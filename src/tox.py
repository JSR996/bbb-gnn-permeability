"""Tox21 toxicity model, and the frozen toxicity term for the reward.

Why this exists. The generator converged on carbon tetrachloride, freon-12
and tetrafluoromethane, and the reward scored them above caffeine. Auditing
BBBP explained it: chloroform, dichloromethane, bromoform and nitrous oxide
are all in the training set labelled BBB+, correctly, because BBBP labels a
TRANSPORT property. Anaesthetics and solvents cross the blood-brain barrier.
Nothing in that label says "drug" and nothing says "safe", so optimizing it
finds exactly what it points at.

Structural-alert catalogues cannot patch this: measured on BBBP, BRENK flags
35% of the real BBB+ drugs and CHEMBL 64%, so as a gate either would reject
much of the answer. The missing piece is a second labelled objective, which
is what Tox21 provides -- 7831 compounds across 12 nuclear-receptor and
stress-response assays.

The term is defined so that higher is better, matching C, QED and SA:

    T(m) = 1 - aggregate(P(active across the 12 assays))

so a molecule predicted inactive everywhere scores near 1 and contributes
positively, and a predicted-active one is pulled down.

IMPORTANT, and this is a limitation rather than a caveat: Tox21 measures 12
specific in-vitro assay endpoints. It is not a safety model. A molecule
scoring well here has not been shown to be safe -- it has been predicted
inactive in twelve assays, on a model trained on a few thousand compounds.
Nothing this pipeline emits is a synthesis candidate.

    python -m src.tox --train        # ~5 min on CPU
    python -m src.tox                # self-check (needs a trained model)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .datasets import ROOT
from .featurize import smiles_to_graph
from .models_tox import TOX21_TASKS, build_tox_model, masked_bce
from .split import scaffold_split

RDLogger.DisableLog("rdApp.*")

TOX_CSV = ROOT / "data" / "tox21" / "tox21.csv"
TOX_DIR = ROOT / "results" / "tox21"


def load_tox21() -> tuple[list, list[str], np.ndarray, np.ndarray]:
    """Graphs, SMILES, targets (n, 12), and an observed-label mask (n, 12)."""
    if not TOX_CSV.exists():
        raise FileNotFoundError(
            f"no Tox21 data at {TOX_CSV}. Fetch it with:\n"
            f"  curl -L https://deepchemdata.s3-us-west-1.amazonaws.com/"
            f"datasets/tox21.csv.gz | gunzip > {TOX_CSV}"
        )
    df = pd.read_csv(TOX_CSV)
    graphs, smiles, y, m = [], [], [], []
    for r in df.itertuples():
        g = smiles_to_graph(r.smiles)
        if g is None:
            continue
        row = np.array([getattr(r, t.replace("-", "_"), np.nan)
                        if hasattr(r, t.replace("-", "_"))
                        else df.loc[r.Index, t] for t in TOX21_TASKS], dtype=float)
        graphs.append(g)
        smiles.append(r.smiles)
        y.append(np.nan_to_num(row, nan=0.0))
        m.append(~np.isnan(row))
    return graphs, smiles, np.array(y), np.array(m, dtype=float)


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(allv):
        sel = allv == v
        if sel.sum() > 1:
            ranks[sel] = ranks[sel].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train(conv: str = "gin", epochs: int = 60, batch_size: int = 128,
          lr: float = 1e-3, patience: int = 15, seed: int = 0,
          device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs, smiles, y, mask = load_tox21()
    # Scaffold split, same discipline as the permeability side: a random split
    # would put close analogues on both sides and overstate generalization.
    tr, va, te = scaffold_split(smiles, seed=seed)
    yt = torch.tensor(y, dtype=torch.float)
    mt = torch.tensor(mask, dtype=torch.float)
    for i, g in enumerate(graphs):
        g.y_tox = yt[i].unsqueeze(0)
        g.mask_tox = mt[i].unsqueeze(0)

    sub = lambda idx: [graphs[i] for i in idx]  # noqa: E731
    dl_tr = DataLoader(sub(tr), batch_size=batch_size, shuffle=True)

    # Per-task pos_weight from the TRAINING split only -- computing it over
    # the whole set would leak the test split's class balance.
    n_pos = (y[tr] * mask[tr]).sum(0)
    n_neg = ((1 - y[tr]) * mask[tr]).sum(0)
    pos_weight = torch.tensor(np.clip(n_neg / np.clip(n_pos, 1, None), 1, 50),
                              dtype=torch.float, device=device)

    model = build_tox_model(conv).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def evaluate(idx) -> tuple[float, np.ndarray]:
        model.eval()
        out = []
        for i in range(0, len(idx), 256):
            b = Batch.from_data_list(sub(idx[i:i + 256])).to(device)
            out.append(torch.sigmoid(model(b)).cpu().numpy())
        p = np.concatenate(out)
        aucs = np.array([
            _roc_auc(y[idx][:, t][mask[idx][:, t] > 0], p[:, t][mask[idx][:, t] > 0])
            for t in range(len(TOX21_TASKS))
        ])
        return float(np.nanmean(aucs)), aucs

    best, best_state, bad, hist = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for b in dl_tr:
            b = b.to(device)
            loss = masked_bce(model(b), b.y_tox, b.mask_tox, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        val_auc, _ = evaluate(va)
        hist.append({"epoch": epoch, "loss": tot / max(len(dl_tr), 1), "val_auc": val_auc})
        if val_auc > best:
            best, bad = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if epoch % 10 == 0:
            print(f"  epoch {epoch:>3}  loss={hist[-1]['loss']:.4f}  val_auc={val_auc:.4f}")

    model.load_state_dict(best_state)
    test_auc, per_task = evaluate(te)
    print(f"\n  best val AUC {best:.4f}   test AUC {test_auc:.4f}")
    for t, a in zip(TOX21_TASKS, per_task):
        print(f"    {t:<14}{a:.4f}")

    out_dir = TOX_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps({
        "conv": conv, "seed": seed, "val_auc": best, "test_auc": test_auc,
        "per_task": dict(zip(TOX21_TASKS, per_task.tolist())),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }, indent=1))
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    print(f"\nwrote {out_dir}")
    return {"val_auc": best, "test_auc": test_auc}


def load_frozen_tox(seed: int = 0, device: str = "cpu", conv: str = "gin",
                    aggregate: str = "mean"):
    """Return tox(smiles) -> P(active), aggregated over the 12 assays.

    Frozen exactly as the permeability classifier is, and for the same
    reason: eval() is load-bearing because BatchNorm in train mode would make
    the score depend on which molecules share the batch, and the reward would
    stop being a function of the molecule.

    `aggregate` is "mean" or "max". max is the conservative reading -- one
    predicted-active assay is enough to penalize -- but it produces a sparser
    gradient, since only the arg-max assay moves the score. mean is the
    default because GRPO needs to rank candidates that are all mostly
    inactive, and mean keeps that ordering informative.
    """
    ckpt = TOX_DIR / f"seed{seed}" / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no toxicity model at {ckpt}. Train it first: python -m src.tox --train"
        )
    net = build_tox_model(conv).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    net.requires_grad_(False)

    if aggregate not in ("mean", "max"):
        raise ValueError(f"unknown aggregate {aggregate!r}")

    @torch.no_grad()
    def tox(smiles: list[str]) -> np.ndarray:
        """P(active) per input SMILES; 0.0 for anything unsanitizable.

        Unsanitizable molecules return 0 rather than NaN because they are
        gated out upstream by the validity check and collapse to the flat
        invalid penalty regardless -- a NaN here would propagate into the
        group's advantage instead.
        """
        out = np.zeros(len(smiles))
        keep, graphs = [], []
        for i, s in enumerate(smiles):
            g = smiles_to_graph(s) if s else None
            if g is not None:
                keep.append(i)
                graphs.append(g)
        if not keep:
            return out
        batch = Batch.from_data_list(graphs).to(device)
        p = torch.sigmoid(net(batch)).cpu().numpy()
        out[keep] = p.max(axis=1) if aggregate == "max" else p.mean(axis=1)
        return out

    return tox


def _self_check() -> None:
    tox = load_frozen_tox()

    # Batch invariance, the same property the permeability classifier is
    # asserted on: a molecule's score must not depend on its company.
    alone = tox(["ClC(Cl)(Cl)Cl"])[0]
    crowded = tox(["ClC(Cl)(Cl)Cl", "CCO", "c1ccccc1"])[0]
    assert abs(alone - crowded) < 1e-6, \
        f"tox(m) moved with batch composition ({alone} vs {crowded})"

    # Unsanitizable input scores 0 rather than NaN or a crash.
    out = tox(["", "not-a-molecule", "CCO"])
    assert out[0] == 0.0 and out[1] == 0.0 and np.isfinite(out).all()

    # Outputs are probabilities.
    p = tox(["ClC(Cl)(Cl)Cl", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O"])
    assert ((0.0 <= p) & (p <= 1.0)).all(), p

    # max must be >= mean on the same molecules, by construction.
    tox_max = load_frozen_tox(aggregate="max")
    probe = ["ClC(Cl)(Cl)Cl", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"]
    assert (tox_max(probe) >= tox(probe) - 1e-6).all()

    try:
        load_frozen_tox(aggregate="median")
        raise AssertionError("unknown aggregate should have been rejected")
    except ValueError:
        pass

    print(f"  tox(CCl4)={p[0]:.3f}  tox(caffeine)={p[1]:.3f}  tox(aspirin)={p[2]:.3f}")
    print("tox self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--conv", default="gin")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.train:
        train(conv=args.conv, epochs=args.epochs, seed=args.seed, device=args.device)
    else:
        _self_check()
