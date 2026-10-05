"""Attributed retraining sweep: which curation step, if any, moves the classifier.

Three stages, each a strict superset of the next in training rows:

  raw       BBBP as shipped
  defects   --dedup --conflicts drop   (74 rows: 54 duplicates, 20 conflicts)
  full      + --restrict               (25 more: 23 solvents, 2 broken)

THE DESIGN PROBLEM THIS EXISTS TO AVOID. `scaffold_split` partitions the
molecule list it is given, so three datasets of different size produce
three different splits. Training each stage through the normal path would
score them on three different test sets, and any difference would confound
"better training data" with "easier test set". It would look like a clean
paired sweep and measure nothing.

So the split is computed ONCE per seed, on the smallest (fully curated)
molecule set, and the resulting validation and test folds are FIXED across
all three stages. Only the training set grows. The question then has one
answer: does having those rows in training help or hurt, measured on
identical molecules.

Rows present in a larger stage but absent from the common universe are
added to training only, and only when their Murcko scaffold does not occur
in the fixed test or validation folds -- otherwise a row deleted by
curation could reintroduce leakage that the original split had excluded.
Any such exclusions are reported rather than silently applied.

    python -m src.curation_sweep --models gin --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from .curate_dataset import canonical, curate
from .datasets import ROOT
from .featurize import smiles_to_graph
from .models import build_model
from .models_edge import EDGE_MODELS, build_edge_model
from .split import murcko_scaffold, scaffold_split

RDLogger.DisableLog("rdApp.*")

OUT_DIR = ROOT / "results" / "curation_sweep"


def stage_frames() -> dict[str, pd.DataFrame]:
    """The three training-row sets, keyed by canonical SMILES."""
    raw = canonical(pd.read_csv(ROOT / "BBBP.csv").dropna(subset=["smiles"]))
    print("\n--- stage: defects ---")
    defects = curate(dedup=True, conflicts="drop", restrict=False)
    print("\n--- stage: full ---")
    full = curate(dedup=True, conflicts="drop", restrict=True)
    return {"raw": raw, "defects": defects, "full": full}


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from .evaluate import compute_metrics  # reuse the project's definitions

    m = {k: v for k, v in compute_metrics(y, p).items()
         if isinstance(v, (int, float))}
    # Brier is the calibration question the conflicting labels raise, and is
    # not in the existing metric set.
    m["brier"] = float(np.mean((p - y) ** 2))
    return m


def run(models: list[str], seeds: list[int], epochs: int, device: str) -> pd.DataFrame:
    stages = stage_frames()
    common = stages["full"]
    rows = []

    for seed in seeds:
        # One split per seed, on the common universe. Fixed for every stage.
        smi = common.canonical_smiles.tolist()
        tr_i, va_i, te_i = scaffold_split(smi, seed=seed, verbose=False)
        val_smiles = {smi[i] for i in va_i}
        test_smiles = {smi[i] for i in te_i}
        held_scaffolds = {murcko_scaffold(s) for s in (val_smiles | test_smiles)}
        base_train = {smi[i] for i in tr_i}

        for stage, df in stages.items():
            extra, blocked = set(), 0
            for s in df.canonical_smiles:
                if s in base_train or s in val_smiles or s in test_smiles:
                    continue
                if murcko_scaffold(s) in held_scaffolds:
                    blocked += 1          # would reintroduce leakage
                else:
                    extra.add(s)
            train_smiles = base_train | extra

            # Training graphs are built from ROWS, not from the set of
            # distinct SMILES. Duplicates matter because a molecule present
            # twice is seen twice per epoch and up-weighted accordingly, and
            # a conflicting pair contributes both of its labels -- which is
            # exactly what training on the raw file does. Building from a set
            # would deduplicate silently and the dedup arm would measure
            # nothing.
            sub = df[df.canonical_smiles.isin(train_smiles)]
            g_tr = []
            for smi, lab in zip(sub.canonical_smiles, sub.p_np):
                g = smiles_to_graph(smi, label=float(lab))
                if g is not None:
                    g_tr.append(g)

            # Evaluation folds come from the common universe, which is
            # already deduplicated and conflict-free, so one graph each.
            lab_common = dict(zip(common.canonical_smiles, common.p_np))

            def eval_graphs(sel):
                out = []
                for smi in sel:
                    if smi in lab_common:
                        g = smiles_to_graph(smi, label=float(lab_common[smi]))
                        if g is not None:
                            out.append(g)
                return out

            g_va, g_te = eval_graphs(val_smiles), eval_graphs(test_smiles)

            for model_name in models:
                res = _fit(model_name, g_tr, g_va, g_te, seed, epochs, device)
                res.update({"stage": stage, "seed": seed, "model": model_name,
                            "n_train": len(g_tr), "n_extra": len(extra),
                            "n_train_distinct": int(sub.canonical_smiles.nunique()),
                            "leak_blocked": blocked,
                            "n_val": len(g_va), "n_test": len(g_te)})
                rows.append(res)
                print(f"  {stage:<8} seed{seed} {model_name:<5} "
                      f"n_train={len(g_tr):<5} auc={res['roc_auc']:.4f} "
                      f"brier={res['brier']:.4f}")
    return pd.DataFrame(rows)


def _fit(model_name, g_tr, g_va, g_te, seed, epochs, device) -> dict:
    from torch_geometric.loader import DataLoader

    from .train import set_seed

    set_seed(seed)
    builder = build_edge_model if model_name in EDGE_MODELS else build_model
    net = builder(model_name).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    dl = DataLoader(g_tr, batch_size=64, shuffle=True)

    @torch.no_grad()
    def predict(gs):
        net.eval()
        from torch_geometric.data import Batch
        out = []
        for i in range(0, len(gs), 256):
            b = Batch.from_data_list(gs[i:i + 256]).to(device)
            out.append(torch.sigmoid(net(b)).cpu().numpy())
        return np.concatenate(out)

    y_va = np.array([float(g.y) for g in g_va])
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        net.train()
        for b in dl:
            b = b.to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(b), b.y.float())
            opt.zero_grad(); loss.backward(); opt.step()
        auc = _metrics(y_va, predict(g_va))["roc_auc"]
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    net.load_state_dict(best_state)
    y_te = np.array([float(g.y) for g in g_te])
    m = _metrics(y_te, predict(g_te))
    m["val_roc_auc"] = best
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gin"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = run(args.models, args.seeds, args.epochs, args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "runs.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-STAGE (fixed val/test folds; only the training set differs)")
    print("=" * 78)
    cols = ["roc_auc", "pr_auc", "brier", "n_train"]
    print(df.groupby("stage")[cols].agg(["mean", "std"]).round(4).to_string())

    print("\npaired per-seed differences:")
    for metric in ("roc_auc", "pr_auc", "brier"):
        piv = df.pivot_table(index=["seed", "model"], columns="stage", values=metric)
        for a, b in (("defects", "raw"), ("full", "defects"), ("full", "raw")):
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna()
                if len(d) > 1:
                    se = d.std(ddof=1) / np.sqrt(len(d))
                    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(d), 1.96)
                    lo, hi = d.mean() - t * se, d.mean() + t * se
                    sig = "yes" if lo * hi > 0 else "no"
                    print(f"  {metric:<8} {a:>8} - {b:<8} {d.mean():+.4f} "
                          f"[{lo:+.4f}, {hi:+.4f}]  sig={sig}")
    print(f"\nwrote {OUT_DIR / 'runs.csv'}")


if __name__ == "__main__":
    main()
