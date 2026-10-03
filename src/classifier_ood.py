"""Step 3: is the reward's classifier trustworthy off-distribution?

The generator's reward currently comes from the gin/gat/gine ensemble
(~0.93-0.94 test ROC-AUC), while the benchmark leader is hybrid/gat at
0.9608. The obvious move is to swap it in. The obvious move may be wrong:
test ROC-AUC measures accuracy on molecules that look like the training set,
and the generator asks about molecules that by construction do not. What
matters is which model degrades gracefully out there, and the benchmark does
not measure that.

So this runs two separate experiments, because they answer different
questions and only one of them has labels:

  LABELED OOD -- the external holdouts. The leak scrub (leak_report.json)
  removed every molecule and every Murcko scaffold group shared with
  training, which is exactly what makes the survivors an out-of-distribution
  test set with ground truth. n=59 and n=55 are far too small for a headline
  generalization claim, but this is a PAIRED comparison -- the same molecules
  scored by both models -- and the paired question ("which is better here")
  is much better powered than the marginal one ("how good is it").

  UNLABELED OOD -- generated molecules. No ground truth exists, so nothing
  here measures correctness. Disagreement between independently trained
  members measures instability: if the ensemble members diverge wildly on
  invented molecules, the reward they average into is partly noise, and no
  amount of in-distribution accuracy fixes that.

    python -m src.classifier_ood
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

from .featurize import smiles_to_graph
from .hybrid_features import compute_descriptor_matrix
from .models import build_model
from .models_edge import EDGE_MODELS, build_edge_model
from .models_hybrid import build_hybrid_model
from .reward import _checkpoint
from .train import RESULTS_DIR

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent

# The reward's current ensemble, and the benchmark leader it might be
# replaced with. Named per (family, operator) so a hybrid gat and a base gat
# never collide.
CURRENT = [("base", "gin"), ("base", "gat"), ("edge", "gine")]
CANDIDATE = [("hybrid", "gat")]


def _load_one(family: str, name: str, dataset: str, seed: int, device: str):
    """One frozen scorer: SMILES list -> P(permeable).

    Each family carries its own skeleton and its own inputs, so they cannot
    share a loader. Hybrid additionally needs the descriptor scaling saved
    beside its checkpoint -- applying training-set mu/sigma is part of the
    model, and recomputing it from the evaluation set would leak.
    """
    if family == "hybrid":
        ckpt = RESULTS_DIR / "hybrid" / dataset / name / f"seed{seed}" / "model.pt"
        scaling = ckpt.parent / "descriptor_scaling.npz"
        if not ckpt.exists():
            raise FileNotFoundError(f"no hybrid checkpoint at {ckpt}")
        z = np.load(scaling)
        mu, sigma = z["mu"], z["sigma"]
        net = build_hybrid_model(name).to(device)
    else:
        ckpt = _checkpoint(RESULTS_DIR, dataset, name, seed)
        builder = build_edge_model if name in EDGE_MODELS else build_model
        net = builder(name).to(device)
        mu = sigma = None
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    net.requires_grad_(False)

    @torch.no_grad()
    def score(smiles: list[str]) -> np.ndarray:
        """NaN for anything unsanitizable, so callers cannot silently average
        a failed parse in as a real prediction."""
        out = np.full(len(smiles), np.nan)
        keep, graphs = [], []
        for i, s in enumerate(smiles):
            g = smiles_to_graph(s) if s else None
            if g is not None:
                keep.append(i)
                graphs.append(g)
        if not keep:
            return out
        if mu is not None:
            desc = compute_descriptor_matrix([smiles[i] for i in keep])
            desc = (desc - mu) / np.where(sigma == 0, 1.0, sigma)
            for g, d in zip(graphs, desc):
                g.descriptors = torch.tensor(d, dtype=torch.float).unsqueeze(0)
        batch = Batch.from_data_list(graphs).to(device)
        out[keep] = torch.sigmoid(net(batch)).cpu().numpy()
        return out

    return score


def load_group(spec, dataset: str, seed: int, device: str):
    return {f"{fam}:{name}": _load_one(fam, name, dataset, seed, device)
            for fam, name in spec}


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC. Written out rather than imported so the uncertainty
    machinery below sits next to the point estimate it qualifies."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties, or tied scores inflate the estimate
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _boot_auc(y, p, draws=2000, seed=0):
    """Stratified bootstrap CI: resample positives and negatives separately so
    every draw keeps the observed class balance. At n=59 with 15 positives an
    unstratified draw can produce a resample with almost no positives, which
    widens the interval for a reason that has nothing to do with the model."""
    rng = np.random.default_rng(seed)
    ipos, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(ipos, len(ipos)), rng.choice(ineg, len(ineg))])
        out.append(_roc_auc(y[idx], p[idx]))
    return np.nanpercentile(out, [2.5, 97.5])


def _boot_paired_delta(y, pa, pb, draws=2000, seed=0):
    """CI on the DIFFERENCE, resampling molecules once and scoring both models
    on that same resample. The two AUCs are strongly correlated because they
    share the test set, so the marginal intervals can overlap heavily while
    the difference is still well determined -- the same pairing argument as
    src/paired_analysis.py, applied to molecules instead of seeds."""
    rng = np.random.default_rng(seed)
    ipos, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(ipos, len(ipos)), rng.choice(ineg, len(ineg))])
        out.append(_roc_auc(y[idx], pa[idx]) - _roc_auc(y[idx], pb[idx]))
    return float(np.nanmean(out)), np.nanpercentile(out, [2.5, 97.5])


def labeled_ood(current, candidate, device: str) -> pd.DataFrame:
    """Both models on the scaffold-scrubbed external holdouts."""
    rows = []
    for src in ("adenot", "wang"):
        path = ROOT / "data" / "external_holdout" / f"{src}_clean.csv"
        if not path.exists():
            print(f"  skip {src}: {path} missing")
            continue
        df = pd.read_csv(path)
        smi_col = next(c for c in df.columns if c.lower() in ("smiles", "smi"))
        lab_col = next(c for c in df.columns
                       if c.lower() in ("label", "y", "bbb", "p_np", "class"))
        smiles = df[smi_col].tolist()
        y = df[lab_col].to_numpy(float)

        # Ensemble members are averaged, matching how the reward consumes them.
        pc = np.nanmean([f(smiles) for f in current.values()], axis=0)
        pk = np.nanmean([f(smiles) for f in candidate.values()], axis=0)
        ok = ~(np.isnan(pc) | np.isnan(pk) | np.isnan(y))
        y, pc, pk = y[ok], pc[ok], pk[ok]

        auc_c, auc_k = _roc_auc(y, pc), _roc_auc(y, pk)
        lo_c, hi_c = _boot_auc(y, pc)
        lo_k, hi_k = _boot_auc(y, pk)
        d, (dlo, dhi) = _boot_paired_delta(y, pk, pc)
        rows.append({
            "source": src, "n": int(ok.sum()), "n_pos": int((y == 1).sum()),
            "current_auc": auc_c, "current_lo": lo_c, "current_hi": hi_c,
            "candidate_auc": auc_k, "candidate_lo": lo_k, "candidate_hi": hi_k,
            "delta": d, "delta_lo": dlo, "delta_hi": dhi,
            "delta_sig": "yes" if dlo * dhi > 0 else "no",
        })
    return pd.DataFrame(rows)


def unlabeled_ood(current, candidate, smiles: list[str],
                  real: list[str]) -> pd.DataFrame:
    """Member disagreement on generated vs real molecules.

    Real molecules are the control: some disagreement is normal, and only the
    INCREASE going off-distribution says the ensemble is extrapolating.
    """
    rows = []
    for label, group in (("current", current), ("candidate", candidate)):
        if len(group) < 2:
            # A single model cannot disagree with itself; reported rather than
            # silently skipped, since it is a real limitation of comparing a
            # 3-member ensemble against a 1-member one.
            rows.append({"group": label, "members": len(group), "note":
                         "single member -- disagreement undefined"})
            continue
        for setname, s in (("real", real), ("generated", smiles)):
            preds = np.array([f(s) for f in group.values()])
            ok = ~np.isnan(preds).any(axis=0)
            preds = preds[:, ok]
            rows.append({
                "group": label, "members": len(group), "set": setname,
                "n": int(ok.sum()),
                "mean_p": float(preds.mean()),
                "member_sd": float(preds.std(axis=0).mean()),
                "max_spread": float((preds.max(0) - preds.min(0)).mean()),
                # Confidence: how often the mean lands in the saturated tails.
                "frac_extreme": float(
                    ((preds.mean(0) > 0.95) | (preds.mean(0) < 0.05)).mean()),
            })
    return pd.DataFrame(rows)


def reward_saturation(run_dir: Path, dataset: str, seed: int,
                      device: str, steps=(0, 10, 25, 50, 75, 100, 150, 199)) -> pd.DataFrame:
    """Within-group spread of C(m) over training, from the saved sample dumps.

    This is the diagnostic that decides whether the permeability term is
    still doing anything. GRPO's advantage (Eq. 6) is group-RELATIVE: only
    differences in reward within a group move the policy. If every molecule
    in a group scores the same C, the term contributes nothing no matter how
    high that score is. So the quantity to watch is sd, not mean -- and the
    mean rising while the sd falls is the signature of a reward that has
    been optimized until it can no longer discriminate.
    """
    from .reward import load_frozen_classifier, sanitizable

    C = load_frozen_classifier(dataset=dataset, device=device)
    rows = []
    for k in steps:
        f = run_dir / "samples" / f"step{k:04d}.json"
        if not f.exists():
            continue
        mols = [m for m in (sanitizable(x) for x in json.load(open(f))) if m is not None]
        if not mols:
            continue
        p = C(mols)
        rows.append({"step": k, "n": len(p), "mean_C": p.mean(), "sd_C": p.std(),
                     "min_C": p.min(), "max_C": p.max(),
                     "frac_over_95": float((p > 0.95).mean())})
    return pd.DataFrame(rows)


def main(dataset: str, seed: int, device: str, gen_path: Path) -> None:
    print("loading models...")
    current = load_group(CURRENT, dataset, seed, device)
    candidate = load_group(CANDIDATE, dataset, seed, device)
    print(f"  current  (reward's ensemble): {list(current)}")
    print(f"  candidate (benchmark leader): {list(candidate)}")

    print("\n" + "=" * 78)
    print("LABELED OOD -- scaffold-scrubbed external holdouts")
    print("=" * 78)
    lab = labeled_ood(current, candidate, device)
    if lab.empty:
        print("  no holdout data found")
    else:
        print(lab.round(4).to_string(index=False))
        print("\n  delta = candidate - current, paired bootstrap over molecules.")
        print("  Marginal CIs are wide at this n and are NOT a generalization")
        print("  claim; the paired delta is the comparison this can support.")

    print("\n" + "=" * 78)
    print("UNLABELED OOD -- disagreement on generated molecules (no ground truth)")
    print("=" * 78)
    gen = json.load(open(gen_path))
    real = pd.read_csv(ROOT / "data" / "raw" / "BBBP.csv")["smiles"].dropna().tolist()
    unl = unlabeled_ood(current, candidate, gen, real)
    print(unl.round(4).to_string(index=False))
    print("\n  member_sd measures INSTABILITY, not error. Nothing here says")
    print("  which model is more correct on invented molecules -- no labels exist.")

    print("\n" + "=" * 78)
    print("REWARD SATURATION -- does C(m) still discriminate within a group?")
    print("=" * 78)
    sat = reward_saturation(gen_path.parent, dataset, seed, device)
    if sat.empty:
        print(f"  no per-step samples under {gen_path.parent / 'samples'}")
    else:
        print(sat.round(4).to_string(index=False))
        print("\n  GRPO's advantage is group-RELATIVE, so only sd_C moves the")
        print("  policy. mean_C rising while sd_C collapses means the term has")
        print("  been optimized until it can no longer rank anything.")

    out = RESULTS_DIR / "classifier_ood"
    out.mkdir(parents=True, exist_ok=True)
    if not lab.empty:
        lab.to_csv(out / "labeled_ood.csv", index=False)
    unl.to_csv(out / "unlabeled_ood.csv", index=False)
    if not sat.empty:
        sat.to_csv(out / "reward_saturation.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--generated", type=Path,
                    default=ROOT / "results" / "gan" / "seed0" / "samples.json")
    args = ap.parse_args()
    main(args.dataset, args.seed, args.device, args.generated)
