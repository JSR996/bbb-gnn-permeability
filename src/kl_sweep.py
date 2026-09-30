"""Step 4: does raising kl_coef hold back the collapse, and at what cost?

The KL term pulls the policy toward the reference policy, which is the MLE
warm start. So the sweep has a trivial degenerate solution: turn kl_coef up
far enough and the generator reproduces the warm start, property spread is
"restored", and nothing has been generated. Scoring this run on "did the
collapse stop" would report that as a success.

It also has a second degenerate solution that the baseline exposed. Mean
reward cannot be the other half of the criterion either, because C(m)
saturates: at step 199 the classifier scores every molecule above 0.95 and
the within-group sd of C is 0.006, so the reward is HIGHEST exactly where the
permeability term has stopped discriminating. "Reward above baseline" is
satisfied by the failure mode.

So each run is scored on three things at once, and a setting only counts if
it holds all three:

  1. diversity retained   -- scaffold fraction and Tanimoto distance, against
                             the warm start (not against BBBP: the warm start
                             is already 0.72-0.87x narrower on descriptors,
                             and no KL setting can beat its own reference)
  2. reward still live    -- sd of C within the group, NOT its mean. This is
                             the quantity GRPO's group-relative advantage
                             actually consumes.
  3. optimization happened -- mean C above the warm start's. Necessary but
                             nowhere near sufficient, for the reason above.

The output is a frontier, not a winner: the question is where the knee is,
and whether one exists at all.

    python -m src.kl_sweep --steps 120 --coefs 0.02,0.1,0.3,1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import ROOT
from .train_gan import OUT_DIR, train

SWEEP_DIR = ROOT / "results" / "kl_sweep"


def summarize_run(run_dir: Path, tail: int = 20) -> dict:
    """Endpoint state of one run, averaged over its last `tail` steps.

    A single final group of 64 is a noisy estimate of every quantity here, and
    the endpoint is what the whole comparison turns on, so it is averaged.
    """
    df = pd.read_csv(run_dir / "history.csv")
    t = df.tail(tail)
    out = {
        "scaffold_frac": t.scaffold_frac.mean(),
        "tanimoto_dist": t.tanimoto_dist.mean(),
        "valid_frac": t.valid_frac.mean(),
        "c_mean": t.c_mean.mean(),
        "kl": t.kl.mean() if "kl" in t else np.nan,
        "unique_smiles_frac": t.unique_smiles_frac.mean(),
    }
    for n in ("tpsa", "hba", "logp", "mw"):
        out[f"{n}_spread_ratio"] = t[f"{n}_spread_ratio"].mean()
    return out


def c_spread(run_dir: Path, dataset: str, device: str, tail: int = 5) -> dict:
    """Within-group sd of C over the last few steps -- criterion 2.

    Recomputed from the saved sample dumps rather than logged during training,
    so it can be measured on runs that predate this script.
    """
    from .classifier_ood import reward_saturation

    steps = sorted(int(p.stem[4:]) for p in (run_dir / "samples").glob("step*.json"))
    sat = reward_saturation(run_dir, dataset, 0, device, steps=tuple(steps[-tail:]))
    if sat.empty:
        return {"sd_C": np.nan, "mean_C": np.nan, "frac_over_95": np.nan}
    return {"sd_C": sat.sd_C.mean(), "mean_C": sat.mean_C.mean(),
            "frac_over_95": sat.frac_over_95.mean()}


def warm_start_reference(dataset: str, device: str, n: int = 640,
                         seed: int = 0) -> dict:
    """The no-GRPO control every run is scored against.

    Sampled at n=640 (10 groups' worth) and then compared at matched n by the
    caller, since scaffold fraction is strongly size-dependent.
    """
    import torch

    from .diversity import summarize
    from .generator import CKPT_DIR, SelfiesGenerator
    from .reward import load_frozen_classifier, sanitizable

    torch.manual_seed(seed)
    gen = SelfiesGenerator.load(CKPT_DIR / f"{dataset}_pretrained.pt", device)
    smiles = gen.sample(n, device=device)["smiles"]
    # Score C in groups of 64, matching how the reward sees them, so the
    # within-group sd is comparable to a training run's.
    C = load_frozen_classifier(dataset=dataset, device=device)
    sds, means = [], []
    for i in range(0, len(smiles), 64):
        mols = [m for m in (sanitizable(s) for s in smiles[i:i + 64]) if m]
        if len(mols) > 1:
            p = C(mols)
            sds.append(p.std())
            means.append(p.mean())
    row = summarize(smiles, group_size=len(smiles))
    row["sd_C"] = float(np.mean(sds))
    row["mean_C"] = float(np.mean(means))
    return row


def main(coefs: list[float], steps: int, dataset: str, group_size: int,
         device: str, seed: int) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print("measuring the warm-start control (no GRPO)...")
    warm = warm_start_reference(dataset, device)
    print(f"  scaffold_frac={warm['scaffold_frac']:.3f}  "
          f"tanimoto={warm['tanimoto_dist']:.3f}  "
          f"mean_C={warm['mean_C']:.3f}  sd_C={warm['sd_C']:.3f}")

    rows = []
    for c in coefs:
        run_dir = SWEEP_DIR / f"kl{c:g}"
        if (run_dir / "history.csv").exists():
            print(f"\nkl_coef={c:g}: reusing {run_dir}")
        else:
            print(f"\nkl_coef={c:g}: training {steps} steps -> {run_dir}")
            train(dataset=dataset, steps=steps, group_size=group_size,
                  kl_coef=c, device=device, seed=seed, out_dir=run_dir)
        row = {"kl_coef": c}
        row.update(summarize_run(run_dir))
        row.update(c_spread(run_dir, dataset, device))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Criterion columns, all relative to the warm start.
    df["scaffold_ret"] = df.scaffold_frac / warm["scaffold_frac"]
    df["tanimoto_ret"] = df.tanimoto_dist / warm["tanimoto_dist"]
    df["sd_C_ret"] = df.sd_C / warm["sd_C"]
    df["c_gain"] = df.mean_C - warm["mean_C"]
    # A setting passes only if it holds diversity, keeps the reward able to
    # rank, AND actually optimized. Thresholds are deliberately loose: this is
    # a screen for where the knee is, not an acceptance test.
    df["passes"] = (
        (df.scaffold_ret >= 0.70) & (df.sd_C_ret >= 0.50) & (df.c_gain > 0.01)
    )

    cols = ["kl_coef", "scaffold_frac", "scaffold_ret", "tanimoto_ret",
            "sd_C", "sd_C_ret", "mean_C", "c_gain", "tpsa_spread_ratio",
            "valid_frac", "passes"]
    print("\n" + "=" * 90)
    print("KL SWEEP -- all ratios are vs the warm start (no-GRPO control)")
    print("=" * 90)
    print(df[cols].round(4).to_string(index=False))
    print(f"\nwarm start: scaffold_frac={warm['scaffold_frac']:.3f}  "
          f"tanimoto={warm['tanimoto_dist']:.3f}  mean_C={warm['mean_C']:.3f}  "
          f"sd_C={warm['sd_C']:.3f}")
    print("\npasses = diversity held (scaffold >= 0.70x) AND reward still ranks")
    print("         (sd_C >= 0.50x) AND optimization happened (mean C gain > 0.01)")
    if not df.passes.any():
        print("\n  NOTHING PASSES. Either no KL setting resolves this, or the")
        print("  frontier lies outside the swept range -- read the columns to")
        print("  see which criterion each setting fails, and on which side.")

    df.to_csv(SWEEP_DIR / "kl_sweep.csv", index=False)
    (SWEEP_DIR / "warm_start.json").write_text(json.dumps(warm, indent=1))
    print(f"\nwrote {SWEEP_DIR / 'kl_sweep.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coefs", default="0.02,0.1,0.3,1.0",
                    help="comma-separated kl_coef values; 0.02 is the baseline")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main([float(x) for x in args.coefs.split(",")], args.steps, args.dataset,
         args.group_size, args.device, args.seed)
