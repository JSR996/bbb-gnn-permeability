"""Paired comparison of reward-guard arms across seeds.

Three configurations at kl=0.3, 150 steps, seeds 0-2:

  raw      the kl=0.3 baseline -- no size gate, linear reward
  gate     hard size gate at 10 heavy atoms, linear reward
  gate_gm  the same gate, with weighted geometric aggregation

The seed here fixes the generator's sampling, the discriminator's init and
its real-batch draws, so the three arms at a given seed see the same rollout
noise. That makes this a PAIRED design for the same reason the architecture
comparison is one (src/paired_analysis.py), and the marginal spread across
seeds overstates the noise on an arm-vs-arm difference. With n=3 the paired
sd carries 2 df, so these are ranked candidates rather than results.

Seed 0 for each arm was run earlier and lives elsewhere; RUNS maps every
(arm, seed) to its directory rather than assuming a layout, so nothing had
to be recomputed or moved to make the paths uniform.

    python -m src.arm_study
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .datasets import ROOT

RDLogger.DisableLog("rdApp.*")

RESULTS = ROOT / "results"

RUNS = {
    ("raw", 0): RESULTS / "c_transform" / "raw",
    ("raw", 1): RESULTS / "arm_study" / "raw_s1",
    ("raw", 2): RESULTS / "arm_study" / "raw_s2",
    ("gate", 0): RESULTS / "guarded" / "gate",
    ("gate", 1): RESULTS / "arm_study" / "gate_s1",
    ("gate", 2): RESULTS / "arm_study" / "gate_s2",
    ("gate_gm", 0): RESULTS / "guarded" / "gate_gm",
    ("gate_gm", 1): RESULTS / "arm_study" / "gate_gm_s1",
    ("gate_gm", 2): RESULTS / "arm_study" / "gate_gm_s2",
}

METRICS = ["scaffold_frac", "tanimoto_dist", "c_mean",
           "tpsa_spread_ratio", "hba_spread_ratio", "valid_frac"]

# Warm start (no GRPO), measured in kl_sweep; the reference every arm is
# trying to beat on diversity while still improving on C.
WARM = {"scaffold_frac": 0.736, "tanimoto_dist": 0.916, "c_mean": 0.835}

_TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def endpoints(tail: int = 20) -> pd.DataFrame:
    rows = []
    for (arm, seed), d in RUNS.items():
        h = d / "history.csv"
        if not h.exists():
            print(f"  missing: {arm} seed{seed} ({d})")
            continue
        t = pd.read_csv(h).tail(tail)
        row = {"arm": arm, "seed": seed}
        row.update({m: t[m].mean() for m in METRICS if m in t})
        # Composition of the final group, which is what "did the solvents
        # go away" actually asks.
        f = d / "samples" / "step0149.json"
        if f.exists():
            mols = [m for m in (Chem.MolFromSmiles(s)
                                for s in json.load(open(f))) if m]
            hv = np.array([m.GetNumHeavyAtoms() for m in mols])
            row["frac_under10"] = float((hv < 10).mean())
            row["median_heavy"] = float(np.median(hv))
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    piv = df.pivot(index="seed", columns="arm", values=metric)
    out = []
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        d = (piv[a] - piv[b]).dropna()
        n = len(d)
        if n < 2:
            continue
        md, sd = d.mean(), d.std(ddof=1)
        se = sd / np.sqrt(n)
        tc = _TCRIT.get(n, 1.96)
        lo, hi = md - tc * se, md + tc * se
        indep = np.sqrt(piv[a].var(ddof=1) + piv[b].var(ddof=1))
        out.append({
            "metric": metric, "pair": f"{a} - {b}", "n": n, "mean_diff": md,
            "paired_sd": sd, "indep_sd": indep,
            "sd_ratio": sd / indep if indep else np.nan,
            "ci95_lo": lo, "ci95_hi": hi,
            "sig": "yes" if lo * hi > 0 else "no",
            "dz": md / sd if sd else np.nan,
        })
    return pd.DataFrame(out)


def main() -> None:
    df = endpoints()
    if df.empty:
        print("no runs found")
        return

    print("=" * 88)
    print("PER-ARM MEANS (last 20 steps, mean +- sd over seeds)")
    print("=" * 88)
    cols = [c for c in METRICS + ["frac_under10", "median_heavy"] if c in df]
    g = df.groupby("arm")[cols].agg(["mean", "std"])
    print(g.round(3).to_string())
    print(f"\n  warm-start control: " +
          "  ".join(f"{k}={v}" for k, v in WARM.items()))

    print("\n" + "=" * 88)
    print("PAIRED DIFFERENCES (per-seed, so rollout noise cancels)")
    print("=" * 88)
    allp = pd.concat([paired(df, m) for m in cols if df[m].notna().all()],
                     ignore_index=True)
    for m in allp.metric.unique():
        sub = allp[allp.metric == m]
        print(f"\n{m}")
        print(sub.drop(columns="metric").round(4).to_string(index=False))

    out = RESULTS / "arm_study"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "endpoints.csv", index=False)
    allp.to_csv(out / "paired.csv", index=False)
    n_sig = (allp.sig == "yes").sum()
    print(f"\n{n_sig}/{len(allp)} arm-metric comparisons separate at 95% with "
          f"n={df.seed.nunique()} seeds")
    print("  At n=3 the paired sd has 2 df, so dz is itself noisy -- read these")
    print("  as a ranking, not as results.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
