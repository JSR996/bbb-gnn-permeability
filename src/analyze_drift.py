"""Plot the collapse trajectory from one or more instrumented GRPO runs.

Reads the per-step history.csv written by `train_gan.train` and produces a
single figure with the four things the first pass is meant to answer:

  1. does chemical diversity collapse (scaffold fraction, Tanimoto distance)
  2. which descriptors drift, and by how much (in reference sd units)
  3. does descriptor *spread* narrow (the signal string-uniqueness missed)
  4. what the reward is doing while that happens

Every panel carries the reference level as a dashed line, so "how far from the
real data" is readable without cross-referencing a table.

    python -m src.analyze_drift results/gan/seed0 results/gan/seed1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .diversity import TRACKED  # noqa: E402


def _smooth(s: pd.Series, w: int) -> pd.Series:
    """Centered rolling mean. A single group of 64 is a noisy estimate of any
    of these quantities, so the trend is plotted over the raw points rather
    than instead of them."""
    return s.rolling(w, center=True, min_periods=1).mean()


def plot(run_dirs: list[Path], out: Path, smooth: int = 9) -> None:
    runs = []
    for d in run_dirs:
        h = d / "history.csv"
        if not h.exists():
            raise FileNotFoundError(f"no history.csv in {d}")
        runs.append((d.name, pd.read_csv(h)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = plt.cm.viridis([i / max(len(runs) - 1, 1) * 0.8 for i in range(len(runs))])

    def panel(ax, col, title, ylabel, ref=None, ref_label=None):
        for (name, df), c in zip(runs, colors):
            if col not in df:
                continue
            ax.plot(df.step, df[col], color=c, alpha=0.18, lw=0.8)
            ax.plot(df.step, _smooth(df[col], smooth), color=c, lw=1.8, label=name)
        if ref is not None:
            ax.axhline(ref, ls="--", c="crimson", lw=1.2, label=ref_label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("generator step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    # 1. structural diversity -- the two metrics that replace SMILES uniqueness
    panel(axes[0, 0], "scaffold_frac", "Scaffold diversity", "distinct scaffolds / group")
    # unique_smiles_frac on the same axes is the whole argument for this work:
    # it stays pinned near 1.0 while the others fall.
    for (name, df), c in zip(runs, colors):
        if "unique_smiles_frac" in df:
            axes[0, 0].plot(df.step, _smooth(df.unique_smiles_frac, smooth),
                            color=c, ls=":", lw=1.4)
    axes[0, 0].plot([], [], color="grey", ls=":", lw=1.4, label="unique SMILES (old metric)")
    axes[0, 0].legend(fontsize=7)

    panel(axes[0, 1], "tanimoto_dist", "Internal diversity", "mean pairwise Tanimoto dist")
    axes[0, 1].legend(fontsize=7)

    panel(axes[0, 2], "valid_frac", "Validity", "sanitizable fraction")

    # 2. descriptor drift, all four on one axis since they share sd units
    ax = axes[1, 0]
    for n, ls in zip(TRACKED, ["-", "--", "-.", ":"]):
        col = f"{n}_drift_sd"
        for (_, df), c in zip(runs, colors):
            if col in df:
                ax.plot(df.step, _smooth(df[col], smooth), color=c, ls=ls, lw=1.5)
        ax.plot([], [], color="grey", ls=ls, lw=1.5, label=n)
    ax.axhline(0, ls="--", c="crimson", lw=1.2, label="BBBP mean")
    ax.set_title("Descriptor drift (reference sd units)", fontsize=10)
    ax.set_xlabel("generator step"); ax.set_ylabel("(gen - ref) / ref sd")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)

    # 3. descriptor spread -- the collapse signal
    ax = axes[1, 1]
    for n, ls in zip(TRACKED, ["-", "--", "-.", ":"]):
        col = f"{n}_spread_ratio"
        for (_, df), c in zip(runs, colors):
            if col in df:
                ax.plot(df.step, _smooth(df[col], smooth), color=c, ls=ls, lw=1.5)
        ax.plot([], [], color="grey", ls=ls, lw=1.5, label=n)
    ax.axhline(1.0, ls="--", c="crimson", lw=1.2, label="BBBP spread")
    ax.set_title("Descriptor spread (ratio to reference sd)", fontsize=10)
    ax.set_xlabel("generator step"); ax.set_ylabel("gen sd / ref sd")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)

    # 4. what the optimizer thinks is happening, for contrast
    panel(axes[1, 2], "c_mean", "Classifier reward (permeability)", "mean C(m) on valid")
    axes[1, 2].legend(fontsize=7)

    fig.suptitle("GRPO baseline: does chemical diversity collapse while reward rises?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def table(run_dirs: list[Path], head: int = 10) -> pd.DataFrame:
    """First-vs-last comparison per run. `head` steps are averaged at each end
    so the endpoints are not single noisy groups."""
    rows = []
    cols = ["scaffold_frac", "tanimoto_dist", "unique_smiles_frac", "valid_frac", "c_mean"]
    cols += [f"{n}_drift_sd" for n in TRACKED] + [f"{n}_spread_ratio" for n in TRACKED]
    for d in run_dirs:
        df = pd.read_csv(d / "history.csv")
        for col in cols:
            if col not in df:
                continue
            a, b = df[col].head(head).mean(), df[col].tail(head).mean()
            rows.append({"run": d.name, "metric": col, "first": a, "last": b,
                         "change": b - a})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smooth", type=int, default=9)
    args = ap.parse_args()

    out = args.out or args.run_dirs[0].parent / "drift_trajectory.png"
    plot(args.run_dirs, out, smooth=args.smooth)

    t = table(args.run_dirs)
    print()
    print(t.pivot(index="metric", columns="run", values=["first", "last"]).round(3).to_string())
    t.to_csv(out.parent / "drift_summary.csv", index=False)
    print(f"\nwrote {out.parent / 'drift_summary.csv'}")
