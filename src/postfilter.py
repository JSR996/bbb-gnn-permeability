"""Post-generation filtering: pool a run's samples, screen them, rank survivors.

Path 2. The active loop is left as the `gate` arm -- size gate only -- and
structural screening happens HERE, after training, where the generator
cannot optimize against it.

That separation is the whole point. A reactive-group penalty inside the
reward was measured to reduce the patterns it was trained against (0.339 ->
0.203) while leaving independent catalogues unmoved (BRENK 0.438 -> 0.406,
within seed noise) -- the generator learned the checklist, not the
chemistry. A filter the policy never sees cannot be gamed, because no
gradient, and no reward, ever reaches it.

The screen is the union of the targeted SMARTS set and BRENK, PAINS A/B/C
and NIH. Using everything is safe here for the same reason it was unsafe in
the loop: nothing is being trained, so no instrument is being consumed.

    python -m src.postfilter results/guarded/gate results/arm_study/gate_s1 \
        --from-step 120 --out results/candidates

TWO CAUTIONS, both measured rather than asserted, and both printed by the
script so they cannot be lost between here and a slide:

  * Passing this screen is NOT evidence a molecule is safe. 39% of the real
    BBB+ drugs in BBBP fail the same union. The filter removes recognized
    liabilities; it says nothing about the ones no catalogue encodes.

  * Ranking by C is only meaningful if C still discriminates. In the
    collapsed baseline the within-group sd of C fell to 0.006 with every
    molecule above 0.95, and MC dropout put the model's own uncertainty
    well above that -- a ranking there is noise. The script reports the
    spread of C among survivors and says so when it is too tight to rank on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog, QED
from rdkit.Chem.FilterCatalog import FilterCatalogParams as P

from .alerts import alert_names, count_alerts
from .datasets import ROOT
from .diversity import summarize

RDLogger.DisableLog("rdApp.*")

CATALOGUES = ("BRENK", "PAINS_A", "PAINS_B", "PAINS_C", "NIH")


def _union_catalog() -> FilterCatalog.FilterCatalog:
    p = P()
    for name in CATALOGUES:
        p.AddCatalog(getattr(P.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(p)


def pool_samples(run_dirs: list[Path], from_step: int = 120) -> list[str]:
    """Canonical, deduplicated SMILES from the per-step dumps.

    Late steps only: early groups are close to the warm start and say
    little about what the trained policy produces. Deduplication is on the
    CANONICAL form, so two spellings of one molecule count once -- without
    it a collapsed run would look productive.
    """
    seen: dict[str, None] = {}
    for d in run_dirs:
        sdir = Path(d) / "samples"
        if not sdir.exists():
            print(f"  no samples/ in {d}")
            continue
        for f in sorted(sdir.glob("step*.json")):
            if int(f.stem[4:]) < from_step:
                continue
            for s in json.load(open(f)):
                m = Chem.MolFromSmiles(s) if s else None
                if m is not None:
                    seen.setdefault(Chem.MolToSmiles(m), None)
    return list(seen)


def screen(smiles: list[str], min_heavy_atoms: int = 10) -> pd.DataFrame:
    """One row per molecule with every rejection reason, not just the first.

    Rows are kept rather than dropped so the discarded set stays inspectable
    -- knowing WHAT was thrown away is how the next round of patterns gets
    chosen, and a script that silently returns survivors hides that.
    """
    cat = _union_catalog()
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        cat_hits = [e.GetDescription() for e in cat.GetMatches(m)]
        targeted = alert_names(m)
        hv = m.GetNumHeavyAtoms()
        rows.append({
            "smiles": s, "heavy_atoms": hv, "mw": Descriptors.MolWt(m),
            "qed": QED.qed(m),
            "n_targeted": len(targeted), "n_catalogue": len(cat_hits),
            "alerts": "|".join(sorted(set(targeted + cat_hits))),
            "undersized": hv < min_heavy_atoms,
            "passes": hv >= min_heavy_atoms and not cat_hits and not targeted,
        })
    return pd.DataFrame(rows)


def add_scores(df: pd.DataFrame, dataset: str = "bbbp",
               device: str = "cpu") -> pd.DataFrame:
    """Frozen-classifier score for the survivors, in batches."""
    from .reward import load_frozen_classifier, sanitizable

    classify = load_frozen_classifier(dataset=dataset, device=device)
    out = np.full(len(df), np.nan)
    idx = np.where(df["passes"].to_numpy())[0]
    for i in range(0, len(idx), 256):
        chunk = idx[i:i + 256]
        mols = [sanitizable(df.smiles.iloc[j]) for j in chunk]
        keep = [(j, m) for j, m in zip(chunk, mols) if m is not None]
        if keep:
            out[[j for j, _ in keep]] = classify([m for _, m in keep])
    df = df.copy()
    df["c_score"] = out
    return df


def novelty(df: pd.DataFrame, dataset: str = "bbbp") -> pd.DataFrame:
    """Flag survivors that are literally training molecules.

    A generator that rediscovers its training set is not generating. This
    is cheap to check and easy to forget.
    """
    from .datasets import DATASETS

    cfg = DATASETS[dataset]
    raw = pd.read_csv(cfg["path"], sep=cfg["sep"])[cfg["smiles_col"]].dropna()
    known = set()
    for s in raw:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            known.add(Chem.MolToSmiles(m))
    df = df.copy()
    df["in_training_set"] = df.smiles.isin(known)
    return df


def report(df: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    total = len(df)
    survivors = df[df["passes"] & ~df["in_training_set"]].copy()
    print("=" * 74)
    print("POST-GENERATION SCREEN")
    print("=" * 74)
    print(f"  pooled, deduplicated : {total}")
    print(f"  undersized           : {int(df.undersized.sum())}")
    print(f"  targeted alert       : {int((df.n_targeted > 0).sum())}")
    print(f"  catalogue alert      : {int((df.n_catalogue > 0).sum())}")
    print(f"  already in training  : {int(df.in_training_set.sum())}")
    print(f"  SURVIVORS            : {len(survivors)} ({len(survivors)/max(total,1):.0%})")

    pre = summarize(df.smiles.tolist())
    post = summarize(survivors.smiles.tolist()) if len(survivors) else None
    if post:
        print(f"\n  scaffold fraction    : {pre['scaffold_frac']:.3f} -> "
              f"{post['scaffold_frac']:.3f}")
        print(f"  tanimoto distance    : {pre['tanimoto_dist']:.3f} -> "
              f"{post['tanimoto_dist']:.3f}")

    if len(survivors) and survivors.c_score.notna().any():
        c = survivors.c_score.dropna()
        print(f"\n  C among survivors    : mean {c.mean():.3f}  sd {c.std():.3f}  "
              f"{(c > 0.95).mean():.0%} above 0.95")
        # The saturation test. MC dropout put the classifier's own
        # uncertainty near 0.95 in logit space; once the spread between
        # molecules falls far below that, ranking on C orders noise.
        if c.std() < 0.02:
            print("  !! C is saturated here (sd < 0.02). Ranking by it orders")
            print("     noise rather than permeability -- rank on something else,")
            print("     or treat the survivor set as unordered.")

    print("\n  Passing this screen is NOT evidence of safety: 39% of the real")
    print("  BBB+ drugs in BBBP fail the same union. It removes recognized")
    print("  liabilities and says nothing about the ones no catalogue encodes.")
    print("  Every survivor is a hypothesis for evaluation, not for synthesis.")

    return survivors.sort_values("c_score", ascending=False).head(top)


def main(run_dirs: list[Path], from_step: int, out: Path, dataset: str,
         device: str, min_heavy_atoms: int, top: int) -> None:
    smiles = pool_samples(run_dirs, from_step)
    print(f"pooled {len(smiles)} distinct molecules from {len(run_dirs)} run(s) "
          f"at step >= {from_step}\n")
    if not smiles:
        return
    df = novelty(add_scores(screen(smiles, min_heavy_atoms), dataset, device), dataset)
    ranked = report(df, top)

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "screened.csv", index=False)
    ranked.to_csv(out / "candidates.csv", index=False)
    print(f"\ntop {len(ranked)} by C:")
    print(ranked[["smiles", "heavy_atoms", "qed", "c_score"]]
          .head(10).round(3).to_string(index=False))
    print(f"\nwrote {out}/screened.csv (all, with reasons) and candidates.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--from-step", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "candidates")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-heavy-atoms", type=int, default=10)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    main(args.run_dirs, args.from_step, args.out, args.dataset, args.device,
         args.min_heavy_atoms, args.top)
