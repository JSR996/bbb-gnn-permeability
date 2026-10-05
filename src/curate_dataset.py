"""Curate BBBP: deduplicate, resolve label conflicts, restrict the domain.

Three independent operations, each toggleable and each reported separately,
because they rest on different justifications and one of them is a
judgement call rather than a fix.

  --dedup     (data defect) 2039 parseable rows hold 1975 distinct
              structures. Identical molecules appear under different names
              -- Trichloromethane and chloroform, Dichloromethane and "18".
              Duplicates inflate whatever region they sit in.

  --conflicts (data defect) 10 structures carry BOTH labels. Aspirin,
              loratadine, atropine, indomethacin and trimetrexate are each
              in the file as 0 and as 1. These are not resolvable from the
              file, so the default is to drop every copy rather than guess;
              --conflicts keep-majority is offered but is a guess.

  --restrict  (JUDGEMENT, not a fix) drop the entries classified as
              industrial solvents or broken structures in
              data/curation_review.csv.

On --restrict specifically. This is NOT correcting mislabelled data. The
solvent rows are real occupational-toxicology BBB measurements and their
labels are right. Nor is the wider class of small halogenated positives an
error: halothane, methoxyflurane, nitrous oxide, cyclopropane, diethyl
ether and chloroform are inhalational anaesthetics, and small, volatile,
lipophilic molecules crossing quickly is the true chemistry of CNS
penetration rather than a shortcut. Of the 65 molecules with <=8 heavy
atoms, only 25 lack any therapeutic history.

What --restrict does is narrow the training distribution to therapeutic
chemotypes, on the hypothesis that occupational-toxicology entries are the
part of the small/greasy/low-TPSA region the generator has no use for. It
costs accuracy on small molecules. At ~31 of 2039 rows the effect on the
classifier is likely to be null or inconclusive, and that should be
pre-committed to rather than discovered.

    python -m src.curate_dataset --dedup --conflicts drop --restrict
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .datasets import ROOT

RDLogger.DisableLog("rdApp.*")

REVIEW = ROOT / "data" / "curation_review.csv"
OUT = ROOT / "data" / "curated_bbbp.csv"

# Reviewed and locked. Everything else in curation_review.csv is retained:
# anaesthetics, sedative-hypnotics and the small therapeutics.
DROP_CLASSES = {"solvent", "broken"}
# Moved from other_drug to solvent on review: primarily a preservative and
# manufacturing reagent rather than a therapeutic agent.
RECLASSIFY = {"OCc1ccccc1": "solvent"}


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    can = []
    for s in df.smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        can.append(Chem.MolToSmiles(m) if m is not None else None)
    out = df.copy()
    out["canonical_smiles"] = can
    return out[out.canonical_smiles.notna()].reset_index(drop=True)


def load_review() -> dict[str, str]:
    if not REVIEW.exists():
        raise FileNotFoundError(
            f"no review file at {REVIEW}. It is the reviewable record of which "
            f"structures are being dropped and why; --restrict must not run "
            f"without it."
        )
    r = pd.read_csv(REVIEW)
    cls = dict(zip(r.canonical_smiles, r.classification))
    cls.update(RECLASSIFY)
    return cls


def curate(dedup: bool, conflicts: str, restrict: bool) -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "BBBP.csv").dropna(subset=["smiles"])
    df = canonical(raw)
    n0 = len(df)
    print(f"parseable rows                : {n0}")
    print(f"distinct structures           : {df.canonical_smiles.nunique()}")

    # 1. Conflicting labels FIRST: resolving them after deduplication would
    #    mean whichever copy happened to survive silently decided the label.
    if conflicts != "keep":
        g = df.groupby("canonical_smiles").p_np.nunique()
        bad = set(g[g > 1].index)
        rows = df.canonical_smiles.isin(bad).sum()
        if conflicts == "drop":
            df = df[~df.canonical_smiles.isin(bad)]
            print(f"label conflicts dropped       : {rows} rows / {len(bad)} structures")
        elif conflicts == "keep-majority":
            keep = []
            for c, sub in df[df.canonical_smiles.isin(bad)].groupby("canonical_smiles"):
                keep.append(sub.p_np.mode().iloc[0])
            print(f"label conflicts resolved by majority: {len(bad)} structures "
                  f"(a guess, not a fix)")
            maj = dict(zip(sorted(bad), keep))
            df = df.copy()
            df.loc[df.canonical_smiles.isin(bad), "p_np"] = (
                df.loc[df.canonical_smiles.isin(bad), "canonical_smiles"].map(maj))
        else:
            raise ValueError(f"unknown --conflicts {conflicts!r}")

    # 2. Deduplicate on structure, keeping the first occurrence.
    if dedup:
        before = len(df)
        df = df.drop_duplicates("canonical_smiles", keep="first")
        print(f"duplicate rows removed        : {before - len(df)}")

    # 3. Domain restriction -- the judgement call.
    if restrict:
        cls = load_review()
        drop = df.canonical_smiles.map(cls).isin(DROP_CLASSES)
        counts = df.canonical_smiles.map(cls)[drop].value_counts().to_dict()
        df = df[~drop]
        print(f"domain-restricted rows removed: {int(drop.sum())}  {counts}")

    print(f"\nfinal rows                    : {len(df)}  "
          f"({n0 - len(df)} removed, {(n0-len(df))/n0:.1%})")
    print(f"positives                     : {int(df.p_np.sum())} "
          f"({df.p_np.mean():.1%})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--conflicts", default="keep",
                    choices=["keep", "drop", "keep-majority"])
    ap.add_argument("--restrict", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not (args.dedup or args.restrict or args.conflicts != "keep"):
        print("nothing to do -- pass --dedup, --conflicts or --restrict")
        return

    df = curate(args.dedup, args.conflicts, args.restrict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["num", "name", "p_np", "smiles"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("  Column order matches BBBP.csv, so datasets.DATASETS['bbbp'] can be")
    print("  pointed at it without other changes. The original is untouched.")


if __name__ == "__main__":
    main()
