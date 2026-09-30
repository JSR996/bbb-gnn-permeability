"""Per-step chemistry diagnostics for the GRPO loop: drift and diversity.

Reward hacking in this pipeline is a *chemical* failure, and the metric that
was being logged (`unique`, the fraction of distinct SMILES strings) cannot
see it. A 40-step run produced 195/199 distinct strings -- which reads as
healthy -- while the property spread had already collapsed (TPSA sd 58.5 ->
18.7 against BBBP, HBA 3.2 -> 0.9). Distinct strings are not distinct
chemistry: minor decorations of one scaffold are 195 unique strings and one
molecule family.

So the diagnostics here are deliberately structural rather than lexical:

  * descriptor drift, in units of the reference sd, so a shift is comparable
    across descriptors with different scales (TPSA spans ~60, HBD spans ~2);
  * descriptor *spread* as a ratio to the reference, which is the number that
    actually moved in the 40-step run and the one string-uniqueness missed;
  * Murcko scaffold count, i.e. how many distinct cores the group is built on;
  * mean pairwise Tanimoto distance over Morgan fingerprints, the standard
    internal-diversity measure.

Nothing here touches the reward or the policy -- it is measurement only, and
`summarize` is a pure function of a list of SMILES so it can be re-run over
saved samples without re-running training.

    python -m src.diversity     # self-check
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .hybrid_features import DESCRIPTOR_NAMES, compute_descriptor_matrix

RDLogger.DisableLog("rdApp.*")

# The four descriptors on the BBB-permeability axis, i.e. the ones the reward
# can be gamed along. They are a subset of DESCRIPTOR_NAMES so the reference
# statistics and the per-step statistics come from the same code path.
TRACKED = ("tpsa", "hba", "logp", "mw")
_TRACKED_IDX = [DESCRIPTOR_NAMES.index(n) for n in TRACKED]

# 2048-bit Morgan radius 2 -- the ECFP4-equivalent default for diversity work.
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _mols(smiles: list[str]) -> list[Chem.Mol]:
    """Sanitizable molecules only. Everything downstream assumes valid input."""
    out = []
    for s in smiles:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(m)
    return out


def scaffold_count(mols: list[Chem.Mol]) -> tuple[int, float]:
    """Distinct Bemis-Murcko scaffolds, absolute and as a fraction of input.

    The fraction is the one to watch: an absolute count falls simply because
    fewer molecules sanitized, which is a validity problem rather than a
    diversity one, and the two should not be read off the same number.
    """
    if not mols:
        return 0, 0.0
    cores = set()
    for m in mols:
        try:
            cores.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            # A scaffold that will not generate is not a diversity signal;
            # dropping it understates diversity slightly, which is the safe
            # direction for a collapse detector.
            continue
    return len(cores), len(cores) / len(mols)


def mean_pairwise_tanimoto_distance(mols: list[Chem.Mol], max_n: int = 256,
                                    seed: int = 0) -> float:
    """Internal diversity: mean 1 - Tanimoto over all distinct pairs.

    1.0 is maximally diverse, 0.0 is every molecule identical. Subsampled
    above `max_n` because the pair count is quadratic and this runs every step.
    """
    if len(mols) < 2:
        return float("nan")
    if len(mols) > max_n:
        idx = np.random.default_rng(seed).choice(len(mols), max_n, replace=False)
        mols = [mols[i] for i in idx]
    fps = [_MORGAN.GetFingerprint(m) for m in mols]
    # BulkTanimotoSimilarity over the upper triangle: each i against j > i, so
    # every unordered pair is counted exactly once and self-similarity never
    # enters the mean.
    sims = []
    for i in range(len(fps) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(1.0 - np.mean(sims))


def reference_band(smiles: list[str], n: int, draws: int = 20,
                   seed: int = 0) -> dict[str, tuple[float, float]]:
    """Mean and sd of the structural metrics over `draws` subsamples of size n.

    Scaffold fraction and, to a lesser extent, Tanimoto distance depend
    strongly on sample size -- more molecules means more repeated scaffolds,
    so the distinct fraction falls. On BBBP it reads 0.758 at n=199 and 0.595
    at n=982, which is a bigger swing than any effect being looked for. So a
    generated sample can only be compared against a reference subsampled to
    the SAME n, and comparing two generated sets of different sizes to each
    other is meaningless without this.
    """
    pool = [s for s in smiles if s]
    n = min(n, len(pool))
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {"scaffold_frac": [], "tanimoto_dist": []}
    for _ in range(draws):
        sub = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
        row = summarize(sub)
        for k in acc:
            acc[k].append(row[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def reference_stats(smiles: list[str]) -> dict[str, tuple[float, float]]:
    """Mean and sd per tracked descriptor over a reference set (BBBP/B3DB).

    Computed once per run and held fixed: drift is measured against the real
    data, not against a moving window of the generator's own output, which
    would normalize away the very trend being looked for.
    """
    mols = _mols(smiles)
    mat = compute_descriptor_matrix([Chem.MolToSmiles(m) for m in mols])
    return {
        n: (float(mat[:, i].mean()), float(mat[:, i].std()))
        for n, i in zip(TRACKED, _TRACKED_IDX)
    }


def summarize(smiles: list[str], ref: dict[str, tuple[float, float]] | None = None,
              group_size: int | None = None) -> dict[str, float]:
    """One row of chemistry diagnostics for a sampled group.

    `smiles` is the RAW group including invalid entries, so validity is
    measured here rather than passed in; every other metric is computed on the
    sanitizable subset only. With `ref`, each tracked descriptor also reports
    drift in reference sd units and spread as a ratio to the reference sd --
    those two are the collapse signal.
    """
    n_total = group_size if group_size is not None else len(smiles)
    mols = _mols(smiles)
    row: dict[str, float] = {
        "n_valid": len(mols),
        "valid_frac": len(mols) / n_total if n_total else 0.0,
        # Kept for continuity with the old logs, and as the contrast case: it
        # is expected to stay near 1.0 while the structural metrics fall.
        "unique_smiles_frac": (
            len({Chem.MolToSmiles(m) for m in mols}) / n_total if n_total else 0.0
        ),
    }

    n_scaf, scaf_frac = scaffold_count(mols)
    row["n_scaffolds"] = n_scaf
    row["scaffold_frac"] = scaf_frac
    row["tanimoto_dist"] = mean_pairwise_tanimoto_distance(mols)

    if not mols:
        for n in TRACKED:
            row[f"{n}_mean"] = float("nan")
            row[f"{n}_sd"] = float("nan")
            if ref:
                row[f"{n}_drift_sd"] = float("nan")
                row[f"{n}_spread_ratio"] = float("nan")
        return row

    mat = compute_descriptor_matrix([Chem.MolToSmiles(m) for m in mols])
    for n, i in zip(TRACKED, _TRACKED_IDX):
        col = mat[:, i]
        row[f"{n}_mean"] = float(col.mean())
        row[f"{n}_sd"] = float(col.std())
        if ref:
            mu, sd = ref[n]
            row[f"{n}_drift_sd"] = float((col.mean() - mu) / (sd + 1e-9))
            row[f"{n}_spread_ratio"] = float(col.std() / (sd + 1e-9))
    return row


def _self_check() -> None:
    # One scaffold decorated many ways: the string-uniqueness trap in miniature.
    decorated = [
        "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC",
        "c1ccccc1CCCCC", "c1ccccc1CCCCCC",
    ]
    diverse = ["CCO", "c1ccncc1", "C1CCCCC1", "CC(=O)Oc1ccccc1C(=O)O",
               "C1COCCN1", "CCCCCCCC(=O)O"]

    d_dec = summarize(decorated)
    d_div = summarize(diverse)

    assert d_dec["unique_smiles_frac"] == 1.0, "all six strings are distinct"
    assert d_dec["n_scaffolds"] == 1, f"one core expected, got {d_dec['n_scaffolds']}"
    assert d_div["n_scaffolds"] > d_dec["n_scaffolds"], "scaffolds must separate these"
    assert d_div["tanimoto_dist"] > d_dec["tanimoto_dist"], \
        f"tanimoto must separate these ({d_div['tanimoto_dist']:.3f} vs {d_dec['tanimoto_dist']:.3f})"
    print(f"  decorated: unique={d_dec['unique_smiles_frac']:.2f}  "
          f"scaffolds={d_dec['n_scaffolds']}  tanimoto={d_dec['tanimoto_dist']:.3f}")
    print(f"  diverse:   unique={d_div['unique_smiles_frac']:.2f}  "
          f"scaffolds={d_div['n_scaffolds']}  tanimoto={d_div['tanimoto_dist']:.3f}")
    print("  ^ identical on the old metric, separated by the new ones")

    # Invalid entries count against validity, and must not crash the rest.
    mixed = summarize(["CCO", "", "not-a-molecule", "c1ccccc1"])
    assert mixed["n_valid"] == 2 and mixed["valid_frac"] == 0.5, mixed
    assert np.isfinite(mixed["tanimoto_dist"])

    # All-invalid must return cleanly rather than raise: early training hits it.
    dead = summarize(["", "xyz"])
    assert dead["n_valid"] == 0 and dead["valid_frac"] == 0.0
    assert np.isnan(dead["tanimoto_dist"])
    assert np.isnan(dead["tpsa_mean"])

    # A single valid molecule has no pairs, so diversity is undefined, not 0.
    assert np.isnan(summarize(["CCO", ""])["tanimoto_dist"])

    # Drift against a reference: identical sets drift 0 and keep spread 1.
    ref = reference_stats(diverse)
    same = summarize(diverse, ref)
    for n in TRACKED:
        assert abs(same[f"{n}_drift_sd"]) < 1e-6, f"{n} drifted against itself"
        assert abs(same[f"{n}_spread_ratio"] - 1.0) < 1e-6, f"{n} spread moved"

    # And a narrow, greasy subset must show the signature we are hunting:
    # negative TPSA drift with spread well under 1.
    greasy = summarize(["CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"], ref)
    assert greasy["tpsa_drift_sd"] < 0, greasy["tpsa_drift_sd"]
    assert greasy["tpsa_spread_ratio"] < 0.5, greasy["tpsa_spread_ratio"]
    print(f"  greasy probe: tpsa drift={greasy['tpsa_drift_sd']:+.2f} sd  "
          f"spread={greasy['tpsa_spread_ratio']:.2f}x")

    # Scaffold fraction must fall as n rises -- the property that makes
    # matched-n comparison mandatory. Built from a pool with real repeats.
    pool = (diverse + decorated) * 12
    small = reference_band(pool, 6, draws=10)["scaffold_frac"][0]
    large = reference_band(pool, 60, draws=10)["scaffold_frac"][0]
    assert small > large, f"scaffold_frac should fall with n ({small} vs {large})"
    print(f"  reference_band: scaffold_frac {small:.3f} at n=6 -> {large:.3f} at n=60")

    print("diversity self-check passed")


if __name__ == "__main__":
    _self_check()
