"""Does the Tox21 term earn its place in the reward? Paired, 4 seeds.

This exists because the first answer was void. `--w-tox`, `--w-alert` and
`--aggregate` were silently dropped between the CLI and `assemble` from
`ac3e304` until the duplicate-call fix, so the run that produced "the Tox21
term does nothing and costs diversity" was a baseline compared against
itself. Anything measured about those flags in that window says nothing.

The arm is `results/factorial_ws/mle_withC` plus `--w-tox 0.5`, same four
seeds and the same per-seed warm start, so the comparison is paired on both
the GRPO seed and the checkpoint -- warm-start variance is ~2x seed variance
here, and pairing on only one of them is what made the VAE read collapse.

The baseline never loaded the tox model, so its `tox_mean` column is logged
against an all-ones placeholder and is not comparable. The toxicity question
therefore has to be answered post hoc, by scoring both arms' molecules with
the same frozen scorer. Real BBBP+ drugs go in as the reference: a term that
drives generated molecules BELOW real approved drugs on the scorer is
buying score, not safety.

    python -m src.tox_arm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from .datasets import ROOT, _load_raw
from .kl_sweep import summarize_run

RDLogger.DisableLog("rdApp.*")

SEEDS = (1, 2, 3, 4)
BASE = ROOT / "results" / "factorial_ws" / "mle_withC"
ARM = ROOT / "results" / "tox_arm"
METRICS = ("scaffold_frac", "tanimoto_dist", "unique_smiles_frac", "valid_frac",
           "c_mean", "kl", "tpsa_spread_ratio", "logp_spread_ratio")
LAST_N = 20


def pool(d: Path, last_n: int = LAST_N) -> list[str]:
    """Canonical SMILES from the final `last_n` sampled steps, deduplicated.

    Taking the tail rather than the whole run keeps the comparison about where
    the policy ENDED, which is what the diversity and toxicity numbers in the
    findings doc refer to.
    """
    seen: set[str] = set()
    for f in sorted((d / "samples").glob("step*.json"))[-last_n:]:
        seen.update(s for s in json.loads(f.read_text()) if s)
    return [Chem.MolToSmiles(m)
            for m in (Chem.MolFromSmiles(s) for s in sorted(seen))
            if m is not None]


def paired_metrics() -> None:
    base = [summarize_run(BASE / f"seed{s}") for s in SEEDS]
    arm = [summarize_run(ARM / f"seed{s}") for s in SEEDS]
    print(f"{'metric':<20}{'base':>9}{'w_tox':>9}"
          + "".join(f"{'d_s' + str(s):>9}" for s in SEEDS)
          + f"{'mean':>9}{'up':>6}")
    print("-" * (20 + 18 + 9 * len(SEEDS) + 15))
    for m in METRICS:
        b = np.array([r[m] for r in base])
        x = np.array([r[m] for r in arm])
        d = x - b
        print(f"{m:<20}{b.mean():>9.3f}{x.mean():>9.3f}"
              + "".join(f"{v:>+9.3f}" for v in d)
              + f"{d.mean():>+9.3f}{f'{int((d > 0).sum())}/{len(SEEDS)}':>6}")


def posthoc_toxicity() -> None:
    from .tox import load_frozen_tox

    tox = load_frozen_tox(device="cpu")
    real = [s for s in _load_raw("bbbp").query("label==1")["smiles"].dropna()
            if Chem.MolFromSmiles(s)]
    print(f"\n{'seed':>6}{'base P(tox)':>13}{'w_tox P(tox)':>14}{'delta':>9}")
    print("-" * 42)
    d = []
    for s in SEEDS:
        b = float(tox(pool(BASE / f"seed{s}")).mean())
        x = float(tox(pool(ARM / f"seed{s}")).mean())
        d.append(x - b)
        print(f"{s:>6}{b:>13.4f}{x:>14.4f}{x - b:>+9.4f}")
    d = np.array(d)
    print("-" * 42)
    print(f"{'mean':>6}{'':>27}{d.mean():>+9.4f}   "
          f"({int((d < 0).sum())}/{len(SEEDS)} seeds less toxic)")
    print(f"\nreal BBBP+ drugs (n={len(real)}): P(tox) = {tox(real).mean():.4f}")
    print("Generated molecules BELOW this line are outside the range of "
          "approved\nCNS drugs on the scorer -- read that as Goodhart, not as safety.")


def main() -> int:
    missing = [d / f"seed{s}" for d in (BASE, ARM) for s in SEEDS
               if not (d / f"seed{s}" / "config.json").exists()]
    if missing:
        print("missing runs:\n  " + "\n  ".join(str(p) for p in missing))
        print("\nGenerate the arm with, for each seed s:\n"
              "  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python -m src.train_gan \\n"
              "      --steps 200 --seed s --group-size 64 --device cpu \\n"
              "      --warm-start results/generator/bbbp_mle_s<s>.pt \\n"
              "      --w-tox 0.5 --out-dir results/tox_arm/seed<s>")
        return 1
    # Guard the premise: an arm whose config still reports w_tox = 0 was run
    # under the dropped-flag bug, and every number below would again be a
    # baseline against itself.
    for s in SEEDS:
        cfg = json.loads((ARM / f"seed{s}" / "config.json").read_text())
        assert cfg["w_tox"] > 0, f"tox_arm/seed{s} recorded w_tox={cfg['w_tox']}"
        assert json.loads((BASE / f"seed{s}" / "config.json").read_text())["w_tox"] == 0
    paired_metrics()
    posthoc_toxicity()
    return 0


if __name__ == "__main__":
    sys.exit(main())
