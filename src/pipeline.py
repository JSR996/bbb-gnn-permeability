"""The whole pipeline, in order, resumable: classifier -> reward -> generator -> screen.

WHY THIS EXISTS. Every stage below already had a runner. What did not exist was
anything that crossed a stage boundary, so the recipe lived as prose in two
markdown files on two branches -- and `documentation/generator_pipeline.md` S3
opened with `git clone` and jumped straight to `python -m src.reward`, silently
assuming a trained classifier because the three ensemble checkpoints happen to
be committed. A clean clone followed it and got a reward built from someone
else's checkpoints. That seam is what this file closes.

THE TWO CLASSIFIERS, WHICH IS THE ONE THING TO UNDERSTAND HERE. The classifier
is trained twice because it has two incompatible jobs:

  reported  --norm batch  -> results/hybrid_batch/   BBBP 0.9214, Wang 0.8495
  reward    --norm graph  -> results/{bbbp,b3db}/    the frozen gin/gat/gine trio

BatchNorm scores higher and is disqualified as a reward: in train mode its
statistics come from the batch, so C(m) would depend on which other molecules
GRPO happened to sample alongside m. The reward ensemble is NOT a hybrid model
-- it is base/edge `gin`/`gat`/`gine` at seed 0 under GraphNorm, which is what
every committed generator result was trained against. Swapping it changes the
meaning of every number in results/brics_kl1/.

RESUME IS BY ARTIFACT, NOT BY BOOKKEEPING. Each stage asks "is the output
already on disk?" and skips if so, which is the same rule `run_all` uses. There
is no state file to get out of sync. The committed warm start, ensemble
checkpoints and three policy seeds mean a default run trains only the seven new
policy seeds.

    python -m src.pipeline --list            # what would run, what would skip
    python -m src.pipeline                   # everything, resuming
    python -m src.pipeline --only reward     # the gate alone
    python -m src.pipeline --from generator  # that stage onward

Self-check: `python -m src.pipeline --self-check` (no training).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- the settled generator configuration ------------------------------------
# kl_coef is the single most consequential number here. The repo default is 0.3,
# validated for the SELFIES action space, and it does NOT transfer: at 0.3 BRICS
# collapses to scaffold 0.643 with 26% literal duplicates per group, because the
# same coefficient yields ~10.7x the KL on this action space. Do not "restore
# the default".
KL_COEF = 1.0
STEPS = 250
GROUP_SIZE = 64
MIN_HEAVY_ATOMS = 10
THREADS = 6          # configuration, not tuning: OpenMP changes reduction order
REWARD_SEED = 0      # fixed before anyone looked at a test table
WARM_START = ROOT / "results" / "generator" / "bbbp_brics_pretrained.pt"
POLICY_SEEDS = list(range(10))
FROM_STEP = 150      # policy has settled by here in a 250-step run

BRICS_DIR = ROOT / "results" / "brics_kl1"
CANDIDATES = ROOT / "results" / "candidates" / "brics10"

# Policy seeds 0-2 are already committed, under two incompatible naming
# conventions, and `rs0_ws0`/`rs0_ws1` differ ONLY in warm start -- identical
# seed and reward_seed. So a run directory's NAME tells you nothing; every axis
# is read from its config.json. New runs use one convention, pseed<n>.
LEGACY_DIRS = {0: "rs0_ws0", 1: "gseed1", 2: "gseed2"}


def run_dir(seed: int) -> Path:
    return BRICS_DIR / LEGACY_DIRS.get(seed, f"pseed{seed}")


def policy_seed_dirs() -> list[Path]:
    """Every settled-point run at reward seed 0 and the default warm start.

    Read from config.json rather than inferred from the directory name, and
    deliberately excluding the rs1_* (reward seed 1) and *_ws1 (second warm
    start) dirs: those are different axes, not more policy seeds.
    """
    out = []
    for d in sorted(BRICS_DIR.glob("*")):
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        c = json.loads(cfg.read_text())
        same = (c.get("kl_coef") == KL_COEF
                and c.get("reward_seed") == REWARD_SEED
                and Path(str(c.get("warm_start", ""))).name == WARM_START.name)
        if same:
            out.append(d)
    return out


@contextlib.contextmanager
def _argv(args: list[str]):
    """Neutralize sys.argv for modules whose main() parses it directly.

    Cheaper than refactoring four argparse mains to take an argv parameter, and
    it keeps those files' CLI behaviour exactly as documented.
    """
    saved = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = saved


# --- stages -----------------------------------------------------------------
# Each is (name, plan_fn, run_fn). plan_fn returns a one-line status string and
# whether anything would actually run.

def plan_contract():
    return "always runs (cheap gate)", True


#: The featurizer hash every committed checkpoint was trained under. This is a
#: content hash of atom_features/bond_features on a fixed probe set, so it moves
#: by itself if the featurizer changes -- a constant someone must remember to
#: bump is a constant that will be wrong exactly once.
EXPECTED_FEATURE_ID = "c0c3bbb8"


def run_contract(force=False):
    import runpy
    from .featurize import EDGE_DIM, FEATURE_ID, NODE_DIM

    print(f"FEATURE_ID = {FEATURE_ID}  NODE_DIM = {NODE_DIM}  EDGE_DIM = {EDGE_DIM}")
    if FEATURE_ID != EXPECTED_FEATURE_ID:
        raise SystemExit(
            f"FEATURE_ID is {FEATURE_ID}, expected {EXPECTED_FEATURE_ID}. The "
            f"featurizer changed, so every committed checkpoint is stale and the "
            f"reward would be meaningless. Retrain, or check out the matching "
            f"commit -- do NOT proceed."
        )
    # Run each module's self-check the way its docstring documents it, rather
    # than calling a _self_check() that only two of the five actually expose.
    for mod in ("featurize", "split", "models", "brics", "datasets"):
        print(f"\n--- python -m src.{mod}")
        with _argv([f"src.{mod}"]):
            runpy.run_module(f"src.{mod}", run_name="__main__")


def _classifier_arms():
    """(label, globs, expected run count) for each arm stage 1 trains.

    The globs are scoped per dataset rather than `root/*/*/seed*`, because at
    the top of results/ that pattern also matches diagnostic trees --
    `norm_split_diag/` alone adds 40 runs and would report base as 120/80. The
    same over-broad glob once folded 30 diagnostic runs into the base table.
    """
    from .train_edge import EDGE_MODELS
    from .train_hybrid import results_dir
    from .models import MODELS
    from .run_all import DATASET_NAMES
    res = ROOT / "results"
    per = lambda root: [f"{ds}/*/seed*/metrics.json" for ds in DATASET_NAMES]
    return [
        ("base",         res,                        per(res), len(MODELS) * 2 * 10),
        ("edge",         res / "edge_ablation",      per(res), len(EDGE_MODELS) * 2 * 10),
        ("hybrid",       results_dir("graph"),       per(res), len(MODELS) * 2 * 10),
        ("hybrid_batch", results_dir("batch"),       per(res), len(MODELS) * 2 * 10),
    ]


def plan_classifier():
    bits, todo = [], False
    for label, root, globs, expected in _classifier_arms():
        have = sum(len(list(root.glob(g))) for g in globs)
        bits.append(f"{label} {have}/{expected}")
        todo |= have < expected
    return ", ".join(bits), todo


def run_classifier(force=False):
    from . import run_all, train_edge, train_hybrid
    f = ["--force"] if force else []
    with _argv(["run_all", "--threads", str(THREADS), *f]):
        run_all.main()
    train_edge.run_all()
    # Both hybrid arms. They differ ONLY in the norm, and hybrid_batch is the
    # reported classifier while hybrid/graph is the batch-independent sibling.
    for norm in ("graph", "batch"):
        train_hybrid.run_all(norm=norm, force=force)


def plan_reward():
    return "always runs (gate -- hard stop on failure)", True


def run_reward(force=False):
    from rdkit import Chem
    from .reward import load_frozen_classifier
    classify = load_frozen_classifier()
    # classify() takes Mol objects, not SMILES. Aspirin and caffeine only as a
    # liveness probe -- the real checks (feature_id per member, fold identity,
    # norm identity) already ran inside load_frozen_classifier.
    probe = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O",
             "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}
    c = classify([Chem.MolFromSmiles(s) for s in probe.values()])
    print("reward ensemble loaded: "
          + ", ".join(f"C({k})={v:.4f}" for k, v in zip(probe, c)))


def plan_generator():
    have = [s for s in POLICY_SEEDS if (run_dir(s) / "config.json").exists()]
    ws = "warm start present" if WARM_START.exists() else "WARM START MISSING (will pretrain, ~9 min)"
    missing = [s for s in POLICY_SEEDS if s not in have]
    return (f"{ws}; policy seeds {len(have)}/{len(POLICY_SEEDS)} done"
            + (f", to train: {missing}" if missing else "")), bool(missing)


def run_generator(force=False):
    from . import brics_generator
    from .train_gan import train

    if not WARM_START.exists() or force:
        print(f"MLE warm start -> {WARM_START.name} (~9 min, NOT optional)")
        brics_generator.pretrain(dataset="bbbp", epochs=30, seed=0,
                                 out_name=WARM_START.name)

    for seed in POLICY_SEEDS:
        out = run_dir(seed)
        if (out / "config.json").exists() and not force:
            print(f"policy seed {seed}: {out.name} already done, skipping")
            continue
        print(f"\npolicy seed {seed} -> {out.name}  (~6 min)")
        train(dataset="bbbp", steps=STEPS, group_size=GROUP_SIZE,
              kl_coef=KL_COEF, min_heavy_atoms=MIN_HEAVY_ATOMS,
              seed=seed, reward_seed=REWARD_SEED,
              warm_start=str(WARM_START), out_dir=out,
              threads=THREADS, generator="brics", device="cpu")


def plan_screen():
    dirs = policy_seed_dirs()
    done = (CANDIDATES / "candidates.csv").exists()
    return (f"{len(dirs)} run dirs -> {CANDIDATES.name}/"
            + (" (already screened)" if done else "")), not done


def run_screen(force=False):
    from .postfilter import main as postfilter_main
    dirs = policy_seed_dirs()
    if not dirs:
        raise SystemExit("no settled-point run dirs found; run the generator stage first")
    print(f"screening {len(dirs)} runs from step >= {FROM_STEP}: "
          f"{[d.name for d in dirs]}")
    postfilter_main(run_dirs=dirs, from_step=FROM_STEP, out=CANDIDATES,
                    dataset="bbbp", device="cpu",
                    min_heavy_atoms=MIN_HEAVY_ATOMS, top=25)


#: The headline scaffold statistic is the mean of the last 25 logged steps of a
#: run, each computed at a matched group of 64. Recovered by fitting the
#: committed 3-seed figure (0.8643, sd 0.0202, SEM 0.0117) rather than taken on
#: trust, so the 10-seed number is computed the same way as the one it extends.
HEADLINE_TAIL = 25


def scaffold_headline(dirs: list[Path]) -> tuple[float, float, dict[str, float]]:
    """(mean, sd, per-run) scaffold diversity over the last HEADLINE_TAIL steps.

    This is a PER-RUN statistic at matched n=64, averaged across runs. It is not
    a property of a pooled set: scaffold fraction reads 0.758 at n=199 and 0.595
    at n=982 on the same BBBP molecules, a swing larger than any effect being
    measured (limitations 4.3/4.4). Never recompute it on the pooled candidate
    set and quote it next to these numbers.
    """
    import numpy as np
    import pandas as pd
    per = {}
    for d in dirs:
        h = pd.read_csv(d / "history.csv")
        per[d.name] = float(h["scaffold_frac"].tail(HEADLINE_TAIL).mean())
    v = np.array(list(per.values()))
    return float(v.mean()), float(v.std(ddof=1)), per


def plan_report():
    return "always runs (read-only aggregation)", True


def run_report(force=False):
    import numpy as np
    from . import collect_all, eval_external_holdout, paired_analysis

    dirs = policy_seed_dirs()
    if dirs:
        mean, sd, per = scaffold_headline(dirs)
        n = len(dirs)
        print(f"\nscaffold diversity, per-run at matched n=64, last "
              f"{HEADLINE_TAIL} steps, over {n} policy seeds:")
        print(f"  mean {mean:.4f}  sd {sd:.4f}  SEM {sd / np.sqrt(n):.4f}")
        for k, v in sorted(per.items()):
            print(f"    {k:<10} {v:.4f}")

        # The n=3 subset must still reproduce the committed figure exactly. If
        # it does not, this statistic is being computed differently from the
        # one it claims to extend, and the 10-seed number is not comparable --
        # a precise wrong headline, which is worse than an imprecise right one.
        legacy = [d for d in dirs if d.name in set(LEGACY_DIRS.values())]
        if len(legacy) == 3:
            m3, sd3, _ = scaffold_headline(legacy)
            print(f"  n=3 subset check: {m3:.4f} +/- {sd3:.4f} "
                  f"(committed: 0.8643 +/- 0.0202)")
            assert abs(m3 - 0.8643) < 5e-4 and abs(sd3 - 0.0202) < 5e-4, (
                f"n=3 subset gives {m3:.4f}/{sd3:.4f}, not 0.8643/0.0202 -- the "
                f"statistic changed, so the {n}-seed figure is not comparable"
            )

    with _argv(["eval_external_holdout"]):
        eval_external_holdout.main()
    with _argv(["collect_all"]):
        collect_all.main()
    paired_analysis.main(all_sources=True, metric="roc_auc")


STAGES = [
    ("contract",   plan_contract,   run_contract),
    ("classifier", plan_classifier, run_classifier),
    ("reward",     plan_reward,     run_reward),
    ("generator",  plan_generator,  run_generator),
    ("screen",     plan_screen,     run_screen),
    ("report",     plan_report,     run_report),
]
NAMES = [s[0] for s in STAGES]


def show_plan() -> None:
    print(f"{'stage':<12} {'would run':<10} status")
    print("-" * 78)
    for name, plan, _ in STAGES:
        status, todo = plan()
        print(f"{name:<12} {'yes' if todo else 'no':<10} {status}")


def _self_check() -> None:
    """Checks the wiring, not the science: no training, no checkpoint loads."""
    assert NAMES == ["contract", "classifier", "reward", "generator",
                     "screen", "report"], NAMES

    # The argv guard must restore sys.argv even when the body raises, or a
    # later stage parses the wrong arguments and the failure is far away.
    saved = sys.argv
    with contextlib.suppress(RuntimeError):
        with _argv(["x"]):
            assert sys.argv == ["x"]
            raise RuntimeError("boom")
    assert sys.argv is saved, "argv not restored after an exception"

    # Legacy names must map to the right policy seeds, since inferring them
    # from the directory name is exactly what goes wrong here.
    assert run_dir(0).name == "rs0_ws0" and run_dir(2).name == "gseed2"
    assert run_dir(7).name == "pseed7"

    # The settled-point selector must exclude the other two axes. rs1_* is a
    # different reward seed, *_ws1 a different warm start; neither is a policy
    # seed, and pooling them would silently widen the "10 seeds" claim.
    if BRICS_DIR.exists():
        picked = {d.name for d in policy_seed_dirs()}
        for excluded in ("rs1_ws0", "rs1_ws1", "rs0_ws1"):
            if (BRICS_DIR / excluded / "config.json").exists():
                assert excluded not in picked, f"{excluded} is not a policy seed"
        if (BRICS_DIR / "rs0_ws0" / "config.json").exists():
            assert "rs0_ws0" in picked, "policy seed 0 missing from the screen set"

    # No arm may count MORE runs than its grid holds. An over-count means the
    # glob is reaching into a sibling tree -- `results/*/*/seed*` picks up
    # norm_split_diag/ and reported base as 120/80 until the globs were scoped.
    for label, root, globs, expected in _classifier_arms():
        have = sum(len(list(root.glob(g))) for g in globs)
        assert have <= expected, f"{label}: {have} runs found, grid holds {expected}"

    # Every stage must report a plan without touching the network or training.
    for name, plan, _ in STAGES:
        status, todo = plan()
        assert isinstance(status, str) and isinstance(todo, bool), name

    assert KL_COEF == 1.0, "kl_coef 0.3 collapses BRICS; see the module docstring"

    # The headline statistic must reproduce the committed 3-seed figure on the
    # three committed runs. This is the check that keeps the 10-seed number
    # commensurable with the one it supersedes.
    legacy = [BRICS_DIR / n for n in LEGACY_DIRS.values()]
    if all((d / "history.csv").exists() for d in legacy):
        m, sd, _ = scaffold_headline(legacy)
        assert abs(m - 0.8643) < 5e-4, f"3-seed scaffold mean is {m:.4f}, not 0.8643"
        assert abs(sd - 0.0202) < 5e-4, f"3-seed scaffold sd is {sd:.4f}, not 0.0202"

    print("pipeline self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--self-check", action="store_true", help="check wiring, run nothing")
    ap.add_argument("--only", choices=NAMES, help="run exactly one stage")
    ap.add_argument("--from", dest="from_stage", choices=NAMES,
                    help="run this stage and everything after it")
    ap.add_argument("--force", action="store_true",
                    help="retrain even where an artifact already exists")
    args = ap.parse_args()

    if args.self_check:
        return _self_check()
    if args.list:
        return show_plan()

    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    elif args.from_stage:
        selected = STAGES[NAMES.index(args.from_stage):]
    else:
        selected = STAGES

    print(f"stages: {' -> '.join(s[0] for s in selected)}\n")
    for name, _, run in selected:
        print(f"\n{'=' * 78}\n== {name}\n{'=' * 78}")
        t0 = time.time()
        run(force=args.force)
        print(f"-- {name} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
