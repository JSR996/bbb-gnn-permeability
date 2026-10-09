# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # requirements.lock pins the verified versions
```

PyG ≥ 2.3 uses `torch.scatter_reduce` natively — the compiled `torch-scatter` /
`torch-sparse` extensions are *not* needed. Everything defaults to CPU; these
graphs are small enough that MPS kernel-launch overhead is a net loss.

## There is no test suite

No pytest, no CI, no Makefile. Instead **every module in `src/` is runnable with
no arguments and runs its own `_self_check()`** — assertions over the invariant
that module exists to protect (e.g. `python -m src.reward` asserts a molecule
scores identically alone and in a group; `python -m src.grpo` asserts the
masking survives padded log-probs). That is the test suite.

Follow the convention when adding a module: a `_self_check()` under
`if __name__ == "__main__"` that fails loudly if the module's core invariant
breaks. `python -m src.train_gan` with no `--steps` runs the self-check rather
than training.

```bash
python -m src.featurize      # featurizer
python -m src.models         # forward-pass shapes
python -m src.datasets       # build caches + verify splits
python -m src.reward && python -m src.grpo
```

## Common commands

```bash
# classifier
python -m src.train --model gcn --dataset bbbp --seed 0   # one run
python -m src.run_all                                     # all 24 base runs
python -m src.train_edge --run-all                        # 18
python -m src.train_dmpnn --run-all                       # 6
python -m src.train_hybrid --run-all                      # 24
python baselines/descriptor_baseline_v2.py                # LightGBM, no torch

# evaluation / aggregation
python -m src.external_holdout        # rebuild leak-free holdouts (RDKit only)
python -m src.eval_external_holdout   # score every checkpoint on them
python -m src.paired_analysis --all-sources
python -m src.collect_all             # -> results/master_comparison.csv

# generation (cold start from a fresh clone is these two lines)
python -m src.generator --pretrain    # MLE warm start, ~1 min CPU
python -m src.train_gan --steps 200   # GRPO, ~0.7 s/step at group 64
```

Runs write to `results/<family>/.../seed<n>/`; classifier checkpoints are
committed, so the reward half never needs retraining.

## Architecture

```
SMILES → RDKit graph (39-dim atom features) → GNN → P(BBB permeable)
                                                  ↘ frozen → reward for a GRPO generator
```

Two halves. The **classifier** is a controlled four-operator comparison
(GCN / GraphSAGE / GIN / GAT) sharing one skeleton — depth, width, norm,
readout, head and training loop identical, only the conv differs, so a gap is
attributable to aggregation rather than capacity. Extensions each live in their
own `models_*.py` + `train_*.py` pair (`_edge`, `_dmpnn`, `_hybrid`) and write to
their own `results/` subtree, deliberately kept out of `MODELS` so the core
comparison stays edge-blind and descriptor-blind. The **generator** is a GRU
over SELFIES conditioned on `z`, trained by GRPO against the frozen classifier
ensemble plus QED, SA and a discriminator.

`src/datasets.py` owns `ROOT` (repo root) and the processed-graph cache; every
module reads data through it rather than building paths itself.

### Invariants that are load-bearing, not stylistic

Violating any of these silently invalidates results rather than raising:

- **The seed reseeds the scaffold split.** Each seed is a different partition,
  so reported ± includes split variance, and models may only be ensembled
  *within* a seed. `results/edge_ablation/` deliberately reuses the core split
  seed-for-seed so a `gin`/`gat`/`gine` ensemble is legitimate; `reward.py`'s
  self-check asserts the saved fold indices match rather than trusting it.
- **The split is balanced, not size-sorted** (`src/split.py`, `balanced=True`).
  DeepChem's size-sorted splitter makes `BBBP.csv` degenerate — the file's back
  half is all class 1, so val/test come out 100% positive and ROC-AUC is
  undefined. Consequence: absolute numbers are **not** comparable to the
  widely quoted 0.65–0.70 BBBP figures. Comparable range is ~0.90.
- **Frozen means `eval()`, not just `requires_grad_(False)`.** The classifier
  and `D_phi` use `BatchNorm1d`; in train mode a molecule's reward would depend
  on which candidates shared its group, so the reward would stop being a
  function of the molecule.
- **Thread count is part of the configuration.** OpenMP changes float reduction
  order and the GRPO loop is chaotic. `OMP_NUM_THREADS` is unreliable (36
  silently binds 18 on the dev machine); pass `--threads` so
  `torch.set_num_threads()` sets it in-process. A driver that sets the env var
  and trusts it contaminated the first term ablation.
- **Paired designs.** Warm-start variance is ~2× GRPO-seed variance, so arm
  comparisons pair on *both* the seed and the per-seed warm start. Pairing on
  only the seed is what made the VAE comparison read as a clean regression when
  it was noise. Run `python -m src.paired_analysis` before claiming any ranking.
- **BBBP and B3DB overlap** (B3DB aggregates BBBP, with some contradictory
  labels). Train and evaluate them independently; cross-testing is leakage.
- **Adenot/Wang are never training data.** The leak filter drops ~96.5% of each,
  leaving n = 59 / 55.

### Defaults encode validated findings

`--kl-coef 0.3` and `--min-heavy-atoms 10` are defaults because they are the
only two interventions that survived validation; everything else that was tried
is off for a recorded reason (logit transform, Tox21 term, alert penalty,
geometric aggregation each made some measured instrument worse). Don't "clean
up" a default without reading the reason — `documentation/limitations.md` is a
numbered ledger where every item carries a status tag
(**[measured] / [mitigated] / [suspected] / [untested]**) and nothing is marked
resolved.

Reward hacking here is measured, not hypothetical: unguarded, scaffold
diversity falls 0.8 → 0.07–0.14 over 200 steps in all 3 seeds while validity
stays at 100%. **Generated molecules are unvalidated and are not synthesis
candidates.** Term ablation found SA, not the classifier, drives the collapse.
Structural screening lives in `src/postfilter.py` *after* training on purpose —
a filter inside the reward taught the generator the checklist, not the
chemistry (targeted SMARTS 0.339 → 0.203 while BRENK stayed flat).

## Writing conventions

Module docstrings here are long and carry the *why*, including the measurement
that motivated the module and what it explicitly does not fix. Several open with
a retraction of an earlier result and the reason it was void. Match that: when a
claim is in a docstring or in `documentation/`, it is expected to be backed by a
number produced in this repo.

Stale: `cns_mpo_reward.py` (repo root) and `data/raw/BBB.csv` are imported and
read by nothing. `baselines/legacy/` uses folds that do **not** match the GNN
runs — never compare against it.
