# BBB Permeability Prediction with Graph Neural Networks

Binary classification of blood–brain-barrier permeability from SMILES alone.
Four message-passing architectures — **GCN**, **GraphSAGE**, **GIN**, **GAT** —
are compared on two datasets under a scaffold split, then combined into ensembles.

```
SMILES → RDKit molecular graph → GNN → P(BBB permeable)
```

The only input feature is the SMILES string. No descriptors, no fingerprints —
every feature the models see is derived from the RDKit molecular graph.

## Setup

```bash
.venv/bin/pip install -r requirements.txt
```

PyTorch Geometric ≥ 2.3 uses `torch.scatter_reduce` natively, so the compiled
`torch-scatter` / `torch-sparse` extensions are **not** required.

## Running

```bash
python -m src.featurize                              # featurizer smoke test
python -m src.models                                 # forward-pass shape check
python -m src.datasets                               # build caches, verify splits
python -m src.train --model gcn --dataset bbbp --seed 0   # one run
python -m src.run_all                                # all 24 runs
python -m src.ensemble                               # combine predictions
python -m src.reward                                 # reward terms + frozen classifier
python -m src.grpo                                   # GRPO advantage + clipped loss
jupyter lab notebooks/results.ipynb                  # tables and figures
```

Runs default to CPU — these graphs are small enough that MPS kernel-launch
overhead makes it slower. `--device mps` is available to test that.

## Layout

| path | role |
|---|---|
| `src/featurize.py` | SMILES → PyG `Data`; 39-dim atom features, bond features stored but unused |
| `src/datasets.py` | load, clean, deduplicate, and cache both datasets |
| `src/split.py` | Bemis–Murcko scaffold split + leak assertions |
| `src/models.py` | one skeleton, four convolution operators |
| `src/train.py` | one `(model, dataset, seed)` run |
| `src/evaluate.py` | metrics; threshold chosen on validation only |
| `src/run_all.py` | 4 models × 2 datasets × 3 seeds |
| `src/ensemble.py` | soft vote, rank average, logistic stacking |
| `src/models_edge.py` | edge-aware operators: `gine`, `gat_edge`, `sage_edge` |
| `src/train_edge.py` | edge ablation runs; writes `results/edge_ablation/` |
| `src/reward.py` | frozen-classifier reward for generation (Eq. 5–7) |
| `src/grpo.py` | group-relative advantage and clipped surrogate (Eq. 6–7) |
| `notebooks/results.ipynb` | EDA, results tables, ROC/PR curves |

Results land in `results/<dataset>/<model>/seed<N>/`, aggregated into
`results/all_runs.csv` and `results/summary.csv`.

## Design decisions worth knowing

**One skeleton, four operators.** Depth, hidden width, normalization, readout,
classifier head and training loop are identical across all four models. Only the
convolution differs, so a performance gap is attributable to the aggregation
scheme rather than to incidental capacity differences. `GATConv` uses
`hidden // heads` channels per head so its concatenated output matches the
others rather than inflating fourfold.

**The scaffold split is balanced, not size-sorted.** The classic DeepChem
`ScaffoldSplitter` sorts scaffold groups by size and fills train first, leaving
val/test with only singleton scaffolds. On `BBBP.csv` that is degenerate: the
file is ordered so its entire back half is class 1, and both folds come out
**100% positive**, making ROC-AUC undefined. The balanced split (Chemprop-style)
shuffles scaffold groups under a per-seed RNG instead. `src/split.py` keeps the
size-sorted variant behind `balanced=False` so the artifact can be reproduced.

Consequence: absolute numbers are **not** comparable to the widely quoted
BBBP scaffold figures near 0.65–0.70, which come from the size-sorted protocol.
The comparable published range for the balanced protocol is around 0.90.

**The seed reseeds the split.** Each seed defines a different scaffold
partition, so the reported ± captures split variance as well as initialization
variance. It also means models can only be ensembled *within* a seed, where all
four share an identical fold.

**Bond features are computed but unused.** `Data.edge_attr` carries bond type,
conjugation and ring membership, but no model consumes it — GCN and GraphSAGE
structurally cannot, so feeding it to GIN/GAT alone would confound the
architecture comparison with an input-signal advantage. That follow-up ablation
now lives in `src/models_edge.py` (`gine`, `gat_edge`, `sage_edge`) and writes
to `results/edge_ablation/`, deliberately kept out of `MODELS` so the four-model
comparison stays edge-blind.

**The datasets overlap.** B3DB aggregates BBBP as one of its sources, and some
shared molecules carry contradictory labels. The two are therefore trained and
evaluated independently; training on one and testing on the other would be a
leaked evaluation.

## Molecule generation (GRPO)

The classifier above is the *scoring* half of a second project: a GAN generator
trained with GRPO to propose molecules that are predicted permeable, drug-like
and synthesizable. `src/reward.py` and `src/grpo.py` implement the parts of that
math formulation which do not depend on the generator architecture. The
generator, the discriminator, and the outer training loop are **not** here yet.

```
generator -> SMILES -> RDKit sanitization -> frozen classifier ensemble
                                          -> QED + SA
                                          -> discriminator     } -> R(m) -> advantage -> clipped update
```

Reward (Eq. 5) is a weighted sum of four terms: the discriminator score, the
frozen permeability classifier, QED, and a synthesizability score. `assemble()`
takes the discriminator score as an argument rather than owning `D_phi` —
`D_phi` trains alongside the generator and is the only non-stationary term, so
keeping it out lets the three stationary terms be tested on their own.

```python
from src.reward import load_frozen_classifier, score_terms, weights_at, assemble

classify = load_frozen_classifier(dataset="bbbp", models=("gin", "gat", "gine"), seed=0)
terms = score_terms(smiles_list, classify)
w = weights_at(step, w0, wf, schedule="linear", k_anneal=1000)
rewards = assemble(terms, d_scores, w)
```

**"Frozen" has to mean `eval()`, not just `requires_grad_(False)`.** The
classifier uses `BatchNorm1d`. In train mode its output depends on the batch it
is scored in, so the same molecule would earn a different reward depending on
which candidates happened to share its group — the reward would stop being a
function of the molecule at all. `load_frozen_classifier` sets `eval()` and
`_self_check` asserts a molecule scores identically alone and in company.

**Ensemble members must share a scaffold split.** The seed reseeds the split, so
averaging checkpoints from different seeds averages models that trained on each
other's held-out molecules. `results/edge_ablation/` reuses the core split
seed-for-seed, which is what makes a `gin`/`gat`/`gine` ensemble legitimate;
`_self_check` asserts the saved fold indices match rather than trusting it.

**Validity means sanitization, not grammar.** A SELFIES string always parses to
*some* molecule, so a generator's syntactic guarantee says nothing about whether
RDKit will sanitize the result. Only sanitization gates the reward; invalid
molecules collapse to a flat penalty and never reach the classifier.

**Weight schedules.** `weights_at` covers the fixed baseline, linear and cosine
annealing (Eq. 6), and the validity-gated switch (Eq. 7) that trades a step
count for a threshold on measured validity rate. It is a pure function of the
step and the caller's own validity history, so no schedule carries hidden state
between steps.

**One update rule for both generator designs.** The formulation leaves open
whether the generator is autoregressive (SELFIES, a sequence of `T` token
actions) or one-shot (MolGAN-style atom/bond tensors, a single joint action).
`clipped_loss` accepts `(N, T)` or `(N,)` log-probs, because the one-shot case
is exactly the `T = 1` degenerate case of the sequential one — a test asserts
the two agree. That fork therefore only has to be decided in the generator; the
reward and the update do not branch on it.

Two traps the self-checks pin down, both of which fail silently rather than
loudly:

- A group whose rewards are all equal — every molecule invalid, which is the
  normal early-training state — has zero standard deviation. Its advantages are
  forced to exactly zero instead of dividing by an epsilon and amplifying float
  noise into a gradient.
- Padded positions in a `(N, T)` batch hold arbitrary values, so `exp()`
  overflows to `inf` there and `inf * 0` is `nan`. Masking has to happen on the
  log-probability difference, *before* the exponential; masking the ratio
  afterwards cannot recover it.

## Data cleaning

| | rows in | invalid SMILES | dupes merged | label conflicts | rows out |
|---|---|---|---|---|---|
| BBBP | 2050 | 11 | 54 | 20 | **1965** (76.3% positive) |
| B3DB | 7807 | 2 | 0 | 0 | **7805** (63.5% positive) |

Deduplication is on the *canonical* SMILES, not the raw string. Molecules whose
duplicate copies disagree on the label are dropped entirely rather than guessed.
