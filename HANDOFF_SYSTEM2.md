# Handoff to system-2 — de-novo generation on BRICS

You own the generation half. System-1 owns the classifier and has frozen the
interface the two halves meet at. This document is that interface, plus the
measurements that should change what you build.

Read it before writing code. Several things here contradict the earlier design
discussion, and each contradiction is backed by a number from this repo.

---

## 0. First five minutes

Your clone probably points only at `sankarhariharan2007-kali`. The work is on
the fork, so add it, then branch off the contract — **do not commit to
`classifier-v2`**, system-1 owns that branch and two machines pushing to it
will collide.

```bash
git remote add myfork https://github.com/JSR996/bbb-gnn-permeability 2>/dev/null
git fetch myfork --tags
git checkout -b brics-generator contract-v2      # your branch, off the frozen contract
git config user.name  "JSR996"
git config user.email "shrenikrjnasa@gmail.com"
```

Then confirm the contract holds on this machine:

```bash
python -m src.featurize      # prints FEATURE_ID -- must be c0c3bbb8
python -m src.brics          # fragment substrate self-check
python -m src.reward         # THE GATE: must pass
python -m src.grpo
```

Push with `git push -u myfork brics-generator`. Pull system-1's later work with
`git fetch myfork && git merge myfork/classifier-v2` when you need the 10-seed
ensemble (section 8).

If `python -m src.reward` passes, your reward is wired to a classifier that
matches this code. If it raises a `feature_id` error, **stop** — do not work
around it. That error means the checkpoints and the featurizer disagree, and
anything you train against them is meaningless. Pull again, or ask system-1.

You never need to poll for system-1. Everything you need is already pushed.

---

## 1. The run contract

The two halves used to be coupled by nothing but `build_model`'s default
arguments. No caller ever passed `in_dim`, checkpoints are bare `state_dict`s,
and `metrics.json` recorded no architecture at all — so changing a default
silently invalidated all 73 committed checkpoints, and the failure surfaced as
`size mismatch for convs.0` far from its cause. `load_dataset` also keyed its
cache on the dataset name alone, so a featurizer change could return stale
graphs and never raise.

That is now closed at one chokepoint:

| piece | where | what it does |
|---|---|---|
| `FEATURE_ID` | `featurize.py` | sha1 of the featurizer's own output on a fixed probe set. **Derived, not declared** — a constant you must remember to bump is one that will be wrong exactly once. Currently `c0c3bbb8`. |
| contract fields | `train.py` → `metrics.json` | `feature_id`, `node_dim`, `edge_dim`, `hidden`, `num_layers`, `dropout`, `heads`, `norm`, `virtual_node`, `split` |
| `run_contract(run_dir)` | `featurize.py` | raises on a `feature_id` mismatch, **returns the recorded architecture** |
| cache key | `datasets.cache_path` | `bbbp-<FEATURE_ID>.pt` — a stale cache is not detected, it simply isn't found |

`reward.load_frozen_classifier` and `eval_external_holdout` both call
`run_contract` and rebuild each checkpoint the way it was *trained*, not the
way today's defaults happen to be. Verified both directions: a `results_v1/`
checkpoint is refused by name, the current ones load.

**If you add a feature that no entry in `featurize._PROBE` exercises, extend
`_PROBE` in the same diff.** Otherwise the hash does not move and the guard
goes blind to exactly the change it exists to catch.

### Frozen architecture

```
NODE_DIM = 39      EDGE_DIM = 7      FEATURE_ID = c0c3bbb8
hidden = 128       num_layers = 3    dropout = 0.3     heads = 4
norm = "layer"     virtual_node = False
split = "murcko+tanimoto@0.6"
reward ensemble = ("gin", "gat", "gine") on bbbp, seed 0
```

**The featurizer did not change.** It is still the same 39 dimensions. What
changed is the normalization, the split, and the fact that all of it is now
recorded. So your side breaks only if you change `featurize.py`.

---

## 2. LayerNorm replaced BatchNorm, and it matters most to you

`BatchNorm1d` made a molecule's logit depend on which molecules shared its
batch. For a classifier that is a training detail. For **your** reward it was a
correctness bug: the same molecule earned a different reward depending on which
candidates happened to land in its GRPO group, so the reward was not a function
of the molecule at all.

Measured by `python -m src.models`, gin on a 4-graph batch, train mode:

| norm | logit drift vs scoring alone |
|---|---|
| batch | **3.97e-02** |
| layer | 3.7e-09 |
| graph | 6.0e-08 |

**`eval()` is still load-bearing**, for a smaller reason: `nn.Dropout` keeps the
forward pass stochastic in train mode. Keep every `eval()` call and keep
`reward._self_check`'s alone-vs-crowded assertion — it is cheap, and it is what
catches a regression back to BatchNorm.

---

## 3. `src/brics.py` — the shared substrate

System-1 owns this file; you import it. One definition of "a fragment" on both
sides, or you optimize a vocabulary the classifier cannot read.

```python
from src.brics import decompose, reassemble, motif_graph, vocab, UNK

frags, junctions = decompose(mol)   # junction = (frag_a, dummy_a, frag_b, dummy_b, bond_type)
parent = reassemble(frags, junctions)
nodes, edges = motif_graph(mol)     # fragment SMILES + the cut bonds
v = vocab(smiles_list, min_count=3) # UNK is always id 0
```

`decompose` is **lossless**: 2039/2039 BBBP and 7805/7805 B3DB molecules
round-trip to an identical canonical SMILES.

Three measured facts that contradict the earlier design chat:

- **`BRICS.BRICSDecompose` has no `keepChiral` argument.** Verified against the
  installed RDKit. Chirality is preserved by default anyway (~97% of assigned
  centers survive).
- **"~150 fragments covers 95%" does not hold here.** At `min_count=3`: BBBP
  keeps 316 fragments covering **75.1%** of occurrences, B3DB 1178 covering
  **81.3%**.
- **12.8% of BBBP and 9.3% of B3DB molecules yield zero BRICS cuts.** The
  whole-molecule scaffold-core fallback is one molecule in ten, not an edge
  case. Your assembly engine must be able to emit a core with no junctions.

Also: the junction carries a **bond type**, and it is not decoration. BRICS cuts
type-7 alkene linkages, not only single bonds. Rebuilding every junction as
`SINGLE` silently hydrogenates 91 of BBBP's 2039 molecules —
`CN(C)CCC=C1c2ccccc2CCc2ccccc21` comes back as `...CCCC1...`. That bug was
caught by the round-trip assertion; do not reintroduce it in your assembler.

### `vocab()` takes YOUR corpus, on purpose

There is no committed shared vocabulary file, because the two sides need
different ones:

- **Classifier** (system-1): built from **training-fold molecules only**, per
  (dataset, seed). A vocabulary over the full dataset is transductive leakage
  under a scaffold split — the split exists to give test molecules unseen
  cores, and a global vocabulary hands every test fragment its own embedding
  slot. Rare fragments map to `UNK`.
- **Generator** (you): the whole corpus is fine, you make no held-out claim.
  But you need a **lower floor or none**: frequency pruning that is a harmless
  long tail for a classifier is a *reconstruction failure* for a generator,
  which cannot emit a fragment it has no token for.

Record whichever you used.

---

## 4. Three decisions that change what you build

### 4.1 Assembly-first. Do not generate a graph and then decompose it.

The action space **is** sequential assembly: pick an attachment point, pick a
compatible fragment, let RDKit form the bond. The molecule is the *output*.

The entire reason BRICS justifies this rebuild is that assembly is valid by
construction, which buys validity, a compact action space and a
synthesizability floor in one move. Generating an atom-level graph first and
decomposing it afterwards pays the full cost of the rebuild while keeping every
problem it was meant to solve: invalid graphs still reachable, no fragment
action space, and atom spam still available to the policy. It sounds like it
gives you both representations. It gives neither guarantee.

### 4.2 Stereochemistry is dropped. Do not build a chiral-pool reward.

Measured on this repo's data, 2026-10-10:

| | BBBP | B3DB |
|---|---|---|
| ≥1 assigned stereocenter | 31.2% | 54.6% |
| **genuine opposite-label stereoisomer pairs** | **0** | **57 groups / 7805 molecules** |
| specified-vs-unspecified annotation artifacts | 3 | 73 |

There is plenty of stereo *annotation* and almost no stereo *signal*. The
featurizer has no chirality features and will not get them.

**The trap, specifically.** `load_frozen_classifier` defaults to
`dataset="bbbp"`, and `train_gan.py` uses that default. A chiral-pool generator
scored by the BBBP ensemble is being rewarded by a model with **zero**
discriminating examples. You would get a flat reward curve and conclude
"stereochemistry doesn't help", when what actually happened is the reward
cannot see it. Either run any stereo arm against the **b3db** ensemble, or do
not run it. Decide before building the chiral pool, not after.

Emitting defined stereochemistry is still fine as a *correctness* property —
`decompose` preserves it, so fragments carry it for free. Just do not expect the
reward to select on it.

### 4.3 One featurization path, already consolidated

There used to be four entry points on generated SMILES: `reward.py`
canonicalized first, `train_gan.py` featurized the raw generator output in three
places. They agreed only by accident. They now all route through
`featurize.graphs_from_smiles(list)`, which canonicalizes and sanitizes once.

**Keep it that way.** If your new generator emits SMILES through a different
path, the reward and the discriminator start seeing different molecules, and it
presents as a training instability rather than as a bug.

---

## 5. What you own, and what you must not touch

**Yours:** `generator.py`, `train_gan.py`, `grpo.py`, `diversity.py`,
`postfilter.py`, `analyze_drift.py`, `kl_sweep.py`, and whatever new modules the
BRICS assembly engine needs.

**System-1's — a change here silently breaks the contract on the other
machine:** `featurize.py`, `split.py`, `models.py`, `models_edge.py`,
`models_hybrid.py`, `brics.py`, `train.py`.

If you need something changed in those, say so rather than editing — the whole
point of `FEATURE_ID` is that a divergence is loud, but it is only loud at the
moment someone reloads a checkpoint, which may be hours after the edit.

Suggested build order, none of which depends on the classifier:

1. BRICS assembly engine — attachment bookkeeping, fragment compatibility by
   BRICS type, bond formation, **canonical trajectory ordering** (populate
   attachment sites in a fixed index order so one molecule has one trajectory,
   not *k!*), `[STOP]`, terminal capping.
2. Hierarchical policy: core selection → (attachment point, fragment).
3. **Terminal-only reward.** Never score a molecule with open `[*]` tags; the
   classifier's prediction on a partial graph is meaningless.
4. MLE warm start on assembly traces decomposed from real molecules. The warm
   start is not optional — from a uniform policy every group is all-invalid,
   the group std is zero, and `group_advantages` correctly returns no gradient.

---

## 6. Things in this repo that will bite you

- **`results_v1/`** is the pre-contract archive. Nothing reads it and
  `run_contract` refuses it. Do not "fix" a load error by pointing at it.
- **`results/generator/`** was deliberately left in place. Those checkpoints are
  self-describing (they store their own vocab and dims) and feature-independent,
  so your existing warm starts still load.
- **D-MPNN was removed** (`models_dmpnn.py`, `train_dmpnn.py`) along with its
  references in `eval_external_holdout`, `paired_analysis` and `collect_all`.
- **The ensemble must share a fold.** `reward._self_check` asserts the
  `test_preds.npz` `idx` arrays are element-wise identical across `gin`, `gat`
  and `gine`. The seed reseeds the split, so members from different seeds have
  trained on each other's held-out molecules.
- **Reward hacking here is measured, not hypothetical.** Unguarded, scaffold
  diversity falls 0.8 → 0.07–0.14 over 200 steps in all 3 seeds while validity
  stays at 100%. Term ablation found **SA**, not the classifier, drives it.
  `kl_coef=0.3` and `min_heavy_atoms=10` are defaults because they are the only
  two interventions that survived validation.
- **Structural screening belongs in `postfilter.py`, after training.** A
  reactive-group penalty inside the reward taught the generator the checklist,
  not the chemistry: the targeted SMARTS it was penalized on fell 0.339 → 0.203
  while BRENK stayed flat.

---

## 7. Usage protocol

Both Claude sessions bill the same Pro account with 5-hour windows. Training
burns CPU, not tokens; writing code burns tokens. **Run one session at a time.**
System-1 used the overnight window to land this contract and launch its sweep;
that sweep needs no session attached.

---

## 8. Checkpoint status

The `gin`/`gat`/`gine` bbbp seed-0 checkpoints on this branch are a **smoke
triple, trained to unblock you**. They are real and correctly trained, but they
are one seed.

**Never report a number from them.** The 10-seed sweep overwrites them with
same-architecture results; the swap is a second tag, and the contract guard
makes a mismatch loud rather than silent.
