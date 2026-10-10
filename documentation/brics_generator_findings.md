# BRICS assembly generator: what it bought, and what it cost

First real runs of the BRICS fragment-assembly generator against the SELFIES
token generator, under one reward. The engine works and validity is near-total.
**Structural diversity is substantially worse, and the mechanism is the compact
action space that was the selling point.**

Everything below is 8 GRPO runs of 250 steps: a 2x2 of reward seed x warm start
per generator family, group 64, `kl_coef=0.3`, `min_heavy_atoms=10` (the
validated defaults), `--threads 6`, GRPO `--seed 0`, GraphNorm reward ensemble.
Artifacts in `results/brics_grpo/` and `results/selfies_control/`.

**What is committed, and what is not.** `history.csv`, `config.json` and
`samples/` for all 8 runs -- every number below is derived from those, and
`samples/` is what `src/postfilter.py` pools. The 8 generator and 8
discriminator checkpoints (50 MB of the 56) are **not** committed: they are
deterministic given the committed warm start, `--seed`, and the pinned thread
count recorded in each `config.json`, so re-running costs ~18 minutes against a
standing limitation about committed artifact size. `src/postfilter.py`'s
`bulk_sample` is the one thing that needs them; the primary screening path reads
`samples/` and works as committed.

---

## 1. The engine reconstructs exactly

`python -m src.brics_assembly` traces every BBBP molecule through the assembly
automaton and replays it: **2039/2039 exact**, 0 untraceable, 0 mismatch,
longest trace 24 actions. Two corrections were needed to get there, both
measured rather than assumed.

**The incoming slot must be part of the action.** Picking the lowest-index
compatible dummy reconstructs only **63.5%**. 948 of 2612 fragments have more
than one slot compatible with some site type, and the wrong pick leaves a
different site set open, so every later lookup is against the wrong site and the
error cascades into a different molecule --
`CC(C)NCC(O)COc1cccc2ccccc12` came back as `CC(C)NCC(O)CNC(C)C`. The slot is
information only the policy can supply. 6051 actions instead of 2614.

**Stereochemistry has to be stripped.** With the slot fixed, reconstruction
reached **81.0%**, and all 387 residual failures were stereo-only -- zero
constitutional. `reassemble` appends the new bond after the dummy is removed, so
the anchor's chiral tag is read against a different neighbour order and the
parity flips. Stripping costs nothing: the featurizer has no chirality features,
so `C(m)` changes by exactly 0.00e+00. A defined but WRONG centre is a false
claim about a molecule for no signal; an undefined one is merely silent.

## 2. "Valid by construction" is narrower than claimed

Type compatibility plus an explicit bond order guarantees **connectivity and
valence, not aromaticity**. Joining two type-compatible aromatic fragments can
yield a ring system RDKit cannot kekulize. Every inspected failure is a
`KekulizeException`; none are valence.

| policy | sanitizes |
|---|---|
| untrained | 95.39% (118/2560 fail) |
| MLE warm start | 100% (0/2048) |
| trained 20-step | 100% (0/2048) |
| during a 250-step run | 99.98% (min 98.44% at one step) |

So the floor is ~95.4%, not 100%, and a drifting policy can still reach a
failure. `AssemblyState.to_mol` returns `""` and it routes through the
invalidity hurdle, which is the correct direction.

**Part of what the warm start teaches is toolkit avoidance, not chemistry.** The
95.4% -> 100% gain is the policy learning to avoid fragment pairs RDKit cannot
kekulize. That is a representation limit of the cheminformatics stack. Do not
report it as "the policy learned valid chemistry".

## 3. The trade, paired on reward seed and warm start

Each BRICS cell is matched to a SELFIES cell on both reward seed and warm start,
under the same GraphNorm ensemble. Window = mean of steps 125-149; per-step
`scaffold_frac` at n=64 is far too noisy to read pointwise (one run swings
0.250 -> 0.797 in 25 steps).

| metric | BRICS − SELFIES | SEM | ratio |
|---|---|---|---|
| `scaffold_frac` | **−0.2701** | 0.0170 | 15.8 |
| `unique_smiles_frac` | **−0.3370** | 0.0313 | 10.8 |
| `tanimoto_dist` | −0.0901 | 0.0107 | 8.4 |
| `rdkit_valid_frac` | +0.0052 | 0.0002 | 25.3 |
| `undersized_frac` | −0.0145 | 0.0008 | 18.2 |
| `c_mean` | +0.0341 | 0.0051 | 6.7 |
| `reward_mean` | +0.1797 | 0.0106 | 17.0 |
| `kl` | +0.7004 | 0.0699 | 10.0 |

n=4 paired cells, 3 df; t(0.05, 3) = 3.18, so every row clears it.

**BRICS wins on everything the reward measures and loses on everything it does
not.** That is the Goodhart signature, stated precisely rather than inferred.

Trajectories, averaged over the four cells:

| family | scaffold @0-25 | @125-150 | @225-250 | C @125-150 |
|---|---|---|---|---|
| BRICS | 0.841 | 0.561 | 0.643 | 0.933 |
| SELFIES | 0.810 | 0.832 | 0.821 | 0.898 |

SELFIES holds diversity flat, as the gate arm was validated to do. BRICS dips to
0.561 by step 150 and partially recovers to 0.643 by 250 -- it does not reach
the 0.07-0.14 of the ungated SELFIES collapse, so this is loss, not collapse.

### The mechanism is exact re-emission

`unique_smiles_frac` is 0.61-0.75 for BRICS against 0.99-1.00 for SELFIES: a
quarter to a third of every group is literal duplicates. Confirmed not to be an
artifact of the fraction's n-dependence -- `n_valid` is 64.0 vs 63.6-63.9, and
absolute distinct scaffolds say the same thing (34-39 vs 53).

This is a different failure signature from the SELFIES collapse, where the
`unique` metric lagged the true collapse by 88-154 steps. Here it moves
concurrently, because a compact discrete action space can revisit a molecule
*exactly*, where a 77-token string space essentially cannot.

**The compactness that prevents atom spam is the same property that permits
re-emission.** The action-space argument for BRICS is not free; it has this cost
on the other side.

## 4. `kl_coef=0.3` does not transplant across action spaces

| family | KL @125-150 | `clip_frac` | actions/molecule | action space |
|---|---|---|---|---|
| BRICS | 0.773 | 0.0217 | 5.5 | 6051 |
| SELFIES | 0.073 | 0.0056 | 38.3 | 77 |

Same nominal coefficient, **10.7x the KL**. The clipped loss is a token-level
masked mean, so per-action KL over 5.5 steps of a 6051-way softmax is not on the
scale of 38.3 steps of a 77-way one, and a single outlier is diluted by 5.5
rather than 38.3. `kl_coef=0.3` survived validation *for SELFIES*; inheriting
the number did not inherit the regularization strength. Any KL claim about the
BRICS generator needs its own sweep.

**The k3 estimator produces extreme outliers here.** `rs0_ws1` recorded
`kl=17327.56` at step 7, with the next step back at 0.0145. The estimator is
`exp(d) - d - 1` with `d = logp_ref - logp`; one action where the policy
disagrees with `pi_ref` by ~10 nats gives exactly that magnitude. At
`kl_coef=0.3` that step's loss was ~5198, i.e. entirely KL, visible as
`clip_frac` 0.169 and `ratio_mean` 0.845. Gradient clipping at 1.0 bounded it
and the run recovered. The estimator was **not** changed, because that would
break comparability with the committed GRPO runs.

## 5. What these runs cannot tell you

- **One GRPO seed.** All 8 runs use `--seed 0`. The paired difference is
  conditional on one rollout-noise draw. The effect is large enough
  (15.8 SEM on scaffold) that this is unlikely to flip it, but it is untested.
- **The 2x2 cannot separate its two noise sources.** One run per cell means zero
  residual df. Point estimates on final scaffold: reward-seed effect −0.072,
  warm-start effect −0.081 -- the same order as each other and as the
  between-cell spread. Separating them needs replication within cells. What the
  matrix does buy is that the headline direction holds in all four cells.
- **The reward change was checked and was negligible.** The committed gate arms
  predate the GraphNorm ensemble, so the first cross-family comparison was
  confounded. Re-running SELFIES under the current reward moved `scaffold_frac`
  by **+0.003** (0.832 vs 0.829), so the committed arms were in fact
  comparable -- but that is a measurement, not an assumption.
- **No claim about synthesizability.** Assembly from BRICS fragments gives a
  retrosynthetic plausibility floor by construction; nothing here tests it.

## 6. What this does not undermine

The case for BRICS was never classifier accuracy -- the hierarchical motif
family settled that separately (replacing atom detail with fragment structure:
−0.1161 ± 0.0107; adding it on top: −0.0051 ± 0.0047, i.e. nothing). The case
was validity by construction and an action space that cannot emit atom spam.

Validity holds: +0.0052 on `rdkit_valid_frac` and −0.0145 on `undersized_frac`,
both against a SELFIES arm that already had the size gate. The action-space
argument is structural and needs no measurement -- a policy over
(attachment point, fragment) has no action that emits a lone atom.

What has changed is that the diversity cost is now measured, large, and
mechanistically explained. A BRICS arm is not a drop-in improvement on the
SELFIES gate arm; it is a different point on the validity/diversity trade, and
the defaults it inherited were tuned for the other generator.
