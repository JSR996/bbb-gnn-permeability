# BRICS assembly generator: what it bought, and what it cost

The BRICS fragment-assembly generator against the SELFIES token generator, under
one reward. The engine reconstructs exactly and validity is near-total. **At each
family's own tuned KL coefficient the two are not distinguishable on structural
diversity**; BRICS trades classifier confidence for validity at the same total
reward.

**This supersedes an earlier version of this document that reported BRICS losing
0.27 of scaffold diversity.** That measurement was real but its interpretation
was wrong: it was taken at `kl_coef=0.3`, a value validated for SELFIES, and the
entire gap closes when the coefficient is set for this action space. Section 4
is the retraction.

Runs: 24 GRPO runs of 250 steps, group 64, `min_heavy_atoms=10`, `--threads 6`,
GRPO `--seed 0`, GraphNorm reward ensemble. A 2x2 of reward seed x warm start per
arm (`results/brics_grpo` kl=0.3, `results/brics_kl1` kl=1.0,
`results/selfies_control` kl=0.3) plus a 4x3 `kl_coef` x `ent_coef` sweep
(`results/brics_sweep`).

**What is committed.** `history.csv`, `config.json` and `samples/` for every
run -- every number here derives from those, and `samples/` is what
`src/postfilter.py` pools. The generator and discriminator checkpoints are not
committed: they are deterministic given the committed warm start, `--seed` and
the pinned thread count in each `config.json`, and they would add ~150 MB
against a standing limitation about artifact size.

---

## 1. The engine reconstructs exactly

`python -m src.brics_assembly`: **2039/2039 exact**, 0 untraceable, 0 mismatch,
longest trace 24 actions. Two corrections were needed, both measured.

**The incoming slot must be part of the action.** Picking the lowest-index
compatible dummy reconstructs only **63.5%**. 948 of 2612 fragments have more
than one slot compatible with some site type; the wrong pick leaves a different
site set open, so every later lookup is against the wrong site and the error
cascades -- `CC(C)NCC(O)COc1cccc2ccccc12` came back as `CC(C)NCC(O)CNC(C)C`.
6051 actions instead of 2614. Do not reintroduce the heuristic.

**Stereochemistry has to be stripped.** With the slot fixed, reconstruction
reached **81.0%** and all 387 residual failures were stereo-only, zero
constitutional: `reassemble` appends the new bond after the dummy is removed, so
the anchor's chiral tag is read against a different neighbour order and the
parity flips. Stripping is free -- the featurizer has no chirality features, so
`C(m)` changes by exactly 0.00e+00 on sulpiride, L-DOPA and quinine. (The graph
tensors are *not* element-wise equal, because canonical re-ordering permutes atom
rows; they match once sorted, and the readout is permutation-invariant. Compare
sorted tensors or compare outputs.) A defined but WRONG centre is a false claim
for no signal; an undefined one is merely silent.

## 2. "Valid by construction" is narrower than it sounds

Type compatibility plus an explicit bond order guarantees **connectivity and
valence, not aromaticity**. Joining two type-compatible aromatic fragments can
yield a ring system RDKit cannot kekulize. Every inspected failure is a
`KekulizeException`; none are valence.

| policy | sanitizes |
|---|---|
| untrained | 95.39% (118/2560 fail) |
| MLE warm start | 100% (0/2048) |
| 250-step runs | 99.98% (min 98.44% at one step) |

The floor is ~95.4%, not 100%. `AssemblyState.to_mol` returns `""` and the
molecule routes through the invalidity hurdle.

**Part of what the warm start teaches is toolkit avoidance, not chemistry.** The
95.4% -> 100% gain is the policy learning to avoid pairs RDKit cannot kekulize,
which is a representation limit of the stack. Do not report it as "the policy
learned valid chemistry".

## 3. The headline: comparable diversity, a validity/confidence trade

Steps 225-250, means over 4 cells. Warm-start reference measured at the same
n=64: scaffold 0.854, unique 0.990.

| arm | scaffold | unique | C | reward | rdkit_valid |
|---|---|---|---|---|---|
| BRICS kl=0.3 (inherited) | 0.643 | 0.744 | 0.932 | 2.280 | 0.9998 |
| **BRICS kl=1.0 (tuned)** | **0.864** | **0.982** | 0.847 | 2.090 | 0.9998 |
| SELFIES kl=0.3 (validated) | 0.821 | 0.995 | 0.900 | 2.080 | 0.9946 |

Paired on reward seed and warm start, BRICS kl=1.0 against SELFIES kl=0.3,
n=4 cells, 3 df, t_crit = 3.182:

| metric | diff | SEM | t | p | verdict |
|---|---|---|---|---|---|
| `scaffold_frac` | +0.0431 | 0.0154 | 2.79 | 0.068 | **not distinguishable** |
| `reward_mean` | +0.0096 | 0.0129 | 0.74 | 0.512 | **not distinguishable** |
| `undersized_frac` | −0.0046 | 0.0015 | −3.09 | 0.054 | not distinguishable |
| `unique_smiles_frac` | −0.0134 | 0.0016 | −8.60 | 0.003 | differs, negligibly |
| `tanimoto_dist` | +0.0045 | 0.0010 | 4.36 | 0.022 | differs, negligibly |
| `c_mean` | −0.0536 | 0.0066 | −8.16 | 0.004 | differs |
| `rdkit_valid_frac` | +0.0052 | 0.0002 | 25.87 | 0.000 | differs |

So at its own coefficient BRICS **matches** SELFIES on scaffold diversity and
total reward, is marginally behind on exact-duplicate rate (0.982 vs 0.995),
buys **+0.0052 validity**, and pays **−0.054 of classifier confidence**. The
trade is validity for confidence, at equal reward -- not diversity for
confidence.

## 4. Retraction: the diversity collapse was a mis-set coefficient

The earlier version of this document reported, correctly as a measurement:
scaffold −0.2701 +/- 0.0170 and `unique_smiles_frac` −0.3370 +/- 0.0313 for
BRICS against SELFIES, both at `kl_coef=0.3`. It then explained the duplicates
as structural:

> "The compactness that prevents atom spam is the same property that permits
> re-emission."

**That explanation was wrong.** Raising the coefficient alone recovers it, paired
on the same cells:

| BRICS kl=1.0 − kl=0.3 | diff | SEM | p |
|---|---|---|---|
| `scaffold_frac` | +0.2212 | 0.0267 | 0.004 |
| `unique_smiles_frac` | +0.2380 | 0.0313 | 0.005 |
| `c_mean` | −0.0850 | 0.0099 | 0.003 |
| `reward_mean` | −0.1907 | 0.0121 | 0.001 |

Re-emission was an under-regularized policy concentrating, not a property of a
compact action space. `kl_coef=0.3` survived validation *for SELFIES* and
transplanted as a number rather than as a strength: at the same coefficient
BRICS showed 10.7x the KL, because the clipped loss is a token-level masked mean
and BRICS takes ~5.1 decisions over a 6051-way softmax against SELFIES' ~38.6
over 77. **Any hyperparameter validated on one action space has to be re-swept on
another.**

## 5. The sweep: KL is the lever, entropy is not

4x3, `kl_coef` x `ent_coef`, scaffold_frac @225-250 (warm start 0.854):

| kl \ ent | 0.0 | 0.02 | 0.1 |
|---|---|---|---|
| 0.1 | 0.464 | 0.411 | 0.613 |
| 0.3 | 0.723 | 0.547 | 0.809 |
| 1.0 | **0.887** | **0.891** | 0.874 |
| 3.0 | **0.906** | 0.879 | 0.873 |

`unique_smiles_frac` on the same grid reaches 0.986-0.993 for every kl >= 1.0,
i.e. the duplicate problem disappears entirely.

**Entropy does nothing once KL is adequate.** At kl=1.0 scaffold is
0.887/0.891/0.874 across ent 0/0.02/0.1 -- flat. It helps only where KL is
missing (0.723 -> 0.809 at kl=0.3), so it substitutes for absent regularization
rather than adding to it. The mechanism is visible in the logs: per-step entropy
rises to ~2.5 at kl >= 1.0 **with no bonus at all**, above the warm start's
2.130, so adequate KL already holds per-step entropy up and an explicit bonus has
nothing left to do.

Sweeping KL alone would have found the same setting. The second axis is what
makes "entropy was never the knob" a finding instead of an assumption.

**Entropy was worth testing on theory.** With ~5.1 decisions against ~38.6, the
same per-step entropy gives far lower molecule-level entropy -- measured, BRICS
has 1.8x the per-step entropy (2.130 vs 1.170) and 24% of the molecule-level
entropy (10.79 vs 45.11). That predicted an entropy bonus should help. It did
not, because KL reaches the same axis indirectly.

**The warm start was not the bottleneck.** A hypothesis worth ruling out was that
`pi_ref` is itself peaked, in which case no `kl_coef` could recover diversity it
never had. At matched n=64 the BRICS warm start is scaffold 0.854 / unique 0.990
against SELFIES' 0.811 / 0.982 -- *more* diverse. The runs lost diversity their
reference genuinely had, which is why raising the anchor's weight recovered it.
Measuring this at n=512 instead gives 0.654 vs 0.765 and inverts the conclusion;
`scaffold_frac` is n-dependent and must be compared at matched n.

**Cost of diversity is C**: 0.947 -> 0.871 -> 0.781 as kl goes 0.3 -> 1.0 -> 3.0.
kl=3.0 buys a little more scaffold (0.906) for much less C (0.781), so 1.0 is the
knee.

## 6. The k3 KL estimator produces extreme outliers here

Left unchanged -- the loss still uses unclipped k3, which keeps the committed
GRPO runs comparable. `kl_clipped` (d clamped to +/-10 nats before the exp) is
logged beside it instead of replacing it.

It earned itself immediately: sweep cell kl=1.0/ent=0.02 recorded `kl=2940.5`
with `kl_clipped=93.3`, so **~97% of the raw mean was a single tail event** -- and
that cell still produced the best-in-column diversity (0.891). Without the
companion statistic that run reads as broken. Similarly `rs0_ws1` at kl=0.3 hit
`kl=17327.56` at step 7 with the next step at 0.0145.

The estimator is `exp(d) - d - 1`; one action where the policy disagrees with
`pi_ref` by ~10 nats contributes exp(10) ~ 22026, and a 6051-way space over ~5
decisions both permits that and averages it over 5 positions rather than 38.
**The outlier is a symptom, not a bug**: it means the policy reached an action the
reference assigns ~zero probability, which is expected when the action space was
rebuilt under it. Gradient clipping at 1.0 bounds the damage.

## 7. Duplicates shrink the effective group

`n_eff_group` and `eff_group_frac` are now logged: distinct SMILES among the
*rankable* rows, which is the set `group_advantages` actually standardizes over.
`eff_group_frac` tracks `unique_smiles_frac` to within 0.002 throughout, so at
kl=0.3 a nominal group of 64 carried ~48 distinct candidates and at kl=0.1 only
~33. Every advantage estimate in the kl=0.3 matrix was computed from a smaller
effective group than `group_size` claims. At kl=1.0 it is 0.987, so the problem
does not arise at the tuned setting.

## 8. What these runs cannot tell you

- **One GRPO seed.** All 24 runs use `--seed 0`; rollout noise is unsampled.
- **`kl_coef=1.0` was selected from section 5's grid**, and one of the four
  paired cells (`rs0_ws0`) is the same configuration as the sweep cell that
  selected it. It reproduced bit-identically (0.8874 both ways, a useful
  determinism check), but it is not independent of the selection; the other
  three cells are new data.
- **SELFIES was not re-swept.** Its `kl_coef=0.3` comes from the repo's earlier
  sweep under a different reward and a different acceptance criterion. Strict
  symmetry would sweep both families under the current reward. If SELFIES also
  improves at kl=1.0, section 3's comparison moves.
- **The 2x2 cannot separate its two noise sources** -- one run per cell is zero
  residual df. At kl=0.3: reward-seed effect −0.072, warm-start effect −0.081,
  the same order as each other and as the between-cell spread.
- **No synthesizability claim.** Assembly from BRICS fragments gives a
  retrosynthetic plausibility floor by construction; nothing here tests it.
- **Nothing here is a synthesis candidate.** Diversity being retained is not
  evidence the molecules are good.

## 9. What stands

The case for BRICS was never classifier accuracy -- the hierarchical motif family
settled that separately (replacing atom detail with fragment structure
−0.1161 +/- 0.0107; adding it on top −0.0051 +/- 0.0047, i.e. nothing).

It was validity by construction and an action space that cannot emit atom spam.
Validity holds (+0.0052 against an arm that already had the size gate), the
action-space argument is structural, and the diversity objection that appeared to
count against it was an artifact of a transplanted coefficient. What remains true
is narrower than the original claim: validity is a ~95.4% untrained floor rather
than a guarantee, because aromaticity is not covered.
