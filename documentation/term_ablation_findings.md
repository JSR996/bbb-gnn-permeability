# Term Ablation: the synthesizability term drives the collapse, not the classifier

Result of `src/term_ablation.py` — 4 terms × 3 seeds, 200 steps, group 64,
each arm paired against a same-seed baseline. Companion to
`reward_hacking_investigation.md`, whose Section 8 inferred the driver from
saturation data without ablating.

**It revises that inference.** The reward is
`R = w_D·D + w_C·C + w_Q·QED + w_S·SA`, and the term that drives the scaffold
collapse is **SA**, not **C**.

---

## 1. Headline

Endpoint = mean of the last 20 steps. `Δ` = (term dropped) − (baseline), same seed.

| dropped | Δ scaffold_frac | seeds agreeing | still collapsed | mean_C |
|---|---|---|---|---|
| **S** (synthesizability) | **+0.426** | **3/3** | **0/3** | 0.972 |
| C (permeability) | +0.080 | 3/3 | 3/3 | 0.786 |
| Q (QED) | +0.021 | 2/3 | 3/3 | 0.984 |
| D (discriminator) | −0.016 | 2/3 | 3/3 | 0.982 |

Per seed, scaffold_frac at the endpoint:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline | 0.055 | 0.100 | 0.192 |
| drop S | **0.365** | **0.821** | **0.441** |
| drop C | 0.174 | 0.202 | 0.212 |
| drop Q | 0.128 | 0.116 | 0.168 |
| drop D | 0.059 | 0.119 | 0.120 |

Dropping SA is the only arm where **no seed collapses** (screen:
endpoint scaffold_frac < 0.30; BBBP at matched n is ~0.6–0.76). Tanimoto
distance roughly triples (0.22–0.31 → 0.67–0.84) and TPSA spread ratio
roughly doubles.

**Reading:** SA rewards *ease of synthesis*, which in practice means small,
common, simple fragments — simple aromatic rings. The term included as a
realism guard is the main thing crushing structural diversity. Dropping C
helps consistently (3/3) but is ~5× smaller and leaves every seed collapsed,
so C is a contributor, not the driver.

## 2. Permeability is largely a side effect

With `w_C = 0` — C still scored and logged, just not steering — `mean_C` falls
only from ~0.98 to **0.786**. BBBP's own reference raw-C mean is 0.745
(`reward.LOGIT_CALIB`). The generator keeps finding molecules the frozen
classifier scores above the real data *without being asked to*.

This is the test the module docstring set up in advance: the classifier's
confidence is substantially a by-product of what Q, S and D select for, not
something the policy has to chase.

## 3. C's saturation is partly self-inflicted

`sd_C` within a group, at the endpoint:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline | 0.005 | 0.008 | 0.011 |
| drop C | 0.082 | 0.057 | 0.084 |

Optimizing C destroys roughly 8× of C's own between-molecule spread — which is
exactly the quantity GRPO consumes, since the advantage is group-relative.
This is consistent with the MC-dropout finding that the saturation is a
property of the probability scale rather than genuine certainty, and it is an
independent argument for the logit transform in `reward.transform_c`.

## 4. Caveats, in order of weight

- **Zeroing is not a marginal contribution.** `group_advantages` standardizes
  the total reward inside the valid subgroup, so removing a high-variance term
  inflates the effective weight of everything that remains. Read a row as
  "what happens when this term stops steering", not "this term's share".
- **The weights are not equal, and not constant.** `w_S` and `w_Q` are fixed at
  0.5 while `w_C` anneals 0.2 → 1.5 and `w_D` 1.0 → 0.3. S dominating despite
  carrying under a third of C's endpoint weight makes the result stronger, not
  weaker — but it does mean the arms are not weight-matched.
- **n = 3.** Sign agreement is the readout; the means are descriptive. Seed 0
  resists rescue hardest (0.365 vs seed 1's 0.821).
- **`collapse_below = 0.30` is an arbitrary screen**, not a calibrated threshold.
- **Dropping S has not been shown to be safe**, only to restore diversity.
  Nothing here checks whether the resulting molecules are synthesizable; that is
  the obvious next measurement.

## 5. Reproducibility: thread count is part of the configuration

This loop is chaotic and its reproducibility is governed by **OpenMP thread
count**, because OpenMP changes the order of float reductions. Measured on
seed 0 against a 6-thread baseline:

| `OMP_NUM_THREADS` | max abs diff in `reward_mean` over 20 steps |
|---|---|
| 6 (matching) | **2.22e-16** — exact |
| 36 | 2.89e-03 — diverges |

A mismatch starts at ~1e-09 at step 1 and reaches ~1e-02 by step 20 — larger
than the effects this ablation measures. Two consequences:

1. **Every run in a comparison must use the same thread count.** With
   `--jobs J`, `term_ablation` gives each arm `cpu_count // J` threads, so the
   baselines must be generated to match. The results above used
   `--jobs 6` on 36 cores = 6 threads, with baselines written at
   `OMP_NUM_THREADS=6`.
2. `train_gan` now records `torch_num_threads` / `omp_num_threads` /
   `mkl_num_threads` in `config.json`, and `--check-baseline` warns on a
   mismatch instead of reporting an unexplained divergence. `torch.get_num_threads()`
   is recorded alongside the env vars because torch clamps to physical cores —
   `OMP_NUM_THREADS=36` binds 18 on this machine.

**The baselines in `results/gan/seed{0,1,2}` were regenerated locally** at 6
threads for this reason; the previously committed ones were produced elsewhere
and do not reproduce here at any thread count available on this machine. The
earlier versions remain in git history.

## 6. What this suggests next

- Ablate **S** against a *synthesizability-aware* check: does dropping it buy
  diversity at the cost of unmakeable molecules, or was the term simply
  mis-weighted? A weight sweep on `w_S` is cheaper than an on/off ablation and
  would answer this directly.
- The guard work on `claude/quirky-goodall-4qkepg` (size gate, geometric
  aggregation, alert penalty) was designed against the hypothesis that C drives
  the collapse. Those arms are worth re-reading against this result.
