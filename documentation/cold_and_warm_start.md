# Cold Start and Warm Start: What Won, and Why

Companion to `open_items.pdf`. That document formalized the three open items;
this one records which concrete option was chosen for each of the two
bootstrapping regimes, what it was measured against, and what is still open.

Everything below is measured on BBBP with the committed seed-0 classifier
ensemble (`gin` + `gat` + `gine`), group size 64 unless stated.

---

## The two winners

| regime | the question | winner | margin |
|---|---|---|---|
| **cold start** | how does `pi_theta` bootstrap when nothing is trained? | **MLE warm start on real SELFIES** (`src/generator.py --pretrain`) | only viable option; the alternatives are structurally excluded, not merely worse |
| **warm start** | how is the advantage computed once the loop is running? | **two-part (hurdle) advantage** (`grpo.group_advantages(valid=...)`) | 0% ranking-signal loss vs 51% for pooled standardization |

---

## 1. Cold start — bootstrapping the policy

### The problem

GRPO needs a reward signal that discriminates *within* a sampled group. From a
uniformly initialized policy, essentially every molecule fails RDKit
sanitization, so every reward is the same flat penalty, the group standard
deviation is zero, and `group_advantages` correctly returns no gradient at all.
Training does not start slowly — it does not start.

### Options considered

**(a) Uniform init + GRPO directly.** Rejected. Waits for validity to appear by
chance. This is the degenerate all-equal-rewards case the advantage estimator
is explicitly built to neutralize, so the estimator and the bootstrap strategy
would be working against each other.

**(b) ZIG-MC — zero-inflated gamma fitted by MCMC.** Rejected on structural
grounds, not performance. Two independent reasons:

1. *MCMC produces samples, not a policy.* The GRPO update is built on the ratio
   `r(theta) = pi_theta(a) / pi_theta_old(a)`, which requires a parameterized
   policy with differentiable log-probabilities. An MCMC sampler over molecule
   space has no `theta` to differentiate, so there is no ratio and no gradient.
   It cannot be connected to `src/grpo.py` at all. MCMC over molecular graphs
   (MARS-style add/delete/mutate proposals against a target proportional to
   `exp(R(m)/T)`) is a legitimate *replacement* for GRPO, but it is a different
   algorithm, not a warm start for this one.
2. *The Gamma component does no work.* Measured on valid molecules,
   `R | valid` has mean 1.241 and sd 0.145, i.e. a coefficient of variation of
   0.117. That implies a Gamma shape parameter of about 73, at which a Gamma is
   indistinguishable from a Gaussian (skew ~0.23) — which is exactly what
   standardization already assumes. Fitting one by MCMC would spend a sampler
   to recover a mean and a variance.

**(c) MLE warm start on real SELFIES. — WINNER.** Teacher-forced cross-entropy
on the BBBP molecules, padding excluded from the loss.

### Measured result

```
bbbp: 2039/2050 encodable, |V|=77
  epoch  1   nll=3.0041   valid=87.5%
  epoch 10   nll=1.4276   valid=100.0%
  epoch 30   nll=0.9614   valid=99.0%
```

~53 s on CPU to reach 97–100% sanitization validity, which is where GRPO has a
gradient to work with. Nothing else was close enough to justify evaluating
further.

### Note on the other meaning of "cold start"

For a *fresh clone*, cold start is two commands and needs no classifier
training — the seed-0 checkpoints the reward depends on are committed under
`results/`:

```bash
python -m src.generator --pretrain     # ~53 s
python -m src.train_gan --steps 200    # ~0.7 s/step at group 32
```

---

## 2. Warm start — the advantage estimator

### The problem

The reward from `reward.assemble` is **zero-inflated by construction**: invalid
molecules collapse to a flat penalty, valid ones take a positive continuous
value. These are two populations, not one sample, and pooled standardization
lets a handful of zeros set the scale the valid molecules are ranked on.

An invalid molecule sits **~8.5 sd below the valid cluster**. That single
outlier inflates the group standard deviation and drags the mean, compressing
the valid molecules toward a uniform "you sanitized" push with genuine ranking
left as the remainder.

### Options compared

Spread = ranking signal among valid molecules. The property wanted is that it
stays **flat** regardless of how many zeros happened to land in the group.

| scheme | 0 invalid | 1 | 2 | 3 | signal lost | penalty delivered | corr(adv, C) |
|---|---|---|---|---|---|---|---|
| pooled standardization (previous) | 1.00 | 0.70 | 0.60 | 0.49 | **51%** | −4.72 | 0.431 |
| **two-part / hurdle — WINNER** | 1.00 | 1.00 | 1.00 | 1.00 | **0%** | −2.00 (chosen) | 0.431 |
| median / MAD | 0.99 | 0.99 | 0.99 | 0.91 | 9% | −7.92 | 0.431 |
| rank-based | 1.00 | 0.98 | 0.97 | 0.95 | 5% | −1.68 | 0.391 |

**One invalid molecule in 64 costs ~31% of the ranking signal; three cost
~51%.**

### Why hurdle over the two no-knob alternatives

Median/MAD and rank-based are close on stability and introduce no
hyperparameter, which made them genuinely tempting. The penalty column decided
it:

- **median/MAD** auto-derives a penalty of −7.92 — roughly an 8x gradient
  weight on invalid molecules that nobody chose and that is hard to reason
  about or tune.
- **rank-based** discards magnitude information; `corr(adv, C)` falls from
  0.431 to 0.391, so it is measurably worse at tracking the permeability term
  the whole project is optimizing.
- **hurdle** introduces `floor` as an explicit hyperparameter. This is the
  point rather than a cost: under pooled standardization the effective penalty
  for invalidity was *already* a hidden knob, set by whatever else happened to
  land in the group, which is not a decision anyone made. `floor` pairs with
  the `invalid_reward` argument `reward.assemble` already exposes — that one
  sets the reward, `floor` sets the advantage it becomes.

### Two corrections worth recording

Both of these were initial hypotheses that measurement overturned, and both are
easy to get wrong again:

1. **The zero spike does not destroy rank ordering.** Standardization is an
   affine map, so `corr(adv, C)` among valid molecules is *identical* under
   pooled and hurdle (0.431 both). The damage is to **scale**, not order. Any
   diagnostic built on correlation will show nothing here.
2. **The "G" in ZIG is inert.** See the cv ~0.117 / shape ~73 argument above.
   The zero-inflation is real and worth modelling; the Gamma is not.

---

## 3. Scope and caveats

- **The warm-start gain is a recovery, not a rescue.** At 97–100% validity the
  typical group holds ~1 invalid molecule, so the practical effect is ~30%
  signal recovery. Pooled standardization already trained successfully
  (C: 0.81 → 0.96). The hurdle form matters most if the warm start is shortened,
  the group is widened, or the schedule anneals harder toward permeability and
  validity slips.
- **`results/gan/` is a 40-step smoke run, not a trained generator.** It exists
  to demonstrate the loop closes end to end. Do not cite its numbers.
- **Reward hacking is visible and unresolved.** Samples drift toward aromatic,
  lipophilic ring systems — the expected direction for a BBB objective. The
  QED/SA terms and the reference-policy KL are meant to hold this back; 40 steps
  is not enough to establish that they do. This is the most important open
  question, and it is not addressed by anything above.
- **`floor = -2.0` is a default, not a tuned value.** It has not been swept.

## 4. Reproducing the measurements

The comparison table came from sampling 60 groups of 64 from the warm-started
generator, scoring them through the real reward, and computing each estimator's
spread among valid molecules bucketed by invalid count. The pipeline pieces are
all in `src/`; the probe scripts were throwaway and are not committed.
