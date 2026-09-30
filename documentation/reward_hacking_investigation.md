# Reward Hacking in the GRPO Loop: Instrumentation, Baseline, and What the Reward Is Actually Doing

Companion to `open_items.pdf` and `cold_and_warm_start.md`. Those two formalize
the math and record the bootstrapping choices. This one records the
investigation into the open question `cold_and_warm_start.md` flagged and did
not resolve:

> **Reward hacking is visible and unresolved.** Samples drift toward aromatic,
> lipophilic ring systems [...] 40 steps is not enough to establish that they
> do. This is the most important open question.

It is now established. This document says how, what was measured, and which of
the initial hypotheses measurement overturned.

Everything below is BBBP, seed-0 frozen classifier ensemble (`gin` + `gat` +
`gine`), group size 64, unless stated.

---

## Summary

| question | answer | evidence |
|---|---|---|
| does the generator hack the reward? | **yes, totally** | scaffold diversity 0.80 → 0.07–0.14 in 200 steps, all 3 seeds |
| would the old logging have caught it? | **no** | validity stays at 100%; unique-SMILES fires 88–154 steps late |
| is it all GRPO's fault? | **no** | the MLE warm start is already 0.72–0.87× narrower than BBBP |
| should the reward use the better classifier? | **no evidence for it** | paired delta on labeled OOD: −0.0001 and +0.0083, neither significant |
| is the reward still informative at step 200? | **no** | `sd C` collapses 0.173 → 0.006; 100% of molecules score > 0.95 |

The last row is the one that changes what to do next, and it was not on the
original list of questions.

---

## 1. The metric that hid the problem

The loop logged `unique` — the fraction of distinct SMILES strings in a group.
On the committed 40-step run that read 195/199 distinct, which was reported as
evidence of no mode collapse. It was not. Distinct strings are not distinct
chemistry: minor decorations of one scaffold produce 195 unique strings and one
molecular family.

`src/diversity.py` replaces it with structural measures. Its self-check pins the
distinction with two six-molecule sets:

| | unique SMILES | scaffolds | Tanimoto dist |
|---|---|---|---|
| six decorations of one benzene | 1.00 | 1 | 0.444 |
| six unrelated molecules | 1.00 | 5 | 0.955 |

Identical on the old metric, cleanly separated by the new ones.

### Metrics now logged every step

- **Murcko scaffold count / fraction** — distinct cores per group
- **mean pairwise Tanimoto distance** — Morgan r=2, 2048-bit; internal diversity
- **per-descriptor drift** — (generated mean − reference mean) / reference sd
- **per-descriptor spread ratio** — generated sd / reference sd, for TPSA, HBA,
  logP, MW
- validity, and `unique` retained deliberately as the contrast case

Reference statistics are computed once on the real data and held fixed.
Measuring drift against a rolling window of the generator's own output would
normalize away the trend being looked for.

### Sample size is not optional here

Scaffold fraction depends strongly on n — more molecules means more repeated
scaffolds, so the distinct fraction falls. BBBP reads **0.758 at n=199** and
**0.595 at n=982**. That swing is larger than any effect being measured, so
every structural comparison goes through `diversity.reference_band`, which
subsamples the reference to matched n over repeated draws. Comparing a
199-molecule sample against the full 2039-molecule reference flatters the
generator; comparing two generated sets of different sizes says nothing at all.

---

## 2. The baseline: 200 steps × 3 seeds

`results/gan/seed{0,1,2}/`. The reward climbs while chemistry is destroyed.

| | first 10 steps | last 10 steps |
|---|---|---|
| scaffold_frac | 0.72 – 0.80 | **0.07 – 0.14** |
| tanimoto_dist | 0.90 – 0.91 | **0.25 – 0.45** |
| tpsa_spread_ratio | 0.41 – 0.48 | 0.12 – 0.18 |
| hba_spread_ratio | 0.46 – 0.51 | 0.11 – 0.17 |
| mw_spread_ratio | 0.54 – 0.58 | 0.14 – 0.22 |
| c_mean (reward) | 0.87 – 0.88 | **0.98 – 0.99** |
| valid_frac | 0.98 – 0.99 | **1.00** |

**Validity never degrades.** It rises. Every metric the loop tracked before this
work says the run is a clean success.

### Lead time

Step at which each metric first falls 20% below its own opening level
(9-step smoothed):

| metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| descriptor spread (TPSA) | 8 | 8 | 8 |
| scaffold fraction | 24 | 15 | 12 |
| Tanimoto distance | 81 | 159 | 136 |
| **unique SMILES** | **112** | **169** | **144** |
| validity | never | never | never |

**Scaffold fraction leads unique-SMILES by 88–154 steps.** At step 40, where the
previously committed run stopped, scaffold diversity had already fallen and
unique SMILES still read 0.98. The old metric is not merely weaker; on the
timescale anyone would actually train at, it is silent.

### Seeds are not interchangeable

Seed 0 collapses structurally much earlier and harder than seeds 1 and 2
(scaffold fraction 0.17 at step 90 against 0.75 and 0.73). A single run would
have misstated the timing in either direction. Multi-seed is not a formality
here.

---

## 3. Attribution: the warm start is already narrow

Not all of the narrowing is reward hacking. Matched-n (n=199), descriptor spread
as a ratio to BBBP:

| descriptor | BBBP | warm start | + GRPO | GRPO's share |
|---|---|---|---|---|
| tpsa | 1.00 | 0.72 | 0.32 | 58% |
| hba | 1.00 | 0.79 | 0.29 | 70% |
| logp | 1.00 | 0.87 | 0.57 | 71% |
| mw | 1.00 | 0.79 | 0.35 | 68% |

The MLE warm start under-disperses before GRPO runs at all — an imperfect
density model of BBBP, which is what an MLE-trained sequence model on 2039
molecules is.

**This bounds what a KL fix can achieve.** The KL term pulls toward the
reference policy, and the reference policy *is* the warm start. Pulling all the
way back to it still leaves a distribution 0.72–0.87× narrower than the real
data. A KL sweep can recover at most the 58–71% GRPO contributes.

Structurally the warm start matches BBBP — scaffold fraction 0.750 ± 0.026 vs
0.765 ± 0.021, Tanimoto 0.917 vs 0.897 — so **all** of the structural collapse
is GRPO's doing.

---

## 4. Is the reward's classifier the right one?

The reward uses `gin`+`gat`+`gine` (~0.93–0.94 test ROC-AUC) while the benchmark
leader is `hybrid/gat` at 0.9608. The tempting conclusion is that the reward
uses the wrong model. That conclusion is not supported, and the framing was
wrong: test ROC-AUC measures accuracy on molecules that look like training data,
and the generator asks about molecules that by construction do not.

Two experiments, because only one of them has labels.

### Labeled OOD — the external holdouts

The leak scrub removed every molecule and every Murcko scaffold group shared
with training, which is exactly what makes the survivors an out-of-distribution
test set *with ground truth*. Paired bootstrap over molecules, `delta` =
candidate − current:

| source | n | n_pos | current AUC | candidate AUC | delta [95% CI] | sig |
|---|---|---|---|---|---|---|
| adenot | 59 | 15 | 0.997 | 0.997 | −0.0001 [−0.012, +0.012] | no |
| wang | 55 | 38 | 0.808 | 0.816 | +0.0083 [−0.051, +0.067] | no |

**No evidence for swapping the classifier.** Both deltas are indistinguishable
from zero.

Two caveats that matter more than the headline:

1. **The two holdouts disagree violently.** 0.997 on Adenot against 0.808 on
   Wang, on datasets constructed by the same scrub. A near-perfect AUC on 59
   molecules should be treated as a red flag about that holdout, not as evidence
   of a good model. Neither number should be quoted as a generalization result.
2. Marginal CIs are wide (Wang: 0.675–0.916). The paired delta is the only
   comparison this sample size supports.

### Unlabeled OOD — disagreement on generated molecules

No ground truth exists, so nothing here measures correctness. Member
disagreement measures *instability*.

| set | n | mean P | member sd | frac in tails (>0.95 or <0.05) |
|---|---|---|---|---|
| real BBBP | 2039 | 0.745 | 0.063 | 34% |
| generated | 200 | **0.991** | **0.005** | **100%** |

**This is the opposite of the predicted result.** The hypothesis was that
ensemble members would diverge off-distribution, making the reward noisy. They
converge — 12× *less* disagreement on invented molecules than on real ones, and
every single generated molecule lands in the saturated tail.

Low disagreement here is not reassurance. Three models trained on the same data
with the same featurization can be confidently wrong together; agreement between
them measures shared inductive bias, not correctness.

---

## 5. The finding that changes the plan: the reward is saturated

GRPO's advantage (Eq. 6) is group-**relative**. Only differences in reward
*within* a group move the policy. A term that assigns every molecule in a group
the same value contributes nothing, however high that value is.

Within-group statistics of the permeability term over training (seed 0):

| step | mean C | **sd C** | min | max | frac > 0.95 |
|---|---|---|---|---|---|
| 0 | 0.830 | **0.173** | 0.239 | 0.999 | 33% |
| 25 | 0.928 | 0.067 | 0.736 | 0.998 | 47% |
| 50 | 0.959 | 0.055 | 0.679 | 0.998 | 71% |
| 100 | 0.964 | 0.070 | 0.484 | 0.999 | 88% |
| 150 | 0.989 | **0.004** | 0.974 | 0.997 | 100% |
| 199 | 0.990 | **0.006** | 0.959 | 0.994 | 100% |

**The spread collapses by a factor of ~30.** By step 150 the classifier says
"definitely permeable" to everything the generator produces, and the
permeability term has stopped ranking anything.

Consequences:

- **The reward curve is misleading.** `c_mean` rising from 0.88 to 0.99 looks
  like optimization succeeding. It is the term going blind.
- **Late training is not optimizing permeability at all.** After ~step 150 the
  policy is driven by the remaining terms (QED, SA, the discriminator) plus
  whatever noise survives in C — while the property and scaffold collapse
  continues.
- **This is not fixed by a better classifier.** A more accurate in-distribution
  model saturates the same way once the generator finds its high-confidence
  region. Section 4 found no accuracy difference off-distribution; this section
  suggests accuracy is not the binding constraint anyway.

### Why this reframes the intervention

The planned success criterion for a KL sweep was "property spread retained AND
classifier reward above the warm-start baseline." The second half is now
unusable as stated: reward is high precisely when the term is saturated, so
"reward above baseline" is satisfied by the failure mode.

A usable criterion has to include **`sd C` staying non-trivial** — evidence that
the permeability term can still tell candidates apart — not just `mean C` being
large.

---

## 6. Corrections to earlier claims in this repo and in review

Recorded because each was a plausible hypothesis that measurement overturned,
and each is easy to get wrong again.

1. **"195/199 distinct samples, so no mode collapse."** Wrong metric. String
   uniqueness is blind to scaffold and property collapse for the first ~100
   steps.
2. **"The collapse is property-space, with only mild scaffold narrowing."** True
   at step 40, and I generalized it too far. At 200 steps scaffold collapse is
   the *more* dramatic effect (0.80 → 0.07).
3. **"The reward uses the wrong classifier."** Overclaimed. It uses a model that
   is not the benchmark leader; whether that matters is an empirical question,
   and the answer so far is no.
4. **"Off-distribution the ensemble will disagree, making the reward noisy."**
   Backwards. It agrees more, and saturates.
5. **"Architecture differences are hopeless at 3 seeds — noise is 3× signal."**
   That used marginal SDs. The design is paired (all models share a split within
   a seed), and the paired SD is a median 0.40× the marginal one on BBBP.
   `gat − gin` is a 1.4 sd effect, not a hopeless one. Still not significant at
   n=3; see §7.

---

## 7. The classifier comparison is paired

`src/paired_analysis.py`. The seed controls the split and every model trains on
the same seed-s split, so the architecture comparison is a paired design.
Verified rather than assumed: all 12 models match on both `test_n` and
`test_n_positive` within every (dataset, seed).

BBBP base GNNs, `test_roc_auc`:

| pair | mean diff | paired sd | indep sd | ratio | dz | sig |
|---|---|---|---|---|---|---|
| gat − gin | +0.0127 | 0.0091 | 0.0581 | 0.16 | +1.40 | no |
| gcn − sage | +0.0064 | 0.0108 | 0.0290 | 0.37 | +0.60 | no |
| gcn − gin | +0.0126 | 0.0246 | 0.0395 | 0.62 | +0.51 | no |
| gat − gcn | +0.0001 | 0.0327 | 0.0478 | 0.68 | +0.00 | no |

Median paired/independent sd = **0.40 on BBBP**, 0.77 on B3DB. **0/6 pairs
separate at 95% with n=3.**

Two cautions built into the script:

- At n=3 the paired sd has **2 df**, so `dz` and the implied
  seeds-to-significance are themselves noisy. Read them as a ranking of
  candidates for a 10-seed run, not as results.
- Statistical separation is not practical relevance. On B3DB,
  `hybrid:gcn − hybrid:gin` shows dz = −14.9 from three remarkably consistent
  differences (−0.00156, −0.00170, −0.00149) — which amount to **0.16% AUC**.
  The script flags magnitudes below 0.005 as `negligible`.

---

## 8. Scope and caveats

- **`floor = −2.0` remains untuned.** Not swept.
- **The external holdouts remain exploratory.** n=59 and n=55 after a 96% scrub.
  They support a paired model-vs-model comparison and nothing stronger. The
  Adenot/Wang disagreement (0.997 vs 0.808) is unexplained and should be
  understood before either is cited.
- **Nothing here tests B3DB.** All generation results are BBBP.
- **The discriminator's contribution is not separated.** `d_var` is logged but
  the reward's terms have not been ablated individually, so "the collapse is
  driven by C" is an inference from the saturation data, not a measured
  decomposition.
- **No intervention has been run yet.** `kl_coef`, the reward weights, and the
  classifier are all untouched from the baseline configuration.

---

## 9. Reproducing

```bash
python -m src.diversity                 # self-check: the metric distinction
python -m src.generator --pretrain      # ~53 s, warm start
python -m src.train_gan --steps 200 --seed 0   # ~11 min at group 64
python -m src.analyze_drift results/gan/seed0 results/gan/seed1 results/gan/seed2
python -m src.paired_analysis --all-sources
python -m src.classifier_ood
```

Per-step sample dumps land in `results/gan/seed<n>/samples/stepNNNN.json`, so
any metric not thought of yet can be computed offline without retraining.
