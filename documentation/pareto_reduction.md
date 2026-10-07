# Pareto reduction vs linear scalarization

Measured, four seeds, paired on both the GRPO seed and the per-seed warm
start. Reproduce with `python -m src.aggregate_arms`; the derivation and a
runnable concave-fold demo are in `src/pareto.py`.

## Why this was tried

The reward `R = w_D*D + w_C*C + w_Q*QED + w_S*SA` is a Minkowski-weighted
linear scalarization of a 4-objective vector, so Theorem 1 of Das & Dennis
(1997) applies verbatim: **no choice of weights reaches a Pareto-optimal
candidate inside a concave fold of the frontier.** Measured on this project's
own sampled groups, a linear reduction reaches **0.48** of the per-group
Pareto front over 2000 random weightings. Every weight experiment in this
repo — the 0.2→1.5 anneal, the term ablation, the `w_S` sweep — searched a
space that provably cannot reach the other half.

The objectives are also nearly uncorrelated (mean pairwise **+0.009**; D vs C
is **−0.258**, realism and permeability actively oppose), which is the regime
where a rank-based reduction has the most room.

`nsga2` ranks by non-dominated sorting plus crowding distance and never
collapses the partial order into a scalar before ranking, so Theorem 1 does
not bite. `composite` keeps that ordering but borrows the linear reward's
scale, as a hedge against a rank reduction starving the advantage of variance.

## What it bought

| metric | linear | nsga2 | seeds | composite | seeds |
|---|---|---|---|---|---|
| tpsa_spread_ratio | 0.410 | **0.524** | 4/4 | **0.519** | 4/4 |
| logp_spread_ratio | 0.683 | **0.784** | 4/4 | **0.785** | 4/4 |
| mw_spread_ratio | 0.481 | **0.585** | 4/4 | **0.602** | 4/4 |
| tanimoto_dist | 0.887 | **0.899** | 4/4 | **0.899** | 4/4 |
| kl to pi_ref | 0.066 | **0.047** | 4/4 | **0.048** | 4/4 |
| c_mean | 0.925 | 0.891 | 0/4 | 0.894 | 0/4 |
| scaffold_frac | 0.811 | 0.780 | 1/4 | 0.785 | 1/4 |
| valid_frac | 0.963 | 0.953 | 1/4 | 0.954 | 2/4 |

("seeds" = seeds improved vs linear, out of 4.)

**Property-distribution coverage is the win, and it is large and unanimous.**
TPSA, logP and MW spread each recover ~0.10–0.12 of the real distribution's
spread on every seed — about a quarter of the remaining gap on TPSA. Property
drift was the item the containment work (`kl_coef` 0.3, the size gate) made
*worse* on TPSA and logP; this is the first intervention that moves it the
right way. The policy also sits closer to `pi_ref` on all four seeds.

The price is 0.034 of classifier confidence. Given that `c_mean` is the term
with the documented saturation and hacking problem, and that real BBBP+ drugs
score **0.868** on the same ensemble — *below* all three arms — paying
confidence for coverage is the right direction of trade.

## What it did not buy, which matters more

| arm | D_real | C(BBBP) | C(B3DB) | gap | vs linear |
|---|---|---|---|---|---|
| linear | 0.765 | 0.924 | 0.888 | +0.036 | — |
| nsga2 | **0.775** | 0.891 | 0.849 | +0.042 | worse on 3/4 |
| composite | 0.773 | 0.894 | 0.848 | +0.046 | worse on **4/4** |
| **real drugs** | **0.940** | 0.868 | 0.898 | **−0.029** | — |

1. **It does not reduce Goodharting.** The instrument gap — training ensemble
   minus the never-seen B3DB judge — gets slightly *worse*, by +0.006 on 3/4
   seeds for `nsga2` and +0.010 on 4/4 for `composite`. Reaching the concave
   folds of the frontier is a different problem from reaching them honestly,
   and this result separates the two cleanly: the arms found more of the
   frontier and mined the selector's blind spots marginally harder while doing
   it. **Theorem 1's headroom buys coverage, not honesty.**

2. **It does not close the realism gap.** `D_real` improves on 4/4 seeds, but
   by +0.009, from 0.765 to 0.775 against real drugs at 0.940. The 0.165 that
   remains is untouched, as it has been in every configuration tried.

3. **It does not fix scaffold collapse** (−0.031, 1/4). This looked like the
   most likely win going in, because crowding distance rewards candidates in
   sparse regions and that is a direct push against collapse. It fails for a
   specific reason: crowding distance acts in **objective** space, and
   scaffold identity is not one of the four objectives. Molecules can spread
   out across D/C/QED/SA while sharing a scaffold. Getting scaffold diversity
   from this machinery means making it an objective, not hoping it correlates.

## Recommendation

**`nsga2`, not `composite`.** They are within noise of each other on
everything the table measures, but `composite` is worse on the instrument gap
on 4/4 seeds against `nsga2`'s 3/4, and `nsga2` needs no weights at all —
which removes the weight-tuning surface that Theorem 1 says was never
reaching half the frontier anyway. The variance-starvation the `composite`
hedge was built against did not materialize here (`adv_std` is ~1.06 in both).

**Not promoted to the default yet.** The gap regression is small but it is in
the wrong direction on the one measurement this project trusts least, and
`c_mean` −0.034 is a real cost. The honest summary is **a better
diversity/coverage operating point at equal-or-slightly-worse honesty**,
which is worth having as a documented arm and worth re-testing the moment
there is a stronger off-instrument judge than B3DB (n=55 after the leak
scrub, and it aggregates BBBP).
