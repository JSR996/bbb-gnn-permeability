# Known Limitations

Everything currently wrong, unproven, or unresolved in this project, as of the
reward-hacking investigation. Companion to `reward_hacking_investigation.md`,
which has the evidence behind most of these.

Each item is marked:

- **[measured]** — demonstrated with numbers in this repo
- **[mitigated]** — a failure mode now contained, with the residue named
- **[suspected]** — plausible, not yet tested
- **[untested]** — a claim the project relies on that nobody has checked

Nothing here is marked "resolved". Two failure modes are contained and one
reporting flaw is fixed; the underlying objective is unchanged.

The generation half is a research prototype. **Nothing it emits is a synthesis
candidate**, and several items below explain why that is not conservatism.

---

## 1. The generator

### 1.1 Reward hacking is total, not partial **[mitigated]**

*Contained by the hard size gate (`--min-heavy-atoms 10`), which routes
undersized molecules through the invalidity hurdle. Across 3 seeds it lifts
scaffold diversity to 0.828 ± 0.004 — above the warm start's 0.736 and far
more stable than the ungated 0.769 ± 0.027 — for 0.004 of mean C, and cuts
molecules under 10 heavy atoms from 10.4% to 2.1%.*

*These are now the **defaults** (`kl_coef=0.3`, `min_heavy_atoms=10`), because
leaving them opt-in meant the documented command reproduced the hack. At
200 steps, seed 0: scaffold 0.055 → 0.838, Tanimoto 0.292 → 0.890, distinct
pooled molecules 190 → 1267, for mean C 0.990 → 0.920.*

**What remains.** The gate blocks one manifestation, not the mechanism. The
generator still optimizes a proxy it can outrun: the gate arm trips BRENK on
43.8% of what it makes against 35.0% for real BBB+ drugs.

**Structural diversity is restored; the property-space hack is not.** Measured
on the new defaults against BBBP, the *direction* of drift is unchanged and on
two descriptors slightly worse — TPSA −0.41 → −0.57 sd, logP +0.22 → +0.39 sd
— even as spread recovers (TPSA 0.30× → 0.46×, HBA 0.26× → 0.45×, logP 0.43×
→ 0.76×). The generator now walks the same low-polarity, lipophilic direction
across a structurally diverse set instead of a collapsed one. That is a better
failure, not an absent one, and it is exactly what 1.3 and 1.4 predict.

Nothing has been run past 200 steps under the gate, and the baseline collapse
took until step ~150 to complete. The original measurement follows.

200 steps × 3 seeds: scaffold fraction 0.80 → 0.07–0.14, Tanimoto distance
0.90 → 0.25–0.45, descriptor spread ~0.45 → ~0.12, while classifier reward
rises 0.88 → 0.99. Validity stays at 100% throughout, so every metric the loop
tracked before this work says the run succeeded.

### 1.2 It generates non-drugs that outscore real drugs **[measured]**
The 150-step logit arm converged on carbon tetrachloride, freon-12, freon-11
and tetrafluoromethane. At the late-training weights CCl₄ scores **2.280**
against caffeine's **2.063**. The kl=0.02 baseline failed differently — 41 of
64 copies of a single biphenyl amide.

### 1.2b Soft in-loop alert penalties do not generalize **[mitigated]**

*Contained by moving structural screening out of the loop entirely
(`src/postfilter.py`).* An `exp(-λN)` penalty at λ=0.3, weight 3 cut the
SMARTS set it was trained against (0.339 → 0.203) while leaving BRENK
(0.438 → 0.406) and BRENK+PAINS+NIH (0.464 → 0.427) inside seed noise. It
learned the checklist, not the chemistry — Goodhart one level up from the
reward hacking it was added to fix. See §10 of
`reward_hacking_investigation.md`.

**What remains.** Post-filtering is containment, not repair: the policy is
unchanged and still produces the same reactive molecules, they are simply
not shipped. Yield is 40% on the gate seeds. And the alert reduction that
motivated the term was itself never statistically resolved — paired
`alert − gate` is −0.250 alerts/mol with CI [−0.558, +0.058] at n=3, while
the C cost *was* resolved.

### 1.3 No intervention fixes it yet **[measured]**
- Raising `kl_coef` to 0.3 restores *structural* diversity (scaffold 1.06× the
  warm start) but barely touches property spread: TPSA 0.33 → 0.57 across a
  **50×** change in `kl_coef`.
- The logit transform restores reward scale but makes scaffold collapse
  **worse** (0.741 → 0.349), because more gradient on a hackable objective
  buys more hacking.
- The Tox21 term makes it worse still (§3.3).

### 1.4 KL cannot reach real-data diversity, by construction **[measured]**
The KL term pulls toward the reference policy, which *is* the MLE warm start,
and the warm start is already at 0.72–0.87× BBBP's descriptor spread before
any RL. No `kl_coef` can produce a distribution wider than its own reference.

### 1.5 Only one guard feature has ever separated the failures **[measured]**
Molecular size (heavy atoms: 5 for every solvent, 14–21 for the drugs, BBBP 5th
percentile 10). Three others were tested and failed: descriptor distance
(CCl₄ 1.087 = caffeine 1.087), applicability domain (CCl₄ 0.429, *above* the
baseline's own collapse molecule at 0.289), and substructure catalogues (BRENK
flags 35% of real BBB+ drugs, CHEMBL 64%).

### 1.6 The size floor now runs in training, and holds for 200 steps **[mitigated]**
Run as the `gate` arm (150 steps × 3 seeds) and now as a default
(`min_heavy_atoms=10`, 200 steps, seed 0). It holds: molecules under 10 heavy
atoms fall to 0.016–0.031 of output against 0.047–0.188 without it, and
scaffold diversity is unharmed (0.826–0.833 gated vs 0.741–0.795 raw).

It is **[mitigated]**, not resolved, for the reason given in 1.1: the gate
blocks one *manifestation* — the tiny-solvent failure — while the mechanism
that produces it is untouched. The gate arm still trips BRENK on 43.8% of its
output against 35.0% for real drugs, and nothing has run past 200 steps.

### 1.7 The reward's terms have now been ablated, and C was the wrong suspect **[measured]**
4 terms × 3 seeds, 200 steps, each arm paired against a same-seed baseline
(`term_ablation_findings.md`). Dropping **SA** recovers scaffold diversity by
**+0.426 on 3/3 seeds** and is the only arm where no seed collapses. Dropping
**C** gives +0.080 — also 3/3, but ~5× smaller, and all three seeds still
collapse. `w_D` and `w_Q` do not separate (−0.016 and +0.021, 2/3 each).

So "the collapse is driven by C" was wrong: SA is the driver. SA rewards ease
of synthesis, which in practice selects small common fragments, and the term
added as a realism guard is the main thing crushing structural diversity.

Two by-products. With `w_C = 0` the term is still scored, and `mean_C` falls
only 0.98 → 0.786 against BBBP's own 0.745 — the classifier's confidence is
largely a side effect of what Q/S/D select for, not something the policy
chases. And `sd_C` rises ~0.008 → ~0.074 when C is dropped: optimising C
destroys ~8× of C's own between-molecule spread, which is the quantity the
group-relative advantage consumes.

Caveat: zeroing is not a marginal contribution — `group_advantages`
standardises the total reward, so dropping a high-variance term inflates the
weight of what remains. And nothing here shows dropping SA is *safe*: no
synthesizability check was run on the resulting molecules. That is the next
measurement, and a `w_S` sweep is cheaper than an on/off ablation.

### 1.8 `floor = -2.0` is untuned **[measured]**
*Swept. −2.0 is at the knee: it is the weakest value that fully deters gate
failure, and stronger values change nothing measurable.*

`results/brics_floor/` — floor in {−1.0, −4.0} against the default −2.0, all at
the settled config (kl=1.0, reward seed 0, ws0, GRPO seed 0, 250 steps).

First, how often it binds at all: **2.71% of rows**, 1.7 of 64 per step. Almost
entirely the size gate (undersized 0.0267) rather than unsanitizability
(0.0004) — so this is the penalty for being too small, not for being invalid.

| floor | scaffold | unique | C | reward | adv_std | undersized |
|---|---|---|---|---|---|---|
| −4.0 | 0.881 | 0.976 | 0.864 | 2.116 | 1.100 | 0.0259 |
| **−2.0** | **0.887** | **0.986** | 0.871 | 2.130 | 1.022 | 0.0267 |
| −1.0 | 0.834 | 0.979 | 0.908 | 2.173 | 0.999 | 0.0347 |

Welch on the last 25 steps against the default, 8 tests, x8 threshold 0.0063:

| comparison | scaffold | C | undersized |
|---|---|---|---|
| −1.0 vs −2.0 | **−0.053** (p<1e-4) | **+0.037** (p<1e-4) | **+0.018** (p=0.0035) |
| −4.0 vs −2.0 | −0.006 (p=0.60) | −0.007 (p=0.34) | −0.001 (p=0.78) |

**Weakening the floor measurably hurts; strengthening it does nothing.** At −1.0
the penalty is too weak to deter gate failure: undersized molecules more than
double (0.0150 → 0.0331 on the last 25 steps) and the policy buys `C` (+0.037)
by drifting toward molecules the gate is meant to exclude — scaffold diversity
falls 0.053 with it. At −2.0 the behaviour is already fully deterred, so −4.0
adds nothing and only inflates `adv_std` (1.02 → 1.10), which is the hurdle
estimator's spread moving away from its design value of 1.0.

So the inherited default is correct and now has a reason: it is the knee of this
curve. One seed, one configuration.

### 1.9 Single-seed interventions **[measured, as a gap]**
The 3-seed discipline was applied to the baseline only. The KL sweep, the
raw-vs-logit A/B and the Tox21 probe are all **seed 0 alone**. Given that seed
0 collapsed markedly harder than seeds 1 and 2 in the baseline (scaffold 0.17
vs 0.75/0.73 at step 90), every intervention result should be treated as
provisional until replicated.

### 1.10 Runs are short, and no long run has been validated **[measured]**
*1000 steps at the settled config. No sustained decay. Validity holds at
0.9998. The settled point is stable over 5x the previous horizon.*

`results/brics_long/kl1_s0_1000` — kl=1.0, reward seed 0, warm start ws0, GRPO
seed 0, 1000 steps, same instrumentation. 100-step window means:

| window | scaffold | unique | C | reward | rdkit_valid |
|---|---|---|---|---|---|
| 0-99 | 0.835 | 0.985 | 0.839 | 1.702 | 0.9998 |
| 200-299 | 0.873 | 0.984 | 0.878 | 2.141 | 0.9997 |
| 400-499 | 0.859 | 0.982 | 0.894 | 2.171 | 0.9997 |
| 600-699 | 0.861 | 0.984 | 0.901 | 2.195 | 1.0000 |
| 800-899 | 0.852 | 0.979 | 0.904 | 2.197 | 1.0000 |
| 900-999 | 0.853 | 0.979 | 0.905 | 2.198 | 0.9998 |

**The apparent decay is a settling transient, not a trend.** Regressing
`scaffold_frac` on step from 250 onward gives p = 0.001 (−0.0196 over 750
steps), but that is driven by settling from 0.873 to ~0.855 between steps 250
and 400. Starting the window later, there is no detectable trend:

| window | slope/step | p |
|---|---|---|
| 250-999 | −2.61e-05 | 0.001 |
| 400-999 | −1.47e-05 | 0.179 |
| 500-999 | −1.48e-05 | 0.299 |
| 700-999 | −1.51e-05 | 0.623 |

**Stated as a bound rather than an absence**, because 600 steps of a statistic
with per-step sd ~0.05 can only rule out so much: any sustained decay is slower
than roughly **0.024 per 1000 steps**, the detection limit here. For scale, the
documented ungated collapse ran at −0.0035/step (0.80 → ~0.10 in 200 steps),
134x faster than even the transient slope; at the detection limit it would take
>30,000 steps to reach 0.10.

`unique_smiles_frac` shows no trend (p = 0.128 from step 250). `c_mean` rises
+0.0276 over 750 steps (p < 0.001) while scaffold is flat — the Goodhart
direction, at a rate that does not matter over any horizon anyone will run.

Caveats: **one seed, one configuration.** 6 tests were run on this trajectory,
and the two significant results survive x6 (scaffold transient 0.006, C rise
<0.001). Validity min 0.9844, mean 0.9998 across all 1000 steps.

### 1.11 Generation has only ever been run on BBBP **[measured]**
*B3DB generation works — validity and duplicate-freedom match BBBP. But the
diversity-RAISING effect is **BBBP-specific**, and that is established at
matched policy drift rather than at a matched coefficient.*

`results/brics_b3db/` — B3DB warm start (30 epochs, nll 1.5336, 100% valid),
then GRPO at five coefficients, 250 steps, reward seed 0, GRPO seed 0. B3DB's
action space is ~11.6k actions against BBBP's 6051 (4795 fragments vs 2484).

| arm | measured KL | scaffold | start | delta | p | unique | C | valid |
|---|---|---|---|---|---|---|---|---|
| **BBBP kl=1.0** | 0.482 | 0.887 | 0.839 | **+0.048** | 0.0007 | 0.986 | 0.871 | 0.9996 |
| B3DB kl=0.1 | 0.913 | 0.441 | 0.843 | **−0.402** | <1e-30 | 0.526 | 0.963 | 0.9998 |
| **B3DB kl=0.3** | **0.503** | 0.687 | 0.843 | **−0.156** | <1e-13 | 0.792 | 0.929 | 0.9998 |
| B3DB kl=1.0 | 0.097 | 0.831 | 0.856 | −0.025 | 0.096 | 0.983 | 0.843 | 0.9999 |
| B3DB kl=3.0 | 0.038 | 0.825 | 0.858 | −0.032 | 0.080 | 0.985 | 0.747 | 0.9998 |
| B3DB kl=10.0 | 0.011 | 0.853 | 0.850 | +0.003 | 0.849 | 0.984 | 0.682 | 0.9997 |

**The decisive row is B3DB kl=0.3**, where measured KL is 0.503 against BBBP's
0.482 — matched drift, 1.04x. At matched drift BBBP *gains* 0.048 of scaffold
diversity and B3DB *loses* 0.156. Opposite signs, both far past the x6
Bonferroni threshold of 0.0083. So the effect is a property of the BBBP
objective, not of the coefficient.

**Comparing at a matched coefficient would have given the wrong answer**, and
nearly did. At kl=1.0 B3DB simply shows no change (p = 0.096), which reads as a
weak null; the coefficient delivers 5x less drift there than it does on BBBP.
Matching by measured effect is what makes the comparison interpretable — the
same lesson as §4c on `kl_coef` not transplanting across action spaces.

**There is no good operating point on B3DB.** Either the policy barely moves
(kl=10.0: KL 0.011, C 0.682, scaffold preserved at 0.853 because almost nothing
was learned) or it learns and degrades diversity (kl=0.3: C 0.929, scaffold
0.687, unique 0.792). BBBP has a setting that does both; B3DB does not, at least
not among these five.

What transfers regardless: **sanitization validity 0.9997–0.9999 at every
coefficient**, and duplicate-freedom at kl >= 1.0 (unique 0.983–0.985). The
assembly engine is dataset-independent; the reward's behaviour is not.

Caveats: one GRPO seed and one warm start per cell. 6 start-vs-end tests across
the item, threshold x6 = 0.0083; the two significant results clear it by >10
orders of magnitude and the BBBP reference clears it at 0.0042.

**The reward-headroom explanation was tested and is EXCLUDED.** The proposed
mechanism was that B3DB's reward has more room left to climb — real molecules
average C 0.663 under the B3DB ensemble against 0.755 under BBBP's — so its
policy climbs further (+0.305 vs +0.112) and pays diversity for the climb.

Tested on the `--reward-seed` axis, which moves BBBP headroom independently of
the dataset. Seed 1 has the most headroom (real-molecule C 0.688), seed 5 the
least (0.787); that 0.098 span exceeds the 0.092 BBBP→B3DB gap that produced
the effect. Three GRPO seeds each, kl=1.0, ws0, 250 steps
(`results/brics_headroom/`):

| reward seed | headroom | C climb | scaffold delta |
|---|---|---|---|
| 1 | 0.688 | +0.113 +/- 0.007 | +0.014 +/- 0.008 |
| 5 | 0.787 | +0.092 +/- 0.004 | +0.012 +/- 0.011 |

| pre-specified | seed1 − seed5 | p | predicted? |
|---|---|---|---|
| C climb higher | +0.0219 | 0.063 | direction yes, not significant |
| scaffold delta lower | **+0.0026** | 0.86 | **no** |

**Half the mechanism holds and the load-bearing half does not.** More headroom
does buy a longer climb in `C` (+0.022, directionally as predicted, though
p = 0.063 against the x2 threshold of 0.025) — which is close to definitional,
since a lower starting C leaves more room. But the *consequence* the hypothesis
requires, that climbing costs diversity, does not follow at all.

This is an exclusion rather than a null: the difference is +0.003 with a 95% CI
of [−0.037, +0.043], while the predicted effect scaled to this headroom span is
**−0.217**, which the CI excludes by 15.8 SEM.

So B3DB's behaviour is not explained by how saturated its reward is. Candidates
not yet excluded include corpus size (7805 vs 2039 molecules), action-space size
(~11.6k vs 6051), and label composition — **no replacement mechanism is proposed
here**, because the data that falsified this one cannot support a successor.

---

## 2. The permeability classifier

### 2.1 The benchmark does not predict the behaviour that matters **[measured]**
Test ROC-AUC measures in-distribution accuracy. The generator queries
out-of-distribution by construction. On the only labelled OOD data available,
the benchmark leader (`hybrid/gat`, 0.9608) and the reward's ensemble
(`gin+gat+gine`, ~0.93) are indistinguishable: paired deltas −0.0001
[−0.012, +0.012] and +0.0083 [−0.051, +0.067].

### 2.2 Ensemble agreement is shared bias, not confidence **[measured]**
On generated molecules member sd is 0.005 against 0.063 on real BBBP — but MC
dropout shows per-model uncertainty is **flat** in logit space (0.86 real, 0.99
collapsed). The apparent certainty is sigmoid compression. `mc_over_member`
rises to 1.25 on collapsed samples, i.e. a single model is *less* certain than
the three-member agreement implies. The three members share a feature set, a
split and a skeleton.

### 2.3 The reward's ranking is noise past a point **[measured]**
In logit space, between-molecule signal falls 2.26 → 0.41 over 200 steps while
MC-dropout noise stays flat near 0.95. **SNR crosses 1 between steps 100 and
150.** Past that the classifier's own uncertainty exceeds the differences it
reports.

### 2.4 No calibration analysis **[untested]**
No reliability diagram, no Brier/ECE, on either in- or out-of-distribution
data. The reward consumes `C(m)` as if it were a probability.

---

## 3. The data

### 3.1 BBBP labels transport, not drug-likeness or safety **[measured]**
This is the root cause of §1.2. Chloroform, dichloromethane, bromoform,
nitrous oxide, cyclopropane, ethanol and vinyl ether are all in BBBP labelled
**BBB+**, correctly — anaesthetics and solvents do cross the barrier. Of the 65
molecules with ≤8 heavy atoms, **62 are positive**. Carbon tetrachloride is not
an extrapolation failure; it interpolates between two positive training
examples. No reward term changes what the label means.

### 3.2 The external holdouts are underpowered and mutually inconsistent **[measured]**
The leak scrub is rigorous and removes 96% of rows: Adenot 1650 → 59, Wang
1592 → 55. They then disagree violently under the same procedure — AUC **0.997
vs 0.808**. A near-perfect result on 59 molecules is a red flag about that
holdout, not evidence of a good model. Neither should be quoted as a
generalization claim; they support paired model-vs-model comparison only.

### 3.3 Tox21 measures the wrong kind of toxicity for this failure **[measured]**
The trained model is sound (test AUC 0.808, scaffold split). It is also close
to an inverse drug-likeness detector here: it scores CCl₄ at 0.251 and
caffeine at 0.302, so the non-toxicity term **rewards the solvents and
penalizes the drugs**, and CCl₄'s margin over caffeine *grows* with the
weight. Tox21's twelve assays are nuclear-receptor and stress-response panels,
which fire on drug-sized aromatic scaffolds; CCl₄'s hepatotoxicity runs
through CYP2E1 activation, not a Tox21 endpoint. The term is implemented and
**off by default**; it should not be switched on as-is.

### 3.4 BBBP's `name` column appears misaligned **[suspected]**
`C=C` is named "ethyl-loflazepate", `C(C)Cl` is named "ethylene(ethene)",
`C(Cl)Cl` is named "18". SMILES/label pairing looks correct where spot-checked
(`CCO`/Ethanol, `C1CC1`/Cyclopropane), so this may be confined to the name
column and to specific rows — but it has not been audited, and nothing should
rely on that column until it has been.

### 3.5 No toxicity or drug-likeness label matched to the failure **[untested]**
What §1.2 needs is an "is this a drug" signal (ClinTox `FDA_APPROVED`) or an
organ/acute endpoint (LD50, DILI). Neither has been obtained or tested.

---

## 4. Statistics and measurement

### 4.1 Three seeds, and nothing separates **[measured]**
0/6 base-GNN pairs separate at 95% on either dataset. Pairing helps
substantially (median paired sd 0.40× the marginal on BBBP; `gat − gin` is a
1.4 sd effect) but is not sufficient. The 10-seed study has not been run.

### 4.2 The paired standard deviations are themselves unreliable **[measured]**
At n=3 the paired sd carries **2 degrees of freedom**, so `dz` and the implied
seeds-to-significance are noisy. `hybrid:gcn − hybrid:gin` shows dz = −14.9
from three consistent differences amounting to **0.16% AUC** — statistically
separable, practically irrelevant.

### 4.3 Structural diversity metrics are sample-size dependent **[measured]**
Scaffold fraction reads 0.758 at n=199 and 0.595 at n=982 on the same BBBP.
That swing exceeds any effect being measured. `diversity.reference_band`
handles this, but any comparison made without it — including comparisons
between two generated sets of different sizes — is invalid.

### 4.4 Most per-step diagnostics come from one group of 64 **[measured]**
*Within-group `scaffold_frac` at n=64 is validated as a COMPARATIVE statistic
and must not be read as a support fraction. The conflation is real but it does
not bite, because a wide-support control scores the same as the policy.*

M=8 groups of N=64 per checkpoint, against real BBBP molecules drawn the same
way as the wide-support control:

| source | within-group | pooled (n=512) | distinct scaffolds, 1→8 groups |
|---|---|---|---|
| trained (kl=1.0, seed 0) | 0.877 +/- 0.013 | 0.678 | 60 → 347 (5.8x) |
| warm start | 0.828 +/- 0.013 | 0.633 | 52 → 324 (6.2x) |
| **real BBBP (control)** | **0.879 +/- 0.011** | **0.633** | 56 → 324 (5.8x) |

**The within-to-pooled drop is a sample-size property, not evidence of a narrow
policy.** Real BBBP molecules show the same drop (ratio 0.72) as the trained
policy (0.77). A distribution with ~1000 scaffolds in it still scores 0.879
within-group at n=64, so 0.879 is approximately the ceiling the sample size
imposes and not a statement about support.

**Re-measured at n=32 groups per arm, identical procedure, exactly 64 molecules
per group: training RAISED diversity rather than degrading it.**

| arm | within-group (n=32) |
|---|---|
| trained (kl=1.0, seed 0) | 0.8838 +/- 0.0077 |
| warm start | 0.8296 +/- 0.0088 |
| real BBBP (control) | 0.8608 +/- 0.0075 |

| comparison | diff | t | p | x3 | verdict |
|---|---|---|---|---|---|
| trained vs warm start | +0.0542 | 4.63 | <0.0001 | <0.001 | **established** |
| warm start vs real | −0.0312 | −2.70 | 0.0089 | 0.027 | survives |
| trained vs real | +0.0229 | 2.13 | 0.037 | 0.11 | **does not survive** |

So at `kl_coef=1.0`, GRPO moved the policy from measurably *below* the
real-molecule control to *at or slightly above* it. **That is the strongest form
of "reward hacking is contained" in this project** — not merely that diversity
failed to collapse, but that optimization increased it. The further claim that
the trained policy *exceeds* real molecules does not survive correction for the
three comparisons and is not made.

**But the statistic resolves support poorly in exactly this band**, so "at the
real level" bounds the support loosely. Pools built from BBBP with a known
number of distinct scaffolds, same statistic:

| scaffolds in pool | within-group @n=64 |
|---|---|
| 1021 | 0.871 +/- 0.013 |
| 500 | 0.844 +/- 0.010 |
| 300 | 0.830 +/- 0.015 |
| 150 | 0.514 +/- 0.022 |
| 20 | 0.268 +/- 0.007 |

It discriminates sharply below ~300 scaffolds and weakly above: a 3.4x change in
support (300 → 1021) moves it only 0.830 → 0.871. Both the trained policy and
the control sit in the weakly-resolved region, so "comparable within-group
score" is consistent with support differing by a factor of two. The
scaffold-accumulation curve (347 distinct over 512 molecules for the trained
policy against 324 for the control) is the better instrument there, and shows no
deficit — stated descriptively, as it is one draw per arm.

**A procedure note that explains an earlier discrepancy.** The warm start
measured 0.854 in one place and 0.828-0.830 here. Cause: `scaffold_frac` is
n-dependent, and the earlier figure divided by the ~62 valid molecules in a
64-sample draw while these divide by exactly 64. Verified: same checkpoint,
32 groups, gives 0.830 with the fixed-64 procedure and 0.854 keeping all valid
from a 64-draw. Every arm in the tables above uses exactly 64, so the
comparisons are matched; cross-measurement comparisons of this statistic are
only valid at matched molecule count.

Consequences for existing numbers:

* Diversity comparisons between arms measured identically remain valid, which
  is every frontier number and the 0.864 headline. The kl frontier spans
  0.46 to 0.91 against a control at 0.879 — far outside what sample size alone
  explains.
* The historical collapse claims (SELFIES scaffold 0.07–0.14) are genuine
  narrowing: they sit an order of magnitude below the real-molecule reference
  under the identical statistic.
* **Absolute within-group values must not be quoted as diversity fractions.**
  "0.88 of scaffolds are distinct" is wrong; "0.88, the same as real drugs at
  this sample size" is right.

Driver: ad hoc (see the commit that closed this item). Not promoted to a module
because it answers a question about a statistic rather than producing a
recurring diagnostic.

---

## 4b. Validation protocols now in force

### 4b.1 The held-out instrument rule
Any reward term meant to suppress a structural feature must be evaluated on
libraries the policy never saw during optimization
(`alerts.cross_instrument_check()`). §10.2 is what happens without it: a
term that looks like a 40% improvement on its own metric and moves nothing
independent. Corollary — never train against the union of every available
instrument, because that consumes the only detector.

### 4b.2 Alert rates are instrument-specific
Rates from different libraries are not comparable. Targeted-vs-BRENK
Jaccard on BBBP BBB+ is 0.36. Every reported rate must name its instrument.
**[measured]**

### 4b.3 Ranking requires discrimination
Ranking candidates by `C` is meaningful only while `C` separates them. At
step 199 of the baseline its within-group sd was 0.006, below the model's
own MC-dropout uncertainty, so a ranking there orders noise. `postfilter`
reports the survivor spread and withholds the recommendation below sd 0.02.
**[measured]**

## 5. Engineering

### 5.0 GraphNorm is the default; BatchNorm still scores higher **[measured]**

`BatchNorm1d` makes a molecule's logit a function of what shares its batch.
That is a training detail for a classifier and a correctness problem for a
frozen reward: under BatchNorm in train mode the same molecule earns a
different reward depending on which candidates land in its GRPO group, so the
reward stops being a function of the molecule. The default moved to **GraphNorm**, which
normalizes per molecule and is batch-independent by construction. Measured by `python -m src.models`,
gin on a 4-graph batch in train mode: batch drifts **3.97e-02**, layer 3.7e-09,
graph 6.0e-08.

**It is not free, and the trade is not obviously worth it.** BatchNorm vs
LayerNorm, 10 seeds, paired on seed (identical split), test ROC-AUC:

| dataset | gcn | sage | gin | gat |
|---|---|---|---|---|
| BBBP | **+0.0215** +/- 0.0079 | +0.0459 +/- 0.0251 | +0.0269 +/- 0.0146 | **+0.0168** +/- 0.0058 |
| B3DB | +0.0030 +/- 0.0032 | -0.0018 +/- 0.0041 | +0.0096 +/- 0.0053 | -0.0007 +/- 0.0032 |

(SEM; bold exceeds 2 SEM. Pooled over all 8 cells, n=80:
**batch - layer = +0.0152 +/- 0.0042**.)

BatchNorm is better on BBBP in all four cells and indistinguishable from zero
on B3DB in all four. That pattern is a **small-dataset effect**: BBBP trains on
1572 molecules where batch statistics regularize, B3DB on 6244 where they do
not matter. Runs are under `results/norm_arm_batch/`; GraphNorm was also tried
and is tied with LayerNorm (+0.0072 +/- 0.0108 on bbbp/gcn), so it does not
recover the gap.

**Why GraphNorm rather than LayerNorm.** Both were the options on the table and
both are batch-independent; GraphNorm measured higher in all four BBBP cells and
~zero in all four B3DB cells, pooled **+0.0077 +/- 0.0043 SEM** over 80 paired
runs. That is 1.8 SEM and does **not** clear 95%, so the switch rests on sign
consistency plus the prior that per-graph normalization suits variable-size
molecules -- not on a proven difference. GraphNorm recovers roughly half the gap
to BatchNorm (pooled over the same 8 cells: batch 0.8959, graph 0.8885,
layer 0.8807). LayerNorm runs are archived under `results_layernorm/`.

**The honest counterargument against the current default.** The
batch-dependence hazard was *already* contained by the `eval()` calls in
`reward.py` and `train_gan.py`, which `reward._self_check` asserts. LayerNorm
buys structural immunity to a bug class this project has repeatedly been bitten
by, at a measured ~0.015-0.02 ROC-AUC on the smaller dataset. That is a
judgement call, not a measurement, and it is **unresolved**. Flipping it is
`--norm batch` plus a re-sweep; `featurize.run_contract` replays whichever norm
a checkpoint recorded, so the generation half does not break either way.

### 5.0b The virtual node helps, and only where receptive field is the bottleneck **[measured]**

`num_layers=3` caps an atom's receptive field at three bonds, so on a
40-heavy-atom drug the two ends of the molecule never see each other -- while
BBB permeability is driven by whole-molecule TPSA, logP and size. The mean+max
readout supplies a global view at the *output*, but the convolutions cannot
condition on it. A per-graph state injected between layers makes the effective
diameter 2 at any molecule size.

Paired on seed, 10 seeds, test ROC-AUC, virtual node minus baseline:

| dataset | gcn | sage | gin | gat |
|---|---|---|---|---|
| BBBP | +0.0033 | +0.0492 | **+0.0319** | +0.0123 |
| B3DB | -0.0055 | -0.0069 | **+0.0114** | +0.0036 |

Pooled over 80 paired runs: **+0.0124 +/- 0.0045 SEM**.

The interesting part is not the mean, it is *which* operators gain. On BBBP the
virtual node **collapses the spread between the four operators from 0.031 to
0.019**, and it does so by lifting the two weakest:

| | gcn | sage | gin | gat | spread |
|---|---|---|---|---|---|
| baseline | 0.883 | 0.857 | 0.864 | 0.888 | 0.031 |
| +virtual node | 0.886 | **0.906** | **0.896** | 0.900 | 0.019 |

SAGE and GIN were the weakest BBBP operators and gain the most (+0.049,
+0.032); GCN and GAT were already the strongest and gain least. That is what a
receptive-field explanation predicts: the operators differed largely in how
well they propagated information across the molecule, and handing all four a
global channel equalizes them. It is also a caution for the architecture
comparison -- part of what `src/models.py` was measuring as "aggregation
scheme" was reach, not aggregation.

B3DB shows no consistent effect (two cells slightly negative), the same
dataset-size pattern as 5.0: the gain appears where data is scarce.

**Not yet the default.** One arm, base family only, and `models_edge`,
`models_hybrid` and `models_tox` raise `NotImplementedError` on
`virtual_node=True`. Runs are under `results/virtual_node/`.

### 5.1 No dependency pinning or setup script **[measured]**
This container started with nothing installed, and the PyTorch CDN is blocked
by the proxy (403), so torch has to come from PyPI. There is no
`requirements.lock` and no SessionStart hook, so every fresh session pays that
cost again.

### 5.2 Self-checks are not run by anything **[measured]**
Every module has a `_self_check()` and they are good, but there is no CI, no
test runner, and no pre-commit hook. They pass because they were run by hand.

### 5.3 Model checkpoints are committed **[measured]**
`results/` carries `.pt` files (11 MB for `results/gan/` alone). Convenient for
cold start — the README leans on it — but it grows with every experiment.

---

## 6. Framing

### 6.1 Outputs are hypotheses, not candidates
No in-silico filter makes a generated molecule safe. These models are trained
on a few thousand compounds; toxicity prediction is substantially harder than
BBB prediction; and §3.3 shows that a sound-looking toxicity model can be
actively anti-correlated with what you want. Anything this pipeline emits is a
**candidate for evaluation**, never for synthesis.

### 6.2 The objective itself is partly adverse
Optimizing for CNS penetration without a counterweight preferentially finds
CNS-penetrant small molecules, and the most reliably CNS-penetrant compounds in
the training data are volatile anaesthetics and chlorinated solvents. This is a
property of the objective, not a bug in the optimizer.

---

## Corrections already made

Recorded because each was a plausible hypothesis that measurement overturned.

1. "195/199 distinct samples, so no mode collapse" — wrong metric; string
   uniqueness is blind for ~100 steps.
2. "The collapse is property-space, with only mild scaffold narrowing" — true
   at step 40, over-generalized; at 200 steps scaffold collapse is larger.
3. "The reward uses the wrong classifier" — overclaimed; no measurable OOD
   difference.
4. "Off-distribution the ensemble will disagree" — backwards; it agrees more.
5. "Architecture differences are hopeless at 3 seeds, noise is 3× signal" —
   used marginal SDs; the design is paired.
6. "The reward has gone blind" — overstated; the information is there, on a
   scale that could not show it.
7. Descriptor distance, applicability domain and Tox21 were each expected to
   catch the solvents. None does.
8. "The alert penalty makes the generator cleaner than approved drugs." True
   only on the instrument it was trained against; on BRENK it is 0.406
   against the drugs' 0.350. Five of six guard features have now failed or
   failed to generalize — the pattern is evidence about counterweighting as
   an approach, not about the individual terms.
