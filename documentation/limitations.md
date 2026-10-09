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

### 1.8 `floor = -2.0` is untuned **[untested]**
Carried over from `cold_and_warm_start.md`, never swept.

### 1.9 Single-seed interventions **[measured, as a gap]**
The 3-seed discipline was applied to the baseline only. The KL sweep, the
raw-vs-logit A/B and the Tox21 probe are all **seed 0 alone**. Given that seed
0 collapsed markedly harder than seeds 1 and 2 in the baseline (scaffold 0.17
vs 0.75/0.73 at step 90), every intervention result should be treated as
provisional until replicated.

### 1.10 Runs are short, and no long run has been validated **[untested]**
150–200 steps. The originally planned 500-step run has correctly not been
attempted, but that means nothing is known about behaviour past step 200.

### 1.11 Generation has only ever been run on BBBP **[untested]**
B3DB is trained and benchmarked on the classifier side but has never been used
as a generation target.

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

### 4.4 Most per-step diagnostics come from one group of 64 **[untested]**
Within-group scaffold fraction conflates "the policy is narrow" with "64 draws
is a small sample". Sampling M groups per checkpoint and comparing within-group
against pooled diversity would separate them. Not done — so every "diversity
collapsed" claim assumes the policy's support genuinely shrank, rather than
having verified it.

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

### 5.0 LayerNorm is the default and it costs accuracy on BBBP **[measured]**

`BatchNorm1d` makes a molecule's logit a function of what shares its batch.
That is a training detail for a classifier and a correctness problem for a
frozen reward: under BatchNorm in train mode the same molecule earns a
different reward depending on which candidates land in its GRPO group, so the
reward stops being a function of the molecule. The default moved to LayerNorm,
which is batch-independent by construction. Measured by `python -m src.models`,
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

**The honest counterargument against the current default.** The
batch-dependence hazard was *already* contained by the `eval()` calls in
`reward.py` and `train_gan.py`, which `reward._self_check` asserts. LayerNorm
buys structural immunity to a bug class this project has repeatedly been bitten
by, at a measured ~0.015-0.02 ROC-AUC on the smaller dataset. That is a
judgement call, not a measurement, and it is **unresolved**. Flipping it is
`--norm batch` plus a re-sweep; `featurize.run_contract` replays whichever norm
a checkpoint recorded, so the generation half does not break either way.

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
