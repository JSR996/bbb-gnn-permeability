# Known Limitations

Everything currently wrong, unproven, or unresolved in this project, as of the
reward-hacking investigation. Companion to `reward_hacking_investigation.md`,
which has the evidence behind most of these.

Each item is marked:

- **[measured]** — demonstrated with numbers in this repo
- **[suspected]** — plausible, not yet tested
- **[untested]** — a claim the project relies on that nobody has checked

The generation half is a research prototype. **Nothing it emits is a synthesis
candidate**, and several items below explain why that is not conservatism.

---

## 1. The generator

### 1.1 Reward hacking is total, not partial **[measured]**
200 steps × 3 seeds: scaffold fraction 0.80 → 0.07–0.14, Tanimoto distance
0.90 → 0.25–0.45, descriptor spread ~0.45 → ~0.12, while classifier reward
rises 0.88 → 0.99. Validity stays at 100% throughout, so every metric the loop
tracked before this work says the run succeeded.

### 1.2 It generates non-drugs that outscore real drugs **[measured]**
The 150-step logit arm converged on carbon tetrachloride, freon-12, freon-11
and tetrafluoromethane. At the late-training weights CCl₄ scores **2.280**
against caffeine's **2.063**. The kl=0.02 baseline failed differently — 41 of
64 copies of a single biphenyl amide.

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

### 1.6 The size floor has never been run in training **[untested]**
It separates on a 7-molecule probe. Whether it holds up as a reward term under
150 steps of adversarial pressure from the generator is unknown, and the
pattern of this investigation is that such things usually do not.

### 1.7 The reward's terms have never been ablated **[untested]**
"The collapse is driven by C" is an inference from the saturation data, not a
measured decomposition. `d_var` is logged but the discriminator's contribution
has never been isolated, and `w_D`, `w_Q`, `w_S` have never been varied.

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

## 5. Engineering

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
