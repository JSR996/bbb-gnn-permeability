# External holdout evaluation — results

Every trained checkpoint (72 total: base GCN/SAGE/GIN/GAT, hybrid GNN+descriptor,
GINE/gat_edge/sage_edge edge-feature ablation, D-MPNN — 2 datasets x 3 seeds
each family) scored against the two leak-free external holdout sets
(`adenot_clean.csv`, n=59; `wang_clean.csv`, n=55), which are scaffold-disjoint
from both BBBP and B3DB. No model has seen these molecules in any form.

Run command: `python -m src.eval_external_holdout --device cpu`
Files: `external_holdout_runs.csv` (144 rows, every checkpoint x holdout, with
confusion matrices) and `external_holdout_summary.csv` (mean +/- std over the
3 seeds per family/dataset/model/holdout).

## Headline finding: the two holdouts behave completely differently

- **Adenot (n=59, 15 pos / 44 neg):** every single architecture, in every
  family, scores ROC-AUC between 0.983 and 1.000. Several hit exactly 1.000
  (base GCN/bbbp, several seeds). This is nearly ceiling performance
  regardless of model choice.
- **Wang (n=55, 38 pos / 17 neg):** ROC-AUC spreads from 0.795 to 0.898 across
  the same checkpoints, and MCC drops sharply (0.40-0.61 vs. 0.68-0.90 on
  Adenot). This is the holdout that's actually discriminating between models.

Practical implication: any generalization claim in the paper should be made
against Wang, not Adenot — Adenot is too easy (at this n, with this class
balance) to distinguish architectures at all. Reporting only an
average-across-both-holdouts number would hide this and make every model look
artificially strong.

## On Wang, ranked by mean ROC-AUC

Top 5:
1. base / bbbp / sage — 0.898 (MCC 0.530)
2. hybrid / bbbp / sage — 0.891 (MCC 0.551)
3. edge_ablation / b3db / sage_edge — 0.885 (MCC 0.583)
4. edge_ablation / b3db / gine — 0.878 (MCC 0.571)
5. hybrid / b3db / gcn — 0.875 (MCC 0.609)

Bottom 5:
- base / bbbp / gin — 0.795 (MCC 0.432)
- edge_ablation / b3db / gat_edge — 0.809 (MCC 0.406)
- base / bbbp / gat — 0.812 (MCC 0.402)
- edge_ablation / bbbp / gine — 0.814 (MCC 0.467)
- hybrid / bbbp / gcn — 0.815 (MCC 0.453)

## The hybrid descriptor advantage does not clearly survive true external shift

In-distribution (scaffold-split test sets), hybrid fusion was the strongest
family across the board. On the Wang holdout that edge does not hold up
consistently:
- SAGE/BBBP: **base (0.898) actually beats hybrid (0.891)** — the descriptor
  concatenation adds nothing here and may add slight noise.
- GCN/B3DB: hybrid (0.875) does clearly beat base (0.840) — here it helps.

So the hybrid model's benefit looks conv-operator- and training-dataset-
dependent rather than a uniform property of adding descriptors. This is worth
a sentence in the paper rather than assuming the in-distribution ranking
transfers.

## D-MPNN and GINE

Neither the D-MPNN reimplementation nor GINE/edge-feature ablation produces a
consistent edge over the plain base GNNs on Wang (D-MPNN/bbbp: 0.826;
GINE/b3db: 0.878, close to the best base result). Same conclusion as the
in-distribution comparison: edge/bond-level information isn't buying much on
this task, on this data.

## Caveat

n=55-59 per holdout is small; individual confusion-matrix cells (in
`external_holdout_runs.csv`) show single-digit false positive/negative counts
driving several of the MCC swings between seeds of the *same* model — treat
the ranking above as suggestive, not a settled result, and consider bootstrap
CIs before quoting these numbers in the paper.
