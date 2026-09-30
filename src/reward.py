"""Reward assembly for GRPO: frozen permeability classifier + QED + SA + D.

Implements Eq. 5 (reward), Eq. 6 (annealed weights) and Eq. 7 (validity-gated
switch) from the continuation draft. The discriminator term is passed IN rather
than owned here -- D_phi is trained by the GAN loop and is the only
non-stationary part of the reward (Section 3.1), so keeping it out of this
module makes the stationary terms independently testable.

    python -m src.reward        # self-check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig
from torch_geometric.data import Batch

from .featurize import smiles_to_graph
from .models import build_model
from .models_edge import EDGE_MODELS, build_edge_model
from .train import RESULTS_DIR

RDLogger.DisableLog("rdApp.*")

# sascorer ships with RDKit but lives in Contrib, which is not on sys.path.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

SA_MIN, SA_MAX = 1.0, 10.0  # sascorer's documented range; lower = easier to make


def _checkpoint(root: Path, dataset: str, model: str, seed: int) -> Path:
    """Locate a run, whether it landed in the core grid or the edge ablation.

    The edge-aware operators (gine, gat_edge) were run separately and live
    under results/edge_ablation/, not results/<dataset>/. Their saved fold
    indices match the core runs seed-for-seed, so the two sets are genuinely
    ensemble-compatible -- see _self_check.
    """
    for path in (
        root / dataset / model / f"seed{seed}" / "model.pt",
        root / "edge_ablation" / dataset / model / f"seed{seed}" / "model.pt",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no checkpoint for {model}/{dataset}/seed{seed}. Train it first: "
        f"python -m src.train --model {model} --dataset {dataset} --seed {seed}"
    )


def load_frozen_classifier(
    dataset: str = "bbbp",
    models: tuple[str, ...] = ("gin", "gat", "gine"),
    seed: int = 0,
    device: str = "cpu",
    results_dir: Path | None = None,
):
    """Return C(mols) -> np.ndarray of mean P(permeable), frozen and stationary.

    eval() is load-bearing, not hygiene: the classifier uses BatchNorm1d, so in
    train mode C(m) would depend on which other molecules share the batch. The
    reward would then vary with group composition and the "frozen C is
    stationary in k" premise of Section 3.1 would be quietly false.

    Checkpoints are combined within a single seed only -- each seed defines a
    different scaffold split (README), so mixing seeds mixes models that saw
    each other's validation molecules.
    """
    root = results_dir or RESULTS_DIR
    nets = []
    for name in models:
        ckpt = _checkpoint(root, dataset, name, seed)
        # Edge-aware operators carry a different skeleton class, so they must
        # be rebuilt by the module that trained them or the state_dict is
        # being loaded into the wrong architecture.
        builder = build_edge_model if name in EDGE_MODELS else build_model
        net = builder(name).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        net.requires_grad_(False)
        nets.append(net)

    @torch.no_grad()
    def classify(mols: list[Chem.Mol]) -> np.ndarray:
        if not mols:
            return np.zeros(0, dtype=float)
        graphs = [smiles_to_graph(Chem.MolToSmiles(m)) for m in mols]
        batch = Batch.from_data_list(graphs).to(device)
        probs = [torch.sigmoid(net(batch)) for net in nets]
        return torch.stack(probs).mean(0).cpu().numpy()

    return classify


def sanitizable(smiles: str) -> Chem.Mol | None:
    """Chemical validity, i.e. RDKit sanitization -- NOT grammar validity.

    Section 1.1: a SELFIES string always parses to some molecule, so the
    generator's syntactic guarantee says nothing about whether the result
    survives sanitization. These are different failure modes; only this one
    gates the reward.
    """
    mol = Chem.MolFromSmiles(smiles)  # returns None iff sanitization fails
    return None if mol is None or mol.GetNumAtoms() == 0 else mol


def score_terms(smiles: list[str], classify) -> dict[str, np.ndarray]:
    """Per-molecule stationary reward terms. Invalid rows are left at zero."""
    n = len(smiles)
    valid = np.zeros(n, dtype=bool)
    c = np.zeros(n)
    qed = np.zeros(n)
    sa = np.zeros(n)

    mols, keep = [], []
    for i, smi in enumerate(smiles):
        mol = sanitizable(smi)
        if mol is not None:
            mols.append(mol)
            keep.append(i)

    if mols:
        idx = np.array(keep)
        valid[idx] = True
        c[idx] = classify(mols)
        qed[idx] = [QED.qed(m) for m in mols]
        # Flip and rescale so higher is better, matching the other three terms.
        raw_sa = np.array([sascorer.calculateScore(m) for m in mols])
        sa[idx] = (SA_MAX - raw_sa) / (SA_MAX - SA_MIN)

    return {"valid": valid, "c": c, "qed": qed, "sa": sa}


def weights_at(
    k: int,
    w0: dict[str, float],
    wf: dict[str, float] | None = None,
    schedule: str = "fixed",
    k_anneal: int = 1000,
    valid_history: list[float] | None = None,
    tau: float = 0.9,
    window: int = 10,
) -> dict[str, float]:
    """Reward weights at generator step k. Eq. 6 (linear/cosine), Eq. 7 (gated).

    Pure function: the validity-gated variant reads the caller's own history
    list rather than owning a rolling buffer, so nothing here carries state
    across steps and each schedule is checkable in isolation.
    """
    if schedule == "fixed" or wf is None:
        return dict(w0)

    if schedule == "linear":
        lam = min(k / k_anneal, 1.0)
    elif schedule == "cosine":
        lam = 0.5 * (1.0 - np.cos(np.pi * min(k / k_anneal, 1.0)))
    elif schedule == "gated":
        # Hard switch once the W-step mean validity clears tau. Before the
        # window has filled there is no evidence to switch on, so stay at w0.
        recent = (valid_history or [])[-window:]
        lam = 1.0 if len(recent) == window and float(np.mean(recent)) >= tau else 0.0
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    return {key: w0[key] + (wf[key] - w0[key]) * lam for key in w0}


# Calibration of the logit transform, measured once on the BBBP reference set
# (n=2039, seed-0 gin/gat/gine ensemble):
#
#   raw C    mean 0.7447   sd 0.2880
#   logit C  mean 1.8591   sd 2.1581
#
# The affine map below sends logit C onto raw C's mean and sd over that
# reference. Without it the transform would also multiply the C term's
# effective weight by ~7.5x, and an A/B against raw C would be confounded by
# scale rather than isolating the change in shape. Recompute with
# `python -m src.reward --calibrate` if the ensemble or dataset changes.
LOGIT_CALIB = {"mu_raw": 0.7447, "sd_raw": 0.2880,
               "mu_logit": 1.8591, "sd_logit": 2.1581}


def transform_c(c: np.ndarray, mode: str = "raw",
                calib: dict[str, float] | None = None) -> np.ndarray:
    """Rescale the permeability term. `raw` is the identity.

    `logit` exists because the classifier's output saturates near 1 where the
    sigmoid compresses real differences into nothing: measured on the
    collapsed baseline, between-molecule sd is 0.006 in probability space but
    0.35 in logit space, and MC-dropout uncertainty is FLAT across that same
    range in logit space (0.86 on real BBBP, 0.99 on collapsed samples) while
    appearing to shrink 3.6x in probability space. The apparent certainty is
    the scale, not the model. GRPO ranks within a group, so a scale that
    compresses differences costs it exactly the signal it consumes.

    The transform is calibrated to leave the term's mean and sd unchanged on
    the reference set, so w_C keeps its meaning across modes.
    """
    if mode == "raw":
        return c
    if mode != "logit":
        raise ValueError(f"unknown c transform {mode!r}")
    k = calib or LOGIT_CALIB
    p = np.clip(c, 1e-6, 1.0 - 1e-6)
    z = (np.log(p / (1.0 - p)) - k["mu_logit"]) / k["sd_logit"]
    return k["mu_raw"] + k["sd_raw"] * z


def assemble(
    terms: dict[str, np.ndarray],
    d_scores: np.ndarray,
    w: dict[str, float],
    invalid_reward: float = 0.0,
    c_transform: str = "raw",
) -> np.ndarray:
    """Eq. 5. Invalid molecules collapse to a flat penalty, bypassing every term.

    `c_transform` rescales the permeability term only; see `transform_c`.
    It is applied here rather than in `score_terms` so that `terms["c"]`
    stays the raw probability for logging and diagnostics, and a history
    written under one mode remains comparable to one written under the other.
    """
    r = (
        w["D"] * np.asarray(d_scores, dtype=float)
        + w["C"] * transform_c(terms["c"], c_transform)
        + w["Q"] * terms["qed"]
        + w["S"] * terms["sa"]
    )
    return np.where(terms["valid"], r, invalid_reward)


def _self_check() -> None:
    # Validity gate: the second string is syntactically fine and chemically not.
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "C(C)(C)(C)(C)C", "CCO", "not-a-molecule"]
    assert sanitizable(smis[0]) is not None
    assert sanitizable(smis[1]) is None, "5-valent carbon must fail sanitization"
    assert sanitizable(smis[3]) is None

    # Ensemble members must share a scaffold split. The seed reseeds the split
    # (README), so averaging models from different folds would mean averaging
    # models that trained on each other's held-out molecules.
    folds = {
        name: np.load(_checkpoint(RESULTS_DIR, "bbbp", name, 0).parent / "test_preds.npz")["idx"]
        for name in ("gin", "gat", "gine")
    }
    for name, idx in folds.items():
        assert np.array_equal(idx, folds["gin"]), f"{name} trained on a different fold"

    # Frozen classifier: same molecule must score identically in any company.
    classify = load_frozen_classifier()
    alone = classify([Chem.MolFromSmiles("CCO")])
    crowded = classify([Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CCN"]])
    assert np.allclose(alone[0], crowded[0], atol=1e-6), (
        f"C(m) moved with batch composition ({alone[0]} vs {crowded[0]}): "
        "BatchNorm is in train mode and the reward is not stationary"
    )

    terms = score_terms(smis, classify)
    assert terms["valid"].tolist() == [True, False, True, False]
    assert terms["c"][1] == 0.0 and terms["qed"][3] == 0.0
    assert (0.0 <= terms["c"][terms["valid"]]).all()
    assert (terms["c"][terms["valid"]] <= 1.0).all()

    w = {"D": 1.0, "C": 1.0, "Q": 0.5, "S": 0.5}
    r = assemble(terms, np.full(4, 0.5), w)
    assert r[1] == 0.0 and r[3] == 0.0, "invalid molecules must not collect reward"
    assert r[0] > 0 and r[2] > 0

    # Schedules. Linear hits the endpoints; gated needs a full window over tau.
    w0 = {"D": 1.0, "C": 0.1, "Q": 0.5, "S": 0.5}
    wf = {"D": 0.1, "C": 1.0, "Q": 0.5, "S": 0.5}
    assert weights_at(0, w0, wf, "linear", k_anneal=100)["C"] == 0.1
    assert weights_at(100, w0, wf, "linear", k_anneal=100)["C"] == 1.0
    assert weights_at(999, w0, wf, "linear", k_anneal=100)["C"] == 1.0, "must clamp"
    assert weights_at(50, w0, wf, "cosine", k_anneal=100)["D"] == 0.55
    assert weights_at(5, w0, wf, "gated", valid_history=[1.0] * 3)["C"] == 0.1
    assert weights_at(5, w0, wf, "gated", valid_history=[1.0] * 10)["C"] == 1.0
    assert weights_at(5, w0, wf, "gated", valid_history=[0.5] * 10)["C"] == 0.1
    assert weights_at(5, w0, wf)["C"] == 0.1, "no wf -> fixed"

    # --- logit transform ------------------------------------------------
    # Identity when asked for, and order-preserving always: the transform may
    # rescale the term but must never reorder two molecules, or it is not a
    # recalibration but a different objective.
    c = np.array([0.05, 0.3, 0.5, 0.9, 0.99, 0.999])
    assert np.array_equal(transform_c(c, "raw"), c)
    t = transform_c(c, "logit")
    assert np.all(np.diff(t) > 0), f"logit transform reordered molecules: {t}"

    # Calibrated to leave the reference mean and sd alone, so w_C keeps its
    # meaning across modes and an A/B isolates shape rather than weight.
    k = LOGIT_CALIB
    at_mean = transform_c(np.array([1 / (1 + np.exp(-k["mu_logit"]))]), "logit")[0]
    assert abs(at_mean - k["mu_raw"]) < 1e-6, at_mean

    # The point of the exercise: differences that the sigmoid compresses to
    # nothing must survive. On the collapsed baseline's observed range the
    # raw spread is ~0.006 and the transform must expand it substantially.
    tight = np.array([0.959, 0.975, 0.990, 0.994])
    gain = transform_c(tight, "logit").std() / tight.std()
    assert gain > 5, f"transform recovered too little spread ({gain:.2f}x)"

    # Saturated inputs must stay finite rather than becoming inf through the
    # log; a single inf would poison the whole group's advantage.
    assert np.all(np.isfinite(transform_c(np.array([0.0, 1.0]), "logit")))

    try:
        transform_c(c, "sigmoid")
        raise AssertionError("unknown transform should have been rejected")
    except ValueError:
        pass

    # And it must reach the reward: same terms, different mode, different R.
    r_raw = assemble(terms, np.full(4, 0.5), w)
    r_log = assemble(terms, np.full(4, 0.5), w, c_transform="logit")
    assert not np.allclose(r_raw, r_log), "c_transform did not reach assemble"
    assert np.array_equal(r_raw[~terms["valid"]], r_log[~terms["valid"]]), \
        "invalid molecules must take the flat penalty under either mode"

    print(f"C(aspirin)={terms['c'][0]:.3f}  QED={terms['qed'][0]:.3f}  "
          f"SA_norm={terms['sa'][0]:.3f}  R={r[0]:.3f}")
    print(f"logit transform: {gain:.1f}x spread recovery on the saturated range")
    print("reward self-check passed")


def _calibrate(dataset: str = "bbbp", device: str = "cpu") -> None:
    """Recompute LOGIT_CALIB against the current ensemble and dataset."""
    import pandas as pd

    from .datasets import DATASETS

    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["path"], sep=cfg["sep"])
    smiles = df[cfg["smiles_col"]].dropna().tolist()
    classify = load_frozen_classifier(dataset=dataset, device=device)
    mols = [m for m in (sanitizable(s) for s in smiles) if m is not None]
    p = np.concatenate([classify(mols[i:i + 256]) for i in range(0, len(mols), 256)])
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(q / (1 - q))
    print(f"n={len(p)}")
    print('LOGIT_CALIB = {"mu_raw": %.4f, "sd_raw": %.4f,' % (p.mean(), p.std()))
    print('               "mu_logit": %.4f, "sd_logit": %.4f}' % (lg.mean(), lg.std()))


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        _calibrate()
    else:
        _self_check()
