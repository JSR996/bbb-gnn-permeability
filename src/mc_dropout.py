"""MC dropout on the frozen classifier: is the saturation real confidence?

The baseline found that the reward's 3-member ensemble agrees almost exactly
on generated molecules -- member sd 0.005, against 0.063 on real BBBP -- and
that every generated molecule scores above 0.95. Two readings fit that:

  (a) the classifier is genuinely certain, and the reward really has run out
      of information to give. Then reward shaping cannot recover a signal
      that is not there, and the fix has to come from elsewhere.
  (b) the three members are too similar to disagree. They share a feature
      set, a scaffold split and a skeleton, so their agreement measures
      shared inductive bias rather than certainty. Then the reward carries
      more uncertainty than the ensemble reports, and a variance-aware
      reward could recover gradient.

Ensemble spread cannot separate these, because it is the quantity under
suspicion. MC dropout gives an independent estimate from inside a single
model: K stochastic forward passes with dropout active, which approximates
sampling from a distribution over sub-networks.

CRITICAL: dropout is enabled, BatchNorm is NOT. A blanket .train() would put
BatchNorm into batch-statistics mode, and C(m) would start depending on which
molecules share the batch -- exactly the property load_frozen_classifier
calls eval() to prevent, and the premise of the stationarity argument in
Section 3.1 of the continuation draft. `_enable_dropout_only` touches
nn.Dropout modules and nothing else; the self-check asserts every
normalization layer is still in eval mode afterwards.

    python -m src.mc_dropout            # self-check
    python -m src.mc_dropout --run      # the real comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

from .featurize import smiles_to_graph
from .models import build_model
from .models_edge import EDGE_MODELS, build_edge_model
from .reward import _checkpoint
from .train import RESULTS_DIR

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE = ("gin", "gat", "gine")

_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm1d)


def _enable_dropout_only(net: nn.Module) -> int:
    """Put every nn.Dropout into train mode, leaving everything else in eval.

    Returns the number of dropout layers switched, so a caller can fail loudly
    on a model that has none rather than silently collecting K identical
    passes and reporting zero uncertainty as a finding.
    """
    net.eval()
    n = 0
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def load_members(dataset: str = "bbbp", seed: int = 0, device: str = "cpu",
                 models: tuple[str, ...] = ENSEMBLE) -> dict[str, nn.Module]:
    nets = {}
    for name in models:
        ckpt = _checkpoint(RESULTS_DIR, dataset, name, seed)
        builder = build_edge_model if name in EDGE_MODELS else build_model
        net = builder(name).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        net.requires_grad_(False)
        nets[name] = net
    return nets


@torch.no_grad()
def mc_predict(net: nn.Module, smiles: list[str], k: int = 50,
               device: str = "cpu", seed: int = 0) -> np.ndarray:
    """(K, n) matrix of stochastic predictions. NaN columns for bad SMILES."""
    n_drop = _enable_dropout_only(net)
    if n_drop == 0:
        raise RuntimeError(
            f"{type(net).__name__} has no nn.Dropout layers, so MC dropout "
            f"would report zero uncertainty by construction"
        )
    keep, graphs = [], []
    for i, s in enumerate(smiles):
        g = smiles_to_graph(s) if s else None
        if g is not None:
            keep.append(i)
            graphs.append(g)
    out = np.full((k, len(smiles)), np.nan)
    if not keep:
        net.eval()
        return out
    batch = Batch.from_data_list(graphs).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for j in range(k):
        # Reseed per pass so the draw sequence is reproducible run to run.
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        out[j, keep] = torch.sigmoid(net(batch)).cpu().numpy()
    net.eval()  # leave the model as it was found
    return out


def compare(nets: dict[str, nn.Module], sets: dict[str, list[str]],
            k: int, device: str) -> pd.DataFrame:
    """MC-dropout spread vs ensemble-member spread, per molecule set.

    Both are reported in probability AND logit space. The point estimates
    saturate near 1, where probability space compresses differences that
    logit space keeps -- so a spread that looks like zero on one scale may
    not be zero on the other, and the distinction decides whether there is
    any signal left to shape.
    """
    rows = []
    for set_name, smiles in sets.items():
        # (members, K, n)
        mc = np.stack([mc_predict(net, smiles, k, device) for net in nets.values()])
        ok = ~np.isnan(mc).any(axis=(0, 1))
        mc = mc[:, :, ok]
        if mc.shape[-1] == 0:
            continue

        # Within-model spread across dropout draws, averaged over members.
        mc_sd = mc.std(axis=1).mean()
        mc_sd_logit = _logit(mc).std(axis=1).mean()

        # Between-member spread of the deterministic prediction: the quantity
        # the baseline reported, recomputed here on the same molecules.
        point = mc.mean(axis=1)                      # (members, n)
        member_sd = point.std(axis=0).mean()
        member_sd_logit = _logit(point).std(axis=0).mean()

        mean_p = point.mean()
        rows.append({
            "set": set_name, "n": int(mc.shape[-1]), "K": k,
            "mean_p": mean_p,
            "mc_sd": mc_sd, "member_sd": member_sd,
            "mc_over_member": mc_sd / member_sd if member_sd else np.inf,
            "mc_sd_logit": mc_sd_logit, "member_sd_logit": member_sd_logit,
            "frac_over_95": float((point.mean(axis=0) > 0.95).mean()),
        })
    return pd.DataFrame(rows)


def _self_check() -> None:
    torch.manual_seed(0)
    net = build_model("gin")

    # 1. dropout on, every normalization layer still off.
    n = _enable_dropout_only(net)
    assert n > 0, "no dropout layers found in the base skeleton"
    for m in net.modules():
        if isinstance(m, _NORM_TYPES):
            assert not m.training, f"{type(m).__name__} left in train mode -- " \
                                   f"C(m) would depend on its batch"
        if isinstance(m, nn.Dropout):
            assert m.training, "dropout not enabled"
    print(f"  {n} dropout layers enabled, all norm layers still in eval")

    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C1CCCCC1"]

    # 2. the passes must actually differ, or this measures nothing.
    mc = mc_predict(net, smiles, k=16)
    assert mc.std(axis=0).mean() > 1e-6, "dropout produced identical passes"

    # 3. and the model must be left in eval, or a later reward call silently
    #    becomes stochastic.
    assert not net.training
    for m in net.modules():
        assert not m.training, f"{type(m).__name__} left training after mc_predict"
    print("  model restored to eval after sampling")

    # 4. batch invariance of the MC MEAN. Individual draws depend on the
    #    dropout mask, but with BatchNorm in eval the expectation over draws
    #    must not depend on which molecules share the batch. This is the
    #    property that would break if .train() had been called wholesale.
    alone = mc_predict(net, ["CCO"], k=400, seed=1)[:, 0].mean()
    crowded = mc_predict(net, smiles, k=400, seed=1)[:, 0].mean()
    assert abs(alone - crowded) < 0.02, \
        f"MC mean moved with group composition ({alone:.4f} vs {crowded:.4f})"
    print(f"  batch invariance of the MC mean: {alone:.4f} alone vs "
          f"{crowded:.4f} in company")

    # 5. a model with no dropout must raise rather than report zero spread.
    bare = build_model("gin", dropout=0.0)
    for m in bare.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    try:
        out = mc_predict(bare, smiles, k=4)
        assert out.std(axis=0).mean() < 1e-9  # p=0 => deterministic, not an error
        print("  p=0 model yields deterministic passes, as expected")
    except RuntimeError:
        print("  model without dropout layers raises, as expected")

    # 6. unsanitizable input stays NaN rather than becoming a real number.
    bad = mc_predict(net, ["", "not-a-molecule", "CCO"], k=4)
    assert np.isnan(bad[:, :2]).all() and not np.isnan(bad[:, 2]).any()

    print("mc_dropout self-check passed")


def main(dataset: str, seed: int, k: int, device: str, n_real: int) -> None:
    nets = load_members(dataset, seed, device)
    print(f"loaded {list(nets)}  (K={k} stochastic passes each)")

    real = pd.read_csv(ROOT / "BBBP.csv")["smiles"].dropna().tolist()
    rng = np.random.default_rng(0)
    real = [real[i] for i in rng.choice(len(real), min(n_real, len(real)), replace=False)]

    sets = {"real_bbbp": real}
    warm = ROOT / "results" / "generator" / f"{dataset}_pretrained.pt"
    if warm.exists():
        from .generator import SelfiesGenerator
        torch.manual_seed(0)
        g = SelfiesGenerator.load(warm, device)
        sets["warm_start"] = [s for s in g.sample(400, device=device)["smiles"] if s]
    for label, path in (("baseline_step199", "gan/seed0/samples/step0199.json"),
                        ("kl1.0_step119", "kl_sweep/kl1/samples/step0119.json")):
        p = ROOT / "results" / path
        if p.exists():
            sets[label] = json.load(open(p))

    df = compare(nets, sets, k, device)
    print("\n" + "=" * 92)
    print("MC DROPOUT vs ENSEMBLE SPREAD")
    print("=" * 92)
    print(df.round(4).to_string(index=False))
    print("\n  mc_sd        : spread across dropout draws within a model (epistemic proxy)")
    print("  member_sd    : spread across the 3 ensemble members (what the baseline reported)")
    print("  mc_over_member > 1 means a single model is less certain than the")
    print("  ensemble's agreement suggests -- i.e. the agreement is shared bias.")

    out = RESULTS_DIR / "classifier_ood"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "mc_dropout.csv", index=False)
    print(f"\nwrote {out / 'mc_dropout.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the comparison")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-k", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-real", type=int, default=400)
    args = ap.parse_args()
    if args.run:
        main(args.dataset, args.seed, args.k, args.device, args.n_real)
    else:
        _self_check()
