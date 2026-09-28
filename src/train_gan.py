"""Outer GAN+GRPO loop: sample -> reward -> advantage -> clipped update.

Closes the diagram. The three pieces that already existed (`reward.py`,
`grpo.py`, `generator.py`) are wired together here, plus D_phi, which is the
same GNN skeleton the permeability classifier uses -- real-vs-fake is the same
graph->logit problem, so it reuses `build_model` rather than introducing a
second architecture.

Two ordering constraints from Section 3 of the continuation draft are structural
here rather than advisory:

  * D_phi is scored in eval() (Section 3.1's stationarity argument applied to D:
    with BatchNorm in train mode, D(m) would depend on which other candidates
    shared the group, so the reward would stop being a function of the molecule).
  * the n_D discriminator steps run strictly AFTER the K-epoch PPO block, never
    interleaved (Section 3.4), so every advantage in a group is computed against
    one fixed phi snapshot.

    python -m src.train_gan --steps 200     # train
    python -m src.train_gan                 # self-check (tiny, no checkpoints)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

from .datasets import ROOT, _load_raw
from .featurize import smiles_to_graph
from .generator import CKPT_DIR, SelfiesGenerator
from .grpo import clipped_loss, group_advantages
from .models import build_model
from .reward import assemble, load_frozen_classifier, score_terms, weights_at

RDLogger.DisableLog("rdApp.*")

OUT_DIR = ROOT / "results" / "gan"

W0 = {"D": 1.0, "C": 0.2, "Q": 0.5, "S": 0.5}   # validity/realism first
WF = {"D": 0.3, "C": 1.5, "Q": 0.5, "S": 0.5}   # then permeability


def _graphs(smiles: list[str]) -> Batch | None:
    """Featurize the sanitizable subset. None if nothing survives."""
    graphs = [g for g in (smiles_to_graph(s) for s in smiles if s) if g is not None]
    return Batch.from_data_list(graphs) if graphs else None


class Discriminator(nn.Module):
    """D_phi: real-vs-fake over molecular graphs. Same skeleton as the classifier."""

    def __init__(self, conv: str = "gin", **kwargs) -> None:
        super().__init__()
        self.net = build_model(conv, **kwargs)

    def forward(self, batch) -> torch.Tensor:
        return self.net(batch)  # logit per graph

    @torch.no_grad()
    def score(self, smiles: list[str], device: str = "cpu") -> np.ndarray:
        """D(m) in [0,1] per input SMILES, 0 for anything unsanitizable.

        eval() is load-bearing for the same reason it is on the frozen
        classifier: BatchNorm in train mode makes D(m) a function of the group
        rather than of the molecule.
        """
        was_training = self.training
        self.eval()
        out = np.zeros(len(smiles))
        keep = [i for i, s in enumerate(smiles) if s and smiles_to_graph(s) is not None]
        if keep:
            batch = _graphs([smiles[i] for i in keep]).to(device)
            out[keep] = torch.sigmoid(self(batch)).cpu().numpy()
        self.train(was_training)
        return out


def _d_step(disc, opt, real_smiles, fake_smiles, device) -> float:
    """One non-saturating GAN discriminator step. Returns BCE, or nan if skipped.

    A step is skipped rather than faked when either side is empty: early in
    training whole groups fail sanitization, and training D on reals alone
    would teach it to output 1 unconditionally.
    """
    real, fake = _graphs(real_smiles), _graphs(fake_smiles)
    if real is None or fake is None:
        return float("nan")
    disc.train()
    logits = disc(Batch.from_data_list(real.to_data_list() + fake.to_data_list()).to(device))
    target = torch.cat([torch.ones(real.num_graphs), torch.zeros(fake.num_graphs)]).to(device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train(
    dataset: str = "bbbp",
    steps: int = 200,
    group_size: int = 64,
    ppo_epochs: int = 4,
    n_d: int = 2,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    invalid_floor: float = -2.0,
    schedule: str = "linear",
    k_anneal: int = 100,
    max_len: int = 72,
    device: str = "cpu",
    out_dir: Path = OUT_DIR,
    log: bool = True,
) -> dict:
    warm = CKPT_DIR / f"{dataset}_pretrained.pt"
    if not warm.exists():
        raise FileNotFoundError(
            f"no warm start at {warm}. GRPO cannot bootstrap from a uniform "
            f"policy -- every group would be all-invalid and carry zero "
            f"advantage. Run it first: "
            f"python -m src.generator --pretrain --dataset {dataset}"
        )
    gen = SelfiesGenerator.load(warm, device)
    # Reference policy for the KL term in the diagram: the pretrained generator,
    # frozen. It is what keeps GRPO from drifting off the space of molecules the
    # MLE warm start actually learned while chasing reward.
    ref = SelfiesGenerator.load(warm, device)
    ref.eval().requires_grad_(False)

    disc = Discriminator().to(device)
    classify = load_frozen_classifier(dataset=dataset, device=device)
    real_pool = [s for s in _load_raw(dataset)["smiles"].dropna().tolist()]

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d)
    rng = np.random.default_rng(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    history, valid_history = [], []

    for k in range(steps):
        # --- 1. sample a group; pi_theta_old is snapshotted here, once -------
        roll = gen.sample(group_size, max_len=max_len, device=device)

        # --- 2. reward against a single frozen phi snapshot ------------------
        terms = score_terms(roll["smiles"], classify)
        d_scores = disc.score(roll["smiles"], device=device)
        valid_history.append(float(terms["valid"].mean()))
        w = weights_at(k, W0, WF, schedule=schedule, k_anneal=k_anneal,
                       valid_history=valid_history)
        rewards = assemble(terms, d_scores, w)
        # Hurdle form: the reward from `assemble` is zero-inflated by
        # construction, so the invalid molecules must not set the scale the
        # valid ones are ranked on (see group_advantages).
        adv = group_advantages(
            torch.as_tensor(rewards, dtype=torch.float, device=device),
            valid=torch.as_tensor(terms["valid"], device=device),
            floor=invalid_floor,
        )

        # --- 3. K PPO epochs, phi held fixed throughout (Section 3.4) --------
        gen.train()
        stats = {}
        for _ in range(ppo_epochs):
            logp = gen.log_probs(roll["actions"], roll["z"])
            with torch.no_grad():
                logp_ref = ref.log_probs(roll["actions"], roll["z"])
            loss, stats = clipped_loss(
                logp, roll["logp_old"], adv, mask=roll["mask"],
                clip_eps=clip_eps, logp_ref=logp_ref, kl_coef=kl_coef,
            )
            opt_g.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_g.step()

        # --- 4. n_D discriminator steps, strictly outside the PPO window -----
        d_loss = float("nan")
        for _ in range(n_d):
            real = [real_pool[i] for i in rng.choice(len(real_pool), group_size)]
            d_loss = _d_step(disc, opt_d, real, roll["smiles"], device)

        row = {
            "step": k,
            "reward_mean": float(rewards.mean()),
            "valid_frac": valid_history[-1],
            "c_mean": float(terms["c"][terms["valid"]].mean()) if terms["valid"].any() else 0.0,
            "qed_mean": float(terms["qed"][terms["valid"]].mean()) if terms["valid"].any() else 0.0,
            # Section 3.3 diagnostic: as D sharpens, the variance of its term
            # drifts relative to the fixed-variance C term, changing their
            # effective weight on A_i even with w_D, w_C literally constant.
            "d_var": float(d_scores.var()),
            "d_loss": d_loss,
            "w_C": w["C"],
            "unique": len({s for s in roll["smiles"] if s}) / group_size,
            **stats,
        }
        history.append(row)
        if log and (k % 10 == 0 or k == steps - 1):
            print(f"step {k:>4}  R={row['reward_mean']:+.3f}  valid={row['valid_frac']:.0%}  "
                  f"C={row['c_mean']:.3f}  D_var={row['d_var']:.4f}  "
                  f"clip={row['clip_frac']:.2f}  kl={row.get('kl', 0):.4f}")

    if log:
        gen.save(out_dir / f"{dataset}_grpo.pt")
        torch.save(disc.state_dict(), out_dir / f"{dataset}_disc.pt")
        with open(out_dir / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        sample = [s for s in gen.sample(200, device=device)["smiles"]
                  if s and Chem.MolFromSmiles(s) is not None]
        (out_dir / "samples.json").write_text(json.dumps(sample, indent=1))
        print(f"\nwrote {out_dir}  ({len(sample)}/200 valid samples)")

    return {"history": history, "generator": gen, "discriminator": disc}


def _self_check() -> None:
    """End-to-end on a toy generator, with a stub classifier -- no checkpoints.

    The point is the wiring and the ordering constraints, not the chemistry:
    that the reward reaches the advantage, that the advantage moves theta, and
    that phi cannot move during the PPO window.
    """
    torch.manual_seed(0)
    from .generator import build_vocab

    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "CCN", "C1CCCCC1", "CCOCC"]
    gen = SelfiesGenerator(build_vocab(smiles), emb=16, hidden=32, z_dim=8)
    disc = Discriminator(hidden=16, num_layers=2)

    # D must be batch-invariant, or the reward is a function of the group.
    alone = disc.score(["CCO"])
    crowded = disc.score(["CCO", "c1ccccc1", "CCN"])
    assert np.allclose(alone[0], crowded[0], atol=1e-6), \
        f"D(m) moved with group composition ({alone[0]} vs {crowded[0]})"

    # Unsanitizable and empty strings score 0 rather than crashing featurization.
    assert disc.score(["", "not-a-molecule", "CCO"]).tolist()[:2] == [0.0, 0.0]

    # An empty fake side must skip the step, not train D on reals alone.
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    before = [p.clone() for p in disc.parameters()]
    assert np.isnan(_d_step(disc, opt_d, smiles, ["", ""], "cpu"))
    assert all(torch.equal(a, b) for a, b in zip(before, disc.parameters())), \
        "D moved on a skipped step"
    assert not np.isnan(_d_step(disc, opt_d, smiles, smiles, "cpu"))

    # One full GRPO step on a stub reward: theta must move, phi must not.
    roll = gen.sample(8, max_len=10)
    terms = score_terms(roll["smiles"], lambda mols: np.full(len(mols), 0.5))
    rewards = assemble(terms, disc.score(roll["smiles"]), W0)
    assert (rewards[~terms["valid"]] == 0).all(), "invalid molecules collected reward"

    adv = group_advantages(
        torch.as_tensor(rewards, dtype=torch.float),
        valid=torch.as_tensor(terms["valid"]),
    )
    # The zero spike must not set the scale the valid molecules are ranked on.
    if terms["valid"].sum() >= 2 and not terms["valid"].all():
        v = torch.as_tensor(terms["valid"])
        assert abs(adv[v].std().item() - 1.0) < 1e-3, \
            f"invalid molecules leaked into the valid subgroup's scale ({adv[v].std():.3f})"
        assert (adv[~v] == -2.0).all(), "invalid molecules did not take the floor"
    phi_before = [p.clone() for p in disc.parameters()]
    theta_before = gen.out.weight.clone()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-2)
    loss, stats = clipped_loss(
        gen.log_probs(roll["actions"], roll["z"]), roll["logp_old"], adv,
        mask=roll["mask"], logp_ref=roll["logp_old"], kl_coef=0.02,
    )
    opt_g.zero_grad()
    loss.backward()
    opt_g.step()

    assert all(torch.equal(a, b) for a, b in zip(phi_before, disc.parameters())), \
        "phi moved inside the PPO window -- advantages are stale by the end of it"
    if adv.abs().sum() > 0:
        assert not torch.equal(theta_before, gen.out.weight), "theta did not move"
    assert abs(stats["ratio_mean"] - 1.0) < 1e-4, \
        f"ratio != 1 on the first epoch ({stats['ratio_mean']}) -- logp_old is not pi_theta_old"

    print(f"valid={terms['valid'].mean():.0%}  R={rewards.mean():+.3f}  "
          f"adv_std={stats['adv_std']:.3f}  ratio={stats['ratio_mean']:.4f}")
    print("train_gan self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="0 runs the self-check")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--n-d", type=int, default=2)
    ap.add_argument("--schedule", default="linear",
                    choices=["fixed", "linear", "cosine", "gated"])
    ap.add_argument("--k-anneal", type=int, default=100)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--invalid-floor", type=float, default=-2.0,
                    help="advantage handed to unsanitizable molecules")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.steps:
        train(dataset=args.dataset, steps=args.steps, group_size=args.group_size,
              ppo_epochs=args.ppo_epochs, n_d=args.n_d, schedule=args.schedule,
              k_anneal=args.k_anneal, kl_coef=args.kl_coef,
              invalid_floor=args.invalid_floor, device=args.device)
    else:
        _self_check()
