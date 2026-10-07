"""VAE-augmented generator pretraining: Eqs. (9)-(13) of the VAE draft.

Replaces the MLE warm start in `generator.pretrain`. The decoder is the
EXISTING `SelfiesGenerator` unchanged -- it already conditions on z through
`z2h` -> initial hidden state, so it is already pi_theta(m | z). What was
missing is that the MLE warm start fed it `z = torch.randn(...)`, an
uninformative draw the decoder could only learn to ignore. This module adds
the encoder q_psi(z | m) that makes z carry information about m.

Nothing downstream changes. The draft is explicit that Sections 4-8 (reward,
group advantage, clipped objective) are untouched, and the code honours that
literally: this writes the same checkpoint format `SelfiesGenerator.load`
already reads, so `train_gan` needs no edit and pi_ref is still pi_theta0.

Two traps the draft names, both implemented and both checked:

  * Trap A, posterior collapse (Sec 1.3). An expressive autoregressive decoder
    can drive q_psi(z|m) -> p(z) for every m, zeroing the KL and falling back
    to the unconditional language model. Countered by KL annealing (Eq. 10)
    and free bits (Eq. 11).
  * Trap B, prior/posterior mismatch (Sec 1.6). Training conditions on
    z ~ q_psi(z|m) but generation conditions on z ~ p(z). `moment_mismatch`
    computes Delta_mom (Eq. 13) -- a leading indicator, checkable before the
    expensive RL phase.

    python -m src.vae_pretrain --train     # pretrain + compare against MLE
    python -m src.vae_pretrain             # self-check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from .datasets import ROOT, _load_raw
from .generator import (CKPT_DIR, EOS, SelfiesGenerator, _tensorize, build_vocab,
                        encode)

RDLogger.DisableLog("rdApp.*")


class Encoder(nn.Module):
    """q_psi(z | m) = N(mu_psi(m), diag(sigma_psi(m)^2)). Eq. (9).

    Bidirectional GRU over the token sequence, mean-pooled over real positions.
    Mean pooling rather than the final hidden state because the sequences are
    right-padded to a common length: the final state of a padded row is the
    state after the padding, which is not a function of the molecule.
    """

    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 256,
                 z_dim: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(2 * hidden, z_dim)
        self.to_logvar = nn.Linear(2 * hidden, z_dim)

    def forward(self, actions: torch.Tensor, mask: torch.Tensor):
        h, _ = self.rnn(self.embed(actions))
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        # logvar is clamped because exp() of a large positive value overflows
        # and of a large negative one underflows sigma to exactly 0, which
        # makes log sigma^2 in Eq. (11) non-finite and poisons the whole batch.
        return self.to_mu(pooled), self.to_logvar(pooled).clamp(-8.0, 8.0)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Eq. (10): z = mu + sigma * eps. Keeps grad_psi defined through z."""
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Eq. (11) summand, per latent dimension, NOT yet summed: (N, dz).

    Kept per-dimension because free bits (Eq. 11) floors each dimension
    separately, and because the count of dimensions above the floor is the
    diagnostic for Trap A.
    """
    return 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)


def free_bits_kl(mu, logvar, lam: float = 0.0) -> torch.Tensor:
    """Eq. (11). Floors each dimension at lam nats so it cannot be zeroed."""
    per_dim = kl_per_dim(mu, logvar)
    if lam > 0.0:
        per_dim = torch.clamp(per_dim, min=lam)
    return per_dim.sum(-1)


def beta_at(k: int, beta_max: float = 1.0, k_anneal: int = 1000,
            schedule: str = "monotonic", k_cyc: int = 500) -> float:
    """Eq. (10). Monotonic ramp or cyclical, mirroring reward.weights_at."""
    if schedule == "monotonic":
        return beta_max * min(k / max(k_anneal, 1), 1.0)
    if schedule == "cyclical":
        frac = (k % max(k_cyc, 1)) / max(k_cyc, 1)
        return beta_max * 0.5 * (1.0 - math.cos(math.pi * frac))
    if schedule == "constant":
        return beta_max
    raise ValueError(f"unknown beta schedule {schedule!r}")


def recon_nll(dec: SelfiesGenerator, actions, mask, z) -> torch.Tensor:
    """Eq. (12)/(16) reconstruction: per-MOLECULE mean over its own T_m tokens.

    The draft normalizes inside the per-molecule sum (1/T_m), not over the
    whole batch. The difference is not cosmetic: a batch-level mean weights a
    long molecule more than a short one, so the ELBO being optimized would not
    be the one written down, and beta would trade against a reconstruction
    term of a different scale.
    """
    lp = dec.log_probs(actions, z)
    m = mask.to(lp.dtype)
    return -((lp * m).sum(-1) / m.sum(-1).clamp(min=1.0))


@torch.no_grad()
def moment_mismatch(enc: Encoder, actions, mask, batch_size: int = 256) -> dict:
    """Eq. (13): Delta_mom for the aggregate posterior. Trap B.

    Sigma_bar follows the law of total variance in Eq. (12): the mean
    within-molecule variance plus the between-molecule covariance of the means.
    The between term is the FULL covariance, not just its diagonal -- an
    aggregate posterior can match the prior on every marginal and still be
    badly correlated, and the Frobenius norm is what would miss it otherwise.
    """
    enc.eval()
    mus, vars_ = [], []
    for i in range(0, len(actions), batch_size):
        mu, logvar = enc(actions[i:i + batch_size], mask[i:i + batch_size])
        mus.append(mu)
        vars_.append(logvar.exp())
    mu_all = torch.cat(mus)
    var_all = torch.cat(vars_)

    mu_bar = mu_all.mean(0)
    within = torch.diag(var_all.mean(0))
    centred = mu_all - mu_bar
    between = centred.T @ centred / max(len(mu_all) - 1, 1)
    sigma_bar = within + between
    eye = torch.eye(sigma_bar.shape[0], device=sigma_bar.device)

    return {
        "delta_mom": float(mu_bar.pow(2).sum() + torch.linalg.norm(sigma_bar - eye)),
        "mu_bar_norm2": float(mu_bar.pow(2).sum()),
        "cov_dev_fro": float(torch.linalg.norm(sigma_bar - eye)),
    }


@torch.no_grad()
def active_units(enc: Encoder, actions, mask, thresh: float = 0.01) -> dict:
    """Trap A readout: how many latent dimensions carry more than `thresh` nats."""
    enc.eval()
    per = []
    for i in range(0, len(actions), 256):
        mu, logvar = enc(actions[i:i + 256], mask[i:i + 256])
        per.append(kl_per_dim(mu, logvar))
    kl = torch.cat(per).mean(0)
    return {"active_units": int((kl > thresh).sum()), "total_units": kl.numel(),
            "kl_total": float(kl.sum())}


def pretrain_vae(
    dataset: str = "bbbp",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_len: int = 72,
    z_dim: int = 64,
    beta_max: float = 0.5,
    k_anneal_epochs: int = 10,
    beta_schedule: str = "monotonic",
    free_bits: float = 0.05,
    device: str = "cpu",
    seed: int = 0,
    out_name: str | None = None,
    log: bool = True,
) -> dict:
    """Eq. (16). Trains encoder and decoder jointly; writes a decoder
    checkpoint in the format `SelfiesGenerator.load` already reads."""
    torch.manual_seed(seed)
    smiles = _load_raw(dataset)["smiles"].dropna().tolist()
    vocab = build_vocab(smiles)
    selfies = [s for s in (encode(x) for x in smiles) if s]

    dec = SelfiesGenerator(vocab, z_dim=z_dim).to(device)
    enc = Encoder(len(vocab), z_dim=z_dim).to(device)
    data = _tensorize(selfies, dec.stoi, max_len).to(device)
    mask = (data != 0).float()
    # The EOS a row ends on is a real decision and must be reconstructed; only
    # what follows it is padding. _tensorize writes EOS then zeros, so the
    # token-nonzero mask already has exactly this shape.
    opt = torch.optim.Adam([*dec.parameters(), *enc.parameters()], lr=lr)

    if log:
        print(f"{dataset}: {len(selfies)} molecules, |V|={len(vocab)}, d_z={z_dim}")

    steps_per_epoch = math.ceil(len(data) / batch_size)
    k = 0
    for ep in range(epochs):
        dec.train()
        enc.train()
        tot_r = tot_k = 0.0
        perm = torch.randperm(len(data), device=device)
        for i in range(0, len(data), batch_size):
            idx = perm[i:i + batch_size]
            a, m = data[idx], mask[idx]
            mu, logvar = enc(a, m)
            z = reparameterize(mu, logvar)
            rec = recon_nll(dec, a, m, z).mean()
            kl = free_bits_kl(mu, logvar, free_bits).mean()
            beta = beta_at(k, beta_max, k_anneal_epochs * steps_per_epoch,
                           beta_schedule)
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*dec.parameters(), *enc.parameters()], 5.0)
            opt.step()
            tot_r += rec.item() * len(idx)
            tot_k += kl.item() * len(idx)
            k += 1

        if log and (ep % 5 == 0 or ep == epochs - 1):
            probe = dec.sample(200, max_len=max_len, device=device)["smiles"]
            valid = sum(bool(s) and Chem.MolFromSmiles(s) is not None for s in probe)
            au = active_units(enc, data, mask)
            print(f"  epoch {ep + 1:>2}  recon={tot_r / len(data):.4f}  "
                  f"kl={tot_k / len(data):.3f}  beta={beta:.3f}  "
                  f"valid={valid / 200:.1%}  active={au['active_units']}/{au['total_units']}")

    diag = {**moment_mismatch(enc, data, mask), **active_units(enc, data, mask)}
    path = CKPT_DIR / (out_name or f"{dataset}_vae_pretrained.pt")
    dec.save(path)
    torch.save({"state": enc.state_dict(), "z_dim": z_dim, "vocab": vocab},
               path.with_name(path.stem + "_encoder.pt"))
    if log:
        print(f"  Delta_mom={diag['delta_mom']:.3f}  "
              f"(||mu_bar||^2={diag['mu_bar_norm2']:.3f}, "
              f"||Sigma_bar-I||_F={diag['cov_dev_fro']:.3f})")
        print(f"saved {path}")
    return {"decoder": dec, "encoder": enc, "diag": diag, "path": path}


def _self_check() -> None:
    torch.manual_seed(0)
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "CCN", "C1CCCCC1", "CCOCC"]
    vocab = build_vocab(smis)
    dec = SelfiesGenerator(vocab, emb=16, hidden=32, z_dim=8)
    enc = Encoder(len(vocab), emb=16, hidden=32, z_dim=8)
    sel = [s for s in (encode(x) for x in smis) if s]
    a = _tensorize(sel, dec.stoi, 16)
    m = (a != 0).float()

    # Eq. (11) closed form: a standard-normal posterior must give exactly 0.
    zero = kl_per_dim(torch.zeros(4, 8), torch.zeros(4, 8))
    assert torch.allclose(zero, torch.zeros_like(zero), atol=1e-7), zero
    # and KL must be non-negative everywhere else.
    rnd = kl_per_dim(torch.randn(64, 8), torch.randn(64, 8) * 0.5)
    assert (rnd >= -1e-6).all(), "closed-form KL went negative"

    # Free bits floors each dim independently and only ever raises the total.
    mu, lv = torch.zeros(4, 8), torch.zeros(4, 8)
    assert torch.allclose(free_bits_kl(mu, lv, 0.0), torch.zeros(4), atol=1e-7)
    assert torch.allclose(free_bits_kl(mu, lv, 0.1), torch.full((4,), 0.8), atol=1e-6), \
        "free bits must floor every one of the 8 dims at 0.1"

    # Eq. (10) schedules.
    assert beta_at(0, 1.0, 100) == 0.0
    assert beta_at(100, 1.0, 100) == 1.0
    assert beta_at(999, 1.0, 100) == 1.0, "monotonic ramp must clamp"
    assert abs(beta_at(250, 1.0, schedule="cyclical", k_cyc=500) - 0.5) < 1e-6

    # Reparameterization must carry gradient into psi (the whole point of
    # Eq. 10 -- sampling z directly would detach the encoder).
    mu, logvar = enc(a, m)
    z = reparameterize(mu, logvar)
    assert z.requires_grad
    (recon_nll(dec, a, m, z).mean() + free_bits_kl(mu, logvar, 0.05).mean()).backward()
    assert enc.to_mu.weight.grad is not None and enc.to_mu.weight.grad.abs().sum() > 0, \
        "no gradient reached the encoder -- reparameterization is broken"
    assert dec.z2h.weight.grad is not None and dec.z2h.weight.grad.abs().sum() > 0, \
        "no gradient reached the decoder's z pathway"

    # Reconstruction is per-molecule normalized (Eq. 16), so it must be
    # invariant to how much padding a row carries.
    z0 = torch.randn(len(a), 8)
    wide = torch.cat([a, torch.zeros(len(a), 6, dtype=torch.long)], 1)
    wm = (wide != 0).float()
    assert torch.allclose(recon_nll(dec, a, m, z0),
                          recon_nll(dec, wide, wm, z0), atol=1e-5), \
        "padding changed the per-molecule reconstruction term"

    # Trap B: a perfectly matched aggregate posterior scores Delta_mom ~ 0.
    class Ideal(Encoder):
        def forward(self, x, msk):
            n = len(x)
            return torch.zeros(n, 8), torch.zeros(n, 8)

    ideal = Ideal(len(vocab), emb=16, hidden=32, z_dim=8)
    big = torch.randint(1, len(vocab), (512, 16))
    d_ideal = moment_mismatch(ideal, big, torch.ones_like(big).float())
    assert d_ideal["delta_mom"] < 0.2, d_ideal

    # and a shifted/inflated one must score clearly worse -- the diagnostic has
    # to be able to FAIL, or logging it says nothing.
    class Bad(Encoder):
        def forward(self, x, msk):
            n = len(x)
            return torch.full((n, 8), 2.0), torch.full((n, 8), 1.0)

    bad = moment_mismatch(Bad(len(vocab), emb=16, hidden=32, z_dim=8),
                          big, torch.ones_like(big).float())
    assert bad["delta_mom"] > d_ideal["delta_mom"] + 1.0, (bad, d_ideal)

    # Trap A: a collapsed posterior must report ~0 active units.
    au = active_units(ideal, big, torch.ones_like(big).float())
    assert au["active_units"] == 0, au

    # The decoder written out must still load as a plain SelfiesGenerator, or
    # train_gan cannot consume it and the draft's "Sections 4-8 unchanged"
    # claim is false in the code.
    tmp = CKPT_DIR / "_vae_selfcheck.pt"
    dec.save(tmp)
    back = SelfiesGenerator.load(tmp)
    assert torch.allclose(back.log_probs(a, z0), dec.log_probs(a, z0), atol=1e-6)
    tmp.unlink()

    print(f"Delta_mom ideal={d_ideal['delta_mom']:.3f} bad={bad['delta_mom']:.3f}")
    print("vae_pretrain self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--beta-max", type=float, default=0.5)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--beta-schedule", default="monotonic",
                    choices=["monotonic", "cyclical", "constant"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default=None,
                    help="checkpoint filename under results/generator/")
    args = ap.parse_args()
    if args.train:
        pretrain_vae(dataset=args.dataset, epochs=args.epochs, z_dim=args.z_dim,
                     beta_max=args.beta_max, free_bits=args.free_bits,
                     beta_schedule=args.beta_schedule, device=args.device,
                     seed=args.seed, out_name=args.out_name)
    else:
        _self_check()
