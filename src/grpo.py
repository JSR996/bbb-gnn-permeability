"""GRPO update: group-relative advantage (Eq. 6) and clipped surrogate (Eq. 7).

Both cases of Section 1 run through this same code. Case A (autoregressive
SELFIES) passes log-probs of shape (N, T); Case B (one-shot atom/bond tensor)
is the degenerate T = 1 of the same call, with Eq. (4)'s factorized joint
summed into a single column. Nothing here needs to know which generator
produced the numbers, so Item 1's fork stays confined to the generator.

    python -m src.grpo          # self-check
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 6, normalized within each group. Accepts (N,) or (G, N).

    A group whose rewards are all equal carries no ranking information -- which
    is the common early-training case where every molecule fails sanitization
    and collects the same flat penalty. Its advantages are forced to exactly
    zero rather than 0/eps, so a degenerate group contributes no gradient
    instead of amplifying float noise into a spurious one.
    """
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    if rewards.shape[-1] < 2:
        raise ValueError(f"group size must be >= 2, got {rewards.shape[-1]}")

    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    return torch.where(std > eps, adv, torch.zeros_like(adv)).squeeze(0)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    total = mask.sum()
    return (x * mask).sum() / total.clamp(min=1.0)


def clipped_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    logp_ref: torch.Tensor | None = None,
    kl_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq. 7 plus the optional reference-policy KL penalty from the diagram.

    logp/logp_old/logp_ref are (N, T) -- or (N,) for Case B, promoted to (N, 1).
    advantages are per-molecule (N,) and broadcast across t, since every token
    of a molecule shares that molecule's group-relative advantage.

    logp_old must come from a detached snapshot taken at sampling time: it is
    pi_theta_old in Eq. 2, and if it carries grad the ratio differentiates
    through its own denominator and the update is no longer PPO.
    """
    if logp.ndim == 1:  # Case B: one joint action per molecule
        logp, logp_old = logp.unsqueeze(-1), logp_old.unsqueeze(-1)
        if logp_ref is not None:
            logp_ref = logp_ref.unsqueeze(-1)
    if mask is not None:
        mask = mask.float()

    adv = advantages.reshape(-1, 1)
    # Mask BEFORE the exp, not after. Padded positions hold arbitrary garbage,
    # so exp() overflows to inf there and inf * 0 is nan -- multiplying by the
    # mask downstream cannot rescue it. Zeroing the difference first pins the
    # padded ratio at exactly 1, which is finite and then masked out cleanly.
    diff = logp - logp_old.detach()
    if mask is not None:
        diff = diff * mask
    ratio = torch.exp(diff)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.min(unclipped, clipped)

    # ponytail: token-level mean, so a long molecule does not outweigh a short
    # one. The draft writes Eq. 7 as a sum over t, which is sequence-level and
    # does carry that length bias -- switch _masked_mean to a per-row sum if
    # you want the draft's exact form.
    loss = -_masked_mean(surrogate, mask)

    stats = {
        "ratio_mean": _masked_mean(ratio, mask).item(),
        "clip_frac": _masked_mean((ratio - 1.0).abs().gt(clip_eps).float(), mask).item(),
        "adv_std": advantages.std().item(),
    }

    if logp_ref is not None and kl_coef > 0.0:
        # k3 estimator: unbiased, low variance, and non-negative by
        # construction, unlike the raw (logp_ref - logp) difference.
        d = logp_ref.detach() - logp
        if mask is not None:
            d = d * mask  # same overflow trap as the ratio above
        kl = torch.exp(d) - d - 1.0
        kl_term = _masked_mean(kl, mask)
        loss = loss + kl_coef * kl_term
        stats["kl"] = kl_term.item()

    return loss, stats


def _self_check() -> None:
    torch.manual_seed(0)

    # Degenerate group: every molecule invalid, so every reward is the penalty.
    flat = group_advantages(torch.zeros(8))
    assert torch.isfinite(flat).all(), "all-equal group produced non-finite advantage"
    assert (flat == 0).all(), f"expected exactly zero, got {flat}"

    adv = group_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.isfinite(adv).all()
    assert abs(adv.mean().item()) < 1e-6, "advantages must be centered"
    assert adv[0] < 0 < adv[-1], "ranking must survive normalization"

    batched = group_advantages(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert batched.shape == (2, 3)
    assert (batched[1] == 0).all(), "degenerate row must not poison the batch"

    try:
        group_advantages(torch.tensor([1.0]))
        raise AssertionError("group of 1 should have been rejected")
    except ValueError:
        pass

    # No policy movement -> ratio 1 -> loss is exactly -mean(advantage).
    logp = torch.randn(4, 6)
    loss, stats = clipped_loss(logp, logp.clone(), adv)
    assert abs(stats["ratio_mean"] - 1.0) < 1e-6
    assert abs(loss.item() + adv.mean().item()) < 1e-6

    # Clipping must bite once the ratio runs away from a positive advantage.
    far = logp + 3.0
    _, stats_far = clipped_loss(far, logp, adv)
    assert stats_far["clip_frac"] == 1.0, stats_far

    # Case B: (N,) log-probs must behave as the T=1 case of the same call.
    a = group_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    flat_lp, old_lp = torch.randn(4), torch.randn(4)
    lb, _ = clipped_loss(flat_lp, old_lp, a)
    lt, _ = clipped_loss(flat_lp.unsqueeze(-1), old_lp.unsqueeze(-1), a)
    assert torch.allclose(lb, lt), "Case B must equal Case A with T=1"

    # Masked padding must not contribute; padded batch == unpadded batch.
    lp3, old3 = torch.randn(4, 3), torch.randn(4, 3)
    ref = clipped_loss(lp3, old3, adv)[0]
    lp5 = torch.cat([lp3, torch.randn(4, 2) * 50], dim=1)
    old5 = torch.cat([old3, torch.randn(4, 2) * 50], dim=1)
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 4)
    assert torch.allclose(ref, clipped_loss(lp5, old5, adv, mask=mask)[0], atol=1e-5), \
        "padding leaked into the loss"

    # KL penalty is non-negative and vanishes when the policies agree.
    _, s_same = clipped_loss(logp, logp, adv, logp_ref=logp, kl_coef=1.0)
    assert abs(s_same["kl"]) < 1e-6, s_same
    _, s_diff = clipped_loss(logp, logp, adv, logp_ref=logp + 0.5, kl_coef=1.0)
    assert s_diff["kl"] > 0, s_diff

    # logp_old arriving with grad must not backpropagate through the ratio.
    live = torch.randn(4, 6, requires_grad=True)
    clipped_loss(logp.requires_grad_(True), live, adv)[0].backward()
    assert live.grad is None, "gradient leaked into pi_theta_old"

    print(f"loss={loss.item():.4f}  ratio={stats['ratio_mean']:.3f}  "
          f"clip_frac={stats['clip_frac']:.3f}")
    print("grpo self-check passed")


if __name__ == "__main__":
    _self_check()
