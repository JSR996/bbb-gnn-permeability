"""Case A generator: autoregressive SELFIES policy pi_theta(a_t | a_<t, z).

Item 1 of the continuation draft forks between autoregressive SELFIES (Case A)
and a one-shot atom/bond tensor (Case B). This is Case A. The fork stays
confined here on purpose -- `grpo.py` already accepts (N, T) or (N,) log-probs,
so switching to Case B would replace this file and nothing else.

Eq. 1 is `logits()`; Eq. 2 is `log_probs()`. Sampling and scoring are split
because PPO needs both: pi_theta_old is a detached snapshot taken once at
sampling time, while pi_theta is recomputed each epoch over the SAME stored
actions. Re-sampling instead of re-scoring would silently pin the ratio at 1.

    python -m src.generator --pretrain   # MLE warm start on real molecules
    python -m src.generator              # self-check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import selfies as sf
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from .datasets import ROOT, _load_raw

RDLogger.DisableLog("rdApp.*")

# "[nop]" is SELFIES' own no-op token: a padded string still decodes, so PAD
# needs no special casing in the decoder.
PAD, BOS, EOS = "[nop]", "[BOS]", "[EOS]"
SPECIALS = [PAD, BOS, EOS]

CKPT_DIR = ROOT / "results" / "generator"


def encode(smiles: str) -> str | None:
    """SMILES -> SELFIES, or None if RDKit/SELFIES cannot represent it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return sf.encoder(Chem.MolToSmiles(mol))
    except sf.EncoderError:
        return None


def build_vocab(smiles: list[str]) -> list[str]:
    """Token list with the three specials pinned to indices 0, 1, 2."""
    selfies = [s for s in (encode(x) for x in smiles) if s]
    alphabet = sf.get_alphabet_from_selfies(selfies)
    return SPECIALS + sorted(alphabet - set(SPECIALS))


class SelfiesGenerator(nn.Module):
    """GRU over SELFIES tokens, conditioned on a latent z via the initial state."""

    def __init__(
        self,
        vocab: list[str],
        emb: int = 128,
        hidden: int = 256,
        layers: int = 1,
        z_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.z_dim, self.hidden, self.layers, self.emb = z_dim, hidden, layers, emb
        self.embed = nn.Embedding(len(vocab), emb, padding_idx=0)
        self.z2h = nn.Linear(z_dim, layers * hidden)
        self.rnn = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, len(vocab))

    def _h0(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.z2h(z)).view(-1, self.layers, self.hidden).transpose(0, 1)

    def logits(self, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Eq. 1, teacher-forced: (N, T) actions -> (N, T, V) logits.

        Input is [BOS, a_1..a_{T-1}] so column t predicts a_t, i.e. every logit
        is conditioned on a_<t only.
        """
        bos = torch.full_like(actions[:, :1], self.stoi[BOS])
        inp = torch.cat([bos, actions[:, :-1]], dim=1)
        h, _ = self.rnn(self.embed(inp), self._h0(z))
        return self.out(h)

    def log_probs(self, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Eq. 2, per token: (N, T). Sum over the mask for log pi_theta(m)."""
        lp = torch.log_softmax(self.logits(actions, z), dim=-1)
        return lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, n: int, max_len: int = 72, device: str = "cpu") -> dict:
        """Roll out n molecules. T is decided by the rollout, not fixed (Sec 1.1).

        The mask covers a_1..a_T inclusive of EOS: the decision to stop is a
        real action the policy took and must earn gradient like any other.
        Everything after it is padding and is masked out.
        """
        was_training = self.training
        self.eval()
        z = torch.randn(n, self.z_dim, device=device)
        h = self._h0(z)
        tok = torch.full((n, 1), self.stoi[BOS], dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        actions, mask = [], []

        for _ in range(max_len):
            o, h = self.rnn(self.embed(tok), h)
            probs = torch.softmax(self.out(o[:, 0]), dim=-1)
            a = torch.multinomial(probs, 1).squeeze(-1)
            # Once a sequence has emitted EOS it is frozen at PAD, so the stored
            # actions stay a valid teacher-forcing input on the rescoring pass.
            a = torch.where(done, torch.zeros_like(a), a)
            mask.append((~done).float())
            actions.append(a)
            done = done | (a == self.stoi[EOS])
            tok = a.unsqueeze(1)
            if bool(done.all()):
                break

        actions = torch.stack(actions, dim=1)
        mask = torch.stack(mask, dim=1)
        logp = self.log_probs(actions, z)
        self.train(was_training)
        return {
            "z": z,
            "actions": actions,
            "mask": mask,
            "logp_old": logp.detach(),
            "smiles": [self.decode(row, m) for row, m in zip(actions, mask)],
        }

    def decode(self, actions: torch.Tensor, mask: torch.Tensor) -> str:
        """Tokens -> SMILES. Always syntactically valid; sanitization is separate."""
        toks = [self.vocab[i] for i, m in zip(actions.tolist(), mask.tolist()) if m]
        body = "".join(t for t in toks if t not in SPECIALS)
        return sf.decoder(body) if body else ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vocab": self.vocab,
                "state": self.state_dict(),
                "z_dim": self.z_dim,
                "hidden": self.hidden,
                "layers": self.layers,
                "emb": self.emb,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SelfiesGenerator":
        ck = torch.load(path, map_location=device, weights_only=False)
        gen = cls(
            ck["vocab"], emb=ck["emb"], hidden=ck["hidden"],
            layers=ck["layers"], z_dim=ck["z_dim"],
        )
        gen.load_state_dict(ck["state"])
        return gen.to(device)


def _tensorize(selfies: list[str], stoi: dict[str, int], max_len: int) -> torch.Tensor:
    """Right-pad token ids to max_len, with EOS appended after the body."""
    rows = []
    for s in selfies:
        ids = [stoi[t] for t in sf.split_selfies(s) if t in stoi][: max_len - 1]
        ids.append(stoi[EOS])
        rows.append(ids + [0] * (max_len - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def pretrain(
    dataset: str = "bbbp",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_len: int = 72,
    device: str = "cpu",
) -> SelfiesGenerator:
    """MLE warm start on real molecules.

    Not optional in practice: from a uniform babbler every sampled group is
    all-invalid, every reward is the same flat penalty, and the group has zero
    standard deviation -- the degenerate case `grpo.group_advantages` zeroes
    out. GRPO would then receive no gradient at all until validity appeared by
    chance, which is what this warm start supplies instead.
    """
    smiles = _load_raw(dataset)["smiles"].dropna().tolist()
    vocab = build_vocab(smiles)
    selfies = [s for s in (encode(x) for x in smiles) if s]
    print(f"{dataset}: {len(selfies)}/{len(smiles)} encodable, |V|={len(vocab)}")

    gen = SelfiesGenerator(vocab).to(device)
    data = _tensorize(selfies, gen.stoi, max_len).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    # Padding contributes no loss: a model rewarded for predicting [nop] after
    # EOS learns sequence length from the padding rather than from chemistry.
    lossf = nn.CrossEntropyLoss(ignore_index=0)

    for ep in range(epochs):
        gen.train()
        perm = torch.randperm(len(data), device=device)
        total = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[perm[i : i + batch_size]]
            z = torch.randn(len(batch), gen.z_dim, device=device)
            logits = gen.logits(batch, z)
            loss = lossf(logits.reshape(-1, len(vocab)), batch.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        probe = gen.sample(200, max_len=max_len, device=device)["smiles"]
        valid = sum(bool(s) and Chem.MolFromSmiles(s) is not None for s in probe)
        print(f"  epoch {ep + 1:>2}  nll={total / len(data):.4f}  valid={valid / 200:.1%}")

    path = CKPT_DIR / f"{dataset}_pretrained.pt"
    gen.save(path)
    print(f"saved {path}")
    return gen


def _self_check() -> None:
    torch.manual_seed(0)
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "CCN", "C1CCCCC1"]
    vocab = build_vocab(smiles)
    assert vocab[:3] == SPECIALS, "specials must hold indices 0,1,2"
    assert len(set(vocab)) == len(vocab), "duplicate token in vocab"

    gen = SelfiesGenerator(vocab, emb=16, hidden=32, z_dim=8)
    out = gen.sample(6, max_len=12)
    n, t = out["actions"].shape
    assert n == 6 and out["mask"].shape == (n, t) and out["logp_old"].shape == (n, t)
    assert not out["logp_old"].requires_grad, "pi_theta_old must be detached"

    # The live region must end on the EOS that closed it, and everything past it
    # must be PAD -- otherwise rescoring feeds the model different inputs.
    eos = gen.stoi[EOS]
    for a, m in zip(out["actions"], out["mask"]):
        live = int(m.sum())
        assert (a[live:] == 0).all(), "live region leaked into padding"
        if live < t:
            assert a[live - 1] == eos, "sequence ended without EOS"

    # Rescoring the stored actions must reproduce the sampled log-probs exactly.
    # If this drifts, the PPO ratio is not 1 at epoch 0 and the update is wrong.
    fresh = gen.log_probs(out["actions"], out["z"])
    assert torch.allclose(fresh, out["logp_old"], atol=1e-6), "sample/rescore mismatch"

    # Causality: perturbing a_T must not move the logits at t <= T.
    ref = gen.logits(out["actions"], out["z"])
    poked = out["actions"].clone()
    poked[:, -1] = (poked[:, -1] + 1) % len(vocab)
    assert torch.allclose(ref[:, :-1], gen.logits(poked, out["z"])[:, :-1], atol=1e-6), \
        "logits at t depend on a_t -- the model is peeking at its own answer"

    # SELFIES' syntactic guarantee: even an untrained babbler decodes to SOMETHING
    # parseable. Section 1.1's point -- it says nothing about sanitization.
    decoded = [s for s in out["smiles"] if s]
    assert decoded, "nothing decoded at all"

    # Gradient reaches the policy through the recomputed log-probs only.
    gen.log_probs(out["actions"], out["z"]).mul(out["mask"]).sum().backward()
    assert gen.out.weight.grad is not None and gen.out.weight.grad.abs().sum() > 0

    # Round trip through save/load must preserve the policy exactly, or a
    # resumed run scores its own stored actions under a different pi_theta_old.
    tmp = CKPT_DIR / "_selfcheck.pt"
    gen.save(tmp)
    same = SelfiesGenerator.load(tmp).log_probs(out["actions"], out["z"])
    tmp.unlink()
    assert torch.allclose(same, fresh, atol=1e-6), "save/load changed the policy"

    print(f"|V|={len(vocab)}  sampled T={t}  decoded={len(decoded)}/{n}")
    print(f"  e.g. {decoded[0][:60]!r}")
    print("generator self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.pretrain:
        pretrain(args.dataset, epochs=args.epochs, device=args.device)
    else:
        _self_check()
