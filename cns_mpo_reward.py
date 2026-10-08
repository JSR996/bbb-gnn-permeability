"""Weighted CNS MPO reward + extra factors + GDPO-style decoupled group advantages.

Plugs into a GRPO loop: score a group of SELFIES -> per-term scores -> per-term
group-normalised advantages -> weighted sum.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

# --- piecewise-linear desirability (Wager et al. CNS MPO) -------------------
def _down(x, hi_score, lo_score):          # 1 for x<=hi_score, 0 for x>=lo_score
    return float(np.clip((lo_score - x) / (lo_score - hi_score), 0.0, 1.0))

def _hump(x, a, b, c, d):                  # 0 below a, ramp a-b, 1 on b-c, ramp c-d
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)

# Each: name -> (fn(mol, ctx) -> value, desirability(value) -> [0,1])
def _props(mol, logd_fn=None, pka_fn=None):
    p = {
        "clogp": Crippen.MolLogP(mol),      # proxy for ClogP
        "mw": Descriptors.MolWt(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
    }
    if logd_fn:
        p["clogd"] = logd_fn(mol)           # needs external tool (not in RDKit)
    if pka_fn:
        p["pka"] = pka_fn(mol)              # most basic centre; external tool
    return p

DESIR = {
    "clogp": lambda x: _down(x, 3, 5),
    "clogd": lambda x: _down(x, 2, 4),
    "mw":    lambda x: _down(x, 360, 500),
    "tpsa":  lambda x: _hump(x, 20, 40, 90, 120),
    "hbd":   lambda x: _down(x, 0.5, 3.5),
    "pka":   lambda x: _down(x, 8, 10),
}

def cns_mpo(mol, weights=None, logd_fn=None, pka_fn=None):
    """Weighted CNS MPO in [0,1]. Missing terms (no logD/pKa fn) are dropped and
    weights renormalised -- report this in the paper."""
    p = _props(mol, logd_fn, pka_fn)
    w = {k: (weights or {}).get(k, 1.0) for k in p}
    s = sum(w.values())
    return sum(w[k] * DESIR[k](p[k]) for k in p) / s

# --- extra factors ----------------------------------------------------------
def sa_norm(mol):
    from rdkit.Chem import RDConfig
    import os, sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    return (10.0 - sascorer.calculateScore(mol)) / 9.0   # 1 = easy

TERMS = {            # name -> fn(mol) -> [0,1]; "C" comes from the GNN ensemble
    "mpo": cns_mpo,
    "qed": QED.qed,
    "sa":  sa_norm,
}

def score_group(smiles_list, clf_prob_fn):
    """Returns dict term -> np.array(G). Invalid molecules score 0 on every term."""
    G = len(smiles_list)
    out = {k: np.zeros(G) for k in [*TERMS, "C"]}
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [i for i, m in enumerate(mols) if m is not None]
    for i in valid:
        for k, f in TERMS.items():
            out[k][i] = f(mols[i])
    if valid:
        out["C"][valid] = clf_prob_fn([smiles_list[i] for i in valid])
    return out

def decoupled_advantage(term_scores, weights, sd_floor=1e-3, clip=5.0):
    """GDPO-style: normalise each term within the group, THEN weight and sum.
    sd_floor stops a saturated term (sd_C ~ 0.005) from being blown up.
    Returns (advantage[G], per-term sd for logging)."""
    adv, sds = 0.0, {}
    for k, x in term_scores.items():
        sd = x.std()
        sds[k] = sd
        z = (x - x.mean()) / max(sd, sd_floor)
        adv = adv + weights.get(k, 1.0) * np.clip(z, -clip, clip)
    return adv, sds

if __name__ == "__main__":
    # smoke test: clf stub returns 0.9
    smi = ["CCN(CC)CCOC(=O)c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "bad"]
    sc = score_group(smi, lambda s: np.full(len(s), 0.9))
    for k, v in sc.items():
        print(k, np.round(v, 3))
    adv, sds = decoupled_advantage(sc, {"mpo": 1, "qed": 1, "sa": 1, "C": 1})
    print("adv", np.round(adv, 3), {k: round(float(v), 3) for k, v in sds.items()})
