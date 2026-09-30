"""Reactive-group alerts as a soft reward term.

Why now, when the same idea failed earlier. Tested against carbon
tetrachloride, substructure catalogues were useless: BRENK caught 2 of 3
solvents and flagged 35% of real BBB+ drugs, CHEMBL caught all 3 and 64%.
Nothing there separated. But that was the wrong job for them. The size gate
now handles degenerate small molecules, and what the generator does next is
different: freed from the solvent optimum it slides into drug-SIZED reactive
electrophiles. Measured on the final groups, the most common alerts in the
guarded arms are long aliphatic chains (17), aldehydes (15), alkyl halides
(9) and imines (14). Catalogues are good at exactly those.

So this is a soft, weighted term rather than a gate. A hard veto is not
available: 39% of real BBB+ drugs in BBBP trip these filters, so rejecting
on one alert would reject a third of the answer.

    S_alerts = exp(-lambda * N_alerts)

A TARGETED SMARTS set is used rather than a whole catalogue, for the reason
above -- the aim is to catch the electrophiles the generator actually
reaches for while tripping on as few real drugs as possible.

    python -m src.alerts        # self-check + false-positive rates
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Reactive or labile groups, each with why it is here. Deliberately narrower
# than BRENK: every pattern below is a stability, reactivity or
# genotoxicity liability rather than a general "unlovely" flag.
ALERT_SMARTS: dict[str, str] = {
    # Strained electrophilic heterocycles -- DNA alkylators. The aziridine in
    # the gate+GM samples is the single worst structure the run produced.
    "aziridine": "C1CN1",
    "epoxide": "C1CO1",
    "thiirane": "C1CS1",
    # Carbonyl electrophiles and labile C=N. Aldehydes and imines dominate
    # the guarded arms' alert counts.
    "aldehyde": "[CX3H1](=O)[#6]",
    "imine": "[CX3]=[NX2][#6,#1]",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "anhydride": "[CX3](=O)O[CX3]=O",
    "acyclic_imide": "[CX3](=O)[NX3][CX3]=O",
    # Hepatotoxic / mutagenic nitrogen.
    "hydrazine": "[NX3][NX3]",
    "azide": "[NX2]=[NX2+0,NX2-]=[NX1-,NX1+0]",
    "nitroso": "[NX2]=O",
    "n_oxide_acyclic": "[NX3][OX2H1]",
    # Michael acceptors -- covalent protein binders.
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    # Latent alkylating agents.
    "alkyl_halide_sp3": "[CX4][F,Cl,Br,I]",
    "halo_pyridine_2": "[F,Cl,Br,I][c]1[n][c][c][c][c]1",
    # Lipophilicity hack: the BBB objective pays for grease, and an
    # unbranched chain is the cheapest way to buy it.
    "long_chain": "[CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2]",
}

_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in ALERT_SMARTS.items()}
_BAD = [k for k, v in _PATTERNS.items() if v is None]
if _BAD:  # a typo here would silently stop a whole class being penalized
    raise ValueError(f"unparsable SMARTS: {_BAD}")


def count_alerts(mol: Chem.Mol) -> int:
    """Number of DISTINCT alert types tripped, not total match count.

    A molecule with three separate long-chain matches has one problem, not
    three, and counting occurrences would let a single motif dominate the
    penalty.
    """
    return sum(1 for p in _PATTERNS.values() if mol.HasSubstructMatch(p))


def alert_names(mol: Chem.Mol) -> list[str]:
    return [k for k, p in _PATTERNS.items() if mol.HasSubstructMatch(p)]


def alert_score(smiles: list[str], lam: float = 0.15) -> np.ndarray:
    """exp(-lam * n_alerts) per molecule; 1.0 for anything unparseable.

    Unparseable returns 1.0 rather than 0.0 because validity is gated
    upstream -- such molecules already take the flat invalid penalty, and a
    0 here would double-penalize through a term that is meant to rank the
    survivors.
    """
    out = np.ones(len(smiles))
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            out[i] = float(np.exp(-lam * count_alerts(m)))
    return out


def _self_check() -> None:
    import pandas as pd

    from .datasets import ROOT

    clean = "CC(=O)Oc1ccccc1C(=O)O"          # aspirin
    aziridine = "CC(NC1=CC=CC=C1)OCC(=O)N2CC2C(C)C3=CC=C(F)C=C3CC"  # from gate+GM
    assert count_alerts(Chem.MolFromSmiles(clean)) == 0, alert_names(Chem.MolFromSmiles(clean))
    az = Chem.MolFromSmiles(aziridine)
    assert "aziridine" in alert_names(az), alert_names(az)

    # Monotone in alert count, and bounded in (0, 1].
    s = alert_score([clean, aziridine], lam=0.15)
    assert s[0] == 1.0 and 0 < s[1] < 1.0, s
    assert alert_score([""], lam=0.15)[0] == 1.0, "invalid must not double-penalize"

    # Distinct types, not occurrences: two long chains are one problem.
    one = Chem.MolFromSmiles("CCCCCCCCCC")
    two = Chem.MolFromSmiles("CCCCCCCCCCN(CCCCCCCCCC)C")
    assert count_alerts(one) == count_alerts(two) == 1, \
        (alert_names(one), alert_names(two))

    # False-positive rate on real BBB+ drugs is the constraint that forces
    # this to be soft rather than a gate. Report it rather than assume it.
    df = pd.read_csv(ROOT / "BBBP.csv").dropna(subset=["smiles"])
    mols = [m for m in (Chem.MolFromSmiles(x) for x in df[df.p_np == 1].smiles) if m]
    n = np.array([count_alerts(m) for m in mols])
    print(f"  BBBP BBB+ (n={len(mols)}): {np.mean(n > 0):.0%} trip >=1 alert, "
          f"mean {n.mean():.2f} types")
    for lam in (0.15, 0.3, 0.5):
        sc = np.exp(-lam * n)
        print(f"    lam={lam}: real-drug score {sc.mean():.3f} mean, "
              f"{np.mean(sc < 0.8):.0%} below 0.8")
    assert np.mean(n > 0) < 0.5, "targeted set should trip fewer drugs than BRENK's 35%"

    print("alerts self-check passed")


if __name__ == "__main__":
    _self_check()
