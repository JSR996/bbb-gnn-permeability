"""Pareto reduction: objective matrix -> per-candidate scalar. The REDUCE step.

Our reward R = wD*D + wC*C + wQ*QED + wS*SA is a Minkowski-weighted linear
scalarization of a 4-objective vector, so Theorem 1 of Das & Dennis (1997)
applies verbatim: no choice of weights reaches a Pareto-optimal candidate
inside a concave fold of the frontier. Measured on our own sampled groups,
a linear reduction reaches only **0.48** of the per-group Pareto front over
2000 random weightings -- every weight experiment we have run searches inside
a space that cannot reach the other half.

The objectives are also almost uncorrelated (mean pairwise +0.009; D vs C is
-0.258, realism and permeability actively oppose), which is the regime where
a rank-based reduction has the most room. At group_size 64 the detection
bound of Pareto-GRPO's Proposition 2 gives 4.6% of the local objective range,
against 17.1% at their G=16.

`nsga2` ranks by dominance in objective space and never collapses the partial
order into a scalar before ranking, so Theorem 1 does not bite. Crowding
distance is the part that matters most here: it rewards candidates in SPARSE
regions of objective space, which is a direct push against the mode collapse
this project keeps measuring -- and unlike the size gate or the KL, it acts
inside the group rather than as an absolute floor.

What this does NOT fix: the group-relative advantage downstream is still
blind to a uniformly bad group (verified: it is invariant to uniform scale
and shift). Pareto reduction addresses reachability, not that blindness.
Only an absolute anchor -- a gate, the KL to pi_ref -- addresses the latter.

    python -m src.pareto      # self-check
"""

from __future__ import annotations

import numpy as np

MODES = ("linear", "nsga2", "composite")


def dominance_count(F: np.ndarray) -> np.ndarray:
    """Number of candidates strictly dominating each row. 0 = Pareto-optimal.

    `j` dominates `i` when it is >= on every objective and > on at least one,
    with all objectives oriented so higher is better.

    ponytail: O(N^2) in the group, which is 64 here -- 4096 comparisons per
    step against a GNN forward pass, so it is free. Switch to a kd-tree or
    Kung's algorithm only if group_size grows past a few hundred.
    """
    n = len(F)
    out = np.zeros(n, dtype=int)
    for i in range(n):
        ge = (F >= F[i]).all(axis=1)
        gt = (F > F[i]).any(axis=1)
        out[i] = int(np.sum(ge & gt))
    return out


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance: how isolated each candidate is.

    Per objective, sort and sum the normalized gap between each point's two
    neighbours. Boundary points are conventionally infinite; they are capped
    at the largest finite value instead, so they stay preferred without
    turning the reduction into "always pick the two extremes" -- an infinity
    would make every interior difference numerically irrelevant.
    """
    n, k = F.shape
    if n < 3:
        return np.zeros(n)
    d = np.zeros(n)
    for j in range(k):
        order = np.argsort(F[:, j])
        rng = F[order[-1], j] - F[order[0], j]
        if rng < 1e-12:          # objective is constant: it separates nothing
            continue
        d[order[0]] = d[order[-1]] = np.inf
        d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / rng
    finite = d[np.isfinite(d)]
    cap = finite.max() if finite.size else 1.0
    return np.where(np.isfinite(d), d, cap)


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def reduce_objectives(
    F: np.ndarray,
    weights: np.ndarray | None = None,
    mode: str = "linear",
    crowd_coef: float = 0.5,
) -> np.ndarray:
    """(N, K) objectives -> (N,) scalar reward. Higher is better throughout.

    linear     weighted sum, the current behaviour.
    nsga2      -dominance rank, plus crowd_coef * crowding distance.
    composite  nsga2 ordering scaled onto the linear reward's spread; the
               best arm in Pareto-GRPO, and the hedge against rank-based
               reduction starving the gradient of variance, which is the
               failure they measured on high-correlation panels.
    """
    if mode not in MODES:
        raise ValueError(f"unknown reduce mode {mode!r}, expected one of {MODES}")
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError(f"expected (N, K) objectives, got shape {F.shape}")

    if mode == "linear":
        if weights is None:
            raise ValueError("linear reduction needs weights")
        return F @ np.asarray(weights, dtype=float)

    rank = -dominance_count(F).astype(float)      # 0 is best, so negate
    score = _z(rank) + crowd_coef * _z(crowding_distance(F))
    if mode == "nsga2":
        return score
    # composite: keep the rank ordering, borrow the linear reward's scale so
    # the downstream advantage sees a comparable spread.
    if weights is None:
        raise ValueError("composite reduction needs weights for its scale")
    lin = F @ np.asarray(weights, dtype=float)
    return _z(score) * lin.std() + lin.mean()


def _self_check() -> None:
    rng = np.random.default_rng(0)

    # Dominance. (1,1) beats everything here, including the axis extremes:
    # it ties them on one objective and wins on the other, which is exactly
    # what weak dominance means. Only a point that is NOT tied-and-beaten
    # survives, so the genuinely incomparable pair is tested separately.
    F = np.array([[1.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    dc = dominance_count(F)
    assert dc.tolist() == [0, 1, 1, 1], dc

    # Incomparable points: each wins on one objective, so neither dominates.
    assert (dominance_count(np.array([[1.0, 0.0], [0.0, 1.0]])) == 0).all()

    # Equal rows dominate nobody: dominance needs a STRICT improvement, or
    # every duplicate in a collapsed group would be ranked below its twin.
    assert (dominance_count(np.ones((5, 3))) == 0).all()

    # Crowding: the point buried inside a tight cluster must score lowest.
    # Its NEIGHBOURS are not a fair comparison -- the cluster's edges each
    # have one far neighbour, so a large gap there is correct, not a bug.
    F = np.array([[0.0, 0.0], [0.50, 0.0], [0.51, 0.0], [0.52, 0.0], [1.0, 0.0]])
    cd = crowding_distance(F)
    assert cd[2] == cd.min() and cd[2] < 0.1, cd
    assert cd[0] == cd.max() and cd[4] == cd.max(), "boundaries must stay preferred"
    assert np.isfinite(cd).all(), "infinities leaked out of crowding distance"
    # A constant objective separates nothing and must not contribute.
    assert np.allclose(crowding_distance(np.c_[F[:, 0], np.ones(5)]), cd)

    # THE MOTIVATION, as an executable claim. Build a frontier with a concave
    # fold and confirm no weighting reaches the folded point while nsga2
    # ranks it top. If this ever fails, the reduction is not buying what the
    # module docstring says it buys.
    # A CONCAVE front: sqrt(x) + sqrt(y) = 1, i.e. x = t^2, y = (1-t)^2. Every
    # point on it is Pareto-optimal (x rises as y falls, so no two compare),
    # but w1*t^2 + w2*(1-t)^2 is convex in t, so a weighted sum is always
    # maximised at an endpoint. A circular arc would NOT work here -- it is
    # convex, and its interior points are genuinely dominated.
    t = np.linspace(0.0, 1.0, 25)
    F = np.c_[t ** 2, (1 - t) ** 2]
    assert (dominance_count(F) == 0).all(), "every point on this front is optimal"
    W = rng.dirichlet(np.ones(2), size=4000)
    winners = {int(np.argmax(F @ w)) for w in W}
    assert winners <= {0, len(F) - 1}, f"only endpoints should win, got {winners}"
    lin_reach = len(winners) / len(F)
    assert lin_reach < 0.15, lin_reach
    # The rank reduction ranks the folded interior alongside the endpoints,
    # because dominance does not care about convexity.
    assert dominance_count(F)[len(F) // 2] == 0

    # Rank reduction is invariant to any monotone rescaling of an objective,
    # which a weighted sum is not -- that invariance is why it is immune to
    # the C term's saturation.
    F = rng.random((40, 3))
    a = reduce_objectives(F, mode="nsga2")
    G = F.copy()
    G[:, 0] = G[:, 0] ** 3                               # monotone, order-preserving
    assert np.allclose(dominance_count(F), dominance_count(G)), \
        "dominance changed under a monotone transform"

    # Composite keeps the nsga2 ORDER but the linear reward's SCALE.
    w = np.array([0.4, 0.3, 0.3])
    comp = reduce_objectives(F, w, mode="composite")
    lin = F @ w
    assert np.array_equal(np.argsort(comp), np.argsort(a)), "composite reordered"
    assert abs(comp.std() - lin.std()) < 1e-6, (comp.std(), lin.std())

    # linear mode must be exactly what it was before.
    assert np.allclose(reduce_objectives(F, w, mode="linear"), lin)

    try:
        reduce_objectives(F, mode="linear")
        raise AssertionError("linear without weights should have been rejected")
    except ValueError:
        pass

    print(f"concave-fold demo: linear reaches {lin_reach:.2f} of the front, "
          f"nsga2 ranks the folded point at dominance 0")
    print("pareto self-check passed")


if __name__ == "__main__":
    _self_check()
