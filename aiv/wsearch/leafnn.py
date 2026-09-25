"""Fixed-support neighborhoods in the loss model's leaf geometry.

A third declared region class, alongside the box and the ball. A candidate is an
anchor row, and its region is the m rows most similar to that anchor under the
leaf kernel of the auxiliary loss model. The class is not nested with the box
family: two rows are close here when the error model routes them to the same
terminal leaves, which is a statement about how the fixed model fails rather
than about where the rows sit in input space.

This is a different use of the same kernel from the partitioning benchmark. A
spectral partition offers only as many candidate regions as it has parts, and
the parts are chosen for cohesion rather than for loss, so only the selection
among them is loss-informed. A neighborhood rule offers one candidate per
observation and keeps support fixed, so it can be ranked and searched like any
other class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class LeafRegion:
    anchor: int
    member_idx: np.ndarray
    threshold: float

    @property
    def m(self) -> int:
        return len(self.member_idx)


@dataclass(frozen=True)
class FrozenLeafRegion:
    """A fixed subset of input space: the rows whose leaf-kernel similarity to a
    stored anchor reaches a stored threshold.

    The anchor's leaf vector and the threshold are frozen together, so
    membership on new rows changes only because new rows occupy the region. The
    auxiliary model is part of the frozen object, since the leaf assignment is
    what defines the metric.
    """

    features: tuple
    anchor_vector: sp.csr_matrix
    threshold: float
    aux: object

    def similarity(self, df) -> np.ndarray:
        X = df[list(self.features)].to_numpy(dtype=float) if hasattr(df, "columns") else np.asarray(df, float)
        return np.asarray(self.aux.phi(X) @ self.anchor_vector.T.todense()).ravel()

    def mask(self, df) -> np.ndarray:
        return self.similarity(df) >= self.threshold


class LeafNeighborhoodConstructor:
    """Maps an anchor row to the m rows nearest it in leaf space.

    The similarity matrix is the sparse leaf-membership product, so it is formed
    once for the discovery rows and reused for every anchor. Ties are broken the
    way the box constructor breaks them, by taking the rows strictly above the
    threshold and filling the remaining places from the tied set in order of
    standardized Euclidean distance, so the region is defined rather than left to
    the sort.
    """

    def __init__(self, aux, X: np.ndarray, m: int, Xs: np.ndarray | None = None):
        self.aux = aux
        self.X = np.asarray(X, dtype=float)
        self.m = int(m)
        self.Z = aux.phi(self.X)
        self.S = np.asarray((self.Z @ self.Z.T).todense())
        self.n = self.S.shape[0]
        self.Xs = np.asarray(Xs, float) if Xs is not None else self.X

    def build(self, anchor: int) -> LeafRegion:
        s = self.S[int(anchor)]
        thr = float(np.sort(s)[::-1][self.m - 1])
        inside = np.flatnonzero(s > thr)
        tied = np.flatnonzero(s == thr)
        need = self.m - len(inside)
        if need > 0 and len(tied) > need:
            d = np.sqrt(((self.Xs[tied] - self.Xs[int(anchor)]) ** 2).sum(1))
            tied = tied[np.argsort(d, kind="stable")[:need]]
        idx = np.sort(np.concatenate([inside, tied[:max(need, 0)]]))
        return LeafRegion(int(anchor), idx, thr)

    def region_for(self, z_std, standardizer):
        """Members of the neighborhood around a continuous anchor.

        The anchor is given in standardized coordinates and mapped back through
        the declared scaling before the auxiliary model assigns its leaves, so
        the metric is the same one the observed-row anchors use.
        """
        raw = standardizer.inverse(np.atleast_2d(np.asarray(z_std, dtype=float)))
        sim = np.asarray((self.Z @ self.aux.phi(raw).T).todense()).ravel()
        return np.argpartition(-sim, self.m - 1)[: self.m]

    def freeze_anchor(self, z_std, standardizer, features) -> FrozenLeafRegion:
        """Freeze a searched anchor: its leaf vector and the realized threshold."""
        raw = standardizer.inverse(np.atleast_2d(np.asarray(z_std, dtype=float)))
        anchor = self.aux.phi(raw)
        sim = np.asarray((self.Z @ anchor.T).todense()).ravel()
        thr = float(np.sort(sim)[::-1][self.m - 1])
        return FrozenLeafRegion(tuple(features), anchor, thr, self.aux)

    def build_all(self):
        return [self.build(a) for a in range(self.n)]

    def freeze(self, region: LeafRegion, features) -> FrozenLeafRegion:
        return FrozenLeafRegion(tuple(features), self.Z[region.anchor],
                                region.threshold, self.aux)


def rank_by_proxy(constructor: LeafNeighborhoodConstructor, regions, g_hat: np.ndarray):
    """Region-averaged loss-model prediction, the same free ranking B1 uses.

    Reads no losses, so its cost is the auxiliary fit whatever the number of
    anchors, and the anchors are every observed row rather than a sample.
    """
    return np.array([g_hat[reg.member_idx].mean() for reg in regions])


def leaf_search(constructor: "LeafNeighborhoodConstructor", ev, meter, budget: int,
                start_anchor, standardizer, sigma: float = 0.6, adapt: float = 0.85,
                restart_below: float = 0.02, rng=None, aux_fit_cost: int = 0):
    """Search the leaf class over a continuous anchor.

    Enumerating observed rows gives one candidate per observation, which is a
    sample of the class rather than a search of it. The anchor need not be an
    observed row: any point of input space has a leaf assignment under the
    auxiliary model, so the class is continuous in the anchor and can be searched
    with the same evolution strategy used for boxes. The objective is the
    measured contrast, not the loss model's prediction of it, because optimizing
    a proxy fitted to the discovery losses buys discovery contrast that does not
    survive confirmation.
    """
    rng = rng or np.random.default_rng(0)
    if aux_fit_cost:
        meter.charge(aux_fit_cost, "loss_model_fit")
    cur = np.asarray(start_anchor, dtype=float).copy()
    idx = constructor.region_for(cur, standardizer)
    cur_M = ev.contrast(idx)
    best = (cur_M, cur.copy(), idx)
    history = []
    while meter.reads < budget:
        cand = cur + sigma * rng.normal(size=len(cur))
        i2 = constructor.region_for(cand, standardizer)
        M2 = ev.contrast(i2)
        if M2 > cur_M:
            cur, cur_M, sigma = cand, M2, sigma / adapt
        else:
            sigma *= adapt
        if sigma < restart_below:
            sigma = 0.6
        if cur_M > best[0]:
            best = (cur_M, cur.copy(), i2)
        history.append((meter.reads, best[0]))
    return best[0], best[1], best[2], history


def exhaustive_contrasts(constructor: "LeafNeighborhoodConstructor", losses: np.ndarray):
    """Exact contrast of every observed row taken as a centroid, in one pass.

    The proximity matrix is already formed, so each centroid's region is an
    argpartition of one of its rows and every contrast follows from the in-region
    sums. On the credit-default task this evaluates all 2500 centroids in under a
    tenth of a second, and its maximum has the highest confirmed contrast and the
    smallest discovery-to-confirmation shrinkage of any method measured, because
    a maximum over the observed centroids is a far milder selection than a
    maximum over a continuous anchor space.

    Returns the contrast per centroid and the member indices per centroid.
    """
    r = np.asarray(losses, dtype=float)
    n, m = constructor.n, constructor.m
    members = np.argpartition(-constructor.S, m - 1, axis=1)[:, :m]
    in_sum = r[members].sum(1)
    contrast = (in_sum / m) / ((r.sum() - in_sum) / (n - m)) - 1.0
    return contrast, members
