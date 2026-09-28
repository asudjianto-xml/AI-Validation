"""K-medoids on the loss model's leaf-kernel dissimilarity.

Medoids rather than centroids, for three reasons specific to this framework.
The dissimilarity is used directly, so no spectral embedding is formed and the
question of how to choose Nystrom landmarks does not arise. A medoid is an
observation, so it has a leaf assignment and can be traced through the trees,
which is what a readable boundary needs and what a k-means centroid cannot
supply. And the clusters are natural groups rather than fixed-size balls, which
is where the strongest confirmed weakness on the credit-default task came from.

The clustering objective never sees the loss. Medoids are placed to minimize
within-cluster dissimilarity and the weakness of the resulting clusters is
measured afterwards, so the partition is not selected on the quantity being
estimated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PamResult:
    medoids: np.ndarray
    labels: np.ndarray
    cost: float
    n_swaps: int
    silhouette: float


def leaf_dissimilarity(aux, X: np.ndarray) -> np.ndarray:
    """One minus the normalized leaf co-membership, a proper dissimilarity."""
    Z = aux.phi(np.asarray(X, dtype=float))
    K = np.asarray((Z @ Z.T).todense()) / aux.kernel.S
    np.fill_diagonal(K, 1.0)
    return 1.0 - K


def _assign(D: np.ndarray, med: np.ndarray):
    """Nearest and second-nearest medoid for every point, which is what makes
    the swap phase a matrix operation rather than a re-assignment per candidate."""
    sub = D[:, med]
    o = np.argsort(sub, axis=1)
    near, second = o[:, 0], o[:, 1] if sub.shape[1] > 1 else o[:, 0]
    d1 = sub[np.arange(len(D)), near]
    d2 = sub[np.arange(len(D)), second]
    return near, d1, d2


def pam(D: np.ndarray, k: int, max_swaps: int = 200, seed: int = 0) -> PamResult:
    """Partitioning around medoids: greedy build, then steepest-descent swaps."""
    n = len(D)
    if not 2 <= k <= n // 2:
        raise ValueError("k must be between 2 and half the sample size")
    med = [int(np.argmin(D.sum(1)))]                       # build
    while len(med) < k:
        cur = D[:, med].min(1)
        gain = np.maximum(cur[None, :] - D, 0.0).sum(1)
        gain[med] = -np.inf
        med.append(int(np.argmax(gain)))
    med = np.array(sorted(med))

    swaps = 0
    for _ in range(max_swaps):
        near, d1, d2 = _assign(D, med)
        best = (0.0, -1, -1)
        for j in range(k):
            owned = near == j
            # cost if medoid j is replaced by each candidate h, all h at once
            new = np.where(owned[None, :], np.minimum(D, d2[None, :]),
                           np.minimum(D, d1[None, :]))
            delta = new.sum(1) - d1.sum()
            delta[med] = np.inf
            h = int(np.argmin(delta))
            if delta[h] < best[0]:
                best = (float(delta[h]), j, h)
        if best[1] < 0:
            break
        med = np.array(sorted(np.where(med == med[best[1]], best[2], med)))
        swaps += 1

    near, d1, d2 = _assign(D, med)
    sil = float(np.mean((d2 - d1) / np.maximum(d2, 1e-12)))
    return PamResult(med, near, float(d1.sum()), swaps, sil)


def pam_build_swap(D: np.ndarray, k: int, max_iter: int = 300, block: int = 512) -> np.ndarray:
    """BUILD then SWAP, reproducing scikit-learn-extra 0.3.0
    ``KMedoids(method="pam", init="build", metric="precomputed")`` exactly.

    Ported from scikit-learn-extra's ``_k_medoids_helper.pyx`` (BSD-3-Clause,
    Timothee Mathieu) so replays need neither that package nor its NumPy 1.x
    extension. Tie-breaking and summation order follow the Cython loops: BUILD
    keeps the last candidate with the largest gain, SWAP keeps the first
    (candidate, medoid) pair with the most negative cost change, and every cost
    is accumulated sequentially (``cumsum``) over non-medoids in index order.
    Returns medoid indices in the order ``medoid_indices_`` reports them.
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and the sample size")

    med = [int(np.argmin(np.sum(D, axis=0)))]              # build
    rest = np.delete(np.arange(n), med[0])
    Dj = D[med[0]].copy()
    for _ in range(k - 1):
        gain = np.empty(len(rest))
        for s in range(0, len(rest), block):
            rows = rest[s:s + block]
            terms = np.maximum(0.0, Dj[rest][None, :] - D[np.ix_(rows, rest)])
            gain[s:s + block] = np.cumsum(terms, axis=1)[:, -1]
        pos = len(gain) - 1 - int(np.argmax(gain[::-1]))   # last maximum
        new = int(rest[pos])
        med.append(new)
        rest = np.delete(rest, pos)
        Dj = np.minimum(Dj, D[:, new])
    med = np.array(med, dtype=np.int64)
    if k == 1:
        return med

    Djs, Ejs = np.sort(D[med], axis=0)[[0, 1]]             # swap
    for _ in range(max_iter):
        old = med.copy()
        other = np.delete(np.arange(n), med)
        dj, ej = Djs[other], Ejs[other]
        cost = np.empty((len(other), k))
        for s in range(0, len(other), block):
            h = other[s:s + block]
            Dhj = D[np.ix_(h, other)]                      # D[id_h, id_j]
            Djh = D[np.ix_(other, h)].T                    # D[id_j, id_h]
            second = Dhj < ej[None, :]
            closer = Djh < dj[None, :]
            for i, id_i in enumerate(med):
                owned = (D[id_i, other] == dj)[None, :]
                terms = np.where(owned, np.where(second, Djh - dj[None, :], (ej - dj)[None, :]),
                                 np.where(closer, Djh - dj[None, :], 0.0))
                own = np.where(D[h, id_i] < Ejs[id_i], D[id_i, h], Ejs[id_i])
                cost[s:s + block, i] = np.cumsum(terms, axis=1)[:, -1] + own
        best = int(np.argmin(cost.ravel()))                # first minimum, candidate-major
        if cost.ravel()[best] < 0:
            h, i = divmod(best, k)
            med[med == med[i]] = other[h]
            Djs, Ejs = np.sort(D[med], axis=0)[[0, 1]]
        if np.array_equal(old, med):
            break
    return med


def medoid_of(D: np.ndarray, members: np.ndarray) -> int:
    """The member with the smallest average dissimilarity to the rest."""
    members = np.asarray(members, dtype=int)
    return int(members[np.argmin(D[np.ix_(members, members)].mean(1))])
