"""XGBoost leaf kernel: sparse leaf-membership matrix + cosine-normalized similarity."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Optional


class LeafKernel:
    """Extracts XGBoost terminal-leaf one-hot fingerprints and computes the
    cosine-normalized leaf kernel K(x_i, x_j) = z(x_i)^T z(x_j) / sum_t alpha_t.

    Each tree contributes a one-hot block of size L_t (number of leaves in tree t).
    Weights alpha_t (uniform here) scale each block; with uniform alpha,
    ||z(x)||^2 = T so K = inner product / T.
    """

    def __init__(self, booster, alpha: Optional[np.ndarray] = None):
        self.booster = booster
        self._fitted = False
        self.alpha = alpha

    def fit(self, X) -> "LeafKernel":
        # apply returns (n, T) array of leaf ids
        leaves = self.booster.apply(X)
        self._build_leaf_index(leaves)
        self._fitted = True
        return self

    def _build_leaf_index(self, leaves: np.ndarray):
        n, T = leaves.shape
        if self.alpha is None:
            self.alpha = np.ones(T, dtype=np.float64)
        # For each tree, map raw leaf ids -> dense column offsets
        self._leaf_maps = []
        offsets = [0]
        for t in range(T):
            uniq = np.unique(leaves[:, t])
            id_map = {int(v): i for i, v in enumerate(uniq)}
            self._leaf_maps.append(id_map)
            offsets.append(offsets[-1] + len(uniq))
        self._offsets = np.array(offsets, dtype=np.int64)
        self.d_total = int(self._offsets[-1])
        self.T = T

    def transform(self, X) -> sp.csr_matrix:
        """Return sparse leaf-membership matrix Z of shape (n, d_total) where
        d_total = sum_t L_t. Each row has exactly T nonzero entries (one per tree),
        each equal to sqrt(alpha_t). Indexing is tree-major: for tree t we write
        rows[t*n : (t+1)*n] = [0..n-1] (np.tile pattern)."""
        leaves = self.booster.apply(X)
        n, T = leaves.shape
        assert T == self.T, "Tree count mismatch"
        rows = np.tile(np.arange(n, dtype=np.int64), T)  # tree-major: matches col-fill
        cols = np.empty(n * T, dtype=np.int64)
        vals = np.empty(n * T, dtype=np.float64)
        sqrt_alpha = np.sqrt(self.alpha)
        flat_idx = 0
        for t in range(T):
            id_map = self._leaf_maps[t]
            off = self._offsets[t]
            col_t = np.fromiter(
                (id_map.get(int(v), -1) for v in leaves[:, t]),
                dtype=np.int64,
                count=n,
            )
            cols[flat_idx : flat_idx + n] = np.where(col_t >= 0, col_t + off, -1)
            vals[flat_idx : flat_idx + n] = sqrt_alpha[t]
            flat_idx += n
        valid = cols >= 0
        Z = sp.csr_matrix(
            (vals[valid], (rows[valid], cols[valid])),
            shape=(n, self.d_total),
        )
        return Z

    @property
    def S(self) -> float:
        """Normalization constant S = sum_t alpha_t."""
        return float(np.sum(self.alpha))

    def kernel(self, Z1: sp.csr_matrix, Z2: sp.csr_matrix) -> np.ndarray:
        """Compute cosine-normalized kernel: K = Z1 Z2^T / S, returned as dense array."""
        K = (Z1 @ Z2.T).toarray() / self.S
        return K

    def topk_neighbors(
        self, Z_query: sp.csr_matrix, Z_train: sp.csr_matrix, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each query row, return (indices, sims) of top-k training neighbors
        under K_xgb. Naive O(n_q * n_tr) implementation; fine for synthetic sizes."""
        sims = self.kernel(Z_query, Z_train)
        n_q = sims.shape[0]
        if k >= sims.shape[1]:
            idx = np.argsort(-sims, axis=1)
            return idx, np.take_along_axis(sims, idx, axis=1)
        # partial top-k
        idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        # sort within top-k by similarity desc
        gathered = np.take_along_axis(sims, idx_part, axis=1)
        order = np.argsort(-gathered, axis=1)
        idx = np.take_along_axis(idx_part, order, axis=1)
        sims_topk = np.take_along_axis(gathered, order, axis=1)
        return idx, sims_topk
