"""The auxiliary loss model and the one-shot benchmarks built on it.

A shallow tree ensemble fitted to the fixed model's observation loss supplies
three things the framework uses separately: a prediction of where loss is high,
a leaf geometry in which regions can be compared, and the feature space in which
the Bayesian optimizer places its prior. Because a boosted ensemble is additive
over leaves, the one-shot benchmark is the plug-in linear predictor in exactly
that feature space, which is what makes the comparison against the surrogate a
statement about sequential measurement rather than about representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import xgboost
from sklearn.tree import DecisionTreeRegressor

from aiv._vendor.kernel_xgb.kernel import LeafKernel


@dataclass
class AuxLossModel:
    """Shallow gradient-boosted regressor with target the observation loss.

    `leaf_values` is the vector v for which g(x) = base + phi(x) . v exactly,
    with phi the weighted one-hot leaf feature of the leaf kernel. The identity
    is checked at fit time rather than assumed, since it is the basis of the
    comparison between the one-shot benchmark and the composite-kernel surrogate.
    """

    depth: int = 3
    n_estimators: int = 200
    learning_rate: float = 0.1
    seed: int = 0
    model: object = None
    kernel: LeafKernel = None
    leaf_values: np.ndarray = None
    base: float = 0.0
    reconstruction_error: float = np.nan

    def fit(self, X: np.ndarray, r: np.ndarray) -> "AuxLossModel":
        X = np.asarray(X, dtype=float)
        r = np.asarray(r, dtype=float)
        self.model = xgboost.XGBRegressor(
            max_depth=self.depth, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, subsample=0.8, colsample_bytree=0.8,
            random_state=self.seed,
        )
        self.model.fit(X, r)
        self.kernel = LeafKernel(self.model).fit(X)
        self.leaf_values = self._leaf_value_vector()
        Z = self.kernel.transform(X)
        offset = self.model.predict(X) - np.asarray(Z @ self.leaf_values).ravel()
        self.base = float(offset.mean())
        self.reconstruction_error = float(np.abs(offset - self.base).max())
        return self

    def _leaf_value_vector(self) -> np.ndarray:
        """Leaf weights laid out in the leaf kernel's column order."""
        df = self.model.get_booster().trees_to_dataframe()
        leaves = df[df.Feature == "Leaf"]
        v = np.zeros(self.kernel.d_total, dtype=float)
        alpha = np.sqrt(self.kernel.alpha)
        for t, id_map in enumerate(self.kernel._leaf_maps):
            off = int(self.kernel._offsets[t])
            sub = leaves[leaves.Tree == t]
            for node, value in zip(sub.Node.to_numpy(), sub.Gain.to_numpy()):
                col = id_map.get(int(node))
                if col is not None:
                    v[off + col] = float(value) / alpha[t]
        return v

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float)

    def phi(self, X) -> sp.csr_matrix:
        """Weighted one-hot leaf features, one row per observation."""
        return self.kernel.transform(np.asarray(X, dtype=float))

    def importance(self, features) -> list:
        gains = np.asarray(self.model.feature_importances_, dtype=float)
        return sorted(zip(list(features), gains.tolist()), key=lambda kv: -kv[1])


def region_embeddings(Phi: sp.csr_matrix, members) -> np.ndarray:
    """Mean leaf feature of each region, the cached embedding the kernel needs.

    Averaging the rows of a region rather than forming pairwise similarities
    turns the region-to-region kernel into one inner product, which is what
    makes the surrogate affordable once the embeddings are stored.
    """
    rows = np.concatenate([np.asarray(a, dtype=int) for a in members])
    counts = np.array([len(a) for a in members])
    ptr = np.concatenate([[0], np.cumsum(counts)])
    sel = sp.csr_matrix(
        (np.repeat(1.0 / counts, counts), rows, ptr),
        shape=(len(members), Phi.shape[0]),
    )
    return np.asarray((sel @ Phi).todense())


def b1_region_scores(embeddings: np.ndarray, aux: AuxLossModel) -> np.ndarray:
    """Region-averaged loss-model prediction, the B1 score.

    Equals the mean of g over the region's members, because a boosted ensemble
    is linear in the leaf features. Ranking candidates by the pointwise
    prediction instead optimizes a different functional than the objective and
    favors regions centered on isolated high-prediction observations.
    """
    return aux.base + embeddings @ aux.leaf_values


def error_tree_region(X: np.ndarray, r: np.ndarray, m: int, ccp_alpha: float = 0.0,
                      max_leaf_nodes: int = 32, seed: int = 0):
    """B0: highest-mean-loss leaf of a pruned regression tree fitted to the loss.

    Returns the member indices of the strongest leaf holding at least `m`
    observations, together with the fitted tree. This is the incumbent practice
    in model risk review and it produces an auditable rule with no search loop.
    """
    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, ccp_alpha=ccp_alpha,
        min_samples_leaf=m, random_state=seed,
    ).fit(np.asarray(X, float), np.asarray(r, float))
    leaf = tree.apply(np.asarray(X, float))
    best, best_mean = None, -np.inf
    for lid in np.unique(leaf):
        idx = np.flatnonzero(leaf == lid)
        if len(idx) >= m and r[idx].mean() > best_mean:
            best, best_mean = idx, float(r[idx].mean())
    if best is None:
        raise ValueError("no leaf reached the required support")
    return best, tree


def loss_kernel_partition(aux: AuxLossModel, X: np.ndarray, r: np.ndarray, m: int,
                          features=None, n_clusters: int = 8, n_landmarks: int = 500,
                          seed: int = 0):
    """B2: partition the leaf geometry, then score each part with the real loss.

    The auxiliary model supplies the representation and the fixed model's loss
    decides which part is weak, so the benchmark cannot be strong merely because
    the loss model fits well.

    A spectral cluster is not a box, and it is not a set the box class can
    express: on the credit-default task the cluster found at 64 parts has
    contrast 3.6 while the best box reaches 1.2, and mapping that cluster onto
    the admissible box centered on it returns 0.4. The cluster is nevertheless
    deployable, because a row can be routed to the cluster whose mean leaf-kernel
    vector it most resembles, so B2 is reported in its own declared class with
    that router as the frozen rule, and separately mapped into the box class
    where a matched-class comparison is wanted. The router reproduces the
    partition only approximately, so its agreement with the fitted labels is
    returned and belongs beside any result that rests on it.
    """
    from aiv._vendor.kernel_xgb.weakness import spectral_cluster_nystrom
    from aiv.weak_kernel import LeafKernelRouter

    Z = aux.phi(X)
    labels, _, _ = spectral_cluster_nystrom(
        Z, n_clusters=n_clusters, m_landmarks=min(n_landmarks, Z.shape[0]), random_state=seed
    )
    labels = np.asarray(labels, dtype=int)
    best, best_mean, best_id = None, -np.inf, None
    for lid in np.unique(labels):
        idx = np.flatnonzero(labels == lid)
        if len(idx) >= m and r[idx].mean() > best_mean:
            best, best_mean, best_id = idx, float(r[idx].mean()), int(lid)
    if best is None:
        raise ValueError("no cluster reached the required support")
    feats = list(features) if features is not None else list(range(X.shape[1]))
    router = LeafKernelRouter.from_labels(aux.kernel, feats, Z, labels, best_id)
    return best, labels, router
