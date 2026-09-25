"""Gates over the standardized covariates.

Two gates are implemented and only the first is searched at present.

`CentroidParams` is the hard nearest-centroid gate of k-means: gamma_h(x) is
one for the nearest center and zero otherwise. Its parameter vector holds the
centers alone, H*d coordinates, which is the smallest gate that the strategy
can search and the one the earlier centroid search used, so the comparison
isolates the change of expert family.

`GateParams` is the Gaussian-mixture gate, gamma_h(x) = pi_h N(x; mu_h,
Sigma_h) / sum_j pi_j N(x; mu_j, Sigma_j). With a shared isotropic covariance
this reduces to the softmax of a negative squared distance to the centers, so
the hard gate is its zero-temperature limit and the cluster-head model's
softmax gate is the special case with the mixing weights dropped and the width
fixed. Searching it adds H width coordinates in the spherical case and H*d in
the diagonal case, plus H mixing logits. It is left unsearched for now because
the centroid gate is cheaper to optimize; the mixture gate is the next thing to
try, and the machinery here is in place for it.

Both gates place the parameters in the raw standardized feature space rather
than in leaf space, so a component is reportable as a location in the
covariates. The strategy's step sizes are per block: centers are in standard
deviations, widths in log units and weights in logits, so a single step size
would move the widths far harder than the centers, and `BLOCK_SCALES` fixes
their relative sizes against one adapting global step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

BLOCK_SCALES = {"mu": 1.0, "log_s": 0.3, "log_pi": 0.3}
_LOG_2PI = float(np.log(2.0 * np.pi))


class Standardizer:
    """Covariate standardization shared by the gate across folds, so that a
    parameter vector means the same thing in every fold."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, float)
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale == 0] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean) / self.scale


# --- hard nearest-centroid gate -------------------------------------------

@dataclass(frozen=True)
class CentroidParams:
    """Nearest-centroid gate; the parameter vector is the centers alone."""

    mu: np.ndarray

    @property
    def n_components(self) -> int:
        return self.mu.shape[0]

    def pack(self) -> np.ndarray:
        return self.mu.ravel().copy()

    def scales(self) -> np.ndarray:
        return np.full(self.mu.size, BLOCK_SCALES["mu"])

    @classmethod
    def unpack(cls, vec: np.ndarray, H: int, d: int) -> "CentroidParams":
        return cls(mu=np.asarray(vec, float).reshape(H, d))

    def responsibilities(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.asarray(Xs, float)
        d2 = ((Xs[:, None, :] - self.mu[None, :, :]) ** 2).sum(axis=2)
        out = np.zeros_like(d2)
        out[np.arange(len(Xs)), np.argmin(d2, axis=1)] = 1.0
        return out


def kmeans_centroids(Xs: np.ndarray, n_components: int, seed: int = 0,
                     n_init: int = 20) -> CentroidParams:
    km = KMeans(n_clusters=n_components, n_init=n_init, random_state=seed).fit(np.asarray(Xs, float))
    return CentroidParams(mu=km.cluster_centers_)


# --- Gaussian-mixture gate (implemented, not yet searched) -----------------

@dataclass(frozen=True)
class GateParams:
    """Mixture parameters. `log_s` is (H,) spherical or (H, d) diagonal."""

    mu: np.ndarray
    log_s: np.ndarray
    log_pi: np.ndarray

    @property
    def n_components(self) -> int:
        return self.mu.shape[0]

    @property
    def diagonal(self) -> bool:
        return self.log_s.ndim == 2

    def pack(self) -> np.ndarray:
        return np.concatenate([self.mu.ravel(), self.log_s.ravel(), self.log_pi.ravel()])

    def scales(self) -> np.ndarray:
        return np.concatenate([
            np.full(self.mu.size, BLOCK_SCALES["mu"]),
            np.full(self.log_s.size, BLOCK_SCALES["log_s"]),
            np.full(self.log_pi.size, BLOCK_SCALES["log_pi"]),
        ])

    @classmethod
    def unpack(cls, vec: np.ndarray, H: int, d: int, diagonal: bool = False) -> "GateParams":
        vec = np.asarray(vec, float)
        n_s = H * d if diagonal else H
        mu = vec[: H * d].reshape(H, d)
        log_s = vec[H * d: H * d + n_s].reshape((H, d) if diagonal else (H,))
        return cls(mu=mu, log_s=log_s, log_pi=vec[H * d + n_s:])

    def log_responsibilities(self, Xs: np.ndarray, log_s_clip=(-4.0, 4.0)) -> np.ndarray:
        Xs = np.asarray(Xs, float)
        n, d = Xs.shape
        H = self.n_components
        s2 = np.exp(2.0 * np.clip(self.log_s, *log_s_clip))
        out = np.empty((n, H))
        for h in range(H):
            var_h = s2[h] if self.diagonal else np.full(d, s2[h])
            diff = Xs - self.mu[h]
            out[:, h] = -0.5 * (((diff * diff) / var_h).sum(axis=1)
                                + np.log(var_h).sum() + d * _LOG_2PI)
        out = out + (self.log_pi - self.log_pi.max())
        out -= out.max(axis=1, keepdims=True)
        return out - np.log(np.exp(out).sum(axis=1, keepdims=True))

    def responsibilities(self, Xs: np.ndarray) -> np.ndarray:
        return np.exp(self.log_responsibilities(Xs))


def kmeans_init(Xs: np.ndarray, n_components: int, seed: int = 0,
                diagonal: bool = False, n_init: int = 20,
                min_var: float = 1e-3) -> GateParams:
    """Moment initialization of the mixture gate from k-means: the cluster
    sizes give the mixing weights and the within-cluster scatter the widths."""
    Xs = np.asarray(Xs, float)
    km = KMeans(n_clusters=n_components, n_init=n_init, random_state=seed).fit(Xs)
    labels, mu = km.labels_, km.cluster_centers_
    counts = np.array([max((labels == h).sum(), 1) for h in range(n_components)], float)
    var = np.empty((n_components, Xs.shape[1]))
    for h in range(n_components):
        members = Xs[labels == h]
        var[h] = np.maximum(members.var(axis=0) if len(members) > 1 else 1.0, min_var)
    log_s = 0.5 * np.log(var) if diagonal else 0.5 * np.log(var.mean(axis=1))
    return GateParams(mu=mu, log_s=log_s, log_pi=np.log(counts / counts.sum()))
