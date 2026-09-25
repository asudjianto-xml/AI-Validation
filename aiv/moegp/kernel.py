"""The XGBoost leaf feature map used as the covariance of the expert processes.

The kernel is taken from the model under validation rather than fitted
alongside the mixture: a depth-2 monotone booster on the training split defines
the leaf partition, and the induced covariance
K(x, x') = z(x)^T z(x') / T is the fraction of trees in which the two rows fall
in the same leaf. Because the map z is explicit, finite and sparse -- one
one-hot block per tree, so at most four columns per tree at depth two -- the
processes can be fitted in the primal representation, and no Nystroem
approximation of K is needed.

The primal dimension is compressed by an eigendecomposition of the
feature-space second-moment matrix on the discovery split. Trees in a boosted
ensemble are highly correlated, so the spectrum concentrates and a few hundred
directions carry essentially all of the trace; the compression is a truncation
of the feature map, not an approximation of a kernel by landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from aiv._vendor.kernel_xgb.kernel import LeafKernel

BASE_PARAMS = dict(max_depth=2, n_estimators=200, learning_rate=0.05,
                   subsample=0.9, colsample_bytree=0.9, random_state=0)


def monotone_string(features, monotone: dict) -> str:
    """XGBoost's tuple-literal encoding of the per-feature constraint directions."""
    return "(" + ",".join(str(monotone.get(f, 0)) for f in features) + ")"


def fit_base(X, y, features, monotone: dict, params: dict | None = None):
    """The monotone booster under validation, in the configuration of the
    earlier mixture-of-experts comparison so the numbers stay commensurable."""
    p = dict(BASE_PARAMS if params is None else params)
    p["monotone_constraints"] = monotone_string(features, monotone)
    model = xgb.XGBClassifier(**p)
    model.fit(np.asarray(X, float), np.asarray(y).astype(int))
    return model


def base_logit(model, X) -> np.ndarray:
    """Raw margin of the base model, which enters each expert as a fixed offset."""
    return model.predict(np.asarray(X, float), output_margin=True).astype(np.float64)


class LeafMap:
    """Dense leaf feature map with K = Psi Psi^T and unit diagonal.

    The leaf index is built once over the union of all rows so that a leaf
    reached only by a held-out row still receives a column; this uses covariates
    alone and no outcomes.
    """

    def __init__(self, booster, X_vocab):
        self._lk = LeafKernel(booster)
        self._lk.fit(np.asarray(X_vocab, float))
        self.n_trees = int(self._lk.T)
        self.dim = int(self._lk.d_total)

    def transform(self, X) -> np.ndarray:
        Z = self._lk.transform(np.asarray(X, float))
        return np.asarray(Z.todense(), dtype=np.float64) / np.sqrt(float(self._lk.S))


@dataclass(frozen=True)
class PrimalBasis:
    """Truncated eigenbasis of the leaf feature map."""

    P: np.ndarray            # (dim, m) projection
    eigenvalues: np.ndarray  # (m,) retained second-moment eigenvalues
    kept_trace: float        # share of the total trace retained

    @property
    def m(self) -> int:
        return self.P.shape[1]

    def project(self, Psi: np.ndarray) -> np.ndarray:
        return np.asarray(Psi, float) @ self.P


def primal_basis(Psi_fit: np.ndarray, var_keep: float = 0.9995,
                 max_dim: int | None = None) -> PrimalBasis:
    """Eigendecompose Psi^T Psi on the fitting rows and keep the leading block.

    No centering: the constant direction lies in the span of the one-hot blocks
    and carries the offset the processes are allowed to use.
    """
    Psi_fit = np.asarray(Psi_fit, float)
    G = Psi_fit.T @ Psi_fit
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    w = np.maximum(w, 0.0)
    total = w.sum()
    keep = int(np.searchsorted(np.cumsum(w) / total, var_keep) + 1)
    keep = min(keep, len(w) if max_dim is None else max_dim)
    return PrimalBasis(P=V[:, :keep], eigenvalues=w[:keep],
                       kept_trace=float(w[:keep].sum() / total))
