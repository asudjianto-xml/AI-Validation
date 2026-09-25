"""Declared geometry for weakness search: standardization and a bin grid.

Both are estimated on the training split alone. They are part of the declared
region representation, so they must not move when discovery or confirmation data
changes -- a frozen region whose scaling was refit on new data is a different
subset of input space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Standardizer:
    """Center and scale, fitted once on training covariates."""

    features: tuple
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame, features=None) -> "Standardizer":
        feats = tuple(features) if features is not None else tuple(X.columns)
        mu = X[list(feats)].to_numpy(dtype=float).mean(0)
        sd = X[list(feats)].to_numpy(dtype=float).std(0)
        sd = np.where(sd <= 0, 1.0, sd)
        return cls(feats, mu, sd)

    def transform(self, X) -> np.ndarray:
        arr = X[list(self.features)].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        return (arr - self.mu) / self.sd

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, float) * self.sd + self.mu


@dataclass(frozen=True)
class BinGrid:
    """Per-feature bin edges, from training quantiles.

    A feature with fewer distinct training values than requested bins keeps its
    distinct values as edges, so a binary or small-count feature is not given
    spurious resolution. Edges are open at both ends (-inf, +inf) so that every
    observation falls inside, and a frozen face snapped to an outer edge means
    the feature is unconstrained on that side.
    """

    features: tuple
    edges: tuple  # one increasing float array per feature, including +-inf

    @classmethod
    def fit(cls, X: pd.DataFrame, features=None, n_bins: int = 20) -> "BinGrid":
        feats = tuple(features) if features is not None else tuple(X.columns)
        edges = []
        for f in feats:
            v = X[f].to_numpy(dtype=float)
            uniq = np.unique(v)
            if len(uniq) <= n_bins:
                inner = (uniq[:-1] + uniq[1:]) / 2.0
            else:
                q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
                inner = np.unique(np.quantile(v, q))
            edges.append(np.concatenate([[-np.inf], inner, [np.inf]]))
        return cls(feats, tuple(edges))

    def n_bins(self, j: int) -> int:
        return len(self.edges[j]) - 1

    def snap_low(self, j: int, value: float) -> float:
        """Largest grid edge at or below `value` (outward snap of a lower face)."""
        e = self.edges[j]
        k = int(np.searchsorted(e, value, side="right")) - 1
        return float(e[max(k, 0)])

    def snap_high(self, j: int, value: float) -> float:
        """Smallest grid edge at or above `value` (outward snap of an upper face)."""
        e = self.edges[j]
        k = int(np.searchsorted(e, value, side="left"))
        return float(e[min(k, len(e) - 1)])
