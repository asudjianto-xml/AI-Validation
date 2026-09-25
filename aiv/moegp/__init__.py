"""Mixture of Gaussian-process experts with an XGBoost leaf kernel.

The object under validation is a monotone gradient-boosted model. Its leaf
partition defines a covariance, K(x, x') = z(x)^T z(x') / T, and the mixture
places one Gaussian process per gate component on that covariance, each
modeling a correction to the base model's logit. The gate is searched by an
evolution strategy; the experts are fitted in closed form given the gate.

The gate is at present the hard nearest-centroid gate of k-means, whose
parameter vector is the centers alone. The Gaussian-mixture gate in `gate` is
implemented and not yet searched; see its module docstring.
"""
from __future__ import annotations

from aiv.moegp.experts import MixtureFit, device_of, fit_mixture, mixture_proba, to_tensor
from aiv.moegp.gate import (
    CentroidParams,
    GateParams,
    Standardizer,
    kmeans_centroids,
    kmeans_init,
)
from aiv.moegp.kernel import (
    BASE_PARAMS,
    LeafMap,
    PrimalBasis,
    base_logit,
    fit_base,
    monotone_string,
    primal_basis,
)
from aiv.moegp.search import Fold, SearchResult, es_search, fit_and_score, make_folds, objective

__all__ = [
    "BASE_PARAMS", "fit_base", "base_logit", "monotone_string", "LeafMap",
    "PrimalBasis", "primal_basis",
    "Standardizer", "CentroidParams", "kmeans_centroids", "GateParams", "kmeans_init",
    "MixtureFit", "fit_mixture", "mixture_proba", "to_tensor", "device_of",
    "Fold", "make_folds", "fit_and_score", "objective", "es_search", "SearchResult",
]
