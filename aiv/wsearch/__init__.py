"""Constrained weakness search: representation, evaluator and cost accounting.

The modules here implement the shared layer every search method in the framework
is measured through -- the declared geometry, the fixed-support region
constructor, the common weakness evaluator and its loss-read meter, the
delete-one-fold replicates, and the overlap-induced noise covariance that the
surrogate model needs. Search algorithms and benchmarks build on this layer and
are not part of it.
"""
from __future__ import annotations

from aiv.wsearch.constructor import (
    Candidate,
    FixedSupportConstructor,
    FrozenRegion,
    Region,
    sample_candidates,
)
from aiv.wsearch.evaluator import (
    Measurement,
    Replicates,
    TouchMeter,
    WeaknessEvaluator,
    distinct_by_overlap,
    max_overlap,
    resampling_covariance,
    overlap_fraction,
    pooled_within_variance,
)
from aiv.wsearch.frontier import RegionPoint, lift_coverage, pareto_front, weakness_frontier
from aiv.wsearch.grid import BinGrid, Standardizer
from aiv.wsearch.leafnn import (
    FrozenLeafRegion,
    LeafNeighborhoodConstructor,
    LeafRegion,
    exhaustive_contrasts,
    leaf_search,
    rank_by_proxy,
)
from aiv.wsearch.twinfolds import energy_distance, twin_folds

__all__ = [
    "BinGrid", "Standardizer",
    "Candidate", "Region", "FrozenRegion", "FixedSupportConstructor", "sample_candidates",
    "TouchMeter", "Measurement", "Replicates", "WeaknessEvaluator",
    "overlap_fraction", "resampling_covariance", "pooled_within_variance",
    "max_overlap", "distinct_by_overlap",
    "twin_folds", "energy_distance",
    "RegionPoint", "lift_coverage", "pareto_front", "weakness_frontier",
    "LeafRegion", "FrozenLeafRegion", "LeafNeighborhoodConstructor", "rank_by_proxy", "leaf_search", "exhaustive_contrasts",
]
