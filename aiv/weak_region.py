"""Weak-region discovery by evolving centroids over the residual field.

Inner array: centroids in standardized feature space, seeded by k-means, mutated
by an evolution strategy. The evaluator uses the model's per-sample
loss on the discovery data. This compact scalar search returns a ranked list;
it does not construct the full Pareto selection record or retain proposal lineage. A centroid's weakness is the mean residual of
its neighborhood; the evolutionary strategy mutates the centroids until they sit
on the weakest neighborhoods, which are the weak regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class Standardizer:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "Standardizer":
        mu = X.mean(0).to_numpy()
        sd = X.std(0).replace(0, 1.0).to_numpy()
        return cls(mu, sd)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return (X.to_numpy() - self.mu) / self.sd


def _neighborhood(centroid: np.ndarray, Xs: np.ndarray, resid: np.ndarray, m: int):
    """Mean residual (weakness) of the m nearest samples to a centroid.

    A neighborhood cannot be larger than the evaluated sample, so the requested
    size is capped at the number of available rows."""
    d = np.sqrt(((Xs - centroid) ** 2).sum(1))
    mm = min(m, len(d))
    idx = np.argpartition(d, mm - 1)[:mm]
    return float(resid[idx].mean()), idx


@dataclass
class WeakRegion:
    centroid: np.ndarray          # standardized feature space
    weakness: float               # mean residual of its neighborhood
    profile: dict                 # feature -> original-scale centroid value
    member_index: np.ndarray      # validation rows in the neighborhood


class CentroidRouter:
    """Route a sample by its nearest centroid among a discovered set. The weak
    segment is the union of the cells of the top-weakness centroids."""

    def __init__(self, features, centroids_orig, weak_ids, mu, sd):
        self.features = list(features)
        self.mu = np.asarray(mu)
        self.sd = np.asarray(sd)
        self.C = (np.asarray(centroids_orig) - self.mu) / self.sd
        self.weak_ids = list(weak_ids)

    def weak_mask(self, df) -> np.ndarray:
        Xs = (df[self.features].to_numpy() - self.mu) / self.sd
        d = ((Xs[:, None, :] - self.C[None, :, :]) ** 2).sum(2)
        nearest = d.argmin(1)
        return np.isin(nearest, self.weak_ids)


class RadiusRouter:
    """The weak segment is a ball around one discovered centroid, sized to a
    target count. Deployable: a sample is in the weak segment iff its standardized
    distance to the centroid is below the radius."""

    def __init__(self, features, centroid_orig, radius, mu, sd):
        self.features = list(features)
        self.mu = np.asarray(mu)
        self.sd = np.asarray(sd)
        self.c = (np.asarray(centroid_orig) - self.mu) / self.sd
        self.r = float(radius)

    def weak_mask(self, df) -> np.ndarray:
        Xs = (df[self.features].to_numpy() - self.mu) / self.sd
        d = np.sqrt(((Xs - self.c) ** 2).sum(1))
        return d < self.r


def radius_router_for_count(system, region, mu, sd, split: str, count: int) -> RadiusRouter:
    """Radius that captures `count` samples of `split` around the region centroid."""
    feats = system.features
    c_std = (np.array([region.profile[f] for f in feats]) - np.asarray(mu)) / np.asarray(sd)
    X = (system.splits[split][feats].to_numpy() - np.asarray(mu)) / np.asarray(sd)
    d = np.sqrt(((X - c_std) ** 2).sum(1))
    r = float(np.partition(d, count)[count])
    return RadiusRouter(feats, [region.profile[f] for f in feats], r, mu, sd)


def weak_membership(system, regions, mu, sd, top_n: int = 1) -> CentroidRouter:
    """Build a router from discovered regions (ranked by weakness). The top_n
    weakest centroids define the weak segment."""
    C = [[r.profile[f] for f in system.features] for r in regions]
    return CentroidRouter(system.features, C, list(range(top_n)), mu, sd)


def discover_weak_regions(
    system, split: str = "discovery", k: int = 10, m: int = 150,
    gens: int = 40, sigma0: float = 0.6, seed: int = 0, features: list | None = None,
) -> tuple[list[WeakRegion], float]:
    """Seed centroids with k-means, then mutate each with a (1+1)-ES to maximize
    its neighborhood residual. Returns weak regions ranked by weakness, and the
    baseline mean residual.

    `features` restricts the search to a declared subset of the system's
    features; the excluded features are held at their population mean, so the
    search cannot move along a dimension the representation does not include.
    Defaults to every feature, which is the ordinary (unrestricted) search."""
    feats = list(features) if features is not None else system.features
    df = system.splits[split]
    std = Standardizer.fit(df[feats])
    Xs = std.transform(df[feats])
    resid = system.row_loss(df)  # evidence for the weakness evaluator
    base = float(resid.mean())

    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    C = km.cluster_centers_.copy()
    weak = np.array([_neighborhood(c, Xs, resid, m)[0] for c in C])

    for g in range(gens):
        sigma = sigma0 * (1 - g / gens) + 0.05
        for j in range(k):
            cand = C[j] + rng.normal(0.0, sigma, size=C.shape[1])
            w, _ = _neighborhood(cand, Xs, resid, m)
            if w > weak[j]:
                C[j], weak[j] = cand, w

    order = np.argsort(-weak)
    regions = []
    for j in order:
        _, idx = _neighborhood(C[j], Xs, resid, m)
        original = C[j] * std.sd + std.mu
        profile = {f: float(v) for f, v in zip(feats, original)}
        regions.append(WeakRegion(C[j], float(weak[j]), profile, df.index.to_numpy()[idx]))
    return regions, base


def hill_climb_weak_regions(
    system, split: str = "discovery", k: int = 10, m: int = 150,
    step: float = 0.5, max_iters: int = 40, seed: int = 0,
) -> tuple[list[WeakRegion], float, int]:
    """Seed centroids with k-means, then move each by best-improvement hill
    climbing: one factor at a time, in standardized units.

    At every iteration, both directions of every feature are tried as a
    one-factor neighbor; the centroid moves to the best strictly-improving
    neighbor found, or stops if none improves (a local optimum under this
    step size and neighborhood, not a claim of global optimality). Same
    k-means seeding as `discover_weak_regions`, so the two share a starting
    population and differ only in how they move it. Returns weak regions
    ranked by weakness, the baseline mean residual, and the total number of
    neighbor evaluations spent."""
    df = system.splits[split]
    std = Standardizer.fit(df[system.features])
    Xs = std.transform(df[system.features])
    resid = system.row_loss(df)
    base = float(resid.mean())

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    C = km.cluster_centers_.copy()
    weak = np.array([_neighborhood(c, Xs, resid, m)[0] for c in C])
    d = C.shape[1]
    evals = 0

    for j in range(k):
        for _ in range(max_iters):
            best_w, best_c = weak[j], None
            for f in range(d):
                for sign in (+1.0, -1.0):
                    cand = C[j].copy()
                    cand[f] += sign * step
                    w, _ = _neighborhood(cand, Xs, resid, m)
                    evals += 1
                    if w > best_w:
                        best_w, best_c = w, cand
            if best_c is None:      # no improving neighbor: local stop
                break
            C[j], weak[j] = best_c, best_w

    order = np.argsort(-weak)
    regions = []
    for j in order:
        _, idx = _neighborhood(C[j], Xs, resid, m)
        original = C[j] * std.sd + std.mu
        profile = {f: float(v) for f, v in zip(system.features, original)}
        regions.append(WeakRegion(C[j], float(weak[j]), profile, df.index.to_numpy()[idx]))
    return regions, base, evals
