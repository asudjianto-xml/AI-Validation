"""Fixed-support region constructor.

A candidate is a center and a shape, and the constructor inflates the region
until it holds exactly `m` discovery observations. Holding support fixed rather
than bounding it below has three consequences the search design depends on: the
support constraint is met by construction, the weakness contrast is comparable
across candidates without a size correction, and one evaluation costs the same
number of loss reads for every method.

Two metrics are available. The box metric is a weighted Chebyshev distance, so a
region is an axis-aligned box whose face half-widths are proportional to the
shape vector; the cube is the uniform-shape special case. The ball metric is
standardized Euclidean and ignores the shape. Boxes and cubes are expressible as
conjunctions of interval predicates; balls are not, and are reported as a
separate declared class rather than as a member of the box family.

Grid snapping belongs to freezing, not to construction: snapping a face outward
can only admit observations, so a snapped region holds at least `m` rather than
exactly `m`. Discovery therefore runs on the exact-support region and `freeze()`
reports the realized support of the auditable rule alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class Candidate:
    """Center in standardized coordinates and a shape vector on the simplex."""

    z: np.ndarray
    w: np.ndarray

    @staticmethod
    def cube(z) -> "Candidate":
        z = np.asarray(z, float)
        return Candidate(z, np.full(z.shape, 1.0 / len(z)))

    def as_tuple(self) -> np.ndarray:
        """Flat coordinate vector used by surrogates and acquisition."""
        return np.concatenate([self.z, self.w])


@dataclass(frozen=True)
class Region:
    """An exact-support region: which discovery rows are inside, and the
    inflation radius that put them there."""

    candidate: Candidate
    member_idx: np.ndarray
    gamma: float
    metric: str

    @property
    def m(self) -> int:
        return len(self.member_idx)


@dataclass(frozen=True)
class FrozenRegion:
    """A fixed subset of input space, in original units.

    `lo`/`hi` are box faces after outward snapping to the bin grid; a face at
    -inf or +inf leaves that feature unconstrained. For a ball, `center` and
    `radius` are used instead and the faces are the bounding box. `support_disc`
    is how many discovery rows the snapped rule actually holds, which is at
    least the `m` the constructor targeted.
    """

    features: tuple
    metric: str
    lo: np.ndarray
    hi: np.ndarray
    center: np.ndarray | None
    radius: float | None
    mu: np.ndarray
    sd: np.ndarray
    support_disc: int
    m_target: int

    def mask(self, df) -> np.ndarray:
        X = df[list(self.features)].to_numpy(dtype=float)
        if self.metric == "ball":
            Z = (X - self.mu) / self.sd
            d = np.sqrt(((Z - self.center) ** 2).sum(1))
            return d <= self.radius
        return np.all((X >= self.lo) & (X <= self.hi), axis=1)


class FixedSupportConstructor:
    """Maps a candidate to the region holding exactly `m` of the reference rows.

    `Xs` is the standardized reference matrix (the discovery covariates). The
    same constructor instance must be used throughout one discovery run, since
    the inflation radius is defined relative to these rows.
    """

    def __init__(self, Xs: np.ndarray, m: int, metric: str = "box"):
        if metric not in ("box", "ball"):
            raise ValueError("metric must be 'box' or 'ball'")
        Xs = np.asarray(Xs, dtype=float)
        if Xs.ndim != 2:
            raise ValueError("Xs must be a two-dimensional standardized matrix")
        if not 1 <= m <= len(Xs):
            raise ValueError("m must be between 1 and the number of reference rows")
        self.Xs = Xs
        self.n, self.p = Xs.shape
        self.m = int(m)
        self.metric = metric

    def _keys(self, cand: Candidate):
        """Inflation radius per row, and the Euclidean tie-break.

        The box radius is a maximum over coordinates, so it is decided by the
        single coordinate with the smallest shape weight. On data with discrete
        or heavily repeated features many rows then share a radius exactly: on
        the credit-default task a typical candidate has tens of rows tied at the
        boundary, so ``the m nearest rows`` does not by itself define a region,
        and two tie-breakings of the same candidate give contrasts differing by
        as much as 0.12. The tie-break is therefore declared rather than left to
        the sort: rows are ordered by radius, and within a radius by standardized
        Euclidean distance to the center, which is the ordering in which a face
        pushed outward admits them.
        """
        d = self.Xs - cand.z
        euclid = np.sqrt((d * d).sum(1))
        if self.metric == "ball":
            return euclid, euclid
        w = np.maximum(np.asarray(cand.w, float), _EPS)
        return np.abs(d / w).max(1), euclid

    def build(self, cand: Candidate) -> Region:
        u, euclid = self._keys(cand)
        idx = np.lexsort((euclid, u))[: self.m]
        gamma = float(u[idx].max())
        return Region(cand, np.sort(idx), gamma, self.metric)

    def build_batch(self, cands, device=None, chunk: int = 512):
        """Regions for many candidates at once.

        Uses torch on the GPU when available, since the cost is one dense
        (candidates x rows) distance matrix and nothing else. Falls back to the
        same computation in numpy.
        """
        cands = list(cands)
        try:
            import torch
        except ImportError:
            return [self.build(c) for c in cands]
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Double precision and a stable two-key sort, so that the batch path
        # resolves boundary ties exactly as `build` does. A top-k on the radius
        # alone would break them by whatever order the kernel happened to
        # produce, which is not reproducible and not the declared region.
        X = torch.as_tensor(self.Xs, dtype=torch.float64, device=dev)
        Z = torch.as_tensor(np.stack([c.z for c in cands]), dtype=torch.float64, device=dev)
        W = torch.as_tensor(np.stack([np.maximum(c.w, _EPS) for c in cands]),
                            dtype=torch.float64, device=dev)
        out = []
        for s in range(0, len(cands), chunk):
            zb, wb = Z[s : s + chunk], W[s : s + chunk]
            diff = X.unsqueeze(0) - zb.unsqueeze(1)              # (b, n, p)
            euclid = diff.pow(2).sum(-1).sqrt()
            u = euclid if self.metric == "ball" else (diff / wb.unsqueeze(1)).abs().amax(-1)
            # Rows strictly inside the m-th radius are all members; the places
            # left over go to the tied rows nearest the center in Euclidean
            # distance. Sorting every row would give the same answer at O(n log
            # n) per candidate, and the tied set is what actually needs ordering.
            gam = torch.kthvalue(u, self.m, dim=1).values
            inside = u < gam.unsqueeze(1)
            key = torch.where(inside, torch.full_like(euclid, -float("inf")),
                              torch.where(u == gam.unsqueeze(1), euclid,
                                          torch.full_like(euclid, float("inf"))))
            idx = torch.topk(key, self.m, dim=1, largest=False).indices
            idx_np, gam_np = idx.cpu().numpy(), gam.cpu().numpy()
            for b in range(idx_np.shape[0]):
                out.append(Region(cands[s + b], np.sort(idx_np[b]), float(gam_np[b]), self.metric))
        return out

    def freeze(self, region: Region, std, grid, X_disc_raw: np.ndarray) -> FrozenRegion:
        """Convert a region to a fixed subset of input space.

        Box faces are taken from the realized members, widened to the inflation
        radius, then snapped outward to bin edges so that the rule reads as a
        conjunction of interval predicates on the declared grid. A face that
        reaches the outermost edge is reported as unconstrained. The ball keeps
        its center and radius, since no interval rule reproduces it.
        """
        cand = region.candidate
        if region.metric == "ball":
            lo_s = cand.z - region.gamma
            hi_s = cand.z + region.gamma
        else:
            w = np.maximum(cand.w, _EPS)
            lo_s = cand.z - region.gamma * w
            hi_s = cand.z + region.gamma * w
        lo_raw, hi_raw = std.inverse(lo_s), std.inverse(hi_s)
        lo = np.array([grid.snap_low(j, lo_raw[j]) for j in range(len(lo_raw))])
        hi = np.array([grid.snap_high(j, hi_raw[j]) for j in range(len(hi_raw))])

        if region.metric == "ball":
            Z = (X_disc_raw - std.mu) / std.sd
            support = int((np.sqrt(((Z - cand.z) ** 2).sum(1)) <= region.gamma).sum())
            center, radius = cand.z.copy(), float(region.gamma)
        else:
            support = int(np.all((X_disc_raw >= lo) & (X_disc_raw <= hi), axis=1).sum())
            center, radius = None, None

        return FrozenRegion(
            features=std.features, metric=region.metric, lo=lo, hi=hi,
            center=center, radius=radius, mu=std.mu, sd=std.sd,
            support_disc=support, m_target=region.m,
        )


def sample_candidates(Xs: np.ndarray, n_cand: int, rng, shape: str = "box",
                      dirichlet_alpha: float = 1.0):
    """Admissible random candidates: centers on observed rows, shapes from a
    Dirichlet centered on the uniform shape.

    Centers are drawn from actual observations rather than from a bounding box,
    so every candidate sits on empirical support. `shape='cube'` fixes the shape
    to uniform, which is the nested subfamily with p free parameters.
    """
    Xs = np.asarray(Xs, float)
    n, p = Xs.shape
    rows = rng.integers(0, n, size=n_cand)
    if shape == "cube":
        W = np.full((n_cand, p), 1.0 / p)
    elif shape == "box":
        W = rng.dirichlet(np.full(p, dirichlet_alpha), size=n_cand)
    else:
        raise ValueError("shape must be 'box' or 'cube'")
    return [Candidate(Xs[r].copy(), W[i]) for i, r in enumerate(rows)]
