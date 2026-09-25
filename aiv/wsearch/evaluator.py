"""The common weakness evaluator, its cost meter and the overlap covariance.

Every method in the framework measures a candidate through this one object, so
that a comparison between a one-shot benchmark and an adaptive searcher reflects
the search rather than the measurement. The evaluator reports the in-region
against out-of-region loss contrast, the secondary criteria, the delete-one-fold
replicates that describe how sensitive the contrast is to the discovery
evidence, and the number of loss reads the measurement consumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class TouchMeter:
    """Loss reads, the common cost currency.

    A method is charged one unit per observation whose loss value it reads.
    Fitting an auxiliary loss model reads every discovery loss once as a target;
    evaluating a candidate reads the `m` in-region losses, the complement mean
    coming from a grand total cached at a one-time cost of `n`. Counting only
    candidate evaluations would place a one-shot benchmark at the origin and an
    exhaustive lattice search there with it.
    """

    reads: int = 0
    by_tag: dict = field(default_factory=dict)

    def charge(self, k: int, tag: str = "eval") -> None:
        self.reads += int(k)
        self.by_tag[tag] = self.by_tag.get(tag, 0) + int(k)

    def reset(self) -> None:
        self.reads = 0
        self.by_tag = {}


@dataclass(frozen=True)
class Measurement:
    m: int
    contrast: float          # M = mean(r | in) / mean(r | out) - 1
    mean_in: float
    mean_out: float
    sd_in: float             # S, within-region loss dispersion
    tail_in: float           # T_tau, exceedance rate in region (nan if no tau)
    var_in: float            # within-region loss variance, feeds the noise model


@dataclass(frozen=True)
class Replicates:
    values: np.ndarray       # delete-one-fold contrasts
    se: float                # grouped jackknife standard error
    floor: float             # minimum replicate contrast


class WeaknessEvaluator:
    """Measures a region against the discovery losses.

    `folds` assigns every discovery row to one of K Twinning folds and enables
    the delete-one-fold replicates. `tau` is the tail threshold, which must come
    from the training split rather than from these losses.
    """

    def __init__(self, r: np.ndarray, folds: np.ndarray | None = None,
                 tau: float | None = None, meter: TouchMeter | None = None):
        r = np.asarray(r, dtype=float)
        if r.ndim != 1:
            raise ValueError("losses must be a one-dimensional array")
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise ValueError("losses must be finite and nonnegative")
        self.r = r
        self.n = len(r)
        self.total = float(r.sum())
        self.tau = tau
        self.meter = meter or TouchMeter()
        self.meter.charge(self.n, "grand_total")

        self.folds = None
        if folds is not None:
            folds = np.asarray(folds, dtype=int)
            if folds.shape != r.shape:
                raise ValueError("fold labels must align with the losses")
            self.folds = folds
            self.K = int(folds.max()) + 1
            self.fold_n = np.bincount(folds, minlength=self.K).astype(float)
            self.fold_sum = np.bincount(folds, weights=r, minlength=self.K)
            if np.any(self.fold_n < 2):
                raise ValueError("every fold needs at least two observations")

    # -- primary measurement ------------------------------------------------

    def measure(self, idx: np.ndarray) -> Measurement:
        idx = np.asarray(idx, dtype=int)
        m = len(idx)
        if m < 2 or m >= self.n:
            raise ValueError("region must hold at least two rows and leave a complement")
        r_in = self.r[idx]
        self.meter.charge(m, "eval")
        s_in = float(r_in.sum())
        mean_in = s_in / m
        mean_out = (self.total - s_in) / (self.n - m)
        if mean_out <= 0:
            raise ValueError("complement mean loss is not positive")
        var_in = float(r_in.var(ddof=1))
        tail = float((r_in > self.tau).mean()) if self.tau is not None else float("nan")
        return Measurement(
            m=m, contrast=mean_in / mean_out - 1.0, mean_in=mean_in,
            mean_out=mean_out, sd_in=float(np.sqrt(var_in)), tail_in=tail, var_in=var_in,
        )

    def contrast(self, idx: np.ndarray) -> float:
        return self.measure(idx).contrast

    # -- replicate measurement ----------------------------------------------

    def replicates(self, idx: np.ndarray) -> Replicates:
        """Delete-one-fold contrasts and the grouped jackknife standard error.

        The replicate deletes a fold from the whole discovery sample rather than
        restricting to it, so the region keeps m(K-1)/K of its members instead
        of m/K. Numerator and denominator are computed on the same replicate
        sample, which is what makes the jackknife applicable to a ratio.
        """
        if self.folds is None:
            raise ValueError("replicates need fold labels")
        idx = np.asarray(idx, dtype=int)
        m = len(idx)
        r_in = self.r[idx]
        s_in = float(r_in.sum())
        f_in = self.folds[idx]
        m_k = np.bincount(f_in, minlength=self.K).astype(float)
        s_k = np.bincount(f_in, weights=r_in, minlength=self.K)
        self.meter.charge(m, "replicate")

        m_rep = m - m_k                                  # in-region rows kept
        n_rep = self.n - self.fold_n                     # total rows kept
        sin_rep = s_in - s_k
        sout_rep = (self.total - self.fold_sum) - sin_rep
        out_rep = n_rep - m_rep
        ok = (m_rep >= 2) & (out_rep >= 2)
        vals = np.full(self.K, np.nan)
        denom = np.where(ok & (sout_rep > 0), sout_rep / np.where(out_rep > 0, out_rep, 1), np.nan)
        num = np.where(ok, sin_rep / np.where(m_rep > 0, m_rep, 1), np.nan)
        vals = num / denom - 1.0
        good = vals[np.isfinite(vals)]
        if len(good) < 2:
            return Replicates(vals, float("nan"), float("nan"))
        K = len(good)
        se = float(np.sqrt((K - 1) / K * ((good - good.mean()) ** 2).sum()))
        return Replicates(vals, se, float(good.min()))


# -- overlap-induced noise covariance ---------------------------------------


def overlap_fraction(members, n: int) -> np.ndarray:
    """Pairwise |A_t and A_t'| / m over a list of membership index arrays.

    Candidates built by a nearest-neighbor or inflation rule share most of their
    members when their centers are close, so their sampling errors are largely
    the same error. This matrix is what makes that shared error explicit.
    """
    members = [np.asarray(a, dtype=int) for a in members]
    m = len(members[0])
    if any(len(a) != m for a in members):
        raise ValueError("fixed-support regions must all hold the same number of rows")
    T = len(members)
    rows = np.repeat(np.arange(T), m)
    cols = np.concatenate(members)
    Z = sp.csr_matrix((np.ones(T * m), (rows, cols)), shape=(T, n))
    return np.asarray((Z @ Z.T).todense()) / m


def resampling_covariance(members, n: int, s2: float, mean_out: float) -> np.ndarray:
    """Covariance of the measured contrasts under resampling of the discovery data.

    With r_i = rho(x_i) + eps_i and Var(eps_i) = s2, the in-region means satisfy
    Cov(mean_t, mean_t') = s2 |A_t and A_t'| / m^2, and the delta method carries
    that through the ratio, dividing by the squared complement mean. The shared
    denominator contributes a further term common to all candidates.

    This is evidentiary uncertainty, not measurement noise. Given the fixed
    discovery losses, the contrast of a candidate is a deterministic function of
    its coordinates: evaluating the same candidate twice returns the same number.
    The quantity here describes how the whole archive would move if the discovery
    sample were redrawn, which is the question the delete-one-fold jackknife
    answers empirically, and it belongs in promotion eligibility rather than in a
    surrogate likelihood. Supplying it as observation noise is not identifiable:
    it is proportional to the overlap-fraction matrix, which is itself a
    similarity kernel over candidates, so the marginal likelihood cannot separate
    it from the signal kernel and collapses the signal amplitude instead.
    """
    frac = overlap_fraction(members, n)
    m = len(np.asarray(members[0]))
    return s2 * frac / (m * mean_out ** 2)


def max_overlap(candidate, archive, n: int) -> np.ndarray:
    """Largest membership overlap of each candidate with any archive region.

    The operational cost of overlap is budget rather than calibration: a
    candidate sharing most of its members with one already measured returns
    almost the same contrast, so the evaluation buys almost nothing.
    """
    cand = [np.asarray(a, dtype=int) for a in candidate]
    arch = [np.asarray(a, dtype=int) for a in archive]
    if not arch:
        return np.zeros(len(cand))
    frac = overlap_fraction(cand + arch, n)
    return frac[: len(cand), len(cand):].max(axis=1)


def distinct_by_overlap(members, n: int, threshold: float = 0.5) -> np.ndarray:
    """Greedy selection of regions no two of which overlap beyond `threshold`.

    Used both to constrain acquisition, so that a proposal duplicating the
    archive is not evaluated, and to count distinct findings at promotion.
    Regions are considered in the order given, so pass them ranked by contrast.
    """
    members = [np.asarray(a, dtype=int) for a in members]
    frac = overlap_fraction(members, n)
    keep = []
    for i in range(len(members)):
        if not keep or frac[i, keep].max() <= threshold:
            keep.append(i)
    return np.array(keep, dtype=int)


def pooled_within_variance(measurements) -> float:
    """Pooled within-region loss variance over evaluated candidates, the s2 the
    noise covariance needs."""
    v = np.array([mm.var_in for mm in measurements], dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        raise ValueError("no finite within-region variances")
    return float(v.mean())
