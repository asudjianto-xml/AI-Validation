"""Confirmation of a frozen region using independent held-out observations."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InflationResult:
    n: int
    statistic: float
    ci_low: float
    ci_high: float
    passed: bool | None
    detail: str


def confirm_loss_inflation(loss, membership, *, threshold=0.0, alpha=0.05,
                           n_boot=2000, seed=0, min_region=20):
    """Paired row-bootstrap percentile interval for region/population loss - 1.

    Freeze membership before supplying held-out losses. Rows must be independent;
    clustered observations need a cluster-level resampling procedure instead.
    Each replicate resamples losses and membership together, retaining the shared
    denominator. Bootstrap replicates approximate the sampling distribution;
    they are not new observations for a confidence interval on their mean.
    This is a single-region assessment, without portfolio multiplicity control.
    """
    loss = np.asarray(loss, dtype=float)
    membership = np.asarray(membership)
    if loss.ndim != 1 or membership.shape != loss.shape or membership.dtype != bool:
        raise ValueError("loss and boolean membership must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(loss)) or np.any(loss < 0):
        raise ValueError("losses must be finite and nonnegative")
    if not 0 < alpha < 1 or n_boot < 100 or min_region < 2:
        raise ValueError("require 0 < alpha < 1, at least 100 replicates and min_region >= 2")
    n = int(membership.sum())
    if n < min_region or not len(loss) or loss.mean() <= 0:
        return InflationResult(n, np.nan, np.nan, np.nan, None,
                               "inconclusive: insufficient region support or zero baseline loss")
    point = float(loss[membership].mean() / loss.mean() - 1)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(len(loss), size=len(loss))
        mask = membership[idx]
        sampled = loss[idx]
        if mask.any() and sampled.mean() > 0:
            boot.append(sampled[mask].mean() / sampled.mean() - 1)
    if len(boot) != n_boot:
        return InflationResult(n, point, np.nan, np.nan, None,
                               "inconclusive: undefined bootstrap ratios")
    lo, hi = map(float, np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    passed = None if threshold is None else lo > threshold
    return InflationResult(n, point, lo, hi, passed,
                           f"inflation={point:.3f} percentile CI=[{lo:.3f},{hi:.3f}]")
