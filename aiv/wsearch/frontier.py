"""The intensity-coverage frontier over candidate weak regions.

A single region is the wrong deliverable. Regions found by different methods sit
at different points of a genuine trade-off, and which one is wanted depends on
the decision: remediation with limited capacity argues for a small severe
region, a retraining decision for a broad one. This module reduces a set of
candidates to the ones that are not dominated, so the choice is made explicitly
rather than by whoever fixed the support floor.

The two axes are lift and coverage, and the choice of axes matters. Contrast
against support does not work, because the contrast is a ratio against the
region's complement, so shrinking the complement inflates it without bound: on
the credit-default task a region holding 97% of the sample reached a contrast of
56 purely because the 76 rows left outside it were unusually easy. Both axes are
then maximised together and nothing is dominated. Lift is measured against the
overall mean instead, so it is bounded and falls as the region grows, while
coverage rises, and the two genuinely compete.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionPoint:
    """One candidate region, measured on whichever split is supplied."""

    label: str
    lift: float            # mean loss in region / overall mean loss
    coverage: float        # share of the sample's total excess loss the region holds
    support: int
    n_conditions: float    # readable conditions in the frozen rule; inf for a kernel rule

    def dominates(self, other: "RegionPoint") -> bool:
        return (self.lift >= other.lift and self.coverage >= other.coverage
                and (self.lift > other.lift or self.coverage > other.coverage))


def lift_coverage(losses, membership, baseline: float | None = None):
    """Intensity and coverage of a region.

    Lift is the region's mean loss over the overall mean, so a lift of one is an
    average region and the quantity cannot run away as the complement shrinks.
    Coverage is the share of the sample's total excess loss -- loss above the
    overall mean -- that falls inside the region, which is what says how much of
    the problem this region accounts for.
    """
    r = np.asarray(losses, dtype=float)
    mask = np.asarray(membership, dtype=bool)
    if r.ndim != 1 or mask.shape != r.shape:
        raise ValueError("losses and boolean membership must be aligned one-dimensional arrays")
    base = float(r.mean()) if baseline is None else float(baseline)
    if base <= 0:
        raise ValueError("baseline loss must be positive")
    total_excess = np.maximum(r - base, 0.0).sum()
    if not mask.any() or total_excess <= 0:
        return float("nan"), float("nan")
    return float(r[mask].mean() / base), float(np.maximum(r[mask] - base, 0.0).sum() / total_excess)


def pareto_front(points):
    """The candidates no other candidate beats on both axes, sorted by lift."""
    pts = [p for p in points if np.isfinite(p.lift) and np.isfinite(p.coverage)]
    front = [a for a in pts if not any(b.dominates(a) for b in pts)]
    return sorted(front, key=lambda p: -p.lift)


def weakness_frontier(losses, regions, n_conditions=None, min_support: int = 40,
                      baseline: float | None = None):
    """Frontier over named candidate regions.

    `regions` maps a label to a boolean membership array on the evaluation split;
    apply the frozen rules to untouched data before calling, so the frontier is
    an out-of-sample one. `n_conditions` gives the number of readable predicates
    behind each rule, used only for reporting: a rule sitting on the frontier
    with one condition is a different proposition from a kernel rule sitting next
    to it, even though the two axes cannot tell them apart.
    """
    n_conditions = n_conditions or {}
    pts = []
    for label, mask in regions.items():
        mask = np.asarray(mask, dtype=bool)
        if int(mask.sum()) < min_support:
            continue
        lift, cov = lift_coverage(losses, mask, baseline)
        pts.append(RegionPoint(label, lift, cov, int(mask.sum()),
                               float(n_conditions.get(label, np.inf))))
    return pareto_front(pts), pts
