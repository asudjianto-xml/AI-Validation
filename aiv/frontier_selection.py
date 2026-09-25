"""Frontier ownership and parent selection over a candidate-by-condition table.

The proposal methods of Chapter 3 -- designed sweep, bandit and evolution --
produce the same object: failure scores $w_k(t)$ for candidate $k$ under
condition $t$, oriented so that larger means weaker. Retention and parent
selection read only that table, which is what allows one selection rule to
serve every proposer.

Ownership here is the scalar per-condition rule of \\cref{sec:frontier}: a
candidate owns a condition when its score is within a margin of that condition's
largest score. It is not local Pareto non-domination across criteria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ownership:
    """Owner sets, coverage shares and the retained candidates behind them."""

    owners: list          # owners[t]: indices of the candidates owning condition t
    coverage: np.ndarray  # C_k, the weighted share of conditions candidate k owns
    retained: np.ndarray  # candidate indices surviving the optional reduction


def _as_table(W) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("scores must be a candidate-by-condition table")
    if not np.all(np.isfinite(W)):
        raise ValueError("scores must be finite")
    return W


def _normalized_weights(weights, n_cond: int) -> np.ndarray:
    if weights is None:
        return np.full(n_cond, 1.0 / n_cond)
    v = np.asarray(weights, dtype=float)
    if v.shape != (n_cond,) or np.any(v < 0) or not np.isfinite(v).all():
        raise ValueError("condition weights must be nonnegative and one per condition")
    total = v.sum()
    if total <= 0:
        raise ValueError("condition weights must not sum to zero")
    return v / total


def frontier_owners(W, delta: float = 0.0, candidates=None) -> list:
    """Owners of each condition: candidates scoring within `delta` of its maximum.

    `candidates` restricts the comparison to a subset, which is how the owner
    sets are recomputed after ownership reduction."""
    W = _as_table(W)
    if delta < 0:
        raise ValueError("the tie margin must be nonnegative")
    keep = np.arange(W.shape[0]) if candidates is None else np.asarray(candidates, dtype=int)
    if keep.size == 0:
        raise ValueError("at least one candidate is required")
    sub = W[keep]
    return [keep[np.flatnonzero(sub[:, t] >= sub[:, t].max() - delta)]
            for t in range(W.shape[1])]


def _shares(W, owners, weights, n_cand: int) -> np.ndarray:
    """C_k = sum_t v_t * 1[k in F_t] / |F_t|: each condition's weight is split
    among its owners, so near-tied candidates share its credit."""
    C = np.zeros(n_cand)
    for t, owner in enumerate(owners):
        C[owner] += weights[t] / len(owner)
    return C


def coverage_shares(W, delta: float = 0.0, weights=None, reduce: bool = False) -> Ownership:
    """Frontier ownership and fractional coverage of a score table.

    With `reduce`, a candidate is dropped when every condition it owns keeps
    another owner without it, and the shares are recomputed from the reduced
    owner sets. Candidates are considered in increasing order of coverage, so
    the smallest shares are tested for removal first."""
    W = _as_table(W)
    n_cand, n_cond = W.shape
    v = _normalized_weights(weights, n_cond)
    owners = frontier_owners(W, delta)
    C = _shares(W, owners, v, n_cand)
    retained = np.arange(n_cand)

    if reduce:
        kept = set(range(n_cand))
        for k in (int(j) for j in np.argsort(C, kind="stable")):
            others = kept - {k}
            if not others:
                continue
            # k is removable when every condition it owns keeps an owner without it,
            # which leaves each column's maximum represented by a retained candidate.
            covered = all(k not in owner or (set(owner) & others)
                          for owner in owners)
            if covered:
                kept = others
        retained = np.array(sorted(kept))
        owners = frontier_owners(W, delta, candidates=retained)
        C = _shares(W, owners, v, n_cand)

    return Ownership(owners=owners, coverage=C, retained=retained)


def selection_probabilities(coverage, retained=None, epsilon: float = 0.05) -> np.ndarray:
    """Parent-selection probabilities from coverage, with a uniform floor.

    pi_k is proportional to C_k over the included candidates, mixed with a
    uniform component so that every included candidate keeps a positive
    probability within a finite budget. Excluded candidates receive zero."""
    C = np.asarray(coverage, dtype=float)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("the exploration floor must lie in [0, 1]")
    idx = np.arange(C.size) if retained is None else np.asarray(retained, dtype=int)
    if idx.size == 0:
        raise ValueError("at least one candidate must be included")
    mass = C[idx]
    base = mass / mass.sum() if mass.sum() > 0 else np.full(idx.size, 1.0 / idx.size)
    pi = np.zeros(C.size)
    pi[idx] = (1.0 - epsilon) * base + epsilon / idx.size
    return pi
