"""Twinning folds for replicate measurement and the initial design.

Twinning partitions a sample into subsets with similar empirical distributions
by approximately minimizing energy distance. Folds are built on the covariates
alone: folds balanced on the realized loss would suppress the very variation the
delete-one-fold replicate is meant to measure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _gpu_twinning():
    """The repository's GPU twinning implementation, when it is importable.

    It lives in `twinning/` at the repository root rather than in an installed
    package, so the path is added on demand. Returning None puts this module on
    its own fallback, which is a traversal of the same shape but without the
    multiplet strategies.
    """
    try:
        import gpu_twinning
        return gpu_twinning
    except ImportError:
        pass
    root = Path(__file__).resolve().parents[2] / "twinning"
    if not root.is_dir():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import gpu_twinning
        return gpu_twinning
    except ImportError:
        return None


def twin_folds(Xs: np.ndarray, K: int, seed: int = 0, strategy: int = 1,
               force_fallback: bool = False) -> np.ndarray:
    """Fold label per row, from the multiplet construction of Vakayil and Joseph.

    Delegates to the repository's GPU twinning implementation when it is
    available. The fallback deals a nearest-neighbor traversal round-robin to the
    folds, which spreads neighboring rows across folds rather than concentrating
    them in one; it reproduces the shape of strategy 3 and is used only where the
    implementation is absent.
    """
    Xs = np.asarray(Xs, dtype=float)
    n = len(Xs)
    if not 2 <= K <= n:
        raise ValueError("K must be between 2 and the number of rows")

    gt = None if force_fallback else _gpu_twinning()
    if gt is not None and K <= n // 2:
        labels = gt.multiplet(Xs, K, strategy=strategy)
        return np.asarray(labels.cpu().numpy() if hasattr(labels, "cpu") else labels, dtype=int)

    order = _nn_chain(Xs, seed)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % K
    return folds


def _nn_chain(Xs: np.ndarray, seed: int) -> np.ndarray:
    """Greedy nearest-neighbor traversal from the row closest to the centroid."""
    n = len(Xs)
    start = int(np.argmin(((Xs - Xs.mean(0)) ** 2).sum(1)))
    unused = np.ones(n, dtype=bool)
    order = np.empty(n, dtype=int)
    cur = start
    for t in range(n):
        order[t] = cur
        unused[cur] = False
        if t == n - 1:
            break
        d = ((Xs - Xs[cur]) ** 2).sum(1)
        d[~unused] = np.inf
        cur = int(np.argmin(d))
    return order


def energy_distance(A: np.ndarray, B: np.ndarray, rng=None, max_n: int = 2000) -> float:
    """Energy distance between two samples, subsampled for tractability.

    Reported per fold against the full discovery covariates in the Twinning
    ablation, where the comparator is a random K-fold partition at the same K.
    Delegates to the repository's implementation when it is available, which
    computes the criterion twinning itself minimizes and needs no subsampling.
    """
    gt = _gpu_twinning()
    if gt is not None:
        return float(gt.energy(A, B, full=True))
    rng = rng or np.random.default_rng(0)
    A, B = np.asarray(A, float), np.asarray(B, float)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    dab = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).mean()
    daa = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)).mean()
    dbb = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(-1)).mean()
    return float(2 * dab - daa - dbb)
