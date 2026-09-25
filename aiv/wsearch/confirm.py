"""Confirmation of frozen regions on untouched evidence.

Discovery estimates are selected statistics and are never reused as confirmatory
evidence. Confirmation applies the frozen membership rule unchanged to data no
part of the search has seen, estimates the same contrast there, and tests it
against a predeclared materiality margin. The margin matters: a null of no
inflation at all rejects for almost any region chosen for high loss on a model
whose performance varies, so it confirms nearly everything and leaves the
substantive question outside the statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Confirmation:
    n_conf: int
    support: int
    contrast: float
    ci_low: float
    ci_high: float
    p_value: float
    margin: float
    z0: float
    accel: float
    detail: str


def _contrast(loss: np.ndarray, inside: np.ndarray) -> float:
    s_in, n_in = loss[inside].sum(), int(inside.sum())
    s_out, n_out = loss[~inside].sum(), int((~inside).sum())
    if n_in < 1 or n_out < 1 or s_out <= 0:
        return np.nan
    return float((s_in / n_in) / (s_out / n_out) - 1.0)


def _jackknife_contrasts(loss: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Delete-one-row contrasts in closed form, for the acceleration constant."""
    s_in, n_in = loss[inside].sum(), int(inside.sum())
    s_out, n_out = loss[~inside].sum(), int((~inside).sum())
    out = np.empty(len(loss))
    mean_out = s_out / n_out
    mean_in = s_in / n_in
    out[inside] = ((s_in - loss[inside]) / (n_in - 1)) / mean_out - 1.0
    out[~inside] = mean_in / ((s_out - loss[~inside]) / (n_out - 1)) - 1.0
    return out


def bca_confirm(loss, membership, *, margin: float = 0.0, alpha: float = 0.05,
                n_boot: int = 4000, seed: int = 0, min_support: int = 30) -> Confirmation:
    """Bias-corrected and accelerated interval and one-sided p-value for H0: M <= margin.

    The contrast is a ratio of means of a right-skewed loss, so a normal
    approximation understates the upper tail; BCa corrects both the median bias
    of the bootstrap distribution and the dependence of its spread on the
    parameter. The p-value inverts the interval: the bootstrap quantile level at
    which the lower endpoint equals the margin is mapped back through the same
    bias and acceleration correction.
    """
    loss = np.asarray(loss, dtype=float)
    inside = np.asarray(membership, dtype=bool)
    if loss.ndim != 1 or inside.shape != loss.shape:
        raise ValueError("losses and boolean membership must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(loss)) or np.any(loss < 0):
        raise ValueError("losses must be finite and nonnegative")
    n, support = len(loss), int(inside.sum())
    if support < min_support or support >= n:
        return Confirmation(n, support, np.nan, np.nan, np.nan, np.nan, margin, np.nan, np.nan,
                            f"inconclusive: support {support} below the required {min_support}")

    theta = _contrast(loss, inside)
    rng = np.random.default_rng(seed)
    idx = rng.integers(n, size=(n_boot, n))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        take = idx[b]
        boot[b] = _contrast(loss[take], inside[take])
    boot = boot[np.isfinite(boot)]
    if len(boot) < n_boot // 2:
        return Confirmation(n, support, theta, np.nan, np.nan, np.nan, margin, np.nan, np.nan,
                            "inconclusive: bootstrap contrast undefined too often")

    z0 = float(norm.ppf(np.clip((boot < theta).mean(), 1 / (2 * len(boot)), 1 - 1 / (2 * len(boot)))))
    jk = _jackknife_contrasts(loss, inside)
    dev = jk.mean() - jk
    denom = 6.0 * (dev @ dev) ** 1.5
    a = float((dev ** 3).sum() / denom) if denom > 0 else 0.0

    def adjust(z):
        d = z0 + z
        return float(norm.cdf(z0 + d / (1.0 - a * d)))

    lo = float(np.quantile(boot, np.clip(adjust(norm.ppf(alpha / 2)), 0.0, 1.0)))
    hi = float(np.quantile(boot, np.clip(adjust(norm.ppf(1 - alpha / 2)), 0.0, 1.0)))

    q = np.clip((boot < margin).mean(), 1 / (2 * len(boot)), 1 - 1 / (2 * len(boot)))
    u = float(norm.ppf(q)) - z0
    p = float(norm.cdf(u / (1.0 + a * u) - z0)) if (1.0 + a * u) != 0 else np.nan

    return Confirmation(n, support, theta, lo, hi, p, margin, z0, a,
                        f"M={theta:.3f} CI=[{lo:.3f},{hi:.3f}] p(M<={margin:.2f})={p:.4g}")


def confirm_frozen(frozen, df, losses, *, margin: float = 0.0, **kwargs) -> Confirmation:
    """Apply a frozen region to confirmation rows and test it there."""
    return bca_confirm(losses, frozen.mask(df), margin=margin, **kwargs)


def benjamini_hochberg(p_values, q: float = 0.05):
    """Step-up false-discovery-rate control; returns the rejection mask and
    adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    ok = np.isfinite(p)
    G = int(ok.sum())
    reject = np.zeros(len(p), dtype=bool)
    adj = np.full(len(p), np.nan)
    if G == 0:
        return reject, adj
    order = np.flatnonzero(ok)[np.argsort(p[ok], kind="mergesort")]
    ranked = p[order]
    adj_sorted = np.minimum.accumulate((ranked * G / np.arange(1, G + 1))[::-1])[::-1]
    adj[order] = np.minimum(adj_sorted, 1.0)
    below = np.flatnonzero(ranked <= q * np.arange(1, G + 1) / G)
    if len(below):
        reject[order[: below[-1] + 1]] = True
    return reject, adj


def holm(p_values, alpha: float = 0.05):
    """Step-down family-wise error control, reported as a sensitivity analysis."""
    p = np.asarray(p_values, dtype=float)
    ok = np.isfinite(p)
    G = int(ok.sum())
    reject = np.zeros(len(p), dtype=bool)
    adj = np.full(len(p), np.nan)
    if G == 0:
        return reject, adj
    order = np.flatnonzero(ok)[np.argsort(p[ok], kind="mergesort")]
    ranked = p[order]
    adj_sorted = np.minimum(np.maximum.accumulate(ranked * (G - np.arange(G))), 1.0)
    adj[order] = adj_sorted
    reject[order] = adj_sorted <= alpha
    return reject, adj
