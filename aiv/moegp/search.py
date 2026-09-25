"""Fold-averaged objective and the (1+1) evolution strategy over the gate.

The objective is the area under the ROC curve of the mixture on held-out rows,
averaged over rotations of a stratified split of the discovery data. The
earlier centroid search scored a single seventy-thirty draw, which put twenty
free parameters against one fixed set of scoring rows; because the expert fit
here is a ridge solve in a precomputed basis rather than a boosting fit, the
rotation costs about what one evaluation used to cost, and the search can no
longer exploit one particular partition.

The strategy is the same (1+1) rule as before so that the comparison is of the
model and not of the optimizer: one Gaussian proposal per step, the step size
growing on success and shrinking on failure, and restarting when it collapses.
Proposals are scaled per parameter block.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from aiv.moegp.experts import fit_mixture, mixture_proba, to_tensor


@dataclass
class Fold:
    """One rotation: rows the experts are fitted on, rows they are scored on."""

    Xs_fit: np.ndarray
    Xs_score: np.ndarray
    y_score: np.ndarray
    Psi_fit: torch.Tensor
    Psi_score: torch.Tensor
    y_fit: torch.Tensor
    off_fit: torch.Tensor
    off_score: torch.Tensor


def make_folds(Xs, Psi, y, offset, n_folds: int, seed: int, device) -> list:
    Xs, y, offset = np.asarray(Xs, float), np.asarray(y).astype(float), np.asarray(offset, float)
    Psi = np.asarray(Psi, float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fit_idx, score_idx in skf.split(Xs, y.astype(int)):
        folds.append(Fold(
            Xs_fit=Xs[fit_idx], Xs_score=Xs[score_idx], y_score=y[score_idx].astype(int),
            Psi_fit=to_tensor(Psi[fit_idx], device), Psi_score=to_tensor(Psi[score_idx], device),
            y_fit=to_tensor(y[fit_idx], device),
            off_fit=to_tensor(offset[fit_idx], device), off_score=to_tensor(offset[score_idx], device),
        ))
    return folds


def fit_and_score(fold: Fold, params, n_irls: int = 5, n_lambda: int = 41,
                  min_support: int = 40):
    """Fit the experts on the fold's fitting rows and return the scoring AUC.

    A gate that starves a component is rejected rather than fitted: with a hard
    assignment a center pushed into a corner leaves an expert with too few rows
    for the ridge to be meaningful, and the objective should not reward it.
    """
    gamma_fit = params.responsibilities(fold.Xs_fit)
    if gamma_fit.sum(axis=0).min() < min_support:
        return None, None
    g_fit = to_tensor(gamma_fit, fold.Psi_fit.device)
    fits = fit_mixture(fold.Psi_fit, fold.y_fit, fold.off_fit, g_fit,
                       n_irls=n_irls, n_lambda=n_lambda)
    g_score = to_tensor(params.responsibilities(fold.Xs_score), fold.Psi_score.device)
    p = mixture_proba(fold.Psi_score, fold.off_score, g_score, fits).cpu().numpy()
    return float(roc_auc_score(fold.y_score, p)), fits


def objective(folds, params, n_irls: int = 5, n_lambda: int = 41,
              min_support: int = 40, penalty: float = -1.0) -> float:
    scores = []
    for fold in folds:
        auc, _ = fit_and_score(fold, params, n_irls=n_irls, n_lambda=n_lambda,
                               min_support=min_support)
        if auc is None:
            return penalty
        scores.append(auc)
    return float(np.mean(scores))


@dataclass
class SearchResult:
    x: np.ndarray
    score: float
    x0: np.ndarray
    score0: float
    n_evals: int
    seconds: float
    trajectory: list = field(default_factory=list)
    n_params: int = 0


def es_search(score_fn, x0, scales, seconds: float = 300.0, max_evals: int | None = None,
              seed: int = 0, sigma0: float = 0.5, shrink: float = 0.85,
              sigma_min: float = 0.02) -> SearchResult:
    """(1+1) evolution strategy with one adapting step size over scaled blocks."""
    x0 = np.asarray(x0, float)
    scales = np.asarray(scales, float)
    rng = np.random.default_rng(seed)
    cur, cur_score = x0.copy(), float(score_fn(x0))
    best0 = cur_score
    sigma = sigma0
    traj = [cur_score]
    t0 = time.time()
    evals = 0
    while time.time() - t0 < seconds and (max_evals is None or evals < max_evals):
        cand = cur + sigma * scales * rng.normal(size=cur.shape)
        s = float(score_fn(cand))
        evals += 1
        if s > cur_score:
            cur, cur_score, sigma = cand, s, sigma / shrink
        else:
            sigma *= shrink
        if sigma < sigma_min:
            sigma = sigma0
        traj.append(cur_score)
    return SearchResult(x=cur, score=cur_score, x0=x0, score0=best0, n_evals=evals,
                        seconds=time.time() - t0, trajectory=traj, n_params=int(x0.size))
