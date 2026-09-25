"""Responsibility-weighted Gaussian-process experts in the leaf feature map.

Each component models a correction to the base model's logit,

    eta_h(x) = f0(x) + g_h(x),      g_h ~ GP(0, sigma_f^2 K),

with K the leaf kernel. In the primal representation g_h(x) = psi(x)^T beta_h
with beta_h ~ N(0, sigma_f^2 I), so the posterior mean is a ridge solution and
only the ratio lambda_h = sigma^2 / sigma_f^2 enters it. The outcome is binary,
so the fit is iteratively reweighted least squares on the Bernoulli likelihood
with the base logit held as an offset, and the gate enters as an observation
weight: component h maximizes sum_i gamma_h(x_i) log p(y_i | eta_h(x_i)) under
the prior. Every row therefore contributes to every component in proportion to
its responsibility, so the effective sample size of an expert is sum_i
gamma_h(x_i) rather than n / H; under a hard gate the two coincide.

At each reweighting the ridge is set by the evidence of the Gaussian
working-response model at the current weights, profiled over the noise scale.
With responsibility weights this is a tempered evidence rather than an exact
marginal likelihood; the effective count sum_i gamma_h(x_i) replaces n in the
profile, so a component holding little mass is not charged a full sample's
worth of evidence. Selecting the ridge requires the eigenvalues of the weighted
Gram and is the dominant cost, so it is done on the first and last reweighting
and the intermediate steps reuse the current value through a Cholesky solve.

Components are fitted as one batch: they share the feature matrix and differ
only in their weights, so the Grams, eigendecompositions and solves stack.

The mixture combines the expert predictions on the probability scale,
p(y = 1 | x) = sum_h gamma_h(x) sigmoid(eta_h(x)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

_P_CLIP = 1e-6
_W_FLOOR = 1e-8


def device_of(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def to_tensor(a, device, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


@dataclass
class MixtureFit:
    """Fitted corrections, one row of `beta` per component."""

    beta: torch.Tensor    # (H, m)
    lam: np.ndarray       # (H,) selected ridge
    n_eff: np.ndarray     # (H,) effective sample size
    n_irls: int


def _lambda_grid(A: torch.Tensor, n_lambda: int) -> torch.Tensor:
    """Ridge grid per component, anchored on the mean eigenvalue of its Gram."""
    scale = torch.clamp(torch.diagonal(A, dim1=-2, dim2=-1).mean(dim=-1), min=1e-12)
    expo = torch.linspace(-8.0, 2.0, n_lambda, dtype=A.dtype, device=A.device)
    return scale[:, None] * torch.pow(torch.tensor(10.0, dtype=A.dtype, device=A.device), expo)[None, :]


def _solve_with_evidence(A, b, zWz, n_eff, n_lambda):
    """Ridge solutions at the evidence-optimal lambda on the grid, per component."""
    s, Q = torch.linalg.eigh(A)                       # (H, m), (H, m, m)
    s = torch.clamp(s, min=0.0)
    bq = torch.einsum("hmk,hm->hk", Q, b)             # Q^T b
    lams = _lambda_grid(A, n_lambda)                  # (H, L)
    denom = s[:, None, :] + lams[:, :, None]          # (H, L, m)
    quad = ((bq[:, None, :] ** 2) / denom).sum(dim=2)
    rss = torch.clamp(zWz[:, None] - quad, min=1e-12)
    logdet = torch.log1p(s[:, None, :] / lams[:, :, None]).sum(dim=2)
    n_safe = torch.clamp(n_eff, min=1e-12)[:, None]
    crit = n_safe * torch.log(rss / n_safe) + logdet
    k = torch.argmin(crit, dim=1)                     # (H,)
    lam = torch.gather(lams, 1, k[:, None]).squeeze(1)
    beta = torch.einsum("hmk,hk->hm", Q, bq / (s + lam[:, None]))
    return beta, lam


def _solve_fixed(A, b, lam):
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    L = torch.linalg.cholesky(A + lam[:, None, None] * eye)
    return torch.cholesky_solve(b[:, :, None], L).squeeze(-1)


def fit_mixture(Psi: torch.Tensor, y: torch.Tensor, offset: torch.Tensor,
                gamma: torch.Tensor, n_irls: int = 5, n_lambda: int = 41,
                tol: float = 1e-5) -> MixtureFit:
    """Weighted IRLS fit of every component's correction, as one batch.

    `gamma` is (n, H); a column of zeros yields a zero correction.
    """
    H = gamma.shape[1]
    G = gamma.T.contiguous()                                    # (H, n)
    n_eff = G.sum(dim=1)
    beta = torch.zeros(H, Psi.shape[1], dtype=Psi.dtype, device=Psi.device)
    lam = torch.zeros(H, dtype=Psi.dtype, device=Psi.device)
    used = 0
    for used in range(1, n_irls + 1):
        eta = offset[None, :] + beta @ Psi.T                    # (H, n)
        p = torch.clamp(torch.sigmoid(eta), _P_CLIP, 1.0 - _P_CLIP)
        w = torch.clamp(p * (1.0 - p), min=_W_FLOOR)
        W = G * w
        z = (eta - offset[None, :]) + (y[None, :] - p) / w
        Wz = W * z
        A = torch.einsum("hn,ni,nj->hij", W, Psi, Psi)
        b = Wz @ Psi
        zWz = (Wz * z).sum(dim=1)
        if used == 1 or used == n_irls:
            new_beta, lam = _solve_with_evidence(A, b, zWz, n_eff, n_lambda)
        else:
            new_beta = _solve_fixed(A, b, lam)
        shift = torch.linalg.norm(new_beta - beta, dim=1)
        scale = torch.linalg.norm(new_beta, dim=1) + 1e-12
        converged = bool((shift / scale < tol).all().item())
        beta = new_beta
        if converged and used > 1:
            if used < n_irls:                                   # settle the ridge once more
                A_f = torch.einsum("hn,ni,nj->hij", W, Psi, Psi)
                beta, lam = _solve_with_evidence(A_f, b, zWz, n_eff, n_lambda)
            break
    return MixtureFit(beta=beta, lam=lam.cpu().numpy(), n_eff=n_eff.cpu().numpy(), n_irls=used)


def mixture_proba(Psi: torch.Tensor, offset: torch.Tensor, gamma: torch.Tensor,
                  fit: MixtureFit) -> torch.Tensor:
    """sum_h gamma_h(x) sigmoid(f0(x) + psi(x)^T beta_h)."""
    eta = offset[None, :] + fit.beta @ Psi.T
    return (gamma.T * torch.sigmoid(eta)).sum(dim=0)
