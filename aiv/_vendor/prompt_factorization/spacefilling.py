"""Space-filling designers (vendored).

Self-contained space-filling design over typed factors: scrambled Sobol / LHS /
grid via `scipy.stats.qmc`, with optional gradient phi_p refinement (`sobol+refine`)
via the analytic-gradient optimizer below. numpy + scipy only; torch is imported
lazily and only for `sobol+refine`. CUDA is optional (CPU fallback).

`SpaceFillingDesigner` implements the `PopulationDesigner` and `ConditionDesigner`
protocols, so it drops into the optimizer's designed-initialization seams.

The `optimize` routine is vendored verbatim from the graphdoe space-filling code
(SPDX-License-Identifier: Apache-2.0), the GPU-accelerated gradient phi_p design
of Jin & Sudjianto; only its import context changed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc



# --------------------------------------------------------------------------
# Vendored phi_p optimizer (Apache-2.0; analytic gradients, O(n^2) memory).
# --------------------------------------------------------------------------
def _loss_and_grad(Z, p, weights=None):
    import torch

    X = torch.sigmoid(Z)
    Xw = X * weights.unsqueeze(0) if weights is not None else X
    M = torch.cdist(Xw, Xw)
    M.clamp_(min=1e-30)
    M.log_()
    M.mul_(-p)
    M.fill_diagonal_(float("-inf"))
    log_phi_full = torch.logsumexp(M.reshape(-1), dim=0)
    loss = (log_phi_full - math.log(2.0)) / p
    log_phi_sum = log_phi_full - math.log(2.0)
    M.mul_((p + 2.0) / p)
    M.sub_(log_phi_sum)
    M.exp_()
    M.fill_diagonal_(0.0)
    G_row_sum = M.sum(dim=1, keepdim=True)
    grad_X = torch.mm(M, X) - G_row_sum * X
    if weights is not None:
        grad_X = grad_X * (weights.unsqueeze(0) ** 2)
    grad_Z = grad_X * X * (1.0 - X)
    return loss, grad_Z


def _get_device():
    import torch

    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def optimize(
    n,
    m,
    *,
    p_final=50,
    p_start=5,
    n_stages=5,
    iters_per_stage=None,
    lr=None,
    use_lbfgs=True,
    device=None,
    dtype=None,
    seed=None,
    X_init=None,
    verbose=False,
    dim_weights=None,
):
    """Optimize n points in [0,1]^m for space-filling (phi_p criterion).

    Returns (X, phi_p, info) with X an (n, m) ndarray in [0,1]^m.
    """
    import torch

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = _get_device()
    if lr is None:
        lr = 1.0 if use_lbfgs else 0.01
    if iters_per_stage is None:
        iters_per_stage = 200 if use_lbfgs else 500
    if seed is not None:
        torch.manual_seed(seed)

    w_t = None
    if dim_weights is not None:
        w_t = torch.tensor(dim_weights, device=device, dtype=dtype)

    if X_init is not None:
        X0 = torch.tensor(X_init, device=device, dtype=dtype).clamp(0.02, 0.98)
    else:
        X0 = torch.rand(n, m, device=device, dtype=dtype) * 0.96 + 0.02
    Z = torch.logit(X0)
    Z.requires_grad_(True)

    ps = np.geomspace(p_start, p_final, n_stages).tolist()

    for _stage, p in enumerate(ps):
        if use_lbfgs:
            opt = torch.optim.LBFGS([Z], lr=lr, max_iter=iters_per_stage, line_search_fn="strong_wolfe")

            def closure():
                with torch.no_grad():
                    loss, grad = _loss_and_grad(Z, p, weights=w_t)
                    Z.grad = grad
                return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam([Z], lr=lr)
            for _ in range(iters_per_stage):
                with torch.no_grad():
                    loss, grad = _loss_and_grad(Z, p, weights=w_t)
                    Z.grad = grad
                opt.step()

    with torch.no_grad():
        X_final = torch.sigmoid(Z)
        Xw = X_final * w_t.unsqueeze(0) if w_t is not None else X_final
        D = torch.cdist(Xw, Xw)
        D.clamp_(min=1e-30)
        D.log_()
        D.mul_(-p_final)
        D.fill_diagonal_(float("-inf"))
        log_phi_full = torch.logsumexp(D.reshape(-1), dim=0)
        phi_p = ((log_phi_full - math.log(2.0)) / p_final).exp().item()
        D2 = torch.cdist(Xw, Xw)
        D2.fill_diagonal_(float("inf"))
        min_dist = D2.min().item()

    return X_final.cpu().numpy(), phi_p, {"min_dist": min_dist, "device": str(device)}


# --------------------------------------------------------------------------
# Designer over typed factors
# --------------------------------------------------------------------------
def _sample_unit_cube(d: int, n: int, method: str, seed: int, verbose: bool, dim_weights):
    if method == "sobol" or method == "sobol+refine":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = max(1, math.ceil(math.log2(max(2, n))))
        base = sampler.random_base2(m=m)[:n]
        if method == "sobol":
            return base
        refined, _, _ = optimize(n, d, p_final=50, seed=seed, X_init=base, verbose=verbose, dim_weights=dim_weights)
        return refined
    if method == "lhs":
        return qmc.LatinHypercube(d=d, seed=seed).random(n=n)
    if method == "grid":
        levels_per = max(2, round(n ** (1.0 / d)))
        axes = [np.linspace(0, 1, levels_per) for _ in range(d)]
        grid = np.array(np.meshgrid(*axes)).T.reshape(-1, d)
        if len(grid) > n:
            idx = np.random.default_rng(seed).choice(len(grid), n, replace=False)
            grid = grid[idx]
        return grid
    raise ValueError(f"Unknown method: {method!r} (use sobol / sobol+refine / lhs / grid)")



