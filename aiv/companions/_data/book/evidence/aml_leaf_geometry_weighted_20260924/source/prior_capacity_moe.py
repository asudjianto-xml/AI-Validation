"""Fitting a gated mixture under the mean and worst-population objectives.

The mixture routes on context with a linear softmax gate and predicts with logistic
experts on the outcome signals. Feature roles are given rather than discovered,
because the question here is how limited expert capacity is allocated between
populations; role discovery is the subject of ``synthetic_population_data``.

Overall mean loss is the population-weighted average of the per-population losses,
so it is the weight vector q equal to the observed population proportions, and the
whole weighted family is swept by varying q over the simplex. The maximum over
populations is not a member of that family, and comparing the two is the point:
a weighted average can only reach the convex hull of the attainable loss set, so
a min-max solution lying off that hull is one no choice of weights would find.

Population labels enter fitting only as the fixed index set over which the maximum
is taken. They are never predictors. The populations are generating truth and so
cannot move with the candidate, which is the condition that makes the maximum
comparable across models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from capacity_moe_data import (CONTEXT_FEATURES, NOISE_FEATURES, SIGNAL_FEATURES,
                               coefficients)

EXPERT_INPUTS = SIGNAL_FEATURES + NOISE_FEATURES


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Standardizer:
    """Training means and scales for the gate block and the expert block."""

    gate_features: tuple
    expert_features: tuple
    gate_mu: np.ndarray
    gate_sd: np.ndarray
    expert_mu: np.ndarray
    expert_sd: np.ndarray


def standardizer(X, gate_features=CONTEXT_FEATURES, expert_features=EXPERT_INPUTS):
    Z = np.asarray(X[list(gate_features)], dtype=float)
    S = np.asarray(X[list(expert_features)], dtype=float)
    return Standardizer(tuple(gate_features), tuple(expert_features),
                        Z.mean(0), Z.std(0) + 1e-12,
                        S.mean(0), S.std(0) + 1e-12)


def _tensors(X, y, group, std: Standardizer, dev):
    Z = (np.asarray(X[list(std.gate_features)], dtype=float) - std.gate_mu) / std.gate_sd
    S = (np.asarray(X[list(std.expert_features)], dtype=float) - std.expert_mu) / std.expert_sd
    y = np.zeros(len(Z)) if y is None else np.asarray(y)
    group = np.zeros(len(Z), dtype=int) if group is None else np.asarray(group)
    return (torch.tensor(Z, dtype=torch.float32, device=dev),
            torch.tensor(S, dtype=torch.float32, device=dev),
            torch.tensor(y, dtype=torch.float32, device=dev),
            torch.tensor(group, dtype=torch.long, device=dev))


@dataclass
class Params:
    """A batch of independent mixtures, one per configuration."""

    gate_w: torch.Tensor   # [C, n_context, K]
    gate_b: torch.Tensor   # [C, K]
    exp_w: torch.Tensor    # [C, n_expert_inputs, K]
    exp_b: torch.Tensor    # [C, K]

    def detached(self):
        return Params(*[t.detach().clone() for t in
                        (self.gate_w, self.gate_b, self.exp_w, self.exp_b)])


def init_params(n_configs: int, n_experts: int, seeds, dev,
                n_gate: int = len(CONTEXT_FEATURES),
                n_expert: int = len(EXPERT_INPUTS)) -> Params:
    gen = torch.Generator(device="cpu")
    w = []
    for s in seeds:
        gen.manual_seed(int(s))
        w.append([torch.randn(n_gate, n_experts, generator=gen),
                  torch.zeros(n_experts),
                  0.5 * torch.randn(n_expert, n_experts, generator=gen),
                  torch.zeros(n_experts)])
    stacked = [torch.stack([w[i][j] for i in range(len(seeds))]).to(dev)
               for j in range(4)]
    assert stacked[0].shape[0] == n_configs, "one seed per configuration"
    return Params(*[t.requires_grad_(True) for t in stacked])


def row_losses(params: Params, Z, S, y) -> torch.Tensor:
    """Per-configuration, per-row negative log-likelihood of the mixture."""
    gate = torch.softmax(torch.einsum("nd,cdk->cnk", Z, params.gate_w)
                         + params.gate_b[:, None, :], dim=-1)
    expert = torch.sigmoid(torch.einsum("nd,cdk->cnk", S, params.exp_w)
                           + params.exp_b[:, None, :])
    p = (gate * expert).sum(-1).clamp(1e-7, 1 - 1e-7)
    return -(y[None, :] * torch.log(p) + (1 - y[None, :]) * torch.log1p(-p))


def group_losses(nll: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Mean loss within each fixed population, shape [C, n_groups]."""
    out = torch.zeros(nll.shape[0], n_groups, device=nll.device, dtype=nll.dtype)
    for g in range(n_groups):
        out[:, g] = nll[:, group == g].mean(1)
    return out


def fit(X, y, group, n_experts: int, weights=None, worst: bool = False,
        seeds=(0,), n_groups: int = 3, steps: int = 4000, lr: float = 0.05,
        std=None, dev=None, gate_features=CONTEXT_FEATURES,
        expert_features=EXPERT_INPUTS):
    """Fit one mixture per configuration.

    ``weights`` gives a [C, n_groups] array of population weights and fits the
    weighted objective; ``worst=True`` fits the maximum over populations instead.
    Configurations are independent, so the batch is summed and differentiated once.
    """
    dev = dev or device()
    std = std if std is not None else standardizer(X, gate_features, expert_features)
    Z, S, yt, gt = _tensors(X, y, group, std, dev)
    if worst:
        n_configs = len(seeds)
        Q = None
    else:
        W = torch.tensor(np.asarray(weights, dtype=float), dtype=torch.float32, device=dev)
        Q = W.repeat_interleave(len(seeds), dim=0)
        n_configs = Q.shape[0]
    params = init_params(n_configs, n_experts,
                         list(seeds) * (n_configs // len(seeds)), dev,
                         n_gate=len(std.gate_features),
                         n_expert=len(std.expert_features))
    opt = torch.optim.Adam([params.gate_w, params.gate_b, params.exp_w, params.exp_b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        L = group_losses(row_losses(params, Z, S, yt), gt, n_groups)
        obj = L.max(1).values if worst else (Q * L).sum(1)
        obj.sum().backward()
        opt.step()
    return params.detached(), std, Q


def evaluate(params: Params, X, y, group, std, n_groups: int = 3, dev=None):
    """Overall and per-population loss of every configuration, as numpy."""
    dev = dev or device()
    Z, S, yt, gt = _tensors(X, y, group, std, dev)
    with torch.no_grad():
        nll = row_losses(params, Z, S, yt)
        L = group_losses(nll, gt, n_groups)
        return nll.mean(1).cpu().numpy(), L.cpu().numpy()


def predict(params: Params, X, std, dev=None) -> np.ndarray:
    """Mixture event probability per configuration, shape [C, n_rows]."""
    dev = dev or device()
    Z, S, _, _ = _tensors(X, None, None, std, dev)
    with torch.no_grad():
        gate = torch.softmax(torch.einsum("nd,cdk->cnk", Z, params.gate_w)
                             + params.gate_b[:, None, :], dim=-1)
        expert = torch.sigmoid(torch.einsum("nd,cdk->cnk", S, params.exp_w)
                               + params.exp_b[:, None, :])
        return (gate * expert).sum(-1).cpu().numpy()


def simplex_grid(n_groups: int = 3, step: float = 0.05) -> np.ndarray:
    """All weight vectors on a regular simplex lattice."""
    k = int(round(1 / step))
    pts = []
    for a in range(k + 1):
        for b in range(k + 1 - a):
            pts.append([a, b, k - a - b])
    return np.array(pts, dtype=float) / k


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """True where no other point is at least as low on both coordinates and lower on one."""
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        keep[i] = not np.any(np.all(others <= p, axis=1) & np.any(others < p, axis=1))
    return keep


def oracle_partition_frontier(norms, n_experts: int, priors, n: int = 400_000,
                              seed: int = 77, **_):
    """Attainable loss of every way of assigning populations to experts.

    Routing is the true population and each expert is the maximum-likelihood
    logistic predictor on the signals for the populations assigned to it, so this
    is the best a mixture of this capacity can do when the gate is exact. It bounds
    what any fitted model can reach and names the arrangement behind each point.
    """
    from itertools import product

    from sklearn.linear_model import LogisticRegression

    from capacity_moe_data import SIGNAL_FEATURES as SIG
    from capacity_moe_data import generate

    X, y, group = generate(n, seed, norms)
    S = np.asarray(X[list(SIG)], dtype=float)
    y = np.asarray(y)
    n_groups = len(priors)

    assignments = [a for a in product(range(n_experts), repeat=n_groups)
                   if len(set(a)) == n_experts]
    # Expert labels are interchangeable; keep one representative per partition.
    seen, unique = set(), []
    for a in assignments:
        key = frozenset(frozenset(g for g in range(n_groups) if a[g] == k)
                        for k in range(n_experts))
        if key not in seen:
            seen.add(key)
            unique.append(a)

    rows = []
    for a in unique:
        L = np.zeros(n_groups)
        blocks = [sorted(g for g in range(n_groups) if a[g] == k) for k in range(n_experts)]
        for block in blocks:
            rows_in = np.isin(group, block)
            model = LogisticRegression(C=1e6, max_iter=2000).fit(S[rows_in], y[rows_in])
            p = model.predict_proba(S[rows_in])[:, 1].clip(1e-7, 1 - 1e-7)
            yy = y[rows_in]
            nll = -(yy * np.log(p) + (1 - yy) * np.log1p(-p))
            for g in block:
                L[g] = nll[group[rows_in] == g].mean()
        rows.append(dict(assignment=list(a), blocks=blocks,
                         group_losses=L.tolist(),
                         mean=float((np.asarray(priors) * L).sum()),
                         worst=float(L.max())))
    return rows
