"""Tests for the mixture of processes on the leaf kernel."""
from __future__ import annotations

import numpy as np
import pytest
torch = pytest.importorskip("torch")  # the torch extra

from aiv.moegp.experts import fit_mixture, mixture_proba, to_tensor
from aiv.moegp.gate import CentroidParams, GateParams, Standardizer, kmeans_centroids, kmeans_init
from aiv.moegp.kernel import LeafMap, fit_base, monotone_string, primal_basis
from aiv.moegp.search import es_search

DEVICE = torch.device("cpu")


def _toy(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))
    return X, rng.binomial(1, p)


def test_monotone_string_orders_by_feature():
    assert monotone_string(["a", "b", "c"], {"b": -1, "c": 1}) == "(0,-1,1)"


def test_leaf_kernel_has_unit_diagonal():
    X, y = _toy()
    feats = [f"x{j}" for j in range(X.shape[1])]
    base = fit_base(X, y, feats, {"x0": 1}, params=dict(max_depth=2, n_estimators=20, random_state=0))
    Psi = LeafMap(base, X).transform(X)
    assert np.allclose((Psi * Psi).sum(axis=1), 1.0)


def test_primal_basis_reproduces_the_kernel_when_untruncated():
    X, y = _toy()
    feats = [f"x{j}" for j in range(X.shape[1])]
    base = fit_base(X, y, feats, {}, params=dict(max_depth=2, n_estimators=20, random_state=0))
    Psi = LeafMap(base, X).transform(X)
    basis = primal_basis(Psi, var_keep=1.0)
    P = basis.project(Psi)
    assert np.allclose(P @ P.T, Psi @ Psi.T, atol=1e-8)


def test_centroid_gate_is_one_hot_and_packs_round_trip():
    Xs = np.random.default_rng(0).normal(size=(50, 3))
    params = kmeans_centroids(Xs, 2, seed=0, n_init=3)
    gamma = params.responsibilities(Xs)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert set(np.unique(gamma)) <= {0.0, 1.0}
    back = CentroidParams.unpack(params.pack(), 2, 3)
    assert np.allclose(back.mu, params.mu)


def test_mixture_gate_normalizes_and_packs_round_trip():
    Xs = np.random.default_rng(0).normal(size=(50, 3))
    params = kmeans_init(Xs, 3, seed=0, n_init=3)
    gamma = params.responsibilities(Xs)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    back = GateParams.unpack(params.pack(), 3, 3, diagonal=False)
    assert np.allclose(back.pack(), params.pack())


def test_narrow_mixture_gate_approaches_the_hard_gate():
    Xs = np.random.default_rng(1).normal(size=(60, 2))
    hard = kmeans_centroids(Xs, 2, seed=0, n_init=3)
    soft = GateParams(mu=hard.mu, log_s=np.full(2, -3.0), log_pi=np.zeros(2))
    assert np.allclose(soft.responsibilities(Xs), hard.responsibilities(Xs), atol=1e-6)


def test_expert_recovers_a_linear_correction_in_the_feature_map():
    rng = np.random.default_rng(0)
    n, m = 600, 5
    Psi = rng.normal(size=(n, m))
    beta = np.array([1.5, -1.0, 0.5, 0.0, 0.0])
    eta = Psi @ beta
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    fit = fit_mixture(to_tensor(Psi, DEVICE), to_tensor(y.astype(float), DEVICE),
                      to_tensor(np.zeros(n), DEVICE), to_tensor(np.ones((n, 1)), DEVICE),
                      n_irls=8)
    assert np.corrcoef(fit.beta[0].cpu().numpy(), beta)[0, 1] > 0.95


def test_zero_responsibility_gives_a_zero_correction():
    rng = np.random.default_rng(0)
    n, m = 200, 4
    Psi, y = rng.normal(size=(n, m)), rng.binomial(1, 0.5, size=n)
    gamma = np.zeros((n, 2)); gamma[:, 0] = 1.0
    fit = fit_mixture(to_tensor(Psi, DEVICE), to_tensor(y.astype(float), DEVICE),
                      to_tensor(np.zeros(n), DEVICE), to_tensor(gamma, DEVICE), n_irls=3)
    assert np.allclose(fit.beta[1].cpu().numpy(), 0.0)
    assert fit.n_eff[1] == 0.0


def test_mixture_probability_lies_between_the_expert_probabilities():
    rng = np.random.default_rng(2)
    n, m = 300, 4
    Psi = to_tensor(rng.normal(size=(n, m)), DEVICE)
    off = to_tensor(rng.normal(size=n) * 0.3, DEVICE)
    y = to_tensor(rng.binomial(1, 0.4, size=n).astype(float), DEVICE)
    g = rng.random((n, 2)); g = g / g.sum(axis=1, keepdims=True)
    gamma = to_tensor(g, DEVICE)
    fit = fit_mixture(Psi, y, off, gamma, n_irls=3)
    p = mixture_proba(Psi, off, gamma, fit).cpu().numpy()
    each = torch.sigmoid(off[None, :] + fit.beta @ Psi.T).cpu().numpy()
    assert np.all(p <= each.max(axis=0) + 1e-12)
    assert np.all(p >= each.min(axis=0) - 1e-12)


def test_search_improves_a_quadratic_objective_and_respects_the_budget():
    target = np.array([1.0, -2.0, 0.5, 0.0])
    res = es_search(lambda x: -float(((x - target) ** 2).sum()), np.zeros(4),
                    np.ones(4), seconds=2.0, max_evals=400, seed=0)
    assert res.score > res.score0
    assert res.n_evals <= 400
    assert len(res.trajectory) == res.n_evals + 1


def test_standardizer_leaves_a_constant_column_alone():
    X = np.column_stack([np.arange(10.0), np.ones(10)])
    Z = Standardizer(X)(X)
    assert np.allclose(Z[:, 1], 0.0)
    assert np.isclose(Z[:, 0].std(), 1.0)
