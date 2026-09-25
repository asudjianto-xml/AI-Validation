"""Iterative search over the admissible region class.

Every method here consumes the same constructor, the same evaluator and the same
loss-read meter, and every one starts from the same Twinning-derived archive, so
a difference between them is a difference in how the budget was allocated rather
than in what was measured or what was searched. Budgets are stated in loss reads
so that the one-shot benchmarks appear on the same axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aiv.wsearch.constructor import Candidate
from aiv.wsearch.evaluator import TouchMeter, WeaknessEvaluator, max_overlap
from aiv.wsearch.gp import CompositeGP, expected_improvement, lengthscale_groups
from aiv.wsearch.lossmodel import region_embeddings

_EPS = 1e-9


@dataclass
class SearchResult:
    method: str
    best_M: float
    best_region: object
    n_evaluations: int
    loss_reads: int
    history: list = field(default_factory=list)   # (loss_reads, best_M_so_far)
    archive: list = field(default_factory=list)   # (candidate, region, M)
    detail: dict = field(default_factory=dict)

    def curve(self):
        """Best contrast against loss reads, for the anytime comparison."""
        return np.array([h[0] for h in self.history]), np.array([h[1] for h in self.history])


class _Run:
    """Shared bookkeeping: evaluate, record, and stop when the budget is spent."""

    def __init__(self, con, ev: WeaknessEvaluator, meter: TouchMeter, budget: int):
        self.con, self.ev, self.meter, self.budget = con, ev, meter, budget
        self.archive, self.history = [], []
        self.best_M, self.best_region = -np.inf, None

    @property
    def spent(self) -> bool:
        return self.meter.reads >= self.budget

    def evaluate(self, cand: Candidate):
        region = self.con.build(cand)
        M = self.ev.contrast(region.member_idx)
        self.archive.append((cand, region, M))
        if M > self.best_M:
            self.best_M, self.best_region = M, region
        self.history.append((self.meter.reads, self.best_M))
        return region, M

    def result(self, method: str, **detail) -> SearchResult:
        return SearchResult(method, self.best_M, self.best_region, len(self.archive),
                            self.meter.reads, self.history, self.archive, detail)


def twinning_archive(Xs: np.ndarray, folds: np.ndarray, n_init: int, rng,
                     shape: str = "box", dirichlet_alpha: float = 1.0):
    """Initial candidates on empirical support, spread across the Twinning folds.

    Centers are observed rows drawn round-robin over folds so that the archive
    covers the observational population before any outcome redirects the search.
    Every adaptive method receives this same archive.
    """
    K = int(folds.max()) + 1
    p = Xs.shape[1]
    picks = []
    per_fold = [np.flatnonzero(folds == k) for k in range(K)]
    for i in range(n_init):
        pool = per_fold[i % K]
        picks.append(int(rng.choice(pool)))
    W = (np.full((n_init, p), 1.0 / p) if shape == "cube"
         else rng.dirichlet(np.full(p, dirichlet_alpha), size=n_init))
    return [Candidate(Xs[r].copy(), W[i]) for i, r in enumerate(picks)]


def _sample_shape(rng, p, alpha=1.0):
    return rng.dirichlet(np.full(p, alpha))


def _seed(run: "_Run", initial):
    """Evaluate the shared initial archive and start from its best member.

    Every adaptive method pays for the same archive and begins from the same
    point, so a later difference between them is allocation rather than a
    luckier start.
    """
    for cand in initial:
        if run.spent:
            break
        run.evaluate(cand)
    idx = int(np.argmax([M for _, _, M in run.archive]))
    return run.archive[idx][0], run.archive[idx][2]


# -- S1 / B4: nonadaptive designs -------------------------------------------


def design_search(con, ev, meter, budget, candidates) -> SearchResult:
    """Evaluate a fixed design and return its best. Twinning design when the
    candidates come from `twinning_archive`, random search when they do not."""
    run = _Run(con, ev, meter, budget)
    for cand in candidates:
        if run.spent:
            break
        run.evaluate(cand)
    return run.result("design")


# -- S2: hill climbing -------------------------------------------------------


def hill_climb(con, ev, meter, budget, initial, step: float = 0.5,
               shape_step: float = 0.3, contract: float = 0.5, min_step: float = 0.02,
               rng=None) -> SearchResult:
    """Best-improvement local search over the center and the shape.

    The neighborhood is one step along each center coordinate in each direction
    plus a reweighting of each face, so it is admissible by construction. The
    step contracts when no neighbor improves, which is what lets the same routine
    both traverse and refine.
    """
    rng = rng or np.random.default_rng(0)
    run = _Run(con, ev, meter, budget)
    cur, cur_M = _seed(run, initial)
    p = len(cur.z)
    while not run.spent and step >= min_step:
        best, best_M = None, cur_M
        for j in range(p):
            for sign in (1.0, -1.0):
                if run.spent:
                    break
                z = cur.z.copy()
                z[j] += sign * step
                _, M = run.evaluate(Candidate(z, cur.w))
                if M > best_M:
                    best, best_M = Candidate(z, cur.w), M
                if run.spent:
                    break
                w = cur.w.copy()
                w[j] = max(w[j] * (1.0 + sign * shape_step), _EPS)
                w = w / w.sum()
                _, M = run.evaluate(Candidate(cur.z, w))
                if M > best_M:
                    best, best_M = Candidate(cur.z, w), M
        if best is None:
            step *= contract
        else:
            cur, cur_M = best, best_M
    return run.result("hill_climb", final_step=step)


# -- S3: evolutionary search -------------------------------------------------


def evolution(con, ev, meter, budget, initial, sigma: float = 0.6,
              sigma_w: float = 0.4, rng=None, adapt: float = 0.85) -> SearchResult:
    """A (1+1) evolution strategy with the one-fifth success rule.

    The center is perturbed by a Gaussian step and the shape by a log-ratio step,
    so the shape stays on the simplex without a projection. The step size grows
    after a success and contracts after a failure, which keeps the strategy from
    either stalling or wandering.
    """
    rng = rng or np.random.default_rng(0)
    run = _Run(con, ev, meter, budget)
    cur, cur_M = _seed(run, initial)
    p = len(cur.z)
    while not run.spent:
        z = cur.z + sigma * rng.normal(size=p)
        w = cur.w * np.exp(sigma_w * rng.normal(size=p))
        w = np.maximum(w, _EPS)
        w = w / w.sum()
        _, M = run.evaluate(Candidate(z, w))
        if M > cur_M:
            cur, cur_M = Candidate(z, w), M
            sigma, sigma_w = sigma / adapt, sigma_w / adapt
        else:
            sigma, sigma_w = sigma * adapt, sigma_w * adapt
    return run.result("evolution", final_sigma=sigma)


# -- S5: Bayesian optimization ----------------------------------------------


def bayes_opt(con, ev, meter, budget, initial, Phi, lam: float = 0.0,
              n_pool: int = 512, omega_max: float = 0.7, rng=None,
              refit_every: int = 1, pool_spread: float = 0.6,
              aux_fit_cost: int = 0) -> SearchResult:
    """Surrogate-guided search with an acquisition overlap constraint.

    A proposal whose membership overlaps the evaluated archive by more than
    `omega_max` is rejected before it is measured: it would return nearly the
    contrast already recorded, so the read buys nothing. The constraint uses
    cached membership indices and costs no loss reads. Set `omega_max` to one to
    disable it, which is the ablation.

    `aux_fit_cost` charges the auxiliary loss model that supplies `Phi`, so that
    the surrogate is not credited with a representation it did not pay for.
    """
    rng = rng or np.random.default_rng(0)
    run = _Run(con, ev, meter, budget)
    if aux_fit_cost:
        meter.charge(aux_fit_cost, "loss_model_fit")
    for cand in initial:
        if run.spent:
            break
        run.evaluate(cand)

    p = len(run.archive[0][0].z)
    groups = lengthscale_groups(p, p)
    n_rejected, n_fits = 0, 0
    while not run.spent:
        coords = np.stack([c.as_tuple() for c, _, _ in run.archive])
        members = [g.member_idx for _, g, _ in run.archive]
        E = region_embeddings(Phi, members)
        y = np.array([M for _, _, M in run.archive])
        if n_fits == 0 or len(run.archive) % refit_every == 0:
            gp = CompositeGP(lam=lam, groups=groups).fit(coords, E, y)
            n_fits += 1

        anchor = run.best_region.candidate
        pool = [Candidate(anchor.z + pool_spread * rng.normal(size=p),
                          _sample_shape(rng, p)) for _ in range(n_pool)]
        pr = con.build_batch(pool)
        pm = [g.member_idx for g in pr]
        omega = max_overlap(pm, members, con.n)
        allowed = np.flatnonzero(omega <= omega_max)
        if not len(allowed):
            n_rejected += n_pool
            pool_spread *= 1.5          # the pool has collapsed onto the archive
            continue
        n_rejected += n_pool - len(allowed)
        Ep = region_embeddings(Phi, [pm[i] for i in allowed])
        Cp = np.stack([pool[i].as_tuple() for i in allowed])
        mu, sd = gp.predict(Cp, Ep)
        ei = expected_improvement(mu, sd, y.max())
        run.evaluate(pool[int(allowed[int(np.argmax(ei))])])
    return run.result("bayes_opt", gp_fits=n_fits, proposals_rejected=int(n_rejected),
                      lam=lam, omega_max=omega_max)
