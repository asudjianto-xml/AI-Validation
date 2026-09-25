"""Does an optimization loop beat one-shot loss localization, and at what cost?

Every method runs to the largest budget once and its anytime curve is read at
the smaller budgets, which is valid because each of them is online: none looks
ahead, so a run truncated at a budget is the run that budget would have given.
Costs are loss reads throughout, with wall-clock reported alongside because the
currency charges nothing for a loss-model prediction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aiv.wsearch.constructor import FixedSupportConstructor, sample_candidates
from aiv.wsearch.evaluator import TouchMeter, WeaknessEvaluator, distinct_by_overlap
from aiv.wsearch.grid import BinGrid, Standardizer
from aiv.wsearch.lossmodel import AuxLossModel, b1_region_scores, region_embeddings
from aiv.wsearch.search import (
    bayes_opt,
    design_search,
    evolution,
    hill_climb,
    twinning_archive,
)
from aiv.wsearch.twinfolds import twin_folds

BUDGETS = (5342, 10000, 20000, 50000)


def at_budget(result, budget):
    """Best contrast the method had reached by `budget` loss reads."""
    reads, best = result.curve()
    ok = reads <= budget
    return float(best[ok].max()) if ok.any() else float("nan")


def run(system=None, m=171, K=10, n_bins=20, n_init=16, max_budget=50000,
        seeds=(0, 1, 2), b1_pool=4000, out_dir="runs/compare"):
    from aiv.predictive import PredictiveSystem

    system = system or PredictiveSystem().fit()
    feats = list(system.features)
    train, disc = system.splits["train"], system.splits["discovery"]
    r = system.row_loss(disc)
    r_train = system.row_loss(train)
    tau = float(np.quantile(r_train, 0.90))
    std = Standardizer.fit(train, feats)
    BinGrid.fit(train, feats, n_bins=n_bins)
    Xs = std.transform(disc)
    X_raw = disc[feats].to_numpy(float)
    folds = twin_folds(Xs, K=K)
    con = FixedSupportConstructor(Xs, m=m, metric="box")
    n = len(r)

    aux = AuxLossModel(depth=3, n_estimators=200, seed=0).fit(X_raw, r)
    Phi = aux.phi(X_raw)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        init = twinning_archive(Xs, folds, n_init, rng)

        # B1: one shot. Its scan costs no loss reads because the loss model
        # predicts rather than reads; wall-clock records what the scan cost.
        mt = TouchMeter()
        ev = WeaknessEvaluator(r, folds=folds, tau=tau, meter=mt)
        t0 = time.time()
        mt.charge(n, "loss_model_fit")
        regs = con.build_batch(sample_candidates(Xs, b1_pool, np.random.default_rng(seed), "box"))
        E = region_embeddings(Phi, [g.member_idx for g in regs])
        b1 = regs[int(np.argmax(b1_region_scores(E, aux)))]
        M_b1 = ev.contrast(b1.member_idx)
        rows.append({"seed": seed, "method": "B1_one_shot", "loss_reads": mt.reads,
                     "wall": time.time() - t0, "M": M_b1, "n_eval": 1,
                     "by_budget": {str(b): (M_b1 if mt.reads <= b else float("nan")) for b in BUDGETS}})

        specs = [
            ("B4_random", lambda mt, ev: design_search(
                con, ev, mt, max_budget,
                sample_candidates(Xs, max_budget // m + 2, np.random.default_rng(seed + 100), "box"))),
            ("S1_twinning_design", lambda mt, ev: design_search(
                con, ev, mt, max_budget,
                twinning_archive(Xs, folds, max_budget // m + 2, np.random.default_rng(seed + 200)))),
            ("S2_hill_climb", lambda mt, ev: hill_climb(
                con, ev, mt, max_budget, init, rng=np.random.default_rng(seed + 300))),
            ("S3_evolution", lambda mt, ev: evolution(
                con, ev, mt, max_budget, init, rng=np.random.default_rng(seed + 400))),
            ("S5_bo_overlap_0.7", lambda mt, ev: bayes_opt(
                con, ev, mt, max_budget, init, Phi, lam=0.0, omega_max=0.7,
                refit_every=5, rng=np.random.default_rng(seed + 500), aux_fit_cost=n)),
            ("S5_bo_no_constraint", lambda mt, ev: bayes_opt(
                con, ev, mt, max_budget, init, Phi, lam=0.0, omega_max=1.0,
                refit_every=5, rng=np.random.default_rng(seed + 600), aux_fit_cost=n)),
        ]
        for name, fn in specs:
            mt = TouchMeter()
            ev = WeaknessEvaluator(r, folds=folds, tau=tau, meter=mt)
            t0 = time.time()
            res = fn(mt, ev)
            members = [g.member_idx for _, g, _ in res.archive]
            order = np.argsort([-M for _, _, M in res.archive])
            rows.append({
                "seed": seed, "method": name, "loss_reads": res.loss_reads,
                "wall": time.time() - t0, "M": res.best_M, "n_eval": res.n_evaluations,
                "distinct_at_0.5": int(len(distinct_by_overlap(
                    [members[i] for i in order], n, 0.5))),
                "detail": res.detail,
                "by_budget": {str(b): at_budget(res, b) for b in BUDGETS},
            })
            print(f"seed {seed} {name:22s} M={res.best_M:.3f} evals={res.n_evaluations:4d} "
                  f"reads={res.loss_reads:6d} wall={time.time()-t0:6.1f}s", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({"budgets": list(BUDGETS), "m": m,
                                                  "rows": rows}, indent=2, default=float))
    return rows


if __name__ == "__main__":
    run()
