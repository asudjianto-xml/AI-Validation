"""Stage 0: estimand and criterion diagnostics.

Run once per dataset, outside every method budget. It fixes the quantities the
rest of the protocol treats as predeclared -- the within-class ceiling, whether a
second weakness criterion carries structure the contrast does not, the stability
floor, and the support the confirmation stage can actually resolve -- and it
measures the candidate overlap that decides whether the correlated-noise model
can matter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from aiv.wsearch.constructor import Candidate, FixedSupportConstructor, sample_candidates
from aiv.wsearch.evaluator import TouchMeter, WeaknessEvaluator, overlap_fraction
from aiv.wsearch.grid import BinGrid, Standardizer
from aiv.wsearch.twinfolds import twin_folds


def required_support(r_train, target_halfwidth=0.25, sizes=(25, 50, 100, 150, 200, 300, 400, 600),
                     n_rep=200, n_boot=400, seed=0):
    """Smallest region support whose confirmation interval for M is narrow enough.

    Losses are drawn from the training distribution, so the calculation uses no
    discovery or confirmation outcome. The returned support is what Eq. (18)
    converts into a prevalence floor.
    """
    rng = np.random.default_rng(seed)
    r_train = np.asarray(r_train, float)
    out = {}
    for m in sizes:
        widths = []
        for _ in range(n_rep):
            inside = rng.choice(r_train, m)
            outside = rng.choice(r_train, 4 * m)
            b = [rng.choice(inside, m).mean() / rng.choice(outside, 4 * m).mean() - 1
                 for _ in range(n_boot)]
            lo, hi = np.quantile(b, [0.025, 0.975])
            widths.append((hi - lo) / 2)
        out[m] = float(np.median(widths))
    ok = [m for m, w in out.items() if w <= target_halfwidth]
    return (min(ok) if ok else None), out


def prevalence_floor(n_req, n_conf, epsilon=0.05):
    """Smallest prevalence whose confirmation support clears n_req with
    probability 1 - epsilon, from the normal lower bound on a binomial count."""
    z = stats.norm.ppf(1 - epsilon)
    pi = np.linspace(1e-4, 0.999, 20000)
    lower = n_conf * pi - z * np.sqrt(n_conf * pi * (1 - pi))
    ok = np.flatnonzero(lower >= n_req)
    return float(pi[ok[0]]) if len(ok) else None


def nondominated_fraction(a, b):
    """Fraction of points on the Pareto frontier of two maximized criteria."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    order = np.argsort(-a, kind="mergesort")
    best, front = -np.inf, np.zeros(len(a), dtype=bool)
    for i in order:
        if b[i] > best:
            front[i], best = True, b[i]
    return float(front.mean())


def run(system=None, split="discovery", n_cand=5000, m=None, K=10, n_bins=20,
        tau_q=0.90, seed=0, out_dir="runs/stage0"):
    from aiv.predictive import PredictiveSystem

    system = system or PredictiveSystem().fit()
    feats = list(system.features)
    train, disc = system.splits["train"], system.splits[split]
    conf = system.splits["confirm"]
    n_disc, n_conf = len(disc), len(conf)

    r_train = system.row_loss(train)
    r = system.row_loss(disc)
    tau = float(np.quantile(r_train, tau_q))

    # -- support the confirmation stage can resolve -------------------------
    n_req, width_curve = required_support(r_train, seed=seed)
    pi_min = prevalence_floor(n_req, n_conf) if n_req else None
    m = int(m or max(n_req or 150, int(np.ceil((pi_min or 0.0) * n_disc))))

    std = Standardizer.fit(train, feats)
    grid = BinGrid.fit(train, feats, n_bins=n_bins)
    Xs = std.transform(disc)
    folds = twin_folds(Xs, K=K, seed=seed)

    meter = TouchMeter()
    ev = WeaknessEvaluator(r, folds=folds, tau=tau, meter=meter)
    rng = np.random.default_rng(seed)

    results = {}
    for cls in ("box", "cube", "ball"):
        metric = "ball" if cls == "ball" else "box"
        shape = "cube" if cls in ("cube", "ball") else "box"
        con = FixedSupportConstructor(Xs, m=m, metric=metric)
        cands = sample_candidates(Xs, n_cand, np.random.default_rng(seed + hash(cls) % 1000), shape)
        regions = con.build_batch(cands)
        meas = [ev.measure(reg.member_idx) for reg in regions]
        M = np.array([x.contrast for x in meas])
        S = np.array([x.sd_in for x in meas])
        T = np.array([x.tail_in for x in meas])

        rep = [ev.replicates(reg.member_idx) for reg in regions]
        se = np.array([x.se for x in rep])
        floor = np.array([x.floor for x in rep])

        top = np.argsort(-M)[:200]
        results[cls] = {
            "m": m, "n_cand": n_cand,
            "ceiling_within_class": float(M.max()),
            "M": {"mean": float(M.mean()), "p50": float(np.median(M)),
                  "p95": float(np.quantile(M, 0.95)), "max": float(M.max())},
            "M_vs_S": {"pearson": float(stats.pearsonr(M, S)[0]),
                       "spearman": float(stats.spearmanr(M, S)[0]),
                       "nondominated": nondominated_fraction(M, S)},
            "M_vs_T": {"pearson": float(stats.pearsonr(M, T)[0]),
                       "spearman": float(stats.spearmanr(M, T)[0]),
                       "nondominated": nondominated_fraction(M, T)},
            "jackknife": {
                "se_median": float(np.median(se)),
                "se_median_top200": float(np.median(se[top])),
                "spearman_M_se": float(stats.spearmanr(M, se)[0]),
                "floor_p10_top200": float(np.quantile(floor[top], 0.10)),
                "floor_median_top200": float(np.median(floor[top])),
            },
            "_arrays": {"M": M, "S": S, "T": T, "se": se, "floor": floor,
                        "members": [reg.member_idx for reg in regions],
                        "top": top},
        }

    # -- overlap among the candidates a searcher actually visits ------------
    con = FixedSupportConstructor(Xs, m=m, metric="box")
    base = sample_candidates(Xs, 1, rng, "box")[0]
    overlap = {}
    for scale in (0.02, 0.05, 0.10, 0.25, 0.50):
        near = [Candidate(base.z + scale * rng.normal(size=len(base.z)), base.w) for _ in range(32)]
        frac = overlap_fraction([x.member_idx for x in con.build_batch(near)], len(Xs))
        off = frac[np.triu_indices_from(frac, k=1)]
        overlap[str(scale)] = {"mean": float(off.mean()), "p90": float(np.quantile(off, 0.90))}
    top_regions = results["box"]["_arrays"]["members"]
    tf = overlap_fraction([top_regions[i] for i in results["box"]["_arrays"]["top"][:64]], len(Xs))
    overlap["archive_top64"] = {
        "mean": float(tf[np.triu_indices_from(tf, k=1)].mean()),
        "max": float(tf[np.triu_indices_from(tf, k=1)].max()),
    }

    # -- descriptive unconstrained ceiling ---------------------------------
    top_m = np.argsort(-r)[:m]
    M_top = float(r[top_m].mean() / np.delete(r, top_m).mean() - 1)

    summary = {
        "n_train": len(train), "n_disc": n_disc, "n_conf": n_conf,
        "features": feats, "K": K, "n_bins": n_bins,
        "tau": tau, "tau_quantile": tau_q,
        "loss_mean_disc": float(r.mean()), "loss_p99_disc": float(np.quantile(r, 0.99)),
        "support": {"n_req": n_req, "pi_min": pi_min, "m": m,
                    "ci_halfwidth_by_m": width_curve},
        "ceiling_unconstrained_top_m": M_top,
        "classes": {k: {kk: vv for kk, vv in v.items() if kk != "_arrays"}
                    for k, v in results.items()},
        "overlap": overlap,
        "touches": dict(meter.by_tag, total=meter.reads),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage0.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "stage0_arrays.npz",
                        **{f"{c}_{k}": results[c]["_arrays"][k]
                           for c in results for k in ("M", "S", "T", "se", "floor")})
    return summary, results


if __name__ == "__main__":
    summary, _ = run()
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}, indent=2))
