"""One-shot benchmarks and the surrogate, run end to end on one task.

Runs B0, B1 and B2 against the same fixed-support constructor and evaluator,
freezes the strongest region from each, confirms them on untouched evidence
under a materiality margin with false-discovery-rate control, and reports the
posterior calibration of the composite-kernel surrogate with and without the
overlap covariance. Everything is charged in loss reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aiv.wsearch.confirm import bca_confirm, benjamini_hochberg, holm
from aiv.wsearch.constructor import Candidate, FixedSupportConstructor, sample_candidates
from aiv.wsearch.evaluator import (
    TouchMeter,
    WeaknessEvaluator,
    distinct_by_overlap,
    max_overlap,
    pooled_within_variance,
    resampling_covariance,
)
from aiv.wsearch.gp import CompositeGP, interval_coverage
from aiv.wsearch.grid import BinGrid, Standardizer
from aiv.wsearch.lossmodel import (
    AuxLossModel,
    b1_region_scores,
    error_tree_region,
    loss_kernel_partition,
    region_embeddings,
)
from aiv.wsearch.twinfolds import twin_folds


def _nearest_candidate(Xs, idx, m, con):
    """Express an arbitrary member set as an admissible fixed-support region.

    A tree leaf or a spectral cluster is not a candidate of the declared class,
    so it is mapped to the admissible region centered on its centroid. The
    reported contrast is then the one the common evaluator measures for every
    method, and the difference from the raw part's contrast is the cost of
    expressing it in the declared representation.
    """
    return con.build(Candidate.cube(Xs[idx].mean(0)))


def run(system=None, m=171, K=10, n_bins=20, n_cand=4000, margin=0.5, q=0.05,
        seed=0, out_dir="runs/benchmarks"):
    from aiv.predictive import PredictiveSystem

    system = system or PredictiveSystem().fit()
    feats = list(system.features)
    train, disc, conf = (system.splits[s] for s in ("train", "discovery", "confirm"))
    r_train = system.row_loss(train)
    r = system.row_loss(disc)
    r_conf = system.row_loss(conf)

    std = Standardizer.fit(train, feats)
    grid = BinGrid.fit(train, feats, n_bins=n_bins)
    Xs = std.transform(disc)
    X_raw = disc[feats].to_numpy(float)
    folds = twin_folds(Xs, K=K, seed=seed)

    con = FixedSupportConstructor(Xs, m=m, metric="box")
    tau = float(np.quantile(r_train, 0.90))

    def fresh():
        """A method's own meter. The grand total it charges at construction is
        the one-time cost of caching the complement mean."""
        mt = TouchMeter()
        return mt, WeaknessEvaluator(r, folds=folds, tau=tau, meter=mt)

    # -- auxiliary loss model, shared by B1, B2 and the surrogate ------------
    aux = AuxLossModel(depth=3, n_estimators=200, seed=seed).fit(X_raw, r)
    fit_cost = len(r)          # every discovery loss read once as a target
    Phi = aux.phi(X_raw)

    results, promoted = {}, []

    # -- B0: error tree -----------------------------------------------------
    mt, ev = fresh()
    leaf_idx, _ = error_tree_region(X_raw, r, m=m, seed=seed)
    mt.charge(len(r), "tree_fit")
    b0 = _nearest_candidate(Xs, leaf_idx, m, con)
    results["B0_error_tree"] = _record(ev, b0, mt, raw_idx=leaf_idx)
    promoted.append(("B0_error_tree", b0))

    # -- B1: region-averaged loss-model localization ------------------------
    mt, ev = fresh()
    mt.charge(fit_cost, "loss_model_fit")
    cands = sample_candidates(Xs, n_cand, np.random.default_rng(seed), "box")
    regions = con.build_batch(cands)
    E_all = region_embeddings(Phi, [g.member_idx for g in regions])
    scores = b1_region_scores(E_all, aux)
    b1 = regions[int(np.argmax(scores))]
    results["B1_loss_model"] = _record(ev, b1, mt)
    promoted.append(("B1_loss_model", b1))

    # -- B2: loss-kernel partitioning ---------------------------------------
    mt, ev = fresh()
    mt.charge(fit_cost, "loss_model_fit")
    part_idx, _b2_labels, b2_router = loss_kernel_partition(
        aux, X_raw, r, m=m, features=feats, n_clusters=8, seed=seed)
    mt.charge(len(r), "cluster_scoring")
    b2 = _nearest_candidate(Xs, part_idx, m, con)
    results["B2_loss_kernel"] = _record(ev, b2, mt, raw_idx=part_idx)
    promoted.append(("B2_loss_kernel", b2))

    # -- B4: random candidate search at a matched read budget ---------------
    budget = max(results[k]["loss_reads"] for k in results)
    mt, ev = fresh()
    n_rand = max(1, (budget - mt.reads) // m)
    rand_regions = con.build_batch(sample_candidates(Xs, n_rand, np.random.default_rng(seed + 1), "box"))
    rand_M = np.array([ev.contrast(g.member_idx) for g in rand_regions])
    best_rand = rand_regions[int(np.argmax(rand_M))]
    results["B4_random"] = _record(ev, best_rand, mt)
    results["B4_random"]["n_candidates"] = n_rand
    promoted.append(("B4_random", best_rand))

    # -- diagnostics, charged to nobody -------------------------------------
    diag_meter, diag_ev = fresh()
    results["B1_loss_model"]["pointwise_alternative"] = _pointwise_gap(
        diag_ev, aux, std, cands, regions, scores
    )

    # -- surrogate calibration: overlap covariance against independent noise --
    results["surrogate"] = {
        f"spread_{sp}": _calibration(con, diag_ev, Phi, Xs, regions, E_all, seed, spread=sp)
        for sp in (0.18, 0.60, 1.20)
    }
    results["evaluator_deterministic"] = determinism_check(
        con, diag_ev, regions[int(np.argmax([diag_ev.contrast(g.member_idx) for g in regions[:50]]))].candidate
    )

    # -- freeze and confirm --------------------------------------------------
    rows = []
    for name, region in promoted:
        frozen = con.freeze(region, std, grid, X_raw)
        c = bca_confirm(r_conf, frozen.mask(conf), margin=margin, n_boot=4000, seed=seed)
        rows.append({
            "method": name,
            "M_disc": results[name]["M"],
            "M_conf": c.contrast,
            "shrinkage": results[name]["M"] - c.contrast,
            "ci": [c.ci_low, c.ci_high],
            "p": c.p_value,
            "support_conf": c.support,
            "support_frozen_disc": frozen.support_disc,
            "snap_inflation": frozen.support_disc / region.m,
        })
    # B2 in its own declared class: the leaf-kernel router is a deployable
    # membership rule, so the cluster need not be expressed as a box to be
    # confirmed. Reported beside the box-mapped variant, never in place of it,
    # because the two answer questions about different region classes.
    c = bca_confirm(r_conf, b2_router.weak_mask(conf), margin=margin, n_boot=4000, seed=seed)
    rows.append({
        "method": "B2_leaf_router", "M_disc": results["B2_loss_kernel"]["M_raw_part"],
        "M_conf": c.contrast, "shrinkage": results["B2_loss_kernel"]["M_raw_part"] - c.contrast,
        "ci": [c.ci_low, c.ci_high], "p": c.p_value, "support_conf": c.support,
        "support_frozen_disc": results["B2_loss_kernel"]["n_raw_part"],
        "snap_inflation": float("nan"),
        "router_label_agreement": b2_router.label_agreement(disc, _b2_labels),
        "note": "own class, support not matched to m",
    })

    p = np.array([x["p"] for x in rows])
    rej_bh, adj_bh = benjamini_hochberg(p, q)
    rej_holm, adj_holm = holm(p, q)
    for i, x in enumerate(rows):
        x.update(bh_reject=bool(rej_bh[i]), bh_adj=float(adj_bh[i]),
                 holm_reject=bool(rej_holm[i]), holm_adj=float(adj_holm[i]))

    summary = {
        "m": m, "margin": margin, "q": q, "n_disc": len(disc), "n_conf": len(conf),
        "aux_reconstruction_error": aux.reconstruction_error,
        "discovery": results,
        "confirmation": rows,
        "diagnostic_reads_uncharged": diag_meter.reads,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmarks.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _record(ev, region, meter, raw_idx=None) -> dict:
    mm = ev.measure(region.member_idx)
    rep = ev.replicates(region.member_idx)
    out = {"M": mm.contrast, "sd_in": mm.sd_in, "tail_in": mm.tail_in,
           "se_jack": rep.se, "floor": rep.floor,
           "loss_reads": int(meter.reads), "reads_by_tag": dict(meter.by_tag), "m": mm.m}
    if raw_idx is not None:
        out["M_raw_part"] = ev.contrast(np.asarray(raw_idx))
        out["n_raw_part"] = int(len(raw_idx))
    return out


def _pointwise_gap(ev, aux, std, cands, regions, region_scores) -> dict:
    """What ranking candidate centers by the pointwise prediction would have found."""
    centers = std.inverse(np.stack([c.z for c in cands]))
    point = aux.predict(centers)
    best_point = regions[int(np.argmax(point))]
    truth = np.array([ev.contrast(g.member_idx) for g in regions])
    return {
        "M_pointwise_choice": ev.contrast(best_point.member_idx),
        "corr_region_avg": float(np.corrcoef(region_scores, truth)[0, 1]),
        "corr_pointwise": float(np.corrcoef(point, truth)[0, 1]),
    }


def _calibration(con, ev, Phi, Xs, regions, E_all, seed, n_local=200, spread=0.18) -> dict:
    """Surrogate behavior on a local archive, where an optimizer concentrates.

    Reports three treatments of the nugget on the same data. The response is
    deterministic given the discovery losses, so none of them is a measurement
    model; the question is which trades fit against coverage acceptably for a
    piecewise-constant response. Also reports the share of evaluations that
    duplicate an earlier one, which is what overlap actually costs.
    """
    rng = np.random.default_rng(seed + 2)
    M_all = np.array([ev.contrast(g.member_idx) for g in regions])
    base = regions[int(np.argmax(M_all))].candidate
    local = [Candidate(base.z + spread * rng.normal(size=len(base.z)), base.w) for _ in range(n_local)]
    lr = con.build_batch(local)
    members = [g.member_idx for g in lr]
    meas = [ev.measure(mi) for mi in members]
    y = np.array([mm.contrast for mm in meas])
    coords = np.stack([c.as_tuple() for c in local])   # center and shape: the Matern
                                                       # component must see both, or two
                                                       # candidates differing only in their
                                                       # faces look identical to it
    E = region_embeddings(Phi, members)

    s2 = pooled_within_variance(meas)
    mean_out = float(np.mean([mm.mean_out for mm in meas]))
    Sigma = resampling_covariance(members, Phi.shape[0], s2, mean_out)
    d = np.sqrt(np.diag(Sigma))
    Sigma_corr = Sigma / np.outer(d, d)
    iu = np.triu_indices_from(Sigma, k=1)

    cut = int(0.7 * n_local)
    tr, te = np.arange(cut), np.arange(cut, n_local)
    out = {
        "spread": spread,
        "y_sd": float(y.std()),
        "mean_overlap": float(Sigma_corr[iu].mean()),
        "duplicate_share_over_0.9": float(np.mean(
            [max_overlap([members[i]], members[:i], Phi.shape[0])[0] > 0.9
             for i in range(1, len(members))])),
        "distinct_at_0.5": int(len(distinct_by_overlap(
            [members[i] for i in np.argsort(-y)], Phi.shape[0], 0.5))),
        "n_evaluated": n_local,
    }

    ref = CompositeGP(lam=0.5).fit(coords[tr], E[tr], y[tr])
    K = ref._k(coords, E) / ref.sigma2
    out["kernel_overlap_correlation"] = float(np.corrcoef(K[iu], Sigma_corr[iu])[0, 1])

    mu, sd = ref.predictive(coords[te], E[te])
    out["fitted_nugget"] = {"coverage90": interval_coverage(mu, sd, y[te], 0.90),
                            "rmse": float(np.sqrt(((mu - y[te]) ** 2).mean())),
                            "sigma2": ref.sigma2, "nugget": ref.noise_scale}

    bare = CompositeGP(lam=0.5, jitter=1e-8).fit(coords[tr], E[tr], y[tr])
    bare.noise_scale = 1e-8
    bare._factor(coords[tr], E[tr], y[tr])
    mu, sd = bare.predict(coords[te], E[te])
    out["jitter_only"] = {"coverage90": interval_coverage(mu, sd, y[te], 0.90),
                          "rmse": float(np.sqrt(((mu - y[te]) ** 2).mean())),
                          "sigma2": bare.sigma2}

    asn = CompositeGP(lam=0.5).fit(coords[tr], E[tr], y[tr], noise=Sigma[np.ix_(tr, tr)])
    mu, sd = asn.predictive(coords[te], E[te], noise_var=np.diag(Sigma)[te],
                            noise_cross=Sigma[np.ix_(tr, te)])
    out["sigma_as_noise"] = {"coverage90": interval_coverage(mu, sd, y[te], 0.90),
                             "rmse": float(np.sqrt(((mu - y[te]) ** 2).mean())),
                             "sigma2": asn.sigma2, "noise_scale": asn.noise_scale,
                             "sigma2_at_lower_bound": bool(asn.sigma2 < 1e-5)}

    for lam, tag in ((1.0, "matern_only"), (0.0, "loss_geometry_only")):
        g = CompositeGP(lam=lam).fit(coords[tr], E[tr], y[tr])
        mu, sd = g.predictive(coords[te], E[te])
        out[tag] = {"coverage90": interval_coverage(mu, sd, y[te], 0.90),
                    "rmse": float(np.sqrt(((mu - y[te]) ** 2).mean()))}
    return out


def determinism_check(con, ev, cand) -> bool:
    """The premise of the nugget argument: repeated evaluation returns the same value."""
    return ev.contrast(con.build(cand).member_idx) == ev.contrast(con.build(cand).member_idx)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))
