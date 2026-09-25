"""The comparison: a monotone booster against mixtures of processes on its own
leaf kernel.

Protocol. The base model is fitted on the training split and is the object
under validation; its leaf partition supplies the covariance. The gate and the
expert corrections are fitted on the discovery split, which the base model has
not seen, so the residuals the processes model are out of sample. The gate is
searched against an area under the curve averaged over three rotations of a
stratified split of discovery, and the confirmation split is read once per
reported configuration.

Because the mixture draws on both the training split, through the kernel, and
the discovery split, through the corrections, a single booster fitted on
discovery alone is not the only comparator it owes an answer to; one fitted on
the two splits together sees the same rows and is reported alongside.

Monotonicity is checked rather than assumed. Each expert is a free linear
functional of the leaf indicators, so it inherits the partition of the
constrained booster and not its constraint, and under a gate that varies with
x the mixture can move against a constraint even where every expert respects
it. The finite-difference sweep reports the rate at which the predicted
probability moves the wrong way along each constrained feature.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from aiv.moegp.experts import device_of, fit_mixture, mixture_proba, to_tensor
from aiv.moegp.gate import CentroidParams, Standardizer, kmeans_centroids
from aiv.moegp.kernel import BASE_PARAMS, LeafMap, base_logit, fit_base, monotone_string, primal_basis
from aiv.moegp.search import es_search, make_folds, objective

VAR_KEEP = 0.999
N_IRLS = 3
N_FOLDS = 3


def _scores(y, p) -> dict:
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p))}


def monotonicity_violations(predict, X, features, monotone, delta: float = 0.25,
                            tol: float = 1e-9) -> dict:
    """Rate at which the predicted probability moves against a constraint.

    Each constrained feature is shifted by `delta` of its standard deviation
    and the sign of the change in predicted probability is compared with the
    required direction.
    """
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    p0 = predict(X)
    out = {}
    for j, f in enumerate(features):
        c = monotone.get(f, 0)
        if c == 0:
            continue
        Xp = X.copy()
        Xp[:, j] = Xp[:, j] + delta * sd[j]
        d = predict(Xp) - p0
        out[f] = float((c * d < -tol).mean())
    out["any_feature_mean"] = float(np.mean([v for k, v in out.items() if k != "any_feature_mean"]))
    return out


class MixturePredictor:
    """Fitted gate and experts, as a callable on raw covariates."""

    def __init__(self, base, leafmap, basis, standardizer, params, fit, device):
        self.base, self.leafmap, self.basis = base, leafmap, basis
        self.std, self.params, self.fit, self.device = standardizer, params, fit, device

    def __call__(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        Psi = to_tensor(self.basis.project(self.leafmap.transform(X)), self.device)
        off = to_tensor(base_logit(self.base, X), self.device)
        gam = to_tensor(self.params.responsibilities(self.std(X)), self.device)
        return mixture_proba(Psi, off, gam, self.fit).cpu().numpy()


def fit_final(base, leafmap, basis, standardizer, params, X_fit, y_fit, device,
              n_irls: int = N_IRLS) -> MixturePredictor:
    Psi = to_tensor(basis.project(leafmap.transform(X_fit)), device)
    off = to_tensor(base_logit(base, X_fit), device)
    y = to_tensor(np.asarray(y_fit, float), device)
    gam = to_tensor(params.responsibilities(standardizer(X_fit)), device)
    fit = fit_mixture(Psi, y, off, gam, n_irls=n_irls)
    return MixturePredictor(base, leafmap, basis, standardizer, params, fit, device)


def run(n_components: int = 2, seeds=(0, 1, 2), seconds: float = 600.0,
        out_dir: str = "runs/moegp", sigma0: float = 0.5,
        max_evals: int | None = None, tag: str = "") -> dict:
    from aiv.predictive import MONOTONE, PredictiveSystem

    t_start = time.time()
    system = PredictiveSystem().fit()
    feats = list(system.features)
    tr, di, co = (system.splits[s] for s in ("train", "discovery", "confirm"))
    Xtr, Xdi, Xco = (d[feats].to_numpy(float) for d in (tr, di, co))
    ytr, ydi, yco = (d[system.target].to_numpy().astype(int) for d in (tr, di, co))

    device = device_of()
    out = {
        "features": feats,
        "monotone": monotone_string(feats, MONOTONE),
        "base_params": {k: v for k, v in BASE_PARAMS.items()},
        "n_components": n_components,
        "sizes": {"train": len(Xtr), "discovery": len(Xdi), "confirm": len(Xco)},
        "settings": {"var_keep": VAR_KEEP, "n_irls": N_IRLS, "n_folds": N_FOLDS,
                     "seconds_per_seed": seconds, "sigma0": sigma0,
                     "max_evals": max_evals, "device": str(device)},
        "baselines": {},
        "mixture": {},
    }

    # --- single-model comparators -----------------------------------------
    def record(key, model):
        p1 = lambda A: model.predict_proba(np.asarray(A, float))[:, 1]
        s = _scores(yco, p1(Xco))
        s["monotonicity"] = monotonicity_violations(p1, Xco, feats, MONOTONE)
        out["baselines"][key] = s
        return model

    base = record("monotone_xgb_train", fit_base(Xtr, ytr, feats, MONOTONE))
    record("monotone_xgb_discovery", fit_base(Xdi, ydi, feats, MONOTONE))
    record("monotone_xgb_train_plus_discovery",
           fit_base(np.vstack([Xtr, Xdi]), np.concatenate([ytr, ydi]), feats, MONOTONE))
    u = xgb.XGBClassifier(**BASE_PARAMS); u.fit(Xdi, ydi)
    record("unconstrained_xgb_discovery", u)

    # --- leaf kernel and its primal basis ---------------------------------
    leafmap = LeafMap(base, np.vstack([Xtr, Xdi, Xco]))
    Psi_di = leafmap.transform(Xdi)
    basis = primal_basis(Psi_di, var_keep=VAR_KEEP)
    out["kernel"] = {"n_trees": leafmap.n_trees, "leaf_dim": leafmap.dim,
                     "primal_dim": basis.m, "kept_trace": basis.kept_trace}

    standardizer = Standardizer(Xdi)
    Xs_di = standardizer(Xdi)
    off_di = base_logit(base, Xdi)
    P_di = basis.project(Psi_di)
    folds = make_folds(Xs_di, P_di, ydi, off_di, n_folds=N_FOLDS, seed=0, device=device)

    def confirm_scores(params) -> tuple:
        pred = fit_final(base, leafmap, basis, standardizer, params, Xdi, ydi, device)
        s = _scores(yco, pred(Xco))
        s["monotonicity"] = monotonicity_violations(pred, Xco, feats, MONOTONE)
        return s, pred

    # --- no gate: one process on the whole discovery split ----------------
    single = CentroidParams(mu=np.zeros((1, Xs_di.shape[1])))
    s1, _ = confirm_scores(single)
    out["mixture"]["single_process_no_gate"] = {
        "inner_auc": objective(folds, single, n_irls=N_IRLS), "confirm": s1}

    # --- the searched mixture, one record per seed ------------------------
    def score_fn(vec):
        return objective(folds, CentroidParams.unpack(vec, n_components, Xs_di.shape[1]),
                         n_irls=N_IRLS)

    records = []
    for seed in seeds:
        init = kmeans_centroids(Xs_di, n_components, seed=seed)
        s_init, _ = confirm_scores(init)
        res = es_search(score_fn, init.pack(), init.scales(), seconds=seconds,
                        max_evals=max_evals, seed=seed, sigma0=sigma0)
        found = CentroidParams.unpack(res.x, n_components, Xs_di.shape[1])
        s_found, _ = confirm_scores(found)
        records.append({
            "seed": seed, "n_params": res.n_params, "n_evals": res.n_evals,
            "seconds": res.seconds,
            "inner_auc_init": res.score0, "inner_auc_searched": res.score,
            "confirm_init": s_init, "confirm_searched": s_found,
            "centroids_standardized": found.mu.tolist(),
            "centroids_raw": (found.mu * standardizer.scale + standardizer.mean).tolist(),
            "trajectory": [float(v) for v in res.trajectory[::10]],
        })
        print(f"seed {seed}: inner {res.score0:.4f} -> {res.score:.4f} "
              f"({res.n_evals} evals, {res.seconds:.0f}s) | confirm AUC "
              f"{s_init['auc']:.4f} -> {s_found['auc']:.4f}", flush=True)

    out["mixture"]["kmeans_gate"] = records
    best = max(records, key=lambda r: r["inner_auc_searched"])
    out["mixture"]["headline_seed"] = best["seed"]
    out["seconds_total"] = time.time() - t_start

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"moegp_H{n_components}{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default="runs/moegp")
    ap.add_argument("--max-evals", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run(n_components=args.components, seeds=tuple(args.seeds), seconds=args.seconds,
        out_dir=args.out_dir, max_evals=args.max_evals, tag=args.tag)


if __name__ == "__main__":
    main()
