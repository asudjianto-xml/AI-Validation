"""Reproduction of the earlier mixture-of-boosters comparison, with the
monotonicity of the mixture measured.

The configuration is the one that produced the earlier numbers: MoDeVa's
mixture of experts with monotone XGBoost experts, k-means centroids passed
directly, everything fitted on the discovery split and read once on
confirmation, and a (1+1) evolution strategy over the centroid coordinates in
standardized space. The strategy is run for a fixed number of evaluations
rather than a fixed wall-clock budget, so the trajectory is reproducible.

The objective is selectable. Area under the curve is the one the earlier run
used; log loss is the alternative and is the better-posed criterion, because it
is additive over rows, so the quantity the search minimizes is the size-weighted
sum of the per-cluster scores that the weakness report gives, and because it is
a proper scoring rule and therefore charges for the predicted level rather than
for the ranking alone.

The check added here is on monotonicity. The constraint is verified on the
fitted expert objects in the earlier run, which establishes that each expert
carries it; it does not establish that the mixture does, because the gate
varies with x and contributes a term to the derivative of the combined
prediction that the experts' own slopes do not control.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

N_EVALS = 181  # the count the earlier wall-clock budget reached


def monotonicity_violations(predict, X, features, monotone, delta: float = 0.25,
                            tol: float = 1e-9) -> dict:
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
        out[f] = float((c * (predict(Xp) - p0) < -tol).mean())
    out["any_feature_mean"] = float(np.mean([v for k, v in out.items() if k != "any_feature_mean"]))
    return out


def run(n_evals: int = N_EVALS, out_dir: str = "runs/moegp",
        objective: str = "auc", tag: str = "") -> dict:
    from modeva.models import MoMoEClassifier, MoXGBClassifier

    from aiv.predictive import MONOTONE, PredictiveSystem

    t0 = time.time()
    system = PredictiveSystem().fit()
    feats = list(system.features)
    va, te = system.splits["discovery"], system.splits["confirm"]
    Xva, Xte = (d[feats].to_numpy(float) for d in (va, te))
    yva, yte = (d["default"].to_numpy().astype(int) for d in (va, te))
    mono = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in feats) + ")"
    XP = dict(max_depth=2, n_estimators=200, learning_rate=0.05, subsample=0.9,
              colsample_bytree=0.9, random_state=0, monotone_constraints=mono)

    Xi, Xs, yi, ys = train_test_split(Xva, yva, train_size=0.7, random_state=0, stratify=yva)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd == 0] = 1.0

    def mixture(C_std, X_fit, y_fit, tag="m"):
        m = MoMoEClassifier(name=tag, n_clusters=len(C_std), cluster_method="kmeans",
                            centroids=C_std * sd + mu, expert="xgboost",
                            feature_names=feats, **XP)
        m.fit(X_fit, y_fit)
        return m

    if objective not in ("auc", "logloss"):
        raise ValueError(f"objective must be auc|logloss, got {objective!r}")

    def criterion(y_true, p) -> float:
        """Oriented so the strategy maximizes."""
        return float(roc_auc_score(y_true, p)) if objective == "auc" else -float(log_loss(y_true, p))

    def inner_score(C_std):
        try:
            m = mixture(C_std, Xi, yi)
        except Exception:
            return -1e9
        return criterion(ys, m.predict_proba(Xs)[:, 1])

    base_inner = MoXGBClassifier(name="b", **XP)
    base_inner.fit(Xi, yi)
    a_base = criterion(ys, base_inner.predict_proba(Xs)[:, 1])
    C0 = KMeans(n_clusters=2, n_init=20, random_state=0).fit((Xi - mu) / sd).cluster_centers_
    a0 = inner_score(C0)
    print(f"inner-split {objective}   base {a_base:.4f}   kmeans centroids {a0:.4f}", flush=True)

    rng = np.random.default_rng(0)
    cur, cur_a, sig = C0.copy(), a0, 0.5
    traj = [cur_a]
    for _ in range(n_evals):
        cand = cur + sig * rng.normal(size=cur.shape)
        a = inner_score(cand)
        if a > cur_a:
            cur, cur_a, sig = cand, a, sig / 0.85
        else:
            sig *= 0.85
        if sig < 0.02:
            sig = 0.5
        traj.append(cur_a)
    print(f"searched {n_evals} centroid sets: inner {objective} {a0:.4f} -> {cur_a:.4f}", flush=True)

    out = {"n_evals": n_evals, "objective": objective, "inner": {"base": a_base, "kmeans": a0, "searched": cur_a},
           "trajectory": [float(v) for v in traj[::10]],
           "centroids_standardized": cur.tolist(), "confirm": {}}

    b2 = MoXGBClassifier(name="bf", **XP)
    b2.fit(Xva, yva)
    p = b2.predict_proba(Xte)[:, 1]
    out["confirm"]["single_monotone_xgb"] = {
        "auc": float(roc_auc_score(yte, p)), "logloss": float(log_loss(yte, p)),
        "monotonicity": monotonicity_violations(
            lambda A: b2.predict_proba(A)[:, 1], Xte, feats, MONOTONE)}

    for tag, C in (("kmeans_centroids", C0), ("searched_centroids", cur)):
        m = mixture(C, Xva, yva, tag=tag)
        p = m.predict_proba(Xte)[:, 1]
        out["confirm"][tag] = {
            "auc": float(roc_auc_score(yte, p)), "logloss": float(log_loss(yte, p)),
            "monotonicity": monotonicity_violations(
                lambda A: m.predict_proba(A)[:, 1], Xte, feats, MONOTONE)}

    for k, v in out["confirm"].items():
        print(f"{k:<24} confirm AUC {v['auc']:.4f}  logloss {v['logloss']:.5f}  "
              f"monotonicity violations {v['monotonicity']['any_feature_mean']:.4f}", flush=True)

    out["seconds_total"] = time.time() - t0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"prior_moe{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-evals", type=int, default=N_EVALS)
    ap.add_argument("--out-dir", default="runs/moegp")
    ap.add_argument("--objective", default="auc", choices=["auc", "logloss"])
    ap.add_argument("--tag", default="")
    run(**vars(ap.parse_args()))
