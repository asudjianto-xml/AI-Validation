"""Per-cluster log score of the model under validation.

The clusters come from the mixture-of-experts search, so the partition is
supervised: the centers were moved to maximize the mixture's held-out
discrimination, which is an objective that reads the outcomes, rather than to
summarize the covariate geometry. The question here is whether a partition
found that way isolates a region where the model under validation is actually
weak, and the comparison that answers it is against the k-means partition of
the same data, which is the unsupervised alternative.

For each cluster the report gives the logarithmic score of the base model, the
score of the mixture, and the two quantities the frontier is stated in: lift,
the cluster's mean loss against the overall mean loss, and coverage, the share
of the sample's total loss the cluster holds.

A high log score on its own does not establish weakness, because the score is
bounded below by the entropy of the outcome and a region of high prevalence
scores badly under any model. The reference reported alongside is the score of
the constant predictor at the cluster's own observed rate, H(pi_h); the excess
over it is the part of the score attributable to the model rather than to the
uncertainty of the outcome, and a positive excess says the model is worse in
that region than knowing nothing beyond the region's base rate. Everything is
measured on the confirmation split, which neither the clustering nor the
experts have seen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_EPS = 1e-12


def log_score(y, p) -> np.ndarray:
    """Row-wise logarithmic score; the mean is the log loss."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def entropy(rate: float) -> float:
    """Log score of the constant predictor at the observed rate."""
    r = min(max(float(rate), _EPS), 1.0 - _EPS)
    return float(-(r * np.log(r) + (1.0 - r) * np.log(1.0 - r)))


def cluster_table(labels, y, loss_base, loss_mix, p_base, n_components) -> list:
    total = loss_base.sum()
    overall = loss_base.mean()
    rows = []
    for h in range(n_components):
        m = labels == h
        n = int(m.sum())
        if n == 0:
            rows.append({"cluster": h, "n": 0})
            continue
        rate = float(y[m].mean())
        const = entropy(rate)
        rows.append({
            "cluster": h, "n": n, "share": float(n / len(y)),
            "default_rate": rate,
            "mean_predicted": float(p_base[m].mean()),
            "constant_log_score": const,
            "excess_over_constant": float(loss_base[m].mean() - const),
            "base_log_score": float(loss_base[m].mean()),
            "mixture_log_score": float(loss_mix[m].mean()),
            "improvement": float(loss_base[m].mean() - loss_mix[m].mean()),
            "lift": float(loss_base[m].mean() / overall),
            "coverage": float(loss_base[m].sum() / total),
        })
    return rows


def run(out_dir: str = "runs/moegp", prior: str = "runs/moegp/prior_moe.json") -> dict:
    from modeva.models import MoMoEClassifier, MoXGBClassifier

    from aiv.predictive import MONOTONE, PredictiveSystem

    rec = json.loads(Path(prior).read_text())
    system = PredictiveSystem().fit()
    feats = list(system.features)
    va, te = system.splits["discovery"], system.splits["confirm"]
    Xva, Xte = (d[feats].to_numpy(float) for d in (va, te))
    yva, yte = (d["default"].to_numpy().astype(int) for d in (va, te))
    mono = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in feats) + ")"
    XP = dict(max_depth=2, n_estimators=200, learning_rate=0.05, subsample=0.9,
              colsample_bytree=0.9, random_state=0, monotone_constraints=mono)

    # the standardization the search used: moments of the inner fitting split
    Xi, _, yi, _ = train_test_split(Xva, yva, train_size=0.7, random_state=0, stratify=yva)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd == 0] = 1.0

    base = MoXGBClassifier(name="base", **XP)
    base.fit(Xva, yva)
    p_base = base.predict_proba(Xte)[:, 1]
    loss_base = log_score(yte, p_base)

    C_km = KMeans(n_clusters=2, n_init=20, random_state=0).fit((Xi - mu) / sd).cluster_centers_
    C_se = np.asarray(rec["centroids_standardized"], float)

    out = {
        "base": {"auc": float(roc_auc_score(yte, p_base)), "log_score": float(loss_base.mean())},
        "partitions": {},
    }
    Xte_s = (Xte - mu) / sd
    for tag, C in (("kmeans", C_km), ("searched", C_se)):
        m = MoMoEClassifier(name=tag, n_clusters=len(C), cluster_method="kmeans",
                            centroids=C * sd + mu, expert="xgboost",
                            feature_names=feats, **XP)
        m.fit(Xva, yva)
        p_mix = m.predict_proba(Xte)[:, 1]
        loss_mix = log_score(yte, p_mix)
        labels = np.argmin(((Xte_s[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
        out["partitions"][tag] = {
            "mixture": {"auc": float(roc_auc_score(yte, p_mix)),
                        "log_score": float(loss_mix.mean())},
            "centroids_raw": (C * sd + mu).tolist(),
            "clusters": cluster_table(labels, yte, loss_base, loss_mix, p_base, len(C)),
        }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "cluster_weakness.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"confirmation split, {len(yte)} rows")
    print(f"base model            log score {out['base']['log_score']:.4f}   "
          f"AUC {out['base']['auc']:.4f}\n")
    for tag, part in out["partitions"].items():
        print(f"{tag} partition   mixture log score {part['mixture']['log_score']:.4f}   "
              f"AUC {part['mixture']['auc']:.4f}")
        print(f"{'cl':>3}{'n':>7}{'share':>7}{'rate':>7}{'pred':>7}{'const':>8}"
              f"{'base':>8}{'excess':>8}{'mixture':>9}{'gain':>8}{'lift':>6}{'cover':>7}")
        for r in part["clusters"]:
            print(f"{r['cluster']:>3}{r['n']:>7}{r['share']:>7.3f}{r['default_rate']:>7.3f}"
                  f"{r['mean_predicted']:>7.3f}{r['constant_log_score']:>8.4f}"
                  f"{r['base_log_score']:>8.4f}{r['excess_over_constant']:>8.4f}"
                  f"{r['mixture_log_score']:>9.4f}{r['improvement']:>8.4f}"
                  f"{r['lift']:>6.2f}{r['coverage']:>7.3f}")
        print("    centroids (raw units):")
        for h, c in enumerate(part["centroids_raw"]):
            print("      " + f"{h}: " + "  ".join(f"{f}={v:.3g}" for f, v in zip(feats, c)))
        print()
    print("wrote", path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/moegp")
    ap.add_argument("--prior", default="runs/moegp/prior_moe.json")
    run(**vars(ap.parse_args()))
