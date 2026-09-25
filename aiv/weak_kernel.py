"""Error-model weak-region discovery for the AI-for-Validation framework.

Recipe: fit a shallow XGBoost regressor whose target is the model-under-test's
absolute error, then (1) read that error model's gain importance for a global
ranking of the factors that drive error, and (2) spectral-cluster the error
model's leaf kernel to localize the weak region, characterized by a
Jensen-Shannon feature profile.

All kernel, spectral-clustering, cluster-error and JS-profile logic is reused
from the `kernel_xgb` library (kernel-xgb) -- this module only orchestrates it,
it does not reimplement any of it:
  - kernel_xgb.kernel.LeafKernel
  - kernel_xgb.weakness.spectral_cluster_nystrom
  - kernel_xgb.weakness.cluster_error_report_classification
  - kernel_xgb.weakness.js_weak_region_profile

`LeafKernelRouter` is this module's own addition. It turns the cluster labels,
which exist only for the analyzed rows, into a membership rule that applies to
held-out rows, so the located region can be confirmed rather than only described.
"""
from __future__ import annotations

import numpy as np
import xgboost

from aiv._vendor.kernel_xgb.kernel import LeafKernel
from aiv._vendor.kernel_xgb.weakness import (
    cluster_error_report_classification,
    js_weak_region_profile,
    spectral_cluster_nystrom,
)


class LeafKernelRouter:
    """Membership rule for the error-model weak region, applicable to unseen rows.

    Spectral clustering assigns labels only to the rows it was run on, so the
    partition alone cannot decide whether a held-out borrower falls in the weak
    region. This router supplies that decision: each cluster is summarized by its
    mean leaf-kernel vector, and a row is routed to the cluster whose mean it is
    most similar to under the same kernel. It is the leaf-space counterpart of
    `aiv.weak_region.CentroidRouter`, which routes by nearest centroid in
    standardized feature space.

    The rule reproduces the spectral partition approximately rather than exactly,
    because the Nystrom spectral embedding has no exact out-of-sample extension
    here. Agreement with the fitted labels is worth reporting before the region is
    frozen. A row landing in a leaf no tree saw during fitting contributes fewer
    nonzero entries, which lowers its similarity to every cluster mean alike.
    """

    def __init__(self, leaf_kernel, features, cluster_means, weak_cluster: int):
        self.leaf_kernel = leaf_kernel
        self.features = list(features)
        self.cluster_means = np.asarray(cluster_means, dtype=float)
        self.weak_cluster = int(weak_cluster)

    @classmethod
    def from_labels(cls, leaf_kernel, features, Z, labels, weak_cluster: int):
        """Mean leaf-kernel vector per cluster; an empty cluster gets a zero mean
        and is therefore never the nearest."""
        k = int(np.max(labels)) + 1
        means = np.zeros((k, Z.shape[1]), dtype=float)
        for j in range(k):
            rows = labels == j
            if rows.any():
                means[j] = np.asarray(Z[rows].mean(axis=0)).ravel()
        return cls(leaf_kernel, features, means, weak_cluster)

    def assign(self, df) -> np.ndarray:
        """Cluster id for each row. The kernel's 1/S normalization is a positive
        constant and does not affect which cluster is nearest, so it is omitted."""
        Z = self.leaf_kernel.transform(df[self.features].values)
        return np.asarray(Z @ self.cluster_means.T).argmax(1)

    def weak_mask(self, df) -> np.ndarray:
        """Boolean membership of the weak region, the interface the confirmation
        procedures in `aiv.confirmation` expect."""
        return self.assign(df) == self.weak_cluster

    def label_agreement(self, df, labels) -> float:
        """Fraction of `labels` the rule reproduces on the rows it was fitted to."""
        return float((self.assign(df) == np.asarray(labels)).mean())


def error_model_weak_regions(
    system,
    split: str = "discovery",
    n_clusters: int = 5,
    depth: int = 2,
    seed: int = 0,
) -> dict:
    """Locate a model's weak region by modeling its error with a shallow XGBoost.

    Parameters
    ----------
    system : object exposing `.features`, `.splits[split]`, `.predict_proba(df)`
             and a binary target column named by `system.target` (default
             "default"), e.g. aiv.predictive.PredictiveSystem.
    split  : which split of `system.splits` to analyze.
    n_clusters, depth, seed : error-model / clustering hyperparameters.

    Returns
    -------
    dict with keys:
      labels          : (n,) cluster assignment from the leaf-kernel spectral clustering.
      weak_cluster    : id of the highest-logloss (weakest) cluster.
      cluster_report  : small per-cluster DataFrame (n, share, event_rate,
                        mean_pred, logloss, logloss_ratio) sorted worst-first.
      overall         : dict of overall logloss / brier / event_rate / n.
      importance      : list of (feature, gain) sorted desc -- global error factors.
      js_profile      : list of (feature, js_divergence, direction) sorted desc --
                        region-local factor characterization for the weak cluster.
      router          : LeafKernelRouter, the frozen membership rule for the weak
                        cluster; `router.weak_mask(df)` applies it to any rows,
                        including a split this function never saw.
    """
    target = getattr(system, "target", "default")
    feats = list(system.features)
    df = system.splits[split]
    X = df[feats]
    Xv = X.values
    y = df[target].to_numpy()
    p = system.predict_proba(df)[:, 1]  # P(default) from the model under test
    ae = np.abs(y - p)                  # absolute error = target of the error model

    # 1. Shallow error model: predict absolute error of the model under test.
    error_model = xgboost.XGBRegressor(
        max_depth=depth,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    error_model.fit(Xv, ae)

    # 2. Global factors: error-model gain importance ranked with feature names.
    gains = np.asarray(error_model.feature_importances_, dtype=float)
    importance = sorted(zip(feats, gains.tolist()), key=lambda kv: -kv[1])

    # 3. Spectral-cluster the error model's leaf kernel (reused from kernel_xgb).
    kern = LeafKernel(error_model).fit(Xv)
    Z = kern.transform(Xv)
    labels, _, _ = spectral_cluster_nystrom(
        Z,
        n_clusters=n_clusters,
        m_landmarks=min(500, len(X)),
        random_state=seed,
    )

    # 4. Per-cluster error report on the model under test; weak = highest logloss.
    report, overall = cluster_error_report_classification(labels, y, p)

    # 5. Region-local factor characterization via JS divergence.
    profile, weak_id = js_weak_region_profile(
        X, labels, report, feature_names=feats, task="classification"
    )

    # 6. A membership rule for the weak cluster, so the region can be frozen and
    #    evaluated on rows this function never saw.
    router = LeafKernelRouter.from_labels(kern, feats, Z, labels, weak_id)

    # Trim outputs to the essentials the caller asked for.
    cols = ["cluster", "n", "cluster_share", "event_rate", "mean_pred",
            "logloss", "logloss_ratio"]
    cluster_report = report[cols].copy()

    js_profile = [
        (
            r["feature"],
            float(r["js_divergence"]),
            f"{r['top_bin']} enriched "
            f"({r['top_bin_weak_prob']:.2f} vs {r['top_bin_rest_prob']:.2f}, "
            f"x{r['top_bin_enrichment']:.2f}); weak_mean={r['weak_mean']:.3g} "
            f"vs rest_mean={r['rest_mean']:.3g}",
        )
        for _, r in profile.iterrows()
    ]

    return {
        "labels": labels,
        "weak_cluster": int(weak_id),
        "cluster_report": cluster_report,
        "overall": overall,
        "importance": importance,
        "js_profile": js_profile,
        "router": router,
    }


if __name__ == "__main__":
    from aiv.predictive import PredictiveSystem

    sysm = PredictiveSystem().fit()
    out = error_model_weak_regions(sysm)

    rep = out["cluster_report"]
    weak = out["weak_cluster"]
    weak_row = rep[rep["cluster"] == weak].iloc[0]
    ov = out["overall"]

    print("=== error-model weak-region discovery ===")
    print(f"n_clusters=5  n={ov['n']}  overall logloss={ov['logloss']:.4f}  "
          f"event_rate={ov['event_rate']:.4f}")
    print(f"weak cluster id={weak}  size={int(weak_row['n'])}  "
          f"share={weak_row['cluster_share']:.3f}  "
          f"logloss={weak_row['logloss']:.4f}  "
          f"lift={weak_row['logloss_ratio']:.2f}x  "
          f"event_rate={weak_row['event_rate']:.4f}")
    print("\ncluster report (worst-first):")
    print(rep.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))
    print("\ntop-4 error-model importances (gain):")
    for f, g in out["importance"][:4]:
        print(f"  {f:<14s} {g:.4f}")
    print("\ntop-4 JS-profile features (weak region):")
    for f, js, direction in out["js_profile"][:4]:
        print(f"  {f:<14s} js={js:.4f}  {direction}")
