"""XGBoost Proximity Weakness Clustering.

Pipeline:
    1. Fit XGBoost on training data.
    2. Build LeafKernel and extract sparse leaf-membership matrix Z (n, d).
    3. Nystrom approximation of K = Z Z^T using m landmark rows:
           C = Z Z_L^T   (n x m)
           W = Z_L Z_L^T (m x m)
           Phi = C W^{-1/2}     so that Phi Phi^T ~ K
    4. Spectral embedding: thin SVD of Phi gives left singular vectors U.
       The top-k columns U_k are the leading eigenvectors of K. Row-normalize
       (Ng-Jordan-Weiss) and run KMeans for the cluster assignment.
    5. Compute residual/error metrics per cluster on a held-out set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from typing import Tuple, Dict, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Nystrom + spectral embedding
# ---------------------------------------------------------------------------

def nystrom_embedding(
    Z: sp.csr_matrix,
    m: int = 500,
    ridge: float = 1e-6,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Low-rank embedding Phi (n, m) such that Phi Phi^T approximates K = Z Z^T.

    Returns Phi (dense float64) and the landmark indices used.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    m_eff = min(m, n)
    landmark_idx = rng.choice(n, size=m_eff, replace=False)
    Z_land = Z[landmark_idx]

    C = (Z @ Z_land.T).toarray().astype(np.float64)
    W = (Z_land @ Z_land.T).toarray().astype(np.float64)
    W = W + ridge * np.eye(W.shape[0])

    vals, vecs = eigh(W)
    vals = np.maximum(vals, ridge)
    W_inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T

    Phi = C @ W_inv_sqrt
    return Phi, landmark_idx


def spectral_coords_from_nystrom(
    Phi: np.ndarray,
    n_components: int,
    row_normalize: bool = True,
) -> np.ndarray:
    """Top-k left singular vectors of Phi = top-k eigenvectors of K = Phi Phi^T.

    With row_normalize=True this is the Ng-Jordan-Weiss spectral embedding:
    each row projected onto the unit sphere of the leading-k spectral subspace.
    """
    U, _, _ = np.linalg.svd(Phi, full_matrices=False)
    U_k = U[:, :n_components]
    if row_normalize:
        norms = np.linalg.norm(U_k, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        U_k = U_k / norms
    return U_k


def spectral_cluster_nystrom(
    Z: sp.csr_matrix,
    n_clusters: int = 10,
    m_landmarks: int = 500,
    ridge: float = 1e-6,
    random_state: int = 42,
    row_normalize: bool = True,
    n_init: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full proximity-kernel spectral clustering.

    Returns
    -------
    labels : (n,) int
    Phi    : (n, m_landmarks) Nystrom embedding
    U_k    : (n, n_clusters) spectral coords used for KMeans
    """
    Phi, _ = nystrom_embedding(Z, m=m_landmarks, ridge=ridge, random_state=random_state)
    U_k = spectral_coords_from_nystrom(Phi, n_clusters, row_normalize=row_normalize)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(U_k)
    return labels, Phi, U_k


# ---------------------------------------------------------------------------
# Cluster error reports
# ---------------------------------------------------------------------------

def cluster_error_report_regression(
    cluster: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    abs_err = np.abs(residual)
    df = pd.DataFrame({
        "cluster": cluster,
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": abs_err,
        "residual": residual,
    })
    overall_mae = float(abs_err.mean())
    overall_rmse = float(np.sqrt(np.mean(residual ** 2)))
    rep = df.groupby("cluster").agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        bias=("residual", "mean"),
        rmse=("residual", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
    ).reset_index()
    rep["mae_ratio"] = rep["mae"] / overall_mae
    rep["rmse_ratio"] = rep["rmse"] / overall_rmse
    rep["cluster_share"] = rep["n"] / len(df)
    rep["weakness_score"] = rep["mae_ratio"] * np.sqrt(rep["cluster_share"])
    rep = rep.sort_values("mae_ratio", ascending=False).reset_index(drop=True)
    overall = {"mae": overall_mae, "rmse": overall_rmse, "n": int(len(df))}
    return rep, overall


def cluster_error_report_classification(
    cluster: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    brier = (y_true - y_prob) ** 2
    logloss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    df = pd.DataFrame({
        "cluster": cluster,
        "y_true": y_true,
        "y_prob": y_prob,
        "brier": brier,
        "logloss": logloss,
    })
    overall_logloss = float(logloss.mean())
    overall_brier = float(brier.mean())
    overall_event = float(y_true.mean())

    rep = df.groupby("cluster").agg(
        n=("y_true", "size"),
        event_rate=("y_true", "mean"),
        mean_pred=("y_prob", "mean"),
        logloss=("logloss", "mean"),
        brier=("brier", "mean"),
    ).reset_index()

    aucs = []
    for c in rep["cluster"].values:
        sub = df[df["cluster"] == c]
        if sub["y_true"].nunique() < 2:
            aucs.append(np.nan)
        else:
            aucs.append(float(roc_auc_score(sub["y_true"], sub["y_prob"])))
    rep["auc"] = aucs
    rep["calibration_bias"] = rep["event_rate"] - rep["mean_pred"]
    rep["logloss_ratio"] = rep["logloss"] / overall_logloss
    rep["brier_ratio"] = rep["brier"] / overall_brier
    rep["cluster_share"] = rep["n"] / len(df)
    rep["weakness_score"] = rep["logloss_ratio"] * np.sqrt(rep["cluster_share"])
    rep = rep.sort_values("logloss_ratio", ascending=False).reset_index(drop=True)
    overall = {
        "logloss": overall_logloss,
        "brier": overall_brier,
        "event_rate": overall_event,
        "n": int(len(df)),
    }
    return rep, overall


# ---------------------------------------------------------------------------
# Spread / contrast diagnostics
# ---------------------------------------------------------------------------

def spread_diagnostics_regression(rep: pd.DataFrame) -> Dict[str, float]:
    """How well-separated are clusters by error?"""
    return {
        "mae_max": float(rep["mae"].max()),
        "mae_min": float(rep["mae"].min()),
        "mae_max_over_min": float(rep["mae"].max() / max(rep["mae"].min(), 1e-12)),
        "rmse_max_over_min": float(rep["rmse"].max() / max(rep["rmse"].min(), 1e-12)),
        "weakness_score_max": float(rep["weakness_score"].max()),
    }


# ---------------------------------------------------------------------------
# Residual-correction lift by cluster
# ---------------------------------------------------------------------------

def cluster_lift_regression(
    cluster: np.ndarray,
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_corrected: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_base = np.asarray(y_pred_base, dtype=np.float64)
    y_pred_corr = np.asarray(y_pred_corrected, dtype=np.float64)
    base_abs = np.abs(y_true - y_pred_base)
    corr_abs = np.abs(y_true - y_pred_corr)
    base_sq = (y_true - y_pred_base) ** 2
    corr_sq = (y_true - y_pred_corr) ** 2
    df = pd.DataFrame({
        "cluster": cluster,
        "base_abs": base_abs, "corr_abs": corr_abs,
        "base_sq": base_sq, "corr_sq": corr_sq,
    })
    rep = df.groupby("cluster").agg(
        n=("base_abs", "size"),
        mae_base=("base_abs", "mean"),
        mae_corr=("corr_abs", "mean"),
        rmse_base=("base_sq", lambda x: float(np.sqrt(np.mean(x)))),
        rmse_corr=("corr_sq", lambda x: float(np.sqrt(np.mean(x)))),
    ).reset_index()
    rep["mae_lift"] = rep["mae_base"] - rep["mae_corr"]
    rep["rmse_lift"] = rep["rmse_base"] - rep["rmse_corr"]
    rep["rel_mae_lift_pct"] = 100.0 * rep["mae_lift"] / rep["mae_base"].clip(lower=1e-12)
    rep = rep.sort_values("mae_lift", ascending=False).reset_index(drop=True)
    overall = {
        "mae_base": float(base_abs.mean()),
        "mae_corr": float(corr_abs.mean()),
        "rmse_base": float(np.sqrt(base_sq.mean())),
        "rmse_corr": float(np.sqrt(corr_sq.mean())),
    }
    overall["mae_lift"] = overall["mae_base"] - overall["mae_corr"]
    overall["rmse_lift"] = overall["rmse_base"] - overall["rmse_corr"]
    return rep, overall


def cluster_lift_classification(
    cluster: np.ndarray,
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_corrected: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    p_base = np.clip(np.asarray(p_base, dtype=np.float64), eps, 1 - eps)
    p_corr = np.clip(np.asarray(p_corrected, dtype=np.float64), eps, 1 - eps)
    ll_base = -(y_true * np.log(p_base) + (1 - y_true) * np.log(1 - p_base))
    ll_corr = -(y_true * np.log(p_corr) + (1 - y_true) * np.log(1 - p_corr))
    br_base = (y_true - p_base) ** 2
    br_corr = (y_true - p_corr) ** 2
    df = pd.DataFrame({
        "cluster": cluster,
        "ll_base": ll_base, "ll_corr": ll_corr,
        "br_base": br_base, "br_corr": br_corr,
        "y_true": y_true, "p_base": p_base, "p_corr": p_corr,
    })
    rep = df.groupby("cluster").agg(
        n=("ll_base", "size"),
        event_rate=("y_true", "mean"),
        mean_p_base=("p_base", "mean"),
        mean_p_corr=("p_corr", "mean"),
        logloss_base=("ll_base", "mean"),
        logloss_corr=("ll_corr", "mean"),
        brier_base=("br_base", "mean"),
        brier_corr=("br_corr", "mean"),
    ).reset_index()
    rep["logloss_lift"] = rep["logloss_base"] - rep["logloss_corr"]
    rep["brier_lift"] = rep["brier_base"] - rep["brier_corr"]
    rep["rel_logloss_lift_pct"] = (
        100.0 * rep["logloss_lift"] / rep["logloss_base"].clip(lower=1e-12)
    )
    rep = rep.sort_values("logloss_lift", ascending=False).reset_index(drop=True)
    overall = {
        "logloss_base": float(ll_base.mean()),
        "logloss_corr": float(ll_corr.mean()),
        "brier_base": float(br_base.mean()),
        "brier_corr": float(br_corr.mean()),
    }
    overall["logloss_lift"] = overall["logloss_base"] - overall["logloss_corr"]
    overall["brier_lift"] = overall["brier_base"] - overall["brier_corr"]
    return rep, overall


# ---------------------------------------------------------------------------
# Weak-region variable profile (Jensen-Shannon divergence)
# ---------------------------------------------------------------------------

def _pick_weakest_cluster(rep: pd.DataFrame, task: str) -> int:
    """Pick the worst cluster from a regression or classification report."""
    if task == "regression":
        sort_col = "mae_ratio" if "mae_ratio" in rep.columns else "mae"
    else:
        sort_col = "logloss_ratio" if "logloss_ratio" in rep.columns else "logloss"
    return int(rep.sort_values(sort_col, ascending=False).iloc[0]["cluster"])


def _bin_series(x: pd.Series, n_bins: int) -> pd.Series:
    """Quantile-bin a numeric series. Falls back to raw categories if too few uniques."""
    if pd.api.types.is_numeric_dtype(x):
        # If few unique values (e.g., one-hot or low-cardinality int), use values directly.
        nunique = int(x.nunique(dropna=True))
        if nunique <= max(2, n_bins // 2):
            return x.fillna("__nan__").astype(str)
        try:
            cuts = pd.qcut(x, q=n_bins, duplicates="drop")
            # qcut can return NaN-only when the column degenerates; fall back to raw values.
            if cuts.dropna().empty:
                return x.fillna("__nan__").astype(str)
            return cuts.astype(object).fillna("__nan__")
        except ValueError:
            return x.fillna("__nan__").astype(str)
    return x.fillna("__nan__").astype(str)


def js_weak_region_profile(
    X: Union[np.ndarray, pd.DataFrame],
    clusters: np.ndarray,
    cluster_report: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    weakest_cluster: Optional[int] = None,
    task: str = "regression",
    n_bins: int = 10,
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, int]:
    """Rank variables by JS divergence between the weak cluster and the rest.

    Returns (profile_df sorted by js_divergence desc, weakest_cluster_id).
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        if feature_names is None:
            feature_names = [f"x{j}" for j in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=list(feature_names))

    if weakest_cluster is None:
        weakest_cluster = _pick_weakest_cluster(cluster_report, task)

    weak_mask = (clusters == weakest_cluster)
    rest_mask = ~weak_mask
    weak_share = float(weak_mask.mean())

    rows = []
    for col in X_df.columns:
        x = X_df[col]
        bins = _bin_series(x, n_bins=n_bins)

        weak_counts = bins[weak_mask].value_counts()
        rest_counts = bins[rest_mask].value_counts()
        all_bins = weak_counts.index.union(rest_counts.index)

        if len(all_bins) == 0:
            # Pathological column (all-NaN bins or fully constant after binning).
            js_dist = 0.0
            js_div = 0.0
            top_bin = "(none)"
            top_bin_enrich = 1.0
            top_bin_weak_prob = 0.0
            top_bin_rest_prob = 0.0
        else:
            p = weak_counts.reindex(all_bins, fill_value=0).astype(float).values + eps
            q = rest_counts.reindex(all_bins, fill_value=0).astype(float).values + eps
            p = p / p.sum()
            q = q / q.sum()
            js_dist = float(jensenshannon(p, q, base=2))
            js_div = js_dist ** 2
            weak_prob = pd.Series(p, index=[str(b) for b in all_bins])
            rest_prob = pd.Series(q, index=[str(b) for b in all_bins])
            enrich = (weak_prob + eps) / (rest_prob + eps)
            top_bin = enrich.idxmax()
            top_bin_enrich = float(enrich.max())
            top_bin_weak_prob = float(weak_prob.loc[top_bin])
            top_bin_rest_prob = float(rest_prob.loc[top_bin])

        if pd.api.types.is_numeric_dtype(x):
            weak_mean = float(x[weak_mask].mean())
            rest_mean = float(x[rest_mask].mean())
            weak_median = float(x[weak_mask].median())
            rest_median = float(x[rest_mask].median())
            weak_std = float(x[weak_mask].std(ddof=0))
            rest_std = float(x[rest_mask].std(ddof=0))
            # standardized mean shift (Cohen's d, pooled sd)
            pooled = np.sqrt(0.5 * (weak_std ** 2 + rest_std ** 2)) + 1e-12
            cohen_d = (weak_mean - rest_mean) / pooled
        else:
            weak_mean = rest_mean = weak_median = rest_median = np.nan
            cohen_d = np.nan

        rows.append({
            "feature": col,
            "js_divergence": js_div,
            "js_distance": js_dist,
            "mean_diff": (weak_mean - rest_mean) if np.isfinite(weak_mean) else np.nan,
            "median_diff": (weak_median - rest_median) if np.isfinite(weak_median) else np.nan,
            "cohen_d": cohen_d,
            "weak_mean": weak_mean,
            "rest_mean": rest_mean,
            "weak_median": weak_median,
            "rest_median": rest_median,
            "top_bin": top_bin,
            "top_bin_enrichment": top_bin_enrich,
            "top_bin_weak_prob": top_bin_weak_prob,
            "top_bin_rest_prob": top_bin_rest_prob,
        })

    profile = (
        pd.DataFrame(rows)
        .sort_values("js_divergence", ascending=False)
        .reset_index(drop=True)
    )
    profile.attrs["weakest_cluster"] = int(weakest_cluster)
    profile.attrs["weak_share"] = weak_share
    return profile, int(weakest_cluster)


# ---------------------------------------------------------------------------
# Spread / contrast diagnostics
# ---------------------------------------------------------------------------

def spread_diagnostics_classification(rep: pd.DataFrame) -> Dict[str, float]:
    return {
        "logloss_max": float(rep["logloss"].max()),
        "logloss_min": float(rep["logloss"].min()),
        "logloss_max_over_min": float(
            rep["logloss"].max() / max(rep["logloss"].min(), 1e-12)
        ),
        "brier_max_over_min": float(
            rep["brier"].max() / max(rep["brier"].min(), 1e-12)
        ),
        "calibration_bias_abs_max": float(rep["calibration_bias"].abs().max()),
        "weakness_score_max": float(rep["weakness_score"].max()),
    }
