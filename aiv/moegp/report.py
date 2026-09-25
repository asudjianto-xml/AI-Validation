"""Assemble the recorded runs into one comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _row(name, auc, ll, mono=None, extra=""):
    m = "     -" if mono is None else f"{mono:6.3f}"
    return f"{name:<44}{auc:8.4f}{ll:10.4f}{m}   {extra}"


def report(run_dir: str = "runs/moegp") -> str:
    d = Path(run_dir)
    lines = [f"{'':<44}{'AUC':>8}{'logloss':>10}{'mono':>7}   notes",
             "-" * 96]

    prior_path = d / "prior_moe.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None

    h2 = json.loads((d / "moegp_H2.json").read_text())
    b = h2["baselines"]
    lines.append("single models (confirmation split)")
    for key, label in [("monotone_xgb_discovery", "monotone XGB, fitted on discovery"),
                       ("monotone_xgb_train", "monotone XGB, fitted on train"),
                       ("monotone_xgb_train_plus_discovery", "monotone XGB, train + discovery"),
                       ("unconstrained_xgb_discovery", "unconstrained XGB, fitted on discovery")]:
        mono = b[key].get("monotonicity", {}).get("any_feature_mean")
        lines.append(_row("  " + label, b[key]["auc"], b[key]["logloss"], mono))

    if prior:
        lines.append("")
        lines.append(f"mixture of monotone XGBoost experts (prior method, {prior['n_evals']} evaluations)")
        for key, label in [("single_monotone_xgb", "single monotone XGB"),
                           ("kmeans_centroids", "k-means centroids"),
                           ("searched_centroids", "searched centroids")]:
            c = prior["confirm"][key]
            lines.append(_row("  " + label, c["auc"], c["logloss"],
                              c["monotonicity"]["any_feature_mean"]))
        lines.append(f"  inner-split AUC  base {prior['inner']['base']:.4f}"
                     f"   k-means {prior['inner']['kmeans']:.4f}"
                     f"   searched {prior['inner']['searched']:.4f}")

    for path in sorted(d.glob("moegp_H*.json")):
        r = json.loads(path.read_text())
        H = r["n_components"]
        lines.append("")
        lines.append(f"mixture of leaf-kernel processes, H = {H}"
                     f"  (primal dim {r['kernel']['primal_dim']}, "
                     f"{r['settings']['n_folds']}-fold objective)")
        s = r["mixture"]["single_process_no_gate"]
        lines.append(_row("  one process, no gate", s["confirm"]["auc"], s["confirm"]["logloss"],
                          s["confirm"]["monotonicity"]["any_feature_mean"],
                          f"inner {s['inner_auc']:.4f}"))
        recs = r["mixture"]["kmeans_gate"]
        for rec in recs:
            tag = " (headline)" if rec["seed"] == r["mixture"]["headline_seed"] else ""
            ci, cs = rec["confirm_init"], rec["confirm_searched"]
            lines.append(_row(f"  seed {rec['seed']}: k-means gate", ci["auc"], ci["logloss"],
                              ci["monotonicity"]["any_feature_mean"],
                              f"inner {rec['inner_auc_init']:.4f}"))
            lines.append(_row(f"  seed {rec['seed']}: searched gate{tag}", cs["auc"], cs["logloss"],
                              cs["monotonicity"]["any_feature_mean"],
                              f"inner {rec['inner_auc_searched']:.4f}, {rec['n_evals']} evals"))
        aucs = [rec["confirm_searched"]["auc"] for rec in recs]
        lines.append(f"  searched across seeds: min {min(aucs):.4f}  median "
                     f"{float(np.median(aucs)):.4f}  max {max(aucs):.4f}")

    lines.append("")
    lines.append("mono = share of confirmation rows whose predicted probability moves against a")
    lines.append("       constraint under a +0.25 sd shift, averaged over the eight constrained features")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/moegp")
    print(report(ap.parse_args().run_dir))
