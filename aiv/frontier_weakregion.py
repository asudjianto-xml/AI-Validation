"""Weak-region discovery wired through Frontier-Discovery.

Inner array: centroids (a borrower profile), a candidate is one centroid, seeded
by k-means. The outer array here holds bootstrap resamples of the discovery split,
so it measures sampling variability within one operating environment rather than
coverage across environments. From these evaluations the Frontier-Discovery engine
builds the selection record of scores, ownership and lineage; the evaluator
is the model's per-sample residual over the centroid's neighborhood, oriented so
higher means weaker. The search proposes new centroids by either an evolution
strategy (Gaussian mutation) or LLM reflection over the frontier. Same engine,
swapped proposer.
"""

from __future__ import annotations

import json
import re

import numpy as np
from sklearn.cluster import KMeans

from aiv._vendor.frontier import EvaluationBatch

# feature ranges for sanitizing and prompting (original units)
RANGES = {
    "employment": (0, 1), "income": (10000, 200000), "dti": (0.0, 0.7),
    "score": (560, 820), "amount": (500, 50000), "tenure": (6, 84),
    "emp_length": (0.0, 20.0), "delinquencies": (0, 8), "savings": (0.0, 60000.0),
    "utilization": (0.0, 1.2),
}


def _profile(features, vec_orig) -> dict:
    return {f: round(float(v), 4) for f, v in zip(features, vec_orig)}


class WeakRegionAdapter:
    """A candidate centroid is scored by the mean residual of its neighborhood."""

    propose_new_texts = None

    def __init__(self, system, split: str = "discovery", m: int = 150, features: list | None = None):
        df = system.splits[split].reset_index(drop=True)
        self.features = list(features) if features is not None else system.features
        self.mu = df[self.features].mean().to_numpy()
        self.sd = df[self.features].std().replace(0, 1.0).to_numpy()
        self.Xs = (df[self.features].to_numpy() - self.mu) / self.sd
        self.resid = system.row_loss(df)
        self.base = float(self.resid.mean())
        self.m = m

    def _centroid_std(self, candidate) -> np.ndarray:
        prof = json.loads(candidate["centroid"])
        if not isinstance(prof, dict) or set(prof) != set(self.features):
            raise ValueError("centroid must contain exactly the declared feature keys")
        v = np.array([float(prof[f]) for f in self.features])
        if not np.all(np.isfinite(v)):
            raise ValueError("centroid values must be finite")
        for f, value in zip(self.features, v):
            lo, hi = RANGES.get(f, (-np.inf, np.inf))
            if not lo <= value <= hi:
                raise ValueError(f"centroid value outside admissible range: {f}")
        return (v - self.mu) / self.sd

    def _weakness(self, c_std: np.ndarray, idx: np.ndarray) -> float:
        Xs, r = self.Xs[idx], self.resid[idx]
        d = np.sqrt(((Xs - c_std) ** 2).sum(1))
        mm = min(self.m, len(idx))
        near = np.argpartition(d, mm - 1)[:mm]
        return float(r[near].mean() / self.base - 1.0)

    def evaluate(self, batch, candidate, capture_traces=False):
        c = self._centroid_std(candidate)
        scores = [self._weakness(c, cond["idx"]) for cond in batch]
        trajs = ([{"cond": cond["name"], "weakness": s} for cond, s in zip(batch, scores)]
                 if capture_traces else None)
        return EvaluationBatch(outputs=[{"weakness": s} for s in scores], scores=scores,
                               trajectories=trajs, objective_scores=None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        comp = components_to_update[0]
        trajs = eval_batch.trajectories or []
        ws = [t["weakness"] for t in trajs]
        summ = (f"mean weakness {np.mean(ws):+.3f} (higher=weaker) over {len(ws)} bootstraps"
                if ws else "")
        recs = [{"Inputs": t["cond"], "Generated Outputs": f"weakness={t['weakness']:+.3f}",
                 "Feedback": summ + " Move the profile toward a weaker region."} for t in trajs]
        return {comp: recs}


def make_conditions(adapter: WeakRegionAdapter, n_boot: int = 8, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    n = len(adapter.Xs)
    return [{"name": f"boot{b}", "idx": rng.choice(n, size=n, replace=True)} for b in range(n_boot)]


def kmeans_seeds(adapter: WeakRegionAdapter, k: int = 8, seed: int = 0) -> list:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(adapter.Xs)
    centers = km.cluster_centers_ * adapter.sd + adapter.mu
    return [{"centroid": json.dumps(_profile(adapter.features, c))} for c in centers]


def es_proposer(adapter: WeakRegionAdapter, sigma: float = 0.5, seed: int = 0):
    """Gaussian mutation of the centroid in standard-deviation units."""
    rng = np.random.default_rng(seed)

    def _propose(candidate, reflective_dataset, components_to_update):
        prof = json.loads(candidate["centroid"])
        new = {}
        for i, f in enumerate(adapter.features):
            val = float(prof.get(f, adapter.mu[i])) + rng.normal(0.0, sigma) * adapter.sd[i]
            lo, hi = RANGES.get(f, (-np.inf, np.inf))
            new[f] = round(min(max(val, lo), hi), 4)
        return {"centroid": json.dumps(new)}

    return _propose


REFLECTION_TEMPLATE = (
    "You are searching for the region of borrowers where a credit default model is WEAKEST, "
    "meaning highest prediction loss. A region is a centroid: a borrower profile of feature "
    "values. Higher weakness is better here, because the goal is to locate the weak region.\n\n"
    "Current centroid profile (JSON, original feature units):\n```\n<curr_param>\n```\n\n"
    "Weakness of this centroid across bootstraps (higher means weaker):\n```\n<side_info>\n```\n\n"
    "Feature ranges: employment {0,1}, income [10k,200k], dti [0,0.7], score [560,820], "
    "amount [500,50k], tenure [6,84], emp_length [0,20], delinquencies [0,8], savings [0,60k], "
    "utilization [0,1.2]. Known model interactions: score x utilization, dti x score, "
    "dti x utilization.\n"
    "Propose a new centroid profile likely to be WEAKER, reasoning briefly about which feature "
    "combination stresses the model. Output ONLY a JSON object with all ten feature keys inside a "
    "single ``` code block."
)


def sanitize_centroid(text: str, features: list) -> dict:
    m = re.search(r"\{.*\}", str(text), re.S)
    raw = {}
    if m:
        try:
            raw = json.loads(m.group())
        except Exception:
            raw = {}
    out = {}
    for f in features:
        lo, hi = RANGES.get(f, (-np.inf, np.inf))
        try:
            v = float(raw.get(f))
            if not np.isfinite(v):
                raise ValueError("non-finite centroid value")
        except Exception:
            v = (lo + hi) / 2 if np.isfinite(lo) else 0.0
        out[f] = round(min(max(v, lo), hi), 4)
    return out
