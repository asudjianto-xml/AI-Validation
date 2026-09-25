"""Predictive adapter for the AI-for-Validation framework.

The credit model is an inherently interpretable functional-ANOVA model: additive
monotone main effects plus a small set of two-way interactions, built in three
steps (`FanovaModel`).

  1. Fit a depth-2 model to identify candidate two-way interactions (pairs of
     features that split together in a shallow tree, ranked by gain).
  2. Fit a main-effect model: depth-4 trees constrained so each tree splits on a
     single variable, giving additive main effects f_j(x_j). Monotonicity
     constraints fix each effect's direction as credit policy requires.
  3. Boost the residual of the main-effect model with two-variable splits over
     only the interactions found in step 1, giving f_jk(x_j, x_k).

`SegmentedSystem` closes the validation loop: once a weak region is found and
confirmed, the data is segmented on that region and a distinct functional-ANOVA
model is built for each segment.

A candidate is a validation test: a data slice plus an optional stress. Executing
a candidate on a condition (a bootstrap resample of an evaluation split) returns a
weakness score, oriented so that a larger value means a more exposed weakness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from aiv.datasets import default_data_path

TARGET = "default"

MONOTONE = {
    "income": -1,
    "dti": +1,
    "score": -1,
    "amount": +1,
    "emp_length": -1,
    "delinquencies": +1,
    "savings": -1,
    "utilization": +1,
    "employment": 0,
    "tenure": 0,
}

N_INTERACTIONS = 5  # number of two-way interactions kept in the fANOVA model


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _three_way_split(csv_path: str | None, target: str, seed: int) -> dict:
    """Train / discovery / confirm. Fixed by seed so systems are comparable.

    A None csv_path falls back to the bundled credit-default dataset."""
    df = pd.read_csv(csv_path or default_data_path())
    train, rest = train_test_split(df, test_size=0.5, random_state=seed, stratify=df[target])
    discovery, confirm = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest[target])
    return {"train": train, "discovery": discovery, "confirm": confirm}


@dataclass
class FanovaModel:
    """A functional-ANOVA model: additive monotone main effects plus two-way
    interactions, built by the three-step procedure."""

    features: list
    seed: int = 0
    interactions: list = field(init=False, default=None)
    bst_main: object = field(init=False, default=None)
    bst_int: object = field(init=False, default=None)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FanovaModel":
        mono = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in self.features) + ")"
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.features)

        # Step 1: identify two-way interactions with a depth-2 model.
        det = xgb.train(
            dict(max_depth=2, eta=0.1, subsample=0.9, colsample_bytree=0.9,
                 objective="binary:logistic", eval_metric="logloss",
                 monotone_constraints=mono, seed=self.seed),
            dtrain, num_boost_round=200,
        )
        self.interactions = self._rank_interactions(det)
        pairs = [[self.features.index(a), self.features.index(b)] for a, b, _ in self.interactions]

        # Step 2: additive monotone main effects (each tree splits on one variable).
        additive = json.dumps([[j] for j in range(len(self.features))]).replace(" ", "")
        self.bst_main = xgb.train(
            dict(max_depth=4, eta=0.05, subsample=0.9, colsample_bytree=1.0,
                 objective="binary:logistic", eval_metric="logloss",
                 monotone_constraints=mono, interaction_constraints=additive, seed=self.seed),
            dtrain, num_boost_round=300,
        )
        margin_main = self.bst_main.predict(dtrain, output_margin=True)

        # Step 3: boost the residual with only the identified two-way interactions.
        inter = json.dumps(pairs).replace(" ", "") if pairs else "[]"
        dtrain_int = xgb.DMatrix(X, label=y, feature_names=self.features, base_margin=margin_main)
        self.bst_int = xgb.train(
            dict(max_depth=2, eta=0.05, subsample=0.9, colsample_bytree=1.0,
                 objective="binary:logistic", eval_metric="logloss",
                 monotone_constraints=mono, interaction_constraints=inter, seed=self.seed),
            dtrain_int, num_boost_round=200,
        )
        return self

    def _rank_interactions(self, booster) -> list:
        det = booster.trees_to_dataframe()
        det = det[det.Feature != "Leaf"]
        pair_gain: dict = {}
        for _, g in det.groupby("Tree"):
            feats = list(dict.fromkeys(g.Feature.tolist()))
            if len(feats) < 2:
                continue
            gain = float(g.Gain.sum())
            combos = list(combinations(sorted(feats), 2))
            for a, b in combos:
                pair_gain[(a, b)] = pair_gain.get((a, b), 0.0) + gain / len(combos)
        ranked = sorted(pair_gain.items(), key=lambda kv: -kv[1])
        return [(a, b, gain) for (a, b), gain in ranked[:N_INTERACTIONS]]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(df[self.features], feature_names=self.features)
        margin_main = self.bst_main.predict(d, output_margin=True)
        d_int = xgb.DMatrix(df[self.features], feature_names=self.features, base_margin=margin_main)
        margin = self.bst_int.predict(d_int, output_margin=True)
        p = _sigmoid(margin)
        return np.column_stack([1.0 - p, p])


def _row_loss(proba1: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(proba1, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


@dataclass
class PredictiveSystem:
    """A single functional-ANOVA credit model plus a train / discovery / confirm split.

    csv_path defaults to the bundled credit-default dataset when left as None."""

    csv_path: str | None = None
    target: str = TARGET
    seed: int = 0
    features: list = field(init=False, default=None)
    splits: dict = field(init=False, default=None)
    model: FanovaModel = field(init=False, default=None)

    def fit(self) -> "PredictiveSystem":
        self.splits = _three_way_split(self.csv_path, self.target, self.seed)
        train = self.splits["train"]
        self.features = [c for c in train.columns if c != self.target]
        self.model = FanovaModel(self.features, self.seed).fit(train[self.features], train[self.target])
        return self

    @property
    def interactions(self) -> list:
        return self.model.interactions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df)

    def row_loss(self, df: pd.DataFrame) -> np.ndarray:
        return _row_loss(self.predict_proba(df)[:, 1], df[self.target].to_numpy())


@dataclass
class SegmentedSystem:
    """Distinct functional-ANOVA models for a weak segment and the rest of the
    data. The segment is the confirmed weak region, given as a slice expression.
    Predictions route by segment membership. csv_path defaults to the bundled
    credit-default dataset when left as None."""

    csv_path: str | None = None
    segment_slice: str | None = None
    member_fn: object = None  # callable(df) -> bool mask; overrides segment_slice
    target: str = TARGET
    seed: int = 0
    features: list = field(init=False, default=None)
    splits: dict = field(init=False, default=None)
    seg_model: FanovaModel = field(init=False, default=None)
    rest_model: FanovaModel = field(init=False, default=None)

    def _mask(self, df) -> np.ndarray:
        if self.member_fn is not None:
            return np.asarray(self.member_fn(df))
        return slice_mask(df, self.segment_slice)

    def fit(self) -> "SegmentedSystem":
        self.splits = _three_way_split(self.csv_path, self.target, self.seed)
        train = self.splits["train"]
        self.features = [c for c in train.columns if c != self.target]
        mask = self._mask(train)
        self.seg_model = FanovaModel(self.features, self.seed).fit(
            train[mask][self.features], train[mask][self.target]
        )
        self.rest_model = FanovaModel(self.features, self.seed).fit(
            train[~mask][self.features], train[~mask][self.target]
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        mask = self._mask(df)
        out = np.zeros((len(df), 2))
        if mask.any():
            out[mask] = self.seg_model.predict_proba(df[mask])
        if (~mask).any():
            out[~mask] = self.rest_model.predict_proba(df[~mask])
        return out

    def row_loss(self, df: pd.DataFrame) -> np.ndarray:
        return _row_loss(self.predict_proba(df)[:, 1], df[self.target].to_numpy())


def slice_mask(df: pd.DataFrame, expr: str) -> np.ndarray:
    if not expr or expr.strip().lower() in ("all", "none", ""):
        return np.ones(len(df), dtype=bool)
    return df.eval(expr).to_numpy()


def apply_stress(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """Apply a simple feature shift, for example "utilization+0.15"."""
    if not expr or expr.strip().lower() in ("none", ""):
        return df
    m = re.match(r"^\s*(\w+)\s*([+\-*])\s*([\d.]+)\s*$", expr)
    if not m:
        raise ValueError(f"unrecognized stress expression: {expr!r}")
    feat, op, val = m.group(1), m.group(2), float(m.group(3))
    out = df.copy()
    if op == "+":
        out[feat] = out[feat] + val
    elif op == "-":
        out[feat] = out[feat] - val
    else:
        out[feat] = out[feat] * val
    return out


class PredictiveAdapter:
    """Evaluate candidates on conditions, conforming to the Frontier-Discovery
    BatchEvaluateFn. A condition is a bootstrap-resample seed of one split; the
    score is weakness, higher meaning weaker. Works with any system that exposes
    `splits` and `row_loss`."""

    def __init__(self, system, split: str = "discovery", min_slice: int = 20):
        self.system = system
        self.split = split
        self.min_slice = min_slice

    def score_condition(self, candidate: dict, seed: int) -> tuple[float, int]:
        df = self.system.splits[self.split]
        resample = df.sample(frac=1.0, replace=True, random_state=seed)
        base_loss = self.system.row_loss(resample).mean()
        sub = resample[slice_mask(resample, candidate.get("slice", ""))]
        if len(sub) < self.min_slice:
            return 0.0, len(sub)
        sub = apply_stress(sub, candidate.get("stress", ""))
        slice_loss = self.system.row_loss(sub).mean()
        return float(slice_loss / base_loss - 1.0), int(len(sub))

    def weakness_series(self, candidate: dict, seeds) -> list[float]:
        return [self.score_condition(candidate, int(s))[0] for s in seeds]

    def evaluate(self, items):
        from aiv._vendor.frontier import EvaluationBatch

        results = []
        for candidate, conditions in items:
            scores, outputs = [], []
            for cond in conditions:
                w, n = self.score_condition(candidate, int(cond))
                scores.append(w)
                outputs.append({"condition": int(cond), "weakness": w, "slice_n": n})
            results.append(EvaluationBatch(outputs=outputs, scores=scores))
        return results
