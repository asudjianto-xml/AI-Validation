"""Calibrated k-means membership over monotonic depth-2 XGBoost experts.

K-means produces labels, not probabilities. A depth-1 XGBoost classifier
learns those labels; sigmoid calibration is fitted to cross-validated scores
with CalibratedClassifierCV(ensemble=False). The membership classifier is then
refitted on all fitting rows. Its calibrated probabilities weight both expert
training and probability averaging at inference. Training weights come from
this final fitted gate, not from a different out-of-fold routing function.

Calibration concerns cluster membership, not the credit outcome. Monotonic
experts do not imply a globally monotonic mixture with an input-dependent gate.
Centroid search and test-set evaluation are deliberately outside this module.
"""
from __future__ import annotations

import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBClassifier


class UnsupportedPartitionError(ValueError):
    """A centroid proposal cannot support every calibrated expert."""


class _FixedCentroids:
    def __init__(self, centers):
        self.cluster_centers_ = np.array(centers, dtype=float, copy=True)

    def predict(self, X):
        return ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(2).argmin(1)


class _Float64GateClassifier(XGBClassifier):
    """Promote XGBoost scores to match float64 calibration weights.

    sklearn 1.6 sigmoid calibration requires matching score/weight dtypes.
    This preserves observation-weight precision instead of downcasting weights.
    """

    def predict_proba(self, X, **kwargs):
        return np.asarray(super().predict_proba(X, **kwargs), dtype=np.float64)


class MonotoneKMeansMoEClassifier(ClassifierMixin, BaseEstimator):
    """A binary-outcome probability mixture of monotonic tree experts.

    Parameters
    ----------
    monotone_constraints : sequence of {-1, 0, 1}
        One direction per feature, in input column order. At least one must
        be nonzero. All experts receive exactly these constraints.
    expert_params : dict or None
        Optional XGBoost hyperparameters. Depth, objective, constraints,
        random state and thread count are controlled here and cannot be
        overridden. Defaults match the existing 200-tree credit comparator.
    calibration_cv : int, default=5
        Stratified folds for sigmoid calibration of cluster membership.
        Each cluster must contain at least this many positive-weight rows.
    gate_n_estimators : int, default=100
        Number of depth-1 trees in the membership classifier.
    random_state : int, default=0
        Fixed seed for k-means, calibration folds and boosters.
    n_jobs : int, default=1
        XGBoost threads. Calibration folds run sequentially.
    gate_features : sequence of integer column indices or None
        Columns used for scaling, hard clustering and calibrated membership.
        None uses every column. All experts always receive all input columns.
    n_clusters : int, default=2
        Number of k-means populations and corresponding experts, at least two.
        For more than two populations the membership classifier is multiclass;
        sigmoid calibration is applied one-vs-rest and normalized across classes.

    fit accepts optional strictly positive observation weights. Scaling,
    k-means, gate calibration and expert fitting all respect these weights.
    Remove zero-weight rows before fitting. All input features must be finite.
    """

    def __init__(self, monotone_constraints, expert_params=None,
                 calibration_cv=5, gate_n_estimators=100, random_state=0,
                 n_jobs=1, gate_features=None, n_clusters=2):
        self.monotone_constraints = monotone_constraints
        self.expert_params = expert_params
        self.calibration_cv = calibration_cv
        self.gate_n_estimators = gate_n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.gate_features = gate_features
        self.n_clusters = n_clusters

    def __setstate__(self, state):
        # Older saved estimators must remain cloneable for evidence reproduction.
        state = dict(state)
        state.setdefault("gate_features", None)
        state.setdefault("n_clusters", 2)
        super().__setstate__(state)

    def fit(self, X, y, sample_weight=None, *, centroids=None, scaler=None):
        """Fit ordinary k-means or evaluate supplied frozen geometry.

        Supply both centroids (n_clusters x number of gate features, standardized
        coordinates) and a fitted training-only StandardScaler for those
        columns in declared order. Neither is refitted; copies isolate
        this candidate from later mutations of the supplied objects.
        """
        # An unsuccessful refit must not leave an old fitted model callable.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        names = np.asarray(X.columns, dtype=object) if hasattr(X, "columns") else None
        X, y = check_X_y(X, y, dtype=np.float64)
        indices = np.arange(X.shape[1]) if self.gate_features is None else np.asarray(self.gate_features)
        if (indices.ndim != 1 or not len(indices)
                or indices.dtype.kind not in "iu"
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= X.shape[1])):
            raise ValueError("gate_features must be distinct valid integer column indices")
        indices = indices.astype(int, copy=True)
        Xgate = X[:, indices]
        if (not isinstance(self.n_clusters, (int, np.integer))
                or isinstance(self.n_clusters, (bool, np.bool_)) or self.n_clusters < 2):
            raise ValueError("n_clusters must be an integer >= 2")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("y must contain both binary labels 0 and 1")
        directions = np.asarray(self.monotone_constraints)
        if (directions.shape != (X.shape[1],)
                or not np.isin(directions, [-1, 0, 1]).all()
                or not np.any(directions != 0)):
            raise ValueError("monotone_constraints must give -1/0/1 per feature, with a nonzero direction")
        if not isinstance(self.calibration_cv, (int, np.integer)) or self.calibration_cv < 2:
            raise ValueError("calibration_cv must be an integer >= 2")
        if not isinstance(self.gate_n_estimators, (int, np.integer)) or self.gate_n_estimators < 1:
            raise ValueError("gate_n_estimators must be a positive integer")
        if not isinstance(self.random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer")
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if (weights.shape != (len(y),) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError("sample_weight must be finite, strictly positive and have one value per row")
        params = dict(n_estimators=200, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9)
        overrides = {} if self.expert_params is None else dict(self.expert_params)
        protected = {"max_depth", "objective", "monotone_constraints", "random_state",
                     "seed", "n_jobs", "nthread", "booster", "num_class", "device",
                     "max_leaves", "grow_policy", "updater", "process_type",
                     "scale_pos_weight", "tree_method"}
        if protected.intersection(overrides):
            raise ValueError(f"expert_params cannot override {sorted(protected.intersection(overrides))}")
        params.update(overrides)
        params.update(max_depth=2, objective="binary:logistic", eval_metric="logloss",
                      monotone_constraints=tuple(int(v) for v in directions),
                      random_state=self.random_state, n_jobs=self.n_jobs,
                      booster="gbtree", tree_method="hist", device="cpu")

        if (centroids is None) != (scaler is None):
            raise ValueError("Supply both centroids and a fitted scaler, or neither")
        if scaler is None:
            scaler = StandardScaler().fit(Xgate, sample_weight=weights)
        else:
            if not isinstance(scaler, StandardScaler):
                raise ValueError("scaler must be a fitted StandardScaler")
            check_is_fitted(scaler)
            if scaler.n_features_in_ != Xgate.shape[1]:
                raise ValueError("scaler feature count differs from fitting data")
            scaler = deepcopy(scaler)
        Xs = scaler.transform(Xgate)
        if centroids is None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
            kmeans.fit(Xs, sample_weight=weights)
        else:
            centers = np.asarray(centroids, float)
            if centers.shape != (self.n_clusters, Xgate.shape[1]) or not np.isfinite(centers).all():
                raise ValueError("centroids must be finite with shape (n_clusters, n_gate_features)")
            kmeans = _FixedCentroids(centers)
        labels = kmeans.predict(Xs)
        counts = np.bincount(labels, minlength=self.n_clusters)
        if np.any(counts < self.calibration_cv):
            raise UnsupportedPartitionError("All k-means clusters need at least calibration_cv rows; collapsed or undersupported gate")
        splits = list(StratifiedKFold(n_splits=self.calibration_cv, shuffle=True,
                                      random_state=self.random_state).split(Xs, labels))
        gate_base = _Float64GateClassifier(max_depth=1, n_estimators=self.gate_n_estimators,
                                  learning_rate=0.1, objective="binary:logistic" if self.n_clusters == 2 else "multi:softprob",
                                  eval_metric="logloss" if self.n_clusters == 2 else "mlogloss", tree_method="hist",
                                  random_state=self.random_state, n_jobs=self.n_jobs)
        gate = CalibratedClassifierCV(gate_base, method="sigmoid", cv=splits,
                                      ensemble=False, n_jobs=1)
        # Never silently replace this with an uncalibrated classifier.
        gate.fit(Xs, labels, sample_weight=weights)
        G = self._gate_matrix(gate, Xs)
        weighted = weights[:, None] * G
        mass = weighted.sum(axis=0)
        if np.any(mass <= 0):
            raise UnsupportedPartitionError("All experts must receive positive training weight")
        experts = []
        for h in range(self.n_clusters):
            expert = XGBClassifier(**params)
            expert.fit(X, y, sample_weight=weighted[:, h])
            experts.append(expert)

        self.n_features_in_ = X.shape[1]
        self.gate_feature_indices_ = indices
        if names is not None:
            self.feature_names_in_ = names.copy()
        self.classes_ = np.array([0, 1])
        self.scaler_ = scaler
        self.kmeans_ = kmeans
        self.gate_ = gate
        self.experts_ = experts
        self.cluster_labels_ = labels
        self.cluster_counts_ = counts
        self.calibration_splits_ = splits
        self.calibration_method_ = "sigmoid on out-of-fold membership scores; ensemble=False"
        self.training_gate_probabilities_ = G
        self.expert_weight_sums_ = mass
        self.expert_effective_sample_sizes_ = mass ** 2 / (weighted ** 2).sum(axis=0)
        self.expert_params_ = params
        return self

    @staticmethod
    def _gate_matrix(gate, Xs):
        n_classes = len(gate.classes_)
        if n_classes < 2 or not np.array_equal(gate.classes_, np.arange(n_classes)):
            raise ValueError("Gate must contain consecutive membership classes starting at 0")
        p = np.asarray(gate.predict_proba(Xs), float)
        if (p.shape != (len(Xs), n_classes) or not np.isfinite(p).all()
                or np.any((p < 0) | (p > 1))
                or not np.allclose(p.sum(axis=1), 1.0)):
            raise ValueError("Gate returned invalid membership probabilities")
        # One binary gate: a single probability and its exact complement.
        return np.column_stack([1.0 - p[:, 1], p[:, 1]]) if n_classes == 2 else p

    def _input(self, X):
        check_is_fitted(self, "experts_")
        if hasattr(X, "columns") and hasattr(self, "feature_names_in_"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError("Input feature names/order differ from fitting")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from fitting")
        return X

    def predict_gate_proba(self, X):
        """Normalized membership weights, shape (n_samples, n_clusters)."""
        X = self._input(X)
        return self._gate_matrix(self.gate_, self._gate_input(X))

    def _gate_input(self, X):
        # Historical serialized all-feature models predate this fitted field.
        indices = getattr(self, "gate_feature_indices_", np.arange(self.n_features_in_))
        return self.scaler_.transform(X[:, indices])

    def predict_cluster(self, X):
        """Hard nearest-centroid labels, distinct from surrogate gate argmax."""
        X = self._input(X)
        return self.kmeans_.predict(self._gate_input(X))

    def predict_expert_proba(self, X):
        """Positive-outcome probability per expert, shape (n_samples, n_clusters)."""
        X = self._input(X)
        return np.column_stack([e.predict_proba(X)[:, 1] for e in self.experts_])

    def predict_proba(self, X):
        X = self._input(X)
        G = self._gate_matrix(self.gate_, self._gate_input(X))
        P = np.column_stack([e.predict_proba(X)[:, 1] for e in self.experts_])
        p = np.sum(G * P, axis=1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[p.argmax(axis=1)]
