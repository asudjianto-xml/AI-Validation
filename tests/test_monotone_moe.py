"""Contract tests for calibrated k-means / monotonic-XGBoost mixtures."""
import json
import pickle
from functools import wraps

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

from aiv.monotone_moe import MonotoneKMeansMoEClassifier


def data():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(240, 3))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, 0] - X[:, 1]))))
    return X, y


def model():
    return MonotoneKMeansMoEClassifier(
        (1, -1, 0), expert_params={"n_estimators": 20},
        gate_n_estimators=15, calibration_cv=3)


@pytest.fixture(scope="module")
def fitted():
    X, y = data()
    return model().fit(X, y), X, y


def test_calibrated_gate_and_identical_training_routing(fitted):
    m, X, _ = fitted
    G = m.predict_gate_proba(X)
    assert G.shape == (len(X), 2)
    assert np.allclose(G.sum(1), 1)
    assert np.all((G > 0) & (G < 1))
    np.testing.assert_array_equal(G, m.training_gate_probabilities_)
    assert m.gate_.method == "sigmoid"
    assert m.gate_.ensemble is False
    assert len(m.gate_.calibrated_classifiers_) == 1
    seen = np.zeros(len(X), int)
    for train, held in m.calibration_splits_:
        assert not np.intersect1d(train, held).size
        assert set(m.cluster_labels_[train]) == {0, 1}
        seen[held] += 1
    np.testing.assert_array_equal(seen, np.ones(len(X)))


def test_actual_expert_training_weights_and_predictions(monkeypatch):
    X, y = data()
    w = np.linspace(0.5, 2.0, len(y))
    captured = []
    original = XGBClassifier.fit

    @wraps(original)
    def fit_spy(self, X, y, **kwargs):
        if self.max_depth == 2:
            captured.append(np.array(kwargs["sample_weight"], copy=True))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", fit_spy)
    m = model().fit(X, y, sample_weight=w)
    assert len(captured) == 2
    G = m.predict_gate_proba(X)
    for h in range(2):
        np.testing.assert_array_equal(captured[h], w * G[:, h])
        # Independently retrain with the contract weights, verifying propagation.
        independent = XGBClassifier(**m.expert_params_).fit(X, y, sample_weight=w * G[:, h])
        np.testing.assert_allclose(independent.predict_proba(X), m.experts_[h].predict_proba(X))
    np.testing.assert_allclose(m.expert_weight_sums_, (w[:, None] * G).sum(0))
    np.testing.assert_allclose(m.scaler_.mean_, np.average(X, axis=0, weights=w))


def test_probability_pool_not_margin_pool(fitted):
    m, X, _ = fitted
    G, P = m.predict_gate_proba(X), m.predict_expert_proba(X)
    actual = m.predict_proba(X)
    np.testing.assert_allclose(actual[:, 1], (G * P).sum(1), rtol=0, atol=1e-14)
    assert np.all(actual[:, 1] >= P.min(1) - 1e-12)
    assert np.all(actual[:, 1] <= P.max(1) + 1e-12)
    margins = np.log(P / (1 - P))
    margin_pool = 1 / (1 + np.exp(-(G * margins).sum(1)))
    assert np.max(np.abs(actual[:, 1] - margin_pool)) > 1e-6
    np.testing.assert_allclose(actual.sum(1), 1)


def test_expert_monotonicity_and_tree_depth(fitted):
    m, X, _ = fitted

    def depth(node):
        return 0 if "leaf" in node else 1 + max(depth(c) for c in node["children"])

    for e in m.experts_:
        assert e.get_params()["monotone_constraints"] == (1, -1, 0)
        assert max(depth(json.loads(t)) for t in e.get_booster().get_dump(dump_format="json")) <= 2
        p = e.predict_proba(X)[:, 1]
        for j, direction in ((0, 1), (1, -1)):
            shifted = X.copy()
            shifted[:, j] += 0.4
            assert np.all(direction * (e.predict_proba(shifted)[:, 1] - p) >= -1e-7)


def test_prediction_is_batch_independent_and_serializable(fitted):
    m, X, _ = fitted
    before = m.scaler_.scale_.copy()
    p = m.predict_proba(X[:5])
    np.testing.assert_allclose(p, np.vstack([m.predict_proba(row[None]) for row in X[:5]]))
    changed_batch = np.vstack([X[:5], np.full((3, X.shape[1]), 1000.0)])
    np.testing.assert_allclose(p, m.predict_proba(changed_batch)[:5])
    np.testing.assert_array_equal(before, m.scaler_.scale_)
    restored = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(p, restored.predict_proba(X[:5]))
    np.testing.assert_array_equal(m.predict_cluster(X), m.kmeans_.predict(m.scaler_.transform(X)))


def test_calibration_failure_propagates_and_invalidates_refit(monkeypatch):
    X, y = data()
    m = model().fit(X, y)

    def fail(*args, **kwargs):
        raise RuntimeError("calibration failed for test")

    monkeypatch.setattr("aiv.monotone_moe.CalibratedClassifierCV.fit", fail)
    with pytest.raises(RuntimeError, match="calibration failed"):
        m.fit(X, y)
    with pytest.raises(NotFittedError):
        m.predict_proba(X)


def test_collapsed_or_undersupported_clusters_rejected():
    X = np.zeros((20, 3))
    X[-1] = 10
    with pytest.raises(ValueError, match="undersupported"):
        model().fit(X, np.tile([0, 1], 10))


@pytest.mark.parametrize("weights", [np.zeros(240), np.full(240, np.nan), np.ones(239), -np.ones(240)])
def test_invalid_observation_weights_rejected(weights):
    X, y = data()
    with pytest.raises(ValueError, match="sample_weight"):
        model().fit(X, y, sample_weight=weights)


def test_feature_order_and_invalid_configuration():
    X, y = data()
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    m = model().fit(df, y)
    with pytest.raises(ValueError, match="names/order"):
        m.predict_proba(df[["b", "a", "c"]])
    with pytest.raises(ValueError, match="cannot override"):
        MonotoneKMeansMoEClassifier((1, -1, 0), expert_params={"max_depth": 3}).fit(X, y)
    with pytest.raises(ValueError, match="monotone_constraints"):
        MonotoneKMeansMoEClassifier((0, 0, 0)).fit(X, y)
    with pytest.raises(ValueError, match="binary"):
        model().fit(X, np.full(len(y), 0.5))

def test_fixed_candidate_geometry(fitted):
    original, X, y = fitted
    centers = original.kmeans_.cluster_centers_.copy()
    candidate = model().fit(X, y, centroids=centers, scaler=original.scaler_)
    np.testing.assert_array_equal(candidate.cluster_labels_, original.cluster_labels_)
    np.testing.assert_allclose(candidate.predict_proba(X), original.predict_proba(X))
    assert candidate.scaler_ is not original.scaler_
    centers[:] = 100
    np.testing.assert_array_equal(candidate.kmeans_.cluster_centers_, original.kmeans_.cluster_centers_)


def test_geometry_is_not_refitted(fitted):
    original, X, y = fitted
    centers = original.kmeans_.cluster_centers_ + 0.05
    changed = X + 0.2
    candidate = model().fit(changed, y, centroids=centers, scaler=original.scaler_)
    np.testing.assert_array_equal(candidate.kmeans_.cluster_centers_, centers)
    np.testing.assert_array_equal(candidate.scaler_.mean_, original.scaler_.mean_)
    expected = ((original.scaler_.transform(changed)[:, None] - centers) ** 2).sum(2).argmin(1)
    np.testing.assert_array_equal(candidate.cluster_labels_, expected)


def test_candidate_geometry_validation(fitted):
    original, X, y = fitted
    from aiv.monotone_moe import UnsupportedPartitionError
    with pytest.raises(ValueError, match="both centroids"):
        model().fit(X, y, centroids=original.kmeans_.cluster_centers_)
    with pytest.raises(ValueError, match="shape"):
        model().fit(X, y, centroids=np.zeros((2, 2)), scaler=original.scaler_)
    with pytest.raises(UnsupportedPartitionError):
        model().fit(X, y, centroids=np.zeros((2, 3)), scaler=original.scaler_)


def test_sparse_gate_excludes_features_in_fit_and_prediction():
    from sklearn.base import clone
    X, y = data()
    estimator = model().set_params(gate_features=(2, 0))
    m = estimator.fit(X, y)
    changed = X.copy()
    changed[:, 1] = np.linspace(-1000, 1000, len(X))
    other = clone(estimator).fit(changed, y)
    for routed in (changed, X):
        np.testing.assert_array_equal(m.predict_cluster(routed), m.predict_cluster(X))
        np.testing.assert_array_equal(m.predict_gate_proba(routed), m.predict_gate_proba(X))
    np.testing.assert_array_equal(other.cluster_labels_, m.cluster_labels_)
    np.testing.assert_array_equal(other.training_gate_probabilities_, m.training_gate_probabilities_)
    assert all(e.n_features_in_ == X.shape[1] for e in m.experts_)
    assert m.scaler_.n_features_in_ == 2
    np.testing.assert_allclose(m.scaler_.mean_, X[:, (2, 0)].mean(0))
    frozen = clone(estimator).fit(X, y, centroids=m.kmeans_.cluster_centers_, scaler=m.scaler_)
    np.testing.assert_array_equal(frozen.predict_proba(X), m.predict_proba(X))
    restored = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(restored.predict_gate_proba(changed), m.predict_gate_proba(X))
    # Experts still respond to the excluded predictor, using all rows and columns.
    assert np.max(abs(m.predict_expert_proba(changed) - m.predict_expert_proba(X))) > 0


@pytest.mark.parametrize("features", [[], [0, 0], [-1], [3], [0.5], [True], [[0, 1]]])
def test_invalid_gate_features(features):
    X, y = data()
    with pytest.raises(ValueError, match="gate_features"):
        model().set_params(gate_features=features).fit(X, y)


def test_historical_model_prediction_compatibility(fitted):
    m, X, _ = fitted
    restored = pickle.loads(pickle.dumps(m))
    del restored.gate_feature_indices_
    del restored.gate_features
    np.testing.assert_array_equal(restored.predict_proba(X), m.predict_proba(X))


def test_three_experts_calibration_weights_and_sparse_routing(monkeypatch):
    from sklearn.base import clone
    X, y = data()
    captured = []
    original = XGBClassifier.fit

    @wraps(original)
    def spy(self, X, y, **kwargs):
        if self.max_depth == 2:
            captured.append((X.shape, np.array(kwargs['sample_weight'])))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(XGBClassifier, 'fit', spy)
    m = model().set_params(gate_features=(0, 2), n_clusters=3).fit(X, y)
    G = m.predict_gate_proba(X)
    assert G.shape == (len(X), 3)
    np.testing.assert_allclose(G.sum(1), 1)
    assert np.all(G > 0)
    assert len(captured) == 3
    for h, (shape, weights) in enumerate(captured):
        assert shape == X.shape
        np.testing.assert_array_equal(weights, G[:, h])
    np.testing.assert_array_equal(m.training_gate_probabilities_, G)
    np.testing.assert_allclose(m.predict_proba(X)[:, 1], (G*m.predict_expert_proba(X)).sum(1))
    changed = X.copy(); changed[:, 1] += 100
    np.testing.assert_array_equal(m.predict_gate_proba(changed), G)
    np.testing.assert_array_equal(m.predict_cluster(changed), m.predict_cluster(X))
    restored = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(restored.predict_proba(X), m.predict_proba(X))
    fixed = clone(m).fit(X, y, centroids=m.kmeans_.cluster_centers_, scaler=m.scaler_)
    np.testing.assert_array_equal(fixed.predict_proba(X), m.predict_proba(X))


@pytest.mark.parametrize('count', [1, 0, -1, True, 2.5])
def test_invalid_cluster_count(count):
    X, y = data()
    with pytest.raises(ValueError, match='n_clusters'):
        model().set_params(n_clusters=count).fit(X, y)


def test_historical_estimator_remains_cloneable_after_loading(fitted):
    from sklearn.base import clone
    m, X, y = fitted
    old = pickle.loads(pickle.dumps(m))
    del old.gate_features
    del old.n_clusters
    restored = pickle.loads(pickle.dumps(old))
    assert restored.get_params()['n_clusters'] == 2
    assert restored.get_params()['gate_features'] is None
    fresh = clone(restored).fit(X,y)
    np.testing.assert_array_equal(fresh.predict_proba(X), m.predict_proba(X))
