"""Read-only loaders for the Chapter 2 clustering evidence. No fitting and no writes."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
from companion import ROOT, BOOK, read, sha

CLUSTERING = BOOK/'evidence/aml_error_clustering_current_20260921'
GEOMETRY = BOOK/'evidence/aml_leaf_geometry_weighted_20260924'
ARMS = ['equal_leaf', 'gain_leaf', 'leaf_output']
NAMES = dict(equal_leaf='Unweighted leaf proximity', gain_leaf='Gain-weighted leaf proximity',
             leaf_output='Weighted leaf proximity')


def check_manifest(folder):
    for name, digest in read(folder/'manifest.json').items():
        assert sha(folder/name) == digest, name


def load_geometry():
    """Hash-checked leaf-geometry comparison: results, frozen selections and assignments."""
    check_manifest(GEOMETRY)
    for path, digest in read(GEOMETRY/'inputs.json').items():
        assert sha(ROOT/path) == digest, path
    results, frozen = read(GEOMETRY/'results.json'), read(GEOMETRY/'frozen_selection.json')
    assert sha(GEOMETRY/'frozen_selection.json') == results['frozen_sha256']
    with np.load(GEOMETRY/'assignments.npz') as saved:
        assignments = {k: saved[k].copy() for k in saved.files}
    return results, frozen, assignments


def load_clustering():
    """Hash-checked error-clustering run: results and discovery observations."""
    check_manifest(CLUSTERING)
    results = read(CLUSTERING/'results.json')
    with np.load(CLUSTERING/'observations.npz') as saved:
        observations = {k: saved[k].copy() for k in saved.files
                        if k.endswith('_discovery')}
    return results, observations


def auxiliary_model():
    from xgboost import XGBRegressor
    assert sha(GEOMETRY/'residual_model.json') == sha(CLUSTERING/'residual_model.json')
    model = XGBRegressor(); model.load_model(CLUSTERING/'residual_model.json')
    return model


def fanova_class(results):
    """Import the recorded purification code after checking it against the run's source hash."""
    path = BOOK/'error_clustering_current/purified_tree_fanova.py'
    assert sha(path) == results['source_hashes']['purified_tree_fanova.py']
    spec = importlib.util.spec_from_file_location('purified_tree_fanova', path)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.DepthTwoFANOVA


def frozen_predictor():
    """Load the source-checked predictor without fitting it."""
    import joblib
    check_manifest(CLUSTERING)
    return joblib.load(CLUSTERING/'base.joblib')
