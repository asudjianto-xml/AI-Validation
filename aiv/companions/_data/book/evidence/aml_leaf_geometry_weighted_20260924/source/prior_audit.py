"""Replay saved models and independently recompute the reported evidence."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from xgboost import XGBRegressor
from run import OUT, sha, leaf_distance, summarize, generate
from purified_tree_fanova import DepthTwoFANOVA


def audit():
    r = json.loads((OUT/'results.json').read_text())
    for name, digest in r['source_hashes'].items():
        assert sha(OUT/'source'/name) == digest
    freeze = json.loads((OUT/'frozen_selection.json').read_text())
    a = np.load(OUT/'observations.npz')
    base = joblib.load(OUT/'base.joblib')
    aux = XGBRegressor(); aux.load_model(OUT/'residual_model.json')
    assert sha(OUT/'base.joblib') == freeze['base_sha256']
    assert sha(OUT/'residual_model.json') == freeze['auxiliary_sha256']
    assert sha(OUT/'frozen_selection.json') == r['frozen_selection_sha256']
    for role in ['train', 'discovery', 'confirmation']:
        X, y, group = generate(r['config'][role+'_n'], r['config'][role+'_seed'])
        np.testing.assert_array_equal(X, a['X_'+role])
        np.testing.assert_array_equal(y, a['y_'+role])
        np.testing.assert_array_equal(group, a['group_'+role])
    medoid_leaves = np.asarray(freeze['medoid_leaves'])
    reconstruction = []
    for split in ['discovery', 'confirmation']:
        X = pd.DataFrame(a['X_'+split], columns=r['features'])
        y, p = a['y_'+split], base.predict_proba(X)[:, 1]
        np.testing.assert_allclose(p, a['p_'+split], atol=1e-14)
        leaves = aux.apply(X).astype(int)
        labels = leaf_distance(leaves, medoid_leaves).argmin(1)
        np.testing.assert_array_equal(labels, a['cluster_'+split])
        assert summarize(y, p, labels) == r[split]
        if split == 'discovery':
            fanova = DepthTwoFANOVA(aux, X).purified('product_marginals')
            np.testing.assert_array_equal(leaves, a['leaf_discovery'])
            for v in r['js']:
                j = r['features'].index(v['feature'])
                edges = np.r_[-np.inf, v['interior_edges'], np.inf]
                worst = labels == r['worst_cluster']
                P = np.histogram(X.iloc[worst, j], edges)[0]
                Q = np.histogram(X.iloc[~worst, j], edges)[0]
                np.testing.assert_allclose(jensenshannon(P,Q,base=2)**2, v['divergence_bits'])
            for j, v in enumerate(fanova.main):
                assert abs(v @ fanova.marginal[j]) < 1e-10
            for (j,k), v in fanova.pair.items():
                assert np.max(abs(v @ fanova.marginal[k])) < 1e-10
                assert np.max(abs(fanova.marginal[j] @ v)) < 1e-10
            indices = np.array(freeze['pam_sample_indices'])
            expected = np.sort(np.random.default_rng(r['config']['pam_sample_seed']).choice(
                len(X), r['config']['pam_sample_n'], replace=False))
            np.testing.assert_array_equal(indices, expected)
            assert set(freeze['medoid_indices']).issubset(set(indices))
            np.testing.assert_array_equal(medoid_leaves, leaves[freeze['medoid_indices']])
            eligible = [v for v in r['discovery']['clusters'] if
                        v['n'] >= r['config']['min_cluster_n'] and
                        v['share'] >= r['config']['min_cluster_share']]
            assert max(eligible,key=lambda v:(v['brier'],-v['cluster']))['cluster'] == r['worst_cluster']
            # Independently count shared leaves, checking tree-specific encoding.
            from sklearn.preprocessing import OneHotEncoder
            H = OneHotEncoder(sparse_output=True, dtype=np.float64).fit_transform(leaves[:150])
            S = (H @ H.T).toarray()/leaves.shape[1]
            for i in range(150):
                np.testing.assert_allclose(S[i], (leaves[:150] == leaves[i]).mean(axis=1))
            np.testing.assert_allclose(np.diag(S), 1)
            assert np.linalg.eigvalsh(S).min() > -1e-10
        error = float(np.max(abs(fanova.predict(X)-aux.predict(X))))
        assert error < 3e-6
        reconstruction.append(error)
    # Genuine pair-only fixture catches a decomposition that simply drops interactions.
    from xgboost import XGBRegressor as Regressor
    grid = np.tile(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32), (100,1))
    target = ((grid[:,0]-.5)*(grid[:,1]-.5)).astype(float)
    model = Regressor(max_depth=2,n_estimators=1,learning_rate=1.,reg_lambda=0.,
                      min_child_weight=0.,gamma=0.,tree_method='exact',base_score=0.,n_jobs=1)
    # Add a main effect so the greedy first split has positive gain.
    target += grid[:,0]
    model.fit(grid,target)
    f = DepthTwoFANOVA(model,grid).purified('product_marginals')
    np.testing.assert_allclose(f.predict(grid),model.predict(grid),atol=3e-6)
    assert np.max(abs(f.pair[0,1])) > .2
    report = dict(status='passed', observations_replayed=sum(r['config'][s+'_n'] for s in ['train','discovery','confirmation']),
                  checks=['data regeneration','model predictions','frozen assignments','all cluster metrics',
                          'JS divergence','leaf kernel identity and PSD','fANOVA centering and reconstruction',
                          'known interaction fixture','freeze hashes','source snapshot hashes','declared PAM subset and medoids','discovery selection'], max_reconstruction_error=max(reconstruction))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    audit()
