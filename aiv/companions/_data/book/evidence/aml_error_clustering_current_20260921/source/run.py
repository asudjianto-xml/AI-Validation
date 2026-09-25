"""Error-aware AML clustering: frozen baseline, depth-two residual model and PAM.

Run with the environment recorded in README.md. Refuses to overwrite evidence.
"""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, r2_score
from sklearn_extra.cluster import KMedoids
import xgboost
from xgboost import XGBRegressor

from aml_capacity_data import FEATURES
import run_aml_frontier_challenge as current_generator
from purified_tree_fanova import DepthTwoFANOVA

BOOK = Path(__file__).resolve().parents[1]
OUT = BOOK / 'evidence/aml_error_clustering_current_20260921'
CONFIG = dict(train_n=40000, discovery_n=40000, confirmation_n=200000,
              pam_sample_n=3000, pam_sample_seed=62206,
              population_event_rates=current_generator.EVENT_RATES.tolist(),
              deviation_norms=current_generator.DEVIATION_NORMS.tolist(),
              hidden_driver_sd=current_generator.HIDDEN_DRIVER_SD.tolist(),
              generator="snapshot of current paper/run_aml_frontier_challenge.py",
              train_seed=62201, discovery_seed=62202, confirmation_seed=62203,
              n_estimators=120, max_depth=2, learning_rate=.05,
              n_clusters=4, min_cluster_n=50, min_cluster_share=.01,
              auxiliary_target='squared residual (Brier loss)',
              fanova_measure='product of discovery empirical marginals',
              selection='largest discovery mean Brier among eligible clusters',
              confirmation='one frozen worst-versus-rest contrast; conditional iid normal interval')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(n, seed):
    return current_generator.generate(n, seed)


def leaf_distance(leaves, medoid_leaves):
    # Blocked query-to-medoid assignment requires O(block_size * k * T) workspace.
    result = np.empty((len(leaves), len(medoid_leaves)))
    for start in range(0, len(leaves), 1000):
        result[start:start+1000] = np.mean(
            leaves[start:start+1000, None, :] != medoid_leaves[None, :, :], axis=2)
    return result


def summarize(y, p, labels):
    loss = (y-p)**2
    rows = []
    for k in range(CONFIG['n_clusters']):
        mask = labels == k
        a, b = loss[mask], loss[~mask]
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        gap = float(a.mean()-b.mean())
        rows.append(dict(cluster=k, n=int(mask.sum()), share=float(mask.mean()),
                         brier=float(a.mean()), rest_brier=float(b.mean()),
                         excess_overall=float(a.mean()-loss.mean()), gap_rest=gap,
                         gap_rest_ci95=[gap-1.96*se, gap+1.96*se],
                         inflation=float(a.mean()/loss.mean()-1),
                         prevalence=float(y[mask].mean()),
                         auc=float(roc_auc_score(y[mask], p[mask]))
                         if len(np.unique(y[mask])) == 2 else None))
    return dict(n=len(y), overall_brier=float(loss.mean()), clusters=rows)


def run():
    if (OUT/'results.json').exists() or (OUT/'protocol.json').exists():
        raise RuntimeError('Evidence already exists; use a new versioned run directory.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'source').mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, OUT/'source'/path.name)
    save(OUT/'protocol.json', CONFIG)
    X, y, g = generate(CONFIG['train_n'], CONFIG['train_seed'])
    Xd, yd, gd = generate(CONFIG['discovery_n'], CONFIG['discovery_seed'])
    base = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
    base.fit(X, y)
    joblib.dump(base, OUT/'base.joblib')
    pd = base.predict_proba(Xd)[:, 1]
    residual = yd-pd
    aux = XGBRegressor(n_estimators=CONFIG['n_estimators'], max_depth=2,
                       learning_rate=.05, objective='reg:squarederror',
                       tree_method='hist', n_jobs=2, random_state=62104,
                       subsample=1., colsample_bytree=1.)
    aux.fit(Xd, residual**2)
    aux.save_model(OUT/'residual_model.json')
    leaves = aux.apply(Xd).astype(np.int32)
    encoder = OneHotEncoder(dtype=np.float32, sparse_output=True)
    pam_indices = np.sort(np.random.default_rng(CONFIG['pam_sample_seed']).choice(
        len(leaves), CONFIG['pam_sample_n'], replace=False))
    H = encoder.fit_transform(leaves[pam_indices])
    start = time.perf_counter()
    similarity = (H @ H.T).toarray()/leaves.shape[1]
    distance = np.clip(1.-similarity, 0., 1.)
    np.fill_diagonal(distance, 0.)
    proximity_seconds = time.perf_counter()-start
    np.testing.assert_allclose(distance[:100, :100],
        leaf_distance(leaves[pam_indices[:100]], leaves[pam_indices[:100]]), atol=1e-7)
    start = time.perf_counter()
    pam = KMedoids(n_clusters=4, metric='precomputed', method='pam', init='build',
                   max_iter=300, random_state=62105).fit(distance)
    pam_seconds = time.perf_counter()-start
    if pam.n_iter_ >= 299:
        raise RuntimeError('PAM did not converge within the declared budget')
    medoids = pam_indices[pam.medoid_indices_]
    labels = leaf_distance(leaves, leaves[medoids]).argmin(axis=1)
    np.testing.assert_array_equal(labels[pam_indices], pam.labels_)
    discovery = summarize(yd, pd, labels)
    eligible = [r for r in discovery['clusters'] if r['n'] >= CONFIG['min_cluster_n']
                and r['share'] >= CONFIG['min_cluster_share']]
    worst = max(eligible, key=lambda r: (r['brier'], -r['cluster']))['cluster']
    # All descriptive choices use discovery only, including histogram boundaries.
    js = []
    for feature in FEATURES:
        edges = np.unique(np.r_[-np.inf, np.quantile(Xd[feature], np.arange(.1, 1., .1)), np.inf])
        a = np.histogram(Xd.loc[labels == worst, feature], edges)[0]
        b = np.histogram(Xd.loc[labels != worst, feature], edges)[0]
        js.append(dict(feature=feature, divergence_bits=float(jensenshannon(a, b, base=2)**2),
                       median_worst=float(Xd.loc[labels == worst, feature].median()),
                       median_rest=float(Xd.loc[labels != worst, feature].median()),
                       interior_edges=edges[1:-1].tolist(),
                       worst_mass=(a/a.sum()).tolist(), rest_mass=(b/b.sum()).tolist()))
    js.sort(key=lambda r: -r['divergence_bits'])
    fanova = DepthTwoFANOVA(aux, Xd).purified('product_marginals')
    np.testing.assert_allclose(fanova.predict(Xd), aux.predict(Xd), atol=3e-6)
    main_importance = np.array([np.dot(v*v, w) for v, w in zip(fanova.main, fanova.marginal)])
    pair_importance = {pair:float(np.sum(v*v*np.outer(fanova.marginal[pair[0]], fanova.marginal[pair[1]])))
                       for pair, v in fanova.pair.items()}
    top_main = np.argsort(-main_importance)[:3].tolist()
    top_pair = max(pair_importance, key=pair_importance.get)
    save(OUT/'frozen_selection.json', dict(worst_cluster=worst, medoid_indices=medoids.tolist(),
         medoid_leaves=leaves[medoids].tolist(), tie_rule='lowest medoid position',
         top_main=top_main, top_pair=top_pair, js=js, pam_sample_indices=pam_indices.tolist(),
         base_sha256=sha(OUT/'base.joblib'), auxiliary_sha256=sha(OUT/'residual_model.json')))
    freeze_hash = sha(OUT/'frozen_selection.json')
    # Confirmation is generated only after the membership rule and selected claim are saved.
    Xc, yc, gc = generate(CONFIG['confirmation_n'], CONFIG['confirmation_seed'])
    pc = base.predict_proba(Xc)[:, 1]
    lc = leaf_distance(aux.apply(Xc).astype(np.int32), leaves[medoids]).argmin(axis=1)
    confirmation = summarize(yc, pc, lc)
    np.testing.assert_allclose(fanova.predict(Xc), aux.predict(Xc), atol=3e-6)
    assert sha(OUT/'frozen_selection.json') == freeze_hash
    np.savez_compressed(OUT/'observations.npz', X_train=X.to_numpy(), y_train=y, group_train=g,
        X_discovery=Xd.to_numpy(), y_discovery=yd, group_discovery=gd, p_discovery=pd,
        residual_discovery=residual, leaf_discovery=leaves, cluster_discovery=labels,
        X_confirmation=Xc.to_numpy(), y_confirmation=yc, group_confirmation=gc,
        p_confirmation=pc, cluster_confirmation=lc)
    results = dict(config=CONFIG, features=FEATURES, discovery=discovery, confirmation=confirmation,
        worst_cluster=worst, js=js, fanova=dict(top_main=[FEATURES[j] for j in top_main],
        top_pair=[FEATURES[j] for j in top_pair], main_variances=main_importance.tolist(),
        pair_variances={','.join(FEATURES[j] for j in pair):v for pair,v in pair_importance.items()},
        max_reconstruction_error=float(np.max(abs(fanova.predict(Xc)-aux.predict(Xc))))),
        auxiliary_r2=dict(discovery=float(r2_score(residual**2, aux.predict(Xd))),
                          confirmation=float(r2_score((yc-pc)**2, aux.predict(Xc)))),
        computation=dict(one_hot_shape=H.shape, one_hot_nnz=H.nnz,
            dense_matrix_bytes=distance.nbytes, proximity_seconds=proximity_seconds,
            pam_seconds=pam_seconds, pam_iterations=int(pam.n_iter_), pam_inertia=float(pam.inertia_)),
        source_hashes={p.name:sha(p) for p in sorted((OUT/'source').glob('*.py'))},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      sklearn=sklearn.__version__, xgboost=xgboost.__version__),
        frozen_selection_sha256=freeze_hash)
    save(OUT/'results.json', results)
    plot(fanova, Xd, results, top_main, top_pair)
    save(OUT/'manifest.json', {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(results, indent=2))


def plot(fanova, Xd, results, top_main, top_pair):
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for ax, j in zip(axs.flat, top_main):
        low, high = np.quantile(Xd.iloc[:, j], [.01, .99])
        grid = np.linspace(low, high, 300)
        values = fanova.main[j][np.searchsorted(fanova.cuts[j], grid, side='right')]
        ax.step(grid, values, where='post', color='#245c89')
        ax.axhline(0, color='gray', linewidth=.6)
        ax.set(xlabel=FEATURES[j].replace('_', ' '), ylabel='Contribution to predicted Brier loss')
    ax = axs.flat[3]
    j, k = top_pair
    a = np.linspace(*np.quantile(Xd.iloc[:, j], [.01, .99]), 100)
    b = np.linspace(*np.quantile(Xd.iloc[:, k], [.01, .99]), 100)
    z = fanova.pair[j,k][np.searchsorted(fanova.cuts[j], a, side='right')[:,None],
                           np.searchsorted(fanova.cuts[k], b, side='right')[None,:]]
    limit = max(abs(z.min()), abs(z.max()))
    im = ax.pcolormesh(a, b, z.T, shading='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax.set(xlabel=FEATURES[j].replace('_',' '), ylabel=FEATURES[k].replace('_',' '))
    fig.colorbar(im, ax=ax, label='Pair contribution')
    fig.savefig(OUT/'fanova.pdf'); fig.savefig(OUT/'fanova.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for split, offset, color in [('discovery', -.18, '#829db0'), ('confirmation', .18, '#245c89')]:
        rows = results[split]['clusters']
        axs[0].bar(np.arange(4)+offset, [r['brier'] for r in rows], width=.36, label=split, color=color)
    axs[0].axhline(results['confirmation']['overall_brier'], linestyle='--', color='black', label='confirmation overall')
    axs[0].set(xticks=range(4), xlabel='Frozen cluster ID', ylabel='Base-model Brier loss')
    axs[0].legend(fontsize=8)
    top = results['js'][:6][::-1]
    axs[1].barh([r['feature'].replace('_',' ') for r in top], [r['divergence_bits'] for r in top], color='#245c89')
    axs[1].set(xlabel='Jensen–Shannon divergence (bits)', title='Discovery: selected worst cluster versus rest')
    fig.savefig(OUT/'clusters.pdf'); fig.savefig(OUT/'clusters.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    run()
