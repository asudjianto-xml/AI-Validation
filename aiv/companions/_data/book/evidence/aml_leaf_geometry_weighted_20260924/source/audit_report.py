"""Replay frozen geometry comparison and generate a standalone results report."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location('geometry_experiment', Path(__file__).with_name('run.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
out = e.OUT
r, f = e.read(out/'results.json'), e.read(out/'frozen_selection.json')
for name, digest in e.read(out/'manifest.json').items():
    assert e.sha(out/name) == digest, name
for name, digest in e.read(out/'inputs.json').items():
    assert e.sha(e.REPO/name) == digest, name
assert e.sha(out/'frozen_selection.json') == r['frozen_sha256']
assert e.sha(out/'protocol.json') == f['protocol_sha256']
a = np.load(e.PRIOR/'observations.npz')
b = np.load(out/'assignments.npz')
model = e.XGBRegressor(); model.load_model(out/'residual_model.json')
# Independent table export: Gain stores prediction values on leaf rows only.
table = model.get_booster().trees_to_dataframe()
leaves = table[table.Feature == 'Leaf']
tables = [{int(row.Node): float(row.Gain) for row in leaves[leaves.Tree == t].itertuples()} for t in range(120)]
weights = table[table.Feature != 'Leaf'].groupby('Tree').Gain.sum().reindex(range(120), fill_value=0).to_numpy()
weights /= weights.sum()
np.testing.assert_allclose(weights, f['gain_weights'], atol=1e-14)
Xc, yc, gc = e.generate(200000, 62403)
for key, value in [('X_confirmation', Xc.to_numpy()), ('y_confirmation', yc), ('group_confirmation', gc)]:
    np.testing.assert_array_equal(b[key], value)
pc = joblib.load(out/'base.joblib').predict_proba(Xc)[:, 1]
np.testing.assert_array_equal(pc, b['p_confirmation'])
codes = a['leaf_discovery'].astype(int)
cc = model.apply(Xc).astype(int)
Z, Zc = e.output_embedding(codes, tables), e.output_embedding(cc, tables)
np.testing.assert_allclose(Zc.sum(1)+f['base_score'], model.predict(Xc), atol=5e-7)
pool = np.asarray(f['pam_pool'])
dist = e.pool_distances(codes[pool], Z[pool], weights)
for arm in e.ARMS:
    med = np.asarray(f['arms'][arm]['medoid_indices'])
    assert np.isin(med, pool).all()
    pam = e.KMedoids(n_clusters=4, metric='precomputed', method='pam', init='build', max_iter=300, random_state=62105).fit(dist[arm])
    np.testing.assert_array_equal(pool[pam.medoid_indices_], med)
    for split, c, z in [('discovery', codes, Z), ('confirmation', cc, Zc)]:
        if arm == 'leaf_output':
            lab = cdist(z, Z[med]).argmin(1)
        else:
            lab = np.empty(len(c), int)
            for start in range(0, len(c), 1000):
                mismatch = c[start:start+1000, None, :] != codes[med][None, :, :]
                d = mismatch.mean(2) if arm == 'equal_leaf' else (mismatch*weights).sum(2)
                lab[start:start+1000] = d.argmin(1)
        np.testing.assert_array_equal(lab, b[split+'_'+arm])
    dr = r['arms'][arm]['discovery']['clusters']
    selected = max((row for row in dr if row['n'] >= 50 and row['share'] >= .01), key=lambda row:(row['brier'], -row['cluster']))['cluster']
    assert selected == f['arms'][arm]['worst_cluster']
    for split, y, p, rows in [('discovery', a['y_discovery'], a['p_discovery'], dr), ('confirmation', yc, pc, r['confirmation'][arm]['clusters'])]:
        for row in rows:
            mask = b[split+'_'+arm] == row['cluster']
            loss = (y-p)**2
            assert mask.sum() == row['n']
            np.testing.assert_allclose([loss[mask].mean(), loss[~mask].mean(), y[mask].mean()], [row['brier'], row['rest_brier'], row['prevalence']], atol=1e-14)
np.testing.assert_array_equal(b['discovery_equal_leaf'], a['cluster_discovery'])
loss = (yc-pc)**2
influences, masks = {}, {}
critical = norm.ppf(1-.05/10)
for arm in e.ARMS:
    mask = b['confirmation_'+arm] == f['arms'][arm]['worst_cluster']; masks[arm] = mask
    n1, n0 = mask.sum(), (~mask).sum()
    gap = loss[mask].mean()-loss[~mask].mean()
    influence = np.where(mask, (loss-loss[mask].mean())*len(loss)/n1, -(loss-loss[~mask].mean())*len(loss)/n0)
    influences[arm] = influence
    se = np.sqrt((len(loss)/(len(loss)-1))*(loss[mask].var()/n1+loss[~mask].var()/n0))
    np.testing.assert_allclose(r['confirmation'][arm]['selected']['simultaneous_ci'], [gap-critical*se, gap+critical*se])
for comparison in r['comparisons']:
    arm = comparison['arm']
    se = (influences[arm]-influences['equal_leaf']).std(ddof=1)/np.sqrt(len(loss))
    gap = r['confirmation'][arm]['selected']['contrast']-r['confirmation']['equal_leaf']['selected']['contrast']
    np.testing.assert_allclose(comparison['simultaneous_ci'], [gap-critical*se, gap+critical*se])
    np.testing.assert_allclose(comparison['selected_jaccard'], (masks[arm]&masks['equal_leaf']).sum()/(masks[arm]|masks['equal_leaf']).sum())
for split in ['discovery', 'confirmation']:
    for arm in e.ARMS:
        for other in e.ARMS:
            assert adjusted_rand_score(b[split+'_'+arm], b[split+'_'+other]) == r['ari'][split][arm][other]
e.save(out/'audit.json', dict(status='PASS', results_sha256=e.sha(out/'results.json'), checks=['input and artifact hashes', 'independent leaf-table extraction and gain weights', 'fresh sample regeneration', 'prediction reconstruction', 'all three PAM fits replayed', '240000 assignments per arm independently replayed', 'baseline exactly reproduced', 'discovery selection rule', 'cluster means and event rates', 'five simultaneous intervals and paired covariance', 'partition ARI and selected-region Jaccard']))

names = dict(equal_leaf='Unweighted leaf proximity', gain_leaf='Gain-weighted leaf proximity', leaf_output='Leaf-output embedding')
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, arm in zip(axes, e.ARMS):
    rows = r['confirmation'][arm]['clusters']; selected = f['arms'][arm]['worst_cluster']
    ax.bar(range(4), [x['brier'] for x in rows], color=['#bc4639' if i == selected else '#487eaa' for i in range(4)])
    ax.plot(range(4), [x['brier'] for x in r['arms'][arm]['discovery']['clusters']], 'ko', markersize=4, label='Discovery')
    ax.axhline(loss.mean(), color='#555555', linestyle='--', linewidth=1, label='Confirmation overall')
    ax.set(title=names[arm], xlabel='Cluster (IDs are local to each method)', xticks=range(4), ylim=(0,.065))
    for i, row in enumerate(rows):
        ax.text(i, row['brier']+.003, f"{row['share']:.1%}", ha='center', fontsize=9)
axes[0].set_ylabel('Mean Brier loss (higher = worse)')
axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('Same frozen AML model, different clustering geometry', fontsize=14)
fig.text(.5,.01,'Bars: fresh confirmation (n = 200,000). Red: region selected on discovery. Labels: population share.', ha='center', fontsize=10)
fig.tight_layout(rect=(0,.045,1,.95))
for suffix in ['png','pdf']: fig.savefig(out/f'clustering_comparison.{suffix}', dpi=180)
lines = ['# Executed AML clustering comparison', '', 'The leaf-output embedding identifies a higher-loss region with substantially stronger contrast on fresh confirmation. The predictive model is unchanged.', '', '## Frozen design', '', 'The same Chapter 2 logistic model, 120 depth-two error trees, 40,000 discovery observations, 3,000-point PAM pool and four clusters are used in all arms. PAM uses deterministic BUILD initialization. Each method selects its largest-mean-Brier eligible cluster on discovery (at least 50 observations and 1% share). All three selections were saved before generating the same fresh 200,000-observation confirmation sample (seed 62403).', '', 'Unweighted proximity counts shared leaves. Gain weighting assigns each tree its normalized sum of internal split gains. The output embedding uses the 120 actual leaf prediction contributions and Euclidean distance, without standardizing tree coordinates or multiplying by shrinkage again. Summed coordinates plus the intercept reproduce auxiliary predictions within 5e-7.', '', '## Fresh confirmation', '', '| Geometry | Selected share | Region Brier | Rest Brier | Contrast | Simultaneous 95% CI |', '|---|---:|---:|---:|---:|---|']
for arm in e.ARMS:
    row=r['confirmation'][arm]['clusters'][f['arms'][arm]['worst_cluster']]; lo,hi=r['confirmation'][arm]['selected']['simultaneous_ci']
    lines.append(f"| {names[arm]} | {row['share']:.2%} | {row['brier']:.6f} | {row['rest_brier']:.6f} | {row['contrast']:.6f} | [{lo:.6f}, {hi:.6f}] |")
lines += ['', f'Overall confirmation Brier is {loss.mean():.6f} for all methods. Contrast means region Brier minus rest-of-population Brier.', '', '| Comparison against unweighted | Contrast increase | Simultaneous 95% CI | Selected-region Jaccard | Partition ARI |', '|---|---:|---|---:|---:|']
for c in r['comparisons']:
    lo,hi=c['simultaneous_ci']; arm=c['arm']
    lines.append(f"| {names[arm]} | {c['contrast_difference']:.6f} | [{lo:.6f}, {hi:.6f}] | {c['selected_jaccard']:.3f} | {r['ari']['confirmation'][arm]['equal_leaf']:.3f} |")
lines += ['', 'Intervals use paired observation-level influence functions and Bonferroni adjustment across five claims: three regional contrasts and two between-method differences. They are conditional on the fitted models and frozen selections; they do not measure variation across training samples or PAM pools.', '', '![Clustering comparison](clustering_comparison.png)', '', '## All confirmation clusters', '', 'Cluster IDs have meaning only within each method. An asterisk marks the discovery-selected region.', '', '| Geometry | Cluster | Share | Brier | Event rate | AUC |', '|---|---:|---:|---:|---:|---:|']
for arm in e.ARMS:
    for row in r['confirmation'][arm]['clusters']:
        mark='*' if row['cluster']==f['arms'][arm]['worst_cluster'] else ''
        lines.append(f"| {names[arm]} | {row['cluster']}{mark} | {row['share']:.2%} | {row['brier']:.6f} | {row['prevalence']:.2%} | {row['auc']:.3f} |")
lines += ['', '## What changed in the selected region?', '', 'Discovery-only feature profiles use Jensen–Shannon divergence in bits between the selected region and its complement, with common discovery-decile bins. These marginal summaries describe membership; they do not establish causal error drivers.', '']
for arm in e.ARMS:
    profiles=r['arms'][arm]['js'][:3]
    lines.append(f"- **{names[arm]}:** "+'; '.join(f"{j['feature']} (JS {j['divergence_bits']:.4f}; median inside {j['median_inside']:.3f}, outside {j['median_outside']:.3f})" for j in profiles)+'.')
lines += ['', '## Interpretation and limits', '', 'The leaf-output contrast is approximately 84% larger than the unweighted contrast, and the selected region covers approximately 20% rather than 15% of confirmation observations. Gain weighting produces a much smaller increase. These are different regions with different coverage, so this is not a matched-coverage comparison. The output embedding changes the full partition substantially (ARI approximately 0.46 versus unweighted).', '', 'This supports using leaf outputs for this Brier-based weakness-discovery example. It demonstrates stronger error localization, not improved predictions or model repair. Higher regional Brier also reflects event prevalence and irreducible uncertainty; it is not by itself proof of misspecification. The selected output region has AUC approximately 0.707, while another output cluster has AUC approximately 0.489: the weakest region depends on the chosen metric.', '', 'This is one fitted auxiliary model, one PAM pool and one synthetic generator. No optimizer, LLM proposal or downstream repair was rerun. The existing chapter experiment remains the recorded baseline; this report is a separate executed comparison.', '', '## Reproducibility', '', 'Runner: `book/leaf_geometry_comparison/run.py` (refuses to overwrite its evidence directory). Audit/report: `book/leaf_geometry_comparison/audit_report.py`. Run using `/home/asudjianto/modeva-mcp-venv/bin/python`. `protocol.json`, `frozen_selection.json`, `assignments.npz`, `results.json`, source snapshots and hashes retain the experiment. `audit.json` records a passing replay.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('PASS: audit and report generated at', out)
