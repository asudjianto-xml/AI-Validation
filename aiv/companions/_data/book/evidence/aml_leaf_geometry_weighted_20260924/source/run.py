"""Frozen-model comparison of three geometries; no outcome-driven tuning."""
from pathlib import Path
import sys,json,hashlib,shutil,time,platform
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist,jensenshannon
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import adjusted_rand_score,roc_auc_score
from sklearn_extra.cluster import KMedoids
from xgboost import XGBRegressor
BOOK=Path(__file__).resolve().parents[1]
REPO=BOOK.parent
PRIOR=BOOK/'evidence/aml_error_clustering_current_20260921'
OUT=BOOK/'evidence/aml_leaf_geometry_weighted_20260924'
sys.path.insert(0,str(BOOK/'error_clustering_current'))
from run import generate,leaf_distance
from aml_capacity_data import FEATURES
sys.path.insert(0,str(BOOK))
from weighted_leaf_geometry import weighted_leaf_embedding
ARMS=['equal_leaf','gain_leaf','leaf_output']
PROTOCOL=dict(version='leaf-geometry-weighted-v2',prior=str(PRIOR.relative_to(REPO)),
    training_n=40000,discovery_n=40000,models='exact frozen Chapter 2 logistic and 120 depth-two error trees',
    arms=dict(equal_leaf='one minus fraction of trees sharing a leaf; float32 original PAM matrix',
              gain_leaf='one minus shared-leaf agreement weighted by sum of internal-node split gains per tree',
              leaf_output='Euclidean distance in tree-leaf indicator coordinates weighted by fitted leaf values; shared-leaf kernel weights are squared values; no normalization or extra shrinkage'),
    pam_pool='same saved 3000 discovery indices in same order',k=4,pam_init='build',pam_max_iter=300,pam_random_state=62105,
    selection='largest mean discovery Brier among clusters with n>=50 and share>=0.01; lowest cluster ID breaks ties',
    assignment_tie='lowest medoid position',confirmation_n=200000,confirmation_seed=62624,
    primary='leaf_output minus equal_leaf selected-region Brier contrast versus rest',
    secondary='gain_leaf minus equal_leaf selected-region Brier contrast versus rest',
    inference='5-claim Bonferroni simultaneous 95% normal intervals: three selected contrasts and two paired differences; conditional on frozen fits and selections',
    descriptive=['all cluster sizes, Brier, event rates and AUC','adjusted Rand index of complete partitions','selected-region Jaccard','discovery feature JS divergence'],
    scope='one fixed data draw, one fixed PAM pool and deterministic initialization; different region sizes; no universal geometry ranking',
    restrictions=['no refitting base or auxiliary','no per-tree coordinate standardization','no confirmation-based model or region reselection','no numerical medoid optimization or LLM call'])
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree_values(model):
    tables=[];gain=[]
    for raw in model.get_booster().get_dump(dump_format='json',with_stats=True):
        leaves={};g=[]
        def visit(node):
            if 'leaf' in node:leaves[node['nodeid']]=node['leaf']
            else:
                g.append(float(node['gain']))
                for child in node['children']:visit(child)
        visit(json.loads(raw));tables.append(leaves);gain.append(sum(g))
    gain=np.asarray(gain);assert (gain>=0).all() and gain.sum()>0
    return tables,gain/gain.sum()
def output_embedding(codes,tables):
    # Historical arm/API name retained; v2 preserves a coordinate per tree-leaf.
    return weighted_leaf_embedding(codes,tables)
def pool_distances(codes,z,weights):
    H=OneHotEncoder(sparse_output=True,dtype=np.float32).fit_transform(codes)
    equal=np.clip(1-(H@H.T).toarray()/codes.shape[1],0,1);np.fill_diagonal(equal,0)
    wcols=np.concatenate([np.repeat(weights[t],len(np.unique(codes[:,t]))) for t in range(codes.shape[1])])
    Hw=H.astype(np.float64).multiply(np.sqrt(wcols)).tocsr()
    gain=np.clip(1-(Hw@Hw.T).toarray(),0,1);np.fill_diagonal(gain,0)
    outputs=cdist(z,z,metric='euclidean');np.fill_diagonal(outputs,0)
    return dict(equal_leaf=equal,gain_leaf=gain,leaf_output=outputs)
def assign(arm,codes,z,mc,mz,weights):
    if arm=='equal_leaf':return leaf_distance(codes,mc).argmin(1)
    if arm=='leaf_output':return cdist(z,mz,metric='euclidean').argmin(1)
    result=np.empty(len(codes),int)
    for start in range(0,len(codes),1000):
        d=np.sum((codes[start:start+1000,None,:]!=mc[None,:,:])*weights,axis=2)
        result[start:start+1000]=d.argmin(1)
    return result
def metrics(y,p,labels):
    loss=(y-p)**2;rows=[]
    for k in range(4):
        mask=labels==k;n=int(mask.sum());a=loss[mask];b=loss[~mask]
        rows.append(dict(cluster=k,n=n,share=float(mask.mean()),brier=float(a.mean()),rest_brier=float(b.mean()),contrast=float(a.mean()-b.mean()),
            prevalence=float(y[mask].mean()),auc=float(roc_auc_score(y[mask],p[mask])) if len(np.unique(y[mask]))==2 else None))
    return dict(n=len(y),overall_brier=float(loss.mean()),overall_prevalence=float(y.mean()),clusters=rows)
def selected_stats(loss,mask,z):
    a,b=loss[mask],loss[~mask];q=mask.mean();gap=float(a.mean()-b.mean())
    influence=mask*(loss-a.mean())/q-(~mask)*(loss-b.mean())/(1-q)
    se=float(influence.std(ddof=1)/np.sqrt(len(loss)))
    return dict(contrast=gap,se=se,simultaneous_ci=[gap-z*se,gap+z*se]),influence
def js_profiles(X,labels,worst):
    mask=labels==worst;rows=[]
    for f in FEATURES:
        edges=np.r_[-np.inf,np.unique(np.quantile(X[f],np.arange(.1,1,.1))),np.inf]
        a=np.histogram(X.loc[mask,f],edges)[0];b=np.histogram(X.loc[~mask,f],edges)[0]
        rows.append(dict(feature=f,divergence_bits=float(jensenshannon(a,b,base=2)**2),median_inside=float(X.loc[mask,f].median()),median_outside=float(X.loc[~mask,f].median()),edges=edges[1:-1].tolist()))
    return sorted(rows,key=lambda r:-r['divergence_bits'])
def main():
    OUT.mkdir(exist_ok=False);save(OUT/'protocol.json',PROTOCOL)
    (OUT/'source').mkdir()
    shutil.copy2(BOOK/'weighted_leaf_geometry.py',OUT/'source/weighted_leaf_geometry.py')
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    for p in (PRIOR/'source').glob('*.py'):shutil.copy2(p,OUT/'source'/('prior_'+p.name))
    save(OUT/'inputs.json',{str(p.relative_to(REPO)):sha(p) for p in [PRIOR/'observations.npz',PRIOR/'base.joblib',PRIOR/'residual_model.json',PRIOR/'frozen_selection.json']})
    shutil.copy2(PRIOR/'base.joblib',OUT/'base.joblib');shutil.copy2(PRIOR/'residual_model.json',OUT/'residual_model.json')
    a=np.load(PRIOR/'observations.npz');X=pd.DataFrame(a['X_discovery'],columns=FEATURES);y=a['y_discovery'];p=a['p_discovery'];codes=a['leaf_discovery'].astype(int)
    model=XGBRegressor();model.load_model(OUT/'residual_model.json')
    np.testing.assert_array_equal(codes,model.apply(X).astype(int));tables,weights=tree_values(model);Z=output_embedding(codes,tables)
    base_score=np.asarray(json.loads(model.get_booster().save_config())['learner']['learner_model_param']['base_score'].strip('[]'),dtype=float).item()
    reconstruction=float(np.max(abs(Z.sum(1)+base_score-model.predict(X))))
    assert reconstruction<5e-7,reconstruction
    prior=read(PRIOR/'frozen_selection.json');pool=np.array(prior['pam_sample_indices']);dist=pool_distances(codes[pool],Z[pool],weights)
    selections={};labels={};timings={}
    for arm in ARMS:
        start=time.perf_counter();pam=KMedoids(n_clusters=4,metric='precomputed',method='pam',init='build',max_iter=300,random_state=62105).fit(dist[arm])
        assert pam.n_iter_<299
        medoids=pool[pam.medoid_indices_];labels[arm]=assign(arm,codes,Z,codes[medoids],Z[medoids],weights)
        np.testing.assert_array_equal(labels[arm][pool],pam.labels_)
        if arm=='equal_leaf':
            np.testing.assert_array_equal(medoids,prior['medoid_indices']);np.testing.assert_array_equal(labels[arm],a['cluster_discovery'])
        m=metrics(y,p,labels[arm]);eligible=[r for r in m['clusters'] if r['n']>=50 and r['share']>=.01];worst=max(eligible,key=lambda r:(r['brier'],-r['cluster']))['cluster']
        selections[arm]=dict(medoid_indices=medoids.tolist(),worst_cluster=worst,discovery=m,js=js_profiles(X,labels[arm],worst))
        timings[arm]=dict(pam_and_assignment_seconds=time.perf_counter()-start,iterations=int(pam.n_iter_),within_geometry_inertia=float(pam.inertia_),distance_bytes=dist[arm].nbytes)
        print(arm,'selected',m['clusters'][worst],flush=True)
    save(OUT/'frozen_selection.json',dict(arms=selections,gain_weights=weights.tolist(),leaf_values=[{str(k):v for k,v in t.items()} for t in tables],pam_pool=pool.tolist(),base_score=base_score,protocol_sha256=sha(OUT/'protocol.json'),base_sha256=sha(OUT/'base.joblib'),auxiliary_sha256=sha(OUT/'residual_model.json')))
    freeze_hash=sha(OUT/'frozen_selection.json')
    # All three choices freeze together before new observations/outcomes are generated.
    Xc,yc,gc=generate(PROTOCOL['confirmation_n'],PROTOCOL['confirmation_seed']);pc=joblib.load(OUT/'base.joblib').predict_proba(Xc)[:,1]
    cc=model.apply(Xc).astype(int);Zc=output_embedding(cc,tables)
    assert np.max(abs(Zc.sum(1)+base_score-model.predict(Xc)))<5e-7
    z=float(norm.ppf(1-.05/(2*5)));confirmation={};influences={};clabels={};masks={}
    for arm in ARMS:
        medoids=selections[arm]['medoid_indices'];lab=assign(arm,cc,Zc,codes[medoids],Z[medoids],weights);clabels[arm]=lab
        mask=lab==selections[arm]['worst_cluster'];masks[arm]=mask
        stats,inf=selected_stats((yc-pc)**2,mask,z);influences[arm]=inf
        confirmation[arm]=dict(**metrics(yc,pc,lab),selected=stats)
    comparisons=[]
    for arm in ARMS[1:]:
        gap=confirmation[arm]['selected']['contrast']-confirmation['equal_leaf']['selected']['contrast'];se=float((influences[arm]-influences['equal_leaf']).std(ddof=1)/np.sqrt(len(yc)))
        comparisons.append(dict(arm=arm,reference='equal_leaf',contrast_difference=gap,se=se,simultaneous_ci=[gap-z*se,gap+z*se],selected_jaccard=float((masks[arm]&masks['equal_leaf']).sum()/(masks[arm]|masks['equal_leaf']).sum())))
    ari={split:{a:{b:float(adjusted_rand_score(labs[a],labs[b])) for b in ARMS} for a in ARMS} for split,labs in [('discovery',labels),('confirmation',clabels)]}
    np.savez_compressed(OUT/'assignments.npz',X_confirmation=Xc.to_numpy(),y_confirmation=yc,group_confirmation=gc,p_confirmation=pc,**{'discovery_'+a:l for a,l in labels.items()},**{'confirmation_'+a:l for a,l in clabels.items()})
    result=dict(protocol=PROTOCOL,arms=selections,confirmation=confirmation,comparisons=comparisons,ari=ari,timings=timings,reconstruction_error=reconstruction,
        tree_weight_diagnostics=dict(gain_first_30=float(weights[:30].sum()),gain_last_30=float(weights[-30:].sum()),weighted_leaf_variance_first_30=float(Z.var(0)[:sum(map(len,tables[:30]))].sum()/Z.var(0).sum()),weighted_leaf_variance_last_30=float(Z.var(0)[sum(map(len,tables[:-30])):].sum()/Z.var(0).sum())),
        frozen_sha256=freeze_hash,critical_value=z,family_size=5,versions=dict(python=platform.python_version(),numpy=np.__version__))
    assert sha(OUT/'frozen_selection.json')==freeze_hash;save(OUT/'results.json',result)
    save(OUT/'manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    print('Confirmation comparisons:',json.dumps(comparisons),flush=True)
if __name__=='__main__':main()
