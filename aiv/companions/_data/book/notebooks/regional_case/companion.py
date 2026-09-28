"""Read-only regional companion utilities. No fitting or live model calls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOOK = ROOT/'book'
RUN = BOOK/'evidence/aml_medoid_loop_leaf_output_20260921'

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load():
    protocol = read(RUN/'protocol.json')
    results = read(RUN/'results.json')
    frozen = read(RUN/'frozen_finalists.json')
    assert results['frozen_sha256'] == sha(RUN/'frozen_finalists.json')
    for field, file in [('protocol_sha256','protocol.json'), ('admission_sha256','reflection/admission.json'),
                        ('base_sha256','base.joblib'), ('original_auxiliary_sha256','original_auxiliary.json'),
                        ('expanded_auxiliary_sha256','expanded_auxiliary.json')]:
        assert frozen[field] == sha(RUN/file)
    assert read(RUN/'reflection/admission.json')['response_sha256'] == sha(RUN/'reflection/response.json')
    for path, digest in read(RUN/'inputs.json').items():
        assert sha(ROOT/path) == digest
    return protocol, results, frozen

def frontier(records):
    """Maximize contrast and share; keep earliest ID for exact duplicate objectives."""
    ordered = sorted((r for r in records if r['valid']), key=lambda r:(-r['objectives'][0],-r['objectives'][1],r['id']))
    kept=[]; best=-1
    for row in ordered:
        if row['objectives'][1] > best:
            kept.append(row); best=row['objectives'][1]
    return kept

def confirmation():
    with np.load(RUN/'confirmation.npz') as saved:
        loss=(saved['y']-saved['p'])**2
        masks={k:saved[k].copy() for k in saved.files if k not in ['X','y','p']}
    return loss,masks

def measure(loss,mask,z):
    inside,outside=loss[mask].mean(),loss[~mask].mean();share=mask.mean()
    influence=mask*(loss-inside)/share-(~mask)*(loss-outside)/(1-share)
    se=influence.std(ddof=1)/np.sqrt(len(loss));gap=inside-outside
    return dict(n=int(mask.sum()),share=float(share),brier=float(inside),rest_brier=float(outside),
                contrast=float(gap),standard_error=float(se),simultaneous_ci=[float(gap-z*se),float(gap+z*se)]),influence

def assignments(candidate, limit=5000):
    """Replay frozen adapter on an explicit confirmation prefix, never for selection."""
    from xgboost import XGBRegressor
    from scipy.spatial.distance import cdist
    model=XGBRegressor(); model.load_model(RUN/'original_auxiliary.json')
    with np.load(RUN/'confirmation.npz') as saved:
        X=saved['X'][:limit];loss=(saved['y'][:limit]-saved['p'][:limit])**2
        expected=saved[candidate['key']][:limit]
    X=pd.DataFrame(X,columns=model.get_booster().feature_names)
    codes=model.apply(X).astype(int);frame=model.get_booster().trees_to_dataframe()
    values=np.empty(codes.shape,dtype=np.float64)
    for tree in range(codes.shape[1]):
        leaves=frame[(frame.Tree==tree)&(frame.Feature=='Leaf')]
        lookup=np.full(int(leaves.Node.max())+1,np.nan)
        lookup[leaves.Node.to_numpy(dtype=int)]=leaves.Gain.to_numpy()
        values[:,tree]=lookup[codes[:,tree]]
    labels=cdist(values,np.asarray(candidate['medoid_outputs']),metric='euclidean').argmin(1)
    mask=np.isin(labels,candidate['discovery']['region']['cells'])
    np.testing.assert_array_equal(mask,expected)
    return labels,loss

QUALIFIERS = {
    'synthetic':'These measurements concern simulated account-months and do not establish operational AML adequacy.',
    'fixed':'The logistic predictor and auxiliary error model remained frozen; search changed region membership.',
    'conditional':'Intervals are approximate simultaneous 95% normal intervals for 19 claims, conditional on the fitted models and frozen regions.',
    'replication':'Three optimizer seeds share one discovery sample, one LLM proposal and one confirmation sample.',
    'comparison':'All three reflection-minus-continuation intervals include zero; a positive advantage is unresolved.',
    'coverage':'The selected regions differ in coverage; contrast is not predictor repair.',
}

def report_packet():
    from book.parallel_evidence.build import OUT, read as load_json
    store=load_json(OUT/'store.json')
    ids=['region:protocol/confirmation_n']
    for seed in [73101,73102,73103]:
        for arm in ['continuation','reflection']:
            ids += [f'region:confirmation/{arm}_{seed}/{field}' for field in ['brier','share','contrast']]
        ids += [f'region:comparison/reflection-continuation_{seed}/{field}' for field in ['contrast_difference','simultaneous_ci']]
    return store, {'required_facts':ids,'required_qualifiers':list(QUALIFIERS)}

def validate_plan(store,packet,plan):
    from book.parallel_evidence.build import verify
    ids=plan.get('facts',[]);quals=plan.get('qualifiers',[])
    if len(ids)!=len(set(ids)) or len(quals)!=len(set(quals)):
        return False,'Duplicate identifiers'
    if set(ids)!=set(packet['required_facts']) or set(quals)!=set(packet['required_qualifiers']):
        return False,'Missing or unexpected evidence/qualification identifiers'
    for id_ in ids:
        if verify(store,{'id':id_,'value':store['facts'][id_]['value']})['verdict']!='Supported':
            return False,'Unsupported fact'
    return True,'Complete source-checked packet'

def render_report(store,packet,plan):
    valid,reason=validate_plan(store,packet,plan)
    if not valid: raise ValueError(reason)
    lines=['Regional discovery report: deterministic assembly from an authored content plan.','']
    for id_ in plan['facts']:
        record=store['facts'][id_]
        lines.append(f"{id_}: {record['value']} [source: {record['source']}#{record['pointer']}]")
    lines += ['']+[QUALIFIERS[q] for q in plan['qualifiers']]
    return '\n'.join(lines)
