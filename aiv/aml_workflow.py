"""Scoped, source-checked AML evidence shared by Chapters 5, 6, 8 and 9.

No model refitting, fuzzy entity resolution, geometric truth gate or semantic
accuracy claim. JSON pointers are the bindings; stored hashes detect source drift.
"""
from pathlib import Path
import hashlib
import json
import math
import os
from numbers import Real

# The studies below are historical repository evidence and are not installed with
# the package. MODEL_VALIDATION_ROOT names the checkout; the default is the
# checkout containing this module. Missing sources raise FileNotFoundError.
ROOT = Path(os.environ.get('MODEL_VALIDATION_ROOT') or Path(__file__).resolve().parents[1]).expanduser()
OUT = ROOT / 'book/evidence/aml_workflow'
STUDIES = {
    'representation': 'paper/aml_end_to_end_20260919',
    'optimizers': 'paper/aml_four_optimizers_20260919',
    'splitting': 'paper/aml_twinning_splits_20260919',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding="utf-8")


def resolve(data, pointer):
    for part in pointer.lstrip('/').split('/') if pointer else []:
        part = part.replace('~1', '/').replace('~0', '~')
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def fact_id(study, pointer):
    return study + ':' + pointer


def build_store(root=ROOT):
    facts = []
    sources = {}
    def add(study, filename, pointer, kind='measurement'):
        source = str(Path(STUDIES[study]) / filename)
        data = read(root / source)
        sources[source] = digest(root / source)
        facts.append(dict(id=fact_id(study, pointer) if filename == 'results.json'
                          else study + ':' + filename + '#' + pointer,
                          study=study, kind=kind, value=resolve(data, pointer),
                          source=source, pointer=pointer, source_sha256=sources[source]))
    for study, directory in STUDIES.items():
        data = read(root / directory / 'results.json')
        if study == 'representation':
            for arm, run in data['runs'].items():
                add(study, 'results.json', f'/runs/{arm}/gate_features', 'specification')
                for role in run['endpoints']:
                    for field in ('confirmation', 'group_losses'):
                        for i in range(len(run['endpoints'][role][field])):
                            add(study, 'results.json', f'/runs/{arm}/endpoints/{role}/{field}/{i}')
            for key in data['worst_reduction_ci']:
                add(study, 'results.json', '/worst_reduction_ci/' + key, 'conditional_interval')
            for filename, pointer, kind in [
                ('reflection/admission.json', '/admitted', 'admission'),
                ('reflection/admission.json', '/gate_features', 'admission'),
                ('reflection/admission.json', '/response_sha256', 'provenance')]:
                add(study, filename, pointer, kind)
        elif study == 'optimizers':
            for method, aggregate in data['aggregates'].items():
                for role in ('mean', 'robust', 'compromise'):
                    for i in range(2):
                        add(study, 'results.json', f'/aggregates/{method}/{role}/mean/{i}')
            add(study, 'results.json', '/comparisons', 'conditional_intervals')
            for key in ('charged_fits', 'distinct_fits'):
                add(study, 'results.json', '/' + key, 'design')
        else:
            for method, aggregate in data['summary'].items():
                for key in ('test_logloss', 'reference_logloss', 'test_minus_reference'):
                    for stat in ('mean', 'sd'):
                        add(study, 'results.json', f'/summary/{method}/{key}/{stat}')
                add(study, 'results.json', f'/summary/{method}/estimation_rmse')
            for key in ('n_pool', 'n_train', 'n_test', 'n_reference', 'replications'):
                add(study, 'results.json', '/' + key, 'design')
        if 'confirmation_n' in data:
            add(study, 'results.json', '/confirmation_n', 'design')
    # Retain proposal, admission and selection artifacts as different graph nodes.
    for study, directory in STUDIES.items():
        names = ['results.json']
        names += (['frozen_selections.json', 'reflection/prompt.txt',
                   'reflection/response.json', 'reflection/admission.json', 'audit.json']
                  if study == 'representation' else
                  ['frozen_selections.json', 'prior_admission.json', 'confirmation_audit.json']
                  if study == 'optimizers' else ['model_freeze.json', 'audit.json'])
        for name in names:
            path = root / directory / name
            if not path.exists():
                raise FileNotFoundError(path)
            sources[path.relative_to(root).as_posix()] = digest(path)
    graph = [{'head': f['id'], 'relation': 'asserted_at',
              'tail': f['source'] + '#' + f['pointer']} for f in facts]
    return dict(schema_version=1, facts=facts, sources=sources, graph=graph,
                scope='Balanced synthetic AML; studies and selection roles remain separate.')


def audit_store(store, root=ROOT):
    for source, expected in store['sources'].items():
        if digest(root / source) != expected:
            raise ValueError('Source drift: ' + source)
    documents = {f['source']: read(root / f['source']) for f in store['facts']}
    ids = set()
    for f in store['facts']:
        prefix = STUDIES.get(f['study'], '') + '/'
        if f['study'] not in STUDIES or not f['source'].startswith(prefix):
            raise ValueError('Study/source scope mismatch')
        filename = f['source'][len(prefix):]
        expected_id = (fact_id(f['study'], f['pointer']) if filename == 'results.json'
                       else f['study'] + ':' + filename + '#' + f['pointer'])
        if f['id'] != expected_id:
            raise ValueError('Identifier/source scope mismatch')
        if f['id'] in ids or resolve(documents[f['source']], f['pointer']) != f['value']:
            raise ValueError('Invalid or duplicate binding: ' + f['id'])
        if f['source_sha256'] != store['sources'][f['source']]:
            raise ValueError('Invalid source hash: ' + f['id'])
        ids.add(f['id'])
    return dict(facts=len(ids), sources=len(store['sources']), all_bindings_checked=True)


def equal(a, b):
    # A boolean is never a numeric measurement, despite bool subclassing int.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, Real) and isinstance(b, Real):
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 0.5e-6
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return a == b


def verify(store, claim, root=ROOT):
    """Structured binding only. Corrupt evidence aborts; malformed claims are unscorable."""
    audit_store(store, root)
    if not isinstance(claim, dict) or set(claim) != {'id', 'value'} or not isinstance(claim['id'], str):
        return dict(verdict='Unscorable', reason='Expected an explicit scoped id and typed value')
    try:
        json.dumps(claim, allow_nan=False)
    except (ValueError, TypeError):
        return dict(verdict='Unscorable', reason='Non-JSON or nonfinite value')
    f = next((f for f in store['facts'] if f['id'] == claim['id']), None)
    if f is None:
        return dict(verdict='Absent', reason='No assertion at this exact scope')
    return dict(verdict='Supported' if equal(claim['value'], f['value']) else 'Contradicted',
                evidence=f)


def verification_cases():
    p = 'representation:/runs/llm/endpoints/compromise/'
    return [
        ('rounded scalar', {'id': p+'confirmation/1', 'value': .597821}, 'Supported'),
        ('wrong value', {'id': p+'confirmation/1', 'value': .4}, 'Contradicted'),
        ('different selection rule', {'id': p.replace('compromise', 'mean')+'confirmation/1', 'value': .597821}, 'Contradicted'),
        ('invented discovery binding', {'id': p+'discovery/1', 'value': .597821}, 'Absent'),
        ('fairness not measured', {'id': 'representation:/fairness_passed', 'value': True}, 'Absent'),
        ('numeric string', {'id': p+'confirmation/1', 'value': '0.597821'}, 'Contradicted'),
        ('wrong study', {'id': 'optimizers:/runs/llm/endpoints/compromise/confirmation/1', 'value': .597821}, 'Absent'),
        ('boolean versus sample size', {'id': 'representation:/confirmation_n', 'value': True}, 'Contradicted'),
        ('confirmation count', {'id': 'representation:/confirmation_n', 'value': 40000}, 'Supported'),
        ('admitted is not confirmed', {'id': 'representation:reflection/admission.json#/admitted', 'value': True}, 'Supported'),
        ('missing binding', {'value': .597821}, 'Unscorable'),
        ('nonfinite measurement', {'id': p+'confirmation/1', 'value': None}, 'Contradicted'),
    ]


def run_verification(store):
    rows = [dict(case=name, claim=claim, expected=expected, **verify(store, claim))
            for name, claim, expected in verification_cases()]
    return dict(cases=rows, passed=sum(r['verdict'] == r['expected'] for r in rows),
                scope='Twelve specified regression cases; not a semantic accuracy estimate.')


def findings(store):
    audit_store(store)
    r, o, t = [read(ROOT / d / 'results.json') for d in STUDIES.values()]
    # Link each published result back to its pre-confirmation freeze record.
    for study, result in [('representation', r), ('optimizers', o)]:
        path = ROOT / STUDIES[study] / 'frozen_selections.json'
        if digest(path) != result['frozen_sha256']:
            raise ValueError('Selection freeze mismatch')
    if digest(ROOT / STUDIES['splitting'] / 'model_freeze.json') != t['model_freeze_sha256']:
        raise ValueError('Splitting model freeze mismatch')
    admission = read(ROOT / STUDIES['representation'] / 'reflection/admission.json')
    if digest(ROOT / STUDIES['representation'] / 'reflection/response.json') != admission['response_sha256']:
        raise ValueError('Proposal/admission mismatch')
    e = r['runs']['llm']['endpoints']
    sd = t['summary']
    return dict(
        representation=dict(reduction=e['mean']['confirmation'][1]-e['compromise']['confirmation'][1],
            conditional_ci=r['worst_reduction_ci']['llm:mean->llm:compromise'],
            scope='Frozen models and populations, paired stratified row bootstrap; no search rerun.',
            full_control_worst=r['runs']['full']['endpoints']['compromise']['confirmation'][1]),
        optimizers=dict(comparisons=o['comparisons'],
            scope='Five paired searches per method; conditional intervals do not establish a winner.'),
        splitting=dict(test_variance_reduction=1-(sd['twin_xy']['test_logloss']['sd']/sd['random']['test_logloss']['sd'])**2,
            random_rmse=sd['random']['estimation_rmse'], joint_rmse=sd['twin_xy']['estimation_rmse'],
            scope='One shared observation pool; lower test variance is not independent confirmation.'),
        materiality='No operational AML acceptance threshold or fairness assessment supplied.',
        freeze_hashes={k: store['sources'][STUDIES[k] + '/' + ('model_freeze.json' if k == 'splitting' else 'frozen_selections.json')] for k in STUDIES})


def packets(store):
    """Explicit report requirements. Qualifiers are authored scope obligations, not measurements."""
    ids = {f['id'] for f in store['facts']}
    definitions = [
        ('tradeoff', 'discovery', 'representation', [
            '/runs/llm/endpoints/mean/confirmation/0', '/runs/llm/endpoints/mean/confirmation/1',
            '/runs/llm/endpoints/compromise/confirmation/0', '/runs/llm/endpoints/compromise/confirmation/1'],
         ['Balanced synthetic challenge; not operational AML prevalence.',
          'Endpoints selected on discovery and frozen before confirmation.']),
        ('control', 'discovery', 'representation', [
            '/runs/llm/endpoints/compromise/confirmation/1', '/runs/full/endpoints/compromise/confirmation/1'],
         ['The full-feature control performs better; no best-representation claim.',
          'No fairness assessment or operational materiality threshold is supplied.']),
        ('interval', 'confirmation', 'representation', [
            '/worst_reduction_ci/llm:mean->llm:compromise', '/confirmation_n'],
         ['Paired stratified bootstrap conditional on the frozen models and populations.',
          'The interval does not include uncertainty from repeating discovery.']),
        ('optimizers', 'confirmation', 'optimizers', [
            '/aggregates/'+m+'/robust/mean/1' for m in ('hill','bandit','es','bayesian')],
         ['Averages of five selected models per method, not ensemble predictions.',
          'All six conditional pairwise intervals include zero; no optimizer winner established.']),
        ('twinning', 'confirmation', 'splitting', [
            '/summary/'+m+'/'+k for m in ('random','twin_xy') for k in ('test_logloss/sd','estimation_rmse')],
         ['Thirty splits of one shared pool, with a separately generated reference sample.',
          'Joint twinning lowers test-score variance but increases optimism and estimation RMSE here.']),
    ]
    result=[]
    for name, split, study, paths, qualifiers in definitions:
        required=[fact_id(study,p) for p in paths]
        assert set(required) <= ids
        result.append(dict(id=name, split=split, study=study, required=required,
                           qualifiers={name+':q'+str(i):q for i,q in enumerate(qualifiers)}))
    return result


def evaluate_plan(store, packet, plan):
    """Score an LLM's structured content plan, not the semantics of free prose."""
    if not isinstance(plan, dict) or set(plan) != {'packet','facts','qualifiers'} or plan.get('packet') != packet['id']:
        return dict(valid=False, coverage=0., qualifier_coverage=0., cost=0, reason='Malformed plan')
    fs, qs = plan['facts'], plan['qualifiers']
    if not isinstance(fs,list) or not isinstance(qs,list) or not all(isinstance(x,str) for x in fs+qs):
        return dict(valid=False, coverage=0., qualifier_coverage=0., cost=0, reason='Malformed identifiers')
    valid = (len(set(fs)) == len(fs) and len(set(qs)) == len(qs)
             and set(fs) <= set(packet['required']) and set(qs) <= set(packet['qualifiers']))
    coverage=len(set(fs)&set(packet['required']))/len(packet['required'])
    qc=len(set(qs)&set(packet['qualifiers']))/len(packet['qualifiers'])
    return dict(valid=valid, coverage=coverage, qualifier_coverage=qc, cost=len(fs)+len(qs),
                admissible=valid and coverage==1 and qc==1)


def render(store, packet, plan):
    score=evaluate_plan(store,packet,plan)
    if not score.get('admissible'):
        raise ValueError('Incomplete or unsupported report plan; abstain')
    index={f['id']:f for f in store['facts']}
    lines=[packet['id'].upper()]
    for fid in plan['facts']:
        f=index[fid]
        lines.append(f"{fid} = {json.dumps(f['value'])} [{f['source']}#{f['pointer']}]")
    lines.extend(packet['qualifiers'][q] for q in plan['qualifiers'])
    return '\n'.join(lines)


def build():
    store=build_store(); summary=audit_store(store)
    save(OUT/'store.json',store);save(OUT/'verification.json',run_verification(store))
    save(OUT/'findings.json',findings(store));save(OUT/'packets.json',packets(store))
    save(OUT/'summary.json',summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(build(),indent=2))
