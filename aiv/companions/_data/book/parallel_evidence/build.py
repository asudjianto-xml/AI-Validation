"""Source-bound parallel AML examples; exact supplied claims, not semantic judging."""
from pathlib import Path
import hashlib
import json
import math
import sys

BOOK = Path(__file__).resolve().parents[1]
ROOT = BOOK.parent
OUT = BOOK / 'evidence/parallel_aml_20260921'
REGION = BOOK / 'evidence/aml_medoid_loop_leaf_output_20260921'
MOE = BOOK / 'evidence/aml_end_to_end_current_20260921'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(document, pointer):
    for token in pointer.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def same_type(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b)
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same_type(x, y) for x, y in zip(a, b))
    return isinstance(a, (str, bool)) or a is None


def equal(a, b):
    if not same_type(a, b):
        return False
    if type(a) in (int, float):
        # Counts are exact; displayed continuous measurements allow six-decimal rounding.
        return a == b if type(b) is int else math.isclose(a, b, rel_tol=0, abs_tol=0.5e-6)
    if isinstance(a, list):
        return all(equal(x, y) for x, y in zip(a, b))
    return a == b


def verify(store, claim):
    if not isinstance(claim, dict) or set(claim) != {'id', 'value'} or not isinstance(claim['id'], str):
        return {'verdict': 'Unscorable', 'reason': 'Expected an explicit id and typed value.'}
    record = store['facts'].get(claim['id'])
    if record is None:
        return {'verdict': 'Absent', 'reason': 'No record at this scoped address.'}
    source = ROOT / record['source']
    if sha(source) != record['source_sha256']:
        raise ValueError('Source hash mismatch: ' + record['source'])
    value = resolve(read(source), record['pointer'])
    if type(value) is not type(record['value']) or value != record['value']:
        raise ValueError('Source pointer/value mismatch: ' + claim['id'])
    if not same_type(claim['value'], value):
        return {'verdict': 'Unscorable', 'reason': 'Value type or shape does not match reference.'}
    verdict = 'Supported' if equal(claim['value'], value) else 'Contradicted'
    return {'verdict': verdict, 'reason': 'Exact scoped reference; declared numeric tolerance.', 'evidence': record}


def construct():
    facts = {}
    def add(id_, source, pointer):
        assert id_ not in facts
        facts[id_] = dict(value=resolve(read(source), pointer), source=str(source.relative_to(ROOT)),
                          source_sha256=sha(source), pointer=pointer)
    # Same 50 measurements as the existing mixture projection, with split-qualified identities.
    r = read(MOE / 'results.json')
    for arm, run in r['runs'].items():
        for role, endpoint in run['endpoints'].items():
            for j, field in enumerate(['overall_logloss', 'worst_logloss']):
                add(f'moe:confirmation/{arm}_{role}/{field}', MOE/'results.json', f'/runs/{arm}/endpoints/{role}/confirmation/{j}')
            for j, population in enumerate(r['populations']):
                add(f'moe:confirmation/{arm}_{role}/{population}_logloss', MOE/'results.json', f'/runs/{arm}/endpoints/{role}/group_losses/{j}')
    add('moe:protocol/confirmation_n', MOE/'results.json', '/confirmation_n')
    for key in r['worst_reduction_ci']:
        add('moe:comparison/'+key+'/ci', MOE/'results.json', '/worst_reduction_ci/'+key)
    r = read(REGION/'results.json')
    for candidate, metrics in r['confirmation'].items():
        for field in metrics:
            add(f'region:confirmation/{candidate}/{field}', REGION/'results.json', f'/confirmation/{candidate}/{field}')
    for i, comparison in enumerate(r['comparisons']):
        name = f"{comparison['a']}-{comparison['b']}_{comparison['seed']}"
        for field in ['contrast_difference', 'standard_error', 'simultaneous_ci']:
            add(f'region:comparison/{name}/{field}', REGION/'results.json', f'/comparisons/{i}/{field}')
    for field in ['overall_brier', 'family_size', 'scope']:
        add('region:experiment/'+field, REGION/'results.json', '/'+field)
    for field in ['version', 'geometry', 'base_model', 'auxiliary', 'confirmation_n', 'confirmation_seed', 'primary_policy', 'inference', 'reflection_calls']:
        add('region:protocol/'+field, REGION/'protocol.json', '/'+field)
    frozen = read(REGION/'frozen_finalists.json')
    for i, candidate in enumerate(frozen['finalists']):
        key = candidate['key']
        for field in ['medoids', 'region/cells', 'region/share', 'region/contrast']:
            add(f'region:discovery/{key}/{field}', REGION/'frozen_finalists.json', f'/finalists/{i}/discovery/{field}')
    for field in ['admitted', 'parent', 'version', 'reason', 'response_sha256']:
        add('region:admission/'+field, REGION/'reflection/admission.json', '/'+field)
    assert r['frozen_sha256'] == sha(REGION/'frozen_finalists.json')
    assert frozen['protocol_sha256'] == sha(REGION/'protocol.json')
    assert frozen['admission_sha256'] == sha(REGION/'reflection/admission.json')
    assert read(REGION/'reflection/admission.json')['response_sha256'] == sha(REGION/'reflection/response.json')
    return {'version': 'parallel-aml-exact-v1', 'facts': facts,
            'scope': 'Saved synthetic experiments; exact supplied mappings only. No free-text extraction or live reporting assessment.'}


def cases():
    return [
        ('moe_correct', 'moe:confirmation/llm_compromise/worst_logloss', .103333, 'Supported'),
        ('moe_false', 'moe:confirmation/llm_compromise/worst_logloss', .4, 'Contradicted'),
        ('moe_missing', 'moe:experiment/fairness_passed', True, 'Absent'),
        ('regional_loss', 'region:confirmation/reflection_73101/brier', .072851, 'Supported'),
        ('regional_contrast', 'region:confirmation/reflection_73101/contrast', .058326, 'Supported'),
        ('regional_share', 'region:confirmation/reflection_73101/share', .115220, 'Supported'),
        ('regional_count', 'region:protocol/confirmation_n', 200000, 'Supported'),
        ('admission', 'region:admission/admitted', True, 'Supported'),
        ('paired_interval', 'region:comparison/reflection-continuation_73101/simultaneous_ci', [-.001723, .002495], 'Supported'),
        ('wrong_metric', 'region:confirmation/reflection_73101/brier', .058326, 'Contradicted'),
        ('wrong_split', 'region:discovery/reflection_73101/region/contrast', .058326, 'Contradicted'),
        ('wrong_study', 'moe:confirmation/llm_compromise/worst_logloss', .058326, 'Contradicted'),
        ('unsupported_repair', 'region:experiment/predictor_repair_gain', .01, 'Absent'),
        ('unsupported_cause', 'region:experiment/fragmentation_cause_confirmed', True, 'Absent'),
        ('numeric_string', 'region:confirmation/reflection_73101/contrast', '0.058326', 'Unscorable'),
        ('boolean_count', 'region:protocol/confirmation_n', True, 'Unscorable'),
        ('interval_shape', 'region:comparison/reflection-continuation_73101/simultaneous_ci', [0], 'Unscorable'),
    ]


def artifacts():
    store = construct()
    verdicts = []
    for name, id_, value, expected in cases():
        claim = {'id': id_, 'value': value}
        result = verify(store, claim)
        assert result['verdict'] == expected, (name, result)
        verdicts.append(dict(case=name, claim=claim, expected=expected, **result))
    # Detect corrupted sources and bindings separately from ordinary missing evidence.
    from copy import deepcopy
    for field, value in [('source_sha256', 'invalid'), ('value', -1.0)]:
        corrupted = deepcopy(store)
        key = 'region:confirmation/reflection_73101/contrast'
        corrupted['facts'][key][field] = value
        try:
            verify(corrupted, {'id': key, 'value': .058326})
        except ValueError:
            pass
        else:
            raise AssertionError('Corruption was not rejected')
    summary = dict(facts_by_study={s:sum(k.startswith(s+':') for k in store['facts']) for s in ['moe','region']},
                   cases=len(verdicts), verdict_counts={v:sum(c['verdict']==v for c in verdicts) for v in ['Supported','Contradicted','Absent','Unscorable']}, corruption_checks=2)
    output = '\n'.join(f"{v['verdict']}: {v['case']}" for v in verdicts)+'\n'
    return {'store.json':store, 'claim_verdicts.json':verdicts, 'summary.json':summary, 'claim_output.txt':output}


def main(check=False):
    generated = artifacts()
    if not check:
        OUT.mkdir(parents=True, exist_ok=True)
    for name, value in generated.items():
        text = value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
        if check:
            assert (OUT/name).read_text() == text, 'Stale projection: '+name
        else:
            (OUT/name).write_text(text)
    print(json.dumps(generated['summary.json'], indent=2))

if __name__ == '__main__':
    main('--check' in sys.argv)
