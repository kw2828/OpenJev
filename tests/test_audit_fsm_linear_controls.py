"""Fabricated saved-evidence checks; no model, fitting or data calls."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('audit_fsm_linear_controls', SCRIPTS/'audit_fsm_linear_controls.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def affine_state():
    backbone = np.arange(588, dtype=np.float64).reshape(3, 196)/1000
    state = {'coefficients': torch.from_numpy(backbone.copy()),
             'residual.weight': torch.full((3, 195), .25, dtype=torch.float64),
             'residual.bias': torch.tensor([1., -2., 3.], dtype=torch.float64)}
    return state, backbone


def test_direct_fold_algebra_and_no_mutation():
    state, backbone = affine_state()
    original = backbone.copy()
    merged = audit.folded_coefficients(state, backbone)
    np.testing.assert_array_equal(merged[:, :-1], original[:, :-1]+.25)
    np.testing.assert_array_equal(merged[:, -1], original[:, -1]+[1., -2., 3.])
    np.testing.assert_array_equal(backbone, original)
    np.testing.assert_array_equal(state['coefficients'].numpy(), original)
    assert not np.shares_memory(merged, backbone)


@pytest.mark.parametrize('mutation', ['backbone', 'extra', 'shape', 'dtype', 'nan', 'overflow'])
def test_fold_certificate_rejects_invalid_checkpoint(mutation):
    state, backbone = affine_state()
    if mutation == 'backbone':
        state['coefficients'][0, 0] = -1.
    elif mutation == 'extra':
        state['hidden_cache'] = torch.zeros(1)
    elif mutation == 'shape':
        state['residual.weight'] = torch.zeros((3, 196), dtype=torch.float64)
    elif mutation == 'dtype':
        state['residual.bias'] = state['residual.bias'].float()
    elif mutation == 'nan':
        state['residual.bias'][0] = float('nan')
    else:
        backbone.fill(np.finfo(float).max)
        state['coefficients'] = torch.from_numpy(backbone.copy())
        state['residual.weight'].fill_(np.finfo(float).max)
    with pytest.raises(ValueError):
        audit.folded_coefficients(state, backbone)


def test_metrics_native_units_and_corruption():
    prediction = np.broadcast_to(np.array([1., 2., 3.]), (32, 128, 3)).copy()
    target = np.zeros_like(prediction)
    result = audit.record_metrics(prediction, target, np.array([2., 3., 4.]))
    assert result['mse'] == 14/3
    assert result['per_channel_rmse'] == [1., 2., 3.]
    assert result['native_output_per_channel_rmse'] == [2., 6., 12.]
    prediction[0, 0, 0] = 10.
    with pytest.raises(ValueError):
        audit.close(audit.record_metrics(prediction, target, np.array([2., 3., 4.])), result)


def test_saved_prediction_parity_preserves_failure():
    original = np.ones((32, 128, 3))
    folded = original.copy()
    folded[0, 0, 0] += 1e-12
    assert audit.prediction_parity(folded, original, rtol=1e-9, atol=1e-9)['passed']
    folded[0, 0, 0] += 1e-5
    record = audit.prediction_parity(folded, original, rtol=1e-9, atol=1e-9)
    assert record['passed'] is False and record['max_abs_difference'] > 1e-5
    folded[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        audit.prediction_parity(folded, original, rtol=1e-9, atol=1e-9)


def evidence():
    fits = [{'family': f, 'status': 'complete'} for f in audit.LINEAR]
    folds = [{'seed': s, 'status': 'complete', 'parity': {'passed': True}} for s in audit.SEEDS]
    evaluations = []
    for family in audit.FAMILIES:
        for seed in (None,) if family in (*audit.LINEAR, audit.NATIVE) else audit.SEEDS:
            evaluations.append({'family': family, 'seed': seed,
                                'rows': [{'record_id': r, 'status': 'complete', 'rmse': 8. if family == audit.CANDIDATE else 10.} for r in audit.DEV],
                                'timing_error': None, 'request_ms': [1.]*24,
                                'median_request_ms': 1., 'persistent_numeric_bytes': 100})
    return evaluations, fits, folds


def set_score(evaluations, family, value, *, seed=None, record=None):
    for e in evaluations:
        if e['family'] == family and (seed is None or e['seed'] == seed):
            for row in e['rows']:
                if record is None or row['record_id'] == record:
                    row['rmse'] = value


def test_all_25_slots_15_families_nine_rules_and_fixed_candidate():
    evaluations, fits, folds = evidence()
    result = audit.decisions(evaluations, fits, folds)
    assert len(evaluations) == 25 and len(fits) == 9 and len(folds) == 3
    assert len(result['families']) == 15 and sum(len(e['rows']) for e in evaluations) == 300
    assert result['passed'] == result['total'] == 9
    assert result['candidate'] == 'tanh_feedback-lr0.0003'
    assert result['strongest_control'] == 'affine_feedback-lr0.0001'


def test_strongest_reference_is_global_and_score_only():
    evaluations, fits, folds = evidence()
    set_score(evaluations, 'varx96-ridge0.001', 7.)
    reference = next(e for e in evaluations if e['family'] == 'varx96-ridge0.001')
    reference.update(timing_error='failed timer', median_request_ms=None, request_ms=[])
    result = audit.decisions(evaluations, fits, folds)
    assert result['strongest_control'] == 'varx96-ridge0.001'
    assert result['conditions']['five_percent_below_strongest_control'] is False
    assert result['conditions']['all_25_evaluations_complete'] is False
    assert result['candidate'] == audit.CANDIDATE


@pytest.mark.parametrize('mutation,condition', [
    ('fit_failed', 'all_9_fits_complete'), ('missing', 'all_25_evaluations_complete'),
    ('duplicate', 'all_25_evaluations_complete'), ('fold_failed', 'all_3_folds_equivalent'),
    ('timing', 'latency_within_ten_percent_tanh_output'), ('storage', 'storage_no_more_than_tanh_output'),
    ('worst_record', 'no_record_over_two_percent_strongest_control'),
    ('amplitude', 'both_amplitudes_below_strongest_control'), ('seed', 'every_seed_below_strongest_control'),
    ('nonfinite', 'all_25_evaluations_complete'),
])
def test_declared_failures_and_omissions_cannot_disappear(mutation, condition):
    evaluations, fits, folds = evidence()
    candidates = [e for e in evaluations if e['family'] == audit.CANDIDATE]
    if mutation == 'fit_failed':
        fits[0]['status'] = 'failed'
    elif mutation == 'missing':
        evaluations.pop()
    elif mutation == 'duplicate':
        candidates[1]['seed'] = candidates[0]['seed']
    elif mutation == 'fold_failed':
        folds[0]['parity']['passed'] = False
    elif mutation == 'timing':
        for e in candidates: e['median_request_ms'] = 1.100001
    elif mutation == 'storage':
        candidates[0]['persistent_numeric_bytes'] = 101
    elif mutation == 'worst_record':
        set_score(evaluations, audit.CANDIDATE, 10.200001, record=audit.DEV[0])
    elif mutation == 'amplitude':
        for e in candidates:
            for r in e['rows']: r['rmse'] = 10. if r['record_id'].startswith('100mV') else 1.
    elif mutation == 'seed':
        set_score(evaluations, audit.CANDIDATE, 10., seed=9201)
    else:
        candidates[0]['rows'][0]['rmse'] = float('nan')
    result = audit.decisions(evaluations, fits, folds)
    assert result['conditions'][condition] is False
    assert result['status'] == 'DEVELOPMENT_FAIL'


def test_exact_inclusive_margins_but_strict_seed_and_amplitude_advantage():
    evaluations, fits, folds = evidence()
    set_score(evaluations, audit.CANDIDATE, 9.5)
    for e in evaluations:
        if e['family'] == audit.CANDIDATE: e['median_request_ms'] = 1.1
    assert audit.decisions(evaluations, fits, folds)['passed'] == 9
    set_score(evaluations, audit.CANDIDATE, 10.)
    result = audit.decisions(evaluations, fits, folds)
    assert not result['conditions']['every_seed_below_strongest_control']
    assert not result['conditions']['both_amplitudes_below_strongest_control']
    assert result['conditions']['no_record_over_two_percent_strongest_control']


@pytest.mark.parametrize('order', audit.ORDERS)
def test_mean_gram_equation_and_unpenalized_intercept(order):
    width = 6*order+4
    gram = np.eye(width)
    cross = np.ones((width, 3))
    alpha = .1
    coefficients = np.full((3, width), 1/1.1)
    coefficients[:, -1] = 1.
    certificate = audit.ridge_certificate(coefficients, gram, cross, order, alpha)
    assert certificate['relative_backward_error'] < 1e-15
    wrong = coefficients.copy(); wrong[:, -1] = 1/1.1
    with pytest.raises(ValueError, match='normal equation'):
        audit.ridge_certificate(wrong, gram, cross, order, alpha)


def test_zero_system_certificate_and_corrupt_coefficient_rejection():
    gram, cross, coefficients = np.zeros((196, 196)), np.zeros((196, 3)), np.zeros((3, 196))
    assert audit.ridge_certificate(coefficients, gram, cross, 32, 1e-6)['relative_backward_error'] == 0.
    coefficients[0, 0] = 1.
    with pytest.raises(ValueError, match='normal equation'):
        audit.ridge_certificate(coefficients, gram, cross, 32, 1e-6)


@pytest.mark.parametrize('architecture,count,total', [('affine_feedback',588,11056), ('tanh_output_only',4779,44592)])
def test_frozen_checkpoint_shape_and_exact_storage(architecture, count, total):
    backbone = np.zeros((3,196))
    kind, mode = architecture.split('_',1)
    state = {'coefficients': torch.from_numpy(backbone.copy())}
    shapes = {'residual.weight': (3,195),'residual.bias': (3,)} if kind == 'affine' else {'residual.0.weight': (24,195),'residual.0.bias': (24,),'residual.2.weight': (3,24),'residual.2.bias': (3,)}
    state.update({k:torch.zeros(s,dtype=torch.float64) for k,s in shapes.items()})
    payload={'model':state,'accepted_updates':2048,'optimizer':{}}
    assert audit.checkpoint_state(payload,architecture,backbone)[1:] == (count,mode)
    spec={'order':32,'residual_kind':kind,'mode':mode,'trainable_parameter_count':count,'buffer_bytes':4704,'numeric_metadata_bytes':16 if kind=='affine' else 24,'dtype':'float64','parameter_count':count,'parameter_bytes':count*8,'state_scalars':192}
    assert audit.validate_spec(spec,order=32,alpha=1e-6,kind=kind,mode=mode)['persistent_numeric_bytes'] == total
    state[next(iter(shapes))].view(-1)[0]=float('nan')
    with pytest.raises(ValueError,match='tensor geometry'):
        audit.checkpoint_state(payload,architecture,backbone)


@pytest.mark.parametrize('order,count,total',[(32,588,6360),(64,1164,12504),(96,1740,18648)])
def test_longer_linear_state_is_charged(order,count,total):
    spec={'order':order,'alpha':.001,'fit_rows':12*(8192-order),'fit_record_ids':list(audit.FIT),'scalar_metadata_bytes':24,'retained_numeric_bytes':count*8+24,'dtype':'float64','parameter_count':count,'parameter_bytes':count*8,'state_scalars':6*order}
    assert audit.validate_spec(spec,order=order,alpha=.001)['persistent_numeric_bytes'] == total
    spec['state_scalars']=192 if order!=32 else 1
    with pytest.raises(ValueError):audit.validate_spec(spec,order=order,alpha=.001)


def authenticated_tree(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    study=tmp_path/'output/fsm-linear-controls-study-v1';study.mkdir(parents=True)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value)
        return path
    sources={}
    for name in audit.SOURCES:
        write(tmp_path/name,'frozen source '+name)
        write(study/'source'/name,'frozen source '+name)
        sources[name]=audit.pin(tmp_path/name)['sha256']
    artifacts={}
    for key in audit.PARENTS:
        if key.startswith('checkpoint_'):
            arch,seed=key[len('checkpoint_'):].rsplit('_',1)
            relative=f'output/fsm-residual-study-v1/{audit.SELECTED[arch]}-{seed}/final.pt'
        elif key in ('coefficients','normalizer','fit_receipt'):
            suffix='.json' if key=='fit_receipt' else '.npz'
            relative='output/fsm-correction-study-v1/'+key+suffix
        elif key in ('residual_summary','residual_closure'):
            relative='output/fsm-residual-study-v1/'+key.removeprefix('residual_')+'.json'
        elif key=='residual_registration':relative='research/fsm-residual-registration.json'
        elif key=='residual_process':relative='output/fsm-residual-engineering-v1/original-process.json'
        else:relative='output/fsm-residual-audit-v1/audit.json'
        write(tmp_path/relative,'opaque '+key)
        artifacts[key]={'path':relative,'sha256':audit.pin(tmp_path/relative)['sha256']}
    def change(key,value):
        path=write(tmp_path/artifacts[key]['path'],value)
        artifacts[key]['sha256']=audit.pin(path)['sha256']
    change('residual_summary',{'status':'DEVELOPMENT_PASS','selected_by_architecture':audit.SELECTED})
    change('residual_closure',{'status':'completed','summary_sha256':artifacts['residual_summary']['sha256']})
    change('residual_process',{'observed_exit_code':0,'summary_sha256':artifacts['residual_summary']['sha256']})
    change('residual_registration',{'parent_artifacts':{k:artifacts[k].copy() for k in ('coefficients','fit_receipt','normalizer')}})
    parent_files={}
    for key,item in artifacts.items():
        if key in ('residual_summary','residual_closure') or key.startswith('checkpoint_'):
            parent_files[audit.parent_key(item['path'],'/old-host/project/output/fsm-residual-study-v1')]=audit.pin(tmp_path/item['path'])
        elif key in ('coefficients','fit_receipt','normalizer'):
            parent_files['parent/'+key+Path(item['path']).suffix]=audit.pin(tmp_path/item['path'])
    change('residual_audit',{'status':'PASS','agreement':True,'inputs':{'study':'/old-host/project/output/fsm-residual-study-v1','registration':audit.pin(tmp_path/artifacts['residual_registration']['path']),'process':audit.pin(tmp_path/artifacts['residual_process']['path']),'files':parent_files}})
    for key,item in artifacts.items():
        target=study/'parent'/(key+Path(item['path']).suffix);target.parent.mkdir(exist_ok=True)
        target.write_bytes((tmp_path/item['path']).read_bytes())
    markdown=write(tmp_path/'research/fsm-linear-controls-protocol.md','frozen protocol')
    plan={'experiment':audit.CONFIG,'source_sha256':sources,'parent_artifacts':artifacts,'protocol_markdown_path':str(markdown.relative_to(tmp_path)),'protocol_markdown_sha256':audit.pin(markdown)['sha256']}
    registration=write(tmp_path/audit.REG,plan)
    monkeypatch.setattr(audit,'REGISTRATION_SHA256',audit.pin(registration)['sha256'])
    write(study/'protocol.json',plan)
    write(study/'summary.json',{'status':'DEVELOPMENT_PASS'})
    summary_sha=audit.pin(study/'summary.json')['sha256']
    write(study/'closure.json',{'status':'completed','source_sha256':sources,'summary_sha256':summary_sha})
    process=write(tmp_path/'process.json',{'observed_exit_code':0,'command':audit.COMMAND,'tool_session_id':123,'observation':'original Codex exec/write_stdin completion','summary_sha256':summary_sha})
    helper=write(tmp_path/'helper.py','independent helper')
    monkeypatch.setattr(audit.common,'__file__',str(helper));monkeypatch.setattr(audit,'COMMON_SHA256',audit.pin(helper)['sha256'])
    return study,process,plan


def test_full_opaque_admission_with_different_clone_root(tmp_path,monkeypatch):
    study,process,plan=authenticated_tree(tmp_path,monkeypatch)
    monkeypatch.setattr(audit,'arrays',lambda *_:pytest.fail('metadata may not decode'))
    admitted,paths,_,_=audit.authenticate(study,process)
    assert admitted==plan and len(paths)==20


@pytest.mark.parametrize('mutation',['source','snapshot','parent_original','parent_copy','process','summary','protocol','registration','helper','parent_audit_join'])
def test_metadata_tamper_stops_before_array_decode(tmp_path,monkeypatch,mutation):
    study,process,plan=authenticated_tree(tmp_path,monkeypatch)
    monkeypatch.setattr(audit,'arrays',lambda *_:pytest.fail('metadata may not decode'))
    if mutation=='source':path=tmp_path/next(iter(audit.SOURCES))
    elif mutation=='snapshot':path=study/'source'/next(iter(audit.SOURCES))
    elif mutation=='parent_original':path=tmp_path/plan['parent_artifacts']['coefficients']['path']
    elif mutation=='parent_copy':path=study/'parent/normalizer.npz'
    elif mutation=='process':
        value=json.loads(process.read_text());value['command']=['other'];process.write_text(json.dumps(value));path=None
    elif mutation=='summary':path=study/'summary.json'
    elif mutation=='protocol':path=tmp_path/plan['protocol_markdown_path']
    elif mutation=='registration':path=tmp_path/audit.REG
    elif mutation=='helper':path=Path(audit.common.__file__)
    else:
        key='residual_audit';path=tmp_path/plan['parent_artifacts'][key]['path'];value=json.loads(path.read_text());value['inputs']['files']['summary.json']['sha256']='0'*64;path.write_text(json.dumps(value));(study/'parent/residual_audit.json').write_bytes(path.read_bytes());plan['parent_artifacts'][key]['sha256']=audit.pin(path)['sha256'];(tmp_path/audit.REG).write_text(json.dumps(plan));(study/'protocol.json').write_text(json.dumps(plan));monkeypatch.setattr(audit,'REGISTRATION_SHA256',audit.pin(tmp_path/audit.REG)['sha256']);path=None
    if path:path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):
        audit.authenticate(study,process)


def test_failed_original_process_rejected_before_any_other_read(tmp_path,monkeypatch):
    process=tmp_path/'process.json';process.write_text('{"observed_exit_code":1}')
    monkeypatch.setattr(audit,'pin',lambda _:pytest.fail('must stop before pinning'))
    with pytest.raises(ValueError,match='successful process'):
        audit.authenticate(tmp_path/'missing',process)


@pytest.mark.parametrize('relative,root',[
    ('output/fsm-residual-study-v1/../outside','/host/output/fsm-residual-study-v1'),
    ('output/fsm-correction-study-v1/summary.json','/host/output/fsm-residual-study-v1'),
    ('/output/fsm-residual-study-v1/summary.json','/host/output/fsm-residual-study-v1'),
    ('output/fsm-residual-study-v1/summary.json','output/fsm-residual-study-v1'),
    ('output/fsm-residual-study-v1/summary.json','/host/output/wrong'),
])
def test_parent_relative_join_rejects_outside_or_wrong_root(relative,root):
    with pytest.raises(ValueError):audit.parent_key(relative,root)


def test_audit_ledger_paths_follow_validated_header_without_model_calls(tmp_path,monkeypatch):
    study=(tmp_path/'study').resolve();process=(tmp_path/'process.json').resolve()
    paths={k:tmp_path/k for k in ('coefficients','normalizer','fit_receipt')}
    monkeypatch.setattr(audit,'authenticate',lambda *_:({'data_sha256':'fake'},paths,{},{}))
    monkeypatch.setattr(audit.common,'inventory',lambda *_:{})
    spec={'order':32,'alpha':1e-6,'fit_rows':97920,'fit_record_ids':list(audit.FIT),'scalar_metadata_bytes':24,'retained_numeric_bytes':4728,'dtype':'float64','parameter_count':588,'parameter_bytes':4704,'state_scalars':192}
    values={paths['fit_receipt']:{'status':'complete','order':32,'alpha':1e-6,'model_spec':spec},
            study/'admission.json':{'source_sha256':'fake','decoded_keys':['u_100mV_train','y_100mV_train','u_200mV_train','y_200mV_train'],'fit_ids':list(audit.FIT),'dev_ids':list(audit.DEV),'fit_samples':98304,'normalizer':'unchanged parent FIT normalization'}}
    values.update({study/(name+'.json'):[] for name in ('fits','folds','parents','evaluations')})
    seen=[]
    def fake_read(path):
        seen.append(path)
        return values[path]
    monkeypatch.setattr(audit,'read',fake_read)
    monkeypatch.setattr(audit,'arrays',lambda path,keys:{'coefficients':np.zeros((3,196))} if path==paths['coefficients'] else {'u_mean':np.zeros(3),'u_scale':np.ones(3),'y_mean':np.zeros(3),'y_scale':np.ones(3)})
    monkeypatch.setattr(audit.torch,'load',lambda *_args,**_kwargs:pytest.fail('no checkpoint calls'))
    with pytest.raises(ValueError,match='ordered nine FIT'):
        audit.audit(study,process)
    assert seen[-4:]==[study/(name+'.json') for name in ('fits','folds','parents','evaluations')]
