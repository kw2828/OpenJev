"""Independent fabricated evidence for the position-observer auditor.

No measured arrays, trained checkpoints, original audits or model calls.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_robot_position_observer as audit


def fixture():
    rates = {'local_affine': .003, 'temporal_affine': .001, 'last_two': None,
             'observer_fixed': None, 'observer_zero': None, **audit.parent_audit.CACHED_RATES}
    cfg = {'inherited_rates': rates, 'selection_scope': 'original DEV2 only'}
    rows = []
    for name in audit.EXPOSED:
        for identity in (*audit.parent_audit.identities(), *audit.identities()):
            for horizon in (64, 128):
                arm = identity['arm']
                error = .75 if arm == 'observer_position' else 1.
                count = 22*horizon*6
                failed = arm == 'observer_learned'
                rows.append({'recording': name, 'fit_key': identity['key'], 'arm': arm,
                             'seed': identity['seed'], 'learning_rate': identity['learning_rate'], 'horizon': horizon,
                             'status': 'FAILED' if failed else 'PASS', 'error': {'type': 'PriorFailed'} if failed else None,
                             'metrics': None if failed else {'standardized_rmse': error, 'standardized_sse': error**2*count,
                             'physical_rmse_deg': error, 'per_joint_rmse_deg': [error]*6,
                             'scalars': count, 'windows': 22, 'horizon': horizon}})
    selection = {'selected_rates': {'observer_position': .001}}
    resources = []
    for identity in audit.resource_identities(selection, rates):
        latency = 1.5 if identity['arm'] == 'observer_position' else 1.
        resources.append({**identity, **audit.storage(identity['arm']), 'status': 'PASS', 'error': None,
                          'timing': {'median_seconds': latency}})
    return cfg, rows, resources


def change_metric(rows, arm, value, recording=None, horizon=None, rate=None):
    for row in rows:
        if row['arm'] == arm and (recording is None or row['recording'] == recording) and (horizon is None or row['horizon'] == horizon) and (rate is None or row['learning_rate'] == rate):
            m = row['metrics']
            m['standardized_rmse'] = value
            m['standardized_sse'] = value*value*m['scalars']


def flags(rows, resources, cfg):
    return [r['passed'] for r in audit.decisions(rows, resources, cfg)['result']['conditions']]


def test_complete_roster_and_all_five_pass_with_failed_parent_retained():
    cfg, rows, resources = fixture()
    result = audit.decisions(rows, resources, cfg)
    assert len(rows) == 488 and len(resources) == 46
    assert len(audit.DISPLAY) == 19 and len(audit.ELIGIBLE) == 18
    assert result['selection']['selected_rates'] == {'observer_position': .001}
    assert result['result']['passed'] == result['result']['total'] == 5
    assert result['result']['best_control_mean'] == 1.
    assert 'observer_learned' not in result['result']['equal_file_means']


@pytest.mark.parametrize('tamper', ('missing_row', 'duplicate_row', 'missing_cost', 'duplicate_cost', 'unknown_status', 'bad_geometry'))
def test_roster_corruptions_reject(tamper):
    cfg, rows, resources = fixture()
    if tamper == 'missing_row': rows.pop()
    elif tamper == 'duplicate_row': rows[-1] = copy.deepcopy(rows[0])
    elif tamper == 'missing_cost': resources.pop()
    elif tamper == 'duplicate_cost': resources[-1] = copy.deepcopy(resources[0])
    elif tamper == 'unknown_status': resources[0]['status'] = 'MAYBE'
    else: rows[0]['metrics']['scalars'] -= 1
    with pytest.raises(ValueError): audit.decisions(rows, resources, cfg)


def test_mean_relative_boundary_and_zero_control_are_not_false_gain():
    cfg, rows, resources = fixture()
    change_metric(rows, 'observer_position', .95)
    assert flags(rows, resources, cfg)[1]
    change_metric(rows, 'observer_position', .950001)
    assert not flags(rows, resources, cfg)[1]
    change_metric(rows, 'observer_position', 0.)
    change_metric(rows, 'persistence', 0.)
    assert not flags(rows, resources, cfg)[1]


def test_gain_compares_all_seventeen_controls_including_historical_and_ridge():
    cfg, rows, resources = fixture()
    for arm in ('joint_temporal_affine', 'causal_ridge_100'):
        clone = copy.deepcopy(rows)
        change_metric(clone, arm, .5)
        assert not flags(clone, resources, cfg)[1]


def test_each_file_uses_new_fixed_control_and_inclusive_margin():
    cfg, rows, resources = fixture()
    change_metric(rows, 'observer_position_fixed', .5, audit.EXPOSED[-1])
    change_metric(rows, 'observer_position', .51, audit.EXPOSED[-1])
    assert flags(rows, resources, cfg)[2]
    change_metric(rows, 'observer_position', .510001, audit.EXPOSED[-1])
    assert not flags(rows, resources, cfg)[2]


def test_h64_failure_blocks_completeness_even_when_h128_and_selection_pass():
    cfg, rows, resources = fixture()
    row = next(r for r in rows if r['arm'] == 'observer_position' and r['learning_rate'] == .001 and r['horizon'] == 64)
    row.update(status='FAILED', metrics=None, error={'type': 'KnownNumeric'})
    assert not flags(rows, resources, cfg)[0]


def test_latency_inclusive_one_point_five_and_nonfinite_cost_rejection():
    cfg, rows, resources = fixture()
    assert flags(rows, resources, cfg)[3]
    for r in resources:
        if r['arm'] == 'observer_position': r['timing']['median_seconds'] = 1.500001
    assert not flags(rows, resources, cfg)[3]
    resources[0]['timing']['median_seconds'] = math.nan
    with pytest.raises(ValueError): audit.decisions(rows, resources, cfg)


def test_frontier_requires_weak_all_three_and_one_strict_axis():
    cfg, rows, resources = fixture()
    change_metric(rows, 'observer_position_fixed', .75)
    for r in resources:
        if r['arm'] == 'observer_position_fixed': r['timing']['median_seconds'] = 1.5
    assert flags(rows, resources, cfg)[4]  # exact three-axis tie does not dominate
    for r in resources:
        if r['arm'] == 'observer_position_fixed': r['timing']['median_seconds'] = 1.4
    assert not flags(rows, resources, cfg)[4]


def test_selection_uses_only_dev2_pooled_sse_and_never_reselects_parent():
    cfg, rows, resources = fixture()
    # Opposite former-confirmation performance must not switch the DEV-tied lower rate.
    for name in audit.EXPOSED[2:]:
        change_metric(rows, 'observer_position', 50., name, rate=.001)
    # The inherited local rate remains .003 even if its unused .001 is better.
    change_metric(rows, 'local_affine', .01, rate=.001)
    result = audit.decisions(rows, resources, cfg)
    assert result['selection']['selected_rates'] == {'observer_position': .001}
    assert result['result']['equal_file_means']['local_affine'] == 1.


def test_no_eligible_primary_retains_three_unavailable_and_all_rules_fail_closed():
    cfg, rows, resources = fixture()
    for row in rows:
        if row['arm'] == 'observer_position':
            row.update(status='FAILED', error={'type': 'FailedTrainingAttempt'}, metrics=None)
    old = {(r['arm'], r['seed']): r for r in resources}
    resources = []
    for identity in audit.resource_identities({'selected_rates': {'observer_position': None}}, cfg['inherited_rates']):
        row = {**old[identity['arm'], identity['seed']], **identity}
        if identity['arm'] == 'observer_position':
            row.update(status='UNAVAILABLE', error={'type': 'UnavailableSelectedRecipe'}, timing=None)
        resources.append(row)
    assert flags(rows, resources, cfg) == [False]*5


def test_score_closed_form_and_causal_window_alignment():
    pred = np.ones((1, 128, 6), np.float64)
    pred[:, 64:] = 3.
    target = np.zeros_like(pred)
    short = audit.scored(pred, target, np.arange(1, 7, dtype=np.float64), 64, np)
    long = audit.scored(pred, target, np.arange(1, 7, dtype=np.float64), 128, np)
    assert short['standardized_sse'] == 384 and short['standardized_rmse'] == 1.
    assert long['standardized_sse'] == 3840 and long['standardized_rmse'] == math.sqrt(5.)
    assert short['physical_rmse_deg'] == math.sqrt(91/6)
    index = np.arange(3636, dtype=np.float64)[:, None]+np.arange(6, dtype=np.float64)[None, :]/10
    record = {'q': index, 'u': index+10000., 'raw_indices': np.arange(0, 90881, 25, dtype=np.int64)}
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    starts, batch, targets = audit.windows(record, norm, np)
    assert starts.tolist() == [64+160*i for i in range(22)]
    assert batch['future_u'][0, 0, 0] == 10095.
    assert targets[0, 0, 0] == 96.


def checkpoint_fixture(mode='observer_position'):
    backbone = {'cell.fake': np.arange(590, dtype=np.float32)}
    gain = np.concatenate((np.eye(6, dtype=np.float32), np.zeros((6, 6), dtype=np.float32)))
    initial = {**backbone, 'gain': gain}
    final = {k:v.copy() for k,v in initial.items()}
    optimizer = {'gain/step': np.array(4096, np.float32), 'gain/exp_avg': np.zeros((12, 6), np.float32),
                 'gain/exp_avg_sq': np.zeros((12, 6), np.float32)} if mode == 'observer_position' else {}
    receipt = {'status': 'PASS', 'completed_updates': 4096 if optimizer else 0}
    return initial, final, optimizer, backbone, receipt


@pytest.mark.parametrize('mode', ('observer_position', 'observer_position_fixed'))
def test_frozen_cell_and_gain_only_adam(mode):
    initial, final, optimizer, backbone, receipt = checkpoint_fixture(mode)
    audit.validate_frozen_evidence(initial, final, optimizer, backbone, mode, receipt, np)
    final['cell.fake'][17] += 1
    with pytest.raises(ValueError, match='immutable cell'):
        audit.validate_frozen_evidence(initial, final, optimizer, backbone, mode, receipt, np)


@pytest.mark.parametrize('tamper', ('old_initial_gain', 'cell_adam', 'missing_moment', 'fixed_updated'))
def test_initializer_evidence_rejects_wrong_initialization_or_optimizer_ownership(tamper):
    mode = 'observer_position_fixed' if tamper == 'fixed_updated' else 'observer_position'
    initial, final, optimizer, backbone, receipt = checkpoint_fixture(mode)
    if tamper == 'old_initial_gain': initial['gain'][6:] = np.eye(6, dtype=np.float32)
    elif tamper == 'cell_adam': optimizer['cell.fake/step'] = np.array(4096, np.float32)
    elif tamper == 'missing_moment': optimizer.pop('gain/exp_avg')
    else: receipt['completed_updates'] = 1
    with pytest.raises(ValueError): audit.validate_frozen_evidence(initial, final, optimizer, backbone, mode, receipt, np)


def test_failed_update_can_retain_attempted_adam_step_but_not_pass():
    initial, final, optimizer, backbone, receipt = checkpoint_fixture()
    receipt.update(status='FAILED', completed_updates=22)
    optimizer['gain/step'] = np.array(23, np.float32)
    final['gain'][1, 1] = np.nan
    audit.validate_frozen_evidence(initial, final, optimizer, backbone, 'observer_position', receipt, np)
    receipt['status'] = 'PASS'
    with pytest.raises(ValueError): audit.validate_frozen_evidence(initial, final, optimizer, backbone, 'observer_position', receipt, np)


def test_float64_gradient_diagnostic_distinguishes_entries_from_norm_overflow():
    gradient = np.zeros((12, 6), np.float32)
    gradient[0, :2] = [3e20, 4e20]
    value = audit.gradient_summary(gradient, np)
    assert value['all_finite'] and value['nonfinite_count'] == 0 and value['numel'] == 72
    assert math.isclose(value['norm64'], 5e20, rel_tol=1e-7)
    assert value['max_abs'] == float(gradient[0, 1])
    gradient[4, 1] = np.inf
    gradient[6, 3] = np.nan
    value = audit.gradient_summary(gradient, np)
    assert value == {'numel': 72, 'nonfinite_count': 2, 'all_finite': False, 'max_abs': None, 'norm64': None}


def test_zero_gradient_has_zero_not_missing_magnitudes():
    assert audit.gradient_summary(np.zeros((12, 6), np.float32), np) == {
        'numel': 72, 'nonfinite_count': 0, 'max_abs': 0., 'norm64': 0., 'all_finite': True}


def test_inherited_failure_and_values_cannot_be_repaired_or_reordered():
    _, rows, _ = fixture()
    parent = [copy.deepcopy(r) for r in rows if r['arm'] not in ('observer_position', 'observer_position_fixed')]
    audit.validate_parent_rows(rows, parent)
    changed = copy.deepcopy(rows)
    changed[0]['metrics']['standardized_rmse'] += 1e-13
    with pytest.raises(ValueError, match='unchanged parent416'):
        audit.validate_parent_rows(changed, parent)
    reordered = list(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError): audit.validate_parent_rows(reordered, parent)


def test_all46_timing_summaries_and_fixed_gain_buffers_are_checked():
    cfg, rows, resources = fixture()
    durations = [1.+i/100 for i in range(20)]
    for r in resources:
        r['timing'] = {'seconds': durations.copy(), 'median_seconds': 1.095, 'p95_seconds': 1.1805,
                       'scope': audit.parent_audit.prior_audit.TIMING_SCOPE}
    selection = audit.decisions(rows, resources, cfg)['selection']
    audit.validate_resources(resources, selection, cfg['inherited_rates'], np)
    fixed = next(r for r in resources if r['arm'] == 'observer_position_fixed')
    assert fixed['parameter_bytes'] == 2360 and fixed['buffer_bytes'] == 288
    assert sum(fixed[k] for k in ('parameter_bytes','buffer_bytes','state_bytes','normalizer_bytes')) == 2888
    resources[0]['timing']['seconds'][10] += .1
    with pytest.raises(ValueError, match='median'):
        audit.validate_resources(resources, selection, cfg['inherited_rates'], np)


def write(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def admission_fixture(root, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', root)
    study, eng = root/audit.STUDY, root/audit.ENGINEERING
    sources = {}
    for name in audit.SOURCES:
        payload = ('opaque source '+name).encode()
        write(root/name, payload); write(study/'sources'/name, payload)
        sources[name] = audit.descriptor(root/name)
    write(root/audit.LAUNCHER, b'opaque launcher')
    qpath = eng/'qualification-01/receipt.json'
    commands = [['.venv/bin/ruff','check',*[n for n in audit.QUALIFICATION_SOURCES if n.endswith('.py')]],
                ['.venv/bin/python','-m','pytest','--noconftest','-q','tests/test_robot_position_observer.py',
                 'tests/test_robot_position_observer_study.py','tests/test_audit_robot_position_observer.py']]
    command_rows = []
    for i, command in enumerate(commands):
        path = qpath.parent/f'command-{i+1:02d}.log'; write(path, b'original fabricated qualification log')
        command_rows.append({'command':command,'returncode':0,'seconds':1.,'log':str(path),
                             'sha256':audit.descriptor(path)['sha256']})
    qualification = {'status':'PASS','sources_unchanged':True,'sources':sources,'launcher':audit.descriptor(root/audit.LAUNCHER),
                     'thread_env':dict.fromkeys(audit.THREADS,'1'),'commands':command_rows,'created_utc':'2026-01-01T00:00:00+00:00'}
    write(qpath, qualification)
    plan = {'version':audit.STUDY_VERSION,'sources':sources,'launcher':qualification['launcher'],
            'created_utc':'2026-01-02T00:00:00+00:00','qualification':audit.pin(qpath)}
    write(root/audit.REGISTRATION, plan); write(study/'registration.json',plan)
    launch = {'command':audit.COMMAND.copy(),'prefit_commit':'a'*40,'started_utc':'2026-01-03T00:00:00+00:00',
              'registration_sha256':audit.descriptor(root/audit.REGISTRATION)['sha256'],'launcher':plan['launcher'],
              'thread_env':dict.fromkeys(audit.THREADS,'1'),'scope':'fabricated admission'}
    write(eng/'run-process-01.log', b'original fabricated terminal log')
    process = {**launch,'returncode':0,'elapsed_seconds':2.,'external_timeout':False,'log':audit.descriptor(eng/'run-process-01.log')}
    write(eng/'run-launch-01.json',launch); write(eng/'run-process-01.json',process)
    runtime = {'python':audit.sys.version,'platform':audit.platform.platform(),'machine':audit.platform.machine(),
               'torch_threads':1,'thread_env':dict.fromkeys(audit.THREADS,'1'),'numpy':'test-numpy','torch':'test-torch',
               'clock':'mach_continuous_time' if audit.sys.platform=='darwin' else 'CLOCK_BOOTTIME'}
    write(study/'runtime.json',runtime); write(study/'receipt.json',{'opaque':'no model execution'})
    write(study/'not-decoded.npz', b'not an NPZ archive')
    write(study/'manifest.json',{'files':{str(p.relative_to(study)):audit.descriptor(p)
          for p in study.rglob('*') if p.is_file() and p.name!='receipt.json'}})
    monkeypatch.setattr(audit.importlib.metadata,'version',lambda package:'test-'+package)
    monkeypatch.setattr(audit.subprocess,'check_output',lambda argv,**kwargs:(root/argv[-1].split(':',1)[1]).read_bytes())
    for key in audit.THREADS: monkeypatch.setenv(key,'1')
    def no_decode(*args,**kwargs): raise AssertionError('arrays decoded before admission')
    monkeypatch.setattr(np,'load',no_decode)
    return study,eng,plan,launch,process,qualification


def test_original_metadata_admits_opaque_arrays_without_decoding(tmp_path,monkeypatch):
    study,eng,plan,_,_,_ = admission_fixture(tmp_path,monkeypatch)
    actual,inputs = audit.original_admission(study,eng/'run-process-01.json')
    assert actual == plan and len(plan['sources']) == 54
    assert inputs['registration']['sha256'] == audit.descriptor(tmp_path/audit.REGISTRATION)['sha256']


@pytest.mark.parametrize('damage',('source','snapshot','launcher','registration','command','launch_join','code','timeout','elapsed',
                                    'log','commit','committed_source','qualification_pin','manifest','runtime'))
def test_original_admission_tampering_fails_before_decode(tmp_path,monkeypatch,damage):
    study,eng,plan,launch,process,_ = admission_fixture(tmp_path,monkeypatch)
    if damage=='source': write(tmp_path/audit.SOURCES[0],b'changed')
    elif damage=='snapshot': write(study/'sources'/audit.SOURCES[0],b'changed')
    elif damage=='launcher': write(tmp_path/audit.LAUNCHER,b'changed')
    elif damage=='registration': write(study/'registration.json',b'changed')
    elif damage=='command': launch['command']=process['command']=['python','unrelated.py']
    elif damage=='launch_join': launch['started_utc']='2026-01-04T00:00:00+00:00'
    elif damage=='code': process['returncode']=1
    elif damage=='timeout': process['external_timeout']=True
    elif damage=='elapsed': process['elapsed_seconds']=7261.
    elif damage=='log': write(eng/'run-process-01.log',b'changed')
    elif damage=='commit': launch['prefit_commit']=process['prefit_commit']='bad'
    elif damage=='committed_source': monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**k:b'changed')
    elif damage=='qualification_pin': write(Path(plan['qualification']['path']),b'changed')
    elif damage=='manifest': write(study/'not-decoded.npz',b'changed')
    else: write(study/'runtime.json',{'python':'different'})
    write(eng/'run-launch-01.json',launch); write(eng/'run-process-01.json',process)
    with pytest.raises(ValueError): audit.original_admission(study,eng/'run-process-01.json')


@pytest.mark.parametrize('raw,kind,value', [(3.,'finite',3.),(0.,'finite',0.),(float('inf'),'positive_infinity',None),
                                         (-float('inf'),'negative_infinity',None),(float('nan'),'nan',None)])
def test_native_nonfinite_encoding_never_clips_or_replaces(raw,kind,value):
    assert audit.native_norm_summary(np.array(raw,np.float32),np) == {'kind':kind,'value':value}


def test_entry_failure_takes_precedence_over_native_aggregate_overflow():
    finite = audit.gradient_summary(np.full((12,6),1e20,np.float32),np)
    assert audit.gradient_outcome(finite,{'kind':'positive_infinity','value':None}) == 'native_norm_overflow'
    assert audit.gradient_outcome(finite,{'kind':'finite','value':1e20}) == 'finite'
    nonfinite = audit.gradient_summary(np.full((12,6),np.inf,np.float32),np)
    assert audit.gradient_outcome(nonfinite,{'kind':'nan','value':None}) == 'nonfinite_gradient_entries'
    assert audit.gradient_outcome(finite,None) == 'forward_or_loss_failure'


def test_immediate_parent_metadata_cannot_be_shadowed_by_ancestor_inputs(tmp_path,monkeypatch):
    import publish_robot_observer_study as publication
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    cfg,_,_ = fixture(); folder = tmp_path/publication.STUDY
    common = ('normalizers.npz','linear.npz','causal_ridge_1.npz','causal_ridge_1.json',
              'causal_ridge_100.npz','causal_ridge_100.json',*(f'batches-{s}.npz' for s in audit.SEEDS))
    inputs,inventory = {},{}
    for index in range(11):
        path = tmp_path/'ancestor'/f'data-{index}'; write(path,b'opaque data bytes')
        inputs[f'data/fake-{index}'] = audit.pin(path)
    for name in common:
        path = tmp_path/'ancestor'/name; write(path,('original common '+name).encode())
        inputs['parent/'+name] = audit.pin(path)
        inventory['parent/'+name] = audit.descriptor(path)
    current = ['results.json','fits.json','resources.json','prediction-attempts.json']
    current += [r['key']+'/final.npz' for r in audit.inherited_identities(cfg['inherited_rates']) if r['arm'] not in audit.REFS]
    current += ['observer_learned-8103-lr0/final.npz']
    for name in current:
        write(folder/name,('immediate parent '+name).encode()); inventory[name] = audit.descriptor(folder/name)
    write(tmp_path/'ancestor/fits.json',b'distinct grandparent fits')
    inputs['parent/fits.json'] = audit.pin(tmp_path/'ancestor/fits.json')
    prior = {'plan':{'inputs':inputs},'inventory':inventory,'audit':{'results':{'selection':{'selected_rates':{
             'local_affine':.003,'temporal_affine':.001,'observer_learned':None}}}}}
    expected,rates = audit.parent_inputs(prior)
    assert len(expected) == 61 and rates == cfg['inherited_rates']
    assert expected['parent/fits.json'] == audit.pin(folder/'fits.json')
    assert expected['parent/fits.json'] != inputs['parent/fits.json']
    assert expected['parent/normalizers.npz'] == inputs['parent/normalizers.npz']
    assert expected['diagnostic/observer_learned-8103-lr0/final.npz'] == audit.pin(folder/'observer_learned-8103-lr0/final.npz')


def prefix(steps=30):
    return {'prefix_steps':steps,'max_state_abs':5.,'max_state_norm64':7.,
            'max_innovation_abs':2.,'max_innovation_norm64':3.}


def diagnostic_fixture():
    gradient = np.zeros((12,6),np.float32); gradient.flat[:2] = (3.,4.)
    good = {'numel':72,'nonfinite_count':0,'max_abs':4.,'norm64':5.,'all_finite':True}
    error = {'type':'FitFailure','message':'nonfinite training gradient norm'}
    records = [{'update':1,'prefix':prefix(),'gradient':good,'native_norm':{'kind':'finite','value':5.},
                'status':'PASS','error':None},
               {'update':2,'prefix':prefix(),'gradient':good,'native_norm':{'kind':'positive_infinity','value':None},
                'status':'FAILED','error':error}]
    summary = {'attempts':2,'backward_calls':2,'clip_calls':2,'nonfinite_gradient_attempts':0,
               'native_norm_nonfinite_attempts':1,'prefix_steps':60,'max_gradient_abs':4.,'max_gradient_norm64':5.,
               'max_state_abs':5.,'max_state_norm64':7.,'max_innovation_abs':2.,'max_innovation_norm64':3.}
    raw = {'gain_gradient':gradient,'native_norm':np.array(np.inf,np.float32)}
    receipt = {'status':'FAILED','completed_updates':1,'error':error}
    trace = [{'update':1,'loss':2.,'gradient_norm_before_clip':5.}]
    return records,summary,raw,receipt,trace


def test_independent_diagnostic_aggregate_and_raw_gradient_reconcile_without_clip_replay():
    args = diagnostic_fixture()
    audit.validate_diagnostics(*args,np)
    assert audit.aggregate_diagnostics(args[0]) == args[1]
    assert args[2]['gain_gradient'].flat[1] == 4 and np.isposinf(args[2]['native_norm'])


@pytest.mark.parametrize('damage',('summary','raw_gradient','raw_native','entry_count','attempt_order','prefix_steps','trace_norm','earlier_failure'))
def test_attempt_or_aggregate_corruption_rejected(damage):
    records,summary,raw,receipt,trace = diagnostic_fixture()
    if damage=='summary': summary['native_norm_nonfinite_attempts']=0
    elif damage=='raw_gradient': raw['gain_gradient'].flat[1]=3.
    elif damage=='raw_native': raw['native_norm']=np.array(5.,np.float32)
    elif damage=='entry_count': records[-1]['gradient']['numel']=73
    elif damage=='attempt_order': records[-1]['update']=3
    elif damage=='prefix_steps': records[-1]['prefix']['prefix_steps']=29
    elif damage=='trace_norm': trace[0]['gradient_norm_before_clip']=6.
    else: records[0].update(status='FAILED',error=receipt['error'])
    with pytest.raises(ValueError): audit.validate_diagnostics(records,summary,raw,receipt,trace,np)


def test_forward_failure_has_empty_raw_bank_and_partial_prefix_not_fabricated_gradient():
    records,_,_,receipt,trace = diagnostic_fixture()
    error = {'type':'FitFailure','message':'nonfinite autoregressive training loss'}
    records[-1].update(prefix=prefix(4),gradient=None,native_norm=None,error=error)
    receipt['error']=error
    expected = {'attempts':2,'backward_calls':1,'clip_calls':1,'nonfinite_gradient_attempts':0,
                'native_norm_nonfinite_attempts':0,'prefix_steps':34,'max_gradient_abs':4.,'max_gradient_norm64':5.,
                'max_state_abs':5.,'max_state_norm64':7.,'max_innovation_abs':2.,'max_innovation_norm64':3.}
    audit.validate_diagnostics(records,expected,{},receipt,trace,np)
    with pytest.raises(ValueError): audit.validate_diagnostics(records,expected,{'gain_gradient':np.zeros((12,6),np.float32)},receipt,trace,np)


def test_after_update_cap_failure_keeps_original_completed_trace_and_preclip_bank():
    records,_,raw,receipt,trace = diagnostic_fixture()
    records = records[:1]
    error = {'type':'FitFailure','message':'native single-fit deadline exceeded'}
    records[0].update(status='FAILED',error=error); receipt['error']=error
    raw['native_norm']=np.array(5.,np.float32)
    audit.validate_diagnostics(records,audit.aggregate_diagnostics(records),raw,receipt,trace,np)
    # Preservation can fail after diagnostics were serialized with PASS.
    records[0].update(status='PASS',error=None)
    receipt['error']={'type':'FitFailure','message':'single-fit wall cap exceeded during preservation'}
    audit.validate_diagnostics(records,audit.aggregate_diagnostics(records),raw,receipt,trace,np)


def probe_fixture():
    records,_,raw,_,_ = diagnostic_fixture()
    identity = audit.probe_schedule()[0]
    record = {**identity,'status':'FAILED','outcome':'native_norm_overflow',
              'error':{'type':'FitFailure','message':'nonfinite training gradient norm'},'prefix':prefix(),
              'loss':2.,'gradient':records[-1]['gradient'],'native_norm':records[-1]['native_norm'],
              'parameter_hash_before':'same','parameter_hash_after':'same','parameters_unchanged':True,
              'frozen_cell_sha256':'cell','seconds':.1,'optimizer_steps':0,'files':{'preclip.npz':{}},'scope':audit.PROBE_SCOPE}
    return record,raw,identity


def test_probe_raw_native_overflow_is_failure_despite_finite_entries_and_float64_norm():
    record,raw,identity = probe_fixture()
    audit.validate_probe(record,raw,identity,'same','cell',np)
    assert record['gradient']['all_finite'] and record['gradient']['norm64']==5.
    record['outcome']='finite'
    with pytest.raises(ValueError): audit.validate_probe(record,raw,identity,'same','cell',np)


@pytest.mark.parametrize('damage',('parameter_mutation','optimizer','cell','native_kind','source_identity','fake_pass'))
def test_no_update_probe_guards(damage):
    record,raw,identity = probe_fixture()
    if damage=='parameter_mutation': record['parameter_hash_after']='different'
    elif damage=='optimizer': record['optimizer_steps']=1
    elif damage=='cell': record['frozen_cell_sha256']='different'
    elif damage=='native_kind': record['native_norm']={'kind':'nan','value':None}
    elif damage=='source_identity': record['batch_index']=1
    else: record.update(status='PASS',error=None,outcome='finite')
    with pytest.raises(ValueError): audit.validate_probe(record,raw,identity,'same','cell',np)


def test_probe_before_backward_failure_preserves_absence_and_original_numeric_error():
    record,_,identity = probe_fixture()
    record.update(outcome='forward_or_loss_failure',prefix=prefix(7),gradient=None,native_norm=None,loss=None,
                  error={'type':'FitFailure','message':'nonfinite structured transition output; no clipping or repair'})
    # Use the first exact qualified output guard, without a fabricated success.
    record['error']['message']=audit.NUMERIC_ERRORS[0]
    audit.validate_probe(record,{},identity,'same','cell',np)
    record['gradient']={'numel':72,'nonfinite_count':0,'max_abs':0.,'norm64':0.,'all_finite':True}
    with pytest.raises(ValueError): audit.validate_probe(record,{},identity,'same','cell',np)


def test_fit_receipt_requires_all_seven_files_and_original_failure_semantics():
    trace = [{'update':1,'loss':2.,'gradient_norm_before_clip':5.}]
    receipt = {'status':'FAILED','error':{'type':'FitFailure','message':'nonfinite training gradient norm'},
               'completed_updates':1,'requested_updates':4096,'learning_rate':.001,'optimizer_seconds':.1,'fit_seconds':.2,
               'fit_cap_scope':'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows',
               'timing_scope':'optimizer loop including batch construction and finite checks, excluding model construction and saved files',
               'diagnostics_scope':audit.DIAGNOSTICS_SCOPE,
               'files':{n:{} for n in ('initial.npz','final.npz','optimizer.npz','trace.json','diagnostics.json','diagnostic-summary.json','last-gradient.npz')}}
    audit.validate_fit_receipt(receipt,trace,.001,1800.)
    receipt['files'].pop('diagnostic-summary.json')
    with pytest.raises(ValueError): audit.validate_fit_receipt(receipt,trace,.001,1800.)


def test_metric_checkpoint_key_cannot_be_relabelled_with_same_scores():
    cfg,rows,resources=fixture(); rows[0]['fit_key']='unrelated-checkpoint'
    with pytest.raises(ValueError): audit.decisions(rows,resources,cfg)
