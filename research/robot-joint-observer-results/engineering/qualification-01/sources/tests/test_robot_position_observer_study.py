"""Fabricated position-observer schedule, selection and evidence contracts.

No measured files, trained checkpoints, model construction, or fit calls.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_position_observer_study as study

CANDIDATE = 'observer_position'
FIXED = 'observer_position_fixed'
SEEDS = (8101, 8102, 8103)
RATES = (.001, .003)
OLD_LEARNED = ('local_affine', 'temporal_affine', 'observer_learned')
OLD_FIXED = ('last_two', 'observer_fixed', 'observer_zero')
CACHED = ('joint_local_affine', 'joint_temporal_affine', 'dense_bounded',
          'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
OLD_FAMILIES = (*OLD_LEARNED, *OLD_FIXED, *CACHED, *REFERENCES)
ELIGIBLE = (CANDIDATE, FIXED, *(a for a in OLD_FAMILIES if a != 'observer_learned'))
CONDITIONS = ('complete_forecasts_and_costs', 'equal_four_file_mean_5pct_vs_best_control',
              'each_file_within_2pct_best_simple', 'latency_within_150pct_last_two',
              'complete_frontier_not_dominated')


def metrics(value, horizon):
    count = 22*horizon*6
    return {'standardized_rmse': value, 'standardized_sse': value*value*count,
            'scalars': count, 'physical_rmse_deg': value,
            'per_joint_rmse_deg': [value]*6, 'windows': 22, 'horizon': horizon}


def recipes():
    """Literal roster independent of the producer identity builder."""
    cfg = study.config()
    rows = []
    for arm in OLD_LEARNED:
        rows += [{'key': f'{arm}-{s}-lr{i}', 'arm': arm, 'seed': s,
                  'learning_rate': r, 'origin': 'fresh'} for s in SEEDS for i, r in enumerate(RATES)]
    for arm in OLD_FIXED:
        rows += [{'key': f'{arm}-{s}-fixed', 'arm': arm, 'seed': s,
                  'learning_rate': None, 'origin': 'fixed'} for s in SEEDS]
    for arm in CACHED:
        rows += [{'key': f'{arm}-{s}-cached', 'arm': arm, 'seed': s,
                  'learning_rate': cfg['inherited_rates'][arm], 'origin': 'cached'} for s in SEEDS]
    rows += [{'key': a, 'arm': a, 'seed': None, 'learning_rate': None,
              'origin': 'reference'} for a in REFERENCES]
    rows += [{'key': f'{CANDIDATE}-{s}-lr{i}', 'arm': CANDIDATE, 'seed': s,
              'learning_rate': r, 'origin': 'fresh'} for s in SEEDS for i, r in enumerate(RATES)]
    rows += [{'key': f'{FIXED}-{s}-fixed', 'arm': FIXED, 'seed': s,
              'learning_rate': None, 'origin': 'fixed'} for s in SEEDS]
    return rows


def fixture_rows():
    cfg = study.config()
    rows = []
    for name in cfg['partitions']['exposed']:
        for item in recipes():
            for horizon in (64, 128):
                failed = item['arm'] == 'observer_learned'
                rows.append({'fit_key': item['key'], **{k: item[k] for k in ('arm', 'seed', 'learning_rate', 'origin')},
                             'recording': name, 'horizon': horizon,
                             'status': 'FAILED' if failed else 'PASS',
                             'error': {'type': 'FailedTrainingAttempt'} if failed else None,
                             'metrics': None if failed else metrics(.75 if item['arm'] == CANDIDATE else 1., horizon)})
    return cfg, rows


def fixture_resources(selection=None):
    if selection is None:
        cfg, rows = fixture_rows()
        selection = study.select(rows, cfg)
    result = []
    for identity in study.resource_identities(selection):
        arm = identity['arm']
        count = {'last_two': 590, 'local_affine': 962, 'temporal_affine': 962,
                 'observer_fixed': 590, 'observer_zero': 590, CANDIDATE: 662, FIXED: 590}.get(arm, 1000)
        buffered = arm in ('observer_fixed', 'observer_zero', FIXED)
        unavailable = identity['origin'] == 'unavailable'
        seconds = 1.5 if arm == CANDIDATE else 1.
        result.append({**identity, 'status': 'UNAVAILABLE' if unavailable else 'PASS',
                       'error': {'type': 'UnavailableSelectedRecipe'} if unavailable else None,
                       'parameters': count, 'parameter_bytes': count*4, 'buffer_bytes': 288 if buffered else 0,
                       'state_bytes': 48, 'normalizer_bytes': 192,
                       'timing': None if unavailable else {'seconds': [seconds]*20,
                       'median_seconds': seconds, 'p95_seconds': seconds}})
    return result


def set_error(rows, arm, value, *, recording=None, rate=None, horizon=None):
    for row in rows:
        if (row['arm'] == arm and (recording is None or row['recording'] == recording)
                and (rate is None or row['learning_rate'] == rate)
                and (horizon is None or row['horizon'] == horizon)):
            row.update(status='PASS', error=None, metrics=metrics(value, row['horizon']))


def rule(rows, resources=None):
    cfg = study.config()
    selected = study.select(rows, cfg)
    return study.evaluate_rule(rows, selected, fixture_resources(selected) if resources is None else resources, cfg)


def flags(result):
    return [r['passed'] for r in result['conditions']]


def test_exact_new_and_retained_rosters_and_independent_passing_witness():
    cfg, rows = fixture_rows()
    expected = recipes()
    actual = study.identities()
    assert len(actual) == len({r['key'] for r in actual}) == 9
    assert {tuple(sorted(r.items())) for r in actual} == {
        tuple(sorted(r.items())) for r in expected if r['arm'] in (CANDIDATE, FIXED)}
    controls = study.inherited_identities()
    assert len(controls) == 40 and len({r['key'] for r in controls}) == 40
    assert len([r for r in controls if r['seed'] is not None]) == 36
    assert 'observer_learned' not in {r['arm'] for r in controls}
    assert len(rows) == 488 and sum(r['arm'] not in (CANDIDATE, FIXED) for r in rows) == 416
    assert len({(r['fit_key'], r['recording']) for r in rows}) == 244
    assert len(fixture_resources()) == 46 and len(ELIGIBLE) == 18
    assert cfg['seeds'] == list(SEEDS) and cfg['learning_rates'] == list(RATES)
    assert cfg['updates'] == 4096 and cfg['batch_size'] == 16
    assert cfg['context'] == 32 and cfg['train_horizon'] == cfg['dev_horizon'] == 128
    assert cfg['fit_cap_seconds'] == 1800 and cfg['wall_cap_seconds'] == 7200
    result = rule(rows)
    assert tuple(r['name'] for r in result['conditions']) == CONDITIONS
    assert flags(result) == [True]*5 and result['passed'] == result['total'] == 5
    assert result['status'] == 'POSITION_OBSERVER_DEVELOPMENT_PASS'
    assert 'observer_learned' not in result['equal_file_means']


@pytest.mark.parametrize('value,passes', [(.95, True), (.950001, False)])
def test_best_control_gain_boundary(value, passes):
    _, rows = fixture_rows(); set_error(rows, CANDIDATE, value)
    assert flags(rule(rows)) == [True, passes, True, True, True]


@pytest.mark.parametrize('arm', tuple(a for a in ELIGIBLE if a != CANDIDATE))
def test_every_one_of_seventeen_controls_can_block_global_gain(arm):
    _, rows = fixture_rows(); set_error(rows, arm, .78125)
    assert flags(rule(rows))[1] is False


def test_zero_error_control_tie_is_not_relative_improvement():
    _, rows = fixture_rows()
    set_error(rows, CANDIDATE, 0.); set_error(rows, 'causal_ridge_100', 0.)
    assert flags(rule(rows))[:2] == [True, False]


@pytest.mark.parametrize('simple', ('last_two', 'local_affine', 'temporal_affine', 'observer_fixed', FIXED))
@pytest.mark.parametrize('value,passes', [(1.02, True), (1.020001, False)])
def test_each_file_harm_guard_uses_each_of_five_simple_controls(simple, value, passes):
    cfg, rows = fixture_rows()
    set_error(rows, CANDIDATE, .5)
    set_error(rows, CANDIDATE, value*.5, recording=cfg['partitions']['exposed'][-1])
    set_error(rows, simple, .5, recording=cfg['partitions']['exposed'][-1])
    assert flags(rule(rows))[2] is passes


def test_only_dev2_h128_pooled_sse_selects_new_rate_without_old_reselection():
    cfg, rows = fixture_rows()
    set_error(rows, CANDIDATE, .75, rate=.003); set_error(rows, CANDIDATE, 0., rate=.001)
    first = next(r for r in rows if r['arm'] == CANDIDATE and r['learning_rate'] == .001
                 and r['recording'] in cfg['partitions']['dev'] and r['horizon'] == 128)
    first['metrics'] = metrics(2., 128)  # SSE 4 exceeds six times .75 squared.
    for name in set(cfg['partitions']['exposed'])-set(cfg['partitions']['dev']):
        set_error(rows, CANDIDATE, 0., rate=.001, recording=name)
        set_error(rows, CANDIDATE, 100., rate=.003, recording=name)
    set_error(rows, CANDIDATE, 1000., rate=.003, horizon=64)
    set_error(rows, 'local_affine', 0., rate=.001)
    selection = study.select(rows, cfg)
    assert selection['selected_rates'] == {CANDIDATE: .003}
    inherited = {r['arm']: r['learning_rate'] for r in study.inherited_identities()}
    assert inherited['local_affine'] == .003 and inherited['temporal_affine'] == .001
    _, tied = fixture_rows()
    assert study.select(tied, cfg)['selected_rates'] == {CANDIDATE: .001}


@pytest.mark.parametrize('seconds,passes', [(1.5, True), (1.500001, False)])
def test_full_request_latency_boundary(seconds, passes):
    _, rows = fixture_rows(); resources = fixture_resources()
    for r in resources:
        if r['arm'] == CANDIDATE:
            r['timing'] = {'seconds': [seconds]*20, 'median_seconds': seconds, 'p95_seconds': seconds}
    assert flags(rule(rows, resources)) == [True, True, True, passes, True]


@pytest.mark.parametrize('axis', ['error', 'latency', 'bytes', 'tie', 'tradeoff'])
def test_frontier_has_three_axes_and_requires_one_strict_improvement(axis):
    _, rows = fixture_rows(); resources = fixture_resources()
    set_error(rows, FIXED, .5 if axis == 'error' else 1. if axis == 'tradeoff' else .75)
    for r in resources:
        if r['arm'] == FIXED:
            r['buffer_bytes'] = 284 if axis == 'bytes' else 288
            seconds = 1. if axis == 'latency' else 1.5
            r['timing'] = {'seconds': [seconds]*20, 'median_seconds': seconds, 'p95_seconds': seconds}
    result = rule(rows, resources)
    assert flags(result)[4] is (axis in ('tie', 'tradeoff'))
    assert result['dominators'] == ([] if axis in ('tie', 'tradeoff') else [FIXED])


def test_parent_failed_observer_is_retained_but_never_a_comparator():
    _, rows = fixture_rows()
    assert sum(r['arm'] == 'observer_learned' and r['status'] == 'FAILED' for r in rows) == 48
    set_error(rows, 'observer_learned', 0.)
    # Even a different descriptive parent value cannot change eligibility.
    assert flags(rule(rows)) == [True]*5
    assert 'observer_learned' not in rule(rows)['costs']


def test_selected_h64_failure_cannot_be_hidden_by_good_primary_scores():
    _, rows = fixture_rows()
    r = next(r for r in rows if r['arm'] == FIXED and r['horizon'] == 64)
    r.update(status='FAILED', error={'type': 'NonfiniteEvaluation'}, metrics=None)
    assert flags(rule(rows)) == [False, True, True, True, True]


def test_unselected_new_rate_failure_stays_visible_without_blocking_selected_recipe():
    _, rows = fixture_rows()
    for r in rows:
        if r['arm'] == CANDIDATE and r['learning_rate'] == .003:
            r.update(status='FAILED', error={'type': 'FailedTrainingAttempt'}, metrics=None)
    assert len(rows) == 488 and flags(rule(rows)) == [True]*5


def test_all_new_rates_failed_retains_three_unavailable_cost_slots():
    cfg, rows = fixture_rows()
    for r in rows:
        if r['arm'] == CANDIDATE:
            r.update(status='FAILED', error={'type': 'FailedTrainingAttempt'}, metrics=None)
    selection = study.select(rows, cfg)
    assert selection['selected_rates'] == {CANDIDATE: None}
    costs = study.resource_identities(selection)
    assert len(costs) == 46
    assert [r for r in costs if r['arm'] == CANDIDATE] == [
        {'key': f'{CANDIDATE}-{s}-unavailable', 'arm': CANDIDATE, 'seed': s,
         'learning_rate': None, 'origin': 'unavailable'} for s in SEEDS]
    assert flags(rule(rows)) == [False]*5


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_recording', 'wrong_geometry'])
def test_metric_roster_corruption_rejects(damage):
    _, rows = fixture_rows(); r = next(r for r in rows if r['arm'] == FIXED)
    if damage == 'missing': rows.remove(r)
    elif damage == 'duplicate': rows.append(copy.deepcopy(r))
    elif damage == 'wrong_recording': r['recording'] = 'unregistered'
    else: r['metrics']['scalars'] -= 1
    with pytest.raises(ValueError): rule(rows)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_rate'])
def test_cost_roster_corruption_rejects(damage):
    _, rows = fixture_rows(); resources = fixture_resources(); r = resources[0]
    if damage == 'missing': resources.remove(r)
    elif damage == 'duplicate': resources.append(copy.deepcopy(r))
    else: r['learning_rate'] = .77
    with pytest.raises(ValueError): rule(rows, resources)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def descriptor(path):
    blob = path.read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def absolute_pin(path):
    return {'path': str(path.resolve()), **descriptor(path)}


def admission_fixture(tmp_path, monkeypatch):
    """Only fabricated source and opaque invalid-array bytes are inspected."""
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    for name in (*study.SOURCES, study.LAUNCHER):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('fabricated source '+name).encode())
    sources = {name: descriptor(tmp_path/name) for name in study.SOURCES}
    assert len(sources) == 54
    inputs = {}
    for i, name in enumerate(sorted(study.expected_input_names())):
        path = tmp_path/'opaque'/str(i); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('invalid NPZ, never decoded: '+name).encode())
        inputs[name] = absolute_pin(path)
    assert len(inputs) == 61
    audit = tmp_path/study.publication.AUDIT
    manifest = tmp_path/study.publication.OUTPUT/'manifest.json'
    write_json(audit, {'status': 'PASS', 'agreement': True})
    write_json(manifest, {'files': {}})
    commands = [
        ['.venv/bin/ruff', 'check', *[n for n in study.QUALIFICATION_SOURCES if n.endswith('.py')]],
        ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
         'tests/test_robot_position_observer.py', 'tests/test_robot_position_observer_study.py',
         'tests/test_audit_robot_position_observer.py']]
    completed = []
    for i, command in enumerate(commands):
        log = tmp_path/f'qualification-{i}.log'; log.write_bytes(b'fabricated closed qualification')
        completed.append({'command': command, 'returncode': 0, 'log': str(log),
                          'sha256': descriptor(log)['sha256'], 'seconds': 1.})
    q = {'status': 'PASS', 'sources_unchanged': True, 'sources': sources,
         'launcher': descriptor(tmp_path/study.LAUNCHER),
         'thread_env': dict.fromkeys(study.old.THREADS, '1'), 'commands': completed}
    qpath = tmp_path/'qualification.json'; write_json(qpath, q)
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'launcher': q['launcher'], 'qualification': absolute_pin(qpath), 'inputs': inputs,
            'parent_publication_manifest': absolute_pin(manifest), 'parent_audit': absolute_pin(audit)}
    path = tmp_path/'registration.json'; write_json(path, plan)
    for name in study.old.THREADS: monkeypatch.setenv(name, '1')
    visits, token = [], object()
    def admit(*args):
        assert args == (tmp_path/study.publication.STUDY, audit, tmp_path/study.publication.ENGINEERING)
        visits.append('closed-parent'); return token
    def expected(prior):
        assert prior is token
        visits.append('inherited-roster'); return inputs
    def forbidden(*args, **kwargs): raise AssertionError('numeric decoder used during metadata admission')
    monkeypatch.setattr(study.publication, 'authenticate', admit)
    monkeypatch.setattr(study, 'expected_inputs', expected)
    monkeypatch.setattr(study.np, 'load', forbidden)
    return path, plan, qpath, q, visits


def test_exact_admission_is_metadata_only_and_parent_closed_first(tmp_path, monkeypatch):
    path, plan, _, _, visits = admission_fixture(tmp_path, monkeypatch)
    actual, sha = study.authenticate(path)
    assert actual == plan and sha == descriptor(path)['sha256']
    assert visits == ['closed-parent', 'inherited-roster']


@pytest.mark.parametrize('damage', ['source', 'sources', 'launcher', 'config', 'qualification_status',
    'qualification_sources', 'qualification_command', 'qualification_returncode', 'qualification_log',
    'qualification_unchanged', 'threads', 'input_bytes', 'input_roster', 'parent_audit',
    'parent_manifest', 'parent_closure'])
def test_admission_tampering_stops_before_any_decoder(tmp_path, monkeypatch, damage):
    path, plan, qpath, q, _ = admission_fixture(tmp_path, monkeypatch)
    if damage == 'source': (tmp_path/study.SOURCES[0]).write_bytes(b'changed')
    elif damage == 'sources': plan['sources'].pop(study.SOURCES[0])
    elif damage == 'launcher': (tmp_path/study.LAUNCHER).write_bytes(b'changed')
    elif damage == 'config': plan['config']['diagnostic_probes'] = 6
    elif damage == 'qualification_status': q['status'] = 'FAILED'
    elif damage == 'qualification_sources': q['sources'] = {}
    elif damage == 'qualification_command': q['commands'][1]['command'][-1] = 'tests/unrelated.py'
    elif damage == 'qualification_returncode': q['commands'][1]['returncode'] = 1
    elif damage == 'qualification_log': Path(q['commands'][0]['log']).write_bytes(b'changed')
    elif damage == 'qualification_unchanged': q['sources_unchanged'] = False
    elif damage == 'threads': monkeypatch.setenv(study.old.THREADS[0], '2')
    elif damage == 'input_bytes': Path(next(iter(plan['inputs'].values()))['path']).write_bytes(b'changed')
    elif damage == 'input_roster': plan['inputs'] = dict(list(plan['inputs'].items())[1:])
    elif damage == 'parent_audit': Path(plan['parent_audit']['path']).write_bytes(b'changed')
    elif damage == 'parent_manifest': Path(plan['parent_publication_manifest']['path']).write_bytes(b'changed')
    else:
        def reject(*args): raise ValueError('original parent not closed')
        monkeypatch.setattr(study.publication, 'authenticate', reject)
    if damage.startswith('qualification_') and damage != 'qualification_log':
        write_json(qpath, q); plan['qualification'] = absolute_pin(qpath)
    write_json(path, plan)
    with pytest.raises(ValueError): study.authenticate(path)


def test_exact_input_roster_includes_attempts_and_separate_failed_probe_checkpoint():
    inputs = study.expected_input_names()
    assert len(inputs) == 61
    assert {'parent/results.json', 'parent/resources.json', 'parent/fits.json',
            'parent/prediction-attempts.json', 'diagnostic/observer_learned-8103-lr0/final.npz'} <= inputs
    assert len([n for n in inputs if n.startswith('data/')]) == 11
    assert len([n for n in inputs if n.startswith('parent/') and n.endswith('/final.npz')]) == 36
    assert not any('22H_58M' in n for n in inputs)


def test_loader_checks_pin_before_decoding_and_preserves_fortran_layout(tmp_path, monkeypatch):
    name = 'parent/normalizers.npz'; path = tmp_path/'fabricated.npz'
    array = np.asfortranarray(np.arange(30, dtype=np.float32).reshape(6, 5))
    np.savez_compressed(path, coefficient=array)
    plan = {'inputs': {name: absolute_pin(path)}}
    loaded = study.load(plan, name)['coefficient']
    assert loaded.flags.f_contiguous and not loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, array)
    path.write_bytes(b'changed invalid bytes')
    def forbidden(*args, **kwargs): raise AssertionError('decoder reached before byte pin')
    monkeypatch.setattr(study.np, 'load', forbidden)
    with pytest.raises(ValueError): study.load(plan, name)
    with pytest.raises(ValueError): study.load(plan, 'data/official-test.mat')


def test_inherited_window_geometry_and_boundary_torque_alignment():
    t = np.arange(300, dtype=np.float64)[:, None]
    record = {'q': np.repeat(t, 6, axis=1), 'u': np.repeat(t+10000., 6, axis=1)}
    batch = study.old.window_batch([record], {'record': np.array([0]), 'start': np.array([64])}, 32, 128)
    np.testing.assert_array_equal(batch['q_context'][0, :, 0], np.arange(64, 96))
    np.testing.assert_array_equal(batch['u_context'][0, :, 0], np.arange(10064, 10096))
    np.testing.assert_array_equal(batch['future_u'][0, :, 0], np.arange(10095, 10223))
    np.testing.assert_array_equal(batch['target'][0, :, 0], np.arange(96, 224))
    assert list(study.old.dev_windows(3636, study.config())) == list(64+160*np.arange(22))


def test_current_parent_fit_ledger_overrides_same_named_grandparent_input(tmp_path, monkeypatch):
    """A repeated parent/fits.json key must never bind the older study's ledger."""
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    current_folder = tmp_path/study.publication.STUDY
    inputs, inventory = {}, {}
    current_names = {'parent/'+n for n in ('results.json', 'resources.json', 'fits.json', 'prediction-attempts.json')}
    current_names |= {'parent/'+r['key']+'/final.npz' for r in study.inherited_identities()
                     if r['arm'] not in REFERENCES}
    current_names.add('diagnostic/observer_learned-8103-lr0/final.npz')
    for i, name in enumerate(sorted(study.expected_input_names())):
        if name in current_names:
            relative = name.removeprefix('parent/').removeprefix('diagnostic/')
            path = current_folder/relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('current audited payload '+name).encode()); inventory[relative] = descriptor(path)
        else:
            path = tmp_path/'older'/str(i); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('opaque shared payload '+name).encode()); inputs[name] = absolute_pin(path)
    older_fits = tmp_path/'older'/'fits.json'; older_fits.write_bytes(b'wrong grandparent fit ledger')
    inputs['parent/fits.json'] = absolute_pin(older_fits)
    prior = {'plan': {'inputs': inputs}, 'inventory': inventory, 'audit': {'results': {
        'result': {'status': 'OBSERVER_DEVELOPMENT_FAIL'},
        'selection': {'selected_rates': {'local_affine': .003, 'temporal_affine': .001, 'observer_learned': None}}}}}
    actual = study.expected_inputs(prior)
    assert len(actual) == 61
    assert actual['parent/fits.json'] == absolute_pin(current_folder/'fits.json')
    assert actual['parent/fits.json'] != inputs['parent/fits.json']
    for name in current_names:
        relative = name.removeprefix('parent/').removeprefix('diagnostic/')
        assert actual[name] == absolute_pin(current_folder/relative)


def test_seven_probe_schedule_has_no_rate_or_evaluation_choice():
    expected = [{'key': f'{origin}-{seed}-batch0', 'seed': seed, 'batch_index': 0, 'gain_origin': origin}
                for seed in SEEDS for origin in ('identity', 'position')]
    expected += [{'key': 'archived-8103-batch22', 'seed': 8103, 'batch_index': 22, 'gain_origin': 'archived'}]
    assert study.probe_schedule() == expected
    assert len({r['key'] for r in expected}) == 7
    assert all('learning_rate' not in r and 'recording' not in r for r in expected)


def toy_gain_model():
    """An algebra fixture, without a robot transition or scientific checkpoint."""
    import torch
    model = torch.nn.Module()
    model.cell = torch.nn.Module()
    model.cell.frozen = torch.nn.Parameter(torch.zeros(590), requires_grad=False)
    model.gain = torch.nn.Parameter(torch.zeros(12, 6))
    return model


def fake_inference(model, batch, prefix):
    prefix.update(prefix_steps=30, max_state_abs=2., max_state_norm64=3.,
                  max_innovation_abs=1., max_innovation_norm64=2.)
    return model.gain.sum().expand(1, 128, 6)


def test_probe_preserves_preclip_gradient_and_never_constructs_optimizer(tmp_path, monkeypatch):
    import torch
    model = toy_gain_model(); before = study.confirmation.parameter_hash(model)
    batch = {'target': np.full((1, 128, 6), -1., dtype=np.float64)}
    monkeypatch.setattr(study, 'infer_diagnostic', fake_inference)
    def forbidden(*args, **kwargs): raise AssertionError('optimizer constructed in no-update probe')
    monkeypatch.setattr(study.torch.optim, 'Adam', forbidden)
    visits = []
    record = study.probe_one(model, batch, study.probe_schedule()[0], tmp_path/'probe', lambda: visits.append('check'))
    assert record['status'] == 'PASS' and record['outcome'] == 'finite' and record['optimizer_steps'] == 0
    assert record['parameters_unchanged'] and record['parameter_hash_before'] == record['parameter_hash_after'] == before
    assert record['prefix']['prefix_steps'] == 30
    assert record['gradient']['numel'] == 72 and record['gradient']['nonfinite_count'] == 0
    assert record['gradient']['max_abs'] == pytest.approx(2.)
    assert record['gradient']['norm64'] == pytest.approx(2*np.sqrt(72.))
    assert record['native_norm']['kind'] == 'finite'
    assert visits == ['check', 'check'] and all(p.grad is None for p in model.parameters())
    with np.load(tmp_path/'probe/preclip.npz', allow_pickle=False) as bank:
        assert set(bank.files) == {'gain_gradient', 'native_norm'}
        np.testing.assert_allclose(bank['gain_gradient'], np.full((12, 6), 2., dtype=np.float32), rtol=1e-6)
        assert bank['native_norm'].shape == ()
    assert all(torch.count_nonzero(p) == 0 for p in model.parameters())


@pytest.mark.parametrize('failure', ['forward', 'native_norm', 'programming'])
def test_probe_failures_retain_original_evidence_without_retry(tmp_path, monkeypatch, failure):
    import torch
    model = toy_gain_model(); visits = []
    batch = {'target': np.full((1, 128, 6), -1., dtype=np.float64)}
    original = study.old.FitFailure('fabricated numerical output') if failure == 'forward' else ValueError('programming defect')
    def forward(m, b, p):
        visits.append('forward')
        if failure in ('forward', 'programming'): raise original
        return fake_inference(m, b, p)
    monkeypatch.setattr(study, 'infer_diagnostic', forward)
    if failure == 'native_norm':
        def clip(*args, **kwargs):
            assert kwargs == {'error_if_nonfinite': False}
            visits.append('clip'); return torch.tensor(float('inf'), dtype=torch.float32)
        monkeypatch.setattr(study.torch.nn.utils, 'clip_grad_norm_', clip)
    if failure == 'programming':
        with pytest.raises(ValueError) as caught:
            study.probe_one(model, batch, study.probe_schedule()[1], tmp_path/'probe', lambda: None)
        assert caught.value is original
    else:
        study.probe_one(model, batch, study.probe_schedule()[1], tmp_path/'probe', lambda: None)
    record = json.loads((tmp_path/'probe/receipt.json').read_text())
    assert record['status'] == ('FATAL' if failure == 'programming' else 'FAILED')
    assert record['optimizer_steps'] == 0 and record['parameters_unchanged']
    assert visits == (['forward', 'clip'] if failure == 'native_norm' else ['forward'])
    with np.load(tmp_path/'probe/preclip.npz', allow_pickle=False) as bank:
        if failure == 'native_norm':
            assert record['outcome'] == 'native_norm_overflow'
            assert record['gradient']['all_finite'] and np.isfinite(bank['gain_gradient']).all()
            assert record['native_norm'] == {'kind': 'positive_infinity', 'value': None}
            assert np.isposinf(bank['native_norm'])
        else:
            assert bank.files == [] and record['gradient'] is None and record['native_norm'] is None


@pytest.mark.parametrize('number,kind', [(3., 'finite'), (float('inf'), 'positive_infinity'),
                                       (-float('inf'), 'negative_infinity'), (float('nan'), 'nan')])
def test_native_norm_record_does_not_silently_repair_nonfinite_values(number, kind):
    import torch
    expected = {'kind': kind, 'value': number if kind == 'finite' else None}
    assert study.native_norm_record(torch.tensor(number, dtype=torch.float32)) == expected


def test_diagnostic_inference_passes_only_public_inputs_and_one_collector():
    import torch
    seen = []
    class Spy:
        def condition(self, q, u, diagnostics=None):
            assert q.dtype == u.dtype == torch.float32 and q.shape == u.shape == (1, 32, 6)
            seen.append(('condition', q.clone(), u.clone(), diagnostics))
            return torch.zeros(1, 12)
        def __call__(self, future, state):
            assert future.dtype == torch.float32 and future.shape == (1, 128, 6)
            seen.append(('forecast', future.clone(), state))
            return torch.zeros(1, 128, 6), state
    batch = {'q_context': np.ones((1, 32, 6)), 'u_context': np.full((1, 32, 6), 2.),
             'future_u': np.full((1, 128, 6), 3.), 'target': object()}
    prefix = {}; result = study.infer_diagnostic(Spy(), batch, prefix)
    assert result.shape == (1, 128, 6) and len(seen) == 2 and seen[0][-1] is prefix


@pytest.mark.parametrize('arm,parameters,buffers', [(CANDIDATE, 662, 0), (FIXED, 590, 72)])
def test_new_storage_charges_gain_whether_parameter_or_fixed_buffer(arm, parameters, buffers):
    from types import SimpleNamespace
    def array(n): return SimpleNamespace(numel=lambda: n, element_size=lambda: 4)
    model = SimpleNamespace(parameters=lambda: [array(parameters)], buffers=lambda: [array(buffers)] if buffers else [])
    result = study.resource_model(model, arm)
    assert result['parameters'] == parameters and result['buffer_bytes'] == buffers*4
    assert result['state_bytes'] == 48 and result['normalizer_bytes'] == 192
    assert sum(result[k] for k in ('parameter_bytes', 'buffer_bytes', 'state_bytes', 'normalizer_bytes')) == 2888
    assert result['input_bytes'] == 9216 and result['output_bytes'] == 6144
    assert result['temporary_workspace'] == 'not measured'


def test_diagnostic_aggregate_keeps_partial_prefix_and_distinguishes_failure_modes():
    records = [
        {'prefix': {'prefix_steps': 30, 'max_state_abs': 3., 'max_state_norm64': 5.,
                    'max_innovation_abs': 1., 'max_innovation_norm64': 2.},
         'gradient': {'all_finite': True, 'max_abs': 4., 'norm64': 5.},
         'native_norm': {'kind': 'finite', 'value': 5.}},
        {'prefix': {'prefix_steps': 30, 'max_state_abs': 4., 'max_state_norm64': 6.,
                    'max_innovation_abs': 2., 'max_innovation_norm64': 3.},
         'gradient': {'all_finite': True, 'max_abs': 1e20, 'norm64': 8e20},
         'native_norm': {'kind': 'positive_infinity', 'value': None}},
        {'prefix': {'prefix_steps': 30}, 'gradient': {'all_finite': False, 'max_abs': None, 'norm64': None},
         'native_norm': {'kind': 'nan', 'value': None}},
        {'prefix': {'prefix_steps': 4, 'max_state_abs': 9.}, 'gradient': None, 'native_norm': None}]
    before = copy.deepcopy(records)
    assert study.aggregate_diagnostics(records) == {
        'attempts': 4, 'backward_calls': 3, 'clip_calls': 3, 'nonfinite_gradient_attempts': 1,
        'native_norm_nonfinite_attempts': 2, 'prefix_steps': 94,
        'max_gradient_abs': 1e20, 'max_gradient_norm64': 8e20, 'max_state_abs': 9.,
        'max_state_norm64': 6., 'max_innovation_abs': 2., 'max_innovation_norm64': 3.}
    assert records == before
    empty = study.aggregate_diagnostics([])
    assert empty['attempts'] == empty['backward_calls'] == empty['clip_calls'] == empty['prefix_steps'] == 0
    assert empty['max_gradient_abs'] is empty['max_gradient_norm64'] is None
    assert empty['max_state_abs'] == empty['max_innovation_abs'] == 0.


@pytest.mark.parametrize('fault', [None, 'second_forward', 'programming'])
def test_toy_trainer_preserves_every_attempt_and_gain_only_adam(tmp_path, monkeypatch, fault):
    model = toy_gain_model(); frozen = study.cell_hash(model); visits = []
    batch = {'target': np.full((1, 128, 6), -1., dtype=np.float64)}
    monkeypatch.setattr(study.old, 'window_batch', lambda *args: batch)
    error = ValueError('unrelated schema defect')
    def infer(m, b, p):
        visits.append('infer')
        if fault == 'programming': raise error
        if fault == 'second_forward' and len(visits) == 2:
            p.update(prefix_steps=3, max_state_abs=8.)
            raise study.old.FitFailure('fabricated forward failure')
        return fake_inference(m, b, p)
    monkeypatch.setattr(study, 'infer_diagnostic', infer)
    cfg = {**study.config(), 'updates': 2}
    kwargs = {'cfg': cfg, 'lr': .001, 'folder': tmp_path/'fit'}
    if fault == 'programming':
        with pytest.raises(ValueError) as caught: study.train_one(model, [], {'record': np.zeros((2, 1), np.int64)}, **kwargs)
        assert caught.value is error
    else:
        study.train_one(model, [], {'record': np.zeros((2, 1), np.int64)}, **kwargs)
    folder = tmp_path/'fit'; receipt = json.loads((folder/'fit-receipt.json').read_text())
    expected_status, completed = ('PASS', 2) if fault is None else ('FAILED', 1) if fault == 'second_forward' else ('FATAL', 0)
    assert receipt['status'] == expected_status and receipt['completed_updates'] == completed
    assert study.cell_hash(model) == frozen and all(p.grad is None for p in model.cell.parameters())
    assert set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json',
                                     'diagnostics.json', 'diagnostic-summary.json', 'last-gradient.npz'}
    trace = json.loads((folder/'trace.json').read_text())
    diagnostics = json.loads((folder/'diagnostics.json').read_text())
    aggregate = json.loads((folder/'diagnostic-summary.json').read_text())
    assert len(trace) == completed and len(diagnostics) == len(visits)
    assert aggregate['attempts'] == len(visits) and aggregate['backward_calls'] == completed
    assert diagnostics[-1]['status'] == expected_status
    with np.load(folder/'optimizer.npz', allow_pickle=False) as optimizer:
        expected = {'gain/'+suffix for suffix in ('step', 'exp_avg', 'exp_avg_sq')} if completed else set()
        assert set(optimizer.files) == expected
        if completed: assert float(optimizer['gain/step']) == completed
    with np.load(folder/'last-gradient.npz', allow_pickle=False) as gradient:
        assert set(gradient.files) == ({'gain_gradient', 'native_norm'} if fault is None else set())


def test_all_scheduled_probes_and_failed_fits_still_precede_first_exposed_decode(tmp_path, monkeypatch):
    """Lifecycle fakes stop exactly at the first exposed-array boundary."""
    from types import SimpleNamespace
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    cfg, full_rows = fixture_rows()
    input_paths = {}
    parent_fits = [{**r, 'effective_status': 'PASS', 'resources': {}}
                   for r in study.inherited_identities() if r['arm'] not in REFERENCES]
    payloads = {
        'parent/results.json': {'rows': [r for r in full_rows if r['arm'] not in (CANDIDATE, FIXED)]},
        'parent/prediction-attempts.json': [{'opaque_parent_attempt': i} for i in range(208)],
        'parent/fits.json': parent_fits,
        'parent/causal_ridge_1.json': {'fit_rows': 1}, 'parent/causal_ridge_100.json': {'fit_rows': 1}}
    for name, value in payloads.items():
        path = tmp_path/'inputs'/name; write_json(path, value); input_paths[name] = absolute_pin(path)
    registration = tmp_path/'registration.json'; write_json(registration, {})
    plan = {'sources': {}, 'inputs': input_paths}
    monkeypatch.setattr(study, 'authenticate', lambda path: (plan, 'fabricated-sha'))
    monkeypatch.setattr(study, 'SOURCES', ())
    bank = {'record': np.zeros((4096, 16), np.int64), 'start': np.zeros((4096, 16), np.int64)}
    events = []
    class AtExposedBoundary(RuntimeError): pass
    def load(plan, name):
        if name.startswith('data/'):
            if name.removeprefix('data/') in cfg['partitions']['exposed']:
                barrier = json.loads((tmp_path/'out/checkpoint-barrier.json').read_text())
                assert barrier['fresh_fit_attempts'] == 6 and barrier['fixed_models'] == 3
                assert barrier['inherited_models'] == 36 and len(barrier['checkpoints']) == 45
                assert barrier['exposed_loads_this_run'] == 0
                assert len([e for e in events if e.startswith('probe:')]) == 7
                assert len([e for e in events if e.startswith('fit:')]) == 6
                raise AtExposedBoundary('intentional stop before measured decoding')
            return {'q': np.zeros((300, 6)), 'u': np.zeros((300, 6))}
        if name == 'parent/normalizers.npz': return {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
        if name == 'parent/linear.npz': return {'coefficient': np.zeros((6, 25))}
        if 'batches-' in name: return bank
        if 'causal_ridge_' in name: return {f'h{h:03d}': np.zeros((1, 6)) for h in range(1, 129)}
        return {'cell.frozen': np.zeros(590, np.float32)}
    monkeypatch.setattr(study, 'load', load)
    monkeypatch.setattr(study.old, 'normalized_record', lambda record, norm: record)
    monkeypatch.setattr(study.old, 'make_batches', lambda *args: bank)
    monkeypatch.setattr(study.old, 'window_batch', lambda *args: {})
    monkeypatch.setattr(study, 'CausalRobotRidge', lambda *args: SimpleNamespace())
    def model_for(*args):
        model = toy_gain_model()
        model.load_state_dict = lambda *args, **kwargs: None
        return model
    monkeypatch.setattr(study, 'model_for', model_for)
    monkeypatch.setattr(study, 'resource_model', lambda *args: {})
    def probe(model, batch, identity, folder, check):
        events.append('probe:'+identity['key']); return {**identity, 'outcome': 'fabricated'}
    def train(model, records, batches, **kwargs):
        folder = kwargs['folder']; folder.mkdir()
        np.savez_compressed(folder/'final.npz', **study.old.weights(model))
        events.append('fit:'+folder.name)
        return {'status': 'FAILED', 'completed_updates': 0}
    monkeypatch.setattr(study, 'probe_one', probe)
    monkeypatch.setattr(study, 'train_one', train)
    # Inherited final copies are opaque and must exist, but never get decoded.
    for identity in study.inherited_identities():
        if identity['arm'] not in REFERENCES:
            name = 'parent/'+identity['key']+'/final.npz'
            path = tmp_path/'inputs'/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'opaque inherited final'); input_paths[name] = absolute_pin(path)
    with pytest.raises(AtExposedBoundary): study.run(registration, tmp_path/'out')
    fits = json.loads((tmp_path/'out/fits.json').read_text())
    assert len(fits) == 45 and sum(r['effective_status'] == 'FAILED' for r in fits) == 6
    assert all(e.startswith('probe:') for e in events[:7]) and all(e.startswith('fit:') for e in events[7:])
    assert json.loads((tmp_path/'out/failure.json').read_text())['type'] == 'AtExposedBoundary'


@pytest.mark.parametrize('damage', ['source', 'qualification', 'uncommitted', 'preexisting'])
def test_launcher_rejects_before_original_job_record_or_child(tmp_path, monkeypatch, damage):
    import launch_robot_position_observer_study as launch
    registration, plan, qpath, q, _ = admission_fixture(tmp_path, monkeypatch)
    copy_path = tmp_path/study.LAUNCHER
    copy_path.write_bytes(Path(launch.__file__).read_bytes())
    plan['launcher'] = descriptor(copy_path); q['launcher'] = plan['launcher']
    engineering = tmp_path/'engineering'; engineering.mkdir()
    output = tmp_path/'out'
    if damage == 'source': (tmp_path/study.SOURCES[0]).write_bytes(b'changed')
    elif damage == 'qualification': q['status'] = 'FAILED'
    elif damage == 'preexisting': output.mkdir()
    write_json(qpath, q); plan['qualification'] = absolute_pin(qpath); write_json(registration, plan)
    monkeypatch.setattr(launch, 'ROOT', tmp_path); monkeypatch.setattr(launch, 'REGISTRATION', registration)
    monkeypatch.setattr(launch, 'ENGINEERING', engineering); monkeypatch.setattr(launch, 'OUTPUT', output)
    def git(command, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']: return 'f'*40
        assert command[:2] == ['git', 'show']
        name = command[2].split(':', 1)[1]
        return b'uncommitted' if damage == 'uncommitted' else (tmp_path/name).read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError('child launched after failed metadata gate')
    monkeypatch.setattr(launch.subprocess, 'check_output', git)
    monkeypatch.setattr(launch.subprocess, 'run', forbidden)
    with pytest.raises(ValueError): launch.main()
    assert not (engineering/'run-launch-01.json').exists()
    assert not (engineering/'run-process-01.json').exists()
