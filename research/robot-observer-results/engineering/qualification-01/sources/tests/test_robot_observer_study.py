"""Fabricated observer harness contracts, without measured files or fit calls."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_observer_study as study

LEARNED = ('local_affine', 'temporal_affine', 'observer_learned')
FIXED = ('last_two', 'observer_fixed', 'observer_zero')
CACHED = ('joint_local_affine', 'joint_temporal_affine', 'dense_bounded',
          'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
FAMILIES = (*LEARNED, *FIXED, *CACHED, *REFERENCES)
CONDITIONS = ('complete_forecasts_and_costs', 'equal_four_file_mean_5pct_vs_best_control',
              'each_file_within_2pct_best_simple', 'latency_within_150pct_last_two',
              'complete_frontier_not_dominated')
SEEDS = (8101, 8102, 8103)
RATES = (.001, .003)


def metrics(value, horizon):
    count = 22*horizon*6
    return {'standardized_rmse': value, 'standardized_sse': value * value * count,
            'scalars': count, 'physical_rmse_deg': value,
            'per_joint_rmse_deg': [value] * 6, 'windows': 22, 'horizon': horizon}


def fixture_rows():
    cfg = study.config()
    # Identities only supply serialized labels. Scientific roster and every
    # score are assembled independently below, not from producer calculations.
    labels = {(r['arm'], r['seed'], r['learning_rate']): r for r in study.identities()}
    recipes = [(a, s, r) for a in LEARNED for s in SEEDS for r in RATES]
    recipes += [(a, s, None) for a in FIXED for s in SEEDS]
    recipes += [(a, s, cfg['cached_rates'][a]) for a in CACHED for s in SEEDS]
    recipes += [(a, None, None) for a in REFERENCES]
    rows = []
    for name in cfg['partitions']['exposed']:
        for arm, seed, rate in recipes:
            item = labels[arm, seed, rate]
            for horizon in (64, 128):
                rows.append({'fit_key': item['key'], 'arm': arm, 'seed': seed,
                             'learning_rate': rate, 'origin': item['origin'],
                             'recording': name, 'horizon': horizon, 'status': 'PASS',
                             'error': None, 'metrics': metrics(.75 if arm == 'observer_learned' else 1., horizon)})
    return cfg, rows


def fixture_resources():
    cfg = study.config()
    rows = []
    for item in study.identities():
        arm = item['arm']
        if arm in LEARNED and item['learning_rate'] != .001:
            continue
        # Literal complete storage includes fixed and zero gain buffers.
        count = {'last_two': 590, 'local_affine': 962, 'temporal_affine': 962,
                 'observer_fixed': 590, 'observer_zero': 590, 'observer_learned': 662}.get(arm, 1000)
        seconds = 1.5 if arm == 'observer_learned' else 1.
        rows.append({'key': item['key'], 'arm': arm, 'seed': item['seed'],
                     'learning_rate': item['learning_rate'], 'origin': item['origin'],
                     'status': 'PASS', 'error': None, 'parameters': count,
                     'parameter_bytes': count * 4, 'buffer_bytes': 288 if arm in FIXED[1:] else 0,
                     'state_bytes': 48, 'normalizer_bytes': 192,
                     'timing': {'seconds': [seconds] * 20,
                                'median_seconds': seconds, 'p95_seconds': seconds}})
    assert cfg['seeds'] == list(SEEDS)
    return rows


def set_error(rows, arm, value, *, recording=None, rate=None, horizon=None):
    for row in rows:
        if (row['arm'] == arm and (recording is None or row['recording'] == recording)
                and (rate is None or row['learning_rate'] == rate)
                and (horizon is None or row['horizon'] == horizon)):
            row['metrics'] = metrics(value, row['horizon'])


def rule(rows, resources=None):
    cfg = study.config()
    return study.evaluate_rule(rows, study.select(rows, cfg),
                               fixture_resources() if resources is None else resources, cfg)


def flags(result):
    return [r['passed'] for r in result['conditions']]


def test_exact_rosters_and_independent_complete_witness():
    cfg, rows = fixture_rows()
    identities = study.identities()
    assert len(identities) == len({r['key'] for r in identities}) == 52
    assert len([r for r in identities if r['arm'] in LEARNED]) == 18
    assert len([r for r in identities if r['arm'] in FIXED]) == 9
    assert len([r for r in identities if r['arm'] in CACHED]) == 21
    assert len([r for r in identities if r['arm'] in REFERENCES]) == 4
    assert {r['arm'] for r in identities} == set(FAMILIES)
    assert cfg['learning_rates'] == list(RATES) and cfg['updates'] == 4096
    assert cfg['batch_size'] == 16 and cfg['context'] == 32
    assert cfg['train_horizon'] == cfg['dev_horizon'] == 128
    assert len(cfg['partitions']['dev']) == 2 and len(cfg['partitions']['exposed']) == 4
    assert len(rows) == 416 and len(fixture_resources()) == 43
    result = rule(rows)
    assert tuple(r['name'] for r in result['conditions']) == CONDITIONS
    assert result['passed'] == result['total'] == 5 and flags(result) == [True] * 5
    assert result['status'] == 'OBSERVER_DEVELOPMENT_PASS'


@pytest.mark.parametrize('value,expected', [(.95, True), (.950001, False)])
def test_global_gain_boundary_is_inclusive(value, expected):
    _, rows = fixture_rows()
    set_error(rows, 'observer_learned', value)
    assert flags(rule(rows)) == [True, expected, True, True, True]


@pytest.mark.parametrize('control', tuple(a for a in FAMILIES if a != 'observer_learned'))
def test_every_declared_control_can_prevent_global_gain(control):
    _, rows = fixture_rows()
    set_error(rows, control, .78125)  # .75 exceeds .95 * .78125.
    assert flags(rule(rows))[1] is False


def test_zero_error_tie_never_means_five_percent_relative_improvement():
    _, rows = fixture_rows()
    set_error(rows, 'observer_learned', 0.)
    set_error(rows, 'causal_ridge_100', 0.)
    assert flags(rule(rows))[:2] == [True, False]


@pytest.mark.parametrize('value,expected', [(1.02, True), (1.020001, False)])
def test_each_of_four_files_has_its_own_harm_guard(value, expected):
    cfg, rows = fixture_rows()
    set_error(rows, 'observer_learned', .5)
    set_error(rows, 'observer_learned', value, recording=cfg['partitions']['exposed'][-1])
    assert flags(rule(rows)) == [True, True, expected, True, True]


@pytest.mark.parametrize('simple', ('last_two', 'local_affine', 'temporal_affine', 'observer_fixed'))
def test_file_guard_uses_the_best_simple_control(simple):
    cfg, rows = fixture_rows()
    set_error(rows, simple, .5, recording=cfg['partitions']['exposed'][0])
    assert flags(rule(rows))[2] is False


def test_old_confirmation_files_and_h64_never_choose_learning_rate():
    cfg, rows = fixture_rows()
    for arm in LEARNED:
        set_error(rows, arm, .75, rate=.001)
        set_error(rows, arm, 1., rate=.003)
        for recording in set(cfg['partitions']['exposed']) - set(cfg['partitions']['dev']):
            set_error(rows, arm, 100., recording=recording, rate=.001)
            set_error(rows, arm, 0., recording=recording, rate=.003)
        set_error(rows, arm, 1000., rate=.001, horizon=64)
    selected = study.select(rows, cfg)
    assert all(selected['selected_rates'][arm] == .001 for arm in LEARNED)
    assert 'selected_ridge' not in selected


def test_selection_uses_pooled_sse_not_average_rmse_and_ties_by_rate():
    cfg, rows = fixture_rows()
    chosen = [r for r in rows if r['arm'] == 'observer_learned'
              and r['recording'] in cfg['partitions']['dev'] and r['horizon'] == 128]
    for row in chosen:
        row['metrics'] = metrics(.75 if row['learning_rate'] == .003 else 0., 128)
    first = next(row for row in chosen if row['learning_rate'] == .001)
    first['metrics'] = metrics(2., 128)
    # sqrt(4/6) > .75 despite unweighted mean 2/6 < .75.
    selected = study.select(rows, cfg)['selected_rates']
    assert selected['observer_learned'] == .003
    assert selected['local_affine'] == selected['temporal_affine'] == .001


@pytest.mark.parametrize('seconds,expected', [(1.5, True), (1.500001, False)])
def test_total_request_latency_margin_is_inclusive(seconds, expected):
    _, rows = fixture_rows()
    resources = fixture_resources()
    for row in resources:
        if row['arm'] == 'observer_learned':
            row['timing'] = {'seconds': [seconds] * 20, 'median_seconds': seconds, 'p95_seconds': seconds}
    assert flags(rule(rows, resources)) == [True, True, True, expected, True]


@pytest.mark.parametrize('axis', ['error', 'latency', 'bytes', 'tie', 'tradeoff'])
def test_frontier_uses_all_three_axes_and_a_strict_improvement(axis):
    _, rows = fixture_rows()
    resources = fixture_resources()
    set_error(rows, 'joint_temporal_affine', .5 if axis == 'error' else 1. if axis == 'tradeoff' else .75)
    for row in resources:
        if row['arm'] == 'joint_temporal_affine':
            row['parameter_bytes'] = 2644 if axis == 'bytes' else 2648
            row['buffer_bytes'] = 0
            seconds = 1. if axis == 'latency' else 1.5
            row['timing'] = {'seconds': [seconds] * 20, 'median_seconds': seconds, 'p95_seconds': seconds}
    result = rule(rows, resources)
    assert flags(result)[4] is (axis in ('tie', 'tradeoff'))
    assert result['dominators'] == ([] if axis in ('tie', 'tradeoff') else ['joint_temporal_affine'])


def test_selector_result_cannot_be_forged_after_development_selection():
    cfg, rows = fixture_rows()
    selection = study.select(rows, cfg)
    selection['selected_rates']['observer_learned'] = .003
    with pytest.raises(ValueError):
        study.evaluate_rule(rows, selection, fixture_resources(), cfg)


@pytest.mark.parametrize('control', FAMILIES)
def test_any_declared_family_failure_blocks_complete_eligibility(control):
    cfg, rows = fixture_rows()
    for row in rows:
        if row['arm'] == control:
            row.update(status='FAILED', metrics=None,
                       error={'type': 'NonfiniteEvaluation', 'message': 'fabricated numeric failure'})
    resources = fixture_resources()
    if control in LEARNED:
        for row in resources:
            if row['arm'] == control:
                row.update(key=f'{control}-{row["seed"]}-unavailable', origin='unavailable',
                           learning_rate=None, status='UNAVAILABLE', timing=None,
                           error={'type': 'UnavailableSelectedRecipe', 'message': 'fabricated'})
    result = study.evaluate_rule(rows, study.select(rows, cfg), resources, cfg)
    assert result['passed'] < 5 and flags(result)[0] is False


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_recording'])
def test_forecast_roster_or_metric_corruption_cannot_be_silently_dropped(damage):
    _, rows = fixture_rows()
    item = next(r for r in rows if r['arm'] == 'observer_zero' and r['horizon'] == 64)
    if damage == 'missing': rows.remove(item)
    elif damage == 'duplicate': rows.append(copy.deepcopy(item))
    else: item['recording'] = 'not-a-declared-recording'
    with pytest.raises(ValueError): rule(rows)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_rate', 'bad_status', 'bad_error'])
def test_complete_cost_roster_cannot_be_bypassed(damage):
    _, rows = fixture_rows()
    resources = fixture_resources()
    item = next(r for r in resources if r['arm'] == 'joint_temporal_affine')
    if damage == 'missing': resources.remove(item)
    elif damage == 'duplicate': resources.append(copy.deepcopy(item))
    elif damage == 'wrong_rate': item['learning_rate'] = .77
    elif damage == 'bad_status': item['status'] = 'UNKNOWN'
    else: item['error'] = {'type': 'Failure', 'message': 'not a PASS'}
    if damage in ('missing', 'duplicate', 'wrong_rate'):
        with pytest.raises(ValueError): rule(rows, resources)
    else:
        assert flags(rule(rows, resources))[0] is False


@pytest.mark.parametrize('damage', ['failed', 'nan'])
def test_selected_h64_failure_is_visible_even_if_every_h128_passes(damage):
    _, rows = fixture_rows()
    item = next(r for r in rows if r['arm'] == 'observer_zero' and r['horizon'] == 64)
    if damage == 'failed':
        item.update(status='FAILED', metrics=None, error={'type': 'NonfiniteEvaluation', 'message': 'fabricated'})
    else:
        item['metrics']['standardized_rmse'] = float('nan')
    assert flags(rule(rows)) == [False, True, True, True, True]


def test_unselected_bad_rate_stays_in_roster_without_invalidating_selected_recipe():
    cfg, rows = fixture_rows()
    for row in rows:
        if row['arm'] in LEARNED and row['learning_rate'] == .003:
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    assert len(rows) == 416
    assert set(study.select(rows, cfg)['selected_rates'].values()) == {.001}
    assert flags(rule(rows)) == [True] * 5


def test_all_failed_learned_rates_leave_three_explicit_unavailable_cost_slots():
    cfg, rows = fixture_rows()
    for row in rows:
        if row['arm'] == 'observer_learned':
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    selection = study.select(rows, cfg)
    assert selection['selected_rates']['observer_learned'] is None
    costs = study.resource_identities(selection)
    assert len(costs) == 43
    assert [r for r in costs if r['arm'] == 'observer_learned'] == [
        {'key': f'observer_learned-{seed}-unavailable', 'arm': 'observer_learned',
         'seed': seed, 'learning_rate': None, 'origin': 'unavailable'} for seed in SEEDS]


@pytest.mark.parametrize('mode,shapes', [('observer_learned', {'gain': (12, 6)}),
                                      ('local_affine', {'head.weight': (12, 30), 'head.bias': (12,)})])
@pytest.mark.parametrize('damage', [None, 'cell_slot', 'missing', 'step', 'shape', 'dtype', 'nan', 'trainable_cell'])
def test_optimizer_evidence_only_has_initializer_owned_complete_moments(mode, shapes, damage):
    params = {name: SimpleNamespace(requires_grad=True, shape=shape) for name, shape in shapes.items()}
    params['cell.raw_matrix'] = SimpleNamespace(requires_grad=False, shape=(2, 12, 12))
    model = SimpleNamespace(named_parameters=lambda: params.items())
    arrays = {name+'/'+suffix: np.zeros(shape, dtype=np.float32)
              for name, shape in shapes.items() for suffix in ('exp_avg', 'exp_avg_sq')}
    arrays.update({name+'/step': np.array(4096., dtype=np.float32) for name in shapes})
    first = next(iter(shapes))
    if damage == 'cell_slot': arrays['cell.raw_matrix/step'] = np.array(4096., dtype=np.float32)
    elif damage == 'missing': del arrays[first+'/exp_avg']
    elif damage == 'step': arrays[first+'/step'][...] = 4095
    elif damage == 'shape': arrays[first+'/exp_avg'] = np.zeros((1,), dtype=np.float32)
    elif damage == 'dtype': arrays[first+'/exp_avg'] = arrays[first+'/exp_avg'].astype(np.float64)
    elif damage == 'nan': arrays[first+'/exp_avg'].flat[0] = np.nan
    elif damage == 'trainable_cell': params['cell.raw_matrix'].requires_grad = True
    if damage is None:
        study.check_optimizer(model, arrays, 4096)
    else:
        with pytest.raises(ValueError): study.check_optimizer(model, arrays, 4096)


@pytest.mark.parametrize('damage', [None, 'hash', 'requires_grad', 'stored_gradient'])
def test_frozen_cell_guard_checks_hash_and_gradient_ownership(monkeypatch, damage):
    parameter = SimpleNamespace(requires_grad=damage == 'requires_grad',
                                grad=object() if damage == 'stored_gradient' else None)
    model = SimpleNamespace(cell=SimpleNamespace(parameters=lambda: [parameter]))
    monkeypatch.setattr(study, 'cell_hash', lambda m: 'different' if damage == 'hash' else 'expected')
    if damage is None: study.check_frozen(model, 'expected')
    else:
        with pytest.raises(ValueError): study.check_frozen(model, 'expected')


@pytest.mark.parametrize('mode,parameters,buffers,total', [
    ('last_two', 590, 0, 2600), ('local_affine', 962, 0, 4088),
    ('temporal_affine', 962, 0, 4088), ('observer_learned', 662, 0, 2888),
    ('observer_fixed', 590, 72, 2888), ('observer_zero', 590, 72, 2888)])
def test_logical_storage_charges_frozen_cell_and_even_zero_gain_buffer(mode, parameters, buffers, total):
    def value(n): return SimpleNamespace(numel=lambda: n, element_size=lambda: 4)
    model = SimpleNamespace(parameters=lambda: [value(parameters)], buffers=lambda: [value(buffers)] if buffers else [])
    result = study.resource_model(model, mode)
    assert result['parameters'] == parameters and result['buffer_bytes'] == 4*buffers
    assert result['state_scalars'] == 12 and result['state_bytes'] == 48
    assert sum(result[k] for k in ('parameter_bytes', 'buffer_bytes', 'state_bytes', 'normalizer_bytes')) == total
    assert result['input_bytes'] == 9216 and result['output_bytes'] == 6144
    assert result['temporary_workspace'] == 'not measured'


@pytest.mark.parametrize('message', ['nonfinite structured transition output; no repair',
    'finite CPU tensor with exact shape/dtype: observer innovation', 'programming-error'])
def test_numerical_failure_translation_is_exact_not_a_general_valueerror_catch(message):
    error = ValueError(message)
    def fail(): raise error
    expected = study.old.FitFailure if message != 'programming-error' else ValueError
    with pytest.raises(expected) as caught: study.ObserverAdapter.numeric(fail)
    if message == 'programming-error': assert caught.value is error
    else: assert caught.value.__cause__ is error


def test_inherited_window_geometry_and_public_only_torque_alignment():
    t = np.arange(300, dtype=np.float64)[:, None]
    record = {'q': np.repeat(t, 6, axis=1), 'u': np.repeat(1000+t, 6, axis=1)}
    batch = study.old.window_batch([record], {'record': np.array([0]), 'start': np.array([64])}, 32, 128)
    np.testing.assert_array_equal(batch['q_context'][0, :, 0], np.arange(64, 96))
    np.testing.assert_array_equal(batch['u_context'][0, :, 0], np.arange(1064, 1096))
    np.testing.assert_array_equal(batch['future_u'][0, :, 0], np.arange(1095, 1223))
    np.testing.assert_array_equal(batch['target'][0, :, 0], np.arange(96, 224))
    assert list(study.old.dev_windows(3636, study.config())) == list(64+160*np.arange(22))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def descriptor(path):
    blob = path.read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def absolute_pin(path):
    return {'path': str(path.resolve()), **descriptor(path)}


def admission_fixture(tmp_path, monkeypatch):
    """Opaque fake sources and inputs; inherited publication admission is stubbed."""
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    for name in study.SOURCES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('fabricated source ' + name).encode())
    launcher = tmp_path/study.LAUNCHER
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_bytes(b'fabricated launcher')
    sources = {name: descriptor(tmp_path/name) for name in study.SOURCES}
    assert len(sources) == 45
    inputs = {}
    for i, name in enumerate(sorted(study.expected_input_names())):
        path = tmp_path/'fake-inputs'/str(i)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('invalid NPZ and never decoded: ' + name).encode())
        inputs[name] = absolute_pin(path)
    assert len(inputs) == 45
    parent_audit = tmp_path/study.publication.AUDIT
    parent_manifest = tmp_path/study.publication.OUTPUT/'manifest.json'
    write_json(parent_audit, {'status': 'PASS', 'agreement': True})
    write_json(parent_manifest, {'files': {}})
    commands = [
        ['.venv/bin/ruff', 'check', *[p for p in study.QUALIFICATION_SOURCES if p.endswith('.py')]],
        ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
         'tests/test_robot_observer_initializer.py', 'tests/test_robot_observer_study.py',
         'tests/test_audit_robot_observer_study.py']]
    completed = []
    for i, command in enumerate(commands):
        log = tmp_path/f'qualification-log-{i}.txt'
        log.write_bytes(b'fabricated closed passing qualification')
        completed.append({'command': command, 'returncode': 0, 'log': str(log),
                          'sha256': descriptor(log)['sha256'], 'seconds': 1.})
    qualification = {'status': 'PASS', 'sources_unchanged': True, 'sources': sources,
                     'launcher': descriptor(launcher), 'thread_env': dict.fromkeys(study.old.THREADS, '1'),
                     'commands': completed}
    qualification_path = tmp_path/'qualification.json'
    write_json(qualification_path, qualification)
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'launcher': descriptor(launcher), 'qualification': absolute_pin(qualification_path),
            'inputs': inputs, 'parent_publication_manifest': absolute_pin(parent_manifest),
            'parent_audit': absolute_pin(parent_audit)}
    registration = tmp_path/'registration.json'
    write_json(registration, plan)
    for variable in study.old.THREADS: monkeypatch.setenv(variable, '1')
    visits = []
    admitted = object()
    def parent(*args):
        assert args == (tmp_path/study.publication.STUDY, parent_audit, tmp_path/study.publication.ENGINEERING)
        visits.append('closed-parent-admission')
        return admitted
    def expected(prior):
        assert prior is admitted
        visits.append('parent-payload-roster')
        return inputs
    monkeypatch.setattr(study.publication, 'authenticate', parent)
    monkeypatch.setattr(study, 'expected_inputs', expected)
    def forbidden(*args, **kwargs): raise AssertionError('numeric decoder reached during admission')
    monkeypatch.setattr(study.np, 'load', forbidden)
    return registration, plan, qualification_path, qualification, visits


def test_admission_authenticates_sources_and_parent_before_any_decoder(tmp_path, monkeypatch):
    registration, plan, _, _, visits = admission_fixture(tmp_path, monkeypatch)
    actual, sha = study.authenticate(registration)
    assert actual == plan and sha == descriptor(registration)['sha256']
    assert visits == ['closed-parent-admission', 'parent-payload-roster']


@pytest.mark.parametrize('damage', ['source', 'source_roster', 'launcher', 'config', 'qualification_status',
    'qualification_sources', 'qualification_command', 'qualification_returncode', 'qualification_log',
    'qualification_source_drift', 'thread_env', 'input_bytes', 'input_roster', 'parent_audit',
    'parent_manifest', 'parent_not_closed'])
def test_admission_blocks_metadata_tampering_without_numeric_access(tmp_path, monkeypatch, damage):
    registration, plan, qualification_path, qualification, _ = admission_fixture(tmp_path, monkeypatch)
    if damage == 'source': (tmp_path/study.SOURCES[0]).write_bytes(b'changed')
    elif damage == 'source_roster': plan['sources'].pop(study.SOURCES[0])
    elif damage == 'launcher': (tmp_path/study.LAUNCHER).write_bytes(b'changed')
    elif damage == 'config': plan['config']['updates'] = 4095
    elif damage == 'qualification_status': qualification['status'] = 'FAILED'
    elif damage == 'qualification_sources': qualification['sources'] = {}
    elif damage == 'qualification_command': qualification['commands'][1]['command'][-1] = 'tests/unrelated.py'
    elif damage == 'qualification_returncode': qualification['commands'][1]['returncode'] = 1
    elif damage == 'qualification_log': Path(qualification['commands'][0]['log']).write_bytes(b'changed')
    elif damage == 'qualification_source_drift': qualification['sources_unchanged'] = False
    elif damage == 'thread_env': monkeypatch.setenv(study.old.THREADS[0], '2')
    elif damage == 'input_bytes': Path(next(iter(plan['inputs'].values()))['path']).write_bytes(b'changed')
    elif damage == 'input_roster': plan['inputs'] = dict(list(plan['inputs'].items())[1:])
    elif damage == 'parent_audit': Path(plan['parent_audit']['path']).write_bytes(b'changed')
    elif damage == 'parent_manifest': Path(plan['parent_publication_manifest']['path']).write_bytes(b'changed')
    else:
        def reject(*args): raise ValueError('parent original closure incomplete')
        monkeypatch.setattr(study.publication, 'authenticate', reject)
    if damage.startswith('qualification_') and damage != 'qualification_log':
        write_json(qualification_path, qualification)
        plan['qualification'] = absolute_pin(qualification_path)
    write_json(registration, plan)
    with pytest.raises(ValueError): study.authenticate(registration)


def test_registered_array_loader_preserves_memory_order_and_checks_pin_before_decode(tmp_path, monkeypatch):
    key = 'parent/normalizers.npz'
    path = tmp_path/'fabricated.npz'
    values = np.asfortranarray(np.arange(30, dtype=np.float32).reshape(6, 5))
    np.savez_compressed(path, coefficient=values)
    plan = {'inputs': {key: absolute_pin(path)}}
    loaded = study.load(plan, key)['coefficient']
    assert loaded.flags.f_contiguous and not loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, values)
    path.write_bytes(b'tampered opaque bytes')
    def forbidden(*args, **kwargs): raise AssertionError('decoder before pin validation')
    monkeypatch.setattr(study.np, 'load', forbidden)
    with pytest.raises(ValueError): study.load(plan, key)
    with pytest.raises(ValueError): study.load(plan, 'data/official-test.mat')


@pytest.mark.parametrize('damage', ['source', 'qualification', 'uncommitted', 'preexisting'])
def test_launcher_guard_blocks_child_and_original_launch_record(tmp_path, monkeypatch, damage):
    import launch_robot_observer_study as launch

    registration, plan, qualification_path, qualification, _ = admission_fixture(tmp_path, monkeypatch)
    launcher_copy = tmp_path/study.LAUNCHER
    launcher_copy.write_bytes(Path(launch.__file__).read_bytes())
    plan['launcher'] = descriptor(launcher_copy)
    qualification['launcher'] = plan['launcher']
    engineering = tmp_path/'engineering'
    engineering.mkdir()
    output = tmp_path/'new-study'
    if damage == 'source': (tmp_path/study.SOURCES[0]).write_bytes(b'changed')
    elif damage == 'qualification': qualification['status'] = 'FAILED'
    elif damage == 'preexisting': output.mkdir()
    write_json(qualification_path, qualification)
    plan['qualification'] = absolute_pin(qualification_path)
    write_json(registration, plan)
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(launch, 'REGISTRATION', registration)
    monkeypatch.setattr(launch, 'ENGINEERING', engineering)
    monkeypatch.setattr(launch, 'OUTPUT', output)
    def git(command, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']: return 'f'*40
        assert command[:2] == ['git', 'show']
        name = command[2].split(':', 1)[1]
        return b'not committed' if damage == 'uncommitted' else (tmp_path/name).read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError('child launched after failed admission')
    monkeypatch.setattr(launch.subprocess, 'check_output', git)
    monkeypatch.setattr(launch.subprocess, 'run', forbidden)
    with pytest.raises(ValueError): launch.main()
    assert not (engineering/'run-launch-01.json').exists()
    assert not (engineering/'run-process-01.json').exists()
