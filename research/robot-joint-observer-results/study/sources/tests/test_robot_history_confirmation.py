"""Fabricated fixed-confirmation contracts; no reserved files or model fits."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_history_confirmation as study

PRIMARY = ('last_two', 'local_affine', 'temporal_affine')
CACHED = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ARMS = (*PRIMARY, *CACHED)
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS = (8101, 8102, 8103)
RATES = {'last_two': .001, 'local_affine': .003, 'temporal_affine': .003, 'dense_bounded': .001,
         'dense_unbounded': .003, 'gru32': .003, 'legacy_instant': .001, 'gru10': .003}
CONFIRM = ('recording_2021_12_15_22H_41M.mat', 'recording_2021_12_15_22H_50M.mat')
CONDITIONS = ('primary_recipes_complete', 'equal_file_mean_5pct_vs_both_locals',
              'each_file_within_2pct_best_local', 'latency_within_125pct_last_two',
              'complete_frontier_not_dominated')


def metric(error, horizon, scalars=120):
    return {'standardized_rmse': error, 'standardized_sse': error**2 * scalars, 'scalars': scalars,
            'physical_rmse_deg': error, 'per_joint_rmse_deg': [error] * 6, 'windows': 1, 'horizon': horizon}


def rows_fixture():
    rows = []
    for name in CONFIRM:
        for arm in (*ARMS, *REFS):
            for seed in SEEDS if arm in ARMS else (None,):
                for horizon in (64, 128):
                    rows.append({'recording': name, 'arm': arm, 'seed': seed,
                                 'learning_rate': RATES.get(arm), 'horizon': horizon, 'status': 'PASS',
                                 'metrics': metric(.8 if arm == 'temporal_affine' else 1., horizon), 'error': None})
    return rows


def resources_fixture():
    counts = dict(zip(ARMS, (590, 962, 962, 806, 806, 5916, 1014, 1296), strict=True))
    resources = []
    for arm in (*ARMS, *REFS):
        for seed in SEEDS if arm in ARMS else (None,):
            count = counts.get(arm, 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0)
            size = 4 if arm in ARMS else 8
            state = {'gru32': 50, 'gru10': 28}.get(arm, 12) if arm in ARMS else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
            seconds = 1.25 if arm == 'temporal_affine' else 1. if arm in ARMS else .1
            resources.append({'arm': arm, 'seed': seed, 'learning_rate': RATES.get(arm),
                              'status': 'PASS', 'error': None,
                              'parameters': count, 'parameter_bytes': count*size, 'state_scalars': state,
                              'state_bytes': state*size, 'buffer_bytes': 16 if arm in REFS[:2] else 0,
                              'normalizer_bytes': 192, 'input_bytes': 9216, 'output_bytes': 6144,
                              'timing': {'seconds': [seconds]*20, 'median_seconds': seconds, 'p95_seconds': seconds}})
    return resources


def flags(result):
    return [condition['passed'] for condition in result['conditions']]


def set_error(rows, arm, value, recording=None):
    for row in rows:
        if row['arm'] == arm and (recording is None or row['recording'] == recording):
            row['metrics'] = metric(value, row['horizon'])


def test_fixed24_checkpoints_and_complete112_row_confirmation_never_reselect(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('confirmation may not select or refit')
    monkeypatch.setattr(study.history, 'select', forbidden)
    monkeypatch.setattr(study.history, 'evaluate_rule', forbidden)
    rows, resources = rows_fixture(), resources_fixture()
    cfg = study.config()
    assert tuple(study.ALL_ARMS) == ARMS and tuple(study.PRIMARY) == PRIMARY
    assert tuple(study.REFERENCES) == REFS and study.FIXED_RATES == RATES
    assert cfg['partitions']['confirm'] == list(CONFIRM)
    assert len(rows) == 112 and len(rows)//2 == 56 and len(resources) == 28
    assert len({(r['arm'], r['seed']) for r in rows if r['arm'] in ARMS}) == 24
    result = study.fixed_rule(rows, resources, cfg)
    assert tuple(c['name'] for c in result['conditions']) == CONDITIONS
    assert flags(result) == [True]*5 and result['passed'] == result['total'] == 5


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_rate', 'wrong_seed', 'new_rate', 'dev_name', 'official_test'])
def test_metric_identity_is_exact_and_does_not_admit_a_better_alternative_recipe(damage):
    rows = rows_fixture()
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif damage == 'wrong_rate': rows[0]['learning_rate'] = .003
    elif damage == 'wrong_seed': rows[0]['seed'] = 8104
    elif damage == 'new_rate': rows[0]['learning_rate'] = .0001
    elif damage == 'dev_name': rows[0]['recording'] = 'recording_2021_12_15_21H_54M.mat'
    else: rows[0]['recording'] = 'recording_2021_12_15_22H_58M.mat'
    with pytest.raises(ValueError): study.fixed_rule(rows, resources_fixture(), study.config())


@pytest.mark.parametrize('error,passed', [(.95, True), (.950001, False)])
def test_inclusive_5pct_gain_boundary(error, passed):
    rows = rows_fixture(); set_error(rows, 'temporal_affine', error)
    result = study.fixed_rule(rows, resources_fixture(), study.config())
    assert flags(result) == [True, passed, True, True, True]


@pytest.mark.parametrize('control', PRIMARY[:2])
def test_zero_error_tie_is_valid_metric_but_not_relative_improvement(control):
    rows = rows_fixture(); set_error(rows, control, 0.); set_error(rows, 'temporal_affine', 0.)
    result = study.fixed_rule(rows, resources_fixture(), study.config())
    assert flags(result)[0] is True and flags(result)[1] is False


@pytest.mark.parametrize('error,passed', [(1.02, True), (1.020001, False)])
def test_better_second_file_cannot_hide_first_file_harm(error, passed):
    rows = rows_fixture(); set_error(rows, 'temporal_affine', error, CONFIRM[0]); set_error(rows, 'temporal_affine', .6, CONFIRM[1])
    result = study.fixed_rule(rows, resources_fixture(), study.config())
    assert flags(result) == [True, True, passed, True, True]


@pytest.mark.parametrize('seconds,passed', [(1.25, True), (1.250001, False)])
def test_complete_request_latency_margin(seconds, passed):
    resources = resources_fixture()
    for item in resources:
        if item['arm'] == 'temporal_affine': item['timing']['median_seconds'] = seconds
    result = study.fixed_rule(rows_fixture(), resources, study.config())
    assert flags(result) == [True, True, True, passed, True]


@pytest.mark.parametrize('axis', ['error', 'time', 'bytes', 'tie', 'tradeoff'])
def test_frontier_compares_all_three_axes_without_promoting_exact_ties(axis):
    rows, resources = rows_fixture(), resources_fixture()
    set_error(rows, 'dense_bounded', .79 if axis == 'error' else .81 if axis == 'tradeoff' else .8)
    for item in resources:
        if item['arm'] == 'dense_bounded':
            item['parameter_bytes'] = 3844 if axis in ('bytes', 'tradeoff') else 3848
            item['timing']['median_seconds'] = 1.24 if axis in ('time', 'tradeoff') else 1.25
    result = study.fixed_rule(rows, resources, study.config())
    assert flags(result)[-1] is (axis in ('tie', 'tradeoff'))


@pytest.mark.parametrize('arm', (*CACHED, *REFS))
def test_invalid_control_remains_in_roster_and_blocks_complete_frontier(arm):
    rows, resources = rows_fixture(), resources_fixture()
    for row in rows:
        if row['arm'] == arm: row.update(status='FAILED', metrics=None, error={'type': 'NonfiniteEvaluation'})
    for item in resources:
        if item['arm'] == arm: item.update(status='FAILED', error={'type': 'NonfiniteTiming'}, timing=None)
    result = study.fixed_rule(rows, resources, study.config())
    assert len(rows) == 112 and flags(result) == [True, True, True, True, False]


def test_failed_primary_stays_failed_without_fallback_to_unselected_checkpoint():
    rows, resources = rows_fixture(), resources_fixture()
    for row in rows:
        if row['arm'] == 'temporal_affine': row.update(status='FAILED', metrics=None, error={'type': 'NonfiniteEvaluation'})
    for item in resources:
        if item['arm'] == 'temporal_affine': item.update(status='FAILED', error={'type': 'NonfiniteTiming'}, timing=None)
    assert flags(study.fixed_rule(rows, resources, study.config())) == [False]*5


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True])
def test_invalid_latency_cannot_satisfy_a_favorable_compute_comparison(value):
    resources = resources_fixture()
    next(r for r in resources if r['arm'] == 'temporal_affine')['timing']['median_seconds'] = value
    result = study.fixed_rule(rows_fixture(), resources, study.config())
    assert flags(result)[3:] == [False, False]


@pytest.mark.parametrize('arm', REFS[:2])
def test_both_ridge_banks_enter_frontier_without_confirmation_recipe_selection(arm):
    rows, resources = rows_fixture(), resources_fixture()
    set_error(rows, arm, .7)
    # A hypothetical small ridge tests the rule's three-axis comparison; the
    # independent resource admission separately enforces actual bank storage.
    item = next(r for r in resources if r['arm'] == arm)
    item['parameter_bytes'] = 0; item['state_bytes'] = 0; item['buffer_bytes'] = 0
    result = study.fixed_rule(rows, resources, study.config())
    assert flags(result) == [True, True, True, True, False]
    assert arm in result['dominators']


def test_equal_file_rule_does_not_silently_change_to_pooled_sse():
    rows = rows_fixture()
    for row in rows:
        if row['arm'] == 'temporal_affine':
            error, count = (1., 1) if row['recording'] == CONFIRM[0] else (.8, 10000)
            row['metrics'] = metric(error, row['horizon'], count)
    result = study.fixed_rule(rows, resources_fixture(), study.config())
    assert result['equal_file_means']['temporal_affine'] == pytest.approx(.9)
    assert flags(result) == [True]*5


def test_causal_windows_use_boundary_torque_and_fixed22_starts_without_refitting(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('confirmation must reuse inherited statistics and references')
    monkeypatch.setattr(study.old, 'normalizers', forbidden)
    monkeypatch.setattr(study.old, 'solve_ridge', forbidden)
    q = np.arange(3636, dtype=np.float64)[:, None] * 10 + np.arange(6)[None, :]
    u = q + 100000.
    record = {'name': CONFIRM[0], 'q': q, 'u': u, 'raw_indices': np.arange(3636, dtype=np.int64)*25}
    norm = {'q_mean': np.arange(6, dtype=np.float64), 'q_std': np.full(6, 10.),
            'u_mean': np.arange(6, dtype=np.float64) + 100000., 'u_std': np.full(6, 10.)}
    before = {name: value.copy() for name, value in norm.items()}
    batch, physical, starts = study.make_windows(record, norm, study.config())
    np.testing.assert_array_equal(starts, 64 + 160*np.arange(22, dtype=np.int64))
    assert starts[-1] == 3424 and batch['q_context'].shape == (22, 32, 6)
    assert batch['target'].shape == batch['future_u'].shape == (22, 128, 6)
    for i, start in enumerate(starts):
        np.testing.assert_array_equal(batch['q_context'][i, :, 0], start + np.arange(32))
        np.testing.assert_array_equal(batch['future_u'][i, :, 0], start + 31 + np.arange(128))
        np.testing.assert_array_equal(batch['target'][i, :, 0], start + 32 + np.arange(128))
    np.testing.assert_array_equal(physical['q_context'][0], q[64:96])
    np.testing.assert_array_equal(physical['future_u'][0], u[95:223])
    for name, value in before.items(): np.testing.assert_array_equal(norm[name], value)
    changed = {**record, 'q': q.copy()}; changed['q'][96:] += 1e6
    altered, _, _ = study.make_windows(changed, norm, study.config())
    for key in ('q_context', 'u_context', 'future_u'):
        np.testing.assert_array_equal(altered[key][0], batch[key][0])
    assert not np.array_equal(altered['target'][0], batch['target'][0])


@pytest.mark.parametrize('damage', ['short', 'float32', 'nonfinite', 'raw_gap', 'zero_scale', 'wrong_name'])
def test_window_builder_rejects_bad_recording_without_trimming_or_imputation(damage):
    record = {'name': CONFIRM[0], 'q': np.zeros((3636, 6)), 'u': np.ones((3636, 6)),
              'raw_indices': np.arange(3636, dtype=np.int64)*25}
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    if damage == 'short': record['q'] = record['q'][:-1]
    elif damage == 'float32': record['u'] = record['u'].astype(np.float32)
    elif damage == 'nonfinite': record['q'][100, 0] = np.inf
    elif damage == 'raw_gap': record['raw_indices'][100] += 1
    elif damage == 'zero_scale': norm['q_std'][0] = 0
    else: record['name'] = 'official_TEST.mat'
    with pytest.raises(ValueError): study.make_windows(record, norm, study.config())


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unknown'])
def test_all28_cost_attempts_required_even_when_forecasts_fail(damage):
    resources = resources_fixture()
    if damage == 'missing': resources.pop()
    elif damage == 'duplicate': resources[-1] = copy.deepcopy(resources[0])
    else: resources[-1]['arm'] = 'unregistered_reference'
    with pytest.raises(ValueError): study.fixed_rule(rows_fixture(), resources, study.config())


def test_family_latency_uses_three_seed_median_and_complete_resident_bytes():
    resources = resources_fixture()
    for item, seconds in zip([r for r in resources if r['arm'] == 'temporal_affine'], (1., 1.2, 100.), strict=True):
        item['timing']['median_seconds'] = seconds
    result = study.fixed_rule(rows_fixture(), resources, study.config())
    assert result['costs']['temporal_affine'] == {'latency': 1.2, 'bytes': 4088}
    assert result['costs']['last_two']['bytes'] == 2600
    assert flags(result) == [True]*5


def public_batch():
    return {'q_context': np.zeros((22, 32, 6)), 'u_context': np.ones((22, 32, 6)),
            'future_u': np.full((22, 128, 6), 2.), 'target': object()}


def test_prediction_boundary_never_supplies_future_targets():
    batch = public_batch(); seen = []
    forecast = np.zeros((22, 128, 6), dtype=np.float64)
    def predict(inputs):
        seen.append(inputs)
        assert set(inputs) == {'q_context', 'u_context', 'future_u'}
        return forecast
    result, error = study.prediction_attempt(predict, batch)
    assert result is forecast and error is None and len(seen) == 1
    assert all(seen[0][k] is batch[k] for k in seen[0])


@pytest.mark.parametrize('message', study.NUMERIC_ERRORS)
def test_known_numerical_forecast_failure_retains_both_horizons(message):
    def fail(inputs): raise study.old.FitFailure(message)
    prediction, error = study.prediction_attempt(fail, public_batch())
    assert prediction is None and error == {'type': 'NonfiniteEvaluation', 'message': message}
    rows = study.old.scored_rows({'arm': 'temporal_affine'}, prediction, np.zeros((22, 128, 6)),
                                 np.ones(6), study.config(), error)
    assert [r['horizon'] for r in rows] == [64, 128]
    assert all(r['status'] == 'FAILED' and r['metrics'] is None for r in rows)


def test_known_ridge_overflow_is_retained_without_relabeling_all_value_errors():
    def fail(inputs): raise ValueError('nonfinite ridge prediction')
    prediction, error = study.prediction_attempt(fail, public_batch())
    assert prediction is None and error == {'type': 'NonfiniteEvaluation', 'message': 'nonfinite ridge prediction'}
    rows = study.old.scored_rows({'arm': 'causal_ridge_1'}, prediction, np.zeros((22, 128, 6)),
                                 np.ones(6), study.config(), error)
    assert [r['horizon'] for r in rows] == [64, 128] and all(r['status'] == 'FAILED' for r in rows)


def test_returned_nonfinite_forecast_is_retained_as_raw_failure_witness():
    forecast = np.zeros((22, 128, 6)); forecast[0, 0, 0] = np.nan
    prediction, error = study.prediction_attempt(lambda batch: forecast, public_batch())
    assert prediction is forecast and error is None
    rows = study.old.scored_rows({}, prediction, np.zeros_like(forecast), np.ones(6), study.config())
    assert len(rows) == 2 and all(r['status'] == 'FAILED' for r in rows)


@pytest.mark.parametrize('error', [ValueError('schema defect'), study.old.FitFailure('unrecognized failure')])
def test_prediction_does_not_relabel_programming_errors(error):
    def fail(inputs): raise error
    with pytest.raises(type(error)) as caught: study.prediction_attempt(fail, public_batch())
    assert caught.value is error


@pytest.mark.parametrize('forecast', [np.zeros((1, 128, 6)), np.zeros((22, 128, 6), dtype=np.float32), None])
def test_prediction_shape_and_dtype_are_not_repaired(forecast):
    with pytest.raises(ValueError): study.prediction_attempt(lambda inputs: forecast, public_batch())


def test_timing_passes_complete_physical_request_and_fixed_repetition_budget(monkeypatch):
    predict, physical, norm, cfg = object(), {'public': object()}, {'fixed': object()}, study.config()
    seen = []
    timing = {'seconds': [1.]*20, 'median_seconds': 1., 'p95_seconds': 1.}
    def timed(*args):
        seen.append(args)
        assert args[3]['timing_warmups'] == 3 and args[3]['timing_repeats'] == 20
        return timing
    monkeypatch.setattr(study.history, 'timed_request', timed)
    assert study.timing_attempt(predict, physical, norm, cfg) == {'status': 'PASS', 'error': None, 'timing': timing}
    assert seen == [(predict, physical, norm, cfg)]


@pytest.mark.parametrize('error', [study.old.FitFailure(study.NUMERIC_ERRORS[0]), ValueError('finite complete timed request'),
                                 ValueError('nonfinite ridge prediction')])
def test_known_timing_failure_is_explicit_attempt_without_retries(monkeypatch, error):
    calls = []
    def fail(*args): calls.append(args); raise error
    monkeypatch.setattr(study.history, 'timed_request', fail)
    row = study.timing_attempt(None, {}, {}, study.config())
    assert len(calls) == 1 and row['status'] == 'FAILED' and row['timing'] is None
    assert row['error'] == {'type': 'NonfiniteTiming', 'message': str(error)}


def test_unrelated_timing_error_stays_fatal(monkeypatch):
    error = ValueError('wrong schema')
    def fail(*args): raise error
    monkeypatch.setattr(study.history, 'timed_request', fail)
    with pytest.raises(ValueError) as caught: study.timing_attempt(None, {}, {}, study.config())
    assert caught.value is error


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    blob = path.read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def descriptor(path):
    blob = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def admission_fixture(tmp_path, monkeypatch):
    """Opaque fake source/data bytes, never actual recordings or valid NPZ."""
    root = tmp_path / 'repo'
    parent = root / 'output/robot-history-initialization-study-v1'
    engineering = root / 'output/robot-history-initialization-engineering-v1'
    audit = root / 'output/robot-history-initialization-audit-v1/audit.json'
    for key, value in (('ROOT', root), ('PARENT', parent), ('PARENT_ENGINEERING', engineering), ('PARENT_AUDIT', audit)):
        monkeypatch.setattr(study, key, value)
    for name in study.old.THREADS: monkeypatch.setenv(name, '1')
    sources = {name: write(root/name, ('fabricated source '+name).encode()) for name in study.SOURCES}
    payloads = {}
    for name in study.PAYLOADS:
        write(parent/name, b'opaque-not-npz-or-measurements'); payloads[name] = descriptor(parent/name)
    paths = {'manifest': parent/'manifest.json', 'receipt': parent/'receipt.json', 'process': engineering/'run-process-01.json',
             'audit': audit, 'audit_process': engineering/'audit-process-01.json'}
    for path in paths.values(): write(path, {'opaque': True})
    write(paths['manifest'], {'files': {name: {k: item[k] for k in ('sha256', 'bytes')} for name, item in payloads.items()}})
    parent_auth = {'plan': {'sources': {n: sources[n] for n in study.history.SOURCES}},
                   'audit': {'results': {'selection': {'selected_rates': dict(RATES)},
                                        'result': {'passed': 5, 'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'}}},
                   'inputs': {str(p): {k: descriptor(p)[k] for k in ('sha256', 'bytes')} for p in paths.values()}}
    import plot_robot_history_initialization as parent_plot
    monkeypatch.setattr(parent_plot, 'authenticate', lambda *args: parent_auth)
    launcher = write(root/'scripts/launch_robot_history_confirmation.py', b'fake launcher')
    commands = [['.venv/bin/ruff', 'check', *study.QUALIFICATION_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                 'tests/test_robot_history_confirmation.py', 'tests/test_audit_robot_history_confirmation.py']]
    qualification = {'status': 'PASS', 'sources_unchanged': True, 'sources': sources, 'launcher': launcher,
                     'thread_env': dict.fromkeys(study.old.THREADS, '1'), 'commands': []}
    for i, command in enumerate(commands):
        path = root/f'qualification-{i}.log'; pin = write(path, b'fabricated original command log')
        qualification['commands'].append({'command': command, 'returncode': 0, 'log': str(path), 'sha256': pin['sha256']})
    qualification_path = root/'qualification.json'; write(qualification_path, qualification)
    archive = root/'output/robot-data-engineering-v1/raw-bundle-attempt-01.rar'
    write(archive, b'not an archive; opaque fake admission fixture')
    monkeypatch.setattr(study, 'ARCHIVE_SHA', descriptor(archive)['sha256'])
    inputs = root/'output/robot-history-confirmation-inputs-v1'
    raw = {}
    for name in CONFIRM:
        path = inputs/'raw_data'/name; write(path, b'not MAT; no numerical data'); raw[name] = descriptor(path)
    extraction = {'scope': 'Opaque extraction and hashing only; no numeric decoding',
                  'command': ['/usr/bin/bsdtar', '-xf', str(archive), '-C', str(inputs), *['raw_data/'+n for n in CONFIRM]],
                  'returncode': 0, 'elapsed_seconds': .25, 'archive': descriptor(archive), 'raw_recordings': raw}
    extraction_path = root/'output/robot-history-confirmation-engineering-v1/raw-extraction-01.json'
    write(extraction_path, extraction)
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'parent_registration_sha256': study.PARENT_SHA, 'parent_closure': {n: descriptor(p) for n, p in paths.items()},
            'parent_payloads': payloads, 'launcher': launcher, 'qualification': descriptor(qualification_path),
            'archive': descriptor(archive), 'extraction': descriptor(extraction_path), 'raw_recordings': raw}
    registration = root/'registration.json'; write(registration, plan)
    def forbidden(*args, **kwargs): pytest.fail('admission must not decode arrays or recordings')
    monkeypatch.setattr(study.np, 'load', forbidden)
    monkeypatch.setattr(study, 'load_recording', forbidden)
    return SimpleNamespace(root=root, plan=plan, path=registration, parent=parent_auth, qualification=qualification,
                           qualification_path=qualification_path, extraction=extraction, extraction_path=extraction_path)


def test_complete_opaque_admission_has36_sources30_selected_payloads_and_no_decoding(tmp_path, monkeypatch):
    fixture = admission_fixture(tmp_path, monkeypatch)
    plan, sha = study.authenticate(fixture.path)
    assert len(plan['sources']) == 36 and len(plan['parent_payloads']) == 30
    assert plan == fixture.plan and sha == descriptor(fixture.path)['sha256']
    finals = {n for n in plan['parent_payloads'] if n.endswith('/final.npz')}
    assert len(finals) == 24
    assert finals == {f'{a}-{s}-lr{int(RATES[a] == .003)}/final.npz' for a in ARMS for s in SEEDS}


@pytest.mark.parametrize('damage', ['source', 'checkpoint', 'checkpoint_identity', 'payload_missing', 'parent_manifest',
                                  'parent_process', 'parent_audit', 'parent_rate', 'parent_gate', 'qualification_status',
                                  'qualification_source', 'qualification_argv', 'qualification_log', 'launcher',
                                  'raw_name', 'raw_alias', 'raw_bytes', 'archive', 'extraction_argv', 'extraction_status',
                                  'extraction_scope', 'extraction_join', 'config_rate', 'official_test', 'threads'])
def test_admission_tampering_fails_before_first_array_decode(tmp_path, monkeypatch, damage):
    f = admission_fixture(tmp_path, monkeypatch); plan = f.plan
    if damage == 'source': (f.root/study.SOURCES[0]).write_bytes(b'changed source')
    elif damage == 'checkpoint': Path(next(v['path'] for k, v in plan['parent_payloads'].items() if k.endswith('/final.npz'))).write_bytes(b'changed weights')
    elif damage == 'checkpoint_identity':
        name = next(k for k in plan['parent_payloads'] if k.endswith('/final.npz'))
        plan['parent_payloads'][name.replace('-lr0/', '-lr1/')] = plan['parent_payloads'].pop(name)
    elif damage == 'payload_missing': plan['parent_payloads'].pop('linear.npz')
    elif damage.startswith('parent_') and damage in ('parent_manifest', 'parent_process', 'parent_audit'):
        Path(plan['parent_closure'][damage.removeprefix('parent_')]['path']).write_bytes(b'changed closure')
    elif damage == 'parent_rate': f.parent['audit']['results']['selection']['selected_rates']['temporal_affine'] = .001
    elif damage == 'parent_gate': f.parent['audit']['results']['result']['passed'] = 4
    elif damage.startswith('qualification_'):
        if damage == 'qualification_status': f.qualification['status'] = 'FAILED'
        elif damage == 'qualification_source': f.qualification['sources'] = {**plan['sources'], study.SOURCES[0]: {'sha256': '0'*64, 'bytes': 1}}
        elif damage == 'qualification_argv': f.qualification['commands'][1]['command'] = ['.venv/bin/python', '-c', 'pass']
        else: Path(f.qualification['commands'][0]['log']).write_bytes(b'changed log')
        write(f.qualification_path, f.qualification); plan['qualification'] = descriptor(f.qualification_path)
    elif damage == 'launcher': (f.root/'scripts/launch_robot_history_confirmation.py').write_bytes(b'changed launcher')
    elif damage in ('raw_name', 'official_test'):
        plan['raw_recordings'] = {**plan['raw_recordings']}
        plan['raw_recordings']['recording_2021_12_15_22H_58M.mat' if damage == 'official_test' else 'unknown.mat'] = plan['raw_recordings'].pop(CONFIRM[0])
    elif damage == 'raw_alias':
        alias = f.root/'alias.mat'; write(alias, b'not MAT; no numerical data')
        plan['raw_recordings'][CONFIRM[0]] = descriptor(alias)
        f.extraction['raw_recordings'] = plan['raw_recordings']
        write(f.extraction_path, f.extraction); plan['extraction'] = descriptor(f.extraction_path)
    elif damage == 'raw_bytes': Path(plan['raw_recordings'][CONFIRM[0]]['path']).write_bytes(b'changed raw fake')
    elif damage == 'archive': Path(plan['archive']['path']).write_bytes(b'changed archive fake')
    elif damage.startswith('extraction_'):
        if damage == 'extraction_argv': f.extraction['command'][-1] = 'raw_data/official_TEST.mat'
        elif damage == 'extraction_status': f.extraction['returncode'] = 1
        elif damage == 'extraction_scope': f.extraction['scope'] = 'already numerically decoded'
        else: f.extraction['archive'] = {**plan['archive'], 'sha256': '0'*64}
        write(f.extraction_path, f.extraction); plan['extraction'] = descriptor(f.extraction_path)
    elif damage == 'config_rate': plan['config']['fixed_rates']['last_two'] = .003
    else: monkeypatch.setenv(study.old.THREADS[0], '2')
    write(f.path, plan)
    with pytest.raises(ValueError): study.authenticate(f.path)


@pytest.mark.parametrize('name', ['fit-data.npz', 'dev-data.npz', 'initial.npz', 'unknown/final.npz'])
def test_inherited_loader_has_no_fit_dev_or_unselected_checkpoint_interface(monkeypatch, name):
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('undeclared decode'))
    with pytest.raises(ValueError): study.load_parent_arrays({'parent_payloads': {}}, name)


def test_inherited_checkpoint_copy_preserves_storage_order_and_ownership(tmp_path, monkeypatch):
    path = tmp_path/'opaque.npz'; write(path, b'opaque fixture only')
    original = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    class Archive:
        files = ('weights',)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key): assert key == 'weights'; return original
    calls = []
    def load(filename, **kwargs): calls.append((filename, kwargs)); return Archive()
    monkeypatch.setattr(study.np, 'load', load)
    name = f'last_two-{SEEDS[0]}-lr0/final.npz'
    plan = {'parent_payloads': {name: descriptor(path)}}
    result = study.load_parent_arrays(plan, name)['weights']
    assert result.flags.f_contiguous and not np.shares_memory(result, original)
    np.testing.assert_array_equal(result, original)
    assert calls == [(str(path), {'allow_pickle': False})]
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): study.load_parent_arrays(plan, name)
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['recording_2021_12_15_22H_58M.mat', 'unknown.mat'])
def test_raw_loader_rejects_official_test_or_unknown_name_before_pin_or_decode(monkeypatch, name):
    monkeypatch.setattr(study, '_pin', lambda *a: pytest.fail('unregistered raw hash'))
    monkeypatch.setattr(study, 'load_recording', lambda *a, **k: pytest.fail('unregistered raw decode'))
    with pytest.raises(ValueError): study.load_confirmation({'raw_recordings': dict.fromkeys(CONFIRM)}, name)


@pytest.mark.parametrize('damage', [None, 'partition', 'preprocessing', 'source', 'name'])
def test_confirmation_loader_checks_explicit_access_and_saved_identity(tmp_path, monkeypatch, damage):
    path = tmp_path/'fake.mat'; write(path, b'opaque fixture')
    pin = descriptor(path)
    record = SimpleNamespace(name=CONFIRM[0], partition='confirm', q=object(), torque=object(), raw_indices=object(),
                             pins={'source_sha256': pin['sha256'], 'source_bytes': pin['bytes'],
                                   'preprocessing_sha256': study.PREPROCESSING_SHA256})
    if damage == 'partition': record.partition = 'dev'
    elif damage == 'preprocessing': record.pins['preprocessing_sha256'] = '0'*64
    elif damage == 'source': record.pins['source_bytes'] += 1
    elif damage == 'name': record.name = CONFIRM[1]
    calls = []
    def load(*args, **kwargs): calls.append((args, kwargs)); return record
    monkeypatch.setattr(study, 'load_recording', load)
    plan = {'raw_recordings': dict.fromkeys(CONFIRM, pin)}
    if damage:
        with pytest.raises(ValueError): study.load_confirmation(plan, CONFIRM[0])
    else:
        result = study.load_confirmation(plan, CONFIRM[0])
        assert result == {'name': CONFIRM[0], 'q': record.q, 'u': record.torque, 'raw_indices': record.raw_indices}
    assert calls == [((path, CONFIRM[0]), {'allow_confirmation': True})]
