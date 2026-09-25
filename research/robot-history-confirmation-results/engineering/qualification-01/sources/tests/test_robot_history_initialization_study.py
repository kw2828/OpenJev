"""Synthetic integration contracts only; no measured data, checkpoints or fits."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_history_initialization_study as study

PRIMARY = ('last_two', 'local_affine', 'temporal_affine')
CACHED = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
CONDITIONS = ('primary_recipes_complete', 'equal_file_mean_5pct_vs_both_locals',
              'each_file_within_2pct_best_local', 'latency_within_125pct_last_two',
              'complete_frontier_not_dominated')


def metric(value, horizon, scalars=120):
    return {'standardized_rmse': value, 'standardized_sse': value**2 * scalars,
            'scalars': scalars, 'physical_rmse_deg': value,
            'per_joint_rmse_deg': [value] * 6, 'windows': 1, 'horizon': horizon}


def fixture_rows():
    cfg = study.config()
    rows = []
    for name in cfg['partitions']['dev']:
        for arm in (*PRIMARY, *CACHED):
            for seed in (8101, 8102, 8103):
                for rate in (.001, .003):
                    for horizon in (64, 128):
                        rows.append({'recording': name, 'arm': arm, 'seed': seed,
                                     'learning_rate': rate, 'horizon': horizon, 'status': 'PASS',
                                     'metrics': metric(.8 if arm == 'temporal_affine' else 1., horizon),
                                     'error': None})
        for arm in REFERENCES:
            for horizon in (64, 128):
                rows.append({'recording': name, 'arm': arm, 'seed': None,
                             'learning_rate': None, 'horizon': horizon, 'status': 'PASS',
                             'metrics': metric(1., horizon), 'error': None})
    return cfg, rows


def fixture_resources():
    rows = []
    for arm in (*PRIMARY, *CACHED):
        count = 962 if arm in ('local_affine', 'temporal_affine') else 590
        for seed in (8101, 8102, 8103):
            seconds = 1.25 if arm == 'temporal_affine' else 1.
            rows.append({'arm': arm, 'seed': seed, 'learning_rate': .001,
                         'parameters': count, 'parameter_bytes': count * 4,
                         'buffer_bytes': 0, 'state_bytes': 48, 'normalizer_bytes': 192,
                         'timing': {'seconds': [seconds] * 20, 'median_seconds': seconds,
                                    'p95_seconds': seconds}})
    rows.extend({'arm': arm, 'seed': None, 'learning_rate': None, 'parameters': 1,
                 'parameter_bytes': 8, 'buffer_bytes': 0, 'state_bytes': 48,
                 'normalizer_bytes': 192, 'timing': {'seconds': [.1] * 20,
                                                  'median_seconds': .1, 'p95_seconds': .1}}
                for arm in REFERENCES)
    return rows


def set_error(rows, arm, value, *, recording=None, rate=None):
    for row in rows:
        if (row['arm'] == arm and (recording is None or row['recording'] == recording)
                and (rate is None or row['learning_rate'] == rate)):
            row['metrics'] = metric(value, row['horizon'])


def rule(rows, resources=None):
    cfg = study.config()
    return study.evaluate_rule(rows, study.select(rows, cfg),
                               fixture_resources() if resources is None else resources, cfg)


def flags(result):
    return [row['passed'] for row in result['conditions']]


def test_fixed_rosters_and_complete_five_condition_witness():
    cfg, rows = fixture_rows()
    assert tuple(study.ARMS) == PRIMARY
    assert set(study.ALL_ARMS) == {*PRIMARY, *CACHED}
    assert tuple(study.REFERENCES) == REFERENCES
    assert cfg['seeds'] == [8101, 8102, 8103]
    assert cfg['learning_rates'] == [.001, .003]
    assert cfg['context'] == 32 and cfg['train_horizon'] == cfg['dev_horizon'] == 128
    assert cfg['updates'] == 4096 and cfg['batch_size'] == 16
    assert len(PRIMARY) * 3 * 2 == 18 and len(CACHED) * 3 * 2 == 30
    assert len(rows) == 208 and len(rows) // 2 == 104
    assert len(fixture_resources()) == 28
    assert len(PRIMARY) * 3 * 2 == 18 and len(PRIMARY) * 3 * 2 * 2 == 36
    selected = study.select(rows, cfg)
    assert set(selected['selected_rates'].values()) == {.001}
    assert 'selected_ridge' not in selected  # Both declared ridge banks enter the frontier.
    result = rule(rows)
    assert result['passed'] == result['total'] == 5
    assert tuple(item['name'] for item in result['conditions']) == CONDITIONS
    assert flags(result) == [True] * 5
    assert result['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'


@pytest.mark.parametrize('value, expected', [(.95, True), (.950001, False)])
def test_five_percent_gain_is_inclusive_and_not_rounded(value, expected):
    _, rows = fixture_rows()
    set_error(rows, 'temporal_affine', value)
    assert flags(rule(rows)) == [True, expected, True, True, True]


@pytest.mark.parametrize('comparison', ['last_two', 'local_affine'])
def test_gain_must_beat_both_primary_controls(comparison):
    _, rows = fixture_rows()
    set_error(rows, comparison, .83)
    assert flags(rule(rows))[1] is False  # .8 is not <= .95 * .83.


@pytest.mark.parametrize('comparison', ['last_two', 'local_affine'])
def test_zero_error_tie_is_valid_accuracy_but_not_a_relative_gain(comparison):
    cfg, rows = fixture_rows()
    set_error(rows, comparison, 0.)
    set_error(rows, 'temporal_affine', 0.)
    selected = study.select(rows, cfg)
    assert selected['selected_rates'][comparison] == .001
    assert selected['selected_rates']['temporal_affine'] == .001
    assert flags(rule(rows))[0] is True
    assert flags(rule(rows))[1] is False


@pytest.mark.parametrize('first_error, expected', [(1.02, True), (1.020001, False)])
def test_file_guard_inclusive_even_when_equal_file_mean_improves(first_error, expected):
    cfg, rows = fixture_rows()
    set_error(rows, 'temporal_affine', first_error, recording=cfg['partitions']['dev'][0])
    set_error(rows, 'temporal_affine', .6, recording=cfg['partitions']['dev'][1])
    assert flags(rule(rows)) == [True, True, expected, True, True]


def test_file_guard_uses_better_local_control_on_each_file():
    cfg, rows = fixture_rows()
    first, second = cfg['partitions']['dev']
    set_error(rows, 'local_affine', .6, recording=first)
    set_error(rows, 'temporal_affine', .65, recording=first)
    set_error(rows, 'temporal_affine', .2, recording=second)
    assert flags(rule(rows)) == [True, True, False, True, True]


@pytest.mark.parametrize('seconds, expected', [(1.25, True), (1.250001, False)])
def test_latency_uses_complete_request_and_inclusive_margin(seconds, expected):
    _, rows = fixture_rows()
    resources = fixture_resources()
    for row in resources:
        if row['arm'] == 'temporal_affine':
            row['timing']['median_seconds'] = seconds
    assert flags(rule(rows, resources)) == [True, True, True, expected, True]


@pytest.mark.parametrize('axis', ['error', 'latency', 'storage', 'none', 'tradeoff'])
def test_three_axis_dominance_requires_all_weak_and_one_strict(axis):
    _, rows = fixture_rows()
    resources = fixture_resources()
    set_error(rows, 'dense_bounded', .79 if axis == 'error' else (.81 if axis == 'tradeoff' else .8))
    for row in resources:
        if row['arm'] == 'dense_bounded':
            row['parameter_bytes'] = 3844 if axis in ('storage', 'tradeoff') else 3848
            row['timing']['median_seconds'] = 1.24 if axis in ('latency', 'tradeoff') else 1.25
    assert flags(rule(rows, resources)) == [True, True, True, True, axis in ('none', 'tradeoff')]


@pytest.mark.parametrize('arm', PRIMARY + CACHED)
def test_missing_entire_neural_control_cannot_create_success(arm):
    _, rows = fixture_rows()
    rows = [row for row in rows if row['arm'] != arm]
    result = rule(rows)
    assert result['passed'] < 5
    assert result['status'] != 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'


@pytest.mark.parametrize('arm', REFERENCES)
@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'failed'])
def test_incomplete_declared_reference_cannot_disappear_from_frontier(arm, damage):
    _, rows = fixture_rows()
    item = next(row for row in rows if row['arm'] == arm and row['horizon'] == 128)
    if damage == 'missing':
        rows.remove(item)
    elif damage == 'duplicate':
        rows.append(copy.deepcopy(item))
    else:
        item.update(status='FAILED', metrics=None)
    assert flags(rule(rows))[-1] is False


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'failed', 'nan', 'wrong_seed'])
def test_bad_recipe_excludes_whole_rate_not_just_bad_seed(damage):
    cfg, rows = fixture_rows()
    item = next(row for row in rows if row['arm'] == 'temporal_affine'
                and row['horizon'] == 128 and row['learning_rate'] == .001)
    if damage == 'missing':
        rows.remove(item)
    elif damage == 'duplicate':
        rows.append(copy.deepcopy(item))
    elif damage == 'failed':
        item.update(status='FAILED', metrics=None)
    elif damage == 'nan':
        item['metrics']['standardized_rmse'] = float('nan')
    else:
        item['seed'] = 999
    assert study.select(rows, cfg)['selected_rates']['temporal_affine'] == .003


def test_pooled_selection_uses_h128_sse_and_counts_not_mean_rmse_or_h64():
    cfg, rows = fixture_rows()
    chosen = [r for r in rows if r['arm'] == 'temporal_affine'
              and r['learning_rate'] == .001 and r['horizon'] == 128]
    for row in rows:
        if row['arm'] == 'temporal_affine':
            row['metrics'] = metric(.2, row['horizon'], 100)
    for row in chosen:
        row['metrics'] = metric(0., 128, 100)
    chosen[0]['metrics'] = metric(2., 128, 1)
    for row in rows:
        if row['arm'] == 'temporal_affine' and row['horizon'] == 64 and row['learning_rate'] == .001:
            row.update(status='FAILED', metrics=None)
    # sqrt(4/501) < .2, while the unweighted mean 2/6 > .2.
    assert study.select(rows, cfg)['selected_rates']['temporal_affine'] == .001


def test_each_ridge_requires_two_distinct_files_even_if_other_bank_is_complete():
    _, rows = fixture_rows()
    chosen = [r for r in rows if r['arm'] == 'causal_ridge_1' and r['horizon'] == 128]
    assert flags(rule(rows))[-1] is True
    chosen[1]['recording'] = chosen[0]['recording']
    assert flags(rule(rows))[-1] is False


@pytest.mark.parametrize('ridge', REFERENCES[:2])
def test_no_ridge_selection_can_hide_a_dominating_bank(ridge):
    _, rows = fixture_rows()
    set_error(rows, ridge, .7)
    result = rule(rows)
    assert flags(result) == [True, True, True, True, False]
    assert result['dominators'] == [ridge]


def test_forged_rate_selection_is_rejected():
    cfg, rows = fixture_rows()
    selected = study.select(rows, cfg)
    selected['selected_rates']['temporal_affine'] = .003
    with pytest.raises(ValueError, match='selection'):
        study.evaluate_rule(rows, selected, fixture_resources(), cfg)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_timing_cannot_be_a_favorable_cost(value):
    _, rows = fixture_rows()
    resources = fixture_resources()
    next(r for r in resources if r['arm'] == 'temporal_affine')['timing']['median_seconds'] = value
    assert flags(rule(rows, resources))[3:] == [False, False]


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_rate', 'negative_bytes'])
def test_comparator_cost_roster_cannot_be_bypassed(damage):
    _, rows = fixture_rows()
    resources = fixture_resources()
    item = next(r for r in resources if r['arm'] == 'dense_bounded')
    if damage == 'missing':
        resources.remove(item)
    elif damage == 'duplicate':
        resources.append(copy.deepcopy(item))
    elif damage == 'wrong_rate':
        item['learning_rate'] = .003
    else:
        item['parameter_bytes'] = -1
    assert flags(rule(rows, resources))[-1] is False


def test_reused_window_boundary_and_inference_never_read_future_positions():
    q = np.arange(1080., dtype=np.float64).reshape(180, 6)
    u = 10000. + q
    data = [{'q': q.copy(), 'u': u.copy()}]
    choices = {'record': np.array([0], dtype=np.int64), 'start': np.array([3], dtype=np.int64)}
    batch = study.old.window_batch(data, choices, 32, 128)
    assert np.array_equal(batch['q_context'][0], q[3:35])
    assert np.array_equal(batch['u_context'][0], u[3:35])
    assert np.array_equal(batch['future_u'][0], u[34:162])
    assert np.array_equal(batch['target'][0], q[35:163])
    class Spy:
        def condition(self, position, torque):
            assert position.shape[1:] == (32, 6) and torque.shape == position.shape
            return position[:, -1]
        def __call__(self, future, state):
            return future + state[:, None], state
    expected = study.old.infer(Spy(), batch).numpy()
    batch['target'] = object()
    assert np.array_equal(study.old.infer(Spy(), batch).numpy(), expected)
    data[0]['q'][35:] = np.nan
    modified = study.old.window_batch(data, choices, 32, 128)
    for key in ('q_context', 'u_context', 'future_u'):
        assert np.array_equal(modified[key], batch[key])


def test_fixed_permutation_pairs_older_inputs_preserves_boundary_and_local_features():
    q = np.arange(384., dtype=np.float64).reshape(2, 32, 6) / 100
    u = 4. + q
    future = np.full((2, 128, 6), 9.)
    marker = object()
    batch = {'q_context': q.copy(), 'u_context': u.copy(), 'future_u': future, 'target': marker}
    altered = study.permuted_batch(batch)
    order = list(range(29, -1, -1)) + [30, 31]
    assert list(study.PERMUTATION) == order
    assert np.array_equal(altered['q_context'], q[:, order])
    assert np.array_equal(altered['u_context'], u[:, order])
    assert not np.shares_memory(altered['q_context'], q)
    assert not np.shares_memory(altered['u_context'], u)
    assert altered['future_u'] is future and altered['target'] is marker
    def local(b):
        position, torque = b['q_context'], b['u_context']
        difference = position[:, 31] - position[:, 30]
        return np.concatenate((position[:, 31], difference, torque[:, 30], difference**2,
                               np.tanh(torque[:, 30])), axis=1)
    original = local(batch)
    changed = local(altered)
    assert np.array_equal(original, changed)
    for arm in PRIMARY[:2]:
        assert study.check_permutation(arm, original, changed)['exact_invariance'] is True
    altered['q_context'][0, 0, 0] = -100
    assert np.array_equal(batch['q_context'], q)
    slopes = lambda b: (b['q_context'][:, :30] * (np.arange(30)-14.5)[None, :, None]).sum(1)/2247.5
    assert not np.allclose(slopes(batch), slopes(study.permuted_batch(batch)))


def test_local_permutation_mismatch_is_validity_failure_not_a_sixth_quality_gate():
    original = np.zeros((2, 128, 6))
    changed = original.copy(); changed[0, 0, 0] = 1.
    for arm in PRIMARY[:2]:
        with pytest.raises(ValueError, match='invariance'):
            study.check_permutation(arm, original, changed)
    record = study.check_permutation('temporal_affine', original, changed)
    assert record['exact_invariance'] is False and record['maximum_absolute_change'] == 1.
    changed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        study.check_permutation('temporal_affine', original, changed)


@pytest.mark.parametrize('failure', ['raised_numeric', 'returned_nonfinite'])
def test_temporal_ood_numeric_failure_is_retained_without_changing_ordinary_rule(monkeypatch, failure):
    cfg, rows = fixture_rows()
    before = rule(rows)
    original = np.zeros((2, 128, 6))
    error = study.old.FitFailure('nonfinite structured transition output; no repair')
    def infer(*args):
        if failure == 'raised_numeric':
            raise error
        return SimpleNamespace(numpy=lambda: np.full_like(original, np.inf))
    monkeypatch.setattr(study.old, 'infer', infer)
    record = study.diagnostic_prediction(object(), 'temporal_affine', {'opaque': object()}, original)
    assert record['status'] == 'FAILED' and record['check'] is None
    assert record['error']['type'] == 'NonfiniteDiagnostic'
    if failure == 'raised_numeric':
        assert record['prediction'] is None and record['error']['message'] == str(error)
    else:
        assert np.isinf(record['prediction']).all()  # Preserve the raw witness, without repair.
    diagnostic_rows = study.old.scored_rows({'arm': 'temporal_affine'}, record['prediction'], object(),
                                            object(), cfg, record['error'])
    assert [r['horizon'] for r in diagnostic_rows] == [64, 128]
    assert all(r['status'] == 'FAILED' and r['metrics'] is None for r in diagnostic_rows)
    assert rule(rows) == before and before['total'] == 5
    for arm in PRIMARY[:2]:
        with pytest.raises(ValueError, match='validity'):
            study.diagnostic_prediction(object(), arm, {}, original)


@pytest.mark.parametrize('failure', ['programming', 'shape'])
def test_diagnostic_does_not_rescue_unrelated_errors(monkeypatch, failure):
    error = ValueError('fabricated programmer error')
    def infer(*args):
        if failure == 'programming':
            raise error
        return SimpleNamespace(numpy=lambda: np.zeros((2, 1, 6)))
    monkeypatch.setattr(study.old, 'infer', infer)
    with pytest.raises(ValueError) as caught:
        study.diagnostic_prediction(object(), 'temporal_affine', {}, np.zeros((2, 128, 6)))
    if failure == 'programming':
        assert caught.value is error
    else:
        assert 'shape' in str(caught.value)


def test_common_initial_digest_and_zero_head_use_exact_owned_parameter_bytes(monkeypatch):
    class Value:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    common = {'weight': np.arange(578, dtype=np.float32), 'bias': np.zeros(12, dtype=np.float32)}
    cell = SimpleNamespace(state_dict=lambda: {k: Value(v) for k, v in common.items()})
    monkeypatch.setattr(study.torch, 'count_nonzero', np.count_nonzero)
    digests = []
    for arm in PRIMARY:
        model = SimpleNamespace(cell=cell, initializer=arm, added_parameter_count=0 if arm == 'last_two' else 372,
                                head=SimpleNamespace(weight=np.zeros((12, 30)), bias=np.zeros(12)))
        result = study.initial_pairing(model)
        assert result['common_parameters'] == 590 and result['zero_head'] is True
        assert result['added_parameters'] == model.added_parameter_count
        digests.append(result['common_cell_sha256'])
    assert len(set(digests)) == 1
    common['weight'][0] = 1.
    assert study.initial_pairing(model)['common_cell_sha256'] != digests[0]
    model.head.bias[0] = 1.
    with pytest.raises(ValueError, match='zero'):
        study.initial_pairing(model)


@pytest.mark.parametrize('arm,count,expected_bytes', [('last_two', 590, 2600),
                                                    ('local_affine', 962, 4088),
                                                    ('temporal_affine', 962, 4088)])
def test_complete_persistent_and_request_costs_with_synthetic_payloads(arm, count, expected_bytes):
    class Payload:
        def __init__(self, n): self.n = n
        def numel(self): return self.n
        def element_size(self): return 4
    model = SimpleNamespace(parameters=lambda: iter([Payload(count)]), buffers=lambda: iter([]))
    item = study.resource_model(model, arm)
    assert item['parameters'] == count and item['parameter_bytes'] == count * 4
    assert item['state_scalars'] == 12 and item['state_bytes'] == 48
    assert item['buffer_bytes'] == 0 and item['normalizer_bytes'] == 192
    assert sum(item[key] for key in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) == expected_bytes
    assert item['input_bytes'] == (32*6*2+128*6)*8 == 9216
    assert item['output_bytes'] == 128*6*8 == 6144
    assert item['temporary_workspace'] == 'not measured'


def test_native_cap_includes_equality_and_keeps_helper_status_separate():
    clock = SimpleNamespace(now_ns=lambda: 2_000_000_000)
    study.check_deadline(clock, 1, 2., whole=False)
    with pytest.raises(study.old.FitFailure):
        study.check_deadline(clock, 0, 2., whole=False)
    with pytest.raises(study.old.WholeStudyTimeout):
        study.check_deadline(clock, 0, 2., whole=True)
    receipt = {'status': 'PASS', 'completed_updates': 4096}
    assert study.effective_status({'fit': receipt, 'effective_status': 'FAILED'}) == 'FAILED'
    assert receipt['status'] == 'PASS'


def fake_admission(tmp_path, monkeypatch):
    import plot_robot_structured
    root = tmp_path / 'repo'
    root.mkdir()
    folders = {group: tmp_path / group for group in ('structured', 'transition')}
    for folder in folders.values():
        folder.mkdir()
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'PARENT_FOLDERS', folders)
    for key in study.old.THREADS:
        monkeypatch.setenv(key, '1')
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path': str(path), **study.old.descriptor(path)}
    sources = {}
    for name in study.SOURCES:
        item = write(root / name, {'fabricated_source': name})
        sources[name] = {key: item[key] for key in ('bytes', 'sha256')}
    payloads = {group: {name: write(folders[group] / name, {'opaque_fake': name})
                       for name in names} for group, names in study.PAYLOADS.items()}
    closures = {}
    for group, folder in folders.items():
        engineering = tmp_path / (group + '-engineering')
        closures[group] = {
            'manifest': write(folder / 'manifest.json', {'files': {
                name: {k: v[k] for k in ('bytes', 'sha256')} for name, v in payloads[group].items()}}),
            'receipt': write(folder / 'receipt.json', {'status': 'PASS'}),
            'process': write(engineering / 'run-process-01.json', {'returncode': 0}),
            'audit': write(tmp_path / (group + '-audit.json'), {'status': 'PASS', 'agreement': True}),
            'audit_process': write(engineering / 'audit-process-01.json', {'returncode': 0}),
        }
    data = {'fabricated.npz': write(tmp_path / 'fabricated.npz', {'opaque_fake': 'no array'})}
    prior_plan = {'data': data, 'parent_registration_sha256': '2' * 64,
                  'parent_closure': closures['transition']}
    prior = {'plan': prior_plan, 'engineering': str(tmp_path / 'structured-engineering'),
             'audit_path': closures['structured']['audit']['path'],
             'inputs': {v['path']: {k: v[k] for k in ('bytes', 'sha256')}
                        for v in closures['structured'].values()}}
    monkeypatch.setattr(plot_robot_structured, 'authenticate', lambda folder: prior)
    monkeypatch.setattr(study.structured, 'authenticate', lambda path: (prior_plan, '1' * 64))
    launcher = root / 'scripts/launch_robot_history_initialization.py'
    launcher.write_text('# fake launcher\n')
    launcher_pin = study.old.descriptor(launcher)
    log = tmp_path / 'qualification.log'
    log.write_text('fabricated qualification\n')
    qualification = write(tmp_path / 'qualification.json', {'status': 'PASS', 'sources': sources,
        'launcher': launcher_pin,
        'commands': [{'returncode': 0, 'log': str(log), 'sha256': study.old.descriptor(log)['sha256']}]})
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'parent_registration_sha256': {'structured': '1' * 64, 'transition': '2' * 64},
            'data': copy.deepcopy(data), 'parent_closure': copy.deepcopy(closures),
            'parent_payloads': payloads, 'qualification': qualification, 'launcher': launcher_pin}
    registration = tmp_path / 'registration.json'
    write(registration, plan)
    return registration, plan


def test_complete_metadata_admission_never_decodes_arrays(tmp_path, monkeypatch):
    registration, plan = fake_admission(tmp_path, monkeypatch)
    assert len(plan['parent_payloads']['structured']) == 131
    assert len(plan['parent_payloads']['transition']) == 32
    assert sum(len(group) for group in plan['parent_payloads'].values()) == 163
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('metadata admission decoded an array'))
    actual, sha = study.authenticate(registration)
    assert actual == plan and sha == hashlib.sha256(registration.read_bytes()).hexdigest()


@pytest.mark.parametrize('damage', ['source', 'config', 'parent_sha', 'closure', 'data',
                                   'payload_missing', 'payload_bytes', 'qualification', 'log', 'launcher'])
def test_provenance_corruption_rejects_before_any_array(tmp_path, monkeypatch, damage):
    registration, plan = fake_admission(tmp_path, monkeypatch)
    if damage == 'source':
        (study.ROOT / study.SOURCES[-1]).write_text('changed')
    elif damage == 'config':
        plan['config']['updates'] = 4095
    elif damage == 'parent_sha':
        plan['parent_registration_sha256']['transition'] = '0' * 64
    elif damage == 'closure':
        plan['parent_closure']['transition']['process']['sha256'] = '0' * 64
    elif damage == 'data':
        plan['data']['fabricated.npz']['sha256'] = '0' * 64
    elif damage == 'payload_missing':
        del plan['parent_payloads']['transition']['gru_residual-8101-lr0/final.npz']
    elif damage == 'payload_bytes':
        Path(plan['parent_payloads']['structured']['dense_bounded-8101-lr0/final.npz']['path']).write_text('changed')
    elif damage == 'qualification':
        path = Path(plan['qualification']['path'])
        item = json.loads(path.read_text()); item['status'] = 'FAILED'
        path.write_text(json.dumps(item)); plan['qualification'].update(study.old.descriptor(path))
    elif damage == 'log':
        (tmp_path / 'qualification.log').write_text('changed')
    else:
        (study.ROOT / 'scripts/launch_robot_history_initialization.py').write_text('changed')
    registration.write_text(json.dumps(plan))
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('array decoded before failed admission'))
    with pytest.raises(ValueError):
        study.authenticate(registration)


def test_parent_array_loading_checks_pin_and_preserves_fortran_layout(tmp_path, monkeypatch):
    path = tmp_path / 'linear.npz'
    value = np.asfortranarray(np.arange(150.).reshape(6, 25))
    np.savez(path, coefficient=value)
    descriptor = {'path': str(path), **study.old.descriptor(path)}
    plan = {'parent_payloads': {'structured': {'linear.npz': descriptor}}}
    loaded = study.load_parent_arrays(plan, 'structured', 'linear.npz')['coefficient']
    assert loaded.flags.f_contiguous and loaded.flags.owndata
    assert np.array_equal(loaded, value)
    descriptor['sha256'] = '0' * 64
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('unadmitted decode'))
    with pytest.raises(ValueError):
        study.load_parent_arrays(plan, 'structured', 'linear.npz')


def test_parent_copy_is_byte_exact_exclusive_and_pin_checked(tmp_path):
    source = tmp_path / 'source.json'
    source.write_bytes(b'{"opaque":"not a model"}\n')
    plan = {'parent_payloads': {'structured': {'fits.json': {
        'path': str(source), **study.old.descriptor(source)}}}}
    destination = tmp_path / 'child' / 'fits.json'
    study.copy_parent(plan, 'structured', 'fits.json', destination)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        study.copy_parent(plan, 'structured', 'fits.json', destination)
    source.write_text('changed')
    with pytest.raises(ValueError):
        study.copy_parent(plan, 'structured', 'fits.json', tmp_path / 'different.json')


@pytest.mark.parametrize('cross_preservation_cap', [False, True])
def test_all_attempts_close_before_first_dev_even_when_fit_preservation_crosses_cap(
        tmp_path, monkeypatch, cross_preservation_cap):
    cfg = study.config()
    cfg['direct_ridges'] = []  # This fixture stops before evaluation, so no reference construction is needed.
    registration = tmp_path / 'registration.json'
    registration.write_text('{}')
    output = tmp_path / 'run'
    monkeypatch.setattr(study, 'authenticate', lambda path: ({'sources': {}}, 'a' * 64))
    monkeypatch.setattr(study, 'config', lambda: cfg)
    monkeypatch.setattr(study, 'SOURCES', ())
    monkeypatch.setattr(study.torch, 'set_num_threads', lambda n: None)
    native = {'now': 0}
    monkeypatch.setattr(study, 'SuspendClock', lambda: SimpleNamespace(
        now_ns=lambda: native['now'], backend='fabricated-clock'))
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6),
            'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    batches = {'record': np.zeros((1, 1), np.int64), 'start': np.zeros((1, 1), np.int64)}
    attempted, cleared, helper_receipts = [], [], []
    sentinel = RuntimeError('fabricated stop on first admitted DEV read')
    def load(plan, name):
        if name.startswith('dev-data-'):
            assert len(attempted) == 18 and len(cleared) == 18
            assert len(list(output.glob('completed-fit-*.json'))) == 48
            barrier = json.loads((output / 'checkpoint-barrier.json').read_text())
            assert barrier['fresh_fit_attempts'] == 18 and barrier['cached_fit_records'] == 30
            assert barrier['fit_records'] == 48 and barrier['dev_loads_this_run'] == 0
            assert len(barrier['checkpoints']) == 48
            raise sentinel
        if name == 'normalizers.npz':
            return norm
        assert name.startswith('fit-data-')
        return {'q': np.zeros((10, 6)), 'u': np.zeros((10, 6))}
    monkeypatch.setattr(study, 'load_arrays', load)
    monkeypatch.setattr(study.old, 'normalizers', lambda *a: norm)
    monkeypatch.setattr(study.old, 'normalized_record', lambda record, normal: record)
    monkeypatch.setattr(study.old, 'make_batches', lambda *a: batches)
    def load_parent(plan, group, name):
        assert group == 'structured'
        if name == 'normalizers.npz': return norm
        if name == 'linear.npz': return {'coefficient': np.zeros((6, 25))}
        assert name.startswith('batches-')
        return batches
    monkeypatch.setattr(study, 'load_parent_arrays', load_parent)
    monkeypatch.setattr(study, 'copy_parent', lambda plan, group, name, destination: Path(destination).write_bytes(b'opaque'))
    def model_for(arm, seed, linear):
        return SimpleNamespace(arm=arm, seed=seed, zero_grad=lambda **k: cleared.append((arm, seed)))
    monkeypatch.setattr(study, 'model_for', model_for)
    monkeypatch.setattr(study, 'initial_pairing', lambda model: {
        'common_cell_sha256': str(model.seed), 'zero_head': True,
        'common_parameters': 590, 'added_parameters': 0 if model.arm == 'last_two' else 372})
    monkeypatch.setattr(study, 'resource_model', lambda *a: {'fabricated': True})
    def train(model, data, order, *, cfg, lr, folder, check, fit_started):
        check()
        assert order is batches and len(data) == 7
        folder.mkdir()
        (folder / 'final.npz').write_bytes(b'opaque fabricated checkpoint')
        attempted.append((model.arm, model.seed, lr))
        receipt = {'status': 'PASS', 'completed_updates': 4096}
        helper_receipts.append(receipt)
        if cross_preservation_cap and len(attempted) == 1:
            native['now'] = int(cfg['fit_cap_seconds'] * 1e9)
        return receipt
    monkeypatch.setattr(study.old, 'train_one', train)
    def cached(plan, out, models, linear):
        assert len(attempted) == 18
        records = []
        for arm in CACHED:
            for seed in cfg['seeds']:
                for index, rate in enumerate(cfg['learning_rates']):
                    key = f'{arm}-{seed}-lr{index}'
                    (out / key).mkdir()
                    (out / key / 'final.npz').write_bytes(b'opaque cached checkpoint')
                    records.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate,
                                    'fit': {'status': 'PASS'}, 'origin': 'cached_parent'})
        return records
    monkeypatch.setattr(study, 'cached_fit_records', cached)
    with pytest.raises(RuntimeError) as caught:
        study.run(registration, output)
    assert caught.value is sentinel
    expected = [(arm, seed, rate) for seed in cfg['seeds'] for rate in cfg['learning_rates'] for arm in PRIMARY]
    assert attempted == expected and all(item['status'] == 'PASS' for item in helper_receipts)
    first = json.loads((output / 'completed-fit-01.json').read_text())
    assert first['effective_status'] == ('FAILED' if cross_preservation_cap else 'PASS')
    assert first['fit']['status'] == 'PASS'
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['completed_fits'] == 48 and failure['metric_rows'] == 0


@pytest.mark.parametrize('damage', ['source', 'qualification', 'commit', 'existing_output'])
def test_launcher_rejects_unfrozen_or_repeated_run_before_subprocess(tmp_path, monkeypatch, damage):
    import launch_robot_history_initialization as launcher
    root = tmp_path / 'repo'
    root.mkdir()
    source_name = 'src/fabricated.py'
    source = root / source_name
    source.parent.mkdir()
    source.write_text('# fabricated source\n')
    launcher_path = root / 'scripts/launch_robot_history_initialization.py'
    launcher_path.parent.mkdir()
    launcher_path.write_text('# fabricated launcher\n')
    registration = root / 'research/robot-history-initialization-registration.json'
    registration.parent.mkdir()
    engineering = root / 'engineering'
    engineering.mkdir()
    output = root / 'study'
    log = engineering / 'qualification.log'
    log.write_text('fabricated PASS\n')
    sources = {source_name: launcher.descriptor(source)}
    qpath = engineering / 'qualification.json'
    qualification = {'status': 'PASS', 'sources': sources,
                     'launcher': launcher.descriptor(launcher_path),
                     'commands': [{'returncode': 0, 'log': str(log),
                                   'sha256': launcher.descriptor(log)['sha256']}]}
    if damage == 'qualification':
        qualification['status'] = 'FAILED'
    qpath.write_text(json.dumps(qualification))
    plan = {'sources': sources, 'launcher': launcher.descriptor(launcher_path),
            'qualification': {'path': str(qpath), **launcher.descriptor(qpath)},
            'config': {'wall_cap_seconds': 1.}}
    registration.write_text(json.dumps(plan))
    for key, value in {'ROOT': root, 'ENGINEERING': engineering, 'REGISTRATION': registration,
                       'OUTPUT': output, '__file__': str(launcher_path)}.items():
        monkeypatch.setattr(launcher, key, value)
    def git(args, **kwargs):
        if args == ['git', 'rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        assert args[:2] == ['git', 'show'] and kwargs['cwd'] == root
        name = args[2].split(':', 1)[1]
        return b'not committed' if damage == 'commit' and name == source_name else (root / name).read_bytes()
    monkeypatch.setattr(launcher.subprocess, 'check_output', git)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('unadmitted scientific launch'))
    if damage == 'source':
        source.write_text('changed')
    elif damage == 'existing_output':
        output.mkdir()
    with pytest.raises(ValueError):
        launcher.main()
    assert not (engineering / 'run-launch-01.json').exists()
    assert not (engineering / 'run-process-01.log').exists()
