"""Fabricated saved-array and scalar witnesses; no actual study invocation."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_robot_observer_study as audit


def rule_fixture():
    cfg = {'partitions': {'dev': ['d0', 'd1'], 'exposed': ['d0', 'd1', 'e0', 'e1']},
           'cached_rates': dict.fromkeys(audit.CACHED, .001), 'selection_scope': 'fixture DEV2-only'}
    recipes = [(a, s, r) for a in audit.LEARNED for s in audit.SEEDS for r in audit.RATES]
    recipes += [(a, s, None) for a in audit.FIXED for s in audit.SEEDS]
    recipes += [(a, s, .001) for a in audit.CACHED for s in audit.SEEDS]
    recipes += [(a, None, None) for a in audit.REFS]
    rows = []
    for recording in cfg['partitions']['exposed']:
        for arm, seed, rate in recipes:
            value = .8 if arm == 'observer_learned' else 1.
            for horizon in (64, 128):
                n = 22*horizon*6
                rows.append({'recording': recording, 'arm': arm, 'seed': seed, 'learning_rate': rate,
                             'horizon': horizon, 'status': 'PASS', 'error': None,
                             'metrics': {'standardized_rmse': value, 'standardized_sse': value*value*n,
                                         'scalars': n, 'physical_rmse_deg': value, 'per_joint_rmse_deg': [value]*6,
                                         'windows': 22, 'horizon': horizon}})
    resources = []
    for arm in (*audit.ARMS, *audit.REFS):
        for seed in audit.SEEDS if arm in audit.ARMS else (None,):
            rate = .001 if arm in (*audit.LEARNED, *audit.CACHED) else None
            resources.append({'arm': arm, 'seed': seed, 'learning_rate': rate, 'status': 'PASS', 'error': None,
                              'timing': {'median_seconds': 1.}, 'parameter_bytes': 100, 'state_bytes': 48,
                              'buffer_bytes': 0, 'normalizer_bytes': 192})
    return cfg, rows, resources


def replace_rmse(rows, arm, value, recordings=None, rate=None):
    for row in rows:
        if row['arm'] == arm and (recordings is None or row['recording'] in recordings) and (rate is None or row['learning_rate'] == rate):
            row['metrics'].update(standardized_rmse=value, standardized_sse=value*value*row['metrics']['scalars'],
                                  physical_rmse_deg=value, per_joint_rmse_deg=[value]*6)


def flags(result):
    return [row['passed'] for row in result['conditions']]


def test_full_416_rows_43_costs_five_rules_and_lower_rate_tie():
    cfg, rows, resources = rule_fixture()
    decision = audit.decisions(rows, resources, cfg)
    selection, result = decision['selection'], decision['result']
    assert len(rows) == 416 and len(resources) == 43
    assert selection['selected_rates']['observer_learned'] == .001
    assert result['status'] == 'OBSERVER_DEVELOPMENT_PASS'
    assert flags(result) == [True]*5 and len(result['equal_file_means']) == 17
    assert result['equal_file_means']['observer_learned'] == pytest.approx(.8)


def test_selection_ignores_both_extra_exposed_files():
    cfg, rows, resources = rule_fixture()
    replace_rmse(rows, 'observer_learned', .7, ['d0', 'd1'], .001)
    replace_rmse(rows, 'observer_learned', .8, ['d0', 'd1'], .003)
    replace_rmse(rows, 'observer_learned', 100., ['e0', 'e1'], .001)
    replace_rmse(rows, 'observer_learned', 0., ['e0', 'e1'], .003)
    decision = audit.decisions(rows, resources, cfg)
    selection, result = decision['selection'], decision['result']
    assert selection['selected_rates']['observer_learned'] == .001
    assert result['equal_file_means']['observer_learned'] == pytest.approx(50.35)
    assert flags(result)[1:3] == [False, False]


def test_strongest_control_not_only_initializer_and_positive_baseline():
    cfg, rows, resources = rule_fixture()
    replace_rmse(rows, 'causal_ridge_100', .7)
    result = audit.decisions(rows, resources, cfg)['result']
    assert flags(result) == [True, False, True, True, False]
    assert 'causal_ridge_100' in result['dominators']
    for arm in ('causal_ridge_100', 'observer_learned'):
        replace_rmse(rows, arm, 0.)
    result = audit.decisions(rows, resources, cfg)['result']
    assert flags(result)[1] is False
    assert result['equal_file_means']['observer_learned'] == 0


@pytest.mark.parametrize('value,expected', [(.95, True), (math.nextafter(.95, math.inf), False)])
def test_five_percent_boundary(value, expected):
    cfg, rows, resources = rule_fixture()
    replace_rmse(rows, 'observer_learned', value)
    assert flags(audit.decisions(rows, resources, cfg)['result'])[1] is expected


@pytest.mark.parametrize('factor,expected', [(1.5, True), (1.500001, False)])
def test_complete_request_latency_boundary(factor, expected):
    cfg, rows, resources = rule_fixture()
    for row in resources:
        if row['arm'] == 'observer_learned': row['timing']['median_seconds'] = factor
    assert flags(audit.decisions(rows, resources, cfg)['result'])[3] is expected


def test_each_file_guard_cannot_be_hidden_by_other_file_gains():
    cfg, rows, resources = rule_fixture()
    replace_rmse(rows, 'observer_learned', .5)
    replace_rmse(rows, 'observer_learned', 1.020001, ['e1'])
    result = audit.decisions(rows, resources, cfg)['result']
    assert flags(result)[:3] == [True, True, False]


@pytest.mark.parametrize('damage', ['missing_metric', 'duplicate_metric', 'wrong_rate', 'missing_cost', 'duplicate_cost'])
def test_complete_identity_rosters_are_mandatory(damage):
    cfg, rows, resources = rule_fixture()
    if damage == 'missing_metric': rows.pop()
    elif damage == 'duplicate_metric': rows[-1] = rows[0]
    elif damage == 'wrong_rate': rows[0]['learning_rate'] = .002
    elif damage == 'missing_cost': resources.pop()
    else: resources[-1] = resources[0]
    with pytest.raises(ValueError): audit.decisions(rows, resources, cfg)


def test_failure_and_h64_scope_and_incomplete_frontier_remain_failures():
    cfg, rows, resources = rule_fixture()
    row = next(r for r in rows if r['arm'] == 'gru10' and r['horizon'] == 64)
    row.update(status='FAILED', error={'type': 'KnownNumeric'}, metrics=None)
    result = audit.decisions(rows, resources, cfg)['result']
    assert flags(result)[0] is False
    resources[-1].update(status='FAILED', error={'type': 'NonfiniteTiming'}, timing=None)
    result = audit.decisions(rows, resources, cfg)['result']
    assert flags(result)[-1] is False and result['frontier_complete'] is False


@pytest.mark.parametrize('damage', ['unknown', 'success_error', 'failed_timing', 'fixed_updates'])
def test_status_evidence_cannot_hide_malformed_failures(damage):
    if damage == 'fixed_updates':
        args = list(frozen_fixture('observer_fixed'))
        args[-1].update(status='FAILED', completed_updates=1)
        with pytest.raises(ValueError): audit.validate_frozen_evidence(*args, np, updates=3)
        return
    cfg, rows, resources = rule_fixture()
    if damage == 'unknown': resources[0]['status'] = 'OTHER'
    elif damage == 'success_error': resources[0]['error'] = {'type': 'KnownNumeric'}
    else: resources[0].update(status='FAILED', error={'type': 'KnownNumeric'})
    with pytest.raises(ValueError): audit.decisions(rows, resources, cfg)


def test_independent_closed_form_metrics_units_and_two_horizons():
    target = np.zeros((2, 128, 6))
    prediction = np.ones_like(target); prediction[:, 64:] = 3
    scale = np.arange(1, 7, dtype=np.float64)
    short, full = [audit.scored(prediction, target, scale, h, np) for h in (64, 128)]
    assert short['standardized_sse'] == 768 and short['standardized_rmse'] == 1
    assert full['standardized_sse'] == 7680 and full['standardized_rmse'] == math.sqrt(5)
    assert short['physical_rmse_deg'] == math.sqrt(91/6)
    assert full['per_joint_rmse_deg'] == pytest.approx(np.arange(1, 7)*math.sqrt(5))


def frozen_fixture(mode='observer_learned', updates=3):
    backbone = {'cell.fabricated': np.arange(590, dtype=np.float32)}
    initial = {k: v.copy() for k, v in backbone.items()}
    if mode in audit.LEARNED[:2]:
        initial.update({'head.weight': np.zeros((12, 30), np.float32), 'head.bias': np.zeros(12, np.float32)})
    elif mode != 'last_two':
        initial['gain'] = np.zeros((12, 6), np.float32) if mode == 'observer_zero' else np.tile(np.eye(6, dtype=np.float32), (2, 1))
    final = {k: v.copy() for k, v in initial.items()}
    optimizer = {}
    if mode in audit.LEARNED:
        for key in set(initial)-set(backbone):
            final[key] += .1
            optimizer[key+'/step'] = np.array(updates, np.float32)
            for suffix in ('exp_avg', 'exp_avg_sq'): optimizer[key+'/'+suffix] = np.ones_like(initial[key])
    receipt = {'status': 'PASS', 'completed_updates': updates if mode in audit.LEARNED else 0}
    return initial, final, optimizer, backbone, mode, receipt


@pytest.mark.parametrize('mode', (*audit.LEARNED, *audit.FIXED))
def test_frozen_cell_and_only_useful_initializer_adam_ownership(mode):
    args = frozen_fixture(mode)
    value = audit.validate_frozen_evidence(*args, np, updates=3)
    assert value['frozen_parameters'] == 590
    assert value['trainable_parameters'] == (372 if mode in audit.LEARNED[:2] else 72 if mode == 'observer_learned' else 0)
    assert value['fixed_gain_scalars'] == (72 if mode in ('observer_fixed', 'observer_zero') else 0)


@pytest.mark.parametrize('damage', ['changed_cell', 'cell_adam', 'wrong_initial_gain', 'partial_adam', 'bad_step', 'nonfinite_pass'])
def test_frozen_checkpoint_corruption_fails_even_after_unsuccessful_fit(damage):
    initial, final, optimizer, backbone, mode, receipt = frozen_fixture()
    if damage == 'changed_cell':
        final['cell.fabricated'][0] = 1
        receipt['status'] = 'FAILED'
    elif damage == 'cell_adam': optimizer['cell.fabricated/step'] = np.array(3, np.float32)
    elif damage == 'wrong_initial_gain': initial['gain'][0, 0] = 0
    elif damage == 'partial_adam': optimizer.pop('gain/exp_avg')
    elif damage == 'bad_step': optimizer['gain/step'][...] = 5
    else: final['gain'][0, 0] = np.nan
    with pytest.raises(ValueError): audit.validate_frozen_evidence(initial, final, optimizer, backbone, mode, receipt, np, updates=3)


def test_native_failed_attempt_may_retain_one_partial_adam_update_but_never_backbone_change():
    args = list(frozen_fixture())
    args[-1].update(status='FAILED', completed_updates=2)
    args[1]['gain'][0, 0] = np.inf
    audit.validate_frozen_evidence(*args, np, updates=3)


def test_window_alignment_uses_only_public_context_and_u31_for_q32():
    indices = np.arange(3636, dtype=np.float64)[:, None]
    record = {'q': np.repeat(indices, 6, axis=1), 'u': np.repeat(10000+indices, 6, axis=1),
              'raw_indices': np.arange(0, 90881, 25, dtype=np.int64)}
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    starts, batch, target = audit.windows(record, norm, np)
    assert starts.tolist() == list(range(64, 3425, 160)) and len(starts) == 22
    assert set(batch) == {'q_context', 'u_context', 'future_u'}
    assert batch['q_context'][0, -1, 0] == 95 and batch['future_u'][0, 0, 0] == 10095
    assert target[0, 0, 0] == 96 and target[0, -1, 0] == 223
    changed = copy.deepcopy(record); changed['q'][96:224] = -99
    _, other, _ = audit.windows(changed, norm, np)
    assert np.array_equal(other['q_context'][0], batch['q_context'][0])


def test_saved_scalar_tolerance_never_changes_identity():
    audit.close({'value': .2}, {'value': .2+1e-13}, 'fixture')
    with pytest.raises(ValueError): audit.close({'count': 3}, {'count': 3.0}, 'fixture')
    with pytest.raises(ValueError): audit.close({'value': .2}, {'value': .21}, 'fixture')


@pytest.mark.parametrize('arm', audit.REFS)
def test_independent_reference_overflow_preserves_qualified_bank_policy(monkeypatch, arm):
    value = np.full((1, 128, 6), np.inf, np.float64)
    monkeypatch.setattr(audit.prior_audit, 'reference_prediction', lambda *args: value)
    if arm in audit.REFS[:2]:
        with pytest.raises(ValueError, match='^nonfinite ridge prediction$'):
            audit.reference_prediction(arm, {}, {}, np)
    else:
        assert audit.reference_prediction(arm, {}, {}, np) is value


def timing_fixture():
    selection = {'selected_rates': dict.fromkeys(audit.LEARNED, .001)}
    rows = []
    seconds = [1.+.01*i for i in range(20)]
    for identity in audit.resource_identities(selection):
        rows.append({**identity, **audit.storage(identity['arm']), 'status': 'PASS', 'error': None,
                     'timing': {'seconds': seconds.copy(), 'median_seconds': 1.095, 'p95_seconds': 1.1805,
                                'scope': audit.prior_audit.TIMING_SCOPE}})
    return selection, rows


def test_all43_complete_request_costs_and_fixed_gain_bytes():
    selection, rows = timing_fixture()
    audit.validate_resources(rows, selection, np)
    gains = [r for r in rows if r['arm'] in ('observer_fixed', 'observer_zero')]
    assert len(gains) == 6 and all(r['buffer_bytes'] == 288 for r in gains)
    assert sum(audit.storage('observer_learned')[k] for k in ('parameter_bytes', 'state_bytes', 'normalizer_bytes', 'buffer_bytes')) == 2888


@pytest.mark.parametrize('damage', ['clock', 'median', 'p95', 'scope', 'buffer', 'identity', 'missing'])
def test_timing_or_storage_tampering_is_not_a_rebenchmark(damage):
    selection, rows = timing_fixture()
    row = next(r for r in rows if r['arm'] == 'observer_fixed')
    if damage == 'clock': row['timing']['seconds'][10] += .1
    elif damage in ('median', 'p95'): row['timing'][damage+'_seconds'] += .01
    elif damage == 'scope': row['timing']['scope'] = 'kernel only'
    elif damage == 'buffer': row['buffer_bytes'] = 0
    elif damage == 'identity': row['seed'] = 999
    else: rows.pop()
    with pytest.raises(ValueError): audit.validate_resources(rows, selection, np)


def test_unavailable_recipe_keeps_three_resource_slots_and_never_zero_cost():
    cfg, rows, resources = rule_fixture()
    for row in rows:
        if row['arm'] == 'observer_learned': row.update(status='FAILED', error={'type': 'FailedTrainingAttempt'}, metrics=None)
    for row in resources:
        if row['arm'] == 'observer_learned':
            row.update(learning_rate=None, status='UNAVAILABLE', error={'type': 'UnavailableSelectedRecipe'}, timing=None)
    result = audit.decisions(rows, resources, cfg)
    assert result['selection']['selected_rates']['observer_learned'] is None
    assert flags(result['result']) == [False]*5
    assert result['result']['costs']['observer_learned'] is None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def early_admission_fixture(root, monkeypatch):
    """Real admission prefix with opaque sources; no parent or decoder calls."""
    monkeypatch.setattr(audit, 'ROOT', root)
    study, eng = root/audit.STUDY, root/audit.ENGINEERING
    sources = {}
    for name in audit.SOURCES:
        payload = ('fabricated source '+name).encode()
        write(root/name, payload); write(study/'sources'/name, payload)
        sources[name] = audit.descriptor(root/name)
    write(root/audit.LAUNCHER, b'fabricated launcher')
    plan = {'version': 'robot-observer-study-v1', 'sources': sources, 'launcher': audit.descriptor(root/audit.LAUNCHER)}
    write(root/audit.REGISTRATION, plan); write(study/'registration.json', plan)
    launch = {'command': audit.COMMAND.copy(), 'prefit_commit': 'f'*40, 'started_utc': '2026-01-03T00:00:00+00:00',
              'registration_sha256': audit.descriptor(root/audit.REGISTRATION)['sha256'], 'launcher': plan['launcher'],
              'thread_env': dict.fromkeys(audit.THREADS, '1'), 'scope': 'fabricated observer admission only'}
    write(eng/'run-process-01.log', b'fabricated original log')
    process = {**launch, 'returncode': 0, 'elapsed_seconds': 1., 'external_timeout': False,
               'log': audit.descriptor(eng/'run-process-01.log')}
    write(eng/'run-launch-01.json', launch); write(eng/'run-process-01.json', process)
    for name in audit.THREADS: monkeypatch.setenv(name, '1')
    def forbidden(*args, **kwargs): raise AssertionError('numeric decode during failed admission')
    monkeypatch.setattr(np, 'load', forbidden)
    return study, eng, plan, launch, process


@pytest.mark.parametrize('damage', ['source', 'snapshot', 'source_roster', 'saved_registration', 'launcher', 'thread',
    'command', 'launch_join', 'returncode', 'timeout', 'elapsed', 'log', 'commit', 'committed_bytes'])
def test_original_source_and_process_fail_before_any_decode(tmp_path, monkeypatch, damage):
    study, eng, plan, launch, process = early_admission_fixture(tmp_path, monkeypatch)
    if damage == 'source': write(tmp_path/audit.SOURCES[0], b'changed')
    elif damage == 'snapshot': write(study/'sources'/audit.SOURCES[0], b'changed')
    elif damage == 'source_roster':
        plan['sources'].pop(audit.SOURCES[0]); write(tmp_path/audit.REGISTRATION, plan)
    elif damage == 'saved_registration': write(study/'registration.json', b'changed')
    elif damage == 'launcher': write(tmp_path/audit.LAUNCHER, b'changed')
    elif damage == 'thread': monkeypatch.setenv(audit.THREADS[0], '2')
    elif damage == 'command':
        launch['command'] = process['command'] = ['python', 'unrelated.py']
    elif damage == 'launch_join': launch['started_utc'] = '2026-01-04T00:00:00+00:00'
    elif damage == 'returncode': process['returncode'] = 1
    elif damage == 'timeout': process['external_timeout'] = True
    elif damage == 'elapsed': process['elapsed_seconds'] = 14461.
    elif damage == 'log': write(eng/'run-process-01.log', b'changed')
    elif damage == 'commit': launch['prefit_commit'] = process['prefit_commit'] = 'not-a-commit'
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda *args, **kwargs: b'not the committed source')
    write(eng/'run-launch-01.json', launch); write(eng/'run-process-01.json', process)
    with pytest.raises(ValueError): audit.authenticate(study, eng/'run-process-01.json')
