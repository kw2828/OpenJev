"""Fabricated arithmetic, timing and raw-processing checks for confirmation audit."""
from __future__ import annotations

import copy
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('confirm_audit_tests', ROOT / 'scripts/audit_robot_history_confirmation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    rows, costs = [], []
    for name in audit.CONFIRM:
        for arm in (*audit.ARMS, *audit.REFS):
            for seed in audit.SEEDS if arm in audit.ARMS else (None,):
                for h in (64, 128):
                    error = .8 if arm == 'temporal_affine' else 1.
                    rows.append({'recording': name, 'arm': arm, 'seed': seed,
                                 'learning_rate': audit.FIXED_RATES.get(arm), 'horizon': h,
                                 'status': 'PASS', 'error': None,
                                 'metrics': {'standardized_rmse': error, 'standardized_sse': error**2 * 22*h*6,
                                             'physical_rmse_deg': error, 'per_joint_rmse_deg': [error]*6,
                                             'scalars': 22*h*6, 'windows': 22, 'horizon': h}})
    for arm in (*audit.ARMS, *audit.REFS):
        for seed in audit.SEEDS if arm in audit.ARMS else (None,):
            costs.append({'arm': arm, 'seed': seed, 'learning_rate': audit.FIXED_RATES.get(arm),
                          'status': 'PASS', 'error': None, 'parameter_bytes': 4000, 'state_bytes': 48,
                          'buffer_bytes': 0, 'normalizer_bytes': 192,
                          'timing': {'median_seconds': .01, 'seconds': [.01]*20,
                                     'p95_seconds': .01, 'scope': audit.TIMING_SCOPE}})
    cfg = audit.config()
    return rows, costs, cfg


def set_error(rows, arm, value, *, recording=None, horizon=None, seed=None):
    for row in rows:
        if row['arm'] == arm and (recording is None or row['recording'] == recording) \
                and (horizon is None or row['horizon'] == horizon) and (seed is None or row['seed'] == seed):
            m = row['metrics']
            m.update(standardized_rmse=value, standardized_sse=value**2 * m['scalars'],
                     physical_rmse_deg=value, per_joint_rmse_deg=[value]*6)


def decide(rows, costs, cfg):
    return audit.decisions(rows, costs, cfg, dict(audit.FIXED_RATES))


def test_independent_rule_complete_roster_and_aggregation():
    rows, costs, cfg = fixture()
    for i, seed in enumerate(audit.SEEDS):
        set_error(rows, 'temporal_affine', [.1, .7, 1.][i], recording=audit.CONFIRM[0], seed=seed)
        set_error(rows, 'temporal_affine', .8, recording=audit.CONFIRM[1], seed=seed)
    result = decide(rows, costs, cfg)
    assert result['equal_file_means']['temporal_affine'] == pytest.approx(.7)
    assert result['passed'] == 5 and result['status'] == 'CONFIRMED_HISTORY_INITIALIZATION'
    assert [c['name'] for c in result['conditions']] == list(audit.CONDITIONS)
    assert len(rows) == 112 and len(costs) == 28


@pytest.mark.parametrize('candidate,passes', [(.95, True), (.950001, False), (1., False)])
def test_relative_improvement_boundary(candidate, passes):
    rows, costs, cfg = fixture(); set_error(rows, 'temporal_affine', candidate)
    result = decide(rows, costs, cfg)
    assert result['conditions'][1]['passed'] is passes


def test_zero_error_tie_is_not_improvement():
    rows, costs, cfg = fixture()
    for arm in audit.PRIMARY:
        set_error(rows, arm, 0.)
    assert not decide(rows, costs, cfg)['conditions'][1]['passed']


def test_equal_mean_gain_does_not_override_one_file_harm():
    rows, costs, cfg = fixture()
    set_error(rows, 'temporal_affine', .7, recording=audit.CONFIRM[0])
    set_error(rows, 'temporal_affine', 1.02001, recording=audit.CONFIRM[1])
    result = decide(rows, costs, cfg)
    assert result['conditions'][1]['passed'] and not result['conditions'][2]['passed']
    set_error(rows, 'temporal_affine', 1.02, recording=audit.CONFIRM[1])
    assert decide(rows, costs, cfg)['conditions'][2]['passed']


@pytest.mark.parametrize('duration,passes', [(.0125, True), (.01250001, False)])
def test_latency_boundary(duration, passes):
    rows, costs, cfg = fixture()
    for row in costs:
        if row['arm'] == 'temporal_affine':
            row['timing']['median_seconds'] = duration
    assert decide(rows, costs, cfg)['conditions'][3]['passed'] is passes


@pytest.mark.parametrize('control', ['gru10', 'causal_ridge_1', 'causal_ridge_100'])
def test_complete_control_frontier_and_strict_dominance(control):
    rows, costs, cfg = fixture(); set_error(rows, control, .8)
    assert control not in decide(rows, costs, cfg)['dominators']
    for row in costs:
        if row['arm'] == control:
            row['parameter_bytes'] -= 1
    result = decide(rows, costs, cfg)
    assert control in result['dominators'] and not result['conditions'][4]['passed']


@pytest.mark.parametrize('arm', ['temporal_affine', 'local_affine', 'gru10', 'causal_ridge_1'])
def test_failed_metric_is_retained_and_not_dropped(arm):
    rows, costs, cfg = fixture()
    row = next(r for r in rows if r['arm'] == arm and r['horizon'] == 128)
    row.update(status='FAILED', error={'type': 'NonfiniteEvaluation'}, metrics=None)
    result = decide(rows, costs, cfg)
    assert arm not in result['equal_file_means'] and not result['frontier_complete']
    assert result['status'] == 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION'


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'wrong_rate', 'wrong_seed', 'official_test'])
def test_metric_identity_tampering_rejected(mutation):
    rows, costs, cfg = fixture()
    if mutation == 'drop': rows.pop()
    elif mutation == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif mutation == 'wrong_rate': rows[0]['learning_rate'] = .003
    elif mutation == 'wrong_seed': rows[0]['seed'] = 9000
    else: rows[0]['recording'] = 'recording_2021_12_15_22H_58M.mat'
    with pytest.raises(ValueError): decide(rows, costs, cfg)


def test_rates_cannot_be_changed_after_confirmation():
    rows, costs, cfg = fixture(); rates = dict(audit.FIXED_RATES); rates['temporal_affine'] = .001
    with pytest.raises(ValueError): audit.decisions(rows, costs, cfg, rates)


@pytest.mark.parametrize('field,value', [('mean_reduction', .01), ('latency_ratio', 2.), ('wall_cap_seconds', 9999.)])
def test_confirmation_cannot_relax_registered_configuration(field, value):
    rows, costs, cfg = fixture(); cfg[field] = value
    with pytest.raises(ValueError): decide(rows, costs, cfg)


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'wrong_rate'])
def test_cost_attempt_identities_are_complete_and_fixed(mutation):
    rows, costs, cfg = fixture()
    if mutation == 'drop': costs.pop()
    elif mutation == 'duplicate': costs[-1] = copy.deepcopy(costs[0])
    else: costs[0]['learning_rate'] = .003
    with pytest.raises(ValueError): decide(rows, costs, cfg)


@pytest.mark.parametrize('status', ['FAILED', 'UNAVAILABLE'])
def test_failed_cost_cannot_be_silently_used(status):
    rows, costs, cfg = fixture()
    row = next(r for r in costs if r['arm'] == 'temporal_affine')
    row.update(status=status, error={'type': 'NonfiniteTiming'})
    result = decide(rows, costs, cfg)
    assert result['costs']['temporal_affine'] is None
    assert not result['conditions'][3]['passed'] and not result['conditions'][4]['passed']


@pytest.mark.parametrize('duration', [0., -1., float('inf'), float('nan')])
def test_invalid_duration_is_not_zero_cost(duration):
    rows, costs, cfg = fixture()
    row = next(r for r in costs if r['arm'] == 'gru10'); row['timing']['median_seconds'] = duration
    assert decide(rows, costs, cfg)['costs']['gru10'] is None


def test_h64_descriptive_cannot_rescue_h128_failure():
    rows, costs, cfg = fixture(); set_error(rows, 'temporal_affine', 1., horizon=128)
    set_error(rows, 'temporal_affine', 0., horizon=64)
    assert decide(rows, costs, cfg)['passed'] != 5


@pytest.mark.parametrize('horizon', [64, 128])
def test_independent_score_matches_hand_calculation(horizon):
    target = np.zeros((2, 128, 6), np.float64)
    predicted = np.broadcast_to(np.arange(1., 7.), target.shape).copy()
    result = audit.scored(predicted, target, np.full(6, 2.), horizon, np)
    assert result['standardized_sse'] == 2*horizon*91
    assert result['standardized_rmse'] == pytest.approx(np.sqrt(91/6))
    assert result['physical_rmse_deg'] == pytest.approx(2*np.sqrt(91/6))
    assert result['per_joint_rmse_deg'] == list(np.arange(2., 13., 2.))
    assert result['scalars'] == 2*horizon*6


@pytest.mark.parametrize('bad', [np.inf, np.nan, 1e308])
def test_prediction_numerical_failure_preserved(bad):
    target = np.zeros((1, 128, 6)); prediction = target.copy(); prediction[0, 0, 0] = bad
    assert audit.scored(prediction, target, np.ones(6), 128, np) is None


def test_bad_target_is_fatal_not_a_model_failure():
    target = np.zeros((1, 128, 6)); target[0, 0, 0] = np.nan
    with pytest.raises(ValueError): audit.scored(np.zeros_like(target), target, np.ones(6), 128, np)


def test_window_ranges_and_future_target_isolation():
    sequence = np.broadcast_to(np.arange(3636.)[:, None], (3636, 6)).copy()
    record = {'q': sequence, 'u': sequence+10000, 'raw_indices': np.arange(3636)*25}
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    starts, batch, target = audit.confirmation_windows(record, norm, np)
    assert np.array_equal(starts, 64+160*np.arange(22))
    assert batch['q_context'][0, -1, 0] == 95 and target[0, 0, 0] == 96
    assert batch['future_u'][0, 0, 0] == 10095 and batch['future_u'][0, -1, 0] == 10222
    assert target[-1, -1, 0] == 3583
    record['q'][96:224] += 100
    _, changed, changed_target = audit.confirmation_windows(record, norm, np)
    for name in batch: assert np.array_equal(batch[name][0], changed[name][0])
    assert not np.array_equal(changed_target[0], target[0])


def test_independent_references_causal_and_analytic():
    batch = {'q_context': np.broadcast_to(np.arange(32.)[None, :, None], (1, 32, 6)).copy(),
             'u_context': np.zeros((1, 32, 6)), 'future_u': np.ones((1, 128, 6))}
    linear = np.zeros((6, 25)); linear[:, :6] = np.eye(6); linear[:, 12:18] = np.eye(6)
    actual = audit.reference_prediction('linear_frozen', {'linear_frozen': linear}, batch, np)
    assert np.array_equal(actual, np.broadcast_to(np.arange(32., 160.)[None, :, None], actual.shape))
    persistence = audit.reference_prediction('persistence', {}, batch, np)
    assert np.array_equal(persistence, np.full((1, 128, 6), 31.))
    coefficients = {}
    for h in range(1, 129):
        a = np.zeros((6, 193+6*h)); a[:, -1] = h; a[:, 192:198] = np.eye(6)
        coefficients[f'h{h:03d}'] = a
    baseline = audit.reference_prediction('causal_ridge_1', {'causal_ridge_1': coefficients}, batch, np)
    changed = copy.deepcopy(batch); changed['future_u'][:, 64:] = 999
    result = audit.reference_prediction('causal_ridge_1', {'causal_ridge_1': coefficients}, changed, np)
    assert np.array_equal(baseline[:, :64], result[:, :64])
    assert np.array_equal(baseline[0, :, 0], np.arange(2., 130.))


@pytest.fixture(scope='module')
def fabricated_mat():
    shape = (6, 90881)
    data = {name: np.zeros(shape) for name in ('q_mot_meas', 'q_se_meas', 'qd_mot_meas', 'qd_se_meas',
            'tau_meas', 'tau_fb_meas', 'q_ref', 'qd_ref', 'tau_ref_ff')}
    data['q_se_meas'][:] = np.arange(1., 7.)[:, None]
    data['q_mot_meas'][:] = np.arange(11., 17.)[:, None]
    data['tau_meas'][:] = np.arange(21., 27.)[:, None]
    data['recording_ok'] = np.ones((1, 1), bool)
    data['time'] = np.arange(90881.)[None, :]/250
    data['READ_ME'] = np.full((17, 2), 'fabricated', dtype=object)
    stream = io.BytesIO(); savemat(stream, data, do_compression=True)
    return stream.getvalue()


def test_raw_processing_selects_correct_channels_and_initializes_constant_filter(fabricated_mat):
    result = audit.independent_recording(fabricated_mat, audit.CONFIRM[0], np)
    assert result['q'].shape == result['u'].shape == (3636, 6)
    np.testing.assert_allclose(result['q'], np.broadcast_to([1., 2., 3., 14., 15., 16.], (3636, 6)), rtol=1e-12)
    np.testing.assert_allclose(result['u'], np.broadcast_to(np.arange(21., 27.), (3636, 6)), rtol=1e-12)
    assert np.array_equal(result['raw_indices'], np.arange(0, 90881, 25))


@pytest.mark.parametrize('name', ['recording_2021_12_15_22H_58M.mat', 'recording_2021_12_15_21H_54M.mat', '../bad.mat'])
def test_closed_raw_names_rejected_before_decoder(monkeypatch, name):
    import scipy.io
    monkeypatch.setattr(scipy.io, 'whosmat', lambda *a, **k: pytest.fail('must reject before MAT decode'))
    with pytest.raises(ValueError): audit.independent_recording(b'not opened', name, np)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_json_rejected(tmp_path, invalid):
    p = tmp_path/'bad.json'; p.write_text('{"x": '+('NaN' if np.isnan(invalid) else 'Infinity')+'}')
    with pytest.raises(ValueError): audit.read(p)


def test_exact_gate_identity_and_scalar_tolerance():
    audit.close({'error': .5}, {'error': .5+1e-13}, 'tiny roundoff')
    with pytest.raises(ValueError): audit.close({'passed': True}, {'passed': 1}, 'gate type')
    with pytest.raises(ValueError): audit.close({'error': .5}, {'error': .501}, 'changed score')


def resource_fixture():
    result = []
    for arm in (*audit.ARMS, *audit.REFS):
        for seed in audit.SEEDS if arm in audit.ARMS else (None,):
            result.append({'key': audit.model_key(arm, seed) if seed is not None else arm,
                           'arm': arm, 'seed': seed, 'learning_rate': audit.FIXED_RATES.get(arm),
                           'status': 'PASS', 'error': None, **audit.storage(arm),
                           'timing': {'seconds': [.01]*19+[.03], 'median_seconds': .01,
                                      'p95_seconds': .011, 'scope': audit.TIMING_SCOPE}})
    return result


def test_raw_timing_summary_and_known_numerical_failure():
    resources = resource_fixture()
    audit.validate_resources(resources, np)
    resources[-4].update(status='FAILED', timing=None,
                         error={'type': 'NonfiniteTiming', 'message': 'nonfinite ridge prediction'})
    audit.validate_resources(resources, np)


@pytest.mark.parametrize('mutation', ['drop', 'order', 'duration', 'median', 'p95', 'bytes', 'scope', 'unknown_failure'])
def test_resource_evidence_cannot_hide_or_rewrite_cost(mutation):
    resources = resource_fixture(); row = resources[0]
    if mutation == 'drop': resources.pop()
    elif mutation == 'order': resources[0], resources[1] = resources[1], resources[0]
    elif mutation == 'duration': row['timing']['seconds'][0] = 0.
    elif mutation == 'median': row['timing']['median_seconds'] = .005
    elif mutation == 'p95': row['timing']['p95_seconds'] = .03
    elif mutation == 'bytes': row['parameter_bytes'] -= 4
    elif mutation == 'scope': row['timing']['scope'] = 'forward only'
    else: row.update(status='FAILED', timing=None, error={'type': 'RuntimeError', 'message': 'unknown defect'})
    with pytest.raises(ValueError): audit.validate_resources(resources, np)


def test_auditor_requires_original_successful_closure_before_other_inputs(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    engineering = tmp_path/'output/robot-history-confirmation-engineering-v1'
    engineering.mkdir(parents=True)
    launch = {'command': list(audit.COMMAND), 'pre_access_commit': 'a'*40,
              'started_utc': '2026-09-25T00:00:00+00:00', 'registration_sha256': 'b'*64,
              'launcher': {}, 'thread_env': dict.fromkeys(audit.THREADS, '1'), 'scope': 'synthetic'}
    process = {**launch, 'returncode': 1, 'elapsed_seconds': 1., 'external_timeout': False, 'log': {}}
    (engineering/'run-launch-01.json').write_text(json.dumps(launch))
    receipt = engineering/'run-process-01.json'; receipt.write_text(json.dumps(process))
    monkeypatch.setattr(audit, 'descriptor', lambda *a: pytest.fail('no other input read before successful closure'))
    with pytest.raises(ValueError, match='original successful closed process'):
        audit.authenticate(tmp_path/'output/robot-history-confirmation-v1', receipt)
