"""Independent fabricated audit qualification, without campaign or model access."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_robot_history_initialization as audit

PRIMARY = ('last_two', 'local_affine', 'temporal_affine')
CACHED = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ARMS = (*PRIMARY, *CACHED)
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS, RATES, DEV = (8101, 8102, 8103), (.001, .003), ('fabricated-first', 'fabricated-second')
PARAMETERS = dict(zip(ARMS, (590, 962, 962, 806, 806, 5916, 1014, 1296), strict=True))
CONDITIONS = ('primary_recipes_complete', 'equal_file_mean_5pct_vs_both_locals',
              'each_file_within_2pct_best_local', 'latency_within_125pct_last_two',
              'complete_frontier_not_dominated')
TIMING_SCOPE = ('batch1 normalization, casting, context32, future128, denormalization and finite check; '
                'no model/disk load; outer validation and operator preparation included; eager CPU')


def metric(error, horizon=128, scalars=120):
    return {'standardized_rmse': error, 'standardized_sse': error**2 * scalars,
            'scalars': scalars, 'physical_rmse_deg': error, 'per_joint_rmse_deg': [error] * 6,
            'windows': 1, 'horizon': horizon}


def scalar_fixture():
    cfg = {'partitions': {'dev': list(DEV)}, 'seeds': list(SEEDS), 'learning_rates': list(RATES),
           'horizons': [64, 128], 'mean_reduction': .05, 'file_harm_ratio': 1.02, 'latency_ratio': 1.25}
    rows = []
    for name in DEV:
        for arm in ARMS:
            for seed in SEEDS:
                for rate in RATES:
                    for horizon in (64, 128):
                        rows.append({'recording': name, 'arm': arm, 'seed': seed, 'learning_rate': rate,
                                     'horizon': horizon, 'status': 'PASS', 'error': None,
                                     'metrics': metric(.8 if arm == 'temporal_affine' else 1., horizon)})
        for arm in REFS:
            for horizon in (64, 128):
                rows.append({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None,
                             'horizon': horizon, 'status': 'PASS', 'error': None, 'metrics': metric(1., horizon)})
    return cfg, rows


def resource_fields(arm):
    neural = arm in ARMS
    count = PARAMETERS[arm] if neural else 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0
    state = {'gru32': 50, 'gru10': 28}.get(arm, 12) if neural else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
    size = 4 if neural else 8
    result = {'parameters': count, 'parameter_bytes': count * size, 'state_scalars': state,
              'state_bytes': state * size, 'buffer_bytes': 16 if arm in REFS[:2] else 0,
              'normalizer_bytes': 192, 'dtype': 'float32' if neural else 'float64',
              'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured'}
    if neural:
        result['inactive_parameters'] = 192 if arm == 'legacy_instant' else 0
    return result


def resource_fixture():
    fits, resources = [], []
    order = [(arm, seed, rate) for seed in SEEDS for rate in RATES for arm in PRIMARY]
    order += [(arm, seed, rate) for seed in SEEDS for rate in RATES for arm in CACHED[:3]]
    order += [(arm, seed, rate) for arm in CACHED[3:] for seed in SEEDS for rate in RATES]
    for arm, seed, rate in order:
        fields = resource_fields(arm)
        fits.append({'key': f'{arm}-{seed}-lr{RATES.index(rate)}', 'arm': arm, 'seed': seed,
                     'learning_rate': rate, 'resources': fields,
                     'origin': 'fresh' if arm in PRIMARY else 'cached_parent'})
        if rate == .001:
            scale = 1.25 if arm == 'temporal_affine' else 1.
            durations = [scale * (.81 + .02 * i) for i in range(20)]
            resources.append({'arm': arm, 'seed': seed, 'learning_rate': rate, **fields,
                              'timing': {'seconds': durations, 'median_seconds': scale,
                                         'p95_seconds': scale * 1.171, 'scope': TIMING_SCOPE}})
    for arm in REFS:
        resources.append({'arm': arm, 'seed': None, **resource_fields(arm),
                          'timing': {'seconds': [.1] * 20, 'median_seconds': .1, 'p95_seconds': .1,
                                     'scope': TIMING_SCOPE}})
    return fits, resources


def flags(result):
    return [item['passed'] for item in result['conditions']]


def set_error(rows, arm, value, recording=None):
    for row in rows:
        if row['arm'] == arm and (recording is None or row['recording'] == recording):
            row['metrics'] = metric(value, row['horizon'])


def test_closed_form_metric_respects_horizon_denominator_and_physical_units():
    target = np.zeros((2, 128, 6))
    prediction = np.ones_like(target)
    prediction[:, 64:] = 3.
    scale = np.arange(1., 7.)
    short = audit.scored(prediction, target, scale, 64, np)
    long = audit.scored(prediction, target, scale, 128, np)
    assert short['standardized_sse'] == 768. and short['standardized_rmse'] == 1.
    assert short['scalars'] == 768 and short['physical_rmse_deg'] == math.sqrt(91 / 6)
    assert short['per_joint_rmse_deg'] == scale.tolist()
    assert long['standardized_sse'] == 7680. and long['scalars'] == 1536
    assert long['standardized_rmse'] == math.sqrt(5)
    np.testing.assert_allclose(long['per_joint_rmse_deg'], math.sqrt(5) * scale, rtol=1e-15)


@pytest.mark.parametrize('value, message', [(float('nan'), 'nonfinite prediction/target'),
                                          (1e300, 'nonfinite metric arithmetic')])
def test_invalid_ordinary_predictions_remain_failed(value, message):
    target = np.zeros((1, 128, 6))
    rows = audit.metric_rows({}, np.full_like(target, value), target, np.ones(6), None, np)
    assert [row['horizon'] for row in rows] == [64, 128]
    assert all(row['status'] == 'FAILED' and row['metrics'] is None
               and row['error'] == {'type': 'NonfiniteEvaluation', 'message': message} for row in rows)


def test_complete_independent_five_condition_witness_and_fixed_counts():
    cfg, rows = scalar_fixture()
    fits, resources = resource_fixture()
    selection, result = audit.decisions(rows, resources, cfg)
    assert len(rows) == 208 and len(rows) // 2 == 104
    assert len(fits) == 48 and sum(fit['origin'] == 'fresh' for fit in fits) == 18
    assert len(resources) == 28
    assert 18 * 4096 == 73728 and 30 * 4096 == 122880
    assert len(PRIMARY) * len(SEEDS) * len(DEV) == 18  # Selected diagnostic banks.
    assert set(selection['selected_rates']) == set(ARMS)
    assert set(selection['selected_rates'].values()) == {.001}
    assert tuple(item['name'] for item in result['conditions']) == CONDITIONS
    assert flags(result) == [True] * 5 and result['passed'] == result['total'] == 5
    assert result['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'


@pytest.mark.parametrize('boundary, expected', [(.95, True), (.950001, False)])
def test_relative_gain_boundary_without_rounding(boundary, expected):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    set_error(rows, 'temporal_affine', boundary)
    _, result = audit.decisions(rows, resources, cfg)
    assert flags(result) == [True, expected, True, True, True]


@pytest.mark.parametrize('baseline', PRIMARY[:2])
def test_zero_error_tie_cannot_become_relative_improvement(baseline):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    set_error(rows, baseline, 0.); set_error(rows, 'temporal_affine', 0.)
    selection, result = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates'][baseline] == .001
    assert flags(result)[0] is True and flags(result)[1] is False


@pytest.mark.parametrize('error, expected', [(1.02, True), (1.020001, False)])
def test_equal_file_average_cannot_hide_file_harm(error, expected):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    set_error(rows, 'temporal_affine', error, DEV[0]); set_error(rows, 'temporal_affine', .6, DEV[1])
    _, result = audit.decisions(rows, resources, cfg)
    assert flags(result) == [True, True, expected, True, True]


@pytest.mark.parametrize('seconds, expected', [(1.25, True), (1.250001, False)])
def test_cost_boundary_uses_complete_request_median(seconds, expected):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    for item in resources:
        if item['arm'] == 'temporal_affine': item['timing']['median_seconds'] = seconds
    _, result = audit.decisions(rows, resources, cfg)
    assert flags(result) == [True, True, True, expected, True]


@pytest.mark.parametrize('axis', ['error', 'latency', 'storage', 'none', 'tradeoff'])
def test_coordinatewise_frontier_requires_one_strict_axis(axis):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    set_error(rows, 'dense_bounded', .79 if axis == 'error' else .81 if axis == 'tradeoff' else .8)
    for item in resources:
        if item['arm'] == 'dense_bounded':
            item['parameter_bytes'] = 3844 if axis in ('storage', 'tradeoff') else 3848
            item['timing']['median_seconds'] = 1.24 if axis in ('latency', 'tradeoff') else 1.25
    _, result = audit.decisions(rows, resources, cfg)
    assert flags(result)[-1] is (axis in ('none', 'tradeoff'))


def test_failed_primary_cannot_be_substituted_or_dropped():
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    for row in rows:
        if row['arm'] == 'temporal_affine':
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    resources = [item for item in resources if item['arm'] != 'temporal_affine']
    selection, result = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates']['temporal_affine'] is None
    assert flags(result) == [False] * 5


@pytest.mark.parametrize('arm', (*CACHED, *REFS))
def test_failed_reference_keeps_rows_and_blocks_only_complete_frontier(arm):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    for row in rows:
        if row['arm'] == arm:
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    resources = [item for item in resources if item['arm'] != arm]
    _, result = audit.decisions(rows, resources, cfg)
    assert len(rows) == 208 and flags(result) == [True, True, True, True, False]
    assert result['frontier_complete'] is False


def test_incomplete_rate_requires_other_full_paired_recipe():
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    target = next(row for row in rows if row['arm'] == 'local_affine' and row['horizon'] == 128)
    target.update(status='FAILED', metrics=None, error={'type': 'NonfiniteEvaluation'})
    selection, _ = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates']['local_affine'] == .003
    assert selection['options']['local_affine'][0]['eligible'] is False
    assert selection['options']['local_affine'][1]['eligible'] is True


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_seed'])
def test_fixed_metric_roster_rejects_omission_and_duplicate_identity(damage):
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    else: rows[0]['seed'] = 99
    with pytest.raises(ValueError):
        audit.decisions(rows, resources, cfg)


def test_pooled_selection_uses_sse_per_scalar_not_mean_case_rmse():
    cfg, rows = scalar_fixture(); _, resources = resource_fixture()
    chosen = [row for row in rows if row['arm'] == 'temporal_affine'
              and row['horizon'] == 128 and row['learning_rate'] == .001]
    for row in rows:
        if row['arm'] == 'temporal_affine': row['metrics'] = metric(.2, row['horizon'], 100)
    for row in chosen: row['metrics'] = metric(0., 128, 100)
    chosen[0]['metrics'] = metric(2., 128, 1)
    selection, _ = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates']['temporal_affine'] == .001  # sqrt(4/501) < .2, while 2/6 > .2.


def test_resource_rows_join_weights_state_buffers_normalizers_and_raw_timings():
    cfg, rows = scalar_fixture(); fits, resources = resource_fixture()
    selection, _ = audit.decisions(rows, resources, cfg)
    audit.validate_resources(resources, fits, selection, rows, {}, np)
    for arm, expected in [('last_two', 2600), ('temporal_affine', 4088), ('gru10', 5488)]:
        item = next(r for r in resources if r['arm'] == arm)
        assert sum(item[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) == expected


@pytest.mark.parametrize('damage', ['timing_sample', 'median', 'p95', 'missing_duration', 'zero_duration',
                                   'bytes', 'normalizer', 'rate', 'seed', 'missing_resource'])
def test_storage_and_timing_tampering_is_not_treated_as_roundoff(damage):
    cfg, rows = scalar_fixture(); fits, resources = resource_fixture()
    selection, _ = audit.decisions(rows, resources, cfg)
    item = resources[0]
    if damage == 'timing_sample': item['timing']['seconds'][10] += .1
    elif damage in ('median', 'p95'): item['timing'][damage + '_seconds'] += .01
    elif damage == 'missing_duration': item['timing']['seconds'].pop()
    elif damage == 'zero_duration': item['timing']['seconds'][0] = 0.
    elif damage == 'bytes': item['parameter_bytes'] -= 4
    elif damage == 'normalizer': item['normalizer_bytes'] -= 8
    elif damage == 'rate': item['learning_rate'] = .003
    elif damage == 'seed': item['seed'] = 88
    else: resources.pop()
    with pytest.raises(ValueError):
        audit.validate_resources(resources, fits, selection, rows, {}, np)


def test_scalar_tolerance_never_changes_scientific_identity_or_gate_roster():
    audit.close({'value': 1.}, {'value': 1. + 5e-11}, 'roundoff')
    for left, right in [(1., 1. + 2e-10), (True, 1), (1, 1.),
                        ({'passed': False}, {'passed': True}), ([1, 2], [1]),
                        ({'a': 1}, {'a': 1, 'b': 2}), (float('nan'), float('nan'))]:
        with pytest.raises(ValueError):
            audit.close(left, right, 'tamper')


def fit_fixture(updates=4096, failed=False):
    receipt = {'status': 'FAILED' if failed else 'PASS', 'error': None,
               'completed_updates': updates, 'requested_updates': 4096, 'learning_rate': .001,
               'optimizer_seconds': 1., 'fit_seconds': 2.,
               'fit_cap_scope': 'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows',
               'timing_scope': 'optimizer loop including batch construction and finite checks, excluding model construction and saved files',
               'files': {name: {} for name in ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json')}}
    if failed:
        receipt['error'] = {'type': 'FitFailure', 'message': 'nonfinite updated Adam state'}
    trace = [{'update': i + 1, 'loss': 1. / (i + 1), 'gradient_norm_before_clip': 2.} for i in range(updates)]
    return receipt, trace


def checkpoint_fixture(arm):
    initial = {'cell.weight': np.arange(578, dtype=np.float32), 'cell.bias': np.zeros(12, np.float32)}
    if arm != 'last_two':
        initial.update({'head.weight': np.zeros((12, 30), np.float32), 'head.bias': np.zeros(12, np.float32)})
    final = {key: value + np.float32(.125) for key, value in initial.items()}
    optimizer = {}
    for name, value in initial.items():
        optimizer.update({name + '/exp_avg': np.full_like(value, .25),
                          name + '/exp_avg_sq': np.full_like(value, .5),
                          name + '/step': np.array(4096., np.float32)})
    return initial, final, optimizer, {key: value.shape for key, value in initial.items()}


def test_complete_and_partial_fit_ledgers_preserve_failure_evidence():
    receipt, trace = fit_fixture()
    audit.validate_fit_receipt(receipt, trace, .001, 1800.)
    for updates in (0, 3, 4096):
        failed, partial = fit_fixture(updates, True)
        audit.validate_fit_receipt(failed, partial, .001, 1800.)
    initial, final, optimizer, shapes = checkpoint_fixture('temporal_affine')
    audit.validate_fit_evidence(initial, final, optimizer, shapes, receipt, np)
    failed, _ = fit_fixture(3, True)
    for name in shapes:
        optimizer[name + '/step'][...] = 4  # Failed update may have advanced Adam before rejection.
    final['head.bias'][0] = np.nan
    optimizer['head.bias/exp_avg_sq'][0] = np.inf
    audit.validate_fit_evidence(initial, final, optimizer, shapes, failed, np)
    audit.validate_fit_evidence(initial, final, {}, shapes, fit_fixture(0, True)[0], np)


@pytest.mark.parametrize('damage', ['trace_gap', 'trace_nan', 'requested', 'rate', 'success_short',
                                   'negative_time', 'cap', 'unknown_failure', 'missing_file'])
def test_fit_receipt_rejects_unexplained_ledger_or_failure(damage):
    receipt, trace = fit_fixture()
    if damage == 'trace_gap': trace[2]['update'] = 4
    elif damage == 'trace_nan': trace[0]['loss'] = float('nan')
    elif damage == 'requested': receipt['requested_updates'] = 4095
    elif damage == 'rate': receipt['learning_rate'] = .003
    elif damage == 'success_short': receipt['completed_updates'] = 4095; trace.pop()
    elif damage == 'negative_time': receipt['optimizer_seconds'] = -1.
    elif damage == 'cap': receipt['fit_seconds'] = 1800.001
    elif damage == 'unknown_failure': receipt.update(status='FAILED', error={'type': 'ValueError', 'message': 'programmer bug'})
    else: receipt['files'].pop('initial.npz')
    with pytest.raises(ValueError):
        audit.validate_fit_receipt(receipt, trace, .001, 1800.)


@pytest.mark.parametrize('damage', ['initial_nan', 'final_nan', 'dtype', 'shape', 'missing_parameter',
                                   'missing_moment', 'foreign_slot', 'missing_all_adam', 'step', 'step_shape'])
def test_every_checkpoint_and_adam_slot_is_owned_and_complete(damage):
    initial, final, optimizer, shapes = checkpoint_fixture('local_affine')
    if damage == 'initial_nan': initial['cell.bias'][0] = np.nan
    elif damage == 'final_nan': final['cell.bias'][0] = np.nan
    elif damage == 'dtype': final['cell.bias'] = final['cell.bias'].astype(np.float64)
    elif damage == 'shape': final['cell.bias'] = final['cell.bias'][:-1]
    elif damage == 'missing_parameter': initial.pop('head.bias')
    elif damage == 'missing_moment': optimizer.pop('head.bias/exp_avg')
    elif damage == 'foreign_slot': optimizer['other/step'] = np.array(4096., np.float32)
    elif damage == 'missing_all_adam': optimizer.clear()
    elif damage == 'step': optimizer['head.bias/step'][...] = 4097
    else: optimizer['head.bias/step'] = np.array([4096.], np.float32)
    with pytest.raises(ValueError):
        audit.validate_fit_evidence(initial, final, optimizer, shapes, fit_fixture()[0], np)


def test_common_initial_pairing_hashes_bytes_and_excludes_only_zero_heads():
    records = [audit.initial_pairing(checkpoint_fixture(arm)[0], arm, np) for arm in PRIMARY]
    assert len({record['common_cell_sha256'] for record in records}) == 1
    assert [record['added_parameters'] for record in records] == [0, 372, 372]
    initial = checkpoint_fixture('temporal_affine')[0]
    initial['cell.weight'][0] += 1.
    assert audit.initial_pairing(initial, 'temporal_affine', np)['common_cell_sha256'] != records[0]['common_cell_sha256']
    initial['head.bias'][0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ValueError, match='zero372'):
        audit.initial_pairing(initial, 'temporal_affine', np)


@pytest.mark.parametrize('damage', [None, 'missing_fit', 'missing_pin', 'changed_pin', 'early_dev', 'fresh_count', 'cached_count'])
def test_all_48_checkpoints_are_bound_before_any_new_dev_decode(damage):
    fits, _ = resource_fixture()
    pins = {fit['key']: {'sha256': hashlib.sha256(fit['key'].encode()).hexdigest(), 'bytes': 123} for fit in fits}
    barrier = {'fresh_fit_attempts': 18, 'cached_fit_records': 30, 'fit_records': 48,
               'dev_loads_this_run': 0, 'dev_exposed_prior': True, 'checkpoints': copy.deepcopy(pins)}
    if damage == 'missing_fit': fits.pop()
    elif damage == 'missing_pin': pins.pop(next(iter(pins)))
    elif damage == 'changed_pin': next(iter(pins.values()))['bytes'] += 1
    elif damage == 'early_dev': barrier['dev_loads_this_run'] = 1
    elif damage == 'fresh_count': barrier['fresh_fit_attempts'] = 17
    elif damage == 'cached_count': barrier['cached_fit_records'] = 29
    if damage is None:
        audit.validate_barrier(barrier, fits, pins)
    else:
        with pytest.raises(ValueError): audit.validate_barrier(barrier, fits, pins)


@pytest.mark.parametrize('raw_witness', [False, True])
def test_temporal_numeric_diagnostic_failure_is_retained_and_never_a_sixth_gate(raw_witness):
    cfg, ordinary = scalar_fixture(); _, resources = resource_fixture()
    before = audit.decisions(ordinary, resources, cfg)
    original = np.zeros((2, 128, 6))
    prediction = np.full_like(original, np.inf) if raw_witness else None
    error = None if raw_witness else {'type': 'NonfiniteDiagnostic', 'message': 'qualified numerical failure'}
    common = {'arm': 'temporal_affine', 'learning_rate': .001, 'seed': 8101, 'recording': DEV[0]}
    check, rows = audit.diagnostic_rows(common, original, prediction, original, np.ones(6), error, np)
    assert check['status'] == 'FAILED' and check['check'] is None
    assert check['prediction_saved'] is raw_witness
    assert [row['horizon'] for row in rows] == [64, 128]
    assert all(row['status'] == 'FAILED' and row['metrics'] is None for row in rows)
    assert audit.decisions(ordinary, resources, cfg) == before


@pytest.mark.parametrize('damage', ['local_error', 'local_changed', 'programming', 'shape', 'nonfinite_original'])
def test_diagnostic_does_not_rescue_local_invariance_or_schema_errors(damage):
    original = np.zeros((1, 128, 6)); prediction = original.copy(); error = None
    common = {'arm': 'temporal_affine', 'learning_rate': .001}
    if damage == 'local_error': common['arm'] = 'last_two'; error = {'type': 'NonfiniteDiagnostic'}
    elif damage == 'local_changed': common['arm'] = 'local_affine'; prediction[0, 0, 0] = 1e-15
    elif damage == 'programming': error = {'type': 'ValueError', 'message': 'unrelated implementation failure'}
    elif damage == 'shape': prediction = prediction[:, :64]
    else: original[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.diagnostic_rows(common, original, prediction, np.zeros((1, 128, 6)), np.ones(6), error, np)


def test_unavailable_diagnostic_recipe_is_explicit_and_has_no_inference():
    check, rows = audit.diagnostic_rows({'arm': 'temporal_affine', 'learning_rate': None},
                                      None, None, np.zeros((1, 128, 6)), np.ones(6), None, np)
    assert check['status'] == 'UNAVAILABLE' and all(row['status'] == 'FAILED' for row in rows)
    with pytest.raises(ValueError):
        audit.diagnostic_rows({'arm': 'temporal_affine', 'learning_rate': None},
                              None, np.zeros((1, 128, 6)), np.zeros((1, 128, 6)), np.ones(6), None, np)


def test_independent_reference_time_alignment_and_no_future_target_access():
    q = np.arange(32 * 6, dtype=np.float64).reshape(1, 32, 6)
    u = 1000 + q
    future = 2000 + np.arange(128 * 6, dtype=np.float64).reshape(1, 128, 6)
    batch = {'q_context': q, 'u_context': u, 'future_u': future, 'target': object()}
    linear = np.zeros((6, 25))
    for j in range(6): linear[j, j] = .5; linear[j, 12+j] = 2.; linear[j, 18+j] = -1.
    actual = audit.reference_prediction('linear_frozen', {'linear_frozen': linear}, batch, np)
    expected = np.empty_like(actual); current = q[:, 31].copy(); previous_u = u[:, 30].copy()
    for t in range(128):
        current = .5 * current + 2. * future[:, t] - previous_u
        expected[:, t] = current; previous_u = future[:, t]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(audit.reference_prediction('persistence', {}, batch, np), np.repeat(q[:, 31:], 128, axis=1))
    bank = {}
    for h in range(1, 129):
        weight = np.zeros((6, 193 + 6*h))
        for j in range(6):
            weight[j, j] = 1.; weight[j, 96+j] = 2.; weight[j, 192+6*(h-1)+j] = 3.; weight[j, -1] = 4.
        bank[f'h{h:03d}'] = weight
    ridge = audit.reference_prediction('causal_ridge_1', {'causal_ridge_1': bank}, batch, np)
    np.testing.assert_array_equal(ridge, q[:, 16:17] + 2.*u[:, 15:16] + 3.*future + 4.)
    changed = {**batch, 'future_u': future.copy(), 'target': 'unread poison'}
    changed['future_u'][:, 64:] += 1e6
    altered = audit.reference_prediction('causal_ridge_1', {'causal_ridge_1': bank}, changed, np)
    np.testing.assert_array_equal(altered[:, :64], ridge[:, :64])
    assert not np.array_equal(altered[:, 64:], ridge[:, 64:])


def opaque_admission(tmp_path, monkeypatch):
    """Construct only temporary JSON/opaque bytes, never a loadable model bank."""
    root = tmp_path / 'repo'
    study = root / 'output/robot-history-initialization-study-v1'
    engineering = root / 'output/robot-history-initialization-engineering-v1'
    registration = root / 'research/robot-history-initialization-registration.json'
    monkeypatch.setattr(audit, 'ROOT', root)
    monkeypatch.setattr(audit, 'COMMIT', 'fabricated-prefit-commit')
    for key in audit.THREADS: monkeypatch.setenv(key, '1')
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
        return tagged(path)
    def desc(path):
        value = path.read_bytes()
        return {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    def tagged(path):
        return {'path': str(path), **desc(path)}
    sources = {}
    for name in audit.SOURCES:
        raw = ('# fabricated source ' + name).encode()
        write(root / name, raw); write(study / 'sources' / name, raw)
        sources[name] = desc(root / name)
    launcher = root / 'scripts/launch_robot_history_initialization.py'
    write(launcher, b'# fabricated launcher')
    data = {'saved.npz': write(root / 'output/ancestor/saved.npz', b'opaque non-array bytes')}
    payloads, closures, parent_shas, parent_inputs = {}, {}, {}, {}
    for group in ('transition', 'structured'):
        folder = root / f'output/robot-{group}-study-v1'
        eng = root / f'output/robot-{group}-engineering-v1'
        payloads[group] = {}
        for name in audit.PAYLOADS[group]:
            value = ('opaque ' + group + '/' + name).encode()
            payloads[group][name] = write(folder / name, value)
            write(study / audit.copied_payload(group, name), value)
        parent_reg = root / f'research/robot-{group}-registration.json'
        write(parent_reg, {'fabricated_parent': group})
        sha = parent_shas[group] = desc(parent_reg)['sha256']
        closure = {
            'manifest': write(folder / 'manifest.json', {'files': {name: desc(Path(item['path'])) for name, item in payloads[group].items()}}),
            'receipt': write(folder / 'receipt.json', {'status': 'PASS', 'registration_sha256': sha}),
            'process': write(eng / 'run-process-01.json', {'returncode': 0, 'registration_sha256': sha})}
        joined = {'manifest': closure['manifest'], 'producer_receipt': closure['receipt'],
                  'run_receipt': closure['process'], 'parent_data': data}
        parent_inputs[group] = joined
        parent_audit = {'status': 'PASS', 'agreement': True, 'registration_sha256': sha, 'inputs': joined}
        if group == 'structured':
            parent_audit.update(source_pins={n: sources[n] for n in audit.SOURCES[:21]},
                                auditor=tagged(root / 'scripts/audit_robot_structured.py'))
        closure['audit'] = write(root / f'output/robot-{group}-audit-v1/audit.json', parent_audit)
        closure['audit_process'] = write(eng / 'audit-process-01.json', {'returncode': 0,
            'audit_output': desc(Path(closure['audit']['path']))})
        closures[group] = closure
    base_config = {'partitions': {'dev': list(DEV)}}
    prior = {'sources': {n: sources[n] for n in audit.SOURCES[:21]}, 'config': base_config,
             'parent_closure': closures['transition'], 'parent_registration_sha256': parent_shas['transition'], 'data': data}
    parent_calls = []
    def parent_auth(folder, process):
        parent_calls.append((folder, process))
        return prior, parent_inputs['structured']
    monkeypatch.setitem(sys.modules, 'audit_robot_structured', SimpleNamespace(authenticate=parent_auth))
    cfg = {**base_config, 'version': 'robot-history-initialization-study-v1', 'arms': list(PRIMARY),
           'comparison_arms': list(ARMS), 'fit_cap_seconds': 1800., 'wall_cap_seconds': 14400.,
           'latency_ratio': 1.25, 'mean_reduction': .05, 'file_harm_ratio': 1.02, 'cached_parent_refit': False,
           'older_permutation': [*range(29, -1, -1), 30, 31], 'diagnostic_gate': False}
    commands = [
        ['.venv/bin/ruff', 'check', 'src/openjev/research/robot_history_initializer.py',
         'tests/test_robot_history_initializer.py', 'scripts/robot_history_initialization_study.py',
         'tests/test_robot_history_initialization_study.py', 'scripts/launch_robot_history_initialization.py'],
        ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_history_initializer.py',
         'tests/test_robot_history_initialization_study.py']]
    qualification = {'status': 'PASS', 'sources': sources, 'launcher': desc(launcher), 'sources_unchanged': True,
                     'thread_env': dict.fromkeys(audit.THREADS, '1'), 'created_utc': '2026-01-01T00:00:00+00:00',
                     'prior_qualification': write(engineering / 'qualification-01.json', {'status': 'FAILED'}), 'commands': []}
    for index, command in enumerate(commands, 1):
        log = engineering / f'qualification-02/command-{index}.log'
        write(log, b'fabricated qualification output')
        qualification['commands'].append({'command': command, 'returncode': 0, 'external_timeout': False,
                                         'log': str(log), 'sha256': desc(log)['sha256'], 'seconds': .5})
    qpath = engineering / 'qualification-02.json'
    plan = {'version': cfg['version'], 'config': cfg, 'sources': sources, 'launcher': desc(launcher),
            'created_utc': '2026-01-01T00:01:00+00:00', 'parent_closure': closures,
            'parent_payloads': payloads, 'parent_registration_sha256': parent_shas, 'data': data,
            'qualification': write(qpath, qualification)}
    clock = 'mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'
    runtime = {'python': sys.version, 'platform': audit.platform.platform(), 'machine': audit.platform.machine(),
               'torch_threads': 1, 'thread_env': dict.fromkeys(audit.THREADS, '1'), 'clock': clock,
               'numpy': 'fabricated-package-version', 'torch': 'fabricated-package-version'}
    monkeypatch.setattr(audit.importlib.metadata, 'version', lambda _: 'fabricated-package-version')
    write(study / 'runtime.json', runtime)
    log = engineering / 'run-process-01.log'; write(log, b'fabricated completed campaign')
    process_path = engineering / 'run-process-01.json'
    launch_path = engineering / 'run-launch-01.json'
    receipt = {'status': 'PASS', 'fits': 48, 'fresh_fit_attempts': 18, 'cached_fit_records': 30,
               'rows': 208, 'permutation_metric_rows': 36, 'raw_decodes': 0, 'reference_refits': 0,
               'parent_refits': 0, 'saved_fit_loads': 7, 'saved_dev_loads': 2,
               'confirmation_access': False, 'official_test_access': False,
               'seconds': 10., 'clock': clock, 'monotonic_seconds': 10.}
    def seal():
        plan['qualification'] = write(qpath, qualification)
        write(registration, plan); write(study / 'registration.json', registration.read_bytes())
        sha = desc(registration)['sha256']; monkeypatch.setattr(audit, 'PLAN_SHA', sha)
        launch = {'command': list(audit.COMMAND), 'prefit_commit': audit.COMMIT,
                  'started_utc': '2026-01-01T00:02:00+00:00', 'registration_sha256': sha,
                  'launcher': desc(launcher), 'thread_env': dict.fromkeys(audit.THREADS, '1'), 'scope': 'fabricated'}
        write(launch_path, launch)
        write(process_path, {**launch, 'returncode': 0, 'elapsed_seconds': 11., 'external_timeout': False, 'log': desc(log)})
        write(study / 'receipt.json', {**receipt, 'registration_sha256': sha})
        write(study / 'manifest.json', {'files': {str(p.relative_to(study)): desc(p) for p in study.rglob('*')
              if p.is_file() and p.name not in ('manifest.json', 'receipt.json')}})
    seal()
    return SimpleNamespace(root=root, study=study, engineering=engineering, process=process_path, launch=launch_path,
                           registration=registration, plan=plan, qualification=qualification, receipt=receipt,
                           write=write, desc=desc, seal=seal, parent_calls=parent_calls)


def test_opaque_admission_joins_all_sources_snapshots_two_parents_and_successful_qualification(tmp_path, monkeypatch):
    fixture = opaque_admission(tmp_path, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata admission must not decode arrays'))
    plan, inputs = audit.authenticate(fixture.study, fixture.process)
    assert plan == fixture.plan and len(plan['sources']) == 29
    assert {group: len(value) for group, value in plan['parent_payloads'].items()} == {'structured': 131, 'transition': 32}
    assert inputs['prior_qualification']['path'].endswith('qualification-01.json')
    assert inputs['qualification']['path'].endswith('qualification-02.json')
    assert len(inputs['qualification_logs']) == 2 and len(fixture.parent_calls) == 1


@pytest.mark.parametrize('damage', ['registration', 'source', 'snapshot', 'copied_payload', 'parent_payload',
                                   'data', 'manifest', 'extra_file', 'qualification_argv', 'qualification_sources',
                                   'qualification_chronology', 'qualification_log', 'prior_qualification',
                                   'process_argv', 'process_commit', 'process_log', 'process_timeout', 'process_returncode',
                                   'producer_fresh_count', 'producer_confirmation', 'runtime', 'parent_audit_join'])
def test_admission_blocks_metadata_or_opaque_bytes_tampering_without_array_access(tmp_path, monkeypatch, damage):
    fixture = opaque_admission(tmp_path, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('admission failure reached an array decoder'))
    if damage == 'registration': fixture.registration.write_bytes(b'{}')
    elif damage in ('source', 'snapshot'):
        prefix = fixture.root if damage == 'source' else fixture.study / 'sources'
        (prefix / audit.SOURCES[0]).write_bytes(b'changed source')
    elif damage in ('copied_payload', 'parent_payload'):
        name, item = next(iter(fixture.plan['parent_payloads']['structured'].items()))
        path = fixture.study / audit.copied_payload('structured', name) if damage == 'copied_payload' else Path(item['path'])
        path.write_bytes(b'changed payload')
    elif damage == 'data': Path(fixture.plan['data']['saved.npz']['path']).write_bytes(b'changed data')
    elif damage == 'manifest': fixture.write(fixture.study / 'manifest.json', {'files': {}})
    elif damage == 'extra_file': fixture.write(fixture.study / 'unlisted.bin', b'opaque extra')
    elif damage == 'qualification_argv':
        fixture.qualification['commands'][1]['command'].append('--unexpected'); fixture.seal()
    elif damage == 'qualification_sources':
        fixture.qualification['sources'] = {}; fixture.seal()
    elif damage == 'qualification_chronology':
        fixture.qualification['created_utc'] = '2026-01-01T00:03:00+00:00'; fixture.seal()
    elif damage == 'qualification_log': Path(fixture.qualification['commands'][0]['log']).write_bytes(b'changed log')
    elif damage == 'prior_qualification': Path(fixture.qualification['prior_qualification']['path']).write_bytes(b'changed original failure')
    elif damage.startswith('process_'):
        process = json.loads(fixture.process.read_text())
        if damage == 'process_argv': process['command'][-1] = 'different-study'
        elif damage == 'process_commit': process['prefit_commit'] = 'different-commit'
        elif damage == 'process_log': fixture.process.with_suffix('.log').write_bytes(b'changed process output')
        elif damage == 'process_timeout': process['external_timeout'] = True
        else: process['returncode'] = 1
        fixture.write(fixture.process, process)
    elif damage.startswith('producer_'):
        fixture.receipt['fresh_fit_attempts' if damage == 'producer_fresh_count' else 'confirmation_access'] = 17 if damage == 'producer_fresh_count' else True
        fixture.seal()
    elif damage == 'runtime':
        runtime = json.loads((fixture.study / 'runtime.json').read_text()); runtime['torch_threads'] = 2
        fixture.write(fixture.study / 'runtime.json', runtime); fixture.seal()
    else:
        item = fixture.plan['parent_closure']['structured']['audit']
        value = json.loads(Path(item['path']).read_text()); value['inputs']['manifest']['sha256'] = '0' * 64
        fixture.plan['parent_closure']['structured']['audit'] = fixture.write(Path(item['path']), value)
        fixture.seal()
    with pytest.raises(ValueError):
        audit.authenticate(fixture.study, fixture.process)


def test_opaque_pin_rejects_symlink_relative_path_and_replaced_bytes(tmp_path):
    path = tmp_path / 'opaque.npz'; path.write_bytes(b'not an archive')
    item = {'path': str(path), 'bytes': 14, 'sha256': hashlib.sha256(b'not an archive').hexdigest()}
    assert audit.checked_pin(item, 'fixture') == item
    alias = tmp_path / 'alias.npz'; alias.symlink_to(path)
    for altered in ({**item, 'path': str(alias)}, {**item, 'path': 'opaque.npz'}, {**item, 'bytes': 13}):
        with pytest.raises(ValueError): audit.checked_pin(altered, 'fixture')
