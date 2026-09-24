"""Independent exact-update trace, paired exposure, saved-state and gate oracles."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_update_learning as a

SPEC = a.StudyProfile('fabricated', 942001, (942101, 942102, 942103), 512, 128, 64, 2, 2, 30.)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def allocation(arm='rounded', *, at_cap=False):
    """Hand-authored exact two-plus-two trace; no controller/model invocation."""
    trace, stages = [], []
    cursor = 0
    model, opt = digest('initial model'), digest('initial optimizer')
    initial_model, initial_opt = model, opt
    boundary = None
    for index, kind, start in ((1, 'prefix', .2), (2, 'joint', 2.2)):
        if index == 2:
            old = opt
            opt = digest('fresh joint optimizer')
            boundary = {'joint_cursor': 0, 'model_sha256': model, 'stage1_optimizer_kind': 'prefix',
                'optimizer_before_sha256': old, 'optimizer_after_sha256': opt,
                'stage2_optimizer_kind': 'joint', 'optimizer_reset': True}
        first_model, first_opt, first_cursor = model, opt, cursor
        for offset in range(2):
            next_model, next_opt = digest(('model', index, offset)), digest(('Adam', index, offset))
            params, size = (3, 2560) if kind == 'prefix' else (4, 2816)
            delta = dict.fromkeys(a.CONTROLLER_WORK, 0)
            delta.update(model_snapshot_calls=1, model_snapshot_tensors=4, model_snapshot_bytes=2816,
                optimizer_snapshot_calls=1, optimizer_snapshot_tensors=3 * params if offset else 0,
                optimizer_snapshot_bytes=2 * size + 4 * params if offset else 0,
                model_hash_calls=2, optimizer_hash_calls=2, check_calls=1, clock_calls=1)
            delta[kind + '_update_calls'] = 1
            trace.append({'attempt': len(trace) + 1, 'stage': index, 'kind': kind, 'deadline_seconds': 30.,
                'start_elapsed': start + .1 + offset * .4, 'completed_elapsed': start + .3 + offset * .4,
                'accepted': True, 'rolled_back': False, 'error': None,
                'cursor_before': cursor, 'cursor_attempted': cursor + int(kind == 'joint'),
                'cursor_retained': cursor + int(kind == 'joint'),
                'model_before_sha256': model, 'optimizer_before_sha256': opt,
                'model_attempted_sha256': next_model, 'optimizer_attempted_sha256': next_opt,
                'model_retained_sha256': next_model, 'optimizer_retained_sha256': next_opt,
                'result': {'loss': .5 + offset, 'diagnostics': {}, 'work': {'fabricated_update_calls': 1}},
                'work_delta': delta})
            model, opt = next_model, next_opt
            cursor += int(kind == 'joint')
        stages.append({'stage': index, 'kind': kind, 'target_updates': 2, 'deadline_seconds': 30.,
            'start_elapsed': start, 'stopped_elapsed': start + .8, 'status': 'PASS', 'termination': 'completed_updates',
            'trace_start': 2 * (index - 1), 'trace_stop': 2 * index, 'attempted_updates': 2, 'accepted_updates': 2,
            'joint_cursor_start': first_cursor, 'joint_cursor_end': cursor,
            'model_start_sha256': first_model, 'optimizer_start_sha256': first_opt,
            'model_end_sha256': model, 'optimizer_end_sha256': opt, 'overrun_seconds': 0.})
    checkpoints = []
    for label, begin, end, state, adam, position in (
        ('initial', .1, .15, initial_model, initial_opt, 0),
        ('boundary', 1.1, 1.2, stages[0]['model_end_sha256'], stages[0]['optimizer_end_sha256'], 0),
        ('final', 3.1, 3.2, model, opt, cursor)):
        checkpoints.append({'label': label, 'start_elapsed': begin, 'completed_elapsed': end, 'seconds': end - begin,
            'joint_cursor': position, 'model_sha256': state, 'optimizer_sha256': adam, 'metadata': {},
            'accepted': True, 'rolled_back': False, 'error': None})
    finish = 30. if at_cap else 3.5
    final_record = {'label': 'summary', 'start_elapsed': 3.3, 'completed_elapsed': finish, 'seconds': finish - 3.3,
        'joint_cursor': 2, 'model_sha256': model, 'optimizer_sha256': opt, 'metadata': {},
        'accepted': True, 'rolled_back': False, 'error': None}
    # Four updates plus initial/boundary/final/summary protections = eight snapshots.
    total = dict.fromkeys(a.CONTROLLER_WORK, 0)
    total.update(validation_calls=1, optimizer_constructions=2, checkpoint_calls=3, final_summary_calls=1,
        model_snapshot_calls=8, model_snapshot_tensors=32, model_snapshot_bytes=22528,
        optimizer_snapshot_calls=8, optimizer_snapshot_tensors=54, optimizer_snapshot_bytes=27208,
        model_hash_calls=23, optimizer_hash_calls=23, prefix_update_calls=2, joint_update_calls=2,
        check_calls=28, clock_calls=25)
    final_work = dict.fromkeys(a.CONTROLLER_WORK, 0)
    final_work.update(model_snapshot_calls=1, model_snapshot_tensors=4, model_snapshot_bytes=2816,
        optimizer_snapshot_calls=1, optimizer_snapshot_tensors=12, optimizer_snapshot_bytes=5648,
        model_hash_calls=2, optimizer_hash_calls=2, check_calls=3, clock_calls=2, final_summary_calls=1)
    return {'version': 'finite-update-allocation-v1', 'arm': 'prefix_then_joint', 'status': 'PASS',
        'termination': 'completed_updates', 'prefix_updates': 2, 'joint_updates': 2, 'max_seconds': 30.,
        'max_attempts': 100000, 'metadata': {}, 'initial_model_sha256': initial_model, 'initial_optimizer_sha256': initial_opt,
        'final_model_sha256': model, 'final_optimizer_sha256': opt, 'joint_cursor': 2,
        'attempted_updates': 4, 'accepted_updates': 4, 'accepted_joint_updates': 2, 'accepted_prefix_updates': 2,
        'trace': trace, 'stages': stages, 'boundary': boundary, 'checkpoints': checkpoints,
        'work': total, 'update_work': {'fabricated_update_calls': 4}, 'timed_seconds': finish, 'overrun_seconds': 0.,
        'final_summary': {}, 'final_summary_record': final_record, 'final_summary_seconds': finish - 3.2,
        'final_summary_work': final_work, 'compute_matched': False, 'timing_scope': 'fabricated',
        'work_scope': 'final summary included'}


@pytest.mark.parametrize('arm', a.ARMS)
@pytest.mark.parametrize('at_cap', [False, True])
def test_hand_trace_exact_counts_and_inclusive_checkpoint_summary_cap(arm, at_cap):
    record = allocation(arm, at_cap=at_cap)
    result = a.validate_allocation(record, arm, spec=SPEC)
    assert result['attempted_updates'] == 4
    assert sum(row['late_updates'] for row in result['stages']) == 0
    assert result['accepted_prefix_updates'] == result['accepted_joint_updates'] == 2


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(max_seconds=40.),
    lambda r: r.update(compute_matched=True),
    lambda r: r.update(accepted_updates=3),
    lambda r: r.update(prefix_updates=1),
    lambda r: r.update(status='FAILED_TIMEOUT'),
    lambda r: r['trace'][1].update(accepted=False, rolled_back=True),
    lambda r: r['trace'][1].update(completed_elapsed=30.1),
    lambda r: r['trace'][1].update(cursor_retained=1),
    lambda r: r['trace'][1].update(optimizer_retained_sha256=digest('wrong retained optimizer')),
    lambda r: r['trace'][1].update(model_before_sha256=digest('broken chain')),
    lambda r: r['trace'][2].update(cursor_before=1),
    lambda r: r['trace'][2].update(deadline_seconds=60.),
    lambda r: r['trace'][2].update(start_elapsed=0.),
    lambda r: r['trace'][0]['work_delta'].update(model_snapshot_bytes=2560),
    lambda r: r['trace'][1]['work_delta'].update(optimizer_restore_calls=1),
    lambda r: r['work'].update(check_calls=0),
    lambda r: r['update_work'].update(fabricated_update_calls=2),
    lambda r: r['stages'][0].update(accepted_updates=1),
    lambda r: r['stages'][0].update(target_updates=3),
    lambda r: r['stages'][0].update(termination='deadline_reached'),
    lambda r: r['boundary'].update(optimizer_reset=False),
    lambda r: r['boundary'].update(optimizer_after_sha256=digest('other fresh optimizer')),
    lambda r: r['checkpoints'][1].update(joint_cursor=1),
    lambda r: r['checkpoints'][2].update(completed_elapsed=30.1),
    lambda r: r['final_summary_record'].update(completed_elapsed=30.1),
    lambda r: r['final_summary_record'].update(accepted=False),
    lambda r: r['final_summary_work'].update(model_restore_calls=1),
    lambda r: r.update(final_model_sha256=digest('other final state')),
])
def test_exact_count_cap_and_callback_corruptions_rejected(mutation):
    record = allocation()
    mutation(record)
    with pytest.raises((ValueError, KeyError)):
        a.validate_allocation(record, 'rounded', spec=SPEC)



def optimizer(kind, steps):
    shapes = list(a.PARAMETER_SHAPES.values())[:3 if kind == 'prefix' else 4]
    group = {'params': list(range(len(shapes))), 'lr': .003, 'betas': [.9, .999], 'eps': 1e-8,
             'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'capturable': False,
             'differentiable': False, 'foreach': None, 'fused': None}
    state = {}
    for index, shape in enumerate(shapes):
        if steps:
            state[str(index)] = {'step': {'kind': 'tensor', 'dtype': 'torch.float32', 'shape': [], 'values': float(steps)},
                'exp_avg': {'kind': 'tensor', 'dtype': 'torch.float64', 'shape': list(shape), 'values': np.zeros(shape).tolist()},
                'exp_avg_sq': {'kind': 'tensor', 'dtype': 'torch.float64', 'shape': list(shape), 'values': np.ones(shape).tolist()}}
    return {'state': state, 'param_groups': [group]}


@pytest.mark.parametrize('kind,tensors,size', [('joint', 12, 5648), ('prefix', 9, 5132)])
def test_optimizer_json_hash_and_actual_storage(kind, tensors, size):
    payload = optimizer(kind, 3)
    value = a.optimizer_state(payload, kind, 3)
    assert value['sha256'] == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert (value['tensors'], value['bytes']) == (tensors, size)
    assert a.optimizer_state(optimizer(kind, 0), kind, 0)['bytes'] == 0


@pytest.mark.parametrize('mutation', [
    lambda p: p['state']['0']['step'].update(values=4.),
    lambda p: p['state']['0']['step'].update(dtype='torch.float64'),
    lambda p: p['state']['0']['exp_avg'].update(shape=[256]),
    lambda p: p['state']['0']['exp_avg_sq']['values'][0][0].__setitem__(0, -1.),
    lambda p: p['state'].pop('3'),
    lambda p: p['param_groups'][0].update(lr=.01),
    lambda p: p['param_groups'][0].update(betas=[.8, .999]),
])
def test_optimizer_boundary_corruptions(mutation):
    payload = optimizer('joint', 3)
    mutation(payload)
    with pytest.raises(ValueError):
        a.optimizer_state(payload, 'joint', 3)


def metric_fixture():
    rows = [{'arm': arm, 'seed': seed, 'horizon': h, 'blind_regret': .3 if arm == 'rounded' else .4,
             'blind_cost_mse': .3, 'observed_cost_mse': .2, 'observed_kl': .05,
             'blind_survival_mae': .01, 'observed_survival_mae': .01, 'shuffled_blind_regret': .5}
            for arm in a.ARMS for seed in SPEC.seeds for h in a.HORIZONS]
    base = [{'horizon': h, 'blind_regret': 1., 'blind_cost_mse': 1.} for h in a.HORIZONS]
    fits = [{'arm': arm, 'seed': seed, 'seconds': 41.} for arm in a.ARMS for seed in SPEC.seeds]
    return rows, base, fits


def test_full_advance_and_original_gates_are_separate():
    rows, baseline, fits = metric_fixture()
    gates = a.gates(rows, baseline, {'train': 400, 'base': 100}, spec=SPEC)
    output = a.allocation_comparisons(rows, fits, gates, spec=SPEC)
    assert output['advance']['passed'] and len(output['advance']['conditions']) == 19
    assert len(output['paired']) == 12
    rows[-1]['blind_regret'] = .6
    gates = a.gates(rows, baseline, {'train': 400, 'base': 100}, spec=SPEC)
    assert not a.allocation_comparisons(rows, fits, gates, spec=SPEC)['advance']['passed']


@pytest.mark.parametrize('fault', ['one_worse_pair', 'insufficient_mean', 'zero_reference', 'support'])
def test_no_averages_or_zero_denominator_rescue(fault):
    rows, baseline, fits = metric_fixture()
    counts = {'train': 400, 'base': 100}
    if fault == 'one_worse_pair':
        next(r for r in rows if r['arm'] == 'rounded' and r['horizon'] == 4)['blind_regret'] = .401
    elif fault == 'insufficient_mean':
        for row in rows:
            if row['arm'] == 'rounded':
                row['blind_regret'] = .37
    elif fault == 'zero_reference':
        for row in rows:
            row['blind_regret'] = 0.
    else:
        counts['train'] = 255
    gates = a.gates(rows, baseline, counts, spec=SPEC)
    result = a.allocation_comparisons(rows, fits, gates, spec=SPEC)
    assert not result['advance']['passed']
    if fault == 'zero_reference':
        assert all(r['relative_reduction'] is None and r['tied_zero'] for r in result['mean_regret'])


def test_profile_whitelist_is_immutable_and_retains_scientific_support(tmp_path):
    with pytest.raises(TypeError):
        a.PROFILES['other'] = SPEC
    with pytest.raises(FrozenInstanceError):
        a.SCIENCE.train_attempts = 8
    with pytest.raises(ValueError, match='profile'):
        a.audit(tmp_path, profile='anything')
    spec = a.PROFILES['engineering-942001']
    rows, baseline, _ = metric_fixture()
    rows = [row for row in rows if row['seed'] == spec.seeds[0]]
    gates = a.gates(rows, baseline, {'train': 8, 'base': 8}, spec=spec)
    assert not any(value['passed'] for criteria in gates.values() for value in criteria.values())


def test_incomplete_manifest_fails_before_any_array_decode(tmp_path, monkeypatch):
    rows, _, fits = metric_fixture()
    del rows
    summary = {'version': 'finite-update-learning-v1', 'config': SPEC.config, 'fits': fits, 'files': {}}
    # Match the declared rotated fit order; absence of payloads is then the first failure.
    summary['fits'] = [{'arm': arm, 'seed': seed} for i, seed in enumerate(SPEC.seeds) for arm in a.ARMS[i:] + a.ARMS[:i]]
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('forbidden array decode'))
    with pytest.raises(ValueError, match='payload roster'):
        a.validate_metadata(tmp_path, summary, spec=SPEC)


def raw_state_hash(arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        metadata = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        result.update(len(metadata).to_bytes(8, 'little') + metadata + value.tobytes())
    return result.hexdigest()


def boundary_fixture():
    allocations, arrays, optimizers = {}, {}, {}
    seed = 942101
    for arm in a.ARMS:
        allocation_record = {'accepted_joint_updates': 1,
            'stages': [{'accepted_updates': 1}, {'accepted_updates': 1}], 'checkpoints': [],
            'boundary': {'optimizer_after_sha256': a.optimizer_state(optimizer('joint', 0), 'joint', 0)['sha256']}}
        for label in ('initial', 'boundary', 'final'):
            values = {name: np.zeros(shape, np.float64) for name, shape in a.PARAMETER_SHAPES.items()}
            if label != 'initial':
                values['hazard_logits'][:] = 1 if label == 'boundary' else 2
            if label == 'final':
                values['cost_logits'][:] = 3
            if label == 'initial' and arm == 'matched_free':
                values['transition_logits'][:] = -np.log(8.)
            arrays[label, arm, seed] = values
            kind = 'prefix' if label != 'final' else 'joint'
            steps = 0 if label == 'initial' else 1
            payload = optimizer(kind, steps)
            optimizers[label, arm, seed] = payload
            allocation_record['checkpoints'].append({'model_sha256': raw_state_hash(values),
                'optimizer_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()})
        allocations[arm, seed] = allocation_record
    return allocations, arrays, optimizers


def test_independent_saved_model_and_optimizer_boundary_join():
    allocations, arrays, optimizers = boundary_fixture()
    before = copy.deepcopy(arrays)
    result = a.verify_boundaries(allocations, arrays, optimizers, np)
    assert len(result) == 3 and all(r['optimizer_and_model_boundaries_joined'] for r in result)
    assert all(np.array_equal(value, before[key][name]) for key, record in arrays.items() for name, value in record.items())


@pytest.mark.parametrize('fault', ['model', 'optimizer', 'prefix_head', 'restart', 'shared_initial', 'dtype'])
def test_decoded_boundary_mutations_rejected(fault):
    allocations, arrays, optimizers = boundary_fixture()
    arm, seed = 'rounded', 942101
    if fault == 'model':
        arrays['final', arm, seed]['transition_logits'][0, 0, 0] = .1
    elif fault == 'optimizer':
        optimizers['final', arm, seed]['state']['0']['exp_avg']['values'][0][0][0] = .1
    elif fault == 'prefix_head':
        arrays['boundary', arm, seed]['cost_logits'][0, 0] = .1
        allocations[arm, seed]['checkpoints'][1]['model_sha256'] = raw_state_hash(arrays['boundary', arm, seed])
    elif fault == 'restart':
        allocations[arm, seed]['boundary']['optimizer_after_sha256'] = digest('bad restart')
    elif fault == 'shared_initial':
        arrays['initial', arm, seed]['hazard_logits'][0, 0] = .1
        allocations[arm, seed]['checkpoints'][0]['model_sha256'] = raw_state_hash(arrays['initial', arm, seed])
    else:
        arrays['final', arm, seed]['hazard_logits'] = arrays['final', arm, seed]['hazard_logits'].astype(np.float32)
    with pytest.raises(ValueError):
        a.verify_boundaries(allocations, arrays, optimizers, np)


def test_independent_separable_logit_balancing_has_uniform_solution():
    # A positive rank-one matrix becomes uniform after one row/column sweep.
    # This hand oracle distinguishes raw equality from probability equality.
    _, arrays, _ = boundary_fixture()
    raw = np.log(np.arange(1., 9.)[:, None] * np.arange(2., 10.)[None, :])
    for arm in ('original_free', 'rounded'):
        arrays['initial', arm, 942101]['transition_logits'][:] = raw
    before = copy.deepcopy(arrays)
    result = a.initial_transition_checks(arrays, np)
    assert len(result) == 1 and result[0]['matched_max_abs'] <= 1e-12
    assert result[0]['row_residual_max'] <= 1e-12
    assert not np.array_equal(arrays['initial', 'original_free', 942101]['transition_logits'],
                              arrays['initial', 'matched_free', 942101]['transition_logits'])
    assert all(value.tobytes() == before[key][name].tobytes()
               for key, record in arrays.items() for name, value in record.items())


@pytest.mark.parametrize('arm,name', [
    ('original_free', 'transition_logits'), ('matched_free', 'transition_logits'),
    ('rounded', 'emission_logits'), ('matched_free', 'hazard_logits'), ('original_free', 'cost_logits'),
])
def test_initial_pairing_rejects_the_specific_incorrect_control(arm, name):
    _, arrays, _ = boundary_fixture()
    arrays['initial', arm, 942101][name].flat[0] += .01
    with pytest.raises(ValueError):
        a.initial_transition_checks(arrays, np)


def test_raw_pairing_includes_signed_zero_bytes():
    _, arrays, _ = boundary_fixture()
    arrays['initial', 'rounded', 942101]['transition_logits'].flat[0] = -0.
    with pytest.raises(ValueError, match='raw transition'):
        a.initial_transition_checks(arrays, np)


@pytest.mark.parametrize('arm', a.ARMS)
def test_field_kernel_prior_and_constructor_work_have_separate_scopes(arm):
    observations = np.array([[0, 4, 4], [1, 2, 3]], np.int64)
    endpoint = a.forward_work(arm, a.endpoint_work(observations, True, np), prefix=False)
    assert len(endpoint) == 71
    assert endpoint['probability_field_calls'] == endpoint['adapter_check_calls'] == 2
    assert endpoint['factorization_calls'] == endpoint['emission_softmax_calls'] == endpoint['hazard_sigmoid_calls'] == 2
    assert endpoint['factor_product_entries'] == 3072
    assert endpoint['prefix_filter_rows'] == 16
    assert endpoint['blind_transition_rows'] == endpoint['event_probability_rows'] == 6
    assert endpoint['observed_branch_calls'] == 3 and endpoint['observed_branch_rows'] == 4
    assert endpoint['absorbed_rows'] == 1
    assert endpoint['cost_head_softmax_calls'] == 3 and endpoint['cost_head_probability_rows'] == 24
    assert endpoint['prior_transition_log_softmax_calls'] == endpoint['prior_emission_log_softmax_calls'] == 0
    if arm == 'rounded':
        assert endpoint['transition_softmax_calls'] == 0
        assert endpoint['row_logsumexp_calls'] == endpoint['column_logsumexp_calls'] == 8
        assert endpoint['completed_sweeps'] == 8 and endpoint['check_calls'] == 22
        assert endpoint['normalization_entries'] == 4096 and endpoint['normalization_vectors'] == 512
    else:
        assert endpoint['transition_softmax_calls'] == 2
        assert endpoint['rounded_transition_calls'] == endpoint['check_calls'] == 0
    prior = a.forward_work(arm, dict.fromkeys(a.WORK_KEYS, 0), prefix=True, prior=True)
    assert prior['factorization_calls'] == prior['probability_field_calls'] == 1
    assert prior['cost_head_softmax_calls'] == 0
    assert prior['prior_emission_log_softmax_calls'] == 1
    assert prior['prior_hazard_logsigmoid_calls'] == 2 and prior['prior_logsigmoid_entries'] == 64
    assert prior['prior_normalization_vectors'] == (8 if arm == 'rounded' else 40)
    assert prior['prior_normalization_entries'] == (32 if arm == 'rounded' else 288)
    construction = a.construction_work(arm)
    assert len(construction) == 71 and construction['adapter_check_calls'] == 1
    assert construction['factorization_calls'] == construction['probability_field_calls'] == 0
    assert construction['matching_copy_entries'] == (256 if arm == 'matched_free' else 0)
    assert construction['rounded_transition_calls'] == (1 if arm == 'matched_free' else 0)
    assert construction['check_calls'] == (11 if arm == 'matched_free' else 0)


def test_all_three_arms_require_the_same_controller_schedule():
    for arm in a.ARMS:
        record = allocation(arm)
        assert record['arm'] == 'prefix_then_joint'
        record['arm'] = arm
        with pytest.raises(ValueError, match='registered allocation'):
            a.validate_allocation(record, arm, spec=SPEC)


def test_relative_gain_cannot_rescue_a_failed_absolute_criterion():
    rows, baseline, fits = metric_fixture()
    next(row for row in rows if row['arm'] == 'rounded' and row['horizon'] == 8)['observed_kl'] = .10001
    gates = a.gates(rows, baseline, {'train': 400, 'base': 100}, spec=SPEC)
    result = a.allocation_comparisons(rows, fits, gates, spec=SPEC)
    assert all(row['blind_regret']['difference'] < 0 for row in result['paired'])
    assert not result['advance']['passed']
    assert result['advance']['name'] == 'UPDATE_MATCHED_ADVANCE'
    assert not result['advance']['conditions']['OBSERVED_FILTERING_EXTRAPOLATION']


def scalar_round(raw):
    """Direct positive-matrix arithmetic, independent of the audit's log-domain path."""
    outputs, corrections, before = [], [], []
    for action in raw:
        x = [[math.exp(float(value)) for value in row] for row in action]
        for _ in range(4):
            x = [[value / math.fsum(row) for value in row] for row in x]
            columns = [math.fsum(row[j] for row in x) for j in range(8)]
            x = [[value / columns[j] for j, value in enumerate(row)] for row in x]
        before.append(x)
        tau = 1 - 1e-8
        y = [[value * min(1., tau / math.fsum(row)) for value in row] for row in x]
        columns = [math.fsum(row[j] for row in y) for j in range(8)]
        z = [[value * min(1., tau / columns[j]) for j, value in enumerate(row)] for row in y]
        r = [1 - math.fsum(row) for row in z]
        c = [1 - math.fsum(row[j] for row in z) for j in range(8)]
        correction = [[r[i] * c[j] / math.fsum(r) for j in range(8)] for i in range(8)]
        corrections.append(correction)
        outputs.append([[z[i][j] + correction[i][j] for j in range(8)] for i in range(8)])
    return np.asarray(outputs), np.asarray(corrections), np.asarray(before)


def test_four_sweep_asymmetric_rounding_matches_independent_scalar_arithmetic():
    raw = np.asarray([[[.2 * ((i * 7 + j * 3 + arm) % 13) - 1.2
                       for j in range(8)] for i in range(8)] for arm in range(4)], np.float64)
    original = raw.tobytes()
    expected, correction, pre = scalar_round(raw)
    actual, diagnostic = a.rounded_transition_from_logits(raw, np)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(diagnostic['correction_mass'], correction.sum((1, 2)), rtol=0, atol=4e-15)
    assert diagnostic['correction_maximum_entry'] == pytest.approx(float(correction.max()), abs=2e-15)
    assert diagnostic['pre_to_final_maximum_change'] == pytest.approx(float(np.abs(expected - pre).max()), abs=2e-15)
    assert diagnostic['sweeps'] == 4 and diagnostic['slack'] == 1e-8 and diagnostic['admitted']
    assert raw.tobytes() == original


def test_already_doubly_stochastic_map_has_prescribed_uniform_mixing():
    matrix = .5 * np.eye(8) + np.full((8, 8), 1 / 16)
    raw = np.broadcast_to(np.log(matrix), (4, 8, 8)).copy()
    actual, diagnostic = a.rounded_transition_from_logits(raw, np)
    expected = (1 - 1e-8) * matrix + np.full((8, 8), 1e-8 / 8)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=0, atol=2e-15)
    np.testing.assert_allclose(diagnostic['correction_mass'], np.full(4, 8e-8), rtol=0, atol=4e-15)
    assert diagnostic['correction_maximum_entry'] == pytest.approx(1e-8 / 8, abs=1e-15)
    assert diagnostic['pre_to_final_maximum_change'] == pytest.approx(7e-8 / 16, abs=2e-15)
    assert not np.array_equal(actual[0], matrix)


def test_diagnostic_roster_is_all_27_actual_maps_without_rounding_free_controls(monkeypatch):
    _, original, _ = boundary_fixture()
    arrays = {(label, arm, seed): {name: value.copy() for name, value in record.items()}
              for (label, arm, _), record in original.items() for seed in SPEC.seeds}
    for (label, arm, _), record in arrays.items():
        if label != 'initial':
            record['transition_logits'][:, 0, :] = .7 if arm == 'rounded' else 1.2
    before = {key: raw_state_hash(record) for key, record in arrays.items()}
    calls, checks = [], []
    original_round = a.rounded_transition_from_logits

    def spy(raw, numpy):
        calls.append(raw.tobytes())
        return original_round(raw, numpy)

    monkeypatch.setattr(a, 'rounded_transition_from_logits', spy)
    rows = a.boundary_transition_diagnostics(arrays, np, check=lambda: checks.append(1))
    assert len(rows) == len(checks) == 27 and len(calls) == 9
    assert {(r['boundary'], r['arm'], r['seed']) for r in rows} == set(arrays)
    assert before == {key: raw_state_hash(record) for key, record in arrays.items()}
    for row in rows:
        assert row['column_residual_max'] <= 1e-12
        if row['arm'] == 'rounded':
            assert row['rounding']['admitted'] and row['row_residual_max'] <= 1e-12
            assert row['correction_status'] == 'reconstructed'
        else:
            assert row['rounding'] is None and row['correction_status'] == 'not_executed'
            assert row['transition_map'] == 'column_softmax'
            if row['boundary'] != 'initial':
                assert row['row_residual_max'] > 1


@pytest.mark.parametrize('fault', ['missing', 'extra', 'nonfinite', 'dtype'])
def test_boundary_transition_diagnostics_reject_incomplete_or_invalid_states(fault):
    _, arrays, _ = boundary_fixture()
    key = 'final', 'rounded', 942101
    if fault == 'missing':
        del arrays[key]
    elif fault == 'extra':
        arrays['unrecorded', 'rounded', 942101] = copy.deepcopy(arrays[key])
    elif fault == 'nonfinite':
        arrays[key]['transition_logits'][0, 0, 0] = np.inf
    else:
        arrays[key]['transition_logits'] = arrays[key]['transition_logits'].astype(np.float32)
    with pytest.raises(ValueError):
        a.boundary_transition_diagnostics(arrays, np)


def test_complete_named_rounding_work_is_charged_for_each_field():
    expected = {
        'rounded_transition_calls': 1, 'row_logsumexp_calls': 4, 'column_logsumexp_calls': 4,
        'normalization_vectors': 256, 'normalization_entries': 2048, 'completed_sweeps': 4,
        'exponential_calls': 1, 'exponential_entries': 256, 'matrix_sum_calls': 7, 'vector_sum_calls': 2,
        'contraction_maximum_calls': 2, 'contraction_division_entries': 64, 'contraction_product_entries': 512,
        'deficit_subtraction_entries': 64, 'deficit_normalization_entries': 32,
        'rank_one_product_entries': 256, 'rank_one_addition_entries': 256, 'correction_mass_sum_calls': 1,
        'logarithm_calls': 1, 'logarithm_entries': 256, 'check_calls': 11,
    }
    assert set(expected) == set(a.KERNEL_WORK)
    prefix = a.forward_work('rounded', dict.fromkeys(a.WORK_KEYS, 0), prefix=True, prior=True)
    endpoint = a.forward_work('rounded', dict.fromkeys(a.WORK_KEYS, 0), prefix=False)
    matched = a.construction_work('matched_free')
    for key, value in expected.items():
        assert prefix[key] == matched[key] == value
        assert endpoint[key] == 2 * value


def test_runtime_ratio_is_descriptive_not_an_advance_condition():
    rows, baseline, fits = metric_fixture()
    for fit in fits:
        fit['seconds'] = 110. if fit['arm'] == 'rounded' else 20.
    original = a.gates(rows, baseline, {'train': 400, 'base': 100}, spec=SPEC)
    result = a.allocation_comparisons(rows, fits, original, spec=SPEC)
    assert result['advance']['passed']
    assert len(result['advance']['conditions']) == 19
    assert all(row['ratio'] == 5.5 for row in result['mean_fit_time'])
    assert not any('time' in key for key in result['advance']['conditions'])


def paired_fixture():
    result = {}
    for arm in a.ARMS:
        for seed in SPEC.seeds:
            rows = []
            for cursor in range(SPEC.joint_updates):
                rows.append({'kind': 'joint', 'cursor_before': cursor,
                    'result': {'diagnostics': {'kind': 'joint', 'cursor': cursor, 'epoch': 0,
                    'batch_offset': cursor, 'indices': [cursor, cursor + 10],
                    'eligible': 2, 'valid_events': 18, 'batch_size': 2,
                    'order_sha256': digest(('order', seed))}}})
            result[arm, seed] = {'trace': rows}
    return result


def test_same_counts_do_not_replace_complete_paired_schedule_identity():
    result = a.paired_batch_checks(paired_fixture(), spec=SPEC)
    assert len(result) == 3
    assert all(row['identical_batch_indices_orders_and_support'] for row in result)


@pytest.mark.parametrize('fault', ['indices', 'order', 'support', 'missing_cursor', 'duplicate_cursor', 'missing_arm'])
def test_paired_exposure_corruption_is_rejected(fault):
    records = paired_fixture()
    key = 'rounded', SPEC.seeds[1]
    rows = records[key]['trace']
    first = rows[0]['result']['diagnostics']
    if fault == 'indices':
        first['indices'].reverse()
    elif fault == 'order':
        first['order_sha256'] = digest('changed order')
    elif fault == 'support':
        first['eligible'] = 1
    elif fault == 'missing_cursor':
        rows.pop()
    elif fault == 'duplicate_cursor':
        rows[1]['cursor_before'] = 0
    else:
        del records[key]
    with pytest.raises(ValueError):
        a.paired_batch_checks(records, spec=SPEC)


def test_whitelisted_cost_probe_does_not_relax_scientific_support_or_counts():
    probe = a.PROFILES['engineering-942201']
    assert probe.config == {'seed_namespace': 942201, 'fit_seeds': [942301],
        'train_attempts': 512, 'dev_attempts': 8, 'batch_size': 64, 'learning_rate': .003,
        'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
        'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.}
    rows, base, _ = metric_fixture()
    rows = [{**row, 'seed': 942301} for row in rows if row['seed'] == SPEC.seeds[0]]
    result = a.gates(rows, base, {'train': 480, 'base': 8}, spec=probe)
    assert not any(record['passed'] for criteria in result.values() for record in criteria.values())
