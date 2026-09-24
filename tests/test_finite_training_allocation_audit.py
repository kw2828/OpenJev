"""Independent hand-constructed allocation traces and boundary corruption checks."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_training_allocation as a

SPEC = a.StudyProfile('fabricated', 939001, (939101, 939102, 939103), 512, 128, 64, 10., 40.)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def allocation(arm='joint_continuous', stop='late'):
    """Two updates per stage; no controller or numerical model constructs this oracle."""
    trace, stages, cursor, loops = [], [], 0, 0
    model, opt = digest('initial model'), digest('initial optimizer')
    initial_model, initial_opt = model, opt
    total = dict.fromkeys(a.CONTROLLER_WORK, 0)
    boundary = None
    for stage_index, deadline in ((1, 10.), (2, 40.)):
        kind = 'prefix' if arm == 'prefix_then_joint' and stage_index == 1 else 'joint'
        start = .03 if stage_index == 1 else stages[0]['stopped_elapsed'] + .3
        if stage_index == 2:
            previous = opt
            if arm != 'joint_continuous':
                opt = digest('fresh joint optimizer')
            boundary = {'joint_cursor': cursor, 'model_sha256': model,
                'stage1_optimizer_kind': 'prefix' if arm == 'prefix_then_joint' else 'joint',
                'optimizer_before_sha256': previous, 'optimizer_after_sha256': opt,
                'stage2_optimizer_kind': 'joint', 'optimizer_reset': arm != 'joint_continuous'}
        model_start, optimizer_start, cursor_start = model, opt, cursor
        trace_start, accepted = len(trace), 0
        opt_steps = cursor if stage_index == 2 and arm == 'joint_continuous' else 0
        endings = (start + .4, deadline + 1 if stop == 'late' else deadline if stop == 'equal' else deadline - .5)
        for offset, completed in enumerate(endings):
            began = start + .01 if offset == 0 else deadline - 1
            eligible = completed <= deadline
            attempted_model = digest(('model', stage_index, offset))
            attempted_opt = digest(('optimizer', stage_index, offset))
            tensor_count, param_bytes = (3, 2560) if kind == 'prefix' else (4, 2816)
            delta = dict.fromkeys(a.CONTROLLER_WORK, 0)
            delta.update(model_snapshot_calls=1, model_snapshot_tensors=4, model_snapshot_bytes=2816,
                optimizer_snapshot_calls=1, optimizer_snapshot_tensors=3 * tensor_count if opt_steps else 0,
                optimizer_snapshot_bytes=2 * param_bytes + 4 * tensor_count if opt_steps else 0,
                model_hash_calls=2 + int(not eligible), optimizer_hash_calls=2 + int(not eligible),
                check_calls=1, clock_calls=1)
            delta[kind + '_update_calls'] = 1
            if not eligible:
                delta.update(model_restore_calls=1, model_restore_tensors=4, model_restore_bytes=2816,
                    optimizer_restore_calls=1, optimizer_restore_tensors=delta['optimizer_snapshot_tensors'],
                    optimizer_restore_bytes=delta['optimizer_snapshot_bytes'])
            trace.append({'attempt': len(trace) + 1, 'stage': stage_index, 'kind': kind, 'deadline_seconds': deadline,
                'start_elapsed': began, 'completed_elapsed': completed, 'accepted': eligible, 'rolled_back': not eligible,
                'error': None, 'cursor_before': cursor, 'cursor_attempted': cursor + int(kind == 'joint'),
                'cursor_retained': cursor + int(eligible and kind == 'joint'),
                'model_before_sha256': model, 'optimizer_before_sha256': opt,
                'model_attempted_sha256': attempted_model, 'optimizer_attempted_sha256': attempted_opt,
                'model_retained_sha256': attempted_model if eligible else model,
                'optimizer_retained_sha256': attempted_opt if eligible else opt,
                'result': {'loss': .5 + offset, 'diagnostics': {}, 'work': {'fabricated_update_calls': 1}},
                'work_delta': delta})
            for key, value in delta.items():
                total[key] += value
            if eligible:
                cursor += int(kind == 'joint')
                accepted += 1
                opt_steps += 1
                model, opt = attempted_model, attempted_opt
        stopped = max(deadline, endings[-1]) + .1
        stages.append({'stage': stage_index, 'kind': kind, 'deadline_seconds': deadline,
            'start_elapsed': start, 'stopped_elapsed': stopped, 'status': 'PASS',
            'termination': 'late_update_rolled_back' if stop == 'late' else 'deadline_reached',
            'trace_start': trace_start, 'trace_stop': len(trace), 'attempted_updates': 2, 'accepted_updates': accepted,
            'joint_cursor_start': cursor_start, 'joint_cursor_end': cursor,
            'model_start_sha256': model_start, 'optimizer_start_sha256': optimizer_start,
            'model_end_sha256': model, 'optimizer_end_sha256': opt, 'overrun_seconds': stopped - deadline})
        loops += int(stop == 'next_loop')
    checkpoints = []
    for label, start, end, state, optimizer, position in (
            ('initial', .01, .02, initial_model, initial_opt, 0),
            ('boundary', stages[0]['stopped_elapsed'] + .1, stages[0]['stopped_elapsed'] + .2,
             stages[0]['model_end_sha256'], stages[0]['optimizer_end_sha256'], stages[0]['joint_cursor_end']),
            ('final', stages[1]['stopped_elapsed'] + .1, stages[1]['stopped_elapsed'] + .2, model, opt, cursor)):
        checkpoints.append({'label': label, 'start_elapsed': start, 'completed_elapsed': end,
            'seconds': end - start, 'joint_cursor': position, 'model_sha256': state, 'optimizer_sha256': optimizer,
            'metadata': {}})
    total.update(validation_calls=1, optimizer_constructions=1 if arm == 'joint_continuous' else 2, checkpoint_calls=3)
    total['model_hash_calls'] += 13
    total['optimizer_hash_calls'] += 13
    total['check_calls'] += 14 + loops
    total['clock_calls'] += 16 + loops
    final_work = dict.fromkeys(a.CONTROLLER_WORK, 0)
    final_work.update(model_hash_calls=1, optimizer_hash_calls=1, check_calls=1, clock_calls=1)
    timed = checkpoints[-1]['completed_elapsed'] + .1
    return {'version': 'finite-training-allocation-v1', 'arm': arm, 'status': 'PASS', 'termination': 'completed_stages',
        'stage1_seconds': 10., 'total_seconds': 40., 'max_attempts': 100000, 'metadata': {},
        'initial_model_sha256': initial_model, 'initial_optimizer_sha256': initial_opt,
        'final_model_sha256': model, 'final_optimizer_sha256': opt, 'joint_cursor': cursor,
        'attempted_updates': 4, 'accepted_updates': sum(r['accepted'] for r in trace),
        'accepted_joint_updates': cursor,
        'accepted_prefix_updates': sum(r['accepted'] and r['kind'] == 'prefix' for r in trace),
        'trace': trace, 'stages': stages, 'boundary': boundary, 'checkpoints': checkpoints,
        'work': total, 'update_work': {'fabricated_update_calls': 4}, 'timed_seconds': timed,
        'overrun_seconds': timed - 40., 'compute_matched': False, 'timing_scope': 'fabricated',
        'final_summary': {}, 'final_summary_seconds': .05, 'final_summary_work': final_work}


@pytest.mark.parametrize('arm', a.ARMS)
@pytest.mark.parametrize('stop', ['late', 'equal', 'next_loop'])
def test_hand_trace_checks_both_deadlines_and_adam_storage(arm, stop):
    result = a.validate_allocation(allocation(arm, stop), arm, spec=SPEC)
    assert result['attempted_updates'] == 4
    assert sum(row['late_updates'] for row in result['stages']) == (2 if stop == 'late' else 0)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(total_seconds=50.),
    lambda r: r.update(compute_matched=True),
    lambda r: r.update(accepted_updates=4),
    lambda r: r['trace'][1].update(accepted=True),
    lambda r: r['trace'][1].update(cursor_retained=2),
    lambda r: r['trace'][1].update(optimizer_retained_sha256=digest('late optimizer retained')),
    lambda r: r['trace'][1].update(model_retained_sha256=digest('late model retained')),
    lambda r: r['trace'][2].update(cursor_before=0),
    lambda r: r['trace'][2].update(deadline_seconds=50.),
    lambda r: r['trace'][2].update(start_elapsed=10.),
    lambda r: r['trace'][0]['work_delta'].update(model_snapshot_bytes=2560),
    lambda r: r['trace'][1]['work_delta'].update(optimizer_restore_bytes=0),
    lambda r: r['work'].update(check_calls=0),
    lambda r: r['update_work'].update(fabricated_update_calls=2),
    lambda r: r['stages'][0].update(accepted_updates=0),
    lambda r: r['stages'][0].update(termination='best_loss_selected'),
    lambda r: r['boundary'].update(optimizer_reset=True),
    lambda r: r['boundary'].update(optimizer_after_sha256=digest('fresh optimizer')),
    lambda r: r['checkpoints'][1].update(joint_cursor=2),
    lambda r: r['checkpoints'][1].update(completed_elapsed=100.),
    lambda r: r['final_summary_work'].update(model_restore_calls=1),
    lambda r: r.update(final_model_sha256=digest('other final state')),
])
def test_controller_corruption_rejected(mutation):
    record = allocation()
    mutation(record)
    with pytest.raises((ValueError, KeyError)):
        a.validate_allocation(record, 'joint_continuous', spec=SPEC)


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
    rows = [{'arm': arm, 'seed': seed, 'horizon': h, 'blind_regret': .3 if arm == 'prefix_then_joint' else .4,
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
    assert output['advance']['passed'] and len(output['advance']['conditions']) == 21
    assert len(output['paired']) == 12
    rows[-1]['blind_regret'] = .6
    gates = a.gates(rows, baseline, {'train': 400, 'base': 100}, spec=SPEC)
    assert not a.allocation_comparisons(rows, fits, gates, spec=SPEC)['advance']['passed']


@pytest.mark.parametrize('fault', ['one_worse_pair', 'insufficient_mean', 'time', 'zero_reference', 'support'])
def test_no_averages_or_zero_denominator_rescue(fault):
    rows, baseline, fits = metric_fixture()
    counts = {'train': 400, 'base': 100}
    if fault == 'one_worse_pair':
        next(r for r in rows if r['arm'] == 'prefix_then_joint' and r['horizon'] == 4)['blind_regret'] = .401
    elif fault == 'insufficient_mean':
        for row in rows:
            if row['arm'] == 'prefix_then_joint':
                row['blind_regret'] = .37
    elif fault == 'time':
        for fit in fits:
            if fit['arm'] == 'prefix_then_joint':
                fit['seconds'] = 43.1
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
    spec = a.PROFILES['engineering-939001']
    rows, baseline, _ = metric_fixture()
    rows = [row for row in rows if row['seed'] == spec.seeds[0]]
    gates = a.gates(rows, baseline, {'train': 8, 'base': 8}, spec=spec)
    assert not any(value['passed'] for criteria in gates.values() for value in criteria.values())


def test_incomplete_manifest_fails_before_any_array_decode(tmp_path, monkeypatch):
    rows, _, fits = metric_fixture()
    del rows
    summary = {'version': 'finite-training-allocation-v1', 'config': SPEC.config, 'fits': fits, 'files': {}}
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
    seed = 939101
    for arm in a.ARMS:
        allocation_record = {'accepted_joint_updates': 2 if arm == 'joint_continuous' else 1,
            'stages': [{'accepted_updates': 1}, {'accepted_updates': 1}], 'checkpoints': [],
            'boundary': {'optimizer_after_sha256': a.optimizer_state(optimizer('joint', 0), 'joint', 0)['sha256']}}
        for label in ('initial', 'boundary', 'final'):
            values = {name: np.zeros(shape, np.float64) for name, shape in a.PARAMETER_SHAPES.items()}
            if label != 'initial':
                values['hazard_logits'][:] = 1 if label == 'boundary' else 2
            if label == 'final' or (label == 'boundary' and arm != 'prefix_then_joint'):
                values['cost_logits'][:] = 3
            arrays[label, arm, seed] = values
            kind = 'prefix' if arm == 'prefix_then_joint' and label != 'final' else 'joint'
            steps = 0 if label == 'initial' else 2 if label == 'final' and arm == 'joint_continuous' else 1
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
    arm, seed = 'prefix_then_joint', 939101
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
