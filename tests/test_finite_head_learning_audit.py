"""Fabricated independent checks for the cost-head initialization audit.

No model or generator is called. The unchanged inherited numerical audit is
source-compared and the new disjoint operation ownership is checked from a
hand-authored population containing found, empty-endpoint and partial batches.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_head_learning as a
import audit_finite_reuse_learning as old

SPEC = a.StudyProfile('fabricated', 946001, (946101,), 5, 5, 2, 1, 3, 30.)


def test_only_declared_audit_function_bodies_change():
    root = Path(__file__).resolve().parents[1]
    def functions(name):
        tree = ast.parse((root / 'scripts' / name).read_text())
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, ast.FunctionDef)}
    original, current = functions('audit_finite_reuse_learning.py'), functions('audit_finite_head_learning.py')
    assert set(original) <= set(current)
    changed = {name for name in original if original[name] != current[name]}
    assert changed <= {'construction_work', 'forward_work', 'boundary_hashes', 'allocation_comparisons',
        'initial_transition_checks', 'boundary_transition_diagnostics', 'verify_boundaries',
        'validate_predictions', 'validate_metadata', 'validate_execution', 'audit'}
    for name in ('gates', 'reconstruct', 'reconstruct_prefix', 'rows_for',
                 'validate_allocation', 'optimizer_state', 'paired_batch_checks'):
        assert original[name] == current[name]


def test_fresh_immutable_profiles_keep_original_scientific_recipe():
    assert a.SCIENCE.namespace == 436260924
    assert a.SCIENCE.seeds == (436261001, 436261002, 436261003, 436261004, 436261005)
    assert (a.SCIENCE.prefix_updates, a.SCIENCE.joint_updates, a.SCIENCE.fit_cap_seconds) == (1024, 3072, 120.)
    assert set(a.PROFILES) == {'science', 'engineering-946001', 'engineering-946201'}
    smoke, probe = a.PROFILES['engineering-946001'], a.PROFILES['engineering-946201']
    assert (smoke.namespace, smoke.seeds, smoke.train_attempts, smoke.batch_size) == (946001, (946101,), 8, 3)
    assert (probe.namespace, probe.seeds, probe.train_attempts, probe.dev_attempts, probe.batch_size,
            probe.prefix_updates, probe.joint_updates) == (946201, (946301,), 512, 8, 64, 32, 64)
    with pytest.raises(TypeError):
        a.PROFILES['arbitrary'] = SPEC
    assert old.NAMESPACE == 434260924 and len(old.ROUTES) == 12
    assert len(a.ROUTES) == 12 and len(a.WORK_KEYS) == 71


def metric_fixture():
    spec = a.StudyProfile('fabricated-gates', 946001, (946101, 946102, 946103, 946104, 946105), 512, 512, 64, 1, 3, 30.)
    rows = [{'arm': arm, 'seed': seed, 'horizon': h, 'blind_regret': .3 if arm == 'rounded_random' else .4,
             'blind_cost_mse': .3, 'observed_cost_mse': .2, 'observed_kl': .05,
             'blind_survival_mae': .01, 'observed_survival_mae': .01, 'shuffled_blind_regret': .5}
            for arm in a.ARMS for seed in spec.seeds for h in a.HORIZONS]
    baseline = [{'horizon': h, 'blind_regret': 1., 'blind_cost_mse': 1.} for h in a.HORIZONS]
    fits = [{'arm': arm, 'seed': seed, 'seconds': 1000. if arm == 'rounded_random' else 1.}
            for arm in a.ARMS for seed in spec.seeds]
    return spec, rows, baseline, fits


def test_fifteen_head_conditions_have_no_anchor_or_runtime_rescue():
    spec, rows, baseline, fits = metric_fixture()
    gates = a.gates(rows, baseline, {'train': 300, 'base': 80}, spec=spec)
    report = a.allocation_comparisons(rows, fits, gates, spec=spec)
    assert report['advance']['passed'] and len(report['advance']['conditions']) == 15
    assert not any('time' in key for key in report['advance']['conditions'])
    assert report['mean_fit_time'][0]['ratio'] == 1000.
    rows[0]['observed_kl'] = 1.
    # Mutate the candidate's short-horizon KL, not a control's criterion.
    for row in rows:
        if row['arm'] == 'rounded_random' and row['seed'] == spec.seeds[0] and row['horizon'] == 1:
            row['observed_kl'] = 1.
    failed = a.gates(rows, baseline, {'train': 300, 'base': 80}, spec=spec)
    assert not a.allocation_comparisons(rows, fits, failed, spec=spec)['advance']['passed']


def test_zero_reference_and_failed_support_do_not_become_relative_gain():
    spec, rows, baseline, fits = metric_fixture()
    for row in rows:
        row['blind_regret'] = 0.
    gates = a.gates(rows, baseline, {'train': 300, 'base': 80}, spec=spec)
    assert not a.allocation_comparisons(rows, fits, gates, spec=spec)['advance']['passed']
    tiny = a.gates(rows, baseline, {'train': 5, 'base': 5}, spec=spec)
    assert not a.allocation_comparisons(rows, fits, tiny, spec=spec)['advance']['passed']


def prefix_fixture():
    p = np.zeros((5, 9, 31), np.float32)
    lengths = np.array([9, 2, 4, 9, 9], np.int64)
    ends = (None, 1, 3, 8, None)
    for i, end in enumerate(ends):
        p[i, 0, 4] = p[i, 0, 9] = 1
        for step in range(1, lengths[i]):
            p[i, step, step % 4] = 1
            p[i, step, 8 if step == end else 4] = 1
    return {'prefix': p, 'lengths': lengths, 'event_mask': np.arange(9)[None] < lengths[:, None],
            'endpoint_rows': np.array([0, -1, -1, -1, 1], np.int64)}


def blank():
    return dict.fromkeys(a.WORK_KEYS, 0)


def prefix_geometry(prefix, indices):
    result = blank()
    result['reset_emission_rows'] = len(indices)
    result['prefix_probability_rows'] = result['prefix_nll_rows'] = sum(int(prefix['lengths'][i]) for i in indices)
    for step in range(1, 9):
        active = sum(step < prefix['lengths'][i] and prefix['prefix'][i, step, 8] == 0 for i in indices)
        result['prefix_filter_calls'] += int(active > 0)
        result['prefix_filter_rows'] += int(active)
    return result


def fields(arm, factors, *, prior=False):
    result = blank()
    result.update(factorization_calls=factors, probability_field_calls=factors, adapter_check_calls=factors,
                  emission_softmax_calls=factors, emission_probability_columns=8 * factors,
                  hazard_sigmoid_calls=factors, hazard_probability_entries=32 * factors,
                  factor_product_entries=1536 * factors, factor_found_reduction_columns=32 * factors)
    if arm != 'matched_free_random':
        # The primitive's independently qualified fixed operation roster is reused as a constant.
        result.update({key: value * factors for key, value in a.KERNEL_WORK.items()})
    else:
        result.update(transition_softmax_calls=factors, transition_probability_columns=32 * factors)
    if prior:
        result.update(prior_emission_log_softmax_calls=1, prior_hazard_logsigmoid_calls=2,
                      prior_normalization_vectors=8, prior_normalization_entries=32, prior_logsigmoid_entries=64)
        if arm == 'matched_free_random':
            result.update(prior_transition_log_softmax_calls=1, prior_normalization_vectors=40,
                          prior_normalization_entries=288)
    return result


def continuation(observations, observed):
    result = blank()
    n, horizon = observations.shape
    result.update(blind_transition_calls=horizon, blind_transition_rows=n * horizon,
                  cost_readout_calls=horizon, cost_readout_rows=n * horizon,
                  cost_head_softmax_calls=horizon, cost_head_probability_rows=8 * horizon)
    if observed:
        for step in range(horizon):
            ordinary = sum(int(row[step]) < 4 for row in observations)
            result['observed_branch_calls'] += int(ordinary > 0)
            result['observed_branch_rows'] += ordinary
            result['observed_conditioning_rows'] += ordinary
            result['absorbed_rows'] += sum(step > 0 and int(row[step - 1]) == 4 for row in observations)
            result['event_probability_rows'] += n
    return result


def add(expected, route, *blocks):
    if not expected[route]:
        expected[route] = blank()
    for block in blocks:
        for name, value in block.items():
            expected[route][name] += value


def execution_fixture():
    prefix = prefix_fixture()
    datasets = {'train': {'case_ids': np.array(['a', 'b']), 'observations': np.array([[0, 4], [2, 0]], np.int64)},
                'base': {'case_ids': np.array(['c', 'd']), 'observations': np.array([[0, 1, 4, 4, 4, 4, 4, 4], [4] * 8], np.int64)}}
    prefixes = {'train': prefix, 'base': copy.deepcopy(prefix)}
    work = {arm: {route: {} for route in a.ROUTES} for arm in a.ARMS}
    allocations, orders = {}, []
    events = int(prefix['event_mask'].sum())
    full_order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([946101, 0, 818]))).permutation(5)
    total = dict.fromkeys(a.UPDATE_WORK, 0)
    for arm in a.ARMS:
        trace = []
        orders.append({'arm': arm, 'seed': 946101, 'epoch': 0, 'indices': full_order.tolist(), 'batch_size': 2})
        w = dict.fromkeys(a.UPDATE_WORK, 0)
        w.update(prefix_updates=1, prefix_event_exposures=events, prefix_full_rollouts=1, backward_passes=1, adam_steps=1)
        trace.append({'kind': 'prefix', 'accepted': True, 'result': {'diagnostics': {'kind': 'prefix',
            'valid_events': events, 'attempts': 5, 'pseudocount': .001, 'cost_head_updated': False}, 'work': w}})
        add(work[arm], 'training_prefix', fields(arm, 1, prior=True), prefix_geometry(prefix, range(5)))
        for cursor in range(3):
            indices = full_order[cursor * 2:(cursor + 1) * 2]
            selected = prefix['endpoint_rows'][indices]
            selected = selected[selected >= 0]
            obs = datasets['train']['observations'][selected]
            batch_events = int(prefix['event_mask'][indices].sum())
            d = {'kind': 'joint', 'cursor': cursor, 'epoch': 0, 'batch_offset': cursor, 'indices': indices.tolist(),
                 'eligible': len(selected), 'valid_events': batch_events, 'batch_size': len(indices),
                 'order_sha256': hashlib.sha256(full_order.astype('<i8').tobytes()).hexdigest()}
            w = dict.fromkeys(a.UPDATE_WORK, 0)
            w.update(joint_updates=1, joint_attempt_exposures=len(indices), joint_case_exposures=len(selected),
                     joint_event_exposures=batch_events, joint_prefix_rollouts=1, zero_endpoint_batches=int(not len(selected)),
                     joint_blind_rollouts=int(bool(len(selected))), joint_observed_rollouts=int(bool(len(selected))),
                     backward_passes=1, adam_steps=1)
            trace.append({'kind': 'joint', 'cursor_before': cursor, 'accepted': True, 'result': {'diagnostics': d, 'work': w}})
            shared = fields(arm, 1)
            shared['operator_marginal_sum_calls'] = int(bool(len(selected)))
            add(work[arm], 'joint_reuse_shared', shared)
            add(work[arm], 'joint_reuse_prefix_nll', prefix_geometry(prefix, indices))
            endpoint = blank()
            if len(selected):
                endpoint.update(reset_emission_rows=len(selected), prefix_filter_calls=8, prefix_filter_rows=8 * len(selected))
            add(work[arm], 'joint_reuse_endpoint_prefix', endpoint)
            for route, observed in (('joint_reuse_blind', False), ('joint_reuse_observed', True)):
                add(work[arm], route, continuation(obs, observed) if len(selected) else blank())
        allocations[arm, 946101] = {'trace': trace, 'attempted_updates': 4}
        for row in trace:
            for key, value in row['result']['work'].items():
                total[key] += value
        obs = datasets['base']['observations']
        for route, observed in (('evaluation_blind', False), ('evaluation_observed', True), ('evaluation_shuffled', False)):
            prefix_block = blank()
            prefix_block.update(reset_emission_rows=2, prefix_filter_calls=8, prefix_filter_rows=16, operator_marginal_sum_calls=1)
            add(work[arm], route, fields(arm, 2), prefix_block, continuation(obs, observed))
        for start in range(0, 5, 2):
            add(work[arm], 'evaluation_prefix', fields(arm, 1), prefix_geometry(prefix, range(start, min(start + 2, 5))))
    counts = {'train_generation_count': 1, 'dev_generation_count': 1, 'model_constructions': 3,
        'oracle_model_constructions': 1, 'fit_count': 3, 'checkpoint_writes': 9, 'optimizer_checkpoint_writes': 9,
        'training_blind_rollouts': total['joint_blind_rollouts'], 'training_observed_rollouts': total['joint_observed_rollouts'],
        'optimizer_attempts': 12, 'optimizer_steps': 12, 'training_case_exposures': 6, 'training_attempt_exposures': 15,
        'training_prefix_rollouts': 12, 'training_prefix_event_exposures': 6 * events,
        'zero_endpoint_batches': total['zero_endpoint_batches'], 'evaluation_blind_rollouts': 3,
        'evaluation_observed_rollouts': 3, 'evaluation_shuffled_rollouts': 3, 'evaluation_case_views': 6,
        'evaluation_prefix_rollouts': 9, 'evaluation_prefix_event_views': 3 * events,
        'oracle_blind_rollouts': 2, 'oracle_observed_rollouts': 2, 'array_decodes': 0, 'checkpoint_decodes': 0,
        'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0, 'readout_snapshot_evaluations': 0,
        'readout_snapshot_softmax_evaluations': 0, 'dynamics_snapshot_evaluations': 0,
        'epoch_order_generations': 3, 'accepted_optimizer_steps': 12, 'accepted_joint_steps': 9, 'accepted_prefix_steps': 3}
    return {'structural_work': work, 'counts': counts}, allocations, datasets, prefixes, orders


def test_saved_exposure_and_disjoint_work_reconstruct_empty_and_partial_batches():
    values = execution_fixture()
    report = a.validate_execution(*values, np, spec=SPEC)
    assert report['routes'] == 36 and report['epoch_permutations_reconstructed'] == 3
    assert report['attempted_update_work']['zero_endpoint_batches'] >= 3
    assert report['attempted_update_work']['joint_attempt_exposures'] == 15
    assert not report['model_updates_replayed']
    for arm in a.ARMS:
        assert values[0]['structural_work'][arm]['training_blind'] == {}
        assert values[0]['structural_work'][arm]['training_observed'] == {}
        assert values[0]['structural_work'][arm]['training_prefix']['probability_field_calls'] == 1
        assert values[0]['structural_work'][arm]['joint_reuse_shared']['probability_field_calls'] == 3


@pytest.mark.parametrize('fault', ['shared_twice', 'joint_in_prefix', 'legacy_blind', 'missing_route',
                                  'wrong_order', 'exposure', 'dropped_attempt', 'privilege'])
def test_new_work_ownership_and_exposure_corruptions_rejected(fault):
    values = execution_fixture()
    summary, allocations, _datasets, _prefixes, orders = values
    work = summary['structural_work']['rounded_random']
    if fault == 'shared_twice':
        work['joint_reuse_shared']['factorization_calls'] *= 2
    elif fault == 'joint_in_prefix':
        work['training_prefix']['prefix_nll_rows'] += 1
    elif fault == 'legacy_blind':
        work['training_blind'] = copy.deepcopy(work['joint_reuse_blind'])
    elif fault == 'missing_route':
        work.pop('joint_reuse_observed')
    elif fault == 'wrong_order':
        orders[0]['indices'][0], orders[0]['indices'][1] = orders[0]['indices'][1], orders[0]['indices'][0]
    elif fault == 'exposure':
        summary['counts']['training_attempt_exposures'] -= 1
    elif fault == 'dropped_attempt':
        allocations['rounded_random', 946101]['trace'].pop()
    else:
        work['joint_reuse_shared']['privileged_prefix_rows'] = 1
    with pytest.raises(ValueError):
        a.validate_execution(*values, np, spec=SPEC)


def test_wrong_producer_and_incomplete_manifest_fail_before_arrays(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: calls.append(args))
    (tmp_path / 'summary.json').write_text(json.dumps({'version': 'finite-update-learning-v1', 'implementation': 'separate'}))
    with pytest.raises(ValueError, match='reuse-only'):
        a.audit(tmp_path, profile='engineering-946001')
    assert calls == []


def test_integration_metadata_cannot_admit_separate_fits():
    source = Path(a.__file__).read_text()
    # Source contract complements root's actual closed-producer smoke audit.
    assert "row['implementation'] == 'reuse'" in source
    assert "row['integration_version'] == 'finite-head-training-v1'" in source
    assert "row['joint_structural_routes'] == list(REUSE_ROUTES)" in source
    assert math.isclose(a.PROFILES['science'].config['learning_rate'], .003)


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


def raw_state_hash(arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        metadata = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        result.update(len(metadata).to_bytes(8, 'little') + metadata + value.tobytes())
    return result.hexdigest()


def boundary_fixture():
    """Uniform matched transport, distinct template/random heads, no model calls."""
    seed = 946101
    head = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 436, 1]))).standard_normal((4, 8))
    template = np.log(np.array([[.925 if action == ((state ^ (state >> 1)) & 3) else .025
                               for state in range(8)] for action in range(4)], np.float64))
    allocations, arrays, optimizers = {}, {}, {}
    for arm in a.ARMS:
        allocation = {'stages': [{'accepted_updates': 1}, {'accepted_updates': 1}], 'checkpoints': [],
            'boundary': {'optimizer_after_sha256': a.optimizer_state(optimizer('joint', 0), 'joint', 0)['sha256']}}
        for label in ('initial', 'boundary', 'final'):
            values = {name: np.zeros(shape, np.float64) for name, shape in a.PARAMETER_SHAPES.items()}
            values['cost_logits'][:] = template if arm == 'rounded_anchor' else head
            if arm == 'matched_free_random':
                values['transition_logits'][:] = -math.log(8.)
            if label != 'initial':
                values['hazard_logits'][:] = 1. if label == 'boundary' else 2.
            if label == 'final':
                values['cost_logits'] += .1
            arrays[label, arm, seed] = values
            payload = optimizer('joint' if label == 'final' else 'prefix', 0 if label == 'initial' else 1)
            optimizers[label, arm, seed] = payload
            metadata = {'dynamics_state_sha256': raw_state_hash({name: values[name] for name in a.DYNAMIC_NAMES}),
                        'head_state_sha256': raw_state_hash({'cost_logits': values['cost_logits']}),
                        'optimizer_state_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
            allocation['checkpoints'].append({'label': label, 'model_sha256': raw_state_hash(values),
                'optimizer_sha256': metadata['optimizer_state_sha256'], 'metadata': metadata})
        allocations[arm, seed] = allocation
    return allocations, arrays, optimizers


def rehash_boundary(allocations, arrays, optimizers, label, arm):
    values = arrays[label, arm, 946101]
    point = next(p for p in allocations[arm, 946101]['checkpoints'] if p['label'] == label)
    point['model_sha256'] = raw_state_hash(values)
    point['metadata']['dynamics_state_sha256'] = raw_state_hash({name: values[name] for name in a.DYNAMIC_NAMES})
    point['metadata']['head_state_sha256'] = raw_state_hash({'cost_logits': values['cost_logits']})
    point['optimizer_sha256'] = point['metadata']['optimizer_state_sha256'] = hashlib.sha256(json.dumps(
        optimizers[label, arm, 946101], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_saved_boundary_controls_use_tensor_bytes_and_distinct_restricted_hashes():
    allocations, arrays, optimizers = boundary_fixture()
    before = {key: {name: value.tobytes() for name, value in record.items()} for key, record in arrays.items()}
    records = a.verify_boundaries(allocations, arrays, optimizers, np)
    assert len(records) == 3 and all(r['optimizer_and_model_boundaries_joined'] for r in records)
    assert all(r['initial_transition_check']['initializer_reconstructions'] == 1 for r in records)
    assert len({r['prefix_pair_check']['boundaries']['boundary']['dynamics_state_sha256'] for r in records}) == 1
    for key, state in arrays.items():
        hashes = a.boundary_hashes(state, np)
        assert hashes['dynamics_state'] == raw_state_hash({name: state[name] for name in a.DYNAMIC_NAMES})
        assert hashes['head_state'] == raw_state_hash({'cost_logits': state['cost_logits']})
        assert hashes['dynamics_state'] != hashes['dynamics']
        assert {name: value.tobytes() for name, value in state.items()} == before[key]
    pairs = a.declared_prefix_pairs(allocations, spec=SPEC)
    assert pairs[0]['all_heads_unchanged_after_prefix'] is True
    assert set(pairs[0]['head_initial_boundary_sha256']) == set(a.ARMS)


@pytest.mark.parametrize('fault', ['random_scale', 'different_random_stream', 'template', 'initial_emission',
    'initial_transition', 'prefix_dynamics', 'prefix_adam', 'prefix_head', 'restricted_hash', 'final_hash'])
def test_actual_head_or_prefix_pair_corruption_fails_even_with_recomputed_file_hashes(fault):
    allocations, arrays, optimizers = boundary_fixture()
    arm, label = 'rounded_random', 'initial'
    if fault == 'random_scale':
        for changed in ('rounded_random', 'matched_free_random'):
            arrays['initial', changed, 946101]['cost_logits'] *= .1
            rehash_boundary(allocations, arrays, optimizers, 'initial', changed)
    elif fault == 'different_random_stream':
        arrays[label, arm, 946101]['cost_logits'][0, 0] += .01
    elif fault == 'template':
        arm = 'rounded_anchor'
        arrays[label, arm, 946101]['cost_logits'][0, 0] += .01
    elif fault == 'initial_emission':
        arrays[label, arm, 946101]['emission_logits'][0, 0] += .01
    elif fault == 'initial_transition':
        arrays[label, arm, 946101]['transition_logits'][0, 0, 0] = -0.
    elif fault == 'prefix_dynamics':
        label = 'boundary'
        arrays[label, arm, 946101]['hazard_logits'][0, 0] += .01
    elif fault == 'prefix_adam':
        label = 'boundary'
        optimizers[label, arm, 946101]['state']['1']['exp_avg']['values'][0][0] = .01
    elif fault == 'prefix_head':
        label, arm = 'boundary', 'matched_free_random'
        arrays[label, arm, 946101]['cost_logits'][0, 0] += .01
    elif fault == 'restricted_hash':
        allocations[arm, 946101]['checkpoints'][0]['metadata']['dynamics_state_sha256'] = '0' * 64
    else:
        allocations[arm, 946101]['checkpoints'][2]['model_sha256'] = '0' * 64
    if fault not in ('restricted_hash', 'final_hash'):
        rehash_boundary(allocations, arrays, optimizers, label, arm)
    with pytest.raises(ValueError):
        a.verify_boundaries(allocations, arrays, optimizers, np)


def head_metadata_fixture(arm):
    random, rounded = arm != 'rounded_anchor', arm != 'matched_free_random'
    work = dict(zip(a.INITIALIZER_WORK_KEYS, (1, 1, 32, 1, 32, 256) if random else (0,) * 6, strict=True))
    pm = {'version': 'finite-head-initialization-v1', 'transport_model_version': 'finite-rounded-model-v1',
        'arm': arm, 'transport_arm': 'rounded' if rounded else 'matched_free', 'seed': 946101,
        'parameters': {name: {'shape': list(shape), 'count': math.prod(shape), 'dtype': 'torch.float64',
            'bytes': 8 * math.prod(shape), 'requires_grad': True} for name, shape in a.PARAMETER_SHAPES.items()},
        'count': 352, 'trainable_count': 352, 'parameter_bytes': 2816, 'buffer_bytes': 0, 'buffers': {},
        'mass_width': 8, 'privileged_prefix': False, 'known_operators': False, 'fixed_cost_readout': False,
        'privileged_readout_initialization': not random, 'readout_delta': None if random else .1,
        'head_initialization': {'policy': 'task_independent_standard_normal' if random else 'inherited_cost_template',
            'generator': 'numpy.random.Generator(PCG64)' if random else None,
            'seed_sequence_entropy': [946101, 436, 1] if random else None,
            'shape': [4, 8], 'dtype': 'float64', 'inherited_template_is_constructed': True},
        'head_initialization_work': work.copy(), 'transition_stored_parameters': 256,
        'transition_ideal_constraint_degrees': 196 if rounded else 224,
        'rounded_sweeps': 4 if rounded else None, 'rounded_slack': 1e-8 if rounded else None,
        'rounded_tolerance': 1e-12 if rounded else None, 'matched_initialization': not rounded,
        'construction_work': a.construction_work(arm)}
    return {'arm': arm, 'seed': 946101, 'model_metadata': pm, 'head_initialization_work': work}


@pytest.mark.parametrize('arm', a.ARMS)
def test_effective_model_metadata_does_not_inherit_wrong_head_privilege(arm):
    a.head_metadata(head_metadata_fixture(arm))


@pytest.mark.parametrize('fault', ['privileged', 'entropy', 'scale', 'work', 'count', 'transport'])
def test_head_policy_metadata_corruptions_rejected(fault):
    row = head_metadata_fixture('rounded_random')
    pm = row['model_metadata']
    if fault == 'privileged':
        pm['privileged_readout_initialization'] = True
    elif fault == 'entropy':
        pm['head_initialization']['seed_sequence_entropy'] = [946101, 1, 436]
    elif fault == 'scale':
        pm['readout_delta'] = .1
    elif fault == 'work':
        row['head_initialization_work']['standard_normal_calls'] = 0
    elif fault == 'count':
        pm['count'] = 320
    else:
        pm['transport_arm'] = 'matched_free'
    with pytest.raises(ValueError):
        a.head_metadata(row)


@pytest.mark.parametrize('fault', ['bad_fifth_seed', 'missing_fifth_seed', 'anchor_only_success', 'candidate_absolute'])
def test_five_seed_primary_rule_has_no_privileged_anchor_or_aggregate_rescue(fault):
    spec, rows, baselines, fits = metric_fixture()
    if fault == 'missing_fifth_seed':
        rows = [r for r in rows if not (r['seed'] == spec.seeds[-1] and r['arm'] == 'rounded_random')]
        with pytest.raises(ValueError):
            a.gates(rows, baselines, {'train': 300, 'base': 80}, spec=spec)
        return
    for row in rows:
        if row['arm'] == 'rounded_anchor':
            row.update(blind_regret=0., blind_cost_mse=0., observed_kl=0.)
        elif row['arm'] == 'rounded_random':
            if (fault == 'bad_fifth_seed' and row['seed'] == spec.seeds[-1] and row['horizon'] == 8
                    or fault == 'anchor_only_success'):
                row['blind_regret'] = .41
            elif fault == 'candidate_absolute' and row['horizon'] == 1:
                row['observed_kl'] = .11
    gates = a.gates(rows, baselines, {'train': 300, 'base': 80}, spec=spec)
    report = a.allocation_comparisons(rows, fits, gates, spec=spec)
    assert not report['advance']['passed']
    assert len(report['advance']['conditions']) == 15
    assert len(report['paired']) == 10 and len(report['anchor_comparisons']) == 20
    assert report['anchor_is_descriptive_only'] is True
    assert not any('anchor' in name for name in report['advance']['conditions'])
