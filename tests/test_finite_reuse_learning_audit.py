"""Fabricated independent checks for reuse-only learning audit deltas.

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
import audit_finite_reuse_learning as a
import audit_finite_update_learning as old

SPEC = a.StudyProfile('fabricated', 944001, (944101,), 5, 5, 2, 1, 3, 30.)


def test_only_declared_audit_function_bodies_change():
    root = Path(__file__).resolve().parents[1]
    def functions(name):
        tree = ast.parse((root / 'scripts' / name).read_text())
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, ast.FunctionDef)}
    original, current = functions('audit_finite_update_learning.py'), functions('audit_finite_reuse_learning.py')
    assert original.keys() == current.keys()
    assert {name for name in original if original[name] != current[name]} == {
        'validate_metadata', 'validate_execution', 'audit'}
    # All metric/gate, independent world, decoded Adam and controller checks remain frozen.
    for name in ('gates', 'allocation_comparisons', 'reconstruct', 'reconstruct_prefix',
                 'rows_for', 'verify_boundaries', 'validate_allocation', 'optimizer_state'):
        assert original[name] == current[name]


def test_fresh_immutable_profiles_keep_original_scientific_recipe():
    assert a.SCIENCE.namespace == 434260924
    assert a.SCIENCE.seeds == (434261001, 434261002, 434261003)
    assert (a.SCIENCE.prefix_updates, a.SCIENCE.joint_updates, a.SCIENCE.fit_cap_seconds) == (1024, 3072, 120.)
    assert set(a.PROFILES) == {'science', 'engineering-944001', 'engineering-944201'}
    smoke, probe = a.PROFILES['engineering-944001'], a.PROFILES['engineering-944201']
    assert (smoke.namespace, smoke.seeds, smoke.train_attempts, smoke.batch_size) == (944001, (944101,), 8, 3)
    assert (probe.namespace, probe.seeds, probe.train_attempts, probe.dev_attempts, probe.batch_size,
            probe.prefix_updates, probe.joint_updates) == (944201, (944301,), 512, 8, 64, 32, 64)
    with pytest.raises(TypeError):
        a.PROFILES['arbitrary'] = SPEC
    assert old.NAMESPACE == 433260924 and len(old.ROUTES) == 7
    assert len(a.ROUTES) == 12 and len(a.WORK_KEYS) == 71


def metric_fixture():
    spec = a.StudyProfile('fabricated-gates', 944001, (944101, 944102, 944103), 512, 128, 64, 1, 3, 30.)
    rows = [{'arm': arm, 'seed': seed, 'horizon': h, 'blind_regret': .3 if arm == 'rounded' else .4,
             'blind_cost_mse': .3, 'observed_cost_mse': .2, 'observed_kl': .05,
             'blind_survival_mae': .01, 'observed_survival_mae': .01, 'shuffled_blind_regret': .5}
            for arm in a.ARMS for seed in spec.seeds for h in a.HORIZONS]
    baseline = [{'horizon': h, 'blind_regret': 1., 'blind_cost_mse': 1.} for h in a.HORIZONS]
    fits = [{'arm': arm, 'seed': seed, 'seconds': 1000. if arm == 'rounded' else 1.}
            for arm in a.ARMS for seed in spec.seeds]
    return spec, rows, baseline, fits


def test_original_nineteen_effect_conditions_have_no_runtime_rescue_or_runtime_gate():
    spec, rows, baseline, fits = metric_fixture()
    gates = a.gates(rows, baseline, {'train': 300, 'base': 80}, spec=spec)
    report = a.allocation_comparisons(rows, fits, gates, spec=spec)
    assert report['advance']['passed'] and len(report['advance']['conditions']) == 19
    assert not any('time' in key for key in report['advance']['conditions'])
    assert report['mean_fit_time'][0]['ratio'] == 1000.
    rows[0]['observed_kl'] = 1.
    # Mutate the candidate's short-horizon KL, not a control's criterion.
    for row in rows:
        if row['arm'] == 'rounded' and row['seed'] == spec.seeds[0] and row['horizon'] == 1:
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
    if arm == 'rounded':
        # The primitive's independently qualified fixed operation roster is reused as a constant.
        result.update({key: value * factors for key, value in a.KERNEL_WORK.items()})
    else:
        result.update(transition_softmax_calls=factors, transition_probability_columns=32 * factors)
    if prior:
        result.update(prior_emission_log_softmax_calls=1, prior_hazard_logsigmoid_calls=2,
                      prior_normalization_vectors=8, prior_normalization_entries=32, prior_logsigmoid_entries=64)
        if arm != 'rounded':
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
    full_order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([944101, 0, 818]))).permutation(5)
    total = dict.fromkeys(a.UPDATE_WORK, 0)
    for arm in a.ARMS:
        trace = []
        orders.append({'arm': arm, 'seed': 944101, 'epoch': 0, 'indices': full_order.tolist(), 'batch_size': 2})
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
        allocations[arm, 944101] = {'trace': trace, 'attempted_updates': 4}
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
    work = summary['structural_work']['rounded']
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
        allocations['rounded', 944101]['trace'].pop()
    else:
        work['joint_reuse_shared']['privileged_prefix_rows'] = 1
    with pytest.raises(ValueError):
        a.validate_execution(*values, np, spec=SPEC)


def test_wrong_producer_and_incomplete_manifest_fail_before_arrays(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: calls.append(args))
    (tmp_path / 'summary.json').write_text(json.dumps({'version': 'finite-update-learning-v1', 'implementation': 'separate'}))
    with pytest.raises(ValueError, match='reuse-only'):
        a.audit(tmp_path, profile='engineering-944001')
    assert calls == []


def test_integration_metadata_cannot_admit_separate_fits():
    source = Path(a.__file__).read_text()
    # Source contract complements root's actual closed-producer smoke audit.
    assert "row['implementation'] == 'reuse'" in source
    assert "row['integration_version'] == 'finite-joint-reuse-training-v1'" in source
    assert "row['joint_structural_routes'] == list(REUSE_ROUTES)" in source
    assert math.isclose(a.PROFILES['science'].config['learning_rate'], .003)
