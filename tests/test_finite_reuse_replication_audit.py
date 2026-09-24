"""Fabricated audit checks for fresh BASE replication plus observation shift.

No model or world generator is called. Scalar gate fixtures retain all five
seeds, while small public histories test both independent observation laws.
Actual producer admission is also covered by the separately owned smoke test.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_reuse_learning as old
import audit_finite_reuse_replication as a

SPEC = a.StudyProfile('fabricated', 945001, (945101,), 5, 5, 2, 1, 3, 30.)


def test_controller_state_and_training_math_remain_frozen():
    root = Path(__file__).resolve().parents[1]
    def functions(name):
        tree = ast.parse((root / 'scripts' / name).read_text())
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, ast.FunctionDef)}
    original, current = functions('audit_finite_reuse_learning.py'), functions('audit_finite_reuse_replication.py')
    assert set(current) - set(original) == {'replication_comparisons', 'validate_reference_metadata'}
    for name in ('verify_boundaries', 'validate_allocation', 'optimizer_state', 'paired_batch_checks',
                 'boundary_hashes', 'rounded_transition_from_logits', 'boundary_transition_diagnostics',
                 'parameter_metadata', 'prefix_work', 'endpoint_work', 'forward_work', 'construction_work'):
        assert original[name] == current[name]
    assert a.reuse_work is old.reuse_work
    assert old.NAMESPACE == 434260924 and len(old.SEEDS) == 3


def test_explicit_five_seed_two_regime_profiles_are_immutable():
    assert a.SCIENCE.namespace == 435260924
    assert a.SCIENCE.seeds == tuple(range(435261001, 435261006))
    assert (a.SCIENCE.train_attempts, a.SCIENCE.dev_attempts) == (512, 512)
    assert (a.SCIENCE.prefix_updates, a.SCIENCE.joint_updates, a.SCIENCE.fit_cap_seconds) == (1024, 3072, 120.)
    assert set(a.PROFILES) == {'science', 'engineering-945001', 'engineering-945201'}
    smoke, probe = a.PROFILES['engineering-945001'], a.PROFILES['engineering-945201']
    assert (smoke.namespace, smoke.seeds, smoke.train_attempts, smoke.dev_attempts, smoke.batch_size) == (945001, (945101,), 8, 8, 3)
    assert (probe.namespace, probe.seeds, probe.train_attempts, probe.dev_attempts, probe.batch_size,
            probe.prefix_updates, probe.joint_updates) == (945201, (945301,), 512, 8, 64, 32, 64)
    assert dict(a.SPLIT_IDS) == {'train': 0, 'base': 1, 'shift': 2}
    assert dict(a.EPSILON) == {'train': .12, 'base': .12, 'shift': .30}
    with pytest.raises(TypeError):
        a.PROFILES['arbitrary'] = SPEC
    with pytest.raises(TypeError):
        a.EPSILON['shift'] = .12


def metric_fixture():
    spec = a.StudyProfile('fabricated-gates', 945001, tuple(range(945101, 945106)), 512, 512, 64, 1, 3, 30.)
    rows = [{'regime': regime, 'arm': arm, 'seed': seed, 'horizon': h,
             'blind_regret': .3 if arm == 'rounded' else .4,
             'blind_cost_mse': .3, 'observed_cost_mse': .2, 'observed_kl': .05,
             'blind_survival_mae': .01, 'observed_survival_mae': .01, 'shuffled_blind_regret': .5}
            for regime in a.REGIMES for arm in a.ARMS for seed in spec.seeds for h in a.HORIZONS]
    baseline = [{'regime': regime, 'horizon': h, 'blind_regret': 1., 'blind_cost_mse': 1.}
                for regime in a.REGIMES for h in a.HORIZONS]
    fits = [{'arm': arm, 'seed': seed, 'seconds': 1000. if arm == 'rounded' else 1.}
            for arm in a.ARMS for seed in spec.seeds]
    counts = {'train': 300, 'base': 80, 'shift': 90}
    return spec, rows, baseline, fits, counts


def test_all_fifty_four_conditions_preserved_and_time_is_descriptive_only():
    spec, rows, baseline, fits, counts = metric_fixture()
    report = a.replication_comparisons(rows, baseline, counts, fits, spec=spec)
    assert report['advance']['passed'] and len(report['advance']['conditions']) == 54
    assert set(report['gates']) == {'base', 'shift'}
    for regime in a.REGIMES:
        comparison = report['allocation_comparisons'][regime]
        assert comparison['advance']['passed'] and len(comparison['advance']['conditions']) == 27
        assert comparison['mean_fit_time'][0]['ratio'] == 1000.
    assert not any('time' in key for key in report['advance']['conditions'])


@pytest.mark.parametrize('regime', ['base', 'shift'])
@pytest.mark.parametrize('fault', ['fifth_seed_pair', 'short_kl', 'blind_survival', 'observed_kl', 'support', 'zero_control'])
def test_neither_other_regime_nor_mean_can_rescue_failure(regime, fault):
    spec, rows, baseline, fits, counts = metric_fixture()
    if fault == 'support':
        counts[regime] = 63
    else:
        for row in rows:
            if row['regime'] != regime:
                continue
            if fault == 'zero_control':
                row['blind_regret'] = 0.
            elif row['arm'] == 'rounded' and row['seed'] == spec.seeds[-1]:
                if fault == 'fifth_seed_pair' and row['horizon'] == 4:
                    row['blind_regret'] = .401
                elif fault == 'short_kl' and row['horizon'] == 2:
                    row['observed_kl'] = .10001
                elif fault == 'blind_survival' and row['horizon'] == 8:
                    row['blind_survival_mae'] = .050001
                elif fault == 'observed_kl' and row['horizon'] == 4:
                    row['observed_kl'] = .10001
    report = a.replication_comparisons(rows, baseline, counts, fits, spec=spec)
    assert not report['advance']['passed']
    assert not report['allocation_comparisons'][regime]['advance']['passed']
    other = 'shift' if regime == 'base' else 'base'
    assert report['allocation_comparisons'][other]['advance']['passed']


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'unknown_regime', 'nonfinite'])
def test_incomplete_or_untyped_regime_metrics_rejected(fault):
    spec, rows, baseline, fits, counts = metric_fixture()
    if fault == 'duplicate':
        rows[-1] = copy.deepcopy(rows[-2])
    elif fault == 'missing':
        rows.pop()
    elif fault == 'unknown_regime':
        rows[0]['regime'] = 'pooled'
    else:
        rows[0]['blind_regret'] = float('nan')
    with pytest.raises(ValueError):
        a.replication_comparisons(rows, baseline, counts, fits, spec=spec)


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
    datasets['shift'] = {'case_ids': np.array(['e', 'f']),
                         'observations': np.array([[1] * 8, [0, 1, 2, 4, 4, 4, 4, 4]], np.int64)}
    prefixes = {'train': prefix, 'base': copy.deepcopy(prefix), 'shift': copy.deepcopy(prefix)}
    shifted = prefixes['shift']
    shifted['lengths'][1] = 3
    shifted['prefix'][1, 1, 8] = 0
    shifted['prefix'][1, 1, 4] = 1
    shifted['prefix'][1, 2, 2] = shifted['prefix'][1, 2, 8] = 1
    shifted['event_mask'] = np.arange(9)[None] < shifted['lengths'][:, None]
    work = {arm: {route: {} for route in a.ROUTES} for arm in a.ARMS}
    allocations, orders = {}, []
    events = int(prefix['event_mask'].sum())
    full_order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([945101, 0, 818]))).permutation(5)
    total = dict.fromkeys(a.UPDATE_WORK, 0)
    for arm in a.ARMS:
        trace = []
        orders.append({'arm': arm, 'seed': 945101, 'epoch': 0, 'indices': full_order.tolist(), 'batch_size': 2})
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
        allocations[arm, 945101] = {'trace': trace, 'attempted_updates': 4}
        for row in trace:
            for key, value in row['result']['work'].items():
                total[key] += value
        for regime in ('base', 'shift'):
            obs = datasets[regime]['observations']
            for route, observed in (('evaluation_blind', False), ('evaluation_observed', True), ('evaluation_shuffled', False)):
                prefix_block = blank()
                prefix_block.update(reset_emission_rows=2, prefix_filter_calls=8, prefix_filter_rows=16, operator_marginal_sum_calls=1)
                add(work[arm], route, fields(arm, 2), prefix_block, continuation(obs, observed))
            for start in range(0, 5, 2):
                add(work[arm], 'evaluation_prefix', fields(arm, 1), prefix_geometry(prefixes[regime], range(start, min(start + 2, 5))))
    counts = {'train_generation_count': 1, 'dev_generation_count': 2, 'model_constructions': 3,
        'oracle_model_constructions': 2, 'fit_count': 3, 'checkpoint_writes': 9, 'optimizer_checkpoint_writes': 9,
        'training_blind_rollouts': total['joint_blind_rollouts'], 'training_observed_rollouts': total['joint_observed_rollouts'],
        'optimizer_attempts': 12, 'optimizer_steps': 12, 'training_case_exposures': 6, 'training_attempt_exposures': 15,
        'training_prefix_rollouts': 12, 'training_prefix_event_exposures': 6 * events,
        'zero_endpoint_batches': total['zero_endpoint_batches'], 'evaluation_blind_rollouts': 6,
        'evaluation_observed_rollouts': 6, 'evaluation_shuffled_rollouts': 6, 'evaluation_case_views': 12,
        'evaluation_prefix_rollouts': 18, 'evaluation_prefix_event_views': 3 * (2 * events + 1),
        'oracle_blind_rollouts': 3, 'oracle_observed_rollouts': 3, 'array_decodes': 0, 'checkpoint_decodes': 0,
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
                                  'wrong_order', 'exposure', 'dropped_attempt', 'privilege', 'shift_work_missing'])
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
        allocations['rounded', 945101]['trace'].pop()
    elif fault == 'shift_work_missing':
        work['evaluation_observed']['cost_readout_rows'] -= 16
    else:
        work['joint_reuse_shared']['privileged_prefix_rows'] = 1
    with pytest.raises(ValueError):
        a.validate_execution(*values, np, spec=SPEC)


def test_wrong_producer_and_incomplete_manifest_fail_before_arrays(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: calls.append(args))
    (tmp_path / 'summary.json').write_text(json.dumps({'version': 'finite-reuse-learning-v1', 'implementation': 'reuse'}))
    with pytest.raises(ValueError, match='reuse-only'):
        a.audit(tmp_path, profile='engineering-945001')
    assert calls == []


def test_integration_metadata_cannot_admit_separate_fits():
    source = Path(a.__file__).read_text()
    # Source contract complements root's actual closed-producer smoke audit.
    assert "row['implementation'] == 'reuse'" in source
    assert "row['integration_version'] == 'finite-joint-reuse-training-v1'" in source
    assert "row['joint_structural_routes'] == list(REUSE_ROUTES)" in source
    assert math.isclose(a.PROFILES['science'].config['learning_rate'], .003)


def hand_world(epsilon):
    """Independent literal rational table, without either production world helper."""
    e = Fraction(str(epsilon))
    branches = np.empty((4, 4, 8, 8), np.float64)
    found = np.zeros((4, 8), np.float64)
    emission = np.array([[float(1 - e if o == j % 4 else e / 3) for j in range(8)]
                         for o in range(4)], np.float64)
    costs = np.array([[-.75 if d == ((j ^ (j >> 1)) & 3) else .25 for j in range(8)]
                      for d in range(4)], np.float64)
    for action in range(4):
        for current in range(8):
            destination = [current ^ 1, (current + 1) % 8,
                           ((current << 1) & 7) | (current >> 2), current ^ 4][action]
            for nxt in range(8):
                t = Fraction(1, 400) + (Fraction(49, 50) if nxt == destination else 0)
                hazard = Fraction(1 + (((nxt >> 2) & 1) ^ (action & 1)), 200)
                found[action, current] += float(t * hazard)
                for odor in range(4):
                    o = 1 - e if odor == nxt % 4 else e / 3
                    branches[action, odor, nxt, current] = float(t * (1 - hazard) * o)
    return branches, found, emission, costs


def hand_saved_split(split, *, epsilon=None):
    spec = a.StudyProfile('hand-prefix', 945001, (945101,), 3, 3, 2, 1, 3, 30.)
    b, found, emission, costs = hand_world(a.EPSILON[split] if epsilon is None else epsilon)
    marginal = b.sum(1)
    prefix = np.zeros((3, 9, 31), np.float32)
    lengths = np.array([9, 3, 9], np.int64)
    prefix_probabilities = np.zeros((3, 9, 5), np.float64)
    posterior = []
    for case in range(3):
        initial = case % 4
        prefix[case, 0, 4 + initial] = prefix[case, 0, 9] = 1
        prefix_probabilities[case, 0, :4] = emission.mean(1)
        state = emission[initial] / emission[initial].sum()
        for step in range(1, lengths[case]):
            action = (case + step) % 4
            odor = 4 if case == 1 and step == 2 else (case + 2 * step) % 4
            prefix[case, step, action] = prefix[case, step, 4 + odor] = 1
            branches = b[action] @ state
            law = np.append(branches.sum(1), found[action] @ state)
            prefix_probabilities[case, step] = law
            state = np.zeros(8) if odor == 4 else branches[odor] / law[odor]
        if case != 1:
            posterior.append(state)
    oracle = {'prefix_state': np.asarray(posterior, np.float64)}
    horizon = 2 if split == 'train' else 8
    ids = np.array([f'ns945001-split{a.SPLIT_IDS[split]}-case{i:010d}' for i in range(3)], dtype='U64')
    selected = np.array([0, 2])
    data = {'prefix': prefix[selected].copy(), 'lengths': np.full(2, 9, np.int64),
            'case_ids': ids[selected].copy(),
            'actions': np.array([[(i + h) % 4 for h in range(horizon)] for i in range(2)], np.int64),
            'observations': np.array([[0] + [4] * (horizon - 1), [1] * horizon], np.int64),
            'blind_costs': np.zeros((2, horizon, 4), np.float64),
            'observed_costs': np.zeros((2, horizon, 4), np.float64),
            'blind_survival': np.zeros((2, horizon), np.float64),
            'observed_survival': np.zeros((2, horizon), np.float64),
            'observed_probabilities': np.zeros((2, horizon, 5), np.float64)}
    for case in range(2):
        blind, observed = posterior[case].copy(), posterior[case].copy()
        terminal = False
        for step, action in enumerate(data['actions'][case]):
            blind = marginal[action] @ blind
            data['blind_costs'][case, step] = costs @ blind
            data['blind_survival'][case, step] = blind.sum()
            if terminal:
                data['observed_probabilities'][case, step, 4] = 1
                continue
            branches = b[action] @ observed
            law = np.append(branches.sum(1), found[action] @ observed)
            prior = branches.sum(0)
            data['observed_costs'][case, step] = costs @ prior
            data['observed_survival'][case, step] = prior.sum()
            data['observed_probabilities'][case, step] = law
            odor = data['observations'][case, step]
            terminal = odor == 4
            observed = np.zeros(8) if terminal else branches[odor] / law[odor]
    prefixes = {'prefix': prefix, 'lengths': lengths, 'case_ids': ids,
                'event_mask': np.arange(9)[None] < lengths[:, None],
                'endpoint_eligible': np.array([True, False, True]),
                'endpoint_rows': np.array([0, -1, 1], np.int64)}
    exact = {name: data[name].copy() for name in a.TARGETS}
    return spec, data, prefixes, oracle, exact, prefix_probabilities


@pytest.mark.parametrize('split', ['train', 'base', 'shift'])
def test_independent_rational_prefix_and_prelabel_targets_in_every_regime(split):
    spec, data, prefixes, oracle, exact, reference = hand_saved_split(split)
    _record, rebuilt, _uniform = a.reconstruct(data, oracle, exact, split, np, spec=spec)
    prefix_record, posterior, actual_reference = a.reconstruct_prefix(prefixes, data, split, np, spec=spec)
    assert prefix_record['retained'] == 2 and prefix_record['discarded_found'] == 1
    np.testing.assert_allclose(posterior, oracle['prefix_state'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_reference, reference, atol=1e-12, rtol=0)
    for field in a.TARGETS:
        np.testing.assert_allclose(rebuilt[field], exact[field], atol=1e-12, rtol=0)
    # A found label still has a pre-label forecast, with absorption only next step.
    assert rebuilt['observed_survival'][0, 1] > 0
    if split != 'train':
        assert not rebuilt['observed_costs'][0, 2:].any()
        assert np.all(rebuilt['observed_probabilities'][0, 2:, 4] == 1)
        assert np.all(rebuilt['blind_survival'][0] > 0)


def test_shift_cannot_use_base_oracle_even_with_shift_case_ids():
    spec, data, _prefix, oracle, exact, _reference = hand_saved_split('shift', epsilon=.12)
    with pytest.raises(ValueError):
        a.reconstruct(data, oracle, exact, 'shift', np, spec=spec)
    proper = hand_saved_split('shift')
    assert np.max(np.abs(proper[3]['prefix_state'] - oracle['prefix_state'])) > .01


def test_prefix_nll_keeps_regime_identity_and_first_found_event():
    spec, data, prefixes, _oracle, _exact, reference = hand_saved_split('shift')
    labels = prefixes['prefix'][:, :, 4:9].argmax(-1)
    nll = np.zeros((3, 9), np.float64)
    expected = []
    for i, j in zip(*np.nonzero(prefixes['event_mask']), strict=True):
        value = -math.log(reference[i, j, labels[i, j]])
        nll[i, j] = value
        expected.append(value)
    row = a.prefix_row(prefixes, {'probabilities': reference, 'nll': nll}, 'rounded', 945101,
                       np, regime='shift', spec=spec)
    assert row['regime'] == 'shift' and row['valid_events'] == 21
    assert row['mean_nll'] == math.fsum(expected) / 21
    assert nll[1, 2] > 0 and not nll[1, 3:].any()
    malformed = copy.deepcopy(prefixes)
    malformed['endpoint_rows'][1] = 0
    with pytest.raises(ValueError):
        a.reconstruct_prefix(malformed, data, 'shift', np, spec=spec)


def reference_metadata_fixture():
    shapes = {'costs': [4, 8], 'known_observed': [4, 4, 8, 8],
              'known_found': [4, 8], 'regime_epsilon': []}
    buffers = {name: {'shape': shape, 'count': math.prod(shape), 'dtype': 'torch.float64',
                     'bytes': 8 * math.prod(shape), 'requires_grad': False} for name, shape in shapes.items()}
    metadata = {regime: {'version': 'finite-regime-reference-v1', 'arm': 'exact_exact', 'seed': 0,
        'mass_width': 8, 'count': 0, 'trainable_count': 0, 'parameter_bytes': 0, 'parameters': {},
        'buffers': copy.deepcopy(buffers), 'buffer_bytes': 8712, 'privileged_prefix': True,
        'known_operators': True, 'fixed_cost_readout': True, 'epsilon': epsilon,
        'regime_state_buffer': 'regime_epsilon',
        'known_operator_source': f'finite_observation_world.world(epsilon={epsilon})'}
        for regime, epsilon in (('base', .12), ('shift', .30))}
    return {'regimes': {'train': {'split_id': 0, 'epsilon': .12}, 'base': {'split_id': 1, 'epsilon': .12},
                        'shift': {'split_id': 2, 'epsilon': .30}}, 'oracle_metadata': metadata,
            'oracle_state_sha256': {'base': 'a' * 64, 'shift': 'b' * 64}}


@pytest.mark.parametrize('fault', [None, 'wrong_law', 'same_state', 'unbound_regime', 'trainable', 'missing_buffer'])
def test_reference_metadata_is_regime_bound_before_arrays(fault):
    summary = reference_metadata_fixture()
    shifted = summary['oracle_metadata']['shift']
    if fault == 'wrong_law':
        shifted['epsilon'] = .12
    elif fault == 'same_state':
        summary['oracle_state_sha256']['shift'] = summary['oracle_state_sha256']['base']
    elif fault == 'unbound_regime':
        summary['regimes']['shift']['split_id'] = 1
    elif fault == 'trainable':
        shifted['trainable_count'] = 1
    elif fault == 'missing_buffer':
        shifted['buffers'].pop('regime_epsilon')
    if fault is None:
        a.validate_reference_metadata(summary)
    else:
        with pytest.raises(ValueError):
            a.validate_reference_metadata(summary)
