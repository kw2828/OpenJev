"""Fabricated timed-stage and checkpoint joins; no model or training replay."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_expected_count_learning as a


@pytest.fixture(autouse=True)
def engineering_ids(monkeypatch):
    monkeypatch.setattr(a, 'NAMESPACE', 937001)
    monkeypatch.setattr(a, 'SEEDS', (937101, 937102, 937103))
    monkeypatch.setattr(a, 'CONFIG', {**a.CONFIG, 'seed_namespace': 937001, 'fit_seeds': [937101, 937102, 937103]})


def timed_record(method='gradient', stop='late'):
    # Four fabricated prefixes have12 total events,8 actions and one found.
    events, attempts, found = 12, 4, 1
    late = stop == 'late'
    extra_loop = stop == 'next_loop'
    completed = 11. if late else 9.5 if extra_loop else 10.
    first_delta = dict.fromkeys(a.TIMED_WORK_KEYS, 0)
    first_delta.update(parameter_snapshot_tensors=3, parameter_snapshot_bytes=2560,
                       parameter_hash_calls=2, clock_calls=1)
    if method == 'gradient':
        first_delta.update(full_gradient_passes=1, torch_diagnostic_passes=1, adam_updates=1,
                           probability_checks=1, gradient_event_exposures=12, check_calls=1)
    else:
        first_delta.update(numpy_expectation_passes=2, map_updates=1, probability_imports=1,
                           probability_checks=3, expectation_event_exposures=24, check_calls=91)
    second_delta = dict(first_delta)
    if late:
        second_delta.update(parameter_restore_tensors=3, parameter_restore_bytes=2560, parameter_hash_calls=3)
    trace = []
    for i, (start, end, before, attempted, retained, delta) in enumerate((
            (.25, 2., 'a', 'b', 'b', first_delta),
            (8., completed, 'b', 'c', 'b' if late else 'c', second_delta))):
        trace.append({'update': i + 1, 'start_elapsed': start, 'completed_elapsed': end,
            'seconds': end - start, 'accepted': not (i == 1 and late), 'rolled_back': i == 1 and late,
            'penalized_log_likelihood_before': -20. + i, 'penalized_log_likelihood_after': -19. + i,
            'roundtrip_max_abs': 0., 'dynamics_before_sha256': before * 64,
            'dynamics_attempted_sha256': attempted * 64, 'dynamics_retained_sha256': retained * 64,
            'work_delta': delta})
    total = {key: first_delta[key] + second_delta[key] for key in a.TIMED_WORK_KEYS}
    total['torch_diagnostic_passes'] += 1
    total['parameter_hash_calls'] += 2
    total['clock_calls'] += 4 + int(extra_loop)
    total['check_calls'] += 4 + int(extra_loop)
    numpy_work = {} if method == 'gradient' else {
        'sequences': 16, 'reset_events': 16, 'action_events': 32, 'ordinary_events': 28,
        'found_events': 4, 'forward_matvecs': 32, 'backward_matvecs': 32,
        'transition_posterior_matrices': 32, 'state_posterior_rows': 48, 'check_calls': 176}
    timed = 11.25 if late else 10.25
    return {'version': 'finite-timed-prefix-v1', 'method': method, 'status': 'PASS',
        'termination': 'late_update_rolled_back' if late else 'budget_reached',
        'budget_seconds': 10., 'max_updates': 100000, 'pseudocount': .001,
        'learning_rate': .003, 'gradient_clip': 5., 'accepted_updates': 1 if late else 2,
        'attempted_updates': 2, 'trace': trace, 'valid_events': events, 'attempts': attempts,
        'initial_penalized_log_likelihood': -20., 'final_penalized_log_likelihood': -19. if late else -18.,
        'initial_dynamics_sha256': 'a' * 64, 'final_dynamics_sha256': ('b' if late else 'c') * 64,
        'initial_head_sha256': 'd' * 64, 'final_head_sha256': 'd' * 64,
        'head_unchanged': True, 'work': total, 'numpy_work': numpy_work,
        'timed_seconds': timed, 'overrun_seconds': timed - 10., 'final_summary_seconds': .1,
        'final_summary_work': {'torch_diagnostic_passes': 1, 'parameter_hash_calls': 2, 'check_calls': 1, 'clock_calls': 1},
        'compute_matched': False, 'hidden_state_input': False, 'optimizer_state_reused_across_steps': method == 'gradient',
        'late_optimizer_state_discarded': method == 'gradient' and late}, events, attempts, found


@pytest.mark.parametrize('method', ['gradient', 'em'])
@pytest.mark.parametrize('stop', ['late', 'equal', 'next_loop'])
def test_actual_work_and_atomic_deadline_paths(method, stop):
    row, events, attempts, found = timed_record(method, stop)
    report = a.validate_timed_result(row, method, events, attempts, found)
    assert report['accepted_updates'] == (1 if stop == 'late' else 2)
    assert report['attempted_updates'] == 2
    assert report['late_updates'] == int(stop == 'late')
    assert not report['historical_updates_independently_replayed']
    assert row['work']['clock_calls'] == (7 if stop == 'next_loop' else 6)
    expected_checks = (186 if method == 'em' else 6) + int(stop == 'next_loop')
    assert row['work']['check_calls'] == expected_checks
    assert row['work']['parameter_snapshot_bytes'] == 5120
    assert row['work']['parameter_restore_bytes'] == (2560 if stop == 'late' else 0)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(status='FAILED_UPDATE_CAP'),
    lambda r: r.update(budget_seconds=11.),
    lambda r: r.update(max_updates=100001),
    lambda r: r.update(pseudocount=.002),
    lambda r: r.update(accepted_updates=2),
    lambda r: r.update(head_unchanged=False),
    lambda r: r.update(compute_matched=True),
    lambda r: r.update(hidden_state_input=True),
    lambda r: r.update(final_head_sha256='e' * 64),
    lambda r: r.update(final_dynamics_sha256='c' * 64),
    lambda r: r.update(final_penalized_log_likelihood=-18.),
    lambda r: r.update(overrun_seconds=0.),
    lambda r: r.update(timed_seconds=10.),
    lambda r: r['trace'][1].update(accepted=True, rolled_back=False),
    lambda r: r['trace'][1].update(dynamics_retained_sha256='c' * 64),
    lambda r: r['trace'][1].update(dynamics_before_sha256='a' * 64),
    lambda r: r['trace'][1].update(start_elapsed=10.),
    lambda r: r['trace'][0].update(start_elapsed=-1.),
    lambda r: r['trace'][0].update(seconds=.5),
    lambda r: r['work'].update(parameter_restore_bytes=0),
    lambda r: r['work'].update(clock_calls=5),
    lambda r: r['work'].update(check_calls=5),
    lambda r: r['trace'][1]['work_delta'].update(parameter_restore_tensors=0),
    lambda r: r['final_summary_work'].update(torch_diagnostic_passes=0),
])
def test_timing_retention_and_accounting_tampering_rejected(mutation):
    row, events, attempts, found = timed_record()
    mutation(row)
    with pytest.raises(ValueError):
        a.validate_timed_result(row, 'gradient', events, attempts, found)


def test_em_monotonicity_is_checked_but_gradient_is_not_selected_by_likelihood():
    em, events, attempts, found = timed_record('em')
    em['trace'][1]['penalized_log_likelihood_after'] = -20.
    with pytest.raises(ValueError, match='EM likelihood'):
        a.validate_timed_result(em, 'em', events, attempts, found)
    gradient, *_ = timed_record('gradient')
    gradient['trace'][1]['penalized_log_likelihood_after'] = -100.
    assert a.validate_timed_result(gradient, 'gradient', events, attempts, found)['late_updates'] == 1


def test_nested_numpy_callbacks_are_joined_not_double_added_to_total():
    row, events, attempts, found = timed_record('em')
    assert row['numpy_work']['check_calls'] == 176
    assert row['work']['check_calls'] == 186
    row['work']['check_calls'] += 176
    with pytest.raises(ValueError, match='timed setup'):
        a.validate_timed_result(row, 'em', events, attempts, found)


def independent_hashes(arrays):
    full = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        header = json.dumps([name, '<f8', list(value.shape)], separators=(',', ':')).encode()
        full.update(len(header).to_bytes(8, 'little') + header + value.astype('<f8').tobytes())
    raw = b''.join(name.encode() + b'\0' + arrays[name].astype('<f8').tobytes()
                   for name in ('transition_logits', 'emission_logits', 'hazard_logits'))
    return {'full': full.hexdigest(), 'dynamics': hashlib.sha256(raw).hexdigest(),
            'head': hashlib.sha256(arrays['cost_logits'].astype('<f8').tobytes()).hexdigest()}


def boundary_fixture():
    records, checkpoints = [], {}
    for seed in a.SEEDS:
        initial = {name: np.zeros(shape, np.float64) for name, shape in a.PARAMETER_SHAPES.items()}
        for arm in a.ARMS:
            first = {name: value.copy() for name, value in initial.items()}
            final = {name: value.copy() for name, value in initial.items()}
            method = a.PRETRAIN_METHODS[arm]
            if method != 'none':
                final['hazard_logits'][:] = .25 if method == 'gradient' else .5
            before, after = independent_hashes(first), independent_hashes(final)
            row = {'arm': arm, 'seed': seed, 'method': method, 'initial_state_sha256': before['full'],
                   'final_state_sha256': after['full'], 'result': {}}
            if method != 'none':
                row['result'] = {'initial_dynamics_sha256': before['dynamics'], 'final_dynamics_sha256': after['dynamics'],
                                 'initial_head_sha256': before['head'], 'final_head_sha256': after['head'],
                                 'trace': [{'dynamics_retained_sha256': after['dynamics']}]}
            records.append(row)
            checkpoints['initial', arm, seed], checkpoints['final', arm, seed] = first, final
    return records, checkpoints


def test_all_decoded_boundaries_join_full_dynamic_head_hashes_and_retained_trace():
    records, checkpoints = boundary_fixture()
    result = a.verify_boundaries(records, checkpoints, np)
    assert len(result) == 9 and all(row['head_bytes_unchanged'] and row['retained_state_joined'] for row in result)
    for arrays in checkpoints.values():
        assert a.boundary_hashes(arrays, np) == independent_hashes(arrays)


@pytest.mark.parametrize('mutation', ['head', 'dynamic', 'trace', 'common_initial', 'dtype', 'extra_tensor', 'none_changed'])
def test_checkpoint_content_cannot_be_replaced_by_matching_metadata_claims(mutation):
    records, checkpoints = boundary_fixture()
    row = next(row for row in records if row['arm'] == 'gradient_prefix')
    final = checkpoints['final', row['arm'], row['seed']]
    if mutation == 'head':
        final['cost_logits'][0, 0] = 1.
    elif mutation == 'dynamic':
        final['hazard_logits'][0, 0] = 1.
    elif mutation == 'trace':
        row['result']['trace'][0]['dynamics_retained_sha256'] = 'f' * 64
    elif mutation == 'common_initial':
        initial = checkpoints['initial', row['arm'], row['seed']]
        initial['transition_logits'][0, 0, 0] = .1
        changed = independent_hashes(initial)
        row['initial_state_sha256'] = changed['full']
        row['result']['initial_dynamics_sha256'] = changed['dynamics']
    elif mutation == 'dtype':
        final['hazard_logits'] = final['hazard_logits'].astype(np.float32)
    elif mutation == 'extra_tensor':
        final['extra'] = np.zeros(1, np.float64)
    else:
        none_row = next(row for row in records if row['arm'] == 'joint_only')
        value = checkpoints['final', none_row['arm'], none_row['seed']]
        value['hazard_logits'][0, 0] = .5
        none_row['final_state_sha256'] = independent_hashes(value)['full']
    with pytest.raises(ValueError):
        a.verify_boundaries(records, checkpoints, np)


def test_joint_initial_state_must_equal_retained_pretraining_and_shared_head():
    records, _checkpoints = boundary_fixture()
    fits = [{'arm': row['arm'], 'seed': row['seed'], 'initial_state_sha256': row['final_state_sha256'],
             'readout_initial': {'matrix': [[0.] * 8 for _ in range(4)]}} for row in records]
    assert len(a.initial_function_checks(fits, records)) == 3
    fits[1]['initial_state_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='joint training'):
        a.initial_function_checks(fits, records)


def test_incomplete_metadata_rejected_before_any_checkpoint_decode(tmp_path, monkeypatch):
    summary = {'version': 'finite-expected-count-learning-v1', 'config': a.CONFIG,
               'fits': [{'arm': arm, 'seed': seed} for i, seed in enumerate(a.SEEDS) for arm in a.ARMS[i:] + a.ARMS[:i]], 'files': {}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('decoded before metadata admission'))
    with pytest.raises(ValueError, match='payload roster'):
        a.audit(tmp_path)


def test_unchanged_absolute_criteria_require_every_seed_and_cannot_average_pretraining_gains():
    rows = [{'arm': arm, 'seed': seed, 'horizon': horizon, 'blind_cost_mse': .2, 'blind_regret': .2,
             'observed_kl': .05, 'blind_survival_mae': .04} for arm in a.ARMS for seed in a.SEEDS for horizon in a.HORIZONS]
    references = [{'horizon': horizon, 'blind_cost_mse': .5, 'blind_regret': .5} for horizon in a.HORIZONS]
    report = a.gates(rows, references, {'train': 256, 'base': 64})
    assert all(criterion['passed'] for criteria in report.values() for criterion in criteria.values())
    next(row for row in rows if row['arm'] == 'em_prefix' and row['seed'] == a.SEEDS[0] and row['horizon'] == 8)['blind_regret'] = .3
    report = a.gates(rows, references, {'train': 256, 'base': 64})
    assert not report['em_prefix']['BLIND_EXTRAPOLATION']['passed']
    assert report['em_prefix']['OBSERVED_FILTERING_EXTRAPOLATION']['passed']
