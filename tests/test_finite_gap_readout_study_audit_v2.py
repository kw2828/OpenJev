"""Independent fabricated checks; no study samples, models, or optimizer calls."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_gap_readout_study_v2 as AUDIT


@pytest.fixture(autouse=True)
def engineering_ids(monkeypatch):
    monkeypatch.setattr(AUDIT, 'NAMESPACE', 936001)
    monkeypatch.setattr(AUDIT, 'TRAIN_NAMESPACE', 934001)
    monkeypatch.setattr(AUDIT, 'SEEDS', (936101, 936102, 936103))
    monkeypatch.setattr(AUDIT, 'CONFIG', {**AUDIT.CONFIG, 'dev_seed_namespace': 936001,
                                      'train_namespace': 934001, 'parent_seeds': [936101, 936102, 936103]})


def problem():
    x = np.repeat(np.eye(8, dtype=np.float64)[:, None, :], 2, axis=1)
    p = np.zeros((4, 8), np.float64)
    p[np.arange(8) % 4, np.arange(8)] = 1.
    c = .25 - p
    states = {'blind_states': x * .5, 'observed_states': x.copy()}
    data = {'case_ids': np.array([f'fabricated-{i}' for i in range(8)]),
            'observations': np.zeros((8, 2), np.int64),
            'blind_costs': np.einsum('as,nhs->nha', c, x * .5),
            'observed_costs': np.einsum('as,nhs->nha', c, x)}
    return states, data, p


def result_record():
    states, data, p = problem()
    zero_gradient = [[0.] * 8 for _ in range(4)]
    certificate = {'objective': 0., 'gradient': zero_gradient, 'fw_gap': 0.,
                   'simplex_violation': 0., 'minimum_probability': 0., 'maximum_probability': 1., 'passed': True}
    history = [{'iteration': 0, 'objective': .234375, 'fw_gap': .46875, 'simplex_violation': 0.,
                'certificate_passed': False, 'objective_nonincreasing': True, 'qualifies': False},
               {'iteration': 10, 'objective': 0., 'fw_gap': 0., 'simplex_violation': 0.,
                'certificate_passed': True, 'objective_nonincreasing': True, 'qualifies': True}]
    record = {'version': 'finite-gap-readout-v1', 'complete': True, 'status': 'SOLVE_PASS',
              'failure_reasons': [], 'initial_probabilities': np.full((4, 8), .25).tolist(),
              'raw_probabilities': p.tolist(), 'probabilities': p.tolist(), 'cost_matrix': (.25 - p).tolist(),
              'objective_initial': .234375, 'objective_raw': 0., 'raw_feasibility': 0.,
              'projection': {'applied': False, 'max_abs': 0., 'l2': 0.}, 'certificate': certificate,
              'method': 'fixed-step-monotone-restarted-fista', 'lipschitz': .078125,
              'termination': 'direct_certificate', 'settings': dict(AUDIT.SETTINGS), 'certificate_history': history,
              'solver': {'success': True, 'status': 0, 'message': 'direct_certificate', 'iterations': 10, 'nfev': 11, 'njev': 10},
              'work': {'gradient_calls': 10, 'quadratic_value_calls': 11, 'projection_calls': 10,
                       'direct_objective_gradient_passes': 2, 'check_calls': 45,
                       'monotonicity_restarts': 0, 'iterations': 10, 'accepted_iterations': 10}}
    return states, data, record

def test_two_route_objective_and_signed_gradient_match_rational_oracle():
    states, data, p = problem()
    report = AUDIT.direct_objective(states, data, np.full((4, 8), .25), np)
    assert report['objective'] == .234375
    np.testing.assert_array_equal(report['gradient'], (.25 - p) * .078125)
    assert report['fw_gap'] == .46875
    assert report['passed'] is False
    optimal = AUDIT.direct_objective(states, data, p, np)
    assert optimal['objective'] == optimal['fw_gap'] == 0.
    assert optimal['passed'] is True


def test_absorbed_rows_stay_in_denominator_and_constant_tracks_mass():
    states, data, p = problem()
    for value in states.values():
        value[:, 1] = 0.
    for name in ('blind_costs', 'observed_costs'):
        data[name][:, 1] = 0.
    data['observations'][:, 0] = 4
    # Both routes use half as many nonzero targets, but the original full denominator.
    report = AUDIT.direct_objective(states, data, np.full((4, 8), .25), np)
    assert report['objective'] == .1171875
    assert AUDIT.direct_objective(states, data, p, np)['objective'] == 0.


def test_complete_hand_optimum_passes_without_solver_import_or_call():
    states, data, record = result_record()
    report = AUDIT.verify_solve(record, states, data, np)
    assert report['independently_certified'] is True
    assert report['objective_change'] == -.234375


@pytest.mark.parametrize('mutation', [
    lambda r: r['solver'].update(success=False),
    lambda r: r['solver'].update(status=9),
    lambda r: r['solver'].update(iterations=20001),
    lambda r: r['work'].update(iterations=20001, accepted_iterations=20001),
    lambda r: r['work'].update(projection_calls=11),
    lambda r: r['certificate'].update(objective=.1),
    lambda r: r.update(objective_initial=.234375 / 2),
    lambda r: r['raw_probabilities'][0].__setitem__(0, 1.000001),
    lambda r: r['probabilities'][1].__setitem__(0, .1),
    lambda r: r.update(complete=False),
])
def test_invalid_solve_or_claim_never_admitted(mutation):
    states, data, record = result_record()
    mutation(record)
    with pytest.raises(ValueError):
        AUDIT.verify_solve(record, states, data, np)


def test_curvature_bound_matches_independent_diagonal_hand_oracle():
    states, _, _ = problem()
    before = {name: value.copy() for name, value in states.items()}
    # Gram has eight equal diagonal entries2.5/64, so Hessian norm is5/64.
    assert AUDIT.scalar_lipschitz(states, np) == 5 / 64
    for name in states:
        np.testing.assert_array_equal(states[name], before[name])

def paired_fixture():
    _states, _data, p = problem()
    n, horizon = 2, 8
    x = np.full((n, horizon, 8), .125, np.float64)
    prefix = np.full((n, 8), .125, np.float64)
    states = {head + '_' + route + '_states': x.copy() for head in AUDIT.HEADS for route in ('blind', 'observed', 'shuffled')}
    states.update({head + '_' + field: prefix.copy() for head in AUDIT.HEADS for field in ('prefix_states', 'shuffled_prefix_states')})
    data = {'case_ids': np.array(['a', 'b']), 'observations': np.zeros((n, horizon), np.int64)}
    solve = {'initial_probabilities': p.tolist(), 'probabilities': p.tolist()}
    predictions = {}
    for head in AUDIT.HEADS:
        stem = f'factorized_{head}__936101__'
        predictions.update({stem + key: np.zeros((n, horizon, 4), np.float64)
                            for key in ('blind_costs', 'observed_costs', 'shuffled_blind_costs')})
        for key in ('blind_survival', 'observed_survival'):
            predictions[stem + key] = np.ones((n, horizon), np.float64)
        predictions[stem + 'observed_probabilities'] = np.full((n, horizon, 5), .2, np.float64)
    return states, predictions, solve, data


def test_separate_replays_must_preserve_all_state_and_noncost_bytes():
    states, predictions, solve, data = paired_fixture()
    result = AUDIT.verify_views('factorized', 936101, states, predictions, solve, data, np)
    assert result['latent_states_bitwise_equal']
    for field in ('solved_blind_states', 'solved_prefix_states', 'solved_shuffled_prefix_states'):
        changed = copy.deepcopy(states)
        changed[field].flat[0] += 1e-13
        with pytest.raises(ValueError, match='bitwise'):
            AUDIT.verify_views('factorized', 936101, changed, predictions, solve, data, np)
    changed_predictions = copy.deepcopy(predictions)
    changed_predictions['factorized_solved__936101__observed_probabilities'].flat[0] += 1e-13
    with pytest.raises(ValueError, match='bitwise'):
        AUDIT.verify_views('factorized', 936101, states, changed_predictions, solve, data, np)


def test_direct_mapping_rejects_cost_change_even_if_other_outputs_match():
    states, predictions, solve, data = paired_fixture()
    predictions['factorized_solved__936101__blind_costs'][0, 0, 0] = .001
    with pytest.raises(ValueError, match='head-to-prior'):
        AUDIT.verify_views('factorized', 936101, states, predictions, solve, data, np)


def test_nonfinite_or_negative_prior_and_postfound_state_rejected():
    states, data, p = problem()
    for invalid in (float('nan'), -.001):
        changed = copy.deepcopy(states)
        changed['blind_states'][0, 0, 0] = invalid
        with pytest.raises(ValueError):
            AUDIT.direct_objective(changed, data, p, np)
    data['observations'][0, 0] = 4
    with pytest.raises(ValueError, match='absorbed'):
        AUDIT.direct_objective(states, data, p, np)


def gate_rows():
    rows = []
    for arm in AUDIT.ARMS:
        for seed in AUDIT.SEEDS:
            for horizon in AUDIT.HORIZONS:
                rows.append({'arm': arm, 'seed': seed, 'regime': 'base', 'horizon': horizon, 'cases': 64,
                    'blind_cost_mse': .2, 'blind_regret': .2, 'observed_cost_mse': .01,
                    'blind_survival_mae': .04, 'observed_survival_mae': .01, 'observed_kl': .05,
                    'shuffled_blind_regret': .3})
    references = [{'regime': 'base', 'horizon': h, 'cases': 64, 'blind_cost_mse': .5, 'blind_regret': .5}
                  for h in AUDIT.HORIZONS]
    return rows, references


def test_all_six_groups_keep_absolute_criteria_separate_from_paired_improvement():
    rows, references = gate_rows()
    gates = AUDIT.gates(rows, references, {'train': 256, 'base': 64})
    assert len(gates) == 6
    assert all(value['passed'] for arm in gates.values() for value in arm.values())
    # One paired solved row gets better than original but still fails absolute H8 quality.
    original = next(row for row in rows if row['arm'] == 'factorized_original' and row['seed'] == AUDIT.SEEDS[0] and row['horizon'] == 8)
    solved = next(row for row in rows if row['arm'] == 'factorized_solved' and row['seed'] == AUDIT.SEEDS[0] and row['horizon'] == 8)
    original['blind_regret'], solved['blind_regret'] = .4, .3
    gates = AUDIT.gates(rows, references, {'train': 256, 'base': 64})
    assert not gates['factorized_solved']['BLIND_EXTRAPOLATION']['passed']
    assert gates['factorized_solved']['SHORT_HORIZON_LEARNING']['passed']
    paired = next(row for row in AUDIT.paired_comparisons(rows) if row['parent'] == 'factorized' and row['seed'] == AUDIT.SEEDS[0] and row['horizon'] == 8)
    assert paired['metrics']['blind_regret']['difference'] == pytest.approx(-.1)
    assert paired['metrics']['blind_regret']['relative_reduction'] == pytest.approx(.25)
    assert len(AUDIT.paired_comparisons(rows)) == 36


@pytest.mark.parametrize('failure', ['missing_view', 'train_support', 'dev_support', 'zero_baseline', 'one_observed_failure'])
def test_gate_cannot_rescue_missing_pairs_or_bad_support_with_averages(failure):
    rows, references = gate_rows()
    counts = {'train': 256, 'base': 64}
    if failure == 'missing_view':
        with pytest.raises(ValueError, match='roster'):
            AUDIT.gates(rows[:-1], references, counts)
        return
    if failure == 'train_support':
        counts['train'] = 255
    elif failure == 'dev_support':
        counts['base'] = 63
    elif failure == 'zero_baseline':
        references[-1]['blind_regret'] = 0.
    else:
        rows[-1]['observed_kl'] = .100001
    result = AUDIT.gates(rows, references, counts)
    assert any(not value['passed'] for arm in result.values() for value in arm.values())


def test_incomplete_payload_roster_fails_before_any_array_decode(tmp_path, monkeypatch):
    import json

    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'summary.json').write_text(json.dumps({'version': 'finite-gap-readout-study-v1',
                                                   'config': AUDIT.CONFIG, 'files': {}}))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('numerical decoder reached before metadata admission'))
    with pytest.raises(ValueError, match='38-payload'):
        AUDIT.audit(folder)


def test_scalar_certificate_preserves_raw_negative_roundoff_semantics():
    states, data, p = problem()
    p[0, 0] += 1e-9
    report = AUDIT.direct_objective(states, data, p, np)
    assert report['simplex_violation'] > 1e-12
    assert report['passed'] is False


def metadata_fixture(tmp_path):
    import json

    upstream = tmp_path / 'prior'
    original = upstream / 'run-01'
    original.mkdir(parents=True)
    (original / 'train.npz').write_bytes(b'opaque fabricated TRAIN')
    parent_fits = []
    for parent in AUDIT.PARENTS:
        for seed in AUDIT.SEEDS:
            name = f'{parent}-{seed}.npz'
            (original / name).write_bytes(f'opaque checkpoint {parent} {seed}'.encode())
            parent_fits.append({'arm': parent, 'seed': seed, 'checkpoint': {'path': name, **AUDIT.descriptor(original / name)},
                                'final_state_sha256': 'a' * 64, 'readout_final': {'matrix': [[0.] * 8 for _ in range(4)]}})
    train_counts = {'retained': 8, 'seed_namespace': AUDIT.TRAIN_NAMESPACE, 'attempted': 512}
    parent_summary = {'fits': parent_fits, 'dataset_counts': {'train': train_counts}}
    (original / 'summary.json').write_text(json.dumps(parent_summary))
    upstream_files = {'run-01/' + path.name: AUDIT.descriptor(path) for path in original.iterdir()}
    folder = tmp_path / 'new-run'
    folder.mkdir()
    for name in AUDIT.COMMON_FILES:
        (folder / name).write_bytes(b'opaque saved payload')
    (folder / 'train.npz').write_bytes((original / 'train.npz').read_bytes())
    (folder / 'config.json').write_text(json.dumps(AUDIT.CONFIG))
    solves = []
    for index, seed in enumerate(AUDIT.SEEDS):
        order = AUDIT.PARENTS[index:] + AUDIT.PARENTS[:index]
        for parent in order:
            for prefix in ('states-train', 'states-base'):
                (folder / f'{prefix}-{parent}-{seed}.npz').write_bytes(b'opaque saved states')
            state_name = f'states-train-{parent}-{seed}.npz'
            source = original / f'{parent}-{seed}.npz'
            row = {'parent': parent, 'seed': seed, 'source_checkpoint': {'path': str(source), **AUDIT.descriptor(source)},
                   'source_state_sha256': 'a' * 64, 'model_state_after': 'a' * 64,
                   'original_cost_matrix': [[0.] * 8 for _ in range(4)], 'frozen_weights': True,
                   'optimizer_weight_updates': 0, 'horizon': 2, 'training_cases': 8,
                   'load_seconds': 0., 'extraction_seconds': 0., 'solve_seconds': 0.,
                   'train_original_head_max_error': 0.,
                   'train_state_file': {'path': state_name, **AUDIT.descriptor(folder / state_name)}}
            (folder / f'solve-{parent}-{seed}.json').write_text(json.dumps(row))
            solves.append(row)
    (folder / 'solves.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in solves))
    barrier = {'solves': [{'parent': row['parent'], 'seed': row['seed'],
                         'file': f"solve-{row['parent']}-{row['seed']}.json",
                         **AUDIT.descriptor(folder / f"solve-{row['parent']}-{row['seed']}.json")} for row in solves],
               'completed_solves': 9, 'dev_generation_count': 0}
    (folder / 'solve-barrier.json').write_text(json.dumps(barrier))
    times = [{'parent': parent, 'seed': seed, 'head': head, 'arm': parent + '_' + head,
              'horizon': 8, 'cases': 4, 'model_state_before': 'a' * 64, 'model_state_after': 'a' * 64,
              'seconds': 0., 'original_head_max_error': 0.}
             for parent in AUDIT.PARENTS for seed in AUDIT.SEEDS for head in AUDIT.HEADS]
    (folder / 'prediction-times.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in times))
    oracle_checks = dict.fromkeys(AUDIT.TARGETS, 0.)
    (folder / 'oracle-base-check.json').write_text(json.dumps(oracle_checks))
    summary = {'version': 'finite-gap-readout-study-v1', 'config': AUDIT.CONFIG,
               'upstream': {'folder': str(upstream), 'run': str(original), 'files': upstream_files},
               'original_train_descriptor': AUDIT.descriptor(original / 'train.npz'), 'original_train_counts': train_counts,
               'dataset_counts': {'base': {'retained': 4}}, 'solves': solves, 'solve_barrier': barrier,
               'prediction_times': times, 'oracle_checks': oracle_checks,
               'files': {path.name: AUDIT.descriptor(path) for path in folder.iterdir()}}
    (folder / 'summary.json').write_text(json.dumps(summary))
    return folder, summary, original


def test_complete_opaque_metadata_joins_require_no_array_or_checkpoint_decode(tmp_path, monkeypatch):
    folder, summary, _ = metadata_fixture(tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata decoded numerical content'))
    report = AUDIT.validate_metadata(folder, summary)
    assert report['payloads'] == 38 and report['views'] == 18
    assert report['checkpoint_decodes'] == 0
    assert report['historical_barrier_independently_replayed'] is False


def test_original_checkpoint_changed_after_inventory_is_rejected(tmp_path):
    folder, summary, original = metadata_fixture(tmp_path)
    (original / f'factorized-{AUDIT.SEEDS[0]}.npz').write_bytes(b'changed source checkpoint')
    with pytest.raises(ValueError, match='original input byte descriptor'):
        AUDIT.validate_metadata(folder, summary)


def test_initial_certificate_has_no_updates_projections_or_quadratic_calls():
    states, data, row = result_record()
    row['initial_probabilities'] = copy.deepcopy(row['probabilities'])
    row['objective_initial'] = 0.
    row['termination'] = 'initial_certificate'
    row['certificate_history'] = [{'iteration': 0, 'objective': 0., 'fw_gap': 0., 'simplex_violation': 0.,
                                   'certificate_passed': True, 'objective_nonincreasing': True, 'qualifies': True}]
    row['solver'].update(message='initial_certificate', iterations=0, nfev=0, njev=0)
    row['work'].update(gradient_calls=0, quadratic_value_calls=0, projection_calls=0,
                       direct_objective_gradient_passes=1, check_calls=3, iterations=0, accepted_iterations=0)
    result = AUDIT.verify_solve(row, states, data, np)
    assert result['history']['iterations'] == 0
    assert result['history']['certificate_checks'] == 1
    assert result['objective_change'] == 0.


def test_restart_work_counts_include_second_gradient_projection_and_value():
    states, data, row = result_record()
    row['work'].update(monotonicity_restarts=2, gradient_calls=12, projection_calls=12,
                       quadratic_value_calls=13, check_calls=51)
    row['solver'].update(nfev=13, njev=12)
    result = AUDIT.verify_solve(row, states, data, np)
    assert result['history']['restarts'] == 2
    assert result['history']['intermediate_iterates_independently_replayed'] is False


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(lipschitz=.078126),
    lambda r: r.update(method='other'),
    lambda r: r['settings'].update(max_iterations=20001),
    lambda r: r['settings'].update(certificate_every=20),
    lambda r: r['settings'].update(monotonicity_roundoff=1e-14),
    lambda r: r['projection'].update(applied=True),
    lambda r: r['projection'].update(max_abs=1e-14),
    lambda r: r['raw_probabilities'][1].__setitem__(0, -0.),
    lambda r: r['work'].update(check_calls=44),
    lambda r: r['work'].update(direct_objective_gradient_passes=3),
    lambda r: r['work'].update(gradient_calls=11),
    lambda r: r['work'].update(quadratic_value_calls=12),
    lambda r: r['work'].update(accepted_iterations=9),
    lambda r: r['work'].update(monotonicity_restarts=11),
    lambda r: r['certificate_history'].pop(0),
    lambda r: r['certificate_history'][0].update(fw_gap=.5),
    lambda r: r['certificate_history'][1].update(iteration=9),
    lambda r: r['certificate_history'][1].update(objective_nonincreasing=False),
    lambda r: r['certificate_history'][1].update(certificate_passed=False),
    lambda r: r['certificate_history'][1].update(qualifies=False),
    lambda r: r.update(termination='final_direct_certificate'),
])
def test_registered_gap_settings_history_and_work_tampering_rejected(mutation):
    states, data, row = result_record()
    mutation(row)
    with pytest.raises(ValueError):
        AUDIT.verify_solve(row, states, data, np)


def test_middle_history_is_attested_but_stopping_at_first_qualified_is_checked():
    states, data, row = result_record()
    final = copy.deepcopy(row['certificate_history'][-1])
    final['iteration'] = 20
    row['certificate_history'][-1].update(objective=.1, fw_gap=.2, certificate_passed=False, qualifies=False)
    row['certificate_history'].append(final)
    row['solver'].update(iterations=20, nfev=21, njev=20)
    row['work'].update(iterations=20, accepted_iterations=20, gradient_calls=20, projection_calls=20,
                       quadratic_value_calls=21, direct_objective_gradient_passes=3, check_calls=86)
    result = AUDIT.verify_solve(row, states, data, np)
    assert result['history']['certificate_checks'] == 3
    assert not result['history']['intermediate_iterates_independently_replayed']
    # A continued run after an earlier qualifying check violates the fixed rule.
    row['certificate_history'][1].update(objective=0., fw_gap=0., certificate_passed=True, qualifies=True)
    with pytest.raises(ValueError, match='first qualified'):
        AUDIT.verify_solve(row, states, data, np)
