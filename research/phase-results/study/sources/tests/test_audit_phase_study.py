"""Independent algebra and provenance guards, with fabricated inputs only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_phase_study as audit


def phase(arm='fixed_phase'):
    result = {name: np.zeros(shape, dtype=np.float32) for name, shape in audit.shapes(arm).items()}
    # Both rotations are pi/2; radius .9999/2.
    result['input_weights'][0, 0] = 1
    result['readout_weights'][0, 0] = 1
    return result


def test_phase_rotation_and_same_index_input_have_hand_impulse_solution():
    got = audit.phase_predict(phase(), np.array([1, 0, 0, 0, 0], np.float32), 'fixed_phase')
    radius = .9999 / 2
    np.testing.assert_allclose(got, [1, 0, -radius ** 2, 0, radius ** 4], atol=1e-7, rtol=1e-6)


def test_nonlinear_readout_is_radial_cubic_not_changed_transition():
    p = phase('nonlinear_readout')
    p['cubic_readout_weights'][0, 0] = 2
    got = audit.phase_predict(p, np.array([2, 0, 0], np.float32), 'nonlinear_readout')
    radius = .9999 / 2
    np.testing.assert_allclose(got, [18, 0, -2 * radius ** 2 - 16 * radius ** 6], atol=2e-6)


def test_energy_uses_previous_state_and_zero_phase_matches_linear_control():
    p = phase('energy_phase')
    u = np.array([1, .5, -.25, 0], np.float32)
    np.testing.assert_array_equal(audit.phase_predict(p, u, 'energy_phase'),
                                  audit.phase_predict(phase(), u, 'fixed_phase'))
    p['phase_logits'][:] = 1
    p['energy_scales_raw'][:] = 1
    got = audit.phase_predict(p, u, 'energy_phase')
    assert got[0] == 1
    assert abs(float(got[1]) - .5) > .01


def test_phase_is_causal_repeatable_and_owns_inputs():
    p = phase('energy_phase')
    p['phase_logits'][:] = .2
    u = np.linspace(-1, 1, 24, dtype=np.float32)
    before = {k: v.copy() for k, v in p.items()}
    original = audit.phase_predict(p, u, 'energy_phase')
    changed = u.copy(); changed[12:] = 10
    np.testing.assert_array_equal(original[:12], audit.phase_predict(p, changed, 'energy_phase')[:12])
    np.testing.assert_array_equal(original, audit.phase_predict(p, u, 'energy_phase'))
    for key in p:
        np.testing.assert_array_equal(p[key], before[key])


def test_gru_reset_gate_applies_to_recurrent_candidate_bias():
    p = {key: np.zeros(shape, np.float32) for key, shape in audit.GRU_SHAPES.items()}
    p['gru.bias_hh_l0'][32] = 2
    p['readout.weight'][0, 0] = 1
    got = audit.gru_predict(p, np.zeros(3, np.float32))
    target = np.tanh(1.)
    np.testing.assert_allclose(got, target * np.array([.5, .75, .875]), atol=1e-7)
    assert abs(float(got[0]) - .5 * np.tanh(2.)) > .05


def test_ar2_recurses_on_predictions_with_current_and_previous_inputs():
    p = {'coefficient': np.array([.5, .25, 1, 2, 0, 0, 0], np.float32)}
    got = audit.ar2_predict(p, np.array([1, 0, 0, 0], np.float32))
    np.testing.assert_array_equal(got, [1, 2.5, 1.5, 1.375])


def test_ar2_cubic_and_bias_order():
    p = {'coefficient': np.array([0, 0, 0, 0, 2, 3, 1], np.float32)}
    np.testing.assert_array_equal(audit.ar2_predict(p, np.zeros(3, np.float32)), [1, 6, 721])


def test_ar2_divergence_is_retained_not_clamped():
    p = {'coefficient': np.array([0, 0, 0, 0, 0, 4, 1], np.float32)}
    got = audit.ar2_predict(p, np.zeros(30, np.float32))
    assert not np.isfinite(got).all()
    assert audit.metric(got, np.zeros(30), 1., skip=0) == {'finite': False, 'rmse': None, 'nrmse': None}


@pytest.mark.parametrize('mutation', [lambda p: p.pop('bias'), lambda p: p.update(extra=np.zeros(1)),
    lambda p: p.update(rho_logits=np.zeros(2)), lambda p: p.update(angle_logits=np.zeros(3, np.float32)),
    lambda p: p.update(bias=np.array(np.nan, np.float32))])
def test_parameter_schema_rejects_invalid_arrays(mutation):
    p = phase(); mutation(p)
    with pytest.raises(ValueError):
        audit.phase_predict(p, np.ones(2, np.float32), 'fixed_phase')


def test_ridge_intercept_is_penalized_and_uses_frozen_basis_order():
    u = np.zeros(5)
    y = np.ones(5) * 2
    coefficient = audit.fit_ridge(u, y, 'static_cubic')
    np.testing.assert_allclose(coefficient, [10 / (5 + audit.RIDGE), 0, 0, 0], rtol=1e-12)
    design, skip = audit.ridge_design(np.array([2., 3.]), None, 'static_cubic')
    np.testing.assert_array_equal(design, [[1, 2, 4, 8], [1, 3, 9, 27]])
    assert skip == 0


def test_ar2_fit_design_uses_only_completed_output_and_input_lags():
    design, skip = audit.ridge_design(np.array([2., 3., 4.]), np.array([5., 6., 7.]), 'cubic_ar2')
    assert skip == 2
    np.testing.assert_array_equal(design[2], [6, 5, 4, 3, 36, 216, 1])


def test_fir_alignment_padding_and_fit_burn_are_distinct():
    u = np.arange(140, dtype=np.float64)
    matrix, skip = audit.ridge_design(u, None, 'fir128')
    assert matrix.shape == (140, 129) and skip == 127
    np.testing.assert_array_equal(matrix[0, :4], [1, 0, 0, 0])
    np.testing.assert_array_equal(matrix[3, :6], [1, 3, 2, 1, 0, 0])
    np.testing.assert_array_equal(matrix[130, 1:], u[130:2:-1])
    coefficients = np.zeros(129); coefficients[0] = 2; coefficients[2] = 3
    np.testing.assert_array_equal(audit.classical_predict(coefficients, u, 'fir128'),
                                  2 + 3 * np.concatenate(([0], u[:-1])))


def test_scoring_skips_common_warmup_but_finite_guard_checks_it():
    target = np.ones(1024)
    predicted = target.copy(); predicted[:512] = 100; predicted[512:] = 3
    assert audit.metric(predicted, target, 4.) == {'finite': True, 'rmse': 2., 'nrmse': .5}
    predicted[0] = np.nan
    assert audit.metric(predicted, target, 4.)['finite'] is False


def fake_csv(*, poison=None):
    rows = ['not-numeric,private-test,\n'] * 131072
    for start, end in ((40650, 83946), (84446, 92638), (93138, 101330)):
        rows[start:end] = ['1.25,-2.5e-2,\n'] * (end - start)
    if poison is not None:
        rows[poison] = 'bad,0,\n'
    return ('V1,V2,\n' + ''.join(rows)).encode()


def test_csv_only_selected_train_intervals_are_numerically_converted():
    got = audit.parse_csv(fake_csv())
    for name, start, end in (('fit', 40650, 83946), ('dev_a', 84446, 92638), ('dev_b', 93138, 101330)):
        np.testing.assert_array_equal(got[name]['raw_indices'], np.arange(start, end))
        np.testing.assert_array_equal(got[name]['u'], np.full(end - start, 1.25))
        np.testing.assert_array_equal(got[name]['y'], np.full(end - start, -.025))


@pytest.mark.parametrize('index', [40650, 83945, 84446, 92637, 93138, 101329])
def test_selected_bad_value_cannot_be_dropped(index):
    with pytest.raises(ValueError, match='decimal'):
        audit.parse_csv(fake_csv(poison=index))


def test_array_reconciliation_rejects_dtype_roster_nan_and_exact_signedzero():
    with pytest.raises(ValueError, match='dtype'):
        audit.array_agreement(np.ones(2, np.float64), np.ones(2, np.float32), 'prediction')
    with pytest.raises(ValueError, match='finite'):
        audit.array_agreement(np.array([np.nan], np.float32), np.ones(1, np.float32), 'prediction')
    with pytest.raises(ValueError, match='bytes'):
        audit.array_agreement(np.array([-0.]), np.array([0.]), 'raw', exact=True)
    audit.array_agreement(np.array([1.00001], np.float32), np.ones(1, np.float32), 'prediction')


def score_fixture():
    rows = {}
    for partition in audit.PARTITIONS:
        row = {}
        for arm in audit.ARMS:
            for seed in audit.SEEDS:
                value = .5 if arm == 'energy_phase' else 1.
                row[f'{arm}/{seed}'] = {'rmse_mv': value, 'mae_mv': value / 2, 'scored_rows': 7680}
        row.update({name: {'rmse_mv': 1., 'mae_mv': .5, 'scored_rows': 7680} for name in audit.REFERENCES})
        rows[partition] = row
    return rows


def test_full_gate_roster_and_both_partitions_required():
    result = audit.evaluate_rule(score_fixture(), .01)
    assert result['passed'] == result['total'] == 21
    assert result['outcome'] == 'ADVANCE_PHASE_MECHANISM'
    assert len({row['name'] for row in result['conditions']}) == 21
    with pytest.raises(ValueError, match='partitions'):
        audit.evaluate_rule({'dev_a': score_fixture()['dev_a']}, .01)


def test_seed_failure_cannot_be_rescued_by_improved_family_mean():
    scores = score_fixture()
    scores['dev_a']['energy_phase/7301']['rmse_mv'] = 1.01
    result = audit.evaluate_rule(scores, .02)
    failed = [row['name'] for row in result['conditions'] if not row['passed']]
    assert failed == ['dev_a_seed7301_energy_not_worse_fixed_phase',
                      'dev_a_seed7301_energy_not_worse_nonlinear_readout']


@pytest.mark.parametrize('control', ['gru16', 'cubic_ar2', *audit.REFERENCES])
def test_every_strong_control_can_block_advance(control):
    scores = score_fixture()
    keys = [f'{control}/{seed}' for seed in audit.SEEDS] if control in audit.ARMS else [control]
    for key in keys:
        scores['dev_b'][key]['rmse_mv'] = .1
    result = audit.evaluate_rule(scores, .01)
    assert [row['name'] for row in result['conditions'] if not row['passed']] == ['dev_b_within_5pct_best_conventional']


def optimizer_fixture(arm='energy_phase', steps=2048):
    return {f'{name}.{field}': (np.array(steps, np.float32) if field == 'step' else np.zeros(shape, np.float32))
            for name, shape in audit.shapes(arm).items() for field in ('exp_avg', 'exp_avg_sq', 'step')}


@pytest.mark.parametrize('mutation', [lambda p: p.pop('bias.step'),
    lambda p: p.update({'bias.step': np.array(2047, np.float32)}),
    lambda p: p.update({'bias.exp_avg': np.array(np.inf, np.float32)}),
    lambda p: p.update({'bias.exp_avg_sq': np.array(-.01, np.float32)}),
    lambda p: p.update({'bias.exp_avg_sq': np.zeros(1, np.float32)})])
def test_saved_adam_counter_shape_and_finite_guards(mutation):
    p = optimizer_fixture()
    audit.validate_optimizer(p, 'energy_phase', 2048)
    mutation(p)
    with pytest.raises(ValueError):
        audit.validate_optimizer(p, 'energy_phase', 2048)


def test_training_trace_is_complete_without_monotonic_loss_requirement():
    trace = {'loss': np.arange(2048, dtype=np.float64), 'gradnorm': np.ones(2048)}
    fit = {'arm': 'energy_phase', 'seed': 7301, 'updates': 2048, 'elapsed_seconds': 2.,
           'loss_initial_batch': 0., 'loss_final_batch': 2047.,
           'state_initialization': 'zero every training sequence', 'dev_or_test_access': False}
    audit.validate_training(trace, fit, arm='energy_phase', seed=7301, seconds=2.)
    fit['dev_or_test_access'] = True
    with pytest.raises(ValueError, match='exact'):
        audit.validate_training(trace, fit, arm='energy_phase', seed=7301, seconds=2.)


def test_exact_file_roster_and_parameter_geometry():
    names = audit.roster()
    assert len(names) == 135
    assert sum(name.endswith('.npz') for name in names) == 64
    assert sum(name.endswith('.npy') for name in names) == 41
    got = audit.resources()['trained_models']
    assert [got[f'{arm}/7301']['parameters'] for arm in audit.ARMS] == [14, 18, 18, 929, 7]
    assert [got[f'{arm}/7301']['state_scalars'] for arm in audit.ARMS] == [4, 4, 4, 16, 3]


def receipt_fixture(tmp_path, monkeypatch, cap=2400):
    import json

    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    wrapper = tmp_path / 'output/phase-engineering-v1/invoke.py'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('# fabricated wrapper\n')
    log = tmp_path / 'run.log'; log.write_text('closed\n')
    pins = {'source.py': {'bytes': 1, 'sha256': 'a' * 64}}
    receipt = {'state': 'EXITED', 'returncode': 0, 'argv': ['python', 'run'], 'elapsed_seconds': 1.,
               'thread_env': {key: '1' for key in audit.THREADS}, 'sources_before': pins, 'sources_after': pins,
               'log_path': 'run.log', 'log': audit.descriptor(log), 'timeout_seconds': cap,
               'wrapper': audit.descriptor(wrapper)}
    path = tmp_path / 'run.json'; path.write_text(json.dumps(receipt))
    return path, pins, receipt


@pytest.mark.parametrize('mutation', [lambda p: p.update(returncode=False), lambda p: p.update(state='RUNNING'),
    lambda p: p.update(elapsed_seconds=2400.01), lambda p: p.update(timeout_seconds=300),
    lambda p: p.update(sources_after={}), lambda p: p.update(thread_env={}),
    lambda p: p.update(log_path='../run.log'), lambda p: p['log'].update(bytes=999),
    lambda p: p['wrapper'].update(sha256='0' * 64), lambda p: p['argv'].append('extra')])
def test_original_closure_guards(tmp_path, monkeypatch, mutation):
    import json

    path, pins, receipt = receipt_fixture(tmp_path, monkeypatch)
    audit.closed_receipt(path, pins, cap=2400, expected_argv=['python', 'run'])
    mutation(receipt); path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit.closed_receipt(path, pins, cap=2400, expected_argv=['python', 'run'])


def admission_fixture(tmp_path, monkeypatch):
    import json
    from importlib.metadata import version

    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    for name in audit.SOURCES:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# opaque fabricated source: ' + name)
    source_pins = {name: audit.descriptor(tmp_path / name) for name in audit.SOURCES}
    engineering = tmp_path / 'output/phase-engineering-v1'; engineering.mkdir(parents=True)
    wrapper = engineering / 'invoke.py'; wrapper.write_text('# fabricated wrapper')
    csv = tmp_path / 'raw.csv'; csv.write_bytes(b'opaque,not,numerically,read')
    monkeypatch.setattr(audit, 'CSV_SHA', audit.descriptor(csv)['sha256'])
    for name in audit.THREADS:
        monkeypatch.setenv(name, '1')

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    def receipt(name, argv, cap, seconds):
        log = engineering / (name + '.log'); log.write_text('fabricated completed log')
        value = {'state': 'EXITED', 'returncode': 0, 'argv': argv, 'elapsed_seconds': seconds,
            'thread_env': {key: '1' for key in audit.THREADS}, 'sources_before': source_pins, 'sources_after': source_pins,
            'log_path': log.name, 'log': audit.descriptor(log), 'timeout_seconds': cap, 'wrapper': audit.descriptor(wrapper)}
        path = engineering / (name + '.json'); write(path, value)
        return path

    qualification = receipt('qualification', audit.QUAL_ARGV, 300, .1)
    plan = {'config': audit.config(), 'sources': source_pins, 'csv': audit.descriptor(csv),
        'qualification': audit.descriptor(qualification), 'environment': {'python': sys.version, 'numpy': np.__version__,
        'torch': version('torch'), 'threads': {key: '1' for key in audit.THREADS}, 'device': 'cpu'}}
    registration = tmp_path / 'research/phase-registration.json'; write(registration, plan)
    folder = tmp_path / 'output/study'
    for name in audit.roster():
        path = folder / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'opaque payload')
    for name in audit.SOURCES:
        (folder / 'sources' / name).write_bytes((tmp_path / name).read_bytes())
    (folder / 'registration.json').write_bytes(registration.read_bytes())
    (folder / 'qualification.json').write_bytes(qualification.read_bytes())
    command = ['.venv/bin/python', 'scripts/phase_study.py', 'run', '--csv', 'raw.csv',
        '--registration', 'research/phase-registration.json', '--qualification',
        'output/phase-engineering-v1/qualification.json', '--output', 'output/study']
    process = receipt('run', command, 2400, 3.)
    run = {'version': 'phase-study-v1', 'registration': audit.descriptor(registration),
        'registration_commit': 'a' * 40, 'csv': plan['csv'], 'models_fitted': 15, 'ridge_fits': 4,
        'optimizer_updates': 15 * 2048, 'fit_seconds': {f'{arm}/{seed}': .05 for arm in audit.ARMS for seed in audit.SEEDS},
        'elapsed_seconds': 2., 'numeric_partitions_loaded': ['fit', 'dev_a', 'dev_b'], 'official_test_values_read': 0,
        'source_pins_verified_before_and_after': True}
    write(folder / 'run.json', run)
    write(folder / 'manifest.json', {'files': {name: audit.descriptor(folder / name) for name in audit.roster()}})
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda command, **kwargs:
                        (tmp_path / command[2].split(':', 1)[1]).read_bytes())
    return folder, csv, process


def test_complete_opaque_admission_without_npz_or_measurement_decoding(tmp_path, monkeypatch):
    folder, csv, process = admission_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit.np, 'load', lambda *args, **kwargs: pytest.fail('array decoding'))
    monkeypatch.setattr(audit, 'parse_csv', lambda *args: pytest.fail('measurement parsing'))
    got = audit.authenticate(folder, csv, process)
    assert got['manifest_files'] == 135
    assert got['original_process_seconds'] == 3.


@pytest.mark.parametrize('change', ['source', 'snapshot', 'payload', 'extra', 'missing', 'run_argv'])
def test_opaque_admission_failure_precedes_numeric_access(tmp_path, monkeypatch, change):
    import json

    folder, csv, process = admission_fixture(tmp_path, monkeypatch)
    if change == 'source':
        (tmp_path / 'scripts/phase_study.py').write_text('changed')
    elif change == 'snapshot':
        (folder / 'sources/scripts/phase_study.py').write_text('changed')
    elif change == 'payload':
        (folder / 'data-fit.npz').write_text('changed')
    elif change == 'extra':
        (folder / 'extra.npz').write_text('unregistered')
    elif change == 'missing':
        (folder / 'references.npz').unlink()
    else:
        receipt = json.loads(process.read_text()); receipt['argv'][1] = 'other.py'
        process.write_text(json.dumps(receipt))
    monkeypatch.setattr(audit.np, 'load', lambda *args, **kwargs: pytest.fail('array decoding before authentication'))
    monkeypatch.setattr(audit, 'parse_csv', lambda *args: pytest.fail('measurement parsing before authentication'))
    with pytest.raises(ValueError):
        audit.audit(folder, csv_path=csv, run_receipt=process)


def test_bandlimited_ridge_coefficient_disagreement_is_not_prediction_failure():
    # Rank-poor deterministic engineering signal, not any Silverbox measurement.
    t = np.arange(8192, dtype=np.float64)
    u = np.sin(.013 * t) + .3 * np.cos(.031 * t) + .2 * np.sin(.077 * t)
    y = .3 * u + .2 * np.sin(.013 * (t - 4)) - .15 * np.cos(.031 * (t - 7)) + .01 * u * u
    design, skip = audit.ridge_design(u, y, 'fir512')
    used = design[skip:]
    coefficient = np.linalg.solve(used.T @ used + audit.RIDGE * np.eye(513), used.T @ y[skip:])
    independent = audit.fit_ridge(u, y, 'fir512')
    certificate = audit.ridge_certificate(coefficient, independent, u, y, 'fir512')
    assert certificate['dimension'] == 513 and certificate['fitting_rows'] == 7681
    unit = np.finfo(np.float64).eps / 2
    assert certificate['backward_error_limit'] == 64 * (513 * unit / (1 - 513 * unit))
    assert all(row['backward_error'] <= certificate['backward_error_limit']
               for row in certificate['solutions'].values())
    assert certificate['fit_prediction_max_abs_difference'] < audit.ATOL


def test_ridge_certificate_rejects_changed_solution_and_wrong_dtype():
    u = np.linspace(-1, 1, 32)
    y = .3 + .2 * u + .1 * u * u
    coefficient = audit.fit_ridge(u, y, 'static_cubic')
    bad = coefficient.copy(); bad[0] += .01
    with pytest.raises(ValueError, match='backward error'):
        audit.ridge_certificate(bad, coefficient, u, y, 'static_cubic')
    with pytest.raises(ValueError, match='finite ridge'):
        audit.ridge_certificate(coefficient.astype(np.float32), coefficient, u, y, 'static_cubic')


def test_zero_information_certificate_has_explicit_zero_denominator():
    u = np.linspace(-1, 1, 32)
    y = np.zeros(32)
    coefficient = audit.fit_ridge(u, y, 'static_cubic')
    result = audit.ridge_certificate(coefficient, coefficient, u, y, 'static_cubic')
    assert result['solutions']['saved'] == {'residual_inf': 0., 'denominator': 0., 'backward_error': 0.}
