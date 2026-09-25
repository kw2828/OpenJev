# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent hand witnesses for the saved-evidence auditor; no production imports."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_fsm_author_nllfr.py'
SPEC = importlib.util.spec_from_file_location('independent_nllfr_audit_test', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def scalar_model():
    return {'A': np.array([[.5]]), 'B_u': np.array([[2., 0., 0.]]),
        'C_y': np.array([[1.], [2.], [-1.]]), 'D_yu': np.diag([1., 3., 4.]),
        'B_w': np.array([[.25]]), 'C_z': np.array([[1.]]),
        'D_yw': np.array([[1.], [-2.], [.5]]), 'D_zu': np.array([[.5, 0., 0.]]),
        'W0': np.array([[2.]]), 'b0': np.array([-1.]),
        'W1': np.array([[.5]]), 'b1': np.array([.5]),
        'W2': np.array([[3.]]), 'b2': np.array([-2.]),
        'u_mean': np.array([2., -3., .5]), 'u_std': np.array([.5, 2., 3.]),
        'y_mean': np.array([-4., 1., 7.]), 'y_std': np.array([3., .25, 2.]),
        'ts': np.array(1/6400., dtype=np.float64)}


def manual_scalar(inputs, initial):
    """Literal single-state arithmetic, independent of matrix implementation."""
    x, values = float(initial), []
    for u0, u1, u2 in inputs:
        h0 = max(0., 2*(x+.5*u0)-1)
        h1 = max(0., .5*h0+.5)
        w = 3*h1-2
        values.append([x+u0+w, 2*x+3*u1-2*w, -x+4*u2+.5*w])
        x = .5*x+2*u0+.25*w
    return np.asarray(values, dtype=np.float64), x


def test_literal_two_step_nonlinear_feedback_and_jacobian():
    m = scalar_model()
    u = np.array([[[3., 4., 5.], [0., 0., 0.]]])
    y, final, jac = audit.trajectory(m, u, np.array([[2.]]), jacobian=True)
    np.testing.assert_array_equal(y, [[[13.5, -1., 22.25], [34.5, -32.5, 3.5625]]])
    np.testing.assert_array_equal(final, [[10.90625]])
    np.testing.assert_array_equal(jac, [[[[4.], [-4.], [.5]], [[5.], [-5.], [.625]]]])
    eps = 1e-6
    yp = manual_scalar(u[0], 2+eps)[0]
    ym = manual_scalar(u[0], 2-eps)[0]
    np.testing.assert_allclose(jac[0, :, :, 0], (yp-ym)/(2*eps), rtol=2e-8, atol=2e-8)


def test_relu_zero_derivative_and_initial_state_ownership():
    m = scalar_model()
    m['D_zu'].fill(0.)
    state, u = np.array([[.5]]), np.zeros((1, 2, 3))
    before = state.copy()
    _, final, jac = audit.trajectory(m, u, state, jacobian=True)
    # First ReLU input is exactly zero, so its selected derivative is zero.
    np.testing.assert_array_equal(jac[0, 0, :, 0], [1., 2., -1.])
    np.testing.assert_array_equal(state, before)
    assert not np.shares_memory(final, state)


def test_second_observation_seed_and_forecast_index_with_physical_normalization():
    m = scalar_model()
    m['B_w'].fill(0.)
    m['D_yw'].fill(0.)
    # x1=2; output at u1=[3,4,5] precedes x2=7, then x3=3.5.
    yn = np.array([[[999., -777., 555.], [5., 16., 18.], [7., 17., 1.]]])
    un = np.array([[[3., 4., 5.], [0., 1., 2.]]])
    fu = np.array([[[4., 2., 1.]]])
    seed, linear = audit.linear_seed(m, yn, un)
    np.testing.assert_allclose(seed, [[2.]], rtol=0, atol=2e-15)
    assert linear['time'] == 'second observed output' and linear['rank'] == 1
    args = (yn*m['y_std']+m['y_mean'], un*m['u_std']+m['u_mean'],
            fu*m['u_std']+m['u_mean'])
    prediction, final, forecast, diagnostic = audit.replay_request(m, *args)
    np.testing.assert_allclose(forecast, [[3.5]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(final, [[9.75]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(prediction, np.array([[[7.5, 13., .5]]])*m['y_std']+m['y_mean'],
                               rtol=0, atol=1e-13)
    assert diagnostic['requests'][0]['status'] == 'GRADIENT_TOL'
    assert diagnostic['requests'][0]['accepted_steps'] == 0
    altered = args[0].copy()
    altered[:, 0] = 1e12
    again = audit.replay_request(m, altered, args[1], args[2])
    np.testing.assert_array_equal(again[0], prediction)
    np.testing.assert_array_equal(again[2], forecast)


def test_gn_nonzero_feedback_recovers_known_state_and_accounts_work():
    m = scalar_model()
    m.update(B_w=np.array([[.02]]), D_yw=np.array([[.1], [.2], [.05]]),
             D_zu=np.array([[.1, 0., 0.]]), W0=np.ones((1, 1)), b0=np.array([.5]),
             W1=np.ones((1, 1)), b1=np.array([.2]), W2=np.ones((1, 1)), b2=np.array([-.1]))
    inputs = np.zeros((1, 17, 3))
    inputs[:, :, 0] = .3
    x, target = 2., []
    for _ in range(17):
        w = x+.63
        target.append([x+.3+.1*w, 2*x+.2*w, -x+.05*w])
        x = .5*x+.6+.02*w
    target = np.array([target])
    final, fitted, row = audit.solve_context(m, target, inputs, np.array([[1.]]))
    np.testing.assert_allclose(fitted, [[2.]], rtol=0, atol=1e-7)
    np.testing.assert_allclose(final, [[x]], rtol=0, atol=1e-7)
    assert row['accepted_steps'] > 0 and row['final_objective'] < row['initial_objective']
    trials = [t for iteration in row['trace'] for t in iteration['trials']]
    assert row['trial_attempts'] == len(trials)
    assert row['jacobian_evaluations'] == len(row['trace'])+1
    assert row['trajectory_evaluations'] == len(row['trace'])+1+len(trials)
    assert row['accepted_steps'] == sum(t['accepted'] for t in trials)
    assert row['final_jacobian_rank'] == 1


def test_zero_jacobian_stays_dead_not_claimed_converged():
    m = scalar_model()
    m['C_y'].fill(0.)
    m['D_yw'].fill(0.)
    final, fitted, row = audit.solve_context(m, np.ones((1, 3, 3)), np.zeros((1, 3, 3)), np.zeros((1, 1)))
    assert row['status'] == 'ZERO_JACOBIAN' and row['accepted_steps'] == 0
    assert row['initial_objective'] == row['final_objective']
    assert row['final_objective'] == pytest.approx(.5, rel=0, abs=1e-15)
    assert row['jacobian_evaluations'] == row['trajectory_evaluations'] == 2
    np.testing.assert_array_equal(fitted, [[0.]])
    assert np.isfinite(final).all()


def test_dense_periodic_initial_state_matches_time_domain_orbit():
    a = np.array([[.3, .1], [0., .6]])
    b = np.array([[.2, -.1, .3], [.1, .2, .05]])
    t = np.arange(8, dtype=np.float64)
    u = np.empty((8, 3, 2))
    for r in range(2):
        u[:, :, r] = np.stack([np.cos(np.pi*t/4)+r, np.sin(np.pi*t/2), (-1.)**t*.3], axis=1)
    expected = []
    for r in range(2):
        forced = np.zeros(2)
        for value in u[:, :, r]:
            forced = a@forced+b@value
        x = np.linalg.solve(np.eye(2)-np.linalg.matrix_power(a, 8), forced)
        states = [x.copy()]
        for value in u[:, :, r]:
            x = a@x+b@value
            states.append(x.copy())
        np.testing.assert_allclose(states[8], states[0], rtol=0, atol=2e-15)
        expected.append(states[6])
    actual = audit.periodic_x0({'A': a, 'B_u': b}, np.fft.rfft(u, axis=0), samples=8, offset=2)
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), rtol=2e-14, atol=2e-15)


def test_full_spectrum_objective_uses_dc_nyquist_once_and_no_channel_divisor():
    m = scalar_model()
    m['C_y'].fill(0.)
    m['D_yu'].fill(0.)
    m['D_yw'].fill(0.)
    t = np.arange(8)
    # DC amplitude16, Nyquist24, one interior amplitude16: (256+576+256)/5.
    targets = np.stack([np.full(8, 2.), 3*(-1.)**t, 4*np.cos(np.pi*t/2)], axis=1)[:, :, None]
    target_spectrum = np.fft.rfft(targets, axis=0)
    loss = audit.native_objective(m, np.zeros((8, 3, 1)), target_spectrum, np.zeros((1, 1)), offset=2)
    assert loss == pytest.approx(1088/5, rel=0, abs=1e-12)


def test_fit_normalization_averages_periods_after_standardization_and_excludes_dev():
    raw = {}
    for a_index, amplitude in enumerate(('100mV', '200mV')):
        scalar = (np.arange(8)[:, None, None]+100*a_index
                  +10*np.arange(6)[None, :, None]+3*np.arange(2)[None, None, :])
        raw['u_'+amplitude+'_train'] = scalar[:, None]*np.array([1., 2., 3.])[None, :, None, None]+np.array([1., -2., 4.])[None, :, None, None]
        raw['y_'+amplitude+'_train'] = scalar[:, None]*np.array([-2., 1., .5])[None, :, None, None]+np.array([3., 4., 5.])[None, :, None, None]
        for variable in ('u', 'y'):
            raw[variable+'_'+amplitude+'_train'][:, :, 3:] = 1e50
    values, norm = audit.raw_fit_preprocessing(raw)
    scalar_mean = 65.
    scalar_variance = 5.25+2500+200/3+2.25
    np.testing.assert_allclose(norm['u_mean'], scalar_mean*np.array([1., 2., 3.])+[1., -2., 4.], rtol=0, atol=1e-13)
    np.testing.assert_allclose(norm['y_mean'], scalar_mean*np.array([-2., 1., .5])+[3., 4., 5.], rtol=0, atol=1e-13)
    np.testing.assert_allclose(norm['u_std'], np.sqrt(scalar_variance)*np.array([1., 2., 3.]), rtol=1e-14)
    assert values['raw_fit_u'].shape == (8, 3, 6, 2)
    assert values['u'].shape == (8, 3, 6) and values['U'].shape == (5, 3, 6)
    for name in ('u', 'y'):
        raw_fit = values['raw_fit_'+name]
        standardized = (raw_fit-norm[name+'_mean'][None, :, None, None])/norm[name+'_std'][None, :, None, None]
        np.testing.assert_allclose(values[name], standardized.mean(axis=3), rtol=0, atol=1e-15)
        np.testing.assert_allclose(values[name.upper()], np.fft.rfft(standardized, axis=0).mean(axis=3), rtol=0, atol=1e-14)


def rows(values):
    if np.isscalar(values):
        values = [values]*12
    return [{'record_id': rid, 'status': 'complete', 'rmse': float(value)}
            for rid, value in zip(audit.DEV, values, strict=True)]


def comparison_fixture(candidate=.5, control=1.):
    old = [{'family': family, 'seed': seed, 'rows': rows(candidate if family == audit.CANDIDATE else 10.)}
           for family, seed in audit.ORDERED]
    return old, rows(9.), rows(control)


def test_metrics_are_normalized_once_and_native_channel_errors_are_scaled():
    prediction = np.broadcast_to(np.array([1., 2., 3.]), (2, 7, 3)).copy()
    result = audit.metrics(prediction, np.zeros_like(prediction), np.array([2., 4., 8.]))
    assert result['mse'] == pytest.approx(14/3) and result['rmse'] == pytest.approx(np.sqrt(14/3))
    assert result['per_channel_rmse'] == [1., 2., 3.]
    assert result['native_output_per_channel_rmse'] == [2., 8., 24.]
    assert result['requests'] == 2 and result['horizon'] == 7
    prediction[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.metrics(prediction, np.zeros_like(prediction), np.ones(3))


def test_physical_score_subtracts_before_dividing_equal_large_values():
    large = np.full((1, 2, 3), 1e308)
    result = audit.physical_metrics(large, large.copy(), np.full(3, .01))
    assert result['mse'] == result['rmse'] == 0.
    assert result['native_output_per_channel_rmse'] == [0., 0., 0.]


def test_physical_subtraction_overflow_cannot_be_hidden_by_dividing_first():
    large = np.full((1, 2, 3), 1e308)
    with pytest.raises(ValueError, match='metric overflow') as raised:
        audit.physical_metrics(large, -large, np.full(3, 1e308))
    assert audit._numeric_failure(raised.value) is True
    with pytest.raises(ValueError, match='physical score dtype') as raised:
        audit.physical_metrics(np.zeros((1, 2, 3), dtype=np.float32), large, np.ones(3))
    assert audit._numeric_failure(raised.value) is False


def test_decisions_equal_record_then_seed_mean_instead_of_pooled_squared_errors():
    old, bla, new = comparison_fixture(control=4.)
    for e in old:
        if e['family'] == audit.CANDIDATE:
            k = e['seed']-9200
            e['rows'] = rows([.5*k]*6+[1.5*k]*6)
    result = audit.decisions(old, bla, new, True)
    assert result['family_means'][audit.CANDIDATE] == 2.
    assert result['per_seed_means'][audit.CANDIDATE] == {'9201': 1., '9202': 2., '9203': 3.}
    assert result['per_amplitude_means'][audit.CANDIDATE] == {'100mV': 1., '200mV': 3.}
    assert result['passed'] == 4 and result['strongest_control'] == audit.NLLFR


def test_paired_seed_control_cannot_be_replaced_by_its_family_mean():
    old, bla, new = comparison_fixture(control=8.)
    other = audit.NEURAL[0]
    for e in old:
        if e['family'] == audit.CANDIDATE:
            e['rows'] = rows({9201: .6, 9202: 1., 9203: 3.}[e['seed']])
        elif e['family'] == other:
            e['rows'] = rows({9201: .5, 9202: 2., 9203: 4.}[e['seed']])
    result = audit.decisions(old, bla, new, True)
    assert result['strongest_control'] == other and result['passed'] == 3
    assert result['conditions']['every_seed_below_strongest_control'] is False


@pytest.mark.parametrize('candidate,expected', [(.95, True), (.9500000001, False)])
def test_five_percent_boundary_is_not_rounded(candidate, expected):
    result = audit.decisions(*comparison_fixture(candidate=candidate), True)
    assert result['conditions']['five_percent_below_strongest_control'] is expected


@pytest.mark.parametrize('failure', ['not_complete', 'new_missing', 'new_nonfinite', 'new_failed',
                                     'bla_missing', 'candidate_failed'])
def test_incomplete_reference_or_candidate_cannot_pass(failure):
    old, bla, new = comparison_fixture()
    complete = failure != 'not_complete'
    if failure == 'new_missing':
        new.pop()
    elif failure == 'new_nonfinite':
        new[0]['rmse'] = np.inf
    elif failure == 'new_failed':
        new[0] = {'record_id': audit.DEV[0], 'status': 'incomplete'}
    elif failure == 'bla_missing':
        bla.pop()
    elif failure == 'candidate_failed':
        next(e for e in old if e['family'] == audit.CANDIDATE)['rows'][0]['status'] = 'failed'
    result = audit.decisions(old, bla, new, complete)
    assert result['passed'] == 0 and result['status'] == 'DO_NOT_CONTINUE_REFERENCE_CHECK'


def test_record_harm_and_amplitude_checks_do_not_collapse_to_overall_mean():
    old, bla, new = comparison_fixture(candidate=.5)
    for e in old:
        if e['family'] == audit.CANDIDATE:
            e['rows'] = rows([1.03]*6+[.1]*6)
    result = audit.decisions(old, bla, new, True)
    assert result['passed'] == 2
    assert not result['conditions']['no_record_over_two_percent_strongest_control']
    assert not result['conditions']['both_amplitudes_below_strongest_control']


def test_exact_parent_slot_identities_and_tie_break():
    old, bla, new = comparison_fixture()
    old[0]['rows'] = rows(1.)
    result = audit.decisions(old, bla, new, True)
    assert result['strongest_control'] == min(old[0]['family'], audit.NLLFR)
    old[0], old[1] = old[1], old[0]
    with pytest.raises(ValueError, match='25 comparison slots'):
        audit.decisions(old, bla, new, True)


def closed_fixture(monkeypatch, tmp_path, status='completed'):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)

    def put(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    producer = put('research/fsm_author/scripts/fit_nllfr.py', '# fabricated producer')
    supervisor = put('research/fsm_author/scripts/run_nllfr_study.py', '# fabricated supervisor')
    plan = {'source_sha256': {str(p.relative_to(tmp_path)): audit.pin(p)['sha256'] for p in (producer, supervisor)}}
    registration = put('registration.json', plan)
    folder = tmp_path/'study'
    for name in plan['source_sha256']:
        put('study/source/'+name, (tmp_path/name).read_text())
    put('study/final.npz', 'Invalid opaque NPZ sentinel. Never decode during admission.')
    put('study/started.json', {'pid': 1234, 'registration': audit.descriptor(registration)})
    log = put('process/process.log', 'Original fabricated terminal log')
    launch = {'phase': 'fit', 'status': 'running', 'command': [str(tmp_path/'research/fsm_author/.venv/bin/python'),
        str(producer), '--registration', str(registration), '--output', str(folder)],
        'registration_sha256': audit.pin(registration)['sha256'], 'timeout_seconds': 18000,
        'rss_cap_bytes': 32*1024**3, 'environment': audit.ENV.copy(),
        'producer': audit.descriptor(producer), 'supervisor': audit.descriptor(supervisor)}
    put('process/launch.json', launch)
    terminal = {**copy.deepcopy(launch), 'status': status, 'pid': 1234,
        'observed_exit_code': 0 if status == 'completed' else -9, 'elapsed_seconds': 12.,
        'end_identity_matches': True, 'identity_error': None, 'peak_polled_child_rss_bytes': 1000,
        'outcome': 'ORIGINAL_PROCESS_COMPLETE' if status == 'completed' else 'INCOMPLETE',
        'log_sha256': audit.pin(log)['sha256'], 'artifacts': audit.inventory(folder)}
    process = put('process/process.json', terminal)

    def forbidden(*_a, **_k):
        raise AssertionError('numerical decode before admission')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    monkeypatch.setattr(audit, 'read_raw', forbidden)
    return plan, registration, folder, process, terminal


@pytest.mark.parametrize('status', ['completed', 'failed', 'timeout', 'memory_limit'])
def test_closed_metadata_admission_retains_incomplete_originals(monkeypatch, tmp_path, status):
    plan, reg, folder, process, terminal = closed_fixture(monkeypatch, tmp_path, status)
    got, files = audit._closed_process(plan, reg, 'fit', folder, process)
    assert got == terminal and files == terminal['artifacts']


@pytest.mark.parametrize('status', ['timeout', 'memory_limit'])
def test_selected_stop_with_exit_zero_race_stays_incomplete(monkeypatch, tmp_path, status):
    plan, reg, folder, process, terminal = closed_fixture(monkeypatch, tmp_path, status)
    terminal['observed_exit_code'] = 0
    process.write_text(json.dumps(terminal))
    got, files = audit._closed_process(plan, reg, 'fit', folder, process)
    assert got['status'] == status and got['outcome'] == 'INCOMPLETE'
    assert got['observed_exit_code'] == 0 and files == terminal['artifacts']


@pytest.mark.parametrize('mutation', ['running', 'exit', 'command', 'environment', 'source',
    'source_snapshot', 'bytes', 'extra', 'missing', 'log', 'pid', 'registration', 'symlink'])
def test_original_launch_inventory_and_source_tamper_precede_array_reads(monkeypatch, tmp_path, mutation):
    plan, reg, folder, process, terminal = closed_fixture(monkeypatch, tmp_path)
    if mutation == 'running':
        terminal['status'] = 'running'
    elif mutation == 'exit':
        terminal['observed_exit_code'] = 1
    elif mutation == 'command':
        terminal['command'][1] = 'other.py'
    elif mutation == 'environment':
        terminal['environment']['PYTHONHASHSEED'] = '2'
    elif mutation == 'source':
        (tmp_path/next(iter(plan['source_sha256']))).write_text('changed')
    elif mutation == 'source_snapshot':
        (folder/'source'/next(iter(plan['source_sha256']))).write_text('changed')
        terminal['artifacts'] = audit.inventory(folder)
    elif mutation == 'bytes':
        terminal['artifacts']['final.npz']['bytes'] += 1
    elif mutation == 'extra':
        (folder/'extra').write_text('changed')
    elif mutation == 'missing':
        (folder/'final.npz').unlink()
    elif mutation == 'log':
        process.with_name('process.log').write_text('changed')
    elif mutation == 'pid':
        terminal['pid'] += 1
    elif mutation == 'registration':
        reg.write_text('{}')
    else:
        (folder/'link').symlink_to(reg)
    process.write_text(json.dumps(terminal))
    with pytest.raises(ValueError):
        audit._closed_process(plan, reg, 'fit', folder, process)


def fit_fixture(stopped=True):
    count = 2 if stopped else 10000
    initial, final = {'path': '/fake/initial', 'sha256': 'a'*64}, {'path': '/fake/final', 'sha256': 'b'*64}
    fit = {'iterations': count, 'author_stop_flag': stopped,
        'status': 'complete' if stopped else 'iteration_cap_reached',
        'initial': initial, 'final': final, 'trainable_scalars': 7473,
        'discarded_vendor_warmup_steps': 1, 'zip_roundtrip_all_numeric_arrays_bitwise_equal': True,
        'optimization_and_preservation_compile_inclusive_seconds': 3.,
        'author_reported_seconds': 2., 'peak_ru_maxrss_bytes': 1000}
    trace = {'loss_history': np.arange(count, dtype=np.float64)+1.,
             'iter_times': np.full(count, .001), 'iter_count': np.array(count, dtype=np.int64),
             'author_stop_flag': np.array(stopped), 'wall_time': np.array(2.)}
    return fit, trace, initial, final


@pytest.mark.parametrize('stopped', [True, False])
def test_raw_iteration_trace_is_not_claimed_as_monotone_accepted_updates(stopped):
    args = fit_fixture(stopped)
    audit.validate_fit(*args)
    assert np.all(np.diff(args[1]['loss_history']) > 0)


@pytest.mark.parametrize('mutation', ['short_trace', 'nan_loss', 'negative_time', 'counter',
    'flag', 'status', 'count_over_cap', 'count_bool', 'trainable_count', 'initial_pin', 'wall_time'])
def test_fit_trace_and_completion_scope_corruption(mutation):
    fit, trace, initial, final = fit_fixture()
    if mutation == 'short_trace':
        trace['loss_history'] = trace['loss_history'][:1]
    elif mutation == 'nan_loss':
        trace['loss_history'][0] = np.nan
    elif mutation == 'negative_time':
        trace['iter_times'][0] = -1.
    elif mutation == 'counter':
        trace['iter_count'] = np.array(1)
    elif mutation == 'flag':
        trace['author_stop_flag'] = np.array(False)
    elif mutation == 'status':
        fit['status'] = 'iteration_cap_reached'
    elif mutation == 'count_over_cap':
        fit['iterations'] = 10001
    elif mutation == 'count_bool':
        fit['iterations'] = True
    elif mutation == 'trainable_count':
        fit['trainable_scalars'] += 1
    elif mutation == 'initial_pin':
        initial = {'path': '/other'}
    else:
        trace['wall_time'] = np.array(3.)
    with pytest.raises(ValueError):
        audit.validate_fit(fit, trace, initial, final)


def costs_fixture():
    arrays = {'opaque_numeric': np.arange(7, dtype=np.float64)}
    timings = []
    for i, (rid, start) in enumerate((r, s) for r in audit.DEV for s in (0, 7936)):
        duration = float(i+1)
        timings.append({'record_id': rid, 'start': start, 'request_ms': duration,
                        'slice_ms': duration*.1, 'initializer_ms': duration*.4,
                        'rollout_ms': duration*.5})
    return {'timings': timings, 'timing_error': None, 'median_request_ms': 12.5,
            'persistent_numeric_bytes': 7*8+28*8+9*8}, arrays


def test_full_request_costs_use_saved_components_median_and_all_storage():
    data, arrays = costs_fixture()
    assert audit.validate_costs(data, arrays) == 352
    data.update(timings=data['timings'][:3], timing_error='FloatingPointError: original failure',
                median_request_ms=None)
    assert audit.validate_costs(data, arrays) == 352


@pytest.mark.parametrize('mutation', ['missing', 'reorder', 'component', 'median', 'storage',
                                    'nonfinite', 'failed_median'])
def test_costs_do_not_impute_failed_timings_or_hide_overhead(mutation):
    data, arrays = costs_fixture()
    if mutation == 'missing':
        data['timings'].pop()
    elif mutation == 'reorder':
        data['timings'][0], data['timings'][1] = data['timings'][1], data['timings'][0]
    elif mutation == 'component':
        data['timings'][0]['initializer_ms'] = 0.
    elif mutation == 'median':
        data['median_request_ms'] = 1.
    elif mutation == 'storage':
        data['persistent_numeric_bytes'] -= 8
    elif mutation == 'nonfinite':
        data['timings'][0]['request_ms'] = np.inf
    else:
        data['timing_error'] = 'failed'
    with pytest.raises(ValueError):
        audit.validate_costs(data, arrays)


def test_unbound_registration_rejects_before_any_numerical_reader(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'REGISTRATION_SHA256', None)

    def forbidden(*_a, **_k):
        raise AssertionError('numeric call before original admission')
    monkeypatch.setattr(audit, 'load_arrays', forbidden)
    monkeypatch.setattr(audit, 'read_raw', forbidden)
    with pytest.raises(ValueError, match='unbound/changed registration'):
        audit.authenticate(tmp_path/'study', tmp_path/'process.json')
