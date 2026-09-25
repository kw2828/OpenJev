# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated causal NL-LFR state inference; no measurements or saved models."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from openjev_fsm_author import linear_context  # noqa: E402
from openjev_fsm_author import nllfr_context as context  # noqa: E402


def model(nx=3):
    """Small nonzero-feedback system, or sparse production-shape witness."""
    nz, nw, width = (16, 8, 64) if nx == 28 else (3, 3, 3)
    a = np.eye(nx)*.55
    a[:3, :3] = [[.91, .015, 0.], [0., .87, .02], [.01, 0., .93]]
    bu = np.zeros((nx, 3))
    bu[:3] = np.eye(3)*.07
    cy = np.zeros((3, nx))
    cy[:, :3] = np.array([[1., .1, 0.], [0., 1., .1], [.1, 0., 1.]])
    bw = np.zeros((nx, nw))
    bw[:3, :3] = np.eye(3)*.015
    cz = np.zeros((nz, nx))
    cz[:3, :3] = np.eye(3)
    dyw = np.zeros((3, nw))
    dyw[:3, :3] = np.eye(3)*.12
    dzu = np.zeros((nz, 3))
    dzu[:3] = np.diag([.5, -.4, .3])
    w0 = np.zeros((width, nz))
    w0[:3, :3] = np.eye(3)
    w1 = np.eye(width)
    w2 = np.zeros((nw, width))
    w2[:3, :3] = np.diag([.25, -.2, .15])
    b0, b1, b2 = np.zeros(width), np.zeros(width), np.zeros(nw)
    b0[:3] = [.05, -.05, .1]
    b1[:3] = [.01, .02, -.01]
    b2[:3] = [.02, -.03, .01]
    return {'A': a, 'B_u': bu, 'C_y': cy, 'D_yu': np.diag([.2, -.1, .3]),
        'B_w': bw, 'C_z': cz, 'D_yw': dyw, 'D_zu': dzu,
        'W0': w0, 'b0': b0, 'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2,
        'u_mean': np.array([2., -3., 5.]), 'u_std': np.array([.5, 2., 3.]),
        'y_mean': np.array([10., -20., 30.]), 'y_std': np.array([2., .25, 4.]),
        'ts': np.array(1/6400)}


def oracle(a, inputs, initial):
    y = np.empty((len(inputs), inputs.shape[1], 3))
    states = np.empty((len(inputs), inputs.shape[1]+1, initial.shape[1]))
    states[:, 0] = initial
    for b in range(len(inputs)):
        for t in range(inputs.shape[1]):
            x, u = states[b, t], inputs[b, t]
            z = a['C_z']@x+a['D_zu']@u
            h = np.array([max(0., v) for v in a['W0']@z+a['b0']])
            h = np.array([max(0., v) for v in a['W1']@h+a['b1']])
            w = a['W2']@h+a['b2']
            y[b, t] = a['C_y']@x+a['D_yu']@u+a['D_yw']@w
            states[b, t+1] = a['A']@x+a['B_u']@u+a['B_w']@w
    return y, states


def fixture(nx=3, batch=2, length=100, horizon=128):
    a = model(nx)
    rng = np.random.default_rng(121)
    inputs = rng.normal(scale=.5, size=(batch, length-1+horizon, 3))
    initial = np.zeros((batch, nx))
    initial[:, :3] = rng.normal(scale=.3, size=(batch, 3))
    y, states = oracle(a, inputs, initial)
    yc = np.concatenate([np.full((batch, 1, 3), 123.), y[:, :length-1]], axis=1)
    return (a, yc*a['y_std']+a['y_mean'],
        inputs[:, :length-1]*a['u_std']+a['u_mean'],
        inputs[:, length-1:]*a['u_std']+a['u_mean'],
        y[:, length-1:]*a['y_std']+a['y_mean'], states, initial)


@pytest.mark.parametrize('nx,batch,length,horizon', [(3, 1, 2, 7), (3, 3, 17, 32), (3, 2, 100, 128), (28, 1, 100, 128)])
def test_known_nonzero_feedback_state_recovery_and_forecast(nx, batch, length, horizon):
    a, yc, uc, fu, target, states, initial = fixture(nx, batch, length, horizon)
    prediction, final, diag = context.predict(a, yc, uc, fu)
    np.testing.assert_allclose(prediction, target, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(final, states[:, -1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(diag['solved_context_start_states'], initial, rtol=1e-6, atol=1e-6)
    forecast, _ = context.condition(a, yc, uc)
    np.testing.assert_allclose(forecast, states[:, length-1], rtol=1e-6, atol=1e-6)
    assert any(r['accepted_steps'] > 0 for r in diag['requests'])
    assert not np.array_equal(diag['linear_seed_states'], initial)
    for record in diag['requests']:
        assert record['final_objective'] <= record['initial_objective']
        assert record['status'] in ('GRADIENT_TOL', 'ITERATION_CAP', 'STALLED')


def test_literal_policy_and_complete_trace_work_accounting():
    a, yc, uc, _, _, _, _ = fixture()
    _, diag = context.condition(a, yc, uc)
    assert diag['policy'] == {'iterations': 16, 'trials': 8, 'damping': .001,
        'armijo': .0001, 'gradient_tol': 1e-8, 'scale_floor': 1e-8, 'rcond': 1e-12}
    for record in diag['requests']:
        trace = record['trace']
        assert [r['iteration'] for r in trace] == list(range(len(trace)))
        assert 1 <= record['directions_considered'] == len(trace) <= 16
        trials = [t for row in trace for t in row['trials']]
        assert record['accepted_steps'] == sum(t['accepted'] for t in trials)
        assert record['jacobian_evaluations'] == len(trace)+1
        assert record['trajectory_evaluations'] == len(trace)+1+len(trials)
        assert record['trial_attempts'] == len(trials) <= 128
        previous = record['initial_objective']
        for row in trace:
            assert row['objective'] <= previous
            assert len(row['trials']) <= 8
            assert [t['alpha'] for t in row['trials']] == [2.**(-i) for i in range(len(row['trials']))]
            accepted = [t for t in row['trials'] if t['accepted']]
            assert len(accepted) <= 1
            if accepted:
                assert accepted[0] is row['trials'][-1]
                assert not accepted[0]['nonfinite']
                assert accepted[0]['objective'] < row['objective']
                previous = accepted[0]['objective']
            else:
                previous = row['objective']
        assert record['final_objective'] <= previous


def test_linear_reduction_uses_second_observation_state_not_end_state():
    a = model()
    a['B_w'].fill(0.)
    a['D_yw'].fill(0.)
    rng = np.random.default_rng(91)
    inputs, initial = rng.normal(size=(2, 104, 3)), rng.normal(size=(2, 3))
    target, states = oracle(a, inputs, initial)
    yn = np.concatenate([np.ones((2, 1, 3))*99, target[:, :99]], axis=1)
    seed, sd = context.linear_seed(a, yn, inputs[:, :99])
    np.testing.assert_allclose(seed, initial, rtol=1e-12, atol=1e-12)
    assert sd['time'] == 'second observed output'
    yc, uc = yn*a['y_std']+a['y_mean'], inputs[:, :99]*a['u_std']+a['u_mean']
    actual, diag = context.condition(a, yc, uc)
    expected, _ = linear_context.condition(a['A'], a['B_u'], a['C_y'], a['D_yu'], yn, inputs[:, :99])
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(actual, states[:, 99], rtol=1e-12, atol=1e-12)
    assert all(r['accepted_steps'] == 0 and r['status'] == 'GRADIENT_TOL' for r in diag['requests'])


def test_rank_deficient_seed_minimum_norm_without_repair():
    a, yc, uc, _, _, _, _ = fixture(nx=28, batch=1)
    state, diag = context.condition(a, yc, uc)
    assert diag['requests'][0]['linear_rank'] == 3
    np.testing.assert_array_equal(diag['linear_seed_states'][:, 3:], 0.)
    np.testing.assert_array_equal(diag['solved_context_start_states'][:, 3:], 0.)
    np.testing.assert_array_equal(state[:, 3:], 0.)


def test_batch_solo_identity_and_no_input_or_model_mutation():
    a, yc, uc, fu, _, _, _ = fixture(batch=3)
    before = {k: v.copy() for k, v in a.items()}
    history = [v.copy() for v in (yc, uc, fu)]
    whole, final, diag = context.predict(a, yc, uc, fu)
    for i in range(3):
        one, end, d = context.predict(a, yc[i:i+1], uc[i:i+1], fu[i:i+1])
        np.testing.assert_allclose(one, whole[i:i+1], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(end, final[i:i+1], rtol=1e-13, atol=1e-13)
        left, right = d['requests'][0], diag['requests'][i]
        np.testing.assert_array_equal(left['final_jacobian_singular_values'], right['final_jacobian_singular_values'])
        assert {k: v for k, v in left.items() if k != 'final_jacobian_singular_values'} == {k: v for k, v in right.items() if k != 'final_jacobian_singular_values'}
    for key in a:
        np.testing.assert_array_equal(a[key], before[key])
    for actual, old in zip((yc, uc, fu), history, strict=True):
        np.testing.assert_array_equal(actual, old)
    assert not np.shares_memory(whole, yc) and not np.shares_memory(final, uc)


def test_causal_suffix_dropped_first_output_and_empty_forecast():
    a, yc, uc, fu, _, _, _ = fixture(batch=1)
    p, final, d = context.predict(a, yc, uc, fu)
    yc[:, 0] += 1e6
    fu[:, 13:] += 5
    changed, _, cd = context.predict(a, yc, uc, fu)
    np.testing.assert_array_equal(p[:, :13], changed[:, :13])
    np.testing.assert_array_equal(d['solved_context_start_states'], cd['solved_context_start_states'])
    assert not np.array_equal(p[:, 13], changed[:, 13])
    empty, state, _ = context.predict(a, yc, uc, fu[:, :0])
    expected, _ = context.condition(a, yc, uc)
    assert empty.shape == (1, 0, 3)
    np.testing.assert_array_equal(state, expected)
    assert final.shape == state.shape
    with pytest.raises(TypeError):
        context.condition(a, yc, uc, future_u=fu)
    with pytest.raises(TypeError):
        context.predict(a, yc, uc, fu, target=p)


@pytest.mark.parametrize('argument', ['y', 'u'])
@pytest.mark.parametrize('mutation', ['float32', 'nonfinite', 'rank', 'channels', 'batch', 'length'])
def test_strict_physical_context_guards(argument, mutation):
    a, yc, uc, _, _, _, _ = fixture(batch=2)
    x = yc if argument == 'y' else uc
    if mutation == 'float32':
        x = x.astype(np.float32)
    elif mutation == 'nonfinite':
        x[0, 0, 0] = np.inf
    elif mutation == 'rank':
        x = x[0]
    elif mutation == 'channels':
        x = x[:, :, :2]
    elif mutation == 'batch':
        x = x[:1]
    else:
        x = x[:, :1]
    with pytest.raises(ValueError):
        context.condition(a, x if argument == 'y' else yc, x if argument == 'u' else uc)


def test_nonfinite_linear_seed_system_and_factorization_fail_without_fallback(monkeypatch):
    a, yc, uc, _, _, _, _ = fixture(batch=1)
    a['A'][:] = 1e308
    with pytest.raises(ValueError, match='nonfinite linear seed'):
        context.condition(a, yc, uc)
    a = model()
    error = np.linalg.LinAlgError('fabricated factorization failure')
    def broken(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(context.np.linalg, 'lstsq', broken)
    with pytest.raises(np.linalg.LinAlgError) as observed:
        context.condition(a, yc, uc)
    assert observed.value is error


def fake_rollout(kind, calls):
    def run(_arrays, inputs, state, *, jacobian=False):
        calls.append((jacobian, state.copy()))
        n = inputs.shape[1]
        x = float(state[0, 0])
        if kind == 'nonfinite_current':
            raise FloatingPointError('current invalid')
        if kind == 'reject_first' and not jacobian and x < -.75:
            raise FloatingPointError('trial invalid')
        value = 1. if kind in ('stall', 'zero') else 1.+x
        prediction = np.full((1, n, 3), value)
        derivative = np.full((1, n, 3, 1), 0. if kind == 'zero' else 1.) if jacobian else None
        return prediction, state.copy(), derivative
    return run


def test_zero_jacobian_explicit_status_and_finite_seed_return(monkeypatch):
    calls = []
    monkeypatch.setattr(context, 'rollout', fake_rollout('zero', calls))
    final, state, record = context._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[2.]]))
    assert record['status'] == 'ZERO_JACOBIAN' and record['accepted_steps'] == 0
    assert record['initial_objective'] == record['final_objective']
    assert record['final_objective'] == pytest.approx(.5, rel=0., abs=1e-15)
    np.testing.assert_array_equal(final, [[2.]])
    np.testing.assert_array_equal(state, [[2.]])
    assert len(calls) == record['trajectory_evaluations'] == 2


def test_exhausted_trials_stall_not_silent_new_initialization(monkeypatch):
    calls = []
    monkeypatch.setattr(context, 'rollout', fake_rollout('stall', calls))
    final, _, record = context._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[0.]]))
    assert record['status'] == 'STALLED' and record['accepted_steps'] == 0
    assert len(record['trace']) == 1 and len(record['trace'][0]['trials']) == 8
    assert record['trajectory_evaluations'] == len(calls) == 10
    assert all(not t['accepted'] and not t['nonfinite'] for t in record['trace'][0]['trials'])
    np.testing.assert_array_equal(final, [[0.]])


def test_nonfinite_trial_counts_then_half_step_accepts_and_cap_is_visible(monkeypatch):
    calls = []
    monkeypatch.setattr(context, 'ITERATIONS', 1)
    monkeypatch.setattr(context, 'rollout', fake_rollout('reject_first', calls))
    final, _, record = context._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[0.]]))
    assert record['status'] == 'ITERATION_CAP' and record['accepted_steps'] == 1
    trials = record['trace'][0]['trials']
    assert len(trials) == 2 and trials[0]['nonfinite'] and trials[1]['accepted']
    assert trials[1]['alpha'] == .5 and record['final_objective'] < record['initial_objective']
    assert record['trajectory_evaluations'] == len(calls) == 4
    np.testing.assert_allclose(final, [[-.5/1.001]], rtol=1e-14, atol=1e-14)


def test_nonnegative_direction_stalls_without_trials(monkeypatch):
    calls = []
    monkeypatch.setattr(context, 'rollout', fake_rollout('linear', calls))
    monkeypatch.setattr(context.np.linalg, 'lstsq', lambda *_a, **_k: (np.array([1.]), None, 1, None))
    _, _, record = context._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[0.]]))
    assert record['status'] == 'STALLED' and record['accepted_steps'] == 0
    assert record['trace'][0]['trials'] == [] and len(calls) == 2


def test_nonfinite_current_trajectory_is_not_a_rejected_trial(monkeypatch):
    calls = []
    monkeypatch.setattr(context, 'rollout', fake_rollout('nonfinite_current', calls))
    with pytest.raises(FloatingPointError, match='current invalid'):
        context._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[0.]]))
    assert len(calls) == 1
