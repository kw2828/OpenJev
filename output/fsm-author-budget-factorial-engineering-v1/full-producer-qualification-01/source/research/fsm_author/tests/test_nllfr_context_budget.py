# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated-only parity, causal boundaries and fixed GN work budgets."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'src'))
sys.path.insert(0, str(HERE))
from test_nllfr_context import fixture  # noqa: E402

from openjev_fsm_author import nllfr_context as original  # noqa: E402
from openjev_fsm_author import nllfr_context_budget as budget  # noqa: E402


def exact(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes(order='C') == expected.tobytes(order='C')
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            exact(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            exact(left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize('nx,batch,length,horizon', [(3, 1, 2, 7), (3, 3, 17, 32),
                                                  (3, 2, 100, 128), (28, 1, 100, 128)])
def test_fixed16_bitwise_entire_return_matches_qualified_original(nx, batch, length, horizon):
    a, y, u, future, *_ = fixture(nx, batch, length, horizon)
    # Measurement inconsistency also exercises nonzero residual stopping paths.
    y[:, 1:, 0] += np.linspace(-.002, .003, length-1)
    exact(budget.predict(a, y, u, future, iterations=16), original.predict(a, y, u, future))
    exact(budget.condition(a, y, u), original.condition(a, y, u))
    assert original.ITERATIONS == 16


@pytest.mark.parametrize('value', [True, False, 16., 64., np.int64(16), None, 0, 1, 15, 17, 65, '16'])
def test_only_exact_int16_or64_admitted_before_numeric_inputs(value):
    with pytest.raises(ValueError, match='fixed context budget'):
        budget.condition(None, None, None, iterations=value)


def exponential_rollout(calls, *, reject_trials=False, fail_current=False):
    """Smooth scalar exp(x) surrogate with its exact derivative, for loop work."""
    def call(_arrays, inputs, state, *, jacobian=False):
        calls.append((jacobian, state.copy()))
        if fail_current:
            raise FloatingPointError('fabricated current failure')
        if reject_trials and not jacobian:
            raise FloatingPointError('fabricated trial failure')
        value = float(np.exp(state[0, 0]))
        shape = (1, inputs.shape[1], 3)
        predicted = np.full(shape, value)
        derivative = np.full((*shape, 1), value) if jacobian else None
        return predicted, state.copy(), derivative
    return call


def test64_cap_exact_first16_directions_trials_and_state_boundary(monkeypatch):
    inputs, target, seed = np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), np.array([[100.]])
    calls16, calls64 = [], []
    monkeypatch.setattr(budget, 'rollout', exponential_rollout(calls16))
    short = budget._solve_one({}, target, inputs, seed, iterations=16)
    monkeypatch.setattr(budget, 'rollout', exponential_rollout(calls64))
    long = budget._solve_one({}, target, inputs, seed, iterations=64)
    d16, d64 = short[2], long[2]
    assert d16['status'] == d64['status'] == 'ITERATION_CAP'
    assert d16['directions_considered'] == 16 and d64['directions_considered'] == 64
    exact(d64['trace'][:16], d16['trace'])
    # Final16 diagnostics and direction17 start use the same retained state.
    jac16, jac64 = ([state for jac, state in calls if jac] for calls in (calls16, calls64))
    exact(jac16, jac64[:17])
    exact(short[1], jac64[16])
    trial16, trial64 = ([state for jac, state in calls if not jac] for calls in (calls16, calls64))
    exact(trial16, trial64[:16])
    assert d64['jacobian_evaluations'] == 65
    assert d64['trial_attempts'] == d64['accepted_steps'] == 64
    assert d64['trajectory_evaluations'] == len(calls64) == 129
    assert d64['trial_attempts'] <= 64*8
    assert d64['final_objective'] <= d16['final_objective']
    exact(seed, np.array([[100.]]))
    assert original.ITERATIONS == 16


def test_early_stop_returns_same_full_state_prediction_and_work_for_both_budgets():
    a, y, u, future, *_ = fixture(batch=2, length=17, horizon=32)
    short = budget.predict(a, y, u, future, iterations=16)
    long = budget.predict(a, y, u, future, iterations=64)
    assert all(r['directions_considered'] < 16 and r['status'] != 'ITERATION_CAP' for r in short[2]['requests'])
    assert long[2]['policy']['iterations'] == 64 and short[2]['policy']['iterations'] == 16
    diag = copy.deepcopy(long[2])
    diag['policy']['iterations'] = 16
    exact((long[0], long[1], diag), short)


def test64_full_rejected_line_search_is_bounded_and_preserves_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(budget, 'rollout', exponential_rollout(calls, reject_trials=True))
    state, solved, diag = budget._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[1.]]), iterations=64)
    assert diag['status'] == 'STALLED' and diag['accepted_steps'] == 0
    assert diag['directions_considered'] == 1 and diag['jacobian_evaluations'] == 2
    assert diag['trajectory_evaluations'] == len(calls) == 10 and diag['trial_attempts'] == 8
    assert [row['alpha'] for row in diag['trace'][0]['trials']] == [2.**(-i) for i in range(8)]
    assert all(row['nonfinite'] and not row['accepted'] for row in diag['trace'][0]['trials'])
    exact(state, np.array([[1.]]))
    exact(solved, state)


def test_bad_current_and_programming_error_propagate_without_repair(monkeypatch):
    monkeypatch.setattr(budget, 'rollout', exponential_rollout([], fail_current=True))
    with pytest.raises(FloatingPointError, match='current failure'):
        budget._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[1.]]), iterations=64)
    error = ValueError('fabricated programmer error')
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(budget, 'rollout', broken)
    with pytest.raises(ValueError) as observed:
        budget._solve_one({}, np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[1.]]), iterations=64)
    assert observed.value is error


@pytest.mark.parametrize('iterations', [16, 64])
def test_target_isolation_current_input_boundary_causal_suffix_and_empty_future(iterations):
    a, y, u, future, target, states, _ = fixture(batch=1, length=100, horizon=128)
    predicted, final, diag = budget.predict(a, y, u, future, iterations=iterations)
    np.testing.assert_allclose(predicted, target, rtol=1e-6, atol=1e-6)
    forecast, _ = budget.condition(a, y, u, iterations=iterations)
    np.testing.assert_allclose(forecast, states[:, 99], rtol=1e-6, atol=1e-6)
    changed_y, changed_future = y.copy(), future.copy()
    changed_y[:, 0] += 1e6  # The state is at the second observation, as qualified.
    changed_future[:, 13:] += 5
    changed, _, changed_diag = budget.predict(a, changed_y, u, changed_future, iterations=iterations)
    exact(diag, changed_diag)
    exact(predicted[:, :13], changed[:, :13])
    assert not np.array_equal(predicted[:, 13], changed[:, 13])
    empty, state, _ = budget.predict(a, y, u, future[:, :0], iterations=iterations)
    assert empty.shape == (1, 0, 3)
    exact(state, forecast)
    assert final.shape == state.shape
    with pytest.raises(TypeError):
        budget.condition(a, y, u, future_u=future, iterations=iterations)
    with pytest.raises(TypeError):
        budget.predict(a, y, u, future, target=target, iterations=iterations)


def test64_batch_independence_ownership_and_no_cache():
    a, y, u, future, *_ = fixture(batch=3, length=17, horizon=32)
    saved = copy.deepcopy((a, y, u, future))
    predicted, final, diag = budget.predict(a, y, u, future, iterations=64)
    for i in range(3):
        one, end, detail = budget.predict(a, y[i:i+1], u[i:i+1], future[i:i+1], iterations=64)
        # Batch-size-dependent GEMM reduction uses the held original tolerance.
        np.testing.assert_allclose(one, predicted[i:i+1], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(end, final[i:i+1], rtol=1e-13, atol=1e-13)
        exact(detail['requests'][0], diag['requests'][i])
    exact((a, y, u, future), saved)
    assert not np.shares_memory(predicted, y) and not np.shares_memory(final, u)
    predicted[:] = 100.
    exact((a, y, u, future), saved)
    exact(budget.predict(a, y, u, future, iterations=64)[1], final)


@pytest.mark.parametrize('change', ['y_float32', 'u_nonfinite', 'u_length', 'y_channels', 'zero_batch', 'bad_model'])
def test64_strict_shapes_dtypes_and_finite_guards(change):
    a, y, u, *_ = fixture(batch=1, length=17, horizon=32)
    if change == 'y_float32':
        y = y.astype(np.float32)
    elif change == 'u_nonfinite':
        u[0, 0, 0] = np.inf
    elif change == 'u_length':
        u = u[:, :-1]
    elif change == 'y_channels':
        y = y[:, :, :2]
    elif change == 'zero_batch':
        y, u = y[:0], u[:0]
    else:
        a['W0'][0, 0] = np.nan
    with pytest.raises(ValueError):
        budget.condition(a, y, u, iterations=64)
