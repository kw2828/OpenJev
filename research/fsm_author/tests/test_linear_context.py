# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated state-space equations only; no author code, weights or data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'src/openjev_fsm_author/linear_context.py'
SPEC = importlib.util.spec_from_file_location('linear_context', SOURCE)
linear = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linear)


def system(n=4):
    A = np.diag(np.linspace(.3, .85, n))
    B = np.arange(n*3, dtype=np.float64).reshape(n, 3)/40-.1
    C = np.eye(3, n)
    if n == 4:
        C[:, 3] = [.5, -.3, .7]
    D = np.array([[.2, -.1, .05], [0., .3, .1], [-.1, .02, -.2]])
    return A, B, C, D


def oracle(A, B, C, D, inputs, initial):
    """Independent scalar batch/time loop, storing every pre/post state."""
    batch, horizon, _ = inputs.shape
    predictions = np.empty((batch, horizon, 3))
    states = np.empty((batch, horizon+1, len(A)))
    states[:, 0] = initial
    for b in range(batch):
        for t in range(horizon):
            predictions[b, t] = C @ states[b, t] + D @ inputs[b, t]
            states[b, t+1] = A @ states[b, t] + B @ inputs[b, t]
    return predictions, states


def request(*, context=100, horizon=128, batch=3, n=4):
    A, B, C, D = system(n)
    rng = np.random.default_rng(712)
    inputs = rng.normal(size=(batch, context-1+horizon, 3))
    initial = rng.normal(size=(batch, n))
    outputs, states = oracle(A, B, C, D, inputs, initial)
    yc = np.concatenate((np.full((batch, 1, 3), 123.), outputs[:, :context-1]), axis=1)
    return (A, B, C, D), yc, inputs[:, :context-1], inputs[:, context-1:], outputs[:, context-1:], states


@pytest.mark.parametrize('context,n', [(2, 3), (3, 4), (100, 4)])
def test_known_hidden_state_and_full_forecast_recovery(context, n):
    matrices, yc, uc, future, expected, states = request(context=context, n=n)
    state, diagnostics = linear.condition(*matrices, yc, uc)
    prediction, final, combined = linear.predict(*matrices, yc, uc, future)
    np.testing.assert_allclose(state, states[:, context-1], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(prediction, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(final, states[:, -1], rtol=1e-11, atol=1e-12)
    assert diagnostics['rank'] == n and diagnostics['rank_deficient'] is False
    assert diagnostics['context_pairs'] == context-1
    assert diagnostics['equations'] == 3*(context-1)
    assert diagnostics['state_dimension'] == n and diagnostics['rcond'] == 1e-12
    assert diagnostics['residual_norm'].shape == (3,)
    assert max(diagnostics['residual_norm']) < 1e-11
    np.testing.assert_array_equal(combined['singular_values'], diagnostics['singular_values'])


def test_direct_feedthrough_and_output_before_advance_hand_oracle():
    A, B = np.array([[.5]]), np.array([[2., 0., 0.]])
    C, D = np.array([[1.], [2.], [-1.]]), np.diag([1., 3., 4.])
    yc = np.array([[[999., 999., 999.], [5., 16., 18.]]])
    uc = np.array([[[3., 4., 5.]]])
    future = np.array([[[11., 13., 17.], [0., 0., 0.]]])
    state, _ = linear.condition(A, B, C, D, yc, uc)
    np.testing.assert_allclose(state, [[7.]], atol=1e-13)
    prediction, final = linear.rollout(A, B, C, D, future, state)
    np.testing.assert_allclose(prediction, [[[18., 53., 61.], [25.5, 51., -25.5]]], atol=1e-13)
    np.testing.assert_allclose(final, [[12.75]], atol=1e-13)


def test_unknown_first_input_is_not_synthesized_or_reused():
    # Independent full chronology: x0=1 and native inputs7,11,13,17 yield
    # y0=38,y1=82,y2=152,y3=298. Only input11,input13 accompany context.
    A, B = np.array([[2.]]), np.array([[1., 0., 0.]])
    C, D = np.array([[3.], [0.], [0.]]), np.diag([5., 0., 0.])
    yc = np.array([[[38., 0., 0.], [82., 0., 0.], [152., 0., 0.]]])
    uc = np.array([[[11., 0., 0.], [13., 0., 0.]]])
    state, _ = linear.condition(A, B, C, D, yc, uc)
    np.testing.assert_allclose(state, [[71.]], rtol=0, atol=1e-12)
    value, final = linear.step(A, B, C, D, state, np.array([[17., 0., 0.]]))
    np.testing.assert_allclose(value, [[298., 0., 0.]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(final, [[159.]], rtol=0, atol=1e-12)


def test_first_observed_output_is_unused_but_last_pair_is_used():
    matrices, yc, uc, future, _, _ = request(context=5)
    original, diagnostic = linear.condition(*matrices, yc, uc)
    changed = yc.copy()
    changed[:, 0] = [-1e7, 2e7, 3e7]
    revised, revised_diagnostic = linear.condition(*matrices, changed, uc)
    np.testing.assert_array_equal(original, revised)
    np.testing.assert_array_equal(diagnostic['residual_norm'], revised_diagnostic['residual_norm'])
    changed[:, -1] += 1.
    revised, _ = linear.condition(*matrices, changed, uc)
    assert not np.allclose(original, revised)
    changed_input = uc.copy()
    changed_input[:, -1] += 1.
    revised, _ = linear.condition(*matrices, yc, changed_input)
    assert not np.allclose(original, revised)
    first, _, _ = linear.predict(*matrices, yc, uc, future)
    altered = future.copy()
    altered[:, 7:] += 100.
    second, _, _ = linear.predict(*matrices, yc, uc, altered)
    np.testing.assert_array_equal(first[:, :7], second[:, :7])
    assert not np.array_equal(first[:, 7], second[:, 7])


def test_rank_deficient_minimum_norm_is_declared_without_jitter():
    A, B = np.eye(2), np.zeros((2, 3))
    C, D = np.array([[1., 0.], [2., 0.], [0., 0.]]), np.eye(3)
    uc = np.arange(18, dtype=np.float64).reshape(2, 3, 3)/10
    yc = np.zeros((2, 4, 3))
    yc[:, 1:] = uc + np.array([3., 6., 0.])
    state, diagnostics = linear.condition(A, B, C, D, yc, uc)
    np.testing.assert_allclose(state, [[3., 0.], [3., 0.]], atol=1e-13)
    assert diagnostics['rank'] == 1 and diagnostics['rank_deficient'] is True
    assert diagnostics['singular_values'][1] == 0.
    assert 'minimum Euclidean norm' in diagnostics['solution_convention']
    # An unobservable state coordinate is not guessed from unavailable history.
    np.testing.assert_array_equal(state[:, 1], np.zeros(2))


def test_short_context_nullspace_can_matter_later_without_recovery_claim():
    A, B = np.array([[0., 1.], [0., 0.]]), np.zeros((2, 3))
    C, D = np.array([[1., 0.], [0., 0.], [0., 0.]]), np.zeros((3, 3))
    yc = np.array([[[0., 0., 0.], [2., 0., 0.]]])
    state, diagnostic = linear.condition(A, B, C, D, yc, np.zeros((1, 1, 3)))
    assert diagnostic['rank'] == 1
    np.testing.assert_array_equal(state, np.zeros((1, 2)))
    # x at observation1 could also have been [2,7], producing forecast7.
    # Minimum norm chooses [2,0]; the finite context cannot distinguish them.


def test_fixed_rcond_excludes_tiny_observable_direction():
    A, B, C, D = np.eye(3), np.zeros((3, 3)), np.diag([1., 1e-13, 1.]), np.zeros((3, 3))
    yc = np.array([[[0., 0., 0.], [1., 2e-13, 3.]]])
    state, diagnostic = linear.condition(A, B, C, D, yc, np.zeros((1, 1, 3)))
    np.testing.assert_allclose(state, [[1., 0., 3.]], rtol=0, atol=1e-14)
    assert diagnostic['rank'] == 2 and diagnostic['rcond'] == 1e-12


def test_inconsistent_context_least_squares_and_residual_diagnostic():
    A, B, C, D = np.ones((1, 1)), np.zeros((1, 3)), np.array([[1.], [0.], [0.]]), np.zeros((3, 3))
    yc = np.array([[[9., 9., 9.], [0., 0., 0.], [2., 0., 0.], [4., 0., 0.]]])
    state, diagnostic = linear.condition(A, B, C, D, yc, np.zeros((1, 3, 3)))
    np.testing.assert_allclose(state, [[2.]], atol=1e-14)
    np.testing.assert_allclose(diagnostic['residual_norm'], [np.sqrt(8.)], atol=1e-14)


def test_full_rank_similarity_changes_coordinates_but_not_forecast():
    matrices, yc, uc, future, expected, _ = request(context=8)
    A, B, C, D = matrices
    T = np.array([[2., .2, 0., 0.], [0., .5, 0., 0.], [0., 0., 1.5, .1], [0., 0., 0., .8]])
    inverse = np.linalg.inv(T)
    actual, _, diagnostics = linear.predict(T@A@inverse, T@B, C@inverse, D, yc, uc, future)
    assert diagnostics['rank'] == 4
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_batch_segmented_step_and_empty_horizon_equivalence():
    matrices, yc, uc, future, _, _ = request(context=9)
    state, _ = linear.condition(*matrices, yc, uc)
    whole, end = linear.rollout(*matrices, future, state)
    first, middle = linear.rollout(*matrices, future[:, :11], state)
    second, final = linear.rollout(*matrices, future[:, 11:], middle)
    np.testing.assert_array_equal(np.concatenate((first, second), axis=1), whole)
    np.testing.assert_array_equal(final, end)
    one, next_state = linear.step(*matrices, state, future[:, 0])
    np.testing.assert_array_equal(one, whole[:, 0])
    _, reference_next = linear.rollout(*matrices, future[:, :1], state)
    np.testing.assert_array_equal(next_state, reference_next)
    for b in range(len(yc)):
        prediction, final, _ = linear.predict(*matrices, yc[b:b+1], uc[b:b+1], future[b:b+1])
        np.testing.assert_allclose(prediction[0], whole[b], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(final[0], end[b], rtol=1e-11, atol=1e-12)
    empty, copied = linear.rollout(*matrices, future[:, :0], state)
    assert empty.shape == (3, 0, 3) and empty.dtype == np.float64
    np.testing.assert_array_equal(copied, state)
    assert not np.shares_memory(copied, state)


def test_readonly_inputs_owned_outputs_no_hidden_state_and_rng_isolation():
    matrices, yc, uc, future, _, _ = request(context=5)
    inputs = (*matrices, yc, uc, future)
    originals = [a.copy() for a in inputs]
    for a in inputs:
        a.setflags(write=False)
    before = np.random.get_state()
    pred, final, diagnostic = linear.predict(*matrices, yc, uc, future)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    for actual, original in zip(inputs, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
        assert not np.shares_memory(pred, actual) and not np.shares_memory(final, actual)
    original_pred = pred.copy()
    pred.fill(999.)
    final.fill(999.)
    diagnostic['singular_values'].fill(999.)
    fresh, _, _ = linear.predict(*matrices, yc, uc, future)
    np.testing.assert_array_equal(fresh, original_pred)


@pytest.mark.parametrize('index', range(7))
@pytest.mark.parametrize('bad', ['float32', 'nonfinite', 'wrong_shape', 'list'])
def test_matrix_context_future_contract(index, bad):
    matrices, yc, uc, future, _, _ = request(context=5)
    args = [a.copy() for a in (*matrices, yc, uc, future)]
    if bad == 'float32':
        args[index] = args[index].astype(np.float32)
    elif bad == 'nonfinite':
        args[index].flat[0] = np.nan
    elif bad == 'wrong_shape':
        args[index] = args[index][:-1]
    else:
        args[index] = args[index].tolist()
    with pytest.raises(ValueError):
        linear.predict(*args)


@pytest.mark.parametrize('bad', ['one_output', 'empty_batch', 'extra_context_input'])
def test_context_boundary_geometry(bad):
    matrices, yc, uc, _, _, _ = request(context=5)
    if bad == 'one_output':
        yc, uc = yc[:, :1], uc[:, :0]
    elif bad == 'empty_batch':
        yc, uc = yc[:0], uc[:0]
    else:
        uc = np.concatenate((uc, uc[:, :1]), axis=1)
    with pytest.raises(ValueError):
        linear.condition(*matrices, yc, uc)


@pytest.mark.parametrize('bad', ['rank', 'empty', 'dtype', 'nan', 'width', 'input'])
def test_step_and_rollout_reject_invalid_caller_state(bad):
    matrices = system()
    state, u = np.ones((2, 4)), np.ones((2, 3))
    if bad == 'rank':
        state = state[0]
    elif bad == 'empty':
        state, u = state[:0], u[:0]
    elif bad == 'dtype':
        state = state.astype(np.float32)
    elif bad == 'nan':
        state[0, 0] = np.inf
    elif bad == 'width':
        state = state[:, :3]
    else:
        u = u.astype(np.float32)
    with pytest.raises(ValueError):
        linear.step(*matrices, state, u)
    with pytest.raises(ValueError):
        linear.rollout(*matrices, u[:, None], state)


def test_overflow_fails_without_repair():
    A, B, C, D = np.array([[1e308]]), np.zeros((1, 3)), np.ones((3, 1)), np.zeros((3, 3))
    with pytest.raises(ValueError, match='nonfinite'):
        linear.condition(A, B, C, D, np.ones((1, 4, 3)), np.zeros((1, 3, 3)))
    with pytest.raises(ValueError, match='nonfinite'):
        linear.step(A, B, C, D, np.array([[2.]]), np.zeros((1, 3)))
    with pytest.raises(ValueError, match='nonfinite'):
        linear.rollout(A, B, C, D, np.zeros((1, 2, 3)), np.array([[2.]]))


def test_nonfinite_solver_output_rejected_without_second_solver_call(monkeypatch):
    matrices, yc, uc, _, _, _ = request(context=5)
    calls = []
    def invalid_solver(matrix, rhs, *, rcond):
        calls.append(rcond)
        return np.full((4, 3), np.nan), np.empty(0), 4, np.ones(4)
    monkeypatch.setattr(linear.np.linalg, 'lstsq', invalid_solver)
    with pytest.raises(ValueError, match='nonfinite'):
        linear.condition(*matrices, yc, uc)
    assert calls == [1e-12]
