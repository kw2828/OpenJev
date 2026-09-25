# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated NL-LFR equations and derivatives; no measurements/checkpoints."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from openjev_fsm_author import linear_context, nllfr  # noqa: E402


def arrays():
    return {
        'A': np.array([[.42, .06, -.02], [0., .55, .04], [.03, 0., .67]]),
        'B_u': np.array([[.2, -.1, .03], [.04, .3, -.02], [-.1, .03, .15]]),
        'C_y': np.array([[1., .2, -.1], [.1, .7, .3], [-.2, .1, .8]]),
        'D_yu': np.array([[.2, -.03, .01], [0., -.1, .04], [.05, .02, .15]]),
        'B_w': np.array([[.02, -.01], [.01, .015], [-.02, .01]]),
        'C_z': np.array([[.3, -.2, .1], [-.1, .2, .4]]),
        'D_yw': np.array([[.1, -.04], [-.03, .08], [.02, .06]]),
        'D_zu': np.array([[.2, -.1, .05], [-.04, .15, .1]]),
        'W0': np.array([[.4, -.2], [-.3, .5], [.2, .1]]),
        'b0': np.array([.7, -.6, .8]),
        'W1': np.array([[.2, -.3, .1], [-.4, .2, .3], [.1, .3, .2]]),
        'b1': np.array([.8, -.7, .6]),
        'W2': np.array([[.3, -.2, .1], [-.1, .4, -.2]]),
        'b2': np.array([-.2, .1]),
        'u_mean': np.array([2., -3., .5]), 'u_std': np.array([.5, 2., 3.]),
        'y_mean': np.array([-4., 1., 7.]), 'y_std': np.array([3., .25, 2.]),
        'ts': np.array(1/6400., dtype=np.float64),
    }


def oracle(a, inputs, initial):
    """Separate column-vector, batch/time simulator with all pre/post states."""
    batch, horizon, _ = inputs.shape
    states = np.empty((batch, horizon+1, initial.shape[1]), dtype=np.float64)
    outputs = np.empty((batch, horizon, 3), dtype=np.float64)
    states[:, 0] = initial
    for b in range(batch):
        for t in range(horizon):
            x, u = states[b, t], inputs[b, t]
            z = a['C_z'] @ x+a['D_zu'] @ u
            h = np.array([max(0., value) for value in a['W0'] @ z+a['b0']])
            h = np.array([max(0., value) for value in a['W1'] @ h+a['b1']])
            w = a['W2'] @ h+a['b2']
            outputs[b, t] = a['C_y'] @ x+a['D_yu'] @ u+a['D_yw'] @ w
            states[b, t+1] = a['A'] @ x+a['B_u'] @ u+a['B_w'] @ w
    return outputs, states


def scalar_arrays():
    a = arrays()
    a.update(A=np.array([[.5]]), B_u=np.array([[2., 0., 0.]]),
             C_y=np.array([[1.], [2.], [-1.]]), D_yu=np.diag([1., 3., 4.]),
             B_w=np.array([[.25]]), C_z=np.array([[1.]]),
             D_yw=np.array([[1.], [-2.], [.5]]), D_zu=np.array([[.5, 0., 0.]]),
             W0=np.array([[2.]]), b0=np.array([-1.]),
             W1=np.array([[.5]]), b1=np.array([.5]),
             W2=np.array([[3.]]), b2=np.array([-2.]))
    return a


def test_hand_nonzero_feedback_feedthrough_and_later_jacobian():
    a = scalar_arrays()
    inputs = np.array([[[3., 4., 5.], [0., 0., 0.]]])
    result, final, derivative = nllfr.rollout(a, inputs, np.array([[2.]]), jacobian=True)
    np.testing.assert_array_equal(result, [[[13.5, -1., 22.25], [34.5, -32.5, 3.5625]]])
    np.testing.assert_array_equal(final, [[10.90625]])
    np.testing.assert_array_equal(derivative, [[[[4.], [-4.], [.5]], [[5.], [-5.], [.625]]]])


@pytest.mark.parametrize('batch,horizon', [(1, 1), (3, 7), (1, 128), (3, 128)])
def test_independent_column_simulator_and_physical_units(batch, horizon):
    a = arrays()
    rng = np.random.default_rng(703)
    inputs = rng.normal(scale=.3, size=(batch, horizon, 3))
    initial = rng.normal(scale=.2, size=(batch, 3))
    expected, states = oracle(a, inputs, initial)
    actual, final, derivative = nllfr.rollout(a, inputs, initial)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(final, states[:, -1], rtol=1e-13, atol=1e-14)
    assert derivative is None
    physical, physical_final = nllfr.physical_rollout(a, inputs*a['u_std']+a['u_mean'], initial)
    np.testing.assert_allclose(physical, expected*a['y_std']+a['y_mean'], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(physical_final, final, rtol=1e-13, atol=1e-14)


def test_zero_feedback_reduces_to_qualified_linear_model():
    a = arrays()
    a['B_w'].fill(0.)
    a['D_yw'].fill(0.)
    rng = np.random.default_rng(9)
    inputs, initial = rng.normal(size=(2, 128, 3)), rng.normal(size=(2, 3))
    value, final, jac = nllfr.rollout(a, inputs, initial, jacobian=True)
    expected, expected_final = linear_context.rollout(
        a['A'], a['B_u'], a['C_y'], a['D_yu'], inputs, initial)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(final, expected_final)
    for time in (0, 1, 31, 127):
        block = a['C_y'] @ np.linalg.matrix_power(a['A'], time)
        np.testing.assert_allclose(jac[:, time], np.broadcast_to(block, (2, 3, 3)),
                                   rtol=1e-12, atol=1e-15)


def test_initial_state_jacobian_central_differences_away_from_kinks():
    a = arrays()
    rng = np.random.default_rng(79)
    inputs, initial = rng.normal(scale=.03, size=(2, 17, 3)), rng.normal(scale=.03, size=(2, 3))
    value, final, jac = nllfr.rollout(a, inputs, initial, jacobian=True)
    plain, plain_final, _ = nllfr.rollout(a, inputs, initial)
    np.testing.assert_array_equal(value, plain)
    np.testing.assert_array_equal(final, plain_final)
    epsilon = 1e-6
    for coordinate in range(3):
        plus, minus = initial.copy(), initial.copy()
        plus[:, coordinate] += epsilon
        minus[:, coordinate] -= epsilon
        yp = oracle(a, inputs, plus)[0]
        ym = oracle(a, inputs, minus)[0]
        np.testing.assert_allclose(jac[..., coordinate], (yp-ym)/(2*epsilon),
                                   rtol=2e-6, atol=1e-10)


def test_jax_autodiff_matches_selected_relu_zero_and_later_steps():
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        a = scalar_arrays()
        a['A'][:] = .5
        a['D_zu'][:] = 0.
        a['W0'][:] = 1.
        a['b0'][:] = 0.
        a['W1'][:] = 1.
        a['b1'][:] = 0.
        a['W2'][:] = 1.
        a['b2'][:] = 0.
        inputs = np.array([[[1., 0., 0.], [-.2, 0., 0.], [.1, 0., 0.]]])
        initial = np.array([[0.]])
        params = {key: jnp.asarray(value) for key, value in a.items()}

        def independent(x):
            output = []
            for u in jnp.asarray(inputs[0]):
                z = params['C_z'] @ x+params['D_zu'] @ u
                w = params['W2'] @ jax.nn.relu(params['W1'] @ jax.nn.relu(
                    params['W0'] @ z+params['b0'])+params['b1'])+params['b2']
                output.append(params['C_y'] @ x+params['D_yu'] @ u+params['D_yw'] @ w)
                x = params['A'] @ x+params['B_u'] @ u+params['B_w'] @ w
            return jnp.stack(output)

        value, _, jac = nllfr.rollout(a, inputs, initial, jacobian=True)
        np.testing.assert_allclose(value[0], np.asarray(independent(jnp.asarray(initial[0]))),
                                   rtol=1e-13, atol=1e-14)
        expected = np.asarray(jax.jacfwd(independent)(jnp.asarray(initial[0])))
        np.testing.assert_allclose(jac[0], expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(jac[0, 0], a['C_y'])


def test_causal_suffix_chunking_batch_independence_and_no_cache():
    a = arrays()
    rng = np.random.default_rng(13)
    inputs, state = rng.normal(size=(3, 12, 3)), rng.normal(size=(3, 3))
    original_a = {key: value.copy() for key, value in a.items()}
    original_inputs, original_state = inputs.copy(), state.copy()
    whole, final, jac = nllfr.rollout(a, inputs, state, jacobian=True)
    left, carry, _ = nllfr.rollout(a, inputs[:, :5], state)
    right, carry_final, _ = nllfr.rollout(a, inputs[:, 5:], carry)
    np.testing.assert_array_equal(whole, np.concatenate((left, right), axis=1))
    np.testing.assert_array_equal(final, carry_final)
    changed = inputs.copy()
    changed[:, 5:] += 19.
    mutated, _, _ = nllfr.rollout(a, changed, state)
    np.testing.assert_array_equal(whole[:, :5], mutated[:, :5])
    for b in range(3):
        single, sf, sj = nllfr.rollout(a, inputs[b:b+1], state[b:b+1], jacobian=True)
        np.testing.assert_allclose(single, whole[b:b+1], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(sf, final[b:b+1], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(sj, jac[b:b+1], rtol=1e-13, atol=1e-14)
    for key in a:
        np.testing.assert_array_equal(a[key], original_a[key])
    np.testing.assert_array_equal(inputs, original_inputs)
    np.testing.assert_array_equal(state, original_state)
    for output in (whole, final, jac):
        assert not np.shares_memory(output, inputs) and not np.shares_memory(output, state)
    empty, empty_state, empty_jac = nllfr.rollout(a, inputs[:, :0], state, jacobian=True)
    assert empty.shape == (3, 0, 3) and empty_jac.shape == (3, 0, 3, 3)
    np.testing.assert_array_equal(empty_state, state)
    assert not np.shares_memory(empty_state, state)


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'A_float32', 'W0_float32', 'nan',
                                     'inf', 'width', 'latent', 'bias', 'scale', 'zero_width'])
def test_model_roster_dtype_shape_and_finiteness(mutation):
    a = arrays()
    if mutation == 'extra':
        a['hidden_cache'] = np.zeros(1)
    elif mutation == 'missing':
        del a['b2']
    elif mutation in ('A_float32', 'W0_float32'):
        key = mutation.split('_')[0]
        a[key] = a[key].astype(np.float32)
    elif mutation in ('nan', 'inf'):
        a['W2'][0, 0] = np.nan if mutation == 'nan' else np.inf
    elif mutation == 'width':
        a['W1'] = np.zeros((2, 3))
    elif mutation == 'latent':
        a['D_yw'] = np.zeros((3, 3))
    elif mutation == 'bias':
        a['b0'] = a['b0'][None]
    elif mutation == 'scale':
        a['u_std'][0] = 0.
    else:
        a['W0'] = np.empty((0, 2))
    with pytest.raises(ValueError):
        nllfr.validate(a)


@pytest.mark.parametrize('mutation', ['empty_batch', 'input_rank', 'input_channels', 'input_float32',
                                     'input_nan', 'state_shape', 'state_float32', 'state_inf', 'flag'])
def test_rollout_input_guards(mutation):
    inputs, state, flag = np.zeros((1, 3, 3)), np.zeros((1, 3)), False
    if mutation == 'empty_batch':
        inputs, state = inputs[:0], state[:0]
    elif mutation == 'input_rank':
        inputs = inputs[0]
    elif mutation == 'input_channels':
        inputs = inputs[:, :, :2]
    elif mutation == 'input_float32':
        inputs = inputs.astype(np.float32)
    elif mutation == 'input_nan':
        inputs[0, 1, 0] = np.nan
    elif mutation == 'state_shape':
        state = state[:, :2]
    elif mutation == 'state_float32':
        state = state.astype(np.float32)
    elif mutation == 'state_inf':
        state[0, 0] = np.inf
    else:
        flag = 1
    with pytest.raises(ValueError):
        nllfr.rollout(arrays(), inputs, state, jacobian=flag)


def test_overflow_is_failure_without_clipping_or_state_repair():
    a = scalar_arrays()
    a['A'][:] = 1e308
    with pytest.raises(FloatingPointError, match='nonfinite NL-LFR rollout'):
        nllfr.rollout(a, np.zeros((1, 2, 3)), np.array([[2.]]))
    a = scalar_arrays()
    a['A'][:] = 1e200
    a['B_w'][:] = 0.
    a['B_u'][:] = 0.
    a['b0'][:] = 0.
    a['b1'][:] = 0.
    a['b2'][:] = 0.
    # Zero trajectory is finite; only initial-state sensitivity overflows.
    nllfr.rollout(a, np.zeros((1, 3, 3)), np.zeros((1, 1)))
    with pytest.raises(FloatingPointError, match='nonfinite NL-LFR sensitivities'):
        nllfr.rollout(a, np.zeros((1, 3, 3)), np.zeros((1, 1)), jacobian=True)


def fake_model(a):
    import equinox.nn._mlp as mlp_module
    import jax

    model = SimpleNamespace(**{key: a[key] for key in ('A', 'B_u', 'C_y', 'D_yu',
                                                       'B_w', 'C_z', 'D_yw', 'D_zu', 'ts')})
    model.norm = SimpleNamespace(**{key: a[key] for key in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    model.func_static = SimpleNamespace(layers=2, bias=True, activation=jax.nn.relu,
        model=SimpleNamespace(layers=tuple(SimpleNamespace(weight=a[f'W{i}'], bias=a[f'b{i}'])
                                         for i in range(3)), final_activation=mlp_module._identity))
    return model


def test_export_owned_exact_values_and_no_retained_original_bla():
    a = arrays()
    model = fake_model(a)
    model._bla = object()
    exported = nllfr.export(model)
    assert set(exported) == set(a)
    for key in a:
        np.testing.assert_array_equal(exported[key], a[key])
        assert exported[key].dtype == np.float64
        assert not np.shares_memory(exported[key], a[key])


@pytest.mark.parametrize('field', ['A', 'ts', 'u_mean', 'W1'])
def test_export_must_not_silently_promote_float32(field):
    a = arrays()
    a[field] = a[field].astype(np.float32)
    with pytest.raises(ValueError):
        nllfr.export(fake_model(a))


@pytest.mark.parametrize('mutation', ['activation', 'final_activation', 'depth', 'bias'])
def test_export_rejects_unsupported_static_function(mutation):
    import jax

    model = fake_model(arrays())
    if mutation == 'activation':
        model.func_static.activation = jax.nn.tanh
    elif mutation == 'final_activation':
        model.func_static.model.final_activation = jax.nn.tanh
    elif mutation == 'depth':
        model.func_static.layers = 1
    else:
        model.func_static.bias = False
    with pytest.raises(ValueError):
        nllfr.export(model)
