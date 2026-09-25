"""Fabricated inference parity, with fixed tolerances before native execution."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_structured_study as study

from openjev.research.native_robot_gru import NativeRobotGRU

RTOL = ATOL = 1e-5


@pytest.fixture
def library():
    value = os.environ.get('ROBOT_TRANSITION_LIBRARY')
    assert value and Path(value).is_file(), 'Explicit compiled library required; never skip qualification'
    return value


def model_fixture(saturated=False):
    coefficient = np.zeros((6, 25), dtype=np.float64, order='F')
    coefficient[:, :6] = .65 * np.eye(6)
    coefficient[:, 6:12] = .2 * np.eye(6)
    coefficient[:, 12:18] = .03 * np.eye(6)
    coefficient[:, 18:24] = .01 * np.eye(6)
    coefficient[:, 24] = np.linspace(-.01, .01, 6)
    model = study.GRU32(197251, coefficient)
    with torch.no_grad():
        model.gru.weight_ih.mul_(.6)
        model.gru.weight_hh.mul_(.6)
        model.head.weight.copy_(.025 * torch.sin(torch.arange(192).reshape(6, 32) / 11))
        model.head.bias.copy_(torch.linspace(-.01, .01, 6))
        if saturated:
            model.gru.bias_ih[:32].fill_(90)
            model.gru.bias_ih[32:64].fill_(-90)
    return model


def physical(batch=2, context=32, horizon=128):
    q = .3 * np.sin(np.arange(batch*context*6).reshape(batch, context, 6) / 31)
    u = .2 * np.cos(np.arange(batch*context*6).reshape(batch, context, 6) / 19)
    future = .4 * np.sin(np.arange(batch*horizon*6).reshape(batch, horizon, 6) / 23)
    norm = {'q_mean': np.linspace(-.2, .2, 6), 'q_std': np.linspace(.5, 1.5, 6),
            'u_mean': np.linspace(-.1, .1, 6), 'u_std': np.linspace(.7, 1.7, 6)}
    return q, u, future, norm


def oracle(model, q, u, future, norm):
    q32 = torch.from_numpy(((q - norm['q_mean']) / norm['q_std']).astype(np.float32))
    u32 = torch.from_numpy(((u - norm['u_mean']) / norm['u_std']).astype(np.float32))
    f32 = torch.from_numpy(((future - norm['u_mean']) / norm['u_std']).astype(np.float32))
    with torch.no_grad():
        state = model.condition(q32, u32)
        predicted, state = model(f32, state)
    return (predicted.numpy().astype(np.float64) * norm['q_std'] + norm['q_mean'],
            torch.cat(tuple(state), dim=1).numpy())


@pytest.mark.parametrize('context,horizon', [(2, 128), (32, 128), (32, 512)])
@pytest.mark.parametrize('saturated', [False, True])
def test_full_request_prediction_and_all_50_state_values(library, context, horizon, saturated):
    model = model_fixture(saturated)
    args = physical(context=context, horizon=horizon)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    original = [value.copy() for value in args[:3]]
    native = NativeRobotGRU(model, library)
    expected, state = oracle(model, *args)
    actual = native.request(*args)
    np.testing.assert_allclose(actual['prediction'], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual['final_state'], state, rtol=RTOL, atol=ATOL)
    assert actual['prediction'].dtype == np.float64 and actual['final_state'].dtype == np.float32
    assert actual['final_state'].shape == (2, 50)
    assert actual['work']['context_gru_steps'] == 2*(context-2)
    assert actual['work']['packed_parameter_bytes'] == 5916*4
    assert actual['work']['ffi_calls'] == 1 and actual['work']['retained_prepared_bytes'] == 0
    for old, current in zip(original, args[:3], strict=True):
        assert np.array_equal(old, current)
    assert all(torch.equal(before[n], p) and p.grad is None for n, p in model.named_parameters())
    storage = native.storage()
    assert storage['parameter_count'] == 5916 and storage['state_bytes_per_stream'] == 200
    assert storage['retained_prepared_bytes'] == storage['retained_gradient_bytes'] == 0


def test_first_step_closed_form_and_torque_alignment(library):
    model = model_fixture()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    q, u, future, norm = physical(context=2, horizon=1)
    norm = {k: np.ones(6) if k.endswith('std') else np.zeros(6) for k in norm}
    actual = NativeRobotGRU(model, library).request(q, u, future, norm)
    expected = .65*q[:, -1]+.2*q[:, -2]+.03*future[:, 0]+.01*u[:, -2]+np.linspace(-.01, .01, 6)
    np.testing.assert_allclose(actual['prediction'][:, 0], expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('sign', [-1.0, 1.0])
def test_finite_parameters_with_signed_infinite_affines_saturate(library, sign):
    """Finite W*x overflows, but sigmoid/tanh yield valid finite GRU states."""
    model = model_fixture()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        largest = torch.finfo(torch.float32).max
        # x[0]=2 makes reset/candidate +/-inf and update -inf. The hidden
        # candidate affine is zero, so both reset limits remain well-defined.
        model.gru.weight_ih[:32, 0].fill_(sign * largest)
        model.gru.weight_ih[32:64, 0].fill_(-largest)
        model.gru.weight_ih[64:, 0].fill_(sign * largest)
    q = np.zeros((1, 2, 6), dtype=np.float64)
    q[0, -1, 0] = 2.0
    u = np.zeros_like(q)
    future = np.zeros((1, 1, 6), dtype=np.float64)
    norm = {key: np.ones(6) if key.endswith('std') else np.zeros(6)
            for key in ('q_mean', 'q_std', 'u_mean', 'u_std')}
    expected, expected_state = oracle(model, q, u, future, norm)
    assert np.isfinite(expected).all() and np.isfinite(expected_state).all()
    np.testing.assert_allclose(expected_state[:, 18:], sign, rtol=RTOL, atol=ATOL)
    actual = NativeRobotGRU(model, library).request(q, u, future, norm)
    np.testing.assert_allclose(actual['prediction'], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual['final_state'], expected_state, rtol=RTOL, atol=ATOL)
    assert np.isfinite(actual['prediction']).all() and np.isfinite(actual['final_state']).all()


def test_causal_suffix_and_observed_prefix_boundary(library):
    model = model_fixture()
    native = NativeRobotGRU(model, library)
    q, u, future, norm = physical()
    original = native.request(q, u, future, norm)['prediction']
    changed_future = future.copy(); changed_future[:, 64:] += 4
    assert np.array_equal(original[:, :64], native.request(q, u, changed_future, norm)['prediction'][:, :64])
    unused = u.copy(); unused[:, -1] += 10
    assert np.array_equal(original, native.request(q, unused, future, norm)['prediction'])
    older_q = q.copy(); older_q[:, -3] += 5
    assert not np.array_equal(original[:, :1], native.request(older_q, u, future, norm)['prediction'][:, :1])


def test_carried_state_and_zero_horizon_are_owned(library):
    model = model_fixture()
    native = NativeRobotGRU(model, library)
    q, u, future, norm = physical(horizon=128)
    q32 = torch.from_numpy(((q-norm['q_mean'])/norm['q_std']).astype(np.float32))
    u32 = torch.from_numpy(((u-norm['u_mean'])/norm['u_std']).astype(np.float32))
    f32 = ((future-norm['u_mean'])/norm['u_std']).astype(np.float32)
    with torch.no_grad():
        initial = torch.cat(tuple(model.condition(q32, u32)), dim=1).numpy()
    full = native.rollout(f32, initial)
    first = native.rollout(f32[:, :64], initial)
    second = native.rollout(f32[:, 64:], first['final_state'])
    assert np.array_equal(full['prediction'], np.concatenate((first['prediction'], second['prediction']), axis=1))
    assert np.array_equal(full['final_state'], second['final_state'])
    empty = native.rollout(f32[:, :0], initial)
    assert empty['prediction'].shape == (2, 0, 6)
    assert np.array_equal(empty['final_state'], initial) and not np.shares_memory(empty['final_state'], initial)


def test_fortran_export_noncontiguous_inputs_and_parameter_refresh(library):
    model = model_fixture()
    assert not model.base_weight.is_contiguous()
    native = NativeRobotGRU(model, library)
    q, u, future, norm = physical()
    q, u, future = (np.asfortranarray(v) for v in (q, u, future))
    before = native.request(q, u, future, norm)['prediction']
    with torch.no_grad():
        model.head.bias.add_(.015)
    after = native.request(q, u, future, norm)['prediction']
    assert not np.array_equal(before, after)
    np.testing.assert_allclose(after, oracle(model, q, u, future, norm)[0], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('damage', ['float32_physical', 'nan', 'bad_scale', 'missing_norm', 'wrong_context', 'gradient', 'parameter_nan'])
def test_invalid_requests_fail_without_repairs(library, damage):
    model = model_fixture()
    native = NativeRobotGRU(model, library)
    q, u, future, norm = physical()
    if damage == 'float32_physical': q = q.astype(np.float32)
    elif damage == 'nan': future[0, 0, 0] = np.nan
    elif damage == 'bad_scale': norm['u_std'][0] = 0
    elif damage == 'missing_norm': norm.pop('q_mean')
    elif damage == 'wrong_context': u = u[:, :-1]
    elif damage == 'gradient': model.head.bias.grad = torch.ones_like(model.head.bias)
    else:
        with torch.no_grad(): model.head.bias[0] = float('nan')
    with pytest.raises(ValueError): native.request(q, u, future, norm)


def test_library_is_explicit():
    with pytest.raises(ValueError, match='explicit library_path'):
        NativeRobotGRU(model_fixture())
