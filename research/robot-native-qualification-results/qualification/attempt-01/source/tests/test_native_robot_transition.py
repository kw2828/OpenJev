"""Native inference qualification on fabricated inputs, never study checkpoints.

Fixed before first compiled execution: CPU float32 predictions AND final state
must match the qualified Torch implementation at rtol=1e-5, atol=1e-5 for both
H128 and H512. No performance assertion, training, or empirical input loading.
The shared library is an explicit caller-supplied build, never auto-discovered.
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from openjev.research.native_robot_transition import NativeRobotTransition
from openjev.research.structured_robot_transition import KINDS, StructuredRobotTransition

RTOL = 1e-5
ATOL = 1e-5
SEED = 97241


@pytest.fixture(scope='session')
def library():
    value = os.environ.get('ROBOT_TRANSITION_LIBRARY')
    if value is None:
        pytest.fail('ROBOT_TRANSITION_LIBRARY must explicitly name the qualified build')
    path = Path(value)
    assert path.is_file() and path.is_absolute()
    return path


def inputs(horizon, batch=2):
    q = np.sin(np.arange(batch*32*6, dtype=np.float64).reshape(batch, 32, 6)/19)*.3
    u = np.cos(np.arange(batch*32*6, dtype=np.float64).reshape(batch, 32, 6)/23)*.4
    future = np.sin(np.arange(batch*horizon*6, dtype=np.float64).reshape(batch, horizon, 6)/17)*.5
    return q.astype(np.float32), u.astype(np.float32), future.astype(np.float32)


def model(kind, profile='active'):
    cell = StructuredRobotTransition(kind, SEED)
    if profile == 'initial':
        return cell
    with torch.no_grad():
        # Every trainable tensor gets nonconstant data, including both forcing
        # experts, metric, all gate affine terms and nonidentical reflectors.
        for index, (name, value) in enumerate(cell.named_parameters()):
            if name in ('raw_matrix', 'reflection_raw', 'decay_raw', 'scale_raw'):
                continue
            wave = torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
            value.copy_(.08*torch.sin(wave*.7+index*.3)+.015*torch.cos(wave*.23-index))
        cell.scale_raw.copy_(torch.linspace(-.25, .7, 12))
        if kind == 'householder':
            wave = torch.arange(88, dtype=torch.float32).reshape(2, 4, 11)
            cell.reflection_raw.copy_(.31*torch.sin(wave*.37)+.09*torch.cos(wave*.71))
            cell.decay_raw.copy_(torch.linspace(3.1, 4.2, 24).reshape(2, 12))
        else:
            wave = torch.arange(288, dtype=torch.float32).reshape(2, 12, 12)
            raw = .87*torch.eye(12)[None]+.012*torch.sin(wave*.47)
            # Bound is active for both bounded controls. Unbounded gets a
            # stable fabricated matrix to isolate parity rather than overflow.
            cell.raw_matrix.copy_(raw*(3. if kind != 'dense_unbounded' else 1.))
        if profile == 'saturated':
            if kind == 'dense_mlp':
                cell.gate[0].bias.copy_(torch.tensor([40., -40.]*4))
                cell.gate[2].bias.copy_(torch.tensor([60., -60.]))
            else:
                cell.gate.reset_update_bias.copy_(torch.tensor([40., -40.]*8))
                cell.gate.candidate_input_bias.copy_(torch.tensor([-30., 30.]*4))
                cell.gate.candidate_hidden_bias.copy_(torch.tensor([20., -20.]*4))
                cell.gate.head_bias.copy_(torch.tensor([-60., 60.]))
    return cell


def torch_rollout(cell, future, state):
    with torch.no_grad():
        predicted, final = cell(torch.from_numpy(future), torch.from_numpy(state))
    return predicted.numpy(), final.numpy()


def initial_state(cell, q, u):
    with torch.no_grad():
        return cell.condition(torch.from_numpy(q), torch.from_numpy(u)).numpy()


def assert_parity(actual, expected_prediction, expected_state):
    assert actual['prediction'].dtype == np.float32
    assert actual['final_state'].dtype == np.float32
    assert actual['prediction'].shape == expected_prediction.shape
    assert actual['final_state'].shape == expected_state.shape
    assert np.isfinite(actual['prediction']).all() and np.isfinite(actual['final_state']).all()
    np.testing.assert_allclose(actual['prediction'], expected_prediction, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual['final_state'], expected_state, rtol=RTOL, atol=ATOL)
    assert isinstance(actual['work'], dict)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('horizon', [128, 512])
@pytest.mark.parametrize('profile', ['initial', 'active', 'saturated'])
def test_fixed_long_horizon_prediction_and_state_parity(kind, horizon, profile, library):
    cell = model(kind, profile)
    q, u, future = inputs(horizon)
    state = initial_state(cell, q, u)
    expected = torch_rollout(cell, future, state)
    native = NativeRobotTransition(cell, library_path=library)
    assert_parity(native.rollout(future, state), *expected)


@pytest.mark.parametrize('kind', KINDS)
def test_finite_affine_overflow_saturates_like_torch(kind, library):
    """Signed infinity inside a saturating activation is not a bad output."""
    cell = model(kind)
    state = np.ones((1, 12), dtype=np.float32)
    future = np.ones((1, 1, 6), dtype=np.float32)
    with torch.no_grad():
        cell.input_matrix.zero_()
        cell.expert_bias.zero_()
        weight = cell.gate[0].weight if kind == 'dense_mlp' else cell.gate.input_weight
        for row in range(weight.shape[0]):
            weight[row].fill_(2e38 if row % 2 == 0 else -2e38)
        affine = torch.ones((1, 12)) @ weight.T
        assert torch.isposinf(affine[:, ::2]).all()
        assert torch.isneginf(affine[:, 1::2]).all()
        assert not torch.isnan(affine).any()
    expected = torch_rollout(cell, future, state)
    assert all(np.isfinite(value).all() for value in expected)
    actual = NativeRobotTransition(cell, library_path=library).rollout(future, state)
    assert_parity(actual, *expected)


@pytest.mark.parametrize('kind', KINDS)
def test_nonfinite_final_gate_logits_fail_after_saturation(kind, library):
    """Allowing affine saturation must not repair an undefined softmax."""
    cell = model(kind)
    state = np.ones((1, 12), dtype=np.float32)
    future = np.ones((1, 1, 6), dtype=np.float32)
    with torch.no_grad():
        cell.input_matrix.zero_()
        cell.expert_bias.zero_()
        if kind == 'dense_mlp':
            cell.gate[0].weight.zero_()
            cell.gate[0].bias.fill_(40.)
            cell.gate[2].weight.fill_(2e38)
            cell.gate[2].bias.zero_()
        else:
            cell.gate.input_weight.zero_()
            cell.gate.reset_update_bias[:8].zero_()
            cell.gate.reset_update_bias[8:].fill_(-40.)
            cell.gate.candidate_input_bias.fill_(40.)
            cell.gate.candidate_hidden_bias.zero_()
            cell.gate.head_weight.fill_(2e38)
            cell.gate.head_bias.zero_()
    # Both logits are +inf, so subtracting their maximum is NaN. Neither
    # implementation may turn that undefined gate into finite predictions.
    with pytest.raises(ValueError):
        torch_rollout(cell, future, state)
    with pytest.raises(ValueError):
        NativeRobotTransition(cell, library_path=library).rollout(future, state)


@pytest.mark.parametrize('kind', KINDS)
def test_causal_suffix_chunk_carry_and_owned_buffers(kind, library):
    cell = model(kind)
    q, u, future = inputs(128)
    state = initial_state(cell, q, u)
    prior_state, prior_future = state.copy(), future.copy()
    native = NativeRobotTransition(cell, library_path=library)
    full = native.rollout(future, state)
    altered = future.copy(); altered[:, 63:] += .3
    changed = native.rollout(altered, state)
    np.testing.assert_array_equal(full['prediction'][:, :63], changed['prediction'][:, :63])
    first = native.rollout(future[:, :63], state)
    tail = native.rollout(future[:, 63:], first['final_state'])
    np.testing.assert_allclose(np.concatenate((first['prediction'], tail['prediction']), 1), full['prediction'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(tail['final_state'], full['final_state'], rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(state, prior_state)
    np.testing.assert_array_equal(future, prior_future)
    assert not np.shares_memory(full['prediction'], full['final_state'])
    for returned in (full['prediction'], full['final_state']):
        assert not np.shares_memory(returned, state) and not np.shares_memory(returned, future)
    saved_prediction = full['prediction'].copy()
    full['final_state'].fill(77)
    np.testing.assert_array_equal(native.rollout(future, state)['prediction'], saved_prediction)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('batch', [1, 3])
def test_empty_horizon_batch_shapes_and_owned_state(kind, batch, library):
    cell = model(kind)
    q, u, future = inputs(0, batch)
    state = initial_state(cell, q, u)
    native = NativeRobotTransition(cell, library_path=library)
    actual = native.rollout(future, state)
    assert actual['prediction'].shape == (batch, 0, 6)
    np.testing.assert_array_equal(actual['final_state'], state)
    assert not np.shares_memory(actual['final_state'], state)
    actual['final_state'].fill(12)
    assert not np.all(state == 12)


@pytest.mark.parametrize('kind', KINDS)
def test_physical_request_matches_pipeline_and_ignores_old_history(kind, library):
    cell = model(kind)
    q, u, future = (a.astype(np.float64) for a in inputs(128))
    normalizers = {'q_mean': np.linspace(-.2, .3, 6), 'q_std': np.linspace(.7, 1.3, 6),
                   'u_mean': np.linspace(.2, -.1, 6), 'u_std': np.linspace(.8, 1.5, 6)}
    qn = ((q-normalizers['q_mean'])/normalizers['q_std']).astype(np.float32)
    un = ((u-normalizers['u_mean'])/normalizers['u_std']).astype(np.float32)
    fn = ((future-normalizers['u_mean'])/normalizers['u_std']).astype(np.float32)
    expected, final = torch_rollout(cell, fn, initial_state(cell, qn, un))
    native = NativeRobotTransition(cell, library_path=library)
    actual = native.request(q, u, future, normalizers)
    assert actual['prediction'].dtype == np.float64
    np.testing.assert_allclose(actual['prediction'], expected.astype(np.float64)*normalizers['q_std']+normalizers['q_mean'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual['final_state'], final, rtol=RTOL, atol=ATOL)
    changed = q.copy(); changed[:, :-2] += 10
    result = native.request(changed, u+7, future, normalizers)
    np.testing.assert_array_equal(result['prediction'], actual['prediction'])
    for value in normalizers.values():
        assert np.isfinite(value).all()


@pytest.mark.parametrize('kind', KINDS)
def test_fresh_preparation_once_per_call_and_no_stale_parameter_cache(kind, library, monkeypatch):
    cell = model(kind)
    q, u, future = inputs(13)
    state = initial_state(cell, q, u)
    native = NativeRobotTransition(cell, library_path=library)
    calls = []
    prepare = cell._prepare
    def tracked():
        calls.append(1)
        return prepare()
    monkeypatch.setattr(cell, '_prepare', tracked)
    first = native.rollout(future, state)
    assert len(calls) == 1
    with torch.no_grad():
        cell.input_matrix[1].add_(.1)
        cell.expert_bias[0].add_(.02)
    second = native.rollout(future, state)
    assert len(calls) == 2 and not np.array_equal(first['prediction'], second['prediction'])
    expected = torch_rollout(cell, future, state)
    assert_parity(second, *expected)


@pytest.mark.parametrize('kind', KINDS)
def test_each_live_parameter_tensor_is_reexported(kind, library):
    cell = model(kind)
    q, u, future = inputs(7)
    state = initial_state(cell, q, u)
    native = NativeRobotTransition(cell, library_path=library)
    for _, parameter in cell.named_parameters():
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.view(-1)[0] += .4
        expected = torch_rollout(cell, future, state)
        assert_parity(native.rollout(future, state), *expected)
        with torch.no_grad():
            parameter.copy_(original)


@pytest.mark.parametrize('damage', ['future_dtype', 'state_dtype', 'future_shape', 'state_shape',
                                   'future_nan', 'future_inf', 'state_nan', 'zero_batch'])
def test_invalid_rollout_inputs_fail_without_repair(damage, library):
    cell = model('householder')
    q, u, future = inputs(4)
    state = initial_state(cell, q, u)
    if damage == 'future_dtype':
        future = future.astype(np.float64)
    elif damage == 'state_dtype':
        state = state.astype(np.float64)
    elif damage == 'future_shape':
        future = future[:, :, :5]
    elif damage == 'state_shape':
        state = state[:, :11]
    elif damage == 'future_nan':
        future[0, 0, 0] = np.nan
    elif damage == 'future_inf':
        future[0, 0, 0] = np.inf
    elif damage == 'state_nan':
        state[0, 0] = np.nan
    else:
        future, state = future[:0], state[:0]
    with pytest.raises(ValueError):
        NativeRobotTransition(cell, library_path=library).rollout(future, state)


@pytest.mark.parametrize('damage', ['q_dtype', 'u_dtype', 'future_dtype', 'q_short', 'q_shape',
                                   'u_nan', 'q_nan', 'norm_key', 'norm_dtype', 'norm_zero', 'norm_nan'])
def test_invalid_physical_request_fails(damage, library):
    cell = model('householder')
    q, u, future = (a.astype(np.float64) for a in inputs(4))
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    if damage == 'q_dtype':
        q = q.astype(np.float32)
    elif damage == 'u_dtype':
        u = u.astype(np.float32)
    elif damage == 'future_dtype':
        future = future.astype(np.float32)
    elif damage == 'q_short':
        q, u = q[:, :1], u[:, :1]
    elif damage == 'q_shape':
        q = q[:, :, :5]
    elif damage == 'u_nan':
        u[0, 0, 0] = np.nan
    elif damage == 'q_nan':
        q[0, 0, 0] = np.nan
    elif damage == 'norm_key':
        norm['extra'] = np.zeros(6)
    elif damage == 'norm_dtype':
        norm['q_mean'] = norm['q_mean'].astype(np.float32)
    elif damage == 'norm_zero':
        norm['u_std'][0] = 0
    else:
        norm['q_mean'][0] = np.nan
    with pytest.raises(ValueError):
        NativeRobotTransition(cell, library_path=library).request(q, u, future, norm)


def test_nonfinite_live_parameters_and_overflow_fail_without_repair(library):
    cell = model('dense_unbounded')
    q, u, future = inputs(8)
    state = initial_state(cell, q, u)
    native = NativeRobotTransition(cell, library_path=library)
    with torch.no_grad():
        cell.raw_matrix.fill_(2e38)
    with pytest.raises(ValueError):
        native.rollout(future, state)
    with torch.no_grad():
        cell.raw_matrix[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        native.rollout(future, state)


def test_library_and_exact_float32_model_are_explicit(library):
    with pytest.raises((ValueError, TypeError)):
        NativeRobotTransition(model('householder'), library_path=None)
    native = NativeRobotTransition(StructuredRobotTransition('householder', SEED, dtype=torch.float64), library_path=library)
    with pytest.raises(ValueError):
        native.rollout(np.zeros((1, 1, 6), np.float32), np.zeros((1, 12), np.float32))


@pytest.mark.parametrize('kind,count,packed,prepared', [
    ('householder', 630, 638, 528), ('dense_bounded', 806, 806, 1208),
    ('dense_unbounded', 806, 806, 1200), ('dense_mlp', 590, 590, 1208),
])
def test_request_work_charges_boundary_preparation_and_copies(kind, count, packed, prepared, library):
    cell = model(kind)
    native = NativeRobotTransition(cell, library_path=library)
    q, u, future = (a.astype(np.float64) for a in inputs(3))
    norm = {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
    result = native.request(q, u, future, norm)
    work = result['work']
    assert work['parameter_validations'] == 2 and work['parameter_preparations'] == 1
    assert work['ffi_calls'] == 1 and work['steps'] == 6
    assert work['packed_parameter_bytes'] == work['parameter_export_piece_bytes'] == packed*4
    assert work['prepared_tensor_bytes'] == prepared
    assert work['input_copy_bytes'] == (2*3*6+2*12)*4
    assert work['output_buffer_bytes'] == (2*3*6+2*12)*4
    assert work['normalized_input_bytes'] == q.nbytes+u.nbytes+future.nbytes
    assert work['cast_input_bytes'] == (q.nbytes+u.nbytes+future.nbytes)//2
    assert work['condition_state_bytes'] == 2*12*4
    assert work['physical_output_bytes'] == 2*3*6*8
    assert work['retained_prepared_bytes'] == 0
    storage = native.storage()
    assert storage['parameter_count'] == count and storage['parameter_bytes'] == count*4
    assert storage['buffer_bytes'] == storage['retained_gradient_bytes'] == storage['retained_prepared_bytes'] == 0
    assert storage['state_bytes_per_stream'] == 48 and storage['normalizer_bytes'] == 192


def test_retained_gradients_are_counted_and_rejected_for_inference(library):
    cell = model('householder')
    native = NativeRobotTransition(cell, library_path=library)
    cell.input_matrix.grad = torch.ones_like(cell.input_matrix)
    assert native.storage()['retained_gradient_bytes'] == 144*4
    with pytest.raises(ValueError, match='cleared parameter gradients'):
        native.rollout(np.zeros((1, 1, 6), np.float32), np.zeros((1, 12), np.float32))


@pytest.mark.parametrize('damage,expected', [
    ('length', 1), ('kind', 1), ('null_parameters', 2), ('null_initial', 2),
    ('output_alias', 2), ('nonfinite_parameter', 3), ('negative_scale', 4),
    ('zero_reflection', 4),
])
def test_direct_abi_rejects_bad_extents_aliases_and_domains(damage, expected, library):
    native = NativeRobotTransition(model('householder'), library_path=library)
    with torch.no_grad():
        packed, _ = native._pack()
    future = np.zeros((1, 1, 6), np.float32)
    state = np.ones((1, 12), np.float32)
    prediction = np.full((1, 1, 6), 17., np.float32)
    final = np.full((1, 12), 19., np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)
    args = [0, 1, 1, packed.ctypes.data_as(pointer), packed.size,
            future.ctypes.data_as(pointer), future.size, state.ctypes.data_as(pointer), state.size,
            prediction.ctypes.data_as(pointer), prediction.size, final.ctypes.data_as(pointer), final.size]
    if damage == 'length':
        args[4] -= 1
    elif damage == 'kind':
        args[0] = 4
    elif damage == 'null_parameters':
        args[3] = pointer()
    elif damage == 'null_initial':
        args[7] = pointer()
    elif damage == 'output_alias':
        args[9] = args[3]
    elif damage == 'nonfinite_parameter':
        packed[0] = np.nan
    elif damage == 'negative_scale':
        packed[0] = -1
    else:
        packed[180:276] = 0
    packed_before = packed.copy()
    assert native._function(*args) == expected
    np.testing.assert_array_equal(packed, packed_before)
    np.testing.assert_array_equal(state, np.ones((1, 12), np.float32))
    np.testing.assert_array_equal(future, np.zeros((1, 1, 6), np.float32))
