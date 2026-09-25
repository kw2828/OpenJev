"""Independent fabricated histories only; no measurements, fits or checkpoints.

Exact equality is required for paired zero-head functions and gradients. Hand
feature/oracle checks use 2e-12; finite differences use 2e-5 relative / 2e-8
absolute tolerance, declared before the first qualification invocation.
"""
import inspect
import math

import numpy as np
import pytest
import torch

from openjev.research.robot_history_initializer import HistoryInitializedDense
from openjev.research.structured_robot_transition import StructuredRobotTransition

MODES = ('last_two', 'local_affine', 'temporal_affine')
AFFINE = MODES[1:]


def histories(dtype=torch.float64, horizon=5):
    index = torch.arange(2 * 32 * 6, dtype=dtype).reshape(2, 32, 6)
    q = torch.sin(index / 19) + index / 173
    u = torch.cos(index / 23) + index / 211
    future = torch.sin(torch.arange(2 * horizon * 6, dtype=dtype).reshape(2, horizon, 6) / 17)
    return q, u, future


def activate_cell(cell):
    """Distinct nonsingular operators with an active gate, safely below the cap."""
    dtype = cell.input_matrix.dtype
    with torch.no_grad():
        cell.raw_matrix[0].copy_(torch.diag(torch.linspace(.5, .75, 12, dtype=dtype)) + .007)
        cell.raw_matrix[1].copy_(torch.diag(torch.linspace(.72, .48, 12, dtype=dtype)) - .004)
        cell.expert_bias.copy_(.01 * torch.sin(torch.arange(24, dtype=dtype).reshape(2, 12)))
        cell.gate[2].weight.copy_(.15 * torch.sin(torch.arange(16, dtype=dtype).reshape(2, 8)))
        cell.gate[2].bias.copy_(torch.tensor([.03, -.02], dtype=dtype))


def activate_head(model):
    if model.initializer != 'last_two':
        dtype = model.head.weight.dtype
        with torch.no_grad():
            model.head.weight.copy_(.003 * torch.cos(torch.arange(360, dtype=dtype).reshape(12, 30) / 17))
            model.head.bias.copy_(torch.linspace(-.01, .02, 12, dtype=dtype))


def hand_features(q, u, mode):
    """Scalar sums and Python tanh, independent of the tensor feature routine."""
    outputs = []
    for qb, ub in zip(q.tolist(), u.tolist(), strict=True):
        dq = [qb[31][j] - qb[30][j] for j in range(6)]
        common = qb[31] + dq + ub[30]
        if mode == 'local_affine':
            extra = [v * v for v in dq] + [math.tanh(v) for v in ub[30]]
        else:
            denominator = sum((t - 14.5) ** 2 for t in range(30))
            assert denominator == 2247.5
            extra = [sum((t - 14.5) * qb[t][j] for t in range(30)) / denominator for j in range(6)]
            extra += [sum(ub[t][j] for t in range(30)) / 30 for j in range(6)]
        outputs.append(common + extra)
    return np.asarray(outputs, dtype=np.float64)


def run_with_grad(model, q, u, future):
    q, u, future = [value.detach().clone().requires_grad_() for value in (q, u, future)]
    state = model.condition(q, u)
    prediction, final = model(future, state)
    loss = prediction.square().sum() + .17 * final.square().sum()
    loss.backward()
    # An unused baseline torque input has None; a zero affine path materializes
    # a zero gradient. Both mean the same mathematical derivative.
    gradients = [torch.zeros_like(v) if v.grad is None else v.grad for v in (q, u, future)]
    return state, prediction, final, gradients


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('active', [False, True])
def test_paired_zero_heads_preserve_exact_function_inputs_and_common_gradients(mode, dtype, active):
    baseline = StructuredRobotTransition('dense_mlp', 984101, dtype=dtype)
    candidate = HistoryInitializedDense(984101, mode, dtype=dtype)
    base_names = list(dict(baseline.named_parameters()))
    assert list(dict(candidate.named_parameters()))[:len(base_names)] == ['cell.' + name for name in base_names]
    for name, value in baseline.named_parameters():
        assert torch.equal(value, dict(candidate.cell.named_parameters())[name])
    if active:
        activate_cell(baseline)
        candidate.cell.load_state_dict(baseline.state_dict())
    if mode in AFFINE:
        assert torch.count_nonzero(candidate.head.weight) == torch.count_nonzero(candidate.head.bias) == 0
    inputs = histories(dtype)
    left, right = run_with_grad(baseline, *inputs), run_with_grad(candidate, *inputs)
    for expected, actual in zip(left[:3], right[:3], strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(left[3], right[3], strict=True):
        assert torch.equal(expected, actual)
    for name, parameter in baseline.named_parameters():
        actual = dict(candidate.cell.named_parameters())[name].grad
        assert parameter.grad is not None and actual is not None
        assert torch.isfinite(actual).all() and torch.equal(parameter.grad, actual)


@pytest.mark.parametrize('mode', AFFINE)
def test_hand_features_and_affine_state_from_scalar_oracle(mode):
    model = HistoryInitializedDense(984102, mode, dtype=torch.float64)
    activate_head(model)
    q, u, _ = histories()
    expected = hand_features(q, u, mode)
    actual = model.initialization_features(q, u)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-12, rtol=2e-12)
    base = np.concatenate((q.numpy()[:, 31], q.numpy()[:, 31] - q.numpy()[:, 30]), axis=-1)
    independent = base + expected @ model.head.weight.detach().numpy().T + model.head.bias.detach().numpy()
    np.testing.assert_allclose(model.condition(q, u).detach().numpy(), independent, atol=2e-12, rtol=2e-12)


def test_temporal_slope_and_mean_use_exact_thirty_older_indices():
    model = HistoryInitializedDense(984103, 'temporal_affine', dtype=torch.float64)
    t = torch.arange(32, dtype=torch.float64)[None, :, None]
    channels = torch.arange(1, 7, dtype=torch.float64)[None, None, :]
    q = 3 * channels + t * channels
    u = 2 * channels + 2 * t
    expected_slope = channels[:, 0]
    expected_mean = 2 * channels[:, 0] + 29
    q[:, 30:] = -1e6
    u[:, 30:] = 1e6
    features = model.initialization_features(q, u)
    assert torch.equal(features[:, 18:24], expected_slope)
    assert torch.equal(features[:, 24:], expected_mean)
    q[:, :30] = channels * 7
    assert torch.equal(model.initialization_features(q, u)[:, 18:24], torch.zeros(1, 6, dtype=torch.float64))


@pytest.mark.parametrize('mode', MODES)
def test_condition_torque31_exclusion_and_public_signature(mode):
    model = HistoryInitializedDense(984104, mode, dtype=torch.float64)
    activate_head(model)
    q, u, future = histories()
    original = model.condition(q, u)
    changed = u.clone(); changed[:, 31] += 900
    assert torch.equal(original, model.condition(q, changed))
    assert tuple(inspect.signature(model.condition).parameters) == ('q_context', 'u_context')
    assert tuple(inspect.signature(model.forward).parameters) == ('future_u', 'state')
    with pytest.raises(TypeError):
        model.condition(q, u, targets=q)
    # The excluded context torque is still valid as the FIRST forecast input.
    future[:, 0] = u[:, 31]
    full, _ = model(future, original)
    one, _ = model.step(original, u[:, 31])
    assert torch.equal(one, full[:, 0])
    changed_future = future.clone(); changed_future[:, 0] += .5
    assert not torch.equal(full[:, 0], model(changed_future, original)[0][:, 0])


@pytest.mark.parametrize('mode', MODES)
def test_active_history_and_fixed_paired_permutation_distinguish_information(mode):
    model = HistoryInitializedDense(984105, mode, dtype=torch.float64)
    activate_head(model)
    t = torch.arange(32, dtype=torch.float64)[None, :, None]
    channel = torch.arange(1, 7, dtype=torch.float64)[None, None, :]
    q = 2 + t * channel
    u = 3 + t + channel
    base = model.condition(q, u)
    older_q = q.clone(); older_q[:, 2] += 7
    older_u = u.clone(); older_u[:, 2] += 9
    order = list(reversed(range(30))) + [30, 31]
    changed_states = [model.condition(older_q, u), model.condition(q, older_u), model.condition(q[:, order], u[:, order])]
    for changed in changed_states:
        assert torch.equal(base, changed) is (mode != 'temporal_affine')
    if mode == 'temporal_affine':
        before = model.initialization_features(q, u)
        after = model.initialization_features(q[:, order], u[:, order])
        assert torch.equal(before[:, :18], after[:, :18])
        assert torch.equal(before[:, 24:], after[:, 24:])
        assert torch.equal(before[:, 18:24], -after[:, 18:24])


@pytest.mark.parametrize('mode', AFFINE)
def test_every_head_coordinate_has_a_nonzero_forecast_gradient(mode):
    model = HistoryInitializedDense(984106, mode, dtype=torch.float64)
    # Positive full mixing gives every initial-state coordinate a path into the
    # six reported positions. All thirty independently computed features >0.
    with torch.no_grad():
        model.cell.raw_matrix.copy_((.4 * torch.eye(12, dtype=torch.float64) + .01).repeat(2, 1, 1))
        model.cell.input_matrix.zero_()
        model.head.weight.fill_(.0001)
        model.head.bias.fill_(.001)
    t = torch.arange(32, dtype=torch.float64)[None, :, None]
    channels = torch.arange(1, 7, dtype=torch.float64)[None, None, :]
    q, u = 1 + t * channels / 32, 1 + t / 32 + channels / 16
    future = torch.zeros(1, 3, 6, dtype=torch.float64)
    assert (hand_features(q, u, mode) > 0).all()
    model(future, model.condition(q, u))[0][:, -1].sum().backward()
    gradients = torch.cat((model.head.weight.grad.flatten(), model.head.bias.grad))
    assert gradients.numel() == 372 and torch.isfinite(gradients).all() and (gradients > 0).all()


@pytest.mark.parametrize('mode', AFFINE)
def test_later_forecast_head_and_public_input_derivatives_by_finite_difference(mode):
    model = HistoryInitializedDense(984107, mode, dtype=torch.float64)
    activate_cell(model.cell); activate_head(model)
    q, u, future = [v.requires_grad_() for v in histories(horizon=4)]
    def value():
        prediction, _ = model(future, model.condition(q, u))
        return prediction[:, -1].square().sum()
    value().backward()
    probes = [(model.head.weight, (0, 0)), (model.head.weight, (8, 20)),
              (model.head.weight, (11, 29)), (model.head.bias, (10,)),
              (q, (0, 31, 1)), (u, (0, 30, 2)), (future, (0, 2, 4))]
    if mode == 'temporal_affine':
        probes += [(q, (0, 3, 2)), (u, (0, 4, 3))]
    for tensor, index in probes:
        derivative = tensor.grad[index].item()
        center = tensor[index].item()
        assert math.isfinite(derivative) and abs(derivative) > 1e-10
        with torch.no_grad():
            tensor[index] = center + 1e-6
            plus = value().item()
            tensor[index] = center - 1e-6
            minus = value().item()
            tensor[index] = center
        assert derivative == pytest.approx((plus - minus) / 2e-6, rel=2e-5, abs=2e-8)
    assert torch.count_nonzero(u.grad[:, 31]) == 0
    if mode == 'local_affine':
        assert torch.count_nonzero(q.grad[:, :30]) == torch.count_nonzero(u.grad[:, :30]) == 0


@pytest.mark.parametrize('mode', MODES)
def test_chunk_causality_batch_independence_and_owned_h0(mode):
    model = HistoryInitializedDense(984108, mode, dtype=torch.float64)
    activate_cell(model.cell); activate_head(model)
    q, u, future = histories(horizon=6)
    originals = [v.clone() for v in (q, u, future)]
    state = model.condition(q, u)
    out, final = model(future, state)
    first, carry = model(future[:, :2], state)
    second, chunk_final = model(future[:, 2:], carry)
    assert torch.equal(torch.cat((first, second), 1), out) and torch.equal(chunk_final, final)
    changed = future.clone(); changed[:, 3:] += 70
    assert torch.equal(model(changed, state)[0][:, :3], out[:, :3])
    separate = torch.cat([model(future[i:i+1], model.condition(q[i:i+1], u[i:i+1]))[0] for i in range(2)])
    torch.testing.assert_close(separate, out, atol=2e-12, rtol=2e-12)
    empty, unchanged = model(future[:, :0], state)
    assert empty.shape == (2, 0, 6) and torch.equal(unchanged, state)
    assert unchanged.data_ptr() != state.data_ptr()
    assert state.data_ptr() not in {q.data_ptr(), u.data_ptr()}
    for before, after in zip(originals, (q, u, future), strict=True):
        assert torch.equal(before, after)
    with torch.no_grad():
        unchanged.add_(12)
    assert torch.equal(model(future, model.condition(q, u))[0], out)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_parameter_storage_rng_isolation_and_no_retained_cache(mode, dtype):
    rng = torch.random.get_rng_state().clone()
    model = HistoryInitializedDense(984109, mode, dtype=dtype)
    assert torch.equal(rng, torch.random.get_rng_state())
    count = 590 if mode == 'last_two' else 962
    size = 4 if dtype == torch.float32 else 8
    spec = model.model_spec()
    assert model.parameter_count == spec['parameter_count'] == count
    assert sum(p.numel() for p in model.parameters()) == count
    assert spec['parameter_bytes'] == count * size
    assert spec['added_parameter_count'] == count - 590
    assert model.state_scalars == spec['state_scalars'] == 12
    assert spec['state_bytes_per_stream'] == 12 * size and spec['buffer_bytes'] == 0
    assert spec['structurally_inactive_parameter_count'] == 0
    assert spec['persistent_cache'] is False and spec['retained_trajectory'] is False
    if dtype == torch.float32:
        assert spec['parameter_bytes'] + spec['state_bytes_per_stream'] + 192 == (2600 if mode == 'last_two' else 4088)
    before = [(id(module), set(vars(module))) for module in model.modules()]
    q, u, future = histories(dtype)
    first = model(future, model.condition(q, u))[0].detach().clone()
    assert before == [(id(module), set(vars(module))) for module in model.modules()]
    assert not list(model.buffers())
    for module in model.modules():
        assert not any(isinstance(v, torch.Tensor) for v in vars(module).values())
    with torch.no_grad():
        model.cell.input_matrix.add_(.05)
    assert not torch.equal(first, model(future, model.condition(q, u))[0])
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('mode', AFFINE)
def test_feature_output_is_owned_graph_attached_and_head_changes_are_not_cached(mode):
    model = HistoryInitializedDense(984110, mode, dtype=torch.float64)
    q, u, future = [v.requires_grad_() for v in histories()]
    features = model.initialization_features(q, u)
    assert features.requires_grad and features.data_ptr() not in {q.data_ptr(), u.data_ptr()}
    before = [v.detach().clone() for v in (q, u)]
    with torch.no_grad():
        features.add_(100)
    assert torch.equal(q, before[0]) and torch.equal(u, before[1])
    expected = model(future, model.condition(q, u))[0].detach().clone()
    with torch.no_grad():
        model.head.bias.add_(.1)
    assert not torch.equal(expected, model(future, model.condition(q, u))[0])


@pytest.mark.parametrize('kwargs', [{'initializer': 'observer'}, {'initializer': ''}, {'seed': True},
                                   {'seed': -1}, {'seed': 2**63}, {'dtype': torch.float16}])
def test_constructor_rejects_undeclared_modes_seeds_and_dtypes(kwargs):
    with pytest.raises(ValueError):
        HistoryInitializedDense(**kwargs)


@pytest.mark.parametrize('corruption', ['short', 'long', 'empty', 'channels', 'torque_shape',
                                      'dtype', 'q_nan', 'u31_inf'])
def test_context_shape_dtype_and_finite_guards(corruption):
    model = HistoryInitializedDense(984111, 'temporal_affine', dtype=torch.float64)
    q, u, _ = histories()
    if corruption == 'short':
        q, u = q[:, :31], u[:, :31]
    elif corruption == 'long':
        q, u = torch.cat((q, q[:, :1]), 1), torch.cat((u, u[:, :1]), 1)
    elif corruption == 'empty':
        q, u = q[:0], u[:0]
    elif corruption == 'channels':
        q, u = q[:, :, :5], u[:, :, :5]
    elif corruption == 'torque_shape':
        u = u[:1]
    elif corruption == 'dtype':
        q = q.float()
    elif corruption == 'q_nan':
        q[:, 0] = float('nan')
    else:
        u[:, 31] = float('inf')
    with pytest.raises(ValueError):
        model.condition(q, u)


@pytest.mark.parametrize('mode', MODES)
def test_finite_inputs_with_nonfinite_feature_or_difference_are_not_repaired(mode):
    model = HistoryInitializedDense(984112, mode, dtype=torch.float64)
    q, u, _ = histories()
    if mode == 'last_two':
        q[:, 30], q[:, 31] = -1e308, 1e308
    elif mode == 'local_affine':
        q[:, 30], q[:, 31] = 0, 1e200
    else:
        q[:, 0] = 1e308
    with pytest.raises(ValueError):
        model.condition(q, u)


@pytest.mark.parametrize('corruption', ['head_nan', 'head_shape', 'rogue_parameter', 'rogue_buffer', 'wrong_cell'])
def test_parameter_and_buffer_roster_guards(corruption):
    model = HistoryInitializedDense(984113, 'local_affine', dtype=torch.float64)
    if corruption == 'head_nan':
        with torch.no_grad():
            model.head.bias[0] = float('nan')
    elif corruption == 'head_shape':
        model.head.weight = torch.nn.Parameter(torch.zeros(12, 29, dtype=torch.float64))
    elif corruption == 'rogue_parameter':
        model.register_parameter('extra', torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
    elif corruption == 'rogue_buffer':
        model.register_buffer('cache', torch.zeros(1, dtype=torch.float64))
    else:
        model.cell = StructuredRobotTransition('dense_bounded', 984113, dtype=torch.float64)
    with pytest.raises(ValueError):
        model.condition(*histories()[:2])


def test_baseline_has_no_features_and_forecast_preserves_guards():
    model = HistoryInitializedDense(984114, 'last_two', dtype=torch.float64)
    q, u, future = histories()
    with pytest.raises(ValueError):
        model.initialization_features(q, u)
    state = model.condition(q, u)
    for bad in (future.float(), future[:, :, :5], torch.full_like(future, float('nan'))):
        with pytest.raises(ValueError):
            model(bad, state)
    with pytest.raises(ValueError):
        model(future, state[:, :11])
