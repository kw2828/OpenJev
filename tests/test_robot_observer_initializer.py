"""Fabricated observer qualification; no measurements, checkpoints or training run."""
import inspect

import numpy as np
import pytest
import torch

from openjev.research.robot_history_initializer import HistoryInitializedDense
from openjev.research.robot_observer_initializer import MODES, FrozenObserverInitializer


def inputs(dtype=torch.float64, batch=2, horizon=5):
    q = torch.arange(batch*32*6, dtype=dtype).reshape(batch, 32, 6)/500
    u = torch.sin(torch.arange(batch*32*6, dtype=dtype).reshape(batch, 32, 6)/21)/4
    future = torch.cos(torch.arange(batch*horizon*6, dtype=dtype).reshape(batch, horizon, 6)/13)/5
    return q, u, future


def cell_arrays(model):
    return {key: value.detach().clone() for key, value in model.cell.state_dict().items()}


def coupled_cell(model):
    """A fabricated stable matrix with all state coordinates influencing q."""
    with torch.no_grad():
        raw = .45*torch.eye(12, dtype=model.cell.raw_matrix.dtype)+.015*torch.ones_like(model.cell.raw_matrix[0])
        model.cell.raw_matrix.copy_(raw.repeat(2, 1, 1))
        model.cell.scale_raw.zero_()


@pytest.mark.parametrize('mode,total,trainable,buffers', [
    ('last_two', 590, 0, 0), ('local_affine', 962, 372, 0), ('temporal_affine', 962, 372, 0),
    ('observer_fixed', 590, 0, 72), ('observer_learned', 662, 72, 0), ('observer_zero', 590, 0, 72)])
@pytest.mark.parametrize('dtype,bytes_per_scalar', [(torch.float32, 4), (torch.float64, 8)])
def test_exact_numeric_state_and_frozen_trainable_storage(mode, total, trainable, buffers, dtype, bytes_per_scalar):
    model = FrozenObserverInitializer(11, mode, dtype=dtype)
    spec = model.model_spec()
    assert model.parameter_count == total and model.trainable_parameter_count == trainable
    assert model.frozen_parameter_count == 590 and model.state_scalars == 12
    assert spec['parameter_bytes'] == total*bytes_per_scalar
    assert spec['trainable_parameter_bytes'] == trainable*bytes_per_scalar
    assert spec['frozen_parameter_bytes'] == 590*bytes_per_scalar
    assert spec['buffer_bytes'] == buffers*bytes_per_scalar and spec['state_bytes_per_stream'] == 12*bytes_per_scalar
    assert spec['gain_buffer_scalars'] == buffers
    assert spec['persistent_cache'] is False and spec['retained_trajectory'] is False
    assert all(not p.requires_grad and p.grad is None for p in model.cell.parameters())
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert names == ({'head.weight', 'head.bias'} if mode.endswith('affine') else {'gain'} if mode == 'observer_learned' else set())


@pytest.mark.parametrize('mode', ['last_two', 'local_affine', 'temporal_affine'])
def test_existing_affine_and_last_two_functions_and_loaded_cell_mapping_are_unchanged(mode):
    q, u, future = inputs()
    old = HistoryInitializedDense(13, mode, dtype=torch.float64)
    model = FrozenObserverInitializer(13, mode, dtype=torch.float64)
    assert list(model.state_dict()) == list(old.state_dict())
    for key, value in old.state_dict().items(): assert torch.equal(model.state_dict()[key], value)
    with torch.no_grad(): old.cell.raw_matrix.mul_(.7)
    model.cell.load_state_dict(old.cell.state_dict(), strict=True)
    if mode != 'last_two':
        with torch.no_grad():
            values = torch.arange(360, dtype=torch.float64).reshape(12, 30)/10000
            old.head.weight.copy_(values); model.head.weight.copy_(values)
    assert torch.equal(model.condition(q, u), old.condition(q, u))
    actual = model(future, model.condition(q, u)); expected = old(future, old.condition(q, u))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert all(not p.requires_grad for p in model.cell.parameters())


def test_initial_equivalences_are_within_observer_pair_and_affine_pair_only():
    q, u, future = inputs()
    models = {mode: FrozenObserverInitializer(3, mode, dtype=torch.float64) for mode in MODES}
    states = {mode: model.condition(q, u) for mode, model in models.items()}
    assert torch.equal(states['last_two'], states['local_affine'])
    assert torch.equal(states['last_two'], states['temporal_affine'])
    assert torch.equal(states['observer_fixed'], states['observer_learned'])
    assert not torch.equal(states['last_two'], states['observer_fixed'])
    assert not torch.equal(states['observer_fixed'], states['observer_zero'])
    left = models['observer_fixed'](future, states['observer_fixed'])
    right = models['observer_learned'](future, states['observer_learned'])
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


@pytest.mark.parametrize('mode', ['observer_fixed', 'observer_learned', 'observer_zero'])
def test_thirty_nonzero_innovation_steps_match_independent_linear_hand_oracle(mode):
    q, u, _ = inputs(batch=1)
    model = FrozenObserverInitializer(5, mode, dtype=torch.float64)
    gain = np.zeros((12, 6)) if mode == 'observer_zero' else np.concatenate((np.eye(6), np.eye(6)))
    if mode == 'observer_learned':
        gain = np.concatenate((.7*np.eye(6), .4*np.eye(6)))
        gain[7, 0] = -.2
    forcing = np.concatenate((.25*np.eye(6), -.125*np.eye(6)))
    with torch.no_grad():
        model.cell.raw_matrix.copy_((.5/.9999)*torch.eye(12, dtype=torch.float64).repeat(2, 1, 1))
        model.cell.scale_raw.zero_(); model.cell.expert_bias.zero_()
        model.cell.input_matrix.copy_(torch.from_numpy(forcing).repeat(2, 1, 1))
        if mode == 'observer_learned': model.gain.copy_(torch.from_numpy(gain))
    qn, un = q.numpy()[0], u.numpy()[0]
    expected = np.concatenate((qn[1], qn[1]-qn[0]))
    innovations = []
    for t in range(2, 32):
        prior = .5*expected + forcing @ un[t-1]
        innovation = qn[t]-prior[:6]
        innovations.append(innovation)
        expected = prior+gain@innovation
    assert np.any(np.abs(innovations) > .01)
    np.testing.assert_allclose(model.condition(q, u).detach().numpy()[0], expected, rtol=1e-12, atol=1e-12)
    if mode == 'observer_fixed':
        torch.testing.assert_close(model.condition(q, u)[:, :6], q[:, 31], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('mode', MODES)
def test_boundary_torque_u31_is_excluded_from_condition_but_used_in_forecast(mode):
    model = FrozenObserverInitializer(17, mode, dtype=torch.float64)
    q, u, future = inputs()
    other = u.clone(); other[:, 31] += 100
    assert torch.equal(model.condition(q, u), model.condition(q, other))
    state = model.condition(q, u)
    changed = future.clone(); changed[:, 0] += .5
    assert not torch.equal(model(future, state)[0][:, 0], model(changed, state)[0][:, 0])


def test_prefix_update_is_causal_and_prepares_operator_only_once(monkeypatch):
    model = FrozenObserverInitializer(19, 'observer_learned', dtype=torch.float64)
    q, u, _ = inputs(batch=1)
    original_prepare, original_step = model.cell._prepare, model.cell._step
    preparations, states, torques = [], [], []
    def prepare(): preparations.append(1); return original_prepare()
    def step(state, torque, prepared):
        states.append(state.detach().clone()); torques.append(torque.detach().clone())
        return original_step(state, torque, prepared)
    monkeypatch.setattr(model.cell, '_prepare', prepare); monkeypatch.setattr(model.cell, '_step', step)
    model.condition(q, u)
    assert len(preparations) == 1 and len(states) == 30
    for t, torque in enumerate(torques, 1): assert torch.equal(torque, u[:, t])
    baseline = states.copy(); states.clear(); torques.clear(); preparations.clear()
    changed = q.clone(); changed[:, 10] += .4
    model.condition(changed, u)
    assert all(torch.equal(a, b) for a, b in zip(baseline[:9], states[:9], strict=True))
    assert not torch.equal(baseline[9], states[9])


@pytest.mark.parametrize('mode', ['local_affine', 'temporal_affine', 'observer_learned'])
def test_only_initializer_gets_gradients_and_an_update_leaves_cell_exact(mode):
    model = FrozenObserverInitializer(23, mode, dtype=torch.float64)
    coupled_cell(model)
    q, u, future = inputs(); q.requires_grad_(); u.requires_grad_(); future.requires_grad_()
    frozen = cell_arrays(model)
    initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    prediction, final = model(future, model.condition(q, u))
    (prediction.square().sum()+.2*final.square().sum()).backward()
    assert all(p.grad is None and not p.requires_grad for p in model.cell.parameters())
    learned = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in learned)
    assert q.grad is not None
    if mode == 'observer_learned': assert q.grad[:, :2].abs().sum() > 0
    assert future.grad is not None and future.grad.abs().sum() > 0
    with torch.no_grad():
        for p in learned: p.add_(p.grad, alpha=-1e-4)
    assert any(not torch.equal(initial[name], p) for name, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(value, model.cell.state_dict()[name]) for name, value in frozen.items())


@pytest.mark.parametrize('index', [(0, 0), (6, 3), (11, 5)])
def test_gain_gradient_traverses_later_prefix_and_forecast_steps(index):
    model = FrozenObserverInitializer(29, 'observer_learned', dtype=torch.float64)
    coupled_cell(model)
    q, u, future = inputs(batch=1, horizon=4)
    def loss():
        prediction, _ = model(future, model.condition(q, u))
        return prediction[:, -1].square().sum()
    derivative = torch.autograd.grad(loss(), model.gain)[0][index].item()
    epsilon = 1e-6
    original = model.gain[index].item()
    with torch.no_grad(): model.gain[index] = original+epsilon
    plus = loss().item()
    with torch.no_grad(): model.gain[index] = original-epsilon
    minus = loss().item()
    with torch.no_grad(): model.gain[index] = original
    assert abs(derivative) > 1e-9
    assert derivative == pytest.approx((plus-minus)/(2*epsilon), rel=2e-5, abs=2e-8)


@pytest.mark.parametrize('mode', MODES)
def test_chunking_empty_horizon_ownership_and_request_reset(mode):
    model = FrozenObserverInitializer(31, mode, dtype=torch.float64)
    q, u, future = inputs()
    q_before, u_before = q.clone(), u.clone()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = model.condition(q, u)
    whole, final = model(future, state)
    one, one_state = model.step(state, future[:, 0])
    first_one, first_state = model(future[:, :1], state)
    assert torch.equal(one, first_one[:, 0]) and torch.equal(one_state, first_state)
    first, middle = model(future[:, :2], state)
    last, chunk_final = model(future[:, 2:], middle)
    assert torch.equal(whole, torch.cat((first, last), 1)) and torch.equal(final, chunk_final)
    empty, empty_final = model(future[:, :0], state)
    assert empty.shape == (2, 0, 6) and torch.equal(empty_final, state)
    assert empty_final.data_ptr() != state.data_ptr()
    again = model.condition(q, u)
    assert torch.equal(state, again) and state.data_ptr() != again.data_ptr()
    with torch.no_grad(): state[0, 0] += 1
    assert torch.equal(q, q_before) and torch.equal(u, u_before)
    assert torch.equal(again, model.condition(q, u))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize('damage', ['q_shape', 'empty_batch', 'u_dtype', 'u31_nan', 'q_inf', 'gain_shape', 'gain_nan',
                                   'fixed_gain', 'unfrozen_cell', 'stored_cell_grad', 'extra_buffer'])
def test_input_and_parameter_errors_are_rejected_without_repairs(damage):
    mode = 'observer_fixed' if damage == 'fixed_gain' else 'observer_learned'
    model = FrozenObserverInitializer(37, mode, dtype=torch.float64)
    q, u, _ = inputs()
    if damage == 'q_shape': q = q[:, :-1]
    elif damage == 'empty_batch': q, u = q[:0], u[:0]
    elif damage == 'u_dtype': u = u.float()
    elif damage == 'u31_nan': u[:, 31, 0] = torch.nan
    elif damage == 'q_inf': q[:, 1, 0] = torch.inf
    elif damage == 'gain_shape': model.gain = torch.nn.Parameter(torch.zeros((6, 12), dtype=torch.float64))
    elif damage == 'gain_nan':
        with torch.no_grad(): model.gain[0, 0] = torch.nan
    elif damage == 'fixed_gain': model.gain[0, 0] = .5
    elif damage == 'unfrozen_cell': model.cell.input_matrix.requires_grad_(True)
    elif damage == 'stored_cell_grad': model.cell.input_matrix.grad = torch.zeros_like(model.cell.input_matrix)
    else: model.register_buffer('hidden_cache', torch.zeros(1))
    with pytest.raises(ValueError): model.condition(q, u)


def test_nonfinite_correction_is_reported_not_clipped_or_ignored():
    model = FrozenObserverInitializer(41, 'observer_learned', dtype=torch.float64)
    q, u, _ = inputs(batch=1)
    with torch.no_grad(): model.gain.fill_(torch.finfo(torch.float64).max)
    q[:, 2] = 100
    with pytest.raises(ValueError, match='nonfinite structured transition output; no repair'):
        model.condition(q, u)
    assert torch.all(model.gain == torch.finfo(torch.float64).max)


@pytest.mark.parametrize('mode', MODES)
def test_local_rng_isolation_owned_models_and_public_only_signature(mode):
    before = torch.random.get_rng_state().clone()
    first = FrozenObserverInitializer(43, mode, dtype=torch.float64)
    second = FrozenObserverInitializer(43, mode, dtype=torch.float64)
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(a, second.state_dict()[name]) and a.data_ptr() != second.state_dict()[name].data_ptr()
               for name, a in first.state_dict().items())
    assert list(inspect.signature(first.condition).parameters) == ['q_context', 'u_context']
    assert list(inspect.signature(first.forward).parameters) == ['future_u', 'state']
    assert list(inspect.signature(first.step).parameters) == ['state', 'inputs']


@pytest.mark.parametrize('kwargs', [{'mode': 'kalman'}, {'seed': True}, {'seed': -1}, {'dtype': torch.int64}])
def test_constructor_rejects_undeclared_identity(kwargs):
    with pytest.raises(ValueError): FrozenObserverInitializer(**kwargs)
