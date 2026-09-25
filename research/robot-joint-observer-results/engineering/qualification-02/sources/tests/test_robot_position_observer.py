"""Fabricated position-observer fixtures only; no measurement or checkpoint IO."""
import math

import numpy as np
import pytest
import torch
from torch import nn

import openjev.research.robot_position_observer as module
from openjev.research.robot_observer_initializer import FrozenObserverInitializer
from openjev.research.robot_position_observer import (
    DIAGNOSTIC_KEYS,
    FrozenPositionObserver,
    gradient_diagnostics,
)


def inputs(dtype=torch.float64, batch=2, horizon=5):
    q = torch.arange(batch*32*6, dtype=dtype).reshape(batch, 32, 6)/500
    u = torch.sin(torch.arange(batch*32*6, dtype=dtype).reshape(batch, 32, 6)/21)/4
    future = torch.cos(torch.arange(batch*horizon*6, dtype=dtype).reshape(batch, horizon, 6)/13)/5
    return q, u, future


def coupled_cell(model):
    with torch.no_grad():
        model.cell.raw_matrix.copy_((.45*torch.eye(12, dtype=model.gain.dtype)+.015*torch.ones_like(model.cell.raw_matrix[0])).repeat(2, 1, 1))
        model.cell.scale_raw.zero_()
        model.cell.gate[2].weight.copy_(torch.arange(16, dtype=model.gain.dtype).reshape(2, 8)/100)


@pytest.mark.parametrize('mode,parameters,buffers,trainable', [('learned', 662, 0, 72), ('fixed', 590, 72, 0)])
@pytest.mark.parametrize('dtype,width', [(torch.float32, 4), (torch.float64, 8)])
def test_exact_cell_gain_storage_and_initialization(mode, parameters, buffers, trainable, dtype, width):
    model = FrozenPositionObserver(11, mode, dtype=dtype)
    spec = model.model_spec()
    assert model.parameter_count == parameters and model.state_scalars == 12
    assert model.added_parameter_count == trainable and model.trainable_parameter_count == trainable
    assert spec['version'] == 'robot-position-observer-v1' and spec['mode'] == mode
    assert spec['gain_initialization'] == '[I;0]'
    assert spec['parameter_bytes'] == width*parameters and spec['buffer_bytes'] == width*buffers
    assert spec['trainable_parameter_bytes'] == width*trainable and spec['frozen_parameter_bytes'] == width*590
    assert spec['gain_buffer_scalars'] == buffers and spec['state_bytes_per_stream'] == width*12
    assert torch.equal(model.gain[:6], torch.eye(6, dtype=dtype)) and torch.count_nonzero(model.gain[6:]) == 0
    assert all(not p.requires_grad and p.grad is None for p in model.cell.parameters())
    assert spec['persistent_cache'] is False and spec['retained_trajectory'] is False
    assert set(model.state_dict()) == {'gain', *('cell.'+n for n in model.cell.state_dict())}


def test_learned_fixed_pair_cell_seed_and_rng_isolation():
    before = torch.random.get_rng_state().clone()
    learned, fixed = FrozenPositionObserver(19), FrozenPositionObserver(19, 'fixed')
    old = FrozenObserverInitializer(19, 'observer_learned')
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(v, old.cell.state_dict()[k]) for k, v in learned.cell.state_dict().items())
    q, u, future = inputs(torch.float32)
    assert torch.equal(learned.condition(q, u), fixed.condition(q, u))
    assert all(torch.equal(a, b) for a, b in zip(learned(future, learned.condition(q, u)), fixed(future, fixed.condition(q, u)), strict=True))
    assert learned.gain.data_ptr() != fixed.gain.data_ptr()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_loading_old_gain_preserves_old_equations_and_all_active_gradients(dtype):
    old = FrozenObserverInitializer(7, 'observer_learned', dtype=dtype)
    coupled_cell(old)
    new = FrozenPositionObserver(7, dtype=dtype)
    new.load_state_dict(old.state_dict(), strict=True)
    original = inputs(dtype, horizon=4)
    histories = [[v.clone().requires_grad_() for v in original] for _ in range(2)]
    outputs, gradients = [], []
    for model, (q, u, future) in zip((old, new), histories, strict=True):
        initial = model.condition(q, u)
        predicted, final = model(future, initial)
        loss = predicted.square().sum()+.2*final.square().sum()
        gradients.append(torch.autograd.grad(loss, (q, u, future, model.gain)))
        outputs.append((initial, predicted, final))
    assert all(torch.equal(a, b) for a, b in zip(*outputs, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(*gradients, strict=True))
    assert all(p.grad is None for p in new.cell.parameters())


@pytest.mark.parametrize('mode', ['fixed', 'learned'])
def test_independent_half_identity_oracle_and_exact_diagnostic_aggregates(mode):
    model = FrozenPositionObserver(5, mode, dtype=torch.float64)
    q, u, _ = inputs(batch=2)
    forcing = np.concatenate((.25*np.eye(6), -.125*np.eye(6)))
    gain = np.concatenate((np.eye(6), np.zeros((6, 6))))
    if mode == 'learned': gain[7, 0] = -.2
    with torch.no_grad():
        model.cell.raw_matrix.copy_((.5/.9999)*torch.eye(12, dtype=torch.float64).repeat(2, 1, 1))
        model.cell.scale_raw.zero_(); model.cell.expert_bias.zero_()
        model.cell.input_matrix.copy_(torch.from_numpy(forcing).repeat(2, 1, 1))
        if mode == 'learned': model.gain.copy_(torch.from_numpy(gain))
    qn, un = q.numpy(), u.numpy()
    state = np.concatenate((qn[:, 1], qn[:, 1]-qn[:, 0]), axis=-1)
    states, innovations = [state.copy()], []
    for t in range(2, 32):
        prior = .5*state+un[:, t-1]@forcing.T
        innovation = qn[:, t]-prior[:, :6]
        state = prior+innovation@gain.T
        states.extend((prior, state)); innovations.append(innovation)
    diagnostics = {}
    observed = model.condition(q, u, diagnostics)
    np.testing.assert_allclose(observed.detach().numpy(), state, rtol=1e-12, atol=1e-12)
    assert set(diagnostics) == set(DIAGNOSTIC_KEYS) and diagnostics['prefix_steps'] == 30
    for name, values in (('state', states), ('innovation', innovations)):
        assert diagnostics['max_'+name+'_abs'] == pytest.approx(max(np.abs(v).max() for v in values), rel=1e-12)
        assert diagnostics['max_'+name+'_norm64'] == pytest.approx(max(np.linalg.norm(v, axis=-1).max() for v in values), rel=1e-12)
    assert all(type(v) in (int, float) for v in diagnostics.values())
    assert not any(isinstance(v, torch.Tensor) for v in diagnostics.values())


def test_optional_diagnostics_leave_values_and_gradients_exact_and_no_cache(monkeypatch):
    model = FrozenPositionObserver(17, dtype=torch.float64); coupled_cell(model)
    before_keys = set(vars(model)); before_state = {k: v.clone() for k, v in model.state_dict().items()}
    original = inputs(horizon=3)
    outputs, grads = [], []
    for diagnostics in (None, {}):
        q, u, future = [v.clone().requires_grad_() for v in original]
        start = model.condition(q, u, diagnostics)
        pred, final = model(future, start)
        grads.append(torch.autograd.grad(pred.square().sum()+final.square().sum(), (q, u, future, model.gain)))
        outputs.append((start, pred, final))
        if diagnostics is not None: assert diagnostics['prefix_steps'] == 30
    assert all(torch.equal(a, b) for a, b in zip(*outputs, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(*grads, strict=True))
    assert set(vars(model)) == before_keys and all(torch.equal(v, model.state_dict()[k]) for k, v in before_state.items())
    def forbidden(*args): raise AssertionError('diagnostic work when disabled')
    monkeypatch.setattr(module, '_prefix_maxima', forbidden)
    model.condition(*original[:2])


def test_prepare_once_thirty_correct_torques_and_prefix_causality(monkeypatch):
    model = FrozenPositionObserver(23, dtype=torch.float64); coupled_cell(model)
    q, u, _ = inputs(batch=1)
    prepare, step = model.cell._prepare, model.cell._step
    preparations, states, torques = [], [], []
    def wrapped_prepare(): preparations.append(1); return prepare()
    def wrapped_step(state, torque, prepared):
        states.append(state.detach().clone()); torques.append(torque.detach().clone()); return step(state, torque, prepared)
    monkeypatch.setattr(model.cell, '_prepare', wrapped_prepare); monkeypatch.setattr(model.cell, '_step', wrapped_step)
    diagnostics = {}; model.condition(q, u, diagnostics)
    assert len(preparations) == 1 and len(states) == diagnostics['prefix_steps'] == 30
    assert all(torch.equal(torque, u[:, t]) for t, torque in enumerate(torques, 1))
    previous = states[:]; states.clear()
    changed = q.clone(); changed[:, 10] += .4
    model.condition(changed, u)
    assert all(torch.equal(a, b) for a, b in zip(previous[:9], states[:9], strict=True))
    assert not torch.equal(previous[9], states[9])


@pytest.mark.parametrize('mode', ['learned', 'fixed'])
def test_u31_exclusion_forecast_chunking_empty_horizon_and_reset(mode):
    model = FrozenPositionObserver(29, mode, dtype=torch.float64)
    q, u, future = inputs(); old_q, old_u = q.clone(), u.clone()
    state = model.condition(q, u)
    changed = u.clone(); changed[:, 31] += 100
    assert torch.equal(state, model.condition(q, changed))
    full, final = model(future, state)
    left, carried = model(future[:, :2], state); right, carried = model(future[:, 2:], carried)
    assert torch.equal(full, torch.cat((left, right), 1)) and torch.equal(final, carried)
    first, step_state = model.step(state, future[:, 0])
    assert torch.equal(first, full[:, 0])
    empty, same = model(future[:, :0], state)
    assert empty.shape == (2, 0, 6) and torch.equal(same, state) and same.data_ptr() != state.data_ptr()
    new = future.clone(); new[:, 0] += .5
    assert not torch.equal(model(new, state)[0][:, 0], first)
    assert torch.equal(state, model.condition(q, u)) and state.data_ptr() != model.condition(q, u).data_ptr()
    assert torch.equal(q, old_q) and torch.equal(u, old_u) and torch.isfinite(step_state).all()


def test_gain_gradient_update_only_changes_gain_and_readonly_diagnostics():
    model = FrozenPositionObserver(31, dtype=torch.float64); coupled_cell(model)
    q, u, future = inputs(); before = {k: v.detach().clone() for k, v in model.cell.state_dict().items()}
    gain = model.gain.detach().clone()
    pred, final = model(future, model.condition(q, u))
    (pred.square().sum()+.2*final.square().sum()).backward()
    assert model.gain.grad is not None and torch.isfinite(model.gain.grad).all() and model.gain.grad.abs().sum() > 0
    raw = model.gain.grad.clone(); summary = gradient_diagnostics(model)
    assert summary['numel'] == 72 and summary['nonfinite_count'] == 0 and summary['all_finite'] is True
    assert torch.equal(raw, model.gain.grad)
    with torch.no_grad(): model.gain.add_(model.gain.grad, alpha=-1e-4)
    assert not torch.equal(gain, model.gain)
    assert all(torch.equal(v, model.cell.state_dict()[k]) for k, v in before.items())
    assert all(not p.requires_grad and p.grad is None for p in model.cell.parameters())


@pytest.mark.parametrize('index', [(0, 0), (6, 3), (11, 5)])
def test_later_forecast_gain_derivative_matches_finite_difference(index):
    model = FrozenPositionObserver(37, dtype=torch.float64); coupled_cell(model)
    q, u, future = inputs(batch=1, horizon=4)
    def loss(): return model(future, model.condition(q, u))[0][:, -1].square().sum()
    derivative = torch.autograd.grad(loss(), model.gain)[0][index].item()
    saved, epsilon = model.gain[index].item(), 1e-6
    with torch.no_grad(): model.gain[index] = saved+epsilon
    plus = loss().item()
    with torch.no_grad(): model.gain[index] = saved-epsilon
    minus = loss().item()
    with torch.no_grad(): model.gain[index] = saved
    assert abs(derivative) > 1e-10
    assert derivative == pytest.approx((plus-minus)/(2*epsilon), rel=3e-5, abs=1e-8)


def test_partial_prefix_diagnostics_preserve_completed_count_without_repair(monkeypatch):
    model = FrozenPositionObserver(41, dtype=torch.float64)
    q, u, _ = inputs(batch=1); step = model.cell._step; calls = []
    def failing(state, torque, prepared):
        calls.append(1)
        if len(calls) == 4:
            bad = torch.full_like(state, float('inf')); return bad[:, :6], bad
        return step(state, torque, prepared)
    monkeypatch.setattr(model.cell, '_step', failing)
    diagnostics = {}
    with pytest.raises(ValueError, match='nonfinite structured transition'):
        model.condition(q, u, diagnostics)
    assert len(calls) == 4 and diagnostics['prefix_steps'] == 3
    assert all(math.isfinite(v) for v in diagnostics.values())
    assert diagnostics['max_innovation_abs'] > 0


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'nan_q', 'inf_u', 'gain_nan', 'gain_shape', 'unfrozen_cell', 'cell_grad', 'extra_buffer', 'fixed_changed'])
def test_malformed_inputs_or_state_fail_without_repair(kind):
    model = FrozenPositionObserver(43, 'fixed' if kind == 'fixed_changed' else 'learned', dtype=torch.float64)
    q, u, _ = inputs()
    if kind == 'shape': q = q[:, :31]
    elif kind == 'dtype': u = u.float()
    elif kind == 'nan_q': q[0, 0, 0] = float('nan')
    elif kind == 'inf_u': u[0, 31, 0] = float('inf')
    elif kind == 'gain_nan':
        with torch.no_grad(): model.gain[0, 0] = float('nan')
    elif kind == 'gain_shape': model.gain = nn.Parameter(torch.zeros(6, 6, dtype=torch.float64))
    elif kind == 'unfrozen_cell': next(model.cell.parameters()).requires_grad_(True)
    elif kind == 'cell_grad':
        parameter = next(model.cell.parameters()); parameter.grad = torch.ones_like(parameter)
    elif kind == 'extra_buffer': model.register_buffer('extra', torch.ones(1, dtype=torch.float64))
    else:
        with torch.no_grad(): model.gain[6, 0] = .1
    with pytest.raises(ValueError): model.condition(q, u)


@pytest.mark.parametrize('diagnostics', [[], {'prefix_steps': 0}, 1])
def test_diagnostics_require_empty_caller_owned_dictionary(diagnostics):
    model = FrozenPositionObserver(); q, u, _ = inputs(torch.float32)
    with pytest.raises(ValueError, match='empty caller-owned'): model.condition(q, u, diagnostics)


@pytest.mark.parametrize('kwargs', [{'mode': 'observer_learned'}, {'mode': None}, {'seed': True}, {'dtype': torch.float16}])
def test_constructor_guards(kwargs):
    with pytest.raises(ValueError): FrozenPositionObserver(**kwargs)


def test_gradient_diagnostics_hand_norm_missing_grad_and_nonfinite_entries():
    model = nn.Module(); model.a = nn.Parameter(torch.zeros(3, dtype=torch.float64)); model.b = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    assert gradient_diagnostics(model) == {'numel': 0, 'nonfinite_count': 0, 'max_abs': 0., 'norm64': 0., 'all_finite': True}
    model.a.grad = torch.tensor([3., 4., 0.], dtype=torch.float64)
    snapshot = model.a.grad.clone()
    assert gradient_diagnostics(model) == {'numel': 3, 'nonfinite_count': 0, 'max_abs': 4., 'norm64': 5., 'all_finite': True}
    assert torch.equal(snapshot, model.a.grad) and model.b.grad is None
    model.b.grad = torch.tensor([float('nan'), float('inf')])
    assert gradient_diagnostics(model) == {'numel': 5, 'nonfinite_count': 2, 'max_abs': None, 'norm64': None, 'all_finite': False}
    assert math.isnan(model.b.grad[0].item()) and math.isinf(model.b.grad[1].item())


def test_finite_float32_entries_can_have_nonfinite_native_clip_norm():
    # Native float32 sum-of-squares overflows although sqrt(72)*1e20 is finite.
    # This is a numerical qualification fixture, never a repair or optimizer step.
    model = nn.Module(); model.gain = nn.Parameter(torch.zeros(72, dtype=torch.float32))
    model.gain.grad = torch.full_like(model.gain, 1e20)
    raw = model.gain.grad.clone(); before = gradient_diagnostics(model)
    assert before['numel'] == 72 and before['nonfinite_count'] == 0 and before['all_finite'] is True
    assert before['max_abs'] == float(raw[0])
    assert before['norm64'] == pytest.approx(math.sqrt(72)*float(raw[0]), rel=1e-15)
    assert torch.equal(raw, model.gain.grad)
    native = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
    assert torch.isinf(native) and math.isfinite(before['norm64'])


def test_gradient_diagnostics_counts_frozen_grads_and_rejects_unsupported_inputs():
    model = nn.Module(); model.a = nn.Parameter(torch.zeros(1), requires_grad=False); model.a.grad = torch.tensor([2.])
    assert gradient_diagnostics(model)['numel'] == 1
    with pytest.raises(ValueError): gradient_diagnostics(object())
    half = nn.Linear(1, 1, bias=False, dtype=torch.float16); half.weight.grad = torch.ones_like(half.weight)
    with pytest.raises(ValueError, match='CPU float32/float64'): gradient_diagnostics(half)
