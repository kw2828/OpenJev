"""Independent fabricated reflection-capacity qualification; no measured inputs."""
import inspect

import numpy as np
import pytest
import torch

from openjev.research.reflection_capacity import ReflectionCapacityRobotTransition
from openjev.research.structured_robot_transition import StructuredRobotTransition

ORACLE_TOL = 2e-11
RADIUS = .9999
ANCHORS = (0, 0, 6, 6) * 3


def inputs(dtype=torch.float64, horizon=6, batch=2):
    q = .3*torch.sin(torch.arange(batch*5*6, dtype=dtype).reshape(batch, 5, 6)/11)
    u = .4*torch.cos(torch.arange(batch*5*6, dtype=dtype).reshape(batch, 5, 6)/13)
    future = .2*torch.sin(torch.arange(batch*horizon*6, dtype=dtype).reshape(batch, horizon, 6)/7+.2)
    return q, u, future


def activate(model):
    """Unequal reflectors, experts and a nonconstant gate, without any fit."""
    dtype = model.input_matrix.dtype
    with torch.no_grad():
        raw = model.reflection_raw
        raw.add_(.18*torch.sin(torch.arange(raw.numel(), dtype=dtype).reshape(raw.shape)/3+.1))
        model.decay_raw.sub_(torch.linspace(.1, 1.2, 24, dtype=dtype).reshape(2, 12))
        model.scale_raw.add_(torch.linspace(-.2, .2, 12, dtype=dtype))
        model.expert_bias.copy_(.02*torch.cos(torch.arange(24, dtype=dtype).reshape(2, 12)))
        model.gate.head_weight.copy_(.3*torch.sin(torch.arange(16, dtype=dtype).reshape(2, 8)/2))
        model.gate.head_bias.copy_(torch.tensor([.15, -.1], dtype=dtype))


def matrix_step(parameters, state, torque, count):
    """NumPy dense-matrix product plus independent zero-history GRU algebra."""
    sigmoid = lambda value: 1/(1+np.exp(-value))
    diagonal = np.exp(3*np.tanh(parameters['scale_raw']))
    result = []
    for x, u in zip(state, torque, strict=True):
        feature = np.concatenate((x[:6], u))
        affine = parameters['gate.input_weight'] @ feature
        reset = sigmoid(affine[:8] + parameters['gate.reset_update_bias'][:8])
        update = sigmoid(affine[8:16] + parameters['gate.reset_update_bias'][8:])
        candidate = np.tanh(affine[16:] + parameters['gate.candidate_input_bias']
                            + reset*parameters['gate.candidate_hidden_bias'])
        hidden = (1-update)*candidate
        logits = parameters['gate.head_weight']@hidden + parameters['gate.head_bias']
        weights = np.exp(logits-logits.max()); weights /= weights.sum()
        operator = np.eye(12)
        for k in range(count):
            raw = np.tanh(parameters['reflection_raw'][:, k])
            prototypes = np.array([np.insert(row, ANCHORS[k], 1.) for row in raw])
            vector = weights @ prototypes
            operator = (np.eye(12) - 2*np.outer(vector, vector)/(vector@vector)) @ operator
        decay = weights @ (RADIUS*sigmoid(parameters['decay_raw']))
        forcing = sum(weights[e]*(parameters['input_matrix'][e]@u + parameters['expert_bias'][e])
                      for e in range(2))
        result.append((np.diag(decay)@operator@np.diag(diagonal)@x + forcing)/diagonal)
    return np.stack(result)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('active', [False, True])
def test_four_is_exact_old_function_and_loaded_parameter_gradient_path(dtype, active):
    old = StructuredRobotTransition('householder', 198301, dtype=dtype)
    new = ReflectionCapacityRobotTransition(4, 198301, dtype=dtype)
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    if active:
        activate(old)
        new.load_state_dict(old.state_dict(), strict=True)
    q, u, future = inputs(dtype)
    old_state = old.condition(q, u).detach().requires_grad_()
    new_state = new.condition(q, u).detach().requires_grad_()
    left_u, right_u = future.clone().requires_grad_(), future.clone().requires_grad_()
    left, left_final = old(left_u, old_state)
    right, right_final = new(right_u, new_state)
    assert torch.equal(left, right) and torch.equal(left_final, right_final)
    weight = torch.linspace(.1, 1., left.numel(), dtype=dtype).reshape(left.shape)
    ((left*weight).sum() + left_final.square().sum()).backward()
    ((right*weight).sum() + right_final.square().sum()).backward()
    assert torch.equal(left_u.grad, right_u.grad) and torch.equal(old_state.grad, new_state.grad)
    assert all(torch.equal(p.grad, dict(new.named_parameters())[name].grad) for name, p in old.named_parameters())


@pytest.mark.parametrize('dtype,tol', [(torch.float32, 4e-6), (torch.float64, 2e-12)])
def test_twelve_initialization_pairs_old_prefix_and_new_local_rng(dtype, tol):
    seed = 198302
    previous = torch.random.get_rng_state().clone()
    old = StructuredRobotTransition('householder', seed, dtype=dtype)
    new = ReflectionCapacityRobotTransition(12, seed, dtype=dtype)
    assert torch.equal(previous, torch.random.get_rng_state())
    for name, value in old.state_dict().items():
        actual = new.state_dict()[name]
        assert torch.equal(value, actual[:, :4] if name == 'reflection_raw' else actual)
    rng = torch.Generator(device='cpu').manual_seed(seed ^ 0x43415031)
    expected = .01*torch.randn((2, 4, 11), generator=rng, dtype=dtype)
    assert torch.equal(new.reflection_raw[:, 4:], expected.repeat_interleave(2, 1))
    assert new.reflection_raw.data_ptr() != old.reflection_raw.data_ptr()
    mixture = torch.tensor([[.5, .5], [1., 0.], [.2, .8]], dtype=dtype)
    actual = new.operators(mixture)
    torch.testing.assert_close(actual['matrix'], .999*torch.eye(12, dtype=dtype).expand(3, 12, 12), rtol=tol, atol=tol)
    for index, anchor in enumerate(ANCHORS):
        torch.testing.assert_close(actual['reflection_vectors'][:, index, anchor], torch.ones(3, dtype=dtype), rtol=0, atol=1e-7 if dtype == torch.float32 else 1e-14)
    q, u, future = inputs(dtype, horizon=3)
    a = old(future, old.condition(q, u))
    b = new(future, new.condition(q, u))
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=tol, atol=tol)


@pytest.mark.parametrize('count', [4, 12])
def test_full_materialized_matrix_oracle_over_multiple_gated_steps(count):
    model = ReflectionCapacityRobotTransition(count, 198303, dtype=torch.float64)
    activate(model)
    q, u, future = inputs(horizon=7)
    initial = np.concatenate((q.numpy()[:, -1], q.numpy()[:, -1]-q.numpy()[:, -2]), axis=1)
    parameters = {name: value.detach().numpy().copy() for name, value in model.named_parameters()}
    state, predictions = initial, []
    for t in range(7):
        state = matrix_step(parameters, state, future[:, t].numpy(), count)
        predictions.append(state[:, :6].copy())
    actual, final = model(future, model.condition(q, u))
    np.testing.assert_allclose(actual.detach().numpy(), np.stack(predictions, 1), rtol=ORACLE_TOL, atol=ORACLE_TOL)
    np.testing.assert_allclose(final.detach().numpy(), state, rtol=ORACLE_TOL, atol=ORACLE_TOL)


@pytest.mark.parametrize('count', [4, 12])
def test_common_metric_operator_and_forced_state_bound(count):
    model = ReflectionCapacityRobotTransition(count, 198304, dtype=torch.float64)
    activate(model)
    mixing = torch.tensor([[1., 0.], [.3, .7], [0., 1.]], dtype=torch.float64)
    diagnostics = model.operators(mixing)
    assert bool((torch.linalg.svdvals(diagnostics['matrix'])[:, 0] <= RADIUS+2e-12).all())
    q, u, future = inputs(horizon=4)
    state = model.condition(q, u)
    scale = torch.exp(3*torch.tanh(model.scale_raw))
    for t in range(4):
        _, result = model.step(state, future[:, t])
        forces = torch.stack([future[:, t]@model.input_matrix[e].T+model.expert_bias[e] for e in range(2)], dim=1)
        upper = RADIUS*torch.linalg.vector_norm(state*scale, dim=1) + torch.linalg.vector_norm(forces, dim=2).max(1).values
        assert bool((torch.linalg.vector_norm(result*scale, dim=1) <= upper+2e-12).all())
        state = result
    assert 'not incremental' in model.model_spec()['stability_scope']


@pytest.mark.parametrize('name,index', [('reflection_raw', (1, 11, 7)), ('reflection_raw', (0, 4, 2)),
                                      ('decay_raw', (0, 8)), ('scale_raw', (9,)),
                                      ('input_matrix', (1, 2, 3)), ('gate.head_weight', (0, 3))])
def test_twelve_late_output_parameter_finite_differences(name, index):
    model = ReflectionCapacityRobotTransition(12, 198305, dtype=torch.float64)
    activate(model)
    q, u, future = inputs(horizon=4, batch=1)
    parameter = dict(model.named_parameters())[name]
    weights = torch.linspace(.2, 1.1, 12, dtype=torch.float64)
    def objective():
        output, final = model(future, model.condition(q, u))
        return (final*weights).sum() + output[:, -1].square().sum()
    objective().backward()
    actual = parameter.grad[index].item()
    center = parameter[index].item()
    with torch.no_grad():
        parameter[index] = center + 1e-6
        plus = objective().item()
        parameter[index] = center - 1e-6
        minus = objective().item()
        parameter[index] = center
    assert actual == pytest.approx((plus-minus)/2e-6, rel=2e-5, abs=2e-8)


def test_twelve_late_step_input_and_state_gradcheck_and_active_extra_gradients():
    model = ReflectionCapacityRobotTransition(12, 198306, dtype=torch.float64)
    activate(model)
    q, u, future = inputs(horizon=3, batch=1)
    state = model.condition(q, u).detach().requires_grad_()
    future = future.requires_grad_()
    def late(inputs, initial):
        output, final = model(inputs, initial)
        return torch.cat((output[:, -1], final), dim=1)
    assert torch.autograd.gradcheck(late, (future, state), eps=1e-6, atol=2e-7, rtol=2e-5)
    (late(future, state)*torch.linspace(.1, 1.1, 18, dtype=torch.float64)).sum().backward()
    assert bool((model.reflection_raw.grad[:, 4:].abs().sum((0, 2)) > 1e-10).all())
    assert model.gate.input_weight.grad.abs().sum() > 1e-10
    assert model.gate.head_weight.grad.abs().sum() > 1e-10
    assert future.grad[:, 0].abs().sum() > 1e-10
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


def test_each_extra_reflector_changes_the_function_when_perturbed():
    model = ReflectionCapacityRobotTransition(12, 198307, dtype=torch.float64)
    q, u, future = inputs(horizon=2, batch=1)
    state = model.condition(q, u)
    before = model(future, state)[1].detach().clone()
    for k in range(4, 12):
        with torch.no_grad():
            center = model.reflection_raw[0, k, 1].item()
            model.reflection_raw[0, k, 1] += .1
        after = model(future, state)[1].detach()
        assert (after-before).abs().max() > 1e-8
        with torch.no_grad(): model.reflection_raw[0, k, 1] = center


@pytest.mark.parametrize('count', [4, 12])
def test_causal_context_chunking_empty_horizon_and_no_output_aliases(count):
    model = ReflectionCapacityRobotTransition(count, 198308, dtype=torch.float64)
    activate(model)
    q, u, future = inputs(horizon=6)
    state = model.condition(q, u)
    q[:, :-2] += 100
    assert torch.equal(state, model.condition(q, u+900))
    saved = state.clone()
    full, final = model(future, state)
    first, carry = model(future[:, :2], state)
    second, chunk_final = model(future[:, 2:], carry)
    assert torch.equal(full, torch.cat((first, second), 1)) and torch.equal(final, chunk_final)
    changed = future.clone(); changed[:, 2:] += 7
    assert torch.equal(full[:, :2], model(changed, state)[0][:, :2])
    empty, unchanged = model(future[:, :0], state)
    assert empty.shape == (2, 0, 6) and torch.equal(unchanged, state)
    unchanged.detach().add_(100)
    assert torch.equal(state, saved)
    final_before = final.clone()
    full.detach().fill_(100)
    assert torch.equal(final, final_before)
    assert tuple(inspect.signature(model.forward).parameters) == ('future_u', 'state')


@pytest.mark.parametrize('count,parameters', [(4, 630), (12, 806)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_storage_exact_roster_rng_and_no_prepared_cache(count, parameters, dtype):
    before = torch.random.get_rng_state().clone()
    model = ReflectionCapacityRobotTransition(count, 198309, dtype=dtype)
    assert torch.equal(before, torch.random.get_rng_state())
    spec = model.model_spec()
    assert model.parameter_count == spec['parameter_count'] == parameters
    assert spec['parameter_bytes'] == parameters*torch.empty((), dtype=dtype).element_size()
    assert spec['state_scalars'] == 12 and spec['state_bytes_per_stream'] == 12*torch.empty((), dtype=dtype).element_size()
    assert spec['buffer_bytes'] == 0 and not list(model.buffers()) and not spec['persistent_cache']
    assert spec['structurally_inactive_parameter_count'] == 0 and spec['reflections'] == count
    assert tuple(model.reflection_raw.shape) == (2, count, 11)


@pytest.mark.parametrize('count', [4, 12])
def test_validation_and_preparation_once_per_rollout(count, monkeypatch):
    model = ReflectionCapacityRobotTransition(count, 198310, dtype=torch.float64)
    q, u, future = inputs()
    state = model.condition(q, u)
    calls = {'validation': 0, 'preparation': 0}
    validation, preparation = model._validate_parameters, model._prepare
    def validate():
        calls['validation'] += 1
        return validation()
    def prepare():
        calls['preparation'] += 1
        return preparation()
    monkeypatch.setattr(model, '_validate_parameters', validate)
    monkeypatch.setattr(model, '_prepare', prepare)
    monkeypatch.setattr(model.gate, 'forward', lambda *a: pytest.fail('no per-step public gate validation'))
    model(future, state)
    assert calls == {'validation': 1, 'preparation': 1}


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'state', 'nan_input', 'parameter', 'buffer', 'overflow'])
def test_input_parameter_and_output_guards(bad):
    model = ReflectionCapacityRobotTransition(12, 198311)
    q, u, future = inputs(torch.float32)
    state = model.condition(q, u)
    if bad == 'dtype': future = future.double()
    elif bad == 'shape': future = future[..., :5]
    elif bad == 'state': state = state[:, :-1]
    elif bad == 'nan_input': future[0, 0, 0] = float('nan')
    elif bad == 'parameter':
        with torch.no_grad(): model.reflection_raw[0, 11, 0] = float('inf')
    elif bad == 'buffer': model.register_buffer('forbidden', torch.zeros(1))
    else:
        with torch.no_grad(): model.input_matrix.fill_(torch.finfo(torch.float32).max)
        future = torch.full_like(future, 10.)
    with pytest.raises(ValueError): model(future, state)


@pytest.mark.parametrize('count,seed,dtype', [(True, 0, torch.float32), (8, 0, torch.float32),
                                            (4.0, 0, torch.float32), (12, True, torch.float32),
                                            (12, -1, torch.float32), (12, 0, torch.float16)])
def test_constructor_guards(count, seed, dtype):
    with pytest.raises(ValueError): ReflectionCapacityRobotTransition(count, seed, dtype=dtype)
