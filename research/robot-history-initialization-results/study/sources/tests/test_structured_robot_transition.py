"""Fabricated transition qualification only; no measured data or model fitting."""
import inspect

import numpy as np
import pytest
import torch

from openjev.research.structured_robot_transition import (
    ANCHORS,
    KINDS,
    RADIUS,
    StructuredRobotTransition,
)


def fixture(dtype=torch.float64, horizon=7):
    q = torch.sin(torch.arange(2*5*6, dtype=dtype).reshape(2, 5, 6)/13)
    u = torch.cos(torch.arange(2*5*6, dtype=dtype).reshape(2, 5, 6)/11)
    future = torch.sin(torch.arange(2*horizon*6, dtype=dtype).reshape(2, horizon, 6)/17)
    return q, u, future


def activate(model):
    with torch.no_grad():
        if model.kind == 'dense_mlp':
            model.gate[2].weight.copy_(torch.sin(torch.arange(16, dtype=model.input_matrix.dtype).reshape(2, 8))*.2)
        else:
            model.gate.head_weight.copy_(torch.sin(torch.arange(16, dtype=model.input_matrix.dtype).reshape(2, 8))*.2)
        if model.kind == 'householder':
            model.reflection_raw.add_(torch.sin(torch.arange(88, dtype=model.input_matrix.dtype).reshape(2, 4, 11))*.3)
            model.decay_raw[0].sub_(.7)
        else:
            model.raw_matrix.add_(torch.sin(torch.arange(288, dtype=model.input_matrix.dtype).reshape(2, 12, 12))*.03)


def oracle(model, state, inputs):
    """Independent per-batch NumPy matrix law, including reset-GRU algebra."""
    p = {k: v.detach().numpy() for k, v in model.named_parameters()}
    sig = lambda x: 1/(1+np.exp(-x))
    scale = np.exp(3*np.tanh(p['scale_raw']))
    outputs = []
    for x, torque in zip(state, inputs, strict=True):
        feature = np.concatenate((x[:6], torque))
        if model.kind == 'dense_mlp':
            hidden = np.tanh(p['gate.0.weight']@feature+p['gate.0.bias'])
            logits = p['gate.2.weight']@hidden+p['gate.2.bias']
        else:
            affine = p['gate.input_weight']@feature
            rz = sig(affine[:16]+p['gate.reset_update_bias'])
            hidden = (1-rz[8:])*np.tanh(affine[16:]+p['gate.candidate_input_bias']+rz[:8]*p['gate.candidate_hidden_bias'])
            logits = p['gate.head_weight']@hidden+p['gate.head_bias']
        mixing = np.exp(logits-logits.max()); mixing /= mixing.sum()
        if model.kind == 'householder':
            matrix = np.eye(12)
            for k, anchor in enumerate(ANCHORS):
                prototypes = [np.insert(np.tanh(p['reflection_raw'][e, k]), anchor, 1.) for e in range(2)]
                vector = mixing[0]*prototypes[0]+mixing[1]*prototypes[1]
                matrix = (np.eye(12)-2*np.outer(vector, vector)/np.dot(vector, vector))@matrix
            decay = RADIUS*(mixing[0]*sig(p['decay_raw'][0])+mixing[1]*sig(p['decay_raw'][1]))
            matrix = np.diag(decay)@matrix
        else:
            operators = []
            for raw in p['raw_matrix']:
                denominator = max(1., np.linalg.svd(raw, compute_uv=False)[0]) if model.kind != 'dense_unbounded' else 1.
                operators.append(RADIUS*raw/denominator)
            matrix = mixing[0]*operators[0]+mixing[1]*operators[1]
        forcing = sum(mixing[e]*(p['input_matrix'][e]@torque+p['expert_bias'][e]) for e in range(2))
        outputs.append((matrix@(x*scale)+forcing)/scale)
    return np.stack(outputs)


@pytest.mark.parametrize('kind', KINDS)
def test_independent_scalar_gate_and_full_matrix_oracle(kind):
    model = StructuredRobotTransition(kind, 97201, dtype=torch.float64)
    activate(model)
    q, u, future = fixture()
    state = model.condition(q, u)
    independent = np.concatenate((q.numpy()[:, -1], q.numpy()[:, -1]-q.numpy()[:, -2]), axis=1)
    predictions = []
    for t in range(future.shape[1]):
        independent = oracle(model, independent, future[:, t].numpy())
        predictions.append(independent[:, :6].copy())
    actual, final = model(future, state)
    np.testing.assert_allclose(actual.detach().numpy(), np.stack(predictions, axis=1), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(final.detach().numpy(), independent, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('dtype,tolerance', [(torch.float64, 2e-12), (torch.float32, 3e-6)])
def test_initial_functions_forcing_gates_and_metric_are_paired(dtype, tolerance):
    models = [StructuredRobotTransition(kind, 97202, dtype=dtype) for kind in KINDS]
    q, u, future = fixture(dtype)
    reference = None
    for model in models:
        assert torch.equal(model.input_matrix, models[0].input_matrix)
        assert torch.equal(model.expert_bias, torch.zeros_like(model.expert_bias))
        assert torch.equal(model.scale_raw, models[0].scale_raw)
        mixing = torch.tensor([[.5, .5], [0., 1.], [.8, .2]], dtype=dtype)
        diagnostic = model.operators(mixing)
        torch.testing.assert_close(diagnostic['matrix'], .999*torch.eye(12, dtype=dtype).expand(3, 12, 12), atol=tolerance, rtol=tolerance)
        state = model.condition(q, u)
        torch.testing.assert_close(model._mix(state, future[:, 0]), torch.full((2, 2), .5, dtype=dtype), atol=0, rtol=0)
        result = model(future, state)[0]
        if reference is None:
            reference = result
        torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)
    for other in models[1:3]:
        for name, value in models[0].gate.state_dict().items():
            assert torch.equal(value, other.gate.state_dict()[name])


@pytest.mark.parametrize('kind', KINDS)
def test_condition_alignment_future_suffix_and_chunk_state(kind):
    model = StructuredRobotTransition(kind, 97203, dtype=torch.float64)
    activate(model)
    q, u, future = fixture()
    state = model.condition(q, u)
    assert torch.equal(state[:, :6], q[:, -1])
    assert torch.equal(state[:, 6:], q[:, -1]-q[:, -2])
    assert torch.equal(state, model.condition(q, u+500))
    early_q = q.clone(); early_q[:, :-2] += 100
    assert torch.equal(state, model.condition(early_q, u))
    full, final = model(future, state)
    first, carry = model(future[:, :3], state)
    second, chunk_final = model(future[:, 3:], carry)
    torch.testing.assert_close(torch.cat((first, second), 1), full, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(chunk_final, final, atol=1e-12, rtol=1e-12)
    changed = future.clone(); changed[:, 3:] += 5
    assert torch.equal(model(changed, state)[0][:, :3], full[:, :3])
    prediction, _ = model.step(state, future[:, 0])
    assert torch.equal(prediction, full[:, 0])
    assert tuple(inspect.signature(model.forward).parameters) == ('future_u', 'state')


@pytest.mark.parametrize('kind', KINDS)
def test_finite_gradient_paths_and_no_stale_prepared_cache(kind):
    model = StructuredRobotTransition(kind, 97204, dtype=torch.float64)
    activate(model)
    q, u, future = fixture()
    future.requires_grad_()
    out, final = model(future, model.condition(q, u))
    (out.square().mean()+final.square().mean()).backward()
    assert future.grad is not None and torch.isfinite(future.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    before = out.detach().clone()
    with torch.no_grad():
        model.input_matrix.add_(.1)
    assert not torch.allclose(before, model(future.detach(), model.condition(q, u))[0])


@pytest.mark.parametrize('parameter,index', [('reflection_raw', (1, 2, 3)), ('decay_raw', (0, 4)), ('scale_raw', (8,))])
def test_householder_direction_decay_and_metric_finite_differences(parameter, index):
    model = StructuredRobotTransition('householder', 97205, dtype=torch.float64)
    activate(model)
    q, u, future = fixture(horizon=3)
    def value():
        return model(future, model.condition(q, u))[0].square().mean()
    loss = value(); loss.backward()
    tensor = getattr(model, parameter)
    analytic = tensor.grad[index].item()
    center = tensor[index].item()
    with torch.no_grad():
        tensor[index] = center+1e-6
        plus = value().item()
        tensor[index] = center-1e-6
        minus = value().item()
        tensor[index] = center
    assert analytic == pytest.approx((plus-minus)/2e-6, rel=2e-5, abs=2e-8)


@pytest.mark.parametrize('kind', KINDS)
def test_declared_operator_norms_and_forcing_bound(kind):
    model = StructuredRobotTransition(kind, 97206, dtype=torch.float64)
    activate(model)
    with torch.no_grad():
        if kind != 'householder':
            model.raw_matrix.mul_(5)
    mixing = torch.tensor([[.5, .5], [1., 0.], [.01, .99]], dtype=torch.float64)
    diagnostics = model.operators(mixing)
    norms = torch.linalg.matrix_norm(diagnostics['matrix'], ord=2)
    if kind == 'dense_unbounded':
        assert (norms > 1).all()
        assert model.model_spec()['radius_cap'] is None
    else:
        assert (norms <= RADIUS+2e-12).all()
        q, u, future = fixture()
        state = model.condition(q, u)
        _, next_state = model.step(state, future[:, 0])
        scale = diagnostics['scale']
        forced = torch.einsum('eij,bj->bei', model.input_matrix, future[:, 0])+model.expert_bias
        bound = RADIUS*torch.linalg.vector_norm(state*scale, dim=-1)+torch.linalg.vector_norm(forced, dim=-1).max(1).values
        assert (torch.linalg.vector_norm(next_state*scale, dim=-1) <= bound+2e-12).all()
    if kind == 'householder':
        v = diagnostics['reflection_vectors']
        assert (v.square().sum(-1) >= 1-1e-14).all()
        for k, anchor in enumerate(ANCHORS):
            assert torch.equal(v[:, k, anchor], torch.ones(3, dtype=torch.float64))
        independent = mixing @ (RADIUS*torch.sigmoid(model.decay_raw))
        assert torch.equal(independent, diagnostics['decay'])


@pytest.mark.parametrize('kind,count', [('householder', 630), ('dense_bounded', 806), ('dense_unbounded', 806), ('dense_mlp', 590)])
def test_counts_rng_ownership_and_empty_rollout(kind, count):
    before = torch.random.get_rng_state().clone()
    model = StructuredRobotTransition(kind, 97207)
    assert torch.equal(before, torch.random.get_rng_state())
    spec = model.model_spec()
    assert model.parameter_count == spec['parameter_count'] == count
    assert spec['parameter_bytes'] == 4*count and spec['state_scalars'] == 12
    assert spec['state_bytes_per_stream'] == 48 and spec['buffer_bytes'] == 0
    assert not dict(model.named_buffers()) and not spec['persistent_cache']
    q, u, future = fixture(torch.float32)
    state = model.condition(q, u)
    q.add_(9)
    original = state.clone()
    out, final = model(future[:, :0], state)
    assert out.shape == (2, 0, 6) and torch.equal(final, state)
    final.add_(10)
    assert torch.equal(state, original)
    out, final = model(future, state)
    out.detach().fill_(900)
    assert torch.isfinite(final).all() and torch.equal(state, original)


@pytest.mark.parametrize('kind', KINDS)
def test_validation_preparation_once_no_public_gate_in_inner_loop(kind, monkeypatch):
    model = StructuredRobotTransition(kind, 97208, dtype=torch.float64)
    q, u, future = fixture()
    state = model.condition(q, u)
    calls = {'validate': 0, 'prepare': 0, 'svd': 0}
    original_validate, original_prepare = model._validate_parameters, model._prepare
    original_norm = torch.linalg.matrix_norm
    def validate():
        calls['validate'] += 1
        return original_validate()
    def prepare():
        calls['prepare'] += 1
        return original_prepare()
    def norm(*args, **kwargs):
        calls['svd'] += 1
        return original_norm(*args, **kwargs)
    monkeypatch.setattr(model, '_validate_parameters', validate)
    monkeypatch.setattr(model, '_prepare', prepare)
    monkeypatch.setattr(torch.linalg, 'matrix_norm', norm)
    if kind != 'dense_mlp':
        def forbidden(*args):
            raise AssertionError('public compact forward revalidates per step')
        monkeypatch.setattr(model.gate, 'forward', forbidden)
    model(future, state)
    assert calls == {'validate': 1, 'prepare': 1, 'svd': int(kind in ('dense_bounded', 'dense_mlp'))}


@pytest.mark.parametrize('kind', KINDS)
def test_input_and_nonfinite_output_guards(kind):
    model = StructuredRobotTransition(kind, 97209)
    q, u, future = fixture(torch.float32)
    for bad in (q[:, :1], q.double(), torch.full_like(q, float('nan'))):
        with pytest.raises(ValueError):
            model.condition(bad, u)
    state = model.condition(q, u)
    for bad in (future.double(), future[:, :, :5], torch.full_like(future, float('inf'))):
        with pytest.raises(ValueError):
            model(bad, state)
    with torch.no_grad():
        model.input_matrix.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match='nonfinite structured transition output; no repair'):
        model(torch.full_like(future, 10), state)


@pytest.mark.parametrize('kind,seed,dtype', [('bad', 0, torch.float32), ('householder', True, torch.float32),
                                            ('householder', -1, torch.float32), ('householder', 0, torch.float16)])
def test_constructor_guards(kind, seed, dtype):
    with pytest.raises(ValueError):
        StructuredRobotTransition(kind, seed, dtype=dtype)
