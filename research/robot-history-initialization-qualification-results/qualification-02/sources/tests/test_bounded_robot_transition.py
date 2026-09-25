"""Independent fabricated equations, causality and operator-bound qualification."""
import json
import math

import numpy as np
import pytest
import torch

from openjev.research.bounded_robot_transition import KINDS, RADIUS, BoundedLPV, LPVState


def wave(length=9, batch=2, dtype=torch.float64):
    time = torch.arange(length, dtype=dtype)[None, :, None]
    joint = torch.arange(6, dtype=dtype)[None, None, :]
    offset = torch.arange(batch, dtype=dtype)[:, None, None]
    return torch.sin(.23 * time + .17 * joint + .11 * offset) + .2 * torch.cos(.41 * time - .1 * joint)


def model(kind='recurrent', active=True, dtype=torch.float64):
    result = BoundedLPV(kind, 8201, dtype=dtype)
    if active and kind != 'constant':
        with torch.no_grad():
            result.gate.weight.copy_(torch.linspace(-.3, .2, 16, dtype=dtype).reshape(2, 8))
            result.gate.bias.copy_(torch.tensor([-.07, .09], dtype=dtype))
    return result


def gru_scalar(values, inputs, hidden):
    first = values['scheduler.weight_ih'] @ inputs + values['scheduler.bias_ih']
    second = values['scheduler.weight_hh'] @ hidden + values['scheduler.bias_hh']
    result = []
    for j in range(8):
        reset = 1 / (1 + math.exp(-float(first[j] + second[j])))
        update = 1 / (1 + math.exp(-float(first[8+j] + second[8+j])))
        candidate = math.tanh(float(first[16+j] + reset * second[16+j]))
        result.append((1-update) * candidate + update * hidden[j])
    return np.asarray(result)


def independent(values, q, u, future, kind):
    """NumPy/scalar recurrence, no model-private helpers or Torch inference."""
    scale = np.exp(3 * np.tanh(values['scale_raw']))
    matrix = np.stack([RADIUS * value / max(1., np.linalg.svd(value, compute_uv=False)[0])
                       for value in values['raw_matrix']])
    x = np.concatenate((q[-1], q[-1] - q[-2]))
    hidden = np.zeros(8)
    if kind == 'recurrent':
        for t in range(1, len(q)-1):
            hidden = gru_scalar(values, np.concatenate((q[t], u[t])), hidden)
    outputs = []
    for torque in future:
        if kind == 'constant':
            mixing = [.5, .5]
        else:
            hidden = gru_scalar(values, np.concatenate((x[:6], torque)),
                                hidden if kind == 'recurrent' else np.zeros(8))
            logits = values['gate.weight'] @ hidden + values['gate.bias']
            odds = np.exp(logits - np.max(logits))
            mixing = odds / odds.sum()
        x = sum(mixing[j] * (matrix[j] @ (scale*x) + values['input_matrix'][j] @ torque
                            + values['expert_bias'][j]) for j in range(2)) / scale
        outputs.append(x[:6].copy())
    return np.asarray(outputs), x, hidden if kind == 'recurrent' else None


@pytest.mark.parametrize('kind', KINDS)
def test_independent_prefix_and_transition_equations(kind):
    cell = model(kind)
    q, u, future = wave(8, 1), wave(8, 1) * -.4, wave(15, 1) * .7
    expected, x, hidden = independent(cell.export_numpy(), q[0].numpy(), u[0].numpy(), future[0].numpy(), kind)
    actual, final = cell.rollout(future, cell.condition(q, u))
    np.testing.assert_allclose(actual.detach().numpy()[0], expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(final.x.detach().numpy()[0], x, rtol=1e-12, atol=1e-13)
    if hidden is None:
        assert final.h is None
    else:
        np.testing.assert_allclose(final.h.detach().numpy()[0], hidden, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_paired_initialization_counts_and_exact_initial_functions(dtype):
    cells = [model(kind, active=False, dtype=dtype) for kind in KINDS]
    assert [c.parameter_count for c in cells] == [1014, 1014, 468]
    assert [c.state_scalars for c in cells] == [20, 12, 12]
    for name in ('raw_matrix', 'input_matrix', 'expert_bias', 'scale_raw'):
        assert all(torch.equal(cells[0].state_dict()[name], c.state_dict()[name]) for c in cells[1:])
    for name, value in cells[0].state_dict().items():
        assert torch.equal(value, cells[1].state_dict()[name])
    q, u, future = wave(7, dtype=dtype), wave(7, dtype=dtype) * .3, wave(dtype=dtype)
    predictions = [c(q, u, future) for c in cells]
    assert all(torch.equal(predictions[0], value) for value in predictions[1:])
    np.testing.assert_allclose(cells[0].operators().scale.detach().numpy(), [1]*6+[10]*6, rtol=2e-7)
    assert cells[1].model_spec()['structurally_inactive_parameter_count'] == 192
    for cell in cells:
        spec = cell.model_spec()
        assert spec['parameter_bytes'] == cell.parameter_count * torch.empty((), dtype=dtype).element_size()
        assert spec['state_bytes_per_stream'] == cell.state_scalars * torch.empty((), dtype=dtype).element_size()
        assert spec['buffer_bytes'] == 0 and not dict(cell.named_buffers())


@pytest.mark.parametrize('kind', KINDS)
def test_condition_alignment_and_next_torque_predicts_next_position(kind):
    cell = model(kind)
    with torch.no_grad():
        cell.raw_matrix.zero_()
        cell.scale_raw.zero_()
        cell.input_matrix.zero_()
        cell.input_matrix[:, :6] = torch.eye(6, dtype=torch.float64)
        cell.expert_bias.zero_()
    q, u = wave(6), wave(6) * .3
    state = cell.condition(q, u)
    assert torch.equal(state.x[:, :6], q[:, -1])
    assert torch.equal(state.x[:, 6:], q[:, -1] - q[:, -2])
    actual, _ = cell.step(state, u[:, -1])
    torch.testing.assert_close(actual, u[:, -1], rtol=1e-14, atol=1e-14)
    changed = u.clone(); changed[:, -1] += 999
    same = cell.condition(q, changed)
    assert torch.equal(state.x, same.x)
    assert state.h is same.h is None or torch.equal(state.h, same.h)
    forecast = wave(5) * -.1
    torch.testing.assert_close(cell(q, u, forecast), forecast, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('kind', KINDS)
def test_rollout_chunking_public_steps_causality_and_owned_state(kind):
    cell = model(kind)
    q, u, future = wave(8), wave(8) * .2, wave(19) * -.3
    initial = cell.condition(q, u)
    saved_x = initial.x.clone()
    whole, final = cell.rollout(future, initial)
    left, middle = cell.rollout(future[:, :7], initial)
    right, last = cell.rollout(future[:, 7:], middle)
    assert torch.equal(whole, torch.cat((left, right), dim=1))
    assert torch.equal(final.x, last.x)
    assert final.h is last.h is None or torch.equal(final.h, last.h)
    current, outputs = initial, []
    for t in range(future.shape[1]):
        value, current = cell.step(current, future[:, t]); outputs.append(value)
    assert torch.equal(whole, torch.stack(outputs, dim=1))
    assert torch.equal(initial.x, saved_x)
    altered = future.clone(); altered[:, 7:] += 5
    assert torch.equal(cell(q, u, altered)[:, :7], whole[:, :7])
    empty, clone = cell.rollout(future[:, :0], initial)
    assert empty.shape == (2, 0, 6) and torch.equal(clone.x, initial.x)
    assert clone.x.data_ptr() != initial.x.data_ptr() != q[:, -1].data_ptr()
    if initial.h is not None:
        assert clone.h.data_ptr() != initial.h.data_ptr()


def test_recurrent_prefix_memory_witness_instant_and_constant_do_not_keep_it():
    q, u, future = wave(9, 1), wave(9, 1) * .3, wave(5, 1) * -.2
    altered = q.clone(); altered[:, 1:-2] += .9
    for kind in KINDS:
        cell = model(kind)
        original = cell(q, u, future)
        changed = cell(altered, u, future)
        if kind == 'recurrent':
            assert torch.max(torch.abs(original - changed)) > 1e-8
        else:
            assert torch.equal(original, changed)


def test_spectral_operators_computed_once_per_forward_without_persistent_cache(monkeypatch):
    cell = model()
    original = torch.linalg.matrix_norm
    calls = []
    def counted(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, 'matrix_norm', counted)
    q, u, future = wave(7), wave(7) * .5, wave(11)
    cell.condition(q, u)
    assert calls == []
    first = cell(q, u, future)
    assert calls == [(2, 12, 12)]
    with torch.no_grad():
        cell.input_matrix.add_(.01)
    second = cell(q, u, future)
    assert calls == [(2, 12, 12)] * 2
    assert not torch.equal(first, second)


@pytest.mark.parametrize('kind', KINDS)
def test_operator_common_norm_and_affine_forcing_bound_not_global_contraction(kind):
    cell = model(kind)
    with torch.no_grad():
        # Distinct singular values and norms above one exercise the cap.
        cell.raw_matrix[0].copy_(torch.diag(torch.linspace(.3, 2.7, 12, dtype=torch.float64)))
        cell.raw_matrix[1].copy_(torch.diag(torch.linspace(3., .2, 12, dtype=torch.float64)))
        cell.expert_bias.copy_(torch.linspace(-.2, .3, 24, dtype=torch.float64).reshape(2, 12))
    operators = cell.operators()
    for matrix in operators.matrix.detach().numpy():
        assert np.linalg.svd(matrix, compute_uv=False)[0] <= RADIUS + 2e-15
    x = torch.linspace(-1., 1.2, 24, dtype=torch.float64).reshape(2, 12)
    state = LPVState(x, torch.full((2, 8), .2, dtype=torch.float64) if kind == 'recurrent' else None)
    inputs = wave(1)[:, 0]
    _, final = cell.step(state, inputs)
    for b in range(2):
        old_norm = torch.linalg.vector_norm(operators.scale * state.x[b])
        new_norm = torch.linalg.vector_norm(operators.scale * final.x[b])
        forcing = max(torch.linalg.vector_norm(cell.input_matrix[j] @ inputs[b] + cell.expert_bias[j]) for j in range(2))
        assert new_norm <= RADIUS * old_norm + forcing + 1e-13
    assert 'not full-system incremental contraction' in cell.model_spec()['stability_scope']


@pytest.mark.parametrize('kind', KINDS)
def test_finite_parameter_and_context_gradients_with_inactive_paths_disclosed(kind):
    cell = model(kind)
    q, u = wave(6).requires_grad_(), (wave(6) * .2).requires_grad_()
    future = (wave(10) * -.3).requires_grad_()
    cell(q, u, future).square().mean().backward()
    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if kind == 'instant' and name == 'scheduler.weight_hh':
            assert torch.count_nonzero(parameter.grad) == 0
        else:
            assert torch.count_nonzero(parameter.grad) > 0
    for tensor in (q, u, future):
        if tensor.grad is not None:
            assert torch.isfinite(tensor.grad).all()
    assert q.grad is not None and future.grad is not None
    if kind == 'recurrent':
        assert u.grad is not None and torch.count_nonzero(u.grad[:, 1:-1]) > 0
        assert torch.count_nonzero(u.grad[:, -1]) == 0
    else:
        assert u.grad is None


def test_zero_gate_blocks_initial_scheduler_gradients_but_gate_and_experts_learn():
    cell = model(active=False)
    cell(wave(6), wave(6)*.2, wave(8)*-.3).square().sum().backward()
    for name, value in cell.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        if name.startswith('scheduler.'):
            assert torch.count_nonzero(value.grad) == 0
    assert torch.count_nonzero(cell.gate.weight.grad) > 0
    assert torch.count_nonzero(cell.raw_matrix.grad) > 0


def test_active_spectral_cap_gradient_matches_finite_difference():
    cell = model('constant')
    with torch.no_grad():
        cell.raw_matrix[0].copy_(torch.diag(torch.linspace(.5, 2., 12, dtype=torch.float64)))
        cell.raw_matrix[0, 0, 1] = .1
    q, u, future = wave(5, 1), wave(5, 1) * .2, wave(3, 1) * .4
    cell(q, u, future).square().sum().backward()
    for name, index in (('raw_matrix', (0, 0, 1)), ('scale_raw', (7,))):
        parameter = getattr(cell, name)
        analytical = parameter.grad[index].item()
        value, epsilon = parameter[index].item(), 1e-6
        with torch.no_grad():
            parameter[index] = value + epsilon
            plus = cell(q, u, future).square().sum().item()
            parameter[index] = value - epsilon
            minus = cell(q, u, future).square().sum().item()
            parameter[index] = value
        assert analytical == pytest.approx((plus-minus)/(2*epsilon), rel=2e-5, abs=1e-8)


@pytest.mark.parametrize('kind', KINDS)
def test_fabricated_float32_adam_steps_and_long_bounded_input_are_finite(kind):
    cell = model(kind, dtype=torch.float32)
    optimizer = torch.optim.Adam(cell.parameters(), lr=.001)
    q, u, future = wave(8, dtype=torch.float32), wave(8, dtype=torch.float32)*.3, wave(12, dtype=torch.float32)*-.2
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = cell(q, u, future).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        output = cell(q, u, wave(1024, dtype=torch.float32))
    assert torch.isfinite(output).all()
    # This finite fixture does not establish contraction of the nonlinear map.
    assert cell.model_spec()['parameter_count'] == (468 if kind == 'constant' else 1014)


def test_rng_ownership_exports_and_no_future_output_argument():
    before = torch.random.get_rng_state().clone()
    cell = model()
    assert torch.equal(before, torch.random.get_rng_state())
    exported = cell.export_numpy()
    original = cell.raw_matrix.detach().clone()
    exported['raw_matrix'][:] = 0
    assert torch.equal(cell.raw_matrix, original)
    document = cell.export_json()
    json.dumps(document, allow_nan=False)
    document['parameters']['raw_matrix'][0][0][0] = 123
    assert torch.equal(cell.raw_matrix, original)
    with pytest.raises(TypeError):
        cell(wave(4), wave(4), wave(5), target=wave(5))


@pytest.mark.parametrize('kind,seed,dtype', [('other', 1, torch.float32), ('constant', True, torch.float32),
                                          ('instant', -1, torch.float32), ('recurrent', 1, torch.float16)])
def test_constructor_metadata_guards(kind, seed, dtype):
    with pytest.raises(ValueError):
        BoundedLPV(kind, seed, dtype=dtype)


@pytest.mark.parametrize('kind', KINDS)
def test_shapes_nonfinite_and_wrong_state_guards(kind):
    cell = model(kind)
    q, u, future = wave(5), wave(5), wave(3)
    for bad in (q[:, :1], q[:, :, :5], q.float(), torch.full_like(q, float('nan'))):
        with pytest.raises(ValueError):
            cell(bad, u, future)
    with pytest.raises(ValueError):
        cell(q, u[:, :3], future)
    with pytest.raises(ValueError):
        cell(q, u, torch.full_like(future, float('inf')))
    state = cell.condition(q, u)
    wrong = state._replace(h=None if kind == 'recurrent' else torch.zeros(2, 8, dtype=torch.float64))
    with pytest.raises(ValueError):
        cell.rollout(future, wrong)
    with pytest.raises(ValueError):
        cell.step(tuple(state), future[:, 0])
    with torch.no_grad():
        cell.input_matrix[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        cell(q, u, future)
