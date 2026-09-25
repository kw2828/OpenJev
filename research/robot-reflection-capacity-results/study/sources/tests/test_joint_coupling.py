"""Fabricated AR2, graph and conditioning checks, without measured robot data."""
import json
import math

import numpy as np
import pytest
import torch

from openjev.research.joint_coupling import ARMS, PARAMETER_SHAPES, JointCoupling, JointState


def base(dtype=torch.float64):
    result = torch.zeros(6, 25, dtype=dtype)
    for j in range(6):
        result[j, j], result[j, 6 + j] = .4, .15
        result[j, 12 + j], result[j, 18 + j] = .1, .05
        result[j, (j + 1) % 6] += .02
        result[j, -1] = (j - 2) * .001
    return result


def wave(length=13, batch=2, dtype=torch.float64):
    t = torch.arange(length, dtype=dtype)[None, :, None]
    j = torch.arange(6, dtype=dtype)[None, None, :]
    b = torch.arange(batch, dtype=dtype)[:, None, None]
    return torch.sin(.37 * t + .21 * j + .11 * b) + .2 * torch.cos(.19 * t - .31 * j)


def model(arm='chain_memory', *, active=True, dtype=torch.float64):
    result = JointCoupling(arm, 951101, base(dtype), dtype=dtype)
    if active:
        with torch.no_grad():
            result.local_gain.copy_(torch.linspace(.02, .07, 6, dtype=dtype))
            result.edge_gain.copy_(torch.linspace(.03, .08, 6, dtype=dtype))
            result.local_bias.fill_(.04)
            result.edge_bias.fill_(-.03)
    return result


def scalar_messages(values, q, previous, torque, old, arm):
    receivers, senders = values['edge_index']
    output = []
    for e, (i, j) in enumerate(zip(receivers, senders, strict=True)):
        features = [q[i], q[i] - previous[i], q[j], q[j] - previous[j], torque[i], torque[j]]
        signal = math.tanh(sum(float(w) * float(x) for w, x in zip(values['edge_encoder'][e], features, strict=True))
                           + float(values['edge_bias'][e]))
        decay = 1 / (1 + math.exp(-float(values['decay_logits'][e])))
        output.append((1 - decay) * signal + (0 if arm == 'chain_instant' else decay * old[e]))
    return output


def scalar_reference(values, q_context, u_context, future, arm):
    """Python scalar arithmetic, including independent observed-prefix alignment."""
    edges = [0.] * 10
    if arm != 'chain_instant':
        for t in range(1, len(q_context) - 1):
            edges = scalar_messages(values, q_context[t], q_context[t - 1], u_context[t], edges, arm)
    q, previous, prevu = list(q_context[-1]), list(q_context[-2]), list(u_context[-2])
    outputs = []
    for torque in future:
        edges = scalar_messages(values, q, previous, torque, edges, arm)
        prediction = []
        feature = list(q) + list(previous) + list(torque) + list(prevu) + [1.]
        for i in range(6):
            linear = sum(float(w) * float(v) for w, v in zip(values['base_weight'][i], feature, strict=True))
            local = math.tanh(sum(float(w) * float(v) for w, v in
                                  zip(values['local_weights'][i], [q[i], q[i] - previous[i], torque[i]], strict=True))
                              + float(values['local_bias'][i]))
            incoming = [edges[e] for e, dest in enumerate(values['edge_index'][0]) if dest == i]
            prediction.append(linear + float(values['local_gain'][i]) * local
                              + float(values['edge_gain'][i]) * sum(incoming) / len(incoming))
        outputs.append(prediction)
        previous, q, prevu = q, prediction, list(torque)
    return np.asarray(outputs), (np.asarray(q), np.asarray(previous), np.asarray(prevu),
                                 None if arm == 'chain_instant' else np.asarray(edges))


def assert_state_equal(left, right):
    for first, second in zip(left, right, strict=True):
        if first is None:
            assert second is None
        else:
            assert torch.equal(first, second)


@pytest.mark.parametrize('arm', ARMS)
def test_independent_scalar_prefix_and_rollout(arm):
    cell = model(arm)
    q, u, future = wave(7, 1), wave(7, 1) * .7, wave(17, 1) * -.3
    expected, expected_state = scalar_reference(cell.export_numpy(), q[0].tolist(), u[0].tolist(),
                                                future[0].tolist(), arm)
    actual, final = cell(future, cell.condition(q, u))
    np.testing.assert_allclose(actual.detach()[0], expected, rtol=1e-12, atol=1e-13)
    for got, want in zip(final, expected_state, strict=True):
        if got is None:
            assert want is None
        else:
            np.testing.assert_allclose(got.detach()[0], want, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_paired_initial_functions_parameters_counts_and_graph(dtype):
    cells = [model(arm, active=False, dtype=dtype) for arm in ARMS]
    assert all(c.parameter_count == 266 for c in cells)
    for name, shape in PARAMETER_SHAPES.items():
        assert tuple(cells[0].state_dict()[name].shape) == shape
        assert all(torch.equal(cells[0].state_dict()[name], c.state_dict()[name]) for c in cells[1:])
    q, u, future = wave(6, dtype=dtype), wave(6, dtype=dtype) * .8, wave(dtype=dtype)
    predictions = [c(future, c.condition(q, u))[0] for c in cells]
    assert all(torch.equal(predictions[0], p) for p in predictions[1:])
    assert [c.state_scalars for c in cells] == [28, 28, 18]
    for cell in cells:
        metadata = cell.parameter_metadata()
        assert metadata['parameter_bytes'] == 266 * torch.empty((), dtype=dtype).element_size()
        assert metadata['state_bytes_per_stream'] == cell.state_scalars * torch.empty((), dtype=dtype).element_size()
        assert metadata['buffer_bytes'] == 160
        assert torch.equal(torch.bincount(cell.edge_index[0], minlength=6), torch.tensor([1, 2, 2, 2, 2, 1]))
    assert cells[0].edge_index.tolist() == [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
    assert cells[1].edge_index.tolist() == [[0, 1, 1, 3, 3, 2, 2, 4, 4, 5], [1, 0, 3, 1, 2, 3, 4, 2, 5, 4]]


@pytest.mark.parametrize('arm', ARMS)
def test_exact_alignment_and_no_base_prediction_during_condition(arm):
    cell = model(arm, active=False)
    with torch.no_grad():
        cell.base_weight.zero_()
        for j in range(6):
            cell.base_weight[j, j] = 2
            cell.base_weight[j, 6 + j] = 3
            cell.base_weight[j, 12 + j] = 5
            cell.base_weight[j, 18 + j] = 7
            cell.base_weight[j, -1] = 11
    q, u = wave(4), wave(4) * .37
    state = cell.condition(q, u)
    assert torch.equal(state.q, q[:, -1]) and torch.equal(state.prevq, q[:, -2])
    assert torch.equal(state.prevu, u[:, -2])
    got, final = cell.step(u[:, -1], state)
    torch.testing.assert_close(got, 2 * q[:, -1] + 3 * q[:, -2] + 5 * u[:, -1] + 7 * u[:, -2] + 11,
                               rtol=1e-14, atol=1e-14)
    assert torch.equal(final.q, got) and torch.equal(final.prevq, q[:, -1])
    assert torch.equal(final.prevu, u[:, -1])
    changed = u.clone()
    changed[:, -1] += 99
    assert_state_equal(state, cell.condition(q, changed))


@pytest.mark.parametrize('arm', ARMS)
def test_causal_chunk_and_step_equivalence_with_owned_state(arm):
    cell = model(arm)
    q, u, future = wave(7), wave(7) * .6, wave(19)
    initial = cell.condition(q, u)
    saved = JointState(*(v.clone() if v is not None else None for v in initial))
    whole, last = cell(future, initial)
    left, middle = cell(future[:, :8], initial)
    right, chunk_last = cell(future[:, 8:], middle)
    assert torch.equal(whole, torch.cat((left, right), dim=1))
    assert_state_equal(last, chunk_last)
    step_state, outputs = initial, []
    for t in range(future.shape[1]):
        value, step_state = cell.step(future[:, t], step_state)
        outputs.append(value)
    assert torch.equal(whole, torch.stack(outputs, dim=1))
    assert_state_equal(last, step_state)
    assert_state_equal(initial, saved)
    altered = future.clone()
    altered[:, 8:] += 2
    assert torch.equal(cell(altered, initial)[0][:, :8], whole[:, :8])
    empty, cloned = cell(future[:, :0], initial)
    assert empty.shape == (2, 0, 6)
    assert_state_equal(initial, cloned)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(initial, cloned, strict=True) if a is not None)
    assert initial.q.data_ptr() != q[:, -1].data_ptr()
    assert last.prevu.data_ptr() != future[:, -1].data_ptr()
    assert last.q.data_ptr() != whole[:, -1].data_ptr()


def test_condition_two_points_needs_no_earlier_position_and_keeps_gradients():
    cell = model()
    q, u = wave(2).requires_grad_(), wave(2).requires_grad_()
    state = cell.condition(q, u)
    assert torch.count_nonzero(state.edges) == 0
    predicted, _ = cell.step(u[:, -1], state)
    predicted.square().sum().backward()
    assert torch.isfinite(q.grad).all() and torch.count_nonzero(q.grad) > 0
    assert torch.isfinite(u.grad).all() and torch.count_nonzero(u.grad) > 0


def test_memory_versus_instant_retains_only_declared_edge_state():
    memory, instant = model(), model('chain_instant')
    with torch.no_grad():
        for cell in (memory, instant):
            cell.base_weight.zero_()
            cell.local_gain.zero_()
            cell.edge_encoder.zero_()
            cell.edge_bias.zero_()
            cell.edge_gain.fill_(1)
    state = memory.initial_state(1)._replace(edges=torch.full((1, 10), .2, dtype=torch.float64))
    inputs = torch.zeros(1, 6, dtype=torch.float64)
    np.testing.assert_allclose(memory.step(inputs, state)[0].detach(), .18, rtol=1e-14)
    instant_output, instant_state = instant.step(inputs, instant.initial_state(1))
    assert torch.count_nonzero(instant_output) == 0 and instant_state.edges is None
    with pytest.raises(ValueError, match='no recurrent edge state'):
        instant.step(inputs, state)


def test_rewiring_changes_residual_function_without_relabeling_joint_io():
    chain, rewired = model(), model('rewired_memory')
    q, u, future = wave(5), wave(5) * .6, wave(8)
    first = chain(future, chain.condition(q, u))[0]
    second = rewired(future, rewired.condition(q, u))[0]
    assert torch.max(torch.abs(first - second)) > 1e-7


@pytest.mark.parametrize('arm', ARMS)
def test_initial_zero_gain_and_later_encoder_gradient_paths(arm):
    cell = model(arm, active=False)
    q, u, future = wave(7), wave(7) * .6, wave(9)
    loss = cell(future, cell.condition(q, u))[0].square().sum()
    loss.backward()
    for name in ('base_weight', 'local_gain', 'edge_gain'):
        assert torch.count_nonzero(getattr(cell, name).grad) > 0
    for name in ('local_weights', 'local_bias', 'edge_encoder', 'edge_bias', 'decay_logits'):
        assert torch.count_nonzero(getattr(cell, name).grad) == 0
    cell.zero_grad(set_to_none=True)
    with torch.no_grad():
        cell.local_gain.fill_(.2)
        cell.edge_gain.fill_(.1)
    cell(future, cell.condition(q, u))[0].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) > 0
               for p in cell.parameters())


def test_conditioning_encoder_gradient_matches_finite_difference():
    cell = model()
    q, u = wave(8, 1), wave(8, 1) * .7
    objective = cell.condition(q, u).edges.square().sum()
    objective.backward()
    for name, index in (('edge_encoder', (3, 0)), ('decay_logits', (3,))):
        parameter = getattr(cell, name)
        analytical = parameter.grad[index].item()
        original, epsilon = parameter[index].item(), 1e-6
        with torch.no_grad():
            parameter[index] = original + epsilon
            plus = cell.condition(q, u).edges.square().sum().item()
            parameter[index] = original - epsilon
            minus = cell.condition(q, u).edges.square().sum().item()
            parameter[index] = original
        assert analytical == pytest.approx((plus - minus) / (2 * epsilon), rel=1e-7, abs=1e-9)


def test_batch_independence_and_changed_parameters_do_not_use_stale_cache():
    cell = model()
    q, u, future = wave(5), wave(5) * .6, wave(8)
    batch = cell(future, cell.condition(q, u))[0]
    single = cell(future[:1], cell.condition(q[:1], u[:1]))[0]
    torch.testing.assert_close(batch[:1], single, rtol=1e-13, atol=1e-14)
    with torch.no_grad():
        cell.edge_gain.add_(.2)
    assert not torch.equal(batch, cell(future, cell.condition(q, u))[0])


def test_local_rng_owned_base_and_exports():
    rng = torch.random.get_rng_state().clone()
    coefficients = base()
    original = coefficients.clone()
    cell = JointCoupling('chain_memory', 951101, coefficients, dtype=torch.float64)
    assert torch.equal(rng, torch.random.get_rng_state())
    coefficients.zero_()
    assert torch.equal(cell.base_weight, original)
    arrays = cell.export_numpy()
    assert set(arrays) == set(PARAMETER_SHAPES) | {'edge_index'}
    arrays['base_weight'][:] = 999
    arrays['edge_index'][:] = 0
    assert torch.equal(cell.base_weight, original)
    assert torch.count_nonzero(cell.edge_index) > 0
    document = cell.export_json()
    json.dumps(document, allow_nan=False)
    document['parameters']['base_weight'][0][0] = 777
    assert cell.base_weight[0, 0].item() == original[0, 0].item()
    assert not document['metadata']['persistent_cache'] and not document['metadata']['retained_trajectory']


@pytest.mark.parametrize('arm', ARMS)
def test_bounded_fabricated_input_stays_finite_without_general_stability_claim(arm):
    cell = model(arm, dtype=torch.float32)
    with torch.no_grad():
        output, final = cell(wave(1024, 1, torch.float32), cell.condition(wave(5, 1, torch.float32),
                                                                      wave(5, 1, torch.float32)))
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value).all() for value in final if value is not None)
    assert cell.parameter_metadata()['stability_scope'].startswith('no boundedness')


@pytest.mark.parametrize('arm,seed,dtype', [('bad', 1, torch.float32), ('chain_memory', True, torch.float32),
                                          ('chain_memory', -1, torch.float32), ('chain_memory', 1, torch.float16)])
def test_constructor_rejects_bad_metadata(arm, seed, dtype):
    with pytest.raises(ValueError):
        JointCoupling(arm, seed, base(torch.float32), dtype=dtype)


@pytest.mark.parametrize('bad', [torch.zeros(25, 6), torch.zeros(6, 25, dtype=torch.float64),
                               torch.full((6, 25), float('nan'))])
def test_constructor_rejects_bad_base(bad):
    with pytest.raises(ValueError):
        JointCoupling('chain_memory', 1, bad)


def test_public_guards_and_no_label_api():
    cell = model()
    state = cell.initial_state(2)
    with pytest.raises(TypeError):
        cell(wave())
    with pytest.raises(TypeError):
        cell(wave(), state, targets=wave())
    for bad in (torch.zeros(0, 3, 6, dtype=torch.float64), wave().float(), wave()[:, :, :5],
                torch.full((2, 3, 6), float('inf'), dtype=torch.float64)):
        with pytest.raises(ValueError):
            cell(bad, state)
    with pytest.raises(ValueError):
        cell.condition(wave(1), wave(1))
    with pytest.raises(ValueError):
        cell.condition(wave(3), wave(2))
    with pytest.raises(ValueError):
        cell.step(wave(1)[:, 0], state._replace(edges=None))
    with pytest.raises(ValueError):
        cell.step(wave(1)[:, 0], tuple(state))
    with torch.no_grad():
        cell.edge_index[0, 0] = 5
    with pytest.raises(ValueError, match='graph'):
        cell(wave(), state)


def test_nonfinite_parameters_and_overflow_fail_without_clipping():
    cell = model(dtype=torch.float32)
    with torch.no_grad():
        cell.base_weight[0, 0] = float('nan')
    with pytest.raises(ValueError, match='base_weight'):
        cell(wave(dtype=torch.float32), cell.initial_state(2))
    cell = model(dtype=torch.float32)
    with torch.no_grad():
        cell.base_weight.fill_(torch.finfo(torch.float32).max)
    state = cell.initial_state(1)._replace(q=torch.ones(1, 6), prevq=torch.ones(1, 6))
    with pytest.raises(ValueError, match='nonfinite'):
        cell.step(torch.ones(1, 6), state)
