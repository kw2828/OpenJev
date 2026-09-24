"""Fabricated causal and mathematical checks; no benchmark data or fitting."""
import json
import math

import numpy as np
import pytest
import torch

from openjev.research.phase_memory import ARMS, BASE_KEYS, PhaseMemory


def sequence(dtype=torch.float64, length=31, batch=3):
    time = torch.arange(length, dtype=dtype)
    return torch.stack([torch.sin(time * .37 + i) + .2 * torch.cos(time * .13)
                        for i in range(batch)])[:, :, None]


def scalar_reference(parameters, inputs, state, arm):
    """Independent scalar equations using exported values, with no model calls."""
    sigmoid = lambda x: 1 / (1 + math.exp(-float(x)))
    current = [[float(v) for v in pair] for pair in state]
    outputs = []
    for value in inputs:
        updated = []
        for j in range(2):
            radius = .9999 * sigmoid(parameters['rho_logits'][j])
            angle = math.pi * sigmoid(parameters['angle_logits'][j])
            if arm == 'energy_phase':
                energy = sum(v * v for v in current[j])
                angle += .5 * math.tanh(float(parameters['phase_logits'][j])) * math.tanh(
                    math.log1p(math.exp(float(parameters['energy_scales_raw'][j]))) * energy)
            first, second = current[j]
            updated.append([
                radius * (math.cos(angle) * first - math.sin(angle) * second)
                + float(parameters['input_weights'][j, 0]) * value,
                radius * (math.sin(angle) * first + math.cos(angle) * second)
                + float(parameters['input_weights'][j, 1]) * value,
            ])
        prediction = sum(updated[j][k] * float(parameters['readout_weights'][j, k])
                         for j in range(2) for k in range(2))
        if arm == 'nonlinear_readout':
            prediction += sum(updated[j][k] * sum(v * v for v in updated[j])
                              * float(parameters['cubic_readout_weights'][j, k])
                              for j in range(2) for k in range(2))
        prediction += float(parameters['feedthrough']) * value + float(parameters['bias'])
        outputs.append(prediction)
        current = updated
    return np.asarray(outputs), np.asarray(current)


@pytest.mark.parametrize('arm', ARMS)
def test_scalar_equations_and_advance_before_readout(arm):
    model = PhaseMemory(arm, 71, dtype=torch.float64)
    with torch.no_grad():
        model.feedthrough.fill_(.23)
        model.bias.fill_(-.11)
        if arm == 'energy_phase':
            model.phase_logits.copy_(torch.tensor([.7, -.4], dtype=torch.float64))
        elif arm == 'nonlinear_readout':
            model.cubic_readout_weights.fill_(.13)
    inputs = sequence(batch=1)
    initial = torch.tensor([[[.3, -.4], [.2, .1]]], dtype=torch.float64)
    expected, final = scalar_reference(model.export_numpy(), inputs[0, :, 0].tolist(), initial[0], arm)
    with torch.no_grad():
        actual, actual_final = model(inputs, initial)
    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual_final[0].numpy(), final, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_counts_initial_functions_and_base_parameters_are_paired(dtype):
    models = [PhaseMemory(arm, 11, dtype=dtype) for arm in ARMS]
    assert [m.parameter_count for m in models] == [18, 14, 18]
    for name in BASE_KEYS:
        assert all(torch.equal(models[0].state_dict()[name], m.state_dict()[name]) for m in models[1:])
    with torch.no_grad():
        values = [m(sequence(dtype)) for m in models]
    assert all(torch.equal(values[0][0], p) and torch.equal(values[0][1], s) for p, s in values[1:])
    np.testing.assert_allclose((.9999 * torch.sigmoid(models[0].rho_logits)).detach().numpy(),
                               [.99, .95], rtol=2e-7)
    np.testing.assert_allclose((torch.sigmoid(models[0].angle_logits)).detach().numpy(),
                               [.1, .3], rtol=2e-7)


@pytest.mark.parametrize('arm', ARMS)
def test_chunked_and_public_step_match_one_forward_without_mutation(arm):
    model = PhaseMemory(arm, 23, dtype=torch.float64)
    inputs = sequence()
    initial = torch.full((3, 2, 2), .17, dtype=torch.float64)
    saved = initial.clone()
    with torch.no_grad():
        whole, final = model(inputs, initial)
        left, middle = model(inputs[:, :12], initial)
        right, final_chunk = model(inputs[:, 12:], middle)
        current, outputs = initial, []
        for t in range(inputs.shape[1]):
            prediction, current = model.step(inputs[:, t, 0], current)
            outputs.append(prediction)
    assert torch.equal(whole, torch.cat((left, right), dim=1))
    assert torch.equal(whole, torch.stack(outputs, dim=1))
    assert torch.equal(final, final_chunk) and torch.equal(final, current)
    assert torch.equal(initial, saved)
    assert final.data_ptr() != initial.data_ptr()


@pytest.mark.parametrize('arm', ARMS)
def test_future_input_cannot_change_prefix_and_module_has_no_retained_state(arm):
    model = PhaseMemory(arm, 37)
    inputs = sequence(torch.float32)
    changed = inputs.clone()
    changed[:, 14:] = 10
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        original, _ = model(inputs)
        poisoned, _ = model(changed)
        repeated, _ = model(inputs)
    assert torch.equal(original[:, :14], poisoned[:, :14])
    assert torch.equal(original, repeated)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not list(model.buffers())


def test_energy_phase_has_causal_parameter_gradients_and_no_detach():
    model = PhaseMemory('energy_phase', 11, dtype=torch.float64)
    inputs = sequence(length=19).requires_grad_()
    state = torch.full((3, 2, 2), .7, dtype=torch.float64, requires_grad=True)
    prediction, final = model(inputs, state)
    loss = prediction.square().mean() + .01 * final.square().mean()
    loss.backward()
    assert model.phase_logits.grad.abs().sum() > 1e-10
    assert torch.equal(model.energy_scales_raw.grad, torch.zeros(2, dtype=torch.float64))
    assert inputs.grad.abs().sum() > 0 and state.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.phase_logits.fill_(.35)
    model(inputs.detach(), state.detach())[0].square().mean().backward()
    assert model.energy_scales_raw.grad.abs().sum() > 1e-10
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_phase_gradient_matches_central_difference():
    model = PhaseMemory('energy_phase', 37, dtype=torch.float64)
    inputs = sequence(length=11, batch=1)
    with torch.no_grad():
        model.phase_logits.fill_(.3)
    loss = model(inputs)[0].square().sum()
    analytic = torch.autograd.grad(loss, model.phase_logits)[0][0].item()
    with torch.no_grad():
        original = model.phase_logits[0].item()
        model.phase_logits[0] = original + 1e-6
        upper = model(inputs)[0].square().sum().item()
        model.phase_logits[0] = original - 1e-6
        lower = model(inputs)[0].square().sum().item()
        model.phase_logits[0] = original
    assert analytic == pytest.approx((upper - lower) / 2e-6, rel=1e-6, abs=1e-10)


@pytest.mark.parametrize('arm', ARMS)
def test_zero_input_norm_decay_and_bounded_input_state_bound(arm):
    model = PhaseMemory(arm, 23, dtype=torch.float64)
    if arm == 'energy_phase':
        with torch.no_grad():
            model.phase_logits.copy_(torch.tensor([1.2, -.9], dtype=torch.float64))
    initial = torch.tensor([[[3., -4.], [-1., 2.]]], dtype=torch.float64)
    with torch.no_grad():
        radius = .9999 * torch.sigmoid(model.rho_logits)
        _, final = model(torch.zeros(1, 79, 1, dtype=torch.float64), initial)
        torch.testing.assert_close(torch.linalg.vector_norm(final, dim=-1),
                                   torch.linalg.vector_norm(initial, dim=-1) * radius**79,
                                   rtol=1e-12, atol=1e-12)
        state = initial
        bound = torch.linalg.vector_norm(initial)
        for inputs in sequence(length=91, batch=1)[0]:
            _, state = model.step(inputs, state)
            bound = radius.max() * bound + torch.linalg.vector_norm(model.input_weights) * inputs.abs()[0]
            assert torch.linalg.vector_norm(state) <= bound + 1e-12


def test_long_nonzero_phase_float32_against_independent_float64_witness():
    model = PhaseMemory('energy_phase', 23)
    with torch.no_grad():
        model.phase_logits.copy_(torch.tensor([.35, -.25]))
    inputs = sequence(torch.float32, length=8192, batch=1)
    expected, final = scalar_reference(model.export_numpy(), inputs[0, :, 0].tolist(), [[0., 0.], [0., 0.]],
                                       'energy_phase')
    with torch.no_grad():
        prediction, state = model(inputs)
    np.testing.assert_allclose(prediction[0].numpy(), expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(state[0].numpy(), final, rtol=1e-4, atol=1e-5)


def test_local_rng_owned_exports_metadata_and_parameter_refresh():
    rng_before = torch.random.get_rng_state().clone()
    model = PhaseMemory('energy_phase', 11)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    arrays = model.export_numpy()
    assert set(arrays) == set(model.state_dict())
    arrays['input_weights'][:] = 999
    assert model.input_weights.abs().max() < 999
    encoded = json.loads(json.dumps(model.export_json(), allow_nan=False))
    assert encoded['metadata']['parameter_count'] == 18
    assert encoded['metadata']['parameter_bytes'] == 72
    assert encoded['metadata']['state_bytes_per_stream'] == 16
    assert encoded['metadata']['state_scalars'] == model.state_scalars == 4
    inputs = sequence(torch.float32)
    with torch.no_grad():
        before = model(inputs)[0]
        model.phase_logits.fill_(.7)
        after = model(inputs)[0]
    assert not torch.equal(before, after)


def test_empty_time_has_owned_final_state_and_no_implicit_labels():
    model = PhaseMemory('fixed_phase', 11)
    state = model.initial_state(2)
    prediction, final = model(torch.empty(2, 0, 1), state)
    assert prediction.shape == (2, 0) and torch.equal(state, final)
    assert state.data_ptr() != final.data_ptr()
    with pytest.raises(TypeError):
        model(sequence(torch.float32), targets=torch.zeros(3, 31))


@pytest.mark.parametrize('args', [('unknown', 11), ('fixed_phase', True), ('fixed_phase', -1)])
def test_invalid_constructor(args):
    with pytest.raises(ValueError):
        PhaseMemory(*args)


@pytest.mark.parametrize('bad', [torch.zeros(2, 3), torch.zeros(2, 3, 2), torch.zeros(0, 3, 1),
                               torch.zeros(2, 3, 1, dtype=torch.float64), torch.full((2, 3, 1), float('nan'))])
def test_invalid_sequence(bad):
    with pytest.raises(ValueError):
        PhaseMemory('energy_phase', 11)(bad)


def test_invalid_state_parameters_and_precision_fail_explicitly():
    with pytest.raises(ValueError):
        PhaseMemory('fixed_phase', 11, dtype=torch.float16)
    model = PhaseMemory('fixed_phase', 11)
    with pytest.raises(ValueError):
        model(sequence(torch.float32), torch.zeros(3, 4))
    with pytest.raises(ValueError):
        model.step(torch.zeros(2, 1), model.initial_state(2))
    with torch.no_grad():
        model.rho_logits[0] = float('inf')
    with pytest.raises(ValueError):
        model(sequence(torch.float32))
