"""Fabricated full-recurrence tangent and mixed-derivative witnesses only."""
import copy

import numpy as np
import pytest
import torch

from openjev.research import fsm_sensitivity as sensitivity
from openjev.research.fsm_residual import FSMResidual
from openjev.research.fsm_sensitivity import (
    gain_profiles,
    regularizer,
    regularizers,
    rollout_primal,
    rollout_tangents,
    sample_directions,
)


def coefficients(order):
    value = np.zeros((3, 6*order+4), dtype=np.float64)
    value[:, 3*(order-1):3*order] = np.array([[.35, .02, .01], [.03, .28, .02], [.01, .04, .22]])
    value[:, 3*order:3*order+3] = np.diag([.2, .3, .4])
    value[:, 6*order:6*order+3] = np.eye(3)*.04
    value[:, -1] = [.01, .02, .03]
    return value


def model(order=1, width=2, *, active=True, mode='feedback', kind='tanh'):
    result = FSMResidual(coefficients(order), order=order, mode=mode,
                         residual_kind=kind, hidden_width=width, seed=31)
    if active and kind == 'tanh':
        with torch.no_grad():
            first, last = result.residual[0], result.residual[2]
            first.weight.copy_(.04+torch.arange(first.weight.numel(), dtype=torch.float64).reshape_as(first.weight)*.001)
            first.bias.copy_(torch.arange(width, dtype=torch.float64)*.01+.02)
            last.weight.copy_(.1+torch.arange(last.weight.numel(), dtype=torch.float64).reshape_as(last.weight)*.01)
            last.bias.copy_(torch.tensor([.03, .04, .05], dtype=torch.float64))
    return result


def inputs(order=1, batch=2, horizon=4):
    state = torch.linspace(.1, .3, batch*6*order, dtype=torch.float64).reshape(batch, 6*order)
    future = torch.linspace(.2, .4, batch*horizon*3, dtype=torch.float64).reshape(batch, horizon, 3)
    generator = torch.Generator(device='cpu').manual_seed(703)
    directions = sample_directions(batch, order, generator=generator)
    return future, state, directions


def native_jvp(cell, future, state, directions):
    """Differentiate the qualified native rollout, not the tangent implementation."""
    predictions, finals = [], []
    for k in range(directions.shape[1]):
        tangent = torch.cat((directions[:, k], torch.zeros_like(directions[:, k])), dim=1)
        _primal, derivative = torch.autograd.functional.jvp(
            lambda value: cell.rollout(future, value), state, tangent, create_graph=True)
        predictions.append(derivative[0])
        finals.append(derivative[1])
    return torch.stack(predictions, dim=1), torch.stack(finals, dim=1)


def linear_matrix_oracle(co, order, directions, horizon):
    """Materialize the VARX state matrix independently from lag coordinates."""
    split, size = 3*order, 6*order
    selector = np.zeros((size+3, size))
    selector[:split, :split] = np.eye(split)
    selector[split+3:, split:] = np.eye(split)
    output = co[:, :-1]@selector
    transition = np.zeros((size, size))
    transition[:split-3, 3:split] = np.eye(split-3)
    transition[split-3:split] = output
    transition[split:size-3, split+3:] = np.eye(split-3)
    tangent = np.concatenate((directions, np.zeros_like(directions)), axis=-1)
    values = []
    for _ in range(horizon):
        values.append(tangent@output.T)
        tangent = tangent@transition.T
    return np.stack(values, axis=2), tangent


@pytest.mark.parametrize('order,horizon', [(1, 4), (3, 9), (32, 128)])
def test_primal_exact_native_rollout_and_independent_linear_companion(order, horizon):
    cell = model(order, active=False)
    future, state, directions = inputs(order, horizon=horizon)
    actual = rollout_tangents(cell, future, state, directions)
    prediction, final = cell.rollout(future, state)
    assert torch.equal(actual['prediction'], prediction)
    assert torch.equal(actual['final_state'], final)
    assert torch.count_nonzero(actual['residual']) == 0
    expected, expected_final = linear_matrix_oracle(coefficients(order), order, directions.numpy(), horizon)
    for key in ('output_tangents', 'linear_output_tangents'):
        np.testing.assert_allclose(actual[key].detach().numpy(), expected, rtol=2e-13, atol=2e-14)
    for key in ('final_tangents', 'linear_final_tangents'):
        np.testing.assert_allclose(actual[key].detach().numpy(), expected_final, rtol=2e-13, atol=2e-14)
        assert torch.count_nonzero(actual[key][:, :, 3*order:]) == 0


@pytest.mark.parametrize('order', [1, 3])
def test_active_tangent_equals_independent_native_jvp_and_centered_state_difference(order):
    cell = model(order)
    future, state, directions = inputs(order, horizon=5)
    actual = rollout_tangents(cell, future, state, directions)
    prediction, final = cell.rollout(future, state)
    assert torch.equal(actual['prediction'], prediction)
    assert torch.equal(actual['final_state'], final)
    tangent, tangent_final = native_jvp(cell, future, state, directions)
    torch.testing.assert_close(actual['output_tangents'], tangent, rtol=3e-12, atol=3e-13)
    torch.testing.assert_close(actual['final_tangents'], tangent_final, rtol=3e-12, atol=3e-13)
    epsilon = 1e-5
    for k in range(directions.shape[1]):
        v = torch.cat((directions[:, k], torch.zeros_like(directions[:, k])), dim=1)
        plus, plus_final = cell.rollout(future, state+epsilon*v)
        minus, minus_final = cell.rollout(future, state-epsilon*v)
        torch.testing.assert_close(actual['output_tangents'][:, k], (plus-minus)/(2*epsilon), rtol=2e-7, atol=2e-10)
        torch.testing.assert_close(actual['final_tangents'][:, k], (plus_final-minus_final)/(2*epsilon), rtol=2e-7, atol=2e-10)


def test_reported_residual_is_each_neural_correction_not_accumulated_forecast_difference():
    cell = model(order=2)
    future, state, directions = inputs(2, horizon=5)
    result = rollout_tangents(cell, future, state, directions)
    weights = {k: v.detach().numpy().copy() for k, v in cell.state_dict().items()}
    past_y, past_u = state[:, :6].numpy().copy(), state[:, 6:].numpy().copy()
    expected = []
    for t in range(5):
        current_u = future[:, t].numpy()
        features = np.concatenate((past_y, current_u, past_u), axis=1)
        hidden = np.tanh(features@weights['residual.0.weight'].T+weights['residual.0.bias'])
        correction = hidden@weights['residual.2.weight'].T+weights['residual.2.bias']
        expected.append(correction)
        predicted = features@weights['coefficients'][:, :-1].T+weights['coefficients'][:, -1]+correction
        past_y = np.concatenate((past_y[:, 3:], predicted), axis=1)
        past_u = np.concatenate((past_u[:, 3:], current_u), axis=1)
    np.testing.assert_allclose(result['residual'].detach().numpy(), np.stack(expected, axis=1), rtol=2e-13, atol=2e-14)
    zero_head = copy.deepcopy(cell)
    with torch.no_grad():
        zero_head.residual[2].weight.zero_()
        zero_head.residual[2].bias.zero_()
    accumulated = result['prediction']-zero_head.rollout(future, state)[0]
    assert not torch.allclose(result['residual'][:, 1:], accumulated[:, 1:], rtol=1e-8, atol=1e-8)


def test_causal_prefix_and_caller_carried_learned_tangent_match_one_full_request():
    cell = model(order=2)
    future, state, directions = inputs(2, horizon=6)
    full = rollout_tangents(cell, future, state, directions)
    first = rollout_tangents(cell, future[:, :2], state, directions)
    second = rollout_tangents(cell, future[:, 2:], first['final_state'], first['final_tangents'][:, :, :6])
    assert torch.equal(full['prediction'], torch.cat((first['prediction'], second['prediction']), dim=1))
    assert torch.equal(full['output_tangents'], torch.cat((first['output_tangents'], second['output_tangents']), dim=2))
    assert torch.equal(full['final_state'], second['final_state'])
    assert torch.equal(full['final_tangents'], second['final_tangents'])
    changed = future.clone()
    changed[:, 2:] += 1.
    altered = rollout_tangents(cell, changed, state, directions)
    assert torch.equal(altered['prediction'][:, :2], full['prediction'][:, :2])
    assert torch.equal(altered['output_tangents'][:, :, :2], full['output_tangents'][:, :, :2])
    scaled = rollout_tangents(cell, future, state, directions*2)
    assert torch.equal(scaled['prediction'], full['prediction'])
    torch.testing.assert_close(scaled['output_tangents'], full['output_tangents']*2, rtol=0, atol=0)


def test_current_and_past_input_values_change_primal_but_have_zero_directional_support():
    co = np.zeros((3, 16), dtype=np.float64)
    co[:, 6:9] = np.diag([2., 3., 4.])
    co[:, 9:12] = np.eye(3)*5.
    co[:, 12:15] = np.eye(3)*7.
    cell = FSMResidual(co, order=2, mode='feedback', hidden_width=2)
    future, state, directions = inputs(2, horizon=4)
    result = rollout_tangents(cell, future, state, directions)
    assert torch.count_nonzero(result['prediction']) > 0
    assert torch.count_nonzero(result['output_tangents']) == 0
    assert torch.count_nonzero(result['linear_output_tangents']) == 0
    assert torch.count_nonzero(result['final_tangents']) == 0


def test_active_relative_penalty_has_full_mixed_derivative_for_all29_parameters():
    cell = model()
    assert cell.parameter_count() == 29
    future, state, directions = inputs(batch=1, horizon=4)
    directions = torch.ones_like(directions)/np.sqrt(3.)
    result = rollout_tangents(cell, future, state, directions)
    penalty = regularizers(cell, result, horizons=(1, 2, 4))['relative_sensitivity']
    assert penalty.item() > 0.
    parameters = tuple(cell.parameters())
    actual = torch.autograd.grad(penalty, parameters)
    learned, _ = native_jvp(cell, future, state, directions)
    baseline, _ = linear_matrix_oracle(coefficients(1), 1, directions.numpy(), 4)
    baseline = torch.from_numpy(baseline)
    oracle_terms = []
    for h in (1, 2, 4):
        learned_gain = learned[:, :, :h].square().sum(dim=(2, 3))/(3*h)
        baseline_gain = baseline[:, :, :h].square().sum(dim=(2, 3))/(3*h)
        assert bool((learned_gain > baseline_gain).all())
        oracle_terms.append((learned_gain-baseline_gain).square())
    oracle = torch.stack(oracle_terms).mean()
    expected = torch.autograd.grad(oracle, parameters)
    for a, b in zip(actual, expected, strict=True):
        assert bool(torch.isfinite(a).all()) and bool((a.abs() > 1e-12).all())
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-12)
    epsilon = 1e-5
    for parameter, gradient in zip(parameters, actual, strict=True):
        for index in range(parameter.numel()):
            original = parameter.detach().reshape(-1)[index].item()
            values = []
            for delta in (epsilon, -epsilon):
                with torch.no_grad():
                    parameter.reshape(-1)[index] = original+delta
                trial = rollout_tangents(cell, future, state, directions)
                values.append(regularizers(cell, trial, horizons=(1, 2, 4))['relative_sensitivity'].item())
            with torch.no_grad():
                parameter.reshape(-1)[index] = original
            assert gradient.reshape(-1)[index].item() == pytest.approx((values[0]-values[1])/(2*epsilon), rel=3e-5, abs=2e-9)


@pytest.mark.parametrize('name', ['relative_sensitivity', 'max_envelope'])
def test_zero_head_penalty_and_every_parameter_gradient_exactly_zero_at_all_registered_horizons(name):
    cell = model(order=32, width=24, active=False)
    future, state, directions = inputs(32, batch=2, horizon=128)
    result = rollout_tangents(cell, future, state, directions)
    assert torch.equal(result['output_tangents'], result['linear_output_tangents'])
    for horizon in (8, 32, 128):
        loss = regularizers(cell, result, horizons=(horizon,))[name]
        assert loss.item() == 0.
        gradients = torch.autograd.grad(loss, tuple(cell.parameters()), retain_graph=True)
        assert all(bool(torch.isfinite(g).all()) and torch.count_nonzero(g) == 0 for g in gradients)


def test_gain_profiles_and_envelopes_use_each_batch_direction_and_horizon_not_global_max():
    # Constant per-channel tangents make each gain equal to its squared amplitude.
    amplitudes = torch.tensor([[1., 3.], [2., 4.]], dtype=torch.float64)
    base = torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64)
    result = {'output_tangents': amplitudes[:, :, None, None].expand(2, 2, 4, 3).clone(),
              'linear_output_tangents': base[:, :, None, None].expand(2, 2, 4, 3).clone(),
              'residual': torch.full((2, 4, 3), 2., dtype=torch.float64)}
    gain, reference = gain_profiles(result, horizons=(1, 4))
    wanted = amplitudes.square()[:, :, None].expand(2, 2, 2)
    torch.testing.assert_close(gain, wanted, rtol=0, atol=0)
    torch.testing.assert_close(reference, base.square()[:, :, None].expand_as(wanted), rtol=0, atol=0)
    cell = model()
    penalties = regularizers(cell, result, horizons=(1, 4))
    assert set(penalties) == {'unregularized', 'l2', 'residual_magnitude', 'total_sensitivity', 'max_envelope', 'relative_sensitivity'}
    assert penalties['unregularized'].item() == 0.
    assert penalties['residual_magnitude'].item() == 4.
    assert penalties['total_sensitivity'].item() == (1+81+16+256)/4
    assert penalties['relative_sensitivity'].item() == (0+25+0+225)/4
    assert penalties['max_envelope'].item() == (0+25+0+49)/4
    expected_l2 = sum(p.detach().square().sum() for p in cell.parameters())/29
    torch.testing.assert_close(penalties['l2'], expected_l2, rtol=1e-15, atol=1e-16)
    assert all(value.dtype == torch.float64 and value.ndim == 0 for value in penalties.values())


def test_horizon_gain_uses_all_earlier_outputs_not_only_endpoint():
    tangents = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)[None, None, :, None].expand(1, 1, 4, 3)
    result = {'output_tangents': tangents, 'linear_output_tangents': tangents*0}
    learned, baseline = gain_profiles(result, horizons=(1, 2, 4))
    torch.testing.assert_close(learned, torch.tensor([[[1., 2.5, 7.5]]], dtype=torch.float64), rtol=0, atol=0)
    assert torch.count_nonzero(baseline) == 0


def test_selected_cheap_regularizers_do_not_require_or_inspect_tangent_outputs():
    cell = model()
    poisoned = {'output_tangents': torch.full((1, 2, 4, 3), float('inf'), dtype=torch.float64),
                'linear_output_tangents': object()}
    assert regularizer(cell, None, 'unregularized').item() == 0.
    assert regularizer(cell, poisoned, 'unregularized').item() == 0.
    expected = sum(p.square().sum() for p in cell.parameters())/29
    torch.testing.assert_close(regularizer(cell, None, 'l2'), expected, rtol=0, atol=0)
    torch.testing.assert_close(regularizer(cell, poisoned, 'l2'), expected, rtol=0, atol=0)
    correction = torch.full((2, 4, 3), 2., dtype=torch.float64)
    assert regularizer(cell, {'residual': correction}, 'residual_magnitude').item() == 4.
    assert regularizer(cell, {**poisoned, 'residual': correction}, 'residual_magnitude').item() == 4.


def test_primal_only_path_matches_native_values_and_gradients_without_tangent_calls(monkeypatch):
    cell = model(order=2)
    future, state, _directions = inputs(2, horizon=4)
    future.requires_grad_()
    state.requires_grad_()
    monkeypatch.setattr(sensitivity, 'rollout_tangents', lambda *a, **k: pytest.fail('cheap path must not propagate tangents'))
    result = rollout_primal(cell, future, state)
    native, final = cell.rollout(future, state)
    assert set(result) == {'prediction', 'final_state', 'residual'}
    assert torch.equal(result['prediction'], native) and torch.equal(result['final_state'], final)
    parameters = (*cell.parameters(), future, state)
    actual = torch.autograd.grad(result['prediction'].square().mean()+result['final_state'].square().mean(), parameters)
    expected = torch.autograd.grad(native.square().mean()+final.square().mean(), parameters)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-13, atol=2e-14)
    assert regularizer(cell, result, 'residual_magnitude').item() > 0.


def test_selected_total_sensitivity_needs_no_baseline_or_residual_bank():
    cell = model()
    learned = torch.full((1, 2, 4, 3), 2., dtype=torch.float64)
    assert regularizer(cell, {'output_tangents': learned}, 'total_sensitivity', horizons=(1, 4)).item() == 16.
    poisoned = {'output_tangents': learned, 'linear_output_tangents': object(),
                'residual': torch.tensor(float('nan'))}
    assert regularizer(cell, poisoned, 'total_sensitivity', horizons=(1, 4)).item() == 16.


def test_direction_generator_is_explicit_unit_rademacher_and_global_rng_is_unchanged():
    before = torch.get_rng_state().clone()
    generator = torch.Generator(device='cpu').manual_seed(998)
    initial = generator.get_state().clone()
    first = sample_directions(3, 8, generator=generator, count=2)
    second = sample_directions(3, 8, generator=generator, count=2)
    clone = torch.Generator(device='cpu').set_state(initial)
    assert torch.equal(first, sample_directions(3, 8, generator=clone, count=2))
    assert torch.equal(second, sample_directions(3, 8, generator=clone, count=2))
    assert first.dtype == torch.float64 and first.device.type == 'cpu' and first.shape == (3, 2, 24)
    torch.testing.assert_close(first.abs(), torch.full_like(first, 1/np.sqrt(24.)), rtol=0, atol=0)
    torch.testing.assert_close(first.square().sum(dim=-1), torch.ones((3, 2), dtype=torch.float64), rtol=2e-15, atol=2e-15)
    assert not torch.equal(initial, generator.get_state())
    assert torch.equal(before, torch.get_rng_state()) and first.data_ptr() != second.data_ptr()


def test_no_caller_model_mutation_or_persistent_tangent_cache():
    cell = model(order=3)
    future, state, directions = inputs(3)
    owned = [x.clone() for x in (future, state, directions)]
    weights = copy.deepcopy(cell.state_dict())
    fields = set(vars(cell))
    first = rollout_tangents(cell, future, state, directions)
    second = rollout_tangents(cell, future, state, directions)
    assert set(vars(cell)) == fields
    assert set(cell.state_dict()) == set(weights)
    assert all(torch.equal(value, weights[name]) for name, value in cell.state_dict().items())
    assert all(torch.equal(a, b) for a, b in zip((future, state, directions), owned, strict=True))
    for key, value in first.items():
        assert torch.equal(value, second[key])
        assert value.data_ptr() != second[key].data_ptr()
        assert all(value.data_ptr() != x.data_ptr() for x in (future, state, directions))
    first['final_state'].detach().fill_(123.)
    assert torch.equal(second['final_state'], cell.rollout(future, state)[1])


@pytest.mark.parametrize('mode,kind', [('output_only', 'tanh'), ('feedback', 'affine')])
def test_only_tanh_feedback_is_admitted(mode, kind):
    cell = model(mode=mode, kind=kind)
    future, state, directions = inputs()
    with pytest.raises(ValueError):
        rollout_tangents(cell, future, state, directions)


@pytest.mark.parametrize('which,kind', [('future', 'dtype'), ('state', 'dtype'), ('directions', 'dtype'),
    ('future', 'shape'), ('state', 'shape'), ('directions', 'shape'),
    ('future', 'nonfinite'), ('state', 'nonfinite'), ('directions', 'nonfinite')])
def test_malformed_or_nonfinite_inputs_fail_without_repair(which, kind):
    values = dict(zip(('future', 'state', 'directions'), inputs(), strict=True))
    if kind == 'dtype':
        values[which] = values[which].float()
    elif kind == 'shape':
        values[which] = values[which][..., :-1]
    else:
        values[which].reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError):
        rollout_tangents(model(), values['future'], values['state'], values['directions'])


@pytest.mark.parametrize('horizons', [(), (0,), (5,), (True,), (1.5,)])
def test_invalid_gain_horizons_are_rejected(horizons):
    value = torch.ones((1, 2, 4, 3), dtype=torch.float64)
    with pytest.raises(ValueError):
        gain_profiles({'output_tangents': value, 'linear_output_tangents': value}, horizons=horizons)


@pytest.mark.parametrize('kwargs', [{'batch_size': 0}, {'batch_size': True}, {'order': 0},
    {'order': 100}, {'order': True}, {'count': 0}, {'count': True}, {'generator': None}])
def test_direction_sampler_rejects_invalid_domain_without_consuming_valid_stream(kwargs):
    generator = torch.Generator(device='cpu').manual_seed(7)
    before = generator.get_state().clone()
    arguments = {'batch_size': 2, 'order': 1, 'count': 2, 'generator': generator, **kwargs}
    with pytest.raises(ValueError):
        sample_directions(**arguments)
    assert torch.equal(before, generator.get_state())


def test_tangent_overflow_and_empty_training_horizon_fail_without_repair():
    co = np.zeros((3, 10), dtype=np.float64)
    co[:, :3] = np.eye(3)*4.
    cell = FSMResidual(co, order=1, mode='feedback', hidden_width=2)
    state = torch.zeros((1, 6), dtype=torch.float64)
    future = torch.zeros((1, 2, 3), dtype=torch.float64)
    directions = torch.full((1, 2, 3), 1e308, dtype=torch.float64)
    before = directions.clone()
    with pytest.raises(ValueError, match='nonfinite sensitivity'):
        rollout_tangents(cell, future, state, directions)
    assert torch.equal(directions, before)
    with pytest.raises(ValueError, match='positive forecast horizon'):
        rollout_tangents(cell, future[:, :0], state, torch.ones_like(directions))
