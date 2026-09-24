"""Fabricated loss identities and matched numerical paths; no empirical inputs."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from openjev.research import finite_action_range_loss as a
from openjev.research import finite_joint_reuse as parent
from openjev.research.finite_head_initialization import make_model as parent_model
from openjev.research.finite_observation_models import objective
from openjev.research.finite_rounded_models import DYNAMICS, RoundedDynamicsModel, torch_objective

SEED = 948101


def fixture(endings=(None, 1, None), horizon=2):
    prefix = torch.zeros((len(endings), 9, 31), dtype=torch.float32)
    lengths = torch.tensor([9 if end is None else end + 1 for end in endings], dtype=torch.int64)
    for row, end in enumerate(endings):
        prefix[row, 0, 4 + row % 4] = prefix[row, 0, 9] = 1
        for step in range(1, int(lengths[row])):
            prefix[row, step, (row + step) % 4] = 1
            prefix[row, step, 8 if step == end else 4 + (row + 2 * step) % 4] = 1
    positions = torch.tensor([i for i, end in enumerate(endings) if end is None], dtype=torch.int64)
    eligible = len(positions)
    actions = torch.arange(eligible * horizon).reshape(eligible, horizon).remainder(4)
    observations = (actions + 1).remainder(4)
    if eligible:
        observations[0, 1:] = 4
    targets = {
        'blind_costs': torch.tensor([.23, -.17, .03, -.09], dtype=torch.float64).expand(eligible, horizon, 4).clone(),
        'observed_costs': torch.tensor([-.13, .21, -.07, -.01], dtype=torch.float64).expand(eligible, horizon, 4).clone(),
        'blind_survival': torch.full((eligible, horizon), .8, dtype=torch.float64),
        'observed_survival': torch.full((eligible, horizon), .9, dtype=torch.float64),
        'observed_probabilities': torch.tensor([.1, .2, .3, .25, .15], dtype=torch.float64).expand(eligible, horizon, 5).clone()}
    for row in range(eligible):
        for step in range(1, horizon):
            if observations[row, step - 1] == 4:
                targets['observed_costs'][row, step] = 0
                targets['observed_survival'][row, step] = 0
                targets['observed_probabilities'][row, step] = torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64)
    return {'prefix': prefix, 'lengths': lengths, 'endpoint_positions': positions,
        'actions': actions, 'observations': observations, 'targets': targets,
        'total_attempts': 11, 'total_survivors': 7, 'total_events': 99}


def test_centered_four_action_bounds_and_sharp_constants():
    # Exhaustive small centered vectors, not a fitted numerical tolerance.
    for values in itertools.product((-2., -1., 0., 1., 2.), repeat=4):
        error = torch.tensor(values, dtype=torch.float64)
        error = error - error.mean()
        mse = error.square().mean()
        value = a.action_range_loss(error.reshape(1, 1, 4))
        assert mse <= value <= 2 * mse
    low = torch.tensor([-1., -1., 1., 1.], dtype=torch.float64)
    high = torch.tensor([-1., 0., 0., 1.], dtype=torch.float64)
    assert a.action_range_loss(low.reshape(1, 1, 4)) == low.square().mean()
    assert a.action_range_loss(high.reshape(1, 1, 4)) == 2 * high.square().mean()
    shifted = (high + 7).reshape(1, 1, 4)
    assert a.action_range_loss(shifted) == 1  # Range needs no centering.
    assert shifted.square().mean() > 2  # MSE sandwich requires centered errors.


def test_greedy_regret_is_bounded_by_error_span_including_ties():
    truths = ([0., 0., 1., 2.], [0., .01, .2, .8], [.3, -.2, .1, -.2])
    predictions = ([0., 0., 0., 0.], [.4, .1, -.3, .2], [-.5, .7, .8, .9])
    for q, p in itertools.product(truths, predictions):
        target, predicted = torch.tensor(q, dtype=torch.float64), torch.tensor(p, dtype=torch.float64)
        error = predicted - target
        regret = target[predicted.argmin()] - target.min()
        assert regret >= 0
        assert regret <= error.max() - error.min() + 4 * torch.finfo(torch.float64).eps
        assert regret.square() <= 4 * a.action_range_loss(error.reshape(1, 1, 4)) + 8 * torch.finfo(torch.float64).eps
    # A first-index prediction tie attains the bound with the wrong action.
    q = torch.tensor([1., 0., 2., 3.], dtype=torch.float64)
    p = torch.tensor([0., 0., 2., 3.], dtype=torch.float64)
    assert q[p.argmin()] - q.min() == 1
    assert a.action_range_loss((p - q).reshape(1, 1, 4)) == .25


def test_range_gradient_finite_difference_away_from_extremum_ties():
    error = torch.tensor([[[-1.4, -.2, .5, 1.1], [.3, -1.2, 1.6, -.7]]], dtype=torch.float64, requires_grad=True)
    a.action_range_loss(error).backward()
    for coordinate in range(error.numel()):
        plus, minus = error.detach().clone(), error.detach().clone()
        plus.view(-1)[coordinate] += 1e-6
        minus.view(-1)[coordinate] -= 1e-6
        finite_difference = (a.action_range_loss(plus) - a.action_range_loss(minus)) / 2e-6
        torch.testing.assert_close(error.grad.view(-1)[coordinate], finite_difference, atol=2e-9, rtol=2e-9)


def test_tie_shared_gradients_and_action_permutation_equivariance():
    base = torch.tensor([[[-2., -2., 2., 2.]]], dtype=torch.float64, requires_grad=True)
    a.action_range_loss(base).backward()
    torch.testing.assert_close(base.grad, torch.tensor([[[-1., -1., 1., 1.]]], dtype=torch.float64), atol=0, rtol=0)
    for permutation in itertools.permutations(range(4)):
        reordered = base.detach()[..., list(permutation)].clone().requires_grad_()
        a.action_range_loss(reordered).backward()
        torch.testing.assert_close(reordered.grad, base.grad[..., list(permutation)], atol=0, rtol=0)
    constant = torch.full((2, 3, 4), 4., dtype=torch.float64, requires_grad=True)
    a.action_range_loss(constant).backward()
    assert torch.count_nonzero(constant.grad) == 0


@pytest.mark.parametrize('arm', a.ARMS)
def test_factory_preserves_exact_model_and_task_independent_head(arm):
    torch_rng, numpy_rng = torch.random.get_rng_state().clone(), np.random.get_state()
    value, reference = a.make_model(arm, SEED), parent_model(a.HEAD_ARMS[arm], SEED)
    assert type(value) is RoundedDynamicsModel
    assert sum(parameter.numel() for parameter in value.parameters()) == 352
    assert list(value.state_dict()) == list(reference.state_dict())
    for name, tensor in value.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[name], atol=0, rtol=0)
    meta = a.model_metadata(value, arm)
    assert meta['arm'] == arm and meta['loss_kind'] == a.LOSS_KINDS[arm]
    assert meta['privileged_readout_initialization'] is False
    assert meta['head_initialization']['seed_sequence_entropy'] == [SEED, 436, 1]
    assert torch.equal(torch_rng, torch.random.get_rng_state())
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]


@pytest.mark.parametrize('transport', ['rounded', 'free'])
@pytest.mark.parametrize('endings', [(None, None, None), (None, 1, None), (1, 3, 8)])
@pytest.mark.parametrize('kind', ['mse', 'double', 'range'])
def test_global_normalization_full_partial_and_empty_endpoint_batches(transport, endings, kind):
    model = a.make_model(f'{transport}_{kind}', SEED)
    values = fixture(endings)
    result = a.joint_objective(model, **values, loss_kind=kind)
    eligible = len(values['endpoint_positions'])
    graph_zero = sum(parameter.sum() * 0 for parameter in model.parameters())
    expected = graph_zero
    if eligible:
        blind, observed, target = result['blind'], result['observed'], values['targets']
        original_mse = (blind['cost_contrasts'] - target['blind_costs']).square().mean()
        endpoint = objective(blind, observed, target)
        if kind == 'double':
            endpoint = endpoint + original_mse
        elif kind == 'range':
            error = blind['cost_contrasts'] - target['blind_costs']
            # Independent four-component pairwise expression for the span.
            differences = torch.stack([error[..., i] - error[..., j] for i in range(4) for j in range(4)], -1)
            endpoint = endpoint - original_mse + differences.amax(-1).square().mean() / 4
        expected = expected + endpoint * eligible / values['total_survivors']
    expected = (expected + result['prefix']['nll'].sum() / values['total_events']) * values['total_attempts'] / len(endings)
    torch.testing.assert_close(result['loss'], expected, atol=2e-14, rtol=2e-14)
    result['loss'].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    work = result['loss_work']
    wanted = a.loss_work_counts()
    wanted['wrapper_calls'] = 1
    if not eligible:
        wanted['zero_endpoint_batches'] = 1
        assert result['blind'] is result['observed'] is None
        assert torch.count_nonzero(model.cost_logits.grad) == 0
    elif kind != 'mse':
        rows = eligible * values['actions'].shape[1]
        wanted.update(blind_error_entries=4 * rows, mse_square_entries=4 * rows, mse_mean_calls=1,
                      weight_scalar_divisions=2, weight_scalar_multiplications=1,
                      loss_adjustment_multiplications=1, loss_adjustment_additions=1)
        if kind == 'range':
            wanted.update(range_max_rows=rows, range_min_rows=rows, range_subtract_entries=rows,
                          range_square_entries=rows, range_scale_entries=rows, range_mean_calls=1,
                          replacement_subtractions=1)
    assert work == wanted
    assert tuple(result['work']) == parent.WORK_ROUTES


@pytest.mark.parametrize('transport', ['rounded', 'free'])
def test_mse_keeps_original_values_gradients_and_three_adam_steps(transport):
    old, new = parent_model(a.HEAD_ARMS[f'{transport}_mse'], SEED), a.make_model(f'{transport}_mse', SEED)
    optimizers = [torch.optim.Adam(model.parameters(), lr=.003) for model in (old, new)]
    for step in range(3):
        values = fixture((None, 1, None) if step != 1 else (1, 3, 8))
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        before = parent.joint_objective(old, **values)
        after = a.joint_objective(new, **values, loss_kind='mse')
        torch.testing.assert_close(before['loss'], after['loss'], atol=0, rtol=0)
        before['loss'].backward()
        after['loss'].backward()
        for first, second in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
        for model in (old, new):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        for first, second in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        left, right = [opt.state_dict() for opt in optimizers]
        assert left['param_groups'] == right['param_groups']
        for index, state in left['state'].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, right['state'][index][name], atol=0, rtol=0)


def test_mse_and_empty_endpoint_return_same_original_loss_tensor(monkeypatch):
    sentinel = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    def frozen(*args, **kwargs):
        return {'loss': sentinel}
    monkeypatch.setattr(a, '_joint_objective', frozen)
    for kind, positions in [('mse', [0]), ('mse', []), ('double', []), ('range', [])]:
        result = a.joint_objective(None, [None], None, positions, None, None, None,
                                  total_attempts=1, total_survivors=1, total_events=1, loss_kind=kind)
        assert result['loss'] is sentinel


def test_all_loss_variants_share_initial_parameters_and_prefix_head_exclusion():
    values = fixture()
    models = {arm: a.make_model(arm, SEED) for arm in a.ARMS}
    for transport in ('rounded', 'free'):
        reference = models[f'{transport}_mse']
        for kind in ('double', 'range'):
            for name, parameter in reference.named_parameters():
                torch.testing.assert_close(parameter, dict(models[f'{transport}_{kind}'].named_parameters())[name], atol=0, rtol=0)
    for model in models.values():
        head = model.cost_logits.detach().clone()
        parameters = [getattr(model, name) for name in DYNAMICS]
        opt = torch.optim.Adam(parameters, lr=.003)
        result = torch_objective(model, values['prefix'], values['lengths'], .001)
        result['loss'].backward()
        assert model.cost_logits.grad is None
        opt.step()
        torch.testing.assert_close(model.cost_logits, head, atol=0, rtol=0)


@pytest.mark.parametrize('bad', [torch.zeros(4, dtype=torch.float64), torch.zeros((0, 2, 4), dtype=torch.float64),
                               torch.zeros((1, 2, 3), dtype=torch.float64), torch.zeros((1, 2, 4), dtype=torch.float32),
                               torch.full((1, 2, 4), float('nan'), dtype=torch.float64)])
def test_invalid_range_inputs_fail(bad):
    with pytest.raises(ValueError):
        a.action_range_loss(bad)


def test_failure_retains_partial_extra_work_without_swallowing_external_error(monkeypatch):
    marker = TimeoutError('fabricated deadline')
    def stop(*args, **kwargs):
        raise marker
    monkeypatch.setattr(a, '_joint_objective', stop)
    with pytest.raises(TimeoutError) as caught:
        a.joint_objective(None, None, None, None, None, None, None,
                          total_attempts=1, total_survivors=1, total_events=1, loss_kind='range')
    assert caught.value is marker
    wanted = a.loss_work_counts()
    wanted['wrapper_calls'] = 1
    assert marker.action_range_loss_work == wanted
