"""Fabricated arithmetic and gradient checks, with no data or state loading."""
from __future__ import annotations

import math

import pytest
import torch

from openjev.research import otto_conditional_label as m
from openjev.research.otto_action_latent_model import make_model


def raw_bank():
    costs = torch.tensor([0., 64., 128., 192.]).repeat(2, 32, 1)
    alive = torch.ones(2, 32, dtype=torch.bool)
    alive[1, :16] = False
    costs[~alive] = 0
    return costs, alive


def test_unconditional_target_includes_found_zeros_and_owns_storage():
    raw, alive = raw_bank()
    before = raw.clone()
    bank = m.centered_bank(raw, alive)
    expected = torch.tensor([-1.5, -.5, .5, 1.5], dtype=torch.float64)
    assert torch.equal(bank[0, 0], expected)
    assert torch.equal(m.select_target(bank, 'mean32'), torch.stack((expected, expected / 2)))
    assert torch.equal(bank[1, :16], torch.zeros(16, 4, dtype=torch.float64))
    bank[0, 0] = 0
    assert torch.equal(raw, before)


def test_shared_scale_is_sampled_second_moment_not_mean_label_second_moment():
    bank = m.centered_bank(*raw_bank())
    result = m.train_scale(bank, variance_floor=1e-6)
    assert result['second_moment_before_floor'] == pytest.approx(.9375)
    assert result['cost_scale'] == float(torch.tensor(math.sqrt(.9375), dtype=torch.float32))
    assert float(m.select_target(bank, 'mean32').square().mean()) == pytest.approx(.78125)
    zero = torch.zeros(2, 32, 4, dtype=torch.float64)
    assert m.train_scale(zero, variance_floor=.04)['cost_scale'] == float(torch.tensor(.2))


def test_last_horizon_only_and_all_found_cases_still_penalize_nonzero_prediction():
    prediction = torch.zeros(2, 8, 4, requires_grad=True)
    with torch.no_grad():
        prediction[1, 7] = torch.tensor([-1., 0., 0., 1.])
    target = torch.zeros(2, 4, dtype=torch.float64)
    loss = m.horizon8_loss(prediction, target, cost_scale=.5)
    assert loss.dtype == torch.float64 and loss.item() == 1.
    loss.backward()
    assert not prediction.grad[:, :7].any()
    torch.testing.assert_close(prediction.grad[1, 7], torch.tensor([-1., 0., 0., 1.]), rtol=0, atol=0)


def test_complete_balanced_cycle_equals_mean_target_gradient_plus_fixed_variance():
    bank = m.centered_bank(*raw_bank())
    prediction = torch.linspace(-1., 1., 64).reshape(2, 8, 4).requires_grad_()
    mean_loss = m.horizon8_loss(prediction, m.select_target(bank, 'mean32'), cost_scale=.3)
    draw_losses = [m.horizon8_loss(prediction, m.select_target(bank, 'sampled',
        indices=torch.tensor([i, (i + 7) % 32])), cost_scale=.3) for i in range(32)]
    cycle_loss = torch.stack(draw_losses).mean()
    variance = (bank - bank.mean(1, keepdim=True)).square().mean() / .3**2
    torch.testing.assert_close(cycle_loss - mean_loss, variance, rtol=1e-12, atol=1e-12)
    actual = torch.autograd.grad(cycle_loss, prediction, retain_graph=True)[0]
    expected = torch.autograd.grad(mean_loss, prediction)[0]
    # The sampled-cycle backward casts 32 separate contributions to float32
    # before accumulating them; the mean-target backward casts just once.
    # Check the real-valued identity in float64, then charge those different
    # rounding paths rather than applying a fixed empirical tolerance.
    last = prediction.detach()[:, 7].double()
    ordered = torch.stack([bank[torch.arange(2), torch.tensor([i, (i + 7) % 32])]
                           for i in range(32)])
    contributions = 2 * (last[None] - ordered) / (.3**2 * 2 * 4 * 32)
    absolute_sum = contributions.abs().sum(0)
    exact_cycle = contributions.sum(0)
    exact_mean = 2 * (last - bank.mean(1)) / (.3**2 * 2 * 4)
    u64 = torch.finfo(torch.float64).eps / 2
    # This also covers mean/centering/reference arithmetic in this fixture.
    arithmetic_bound = (64 * u64 / (1 - 64 * u64)) * absolute_sum
    assert bool(((exact_cycle - exact_mean).abs() <= arithmetic_bound).all())
    u32 = torch.finfo(torch.float32).eps / 2
    # One cast per contribution plus at most 31 float32 additions on a path.
    gamma32 = 32 * u32 / (1 - 32 * u32)
    cycle_bound = gamma32 * absolute_sum + arithmetic_bound
    mean_bound = u32 * exact_mean.abs() + arithmetic_bound
    assert bool(((actual[:, 7].double() - exact_cycle).abs() <= cycle_bound).all())
    assert bool(((expected[:, 7].double() - exact_mean).abs() <= mean_bound).all())
    assert bool(((actual[:, 7].double() - expected[:, 7].double()).abs()
                 <= cycle_bound + mean_bound + arithmetic_bound).all())
    assert not actual[:, :7].any() and not expected[:, :7].any()


def test_selected_target_is_owned_and_inputs_have_no_gradient_history():
    bank = m.centered_bank(*raw_bank())
    before = bank.clone()
    target = m.select_target(bank, 'sampled', indices=torch.tensor([1, 31]))
    target.zero_()
    assert torch.equal(before, bank)
    with pytest.raises(ValueError, match='gradient'):
        m.select_target(bank.requires_grad_(), 'mean32')


@pytest.mark.parametrize('mutation', ['nan', 'negative', 'found_nonzero', 'wrong_dtype', 'wrong_shape', 'grad'])
def test_raw_bank_rejects_invalid_inputs(mutation):
    raw, alive = raw_bank()
    if mutation == 'nan':
        raw[0, 0, 0] = float('nan')
    elif mutation == 'negative':
        raw[0, 0, 0] = -1
    elif mutation == 'found_nonzero':
        raw[1, 0, 0] = 1
    elif mutation == 'wrong_dtype':
        raw = raw.double()
    elif mutation == 'wrong_shape':
        raw = raw[:, :31]
    else:
        raw.requires_grad_()
    with pytest.raises(ValueError):
        m.centered_bank(raw, alive)


@pytest.mark.parametrize('indices', [None, torch.tensor([0., 1.]), torch.tensor([-1, 0]), torch.tensor([0, 32]), torch.tensor([0])])
def test_sampled_indices_are_explicit_and_bounded(indices):
    with pytest.raises(ValueError):
        m.select_target(m.centered_bank(*raw_bank()), 'sampled', indices=indices)


@pytest.mark.parametrize('scale', [True, 0., -1., float('nan'), float('inf'), 1e-300])
def test_invalid_or_overflowing_scale_fails_without_mutating_prediction(scale):
    prediction = torch.ones(2, 8, 4, requires_grad=True)
    before = prediction.detach().clone()
    with pytest.raises(ValueError):
        m.horizon8_loss(prediction, torch.zeros(2, 4, dtype=torch.float64), cost_scale=scale)
    assert torch.equal(prediction, before) and prediction.grad is None


def test_mean_rejects_indices_and_loss_rejects_uncentered_targets():
    bank = m.centered_bank(*raw_bank())
    with pytest.raises(ValueError):
        m.select_target(bank, 'mean32', indices=torch.tensor([0, 0]))
    with pytest.raises(ValueError, match='centered'):
        m.horizon8_loss(torch.zeros(2, 8, 4), torch.ones(2, 4, dtype=torch.float64), cost_scale=1.)


def test_same_initial_gru_and_identical_bank_give_identical_updates_and_unused_heads():
    raw, alive = raw_bank()
    raw[1] = raw[0]
    alive[:] = True
    bank = m.centered_bank(raw, alive)
    models = [make_model('action_recurrent', 73, cost_scale=.5) for _ in range(2)]
    prefix = torch.linspace(-.2, .2, 2 * 9 * 31).reshape(2, 9, 31)
    lengths = torch.full((2,), 9, dtype=torch.int64)
    actions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0]])
    initial = {key: value.clone() for key, value in models[0].state_dict().items()}
    for model, arm in zip(models, m.ARMS, strict=True):
        for name, parameter in model.named_parameters():
            if name.startswith(('outcome_head.', 'aux_head.')):
                parameter.requires_grad_(False)
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in parameters) == 8096
        optimizer = torch.optim.Adam(parameters, lr=.003)
        # Two updates get beyond the shared zero-initialized cost-head barrier.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model.blind_rollout(prefix, lengths, actions)
            target = m.select_target(bank, arm, indices=torch.tensor([5, 17]) if arm == 'sampled' else None)
            m.horizon8_loss(result['cost_contrasts'], target, cost_scale=.5).backward()
            assert model.outcome_head.weight.grad is None and model.aux_head.weight.grad is None
            optimizer.step()
        for key, value in model.state_dict().items():
            if key.startswith(('outcome_head.', 'aux_head.')) or key == 'cost_scale':
                assert torch.equal(value, initial[key])
        assert not torch.equal(model.assimilation.weight_ih, initial['assimilation.weight_ih'])
        assert not torch.equal(model.transition.weight_ih, initial['transition.weight_ih'])
    assert all(torch.equal(value, models[1].state_dict()[key]) for key, value in models[0].state_dict().items())
