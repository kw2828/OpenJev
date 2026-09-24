"""Fabricated adapter checks; no synthetic study split or empirical data is read."""
from __future__ import annotations

import math

import pytest
import torch

from openjev.research.finite_observation_models import (
    ARMS,
    make_model,
    metadata,
    objective,
    soft_cross_entropy,
)


def public_inputs(horizon=8):
    prefix = torch.zeros((2, 5, 31), dtype=torch.float32)
    lengths = torch.tensor([3, 5], dtype=torch.int64)
    for lane, length in enumerate(lengths.tolist()):
        prefix[lane, 0, 9] = 1
        for time in range(length):
            prefix[lane, time, 4 + (time + lane) % 4] = 1
            if time:
                prefix[lane, time, (time + lane) % 4] = 1
    actions = torch.arange(2 * horizon, dtype=torch.int64).reshape(2, horizon) % 4
    observations = (actions + 1) % 4
    return prefix, lengths, actions, observations


def assert_predictions(prediction, horizon):
    assert prediction['cost_contrasts'].shape == (2, horizon, 4)
    assert prediction['survival_mass'].shape == (2, horizon)
    for name in ('cost_contrasts', 'survival_mass'):
        assert prediction[name].dtype == torch.float64
        assert prediction[name].device.type == 'cpu'
        assert torch.isfinite(prediction[name]).all()
    # GRU cost centering happens in float32; float64 operator centering is tighter.
    assert prediction['cost_contrasts'].sum(-1).abs().max() < 5e-7
    survival = prediction['survival_mass']
    assert ((survival >= 0) & (survival <= 1 + 1e-12)).all()
    if 'probabilities' in prediction:
        p = prediction['probabilities']
        assert p.shape == (2, horizon, 5) and p.dtype == torch.float64
        assert torch.isfinite(p).all() and (p >= 0).all()
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]), rtol=0, atol=3e-7)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('horizon', (2, 8))
def test_declared_shapes_precision_and_probability_mass(arm, horizon):
    model = make_model(arm, 913)
    prefix, lengths, actions, observations = public_inputs(horizon)
    blind = model.blind_rollout(prefix, lengths, actions)
    observed = model.observed_rollout(prefix, lengths, actions, observations)
    assert_predictions(blind, horizon)
    assert_predictions(observed, horizon)
    assert 'probabilities' in observed
    if arm != 'gru':
        assert observed['adapter_extra_work'] == {
            'operator_materializations': 1, 'event_probability_rows': 2 * horizon}
        assert observed['work']['operator_observed_softmax_calls'] == 1
        assert observed['work']['operator_blind_softmax_calls'] == int(arm.startswith('untied'))
        assert 'core only' in observed['work_scope']
    assert (blind['survival_mass'][:, 1:] <= blind['survival_mass'][:, :-1] + 1e-12).all()


@pytest.mark.parametrize('arm', ARMS)
def test_current_label_and_future_actions_do_not_change_earlier_forecasts(arm):
    model = make_model(arm, 71)
    prefix, lengths, actions, observations = public_inputs()
    original = model.observed_rollout(prefix, lengths, actions, observations)
    changed_observations = observations.clone()
    changed_observations[:, 2] = (changed_observations[:, 2] + 1) % 4
    changed = model.observed_rollout(prefix, lengths, actions, changed_observations)
    for name in ('cost_contrasts', 'survival_mass', 'probabilities'):
        assert torch.equal(original[name][:, :3], changed[name][:, :3])
    assert not torch.equal(original['probabilities'][:, 3:], changed['probabilities'][:, 3:])
    future_actions = actions.clone()
    future_actions[:, 4:] = (future_actions[:, 4:] + 2) % 4
    future = model.observed_rollout(prefix, lengths, future_actions, observations)
    for name in ('cost_contrasts', 'survival_mass', 'probabilities'):
        assert torch.equal(original[name][:, :4], future[name][:, :4])
    blind = model.blind_rollout(prefix, lengths, actions)
    blind_future = model.blind_rollout(prefix, lengths, future_actions)
    assert torch.equal(blind['cost_contrasts'][:, :4], blind_future['cost_contrasts'][:, :4])


@pytest.mark.parametrize('arm', ARMS)
def test_padding_poison_is_ignored_but_consumed_poison_rejected(arm):
    model = make_model(arm, 81)
    prefix, lengths, actions, observations = public_inputs(2)
    before = prefix.clone()
    reference = model.observed_rollout(prefix, lengths, actions, observations)
    padded = prefix.clone()
    padded[0, 3:] = torch.nan
    actual = model.observed_rollout(padded, lengths, actions, observations)
    for name in ('cost_contrasts', 'survival_mass', 'probabilities'):
        assert torch.equal(reference[name], actual[name])
    assert torch.equal(prefix, before)
    assert torch.isnan(padded[0, 3:]).all()
    consumed = prefix.clone()
    consumed[0, 1, 10] = torch.inf
    with pytest.raises(ValueError):
        model.observed_rollout(consumed, lengths, actions, observations)


@pytest.mark.parametrize('arm', ARMS)
def test_found_prediction_is_prelabel_then_exactly_absorbing(arm):
    model = make_model(arm, 38)
    prefix, lengths, actions, observations = public_inputs()
    live = model.observed_rollout(prefix, lengths, actions, observations)
    observations[:, 2:] = 4
    found = model.observed_rollout(prefix, lengths, actions, observations)
    for name in ('cost_contrasts', 'survival_mass', 'probabilities'):
        assert torch.equal(found[name][:, :3], live[name][:, :3])
    assert torch.equal(found['cost_contrasts'][:, 3:], torch.zeros((2, 5, 4), dtype=torch.float64))
    assert torch.equal(found['survival_mass'][:, 3:], torch.zeros((2, 5), dtype=torch.float64))
    terminal = torch.zeros((2, 5, 5), dtype=torch.float64)
    terminal[..., 4] = 1
    assert torch.equal(found['probabilities'][:, 3:], terminal)
    observations[0, 4] = 0
    with pytest.raises(ValueError):
        model.observed_rollout(prefix, lengths, actions, observations)


@pytest.mark.parametrize('initialization', ('dense', 'retentive'))
def test_tied_untied_share_parameters_and_initial_blind_function(initialization):
    tied = make_model('tied_' + initialization, 514)
    untied = make_model('untied_' + initialization, 514)
    shared, separate = tied.state_dict(), untied.state_dict()
    assert set(separate) - set(shared) == {'core.blind_logits'}
    for name in shared:
        assert torch.equal(shared[name], separate[name]), name
    inputs = public_inputs()
    paired_bound = 64 * 14 * 8 * torch.finfo(torch.float64).eps
    tied_blind = tied.blind_rollout(*inputs[:3])
    untied_blind = untied.blind_rollout(*inputs[:3])
    for name in ('cost_contrasts', 'survival_mass'):
        torch.testing.assert_close(tied_blind[name], untied_blind[name], rtol=0, atol=paired_bound)
    tied_observed = tied.observed_rollout(*inputs)
    untied_observed = untied.observed_rollout(*inputs)
    for name in ('cost_contrasts', 'survival_mass', 'probabilities'):
        assert torch.equal(tied_observed[name], untied_observed[name])


@pytest.mark.parametrize('arm', ARMS)
def test_constructor_and_rollouts_preserve_caller_rng_and_use_explicit_precision(arm):
    random_state = torch.random.get_rng_state().clone()
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = make_model(arm, 95)
    finally:
        torch.set_default_dtype(old_dtype)
    assert torch.equal(torch.random.get_rng_state(), random_state)
    expected = {torch.float32} if arm == 'gru' else {torch.float32, torch.float64}
    assert {p.dtype for p in model.parameters()} == expected
    assert all(p.device.type == 'cpu' for p in model.parameters())
    inputs = public_inputs(2)
    model.blind_rollout(*inputs[:3])
    model.observed_rollout(*inputs)
    assert torch.equal(torch.random.get_rng_state(), random_state)


def test_retentive_initialization_has_large_diagonal_without_zero_offdiagonals():
    dense = make_model('tied_dense', 631).core.operators()['blind']
    retained = make_model('tied_retentive', 631).core.operators()['blind']
    conditional = retained / retained.sum(dim=1, keepdim=True)
    diagonal = conditional.diagonal(dim1=1, dim2=2)
    assert (diagonal > .90).all()
    mask = ~torch.eye(14, dtype=torch.bool)[None].expand(4, -1, -1)
    assert (conditional[mask] > 0).all() and (conditional[mask] < .01).all()
    dense_conditional = dense / dense.sum(dim=1, keepdim=True)
    assert (dense_conditional.diagonal(dim1=1, dim2=2) < .20).all()


@pytest.mark.parametrize('arm', ARMS)
def test_parameter_storage_counts_are_explicit_and_complete(arm):
    model = make_model(arm, 91)
    info = metadata(model)
    expected_count = 11181 if arm == 'gru' else (9618 if arm.startswith('untied') else 8778)
    expected_bytes = 44724 if arm == 'gru' else (56448 if arm.startswith('untied') else 49728)
    assert info['parameter_count'] == expected_count
    assert info['parameter_bytes'] == expected_bytes
    assert info['buffer_bytes'] == (0 if arm == 'gru' else 8)
    assert set(info['parameters']) == dict(model.named_parameters()).keys()
    assert all(row['count'] > 0 and row['bytes'] in (4 * row['count'], 8 * row['count'])
               for row in info['parameters'].values())


@pytest.mark.parametrize('arm', ARMS)
def test_combined_objective_has_finite_gradients_and_adam_updates_each_parameter_group(arm):
    model = make_model(arm, 36)
    inputs = public_inputs(2)
    targets = {
        'blind_costs': torch.tensor([.4, -.3, .1, -.2], dtype=torch.float64).expand(2, 2, 4),
        'observed_costs': torch.tensor([-.2, .1, .3, -.2], dtype=torch.float64).expand(2, 2, 4),
        'blind_survival': torch.tensor([.90, .80], dtype=torch.float64).expand(2, 2),
        'observed_survival': torch.full((2, 2), .95, dtype=torch.float64),
        'observed_probabilities': torch.tensor([.1, .2, .3, .25, .15], dtype=torch.float64).expand(2, 2, 5),
    }
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    loss = objective(model.blind_rollout(*inputs[:3]), model.observed_rollout(*inputs), targets)
    assert loss.ndim == 0 and loss.dtype == torch.float64 and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        assert not torch.equal(parameter, before[name]), name


def test_soft_cross_entropy_preserves_zero_support_and_tiny_probability():
    terminal = torch.tensor([[0., 0., 0., 0., 1.]], dtype=torch.float64)
    assert soft_cross_entropy(terminal, terminal).item() == 0
    probability = torch.tensor([[1e-200, .25, .25, .25, .25]], dtype=torch.float64)
    target = torch.tensor([[1., 0., 0., 0., 0.]], dtype=torch.float64)
    assert soft_cross_entropy(probability, target).item() == pytest.approx(200 * math.log(10), rel=1e-14)
    with pytest.raises(ValueError):
        soft_cross_entropy(terminal, target)
    with pytest.raises(ValueError):
        soft_cross_entropy(terminal * 2, terminal)
    with pytest.raises(ValueError):
        soft_cross_entropy(terminal, terminal * 2)


@pytest.mark.parametrize('arm', ARMS)
def test_event_underflow_is_rejected_instead_of_clipped(arm):
    model = make_model(arm, 411)
    with torch.no_grad():
        if arm == 'gru':
            model.event_head.weight.zero_()
            model.event_head.bias.zero_()
            model.event_head.bias[0] = -1000
        else:
            model.core.observed_logits[:, :14] = -1000
    with pytest.raises(ValueError):
        model.observed_rollout(*public_inputs(2))


@pytest.mark.parametrize('arm,seed', [('other', 0), ('gru', True), ('gru', -1), ('gru', 2**32)])
def test_invalid_constructor_is_rejected(arm, seed):
    with pytest.raises(ValueError):
        make_model(arm, seed)
