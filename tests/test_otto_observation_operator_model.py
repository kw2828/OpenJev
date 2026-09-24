"""Independent fabricated probability/causality checks for prospective operators.

These are engineering tests, not an empirical model comparison. Exact rational
branch enumeration supplies the main oracle; no native inputs or files load.
"""
from __future__ import annotations

import inspect
import itertools
from fractions import Fraction

import pytest
import torch

from openjev.research import otto_observation_operator_model as component

DTYPE = torch.float64
BASE = (((8, 4), (4, 8)), ((4, 2), (2, 4)),
        ((2, 1), (1, 2)), ((1, 2), (1, 1)))
READOUT = ((1, -1), (-1, 1), (2, 0), (-2, 0))


def matrices():
    base = torch.tensor(BASE, dtype=DTYPE) / 64
    return torch.stack((base, base.flip(-1, -2), base / 2, base * .75))


def oracle_model(kind='tied'):
    model = component.make_model(kind, seed=733, width=2, cost_scale=1.)
    b = matrices()
    found = 1 - b.sum((1, 2))
    with torch.no_grad():
        model.observed_logits.copy_(torch.cat((b.flatten(1, 2), found[:, None]), dim=1).log())
        if kind == 'untied':
            model.blind_logits.copy_(torch.cat((b.sum(1), found[:, None]), dim=1).log())
        model.readout.weight.copy_(torch.tensor(READOUT, dtype=DTYPE))
    return model


def assert_close(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), rtol=2e-12, atol=2e-14)


def rational_matvec(matrix, vector):
    return tuple(sum(matrix[i][j] * vector[j] for j in range(2)) for i in range(2))


def rational_b(action, odor):
    values = BASE[odor]
    if action == 1:
        values = tuple(tuple(values[1 - i][1 - j] for j in range(2)) for i in range(2))
    factor = Fraction(1, 2) if action == 2 else Fraction(3, 4) if action == 3 else Fraction(1)
    return tuple(tuple(Fraction(values[i][j], 64) * factor for j in range(2)) for i in range(2))


def rational_branches(actions):
    for observations in itertools.product(range(4), repeat=len(actions)):
        state = (Fraction(3, 4), Fraction(1, 4))
        for action, odor in zip(actions, observations, strict=True):
            state = rational_matvec(rational_b(action, odor), state)
        weight = sum(state)
        yield weight, tuple(x / weight for x in state)


def state0():
    return torch.tensor([[.75, .25]], dtype=DTYPE)


def test_hand_derived_two_state_event_mass_posterior_and_preobservation_cost():
    model = oracle_model()
    action = torch.tensor([0], dtype=torch.int64)
    blind = model.blind_step(state0(), action)
    assert_close(blind['prior_state'], [[54 / 256, 39 / 256]])
    assert_close(blind['survival_mass'], [93 / 256])
    assert_close(blind['found_increment'], [163 / 256])
    assert_close(blind['cost_contrasts'], [[15 / 256, -15 / 256, 108 / 256, -108 / 256]])
    for odor, numerator in enumerate((48, 24, 12, 9, 163)):
        observed = model.observed_step(state0(), action, torch.tensor([odor], dtype=torch.int64))
        assert_close(observed['evidence'], [numerator / 256])
        assert_close(observed['prior_state'], blind['prior_state'])
        assert_close(observed['cost_contrasts'], blind['cost_contrasts'])
        if odor == 0:
            assert_close(observed['posterior_state'], [[7 / 12, 5 / 12]])
        if odor == 4:
            assert torch.equal(observed['posterior_state'], torch.zeros((1, 2), dtype=DTYPE))


@pytest.mark.parametrize('kind', ('tied', 'untied'))
def test_every_action_is_column_stochastic_including_found(kind):
    model = component.make_model(kind, seed=93)
    operators = model.operators()
    b, found, a, blind_found = (operators[k] for k in ('observed', 'found', 'blind', 'blind_found'))
    assert b.shape == (4, 4, 14, 14) and a.shape == (4, 14, 14)
    assert b.dtype == a.dtype == DTYPE and bool((b > 0).all()) and bool((found > 0).all())
    assert_close(b.sum((1, 2)) + found, torch.ones((4, 14), dtype=DTYPE))
    assert_close(a.sum(1) + blind_found, torch.ones((4, 14), dtype=DTYPE))
    assert_close(a, b.sum(1))


def test_three_horizon_blind_prediction_equals_weighted_conditional_linear_readout():
    model = oracle_model()
    actions = (0, 1, 2)
    state = state0()
    found = 0.
    for horizon, action in enumerate(actions, start=1):
        result = model.blind_step(state, torch.tensor([action], dtype=torch.int64))
        state = result['prior_state']
        found += float(result['found_increment'][0])
        branches = list(rational_branches(actions[:horizon]))
        survival = sum(weight for weight, _ in branches)
        expected = [sum(weight * sum(Fraction(READOUT[i][j]) * posterior[j] for j in range(2))
                        for weight, posterior in branches) for i in range(4)]
        assert_close(result['cost_contrasts'], [[float(x) for x in expected]])
        assert_close(result['survival_mass'], [float(survival)])
        assert found == pytest.approx(float(1 - survival), rel=2e-12, abs=2e-14)
        assert float(result['cost_contrasts'].sum()) == pytest.approx(0., abs=2e-14)


def test_nonlinear_readout_is_a_counterexample_to_branch_marginal_identity():
    branches = list(rational_branches((0,)))
    branch_prediction = sum(weight * posterior[0] ** 2 for weight, posterior in branches)
    blind_first_coordinate = sum(weight * posterior[0] for weight, posterior in branches)
    # Found branches output zero in both expressions. Squaring the marginalized
    # surviving state is not the weighted mean of conditional squared outputs.
    assert branch_prediction != blind_first_coordinate ** 2
    assert float(branch_prediction - blind_first_coordinate ** 2) > .05


def test_zero_state_blind_absorption_and_observed_found_are_exact():
    model = oracle_model()
    state = torch.zeros((1, 2), dtype=DTYPE)
    action = torch.tensor([2], dtype=torch.int64)
    blind = model.blind_step(state, action)
    for key in ('prior_state', 'cost_contrasts', 'survival_mass', 'found_increment'):
        assert not bool(blind[key].any())
    observed = model.observed_step(state, action, torch.tensor([4], dtype=torch.int64))
    assert not bool(observed['posterior_state'].any())
    assert_close(observed['evidence'], [1.])
    with pytest.raises(ValueError):
        model.observed_step(state, action, torch.tensor([0], dtype=torch.int64))


def test_tied_observed_and_blind_losses_both_update_the_same_operators():
    model = oracle_model()
    state = state0()
    action = torch.tensor([0], dtype=torch.int64)
    observed = model.observed_step(state, action, torch.tensor([0], dtype=torch.int64))
    loss = -observed['evidence'].log().sum() + observed['posterior_state'][0, 0].square()
    loss.backward()
    assert model.observed_logits.grad is not None and float(model.observed_logits.grad.abs().sum()) > 0
    model.zero_grad(set_to_none=True)
    model.blind_step(state, action)['cost_contrasts'].square().sum().backward()
    assert model.observed_logits.grad is not None and float(model.observed_logits.grad.abs().sum()) > 0
    assert model.readout.weight.grad is not None and float(model.readout.weight.grad.abs().sum()) > 0


def test_untied_branches_isolate_blind_and_observed_gradients():
    model = oracle_model('untied')
    action = torch.tensor([0], dtype=torch.int64)
    observed_before = model.observed_logits.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.blind_step(state0(), action)['cost_contrasts'].square().sum().backward()
    assert model.observed_logits.grad is None
    assert model.blind_logits.grad is not None and float(model.blind_logits.grad.abs().sum()) > 0
    optimizer.step()
    torch.testing.assert_close(model.observed_logits, observed_before, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    model.observed_step(state0(), action, torch.tensor([0], dtype=torch.int64))['evidence'].log().sum().backward()
    assert model.blind_logits.grad is None
    assert model.observed_logits.grad is not None and float(model.observed_logits.grad.abs().sum()) > 0


def test_impossible_underflowed_observation_fails_instead_of_clamping():
    model = oracle_model()
    with torch.no_grad():
        model.observed_logits[0, :2] = -1e6
    with pytest.raises(ValueError):
        model.observed_step(state0(), torch.tensor([0], dtype=torch.int64), torch.tensor([0], dtype=torch.int64))


@pytest.mark.parametrize('bad', ('dtype', 'negative', 'nan', 'mass', 'shape', 'action_dtype', 'action_range'))
def test_bad_state_and_action_rejected_without_caller_mutation(bad):
    model = oracle_model()
    state, action = state0(), torch.tensor([0], dtype=torch.int64)
    if bad == 'dtype':
        state = state.float()
    elif bad == 'negative':
        state[0] = torch.tensor([1.1, -.1])
    elif bad == 'nan':
        state[0, 0] = float('nan')
    elif bad == 'mass':
        state *= 2
    elif bad == 'shape':
        state = state[0]
    elif bad == 'action_dtype':
        action = action.float()
    else:
        action[0] = 4
    before, before_action = state.clone(), action.clone()
    with pytest.raises(ValueError):
        model.blind_step(state, action)
    torch.testing.assert_close(state, before, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(action, before_action, rtol=0, atol=0, equal_nan=True)


def prefix_fixture():
    # Deterministic features without consuming the caller's RNG.
    prefix = torch.arange(2 * 5 * 31, dtype=torch.float32).reshape(2, 5, 31) / 1000
    lengths = torch.tensor([3, 5], dtype=torch.int64)
    actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64)
    return prefix, lengths, actions


def test_constructor_preserves_rng_and_paired_initialization_costs():
    before = torch.random.get_rng_state().clone()
    tied = component.make_model('tied', seed=19, width=2, cost_scale=.125)
    untied = component.make_model('untied', seed=19, width=2, cost_scale=.125)
    assert torch.equal(torch.random.get_rng_state(), before)
    left, right = tied.state_dict(), untied.state_dict()
    assert set(right) - set(left) == {'blind_logits'}
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    prefix, lengths, actions = prefix_fixture()
    a, b = tied.blind_rollout(prefix, lengths, actions), untied.blind_rollout(prefix, lengths, actions)
    assert_close(a['prefix_state'], b['prefix_state'])
    assert_close(a['prior_states'], b['prior_states'])
    assert_close(a['cost_contrasts'], b['cost_contrasts'])
    assert a['cost_contrasts'].dtype == DTYPE
    assert tied.assimilation.weight_ih.dtype == torch.float32
    assert tied.projection.weight.dtype == tied.readout.weight.dtype == DTYPE
    assert tied.readout.bias is None


def test_prefix_padding_and_future_actions_cannot_change_earlier_forecasts():
    model = oracle_model()
    prefix, lengths, actions = prefix_fixture()
    clean = model.blind_rollout(prefix, lengths, actions)
    poisoned = prefix.clone()
    poisoned[0, 3:] = float('nan')
    altered = actions.clone()
    altered[:, 3] = (altered[:, 3] + 1) % 4
    changed = model.blind_rollout(poisoned, lengths, altered)
    assert_close(clean['prefix_state'], changed['prefix_state'])
    assert_close(clean['prior_states'][:, :3], changed['prior_states'][:, :3])
    assert_close(clean['cost_contrasts'][:, :3], changed['cost_contrasts'][:, :3])
    assert 'observations' not in inspect.signature(model.blind_rollout).parameters
    with pytest.raises(TypeError):
        model.blind_rollout(prefix, lengths, actions, observations=torch.zeros_like(actions))
    assert torch.isnan(poisoned[0, 3:]).all()
    invalid = prefix.clone()
    invalid[1, 1, 0] = float('nan')
    with pytest.raises(ValueError):
        model.blind_rollout(invalid, lengths, actions)


def test_observed_labels_only_affect_later_predictions_and_found_absorbs():
    model = oracle_model()
    prefix, lengths, actions = prefix_fixture()
    labels = torch.zeros_like(actions)
    base = model.observed_rollout(prefix, lengths, actions, labels)
    changed = labels.clone()
    changed[:, 1] = 3
    other = model.observed_rollout(prefix, lengths, actions, changed)
    assert_close(base['cost_contrasts'][:, :2], other['cost_contrasts'][:, :2])
    assert_close(base['prior_states'][:, :2], other['prior_states'][:, :2])
    assert not torch.allclose(base['prior_states'][:, 2], other['prior_states'][:, 2], atol=1e-14, rtol=1e-14)
    terminal = labels.clone()
    terminal[:, 1:] = 4
    ended = model.observed_rollout(prefix, lengths, actions, terminal)
    assert_close(base['cost_contrasts'][:, :2], ended['cost_contrasts'][:, :2])
    assert not bool(ended['posterior_states'][:, 1:].any())
    assert not bool(ended['cost_contrasts'][:, 2:].any())
    assert not bool(ended['prior_states'][:, 2:].any())
    assert_close(ended['evidence'][:, 2:], torch.ones((2, 2), dtype=DTYPE))
    terminal[0, -1] = 0
    with pytest.raises(ValueError):
        model.observed_rollout(prefix, lengths, actions, terminal)


def test_untied_observed_panel_ignores_independent_blind_operator():
    model = oracle_model('untied')
    prefix, lengths, actions = prefix_fixture()
    observations = torch.zeros_like(actions)
    before = model.observed_rollout(prefix, lengths, actions, observations)
    blind_before = model.blind_rollout(prefix, lengths, actions)
    with torch.no_grad():
        model.blind_logits[:, 0, :] += 2
    after = model.observed_rollout(prefix, lengths, actions, observations)
    blind_after = model.blind_rollout(prefix, lengths, actions)
    assert_close(before['prior_states'], after['prior_states'])
    assert_close(before['posterior_states'], after['posterior_states'])
    assert_close(before['cost_contrasts'], after['cost_contrasts'])
    assert not torch.allclose(blind_before['prior_states'], blind_after['prior_states'])


@pytest.mark.parametrize('kind', ('tied', 'untied'))
def test_normal_and_blind_rollout_gradients_and_adam_reach_shared_encoder_and_operators(kind):
    model = oracle_model(kind)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    prefix, lengths, actions = prefix_fixture()
    blind = model.blind_rollout(prefix, lengths, actions)
    observed = model.observed_rollout(prefix, lengths, actions, torch.zeros_like(actions))
    loss = (blind['cost_contrasts'].square().sum() + observed['cost_contrasts'].square().sum()
            - observed['evidence'].log().sum())
    loss.backward()
    for parameter in (model.assimilation.weight_ih, model.projection.weight,
                      model.observed_logits, model.readout.weight):
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0
    optimizer.step()
    changed = ('assimilation.weight_ih', 'projection.weight', 'observed_logits', 'readout.weight')
    if kind == 'untied':
        changed += ('blind_logits',)
    for name in changed:
        parameter = dict(model.named_parameters())[name]
        assert bool(torch.isfinite(parameter).all()) and not torch.equal(parameter, before[name])


@pytest.mark.parametrize('kind', ('tied', 'untied'))
def test_parameter_storage_and_physical_work_have_independent_counts(kind):
    model = component.make_model(kind, seed=93, width=14)
    metadata = model.parameter_metadata()
    # GRU:3*28*(31+28+2); projection:14*(28+1); readout:4*14.
    # Observed columns:4*(4*14+1)*14; independent blind:4*(14+1)*14.
    assert metadata['count'] == metadata['trainable_count'] == (8778 if kind == 'tied' else 9618)
    assert metadata['parameter_bytes'] == (49728 if kind == 'tied' else 56448)
    assert metadata['buffer_bytes'] == 8
    assert metadata['effective_counts'] == {'encode_prefix': 5530,
        'blind': 8778 if kind == 'tied' else 6426, 'observed': 8778,
        'combined': 8778 if kind == 'tied' else 9618}
    prefix, lengths, actions = prefix_fixture()
    blind = model.blind_rollout(prefix, lengths, actions)
    expected = {'prefix_assimilation_calls': 5, 'prefix_assimilation_rows': 8,
        'projection_calls': 1, 'projection_rows': 2, 'prefix_softmax_calls': 1,
        'operator_observed_softmax_calls': 1, 'operator_blind_softmax_calls': int(kind == 'untied'),
        'operator_marginal_sum_calls': 1, 'blind_transition_calls': 4, 'blind_transition_rows': 8,
        'observed_branch_calls': 0, 'observed_branch_rows': 0, 'observed_conditioning_rows': 0,
        'cost_readout_calls': 4, 'cost_readout_rows': 8, 'absorbed_rows': 0}
    assert blind['work'] == expected
    observations = torch.tensor([[0, 4, 4, 4], [0, 4, 4, 4]], dtype=torch.int64)
    observed = model.observed_rollout(prefix, lengths, actions, observations)
    assert observed['work'] == {**expected, 'observed_branch_calls': 1, 'observed_branch_rows': 2,
                              'observed_conditioning_rows': 2, 'absorbed_rows': 4}
