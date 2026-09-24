"""Engineering-only scalar and causal checks; no scientific seed or generator."""
import math
from fractions import Fraction

import pytest
import torch

from openjev.research.finite_cost_readout_models import make_model as old_model
from openjev.research.finite_cost_readout_models import prefix_predictions as old_prefix
from openjev.research.finite_factorized_dynamics_models import (
    ARMS,
    FACTOR_WORK,
    FACTOR_WORK_KEYS,
    HAZARD_SEED_XOR,
    make_model,
    prefix_predictions,
)
from openjev.research.otto_observation_operator_model import _work

SEED = 934101


def public_prefix(found_step=None):
    prefix = torch.zeros((1, 9, 31), dtype=torch.float32)
    prefix[0, 0, 6] = prefix[0, 0, 9] = 1
    length = 9 if found_step is None else found_step + 1
    for step in range(1, length):
        prefix[0, step, (step + 1) % 4] = 1
        prefix[0, step, 8 if step == found_step else 4 + step % 4] = 1
    return prefix, torch.tensor([length], dtype=torch.int64)


def blocks():
    return (torch.tensor([[0, 1, 3, 2, 2, 1, 0, 3]], dtype=torch.int64),
            torch.tensor([[0, 2, 1, 3, 1, 0, 4, 4]], dtype=torch.int64))


def all_rollouts(model):
    prefix, lengths = public_prefix()
    actions, observations = blocks()
    return model.blind_rollout(prefix, lengths, actions), model.observed_rollout(prefix, lengths, actions, observations)


def assert_outputs(a, b, *, exact=False):
    assert set(a) == set(b)
    for key in a:
        if key == 'work':
            continue
        if a[key] is None:
            assert b[key] is None
        elif exact:
            assert torch.equal(a[key], b[key]), key
        else:
            torch.testing.assert_close(a[key], b[key], atol=1e-12, rtol=0, msg=key)


def rational_world():
    # This fabricated world deliberately differs from the registered task.
    transition = [[[Fraction(1 + 8 * (n == (s + a) % 8), 16) for s in range(8)]
                   for n in range(8)] for a in range(4)]
    emission = [[Fraction(5 if o == s % 4 else 1, 8) for s in range(8)] for o in range(4)]
    hazard = [[Fraction(1 + (a + n) % 3, 32) for n in range(8)] for a in range(4)]
    b = [[[[emission[o][n] * (1 - hazard[a][n]) * transition[a][n][s] for s in range(8)]
           for n in range(8)] for o in range(4)] for a in range(4)]
    found = [[sum(hazard[a][n] * transition[a][n][s] for n in range(8)) for s in range(8)] for a in range(4)]
    return transition, emission, hazard, b, found


def f64(values):
    if isinstance(values, list):
        return torch.stack([f64(value) for value in values])
    return torch.tensor(float(values), dtype=torch.float64)


def inject_rational(model):
    transition, emission, hazard, branches, found = rational_world()
    with torch.no_grad():
        if model.study_arm == 'factorized':
            model.transition_logits.copy_(f64(transition).log())
            model.emission_logits.copy_(f64(emission).log())
            model.hazard_logits.copy_((f64(hazard) / (1 - f64(hazard))).log())
        else:
            model.reset_logits.copy_(f64(emission).log())
            model.observed_logits.copy_(torch.cat((f64(branches).reshape(4, 32, 8), f64(found)[:, None]), 1).log())
    return emission, branches, found


def scalar_event(state, action, branches, found):
    masses = [[sum(branches[action][o][n][s] * state[s] for s in range(8)) for n in range(8)] for o in range(4)]
    law = [sum(row) for row in masses] + [sum(found[action][s] * state[s] for s in range(8))]
    return masses, law


def scalar_prefix(prefix, lengths, emission, branches, found):
    label = int(prefix[0, 0, 4:9].argmax())
    reset = [sum(emission[o]) / 8 for o in range(4)] + [Fraction(0)]
    state = [emission[label][s] / 8 / reset[label] for s in range(8)]
    laws = [reset]
    for step in range(1, int(lengths[0])):
        action = int(prefix[0, step, :4].argmax())
        label = int(prefix[0, step, 4:9].argmax())
        masses, law = scalar_event(state, action, branches, found)
        laws.append(law)
        state = [Fraction(0)] * 8 if label == 4 else [value / law[label] for value in masses[label]]
    laws.extend([[Fraction(0)] * 5 for _ in range(9 - len(laws))])
    return state, laws


def test_initial_functions_match_and_local_rng_is_unchanged():
    rng = torch.random.get_rng_state().clone()
    candidate, matched = make_model('factorized', SEED), make_model('matched_free', SEED)
    assert torch.equal(rng, torch.random.get_rng_state())
    a, b = candidate.dynamics_snapshot(), matched.dynamics_snapshot()
    for key in ('reset_emission', 'observed', 'found', 'blind'):
        torch.testing.assert_close(a[key], b[key], atol=1e-12, rtol=0)
    assert torch.equal(candidate.cost_logits, matched.cost_logits)
    for first, second in zip(all_rollouts(candidate), all_rollouts(matched), strict=True):
        assert_outputs(first, second)
    prefix, lengths = public_prefix()
    actions, observations = blocks()
    assert_outputs(candidate.blind_rollout(prefix, lengths, actions[:, :2]),
                   matched.blind_rollout(prefix, lengths, actions[:, :2]))
    assert_outputs(candidate.observed_rollout(prefix, lengths, actions[:, :2], observations[:, :2]),
                   matched.observed_rollout(prefix, lengths, actions[:, :2], observations[:, :2]))
    for terminal in (None, 1, 4, 8):
        prefix, lengths = public_prefix(terminal)
        assert_outputs(prefix_predictions(candidate, prefix, lengths), prefix_predictions(matched, prefix, lengths))


def test_dense_free_is_bitwise_the_existing_learned_head_model():
    new, old = make_model('dense_free', SEED), old_model('learned_readout', SEED)
    assert set(new.state_dict()) == set(old.state_dict())
    assert all(torch.equal(value, old.state_dict()[name]) for name, value in new.state_dict().items())
    for first, second in zip(all_rollouts(new), all_rollouts(old), strict=True):
        assert_outputs(first, second, exact=True)
        assert first['work'] == {**second['work'], **dict.fromkeys(FACTOR_WORK_KEYS, 0)}
    prefix, lengths = public_prefix(5)
    a, b = prefix_predictions(new, prefix, lengths), old_prefix(old, prefix, lengths)
    assert_outputs(a, b, exact=True)
    assert a['work'] == {**b['work'], **dict.fromkeys(FACTOR_WORK_KEYS, 0)}


def test_generic_flat_factorization_matches_33_way_initializer_without_true_dynamics():
    model = make_model('factorized', SEED)
    with torch.no_grad():
        model.transition_logits.zero_()
        model.emission_logits.zero_()
        model.hazard_logits.fill_(-math.log(32))
    snapshot = model.dynamics_snapshot()
    torch.testing.assert_close(snapshot['observed'], torch.full((4, 4, 8, 8), 1 / 33, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(snapshot['found'], torch.full((4, 8), 1 / 33, dtype=torch.float64), atol=1e-15, rtol=0)
    prefix, lengths = public_prefix()
    torch.testing.assert_close(model.encode_prefix(prefix, lengths), torch.full((1, 8), 1 / 8, dtype=torch.float64), atol=1e-15, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_independent_fraction_prefix_and_forecast_oracle(arm):
    model = make_model(arm, SEED)
    emission, branches, found = inject_rational(model)
    snapshot = model.dynamics_snapshot()
    torch.testing.assert_close(snapshot['observed'], f64(branches), atol=1e-15, rtol=0)
    torch.testing.assert_close(snapshot['found'], f64(found), atol=1e-15, rtol=0)
    prefix, lengths = public_prefix()
    state, laws = scalar_prefix(prefix, lengths, emission, branches, found)
    torch.testing.assert_close(model.encode_prefix(prefix, lengths)[0], f64(state), atol=1e-12, rtol=0)
    likelihood = prefix_predictions(model, prefix, lengths)
    torch.testing.assert_close(likelihood['probabilities'][0], f64(laws), atol=1e-12, rtol=0)
    actions, observations = blocks()
    blind, observed = all_rollouts(model)
    blind_state, observed_state = state[:], state[:]
    preferred = [0, 1, 3, 2, 2, 3, 1, 0]
    costs = [[Fraction(-27, 40) if d == preferred[s] else Fraction(9, 40) for s in range(8)] for d in range(4)]
    for step in range(8):
        action, label = int(actions[0, step]), int(observations[0, step])
        masses, _law = scalar_event(blind_state, action, branches, found)
        blind_state = [sum(masses[o][n] for o in range(4)) for n in range(8)]
        expected_cost = [sum(costs[d][s] * blind_state[s] for s in range(8)) for d in range(4)]
        torch.testing.assert_close(blind['cost_contrasts'][0, step], f64(expected_cost), atol=1e-12, rtol=0)
        assert float(blind['survival_mass'][0, step]) == pytest.approx(float(sum(blind_state)), abs=1e-12)
        if sum(observed_state) == 0:
            expected_law = [Fraction(0)] * 4 + [Fraction(1)]
            expected_prior = [Fraction(0)] * 8
        else:
            masses, expected_law = scalar_event(observed_state, action, branches, found)
            expected_prior = [sum(masses[o][n] for o in range(4)) for n in range(8)]
            observed_state = [Fraction(0)] * 8 if label == 4 else [v / expected_law[label] for v in masses[label]]
        torch.testing.assert_close(observed['probabilities'][0, step], f64(expected_law), atol=1e-12, rtol=0)
        torch.testing.assert_close(observed['prior_states'][0, step], f64(expected_prior), atol=1e-12, rtol=0)
        torch.testing.assert_close(observed['posterior_states'][0, step], f64(observed_state), atol=1e-12, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_terminal_prefix_scored_once_and_padding_zero(arm):
    model = make_model(arm, SEED)
    emission, branches, found = inject_rational(model)
    prefix, lengths = public_prefix(3)
    _state, laws = scalar_prefix(prefix, lengths, emission, branches, found)
    result = prefix_predictions(model, prefix, lengths)
    torch.testing.assert_close(result['probabilities'][0], f64(laws), atol=1e-12, rtol=0)
    assert torch.equal(result['nll'][:, 4:], torch.zeros((1, 5), dtype=torch.float64))
    for step in range(4):
        label = int(prefix[0, step, 4:9].argmax())
        expected = -f64(laws[step][label]).log()
        torch.testing.assert_close(result['nll'][0, step], expected, atol=1e-12, rtol=0)
    assert result['work']['prefix_nll_rows'] == result['work']['prefix_probability_rows'] == 4
    assert result['work']['prefix_filter_rows'] == 2


def objective(model):
    blind, observed = all_rollouts(model)
    prefix, lengths = public_prefix(4)
    target = torch.tensor([.2, -.3, .4, -.3], dtype=torch.float64)
    return ((blind['cost_contrasts'][:, :2] - target).square().mean()
            + (observed['cost_contrasts'][:, :2] - target.flip(0)).square().mean()
            + blind['survival_mass'][:, :2].square().mean()
            - observed['probabilities'][0, 1, 2].log()
            + prefix_predictions(model, prefix, lengths)['nll'].sum() / lengths.sum())


@pytest.mark.parametrize('arm', ARMS)
def test_all_parameters_receive_combined_gradients_and_adam_updates(arm):
    model = make_model(arm, SEED)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss = objective(model)
    loss.backward()
    for value in model.parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all() and bool((value.grad != 0).any())
    torch.optim.Adam(model.parameters(), lr=.003).step()
    assert all(not torch.equal(value, original[name]) for name, value in model.named_parameters())
    assert torch.isfinite(objective(model))
    assert not dict(model.named_buffers())


def test_shared_emission_has_reset_and_future_gradient_paths():
    model = make_model('factorized', SEED)
    snapshot = model.dynamics_snapshot()
    reset_gradient = torch.autograd.grad(snapshot['reset_emission'][0, 0], model.emission_logits, retain_graph=True)[0]
    future_gradient = torch.autograd.grad(snapshot['observed'][1, 2, 3, 4], model.emission_logits)[0]
    assert bool((reset_gradient != 0).any()) and bool((future_gradient != 0).any())
    assert not hasattr(model, 'reset_logits')


@pytest.mark.parametrize('arm', ARMS)
def test_prefix_likelihood_has_no_head_gradient(arm):
    model = make_model(arm, SEED)
    prefix, lengths = public_prefix(4)
    prefix_predictions(model, prefix, lengths)['nll'].sum().backward()
    assert model.cost_logits.grad is None
    assert all(value.grad is not None for name, value in model.named_parameters() if name != 'cost_logits')


@pytest.mark.parametrize('arm', ARMS)
def test_probability_mass_linear_costs_and_absorption(arm):
    model = make_model(arm, SEED)
    operators = model.operators()
    torch.testing.assert_close(operators['observed'].sum((1, 2)) + operators['found'],
                               torch.ones((4, 8), dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.equal(operators['blind'], operators['observed'].sum(1))
    first = torch.tensor([[.1, .2, .05, .15, .1, .2, .05, .15]], dtype=torch.float64)
    second = first.flip(-1)
    expected = .3 * model._read(first, _work()) + .7 * model._read(second, _work())
    torch.testing.assert_close(model._read(.3 * first + .7 * second, _work()), expected, atol=1e-15, rtol=0)
    torch.testing.assert_close(model._read(.4 * first, _work()), .4 * model._read(first, _work()), atol=1e-15, rtol=0)
    zero = model.observed_step(torch.zeros_like(first), torch.tensor([0]), torch.tensor([4]))
    assert not zero['cost_contrasts'].any() and not zero['survival_mass'].any()
    assert float(zero['evidence'][0]) == 1
    with pytest.raises(ValueError):
        model.observed_step(torch.zeros_like(first), torch.tensor([0]), torch.tensor([0]))


@pytest.mark.parametrize('arm', ARMS)
def test_current_labels_and_future_actions_do_not_change_prior_predictions(arm):
    model = make_model(arm, SEED)
    prefix, lengths = public_prefix()
    actions, observations = blocks()
    other = observations.clone()
    other[0, 1] = 3
    a = model.observed_rollout(prefix, lengths, actions, observations)
    b = model.observed_rollout(prefix, lengths, actions, other)
    assert torch.equal(a['cost_contrasts'][:, :2], b['cost_contrasts'][:, :2])
    assert torch.equal(a['probabilities'][:, :2], b['probabilities'][:, :2])
    other_actions = actions.clone()
    other_actions[0, -1] = 2
    a = model.blind_rollout(prefix, lengths, actions)
    b = model.blind_rollout(prefix, lengths, other_actions)
    assert torch.equal(a['cost_contrasts'][:, :7], b['cost_contrasts'][:, :7])
    altered = prefix.clone()
    altered[0, 4, 4:9] = 0
    altered[0, 4, 7] = 1
    a, b = prefix_predictions(model, prefix, lengths), prefix_predictions(model, altered, lengths)
    assert torch.equal(a['probabilities'][:, :5], b['probabilities'][:, :5])


@pytest.mark.parametrize('arm', ARMS)
def test_owned_diagnostics_and_inputs_are_not_mutated(arm):
    model = make_model(arm, SEED)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    prefix, lengths = public_prefix()
    original = prefix.clone()
    result = model.dynamics_snapshot()
    for key in ('reset_emission', 'observed', 'found', 'blind'):
        result[key].detach().zero_()
    model.readout_matrix().detach().zero_()
    prefix_predictions(model, prefix, lengths)
    assert torch.equal(prefix, original)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('arm', ARMS)
def test_actual_storage_and_work_geometry(arm):
    model = make_model(arm, SEED)
    metadata = model.parameter_metadata()
    count = 352 if arm == 'factorized' else 1120
    assert metadata['arm'] == arm and metadata['count'] == metadata['trainable_count'] == count
    assert metadata['parameter_bytes'] == 8 * count and metadata['buffer_bytes'] == 0
    assert all(value.dtype == torch.float64 and value.device.type == 'cpu' for value in model.parameters())
    assert metadata['hazard_initialization_seed'] == (None if arm == 'dense_free' else SEED ^ HAZARD_SEED_XOR)
    assert set(dict(model.named_parameters())) == ({'transition_logits', 'emission_logits', 'hazard_logits', 'cost_logits'}
                                                  if arm == 'factorized' else {'reset_logits', 'observed_logits', 'cost_logits'})
    for result in all_rollouts(model):
        work = result['work']
        for key, amount in FACTOR_WORK.items():
            assert work[key] == (2 * amount if arm == 'factorized' else 0)
        assert work['operator_marginal_sum_calls'] == 1
        assert work['reset_emission_softmax_calls'] == work['prefix_operator_softmax_calls'] == work['operator_observed_softmax_calls'] == (0 if arm == 'factorized' else 1)
        assert work['prefix_filter_rows'] == 8 and work['reset_emission_rows'] == 1
        assert work['cost_head_softmax_calls'] == 8 and work['cost_head_probability_rows'] == 64
    prefix, lengths = public_prefix(3)
    work = prefix_predictions(model, prefix, lengths)['work']
    for key, amount in FACTOR_WORK.items():
        assert work[key] == (amount if arm == 'factorized' else 0)
    assert work['cost_head_softmax_calls'] == work['cost_head_probability_rows'] == 0
    snapshot = model.dynamics_snapshot()
    assert snapshot['work']['operator_marginal_sum_calls'] == 1
    assert all(type(value) is int and value >= 0 for value in snapshot['work'].values())
    construction = metadata['construction_work']
    for key, amount in FACTOR_WORK.items():
        assert construction[key] == (amount if arm == 'matched_free' else 0)
    assert construction['matching_log_calls'] == (2 if arm == 'matched_free' else 0)
    assert construction['matching_log_entries'] == (1088 if arm == 'matched_free' else 0)


@pytest.mark.parametrize('arm', ARMS)
def test_invalid_public_input_and_oracle_arguments_are_rejected(arm):
    model = make_model(arm, SEED)
    prefix, lengths = public_prefix(3)
    with pytest.raises(TypeError):
        prefix_predictions(model, prefix, lengths, oracle_prefix=torch.ones((1, 8)))
    with pytest.raises(TypeError):
        model.encode_prefix(prefix, lengths, oracle_prefix=torch.ones((1, 8)))
    prefix[0, 5, 0] = 1
    with pytest.raises(ValueError):
        prefix_predictions(model, prefix, lengths)
    prefix, lengths = public_prefix()
    prefix[0, 2, 20] = 1
    with pytest.raises(ValueError):
        model.blind_rollout(prefix, lengths, blocks()[0])


@pytest.mark.parametrize('name,value', [('transition_logits', -10000.), ('emission_logits', -10000.),
    ('hazard_logits', 10000.), ('hazard_logits', -10000.), ('cost_logits', float('nan'))])
def test_factor_underflow_saturation_and_nonfinite_fail_without_parameter_repair(name, value):
    model = make_model('factorized', SEED)
    with torch.no_grad():
        getattr(model, name).reshape(-1)[0] = value
    before = getattr(model, name).detach().clone()
    with pytest.raises(ValueError):
        all_rollouts(model)
    torch.testing.assert_close(getattr(model, name), before, equal_nan=True)


@pytest.mark.parametrize('arm,seed', [('other', SEED), ('factorized', True), ('dense_free', -1)])
def test_invalid_factory(arm, seed):
    with pytest.raises(ValueError):
        make_model(arm, seed)
