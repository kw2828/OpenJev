"""Fabricated public-token, rational-filter and paired-gradient checks only."""
from fractions import Fraction

import pytest
import torch

from openjev.research.finite_factor_models import FactorModel
from openjev.research.finite_shared_filter_models import ARMS, make_model

SEED = 930101


def tokens(actions=(1, 2), odors=(2, 0, 3), *, padding=0):
    assert len(odors) == len(actions) + 1
    prefix = torch.zeros((1, len(odors) + padding, 31), dtype=torch.float32)
    prefix[0, 0, 9] = 1
    for t, odor in enumerate(odors):
        prefix[0, t, 4 + odor] = 1
        if t:
            prefix[0, t, actions[t - 1]] = 1
    return prefix, torch.tensor([len(odors)], dtype=torch.int64)


def rational_law():
    emission = [[Fraction(4 if odor == state % 4 else 1, 7)
                 for state in range(8)] for odor in range(4)]
    branches = [[[[Fraction(0) for _ in range(8)] for _ in range(8)]
                 for _ in range(4)] for _ in range(4)]
    found = [[Fraction(0) for _ in range(8)] for _ in range(4)]
    for action in range(4):
        for current in range(8):
            weights = [[1 + (next_state + 2 * current + 3 * odor + action) % 7
                        for next_state in range(8)] for odor in range(4)]
            terminal = 3 + (action + current) % 3
            denominator = terminal + sum(sum(row) for row in weights)
            found[action][current] = Fraction(terminal, denominator)
            for odor in range(4):
                for next_state in range(8):
                    branches[action][odor][next_state][current] = Fraction(
                        weights[odor][next_state], denominator)
    return emission, branches, found


def injected(arm='shared_filter'):
    model = make_model(arm, SEED)
    emission, branches, found = rational_law()
    e = torch.tensor([[float(x) for x in row] for row in emission], dtype=torch.float64)
    b = torch.tensor([[[[float(x) for x in row] for row in odor]
                       for odor in action] for action in branches], dtype=torch.float64)
    f = torch.tensor([[float(x) for x in row] for row in found], dtype=torch.float64)
    logits = torch.cat((b.reshape(4, 32, 8), f[:, None]), 1).log()
    with torch.no_grad():
        model.reset_logits.copy_(e.log())
        model.observed_logits.copy_(logits)
        if arm == 'untied_filter':
            model.prefix_observed_logits.copy_(logits)
    return model


def product(matrix, vector):
    return [sum((value * mass for value, mass in zip(row, vector, strict=True)), Fraction(0))
            for row in matrix]


def condition(values):
    return [x / sum(values) for x in values]


def manual_prefix():
    emission, branches, _ = rational_law()
    state = condition([x / 8 for x in emission[2]])
    for action, odor in ((1, 0), (2, 3)):
        state = condition(product(branches[action][odor], state))
    return state


def costs(state):
    return [sum((Fraction(-3 if decision == ((s ^ (s >> 1)) & 3) else 1, 4) * mass
                 for s, mass in enumerate(state)), Fraction(0)) for decision in range(4)]


@pytest.mark.parametrize('arm', ARMS[:2])
def test_rational_prefix_and_blind_unconditional_costs(arm):
    model = injected(arm)
    prefix, lengths = tokens(padding=2)
    expected = manual_prefix()
    torch.testing.assert_close(model.encode_prefix(prefix, lengths),
                               torch.tensor([[float(x) for x in expected]], dtype=torch.float64),
                               rtol=0, atol=3e-15)
    _, branches, _ = rational_law()
    actions = torch.tensor([[0, 3]], dtype=torch.int64)
    result = model.blind_rollout(prefix, lengths, actions)
    for step, action in enumerate((0, 3)):
        marginal = [[sum(branches[action][odor][n][s] for odor in range(4))
                     for s in range(8)] for n in range(8)]
        expected = product(marginal, expected)
        torch.testing.assert_close(result['prior_states'][0, step],
                                   torch.tensor([float(x) for x in expected], dtype=torch.float64),
                                   rtol=0, atol=3e-15)
        torch.testing.assert_close(result['cost_contrasts'][0, step],
                                   torch.tensor([float(x) for x in costs(expected)], dtype=torch.float64),
                                   rtol=0, atol=3e-15)
        assert float(result['survival_mass'][0, step]) == pytest.approx(float(sum(expected)), abs=3e-15)
    assert float(result['survival_mass'][0, 1]) < float(result['survival_mass'][0, 0]) < 1


def test_reset_observation_has_no_transition_or_hazard():
    model = injected()
    prefix, lengths = tokens(actions=(), odors=(2,))
    emission, _, _ = rational_law()
    expected = condition(emission[2])
    torch.testing.assert_close(model.encode_prefix(prefix, lengths),
                               torch.tensor([[float(x) for x in expected]], dtype=torch.float64),
                               rtol=0, atol=2e-15)
    before = model.encode_prefix(prefix, lengths).detach().clone()
    with torch.no_grad():
        model.observed_logits.add_(torch.arange(33, dtype=torch.float64)[None, :, None])
    assert torch.equal(model.encode_prefix(prefix, lengths), before)


def test_full_observed_event_law_and_prediction_precedes_current_label():
    model = injected()
    prefix, lengths = tokens()
    actions = torch.tensor([[0, 3, 2]], dtype=torch.int64)
    observations = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    result = model.observed_rollout(prefix, lengths, actions, observations)
    state = manual_prefix()
    _, branches, found = rational_law()
    events = [sum(product(branch, state)) for branch in branches[0]]
    events.append(sum(found[0][s] * state[s] for s in range(8)))
    torch.testing.assert_close(result['probabilities'][0, 0],
                               torch.tensor([float(x) for x in events], dtype=torch.float64),
                               rtol=0, atol=3e-15)
    posterior = condition(product(branches[0][1], state))
    torch.testing.assert_close(result['posterior_states'][0, 0],
                               torch.tensor([float(x) for x in posterior], dtype=torch.float64),
                               rtol=0, atol=3e-15)
    alternate = model.observed_rollout(prefix, lengths, actions, torch.tensor([[3, 4, 4]]))
    for key in ('cost_contrasts', 'survival_mass', 'probabilities'):
        assert torch.equal(result[key][:, 0], alternate[key][:, 0])
    assert not torch.equal(result['cost_contrasts'][:, 1], alternate['cost_contrasts'][:, 1])
    assert torch.equal(result['cost_contrasts'][:, 2], torch.zeros((1, 4), dtype=torch.float64))
    assert float(result['survival_mass'][0, 2]) == 0
    assert torch.equal(result['probabilities'][0, 2], torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64))


def test_initial_pairing_gru_identity_and_global_rng_preservation():
    rng = torch.random.get_rng_state().clone()
    shared, untied, gru = (make_model(arm, SEED) for arm in ARMS)
    original = FactorModel('learned_learned', SEED)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert type(gru) is FactorModel
    assert torch.equal(shared.reset_logits, untied.reset_logits)
    assert torch.equal(shared.observed_logits, untied.observed_logits)
    assert torch.equal(untied.prefix_observed_logits, untied.observed_logits)
    assert untied.prefix_observed_logits.data_ptr() != untied.observed_logits.data_ptr()
    assert torch.equal(shared.observed_logits, gru.observed_logits)
    for key, value in original.state_dict().items():
        assert torch.equal(value, gru.state_dict()[key])
    prefix, lengths = tokens()
    actions = torch.tensor([[0, 2, 1, 3, 1, 2, 0, 3]], dtype=torch.int64)
    a, b = (m.blind_rollout(prefix, lengths, actions) for m in (shared, untied))
    for key in ('prefix_state', 'prior_states', 'cost_contrasts', 'survival_mass'):
        assert torch.equal(a[key], b[key])


def test_shared_gradient_is_sum_of_separate_prefix_and_forecast_routes():
    shared, untied = injected(), injected('untied_filter')
    prefix, lengths = tokens()
    actions, observations = torch.tensor([[0, 3]]), torch.tensor([[1, 2]])
    def loss(model):
        blind = model.blind_rollout(prefix, lengths, actions)
        observed = model.observed_rollout(prefix, lengths, actions, observations)
        weights = torch.tensor([1., 2., 4., 8.], dtype=torch.float64)
        return ((blind['cost_contrasts'] * weights).sum()
                + (observed['cost_contrasts'] * weights.flip(0)).sum()
                - observed['probabilities'][0, 1, 2].log()
                + blind['survival_mass'].square().sum())
    for model in (shared, untied):
        loss(model).backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert bool((parameter.grad != 0).any())
    torch.testing.assert_close(shared.observed_logits.grad,
                               untied.observed_logits.grad + untied.prefix_observed_logits.grad,
                               rtol=2e-13, atol=2e-15)
    torch.testing.assert_close(shared.reset_logits.grad, untied.reset_logits.grad, rtol=2e-13, atol=2e-15)
    before = shared.costs.clone()
    torch.optim.Adam(shared.parameters(), lr=.003).step()
    assert torch.equal(before, shared.costs)


def test_untied_prefix_only_gradient_does_not_touch_forecast_parameters():
    model = injected('untied_filter')
    prefix, lengths = tokens()
    (model.encode_prefix(prefix, lengths) * torch.arange(8, dtype=torch.float64)).sum().backward()
    assert model.observed_logits.grad is None
    assert bool((model.prefix_observed_logits.grad != 0).any())
    assert bool((model.reset_logits.grad != 0).any())


def test_work_counts_include_separate_prefix_and_forecast_normalizations():
    first, first_lengths = tokens(actions=(1, 2, 3), odors=(0, 1, 2, 3))
    second, second_lengths = tokens(actions=(2,), odors=(3, 1), padding=2)
    prefix, lengths = torch.cat((first, second)), torch.cat((first_lengths, second_lengths))
    result = make_model('shared_filter', SEED).blind_rollout(prefix, lengths, torch.tensor([[0, 1], [2, 3]]))
    work = result['work']
    assert {key: work[key] for key in ('prefix_filter_calls', 'prefix_filter_rows',
                                      'reset_emission_softmax_calls', 'reset_emission_rows',
                                      'prefix_operator_softmax_calls', 'operator_observed_softmax_calls')} == {
        'prefix_filter_calls': 3, 'prefix_filter_rows': 4, 'reset_emission_softmax_calls': 1,
        'reset_emission_rows': 2, 'prefix_operator_softmax_calls': 1, 'operator_observed_softmax_calls': 1}
    assert work['prefix_assimilation_calls'] == work['projection_calls'] == 0
    assert work['blind_transition_calls'] == 2 and work['blind_transition_rows'] == 4
    assert all(type(value) is int and value >= 0 for value in work.values())


@pytest.mark.parametrize('arm,count', [('shared_filter', 1088), ('untied_filter', 2144)])
def test_exact_parameter_inventory_precision_and_storage(arm, count):
    model = make_model(arm, SEED)
    metadata = model.parameter_metadata()
    assert metadata['count'] == metadata['trainable_count'] == count
    assert metadata['parameter_bytes'] == count * 8
    assert metadata['buffer_bytes'] == 256
    assert metadata['operator_initialization_seed'] == SEED ^ 0x9E3779B9
    assert metadata['reset_initialization_seed'] == SEED ^ 0x85EBCA6B
    assert set(dict(model.named_buffers())) == {'costs'}
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert not metadata['privileged_prefix'] and not metadata['known_operators']
    with torch.no_grad():
        model.costs[0, 0] += .01
    with pytest.raises(ValueError, match='unchanged fixed'):
        model.parameter_metadata()


@pytest.mark.parametrize('corruption', ['reset_action', 'missing_reset', 'second_reset', 'two_actions',
                                       'two_odors', 'found_prefix', 'hidden_feature', 'padding', 'nan_padding'])
def test_public_grammar_and_padding_reject_malformed_input(corruption):
    prefix, lengths = tokens(padding=1)
    if corruption == 'reset_action':
        prefix[0, 0, 0] = 1
    elif corruption == 'missing_reset':
        prefix[0, 0, 9] = 0
    elif corruption == 'second_reset':
        prefix[0, 1, 9] = 1
    elif corruption == 'two_actions':
        prefix[0, 1, 0] = 1
    elif corruption == 'two_odors':
        prefix[0, 1, 5] = 1
    elif corruption == 'found_prefix':
        prefix[0, 1, 4:8] = 0
        prefix[0, 1, 8] = 1
    elif corruption == 'hidden_feature':
        prefix[0, 1, 10] = 1
    else:
        prefix[0, -1, 0] = float('nan') if corruption == 'nan_padding' else 1
    with pytest.raises(ValueError):
        make_model('shared_filter', SEED).encode_prefix(prefix, lengths)


def test_no_oracle_api_and_absorbing_input_guard():
    model = make_model('shared_filter', SEED)
    prefix, lengths = tokens()
    actions = torch.tensor([[0, 1]])
    oracle = torch.full((1, 8), 1 / 8, dtype=torch.float64)
    with pytest.raises(TypeError):
        model.encode_prefix(prefix, lengths, oracle_prefix=oracle)
    with pytest.raises(TypeError):
        model.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
    with pytest.raises(TypeError):
        model.observed_rollout(prefix, lengths, actions, actions, oracle_prefix=oracle)
    with pytest.raises(ValueError, match='absorbing found suffix'):
        model.observed_rollout(prefix, lengths, actions, torch.tensor([[4, 0]]))


@pytest.mark.parametrize('parameter', ['reset_logits', 'observed_logits', 'prefix_observed_logits'])
def test_finite_softmax_underflow_is_not_clipped(parameter):
    model = make_model('untied_filter', SEED)
    with torch.no_grad():
        getattr(model, parameter).reshape(-1)[0] = -10000
    prefix, lengths = tokens()
    with pytest.raises(ValueError, match='underflow'):
        model.blind_rollout(prefix, lengths, torch.tensor([[0]]))


def test_future_actions_do_not_change_earlier_outputs_and_inputs_are_owned():
    model = injected()
    prefix, lengths = tokens(padding=1)
    saved = prefix.clone()
    first = model.blind_rollout(prefix, lengths, torch.tensor([[0, 1, 2]]))
    other = model.blind_rollout(prefix, lengths, torch.tensor([[0, 1, 3]]))
    assert torch.equal(first['cost_contrasts'][:, :2], other['cost_contrasts'][:, :2])
    assert torch.equal(prefix, saved)
    operators = model.operators()
    operators['observed'].detach().zero_()
    assert bool((model.operators()['observed'] > 0).all())


@pytest.mark.parametrize('arm,seed', [('other', SEED), ('shared_filter', True), ('untied_filter', -1)])
def test_invalid_factory_inputs(arm, seed):
    with pytest.raises(ValueError):
        make_model(arm, seed)
