"""Fabricated engineering namespace only; no registered study cases are used."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from openjev.research import finite_observation_world as fw
from openjev.research.finite_prefix_learning import (
    ARMS,
    generate_attempt_split,
    make_model,
    prefix_predictions,
)

NAMESPACE, SEED = 931001, 931101


def tokens(*, found_step=None):
    prefix = torch.zeros((1, 9, 31), dtype=torch.float32)
    prefix[0, 0, 6] = prefix[0, 0, 9] = 1
    length = 9 if found_step is None else found_step + 1
    for step in range(1, length):
        prefix[0, step, (step + 1) % 4] = 1
        prefix[0, step, 8 if step == found_step else 4 + (2 * step + 1) % 4] = 1
    return prefix, torch.tensor([length], dtype=torch.int64)


def rational_world():
    emission = [[Fraction(22 if odor == state % 4 else 1, 25)
                 for state in range(8)] for odor in range(4)]
    branches = [[[[Fraction(0) for _ in range(8)] for _ in range(8)]
                 for _ in range(4)] for _ in range(4)]
    found = [[Fraction(0) for _ in range(8)] for _ in range(4)]
    for action in range(4):
        for state in range(8):
            destination = (state ^ 1, (state + 1) % 8,
                           ((state << 1) & 7) | (state >> 2), state ^ 4)[action]
            for next_state in range(8):
                transition = Fraction(1, 400) + (Fraction(49, 50) if next_state == destination else 0)
                hazard = Fraction(1 + (((next_state >> 2) & 1) ^ (action & 1)), 200)
                found[action][state] += transition * hazard
                for odor in range(4):
                    branches[action][odor][next_state][state] = (
                        transition * (1 - hazard) * emission[odor][next_state])
    return emission, branches, found


def exact_model():
    model = make_model('endpoint_only', SEED)
    emission, branches, found = rational_world()
    e = torch.tensor([[float(x) for x in row] for row in emission], dtype=torch.float64)
    b = torch.tensor([[[[float(x) for x in row] for row in odor]
                       for odor in action] for action in branches], dtype=torch.float64)
    f = torch.tensor([[float(x) for x in row] for row in found], dtype=torch.float64)
    with torch.no_grad():
        model.reset_logits.copy_(e.log())
        model.observed_logits.copy_(torch.cat((b.reshape(4, 32, 8), f[:, None]), 1).log())
    return model


def manual_laws(prefix, length):
    emission, branches, found = rational_world()
    state = [Fraction(1, 8)] * 8
    result = [[sum(emission[odor][s] * state[s] for s in range(8)) for odor in range(4)] + [Fraction(0)]]
    label = int(prefix[0, 0, 4:9].argmax())
    mass = [emission[label][s] * state[s] for s in range(8)]
    state = [value / sum(mass) for value in mass]
    for step in range(1, length):
        action, label = int(prefix[0, step, :4].argmax()), int(prefix[0, step, 4:9].argmax())
        joint = [[sum(branches[action][odor][n][s] * state[s] for s in range(8))
                  for n in range(8)] for odor in range(4)]
        probabilities = [sum(row) for row in joint] + [sum(found[action][s] * state[s] for s in range(8))]
        result.append(probabilities)
        if label == 4:
            break
        state = [value / probabilities[label] for value in joint[label]]
    result += [[Fraction(0)] * 5] * (9 - length)
    return torch.tensor([[float(x) for x in row] for row in result], dtype=torch.float64)


@pytest.mark.parametrize('split,horizon', [(0, 2), (1, 8), (2, 4)])
def test_surviving_cases_bitwise_match_frozen_generator(split, horizon):
    new = generate_attempt_split(split, 32, horizon, seed_namespace=NAMESPACE)
    old = fw.generate_split(split, 32, horizon, include_oracle=True, seed_namespace=NAMESPACE)
    assert set(new['data']) == set(old['data'])
    for name in old['data']:
        assert new['data'][name].dtype == old['data'][name].dtype
        assert np.array_equal(new['data'][name], old['data'][name])
    assert np.array_equal(new['oracle']['prefix_state'], old['oracle']['prefix_beliefs'][:, -1])
    p, c = new['prefix_data'], new['counts']
    assert set(p) == {'prefix', 'lengths', 'case_ids', 'event_mask', 'endpoint_eligible', 'endpoint_rows'}
    assert p['case_ids'].dtype == np.dtype('<U64')
    assert np.array_equal(p['event_mask'], np.arange(9)[None] < p['lengths'][:, None])
    assert np.array_equal(p['endpoint_rows'][p['endpoint_eligible']], np.arange(len(new['data']['prefix'])))
    assert np.all(p['endpoint_rows'][~p['endpoint_eligible']] == -1)
    assert np.array_equal(p['case_ids'][p['endpoint_eligible']], new['data']['case_ids'])
    assert np.array_equal(p['prefix'][p['endpoint_eligible']], new['data']['prefix'])
    assert np.all(p['prefix'][~p['event_mask']] == 0)
    assert c['attempted'] == c['retained'] + c['discarded_found'] == 32
    assert c['valid_prefix_events'] == int(p['lengths'].sum()) == 32 + c['prefix_event_draws']
    assert c['retained'] == old['counts']['retained'] and c['discarded_found'] == old['counts']['excluded_found']
    for name in ('prefix_found_by_step', 'initial_odor_draws', 'prefix_event_draws', 'prefix_action_draws',
                 'forecast_action_draws', 'forecast_event_draws', 'forecast_found_cases'):
        assert c[name] == old['counts'][name]


def test_first_found_is_retained_without_replacements(monkeypatch):
    draws = []
    def first_found(_rng, probabilities):
        draws.append(len(probabilities))
        return 1 if len(probabilities) == 4 else 4
    monkeypatch.setattr(fw, '_sample', first_found)
    result = generate_attempt_split(0, 3, 2, seed_namespace=NAMESPACE)
    prefix, counts = result['prefix_data'], result['counts']
    assert draws == [4, 5] * 3
    assert counts['retained'] == 0 and counts['discarded_found'] == 3
    assert counts['valid_prefix_events'] == 6 and counts['prefix_action_draws'] == 24
    assert counts['forecast_action_draws'] == counts['forecast_event_draws'] == 0
    assert counts['prefix_found_by_step'] == [3] + [0] * 7
    assert np.array_equal(prefix['lengths'], [2, 2, 2])
    assert np.all(prefix['prefix'][:, 1, 8] == 1) and np.all(prefix['prefix'][:, 2:] == 0)
    assert np.array_equal(prefix['event_mask'].sum(-1), [2, 2, 2])
    assert result['data']['prefix'].shape == (0, 9, 31)
    assert result['data']['observed_probabilities'].shape == (0, 2, 5)
    assert result['oracle']['prefix_state'].shape == (0, 8)


def test_last_step_found_is_ineligible_despite_full_length(monkeypatch):
    event_index = 0
    def delayed_found(_rng, probabilities):
        nonlocal event_index
        if len(probabilities) == 4:
            return 0
        event_index += 1
        return 4 if event_index == 8 else 0
    monkeypatch.setattr(fw, '_sample', delayed_found)
    result = generate_attempt_split(0, 1, 2, seed_namespace=NAMESPACE)
    p = result['prefix_data']
    assert p['lengths'][0] == 9 and p['event_mask'][0].all()
    assert not p['endpoint_eligible'][0] and p['endpoint_rows'][0] == -1
    assert p['prefix'][0, 8, 8] == 1 and result['counts']['valid_prefix_events'] == 9


def test_empty_shapes_ownership_and_rng_isolation():
    rng = np.random.get_state()
    empty = generate_attempt_split(0, 0, 2, seed_namespace=NAMESPACE)
    assert empty['prefix_data']['prefix'].shape == (0, 9, 31)
    assert empty['prefix_data']['event_mask'].shape == (0, 9)
    assert empty['counts']['valid_prefix_events'] == 0
    first = generate_attempt_split(0, 2, 2, seed_namespace=NAMESPACE)
    second = generate_attempt_split(0, 2, 2, seed_namespace=NAMESPACE)
    for before, after in zip(rng, np.random.get_state(), strict=True):
        assert np.array_equal(before, after)
    first['prefix_data']['prefix'].fill(9)
    assert np.all(second['prefix_data']['prefix'] <= 1)
    assert np.all(first['data']['prefix'] <= 1)


@pytest.mark.parametrize('found_step', [None, 1, 3, 8])
def test_exact_world_prelabel_probabilities_and_terminal_masks(found_step):
    prefix, lengths = tokens(found_step=found_step)
    result = prefix_predictions(exact_model(), prefix, lengths)
    expected = manual_laws(prefix, int(lengths[0]))
    torch.testing.assert_close(result['probabilities'][0], expected, rtol=0, atol=3e-15)
    for step in range(9):
        if step < int(lengths[0]):
            label = int(prefix[0, step, 4:9].argmax())
            torch.testing.assert_close(result['nll'][0, step], -expected[step, label].log(), rtol=0, atol=3e-14)
        else:
            assert float(result['nll'][0, step]) == 0
            assert torch.equal(result['probabilities'][0, step], torch.zeros(5, dtype=torch.float64))
    assert float(result['probabilities'][0, 0, 4]) == 0
    assert result['work']['prefix_nll_rows'] == result['work']['prefix_probability_rows'] == int(lengths[0])


def test_current_label_changes_nll_but_not_its_prediction():
    model = exact_model()
    prefix, lengths = tokens()
    alternative = prefix.clone()
    alternative[0, 2, 4:9] = 0
    alternative[0, 2, 4] = 1
    first = prefix_predictions(model, prefix, lengths)
    changed = prefix_predictions(model, alternative, lengths)
    assert torch.equal(first['probabilities'][:, :3], changed['probabilities'][:, :3])
    assert not torch.equal(first['nll'][:, 2], changed['nll'][:, 2])
    assert not torch.equal(first['probabilities'][:, 3], changed['probabilities'][:, 3])


def test_reset_prediction_is_independent_of_reset_label():
    prefix, lengths = tokens()
    alternate = prefix.clone()
    alternate[0, 0, 4:8] = 0
    alternate[0, 0, 4] = 1
    model = exact_model()
    first = prefix_predictions(model, prefix, lengths)
    second = prefix_predictions(model, alternate, lengths)
    assert torch.equal(first['probabilities'][:, 0], second['probabilities'][:, 0])
    assert not torch.equal(first['probabilities'][:, 1], second['probabilities'][:, 1])


def test_pair_identity_no_oracle_and_no_new_parameters():
    rng = torch.random.get_rng_state().clone()
    baseline, candidate = (make_model(arm, SEED) for arm in ARMS)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert type(baseline) is type(candidate) and baseline.arm == candidate.arm == 'shared_filter'
    assert baseline.parameter_metadata()['count'] == candidate.parameter_metadata()['count'] == 1088
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    prefix, lengths = tokens()
    with pytest.raises(TypeError):
        prefix_predictions(candidate, prefix, lengths, oracle_prefix=torch.full((1, 8), .125))
    with pytest.raises(TypeError):
        candidate.encode_prefix(prefix, lengths, oracle_prefix=torch.full((1, 8), .125))


def test_prefix_likelihood_gradients_and_fixed_buffer_preservation():
    model = exact_model()
    first, first_lengths = tokens()
    second, second_lengths = tokens(found_step=3)
    prefix, lengths = torch.cat((first, second)), torch.cat((first_lengths, second_lengths))
    original = prefix.clone()
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    result = prefix_predictions(model, prefix, lengths)
    assert torch.equal(prefix, original)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])
    # Global event denominator, not the mean of per-prefix averages.
    objective = result['nll'].sum() / lengths.sum()
    objective.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert bool((parameter.grad != 0).any())
    assert model.costs.grad is None
    torch.optim.Adam(model.parameters(), lr=.003).step()
    assert torch.equal(model.costs, state_before['costs'])
    assert any(not torch.equal(value, state_before[key]) for key, value in model.named_parameters())


def test_exact_work_counts_include_found_scoring_without_conditioning():
    survivor, survivor_length = tokens()
    terminal, terminal_length = tokens(found_step=2)
    result = prefix_predictions(exact_model(), torch.cat((survivor, terminal)),
                                torch.cat((survivor_length, terminal_length)))
    work = result['work']
    assert work['prefix_nll_rows'] == work['prefix_probability_rows'] == 12
    assert work['prefix_filter_calls'] == 8 and work['prefix_filter_rows'] == 9
    assert work['reset_emission_rows'] == 2
    assert work['reset_emission_softmax_calls'] == work['prefix_operator_softmax_calls'] == 1
    assert work['operator_observed_softmax_calls'] == work['cost_readout_calls'] == 0
    assert all(type(value) is int and value >= 0 for value in work.values())


@pytest.mark.parametrize('corruption', ['padding', 'nan', 'hidden', 'early_found', 'no_terminal', 'reset_found', 'no_action'])
def test_invalid_prefix_grammar_and_padding_fail_closed(corruption):
    prefix, lengths = tokens(found_step=3)
    if corruption == 'padding':
        prefix[0, 4, 0] = 1
    elif corruption == 'nan':
        prefix[0, 4, 0] = float('nan')
    elif corruption == 'hidden':
        prefix[0, 1, 10] = 1
    elif corruption == 'early_found':
        prefix[0, 1, 4:9] = 0
        prefix[0, 1, 8] = 1
    elif corruption == 'no_terminal':
        prefix[0, 3, 8] = 0
        prefix[0, 3, 4] = 1
    elif corruption == 'reset_found':
        prefix[0, 0, 4:9] = 0
        prefix[0, 0, 8] = 1
    else:
        prefix[0, 1, :4] = 0
    with pytest.raises(ValueError):
        prefix_predictions(make_model('endpoint_only', SEED), prefix, lengths)


@pytest.mark.parametrize('name', ['reset_logits', 'observed_logits'])
def test_zero_probability_and_nonfinite_state_are_never_repaired(name):
    model = make_model('endpoint_plus_prefix', SEED)
    prefix, lengths = tokens()
    with torch.no_grad():
        getattr(model, name).reshape(-1)[0] = -10000
    with pytest.raises(ValueError, match='underflow'):
        prefix_predictions(model, prefix, lengths)
    with torch.no_grad():
        getattr(model, name).reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        prefix_predictions(model, prefix, lengths)


@pytest.mark.parametrize('split,attempts,horizon,namespace', [
    (True, 0, 2, NAMESPACE), (3, 0, 2, NAMESPACE), (0, -1, 2, NAMESPACE),
    (0, 0, 9, NAMESPACE), (0, 0, 2, True)])
def test_invalid_generation_parameters(split, attempts, horizon, namespace):
    with pytest.raises(ValueError):
        generate_attempt_split(split, attempts, horizon, seed_namespace=namespace)


def test_invalid_arm_and_seed():
    with pytest.raises(ValueError):
        make_model('unknown', SEED)
    with pytest.raises(ValueError):
        make_model('endpoint_only', True)
