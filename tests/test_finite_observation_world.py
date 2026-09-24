"""Independent exact-world and target-generation checks on fabricated data only."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch

from openjev.research import finite_observation_world as generator
from openjev.research.otto_observation_operator_model import make_model


def reference(epsilon):
    """Scalar rational definition, independent of the generator's array algebra."""
    epsilon = Fraction(str(epsilon))
    emission = [[1 - epsilon if odor == state % 4 else epsilon / 3
                 for state in range(8)] for odor in range(4)]
    b = [[[[Fraction(0) for _ in range(8)] for _ in range(8)] for _ in range(4)] for _ in range(4)]
    found = [[Fraction(0) for _ in range(8)] for _ in range(4)]
    for action in range(4):
        for source in range(8):
            destination = (source ^ 1, (source + 1) % 8,
                           ((source * 2) % 8) + source // 4, source ^ 4)[action]
            for target in range(8):
                transition = Fraction(1, 400) + (Fraction(49, 50) if target == destination else 0)
                hazard = Fraction(1 + ((target // 4) ^ (action % 2)), 200)
                found[action][source] += transition * hazard
                for odor in range(4):
                    b[action][odor][target][source] = transition * (1 - hazard) * emission[odor][target]
    a = [[[sum(b[action][odor][i][j] for odor in range(4)) for j in range(8)]
          for i in range(8)] for action in range(4)]
    costs = [[Fraction(-3, 4) if action == ((state ^ (state >> 1)) % 4) else Fraction(1, 4)
              for state in range(8)] for action in range(4)]
    return {'B': b, 'found': found, 'A': a, 'costs': costs, 'emission': emission}


def as_float(value):
    return np.asarray(value, dtype=np.float64)


def matvec(matrix, state):
    return tuple(sum(value * state[j] for j, value in enumerate(row)) for row in matrix)


def branch_oracle(ref, initial, actions):
    branches = []
    for observations in itertools.product(range(4), repeat=len(actions)):
        state = initial
        for action, odor in zip(actions, observations, strict=True):
            state = matvec(ref['B'][action][odor], state)
        mass = sum(state)
        branches.append((mass, tuple(value / mass for value in state)))
    return branches


@pytest.mark.parametrize('epsilon', (.12, .30))
def test_world_matches_rational_transition_hazard_emission_and_cost_definition(epsilon):
    actual, exact = generator.world(epsilon), reference(epsilon)
    assert set(actual) == {'B', 'found', 'A', 'costs', 'emission'}
    for name in exact:
        assert actual[name].dtype == np.float64
        np.testing.assert_allclose(actual[name], as_float(exact[name]), rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(actual['B'].sum((1, 2)) + actual['found'], 1., rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual['emission'].sum(0), 1., rtol=0, atol=1e-14)
    np.testing.assert_array_equal(actual['costs'].sum(0), np.zeros(8))
    assert (actual['B'] > 0).all() and (actual['found'] > 0).all()
    # The sensing shift changes observation partitions, not transition survival
    # or physical decision costs.
    other = generator.world(.30 if epsilon == .12 else .12)
    np.testing.assert_allclose(other['A'], actual['A'], rtol=2e-14, atol=1e-16)
    np.testing.assert_array_equal(other['found'], actual['found'])
    np.testing.assert_array_equal(other['costs'], actual['costs'])
    actual['B'].fill(0)
    assert generator.world(epsilon)['B'].min() > 0


@pytest.mark.parametrize('horizon', (1, 2, 3))
def test_h1_h2_h3_branch_enumeration_equals_unconditional_mass_and_linear_cost(horizon):
    ref, actual = reference(.12), generator.world(.12)
    initial = tuple(Fraction(i, 36) for i in range(1, 9))
    actions = (0, 1, 2)[:horizon]
    branches = branch_oracle(ref, initial, actions)
    survival = sum(weight for weight, _ in branches)
    expected_state = tuple(sum(weight * posterior[j] for weight, posterior in branches) for j in range(8))
    expected_cost = tuple(sum(weight * matvec(ref['costs'], posterior)[i] for weight, posterior in branches)
                          for i in range(4))
    state = as_float(initial)
    found = 0.
    for action in actions:
        found += actual['found'][action] @ state
        state = actual['A'][action] @ state
    np.testing.assert_allclose(state, as_float(expected_state), rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(actual['costs'] @ state, as_float(expected_cost), rtol=3e-13, atol=1e-16)
    assert state.sum() == pytest.approx(float(survival), abs=1e-14)
    assert found == pytest.approx(float(1 - survival), abs=1e-14)
    assert sum(expected_cost) == 0


def inject_reference(kind='tied'):
    ref = reference(.12)
    model = make_model(kind, seed=91, width=8, cost_scale=1.)
    b, found, a = (torch.tensor(as_float(ref[name]), dtype=torch.float64) for name in ('B', 'found', 'A'))
    with torch.no_grad():
        model.observed_logits.copy_(torch.cat((b.flatten(1, 2), found[:, None]), 1).log())
        model.readout.weight.copy_(torch.tensor(as_float(ref['costs']), dtype=torch.float64))
        if kind == 'untied':
            model.blind_logits.copy_(torch.cat((a, found[:, None]), 1).log())
    return model, ref


@pytest.mark.parametrize('kind', ('tied', 'untied'))
def test_oracle_injection_is_an_external_realizability_check_not_learned_performance(kind):
    model, ref = inject_reference(kind)
    initial = tuple(Fraction(i, 36) for i in range(1, 9))
    state = torch.tensor([as_float(initial).tolist()], dtype=torch.float64)
    expected = initial
    for action in (0, 1, 2):
        expected = matvec(ref['A'][action], expected)
        result = model.blind_step(state, torch.tensor([action], dtype=torch.int64))
        state = result['prior_state']
        np.testing.assert_allclose(state.detach().numpy()[0], as_float(expected), rtol=3e-12, atol=1e-15)
        np.testing.assert_allclose(result['cost_contrasts'].detach().numpy()[0], as_float(matvec(ref['costs'], expected)),
                                   rtol=3e-12, atol=1e-15)
    for odor in range(5):
        predicted = model.observed_step(torch.tensor([as_float(initial).tolist()], dtype=torch.float64),
            torch.tensor([3], dtype=torch.int64), torch.tensor([odor], dtype=torch.int64))
        prior = matvec(ref['A'][3], initial)
        np.testing.assert_allclose(predicted['prior_state'].detach().numpy()[0], as_float(prior), rtol=3e-12, atol=1e-15)
        if odor == 4:
            evidence = sum(ref['found'][3][j] * initial[j] for j in range(8))
            assert not bool(predicted['posterior_state'].any())
        else:
            branch = matvec(ref['B'][3][odor], initial)
            evidence = sum(branch)
            np.testing.assert_allclose(predicted['posterior_state'].detach().numpy()[0],
                as_float(tuple(value / evidence for value in branch)), rtol=3e-12, atol=1e-15)
        assert float(predicted['evidence'].detach()[0]) == pytest.approx(float(evidence), abs=1e-14)


def public_prefix_posterior(prefix, ref):
    # Public observations alone reconstruct the exact target-generation belief.
    odor = int(prefix[0, 4:9].argmax())
    state = tuple(value / 8 for value in ref['emission'][odor])
    state = tuple(value / sum(state) for value in state)
    histories = [state]
    for row in prefix[1:]:
        action, odor = int(row[:4].argmax()), int(row[4:9].argmax())
        branch = matvec(ref['B'][action][odor], state)
        state = tuple(value / sum(branch) for value in branch)
        histories.append(state)
    return histories


@pytest.mark.parametrize('split', (0, 1, 2))
def test_generated_targets_match_independent_public_history_filter_and_preobservation_timing(split):
    result = generator.generate_split(split, 5, 3, include_oracle=True, seed_namespace=919001)
    d, oracle, counts = result['data'], result['oracle'], result['counts']
    assert counts['retained'] > 0
    ref = reference(.30 if split == 2 else .12)
    for i, prefix in enumerate(d['prefix']):
        histories = public_prefix_posterior(prefix, ref)
        np.testing.assert_allclose(oracle['prefix_beliefs'][i], as_float(histories), rtol=2e-11, atol=1e-15)
        blind = observed = histories[-1]
        for h, action in enumerate(d['actions'][i]):
            blind = matvec(ref['A'][int(action)], blind)
            np.testing.assert_allclose(d['blind_costs'][i, h], as_float(matvec(ref['costs'], blind)), rtol=2e-11, atol=1e-15)
            assert d['blind_survival'][i, h] == pytest.approx(float(sum(blind)), abs=1e-13)
            if sum(observed) == 0:
                np.testing.assert_array_equal(d['observed_probabilities'][i, h], [0, 0, 0, 0, 1])
                assert not d['observed_costs'][i, h].any() and d['observations'][i, h] == 4
                continue
            branches = [matvec(ref['B'][int(action)][odor], observed) for odor in range(4)]
            prior = tuple(sum(branch[j] for branch in branches) for j in range(8))
            probabilities = [sum(branch) for branch in branches]
            probabilities.append(sum(ref['found'][int(action)][j] * observed[j] for j in range(8)))
            np.testing.assert_allclose(d['observed_costs'][i, h], as_float(matvec(ref['costs'], prior)), rtol=2e-11, atol=1e-15)
            np.testing.assert_allclose(d['observed_probabilities'][i, h], as_float(probabilities), rtol=2e-11, atol=1e-15)
            np.testing.assert_allclose(oracle['observed_prior'][i, h], as_float(prior), rtol=2e-11, atol=1e-15)
            assert d['observed_survival'][i, h] == pytest.approx(float(sum(prior)), abs=1e-13)
            odor = int(d['observations'][i, h])
            observed = (Fraction(0),) * 8 if odor == 4 else tuple(v / probabilities[odor] for v in branches[odor])
            np.testing.assert_allclose(oracle['observed_posterior'][i, h], as_float(observed), rtol=2e-11, atol=1e-15)


def test_public_schema_excludes_oracle_and_retains_only_declared_onehot_features():
    result = generator.generate_split(0, 12, 8, seed_namespace=919001)
    d, n = result['data'], result['counts']['retained']
    assert result['oracle'] is None
    assert set(d) == {'prefix', 'lengths', 'actions', 'observations', 'blind_costs', 'blind_survival',
                      'observed_costs', 'observed_survival', 'observed_probabilities', 'case_ids'}
    assert d['prefix'].dtype == np.float32 and d['prefix'].shape == (n, 9, 31)
    assert d['lengths'].dtype == np.int64 and (d['lengths'] == 9).all()
    assert d['actions'].dtype == d['observations'].dtype == np.int64
    assert ((d['prefix'] == 0) | (d['prefix'] == 1)).all()
    assert not d['prefix'][:, :, 10:].any() and not d['prefix'][:, 0, :4].any()
    np.testing.assert_array_equal(d['prefix'][:, 1:, :4].sum(-1), np.ones((n, 8)))
    np.testing.assert_array_equal(d['prefix'][:, :, 4:9].sum(-1), np.ones((n, 9)))
    assert not d['prefix'][:, :, 8].any()  # Found prefixes never enter the dataset.
    np.testing.assert_array_equal(d['prefix'][:, :, 9], np.tile([1] + [0] * 8, (n, 1)))
    assert np.all(np.diff(d['blind_survival'], axis=1) <= 1e-14)
    assert np.max(np.abs(d['blind_costs'].sum(-1))) < 1e-13
    assert np.max(np.abs(d['observed_costs'].sum(-1))) < 1e-13


def test_seeded_generator_is_reproducible_without_mutating_global_rng_or_previous_outputs():
    before = np.random.get_state()
    a = generator.generate_split(1, 7, 2, include_oracle=True, seed_namespace=919001)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    b = generator.generate_split(1, 7, 2, include_oracle=True, seed_namespace=919001)
    assert a['counts'] == b['counts']
    for group in ('data', 'oracle'):
        for key in a[group]:
            np.testing.assert_array_equal(a[group][key], b[group][key])
            assert not np.shares_memory(a[group][key], b[group][key])
    original = b['data']['prefix'].copy()
    a['data']['prefix'].fill(999)
    np.testing.assert_array_equal(b['data']['prefix'], original)
    c = generator.generate_split(0, 7, 2, seed_namespace=919001)
    assert not set(c['data']['case_ids']) & set(b['data']['case_ids'])


def test_extending_requested_horizon_leaves_prefix_and_earlier_blind_targets_unchanged():
    short, long = (generator.generate_split(1, 7, h, seed_namespace=919001)['data'] for h in (2, 8))
    for name in ('prefix', 'lengths', 'case_ids'):
        np.testing.assert_array_equal(short[name], long[name])
    np.testing.assert_array_equal(short['actions'], long['actions'][:, :2])
    np.testing.assert_allclose(short['blind_costs'], long['blind_costs'][:, :2], rtol=0, atol=0)
    np.testing.assert_allclose(short['blind_survival'], long['blind_survival'][:, :2], rtol=0, atol=0)
    np.testing.assert_allclose(short['observed_costs'][:, 0], long['observed_costs'][:, 0], rtol=0, atol=0)
    # Drawing a longer committed action block changes subsequent RNG state.
    # Observed labels or second-step conditional targets need not match.


def test_counts_and_absorbing_suffixes_without_replacement():
    result = generator.generate_split(2, 64, 8, seed_namespace=919001)
    d, count = result['data'], result['counts']
    n = len(d['case_ids'])
    assert n + count['excluded_found'] == count['attempts'] == 64
    assert sum(count['prefix_found_by_step']) == count['excluded_found']
    expected_prefix_events = n * 8 + sum((step + 1) * value for step, value in enumerate(count['prefix_found_by_step']))
    assert count['prefix_event_draws'] == expected_prefix_events
    assert count['initial_odor_draws'] == 64 and count['prefix_action_draws'] == 64 * 8
    assert count['forecast_action_draws'] == n * 8
    found = d['observations'] == 4
    np.testing.assert_array_equal(found, np.maximum.accumulate(found, axis=1))
    assert count['forecast_found_cases'] == int(found.any(1).sum())
    sampled = sum(int(np.flatnonzero(row)[0]) + 1 if row.any() else 8 for row in found)
    assert count['forecast_event_draws'] == sampled
    for i in range(n):
        for h in range(1, 8):
            if found[i, h - 1]:
                assert not d['observed_costs'][i, h].any() and d['observed_survival'][i, h] == 0
                np.testing.assert_array_equal(d['observed_probabilities'][i, h], [0, 0, 0, 0, 1])
    attempts = [int(case.split('case')[1]) for case in d['case_ids']]
    assert attempts == sorted(set(attempts)) and all(0 <= index < 64 for index in attempts)


def test_empty_dataset_retains_exact_shapes_and_optional_oracle():
    result = generator.generate_split(0, 0, 2, include_oracle=True, seed_namespace=919001)
    assert result['counts']['attempts'] == result['counts']['retained'] == 0
    assert result['data']['prefix'].shape == (0, 9, 31)
    assert result['data']['observed_probabilities'].shape == (0, 2, 5)
    assert result['oracle']['prefix_beliefs'].shape == (0, 9, 8)
    assert result['oracle']['observed_posterior'].shape == (0, 2, 8)


@pytest.mark.parametrize('epsilon', (0, 1, -.1, float('nan'), float('inf'), True))
def test_invalid_world_parameters_fail(epsilon):
    with pytest.raises(ValueError):
        generator.world(epsilon)


@pytest.mark.parametrize('args', ((True, 1, 2), (3, 1, 2), (0, -1, 2), (0, True, 2),
                                 (0, 1, 0), (0, 1, 9), (0, 1, True)))
def test_invalid_split_parameters_fail(args):
    with pytest.raises(ValueError):
        generator.generate_split(*args, seed_namespace=919001)
    with pytest.raises(ValueError):
        generator.generate_split(0, 1, 2, include_oracle=1, seed_namespace=919001)


def test_forced_prefix_found_has_no_replacement_or_forecast_draws(monkeypatch):
    events = iter((0, 4))
    calls = []

    def event(_rng, probabilities):
        value = next(events)
        assert probabilities[value] > 0
        calls.append(value)
        return value

    monkeypatch.setattr(generator, '_sample', event)
    result = generator.generate_split(0, 1, 8, seed_namespace=919001)
    assert calls == [0, 4]
    assert result['counts']['retained'] == 0 and result['counts']['excluded_found'] == 1
    assert result['counts']['prefix_found_by_step'] == [1] + [0] * 7
    assert result['counts']['forecast_action_draws'] == result['counts']['forecast_event_draws'] == 0
    assert result['data']['prefix'].shape == (0, 9, 31)


def test_forced_forecast_found_retains_own_prior_then_zero_observed_suffix(monkeypatch):
    # Initial odor, eight public nonterminal odors, then first forecast found.
    events = iter([0] * 9 + [4])

    def event(_rng, probabilities):
        value = next(events)
        assert probabilities[value] > 0
        return value

    monkeypatch.setattr(generator, '_sample', event)
    result = generator.generate_split(0, 1, 8, include_oracle=True, seed_namespace=919001)
    d = result['data']
    assert result['counts']['retained'] == 1 and result['counts']['forecast_event_draws'] == 1
    assert (d['observations'] == 4).all()
    ref = reference(.12)
    state = public_prefix_posterior(d['prefix'][0], ref)[-1]
    prior = matvec(ref['A'][int(d['actions'][0, 0])], state)
    np.testing.assert_allclose(d['observed_costs'][0, 0], as_float(matvec(ref['costs'], prior)), rtol=2e-11, atol=1e-15)
    assert d['observed_survival'][0, 0] > 0 and not d['observed_survival'][0, 1:].any()
    assert not d['observed_costs'][0, 1:].any() and not result['oracle']['observed_posterior'].any()
    np.testing.assert_array_equal(d['observed_probabilities'][0, 1:, 4], np.ones(7))
    assert (d['blind_survival'] > 0).all()  # Realized found cannot enter blind targets.


@pytest.mark.parametrize('namespace', (True, -1, 2**32, '919001'))
def test_invalid_namespace_fails_before_generation(namespace):
    with pytest.raises(ValueError):
        generator.generate_split(0, 0, 1, seed_namespace=namespace)
