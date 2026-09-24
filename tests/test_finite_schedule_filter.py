"""Independent fabricated path oracles for the public schedule filter."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from openjev.research.finite_schedule_filter import ScheduleFilter, schedule_bank


def fields():
    transition = np.full((4, 8, 8), 1 / 16, dtype=np.float64)
    for action in range(4):
        for state in range(8):
            transition[action, (state + action + 1) % 8, state] += .5
    hazard = np.array([[1 / 8 + ((s + a) % 2) / 16 for s in range(8)] for a in range(4)], dtype=np.float64)
    emission = np.full((4, 8), .125, dtype=np.float64)
    for s in range(8):
        emission[s % 4, s] = .625
    costs = np.array([[-.75 if s % 4 == a else .25 for s in range(8)] for a in range(4)], dtype=np.float64)
    return {'A': transition * (1 - hazard[:, :, None]),
            'found': np.sum(transition * hazard[:, :, None], axis=1), 'emission': emission, 'costs': costs}


def scalar_emission(base, epsilon, odor, state):
    # Rational arithmetic independent of the production vector interpolation.
    q = (Fraction(str(epsilon)) - Fraction(3, 25)) / Fraction(63, 100)
    return (1 - q) * Fraction(float(base[odor, state])) + q / 4


def paths(law, epsilons, prior, actions, labels):
    """Sum physical-state paths and schedules directly, without filter recursion."""
    assert len(labels) == len(actions) + 1
    assert labels[0] < 4 and 4 not in labels[:-1]
    terminal = labels[-1] == 4
    length = len(labels) - int(terminal)
    joint = [[Fraction(0) for _ in range(8)] for _ in prior]
    for h, weight in enumerate(prior):
        for state_path in itertools.product(range(8), repeat=length):
            mass = Fraction(float(weight)) / 8
            mass *= scalar_emission(law['emission'], epsilons[h, 0], labels[0], state_path[0])
            for t in range(1, length):
                mass *= Fraction(float(law['A'][actions[t - 1], state_path[t], state_path[t - 1]]))
                mass *= scalar_emission(law['emission'], epsilons[h, t], labels[t], state_path[t])
            if terminal:
                mass *= Fraction(float(law['found'][actions[-1], state_path[-1]]))
            joint[h][state_path[-1]] += mass
    likelihood = sum(sum(row) for row in joint)
    if terminal:
        return likelihood, [Fraction(0)] * 8, [Fraction(0)] * len(prior)
    state = [sum(row[s] for row in joint) / likelihood for s in range(8)]
    schedule = [sum(row) / likelihood for row in joint]
    return likelihood, state, schedule


def test_bank_is_exact_declared_family_prior_and_owned():
    bank, prior = schedule_bank()
    assert bank.shape == (36, 33) and bank.dtype == prior.dtype == np.float64
    assert prior.shape == (36,) and np.all(bank[0] == .12) and np.all(bank[1] == .30)
    np.testing.assert_array_equal(prior[:2], [.25, .25])
    np.testing.assert_array_equal(prior[2:], np.full(34, .25 / 17))
    assert prior.sum() == pytest.approx(1, abs=1e-14)
    for offset, switch in enumerate(range(8, 25)):
        assert np.all(bank[2 + offset, :switch] == .12) and np.all(bank[2 + offset, switch:] == .30)
        assert np.all(bank[19 + offset, :switch] == .30) and np.all(bank[19 + offset, switch:] == .12)
    short, _ = schedule_bank(24)
    np.testing.assert_array_equal(short, bank[:, :25])
    bank.fill(0); prior.fill(0)
    assert schedule_bank()[0].min() == .12 and schedule_bank()[1].sum() > .99


@pytest.mark.parametrize('hypotheses', [1, 2])
def test_filter_matches_independent_hidden_path_enumeration_and_found(hypotheses):
    law = fields()
    eps = np.array([[.12, .30, .30, .30], [.30, .30, .12, .12]], np.float64)[:hypotheses]
    prior = np.array([1.] if hypotheses == 1 else [.375, .625], np.float64)
    actions = np.array([[2, 0, 3]], np.int64)
    labels = np.array([[2, 1, 3, 4]], np.int64)
    result = ScheduleFilter(law, eps, prior).filter(actions, labels)
    previous = Fraction(1)
    for boundary in range(4):
        a, history = actions[0, :boundary].tolist(), labels[0, :boundary + 1].tolist()
        expected = []
        for event in range(5):
            if boundary == 0 and event == 4:
                expected.append(Fraction(0))
            else:
                likelihood, _, _ = paths(law, eps, prior, a, [*history[:-1], event])
                expected.append(likelihood / previous)
        likelihood, state, schedule = paths(law, eps, prior, a, history)
        np.testing.assert_allclose(result['probabilities'][0, boundary], np.array(expected, float), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result['post_states'][0, boundary], np.array(state, float), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result['schedule_posterior'][0, boundary], np.array(schedule, float), rtol=1e-12, atol=1e-14)
        cost = [sum(Fraction(float(law['costs'][d, s])) * state[s] for s in range(8)) for d in range(4)]
        np.testing.assert_allclose(result['post_costs'][0, boundary], np.array(cost, float), rtol=1e-12, atol=1e-14)
        previous = likelihood


def test_joint_schedule_state_correlation_cannot_be_replaced_by_marginals():
    law = fields()
    law['A'] = np.broadcast_to(np.eye(8) * .875, (4, 8, 8)).copy()
    law['found'].fill(.125)
    eps = np.array([[.12, .12], [.75, .75]], np.float64)
    model = ScheduleFilter(law, eps, np.array([.5, .5], np.float64))
    out = model.filter(np.array([[0]], np.int64), np.array([[0, 0]], np.int64))
    # Direct exact values: E[O_h(0,z)^2]/E[O_h(0,z)], versus prematurely
    # replacing posterior(h,z) with posterior(h)*posterior(z).
    assert out['probabilities'][0, 1, 0] == pytest.approx(77 / 256, abs=1e-14)
    factorized_value = 133 / 512
    assert out['probabilities'][0, 1, 0] - factorized_value == pytest.approx(21 / 512, abs=1e-14)
    np.testing.assert_allclose(out['probabilities'][0, 0], [.25, .25, .25, .25, 0], atol=1e-14)


def test_expected_costs_integrate_schedule_uncertainty_before_argmin():
    law = fields()
    raw = np.array([[0 if s % 4 == 0 else 2 for s in range(8)], [1] * 8, [3] * 8, [4] * 8], np.float64)
    law['costs'] = raw - raw.mean(0)
    eps = np.array([[.12], [.75]], np.float64)
    public = (np.empty((1, 0), np.int64), np.array([[0]], np.int64))
    mixture = ScheduleFilter(law, eps, np.array([.5, .5], np.float64)).filter(*public)['post_costs'][0, 0]
    private = [ScheduleFilter(law, eps[h:h+1], np.ones(1, np.float64)).filter(*public)['post_costs'][0, 0] for h in range(2)]
    np.testing.assert_allclose(mixture, (private[0] + private[1]) / 2, rtol=0, atol=1e-14)
    assert [int(x.argmin()) for x in private] == [0, 1]
    assert mixture.argmin() == 1
    assert np.bincount([x.argmin() for x in private], minlength=4).argmax() == 0


def test_prefix_causality_and_current_event_prediction_precedes_assimilation():
    eps = np.array([[.12, .12, .30, .30], [.30, .30, .12, .12]], np.float64)
    model = ScheduleFilter(fields(), eps, np.array([.5, .5], np.float64))
    a = np.array([[0, 1, 2]], np.int64)
    left = model.filter(a, np.array([[0, 1, 2, 3]], np.int64))
    right = model.filter(np.array([[0, 3, 0]], np.int64), np.array([[0, 2, 0, 1]], np.int64))
    np.testing.assert_array_equal(left['probabilities'][:, :2], right['probabilities'][:, :2])
    for name in ('post_states', 'post_costs', 'schedule_posterior'):
        np.testing.assert_array_equal(left[name][:, :1], right[name][:, :1])
    assert not np.array_equal(left['post_states'][:, 1], right['post_states'][:, 1])


def test_first_found_is_scored_then_all_joint_mass_and_cost_absorb():
    model = ScheduleFilter(fields(), np.full((1, 4), .12), np.ones(1))
    result = model.filter(np.array([[0, 1, 2]], np.int64), np.array([[0, 4, 4, 4]], np.int64))
    assert 0 < result['probabilities'][0, 1, 4] < 1
    np.testing.assert_array_equal(result['probabilities'][0, 2:], [[0, 0, 0, 0, 1]] * 2)
    for name in ('post_states', 'post_costs', 'schedule_posterior'):
        assert not result[name][:, 1:].any()
    forks = model.fork(result['post_states'], np.zeros((1, 4, 8), np.int64))
    assert not forks['costs'][:, 1:].any() and not forks['survival'][:, 1:].any()


def test_impossible_ordinary_or_found_event_fails_without_probability_floor():
    law = fields(); law['emission'].fill(0); law['emission'][0] = 1
    with pytest.raises(ValueError, match='positive observed'):
        ScheduleFilter(law, np.array([[.12]]), np.ones(1)).filter(np.empty((1, 0), np.int64), np.array([[1]], np.int64))
    law = fields(); law['A'] = np.broadcast_to(np.eye(8), (4, 8, 8)).copy(); law['found'].fill(0)
    with pytest.raises(ValueError, match='positive observed'):
        ScheduleFilter(law, np.full((1, 2), .12), np.ones(1)).filter(np.array([[0]], np.int64), np.array([[0, 4]], np.int64))


def test_forks_match_scalar_unconditional_transport_and_ignore_schedule_law():
    law = fields()
    first = ScheduleFilter(law, np.full((1, 2), .12), np.ones(1))
    second = ScheduleFilter(law, np.full((1, 2), .75), np.ones(1))
    states = np.array([[[1 / 8] * 8, [0] * 8]], np.float64)
    actions = np.array([[[0, 3, 2, 1, 0, 2, 3, 1], [3] * 8]], np.int64)
    result = first.fork(states, actions)
    current = [Fraction(1, 8)] * 8
    for h, a in enumerate(actions[0, 0]):
        current = [sum(Fraction(float(law['A'][a, j, k])) * current[k] for k in range(8)) for j in range(8)]
        cost = [sum(Fraction(float(law['costs'][d, s])) * current[s] for s in range(8)) for d in range(4)]
        np.testing.assert_allclose(result['costs'][0, 0, h], np.array(cost, float), atol=1e-14, rtol=1e-12)
        assert result['survival'][0, 0, h] == pytest.approx(float(sum(current)), abs=1e-14)
    assert not result['costs'][0, 1].any() and not result['survival'][0, 1].any()
    for key, value in second.fork(states, actions).items():
        np.testing.assert_array_equal(value, result[key])


def test_owned_inputs_outputs_determinism_chunking_and_no_global_rng():
    law = fields(); eps = np.array([[.12, .30, .30], [.30, .12, .12]], np.float64); prior = np.array([.5, .5])
    model = ScheduleFilter(law, eps, prior)
    actions = np.array([[0, 1], [1, 3], [2, 0]], np.int64)
    labels = np.array([[0, 1, 2], [3, 4, 4], [1, 2, 0]], np.int64)
    saved_actions, saved_labels = actions.copy(), labels.copy()
    before = np.random.get_state(); output = model.filter(actions, labels); after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    for value in law.values():
        value.fill(99)
    eps.fill(.75); prior.fill(0)
    repeated = model.filter(actions, labels)
    pieces = [model.filter(actions[i:i+1], labels[i:i+1]) for i in range(3)]
    for key in output:
        np.testing.assert_array_equal(output[key], repeated[key])
        np.testing.assert_allclose(output[key], np.concatenate([piece[key] for piece in pieces]), atol=1e-14, rtol=1e-12)
        assert not np.shares_memory(output[key], repeated[key])
        output[key].fill(99)
    np.testing.assert_array_equal(actions, saved_actions); np.testing.assert_array_equal(labels, saved_labels)
    empty = model.filter(actions[:0], labels[:0])
    assert empty['probabilities'].shape == (0, 3, 5) and empty['schedule_posterior'].shape == (0, 3, 2)


@pytest.mark.parametrize('steps', [True, 23, 24., -1])
def test_invalid_bank_steps(steps):
    with pytest.raises(ValueError):
        schedule_bank(steps)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'A_shape', 'A_dtype', 'A_negative', 'A_mass',
                                    'found_range', 'emission_mass', 'cost_nan', 'eps_dtype', 'eps_low',
                                    'eps_high', 'eps_nan', 'prior_mass', 'prior_negative'])
def test_constructor_rejects_malformed_laws(mutation):
    law, eps, prior = fields(), np.full((2, 2), .12), np.array([.5, .5])
    if mutation == 'missing': del law['costs']
    elif mutation == 'extra': law['private_path'] = np.zeros(1)
    elif mutation == 'A_shape': law['A'] = law['A'][:, :7]
    elif mutation == 'A_dtype': law['A'] = law['A'].astype(np.float32)
    elif mutation == 'A_negative': law['A'][0, 0, 0] = -.1
    elif mutation == 'A_mass': law['A'][0, 0, 0] += .1
    elif mutation == 'found_range': law['found'][0, 0] = 2
    elif mutation == 'emission_mass': law['emission'][0, 0] += .1
    elif mutation == 'cost_nan': law['costs'][0, 0] = np.nan
    elif mutation == 'eps_dtype': eps = eps.astype(np.float32)
    elif mutation == 'eps_low': eps[0, 0] = .119
    elif mutation == 'eps_high': eps[0, 0] = .751
    elif mutation == 'eps_nan': eps[0, 0] = np.nan
    elif mutation == 'prior_mass': prior[:] = 0
    elif mutation == 'prior_negative': prior[:] = [-.1, 1.1]
    with pytest.raises(ValueError):
        ScheduleFilter(law, eps, prior)


@pytest.mark.parametrize('actions,labels', [
    (np.array([[0.]]), np.array([[0, 1]], np.int64)),
    (np.array([[4]], np.int64), np.array([[0, 1]], np.int64)),
    (np.array([[0]], np.int64), np.array([[4, 4]], np.int64)),
    (np.array([[0]], np.int64), np.array([[0, 5]], np.int64)),
    (np.array([[0]], np.int64), np.array([[0., 1.]])),
])
def test_invalid_public_inputs(actions, labels):
    with pytest.raises(ValueError):
        ScheduleFilter(fields(), np.full((1, 2), .12), np.ones(1)).filter(actions, labels)


def test_nonabsorbing_suffix_and_invalid_fork_rejected():
    model = ScheduleFilter(fields(), np.full((1, 3), .12), np.ones(1))
    with pytest.raises(ValueError, match='absorbing'):
        model.filter(np.array([[0, 1]], np.int64), np.array([[0, 4, 0]], np.int64))
    states = np.full((1, 3, 8), .125)
    for malformed in (np.zeros((1, 3, 7), np.int64), np.full((1, 3, 8), 4, np.int64), np.zeros((1, 3, 8))):
        with pytest.raises(ValueError):
            model.fork(states, malformed)
    for value in (-.1, np.nan, .5):
        with pytest.raises(ValueError):
            model.fork(np.full_like(states, value), np.zeros((1, 3, 8), np.int64))
