"""Independent fabricated arithmetic and classification checks; no study data."""
from __future__ import annotations

import copy
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
a = importlib.import_module('audit_schedule_headroom')


def fixture():
    n, length = 4, 33
    labels = np.zeros((n, length), np.int64)
    labels[0, 8:] = 4; labels[2, 9:] = 4; labels[3, 10:] = 4
    lengths = np.array([9, 33, 10, 11], np.int64)
    probability = np.full((n, length, 5), .2)
    probability[:, 0] = (.25, .25, .25, .25, 0)
    truth = np.zeros((n, length, 8, 4))
    post = np.zeros((n, length, 4))
    for case in range(n):
        for time in range(length):
            if labels[case, time] != 4:
                truth[case, time, 3] = np.array([-.75, .25, .25, .25])*.1
                truth[case, time, 7] = np.array([-.75, .25, .25, .25])*.3
                post[case, time] = np.array([-.75, .25, .25, .25])*.2
            if time >= lengths[case]:
                probability[case, time] = (0, 0, 0, 0, 1)
    predicted = np.zeros((n, length, 2, 4)); predicted[..., 1] = -1
    predicted_post = np.zeros_like(post); predicted_post[..., 1] = -1
    data = {'public/observations': labels, 'public/lengths': lengths,
            'targets/fork_costs': truth, 'targets/post_costs': post,
            'audit/family': np.arange(4, dtype=np.int64)}
    predictions = {'probabilities': probability, 'fork_costs': predicted, 'post_costs': predicted_post}
    return data, predictions


def test_fixed50_denominator_retains_early_found_and_family_zero():
    data, predictions = fixture()
    row = a.metrics(data, predictions)
    assert row['primary_regret'] == pytest.approx(.2*28/100)
    assert row['h4_regret'] == pytest.approx(.1*28/100)
    assert row['h8_regret'] == pytest.approx(.3*28/100)
    assert row['post_regret'] == pytest.approx(.2*28/100)
    assert row['per_episode_regret'] == pytest.approx([0, .2, .2/25, .4/25])
    assert row['family_regret'][0] == {'family': 0, 'episodes': 1, 'primary_regret': 0}
    assert row['event_count'] == 63 and row['found_episodes'] == 3
    assert row['event_nll'] == pytest.approx((4*math.log(4)+59*math.log(5))/63)
    # Reweighting only supported boundaries would give .2, not the registered .056.
    assert row['primary_regret'] != pytest.approx(.2)


def test_first_found_scored_padding_unscored_and_argmin_tie_lowest():
    data, predictions = fixture()
    before = a.metrics(data, predictions)
    predictions['probabilities'][0, 9:] = .2
    assert a.metrics(data, predictions) == before
    predictions['probabilities'][0, 8, 4] = .1
    changed = a.metrics(data, predictions)
    assert changed['event_nll']-before['event_nll'] == pytest.approx(math.log(2)/63)
    predictions['fork_costs'].fill(0); predictions['post_costs'].fill(0)
    tied = a.metrics(data, predictions)
    assert tied['primary_regret'] == tied['post_regret'] == 0


def rows(learned=.5, true=.4, true_static=.9):
    result = []
    for cohort in range(5):
        for arm in a.ARMS:
            value = {'learned_exact': learned, 'true_exact': true, 'true_static2': true_static}.get(arm, 1.)
            result.append({'cohort': cohort, 'arm': arm, 'primary_regret': value,
                'family_regret': [{'family': f, 'episodes': 128, 'primary_regret': value} for f in range(4)]})
    return result


@pytest.mark.parametrize(('learned', 'true', 'true_static', 'classification'), (
    (.5, .4, .9, 'LEARNED_MODEL_HEADROOM'),
    (1., .4, .9, 'MODEL_MISMATCH_HEADROOM'),
    (1., .4, .4, 'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY'),
    (1., 1., 1., 'NO_REGISTERED_HEADROOM'),
))
def test_exact_four_way_classification(learned, true, true_static, classification):
    result = a.classify(rows(learned, true, true_static))
    assert result['classification'] == classification
    assert set(result['candidates']) == {'learned_exact', 'true_exact'}
    assert all(len(x['conditions']) == 15 for x in result['candidates'].values())
    assert len(result['true_history_conditions']) == 2
    assert set(result['means']) == set(a.ARMS)


def test_original_base_guard_alone_blocks_learned_admission():
    values = rows()
    for row in values:
        if row['arm'] == 'learned_exact':
            row['family_regret'][0]['primary_regret'] = 1.06
    result = a.classify(values)
    conditions = result['candidates']['learned_exact']['conditions']
    assert sum(conditions.values()) == 14
    assert not conditions['base/unchanged/noninferiority']
    assert result['classification'] == 'MODEL_MISMATCH_HEADROOM'


def test_strict_cohort_wins_and_positive_control_not_rescued_by_large_mean_gain():
    values = rows()
    for row in values:
        if row['arm'] == 'learned_exact':
            row['primary_regret'] = .01 if row['cohort'] < 3 else 1.
    conditions = a.classify(values)['candidates']['learned_exact']['conditions']
    assert conditions['unchanged/mean_gain10pct']
    assert not conditions['unchanged/paired_wins4of5']
    for row in values:
        if row['arm'] == 'unchanged':
            row['primary_regret'] = 0.
    assert not a.classify(values)['candidates']['learned_exact']['conditions']['unchanged/mean_gain10pct']


@pytest.mark.parametrize('bad', ('missing', 'duplicate', 'wrong_arm', 'missing_base'))
def test_classification_roster_and_family_support_fail_closed(bad):
    values = rows()
    if bad == 'missing':
        values.pop()
    elif bad == 'duplicate':
        values[-1] = copy.deepcopy(values[0])
    elif bad == 'wrong_arm':
        values[-1]['arm'] = 'winner'
    else:
        values[0]['family_regret'][0]['episodes'] = 0
    with pytest.raises(ValueError):
        a.classify(values)


def fabricated_fields():
    return {'transition': np.tile(np.eye(8), (4, 1, 1)), 'hazard': np.full((4, 8), .125),
            'emission': np.tile(np.array([.625, .125, .125, .125])[:, None], (1, 8)),
            'costs': np.tile(np.array([-.375, -.125, .125, .375])[:, None], (1, 8))}


@pytest.mark.parametrize('kind', ('exact', 'static2'))
def test_exact_posterior_against_scalar_hypothesis_likelihood_products(kind):
    fields = fabricated_fields()
    labels = np.array([[0]+[1]*31+[4]], np.int64)
    actions = np.zeros((1, 32), np.int64)
    forks = np.zeros((1, 33, 8), np.int64)
    actual = a.exact_predictions(fields, {'actions': actions, 'observations': labels}, forks, kind)
    descriptions = [(0, -1), (1, -1)]
    if kind == 'exact':
        descriptions += [(family, switch) for family in (2, 3) for switch in range(8, 25)]
    weights = [(.5 if kind == 'static2' else .25) if f < 2 else .25/17 for f, _ in descriptions]
    for time in range(32):
        for index, (family, switch) in enumerate(descriptions):
            low = family == 0 or (family == 2 and time < switch) or (family == 3 and time >= switch)
            q = 0. if low else 2/7
            odor = int(labels[0, time])
            likelihood = (1-q)*([.625, .125, .125, .125][odor])+q*.25
            weights[index] *= likelihood*(1 if time == 0 else .875)
        total = math.fsum(weights)
        expected = [value/total for value in weights]
        np.testing.assert_allclose(actual['schedule_posterior'][0, time], expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual['post_states'][0, time], np.full(8, .125), atol=1e-14)
        np.testing.assert_allclose(actual['fork_survival'][0, time], [.875**4, .875**8], atol=1e-14)
    assert actual['probabilities'][0, 32, 4] == pytest.approx(.125)
    assert not actual['post_states'][0, 32].any()
    assert not actual['schedule_posterior'][0, 32].any()
    assert not actual['fork_costs'][0, 32].any()


def test_mixture_prior_order_and_future_labels_cannot_change_earlier_laws():
    bank, prior = a.schedule_bank(33, 'exact')
    assert bank.shape == (36, 33) and prior.sum() == pytest.approx(1)
    np.testing.assert_array_equal(bank[:2], np.array([[.12]*33, [.30]*33]))
    assert bank[2, 7] == .12 and bank[2, 8] == .30
    assert bank[19, 7] == .30 and bank[19, 8] == .12
    fields = fabricated_fields()
    actions = np.zeros((1, 32), np.int64); labels = np.zeros((1, 33), np.int64)
    forks = np.zeros((1, 33, 8), np.int64)
    original = a.exact_predictions(fields, {'actions': actions, 'observations': labels}, forks)
    labels[:, 12:] = 2
    changed = a.exact_predictions(fields, {'actions': actions, 'observations': labels}, forks)
    np.testing.assert_array_equal(original['probabilities'][:, :13], changed['probabilities'][:, :13])
    np.testing.assert_array_equal(original['post_states'][:, :12], changed['post_states'][:, :12])


def test_private_reference_reset_and_found_hand_values():
    data = {'public/actions': np.zeros((1, 1), np.int64), 'public/observations': np.array([[0, 4]], np.int64),
            'audit/epsilons': np.array([[.12, .30]]), 'targets/fork_actions': np.zeros((1, 2, 8), np.int64)}
    result = a.reconstruct_targets(data)
    np.testing.assert_allclose(result['probabilities'][0, 0], [.25, .25, .25, .25, 0], atol=1e-15)
    np.testing.assert_allclose(result['post_states'][0, 0], [.44, .02, .02, .02, .44, .02, .02, .02], atol=1e-15)
    assert result['probabilities'][0, 1, 4] == pytest.approx(.0075)
    assert not result['post_states'][0, 1].any() and not result['fork_costs'][0, 1].any()


def test_prediction_validation_rejects_fork_and_public_mixture_corruption():
    fields = fabricated_fields()
    public = {'actions': np.zeros((1, 32), np.int64), 'observations': np.zeros((1, 33), np.int64)}
    forks = np.zeros((1, 33, 8), np.int64)
    data = {'public/actions': public['actions'], 'public/observations': public['observations'],
            'public/lengths': np.array([33], np.int64), 'targets/fork_actions': forks}
    prediction = a.exact_predictions(fields, public, forks)
    a.validate_predictions(data, prediction, fields, 'learned_exact')
    for name in ('fork_costs', 'schedule_posterior', 'probabilities'):
        bad = {key: value.copy() for key, value in prediction.items()}
        bad[name].flat[0] += .01
        with pytest.raises(ValueError):
            a.validate_predictions(data, bad, fields, 'learned_exact')


def test_copied_state_schema_and_frozen_field_bytes():
    fields = fabricated_fields()
    state = {name: value.copy() for name, value in fields.items()}
    state['global_logit'] = np.array(.5)
    a.verify_state(state, fields, 'global')
    state['transition'][0, 0, 0] += .01
    with pytest.raises(ValueError):
        a.verify_state(state, fields, 'global')
