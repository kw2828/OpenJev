"""Fabricated oracle-regret and causal terminal-policy qualification tests."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from openjev.research import otto_action_latent_metrics as hard
from openjev.research import otto_belief_distillation_metrics as metrics


def inputs(n=2):
    p = np.broadcast_to([.4, .2, .1, .2, .1], (n, 8, 5)).copy()
    return {'outcomes': np.zeros((n, 8), np.int64), 'outcome_predictions': p,
            'predicted_costs': np.zeros((n, 8, 4)),
            'raw_costs': np.broadcast_to([1., 0., 2., 3.], (n, 8, 4)).copy(),
            'legal': np.ones((n, 8, 4), np.bool_),
            'oracle_probabilities': np.broadcast_to([.5, .25, .25, 0, 0], (n, 8, 5)).copy(),
            'case_ids': [f'case{i}' for i in range(n)], 'regimes': ['lambda3'] * n,
            'family': 'recurrent_soft', 'fit_seed': 1, 'condition': 'gap', 'prediction_kind': 'probabilities'}


def test_known_kl_brier_excess_and_expected_scores_retain_hard_scores():
    value = inputs()
    result = metrics.score(**value)
    original = hard.score(**{k: v for k, v in value.items() if k != 'oracle_probabilities'})
    assert {k: v for k, v in result.items() if k != 'oracle'} == original
    leaf = result['oracle']['groups']['long']['all_rows']['overall']
    p, q = value['outcome_predictions'][0, 0], value['oracle_probabilities'][0, 0]
    expected_kl = .5 * math.log(.5 / .4) + .25 * math.log(.25 / .2) + .25 * math.log(.25 / .1)
    assert leaf['case_weighted_kl'] == pytest.approx(expected_kl)
    assert leaf['case_weighted_brier_excess'] == pytest.approx(sum((p - q) ** 2))
    assert leaf['case_weighted_expected_log_score'] - leaf['case_weighted_oracle_entropy'] == pytest.approx(expected_kl)
    assert leaf['case_weighted_expected_brier'] - leaf['case_weighted_oracle_brier'] == pytest.approx(sum((p - q) ** 2))
    assert leaf['case_weighted_sampled_log_score'] == pytest.approx(-math.log(.4))
    assert result['oracle']['resolved_rows'] == 0 and result['oracle']['unresolved_rows'] == 16
    assert result['oracle']['groups']['long']['all_rows'] == result['oracle']['groups']['long']['unresolved_rows']


def test_oracle_prediction_has_zero_regret_including_zero_masses():
    value = inputs(1)
    value['outcome_predictions'] = value['oracle_probabilities'].copy()
    report = metrics.score(**value)
    leaf = report['oracle']['groups']['all']['all_rows']['overall']
    assert leaf['case_weighted_kl'] == leaf['case_weighted_brier_excess'] == 0
    assert leaf['case_weighted_expected_log_score'] == leaf['case_weighted_oracle_entropy']


def test_normal_shortcut_starts_after_first_found_not_at_event():
    value = inputs(1)
    value['condition'] = 'normal'
    value['outcomes'][0, 2:] = 4
    value['legal'][0, 2:] = False
    value['oracle_probabilities'][0, 2] = [.5, 0, 0, 0, .5]
    value['oracle_probabilities'][0, 3:] = [0, 0, 0, 0, 1]
    value['outcome_predictions'][0, 3:] = [0, 0, 0, 0, 1]
    report = metrics.score(**value)
    oracle = report['oracle']
    assert oracle['resolved_rows'] == 5 and oracle['unresolved_rows'] == 3
    event = oracle['per_horizon']['3']['unresolved_rows']['overall']
    assert event['scored_rows'] == 1 and event['case_weighted_sampled_log_score'] == pytest.approx(-math.log(.1))
    resolved = oracle['groups']['long']['unresolved_rows']['overall']
    assert resolved['supported_cases'] == resolved['scored_rows'] == 0
    assert resolved['unsupported_case_ids'] == ['case0'] and resolved['case_weighted_kl'] is None
    all_rows = oracle['groups']['long']['all_rows']['overall']
    assert all_rows['scored_rows'] == 4 and all_rows['case_weighted_kl'] == 0
    assert all_rows['case_weighted_sampled_log_score'] == 0
    # The gap counterpart receives no terminal feedback and retains all rows.
    value['condition'] = 'gap'
    value['oracle_probabilities'][:] = [.4, .1, .1, .1, .3]
    value['outcome_predictions'][:] = [.2] * 5
    gap = metrics.score(**value)
    assert gap['oracle']['resolved_rows'] == 0 and gap['oracle']['unresolved_rows'] == 8


def test_case_weights_and_unsupported_cases_are_explicit():
    value = inputs(3)
    value['case_ids'] = ['a', 'a', 'b']
    value['oracle_probabilities'][:] = [1, 0, 0, 0, 0]
    value['outcome_predictions'][:2] = [.5, .5, 0, 0, 0]
    value['outcome_predictions'][2] = [.25, .75, 0, 0, 0]
    leaf = metrics.score(**value)['oracle']['groups']['all']['all_rows']['overall']
    assert leaf['declared_cases'] == 2 and leaf['blocks'] == 3
    assert leaf['case_weighted_kl'] == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert leaf['case_weighted_kl'] != pytest.approx((2 * math.log(2) + math.log(4)) / 3)


def test_stable_logits_keep_expected_log_loss_when_probabilities_underflow():
    value = inputs(1)
    value['prediction_kind'] = 'logits'
    value['outcome_predictions'][:] = [0, 1000, -1000, 0, 0]
    value['oracle_probabilities'][:] = [1, 0, 0, 0, 0]
    leaf = metrics.score(**value)['oracle']['groups']['all']['all_rows']['overall']
    assert leaf['case_weighted_kl'] == leaf['case_weighted_expected_log_score'] == 1000


@pytest.mark.parametrize('change', ['observed_impossible', 'prediction_zero', 'oracle_nan', 'oracle_negative',
                                   'oracle_sum', 'shape', 'unresolved_repair', 'known_prediction', 'known_oracle'])
def test_invalid_or_impossible_oracle_contract_is_not_clipped(change):
    value = inputs(1)
    if change == 'observed_impossible':
        value['oracle_probabilities'][:] = [0, 1, 0, 0, 0]
    elif change == 'prediction_zero':
        value['outcome_predictions'][:] = [1, 0, 0, 0, 0]
    elif change == 'oracle_nan':
        value['oracle_probabilities'][0, 0, 4] = np.nan
    elif change == 'oracle_negative':
        value['oracle_probabilities'][0, 0] = [1.1, -.1, 0, 0, 0]
    elif change == 'oracle_sum':
        value['oracle_probabilities'][:] = [.3] * 5
    elif change == 'shape':
        value['oracle_probabilities'] = value['oracle_probabilities'][:, :7]
    elif change == 'unresolved_repair':
        value['outcome_predictions'][0, 0] = [0, 0, 0, 0, 1]
    else:
        value['condition'] = 'normal'
        value['outcomes'][:] = 4
        value['legal'][:] = False
        value['oracle_probabilities'][:] = [0, 0, 0, 0, 1]
        value['outcome_predictions'][:] = [0, 0, 0, 0, 1]
        field = 'outcome_predictions' if change == 'known_prediction' else 'oracle_probabilities'
        value[field][0, 1] = [.1, 0, 0, 0, .9]
    with pytest.raises(ValueError):
        metrics.score(**value)


def test_no_input_or_global_rng_mutation():
    value = inputs()
    arrays = {k: v.copy() for k, v in value.items() if isinstance(v, np.ndarray)}
    for key in arrays:
        value[key].flags.writeable = False
    state = np.random.get_state()
    metrics.score(**value)
    for key, expected in arrays.items():
        np.testing.assert_array_equal(value[key], expected)
    actual = np.random.get_state()
    assert actual[0] == state[0] and actual[2:] == state[2:]
    np.testing.assert_array_equal(actual[1], state[1])


def test_unchanged_hard_gate_never_selects_from_oracle_regret():
    reports = []
    families = ('recurrent_soft', 'recurrent_sampled', 'action_blind_soft', 'direct_soft')
    for condition in ('gap', 'normal'):
        for family in families:
            for seed in (1, 2, 3):
                value = inputs()
                value.update(family=family, condition=condition, fit_seed=seed, regimes=['lambda3', 'lambda4'])
                reports.append(metrics.score(**value))
    options = {'candidate': families[0], 'controls': families[1:], 'fit_seeds': (1, 2, 3),
        'regimes': ('lambda3', 'lambda4'), 'thresholds': {'long_log_relative_gain': .01,
            'long_gap_relative_gain': .05, 'normal_log_relative_tolerance': .01,
            'normal_gap_relative_tolerance': .01, 'minimum_supported_cases': 1}}
    old = hard.evaluate_reports(reports, **options)
    result = metrics.evaluate_reports(reports, **options)
    assert result['cells'] == old['cells'] and result['passed'] is old['passed'] is False
    assert result['oracle_diagnostics_used_for_selection'] is False
    changed = copy.deepcopy(reports)
    changed[0]['oracle']['groups']['long']['all_rows']['overall']['case_weighted_kl'] = 1e8
    assert metrics.evaluate_reports(changed, **options) == result
    changed[0]['oracle']['target_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='share paired oracle'):
        metrics.evaluate_reports(changed, **options)


def test_opposite_action_oracle_js_tv_keep_zero_effects_and_case_weights():
    p = np.zeros((3, 4, 5))
    q = np.zeros_like(p)
    p[:, :, 0] = 1
    q[:2, :, 1], q[2, :, 0] = 1, 1
    actions = np.tile([0, 1, 2, 3], (3, 1)).astype(np.int64)
    args = {'case_ids': ['a', 'a', 'b'], 'regimes': ['lambda3'] * 3,
            'actions': actions, 'alternate_actions': actions ^ 1}
    report = metrics.action_sensitivity(p, q, **args)
    leaf = report['overall']['overall']
    assert leaf['case_weighted_jensen_shannon'] == pytest.approx(math.log(2) / 2)
    assert leaf['case_weighted_total_variation'] == .5
    assert leaf['case_weighted_oracle_signal'] == 1
    assert leaf['declared_cases'] == 2 and leaf['scored_rows'] == 12
    assert report['case_filtering'] is False
    swapped = metrics.action_sensitivity(q, p, **args)
    assert swapped == report
    same = metrics.action_sensitivity(p, p, **args)
    assert same['overall']['overall']['case_weighted_jensen_shannon'] == 0
    assert same['overall']['overall']['case_weighted_total_variation'] == 0
    args['alternate_actions'] = actions[:, ::-1].copy()
    with pytest.raises(ValueError, match='fixed opposite-action'):
        metrics.action_sensitivity(p, q, **args)


def test_js_handles_positive_subnormal_mass_without_epsilon():
    p = np.zeros((1, 1, 5))
    p[0, 0] = [1, np.nextafter(0., 1.), 0, 0, 0]
    q = np.zeros_like(p)
    q[0, 0, 0] = 1
    actions = np.zeros((1, 1), np.int64)
    result = metrics.action_sensitivity(p, q, case_ids=['a'], regimes=['lambda3'],
                                        actions=actions, alternate_actions=actions ^ 1)
    assert math.isfinite(result['overall']['overall']['case_weighted_jensen_shannon'])


def test_signed_model_effect_error_is_not_only_magnitude_and_keeps_zero_signal():
    p = np.zeros((2, 1, 5))
    p[:, :, 0] = 1
    alt = p.copy()
    alt[0, 0] = [0, 1, 0, 0, 0]
    perfect = metrics.model_effect_error(p, alt, p, alt)
    np.testing.assert_array_equal(perfect['oracle_signal'], [[2], [0]])
    np.testing.assert_array_equal(perfect['model_effect_error'], 0)
    insensitive = metrics.model_effect_error(p, alt, p, p)
    np.testing.assert_array_equal(insensitive['model_effect_error'], [[2], [0]])
    wrong_direction = metrics.model_effect_error(p, alt, alt, p)
    np.testing.assert_array_equal(wrong_direction['model_effect_error'], [[8], [0]])
    model_alt = alt.copy()
    model_alt[1, 0] = [0, 1, 0, 0, 0]
    actions = np.zeros((2, 1), np.int64)
    report = metrics.action_sensitivity(p, alt, case_ids=['a', 'b'], regimes=['lambda3'] * 2,
        actions=actions, alternate_actions=actions ^ 1, predicted_original=p, predicted_alternate=model_alt)
    leaf = report['overall']['overall']
    assert leaf['case_weighted_oracle_signal'] == leaf['case_weighted_model_effect_error'] == 1
    assert leaf['scored_rows'] == leaf['declared_cases'] == 2
    assert report['model_effect_included'] is True and report['case_filtering'] is False
    with pytest.raises(ValueError, match='both paired model'):
        metrics.action_sensitivity(p, alt, case_ids=['a', 'b'], regimes=['lambda3'] * 2,
            actions=actions, alternate_actions=actions ^ 1, predicted_original=p)
