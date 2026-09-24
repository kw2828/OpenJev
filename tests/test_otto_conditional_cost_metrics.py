"""Fabricated fixed-horizon identities, support and numerical contract tests."""
from __future__ import annotations

import copy
import math

import pytest

from openjev.research import otto_conditional_cost_metrics as metrics


def branch(weight, costs, legal=(True, True, True, True)):
    return {'weight': weight, 'costs': list(costs), 'legal': list(legal)}


def example():
    return [{'case_id': 'a', 'branches': [branch(.2, [0., 4., 8., 12.]),
                                         branch(.3, [6., 2., 8., 12.])]}]


def test_known_conditional_identity_and_population_variances():
    inputs = example()
    before = copy.deepcopy(inputs)
    result = metrics.decompose_cases(inputs, {'a': 0}, horizon=6)
    row = result['cases'][0]
    assert row['support_mass'] == .5 and row['defined'] is True
    assert row['conditional_mean_costs'] == pytest.approx([3.6, 2.8, 8., 12.])
    assert row['conditional_cost_variances'] == pytest.approx([8.64, .96, 0., 0.])
    assert row['blind_optimal_action'] == 1
    assert row['conditional_oracle_cost'] == pytest.approx(1.2)
    assert row['information_advantage'] == pytest.approx(1.6)
    assert row['approximation_regret'] == pytest.approx(.8)
    assert row['total_regret'] == pytest.approx(2.4)
    assert row['conditional_oracle_cost_variance'] == pytest.approx(.96)
    assert row['conditional_chosen_regret_variance'] == pytest.approx(3.84)
    assert row['total_regret'] == pytest.approx(row['information_advantage'] + row['approximation_regret'])
    assert row['identity']['within_roundoff'] is True
    assert result['horizon'] == 6 and result['admits_execution'] is False
    assert inputs == before


def test_equal_case_aggregation_is_not_survival_mass_weighted():
    cases = example() + [{'case_id': 'b', 'branches': [branch(.1, [10., 0., 5., 6.])]},
                        {'case_id': 'empty', 'branches': []}]
    result = metrics.decompose_cases(cases, {'a': 0, 'b': 0, 'empty': 2}, horizon=8)
    aggregate = result['aggregate']
    assert aggregate['declared_cases'] == 3 and aggregate['supported_cases'] == 2
    assert aggregate['unsupported_case_ids'] == ['empty']
    assert aggregate['support_mass_sum'] == pytest.approx(.6)
    assert aggregate['mean_support_mass'] == pytest.approx(.2)
    assert aggregate['case_weighted_total_regret'] == pytest.approx(6.2)
    assert aggregate['case_weighted_information_advantage'] == pytest.approx(.8)
    assert aggregate['case_weighted_approximation_regret'] == pytest.approx(5.4)
    assert aggregate['case_weighted_total_regret'] != pytest.approx((.5 * 2.4 + .1 * 10.) / .6)


def test_empty_and_zero_weight_cases_remain_undefined_without_resampling():
    cases = [{'case_id': 'empty', 'branches': []},
             {'case_id': 'zero', 'branches': [branch(0., [1., 2., 3., 4.])]}]
    result = metrics.decompose_cases(cases, {'empty': 3, 'zero': 2}, horizon=1)
    assert len(result['cases']) == 2
    for row in result['cases']:
        assert row['defined'] is False and row['support_mass'] == 0.
        assert row['total_regret'] is row['conditional_mean_costs'] is row['identity'] is None
        assert row['blind_optimal_action'] is None
    assert result['cases'][0]['legal'] is None
    assert result['cases'][1]['legal'] == [True] * 4
    assert result['aggregate']['supported_cases'] == 0
    assert result['aggregate']['case_weighted_total_regret'] is None
    assert result['aggregate']['unsupported_case_ids'] == ['empty', 'zero']


def compare_quantities(left, right):
    for key in (*metrics.SCALARS, *metrics.VECTORS):
        assert left[key] == pytest.approx(right[key])
    assert left['blind_optimal_action'] == right['blind_optimal_action']


def test_branch_splitting_and_permutation_preserve_conditional_quantities():
    reference = metrics.decompose_cases(example(), {'a': 0}, horizon=4)['cases'][0]
    split = example()
    first = split[0]['branches'].pop(0)
    split[0]['branches'].extend([branch(.1, first['costs']), branch(.1, first['costs'])])
    actual = metrics.decompose_cases(split, {'a': 0}, horizon=4)['cases'][0]
    assert actual['branches'] == 3 and actual['support_mass'] == reference['support_mass']
    compare_quantities(actual, reference)
    split[0]['branches'].reverse()
    permuted = metrics.decompose_cases(split, {'a': 0}, horizon=4)['cases'][0]
    compare_quantities(permuted, reference)


def test_rescaling_unconditional_branch_mass_preserves_conditional_estimand():
    reference = metrics.decompose_cases(example(), {'a': 0}, horizon=4)['cases'][0]
    cases = example()
    for row in cases[0]['branches']:
        row['weight'] *= .5
    actual = metrics.decompose_cases(cases, {'a': 0}, horizon=4)['cases'][0]
    assert actual['support_mass'] == reference['support_mass'] * .5
    compare_quantities(actual, reference)


def test_case_permutation_preserves_equal_case_aggregates():
    cases = example() + [{'case_id': 'b', 'branches': [branch(1., [2., 0., 4., 5.])]}]
    choices = {'a': 0, 'b': 0}
    first = metrics.decompose_cases(cases, choices, horizon=3)
    second = metrics.decompose_cases(list(reversed(cases)), choices, horizon=3)
    assert first['aggregate'] == second['aggregate']
    assert {r['case_id']: r for r in first['cases']} == {r['case_id']: r for r in second['cases']}


def test_only_common_legal_actions_define_both_minima():
    legal = [False, True, False, True]
    cases = [{'case_id': 'a', 'branches': [branch(.5, [0., 4., 0., 7.], legal),
                                         branch(.5, [0., 8., 0., 3.], legal)]}]
    row = metrics.decompose_cases(cases, {'a': 1}, horizon=2)['cases'][0]
    assert row['blind_optimal_action'] == 3
    assert row['conditional_oracle_cost'] == 3.5
    assert row['information_advantage'] == 1.5
    assert row['approximation_regret'] == 1.
    assert row['total_regret'] == 2.5
    with pytest.raises(ValueError, match='chosen action'):
        metrics.decompose_cases(cases, {'a': 0}, horizon=2)
    cases[0]['branches'][0]['weight'] = 0.
    cases[0]['branches'][0]['legal'][0] = True
    with pytest.raises(ValueError, match='common legal'):
        metrics.decompose_cases(cases, {'a': 1}, horizon=2)


def test_tied_legal_minimum_uses_lowest_legal_action():
    cases = [{'case_id': 'a', 'branches': [branch(1., [0., 2., 2., 2.], [False, True, True, True])]}]
    row = metrics.decompose_cases(cases, {'a': 3}, horizon=2)['cases'][0]
    assert row['blind_optimal_action'] == 1
    assert row['information_advantage'] == row['approximation_regret'] == row['total_regret'] == 0.


def test_zero_weight_branch_is_retained_but_contributes_no_cost():
    cases = example()
    cases[0]['branches'].append(branch(0., [1e308] * 4))
    row = metrics.decompose_cases(cases, {'a': 0}, horizon=6)['cases'][0]
    assert row['branches'] == 3 and row['positive_weight_branches'] == 2
    assert row['total_regret'] == pytest.approx(2.4)
    assert row['conditional_cost_variances'] == pytest.approx([8.64, .96, 0., 0.])


def test_decomposition_roundoff_is_recorded_without_erasing_direct_regret():
    cases = [{'case_id': 'a', 'branches': [branch(.5, [1e16 + 2., 1e16, 1e16, 1e16]),
                                         branch(.5, [1e16, 1e16, 1e16, 1e16])]}]
    row = metrics.decompose_cases(cases, {'a': 0}, horizon=8)['cases'][0]
    assert row['total_regret'] == 1.
    assert row['approximation_regret'] == 0.
    assert row['identity']['residual'] == 1.
    assert any(r['quantity'] == 'decomposition_identity' and r['raw'] == 1. for r in row['roundoff_records'])


def test_tiny_probability_mass_roundoff_is_explicit_not_clipped():
    weight = math.nextafter(1., math.inf)
    cases = [{'case_id': 'a', 'branches': [branch(weight, [1., 2., 3., 4.])]}]
    row = metrics.decompose_cases(cases, {'a': 0}, horizon=1)['cases'][0]
    assert row['support_mass'] == weight
    assert row['roundoff_records'][0]['quantity'] == 'support_mass_above_one'
    assert row['roundoff_records'][0]['raw'] > 0.


def test_negative_roundoff_guard_never_hides_a_material_error():
    records = []
    assert metrics._difference(1., math.nextafter(1., math.inf), 'test', records) == 0.
    assert len(records) == 1 and records[0]['raw'] < 0. and records[0]['stored'] == 0.
    with pytest.raises(ValueError, match='nonnegative decomposition'):
        metrics._difference(1., 2., 'test', records)


@pytest.mark.parametrize('weight', [-.1, math.nan, math.inf, True, 1.000001, 10**400])
def test_invalid_probability_weight_rejected(weight):
    cases = [{'case_id': 'a', 'branches': [branch(weight, [1., 2., 3., 4.])]}]
    with pytest.raises(ValueError):
        metrics.decompose_cases(cases, {'a': 0}, horizon=1)


@pytest.mark.parametrize('value', [-1., math.nan, math.inf, True, '1', 10**400])
def test_invalid_teacher_cost_rejected_even_at_zero_weight(value):
    cases = [{'case_id': 'a', 'branches': [branch(0., [value, 2., 3., 4.])]}]
    with pytest.raises(ValueError):
        metrics.decompose_cases(cases, {'a': 0}, horizon=1)


@pytest.mark.parametrize('legal', [[False] * 4, [1, 1, 1, 1], [True] * 3, [True, True, True, None]])
def test_invalid_legal_mask_rejected(legal):
    cases = [{'case_id': 'a', 'branches': [branch(1., [1., 2., 3., 4.], legal)]}]
    with pytest.raises(ValueError):
        metrics.decompose_cases(cases, {'a': 0}, horizon=1)


@pytest.mark.parametrize('choices', [{}, {'a': 0, 'extra': 1}, {'a': True}, {'a': -1}, {'a': 4}, {'a': 1.0}])
def test_exact_one_integer_choice_per_case(choices):
    with pytest.raises(ValueError):
        metrics.decompose_cases(example(), choices, horizon=1)


@pytest.mark.parametrize('horizon', [0, -1, True, 1., None])
def test_horizon_is_explicit_single_positive_integer(horizon):
    with pytest.raises(ValueError):
        metrics.decompose_cases(example(), {'a': 0}, horizon=horizon)


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'extra', 'cost_shape', 'mass'])
def test_malformed_case_and_branch_contracts_fail_closed(change):
    cases = example()
    if change == 'duplicate':
        cases.append(copy.deepcopy(cases[0]))
    elif change == 'missing':
        del cases[0]['branches'][0]['costs']
    elif change == 'extra':
        cases[0]['branches'][0]['outcome'] = 4
    elif change == 'cost_shape':
        cases[0]['branches'][0]['costs'].pop()
    else:
        cases[0]['branches'][0]['weight'] = .8
    with pytest.raises(ValueError):
        metrics.decompose_cases(cases, {'a': 0}, horizon=2)


def test_arithmetic_overflow_leaves_caller_inputs_unchanged():
    cases = [{'case_id': 'a', 'branches': [branch(.5, [0.] * 4), branch(.5, [1e308] * 4)]}]
    choices = {'a': 0}
    before = copy.deepcopy((cases, choices))
    with pytest.raises(ValueError, match='finite'):
        metrics.decompose_cases(cases, choices, horizon=4)
    assert (cases, choices) == before
