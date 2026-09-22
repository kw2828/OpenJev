"""Hand-computed saved-record fixtures only; no arrays, models or empirical input."""
from __future__ import annotations

import json
import math
import random
from dataclasses import FrozenInstanceError

import pytest

from openjev.research.otto_teacher_costs import ContinuationRecord, summarize_costs


def panel():
    # Paired differences for action1-action3 are 1,2,3, despite larger row costs.
    return [ContinuationRecord(rep, action, cost, True)
            for rep, first, second in [(8, 2, 1), (3, 5, 3), (21, 9, 6)]
            for action, cost in [(1, first), (3, second)]]


def tiny_panel():
    return [ContinuationRecord(rep, action, 1+action, True)
            for rep in (7, 11) for action in (0, 1)]


def test_high_horizon_error_and_centering_preserve_small_integer_differences():
    horizon = 2**53-1
    records = [ContinuationRecord(0, 0, horizon, True), ContinuationRecord(1, 0, horizon-1, True),
               ContinuationRecord(0, 1, 1, True), ContinuationRecord(1, 1, 1, True)]
    result = summarize_costs(records, eligible_actions=[0, 1], replicate_ids=[0, 1], horizon=horizon)
    # Two integer differences one unit apart have paired SE exactly0.5, even
    # when their half-integer mean is not representable as a float.
    assert result['pairs'][0]['paired_standard_error'] == 0.5
    common_offset = [ContinuationRecord(rep, action, horizon-rep-action, True)
                     for rep in (0, 1) for action in (0, 1)]
    centered = summarize_costs(common_offset, eligible_actions=[0, 1], replicate_ids=[0, 1], horizon=horizon)
    # Individually rounded action means can coincide at this offset, although
    # exact integer centering must retain their one-step separation.
    assert [row['centered_mean_cost'] for row in centered['actions']] == [0.5, -0.5]
    assert centered['pairs'][0]['mean_difference'] == 1
    assert centered['pairs'][0]['paired_standard_error'] == 0


def test_hand_computed_means_centering_and_paired_standard_error():
    result = summarize_costs(panel(), eligible_actions=[3, 1], replicate_ids=[8, 3, 21], horizon=10)
    assert [row['action'] for row in result['actions']] == [1, 3]
    assert [row['mean_cost'] for row in result['actions']] == pytest.approx([16/3, 10/3])
    assert [row['centered_mean_cost'] for row in result['actions']] == pytest.approx([1, -1])
    assert all(row['success_fraction'] == 1 and row['censored_fraction'] == 0 for row in result['actions'])
    pair = result['pairs'][0]
    assert (pair['first_action'], pair['second_action']) == (1, 3)
    assert pair['mean_difference'] == pytest.approx(2)
    # Sample variance of [1,2,3] is1, so the paired SE is1/sqrt(3).
    assert pair['paired_standard_error'] == pytest.approx(1/math.sqrt(3))
    assert pair['hoeffding_interval'] == [-9, 9]
    assert pair['resolved_direction'] == 'unresolved'


def test_paired_offsets_cancel_common_shocks_without_false_certainty():
    records = [ContinuationRecord(rep, action, cost+2*action, True)
               for rep, cost in enumerate([1, 4, 8, 11]) for action in (0, 1)]
    result = summarize_costs(records, eligible_actions=[0, 1], replicate_ids=range(4), horizon=13)
    pair = result['pairs'][0]
    assert pair['mean_difference'] == -2
    assert pair['paired_standard_error'] == 0
    # Identical observed differences do not collapse a distribution-free bound.
    assert pair['hoeffding_radius'] > 0
    assert pair['hoeffding_interval'][0] < pair['hoeffding_interval'][1]
    assert pair['resolved_direction'] == 'unresolved'


def test_success_at_cap_and_censoring_have_same_cost_but_distinct_status():
    records = [ContinuationRecord(10, 0, 5, True), ContinuationRecord(20, 0, 5, False),
               ContinuationRecord(10, 1, 2, True), ContinuationRecord(20, 1, 5, True)]
    result = summarize_costs(records, eligible_actions=[0, 1], replicate_ids=[10, 20], horizon=5)
    assert result['actions'] == [
        {'action': 0, 'mean_cost': 5.0, 'centered_mean_cost': 0.75, 'success_fraction': 0.5, 'censored_fraction': 0.5},
        {'action': 1, 'mean_cost': 3.5, 'centered_mean_cost': -0.75, 'success_fraction': 1.0, 'censored_fraction': 0.0},
    ]
    assert result['pairs'][0]['mean_difference'] == 1.5
    assert result['pairs'][0]['paired_standard_error'] == 1.5


def test_horizon_one_is_unresolved_even_when_interval_touches_zero():
    records = [ContinuationRecord(rep, action, 1, (rep+action) % 2 == 0)
               for rep in (0, 1) for action in range(4)]
    result = summarize_costs(records, eligible_actions=range(4), replicate_ids=[0, 1], horizon=1)
    assert all(row['mean_cost'] == 1 and row['centered_mean_cost'] == 0 for row in result['actions'])
    assert result['pair_count'] == 6
    for pair in result['pairs']:
        assert pair['mean_difference'] == pair['paired_standard_error'] == pair['hoeffding_radius'] == 0
        assert pair['hoeffding_interval'] == [0, 0]
        assert pair['resolved_direction'] == 'unresolved'


def test_output_is_invariant_to_row_and_declaration_order():
    records = panel()
    expected = summarize_costs(records, eligible_actions=[1, 3], replicate_ids=[3, 8, 21], horizon=10)
    random.Random(42).shuffle(records)
    actual = summarize_costs(records, eligible_actions=[3, 1], replicate_ids=[21, 8, 3], horizon=10)
    assert actual == expected


def test_four_actions_use_all_six_simultaneous_pair_bounds():
    records = [ContinuationRecord(rep, action, action+1, True) for rep in range(400) for action in range(4)]
    result = summarize_costs(records, eligible_actions=[3, 1, 0, 2], replicate_ids=range(400), horizon=4)
    assert (result['replicate_count'], result['action_count'], result['pair_count']) == (400, 4, 6)
    assert [(p['first_action'], p['second_action']) for p in result['pairs']] == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # Six comparisons, not one: log(2*6/.05)=log(240).
    expected_radius = 6*math.sqrt(math.log(240)/800)
    for pair in result['pairs']:
        assert pair['mean_difference'] == pair['first_action']-pair['second_action']
        assert pair['hoeffding_radius'] == pytest.approx(expected_radius)
        assert pair['resolved_direction'] == 'first_lower'
        assert -3 <= pair['hoeffding_interval'][0] <= pair['hoeffding_interval'][1] <= 3


def test_difference_sign_resolves_the_correct_lower_cost_action():
    for first_cost, second_cost, expected in [(1, 2, 'first_lower'), (2, 1, 'second_lower')]:
        records = [ContinuationRecord(rep, action, cost, True)
                   for rep in range(64) for action, cost in [(0, first_cost), (2, second_cost)]]
        result = summarize_costs(records, eligible_actions=[2, 0], replicate_ids=range(64), horizon=2)
        assert result['pairs'][0]['mean_difference'] == first_cost-second_cost
        assert result['pairs'][0]['resolved_direction'] == expected


def test_one_replicate_has_no_estimated_standard_error_but_keeps_bound():
    records = [ContinuationRecord(17, 0, 1, True), ContinuationRecord(17, 3, 4, False)]
    result = summarize_costs(records, eligible_actions=[0, 3], replicate_ids=[17], horizon=4)
    pair = result['pairs'][0]
    assert result['replicate_count'] == 1
    assert pair['mean_difference'] == -3
    assert pair['paired_standard_error'] is None
    assert math.isfinite(pair['hoeffding_radius']) and pair['hoeffding_radius'] > 0
    assert pair['hoeffding_interval'] == [-3, 3]
    assert pair['resolved_direction'] == 'unresolved'


def test_single_action_needs_no_pairwise_logarithm():
    records = [ContinuationRecord(rep, 2, rep+1, True) for rep in range(3)]
    result = summarize_costs(records, eligible_actions=[2], replicate_ids=range(3), horizon=3)
    assert result['pairs'] == [] and result['pair_count'] == 0
    assert result['actions'] == [
        {'action': 2, 'mean_cost': 2.0, 'centered_mean_cost': 0.0, 'success_fraction': 1.0, 'censored_fraction': 0.0}]


def test_smallest_positive_alpha_keeps_finite_json_compatible_bounds():
    records = [ContinuationRecord(rep, action, action+1, True) for rep in (0, 1) for action in range(4)]
    alpha = float.fromhex('0x0.0000000000001p-1022')
    result = summarize_costs(records, eligible_actions=range(4), replicate_ids=[0, 1], horizon=4, familywise_alpha=alpha)
    assert result['familywise_alpha'] == alpha
    for pair in result['pairs']:
        assert math.isfinite(pair['hoeffding_radius']) and pair['hoeffding_radius'] > 0
        assert pair['hoeffding_interval'] == [-3, 3]
        assert pair['resolved_direction'] == 'unresolved'
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_eligible_actions_are_nonempty_unique_numeric_ids():
    for actions in ([], [0, 0], [-1], [4], [True], [1.0], [0, '1']):
        with pytest.raises((ValueError, TypeError)):
            summarize_costs(tiny_panel(), eligible_actions=actions, replicate_ids=[7, 11], horizon=3)


def test_replicate_ids_are_nonempty_unique_nonnegative_integers():
    for replicates in ([], [7, 7], [-1], [True], [7.0], [7, '11']):
        with pytest.raises((ValueError, TypeError)):
            summarize_costs(tiny_panel(), eligible_actions=[0, 1], replicate_ids=replicates, horizon=3)


def test_horizon_and_alpha_reject_boolean_nonfinite_or_out_of_range_values():
    for horizon in (0, -1, True, 3.0, '3'):
        with pytest.raises((ValueError, TypeError)):
            summarize_costs(tiny_panel(), eligible_actions=[0, 1], replicate_ids=[7, 11], horizon=horizon)
    for alpha in (0, 1, -0.1, 1.1, True, False, math.nan, math.inf, -math.inf, '0.05'):
        with pytest.raises((ValueError, TypeError)):
            summarize_costs(tiny_panel(), eligible_actions=[0, 1], replicate_ids=[7, 11], horizon=3, familywise_alpha=alpha)


def test_wrong_record_type_or_field_types_are_rejected():
    malformed = [None, (7, 0, 1, True), {'replicate_id': 7, 'first_action': 0, 'steps': 1, 'found': True}]
    for record in malformed:
        with pytest.raises((ValueError, TypeError)):
            summarize_costs([record, *tiny_panel()[1:]], eligible_actions=[0, 1], replicate_ids=[7, 11], horizon=3)
    for fields in [(True, 0, 1, True), (7.0, 0, 1, True), (7, False, 1, True), (7, 0.0, 1, True),
                   (7, 0, True, True), (7, 0, 1.0, True), (7, 0, 1, 1), (7, 0, 1, 'true')]:
        with pytest.raises((ValueError, TypeError)):
            records = [ContinuationRecord(*fields), *tiny_panel()[1:]]
            summarize_costs(records, eligible_actions=[0, 1], replicate_ids=[7, 11], horizon=3)


def test_missing_duplicate_and_extra_panel_cells_are_rejected():
    records = tiny_panel()
    for broken in (records[:-1], [*records, records[0]],
                   [*records, ContinuationRecord(12, 0, 1, True)],
                   [*records, ContinuationRecord(7, 2, 1, True)]):
        with pytest.raises((ValueError, TypeError)):
            summarize_costs(broken, eligible_actions=[0, 1], replicate_ids=[7, 11], horizon=3)


def test_invalid_cost_or_premature_censoring_is_rejected():
    for steps, found in [(0, True), (-1, True), (4, True), (2, False)]:
        with pytest.raises((ValueError, TypeError)):
            records = [ContinuationRecord(7, 0, steps, found), *tiny_panel()[1:]]
            summarize_costs(records, eligible_actions=[0, 1], replicate_ids=[7, 11], horizon=3)


def test_inputs_are_unchanged_and_assumptions_are_explicit():
    records, actions, replicates = panel(), [3, 1], [21, 8, 3]
    original_records, original_actions, original_replicates = records.copy(), actions.copy(), replicates.copy()
    result = summarize_costs(records, eligible_actions=actions, replicate_ids=replicates, horizon=10)
    assert (records, actions, replicates) == (original_records, original_actions, original_replicates)
    with pytest.raises(FrozenInstanceError):
        records[0].steps = 1
    assert result['horizon'] == 10 and result['familywise_alpha'] == 0.05
    assert result['scope'] and result['assumptions']
    assumptions = json.dumps(result['assumptions']).lower()
    assert 'nonterminal' in assumptions and 'teacher' in assumptions and 'horizon' in assumptions
    json.dumps(result, allow_nan=False)
