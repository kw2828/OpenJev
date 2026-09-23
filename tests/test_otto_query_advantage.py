"""Hand-derived synthetic records only; no empirical arrays or scientific calls."""
from __future__ import annotations

import copy
import json
import math

import pytest

from openjev.research.otto_query_advantage import summarize_advantage, summarize_signal
from openjev.research.otto_teacher_costs import ContinuationRecord


def make_panel(differences, *, same=False, horizon=32, replicas=None):
    replicas = list(range(len(differences))) if replicas is None else replicas
    records = []
    for rep, difference in zip(replicas, differences, strict=True):
        if same:
            assert difference == 0
            records.append(ContinuationRecord(rep, 1, 2, True))
        else:
            records.extend((ContinuationRecord(rep, 1, 8 + difference, True),
                            ContinuationRecord(rep, 3, 8, True)))
    return summarize_advantage(records, analytic_action=1, neural_action=1 if same else 3,
                               replicate_ids=replicas, horizon=horizon)


def row(anchor, episode, differences, *, same=False):
    return {"anchor_id": anchor, "episode_id": episode, "reduction": make_panel(differences, same=same)}


def test_hand_paired_sign_variance_and_canonical_half_split():
    # Sorted replicate IDs put [-2,0] in the first half and [2,4] in the second.
    panel = make_panel([4, -2, 2, 0], replicas=[9, 2, 7, 5])
    assert panel["replicate_ids"] == [2, 5, 7, 9]
    assert panel["differences"] == [-2, 0, 2, 4]
    assert panel["sum_difference"] == 4
    assert panel["sum_squared_difference"] == 24
    assert panel["variance_numerator"] == 80
    assert panel["mean_advantage"] == 1
    assert panel["sample_variance"] == pytest.approx(20 / 3)
    assert panel["variance_of_mean"] == pytest.approx(5 / 3)
    assert panel["paired_standard_error"] == pytest.approx(math.sqrt(5 / 3))
    assert [part["mean_advantage"] for part in panel["halves"]] == [-1, 3]
    assert [part["sign"] for part in panel["halves"]] == [-1, 1]
    assert (panel["positive_count"], panel["negative_count"], panel["tie_count"]) == (2, 1, 1)
    assert not panel["split_same_sign"] and not panel["split_same_nonzero_sign"]
    records = [ContinuationRecord(**record) for record in reversed(panel["records"])]
    assert summarize_advantage(records, analytic_action=1, neural_action=3,
                               replicate_ids=[9, 7, 5, 2], horizon=32) == panel


def test_capped_cost_preserves_found_at_horizon_and_unsuccessful_status():
    records = [ContinuationRecord(rep, action, cost, found)
               for rep, a, b in [(0, (5, True), (5, False)), (1, (5, False), (2, True)),
                                 (2, (2, True), (5, False)), (3, (5, False), (5, False))]
               for action, (cost, found) in ((0, a), (2, b))]
    panel = summarize_advantage(records, analytic_action=0, neural_action=2,
                               replicate_ids=range(4), horizon=5)
    assert panel["differences"] == [0, 3, -3, 0]
    assert panel["mean_advantage"] == 0
    assert panel["branches"]["analytic"]["censored_count"] == 2
    assert panel["branches"]["analytic"]["at_cap_count"] == 3
    assert panel["branches"]["neural"]["censored_count"] == 3
    assert panel["both_censored_count"] == 1
    assert panel["tie_count"] == 2 and not panel["structural_zero"]


def test_same_action_reuses_one_real_record_and_keeps_censor_information():
    records = [ContinuationRecord(rep, 2, steps, found)
               for rep, steps, found in [(0, 1, True), (1, 32, False), (2, 32, True), (3, 6, True)]]
    panel = summarize_advantage(records, analytic_action=2, neural_action=2,
                               replicate_ids=range(4), horizon=32)
    assert panel["physical_record_count"] == 4
    assert panel["structural_zero"] and panel["differences"] == [0] * 4
    assert panel["mean_advantage"] == panel["sample_variance"] == panel["paired_standard_error"] == 0
    assert panel["branches"]["analytic"] == panel["branches"]["neural"]
    assert panel["both_censored_count"] == 1
    assert panel["split_same_sign"] and not panel["split_same_nonzero_sign"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_advantage(records + records, analytic_action=2, neural_action=2,
                            replicate_ids=range(4), horizon=32)


def test_hand_equal_episode_moments_and_centered_repeatability():
    rows = [row(2, "b", [4] * 4), row(0, "a", [0] * 4), row(1, "a", [2] * 4)]
    result = summarize_signal(rows, episode_ids=["b", "a"])
    assert [entry["weight"] for entry in result["weights"]] == [0.25, 0.25, 0.5]
    stats = result["all"]["statistics"]
    assert stats["mean_advantage"] == 2.5
    assert stats["second_moment"] == stats["untruncated_signal"] == 9
    assert stats["estimated_mean_noise"] == 0
    assert stats["cross_half_covariance"] == stats["pooled_half_variance"] == 2.75
    assert stats["repeatability"] == stats["half_correlation"] == 1
    assert stats["uncentered_reliability"] == 1
    assert result == summarize_signal(list(reversed(rows)), episode_ids=["a", "b"])


def test_different_action_subset_renormalizes_original_weights_not_episodes():
    rows = [row(0, "a", [0] * 4, same=True), row(1, "a", [2] * 4), row(2, "b", [4] * 4)]
    result = summarize_signal(rows, episode_ids=["a", "b"])
    group = result["different_action"]
    assert group["anchor_count"] == 2 and group["original_weight_mass"] == 0.75
    assert group["statistics"]["mean_advantage"] == pytest.approx(10 / 3)
    # New equal-episode weighting would incorrectly return 3.
    assert group["statistics"]["mean_advantage"] != 3
    assert result["all"]["statistics"]["structural_zero_fraction"] == 0.25
    assert group["statistics"]["structural_zero_fraction"] == 0


def test_signed_noise_estimate_is_unclipped_and_constant_signal_has_no_selectivity():
    noisy = summarize_signal([row(0, "a", [-2, 0, 0, 4])], episode_ids=["a"])["all"]["statistics"]
    assert noisy["second_moment"] == 0.25
    assert noisy["estimated_mean_noise"] == pytest.approx(19 / 12)
    assert noisy["untruncated_signal"] == pytest.approx(-4 / 3)
    assert noisy["uncentered_reliability"] == pytest.approx(-16 / 3)
    assert noisy["cross_half_second_moment"] == -2
    assert noisy["pooled_half_variance"] == 0 and noisy["repeatability"] is None
    assert noisy["half_correlation"] is None
    constant = summarize_signal([row(i, "a", [2] * 4) for i in range(3)], episode_ids=["a"])
    stats = constant["all"]["statistics"]
    assert stats["uncentered_reliability"] == 1
    assert stats["cross_half_covariance"] == stats["pooled_half_variance"] == 0
    assert stats["repeatability"] is None


def test_anticorrelated_halves_have_negative_centered_repeatability():
    rows = [row(0, "a", [0, 0, 4, 4]), row(1, "b", [4, 4, 0, 0])]
    stats = summarize_signal(rows, episode_ids=["a", "b"])["all"]["statistics"]
    assert stats["cross_half_covariance"] == -4
    assert stats["pooled_half_variance"] == 4
    assert stats["repeatability"] == stats["half_correlation"] == -1


def test_empty_differing_subset_is_explicit_and_zero_denominators_are_not_success():
    result = summarize_signal([row(0, "a", [0] * 4, same=True)], episode_ids=["a"])
    assert result["different_action"] == {"anchor_count": 0, "episode_count": 0,
                                          "original_weight_mass": 0.0, "statistics": None}
    stats = result["all"]["statistics"]
    assert stats["repeatability"] is stats["half_correlation"] is stats["uncentered_reliability"] is None
    assert stats["zero_difference_fraction"] == stats["structural_zero_fraction"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "premature", "boolean", "found_type"])
def test_malformed_endpoint_panels_are_not_reduced(change):
    records = [ContinuationRecord(**record) for record in make_panel([0, 1, 2, 3])["records"]]
    if change == "missing":
        records.pop()
    elif change == "extra":
        records.append(ContinuationRecord(0, 0, 1, True))
    elif change == "duplicate":
        records.append(records[0])
    elif change == "premature":
        records[0] = ContinuationRecord(0, 1, 3, False)
    elif change == "boolean":
        records[0] = ContinuationRecord(True, 1, 3, True)
    else:
        records[0] = ContinuationRecord(0, 1, 3, 1)
    with pytest.raises(ValueError):
        summarize_advantage(records, analytic_action=1, neural_action=3, replicate_ids=range(4), horizon=32)


def test_missing_episode_duplicate_anchor_and_corrupted_reduction_are_rejected():
    rows = [row(0, "a", [1] * 4)]
    for invalid, episodes in ((rows, ["a", "b"]), (rows + rows, ["a"]), (rows, ["a", "a"]),
                              (rows, ["b"]), ([], ["a"])):
        with pytest.raises(ValueError):
            summarize_signal(invalid, episode_ids=episodes)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["reduction"]["mean_advantage"] = 2.0
    with pytest.raises(ValueError, match="disagrees"):
        summarize_signal(corrupted, episode_ids=["a"])
    corrupted[0]["reduction"]["mean_advantage"] = True  # True == 1.0 is not a valid saved statistic.
    with pytest.raises(ValueError, match="disagrees"):
        summarize_signal(corrupted, episode_ids=["a"])


def test_declarations_require_even_replicates_bounded_horizon_and_matching_panels():
    records = [ContinuationRecord(**record) for record in make_panel([0] * 4)["records"]]
    for replicas, horizon, action in (([0, 1], 32, 1), ([0, 1, 2], 32, 1), ([0, 1, 2, 2], 32, 1),
                                       ([True, 1, 2, 3], 32, 1), (range(4), True, 1),
                                       (range(4), 2189, 1), (range(4), 32, False)):
        with pytest.raises(ValueError):
            summarize_advantage(records, analytic_action=action, neural_action=3,
                                replicate_ids=replicas, horizon=horizon)
    rows = [row(0, "a", [0] * 4), {"anchor_id": 1, "episode_id": "a",
                                     "reduction": make_panel([0] * 6)}]
    with pytest.raises(ValueError, match="share"):
        summarize_signal(rows, episode_ids=["a"])


def test_input_nonmutation_detached_outputs_and_interrupted_records():
    rows = [row(0, "a", [1, 2, 3, 4])]
    original = copy.deepcopy(rows)
    result = summarize_signal(rows, episode_ids=["a"])
    assert rows == original
    result["replicate_ids"].append(99)
    result["weights"][0]["weight"] = -1
    assert rows == original

    def interrupted():
        yield ContinuationRecord(0, 1, 1, True)
        raise InterruptedError("original record read failed")

    with pytest.raises(InterruptedError, match="original record read failed"):
        summarize_advantage(interrupted(), analytic_action=1, neural_action=3,
                            replicate_ids=range(4), horizon=32)
