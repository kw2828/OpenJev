"""Artificial arithmetic checks only; no source task examples or models."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("alignment_independent_audit", Path(__file__).with_name("audit.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture_rows():
    result = []
    for i, (target, prior) in enumerate(((2, 0), (1, 0), (0, 0), (2, 2))):
        ids = ("reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False")
        result.append({"row_index": i, "split": "train", "admission": "admitted", "heldout_service": True,
                       "candidate_count": 4, "candidate_types": [0, 1, 2, 3], "current_label_index": target,
                       "previous_current_index": prior, "current_candidate_id": ids[target],
                       "previous_candidate_id": ids[prior], "current_value_group": a.VALUES[target],
                       "derived_bin": "first_assignment" if target != prior else
                           "unmentioned_retention" if target == 0 else "assigned_retention",
                       "service": "A" if i % 2 else "B", "dialogue_id": str(i)})
    return result


def packet():
    logs = np.full((4, 12), -np.inf, np.float32)
    logs[:, :4] = np.log(np.array([[.35, .05, .30, .30], [.1, .7, .1, .1],
                                  [.8, .1, .05, .05], [.1, .1, .7, .1]], np.float64))
    return {"row_indices": np.arange(4, dtype=np.int64), "log_probs": logs}


def test_actual_candidate_branch_not_largest_branch_mass():
    rows = fixture_rows(); meta = a.row_metadata(rows)
    scores, choice = a.score(rows, meta, packet())
    assert choice.tolist() == [0, 1, 0, 2]
    assert scores["decisions"]["wrong_selected_branch"] == {"numerator": 1, "denominator": 2}
    assert scores["decisions"]["accuracy"] == {"numerator": 1, "denominator": 2}
    assert scores["decisions"]["true_false_positive_rate"]["denominator"] == 2
    assert scores["decisions"]["dontcare_false_positive_rate"]["denominator"] == 3
    assert scores["cells"]["heldout_service/changed"]["nll"]["row"] == pytest.approx(-.5*(np.log(.3)+np.log(.7)))


def gate_fits():
    result = {}
    for name in a.ORDER:
        aligned = name.startswith("token_aligned-")
        nums = (520 if aligned else 500, 280 if aligned else 300, 100, 20, 20)
        result[name] = {"decisions": {key: {"numerator": num, "denominator": 1000}
                                      for key, num in zip(a.METRICS, nums, strict=True)}}
    return result


def test_exact_boundaries_and_all_22_checks():
    fits = gate_fits(); rule = a.continuation(fits)
    assert rule["passed"] and rule["checks_passed"] == rule["total_checks"] == 22
    fits["token_aligned-6201"]["decisions"]["retained_error"]["numerator"] += 15
    assert a.continuation(fits)["passed"]  # Three-seed mean exactly +.005.
    fits["token_aligned-6201"]["decisions"]["retained_error"]["numerator"] += 1
    assert not a.continuation(fits)["passed"]


def test_one_seed_regression_cannot_be_hidden_by_mean():
    fits = gate_fits()
    for seed, number in zip(a.SEEDS, (499, 550, 550), strict=True):
        fits[f"token_aligned-{seed}"]["decisions"]["accuracy"]["numerator"] = number
    rule = a.continuation(fits)
    assert rule["checks"]["flat_stratum/mean_accuracy"]["passed"]
    assert not rule["checks"]["flat_stratum/seed_6201_accuracy_nonworse"]["passed"]
    assert not rule["passed"]


def test_missing_type_support_fails_without_division_by_zero():
    fits = gate_fits()
    for value in fits.values():
        value["decisions"]["true_false_positive_rate"] = {"numerator": 0, "denominator": 0}
    assert not a.continuation(fits)["passed"]


@pytest.mark.parametrize("bad", ["row", "mass", "pad", "nan", "dtype"])
def test_invalid_saved_predictions_rejected(bad):
    rows = fixture_rows(); item = packet()
    if bad == "row": item["row_indices"][1] = 0
    elif bad == "mass": item["log_probs"][0, :4] = -.1
    elif bad == "pad": item["log_probs"][0, 5] = -5
    elif bad == "nan": item["log_probs"][0, 0] = np.nan
    else: item["log_probs"] = item["log_probs"].astype(np.float64)
    with pytest.raises(ValueError): a.score(rows, a.row_metadata(rows), item)


def test_log_domain_nll_has_no_probability_floor():
    rows = fixture_rows(); item = packet()
    item["log_probs"][0, :4] = np.array([0., -1000., -1000., -1000.], np.float32)
    scores, _ = a.score(rows, a.row_metadata(rows), item)
    assert scores["cells"]["heldout_service/changed"]["nll"]["row"] > 500


def test_correctness_quadrants_exhaust_same_rows():
    meta = a.row_metadata(fixture_rows())
    counts = a.pair_counts(meta, np.array([2, 0, 0, 0]), np.array([0, 1, 0, 2]))
    assert counts["heldout_service/all"] == {"rows": 4, "wrong_to_correct": 1,
                                            "correct_to_wrong": 2, "both_correct": 1, "both_wrong": 0}


def test_duplicate_rows_and_previous_mapping_rejected():
    rows = fixture_rows(); rows[1]["row_index"] = 0
    with pytest.raises(ValueError): a.row_metadata(rows)
    rows = copy.deepcopy(fixture_rows()); rows[0]["previous_candidate_id"] = rows[0]["current_candidate_id"]
    with pytest.raises(ValueError): a.row_metadata(rows)
