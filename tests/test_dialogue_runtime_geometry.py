"""Invented metadata only; no actor, tokenizer, model, timing or file input."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/dialogue_runtime_geometry.py"
SPEC = importlib.util.spec_from_file_location("runtime_geometry_tested", SOURCE)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def profile(did, split="dev", **changes):
    work = dict.fromkeys(geometry.FIELDS, 4)
    work.update(chunk_tokens=254, chunk_batch_size=32, truncated_tokens=0,
                overlength_texts_chunked=0, **changes)
    return {"split": split, "dialogue_id": did, "work": work}


def fixture():
    dev = [profile("warm0", queries=100), profile("warm1", queries=200),
           profile("e0", queries=2, public_user_turns=10),
           profile("e1", queries=6, public_user_turns=2),
           profile("v0", queries=3), profile("v1", queries=8)]
    calibration = [profile("at-low", "train", queries=2), profile("at-high", "train", queries=8),
                   profile("below", "train", queries=1), profile("above", "train", queries=9)]
    sample = {"warmup_ids": ["warm0", "warm1"],
              "estimator": {"dialogue_ids": ["e1", "e0"]},
              "verification": {"dialogue_ids": ["v0", "v1"]}}
    return dev, calibration, sample


def test_exact_ranges_all_fields_and_closed_boundary_uncovered_membership():
    result = geometry.report_geometry(*fixture())
    assert result["cohorts"]["estimator"]["fields"]["queries"] == {"min": 2, "max": 6, "range": 4}
    assert result["cohorts"]["verification"]["fields"]["queries"] == {"min": 3, "max": 8, "range": 5}
    assert result["cohorts"]["timed_union"]["fields"]["queries"] == {"min": 2, "max": 8, "range": 6}
    assert result["cohorts"]["calibration"]["fields"]["queries"] == {"min": 1, "max": 9, "range": 8}
    for cohort in result["cohorts"].values():
        assert set(cohort["fields"]) == set(geometry.FIELDS)
        assert "chunk_tokens" not in cohort["fields"]
    field = result["uncovered_calibration"]["timed_union"]["by_field"]["queries"]
    assert field == {"below_min_count": 1, "below_min_dialogue_ids": ["below"],
                     "above_max_count": 1, "above_max_dialogue_ids": ["above"]}
    assert result["uncovered_calibration"]["timed_union"]["outside_any_field_dialogue_ids"] == ["below", "above"]
    assert result["settings"] == {"chunk_tokens": 254, "chunk_batch_size": 32}
    assert result["used_for_admission"] is False and result["diagnostic_only"] is True
    assert "admitted" not in result and "passed" not in result


def test_each_block_reported_separately_warmup_outliers_not_counted():
    result = geometry.report_geometry(*fixture())
    assert result["cohorts"]["timed_union"]["dialogue_ids"] == ["e1", "e0", "v0", "v1"]
    assert result["source_profile_counts"] == {"dev": 6, "calibration": 4,
                                               "warmups_excluded_from_measured_ranges": 2}
    assert result["uncovered_calibration"]["estimator"]["by_field"]["queries"]["above_max_dialogue_ids"] == ["at-high", "above"]
    assert result["uncovered_calibration"]["verification"]["by_field"]["queries"]["below_min_dialogue_ids"] == ["at-low", "below"]


def test_zero_fields_have_zero_range_without_ratios_or_division():
    result = geometry.report_geometry(*fixture())
    for name in ("truncated_tokens", "overlength_texts_chunked"):
        assert result["cohorts"]["timed_union"]["fields"][name] == {"min": 0, "max": 0, "range": 0}
        assert result["uncovered_calibration"]["timed_union"]["by_field"][name]["above_max_count"] == 0


def test_no_input_mutation_and_json_only_result():
    args = fixture()
    before = copy.deepcopy(args)
    result = geometry.report_geometry(*args)
    assert args == before
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    result["cohorts"]["estimator"]["dialogue_ids"].append("new")
    assert args == before


def test_same_bare_id_across_source_splits_is_valid():
    dev, calibration, sample = fixture()
    calibration[0]["dialogue_id"] = "e0"
    result = geometry.report_geometry(dev, calibration, sample)
    assert result["cohorts"]["calibration"]["source_split"] == "train"
    assert result["cohorts"]["estimator"]["source_split"] == "dev"
    assert "e0" in result["cohorts"]["calibration"]["dialogue_ids"]


def test_marginal_coverage_does_not_claim_joint_shape_coverage():
    dev = [profile("w0"), profile("w1"), profile("e", queries=10, public_user_turns=2),
           profile("v", queries=2, public_user_turns=10)]
    calibration = [profile("joint-unmeasured", "train", queries=8, public_user_turns=8)]
    sample = {"warmup_ids": ["w0", "w1"], "estimator": {"dialogue_ids": ["e"]},
              "verification": {"dialogue_ids": ["v"]}}
    result = geometry.report_geometry(dev, calibration, sample)
    assert result["uncovered_calibration"]["timed_union"]["outside_any_field_count"] == 0
    assert result["joint_batch_coverage"]["status"] == "unknown"
    assert result["used_for_admission"] is False


@pytest.mark.parametrize("defect", ["duplicate_dev", "duplicate_cal", "wrong_split", "missing_field", "extra_field",
                                     "bool", "float", "negative", "settings", "missing_id", "overlap",
                                     "warmup_overlap", "duplicate_sample", "empty_sample", "empty_profiles"])
def test_invalid_metadata_fails_while_numeric_outliers_do_not(defect):
    dev, calibration, sample = fixture()
    if defect == "duplicate_dev":
        dev.append(copy.deepcopy(dev[0]))
    elif defect == "duplicate_cal":
        calibration.append(copy.deepcopy(calibration[0]))
    elif defect == "wrong_split":
        calibration[0]["split"] = "dev"
    elif defect == "missing_field":
        dev[0]["work"].pop("encoder_calls")
    elif defect == "extra_field":
        dev[0]["work"]["made_up"] = 1
    elif defect in ("bool", "float", "negative"):
        calibration[0]["work"]["encoder_calls"] = {"bool": True, "float": 1.0, "negative": -1}[defect]
    elif defect == "settings":
        calibration[0]["work"]["chunk_tokens"] = 128
    elif defect == "missing_id":
        sample["estimator"]["dialogue_ids"][0] = "absent"
    elif defect == "overlap":
        sample["verification"]["dialogue_ids"][0] = "e0"
    elif defect == "warmup_overlap":
        sample["estimator"]["dialogue_ids"][0] = "warm0"
    elif defect == "duplicate_sample":
        sample["estimator"]["dialogue_ids"] = ["e0", "e0"]
    elif defect == "empty_sample":
        sample["verification"]["dialogue_ids"] = []
    else:
        calibration = []
    with pytest.raises(ValueError):
        geometry.report_geometry(dev, calibration, sample)
