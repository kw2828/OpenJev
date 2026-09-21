"""Synthetic public geometry and paid durations; no files, models or real IDs."""
from __future__ import annotations

import copy
import hashlib
import math

import pytest

from openjev.research import dialogue_runtime_projection as runtime


def public_profiles():
    return [{"split": "dev", "dialogue_id": f"synthetic-{index:04d}",
             "work": {"encoder_calls": 1 + index % 4, "padded_attention_positions": 100 + index,
                      "real_question_updates": 2 + index % 9}}
            for index in range(2363)]


def independent_rank(did):
    return hashlib.sha256(("openjev-calibration-runtime-v2:" + did).encode()).hexdigest(), did


def work(count=64, calls=128, attention=6400, updates=256):
    return dict(zip(runtime.WORK_MEASURES, (count, calls, attention, updates), strict=True))


def prediction(fit_id=None, seconds=5):
    return runtime.predict_verification(fit_id or runtime.FIT_ORDER[0], seconds, work(),
                                        work(calls=256, attention=9600, updates=512))


def projection_fixture():
    records = {fit_id: runtime.validate_verification(prediction(fit_id), 15) for fit_id in runtime.FIT_ORDER}
    calibration = work(count=512, calls=1024, attention=64000, updates=2048)
    costs = {"loading_seconds_by_fit": dict.fromkeys(runtime.FIT_ORDER, 1),
             "warmup_seconds_by_fit": dict.fromkeys(runtime.FIT_ORDER, 2),
             "preparation_parent_wall_seconds": 30, "pilot_nonblock_overhead_seconds": 15}
    return records, calibration, costs


def test_hash_sample_is_fixed_disjoint_ordered_and_excludes_legacy_warmups():
    profiles = public_profiles()
    ids = sorted((p["dialogue_id"] for p in profiles), key=independent_rank)
    warmups = ids[:2]
    before = copy.deepcopy(profiles)
    selected = runtime.select_timing_sample(profiles, warmups)
    assert selected == runtime.select_timing_sample(list(reversed(profiles)), warmups)
    estimator, verification = selected["estimator"], selected["verification"]
    assert estimator["dialogue_ids"] == ids[2:66]
    assert verification["dialogue_ids"] == ids[66:130]
    assert not set(estimator["dialogue_ids"]) & set(verification["dialogue_ids"])
    assert not set(warmups) & set(estimator["dialogue_ids"] + verification["dialogue_ids"])
    assert selected["warmup_ids"] == warmups
    assert selected["salt"] == "openjev-calibration-runtime-v2:"
    expected_fits = [f"{arm}-{seed}" for seed in (6901, 6902, 6903)
                     for arm in ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")]
    assert selected["fit_order"] == expected_fits and len(set(expected_fits)) == 12
    assert selected["counts"]["timed_forwards"] == 1536
    assert selected["counts"]["paid_parity_checked_warmups"] == 24
    indexed = {p["dialogue_id"]: p for p in profiles}
    for block in (estimator, verification):
        assert block["work"]["dialogue_count"] == 64 and len(block["profiles"]) == 64
        for key in runtime.WORK_MEASURES[1:]:
            assert block["work"][key] == sum(indexed[did]["work"][key] for did in block["dialogue_ids"])
        assert [p["dialogue_id"] for p in block["profiles"]] == block["dialogue_ids"]
        assert all(p["salted_id_sha256"] == independent_rank(p["dialogue_id"])[0] for p in block["profiles"])
    assert profiles == before
    estimator["profiles"][0]["work"]["encoder_calls"] += 99
    assert profiles == before


def test_selection_ignores_work_magnitude_and_never_accesses_outcomes():
    class PublicOnly(dict):
        def __getitem__(self, key):
            assert key in {"split", "dialogue_id", "work"}, f"Nonpublic selection field: {key}"
            return super().__getitem__(key)

    profiles = public_profiles()
    warmups = [p["dialogue_id"] for p in profiles[:2]]
    expected = runtime.select_timing_sample(profiles, warmups)
    changed = [PublicOnly({**p, "work": {key: value * 1000 for key, value in p["work"].items()},
                           "model_score": object(), "target": object(), "case_seconds": object()}) for p in profiles]
    actual = runtime.select_timing_sample(changed, warmups)
    for name in ("estimator", "verification"):
        assert actual[name]["dialogue_ids"] == expected[name]["dialogue_ids"]


@pytest.mark.parametrize("defect", ["missing_profile", "extra_profile", "duplicate_id", "train_split", "empty_id",
                                    "unknown_warmup", "duplicate_warmup", "one_warmup", "three_warmups",
                                    "zero_work", "negative_work", "float_work", "boolean_work"])
def test_sampling_requires_complete_public_cohort_and_exact_work_metadata(defect):
    profiles = public_profiles()
    warmups = [p["dialogue_id"] for p in profiles[:2]]
    if defect == "missing_profile":
        profiles.pop()
    elif defect == "extra_profile":
        profiles.append(profiles[0])
    elif defect == "duplicate_id":
        profiles[-1]["dialogue_id"] = profiles[0]["dialogue_id"]
    elif defect == "train_split":
        profiles[0]["split"] = "train"
    elif defect == "empty_id":
        profiles[0]["dialogue_id"] = ""
    elif defect == "unknown_warmup":
        warmups[0] = "not-evaluated"
    elif defect == "duplicate_warmup":
        warmups[1] = warmups[0]
    elif defect == "one_warmup":
        warmups.pop()
    elif defect == "three_warmups":
        warmups.append(profiles[2]["dialogue_id"])
    else:
        profiles[0]["work"]["encoder_calls"] = {"zero_work": 0, "negative_work": -1,
                                                 "float_work": 1.0, "boolean_work": True}[defect]
    with pytest.raises(ValueError):
        runtime.select_timing_sample(profiles, warmups)


def test_verification_prediction_uses_maximum_of_all_four_ratios_and_two_times_margin():
    estimate = prediction()
    assert estimate["work_ratios"] == {"dialogue_count": 1.0, "encoder_calls": 2.0,
                                       "padded_attention_positions": 1.5, "real_question_updates": 2.0}
    assert estimate["maximum_work_ratio"] == 2 and estimate["safety_factor"] == 2
    assert estimate["predicted_seconds"] == 20
    lower_other_work = runtime.predict_verification(runtime.FIT_ORDER[0], 5, work(),
                                                     work(calls=32, attention=1600, updates=64))
    assert lower_other_work["maximum_work_ratio"] == 1
    assert lower_other_work["predicted_seconds"] == 10


@pytest.mark.parametrize("actual,passed", [(20.0, True), (math.nextafter(20., math.inf), False),
                                          (math.nextafter(20., -math.inf), True)])
def test_verification_boundary_is_inclusive_without_epsilon_and_keeps_prediction(actual, passed):
    estimate = prediction()
    before = copy.deepcopy(estimate)
    result = runtime.validate_verification(estimate, actual)
    assert result["passed"] is passed and result["actual_seconds"] == actual
    assert result["actual_over_prediction"] == actual / 20
    assert result["prediction"] == before and estimate == before
    result["prediction"]["estimator_work"]["encoder_calls"] += 1
    assert estimate == before


@pytest.mark.parametrize("defect", ["margin", "predicted_seconds", "ratio", "maximum_ratio", "unknown_fit", "extra_field"])
def test_tampered_prediction_arithmetic_is_not_accepted(defect):
    estimate = prediction()
    if defect == "margin":
        estimate["safety_factor"] = 3
    elif defect == "predicted_seconds":
        estimate["predicted_seconds"] += 1
    elif defect == "ratio":
        estimate["work_ratios"]["encoder_calls"] = 3
    elif defect == "maximum_ratio":
        estimate["maximum_work_ratio"] = 3
    elif defect == "unknown_fit":
        estimate["fit_id"] = "another-fit"
    else:
        estimate["selected_from_validation"] = True
    with pytest.raises(ValueError):
        runtime.validate_verification(estimate, 15)


@pytest.mark.parametrize("seconds", [0, -1, True, math.nan, math.inf, -math.inf, 1e308])
def test_estimator_cost_must_be_positive_finite_and_yield_finite_prediction(seconds):
    with pytest.raises(ValueError):
        prediction(seconds=seconds)


@pytest.mark.parametrize("field", runtime.WORK_MEASURES)
def test_each_work_measure_is_required_and_positive(field):
    geometry = work()
    geometry.pop(field)
    with pytest.raises(ValueError, match="four work measures"):
        runtime.predict_verification(runtime.FIT_ORDER[0], 5, geometry, work())
    geometry = work()
    geometry[field] = 0
    with pytest.raises(ValueError):
        runtime.predict_verification(runtime.FIT_ORDER[0], 5, geometry, work())


def test_both_timing_blocks_are_fixed_at_64_and_no_extra_proxy_is_allowed():
    for estimator, verification in ((work(count=63), work()), (work(), work(count=65))):
        with pytest.raises(ValueError, match="dialogue count"):
            runtime.predict_verification(runtime.FIT_ORDER[0], 5, estimator, verification)
    with pytest.raises(ValueError, match="four work measures"):
        runtime.predict_verification(runtime.FIT_ORDER[0], 5, {**work(), "chosen_metric": 1}, work())


def test_projection_includes_every_fit_and_all_declared_paid_components():
    records, calibration, costs = projection_fixture()
    before = copy.deepcopy((records, calibration, costs))
    result = runtime.project_calibration(records, calibration, **costs)
    assert result["fit_order"] == list(runtime.FIT_ORDER)
    assert result["projected_dialogue_forwards"] == 6144
    assert set(result["fits"]) == set(result["verification"]) == set(runtime.FIT_ORDER)
    assert result["components"] == {"projected_calibration_forward_seconds": 1200.0,
                                    "loading_seconds": 12.0, "warmup_seconds": 24.0,
                                    "preparation_parent_wall_seconds": 30.0,
                                    "pilot_nonblock_overhead_seconds": 15.0}
    assert result["total_seconds"] == 1281 and result["threshold_seconds"] == 1800
    assert result["admitted"] and result["all_verifications_passed"]
    assert result["verifications_passed"] == result["verifications_total"] == 12
    assert result["legacy_v1_admission_revised"] is False and result["scientific_conditions_changed"] is False
    for item in result["fits"].values():
        assert item["work_ratios"] == {"dialogue_count": 8., "encoder_calls": 8.,
                                       "padded_attention_positions": 10., "real_question_updates": 8.}
        assert item["projected_seconds"] == 100
    assert (records, calibration, costs) == before


def test_first_fit_timing_outlier_cannot_be_dropped_or_replaced_by_other_fits():
    records, calibration, costs = projection_fixture()
    first = runtime.FIT_ORDER[0]
    records[first] = runtime.validate_verification(prediction(first, seconds=50), 15)
    result = runtime.project_calibration(records, calibration, **costs)
    assert result["fits"][first]["projected_seconds"] == 1000
    assert result["components"]["projected_calibration_forward_seconds"] == 2100
    assert not result["admitted"] and result["all_verifications_passed"]


def test_failed_verification_refuses_admission_without_refitting_projection():
    records, calibration, costs = projection_fixture()
    baseline = runtime.project_calibration(records, calibration, **costs)
    last = runtime.FIT_ORDER[-1]
    records[last] = runtime.validate_verification(records[last]["prediction"], 1000)
    failed = runtime.project_calibration(records, calibration, **costs)
    assert failed["components"] == baseline["components"] and failed["fits"] == baseline["fits"]
    assert failed["total_seconds"] == baseline["total_seconds"] and failed["projection_within_threshold"]
    assert not failed["admitted"] and not failed["all_verifications_passed"]
    assert failed["verifications_passed"] == 11 and failed["verification"][last]["actual_seconds"] == 1000


def test_projection_1800_boundary_is_unchanged_and_has_no_epsilon():
    records, calibration, costs = projection_fixture()
    costs["pilot_nonblock_overhead_seconds"] = 534
    boundary = runtime.project_calibration(records, calibration, **costs)
    assert boundary["total_seconds"] == 1800 and boundary["admitted"]
    costs["pilot_nonblock_overhead_seconds"] += 2 * math.ulp(1800.)
    above = runtime.project_calibration(records, calibration, **costs)
    assert above["total_seconds"] > 1800 and not above["admitted"]


@pytest.mark.parametrize("field", ["records", "loading_seconds_by_fit", "warmup_seconds_by_fit"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_projection_rejects_incomplete_or_extended_fit_membership(field, mutation):
    records, calibration, costs = projection_fixture()
    values = records if field == "records" else costs[field]
    if mutation == "missing":
        values.pop(runtime.FIT_ORDER[-1])
    else:
        values["unplanned-fit"] = next(iter(values.values()))
    with pytest.raises(ValueError, match="Exactly all twelve"):
        runtime.project_calibration(records, calibration, **costs)


@pytest.mark.parametrize("defect", ["wrong_fit", "forged_pass", "different_sample_work", "wrong_calibration_count"])
def test_projection_requires_unchanged_matching_records_and_fixed_cohort_geometry(defect):
    records, calibration, costs = projection_fixture()
    first, last = runtime.FIT_ORDER[0], runtime.FIT_ORDER[-1]
    if defect == "wrong_fit":
        records[last] = records[first]
    elif defect == "forged_pass":
        records[last]["passed"] = False
    elif defect == "different_sample_work":
        altered = runtime.predict_verification(last, 5, work(calls=129), work(calls=256, attention=9600, updates=512))
        records[last] = runtime.validate_verification(altered, 15)
    else:
        calibration["dialogue_count"] = 511
    with pytest.raises(ValueError):
        runtime.project_calibration(records, calibration, **costs)


@pytest.mark.parametrize("cost", ["loading_seconds_by_fit", "warmup_seconds_by_fit",
                                   "preparation_parent_wall_seconds", "pilot_nonblock_overhead_seconds"])
@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, True])
def test_paid_component_errors_fail_closed(cost, bad):
    records, calibration, costs = projection_fixture()
    if cost.endswith("by_fit"):
        costs[cost][runtime.FIT_ORDER[0]] = bad
    else:
        costs[cost] = bad
    with pytest.raises(ValueError):
        runtime.project_calibration(records, calibration, **costs)
