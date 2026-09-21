"""Pure sampling and fixed estimator-only arithmetic for runtime pilot V2.

No files, corpus records, model imports or timing calls. The caller authenticates
the public workload metadata, pays and parity-checks both legacy DEV warmups,
and publishes each prediction before starting its verification block. A pure
function cannot witness that publication order or complete neural execution.
The failed V1 admission and the eleven scientific conditions remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping

VERSION = "dialogue-calibration-runtime-v2"
SALT = "openjev-calibration-runtime-v2:"
DEV_DIALOGUES = 2363
ESTIMATOR_DIALOGUES = VERIFICATION_DIALOGUES = 64
CALIBRATION_DIALOGUES = 512
SAFETY_FACTOR = 2.0
THRESHOLD_SECONDS = 1800
WORK_MEASURES = ("dialogue_count", "encoder_calls", "padded_attention_positions", "real_question_updates")
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
SEEDS = (6901, 6902, 6903)
FIT_ORDER = tuple(f"{arm}-{seed}" for seed in SEEDS for arm in ARMS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(dialogue_id):
    require(type(dialogue_id) is str and dialogue_id, "Nonempty public dialogue ID")
    return hashlib.sha256((SALT + dialogue_id).encode()).hexdigest(), dialogue_id


def _work(work, count):
    require(isinstance(work, Mapping) and set(work) == set(WORK_MEASURES), "Exact four work measures")
    require(all(type(work[key]) is int and work[key] > 0 for key in WORK_MEASURES),
            "Positive integer complete-block work")
    require(work["dialogue_count"] == count, "Fixed complete-block dialogue count")
    return {key: work[key] for key in WORK_MEASURES}


def _seconds(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "Finite nonnegative paid seconds; estimator must be positive")
    return float(value)


def _ratios(target, estimator):
    try:
        ratios = {key: target[key] / estimator[key] for key in WORK_MEASURES}
    except OverflowError as error:
        raise ValueError("Finite work ratios") from error
    require(all(math.isfinite(value) and value > 0 for value in ratios.values()), "Finite positive work ratios")
    return ratios


def _scaled_seconds(seconds, ratios):
    result = SAFETY_FACTOR * seconds * max(ratios.values())
    require(math.isfinite(result) and result > 0, "Finite positive fixed-margin prediction")
    return result


def select_timing_sample(public_profiles, replay_ids):
    """Fixed 128-ID sample from the complete evaluated DEV public workload.

    Both legacy replay IDs are excluded before hash ranking. The first 64 IDs
    are estimator-only and the next 64 are verification-only. Every fit uses
    exactly the returned order, with no length/timing/outcome-based selection.
    """
    require(type(public_profiles) is list and len(public_profiles) == DEV_DIALOGUES,
            "Exactly all 2363 evaluated DEV public profiles")
    require(type(replay_ids) in (list, tuple) and len(replay_ids) == 2
            and all(type(did) is str and did for did in replay_ids) and len(set(replay_ids)) == 2,
            "Exactly two distinct legacy replay IDs")
    indexed = {}
    for profile in public_profiles:
        require(isinstance(profile, Mapping) and profile["split"] == "dev", "Original DEV profile identity")
        did = profile["dialogue_id"]
        require(type(did) is str and did and did not in indexed, "Distinct public DEV workload identities")
        geometry = profile["work"]
        require(isinstance(geometry, Mapping), "Public workload mapping")
        work = _work({"dialogue_count": 1, **{key: geometry[key] for key in WORK_MEASURES[1:]}}, 1)
        indexed[did] = {"dialogue_id": did, "salted_id_sha256": rank(did)[0], "work": work}
    require(set(replay_ids) <= set(indexed), "Legacy replay IDs belong to evaluated DEV")
    ranked = sorted(set(indexed) - set(replay_ids), key=rank)
    selected = ranked[:ESTIMATOR_DIALOGUES + VERIFICATION_DIALOGUES]
    require(len(selected) == 128 and len(set(selected)) == 128, "Complete distinct 128-dialogue sample")

    def block(ids):
        profiles = [copy.deepcopy(indexed[did]) for did in ids]
        total = {key: sum(p["work"][key] for p in profiles) for key in WORK_MEASURES}
        return {"dialogue_ids": list(ids), "profiles": profiles, "work": total}

    return {
        "version": VERSION, "salt": SALT, "source_split": "dev", "analysis_role": "runtime_qualification",
        "fit_order": list(FIT_ORDER), "warmup_ids": list(replay_ids),
        "estimator": block(selected[:ESTIMATOR_DIALOGUES]),
        "verification": block(selected[ESTIMATOR_DIALOGUES:]),
        "counts": {"evaluated_dev_dialogues": DEV_DIALOGUES, "excluded_legacy_replay_dialogues": 2,
                   "eligible_dialogues": DEV_DIALOGUES - 2, "selected_dialogues": 128,
                   "estimator_dialogues": ESTIMATOR_DIALOGUES, "verification_dialogues": VERIFICATION_DIALOGUES,
                   "fits": len(FIT_ORDER), "timed_forwards": 128 * len(FIT_ORDER),
                   "paid_parity_checked_warmups": 2 * len(FIT_ORDER)},
        "selection_rule": "Exclude both legacy replay IDs, sort by salted SHA256 then ID; first 64 estimate, next 64 verify",
        "scope": "Work metadata only; caller authenticates input identity and freezes this exact order before model calls",
    }


def predict_verification(fit_id, estimator_seconds, estimator_work, verification_work):
    """Produce the complete record that must be frozen before verification."""
    require(fit_id in FIT_ORDER, "One of the twelve fixed fit IDs")
    seconds = _seconds(estimator_seconds, positive=True)
    estimator = _work(estimator_work, ESTIMATOR_DIALOGUES)
    verification = _work(verification_work, VERIFICATION_DIALOGUES)
    ratios = _ratios(verification, estimator)
    return {
        "version": VERSION, "fit_id": fit_id, "estimator_seconds": seconds,
        "estimator_work": estimator, "verification_work": verification,
        "work_ratios": ratios, "maximum_work_ratio": max(ratios.values()),
        "safety_factor": SAFETY_FACTOR, "predicted_seconds": _scaled_seconds(seconds, ratios),
        "estimation_data": "Entire first 64-dialogue estimator block only; no dropped first-fit or timing outliers",
        "freeze_required": "Publish and bind this record before timing any verification dialogue",
    }


def validate_verification(prediction, actual_seconds):
    """Check the prior prediction; never update its margin or estimator."""
    require(type(prediction) is dict, "Frozen prediction mapping")
    expected = predict_verification(prediction["fit_id"], prediction["estimator_seconds"],
                                    prediction["estimator_work"], prediction["verification_work"])
    require(prediction == expected, "Unchanged fixed verification prediction arithmetic and fields")
    actual = _seconds(actual_seconds, positive=True)
    return {"version": VERSION, "prediction": copy.deepcopy(prediction), "actual_seconds": actual,
            "actual_over_prediction": actual / prediction["predicted_seconds"],
            "passed": actual <= prediction["predicted_seconds"],
            "scope": "Verification only; equality passes without epsilon and verification does not refit the estimator"}


def _fit_mapping(values, name):
    require(isinstance(values, Mapping) and set(values) == set(FIT_ORDER), "Exactly all twelve " + name)


def project_calibration(records_by_fit, calibration_work, *, loading_seconds_by_fit, warmup_seconds_by_fit,
                        preparation_parent_wall_seconds, pilot_nonblock_overhead_seconds):
    """Project 12 x 512 forwards using only each fit's complete estimator block.

    Warmup seconds are the paid sum of the two legacy parity-checked dialogues
    for each fit. The caller supplies non-block pilot overhead: parent wall
    time outside loading, warmups, estimator and verification blocks. It must
    account for the whole pilot without counting those disjoint intervals twice.
    Verification failures are reported and refuse admission, without changing
    any estimator coefficient or excluding any fit from the cost calculation.
    """
    _fit_mapping(records_by_fit, "verification records")
    _fit_mapping(loading_seconds_by_fit, "loading costs")
    _fit_mapping(warmup_seconds_by_fit, "two-warmup costs")
    target = _work(calibration_work, CALIBRATION_DIALOGUES)
    preparation = _seconds(preparation_parent_wall_seconds)
    overhead = _seconds(pilot_nonblock_overhead_seconds)
    fits, validations, loads, warmups, common_geometry = {}, {}, {}, {}, None
    for fit_id in FIT_ORDER:
        record = records_by_fit[fit_id]
        require(type(record) is dict, "Complete verification record mapping")
        checked = validate_verification(record["prediction"], record["actual_seconds"])
        require(record == checked and record["prediction"]["fit_id"] == fit_id,
                "Unchanged verification result bound to the matching fit")
        prediction = record["prediction"]
        geometry = (prediction["estimator_work"], prediction["verification_work"])
        require(common_geometry is None or geometry == common_geometry, "All twelve fits use identical sample work geometry")
        common_geometry = geometry
        ratios = _ratios(target, prediction["estimator_work"])
        fits[fit_id] = {
            "estimator_seconds": prediction["estimator_seconds"], "estimator_work": dict(prediction["estimator_work"]),
            "calibration_work": dict(target), "work_ratios": ratios, "maximum_work_ratio": max(ratios.values()),
            "safety_factor": SAFETY_FACTOR, "projected_seconds": _scaled_seconds(prediction["estimator_seconds"], ratios),
        }
        validations[fit_id] = copy.deepcopy(checked)
        loads[fit_id] = _seconds(loading_seconds_by_fit[fit_id])
        warmups[fit_id] = _seconds(warmup_seconds_by_fit[fit_id])
    try:
        components = {
            "projected_calibration_forward_seconds": math.fsum(item["projected_seconds"] for item in fits.values()),
            "loading_seconds": math.fsum(loads.values()), "warmup_seconds": math.fsum(warmups.values()),
            "preparation_parent_wall_seconds": preparation, "pilot_nonblock_overhead_seconds": overhead,
        }
        total = math.fsum(components.values())
    except OverflowError as error:
        raise ValueError("Finite full calibration projection") from error
    require(math.isfinite(total), "Finite full calibration projection")
    verified = all(item["passed"] for item in validations.values())
    return {
        "version": VERSION, "fit_order": list(FIT_ORDER), "fits": fits, "verification": validations,
        "loading_seconds_by_fit": loads, "warmup_seconds_by_fit": warmups,
        "components": components, "total_seconds": total, "threshold_seconds": THRESHOLD_SECONDS,
        "projection_within_threshold": total <= THRESHOLD_SECONDS,
        "all_verifications_passed": verified, "verifications_passed": sum(r["passed"] for r in validations.values()),
        "verifications_total": len(FIT_ORDER), "admitted": verified and total <= THRESHOLD_SECONDS,
        "projected_dialogue_forwards": len(FIT_ORDER) * CALIBRATION_DIALOGUES,
        "legacy_v1_admission_revised": False, "scientific_conditions_changed": False,
        "scope": "Fixed 2x estimator-only screening; all twelve verification blocks must pass. "
                 "The prior failed cost method is preserved; this is not a runtime guarantee or scientific result.",
    }
