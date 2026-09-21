"""Pure marginal geometry diagnostics for the fixed runtime sample.

Accepts saved per-dialogue work profiles, never actors, tokens or labels. This
report cannot admit or reject a runtime experiment. The caller authenticates
the profiles and independently validates the fixed sample selection.
"""
from __future__ import annotations

from collections.abc import Mapping

VERSION = "dialogue-runtime-geometry-v1"
SETTINGS = ("chunk_tokens", "chunk_batch_size")
FIELDS = (
    "unique_texts", "input_texts", "content_tokens", "chunks", "encoder_sequences",
    "special_token_positions", "valid_token_positions", "padded_token_positions",
    "padding_token_positions", "padded_attention_positions", "encoder_calls",
    "overlength_texts_chunked", "truncated_tokens", "max_chunk_tokens_with_special",
    "public_user_turns", "queries", "schema_text_occurrences", "candidate_text_occurrences",
    "max_candidates", "real_question_updates", "real_candidate_updates",
    "padded_candidate_positions", "lexical_scalars", "lexical_bytes",
    "max_content_tokens_per_text",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _profiles(records, split):
    require(type(records) is list and records, "Nonempty saved profile list")
    result = {}
    for record in records:
        require(isinstance(record, Mapping)
                and set(record) == {"split", "dialogue_id", "work"}
                and record["split"] == split, "Exact split-qualified work profile")
        did, work = record["dialogue_id"], record["work"]
        require(type(did) is str and did and did not in result, "Distinct dialogue IDs within split")
        require(isinstance(work, Mapping) and set(work) == set(FIELDS) | set(SETTINGS),
                "Complete declared saved work fields")
        require(all(type(value) is int and value >= 0 for value in work.values()),
                "Nonnegative integer geometry without Boolean or floating values")
        require(all(work[key] > 0 for key in SETTINGS), "Positive chunk settings")
        result[did] = work
    return result


def _ids(value, name):
    require(type(value) is list and value and all(type(did) is str and did for did in value)
            and len(value) == len(set(value)), "Distinct nonempty " + name + " IDs")
    return list(value)


def _ranges(indexed, ids, split):
    fields = {}
    for key in FIELDS:
        values = [indexed[did][key] for did in ids]
        low, high = min(values), max(values)
        fields[key] = {"min": low, "max": high, "range": high - low}
    return {"source_split": split, "dialogue_count": len(ids), "dialogue_ids": list(ids), "fields": fields}


def report_geometry(devprofiles, calprofiles, sample):
    """Compare estimator, verification and calibration marginal field ranges.

    ``devprofiles`` and ``calprofiles`` are the ``profiles`` lists produced by
    aggregate_work_profiles, with DEV and TRAIN identities respectively.
    ``sample`` is the runtime selector's mapping containing estimator and
    verification ``dialogue_ids`` plus two excluded ``warmup_ids``. Tiny
    synthetic lists are supported; the caller owns the 2363/512/64/64 counts.

    Uncovered means strictly outside a measured field's closed min/max range.
    This is descriptive marginal coverage, not joint-shape coverage, an upper
    runtime bound, or an admission condition. Outliers never raise an error.
    """
    dev, calibration = _profiles(devprofiles, "dev"), _profiles(calprofiles, "train")
    require(isinstance(sample, Mapping)
            and isinstance(sample.get("estimator"), Mapping)
            and isinstance(sample.get("verification"), Mapping), "Runtime sample mapping")
    estimator = _ids(sample["estimator"]["dialogue_ids"], "estimator")
    verification = _ids(sample["verification"]["dialogue_ids"], "verification")
    warmups = _ids(sample["warmup_ids"], "warmup")
    require(len(warmups) == 2 and not set(estimator) & set(verification)
            and not set(warmups) & (set(estimator) | set(verification)), "Disjoint sample and two legacy warmups")
    require(set(estimator + verification + warmups) <= set(dev), "All sampled IDs belong to supplied DEV")
    settings = {key: next(iter(dev.values()))[key] for key in SETTINGS}
    require(all(all(work[key] == value for key, value in settings.items())
                for work in (*dev.values(), *calibration.values())), "Shared declared chunk settings")
    groups = {"estimator": estimator, "verification": verification, "timed_union": estimator + verification}
    ranges = {name: _ranges(dev, ids, "dev") for name, ids in groups.items()}
    calibration_ids = list(calibration)
    ranges["calibration"] = _ranges(calibration, calibration_ids, "train")
    uncovered = {}
    for name, ids in groups.items():
        by_field, outside = {}, set()
        for key in FIELDS:
            bounds = ranges[name]["fields"][key]
            below = [did for did in calibration_ids if calibration[did][key] < bounds["min"]]
            above = [did for did in calibration_ids if calibration[did][key] > bounds["max"]]
            outside.update(below)
            outside.update(above)
            by_field[key] = {"below_min_count": len(below), "below_min_dialogue_ids": below,
                             "above_max_count": len(above), "above_max_dialogue_ids": above}
        uncovered[name] = {"reference_dialogue_count": len(ids),
                           "calibration_dialogue_count": len(calibration_ids),
                           "outside_any_field_count": len(outside),
                           "outside_any_field_dialogue_ids": [did for did in calibration_ids if did in outside],
                           "by_field": by_field}
    return {
        "version": VERSION, "diagnostic_only": True, "used_for_admission": False,
        "settings": settings, "fields": list(FIELDS),
        "source_profile_counts": {"dev": len(dev), "calibration": len(calibration),
                                  "warmups_excluded_from_measured_ranges": len(warmups)},
        "cohorts": ranges, "uncovered_calibration": uncovered,
        "uncovered_definition": "Strictly below the minimum or above the maximum of a measured marginal field; endpoints are included.",
        "range_definition": "Per-dialogue maximum minus minimum; neither a cohort total nor an elapsed-time estimate.",
        "joint_batch_coverage": {
            "status": "unknown",
            "reason": "Saved profiles omit ordered encoder batch (sequence_count, padded_width) pairs and token-length layouts. Aggregate counts and maxima cannot recover them.",
        },
        "scope": "Public work metadata only. Ranges do not establish joint-shape coverage, independent observations, runtime bounds or model quality. Split-qualified identities are separate. No change to sample selection or admission.",
    }
