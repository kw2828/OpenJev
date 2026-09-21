"""Pure complete-cohort reporting for the fixed output-temperature control.

No files, models, cohort selection or DEV-based fitting. The reader must first
authenticate all twelve completed fits, the exact 512-dialogue calibration
cohort, and the historical raw FAIL result. Small synthetic cohorts are valid
here. Calibration rows retain TRAIN identity throughout.
"""
from __future__ import annotations

import json
import math

import numpy as np

from openjev.research import dialogue_observation_metrics as original
from openjev.research.dialogue_temperature import fit_temperature, output_logs, validate

VERSION = "dialogue-calibration-metrics-v1"
ROUTES = ("raw", "normalized", "calibrated")
DECISION_CHECKS = (
    "unseen_macro_gain_1pp", "unseen_macro_strict_paired_wins",
    "seen_macro_deficit_at_most_1pp",
    "seen_assigned_retention_error_increase_at_most_half_pp",
    "unseen_assigned_retention_error_increase_at_most_half_pp",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _indices(indices, count):
    require(isinstance(indices, np.ndarray) and indices.dtype == np.int64
            and np.array_equal(indices, np.arange(count, dtype=np.int64)),
            "Saved canonical row indices must be contiguous and complete")


def _top_mask(logs):
    return logs == logs.max(axis=1, keepdims=True)


def score_transformed(logs64, dev_rows, *, row_indices, raw_logs):
    """Preserve original group definitions while scoring float64 outputs directly.

    This scorer never clips, floors, renormalizes or casts transformed logs to
    float32. The reference is the original saved float32 output on the exact
    same canonical ledger. Every top tie and first canonical argmax must match.
    """
    layout = original._layout(dev_rows)
    original._validate(raw_logs, row_indices, layout)
    _indices(row_indices, len(dev_rows))
    mask = layout["mask"]
    require(isinstance(logs64, np.ndarray) and logs64.dtype == np.float64 and logs64.shape == mask.shape,
            "Transformed float64 log matrix matches complete canonical candidate support")
    require(np.isfinite(logs64[mask]).all() and np.isneginf(logs64[~mask]).all(),
            "Finite transformed support and exact negative-infinity padding")
    probabilities = np.exp(logs64)
    mass_errors = np.abs(probabilities.sum(1) - 1.0)
    require(float(logs64[mask].max()) <= math.log1p(original.TOLERANCE)
            and bool(np.all(mass_errors <= original.TOLERANCE)), "Transformed probability mass")
    choices = logs64.argmax(1)
    require(np.array_equal(choices, raw_logs.argmax(1)), "Exact canonical first argmax unchanged")
    top = _top_mask(logs64)
    require(np.array_equal(top, _top_mask(raw_logs)), "Complete top-tie masks unchanged")
    labels = layout["labels"]
    residual = probabilities.copy()
    residual[np.arange(len(labels)), labels] -= 1.0
    losses = {"nll": -logs64[np.arange(len(labels)), labels], "brier": np.square(residual).sum(1)}
    require(all(np.isfinite(values).all() for values in losses.values()), "Finite transformed proper scores")
    validation = {
        "rows": len(dev_rows), "supported_candidate_positions": int(mask.sum()),
        "maximum_mass_error": float(mass_errors.max()), "tolerance": original.TOLERANCE,
        "exact_top1_tie_rows": int((top.sum(1) > 1).sum()),
        "float64_exponent_underflow_positions": int((probabilities[mask] == 0).sum()),
        "dtype": "float64", "choices_unchanged": True, "top_tie_masks_unchanged": True,
        "policy": "Direct transformed float64 logs and candidate-sum Brier; no clipping, floor or repair",
    }
    return {**original._grouped(layout, choices, losses), "validation": validation}


def _calibration_targets(logs, indices, rows):
    """Validate TRAIN evaluator identity without adapting it to a DEV ledger."""
    require(type(rows) is list and rows, "Nonempty canonical calibration ledger")
    _indices(indices, len(rows))
    require(isinstance(logs, np.ndarray) and logs.dtype == np.float32
            and logs.shape == (len(rows), original.MAX_CANDIDATES), "Raw saved calibration float32 log matrix")
    masks = np.zeros(logs.shape, bool)
    labels, source_keys, endpoints, schemas, query_positions = [], set(), set(), {}, {}
    source_counts, query_indices, position_queries, public_turns = {}, {}, {}, {}
    dialogues, services = set(), set()
    for i, row in enumerate(rows):
        require(type(row) is dict and type(row["row_index"]) is int and row["row_index"] == i
                and row["split"] == row["source_split"] == "train" and row["analysis_role"] == "calibration"
                and row["unseen"] is False, "Calibration retains canonical TRAIN source and separate analysis role")
        did, service, slot, query_id = (row[key] for key in ("dialogue_id", "service", "slot", "query_id"))
        require(all(type(value) is str and value for value in (did, service, slot, query_id))
                and json.loads(query_id) == [service, slot], "Calibration dialogue and public query identity")
        require(all(type(row[key]) is int and row[key] >= 0 for key in
                    ("source_row_index", "time", "turn_index", "query_index", "query_position")),
                "Calibration source and public chronology indices")
        source_key, endpoint = (did, row["source_row_index"]), (did, row["time"], row["query_index"])
        require(source_key not in source_keys and endpoint not in endpoints, "Distinct calibration endpoint identity")
        require(row["source_row_index"] == source_counts.get(did, 0), "Contiguous calibration source-row order")
        source_counts[did] = source_counts.get(did, 0) + 1
        source_keys.add(source_key)
        endpoints.add(endpoint)
        ids, values = row["candidate_ids"], row["candidate_values"]
        require(type(ids) is list and type(values) is list and 3 <= len(ids) <= original.MAX_CANDIDATES
                and len(values) == len(ids) and all(type(cid) is str for cid in ids)
                and len(set(ids)) == len(ids) and ids.count(original.NONE) == ids.count(original.DONTCARE) == 1,
                "Complete distinct canonical calibration candidates")
        for cid, value in zip(ids, values, strict=True):
            require(value is None if cid in (original.NONE, original.DONTCARE) else
                    type(value) is str and bool(value.strip()) and cid == "value:" + value,
                    "Calibration candidate identity and value binding")
        schema = (query_id, service, slot, tuple(ids), tuple(values))
        require(schemas.setdefault(row["query_index"], schema) == schema, "Stable calibration query schema")
        require(query_indices.setdefault(query_id, row["query_index"]) == row["query_index"],
                "Canonical calibration query ID has one index")
        key = (did, row["query_position"])
        require(query_positions.setdefault(key, row["query_index"]) == row["query_index"],
                "Stable calibration query position")
        require(position_queries.setdefault((did, row["query_index"]), row["query_position"]) == row["query_position"],
                "One calibration query position per dialogue")
        require(public_turns.setdefault((did, row["time"]), row["turn_index"]) == row["turn_index"],
                "Consistent calibration public USER chronology")
        label = row["label_index"]
        require(type(label) is int and 0 <= label < len(ids) and row["label_id"] == ids[label],
                "Calibration target bound to canonical candidate ID")
        bin_name = row["bin"]
        require(bin_name in original.BINS
                and row["stratum"] == (bin_name if bin_name in original.STRATA[:2] else "changed")
                and type(row["dontcare"]) is bool and row["dontcare"] == (ids[label] == original.DONTCARE),
                "Original calibration transition and DONTCARE metadata")
        masks[i, :len(ids)] = True
        labels.append(label)
        dialogues.add(did)
        services.add(service)
    require(np.isfinite(logs[masks]).all() and np.isneginf(logs[~masks]).all(),
            "Saved calibration finite support and exact padding")
    targets = np.asarray(labels, dtype=np.int64)
    values, _, support = validate(logs, targets)
    require(np.array_equal(support, masks), "Calibration support matches canonical candidates")
    return targets, {
        "rows": len(rows), "dialogues": len(dialogues), "services": sorted(services),
        "source_split": "train", "analysis_role": "calibration",
        "supported_candidate_positions": int(masks.sum()),
        "maximum_mass_error": float(np.abs(np.exp(values).sum(1) - 1).max()),
        "scope": "Canonical endpoint validation only; exact 512-dialogue authenticated membership required by reader",
    }


def evaluate_fit(raw_dev_logs, dev_indices, dev_rows, calibration_logs, calibration_indices, calibration_rows):
    """Fit on calibration endpoints alone, then score all three fixed DEV routes."""
    targets, calibration_validation = _calibration_targets(calibration_logs, calibration_indices, calibration_rows)
    raw = original.score_panels(raw_dev_logs, dev_rows, row_indices=dev_indices)
    fitted = fit_temperature(calibration_logs, targets)
    normalized = output_logs(raw_dev_logs, 1.0)
    calibrated = output_logs(raw_dev_logs, fitted["beta"])
    # Validate the calibration transformation too: the scalar must preserve the
    # entire tied-top set in both source cohorts, not just the winning index.
    calibration_transformed = output_logs(calibration_logs, fitted["beta"])
    require(np.array_equal(_top_mask(calibration_logs), _top_mask(calibration_transformed))
            and np.array_equal(calibration_logs.argmax(1), calibration_transformed.argmax(1)),
            "Calibration first argmax and complete top-tie masks unchanged")
    return {"raw": raw,
            "normalized": score_transformed(normalized, dev_rows, row_indices=dev_indices, raw_logs=raw_dev_logs),
            "calibrated": score_transformed(calibrated, dev_rows, row_indices=dev_indices, raw_logs=raw_dev_logs),
            "temperature": fitted, "calibration_validation": calibration_validation}


def _decisions(value):
    if not isinstance(value, dict):
        return value
    return {key: _decisions(item) for key, item in value.items() if key not in ("nll", "brier", "validation")}


def calibration_criteria(raw_fits, normalized_fits, calibrated_fits):
    """Eleven conditions for this control, separately retaining all raw checks."""
    routes = dict(zip(ROUTES, (raw_fits, normalized_fits, calibrated_fits), strict=True))
    original_gates = {}
    for name, fits in routes.items():
        original._check_fits(fits)
        original_gates[name] = original.criteria(fits)
        for fit_id in raw_fits:
            require(_decisions(fits[fit_id]) == _decisions(raw_fits[fit_id]),
                    "All route decision metrics and subgroup supports must remain identical")
    raw_checks = {check["name"]: check for check in original_gates["raw"]["checks"]}
    for name in ("normalized", "calibrated"):
        checks = {check["name"]: check for check in original_gates[name]["checks"]}
        require(all(checks[key] == raw_checks[key] for key in DECISION_CHECKS),
                "All five original decision conditions unchanged")
    checks = list(original_gates["calibrated"]["checks"])
    for arm, metric, strict in (("trainable_numbers", "nll", True), ("trainable_numbers", "brier", False),
                                ("frozen_numbers", "nll", False), ("frozen_numbers", "brier", False)):
        after = [calibrated_fits[f"{arm}-{seed}"]["panels"]["unseen"]["micro"][metric] for seed in original.SEEDS]
        before = [raw_fits[f"{arm}-{seed}"]["panels"]["unseen"]["micro"][metric] for seed in original.SEEDS]
        present = all(type(value) in (int, float) and math.isfinite(value) for value in after + before)
        a, b = (original._mean(after), original._mean(before)) if present else (None, None)
        checks.append({
            "name": f"{arm}_unseen_micro_{metric}_" + ("improves_own_raw" if strict else "nonworse_own_raw"),
            "passed": bool(present and (a < b if strict else a <= b)),
            "paired_seed_values": [x - y for x, y in zip(after, before, strict=True)] if present else None,
            "mean": a - b if present else None, "calibrated_mean": a, "raw_mean": b,
            "threshold": 0.0, "comparison": "lt" if strict else "le",
            "arithmetic": "float64 arm means, no epsilon",
        })
    require(len(checks) == 11, "Exactly eleven calibration-control conditions")
    return {"passed": all(check["passed"] for check in checks), "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": 11, "checks": checks, "original_conditions": original_gates,
            "decision_conditions_unchanged": True, "complete_fit_membership": True,
            "raw_result_revised": False,
            "scope": "Separate development control; raw historical gate is retained, never reversed. "
                     "Reader must authenticate actual raw FAIL with six of seven passed."}


def summarize(fit_results):
    """All twelve fits, full factorial trees and paired changes for every arm."""
    require(type(fit_results) is dict and set(fit_results) == original._fit_names(),
            "Exactly all four arms by three seeds required")
    routes = {name: {fit_id: result[name] for fit_id, result in fit_results.items()} for name in ROUTES}
    gate = calibration_criteria(routes["raw"], routes["normalized"], routes["calibrated"])
    factorial = {name: original.factorial_summary(fits) for name, fits in routes.items()}
    trees = {name: {fit_id: original._metric_tree(fit) for fit_id, fit in fits.items()}
             for name, fits in routes.items()}
    changes = {}
    for after, before in (("normalized", "raw"), ("calibrated", "raw"), ("calibrated", "normalized")):
        arms = {}
        for arm in original.ARMS:
            seeds = {str(seed): original._combine([trees[after][f"{arm}-{seed}"], trees[before][f"{arm}-{seed}"]],
                                                  lambda values: values[0] - values[1])
                     for seed in original.SEEDS}
            arms[arm] = {"seeds": seeds, "mean": original._combine(list(seeds.values()), original._mean)}
        changes[f"{after}_minus_{before}"] = {"definition": f"{after} minus {before}", "arms": arms}
    return {"version": VERSION, "fits": fit_results, "factorial": factorial,
            "paired_changes": changes, "continuation": gate,
            "scope": "All three optimization seeds exposed; equal-seed descriptive means, no confidence intervals. "
                     "DEV informed this direction; no untouched-confirmation or architecture claim."}
