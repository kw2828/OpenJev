"""Posthoc saved-output diagnosis of conditional decision errors; no inference.

The closed study's result cannot change. This describes prior-choice errors,
training support and additive NLL contributions before a new method is chosen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import report_dialogue_conditional as report

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"runs/dialogue-conditional-v1/training-01"
RUN_PIN = "df3c172bae7163292b54cdd9a3b1d6e3bb5a07acb49b62be4f0c0dfc68efbe75"
ANALYSIS = ROOT/"output/dialogue-conditional-v1/analysis-01/summary.json"
ANALYSIS_PIN = "fb065bcc05550734f7fbe8a24def71b4b69ae5288c06dabbe1caddf39fb50e20"
VALUES = report.VALUES
require = report.require


def value_class(candidate_id, value, boolean_slot):
    if candidate_id == "reserved:NOT_MENTIONED":
        return "none"
    if candidate_id == "reserved:DONTCARE":
        return "dontcare"
    return value.strip().casefold() if boolean_slot else "other"


def classes_for_rows(rows, queries):
    result = []
    for row in rows:
        q = queries[row["query_index"]]
        require(q["query_id"] == row["query_id"] and len(q["candidate_ids"]) == row["candidate_count"],
                "Schema identity")
        classes = [value_class(cid, value, q["boolean_slot"])
                   for cid, value in zip(q["candidate_ids"], q["candidate_values"], strict=True)]
        require(all(value in VALUES for value in classes), "Unknown candidate type")
        require(classes[row["current_label_index"]] == row["current_value_group"], "Target type identity")
        result.append(classes)
    return result


def diagnostic_groups(rows):
    groups = report.groups(rows)
    for panel, unseen in (("seen", False), ("unseen", True)):
        for transition in report.BINS:
            for value in VALUES:
                groups[f"{panel}/transition_value/{transition}/{value}"] = np.asarray([
                    r["unseen"] == unseen and r["derived_bin"] == transition and r["current_value_group"] == value
                    for r in rows], dtype=bool)
    return groups


def row_quantities(rows, classes, packet):
    probabilities, _ = report.validate_predictions(rows, **packet)
    logs = packet["log_probs"].astype(np.float64)
    target = np.asarray([r["current_label_index"] for r in rows])
    prior = np.asarray([r["previous_current_index"] for r in rows])
    choice = logs.argmax(axis=1)
    changed = target != prior
    correct = choice == target
    prior_error = changed & (choice == prior)
    other_wrong = ~correct & ~prior_error
    require(np.all(correct.astype(int)+prior_error+other_wrong == 1), "Outcome partition")
    index = np.arange(len(rows))
    other = probabilities.copy()
    other[index, target] = -1.
    other[index, prior] = -1.
    return {"changed": changed, "correct": correct, "prior_error": prior_error, "other_wrong": other_wrong,
            "target_probability": probabilities[index, target], "prior_probability": probabilities[index, prior],
            "strongest_other_probability": other.max(axis=1), "target_nll": -logs[index, target],
            "target_minus_prior_log_probability": logs[index, target]-logs[index, prior],
            "top1_tied": (logs == logs.max(axis=1, keepdims=True)).sum(axis=1) > 1,
            "choice_classes": np.asarray([c[int(j)] for c, j in zip(classes, choice, strict=True)])}


def describe(rows, quantities, selected):
    result = report.denominators(rows, selected)
    changed = selected & quantities["changed"]
    result.update(changed_rows=int(changed.sum()), retained_rows=int((selected & ~quantities["changed"]).sum()),
                  prediction_events=result["rows"])
    for key in ("correct", "prior_error", "other_wrong", "top1_tied"):
        result[key+"_count"] = int(quantities[key][selected].sum())
    for key in ("target_probability", "prior_probability", "strongest_other_probability", "target_nll"):
        result["mean_"+key] = float(quantities[key][selected].mean()) if result["rows"] else None
    result["changed_mean_target_minus_prior_log_probability"] = (
        float(quantities["target_minus_prior_log_probability"][changed].mean()) if changed.any() else None)
    wrong = selected & ~quantities["correct"]
    result["wrong_choice_classes"] = {value: int((wrong & (quantities["choice_classes"] == value)).sum()) for value in VALUES}
    require(result["correct_count"]+result["prior_error_count"]+result["other_wrong_count"] == result["rows"],
            "Group outcome partition")
    require(sum(result["wrong_choice_classes"].values()) == result["rows"]-result["correct_count"], "Wrong type partition")
    return result


def training_support(rows, classes):
    require(all(r["split"] == "train" and r["admission"] == "admitted" for r in rows), "Admitted training rows only")
    result = {"rows": len(rows), "transition_value": {}, "previous_to_current_value": {}}
    for transition in report.BINS:
        for value in VALUES:
            selected = np.asarray([r["derived_bin"] == transition and r["current_value_group"] == value for r in rows])
            result["transition_value"][transition+"/"+value] = report.denominators(rows, selected)
    for previous in VALUES:
        for current in VALUES:
            selected = np.asarray([c[r["previous_current_index"]] == previous and r["current_value_group"] == current
                                   for r, c in zip(rows, classes, strict=True)])
            result["previous_to_current_value"][previous+"/"+current] = report.denominators(rows, selected)
    require(sum(c["rows"] for c in result["transition_value"].values()) == len(rows)
            == sum(c["rows"] for c in result["previous_to_current_value"].values()), "Training support partition")
    return result


def contributions(rows, quantities_by_fit, published):
    primary = np.asarray([r["unseen"] and r["derived_bin"] in report.BINS[2:] for r in rows])
    n = int(primary.sum())
    require(n > 0 and n == published["primary"]["rows"], "Primary support")
    result = {"primary_rows": n, "groups": {}}
    for value in VALUES:
        selected = primary & np.asarray([r["current_value_group"] == value for r in rows])
        group = {**report.denominators(rows, selected), "seeds": []}
        for seed in report.SEEDS:
            slot = quantities_by_fit[f"slot-{seed}"]["target_nll"][selected]
            candidate = quantities_by_fit[f"candidate-{seed}"]["target_nll"][selected]
            group["seeds"].append({"seed": seed, "slot_global_denominator_nll_contribution": float(slot.sum()/n),
                "candidate_global_denominator_nll_contribution": float(candidate.sum()/n),
                "candidate_minus_slot_global_denominator_contribution": float((candidate-slot).sum()/n),
                "conditional_mean_difference": float((candidate-slot).mean()) if selected.any() else None})
        group["mean_candidate_minus_slot_global_denominator_contribution"] = float(np.mean([
            p["candidate_minus_slot_global_denominator_contribution"] for p in group["seeds"]]))
        result["groups"][value] = group
    reconstructed = []
    for i, seed in enumerate(report.SEEDS):
        delta = sum(g["seeds"][i]["candidate_minus_slot_global_denominator_contribution"] for g in result["groups"].values())
        expected = published["primary"]["paired"][i]
        require(expected["seed"] == seed and abs(delta-expected["candidate_minus_slot_nll"]) < 1e-12, "Paired NLL reconstruction")
        reconstructed.append(delta)
    require(abs(float(np.mean(reconstructed))-published["primary"]["mean_candidate_minus_slot_nll"]) < 1e-12,
            "Pooled NLL reconstruction")
    result["reconstructed_paired_differences"] = reconstructed
    return result


def analyze(train, dev, queries, predictions, published):
    require(set(predictions) == set(report.FIT_ORDER), "All nine fits required")
    dev_classes = classes_for_rows(dev, queries)
    quantities = {name: row_quantities(dev, dev_classes, packet) for name, packet in predictions.items()}
    group_masks = diagnostic_groups(dev)
    result = {"kind": "Posthoc descriptive diagnosis; no new model or rescored decision",
              "training_support": training_support(train, classes_for_rows(train, queries)),
              "fits": {name: {group: describe(dev, values, mask) for group, mask in group_masks.items()}
                       for name, values in quantities.items()},
              "primary_decomposition": contributions(dev, quantities, published),
              "groups": {group: {**report.denominators(dev, mask), "prediction_events_across_nine_fits": 9*int(mask.sum())}
                         for group, mask in group_masks.items()},
              "original_continuation_allowed": published["continuation_allowed"],
              "new_primary_or_continuation_rule": False,
              "interpretation_limits": ["Predictions at the previous value are output reliance, not proven recurrent inertia; scorer has no recurrent carry.",
                  "Correct and prior coincide on retained rows; prior_error counts changed rows only.",
                  "Probabilities do not establish information absence, calibrated uncertainty or causal mechanisms.",
                  "All nine fits share the same rows. Prediction events are not independent data points.",
                  "Analysis chosen after the original negative result; cannot rescue its failed rule."]}
    return result


def execute(out, source_pin, tests_pin, protocol_pin):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    paths = {"source": Path(__file__), "tests": ROOT/"tests/test_diagnose_dialogue_conditional.py",
             "protocol": ROOT/"research/dialogue-conditional-error-protocol.md"}
    pins = {"source": source_pin, "tests": tests_pin, "protocol": protocol_pin}
    try:
        require(all(report.digest(paths[k]) == v for k, v in pins.items()), "External diagnostic source pins")
        require(report.digest(ANALYSIS) == ANALYSIS_PIN, "Original analysis identity")
        published = report.read_json(ANALYSIS)
        _, plan, _, dev, _ = report.authenticate_run(RUN, RUN_PIN)
        train, joined_dev = report.read_ledger(Path(plan["prepared_path"]), plan)
        require(dev == joined_dev, "Canonical development join")
        queries = report.read_json(Path(plan["prepared_path"])/"catalog.json")["queries"]
        predictions = {name: report.load_npz(RUN/"fits"/name/"dev-predictions.npz", ("log_probs", "row_indices"))
                       for name in report.FIT_ORDER}
        result = analyze(train, dev, queries, predictions, published)
        require(all(report.digest(paths[k]) == v for k, v in pins.items()), "Diagnostic source drift")
        (out/"summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
        receipt = {"status": "completed", "source_pins": pins, "input_run_completed_sha256": RUN_PIN,
                   "input_analysis_sha256": ANALYSIS_PIN, "summary_sha256": report.digest(out/"summary.json"),
                   "wall_seconds": time.monotonic()-start, "model_calls": 0, "training_calls": 0,
                   "original_continuation_allowed": result["original_continuation_allowed"],
                   "posthoc": True, "comparison_result_changed": False}
        (out/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        return receipt
    except BaseException as error:
        try:
            (out/"failed.json").write_text(json.dumps({"status": "failed", "error": repr(error)})+"\n")
        except BaseException as secondary:  # noqa: BLE001 - retain original error
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note("Diagnostic failure receipt write failed: " + repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tests-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.out, args.source_sha256, args.tests_sha256, args.protocol_sha256)))
