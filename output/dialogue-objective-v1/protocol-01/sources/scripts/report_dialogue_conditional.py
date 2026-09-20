"""Saved-output analysis for the privileged conditional-observation diagnostic.

This module imports no model, training code or neural runtime. The metric
functions consume complete aligned final predictions, never generate them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = (5301, 5302, 5303)
MODES = ("mean", "slot", "candidate")
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
VALUES = ("none", "true", "false", "dontcare", "other")
MAX_CANDIDATES = 12
TOLERANCE = 2e-6
ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-conditional-training-v1"
FIT_ORDER = [f"{mode}-{seed}" for seed, modes in zip(SEEDS, (MODES, ("slot", "candidate", "mean"),
                    ("candidate", "mean", "slot")), strict=True) for mode in modes]
SOURCE_NAMES = {
    "research/dialogue-conditional-observation-design.md", "research/dialogue-conditional-preparation-protocol.md",
    "scripts/prepare_dialogue_conditional.py", "scripts/prepare_dialogue_tokens.py", "scripts/study_dialogue_copy.py",
    "scripts/study_dialogue_memory.py", "scripts/study_dialogue_tokens.py", "src/openjev/research/dialogue_copy_features.py",
    "src/openjev/research/dialogue_state_data.py", "tests/test_prepare_dialogue_conditional.py",
    "src/openjev/research/dialogue_conditional_observation.py", "tests/test_dialogue_conditional_observation.py",
    "src/openjev/research/dialogue_copy_memory.py", "scripts/study_dialogue_conditional.py",
    "tests/test_study_dialogue_conditional.py", "scripts/report_dialogue_conditional.py",
    "tests/test_report_dialogue_conditional.py", "research/dialogue-conditional-training-protocol.md",
}
FEATURE_PARENTS = {"pooled": "runs/sgd-state-v1/features-02", "lexical": "runs/dialogue-copy-v1/lexical-01",
                   "tokens": "runs/dialogue-token-v1/features-01"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_rows(rows):
    require(type(rows) is list and rows, "Nonempty admitted development rows required")
    identities = []
    for row in rows:
        require(row["split"] == "dev" and row["admission"] == "admitted", "Wrong analysis cohort")
        require(type(row["row_index"]) is int and row["row_index"] >= 0, "Row identity")
        require(type(row["candidate_count"]) is int and 3 <= row["candidate_count"] <= MAX_CANDIDATES,
                "Candidate count")
        require(all(type(row[k]) is int and 0 <= row[k] < row["candidate_count"]
                    for k in ("current_label_index", "previous_current_index")), "Label range")
        require(type(row["unseen"]) is bool and row["derived_bin"] in BINS
                and row["current_value_group"] in VALUES, "Evaluator group")
        require(all(type(row[k]) is str and row[k] for k in ("dialogue_id", "query_id")), "Group identities")
        previous, current = row["previous_candidate_id"], row["current_candidate_id"]
        require(all(type(x) is str and x for x in (previous, current)), "Canonical label identities")
        expected = ("unmentioned_retention" if previous == current == "reserved:NOT_MENTIONED"
                    else "assigned_retention" if previous == current
                    else "first_assignment" if previous == "reserved:NOT_MENTIONED"
                    else "clear" if current == "reserved:NOT_MENTIONED" else "revision")
        require(row["derived_bin"] == expected, "Transition identity")
        require((previous == current) == (row["previous_current_index"] == row["current_label_index"]),
                "Canonical previous-index mapping")
        identities.append(row["row_index"])
    require(identities == sorted(set(identities)), "Duplicate or noncanonical prediction rows")
    return np.asarray(identities, dtype=np.int64)


def validate_predictions(rows, log_probs, row_indices):
    expected = validate_rows(rows)
    require(isinstance(row_indices, np.ndarray) and row_indices.dtype == np.int64
            and np.array_equal(row_indices, expected), "Prediction row identity/order")
    require(isinstance(log_probs, np.ndarray) and log_probs.dtype == np.float32
            and log_probs.shape == (len(rows), MAX_CANDIDATES), "Prediction dtype/shape")
    mask = np.arange(MAX_CANDIDATES)[None, :] < np.asarray([r["candidate_count"] for r in rows])[:, None]
    require(np.isfinite(log_probs[mask]).all() and (log_probs[mask] <= 0).all()
            and np.isneginf(log_probs[~mask]).all(), "Invalid supported/masked log probabilities")
    probabilities = np.exp(log_probs.astype(np.float64))
    errors = np.abs(probabilities.sum(axis=1) - 1.)
    require((errors <= TOLERANCE).all(), "Unnormalized raw saved probabilities")
    return probabilities, float(errors.max())


def groups(rows):
    unseen = np.asarray([r["unseen"] for r in rows], dtype=bool)
    bins = np.asarray([r["derived_bin"] for r in rows])
    values = np.asarray([r["current_value_group"] for r in rows])
    changed = np.isin(bins, BINS[2:])
    named = {"all": np.ones(len(rows), dtype=bool), "changed": changed, "retained": ~changed}
    named.update({"transition/" + name: bins == name for name in BINS})
    for name in VALUES:
        selected = values == name
        named["value/" + name] = selected
        named["value/" + name + "/changed"] = selected & changed
        named["value/" + name + "/retained"] = selected & ~changed
    return {panel + "/" + name: mask & population
            for panel, population in (("seen", ~unseen), ("unseen", unseen))
            for name, mask in named.items()}


def denominators(rows, selected):
    subset = [rows[int(i)] for i in np.flatnonzero(selected)]
    return {"rows": len(subset), "dialogues": len({r["dialogue_id"] for r in subset}),
            "distinct_schema_queries": len({r["query_id"] for r in subset}),
            "dialogue_query_streams": len({(r["dialogue_id"], r["query_id"]) for r in subset})}


def summarize_values(rows, selected, values):
    result = denominators(rows, selected)
    if not result["rows"]:
        result.update(row_mean=None, equal_dialogue_mean=None)
        return result
    by_dialogue = defaultdict(list)
    for i in np.flatnonzero(selected):
        by_dialogue[rows[int(i)]["dialogue_id"]].append(float(values[i]))
    result["row_mean"] = float(np.mean(values[selected], dtype=np.float64))
    result["equal_dialogue_mean"] = float(np.mean([np.mean(v, dtype=np.float64) for v in by_dialogue.values()]))
    return result


def fit_metrics(rows, log_probs, row_indices):
    probabilities, max_mass_error = validate_predictions(rows, log_probs, row_indices)
    targets = np.asarray([r["current_label_index"] for r in rows], dtype=np.int64)
    nll = -log_probs[np.arange(len(rows)), targets].astype(np.float64)
    correct = (log_probs.argmax(axis=1) == targets).astype(np.float64)
    residual = probabilities.copy()
    residual[np.arange(len(rows)), targets] -= 1.
    brier = np.square(residual).sum(axis=1)
    cells = {}
    for name, selected in groups(rows).items():
        cell = denominators(rows, selected)
        for metric, values in (("accuracy", correct), ("nll", nll), ("brier", brier)):
            stats = summarize_values(rows, selected, values)
            cell[metric] = stats["row_mean"]
            cell["equal_dialogue_" + metric] = stats["equal_dialogue_mean"]
        cells[name] = cell
    return {"rows": len(rows), "maximum_raw_probability_mass_error": max_mass_error, "cells": cells}


def reference_metrics(rows, row_indices, previous_indices, literal_indices):
    expected = validate_rows(rows)
    require(isinstance(row_indices, np.ndarray) and row_indices.dtype == np.int64
            and np.array_equal(row_indices, expected), "Reference row identity/order")
    previous = np.asarray([r["previous_current_index"] for r in rows], dtype=np.int64)
    targets = np.asarray([r["current_label_index"] for r in rows], dtype=np.int64)
    counts = np.asarray([r["candidate_count"] for r in rows], dtype=np.int64)
    for values in (previous_indices, literal_indices):
        require(isinstance(values, np.ndarray) and values.dtype == np.int64 and values.shape == expected.shape
                and ((values >= 0) & (values < counts)).all(), "Reference support/dtype/shape")
    require(np.array_equal(previous_indices, previous), "Privileged previous reference differs")
    result = {}
    for method, indices in (("previous_gold_carry", previous_indices), ("literal_carry", literal_indices)):
        correct = (indices == targets).astype(np.float64)
        cells = {}
        for name, selected in groups(rows).items():
            stats = summarize_values(rows, selected, correct)
            cells[name] = {k: v for k, v in stats.items() if k not in ("row_mean", "equal_dialogue_mean")}
            cells[name].update(accuracy=stats["row_mean"], equal_dialogue_accuracy=stats["equal_dialogue_mean"])
        result[method] = {"cells": cells, "scope": "Deterministic reference accuracy only; no finite probability loss assigned"}
    return result


def analyze(rows, predictions, references):
    """Compute every fixed output from all nine aligned final fits.

    predictions maps (method, seed) to {log_probs, row_indices}; references has
    row_indices, previous_indices and literal_indices. Completion/manifest
    authentication is a separate prerequisite for the command-line reader.
    """
    expected = {(method, seed) for method in MODES for seed in SEEDS}
    require(type(predictions) is dict and set(predictions) == expected, "All nine final fits required")
    metrics = {}
    for method, seed in sorted(expected, key=lambda x: (x[1], MODES.index(x[0]))):
        packet = predictions[(method, seed)]
        require(set(packet) == {"log_probs", "row_indices"}, "Prediction payload fields")
        metrics[f"{method}-{seed}"] = fit_metrics(rows, **packet)
    require(type(references) is dict and set(references) == {"row_indices", "previous_indices", "literal_indices"},
            "Reference payload fields")
    refs = reference_metrics(rows, **references)
    primary_cells = [metrics[f"slot-{seed}"]["cells"]["unseen/changed"] for seed in SEEDS]
    support = primary_cells[0]["rows"]
    require(all(c["rows"] == support for c in primary_cells), "Paired primary support")
    paired = []
    for seed in SEEDS:
        slot = metrics[f"slot-{seed}"]["cells"]["unseen/changed"]["nll"]
        candidate = metrics[f"candidate-{seed}"]["cells"]["unseen/changed"]["nll"]
        delta = candidate - slot if support else None
        paired.append({"seed": seed, "slot_nll": slot, "candidate_nll": candidate,
                       "candidate_minus_slot_nll": delta,
                       "relative_nll_change": delta / slot if support and slot > 0 else None})
    delta = float(np.mean([p["candidate_minus_slot_nll"] for p in paired])) if support else None
    slot_mean = float(np.mean([p["slot_nll"] for p in paired])) if support else None
    return {"fits": metrics, "references": refs,
            "primary": {"panel": "unseen/changed", "weighting": "row mean within fit, paired difference then three-seed mean",
                        **{k: primary_cells[0][k] for k in ("rows", "dialogues", "distinct_schema_queries", "dialogue_query_streams")},
                        "paired": paired, "mean_candidate_minus_slot_nll": delta,
                        "relative_mean_nll_change": delta / slot_mean if support and slot_mean > 0 else None,
                        "all_three_seed_differences_negative": bool(support and all(p["candidate_minus_slot_nll"] < 0 for p in paired)),
                        "primary_nll_rule_passed": bool(support and delta < 0 and all(p["candidate_minus_slot_nll"] < 0 for p in paired))},
            "architecture_advantage_established": False,
            "scope": "Privileged conditional diagnostic on exposed development data; technical admission must be verified separately; no significance, calibration, rollout or architecture-novelty claim"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON number: " + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    return decode(Path(path).read_text())


def bounded_path(root, name):
    root = Path(root).resolve()
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe relative path")
    target = root/name
    require(not target.is_symlink() and target.resolve().is_relative_to(root), "Escaping payload path")
    return target


def verify_file(path, item):
    require(set(item) == {"bytes", "sha256"} and type(item["bytes"]) is int and item["bytes"] >= 0,
            "Malformed file identity")
    require(Path(path).is_file() and Path(path).stat().st_size == item["bytes"]
            and digest(path) == item["sha256"], "Payload identity: " + str(path))


def verify_manifest(root, files, expected):
    require(type(files) is dict and set(files) == set(expected), "Manifest closure")
    actual = {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()}
    require(actual == set(expected) | {"completed.json"}, "Directory closure")
    for name, item in files.items():
        verify_file(bounded_path(root, name), item)


def read_ledger(prepared, plan):
    """Decode authenticated metadata and independently check canonical labels."""
    with (prepared/"rows.jsonl").open() as stream:
        ledger = [decode(line) for line in stream]
    catalog = read_json(prepared/"catalog.json")["queries"]
    require([q["query_index"] for q in catalog] == list(range(len(catalog))), "Catalog query ordering")
    train, dev = [], []
    for index, row in enumerate(ledger):
        require(type(row["row_index"]) is int and row["row_index"] == index, "Ledger ordering")
        qi = row["query_index"]
        require(type(qi) is int and 0 <= qi < len(catalog), "Query identity")
        q = catalog[qi]
        require(all(row[k] == q[k] for k in ("query_id", "split", "service", "slot")), "Row/catalog join")
        ids = q["candidate_ids"]
        label = row["current_label_index"]
        require(type(label) is int and 0 <= label < len(ids) and len(ids) == row["candidate_count"]
                and ids[label] == row["current_candidate_id"], "Canonical current target")
        require(row["split"] in ("train", "dev") and type(row["unseen"]) is bool,
                "Metadata split/panel")
        candidate_value = q["candidate_values"][label]
        value_group = ("none" if ids[label] == "reserved:NOT_MENTIONED"
                       else "dontcare" if ids[label] == "reserved:DONTCARE"
                       else candidate_value.strip().casefold() if q["boolean_slot"] else "other")
        require(row["current_value_group"] == value_group and row["boolean_slot"] == q["boolean_slot"],
                "Canonical value category")
        if row["admission"] != "admitted":
            continue
        pi, previous = row["previous_current_index"], row["previous_row_index"]
        require(type(pi) is int and 0 <= pi < len(ids) and ids[pi] == row["previous_candidate_id"],
                "Canonical prior mapping")
        require(type(previous) is int and 0 <= previous < index, "Prior row address")
        old = ledger[previous]
        require(all(old[k] == row[k] for k in ("split", "dialogue_id", "service", "slot"))
                and old["time"] == row["time"]-1 and old["current_candidate_id"] == ids[pi], "Adjacent prior identity")
        expected_bin = ("unmentioned_retention" if ids[pi] == ids[label] == "reserved:NOT_MENTIONED"
                        else "assigned_retention" if ids[pi] == ids[label]
                        else "first_assignment" if ids[pi] == "reserved:NOT_MENTIONED"
                        else "clear" if ids[label] == "reserved:NOT_MENTIONED" else "revision")
        require(row["derived_bin"] == expected_bin, "Canonical training/evaluation transition")
        (train if row["split"] == "train" else dev).append(row)
    require(len(train) == plan["admitted_train_rows"] and len(dev) == plan["admitted_dev_rows"], "Admitted counts")
    validate_rows(dev)
    return train, dev


def operation_counts(n, d, updates, evaluation_batches):
    return {"forward_attempted": updates+evaluation_batches, "forward_returned": updates+evaluation_batches,
            "backward_attempted": updates, "backward_returned": updates, "optimizer_attempted": updates,
            "optimizer_returned": updates, "training_rows": 20*n, "evaluation_rows": d}


def check_normalization(record, batches, rows, work):
    require(record["batches"] == batches and record["rows"] == rows
            and record["supported_candidates"] == work["supported_candidate_positions"]
            and record["masked_candidates"] == work["padded_candidate_positions"]-work["supported_candidate_positions"],
            "Normalization coverage")
    require(0 <= record["max_abs_mass_error"] <= TOLERANCE
            and math.isfinite(record["min_supported_log_prob"])
            and record["min_supported_log_prob"] <= record["max_supported_log_prob"] <= 0,
            "Normalization extrema")


def expected_geometry(rows, mode):
    b, c = len(rows), max(r["candidate_count"] for r in rows)
    lengths = [r["cache"]["token_stop"]-r["cache"]["token_start"] for r in rows]
    length = 0 if mode == "mean" else max(lengths)
    shapes = {"observation": [b, 384] if mode == "mean" else [b, length, 384],
              "query": [b, 384], "candidates": [b, c, 384], "candidate_mask": [b, c],
              "lexical": [b, c, 10], "previous_onehot": [b, c]}
    if mode != "mean":
        shapes.update(token_mask=[b, length], token_prior=[b, length])
    floats = sum(math.prod(shape) for key, shape in shapes.items() if key not in ("candidate_mask", "token_mask"))
    return {"rows": b, "supported_candidate_positions": sum(r["candidate_count"] for r in rows),
            "padded_candidate_positions": b*c, "supported_token_positions": 0 if mode == "mean" else sum(lengths),
            "padded_token_positions": b*length, "attention_score_positions": b*c*length,
            "token_key_positions": b*length, "attention_query_positions": 0 if mode == "mean" else b*c,
            "scorer_positions": b*c, "float_input_scalars": floats, "float_input_bytes": 4*floats,
            "boolean_input_bytes": b*c+b*length, "max_token_length": length, "max_candidate_count": c}, shapes


def check_journal(path, rows, orders, record):
    totals = defaultdict(int)
    normalization = defaultdict(int)
    max_error, min_log, max_log = 0., math.inf, -math.inf
    expected = ((epoch, start, order[start:start+256])
                for epoch, order in enumerate(orders) for start in range(0, len(rows), 256))
    count = 0
    with Path(path).open() as stream:
        for count, line in enumerate(stream, 1):
            step = next(expected, None)
            require(step is not None, "Extra optimizer journal update")
            epoch, start, indices = step
            batch = [rows[int(i)] for i in indices]
            event = decode(line)
            require(event["epoch"] == epoch and event["start"] == start and event["update"] == count
                    and event["row_indices"] == [r["row_index"] for r in batch], "Update row membership/order")
            require(math.isfinite(event["weighted_loss"]) and event["weighted_loss"] >= 0,
                    "Invalid weighted training loss")
            require(event["phase_seconds"] and all(math.isfinite(v) and v >= 0 for v in event["phase_seconds"].values()),
                    "Update timing")
            work = event["work"]
            expected_work, shapes = expected_geometry(batch, record["method"])
            require(work == expected_work and event["actor_shapes"] == shapes, "Update work and actor geometry")
            check_normalization(event["normalization"], 1, len(batch), work)
            for key, value in work.items():
                if not key.startswith("max_"):
                    totals[key] += value
            norm = event["normalization"]
            for key in ("batches", "rows", "supported_candidates", "masked_candidates"):
                normalization[key] += norm[key]
            max_error = max(max_error, norm["max_abs_mass_error"])
            min_log = min(min_log, norm["min_supported_log_prob"])
            max_log = max(max_log, norm["max_supported_log_prob"])
    require(next(expected, None) is None and count == record["counts"]["optimizer_returned"], "Missing optimizer updates")
    normalization.update(max_abs_mass_error=max_error, min_supported_log_prob=min_log, max_supported_log_prob=max_log)
    require(dict(totals) == record["training_work"] and dict(normalization) == record["training_normalization"],
            "Journal aggregate reconciliation")


def authenticate_run(run, pin):
    """Complete technical admission before reading any development prediction."""
    run = Path(run).resolve()
    require(digest(run/"completed.json") == pin, "External run completion identity")
    done = read_json(run/"completed.json")
    require(done["status"] == "completed" and done["version"] == VERSION and done["phase"] == "train"
            and done["no_retry"] is True and done["completed_fits"] == FIT_ORDER and done["expected_fits"] == FIT_ORDER
            and done["quality_metrics_computed"] is False and done["encoder_calls"] == 0
            and done["test_contents_accessed"] is False, "Nine-fit terminal scope")
    expected = {"started.json", "plan.json", "references.npz"} | {f"orders-{seed}.npy" for seed in SEEDS}
    expected |= {f"fits/{fit}/{name}" for fit in FIT_ORDER
                 for name in ("completed.json", "weights.pt", "updates.jsonl", "dev-predictions.npz")}
    verify_manifest(run, done["files"], expected)
    require(digest(run/"plan.json") == done["plan_sha256"], "Saved plan identity")
    plan = read_json(run/"plan.json")
    config = plan["config"]
    require(plan["version"] == VERSION and plan["expected_fits"] == FIT_ORDER
            and set(plan["source_sha256"]) == SOURCE_NAMES
            and config == {"methods": list(MODES), "seeds": list(SEEDS), "epochs": 20, "batch_size": 256,
                           "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.,
                           "input_dim": 384, "projection_dim": 64, "hidden_dim": 64,
                           "threads": 4, "interop_threads": 1, "dtype": "float32", "deterministic": True},
            "Fixed training recipe")
    require(plan["limits"] == {"wall_seconds": 3600., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
            and 0 < done["wall_seconds"] <= 3600 and 0 < done["process_lifetime_peak_rss_bytes"] <= 6*1024**3
            and sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) <= 512*1024**2, "Complete run cost limits")
    require(done["runtime"] == plan["runtime"] and done["source_sha256"] == plan["source_sha256"], "Source/runtime witness")
    request = read_json(run/"started.json")["request"]
    require(request["plan_sha256"] == done["plan_sha256"], "Run request plan pin")
    frozen = Path(request["plan"]).resolve().parent
    require(frozen.is_relative_to(ROOT.resolve()) and digest(frozen/"plan.json") == done["plan_sha256"], "Frozen plan binding")
    frozen_done = read_json(frozen/"completed.json")
    freeze_files = {"started.json", "plan.json", "prepared-completed.json"} | {f"orders-{s}.npy" for s in SEEDS}
    freeze_files |= {"sources/"+p for p in plan["source_sha256"]}
    require(frozen_done["status"] == "completed" and frozen_done["phase"] == "freeze"
            and frozen_done["plan_sha256"] == done["plan_sha256"], "Frozen completion")
    verify_manifest(frozen, frozen_done["files"], freeze_files)
    for name, value in plan["source_sha256"].items():
        require(digest(bounded_path(ROOT, name)) == value and digest(bounded_path(frozen/"sources", name)) == value,
                "Frozen source identity")
    prepared = Path(plan["prepared_path"]).resolve()
    require(prepared.is_relative_to(ROOT.resolve()) and digest(prepared/"completed.json") == plan["prepared_completed_sha256"]
            == done["prepared_completed_sha256"] == digest(frozen/"prepared-completed.json"), "Prepared external identity")
    prepared_done = read_json(prepared/"completed.json")
    require(prepared_done["status"] == "completed" and prepared_done["phase"] == "prepare"
            and prepared_done["files"] == plan["prepared_files"], "Prepared completion scope")
    verify_manifest(prepared, prepared_done["files"], {"started.json", "catalog.json", "summary.json", "rows.jsonl"})
    train, dev = read_ledger(prepared, plan)
    n, d = len(train), len(dev)
    updates, evaluation_batches = 20*math.ceil(n/256), math.ceil(d/256)
    require(plan["updates_per_fit"] == updates and plan["evaluation_batches_per_fit"] == evaluation_batches, "Operation recipe")
    counts = [sum(r["derived_bin"] == "unmentioned_retention" for r in train),
              sum(r["derived_bin"] == "assigned_retention" for r in train), sum(r["derived_bin"] in BINS[2:] for r in train)]
    require(all(counts) and plan["objective"] == {"counts": counts, "weights": [n/(3*v) for v in counts]}, "Training-only objective")
    expected_counts = operation_counts(n, d, updates, evaluation_batches)
    require(done["progress"]["completed_fits"] == FIT_ORDER and done["progress"]["active_fit"] is None
            and done["progress"]["totals"] == {k: 9*v for k, v in expected_counts.items()}, "Whole run coverage")
    records = {}
    for seed in SEEDS:
        order_path = run/f"orders-{seed}.npy"
        require(digest(order_path) == plan["orders"][str(seed)]["sha256"], "Saved orders identity")
        orders = np.load(order_path, allow_pickle=False)
        rng = np.random.default_rng(seed)
        require(orders.dtype == np.int64 and orders.shape == (20, n)
                and all(np.array_equal(order, rng.permutation(n)) for order in orders), "Fixed full permutations")
        for mode in MODES:
            fit = f"{mode}-{seed}"
            destination = run/"fits"/fit
            record = read_json(destination/"completed.json")
            require(record["status"] == "completed" and record["method"] == mode and record["seed"] == seed
                    and record["epochs"] == 20 and record["counts"] == expected_counts
                    and record["plan_sha256"] == done["plan_sha256"]
                    and record["orders_sha256"] == digest(order_path), "Fit terminal/coverage identity")
            configuration = record["configuration"]
            fixed_configuration = {"class": "DialogueConditionalObservation", "version": "dialogue-conditional-observation-v1",
                "mode": mode, "input_dim": 384, "projection_dim": 64, "hidden_dim": 64, "feature_dim": 396,
                "parameters": 99393 if mode == "mean" else 173121, "shared_scorer_parameters": 99393,
                "attention_parameters": 0 if mode == "mean" else 73728, "zero_input_parameters": 64, "softmax_shift_parameters": 1,
                "attention_width": None if mode == "mean" else 64,
                "schema_pair": None if mode == "mean" else "[query;query]" if mode == "slot" else "[query;candidate]",
                "lexical_fields": ["user_match", "system_match", "unique_longest_user", "unique_longest_system",
                                   "literal_current", "literal_previous", "is_none", "is_dontcare",
                                   "affirmative_cue_for_true", "negative_cue_for_false"]}
            require(all(configuration.get(k) == v for k, v in fixed_configuration.items()), "Model configuration witness")
            require(0 < record["wall_seconds"] <= done["wall_seconds"]
                    and 0 < record["evaluation"]["wall_seconds"] <= record["wall_seconds"]
                    and 0 < record["process_lifetime_peak_rss_bytes"] <= done["process_lifetime_peak_rss_bytes"], "Fit cost witness")
            for key in ("initial_common_sha256", "initial_attention_sha256"):
                value = record[key]
                if key == "initial_attention_sha256" and mode == "mean":
                    require(value is None, "Mean has no attention initialization")
                else:
                    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                            "Initializer digest format")
            verify_manifest(destination, record["files"], {"weights.pt", "updates.jsonl", "dev-predictions.npz"})
            schedule = plan["work_schedules"][fit]
            require(record["training_work"] == schedule["training"]["totals"]
                    and record["evaluation"]["work"] == schedule["evaluation"]["totals"]
                    and record["evaluation"]["rows"] == d, "Fit work coverage")
            eval_totals = defaultdict(int)
            for start in range(0, d, 256):
                work, _ = expected_geometry(dev[start:start+256], mode)
                for key, value in work.items():
                    if not key.startswith("max_"):
                        eval_totals[key] += value
            require(dict(eval_totals) == record["evaluation"]["work"], "Evaluation work reconstruction")
            check_normalization(record["training_normalization"], updates, 20*n, record["training_work"])
            check_normalization(record["evaluation"]["normalization"], evaluation_batches, d, record["evaluation"]["work"])
            check_journal(destination/"updates.jsonl", train, orders, record)
            records[fit] = record
        require(len({records[f"{m}-{seed}"]["initial_common_sha256"] for m in MODES}) == 1
                and records[f"mean-{seed}"]["initial_attention_sha256"] is None
                and records[f"slot-{seed}"]["initial_attention_sha256"] == records[f"candidate-{seed}"]["initial_attention_sha256"],
                "Paired initialization witness")
    require(done["fits"] == [{"method": records[f]["method"], "seed": records[f]["seed"], "counts": expected_counts}
                             for f in FIT_ORDER], "Root and fit summary reconciliation")
    return done, plan, prepared_done, dev, records


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(keys), "Array payload fields")
        return {key: archive[key] for key in keys}


def check_literal_reference(rows, references, plan, prepared_done):
    path = Path(plan["feature_headers"]["lexical"]["path"]).resolve()
    require(path.is_relative_to(ROOT.resolve()), "Literal feature source path")
    verify_file(path, prepared_done["authenticated_inputs"][str(path)])
    lexical = np.load(path, mmap_mode="r", allow_pickle=False)
    require(lexical.dtype == np.float32 and list(lexical.shape) == plan["feature_headers"]["lexical"]["shape"]
            and not lexical.flags.writeable, "Literal feature array")
    expected = []
    for row in rows:
        c, start = row["candidate_count"], row["cache"]["lexical_start"]
        require(row["cache"]["lexical_stride"] == 10 and row["cache"]["lexical_candidates"] == c
                and type(start) is int and 0 <= start and start+10*c <= len(lexical), "Literal address")
        values = lexical[start:start+10*c].reshape(c, 10)[:, 4]
        require(np.logical_or(values == 0, values == 1).all() and np.count_nonzero(values == 1) == 1,
                "Exact supported literal one-hot required")
        expected.append(int(np.flatnonzero(values == 1)[0]))
    require(np.array_equal(references["literal_indices"], np.asarray(expected, np.int64)), "Saved literal reference reconstruction")


def inherited_costs(prepared_done):
    result = {}
    for name, relative in FEATURE_PARENTS.items():
        parent = bounded_path(ROOT, relative)
        receipt_path = parent/"completed.json"
        identity = prepared_done["authenticated_inputs"][str(receipt_path)]
        verify_file(receipt_path, identity)
        receipt = read_json(receipt_path)
        require(receipt["status"] == "completed" and math.isfinite(receipt["wall_seconds"])
                and receipt["wall_seconds"] >= 0, "Inherited cache cost witness")
        payload_bytes = 0
        for file, entry in receipt["files"].items():
            path = bounded_path(parent, file)
            recorded = prepared_done["authenticated_inputs"][str(path)]
            require(recorded["sha256"] == (entry if type(entry) is str else entry["sha256"]), "Inherited cache file binding")
            if type(entry) is dict:
                require(recorded["bytes"] == entry["bytes"], "Inherited cache byte binding")
            payload_bytes += recorded["bytes"]
        result[name] = {"completed_sha256": identity["sha256"], "recorded_wall_seconds": receipt["wall_seconds"],
                        "wall_scope": receipt.get("wall_scope", "Whole preparation wall as recorded by the inherited producer; no retrospective scope expansion"),
                        "recorded_manifest_payload_bytes": payload_bytes, "completion_receipt_bytes": identity["bytes"],
                        "interpretation": "Prior shared cache preparation, charged separately; not rerun or allocated per arm"}
    return result


def execute_report(run, pin, out):
    started = time.monotonic()
    run, out = Path(run).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        done, plan, prepared_done, rows, records = authenticate_run(run, pin)
        cache_costs = inherited_costs(prepared_done)
        refs = load_npz(run/"references.npz", ("row_indices", "previous_indices", "literal_indices"))
        check_literal_reference(rows, refs, plan, prepared_done)
        predictions = {(m, s): load_npz(run/"fits"/f"{m}-{s}"/"dev-predictions.npz", ("log_probs", "row_indices"))
                       for m in MODES for s in SEEDS}
        summary = analyze(rows, predictions, refs)
        summary.update(technical_admission_passed=True, continuation_allowed=summary["primary"]["primary_nll_rule_passed"],
                       source_run_completed_sha256=pin, plan_sha256=done["plan_sha256"],
                       cost={"whole_run_wall_seconds": done["wall_seconds"], "whole_run_wall_scope": done["wall_scope"],
                             "process_lifetime_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
                             "output_bytes": sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
                             "fits": {name: {k: record[k] for k in ("wall_seconds", "counts", "configuration",
                                                                    "training_work", "evaluation")} for name, record in records.items()},
                             "inherited_cache_preparation": cache_costs,
                             "conditional_metadata_preparation": {"recorded_wall_seconds": prepared_done["wall_seconds"],
                                 "wall_scope": prepared_done["wall_scope"],
                                 "manifest_payload_bytes": sum(v["bytes"] for v in prepared_done["files"].values())},
                             "cache_cost_scope": "Separate recorded costs; do not sum overlapping nested timing scopes or call shared caches free"})
        (out/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
        receipt = {"status": "completed", "source_run": str(run), "source_run_completed_sha256": pin,
                   "reporter_sha256": digest(__file__), "summary_sha256": digest(out/"summary.json"),
                   "analysis_wall_seconds": time.monotonic()-started, "model_calls": 0,
                   "technical_admission_passed": True, "continuation_allowed": summary["continuation_allowed"]}
        (out/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")
        return receipt
    except BaseException as error:
        try:
            (out/"failed.json").write_text(json.dumps({"status": "failed", "error_type": type(error).__name__,
                                                      "error": str(error), "source_run_completed_sha256": pin}, indent=2)+"\n")
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note("Failure receipt write failed: " + repr(secondary))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute_report(args.run, args.completed_sha256, args.out)))


if __name__ == "__main__":
    main()
