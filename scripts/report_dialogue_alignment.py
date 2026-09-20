"""Saved-only behavioral analysis of the nine-fit token-alignment campaign.

Imports one hash-pinned, model-free metric reader. Never loads model weights,
producer code, encoders or random generators. All nine fits precede scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRIC_SOURCE = "scripts/report_dialogue_typed_v2.py"
METRIC_PIN = "6c97b4839352083885a3df5240cb1d3564a9652ec90ed30948a5bbada8b54365"
VERSION = "dialogue-token-alignment-scientific-v1"
METHODS = ("flat_stratum", "token_mean", "token_aligned")
SEEDS = (6201, 6202, 6203)
FITS = {f"{method}-{seed}" for method in METHODS for seed in SEEDS}
FIT_ORDER = [f"{m}-{s}" for i, s in enumerate(SEEDS) for m in METHODS[i:]+METHODS[:i]]
PRIMARY = "heldout_service/changed"
REPORT_LIMITS = {"wall_seconds": 60, "rss_bytes": 6*1024**3, "output_bytes": 32*1024**2}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


require(digest(ROOT/METRIC_SOURCE) == METRIC_PIN, "Frozen metric source identity")
_spec = importlib.util.spec_from_file_location("_alignment_frozen_metrics", ROOT/METRIC_SOURCE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
read, write, safe, check_manifest = base.read, base.write, base.safe, base.check_manifest
CONFIG = {**base.CONFIG, "methods": list(METHODS), "seeds": list(SEEDS), "microbatch_size": 32}
LIMITS = {"wall_seconds": 7200., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
CAPACITY_PLAN_PIN = "3b28aac097d9c8c4f3eba38d9ee7ab50f3ed9786cd36249e4772005022f20eea"
CAPACITY_FREEZE_PIN = "d037a378a62f4c81ae55cf28e1e4e7a1c901c5059c0326f05ae18207fd0f33c3"
SCHEMA_PIN = "630116d89e87ada14ad6dbe0028680acd6adaa5413f850364e3232aa6762168a"
NEW_SOURCES = {"scripts/study_dialogue_alignment.py", "tests/test_study_dialogue_alignment.py",
               "scripts/report_dialogue_alignment.py", "tests/test_report_dialogue_alignment.py",
               "research/dialogue-token-alignment-scientific-protocol.md", METRIC_SOURCE,
               "tests/test_dialogue_typed_report_v2.py"}


def decision_vectors(rows, choice):
    target = np.asarray([r["current_label_index"] for r in rows])
    target_branch = np.asarray([min(r["candidate_types"][int(t)], 2) for r, t in zip(rows, target, strict=True)])
    selected_branch = np.asarray([min(r["candidate_types"][int(c)], 2) for r, c in zip(rows, choice, strict=True)])
    correct = choice == target
    branch_wrong = target_branch != selected_branch
    value_wrong = ~correct & ~branch_wrong
    require(np.array_equal(correct.astype(int)+branch_wrong+value_wrong, np.ones(len(rows))), "Disjoint decision partition")
    return {"accuracy": correct.astype(np.float64), "wrong_selected_branch": branch_wrong.astype(np.float64),
            "wrong_value": value_wrong.astype(np.float64)}


def branch_losses(rows, packet):
    logs = packet["log_probs"].astype(np.float64)
    total = -logs[np.arange(len(rows)), [r["current_label_index"] for r in rows]]
    branch = np.empty(len(rows), np.float64)
    for i, row in enumerate(rows):
        kinds = np.minimum(np.asarray(row["candidate_types"]), 2)
        selected = logs[i, :len(kinds)][kinds == kinds[row["current_label_index"]]]
        maximum = selected.max()
        branch[i] = -(maximum+np.log(np.exp(selected-maximum).sum()))
    value = total-branch
    require(np.isfinite(branch).all() and np.isfinite(value).all(), "Finite raw-log loss decomposition")
    return {"branch_nll": branch, "within_branch_nll": value}


def count(vector, mask):
    return {"numerator": int(vector[mask].sum()), "denominator": int(mask.sum())}


def fraction(record):
    n, d = record["numerator"], record["denominator"]
    require(type(n) is int and type(d) is int and 0 <= n <= d, "Canonical count fraction")
    return Fraction(n, d) if d else None


def continuation(fits):
    require(set(fits) == FITS, "All nine final fits required")
    checks, comparisons = [], {}
    for control in METHODS[:2]:
        paired = []
        for seed in SEEDS:
            a, b = (fits[f"{m}-{seed}"] for m in ("token_aligned", control))
            deltas = {}
            for metric in ("accuracy", "wrong_selected_branch", "retained_error", "true_false_positive_rate", "dontcare_false_positive_rate"):
                ac, bc = a["decisions"][metric], b["decisions"][metric]
                require(ac["denominator"] == bc["denominator"], "Paired decision denominator")
                ar, br = fraction(ac), fraction(bc)
                deltas[metric] = None if ar is None or br is None else ar-br
            paired.append({"seed": seed, "deltas": deltas})
        for metric, relation, threshold in (("accuracy", ">=", Fraction(1, 50)),
                                             ("wrong_selected_branch", "<=", -Fraction(1, 50)),
                                             ("retained_error", "<=", Fraction(1, 200)),
                                             ("true_false_positive_rate", "<=", Fraction(1, 200)),
                                             ("dontcare_false_positive_rate", "<=", Fraction(1, 200))):
            values = [p["deltas"][metric] for p in paired]
            delta = sum(values, Fraction())/3 if all(v is not None for v in values) else None
            passed = delta is not None and (delta >= threshold if relation == ">=" else delta <= threshold)
            checks.append({"control": control, "name": "mean_"+metric, "difference": None if delta is None else float(delta),
                           "exact_difference": None if delta is None else [delta.numerator, delta.denominator],
                           "threshold": float(threshold), "relation": relation, "passed": bool(passed)})
        for p in paired:
            for metric, relation in (("accuracy", ">="), ("wrong_selected_branch", "<=")):
                delta = p["deltas"][metric]
                passed = delta is not None and (delta >= 0 if relation == ">=" else delta <= 0)
                checks.append({"control": control, "name": f"seed_{p['seed']}_{metric}_nonworse",
                               "difference": None if delta is None else float(delta), "threshold": 0.,
                               "relation": relation, "passed": bool(passed)})
        comparisons[control] = [{"seed": p["seed"], "aligned_minus_control":
                                 {k: None if v is None else float(v) for k, v in p["deltas"].items()}} for p in paired]
    require(len(checks) == 22, "Frozen expanded check count")
    return {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks),
            "total_checks": 22, "checks": checks, "paired": comparisons,
            "scope": "Pooled-row primary decision rates, exact count-fraction seed means; both controls required, no seed or alternative-winner selection. Technical all-nine completion is separately mandatory."}


def aggregate(rows, packets, references):
    require(set(packets) == FITS, "All nine final fits required before metrics")
    identities = base.validate_rows(rows)
    masks = base.group_masks(rows)
    groups = {key: base.layout(rows, mask) for key, mask in masks.items()}
    fits, choices, vectors = {}, {}, {}
    for name in FIT_ORDER:
        values, choice, mass_error = base.predictions(rows, packets[name])
        values.update(branch_losses(rows, packets[name]))
        decisions = decision_vectors(rows, choice)
        values.update(decisions)
        counts = base.decision_counts(rows, choice)
        counts.update({k: count(decisions[k], masks[PRIMARY]) for k in ("accuracy", "wrong_selected_branch")})
        fits[name] = {"cells": {key: base.describe(values, group) for key, group in groups.items()},
                      "decisions": counts, "maximum_mass_error": mass_error,
                      "changed_per_service": {service: base.describe(values, base.layout(rows,
                          masks["all/changed"] & np.asarray([r["service"] == service for r in rows])))
                          for service in sorted({r["service"] for r in rows})}}
        choices[name], vectors[name] = choice, values
    require(set(references) == {"row_indices", "previous_indices", "literal_indices"}, "Reference fields")
    require(references["row_indices"].dtype == np.int64 and np.array_equal(references["row_indices"], identities), "Reference row order")
    target = np.asarray([r["current_label_index"] for r in rows])
    support = np.asarray([r["candidate_count"] for r in rows])
    refs = {}
    for method in ("previous", "literal"):
        choice = references[method+"_indices"]
        require(choice.dtype == np.int64 and choice.shape == target.shape and ((choice >= 0) & (choice < support)).all(), "Reference support")
        if method == "previous": require(np.array_equal(choice, [r["previous_current_index"] for r in rows]), "Exact privileged previous reference")
        refs[method] = {key: base.describe({"accuracy": (choice == target).astype(np.float64)}, group) for key, group in groups.items()}
    pairs = {}
    for control in METHODS[:2]:
        pairs[control] = {}
        for seed in SEEDS:
            a, b = f"token_aligned-{seed}", f"{control}-{seed}"
            ac, bc = choices[a] == target, choices[b] == target
            pairs[control][str(seed)] = {key: {
                "rows": group["rows"], "correct_to_wrong": int((bc & ~ac & masks[key]).sum()),
                "wrong_to_correct": int((~bc & ac & masks[key]).sum()),
                "both_correct": int((bc & ac & masks[key]).sum()), "both_wrong": int((~bc & ~ac & masks[key]).sum()),
                "aligned_minus_control": {metric: base.means(vectors[a][metric]-vectors[b][metric], group)
                                           for metric in vectors[a]}} for key, group in groups.items()}
    rule = continuation(fits)
    return {"version": VERSION, "fits": fits, "references": refs, "pairs": pairs,
            "groups_per_fit": len(groups), "continuation": rule, "continuation_allowed": rule["passed"],
            "scope": "Historically exposed official TRAIN service-held-out development cohort with supplied gold previous state. Equal-service/equal-dialogue loss and accuracy, sparse TRUE/DONTCARE recall and repair/harm are descriptive. Wrong branch means branch of selected candidate, not maximum branch probability. No fresh-confirmation, significance, novelty or population-generalization claim."}


def summed(records):
    result = {}
    for record in records:
        for key, value in record.items():
            if not key.startswith(("max_", "schema_max_")):
                result[key] = result.get(key, 0)+value
    return result


def merge_norm(records):
    result = {}
    for record in records:
        for key, value in record.items():
            if key == "min_supported_log_prob": result[key] = min(result.get(key, value), value)
            elif key in ("max_supported_log_prob", "max_abs_mass_error"): result[key] = max(result.get(key, value), value)
            else: result[key] = result.get(key, 0)+value
    return result


def public_work(rows, method, schema_lengths):
    """Reconstruct actor payload counters from public support/address geometry."""
    pieces, padding = [], 0
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        b, c = len(batch), max(r["candidate_count"] for r in batch)
        lengths = [r["cache"]["token_stop"]-r["cache"]["token_start"] for r in batch]
        require(all(type(v) is int and v > 0 for v in lengths), "Positive public context lengths")
        length, supported = max(lengths), sum(r["candidate_count"] for r in batch)
        floats = b*(length*384+length)+b*384+b*c*(384+10+1)
        work = {"rows": b, "supported_candidate_positions": supported, "padded_candidate_positions": b*c,
                "supported_token_positions": sum(lengths), "padded_token_positions": b*length,
                "attention_score_positions": b*c*length, "token_key_positions": b*length,
                "attention_query_positions": b*c, "scorer_positions": b*c,
                "float_input_scalars": floats, "float_input_bytes": 4*floats,
                "boolean_input_bytes": b*c+b*length, "candidate_type_input_bytes": 8*b*c}
        if method != "flat_stratum":
            sizes = [schema_lengths[r["query_index"]] for r in batch]
            require(all(len(v) == r["candidate_count"] for r, v in zip(batch, sizes, strict=True)), "Schema support geometry")
            s = max(max(v) for v in sizes)
            work.update({"schema_supported_candidate_positions": supported, "schema_padded_candidate_positions": b*c,
                         "schema_supported_context_token_positions": sum(lengths), "schema_padded_context_token_positions": b*length,
                         "schema_supported_schema_token_positions": sum(map(sum, sizes)), "schema_padded_schema_token_positions": b*c*s,
                         "schema_supported_pairwise_positions": sum(n*sum(v) for n, v in zip(lengths, sizes, strict=True)),
                         "schema_padded_pairwise_positions": b*c*length*s,
                         "schema_schema_float_input_bytes": 4*b*c*s*385, "schema_schema_boolean_input_bytes": b*c*s})
        pieces.append(work)
        padding += b*c-supported
    return summed(pieces), len(pieces), padding


def check_normalization(record, rows, microbatches, padding):
    base.normalization(record, rows, microbatches)
    require(record["masked_candidates"] == padding, "Actual microbatch support/padding coverage")


def finite_cost(value, name, *, positive=True):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0), name)
    return value


def copied_order(item, capacity, seed, locations, nfit):
    """Capacity freezes bind order files under payloads, not an orders mapping."""
    filename = f"orders-{seed}.npy"
    require(item["file"] == filename and item["sha256"] == capacity["payloads"][filename]["sha256"]
            and item["shape"] == [20, nfit], "Inherited order record")
    require(all(digest(Path(path)/filename) == item["sha256"] for path in locations), "Copied order bytes")
    values = np.load(Path(locations[0])/filename, allow_pickle=False)
    require(values.dtype == np.int64 and values.shape == (20, nfit)
            and all(np.array_equal(np.sort(epoch), np.arange(nfit)) for epoch in values), "Full epoch permutations")
    return values


def authenticate_metadata(run, done, plan):
    prepared = Path(plan["prepared_path"]).resolve()
    require(digest(prepared/"completed.json") == plan["prepared_completed_sha256"], "Prepared metadata pin")
    prep = read(prepared/"completed.json")
    check_manifest(prepared, prep["files"], {"started.json", "catalog.json", "rows.jsonl", "summary.json"})
    canonical = {r["row_index"]: r for r in (base.decode(line) for line in (prepared/"rows.jsonl").read_text().splitlines())}
    catalog = read(prepared/"catalog.json")["queries"]
    rows = [base.decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    require(base.validate_rows(rows).tolist() == plan["evaluation_row_indices"], "Frozen evaluation order")
    held = set(plan["split"]["heldout_services"])
    for row in rows:
        require(row == {**canonical[row["row_index"]], "heldout_service": row["service"] in held,
                        "candidate_types": base.expected_types(catalog[row["query_index"]])}, "Exact canonical metadata join")
    fit = [canonical[i] for i in plan["fit_row_indices"]]
    require(plan["fit_row_indices"] == sorted(set(plan["fit_row_indices"]))
            and all(r["split"] == "train" and r["admission"] == "admitted" and r["service"] not in held for r in fit), "Fixed fit cohort")
    require(not ({r["dialogue_id"] for r in rows} & {r["dialogue_id"] for r in fit}), "Dialogue split leakage")
    require(set(plan["fit_row_indices"]) | set(plan["evaluation_row_indices"]) ==
            {i for i, r in canonical.items() if r["split"] == "train" and r["admission"] == "admitted"}, "All admitted TRAIN coverage")
    require(plan["objective"] == base.objective_from_rows(fit), "Fit-only three-stratum objective")
    split = ROOT/"output/dialogue-typed-v1/split-design-01"
    require(digest(split/"receipt.json") == plan["split_receipt_sha256"], "Original split identity")
    receipt = read(split/"receipt.json")
    for name, item in receipt["files"].items():
        require(digest(safe(split, name)) == item["sha256"] and safe(split, name).stat().st_size == item["bytes"], "Split payload")
    base.validate_split_membership(read(split/"membership.json")["admitted_row_indices"], fit, rows, held)
    schema = Path(plan["schema_cache_path"]).resolve()
    require(plan["schema_completed_sha256"] == SCHEMA_PIN and digest(schema/"completed.json") == SCHEMA_PIN, "Schema cache identity")
    cached = read(schema/"completed.json")
    check_manifest(schema, cached["files"], set(cached["files"]))
    index = read(schema/"index.json")
    offsets = np.load(schema/"offsets.npy", allow_pickle=False)
    require(offsets.dtype == np.int64 and offsets.ndim == 1 and offsets[0] == 0 and (np.diff(offsets) > 0).all(), "Schema integer offsets")
    queries = {q["query_index"]: q for q in index["queries"]}
    require(len(queries) == 53 and index["candidate_occurrences"] == 307 and index["unique_texts"] == 360
            and len(offsets) == 361 and set(queries) == {r["query_index"] for r in fit+rows}, "All supplied TRAIN schemas")
    for row in fit+rows:
        q = queries[row["query_index"]]
        require(q["candidate_feature_indices"] == row["cache"]["candidate_feature_indices"]
                and q["candidate_ids"] == catalog[row["query_index"]]["candidate_ids"], "Schema candidate address/order")
    lengths = {qi: [int(offsets[i+1]-offsets[i]) for i in q["candidate_token_ids"]] for qi, q in queries.items()}
    return fit, rows, lengths, cached


def authenticate_run(run, expected_sha):
    """Complete identity/cost authentication precedes prediction-array loading."""
    run = Path(run).resolve()
    require(digest(run/"completed.json") == expected_sha, "External run completion digest")
    done, plan, started = (read(run/name) for name in ("completed.json", "plan.json", "started.json"))
    require(done["status"] == "completed" and done["phase"] == "train" and done["version"] == VERSION,
            "All-nine completed campaign required")
    require(done["completed_fits"] == done["expected_fits"] == plan["expected_fits"] == FIT_ORDER, "All nine identities/order")
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"} | {f"orders-{s}.npy" for s in SEEDS}
    names |= {f"fits/{fit}/{name}" for fit in FITS for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    check_manifest(run, done["files"], names)
    require(len(names)+1 == 44, "Exact 44-file closure")
    plan_sha = digest(run/"plan.json")
    require(done["plan_sha256"] == plan_sha and plan["version"] == VERSION and plan["config"] == CONFIG
            and plan["limits"] == done["limits"] == LIMITS, "Frozen configuration/budget")
    require(plan["original_capacity_admitted"] is False and done["original_capacity_admitted"] is False,
            "Original failed capacity screen remains failed")
    require(plan["quality_metrics_in_runner"] is False and plan["no_retry"] is True
            and done["quality_metrics_computed"] is False and done["encoder_calls"] == 0
            and done["official_dev_inference"] is False and done["test_contents_accessed"] is False,
            "Fixed closed training scope")
    require(started["phase"] == "train" and started["version"] == VERSION
            and started["runtime"] == plan["runtime"] == done["runtime"], "Runtime witnesses")
    frozen = Path(started["request"]["plan"]).resolve().parent
    require(started["request"]["plan_sha256"] == plan_sha and digest(frozen/"plan.json") == plan_sha, "Frozen plan identity")
    freeze = read(frozen/"completed.json")
    sources = plan["source_sha256"]
    require(sources == done["source_sha256"] == freeze["source_sha256"], "Source maps")
    freeze_names = {"started.json", "plan.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+s for s in sources}
    require(freeze["status"] == "completed" and freeze["phase"] == "freeze" and freeze["plan_sha256"] == plan_sha, "Successful metadata freeze")
    check_manifest(frozen, freeze["files"], freeze_names)
    for name, pin in sources.items():
        require(digest(safe(ROOT, name)) == digest(safe(frozen/"sources", name)) == pin, "Live/frozen source: "+name)
    capacity_path = Path(plan["capacity_plan_path"]).resolve()
    require(plan["capacity_plan_sha256"] == CAPACITY_PLAN_PIN and digest(capacity_path) == CAPACITY_PLAN_PIN, "Exact inherited order plan")
    capacity = read(capacity_path)
    require(plan["capacity_freeze_completed_sha256"] == CAPACITY_FREEZE_PIN
            and digest(capacity_path.parent/"completed.json") == CAPACITY_FREEZE_PIN, "Inherited freeze completion")
    parent = read(capacity_path.parent/"completed.json")
    check_manifest(capacity_path.parent, parent["files"], set(capacity["payloads"]) | {"plan.json"})
    require(plan["schema_cache_path"] == capacity["schema_cache"] and plan["schema_completed_sha256"] == capacity["schema_completed_sha256"]
            and plan["prepared_completed_sha256"] == capacity["prepared_completed_sha256"]
            and plan["split_receipt_sha256"] == capacity["split_receipt_sha256"]
            and plan["feature_headers"] == capacity["feature_headers"] and plan["objective"] == capacity["objective"]
            and plan["schema_preparation"] == capacity["schema_preparation"], "Inherited cache/data/recipe identity")
    fit_rows, rows, lengths, cached = authenticate_metadata(run, done, plan)
    require(len(fit_rows) == 29211 and len(rows) == 13599
            and int(base.group_masks(rows)[PRIMARY].sum()) == 578
            and int(base.group_masks(rows)["heldout_service/retained"].sum()) == 7241, "Frozen cohort sizes")
    expected_sources = set(capacity["source_sha256"]) | set(cached["source_sha256"]) | NEW_SOURCES
    require(set(sources) == expected_sources and sources[METRIC_SOURCE] == METRIC_PIN, "Exact imported source closure")
    orders = {}
    for seed in SEEDS:
        item = plan["orders"][str(seed)]
        orders[seed] = copied_order(item, capacity, seed, (run, frozen, capacity_path.parent), len(fit_rows))
    updates, evaluations = 20*math.ceil(len(fit_rows)/256), math.ceil(len(rows)/256)
    train_micro, eval_micro = 20*math.ceil(len(fit_rows)/32), math.ceil(len(rows)/32)
    require(plan["updates_per_fit"] == updates and plan["evaluation_batches_per_fit"] == evaluations, "Effective batch budget")
    require(plan["training_microbatches_per_fit"] == train_micro
            and plan["evaluation_microbatches_per_fit"] == eval_micro, "Microbatch budget")
    counts = {"forward_attempted": train_micro+eval_micro, "forward_returned": train_micro+eval_micro,
              "backward_attempted": train_micro, "backward_returned": train_micro,
              "optimizer_attempted": updates, "optimizer_returned": updates,
              "training_rows": 20*len(fit_rows), "evaluation_rows": len(rows)}
    fit_receipts = {}
    for name in FIT_ORDER:
        method, seed_text = name.rsplit("-", 1); seed = int(seed_text)
        dest = run/"fits"/name; record = read(dest/"completed.json")
        require(record["status"] == "completed" and record["method"] == method and record["seed"] == seed
                and record["version"] == VERSION and record["epochs"] == 20 and record["original_capacity_admitted"] is False
                and record["plan_sha256"] == plan_sha and record["orders_sha256"] == plan["orders"][seed_text]["sha256"]
                and record["counts"] == counts, "Fit identity/work: "+name)
        require(record["training_effective_batches"] == updates and record["training_microbatches"] == train_micro
                and record["evaluation"]["effective_batches"] == evaluations and record["evaluation"]["microbatches"] == eval_micro,
                "Full effective/microbatch receipts")
        check_manifest(dest, record["files"], {"weights.pt", "updates.jsonl", "predictions.npz"})
        config = record["configuration"]
        core = {"input_dim": 384, "projection_dim": 64, "hidden_dim": 64, "feature_dim": 401}
        core.update({"class": "DialogueTypedObservation", "mode": "flat", "parameters": 173506,
                     "version": "dialogue-typed-observation-v1"} if method == "flat_stratum" else
                    {"class": "DialogueTokenAlignment", "mode": "mean" if method == "token_mean" else "aligned",
                     "parameters": 124482, "common_scorer_parameters": 75138, "version": "dialogue-token-alignment-v1"})
        require(all(config.get(k) == v for k, v in core.items()), "Registered model configuration")
        for key in ("initial_state_sha256", "initial_common_sha256"):
            pin = record[key]
            require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin), "Initializer digest")
        journal = [base.decode(line) for line in (dest/"updates.jsonl").read_text().splitlines()]
        require(len(journal) == updates, "Journal update count")
        j, journal_wall, norms, works = 0, 0., [], []
        for epoch, order in enumerate(orders[seed]):
            for start in range(0, len(fit_rows), 256):
                batch = [fit_rows[int(i)] for i in order[start:start+256]]; entry = journal[j]; j += 1
                require(entry["epoch"] == epoch and entry["start"] == start
                        and entry["row_indices"] == [r["row_index"] for r in batch], "Exact journal row coverage")
                finite_cost(entry["weighted_loss"], "Finite training loss", positive=False)
                journal_wall += finite_cost(entry["wall_seconds"], "Update wall time")
                work, micro, padding = public_work(batch, method, lengths)
                require(entry["work"] == work and entry["microbatches"] == micro, "Independent public actor work")
                check_normalization(entry["normalization"], batch, micro, padding)
                norms.append(entry["normalization"]); works.append(work)
        require(merge_norm(norms) == record["training_normalization"] and summed(works) == record["training_work"], "Training witness totals")
        eval_work, micro, padding = public_work(rows, method, lengths)
        require(micro == eval_micro and record["evaluation"]["work"] == eval_work, "Evaluation actor work")
        check_normalization(record["evaluation"]["normalization"], rows, micro, padding)
        train_wall = finite_cost(record["training_wall_seconds"], "Training seconds")
        eval_wall = finite_cost(record["evaluation"]["wall_seconds"], "Evaluation seconds")
        fit_wall = finite_cost(record["wall_seconds"], "Fit seconds")
        require(journal_wall <= train_wall and train_wall+eval_wall <= fit_wall <= done["wall_seconds"], "Timing scopes")
        require(0 < record["process_lifetime_peak_rss_bytes"] <= done["process_lifetime_peak_rss_bytes"], "Lifetime RSS ordering")
        fit_receipts[name] = record
    for seed in SEEDS:
        records = [fit_receipts[f"{m}-{seed}"] for m in METHODS]
        require(len({r["initial_common_sha256"] for r in records}) == 1
                and records[1]["initial_state_sha256"] == records[2]["initial_state_sha256"], "Matched common/new-model initializers")
        require(records[0]["initializer_witness"] == records[1]["initializer_witness"] == records[2]["initializer_witness"], "Paired initializer witness")
        witness = records[0]["initializer_witness"]
        require(witness["full_state_sha256"] == {m: fit_receipts[f"{m}-{seed}"]["initial_state_sha256"] for m in METHODS}
                and witness["common_state_sha256"] == {m: fit_receipts[f"{m}-{seed}"]["initial_common_sha256"] for m in METHODS}, "Actual initializers match witness")
    require(done["progress"]["completed_fits"] == FIT_ORDER and done["progress"]["active_fit"] is None
            and done["progress"]["totals"] == {k: 9*v for k, v in counts.items()}, "Root complete work totals")
    size = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    require(0 < finite_cost(done["wall_seconds"], "Whole wall") <= LIMITS["wall_seconds"]
            and 0 < done["process_lifetime_peak_rss_bytes"] <= LIMITS["rss_bytes"] and size <= LIMITS["output_bytes"]
            and sum(r["wall_seconds"] for r in fit_receipts.values()) <= done["wall_seconds"], "Whole execution caps")
    return rows, done, plan, fit_receipts, size, cached


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); prior_handler = None
    source_pin = digest(__file__)
    def check():
        require(time.monotonic()-started <= REPORT_LIMITS["wall_seconds"], "Report wall cap")
        require(peak_rss() <= REPORT_LIMITS["rss_bytes"], "Report RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= REPORT_LIMITS["output_bytes"], "Report output cap")
    def expired(_sig, _frame): raise TimeoutError("Saved-only report wall cap exceeded")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        prior_handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, REPORT_LIMITS["wall_seconds"])
        run = Path(args.run).resolve()
        rows, done, plan, records, size, cached = authenticate_run(run, args.run_sha256)
        check()
        packets = {}
        for name in FIT_ORDER:
            with np.load(run/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                packets[name] = {key: archive[key] for key in archive.files}
        with np.load(run/"references.npz", allow_pickle=False) as archive:
            refs = {key: archive[key] for key in archive.files}
        summary = aggregate(rows, packets, refs)
        summary.update(status="completed", technical_validity_passed=True, execution_completed_sha256=args.run_sha256,
                       plan_sha256=done["plan_sha256"], source_sha256=done["source_sha256"], split=plan["split"],
                       objective=plan["objective"], original_capacity_admitted=False,
                       costs={"whole_wall_seconds": done["wall_seconds"], "execution_bytes": size,
                         "process_lifetime_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
                         "schema_preparation": {"wall_seconds": cached["wall_seconds"], "completed_sha256": SCHEMA_PIN,
                                                "payload_bytes": cached["payload_bytes"], "work": cached["work"]},
                         "process_load_start": done["process_load_start"], "process_load_end": done["process_load_end"],
                         "per_fit": {name: {"wall_seconds": r["wall_seconds"], "training_wall_seconds": r["training_wall_seconds"],
                             "evaluation_wall_seconds": r["evaluation"]["wall_seconds"], "counts": r["counts"],
                             "training_work": r["training_work"], "evaluation_work": r["evaluation"]["work"],
                             "parameters": r["configuration"]["parameters"]} for name, r in records.items()},
                         "scope": "Whole campaign includes authentication, initialization, actor assembly, training/evaluation, serialization and hashing. Training and evaluation are nested fit scopes, never added again to fit or whole time. Schema preparation and inherited frozen preprocessing are separate, not free. RSS is process-lifetime high-water; normalization, timing, gradients and initializer tensors are source-bound execution witnesses, not model replay."})
        check_manifest(run, done["files"], set(done["files"]))
        require(digest(run/"completed.json") == args.run_sha256 and digest(__file__) == source_pin, "End identity stability")
        require(all(digest(safe(ROOT, name)) == pin for name, pin in done["source_sha256"].items()), "End source stability")
        check(); write(out/"summary.json", summary)
        with (out/"report.md").open("x") as stream: stream.write(report_text(summary))
        check()
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "execution_completed_sha256": args.run_sha256,
              "plan_sha256": done["plan_sha256"], "source_sha256": source_pin, "metric_source_sha256": METRIC_PIN,
              "technical_validity_passed": True, "continuation_allowed": summary["continuation_allowed"],
              "execution_files": len(done["files"])+1, "execution_members": done["files"],
              "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()},
              "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": peak_rss(), "limits": REPORT_LIMITS,
              "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0,
              "scope": "Saved-only metrics and exact behavioral count fractions. Opaque checkpoints hashed only. Exact previous-state reference reconstructed; literal reference is bound to producer/source/manifest, not independently recomputed from raw lexical inputs. All-nine completion precedes score decoding."})
        check()
        return summary["continuation_allowed"]
    except BaseException as error:
        if prior_handler is not None: signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists(): (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "error": repr(error), "source_sha256": source_pin,
                  "execution_completed_sha256": args.run_sha256, "wall_seconds": time.monotonic()-started,
                  "model_calls": 0, "encoder_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - retain original error
            if callable(getattr(error, "add_note", None)): error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if prior_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior_handler)


def report_text(summary):
    rule = summary["continuation"]
    lines = ["# Token-alignment development campaign", "", summary["scope"], "",
             f"Behavioral continuation: **{'PASS' if rule['passed'] else 'FAIL'}**, {rule['checks_passed']}/22 expanded checks.",
             "All nine final fits passed separate technical authentication before any quality arrays were decoded.", "",
             "| Arm | Seed | Changed correct | Changed wrong branch | Equal-service NLL | TRUE recall | DONTCARE recall |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def rate(c): return f"{c['numerator']}/{c['denominator']}" if c["denominator"] else "undefined"
    for method in METHODS:
        for seed in SEEDS:
            fit = summary["fits"][f"{method}-{seed}"]; d = fit["decisions"]
            loss = fit["cells"][PRIMARY]["nll"]["equal_service"]
            lines.append(f"| {method} | {seed} | {rate(d['accuracy'])} | {rate(d['wrong_selected_branch'])} | "
                         f"{loss if loss is not None else 'undefined'} | {rate(d['true_changed_recall'])} | {rate(d['dontcare_changed_recall'])} |")
    lines += ["", "Raw NLL is computed directly from saved float32 logs promoted to float64; no floors or repaired distributions.",
              "Wrong branch refers to the selected candidate. Branch and within-branch losses are descriptive and add to total NLL.",
              "Previous-value and literal-register references have accuracy only. Previous value is evaluator-provided privileged input.",
              "The earlier capacity screen remains NOT ADMITTED. This campaign used a separate 7,200-second allocation before quality access.",
              "", "| Fit | Training seconds | Evaluation seconds | Fit seconds |",
              "|---|---:|---:|---:|"]
    for name, c in summary["costs"]["per_fit"].items():
        lines.append(f"| {name} | {c['training_wall_seconds']:.6f} | {c['evaluation_wall_seconds']:.6f} | {c['wall_seconds']:.6f} |")
    lines += ["", f"Whole-run wall time: {summary['costs']['whole_wall_seconds']:.6f} seconds.",
              summary["costs"]["scope"], "", "All panel/transition/value means, services, paired repairs and new errors are in summary.json."]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps({"continuation_allowed": execute(parser.parse_args())}))
