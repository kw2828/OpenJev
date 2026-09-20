"""Independent primary-result arithmetic, with no model/reporter imports.

Run only after all nine completed fits and the main report are externally pinned.
Saved source/runtime/timing/normalization witnesses are authenticated, not replayed.
This audit does not independently reproduce every descriptive subgroup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("flat_stratum", "token_mean", "token_aligned")
SEEDS = (6201, 6202, 6203)
ORDER = [f"{m}-{s}" for i, s in enumerate(SEEDS) for m in METHODS[i:]+METHODS[:i]]
VERSION = "dialogue-token-alignment-independent-primary-audit-v1"
LIMITS = {"wall_seconds": 7200., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
AUDIT_SECONDS = 60
MASS_TOLERANCE = 2e-6
VALUES = ("none", "dontcare", "true", "false", "other")
METRICS = ("accuracy", "wrong_selected_branch", "retained_error", "true_false_positive_rate", "dontcare_false_positive_rate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON scalar: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(parent, name):
    parent = Path(parent).resolve()
    result = (parent/name).resolve()
    require(result.is_relative_to(parent) and result != parent, "Unsafe payload path")
    return result


def manifest(parent, members, *, exact=False, terminal="completed.json"):
    require(type(members) is dict and members, "Nonempty payload manifest")
    for name, record in members.items():
        file = safe(parent, name)
        require(file.is_file() and file.stat().st_size == record["bytes"]
                and sha(file) == record["sha256"], "Payload binding: " + name)
    if exact:
        actual = {p.relative_to(parent).as_posix() for p in Path(parent).rglob("*") if p.is_file()}
        require(actual == set(members) | {terminal}, "Exact payload closure")


def fraction(record):
    n, d = record["numerator"], record["denominator"]
    require(type(n) is int and type(d) is int and 0 <= n <= d, "Integer count fraction")
    return Fraction(n, d) if d else None


def row_metadata(rows):
    """Derive supports directly from candidate identity and public type codes."""
    ids, labels, priors, sizes, types, held = [], [], [], [], [], []
    for row in rows:
        require(row["split"] == "train" and row["admission"] == "admitted", "TRAIN admitted cohort")
        require(type(row["heldout_service"]) is bool, "Boolean panel")
        n = row["candidate_count"]
        require(type(n) is int and 3 <= n <= 12, "Candidate count")
        codes = row["candidate_types"]
        require(len(codes) == n and all(type(c) is int and 0 <= c <= 4 for c in codes)
                and codes.count(0) == codes.count(1) == 1
                and codes.count(2) <= 1 and codes.count(3) <= 1, "Public candidate types")
        y, old = row["current_label_index"], row["previous_current_index"]
        require(type(y) is int and type(old) is int and 0 <= y < n and 0 <= old < n, "Target/prior support")
        require(row["current_value_group"] == VALUES[codes[y]], "Target type identity")
        require((y == old) == (row["current_candidate_id"] == row["previous_candidate_id"]), "Previous candidate mapping")
        transition = ("unmentioned_retention" if y == old and codes[y] == 0 else
                      "assigned_retention" if y == old else "first_assignment" if codes[old] == 0 else
                      "clear" if codes[y] == 0 else "revision")
        require(row["derived_bin"] == transition, "Adjacent transition identity")
        require(type(row["row_index"]) is int and row["row_index"] >= 0, "Row index")
        ids.append(row["row_index"]); labels.append(y); priors.append(old)
        sizes.append(n); types.append(codes+[-1]*(12-n)); held.append(row["heldout_service"])
    require(ids == sorted(set(ids)), "Unique ordered row identity")
    result = {"ids": np.asarray(ids, np.int64), "labels": np.asarray(labels, np.int64),
              "sizes": np.asarray(sizes), "types": np.asarray(types), "held": np.asarray(held, bool)}
    result["changed"] = result["labels"] != np.asarray(priors)
    result["target_types"] = result["types"][np.arange(len(rows)), result["labels"]]
    return result


def counted(event, eligible):
    return {"numerator": int(np.count_nonzero(event & eligible)), "denominator": int(np.count_nonzero(eligible))}


def mean_metrics(vectors, mask, rows):
    selected = np.flatnonzero(mask)
    require(len(selected) > 0, "Nonempty primary support")
    result = {"rows": len(selected)}
    groupings = {name: sorted({rows[i][key] for i in selected})
                 for name, key in (("equal_service", "service"), ("equal_dialogue", "dialogue_id"))}
    for metric, values in vectors.items():
        means = {"row": math.fsum(float(values[i]) for i in selected)/len(selected)}
        for name, key in (("equal_service", "service"), ("equal_dialogue", "dialogue_id")):
            buckets = {g: [] for g in groupings[name]}
            for i in selected:
                buckets[rows[i][key]].append(float(values[i]))
            means[name] = math.fsum(math.fsum(v)/len(v) for v in buckets.values())/len(buckets)
        result[metric] = means
    return result


def score(rows, meta, packet):
    require(set(packet) == {"row_indices", "log_probs"}, "Prediction member schema")
    raw = packet["log_probs"]
    require(packet["row_indices"].dtype == np.int64 and np.array_equal(packet["row_indices"], meta["ids"]), "Prediction row identity")
    require(raw.dtype == np.float32 and raw.shape == (len(rows), 12), "Saved float32 prediction shape")
    support = np.arange(12)[None, :] < meta["sizes"][:, None]
    require(np.isfinite(raw[support]).all() and np.all(raw[support] <= 0)
            and np.isneginf(raw[~support]).all(), "Finite supported / negative-infinite padded log probabilities")
    logs = raw.astype(np.float64)
    probabilities = np.exp(logs)
    error = float(np.max(np.abs(np.sum(probabilities, axis=1)-1.)))
    require(error <= MASS_TOLERANCE, "Saved probability normalization")
    choices = np.argmax(logs, axis=1)
    selected_types = meta["types"][np.arange(len(rows)), choices]
    correct = choices == meta["labels"]
    branch_wrong = np.minimum(selected_types, 2) != np.minimum(meta["target_types"], 2)
    wrong_value = ~correct & ~branch_wrong
    residual = probabilities.copy()
    residual[np.arange(len(rows)), meta["labels"]] -= 1.
    vectors = {"accuracy": correct.astype(float), "nll": -logs[np.arange(len(rows)), meta["labels"]],
               "brier": np.einsum("ij,ij->i", residual, residual),
               "wrong_selected_branch": branch_wrong.astype(float), "wrong_value": wrong_value.astype(float)}
    primary = meta["held"] & meta["changed"]
    retained = meta["held"] & ~meta["changed"]
    counts = {"accuracy": counted(correct, primary), "wrong_selected_branch": counted(branch_wrong, primary),
              "retained_error": counted(~correct, retained)}
    for name, code in (("true", 2), ("dontcare", 1)):
        eligible = meta["held"] & (meta["target_types"] != code) & np.any(meta["types"] == code, axis=1)
        counts[name+"_false_positive_rate"] = counted(selected_types == code, eligible)
        counts[name+"_changed_recall"] = counted(correct, primary & (meta["target_types"] == code))
    return {"decisions": counts, "maximum_mass_error": error,
            "cells": {name: mean_metrics(vectors, mask, rows) for name, mask in
                      (("heldout_service/all", meta["held"]), ("heldout_service/changed", primary),
                       ("heldout_service/retained", retained))}}, choices


def continuation(fits):
    """Five seed-mean inequalities and six paired inequalities per control."""
    require(set(fits) == set(ORDER), "All nine fits")
    result = {}
    boundaries = {"accuracy": (Fraction(1, 50), True), "wrong_selected_branch": (-Fraction(1, 50), False),
                  "retained_error": (Fraction(1, 200), False), "true_false_positive_rate": (Fraction(1, 200), False),
                  "dontcare_false_positive_rate": (Fraction(1, 200), False)}
    for control in METHODS[:2]:
        paired = {}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                aligned = fits[f"token_aligned-{seed}"]["decisions"][metric]
                baseline = fits[f"{control}-{seed}"]["decisions"][metric]
                require(aligned["denominator"] == baseline["denominator"], "Paired denominator")
                a, b = fraction(aligned), fraction(baseline)
                values.append(None if a is None or b is None else a-b)
            paired[metric] = values
            difference = None if None in values else sum(values, Fraction(0))/len(SEEDS)
            threshold, lower = boundaries[metric]
            result[control+"/mean_"+metric] = {"exact_difference": None if difference is None else [difference.numerator, difference.denominator],
                "difference": None if difference is None else float(difference),
                "passed": difference is not None and (difference >= threshold if lower else difference <= threshold)}
        for metric in ("accuracy", "wrong_selected_branch"):
            for seed, difference in zip(SEEDS, paired[metric], strict=True):
                result[f"{control}/seed_{seed}_{metric}_nonworse"] = {
                    "difference": None if difference is None else float(difference),
                    "passed": difference is not None and (difference >= 0 if metric == "accuracy" else difference <= 0)}
    require(len(result) == 22, "Twenty-two behavioral checks")
    return {"passed": all(c["passed"] for c in result.values()),
            "checks_passed": sum(c["passed"] for c in result.values()), "total_checks": 22, "checks": result}


def pair_counts(meta, a, b):
    ac, bc = a == meta["labels"], b == meta["labels"]
    return {name: {"rows": int(mask.sum()), "wrong_to_correct": int((ac & ~bc & mask).sum()),
                   "correct_to_wrong": int((~ac & bc & mask).sum()), "both_correct": int((ac & bc & mask).sum()),
                   "both_wrong": int((~ac & ~bc & mask).sum())}
            for name, mask in (("heldout_service/all", meta["held"]),
                               ("heldout_service/changed", meta["held"] & meta["changed"]),
                               ("heldout_service/retained", meta["held"] & ~meta["changed"]))}


def compare_report(fits, pairs, rule, report):
    require(set(report["fits"]) == set(ORDER), "Report all-nine membership")
    for name in ORDER:
        expected = report["fits"][name]
        require(fits[name]["decisions"] == expected["decisions"], "Decision counts: "+name)
        for group, actual in fits[name]["cells"].items():
            saved = expected["cells"][group]
            require(actual["rows"] == saved["rows"], "Metric denominator")
            for metric in ("accuracy", "nll", "brier", "wrong_selected_branch", "wrong_value"):
                for weighting, value in actual[metric].items():
                    require(math.isclose(value, saved[metric][weighting], rel_tol=2e-12, abs_tol=2e-12),
                            "Primary metric mismatch: "+name+"/"+group+"/"+metric+"/"+weighting)
        require(math.isclose(fits[name]["maximum_mass_error"], expected["maximum_mass_error"], abs_tol=2e-15), "Mass witness")
    for control, seeds in pairs.items():
        for seed, groups in seeds.items():
            for group, actual in groups.items():
                require(all(report["pairs"][control][seed][group][k] == v for k, v in actual.items()), "Paired repair/harm table")
    saved = report["continuation"]
    require(all(saved[k] == rule[k] for k in ("passed", "checks_passed", "total_checks")), "Scientific result")
    checks = {c["control"]+"/"+c["name"]: c for c in saved["checks"]}
    require(set(checks) == set(rule["checks"]), "Exact behavioral check identities")
    for name, actual in rule["checks"].items():
        require(checks[name]["passed"] is actual["passed"], "Behavioral check: "+name)
        for key in actual.keys()-{"passed"}:
            require(checks[name][key] == actual[key], "Exact behavioral difference: "+name)


def candidate_codes(query):
    result = []
    for identity, value in zip(query["candidate_ids"], query["candidate_values"], strict=True):
        if identity == "reserved:NOT_MENTIONED": code = 0
        elif identity == "reserved:DONTCARE": code = 1
        elif query["boolean_slot"]:
            require(value.strip().casefold() in ("true", "false"), "Boolean ontology value")
            code = 2 if value.strip().casefold() == "true" else 3
        else: code = 4
        result.append(code)
    return result


def norm_counts(rows):
    counts = {"batches": 0, "rows": len(rows), "supported_candidates": 0, "masked_candidates": 0}
    for i in range(0, len(rows), 32):
        sizes = [r["candidate_count"] for r in rows[i:i+32]]
        counts["batches"] += 1
        counts["supported_candidates"] += sum(sizes)
        counts["masked_candidates"] += len(sizes)*max(sizes)-sum(sizes)
    return counts


def norm_witness(record, counts):
    extrema = ("max_abs_mass_error", "min_supported_log_prob", "max_supported_log_prob")
    require(set(record) == set(counts) | set(extrema), "Normalization schema")
    require(all(type(record[k]) is int and record[k] == v for k, v in counts.items()), "Normalization coverage")
    require(all(type(record[k]) in (int, float) and math.isfinite(record[k]) for k in extrema), "Finite normalization extrema")
    require(0 <= record["max_abs_mass_error"] <= MASS_TOLERANCE
            and record["min_supported_log_prob"] <= record["max_supported_log_prob"] <= 0, "Normalization bounds")


def fit_witness(run, name, plan, fit_rows, evaluation, order):
    method, seed_text = name.rsplit("-", 1)
    dest = run/"fits"/name
    receipt = read(dest/"completed.json")
    require(receipt["status"] == "completed" and receipt["method"] == method and receipt["seed"] == int(seed_text)
            and receipt["plan_sha256"] == sha(run/"plan.json")
            and receipt["orders_sha256"] == plan["orders"][seed_text]["sha256"], "Fit identity: "+name)
    require(set(receipt["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}, "Fit membership")
    for key, record in receipt["files"].items():
        require(record == plan["_run_files"][f"fits/{name}/{key}"], "Fit/root manifest agreement")
    require(receipt["epochs"] == 20 and receipt["training_effective_batches"] == 2300
            and receipt["training_microbatches"] == 18260, "Fit full training coverage")
    expected = {"forward_attempted": 18685, "forward_returned": 18685,
                "backward_attempted": 18260, "backward_returned": 18260,
                "optimizer_attempted": 2300, "optimizer_returned": 2300,
                "training_rows": 584220, "evaluation_rows": 13599}
    require(receipt["counts"] == expected, "Fit actual attempted/returned counters")
    aggregate, aggregate_work = {}, {}
    lines = iter((dest/"updates.jsonl").read_text().splitlines())
    updates, update_seconds = 0, 0.
    for epoch, permutation in enumerate(order):
        for start in range(0, len(fit_rows), 256):
            entry = decode(next(lines))
            batch = [fit_rows[int(i)] for i in permutation[start:start+256]]
            coverage = norm_counts(batch)
            require(entry["epoch"] == epoch and entry["start"] == start
                    and entry["row_indices"] == [r["row_index"] for r in batch]
                    and entry["microbatches"] == coverage["batches"], "Paired journal update identity")
            require(math.isfinite(entry["weighted_loss"]) and entry["weighted_loss"] >= 0
                    and math.isfinite(entry["wall_seconds"]) and entry["wall_seconds"] > 0, "Finite update witnesses")
            norm_witness(entry["normalization"], coverage)
            for key, value in entry["normalization"].items():
                if key.startswith("max_"): aggregate[key] = max(aggregate.get(key, value), value)
                elif key.startswith("min_"): aggregate[key] = min(aggregate.get(key, value), value)
                else: aggregate[key] = aggregate.get(key, 0)+value
            for key, value in entry["work"].items():
                require(type(value) is int and value >= 0, "Nonnegative integral work")
                aggregate_work[key] = aggregate_work.get(key, 0)+value
            updates += 1; update_seconds += entry["wall_seconds"]
    require(next(lines, None) is None and updates == 2300, "Exact journal length")
    require(aggregate == receipt["training_normalization"] and aggregate_work == receipt["training_work"], "Training aggregate witnesses")
    ev = receipt["evaluation"]
    require(ev["effective_batches"] == 54 and ev["microbatches"] == 425, "Evaluation batch coverage")
    norm_witness(ev["normalization"], norm_counts(evaluation))
    require(0 < update_seconds <= receipt["training_wall_seconds"] and 0 < ev["wall_seconds"]
            and receipt["training_wall_seconds"]+ev["wall_seconds"] <= receipt["wall_seconds"] <= 7200
            and 0 < receipt["process_lifetime_peak_rss_bytes"] <= LIMITS["rss_bytes"], "Fit paid cost witnesses")
    witness = receipt["initializer_witness"]
    require(set(witness["full_state_sha256"]) == set(witness["common_state_sha256"]) == set(METHODS)
            and witness["full_state_sha256"]["token_mean"] == witness["full_state_sha256"]["token_aligned"]
            and len(set(witness["common_state_sha256"].values())) == 1
            and receipt["initial_state_sha256"] == witness["full_state_sha256"][method]
            and receipt["initial_common_sha256"] == witness["common_state_sha256"][method], "Recorded initializer pairing")
    return receipt


def authenticate(args):
    """Authenticate all completed bytes/metadata before decoding quality arrays."""
    run, report = Path(args.run).resolve(), Path(args.report).resolve()
    require(sha(run/"completed.json") == args.completed_sha256, "External execution pin")
    require(sha(run/"plan.json") == args.plan_sha256, "External plan pin")
    require(sha(report/"receipt.json") == args.report_receipt_sha256, "External report pin")
    done, plan, started = read(run/"completed.json"), read(run/"plan.json"), read(run/"started.json")
    rr = read(report/"receipt.json")
    require(rr["status"] == "completed" and rr["technical_validity_passed"] is True
            and rr["execution_completed_sha256"] == args.completed_sha256
            and rr["plan_sha256"] == args.plan_sha256, "Successful matching primary report")
    manifest(report, rr["files"], exact=True, terminal="receipt.json")
    require(done["status"] == "completed" and done["phase"] == "train"
            and done["version"] == plan["version"] == "dialogue-token-alignment-scientific-v1"
            and done["plan_sha256"] == args.plan_sha256
            and done["completed_fits"] == done["expected_fits"] == plan["expected_fits"] == ORDER,
            "All-nine completed campaign")
    expected_files = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"}
    expected_files |= {f"orders-{s}.npy" for s in SEEDS}
    expected_files |= {f"fits/{f}/{p}" for f in ORDER for p in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    require(set(done["files"]) == expected_files and len(expected_files) == 43, "Exact 44-file run")
    manifest(run, done["files"], exact=True)
    require(plan["limits"] == done["limits"] == LIMITS and done["no_retry"] is True
            and plan["original_capacity_admitted"] is False and done["original_capacity_admitted"] is False,
            "Separate fixed allocation")
    require(0 < done["wall_seconds"] <= 7200 and 0 < done["process_lifetime_peak_rss_bytes"] <= LIMITS["rss_bytes"]
            and sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Whole-run budgets")
    require(started["runtime"] == done["runtime"] == plan["runtime"]
            and started["request"]["plan_sha256"] == args.plan_sha256, "Recorded runtime/request")
    freeze = Path(started["request"]["plan"]).resolve().parent
    require(sha(freeze/"plan.json") == args.plan_sha256, "Original scientific freeze")
    frozen = read(freeze/"completed.json")
    require(frozen["status"] == "completed" and frozen["phase"] == "freeze"
            and frozen["plan_sha256"] == args.plan_sha256 and frozen["model_calls"] == 0, "Metadata-only freeze")
    manifest(freeze, frozen["files"], exact=True)
    require(plan["source_sha256"] == done["source_sha256"] == frozen["source_sha256"], "Source closure")
    for name, pin in plan["source_sha256"].items():
        require(sha(safe(ROOT, name)) == sha(safe(freeze/"sources", name)) == pin, "Live/snapshot source: "+name)
    require(rr["source_sha256"] == sha(ROOT/"scripts/report_dialogue_alignment.py"), "Main reporter source")
    cfg = plan["config"]
    for key, value in {"epochs": 20, "batch_size": 256, "microbatch_size": 32, "learning_rate": .001,
                       "weight_decay": .0001, "gradient_clip": 1., "threads": 4, "interop_threads": 1,
                       "dtype": "float32", "deterministic": True, "methods": list(METHODS), "seeds": list(SEEDS)}.items():
        require(cfg[key] == value, "Fixed training recipe: "+key)
    for key, value in {"updates_per_fit": 2300, "training_microbatches_per_fit": 18260,
                       "evaluation_batches_per_fit": 54, "evaluation_microbatches_per_fit": 425}.items():
        require(plan[key] == value, "Fixed full work: "+key)
    capacity = Path(plan["capacity_plan_path"])
    require(sha(capacity) == plan["capacity_plan_sha256"] == "3b28aac097d9c8c4f3eba38d9ee7ab50f3ed9786cd36249e4772005022f20eea"
            and sha(capacity.parent/"completed.json") == plan["capacity_freeze_completed_sha256"], "Original capacity freeze pins")
    cap_plan, cap_done = read(capacity), read(capacity.parent/"completed.json")
    manifest(capacity.parent, cap_done["files"], exact=True)
    prepared = Path(plan["prepared_path"])
    require(sha(prepared/"completed.json") == plan["prepared_completed_sha256"], "Prepared metadata input")
    prep = read(prepared/"completed.json"); manifest(prepared, prep["files"], exact=True)
    all_rows = [decode(v) for v in (prepared/"rows.jsonl").read_text().splitlines()]
    canonical = {r["row_index"]: r for r in all_rows}
    require(len(canonical) == len(all_rows), "Unique original metadata rows")
    catalog = read(prepared/"catalog.json")["queries"]
    rows = [decode(v) for v in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    meta = row_metadata(rows)
    require(meta["ids"].tolist() == plan["evaluation_row_indices"] and len(rows) == 13599, "Frozen evaluation rows")
    held = set(plan["split"]["heldout_services"])
    for row in rows:
        original = canonical[row["row_index"]]
        require(row == {**original, "heldout_service": original["service"] in held,
                        "candidate_types": candidate_codes(catalog[row["query_index"]])}, "Original row/type join")
    fit_rows = [canonical[i] for i in plan["fit_row_indices"]]
    require(len(fit_rows) == 29211 and plan["fit_row_indices"] == sorted(set(plan["fit_row_indices"]))
            and not ({r["dialogue_id"] for r in fit_rows} & {r["dialogue_id"] for r in rows}), "Fixed disjoint fit dialogues")
    split = ROOT/"output/dialogue-typed-v1/split-design-01"
    require(sha(split/"receipt.json") == plan["split_receipt_sha256"], "Original split receipt")
    manifest(split, read(split/"receipt.json")["files"])
    membership = read(split/"membership.json")["admitted_row_indices"]
    require(membership["fit"] == plan["fit_row_indices"] and membership["evaluation"] == plan["evaluation_row_indices"]
            and membership["primary_heldout_service"] == meta["ids"][meta["held"]].tolist(), "Original service split membership")
    require(meta["held"].sum() == 7819 and (meta["held"] & meta["changed"]).sum() == 578
            and (meta["held"] & ~meta["changed"]).sum() == 7241 and len(held) == 6, "Frozen primary support")
    changed_primary = meta["held"] & meta["changed"]
    require([int((changed_primary & (meta["target_types"] == code)).sum()) for code in (2, 1, 3, 0)]
            == [29, 5, 0, 0], "Sparse TRUE/DONTCARE/FALSE/clear support")
    schema = Path(plan["schema_cache_path"])
    require(sha(schema/"completed.json") == plan["schema_completed_sha256"] == cap_plan["schema_completed_sha256"], "Schema input pin")
    manifest(schema, read(schema/"completed.json")["files"], exact=True)
    orders = {}
    for seed in SEEDS:
        filename = f"orders-{seed}.npy"
        require(sha(run/filename) == sha(freeze/filename) == sha(capacity.parent/filename)
                == plan["orders"][str(seed)]["sha256"], "Copied original paired order")
        values = np.load(run/filename, allow_pickle=False)
        require(values.dtype == np.int64 and values.shape == (20, 29211)
                and all(np.array_equal(np.sort(v), np.arange(29211)) for v in values), "Full epoch permutations")
        orders[seed] = values
    plan["_run_files"] = done["files"]
    fits = {name: fit_witness(run, name, plan, fit_rows, rows, orders[int(name.rsplit("-", 1)[1])]) for name in ORDER}
    for seed in SEEDS:
        witnesses = [fits[f"{m}-{seed}"]["initializer_witness"] for m in METHODS]
        require(all(v == witnesses[0] for v in witnesses), "Same paired initializer witness")
    totals = {k: sum(f["counts"][k] for f in fits.values()) for k in fits[ORDER[0]]["counts"]}
    require(done["progress"]["totals"] == totals and done["progress"]["active_fit"] is None
            and done["progress"]["completed_fits"] == ORDER
            and sum(f["wall_seconds"] for f in fits.values()) <= done["wall_seconds"], "Whole completed work ledger")
    report_summary = read(report/"summary.json")
    require(report_summary["technical_validity_passed"] is True, "Main report technical admission")
    return rows, meta, done, report_summary, totals


def execute(args):
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = {k: str(v) for k, v in vars(args).items()}
    old_handler = None
    def timeout(_signal, _frame):
        raise TimeoutError("Independent audit 60-second wall limit")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "No overlapping timer")
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, AUDIT_SECONDS)
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sha(__file__),
                                   "wall_seconds_limit": AUDIT_SECONDS, "model_calls": 0, "no_retry": True})
        rows, meta, done, report, totals = authenticate(args)
        fits, choices = {}, {}
        for name in ORDER:
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as arrays:
                fits[name], choices[name] = score(rows, meta, {k: arrays[k] for k in arrays.files})
        pairs = {control: {str(seed): pair_counts(meta, choices[f"token_aligned-{seed}"], choices[f"{control}-{seed}"])
                           for seed in SEEDS} for control in METHODS[:2]}
        rule = continuation(fits)
        compare_report(fits, pairs, rule, report)
        result = {"status": "completed", "version": VERSION, "agreement": True, "technical_validity_passed": True,
                  "fits": fits, "pairs": pairs, "continuation": rule, "operation_totals": totals,
                  "primary_support_per_fit": {"all": int(meta["held"].sum()),
                      "changed": int((meta["held"] & meta["changed"]).sum()),
                      "retained": int((meta["held"] & ~meta["changed"]).sum()),
                      "changed_types": {value: int((meta["held"] & meta["changed"] & (meta["target_types"] == code)).sum())
                                        for code, value in enumerate(VALUES)}},
                  "execution_wall_seconds": done["wall_seconds"], "execution_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
                  "scope": "Independent NumPy primary metrics/rates, exact 22 count-fraction checks, paired correctness tables, row/source/input/file bindings and recorded normalization/count coverage. No model/encoder/checkpoint deserialization. Other descriptive subgroups, full geometry/work formulas, configuration/parameter validity and literal-reference generation inherit the authenticated main report. Neural calculations, initializer digests, timing/RSS and gradient checks remain source-bound witnesses, not independently replayed."}
        write(out/"summary.json", result)
        require(sha(Path(args.run)/"completed.json") == args.completed_sha256
                and sha(Path(args.report)/"receipt.json") == args.report_receipt_sha256, "Final receipt stability")
        files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "request": request,
                                  "files": files, "source_sha256": sha(__file__), "model_calls": 0,
                                  "no_retry": True, "agreement": True, "continuation_passed": rule["passed"],
                                  "wall_seconds": time.monotonic()-started,
                                  "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        require(time.monotonic()-started <= AUDIT_SECONDS, "Final audit wall cap")
        return result
    except BaseException as error:
        if old_handler is not None: signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists(): (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                                       "error": repr(error), "wall_seconds": time.monotonic()-started,
                                       "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original audit failure
            if callable(getattr(error, "add_note", None)): error.add_note("Preservation failure: "+repr(secondary))
        raise
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--report-receipt-sha256", required=True)
    parser.add_argument("--out", required=True)
    result = execute(parser.parse_args())
    print(json.dumps({"agreement": result["agreement"], "scientific_passed": result["continuation"]["passed"]}))
