"""Corrected saved-only reader for the paired typed/support experiment.

No model, producer, reporter helper, checkpoint deserializer or random generator
is imported. Original metrics and rule arithmetic are unchanged. The reader now
validates all six fields of the authenticated split projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-typed-v1"
SEEDS = (6101, 6102, 6103)
METHODS = ("flat_stratum", "flat_balanced", "typed_stratum", "typed_balanced")
FITS = {f"{method}-{seed}" for method in METHODS for seed in SEEDS}
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
VALUES = ("none", "dontcare", "true", "false", "other")
BASELINE, CANDIDATE = "flat_balanced", "typed_balanced"
TOLERANCE = 2e-6
CONFIG = {"methods": list(METHODS), "seeds": list(SEEDS), "epochs": 20, "batch_size": 256,
          "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1., "input_dim": 384,
          "projection_dim": 64, "hidden_dim": 64, "threads": 4, "interop_threads": 1,
          "dtype": "float32", "deterministic": True}
LIMITS = {"wall_seconds": 3600., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
ORIGINAL_REPORTER_SHA256 = "d5fabd0afd020e2d4cd966aee396f1a8c34d9d1b2a6259c1fcec79e20e61d651"
FAILED_REPORT_SHA256 = "b54d84e599ffd86a24f43ae127295d6ad478d1171838f7348d7d6b87bfac0ec1"
SOURCE_NAMES = {
    "research/dialogue-conditional-observation-design.md", "research/dialogue-conditional-preparation-protocol.md",
    "scripts/prepare_dialogue_conditional.py", "scripts/prepare_dialogue_tokens.py", "scripts/study_dialogue_copy.py",
    "scripts/study_dialogue_memory.py", "scripts/study_dialogue_tokens.py", "src/openjev/research/dialogue_copy_features.py",
    "src/openjev/research/dialogue_state_data.py", "tests/test_prepare_dialogue_conditional.py",
    "src/openjev/research/dialogue_conditional_observation.py", "tests/test_dialogue_conditional_observation.py",
    "src/openjev/research/dialogue_copy_memory.py", "scripts/study_dialogue_conditional.py",
    "tests/test_study_dialogue_conditional.py", "scripts/report_dialogue_conditional.py",
    "tests/test_report_dialogue_conditional.py", "research/dialogue-conditional-training-protocol.md",
    "scripts/study_dialogue_typed.py", "tests/test_study_dialogue_typed.py", "scripts/report_dialogue_typed.py",
    "tests/test_report_dialogue_typed.py", "src/openjev/research/dialogue_typed_observation.py",
    "tests/test_dialogue_typed_observation.py", "research/dialogue-typed-protocol.md",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON scalar " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text())


def validate_rows(rows):
    require(type(rows) is list and rows, "Nonempty evaluation rows required")
    identities = []
    for row in rows:
        require(row["split"] == "train" and row["admission"] == "admitted", "Official TRAIN cohort required")
        require(type(row["row_index"]) is int and row["row_index"] >= 0, "Row index")
        require(type(row["heldout_service"]) is bool, "Held-out service flag")
        require(all(type(row[k]) is str and row[k] for k in ("dialogue_id", "query_id", "service")), "Grouping identity")
        count = row["candidate_count"]
        require(type(count) is int and 3 <= count <= 12, "Candidate support")
        kinds = row["candidate_types"]
        require(type(kinds) is list and len(kinds) == count and all(type(t) is int and 0 <= t <= 4 for t in kinds), "Candidate types")
        require(kinds.count(0) == kinds.count(1) == 1 and kinds.count(2) <= 1 and kinds.count(3) <= 1, "Canonical type support")
        current, previous = row["current_label_index"], row["previous_current_index"]
        require(all(type(i) is int and 0 <= i < count for i in (current, previous)), "Target/prior support")
        require(row["current_value_group"] == VALUES[kinds[current]], "Schema target type")
        old_id, new_id = row["previous_candidate_id"], row["current_candidate_id"]
        require(all(type(v) is str and v for v in (old_id, new_id)), "Canonical candidate identities")
        require((current == previous) == (old_id == new_id), "Atomic prior/target mapping")
        require((kinds[current] == 0) == (new_id == "reserved:NOT_MENTIONED")
                and (kinds[current] == 1) == (new_id == "reserved:DONTCARE")
                and (kinds[previous] == 0) == (old_id == "reserved:NOT_MENTIONED")
                and (kinds[previous] == 1) == (old_id == "reserved:DONTCARE"), "Reserved identity mapping")
        transition = (BINS[0] if current == previous and kinds[current] == 0 else BINS[1] if current == previous
                      else BINS[2] if kinds[previous] == 0 else BINS[4] if kinds[current] == 0 else BINS[3])
        require(row["derived_bin"] == transition, "Transition identity")
        identities.append(row["row_index"])
    require(identities == sorted(set(identities)), "Unique canonical evaluation order")
    for service in {r["service"] for r in rows}:
        require(len({r["heldout_service"] for r in rows if r["service"] == service}) == 1, "Service split consistency")
    return np.asarray(identities, dtype=np.int64)


def group_masks(rows):
    held = np.asarray([r["heldout_service"] for r in rows])
    bins = np.asarray([r["derived_bin"] for r in rows])
    values = np.asarray([r["current_value_group"] for r in rows])
    changed = np.isin(bins, BINS[2:])
    masks = {"all": np.ones(len(rows), dtype=bool), "changed": changed, "retained": ~changed}
    masks.update({"transition/"+v: bins == v for v in BINS})
    for value in VALUES:
        selected = values == value
        masks["value/"+value] = selected
        masks["value/"+value+"/changed"] = selected & changed
        masks["value/"+value+"/retained"] = selected & ~changed
        masks.update({"transition_value/"+transition+"/"+value: selected & (bins == transition) for transition in BINS})
    return {panel+"/"+name: population & mask for panel, population in
            (("all", np.ones(len(rows), dtype=bool)), ("heldout_service", held), ("seen_service", ~held))
            for name, mask in masks.items()}


def layout(rows, selected):
    indices = np.flatnonzero(selected)
    services = np.asarray([rows[int(i)]["service"] for i in indices])
    dialogs = np.asarray([rows[int(i)]["dialogue_id"] for i in indices])
    service_names, service_index = np.unique(services, return_inverse=True)
    dialogue_names, dialogue_index = np.unique(dialogs, return_inverse=True)
    return {"indices": indices, "service_names": service_names, "service_index": service_index,
            "dialogue_index": dialogue_index, "rows": len(indices), "services": len(service_names),
            "dialogues": len(dialogue_names), "schema_queries": len({rows[int(i)]["query_id"] for i in indices})}


def means(values, group):
    if not group["rows"]:
        return {"row": None, "equal_service": None, "equal_dialogue": None}
    selected = values[group["indices"]]
    result = {"row": float(np.mean(selected, dtype=np.float64))}
    for name in ("service", "dialogue"):
        ids = group[name+"_index"]
        result["equal_"+name] = float(np.mean(np.bincount(ids, weights=selected)/np.bincount(ids)))
    return result


def describe(values, group):
    return {**{k: group[k] for k in ("rows", "services", "dialogues", "schema_queries")},
            **{name: means(vector, group) for name, vector in values.items()}}


def predictions(rows, packet):
    require(set(packet) == {"row_indices", "log_probs"}, "Prediction keys")
    identities = validate_rows(rows)
    logs, ids = packet["log_probs"], packet["row_indices"]
    require(ids.dtype == np.int64 and np.array_equal(ids, identities), "Prediction row identity/order")
    require(logs.dtype == np.float32 and logs.shape == (len(rows), 12), "Prediction shape/dtype")
    mask = np.arange(12)[None] < np.asarray([r["candidate_count"] for r in rows])[:, None]
    require(np.isfinite(logs[mask]).all() and (logs[mask] <= 0).all() and np.isneginf(logs[~mask]).all(), "Supported/masked logs")
    logs = logs.astype(np.float64)
    probability = np.exp(logs)
    mass_error = float(np.abs(probability.sum(1)-1).max())
    require(mass_error <= TOLERANCE, "Unnormalized saved probability mass")
    target = np.asarray([r["current_label_index"] for r in rows])
    choice = logs.argmax(1)
    brier_residual = probability.copy()
    brier_residual[np.arange(len(rows)), target] -= 1
    vectors = {"accuracy": (choice == target).astype(np.float64),
               "nll": -logs[np.arange(len(rows)), target], "brier": np.square(brier_residual).sum(1)}
    return vectors, choice, mass_error


def decision_counts(rows, choice):
    held = np.asarray([r["heldout_service"] for r in rows])
    target = np.asarray([r["current_label_index"] for r in rows])
    changed = target != np.asarray([r["previous_current_index"] for r in rows])
    selected_types = np.asarray([r["candidate_types"][int(c)] for r, c in zip(rows, choice, strict=True)])
    target_types = np.asarray([r["candidate_types"][r["current_label_index"]] for r in rows])
    result = {}
    for value, code in (("true", 2), ("dontcare", 1)):
        positive = held & changed & (target_types == code)
        negative = held & (target_types != code) & np.asarray([code in r["candidate_types"] for r in rows])
        result[value+"_changed_recall"] = {"numerator": int(((choice == target) & positive).sum()), "denominator": int(positive.sum())}
        result[value+"_false_positive_rate"] = {"numerator": int(((selected_types == code) & negative).sum()), "denominator": int(negative.sum())}
    retained = held & ~changed
    result["retained_error"] = {"numerator": int(((choice != target) & retained).sum()), "denominator": int(retained.sum())}
    return result


def continuation(fits):
    """One fixed mechanism comparison; exact count fractions, no seed selection."""
    comparisons = []
    paired = []
    for seed in SEEDS:
        base, typed = (fits[f"{method}-{seed}"] for method in (BASELINE, CANDIDATE))
        b = base["cells"]["heldout_service/changed"]["nll"]["equal_service"]
        t = typed["cells"]["heldout_service/changed"]["nll"]["equal_service"]
        paired.append({"seed": seed, "flat": b, "typed": t, "difference": None if b is None or t is None else t-b})
    supported = all(p["flat"] is not None and p["typed"] is not None for p in paired)
    baseline = math.fsum(p["flat"] for p in paired)/3 if supported else None
    typed = math.fsum(p["typed"] for p in paired)/3 if supported else None
    comparisons.append({"name": "equal_service_changed_nll_reduction", "flat": baseline, "typed": typed,
                        "relative_reduction": (baseline-typed)/baseline if supported and baseline > 0 else None,
                        "threshold": 0.05, "passed": bool(supported and baseline > 0 and typed <= 0.95*baseline)})
    for p in paired:
        comparisons.append({"name": f"seed_{p['seed']}_nll_nonworse", "difference": p["difference"],
                            "passed": p["difference"] is not None and p["difference"] <= 0})
    decisions = {}
    for name, limit, minimum in (("true_changed_recall", Fraction(1, 20), True),
                                 ("dontcare_changed_recall", Fraction(1, 20), True),
                                 ("true_false_positive_rate", Fraction(1, 200), False),
                                 ("dontcare_false_positive_rate", Fraction(1, 200), False),
                                 ("retained_error", Fraction(1, 200), False)):
        entries = [{"seed": seed, **{m: fits[f"{m}-{seed}"]["decisions"][name] for m in (BASELINE, CANDIDATE)}} for seed in SEEDS]
        valid = all(e[m]["denominator"] > 0 for e in entries for m in (BASELINE, CANDIDATE))
        if valid:
            require(all(e[BASELINE]["denominator"] == e[CANDIDATE]["denominator"] for e in entries), "Paired decision denominator")
            rates = {m: sum((Fraction(e[m]["numerator"], e[m]["denominator"]) for e in entries), Fraction())/3 for m in (BASELINE, CANDIDATE)}
            delta = rates[CANDIDATE]-rates[BASELINE]
            passed = delta >= limit if minimum else delta <= limit
        else:
            rates, delta, passed = {m: None for m in (BASELINE, CANDIDATE)}, None, False
        decisions[name] = {"paired": entries, "mean_rates": {m: None if v is None else float(v) for m, v in rates.items()},
                           "difference": None if delta is None else float(delta)}
        comparisons.append({"name": name, "difference": None if delta is None else float(delta),
                            "threshold": float(limit), "relation": ">=" if minimum else "<=", "passed": bool(passed)})
    return {"comparison": CANDIDATE+" versus "+BASELINE, "paired_nll": paired, "decision_counts": decisions,
            "checks": comparisons, "checks_passed": sum(c["passed"] for c in comparisons),
            "total_checks": 9, "passed": all(c["passed"] for c in comparisons),
            "scope": "Prospective engineering screen; no significance or population guarantee; no alternative winner"}


def aggregate(rows, packets, references):
    validate_rows(rows)
    require(set(packets) == FITS, "All twelve final fits required")
    groups = {name: layout(rows, mask) for name, mask in group_masks(rows).items()}
    fits = {}
    for name in sorted(FITS):
        vectors, choice, error = predictions(rows, packets[name])
        services = {}
        for service in sorted({r["service"] for r in rows}):
            mask = np.asarray([r["service"] == service and r["derived_bin"] in BINS[2:] for r in rows])
            services[service] = {"heldout_service": next(r["heldout_service"] for r in rows if r["service"] == service), **describe(vectors, layout(rows, mask))}
        fits[name] = {"cells": {key: describe(vectors, group) for key, group in groups.items()},
                      "changed_per_service": services, "decisions": decision_counts(rows, choice), "maximum_mass_error": error}
    require(set(references) == {"row_indices", "previous_indices", "literal_indices", "type_frequency_indices"}, "Reference fields")
    expected = validate_rows(rows)
    require(references["row_indices"].dtype == np.int64 and np.array_equal(expected, references["row_indices"]), "Reference row identity")
    target = np.asarray([r["current_label_index"] for r in rows])
    counts = np.asarray([r["candidate_count"] for r in rows])
    refs = {}
    for method in ("previous", "literal", "type_frequency"):
        values = references[method+"_indices"]
        require(values.dtype == np.int64 and values.shape == expected.shape and ((values >= 0) & (values < counts)).all(), "Reference support")
        if method == "previous":
            require(np.array_equal(values, [r["previous_current_index"] for r in rows]), "Exact privileged prior reference")
        refs[method] = {key: describe({"accuracy": (values == target).astype(np.float64)}, group) for key, group in groups.items()}
    # All factorial cells and paired differences are descriptive, never alternative gates.
    contrasts = {}
    pairs = {"typing_stratum": ("typed_stratum", "flat_stratum"), "typing_balanced": ("typed_balanced", "flat_balanced"),
             "weighting_flat": ("flat_balanced", "flat_stratum"), "weighting_typed": ("typed_balanced", "typed_stratum")}
    for group in groups:
        contrasts[group] = {}
        for metric in ("accuracy", "nll", "brier"):
            contrasts[group][metric] = {}
            for weight in ("row", "equal_service", "equal_dialogue"):
                values = {m: [fits[f"{m}-{seed}"]["cells"][group][metric][weight] for seed in SEEDS] for m in METHODS}
                available = all(v is not None for v in values[METHODS[0]])
                result = {}
                for label, (a, b) in pairs.items():
                    delta = [x-y for x, y in zip(values[a], values[b], strict=True)] if available else [None]*3
                    result[label] = {"paired_differences": delta, "mean_difference": math.fsum(delta)/3 if available else None}
                interaction = [values["typed_balanced"][i]-values["flat_balanced"][i]-values["typed_stratum"][i]+values["flat_stratum"][i] for i in range(3)] if available else [None]*3
                result["interaction"] = {"paired_differences": interaction, "mean_difference": math.fsum(interaction)/3 if available else None}
                contrasts[group][metric][weight] = result
    rule = continuation(fits)
    return {"version": VERSION, "fits": fits, "references": refs, "descriptive_factorial_contrasts": contrasts,
            "continuation": rule, "continuation_allowed": rule["passed"], "groups_per_fit": len(groups),
            "scope": "Official TRAIN service-held-out development data, historically exposed in prior fits. Gold previous value supplied. References have accuracy only. All twelve fits retained; no recurrent, calibration, novelty or fresh-confirmation claim."}


def safe(base, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe manifest path")
    result = base/path
    require(result.resolve().is_relative_to(base.resolve()) and not result.is_symlink(), "Unsafe manifest member")
    return result


def check_manifest(directory, mapping, names, *, completion=True):
    require(set(mapping) == set(names), "Exact manifest membership")
    expected = set(names) | ({"completed.json"} if completion else set())
    require({p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} == expected,
            "Exact directory membership")
    for name, entry in mapping.items():
        path = safe(directory, name)
        require(set(entry) == {"sha256", "bytes"} and type(entry["bytes"]) is int
                and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Payload hash/size: "+name)


def normalization(record, rows, batches):
    require(set(record) == {"batches", "rows", "supported_candidates", "masked_candidates", "max_abs_mass_error",
                            "min_supported_log_prob", "max_supported_log_prob"}, "Normalization witness keys")
    require(record["batches"] == batches and record["rows"] == len(rows)
            and record["supported_candidates"] == sum(r["candidate_count"] for r in rows), "Normalization coverage")
    require(all(type(record[k]) is int and record[k] >= 0 for k in ("batches", "rows", "supported_candidates", "masked_candidates")), "Normalization counts")
    require(all(type(record[k]) in (int, float) and math.isfinite(record[k]) for k in
                ("max_abs_mass_error", "min_supported_log_prob", "max_supported_log_prob")), "Finite normalization witness")
    require(0 <= record["max_abs_mass_error"] <= TOLERANCE and record["min_supported_log_prob"] <= record["max_supported_log_prob"] <= 0,
            "Normalization witness limits")


def expected_types(query):
    result = []
    for cid, value in zip(query["candidate_ids"], query["candidate_values"], strict=True):
        if cid == "reserved:NOT_MENTIONED": code = 0
        elif cid == "reserved:DONTCARE": code = 1
        elif query["boolean_slot"]:
            require(value.strip().casefold() in ("true", "false"), "Boolean ontology")
            code = 2 if value.strip().casefold() == "true" else 3
        else: code = 4
        result.append(code)
    return result


def objective_from_rows(rows):
    """Fit-only objective: preserve one-third total weight in each stratum."""
    counts = [[0]*5 for _ in range(3)]
    for row in rows:
        stratum = 0 if row["derived_bin"] == BINS[0] else 1 if row["derived_bin"] == BINS[1] else 2
        counts[stratum][VALUES.index(row["current_value_group"])] += 1
    totals = [sum(c) for c in counts]
    require(all(totals), "Three nonempty fit strata required")
    categories = [sum(n > 0 for n in c) for c in counts]
    size = len(rows)
    return {"rows": size, "value_order": list(VALUES), "counts": counts, "stratum_counts": totals,
            "nonempty_categories": categories,
            "weights": {"stratum": [[size/(3*totals[s]) if count else None for count in counts[s]] for s in range(3)],
                        "balanced": [[size/(3*categories[s]*count) if count else None for count in counts[s]] for s in range(3)]}}


def validate_split_membership(admitted, fit_rows, evaluation_rows, heldout_services):
    """Verify every projected subpanel without dropping metadata fields."""
    held = set(heldout_services)
    fit_services = {row["service"] for row in fit_rows}
    primary = [row for row in evaluation_rows if row["service"] in held]
    secondary = [row for row in evaluation_rows if row["service"] not in held]
    expected = {"fit": fit_rows, "evaluation": evaluation_rows,
                "primary_heldout_service": primary, "secondary_nonheldout_service": secondary,
                "secondary_seen_in_fit": [row for row in secondary if row["service"] in fit_services],
                "secondary_absent_from_fit": [row for row in secondary if row["service"] not in fit_services]}
    require(type(admitted) is dict and set(admitted) == set(expected), "Original split projection keys")
    for name, rows in expected.items():
        values = admitted[name]
        require(type(values) is list and all(type(i) is int for i in values)
                and values == [row["row_index"] for row in rows], "Original split subpanel identity: "+name)


def correction_metadata(run_sha256):
    original = ROOT/"scripts/report_dialogue_typed.py"
    failed = ROOT/"output/dialogue-typed-v1/analysis-01/failed.json"
    require(digest(original) == ORIGINAL_REPORTER_SHA256, "Frozen original reporter identity")
    require(digest(failed) == FAILED_REPORT_SHA256, "Preserved failed report identity")
    failure = read(failed)
    require(failure["status"] == "failed" and failure["execution_completed_sha256"] == run_sha256
            and failure["model_calls"] == 0, "Failure belongs to unchanged training execution")
    return {"kind": "Saved-reader split-schema correction; no changed predictions, metrics or rule",
            "original_reporter_sha256": ORIGINAL_REPORTER_SHA256, "original_failed_report_sha256": FAILED_REPORT_SHA256,
            "corrected_reporter_sha256": digest(__file__),
            "corrected_tests_sha256": digest(ROOT/"tests/test_dialogue_typed_report_v2.py"),
            "original_failure_wall_seconds": failure["wall_seconds"], "training_rerun": False,
            "membership_fields_verified": ["fit", "evaluation", "primary_heldout_service", "secondary_nonheldout_service",
                                           "secondary_seen_in_fit", "secondary_absent_from_fit"]}


def authenticate_run(run, expected_sha):
    """Hash all payloads and opaque weights before loading any prediction array."""
    run = Path(run).resolve()
    require(digest(run/"completed.json") == expected_sha, "External run completion digest")
    done = read(run/"completed.json")
    plan, started = read(run/"plan.json"), read(run/"started.json")
    order = [f"{method}-{seed}" for i, seed in enumerate(SEEDS) for method in METHODS[i:]+METHODS[:i]]
    require(done["status"] == "completed" and done["phase"] == "train" and done["version"] == VERSION,
            "Complete typed training run required")
    require(done["completed_fits"] == done["expected_fits"] == plan["expected_fits"] == order, "All twelve fit identities/order")
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"} | {f"orders-{s}.npy" for s in SEEDS}
    names |= {f"fits/{fit}/{name}" for fit in FITS for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    check_manifest(run, done["files"], names)
    plan_sha = digest(run/"plan.json")
    require(done["plan_sha256"] == plan_sha and plan["version"] == VERSION and plan["config"] == CONFIG
            and plan["limits"] == done["limits"] == LIMITS, "Frozen recipe/limits")
    require(started["phase"] == "train" and started["version"] == VERSION and started["runtime"] == plan["runtime"] == done["runtime"], "Runtime witness agreement")
    sources = plan["source_sha256"]
    require(set(sources) == SOURCE_NAMES and sources == done["source_sha256"], "Exact 25-source closure")
    frozen = Path(started["request"]["plan"]).resolve().parent
    require(started["request"]["plan_sha256"] == plan_sha and digest(frozen/"plan.json") == plan_sha, "Published freeze plan identity")
    freeze = read(frozen/"completed.json")
    freeze_names = {"started.json", "plan.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+s for s in sources}
    require(freeze["status"] == "completed" and freeze["phase"] == "freeze" and freeze["plan_sha256"] == plan_sha, "Freeze completion")
    check_manifest(frozen, freeze["files"], freeze_names)
    for name, pin in sources.items():
        require(digest(safe(ROOT, name)) == digest(safe(frozen/"sources", name)) == pin, "Live/frozen source hash: "+name)
    prepared = Path(plan["prepared_path"]).resolve()
    require(digest(prepared/"completed.json") == plan["prepared_completed_sha256"], "Prepared metadata identity")
    prep = read(prepared/"completed.json")
    check_manifest(prepared, prep["files"], {"started.json", "catalog.json", "rows.jsonl", "summary.json"})
    canonical = {r["row_index"]: r for r in (decode(line) for line in (prepared/"rows.jsonl").read_text().splitlines())}
    catalog = read(prepared/"catalog.json")["queries"]
    rows = [decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    require(validate_rows(rows).tolist() == plan["evaluation_row_indices"], "Frozen evaluation row identities")
    held = set(plan["split"]["heldout_services"])
    for row in rows:
        expected = canonical[row["row_index"]]
        query = catalog[row["query_index"]]
        require(row == {**expected, "heldout_service": row["service"] in held, "candidate_types": expected_types(query)}, "Exact canonical evaluation join")
    fit_rows = [canonical[i] for i in plan["fit_row_indices"]]
    require(plan["fit_row_indices"] == sorted(set(plan["fit_row_indices"])), "Unique canonical fit rows")
    require(all(r["split"] == "train" and r["admission"] == "admitted" and r["service"] not in held for r in fit_rows), "Fit cohort")
    require(plan["objective"] == objective_from_rows(fit_rows), "Fit-only loss counts/weights")
    require(not ({r["dialogue_id"] for r in rows} & {r["dialogue_id"] for r in fit_rows}), "Fit/evaluation dialogue leakage")
    require(set(plan["fit_row_indices"]) | set(plan["evaluation_row_indices"]) == {i for i, r in canonical.items() if r["split"] == "train" and r["admission"] == "admitted"}, "All admitted TRAIN rows covered")
    split_root = ROOT/"output/dialogue-typed-v1/split-design-01"
    require(digest(split_root/"receipt.json") == plan["split_receipt_sha256"], "Original public service projection identity")
    split_receipt = read(split_root/"receipt.json")
    require(split_receipt["status"] == "completed" and split_receipt["alternate_splits_evaluated"] == 0, "Fixed single split")
    for name, item in split_receipt["files"].items():
        p = safe(split_root, name)
        require(digest(p) == item["sha256"] and p.stat().st_size == item["bytes"], "Split projection payload")
    membership = read(split_root/"membership.json")
    validate_split_membership(membership["admitted_row_indices"], fit_rows, rows, held)
    orders = {}
    for seed in SEEDS:
        item = plan["orders"][str(seed)]
        require(item["file"] == f"orders-{seed}.npy" and digest(run/item["file"]) == item["sha256"], "Frozen orders digest")
        values = np.load(run/item["file"], allow_pickle=False)
        require(values.dtype == np.int64 and values.shape == (20, len(fit_rows))
                and all(np.array_equal(np.sort(epoch), np.arange(len(fit_rows))) for epoch in values), "Complete epoch permutations")
        orders[seed] = values
    updates, evaluations = 20*math.ceil(len(fit_rows)/256), math.ceil(len(rows)/256)
    require(plan["updates_per_fit"] == updates and plan["evaluation_batches_per_fit"] == evaluations, "Full update/evaluation budget")
    counts = {"forward_attempted": updates+evaluations, "forward_returned": updates+evaluations,
              "backward_attempted": updates, "backward_returned": updates, "optimizer_attempted": updates,
              "optimizer_returned": updates, "training_rows": 20*len(fit_rows), "evaluation_rows": len(rows)}
    fit_receipts = {}
    for name in order:
        method, seed_text = name.rsplit("-", 1); seed = int(seed_text)
        dest = run/"fits"/name; record = read(dest/"completed.json")
        require(record["status"] == "completed" and record["method"] == method and record["seed"] == seed
                and record["plan_sha256"] == plan_sha and record["orders_sha256"] == plan["orders"][seed_text]["sha256"]
                and record["counts"] == counts, "Fit identity/budget: "+name)
        check_manifest(dest, record["files"], {"weights.pt", "updates.jsonl", "predictions.npz"})
        config = record["configuration"]
        for key, value in {"class": "DialogueTypedObservation", "version": "dialogue-typed-observation-v1", "mode": method.split("_")[0],
                           "attention_mode": "candidate", "input_dim": 384, "projection_dim": 64, "hidden_dim": 64,
                           "feature_dim": 401, "parameters": 173506, "attention_parameters": 73728,
                           "shared_scorer_parameters": 99778, "schema_pair": "[query;candidate]"}.items():
            require(config[key] == value, "Fit configuration: "+key)
        initial = record["initial_state_sha256"]
        require(type(initial) is str and len(initial) == 64 and all(c in "0123456789abcdef" for c in initial), "Initial state digest")
        journal = [decode(line) for line in (dest/"updates.jsonl").read_text().splitlines()]
        require(len(journal) == updates, "Journal update count")
        j = 0; aggregate_norm = {}; work = {}; journal_wall = 0.
        for epoch, values in enumerate(orders[seed]):
            for start in range(0, len(fit_rows), 256):
                batch = [fit_rows[int(i)] for i in values[start:start+256]]; entry = journal[j]; j += 1
                require(entry["epoch"] == epoch and entry["start"] == start
                        and entry["row_indices"] == [r["row_index"] for r in batch], "Exact paired journal coverage")
                require(math.isfinite(entry["weighted_loss"]) and entry["weighted_loss"] >= 0 and entry["wall_seconds"] > 0, "Finite paid update")
                journal_wall += entry["wall_seconds"]
                norm = entry["normalization"]; normalization(norm, batch, 1)
                require(norm["masked_candidates"] == len(batch)*max(r["candidate_count"] for r in batch)-sum(r["candidate_count"] for r in batch), "Batch padding coverage")
                for k, v in norm.items():
                    if k == "min_supported_log_prob": aggregate_norm[k] = min(aggregate_norm.get(k, v), v)
                    elif k in ("max_supported_log_prob", "max_abs_mass_error"): aggregate_norm[k] = max(aggregate_norm.get(k, v), v)
                    else: aggregate_norm[k] = aggregate_norm.get(k, 0)+v
                for k, v in entry["work"].items():
                    if not k.startswith("max_"): work[k] = work.get(k, 0)+v
        require(aggregate_norm == record["training_normalization"] and work == record["training_work"], "Training witness aggregate")
        normalization(record["evaluation"]["normalization"], rows, evaluations)
        eval_padding = sum(len(batch)*max(r["candidate_count"] for r in batch)-sum(r["candidate_count"] for r in batch)
                           for start in range(0, len(rows), 256) if (batch := rows[start:start+256]))
        require(record["evaluation"]["normalization"]["masked_candidates"] == eval_padding, "Evaluation padding coverage")
        train_wall, eval_wall = record["training_wall_seconds"], record["evaluation"]["wall_seconds"]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in
                    (train_wall, eval_wall, record["wall_seconds"], done["wall_seconds"])), "Finite timing witnesses")
        require(0 < journal_wall <= train_wall and 0 < eval_wall
                and train_wall+eval_wall <= record["wall_seconds"] <= done["wall_seconds"], "Fit timing scopes")
        fit_receipts[name] = record
    for seed in SEEDS:
        require(len({fit_receipts[f"{m}-{seed}"]["initial_state_sha256"] for m in METHODS}) == 1, "Four-arm full initialization pairing")
    require(done["progress"]["completed_fits"] == order and done["progress"]["active_fit"] is None
            and done["progress"]["totals"] == {k: 12*v for k, v in counts.items()}, "Root full work coverage")
    size = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    require(0 < done["wall_seconds"] <= LIMITS["wall_seconds"] and 0 < done["process_lifetime_peak_rss_bytes"] <= LIMITS["rss_bytes"]
            and size <= LIMITS["output_bytes"] and sum(r["wall_seconds"] for r in fit_receipts.values()) <= done["wall_seconds"], "Whole-run cost bounds")
    return rows, done, plan, fit_receipts, size


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def report_text(summary):
    rule = summary["continuation"]
    lines = ["# Typed/support conditional development study", "", summary["scope"], "",
             f"Fixed mechanism screen: **{'PASS' if rule['passed'] else 'FAIL'}**, {rule['checks_passed']}/{rule['total_checks']} checks passed.",
             "Technical completion and source/row/weight bindings passed for all twelve fits.", "",
             "| Arm | Seed | Held-out changed equal-service NLL | TRUE recall | DONTCARE recall | Retained error |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for seed in SEEDS:
            fit = summary["fits"][f"{method}-{seed}"]
            loss = fit["cells"]["heldout_service/changed"]["nll"]["equal_service"]
            rates = []
            for name in ("true_changed_recall", "dontcare_changed_recall", "retained_error"):
                c = fit["decisions"][name]
                rates.append(f"{c['numerator']}/{c['denominator']}" if c["denominator"] else "undefined (0 rows)")
            lines.append(f"| {method} | {seed} | {loss if loss is not None else 'undefined'} | "+" | ".join(rates)+" |")
    lines += ["", "All panel, transition/value, service and dialogue means and factorial contrasts are in summary.json.",
              "Reference outputs are accuracy-only. Three seeds are repeated optimizations, not independent service samples.",
              "Runtime, normalization extrema and initializer digests are authenticated execution witnesses; no model or checkpoint replay was performed."]
    costs = summary["costs"]
    lines += ["", f"Whole-run wall time: {costs['whole_wall_seconds']:.6f} seconds. Process-lifetime peak RSS: {costs['process_lifetime_peak_rss_bytes']} bytes.",
              "", "| Fit | Training seconds | Evaluation seconds | Fit seconds |",
              "|---|---:|---:|---:|"]
    for name, cost in costs["per_fit"].items():
        lines.append(f"| {name} | {cost['training_wall_seconds']:.6f} | {cost['evaluation_wall_seconds']:.6f} | {cost['wall_seconds']:.6f} |")
    lines += ["", costs["scope"]]
    return "\n".join(lines)+"\n"


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        run = Path(args.run).resolve()
        correction = correction_metadata(args.run_sha256)
        rows, done, plan, fits, size = authenticate_run(run, args.run_sha256)
        packets = {}
        for name in FITS:
            with np.load(run/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                packets[name] = {k: archive[k] for k in archive.files}
        with np.load(run/"references.npz", allow_pickle=False) as archive:
            refs = {k: archive[k] for k in archive.files}
        type_counts = np.asarray(plan["objective"]["counts"], dtype=np.int64).sum(0)
        frequency_choices = []
        for row in rows:
            types = row["candidate_types"]
            frequency_choices.append(int(np.argmax([(type_counts[t]+1)/types.count(t) for t in types])))
        require(np.array_equal(refs["type_frequency_indices"], np.asarray(frequency_choices, dtype=np.int64)), "Fit-only type-frequency reference")
        summary = aggregate(rows, packets, refs)
        summary.update(status="completed", analysis_correction=correction, technical_validity_passed=True, execution_completed_sha256=args.run_sha256,
                       plan_sha256=done["plan_sha256"], source_sha256=done["source_sha256"], split=plan["split"],
                       objective=plan["objective"], costs={"whole_wall_seconds": done["wall_seconds"],
                       "process_lifetime_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"], "execution_bytes": size,
                       "scope": "Training includes optimizer setup, training and checkpoint write; evaluation includes actor/model evaluation and prediction write; fit adds receipt preparation/payload hashing. Whole run also pays shared authentication, cache load and initialization. RSS is process-lifetime high-water, not a sum of fit peaks. Inherited preprocessing/preflight are separate costs, not included or claimed free.",
                       "per_fit": {k: {"wall_seconds": v["wall_seconds"], "training_wall_seconds": v["training_wall_seconds"],
                                      "evaluation_wall_seconds": v["evaluation"]["wall_seconds"], "counts": v["counts"], "training_work": v["training_work"],
                                      "evaluation_work": v["evaluation"]["work"]} for k, v in fits.items()}})
        check_manifest(run, done["files"], set(done["files"]))
        require(digest(run/"completed.json") == args.run_sha256, "Completion changed during reporting")
        require(all(digest(safe(ROOT, name)) == pin for name, pin in done["source_sha256"].items()), "End source stability")
        write(out/"summary.json", summary)
        with (out/"report.md").open("x") as stream:
            stream.write(report_text(summary))
            stream.write("\nThis corrected saved reader validates all six split-projection fields. The original pre-scoring failure is preserved; no training or metric changes occurred.\n")
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "execution_completed_sha256": args.run_sha256,
              "source_sha256": digest(__file__), "analysis_correction": correction, "plan_sha256": done["plan_sha256"], "technical_validity_passed": True,
              "continuation_allowed": summary["continuation_allowed"], "execution_files": len(done["files"])+1,
              "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()},
              "wall_seconds": time.monotonic()-started, "model_calls": 0,
              "scope": "Saved-only analysis. Weights hashed, never deserialized. Exact prior and fit-type-frequency references reconstructed; literal reference inherited from hashed execution. Source/test-bound actor semantics; execution clocks and state/gradient checks are recorded witnesses."})
        return summary["continuation_allowed"]
    except BaseException as error:
        try: write(out/"failed.json", {"status": "failed", "error": repr(error), "execution_completed_sha256": args.run_sha256,
                                     "wall_seconds": time.monotonic()-started, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original reporting error
            if callable(getattr(error, "add_note", None)): error.add_note("Failure receipt error: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps({"continuation_allowed": execute(parser.parse_args())}))
