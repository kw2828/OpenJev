"""Saved-only six-fit objective report. No producer/model imports or float caches.

Hash-pinned alignment/typed readers supply metadata and work primitives. New
readout construction, aggregation and thirteen decisions are defined here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import signal
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/report_dialogue_alignment.py"
HELPER_PIN = "68674dfd7ef456a6353a0567a9b9e54e2dc1cdbc2d9d59fc25ca92ee2d45539e"
VERSION = "dialogue-objective-v1"
METHODS, SEEDS = ("stratum", "uniform"), (6201, 6202, 6203)
FIT_ORDER = ["stratum-6201", "uniform-6201", "uniform-6202", "stratum-6202", "stratum-6203", "uniform-6203"]
CATEGORIES = ("stratum-original", "stratum-corrected", "uniform-original", "uniform-reweighted")
COUNTS = (17666, 9246, 2299)
STRATA = ("all", "changed", "retained", "unmentioned_retention", "assigned_retention")
PARENT_PIN = "e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1"
PARENT_FREEZE_PIN = "8a2fb2659efed6ac187e92095eef3b3214beacf4ddcbcd7101664c9ef4c57e5f"
HISTORICAL = {
    "summary": ("output/dialogue-weight-prior-v1/diagnostic-01/summary.json", "97e6388f89020628854b3274b8c6d05f8524ea69e7a31931738840ff1bbbd961"),
    "receipt": ("output/dialogue-weight-prior-v1/diagnostic-01/receipt.json", "a0b5209b81a81f33bcebb833b3badc8e7fcf4144ceda5eb2f64bd03adeb67975"),
    "audit": ("output/dialogue-weight-prior-v1/audit-01/receipt.json", "b1f802b477fddfc3c11bf304588753780536d580ec36037acba04290bc96d771"),
}
REPORT_LIMITS = {"wall_seconds": 60., "rss_bytes": 1024**3, "output_bytes": 64*1024**2}
LIMITS = {"wall_seconds": 6000., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
FREEZE_LIMITS = {"wall_seconds": 60., "rss_bytes": 1024**3, "output_bytes": 128*1024**2}
NEW_SOURCES = {"scripts/study_dialogue_objective.py", "tests/test_study_dialogue_objective.py",
               "scripts/report_dialogue_objective.py", "tests/test_report_dialogue_objective.py",
               "scripts/audit_dialogue_objective.py", "tests/test_audit_dialogue_objective.py",
               "scripts/audit_dialogue_commitment.py", "tests/test_audit_dialogue_commitment.py",
               "research/dialogue-objective-protocol.md"}
SCOPE = ("Six fresh fits on historically exposed official TRAIN, with privileged gold previous values. "
         "Four fixed readouts per seed; no selection or new architecture claim. Earlier 18/22 FAIL is unchanged. "
         "Main authentication hashes all execution payloads including opaque weights, and decodes prepared/split "
         "metadata plus authenticated schema index/integer offsets for work reconstruction. No float cache, "
         "encoder, producer, model or checkpoint deserializer is loaded. Execution/initializer/normalization "
         "receipts are source-bound witnesses, not training replay. Historical metrics are pinned references, "
         "not contemporaneous equal-capacity fits; historical row-level paired repairs are unavailable.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


require(digest(ROOT/HELPER) == HELPER_PIN, "Frozen saved-only helper")
_spec = importlib.util.spec_from_file_location("_objective_saved_helper", ROOT/HELPER)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
base = old.base
read, write, safe, check_manifest = base.read, base.write, base.safe, base.check_manifest
CONFIG = {**old.CONFIG, "methods": list(METHODS)}


def average(trees):
    first = trees[0]
    if isinstance(first, dict):
        require(all(set(t) == set(first) for t in trees), "Mean schema")
        return {k: average([t[k] for t in trees]) for k in first}
    if first is None:
        require(all(t is None for t in trees), "Undefined mean support")
        return None
    require(all(t is not None and math.isfinite(t) for t in trees), "Finite mean")
    return math.fsum(trees)/len(trees)


def difference(a, b):
    if isinstance(a, dict):
        require(set(a) == set(b), "Paired metric schema")
        return {k: difference(a[k], b[k]) for k in a}
    require((a is None) == (b is None), "Paired support")
    return None if a is None else a-b


def layouts(rows):
    labels = np.asarray([r["current_label_index"] for r in rows])
    previous = np.asarray([r["previous_current_index"] for r in rows])
    types = np.full((len(rows), 12), -1, np.int64)
    for i, row in enumerate(rows): types[i, :row["candidate_count"]] = row["candidate_types"]
    target_types = types[np.arange(len(rows)), labels]
    changed = labels != previous
    masks = dict(zip(STRATA, (np.ones(len(rows), bool), changed, ~changed,
                             ~changed & (target_types == 0), ~changed & (target_types != 0)), strict=True))
    held = np.asarray([r["heldout_service"] for r in rows])
    cells = {f"{p}/{s}": base.layout(rows, pop & mask) for p, pop in
             (("all", np.ones(len(rows), bool)), ("seen_service", ~held), ("heldout_service", held)) for s, mask in masks.items()}
    names = np.asarray([r["service"] for r in rows])
    services = {str(name): {s: base.layout(rows, (names == name) & mask) for s, mask in masks.items()} for name in np.unique(names)}
    return {"labels": labels, "previous": previous, "types": types, "target_types": target_types}, cells, services


def readouts(raw, sizes, previous, types, method):
    """Only public support, prior and candidate types determine correction."""
    require(method in METHODS, "Objective identity")
    require(raw.dtype == np.float32 and raw.shape == (len(sizes), 12), "Raw log shape/dtype")
    mask = np.arange(12)[None] < sizes[:, None]
    require(np.isfinite(raw[mask]).all() and (raw[mask] <= 0).all() and np.isneginf(raw[~mask]).all(), "Raw log support")
    require(previous.shape == sizes.shape and previous.dtype.kind in "iu" and ((previous >= 0) & (previous < sizes)).all(), "Public prior")
    require(types.shape == raw.shape and types.dtype.kind in "iu" and (types[~mask] == -1).all()
            and ((types[mask] >= 0) & (types[mask] <= 4)).all(), "Public types")
    raw64 = raw.astype(np.float64)
    mass_error = float(np.abs(np.exp(raw64).sum(1)-1).max())
    require(mass_error <= 2e-6, "Raw float32 normalization")
    z = np.logaddexp.reduce(raw64, axis=1)
    original = raw64-z[:, None]
    weights = sum(COUNTS)/(3*np.asarray(COUNTS, np.float64))
    logw = np.full(raw.shape, np.log(weights[2]), np.float64)
    logw[np.arange(len(raw)), previous] = np.log(weights[np.where(types[np.arange(len(raw)), previous] == 0, 0, 1)])
    adjusted = raw64 + (-1 if method == "stratum" else 1)*logw
    adjusted -= np.logaddexp.reduce(adjusted, axis=1)[:, None]
    result = {"original": original, "corrected" if method == "stratum" else "reweighted": adjusted}
    require(all(np.isfinite(v[mask]).all() and np.isneginf(v[~mask]).all()
                and float(np.abs(np.exp(v).sum(1)-1).max()) <= 1e-12 for v in result.values()), "Float64 readout normalization")
    require(np.array_equal(original.argmax(1), raw.argmax(1)), "Original hard choice identity")
    return result, {"raw_maximum_mass_error": mass_error, "max_abs_log_z": float(np.abs(z).max()),
                    "mean_log_z": float(z.mean())}, z


def vectors(logs, meta):
    choice = logs.argmax(1); target = meta["labels"]
    selected = meta["types"][np.arange(len(logs)), choice]
    correct = choice == target
    branch = np.minimum(selected, 2) != np.minimum(meta["target_types"], 2)
    residual = np.exp(logs); residual[np.arange(len(logs)), target] -= 1
    values = {"accuracy": correct.astype(float), "error": (~correct).astype(float),
              "wrong_selected_branch": branch.astype(float), "wrong_value": (~correct & ~branch).astype(float),
              "nll": -logs[np.arange(len(logs)), target], "brier": np.square(residual).sum(1)}
    ties = (logs == logs.max(1, keepdims=True)).sum(1) > 1
    return values, choice, selected, ties


def describe(values, choice, selected, ties, meta, group):
    ids = group["indices"]; rare = {}
    for name, code in (("true", 2), ("dontcare", 1)):
        for kind in ("recall", "false_positive"):
            eligible = meta["target_types"] == code if kind == "recall" else (
                (meta["target_types"] != code) & (meta["types"] == code).any(1))
            hit = choice == meta["labels"] if kind == "recall" else selected == code
            n, d = int((eligible & hit)[ids].sum()), int(eligible[ids].sum())
            rare[name+"_"+kind] = {"numerator": n, "denominator": d, "rate": n/d if d else None}
    return {"rows": len(ids), "counts": {k: int(values[v][ids].sum()) for k, v in
            (("correct", "accuracy"), ("error", "error"), ("wrong_selected_branch", "wrong_selected_branch"), ("wrong_value", "wrong_value"))},
            "rare": rare, "metrics": {k: base.means(v, group) for k, v in values.items()}, "exact_tie_rows": int(ties[ids].sum())}


def paired_cell(a, b, ca, cb, labels, group):
    ids = group["indices"]; ac, bc = ca[ids] == labels[ids], cb[ids] == labels[ids]
    return {"rows": len(ids), "count_differences": difference(a["counts"], b["counts"]),
            "metric_differences": difference(a["metrics"], b["metrics"]),
            "rare_rate_differences": {k: difference(a["rare"][k]["rate"], b["rare"][k]["rate"]) for k in a["rare"]},
            "paired": {"rows": len(ids), "wrong_to_correct": int((ac & ~bc).sum()), "correct_to_wrong": int((~ac & bc).sum()),
                       "both_correct": int((ac & bc).sum()), "both_wrong": int((~ac & ~bc).sum())}}


def exact_rate(fit, stratum, metric, weighting, heldout_services):
    def value(cell):
        n, d = cell["counts"]["correct" if metric == "accuracy" else "error"], cell["rows"]
        require(type(n) is int and type(d) is int and 0 <= n <= d, "Exact decision counts")
        return Fraction(n, d) if d else None
    if weighting == "row": return value(fit["cells"]["heldout_service/"+stratum])
    parts = [value(fit["services"][s][stratum]) for s in heldout_services]
    parts = [p for p in parts if p is not None]
    return sum(parts, Fraction())/len(parts) if parts else None


def continuation(fits, historical, heldout_services):
    require(set(fits) == {f"{c}-{s}" for c in CATEGORIES for s in SEEDS}, "All twelve fixed readouts")
    require(bool(heldout_services), "Primary service support")
    checks = {}
    def delta(a, b, stratum, metric, weighting):
        if metric == "nll":
            av, bv = (f["cells"]["heldout_service/"+stratum]["metrics"][metric][weighting] for f in (a, b))
            return av-bv if av is not None and bv is not None and math.isfinite(av) and math.isfinite(bv) else None
        av, bv = (exact_rate(f, stratum, metric, weighting, heldout_services) for f in (a, b))
        return av-bv if av is not None and bv is not None else None
    def record(name, ds, relation, threshold):
        supported = all(d is not None for d in ds)
        mean = (sum(ds, Fraction())/3 if all(isinstance(d, Fraction) for d in ds) else math.fsum(ds)/3) if supported else None
        passed = supported and (mean >= threshold if relation == ">=" else mean <= threshold)
        checks[name] = {"passed": bool(passed), "seed_differences": [None if d is None else float(d) for d in ds],
                        "mean_difference": None if mean is None else float(mean), "relation": relation, "threshold": float(threshold)}
    for scope, comparator in (("objective", "stratum-corrected"), ("practical", "flat_stratum-corrected")):
        pairs = [(fits[f"uniform-original-{s}"], (fits if scope == "objective" else historical["fits"])[f"{comparator}-{s}"]) for s in SEEDS]
        rules = (("changed", "accuracy", ">=", Fraction(1, 50)), ("retained", "error", "<=", Fraction(0)), ("all", "nll", "<=", 0.)) if scope == "objective" else (
            ("all", "accuracy", ">=", Fraction(1, 400)), ("all", "nll", "<=", 0.))
        for stratum, metric, relation, threshold in rules:
            for weighting in ("row", "equal_service"):
                label = ("changed_accuracy" if stratum == "changed" else "retained_error" if stratum == "retained" else metric)
                record(scope+"_"+label+"_"+weighting, [delta(a, b, stratum, metric, weighting) for a, b in pairs], relation, threshold)
        if scope == "practical":
            ds = []
            for a, b in pairs:
                ca, cb = (f["cells"]["heldout_service/all"]["rare"]["true_false_positive"] for f in (a, b))
                require(ca["denominator"] == cb["denominator"], "Historical TRUE support")
                ds.append(Fraction(ca["numerator"]-cb["numerator"], ca["denominator"]) if ca["denominator"] else None)
            record("practical_true_false_positive", ds, "<=", Fraction(1, 200))
        common = []
        for seed, (a, b) in zip(SEEDS, pairs, strict=True):
            conditions = []
            for weighting in ("row", "equal_service"):
                changed = delta(a, b, "changed" if scope == "objective" else "all", "accuracy", weighting)
                harm = delta(a, b, "retained" if scope == "objective" else "all", "error" if scope == "objective" else "nll", weighting)
                conditions.extend((changed is not None and changed > 0, harm is not None and harm <= 0))
            common.append({"seed": seed, "passed": bool(all(conditions))})
        checks[scope+"_joint_seeds"] = {"passed": sum(v["passed"] for v in common) >= 2, "seeds": common,
                                       "common_seeds": sum(v["passed"] for v in common), "required": 2}
    require(len(checks) == 13, "Exactly thirteen decisions")
    objective = all(c["passed"] for n, c in checks.items() if n.startswith("objective_"))
    practical = all(c["passed"] for n, c in checks.items() if n.startswith("practical_"))
    return {"passed": objective and practical, "objective_passed": objective, "practical_passed": practical,
            "checks_passed": sum(c["passed"] for c in checks.values()), "total_checks": 13, "checks": checks}


def aggregate(rows, packets, references, historical):
    require(set(packets) == set(FIT_ORDER), "All six final prediction packets required")
    ids = base.validate_rows(rows); meta, cells, services = layouts(rows)
    sizes = np.asarray([r["candidate_count"] for r in rows])
    fits, choices, source_norm = {}, {}, {}
    for name in FIT_ORDER:
        method, seed = name.rsplit("-", 1); packet = packets[name]
        require(set(packet) == {"row_indices", "log_probs"} and packet["row_indices"].dtype == np.int64
                and np.array_equal(packet["row_indices"], ids), "Atomic prediction row alignment")
        outputs, witness, z = readouts(packet["log_probs"], sizes, meta["previous"], meta["types"], method)
        source_norm[name] = {**witness, "nll_normalized_minus_original_raw": {k: base.means(z, g) for k, g in cells.items()}}
        for readout, logs in outputs.items():
            key = f"{method}-{readout}-{seed}"; values, choice, selected, ties = vectors(logs, meta)
            choices[key] = choice
            fits[key] = {"source_fit": name, "method": method, "readout": readout, "seed": int(seed),
                "validation": {"maximum_mass_error": float(np.abs(np.exp(logs).sum(1)-1).max()), "exact_tie_rows": int(ties.sum())},
                "cells": {k: describe(values, choice, selected, ties, meta, g) for k, g in cells.items()},
                "services": {s: {k: describe(values, choice, selected, ties, meta, g) for k, g in groups.items()} for s, groups in services.items()}}
    pairs = {}
    for label, a, b in (("primary", "uniform-original", "stratum-corrected"), ("secondary", "uniform-reweighted", "stratum-original")):
        seeds = {}
        for seed in SEEDS:
            ak, bk = f"{a}-{seed}", f"{b}-{seed}"
            seeds[str(seed)] = {"cells": {k: paired_cell(fits[ak]["cells"][k], fits[bk]["cells"][k], choices[ak], choices[bk], meta["labels"], g) for k, g in cells.items()},
                "services": {s: {k: paired_cell(fits[ak]["services"][s][k], fits[bk]["services"][s][k], choices[ak], choices[bk], meta["labels"], g)
                                 for k, g in groups.items()} for s, groups in services.items()}}
        pairs[label] = {"seeds": seeds, "mean": average(list(seeds.values()))}
    require(set(references) == {"row_indices", "previous_indices", "literal_indices"}
            and references["row_indices"].dtype == np.int64 and np.array_equal(references["row_indices"], ids), "Reference identity")
    refs = {}
    for name in ("previous", "literal"):
        choice = references[name+"_indices"]
        require(choice.dtype == np.int64 and choice.shape == ids.shape and ((choice >= 0) & (choice < sizes)).all(), "Reference support")
        if name == "previous": require(np.array_equal(choice, meta["previous"]), "Exact previous reference")
        values = (choice == meta["labels"]).astype(float)
        refs[name] = {"cells": {k: {"rows": g["rows"], "accuracy": base.means(values, g)} for k, g in cells.items()},
                      "scope": "Accuracy only; privileged exact previous value" if name == "previous" else "Accuracy only; literal construction is source-bound, not replayed from float lexical cache"}
    historical_differences = {category: {oldcategory: {str(seed): {
        "cells": {k: difference(fits[f"{category}-{seed}"]["cells"][k]["metrics"], historical["fits"][f"{oldcategory}-{seed}"]["cells"][k]["metrics"]) for k in cells},
        "services": {s: {k: difference(fits[f"{category}-{seed}"]["services"][s][k]["metrics"], historical["fits"][f"{oldcategory}-{seed}"]["services"][s][k]["metrics"])
                          for k in groups} for s, groups in services.items()}} for seed in SEEDS}
        for oldcategory in historical["means"]} for category in CATEGORIES}
    held = sorted({r["service"] for r in rows if r["heldout_service"]})
    return {"version": VERSION, "scope": SCOPE, "fits": fits, "pairs": pairs, "source_normalization": source_norm,
            "means": {c: {section: average([fits[f"{c}-{s}"][section] for s in SEEDS]) for section in ("cells", "services")} for c in CATEGORIES},
            "historical": historical, "historical_metric_differences": historical_differences, "references": refs,
            "continuation": continuation(fits, historical, held), "correction_weights": {"fit_counts": list(COUNTS),
                "analytic_float64": (sum(COUNTS)/(3*np.asarray(COUNTS, np.float64))).tolist(),
                "training_float32_representation": (sum(COUNTS)/(3*np.asarray(COUNTS, np.float64))).astype(np.float32).astype(float).tolist()},
            "mean_scope": "Equal three-seed averages on identical rows, not independent new examples. Equal-service metrics omit only services with zero rows in the declared stratum; undefined global denominators cannot pass."}


def historical_inputs(rows_digest):
    for filename, pin in HISTORICAL.values(): require(digest(ROOT/filename) == pin, "Historical external digest")
    summary, receipt, audit = (read(ROOT/HISTORICAL[k][0]) for k in ("summary", "receipt", "audit"))
    require(receipt["status"] == audit["status"] == "completed" and audit["agreement"] is True
            and receipt["files"]["summary.json"]["sha256"] == HISTORICAL["summary"][1]
            and receipt["input_sha256"]["rows"] == rows_digest, "Historical same rows and successful audit")
    categories = {f"{m}-{r}" for m in ("flat_stratum", "token_mean", "token_aligned") for r in ("original", "corrected")}
    require(set(summary["fits"]) == {f"{c}-{s}" for c in categories for s in SEEDS}
            and set(summary["means"]) == categories, "All historical references")
    return {"fits": summary["fits"], "means": summary["means"], "provenance": {k: {"path": p, "sha256": h} for k, (p, h) in HISTORICAL.items()},
            "cost_scope": "Historical prediction fits remain their original collection, not new equal-capacity measurements; diagnostic timing below is analysis only.",
            "diagnostic_wall_seconds": receipt["wall_seconds"], "diagnostic_peak_rss_bytes": receipt["process_lifetime_peak_rss_bytes"]}


def metadata(run, plan):
    """Authenticate metadata bytes before decoding; never open float caches."""
    prepared = Path(plan["prepared_path"])
    require(digest(prepared/"completed.json") == plan["prepared_completed_sha256"], "Prepared metadata identity")
    prep = read(prepared/"completed.json")
    check_manifest(prepared, prep["files"], {"started.json", "catalog.json", "rows.jsonl", "summary.json"})
    canonical = {r["row_index"]: r for r in (base.decode(s) for s in (prepared/"rows.jsonl").read_text().splitlines())}
    catalog = read(prepared/"catalog.json")["queries"]
    rows = [base.decode(s) for s in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    require(base.validate_rows(rows).tolist() == plan["evaluation_row_indices"], "Frozen evaluation row order")
    held = set(plan["split"]["heldout_services"])
    for row in rows:
        require(row == {**canonical[row["row_index"]], "heldout_service": row["service"] in held,
                        "candidate_types": base.expected_types(catalog[row["query_index"]])}, "Canonical metadata join")
    fit = [canonical[i] for i in plan["fit_row_indices"]]
    require(plan["fit_row_indices"] == sorted(set(plan["fit_row_indices"])) and len(fit) == 29211 and len(rows) == 13599
            and all(r["split"] == "train" and r["admission"] == "admitted" and r["service"] not in held for r in fit)
            and not ({r["dialogue_id"] for r in fit} & {r["dialogue_id"] for r in rows}), "Disjoint fixed TRAIN cohorts")
    require(set(plan["fit_row_indices"]) | set(plan["evaluation_row_indices"]) ==
            {i for i, r in canonical.items() if r["split"] == "train" and r["admission"] == "admitted"}, "Complete admitted TRAIN")
    objective = base.objective_from_rows(fit)
    require(plan["objective"] == objective and objective["stratum_counts"] == list(COUNTS), "Fit-only objective counts")
    split = ROOT/"output/dialogue-typed-v1/split-design-01"
    require(digest(split/"receipt.json") == plan["split_receipt_sha256"], "Original split pin")
    split_done = read(split/"receipt.json")
    for name, item in split_done["files"].items():
        p = safe(split, name)
        require(p.stat().st_size == item["bytes"] and digest(p) == item["sha256"], "Split bytes")
    base.validate_split_membership(read(split/"membership.json")["admitted_row_indices"], fit, rows, held)
    _, groups, _ = layouts(rows)
    require([groups["heldout_service/"+s]["rows"] for s in STRATA] == [7819, 578, 7241, 4032, 3209], "Primary fixed support")
    schema = Path(plan["schema_cache_path"])
    require(plan["schema_completed_sha256"] == old.SCHEMA_PIN and digest(schema/"completed.json") == old.SCHEMA_PIN, "Schema completion pin")
    cached = read(schema/"completed.json")
    for name in ("index.json", "offsets.npy"):
        item = cached["files"][name]; p = schema/name
        require(p.stat().st_size == item["bytes"] and digest(p) == item["sha256"], "Schema integer metadata identity")
    index = read(schema/"index.json"); offsets = np.load(schema/"offsets.npy", allow_pickle=False)
    require(offsets.dtype == np.int64 and offsets.shape == (361,) and offsets[0] == 0 and (np.diff(offsets) > 0).all(), "Schema offsets")
    queries = {q["query_index"]: q for q in index["queries"]}
    require(len(queries) == 53 and index["candidate_occurrences"] == 307 and index["unique_texts"] == 360
            and set(queries) == {r["query_index"] for r in fit+rows}, "Complete supplied schema coverage")
    for row in fit+rows:
        q = queries[row["query_index"]]
        require(q["candidate_feature_indices"] == row["cache"]["candidate_feature_indices"]
                and q["candidate_ids"] == catalog[row["query_index"]]["candidate_ids"], "Schema candidate alignment")
    lengths = {qi: [int(offsets[i+1]-offsets[i]) for i in q["candidate_token_ids"]] for qi, q in queries.items()}
    return fit, rows, lengths, cached


def authenticate_run(run, expected_sha):
    """Reject incomplete/corrupt receipts before any prediction decoding."""
    run = Path(run).resolve()
    require(digest(run/"completed.json") == expected_sha, "External completed digest")
    done, plan, start = (read(run/name) for name in ("completed.json", "plan.json", "started.json"))
    require(done["status"] == "completed" and done["phase"] == "train" and done["version"] == VERSION
            and done["completed_fits"] == done["expected_fits"] == plan["expected_fits"] == FIT_ORDER, "Complete six-fit campaign")
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"} | {f"orders-{s}.npy" for s in SEEDS}
    names |= {f"fits/{fit}/{name}" for fit in FIT_ORDER for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    check_manifest(run, done["files"], names)
    require(len(names) == 31, "Exact 32-file execution")
    plan_sha = digest(run/"plan.json")
    require(done["plan_sha256"] == plan_sha and plan["version"] == VERSION and plan["config"] == CONFIG
            and done["limits"] == plan["limits"] == LIMITS and plan["freeze_limits"] == FREEZE_LIMITS, "Recipe and caps")
    require(plan["original_capacity_admitted"] is False and plan["quality_metrics_in_runner"] is False
            and plan["no_retry"] is done["no_retry"] is True and done["quality_metrics_computed"] is False
            and done["encoder_calls"] == 0 and done["official_dev_inference"] is False and done["test_contents_accessed"] is False,
            "Closed training scope")
    require(start["version"] == VERSION and start["phase"] == "train" and start["runtime"] == plan["runtime"] == done["runtime"], "Runtime witnesses")
    frozen = Path(start["request"]["plan"]).resolve().parent
    require(start["request"]["plan_sha256"] == plan_sha and digest(frozen/"plan.json") == plan_sha, "Published plan identity")
    freeze = read(frozen/"completed.json"); sources = plan["source_sha256"]
    require(freeze["status"] == "completed" and freeze["phase"] == "freeze" and freeze["plan_sha256"] == plan_sha
            and freeze["version"] == VERSION and freeze["runtime"] == plan["runtime"] and freeze["limits"] == FREEZE_LIMITS
            and freeze["model_calls"] == freeze["encoder_calls"] == 0
            and sources == freeze["source_sha256"] == done["source_sha256"], "Successful frozen source identity")
    fnames = {"started.json", "plan.json", "parent-plan.json", "parent-completed.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+n for n in sources}
    check_manifest(frozen, freeze["files"], fnames)
    require(0 <= freeze["wall_seconds"] <= FREEZE_LIMITS["wall_seconds"]
            and 0 < freeze["process_lifetime_peak_rss_bytes"] <= FREEZE_LIMITS["rss_bytes"]
            and sum(p.stat().st_size for p in frozen.rglob("*") if p.is_file()) <= FREEZE_LIMITS["output_bytes"], "Freeze caps")
    for name, pin in sources.items(): require(digest(safe(ROOT, name)) == digest(safe(frozen/"sources", name)) == pin, "Live/frozen source: "+name)
    parent_path = Path(plan["parent_plan_path"])
    require(plan["parent_plan_sha256"] == done["parent_plan_sha256"] == PARENT_PIN
            and digest(parent_path) == digest(frozen/"parent-plan.json") == PARENT_PIN
            and plan["parent_freeze_completed_sha256"] == PARENT_FREEZE_PIN
            and digest(parent_path.parent/"completed.json") == digest(frozen/"parent-completed.json") == PARENT_FREEZE_PIN, "Parent freeze lineage")
    parent = read(parent_path)
    changed = {"version", "config", "limits", "source_sha256", "allocation", "expected_fits", "initialization"}
    require(all(plan[k] == v for k, v in parent.items() if k not in changed), "Exact inherited metadata")
    require(set(sources) == set(parent["source_sha256"]) | NEW_SOURCES and
            all(sources[k] == v for k, v in parent["source_sha256"].items()), "Exact transitive source closure")
    require(sources[HELPER] == HELPER_PIN and sources[old.METRIC_SOURCE] == old.METRIC_PIN, "Executed helper pins")
    require(plan["model_method"] == "token_aligned" and plan["objective_weightings"] == {
        "stratum": "Inherited fixed weights; sum weighted CE divided by effective row count",
        "uniform": "Every training row weight exactly one; same effective row denominator"}, "Two objective definitions")
    fit_rows, rows, lengths, cached = metadata(run, plan)
    orders = {}
    for seed in SEEDS:
        item = plan["orders"][str(seed)]; filename = f"orders-{seed}.npy"
        require(item == parent["orders"][str(seed)] and item["file"] == filename
                and all(digest(p/filename) == item["sha256"] for p in (run, frozen, parent_path.parent)), "Byte-identical paired orders")
        values = np.load(run/filename, allow_pickle=False)
        require(values.dtype == np.int64 and values.shape == (20, len(fit_rows))
                and all(np.array_equal(np.sort(v), np.arange(len(fit_rows))) for v in values), "Every full epoch permutation")
        orders[seed] = values
    updates, train_micro, eval_micro, eval_batches = 2300, 18260, 425, 54
    require([plan[k] for k in ("updates_per_fit", "training_microbatches_per_fit", "evaluation_microbatches_per_fit", "evaluation_batches_per_fit")]
            == [updates, train_micro, eval_micro, eval_batches], "Complete effective/micro budgets")
    counts = {"forward_attempted": train_micro+eval_micro, "forward_returned": train_micro+eval_micro,
              "backward_attempted": train_micro, "backward_returned": train_micro,
              "optimizer_attempted": updates, "optimizer_returned": updates, "training_rows": 20*len(fit_rows), "evaluation_rows": len(rows)}
    records = {}
    # Both arms share geometry/order; calculate independently once per seed.
    for seed in SEEDS:
        expected = []
        for epoch, order in enumerate(orders[seed]):
            for offset in range(0, len(fit_rows), 256):
                batch = [fit_rows[int(i)] for i in order[offset:offset+256]]
                work, micro, padding = old.public_work(batch, "token_aligned", lengths)
                expected.append((epoch, offset, batch, work, micro, padding))
        for method in METHODS:
            name = f"{method}-{seed}"; dest = run/"fits"/name; rec = read(dest/"completed.json")
            require(rec["status"] == "completed" and rec["version"] == VERSION and rec["method"] == rec["objective_weighting"] == method
                    and rec["model_method"] == "token_aligned" and rec["seed"] == seed and rec["epochs"] == 20
                    and rec["plan_sha256"] == plan_sha and rec["orders_sha256"] == plan["orders"][str(seed)]["sha256"]
                    and rec["counts"] == counts and rec["training_effective_batches"] == updates and rec["training_microbatches"] == train_micro
                    and rec["evaluation"]["effective_batches"] == eval_batches and rec["evaluation"]["microbatches"] == eval_micro
                    and rec["original_capacity_admitted"] is False and rec["allocation"] == done["allocation"] == plan["allocation"], "Fit identity/counts")
            check_manifest(dest, rec["files"], {"weights.pt", "updates.jsonl", "predictions.npz"})
            core = {"class": "DialogueTokenAlignment", "mode": "aligned", "version": "dialogue-token-alignment-v1",
                    "input_dim": 384, "projection_dim": 64, "hidden_dim": 64, "feature_dim": 401,
                    "parameters": 124482, "common_scorer_parameters": 75138}
            require(all(rec["configuration"].get(k) == v for k, v in core.items()), "Unchanged aligned configuration")
            for k in ("initial_state_sha256", "initial_common_sha256"):
                pin = rec[k]; require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin), "Initializer digest format")
            journal = [base.decode(s) for s in (dest/"updates.jsonl").read_text().splitlines()]
            require(len(journal) == len(expected), "Full journal")
            norms, works, seconds = [], [], 0.
            for entry, (epoch, offset, batch, work, micro, padding) in zip(journal, expected, strict=True):
                require(entry["epoch"] == epoch and entry["start"] == offset and entry["row_indices"] == [r["row_index"] for r in batch]
                        and entry["work"] == work and entry["microbatches"] == micro, "Exact batch rows/work")
                old.finite_cost(entry["weighted_loss"], "Loss witness", positive=False)
                seconds += old.finite_cost(entry["wall_seconds"], "Batch cost")
                old.check_normalization(entry["normalization"], batch, micro, padding)
                norms.append(entry["normalization"]); works.append(work)
            require(old.merge_norm(norms) == rec["training_normalization"] and old.summed(works) == rec["training_work"], "Training witness totals")
            work, micro, padding = old.public_work(rows, "token_aligned", lengths)
            require(work == rec["evaluation"]["work"] and micro == eval_micro, "Evaluation geometry")
            old.check_normalization(rec["evaluation"]["normalization"], rows, micro, padding)
            tw, ew, fw = (old.finite_cost(v, "Fit timing") for v in (rec["training_wall_seconds"], rec["evaluation"]["wall_seconds"], rec["wall_seconds"]))
            require(seconds <= tw and tw+ew <= fw <= done["wall_seconds"]
                    and 0 < rec["process_lifetime_peak_rss_bytes"] <= done["process_lifetime_peak_rss_bytes"], "Nested timing/RSS scopes")
            records[name] = rec
        pair = [records[f"{m}-{seed}"] for m in METHODS]
        require(pair[0]["initializer_witness"] == pair[1]["initializer_witness"] and pair[0]["configuration"] == pair[1]["configuration"], "Paired initialization/configuration")
        witness = pair[0]["initializer_witness"]
        for field, key in (("full_state_sha256", "initial_state_sha256"), ("common_state_sha256", "initial_common_sha256")):
            require(witness[field] == {m: records[f"{m}-{seed}"][key] for m in METHODS}
                    and len(set(witness[field].values())) == 1
                    and pair[0][key] == witness["original_sequence"][field]["token_aligned"], "Full paired historical-draw initializer witnesses")
    require(done["progress"]["completed_fits"] == FIT_ORDER and done["progress"]["active_fit"] is None
            and done["progress"]["totals"] == {k: 6*v for k, v in counts.items()}, "All six work totals")
    size = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    require(0 < old.finite_cost(done["wall_seconds"], "Whole wall") <= LIMITS["wall_seconds"]
            and 0 < done["process_lifetime_peak_rss_bytes"] <= LIMITS["rss_bytes"] and size <= LIMITS["output_bytes"]
            and sum(r["wall_seconds"] for r in records.values()) <= done["wall_seconds"], "Whole execution caps")
    return rows, done, plan, records, size, cached


def report_text(summary):
    c = summary["continuation"]
    lines = ["# Fresh objective comparison", "", summary["scope"], "",
             (f"Continuation **{'PASS' if c['passed'] else 'FAIL'}**, {c['checks_passed']}/13 checks. "
              f"Objective evidence: {c['objective_passed']}; practical evidence: {c['practical_passed']}."), "",
             "| Readout | Seed | Changed accuracy | Retained error | Overall NLL | Equal-service NLL |",
             "|---|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        for seed in SEEDS:
            cells = summary["fits"][f"{category}-{seed}"]["cells"]
            values = (cells["heldout_service/changed"]["metrics"]["accuracy"]["row"], cells["heldout_service/retained"]["metrics"]["error"]["row"],
                      cells["heldout_service/all"]["metrics"]["nll"]["row"], cells["heldout_service/all"]["metrics"]["nll"]["equal_service"])
            lines.append(f"| {category} | {seed} | "+" | ".join("undefined" if v is None else f"{v:.9f}" for v in values)+" |")
    lines += ["", "All readouts, seed means, historical families, service effects and paired repair/harm counts are in summary.json.",
              "No confidence interval, significance or architecture novelty claim. Tiny rare-category supports remain descriptive.", "",
              "| Fit | Training seconds | Evaluation seconds | Fit seconds |", "|---|---:|---:|---:|"]
    for name, r in summary["costs"]["per_fit"].items():
        lines.append(f"| {name} | {r['training_wall_seconds']:.6f} | {r['evaluation_wall_seconds']:.6f} | {r['wall_seconds']:.6f} |")
    lines += ["", f"Whole campaign: {summary['costs']['whole_wall_seconds']:.6f} seconds.", summary["costs"]["scope"]]
    return "\n".join(lines)+"\n"


def execute(args):
    out, run = Path(args.out).resolve(), Path(args.run).resolve()
    require(not out.is_relative_to(run) and not run.is_relative_to(out), "Separate output tree")
    out.mkdir(parents=True, exist_ok=False); started = time.monotonic(); prior = None; source_pin = None
    def check():
        require(time.monotonic()-started <= REPORT_LIMITS["wall_seconds"] and old.peak_rss() <= REPORT_LIMITS["rss_bytes"], "Report time/RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= REPORT_LIMITS["output_bytes"], "Report storage cap")
    def expired(_sig, _frame): raise TimeoutError("Saved-only report cap")
    try:
        source_pin = digest(__file__)
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        prior = signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, REPORT_LIMITS["wall_seconds"])
        write(out/"started.json", {"request": {k: str(v) for k, v in vars(args).items()}, "source_sha256": source_pin, "limits": REPORT_LIMITS})
        rows, done, plan, records, size, cached = authenticate_run(run, args.run_sha256)
        history = historical_inputs(done["files"]["evaluation-rows.jsonl"]["sha256"]); check()
        packets = {}
        for name in FIT_ORDER:
            with np.load(run/"fits"/name/"predictions.npz", allow_pickle=False) as z: packets[name] = {k: z[k] for k in z.files}
        with np.load(run/"references.npz", allow_pickle=False) as z: refs = {k: z[k] for k in z.files}
        summary = aggregate(rows, packets, refs, history)
        summary.update(status="completed", technical_validity_passed=True, execution_completed_sha256=args.run_sha256,
                       plan_sha256=done["plan_sha256"], source_sha256=plan["source_sha256"], split=plan["split"],
                       costs={"whole_wall_seconds": done["wall_seconds"], "execution_bytes": size,
                         "process_lifetime_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
                         "per_fit": {n: {"wall_seconds": r["wall_seconds"], "training_wall_seconds": r["training_wall_seconds"],
                             "evaluation_wall_seconds": r["evaluation"]["wall_seconds"], "counts": r["counts"],
                             "parameters": r["configuration"]["parameters"], "training_work": r["training_work"], "evaluation_work": r["evaluation"]["work"]} for n, r in records.items()},
                         "schema_preparation": {"wall_seconds": cached["wall_seconds"], "completed_sha256": old.SCHEMA_PIN},
                         "scope": "Whole includes authentication, initialization, training, evaluation and IO. Training/evaluation nest inside each fit and are not added again. Schema preparation is separate historical cost. RSS is process-lifetime high-water."})
        check_manifest(run, done["files"], set(done["files"]))
        require(digest(run/"completed.json") == args.run_sha256 and digest(__file__) == source_pin, "End source/input identity")
        write(out/"summary.json", summary)
        with (out/"report.md").open("x") as stream: stream.write(report_text(summary))
        check()
        files = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        receipt = {"status": "completed", "version": VERSION, "technical_validity_passed": True,
                   "continuation_passed": summary["continuation"]["passed"], "execution_completed_sha256": args.run_sha256,
                   "plan_sha256": done["plan_sha256"], "source_sha256": {"scripts/report_dialogue_objective.py": source_pin, HELPER: HELPER_PIN, old.METRIC_SOURCE: old.METRIC_PIN},
                   "files": files, "execution_members": done["files"], "historical": history["provenance"],
                   "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": old.peak_rss(),
                   "limits": REPORT_LIMITS, "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0, "no_retry": True, "scope": SCOPE}
        write(out/"receipt.json", receipt); check()
        return summary["continuation"]["passed"]
    except BaseException as error:
        if prior is not None: signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists(): (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "error": repr(error), "source_sha256": source_pin,
                  "execution_completed_sha256": args.run_sha256, "wall_seconds": time.monotonic()-started,
                  "model_calls": 0, "encoder_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            if callable(getattr(error, "add_note", None)): error.add_note("Failure preservation: "+repr(secondary))
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps({"continuation_allowed": execute(parser.parse_args())}))
