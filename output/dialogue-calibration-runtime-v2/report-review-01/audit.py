"""Independent saved-only temperatures, proper scores and eleven-condition audit.

Only the source-pinned report reader's authentication is inherited. No producer
temperature, score, evaluate_fit or criteria function is used. No neural calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import resource
import signal
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
VERSION = "dialogue-runtime-calibration-independent-report-audit-v1"
LIMITS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
READER = "scripts/report_dialogue_runtime_calibration.py"
READER_PIN = "3d7da51f256cceb18ae46cb11b4d9e77f3d28544cea3eeecc4166965a44bf009"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
SEEDS = (6901, 6902, 6903)
FITS = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
ROUTES, PANELS = ("raw", "normalized", "calibrated"), ("all", "seen", "unseen")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
NONE, DONTCARE = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
SCOPE = ("Independent calibration-only bounded beta optimization and all twelve fits' raw/normalized/calibrated "
         "DEV all/seen/unseen micro and three-stratum counts, accuracy, direct-log NLL and candidate-sum Brier; "
         "full top-tie and canonical first-choice preservation; all eleven conditions and all four phase costs. "
         "Authentication, canonical cohort provenance, internal work/configuration and original complete-tree exactness "
         "are inherited from the pinned completed report reader. Per-service/type/bin/factorial/paired-change trees "
         "outside these panels are authenticated but not numerically replayed. No model, optimizer, tokenizer, "
         "checkpoint deserialization, TEST, new predictions or DEV-driven temperature selection.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class Budget:
    def __init__(self, out, clock):
        self.out, self.clock = out, clock
        self.deadline = clock.deadline_after(LIMITS["wall_seconds"])
        self.progress, self.last = {}, None

    def check(self):
        now = self.clock.now_ns()
        if now >= self.deadline.expires_ns:
            raise TimeoutError("Independent calibration audit deadline expired")
        require(rss() <= LIMITS["rss_bytes"], "Audit RSS limit")
        self.last = now - self.deadline.started_ns

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Audit output limit")

    def elapsed(self):
        self.check()
        return self.last / 1e9


class Checks:
    def __init__(self):
        self.count, self.max_numeric_error = 0, 0.

    def compare(self, actual, expected, label, *, tolerance=5e-12):
        if isinstance(expected, dict):
            require(type(actual) is dict and set(expected) <= set(actual), "Missing audited fields: " + label)
            for key, value in expected.items():
                self.compare(actual[key], value, label + "/" + key, tolerance=tolerance)
        elif isinstance(expected, list):
            require(type(actual) is list and len(actual) == len(expected), "List coverage: " + label)
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                self.compare(a, b, f"{label}/{i}", tolerance=tolerance)
        else:
            self.count += 1
            if type(expected) is float:
                require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                        and math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance), "Numeric mismatch: " + label)
                self.max_numeric_error = max(self.max_numeric_error, abs(actual - expected))
            else:
                require(type(actual) is type(expected) and actual == expected, "Exact mismatch: " + label)


def authenticate(args, budget):
    require(sha(args.plan) == args.plan_sha256, "External analysis-plan pin")
    proposed = read(args.plan)
    require(proposed["sources"][READER] == READER_PIN, "Reviewed report reader pin")
    for name, pin in proposed["sources"].items():
        budget.check()
        require(sha(ROOT / name) == pin, "Pinned analysis source before import")
    reader = importlib.import_module("report_dialogue_runtime_calibration")
    report = (ROOT / proposed["out"]).resolve()
    request = SimpleNamespace(plan=args.plan, plan_sha256=args.plan_sha256, out=report)
    plan, ctx, phases, terminals, published = reader.authenticate(request, budget)
    receipt_path = report / "receipt.json"
    require(sha(receipt_path) == args.report_sha256, "External complete report pin")
    receipt = read(receipt_path)
    require(receipt["version"] == plan["version"] and receipt["status"] == "completed"
            and receipt["technical_validity_passed"] is True and receipt["complete_fits"] == 12
            and receipt["plan_sha256"] == args.plan_sha256 and receipt["source_sha256"] == plan["sources"]
            and receipt["control_plan_sha256"] == plan["control_plan"]["sha256"]
            and receipt["phases"] == plan["phases"] and receipt["limits"] == LIMITS
            and receipt["model_calls"] == 0 and receipt["official_test_opened"] is False
            and receipt["original_raw_failure_preserved"] is True and receipt["legacy_v1_cost_failure_preserved"] is True
            and receipt["runtime_v2_admission_passed"] is True, "Complete authenticated report identity")
    expected = {"started.json", "summary.json", "report.md"} | {f"fits/{name}.json" for name in FITS}
    require(set(receipt["files"]) == expected, "All twelve complete fit diagnostics")
    reader.common.manifest(report, receipt["files"], "receipt.json")
    start = read(report / "started.json")
    require(start["request"] == receipt["request"] and start["limits"] == receipt["limits"]
            and all(start[k] == receipt[k] for k in ("version", "clock_backend", "started_ns", "deadline_ns"))
            and Path(receipt["request"]["plan"]).resolve() == args.plan.resolve()
            and receipt["request"]["plan_sha256"] == args.plan_sha256
            and Path(receipt["request"]["out"]).resolve() == report, "Report launch and request binding")
    require(receipt["timing_available"] is True and receipt["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}
            and receipt["deadline_ns"] - receipt["started_ns"] == 300_000_000_000
            and 0 < receipt["elapsed_ns"] == receipt["finished_ns"] - receipt["started_ns"] < 300_000_000_000
            and receipt["wall_seconds"] == receipt["elapsed_ns"] / 1e9
            and 0 < receipt["peak_rss_bytes"] <= LIMITS["rss_bytes"]
            and sum(v["bytes"] for v in receipt["files"].values()) + receipt_path.stat().st_size <= LIMITS["output_bytes"],
            "Bounded successful report timing and resources")
    summary = read(report / "summary.json")
    require(summary["status"] == "completed" and summary["technical_validity_passed"] is True
            and summary["complete_fits"] == 12 and summary["plan_sha256"] == args.plan_sha256
            and summary["control_plan_sha256"] == plan["control_plan"]["sha256"] and summary["phases"] == plan["phases"]
            and set(summary["fits"]) == set(summary["fit_panels"]) == set(summary["temperatures"]) == set(FITS),
            "Complete report summary identity")
    for name in FITS:
        path = f"fits/{name}.json"
        require(summary["fits"][name] == {"fit_id": name, "path": path, **receipt["files"][path]}, "Fit index identity")
    return reader, plan, ctx, phases, terminals, published, report, receipt, summary


def layout(path, count, split, budget):
    labels, widths, unseen, strata, dialogues, services = [], [], [], [], [], set()
    source_keys, endpoint_keys = set(), set()
    with Path(path).open() as stream:
        for i, line in enumerate(stream):
            budget.check()
            row = json.loads(line)
            require(type(row["row_index"]) is int and row["row_index"] == i and row["split"] == split, "Canonical complete row identity")
            ids, values, target = row["candidate_ids"], row["candidate_values"], row["label_index"]
            require(3 <= len(ids) <= 12 and len(set(ids)) == len(ids) == len(values)
                    and ids.count(NONE) == ids.count(DONTCARE) == 1
                    and type(target) is int and 0 <= target < len(ids) and row["label_id"] == ids[target], "Candidate and target identity")
            require(all(v is None if cid in (NONE, DONTCARE) else type(v) is str and bool(v.strip()) and cid == "value:" + v
                        for cid, v in zip(ids, values, strict=True)), "Exact public value identities")
            require(type(row["unseen"]) is bool and row["bin"] in (*STRATA[:2], "first_assignment", "revision", "clear"), "Panel and transition identity")
            stratum = row["bin"] if row["bin"] in STRATA[:2] else "changed"
            require(row["stratum"] == stratum, "Original stratum binding")
            source = (row["dialogue_id"], row["source_row_index"])
            endpoint = (row["dialogue_id"], row["time"], row["query_index"])
            require(source not in source_keys and endpoint not in endpoint_keys, "No duplicate source endpoints")
            source_keys.add(source)
            endpoint_keys.add(endpoint)
            if split == "train":
                require(row["source_split"] == "train" and row["analysis_role"] == "calibration" and row["unseen"] is False,
                        "Only calibration TRAIN targets can fit temperature")
            labels.append(target)
            widths.append(len(ids))
            unseen.append(row["unseen"])
            strata.append(stratum)
            dialogues.append(row["dialogue_id"])
            services.add(row["service"])
    require(len(labels) == count, "Complete canonical ledger count")
    return {"labels": np.asarray(labels, np.int64), "mask": np.arange(12)[None, :] < np.asarray(widths)[:, None],
            "unseen": np.asarray(unseen, bool), "strata": np.asarray(strata), "dialogues": list(dict.fromkeys(dialogues)),
            "services": sorted(services)}


def load(path, meta):
    with np.load(path, allow_pickle=False) as packet:
        require(set(packet.files) == {"log_probs", "row_indices"}, "Exact raw packet fields")
        logs, indices = packet["log_probs"], packet["row_indices"]
    require(logs.dtype == np.float32 and logs.shape == meta["mask"].shape and indices.dtype == np.int64
            and np.array_equal(indices, np.arange(len(logs), dtype=np.int64)), "Raw float32 canonical prediction identity")
    require(np.isfinite(logs[meta["mask"]]).all() and np.isneginf(logs[~meta["mask"]]).all(), "Finite supported logs and negative-infinity padding")
    mass = np.exp(logs.astype(np.float64)).sum(1)
    require(np.max(np.abs(mass - 1.)) <= 2e-6 and float(logs[meta["mask"]].max()) <= math.log1p(2e-6), "Raw mass without repair")
    return logs.astype(np.float64)


def mean(values):
    return math.fsum(map(float, values)) / len(values)


def normalized(logs, beta):
    # Independently use logaddexp reduction, rather than producer exp/sum/log.
    require(math.isfinite(beta) and beta > 0, "Positive finite beta")
    shifted = (logs - logs.max(1, keepdims=True)) * beta
    result = shifted - np.logaddexp.reduce(shifted, axis=1)[:, None]
    require(np.isfinite(result[np.isfinite(logs)]).all() and np.isneginf(result[~np.isfinite(logs)]).all(), "Normalized finite support")
    require(np.array_equal(result.argmax(1), logs.argmax(1))
            and np.array_equal(result == result.max(1, keepdims=True), logs == logs.max(1, keepdims=True)), "Entire top tie set and first choice preserved")
    require(float(np.max(np.abs(np.exp(result).sum(1) - 1.))) <= 2e-6, "Transformed mass")
    return result


def fit_beta(logs, targets, budget):
    centered = logs - logs.max(1, keepdims=True)
    observed = centered[np.arange(len(targets)), targets]
    finite = np.isfinite(centered)
    terms = np.where(finite, centered, 0.)

    def objective(beta):
        budget.check()
        q = normalized(logs, beta)
        return mean(-q[np.arange(len(targets)), targets]), mean((np.exp(q) * terms).sum(1) - observed)

    low, high = 0.125, 8.
    _, dlow = objective(low)
    _, dhigh = objective(high)
    require(math.isfinite(dlow) and math.isfinite(dhigh) and dlow <= dhigh + 1e-12, "Finite convex endpoint derivatives")
    steps = 0
    if bool(np.all(centered[finite] == 0)):
        beta, location = 1., "unidentified_uniform_identity"
    elif dlow >= 0:
        beta, location = low, "lower_bound"
    elif dhigh <= 0:
        beta, location = high, "upper_bound"
    else:
        left, right = low, high
        for _ in range(64):
            middle = (left + right) / 2
            _, derivative = objective(middle)
            require(math.isfinite(derivative), "Finite bisection derivative")
            if derivative >= 0:
                right = middle
            else:
                left = middle
        beta, location, steps = (left + right) / 2, "interior", 64
    before, _ = objective(1.)
    after, derivative = objective(beta)
    raw = mean(-logs[np.arange(len(targets)), targets])
    require(after <= before + 1e-12, "Calibration-only optimum cannot worsen its objective")
    return {"beta": beta, "temperature": 1 / beta, "beta_bounds": [low, high], "location": location,
            "bisection_steps": steps, "endpoints": len(targets), "raw_nll_before": raw,
            "normalized_nll_before": before, "normalized_nll_after": after,
            "normalization_only_nll_drift": before - raw, "derivative_at_lower_bound": dlow,
            "derivative_at_upper_bound": dhigh, "derivative_at_solution": derivative,
            "selection_data": "calibration_only", "choices_unchanged": True}


def score(logs, meta):
    labels, choices = meta["labels"], logs.argmax(1)
    p = np.exp(logs)
    nll = -logs[np.arange(len(logs)), labels]
    # Algebraic multiclass Brier avoids copying the producer residual matrix.
    brier = np.square(p).sum(1) - 2 * p[np.arange(len(logs)), labels] + 1
    correct = choices == labels

    def cell(selected):
        n, hits = int(selected.sum()), int(correct[selected].sum())
        return {"count": n, "correct": hits, "incorrect": n - hits, "accuracy": hits / n if n else None,
                "error": (n - hits) / n if n else None,
                "nll": mean(nll[selected]) if n else None, "brier": mean(brier[selected]) if n else None}

    result = {}
    for name in PANELS:
        selected = np.ones(len(logs), bool) if name == "all" else meta["unseen"] if name == "unseen" else ~meta["unseen"]
        strata = {s: cell(selected & (meta["strata"] == s)) for s in STRATA}
        macro = {k: mean([strata[s][k] for s in STRATA]) if all(strata[s][k] is not None for s in STRATA) else None
                 for k in ("accuracy", "nll", "brier")}
        result[name] = {"micro": cell(selected), "strata": strata, "macro_three": macro}
    return result


def criteria(fits):
    result = []

    def macro(panel):
        cells = panel["strata"]
        return sum((Fraction(cells[s]["correct"], cells[s]["count"]) for s in STRATA), Fraction()) / 3 if all(cells[s]["count"] for s in STRATA) else None

    def delta(panel, field):
        pairs = [(field(fits[f"trainable_numbers-{s}"][panel]), field(fits[f"frozen_numbers-{s}"][panel])) for s in SEEDS]
        return [a - b if a is not None and b is not None else None for a, b in pairs]

    def exact(name, values, threshold, comparison):
        available = all(v is not None for v in values)
        m = sum(values, Fraction()) / 3 if available else None
        result.append({"name": name, "passed": bool(available and (m >= threshold if comparison == "ge" else m <= threshold)),
                       "paired_seed_values": [float(v) if v is not None else None for v in values],
                       "mean": float(m) if available else None, "threshold": float(threshold), "comparison": comparison})

    unseen = delta("unseen", macro)
    exact("unseen_macro_gain_1pp", unseen, Fraction(1, 100), "ge")
    wins = sum(v > 0 for v in unseen) if all(v is not None for v in unseen) else None
    result.append({"name": "unseen_macro_strict_paired_wins", "passed": wins is not None and wins >= 2, "wins": wins, "required": 2})
    for metric in ("nll", "brier"):
        a = [fits[f"trainable_numbers-{s}"]["unseen"]["micro"][metric] for s in SEEDS]
        b = [fits[f"frozen_numbers-{s}"]["unseen"]["micro"][metric] for s in SEEDS]
        require(all(v is not None and math.isfinite(v) for v in a + b), "Nonempty proper-score denominator")
        am, bm = mean(a), mean(b)
        result.append({"name": f"unseen_micro_{metric}_nonworse", "passed": am <= bm,
                       "paired_seed_values": [x-y for x, y in zip(a, b, strict=True)], "mean": am-bm,
                       "primary_mean": am, "control_mean": bm, "threshold": 0., "comparison": "le"})
    exact("seen_macro_deficit_at_most_1pp", delta("seen", macro), -Fraction(1, 100), "ge")

    def error(panel):
        c = panel["strata"]["assigned_retention"]
        return Fraction(c["incorrect"], c["count"]) if c["count"] else None

    for panel in ("seen", "unseen"):
        exact(f"{panel}_assigned_retention_error_increase_at_most_half_pp", delta(panel, error), Fraction(1, 200), "le")
    return {"checks": result, "checks_total": 7, "checks_passed": sum(c["passed"] for c in result), "passed": all(c["passed"] for c in result)}


def continuation(routes):
    originals = {route: criteria(fits) for route, fits in routes.items()}
    checks = list(originals["calibrated"]["checks"])
    for arm, metric, strict in (("trainable_numbers", "nll", True), ("trainable_numbers", "brier", False),
                                ("frozen_numbers", "nll", False), ("frozen_numbers", "brier", False)):
        a = [routes["calibrated"][f"{arm}-{s}"]["unseen"]["micro"][metric] for s in SEEDS]
        b = [routes["raw"][f"{arm}-{s}"]["unseen"]["micro"][metric] for s in SEEDS]
        require(all(v is not None and math.isfinite(v) for v in a + b), "Nonempty own-raw proper-score denominator")
        am, bm = mean(a), mean(b)
        checks.append({"name": f"{arm}_unseen_micro_{metric}_" + ("improves_own_raw" if strict else "nonworse_own_raw"),
                       "passed": am < bm if strict else am <= bm, "paired_seed_values": [x-y for x, y in zip(a, b, strict=True)],
                       "mean": am-bm, "calibrated_mean": am, "raw_mean": bm, "threshold": 0., "comparison": "lt" if strict else "le"})
    return {"checks": checks, "checks_total": 11, "checks_passed": sum(c["passed"] for c in checks),
            "passed": all(c["passed"] for c in checks), "original_conditions": originals,
            "decision_conditions_unchanged": True, "complete_fit_membership": True, "raw_result_revised": False}


def recompute(args, authenticated, budget):
    _, plan, ctx, phases, terminals, published, report, receipt, recorded = authenticated
    prepared, inferred = ((ROOT / plan["phases"][p]["path"]).resolve() for p in ("prepare", "infer"))
    dev = layout(Path(ctx["prior"].run) / "evaluation-rows.jsonl", 62329, "dev", budget)
    cal = layout(prepared / "evaluation-rows.jsonl", 13333, "train", budget)
    require(len(cal["dialogues"]) == 512 and cal["dialogues"] == read(prepared / "selection.json")["selected_ids"], "Exact complete calibration cohort")
    checks, fits, routes = Checks(), {}, {r: {} for r in ROUTES}
    for name in FITS:
        budget.progress.update(fit_id=name, operation="independent-calibration-only-optimization")
        raw = load(Path(ctx["prior"].run) / name / "predictions.npz", dev)
        calibration = load(inferred / name / "predictions.npz", cal)
        fitted = fit_beta(calibration, cal["labels"], budget)
        observed = read(report / "fits" / (name + ".json"))
        require(set(observed) == {"raw", "normalized", "calibrated", "temperature", "calibration_validation"}, "Exact fit routes")
        checks.compare(observed["temperature"], fitted, name + "/temperature", tolerance=5e-11)
        checks.compare(recorded["temperatures"][name], fitted, name + "/summary-temperature", tolerance=5e-11)
        checks.compare(observed["calibration_validation"], {"rows": 13333, "dialogues": 512, "services": cal["services"],
                       "source_split": "train", "analysis_role": "calibration", "supported_candidate_positions": int(cal["mask"].sum()),
                       "maximum_mass_error": float(np.abs(np.exp(calibration).sum(1)-1).max())}, name + "/calibration-validation")
        for route in ROUTES:
            budget.check()
            logs = raw if route == "raw" else normalized(raw, 1. if route == "normalized" else fitted["beta"])
            value = score(logs, dev)
            routes[route][name] = value
            checks.compare(observed[route]["panels"], value, name + "/" + route)
            checks.compare(recorded["fit_panels"][name][route], value, name + "/summary/" + route)
            top = logs == logs.max(1, keepdims=True)
            validation = {"rows": 62329, "supported_candidate_positions": int(dev["mask"].sum()),
                          "maximum_mass_error": float(np.abs(np.exp(logs).sum(1)-1).max()), "tolerance": 2e-6,
                          "exact_top1_tie_rows": int((top.sum(1) > 1).sum()),
                          "float64_exponent_underflow_positions": int((np.exp(logs)[dev["mask"]] == 0).sum())}
            if route != "raw":
                validation.update(dtype="float64", choices_unchanged=True, top_tie_masks_unchanged=True)
            checks.compare(observed[route]["validation"], validation, name + "/validation/" + route)
            if route == "raw":
                checks.compare(published["fits"][name]["panels"], value, name + "/old-raw")
        fits[name] = {"temperature": fitted, "panels": {r: routes[r][name] for r in ROUTES}}
        budget.progress.setdefault("completed_fits", []).append(name)
        del raw, calibration, logs
    gate = continuation(routes)
    checks.compare(recorded["continuation"], gate, "all-eleven-conditions")
    checks.compare(published["continuation"], gate["original_conditions"]["raw"], "unchanged-old-seven")
    require(gate["original_conditions"]["raw"]["passed"] is False and gate["original_conditions"]["raw"]["checks_passed"] == 6, "Historical raw FAIL6/7 retained")
    checks.compare(receipt, {"continuation_passed": gate["passed"], "checks_passed": gate["checks_passed"], "checks_total": 11,
                   "original_raw_checks_passed": 6, "original_raw_checks_total": 7}, "receipt outcome")
    costs = {p: {"parent_wall_seconds": terminals[p]["wall_seconds"], "receipt_wall_seconds": phases[p]["wall_seconds"],
                  "peak_rss_bytes": phases[p]["peak_rss_bytes"], "sampled_mps_driver_max_bytes": phases[p]["sampled_mps_driver_max_bytes"]}
             for p in ("prepare", "qualify", "pilot", "infer")}
    prior_seconds = math.fsum(v["parent_wall_seconds"] for v in costs.values())
    checks.compare(recorded["costs"], {"separate_phases": costs, "nonnested_prior_phase_seconds": prior_seconds}, "separate phase cost")
    checks.compare(receipt["new_control_total_seconds"], prior_seconds + receipt["wall_seconds"], "control plus report cost once")
    return {"version": VERSION, "status": "completed", "agreement": True, "fits": fits, "continuation": gate,
            "cells": 12 * 3 * 3 * 4, "scalar_checks": checks.count, "maximum_numeric_difference": checks.max_numeric_error,
            "temperature_fit_count": 12, "calibration_rows_per_fit": 13333, "dev_rows_per_fit": 62329,
            "nonnested_prior_phase_seconds": prior_seconds, "control_and_report_seconds": prior_seconds + receipt["wall_seconds"],
            "scope": SCOPE, "comparison_tolerances": {"scores": 5e-12, "temperature_fields": 5e-11,
                 "decisions": "Independent comparisons use no epsilon; counts, ties and outcome booleans must match exactly."}}


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    budget = clock = handler = None
    source_pin = None
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        source_pin = sha(__file__)
        require(sha(ROOT / CLOCK) == CLOCK_PIN, "Qualified native clock source")
        clock = importlib.import_module("openjev.research.suspend_clock").SuspendClock()
        require(clock.backend in {"mach_continuous_time", "CLOCK_BOOTTIME"}, "Native audit clock")
        budget = Budget(args.out, clock)

        def expired(*_):
            raise TimeoutError("Supplementary audit alarm expired")

        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 300)
        write(args.out / "started.json", {"version": VERSION, "request": request, "source_sha256": source_pin, "limits": LIMITS, "model_calls": 0})
        authenticated = authenticate(args, budget)
        summary = recompute(args, authenticated, budget)
        write(args.out / "summary.json", summary)
        authenticate(args, budget)
        require(sha(__file__) == source_pin, "Stable independent source")
        budget.storage()
        elapsed = budget.last
        report = authenticated[6]
        receipt = {"version": VERSION, "status": "completed", "agreement": True, "request": request,
                   "source_sha256": source_pin, "inherited_reader_sha256": READER_PIN, "clock_source_sha256": CLOCK_PIN,
                   "plan_sha256": args.plan_sha256, "producer_receipt_sha256": args.report_sha256,
                   "producer_summary_sha256": sha(report / "summary.json"), "scope": SCOPE, "limits": LIMITS,
                   "files": {name: {"sha256": sha(args.out / name), "bytes": (args.out / name).stat().st_size}
                             for name in ("started.json", "summary.json")}, "clock_backend": clock.backend,
                   "started_ns": budget.deadline.started_ns, "finished_ns": budget.deadline.started_ns + elapsed,
                   "deadline_ns": budget.deadline.expires_ns, "elapsed_ns": elapsed, "wall_seconds": elapsed / 1e9,
                   "timing_available": True, "peak_rss_bytes": rss(), "model_calls": 0, "official_test_opened": False,
                   "scalar_checks": summary["scalar_checks"], "cells": summary["cells"],
                   "continuation_passed": summary["continuation"]["passed"], "checks_passed": summary["continuation"]["checks_passed"],
                   "checks_total": 11, "original_raw_failure_preserved": True}
        write(args.out / "receipt.json", receipt)
        budget.storage()
        return receipt
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "agreement": False,
                  "request": request, "source_sha256": source_pin, "error_type": type(error).__name__, "error": str(error),
                  "timing_available": False, "wall_seconds": None, "elapsed_ns": None,
                  "last_successful_elapsed_ns": None if budget is None else budget.last,
                  "progress": None if budget is None else budget.progress, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve first failure
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "report-sha256"):
        parser.add_argument("--" + name, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({"status": result["status"], "agreement": result["agreement"], "checks_passed": result["checks_passed"]}))
