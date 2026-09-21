"""Saved reporting audit; no posterior, simulator, runner or scoring imports.

Rebuilds aggregates from every persisted prefix and joins qualification scores
to authenticated original transitions. TV/KL, support-mask contents, actual
execution and timing truth remain inherited rather than independently replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
ANALYTIC = "scripts/audit_otto_large_memory.py"
ANALYTIC_PIN = "1ad5a080f58177751dadf0eaab0cd9e2c7b37d7e9d84b50cc32f4046a66155ec"
PRODUCER = "scripts/study_otto_spectral_memory.py"
SOURCES = {PRODUCER, "src/openjev/research/otto_spectral_memory.py", CLOCK, ANALYTIC,
           "tests/test_otto_spectral_memory.py", "tests/test_otto_spectral_study.py",
           "research/otto-spectral-memory-protocol.md"}
RANKS = (4, 8, 16)
CANDIDATES = tuple(f"dct{q}_{fill}" for q in RANKS for fill in ("neutral", "nearest"))
QUAL = ("exact_log", "dct53_neutral", "dct53_nearest")
ARMS = ("full_bayes", "exact_log", *CANDIDATES, "dct53_neutral", "dct53_nearest", "recent32", "recent32_hard")
METRICS = ("tv_to_full", "kl_full_to_approx", "full_objective_excess", "matches_full_action")
TIMES = ("initialization_seconds", "update_seconds", "decode_seconds", "planner_seconds")
LIMITS = {"native_seconds": 300, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}
PAYLOADS = {"started.json", "summary.json", "cases.jsonl", "prefixes.jsonl"}
SCOPE = ("Reporting consistency audit of all 2164 saved prefixes, 96 cases and 12 arms: "
         "score/action/tie/excess joins, original saved-score qualification, rotations, support-count "
         "witnesses, aggregate weights, timings and array-storage arithmetic. Does not independently "
         "reconstruct posteriors, TV/KL values, support-mask contents, fill-TV values, physical sensor "
         "semantics, execution, timing/RSS truth or search utility. No native environment/model calls.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def locate(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def finite(value, name, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value), f"finite number: {name}")
    require(low is None or value >= low, f"lower bound: {name}")
    require(high is None or value <= high, f"upper bound: {name}")
    return value


def integer(value, name, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), f"integer: {name}")
    return value


class Compare:
    def __init__(self):
        self.checks, self.maximum_difference = 0, 0.0

    def same(self, actual, expected, name, tolerance=1e-12):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"keys: {name}")
            for key, value in expected.items():
                self.same(actual[key], value, f"{name}.{key}", tolerance)
        elif isinstance(expected, (tuple, list)):
            require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), f"length: {name}")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.same(left, right, f"{name}.{index}", tolerance)
        elif type(expected) is float:
            finite(actual, name)
            delta = abs(actual - expected)
            self.maximum_difference = max(self.maximum_difference, delta)
            require(delta <= tolerance, f"numeric mismatch: {name}: {actual!r} vs {expected!r}")
            self.checks += 1
        else:
            require(type(actual) is type(expected) and actual == expected, f"exact mismatch: {name}")
            self.checks += 1


def manifest(directory, receipt, names, budget):
    require(not directory.is_symlink() and set(receipt["files"]) == names
            and {p.name for p in directory.iterdir()} == names | {"receipt.json"}, "exact completed payload closure")
    for name, item in receipt["files"].items():
        require(Path(name).name == name, "safe payload name")
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
                and sha(path) == item["sha256"], f"payload identity: {name}")
        budget()


def authenticate(args, budget):
    require(sha(args.plan) == args.plan_sha256 and sha(args.run / "receipt.json") == args.receipt_sha256, "external input pins")
    plan, done = read(args.plan), read(args.run / "receipt.json")
    require(plan["version"] == "otto-spectral-memory-v1" and plan["status"] == "frozen_before_saved_study"
            and plan["limits"] == done["limits"] == LIMITS and set(plan["sources"]) == SOURCES, "frozen study identity")
    require(done["status"] == "completed" and done["plan_sha256"] == args.plan_sha256
            and done["inputs"] == plan["inputs"] and done["source_sha256"] == plan["sources"][PRODUCER], "completed same study")
    require(plan["sources"][CLOCK] == CLOCK_PIN and plan["sources"][ANALYTIC] == ANALYTIC_PIN, "inherited helper pins")
    for name, pin in plan["sources"].items():
        require(not (ROOT / name).is_symlink() and sha(ROOT / name) == pin, f"frozen source: {name}")
    manifest(args.run, done, PAYLOADS, budget)
    cfg = plan["configuration"]
    require(cfg == {"ranks": list(RANKS), "extensions": ["neutral", "nearest"], "qualification_rank": 53,
                   "arms": list(ARMS), "cases": 96, "decisions": 2164, "eligible_prefixes": 538,
                   "eligibility": "completed_steps > 32", "score_tolerance": 1e-8,
                   "qualification_tv_tolerance": 1e-10, "rotation": "(case_index + completed_steps) modulo12"}, "fixed complete design")
    require(done["environment_calls"] == done["model_calls"] == done["training_updates"] == 0, "saved only")
    require(done["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
            and done["finished_ns"] - done["started_ns"] == done["elapsed_ns"]
            and done["wall_seconds"] == done["elapsed_ns"] / 1e9
            and 0 <= done["wall_seconds"] < 300 and 0 <= done["peak_rss_bytes"] <= LIMITS["rss_bytes"], "producer resource witnesses")
    inputs = plan["inputs"]
    original = locate(inputs["run"])
    external = [(original / "receipt.json", inputs["receipt_sha256"]),
                (original / "summary.json", inputs["summary_sha256"]),
                (locate(inputs["plan"]), inputs["plan_sha256"]),
                (locate(inputs["terminal"]), inputs["terminal_sha256"])]
    auditdir = locate(inputs["audit"])
    external += [(auditdir / "receipt.json", inputs["audit_receipt_sha256"]),
                 (auditdir / "summary.json", inputs["audit_summary_sha256"])]
    for path, pin in external:
        require(sha(path) == pin, f"inherited external pin: {path.name}")
    parent, terminal, agreed = read(original / "receipt.json"), read(locate(inputs["terminal"])), read(auditdir / "receipt.json")
    require(parent["status"] == terminal["status"] == agreed["status"] == "completed"
            and agreed["agreement"] is True and agreed["source_sha256"] == ANALYTIC_PIN
            and parent["completed_episodes"] == 768 and parent["learned_pilot_opportunity"] is False, "successful complete inherited run/audit")
    require(agreed["producer_receipt_sha256"] == inputs["receipt_sha256"]
            and agreed["producer_summary_sha256"] == inputs["summary_sha256"]
            and agreed["plan_sha256"] == inputs["plan_sha256"]
            and agreed["terminal_sha256"] == inputs["terminal_sha256"]
            and parent["plan_sha256"] == inputs["plan_sha256"], "inherited audit lineage")
    require(terminal["returncode"] == 0 and terminal["group_absent"] is True and terminal["timed_out"] is False
            and terminal["error"] is None and terminal["clock_error"] is None, "actual successful native parent")
    manifest(auditdir, agreed, {"started.json", "summary.json"}, budget)
    oldnames = {"imports.json", "qualification.json", "public-kernel.npz", "transitions.jsonl", "episodes.jsonl", "summary.json"}
    oldnames |= {f"qualification-{i:02}.json" for i in range(1, 18)}
    manifest(original, parent, oldnames, budget)
    oldplan = read(locate(inputs["plan"]))
    require(parent["sources"] == oldplan["sources"] and parent["upstream_sources"] == oldplan["upstream_sources"], "inherited source identity")
    for base, pins in ((ROOT, oldplan["sources"]), (ROOT / "tmp/otto-source-review-01", oldplan["upstream_sources"])):
        for name, pin in pins.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(base / name) == pin, "unchanged inherited source")
    return plan, done, original, external


def weighted(rows, weights=None):
    if not rows:
        return None
    weights = [1.0] * len(rows) if weights is None else weights
    require(len(rows) == len(weights) and all(w > 0 and math.isfinite(w) for w in weights), "positive weights")
    keys = set(rows[0])
    require(all(set(row) == keys for row in rows), "complete metrics")
    total = math.fsum(weights)
    return {key: math.fsum(float(row[key]) * weight for row, weight in zip(rows, weights, strict=True)) / total for key in keys}


def pooled(cases, mixture, eligible=False):
    selected = [c for c in cases if c["eligible_prefixes" if eligible else "decisions"]]
    field, count = ("eligible_means", "eligible_prefixes") if eligible else ("all_means", "decisions")
    records, counts = [c[field] for c in selected], [c[count] for c in selected]
    weights = [mixture[str(c["initial_hit"])] / 32 for c in selected]
    strata = {}
    for hit in (1, 2, 3):
        chosen = [c for c in selected if c["initial_hit"] == hit]
        strata[str(hit)] = {"all_cases": sum(c["initial_hit"] == hit for c in cases), "selected_cases": len(chosen),
                            "prefixes": sum(c[count] for c in chosen), "case_means": weighted([c[field] for c in chosen]),
                            "prefix_means": weighted([c[field] for c in chosen], [c[count] for c in chosen])}
    return {"all_cases": len(cases), "selected_cases": len(selected), "prefixes": sum(counts),
            "selected_case_mixture_mass": math.fsum(weights), "by_initial_hit": strata,
            "unweighted_case_means": weighted(records), "mixture_weighted_case_means": weighted(records, weights),
            "mixture_weighted_prefix_means": weighted(records, [w * n for w, n in zip(weights, counts, strict=True)])}


def choice(scores, position):
    require(isinstance(scores, list) and len(scores) == 4, "four action scores")
    valid = [a for a in range(4) if 0 <= position[a // 2] + (-1 if a % 2 == 0 else 1) < 53]
    require(all((score is not None) == (a in valid) for a, score in enumerate(scores)), "boundary action support")
    for a in valid:
        finite(scores[a], "action score")
    best = min(scores[a] for a in valid)
    tied = [a for a in valid if abs(scores[a] - best) < 1e-10]
    return tied[0], len(tied), best


def storage(value, arm, compare):
    grid, mask, kernel = 53 * 53 * 8, 53 * 53, 4 * 107 * 107 * 8
    if arm in ("full_bayes", "recent32", "recent32_hard"):
        mutable = {"support_or_visited_mask": mask}
        immutable = {}
        if arm == "full_bayes":
            mutable["probability_grid"] = grid
        else:
            mutable["32_public_observation_ring"] = 32 * 3 * 8
            immutable["initial_log_prior"] = grid
        for key, expected in (("mutable_arrays", mutable), ("immutable_arrays", immutable),
                              ("mutable_array_bytes", sum(mutable.values())), ("immutable_array_bytes", sum(immutable.values())),
                              ("decoder_return_array_bytes", 2 * grid), ("python_metadata_included", False)):
            compare.same(value[key], expected, f"{arm}.storage.{key}")
    else:
        q = 53 if arm == "exact_log" else int(arm.split("_")[0][3:])
        for key, expected in (("q", q), ("evidence_array_bytes", q * q * 8), ("support_mask_bytes", mask),
                              ("state_array_bytes", q * q * 8 + mask), ("shared_array_bytes", kernel + 3 * grid),
                              ("float64_grid_bytes", grid), ("persistent_full_belief", False),
                              ("persistent_history", False), ("persistent_increment_cache", False)):
            compare.same(value[key], expected, f"{arm}.storage.{key}")


def analyze(run, original, summary, done, compare, budget):
    # Only public packet, recorded action and scores enter the per-prefix joins.
    native = {}
    with (original / "transitions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["arm"] == "space_full":
                native.setdefault(row["seed"], []).append({"public": row["public"],
                    **({} if row["kind"] == "reset" else {"action": row["action"], "scores": row["scores"]})})
    cases = [json.loads(line) for line in (run / "cases.jsonl").open()]
    require([c["seed"] for c in cases] == list(range(610001, 610097)) and set(native) == set(range(610001, 610097)), "96 ordered cases")
    rows = iter(json.loads(line) for line in (run / "prefixes.jsonl").open())
    progress = {"completed_cases": 96, "prefixes": 0, "eligible_prefixes": 0,
                "qualified_prefixes": dict.fromkeys(("full_bayes", *QUAL), 0),
                "maximum_score_error": dict.fromkeys(("full_bayes", *QUAL), 0.0),
                "maximum_qualification_tv": dict.fromkeys(QUAL, 0.0),
                "reference_zero_probability_on_hard_support_prefixes": 0,
                "maximum_reference_zero_probability_on_hard_support_cells": 0}
    for ci, case in enumerate(cases):
        budget()
        trace = native[case["seed"]]
        count = len(trace) - 1
        compare.same({k: case[k] for k in ("block", "initial_hit", "decisions", "eligible_prefixes")},
                     {"block": ci // 12, "initial_hit": 1 + (ci % 12) // 4, "decisions": count,
                      "eligible_prefixes": max(0, count - 33)}, "case coverage")
        all_values, eligible_values, previous_excluded = [], [], 0
        times = {arm: {key: [] for key in TIMES[1:]} for arm in ARMS}
        underflow = {arm: {"prefixes_with_probability_underflow": 0, "maximum_underflow_cells": 0} for arm in ARMS}
        for t in range(count):
            row = next(rows, None)
            require(row is not None, "missing prefix")
            compare.same({k: row[k] for k in ("seed", "block", "initial_hit", "completed_steps")},
                         {"seed": case["seed"], "block": ci // 12, "initial_hit": case["initial_hit"], "completed_steps": t}, "prefix identity")
            compare.same(row["position"], trace[t]["public"]["position"], "public position")
            require(trace[t]["public"]["done"] is False, "no terminal observation input")
            shift = (ci + t) % 12
            compare.same(row["arm_order"], list(ARMS[shift:] + ARMS[:shift]), "rotation")
            require(set(row["modes"]) == set(ARMS), "complete arms")
            mask = integer(row["persistent_excluded_cells"], "mask", max(1, previous_excluded), 2808)
            previous_excluded = mask
            modes, flat = row["modes"], {}
            full_scores = modes["full_bayes"]["scores"]
            full_action, _, best = choice(full_scores, row["position"])
            for arm in ARMS:
                mode = modes[arm]
                action, ties, _ = choice(mode["scores"], row["position"])
                compare.same(mode["action"], action, "first action")
                compare.same(mode["tie_count"], ties, "tie count")
                compare.same(mode["matches_full_action"], action == full_action, "action agreement")
                compare.same(mode["full_objective_excess"], float(full_scores[action] - best), "heuristic excess")
                finite(mode["tv_to_full"], "TV", 0, 1 + 1e-12)
                finite(mode["kl_full_to_approx"], "KL", -1e-10)
                support = integer(mode["finite_log_support_cells"], "log support", 1, 2809)
                positive = integer(mode["positive_probability_cells"], "probability support", 1, support)
                missing = integer(mode["probability_underflow_cells"], "underflow", 0, 2809)
                compare.same(missing, support - positive, "support difference")
                if arm in (*QUAL, *CANDIDATES, "recent32_hard"):
                    compare.same(support, 2809 - mask, "hard-mask support count")
                elif arm == "recent32":
                    require(support >= 2809 - mask, "recent control cannot add forgotten hard exclusions")
                underflow[arm]["prefixes_with_probability_underflow"] += int(missing > 0)
                underflow[arm]["maximum_underflow_cells"] = max(underflow[arm]["maximum_underflow_cells"], missing)
                if arm in ("full_bayes", *QUAL):
                    compare.same(action, trace[t + 1]["action"], "qualified saved action")
                    recorded = trace[t + 1]["scores"]
                    require(all((a is None) == (b is None) for a, b in zip(mode["scores"], recorded, strict=True)), "qualified score support")
                    error = max(abs(a - b) for a, b in zip(mode["scores"], recorded, strict=True) if a is not None)
                    require(error <= 1e-8, "qualified original-score tolerance")
                    progress["maximum_score_error"][arm] = max(progress["maximum_score_error"][arm], error)
                    progress["qualified_prefixes"][arm] += 1
                if arm in QUAL:
                    require(mode["tv_to_full"] <= 1e-10, "qualified posterior witness")
                    progress["maximum_qualification_tv"][arm] = max(progress["maximum_qualification_tv"][arm], mode["tv_to_full"])
                for key in TIMES[1:]:
                    times[arm][key].append(finite(mode["timing"][key], key, 0))
                if t == 0:
                    compare.same(mode["timing"]["update_seconds"], 0.0, "no initial update")
                flat.update({f"{arm}.{key}": float(mode[key]) for key in METRICS})
            compare.same(modes["full_bayes"]["tv_to_full"], 0.0, "reference TV")
            compare.same(modes["full_bayes"]["kl_full_to_approx"], 0.0, "reference KL")
            lost = 2809 - mask - modes["full_bayes"]["positive_probability_cells"]
            compare.same(row["reference_zero_probability_on_retained_support_cells"], lost, "reference underflow witness")
            require(lost >= 0, "reference retained support")
            progress["reference_zero_probability_on_hard_support_prefixes"] += int(lost > 0)
            progress["maximum_reference_zero_probability_on_hard_support_cells"] = max(progress["maximum_reference_zero_probability_on_hard_support_cells"], lost)
            require(set(row["fill_sensitivity"]) == {str(q) for q in RANKS}, "all fill contrasts")
            for q in RANKS:
                pair = row["fill_sensitivity"][str(q)]
                a, b = (modes[f"dct{q}_{fill}"] for fill in ("neutral", "nearest"))
                compare.same(pair["actions_disagree"], a["action"] != b["action"], "fill action sensitivity")
                tv = finite(pair["tv_between_extensions"], "fill TV", 0, 1 + 1e-12)
                require(abs(a["tv_to_full"] - b["tv_to_full"]) <= tv + 1e-12
                        and tv <= a["tv_to_full"] + b["tv_to_full"] + 1e-12, "TV triangle consistency")
                flat.update({f"fill.q{q}.{key}": float(value) for key, value in pair.items()})
            all_values.append(flat)
            if t > 32:
                eligible_values.append(flat)
                progress["eligible_prefixes"] += 1
            progress["prefixes"] += 1
        compare.same(case["all_means"], weighted(all_values), "per-case all means")
        compare.same(case["eligible_means"], weighted(eligible_values), "per-case eligible means")
        compare.same(case["underflow"], underflow, "per-case underflow")
        for arm in ARMS:
            timing = case["timing"][arm]
            finite(timing["initialization_seconds"], "initialization", 0)
            for key in TIMES[1:]:
                compare.same(timing[key], math.fsum(times[arm][key]), "case timing sum", 1e-9)
            compare.same(timing["total_seconds"], math.fsum(timing[key] for key in TIMES), "disjoint timing total", 1e-9)
            for key, expected in (("update_calls", count - 1), ("decode_calls", count), ("planner_calls", count)):
                compare.same(timing[key], expected, "operation counts")
            storage(case["storage"][arm], arm, compare)
            compare.same(case["storage"][arm], summary["storage_by_arm"][arm], "constant storage")
    require(next(rows, None) is None, "no extra prefixes")
    require(progress["prefixes"] == 2164 and progress["eligible_prefixes"] == 538, "complete fixed populations")
    compare.same(summary["progress"], progress, "summary progress")
    compare.same(done["progress"], progress, "receipt progress")
    compare.same(summary["qualification_rates"], {arm: 1.0 for arm in ("full_bayes", *QUAL)}, "all qualification rates")
    compare.same(summary["underflow_by_arm"], {arm: {
        "prefixes_with_probability_underflow": sum(c["underflow"][arm]["prefixes_with_probability_underflow"] for c in cases),
        "maximum_underflow_cells": max(c["underflow"][arm]["maximum_underflow_cells"] for c in cases)} for arm in ARMS},
        "aggregate underflow")
    weights = summary["initial_hit_weights"]
    require(set(weights) == {"1", "2", "3"} and all(finite(w, "mixture", 0, 1) > 0 for w in weights.values())
            and abs(math.fsum(weights.values()) - 1) < 1e-12, "initial mixture")
    native_summary = read(original / "summary.json")
    compare.same(weights, native_summary["initial_hit_weights"], "original initial mixture")
    compare.same(summary["all_prefixes"], pooled(cases, weights), "all aggregates")
    compare.same(summary["after32_prefixes"], pooled(cases, weights, True), "eligible aggregates")
    case_weights = [weights[str(c["initial_hit"])] / 32 for c in cases]
    for arm in ARMS:
        rows_t = [{key: c["timing"][arm][key] for key in (*TIMES, "total_seconds")} for c in cases]
        compare.same(summary["mixture_weighted_per_case_timing_seconds"][arm], weighted(rows_t, case_weights), "weighted timings", 1e-9)
    shared = finite(summary["shared_model_initialization_seconds"], "shared initialization", 0)
    measured = math.fsum([shared, *(c["timing"][a]["total_seconds"] for c in cases for a in ARMS)])
    compare.same(summary["sum_measured_controller_seconds"], measured, "sum all controllers once", 1e-9)
    wall = finite(summary["whole_elapsed_before_summary_seconds"], "elapsed", measured - 1e-9, done["wall_seconds"])
    compare.same(summary["unattributed_elapsed_before_summary_seconds"], wall - measured, "whole overhead", 1e-9)
    grid, kernel = 53 * 53 * 8, 4 * 107 * 107 * 8
    for key, value in (("kernel_array_bytes", kernel), ("initial_log_prior_array_bytes", 3 * grid), ("shared_array_bytes", kernel + 3 * grid)):
        compare.same(summary["shared_model"][key], value, "shared storage")
    compare.same(summary["additional_readonly_planner_kernel_bytes"], kernel, "additional kernel")
    compare.same(summary["decoder_return_arrays_bytes_per_arm"], 2 * grid, "decoder outputs")
    compare.same(summary["arm_roles"], {arm: "compression candidate" if arm in CANDIDATES else
        "qualification only" if arm in QUAL else "reference" if arm == "full_bayes" else
        "recent-history control" for arm in ARMS}, "all arm roles")
    require(set(summary["numerical_thread_environment"]) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}
            and set(summary["numerical_thread_environment"].values()) == {"1"}, "recorded single numerical thread")
    require(summary["original_learned_pilot_opportunity"] is False and summary["new_quality_gate"] is None, "no reversed gate")
    return {"cases": 96, "prefixes": 2164, "eligible_prefixes": 538, "arms": 12, "arm_prefixes": 2164 * 12,
            "qualification_arm_prefixes": 2164 * 4, "original_gate_revised": False,
            "recomputed_all_prefixes": pooled(cases, weights), "recomputed_after32_prefixes": pooled(cases, weights, True)}


def execute(args):
    args.run, args.plan, args.out = args.run.resolve(), args.plan.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source = None
    started = last = None
    comparison = Compare()
    try:
        source = sha(__file__)
        require(sha(ROOT / CLOCK) == CLOCK_PIN, "qualified native clock")
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research.suspend_clock import SuspendClock
        clock = SuspendClock()
        started = clock.now_ns()

        def budget():
            nonlocal last
            last = clock.now_ns()
            require(last - started < 300 * 10**9, "native audit deadline")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) <= LIMITS["rss_bytes"], "audit RSS")
            require(sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "audit output")

        def timeout(*_):
            raise TimeoutError("audit emergency timer expired")

        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, 300)
        write(args.out / "started.json", {"source_sha256": source, "request": {k: str(v) for k, v in vars(args).items()},
              "limits": LIMITS, "scope": SCOPE, "clock_backend": clock.backend, "started_ns": started})
        plan, done, original, external = authenticate(args, budget)
        summary = read(args.run / "summary.json")
        comparison.same(summary["configuration"], plan["configuration"], "summary configuration")
        result = analyze(args.run, original, summary, done, comparison, budget)
        write(args.out / "summary.json", {"agreement": True, "scope": SCOPE, "checks": comparison.checks,
              "maximum_absolute_difference": comparison.maximum_difference, **result})
        manifest(args.run, done, PAYLOADS, budget)
        for path, pin in external + [(args.plan, args.plan_sha256), (args.run / "receipt.json", args.receipt_sha256), (Path(__file__), source)]:
            require(sha(path) == pin, "unchanged external/source pin")
        for name, pin in plan["sources"].items():
            require(sha(ROOT / name) == pin, "unchanged producer sources")
        budget()
        receipt = {"status": "completed", "agreement": True, "scope": SCOPE, "source_sha256": source,
                   "clock_source_sha256": CLOCK_PIN, "plan_sha256": args.plan_sha256,
                   "producer_receipt_sha256": args.receipt_sha256,
                   "producer_summary_sha256": done["files"]["summary.json"]["sha256"],
                   "checks": comparison.checks, "maximum_absolute_difference": comparison.maximum_difference,
                   "limits": LIMITS, "model_calls": 0, "environment_calls": 0, "clock_backend": clock.backend,
                   "started_ns": started, "finished_ns": last, "elapsed_ns": last - started,
                   "wall_seconds": (last - started) / 1e9, "timing_scope": "Through pre-receipt check; publication rechecked before return.",
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                   "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in args.out.iterdir() if p.is_file()}}
        write(args.out / "receipt.json", receipt)
        budget()
        print(json.dumps({"status": "completed", "agreement": True, "checks": comparison.checks,
                          "maximum_absolute_difference": comparison.maximum_difference}), flush=True)
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"status": "failed", "agreement": False, "source_sha256": source,
                  "scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(), "checks": comparison.checks,
                  "maximum_absolute_difference": comparison.maximum_difference,
                  "last_successful_elapsed_ns": None if last is None or started is None else last - started,
                  "timing_available": False, "wall_seconds": None})
        except BaseException as secondary:  # noqa: BLE001 - preserve the primary failure.
            if hasattr(error, "add_note"):
                error.add_note(f"Failure publication also failed: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())
