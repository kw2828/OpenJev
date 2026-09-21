"""Independent primary arithmetic audit, usable only after all twelve fits.

Stdlib, NumPy and the standalone clock only. No producer/reporter/metric/model imports. Full execution
bytes, frozen sources, terminal process witness and report are authenticated;
training coverage and tensor/gradient provenance remain inherited from that
report. Reconstructs 144 panel/stratum fit cells, 24 literal cells, arm means and
seven scientific conditions. No service/type/factorial-interaction re-audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[3]
VERSION = "dialogue-observation-independent-audit-v2"
STUDY = "dialogue-observation-scientific-v2"
REPORT = "dialogue-observation-report-v2"
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
SEEDS = (6901, 6902, 6903)
FIT_ORDER = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
PANELS = ("all", "seen", "unseen")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
BINS = (*STRATA[:2], "first_assignment", "revision", "clear")
NONE, DC = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
WATCHDOG = "scripts/supervise_dialogue_observation_v2.py"
REPORTER = "scripts/report_dialogue_observation_v2.py"
METRICS = "src/openjev/research/dialogue_observation_metrics.py"
REPORTER_PIN = "8254df8f1973c02810d1fc34034a8546d02042d8067214f0de19e0e3ae21c4be"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_TEST = "tests/test_suspend_clock.py"
OLD_PLAN = "output/dialogue-observation-learning-v1/scientific-freeze-01/plan.json"
OLD_PLAN_PIN = "acb79b4600c66966762895d28eb2dc1d2be15c761d677c5e87c5750dde47f237"
FAILED_PIN = "41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f"
V2_SOURCES = {"scripts/study_dialogue_observation_v2.py", "tests/test_study_dialogue_observation_v2.py",
              REPORTER, "tests/test_report_dialogue_observation_v2.py", WATCHDOG,
              "tests/test_supervise_dialogue_observation_v2.py", CLOCK, CLOCK_TEST,
              "research/dialogue-observation-learning-protocol-v2.md",
              "output/dialogue-observation-learning-v2/result-audit-01/audit.py",
              "output/dialogue-observation-learning-v2/result-audit-01/test_audit.py"}
METRICS_PIN = "e0913097f05b59c18d2e57b6cf295440639f979ea260ed46545e53ea229be766"
CAPS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 256 * 1024**2}
SCOPE = ("Independent saved probability validation, canonical identity and 144 fit plus24 literal panel/stratum cells, "
         "equal-seed arm means and all seven conditions. Complete execution/freeze/source/report/supervisor bytes authenticated. "
         "Detailed prepared joins, per-update encoder/state/gradient witnesses, initializer truth and lexical-register semantics "
         "inherit the successful production report; no model replay, checkpoint decoding, service/type tables, "
         "paired descriptive contrasts or factorial interaction independently recomputed. No efficacy beyond exposed DEV.")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(x):
    return type(x) is str and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def describe(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def integer(x):
    return type(x) is int and x >= 0


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def manifest(directory, expected, terminal):
    members = set()
    for path in directory.rglob("*"):
        need(not path.is_symlink(), "No symlink artifacts")
        if path.is_file() and path != directory / terminal:
            members.add(path.relative_to(directory).as_posix())
    need(members == set(expected), "Exact artifact closure")
    for name, entry in expected.items():
        need(not Path(name).is_absolute() and ".." not in Path(name).parts, "Safe payload name")
        need(set(entry) == {"sha256", "bytes"} and digest(entry["sha256"]) and integer(entry["bytes"])
             and describe(directory / name) == entry, "Payload identity: " + name)


def failure_lineage(allocation, sources):
    ref = allocation["failed_attempt"]
    need(allocation["version"] == "dialogue-observation-allocation-v2"
         and set(ref) == {"path", "sha256"} and ref["sha256"] == FAILED_PIN
         and sha(ref["path"]) == FAILED_PIN, "Pinned preserved failure")
    prior = read(ref["path"])
    need(prior["status"] == "failed_technical_timing" and prior["required_fits"] == 12
         and prior["resume_permitted"] is False and prior["partial_scoring_permitted"] is False
         and prior["quality_metrics_opened"] is False and prior["individual_predictions_decoded"] is False
         and prior["weights_loaded"] is False and prior["process_group_absent"] is True, "Failed campaign remains unscored")
    need(sha(ROOT / OLD_PLAN) == OLD_PLAN_PIN, "Original source-plan pin")
    old = read(ROOT / OLD_PLAN)["source_sha256"]
    need(len(old) == 53 and set(sources) == set(old) | V2_SOURCES
         and all(sources[name] == pin for name, pin in old.items()), "Preserved 53 plus 11 new sources")


def timing(launch, end, worker_start, worker_end, cap):
    """Independent integer envelope; no subtraction of civil timestamps."""
    backend = launch["clock_backend"]
    need(backend in ("mach_continuous_time", "CLOCK_BOOTTIME") and type(cap) is int and cap > 0,
         "Native timer and positive integral cap")
    need(integer(launch["started_ns"]) and integer(launch["deadline_ns"])
         and launch["deadline_ns"] == launch["started_ns"] + cap * 1_000_000_000, "Absolute allocation deadline")
    for record in (end, worker_end):
        need(all(integer(record[k]) for k in ("started_ns", "finished_ns", "elapsed_ns", "deadline_ns")),
             "Integral elapsed timing record")
        need(record["timing_available"] is True and record["clock_backend"] == backend and record["deadline_ns"] == launch["deadline_ns"]
             and record["started_ns"] <= record["finished_ns"] < record["deadline_ns"]
             and record["elapsed_ns"] == record["finished_ns"] - record["started_ns"]
             and finite(record["wall_seconds"]) and record["wall_seconds"] == record["elapsed_ns"] / 1_000_000_000,
             "Consistent suspend-inclusive interval")
    need(end["status"] == "completed" and end["timing_available"] is True
         and end["started_ns"] == launch["started_ns"], "Successful parent elapsed witness")
    need(worker_start["timing_available"] is True and worker_start["clock_backend"] == backend and worker_start["deadline_ns"] == launch["deadline_ns"]
         and integer(worker_start["started_ns"]) and worker_start["started_ns"] == worker_end["started_ns"]
         and worker_start["parent_started_ns"] == worker_end["parent_started_ns"] == launch["started_ns"], "Inherited worker deadline")
    need(launch["started_ns"] <= worker_start["started_ns"] <= worker_end["finished_ns"] <= end["finished_ns"]
         < launch["deadline_ns"], "Worker enclosed by strictly on-time parent")
    need(finite(launch["started_unix"]) and launch["started_unix"] > 0
         and finite(end["finished_unix"]) and end["finished_unix"] > 0, "Civil provenance only")


def authenticate(args):
    """Never decode evaluator rows or NPZ before every terminal/report join."""
    external = {args.run / "completed.json": args.completed_sha256, args.plan: args.plan_sha256,
                args.launch: args.launch_sha256, args.terminal: args.terminal_sha256,
                args.report / "receipt.json": args.report_receipt_sha256,
                args.report / "summary.json": args.report_summary_sha256}
    for path, pin in external.items():
        need(digest(pin) and sha(path) == pin, "External input pin: " + str(path))
    done, plan = read(args.run / "completed.json"), read(args.plan)
    need(done["status"] == "completed" and done["phase"] == "train" and done["version"] == STUDY
         and done["fit_count"] == 12 and [x["fit_id"] for x in done["fits"]] == FIT_ORDER, "Complete ordered twelve-fit run")
    root_files = {"started.json", "allocation.json", "plan.json", "evaluation-rows.jsonl", "references.json"}
    root_files |= {f"{fit}/{name}" for fit in FIT_ORDER for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    need(set(done["files"]) == root_files, "54-file scientific run closure")
    manifest(args.run, done["files"], "completed.json")
    need(done["plan_sha256"] == args.plan_sha256 == sha(args.run / "plan.json")
         and plan["version"] == STUDY and plan["previous_plan_sha256"] == OLD_PLAN_PIN and plan["fit_order"] == FIT_ORDER, "Frozen scientific plan identity")
    need(plan["source_sha256"] == done["source_sha256"] and len(plan["source_sha256"]) == 64
         and plan["source_sha256"][REPORTER] == REPORTER_PIN and plan["source_sha256"][METRICS] == METRICS_PIN,
         "Frozen scientific source closure")
    frozen = read(args.plan.parent / "completed.json")
    need(frozen["status"] == "completed" and frozen["phase"] == "freeze" and frozen["version"] == STUDY
         and frozen["plan_sha256"] == args.plan_sha256 and frozen["source_sha256"] == plan["source_sha256"], "Successful freeze")
    need(set(frozen["files"]) == {"started.json", "allocation.json", "plan.json", "evaluation-rows.jsonl", "references.json"}
         | {"sources/" + name for name in plan["source_sha256"]}, "Freeze snapshot closure")
    manifest(args.plan.parent, frozen["files"], "completed.json")
    for name, pin in plan["source_sha256"].items():
        need(digest(pin) and sha(ROOT / name) == pin and sha(args.plan.parent / "sources" / name) == pin, "Source bytes: " + name)
    for name in ("allocation.json", "evaluation-rows.jsonl", "references.json"):
        need(describe(args.run / name) == describe(args.plan.parent / name), "Frozen copy: " + name)
    allocation = read(args.run / "allocation.json")
    need(allocation == plan["allocation"] and sha(args.run / "allocation.json") == done["allocation_sha256"] == plan["allocation_sha256"],
         "Allocation identity")
    failure_lineage(allocation, plan["source_sha256"])
    freeze_start = read(args.plan.parent / "started.json")
    need(frozen["timing_available"] is True and freeze_start["timing_available"] is True
         and frozen["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
         and all(integer(frozen[k]) for k in ("started_ns", "finished_ns", "deadline_ns", "elapsed_ns"))
         and frozen["started_ns"] <= frozen["finished_ns"] < frozen["deadline_ns"]
         and frozen["elapsed_ns"] == frozen["finished_ns"] - frozen["started_ns"]
         and finite(frozen["wall_seconds"]) and frozen["wall_seconds"] == frozen["elapsed_ns"] / 1_000_000_000
         and frozen["deadline_ns"] == frozen["started_ns"] + allocation["freeze_limits"]["wall_seconds"] * 1_000_000_000
         and all(freeze_start[k] == frozen[k] for k in ("clock_backend", "started_ns", "deadline_ns", "parent_started_ns"))
         and frozen["parent_started_ns"] == frozen["started_ns"]
         and freeze_start["supervision_sha256"] is None and frozen["supervision_sha256"] is None, "Standalone freeze timing")
    for name in ("cost_completed", "cost_audit", "protocol"):
        reference = allocation[name]
        need(sha(reference["path"]) == reference["sha256"], "Allocation-bound parent: " + name)
    prior_done, prior_audit = read(allocation["cost_completed"]["path"]), read(allocation["cost_audit"]["path"])
    need(prior_done["status"] == "completed" and prior_audit["status"] == "completed" and prior_audit["agreement"] is True
         and prior_audit["execution_completed_sha256"] == allocation["cost_completed"]["sha256"], "Qualified parent terminal witness")
    limits = allocation["limits"]
    need(finite(done["wall_seconds"]) and 0 < done["wall_seconds"] < limits["wall_seconds"]
         and integer(done["peak_rss_bytes"]) and 0 < done["peak_rss_bytes"] <= limits["rss_bytes"]
         and 0 <= done["sampled_mps_current_max_bytes"] <= done["sampled_mps_driver_max_bytes"] <= limits["mps_driver_bytes"]
         and sum(x["bytes"] for x in done["files"].values()) + (args.run / "completed.json").stat().st_size <= limits["output_bytes"],
         "Recorded execution resource limits")
    need(done["counts"] == plan["expected"]["all_fits"] and plan["expected"]["fits"] == 12
         and done["quality_scoring_in_runner"] is False and done["official_test_opened"] is False
         and done["external_model_api_calls"] == 0, "Complete no-selection execution witness")
    initial, encoder = {}, None
    for item in done["fits"]:
        path = args.run / item["fit_id"] / "completed.json"
        need(sha(path) == item["completed_sha256"], "Fit completion join")
        fit = read(path)
        arm, seed = item["fit_id"].rsplit("-", 1)
        need(fit["version"] == STUDY and fit["status"] == "completed" and fit["fit_id"] == item["fit_id"]
             and fit["arm"] == arm and fit["seed"] == int(seed) and fit["evaluation"]["rows"] == plan["expected"]["per_fit"]["evaluation_endpoints"],
             "Fit identity/complete evaluation")
        need(set(fit["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}, "Fit payload closure")
        for name, desc in fit["files"].items():
            need(desc == done["files"][item["fit_id"] + "/" + name], "Root/fit payload identity")
        need({k: fit["counts"][k] for k in plan["expected"]["per_fit"]} == plan["expected"]["per_fit"]
             and fit["encoder_work"] == plan["expected"]["encoder_work_per_fit"]
             and {k: fit["invariants"][k] for k in plan["expected"]["state_counts_per_fit"]} == plan["expected"]["state_counts_per_fit"],
             "Complete source-bound work witness")
        state = fit["initial_sha256"]
        need(set(state) == {"encoder", "memory"} and all(digest(v) for v in state.values())
             and initial.setdefault(seed, state) == state, "Paired initialization witness")
        encoder = state["encoder"] if encoder is None else encoder
        need(state["encoder"] == encoder and digest(fit["final_encoder_sha256"])
             and (fit["final_encoder_sha256"] != encoder) == arm.startswith("trainable"), "Frozen/trainable encoder witness")
    launch, end = read(args.launch), read(args.terminal)
    need(launch["command"] == end["command"] and type(launch["command"]) is list, "Supervisor command identity")
    command = launch["command"]
    need(Path(command[0]).resolve() == Path(sys.executable).resolve(), "Supervisor interpreter")
    tail = command[2:] if len(command) > 1 and command[1] == "-u" else command[1:]
    need(len(tail) == 10 and (ROOT / tail[0]).resolve() == ROOT / "scripts/study_dialogue_observation_v2.py"
         and tail[1] == "train", "Supervised scientific program")
    flags = dict(zip(tail[2::2], tail[3::2], strict=True))
    started = read(args.run / "started.json")
    need(set(flags) == {"--plan", "--plan-sha256", "--out", "--supervision"} and flags["--plan-sha256"] == args.plan_sha256
         and Path(flags["--plan"]).resolve() == args.plan.resolve()
         and Path(flags["--out"]).resolve() == Path(started["request"]["out"]).resolve()
         and Path(flags["--supervision"]).resolve() == args.launch.resolve()
         == Path(started["request"]["supervision"]).resolve()
         and started["request"]["command"] == "train" and started["request"]["plan_sha256"] == args.plan_sha256,
         "Supervised plan/output request")
    need(integer(launch["pid"]) and launch["pid"] > 0 and launch["pid"] == launch["pgid"] == end["pgid"]
         and launch["watchdog_sha256"] == plan["source_sha256"][WATCHDOG] and launch["cap_seconds"] == limits["wall_seconds"]
         and type(end["returncode"]) is int and end["returncode"] == 0 and end["error"] is None
         and end["timed_out"] is False and end["group_absent"] is True, "Successful independent supervisor")
    need(launch["version"] == end["version"] == "dialogue-observation-supervision-v2"
         and launch["clock_source_sha256"] == plan["source_sha256"][CLOCK]
         and all(end[k] == launch[k] for k in ("parent_pid", "pid", "pgid", "cwd", "cap_seconds",
                                               "watchdog_sha256", "clock_source_sha256", "started_unix"))
         and integer(launch["parent_pid"]) and launch["parent_pid"] > 0
         and end["clock_error"] is None and end["cleanup"]["reaped"] is True
         and end["cleanup"]["errors"] == [] and end["cleanup"]["group_absent"] is True,
         "Supervisor origin, clock and cleanup")
    need(started["supervision_sha256"] == done["supervision_sha256"] == args.launch_sha256,
         "Worker launch digest identity")
    timing(launch, end, started, done, limits["wall_seconds"])
    report = read(args.report / "receipt.json")
    need(report["version"] == REPORT and report["status"] == "completed" and report["technical_validity_passed"] is True
         and report["execution_completed_sha256"] == args.completed_sha256 and report["plan_sha256"] == args.plan_sha256
         and report["launch_sha256"] == args.launch_sha256 and report["terminal_sha256"] == args.terminal_sha256
         and report["execution_source_sha256"] == plan["source_sha256"], "Successful production report binding")
    need(set(report["source_sha256"]) == {REPORTER, METRICS, "tests/test_report_dialogue_observation_v2.py", "tests/test_dialogue_observation_metrics.py", CLOCK, CLOCK_TEST}
         and all(plan["source_sha256"][name] == pin for name, pin in report["source_sha256"].items()), "Report implementation identity")
    need(set(report["files"]) == {"started.json", "summary.json", "report.md"}, "Report closure")
    manifest(args.report, report["files"], "receipt.json")
    summary = read(args.report / "summary.json")
    need(summary["status"] == "completed" and summary["version"] == REPORT and summary["technical_validity_passed"] is True
         and summary["technical_complete_fits"] == 12 and summary["execution_completed_sha256"] == args.completed_sha256
         and summary["plan_sha256"] == args.plan_sha256 and set(summary["fits"]) == set(FIT_ORDER), "Complete authenticated scientific report")
    need(sha(args.run / "evaluation-rows.jsonl") == plan["evaluation_rows_sha256"]
         and sha(args.run / "references.json") == plan["references_sha256"], "Canonical evaluator/reference pins")
    return done, plan, report, summary, external


def layout(rows):
    need(type(rows) is list and rows, "Nonempty evaluator rows")
    n = len(rows)
    targets, counts, bins, unseen = [], [], [], []
    endpoint_ids, source_ids, schemas, panels = set(), set(), {}, {}
    for i, row in enumerate(rows):
        need(row["row_index"] == i and integer(row["row_index"]) and row["split"] == "dev"
             and all(integer(row[k]) for k in ("source_row_index", "query_index", "time"))
             and all(type(row[k]) is str and row[k] for k in ("dialogue_id", "service", "slot")), "Canonical evaluator order/identity")
        source = (row["dialogue_id"], row["source_row_index"])
        endpoint = (row["dialogue_id"], row["time"], row["query_index"])
        need(source not in source_ids and endpoint not in endpoint_ids, "No duplicate scored endpoints")
        source_ids.add(source)
        endpoint_ids.add(endpoint)
        ids, values = row["candidate_ids"], row["candidate_values"]
        need(type(ids) is list and type(values) is list and 3 <= len(ids) <= 12 and len(ids) == len(values)
             and all(type(x) is str for x in ids) and len(set(ids)) == len(ids)
             and ids.count(NONE) == ids.count(DC) == 1, "Distinct supplied support and reserved identities")
        for cid, value in zip(ids, values, strict=True):
            need(value is None if cid in (NONE, DC) else type(value) is str and value.strip() and cid == "value:" + value,
                 "Canonical ordinary/reserved value identity")
        schema = (row["service"], row["slot"], tuple(ids), tuple(values))
        need(schemas.setdefault(row["query_index"], schema) == schema, "Stable candidate schema")
        label = row["label_index"]
        need(integer(label) and label < len(ids) and ids[label] == row["label_id"], "Target index/ID identity")
        need(row["bin"] in BINS and type(row["unseen"]) is bool
             and panels.setdefault(row["service"], row["unseen"]) == row["unseen"], "Transition/service panel")
        stratum = row["bin"] if row["bin"] in STRATA[:2] else "changed"
        need("stratum" not in row or row["stratum"] == stratum, "Declared stratum")
        need("dontcare" not in row or (type(row["dontcare"]) is bool and row["dontcare"] == (row["label_id"] == DC)), "Reserved DONTCARE flag")
        targets.append(label)
        counts.append(len(ids))
        bins.append(stratum)
        unseen.append(row["unseen"])
    mask = np.arange(12)[None, :] < np.array(counts)[:, None]
    return {"target": np.array(targets, np.int64), "mask": mask, "strata": np.array(bins),
            "unseen": np.array(unseen, bool), "rows": n}


def group_scores(choices, info, nll=None, brier=None):
    correct = choices == info["target"]
    def cell(mask):
        count = int(mask.sum())
        hits = int(np.count_nonzero(correct & mask))
        result = {"count": count, "correct": hits, "incorrect": count-hits,
                  "accuracy": hits/count if count else None, "error": (count-hits)/count if count else None}
        if nll is not None:
            result.update(nll=math.fsum(float(x) for x in nll[mask])/count if count else None,
                          brier=math.fsum(float(x) for x in brier[mask])/count if count else None)
        return result
    result = {}
    for panel in PANELS:
        selected = np.ones(info["rows"], bool) if panel == "all" else info["unseen"] if panel == "unseen" else ~info["unseen"]
        strata = {s: cell(selected & (info["strata"] == s)) for s in STRATA}
        names = ("accuracy", "nll", "brier") if nll is not None else ("accuracy",)
        macro = {k: math.fsum(strata[s][k] for s in STRATA)/3 if all(strata[s]["count"] for s in STRATA) else None for k in names}
        result[panel] = {"micro": cell(selected), "strata": strata, "macro_three": macro}
    return result


def score(logs, indices, info):
    n, mask = info["rows"], info["mask"]
    need(isinstance(indices, np.ndarray) and indices.dtype == np.int64 and np.array_equal(indices, np.arange(n)), "Raw row alignment")
    need(isinstance(logs, np.ndarray) and logs.dtype == np.float32 and logs.shape == (n, 12), "Raw float32 N by12 logs")
    need(np.isfinite(logs[mask]).all() and np.isneginf(logs[~mask]).all(), "Finite supported logs and negative-infinite padding")
    logs64 = logs.astype(np.float64)
    probabilities = np.exp(logs64)
    mass_error = float(np.max(np.abs(np.sum(probabilities, axis=1)-1)))
    need(mass_error <= 2e-6, "Raw supported mass tolerance")
    choices = np.argmax(logs64, axis=1)
    nll = -logs64[np.arange(n), info["target"]]
    # Independent candidate sum, retaining finite-log underflow without flooring.
    brier = np.zeros(n, np.float64)
    for candidate in range(12):
        residual = probabilities[:, candidate] - (info["target"] == candidate)
        brier += residual * residual
    return {"panels": group_scores(choices, info, nll, brier),
            "validation": {"rows": n, "supported_candidate_positions": int(mask.sum()), "maximum_mass_error": mass_error,
                           "tolerance": 2e-6, "exact_top1_tie_rows": int(((logs64 == logs64.max(axis=1)[:, None]).sum(axis=1) > 1).sum()),
                           "float64_exponent_underflow_positions": int(np.count_nonzero(probabilities[mask] == 0))}}


def literal(choices, indices, rows, info):
    need(type(indices) is list and all(integer(x) for x in indices) and indices == list(range(info["rows"])), "Literal row identity")
    need(type(choices) is list and len(choices) == info["rows"], "Literal coverage")
    for choice, row in zip(choices, rows, strict=True):
        need(integer(choice) and choice < len(row["candidate_ids"]) and row["candidate_ids"][choice] != DC,
             "Literal choice support; never reserved DONTCARE")
    return {"panels": group_scores(np.asarray(choices, np.int64), info)}


def macro_fraction(fit, panel):
    groups = fit["panels"][panel]["strata"]
    return sum((Fraction(groups[s]["correct"], groups[s]["count"]) for s in STRATA), Fraction())/3 if all(groups[s]["count"] for s in STRATA) else None


def decision(fits):
    need(set(fits) == set(FIT_ORDER), "All four arms and all three seeds")
    control = [fits[f"frozen_numbers-{seed}"] for seed in SEEDS]
    treatment = [fits[f"trainable_numbers-{seed}"] for seed in SEEDS]
    checks = []
    def count_rule(name, pairs, threshold, op):
        valid = all(a is not None and b is not None for a, b in pairs)
        deltas = [a-b for a, b in pairs] if valid else None
        mean = sum(deltas, Fraction())/3 if valid else None
        checks.append({"name": name, "passed": valid and (mean >= threshold if op == "ge" else mean <= threshold),
                       "paired_seed_values": [float(d) for d in deltas] if valid else [None if a is None or b is None else float(a-b) for a,b in pairs],
                       "mean": float(mean) if valid else None, "threshold": float(threshold), "comparison": op, "arithmetic": "exact"})
    unseen = [(macro_fraction(a, "unseen"), macro_fraction(b, "unseen")) for a,b in zip(treatment,control,strict=True)]
    count_rule("unseen_macro_gain_1pp", unseen, Fraction(1,100), "ge")
    wins = sum(a>b for a,b in unseen) if all(a is not None and b is not None for a,b in unseen) else None
    checks.append({"name": "unseen_macro_strict_paired_wins", "passed": wins is not None and wins >= 2,
                   "wins": wins, "required": 2, "arithmetic": "exact"})
    for name in ("nll", "brier"):
        a = [f["panels"]["unseen"]["micro"][name] for f in treatment]
        b = [f["panels"]["unseen"]["micro"][name] for f in control]
        valid = all(finite(v) for v in a+b)
        am, bm = (math.fsum(a)/3, math.fsum(b)/3) if valid else (None,None)
        checks.append({"name": f"unseen_micro_{name}_nonworse", "passed": valid and am <= bm,
                       "paired_seed_values": [x-y for x,y in zip(a,b,strict=True)] if valid else None,
                       "mean": am-bm if valid else None, "primary_mean": am, "control_mean": bm,
                       "threshold": 0., "comparison": "le", "arithmetic": "float64 arm means, no epsilon"})
    count_rule("seen_macro_deficit_at_most_1pp", [(macro_fraction(a,"seen"),macro_fraction(b,"seen")) for a,b in zip(treatment,control,strict=True)], -Fraction(1,100), "ge")
    for panel in ("seen", "unseen"):
        pairs = []
        for a,b in zip(treatment,control,strict=True):
            cells = [f["panels"][panel]["strata"]["assigned_retention"] for f in (a,b)]
            pairs.append(tuple(Fraction(c["incorrect"],c["count"]) if c["count"] else None for c in cells))
        count_rule(f"{panel}_assigned_retention_error_increase_at_most_half_pp", pairs, Fraction(1,200), "le")
    return {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks), "checks_total": 7, "checks": checks}


class Comparison:
    def __init__(self):
        self.count = 0
    def check(self, expected, observed, name="root"):
        if isinstance(expected, dict):
            need(isinstance(observed, dict) and set(expected) <= set(observed), "Missing fields: " + name)
            for key, value in expected.items():
                self.check(value, observed[key], name+"/"+key)
        elif isinstance(expected, list):
            need(type(observed) is list and len(expected) == len(observed), "List identity: " + name)
            for i, (a,b) in enumerate(zip(expected,observed,strict=True)):
                self.check(a,b,name+"/"+str(i))
        else:
            self.count += 1
            if type(expected) is float:
                need(type(observed) in (int,float) and math.isfinite(observed)
                     and math.isclose(expected,observed,rel_tol=1e-12,abs_tol=1e-12), "Numeric disagreement: " + name)
            else:
                need(type(expected) is type(observed) and expected == observed, "Exact disagreement: " + name)


def mean_panels(fits):
    def average(cells):
        result = {}
        for key, value in cells[0].items():
            if isinstance(value,dict): result[key] = average([c[key] for c in cells])
            elif key in ("accuracy","error","nll","brier"):
                values = [c[key] for c in cells]
                result[key] = math.fsum(values)/3 if all(x is not None for x in values) else None
        return result
    return {arm: {"panels": average([fits[f"{arm}-{seed}"]["panels"] for seed in SEEDS])} for arm in ARMS}


def recompute(run, rows, references, published):
    info = layout(rows)
    need(set(published["fits"]) == set(FIT_ORDER), "All published fits")
    checks = Comparison()
    fits = {}
    for name in FIT_ORDER:
        with np.load(run/name/"predictions.npz",allow_pickle=False) as data:
            need(set(data.files) == {"row_indices","log_probs"}, "Exact raw prediction fields")
            fits[name] = score(data["log_probs"],data["row_indices"],info)
        checks.check(fits[name],published["fits"][name],"fits/"+name)
    literals = {name: literal(references[name],references["row_indices"],rows,info) for name in ("original","numbers")}
    for name, result in literals.items():
        checks.check(result,published["references"][name],"references/"+name)
    families = mean_panels(fits)
    for name, result in families.items():
        need(published["factorial"]["families"][name]["fit_names"] == [f"{name}-{s}" for s in SEEDS], "No selected seed mean")
        checks.check(result,published["factorial"]["families"][name]["mean"],"families/"+name)
    gate = decision(fits)
    checks.check(gate,published["continuation"],"continuation")
    return {"fits":fits,"references":literals,"families":families,"continuation":gate,"scalar_checks":checks.count,
            "fit_cells":144,"literal_cells":24,"rows_per_fit":info["rows"]}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value*1024


def execute(args):
    args.out.mkdir(parents=True,exist_ok=False)
    request = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    sources, handler = {}, None
    clock, deadline, elapsed_ns = None, None, None
    def budget():
        nonlocal elapsed_ns
        now = clock.now_ns()
        elapsed_ns = now - deadline.started_ns
        need(now < deadline.expires_ns and peak_rss() <= CAPS["rss_bytes"], "Audit wall/RSS cap")
        need(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) <= CAPS["output_bytes"], "Audit output cap")
    try:
        clock = SuspendClock()
        deadline = clock.deadline_after(CAPS["wall_seconds"])
        def timeout(*_): raise TimeoutError("Audit wall cap")
        handler = signal.signal(signal.SIGALRM,timeout)
        signal.setitimer(signal.ITIMER_REAL,CAPS["wall_seconds"])
        sources = {name:sha(Path(__file__).parent/name) for name in ("audit.py","test_audit.py")}
        write(args.out/"started.json",{"version":VERSION,"request":request,"source_sha256":sources,"limits":CAPS,"scope":SCOPE})
        done,plan,report,published,external = authenticate(args)
        budget()
        with (args.run/"evaluation-rows.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        need(len(rows) == plan["expected"]["per_fit"]["evaluation_endpoints"], "Complete canonical endpoint count")
        result = recompute(args.run,rows,read(args.run/"references.json"),published)
        need(report["continuation_passed"] is result["continuation"]["passed"]
             and report["scientific_checks_passed"] == result["continuation"]["checks_passed"]
             and report["scientific_checks_total"] == 7, "Report receipt decision agreement")
        result.update(version=VERSION,status="completed",agreement=True,scope=SCOPE,
                      execution_completed_sha256=args.completed_sha256,plan_sha256=args.plan_sha256)
        write(args.out/"summary.json",result)
        manifest(args.run,done["files"],"completed.json")
        manifest(args.report,report["files"],"receipt.json")
        for path,pin in external.items(): need(sha(path)==pin,"End input stability")
        for name,pin in plan["source_sha256"].items(): need(sha(ROOT/name)==pin,"End scientific source stability")
        for name,pin in sources.items(): need(sha(Path(__file__).parent/name)==pin,"End audit source stability")
        budget()
        receipt = {"version":VERSION,"status":"completed","agreement":True,"request":request,"source_sha256":sources,
                   "producer_receipt_sha256":args.report_receipt_sha256,"producer_summary_sha256":args.report_summary_sha256,
                   "execution_completed_sha256":args.completed_sha256,"plan_sha256":args.plan_sha256,
                   "launch_sha256":args.launch_sha256,"terminal_sha256":args.terminal_sha256,
                   "continuation_passed":result["continuation"]["passed"],"scientific_checks_passed":result["continuation"]["checks_passed"],
                   "scalar_checks":result["scalar_checks"],"fit_cells":144,"literal_cells":24,"limits":CAPS,"scope":SCOPE,
                   "files":{name:describe(args.out/name) for name in ("started.json","summary.json")},
                   "wall_seconds":elapsed_ns/1_000_000_000,"elapsed_ns":elapsed_ns,"clock_backend":clock.backend,
                   "clock_source_sha256":sha(ROOT/CLOCK),"started_ns":deadline.started_ns,
                   "finished_ns":deadline.started_ns+elapsed_ns,"deadline_ns":deadline.expires_ns,
                   "timing_available":True,"peak_rss_bytes":peak_rss(),"model_calls":0}
        write(args.out/"receipt.json",receipt)
        budget()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if (args.out/"receipt.json").exists(): (args.out/"receipt.json").rename(args.out/"invalid-receipt.json")
            write(args.out/"failed.json",{"version":VERSION,"status":"failed","request":request,"source_sha256":sources,
                  "error_type":type(error).__name__,"error":str(error),"wall_seconds":None,"elapsed_ns":None,
                  "timing_available":False,"last_successful_elapsed_ns":elapsed_ns,
                  "elapsed_scope":"Terminal timing unavailable; last successful check retained separately",
                  "peak_rss_bytes":peak_rss(),"scope":SCOPE,"agreement":False,"model_calls":0})
        except BaseException as secondary:  # noqa: BLE001 - preserve initial audit error
            error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,handler)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","plan","launch","terminal","report","out"): parser.add_argument("--"+name,type=Path,required=True)
    for name in ("completed-sha256","plan-sha256","launch-sha256","terminal-sha256","report-receipt-sha256","report-summary-sha256"):
        parser.add_argument("--"+name,required=True)
    print(json.dumps(execute(parser.parse_args()),sort_keys=True))
