"""Independent saved-output scoring of the complete, fixed warm-start study.

Authenticate source/data/output manifests and actual successful supervision
before decoding evaluator rows or prediction arrays. No model calls, weight
loading, probability repair, temperature fitting, or outcome-based selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import resource
import signal
import sys
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np

from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-warm-pooling-report-v1"
STUDY_VERSION = "dialogue-warm-pooling-v1"
ARMS = ("pooled", "schema_attention", "belief_query", "state_token")
COMPARATORS = ("pooled", "schema_attention", "state_token", "untouched")
SEEDS = (6901, 6902, 6903)
FIT_ORDER = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
SCORE_ORDER = [f"{arm}-{seed}" for seed in SEEDS for arm in (*ARMS, "untouched")]
QUALIFICATION_ORDER = [f"{arm}-{seed}" for seed in SEEDS for arm in ("untouched", *ARMS)]
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
NONE, DONTCARE = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
MAX_CANDIDATES, TRAIN_DIALOGUES, DEV_DIALOGUES, DEV_ROWS, EPOCHS, BATCH = 12, 2017, 2363, 62329, 5, 32
SOURCES = {"scripts/report_dialogue_warm_pooling.py", "tests/test_report_dialogue_warm_pooling.py",
           "src/openjev/research/dialogue_observation_metrics.py", "tests/test_dialogue_observation_metrics.py"}
RUNNER = "scripts/study_dialogue_warm_pooling.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK = "src/openjev/research/suspend_clock.py"
BASE = "output/dialogue-warm-pooling-v1"
LIMITS = {"wall_seconds": 600, "rss_bytes": 4*1024**3, "output_bytes": 128*1024**2}
SCOPE = ("Exposed full-cohort development comparison, all three fixed seeds. Raw probabilities only, no temperature or calibration. "
         "The trained encoders are frozen during matched head continuation with fresh optimizers. "
         "A pass permits a subsequent prospective study, not novelty, transfer, significance or ICLR readiness. "
         "The state-token arm is one added key/value pooling control, not a general prior-state encoder comparison.")

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
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def descriptor(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def members(directory, receipt="completed.json"):
    require(Path(directory).is_dir() and not Path(directory).is_symlink(), "Regular artifact directory")
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        require(not path.is_symlink(), "No artifact symlinks")
        if path.is_file() and path != Path(directory) / receipt:
            result[path.relative_to(directory).as_posix()] = descriptor(path)
    return result


def manifest(directory, expected, receipt="completed.json"):
    require(type(expected) is dict and expected, "Nonempty artifact manifest")
    for name, value in expected.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and relative.as_posix() == name
                and all(part not in ("", ".", "..") for part in relative.parts)
                and type(value) is dict and set(value) == {"sha256", "bytes"}
                and pin(value["sha256"]) and type(value["bytes"]) is int and value["bytes"] >= 0,
                "Canonical artifact member descriptor")
    require(members(directory, receipt) == expected, "Exact complete artifact bytes and membership")


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy")}


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def parent_terminal(terminal, launch, done, study, run, plan_path, plan_pin):
    require(terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["group_absent"] is True and terminal["timed_out"] is False
            and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["timing_available"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["reaped"] is True and not terminal["cleanup"]["errors"], "Actual successful supervisor and cleanup required")
    for key in ("version", "command", "cwd", "parent_pid", "pid", "pgid", "cap_seconds", "clock_backend",
                "started_ns", "deadline_ns", "watchdog_sha256", "clock_source_sha256"):
        require(terminal[key] == launch[key], "Supervisor launch/terminal identity: "+key)
    require(launch["version"] == "dialogue-observation-supervision-v2"
            and Path(launch["cwd"]).resolve() == ROOT.resolve() and launch["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}
            and launch["cap_seconds"] == study["limits"]["run"]["wall_seconds"] == 7200
            and all(integer(launch[k], 1) for k in ("pid", "pgid", "parent_pid")) and launch["pid"] == launch["pgid"]
            and launch["watchdog_sha256"] == study["source_sha256"][SUPERVISOR]
            and launch["clock_source_sha256"] == study["source_sha256"][CLOCK], "Bound native supervision")
    request = done["request"]
    require(set(request) == {"command", "plan", "plan_sha256", "supervision", "out"}
            and request["command"] == "run" and request["plan_sha256"] == plan_pin
            and Path(request["plan"]).resolve() == plan_path.resolve() and Path(request["out"]).resolve() == run.resolve(),
            "Complete worker request binding")
    command = launch["command"]
    require(type(command) is list and len(command) >= 3 and Path(command[0]).resolve() == Path(sys.executable).resolve(), "Worker interpreter identity")
    tail = command[1:]
    if tail[0] == "-u":
        tail = tail[1:]
    require(len(tail) == 10 and Path(tail[0]).resolve() == (ROOT/RUNNER).resolve() and tail[1] == "run", "Exact worker command form")
    pairs = list(zip(tail[2::2], tail[3::2], strict=True))
    flags = dict(pairs)
    require(len(flags) == len(pairs) == 4 and set(flags) == {"--plan", "--plan-sha256", "--supervision", "--out"}
            and flags["--plan-sha256"] == plan_pin
            and all(Path(flags["--"+k]).resolve() == Path(request[k]).resolve() for k in ("plan", "supervision", "out")), "Worker command/request identity")
    require(all(integer(v) for v in (launch["started_ns"], launch["deadline_ns"], done["started_ns"], done["worker_started_ns"],
                                     done["finished_ns"], terminal["finished_ns"], done["elapsed_ns"], terminal["elapsed_ns"]))
            and done["clock_backend"] == launch["clock_backend"] and done["timing_available"] is True
            and done["started_ns"] == launch["started_ns"] <= done["worker_started_ns"] <= done["finished_ns"] <= terminal["finished_ns"]
            < done["deadline_ns"] == launch["deadline_ns"]
            and launch["deadline_ns"]-launch["started_ns"] == 7200_000_000_000
            and done["elapsed_ns"] == done["finished_ns"]-done["started_ns"]
            and terminal["elapsed_ns"] == terminal["finished_ns"]-launch["started_ns"]
            and done["wall_seconds"] == done["elapsed_ns"]/1e9 and terminal["wall_seconds"] == terminal["elapsed_ns"]/1e9,
            "Consistent on-time native parent and worker intervals")


class Budget:
    def __init__(self, out):
        self.out, self.clock = out, SuspendClock()
        require(self.clock.backend in {"mach_continuous_time", "CLOCK_BOOTTIME"}, "Native report clock")
        self.deadline = self.clock.deadline_after(LIMITS["wall_seconds"])

    def check(self):
        self.deadline.check()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        require(rss <= LIMITS["rss_bytes"], "Report RSS cap")

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Report output cap")

    def elapsed(self):
        self.check()
        return (self.clock.now_ns()-self.deadline.started_ns)/1e9



def metric_api():
    return importlib.import_module("openjev.research.dialogue_observation_metrics")


def score(logs, indices, rows):
    """Unchanged original raw scoring plus stronger public endpoint validation."""
    sources, local_queries, positions, turns = {}, {}, {}, {}
    for row in rows:
        did, query = row["dialogue_id"], row["query_index"]
        require(row["source_row_index"] == sources.get(did, 0), "Complete source endpoint order per dialogue")
        sources[did] = sources.get(did, 0)+1
        require(json.loads(row["query_id"]) == [row["service"], row["slot"]]
                and integer(row["turn_index"]) and integer(row["query_position"]), "Canonical public query and chronology")
        require(local_queries.setdefault((did, row["query_position"]), query) == query
                and positions.setdefault((did, query), row["query_position"]) == row["query_position"]
                and turns.setdefault((did, row["time"]), row["turn_index"]) == row["turn_index"], "Stable public local query and turn bindings")
        stratum = row["bin"] if row["bin"] in STRATA[:2] else "changed"
        require(row["stratum"] == stratum and integer(row["stratum_index"])
                and row["stratum_index"] == STRATA.index(stratum), "Canonical transition stratum")
    scored = metric_api().score_panels(logs, rows, row_indices=indices)
    return {**scored, "recovery": recovery(logs, rows)}



def recovery(logs, rows):
    """Descriptive next scored endpoint; fit-dependent subsets are not a gate."""
    correct = logs.argmax(1) == np.asarray([r["label_index"] for r in rows])
    previous, after_error, after_correct, gaps = {}, np.zeros(len(rows), bool), np.zeros(len(rows), bool), []
    for index in sorted(range(len(rows)), key=lambda i: (rows[i]["dialogue_id"], rows[i]["query_index"], rows[i]["time"])):
        row = rows[index]
        key = row["dialogue_id"], row["query_index"]
        if key in previous:
            old = previous[key]
            gap = row["time"]-rows[old]["time"]
            require(gap > 0 and row["turn_index"] > rows[old]["turn_index"], "Strict within-query public chronology")
            if correct[old]:
                after_correct[index] = True
            else:
                after_error[index] = True
                gaps.append(gap)
        previous[key] = index

    def cell(selected):
        count, hits = int(selected.sum()), int((selected & correct).sum())
        return {"count": count, "correct": hits, "incorrect": count-hits,
                "accuracy": hits/count if count else None, "error": (count-hits)/count if count else None}

    unseen = np.asarray([r["unseen"] for r in rows], dtype=bool)
    panels = {}
    for name in ("all", "seen", "unseen"):
        selected = np.ones(len(rows), bool) if name == "all" else unseen if name == "unseen" else ~unseen
        panels[name] = {"after_previous_scored_error": cell(selected & after_error),
                        "after_previous_scored_correct": cell(selected & after_correct)}
    return {"panels": panels, "eligible_pairs": int((after_error | after_correct).sum()),
            "first_scored_endpoints": len(previous), "after_error_nonadjacent_public_turns": sum(g > 1 for g in gaps),
            "after_error_mean_public_turn_gap": math.fsum(gaps)/len(gaps) if gaps else None,
            "after_error_max_public_turn_gap": max(gaps) if gaps else None,
            "scope": "Next scored endpoint of the same dialogue/query after this model's own previous prediction. "
                     "Spacing may include unscored public turns. Each model's error subset differs; descriptive, not causal recovery or a continuation condition."}

def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def accuracy(cell):
    require(integer(cell["count"], 1) and integer(cell["correct"])
            and cell["correct"] <= cell["count"], "Positive integer metric support")
    return Fraction(cell["correct"], cell["count"])


def macro(fit, panel):
    return sum((accuracy(fit["panels"][panel]["strata"][s]) for s in STRATA), Fraction())/3


def criteria(fits):
    """Thirty-two fixed conditions; no tolerance, rounding, or missing-fit pass."""
    require(type(fits) is dict and set(fits) == set(SCORE_ORDER), "All twelve adapted and three untouched fits required")
    support = None
    for fit in fits.values():
        counts = []
        for panel in ("all", "seen", "unseen"):
            groups = fit["panels"][panel]
            macro(fit, panel)
            count = groups["micro"]["count"]
            require(integer(count, 1) and sum(groups["strata"][s]["count"] for s in STRATA) == count,
                    "Positive exhaustive panel strata")
            require(all(type(groups["micro"][k]) in (int, float) and math.isfinite(groups["micro"][k])
                        for k in ("nll", "brier")), "Finite raw proper scores")
            counts.extend([count, *(groups["strata"][s]["count"] for s in STRATA)])
        require(fit["panels"]["all"]["micro"]["count"] == fit["panels"]["seen"]["micro"]["count"]
                + fit["panels"]["unseen"]["micro"]["count"], "Seen/unseen exhaustive partition")
        require(all(fit["panels"]["all"]["strata"][s][k] == sum(fit["panels"][p]["strata"][s][k]
                    for p in ("seen", "unseen")) for s in STRATA for k in ("count", "correct")), "Stratum partition reconstruction")
        require(support is None or support == counts, "Identical full cohort support for every fit")
        support = counts
    checks = []
    for control in COMPARATORS:
        new = [fits[f"belief_query-{s}"] for s in SEEDS]
        old = [fits[f"{control}-{s}"] for s in SEEDS]

        def exact(name, values, threshold, comparison, control=control):
            mean = sum(values, Fraction())/3
            passed = mean >= threshold if comparison == "ge" else mean <= threshold
            checks.append({"comparator": control, "name": name, "passed": bool(passed), "comparison": comparison,
                           "threshold": float(threshold), "threshold_exact": rational(threshold),
                           "paired_seed_changes": {str(s): float(v) for s, v in zip(SEEDS, values, strict=True)},
                           "paired_seed_changes_exact": {str(s): rational(v) for s, v in zip(SEEDS, values, strict=True)},
                           "mean_paired_change": float(mean), "mean_paired_change_exact": rational(mean),
                           "arithmetic": "Exact integer-count fractions"})

        gains = [macro(a, "unseen")-macro(b, "unseen") for a, b in zip(new, old, strict=True)]
        exact("unseen_macro_gain_at_least_1pp", gains, Fraction(1, 100), "ge")
        checks.append({"comparator": control, "name": "unseen_macro_positive_at_least_two_seeds",
                       "passed": sum(v > 0 for v in gains) >= 2, "positive_seeds": sum(v > 0 for v in gains),
                       "required_positive_seeds": 2,
                       "paired_seed_changes": {str(s): float(v) for s, v in zip(SEEDS, gains, strict=True)},
                       "paired_seed_changes_exact": {str(s): rational(v) for s, v in zip(SEEDS, gains, strict=True)},
                       "arithmetic": "Exact strict-positive integer-count fractions"})
        exact("unseen_changed_gain_at_least_1pp", [accuracy(a["panels"]["unseen"]["strata"]["changed"])
              - accuracy(b["panels"]["unseen"]["strata"]["changed"]) for a, b in zip(new, old, strict=True)], Fraction(1, 100), "ge")
        for metric in ("nll", "brier"):
            after = [a["panels"]["unseen"]["micro"][metric] for a in new]
            before = [b["panels"]["unseen"]["micro"][metric] for b in old]
            after_mean, before_mean = math.fsum(after)/3, math.fsum(before)/3
            changes = [a-b for a, b in zip(after, before, strict=True)]
            checks.append({"comparator": control, "name": "unseen_"+metric+"_nonworse", "passed": after_mean <= before_mean,
                           "belief_query_mean": after_mean, "comparator_mean": before_mean,
                           "difference_of_arm_means": after_mean-before_mean,
                           "paired_seed_changes": dict(zip(map(str, SEEDS), changes, strict=True)),
                           "mean_paired_change": math.fsum(changes)/3,
                           "arithmetic": "Float64 math.fsum equal-seed arm means; direct <= with no epsilon"})
        exact("seen_macro_decline_at_most_1pp", [macro(a, "seen")-macro(b, "seen") for a, b in zip(new, old, strict=True)],
              Fraction(-1, 100), "ge")
        for panel in ("seen", "unseen"):
            exact(panel+"_assigned_retention_error_increase_at_most_half_pp",
                  [accuracy(b["panels"][panel]["strata"]["assigned_retention"])
                   - accuracy(a["panels"][panel]["strata"]["assigned_retention"]) for a, b in zip(new, old, strict=True)],
                  Fraction(1, 200), "le")
    require(len(checks) == 32, "Exactly thirty-two prospective conditions")
    return {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": 32, "checks": checks,
            "scope": "All eight conditions must pass against each of four fixed comparators; all three seeds retained. "
                     "Raw untransformed outputs. Descriptive full-cohort metrics do not replace unseen primary comparisons."}


def expected_members():
    result = {"started.json", "plan.json", "rows-train.jsonl", "rows-dev.jsonl", "qualification.json"}
    result |= {f"cache-{seed}/"+name for seed in SEEDS for name in ("tokens.npy", "pooled.npy", "offsets.npy", "index.json", "completed.json")}
    result |= {"qualification/"+fit+"/"+name for fit in QUALIFICATION_ORDER for name in ("predictions.npz", "completed.json")}
    result |= {fit+"/"+name for fit in FIT_ORDER for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    return result


def upstream_authenticate(path, digest, run):
    runner = importlib.import_module("study_dialogue_warm_pooling")
    return runner.authenticate(SimpleNamespace(command="run", plan=path, plan_sha256=digest, out=run))


def authenticate(args, budget):
    """No labels, prediction decoding or model loading before complete acceptance."""
    require(args.plan.resolve() == (ROOT/BASE/"report-plan-01.json").resolve()
            and args.out.resolve() == (ROOT/BASE/"report-01").resolve(), "Fixed separate report paths")
    require(pin(args.plan_sha256) and sha(args.plan) == args.plan_sha256, "External report plan pin")
    plan = read(args.plan)
    require(set(plan) == {"version", "limits", "sources", "runtime", "synthetic_qualification", "study_plan", "run", "out"}
            and plan["version"] == VERSION and plan["limits"] == LIMITS and plan["runtime"] == runtime()
            and (ROOT/plan["out"]).resolve() == args.out.resolve() and set(plan["sources"]) == SOURCES,
            "Exact prospective reporting contract")
    for name, digest in plan["sources"].items():
        require(pin(digest) and sha(ROOT/name) == digest, "Report source pin")
    qual = plan["synthetic_qualification"]
    require(set(qual) == {"path", "sha256"} and pin(qual["sha256"]) and sha(ROOT/qual["path"]) == qual["sha256"], "Synthetic qualification pin")
    qualified = read(ROOT/qual["path"])
    require(qualified["status"] == "completed" and qualified["actual_exit_code"] == 0
            and qualified["source_sha256"] == plan["sources"] and qualified["model_calls"] == 0
            and qualified["real_task_inputs_read"] is False, "Same-source synthetic qualification")
    study_binding, binding = plan["study_plan"], plan["run"]
    require(set(study_binding) == {"path", "sha256"} and pin(study_binding["sha256"])
            and sha(ROOT/study_binding["path"]) == study_binding["sha256"], "Prospective study pin before helper execution")
    study_path = ROOT/study_binding["path"]
    study = read(study_path)
    for name, digest in study["source_sha256"].items():
        require(pin(digest) and sha(ROOT/name) == digest, "Study source closure before read-only authentication")
        budget.check()
    require({RUNNER, SUPERVISOR, CLOCK}.issubset(study["source_sha256"]), "Complete directly used helper pins")
    require(set(binding) == {"path", "completed_sha256", "terminal"}
            and set(binding["terminal"]) == {"path", "sha256"}, "Complete run and external parent bindings")
    run = ROOT/binding["path"]
    authenticated, _ = upstream_authenticate(study_path, study_binding["sha256"], run)
    require(authenticated == study and study["version"] == STUDY_VERSION and study["fit_order"] == FIT_ORDER
            and all(len(study["selected"][s]) == len(set(study["selected"][s])) == count
                    for s, count in (("train", TRAIN_DIALOGUES), ("dev", DEV_DIALOGUES))), "Entire original cohort and all twelve fixed fits")
    require(pin(binding["completed_sha256"]) and sha(run/"completed.json") == binding["completed_sha256"], "External run completion pin")
    done = read(run/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "run" and done["version"] == STUDY_VERSION
            and done["fit_order"] == FIT_ORDER and [f["fit_id"] for f in done["fits"]] == FIT_ORDER
            and done["plan_sha256"] == study_binding["sha256"] and done["source_sha256"] == study["source_sha256"]
            and done["quality_metrics_computed"] is False and done["official_test_opened"] is False
            and done["temperature_applied"] is False and done["row_counts"]["dev"] == DEV_ROWS
            and done["row_counts"]["train"] == sum(study["loss_counts"]["train"].values())
            and done["limits"] == study["limits"]["run"] and done["peak_rss_bytes"] <= done["limits"]["rss_bytes"]
            and done["sampled_mps_driver_max_bytes"] <= done["limits"]["mps_driver_bytes"], "Complete raw study and resource acceptance")
    require(set(done["files"]) == expected_members(), "Exact successful study payload set")
    manifest(run, done["files"])
    require(sum(v["bytes"] for v in done["files"].values())+(run/"completed.json").stat().st_size <= done["limits"]["output_bytes"],
            "Full saved study storage cap")
    require(sha(run/"plan.json") == study_binding["sha256"], "Saved prospective plan unchanged")
    started = read(run/"started.json")
    require(started["request"] == done["request"] and started["limits"] == done["limits"]
            and started["clock_backend"] == done["clock_backend"] and started["worker_started_ns"] == done["worker_started_ns"]
            and started["parent_started_ns"] == done["started_ns"] and started["deadline_ns"] == done["deadline_ns"]
            and started["supervision_sha256"] == done["supervision_sha256"], "Started/completed worker join")
    launch_path = Path(done["request"]["supervision"])
    require(sha(launch_path) == done["supervision_sha256"], "Successful worker launch identity")
    terminal_path = ROOT/binding["terminal"]["path"]
    require(pin(binding["terminal"]["sha256"]) and sha(terminal_path) == binding["terminal"]["sha256"], "External supervisor terminal pin")
    terminal = read(terminal_path)
    parent_terminal(terminal, read(launch_path), done, study, run, study_path, study_binding["sha256"])
    require([c["seed"] for c in done["caches"]] == list(SEEDS), "All three trained encoder caches")
    for cache in done["caches"]:
        seed = cache["seed"]
        directory = run/f"cache-{seed}"
        require(read(directory/"completed.json") == cache and cache["status"] == "completed"
                and cache["encoder_sha256"] == study["checkpoints"]["fits"][str(seed)]["encoder_sha256"], "Exact frozen trained encoder cache")
        require(set(cache["files"]) == {"tokens.npy", "pooled.npy", "offsets.npy", "index.json"}, "Complete raw-token and pooled cache")
        manifest(directory, cache["files"])
        budget.check()
    qualification = read(run/"qualification.json")
    require(qualification == {"status": "completed", "paths": done["qualification_paths"], "training_started": False,
                              "optimizer_created": False, "temperature_applied": False}
            and [q["fit_id"] for q in done["qualification_paths"]] == QUALIFICATION_ORDER,
            "All fifteen qualifications completed before any optimizer")
    initials = {}
    for record in done["qualification_paths"]:
        name = record["fit_id"]
        arm, seed = name.rsplit("-", 1)
        directory = run/"qualification"/name
        require(read(directory/"completed.json") == record and record["status"] == "completed"
                and record["rows"] == DEV_ROWS and record["counts"]["evaluation_forwards"] == DEV_DIALOGUES
                and set(record["files"]) == {"predictions.npz"} and pin(record["initial_state_sha256"]), "Entire qualification path and raw predictions")
        manifest(directory, record["files"])
        parity = record["parity"]
        require(parity["passed"] is True and parity["endpoints"] == DEV_ROWS
                and type(parity["selected_choice_changes"]) is int and parity["selected_choice_changes"] == 0
                and type(parity["top_tie_mask_changes"]) is int and parity["top_tie_mask_changes"] == 0
                and 0 <= parity["maximum_log_difference"] <= 1e-5
                and 0 <= parity["maximum_probability_difference"] <= 1e-6, "Full original strict parity including all top ties")
        if arm != "untouched":
            require(initials.setdefault(seed, record["initial_state_sha256"]) == record["initial_state_sha256"], "Paired full zero-residual initialization")
        budget.check()
    for record in done["fits"]:
        name = record["fit_id"]
        arm, seed = name.rsplit("-", 1)
        directory = run/name
        require(sha(directory/"completed.json") == record["completed_sha256"], "Nested complete fit pin")
        fit = read(directory/"completed.json")
        require(fit == {k: v for k, v in record.items() if k != "completed_sha256"}
                and fit["status"] == "completed" and fit["method"] == arm and fit["seed"] == int(seed)
                and fit["initial_state_sha256"] == initials[seed] and pin(fit["final_state_sha256"])
                and set(fit["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}, "Exact adapted fit and qualified initialization")
        manifest(directory, fit["files"])
        counts, evaluation = fit["training_counts"], fit["evaluation"]
        require(counts["optimizer_attempts"] == counts["optimizer_updates"] == EPOCHS*math.ceil(TRAIN_DIALOGUES/BATCH)
                and counts["forward_attempts"] == counts["forward_returns"] == counts["training_forwards"]
                == counts["backward_attempts"] == counts["backward_returns"] == EPOCHS*TRAIN_DIALOGUES
                and counts["training_endpoints"] == EPOCHS*done["row_counts"]["train"]
                and evaluation["rows"] == DEV_ROWS and evaluation["counts"]["evaluation_forwards"] == DEV_DIALOGUES
                and evaluation["counts"]["forward_attempts"] == evaluation["counts"]["forward_returns"] == DEV_DIALOGUES,
                "All fixed weighted updates and complete autonomous evaluation")
        budget.check()
    return plan, study, done, terminal, run


def read_rows(path, budget):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "Nonempty evaluator row")
            result.append(json.loads(line))
            budget.check()
    return result


def compute(plan, study, done, terminal, run, budget):
    rows = read_rows(run/"rows-dev.jsonl", budget)
    original = read_rows(Path(study["checkpoints"]["run"])/"evaluation-rows.jsonl", budget)
    require(rows == original and len(rows) == DEV_ROWS
            and list(dict.fromkeys(row["dialogue_id"] for row in rows)) == study["selected"]["dev"],
            "Entire unchanged original DEV endpoint ledger and cohort order")
    fits = {}
    for name in SCORE_ORDER:
        directory = run/"qualification"/name if name.startswith("untouched-") else run/name
        with np.load(directory/"predictions.npz", allow_pickle=False) as packet:
            require(set(packet.files) == {"log_probs", "row_indices"}, "Exact saved prediction packet")
            fits[name] = score(packet["log_probs"], packet["row_indices"], rows)
        budget.check()
    continuation = criteria(fits)
    api = metric_api()
    trees = {name: api._metric_tree(value) for name, value in fits.items()}
    families = {arm: {"fit_ids": [f"{arm}-{seed}" for seed in SEEDS],
                     "mean": api._combine([trees[f"{arm}-{seed}"] for seed in SEEDS], api._mean)} for arm in (*ARMS, "untouched")}
    contrasts = {}
    for comparator in COMPARATORS:
        seeds = {str(seed): api._combine([trees[f"belief_query-{seed}"], trees[f"{comparator}-{seed}"]],
                                         lambda values: values[0]-values[1]) for seed in SEEDS}
        contrasts[comparator] = {"definition": "belief_query minus "+comparator, "paired_seeds": seeds,
                                 "mean_paired_change": api._combine(list(seeds.values()), api._mean)}
    return {"version": VERSION, "status": "completed", "technical_validity_passed": True,
            "complete_adapted_fits": len(FIT_ORDER), "complete_untouched_references": len(SEEDS), "scored_fits": len(fits),
            "fit_order": SCORE_ORDER, "fits": fits, "families": families, "contrasts": contrasts, "continuation": continuation,
            "model_calls": 0, "temperature_fitted": False, "probabilities_transformed": False, "official_test_opened": False,
            "bindings": {"study_plan": plan["study_plan"], "run": plan["run"]},
            "costs": {"actual_parent_wall_seconds": terminal["wall_seconds"], "worker_wall_seconds": done["wall_seconds"],
                      "caches": done["caches"], "qualification_paths": done["qualification_paths"],
                      "fits": {f["fit_id"]: {k: f[k] for k in ("training_counts", "training_work", "training_seconds", "evaluation",
                              "wall_seconds", "configuration", "registered_parameters", "gradient_present_parameters", "nonzero_gradient_parameters")}
                               for f in done["fits"]},
                      "scope": "Parent total includes loading, all caches, fifteen qualification paths, twelve adaptations and cleanup. "
                               "Nested components are not added. Reporting, synthetic qualification and development are separate; not a serving benchmark."},
            "scope": SCOPE}


def report_text(summary):
    gate = summary["continuation"]
    lines = ["# Warm-start belief pooling: complete development study", "",
             (f"All 12 adapted fits and 3 untouched references completed. Continuation: {'PASS' if gate['passed'] else 'FAIL'}, "
              f"{gate['checks_passed']}/32 conditions passed."), "", SCOPE, "",
             "| Fit | Unseen macro accuracy | Unseen changed accuracy | Unseen NLL | Unseen Brier | Seen macro accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in SCORE_ORDER:
        fit = summary["fits"][name]
        unseen, seen = fit["panels"]["unseen"], fit["panels"]["seen"]
        lines.append(f"| {name} | {100*unseen['macro_three']['accuracy']:.4f}% | "
                     f"{100*unseen['strata']['changed']['accuracy']:.4f}% | {unseen['micro']['nll']:.8g} | "
                     f"{unseen['micro']['brier']:.8g} | {100*seen['macro_three']['accuracy']:.4f}% |")
    lines += ["", ("All three seeds remain visible. Macro accuracy equally weights the three strata; proper scores average endpoints. "
               "The gate uses exact integer-count fractions for accuracy and retention. Display rounding never changes its decisions."),
              "", "| Comparator | Condition | Result |", "|---|---|---|"]
    lines.extend(f"| {c['comparator']} | {c['name']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in gate["checks"])
    lines += ["", (f"Actual supervised parent interval: {summary['costs']['actual_parent_wall_seconds']:.3f} seconds, including all loading, "
               "caching, full-DEV qualification and adaptation."), "",
              ("[summary.json](summary.json) retains every raw fit, all/seen/unseen panels, transition bins, candidate types, services, "
              "equal-seed means and every paired seed contrast. Recovery follows the same query after each model's previous scored error; "
              "unscored turns may intervene and the model-dependent subsets are descriptive only. Empty descriptive groups are null. No transformed-probability "
               "result, early checkpoint or favorable seed replaces the fixed raw comparison."), ""]
    return "\n".join(lines)


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    budget = handler = None
    try:
        budget = Budget(args.out)

        def expired(*_):
            raise TimeoutError("Supplementary report deadline expired")

        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        plan, study, done, terminal, run = authenticate(args, budget)
        summary = compute(plan, study, done, terminal, run, budget)
        write(args.out/"summary.json", summary)
        with (args.out/"report.md").open("x") as stream:
            stream.write(report_text(summary))
        authenticate(args, budget)
        budget.storage()
        receipt = {"version": VERSION, "status": "completed", "plan_sha256": args.plan_sha256,
                   "source_sha256": plan["sources"], "study_plan": plan["study_plan"], "run": plan["run"],
                   "files": members(args.out, "receipt.json"), "complete_adapted_fits": 12,
                   "complete_untouched_references": 3, "scored_fits": 15, "model_calls": 0,
                   "temperature_fitted": False, "official_test_opened": False, "continuation": summary["continuation"],
                   "wall_seconds": budget.elapsed(), "clock_backend": budget.clock.backend, "scope": SCOPE}
        require(set(receipt["files"]) == {"summary.json", "report.md"}, "Exact report payload closure")
        write(args.out/"receipt.json", receipt)
        budget.storage()
        return receipt
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out/"receipt.json").exists():
                (args.out/"receipt.json").rename(args.out/"invalid-receipt.json")
            write(args.out/"failed.json", {"version": VERSION, "status": "failed", "error_type": type(error).__name__,
                                          "error": str(error), "model_calls": 0, "scientific_result_qualified": False,
                                          "scope": "Original failure and partial outputs retained; no automatic retry"})
        except BaseException as secondary:  # noqa: BLE001 - preserve the primary failure
            error.add_note("Failure receipt error: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    result = execute(arguments)
    print(json.dumps({"status": result["status"], "checks_passed": result["continuation"]["checks_passed"],
                      "checks_total": 32, "receipt_sha256": sha(arguments.out/"receipt.json")}))
