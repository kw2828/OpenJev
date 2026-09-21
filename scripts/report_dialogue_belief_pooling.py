"""Independent saved-output scoring of the complete, fixed pooling pilot.

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
VERSION = "dialogue-belief-pooling-report-v1"
PILOT_VERSION = "dialogue-belief-pooling-pilot-v1"
ARMS = ("pooled", "schema_attention", "belief_query", "state_token")
COMPARATORS = ("pooled", "schema_attention", "state_token")
SEEDS = (7101, 7102)
FIT_ORDER = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
BINS = (*STRATA[:2], "first_assignment", "revision", "clear")
NONE, DONTCARE = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
MAX_CANDIDATES, DIALOGUES, EPOCHS, BATCH = 12, 128, 3, 8
TOLERANCE = 2e-6
SOURCES = {"scripts/report_dialogue_belief_pooling.py", "tests/test_report_dialogue_belief_pooling.py"}
RUNNER = "scripts/pilot_dialogue_belief_pooling.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK = "src/openjev/research/suspend_clock.py"
BASE = "output/dialogue-belief-pooling-pilot-v1"
LIMITS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
SCOPE = ("Exposed development pilot: all four arms and both fixed seeds. Raw outputs only; no temperature or calibration. "
         "A pass admits a larger matched development experiment, not novelty, significance, transfer or ICLR readiness. "
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


def layout(rows):
    """Validate independent canonical DEV endpoint identity and candidate order."""
    require(type(rows) is list and rows, "Nonempty saved DEV endpoint ledger")
    mask = np.zeros((len(rows), MAX_CANDIDATES), bool)
    labels, strata, bins, unseen = [], [], [], []
    endpoints, sources, schemas, services, local_queries, local_positions, turns = set(), set(), {}, {}, {}, {}, {}
    source_count = {}
    for index, row in enumerate(rows):
        require(type(row) is dict and integer(row["row_index"]) and row["row_index"] == index
                and row["split"] == "dev", "Exact contiguous DEV endpoint indices")
        did, service, slot, query_id = (row[k] for k in ("dialogue_id", "service", "slot", "query_id"))
        require(all(type(v) is str and v for v in (did, service, slot, query_id))
                and json.loads(query_id) == [service, slot], "Public query identity")
        require(all(integer(row[k]) for k in ("source_row_index", "time", "turn_index", "query_index", "query_position")),
                "Nonnegative source and public chronology indices")
        require(row["source_row_index"] == source_count.get(did, 0), "Complete source endpoint order per dialogue")
        source_count[did] = source_count.get(did, 0) + 1
        endpoint, source = (did, row["time"], row["query_index"]), (did, row["source_row_index"])
        require(endpoint not in endpoints and source not in sources, "Distinct canonical endpoints")
        endpoints.add(endpoint)
        sources.add(source)
        ids, values = row["candidate_ids"], row["candidate_values"]
        require(type(ids) is list and type(values) is list and 3 <= len(ids) <= MAX_CANDIDATES
                and len(values) == len(ids) and all(type(v) is str for v in ids)
                and len(set(ids)) == len(ids) and ids.count(NONE) == ids.count(DONTCARE) == 1,
                "Full unique candidate support including reserved candidates")
        require(all(value is None if cid in (NONE, DONTCARE) else
                    type(value) is str and bool(value.strip()) and cid == "value:" + value
                    for cid, value in zip(ids, values, strict=True)), "Canonical candidate ID/value binding")
        schema = (query_id, tuple(ids), tuple(values))
        require(schemas.setdefault(row["query_index"], schema) == schema
                and local_queries.setdefault((did, row["query_position"]), row["query_index"]) == row["query_index"]
                and local_positions.setdefault((did, row["query_index"]), row["query_position"]) == row["query_position"]
                and turns.setdefault((did, row["time"]), row["turn_index"]) == row["turn_index"],
                "Consistent global schema, local query position and public time")
        y = row["label_index"]
        require(integer(y) and y < len(ids) and row["label_id"] == ids[y], "Canonical target ID and index")
        bin_name = row["bin"]
        require(bin_name in BINS, "Known transition subtype")
        stratum = bin_name if bin_name in STRATA[:2] else "changed"
        require(row["stratum"] == stratum and integer(row["stratum_index"])
                and row["stratum_index"] == STRATA.index(stratum) and type(row["unseen"]) is bool
                and type(row["dontcare"]) is bool and row["dontcare"] == (ids[y] == DONTCARE)
                and services.setdefault(service, row["unseen"]) == row["unseen"], "Transition and service exposure binding")
        mask[index, :len(ids)] = True
        labels.append(y)
        strata.append(stratum)
        bins.append(bin_name)
        unseen.append(row["unseen"])
    return {"mask": mask, "labels": np.asarray(labels, dtype=np.int64), "strata": np.asarray(strata),
            "bins": np.asarray(bins), "unseen": np.asarray(unseen, dtype=bool)}


def score(logs, indices, rows):
    data = layout(rows)
    n, mask = len(rows), data["mask"]
    require(isinstance(indices, np.ndarray) and indices.dtype == np.int64
            and np.array_equal(indices, np.arange(n, dtype=np.int64)), "Exact prediction row indices")
    require(isinstance(logs, np.ndarray) and logs.dtype == np.float32 and logs.shape == mask.shape,
            "Complete saved float32 raw log probabilities")
    require(np.isfinite(logs[mask]).all() and np.isneginf(logs[~mask]).all(), "Finite support and exact negative-infinity padding")
    require(float(logs[mask].max()) <= math.log1p(TOLERANCE), "Supported probability upper bound")
    probabilities = np.exp(logs.astype(np.float64))
    errors = np.abs(probabilities.sum(1) - 1.)
    require(bool(np.all(errors <= TOLERANCE)), "Saved probability mass without repair")
    choices = logs.argmax(1)
    correct = choices == data["labels"]
    nll = -logs[np.arange(n), data["labels"]].astype(np.float64)
    residual = probabilities.copy()
    residual[np.arange(n), data["labels"]] -= 1.
    brier = np.square(residual).sum(1)
    require(np.isfinite(nll).all() and np.isfinite(brier).all(), "Finite complete proper scores")

    def cell(selected):
        count, hits = int(selected.sum()), int((correct & selected).sum())
        return {"count": count, "correct": hits, "incorrect": count-hits,
                "accuracy": hits/count if count else None, "error": (count-hits)/count if count else None,
                "nll": math.fsum(map(float, nll[selected]))/count if count else None,
                "brier": math.fsum(map(float, brier[selected]))/count if count else None}

    panels = {}
    for panel in ("all", "seen", "unseen"):
        selected = np.ones(n, bool) if panel == "all" else (data["unseen"] if panel == "unseen" else ~data["unseen"])
        groups = {name: cell(selected & (data["strata"] == name)) for name in STRATA}
        macro = sum((Fraction(g["correct"], g["count"]) for g in groups.values()), Fraction())/3 if all(g["count"] for g in groups.values()) else None
        panels[panel] = {"micro": cell(selected), "strata": groups,
                         "bins": {name: cell(selected & (data["bins"] == name)) for name in BINS},
                         "macro_accuracy": None if macro is None else float(macro), "macro_accuracy_exact": rational(macro)}
    require(all(panels["all"]["strata"][name]["count"] > 0 for name in STRATA), "All three primary strata require support")
    previous, after_wrong, after_correct, gaps = {}, np.zeros(n, bool), np.zeros(n, bool), []
    wrong_gaps = []
    for index in sorted(range(n), key=lambda i: (rows[i]["dialogue_id"], rows[i]["query_index"], rows[i]["time"])):
        row = rows[index]
        key = (row["dialogue_id"], row["query_index"])
        if key in previous:
            old_index = previous[key]
            gap = row["time"] - rows[old_index]["time"]
            require(gap > 0 and row["turn_index"] > rows[old_index]["turn_index"], "Strict within-query public chronology")
            gaps.append(gap)
            if correct[old_index]:
                after_correct[index] = True
            else:
                after_wrong[index] = True
                wrong_gaps.append(gap)
        previous[key] = index
    recovery = {"after_previous_scored_error": cell(after_wrong), "after_previous_scored_correct": cell(after_correct),
                "eligible_pairs": len(gaps), "first_scored_endpoints": len(previous),
                "after_error_nonadjacent_public_turns": sum(gap > 1 for gap in wrong_gaps),
                "after_error_mean_public_turn_gap": math.fsum(wrong_gaps)/len(wrong_gaps) if wrong_gaps else None,
                "after_error_max_public_turn_gap": max(wrong_gaps) if wrong_gaps else None,
                "scope": "Next scored endpoint for the same dialogue/query after this fit's previous scored prediction. "
                         "Spacing may include unscored public turns. Subsets depend on each model and are descriptive, not matched causal recovery tests or continuation conditions."}
    return {"panels": panels, "recovery": recovery,
            "validation": {"rows": n, "dialogues": len({r["dialogue_id"] for r in rows}),
                           "supported_positions": int(mask.sum()), "maximum_mass_error": float(errors.max()),
                           "top_tie_rows": int(((logs == logs.max(1, keepdims=True)).sum(1) > 1).sum()),
                           "float64_underflow_positions": int((probabilities[mask] == 0).sum()),
                           "policy": "Direct raw-log NLL, float64 candidate-sum Brier, canonical first argmax; no floor, clipping, normalization or temperature"}}


def rational(value):
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def macro(fit):
    groups = fit["panels"]["all"]["strata"]
    require(all(integer(groups[s]["count"], 1) and integer(groups[s]["correct"])
                and groups[s]["correct"] <= groups[s]["count"] for s in STRATA), "Complete integer primary supports")
    return sum((Fraction(groups[s]["correct"], groups[s]["count"]) for s in STRATA), Fraction()) / 3


def criteria(fits):
    require(set(fits) == set(FIT_ORDER), "Every one of eight fixed fits required")
    supports = None
    for fit in fits.values():
        macro(fit)
        current = tuple(fit["panels"]["all"]["strata"][s]["count"] for s in STRATA)
        require(supports is None or supports == current, "All arms and seeds share the same primary support")
        supports = current
        require(fit["panels"]["all"]["micro"]["count"] == sum(current), "Exhaustive primary strata")
        require(all(type(fit["panels"]["all"]["micro"][k]) in (int, float)
                    and math.isfinite(fit["panels"]["all"]["micro"][k]) for k in ("nll", "brier")), "Finite primary proper scores")
    checks = []
    for control in COMPARATORS:
        new = [fits[f"belief_query-{s}"] for s in SEEDS]
        old = [fits[f"{control}-{s}"] for s in SEEDS]
        differences = [macro(a)-macro(b) for a, b in zip(new, old, strict=True)]
        mean = sum(differences, Fraction())/len(SEEDS)
        checks.append({"comparator": control, "name": "macro_gain_at_least_1pp", "passed": mean >= Fraction(1, 100),
                       "mean_paired_change": float(mean), "mean_paired_change_exact": rational(mean),
                       "paired_changes_exact": [rational(v) for v in differences], "threshold_exact": rational(Fraction(1, 100))})
        checks.append({"comparator": control, "name": "macro_strictly_positive_each_seed", "passed": all(v > 0 for v in differences),
                       "paired_changes": [float(v) for v in differences], "paired_changes_exact": [rational(v) for v in differences]})
        for metric in ("nll", "brier"):
            a = [fit["panels"]["all"]["micro"][metric] for fit in new]
            b = [fit["panels"]["all"]["micro"][metric] for fit in old]
            after, before = math.fsum(a)/len(SEEDS), math.fsum(b)/len(SEEDS)
            checks.append({"comparator": control, "name": metric+"_nonworse", "passed": after <= before,
                           "belief_query_mean": after, "control_mean": before, "change_of_means": after-before,
                           "paired_changes": [x-y for x, y in zip(a, b, strict=True)], "arithmetic": "float64 equal-seed arm means; no epsilon"})
        for stratum in STRATA[:2]:
            differences = []
            for a, b in zip(new, old, strict=True):
                ac, bc = (f["panels"]["all"]["strata"][stratum] for f in (a, b))
                differences.append(Fraction(ac["count"]-ac["correct"], ac["count"])-Fraction(bc["count"]-bc["correct"], bc["count"]))
            mean = sum(differences, Fraction())/len(SEEDS)
            checks.append({"comparator": control, "name": stratum+"_error_nonworse", "passed": mean <= 0,
                           "mean_paired_change": float(mean), "mean_paired_change_exact": rational(mean),
                           "paired_changes_exact": [rational(v) for v in differences]})
    require(len(checks) == 18, "Exactly eighteen prospective conditions")
    return {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": 18, "checks": checks, "population": "all selected DEV endpoints",
            "scope": "Fixed prospective pilot gate; descriptive recovery and seen/unseen subsets do not alter the decision"}


def upstream_authenticate(path, digest, run):
    runner = importlib.import_module("pilot_dialogue_belief_pooling")
    return runner.authenticate(SimpleNamespace(command="run", plan=path, plan_sha256=digest, out=run))


def parent_terminal(terminal, launch, done, pilot, run, plan_path, plan_pin):
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
            and launch["cap_seconds"] == pilot["limits"]["run"]["wall_seconds"] == 1800
            and all(integer(launch[k], 1) for k in ("pid", "pgid", "parent_pid")) and launch["pid"] == launch["pgid"]
            and launch["watchdog_sha256"] == pilot["source_sha256"][SUPERVISOR]
            and launch["clock_source_sha256"] == pilot["source_sha256"][CLOCK], "Bound native supervision")
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
            and launch["deadline_ns"]-launch["started_ns"] == 1800_000_000_000
            and done["elapsed_ns"] == done["finished_ns"]-done["started_ns"]
            and terminal["elapsed_ns"] == terminal["finished_ns"]-launch["started_ns"]
            and done["wall_seconds"] == done["elapsed_ns"]/1e9 and terminal["wall_seconds"] == terminal["elapsed_ns"]/1e9,
            "Consistent on-time native parent and worker intervals")


def authenticate(args, budget):
    require(args.plan.resolve() == (ROOT/BASE/"report-plan-01.json").resolve()
            and args.out.resolve() == (ROOT/BASE/"report-01").resolve(), "Fixed separate report paths")
    require(pin(args.plan_sha256) and sha(args.plan) == args.plan_sha256, "External report plan pin")
    plan = read(args.plan)
    require(set(plan) == {"version", "limits", "sources", "runtime", "synthetic_qualification", "pilot_plan", "run", "out"}
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
            and qualified["real_task_inputs_read"] is False, "Same-source artificial qualification")
    pilot_binding, binding = plan["pilot_plan"], plan["run"]
    require(set(pilot_binding) == {"path", "sha256"} and pin(pilot_binding["sha256"])
            and sha(ROOT/pilot_binding["path"]) == pilot_binding["sha256"], "Prospective pilot pin before helper execution")
    pilot_path = ROOT/pilot_binding["path"]
    pilot = read(pilot_path)
    for name, digest in pilot["source_sha256"].items():
        require(pin(digest) and sha(ROOT/name) == digest, "Pilot source closure before read-only authentication")
        budget.check()
    require({RUNNER, SUPERVISOR, CLOCK}.issubset(pilot["source_sha256"]), "Complete directly used helper pins")
    require(set(binding) == {"path", "completed_sha256", "terminal"}
            and set(binding["terminal"]) == {"path", "sha256"}, "Complete run and external parent bindings")
    run = ROOT/binding["path"]
    authenticated, _ = upstream_authenticate(pilot_path, pilot_binding["sha256"], run)
    require(authenticated == pilot and pilot["version"] == PILOT_VERSION and pilot["fit_order"] == FIT_ORDER
            and all(len(pilot["selected"][split]) == len(set(pilot["selected"][split])) == DIALOGUES for split in ("train", "dev")),
            "Complete fixed two-seed four-arm pilot identity")
    require(pin(binding["completed_sha256"]) and sha(run/"completed.json") == binding["completed_sha256"], "External run completion pin")
    done = read(run/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "run" and done["version"] == PILOT_VERSION
            and done["fit_order"] == FIT_ORDER and [f["fit_id"] for f in done["fits"]] == FIT_ORDER
            and done["plan_sha256"] == pilot_binding["sha256"] and done["source_sha256"] == pilot["source_sha256"]
            and done["selected"] == pilot["selected"] and done["quality_metrics_computed"] is False
            and done["official_test_opened"] is False and done["limits"] == pilot["limits"]["run"]
            and done["peak_rss_bytes"] <= done["limits"]["rss_bytes"]
            and done["sampled_mps_driver_max_bytes"] <= done["limits"]["mps_driver_bytes"], "Complete fixed pilot and resource acceptance")
    expected = {"started.json", "plan.json", "rows-train.jsonl", "rows-dev.jsonl"}
    expected |= {"cache/"+name for name in ("tokens.npy", "pooled.npy", "offsets.npy", "index.json", "completed.json")}
    expected |= {fit+"/"+name for fit in FIT_ORDER for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    require(set(done["files"]) == expected, "Exact successful pilot payload set")
    manifest(run, done["files"])
    require(sha(run/"plan.json") == pilot_binding["sha256"], "Saved prospective plan unchanged")
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
    parent_terminal(terminal, read(launch_path), done, pilot, run, pilot_path, pilot_binding["sha256"])
    cache = read(run/"cache/completed.json")
    require(cache == {"status": "completed", **done["cache"]}, "Complete cache witness")
    manifest(run/"cache", cache["files"])
    require(set(cache["files"]) == {"tokens.npy", "pooled.npy", "offsets.npy", "index.json"}
            and math.isfinite(cache["maximum_pooled_error"]) and 0 <= cache["maximum_pooled_error"] <= 2e-5,
            "Frozen original pooled-vector parity")
    initial = {}
    expected_updates = EPOCHS * math.ceil(DIALOGUES/BATCH)
    for record in done["fits"]:
        name = record["fit_id"]
        arm, seed = name.rsplit("-", 1)
        path = run/name
        require(sha(path/"completed.json") == record["completed_sha256"], "Nested complete fit pin")
        fit = read(path/"completed.json")
        require(fit == {k: v for k, v in record.items() if k != "completed_sha256"}
                and fit["status"] == "completed" and fit["method"] == arm and fit["seed"] == int(seed)
                and fit["evaluation_rows"] == done["row_counts"]["dev"], "Complete nested fit identity and endpoint coverage")
        manifest(path, fit["files"])
        require(set(fit["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}, "Exact fit artifacts")
        counts = fit["counts"]
        require(counts["optimizer_updates"] == counts["optimizer_attempts"] == expected_updates
                and counts["training_forwards"] == counts["backward_attempts"] == counts["backward_returns"] == EPOCHS*DIALOGUES
                and counts["evaluation_forwards"] == DIALOGUES
                and counts["forward_attempts"] == counts["forward_returns"] == (EPOCHS+1)*DIALOGUES
                and counts["training_endpoints"] == EPOCHS*done["row_counts"]["train"], "All fixed updates, autonomous forwards and training endpoints")
        require(initial.setdefault(seed, fit["initial_state_sha256"]) == fit["initial_state_sha256"], "Paired full initialization")
        for phase, forwards in (("training_work", EPOCHS*DIALOGUES), ("evaluation_work", DIALOGUES)):
            work = fit[phase]
            require(work["forward_attempts"] == work["forward_returns"] == forwards
                    and work["observation_attempts"] == work["observation_returns"] == work["step_attempts"]
                    == work["step_returns"] == work["state_checks"] >= forwards
                    and work["real_question_updates"] >= work["state_checks"]
                    and (work["attention_positions"] == 0 if arm == "pooled" else work["attention_positions"] > 0),
                    "Complete internal observation/state work")
        require(counts["public_turns"] == fit["training_work"]["state_checks"]+fit["evaluation_work"]["state_checks"]
                and counts["question_updates"] == fit["training_work"]["real_question_updates"]+fit["evaluation_work"]["real_question_updates"],
                "Public and flattened question work identity")
        budget.check()
    return plan, pilot, done, terminal, run


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


def compute(plan, pilot, done, terminal, run, budget):
    rows = []
    with (run/"rows-dev.jsonl").open() as stream:
        for line in stream:
            require(bool(line.strip()), "Nonempty evaluator row")
            rows.append(json.loads(line))
            budget.check()
    require(len(rows) == done["row_counts"]["dev"]
            and list(dict.fromkeys(row["dialogue_id"] for row in rows)) == pilot["selected"]["dev"], "Entire selected DEV population in frozen order")
    fits = {}
    for name in FIT_ORDER:
        with np.load(run/name/"predictions.npz", allow_pickle=False) as packet:
            require(set(packet.files) == {"log_probs", "row_indices"}, "Exact prediction packet members")
            fits[name] = score(packet["log_probs"], packet["row_indices"], rows)
        budget.check()
    continuation = criteria(fits)
    families = {}
    for arm in ARMS:
        selected = [fits[f"{arm}-{seed}"] for seed in SEEDS]
        families[arm] = {"fit_ids": [f"{arm}-{seed}" for seed in SEEDS],
                         "all_dev_mean": {"macro_accuracy": float(sum((macro(f) for f in selected), Fraction())/len(SEEDS)),
                                          **{k: math.fsum(f["panels"]["all"]["micro"][k] for f in selected)/len(SEEDS) for k in ("nll", "brier")}}}
    return {"version": VERSION, "status": "completed", "technical_validity_passed": True, "complete_fits": len(fits),
            "fit_order": FIT_ORDER, "fits": fits, "families": families, "continuation": continuation,
            "model_calls": 0, "temperature_fitted": False, "probabilities_transformed": False, "official_test_opened": False,
            "bindings": {"pilot_plan": plan["pilot_plan"], "run": plan["run"]},
            "costs": {"actual_parent_wall_seconds": terminal["wall_seconds"], "worker_wall_seconds": done["wall_seconds"],
                      "cache": done["cache"], "fits": {f["fit_id"]: {k: f[k] for k in
                       ("training_seconds", "evaluation_seconds", "wall_seconds", "registered_parameters", "gradient_present_parameters",
                        "nonzero_gradient_parameters", "configuration", "training_work", "evaluation_work")} for f in done["fits"]},
                      "scope": "Parent total already includes shared caching, loading, all fits and cleanup. Nested components are not added. "
                               "Reporter, synthetic qualification and human development are separate; not a serving benchmark."},
            "scope": SCOPE}


def report_text(summary):
    gate = summary["continuation"]
    lines = ["# Belief-conditioned pooling: complete development pilot", "",
             f"All 8 fits completed. Continuation: {'PASS' if gate['passed'] else 'FAIL'}, {gate['checks_passed']}/18 conditions passed.",
             "", SCOPE, "", "| Fit | All-DEV macro accuracy | NLL | Brier |", "|---|---:|---:|---:|"]
    for name in FIT_ORDER:
        panel = summary["fits"][name]["panels"]["all"]
        lines.append(f"| {name} | {100*panel['macro_accuracy']:.4f}% | {panel['micro']['nll']:.8g} | {panel['micro']['brier']:.8g} |")
    lines += ["", "Accuracy equally averages three strata; proper scores average endpoints. Both fixed seeds are retained.",
              "", "| Comparator | Condition | Result |", "|---|---|---|"]
    lines.extend(f"| {c['comparator']} | {c['name']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in gate["checks"])
    lines += ["", f"Actual supervised parent interval: {summary['costs']['actual_parent_wall_seconds']:.3f} seconds, including shared caching and all eight fits.",
              "", ("[summary.json](summary.json) retains every fit, both seeds, seen/unseen panels, transition strata and changed subtypes. "
              "Missing descriptive support is null. Recovery concerns the next scored endpoint in the same dialogue/query; "
              "unscored turns may intervene and each model selects a different prior-error subset. It is descriptive and never changes the gate."), ""]
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
        plan, pilot, done, terminal, run = authenticate(args, budget)
        summary = compute(plan, pilot, done, terminal, run, budget)
        write(args.out/"summary.json", summary)
        with (args.out/"report.md").open("x") as stream:
            stream.write(report_text(summary))
        authenticate(args, budget)
        budget.storage()
        receipt = {"version": VERSION, "status": "completed", "plan_sha256": args.plan_sha256,
                   "source_sha256": plan["sources"], "pilot_plan": plan["pilot_plan"], "run": plan["run"],
                   "files": members(args.out, "receipt.json"), "complete_fits": 8, "model_calls": 0,
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
                      "checks_total": 18, "receipt_sha256": sha(arguments.out/"receipt.json")}))
