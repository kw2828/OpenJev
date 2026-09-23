"""Metadata-only closure report for an unsuccessful original collection.

Does not import scientific modules, decode arrays, open compressed journals,
score forecasts, resume work or evaluate continuation criteria. Complete episode
credit is limited to the worker's acknowledged prefix with matching rows and
boundaries. Other complete lines are observed bytes, not proven durable records.
Partial final JSONL bytes are described, never repaired/read as
a completed record. Gzip contents remain opaque authenticated bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-score-forecast-stop-summary-v1"
COLLECTOR = "scripts/collect_otto_score_forecasts.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def pinned(value, expected):
    require(descriptor(value) == {key: expected[key] for key in ("sha256", "bytes")},
            f"unchanged input/payload: {value}")


def jsonl_prefix(path, maximum):
    """Read newline-terminated JSON records only; no partial-tail completion."""
    if not path.exists():
        return [], {"present": False, "complete_records": 0, "uncredited_tail_bytes": 0}
    records, tail = [], b""
    with regular(path).open("rb") as stream:
        while line := stream.readline(2 * 1024**2 + 1):
            require(len(line) <= 2 * 1024**2, "bounded original metadata row")
            if not line.endswith(b"\n"):
                tail = line
                require(not stream.read(1), "partial metadata only at physical file end")
                break
            records.append(json.loads(line))
            require(len(records) <= maximum, "fixed metadata record cap")
    return records, {"present": True, "complete_records": len(records),
                     "uncredited_tail_bytes": len(tail),
                     "uncredited_tail_sha256": hashlib.sha256(tail).hexdigest() if tail else None}


def summarize(args):
    plan_path, receipt_path, terminal_path = map(regular, (args.plan, args.receipt, args.terminal))
    plan, receipt, terminal = read(plan_path), read(receipt_path), read(terminal_path)
    directory = receipt_path.parent
    require(plan["version"] == receipt["version"] == "otto-score-forecast-collection-v1"
            and plan["status"] == "frozen_before_collection", "original collection identity")
    require(receipt["status"] != "completed" or terminal["status"] != "completed"
            or terminal["returncode"] != 0 or terminal["timed_out"], "unsuccessful original allocation only")
    require(terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True,
            "original process group fully closed and reaped")
    require(receipt["plan_sha256"] == descriptor(plan_path)["sha256"], "original plan hash")
    require(receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"]
            and receipt["native_inputs"] == plan["native_inputs"], "original source/input maps")
    for name, pin in plan["sources"].items():
        require(descriptor(name)["sha256"] == pin, "frozen source: " + name)
    for entry in [*plan["inputs"].values(), *plan["native_inputs"].values()]:
        pinned(entry["path"], entry)
    require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"},
            "exact retained partial file inventory")
    for name, entry in receipt["files"].items():
        require(Path(name).name == name, "flat retained payload name")
        pinned(directory / name, entry)
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"), str(ROOT / COLLECTOR), "run"]
            and len(command) == 11 and len(set(command[3::2])) == 4, "original absolute native command")
    options = dict(zip(command[3::2], command[4::2], strict=True))
    require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
            and options["--plan"] == str(plan_path) and options["--plan-sha256"] == receipt["plan_sha256"]
            and options["--output"] == str(directory), "original invocation joins")
    launch_path = regular(options["--supervision"])
    launch = read(launch_path)
    require(descriptor(launch_path)["sha256"] == receipt["supervision_sha256"]
            and all(terminal[key] == value for key, value in launch.items()), "original launch/terminal join")
    require(launch["cwd"] == str(ROOT) and launch["cap_seconds"] == 900
            and launch["deadline_ns"] == launch["started_ns"] + 900 * 10**9
            and launch["clock_source_sha256"] == plan["sources"][CLOCK]
            and launch["watchdog_sha256"] == plan["sources"][SUPERVISOR], "original frozen parent allocation")
    started = read(directory / "started.json")
    require(started["launch"] == launch, "original started witness")
    declared = plan["cohort"]
    require(len(declared) == 90 and [r["episode_index"] for r in declared] == list(range(90))
            and len({r["episode_id"] for r in declared}) == 90
            and [r["stage"] for r in declared] == ["train"] * 54 + ["valid"] * 36,
            "complete declared cohort identities")
    rows, row_status = jsonl_prefix(directory / "episodes.jsonl", 90)
    boundaries, boundary_status = jsonl_prefix(directory / "episode-boundaries.jsonl", 180)
    acknowledged = receipt["completed_episodes"]
    require(type(acknowledged) is int and 0 <= acknowledged <= len(rows) <= min(90, acknowledged + 1),
            "acknowledged prefix with at most one unacknowledged row")
    for row, identity in zip(rows, declared, strict=False):
        require({key: row[key] for key in identity} == identity, "episode rows follow declared prefix")
        require(type(row["steps"]) is int and 1 <= row["steps"] <= 2188
                and row["end_row"] - row["start_row"] == row["steps"], "complete row bookkeeping")
    attempts, returns, pending = [], [], None
    stage_offset = {"train": 0, "valid": 0}
    for event in boundaries:
        index = len(returns)
        require(index < 90, "bounded boundary prefix")
        identity = declared[index]
        require({key: event[key] for key in identity} == identity, "boundary follows declared prefix")
        if event["event"] == "attempt":
            require(pending is None and event["start_row"] == stage_offset[identity["stage"]],
                    "single pending episode at exact stage offset")
            pending = identity
            attempts.append(event)
        else:
            require(event["event"] == "return" and pending == identity and index < len(rows),
                    "return has original attempt and complete episode row")
            row = rows[index]
            require(all(event[key] == row[key] for key in ("steps", "start_row", "end_row")),
                    "returned boundary row join")
            stage_offset[identity["stage"]] = row["end_row"]
            returns.append(event)
            pending = None
    require(acknowledged <= len(returns) <= min(90, acknowledged + 1)
            and len(rows) <= len(returns) + 1, "only acknowledged durable prefix receives completion credit")
    worker_pending = receipt.get("pending_episode")
    pending_relation = "none"
    if worker_pending is not None:
        if acknowledged < 90 and worker_pending == declared[acknowledged]:
            pending_relation = "next_unacknowledged_episode"
        else:
            require(acknowledged > 0 and worker_pending == declared[acknowledged - 1]
                    and len(rows) >= acknowledged and len(returns) >= acknowledged,
                    "only just-acknowledged pending-clear interruption also allowed")
            pending_relation = "just_acknowledged_episode_before_pending_clear"
    require(receipt["training_updates"] == 0, "collection worker performed no fitting")
    calls = receipt["calls"]
    require(set(calls) == set(plan["call_caps"]), "exact original operation channels")
    for channel, value in calls.items():
        require(type(value["attempted"]) is int and type(value["returned"]) is int
                and 0 <= value["returned"] <= value["attempted"] <= plan["call_caps"][channel]
                and type(value["seconds"]) in (int, float) and math.isfinite(value["seconds"])
                and value["seconds"] >= 0, "bounded original attempt/return accounting")
    phases = {}
    started_ids = {row["episode_id"] for row in attempts}
    started_ids.update(row["episode_id"] for row in declared[:acknowledged])
    if worker_pending is not None:
        started_ids.add(worker_pending["episode_id"])
    for stage in ("train", "valid"):
        count = sum(r["stage"] == stage for r in declared[:acknowledged])
        attempted = sum(r["stage"] == stage for r in attempts)
        phases[stage] = {"declared_episodes": 54 if stage == "train" else 36,
                         "acknowledged_complete_episodes": count, "observed_attempt_boundaries": attempted,
                         "unstarted_episodes": sum(row["stage"] == stage and row["episode_id"] not in started_ids
                                                   for row in declared),
                         "dataset_present_unread": f"{stage}.npz" in receipt["files"],
                         "array_decode": "not performed"}
    return {"version": VERSION, "status": "incomplete_collection_verified", "rule": "not_evaluated",
            "meaning": "Original collection allocation did not close successfully; metadata closure only, no efficacy inference.",
            "source": {"path": str(Path(__file__).resolve()), **descriptor(Path(__file__))},
            "inputs": {name: {"path": str(path), **descriptor(path)} for name, path in
                       (("plan", plan_path), ("receipt", receipt_path), ("terminal", terminal_path), ("launch", launch_path))},
            "verified_sources": len(plan["sources"]), "verified_payloads": len(receipt["files"]),
            "payloads": receipt["files"], "declared_episodes": 90,
            "acknowledged_complete_episodes": acknowledged, "complete_line_episode_rows": len(rows),
            "observed_attempt_boundaries": len(attempts), "observed_return_boundaries": len(returns),
            "completion_scope": "Only the receipt-acknowledged prefix has original durable completion credit; other complete lines are observed bytes only.",
            "unacknowledged_episode_rows": len(rows) - acknowledged,
            "metadata_files": {"episodes.jsonl": row_status, "episode-boundaries.jsonl": boundary_status},
            "pending_episode": worker_pending, "pending_episode_relation": pending_relation,
            "boundary_pending_episode": pending,
            "pending_action": receipt.get("pending_action"), "pending_calls": receipt.get("pending"),
            "pending_emission": receipt.get("pending_emission"), "calls": calls,
            "calls_scope": "Original recorded attempt/return counters; compressed operation journals are not replayed.",
            "phases": phases, "fitting": "not performed by this collection worker; not admitted by this report",
            "forecast_criteria": "all 45 not evaluated", "arrays_read": 0, "scientific_calls": 0,
            "gzip_files": {name: "opaque bytes authenticated; completeness not inspected or repaired"
                           for name in receipt["files"] if name.endswith(".gz")},
            "worker_status": receipt["status"], "worker_error": receipt.get("error"),
            "worker_cleanup_errors": receipt.get("cleanup_errors", []),
            "terminal_status": terminal["status"], "terminal_returncode": terminal["returncode"],
            "terminal_timed_out": terminal["timed_out"], "terminal_error": terminal.get("error"),
            "terminal_clock_error": terminal.get("clock_error"), "terminal_cleanup": terminal["cleanup"],
            "parent_wall_seconds": terminal.get("wall_seconds"), "original_cap_seconds": 900}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "receipt", "terminal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and ".." not in args.output.parts
            and not args.output.exists() and not any(p.is_symlink() for p in args.output.parents),
            "exclusive contained absolute report output")
    try:
        value = summarize(args)
    except Exception as error:
        value = {"version": VERSION, "status": "stop_summary_failed", "rule": "not_evaluated", "error": repr(error)}
        with args.output.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        raise
    with args.output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": value["status"], "rule": "not_evaluated", "report": descriptor(args.output)}))


if __name__ == "__main__":
    main()
