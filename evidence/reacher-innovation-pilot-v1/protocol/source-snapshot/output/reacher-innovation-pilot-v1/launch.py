"""One exclusive development-pilot child with an authoritative outer timeout.

No retry or resume. The fixed 1800-second budget includes child preflight and
launcher bookkeeping. Cleanup/failure preservation may extend past the budget;
such an attempt cannot be recorded as completed. No work occurs at import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
PROCESS = ROOT / "output/reacher-innovation-pilot-v1/process"
ATTEMPT = ROOT / "runs/reacher-innovation-pilot-v1/attempt"
PLAN = ROOT / "evidence/reacher-innovation-pilot-v1/protocol/plan.json"
CHILD = ROOT / "output/reacher-innovation-pilot-v1/run_pilot.py"
PYTHON = ROOT / ".venv-robotics/bin/python"
CAP_SECONDS = 1800


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log_members():
    return {name: {"sha256": sha(PROCESS / name), "bytes": (PROCESS / name).stat().st_size}
            for name in ("stdout.log", "stderr.log") if (PROCESS / name).is_file()}


def stop_owned_child(child, notes):
    """Stop only the process group created by this invocation; never retry it."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except BaseException as error:  # noqa: BLE001
        notes.append("Owned child-group termination failed: " + repr(error))
    try:
        child.wait(timeout=5)
    except BaseException as error:  # noqa: BLE001
        notes.append("Owned child cleanup wait failed: " + repr(error))


def launch(expected_plan_sha256, published_commit):
    require(isinstance(expected_plan_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256),
            "Explicit lowercase SHA-256 plan identity required")
    require(isinstance(published_commit, str) and re.fullmatch(r"[0-9a-f]{40}", published_commit),
            "Explicit full lowercase published commit required")
    start = time.monotonic()
    PROCESS.mkdir(parents=True, exist_ok=False)
    deadline = start + CAP_SECONDS
    argv = [str(PYTHON), "-u", str(CHILD), "run", "--plan-sha256", expected_plan_sha256,
            "--published-commit", published_commit]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
    child, exit_code, phase = None, None, "admission"
    child_wait_seconds, subprocess_timeout_seconds = None, None
    source_sha = None
    try:
        source_sha = sha(SOURCE)
        require(sha(PLAN) == expected_plan_sha256, "External plan file digest mismatch")
        require(not ATTEMPT.exists(), "Existing pilot attempt cannot be launched or resumed")
        write_json(PROCESS / "started.json", {
            "utc": datetime.now(UTC).isoformat(), "argv": argv, "cwd": str(ROOT),
            "environment_overrides": {"PYTHONPATH": environment["PYTHONPATH"]},
            "plan_sha256": expected_plan_sha256, "published_commit": published_commit,
            "source_sha256": source_sha, "cap_seconds": CAP_SECONDS,
            "no_retry": True, "scope": "development pilot subprocess; no efficacy determination"})
        phase = "subprocess"
        with (PROCESS / "stdout.log").open("xb") as stdout, (PROCESS / "stderr.log").open("xb") as stderr:
            require(time.monotonic() < deadline, "Outer pilot cap exhausted before spawn")
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stdout,
                                     stderr=stderr, start_new_session=True)
            subprocess_timeout_seconds = deadline - time.monotonic()
            require(subprocess_timeout_seconds > 0, "Outer pilot cap exhausted during spawn")
            child_start = time.monotonic()
            try:
                exit_code = child.wait(timeout=subprocess_timeout_seconds)
            finally:
                child_wait_seconds = time.monotonic() - child_start
        phase = "terminal_validation"
        require(exit_code == 0, f"Pilot child exit was {exit_code}")
        require(not (ATTEMPT / "failed.json").exists()
                and not (ATTEMPT / "invalid-completion.json").exists(), "Pilot child retained a failure marker")
        completed_path = ATTEMPT / "completed.json"
        completed_bytes = completed_path.read_bytes()
        completed = json.loads(completed_bytes)
        require(completed.get("plan_sha256") == expected_plan_sha256
                and type(completed.get("fits")) is int and completed["fits"] == 12
                and type(completed.get("evaluations")) is int and completed["evaluations"] == 48
                and completed.get("development_only") is True
                and completed.get("native_control_measured") is False
                and completed.get("gate_status") == "requires_saved_output_review",
                "Child completion identity/scope/coverage mismatch")
        child_wall = completed.get("wall_seconds")
        require(type(child_wall) in (int, float) and math.isfinite(child_wall) and 0 <= child_wall < CAP_SECONDS,
                "Child completion wall time invalid or over cap")
        require(sha(PLAN) == expected_plan_sha256 and sha(SOURCE) == source_sha,
                "Plan or launcher source changed during execution")
        receipt = {"status": "completed", "utc": datetime.now(UTC).isoformat(),
            "argv": argv, "cwd": str(ROOT), "plan_sha256": expected_plan_sha256,
            "published_commit": published_commit, "source_sha256": source_sha,
            "exit_code": exit_code, "cap_seconds": CAP_SECONDS,
            "subprocess_timeout_seconds": subprocess_timeout_seconds,
            "child_wait_seconds": child_wait_seconds, "child_recorded_wall_seconds": child_wall,
            "child_completed_sha256": hashlib.sha256(completed_bytes).hexdigest(),
            "logs": log_members(), "no_retry": True,
            "development_only": True, "gate_status": "requires_saved_output_review",
            "wall_seconds_scope": "through validation/log hashing, before own receipt write; outer deadline checked after writing"}
        require(time.monotonic() < deadline, "Outer pilot cap exceeded before completion write")
        receipt["wall_seconds"] = time.monotonic() - start
        phase = "completion_write"
        write_json(PROCESS / "completed.json", receipt)
        require(time.monotonic() < deadline, "Outer pilot cap exceeded during completion write")
        return receipt
    except BaseException as error:
        original_traceback = traceback.format_exc()
        cleanup_notes = []
        stop_owned_child(child, cleanup_notes)
        if child is not None:
            exit_code = child.poll()
        try:
            completed_path = PROCESS / "completed.json"
            if completed_path.exists():
                completed_path.rename(PROCESS / "invalid-completion.json")
        except BaseException as preservation_error:  # noqa: BLE001
            cleanup_notes.append("Outer completion demotion failed: " + repr(preservation_error))
        try:
            write_json(PROCESS / "failed.json", {"status": "failed", "utc": datetime.now(UTC).isoformat(),
                "argv": argv, "cwd": str(ROOT), "plan_sha256": expected_plan_sha256,
                "published_commit": published_commit, "source_sha256": source_sha,
                "phase": phase, "exit_code": exit_code, "cap_seconds": CAP_SECONDS,
                "timeout": isinstance(error, subprocess.TimeoutExpired),
                "subprocess_timeout_seconds": subprocess_timeout_seconds,
                "child_wait_seconds": child_wait_seconds, "wall_seconds_including_cleanup": time.monotonic() - start,
                "error": repr(error), "traceback": original_traceback, "cleanup_notes": cleanup_notes,
                "logs": log_members(), "no_retry": True, "automatic_resume": False,
                "cleanup_may_exceed_cap": True})
        except BaseException as preservation_error:  # noqa: BLE001
            cleanup_notes.append("Failure receipt preservation failed: " + repr(preservation_error))
        for note in cleanup_notes:
            error.add_note(note)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--published-commit", required=True)
    arguments = parser.parse_args()
    launch(arguments.plan_sha256, arguments.published_commit)
