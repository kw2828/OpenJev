"""Fabricated complete process records; no policies, arrays or live run reads."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_query_advantage_audit_v2_test", ROOT / "scripts/audit_otto_query_advantage_v2.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def process_fixture(*, unbuffered=False):
    root = Path("/fabricated/OpenJev")
    base = root / "output/otto-query-advantage-v2"
    plan, run, launch_path = base / "plan-01.json", base / "run-01", base / "supervision-01.launch.json"
    plan_pin, launch_pin = "1" * 64, "2" * 64
    command = [str(root / ".venv-otto-released-native/bin/python")]
    if unbuffered:
        command.append("-u")
    command += [str(root / audit.PRODUCER), "run", "--plan", str(plan), "--plan-sha256", plan_pin,
                "--supervision", str(launch_path), "--output", str(run)]
    tick = 1000000000000000
    launch = {"version": "dialogue-observation-supervision-v2", "command": command,
        "cwd": str(root), "cap_seconds": 900, "pid": 901, "pgid": 901, "parent_pid": 900,
        "started_ns": tick, "deadline_ns": tick + 900 * 10**9,
        "clock_backend": "mach_continuous_time", "clock_source_sha256": audit.CLOCK_PIN,
        "watchdog_sha256": audit.SUPERVISOR_PIN, "started_unix": 1700000000.}
    terminal = {**copy.deepcopy(launch), "status": "completed", "returncode": 0, "timed_out": False,
        "error": None, "clock_error": None, "group_absent": True, "finished_ns": tick + 20 * 10**9,
        "elapsed_ns": 20 * 10**9, "wall_seconds": 20., "finished_unix": 1700000020.,
        "timing_available": True,
        "cleanup": {"errors": [], "group_absent": True, "reaped": True, "signals": [], "group_poll_attempts": 1}}
    worker = {"status": "completed", "complete": True, "requires_successful_original_supervisor": True,
        "pending": [], "pending_publications": [], "sampler_pending": [],
        "journal_closure": {"pending": [], "errors": [], "poisoned": False, "close_seconds": .002},
        "plan_sha256": plan_pin, "supervision_sha256": launch_pin,
        "started_ns": tick + 10**9, "finished_ns": tick + 19 * 10**9, "wall_seconds": 18.}
    started = {"request": {"mode": "run", "plan": str(plan), "plan_sha256": plan_pin,
                          "supervision": str(launch_path), "output": str(run)},
               "launch": copy.deepcopy(launch), "started_ns": worker["started_ns"], "clock_backend": launch["clock_backend"]}
    arguments = {"root": root, "plan_path": plan, "plan_sha256": plan_pin, "run_path": run,
                 "launch_path": launch_path, "launch_sha256": launch_pin}
    return [worker, terminal, launch, started], arguments


@pytest.mark.parametrize("unbuffered", [False, True])
def test_complete_absolute_original_launch_contract(unbuffered):
    records, kwargs = process_fixture(unbuffered=unbuffered)
    before = copy.deepcopy(records)
    result = audit.validate_producer_process(*records, **kwargs)
    assert result["producer"] == "scripts/study_otto_query_advantage_v2.py"
    assert result["canonical_command"][:3] == [str(kwargs["root"] / ".venv-otto-released-native/bin/python"),
                                               str(kwargs["root"] / audit.PRODUCER), "run"]
    assert result["worker_seconds"] == 18 and result["parent_seconds"] == 20
    assert records == before


@pytest.mark.parametrize("defect", ["relative_script", "wrong_script", "wrong_interpreter", "wrong_mode", "duplicate_option", "extra_option"])
def test_actual_command_contract_rejects_noncanonical_launch(defect):
    records, kwargs = process_fixture()
    command = records[1]["command"]
    if defect == "relative_script":
        command[1] = audit.PRODUCER
    elif defect == "wrong_script":
        command[1] = str(kwargs["root"] / "scripts/study_otto_query_advantage.py")
    elif defect == "wrong_interpreter":
        command[0] = str(kwargs["root"] / ".venv/bin/python")
    elif defect == "wrong_mode":
        command[2] = "plan"
    elif defect == "duplicate_option":
        command[9] = "--plan"
    else:
        command += ["--retry", "1"]
    # Make the embedded copies agree, so the intended command check is tested.
    records[2]["command"] = copy.deepcopy(command)
    records[3]["launch"]["command"] = copy.deepcopy(command)
    with pytest.raises(ValueError, match="invocation|option joins"):
        audit.validate_producer_process(*records, **kwargs)


@pytest.mark.parametrize("defect", ["timeout", "cleanup_error", "group_present", "wrong_parent", "late_worker", "clock_identity", "wrong_elapsed"])
def test_original_terminal_failure_cannot_admit_worker(defect):
    records, kwargs = process_fixture()
    worker, terminal, launch, started = records
    if defect == "timeout":
        terminal.update(status="failed", returncode=1, timed_out=True, error="TimeoutError('deadline')")
    elif defect == "cleanup_error":
        terminal["cleanup"]["errors"] = ["unreaped child"]
    elif defect == "group_present":
        terminal["cleanup"]["group_absent"] = False
    elif defect == "wrong_parent":
        for record in (terminal, launch, started["launch"]):
            record["parent_pid"] = record["pid"]
    elif defect == "late_worker":
        worker["finished_ns"] = terminal["deadline_ns"] + 1
    elif defect == "clock_identity":
        for record in (terminal, launch, started["launch"]):
            record["clock_source_sha256"] = "f" * 64
    else:
        terminal["elapsed_ns"] += 1
    with pytest.raises(ValueError):
        audit.validate_producer_process(*records, **kwargs)


@pytest.mark.parametrize("defect", ["plan_pin", "launch_pin", "pending", "start_request", "start_clock", "incomplete"])
def test_worker_start_and_external_pin_joins_are_required(defect):
    records, kwargs = process_fixture()
    if defect == "plan_pin":
        records[0]["plan_sha256"] = "3" * 64
    elif defect == "launch_pin":
        kwargs["launch_sha256"] = "4" * 64
    elif defect == "pending":
        records[0]["sampler_pending"] = [{"anchor_id": 0, "operation": "teacher_choose"}]
    elif defect == "start_request":
        records[3]["request"]["output"] += "-other"
    elif defect == "start_clock":
        records[3]["started_ns"] += 1
    else:
        records[0]["complete"] = False
    with pytest.raises(ValueError):
        audit.validate_producer_process(*records, **kwargs)


@pytest.mark.parametrize("defect", ["pending", "errors", "poisoned", "close_seconds"])
def test_durable_journal_must_close_before_success(defect):
    records, kwargs = process_fixture()
    bad = {"pending": [{"id": 2, "stage": "fsync"}], "errors": ["write failed"],
           "poisoned": True, "close_seconds": float("nan")}
    records[0]["journal_closure"][defect] = bad[defect]
    with pytest.raises(ValueError, match="journal fully closed"):
        audit.validate_producer_process(*records, **kwargs)


def test_scientific_analysis_ast_preserved_from_frozen_v1():
    # The change is process/seed admission only. This reads source, not evidence.
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)}

    old = functions(ROOT / "scripts/audit_otto_query_advantage.py")
    new = functions(ROOT / "scripts/audit_otto_query_advantage_v2.py")
    for name in ("panel_statistics", "weighted_statistics", "signal_statistics", "bootstrap_admission"):
        assert new[name] == old[name], name
    assert audit.FIRST_SEEDS == {"base": 19500001, "shift": 19600001}
    assert audit.LABEL_SEED == 19700001 and audit.BOOTSTRAP_SEED == 19800001
    assert audit.BOOTSTRAP_DRAWS == 2000
