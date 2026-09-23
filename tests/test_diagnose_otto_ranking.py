"""Fabricated provenance/process checks; no scientific data or model calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("_ranking_runner_fixture",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_otto_ranking.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_hash_rejects_relative_paths_and_symlinks_and_detects_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    p = tmp_path / "input.json"; p.write_bytes(b"original")
    assert runner.pin(p) == {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}
    with pytest.raises(ValueError):
        runner.pin(Path("input.json"))
    link = tmp_path / "linked.json"; link.symlink_to(p)
    with pytest.raises(ValueError):
        runner.pin(link)
    first = runner.pin(p); p.write_bytes(b"changed")
    assert runner.pin(p) != first


def process_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for name in runner.SOURCES[-2:]:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
    plan, output, launch_path = tmp_path / "plan.json", tmp_path / "run", tmp_path / "launch.json"
    plan.write_text("{}")
    plan_pin = runner.pin(plan)["sha256"]
    command = [str(tmp_path / ".venv/bin/python"), str(tmp_path / "scripts/diagnose_otto_ranking.py"),
        "run", "--plan", str(plan), "--plan-sha256", plan_pin, "--supervision", str(launch_path), "--output", str(output)]
    launch = {"command": command, "cap_seconds": 240, "cwd": str(tmp_path), "started_ns": 100,
        "deadline_ns": 240 * 10**9 + 100, "pid": 123, "pgid": 123, "parent_pid": 122,
        "watchdog_sha256": runner.pin(tmp_path / runner.SOURCES[-2])["sha256"],
        "clock_source_sha256": runner.pin(tmp_path / runner.SOURCES[-1])["sha256"]}
    launch_path.write_text(json.dumps(launch))
    worker = {"status": "completed", "plan_sha256": plan_pin,
        "supervision_sha256": runner.pin(launch_path)["sha256"], "started_ns": 110, "finished_ns": 200}
    terminal = {**launch, "finished_ns": 220, "status": "completed", "returncode": 0,
        "timed_out": False, "group_absent": True, "cleanup": {"reaped": True, "errors": []},
        "error": None, "clock_error": None}
    args = {"script": "diagnose_otto_ranking.py", "output": output,
            "plan": plan, "plan_pin": plan_pin, "cap": 240}
    return worker, terminal, args, launch_path


def test_original_process_join_accepts_exact_successful_parent(tmp_path, monkeypatch):
    worker, terminal, args, _ = process_fixture(tmp_path, monkeypatch)
    before = copy.deepcopy((worker, terminal))
    result = runner.original_join(worker, terminal, **args)
    assert result["--plan"] == str(args["plan"])
    assert (worker, terminal) == before


@pytest.mark.parametrize("fault", ["timeout", "live_group", "not_reaped", "clock", "returncode",
                                  "late", "cap", "prefix", "duplicate", "launch_drift", "source_drift"])
def test_original_process_join_rejects_incomplete_or_substituted_evidence(tmp_path, monkeypatch, fault):
    worker, terminal, args, launch = process_fixture(tmp_path, monkeypatch)
    if fault == "timeout": terminal["timed_out"] = True
    elif fault == "live_group": terminal["group_absent"] = False
    elif fault == "not_reaped": terminal["cleanup"]["reaped"] = False
    elif fault == "clock": terminal["clock_error"] = "unavailable"
    elif fault == "returncode": terminal["returncode"] = 1
    elif fault == "late": worker["finished_ns"] = terminal["deadline_ns"] + 1
    elif fault == "cap": terminal["cap_seconds"] = 241
    elif fault == "prefix": terminal["command"][1] = "/replacement.py"
    elif fault == "duplicate": terminal["command"] += ["--plan", str(args["plan"])]
    elif fault == "launch_drift": launch.write_text("{}")
    else: (tmp_path / runner.SOURCES[-1]).write_text("changed")
    with pytest.raises((ValueError, KeyError)):
        runner.original_join(worker, terminal, **args)


def test_output_write_is_exclusive_and_refuses_nonfinite_json(tmp_path):
    p = tmp_path / "out.json"
    runner.write(p, {"a": 1})
    with pytest.raises(FileExistsError):
        runner.write(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}
    with pytest.raises(ValueError):
        runner.write(tmp_path / "invalid.json", {"a": float("nan")})


def test_saved_json_size_is_checked_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    p = tmp_path / "large.json"
    with p.open("wb") as stream:
        stream.truncate(80 * 1024**2 + 1)
    with pytest.raises(ValueError, match="bounded"):
        runner.read(p)


def test_hydration_restores_qualified_window_contract_without_touching_arrays():
    import numpy as np

    from openjev.research import otto_score_forecast_data as base

    features = np.zeros((2, 31), np.float32)
    features[:, 15] = np.arange(2, dtype=np.float32) / 2188
    features[:, 16] = features[:, 15]
    features[:, 17] = 1
    original = base.build_windows([{"id": "synthetic", "regime": "lambda3", "features": features,
        "teacher_scores": np.zeros((2, 4), np.float32), "legal": np.ones((2, 4), np.bool_)}])
    arrays = {k: v for k, v in original.items() if isinstance(v, np.ndarray)}
    metadata = {"version": original["version"], "episode_ids": ["synthetic"], "episode_regimes": ["lambda3"]}
    windows = runner.hydrate_windows(arrays, metadata)
    result = base._metric_inputs(windows, np.zeros_like(arrays["targets"]))
    assert result[0] == ("synthetic",) and result[1] == ("lambda3",)
    assert all(windows[k] is v for k, v in arrays.items())
    assert "version" not in arrays
    assert isinstance(metadata["episode_ids"], list)


def test_publish_rejects_oversized_encoded_result_before_creating_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LIMITS", {**runner.LIMITS, "output_bytes": 1024**2 + 32})
    job = runner.Run.__new__(runner.Run); job.out = tmp_path; job.check = lambda: None
    with pytest.raises(ValueError, match="encoded output"):
        job.publish("large.json", {"large": "x" * 100})
    assert not (tmp_path / "large.json").exists()
    job.publish("small.json", {"a": 1})
    assert (tmp_path / "small.json").read_bytes() == b'{"a":1}\n'


def test_clock_failure_still_demotes_completed_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    class FailedClock:
        def now_ns(self):
            raise RuntimeError("clock unavailable")
    job = runner.Run.__new__(runner.Run)
    job.created, job.out, job.clock, job.start = True, tmp_path, FailedClock(), 0
    job.receipt = {"status": "completed"}
    runner.write(tmp_path / "receipt.json", {"status": "completed"})
    job.fail(RuntimeError("original failure"))
    assert json.loads((tmp_path / "receipt.invalid.json").read_text())["status"] == "completed"
    failure = json.loads((tmp_path / "receipt.json").read_text())
    assert failure["status"] == "failed" and failure["finished_ns"] is None
    assert "original failure" in failure["error"] and "clock unavailable" in failure["clock_error"]
