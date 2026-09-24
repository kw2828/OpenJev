"""Fabricated metadata, deliberately invalid numerical payloads, no live data."""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("residual_native_bridge_tests",
                                             Path(__file__).parents[1] / "scripts/otto_residual_native_bridge.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def pin(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def saved(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return {"path": str(path), **pin(path)}


def opaque(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a model, array, journal, or JSON; hash only\x00\xff")
    return {"path": str(path), **pin(path)}


def launch(root, command, start, cap, pid):
    return {"version": "dialogue-observation-supervision-v2", "command": command, "cwd": str(root),
            "parent_pid": pid - 1, "pid": pid, "pgid": pid, "cap_seconds": cap,
            "clock_backend": "mach_continuous_time", "clock_source_sha256": M.PINS[M.CLOCK],
            "watchdog_sha256": M.PINS[M.SUPERVISOR], "started_ns": start,
            "deadline_ns": start + cap * 10**9}


def terminal(started, finish):
    return {**started, "status": "completed", "returncode": 0, "timed_out": False,
            "group_absent": True, "timing_available": True, "error": None, "clock_error": None,
            "finished_ns": finish, "cleanup": {"reaped": True, "group_absent": True, "errors": []}}


def metadata_fixture(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    sources = {}
    for name in (*M.PINS, M.SELF):
        sources[name] = opaque(root / name)["sha256"]
    monkeypatch.setattr(M, "PINS", {name: sources[name] for name in M.PINS})
    qualified = {name: pin(root / name) for name in sources}
    log = opaque(root / "engineering/check.log")
    engineering = saved(root / "engineering/receipt.json", {
        "status": "passed", "source_before": qualified, "source_after": qualified,
        "sources_unchanged": True, "commands": [{"returncode": 0, "timed_out": False,
                                                  "reaped": True, "group_absent": True}],
        "files": {"check.log": {k: log[k] for k in ("sha256", "bytes")}}})
    inherited = saved(root / "inherited.json", {"metadata": "not interpreted by consumer"})
    native = {"weights": opaque(root / "weights.h5"), "kernel": opaque(root / "kernel.npz")}
    runtime = {"python_executable": str(root / M.INTERPRETER), "python_version": "synthetic",
               "all_distributions": {"synthetic": "1"}}
    names = {"started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
             "episode-boundaries.jsonl", "episodes.jsonl", "dev.npz", "costs.json", "summary.json",
             *{name + ".jsonl.gz" for name in ("work", "weights", "forwards", "transitions", "samples")}}
    plan = {"version": "otto-residual-collection-v1", "status": "frozen_before_collection", "phase": "dev",
            "inputs": {"engineering": engineering, "collection_plan": inherited, "seed_review": inherited},
            "sources": sources, "native_inputs": native, "runtime": runtime,
            "limits": {"native_seconds": 3600}, "payloads": sorted(names)}
    plan_desc = saved(root / "collection-plan.json", plan)
    collection_dir = root / "collection"
    collection_dir.mkdir()
    command = [str(root / M.INTERPRETER), str(root / M.COLLECTOR), "run",
               "--plan", plan_desc["path"], "--plan-sha256", plan_desc["sha256"],
               "--supervision", str(root / "collection.launch.json"), "--output", str(collection_dir)]
    collection_launch = launch(root, command, 100, 3600, 200)
    launch_desc = saved(root / "collection.launch.json", collection_launch)
    for name in names - {"started.json"}:
        opaque(collection_dir / name)
    saved(collection_dir / "started.json", {"launch": collection_launch, "started_ns": 110,
        "request": {"mode": "run", "plan": plan_desc["path"], "plan_sha256": plan_desc["sha256"],
                    "supervision": launch_desc["path"], "output": str(collection_dir)}})
    collection = {"version": plan["version"], "phase": "dev", "status": "completed", "complete": True,
        "plan_sha256": plan_desc["sha256"], "sources": sources, "inputs": plan["inputs"],
        "native_inputs": native, "limits": plan["limits"], "completed_episodes": 18, "dev_episodes": 18,
        "confirm_episodes": 0, "master_episodes": 54, "training_updates": 0, "old_test_array_decodes": 0,
        "pending": [], "pending_episode": None, "pending_action": None, "pending_emission": None,
        "requires_successful_original_supervisor": True, "started_ns": 110, "finished_ns": 800,
        "supervision_sha256": launch_desc["sha256"], "files": {name: pin(collection_dir / name) for name in names}}
    collection_desc = saved(collection_dir / "receipt.json", collection)
    terminal_desc = saved(root / "collection.terminal.json", terminal(collection_launch, 900))
    worker_inputs = {"collection_plan": plan_desc, "collection_receipt": collection_desc,
                     "collection_terminal": terminal_desc, "engineering": engineering}
    out = root / "bridge"
    out.mkdir()
    command = [str(root / M.INTERPRETER), "-u", str(root / M.SELF), "run"]
    for role, row in worker_inputs.items():
        flag = "--" + role.replace("_", "-")
        command.extend((flag, row["path"], flag + "-sha256", row["sha256"]))
    command.extend(("--supervision", str(root / "bridge.launch.json"), "--output", str(out)))
    bridge_launch = launch(root, command, 1000, 60, 300)
    bridge_launch_desc = saved(root / "bridge.launch.json", bridge_launch)
    saved(out / "started.json", {"launch": bridge_launch, "started_ns": 1010, "inputs": worker_inputs})
    saved(out / "runtime.json", runtime)
    receipt = {"version": M.VERSION, "status": "completed", "complete": True, "pending": None,
        "counts": dict(M.ZERO_COUNTS), "requires_successful_original_supervisor": True,
        "bridge_sources": {**M.PINS, M.SELF: sources[M.SELF]}, "sources": sources,
        "inputs": worker_inputs, "native_runtime": runtime, "native_inputs": native,
        "collection_directory": str(collection_dir),
        "collection_payloads": {name: {"path": str(collection_dir / name), **pin(collection_dir / name)} for name in names},
        "started_ns": 1010, "finished_ns": 1800, "supervision_sha256": bridge_launch_desc["sha256"],
        "files": {name: pin(out / name) for name in M.PAYLOADS}}
    receipt_desc = saved(out / "receipt.json", receipt)
    parent_desc = saved(root / "bridge.terminal.json", terminal(bridge_launch, 1900))
    inputs = {**{k: worker_inputs[k] for k in M.ROLES[:3]},
              "bridge_receipt": receipt_desc, "bridge_terminal": parent_desc}
    return inputs, receipt, plan, collection


def test_consumer_accepts_native_evidence_without_native_runtime_or_numerical_reads(tmp_path, monkeypatch):
    inputs, _receipt, plan, collection = metadata_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "load", lambda *_a: pytest.fail("consumer imported project code"))
    original = M.read
    reads = []

    def metadata_only(path):
        reads.append(Path(path).name)
        assert Path(path).suffix == ".json"
        assert Path(path).name not in {"summary.json", "costs.json", "setup.json", "cohort.json"}
        return original(path)

    monkeypatch.setattr(M, "read", metadata_only)
    result = M.verify_bridge(inputs)
    assert result == (plan, collection, tmp_path / "collection")
    assert "runtime.json" in reads
    assert "dev.npz" not in reads


@pytest.mark.parametrize("target", ["source", "native", "collection", "engineering", "bridge_payload", "launch"])
def test_opaque_mutations_rejected_without_decoding(tmp_path, monkeypatch, target):
    inputs, receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    paths = {"source": tmp_path / M.OLD_RUNNER, "native": Path(receipt["native_inputs"]["weights"]["path"]),
             "collection": tmp_path / "collection/dev.npz", "engineering": tmp_path / "engineering/check.log",
             "bridge_payload": tmp_path / "bridge/runtime.json", "launch": tmp_path / "bridge.launch.json"}
    with paths[target].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        M.verify_bridge(inputs)


@pytest.mark.parametrize("field,value", [("complete", False), ("status", "failed"), ("pending", "authentication"),
                                        ("requires_successful_original_supervisor", False)])
def test_even_rebound_incomplete_receipt_cannot_be_promoted(tmp_path, monkeypatch, field, value):
    inputs, receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    receipt[field] = value
    inputs["bridge_receipt"] = saved(Path(inputs["bridge_receipt"]["path"]), receipt)
    with pytest.raises(ValueError, match="complete source-bound"):
        M.verify_bridge(inputs)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("returncode", 1), ("timed_out", True),
                                        ("group_absent", False), ("finished_ns", 60 * 10**9 + 1000)])
def test_original_supervisor_must_close_successfully(tmp_path, monkeypatch, field, value):
    inputs, _receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    parent = M.read(inputs["bridge_terminal"]["path"])
    parent[field] = value
    inputs["bridge_terminal"] = saved(Path(inputs["bridge_terminal"]["path"]), parent)
    with pytest.raises(ValueError, match="successful original parent"):
        M.verify_bridge(inputs)


@pytest.mark.parametrize("counter", tuple(M.ZERO_COUNTS))
def test_scientific_work_forbidden_even_with_rebound_receipt(tmp_path, monkeypatch, counter):
    inputs, receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    receipt["counts"][counter] = 1
    inputs["bridge_receipt"] = saved(Path(inputs["bridge_receipt"]["path"]), receipt)
    with pytest.raises(ValueError, match="complete source-bound"):
        M.verify_bridge(inputs)


def test_consumer_cannot_substitute_new_engineering_or_another_collection(tmp_path, monkeypatch):
    inputs, _receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    invalid = copy.deepcopy(inputs)
    invalid["engineering"] = saved(tmp_path / "new-engineering.json", {"different": "qualification"})
    with pytest.raises(ValueError, match="consumer bridge roles"):
        M.verify_bridge(invalid)
    inputs["collection_plan"] = saved(tmp_path / "other-plan.json", {"different": "collection"})
    with pytest.raises(ValueError, match="same original collection"):
        M.verify_bridge(inputs)


def test_payload_roster_and_symlink_cannot_hide_extra_evidence(tmp_path, monkeypatch):
    inputs, _receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    extra = tmp_path / "bridge/extra.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="closed payload roster"):
        M.verify_bridge(inputs)
    extra.unlink()
    target = tmp_path / "collection/dev.npz"
    moved = tmp_path / "moved.npz"
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError, match="contained regular"):
        M.verify_bridge(inputs)


def test_import_guard_blocks_both_import_routes_and_restores_state(monkeypatch):
    forbidden = "fabricated_residual_numerical_library"
    monkeypatch.setattr(M, "BLOCKED_IMPORTS", frozenset({forbidden}))
    original, meta = builtins.__import__, list(sys.meta_path)
    with M.metadata_imports_only():
        for action in (lambda: builtins.__import__(forbidden), lambda: importlib.import_module(forbidden)):
            with pytest.raises(ValueError, match="forbidden numerical import"):
                action()
    assert builtins.__import__ is original
    assert sys.meta_path == meta
    monkeypatch.setitem(sys.modules, forbidden, SimpleNamespace())
    with pytest.raises(ValueError, match="clean metadata-only"), M.metadata_imports_only():
        pytest.fail("loaded numerical module crossed worker boundary")


def test_wrong_interpreter_preserves_failure_receipt_before_any_module_load(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    monkeypatch.setattr(M, "load", lambda *_a: pytest.fail("wrong runtime imported code"))
    output = tmp_path / "failed-bridge"
    with pytest.raises(ValueError, match="exact original native interpreter"):
        M.execute(SimpleNamespace(output=output))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["complete"] is False
    assert receipt["pending"] == "admission" and receipt["counts"] == M.ZERO_COUNTS
    assert set(output.iterdir()) == {output / "receipt.json"}
    with pytest.raises(ValueError, match="exclusive contained output"):
        M.execute(SimpleNamespace(output=output))


def test_command_rejects_duplicate_flags_and_foreign_interpreter(tmp_path, monkeypatch):
    inputs, receipt, _plan, _collection = metadata_fixture(tmp_path, monkeypatch)
    command = M.read(tmp_path / "bridge.launch.json")["command"]
    assert M.command_options(command, M.SELF, M.bridge_flags())["--collection-plan"] == inputs["collection_plan"]["path"]
    wrong = command.copy()
    wrong[0] = str(tmp_path / ".venv/bin/python")
    with pytest.raises(ValueError, match="exact native command"):
        M.command_options(wrong, M.SELF, M.bridge_flags())
    wrong = command.copy()
    wrong[wrong.index("--collection-receipt")] = "--collection-plan"
    with pytest.raises(ValueError, match="exact unique command flags"):
        M.command_options(wrong, M.SELF, M.bridge_flags())
    assert receipt["native_runtime"]["python_executable"] != sys.executable
