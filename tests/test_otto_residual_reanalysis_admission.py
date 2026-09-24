"""Independent metadata admission checks using fabricated files only.

These tests exercise the new runner's admission rather than replacing it. The
bridge verifier's own contract and real cross-interpreter smoke are qualified
separately; no fixture here represents an empirical collection or process.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "independent_residual_reanalysis_admission", REPOSITORY / "scripts/run_otto_residual_reanalysis.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
BRIDGE_ROLES = {"collection_plan", "collection_receipt", "collection_terminal", "bridge_receipt", "bridge_terminal"}


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def source_file(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fabricated source, never imported\n")
    return M.descriptor(path)["sha256"]


def admission_fixture(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.setattr(M, "CLOSED_SCREEN", "closed-screen.json")
    sources = {name: source_file(root, name) for name in (M.BRIDGE, "scripts/old-qualified-source.py")}
    old_engineering = json_file(root / "old-engineering.json", {"fabricated": "original qualification"})
    collection_plan = {"inputs": {"engineering": old_engineering}, "phase": "dev"}
    collection_receipt = {"fabricated": "completed collection"}
    inputs = {
        "collection_plan": json_file(root / "collection-plan.json", collection_plan),
        "collection_receipt": json_file(root / "collection-receipt.json", collection_receipt),
        "collection_terminal": json_file(root / "collection-terminal.json", {"finished_ns": 10}),
        "engineering": json_file(root / "new-engineering.json", {"fabricated": "new qualification"}),
        "bridge_receipt": json_file(root / "bridge-receipt.json", {"sources": sources}),
        "bridge_terminal": json_file(root / "bridge-terminal.json", {"finished_ns": 20}),
    }
    closed = {"status": "DEV_TECHNICAL_FAIL", "scientific_decision": None,
              "evaluation_views_completed": 0, "old_test_access": False,
              "collection": {"receipt": inputs["collection_receipt"], "original_supervisor": inputs["collection_terminal"]}}
    closure = json_file(root / M.CLOSED_SCREEN, closed)
    monkeypatch.setattr(M, "CLOSED_SCREEN_PIN", closure["sha256"])
    calls = []
    result = (collection_plan, collection_receipt, root)

    def verify_bridge(received):
        assert set(received) == BRIDGE_ROLES
        assert received == {role: inputs[role] for role in BRIDGE_ROLES}
        calls.append(copy.deepcopy(received))
        return result

    def loaded(path, _name):
        assert path == M.BRIDGE, "evaluator admission must not import native collection authentication"
        return SimpleNamespace(verify_bridge=verify_bridge)

    monkeypatch.setattr(M, "load", loaded)
    return SimpleNamespace(root=root, sources=sources, inputs=inputs, closed=closed,
                           calls=calls, result=result, bridge=SimpleNamespace(verify_bridge=verify_bridge))


def test_collection_admission_routes_exact_five_roles_without_old_engineering_recursion(tmp_path, monkeypatch):
    case = admission_fixture(tmp_path, monkeypatch)
    assert M.authenticate_collection(case.inputs, case.sources) == case.result
    assert len(case.calls) == 1
    assert case.result[0]["inputs"]["engineering"]["path"] != case.inputs["engineering"]["path"]
    assert "engineering" not in case.calls[0]


@pytest.mark.parametrize("role", sorted(BRIDGE_ROLES | {"engineering"}))
def test_every_external_input_is_rehashed_before_bridge_import(tmp_path, monkeypatch, role):
    case = admission_fixture(tmp_path, monkeypatch)
    path = Path(case.inputs[role]["path"])
    path.write_text(path.read_text() + " ")
    monkeypatch.setattr(M, "load", lambda *_args: pytest.fail("changed input reached a module import"))
    with pytest.raises(ValueError, match="exact input descriptor"):
        M.authenticate_collection(case.inputs, case.sources)


@pytest.mark.parametrize("defect", ("missing_role", "extra_role", "closure_pin", "bridge_source_pin"))
def test_role_and_immutable_source_failures_precede_bridge_import(tmp_path, monkeypatch, defect):
    case = admission_fixture(tmp_path, monkeypatch)
    if defect == "missing_role":
        del case.inputs["bridge_terminal"]
    elif defect == "extra_role":
        case.inputs["confirmation"] = case.inputs["engineering"].copy()
    elif defect == "closure_pin":
        monkeypatch.setattr(M, "CLOSED_SCREEN_PIN", "0" * 64)
    else:
        case.sources[M.BRIDGE] = "0" * 64
    monkeypatch.setattr(M, "load", lambda *_args: pytest.fail("invalid admission reached a module import"))
    with pytest.raises(ValueError):
        M.authenticate_collection(case.inputs, case.sources)


@pytest.mark.parametrize("defect", ("status", "decision", "prior_views", "test_opened", "receipt", "terminal"))
def test_stopped_screen_status_and_exact_collection_join_are_required(tmp_path, monkeypatch, defect):
    case = admission_fixture(tmp_path, monkeypatch)
    if defect == "status":
        case.closed["status"] = "DEV_PASS"
    elif defect == "decision":
        case.closed["scientific_decision"] = {"passed": True}
    elif defect == "prior_views":
        case.closed["evaluation_views_completed"] = 1
    elif defect == "test_opened":
        case.closed["old_test_access"] = True
    else:
        key = "receipt" if defect == "receipt" else "original_supervisor"
        case.closed["collection"][key] = json_file(case.root / "different-collection.json", {"fabricated": defect})
    pin = json_file(case.root / M.CLOSED_SCREEN, case.closed)
    monkeypatch.setattr(M, "CLOSED_SCREEN_PIN", pin["sha256"])
    monkeypatch.setattr(M, "load", lambda *_args: pytest.fail("invalid closure reached bridge import"))
    with pytest.raises(ValueError, match="not a retry"):
        M.authenticate_collection(case.inputs, case.sources)


@pytest.mark.parametrize("defect", ("missing_source", "changed_source", "bridge_rejected"))
def test_native_proof_is_required_and_cannot_escape_new_source_closure(tmp_path, monkeypatch, defect):
    case = admission_fixture(tmp_path, monkeypatch)
    if defect == "bridge_rejected":
        def rejected(_inputs):
            raise ValueError("fabricated incomplete bridge")
        monkeypatch.setattr(M, "load", lambda *_args: SimpleNamespace(verify_bridge=rejected))
    else:
        # The receipt still binds its original source map; only the consumer's
        # proposed qualification is defective here.
        case.sources = case.sources.copy()
        if defect == "missing_source":
            del case.sources["scripts/old-qualified-source.py"]
        else:
            case.sources["scripts/old-qualified-source.py"] = "0" * 64
    with pytest.raises(ValueError):
        M.authenticate_collection(case.inputs, case.sources)


@pytest.mark.parametrize("runtime_matches", (False, True))
def test_bridge_does_not_replace_training_runtime_requirement(tmp_path, monkeypatch, runtime_matches):
    case = admission_fixture(tmp_path, monkeypatch)
    training_runtime = {"executable": str(case.root / ".venv/bin/python"), "fabricated_inventory": "training"}
    lineage = {"sources": case.sources.copy(), "runtime": training_runtime, "checkpoints": {}}
    loaded = M.load

    def metadata_module(path, name):
        if path == M.LINEAGE:
            return SimpleNamespace(authenticate=lambda: copy.deepcopy(lineage))
        return loaded(path, name)

    monkeypatch.setattr(M, "load", metadata_module)
    monkeypatch.setattr(M, "engineering", lambda path: case.sources.copy())
    monkeypatch.setattr(M, "runtime_record", lambda: training_runtime if runtime_matches else {"fabricated_inventory": "native"})
    if runtime_matches:
        bound = M.authenticate("evaluate", case.inputs)
        assert bound["collection"] == case.result and bound["lineage"] == lineage
        assert bound["producer"] is None
    else:
        with pytest.raises(ValueError, match="original fixed numerical runtime"):
            M.authenticate("evaluate", case.inputs)
    assert len(case.calls) == 1


@pytest.mark.parametrize("bridge_finished", (999, 1000, 1001))
def test_worker_requires_bridge_closed_before_its_original_launch(tmp_path, monkeypatch, bridge_finished):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.chdir(root)
    for key in M.THREADS:
        monkeypatch.setenv(key, "1")
    for name, pin_name in ((M.CLOCK, "CLOCK_PIN"), (M.SUPERVISOR, "SUPERVISOR_PIN")):
        monkeypatch.setattr(M, pin_name, source_file(root, name))
    runtime = {"fabricated": "training runtime"}
    bridge = json_file(root / "bridge-terminal.json", {"finished_ns": bridge_finished})
    plan = {"version": M.VERSION, "phase": "evaluate", "status": "frozen_before_execution",
            "configuration": M.CONFIG, "limits": M.LIMITS["evaluate"], "payloads": sorted(M.PAYLOADS["evaluate"]),
            "runtime": runtime, "inputs": {"bridge_terminal": bridge}, "sources": {}, "lineage": {}}
    pin = json_file(root / "plan.json", plan)
    out = root / "worker"
    out.mkdir()
    args = SimpleNamespace(mode="run", plan=Path(pin["path"]), plan_sha256=pin["sha256"],
                           supervision=root / "launch.json", output=out)
    executable = str(root / M.INTERPRETER)
    argv = [str(root / M.SELF), "run", "--plan", str(args.plan), "--plan-sha256", args.plan_sha256,
            "--supervision", str(args.supervision), "--output", str(out)]
    monkeypatch.setattr(M.sys, "executable", executable)
    monkeypatch.setattr(M.sys, "argv", argv)
    launch = {"command": [executable, *argv], "pid": os.getpid(), "pgid": os.getpid(), "parent_pid": os.getppid(),
              "cwd": str(root), "cap_seconds": 900, "clock_backend": "fabricated-clock",
              "clock_source_sha256": M.CLOCK_PIN, "watchdog_sha256": M.SUPERVISOR_PIN,
              "started_ns": 1000, "deadline_ns": 1000 + 900 * 10**9}
    json_file(args.supervision, launch)
    clock = SimpleNamespace(now_ns=lambda: 1001, backend="fabricated-clock")

    def loaded(path, _name):
        assert path == M.CLOCK
        return SimpleNamespace(SuspendClock=lambda: clock)

    monkeypatch.setattr(M, "load", loaded)
    monkeypatch.setattr(M, "runtime_record", lambda: runtime)
    monkeypatch.setattr(M, "authenticate", lambda *_args: {"sources": {}, "lineage": {}})
    worker = M.Run(args)
    monkeypatch.setattr(worker, "check", lambda **_kwargs: None)
    events = []
    monkeypatch.setattr(worker, "event", events.append)
    if bridge_finished > launch["started_ns"]:
        with pytest.raises(ValueError, match="bridge closes before evaluation"):
            worker.bind()
        assert not events and not (out / "started.json").exists()
    else:
        worker.bind()
        assert events == [{"event": "admitted", "phase": "evaluate", "before_numerical_imports": True}]
        assert worker.receipt["counts"]["array_decodes"] == 0


def test_algorithm_and_worker_numerical_methods_match_closed_source_ast():
    original = ast.parse((REPOSITORY / "scripts/run_otto_residual.py").read_text())
    repaired = ast.parse((REPOSITORY / "scripts/run_otto_residual_reanalysis.py").read_text())
    old_top = {node.name: node for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    new_top = {node.name: node for node in repaired.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for name in ("equal_nested", "validate_checkpoints", "expected_cache_work", "validate_counts", "report_view"):
        assert ast.dump(old_top[name], include_attributes=False) == ast.dump(new_top[name], include_attributes=False)
    old_methods = {node.name: node for node in old_top["Run"].body if isinstance(node, ast.FunctionDef)}
    new_methods = {node.name: node for node in new_top["Run"].body if isinstance(node, ast.FunctionDef)}
    assert set(old_methods) == set(new_methods)
    for name in set(old_methods) - {"bind"}:
        assert ast.dump(old_methods[name], include_attributes=False) == ast.dump(new_methods[name], include_attributes=False), name
    assert M.CONFIG["views"] == 72 and M.CONFIG["methods"] == 9 and M.CONFIG["confirm_admitted"] is False
    assert M.CONFIG["prior_confirmation_admitted"] is False and set(M.ROLES) == {"evaluate", "audit"}
    assert M.LIMITS == {"evaluate": {"seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 1024**3},
                        "audit": {"seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 256 * 1024**2}}
