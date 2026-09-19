"""Synthetic orchestration tests; renderer authentication is an injected boundary.

These tests do not certify the separate renderer, whose pure checks have their
own verification. They exercise exit/hash ordering, unchanged gates, exclusive
output and retained failure semantics without any production artifact reads.
"""

import copy
import hashlib
import importlib.util
import json
import os
import py_compile
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).with_name("verify_terminal.py")
    spec = importlib.util.spec_from_file_location("synthetic_terminal_verifier", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def args(v, tmp_path):
    return SimpleNamespace(plan=tmp_path / "inputs/protocol/plan.json", expected_plan_sha256=v.PLAN_SHA,
        execution=tmp_path / "inputs/execution", expected_completion_sha256="1" * 64, execution_exit_code=0,
        audit=tmp_path / "inputs/audit", expected_audit_receipt_sha256="2" * 64, audit_exit_code=0,
        out=tmp_path / "publication/terminal-verification.json")


def synthetic_boundary(v, tmp_path, monkeypatch, *, passed):
    a = args(v, tmp_path)
    monkeypatch.setattr(v, "ROOT", tmp_path)
    monkeypatch.setattr(v, "FALLBACK_FAILURE_ROOT", tmp_path / "fallback-failures")
    sources = {}
    for index in range(90):
        name = f"sources/{index:03d}.txt"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(str(index))
        sources[name] = v.sha(path)
    gate = {"passed": passed, "checks": [{"name": str(index), "passed": passed if index == 0 else True}
                                         for index in range(25)], "secondary_cannot_rescue_primary": True}
    summary = {"continuation_gate": gate, "costs": {"new_optimizer_steps": 0,
        "execution_wall_seconds": 10., "audit_validation_wall_seconds": 20.}, "new_fits": 0,
        "new_model_calls": 0, "native_control_transitions_checked": 163200,
        "native_nominal_candidate_transitions_checked": 78741504,
        "native_nominal_selected_transitions_checked": 28800,
        "public_observer_transitions_checked": 310464, "native_max_abs_error": 0.}
    summary_hash = save(a.audit / "summary.json", summary)
    completion_hash = save(a.execution / "completed.json", {"restored_models": 12, "control_rows": 51})
    receipt = {"execution_completed_sha256": completion_hash}
    a.expected_completion_sha256 = completion_hash
    a.expected_audit_receipt_sha256 = save(a.audit / "receipt.json", receipt)
    plan = {"sources": sources, "cap_seconds": 5700, "audit_cap_seconds": 3300}
    inputs = {"audit_summary_sha256": summary_hash, "execution_member_count": 9505}
    calls = []

    def checked(path, expected):
        path = Path(path)
        calls.append((str(path), expected))
        # The production plan SHA is deliberately fixed in the verifier. The
        # injected renderer has already authenticated a synthetic plan stand-in.
        if path == a.plan:
            v.require(expected == v.PLAN_SHA, "Synthetic plan pin")
            return path
        v.require(path.is_file() and v.sha(path) == expected, "Synthetic hash authentication")
        return path

    renderer = SimpleNamespace(authenticate=lambda _: (plan, receipt, copy.deepcopy(summary), inputs),
        checked=checked, read=lambda p: json.loads(Path(p).read_text()), member=lambda root, relative: root / relative)
    monkeypatch.setattr(v, "load_renderer", lambda: renderer)
    return a, renderer, summary, calls


@pytest.mark.parametrize("exit_codes", [(1, 0), (0, 7), (False, 0), (0, False)])
def test_nonzero_or_boolean_exit_rejected_before_renderer_load(tmp_path, monkeypatch, exit_codes):
    v = module()
    a = args(v, tmp_path)
    a.execution_exit_code, a.audit_exit_code = exit_codes
    monkeypatch.setattr(v, "load_renderer", lambda: pytest.fail("Artifact authentication started before exit guard"))
    with pytest.raises(ValueError, match="exactly zero"):
        v.verify(a)
    assert not a.out.exists()
    failures = list(a.out.parent.glob("terminal-verification.failed-*.json"))
    assert len(failures) == 1 and json.loads(failures[0].read_text())["status"] == "failed"


@pytest.mark.parametrize("passed", [True, False])
def test_completed_receipt_preserves_failed_or_passed_qualification(tmp_path, monkeypatch, passed):
    v = module()
    a, _, summary, calls = synthetic_boundary(v, tmp_path, monkeypatch, passed=passed)
    result = v.verify(a)
    assert result["status"] == "completed" and result["qualification_passed"] is passed
    assert result["checks_passed"] == (25 if passed else 24) and result["total_checks"] == 25
    assert result["continuation_gate"] == summary["continuation_gate"]
    assert result["gate_sha256"] == v.canonical_hash(summary["continuation_gate"])
    assert result["bound_sources_verified"] == 90 and result["restored_checkpoints"] == 12 and result["control_rows"] == 51
    assert result["native_nominal_selected_transitions_checked"] == 28800
    assert len([path for path, _ in calls if "/sources/" in path]) == 90
    assert json.loads(a.out.read_text()) == result
    assert not list(a.out.parent.glob("*.failed-*.json"))


def test_wrong_external_completion_rejected_before_renderer_authenticate(tmp_path, monkeypatch):
    v = module()
    a, renderer, _, _ = synthetic_boundary(v, tmp_path, monkeypatch, passed=False)
    a.expected_completion_sha256 = "0" * 64
    renderer.authenticate = lambda _: pytest.fail("Summary read before expected completion hash")
    with pytest.raises(ValueError, match="hash authentication"):
        v.verify(a)
    assert not a.out.exists()


def test_existing_terminal_never_overwritten_and_failures_are_separate(tmp_path, monkeypatch):
    v = module()
    a = args(v, tmp_path)
    a.out.parent.mkdir(parents=True)
    a.out.write_text("prior receipt")
    monkeypatch.setattr(v, "load_renderer", lambda: pytest.fail("Existing receipt guard failed"))
    for _ in range(2):
        with pytest.raises(ValueError, match="no overwrite"):
            v.verify(a)
    assert a.out.read_text() == "prior receipt"
    assert len(list(a.out.parent.glob("terminal-verification.failed-*.json"))) == 2


def test_changed_gate_file_rejected_without_receipt(tmp_path, monkeypatch):
    v = module()
    a, renderer, summary, _ = synthetic_boundary(v, tmp_path, monkeypatch, passed=False)
    original = renderer.authenticate

    def change(_):
        result = original(None)
        changed = copy.deepcopy(summary)
        changed["continuation_gate"]["passed"] = True
        save(a.audit / "summary.json", changed)
        return result

    renderer.authenticate = change
    with pytest.raises(ValueError, match="hash authentication"):
        v.verify(a)
    assert not a.out.exists()


def test_changed_bound_source_rejected_before_terminal_write(tmp_path, monkeypatch):
    v = module()
    a, renderer, _, _ = synthetic_boundary(v, tmp_path, monkeypatch, passed=True)
    original = renderer.authenticate

    def change(_):
        result = original(None)
        (tmp_path / "sources/089.txt").write_text("changed")
        return result

    renderer.authenticate = change
    with pytest.raises(ValueError, match="hash authentication"):
        v.verify(a)
    assert not a.out.exists()


def test_renderer_source_pin_rejects_unreviewed_module_without_import(tmp_path, monkeypatch):
    v = module()
    path = tmp_path / "unreviewed.py"
    path.write_text("raise AssertionError('must not import')")
    monkeypatch.setattr(v, "RENDERER", path)
    with pytest.raises(ValueError, match="reviewed renderer source"):
        v.load_renderer()


def test_loader_executes_authenticated_bytes_despite_valid_stale_bytecode(tmp_path, monkeypatch):
    v = module()
    path = tmp_path / "renderer.py"
    header = f"PLAN_SHA = {v.PLAN_SHA!r}\nSTUDY = {v.STUDY!r}\n"
    stale = header + "MARKER = 'stale'\n"
    fresh = header + "MARKER = 'fresh'\n"
    assert len(stale) == len(fresh)
    path.write_text(stale)
    timestamp = path.stat()
    py_compile.compile(str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    path.write_text(fresh)
    os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    # Demonstrate that a normal timestamp-cache import would use the stale code.
    spec = importlib.util.spec_from_file_location("stale_cache_control", path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    assert control.MARKER == "stale"
    monkeypatch.setattr(v, "RENDERER", path)
    monkeypatch.setattr(v, "RENDERER_SHA", v.sha(path))
    assert v.load_renderer().MARKER == "fresh"
