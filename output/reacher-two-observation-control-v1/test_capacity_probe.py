"""Byte-only capacity driver checks. No Torch, NumPy, model or native calls."""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("two_observation_capacity_probe", Path(__file__).with_name("capacity_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def rehearsal(tmp_path, monkeypatch):
    module = types.ModuleType("openjev.research.reacher_two_observation_experiment")
    module.SCHEMA = "fixture"
    module.SOURCE_PATHS_NEW = ("sources/000.py",)
    module.runtime = lambda: {"runtime": "synthetic"}
    module.expected_members = lambda settings: {"one.bin", "two.bin"}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import openjev.research
    monkeypatch.setattr(openjev.research, "reacher_two_observation_experiment", module, raising=False)
    source = {}
    for i in range(116):
        path = tmp_path / f"sources/{i:03d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(i).encode())
        source[path.relative_to(tmp_path).as_posix()] = probe.sha(path)
    for name in (probe.HELPER, probe.TEST_HELPER):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic helper bytes")
    parent = tmp_path / "rehearsal"
    parent.mkdir()
    execution = parent / "execution"
    execution.mkdir()
    for name in module.expected_members({}):
        (execution / name).write_bytes(name.encode())
    members = {name: probe.sha(execution / name) for name in module.expected_members({})}
    plan = {"schema": "fixture", "engineering": True, "sources": source, "settings": {},
            "runtime": module.runtime(), "historical_registries": {"engineering_sources": {}}}
    probe.write(parent / "plan.json", plan)
    done = {"status": "completed", "plan_sha256": probe.sha(parent / "plan.json"), "files": members}
    probe.write(execution / "completed.json", done)
    summary = {"saved_output_only": True, "execution_new_fits": 3, "coverage": {"control_rows": 42, "evaluated_models": 9},
               "native_max_abs_error": 0, "public_observer_max_abs_error": 0, "new_model_calls": 0, "new_optimizer_steps": 0}
    probe.write(parent / "audit/summary.json", summary)
    (parent / "audit/README.md").write_text("fake saved audit")
    audit = {"status": "completed", "engineering": True, "saved_output_only": True,
             "plan_sha256": done["plan_sha256"], "execution_completed_sha256": probe.sha(execution / "completed.json"),
             "execution_members": members, "source_sha256": source, "runtime": module.runtime(),
             "files": {name: probe.sha(parent / "audit" / name) for name in ("summary.json", "README.md")}}
    probe.write(parent / "audit/receipt.json", audit)
    qual = {"schema": "reacher-two-observation-rehearsal-qualification-v1", "status": "completed", "engineering": True,
            "saved_output_audit_completed": True, "scientific_draws": 0, "native_replay_max_abs_error": 0,
            "new_fits": 3, "inherited_models": 6, "restored_final_models": 9, "control_rows": 42,
            "audit_receipt_sha256": probe.sha(parent / "audit/receipt.json"), "plan_sha256": done["plan_sha256"],
            "execution_completed_sha256": audit["execution_completed_sha256"], "source_sha256": source,
            "runtime": module.runtime(), "files": {name: value["sha256"] for name, value in probe.inventory(parent, float("inf")).items()},
            "engineering_gate": {"passed": False, "checks_passed": 0}}
    probe.write(parent / "qualification.json", qual)
    return {"plan_path": "rehearsal/plan.json", "audit_path": "rehearsal/audit/receipt.json",
            "audit_sha256": probe.sha(parent / "audit/receipt.json"), "execution_path": "rehearsal/execution",
            "qualification_path": "rehearsal/qualification.json", "qualification_sha256": probe.sha(parent / "qualification.json")}


def fake_backend(monkeypatch):
    called = []
    def measure(root, out, binding, deadline, progress):
        called.append("measure")
        probe.write(out / "data.json", {"fixture": True})
        return {"rows": probe.row_manifest()}
    monkeypatch.setattr(probe, "_measure", measure)
    return called


def test_import_and_fixed_scope():
    assert len(probe.row_manifest()) == 14
    assert sum("arm" in row for row in probe.row_manifest()) == 9
    assert {row["reference"] for row in probe.row_manifest() if "reference" in row} == set(probe.REFERENCES)
    assert all(row["panel"] == "shift" for row in probe.row_manifest() if "reference" in row)
    assert not any(name in vars(probe) for name in ("torch", "numpy", "runner"))


def test_qualified_failed_tiny_gate_still_admits_engineering(tmp_path, monkeypatch):
    ref = rehearsal(tmp_path, monkeypatch)
    got = probe.bind_rehearsal(tmp_path, ref, float("inf"))
    assert len(got["scientific_sources"]) == 116
    assert len(got["sources"]) == 118
    assert got["reference"] == ref


@pytest.mark.parametrize("change", ["missing", "corrupt", "source", "extra", "failed_qualification", "wrong_runtime"])
def test_rehearsal_fail_closed_before_backend(tmp_path, monkeypatch, change):
    ref = rehearsal(tmp_path, monkeypatch)
    if change == "missing":
        (tmp_path / "rehearsal/execution/one.bin").unlink()
    elif change == "corrupt":
        (tmp_path / "rehearsal/execution/one.bin").write_bytes(b"changed")
    elif change == "source":
        (tmp_path / "sources/115.py").write_bytes(b"changed")
    elif change == "extra":
        (tmp_path / "rehearsal/execution/failed.json").write_bytes(b"{}")
    elif change == "failed_qualification":
        path = tmp_path / ref["qualification_path"]
        body = probe.read(path); body["status"] = "failed"; path.unlink(); probe.write(path, body)
        ref["qualification_sha256"] = probe.sha(path)
    else:
        sys.modules["openjev.research.reacher_two_observation_experiment"].runtime = lambda: {"runtime": "other"}
    called = fake_backend(monkeypatch)
    attempt = tmp_path / "attempt"
    with pytest.raises(ValueError):
        probe.run_profile(attempt, root=tmp_path, rehearsal=ref, cap_seconds=100, audit_cap_seconds=100, execute_engineering=True)
    assert not called
    assert probe.read(attempt / "failed.json")["automatic_retry"] is False
    assert not (attempt / "completed.json").exists()


@pytest.mark.parametrize("phase", ["run", "audit"])
def test_authorization_precedes_any_artifact_access(tmp_path, phase):
    path = tmp_path / "absent"
    with pytest.raises(ValueError, match="authorization"):
        if phase == "run":
            probe.run_profile(path, root=tmp_path, rehearsal={}, cap_seconds=1, audit_cap_seconds=1)
        else:
            probe.audit_profile(path, "f" * 64, root=tmp_path)
    assert not path.exists() and not path.with_name("absent-audit").exists()


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, float("nan"), None])
def test_bad_phase_cap_retained_without_admission(tmp_path, monkeypatch, cap):
    monkeypatch.setattr(probe, "bind_rehearsal", lambda *args: pytest.fail("Admission must not run"))
    path = tmp_path / "attempt"
    with pytest.raises(ValueError, match="phase caps"):
        probe.run_profile(path, root=tmp_path, rehearsal={}, cap_seconds=cap, audit_cap_seconds=2, execute_engineering=True)
    assert (path / "failed.json").exists()


def test_exact_inventory_and_exclusive_attempt(tmp_path, monkeypatch):
    ref = rehearsal(tmp_path, monkeypatch); called = fake_backend(monkeypatch)
    path = tmp_path / "attempt"
    probe.run_profile(path, root=tmp_path, rehearsal=ref, cap_seconds=100, audit_cap_seconds=100, execute_engineering=True)
    done = probe.read(path / "completed.json")
    probe.verify_members(path, done["files"], float("inf"), extra=("completed.json",))
    assert called == ["measure"] and done["qualification_ready"] is False
    assert len([name for name in done["files"] if name.startswith("source-snapshot/")]) == 118
    with pytest.raises(FileExistsError):
        probe.run_profile(path, root=tmp_path, rehearsal=ref, cap_seconds=100, audit_cap_seconds=100, execute_engineering=True)
    assert called == ["measure"]
    (path / "execution/data.json").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="mismatch"):
        probe.verify_members(path, done["files"], float("inf"), extra=("completed.json",))


def test_failure_after_completion_invalidates_receipt(tmp_path, monkeypatch):
    ref = rehearsal(tmp_path, monkeypatch); fake_backend(monkeypatch)
    path = tmp_path / "attempt"
    original = probe.check
    def check(deadline):
        if (path / "completed.json").exists():
            raise TimeoutError("post-write cap")
        original(deadline)
    monkeypatch.setattr(probe, "check", check)
    with pytest.raises(TimeoutError, match="post-write"):
        probe.run_profile(path, root=tmp_path, rehearsal=ref, cap_seconds=100, audit_cap_seconds=100, execute_engineering=True)
    assert (path / "invalid-completion.json").exists() and (path / "failed.json").exists()
    assert not (path / "completed.json").exists()


def projection_inputs():
    measured = {"rows": [{**row, "whole_call_seconds": 10.} for row in probe.row_manifest()],
                "actual_training_updates": 24, "phases": {"full48_trainer_setup_seconds": 5., "fit_seconds": 50.}}
    audited = {"rows": [{**row, "audit_seconds": 2.} for row in probe.row_manifest()],
               "training_audit_seconds": 1., "wall_seconds_before_report": 34.}
    fit = {"updates": 24, "wall_seconds": 50., "constructor_seconds": 5., "update_call_seconds": 40., "log_validation_flush_seconds": 1.}
    return measured, audited, fit, {"wall_seconds": 210.}


def test_projection_hand_computable_and_never_qualified():
    values = projection_inputs(); original = copy.deepcopy(values)
    result = probe.projection(*values)
    assert result["execution_components_seconds"] == {"training": 41 * 144 + 3 * 9,
           "learned_control": 270., "references": 150., "measured_outer_overhead_proxy": 15.}
    assert result["nominal_projected_seconds"] == {"execution": 6366., "audit": 233.}
    assert result["target"]["training_updates"] == 3456 and result["target"]["control_rows"] == 42
    assert result["qualification_ready"] is False and "pending" in result["status"]
    assert "cap_seconds" not in result and len(result["unmeasured_or_nonbinding"]) == 6
    assert values == original


@pytest.mark.parametrize("change", ["row_missing", "row_duplicate", "row_order", "negative", "nan", "updates", "nested_fit", "outer_wall"])
def test_projection_rejects_missing_or_impossible_measurements(change):
    values = projection_inputs(); measured, audited, fit, done = values
    if change == "row_missing": measured["rows"].pop()
    elif change == "row_duplicate": measured["rows"][-1] = measured["rows"][-2]
    elif change == "row_order": audited["rows"].reverse()
    elif change == "negative": measured["rows"][0]["whole_call_seconds"] = -1.
    elif change == "nan": audited["training_audit_seconds"] = float("nan")
    elif change == "updates": fit["updates"] = 23
    elif change == "nested_fit": fit["wall_seconds"] = 1.
    else: done["wall_seconds"] = 1.
    with pytest.raises(ValueError): probe.projection(*values)


def test_audit_corruption_preserves_separate_failure(tmp_path, monkeypatch):
    ref = rehearsal(tmp_path, monkeypatch); fake_backend(monkeypatch)
    path = tmp_path / "attempt"
    probe.run_profile(path, root=tmp_path, rehearsal=ref, cap_seconds=100, audit_cap_seconds=100, execute_engineering=True)
    digest = probe.sha(path / "completed.json")
    (path / "execution/data.json").write_bytes(b"changed")
    monkeypatch.setattr(probe, "_audit_measurements", lambda *args: pytest.fail("No replay on corrupt inputs"))
    with pytest.raises(ValueError, match="mismatch"):
        probe.audit_profile(path, digest, root=tmp_path, execute_engineering=True)
    assert (tmp_path / "attempt-audit/failed.json").exists()
    assert probe.sha(path / "completed.json") == digest
    with pytest.raises(FileExistsError): probe.audit_profile(path, digest, root=tmp_path, execute_engineering=True)


@pytest.mark.parametrize("relative", ["../x", "/tmp/x", "a//x", "./x", "a\\x"])
def test_unsafe_member_paths_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="Safe relative"):
        probe.safe(tmp_path, relative)


@pytest.mark.parametrize("post_completion_cap", [False, True])
def test_complete_fake_audit_keeps_template_unqualified(tmp_path, monkeypatch, post_completion_cap):
    from openjev.research import reacher_two_observation_protocol as protocol
    ref = rehearsal(tmp_path, monkeypatch)
    def measure(root, out, binding, deadline, progress):
        measured, _, fit, _ = projection_inputs()
        for row in measured["rows"]: row["whole_call_seconds"] *= 1e-8
        measured["phases"] = {key: value * 1e-8 for key, value in measured["phases"].items()}
        for key in fit:
            if key.endswith("seconds"): fit[key] *= 1e-8
        probe.write(out / "measurement.json", measured)
        probe.write(out / "fake-fit.json", fit)
        return {"rows": measured["rows"]}
    def audit(root, execution, out, binding, deadline, progress):
        settings = protocol.settings(engineering=True)
        settings["rng_namespace"] = probe.NAMESPACE
        probe.write(out / "training-audit.json", {"fixture": True})
        return {"rows": [{**row, "audit_seconds": 1e-8} for row in probe.row_manifest()],
                "training_audit_seconds": 1e-8, "fit": probe.read(execution / "fake-fit.json"),
                "settings": settings, "new_model_calls": 0, "new_optimizer_steps": 0}
    monkeypatch.setattr(probe, "_measure", measure)
    monkeypatch.setattr(probe, "_audit_measurements", audit)
    path = tmp_path / "attempt"
    probe.run_profile(path, root=tmp_path, rehearsal=ref, cap_seconds=100, audit_cap_seconds=100, execute_engineering=True)
    digest = probe.sha(path / "completed.json")
    out = tmp_path / "attempt-audit"
    if post_completion_cap:
        check = probe.check
        def capped(deadline):
            if (out / "completed.json").exists(): raise TimeoutError("final audit cap")
            check(deadline)
        monkeypatch.setattr(probe, "check", capped)
        with pytest.raises(TimeoutError, match="final audit cap"):
            probe.audit_profile(path, digest, root=tmp_path, execute_engineering=True)
        assert (out / "invalid-completion.json").exists() and (out / "failed.json").exists()
    else:
        assert probe.audit_profile(path, digest, root=tmp_path, execute_engineering=True) == out
        done = probe.read(out / "completed.json")
        probe.verify_members(out, done["files"], float("inf"), extra=("completed.json",))
        template = probe.read(out / "qualification-template.json")
        assert template["status"] == "pending_independent_sizing_review"
        assert template["qualification_ready"] is False and template["review_sha256"] is None
        assert set(template["measured_components"]) == {"training", "learned_control", "physics_references", "audit", "storage_and_hashing"}
        assert template["coverage"]["control_rows"] == 42
        for item in template["measured_components"].values():
            assert item["artifact"] in template["files"]
        assert done["new_model_calls"] == done["new_optimizer_steps"] == 0
    assert probe.sha(path / "completed.json") == digest
