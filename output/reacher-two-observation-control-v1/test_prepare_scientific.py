"""Fake readiness/preparation and real tiny Git objects; no scored allocations."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("two_observation_prepare_scientific", Path(__file__).with_name("prepare_scientific.py"))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_bound(root, relative, digest):
    path = Path(root) / relative
    prep.require(path.is_file() and prep.sha(path) == digest, "Bound fixture bytes")
    return json.loads(path.read_text())


def git_fixture(root):
    prep.git(root, "init", "-q")
    sources = {}
    for i in range(116):
        path = root / f"src/file{i:03d}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"fixture = {i}\n")
        sources[path.relative_to(root).as_posix()] = prep.sha(path)
    prep.git(root, "add", "--", "src")
    prep.git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Synthetic source closure")
    return sources, prep.git(root, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    sources, commit = git_fixture(tmp_path)
    events, runtime = [], {"runtime": "fake"}
    for name in (prep.HELPER, prep.TEST_HELPER, prep.CAPACITY_HELPER, prep.CAPACITY_TESTS, "engineering/old.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    profile = sources | {name: prep.sha(tmp_path / name) for name in (prep.CAPACITY_HELPER, prep.CAPACITY_TESTS, "engineering/old.py")}
    rehearsal = {"plan_sha256": "b" * 64, "source_sha256": sources, "runtime": runtime}
    history = {"numpy": [{"historical": 7}], "torch": [{"historical": 9}], "descriptors": [{"prior": "fixture"}],
               "literal_calls": [{"source_path": "src/file000.py", "numpy_registry": {}, "torch_registry": {"historical": 410}}],
               "engineering_sources": {"engineering/old.py": profile["engineering/old.py"]}}
    measurement = {"whole_measurement": {"status": "completed", "engineering": True,
        "namespace": "reacher-two-observation-engineering-capacity-v1", "rehearsal": {"qualification_sha256": prep.REHEARSAL["sha256"]},
        "source_sha256": sources, "runtime": runtime, "new_fits": 1, "optimizer_updates": 24, "row_count": 14,
        "profile_source_sha256": profile}, "new_model_calls": 0, "new_optimizer_steps": 0}
    prep.write(tmp_path / "capacity/measurement-audit.json", measurement)
    capacity_body = {"status": "completed", "source_sha256": sources, "runtime": runtime,
        "files": {"measurement-audit.json": prep.sha(tmp_path / "capacity/measurement-audit.json")},
        "projected_seconds": {"execution": 100, "audit": 100}}
    prep.write(tmp_path / "capacity/qualification.json", capacity_body)
    capacity = {"path": "capacity/qualification.json", "sha256": prep.sha(tmp_path / "capacity/qualification.json")}
    def bound(root, path, digest):
        if path == prep.lineage_refs()["prerequisite"]["plan_path"]:
            return {"sources": sources}
        if path == prep.REHEARSAL_PLAN:
            return {"engineering": True, "sources": sources, "historical_registries": copy.deepcopy(history)}
        return read_bound(root, path, digest)
    def checked(root, path, digest):
        full = Path(root) / path
        prep.require(full.is_file() and prep.sha(full) == digest, "Bound source fixture")
        return full
    def readiness(root, refs, settings, supplied_sources, observed, execution_cap, audit_cap, engineering):
        events.append("readiness")
        body = read_bound(root, refs["capacity"]["path"], refs["capacity"]["sha256"])
        prep.require(body["status"] == "completed", "Completed readiness")
        prep.require(body["source_sha256"] == supplied_sources and body["runtime"] == observed, "Readiness source/runtime")
        prep.require(execution_cap > body["projected_seconds"]["execution"] and audit_cap > body["projected_seconds"]["audit"], "Reviewed caps exceed projections")
        for name, digest in body["files"].items():
            checked(root, str(Path(refs["capacity"]["path"]).parent / name), digest)
        return {"capacity": {**refs["capacity"], "receipt": body}, "rehearsal": {**refs["rehearsal"], "receipt": rehearsal}}
    def prepare(root, settings, **kwargs):
        events.append("prepare_manifest")
        assert kwargs["engineering"] is False and kwargs["lineage_refs"] == prep.lineage_refs()
        return {"sources": sources, "engineering": False, "content_sha256": "c" * 64,
                "engineering_evidence": kwargs["engineering_evidence"]}
    def validate(plan, expected, **kwargs):
        events.append("validate_freeze")
        ref = kwargs["freeze_ref"]
        freeze = read_bound(kwargs["root"], ref["path"], ref["sha256"])
        assert freeze["plan_sha256"] == expected and freeze["no_retry"] is True
        return SimpleNamespace(execution_authorized=True, engineering=False)
    def torch_manifest(roles):
        events.append("historical_literal_identity")
        assert roles == {"synthetic_constructors_and_orders": 410}
        return {"fake": "manifest, no generator"}
    experiment = SimpleNamespace(runtime=lambda: runtime, read_bound=bound, _sources=lambda *args: sources,
        _engineering_evidence=readiness, checked=checked, sha=prep.sha, _path=lambda root, name: Path(root) / name,
        prepare=prepare, validate_plan=validate, identity=identity)
    protocol = SimpleNamespace(settings=lambda: {"namespace": "symbolic_only"}, STUDY="reacher-two-observation-study-v1")
    streams = SimpleNamespace(torch_generator_manifest=torch_manifest)
    monkeypatch.setattr(prep, "dependencies", lambda: (experiment, protocol, streams))
    return SimpleNamespace(root=tmp_path, sources=sources, commit=commit, events=events, experiment=experiment,
                           capacity=capacity, history=history, capacity_body=capacity_body, measurement=measurement)


def call(case, **extra):
    return prep.prepare(case.root / "preparation", root=case.root, capacity=case.capacity, source_commit=case.commit,
                        cap_seconds=200, audit_cap_seconds=200, authorize_preparation_and_freeze=True, **extra)


def test_complete_fake_preparation_and_separate_freeze(fixture):
    result = call(fixture)
    folder = fixture.root / "preparation"
    assert fixture.events == ["readiness", "historical_literal_identity", "prepare_manifest", "validate_freeze"]
    done = json.loads((folder / "completed.json").read_text())
    assert done["status"] == "frozen_not_run" and done["scientific_execution_started"] is False
    assert done["scientific_samples"] == done["new_optimizer_steps"] == done["new_native_calls"] == 0
    assert result["sha256"] == prep.sha(folder / "plan.json")
    assert result["freeze"]["sha256"] == prep.sha(folder / "freeze.json")
    assert not (folder / "freeze-candidate.json").exists()
    request = json.loads((folder / "request.json").read_text())
    got = request["historical_registries"]
    for name in ("numpy", "torch", "descriptors"):
        assert got[name] == fixture.history[name]
    assert got["literal_calls"][-1]["torch_registry"] == {"synthetic_constructors_and_orders": 410}
    assert got["literal_calls"][-1]["source_sha256"] == prep.sha(fixture.root / prep.CAPACITY_HELPER)
    assert len([name for name in done["files"] if name.startswith("source-snapshot/")]) == 121
    for name, digest in done["files"].items(): assert prep.sha(folder / name) == digest
    with pytest.raises(FileExistsError): call(fixture)


def test_authorization_before_any_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "dependencies", lambda: pytest.fail("No imports or readiness reads"))
    with pytest.raises(ValueError, match="authorization"):
        prep.prepare(tmp_path / "out", root=tmp_path, capacity={}, source_commit="a" * 40, cap_seconds=200, audit_cap_seconds=200)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["missing", "bad_hash", "pending", "changed_source", "caps", "real_commit_missing", "commit_wrong_bytes", "capacity_helper_drift"])
def test_fail_before_scored_allocation(fixture, kind):
    if kind == "missing": (fixture.root / fixture.capacity["path"]).unlink()
    elif kind == "bad_hash": fixture.capacity["sha256"] = "f" * 64
    elif kind in ("pending", "changed_source", "caps"):
        body = fixture.capacity_body
        if kind == "pending": body["status"] = "pending_review"
        elif kind == "changed_source": body["source_sha256"] = {}
        else: body["projected_seconds"]["execution"] = 200
        path = fixture.root / fixture.capacity["path"]
        path.unlink(); prep.write(path, body); fixture.capacity["sha256"] = prep.sha(path)
    elif kind == "real_commit_missing": fixture.commit = "f" * 40
    elif kind == "commit_wrong_bytes":
        path = fixture.root / "src/file115.py"; path.write_text("changed\n")
        fixture.sources["src/file115.py"] = prep.sha(path)
    else: (fixture.root / prep.CAPACITY_HELPER).write_text("changed\n")
    with pytest.raises(ValueError): call(fixture)
    assert "prepare_manifest" not in fixture.events
    assert not (fixture.root / "preparation/plan.json").exists()
    assert json.loads((fixture.root / "preparation/failed.json").read_text())["automatic_retry"] is False


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, float("inf"), None])
def test_invalid_caps_retained_before_imports(tmp_path, monkeypatch, cap):
    monkeypatch.setattr(prep, "dependencies", lambda: pytest.fail("No readiness imports"))
    with pytest.raises(ValueError, match="reviewed phase caps"):
        prep.prepare(tmp_path / "out", root=tmp_path, capacity={}, source_commit="a" * 40,
                     cap_seconds=cap, audit_cap_seconds=200, authorize_preparation_and_freeze=True)
    assert (tmp_path / "out/failed.json").exists()


def test_validation_failure_invalidates_freeze_candidate(fixture):
    def reject(*args, **kwargs): raise ValueError("Rejected final freeze")
    fixture.experiment.validate_plan = reject
    with pytest.raises(ValueError, match="Rejected final freeze"): call(fixture)
    folder = fixture.root / "preparation"
    assert (folder / "plan.json").exists() and (folder / "invalid-freeze-candidate.json").exists()
    assert (folder / "failed.json").exists() and not (folder / "freeze.json").exists()
    assert not (folder / "completed.json").exists()


def test_real_commit_blob_proof_and_dirty_bytes(tmp_path):
    sources, commit = git_fixture(tmp_path)
    proof = prep.verify_source_commit(tmp_path, sources, commit)
    assert proof["verified_source_files"] == 116 and len(proof["git_blob_ids"]) == 116
    (tmp_path / "src/file010.py").write_text("edited\n")
    with pytest.raises(ValueError, match="Current source differs"):
        prep.verify_source_commit(tmp_path, sources, commit)
    sources["src/file010.py"] = prep.sha(tmp_path / "src/file010.py")
    with pytest.raises(ValueError, match="Commit source bytes differ"):
        prep.verify_source_commit(tmp_path, sources, commit)


def test_capacity_measurement_ambiguity_rejected(fixture):
    fixture.capacity_body["files"]["another/measurement-audit.json"] = "d" * 64
    with pytest.raises(ValueError, match="Exactly one"):
        prep.capacity_source_binding(fixture.experiment, fixture.root, fixture.capacity, fixture.capacity_body)


def test_actual_delayed_dependencies_import_without_preparing():
    experiment, protocol, streams = prep.dependencies()
    assert experiment.__name__ == "openjev.research.reacher_two_observation_experiment"
    assert protocol.__name__ == "openjev.research.reacher_two_observation_protocol"
    assert streams.__name__ == "openjev.research.reacher_two_observation_streams"
    assert callable(experiment.prepare) and callable(experiment.validate_plan)


@pytest.mark.parametrize("broken_step", ["rename", "failed_receipt"])
def test_preservation_error_cannot_replace_original(fixture, monkeypatch, broken_step):
    def reject(*args, **kwargs): raise ValueError("Original validation error")
    fixture.experiment.validate_plan = reject
    if broken_step == "rename":
        original = Path.rename
        def rename(path, target):
            if path.name == "freeze-candidate.json": raise OSError("Synthetic demotion failure")
            return original(path, target)
        monkeypatch.setattr(Path, "rename", rename)
    else:
        original = prep.write
        def write(path, value):
            if path.name == "failed.json": raise OSError("Synthetic receipt failure")
            return original(path, value)
        monkeypatch.setattr(prep, "write", write)
    with pytest.raises(ValueError, match="Original validation error") as caught:
        call(fixture)
    assert len(caught.value.__notes__) == 1
    assert "Could not" in caught.value.__notes__[0]
    assert not (fixture.root / "preparation/completed.json").exists()
    if broken_step == "rename":
        failed = json.loads((fixture.root / "preparation/failed.json").read_text())
        assert "freeze-candidate" in failed["notes"][0]
    else:
        assert (fixture.root / "preparation/invalid-freeze-candidate.json").exists()
