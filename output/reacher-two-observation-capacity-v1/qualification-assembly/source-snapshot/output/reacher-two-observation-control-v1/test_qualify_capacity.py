"""Focused saved-byte and sizing-contract tests; no numerical work or assembly."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("qualify_capacity_under_test", Path(__file__).with_name("qualify_capacity.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


def sizing():
    review = {"status": "reviewed", "measurement_completed_sha256": "a" * 64,
              "repaired_audit_completed_sha256": "b" * 64, "source_sha256": {"fixture": "c" * 64},
              "runtime": {"fixture": True}, "projected_seconds": {"execution": 100., "audit": 50.},
              "approved_caps": {"execution": 120, "audit": 70}}
    kwargs = {"measurement_sha": "a" * 64, "audit_sha": "b" * 64, "sources": review["source_sha256"],
              "runtime": review["runtime"], "caps": dict(review["approved_caps"])}
    return review, kwargs


def test_reviewed_values_are_copied_not_invented():
    review, kwargs = sizing(); before = copy.deepcopy(review)
    result = qualify.review_sizing(review, **kwargs)
    assert result == {"execution": 100., "audit": 50.}
    result["audit"] = 1.
    assert review == before


@pytest.mark.parametrize("bad", ["pending", "audit_sha", "caps", "boundary", "nan", "bool"])
def test_external_review_cannot_be_bypassed(bad):
    review, kwargs = sizing()
    if bad == "pending": review["status"] = "draft"
    elif bad == "audit_sha": review["repaired_audit_completed_sha256"] = "f" * 64
    elif bad == "caps": kwargs["caps"]["audit"] = 71
    elif bad == "boundary": review["projected_seconds"]["audit"] = 70
    elif bad == "nan": review["projected_seconds"]["execution"] = float("nan")
    else: review["projected_seconds"]["audit"] = True
    with pytest.raises(ValueError): qualify.review_sizing(review, **kwargs)


def test_completed_tree_missing_corrupt_and_extra_members(tmp_path):
    base = qualify.original(qualify.ROOT)
    (tmp_path / "data.bin").write_bytes(b"original")
    receipt = {"files": base.inventory(tmp_path, float("inf"))}
    (tmp_path / "completed.json").write_text("{}")
    assert set(qualify.checked_tree(base, tmp_path, receipt, "completed.json")) == {"data.bin", "completed.json"}
    (tmp_path / "data.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksums"):
        qualify.checked_tree(base, tmp_path, receipt, "completed.json")
    (tmp_path / "data.bin").unlink()
    with pytest.raises(ValueError, match="membership"):
        qualify.checked_tree(base, tmp_path, receipt, "completed.json")
    (tmp_path / "data.bin").write_bytes(b"original")
    (tmp_path / "unmentioned.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="membership"):
        qualify.checked_tree(base, tmp_path, receipt, "completed.json")


def test_component_paths_remain_within_qualified_members(tmp_path):
    labels = ("training", "learned_control", "physics_references", "audit", "storage_and_hashing")
    raw = {name: {"wall_seconds": 1., "units": 1,
        "artifact": "training-audit.json" if name == "training" else "measurement-audit.json"} for name in labels}
    result = qualify.copy_components(raw, tmp_path / "repaired", tmp_path)
    assert result["training"]["artifact"] == "repaired/training-audit.json"
    assert all(result[name]["artifact"] == "repaired/measurement-audit.json" for name in labels[1:])
    raw["audit"]["artifact"] = "../measurement-audit.json"
    with pytest.raises(ValueError, match="Exact existing"):
        qualify.copy_components(raw, tmp_path / "repaired", tmp_path)


def test_no_authorization_no_original_or_receipt_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(qualify, "original", lambda *args: pytest.fail("No artifact reads"))
    with pytest.raises(ValueError, match="authorization"):
        qualify.assemble({}, tmp_path / "qualification.json", root=tmp_path)
    assert not (tmp_path / "qualification-assembly").exists()


def test_malformed_request_failure_retained_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(qualify, "original", lambda *args: pytest.fail("No receipt reads"))
    with pytest.raises(ValueError, match="Exact explicit"):
        qualify.assemble({}, tmp_path / "qualification.json", root=tmp_path, authorize_saved_qualification=True)
    assert (tmp_path / "qualification-assembly/failed.json").exists()
    assert not (tmp_path / "qualification.json").exists()
    with pytest.raises(FileExistsError):
        qualify.assemble({}, tmp_path / "qualification.json", root=tmp_path, authorize_saved_qualification=True)


def process_fixture(tmp_path):
    base = qualify.original(qualify.ROOT)
    references, locations = {}, {name: tmp_path / name for name in qualify.WRAPPERS}
    old_helper = "d" * 64
    repaired = {"wall_seconds": 4., "audit_repair": {"repair_source_sha256": {qualify.CONTROL + "capacity_audit_repair.py": "e" * 64}}}
    for kind, name in qualify.WRAPPERS.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fake wrapper\n")
        folder = tmp_path / (kind + "-process"); folder.mkdir()
        started = {"driver_sha256": base.sha(path), "helper_sha256": old_helper if kind == "attempt01" else qualify.ORIGINAL_SHA}
        if kind == "repair":
            command = ["python", str(tmp_path / (qualify.CONTROL + "capacity_audit_repair.py")), str(locations["attempt02"]),
                       str(locations["repair"]), "--completed-sha256", qualify.MEASUREMENT_SHA]
            phase = {"phase": "saved_audit_repair", "exit_code": 0, "wall_seconds": 5., "command": command, "automatic_retry": False}
            started.update(command=command, cwd=str(tmp_path), helper_sha256="e" * 64)
            terminal = {"status": "completed", **phase}
            base.write(folder / "process.json", phase)
            (folder / "as-run-driver.py").write_bytes(path.read_bytes())
            for name in ("audit.stdout", "audit.stderr"): (folder / name).write_bytes(b"")
        else:
            phases = []
            for label, code in ([("run", 1)] if kind == "attempt01" else [("run", 0), ("audit", 1)]):
                command = ["python", str(tmp_path / qualify.ORIGINAL), label, str(locations[kind])]
                if label == "audit": command += ["--completed-sha256", qualify.MEASUREMENT_SHA]
                phase = {"phase": label, "exit_code": code, "wall_seconds": 5., "command": command}
                base.write(folder / f"{label}-process.json", phase)
                base.write(folder / f"{label}-invocation.json", {"command": command, "cwd": str(tmp_path)})
                for suffix in ("stdout", "stderr"): (folder / f"{label}.{suffix}").write_bytes(b"")
                phases.append(phase)
            terminal = {"status": "failed", "phase": phases[-1]["phase"], "phases": phases,
                        "wall_seconds": 1. + sum(item["wall_seconds"] for item in phases), "automatic_retry": False}
        base.write(folder / "started.json", started)
        terminal_path = folder / ("completed.json" if kind == "repair" else "failed.json")
        base.write(terminal_path, terminal)
        references[kind] = {"path": terminal_path.relative_to(tmp_path).as_posix(), "sha256": base.sha(terminal_path)}
    return base, references, {"wall_seconds": 4.}, repaired, {"sources": {qualify.ORIGINAL: old_helper}}, {"wall_seconds": 4.}, locations


def test_actual_process_shapes_all_failures_and_success_retained(tmp_path):
    base, refs, measured, repaired, failed01, failed02, locations = process_fixture(tmp_path)
    rows, wrappers = qualify.process_receipts(base, tmp_path, refs, measured, repaired, failed01, failed02, locations=locations)
    assert [p["exit_code"] for p in rows["attempt01"]["phases"]] == [1]
    assert [p["exit_code"] for p in rows["attempt02"]["phases"]] == [0, 1]
    assert [p["exit_code"] for p in rows["repair"]["phases"]] == [0]
    assert set(wrappers) == set(qualify.WRAPPERS.values())
    (tmp_path / "repair-process/audit.stderr").unlink()
    with pytest.raises(ValueError, match="output streams retained"):
        qualify.process_receipts(base, tmp_path, refs, measured, repaired, failed01, failed02, locations=locations)


def test_process_command_cannot_point_at_another_measurement(tmp_path):
    base, refs, measured, repaired, failed01, failed02, locations = process_fixture(tmp_path)
    locations["attempt02"] = tmp_path / "other-measurement"
    with pytest.raises(ValueError, match="artifact identity"):
        qualify.process_receipts(base, tmp_path, refs, measured, repaired, failed01, failed02, locations=locations)
