"""Temporary constructed-byte tests only; never reads scientific output trees.

The small orchestration tests replace scientific authenticators with explicit
fixture stubs. They validate packaging state transitions, not scientific lineage.
The report/prior-release tests exercise those real validators on synthetic bytes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).with_name("package.py")
    spec = importlib.util.spec_from_file_location("synthetic_geometry_memory_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, payload=b"synthetic fixture only\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def gate(helper, *, passed=False):
    names = []
    for comparator in ("encoded_current_gru", "cached_gru"):
        for panel in ("ordinary", "shift"):
            names.append(f"gap_mean/{panel}/{comparator}")
            names.extend(f"gap_pair/{panel}/{comparator}/{pair}" for pair in helper.PAIRS)
        names.append(f"full_mean/{comparator}")
    names.extend(f"competence/{panel}/residual_gru-{pair}" for panel in ("ordinary", "shift") for pair in helper.PAIRS)
    names.append("competence/ordinary/known_state")
    return {"passed": passed, "checks": [{"name": name, "left": 1. if passed else 2.,
        "right": 1., "comparison": "le", "passed": passed} for name in names]}


def test_exact_independent_coverage(helper):
    names = helper.expected_members()
    assert len(names) == 9505
    assert len(helper.inherited_members()) == 74
    assert len(helper.fit_names()) == 12
    assert sum(name.startswith("inherited/fits/") for name in names) == 60
    assert sum("/physics/" in name for name in names) == 900
    assert sum(name.endswith("observer-final.json") for name in names) == 6
    assert "control/shift/cached_mlp-pair2/scoring/049.npz" in names
    assert "control/full/known_state/physics/000.json" in names


@pytest.mark.parametrize("passed", [False, True])
def test_gate_preserves_pass_and_failure(helper, passed):
    value = gate(helper, passed=passed)
    before = json.dumps(value, sort_keys=True)
    helper.validate_gate(value)
    assert json.dumps(value, sort_keys=True) == before


@pytest.mark.parametrize("change", ["missing", "duplicate", "arithmetic", "overall"])
def test_gate_corruption_rejected(helper, change):
    value = gate(helper)
    if change == "missing":
        value["checks"].pop()
    elif change == "duplicate":
        value["checks"][-1] = value["checks"][0]
    elif change == "arithmetic":
        value["checks"][0]["passed"] = True
    else:
        value["passed"] = True
    with pytest.raises(ValueError):
        helper.validate_gate(value)


@pytest.mark.parametrize("change", ["corrupt", "missing", "extra", "symlink"])
def test_exact_member_binding_rejects_changes(helper, tmp_path, change):
    path = put(tmp_path / "data.bin", b"immutable")
    expected = {"data.bin": helper.sha(path)}
    if change == "corrupt":
        path.write_bytes(b"mutated")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        put(tmp_path / "undeclared.bin")
    else:
        path.unlink()
        target = put(tmp_path.parent / (tmp_path.name + "-external"))
        path.symlink_to(target)
    with pytest.raises(ValueError):
        helper.bind_members(tmp_path, expected)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "./a", "a//b", "a\\b", ""])
def test_unsafe_paths_rejected(helper, tmp_path, name):
    with pytest.raises(ValueError):
        helper.child(tmp_path, name)


def test_deterministic_archive_and_corruption_detection(helper, tmp_path):
    root = tmp_path / "inputs"
    paths = {put(root / "a.txt", b"abc"), put(root / "nested/empty", b""), put(root / "z.bin", bytes(range(256)))}
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    manifest = helper.write_archive(root, paths, first)
    other = helper.write_archive(root, paths, second)
    assert manifest == other and helper.sha(first) == helper.sha(second)
    helper.verify_archive(first, manifest)
    modified = [dict(row) for row in manifest]
    modified[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member bytes"):
        helper.verify_archive(first, modified)
    with pytest.raises(ValueError):
        helper.verify_archive(first, manifest[:-1])
    with pytest.raises(FileExistsError):
        helper.write_archive(root, paths, first)


@pytest.mark.parametrize("length", [9, 10, 11, 20, 21])
def test_split_boundary_and_ordered_byte_reassembly(helper, tmp_path, length):
    archive = put(tmp_path / "bytes.tar.gz", bytes(range(length)))
    assets, split = helper.split_assets(archive, tmp_path, max_asset_bytes=10, part_bytes=10)
    assert split is (length > 10)
    assert all(0 < row["bytes"] <= 10 for row in assets)
    assert b"".join((tmp_path / row["name"]).read_bytes() for row in assets) == archive.read_bytes()
    assert all(helper.sha(tmp_path / row["name"]) == row["sha256"] for row in assets)
    if split:
        with pytest.raises(FileExistsError):
            helper.split_assets(archive, tmp_path, max_asset_bytes=10, part_bytes=10)


def report_fixture(helper, root):
    folder = root / "report"
    folder.mkdir()
    renderer = put(root / f"output/{helper.STUDY}/render.py")
    plan, audit_sha = {"sources": {}}, "a" * 64
    audit = {"files": {"summary.json": "b" * 64}, "execution_completed_sha256": "c" * 64,
             "execution_members": {}}
    summary = {"continuation_gate": gate(helper), "costs": {"synthetic": True},
               "paired_descriptive_comparisons": {}, "control": {}}
    rows = []
    for panel in helper.PANELS:
        summary["control"][panel] = {}
        for label in (*helper.fit_names(), *helper.REFERENCES):
            summary["control"][panel][label] = {"mean_cost": 1., "episode_costs": [1.] * 64}
            rows.append({"panel": panel, "policy": label, "mean_native_cost": 1.})
    replay = {"models": {}, "frames": 50, "frame_duration_ms": 100, "playback_seconds": 5,
              "native_episode_seconds": 1, "native_pixels": False}
    for arm in helper.ARMS:
        label = arm + "-pair0"
        replay["models"][label] = {"case_index": 0, "pair": "pair0", "panel": "ordinary", "native_cost": 1.,
            "episodes_npz_sha256": "d" * 64, "episodes_json_sha256": "e" * 64}
        for extension, value in (("npz", "d" * 64), ("json", "e" * 64)):
            audit["execution_members"][f"control/ordinary/{label}/episodes.{extension}"] = value
    inputs = {"plan_sha256": helper.EXPECTED_PLAN, "audit_receipt_sha256": audit_sha,
        "audit_summary_sha256": audit["files"]["summary.json"], "execution_completed_sha256": audit["execution_completed_sha256"],
        "execution_member_count": 9505, "frozen_source_count": 90}
    report = {"status": "completed", "study": helper.STUDY, "engineering": False, "inputs": inputs,
        "continuation_gate": summary["continuation_gate"], "costs": summary["costs"], "replay": replay,
        "paired_descriptive_comparisons": {}, "new_model_calls": 0, "new_native_calls": 0,
        "control_rows": rows, "work_counts": {}}
    helper.write(folder / "report.json", report)
    names = ["report.json", "tables.md", "fixed-case-replay.gif", "replay-preview.png"]
    names += [f"{stem}.{extension}" for stem in ("native-costs", "utility-vs-cost", "gate-checks") for extension in ("png", "svg", "pdf")]
    for name in names[1:]:
        put(folder / name)
    receipt = {"status": "completed", "study": helper.STUDY, "engineering": False, "inputs": inputs,
        "scope": "saved-artifact reporting and fixed-case schematic replay only", "source_sha256": {},
        "new_model_calls": 0, "new_policy_calls": 0, "new_native_calls": 0, "control_rows": 51,
        "learned_rows": 36, "reference_rows": 15, "inherited_models": 12,
        "renderer_source_sha256": helper.sha(renderer), "gate_passed": False, "checks_passed": 0,
        "work_counts": {}, "replay": replay, "files": {name: helper.sha(folder / name) for name in names}}
    helper.write(folder / "receipt.json", receipt)
    return folder, helper.sha(folder / "receipt.json"), plan, audit, summary, audit_sha


def test_real_report_validator_keeps_failed_gate(helper, tmp_path):
    args = report_fixture(helper, tmp_path)
    paths = helper.validate_reporting(tmp_path, *args)
    assert len(paths) == 15  # thirteen outputs, receipt and exact renderer source


@pytest.mark.parametrize("change", ["missing", "corrupt", "wrong_external_sha", "changed_source"])
def test_real_report_validation_rejects_binding_failure(helper, tmp_path, change):
    args = list(report_fixture(helper, tmp_path))
    if change == "missing":
        (args[0] / "gate-checks.pdf").unlink()
    elif change == "corrupt":
        (args[0] / "tables.md").write_bytes(b"changed")
    elif change == "changed_source":
        (tmp_path / f"output/{helper.STUDY}/render.py").write_bytes(b"changed")
    else:
        args[1] = "0" * 64
    with pytest.raises(ValueError):
        helper.validate_reporting(tmp_path, *args)


def orchestrated_fixture(helper, tmp_path, monkeypatch):
    """Stub only authentication layers for isolated packaging-state tests."""
    root = tmp_path / "constructed-fixture"
    root.mkdir()
    source = put(root / f"output/{helper.STUDY}/package.py", Path(helper.__file__).read_bytes())
    data = put(root / "synthetic/data.bin")
    prep = put(root / "synthetic/preparation.bin")
    report = put(root / "synthetic/report.bin")
    previous = put(root / "synthetic/previous.bin")
    for name in ("LICENSE", "pyproject.toml", "uv.lock"):
        put(root / name)
    monkeypatch.setattr(helper, "ROOT", root)
    monkeypatch.setattr(helper, "__file__", str(source))
    plan, audit, summary, readiness = {"sources": {}}, {"execution_completed_sha256": "0" * 64}, {
        "continuation_gate": gate(helper)}, {"required_free_disk_bytes": 0}
    monkeypatch.setattr(helper, "validate_scored", lambda *args: (plan, audit, summary, readiness, {data}))
    monkeypatch.setattr(helper, "validate_reporting", lambda *args: {report})
    monkeypatch.setattr(helper, "previous_release_dependency", lambda *args: ({previous}, {"scope": "synthetic fixture"}))
    monkeypatch.setattr(helper, "preparation_files", lambda *args: {prep})
    monkeypatch.setattr(helper.shutil, "disk_usage", lambda *args: SimpleNamespace(free=2**40))
    args = argparse.Namespace(completed_authorized=True, expected_plan_sha256=helper.EXPECTED_PLAN,
        expected_audit_receipt_sha256="a" * 64, expected_reporting_receipt_sha256="b" * 64,
        expected_terminal_verification_sha256="c" * 64, expected_previous_release_verification_sha256="d" * 64,
        out=f"output/{helper.STUDY}/publication-v1", reporting_directory="reporting",
        previous_package_directory="previous", previous_release_verification="prior.json", reporting_file=[])
    return root, args


def test_orchestration_preserves_false_qualification_and_exclusive_output(helper, tmp_path, monkeypatch):
    root, args = orchestrated_fixture(helper, tmp_path, monkeypatch)
    receipt = helper.package(args)
    out = root / args.out
    assert receipt["status"] == "verified" and receipt["scientific_status"]["continuation_passed"] is False
    assert receipt["scientific_status"]["checks_passed"] == 0 and receipt["uploaded"] is False
    before = {path.name: helper.sha(path) for path in out.iterdir()}
    with pytest.raises(ValueError, match="Exclusive"):
        helper.package(args)
    assert before == {path.name: helper.sha(path) for path in out.iterdir()}
    for bundle in helper.read(out / "manifest.json")["archives"]:
        helper.verify_archive(out / bundle["name"], bundle["members"])


def test_authorized_preflight_failure_is_retained(helper, tmp_path, monkeypatch):
    root, args = orchestrated_fixture(helper, tmp_path, monkeypatch)
    args.expected_audit_receipt_sha256 = "not-a-hash"
    with pytest.raises(ValueError):
        helper.package(args)
    out = root / args.out
    assert set(helper.files(out)) == {"packaging-started.json", "packaging-failed.json"}
    failed = helper.read(out / "packaging-failed.json")
    assert failed["stage"] == "authorized-preflight" and failed["automatic_retry"] is False


def test_unauthorized_invocation_has_no_artifact_reads_or_output(helper, tmp_path, monkeypatch):
    root, args = orchestrated_fixture(helper, tmp_path, monkeypatch)
    args.completed_authorized = False
    monkeypatch.setattr(helper, "read", lambda *args: pytest.fail("No unauthorized reads"))
    with pytest.raises(ValueError, match="authorization"):
        helper.package(args)
    assert not (root / args.out).exists()


def test_failure_after_archiving_preserves_both_archives(helper, tmp_path, monkeypatch):
    root, args = orchestrated_fixture(helper, tmp_path, monkeypatch)
    original = helper.validate_scored
    calls = 0
    def reject_second(*values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic late binding change")
        return original(*values)
    monkeypatch.setattr(helper, "validate_scored", reject_second)
    with pytest.raises(ValueError, match="late binding"):
        helper.package(args)
    out = root / args.out
    assert len(list(out.glob("*.tar.gz"))) == 2 and not (out / "receipt.json").exists()
    assert helper.read(out / "packaging-failed.json")["stage"] == "reauthenticate-all-input-bindings"


def test_missing_previous_verification_never_falls_back(helper, tmp_path):
    with pytest.raises(ValueError, match="Missing regular member"):
        helper.previous_release_dependency(tmp_path, tmp_path / "missing.json", "a" * 64, tmp_path / "package", {})


def previous_fixture(helper, root, *, omit_failure=False):
    """Saved publication metadata fixture; no remote assets or network calls."""
    folder = root / "prior-package"
    folder.mkdir()
    sources = {f"src/synthetic{i:02d}.py": "a" * 64 for i in range(80)}
    science = [{"path": "synthetic/previous-source.py", "bytes": 1, "sha256": "b" * 64}]
    failed_names = ["output/reacher-geometry-capacity-v1/attempt/execution/failed.json",
                    "output/reacher-geometry-rehearsal-v1/attempt-01/audit/failed.json"]
    if omit_failure:
        failed_names.pop()
    prep = [{"path": name, "bytes": 1, "sha256": "c" * 64} for name in failed_names]
    archives, assets = [], []
    for index, (rows, scope) in enumerate(((science, "completed_scored_study_all_results"),
                                          (prep, "engineering_only_no_effectiveness_claim"))):
        name = f"synthetic-prior-{index}.tar.gz"
        archive = {"name": name, "scope": scope, "bytes": 100, "sha256": "d" * 64,
            "member_count": len(rows), "members": rows, "all_members_reopened_and_verified": True,
            "split_for_release": False, "concatenation_verified": False, "ordered_release_assets": [name]}
        archives.append(archive)
        assets.append({"name": name, "bytes": 100, "sha256": "d" * 64, "bundle": name, "scope": scope})
    helper.write(folder / "manifest.json", {"study": helper.PREVIOUS_STUDY, "archives": archives})
    receipt = {"status": "verified", "study": helper.PREVIOUS_STUDY, "plan_sha256": helper.PREVIOUS_PLAN,
        "audit_receipt_sha256": helper.PREVIOUS_AUDIT, "execution_completed_sha256": "e" * 64,
        "source_sha256": sources, "scientific_status": {"continuation_passed": False},
        "manifest_sha256": helper.sha(folder / "manifest.json"),
        "archives": [{key: value for key, value in row.items() if key != "members"} for row in archives],
        "release_assets": assets, "member_count": len(science) + len(prep)}
    helper.write(folder / "receipt.json", receipt)
    helper.write(folder / "packaging-started.json", {"scope": "synthetic fixture"})
    put(folder / "README.md")
    sidecars = [{"name": name, "bytes": (folder / name).stat().st_size, "sha256": helper.sha(folder / name)}
                for name in helper.SIDECARS if name != "SHA256SUMS"]
    (folder / "SHA256SUMS").write_text("".join(f"{row['sha256']}  {row['name']}\n" for row in [*sidecars, *assets]))
    sidecars.append({"name": "SHA256SUMS", "bytes": (folder / "SHA256SUMS").stat().st_size,
                     "sha256": helper.sha(folder / "SHA256SUMS")})
    remote = [{**row, "asset_id": index + 1,
        "browser_download_url": "https://github.com/kw2828/OpenJev/releases/download/research-reacher-geometry-score-v1/" + row["name"]}
        for index, row in enumerate([*assets, *sidecars])]
    verification = {"status": "published_and_verified", "repository": "kw2828/OpenJev",
        "release_tag": "research-reacher-geometry-score-v1", "release_id": 1,
        "release_url": "https://github.com/kw2828/OpenJev/releases/tag/research-reacher-geometry-score-v1",
        "target_commit": "f" * 40, "resolved_tag_commit": "f" * 40,
        "audit_receipt_sha256": helper.PREVIOUS_AUDIT, "new_model_calls": 0, "new_native_calls": 0,
        "package_receipt_sha256": helper.sha(folder / "receipt.json"), "scientific_status": receipt["scientific_status"],
        "assets": remote, "public_sidecar_downloads": [row for row in sidecars if row["name"] in ("receipt.json", "manifest.json", "SHA256SUMS")],
        "verified_utc": "synthetic fixture time", "published_at": "synthetic fixture time", "publisher_sha256": "f" * 64}
    verification_path = root / "prior-verification.json"
    helper.write(verification_path, verification)
    plan = {"sources": sources, "geometry_source": {"plan_sha256": helper.PREVIOUS_PLAN,
        "audit_receipt_sha256": helper.PREVIOUS_AUDIT, "completed_sha256": "e" * 64}}
    return verification_path, helper.sha(verification_path), folder, plan


def test_real_previous_release_validator_retains_failed_ancestor_identity(helper, tmp_path):
    args = previous_fixture(helper, tmp_path)
    paths, dependency = helper.previous_release_dependency(tmp_path, *args)
    assert len(paths) == 6
    assert len(dependency["retained_failed_ancestors"]) == 2
    assert dependency["mode"] == "mandatory_previously_verified_release"


@pytest.mark.parametrize("change", ["manifest", "sidecar", "missing", "parent", "missing_failure"])
def test_real_previous_release_validation_rejects_partial_or_changed_evidence(helper, tmp_path, change):
    args = previous_fixture(helper, tmp_path, omit_failure=change == "missing_failure")
    if change == "manifest":
        (args[2] / "manifest.json").write_bytes(b"{}")
    elif change == "sidecar":
        (args[2] / "README.md").write_bytes(b"modified")
    elif change == "missing":
        (args[2] / "SHA256SUMS").unlink()
    elif change == "parent":
        args[3]["geometry_source"]["completed_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        helper.previous_release_dependency(tmp_path, *args)


def test_failure_after_receipt_invalidates_success(helper, tmp_path, monkeypatch):
    root, args = orchestrated_fixture(helper, tmp_path, monkeypatch)
    original = Path.open
    def fail_checksums(self, mode="r", *positional, **keywords):
        if self.name == "SHA256SUMS" and mode == "x":
            raise OSError("synthetic sidecar write failure")
        return original(self, mode, *positional, **keywords)
    monkeypatch.setattr(Path, "open", fail_checksums)
    with pytest.raises(OSError, match="sidecar write"):
        helper.package(args)
    out = root / args.out
    assert not (out / "receipt.json").exists() and (out / "incomplete-receipt.json").exists()
    assert (out / "packaging-failed.json").exists() and len(list(out.glob("*.tar.gz"))) == 2


def test_storage_preflight_keeps_reserve(helper, tmp_path, monkeypatch):
    payload = put(tmp_path / "bytes", b"x" * 100)
    monkeypatch.setattr(helper.shutil, "disk_usage", lambda *args: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="Insufficient space"):
        helper.storage_preflight(tmp_path, ({payload},), {"required_free_disk_bytes": 0})
