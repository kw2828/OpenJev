"""Package the closed engineering rehearsal without importing experiment code."""

from __future__ import annotations

import datetime
import gzip
import hashlib
import io
import json
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/reacher-cache-rehearsal-v1"
ATTEMPT = BASE / "attempt-01"
EVIDENCE = ROOT / "evidence/reacher-cache-rehearsal-v1"
PUBLICATION = BASE / "publication-v1"
EXPECTED_PLAN = "c56f6b73c19b0c3ac17bf55a0f1ac95b2513e2688561d52907cb150ff57c1279"
EXPECTED_AUDIT = "77853eb8c8166651054f5e2fef185744809cf686a5c0fa8f7415b3782d9b6e34"
SCOPE = "engineering_rehearsal_only_not_scientific_efficacy"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid member name")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
            "Unsafe or noncanonical member name")
    return name


def checked(root, name, digest):
    safe_name(name)
    path = root / name
    require(path.is_file() and not path.is_symlink(), f"Regular file required: {name}")
    require(root.resolve() in path.resolve().parents, f"Outside declared root: {name}")
    require(sha(path) == digest, f"Hash differs: {name}")
    return path


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    result = json.loads(path.read_text(), parse_constant=reject)

    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), "Nonfinite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    finite(result)
    return result


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def files(root):
    paths = sorted(root.rglob("*"))
    require(not root.is_symlink() and not any(path.is_symlink() for path in paths), "No symlink members")
    require(all(path.is_dir() or path.is_file() for path in paths), "Only regular files/directories")
    return {path.relative_to(root).as_posix(): path for path in paths if path.is_file()}


def authenticate_members(folder, expected, *, extra=()):
    require(set(files(folder)) == set(expected) | set(extra), f"Exact membership differs: {folder}")
    for name, digest in expected.items():
        checked(folder, name, digest)


def archive_write(path, sources, manifest):
    require(set(sources) == set(manifest), "Archive input membership differs")
    with (path.open("xb") as raw,
          gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as compressed,
          tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive):
        for name, source in sorted(sources.items()):
            safe_name(name)
            data = source.read_bytes()
            item = manifest[name]
            require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                    f"Input changed during packaging: {name}")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))


def archive_verify(path, manifest):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        require([member.name for member in members] == sorted(manifest), "Archive membership/order differs")
        for member in members:
            safe_name(member.name)
            require(member.isfile() and member.mode == 0o644 and member.mtime == 0
                    and member.uid == member.gid == 0 and member.uname == member.gname == "",
                    "Archive type/header differs")
            with archive.extractfile(member) as stream:
                data = stream.read()
            item = manifest[member.name]
            require(len(data) == member.size == item["bytes"]
                    and hashlib.sha256(data).hexdigest() == item["sha256"],
                    f"Reopened archive bytes differ: {member.name}")
    return len(members)


def authenticate():
    require(not (BASE / "failed.json").exists(), "Launcher has failed terminal")
    plan = read(checked(ATTEMPT, "plan.json", EXPECTED_PLAN))
    audit = read(checked(ATTEMPT, "audit/receipt.json", EXPECTED_AUDIT))
    terminal, started = read(BASE / "completed.json"), read(BASE / "started.json")
    fixture, completed = read(ATTEMPT / "fixture.json"), read(ATTEMPT / "execution/completed.json")
    summary = read(ATTEMPT / "audit/summary.json")
    require(plan["engineering"] is audit["engineering"] is summary["engineering"] is True,
            "Explicit engineering scope required")
    require(plan["rng_namespace"] == "reacher-cache-engineering-whole-tree-v1", "Engineering namespace")
    require(all(value["status"] == "completed" for value in (terminal, audit, completed, summary)),
            "Completed terminal phases required")
    require(all(value["plan_sha256"] == EXPECTED_PLAN for value in (terminal, audit, fixture, completed, summary)),
            "Plan bindings differ")
    require(terminal["audit_receipt_sha256"] == EXPECTED_AUDIT, "External audit hash differs")
    require(audit["source_sha256"] == plan["sources"] and audit["runtime"] == plan["runtime"],
            "Audit source/runtime identity")
    require(len(plan["sources"]) == 70, "Exactly 70 plan-bound sources required")
    for name, digest in plan["sources"].items():
        checked(ROOT, name, digest)
    checked(BASE, "run.py", started["script_sha256"])
    checked(ROOT, "tests/reacher_cache_fixture.py", plan["fixture_source_sha256"])
    require(fixture["production_authentication"] is False and fixture["inherited_checkpoint_calls"] == 0,
            "Explicit fixture lineage boundary required")
    require(audit["saved_output_only"] is summary["saved_output_only"] is True, "Saved audit required")
    require(summary["new_model_calls"] == summary["new_policy_calls"] == summary["new_fits"] == 0,
            "No audit neural or training calls")
    require(completed["fits"] == summary["coverage"]["fits"] == 15
            and completed["control_rows"] == summary["coverage"]["control_rows"] == 60
            and completed["prediction_episodes"] == 2 and completed["astra_calls"] == 0,
            "Complete rehearsal membership")
    require(summary["coverage"]["total_optimizer_updates"] == 30
            and summary["native_transitions_checked"] == 3300
            and summary["native_max_abs_error"] == 0
            and summary["public_observer_transitions_checked"] == 147,
            "Native and update coverage")
    require(summary["phase_boundary"]["all_fifteen_fits_restored_before_evaluation"] is True,
            "All-class restoration boundary")
    execution = ATTEMPT / "execution"
    require(sha(execution / "completed.json") == audit["execution_completed_sha256"]
            == summary["execution_completed_sha256"], "Execution receipt differs")
    require(audit["execution_members"] == completed["files"], "Execution manifest differs")
    authenticate_members(execution, completed["files"], extra=("completed.json",))
    authenticate_members(ATTEMPT / "audit", audit["files"], extra=("receipt.json",))
    require(not any("failed" in name or "partial-" in name or "over-cap" in name for name in files(ATTEMPT)),
            "Failure/partial artifact in completed rehearsal")
    require(completed["wall_seconds"] == terminal["execution"]["wall_seconds"]
            == summary["costs"]["new_execution_wall_seconds"] <= plan["cap_seconds"] == 300,
            "Execution cap binding")
    require(0 < summary["costs"]["audit_validation_wall_seconds"] <= plan["audit_cap_seconds"] == 300,
            "Audit cap binding")
    return plan, audit, summary, terminal, completed


def main():
    plan, _audit, summary, terminal, completed = authenticate()
    raw = {f"attempt-01/{name}": path for name, path in files(ATTEMPT).items()}
    raw.update({name: BASE / name for name in ("run.py", "started.json", "completed.json")})
    raw_identity = {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                    for name, path in raw.items()}
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    PUBLICATION.mkdir(exist_ok=False)
    try:
        snapshot = EVIDENCE / "source-snapshot"
        sources = dict(raw)
        capture = {**plan["sources"], "tests/reacher_cache_fixture.py": plan["fixture_source_sha256"]}
        for name, digest in capture.items():
            source = checked(ROOT, name, digest)
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            require(sha(destination) == digest, f"Snapshot differs: {name}")
            sources[f"source-snapshot/{name}"] = destination
        for source, name in ((BASE / "started.json", "supervision-started.json"),
                             (BASE / "completed.json", "supervision-completed.json"),
                             (BASE / "run.py", "launcher.py"),
                             (ATTEMPT / "fixture.json", "fixture-description.json")):
            shutil.copyfile(source, EVIDENCE / name)
        manifest = {"schema": "reacher-cache-rehearsal-bundle-v1", "scope": SCOPE,
            "members": {name: {"sha256": sha(path), "bytes": path.stat().st_size,
                                "classification": "source_snapshot" if name.startswith("source-snapshot/")
                                else "original_closed_rehearsal"}
                        for name, path in sorted(sources.items())}}
        write(EVIDENCE / "bundle-manifest.json", manifest)
        archive = PUBLICATION / "reacher-cache-rehearsal-v1.tar.gz"
        archive_write(archive, sources, manifest["members"])
        archive_verify(archive, manifest["members"])
        result = {"scope": SCOPE, "status": "completed", "plan_sha256": EXPECTED_PLAN,
            "original_audit_receipt_sha256": EXPECTED_AUDIT,
            "original_audit_summary_sha256": sha(ATTEMPT / "audit/summary.json"),
            "execution_completed_sha256": sha(ATTEMPT / "execution/completed.json"),
            "fixture": {key: plan[key] for key in ("rng_namespace", "train_episodes", "epochs", "batch_size",
                "hidden_size", "mlp_width", "prediction_episodes", "control_episodes", "steps", "planning_horizon")},
            "coverage": summary["coverage"], "phase_boundary": summary["phase_boundary"],
            "audit": {"native_transitions_checked": 3300, "native_max_abs_error": 0,
                "native_breakdown": {"training": 200, "prediction": 100, "control": 3000},
                "public_observer_transitions_checked": 147, "new_model_calls": 0,
                "new_policy_calls": 0, "new_fits": 0, "production_lineage_authenticated": False,
                "substitutions": ["Explicit synthetic historical-lineage metadata",
                                  "Four engineering training records, still replayed natively"]},
            "costs": summary["costs"], "supervisor_wall_seconds": terminal["wall_seconds"],
            "runtime_recorded_in_fixture_plan": plan["runtime"],
            "raw_gate_output": {"preserved_losslessly": True,
                "archive_member": "attempt-01/audit/summary.json", "scientific_interpretation_authorized": False},
            "limits": "Engineering integration only. No efficacy, architecture superiority or equivalence claim. "
                "The audit does not independently rerun neural forward passes or optimizer updates. "
                "External historical artifacts and installed runtime dependencies are not recursively bundled."}
        write(EVIDENCE / "summary.json", result)
        text = f'''# Reacher cache engineering rehearsal

The first retained rehearsal completed **15 fits and 60 controller rows**. Its saved-output audit replayed **3,300 native transitions with zero discrepancy**, plus 147 supplied-physics observer transitions. This validates the reduced fixture's integration, not model effectiveness.

The [fixture](fixture-description.json) used four engineering training episodes, one epoch and two updates per fit, GRU width 4, MLP width 7, two prediction episodes and one control case per row. All five architectures and three initialization pairs were retained. All 15 actual classes were restored before fresh evaluation. The only substituted audit boundaries were historical lineage and the inherited-training corpus identity; all four fixture training episodes were still replayed.

Execution took {completed['wall_seconds']:.6f} seconds, saved-output audit validation {summary['costs']['audit_validation_wall_seconds']:.6f} seconds, and the [launcher](supervision-completed.json) recorded {terminal['wall_seconds']:.6f} seconds overall. These nested fixture timings are not isolated latency or full-study estimates. See the [coverage and accounting](summary.json).

[The archive manifest](bundle-manifest.json) binds every original file in `attempt-01`, the closed launcher and terminal files, all 70 plan-bound source snapshots and the [fixture helper](source-snapshot/tests/reacher_cache_fixture.py). Every uncompressed archive member was reopened and checked against its path, length and SHA-256. The [receipt](receipt.json) records the archive identity and unchanged input checks. The archive remains local at `output/reacher-cache-rehearsal-v1/publication-v1/reacher-cache-rehearsal-v1.tar.gz`; no upload is claimed.

Raw fixture gate and utility outputs remain losslessly preserved inside the archive. They are **engineering outputs, not scientific results**. Packaging made no model, training or native calls and did not rerun tests. External historical dependencies are not recursively included, so this is a verification bundle rather than a standalone production-lineage reproduction. Original source licenses remain applicable; packaging adds no separate data or third-party license claim.
'''
        with (EVIDENCE / "README.md").open("x") as stream:
            stream.write(text)
        require({f"attempt-01/{name}" for name in files(ATTEMPT)}
                == {name for name in raw_identity if name.startswith("attempt-01/")}, "Original membership changed")
        for name, item in raw_identity.items():
            require(sha(raw[name]) == item["sha256"] and raw[name].stat().st_size == item["bytes"],
                    f"Original changed: {name}")
        for name, digest in capture.items():
            checked(ROOT, name, digest)
        receipt = {"status": "completed", "scope": "saved-byte packaging of engineering rehearsal only",
            "packaged_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": EXPECTED_PLAN, "original_audit_receipt_sha256": EXPECTED_AUDIT,
            "original_audit_summary_sha256": result["original_audit_summary_sha256"],
            "source_sha256": plan["sources"], "fixture_helper_sha256": plan["fixture_source_sha256"],
            "launcher_sha256": sha(BASE / "run.py"), "original_file_count": len(raw),
            "closed_attempt_file_count": len(raw) - 3, "source_snapshot_members": len(capture),
            "original_inputs_unchanged": True, "all_captured_sources_still_match": True,
            "archive": {"path": archive.relative_to(ROOT).as_posix(), "sha256": sha(archive),
                "bytes": archive.stat().st_size, "member_count": len(manifest["members"]),
                "uncompressed_member_bytes": sum(row["bytes"] for row in manifest["members"].values()),
                "reopened_all_members_verified": True, "deterministic_headers": True, "uploaded": False},
            "verification": {"plan_external_hash_matches": True, "audit_external_hash_matches": True,
                "execution_and_audit_manifest_hashes_checked": True, "new_learned_model_calls": 0,
                "new_native_replay_calls": 0, "new_training_calls": 0,
                "original_gate_outputs_preserved_without_scientific_interpretation": True},
            "packager": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__))},
            "files": {name: sha(path) for name, path in files(EVIDENCE).items()}}
        write(EVIDENCE / "receipt.json", receipt)
        checksums = [*files(EVIDENCE).values(), archive, Path(__file__)]
        with (EVIDENCE / "SHA256SUMS").open("x") as stream:
            stream.write("".join(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in checksums))
        print(json.dumps({"receipt_sha256": sha(EVIDENCE / "receipt.json"), "archive": receipt["archive"],
                          "original_file_count": len(raw), "source_snapshot_members": len(capture)}))
    except BaseException as error:
        if not (EVIDENCE / "failed.json").exists():
            write(EVIDENCE / "failed.json", {"status": "failed", "scope": SCOPE, "error": repr(error),
                  "partial_outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()
