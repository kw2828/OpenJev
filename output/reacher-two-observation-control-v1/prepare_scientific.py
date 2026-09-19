"""Prepare and freeze a study only after externally authenticated readiness.

The one explicit authorization covers preparation AND freeze, saved as separate
artifacts. Import performs no preparation or seed allocation. There is no execution,
training or simulator entry point. Explicit caps and a real source Git commit are
mandatory; all 116 scientific bytes must match that commit before new scored
manifest allocation. Failed preparations remain exclusive and cannot be retried.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = "output/reacher-two-observation-control-v1/prepare_scientific.py"
TEST_HELPER = "output/reacher-two-observation-control-v1/test_prepare_scientific.py"
CAPACITY_HELPER = "output/reacher-two-observation-control-v1/capacity_probe.py"
CAPACITY_TESTS = "output/reacher-two-observation-control-v1/test_capacity_probe.py"
REHEARSAL = {"path": "output/reacher-two-observation-rehearsal-v1/attempt-01/qualification.json",
             "sha256": "6066f071cbe3ec48f45efdbfce46d2c397da14dce39063188ac838e213cafce9"}
REHEARSAL_PLAN = "output/reacher-two-observation-rehearsal-v1/attempt-01/plan.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def dependencies():
    from openjev.research import reacher_two_observation_experiment as experiment
    from openjev.research import reacher_two_observation_protocol as protocol
    from openjev.research import reacher_two_observation_streams as streams
    return experiment, protocol, streams


def lineage_refs():
    return {
        "cache": {"plan_path": "evidence/reacher-cache-ablation-v1/protocol/plan.json",
                  "plan_sha256": "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa",
                  "audit_path": "evidence/reacher-cache-ablation-v1/audit/receipt.json",
                  "audit_receipt_sha256": "d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790",
                  "execution_path": "runs/reacher-cache-ablation-v1/attempt"},
        "prerequisite": {"plan_path": "evidence/reacher-geometry-memory-v1/protocol/plan.json",
                         "plan_sha256": "23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee",
                         "audit_path": "evidence/reacher-geometry-memory-v1/audit/receipt.json",
                         "audit_receipt_sha256": "cb64ded7bbcc4ad6e342f9ac3b431e085e754f77075bb2e4852d6b82d29a54c6",
                         "execution_path": "runs/reacher-geometry-memory-v1/attempt",
                         "terminal_path": "evidence/reacher-geometry-memory-v1/terminal-verification.json",
                         "terminal_sha256": "8c65730d6547b118eaeaca86ef61857ef0e4f52145f3eea214316f23f32dbae2",
                         "terminal_kind": "independent_terminal_verification"}}


def git(root, *args):
    value = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    require(value.returncode == 0, "Git object verification failed: " + value.stderr.decode(errors="replace").strip())
    return value.stdout


def verify_source_commit(root, sources, commit):
    require(type(commit) is str and len(commit) == 40 and set(commit) <= set("0123456789abcdef"), "Explicit full lowercase source commit")
    require(type(sources) is dict and len(sources) == 116, "Exact 116-source map")
    actual = git(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    require(actual == commit, "Source object must itself be a real Git commit")
    blobs = {}
    for name, digest in sorted(sources.items()):
        require(type(name) is str and not Path(name).is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")), "Safe source path")
        current = Path(root).resolve()
        for part in name.split("/"):
            current /= part
            require(not current.is_symlink(), "No source symlink")
        require(current.is_file() and sha(current) == digest, "Current source differs: " + name)
        tree = git(root, "ls-tree", "-z", commit, "--", name).split(b"\0")
        require(len(tree) == 2 and tree[-1] == b"", "Exact tracked source member: " + name)
        header, member = tree[0].split(b"\t", 1)
        mode, kind, object_id = header.decode().split(" ")
        require(mode in ("100644", "100755") and kind == "blob" and member.decode() == name, "Regular Git source blob: " + name)
        payload = git(root, "cat-file", "blob", object_id)
        require(hashlib.sha256(payload).hexdigest() == digest, "Commit source bytes differ: " + name)
        blobs[name] = object_id
    return {"source_commit": commit, "source_sha256": dict(sources), "git_blob_ids": blobs,
            "verified_source_files": 116, "scope": "Actual Git object bytes and current regular-file bytes; no clean-tree or branch-tip assumption."}


def capacity_source_binding(experiment, root, reference, receipt):
    """Find the one completed capacity measurement through its qualified members."""
    candidates = [name for name in receipt["files"] if Path(name).name == "measurement-audit.json"]
    require(len(candidates) == 1, "Exactly one independently audited capacity measurement artifact")
    name = candidates[0]
    path = (Path(reference["path"]).parent / name).as_posix()
    payload = experiment.read_bound(root, path, receipt["files"][name])
    whole = payload["whole_measurement"]
    require(whole["status"] == "completed" and whole["engineering"] is True
            and whole["namespace"] == "reacher-two-observation-engineering-capacity-v1"
            and whole["rehearsal"]["qualification_sha256"] == REHEARSAL["sha256"]
            and whole["source_sha256"] == receipt["source_sha256"] and whole["runtime"] == receipt["runtime"]
            and whole["new_fits"] == 1 and whole["optimizer_updates"] == 24 and whole["row_count"] == 14
            and payload["new_model_calls"] == payload["new_optimizer_steps"] == 0,
            "Capacity measurement is completed engineering workload with saved-only audit")
    require(type(whole["profile_source_sha256"]) is dict
            and all(name in whole["profile_source_sha256"] for name in (CAPACITY_HELPER, CAPACITY_TESTS)), "Both actual capacity sources are bound")
    for name in (CAPACITY_HELPER, CAPACITY_TESTS):
        experiment.checked(root, name, whole["profile_source_sha256"][name])
    return {"artifact": path, "sha256": receipt["files"][candidates[0]],
            "profile_source_sha256": whole["profile_source_sha256"]}


def historical_registries(experiment, streams, root, rehearsal, capacity_binding):
    # Reuse the exact authenticated ordered old registry inventory. The existing
    # experiment validates it against the ORIGINAL production prerequisite and
    # adds that completed study's own five full64 namespaces and all child states.
    saved = experiment.read_bound(root, REHEARSAL_PLAN, rehearsal["plan_sha256"])
    require(saved["engineering"] is True and saved["sources"] == rehearsal["source_sha256"], "Bound rehearsal history source")
    history = copy.deepcopy(saved["historical_registries"])
    source = capacity_binding["profile_source_sha256"]
    old = history["engineering_sources"]
    require(all(name not in old or old[name] == digest for name, digest in source.items()
                if name not in saved["sources"]), "No conflicting as-run engineering source attribution")
    old.update({name: digest for name, digest in source.items() if name not in saved["sources"]})
    for name in (HELPER, TEST_HELPER):
        old[name] = experiment.sha(experiment._path(root, name))
    require(not any(row["source_path"] == CAPACITY_HELPER for row in history["literal_calls"]), "Append capacity literal attribution exactly once")
    roles = {"synthetic_constructors_and_orders": 410}
    history["literal_calls"].append({"source_path": CAPACITY_HELPER, "source_sha256": source[CAPACITY_HELPER],
        "numpy_registry": {}, "numpy_generators": {}, "torch_registry": roles,
        "torch_generators": streams.torch_generator_manifest(roles)})
    return history


def snapshot(root, out, sources, experiment):
    for name, digest in sorted(sources.items()):
        original = experiment.checked(root, name, digest)
        target = out / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as source, target.open("xb") as saved:
            shutil.copyfileobj(source, saved)
        require(sha(target) == digest, "Retained source snapshot differs")


def prepare(out, *, root, capacity, source_commit, cap_seconds, audit_cap_seconds, authorize_preparation_and_freeze=False):
    require(authorize_preparation_and_freeze is True, "Explicit scientific preparation-and-freeze authorization required")
    root, out = Path(root).resolve(strict=True), Path(out).resolve()
    require(out.is_relative_to(root), "Preparation must be retained under the explicit repository")
    out.mkdir(parents=True, exist_ok=False)
    begin, phase = time.monotonic(), "admission"
    try:
        write(out / "started.json", {"status": "started", "scope": "preparation and freeze only; never execution",
            "requested_capacity": capacity, "source_commit": source_commit,
            "requested_execution_cap_seconds": repr(cap_seconds), "requested_audit_cap_seconds": repr(audit_cap_seconds),
            "helper_sha256": sha(__file__), "automatic_retry": False})
        require(all(type(value) is int and value > 0 for value in (cap_seconds, audit_cap_seconds)), "Explicit positive reviewed phase caps, no defaults")
        require(type(capacity) is dict and set(capacity) == {"path", "sha256"}, "External completed capacity qualification reference")
        experiment, protocol, streams = dependencies()
        settings, observed_runtime, refs = protocol.settings(), experiment.runtime(), lineage_refs()
        # Readiness precedes historical manifest construction and new scored
        # allocation. _engineering_evidence is the reviewed common admission.
        parent = experiment.read_bound(root, refs["prerequisite"]["plan_path"], refs["prerequisite"]["plan_sha256"])
        sources = experiment._sources(root, parent["sources"])
        phase = "readiness"
        evidence_refs = {"rehearsal": dict(REHEARSAL), "capacity": copy.deepcopy(capacity)}
        evidence = experiment._engineering_evidence(root, evidence_refs, settings, sources, observed_runtime,
                                                    cap_seconds, audit_cap_seconds, False)
        phase = "source_commit"
        proof = verify_source_commit(root, sources, source_commit)
        write(out / "source-commit.json", proof)
        measured = capacity_source_binding(experiment, root, capacity, evidence["capacity"]["receipt"])
        phase = "history"
        history = historical_registries(experiment, streams, root, evidence["rehearsal"]["receipt"], measured)
        write(out / "request.json", {"settings": settings, "lineage_refs": refs,
            "historical_registries": history, "runtime": observed_runtime,
            "cap_seconds": cap_seconds, "audit_cap_seconds": audit_cap_seconds,
            "engineering_evidence": evidence_refs, "capacity_measurement_binding": measured,
            "source_commit": source_commit, "caps_authority": "Explicit caller-supplied reviewed caps exceeding the authenticated capacity projections."})
        snapshot(root, out, sources | history["engineering_sources"], experiment)
        phase = "prepare_manifest"
        plan = experiment.prepare(root, settings, lineage_refs=refs, historical_registries=history,
            runtime=observed_runtime, cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
            engineering_evidence=evidence_refs, engineering=False)
        require(plan["sources"] == sources and plan["engineering"] is False, "Prepared scientific source closure unchanged")
        write(out / "plan.json", plan)
        plan_sha256 = sha(out / "plan.json")
        candidate = out / "freeze-candidate.json"
        freeze = {"schema": "reacher-two-observation-freeze-v1", "status": "frozen", "study": protocol.STUDY,
            "engineering": False, "plan_sha256": plan_sha256, "content_sha256": plan["content_sha256"],
            "source_sha256": sources, "runtime": observed_runtime,
            "engineering_evidence_sha256": experiment.identity(plan["engineering_evidence"]),
            "source_commit": source_commit, "no_retry": True}
        write(candidate, freeze)
        phase = "freeze_validation"
        context = experiment.validate_plan(plan, plan_sha256, root=root, runtime=experiment.runtime(),
            engineering=False, plan_bytes=(out / "plan.json").read_bytes(),
            freeze_ref={"path": candidate.relative_to(root).as_posix(), "sha256": sha(candidate)})
        require(context.execution_authorized is True and context.engineering is False, "Full external freeze validation")
        require(verify_source_commit(root, sources, source_commit) == proof, "Source commit binding changed during preparation")
        candidate.rename(out / "freeze.json")
        files = {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
        write(out / "completed.json", {"status": "frozen_not_run", "study": protocol.STUDY,
            "engineering": False, "plan_sha256": plan_sha256, "freeze_sha256": sha(out / "freeze.json"),
            "source_commit": source_commit, "source_sha256": sources, "runtime": observed_runtime,
            "files": files, "wall_seconds": time.monotonic() - begin, "new_model_calls": 0,
            "new_optimizer_steps": 0, "new_native_calls": 0, "scientific_samples": 0,
            "scored_manifest_allocated": True, "scientific_execution_started": False,
            "scope": "Preparation derives the prospective scored seed/state manifest only after readiness; it draws no scientific samples and launches nothing."})
        return {"path": str(out / "plan.json"), "sha256": plan_sha256,
                "freeze": {"path": str(out / "freeze.json"), "sha256": sha(out / "freeze.json")},
                "completed_sha256": sha(out / "completed.json"), "scientific_execution_started": False}
    except BaseException as error:
        for filename in ("freeze.json", "freeze-candidate.json", "completed.json"):
            try:
                source = out / filename
                if source.exists():
                    source.rename(out / ("invalid-" + filename))
            except BaseException as preservation_error:  # noqa: BLE001
                error.add_note(f"Could not demote {filename}: {preservation_error!r}")
        try:
            write(out / "failed.json", {"status": "failed", "phase": phase, "error": repr(error),
                "notes": list(getattr(error, "__notes__", ())),
                "wall_seconds": time.monotonic() - begin, "automatic_retry": False,
                "new_model_calls": 0, "new_optimizer_steps": 0, "new_native_calls": 0,
                "scientific_samples": 0, "scientific_execution_started": False})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Could not retain failed preparation receipt: {preservation_error!r}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--capacity-qualification", required=True)
    parser.add_argument("--capacity-qualification-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cap-seconds", type=int, required=True)
    parser.add_argument("--audit-cap-seconds", type=int, required=True)
    parser.add_argument("--authorize-preparation-and-freeze", action="store_true")
    args = parser.parse_args()
    result = prepare(args.out, root=args.root, capacity={"path": args.capacity_qualification,
        "sha256": args.capacity_qualification_sha256}, source_commit=args.source_commit,
        cap_seconds=args.cap_seconds, audit_cap_seconds=args.audit_cap_seconds,
        authorize_preparation_and_freeze=args.authorize_preparation_and_freeze)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
