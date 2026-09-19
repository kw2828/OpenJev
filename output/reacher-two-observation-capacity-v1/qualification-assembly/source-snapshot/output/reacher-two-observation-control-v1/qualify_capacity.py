"""Assemble final engineering capacity qualification from saved evidence only.

No sizing estimates, caps, model work or replay are generated here. A completed
measurement, repaired saved audit, retained failures/process exits, and explicit
independent sizing review are mandatory. Output lives at their common ancestor
so original artifacts remain in place and every receipt member is a safe child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = "output/reacher-two-observation-control-v1/qualify_capacity.py"
TEST_HELPER = "output/reacher-two-observation-control-v1/test_qualify_capacity.py"
ORIGINAL = "output/reacher-two-observation-control-v1/capacity_probe.py"
ORIGINAL_TEST = "output/reacher-two-observation-control-v1/test_capacity_probe.py"
ORIGINAL_SHA = "adc72736ff277b0dee80fbeb644b30e3e1bebbb0fae03be36fde94a70728c063"
ORIGINAL_TEST_SHA = "90b76bb5dbfcefe63da4a3ff4411cf138ba1a208a3b242e19c741b4273a29a88"
MEASUREMENT_SHA = "416d8b10fd8bd36758db2a03f9b62b4270cfcc079b73558b87072cf593739e13"
CONTROL = "output/reacher-two-observation-control-v1/"
WRAPPERS = {"attempt01": CONTROL + "run_capacity.py", "attempt02": CONTROL + "run_capacity_attempt02.py",
            "repair": CONTROL + "run_capacity_audit_repair01.py"}


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def original(root):
    path = root / ORIGINAL
    require(sha(path) == ORIGINAL_SHA and sha(root / ORIGINAL_TEST) == ORIGINAL_TEST_SHA, "Original as-run sources unchanged")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == ORIGINAL_SHA, "Exact original helper source")
    module = types.ModuleType("_capacity_qualification_original")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Positive finite " + label)
    return value


def read_reference(base, root, reference):
    require(type(reference) is dict and set(reference) == {"path", "sha256"}, "Explicit external file reference")
    path = base.bound(root, reference["path"], reference["sha256"])
    return path, base.read(path)


def checked_tree(base, folder, receipt, terminal):
    actual = base.inventory(folder, float("inf"))
    require(set(actual) == set(receipt["files"]) | {terminal}, "Exact completed artifact membership")
    require(all(actual[name] == value for name, value in receipt["files"].items()), "Completed artifact checksums/sizes")
    return actual


def review_sizing(review, *, measurement_sha, audit_sha, sources, runtime, caps):
    require(review.get("status") == "reviewed" and review.get("measurement_completed_sha256") == measurement_sha
            and review.get("repaired_audit_completed_sha256") == audit_sha
            and review.get("source_sha256") == sources and review.get("runtime") == runtime,
            "Independent review binds these completed phases and exact source/runtime")
    require(type(caps) is dict and set(caps) == {"execution", "audit"}
            and all(type(v) is int and v > 0 for v in caps.values())
            and review.get("approved_caps") == caps, "Exact explicit externally reviewed caps")
    projected = review["projected_seconds"]
    require(type(projected) is dict and set(projected) == {"execution", "audit"}, "Both reviewed projections")
    for phase in caps:
        require(number(projected[phase], phase + " projection") < caps[phase], "Reviewed caps must exceed reviewed projections")
    return dict(projected)


def process_receipts(base, root, references, measurement, repaired, failed01, failed02, *, locations):
    require(type(references) is dict and set(references) == set(WRAPPERS), "All three actual process histories")
    result, wrappers = {}, {}
    for kind, reference in references.items():
        path, terminal = read_reference(base, root, reference)
        folder = path.parent
        started = base.read(folder / "started.json")
        wrapper = base.bound(root, WRAPPERS[kind], started["driver_sha256"])
        wrappers[WRAPPERS[kind]] = sha(wrapper)
        number(terminal["wall_seconds"], "outer process wall")
        if kind == "repair":
            require(terminal["status"] == "completed" and terminal["phase"] == "saved_audit_repair"
                    and type(terminal["exit_code"]) is int and terminal["exit_code"] == 0
                    and terminal["automatic_retry"] is False, "Actual repair subprocess exit zero")
            phase = base.read(folder / "process.json")
            require(all(terminal[key] == value for key, value in phase.items()), "Repair terminal/process identity")
            require(started["helper_sha256"] == repaired["audit_repair"]["repair_source_sha256"][CONTROL + "capacity_audit_repair.py"], "Reviewed repair helper executed")
            command = phase["command"]
            require(command == started["command"] and "--completed-sha256" in command
                    and command[command.index("--completed-sha256") + 1] == MEASUREMENT_SHA
                    and command[1:4] == [str(root / (CONTROL + "capacity_audit_repair.py")),
                                         str(locations["attempt02"]), str(locations["repair"])]
                    and started["cwd"] == str(root),
                    "Repair command binds original measurement")
            require(sha(folder / "as-run-driver.py") == started["driver_sha256"], "Actual repair wrapper snapshot")
            phases = [phase]
        else:
            wanted = [("run", 1)] if kind == "attempt01" else [("run", 0), ("audit", 1)]
            require(terminal["status"] == "failed" and terminal["automatic_retry"] is False
                    and terminal["phase"] == wanted[-1][0] and len(terminal["phases"]) == len(wanted), "Original failed process history retained")
            phases = terminal["phases"]
            for phase, (name, code) in zip(phases, wanted, strict=True):
                invocation = base.read(folder / f"{name}-invocation.json")
                require(phase["phase"] == name and type(phase["exit_code"]) is int and phase["exit_code"] == code
                        and phase == base.read(folder / f"{name}-process.json")
                        and phase["command"] == invocation["command"] and invocation["cwd"] == str(root)
                        and phase["command"][1:4] == [str(root / ORIGINAL), name, str(locations[kind])], "Actual phase exit/command/artifact identity")
                if name == "audit":
                    require(phase["command"][phase["command"].index("--completed-sha256") + 1] == MEASUREMENT_SHA, "Original audit pins completed measurement")
            require(sum(number(phase["wall_seconds"], "phase wall") for phase in phases) <= terminal["wall_seconds"], "All actual process phases charged")
            expected_helper = failed01["sources"][ORIGINAL] if kind == "attempt01" else ORIGINAL_SHA
            require(started["helper_sha256"] == expected_helper, "Actual helper version in each attempt")
        members = base.inventory(folder, float("inf"))
        required = {"started.json", path.name}
        if kind == "repair":
            required |= {"process.json", "as-run-driver.py", "audit.stdout", "audit.stderr"}
        else:
            required |= {f"{phase['phase']}{suffix}" for phase in phases
                         for suffix in ("-invocation.json", "-process.json", ".stdout", ".stderr")}
        require(required <= set(members), "All actual subprocess receipts and output streams retained")
        result[kind] = {**reference, "wall_seconds": terminal["wall_seconds"], "phases": phases,
                        "folder": folder, "files": members}
    require(result["attempt02"]["phases"][0]["wall_seconds"] >= measurement["wall_seconds"]
            and result["attempt02"]["phases"][1]["wall_seconds"] >= failed02["wall_seconds"]
            and result["repair"]["wall_seconds"] >= repaired["wall_seconds"], "Inner work is charged within actual subprocess timing")
    return result, wrappers


def assemble(request, out, *, root=ROOT, authorize_saved_qualification=False):
    require(authorize_saved_qualification is True, "Explicit saved qualification authorization required")
    root, out = Path(root).resolve(strict=True), Path(out).resolve()
    require(out.is_relative_to(root) and out.parent.is_dir() and not out.exists(), "Exclusive qualification file under existing evidence ancestor")
    tracking = out.with_name(out.stem + "-assembly")
    tracking.mkdir(exist_ok=False)
    begin, phase = time.monotonic(), "admission"
    try:
        write(tracking / "started.json", {"status": "started", "scope": "saved engineering qualification only",
            "request": request, "automatic_retry": False, "new_model_calls": 0, "new_native_calls": 0})
        require(type(request) is dict and set(request) == {"measurement", "repaired_audit", "failed_measurement", "failed_audit", "processes", "sizing_review", "caps"}, "Exact explicit qualification inputs")
        base = original(root)
        measurement_path, measurement = read_reference(base, root, request["measurement"])
        require(request["measurement"]["sha256"] == MEASUREMENT_SHA, "Pinned completed capacity measurement")
        audit_path, audited = read_reference(base, root, request["repaired_audit"])
        require(measurement["status"] == audited["status"] == "completed"
                and measurement["engineering"] is audited["engineering"] is True
                and audited["version"] == "reacher-two-observation-capacity-audit-repair-v1"
                and audited["execution_completed_sha256"] == MEASUREMENT_SHA
                and audited["source_sha256"] == measurement["source_sha256"]
                and audited["profile_source_sha256"] == measurement["profile_source_sha256"]
                and audited["runtime"] == measurement["runtime"], "Completed repaired audit binds unchanged measurement")
        require(measurement["profile_source_sha256"][ORIGINAL] == ORIGINAL_SHA
                and measurement["profile_source_sha256"][ORIGINAL_TEST] == ORIGINAL_TEST_SHA, "True measured constructor/order source versions retained")
        require(len(measurement["source_sha256"]) == 116
                and 0 < measurement["wall_seconds"] < measurement["cap_seconds"]
                and 0 < audited["wall_seconds"] < audited["cap_seconds"]
                and audited["new_model_calls"] == audited["new_optimizer_steps"] == 0, "Complete source count and bounded saved-only phases")
        trees = [(measurement_path.parent, checked_tree(base, measurement_path.parent, measurement, "completed.json")),
                 (audit_path.parent, checked_tree(base, audit_path.parent, audited, "completed.json"))]
        metrics = base.read(audit_path.parent / "measurement-audit.json")
        require(metrics["whole_measurement"] == measurement and metrics["audit_repair"] == audited["audit_repair"]
                and metrics["native_control_transitions_checked"] == 44800
                and metrics["native_nominal_candidate_transitions_checked"] == 26247168
                and metrics["native_nominal_selected_transitions_checked"] == 9600
                and metrics["new_model_calls"] == metrics["new_optimizer_steps"] == 0, "All measured rows independently replayed with no neural work")
        require(len(metrics["rows"]) == 14 and all(all(row[key] == value for key, value in expected.items())
                for row, expected in zip(metrics["rows"], base.row_manifest(), strict=True)), "Exact fourteen audited row identities")
        failed_path, failed = read_reference(base, root, request["failed_measurement"])
        require(failed["status"] == "failed" and failed["exception_type"] == "ImportError"
                and "GRUResidualRewardWorldModel" in failed["error"] and failed["automatic_retry"] is False, "Original pre-numerical import failure retained")
        old_binding = base.read(failed_path.parent / "source-binding.json")
        old_files = base.inventory(failed_path.parent, float("inf"))
        require(set(old_files) == {"started.json", "source-binding.json", "failed.json"}
                | {"source-snapshot/" + name for name in old_binding["sources"]}, "No numerical work hidden in failed first attempt")
        for name, digest in old_binding["sources"].items():
            base.bound(failed_path.parent, "source-snapshot/" + name, digest)
        trees.append((failed_path.parent, old_files))
        failure_path, failed_audit = read_reference(base, root, request["failed_audit"])
        previous = audited["audit_repair"]["previous_failed_audit"]
        require(all(previous[key] == request["failed_audit"][key] for key in ("path", "sha256")), "Original failed audit is same one admitted by repair")
        failure_files = base.inventory(failure_path.parent, float("inf"))
        require(failure_files == previous["files"] and failed_audit["status"] == "failed", "Failed audit tree unchanged")
        trees.append((failure_path.parent, failure_files))
        processes, wrappers = process_receipts(base, root, request["processes"], measurement, audited, old_binding, failed_audit,
            locations={"attempt01": failed_path.parent, "attempt02": measurement_path.parent, "repair": audit_path.parent})
        trees.extend((row["folder"], row["files"]) for row in processes.values())
        review_path, review = read_reference(base, root, request["sizing_review"])
        projected = review_sizing(review, measurement_sha=MEASUREMENT_SHA, audit_sha=request["repaired_audit"]["sha256"],
            sources=measurement["source_sha256"], runtime=measurement["runtime"], caps=request["caps"])
        from openjev.research import reacher_two_observation_experiment as experiment
        from openjev.research import reacher_two_observation_protocol as protocol
        require(experiment.runtime() == measurement["runtime"], "Current runtime equals actual measurements")
        for name, digest in measurement["profile_source_sha256"].items(): base.bound(root, name, digest)
        phase = "retained_members"
        sources = wrappers | {name: sha(root / name) for name in (HELPER, TEST_HELPER)}
        for name, digest in sources.items():
            source = base.bound(root, name, digest)
            target = tracking / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, target.open("xb") as saved: shutil.copyfileobj(incoming, saved)
            require(sha(target) == digest, "Qualification/wrapper source snapshot")
        files = {}
        for folder, members in trees:
            for name, value in members.items():
                path = folder / name
                require(path.is_relative_to(out.parent), "All retained artifacts must descend from qualification parent")
                relative = path.relative_to(out.parent).as_posix()
                require(relative not in files, "No overlapping retained evidence trees")
                files[relative] = value["sha256"]
        require(review_path.is_relative_to(out.parent), "Independent review retained under qualification parent")
        files[review_path.relative_to(out.parent).as_posix()] = request["sizing_review"]["sha256"]
        files.update({(tracking / name).relative_to(out.parent).as_posix(): value["sha256"]
                      for name, value in base.inventory(tracking, float("inf")).items()})
        require(sum(Path(name).name == "measurement-audit.json" for name in files) == 1, "Exactly one repaired measurement audit in qualification")
        template = base.read(audit_path.parent / "qualification-template.json")
        components = copy_components(metrics["measured_components"], audit_path.parent, out.parent)
        require(all(row["artifact"] in files for row in components.values()), "All measured component artifacts retained")
        require(template["full_shape"] == {key: metrics["settings"][key] for key in ("hidden_size", "train_episodes", "batch_size", "steps", "control_episodes", "planning_horizon", "action_block")}
                and template["coverage"] == protocol.coverage(metrics["settings"]), "Exact target shape/coverage")
        result = {"schema": "reacher-two-observation-capacity-qualification-v1", "status": "completed", "engineering": True,
            "study": protocol.STUDY, "source_sha256": measurement["source_sha256"], "runtime": measurement["runtime"],
            "files": files, "wall_seconds": sum(row["wall_seconds"] for row in processes.values()),
            "full_shape": template["full_shape"], "coverage": template["coverage"], "measured_components": components,
            "projected_seconds": projected, "review_sha256": request["sizing_review"]["sha256"], "approved_caps": request["caps"],
            "inputs": request, "actual_process_phases": {key: row["phases"] for key, row in processes.items()},
            "qualification_assembly_seconds": time.monotonic() - begin, "qualification_source_sha256": sources,
            "original_profile_source_sha256": measurement["profile_source_sha256"],
            "repair_source_sha256": audited["audit_repair"]["repair_source_sha256"],
            "new_model_calls": 0, "new_native_calls": 0, "new_optimizer_steps": 0,
            "scope": "Engineering feasibility and reviewed sizing only; all failures retained. No scientific effectiveness or launch authorization."}
        write(out, result)
        return {"path": str(out), "sha256": sha(out), "members": len(files)}
    except BaseException as error:
        try:
            if out.exists(): out.rename(out.with_name("invalid-" + out.name))
            write(tracking / "failed.json", {"status": "failed", "phase": phase, "error": repr(error),
                "wall_seconds": time.monotonic() - begin, "automatic_retry": False})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Failed qualification could not be retained: {preservation_error!r}")
        raise


def copy_components(components, audit_folder, ancestor):
    require(set(components) == {"training", "learned_control", "physics_references", "audit", "storage_and_hashing"}, "All five measured workload components")
    result = {}
    for key, row in components.items():
        require(set(row) == {"wall_seconds", "units", "artifact"} and type(row["units"]) is int and row["units"] > 0, "Actual measured component schema")
        require(row["artifact"] == ("training-audit.json" if key == "training" else "measurement-audit.json"), "Exact existing measured component artifact")
        number(row["wall_seconds"], key + " measured wall")
        result[key] = {**row, "artifact": (audit_folder / row["artifact"]).relative_to(ancestor).as_posix()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--authorize-saved-qualification", action="store_true")
    args = parser.parse_args()
    require(args.authorize_saved_qualification, "Explicit saved qualification authorization required")
    require(sha(args.request) == args.request_sha256, "External request bytes")
    result = assemble(json.loads(args.request.read_text()), args.out, root=args.root, authorize_saved_qualification=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
