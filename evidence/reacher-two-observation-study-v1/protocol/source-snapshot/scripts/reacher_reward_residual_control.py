"""Fresh evaluation of every unchanged final fit from the stopped residual pilot."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import reacher_reward_residual_study as legacy
import torch

from openjev.research.reacher_random_streams import (
    FIELDS,
    concrete_streams,
    domain_bases,
    generator_manifest,
    validate_separation,
)
from openjev.research.robotics_reacher import collect_episode

ROOT = Path(__file__).resolve().parents[1]
base = legacy.base
sha, write, check_cap = base.sha, base.write, base.check_cap
STUDY = "reacher-reward-residual-control-v2"
SOURCE_PLAN = "evidence/reacher-reward-residual-v1/protocol/plan.json"
SOURCE_PLAN_SHA = "df9929ea6320e24ba32ec8e8fdd84f88f8435ed03db1bea655ec31d82345e0fd"
INVALID_PATH = "evidence/reacher-reward-residual-v1/invalid.json"
INVALID_SHA = "eeb892b92a45521816bdfdf873ca16547ed9794f3daf53f0607e54a91ff3daee"
INHERITANCE_PATH = "evidence/reacher-reward-residual-v1/fit-inheritance.json"
INHERITANCE_SHA = "c867c57768556e17467438d8f390f1de2ee2e959ca1ed5513d9c96223a57cebb"
SOURCES = legacy.SOURCES + (
    "scripts/reacher_reward_residual_control.py",
    "scripts/audit_reacher_reward_residual_control.py",
    "src/openjev/research/reacher_random_streams.py",
    "tests/test_reacher_random_streams.py",
    "tests/test_reacher_reward_residual_control.py",
    "tests/test_audit_reacher_reward_residual_control.py",
)
CHANGED_PARENT_FIELDS = {"version", "study", "sources", "runtime", "stop", "data", "models", "control", *FIELDS}
NEW_FIELDS = {"fit_source", "random_stream_contract"}
INTENTIONAL_REUSE = [
    "All fits share one prediction corpus; copied training is the sole inherited data cohort.",
    "Every control arm and sensing panel shares case reset and disturbance streams.",
    "Ordinary/shift share a phase draw; full sensing allocates but does not consume it.",
    "All models/panels share each decision's batched candidate bank; cases occupy distinct draws in that bank.",
    "Particle filters share corresponding case child streams across panels; children are distinct by purpose.",
    "Uniform reference streams are shared across sensing panels; paired bootstrap indices are reused across comparisons.",
]


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, f"Inherited identity mismatch: {path}")
    return path


def authenticate_fit_source(source):
    """Read only pre-evaluation fit identities and the stopped terminal receipt."""
    require(set(source) == {
        "execution_path", "plan_path", "plan_sha256", "invalid_receipt_path", "invalid_receipt_sha256",
        "inheritance_receipt_path", "inheritance_receipt_sha256", "all_fits_boundary_sha256",
        "failure_receipt_sha256", "members", "inherited_fit_wall_seconds", "prior_invalid_attempt_wall_seconds",
    }, "Fit source schema")
    parent = legacy.validate(checked(ROOT / source["plan_path"], source["plan_sha256"]), source["plan_sha256"])
    invalid = read(checked(ROOT / source["invalid_receipt_path"], source["invalid_receipt_sha256"]))
    inheritance = read(checked(ROOT / source["inheritance_receipt_path"], source["inheritance_receipt_sha256"]))
    original = ROOT / source["execution_path"]
    require(original.is_dir() and not original.is_symlink() and not (original / "completed.json").exists(),
            "Original attempt must remain stopped, not completed")
    failed = read(checked(original / "failed.json", source["failure_receipt_sha256"]))
    boundary = read(checked(original / "all-fits-completed.json", source["all_fits_boundary_sha256"]))
    require(invalid["status"] == "invalid_protocol_stopped" and invalid["resumable"] is False
            and invalid["process_exit_code"] == 130 and invalid["plan_sha256"] == source["plan_sha256"],
            "Invalid attempt identity/state")
    require(invalid["failure_receipt_sha256"] == source["failure_receipt_sha256"]
            and invalid["all_six_fits_completed_sha256"] == source["all_fits_boundary_sha256"],
            "Invalid attempt receipt binding")
    require(inheritance["status"] == "all_six_fits_authenticated_for_unselected_inheritance"
            and inheritance["source_plan_sha256"] == source["plan_sha256"]
            and inheritance["source_invalid_receipt_sha256"] == source["invalid_receipt_sha256"]
            and inheritance["new_fits"] == 0 and inheritance["new_model_calls"] == 0
            and inheritance["paired_initial_tensors_identical"] is True
            and inheritance["training_roles_have_no_seed_collisions"] is True
            and inheritance["efficacy_metrics_read"] is False,
            "Fit inheritance identity/scope")
    names = parent["fit_order"]
    require(boundary["fit_order"] == names and boundary["plan_sha256"] == source["plan_sha256"],
            "Original all-fit boundary/order")
    members = {f"fits/{name}/{file}" for name in names
               for file in ("initial-weights.pt", "weights.pt", "training.json", "completed.json")}
    require(set(source["members"]) == members and source["members"] == invalid["completed_fit_members"],
            "Every original final fit must be inherited")
    receipts = {f"fits/{name}/completed.json": inheritance["fit_receipts_sha256"][name] for name in names}
    require(boundary["files"] == receipts and set(inheritance["fits"]) == set(names), "All-fit receipt coverage")
    for member, digest in source["members"].items():
        checked(original / member, digest)
    fit_wall = 0.0
    for name in names:
        receipt = read(original / "fits" / name / "completed.json")
        require(source["members"][f"fits/{name}/completed.json"] == receipts[f"fits/{name}/completed.json"],
                "Inherited fit receipt hash")
        for file, digest in receipt["files"].items():
            require(source["members"][f"fits/{name}/{file}"] == digest, "Inherited nested fit hash")
        fit_wall += receipt["wall_seconds"]
    require(math.isfinite(fit_wall) and fit_wall > 0
            and fit_wall == source["inherited_fit_wall_seconds"] == inheritance["inherited_fit_wall_seconds"],
            "Inherited fit cost binding")
    require(source["prior_invalid_attempt_wall_seconds"] == invalid["wall_seconds"] == failed["wall_seconds"]
            and fit_wall <= boundary["elapsed_seconds"] <= failed["wall_seconds"], "Prior attempt cost/chronology")
    return parent


def default_fit_source():
    invalid = read(checked(ROOT / INVALID_PATH, INVALID_SHA))
    inheritance = read(checked(ROOT / INHERITANCE_PATH, INHERITANCE_SHA))
    source = {
        "execution_path": "runs/reacher-reward-residual-v1/execution",
        "plan_path": SOURCE_PLAN, "plan_sha256": SOURCE_PLAN_SHA,
        "invalid_receipt_path": INVALID_PATH, "invalid_receipt_sha256": INVALID_SHA,
        "inheritance_receipt_path": INHERITANCE_PATH, "inheritance_receipt_sha256": INHERITANCE_SHA,
        "all_fits_boundary_sha256": invalid["all_six_fits_completed_sha256"],
        "failure_receipt_sha256": invalid["failure_receipt_sha256"],
        "members": invalid["completed_fit_members"],
        "inherited_fit_wall_seconds": inheritance["inherited_fit_wall_seconds"],
        "prior_invalid_attempt_wall_seconds": invalid["wall_seconds"],
    }
    authenticate_fit_source(source)
    return source


def stream_contract(plan):
    source = plan["fit_source"]
    priors = [
        {"plan_path": plan["training_source"]["plan_path"],
         "plan_sha256": plan["training_source"]["plan_sha256"], "include_train": True},
        {"plan_path": source["plan_path"], "plan_sha256": source["plan_sha256"], "include_train": False},
    ]
    registries = [concrete_streams(read(checked(ROOT / item["plan_path"], item["plan_sha256"])),
                                  include_train=item["include_train"]) for item in priors]
    registry = validate_separation(plan, prior_registries=registries)
    return {"namespace": STUDY, "registry": registry, "generators": generator_manifest(registry),
            "priors": priors, "intentional_reuse": INTENTIONAL_REUSE, "draws_for_manifest": 0}


def validate_parent_settings(plan, parent):
    require(set(plan) == set(parent) | NEW_FIELDS, "Recovery plan field membership")
    for key, value in parent.items():
        if key not in CHANGED_PARENT_FIELDS:
            require(plan[key] == value, f"Frozen parent setting changed: {key}")
    require(plan["study"] == STUDY and plan["version"] == 2, "Recovery study/version")
    require(all(plan[field] == seed for field, seed in domain_bases(STUDY).items()), "Prospective domain seed binding")


def prepare(out, *, fit_source=None):
    source = default_fit_source() if fit_source is None else fit_source
    parent = authenticate_fit_source(source)
    plan = dict(parent)
    plan.update(version=2, study=STUDY, sources={name: sha(ROOT / name) for name in SOURCES},
                runtime=base.runtime(), fit_source=source, **domain_bases(STUDY))
    plan.update(
        data="Unchanged authenticated training corpus and ALL final fits from the stopped v1. Fresh 96 prediction and 64 paired control cases. Complete role-expanded seed/state manifest excludes both previous evaluations and original training. No selection from partial evaluation.",
        models="Evaluation only: load every inherited final checkpoint, no training, warm start, replacement seed or epoch selection. Preserve paired initialization and minibatch order; charge inherited fitting separately from new evaluation.",
        control="Original full/ordinary/shift panels, memory-reset diagnostics, four references and identical 64-candidate 12-step planner. All inherited checkpoints authenticated and loaded before fresh data or model inference. Same seventeen useful-effect criteria.",
        stop="New evaluation-only attempt with one1800second cooperative cap from function entry through validation, copying, model loading, prediction data, all prediction/control and final member hashing. No retries, resumes, extensions or selection. Stopped v1 is never resumed. Include all inherited fitting costs separately and prior invalid attempt cost without doublecount. Audit only saved outputs/native replay.",
    )
    plan["random_stream_contract"] = stream_contract(plan)
    validate_parent_settings(plan, parent)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, expected):
    plan = read(checked(path, expected))
    require(set(plan["sources"]) == set(SOURCES), "Frozen source membership")
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    require(plan["runtime"] == base.runtime(), "Frozen runtime changed")
    parent = authenticate_fit_source(plan["fit_source"])
    validate_parent_settings(plan, parent)
    require(plan["random_stream_contract"] == stream_contract(plan), "Random stream manifest changed")
    return plan


def copy_verified(source, destination, digest, deadline):
    check_cap(deadline)
    checked(source, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "Inherited destination already exists")
    shutil.copyfile(source, destination)
    checked(destination, digest)


def run(path, expected, out):
    start, started_unix = time.monotonic(), time.time()
    plan = validate(path, expected)
    out.mkdir(parents=True, exist_ok=False)
    deadline = start + plan["cap_seconds"]
    torch.set_num_threads(plan["threads"])
    torch.use_deterministic_algorithms(True)
    write(out / "started.json", {"plan_sha256": expected, "unix_time": started_unix})
    progress = {"phase": "inherit-training-and-fits"}
    try:
        train_source, source = plan["training_source"], plan["fit_source"]
        for name, digest in train_source["members"].items():
            copy_verified(ROOT / train_source["execution_path"] / name, out / name, digest, deadline)
        write(out / "train-provenance.json", train_source)
        for name, digest in source["members"].items():
            copy_verified(ROOT / source["execution_path"] / name, out / name, digest, deadline)
        for filename, relative, digest in (
            ("inherited-plan.json", source["plan_path"], source["plan_sha256"]),
            ("invalid-attempt.json", source["invalid_receipt_path"], source["invalid_receipt_sha256"]),
            ("fit-inheritance.json", source["inheritance_receipt_path"], source["inheritance_receipt_sha256"]),
            ("source-all-fits-completed.json", str(Path(source["execution_path"]) / "all-fits-completed.json"), source["all_fits_boundary_sha256"]),
            ("source-failed.json", str(Path(source["execution_path"]) / "failed.json"), source["failure_receipt_sha256"]),
        ):
            copy_verified(ROOT / relative, out / filename, digest, deadline)
        write(out / "fit-provenance.json", source)
        write(out / "random-streams.json", plan["random_stream_contract"])
        models = {}
        for name in plan["fit_order"]:
            check_cap(deadline)
            kind, seed = name.rsplit("-", 1)
            model = legacy.model_for(plan, kind, int(seed))
            model.load_state_dict(torch.load(out / "fits" / name / "weights.pt", map_location="cpu", weights_only=True), strict=True)
            model.requires_grad_(False)
            models[name] = model.eval()
        write(out / "inherited-fits-ready.json", {
            "plan_sha256": expected, "fit_order": list(models),
            "files": {f"fits/{name}/completed.json": sha(out / "fits" / name / "completed.json") for name in models},
            "unix_time": time.time(), "elapsed_seconds": time.monotonic() - start,
            "inherited_fit_wall_seconds": source["inherited_fit_wall_seconds"], "new_fits": 0,
        })
        evaluation_start = time.monotonic() - start
        write(out / "evaluation-started.json", {
            "plan_sha256": expected, "inherited_fits_ready_sha256": sha(out / "inherited-fits-ready.json"),
            "random_streams_sha256": sha(out / "random-streams.json"),
            "unix_time": time.time(), "elapsed_seconds": evaluation_start,
        })
        prediction = []
        for i in range(plan["prediction_episodes"]):
            progress = {"phase": "prediction-data", "episode": i}
            check_cap(deadline)
            prediction.append(collect_episode(plan["prediction_seed"] + i, base.schedule(plan, 100000 + i),
                noise_std=plan["noise_std"], noise_seed=plan["noise_seed"] + 100000 + i,
                action_seed=plan["exploration_seed"] + 100000 + i, policy="mixed"))
        base.save_records(out / "prediction", prediction)
        (out / "predictions").mkdir()
        for name, model in models.items():
            progress = {"phase": "prediction", "model": name}
            np.savez_compressed(out / "predictions" / f"{name}.npz",
                **base.prediction_record(plan, model, prediction, deadline))
        for panel in ("full", "ordinary", "shift"):
            folder = out / "control" / panel
            folder.mkdir(parents=True, exist_ok=False)
            for name, model in models.items():
                for reset in ((False,) if panel == "full" else (False, True)):
                    label = name + ("-reset" if reset else "")
                    progress = {"phase": "control", "panel": panel, "model": label}
                    records, extra = base.learned_control(plan, model, panel, reset, deadline, progress)
                    base.save_records(folder / label, records)
                    np.savez_compressed(folder / f"{label}-planning.npz", **extra)
                    print(json.dumps({"completed_control": f"{panel}/{label}"}), flush=True)
            for arm in ("known_state", "particle", "zero", "uniform"):
                progress = {"phase": "control", "panel": panel, "model": arm}
                records, extra = base.reference_control(plan, panel, arm, deadline, progress)
                base.save_records(folder / arm, records)
                np.savez_compressed(folder / f"{arm}-planning.npz", **extra)
                print(json.dumps({"completed_control": f"{panel}/{arm}"}), flush=True)
        validate(path, expected)
        members = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        check_cap(deadline)
        write(out / "completed.json", {
            "status": "completed", "plan_sha256": expected, "fits": len(models), "new_fits": 0, "astra_calls": 0,
            "wall_seconds": time.monotonic() - start, "evaluation_started_elapsed_seconds": evaluation_start,
            "inherited_fit_wall_seconds": source["inherited_fit_wall_seconds"],
            "prior_invalid_attempt_wall_seconds": source["prior_invalid_attempt_wall_seconds"], "files": members,
        })
    except BaseException as error:
        write(out / "failed.json", {"error": repr(error), "wall_seconds": time.monotonic() - start,
                                    "progress": progress, "new_fits": 0})
        raise


def audit(path, expected, execution, out):
    from audit_reacher_reward_residual_control import audit_saved
    return audit_saved(validate(path, expected), expected, execution, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.out)))
    elif args.mode == "run":
        run(args.plan, args.expected_plan_sha256, args.out)
    else:
        result = audit(args.plan, args.expected_plan_sha256, args.execution, args.out)
        print(json.dumps({"status": result["status"], "continuation_gate": result["continuation_gate"]["passed"]}))


if __name__ == "__main__":
    main()
