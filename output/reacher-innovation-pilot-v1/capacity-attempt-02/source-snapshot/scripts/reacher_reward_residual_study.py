"""Matched six-fit test of a known actuator-cost term in learned planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import reacher_world_model_study as base
import torch

from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import sequence_loss
from openjev.research.robotics_reacher import collect_episode

ROOT = Path(__file__).resolve().parents[1]
SOURCES = base.SOURCES + (
    "scripts/reacher_reward_residual_study.py",
    "scripts/audit_reacher_reward_residual_study.py",
    "src/openjev/research/reacher_reward_residual.py",
    "tests/test_reacher_reward_residual.py",
    "tests/test_reacher_reward_residual_study.py",
    "tests/test_audit_reacher_reward_residual_study.py",
)
PARENT_PLAN = "evidence/reacher-world-model-v1/protocol/plan.json"
PARENT_AUDIT = "evidence/reacher-world-model-v1/audit/receipt.json"
PARENT_PLAN_SHA = "10b9d22aaaa178c8293908f1d096a29ed7994760c997372dc865d240c224133b"
PARENT_AUDIT_SHA = "da24bee8bf7030e9e8a75f3baef5421b6d05b81483483cbdc7258a98098abb27"
COHORT_KEYS = ("train_episodes", "steps", "train_seed", "schedule_seed", "noise_seed",
               "exploration_seed", "noise_std", "ordinary_gap", "shift_gap")
sha, write, check_cap = base.sha, base.write, base.check_cap


def authenticate_training(source):
    plan_path = ROOT / source["plan_path"]
    audit_path = ROOT / source["audit_receipt_path"]
    if sha(plan_path) != source["plan_sha256"] or sha(audit_path) != source["audit_receipt_sha256"]:
        raise ValueError("Training parent identity mismatch")
    parent = json.loads(plan_path.read_text())
    receipt = json.loads(audit_path.read_text())
    if receipt["status"] != "completed" or receipt["plan_sha256"] != source["plan_sha256"]:
        raise ValueError("Training parent audit incomplete/mismatched")
    if source["cohort_plan"] != {key: parent[key] for key in COHORT_KEYS}:
        raise ValueError("Training parent cohort settings mismatch")
    if set(source["members"]) != {"train.npz", "train.json"}:
        raise ValueError("Training parent member set mismatch")
    for name, digest in source["members"].items():
        path = ROOT / source["execution_path"] / name
        if path.is_symlink() or sha(path) != digest or receipt["execution_members"][name] != digest:
            raise ValueError("Training parent member changed")


def default_training_source():
    parent = json.loads((ROOT / PARENT_PLAN).read_text())
    receipt = json.loads((ROOT / PARENT_AUDIT).read_text())
    source = {
        "execution_path": "runs/reacher-world-model-v1/execution",
        "plan_path": PARENT_PLAN, "plan_sha256": PARENT_PLAN_SHA,
        "audit_receipt_path": PARENT_AUDIT, "audit_receipt_sha256": PARENT_AUDIT_SHA,
        "members": {name: receipt["execution_members"][name] for name in ("train.npz", "train.json")},
        "cohort_plan": {key: parent[key] for key in COHORT_KEYS},
    }
    authenticate_training(source)
    return source


def prepare(out, *, training_source=None):
    source = default_training_source() if training_source is None else training_source
    authenticate_training(source)
    parent = json.loads((ROOT / source["plan_path"]).read_text())
    plan = dict(parent)
    plan.update(
        version=1, study="reacher-reward-residual-v1",
        sources={name: sha(ROOT / name) for name in SOURCES}, runtime=base.runtime(),
        training_source=source, kinds=["free", "residual"], fit_seeds=[271, 283, 293],
        fit_order=["free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293"],
        epochs=48, prediction_episodes=96, control_episodes=64,
        prediction_seed=66200001, control_seed=66300001, schedule_seed=66400001,
        noise_seed=66500001, exploration_seed=66600001, candidate_seed=66700001,
        filter_seed=66800001, bootstrap_seed=66900001,
        data="Copy the original authenticated 768 training episodes bit-for-bit; native total reward and observed-angle targets only. Parent cohort settings are explicit. Generate fresh 96 prediction and64 control episodes with new environment, schedule, noise and action streams. No evaluation selection of data or checkpoint.",
        models="Two parameter-identical GRUs: free total-reward head versus learned residual minus expected clipped Gaussian actuator cost. Known command/noise scale only. Same seed-paired initialization, minibatch order, optimizer and48epochs/1152updates. Final checkpoint only; fixed interleaved fit order. Analytical term costs are charged. Conventional reward structure, no novel RL or connectome claim.",
        control="Same64 paired fresh cases and candidate banks under full, six-step and ten-step gaps. Both arms also reset memory at gap onset in ordinary/shift as diagnostics. Same12-step64-candidate MPC, no terminal value, four original supplied-physics/floor references. All six final fits before prediction/control evaluation.",
        criteria="All results complete. Known-state physics must reduce ordinary cost by10% versus zero. Every residual fit must reduce ordinary AND shift cost by10% versus zero. Residual family mean must reduce cost by5% versus free in both, and each paired seed must strictly improve both. Mean heldout reward MSE must reduce by10%; mean one-step angle MSE must not exceed1.05times free. All checks required. Resets descriptive only; success earns a strong-history memory comparison, not a connectome sweep or novel architecture claim.",
        stop="One1800second cooperative whole-run cap, including copying/preparation, training and all evaluation. No retries, resumes, replacement seeds, cap extensions or checkpoint selection. Save phase counters on failure. Audit replays saved native transitions but makes no new training, learned-model, policy or MPC calls.",
    )
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, expected):
    if sha(path) != expected:
        raise ValueError("External frozen plan identity mismatch")
    plan = json.loads(path.read_text())
    if set(plan["sources"]) != set(SOURCES):
        raise ValueError("Frozen source membership mismatch")
    for name, digest in plan["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if plan["runtime"] != base.runtime():
        raise ValueError("Frozen runtime changed")
    names = {f"{kind}-{seed}" for kind in plan["kinds"] for seed in plan["fit_seeds"]}
    if (plan["kinds"] != ["free", "residual"] or len(plan["fit_order"]) != len(names)
            or set(plan["fit_order"]) != names or len(set(plan["fit_seeds"])) != len(plan["fit_seeds"])):
        raise ValueError("Frozen fit membership/order invalid")
    cohort = plan["training_source"]["cohort_plan"]
    if any(plan[key] != cohort[key] for key in ("train_episodes", "steps", "noise_std", "ordinary_gap")):
        raise ValueError("Training task changed")
    authenticate_training(plan["training_source"])
    return plan


def model_for(plan, kind, seed):
    if kind not in ("free", "residual"):
        raise ValueError("Unknown reward-head arm")
    torch.manual_seed(seed)
    return GRUResidualRewardWorldModel(hidden_size=plan["hidden_size"],
        noise_std=plan["noise_std"], residual_reward=(kind == "residual"))


def fit(plan, kind, seed, data, out, deadline, progress):
    begin = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    model = model_for(plan, kind, seed).train()
    torch.save(model.state_dict(), out / "initial-weights.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    generator = torch.Generator().manual_seed(seed + 4100000)
    order_hash = hashlib.sha256()
    updates, logs = 0, []
    for epoch in range(plan["epochs"]):
        permutation = torch.randperm(len(data[0]), generator=generator)
        order_hash.update(permutation.numpy().tobytes())
        metrics = []
        for indices in permutation.split(plan["batch_size"]):
            progress.update(epoch=epoch + 1, completed_updates=updates)
            check_cap(deadline)
            optimizer.zero_grad(set_to_none=True)
            loss, values = sequence_loss(model, *(x[indices] for x in data), **{
                key: plan[key] for key in ("rollout_horizon", "rollout_weight", "reward_scale",
                                          "kl_weight", "kl_balance", "free_nats")})
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            metrics.append({**values, "gradient_norm": float(norm)})
            updates += 1
        logs.append({"epoch": epoch + 1, **{key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}})
        write(out / "training.json", logs)
    torch.save(model.state_dict(), out / "weights.pt")
    members = {p.name: sha(p) for p in out.iterdir()}
    write(out / "completed.json", {
        "kind": kind, "seed": seed, "updates": updates,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_weights_sha256": members["initial-weights.pt"],
        "minibatch_order_sha256": order_hash.hexdigest(),
        "residual_reward": kind == "residual", "noise_std": plan["noise_std"],
        "wall_seconds": time.monotonic() - begin,
        "files": members,
    })
    return model.eval()


def run(path, expected, out):
    start = time.monotonic()
    started_unix = time.time()
    plan = validate(path, expected)
    out.mkdir(parents=True, exist_ok=False)
    deadline = start + plan["cap_seconds"]
    torch.set_num_threads(plan["threads"])
    torch.use_deterministic_algorithms(True)
    write(out / "started.json", {"plan_sha256": expected, "unix_time": started_unix})
    progress = {"phase": "copy-training"}
    try:
        source = plan["training_source"]
        for name, digest in source["members"].items():
            check_cap(deadline)
            shutil.copyfile(ROOT / source["execution_path"] / name, out / name)
            if sha(out / name) != digest:
                raise ValueError("Copied training data identity mismatch")
        write(out / "train-provenance.json", source)
        train = base.load_records(out / "train")
        prediction = []
        for i in range(plan["prediction_episodes"]):
            progress.update(phase="prediction-data", episode=i)
            check_cap(deadline)
            prediction.append(collect_episode(plan["prediction_seed"] + i, base.schedule(plan, 100000 + i),
                noise_std=plan["noise_std"], noise_seed=plan["noise_seed"] + 100000 + i,
                action_seed=plan["exploration_seed"] + 100000 + i, policy="mixed"))
        base.save_records(out / "prediction", prediction)
        tensors = base.learning_tensors(train)
        models = {}
        for name in plan["fit_order"]:
            kind, seed = name.rsplit("-", 1)
            progress = {"phase": "fit", "model": name}
            models[name] = fit(plan, kind, int(seed), tensors, out / "fits" / name, deadline, progress)
            print(json.dumps({"completed_fit": name}), flush=True)
        write(out / "all-fits-completed.json", {
            "plan_sha256": expected, "fit_order": list(models),
            "files": {f"fits/{name}/completed.json": sha(out / "fits" / name / "completed.json") for name in models},
            "unix_time": time.time(), "elapsed_seconds": time.monotonic() - start,
        })
        evaluation_start = time.monotonic() - start
        write(out / "evaluation-started.json", {
            "plan_sha256": expected, "all_fits_completed_sha256": sha(out / "all-fits-completed.json"),
            "unix_time": time.time(), "elapsed_seconds": evaluation_start,
        })
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
            "status": "completed", "plan_sha256": expected, "fits": len(models), "astra_calls": 0,
            "wall_seconds": time.monotonic() - start,
            "evaluation_started_elapsed_seconds": evaluation_start,
            "files": members,
        })
    except BaseException as error:
        write(out / "failed.json", {"error": repr(error), "wall_seconds": time.monotonic() - start,
                                    "progress": progress})
        raise


def audit(path, expected, execution, out):
    from audit_reacher_reward_residual_study import audit_saved

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
        result = prepare(args.out)
    elif args.mode == "run":
        result = run(args.plan, args.expected_plan_sha256, args.out)
    else:
        result = audit(args.plan, args.expected_plan_sha256, args.execution, args.out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
