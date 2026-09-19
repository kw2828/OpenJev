"""Independent synthetic evidence fixtures; no learned inference or training."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "audit_reward_fixture", ROOT / "scripts/audit_reacher_reward_residual_study.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
base = audit.base
DIGEST = "a" * 64


def plan_fixture():
    seeds = [271, 283, 293]
    return {
        "steps": 50,
        "kinds": ["free", "residual"],
        "fit_seeds": seeds,
        "fit_order": ["free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293"],
        "train_episodes": 2,
        "prediction_episodes": 1,
        "control_episodes": 1,
        "epochs": 2,
        "batch_size": 1,
        "hidden_size": 4,
        "stochastic_size": 2,
        "width": 8,
        "window": 3,
        "rollout_horizon": 5,
        "reward_scale": 4.0,
        "rollout_weight": 0.5,
        "kl_weight": 0.01,
        "noise_std": 0.05,
        "ordinary_gap": 6,
        "shift_gap": 10,
        "prediction_seed": 2101,
        "control_seed": 3101,
        "schedule_seed": 4101,
        "noise_seed": 5101,
        "exploration_seed": 6101,
        "candidate_seed": 7101,
        "planning_horizon": 12,
        "action_block": 3,
        "candidates": 2,
        "cap_seconds": 100.0,
        "filter_seed": 9101,
        "bootstrap_seed": 8101,
        "bootstrap_samples": 32,
        "sources": {"synthetic.py": "b" * 64},
        "runtime": {"synthetic": True},
    }


def fake_record(plan, split, index=0, panel="ordinary"):
    stream = {"train": 0, "prediction": 100000, "control": 200000}[split]
    valid = base.expected_schedule(plan, index if split == "control" else stream + index, panel)
    packets = np.zeros((51, 8), np.float32)
    packets[:, :2] = 1
    packets[:, 6] = valid
    packets[~np.array(valid), :4] = 0
    record = {
        "policy": {"packets": packets, "commands": np.zeros((50, 2), np.float32)},
        "audit": {
            key: np.zeros(shape)
            for key, shape in {
                "qpos": (51, 4),
                "qvel": (51, 4),
                "raw_obs": (51, 10),
                "integration_state": (51, 49),
                "time": (51,),
                "rewards": (50,),
                "reward_dist": (50,),
                "reward_ctrl": (50,),
                "applied_actions": (50, 2),
                "actuator_noise": (50, 2),
            }.items()
        },
        "metadata": {
            "seed": plan[f"{split}_seed"] + index,
            "noise_seed": plan["noise_seed"] + stream + index,
            "noise_std": plan["noise_std"],
            "sensor_schedule": valid,
            "policy_keys": ["packets", "commands"],
        },
    }
    record["audit"]["raw_obs"][:, :2] = 1
    record["audit"]["rewards"][:] = -1
    if split != "control":
        action_seed = plan["exploration_seed"] + stream + index
        mode = str(
            np.random.default_rng(action_seed).choice(
                ["ik_pd", "random_low", "random_high"], p=[0.5, 0.25, 0.25]
            )
        )
        record["metadata"].update(
            action_seed=action_seed,
            requested_policy="mixed",
            collector_policy=mode,
            action_hold=4,
            exploration_std={"ik_pd": 0.12, "random_low": 0.3, "random_high": 0.8}[mode],
        )
    return record


def save_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{category}__{key}": np.stack([r[category][key] for r in records])
        for category in ("policy", "audit")
        for key in records[0][category]
    }
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    base.write(path.with_suffix(".json"), [r["metadata"] for r in records])


def bind_source(tmp_path, plan, execution):
    parent = tmp_path / "parent"
    parent.mkdir()
    old = copy.deepcopy(plan)
    old.update(
        train_seed=101,
        prediction_seed=201,
        control_seed=301,
        schedule_seed=401,
        noise_seed=501,
        exploration_seed=601,
        candidate_seed=701,
        filter_seed=901,
    )
    old["fit_seeds"] = [211, 223, 239]
    base.write(parent / "plan.json", old)
    records = [fake_record(old, "train", i) for i in range(plan["train_episodes"])]
    save_records(parent / "train", records)
    members = {name: base.sha(parent / name) for name in ("train.npz", "train.json")}
    base.write(parent / "completed.json", {"status": "completed", "files": members})
    base.write(
        parent / "audit-receipt.json",
        {
            "status": "completed",
            "plan_sha256": base.sha(parent / "plan.json"),
            "execution_completed_sha256": base.sha(parent / "completed.json"),
            "execution_members": members,
        },
    )
    source = {
        "execution_path": str(parent),
        "plan_path": str(parent / "plan.json"),
        "plan_sha256": base.sha(parent / "plan.json"),
        "audit_receipt_path": str(parent / "audit-receipt.json"),
        "audit_receipt_sha256": base.sha(parent / "audit-receipt.json"),
        "members": members,
        "cohort_plan": {key: old[key] for key in audit.COHORT_FIELDS},
    }
    plan["training_source"] = source
    base.write(execution / "train-provenance.json", source)
    for name in members:
        (execution / name).write_bytes((parent / name).read_bytes())
    return records


def rebind(execution, *, nested=False):
    if nested:
        for folder in (execution / "fits").iterdir():
            receipt = base.read(folder / "completed.json")
            receipt["files"] = {
                name: base.sha(folder / name) for name in audit.FIT_FILES if (folder / name).exists()
            }
            if (folder / "initial-weights.pt").exists():
                receipt["initial_weights_sha256"] = base.sha(folder / "initial-weights.pt")
            base.write(folder / "completed.json", receipt)
        boundary = base.read(execution / "all-fits-completed.json")
        boundary["files"] = {name: base.sha(execution / name) for name in boundary["files"]}
        base.write(execution / "all-fits-completed.json", boundary)
        evaluation = base.read(execution / "evaluation-started.json")
        evaluation["all_fits_completed_sha256"] = base.sha(execution / "all-fits-completed.json")
        base.write(execution / "evaluation-started.json", evaluation)
    receipt = base.read(execution / "completed.json")
    receipt["files"] = {
        str(path.relative_to(execution)): base.sha(path)
        for path in execution.rglob("*")
        if path.is_file() and path != execution / "completed.json"
    }
    base.write(execution / "completed.json", receipt)


def build_fixture(tmp_path, plan):
    execution = tmp_path / "execution"
    execution.mkdir()
    train = bind_source(tmp_path, plan, execution)
    base.write(execution / "started.json", {"plan_sha256": DIGEST, "unix_time": 1.0})
    save_records(execution / "prediction", [fake_record(plan, "prediction")])
    packets = base.stack(train, "policy", "packets")
    batches = 2
    for name in plan["fit_order"]:
        kind, seed = name.rsplit("-", 1)
        seed = int(seed)
        folder = execution / "fits" / name
        folder.mkdir(parents=True)
        initial = {
            key: torch.full(shape, seed / 1000) for key, shape in base.checkpoint_shapes(plan, "gru").items()
        }
        torch.save(initial, folder / "initial-weights.pt")
        torch.save({key: value + 0.01 for key, value in initial.items()}, folder / "weights.pt")
        row = dict.fromkeys(base.LOG_KEYS, 0.0)
        row.update(
            loss=1.0,
            observation_mse=1.0,
            gradient_norm=1.0,
            valid_observation_targets=float((packets[:, 1:, 6] > 0.5).sum() / batches),
            valid_rollout_starts=float((packets[:, :46, 6] > 0.5).sum() / batches),
        )
        base.write(folder / "training.json", [row | {"epoch": i + 1} for i in range(plan["epochs"])])
        order = torch.Generator().manual_seed(seed + 4100000)
        digest = hashlib.sha256(
            b"".join(
                torch.randperm(plan["train_episodes"], generator=order).numpy().tobytes()
                for _ in range(plan["epochs"])
            )
        ).hexdigest()
        base.write(
            folder / "completed.json",
            {
                "kind": kind,
                "seed": seed,
                "updates": 4,
                "parameters": sum(x.numel() for x in initial.values()),
                "wall_seconds": 0.001,
                "residual_reward": kind == "residual",
                "noise_std": 0.05,
                "minibatch_order_sha256": digest,
                "initial_weights_sha256": base.sha(folder / "initial-weights.pt"),
                "files": {file: base.sha(folder / file) for file in audit.FIT_FILES},
            },
        )
        (execution / "predictions").mkdir(exist_ok=True)
        np.savez_compressed(
            execution / "predictions" / f"{name}.npz",
            one=np.ones((1, 50, 4)),
            reward=-np.ones((1, 50)),
            multi=np.ones((1, 46, 4)),
        )
    base.write(
        execution / "all-fits-completed.json",
        {
            "plan_sha256": DIGEST,
            "fit_order": plan["fit_order"],
            "files": {
                f"fits/{name}/completed.json": base.sha(execution / "fits" / name / "completed.json")
                for name in plan["fit_order"]
            },
            "unix_time": 1.2,
            "elapsed_seconds": 0.2,
        },
    )
    base.write(
        execution / "evaluation-started.json",
        {
            "plan_sha256": DIGEST,
            "all_fits_completed_sha256": base.sha(execution / "all-fits-completed.json"),
            "unix_time": 1.21,
            "elapsed_seconds": 0.21,
        },
    )
    for panel in base.PANELS:
        for arm in base.arms(plan, panel):
            record = fake_record(plan, "control", panel=panel)
            if arm == "uniform":
                record["policy"]["commands"] = (
                    np.random.default_rng(plan["candidate_seed"] + 200000)
                    .uniform(-1, 1, (50, 1, 2))[:, 0]
                    .astype(np.float32)
                )
            path = execution / "control" / panel / arm
            save_records(path, [record])
            values = {
                "candidate_scores": np.zeros((1, 50, 2)),
                "batch_decision_seconds": np.full(50, 0.00001),
                "planner_used": np.array(arm not in ("zero", "uniform")),
                "setup_seconds": np.array(0.0001),
            }
            if arm not in base.REFERENCES:
                values.update(
                    selected_predicted_angles=np.zeros((1, 50, 4)),
                    selected_predicted_reward=np.zeros((1, 50)),
                )
            np.savez_compressed(path.parent / f"{arm}-planning.npz", **values)
    base.write(
        execution / "completed.json",
        {
            "status": "completed",
            "plan_sha256": DIGEST,
            "fits": 6,
            "astra_calls": 0,
            "wall_seconds": 1.0,
            "evaluation_started_elapsed_seconds": 0.21,
            "files": {},
        },
    )
    rebind(execution)
    return execution


@pytest.fixture
def saved(tmp_path, monkeypatch):
    plan = plan_fixture()
    execution = build_fixture(tmp_path, plan)
    checked = []

    def replay(record):
        checked.append(record)
        return {"transitions": 50, "max_abs_error": 0.0, "saved_output_only": True, "new_policy_calls": 0}

    monkeypatch.setattr(base, "native_replay", replay)
    return plan, execution, tmp_path / "audit", checked


def test_complete_every_fit_and_reset_cohort_with_old_train_fresh_evaluation(saved):
    plan, execution, out, checked = saved
    summary = audit.audit_saved(plan, DIGEST, execution, out)
    assert len(checked) == 45 and summary["native_transitions_checked"] == 2250
    assert len(summary["fits"]) == 6 and len(summary["continuation_gate"]["checks"]) == 17
    assert set(summary["control"]["full"]) == set(base.arms(plan, "full"))
    assert len(summary["control"]["ordinary"]) == 16
    assert list(summary["fits"]) == plan["fit_order"]
    assert all(item["initial_tensors_identical"] for item in summary["paired_training"].values())
    assert summary["training_source"] == plan["training_source"]
    assert summary["new_model_calls"] == summary["new_policy_calls"] == summary["new_mpc_calls"] == 0
    receipt = base.read(out / "receipt.json")
    assert receipt["files"]["summary.json"] == base.sha(out / "summary.json")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_initial", "member set"),
        ("extra_checkpoint", "member set"),
        ("skip_config", "member set"),
        ("paired_initial", "Paired initial tensors"),
        ("bad_order", "Minibatch order"),
        ("mode_bool", "Reward-head configuration"),
        ("mode_integer", "Reward-head configuration"),
        ("noise", "Reward noise configuration"),
        ("updates", "update count"),
        ("kl", "GRU KL"),
        ("weights_shape", "checkpoint weight"),
        ("weights_nan", "checkpoint weight"),
    ],
)
def test_training_and_checkpoint_corruption_survives_rehash_but_not_audit(saved, mutation, match):
    plan, execution, out, _ = saved
    folder = execution / "fits/residual-283"
    if mutation == "missing_initial":
        (folder / "initial-weights.pt").unlink()
    elif mutation == "extra_checkpoint":
        torch.save({}, folder / "best-weights.pt")
    elif mutation == "skip_config":
        (execution / "predictions/residual-283.npz").unlink()
    elif mutation in ("paired_initial", "weights_shape", "weights_nan"):
        path = folder / ("initial-weights.pt" if mutation == "paired_initial" else "weights.pt")
        state = torch.load(path, weights_only=True)
        key = next(iter(state))
        if mutation == "paired_initial":
            state[key] = state[key] + 0.1
        if mutation == "weights_shape":
            state[key] = state[key].flatten()
        if mutation == "weights_nan":
            state[key].fill_(float("nan"))
        torch.save(state, path)
    elif mutation == "kl":
        rows = base.read(folder / "training.json")
        rows[0]["kl_nats"] = 1.0
        base.write(folder / "training.json", rows)
    else:
        receipt = base.read(folder / "completed.json")
        key, value = {
            "bad_order": ("minibatch_order_sha256", "f" * 64),
            "mode_bool": ("residual_reward", False),
            "mode_integer": ("residual_reward", 1),
            "noise": ("noise_std", 0.1),
            "updates": ("updates", 3),
        }[mutation]
        receipt[key] = value
        base.write(folder / "completed.json", receipt)
    rebind(execution, nested=mutation not in ("missing_initial", "extra_checkpoint", "skip_config"))
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("copied_bytes", "Copied training member hash"),
        ("provenance", "Copied training provenance"),
        ("cohort", "Parent cohort plan binding"),
        ("parent_plan", "Parent plan hash"),
        ("parent_receipt", "Parent audit receipt hash"),
        ("old_seed_for_prediction", "prediction seed"),
        ("new_seed_for_train", "Copied training member hash"),
        ("boundary_hash", "Evaluation boundary hash"),
        ("boundary_order", "Fit boundary order"),
        ("boundary_time", "Fit/evaluation chronology"),
        ("boundary_members", "Fit boundary membership"),
        ("evaluation_time", "Completion evaluation timing"),
        ("cap", "cap exceeded"),
        ("hash", "member hash"),
    ],
)
def test_provenance_stream_and_boundary_corruption(saved, mutation, match):
    plan, execution, out, _ = saved
    if mutation in ("copied_bytes", "new_seed_for_train"):
        path = execution / "train.json"
        value = base.read(path)
        value[0]["seed"] += 2000
        base.write(path, value)
    elif mutation == "provenance":
        value = base.read(execution / "train-provenance.json")
        value["noise"] = "wrong"
        base.write(execution / "train-provenance.json", value)
    elif mutation == "cohort":
        plan["training_source"]["cohort_plan"]["noise_seed"] += 1
        base.write(execution / "train-provenance.json", plan["training_source"])
    elif mutation == "parent_plan":
        Path(plan["training_source"]["plan_path"]).write_text("{}")
    elif mutation == "parent_receipt":
        Path(plan["training_source"]["audit_receipt_path"]).write_text("{}")
    elif mutation == "old_seed_for_prediction":
        path = execution / "prediction.json"
        value = base.read(path)
        value[0]["seed"] = 201
        base.write(path, value)
    elif mutation.startswith("boundary_"):
        if mutation == "boundary_hash":
            path = execution / "evaluation-started.json"
            value = base.read(path)
            value["all_fits_completed_sha256"] = "f" * 64
        else:
            path = execution / "all-fits-completed.json"
            value = base.read(path)
            if mutation == "boundary_order":
                value["fit_order"] = value["fit_order"][::-1]
            if mutation == "boundary_time":
                value["elapsed_seconds"] = 0.3
            if mutation == "boundary_members":
                value["files"].pop(next(iter(value["files"])))
        base.write(path, value)
        if mutation != "boundary_hash":
            evaluation = base.read(execution / "evaluation-started.json")
            evaluation["all_fits_completed_sha256"] = base.sha(path)
            base.write(execution / "evaluation-started.json", evaluation)
    elif mutation == "evaluation_time":
        path = execution / "completed.json"
        value = base.read(path)
        value["evaluation_started_elapsed_seconds"] = 0.22
        base.write(path, value)
    elif mutation == "cap":
        path = execution / "completed.json"
        value = base.read(path)
        value["wall_seconds"] = 101.0
        base.write(path, value)
    elif mutation == "hash":
        (execution / "prediction.json").write_text("[]")
    if mutation != "hash":
        rebind(execution)
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()


def passing_metrics(plan):
    controls = {
        panel: {arm: {"mean_cost": 100.0, "episode_costs": [100.0]} for arm in base.arms(plan, panel)}
        for panel in base.PANELS
    }
    for panel in base.PANELS:
        controls[panel]["known_state"] = {"mean_cost": 90.0, "episode_costs": [90.0]}
        for seed in plan["fit_seeds"]:
            controls[panel][f"residual-{seed}"] = {"mean_cost": 90.0, "episode_costs": [90.0]}
            controls[panel][f"residual-{seed}-reset"] = (
                {"mean_cost": 1.0, "episode_costs": [1.0]} if panel != "full" else None
            )
        if panel == "full":
            for seed in plan["fit_seeds"]:
                controls[panel].pop(f"residual-{seed}-reset")
    predictions = {
        name: {
            "reward_mse": 0.9 if name.startswith("residual") else 1.0,
            "one_step": {"mse": 1.05 if name.startswith("residual") else 1.0},
        }
        for name in base.configurations(plan)
    }
    return predictions, controls


def test_exact_thresholds_and_reset_results_are_not_gate_inputs():
    plan = plan_fixture()
    predictions, controls = passing_metrics(plan)
    gate, _ = audit.qualify(plan, predictions, controls)
    assert gate["passed"] and len(gate["checks"]) == 17 and gate["reset_is_descriptive_only"]
    assert all(item["passed"] for item in gate["checks"])
    controls["shift"]["residual-293"] = {"mean_cost": 100.0, "episode_costs": [100.0]}
    gate, _ = audit.qualify(plan, predictions, controls)
    assert not gate["passed"]
    assert not next(x for x in gate["checks"] if x["name"] == "residual-293/shift/vs_paired_free")["passed"]


@pytest.mark.parametrize("criterion", ["zero", "pair", "family", "reward", "angle", "physics"])
def test_one_requirement_failure_cannot_be_hidden_by_other_seeds_or_panels(criterion):
    plan = plan_fixture()
    predictions, controls = passing_metrics(plan)
    if criterion == "zero":
        controls["shift"]["residual-283"] = {"mean_cost": 90.000001, "episode_costs": [90.000001]}
    if criterion == "pair":
        controls["shift"]["free-283"] = {"mean_cost": 90.0, "episode_costs": [90.0]}
    if criterion == "family":
        for seed in plan["fit_seeds"]:
            controls["shift"][f"free-{seed}"] = {"mean_cost": 94.0, "episode_costs": [94.0]}
    if criterion == "reward":
        predictions["residual-283"]["reward_mse"] = 0.900001
    if criterion == "angle":
        predictions["residual-283"]["one_step"]["mse"] = 1.050001
    if criterion == "physics":
        controls["ordinary"]["known_state"]["mean_cost"] = 90.000001
    gate, _ = audit.qualify(plan, predictions, controls)
    assert not gate["passed"]


def test_missing_nonfinite_metrics_rejected_and_private_shuffle_does_not_advance_global_rng():
    plan = plan_fixture()
    predictions, controls = passing_metrics(plan)
    predictions["residual-271"]["reward_mse"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite qualification"):
        audit.qualify(plan, predictions, controls)
    predictions, controls = passing_metrics(plan)
    predictions.pop("free-293")
    with pytest.raises(ValueError, match="Qualification coverage"):
        audit.qualify(plan, predictions, controls)
    before = torch.random.get_rng_state().clone()
    assert len(audit.minibatch_order_sha256(plan, 271)) == 64
    assert torch.equal(torch.random.get_rng_state(), before)


def test_native_failure_and_output_reuse_are_not_suppressed(saved, monkeypatch):
    plan, execution, out, _ = saved

    def fail(_record):
        raise ValueError("Native replay mismatch")

    monkeypatch.setattr(base, "native_replay", fail)
    with pytest.raises(ValueError, match="Native replay mismatch"):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        audit.audit_saved(plan, DIGEST, execution, out)


@pytest.mark.parametrize("namespace", ["noise", "schedule", "exploration", "candidate", "filter"])
def test_parent_stochastic_stream_reuse_is_rejected(saved, namespace):
    plan, execution, out, _ = saved
    parent = base.read(plan["training_source"]["plan_path"])
    plan[namespace + "_seed"] = parent[namespace + "_seed"]
    with pytest.raises(ValueError, match="stochastic streams overlap"):
        audit.audit_saved(plan, DIGEST, execution, out)


def test_parent_paths_resolve_against_repository_not_process_cwd(saved, tmp_path, monkeypatch):
    plan, execution, out, _ = saved
    # Use a synthetic root containing a source folder, then move the process elsewhere.
    monkeypatch.setattr(audit, "ROOT", execution.parent)
    source = plan["training_source"]
    for key in ("execution_path", "plan_path", "audit_receipt_path"):
        source[key] = str(Path(source[key]).relative_to(execution.parent))
    base.write(execution / "train-provenance.json", source)
    rebind(execution)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    summary = audit.audit_saved(plan, DIGEST, execution, out)
    assert summary["status"] == "completed"
