"""Synthetic saved-output checks; no model training or planner calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_reacher_world_model_study.py"
spec = importlib.util.spec_from_file_location("audit_reacher_fixture", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
DIGEST = "a" * 64


def fixture_plan():
    return {
        "steps": 50,
        "kinds": ["history", "gru", "rssm"],
        "fit_seeds": [211, 223, 239],
        "train_episodes": 1,
        "prediction_episodes": 1,
        "control_episodes": 1,
        "epochs": 1,
        "batch_size": 1,
        "rollout_horizon": 5,
        "reward_scale": 4.0,
        "kl_weight": 0.01,
        "rollout_weight": 0.5,
        "ordinary_gap": 6,
        "shift_gap": 10,
        "noise_std": 0.05,
        "train_seed": 101,
        "prediction_seed": 201,
        "control_seed": 301,
        "schedule_seed": 401,
        "noise_seed": 501,
        "exploration_seed": 601,
        "candidate_seed": 701,
        "planning_horizon": 12,
        "action_block": 3,
        "candidates": 2,
        "cap_seconds": 100.0,
        "bootstrap_seed": 801,
        "bootstrap_samples": 64,
        "hidden_size": 4,
        "stochastic_size": 2,
        "window": 3,
        "width": 8,
        "sources": {"audit.py": "b" * 64},
        "runtime": {"synthetic": True},
    }


def record(plan, split, panel="ordinary"):
    stream = {"train": 0, "prediction": 100000, "control": 200000}[split]
    valid = audit.expected_schedule(plan, 0 if split == "control" else stream, panel)
    packets = np.zeros((51, 8), np.float32)
    packets[:, 0] = packets[:, 1] = 1
    packets[:, 6] = valid
    packets[~np.array(valid), :4] = 0
    result = {
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
            "seed": plan[f"{split}_seed"],
            "noise_seed": plan["noise_seed"] + stream,
            "noise_std": plan["noise_std"],
            "sensor_schedule": valid,
            "policy_keys": ["packets", "commands"],
        },
    }
    result["audit"]["raw_obs"][:, :2] = 1
    result["audit"]["rewards"][:] = -1
    if split != "control":
        seed = plan["exploration_seed"] + stream
        mode = str(
            np.random.default_rng(seed).choice(["ik_pd", "random_low", "random_high"], p=[0.5, 0.25, 0.25])
        )
        result["metadata"].update(
            action_seed=seed,
            requested_policy="mixed",
            collector_policy=mode,
            action_hold=4,
            exploration_std={"ik_pd": 0.12, "random_low": 0.3, "random_high": 0.8}[mode],
        )
    return result


def save_records(base, records):
    base.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{category}__{key}": np.stack([r[category][key] for r in records])
        for category in ("policy", "audit")
        for key in records[0][category]
    }
    np.savez_compressed(base.with_suffix(".npz"), **arrays)
    audit.write(base.with_suffix(".json"), [r["metadata"] for r in records])


def rebind(execution):
    complete = audit.read(execution / "completed.json")
    complete["files"] = {
        str(p.relative_to(execution)): audit.sha(p)
        for p in execution.rglob("*")
        if p.is_file() and p.name != "completed.json"
    }
    # Nested fit completion receipts must also be included.
    for p in (execution / "fits").glob("*/completed.json"):
        complete["files"][str(p.relative_to(execution))] = audit.sha(p)
    audit.write(execution / "completed.json", complete)


def build_fixture(execution, plan):
    execution.mkdir()
    audit.write(execution / "started.json", {"plan_sha256": DIGEST, "unix_time": 1.0})
    train, prediction = record(plan, "train"), record(plan, "prediction")
    save_records(execution / "train", [train])
    save_records(execution / "prediction", [prediction])
    for name in audit.configurations(plan):
        folder = execution / "fits" / name
        folder.mkdir(parents=True)
        kind, seed = name.rsplit("-", 1)
        weights = {key: torch.ones(shape) for key, shape in audit.checkpoint_shapes(plan, kind).items()}
        torch.save(weights, folder / "weights.pt")
        row = dict.fromkeys(audit.LOG_KEYS, 0.0)
        row.update(
            epoch=1,
            loss=1.0,
            observation_mse=1.0,
            gradient_norm=1.0,
            valid_observation_targets=float(train["policy"]["packets"][1:, 6].sum()),
            valid_rollout_starts=float(train["policy"]["packets"][:46, 6].sum()),
        )
        audit.write(folder / "training.json", [row])
        kind, seed = name.rsplit("-", 1)
        audit.write(
            folder / "completed.json",
            {
                "kind": kind,
                "seed": int(seed),
                "updates": 1,
                "parameters": sum(x.numel() for x in weights.values()),
                "wall_seconds": 0.001,
                "files": {p.name: audit.sha(p) for p in folder.iterdir()},
            },
        )
        (execution / "predictions").mkdir(exist_ok=True)
        np.savez_compressed(
            execution / "predictions" / f"{name}.npz",
            one=np.ones((1, 50, 4)),
            reward=-np.ones((1, 50)),
            multi=np.ones((1, 46, 4)),
        )
    for panel in audit.PANELS:
        for arm in audit.arms(plan, panel):
            item = record(plan, "control", panel)
            if arm == "uniform":
                item["policy"]["commands"] = (
                    np.random.default_rng(plan["candidate_seed"] + 200000)
                    .uniform(-1, 1, (50, 1, 2))[:, 0]
                    .astype(np.float32)
                )
            base = execution / "control" / panel / arm
            save_records(base, [item])
            values = {
                "candidate_scores": np.zeros((1, 50, plan["candidates"])),
                "batch_decision_seconds": np.full(50, 0.00001),
                "planner_used": np.array(arm not in ("zero", "uniform")),
                "setup_seconds": np.array(0.0001),
            }
            if arm not in audit.REFERENCES:
                values.update(
                    selected_predicted_angles=np.zeros((1, 50, 4)),
                    selected_predicted_reward=np.zeros((1, 50)),
                )
            np.savez_compressed(base.parent / f"{arm}-planning.npz", **values)
    audit.write(
        execution / "completed.json",
        {
            "status": "completed",
            "plan_sha256": DIGEST,
            "wall_seconds": 1.0,
            "fits": 9,
            "astra_calls": 0,
            "files": {},
        },
    )
    rebind(execution)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    plan = fixture_plan()
    execution = tmp_path / "execution"
    build_fixture(execution, plan)
    checked = []

    def replay(item):
        checked.append(item)
        return {"transitions": 50, "max_abs_error": 0.0, "saved_output_only": True, "new_policy_calls": 0}

    monkeypatch.setattr(audit, "native_replay", replay)
    return plan, execution, tmp_path / "audit", checked


def test_complete_all_fits_all_trajectories_and_receipt(saved):
    plan, execution, out, checked = saved
    result = audit.audit_saved(plan, DIGEST, execution, out)
    assert len(checked) == 53
    assert result["native_transitions_checked"] == 2650
    assert len(result["fits"]) == 9
    assert len(result["continuation_gate"]["checks"]) == 27
    assert not result["continuation_gate"]["passed"]
    receipt = audit.read(out / "receipt.json")
    assert receipt["files"]["summary.json"] == audit.sha(out / "summary.json")
    assert len(receipt["execution_members"]) == len(audit.expected_members(plan))
    assert result["new_model_calls"] == result["new_policy_calls"] == result["new_mpc_calls"] == 0


def test_prediction_masks_and_causal_persistence():
    plan = fixture_plan()
    item = record(plan, "prediction")
    packets = item["policy"]["packets"]
    packets[:, :4] = 0
    packets[:, 6] = 1
    packets[0, :4] = 2
    packets[1:3, 6] = 0
    packets[3:, :4] = 3
    before = audit.persistence_prefix(packets[None])
    assert np.all(before[0, :3] == 2)
    changed = packets.copy()
    changed[4:, :4] = 999
    np.testing.assert_equal(audit.persistence_prefix(changed[None])[:, :4], before[:, :4])
    values = {"one": np.zeros((1, 50, 4)), "reward": -np.ones((1, 50)), "multi": np.zeros((1, 46, 4))}
    metrics = audit.prediction_metrics(plan, [item], values)
    assert metrics["one_step"] == {"mse": 9.0, "targets": 48, "components": 192}
    assert metrics["open_loop"]["targets"] == 46
    assert metrics["open_loop_valid_root_and_endpoint"]["targets"] == 44
    assert metrics["blackout_angle"]["targets"] == 2
    assert metrics["reward_mse"] == 0


def test_candidate_bank_matches_independent_full_generation():
    plan = fixture_plan() | {"candidates": 11}
    for t in (0, 42, 49):
        h = min(12, 50 - t)
        raw = np.random.default_rng(plan["candidate_seed"] + t).normal(0, 0.25, (3, 11, math_ceil(h / 3), 2))
        raw[:, 5:] *= 3
        bank = np.clip(np.repeat(raw, 3, axis=2)[:, :, :h], -1, 1).astype(np.float32)
        bank[:, 0] = 0
        for k, a in enumerate(((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.2, 0.2), (-0.2, -0.2)), 1):
            bank[:, k] = a
        np.testing.assert_equal(audit.candidate_first_actions(plan, t, 3), bank[:, :, 0])


def math_ceil(value):
    return int(np.ceil(value))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("hash", "member hash"),
        ("extra", "member set"),
        ("missing", "member set"),
        ("plan", "completion identity"),
        ("cap", "cap exceeded"),
        ("nonfinite_time", "Nonfinite JSON"),
        ("noise_seed", "noise seed"),
        ("schedule", "sensor schedule"),
        ("collector", "Exploration mixture"),
        ("updates", "update count"),
        ("weights", "checkpoint weight"),
        ("loss", "loss arithmetic"),
        ("command", "disagrees"),
        ("floor_score", "placeholder"),
        ("timing", "finite/timing"),
        ("prediction", "Prediction shape"),
        ("on_policy", "On-policy prediction"),
    ],
)
def test_tampered_saved_outputs_rejected_even_after_rehash(saved, mutation, match):
    plan, execution, out, _ = saved
    if mutation == "hash":
        (execution / "train.json").write_text("[]")
    elif mutation == "extra":
        (execution / "unexpected").write_text("x")
        rebind(execution)
    elif mutation == "missing":
        (execution / "predictions/gru-211.npz").unlink()
        rebind(execution)
    elif mutation in ("plan", "cap", "nonfinite_time"):
        path = execution / "completed.json"
        item = audit.read(path)
        item.update(
            {
                "plan": {"plan_sha256": "f" * 64},
                "cap": {"wall_seconds": 101},
                "nonfinite_time": {"wall_seconds": float("nan")},
            }[mutation]
        )
        path.write_text(json.dumps(item))
    elif mutation in ("noise_seed", "schedule", "collector"):
        path = execution / "train.json"
        item = audit.read(path)
        if mutation == "noise_seed":
            item[0]["noise_seed"] += 1
        if mutation == "schedule":
            item[0]["sensor_schedule"][7] = False
        if mutation == "collector":
            item[0]["exploration_std"] = 9
        audit.write(path, item)
        rebind(execution)
    elif mutation in ("updates", "weights", "loss"):
        folder = execution / "fits/gru-211"
        complete = audit.read(folder / "completed.json")
        if mutation == "updates":
            complete["updates"] += 1
        if mutation == "weights":
            weights = torch.load(folder / "weights.pt", weights_only=True)
            next(iter(weights.values())).fill_(float("nan"))
            torch.save(weights, folder / "weights.pt")
        if mutation == "loss":
            item = audit.read(folder / "training.json")
            item[0]["loss"] = 99
            audit.write(folder / "training.json", item)
        complete["files"] = {name: audit.sha(folder / name) for name in complete["files"]}
        audit.write(folder / "completed.json", complete)
        rebind(execution)
    else:
        if mutation == "prediction":
            path = execution / "predictions/gru-211.npz"
        elif mutation == "command":
            path = execution / "control/ordinary/gru-211.npz"
        else:
            path = (
                execution
                / f"control/ordinary/{'zero' if mutation == 'floor_score' else 'gru-211'}-planning.npz"
            )
        item = audit.load_npz(path)
        if mutation == "prediction":
            item["multi"] = np.zeros((1, 45, 4))
        if mutation == "command":
            item["policy__commands"][0, 0] = 0.4
        if mutation == "floor_score":
            item["candidate_scores"][:] = 1
        if mutation == "timing":
            item["batch_decision_seconds"][0] = -1
        if mutation == "on_policy":
            item["selected_predicted_angles"] = np.zeros((1, 50, 3))
        np.savez_compressed(path, **item)
        rebind(execution)
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()


def test_gate_uses_every_seed_and_both_shift_panels():
    plan = fixture_plan()
    controls = {
        panel: {arm: {"mean_cost": 100.0, "episode_costs": [100.0]} for arm in audit.arms(plan, panel)}
        for panel in audit.PANELS
    }
    predictions = {
        name: {"one_step": {"mse": 0.1}, "persistence_one_step": {"mse": 1.0}}
        for name in audit.configurations(plan)
    }
    for panel in ("ordinary", "shift"):
        controls[panel]["known_state"].update(mean_cost=70.0, episode_costs=[70.0])
        for kind in ("gru", "rssm"):
            for seed in plan["fit_seeds"]:
                controls[panel][f"{kind}-{seed}"].update(mean_cost=80.0, episode_costs=[80.0])
    gate, comparisons = audit.qualify(plan, predictions, controls)
    assert gate["passed"] and all(x["passed"] for x in gate["checks"])
    assert comparisons["gru"]["shift"]["vs_history"]["mean_cost_difference"] == -20
    controls["shift"]["gru-239"]["mean_cost"] = 99
    predictions["rssm-223"]["one_step"]["mse"] = 1
    gate, _ = audit.qualify(plan, predictions, controls)
    assert not gate["passed"]


def test_component_timing_and_empty_audit_destination(saved):
    plan, execution, out, _ = saved
    item = audit.read(execution / "completed.json")
    item["wall_seconds"] = 0.000001
    audit.write(execution / "completed.json", item)
    with pytest.raises(ValueError, match="Component timings"):
        audit.audit_saved(plan, DIGEST, execution, out)
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        audit.audit_saved(plan, DIGEST, execution, out)


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_algebraic_checkpoint_schema_matches_frozen_constructor_without_forward(kind):
    from openjev.research.reacher_world_models import make_world_model

    plan = fixture_plan()
    with torch.random.fork_rng():
        model = make_world_model(
            kind, **{k: plan[k] for k in ("hidden_size", "stochastic_size", "window", "width")}
        )
    assert audit.checkpoint_shapes(plan, kind) == {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("junk", "membership"),
        ("dtype", "checkpoint weight"),
        ("shape", "checkpoint weight"),
        ("kl", "Non-RSSM KL"),
    ],
)
def test_rehashed_checkpoint_schema_and_kl_tamper(saved, mutation, match):
    plan, execution, out, _ = saved
    folder = execution / "fits/gru-211"
    state = torch.load(folder / "weights.pt", weights_only=True)
    key = next(iter(state))
    if mutation == "junk":
        state = {"junk": torch.zeros(sum(x.numel() for x in state.values()))}
    if mutation == "dtype":
        state[key] = state[key].double()
    if mutation == "shape":
        state[key] = state[key].flatten()
    if mutation == "kl":
        logs = audit.read(folder / "training.json")
        logs[0]["kl_nats"] = 1.0
        logs[0]["loss"] += plan["kl_weight"]
        audit.write(folder / "training.json", logs)
    else:
        torch.save(state, folder / "weights.pt")
    completed = audit.read(folder / "completed.json")
    completed["files"] = {name: audit.sha(folder / name) for name in completed["files"]}
    audit.write(folder / "completed.json", completed)
    rebind(execution)
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("range", "score range"),
        ("reward", "score/reward"),
        ("setup", "setup timing"),
        ("flag", "Planning shape/identity"),
    ],
)
def test_planning_algebra_and_setup_cannot_be_forged_by_rehash(saved, mutation, match):
    plan, execution, out, _ = saved
    path = execution / "control/ordinary/gru-211-planning.npz"
    values = audit.load_npz(path)
    if mutation == "range":
        values["candidate_scores"][:] = 1.0
    if mutation == "reward":
        values["selected_predicted_reward"][:, -1] = -1.0
    if mutation == "setup":
        values["setup_seconds"] = np.array(-1.0)
    if mutation == "flag":
        values["planner_used"] = np.array(1)
    np.savez_compressed(path, **values)
    rebind(execution)
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)


def test_native_replay_failures_are_not_silently_skipped(saved, monkeypatch):
    plan, execution, out, _ = saved

    def reject(_record):
        raise ValueError("Native raw_obs replay mismatch")

    monkeypatch.setattr(audit, "native_replay", reject)
    with pytest.raises(ValueError, match="raw_obs replay mismatch"):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()
