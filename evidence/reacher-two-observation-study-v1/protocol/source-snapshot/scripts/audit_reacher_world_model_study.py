"""Recompute the frozen Reacher pilot from complete saved outputs only.

The caller authenticates the external plan, source membership and runtime. This
module binds those identities in its receipt, checks every saved member, replays
native transitions, and never imports or calls a learned model or controller.
Checkpoint/log checks authenticate reported training, not a training replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from openjev.research.robotics_reacher import native_replay

PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "zero", "uniform")
POLICY_KEYS = {"packets", "commands"}
AUDIT_KEYS = {
    "qpos",
    "qvel",
    "raw_obs",
    "integration_state",
    "time",
    "rewards",
    "reward_dist",
    "reward_ctrl",
    "applied_actions",
    "actuator_noise",
}
LOG_KEYS = {
    "epoch",
    "loss",
    "observation_mse",
    "reward_mse",
    "kl_nats",
    "rollout_observation_mse",
    "rollout_reward_mse",
    "gradient_norm",
    "valid_observation_targets",
    "valid_rollout_starts",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label, *, positive=False):
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        label,
    )
    return float(value)


def configurations(plan):
    return [f"{kind}-{seed}" for kind in plan["kinds"] for seed in plan["fit_seeds"]]


def arms(plan, panel):
    base = configurations(plan)
    resets = [name + "-reset" for name in base if not name.startswith("history-")]
    return base + (resets if panel != "full" else []) + list(REFERENCES)


def expected_members(plan):
    names = {"started.json", "train.npz", "train.json", "prediction.npz", "prediction.json"}
    for name in configurations(plan):
        names.update(f"fits/{name}/{file}" for file in ("weights.pt", "training.json", "completed.json"))
        names.add(f"predictions/{name}.npz")
    for panel in PANELS:
        for arm in arms(plan, panel):
            names.update(f"control/{panel}/{arm}{suffix}" for suffix in (".npz", ".json", "-planning.npz"))
    return names


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), f"Duplicate NPZ member: {path}")
        if keys is not None:
            require(set(saved.files) == set(keys), f"NPZ membership: {path}")
        result = {key: saved[key] for key in saved.files}
    for key, value in result.items():
        require(
            value.dtype.kind in "fibu" and np.isfinite(value).all(),
            f"Nonfinite/nonnumeric array: {path}:{key}",
        )
    return result


def load_records(base, count):
    meta = read(base.with_suffix(".json"))
    require(isinstance(meta, list) and len(meta) == count, f"Episode metadata count: {base}")
    keys = {f"policy__{k}" for k in POLICY_KEYS} | {f"audit__{k}" for k in AUDIT_KEYS}
    data = load_npz(base.with_suffix(".npz"), keys)
    require(all(v.ndim > 0 and len(v) == count for v in data.values()), f"Episode array count: {base}")
    records = []
    for i, item in enumerate(meta):
        record = {"policy": {}, "audit": {}, "metadata": item}
        for key, value in data.items():
            category, name = key.split("__", 1)
            record[category][name] = value[i]
        records.append(record)
    return records


def expected_schedule(plan, index, panel="ordinary"):
    valid = np.ones(plan["steps"] + 1, dtype=bool)
    if panel != "full":
        phase = int(np.random.default_rng(plan["schedule_seed"] + index).integers(0, 4))
        gap = plan["shift_gap"] if panel == "shift" else plan["ordinary_gap"]
        for start in (8 + phase, 28 + phase):
            valid[start : start + gap] = False
    return valid.tolist()


def audit_cohort(plan, records, split, panel="ordinary"):
    stream = {"train": 0, "prediction": 100000, "control": 200000}[split]
    transitions, max_error = 0, 0.0
    for i, record in enumerate(records):
        meta = record["metadata"]
        require(meta["seed"] == plan[f"{split}_seed"] + i, f"{split} seed identity")
        require(meta["noise_seed"] == plan["noise_seed"] + stream + i, f"{split} noise seed")
        require(meta["noise_std"] == plan["noise_std"], f"{split} actuator noise scale")
        schedule_index = i if split == "control" else stream + i
        require(
            meta["sensor_schedule"] == expected_schedule(plan, schedule_index, panel),
            f"{split} sensor schedule",
        )
        require(meta["policy_keys"] == ["packets", "commands"], "Policy input allowlist")
        if split != "control":
            action_seed = plan["exploration_seed"] + stream + i
            require(
                meta["action_seed"] == action_seed and meta["requested_policy"] == "mixed",
                "Exploration stream",
            )
            mode = str(
                np.random.default_rng(action_seed).choice(
                    ["ik_pd", "random_low", "random_high"], p=[0.5, 0.25, 0.25]
                )
            )
            require(
                meta["collector_policy"] == mode
                and meta["action_hold"] == 4
                and meta["exploration_std"] == {"ik_pd": 0.12, "random_low": 0.3, "random_high": 0.8}[mode],
                "Exploration mixture",
            )
        replay = native_replay(record)
        require(
            replay["transitions"] == plan["steps"]
            and replay["new_policy_calls"] == 0
            and replay["saved_output_only"] is True,
            "Native replay completeness",
        )
        transitions += replay["transitions"]
        max_error = max(max_error, replay["max_abs_error"])
    return {"episodes": len(records), "transitions": transitions, "max_abs_error": max_error}


def stack(records, category, key):
    return np.stack([r[category][key] for r in records])


def masked_mse(prediction, target, mask):
    require(prediction.shape == target.shape and mask.shape == target.shape[:-1], "MSE shape mismatch")
    count = int(mask.sum())
    return {
        "mse": float(np.mean((prediction[mask].astype(float) - target[mask]) ** 2)) if count else None,
        "targets": count,
        "components": count * target.shape[-1],
    }


def persistence_prefix(packets):
    last = packets[:, 0, :4].copy()
    saved = []
    for t in range(packets.shape[1] - 1):
        last = np.where(packets[:, t, 6:7] > 0.5, packets[:, t, :4], last)
        saved.append(last.copy())
    return np.stack(saved, 1)


def prediction_metrics(plan, records, values):
    n, steps, h = len(records), plan["steps"], plan["rollout_horizon"]
    shapes = {"one": (n, steps, 4), "reward": (n, steps), "multi": (n, steps - h + 1, 4)}
    require(
        set(values) == set(shapes)
        and all(values[k].shape == shape and np.isfinite(values[k]).all() for k, shape in shapes.items()),
        "Prediction shape/finite",
    )
    packets = stack(records, "policy", "packets")
    native_angles = stack(records, "audit", "raw_obs")[:, :, :4]
    valid = packets[:, :, 6] > 0.5
    persistence = persistence_prefix(packets)
    one_mask = valid[:, 1:]
    multi_mask = valid[:, h:]
    root_mask = multi_mask & valid[:, : steps - h + 1]
    return {
        "one_step": masked_mse(values["one"], packets[:, 1:, :4], one_mask),
        "persistence_one_step": masked_mse(persistence, packets[:, 1:, :4], one_mask),
        "open_loop": masked_mse(values["multi"], packets[:, h:, :4], multi_mask),
        "persistence_open_loop": masked_mse(persistence[:, : steps - h + 1], packets[:, h:, :4], multi_mask),
        "open_loop_valid_root_and_endpoint": masked_mse(values["multi"], packets[:, h:, :4], root_mask),
        "blackout_angle": masked_mse(values["one"], native_angles[:, 1:], ~one_mask),
        "reward_mse": float(
            np.mean((values["reward"].astype(float) - stack(records, "audit", "rewards")) ** 2)
        ),
    }


def checkpoint_shapes(plan, kind):
    """Frozen architecture schema derived algebraically; no model construction."""
    shapes = {}

    def linear(name, inputs, outputs):
        shapes[name + ".weight"] = (outputs, inputs)
        shapes[name + ".bias"] = (outputs,)

    def mlp(name, inputs, hidden, outputs):
        linear(name + ".0", inputs, hidden)
        linear(name + ".2", hidden, outputs)

    def gru(name, inputs, hidden):
        shapes[name + ".weight_ih"] = (3 * hidden, inputs)
        shapes[name + ".weight_hh"] = (3 * hidden, hidden)
        shapes[name + ".bias_ih"] = shapes[name + ".bias_hh"] = (3 * hidden,)

    hidden, width, stochastic = plan["hidden_size"], plan["width"], plan["stochastic_size"]
    if kind == "history":
        linear("encoder.0", 10 * plan["window"], width)
        linear("encoder.2", width, width)
        linear("observation_head", width, 4)
        linear("reward_head", width, 1)
    elif kind == "gru":
        gru("observation_update", 8, hidden)
        gru("transition", 6, hidden)
        mlp("observation_head", hidden, hidden, 4)
        mlp("reward_head", hidden + 2, hidden, 1)
    elif kind == "rssm":
        gru("transition", stochastic + 6, hidden)
        mlp("prior", hidden, hidden, 2 * stochastic)
        mlp("posterior", hidden + 8, hidden, 2 * stochastic)
        mlp("observation_head", hidden + stochastic, hidden, 4)
        mlp("reward_head", hidden + stochastic + 2, hidden, 1)
    else:
        raise ValueError("Unknown checkpoint kind")
    return shapes


def audit_fit(plan, execution, name, train):
    folder = execution / "fits" / name
    completed = read(folder / "completed.json")
    kind, seed = name.rsplit("-", 1)
    require(completed["kind"] == kind and completed["seed"] == int(seed), "Fit identity")
    batches = math.ceil(plan["train_episodes"] / plan["batch_size"])
    require(completed["updates"] == plan["epochs"] * batches, "Fit update count")
    require(set(completed["files"]) == {"weights.pt", "training.json"}, "Fit member set")
    for member, digest in completed["files"].items():
        require(sha(folder / member) == digest, "Fit member hash")
    finite_number(completed["wall_seconds"], "Fit timing", positive=True)
    state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    expected_shapes = checkpoint_shapes(plan, kind)
    require(isinstance(state, dict) and set(state) == set(expected_shapes), "Checkpoint state membership")
    for key, value in state.items():
        require(
            isinstance(key, str)
            and isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.dtype == torch.float32
            and tuple(value.shape) == expected_shapes[key]
            and torch.isfinite(value).all().item(),
            "Nonfinite/invalid checkpoint weight",
        )
    require(sum(v.numel() for v in state.values()) == completed["parameters"], "Checkpoint parameter count")
    logs = read(folder / "training.json")
    require(len(logs) == plan["epochs"], "Training epoch count")
    packets = stack(train, "policy", "packets")
    expected_targets = float((packets[:, 1:, 6] > 0.5).sum() / batches)
    expected_roots = float(
        (packets[:, : plan["steps"] - plan["rollout_horizon"] + 1, 6] > 0.5).sum() / batches
    )
    for epoch, row in enumerate(logs, 1):
        require(set(row) == LOG_KEYS and row["epoch"] == epoch, "Training epoch membership/order")
        for key, value in row.items():
            finite_number(value, f"Training finite metric: {key}")
        require(kind == "rssm" or row["kl_nats"] == 0, "Non-RSSM KL must be zero")
        expected = (
            row["observation_mse"]
            + plan["reward_scale"] * row["reward_mse"]
            + plan["kl_weight"] * row["kl_nats"]
            + plan["rollout_weight"]
            * (row["rollout_observation_mse"] + plan["reward_scale"] * row["rollout_reward_mse"])
        )
        require(math.isclose(row["loss"], expected, rel_tol=1e-5, abs_tol=1e-6), "Training loss arithmetic")
        require(
            math.isclose(row["valid_observation_targets"], expected_targets, abs_tol=1e-6)
            and math.isclose(row["valid_rollout_starts"], expected_roots, abs_tol=1e-6),
            "Training target count",
        )
    return {k: completed[k] for k in ("kind", "seed", "updates", "parameters", "wall_seconds")} | {
        "epochs": logs
    }


def candidate_first_actions(plan, step, count):
    """Reconstruct only the frozen candidate bank, never evaluate a candidate."""
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    chunks = math.ceil(horizon / plan["action_block"])
    bank = np.random.default_rng(plan["candidate_seed"] + step).normal(
        0.0, 0.25, (count, plan["candidates"], chunks, 2)
    )
    bank[:, plan["candidates"] // 2 :] *= 3
    first = np.clip(bank[:, :, 0], -1.0, 1.0).astype(np.float32)
    first[:, 0] = 0
    for k, action in enumerate(((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.2, 0.2), (-0.2, -0.2)), 1):
        if k < plan["candidates"]:
            first[:, k] = action
    return first


def control_metrics(plan, records, values, arm):
    n, steps, k = len(records), plan["steps"], plan["candidates"]
    learned = arm not in REFERENCES
    planned = arm not in ("zero", "uniform")
    keys = {"candidate_scores", "batch_decision_seconds", "planner_used", "setup_seconds"}
    if learned:
        keys |= {"selected_predicted_angles", "selected_predicted_reward"}
    require(set(values) == keys, "Planning membership")
    score, times, flag = (
        values[key] for key in ("candidate_scores", "batch_decision_seconds", "planner_used")
    )
    require(
        score.shape == (n, steps, k)
        and times.shape == (steps,)
        and flag.shape == ()
        and flag.dtype == np.bool_
        and bool(flag) == planned,
        "Planning shape/identity",
    )
    require(
        np.isfinite(score).all() and np.isfinite(times).all() and (times > 0).all(), "Planning finite/timing"
    )
    setup = values["setup_seconds"]
    require(setup.shape == () and np.isfinite(setup) and float(setup) > 0, "Planning setup timing")
    commands = stack(records, "policy", "commands")
    uniform = np.random.default_rng(plan["candidate_seed"] + 200000)
    for t in range(steps):
        if planned:
            bank = candidate_first_actions(plan, t, n)
            expected = bank[np.arange(n), score[:, t].argmax(1)]
        elif arm == "uniform":
            expected = uniform.uniform(-1.0, 1.0, (n, 2)).astype(np.float32)
        else:
            expected = np.zeros((n, 2), dtype=np.float32)
        require(
            np.array_equal(commands[:, t], expected), "Saved command disagrees with saved scores/floor stream"
        )
    if not planned:
        require(np.all(score == 0), "Floor placeholder scores must be zero")
    rewards = stack(records, "audit", "rewards")
    require(np.all(rewards <= 1e-12), "Native reward sign")
    costs = -rewards.sum(1)
    result = {
        "episode_costs": costs.tolist(),
        "mean_cost": float(costs.mean()),
        "planner_used": planned,
        "setup_seconds": float(setup),
        "decision_wall_seconds": float(times.sum()),
        "batch_latency_seconds": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "p95": float(np.quantile(times, 0.95)),
        },
        "per_case_amortized_seconds": float(times.mean() / n),
    }
    if learned:
        angles, predicted_reward = values["selected_predicted_angles"], values["selected_predicted_reward"]
        require(
            angles.shape == (n, steps, 4)
            and predicted_reward.shape == (n, steps)
            and np.isfinite(angles).all()
            and np.isfinite(predicted_reward).all(),
            "On-policy prediction shape/finite",
        )
        for t in range(steps):
            horizon = min(plan["planning_horizon"], steps - t)
            require(
                np.all(score[:, t] <= 2e-5) and np.all(score[:, t] >= -2.5 * horizon - 2e-5),
                "Learned candidate score range",
            )
            selected = score[:, t].max(1)
            first = np.clip(predicted_reward[:, t], -2.5, 0)
            require(
                np.all(selected <= first + 2e-5) and np.all(selected >= first - 2.5 * (horizon - 1) - 2e-5),
                "Selected score/reward arithmetic",
            )
        valid = stack(records, "policy", "packets")[:, 1:, 6] > 0.5
        actual = stack(records, "audit", "raw_obs")[:, 1:, :4]
        result["on_policy_prediction"] = {
            "all_angle": masked_mse(angles, actual, np.ones_like(valid)),
            "observed_angle": masked_mse(angles, actual, valid),
            "blackout_angle": masked_mse(angles, actual, ~valid),
            "reward_mse": float(np.mean((predicted_reward.astype(float) - rewards) ** 2)),
            "mean_reward_prediction_bias": float(np.mean(predicted_reward - rewards)),
        }
    return result


def paired_description(candidate, reference, plan):
    differences = np.asarray(candidate, float) - np.asarray(reference, float)
    require(differences.ndim == 1 and np.isfinite(differences).all(), "Bootstrap cases")
    indices = np.random.default_rng(plan["bootstrap_seed"]).integers(
        0, len(differences), (plan["bootstrap_samples"], len(differences))
    )
    means = differences[indices].mean(1)
    return {
        "mean_cost_difference": float(differences.mean()),
        "episode_paired_percentile_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "cases": len(differences),
        "conditional_on_saved_fits": True,
    }


def qualify(plan, predictions, controls):
    checks = []

    def add(name, actual, threshold, direction="le"):
        passed = actual is not None and (actual <= threshold if direction == "le" else actual >= threshold)
        checks.append(
            {
                "name": name,
                "actual": actual,
                "threshold": threshold,
                "direction": direction,
                "passed": bool(passed),
            }
        )

    add(
        "physics_ordinary_vs_zero",
        controls["ordinary"]["known_state"]["mean_cost"],
        0.9 * controls["ordinary"]["zero"]["mean_cost"],
    )
    kinds, comparisons = {}, {}
    for kind in ("gru", "rssm"):
        start = len(checks)
        for seed in plan["fit_seeds"]:
            name = f"{kind}-{seed}"
            for panel in ("ordinary", "shift"):
                add(
                    f"{name}/{panel}/vs_zero",
                    controls[panel][name]["mean_cost"],
                    0.9 * controls[panel]["zero"]["mean_cost"],
                )
            add(
                f"{name}/prediction_vs_persistence",
                predictions[name]["one_step"]["mse"],
                0.9 * predictions[name]["persistence_one_step"]["mse"],
            )
        comparisons[kind] = {}
        for panel in ("ordinary", "shift"):

            def mean_cases(label, suffix="", panel=panel):
                return np.mean(
                    [
                        controls[panel][f"{label}-{seed}{suffix}"]["episode_costs"]
                        for seed in plan["fit_seeds"]
                    ],
                    axis=0,
                )

            recurrent, history, reset = mean_cases(kind), mean_cases("history"), mean_cases(kind, "-reset")
            add(f"{kind}/{panel}/vs_history", float(recurrent.mean()), 0.95 * float(history.mean()))
            add(
                f"{kind}/{panel}/reset_degradation", float(reset.mean()), 1.05 * float(recurrent.mean()), "ge"
            )
            comparisons[kind][panel] = {
                "mean_cost": float(recurrent.mean()),
                "history_mean_cost": float(history.mean()),
                "reset_mean_cost": float(reset.mean()),
                "vs_history": paired_description(recurrent, history, plan),
                "reset_minus_intact": paired_description(reset, recurrent, plan),
            }
        kinds[kind] = {"passed": checks[0]["passed"] and all(c["passed"] for c in checks[start:])}
    return {
        "passed": any(item["passed"] for item in kinds.values()),
        "kinds": kinds,
        "checks": checks,
    }, comparisons


def audit_saved(plan, expected_plan_sha256, execution: Path, out: Path):
    """Authenticate complete saved execution and produce a standalone audit.

    ``plan`` must already have passed the runner's external SHA/source/runtime
    validation. No incomplete input, checkpoint selection, policy or MPC replay.
    Native replay verifies recorded actions, including every actuator-noise draw.
    """
    execution, out = Path(execution), Path(out)
    require(not out.exists(), "Audit output already exists")
    require(
        isinstance(expected_plan_sha256, str)
        and len(expected_plan_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_plan_sha256),
        "External plan SHA",
    )
    require(
        plan["steps"] == 50 and plan["kinds"] == ["history", "gru", "rssm"],
        "Pilot environment/model membership",
    )
    require(
        len(set(plan["fit_seeds"])) == len(plan["fit_seeds"]) and len(plan["fit_seeds"]) > 0,
        "Fit seed uniqueness",
    )
    completed = read(execution / "completed.json")
    expected = expected_members(plan)
    actual = {str(p.relative_to(execution)) for p in execution.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in execution.rglob("*")), "Execution symlinks forbidden")
    require(
        actual == expected | {"completed.json"} and set(completed["files"]) == expected,
        "Execution member set",
    )
    require(
        completed["status"] == "completed"
        and completed["plan_sha256"] == expected_plan_sha256
        and completed["fits"] == len(configurations(plan))
        and completed["astra_calls"] == 0,
        "Execution completion identity",
    )
    wall = finite_number(completed["wall_seconds"], "Execution timing", positive=True)
    require(wall <= plan["cap_seconds"], "Whole-run cap exceeded")
    for member in sorted(expected):
        require(sha(execution / member) == completed["files"][member], f"Execution member hash: {member}")
    started = read(execution / "started.json")
    require(started["plan_sha256"] == expected_plan_sha256, "Started plan binding")
    finite_number(started["unix_time"], "Started timestamp", positive=True)
    train = load_records(execution / "train", plan["train_episodes"])
    prediction = load_records(execution / "prediction", plan["prediction_episodes"])
    replay = {
        "train": audit_cohort(plan, train, "train"),
        "prediction": audit_cohort(plan, prediction, "prediction"),
    }
    fits = {name: audit_fit(plan, execution, name, train) for name in configurations(plan)}
    predictions = {
        name: prediction_metrics(
            plan, prediction, load_npz(execution / "predictions" / f"{name}.npz", {"one", "reward", "multi"})
        )
        for name in configurations(plan)
    }
    controls = {}
    for panel in PANELS:
        controls[panel] = {}
        for arm in arms(plan, panel):
            records = load_records(execution / "control" / panel / arm, plan["control_episodes"])
            replay[f"{panel}/{arm}"] = audit_cohort(plan, records, "control", panel)
            values = load_npz(execution / "control" / panel / f"{arm}-planning.npz")
            controls[panel][arm] = control_metrics(plan, records, values, arm)
    timed = sum(item["wall_seconds"] for item in fits.values()) + sum(
        item["setup_seconds"] + item["decision_wall_seconds"]
        for panel in controls.values()
        for item in panel.values()
    )
    require(timed <= wall + 1e-6, "Component timings exceed whole-run wall time")
    gate, comparisons = qualify(plan, predictions, controls)
    summary = {
        "version": 1,
        "status": "completed",
        "plan_sha256": expected_plan_sha256,
        "execution_completed_sha256": sha(execution / "completed.json"),
        "saved_output_only": True,
        "new_model_calls": 0,
        "new_policy_calls": 0,
        "new_mpc_calls": 0,
        "native_transitions_checked": sum(item["transitions"] for item in replay.values()),
        "native_max_abs_error": max(item["max_abs_error"] for item in replay.values()),
        "wall_seconds": wall,
        "recorded_fit_setup_and_decision_seconds": timed,
        "fits": fits,
        "predictions": predictions,
        "control": controls,
        "cohorts": replay,
        "paired_descriptive_comparisons": comparisons,
        "continuation_gate": gate,
        "limits": [
            "Source/runtime/plan were authenticated by the caller. Hashes and finite weights authenticate saved training artifacts; training and learned predictions are not independently rerun.",
            "One/five-step primary angle MSE averages four components only at valid endpoint observations. Five-step additionally reports valid-root/endpoint restriction. Missing-angle truth is audit-only and excluded from model inputs and training angle targets.",
            "Bootstrap resamples paired episodes after averaging the saved fits. It is conditional on these fits, not architecture-level uncertainty.",
            "Decision times include batch preparation, filtering and planning; per-case figures are amortized batch costs, not individual response deadlines. Setup is separately measured and charged to whole-run time; learned model construction is included in fit time.",
            "Zero/uniform candidate scores are zero placeholders, not model predictions. Supplied-physics references have model-class privilege; known_state additionally has current simulator state.",
            "The fixed final checkpoints and frozen continuation criteria are a pilot qualification, not biological-topology evidence or an ICLR result.",
        ],
    }
    # Check binding again after replay, guarding accidental concurrent mutation.
    require(
        sha(execution / "completed.json") == summary["execution_completed_sha256"],
        "Completion changed during audit",
    )
    for member in expected:
        require(sha(execution / member) == completed["files"][member], "Execution changed during audit")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "summary.json", summary)
    lines = [
        "# Reacher world-model qualification",
        "",
        f"Continuation gate: **{'PASS' if gate['passed'] else 'FAIL'}**.",
        "",
        f"Replayed {summary['native_transitions_checked']:,} saved native transitions. {len(fits)} fixed final fits; no new model, policy or planning calls.",
        "",
        "| Panel | Arm | Mean episode cost |",
        "|---|---|---:|",
    ]
    for panel, results in controls.items():
        lines.extend(f"| {panel} | {name} | {item['mean_cost']:.6f} |" for name, item in results.items())
    lines += [
        "",
        "All episode costs, fit curves, prediction masks, case-paired descriptive intervals and every continuation check are in `summary.json`.",
        "",
    ]
    lines.extend(f"- {limit}" for limit in summary["limits"])
    (out / "README.md").write_text("\n".join(lines) + "\n")
    write(
        out / "receipt.json",
        {
            "status": "completed",
            "plan_sha256": expected_plan_sha256,
            "source_sha256": plan["sources"],
            "runtime": plan["runtime"],
            "plan_canonical_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "execution_completed_sha256": summary["execution_completed_sha256"],
            "execution_members": completed["files"],
            "saved_output_only": True,
            "files": {name: sha(out / name) for name in ("summary.json", "README.md")},
        },
    )
    return summary
