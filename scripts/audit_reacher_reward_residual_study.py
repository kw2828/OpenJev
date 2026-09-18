"""Saved-output audit of a paired free/residual reward-head experiment.

The caller authenticates the frozen plan, runtime and source set. Reused helpers
only read records, replay recorded native transitions and perform arithmetic.
No learned model, policy or controller is imported or rerun here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import audit_reacher_world_model_study as base
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-reward-residual-v1"
COHORT_FIELDS = {
    "train_episodes",
    "steps",
    "train_seed",
    "schedule_seed",
    "noise_seed",
    "exploration_seed",
    "noise_std",
    "ordinary_gap",
    "shift_gap",
}
FIT_FILES = {"initial-weights.pt", "weights.pt", "training.json"}


def expected_members(plan):
    return base.expected_members(plan) | {
        "train-provenance.json",
        "all-fits-completed.json",
        "evaluation-started.json",
        *(f"fits/{name}/initial-weights.pt" for name in base.configurations(plan)),
    }


def minibatch_order_sha256(plan, seed):
    """Reconstruct only the prescribed integer order, using a private generator."""
    generator = torch.Generator().manual_seed(seed + 4100000)
    digest = hashlib.sha256()
    for _ in range(plan["epochs"]):
        digest.update(torch.randperm(plan["train_episodes"], generator=generator).numpy().tobytes())
    return digest.hexdigest()


def validate_training_source(plan, execution):
    source = plan["training_source"]
    base.require(
        set(source)
        == {
            "execution_path",
            "plan_path",
            "plan_sha256",
            "audit_receipt_path",
            "audit_receipt_sha256",
            "members",
            "cohort_plan",
        },
        "Training source schema",
    )
    base.require(base.read(execution / "train-provenance.json") == source, "Copied training provenance")
    base.require(set(source["members"]) == {"train.npz", "train.json"}, "Training source member set")
    parent_plan_path = ROOT / source["plan_path"]
    parent_receipt_path = ROOT / source["audit_receipt_path"]
    base.require(base.sha(parent_plan_path) == source["plan_sha256"], "Parent plan hash")
    base.require(base.sha(parent_receipt_path) == source["audit_receipt_sha256"], "Parent audit receipt hash")
    parent_plan, parent_receipt = base.read(parent_plan_path), base.read(parent_receipt_path)
    base.require(
        parent_receipt["status"] == "completed" and parent_receipt["plan_sha256"] == source["plan_sha256"],
        "Parent completed audit identity",
    )
    expected_cohort = {key: parent_plan[key] for key in COHORT_FIELDS}
    base.require(source["cohort_plan"] == expected_cohort, "Parent cohort plan binding")
    base.require(
        plan["train_episodes"] == expected_cohort["train_episodes"]
        and plan["steps"] == expected_cohort["steps"]
        and plan["noise_std"] == expected_cohort["noise_std"],
        "Copied training task/count",
    )
    original = ROOT / source["execution_path"]
    base.require(
        base.sha(original / "completed.json") == parent_receipt["execution_completed_sha256"],
        "Parent execution receipt hash",
    )
    for name, digest in source["members"].items():
        base.require(
            parent_receipt["execution_members"][name] == digest
            and base.sha(original / name) == digest
            and base.sha(execution / name) == digest,
            "Copied training member hash",
        )
    # Episode identity is defined by reset seeds; reused training is the sole intentional overlap.
    fresh = {
        split: set(range(plan[f"{split}_seed"], plan[f"{split}_seed"] + plan[f"{split}_episodes"]))
        for split in ("prediction", "control")
    }
    inherited = set()
    for split in ("train", "prediction", "control"):
        inherited.update(
            range(
                parent_plan[f"{split}_seed"], parent_plan[f"{split}_seed"] + parent_plan[f"{split}_episodes"]
            )
        )
    base.require(
        not fresh["prediction"] & fresh["control"] and all(not seeds & inherited for seeds in fresh.values()),
        "Fresh evaluation reset seeds overlap",
    )
    old_streams, new_streams = (
        stochastic_streams(parent_plan, include_train=True),
        stochastic_streams(plan, include_train=False),
    )
    base.require(
        all(not old_streams[key] & new_streams[key] for key in old_streams),
        "Fresh evaluation stochastic streams overlap parent",
    )
    return expected_cohort


def stochastic_streams(plan, *, include_train):
    """Seed identities by purpose; control arms/panels intentionally share draws."""

    def cohort(base_key, offset, count):
        return set(range(plan[base_key] + offset, plan[base_key] + offset + count))

    streams = {
        "noise": cohort("noise_seed", 100000, plan["prediction_episodes"])
        | cohort("noise_seed", 200000, plan["control_episodes"]),
        "schedule": cohort("schedule_seed", 100000, plan["prediction_episodes"])
        | cohort("schedule_seed", 0, plan["control_episodes"]),
        "exploration": cohort("exploration_seed", 100000, plan["prediction_episodes"]),
        "candidate": cohort("candidate_seed", 0, plan["steps"]) | {plan["candidate_seed"] + 200000},
        "filter": cohort("filter_seed", 0, plan["control_episodes"]),
    }
    if include_train:
        for purpose in ("noise", "schedule", "exploration"):
            streams[purpose] |= cohort(purpose + "_seed", 0, plan["train_episodes"])
    return streams


def load_weights(plan, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    shapes = base.checkpoint_shapes(plan, "gru")
    base.require(isinstance(state, dict) and set(state) == set(shapes), "Checkpoint state membership")
    for key, value in state.items():
        base.require(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            and tuple(value.shape) == shapes[key]
            and torch.isfinite(value).all().item(),
            "Nonfinite/invalid checkpoint weight",
        )
    return state


def audit_fit(plan, execution, name, train):
    folder = execution / "fits" / name
    completed = base.read(folder / "completed.json")
    kind, seed_text = name.rsplit("-", 1)
    seed = int(seed_text)
    base.require(completed["kind"] == kind and completed["seed"] == seed, "Fit identity")
    base.require(
        type(completed["residual_reward"]) is bool and completed["residual_reward"] is (kind == "residual"),
        "Reward-head configuration binding",
    )
    base.require(
        type(completed["noise_std"]) in (int, float) and completed["noise_std"] == plan["noise_std"],
        "Reward noise configuration binding",
    )
    batches = math.ceil(plan["train_episodes"] / plan["batch_size"])
    base.require(
        type(completed["updates"]) is int and completed["updates"] == plan["epochs"] * batches,
        "Fit update count",
    )
    base.require(set(completed["files"]) == FIT_FILES, "Fit member set")
    for member, digest in completed["files"].items():
        base.require(base.sha(folder / member) == digest, "Fit member hash")
    base.require(
        completed["initial_weights_sha256"] == completed["files"]["initial-weights.pt"],
        "Initial checkpoint hash binding",
    )
    wanted_order = minibatch_order_sha256(plan, seed)
    base.require(completed["minibatch_order_sha256"] == wanted_order, "Minibatch order hash")
    base.finite_number(completed["wall_seconds"], "Fit timing", positive=True)
    initial, final = (
        load_weights(plan, folder / filename) for filename in ("initial-weights.pt", "weights.pt")
    )
    count = sum(value.numel() for value in final.values())
    base.require(
        type(completed["parameters"]) is int and count == completed["parameters"],
        "Checkpoint parameter count",
    )
    logs = base.read(folder / "training.json")
    base.require(isinstance(logs, list) and len(logs) == plan["epochs"], "Training epoch count")
    packets = base.stack(train, "policy", "packets")
    targets = float((packets[:, 1:, 6] > 0.5).sum() / batches)
    roots = float((packets[:, : plan["steps"] - plan["rollout_horizon"] + 1, 6] > 0.5).sum() / batches)
    for epoch, row in enumerate(logs, 1):
        base.require(
            set(row) == base.LOG_KEYS and type(row["epoch"]) is int and row["epoch"] == epoch,
            "Training epoch membership/order",
        )
        for key, value in row.items():
            base.finite_number(value, f"Training finite metric: {key}")
        base.require(row["kl_nats"] == 0, "GRU KL must be zero")
        objective = (
            row["observation_mse"]
            + plan["reward_scale"] * row["reward_mse"]
            + plan["rollout_weight"]
            * (row["rollout_observation_mse"] + plan["reward_scale"] * row["rollout_reward_mse"])
        )
        base.require(
            math.isclose(row["loss"], objective, rel_tol=1e-5, abs_tol=1e-6), "Training loss arithmetic"
        )
        base.require(
            math.isclose(row["valid_observation_targets"], targets, abs_tol=1e-6)
            and math.isclose(row["valid_rollout_starts"], roots, abs_tol=1e-6),
            "Training target count",
        )
    result = {
        key: completed[key]
        for key in (
            "kind",
            "seed",
            "updates",
            "parameters",
            "wall_seconds",
            "residual_reward",
            "noise_std",
            "initial_weights_sha256",
            "minibatch_order_sha256",
        )
    }
    result["epochs"] = logs
    return result, initial


def audit_boundaries(plan, expected_hash, execution, completed, fits):
    started = base.read(execution / "started.json")
    boundary = base.read(execution / "all-fits-completed.json")
    evaluation = base.read(execution / "evaluation-started.json")
    for item in (started, boundary, evaluation):
        base.require(item["plan_sha256"] == expected_hash, "Phase plan binding")
        base.finite_number(item["unix_time"], "Phase timestamp", positive=True)
    base.require(
        set(boundary) == {"plan_sha256", "fit_order", "files", "unix_time", "elapsed_seconds"},
        "Fit boundary schema",
    )
    base.require(
        set(evaluation) == {"plan_sha256", "all_fits_completed_sha256", "unix_time", "elapsed_seconds"},
        "Evaluation boundary schema",
    )
    base.require(boundary["fit_order"] == plan["fit_order"], "Fit boundary order")
    members = {f"fits/{name}/completed.json" for name in base.configurations(plan)}
    base.require(set(boundary["files"]) == members, "Fit boundary membership")
    for member, digest in boundary["files"].items():
        base.require(base.sha(execution / member) == digest, "Fit boundary member hash")
    base.require(
        evaluation["all_fits_completed_sha256"] == base.sha(execution / "all-fits-completed.json"),
        "Evaluation boundary hash",
    )
    fit_elapsed = base.finite_number(boundary["elapsed_seconds"], "Fit boundary elapsed", positive=True)
    eval_elapsed = base.finite_number(
        evaluation["elapsed_seconds"], "Evaluation boundary elapsed", positive=True
    )
    base.require(
        fit_elapsed <= eval_elapsed <= completed["wall_seconds"]
        and started["unix_time"] <= boundary["unix_time"] <= evaluation["unix_time"],
        "Fit/evaluation chronology",
    )
    base.require(
        completed["evaluation_started_elapsed_seconds"] == eval_elapsed,
        "Completion evaluation timing binding",
    )
    base.require(
        sum(item["wall_seconds"] for item in fits.values()) <= fit_elapsed + 1e-6,
        "Fit times exceed pre-evaluation boundary",
    )
    return {
        "fit_order": boundary["fit_order"],
        "all_fits_completed_sha256": base.sha(execution / "all-fits-completed.json"),
        "evaluation_started_sha256": base.sha(execution / "evaluation-started.json"),
        "all_fits_elapsed_seconds": fit_elapsed,
        "evaluation_started_elapsed_seconds": eval_elapsed,
        "scope": "Source-bound logged phase chronology, not an independent process observer.",
    }


def qualify(plan, predictions, controls):
    expected = set(base.configurations(plan))
    base.require(set(predictions) == expected and set(controls) == set(base.PANELS), "Qualification coverage")
    for panel in base.PANELS:
        base.require(set(controls[panel]) == set(base.arms(plan, panel)), "Control arm coverage")
    checks = []

    def add(name, value, threshold, *, strict=False):
        base.require(
            type(value) in (int, float)
            and math.isfinite(value)
            and type(threshold) in (int, float)
            and math.isfinite(threshold),
            "Nonfinite qualification metric",
        )
        passed = value < threshold if strict else value <= threshold
        checks.append(
            {
                "name": name,
                "actual": value,
                "threshold": threshold,
                "direction": "lt" if strict else "le",
                "passed": bool(passed),
            }
        )

    def cases(panel, kind, suffix=""):
        return np.mean(
            [controls[panel][f"{kind}-{seed}{suffix}"]["episode_costs"] for seed in plan["fit_seeds"]], axis=0
        )

    add(
        "physics_ordinary_vs_zero",
        controls["ordinary"]["known_state"]["mean_cost"],
        0.9 * controls["ordinary"]["zero"]["mean_cost"],
    )
    comparisons = {}
    for panel in ("ordinary", "shift"):
        comparisons[panel] = {}
        for seed in plan["fit_seeds"]:
            name = f"residual-{seed}"
            add(
                f"{name}/{panel}/vs_zero",
                controls[panel][name]["mean_cost"],
                0.9 * controls[panel]["zero"]["mean_cost"],
            )
            add(
                f"{name}/{panel}/vs_paired_free",
                controls[panel][name]["mean_cost"],
                controls[panel][f"free-{seed}"]["mean_cost"],
                strict=True,
            )
        free, residual = cases(panel, "free"), cases(panel, "residual")
        add(f"residual/{panel}/mean_vs_free", float(residual.mean()), 0.95 * float(free.mean()))
        comparisons[panel]["free_mean_cost"] = float(free.mean())
        comparisons[panel]["residual_mean_cost"] = float(residual.mean())
        comparisons[panel]["residual_minus_free"] = base.paired_description(residual, free, plan)
        for kind in plan["kinds"]:
            intact, reset = cases(panel, kind), cases(panel, kind, "-reset")
            comparisons[panel][f"{kind}_reset_minus_intact"] = base.paired_description(reset, intact, plan)
    means = {
        kind: {
            "reward_mse": float(
                np.mean([predictions[f"{kind}-{seed}"]["reward_mse"] for seed in plan["fit_seeds"]])
            ),
            "one_step_angle_mse": float(
                np.mean([predictions[f"{kind}-{seed}"]["one_step"]["mse"] for seed in plan["fit_seeds"]])
            ),
        }
        for kind in plan["kinds"]
    }
    add("residual/reward_mse_vs_free", means["residual"]["reward_mse"], 0.9 * means["free"]["reward_mse"])
    add(
        "residual/one_step_angle_noninferiority",
        means["residual"]["one_step_angle_mse"],
        1.05 * means["free"]["one_step_angle_mse"],
    )
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "checks": checks,
        "kinds": {"residual": {"passed": passed}},
        "prediction_means": means,
        "reset_is_descriptive_only": True,
    }, comparisons


def audit_saved(plan, expected_plan_sha256, execution: Path, out: Path):
    """Audit complete saved outputs after caller source/runtime authentication."""
    execution, out = Path(execution), Path(out)
    base.require(not out.exists() and not out.is_symlink(), "Audit output already exists")
    base.require(
        isinstance(expected_plan_sha256, str)
        and len(expected_plan_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_plan_sha256),
        "External plan SHA",
    )
    base.require(
        plan["steps"] == 50 and plan["kinds"] == ["free", "residual"], "Pilot environment/model membership"
    )
    names = base.configurations(plan)
    base.require(
        bool(plan["fit_seeds"])
        and all(type(seed) is int and seed >= 0 for seed in plan["fit_seeds"])
        and len(names) == len(set(names))
        and set(plan["fit_order"]) == set(names)
        and len(plan["fit_order"]) == len(names),
        "Fit seed/order membership",
    )
    completed = base.read(execution / "completed.json")
    completion_hash = base.sha(execution / "completed.json")
    expected = expected_members(plan)
    actual = {str(path.relative_to(execution)) for path in execution.rglob("*") if path.is_file()}
    base.require(
        not execution.is_symlink() and not any(path.is_symlink() for path in execution.rglob("*")),
        "Execution symlinks forbidden",
    )
    base.require(
        actual == expected | {"completed.json"} and set(completed["files"]) == expected,
        "Execution member set",
    )
    base.require(
        completed["status"] == "completed"
        and completed["plan_sha256"] == expected_plan_sha256
        and completed["fits"] == len(names)
        and completed["astra_calls"] == 0,
        "Execution completion identity",
    )
    wall = base.finite_number(completed["wall_seconds"], "Execution timing", positive=True)
    base.require(wall <= plan["cap_seconds"], "Whole-run cap exceeded")
    for member in expected:
        base.require(
            base.sha(execution / member) == completed["files"][member], f"Execution member hash: {member}"
        )
    cohort_plan = validate_training_source(plan, execution)
    train = base.load_records(execution / "train", plan["train_episodes"])
    prediction = base.load_records(execution / "prediction", plan["prediction_episodes"])
    replay = {
        "train": base.audit_cohort(cohort_plan, train, "train"),
        "prediction": base.audit_cohort(plan, prediction, "prediction"),
    }
    fits, initial = {}, {}
    for name in plan["fit_order"]:
        fits[name], initial[name] = audit_fit(plan, execution, name, train)
    pairs = {}
    for seed in plan["fit_seeds"]:
        left, right = initial[f"free-{seed}"], initial[f"residual-{seed}"]
        base.require(all(torch.equal(left[key], right[key]) for key in left), "Paired initial tensors differ")
        base.require(
            fits[f"free-{seed}"]["minibatch_order_sha256"]
            == fits[f"residual-{seed}"]["minibatch_order_sha256"],
            "Paired minibatch order differs",
        )
        pairs[str(seed)] = {
            "initial_tensors_identical": True,
            "parameters": fits[f"free-{seed}"]["parameters"],
            "minibatch_order_sha256": fits[f"free-{seed}"]["minibatch_order_sha256"],
        }
    del initial
    boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed, fits)
    predictions = {
        name: base.prediction_metrics(
            plan,
            prediction,
            base.load_npz(execution / "predictions" / f"{name}.npz", {"one", "reward", "multi"}),
        )
        for name in names
    }
    controls = {}
    for panel in base.PANELS:
        controls[panel] = {}
        for arm in base.arms(plan, panel):
            records = base.load_records(execution / "control" / panel / arm, plan["control_episodes"])
            replay[f"{panel}/{arm}"] = base.audit_cohort(plan, records, "control", panel)
            values = base.load_npz(execution / "control" / panel / f"{arm}-planning.npz")
            controls[panel][arm] = base.control_metrics(plan, records, values, arm)
    decision_and_setup = sum(
        item["setup_seconds"] + item["decision_wall_seconds"]
        for panel in controls.values()
        for item in panel.values()
    )
    timed = sum(item["wall_seconds"] for item in fits.values()) + decision_and_setup
    base.require(timed <= wall + 1e-6, "Component timings exceed whole-run wall time")
    base.require(
        decision_and_setup <= wall - boundary["evaluation_started_elapsed_seconds"] + 1e-6,
        "Control timings exceed evaluation interval",
    )
    gate, comparisons = qualify(plan, predictions, controls)
    summary = {
        "version": VERSION,
        "status": "completed",
        "plan_sha256": expected_plan_sha256,
        "execution_completed_sha256": completion_hash,
        "saved_output_only": True,
        "new_model_calls": 0,
        "new_policy_calls": 0,
        "new_mpc_calls": 0,
        "native_transitions_checked": sum(item["transitions"] for item in replay.values()),
        "native_max_abs_error": max(item["max_abs_error"] for item in replay.values()),
        "wall_seconds": wall,
        "recorded_fit_setup_and_decision_seconds": timed,
        "training_source": plan["training_source"],
        "paired_training": pairs,
        "phase_boundary": boundary,
        "fits": fits,
        "predictions": predictions,
        "control": controls,
        "cohorts": replay,
        "paired_descriptive_comparisons": comparisons,
        "continuation_gate": gate,
        "limits": [
            "Caller authenticates frozen source/runtime/plan. Hashes, exact checkpoint schemas, paired initial tensors and regenerated minibatch-order digests authenticate recorded training; no optimizer or learned predictions are rerun.",
            "Only the original train cohort is copied. Fresh prediction/control reset identities and their planned public sensor and hidden-noise streams are checked. Native replay covers every recorded transition without new policy or MPC calls.",
            "The treatment has supplied knowledge of the actuator cost and noise distribution. Training targets remain total native executed reward and observed angles; realized noise, applied actions and privileged state are audit-only.",
            "All-fits and evaluation-start receipts establish source-bound logged chronology, not an independent process observer. Final checkpoints only; reset interventions are descriptive and cannot affect continuation.",
            "Prediction masks and persistence use valid endpoint observations; missing-state truth is an audit-only diagnostic. Five-step prediction also reports valid-root/endpoint results.",
            "Case-paired bootstrap is conditional on these fitted seeds and is descriptive, not fit or architecture uncertainty. Every paired seed and both ordinary/shift panels must pass the predefined useful-effect checks.",
            "Setup and batch decision wall times are charged; per-case latency is amortized, not an individual deadline. Whole-run cap is cooperative. Supplied-physics references and zero/uniform floors retain their original privilege and score-placeholder labels.",
            "This is a known-reward-structure qualification, not a new RL algorithm, recurrent-memory advantage, connectome result or ICLR claim.",
        ],
    }
    base.require(base.sha(execution / "completed.json") == completion_hash, "Completion changed during audit")
    for member in expected:
        base.require(
            base.sha(execution / member) == completed["files"][member], "Execution changed during audit"
        )
    out.mkdir(parents=True, exist_ok=False)
    base.write(out / "summary.json", summary)
    lines = [
        "# Reacher reward-residual qualification",
        "",
        f"Continuation gate: **{'PASS' if gate['passed'] else 'FAIL'}** ({sum(check['passed'] for check in gate['checks'])}/{len(gate['checks'])} checks).",
        "",
        f"{len(fits)} final fits; {summary['native_transitions_checked']:,} saved native transitions replayed. No new model, policy or planning calls.",
        "",
        "| Panel | Arm | Mean episode cost |",
        "|---|---|---:|",
    ]
    for panel, results in controls.items():
        lines.extend(f"| {panel} | {name} | {item['mean_cost']:.6f} |" for name, item in results.items())
    lines += [
        "",
        "All paired initialization/order checks, fit curves, prediction and reset diagnostics, case-paired intervals and continuation criteria are in `summary.json`.",
        "",
    ]
    lines.extend(f"- {limit}" for limit in summary["limits"])
    (out / "README.md").write_text("\n".join(lines) + "\n")
    base.write(
        out / "receipt.json",
        {
            "status": "completed",
            "version": VERSION,
            "plan_sha256": expected_plan_sha256,
            "source_sha256": plan["sources"],
            "runtime": plan["runtime"],
            "plan_canonical_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "execution_completed_sha256": completion_hash,
            "execution_members": completed["files"],
            "training_source": plan["training_source"],
            "saved_output_only": True,
            "files": {name: base.sha(out / name) for name in ("summary.json", "README.md")},
        },
    )
    return summary
