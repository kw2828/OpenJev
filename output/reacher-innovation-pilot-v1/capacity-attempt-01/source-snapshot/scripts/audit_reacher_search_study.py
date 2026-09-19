"""Independent saved-output checks for the prospective Reacher search study.

Proposal reconstruction uses only recorded scores and frozen innovations. Native
replay executes saved actions; no learned model, policy or MPC is imported.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import audit_reacher_reward_residual_control as inherited
import numpy as np

from openjev.research import reacher_search_protocol as protocol
from openjev.research import robotics_reacher as native
from openjev.research.reacher_adaptive_search import SearchInputs, search

base = inherited.base
ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-search-v1"
TRACE_KEYS = {
    "method",
    "horizon",
    "action_block",
    "candidate_ids",
    "input_identities",
    "candidate_evaluations_per_case",
    "candidate_evaluations",
    "imagined_transitions_per_case",
    "imagined_transitions",
    "callback_sizes",
    "search_seconds",
    "stages",
}
STAGE_KEYS = {
    "name",
    "start",
    "stop",
    "innovation_source",
    "innovation_count",
    "mean_candidate_id",
    "array_fields",
}
STAGE_ARRAYS = ("proposal_mean", "proposal_std", "fixed_scales", "source_elite_ids")


def array(value, shape, dtype, label):
    base.require(
        isinstance(value, np.ndarray)
        and value.shape == shape
        and value.dtype == np.dtype(dtype)
        and np.isfinite(value).all(),
        label,
    )
    return value


def clipped_returns(raw):
    """Match the scorer's sequential float32 accumulation, not a tree sum."""
    base.require(
        raw.ndim == 3 and raw.dtype == np.float32 and np.isfinite(raw).all(),
        "Raw reward prediction shape/type",
    )
    result = np.zeros(raw.shape[:2], dtype=np.float32)
    for step in range(raw.shape[2]):
        result += np.clip(raw[:, :, step], -2.5, 0)
    return result.astype(np.float64)


def audit_innovations(plan, stem, prefix, count):
    data = base.load_npz(
        Path(stem).with_suffix(".npz"), {name.replace("/", "_") for name in protocol.INPUT_NAMES}
    )
    meta = base.read(Path(stem).with_suffix(".json"))
    base.require(
        set(meta)
        == {"prefix", "input_identities", "shapes", "unused_anchor_draws", "full_horizon_innovations"},
        "Innovation metadata schema",
    )
    expected = protocol.draw_inputs(plan, prefix, count)
    values = (expected.initial, expected.random_extra, *expected.cem)
    shapes = {}
    for name, value in zip(protocol.INPUT_NAMES, values, strict=True):
        key = name.replace("/", "_")
        array(data[key], value.shape, np.float64, "Innovation shape/dtype")
        base.require(np.array_equal(data[key], value), "Innovation differs from prospective stream")
        shapes[name] = list(value.shape)
    base.require(
        meta["prefix"] == prefix
        and meta["input_identities"] == dict(expected.identities())
        and meta["shapes"] == shapes
        and type(meta["unused_anchor_draws"]) is int
        and meta["unused_anchor_draws"] == 7
        and meta["full_horizon_innovations"] is True,
        "Innovation metadata binding",
    )
    return expected


def audit_trace(plan, inputs: SearchInputs, stem: Path, *, step: int):
    """Reconstruct all paid proposals using stored scalar scores as a lookup.

    The only callback indexes saved data. Each proposed command bank must match
    the stored float32 chunks exactly before the callback returns any score.
    """
    meta = base.read(stem.with_suffix(".json"))
    data = base.load_npz(stem.with_suffix(".npz"))
    base.require(set(meta) == TRACE_KEYS, "Search trace metadata membership")
    method, n = meta["method"], inputs.initial.shape[0]
    base.require(method in plan["planners"], "Search method")
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    chunks = math.ceil(horizon / plan["action_block"])
    count = 64 if method == "rs64" else 256
    commands = array(data["chunks"], (n, count, chunks, 2), np.float32, "Search chunks")
    base.require(np.all(np.abs(commands) <= 1), "Search command clipping")
    scores = array(data["scores"], (n, count), np.float64, "Search scores")
    raw = array(data["raw_rewards"], (n, count, horizon), np.float32, "Search raw rewards")
    selected = array(data["selected_ids"], (n,), np.int64, "Search selected IDs")
    base.require(
        np.allclose(scores, clipped_returns(raw), rtol=1e-6, atol=2e-5),
        "Clipped sequential reward arithmetic",
    )
    sequences = np.repeat(commands, plan["action_block"], axis=2)[:, :, :horizon]
    cursor, sizes = 0, []

    def recorded_scores(bank):
        nonlocal cursor
        size = bank.shape[1]
        base.require(
            np.array_equal(bank, sequences[:, cursor : cursor + size]),
            "Reconstructed search proposals differ",
        )
        values = scores[:, cursor : cursor + size]
        cursor += size
        sizes.append(size)
        return values

    result = search(
        method,
        inputs,
        recorded_scores,
        step=step,
        steps=plan["steps"],
        planning_horizon=plan["planning_horizon"],
        action_block=plan["action_block"],
    )
    base.require(cursor == count and meta["callback_sizes"] == sizes, "Paid scorer call schedule")
    base.require(meta["input_identities"] == dict(result.input_identities), "Search innovation identities")
    base.require(meta["candidate_ids"] == list(result.candidate_ids), "Global candidate identity/order")
    base.require(np.array_equal(selected, result.selected_ids), "Global best selection/tie order")
    for name in (
        "horizon",
        "action_block",
        "candidate_evaluations_per_case",
        "candidate_evaluations",
        "imagined_transitions_per_case",
        "imagined_transitions",
    ):
        base.require(
            type(meta[name]) is int and meta[name] == getattr(result, name), f"Search budget: {name}"
        )
    base.require(
        isinstance(meta["stages"], list) and len(meta["stages"]) == len(result.stages),
        "Search stage coverage",
    )
    expected_members = {"chunks", "scores", "raw_rewards", "selected_ids"}
    for index, (saved, stage) in enumerate(zip(meta["stages"], result.stages, strict=True)):
        base.require(set(saved) == STAGE_KEYS, "Search stage metadata schema")
        for field in STAGE_KEYS - {"array_fields"}:
            base.require(saved[field] == getattr(stage, field), f"Search stage field: {field}")
        fields = {
            field: f"stage_{index}_{field}" for field in STAGE_ARRAYS if getattr(stage, field) is not None
        }
        base.require(saved["array_fields"] == fields, "Search stage array membership")
        expected_members.update(fields.values())
        for field, key in fields.items():
            value = getattr(stage, field)
            base.require(
                key in data and data[key].dtype == value.dtype and np.array_equal(data[key], value),
                f"Search stage reconstruction: {field}",
            )
    base.require(set(data) == expected_members, "Search trace array membership")
    seconds = base.finite_number(meta["search_seconds"], "Search timing", positive=True)
    rows = np.arange(n)
    return {
        "result": result,
        "raw_rewards": raw,
        "search_seconds": seconds,
        "selected_clipped_return": scores[rows, selected],
        "selected_raw_return": raw[rows, selected].astype(np.float64).sum(axis=-1),
        "clipped_predictions": int(((raw < -2.5) | (raw > 0)).sum()),
        "reward_predictions": int(raw.size),
    }


def average_ranks(values):
    """Deterministic average ranks for ties, without external statistics code."""
    values = np.asarray(values, dtype=np.float64)
    base.require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Rank input")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def rank_agreement(prediction, native):
    left, right = average_ranks(prediction), average_ranks(native)
    base.require(left.shape == right.shape, "Rank alignment")
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    # A constant vector has undefined rank correlation, not zero correlation.
    return None if denominator == 0 else float(np.clip(left @ right / denominator, -1, 1))


def diagnostic_metrics(trace, native_returns, selected_slot):
    """Native labels never influence proposal reconstruction or selection."""
    native_returns = np.asarray(native_returns, dtype=np.float64)
    base.require(
        native_returns.ndim == 2
        and native_returns.shape[0] >= 64
        and native_returns.shape[1] > 0
        and np.isfinite(native_returns).all(),
        "Diagnostic return matrix",
    )
    base.require(
        type(selected_slot) is int and 64 <= selected_slot < len(native_returns), "Selected union slot"
    )
    result, raw = trace["result"], trace["raw_rewards"]
    base.require(result.scores.shape[0] == 1, "Diagnostic root must be a single case")
    expected = native_returns.mean(axis=1)
    actual = float(expected[selected_slot])
    return {
        "common_bank_spearman": rank_agreement(result.scores[0, :64], expected[:64]),
        "native_selected_return": actual,
        "native_selected_branch_returns": native_returns[selected_slot].tolist(),
        "finite_set_regret": float(expected.max() - actual),
        "evaluated_union_slots": len(expected),
        "selected_raw_prediction_bias": float(trace["selected_raw_return"][0] - actual),
        "selected_clipped_prediction_bias": float(trace["selected_clipped_return"][0] - actual),
        "common_bank_raw_return_mse": float(
            np.mean((raw[0, :64].astype(float).sum(-1) - expected[:64]) ** 2)
        ),
        "common_bank_clipped_return_mse": float(np.mean((result.scores[0, :64] - expected[:64]) ** 2)),
        "reward_prediction_clipping_fraction": trace["clipped_predictions"] / trace["reward_predictions"],
        "regret_scope": "Best mean native return within the evaluated finite union; not an optimal controller.",
    }


def replay_branches(record, step, commands, noise, saved, *, atol=1e-10):
    """Replay every recorded branch from its common authenticated physical root.

    `noise` must already have been regenerated from the prospective branch
    streams by the caller. No randomness is sampled in this native replay.
    """
    base.require(type(step) is int and 0 <= step < 50, "Native branch root index")
    base.require(
        commands.ndim == 3
        and commands.shape[-1] == 2
        and commands.dtype == np.float32
        and np.isfinite(commands).all()
        and np.all(np.abs(commands) <= 1),
        "Native branch commands",
    )
    slots, horizon, _ = commands.shape
    base.require(horizon > 0 and step + horizon <= 50, "Native branch crosses terminal boundary")
    base.require(
        noise.ndim == 3
        and noise.shape[1:] == (horizon, 2)
        and noise.dtype == np.float64
        and np.isfinite(noise).all()
        and len(noise) > 0,
        "Native branch noise",
    )
    branches = len(noise)
    env = native.make_env()
    try:
        identity = native.native_identity(env)
        base.require(
            all(record["metadata"].get(key) == value for key, value in identity.items()),
            "Native branch environment identity",
        )
        env.reset(seed=record["metadata"]["seed"])
        state_width = native.mujoco.mj_stateSize(env.unwrapped.model, native.STATE_SPEC)
        shapes = {
            "qpos": (horizon + 1, 4),
            "qvel": (horizon + 1, 4),
            "raw_obs": (horizon + 1, 10),
            "integration_state": (horizon + 1, state_width),
            "time": (horizon + 1,),
            "commands": (horizon, 2),
            "applied_actions": (horizon, 2),
            "actuator_noise": (horizon, 2),
            "rewards": (horizon,),
            "reward_dist": (horizon,),
            "reward_ctrl": (horizon,),
            "terminated": (horizon,),
            "truncated": (horizon,),
        }
        base.require(set(saved) == set(shapes), "Native branch record membership")
        for key, shape in shapes.items():
            dtype = (
                np.bool_
                if key in ("terminated", "truncated")
                else np.float32
                if key == "commands"
                else np.float64
            )
            array(saved[key], (slots, branches, *shape), dtype, f"Native branch {key} shape/type")
        expected_commands = np.broadcast_to(commands[:, None], (slots, branches, horizon, 2))
        expected_noise = np.broadcast_to(noise[None], (slots, branches, horizon, 2))
        base.require(
            np.array_equal(saved["commands"], expected_commands), "Native branch command union binding"
        )
        base.require(
            np.array_equal(saved["actuator_noise"], expected_noise), "Native branch shared noise binding"
        )
        applied = np.clip(expected_commands.astype(np.float64) + expected_noise, -1, 1)
        base.require(np.array_equal(saved["applied_actions"], applied), "Native branch actuator arithmetic")
        root = record["audit"]["integration_state"][step]
        base.require(
            np.array_equal(
                saved["integration_state"][:, :, 0], np.broadcast_to(root, (slots, branches, state_width))
            ),
            "Native branch root state binding",
        )
        max_error = 0.0

        def compare(key, value, slot, branch, index):
            nonlocal max_error
            error = float(np.max(np.abs(np.asarray(value) - saved[key][slot, branch, index])))
            base.require(math.isfinite(error) and error <= atol, f"Native branch replay mismatch: {key}")
            max_error = max(max_error, error)

        for slot in range(slots):
            for branch in range(branches):
                native.restore_native(env, {"integration_state": root, "step": step})
                raw = env.unwrapped._get_obs()
                for index in range(horizon + 1):
                    for key, value in {
                        "qpos": env.unwrapped.data.qpos,
                        "qvel": env.unwrapped.data.qvel,
                        "raw_obs": raw,
                        "integration_state": native._integration_state(env),
                        "time": env.unwrapped.data.time,
                    }.items():
                        compare(key, value, slot, branch, index)
                    if index == horizon:
                        break
                    raw, reward, terminated, truncated, info = env.step(applied[slot, branch, index])
                    base.require(
                        not terminated and bool(truncated) == (step + index == 49),
                        "Native branch terminal semantics",
                    )
                    base.require(
                        saved["terminated"][slot, branch, index] == terminated
                        and saved["truncated"][slot, branch, index] == truncated,
                        "Native branch recorded terminal flags",
                    )
                    for key, value in (
                        ("rewards", reward),
                        ("reward_dist", info["reward_dist"]),
                        ("reward_ctrl", info["reward_ctrl"]),
                    ):
                        compare(key, value, slot, branch, index)
        return {
            "transitions": slots * branches * horizon,
            "max_abs_error": max_error,
            "saved_output_only": True,
            "new_policy_calls": 0,
        }
    finally:
        env.close()


def qualify_search(plan, controls):
    """Fresh native control thresholds, separate from historical prediction tests."""
    bootstrap_plan = plan | {"bootstrap_seed": protocol.seed(plan, "analysis/bootstrap/0")}
    names = base.configurations(plan)
    expected = {f"{name}-{method}" for name in names for method in plan["planners"]} | set(base.REFERENCES)
    base.require(
        set(controls) == set(base.PANELS) and all(set(panel) == expected for panel in controls.values()),
        "Search control coverage",
    )
    checks, competence = [], {}

    def add(target, name, actual, threshold, strict=False):
        base.require(
            type(actual) in (float, int)
            and math.isfinite(actual)
            and type(threshold) in (float, int)
            and math.isfinite(threshold),
            "Finite control qualification",
        )
        target.append(
            {
                "name": name,
                "actual": actual,
                "threshold": threshold,
                "direction": "lt" if strict else "le",
                "passed": bool(actual < threshold if strict else actual <= threshold),
            }
        )

    def mean_cases(panel, kind, method):
        values = [controls[panel][f"{kind}-{seed}-{method}"]["episode_costs"] for seed in plan["fit_seeds"]]
        values = np.asarray(values, dtype=np.float64)
        base.require(
            values.shape == (len(plan["fit_seeds"]), plan["control_episodes"]) and np.isfinite(values).all(),
            "Paired control episode alignment",
        )
        return values.mean(0)

    comparisons = {}
    for method in plan["planners"]:
        rows = []
        add(
            rows,
            "physics_ordinary_vs_zero",
            controls["ordinary"]["known_state"]["mean_cost"],
            0.9 * controls["ordinary"]["zero"]["mean_cost"],
        )
        for panel in ("ordinary", "shift"):
            for seed in plan["fit_seeds"]:
                residual = controls[panel][f"residual-{seed}-{method}"]["mean_cost"]
                add(
                    rows,
                    f"residual-{seed}/{panel}/vs_zero",
                    residual,
                    0.9 * controls[panel]["zero"]["mean_cost"],
                )
                add(
                    rows,
                    f"residual-{seed}/{panel}/vs_paired_free",
                    residual,
                    controls[panel][f"free-{seed}-{method}"]["mean_cost"],
                    strict=True,
                )
            free, residual = mean_cases(panel, "free", method), mean_cases(panel, "residual", method)
            add(rows, f"residual/{panel}/mean_vs_free", float(residual.mean()), 0.95 * float(free.mean()))
        competence[method] = {
            "passed": all(row["passed"] for row in rows),
            "checks": rows,
            "scope": "Fresh control-only checks. Historical held-out prediction and reset endpoints are not rerun.",
        }
    for panel in ("ordinary", "shift"):
        cem, reference = mean_cases(panel, "residual", "cem256"), mean_cases(panel, "residual", "rs256")
        add(checks, f"residual/{panel}/cem_mean_vs_rs256", float(cem.mean()), 0.95 * float(reference.mean()))
        for seed in plan["fit_seeds"]:
            add(
                checks,
                f"residual-{seed}/{panel}/cem_vs_rs256",
                controls[panel][f"residual-{seed}-cem256"]["mean_cost"],
                controls[panel][f"residual-{seed}-rs256"]["mean_cost"],
                strict=True,
            )
        comparisons[panel] = {
            "residual_cem_minus_rs256": base.paired_description(cem, reference, bootstrap_plan)
        }
        for kind in plan["kinds"]:
            high, low = mean_cases(panel, kind, "rs256"), mean_cases(panel, kind, "rs64")
            comparisons[panel][f"{kind}_rs256_minus_rs64"] = base.paired_description(
                high, low, bootstrap_plan
            )
    return (
        {
            "passed": all(row["passed"] for row in checks),
            "checks": checks,
            "scope": "Native control cost: residual-family CEM versus RS256 at equal scheduled scoring budget.",
        },
        competence,
        comparisons,
    )


INHERITED_FILES = {
    "train.npz",
    "train.json",
    "train-provenance.json",
    "fit-provenance.json",
    "inherited-plan.json",
    "invalid-attempt.json",
    "fit-inheritance.json",
    "source-all-fits-completed.json",
    "source-failed.json",
}
PARENT_FIELDS = {
    "plan_path",
    "plan_sha256",
    "audit_path",
    "audit_receipt_sha256",
    "execution_path",
    "completed_sha256",
    "summary_sha256",
    "members",
    "prior_costs",
}
PLAN_CHANGES = {
    "study",
    "version",
    "sources",
    "runtime",
    "cap_seconds",
    "data",
    "models",
    "control",
    "criteria",
    "stop",
    "random_stream_contract",
}
PLAN_ADDITIONS = {
    "parent_source",
    "planners",
    "panels",
    "references",
    "diagnostic_episodes",
    "diagnostic_branches",
    "execution_order",
    "search_parameters",
    "search_criteria",
    "competence_criteria",
    "legacy_seed_fields_unused",
    "rng_namespace",
    "engineering_rng_namespaces",
}


def inherited_members(plan):
    return INHERITED_FILES | {
        f"fits/{name}/{file}" for name in plan["fit_order"] for file in inherited.FIT_FILES
    }


def control_order(plan):
    rows = []
    for panel in plan["panels"]:
        rows.extend(
            {"panel": panel, "fit": name, "planner": method, "label": f"{name}-{method}"}
            for name in plan["fit_order"]
            for method in plan["planners"]
        )
        rows.extend({"panel": panel, "reference": name, "label": name} for name in plan["references"])
    return rows


def expected_members(plan):
    members = inherited_members(plan) | {
        "source-plan.json",
        "source-audit-receipt.json",
        "source-completed.json",
        "source-summary.json",
        "parent-provenance.json",
        "random-streams.json",
        "started.json",
        "inherited-fits-ready.json",
        "evaluation-started.json",
        "control-completed.json",
        "diagnostic-completed.json",
        "costs.json",
        "diagnostic/cohort.npz",
        "diagnostic/cohort.json",
    }
    for t in range(plan["steps"]):
        members.update(f"innovations/control/{t:03d}.{suffix}" for suffix in ("npz", "json"))
    for row in control_order(plan):
        stem = f"control/{row['panel']}/{row['label']}"
        members.update(f"{stem}/{file}" for file in ("episodes.npz", "episodes.json", "timings.json"))
        if "reference" in row:
            members.add(f"{stem}/planning.npz")
        else:
            members.add(f"{stem}/executed_predictions.npz")
            members.update(
                f"{stem}/decisions/{t:03d}.{suffix}"
                for t in range(plan["steps"])
                for suffix in ("npz", "json")
            )
    for index in range(plan["diagnostic_episodes"]):
        for ordinal in range(4):
            label = f"{index:03d}-{ordinal:02d}"
            members.update(f"innovations/diagnostic/{label}.{suffix}" for suffix in ("npz", "json"))
            stem = f"diagnostic/roots/{label}"
            members.update(
                f"{stem}/{file}" for file in ("root.json", "union.npz", "native.npz", "timings.json")
            )
            members.update(f"{stem}/history-{panel}.npz" for panel in plan["panels"])
            members.update(
                f"{stem}/search/{panel}/{name}/{method}.{suffix}"
                for panel in plan["panels"]
                for name in plan["fit_order"]
                for method in plan["planners"]
                for suffix in ("npz", "json")
            )
    return members


def checked(path, digest):
    path = Path(path)
    base.require(
        path.is_file() and not path.is_symlink() and protocol.sha(path) == digest,
        f"Source identity mismatch: {path}",
    )
    return path


def validate_parent(plan, execution):
    source = plan["parent_source"]
    base.require(
        set(source) == PARENT_FIELDS and base.read(execution / "parent-provenance.json") == source,
        "Parent source schema/provenance",
    )
    copies = {
        "source-plan.json": (ROOT / source["plan_path"], source["plan_sha256"]),
        "source-audit-receipt.json": (ROOT / source["audit_path"], source["audit_receipt_sha256"]),
        "source-completed.json": (
            ROOT / source["execution_path"] / "completed.json",
            source["completed_sha256"],
        ),
        "source-summary.json": (
            (ROOT / source["audit_path"]).parent / "summary.json",
            source["summary_sha256"],
        ),
    }
    for member, (origin, digest) in copies.items():
        checked(origin, digest)
        checked(execution / member, digest)
    parent, receipt = (
        base.read(execution / "source-plan.json"),
        base.read(execution / "source-audit-receipt.json"),
    )
    completion, summary = (
        base.read(execution / "source-completed.json"),
        base.read(execution / "source-summary.json"),
    )
    base.require(
        parent["study"] == inherited.VERSION
        and parent["version"] == 2
        and receipt["status"] == "completed"
        and receipt["version"] == inherited.VERSION
        and receipt["saved_output_only"] is True
        and receipt["plan_sha256"] == source["plan_sha256"],
        "Parent completed audit identity",
    )
    base.require(
        receipt["source_sha256"] == parent["sources"]
        and receipt["runtime"] == parent["runtime"]
        and receipt["execution_completed_sha256"] == source["completed_sha256"]
        and receipt["files"]["summary.json"] == source["summary_sha256"]
        and receipt["costs"] == source["prior_costs"],
        "Parent receipt nested binding",
    )
    base.require(
        completion["status"] == "completed"
        and completion["plan_sha256"] == source["plan_sha256"]
        and completion["files"] == receipt["execution_members"]
        and completion["wall_seconds"] == source["prior_costs"]["new_evaluation_wall_seconds"],
        "Parent completion/cost binding",
    )
    base.require(
        summary["status"] == "completed"
        and summary["plan_sha256"] == source["plan_sha256"]
        and summary["execution_completed_sha256"] == source["completed_sha256"]
        and summary["costs"] == source["prior_costs"],
        "Parent historical summary binding",
    )
    for name, digest in parent["sources"].items():
        checked(ROOT / name, digest)
    base.require(set(source["members"]) == inherited_members(parent), "Complete inherited member coverage")
    for name, digest in source["members"].items():
        base.require(receipt["execution_members"][name] == digest, "Inherited member audit binding")
        checked(ROOT / source["execution_path"] / name, digest)
        checked(execution / name, digest)
    base.require(set(plan) == set(parent) | PLAN_ADDITIONS, "Search plan field membership")
    for key in set(parent) - PLAN_CHANGES:
        base.require(plan[key] == parent[key], f"Inherited configuration changed: {key}")
    base.require(
        plan["study"] == VERSION
        and type(plan["version"]) is int
        and plan["version"] == 1
        and plan["legacy_seed_fields_unused"] is True
        and plan["steps"] == 50
        and plan["planning_horizon"] == 12
        and plan["action_block"] == 3
        and plan["planners"] == list(protocol.PLANNERS)
        and plan["panels"] == list(protocol.PANELS)
        and plan["references"] == list(protocol.REFERENCES)
        and plan["execution_order"] == control_order(plan),
        "Search scope/order binding",
    )
    base.require(
        plan["search_criteria"]
        == {
            "family_improvement": 0.05,
            "every_residual_fit_strictly_improves": True,
            "panels": ["ordinary", "shift"],
        },
        "Search qualification changed",
    )
    base.require(
        plan["cap_seconds"] == 3600
        and plan["control_episodes"] == 64
        and plan["diagnostic_episodes"] == 16
        and plan["diagnostic_branches"] == 4
        and plan["fit_seeds"] == [271, 283, 293]
        and plan["fit_order"]
        == ["free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293"],
        "Full prespecified study coverage",
    )
    base.require(
        plan["search_parameters"]
        == {
            "elites": 8,
            "minimum_std": 0.001,
            "reward_clip": [-2.5, 0.0],
            "callback_sizes": {"rs64": [64], "rs256": [256], "cem256": [64, 64, 64, 64]},
            "diagnostic_root_offsets": [6, "first_gap_start+2", "ordinary_gap_end", 47],
            "diagnostic_identity_slots": 118,
            "deduplicate_native_sequences": False,
            "warm_start": False,
            "elite_carryover": False,
            "proposal_momentum": 0.0,
        },
        "Search mechanism changed",
    )
    base.require(
        plan["competence_criteria"]
        == {
            "versus_zero_improvement": 0.10,
            "versus_free_family_improvement": 0.05,
            "each_paired_residual_fit_strictly_beats_free": True,
            "panels": ["ordinary", "shift"],
            "physics_panel": "ordinary",
            "planners": list(protocol.PLANNERS),
            "checks_per_planner": 15,
            "memory_qualification": False,
            "prediction_qualification": False,
        },
        "Control competence criteria changed",
    )
    base.require(
        plan["rng_namespace"] == protocol.SCORED_NAMESPACE
        and plan["engineering_rng_namespaces"] == list(protocol.ENGINEERING_NAMESPACES),
        "Scored streams must exclude every engineering namespace",
    )
    # Current source membership/runtime is validated by the CLI before audit_saved.
    # Repeat content hashes here so direct callers cannot silently use edited source.
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    return parent, summary


def validate_streams(plan, parent, execution):
    contract = plan["random_stream_contract"]
    base.require(base.read(execution / "random-streams.json") == contract, "Copied random stream contract")
    priors = [
        *parent["random_stream_contract"]["priors"],
        {
            "plan_path": plan["parent_source"]["plan_path"],
            "plan_sha256": plan["parent_source"]["plan_sha256"],
            "include_train": False,
        },
    ]
    base.require(
        len(priors) == 3 and contract["priors"] == priors, "All three prior stream namespaces required"
    )
    registries = []
    for item in priors:
        base.require(
            set(item) == {"plan_path", "plan_sha256", "include_train"}
            and type(item["include_train"]) is bool,
            "Prior stream schema",
        )
        previous = base.read(checked(ROOT / item["plan_path"], item["plan_sha256"]))
        registries.append(inherited.streams.concrete_streams(previous, include_train=item["include_train"]))
    expected = protocol.stream_contract(plan, registries, priors)
    base.require(contract == expected, "Prospective seed/generator state manifest changed")
    base.require(priors[0]["include_train"] is True, "Original training stream exclusion required")
    return {
        "streams": len(expected["registry"]),
        "generators": len(expected["generators"]),
        "priors_checked": 3,
        "engineering_namespaces_checked": len(expected["engineering_exclusions"]),
        "draws_for_manifest": 0,
        "cross_role_seed_and_initial_state_disjoint": True,
    }


def audit_cohort(plan, records, split, panel):
    count = plan[f"{split}_episodes"]
    base.require(len(records) == count and split in ("control", "diagnostic"), "Fresh cohort coverage")
    transitions, maximum = 0, 0.0
    for index, record in enumerate(records):
        meta = record["metadata"]
        base.require(
            meta["seed"] == protocol.seed(plan, f"{split}/reset/{index}")
            and meta["noise_seed"] == protocol.seed(plan, f"{split}/actuator_noise/{index}")
            and meta["noise_std"] == plan["noise_std"]
            and meta["sensor_schedule"] == protocol.schedule(plan, index, panel, split).tolist()
            and meta["policy_keys"] == ["packets", "commands"],
            "Fresh cohort stream/schedule binding",
        )
        if split == "diagnostic":
            action_seed = protocol.seed(plan, f"diagnostic/exploration/{index}")
            mode = str(
                np.random.default_rng(action_seed).choice(
                    ["ik_pd", "random_low", "random_high"], p=[0.5, 0.25, 0.25]
                )
            )
            base.require(
                meta["action_seed"] == action_seed
                and meta["requested_policy"] == "mixed"
                and meta["collector_policy"] == mode
                and meta["action_hold"] == 4
                and meta["exploration_std"] == {"ik_pd": 0.12, "random_low": 0.3, "random_high": 0.8}[mode],
                "Diagnostic exploration stream",
            )
        replay = native.native_replay(record)
        base.require(
            replay["transitions"] == plan["steps"]
            and replay["new_policy_calls"] == 0
            and replay["saved_output_only"] is True,
            "Native cohort replay completeness",
        )
        transitions += replay["transitions"]
        maximum = max(maximum, replay["max_abs_error"])
    return {"episodes": count, "transitions": transitions, "max_abs_error": maximum}


def common_bank(plan, inputs, step):
    # The callback supplies constant saved-independent values, no model or physics.
    return search(
        "rs64",
        inputs,
        lambda bank: np.zeros(bank.shape[:2], dtype=np.float64),
        step=step,
        steps=plan["steps"],
        planning_horizon=plan["planning_horizon"],
        action_block=plan["action_block"],
    ).sequences


def timing_row(plan, path, learned):
    value = base.read(path)
    extra = (
        {"observation_assimilations", "executed_action_advances"}
        if learned
        else {"candidate_evaluations_per_decision"}
    )
    base.require(
        set(value)
        == {"setup_seconds", "decision_seconds", "native_step_seconds", "row_wall_seconds"} | extra,
        "Control timing schema",
    )
    for key in ("decision_seconds", "native_step_seconds"):
        base.require(
            isinstance(value[key], list) and len(value[key]) == plan["steps"], "Control timing step coverage"
        )
        for item in value[key]:
            base.finite_number(item, key, positive=True)
    for key in ("setup_seconds", "row_wall_seconds"):
        base.finite_number(value[key], key, positive=True)
    base.require(
        value["setup_seconds"] + sum(value["decision_seconds"]) + sum(value["native_step_seconds"])
        <= value["row_wall_seconds"] + 1e-6,
        "Nested control timing exceeds row wall",
    )
    if learned:
        base.require(
            all(
                type(value[key]) is int and value[key] == plan["control_episodes"] * plan["steps"]
                for key in extra
            ),
            "Executed model update counts",
        )
    return value


def audit_control(plan, folder, row, records, inputs):
    n, steps = plan["control_episodes"], plan["steps"]
    learned = "fit" in row
    times = timing_row(plan, folder / "timings.json", learned)
    commands = base.stack(records, "policy", "commands")
    rewards = base.stack(records, "audit", "rewards")
    array(commands, (n, steps, 2), np.float32, "Control command shape/type")
    costs = -rewards.sum(1)
    base.require(np.isfinite(costs).all(), "Finite native control costs")
    counts, imagined, clipped, predictions = 0, 0, 0, 0
    trace_seconds = []
    diagnostics = {}
    if learned:
        for t in range(steps):
            trace = audit_trace(plan, inputs[t], folder / "decisions" / f"{t:03d}", step=t)
            result = trace["result"]
            base.require(result.method == row["planner"], "Control planner identity")
            base.require(
                np.array_equal(commands[:, t], result.selected_actions),
                "Executed action differs from global best",
            )
            base.require(
                trace["search_seconds"] <= times["decision_seconds"][t] + 1e-6, "Uncharged search time"
            )
            counts += result.candidate_evaluations
            imagined += result.imagined_transitions
            clipped += trace["clipped_predictions"]
            predictions += trace["reward_predictions"]
            trace_seconds.append(trace["search_seconds"])
        executed = base.load_npz(folder / "executed_predictions.npz", {"angles", "rewards"})
        angle = array(executed["angles"], (n, steps, 4), np.float32, "Executed angle predictions")
        reward = array(executed["rewards"], (n, steps), np.float32, "Executed reward predictions")
        truth = base.stack(records, "audit", "raw_obs")[:, 1:, :4]
        valid = base.stack(records, "policy", "packets")[:, 1:, 6].astype(bool)
        diagnostics = {
            "on_policy_angle_mse": base.masked_mse(angle, truth, np.ones_like(valid)),
            "on_policy_valid_angle_mse": base.masked_mse(angle, truth, valid),
            "on_policy_blackout_angle_mse": base.masked_mse(angle, truth, ~valid) if (~valid).any() else None,
            "on_policy_reward_mse": float(np.mean((reward.astype(float) - rewards) ** 2)),
        }
    else:
        arm = row["reference"]
        planned = arm in ("known_state", "particle")
        values = base.load_npz(folder / "planning.npz", {"candidate_scores", "planner_used"})
        scores = array(values["candidate_scores"], (n, steps, 64), np.float64, "Reference scores")
        used = array(values["planner_used"], (), np.bool_, "Reference planning flag")
        base.require(
            bool(used) == planned
            and type(times["candidate_evaluations_per_decision"]) is int
            and times["candidate_evaluations_per_decision"] == (64 if planned else 0),
            "Reference budget/flag",
        )
        rng = np.random.default_rng(protocol.seed(plan, "floor/uniform/0"))
        for t in range(steps):
            if planned:
                bank = common_bank(plan, inputs[t], t)
                wanted = bank[np.arange(n), scores[:, t].argmax(1), 0]
                counts += n * 64
                imagined += n * 64 * bank.shape[2]
            else:
                base.require(np.all(scores[:, t] == 0), "Floor scores must be zero placeholders")
                wanted = (
                    np.zeros((n, 2), dtype=np.float32)
                    if arm == "zero"
                    else rng.uniform(-1, 1, (n, 2)).astype(np.float32)
                )
            base.require(np.array_equal(commands[:, t], wanted), "Reference action/score/stream mismatch")
    decisions = np.asarray(times["decision_seconds"], dtype=float)
    return {
        "episode_costs": costs.tolist(),
        "mean_cost": float(costs.mean()),
        "setup_seconds": times["setup_seconds"],
        "decision_wall_seconds": float(decisions.sum()),
        "native_step_seconds": float(sum(times["native_step_seconds"])),
        "row_wall_seconds": times["row_wall_seconds"],
        "batch_latency_seconds": {
            "mean": float(decisions.mean()),
            "p50": float(np.quantile(decisions, 0.5)),
            "p95": float(np.quantile(decisions, 0.95)),
            "max": float(decisions.max()),
        },
        "per_case_amortized_seconds": float(decisions.mean() / n),
        "decision_seconds": times["decision_seconds"],
        "search_seconds": trace_seconds,
        "candidate_evaluations": counts,
        "imagined_transitions": imagined,
        "clipping_fraction": clipped / predictions if predictions else None,
        "planner_used": learned or row.get("reference") in ("known_state", "particle"),
        **diagnostics,
    }


def audit_root(plan, execution, record, index, ordinal):
    label = f"{index:03d}-{ordinal:02d}"
    folder = execution / "diagnostic" / "roots" / label
    meta = base.read(folder / "root.json")
    step = protocol.root_steps(plan, index)[ordinal]
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    inputs = audit_innovations(
        plan, execution / "innovations" / "diagnostic" / label, f"planner/diagnostic/{index}/{ordinal}", 1
    )
    members, traces, order = {}, {}, []
    commands = list(common_bank(plan, inputs, step)[0])
    ids = [f"common/{i}" for i in range(64)]
    for panel in plan["panels"]:
        history_path = folder / f"history-{panel}.npz"
        history = base.load_npz(history_path, {"packets", "commands"})
        expected = protocol.public_history(record, protocol.schedule(plan, index, panel, "diagnostic"), step)
        for key in expected:
            array(history[key], expected[key].shape, np.float32, "Public diagnostic history shape/type")
            base.require(
                np.array_equal(history[key], expected[key]),
                "Diagnostic public history/causal prefix mismatch",
            )
        members[history_path.name] = protocol.sha(history_path)
        for name in plan["fit_order"]:
            for method in plan["planners"]:
                trace = audit_trace(plan, inputs, folder / "search" / panel / name / method, step=step)
                base.require(trace["result"].method == method, "Diagnostic planner identity")
                traces[(panel, name, method)] = trace
                commands.append(trace["result"].selected_sequences[0])
                ids.append(f"selected/{panel}/{name}/{method}")
                order.append(
                    {"panel": panel, "fit": name, "planner": method, "union_index": len(commands) - 1}
                )
    commands = np.stack(commands)
    expected_meta = {
        "episode_index": index,
        "root_ordinal": ordinal,
        "step": step,
        "phase": protocol.phase(plan, index, "diagnostic"),
        "horizon": horizon,
        "union_ids": ids,
        "search_order": order,
        "history_members": members,
        "identity_slots": len(commands),
        "unique_sequence_count": len({row.tobytes() for row in commands}),
        "duplicates_retained": True,
    }
    base.require(meta == expected_meta, "Diagnostic root identity/history/union coverage")
    union = base.load_npz(folder / "union.npz", {"commands", "noise"})
    array(union["commands"], commands.shape, np.float32, "Diagnostic union command shape/type")
    base.require(np.array_equal(union["commands"], commands), "Diagnostic common/selected sequence union")
    noise = np.stack(
        [
            np.random.default_rng(
                protocol.seed(plan, f"diagnostic/branch_noise/{index}/{ordinal}/{branch}")
            ).normal(0.0, plan["noise_std"], (horizon, 2))
            for branch in range(plan["diagnostic_branches"])
        ]
    )
    array(union["noise"], noise.shape, np.float64, "Diagnostic branch noise shape/type")
    base.require(np.array_equal(union["noise"], noise), "Diagnostic branch stream mismatch")
    saved = base.load_npz(folder / "native.npz")
    replay = replay_branches(record, step, commands, noise, saved)
    returns = saved["rewards"].sum(-1)
    metrics = {panel: {} for panel in plan["panels"]}
    for row in order:
        key = (row["panel"], row["fit"], row["planner"])
        metrics[row["panel"]][f"{row['fit']}-{row['planner']}"] = diagnostic_metrics(
            traces[key], returns, row["union_index"]
        )
    times = base.read(folder / "timings.json")
    time_keys = {"belief_seconds", "search_seconds", "native_and_storage_seconds", "row_wall_seconds"}
    base.require(
        set(times)
        == time_keys
        | {"identity_slots", "native_transitions", "observation_assimilations", "history_action_advances"},
        "Diagnostic timing schema",
    )
    for key in time_keys:
        base.finite_number(times[key], key, positive=True)
    base.require(
        math.isclose(
            times["search_seconds"],
            sum(trace["search_seconds"] for trace in traces.values()),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ),
        "Diagnostic search timing arithmetic",
    )
    base.require(
        sum(times[key] for key in time_keys - {"row_wall_seconds"}) <= times["row_wall_seconds"] + 1e-6,
        "Diagnostic nested timings exceed row",
    )
    expected_counts = {
        "identity_slots": len(commands),
        "native_transitions": replay["transitions"],
        "observation_assimilations": len(plan["panels"]) * len(plan["fit_order"]) * (step + 1),
        "history_action_advances": len(plan["panels"]) * len(plan["fit_order"]) * step,
    }
    base.require(
        all(type(times[key]) is int and times[key] == value for key, value in expected_counts.items()),
        "Diagnostic counts",
    )
    return {**expected_meta, "metrics": metrics, "timing": times, "native_replay": replay}


def aggregate_diagnostics(plan, roots):
    expected = {(index, ordinal) for index in range(plan["diagnostic_episodes"]) for ordinal in range(4)}
    base.require(
        len(roots) == len(expected) and {(r["episode_index"], r["root_ordinal"]) for r in roots} == expected,
        "Diagnostic root coverage",
    )
    result = {panel: {} for panel in plan["panels"]}
    for panel in plan["panels"]:
        for name in plan["fit_order"]:
            for method in plan["planners"]:
                label = f"{name}-{method}"
                rows = [root["metrics"][panel][label] for root in roots]
                ranks = [
                    row["common_bank_spearman"] for row in rows if row["common_bank_spearman"] is not None
                ]
                values = {
                    "common_bank_spearman_mean": float(np.mean(ranks)) if ranks else None,
                    "rank_defined_roots": len(ranks),
                    "rank_undefined_roots": len(rows) - len(ranks),
                }
                for key in (
                    "native_selected_return",
                    "finite_set_regret",
                    "selected_raw_prediction_bias",
                    "selected_clipped_prediction_bias",
                    "common_bank_raw_return_mse",
                    "common_bank_clipped_return_mse",
                    "reward_prediction_clipping_fraction",
                ):
                    values[key + "_mean"] = float(np.mean([row[key] for row in rows]))
                result[panel][label] = values
    return {
        "roots": roots,
        "aggregate": result,
        "aggregation": "Equal weight per prespecified physical root. Rank averages exclude undefined roots with counts. Four branches estimate returns conditional on that root; no fit-level confidence interval.",
    }


def audit_boundaries(plan, expected, execution, completed):
    started = base.read(execution / "started.json")
    ready = base.read(execution / "inherited-fits-ready.json")
    evaluation = base.read(execution / "evaluation-started.json")
    control = base.read(execution / "control-completed.json")
    diagnostic = base.read(execution / "diagnostic-completed.json")
    base.require(
        set(started) == {"plan_sha256", "unix_time"}
        and set(ready) == {"plan_sha256", "fit_order", "new_fits", "files", "elapsed_seconds", "unix_time"}
        and set(evaluation)
        == {
            "plan_sha256",
            "elapsed_seconds",
            "unix_time",
            "inherited_fits_ready_sha256",
            "random_streams_sha256",
        }
        and set(control) == {"plan_sha256", "rows", "elapsed_seconds"}
        and set(diagnostic) == {"plan_sha256", "roots", "native_transitions", "elapsed_seconds"},
        "Phase boundary schema",
    )
    base.require(
        all(row["plan_sha256"] == expected for row in (started, ready, evaluation, control, diagnostic)),
        "Phase plan identity",
    )
    base.require(
        type(ready["new_fits"]) is int
        and ready["new_fits"] == 0
        and ready["fit_order"] == plan["fit_order"]
        and ready["files"]
        == {
            f"fits/{name}/weights.pt": plan["parent_source"]["members"][f"fits/{name}/weights.pt"]
            for name in plan["fit_order"]
        },
        "All inherited fits ready before evaluation",
    )
    base.require(
        evaluation["inherited_fits_ready_sha256"] == protocol.sha(execution / "inherited-fits-ready.json")
        and evaluation["random_streams_sha256"] == protocol.sha(execution / "random-streams.json")
        and evaluation["elapsed_seconds"] == completed["evaluation_started_elapsed_seconds"],
        "Evaluation boundary binding",
    )
    elapsed = [
        base.finite_number(row["elapsed_seconds"], "Phase elapsed time", positive=True)
        for row in (ready, evaluation, control, diagnostic)
    ]
    stamps = [
        base.finite_number(row["unix_time"], "Phase timestamp", positive=True)
        for row in (started, ready, evaluation)
    ]
    base.require(
        elapsed == sorted(elapsed) and elapsed[-1] <= completed["wall_seconds"] and stamps == sorted(stamps),
        "Phase chronology",
    )
    base.require(
        type(control["rows"]) is int
        and control["rows"] == len(control_order(plan))
        and type(diagnostic["roots"]) is int
        and diagnostic["roots"] == plan["diagnostic_episodes"] * 4,
        "Completed phase coverage",
    )
    return {
        "inherited_ready_elapsed_seconds": ready["elapsed_seconds"],
        "evaluation_started_elapsed_seconds": evaluation["elapsed_seconds"],
        "control_completed_elapsed_seconds": control["elapsed_seconds"],
        "diagnostic_completed_elapsed_seconds": diagnostic["elapsed_seconds"],
        "diagnostic_native_transitions": diagnostic["native_transitions"],
        "logged_all_fits_before_evaluation": True,
        "independent_process_observer": False,
    }


def audit_costs(plan, execution, completed, boundary, controls, roots):
    saved = base.read(execution / "costs.json")
    keys = {
        "inherited_setup_seconds",
        "innovation_generation_and_storage_seconds",
        "control_row_wall_seconds",
        "control_setup_seconds",
        "control_decision_seconds",
        "control_native_step_seconds",
        "diagnostic_collection_seconds",
        "diagnostic_root_wall_seconds",
    }
    base.require(set(saved) == keys | {"prior_costs", "new_fits", "accounting"}, "Cost schema")
    for key in keys:
        base.finite_number(saved[key], key, positive=True)
    base.require(
        saved["prior_costs"] == plan["parent_source"]["prior_costs"]
        and type(saved["new_fits"]) is int
        and saved["new_fits"] == 0
        and isinstance(saved["accounting"], str)
        and bool(saved["accounting"]),
        "Cost provenance",
    )
    rows = [row for panel in controls.values() for row in panel.values()]
    calculated = {
        "inherited_setup_seconds": boundary["evaluation_started_elapsed_seconds"],
        "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in rows),
        "control_setup_seconds": sum(row["setup_seconds"] for row in rows),
        "control_decision_seconds": sum(row["decision_wall_seconds"] for row in rows),
        "control_native_step_seconds": sum(row["native_step_seconds"] for row in rows),
        "diagnostic_root_wall_seconds": sum(root["timing"]["row_wall_seconds"] for root in roots),
    }
    base.require(
        all(
            math.isclose(saved[key], value, rel_tol=1e-12, abs_tol=1e-8) for key, value in calculated.items()
        ),
        "Cost reconstruction arithmetic",
    )
    control_wall = (
        saved["inherited_setup_seconds"]
        + saved["innovation_generation_and_storage_seconds"]
        + saved["control_row_wall_seconds"]
    )
    native_wall = (
        control_wall + saved["diagnostic_collection_seconds"] + saved["diagnostic_root_wall_seconds"]
    )
    base.require(
        control_wall <= boundary["control_completed_elapsed_seconds"] + 1e-6
        and native_wall <= boundary["diagnostic_completed_elapsed_seconds"] + 1e-6
        and native_wall <= completed["wall_seconds"] + 1e-6,
        "Nonoverlapping costs exceed phase/whole-run timing",
    )
    prior = saved["prior_costs"]
    fitwall = base.finite_number(prior["inherited_fit_wall_seconds"], "Inherited fit cost", positive=True)
    oldwall = base.finite_number(
        prior["cumulative_attempt_wall_seconds"], "Prior cumulative cost", positive=True
    )
    base.require(
        completed["inherited_fit_wall_seconds"] == fitwall
        and completed["prior_cumulative_attempt_wall_seconds"] == oldwall
        and completed["cumulative_attempt_wall_seconds"] == oldwall + completed["wall_seconds"],
        "Cumulative cost double counting/binding",
    )
    return {
        **saved,
        "new_evaluation_wall_seconds": completed["wall_seconds"],
        "inherited_fit_wall_seconds": fitwall,
        "prior_cumulative_attempt_wall_seconds": oldwall,
        "cumulative_attempt_wall_seconds": oldwall + completed["wall_seconds"],
        "fresh_evaluation_plus_inherited_fits_seconds": completed["wall_seconds"] + fitwall,
    }


def validate_members(plan, expected_hash, execution):
    base.require(
        isinstance(expected_hash, str)
        and len(expected_hash) == 64
        and all(c in "0123456789abcdef" for c in expected_hash),
        "External plan SHA",
    )
    completion = base.read(execution / "completed.json")
    keys = {
        "status",
        "version",
        "plan_sha256",
        "files",
        "fits",
        "new_fits",
        "astra_calls",
        "wall_seconds",
        "evaluation_started_elapsed_seconds",
        "control_rows",
        "diagnostic_roots",
        "inherited_fit_wall_seconds",
        "prior_cumulative_attempt_wall_seconds",
        "cumulative_attempt_wall_seconds",
    }
    base.require(
        set(completion) == keys
        and completion["status"] == "completed"
        and completion["version"] == VERSION
        and completion["plan_sha256"] == expected_hash,
        "Completed study identity/schema",
    )
    for key, value in {
        "fits": len(plan["fit_order"]),
        "new_fits": 0,
        "astra_calls": 0,
        "control_rows": len(control_order(plan)),
        "diagnostic_roots": plan["diagnostic_episodes"] * 4,
    }.items():
        base.require(type(completion[key]) is int and completion[key] == value, f"Completed coverage: {key}")
    base.require(
        base.finite_number(completion["wall_seconds"], "Execution wall", positive=True)
        <= plan["cap_seconds"],
        "Whole-run cap exceeded",
    )
    members = expected_members(plan)
    paths = list(execution.rglob("*"))
    base.require(
        not execution.is_symlink() and not any(path.is_symlink() for path in paths),
        "Execution symlinks forbidden",
    )
    actual = {str(path.relative_to(execution)) for path in paths if path.is_file()}
    base.require(
        actual == members | {"completed.json"} and set(completion["files"]) == members,
        "Complete exact execution membership",
    )
    for member in members:
        checked(execution / member, completion["files"][member])
    return completion


def audit_saved(plan, expected_plan_sha256, execution: Path, out: Path):
    """Audit complete artifacts. The CLI separately authenticates runtime and plan bytes."""
    import hashlib

    execution, out = Path(execution), Path(out)
    base.require(not out.exists(), "Exclusive audit output required")
    begin = time.monotonic()
    completed = validate_members(plan, expected_plan_sha256, execution)
    completion_hash = protocol.sha(execution / "completed.json")
    parent, historical = validate_parent(plan, execution)
    streams = validate_streams(plan, parent, execution)
    old_plan, inheritance = inherited.validate_inheritance(parent, execution)
    training_plan = inherited.prior.validate_training_source(parent, execution)
    train = base.load_records(execution / "train", plan["train_episodes"])
    replay = {"inherited_training": base.audit_cohort(training_plan, train, "train")}
    fits, initial = {}, {}
    for name in plan["fit_order"]:
        fits[name], initial[name] = inherited.prior.audit_fit(old_plan, execution, name, train)
        base.require(
            {key: value for key, value in fits[name].items() if key != "epochs"} == inheritance["fits"][name],
            "Inherited fit metadata audit mismatch",
        )
    for seed in plan["fit_seeds"]:
        left, right = initial[f"free-{seed}"], initial[f"residual-{seed}"]
        base.require(
            set(left) == set(right) and all(inherited.torch.equal(left[key], right[key]) for key in left),
            "Inherited paired initial tensors differ",
        )
        base.require(
            fits[f"free-{seed}"]["minibatch_order_sha256"]
            == fits[f"residual-{seed}"]["minibatch_order_sha256"],
            "Inherited paired training order differs",
        )
    del initial
    base.require(
        math.isclose(
            sum(fit["wall_seconds"] for fit in fits.values()),
            plan["parent_source"]["prior_costs"]["inherited_fit_wall_seconds"],
            rel_tol=1e-12,
            abs_tol=1e-9,
        ),
        "Inherited fit cost arithmetic",
    )
    boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed)
    inputs = [
        audit_innovations(
            plan,
            execution / "innovations" / "control" / f"{t:03d}",
            f"planner/control/{t}",
            plan["control_episodes"],
        )
        for t in range(plan["steps"])
    ]
    controls, initial_states = {panel: {} for panel in plan["panels"]}, None
    for row in control_order(plan):
        folder = execution / "control" / row["panel"] / row["label"]
        records = base.load_records(folder / "episodes", plan["control_episodes"])
        replay[f"control/{row['panel']}/{row['label']}"] = audit_cohort(
            plan, records, "control", row["panel"]
        )
        starts = base.stack(records, "audit", "integration_state")[:, 0]
        if initial_states is None:
            initial_states = starts
        base.require(np.array_equal(initial_states, starts), "Control case pairing differs")
        controls[row["panel"]][row["label"]] = audit_control(plan, folder, row, records, inputs)
    cohort = base.load_records(execution / "diagnostic" / "cohort", plan["diagnostic_episodes"])
    replay["diagnostic/cohort"] = audit_cohort(plan, cohort, "diagnostic", "full")
    roots = [
        audit_root(plan, execution, record, index, ordinal)
        for index, record in enumerate(cohort)
        for ordinal in range(4)
    ]
    for root in roots:
        replay[f"diagnostic/{root['episode_index']}/{root['root_ordinal']}"] = root["native_replay"]
    base.require(
        type(boundary["diagnostic_native_transitions"]) is int
        and boundary["diagnostic_native_transitions"]
        == sum(root["native_replay"]["transitions"] for root in roots),
        "Diagnostic completed transition count",
    )
    native_counts = {
        "control": len(control_order(plan)) * plan["control_episodes"] * plan["steps"],
        "diagnostic_cohort": plan["diagnostic_episodes"] * plan["steps"],
        "diagnostic_branches": sum(
            (64 + len(plan["panels"]) * len(plan["fit_order"]) * len(plan["planners"]))
            * plan["diagnostic_branches"]
            * min(plan["planning_horizon"], plan["steps"] - step)
            for index in range(plan["diagnostic_episodes"])
            for step in protocol.root_steps(plan, index)
        ),
        "inherited_training": plan["train_episodes"] * plan["steps"],
    }
    native_counts["fresh_evaluation"] = sum(
        value for key, value in native_counts.items() if key != "inherited_training"
    )
    base.require(
        sum(row["transitions"] for row in replay.values())
        == native_counts["fresh_evaluation"] + native_counts["inherited_training"],
        "Derived native transition totals",
    )
    learned_imagined = sum(
        row["imagined_transitions"]
        for panel in controls.values()
        for name, row in panel.items()
        if name not in plan["references"]
    )
    wanted_imagined = (
        len(plan["panels"])
        * len(plan["fit_order"])
        * plan["control_episodes"]
        * (64 + 256 + 256)
        * sum(min(plan["planning_horizon"], plan["steps"] - t) for t in range(plan["steps"]))
    )
    base.require(learned_imagined == wanted_imagined, "Derived learned imagined transition totals")
    costs = audit_costs(plan, execution, completed, boundary, controls, roots)
    gate, competence, comparisons = qualify_search(plan, controls)
    costs["audit_validation_wall_seconds"] = time.monotonic() - begin
    summary = {
        "version": VERSION,
        "status": "completed",
        "plan_sha256": expected_plan_sha256,
        "execution_completed_sha256": completion_hash,
        "saved_output_only": True,
        "new_model_calls": 0,
        "new_policy_calls": 0,
        "new_mpc_calls": 0,
        "new_fits": 0,
        "native_transitions_checked": sum(row["transitions"] for row in replay.values()),
        "native_max_abs_error": max(row["max_abs_error"] for row in replay.values()),
        "native_transition_counts": native_counts,
        "learned_control_imagined_transitions": learned_imagined,
        "wall_seconds": completed["wall_seconds"],
        "costs": costs,
        "source_lineage": plan["parent_source"],
        "random_streams": streams,
        "phase_boundary": boundary,
        "fits": fits,
        "control": controls,
        "continuation_gate": gate,
        "control_competence": competence,
        "historical_parent_qualification": {
            "plan_sha256": plan["parent_source"]["plan_sha256"],
            "audit_receipt_sha256": plan["parent_source"]["audit_receipt_sha256"],
            "continuation_gate": historical["continuation_gate"],
            "rerun": False,
        },
        "paired_descriptive_comparisons": comparisons,
        "diagnostics": aggregate_diagnostics(plan, roots),
        "cohorts": replay,
        "limits": [
            "All six final fits are inherited unchanged and unselected. No new training, prediction-cohort or memory/reset qualification is performed.",
            "The primary search gate measures actual native control cost. Control competence is separately reported for every planner. Historical 17 checks remain historical.",
            "Diagnostic regret is relative to the evaluated finite union, not a globally optimal controller. Duplicate identity slots are retained and unique sequences counted.",
            "Rank correlation excludes undefined constant-vector roots and reports coverage. Four noise branches and case bootstrap intervals are descriptive conditional on the fixed fits.",
            "Native replay checks stored actions and physical outcomes. Proposal reconstruction reads recorded model scores; it does not independently verify neural forward outputs.",
            "Raw and clipped predictions are retained separately. Supplied-physics references have model/state privileges; their scored values are not learned predictions.",
            "Wall time includes setup, proposal generation, scoring, state updates and storage. Counts describe scored candidates and imagined steps, not total FLOPs or single-agent deadlines.",
            "The prior cumulative cost already includes fitting. New evaluation cost is added once. Phase receipts establish source-bound logged chronology, not an independent observer.",
        ],
    }
    # Refuse concurrent mutation, inserted partials, missing roots, or terminal changes.
    base.require(
        validate_members(plan, expected_plan_sha256, execution) == completed
        and protocol.sha(execution / "completed.json") == completion_hash,
        "Execution changed during audit",
    )
    out.mkdir(parents=True, exist_ok=False)
    base.write(out / "summary.json", summary)
    lines = [
        "# Reacher adaptive-search saved-output audit",
        "",
        f"Primary search gate: **{'PASS' if gate['passed'] else 'FAIL'}** ({sum(row['passed'] for row in gate['checks'])}/{len(gate['checks'])}).",
        "",
        f"{len(fits)} inherited fits; {len(control_order(plan))} control rows; {len(roots)} diagnostic roots; {summary['native_transitions_checked']:,} native transitions checked.",
        "",
        "| Panel | Arm | Mean native cost |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| {panel} | {name} | {row['mean_cost']:.6f} |"
        for panel, rows in controls.items()
        for name, row in rows.items()
    )
    lines.extend(["", *(f"- {limit}" for limit in summary["limits"])])
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
            "parent_source": plan["parent_source"],
            "random_stream_contract": plan["random_stream_contract"],
            "costs": costs,
            "saved_output_only": True,
            "files": {name: protocol.sha(out / name) for name in ("summary.json", "README.md")},
        },
    )
    return summary
