"""Bounded, evaluation-only search comparison on all six inherited Reacher fits."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import reacher_reward_residual_control as recovery
import torch

from openjev.research import reacher_search_protocol as protocol
from openjev.research.reacher_adaptive_search import ANCHORS, search
from openjev.research.reacher_physics_control import ParticleFilter, PhysicsMPC
from openjev.research.reacher_random_streams import concrete_streams
from openjev.research.reacher_world_models import repeat_index
from openjev.research.robotics_reacher import (
    ReacherEpisode,
    _integration_state,
    collect_episode,
    make_env,
    restore_native,
)

ROOT = Path(__file__).resolve().parents[1]
base, legacy = recovery.base, recovery.legacy
sha, read, write, require = protocol.sha, protocol.read, protocol.write, protocol.require
check_cap = base.check_cap
STUDY = protocol.STUDY
PARENT_PLAN = "evidence/reacher-reward-residual-control-v2/protocol/plan.json"
PARENT_PLAN_SHA = "4c21a1505245314a923325ca0aae141daed6a737170d870272860bd00857561f"
PARENT_AUDIT = "evidence/reacher-reward-residual-control-v2/audit/receipt.json"
PARENT_AUDIT_SHA = "af87d1e4f1aa4bfe72a2730e6d33c1a6a455b573ea4fb27c240812bc61c434dd"
PARENT_EXECUTION = "runs/reacher-reward-residual-control-v2/execution"
SOURCES = recovery.SOURCES + (
    "src/openjev/research/reacher_adaptive_search.py",
    "tests/test_reacher_adaptive_search.py",
    "src/openjev/research/reacher_search_protocol.py",
    "scripts/reacher_search_study.py",
    "scripts/audit_reacher_search_study.py",
    "tests/test_reacher_search_study.py",
    "tests/test_audit_reacher_search_study.py",
)
INHERITED_FILES = (
    "train.npz", "train.json", "train-provenance.json", "fit-provenance.json", "inherited-plan.json",
    "invalid-attempt.json", "fit-inheritance.json", "source-all-fits-completed.json", "source-failed.json",
)
PARENT_SOURCE_FIELDS = {
    "plan_path", "plan_sha256", "audit_path", "audit_receipt_sha256", "execution_path",
    "completed_sha256", "summary_sha256", "members", "prior_costs",
}
CHANGED_FIELDS = {"study", "version", "sources", "runtime", "cap_seconds", "data", "models", "control",
                  "criteria", "stop", "random_stream_contract"}
NEW_FIELDS = {"parent_source", "planners", "panels", "references", "diagnostic_episodes",
              "diagnostic_branches", "execution_order", "search_parameters", "search_criteria",
              "competence_criteria", "legacy_seed_fields_unused", "rng_namespace", "engineering_rng_namespaces"}


def checked(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, f"Identity mismatch: {path}")
    return path


def inherited_members(parent):
    return set(INHERITED_FILES) | {f"fits/{name}/{file}" for name in parent["fit_order"]
                                  for file in ("initial-weights.pt", "weights.pt", "training.json", "completed.json")}


def default_parent_source():
    parent = read(checked(ROOT / PARENT_PLAN, PARENT_PLAN_SHA))
    receipt = read(checked(ROOT / PARENT_AUDIT, PARENT_AUDIT_SHA))
    result = {
        "plan_path": PARENT_PLAN, "plan_sha256": PARENT_PLAN_SHA,
        "audit_path": PARENT_AUDIT, "audit_receipt_sha256": PARENT_AUDIT_SHA,
        "execution_path": PARENT_EXECUTION, "completed_sha256": receipt["execution_completed_sha256"],
        "summary_sha256": receipt["files"]["summary.json"],
        "members": {name: receipt["execution_members"][name] for name in sorted(inherited_members(parent))},
        "prior_costs": dict(receipt["costs"]),
    }
    authenticate_parent(result)
    return result


def authenticate_parent(source):
    require(set(source) == PARENT_SOURCE_FIELDS, "Parent source fields")
    parent = read(checked(ROOT / source["plan_path"], source["plan_sha256"]))
    receipt_path = checked(ROOT / source["audit_path"], source["audit_receipt_sha256"])
    receipt = read(receipt_path)
    require(parent["study"] == "reacher-reward-residual-control-v2" and parent["version"] == 2,
            "Parent study identity")
    require(receipt["status"] == "completed" and receipt["version"] == parent["study"]
            and receipt["plan_sha256"] == source["plan_sha256"] and receipt["saved_output_only"] is True,
            "Parent audited terminal boundary")
    require(receipt["source_sha256"] == parent["sources"] and receipt["runtime"] == parent["runtime"],
            "Parent audit source/runtime binding")
    for name, digest in parent["sources"].items():
        checked(ROOT / name, digest)
    require(set(source["members"]) == inherited_members(parent), "All six inherited fits and provenance required")
    require(source["members"] == {name: receipt["execution_members"][name] for name in source["members"]},
            "Inherited member binding to authoritative audit")
    require(source["completed_sha256"] == receipt["execution_completed_sha256"]
            and source["summary_sha256"] == receipt["files"]["summary.json"]
            and source["prior_costs"] == receipt["costs"], "Parent receipt nested binding")
    execution = ROOT / source["execution_path"]
    require(execution.is_dir() and not execution.is_symlink() and not (execution / "failed.json").exists(),
            "Parent execution must be completed")
    completed = read(checked(execution / "completed.json", source["completed_sha256"]))
    require(completed["status"] == "completed" and completed["plan_sha256"] == source["plan_sha256"]
            and completed["files"] == receipt["execution_members"], "Parent execution/audit members")
    checked(receipt_path.parent / "summary.json", source["summary_sha256"])
    for name, digest in source["members"].items():
        checked(execution / name, digest)
    require(parent["fit_order"] == ["free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293"]
            and parent["kinds"] == ["free", "residual"] and parent["fit_seeds"] == [271, 283, 293],
            "All six original final fits in unchanged order")
    for name in parent["fit_order"]:
        fit = read(execution / "fits" / name / "completed.json")
        kind, fit_seed = name.rsplit("-", 1)
        require(fit["kind"] == kind and fit["seed"] == int(fit_seed), "Inherited fit identity")
        for file, digest in fit["files"].items():
            require(source["members"][f"fits/{name}/{file}"] == digest, "Inherited nested fit hash")
    for key in ("inherited_fit_wall_seconds", "cumulative_attempt_wall_seconds", "new_evaluation_wall_seconds"):
        require(math.isfinite(source["prior_costs"][key]) and source["prior_costs"][key] > 0, "Prior cost boundary")
    require(source["prior_costs"]["new_evaluation_wall_seconds"] == completed["wall_seconds"],
            "Parent actual execution cost")
    return parent


def execution_order(plan):
    rows = []
    for panel in plan["panels"]:
        for name in plan["fit_order"]:
            rows.extend({"panel": panel, "fit": name, "planner": method, "label": f"{name}-{method}"}
                        for method in plan["planners"])
        rows.extend({"panel": panel, "reference": arm, "label": arm} for arm in plan["references"])
    return rows


def stream_contract(plan):
    parent = read(checked(ROOT / plan["parent_source"]["plan_path"], plan["parent_source"]["plan_sha256"]))
    priors = [*parent["random_stream_contract"]["priors"], {
        "plan_path": plan["parent_source"]["plan_path"],
        "plan_sha256": plan["parent_source"]["plan_sha256"], "include_train": False,
    }]
    previous = [concrete_streams(read(checked(ROOT / row["plan_path"], row["plan_sha256"])),
                                 include_train=row["include_train"]) for row in priors]
    return protocol.stream_contract(plan, previous, priors)


def validate_settings(plan, parent):
    require(set(plan) == set(parent) | NEW_FIELDS, "Search plan field membership")
    for key, value in parent.items():
        if key not in CHANGED_FIELDS:
            require(plan[key] == value, f"Inherited configuration changed: {key}")
    require(plan["study"] == STUDY and type(plan["version"]) is int
            and plan["version"] == 1 and plan["cap_seconds"] == 3600,
            "Search study/version/cap")
    require(plan["rng_namespace"] == protocol.SCORED_NAMESPACE
            and plan["engineering_rng_namespaces"] == list(protocol.ENGINEERING_NAMESPACES),
            "Scored namespace and all engineering exclusions must remain frozen")
    require(plan["steps"] == 50 and plan["control_episodes"] == 64 and plan["planning_horizon"] == 12
            and plan["action_block"] == 3 and plan["candidates"] == 64, "Fixed native/search scope")
    require(plan["diagnostic_episodes"] == 16 and plan["diagnostic_branches"] == 4, "Diagnostic scope")
    require(plan["panels"] == list(protocol.PANELS) and plan["planners"] == list(protocol.PLANNERS)
            and plan["references"] == list(protocol.REFERENCES), "All planner/panel/reference arms")
    require(plan["execution_order"] == execution_order(plan), "Complete frozen execution order")
    require(plan["search_parameters"] == {
        "elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
        "callback_sizes": {"rs64": [64], "rs256": [256], "cem256": [64, 64, 64, 64]},
        "diagnostic_root_offsets": [6, "first_gap_start+2", "ordinary_gap_end", 47],
        "diagnostic_identity_slots": 118, "deduplicate_native_sequences": False,
        "warm_start": False, "elite_carryover": False, "proposal_momentum": 0.,
    }, "Search mechanism changed")
    require(plan["search_criteria"] == {"family_improvement": .05, "every_residual_fit_strictly_improves": True,
                                        "panels": ["ordinary", "shift"]}, "Search gate changed")
    require(plan["competence_criteria"] == {
        "versus_zero_improvement": .10, "versus_free_family_improvement": .05,
        "each_paired_residual_fit_strictly_beats_free": True, "panels": ["ordinary", "shift"],
        "physics_panel": "ordinary", "planners": list(protocol.PLANNERS), "checks_per_planner": 15,
        "memory_qualification": False, "prediction_qualification": False,
    },
            "Competence boundary changed")
    require(plan["legacy_seed_fields_unused"] is True, "New named-stream interface required")


def prepare(out, *, parent_source=None):
    source = default_parent_source() if parent_source is None else parent_source
    parent = authenticate_parent(source)
    plan = dict(parent)
    plan.update(
        study=STUDY, version=1, cap_seconds=3600, sources={name: sha(ROOT / name) for name in SOURCES},
        rng_namespace=protocol.SCORED_NAMESPACE, engineering_rng_namespaces=list(protocol.ENGINEERING_NAMESPACES),
        runtime=base.runtime(), parent_source=source, planners=list(protocol.PLANNERS),
        panels=list(protocol.PANELS), references=list(protocol.REFERENCES), diagnostic_episodes=16,
        diagnostic_branches=4, legacy_seed_fields_unused=True,
        search_parameters={
            "elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
            "callback_sizes": {"rs64": [64], "rs256": [256], "cem256": [64, 64, 64, 64]},
            "diagnostic_root_offsets": [6, "first_gap_start+2", "ordinary_gap_end", 47],
            "diagnostic_identity_slots": 118, "deduplicate_native_sequences": False,
            "warm_start": False, "elite_carryover": False, "proposal_momentum": 0.,
        },
        search_criteria={"family_improvement": .05, "every_residual_fit_strictly_improves": True,
                         "panels": ["ordinary", "shift"]},
        competence_criteria={
            "versus_zero_improvement": .10, "versus_free_family_improvement": .05,
            "each_paired_residual_fit_strictly_beats_free": True, "panels": ["ordinary", "shift"],
            "physics_panel": "ordinary", "planners": list(protocol.PLANNERS), "checks_per_planner": 15,
            "memory_qualification": False, "prediction_qualification": False,
        },
        data="No fresh fitting or held-out prediction cohort. Inherit all six final fits unchanged. 64 new paired control cases; 16 separate mixed-policy diagnostic episodes. Every seed/generator excludes original training and all three prior evaluations.",
        models="Load all six authenticated final checkpoints before any fresh draw or learned forward call. Preserve original model and reward-head configuration. No checkpoint/model selection; all results retained.",
        control="All six fits x three planners x three panels, 64 cases each. Four references per panel use the same new 64-candidate bank. Full-horizon draws, explicit CEM proposal trace and raw/clipped imagined rewards. No reset intervention or new memory claim.",
        criteria="Eight search checks: residual CEM256 improves family mean cost by at least 5% versus RS256 on ordinary and shifted panels, and every paired residual fit strictly improves both. Each planner retains 15 fresh competence checks: physics ordinary improves at least 10% versus zero; all residual fits improve at least 10% versus zero and strictly versus paired free fits on both panels; residual family improves at least 5% versus free on both. The historical 17 criteria remain historical; no fresh prediction or memory qualification.",
        stop="Single 3600-second cooperative cap from entry through validation, copying, all loading/collection/scoring/diagnostics, trace writes and final hashing. No retries, resumes, substitute seeds or extensions. Exclusive artifacts and preserved partial/failure receipts. Independent audit uses saved proposals/rewards and native replay only.",
    )
    plan["execution_order"] = execution_order(plan)
    plan["random_stream_contract"] = stream_contract(plan)
    validate_settings(plan, parent)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, expected):
    plan = read(checked(path, expected))
    require(set(plan["sources"]) == set(SOURCES), "Frozen source membership")
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    require(plan["runtime"] == base.runtime(), "Frozen runtime changed")
    validate_settings(plan, authenticate_parent(plan["parent_source"]))
    require(plan["random_stream_contract"] == stream_contract(plan), "Named stream manifest changed")
    return plan


@torch.no_grad()
def score_search(model, state, inputs, method, step, plan, deadline):
    start = time.perf_counter()
    raw, sizes = [], []

    def score(bank):
        check_cap(deadline)
        n, k, horizon, _ = bank.shape
        imagined = repeat_index(state, torch.arange(n).repeat_interleave(k))
        total, steps = torch.zeros(n * k, dtype=torch.float32), []
        for t in range(horizon):
            check_cap(deadline)
            imagined, _, reward = model.advance(imagined, torch.from_numpy(bank[:, :, t].reshape(n * k, 2).copy()))
            require(reward.shape == (n * k,) and reward.dtype == torch.float32
                    and torch.isfinite(reward).all().item(), "Invalid shape, dtype or nonfinite learned reward prediction")
            total += reward.clamp(-2.5, 0.)
            steps.append(reward.reshape(n, k).numpy().copy())
        raw.append(np.stack(steps, axis=2))
        sizes.append(k)
        return total.reshape(n, k).numpy()

    result = search(method, inputs, score, step=step, steps=plan["steps"],
                    planning_horizon=plan["planning_horizon"], action_block=plan["action_block"])
    check_cap(deadline)
    return result, np.concatenate(raw, axis=1), sizes, time.perf_counter() - start


def common_bank(inputs, step, plan):
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    require(horizon > 0, "No candidate bank beyond terminal boundary")
    chunks = (horizon + plan["action_block"] - 1) // plan["action_block"]
    value = inputs.initial[:, :, :chunks] * np.r_[np.full(32, .25), np.full(32, .75)][None, :, None, None]
    for i, anchor in enumerate(ANCHORS):
        value[:, i] = anchor
    return np.repeat(np.clip(value, -1, 1).astype(np.float32), plan["action_block"], axis=2)[:, :, :horizon]


def control_envs(plan, panel, deadline=float("inf")):
    envs, packets = [], []
    try:
        for i in range(plan["control_episodes"]):
            check_cap(deadline)
            env = ReacherEpisode(noise_std=plan["noise_std"])
            envs.append(env)
            packets.append(env.reset(protocol.seed(plan, f"control/reset/{i}"),
                protocol.schedule(plan, i, panel), noise_seed=protocol.seed(plan, f"control/actuator_noise/{i}")))
        return envs, np.stack(packets)
    except BaseException:
        for env in envs:
            env.close()
        raise


def save_episodes(stem, envs):
    require(not Path(stem).with_suffix(".npz").exists() and not Path(stem).with_suffix(".json").exists(),
            "Episode output must be exclusive")
    base.save_records(Path(stem), [env.episode_record() for env in envs])


def save_partial_episodes(stem, envs):
    records = [env.episode_record() for env in envs]
    lengths = [len(record["policy"]["commands"]) for record in records]
    if len(set(lengths)) == 1:
        base.save_records(Path(stem), records)
    else:
        # A native error can occur halfway through a case batch. Preserve each
        # case separately rather than attempting to stack unequal histories.
        folder = Path(stem)
        folder.mkdir(parents=True, exist_ok=False)
        for i, record in enumerate(records):
            base.save_records(folder / f"{i:03d}", [record])
        write(folder / "manifest.json", {"completed_steps_by_case": lengths})


@torch.no_grad()
def learned_control(plan, model, panel, method, inputs_by_step, out, deadline, progress):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    envs = []
    try:
        envs, packets = control_envs(plan, panel, deadline)
        state = model.initial(len(envs))
        setup = time.perf_counter() - begin
        decisions, native_times, angles, rewards = [], [], [], []
        for t in range(plan["steps"]):
            progress["step"] = t
            check_cap(deadline)
            tick = time.perf_counter()
            inputs = protocol.load_inputs(inputs_by_step[t])
            state = model.assimilate(state, torch.from_numpy(packets))
            result, raw, sizes, search_seconds = score_search(model, state, inputs, method, t, plan, deadline)
            state, predicted_angles, predicted_reward = model.advance(
                state, torch.from_numpy(result.selected_actions.copy()))
            require(torch.isfinite(predicted_angles).all().item() and torch.isfinite(predicted_reward).all().item(),
                    "Nonfinite executed-action prediction")
            angles.append(predicted_angles.numpy().copy())
            rewards.append(predicted_reward.numpy().copy())
            protocol.save_trace(out / "decisions" / f"{t:03d}", result, raw, sizes, search_seconds)
            decisions.append(time.perf_counter() - tick)
            check_cap(deadline)
            tick = time.perf_counter()
            packets = np.stack([env.step(action) for env, action in zip(envs, result.selected_actions, strict=True)])
            native_times.append(time.perf_counter() - tick)
        save_episodes(out / "episodes", envs)
        protocol.save_npz(out / "executed_predictions.npz", angles=np.stack(angles, 1), rewards=np.stack(rewards, 1))
        result = {"setup_seconds": setup, "decision_seconds": decisions, "native_step_seconds": native_times,
                  "row_wall_seconds": time.perf_counter() - begin,
                  "observation_assimilations": len(envs) * plan["steps"],
                  "executed_action_advances": len(envs) * plan["steps"]}
        write(out / "timings.json", result)
        check_cap(deadline)
        return result
    except BaseException:
        if envs and not (out / "episodes.npz").exists():
            save_partial_episodes(out / "partial-episodes", envs)
        raise
    finally:
        for env in envs:
            env.close()


def reference_control(plan, panel, arm, inputs_by_step, out, deadline, progress):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    envs, template = [], None
    try:
        envs, packets = control_envs(plan, panel, deadline)
        template = make_env()
        physics = PhysicsMPC(template.unwrapped.model, frame_skip=2)
        filters = [ParticleFilter(template.unwrapped.model, p,
            seed=protocol.seed(plan, f"planner/particle_filter/{i}"), particles=plan["particles"],
            noise_std=plan["noise_std"], measurement_std=plan["filter_bandwidth"], frame_skip=2)
            for i, p in enumerate(packets)] if arm == "particle" else []
        uniform = np.random.default_rng(protocol.seed(plan, "floor/uniform/0"))
        previous = np.zeros((len(envs), 2), dtype=np.float32)
        setup = time.perf_counter() - begin
        scores, decisions, native_times = [], [], []
        for t in range(plan["steps"]):
            progress["step"] = t
            check_cap(deadline)
            tick = time.perf_counter()
            score = np.zeros((len(envs), 64), dtype=np.float64)
            if arm == "zero":
                actions = np.zeros((len(envs), 2), dtype=np.float32)
            elif arm == "uniform":
                actions = uniform.uniform(-1, 1, (len(envs), 2)).astype(np.float32)
            else:
                require(arm in ("known_state", "particle"), "Unknown reference arm")
                bank = common_bank(protocol.load_inputs(inputs_by_step[t]), t, plan)
                actions = []
                for i, env in enumerate(envs):
                    check_cap(deadline)
                    if arm == "known_state":
                        record = env.audit_record()
                        qpos, qvel = record["qpos"], record["qvel"]
                    else:
                        if t:
                            filters[i].update(previous[i], packets[i])
                        qpos, qvel = filters[i].estimate()
                    action, costs = physics.plan(qpos, qvel, bank[i])
                    actions.append(action)
                    score[i] = -costs
                actions = np.asarray(actions, dtype=np.float32)
            scores.append(score)
            decisions.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            packets = np.stack([env.step(action) for env, action in zip(envs, actions, strict=True)])
            native_times.append(time.perf_counter() - tick)
            previous = actions
        save_episodes(out / "episodes", envs)
        protocol.save_npz(out / "planning.npz", candidate_scores=np.stack(scores, 1),
                          planner_used=np.array(arm in ("known_state", "particle")))
        result = {"setup_seconds": setup, "decision_seconds": decisions, "native_step_seconds": native_times,
                  "row_wall_seconds": time.perf_counter() - begin,
                  "candidate_evaluations_per_decision": 64 if arm in ("known_state", "particle") else 0}
        write(out / "timings.json", result)
        check_cap(deadline)
        return result
    except BaseException:
        if envs and not (out / "episodes.npz").exists():
            save_partial_episodes(out / "partial-episodes", envs)
        raise
    finally:
        for env in envs:
            env.close()
        if template is not None:
            template.close()


@torch.no_grad()
def belief_at(model, packets, commands, deadline):
    require(packets.dtype == commands.dtype == np.float32 and packets.shape == (len(commands) + 1, 8)
            and commands.ndim == 2 and commands.shape[1] == 2, "Public history shapes/dtypes")
    state = model.initial(1)
    for t, observation in enumerate(packets):
        check_cap(deadline)
        state = model.assimilate(state, torch.from_numpy(observation[None].copy()))
        if t < len(commands):
            state, _, _ = model.advance(state, torch.from_numpy(commands[t:t + 1].copy()))
    return state


def native_branch(root_record, commands, noise, deadline, env=None):
    """Clone complete physics/time-limit state; disturbance is explicit, never replay RNG."""
    commands, noise = np.asarray(commands), np.asarray(noise)
    require(commands.ndim == 2 and commands.shape[1] == 2 and noise.shape == commands.shape
            and commands.dtype == np.float32 and noise.dtype == np.float64
            and np.isfinite(commands).all() and np.isfinite(noise).all()
            and np.all(np.abs(commands) <= 1), "Invalid native branch commands/noise")
    root = root_record["step"]
    require(type(root) is int and 0 <= root < 50 and 1 <= len(commands) <= 50 - root,
            "Invalid native branch time boundary")
    own = env is None
    if own:
        env = make_env()
        # This initialization is solely for standalone engineering fixtures.
        # Scored execution always supplies its named-seed initialized template.
        env.reset(seed=0)
    state_keys = ("qpos", "qvel", "raw_obs", "integration_state", "time")
    records = {key: [] for key in state_keys}
    records.update({key: [] for key in ("rewards", "reward_dist", "reward_ctrl", "applied_actions",
                                       "terminated", "truncated")})

    def capture():
        native = env.unwrapped
        records["qpos"].append(native.data.qpos.copy())
        records["qvel"].append(native.data.qvel.copy())
        records["raw_obs"].append(native._get_obs().copy())
        records["integration_state"].append(_integration_state(env))
        records["time"].append(float(native.data.time))

    try:
        check_cap(deadline)
        restore_native(env, root_record)
        capture()
        for t, (command, disturbance) in enumerate(zip(commands, noise, strict=True)):
            check_cap(deadline)
            applied = np.clip(command.astype(np.float64) + disturbance, -1, 1)
            _, reward, terminated, truncated, info = env.step(applied)
            require(not terminated and bool(truncated) == (root + t + 1 == 50), "Unexpected native boundary")
            records["applied_actions"].append(applied)
            records["rewards"].append(float(reward))
            records["reward_dist"].append(float(info["reward_dist"]))
            records["reward_ctrl"].append(float(info["reward_ctrl"]))
            records["terminated"].append(bool(terminated))
            records["truncated"].append(bool(truncated))
            capture()
        return {**{key: np.asarray(values) for key, values in records.items()},
                "commands": commands.copy(), "actuator_noise": noise.copy()}
    finally:
        if own:
            env.close()


def diagnostic_root(plan, models, episode, index, ordinal, inputs_stem, out, deadline, progress, template):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    t = protocol.root_steps(plan, index)[ordinal]
    horizon = min(plan["planning_horizon"], plan["steps"] - t)
    inputs = protocol.load_inputs(inputs_stem)
    bank = common_bank(inputs, t, plan)[0]
    sequences = [sequence.copy() for sequence in bank]
    ids = [f"common/{i}" for i in range(64)]
    history_members, order, search_seconds, belief_seconds = {}, [], 0., 0.
    for panel in plan["panels"]:
        history = protocol.public_history(episode, protocol.schedule(plan, index, panel, "diagnostic"), t)
        history_file = out / f"history-{panel}.npz"
        protocol.save_npz(history_file, **history)
        history_members[history_file.name] = sha(history_file)
        for name in plan["fit_order"]:
            tick = time.perf_counter()
            state = belief_at(models[name], **history, deadline=deadline)
            belief_seconds += time.perf_counter() - tick
            for method in plan["planners"]:
                check_cap(deadline)
                progress.update(panel=panel, fit=name, planner=method)
                result, raw, sizes, seconds = score_search(models[name], state, inputs, method, t, plan, deadline)
                protocol.save_trace(out / "search" / panel / name / method, result, raw, sizes, seconds)
                search_seconds += seconds
                sequences.append(result.selected_sequences[0].copy())
                ids.append(f"selected/{panel}/{name}/{method}")
                order.append({"panel": panel, "fit": name, "planner": method, "union_index": len(sequences) - 1})
    commands = np.stack(sequences)
    noise = np.stack([np.random.default_rng(protocol.seed(plan, f"diagnostic/branch_noise/{index}/{ordinal}/{b}"))
                      .normal(0., plan["noise_std"], (horizon, 2)) for b in range(plan["diagnostic_branches"])])
    root_meta = {
        "episode_index": index, "root_ordinal": ordinal, "step": t, "phase": protocol.phase(plan, index, "diagnostic"),
        "horizon": horizon, "union_ids": ids, "search_order": order, "history_members": history_members,
        "identity_slots": len(commands), "unique_sequence_count": len({sequence.tobytes() for sequence in commands}),
        "duplicates_retained": True,
    }
    write(out / "root.json", root_meta)
    protocol.save_npz(out / "union.npz", commands=commands, noise=noise)
    root_record = {"step": t, "integration_state": episode["audit"]["integration_state"][t]}
    completed, indices = [], []
    tick = time.perf_counter()
    try:
        for candidate, sequence in enumerate(commands):
            for branch in range(plan["diagnostic_branches"]):
                progress.update(native_candidate=candidate, native_branch=branch)
                result = native_branch(root_record, sequence, noise[branch], deadline, env=template)
                completed.append(result)
                indices.append((candidate, branch))
        arrays = {key: np.stack([row[key] for row in completed]).reshape(
            len(commands), plan["diagnostic_branches"], *completed[0][key].shape) for key in completed[0]}
        protocol.save_npz(out / "native.npz", **arrays)
    except BaseException:
        if completed:
            protocol.save_npz(out / "native-partial.npz", indices=np.asarray(indices, dtype=np.int64),
                              **{key: np.stack([row[key] for row in completed]) for key in completed[0]})
        raise
    result = {"belief_seconds": belief_seconds, "search_seconds": search_seconds,
              "native_and_storage_seconds": time.perf_counter() - tick,
              "row_wall_seconds": time.perf_counter() - begin, "identity_slots": len(commands),
              "native_transitions": len(commands) * plan["diagnostic_branches"] * horizon,
              "observation_assimilations": len(plan["panels"]) * len(models) * (t + 1),
              "history_action_advances": len(plan["panels"]) * len(models) * t}
    write(out / "timings.json", result)
    check_cap(deadline)
    return result


def copy_verified(source, destination, digest, deadline):
    check_cap(deadline)
    checked(source, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "Inherited destination exists")
    shutil.copyfile(source, destination)
    checked(destination, digest)
    check_cap(deadline)


def run(path, expected, out):
    start, started_unix = time.monotonic(), time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "validate"}
    try:
        plan = validate(path, expected)
        deadline = start + plan["cap_seconds"]
        check_cap(deadline)
        torch.set_num_threads(plan["threads"])
        torch.use_deterministic_algorithms(True)
        write(out / "started.json", {"plan_sha256": expected, "unix_time": started_unix})
        source = plan["parent_source"]
        progress["phase"] = "inherit-all-fits"
        for member, digest in source["members"].items():
            copy_verified(ROOT / source["execution_path"] / member, out / member, digest, deadline)
        for name, origin, digest in (
            ("source-plan.json", ROOT / source["plan_path"], source["plan_sha256"]),
            ("source-audit-receipt.json", ROOT / source["audit_path"], source["audit_receipt_sha256"]),
            ("source-completed.json", ROOT / source["execution_path"] / "completed.json", source["completed_sha256"]),
            ("source-summary.json", (ROOT / source["audit_path"]).parent / "summary.json", source["summary_sha256"]),
        ):
            copy_verified(origin, out / name, digest, deadline)
        write(out / "parent-provenance.json", source)
        write(out / "random-streams.json", plan["random_stream_contract"])
        models = {}
        for name in plan["fit_order"]:
            check_cap(deadline)
            progress["fit"] = name
            kind, fit_seed = name.rsplit("-", 1)
            model = legacy.model_for(plan, kind, int(fit_seed))
            model.load_state_dict(torch.load(out / "fits" / name / "weights.pt", weights_only=True,
                                            map_location="cpu"), strict=True)
            model.requires_grad_(False)
            models[name] = model.eval()
        write(out / "inherited-fits-ready.json", {
            "plan_sha256": expected, "fit_order": list(models), "new_fits": 0,
            "files": {f"fits/{name}/weights.pt": sha(out / "fits" / name / "weights.pt") for name in models},
            "elapsed_seconds": time.monotonic() - start, "unix_time": time.time(),
        })
        evaluation_start = time.monotonic() - start
        write(out / "evaluation-started.json", {
            "plan_sha256": expected, "elapsed_seconds": evaluation_start, "unix_time": time.time(),
            "inherited_fits_ready_sha256": sha(out / "inherited-fits-ready.json"),
            "random_streams_sha256": sha(out / "random-streams.json"),
        })
        progress = {"phase": "fresh-innovations"}
        tick = time.perf_counter()
        inputs_by_step = []
        for t in range(plan["steps"]):
            check_cap(deadline)
            prefix = f"planner/control/{t}"
            stem = out / "innovations" / "control" / f"{t:03d}"
            protocol.save_inputs(stem, protocol.draw_inputs(plan, prefix, plan["control_episodes"]), prefix)
            inputs_by_step.append(stem)
        for index in range(plan["diagnostic_episodes"]):
            for ordinal in range(4):
                check_cap(deadline)
                prefix = f"planner/diagnostic/{index}/{ordinal}"
                protocol.save_inputs(out / "innovations" / "diagnostic" / f"{index:03d}-{ordinal:02d}",
                                     protocol.draw_inputs(plan, prefix, 1), prefix)
        innovation_seconds = time.perf_counter() - tick
        control_timings = []
        for row in plan["execution_order"]:
            progress = {"phase": "control", **row}
            folder = out / "control" / row["panel"] / row["label"]
            if "reference" in row:
                timing = reference_control(plan, row["panel"], row["reference"], inputs_by_step,
                                           folder, deadline, progress)
            else:
                timing = learned_control(plan, models[row["fit"]], row["panel"], row["planner"],
                                         inputs_by_step, folder, deadline, progress)
            control_timings.append(timing)
            print(json.dumps({"completed_control": f"{row['panel']}/{row['label']}"}), flush=True)
        write(out / "control-completed.json", {"plan_sha256": expected, "rows": len(control_timings),
                                                "elapsed_seconds": time.monotonic() - start})
        progress = {"phase": "diagnostic-collection"}
        tick = time.perf_counter()
        cohort = []
        (out / "diagnostic").mkdir()
        try:
            for index in range(plan["diagnostic_episodes"]):
                progress["episode"] = index
                check_cap(deadline)
                cohort.append(collect_episode(protocol.seed(plan, f"diagnostic/reset/{index}"),
                    protocol.schedule(plan, index, "full", "diagnostic"), noise_std=plan["noise_std"],
                    noise_seed=protocol.seed(plan, f"diagnostic/actuator_noise/{index}"),
                    action_seed=protocol.seed(plan, f"diagnostic/exploration/{index}"), policy="mixed"))
        except BaseException:
            if cohort:
                base.save_records(out / "diagnostic" / "partial-cohort", cohort)
            raise
        base.save_records(out / "diagnostic" / "cohort", cohort)
        collection_seconds = time.perf_counter() - tick
        diagnostic_timings = []
        template = make_env()
        try:
            template.reset(seed=protocol.seed(plan, "diagnostic/native_template/0"))
            for index, episode in enumerate(cohort):
                for ordinal in range(4):
                    progress = {"phase": "diagnostic-root", "episode": index, "root": ordinal}
                    label = f"{index:03d}-{ordinal:02d}"
                    diagnostic_timings.append(diagnostic_root(plan, models, episode, index, ordinal,
                        out / "innovations" / "diagnostic" / label,
                        out / "diagnostic" / "roots" / label, deadline, progress, template))
                    print(json.dumps({"completed_diagnostic_root": label}), flush=True)
        finally:
            template.close()
        write(out / "diagnostic-completed.json", {"plan_sha256": expected, "roots": len(diagnostic_timings),
            "native_transitions": sum(row["native_transitions"] for row in diagnostic_timings),
            "elapsed_seconds": time.monotonic() - start})
        write(out / "costs.json", {
            "inherited_setup_seconds": evaluation_start, "innovation_generation_and_storage_seconds": innovation_seconds,
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in control_timings),
            "control_setup_seconds": sum(row["setup_seconds"] for row in control_timings),
            "control_decision_seconds": sum(sum(row["decision_seconds"]) for row in control_timings),
            "control_native_step_seconds": sum(sum(row["native_step_seconds"]) for row in control_timings),
            "diagnostic_collection_seconds": collection_seconds,
            "diagnostic_root_wall_seconds": sum(row["row_wall_seconds"] for row in diagnostic_timings),
            "prior_costs": source["prior_costs"], "new_fits": 0,
            "accounting": "Decision timing includes loading, proposals, scoring, state updates and search trace writes. Row totals include setup/native/storage. Batch timing is throughput, not single-agent latency. Prior cumulative attempt cost already includes original fitting. Final validation/hashing remain inside total execution wall time.",
        })
        progress = {"phase": "final-validation-and-hashing"}
        validate(path, expected)
        members = {}
        for file in sorted(out.rglob("*")):
            check_cap(deadline)
            require(not file.is_symlink(), "Execution symlink")
            if file.is_file():
                members[str(file.relative_to(out))] = sha(file)
        check_cap(deadline)
        wall = time.monotonic() - start
        write(out / "completed.json", {
            "status": "completed", "version": STUDY, "plan_sha256": expected, "files": members,
            "fits": len(models), "new_fits": 0, "astra_calls": 0, "wall_seconds": wall,
            "evaluation_started_elapsed_seconds": evaluation_start, "control_rows": len(control_timings),
            "diagnostic_roots": len(diagnostic_timings),
            "inherited_fit_wall_seconds": source["prior_costs"]["inherited_fit_wall_seconds"],
            "prior_cumulative_attempt_wall_seconds": source["prior_costs"]["cumulative_attempt_wall_seconds"],
            "cumulative_attempt_wall_seconds": source["prior_costs"]["cumulative_attempt_wall_seconds"] + wall,
        })
        check_cap(deadline)
    except BaseException as error:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "over-cap-completion.json")
        write(out / "failed.json", {"status": "failed", "error": repr(error), "plan_sha256": expected,
              "wall_seconds": time.monotonic() - start, "progress": progress, "new_fits": 0})
        raise


def audit(path, expected, execution, out):
    from audit_reacher_search_study import audit_saved
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
        print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
