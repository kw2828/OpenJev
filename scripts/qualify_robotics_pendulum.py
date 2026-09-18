"""Frozen observation-only qualification, not a learned architecture benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium
import numpy as np

from openjev.research.robotics_pendulum import (
    Config,
    history_estimate,
    observation,
    transition,
)

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "current_nominal", "velocity_nominal", "history_identified",
    "recent3_identified", "known_state_physics", "uniform",
)
SOURCES = (
    "scripts/qualify_robotics_pendulum.py",
    "src/openjev/research/robotics_pendulum.py",
    "tests/test_robotics_pendulum.py",
    "tests/test_qualify_robotics_pendulum.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime():
    from gymnasium.envs.classic_control import pendulum

    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "gymnasium": gymnasium.__version__, "platform": platform.platform(),
        "native_pendulum_sha256": sha(pendulum.__file__),
    }


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        "version": 1,
        "sources": {p: sha(ROOT / p) for p in SOURCES},
        "runtime": runtime(), "config": asdict(Config()),
        "cases": 32, "steps": 200, "horizon": 32, "candidates": 64,
        "action_block": 4, "case_seed": 63100001,
        "candidate_seed": 63110001, "diagnostic_seed": 63120001,
        "bootstrap_seed": 63130001, "bootstrap_samples": 4096,
        "gains": [0.65, 0.8, 1., 1.2, 1.35],
        "panels": ["stationary", "switch"], "switch_step": 100,
        "arms": list(ARMS), "probe_commands": [0.8, -0.8],
        "cap_seconds": 600,
        "policy_information": "Float32 cos/sin history and executed command history only. No rewards, hidden velocities, applied torque, gain or episode seed enter the four observation-based estimators. All use the supplied native physical equations and dt; this is a privileged model-class comparison, not model learning. The known_state_physics arm additionally receives true current state and gain. Uniform is a floor, not a compute-matched planner.",
        "design": "All arms receive the same two initial probe commands, included in total cost. Thereafter all MPC arms receive exactly the same random action candidates at each time and case. Actions are held in four-step blocks inside a 32-step undiscounted horizon, with zero/+2/-2 constant candidates included. Execute only the first action and replan. Horizon truncates at the episode end. No terminal value, warm-start selection or tuning. Full-history and last-three-observation estimators use nominal gain1 when no transition identifies gain. No fit or hidden-state supervision.",
        "panels_definition": "Same32 initial states in both panels. Initial gains cycle through the five declared values. Stationary gains never change. Switch gains change at step100 from g to 2-g; the change is not announced to any observation-based controller. These are fixed development diagnostics, not standard leaderboard tasks.",
        "diagnostics": "256 paired initial states with48 common commands in[-1,1], compare gain.65 against1.35 and zero-command controls. Measure trajectory distinction, velocity/gain recovery and informative counts. Physics-based recoverability is a sufficiency diagnostic, not evidence of neural learning.",
        "endpoints": "Native positive episode cost=-sum(reward), all cases retained. Also report post-probe costs and each gain cell descriptively. Report paired cost differences against known_state_physics with4096 episode bootstrap resamples and95% intervals. No optimality, architecture uncertainty or significance claim. Last3 sufficiency requires mean stationary cost<=1.05 times known_state_physics and <=.90 times current_nominal. Known-state reference competence requires stationary mean cost<=.80 times uniform. Numerical observability requires max final velocity error<=1e-4 and max full-history gain error<=1e-3 among informative trajectories, with>=90% informative; zero commands must not distinguish gains. All are required to call this short-history qualification successful. Otherwise preserve diagnostic failure, no automatic tuning. Either outcome is insufficient for a topology claim.",
        "stop": "If qualified, do not launch a long-memory/connectome sweep on this clean constant-gain task. A harder task requires independent motivation and a new prospective protocol. No retries, replacement cases, extensions or checkpoint selection. Failure retains receipts.",
        "audit": "Validate all source/runtime bindings, hashes and complete artifact membership. Replay every saved transition through installed Gymnasium with logged applied torque, recompute metrics and paired intervals from saved arrays only. No new policy or planner evaluations during audit.",
    }
    write(out / "plan.json", spec)
    return {"plan": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(plan):
    if set(plan["sources"]) != set(SOURCES):
        raise ValueError("Frozen source membership changed")
    for path, digest in plan["sources"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    if runtime() != plan["runtime"]:
        raise ValueError("Frozen runtime changed")


def candidates(plan, step, batch):
    rng = np.random.default_rng(plan["candidate_seed"] + step)
    horizon = min(plan["horizon"], plan["steps"] - step)
    chunks = (horizon + plan["action_block"] - 1) // plan["action_block"]
    result = np.repeat(
        rng.uniform(-2., 2., (batch, plan["candidates"], chunks)),
        plan["action_block"], axis=-1,
    )[..., :horizon]
    result[:, 0] = 0.
    result[:, 1] = 2.
    result[:, 2] = -2.
    return result


def mpc(states, gains, action_candidates, config):
    batch, count, horizon = action_candidates.shape
    rollout = np.repeat(states, count, axis=0)
    gain = np.repeat(gains, count)
    costs = np.zeros(batch * count)
    for t in range(horizon):
        rollout, reward = transition(
            rollout, action_candidates[:, :, t].reshape(-1), gain, config,
        )
        costs -= reward
    best = costs.reshape(batch, count).argmin(1)
    return action_candidates[np.arange(batch), best, 0]


def estimate(arm, observations, commands, config):
    count = len(observations)
    theta = np.arctan2(observations[:, -1, 1], observations[:, -1, 0])
    if arm == "current_nominal":
        omega, gain, informative = np.zeros(count), np.ones(count), np.zeros(count, int)
    else:
        history = observations[:, -3:] if arm == "recent3_identified" else observations
        previous = commands[:, -(history.shape[1] - 1):] if history.shape[1] > 1 else commands[:, :0]
        omega, gain, informative = history_estimate(history, previous, config)
        if arm == "velocity_nominal":
            gain = np.ones(count)
    return np.stack((theta, omega), axis=-1), gain, informative


def rollout(plan, panel, arm, *, deadline):
    config = Config(**plan["config"])
    count, steps = plan["cases"], plan["steps"]
    rng = np.random.default_rng(plan["case_seed"])
    states = np.empty((count, steps + 1, 2))
    states[:, 0, 0] = rng.uniform(-np.pi, np.pi, count)
    states[:, 0, 1] = rng.uniform(-1., 1., count)
    observations = np.empty((count, steps + 1, 2), np.float32)
    observations[:, 0] = observation(states[:, 0])
    commands = np.empty((count, steps))
    rewards = np.empty_like(commands)
    true_gains = np.empty_like(commands)
    estimated = np.full((count, steps, 3), np.nan)
    informative = np.zeros((count, steps), dtype=np.int64)
    initial_gains = np.resize(np.array(plan["gains"]), count)
    uniform_rng = np.random.default_rng(plan["candidate_seed"] + 100000)
    begin = time.monotonic()
    planner_calls = 0
    for t in range(steps):
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed qualification runtime cap")
        gain = 2. - initial_gains if panel == "switch" and t >= plan["switch_step"] else initial_gains
        true_gains[:, t] = gain
        if t < len(plan["probe_commands"]):
            action = np.full(count, plan["probe_commands"][t])
        elif arm == "uniform":
            action = uniform_rng.uniform(-2., 2., count)
        else:
            if arm == "known_state_physics":
                guessed_state, guessed_gain = states[:, t].copy(), gain.copy()
                info = np.zeros(count, dtype=int)
            else:
                guessed_state, guessed_gain, info = estimate(
                    arm, observations[:, :t + 1], commands[:, :t], config,
                )
            estimated[:, t, :2] = guessed_state
            estimated[:, t, 2] = guessed_gain
            informative[:, t] = info
            action = mpc(guessed_state, guessed_gain, candidates(plan, t, count), config)
            planner_calls += count
        commands[:, t] = action
        states[:, t + 1], rewards[:, t] = transition(states[:, t], action, gain, config)
        observations[:, t + 1] = observation(states[:, t + 1])
    return {
        "state": states, "observation": observations, "command": commands,
        "reward": rewards, "gain": true_gains, "estimated": estimated,
        "informative": informative,
    }, {"wall_seconds": time.monotonic() - begin, "planner_calls": planner_calls}


def observability(plan):
    cfg = Config(**plan["config"])
    rng = np.random.default_rng(plan["diagnostic_seed"])
    batch, steps = 256, 48
    initial = np.stack((rng.uniform(-np.pi, np.pi, batch), rng.uniform(-1, 1, batch)), -1)
    commands = rng.uniform(-1, 1, (batch, steps))
    result = {"commands": commands, "initial": initial}
    for label, command in (("excited", commands), ("zero", commands * 0)):
        for gain in (0.65, 1.35):
            states = np.empty((batch, steps + 1, 2))
            states[:, 0] = initial
            for t in range(steps):
                states[:, t + 1], _ = transition(states[:, t], command[:, t], np.full(batch, gain), cfg)
            obs = observation(states.reshape(-1, 2)).reshape(batch, steps + 1, 2).astype(np.float32)
            omega, estimate_gain, counts = history_estimate(obs, command, cfg)
            stem = f"{label}_{gain}"
            result.update({
                stem + "_state": states, stem + "_observation": obs,
                stem + "_omega": omega, stem + "_gain": estimate_gain,
                stem + "_count": counts,
            })
    return result


def run(plan_path, out, *, expected_plan_sha256):
    if sha(plan_path) != expected_plan_sha256:
        raise ValueError("Plan digest differs from external frozen identity")
    plan = json.loads(plan_path.read_text())
    validate(plan)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    plan_digest = sha(plan_path)
    write(out / "started.json", {"plan_sha256": plan_digest, "unix_time": time.time()})
    try:
        np.savez_compressed(out / "observability.npz", **observability(plan))
        if time.monotonic() - start >= plan["cap_seconds"]:
            raise TimeoutError("Fixed qualification runtime cap after diagnostics")
        timing = {}
        for panel in plan["panels"]:
            for arm in plan["arms"]:
                arrays, costs = rollout(plan, panel, arm, deadline=start + plan["cap_seconds"])
                name = f"{panel}-{arm}"
                np.savez_compressed(out / f"{name}.npz", **arrays)
                timing[name] = costs
                print(json.dumps({"finished": name, **costs}), flush=True)
        write(out / "timing.json", timing)
        validate(plan)
        if sha(plan_path) != plan_digest:
            raise ValueError("Frozen plan changed during execution")
        elapsed = time.monotonic() - start
        if elapsed >= plan["cap_seconds"]:
            raise TimeoutError("Fixed qualification runtime cap before completion")
        write(out / "completed.json", {
            "status": "completed", "plan_sha256": plan_digest,
            "wall_seconds": elapsed, "fits": 0,
            "neural_calls": 0, "astra_calls": 0,
            "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        })
    except BaseException as error:
        write(out / "failed.json", {"error": repr(error), "wall_seconds": time.monotonic() - start})
        raise


def paired_interval(values, plan):
    rng = np.random.default_rng(plan["bootstrap_seed"])
    indices = rng.integers(0, len(values), (plan["bootstrap_samples"], len(values)))
    return np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()


def native_replay(states, commands, gains, config):
    """Independent installed-simulator replay of recorded physical transitions."""
    from gymnasium.envs.classic_control.pendulum import PendulumEnv

    if commands.ndim != 2 or min(commands.shape) < 1:
        raise ValueError("Invalid command dimensions")
    n, steps = commands.shape
    if states.shape != (n, steps + 1, 2) or gains.shape != commands.shape:
        raise ValueError("Invalid replay array shapes")
    if not all(np.isfinite(x).all() for x in (states, commands, gains)):
        raise ValueError("Nonfinite replay arrays")
    native = PendulumEnv()
    for key, value in config.items():
        setattr(native, {"mass": "m", "length": "l"}.get(key, key), value)
    rewards = np.empty_like(commands)
    max_error = 0.
    for i in range(n):
        for t in range(steps):
            native.state = states[i, t].copy()
            applied = np.clip(np.clip(commands[i, t], -native.max_torque, native.max_torque) * gains[i, t], -native.max_torque, native.max_torque)
            _, reward, _, _, _ = native.step(np.array([applied]))
            error = float(np.max(np.abs(native.state - states[i, t + 1])))
            max_error = max(max_error, error)
            if error > 1e-10:
                raise ValueError("Independent Gymnasium transition replay mismatch")
            rewards[i, t] = reward
    native.close()
    return rewards, max_error, n * steps


def independent_estimator_replay(obs, commands, cfg):
    """Scalar per-case least-squares derivation; no policy/planner evaluation."""
    omega, gains, counts = [], [], []
    eps = np.finfo(obs.dtype).eps
    speed_tol = 16 * eps / cfg["dt"] + 1e-10 * cfg["max_speed"]
    gravity = 3 * cfg["g"] / (2 * cfg["length"])
    coefficient = 3 / (cfg["mass"] * cfg["length"] ** 2)
    torque_tol = (2 * speed_tol / cfg["dt"] + abs(gravity) * 16 * eps) / coefficient + 1e-10 * cfg["max_torque"]
    for angles, issued in zip(obs, commands, strict=True):
        theta = np.arctan2(angles[:, 1].astype(float), angles[:, 0].astype(float))
        velocity = (np.diff(theta) + np.pi) % (2 * np.pi) - np.pi
        velocity = np.clip(velocity / cfg["dt"], -cfg["max_speed"], cfg["max_speed"])
        x, y = [], []
        for t in range(1, len(issued)):
            command = np.clip(issued[t], -cfg["max_torque"], cfg["max_torque"])
            torque = ((velocity[t] - velocity[t - 1]) / cfg["dt"] - gravity * np.sin(theta[t])) / coefficient
            if (abs(command) > 1e-8 * cfg["max_torque"]
                    and abs(velocity[t]) < cfg["max_speed"] - speed_tol
                    and abs(torque) < cfg["max_torque"] - torque_tol):
                x.append(command)
                y.append(torque)
        gains.append(float(np.linalg.lstsq(np.array(x)[:, None], np.array(y), rcond=None)[0][0]) if x else 1.)
        omega.append(velocity[-1])
        counts.append(len(x))
    return np.array(omega), np.array(gains), np.array(counts)


def audit(plan_path, execution, out, *, expected_plan_sha256):
    begin = time.monotonic()
    if sha(plan_path) != expected_plan_sha256:
        raise ValueError("Plan digest differs from external frozen identity")
    plan = json.loads(plan_path.read_text())
    validate(plan)
    completed = json.loads((execution / "completed.json").read_text())
    if completed["status"] != "completed" or completed["plan_sha256"] != sha(plan_path):
        raise ValueError("Unbound execution")
    if not 0 <= completed["wall_seconds"] <= plan["cap_seconds"]:
        raise ValueError("Execution exceeds fixed runtime cap")
    expected = {"started.json", "observability.npz", "timing.json"}
    expected |= {f"{p}-{a}.npz" for p in plan["panels"] for a in plan["arms"]}
    if set(completed["files"]) != expected:
        raise ValueError("Missing or extra execution artifacts")
    if {p.name for p in execution.iterdir()} != expected | {"completed.json"}:
        raise ValueError("Unexpected execution directory membership")
    for name, digest in completed["files"].items():
        if sha(execution / name) != digest:
            raise ValueError(f"Changed execution artifact {name}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        summary, costs, post_probe = {}, {}, {}
        timing = json.loads((execution / "timing.json").read_text())
        if set(timing) != {f"{p}-{a}" for p in plan["panels"] for a in plan["arms"]}:
            raise ValueError("Bad timing membership")
        total_timed = 0.
        max_state_error = max_reward_error = 0.
        transitions = 0
        for panel in plan["panels"]:
            costs[panel] = {}
            post_probe[panel] = {}
            for arm in plan["arms"]:
                record = timing[f"{panel}-{arm}"]
                expected_calls = 0 if arm == "uniform" else plan["cases"] * (plan["steps"] - len(plan["probe_commands"]))
                if record["planner_calls"] != expected_calls or not np.isfinite(record["wall_seconds"]) or record["wall_seconds"] < 0:
                    raise ValueError("Bad planner or timing counters")
                total_timed += record["wall_seconds"]
                with np.load(execution / f"{panel}-{arm}.npz", allow_pickle=False) as archive:
                    a = {k: archive[k] for k in archive.files}
                    n, steps = plan["cases"], plan["steps"]
                    if a["state"].shape != (n, steps + 1, 2) or a["reward"].shape != (n, steps):
                        raise ValueError("Bad trajectory dimensions")
                    if a["estimated"].shape != (n, steps, 3) or a["informative"].shape != (n, steps):
                        raise ValueError("Bad estimator recording dimensions")
                    unused = np.zeros((n, steps), bool)
                    unused[:, :len(plan["probe_commands"])] = True
                    if arm == "uniform":
                        unused[:] = True
                    if not np.isnan(a["estimated"][unused]).all() or not np.isfinite(a["estimated"][~unused]).all():
                        raise ValueError("Invalid estimator values or missing-slot mask")
                    if a["informative"].dtype.kind not in "iu" or np.any(a["informative"] < 0):
                        raise ValueError("Invalid informative counts")
                    if np.any(np.abs(a["command"]) > plan["config"]["max_torque"]):
                        raise ValueError("Out-of-range control command")
                    rng = np.random.default_rng(plan["case_seed"])
                    initial = np.stack((rng.uniform(-np.pi, np.pi, n), rng.uniform(-1, 1, n)), -1)
                    gains = np.repeat(np.resize(np.array(plan["gains"]), n)[:, None], steps, axis=1)
                    if panel == "switch":
                        gains[:, plan["switch_step"]:] = 2. - gains[:, plan["switch_step"]:]
                    if not np.array_equal(a["state"][:, 0], initial) or not np.array_equal(a["gain"], gains):
                        raise ValueError("Case or gain schedule differs from protocol")
                    if not np.all(a["command"][:, :2] == np.array(plan["probe_commands"])):
                        raise ValueError("Probe actions differ from protocol")
                    replayed_rewards, error, replayed = native_replay(a["state"], a["command"], a["gain"], plan["config"])
                    reward_error = float(np.max(np.abs(replayed_rewards - a["reward"])))
                    if reward_error > 1e-10:
                        raise ValueError("Independent reward replay mismatch")
                    max_state_error = max(max_state_error, error)
                    max_reward_error = max(max_reward_error, reward_error)
                    transitions += replayed
                    if not np.array_equal(observation(a["state"].reshape(-1, 2)).astype(np.float32).reshape(n, steps + 1, 2), a["observation"]):
                        raise ValueError("Observation recording mismatch")
                    if not np.all(np.isfinite(a["reward"])):
                        raise ValueError("Nonfinite costs")
                    for t in range(len(plan["probe_commands"]), steps):
                        if arm == "uniform":
                            if np.any(a["informative"][:, t] != 0):
                                raise ValueError("Unexpected uniform-controller estimates")
                            continue
                        if arm == "known_state_physics":
                            state, gain, count = a["state"][:, t], a["gain"][:, t], np.zeros(n, int)
                        else:
                            state, gain, count = estimate(arm, a["observation"][:, :t + 1], a["command"][:, :t], Config(**plan["config"]))
                        expected_estimate = np.column_stack((state, gain))
                        if not np.array_equal(expected_estimate, a["estimated"][:, t]) or not np.array_equal(count, a["informative"][:, t]):
                            raise ValueError("Saved-history estimator arithmetic replay mismatch")
                    costs[panel][arm] = -a["reward"].sum(1)
                    post_probe[panel][arm] = -a["reward"][:, len(plan["probe_commands"]):].sum(1)
            reference = costs[panel]["known_state_physics"]
            summary[panel] = {}
            for arm, values in costs[panel].items():
                difference = values - reference
                summary[panel][arm] = {
                    "mean_cost": float(values.mean()),
                    "mean_post_probe_cost": float(post_probe[panel][arm].mean()),
                    "costs": values.tolist(),
                    "paired_mean_difference": float(difference.mean()),
                    "paired_difference_interval": paired_interval(difference, plan),
                    "initial_gain_cells": {
                        str(g): {
                            "count": int((np.resize(np.array(plan["gains"]), len(values)) == g).sum()),
                            "mean_cost": float(values[np.resize(np.array(plan["gains"]), len(values)) == g].mean()),
                            "paired_mean_difference": float(difference[np.resize(np.array(plan["gains"]), len(values)) == g].mean()),
                        }
                        for g in plan["gains"]
                        if np.any(np.resize(np.array(plan["gains"]), len(values)) == g)
                    },
                }
        if total_timed > completed["wall_seconds"] + 1e-6:
            raise ValueError("Component times exceed execution time")
        diagnostic = {}
        with np.load(execution / "observability.npz", allow_pickle=False) as archive:
            d = {k: archive[k] for k in archive.files}
            rng = np.random.default_rng(plan["diagnostic_seed"])
            expected_initial = np.stack((rng.uniform(-np.pi, np.pi, 256), rng.uniform(-1, 1, 256)), -1)
            expected_commands = rng.uniform(-1, 1, (256, 48))
            if not np.array_equal(d["initial"], expected_initial) or not np.array_equal(d["commands"], expected_commands):
                raise ValueError("Diagnostic input cohort changed")
            for label in ("excited", "zero"):
                for gain in (0.65, 1.35):
                    stem = f"{label}_{gain}"
                    commands = d["commands"] if label == "excited" else np.zeros_like(d["commands"])
                    if not np.array_equal(d[stem + "_state"][:, 0], expected_initial):
                        raise ValueError("Diagnostic initial state changed")
                    recorded_obs = d[stem + "_observation"]
                    expected_obs = observation(d[stem + "_state"].reshape(-1, 2)).reshape(256, 49, 2).astype(np.float32)
                    if not np.array_equal(recorded_obs, expected_obs):
                        raise ValueError("Diagnostic observations changed")
                    _, error, replayed = native_replay(d[stem + "_state"], commands, np.full(commands.shape, gain), plan["config"])
                    max_state_error = max(max_state_error, error)
                    transitions += replayed
                    recomputed = independent_estimator_replay(recorded_obs, commands, plan["config"])
                    for suffix, value in zip(("omega", "gain", "count"), recomputed, strict=True):
                        saved = d[stem + "_" + suffix]
                        if saved.shape != value.shape or not np.isfinite(saved).all():
                            raise ValueError("Bad diagnostic estimates")
                        if not np.allclose(saved, value, atol=1e-9, rtol=1e-9):
                            raise ValueError("Independent estimator replay mismatch")
                    mask = d[stem + "_count"] > 0
                    diagnostic[stem] = {
                        "informative_fraction": float(mask.mean()),
                        "max_velocity_error": float(np.abs(d[stem + "_omega"] - d[stem + "_state"][:, -1, 1]).max()),
                        "max_informative_gain_error": float(np.abs(d[stem + "_gain"][mask] - gain).max()) if mask.any() else None,
                    }
                diff = d[f"{label}_0.65_observation"] - d[f"{label}_1.35_observation"]
                diagnostic[label + "_trajectory_difference_rms"] = float(np.sqrt(np.square(diff).mean()))
        stationary = summary["stationary"]
        checks = {
            "short_history_within_5pct_reference": stationary["recent3_identified"]["mean_cost"] <= 1.05 * stationary["known_state_physics"]["mean_cost"],
            "short_history_beats_current_10pct": stationary["recent3_identified"]["mean_cost"] <= .9 * stationary["current_nominal"]["mean_cost"],
            "known_state_beats_uniform_20pct": stationary["known_state_physics"]["mean_cost"] <= .8 * stationary["uniform"]["mean_cost"],
            "observable_velocity": all(diagnostic[f"excited_{g}"]["max_velocity_error"] <= 1e-4 for g in (.65, 1.35)),
            "observable_gain": all(diagnostic[f"excited_{g}"]["informative_fraction"] >= .9 and diagnostic[f"excited_{g}"]["max_informative_gain_error"] <= 1e-3 for g in (.65, 1.35)),
            "zero_action_unidentifiable": diagnostic["zero_trajectory_difference_rms"] == 0. and all(diagnostic[f"zero_{g}"]["informative_fraction"] == 0. for g in (.65, 1.35)),
        }
        result = {
            "scope": "Supplied-physics diagnostic, no trained models or biological advantage claim",
            "panels": summary, "observability": diagnostic, "checks": checks,
            "short_history_qualified": all(checks.values()),
            "independent_replayed_transitions": transitions,
            "max_state_error": max_state_error, "max_reward_error": max_reward_error,
        }
        write(out / "summary.json", result)
        lines = [
            "# Pendulum observation and control qualification", "",
            "This is a supplied-physics diagnostic. No neural models were trained.", "",
            f"Short-history qualification: **{'PASS' if all(checks.values()) else 'FAIL'}**.", "",
            "| Controller | Stationary mean cost | Switched-gain mean cost |",
            "|---|---:|---:|",
        ]
        for arm in plan["arms"]:
            lines.append(f"| {arm} | {summary['stationary'][arm]['mean_cost']:.3f} | {summary['switch'][arm]['mean_cost']:.3f} |")
        lines += ["", "Lower native episode cost is better. Every controller receives the same two initial probe actions, included in cost. MPC candidates and budgets are paired. The known-state reference uses finite random-shooting planning and is not an optimal controller.", "", f"Independent Gymnasium replay checked {transitions:,} saved transitions. Paired episode intervals and every case are in summary.json. They do not measure training-seed or architecture uncertainty.", "", "A pass rules out expanding this clean constant-gain task into a long-memory topology study. A failure retains the fixed diagnostic and requires diagnosis, not a claim that biological recurrence is needed."]
        (out / "README.md").write_text("\n".join(lines) + "\n")
        write(out / "receipt.json", {
            "status": "completed", "plan_sha256": sha(plan_path),
            "execution_receipt_sha256": sha(execution / "completed.json"),
            "saved_output_only": True, "new_policy_calls": 0,
            "new_mpc_decisions": 0,
            "estimator_arithmetic_replayed_from_saved_histories": True,
            "independent_transition_replay": "installed Gymnasium PendulumEnv",
            "wall_seconds": time.monotonic() - begin,
            "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        })
        return {"checks": checks, "summary": str(out / "summary.json")}
    except BaseException as error:
        write(out / "failed.json", {"error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.out)
    elif args.mode == "run":
        result = run(args.plan, args.out, expected_plan_sha256=args.expected_plan_sha256)
    else:
        result = audit(args.plan, args.execution, args.out, expected_plan_sha256=args.expected_plan_sha256)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
