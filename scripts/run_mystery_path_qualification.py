"""Frozen, bounded qualification of public route memory in the official environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from collections import deque
from dataclasses import astuple
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import memory_gym
import numpy as np
from memory_gym.mystery_path_grid import GridMysteryPathEnv

from openjev.research.mystery_path_memory import MemoryController, _retained_size
from openjev.research.mystery_path_observation import parse_observation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/mystery-path-qualification-v1"
MODES = ("full", "last16", "last32", "erase_on_failure", "reference")
DELTAS = ((0, -1), (-1, 0), (0, 1), (1, 0))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def record_failure(path, obj):
    """A secondary receipt failure must not replace the original exception."""
    try:
        save_json(path, obj)
    except BaseException as error:  # noqa: BLE001 - preserve an already active failure
        print(f"Could not save failure receipt {path}: {error!r}", file=sys.stderr)


def close_env(env, out):
    active_error = sys.exc_info()[0] is not None
    try:
        env.close()
    except BaseException:
        record_failure(out / "cleanup-error.json", {"error": traceback.format_exc()})
        if not active_error:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "completion-before-cleanup-error.json")
            raise


def native_truth(env):
    return (int(env.agent.grid_position[0]), int(env.agent.grid_position[1]),
            int(env.agent.rotation // 90), bool(env.is_off_path))


def reference_memory(safe, goal):
    seen = set()
    return {"serialized_state_bytes": len(json.dumps({"safe": sorted(safe), "goal": goal},
                                                     sort_keys=True, separators=(",", ":")).encode()),
            "retained_python_bytes": _retained_size(safe, seen) + _retained_size(goal, seen)}


def check_frame(env, frame):
    obs = parse_observation(frame)
    assert astuple(obs) == native_truth(env), (astuple(obs), native_truth(env))
    return obs


def reference_action(obs, safe, goal):
    """Evaluator-only shortest primitive-action route. Never passed to public policies."""
    if obs.off_path:
        return 0
    start = (obs.x, obs.y, obs.heading)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        (x, y, h), commands = queue.popleft()
        if (x, y) == goal:
            return commands[0] if commands else 0
        dx, dy = DELTAS[h]
        for action, nxt in ((3, (x + dx, y + dy, h)),
                            (1, (x, y, (h + 1) % 4)),
                            (2, (x, y, (h - 1) % 4))):
            if nxt[:2] not in safe or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, commands + (action,)))
    raise ValueError("Known route is disconnected")


def layout(env):
    safe = {(int(n.x), int(n.y)) for n in env.mystery_path.path}
    goal = tuple(int(x) for x in env.end)
    obj = {"safe_cells": sorted(safe), "start": [int(x) for x in env.start], "goal": goal}
    encoded = json.dumps(obj, sort_keys=True).encode()
    return safe, goal, hashlib.sha256(encoded).hexdigest()


def verify_bindings():
    bindings = json.loads((BASE / "bindings.json").read_text())
    for rel, sha in bindings["sources"].items():
        assert digest(ROOT / rel) == sha, rel
    installed = Path(memory_gym.__file__).parent
    for rel, sha in bindings["upstream_python"].items():
        assert digest(installed / rel) == sha, rel
    for package, version in bindings["packages"].items():
        assert importlib.metadata.version(package) == version, package
    assert sys.version == bindings["python"]
    return bindings


def preflight(out):
    """Four declared engineering seeds, parser and reference only, never qualification."""
    out.mkdir(parents=True, exist_ok=False)
    env = None
    records = []
    try:
        save_json(out / "started.json", {"seeds": [0, 1, 2, 3], "max_steps": 512,
                                        "purpose": "pixel adapter, forced-return and reference verification"})
        env = GridMysteryPathEnv()
        for seed in range(4):
            frame, _ = env.reset(seed=seed)
            obs = check_frame(env, frame)
            safe, goal, sha = layout(env)
            # Use privileged cells only for an adapter test that deliberately observes a fall.
            # Find an unsafe neighbor of a reachable safe cell other than the goal.
            candidates = [(x, y, h, (x + dx, y + dy)) for x, y in sorted(safe - {goal})
                          for h, (dx, dy) in enumerate(DELTAS)
                          if 0 <= x + dx < 7 and 0 <= y + dy < 7
                          and (x + dx, y + dy) not in safe]
            assert candidates
            x, y, heading, bad = candidates[0]
            phase = "approach"
            falls = 0
            steps = []
            for _ in range(128):
                if phase == "approach" and (obs.x, obs.y) == (x, y):
                    phase = "face"
                if phase == "face" and obs.heading == heading:
                    action = 3
                    phase = "fall"
                elif phase == "face":
                    action = 1
                elif phase == "approach":
                    action = reference_action(obs, safe, (x, y))
                elif phase == "fall":
                    assert obs.off_path and (obs.x, obs.y) == bad
                    action = 2  # Deliberately ignored by the official forced-return transition.
                    phase = "return"
                else:
                    if phase == "return":
                        assert not obs.off_path and (obs.x, obs.y) == tuple(env.start)
                        assert obs.heading == heading
                        phase = "finish"
                    action = reference_action(obs, safe, goal)
                frame, reward, done, truncated, info = env.step(action)
                obs = check_frame(env, frame)
                falls += int(obs.off_path)
                steps.append({"action": action, "observation": astuple(obs), "reward": reward})
                assert not truncated
                if done:
                    assert info["success"] == 1 and falls == 1 and phase == "finish"
                    break
            else:
                raise AssertionError("Preflight reference failed to finish")
            records.append({"seed": seed, "layout_sha256": sha, "steps": steps})
        save_json(out / "completed.json", {"status": "passed", "episodes": records,
                                           "total_steps": sum(len(r["steps"]) for r in records)})
    except BaseException:
        record_failure(out / "failed.json", {"error": traceback.format_exc(), "completed_episodes": records})
        raise
    finally:
        if env is not None:
            close_env(env, out)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = None
    total_steps = 0
    completed = 0
    try:
        save_json(out / "started.json", {"status": "started", "pid": os.getpid(),
                                         "platform": platform.platform(), "wall_cap_seconds": 900})
        verify_bindings()
        protocol = json.loads((BASE / "protocol.json").read_text())
        seeds = json.loads((BASE / "inputs.json").read_text())["seeds"]
        assert len(seeds) == len(set(seeds)) == 64 and not set(seeds).intersection(range(4))
        assert protocol["episodes"] == 1280
        (out / "episodes").mkdir()
        env = GridMysteryPathEnv()
        layout_hashes = {}
        with (out / "episodes.jsonl").open("x", buffering=1) as ledger:
            for index, seed in enumerate(seeds):
                for order in range(4):
                    priority = tuple((order + i) % 4 for i in range(4))
                    for mode in MODES:
                        episode_start = time.perf_counter()
                        frame, _ = env.reset(seed=seed)
                        reset_seconds = time.perf_counter() - episode_start
                        safe, goal, sha = layout(env)
                        if index in layout_hashes:
                            assert layout_hashes[index] == sha
                        else:
                            layout_hashes[index] = sha
                        tick = time.perf_counter()
                        obs = parse_observation(frame)
                        parsing_seconds = time.perf_counter() - tick
                        assert astuple(obs) == native_truth(env)
                        controller = None if mode == "reference" else MemoryController(mode, priority)
                        if controller is not None:
                            controller.reset(obs)
                        frames, observed = [frame], [astuple(obs)]
                        actions, rewards, terminals = [], [], []
                        memory_sizes, timings = [], []
                        env_seconds = policy_seconds = update_seconds = 0.0
                        falls = 0
                        try:
                            for step in range(128):
                                if time.perf_counter() - started > 900:
                                    raise TimeoutError("Frozen execution wall cap exceeded")
                                tick = time.perf_counter()
                                action = (reference_action(obs, safe, goal) if controller is None
                                          else controller.act(obs))
                                policy_time = time.perf_counter() - tick
                                assert type(action) is int and 0 <= action < 4
                                tick = time.perf_counter()
                                frame, reward, done, truncated, info = env.step(action)
                                env_time = time.perf_counter() - tick
                                total_steps += 1
                                # Preserve a returned native transition before downstream validation.
                                frames.append(frame)
                                actions.append(action)
                                rewards.append(reward)
                                terminals.append(done)
                                tick = time.perf_counter()
                                after = parse_observation(frame)
                                parse_time = time.perf_counter() - tick
                                assert astuple(after) == native_truth(env)
                                assert not truncated and reward in (0.0, 1.0)
                                tick = time.perf_counter()
                                if controller is not None:
                                    controller.observe(obs, action, after, reward)
                                update_time = time.perf_counter() - tick
                                if controller is not None:
                                    sizes = controller.memory_bytes()
                                else:
                                    sizes = reference_memory(safe, goal)
                                memory_sizes.append([sizes["serialized_state_bytes"], sizes["retained_python_bytes"]])
                                timings.append([policy_time, env_time, parse_time, update_time])
                                observed.append(astuple(after))
                                falls += int(after.off_path)
                                env_seconds += env_time
                                policy_seconds += policy_time
                                parsing_seconds += parse_time
                                update_seconds += update_time
                                obs = after
                                if done:
                                    assert info["success"] == int(reward == 1)
                                    assert info["num_fails"] == falls
                                    break
                            assert done and len(actions) <= 128
                        except BaseException:
                            try:
                                np.savez_compressed(out / "partial-episode.npz", frames=np.asarray(frames),
                                                    observations=observed, actions=actions, rewards=rewards,
                                                    terminals=terminals, timings=timings, memory_sizes=memory_sizes)
                            except BaseException as error:  # noqa: BLE001 - preserve original exception
                                print(f"Could not preserve partial episode: {error!r}", file=sys.stderr)
                            raise
                        tick = time.perf_counter()
                        relative = f"episodes/{index:03d}-{mode}-order{order}.npz"
                        np.savez_compressed(out / relative, frames=np.asarray(frames),
                                            observations=np.asarray(observed, dtype=np.int16),
                                            actions=np.asarray(actions, dtype=np.uint8),
                                            rewards=np.asarray(rewards, dtype=np.float64),
                                            terminals=np.asarray(terminals, dtype=np.bool_),
                                            timings=np.asarray(timings), memory_sizes=np.asarray(memory_sizes, dtype=np.int64))
                        payload_sha = digest(out / relative)
                        serialization_seconds = time.perf_counter() - tick
                        elapsed = time.perf_counter() - episode_start
                        record = {"layout_index": index, "seed": seed, "order_index": order, "mode": mode,
                                      "success": bool(info["success"]), "steps": len(actions), "falls": falls,
                                      "elapsed_seconds": elapsed, "environment_seconds": env_seconds,
                                      "policy_seconds": policy_seconds, "parsing_seconds": parsing_seconds,
                                      "update_seconds": update_seconds, "serialization_seconds": serialization_seconds,
                                      "reset_seconds": reset_seconds,
                                      "evaluator_seconds": elapsed-reset_seconds-env_seconds-policy_seconds-parsing_seconds-update_seconds-serialization_seconds,
                                      "peak_serialized_state_bytes": max(s[0] for s in memory_sizes),
                                      "peak_retained_python_bytes": max(s[1] for s in memory_sizes),
                                      "npz_path": relative, "npz_sha256": payload_sha, "layout_sha256": sha}
                        ledger.write(json.dumps(record, allow_nan=False) + "\n")
                        completed += 1
                if (index + 1) % 8 == 0:
                    print(json.dumps({"layouts_completed": index + 1, "episodes": completed,
                                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
        assert completed == 1280 and total_steps <= 163840
        assert time.perf_counter() - started <= 900
        save_json(out / "completed.json", {"status": "complete", "episodes": completed,
                                           "total_steps": total_steps, "elapsed_seconds": time.perf_counter()-started,
                                           "episodes_sha256": digest(out / "episodes.jsonl"),
                                           "protocol_sha256": digest(BASE / "protocol.json"),
                                           "bindings_sha256": digest(BASE / "bindings.json"),
                                           "inputs_sha256": digest(BASE / "inputs.json")})
        if time.perf_counter() - started > 900:
            (out / "completed.json").rename(out / "late-completion.json")
            raise TimeoutError("Completion write exceeded execution wall cap")
    except BaseException:
        record_failure(out / "failed.json", {"status": "failed", "error": traceback.format_exc(),
                                       "episodes_completed": completed, "steps": total_steps,
                                       "elapsed_seconds": time.perf_counter()-started})
        raise
    finally:
        if env is not None:
            close_env(env, out)


def audit(attempt):
    out = attempt / "audit"
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    env = None
    count = total_steps = 0
    try:
        save_json(out / "started.json", {"status": "started", "wall_cap_seconds": 900, "pid": os.getpid()})
        verify_bindings()
        assert not (attempt / "failed.json").exists()
        assert not (attempt / "late-completion.json").exists()
        completion = json.loads((attempt / "completed.json").read_text())
        assert completion["status"] == "complete"
        for name in ("protocol", "bindings", "inputs"):
            assert completion[name + "_sha256"] == digest(BASE / (name + ".json"))
        assert digest(attempt / "episodes.jsonl") == completion["episodes_sha256"]
        records = [json.loads(x) for x in (attempt / "episodes.jsonl").read_text().splitlines()]
        assert len(records) == 1280
        expected = {(i, o, m) for i in range(64) for o in range(4) for m in MODES}
        assert {(r["layout_index"], r["order_index"], r["mode"]) for r in records} == expected
        seeds = json.loads((BASE / "inputs.json").read_text())["seeds"]
        env = GridMysteryPathEnv()
        with (out / "episodes.jsonl").open("x", buffering=1) as ledger:
            for row in records:
                assert row["seed"] == seeds[row["layout_index"]]
                assert row["npz_path"] == f'episodes/{row["layout_index"]:03d}-{row["mode"]}-order{row["order_index"]}.npz'
                source = attempt / row["npz_path"]
                assert digest(source) == row["npz_sha256"]
                with np.load(source, allow_pickle=False) as archive:
                    payload = {name: archive[name] for name in archive.files}
                    n = row["steps"]
                    expected_arrays = {"frames": ((n+1,84,84,3), np.dtype('uint8')),
                                       "observations": ((n+1,4), np.dtype('int16')),
                                       "actions": ((n,), np.dtype('uint8')),
                                       "rewards": ((n,), np.dtype('float64')),
                                       "terminals": ((n,), np.dtype('bool')),
                                       "timings": ((n,4), np.dtype('float64')),
                                       "memory_sizes": ((n,2), np.dtype('int64'))}
                    assert set(payload) == set(expected_arrays)
                    for name, (shape, dtype) in expected_arrays.items():
                        assert payload[name].shape == shape and payload[name].dtype == dtype
                    frame, _ = env.reset(seed=row["seed"])
                    safe, goal, layout_sha = layout(env)
                    assert layout_sha == row["layout_sha256"]
                    assert np.array_equal(frame, payload["frames"][0])
                    obs = check_frame(env, frame)
                    assert np.array_equal(astuple(obs), payload["observations"][0])
                    priority = tuple((row["order_index"] + i) % 4 for i in range(4))
                    controller = None if row["mode"] == "reference" else MemoryController(row["mode"], priority)
                    if controller is not None:
                        controller.reset(obs)
                    falls = 0
                    assert len(payload["actions"]) == row["steps"]
                    for step, action in enumerate(payload["actions"]):
                        assert time.perf_counter()-started <= 900
                        expected_action = (reference_action(obs, safe, goal) if controller is None
                                           else controller.act(obs))
                        assert int(action) == expected_action
                        frame, reward, done, truncated, info = env.step(int(action))
                        total_steps += 1
                        observed = check_frame(env, frame)
                        assert np.array_equal(frame, payload["frames"][step+1])
                        assert np.array_equal(astuple(observed), payload["observations"][step+1])
                        assert reward == payload["rewards"][step]
                        assert done == payload["terminals"][step] and not truncated
                        assert bool(done) == (step == row["steps"]-1)
                        if controller is not None:
                            controller.observe(obs, int(action), observed, reward)
                            sizes = controller.memory_bytes()
                            assert np.array_equal(payload["memory_sizes"][step],
                                                  [sizes["serialized_state_bytes"], sizes["retained_python_bytes"]])
                        else:
                            sizes = reference_memory(safe, goal)
                            assert np.array_equal(payload["memory_sizes"][step],
                                                  [sizes["serialized_state_bytes"], sizes["retained_python_bytes"]])
                        obs = observed
                        falls += int(observed.off_path)
                    assert bool(info["success"]) == row["success"]
                    assert falls == row["falls"] == info["num_fails"]
                    assert np.isfinite(payload["timings"]).all() and (payload["timings"] >= 0).all()
                    for column, key in ((0, "policy_seconds"), (1, "environment_seconds"), (3, "update_seconds")):
                        assert abs(float(payload["timings"][:, column].sum()) - row[key]) <= 1e-9
                    assert float(payload["timings"][:, 2].sum()) <= row["parsing_seconds"] + 1e-9
                    assert np.array_equal(payload["memory_sizes"].max(axis=0),
                                          [row["peak_serialized_state_bytes"], row["peak_retained_python_bytes"]])
                ledger.write(json.dumps({"npz_path": row["npz_path"], "status": "passed", "steps": row["steps"]})+"\n")
                count += 1
                if count % 160 == 0:
                    print(json.dumps({"audit_episodes": count, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        assert total_steps == completion["total_steps"] and count == 1280
        save_json(out / "completed.json", {"status": "passed", "episodes": count, "steps": total_steps,
                                           "episodes_sha256": completion["episodes_sha256"],
                                           "elapsed_seconds": time.perf_counter()-started,
                                           "audit_ledger_sha256": digest(out / "episodes.jsonl")})
        if time.perf_counter() - started > 900:
            (out / "completed.json").rename(out / "late-completion.json")
            raise TimeoutError("Completion write exceeded audit wall cap")
    except BaseException:
        record_failure(out / "failed.json", {"error": traceback.format_exc(), "episodes": count, "steps": total_steps})
        raise
    finally:
        if env is not None:
            close_env(env, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "audit"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"preflight": preflight, "run": run, "audit": audit}[args.command](args.out)
