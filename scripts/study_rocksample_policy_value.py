"""One frozen classical RockSample control screen, with complete public traces."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.rocksample_memory_controls import PublicTransition, apply_transition, reconstruct
from openjev.research.rocksample_particle_belief import ParticleRockBelief
from openjev.research.rocksample_quality_belief import QualityRockBelief
from openjev.research.rocksample_route_planner import MAX_CHECKS, plan_action
from openjev.research.suspend_clock import SuspendClock

ARMS = ("exit", "full", "recent128", "latest", "quality", "privileged")
MAP_SEEDS = tuple(range(12001, 12009))
RESET_SEEDS = tuple(range(22001, 22005))
HORIZON = 1000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_position(obs):
    array = np.asarray(obs)
    if array.shape != (33,) or not np.isfinite(array).all():
        raise ValueError("expected finite raw33 observation")
    coordinates = array[:22].reshape(2, 11)
    if not np.isin(coordinates, (0, 1)).all() or not np.all(coordinates.sum(axis=1) == 1):
        raise ValueError("public coordinates must be one-hot")
    return tuple(int(x) for x in np.argmax(coordinates, axis=1))


def play_episode(env, params, arm, reset_seed, transition_seed, prior, jax, emit, check, counters):
    """Simulator state stays here; public beliefs only receive PublicTransition."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    obs, state = env.reset(jax.random.PRNGKey(reset_seed), params)
    obs = np.asarray(obs)
    position = public_position(obs)
    start = time.perf_counter()
    if arm == "exit":
        belief = None
    elif arm == "privileged":
        # The only explicit hidden-information path, labeled in every output.
        belief = ParticleRockBelief.from_hypotheses(np.asarray(env.rock_positions)[None],
                    qualities=np.asarray(state.env_state.rock_morality)[None])
    elif arm == "quality":
        belief = QualityRockBelief()
    else:
        belief = prior.copy()
    filter_seconds = time.perf_counter() - start
    planning_seconds = env_seconds = 0.
    history, ledger = [], np.zeros((11, 10), dtype=bool)
    key = jax.random.PRNGKey(transition_seed)
    total = discounted = 0.
    step = checks = samples = good = bad = empty = plans = 0
    ess_ok = ess_count = 0
    minimum_ess = None
    maximum_weight = None
    kinds = {"sense": 0, "exploit": 0, "exit": 0}
    work_totals = {}
    done = truncated = False
    emit("transitions", {"step": -1, "observation": obs.tolist(), "event": "reset"})
    while not done:
        check()
        if step >= HORIZON:
            raise ValueError("native time limit failed to end episode")
        t = time.perf_counter()
        if arm in ("recent128", "latest"):
            belief = reconstruct(prior, history, mode=arm)
        diagnostics = belief.diagnostics() if belief is not None else {"kind": "none", "ess": None}
        filter_seconds += time.perf_counter() - t
        if arm != "exit" and diagnostics["ess"] is not None:
            ess_count += 1
            ess_ok += diagnostics["ess"] >= 8
            minimum_ess = diagnostics["ess"] if minimum_ess is None else min(minimum_ess, diagnostics["ess"])
            maximum_weight = diagnostics["max_weight"] if maximum_weight is None else max(maximum_weight, diagnostics["max_weight"])
        t = time.perf_counter()
        if arm == "exit":
            plan = {"kind": "exit", "actions": [1] * min(10 - position[1], HORIZON - step), "work": {}}
        else:
            plan = plan_action(belief, position, ledger, HORIZON - step, MAX_CHECKS - checks)
        plan_seconds = time.perf_counter() - t
        planning_seconds += plan_seconds
        actions = plan["actions"]
        if not actions or len(actions) > HORIZON - step or any(type(a) is not int or not 0 <= a < 16 for a in actions):
            raise ValueError("invalid committed macro")
        if sum(a >= 5 for a in actions) > MAX_CHECKS - checks:
            raise ValueError("sensing budget exceeded")
        plans += 1
        kinds[plan["kind"]] += 1
        for name, value in plan["work"].items():
            work_totals[name] = max(work_totals.get(name, 0), value) if name.startswith("maximum_") else work_totals.get(name, 0) + value
        emit("decisions", {"step": step, "position": position, "remaining_steps": HORIZON - step,
                           "checks_remaining": MAX_CHECKS - checks, "sampled_cells": int(ledger.sum()),
                           "belief_diagnostics": diagnostics, "plan_seconds": plan_seconds, "plan": plan})
        for ai, action in enumerate(actions):
            check()
            previous = position
            if action == 4 and ledger[previous]:
                raise ValueError("controller attempted to resample a depleted cell")
            key, step_key = jax.random.split(key)
            if counters["steps_attempted"] >= 192000:
                raise ValueError("global native step allowance exhausted")
            counters["steps_attempted"] += 1
            counters["active_step"] = step
            t = time.perf_counter()
            next_obs, state, reward, ended, info = env.step(step_key, state, action, params)
            next_obs, reward, done = np.asarray(next_obs), float(reward), bool(ended)
            env_seconds += time.perf_counter() - t
            counters["steps_returned"] += 1
            truncated = bool(info["truncated"])
            if not np.isfinite(reward) or reward not in (-10., 0., 10.):
                raise ValueError("invalid native reward")
            emit("transitions", {"step": step, "action": action, "observation": next_obs.tolist(),
                                 "reward": reward, "done": done, "truncated": truncated,
                                 "time_limit_reached": bool(info["time_limit_reached"]), "plan_index": plans - 1})
            total += reward
            discounted += (0.99 ** step) * reward
            checks += action >= 5
            samples += action == 4
            if action == 4:
                ledger[previous] = True
                good += reward == 10
                bad += reward == -10
                empty += reward == 0
            if not done:
                readings = next_obs[22:]
                selected = np.flatnonzero(readings)
                if action >= 5:
                    if selected.tolist() != [action - 5] or readings[action - 5] not in (-1., 1.):
                        raise ValueError("wrong check outcome channel")
                    reading = int(readings[action - 5])
                else:
                    if len(selected):
                        raise ValueError("noncheck action unexpectedly reports a reading")
                    reading = None
                event = PublicTransition(previous, action, reading)
                history.append(event)
                t = time.perf_counter()
                if arm in ("full", "quality", "privileged"):
                    belief = apply_transition(belief, event)
                filter_seconds += time.perf_counter() - t
                position = public_position(next_obs)
            step += 1
            if done:
                if ai != len(actions) - 1:
                    raise ValueError("native boundary occurred inside a committed macro")
                break
    if samples != good + bad + empty or checks > MAX_CHECKS:
        raise ValueError("episode accounting failed")
    return {"arm": arm, "raw_return": total, "discounted_return_099": discounted, "steps": step,
            "checks": checks, "samples": samples, "good_samples": good, "bad_samples": bad,
            "empty_samples": empty, "exited": not truncated, "truncated": truncated,
            "plans": plans, "plan_kinds": kinds, "work": work_totals,
            "filter_seconds": filter_seconds, "planning_seconds": planning_seconds,
            "controller_seconds": filter_seconds + planning_seconds, "environment_seconds": env_seconds,
            "ess_adequate_plans": int(ess_ok), "ess_evaluated_plans": ess_count,
            "minimum_ess": minimum_ess, "maximum_map_weight": maximum_weight,
            "information": "true map and qualities" if arm == "privileged" else "public history only"}


def summarize(rows):
    identities = [(r["map_seed"], r["reset_seed"], r["arm"]) for r in rows]
    expected = {(m, r, a) for m in MAP_SEEDS for r in RESET_SEEDS for a in ARMS}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("incomplete or duplicated fixed cohort")
    by_map = []
    metrics = ("raw_return", "discounted_return_099", "steps", "checks", "samples", "good_samples",
               "bad_samples", "empty_samples", "controller_seconds", "filter_seconds", "planning_seconds",
               "environment_seconds", "plans", "exited", "truncated")
    for m in MAP_SEEDS:
        scores = {a: {k: float(np.mean([r[k] for r in rows if r["map_seed"] == m and r["arm"] == a]))
                      for k in metrics} for a in ARMS}
        by_map.append({"map_seed": m, "scores": scores})
    means = {a: {k: float(np.mean([m["scores"][a][k] for m in by_map])) for k in metrics} for a in ARMS}
    criteria = []
    for control in ("recent128", "latest", "quality"):
        gains = [m["scores"]["full"]["raw_return"] - m["scores"][control]["raw_return"] for m in by_map]
        gain = means["full"]["raw_return"] - means[control]["raw_return"]
        criteria.extend([{"name": f"raw_gain_vs_{control}", "value": gain, "passes": gain >= 5},
                         {"name": f"positive_maps_vs_{control}", "value": sum(x > 0 for x in gains),
                          "map_gains": gains, "passes": sum(x > 0 for x in gains) >= 6}])
    for arm, threshold in (("full", 5), ("privileged", 20)):
        gain = means[arm]["raw_return"] - means["exit"]["raw_return"]
        criteria.append({"name": f"{arm}_gain_vs_exit", "value": gain, "passes": gain >= threshold})
    full = [r for r in rows if r["arm"] == "full"]
    adequate = sum(r["ess_adequate_plans"] for r in full)
    denominator = sum(r["ess_evaluated_plans"] for r in full)
    criteria.extend([{"name": "finite_particle_adequacy", "value": adequate / denominator if denominator else 0,
                      "numerator": adequate, "denominator": denominator, "passes": denominator > 0 and adequate * 10 >= 9 * denominator},
                     {"name": "complete_cohort", "value": len(rows), "passes": True}])
    return {"scope": "classical policy-value screen, not a trained architecture result", "episodes": len(rows),
            "scores_equal_map_means": means, "by_map": by_map, "criteria": criteria,
            "learned_pilot_admitted": all(c["passes"] for c in criteria)}


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "started", "steps_attempted": 0, "steps_returned": 0, "completed_episodes": 0}
    check = None
    try:
        clock = SuspendClock()
        t0 = clock.now_ns()
        while not args.supervision.exists():
            if clock.now_ns() - t0 > 5 * 10**9:
                raise ValueError("supervisor missing")
            time.sleep(.01)
        launch = json.loads(args.supervision.read_text())
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        if (command != [sys.executable, *sys.argv] or launch["pid"] != os.getpid()
                or launch["pgid"] != os.getpgrp() or launch["parent_pid"] != os.getppid()
                or launch["cap_seconds"] != 1800 or launch["clock_backend"] != clock.backend
                or Path(launch["cwd"]).resolve() != ROOT or Path.cwd().resolve() != ROOT
                or launch["deadline_ns"] != launch["started_ns"] + 1800 * 10**9):
            raise ValueError("supervisor binding mismatch")
        if sha(args.plan) != args.plan_sha256:
            raise ValueError("externally pinned plan changed")
        plan = json.loads(args.plan.read_text())
        receipt.update(plan_sha256=sha(args.plan), supervision_sha256=sha(args.supervision), sources=plan["sources"])

        def authenticate():
            for name, pin in plan["sources"].items():
                if sha(ROOT / name) != pin:
                    raise ValueError(f"frozen source changed: {name}")

        def check():
            if clock.now_ns() >= launch["deadline_ns"]:
                raise TimeoutError("native deadline expired")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            receipt["peak_rss_bytes"] = peak
            if peak > 8 * 1024**3 or sum(p.stat().st_size for p in output.iterdir()) > 512 * 1024**2:
                raise MemoryError("resource cap exceeded")
            if receipt["steps_attempted"] > 192000:
                raise ValueError("step cap exceeded")

        authenticate()
        check()
        from qualify_rocksample_runtime import authenticate_source
        authenticate_source()
        qualified = ROOT / "output/rocksample-runtime-v1/qualification-01"
        if sha(qualified / "completed.json") != plan["runtime_qualification_sha256"]:
            raise ValueError("qualified runtime changed")
        qualification = json.loads((qualified / "completed.json").read_text())
        for name, expected in qualification["files"].items():
            path = qualified / name
            if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"qualification payload changed: {name}")
        terminal_path = qualified.parent / "qualification-process-01.terminal.json"
        if sha(terminal_path) != plan["runtime_terminal_sha256"]:
            raise ValueError("runtime terminal changed")
        if json.loads(terminal_path.read_text())["status"] != "completed":
            raise ValueError("runtime qualification did not complete")
        imports = json.loads((qualified / "imports.json").read_text())
        if sha(imports["gymnax_environment"]["path"]) != imports["gymnax_environment"]["sha256"]:
            raise ValueError("qualified Gymnax source changed")
        for name, pin in imports["upstream_imported_source_sha256"].items():
            if sha(ROOT / "tmp/pobax-source-review-01" / name) != pin:
                raise ValueError(f"qualified upstream import changed: {name}")
        for name, version in imports["versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"runtime version changed: {name}")
        sys.path.insert(0, str(ROOT / "tmp/pobax-source-review-01"))
        import jax
        from pobax.envs.jax.rocksample import RockSample
        from pobax.envs.wrappers.gymnax import TimeLimitWrapper
        if any(d.platform != "cpu" for d in jax.devices()):
            raise ValueError("CPU only")
        prior = ParticleRockBelief()
        receipt["particle_bank_sha256"] = hashlib.sha256(prior.maps.tobytes()).hexdigest()
        rows = []
        streams = {name: (output / f"{name}.jsonl").open("x") for name in ("transitions", "decisions", "episodes")}
        try:
            for mi, map_seed in enumerate(MAP_SEEDS):
                base = RockSample(jax.random.PRNGKey(map_seed), config_path=ROOT / "tmp/pobax-source-review-01/pobax/envs/configs/rocksample_11_11_config.json")
                env, params = TimeLimitWrapper(base), base.default_params
                if params.max_steps_in_episode != HORIZON:
                    raise ValueError("native horizon changed")
                for ei, reset_seed in enumerate(RESET_SEEDS):
                    ci = mi * 4 + ei
                    order = ARMS[ci % len(ARMS):] + ARMS[:ci % len(ARMS)]
                    for arm in order:
                        identity = {"map_seed": map_seed, "reset_seed": reset_seed, "arm": arm}
                        receipt["active_case"] = identity
                        receipt["active_step"] = None

                        def emit(kind, value, identity=identity):
                            streams[kind].write(json.dumps({**identity, **value}, allow_nan=False) + "\n")
                            streams[kind].flush()

                        row = {**identity, **play_episode(env, params, arm, reset_seed, 430000 + ci,
                                                          prior, jax, emit, check, receipt)}
                        emit("episodes", row)
                        rows.append(row)
                        receipt["completed_episodes"] += 1
                        print(json.dumps({"completed_episodes": len(rows), "returned_steps": receipt["steps_returned"]}), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        if receipt["steps_attempted"] != receipt["steps_returned"]:
            raise ValueError("unreturned transition")
        if sum(r["steps"] for r in rows) != receipt["steps_returned"] or len(rows) != receipt["completed_episodes"]:
            raise ValueError("episode and process accounting disagree")
        summary = summarize(rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        authenticate()
        authenticate_source()
        check()
        receipt.update(status="completed", native_elapsed_seconds=(clock.now_ns() - launch["started_ns"]) / 1e9,
                       learned_pilot_admitted=summary["learned_pilot_admitted"], training_updates=0, external_model_calls=0,
                       worker_timing_scope="native elapsed through final summary and source checks, before receipt hashing/publication; supervisor terminal contains whole-run elapsed",
                       controller_timing_scope="belief initialization, reconstruction, assimilation and diagnostics plus planner calls; excludes trace I/O, reset, imports, JIT and bookkeeping; whole-run native time includes these")
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        original = sys.exception()
        try:
            receipt["files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            if receipt["status"] == "completed":
                try:
                    check()
                except BaseException as late_error:
                    receipt.update(status="failed", error=repr(late_error), late_failure=True)
                    try:
                        (output / "receipt.json").rename(output / "late-completed.json")
                        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
                    except BaseException as publication_error:  # noqa: BLE001 - preserve the original late failure
                        late_error.add_note(f"Late failure publication also failed: {publication_error!r}")
                    raise
        except BaseException as error:
            if original is None:
                raise
            original.add_note(f"Receipt publication also failed: {error!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    execute(parser.parse_args())
