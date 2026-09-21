"""Fresh-map representation screen reusing the frozen public controller loop."""
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
import study_rocksample_policy_value as baseline

from openjev.research.rocksample_factorized_planning import FactorizedRockBelief
from openjev.research.rocksample_particle_belief import ParticleRockBelief
from openjev.research.suspend_clock import SuspendClock

ARMS = ("exit", "factorized_full", "factorized_recent128", "factorized_latest",
        "particle_full", "quality", "known_map", "privileged")
MAP_SEEDS = tuple(range(13001, 13009))
RESET_SEEDS = tuple(range(23001, 23005))
HORIZON = 1000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def play_episode(env, params, arm, reset_seed, transition_seed, prior, jax, emit, check, counters):
    if arm not in ARMS:
        raise ValueError("unknown representation arm")
    start = time.perf_counter()
    role = arm
    if arm.startswith("factorized_"):
        role = arm.removeprefix("factorized_")
        episode_prior = FactorizedRockBelief()
    elif arm == "known_map":
        # Deliberately supplied map; no quality/state access in this adapter.
        episode_prior = ParticleRockBelief.from_hypotheses(np.asarray(env.rock_positions)[None], qualities=0.5)
        role = "full"
    else:
        episode_prior = prior
        if arm == "particle_full":
            role = "full"
    construction_seconds = time.perf_counter() - start
    diagnostics = {"planning_decisions": 0, "maximum_cell_occupancy": None,
                   "overfull_cell_decisions": 0, "overfull_cells_total": 0,
                   "minimum_expected_reward": None, "maximum_expected_reward": None}

    def record(kind, value):
        if kind == "decisions" and arm.startswith("factorized_"):
            info = value["belief_diagnostics"]
            numbers = [info[k] for k in ("maximum_cell_occupancy", "minimum_expected_reward", "maximum_expected_reward")]
            if not np.isfinite(numbers).all():
                raise ValueError("nonfinite factorized diagnostics")
            diagnostics["planning_decisions"] += 1
            diagnostics["overfull_cell_decisions"] += info["overfull_cells"] > 0
            diagnostics["overfull_cells_total"] += info["overfull_cells"]
            for name in ("maximum_cell_occupancy", "minimum_expected_reward", "maximum_expected_reward"):
                current = diagnostics[name]
                combine = min if name.startswith("minimum_") else max
                diagnostics[name] = info[name] if current is None else combine(current, info[name])
        emit(kind, value)

    result = baseline.play_episode(env, params, role, reset_seed, transition_seed,
                                   episode_prior, jax, record, check, counters)
    result["arm"] = arm
    result["filter_seconds"] += construction_seconds
    result["controller_seconds"] += construction_seconds
    result["representation_construction_seconds"] = construction_seconds
    result["representation_diagnostics"] = diagnostics
    if arm == "known_map":
        result["information"] = "supplied true map; qualities inferred from public history"
    return result


def summarize(rows):
    ids = [(r["map_seed"], r["reset_seed"], r["arm"]) for r in rows]
    expected = {(m, r, a) for m in MAP_SEEDS for r in RESET_SEEDS for a in ARMS}
    if len(ids) != 256 or set(ids) != expected:
        raise ValueError("incomplete or duplicated fixed representation cohort")
    metrics = ("raw_return", "discounted_return_099", "steps", "checks", "samples", "good_samples",
               "bad_samples", "empty_samples", "controller_seconds", "filter_seconds", "planning_seconds",
               "environment_seconds", "representation_construction_seconds", "plans", "exited", "truncated")
    maps = [{"map_seed": m, "scores": {a: {k: float(np.mean([r[k] for r in rows if r["map_seed"] == m and r["arm"] == a]))
                                            for k in metrics} for a in ARMS}} for m in MAP_SEEDS]
    means = {a: {k: float(np.mean([m["scores"][a][k] for m in maps])) for k in metrics} for a in ARMS}
    criteria = []
    for control in ("factorized_recent128", "factorized_latest", "quality", "particle_full"):
        gains = [m["scores"]["factorized_full"]["raw_return"] - m["scores"][control]["raw_return"] for m in maps]
        gain = means["factorized_full"]["raw_return"] - means[control]["raw_return"]
        criteria.extend([{"name": f"raw_gain_vs_{control}", "value": gain, "passes": gain >= 5},
                         {"name": f"positive_maps_vs_{control}", "value": sum(x > 0 for x in gains),
                          "map_gains": gains, "passes": sum(x > 0 for x in gains) >= 6}])
    for arm, threshold in (("factorized_full", 5), ("known_map", 20), ("privileged", 20)):
        gain = means[arm]["raw_return"] - means["exit"]["raw_return"]
        criteria.append({"name": f"{arm}_gain_vs_exit", "value": gain, "passes": gain >= threshold})
    all_diagnostics = {}
    for arm in ("factorized_full", "factorized_recent128", "factorized_latest"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        group = [r["representation_diagnostics"] for r in arm_rows]
        for row, diagnostic in zip(arm_rows, group, strict=True):
            count = diagnostic["planning_decisions"]
            overfull = diagnostic["overfull_cell_decisions"]
            cells = diagnostic["overfull_cells_total"]
            if (any(type(x) is not int or x < 0 for x in (count, overfull, cells))
                    or count <= 0 or count != row["plans"] or overfull > count
                    or not overfull <= cells <= 110 * overfull):
                raise ValueError("factorized diagnostic coverage or counts disagree")
        all_diagnostics[arm] = {k: sum(g[k] for g in group) for k in ("planning_decisions", "overfull_cell_decisions", "overfull_cells_total")}
        for name in ("maximum_cell_occupancy", "minimum_expected_reward", "maximum_expected_reward"):
            values = [g[name] for g in group]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite or missing factorized summary")
            all_diagnostics[arm][name] = min(values) if name.startswith("minimum_") else max(values)
    full = all_diagnostics["factorized_full"]
    bounds = (full["maximum_cell_occupancy"] <= 1 and full["overfull_cells_total"] == 0
              and full["minimum_expected_reward"] >= -10 and full["maximum_expected_reward"] <= 10)
    criteria.extend([{"name": "complete_cohort", "value": len(rows), "passes": True},
                     {"name": "factorized_observed_forecast_bounds", "value": full, "passes": bounds}])
    return {"scope": "state representation and information diagnostic; no learned architecture result", "episodes": len(rows),
            "scores_equal_map_means": means, "by_map": maps, "representation_diagnostics": all_diagnostics,
            "criteria": criteria, "learned_pilot_admitted": all(c["passes"] for c in criteria)}


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

                        row = {**identity, **play_episode(env, params, arm, reset_seed, 440000 + ci,
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
