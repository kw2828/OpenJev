"""Fixed native exploration diagnostic; forecasting evidence, not policy efficacy."""
from __future__ import annotations

import argparse
import hashlib
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
from openjev.research.rocksample_public_belief import PublicRockBelief
from openjev.research.suspend_clock import SuspendClock

ARMS = ("prior", "full", "recent32", "recent128", "latest_check")
MAP_SEEDS = tuple(range(11001, 11009))
RESET_SEEDS = tuple(range(21001, 21005))
STEPS = 256


def position(observation):
    return np.argmax(np.asarray(observation)[:22].reshape(2, 11), axis=1)


def forecast(history, full, coordinate, rock):
    """Only completed transitions enter this function; the next outcome is absent."""
    previous = [i for i, row in enumerate(history) if row[1] == rock + 5]
    latest = previous[-1] if previous else None
    predictions = {"prior": 0.5, "full": float(full.probabilities(coordinate)[rock])}
    for arm, start in (("recent32", max(0, len(history) - 32)),
                       ("recent128", max(0, len(history) - 128)),
                       ("latest_check", latest if latest is not None else 0)):
        belief = PublicRockBelief()
        for prior_position, action, observation, done in history[start:]:
            belief.update(prior_position, action, observation, done)
        predictions[arm] = float(belief.probabilities(coordinate)[rock])
    lag = len(history) - latest if latest is not None else None
    age = "never" if lag is None else "le32" if lag <= 32 else "33to128" if lag <= 128 else "gt128"
    return predictions, {"age": age, "prior_reading_age": lag,
                         "prior_checks": len(previous),
                         "samples_since_last_check": sum(row[1] == 4 for row in history[(latest + 1) if latest is not None else 0:])}


def losses(predictions, positive):
    result = {}
    for arm, probability in predictions.items():
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"invalid probability from {arm}")
        likelihood = probability if positive else 1 - probability
        if likelihood <= 0:
            raise ValueError(f"impossible event predicted by {arm}; no clipping allowed")
        result[arm] = {"nll": float(-np.log(likelihood)),
                       "brier": float((probability - int(positive)) ** 2)}
    return result


def aggregate(rows):
    return {arm: {metric: float(np.mean([r["losses"][arm][metric] for r in rows]))
                  for metric in ("nll", "brier")} for arm in ARMS} if rows else None


def summarize(records):
    by_map = []
    for seed in MAP_SEEDS:
        rows = [r for r in records if r["map_seed"] == seed]
        by_map.append({"map_seed": seed, "checks": len(rows), "scores": aggregate(rows),
                       "ages": {age: {"count": len(sub := [r for r in rows if r["age"] == age]),
                                       "scores": aggregate(sub)} for age in ("never", "le32", "33to128", "gt128")}})
    if any(m["scores"] is None for m in by_map):
        raise ValueError("each fixed map must have scored checks")
    means = {a: {k: float(np.mean([m["scores"][a][k] for m in by_map]))
                 for k in ("nll", "brier")} for a in ARMS}
    criteria = []
    for control in ("recent128", "latest_check"):
        gains = [m["scores"][control]["nll"] - m["scores"]["full"]["nll"] for m in by_map]
        criteria.extend([
            {"name": f"relative_nll_gain_vs_{control}", "value": 1 - means["full"]["nll"] / means[control]["nll"],
             "passes": means["full"]["nll"] <= 0.9 * means[control]["nll"]},
            {"name": f"positive_maps_vs_{control}", "value": sum(x > 0 for x in gains),
             "passes": sum(x > 0 for x in gains) >= 6, "map_gains": gains}])
    delayed = [r for r in records if r["age"] == "gt128"]
    criteria.extend([
        {"name": "delayed_endpoints", "value": len(delayed), "passes": len(delayed) >= 32},
        {"name": "delayed_maps", "value": len({r["map_seed"] for r in delayed}),
         "passes": len({r["map_seed"] for r in delayed}) >= 4}])
    delayed_maps = [m["ages"]["gt128"]["scores"] for m in by_map if m["ages"]["gt128"]["count"]]
    delayed_mean = {a: {k: float(np.mean([m[a][k] for m in delayed_maps]))
                         for k in ("nll", "brier")} for a in ARMS} if delayed_maps else None
    return {"scope": "fixed exploratory prediction diagnostic; no training or control comparison",
            "scores_equal_map_means": means, "by_map": by_map, "check_endpoints": len(records),
            "delayed_equal_map_means": delayed_mean, "delayed_contributing_maps": len(delayed_maps),
            "pooled_age_breakdown": {a: {"count": len(sub := [r for r in records if r["age"] == a]),
                                             "scores": aggregate(sub)} for a in ("never", "le32", "33to128", "gt128")},
            "criteria": criteria, "memory_pilot_admitted": all(c["passes"] for c in criteria)}


def run(output, supervision):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "running", "supervision": str(supervision), "transitions": 0,
               "completed_fragments": 0, "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__), ROOT / "src/openjev/research/rocksample_public_belief.py",
                         ROOT / "research/rocksample-public-memory-diagnostic-protocol.md",
                         ROOT / "scripts/qualify_rocksample_runtime.py",
                         ROOT / "scripts/supervise_dialogue_observation_v2.py",
                         ROOT / "src/openjev/research/suspend_clock.py",
                         ROOT / "research/rocksample-runtime-requirements.lock.txt"]}}
    try:
        import importlib.metadata

        from qualify_rocksample_runtime import authenticate_source
        clock = SuspendClock()
        wait_start = clock.now_ns()
        while not supervision.exists():
            if clock.now_ns() - wait_start > 5 * 10**9:
                raise ValueError("supervisor launch missing")
            time.sleep(0.01)
        launch = json.loads(supervision.read_text())
        command = list(launch["command"])
        if len(command) > 1 and command[1] == "-u":
            command.pop(1)
        if (command != [sys.executable, *sys.argv] or launch["pid"] != os.getpid()
                or launch["pgid"] != os.getpgrp() or launch["parent_pid"] != os.getppid()
                or Path(launch["cwd"]).resolve() != Path.cwd().resolve()
                or launch["cap_seconds"] != 600 or launch["clock_backend"] != clock.backend
                or launch["deadline_ns"] != launch["started_ns"] + 600 * 10**9):
            raise ValueError("supervisor binding mismatch")
        receipt["supervision_sha256"] = hashlib.sha256(supervision.read_bytes()).hexdigest()
        receipt["native_started_ns"] = launch["started_ns"]

        def check_resources():
            if clock.now_ns() >= launch["deadline_ns"]:
                raise TimeoutError("native diagnostic deadline expired")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak *= 1 if sys.platform == "darwin" else 1024
            receipt["peak_rss_bytes"] = peak
            if peak > 8 * 1024**3 or sum(p.stat().st_size for p in output.iterdir()) > 128 * 1024**2:
                raise MemoryError("diagnostic resource cap exceeded")

        check_resources()
        qualification = ROOT / "output/rocksample-runtime-v1/qualification-01"
        completed = json.loads((qualification / "completed.json").read_text())
        terminal = json.loads((qualification.parent / "qualification-process-01.terminal.json").read_text())
        if completed["status"] != "completed" or terminal["status"] != "completed" or terminal["returncode"] != 0:
            raise ValueError("runtime must be successfully qualified")
        if completed["source_sha256"] != receipt["source_hashes"]["scripts/qualify_rocksample_runtime.py"]:
            raise ValueError("qualification source changed")
        for name, metadata in completed["files"].items():
            if hashlib.sha256((qualification / name).read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("qualification evidence changed")
        imported = json.loads((qualification / "imports.json").read_text())
        for name, version in imported["versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"runtime dependency changed: {name}")
        for name, pin in imported["upstream_imported_source_sha256"].items():
            if hashlib.sha256((ROOT / "tmp/pobax-source-review-01" / name).read_bytes()).hexdigest() != pin:
                raise ValueError(f"qualified imported source changed: {name}")
        receipt["qualified_runtime_sha256"] = hashlib.sha256((qualification / "completed.json").read_bytes()).hexdigest()
        authenticate_source()
        if importlib.metadata.version("gymnax") != "0.0.9":
            raise ValueError("qualified gymnax 0.0.9 required")
        sys.path.insert(0, str(ROOT / "tmp/pobax-source-review-01"))
        import jax
        from pobax.envs.jax.rocksample import RockSample
        if any(d.platform != "cpu" for d in jax.devices()):
            raise ValueError("CPU-only diagnostic required")
        records = []
        with (output / "traces.jsonl").open("x") as traces, (output / "predictions.jsonl").open("x") as scored:
            for mi, map_seed in enumerate(MAP_SEEDS):
                env = RockSample(jax.random.PRNGKey(map_seed), config_path=ROOT / "tmp/pobax-source-review-01/pobax/envs/configs/rocksample_11_11_config.json")
                params = env.default_params
                for ei, reset_seed in enumerate(RESET_SEEDS):
                    obs, state = env.reset(jax.random.PRNGKey(reset_seed), params)
                    rng = np.random.default_rng(31000 + mi * 4 + ei)
                    key = jax.random.PRNGKey(41000 + mi * 4 + ei)
                    history, belief = [], PublicRockBelief()
                    traces.write(json.dumps({"map_seed": map_seed, "reset_seed": reset_seed, "step": -1,
                                             "observation": np.asarray(obs).tolist()}) + "\n")
                    for step in range(STEPS):
                        coordinate = position(obs)
                        choice = int(rng.integers(20))
                        action = choice // 3 if choice < 12 else 4 if choice < 16 else 5 + int(rng.integers(11))
                        if action == 1 and coordinate[1] == 9:
                            action = 3
                        if action >= 5:
                            predictions, strata = forecast(history, belief, coordinate, action - 5)
                        key, step_key = jax.random.split(key)
                        obs, state, _, done, _ = env.step(step_key, state, action, params)
                        obs, done = np.asarray(obs), bool(done)
                        receipt["transitions"] += 1
                        traces.write(json.dumps({"map_seed": map_seed, "reset_seed": reset_seed, "step": step,
                                                 "action": action, "observation": obs.tolist(), "done": done,
                                                 "pending_forecast": predictions if action >= 5 else None}) + "\n")
                        traces.flush()
                        if done:
                            raise ValueError("unexpected boundary in fixed exploration fragment")
                        if action >= 5:
                            record = {"map_seed": map_seed, "reset_seed": reset_seed, "step": step,
                                      "rock": action - 5, **strata, "predictions": predictions,
                                      "positive": bool(obs[22 + action - 5] > 0),
                                      "losses": losses(predictions, obs[22 + action - 5] > 0)}
                            records.append(record)
                            scored.write(json.dumps(record, allow_nan=False) + "\n")
                        belief.update(coordinate, action, obs, done)
                        history.append((coordinate, action, obs, done))
                        if step % 32 == 0:
                            scored.flush()
                            check_resources()
                    receipt["completed_fragments"] += 1
                    print(json.dumps({"fragments": receipt["completed_fragments"], "checks": len(records)}), flush=True)
        if receipt["transitions"] != len(MAP_SEEDS) * len(RESET_SEEDS) * STEPS:
            raise ValueError("incomplete fixed collection")
        summary = summarize(records)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        authenticate_source()
        for name, pin in receipt["source_hashes"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != pin:
                raise ValueError(f"diagnostic source changed: {name}")
        if hashlib.sha256(supervision.read_bytes()).hexdigest() != receipt["supervision_sha256"]:
            raise ValueError("supervisor launch changed")
        check_resources()
        receipt["native_elapsed_seconds"] = (clock.now_ns() - launch["started_ns"]) / 1e9
        receipt.update(status="completed", check_endpoints=len(records), memory_pilot_admitted=summary["memory_pilot_admitted"])
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        original = sys.exception()
        try:
            receipt["elapsed_seconds"] = time.monotonic() - started
            receipt["output_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        except BaseException as publication_error:
            if original is None:
                raise
            original.add_note(f"Receipt publication also failed: {publication_error!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.supervision)
