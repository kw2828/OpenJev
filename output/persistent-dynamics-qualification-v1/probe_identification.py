"""One seed410 engineering diagnostic, not a scientific control benchmark."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from openjev.research.reacher_tracking_dynamics import TrackingDynamicsEpisode, nominal_model
from openjev.research.reacher_tracking_identification import WindowedGainIdentifier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/persistent-dynamics-qualification-v1/identification-engineering-01"
HORIZON = 80
SOURCES = [Path(__file__).relative_to(ROOT).as_posix(),
    "src/openjev/research/reacher_tracking_dynamics.py",
    "src/openjev/research/reacher_tracking_identification.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    # Conditions and admission rule are fixed before native reset or inference.
    cases = [("low", .7, .7, False), ("high", 1.3, 1.3, False),
             ("low-to-high", .7, 1.3, False), ("high-to-low", 1.3, .7, False),
             ("zero-low", .7, .7, True), ("zero-high", 1.3, 1.3, True)]
    protocol = {"engineering_only": True, "seed": 410, "horizon": HORIZON,
        "cases": cases, "change_step": 40, "window": 20,
        "gain_grid": np.linspace(.5, 1.5, 21).tolist(), "prior_gain": 1.,
        "pulse_blocks": [[.015, 0.], [0., -.015], [-.015, 0.], [0., .015]],
        "block_steps": 10, "repetitions": 2, "noise": "exactly zero", "sensing": "full",
        "targets": [[0, .12, -.04], [20, .08, .10], [60, -.08, -.10]],
        "positive_check": "mean absolute gain error <=0.1 over final10 completed transitions, each of4 pulse cases",
        "negative_check": "both zero cases retain gain1 and exactly flat banks for all80 updates; trajectories identical",
        "native_qualification": False, "learned_models": 0, "scientific_seed_allocation": False,
        "cap_seconds": 30, "automatic_retry": False,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES}}
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "started.json", protocol)
    begin, rows, outputs = time.perf_counter(), [], {}
    try:
        targets = np.empty((HORIZON+1, 2))
        targets[:20], targets[20:60], targets[60:] = [.12, -.04], [.08, .10], [-.08, -.10]
        pulses = np.tile(np.repeat(np.array(protocol["pulse_blocks"]), 10, axis=0), (2, 1))
        for name, before, after, zero in cases:
            gains = np.r_[np.full(40, before), np.full(40, after)]
            commands = np.zeros_like(pulses) if zero else pulses
            episode = TrackingDynamicsEpisode(horizon=HORIZON, target_path=targets,
                gear_multiplier=gains, sensor_schedule=np.ones(HORIZON+1, dtype=bool),
                noise=np.zeros((HORIZON, 2)))
            estimates, velocities, traces, predicted = [], [], [], []
            try:
                initial = episode.reset(410)
                model = WindowedGainIdentifier(nominal_model(), initial, gains=np.array(protocol["gain_grid"]),
                    window=20, prior_gain=1., frame_skip=2)
                for command in commands:
                    if time.perf_counter()-begin > protocol["cap_seconds"]:
                        raise TimeoutError("Engineering diagnostic cap exceeded")
                    observed = episode.step(command)
                    _, velocity, estimate = model.update(command, observed)
                    trace = model.snapshot()["last_trace"]
                    estimates.append(estimate)
                    velocities.append(velocity[:2])
                    traces.append(trace["window_residual_sums"])
                    predicted.append(trace["candidate_predictions"])
                # Privileged values are read only after the public estimator finishes.
                record = episode.episode_record()
            finally:
                episode.close()
            states = record["audit"]["decision_states"]
            true_qpos = np.stack([state["qpos"] for state in states])
            true_qvel = np.stack([state["qvel"] for state in states])
            errors = np.abs(np.array(estimates)-gains)
            flat = np.all(np.array(traces) == np.array(traces)[:, :1], axis=1)
            passed = (bool(np.all(np.array(estimates)==1.) and np.all(flat)) if zero
                      else bool(errors[-10:].mean() <= .1))
            filename = name + ".npz"
            with (OUT / filename).open("xb") as stream:
                np.savez_compressed(stream, packets=record["policy"]["packets"],
                    commands=record["policy"]["commands"], gains=gains,
                    estimates=np.array(estimates), estimated_velocity=np.array(velocities),
                    true_qpos=true_qpos, true_qvel=true_qvel,
                    residual_sums=np.array(traces), predictions=np.array(predicted))
            outputs[name] = true_qpos
            rows.append({"name": name, "zero_commands": zero,
                "last10_mean_absolute_gain_error": float(errors[-10:].mean()),
                "final_gain": estimates[-1], "flat_bank_count": int(flat.sum()),
                "last10_mean_absolute_velocity_error": float(np.abs(np.array(velocities)-true_qvel[1:,:2])[-10:].mean()),
                "engineering_check_passed": passed, "native_decisions": HORIZON,
                "identifier_costs": model.snapshot()["costs"],
                "file": filename, "sha256": sha(OUT / filename)})
        zero_equal = bool(np.array_equal(outputs["zero-low"], outputs["zero-high"]))
        write(OUT / "completed.json", {"status": "completed", "engineering_only": True,
            "started_sha256": sha(OUT / "started.json"), "rows": rows,
            "zero_gain_trajectories_equal": zero_equal,
            "engineering_checks_passed": all(row["engineering_check_passed"] for row in rows) and zero_equal,
            "native_decisions": 480, "identifier_native_transitions": 10080,
            "wall_seconds": time.perf_counter()-begin,
            "limits": "One repeated seed410 initial condition, deterministic pulses, no noise, no planning or learned models. Does not qualify closed-loop adaptation or benchmark effectiveness."})
    except BaseException as error:
        write(OUT / "failed.json", {"status": "failed", "error": repr(error), "completed_cases": rows,
                                   "wall_seconds": time.perf_counter()-begin, "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
