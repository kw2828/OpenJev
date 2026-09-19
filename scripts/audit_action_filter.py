"""Audit completed action-filter predictions without importing or running models.

The producer binds initial/final weights, but does not seal every prediction or
loss file. This auditor authenticates those bindings and seals the remaining
saved bytes at audit entry/exit. It does not numerically replay training, prove
the reported optimizer steps, or attest subprocess execution independently.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("transported_delta", "decay_delta", "gru", "history16", "diagonal_filter")
SEEDS = (101, 202, 303)
REFERENCES = ("ridge16", "hold_last")
SOURCES = {
    "scripts/train_action_filter.py", "scripts/prepare_action_filter_data.py",
    "src/openjev/research/action_filter_models.py", "tests/test_action_filter_models.py",
    "tests/test_action_filter_data.py", "tests/test_action_filter_training.py",
    "research/action-filter-protocol.md",
}
CONTINUATION = (
    "transported_delta improves >=5% over every other neural family, ridge16 and hold_last; "
    "all 3 seeds improve over paired neural fits and both references; "
    "median full-window predictor latency <=1.5x GRU"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    require(value > 0 if positive else value >= 0, f"negative/zero {name}")
    return float(value)


def members(folder):
    paths = list(folder.rglob("*"))
    require(not any(p.is_symlink() for p in paths), f"symlink in {folder}")
    return {str(p.relative_to(folder)): sha(p) for p in sorted(paths) if p.is_file()}


def check_metrics(saved, calculated):
    for key in ("mse", "horizon_mse"):
        left = np.asarray(saved[key], dtype=np.float64)
        right = np.asarray(calculated[key], dtype=np.float64)
        require(left.shape == right.shape and np.isfinite(left).all()
                and np.allclose(left, right, rtol=1e-10, atol=1e-12), f"metric mismatch: {key}")


def errors(predictions, targets, raw_ids, window_ids):
    require(predictions.shape == targets.shape == (176, 10, 9), "prediction/target shape")
    require(predictions.dtype in (np.float32, np.float64)
            and np.isfinite(predictions).all() and np.isfinite(targets).all(), "nonfinite predictions")
    squared = np.square(predictions.astype(np.float64) - targets.astype(np.float64))
    by_window_horizon = squared.mean(axis=2)
    trajectory = []
    for index, source in enumerate(raw_ids):
        mask = window_ids[:, 0] == index
        require(int(mask.sum()) == 16, "trajectory window count")
        trajectory.append({"source_id": int(source), "windows": 16,
                           "mse": float(squared[mask].mean()),
                           "horizon_mse": squared[mask].mean(axis=(0, 2)).tolist()})
    return {"mse": float(squared.mean()), "horizon_mse": squared.mean(axis=(0, 2)).tolist(),
            "per_trajectory": trajectory}, by_window_horizon


def improvement(candidate, comparator):
    # A zero baseline cannot establish a strict improvement. Do not emit NaN/inf.
    return None if comparator == 0 else 100.0 * (comparator - candidate) / comparator


def aggregate(per_fit, references):
    families = {}
    for mode in MODES:
        rows = [per_fit[f"{mode}-{seed}"] for seed in SEEDS]
        samples = np.concatenate([np.asarray(row["window_latency_ms"]["samples"]) for row in rows])
        families[mode] = {
            "fit_names": [f"{mode}-{seed}" for seed in SEEDS],
            "fit_mse": [row["mse"] for row in rows],
            "mean_mse": float(np.mean([row["mse"] for row in rows], dtype=np.float64)),
            "horizon_mse": np.mean([row["horizon_mse"] for row in rows], axis=0).tolist(),
            "pooled_window_latency_median_ms": float(np.median(samples)),
            "median_of_fit_latency_medians_ms": float(np.median([
                row["window_latency_ms"]["median"] for row in rows])),
        }
    checks, contrasts = [], {}
    candidate = families["transported_delta"]["mean_mse"]
    comparators = (*MODES[1:], *REFERENCES)
    for name in comparators:
        baseline = references[name]["mse"] if name in REFERENCES else families[name]["mean_mse"]
        passed = baseline > 0 and Decimal(str(candidate)) <= Decimal(".95") * Decimal(str(baseline))
        checks.append({"name": f"family_5percent_vs_{name}", "passed": passed,
                       "candidate_mse": candidate, "comparator_mse": baseline,
                       "rule": "candidate <= 0.95 * positive comparator"})
        contrasts[name] = {"family_improvement_percent": improvement(candidate, baseline), "paired": []}
    for seed in SEEDS:
        value = per_fit[f"transported_delta-{seed}"]["mse"]
        for name in comparators:
            baseline = references[name]["mse"] if name in REFERENCES else per_fit[f"{name}-{seed}"]["mse"]
            checks.append({"name": f"pair_{seed}_positive_vs_{name}", "passed": value < baseline,
                           "candidate_mse": value, "comparator_mse": baseline,
                           "rule": "candidate < comparator"})
            contrasts[name]["paired"].append({"seed": seed, "candidate_mse": value,
                "comparator_mse": baseline, "improvement_percent": improvement(value, baseline)})
    candidate_time = families["transported_delta"]["pooled_window_latency_median_ms"]
    gru_time = families["gru"]["pooled_window_latency_median_ms"]
    checks.append({"name": "pooled_median_latency_vs_gru", "passed": bool(gru_time > 0 and
                   Decimal(str(candidate_time)) <= Decimal("1.5") * Decimal(str(gru_time))),
                   "candidate_ms": candidate_time, "gru_ms": gru_time,
                   "rule": "median of 300 candidate samples <= 1.5 * median of 300 GRU samples"})
    require(len(checks) == 25, "criterion coverage")
    return families, contrasts, {"passed": all(row["passed"] for row in checks),
        "checks_passed": sum(row["passed"] for row in checks), "total_checks": 25, "checks": checks}


def validate_inputs(experiment, protocol_sha256):
    path = experiment / "protocol.json"
    require(sha(path) == protocol_sha256, "external protocol hash")
    protocol = read_json(path)
    required = {"schema": "action-filter-v1", "modes": list(MODES), "seeds": list(SEEDS),
                "context": 32, "horizon": 10, "windows_per_trajectory": 16, "epochs": 20,
                "batch_size": 32, "learning_rate": .003, "optimizer": "Adam",
                "clip_gradient_norm": 1., "checkpoint": "last epoch, no selection",
                "continuation": CONTINUATION}
    for key, value in required.items():
        require(protocol[key] == value, f"frozen protocol {key}")
    require(set(protocol["sources"]) == SOURCES, "source closure")
    snapshot = members(experiment / "source-snapshot")
    require(snapshot == protocol["sources"], "source snapshot closure/hash")
    require({p: sha(ROOT / p) for p in SOURCES} == snapshot, "current source hashes")
    environment = protocol["environment"]
    require(environment == {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
            "numpy": np.__version__, "platform": platform.platform(), "threads": 1}, "runtime identity")
    data = Path(protocol["data_path"])
    require(members(data) == protocol["data_hashes"], "prepared data membership/hash")
    done, manifest = read_json(data / "completed.json"), read_json(data / "manifest.json")
    require(done["status"] == "completed" and done["new_model_calls"] == done["new_native_calls"] == 0,
            "preparation status")
    require(set(done["files"]) == set(protocol["data_hashes"]) - {"completed.json"}, "preparation members")
    for name, entry in done["files"].items():
        require(entry["sha256"] == protocol["data_hashes"][name]
                and entry["bytes"] == (data / name).stat().st_size, f"preparation file {name}")
    require(done["source_sha256"] == manifest["source_sha256"] == snapshot["scripts/prepare_action_filter_data.py"],
            "preparation source")
    require(done["input_sha256"] == manifest["source"]["sha256"], "original corpus hash")
    expected_splits = {"train": list(range(30)), "dev": list(range(41, 50)), "test": list(range(30, 41))}
    require(manifest["splits"] == expected_splits, "whole-parent splits")
    for split, ids in expected_splits.items():
        with np.load(data / f"{split}.npz", allow_pickle=False) as arrays:
            require(set(arrays.files) == {"obs", "actions", "source_ids"}, "prepared array fields")
            require(np.array_equal(arrays["source_ids"], ids), "prepared parent IDs")
    hashes = manifest["trajectory_sha256"]
    sets = [{hashes[str(i)] for i in ids} for ids in expected_splits.values()]
    require(all(not (sets[i] & sets[j]) for i in range(3) for j in range(i)), "duplicate parent hashes")
    require(manifest["timing"]["nominal_dt_seconds"] == .002, "time scale")
    return protocol, manifest, data


def audit(experiment, output, *, protocol_sha256, completed_sha256):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    try:
        run = experiment / "run-01"
        require((run / "completed.json").is_file() and not (run / "failed.json").exists(), "run not completed")
        require(sha(run / "completed.json") == completed_sha256, "external completion hash")
        protocol, manifest, data = validate_inputs(experiment, protocol_sha256)
        names = [f"{mode}-{seed}" for seed in SEEDS for mode in MODES]
        expected = {"started.json", "training-completed.json", "evaluation-inputs.npz", "ridge16.pt",
                    "ridge16-predictions.npy", "hold_last-predictions.npy", "completed.json"}
        expected.update(name + suffix for name in names for suffix in
                        ("-initial.pt", ".pt", "-losses.npy", "-fit.json", "-predictions.npy", "-evaluation.json"))
        before = members(run)
        require(set(before) == expected and len(before) == 97, "exact 97-file execution closure")
        done = read_json(run / "completed.json")
        training = read_json(run / "training-completed.json")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "started binding")
        require(done["protocol_sha256"] == protocol_sha256 and training["test_accessed"] is False,
                "training/evaluation boundary declaration")
        require(done["counts"] == {"train_windows": 480, "dev_windows": 144, "test_windows": 176,
                                  "neural_fits": 15}, "coverage")
        require(len(done["fits"]) == len(training["fits"]) == 15, "all fits")
        starts = np.linspace(0, 1708, 16, dtype=int)
        with np.load(run / "evaluation-inputs.npz", allow_pickle=False) as arrays:
            require(set(arrays.files) == {"target", "window_ids", "raw_trajectory_ids",
                                         "train_window_ids", "dev_window_ids"}, "evaluation fields")
            target = arrays["target"]
            ids = arrays["window_ids"]
            raw_ids = arrays["raw_trajectory_ids"]
            require(target.dtype == np.float32 and target.shape == (176, 10, 9), "target shape/dtype")
            require(np.array_equal(raw_ids, manifest["splits"]["test"]), "test parent IDs")
            for key, count in (("window_ids", 11), ("train_window_ids", 30), ("dev_window_ids", 9)):
                require(np.array_equal(arrays[key], [(i, s) for i in range(count) for s in starts]), key)
        with np.load(data / "test.npz", allow_pickle=False) as arrays:
            observations = arrays["obs"]
            expected_target = np.stack([observations[i, s + 32:s + 42] for i, s in ids])
            root = np.stack([observations[i, s + 31] for i, s in ids])
        require(np.array_equal(target, expected_target), "target values not exact prepared targets")
        per_fit, window_errors = {}, {}
        for index, name in enumerate(names):
            fit = read_json(run / f"{name}-fit.json")
            evaluation = read_json(run / f"{name}-evaluation.json")
            require(fit == training["fits"][index], "training receipt order/identity")
            require(evaluation == done["fits"][index], "evaluation completion identity")
            require({k: v for k, v in evaluation.items() if k not in ("test", "window_latency_ms")} == fit,
                    "fit changed across evaluation")
            require(f"{fit['mode']}-{fit['seed']}" == name and fit["updates"] == 300, "fit identity/update count")
            require(fit["context_frames"] == (16 if fit["mode"] == "history16" else 32), "context count")
            require(fit["state_floats"] == (128 if fit["mode"] == "diagonal_filter" else 64), "state count")
            require(type(fit["parameters"]) is int and fit["parameters"] > 0, "parameter count")
            finite(fit["train_seconds"], "fit wall time", positive=True)
            require(fit["checkpoint_sha256"] == before[f"{name}.pt"]
                    and fit["initial_sha256"] == before[f"{name}-initial.pt"], "weight file bindings")
            losses = np.load(run / f"{name}-losses.npy", allow_pickle=False)
            require(losses.shape == (300,) and np.isfinite(losses).all() and (losses >= 0).all(), "loss records")
            predictions = np.load(run / f"{name}-predictions.npy", allow_pickle=False)
            require(predictions.dtype == np.float32, "neural prediction dtype")
            calculated, window_errors[name] = errors(predictions, target, raw_ids, ids)
            check_metrics(evaluation["test"], calculated)
            timer = evaluation["window_latency_ms"]
            times = np.asarray(timer["samples"], dtype=np.float64)
            require(times.shape == (100,) and np.isfinite(times).all() and (times > 0).all(), "latency samples")
            require(math.isclose(timer["median"], float(np.median(times)), rel_tol=1e-12)
                    and math.isclose(timer["p95"], float(np.percentile(times, 95)), rel_tol=1e-12), "latency reductions")
            per_fit[name] = {**calculated, "mode": fit["mode"], "seed": fit["seed"],
                            "updates": 300, "parameters": fit["parameters"], "state_floats": fit["state_floats"],
                            "train_seconds": fit["train_seconds"], "checkpoint_sha256": fit["checkpoint_sha256"],
                            "initial_sha256": fit["initial_sha256"], "window_latency_ms": timer,
                            "epoch_training_mse": losses.reshape(20, 15).mean(axis=1).tolist()}
        references = {}
        require(set(done["references"]) == set(REFERENCES), "all references")
        for name in REFERENCES:
            predictions = np.load(run / f"{name}-predictions.npy", allow_pickle=False)
            if name == "hold_last":
                require(np.array_equal(predictions, np.repeat(root[:, None], 10, axis=1)), "hold-last reference")
            references[name], window_errors[name] = errors(predictions, target, raw_ids, ids)
            check_metrics(done["references"][name], references[name])
        families, contrasts, gate = aggregate(per_fit, references)
        for mode in MODES:
            families[mode]["per_trajectory"] = [{"source_id": int(raw), "windows_per_fit": 16,
                "mean_mse": float(np.mean([per_fit[f"{mode}-{s}"]["per_trajectory"][i]["mse"] for s in SEEDS]))}
                for i, raw in enumerate(raw_ids)]
        execution_seconds = finite(done["wall_seconds"], "execution wall", positive=True)
        train_seconds = sum(row["train_seconds"] for row in per_fit.values())
        ridge_seconds = finite(done["references"]["ridge16"]["fit_and_batch_evaluate_seconds"], "ridge time")
        timing_seconds = sum(sum(row["window_latency_ms"]["samples"]) for row in per_fit.values()) / 1000
        require(train_seconds + ridge_seconds + timing_seconds <= execution_seconds + 1e-6, "nested cost bounds")
        summary = {"status": "completed", "study": "action-filter-v1", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "counts": {**done["counts"],
                "seeds": 3, "recorded_updates_per_fit": 300, "recorded_updates_total": 4500,
                "prediction_arrays": 17, "squared_error_coordinates": 17 * 176 * 10 * 9,
                "independent_test_parent_ids": raw_ids.tolist()},
            "per_fit": per_fit, "families": families, "references": references,
            "contrasts": contrasts, "continuation_gate": gate,
            "costs": {"execution_wall_seconds": execution_seconds, "neural_training_seconds": train_seconds,
                "ridge_fit_and_batch_evaluation_seconds": ridge_seconds, "recorded_timing_calls_seconds": timing_seconds,
                "scope": "Nested measured components, not additive to execution wall; training timers omit subsequent dev evaluation and checkpoint writes."},
            "limits": ["Exploratory offline conditional forecasting: 10 steps are nominally 20 ms.",
                "11 test parent trajectories, not 176 independent replicates; no terrain-disjoint or shifted-overlap proof.",
                "Applied torques condition forecasts; this does not establish issued-action causality or policy utility.",
                "No model, checkpoint deserialization, native, RNG, gradient or optimizer calls by this audit.",
                "300 finite loss records and reported updates checked; numerical optimizer updates and weights are not replayed.",
                "Initial/final weights are producer-hash-bound; remaining payload hashes are newly sealed by this audit.",
                "Test-after-training order is source/receipt-bound, not independently process-attested.",
                "Ridge fitting and neural inference are not replayed; saved predictions and hold-last arithmetic are checked.",
                "Gate latency uses the median of all 300 saved samples per family; warmups and runtime overhead remain outside those samples.",
                "No architecture novelty, biological mechanism, uncertainty calibration or real-robot claim."]}
        write_json(output / "summary.json", summary)
        with (output / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before, "execution changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "execution_members": before,
            "source_hashes": protocol["sources"], "data_hashes": protocol["data_hashes"],
            "auditor_sha256": sha(__file__), "files": members(output),
            "qualification_passed": gate["passed"], "checks_passed": gate["checks_passed"], "total_checks": 25,
            "new_model_calls": 0, "new_native_calls": 0, "new_optimizer_calls": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(output / "failed.json", {"status": "failed", "error": repr(error),
                "wall_seconds": time.perf_counter() - start, "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            error.add_note(f"Could not preserve failure receipt: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.experiment, args.output, protocol_sha256=args.protocol_sha256,
                           completed_sha256=args.completed_sha256), allow_nan=False))
