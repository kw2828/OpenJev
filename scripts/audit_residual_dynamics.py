"""Authenticate completed residual forecasts and audit saved errors, without models.

No training, optimizer, model inference, native simulator or random generator is
invoked. Weight files are authenticated bytes, not executable checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
import zipfile
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (411, 512, 613)
VARIANTS = ("none", "bias", "public", "latent", "history16")
PANELS = ("test_sin", "test_zigzag")
REFERENCES = ("ridge16", "hold_last")
SOURCES = {
    "scripts/train_residual_dynamics.py", "scripts/prepare_residual_dynamics_data.py",
    "src/openjev/research/residual_dynamics.py", "tests/test_residual_dynamics.py",
    "tests/test_residual_dynamics_data.py", "tests/test_residual_dynamics_training.py",
    "research/residual-dynamics-protocol.md",
}
GATE = ("latent RLS >=10% mean improvement against each other variant, ridge and hold on BOTH panels; "
        "all 3 fits improve against paired variants and references on BOTH panels; "
        "pooled median full-window latency <=2x same-backbone no-adaptation (49 checks)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def members(folder):
    paths = sorted(folder.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "symlink in authenticated tree")
    return {str(p.relative_to(folder)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in paths if p.is_file()}


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    require(value > 0 if positive else value >= 0, f"negative/zero {name}")
    return float(value)


def weight_payload(path):
    """Compare paired initialization storage/metadata without unpickling anything."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate weight archive member")
        require(all("/" in name for name in names), "weight archive prefix")
        require(len({name.split("/", 1)[0] for name in names}) == 1, "weight archive roots")
        return {name.split("/", 1)[1]: hashlib.sha256(archive.read(name)).hexdigest()
                for name in names if not name.endswith("/.data/serialization_id")}


def check_metrics(saved, calculated):
    for key in ("mse", "horizon_mse"):
        a, b = np.asarray(saved[key], dtype=np.float64), np.asarray(calculated[key], dtype=np.float64)
        require(a.shape == b.shape and np.isfinite(a).all()
                and np.allclose(a, b, rtol=1e-10, atol=1e-12), f"saved {key} mismatch")


def errors(pred, target, ids, scales):
    require(pred.shape == target.shape == (160, 25, 9), "forecast shape")
    require(pred.dtype in (np.dtype("float32"), np.dtype("float64"))
            and np.isfinite(pred).all() and np.isfinite(target).all(), "invalid forecast")
    squared = np.square(pred.astype(np.float64) - target.astype(np.float64))
    per_trajectory = [{"source_id": i, "windows": int((ids == i).sum()),
                       "mse": float(squared[ids == i].mean()),
                       "horizon_mse": squared[ids == i].mean(axis=(0, 2)).tolist()}
                      for i in range(10)]
    require(all(row["windows"] == 16 for row in per_trajectory), "trajectory coverage")
    channels = squared.mean(axis=(0, 1))
    return {"mse": float(squared.mean()), "horizon_mse": squared.mean(axis=(0, 2)).tolist(),
            "per_trajectory": per_trajectory,
            "posthoc_channel_mse_normalized": channels.tolist(),
            "posthoc_channel_mse_source_coordinates": (channels * np.square(scales)).tolist(),
            "posthoc_position_rmse_source_units": float(np.sqrt(np.sum(channels[:3] * np.square(scales[:3]))))}, squared.mean(axis=2)


def improvement(candidate, comparator):
    return None if comparator == 0 else 100.0 * (comparator - candidate) / comparator


def aggregate(rows, references):
    families, contrasts, checks, latency = {}, {}, [], {}
    for variant in VARIANTS:
        samples = np.asarray([x for panel in PANELS for seed in SEEDS
                              for x in rows[panel][f"{variant}-{seed}"]["window_latency_ms"]["samples"]])
        require(samples.shape == (300,), "pooled timing coverage")
        latency[variant] = {"samples": 300, "median_ms": float(np.median(samples)),
                            "p95_ms": float(np.percentile(samples, 95))}
    comparators = (*[v for v in VARIANTS if v != "latent"], *REFERENCES)
    for panel in PANELS:
        families[panel], contrasts[panel] = {}, {}
        for variant in VARIANTS:
            fits = [rows[panel][f"{variant}-{seed}"] for seed in SEEDS]
            families[panel][variant] = {
                "fit_names": [f"{variant}-{seed}" for seed in SEEDS],
                "fit_mse": [f["mse"] for f in fits],
                "mean_mse": float(np.mean([f["mse"] for f in fits])),
                "horizon_mse": np.mean([f["horizon_mse"] for f in fits], axis=0).tolist(),
                "posthoc_channel_mse_normalized": np.mean([f["posthoc_channel_mse_normalized"] for f in fits], axis=0).tolist(),
                "posthoc_channel_mse_source_coordinates": np.mean([f["posthoc_channel_mse_source_coordinates"] for f in fits], axis=0).tolist(),
                "posthoc_position_rmse_source_units": float(np.sqrt(np.sum(np.mean([
                    f["posthoc_channel_mse_source_coordinates"][:3] for f in fits], axis=0)))),
                "per_trajectory": [{"source_id": i, "windows_per_fit": 16,
                    "mean_mse": float(np.mean([f["per_trajectory"][i]["mse"] for f in fits]))}
                    for i in range(10)]}
        candidate = families[panel]["latent"]["mean_mse"]
        for other in comparators:
            baseline = (references[panel][other]["mse"] if other in REFERENCES
                        else families[panel][other]["mean_mse"])
            passed = baseline > 0 and Decimal(str(candidate)) <= Decimal(".90") * Decimal(str(baseline))
            checks.append({"name": f"{panel}/family_10percent_vs_{other}", "passed": passed,
                           "candidate_mse": candidate, "comparator_mse": baseline,
                           "rule": "candidate <= 0.90 * positive comparator"})
            contrast = {"family_improvement_percent": improvement(candidate, baseline), "paired": []}
            for seed in SEEDS:
                value = rows[panel][f"latent-{seed}"]["mse"]
                comparison = (references[panel][other]["mse"] if other in REFERENCES
                              else rows[panel][f"{other}-{seed}"]["mse"])
                checks.append({"name": f"{panel}/pair_{seed}_positive_vs_{other}", "passed": value < comparison,
                               "candidate_mse": value, "comparator_mse": comparison,
                               "rule": "candidate < comparator"})
                contrast["paired"].append({"seed": seed, "candidate_mse": value,
                    "comparator_mse": comparison, "improvement_percent": improvement(value, comparison)})
            contrasts[panel][other] = contrast
    value, comparison = latency["latent"]["median_ms"], latency["none"]["median_ms"]
    checks.append({"name": "pooled_median_latency_vs_none", "passed": comparison > 0 and
                   Decimal(str(value)) <= Decimal(2) * Decimal(str(comparison)),
                   "candidate_ms": value, "comparator_ms": comparison,
                   "rule": "median of 300 latent samples <= 2 * median of 300 no-adaptation samples"})
    require(len(checks) == 49, "criterion count")
    return families, contrasts, latency, {"passed": all(c["passed"] for c in checks),
        "checks_passed": sum(c["passed"] for c in checks), "total_checks": 49, "checks": checks}


def validate_inputs(experiment, expected):
    require(sha(experiment / "protocol.json") == expected, "external protocol hash")
    protocol = read_json(experiment / "protocol.json")
    fields = {"study": "residual-dynamics-v1", "context": 32, "horizon": 25, "epochs": 30,
        "batch": 32, "seeds": list(SEEDS), "variants": list(VARIANTS), "panels": list(PANELS),
        "learning_rate": .001, "optimizer": "Adam", "gradient_norm_cap": 1.,
        "checkpoint_selection": "final epoch, no dev/test selection", "rls_prior_precision": 1.,
        "rls_forgetting": 1., "neural_fits": 6, "per_fit_updates": 690, "gate": GATE}
    require(all(protocol[k] == v for k, v in fields.items()), "protocol settings")
    require(set(protocol["sources"]) == SOURCES, "seven-source closure")
    snapshot = {k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}
    require(snapshot == protocol["sources"] == {p: sha(ROOT / p) for p in SOURCES}, "source identities")
    require(protocol["runtime"] == {"python": platform.python_version(),
        "torch": importlib.metadata.version("torch"), "numpy": np.__version__,
        "platform": platform.platform(), "threads": 1}, "runtime identity")
    data = Path(protocol["data_path"])
    bound = members(data)
    require({k: v["sha256"] for k, v in bound.items()} == protocol["data_hashes"], "data hashes")
    require(set(bound) == {"completed.json", "manifest.json", "normalization.npz",
                           "train.npz", "dev.npz", "test_sin.npz", "test_zigzag.npz"}, "data closure")
    done, manifest = read_json(data / "completed.json"), read_json(data / "manifest.json")
    require(done["status"] == "completed" and done["new_model_calls"] == done["new_native_calls"]
            == done["new_random_draws"] == 0, "preparation status")
    require(done["files"] == {k: v for k, v in bound.items() if k != "completed.json"}, "preparation seal")
    require(done["source_sha256"] == manifest["source_sha256"]
            == snapshot["scripts/prepare_residual_dynamics_data.py"], "preparer identity")
    require(done["input_sha256"] == {k: v["sha256"] for k, v in manifest["sources"].items()}, "archive bindings")
    require(len(set(done["input_sha256"].values())) == 3, "distinct source archives")
    specifications = {"train": ("old", list(range(30)), list(range(0, 1151, 50))),
                      "dev": ("old", list(range(41, 50)), [0, 594, 1189]),
                      **{panel: (panel.removeprefix("test_"), list(range(10)),
                                  np.linspace(0, 9439, 16, dtype=int).tolist()) for panel in PANELS}}
    for split, (archive, ids, starts) in specifications.items():
        require(manifest["splits"][split] == {"archive": archive, "source_ids": ids, "starts": starts}, "split identity")
        with np.load(data / f"{split}.npz", allow_pickle=False) as arrays:
            require(set(arrays.files) == {"obs", "actions", "source_ids", "window_starts"}, "data fields")
            require(np.array_equal(arrays["source_ids"], np.repeat(ids, len(starts)))
                    and np.array_equal(arrays["window_starts"], np.tile(starts, len(ids))), "window provenance")
            require(arrays["obs"].shape == (len(ids) * len(starts), 57, 9)
                    and arrays["actions"].shape == (len(ids) * len(starts), 56, 40), "data shapes")
            require(arrays["obs"].dtype == arrays["actions"].dtype == np.float32
                    and np.isfinite(arrays["obs"]).all() and np.isfinite(arrays["actions"]).all(), "data finiteness")
    require(manifest["excluded_old_ids"] == list(range(30, 41)) and manifest["raw_stride"] == 10
            and manifest["nominal_forecast_seconds"] == .5, "excluded IDs or time scale")
    return protocol, manifest, data


def audit(experiment, output, *, protocol_sha256, completed_sha256):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    try:
        run = experiment / "run-01"
        require(sha(run / "completed.json") == completed_sha256, "external completion hash")
        protocol, manifest, data = validate_inputs(experiment, protocol_sha256)
        fits = [f"{variant}-{seed}" for seed in SEEDS for variant in ("none", "history16")]
        names = [f"{panel}-{variant}-{seed}" for panel in PANELS for seed in SEEDS for variant in VARIANTS]
        expected = {"started.json", "training-completed.json", "ridge16.pt", "completed.json"}
        expected.update(f + suffix for f in fits for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
        expected.update(n + suffix for n in names for suffix in ("-predictions.npy", "-evaluation.json"))
        expected.update(f"{panel}-targets.npy" for panel in PANELS)
        expected.update(f"{panel}-{ref}-predictions.npy" for panel in PANELS for ref in REFERENCES)
        before = members(run)
        require(set(before) == expected and len(before) == 94, "exact 94-file completed execution")
        done, training = read_json(run / "completed.json"), read_json(run / "training-completed.json")
        require(done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256, "completion identity")
        require(done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "producer payload seal")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "start identity")
        require(training["test_arrays_opened"] is False and training["fits"] == done["fits"]
                and len(done["fits"]) == 6, "training boundary")
        fit_records = {}
        for index, name in enumerate(fits):
            fit = read_json(run / f"{name}-fit.json")
            require(fit == done["fits"][index] and f"{fit['variant']}-{fit['seed']}" == name, "fit identity/order")
            require(fit["updates"] == 690 and fit["parameters"] == 39369
                    and fit["state_elements"] == {"obs": 9, "previous_obs": 9, "delta": 9,
                                                  "hidden": 64, "has_obs": 1}, "fit counts")
            require(fit["initial_sha256"] == before[f"{name}-initial.pt"]["sha256"]
                    and fit["checkpoint_sha256"] == before[f"{name}.pt"]["sha256"], "checkpoint bindings")
            losses = np.load(run / f"{name}-losses.npy", allow_pickle=False)
            require(losses.shape == (690,) and np.isfinite(losses).all() and (losses >= 0).all(), "loss records")
            finite(fit["train_seconds"], "fit seconds", positive=True)
            finite(fit["development"]["mse"], "declared development MSE")
            dev_h = np.asarray(fit["development"]["horizon_mse"])
            require(dev_h.shape == (25,) and np.isfinite(dev_h).all() and (dev_h >= 0).all(), "development horizons")
            fit_records[name] = {**fit, "loss_records": 690}
        for seed in SEEDS:
            require(weight_payload(run / f"none-{seed}-initial.pt")
                    == weight_payload(run / f"history16-{seed}-initial.pt"), "paired initial storage differs")
        require(len(done["evaluations"]) == 30 and set(done["references"]) == set(PANELS), "row coverage")
        with np.load(data / "normalization.npz", allow_pickle=False) as arrays:
            scales, channel_std = arrays["obs_scale"], arrays["obs_std"]
            require(scales.shape == channel_std.shape == (9,) and np.isfinite(scales).all()
                    and np.isfinite(channel_std).all() and (scales > 0).all() and (channel_std >= 0).all(), "normalization channels")
            require(np.array_equal(scales, np.where(channel_std == 0, 1., channel_std)), "normalization scale rule")
        rows, references, window_errors = {}, {}, {}
        for panel in PANELS:
            with np.load(data / f"{panel}.npz", allow_pickle=False) as arrays:
                observations, ids = arrays["obs"], arrays["source_ids"]
            target = np.load(run / f"{panel}-targets.npy", allow_pickle=False)
            require(target.dtype == np.float32 and np.array_equal(target, observations[:, 32:]), "bound target identity")
            rows[panel], references[panel] = {}, {}
            for seed in SEEDS:
                for variant in VARIANTS:
                    name, prefix = f"{variant}-{seed}", f"{panel}-{variant}-{seed}"
                    row = read_json(run / f"{prefix}-evaluation.json")
                    require(row == done["evaluations"][names.index(prefix)] and row["panel"] == panel
                            and row["variant"] == variant and row["seed"] == seed, "evaluation identity/order")
                    backbone = f"{'history16' if variant == 'history16' else 'none'}-{seed}"
                    require(row["backbone_sha256"] == fit_records[backbone]["checkpoint_sha256"], "shared backbone")
                    pred = np.load(run / f"{prefix}-predictions.npy", allow_pickle=False)
                    require(pred.dtype == np.float32, "neural prediction precision")
                    calculated, window_errors[prefix] = errors(pred, target, ids, scales)
                    check_metrics(row["test"], calculated)
                    timer = row["window_latency_ms"]
                    samples = np.asarray(timer["samples"], dtype=np.float64)
                    require(samples.shape == (50,) and np.isfinite(samples).all() and (samples > 0).all(), "timing samples")
                    require(math.isclose(timer["median"], float(np.median(samples)), rel_tol=1e-12)
                            and math.isclose(timer["p95"], float(np.percentile(samples, 95)), rel_tol=1e-12), "timing reductions")
                    rows[panel][name] = {**calculated, "variant": variant, "seed": seed,
                        "backbone_sha256": row["backbone_sha256"], "window_latency_ms": timer}
            require(set(done["references"][panel]) == set(REFERENCES), "reference coverage")
            for ref in REFERENCES:
                prefix = f"{panel}-{ref}"
                pred = np.load(run / f"{prefix}-predictions.npy", allow_pickle=False)
                if ref == "hold_last":
                    require(np.array_equal(pred, np.repeat(observations[:, 31:32], 25, axis=1)), "hold-last values")
                calculated, window_errors[prefix] = errors(pred, target, ids, scales)
                check_metrics(done["references"][panel][ref], calculated)
                references[panel][ref] = calculated
        families, contrasts, latency, gate = aggregate(rows, references)
        wall = finite(done["wall_seconds"], "execution wall", positive=True)
        train_seconds = sum(f["train_seconds"] for f in fit_records.values())
        ridge_seconds = finite(done["ridge_fit_seconds"], "ridge fit time", positive=True)
        require(training["ridge_fit_seconds"] == ridge_seconds, "ridge boundary time")
        timing_seconds = sum(sum(r["window_latency_ms"]["samples"]) for p in rows.values() for r in p.values()) / 1000
        require(train_seconds + ridge_seconds + timing_seconds <= wall + 1e-6, "nested cost bounds")
        summary = {"status": "completed", "study": "residual-dynamics-v1", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "fits": fit_records, "per_fit": rows,
            "families": families, "references": references, "contrasts": contrasts, "latency": latency,
            "posthoc_channel_diagnostic": {"channel_names": manifest["feature_order"],
                "training_channel_std": channel_std.tolist(), "normalization_scale": scales.tolist(),
                "scope": "Post hoc descriptive decomposition only, excluded from the gate. Source-coordinate MSE equals normalized MSE times training scale squared. Position RMSE is sqrt(sum of XYZ coordinate MSE), pooled over windows/horizons and, for families, all three fits before square root. XYZ use source position units; sine/cosine errors are dimensionless, not angular error. No combined physical-unit primary metric."},
            "continuation_gate": gate, "counts": {"sources": 7, "execution_files": 94,
                "neural_fits": 6, "recorded_updates_per_fit": 690, "recorded_updates_total": 4140,
                "neural_prediction_rows": 30, "reference_rows": 4, "train_windows": 720, "dev_windows": 27,
                "test_windows_per_panel": 160, "parent_trajectories_per_panel": 10,
                "squared_error_coordinates": 34 * 160 * 25 * 9, "timing_samples_per_variant": 300},
            "costs": {"execution_wall_seconds": wall, "neural_training_seconds": train_seconds,
                "ridge_fit_seconds": ridge_seconds, "recorded_timing_calls_seconds": timing_seconds,
                "scope": "Nested components, not additional costs. Execution receipt wall excludes its final payload hashing and receipt write. Training excludes dev inference and checkpoint writes."},
            "limits": ["Offline conditional forecasts given recorded future applied torques, not issued-action or control utility evidence.",
                "Nominal forecast span 500 ms; windows share 10 parent trajectories per panel, not 160 independent replicates.",
                "Distinct archives and declared prior identity inspection do not establish independent seeds, terrain or regimes.",
                "No model/optimizer/native/RNG calls or checkpoint unpickling; no gradient, RLS, ridge or neural inference replay.",
                "Recorded losses and producer-reported updates are checked; optimizer operations and batch permutations are source-bound, not replayed.",
                "Paired initial archive storage and all metadata bytes match after removing archive prefix and serialization ID.",
                "Development metrics have no saved predictions and are validated for finiteness only, not independently recomputed.",
                "Train-before-test order is source/receipt-bound, not independently process-attested.",
                "Gate latency pools all 300 measured complete-window calls per variant; it is not streaming latency or matched memory.",
                "No novelty, biological mechanism, calibrated uncertainty, source-paper reproduction or real-robot claim."],
            "data_provenance": {"splits": manifest["splits"], "input_sha256": {k: v["sha256"] for k, v in manifest["sources"].items()}}}
        write_json(output / "summary.json", summary)
        with (output / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before, "execution changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "execution_members": before,
            "source_hashes": protocol["sources"], "data_hashes": protocol["data_hashes"],
            "auditor_sha256": sha(__file__), "files": members(output), "qualification_passed": gate["passed"],
            "checks_passed": gate["checks_passed"], "total_checks": 49, "new_model_calls": 0,
            "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(output / "failed.json", {"status": "failed", "error": repr(error),
                "auditor_sha256": sha(__file__), "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the actual audit failure
            error.add_note(f"Could not preserve failure receipt: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.output, protocol_sha256=args.protocol_sha256,
                   completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "checks_passed", "total_checks", "wall_seconds")}))
