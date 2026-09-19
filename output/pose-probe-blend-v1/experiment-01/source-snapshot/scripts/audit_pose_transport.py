"""Saved-pose arithmetic audit; no inference, fitting, optimizer or native calls.

Torch is used only for weights_only tensor deserialization. All pose errors,
family reductions and continuation criteria are independently computed in NumPy.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (701, 802, 903)
VARIANTS = ("world", "body", "transport", "history16")
REFERENCES = ("hold", "cv1", "cv16", "ls16", "body16", "ridge16")
PANELS = ("test_sin", "test_zigzag")
ENDPOINTS = ("position", "rotation")
SOURCES = {
    "scripts/train_pose_transport.py", "scripts/audit_pose_transport.py",
    "src/openjev/research/pose_transport.py", "src/openjev/research/pose_references.py",
    "src/openjev/research/rigid_motion.py", "tests/test_pose_transport.py",
    "tests/test_pose_references.py", "tests/test_rigid_motion.py",
    "tests/test_pose_transport_training.py", "tests/test_audit_pose_transport.py",
    "research/pose-transport-protocol.md",
}
GATE = ("transport at least10% better on each physical endpoint against every learned and classical control "
        "on both panels; all3 paired fit endpoints nonworse; at least8/10 parents nonworse for each "
        "endpoint/control/panel; strictly lower MSE after each leave-one-parent-out; median full-window "
        "latency<=1.5x body. Failure stays failed.")
ROTATION_TOLERANCE = 1e-4


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
    require(not folder.is_symlink() and not any(p.is_symlink() for p in paths), "symlink in evidence")
    return {str(p.relative_to(folder)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in paths if p.is_file()}


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"invalid {name}")
    return float(value)


def rotation_quality(rotation):
    require(rotation.shape[-2:] == (3, 3) and rotation.dtype in (np.float32, np.float64)
            and np.isfinite(rotation).all(), "invalid rotation shape/dtype/values")
    r = rotation.astype(np.float64)
    orth = float(np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3))))
    determinant = float(np.max(np.abs(np.linalg.det(r) - 1)))
    require(max(orth, determinant) <= ROTATION_TOLERANCE, "matrix is not a proper rotation")
    return orth, determinant


def rotation_angles(prediction, target):
    require(prediction.shape == target.shape, "rotation shape mismatch")
    rotation_quality(prediction)
    rotation_quality(target)
    relative = np.swapaxes(prediction.astype(np.float64), -1, -2) @ target.astype(np.float64)
    skew = np.stack((relative[..., 2, 1] - relative[..., 1, 2],
                     relative[..., 0, 2] - relative[..., 2, 0],
                     relative[..., 1, 0] - relative[..., 0, 1]), axis=-1) * .5
    sine = np.sqrt(np.sum(np.square(skew), axis=-1))
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) * .5, -1., 1.)
    return np.arctan2(sine, cosine)


def public_pose(raw):
    """Independent Rz(yaw) Ry(pitch) Rx(roll) conversion of unnormalized fields."""
    require(raw.shape[-1] == 9 and np.isfinite(raw).all(), "public pose fields")
    sine, cosine = raw[..., 3:6], raw[..., 6:9]
    require((np.hypot(sine, cosine) >= 1e-8).all(), "degenerate public orientation")
    roll, pitch, yaw = np.moveaxis(np.arctan2(sine, cosine), -1, 0)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    matrix = np.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                       sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                       -sp, cp * sr, cp * cr), axis=-1).reshape(*raw.shape[:-1], 3, 3)
    return raw[..., :3].astype(np.float32), matrix.astype(np.float32)


def error_arrays(p, rotation, target_p, target_r):
    require(p.shape == target_p.shape and p.shape[-1] == 3 and rotation.shape == (*p.shape[:-1], 3, 3),
            "pose shape mismatch")
    require(p.dtype in (np.float32, np.float64) and target_p.dtype in (np.float32, np.float64)
            and np.isfinite(p).all() and np.isfinite(target_p).all(), "invalid position")
    return {"position": np.square(p.astype(np.float64) - target_p.astype(np.float64)).sum(axis=-1),
            "rotation": np.square(rotation_angles(rotation, target_r))}


def reduce_errors(values, ids):
    require(set(values) == set(ENDPOINTS) and ids.ndim == 2 and ids.shape[1] == 2, "error fields/IDs")
    parents = sorted(np.unique(ids[:, 0]).tolist())
    result = {}
    for endpoint in ENDPOINTS:
        array = values[endpoint]
        require(array.ndim == 2 and len(array) == len(ids) and np.isfinite(array).all()
                and (array >= 0).all(), "invalid squared errors")
        parent_mse = [float(array[ids[:, 0] == i].mean()) for i in parents]
        result[endpoint] = {"mse": float(array.mean()), "rmse": float(np.sqrt(array.mean())),
            "horizon_mse": array.mean(axis=0).tolist(), "horizon_rmse": np.sqrt(array.mean(axis=0)).tolist(),
            "parent_ids": parents, "parent_mse": parent_mse, "parent_rmse": np.sqrt(parent_mse).tolist()}
    return result


def validate_metrics(saved, metrics, rotation):
    expected = {"position_rmse_m": metrics["position"]["rmse"],
                "rotation_rmse_rad": metrics["rotation"]["rmse"],
                "composite": (metrics["position"]["mse"] + metrics["rotation"]["mse"]) / .1 ** 2}
    require(set(saved) == set(expected) | {"rotation_orthogonality_max", "rotation_determinant_max_error"},
            "declared metric fields")
    for key, value in expected.items():
        finite(saved[key], key)
        require(math.isclose(saved[key], value, rel_tol=1e-9, abs_tol=1e-12), f"metric mismatch {key}")
    # Producer quality summaries use original float32 arithmetic, unlike error reductions.
    for key, value in zip(("rotation_orthogonality_max", "rotation_determinant_max_error"),
                          rotation_quality(rotation), strict=True):
        finite(saved[key], key)
        require(math.isclose(saved[key], value, rel_tol=.01, abs_tol=3e-7), f"quality mismatch {key}")


def pooled(rows):
    result = {}
    for endpoint in ENDPOINTS:
        fields = ("mse", "horizon_mse", "parent_mse")
        values = {k: np.mean([r[endpoint][k] for r in rows], axis=0) for k in fields}
        result[endpoint] = {"mse": float(values["mse"]), "rmse": float(np.sqrt(values["mse"])),
            "horizon_mse": values["horizon_mse"].tolist(), "horizon_rmse": np.sqrt(values["horizon_mse"]).tolist(),
            "parent_ids": rows[0][endpoint]["parent_ids"], "parent_mse": values["parent_mse"].tolist(),
            "parent_rmse": np.sqrt(values["parent_mse"]).tolist()}
    return result


def percent(candidate, comparator):
    return None if comparator == 0 else 100 * (comparator - candidate) / comparator


def aggregate(rows):
    families, references, latency, comparisons, checks = {}, {}, {}, {}, []
    for variant in (*VARIANTS, *REFERENCES):
        labels = [f"{variant}-{seed}" for seed in SEEDS] if variant in VARIANTS else [variant]
        samples = np.asarray([t for panel in PANELS for label in labels for t in rows[panel][label]["latency_ms"]])
        require(samples.shape == ((120,) if variant in VARIANTS else (40,))
                and np.isfinite(samples).all() and (samples > 0).all(), "latency coverage")
        latency[variant] = {"samples": len(samples), "median_ms": float(np.median(samples)),
                            "p95_ms": float(np.percentile(samples, 95))}
    for panel in PANELS:
        families[panel] = {v: pooled([rows[panel][f"{v}-{seed}"]["errors"] for seed in SEEDS]) for v in VARIANTS}
        references[panel] = {v: rows[panel][v]["errors"] for v in REFERENCES}
        comparisons[panel] = {}
        for control in (*[v for v in VARIANTS if v != "transport"], *REFERENCES):
            comparisons[panel][control] = {}
            for endpoint in ENDPOINTS:
                candidate = families[panel]["transport"][endpoint]
                comparator = (families[panel][control] if control in VARIANTS else references[panel][control])[endpoint]
                base = f"{panel}/{control}/{endpoint}"
                checks.append({"name": base + "/family_10percent", "passed": comparator["rmse"] > 0 and
                    Decimal(str(candidate["rmse"])) <= Decimal(".90") * Decimal(str(comparator["rmse"])),
                    "candidate": candidate["rmse"], "comparator": comparator["rmse"], "rule": "RMSE <=0.90*positive control"})
                paired = []
                for seed in SEEDS:
                    c = rows[panel][f"transport-{seed}"]["errors"][endpoint]["mse"]
                    b = rows[panel][f"{control}-{seed}" if control in VARIANTS else control]["errors"][endpoint]["mse"]
                    checks.append({"name": base + f"/pair_{seed}_nonworse", "passed": c <= b,
                                   "candidate": c, "comparator": b, "rule": "MSE <= control"})
                    paired.append({"seed": seed, "candidate_mse": c, "control_mse": b,
                                   "rmse_improvement_percent": percent(math.sqrt(c), math.sqrt(b))})
                c = np.asarray(candidate["parent_mse"])
                b = np.asarray(comparator["parent_mse"])
                require(c.shape == b.shape == (10,) and candidate["parent_ids"] == comparator["parent_ids"] == list(range(10)),
                        "parent coverage")
                count = int(np.sum(c <= b))
                checks.append({"name": base + "/parents_nonworse", "passed": count >= 8,
                               "count": count, "total": 10, "rule": "at least8/10 parent MSE <= control"})
                loo = []
                for i in range(10):
                    cm, bm = float(np.delete(c, i).mean()), float(np.delete(b, i).mean())
                    checks.append({"name": base + f"/leave_out_{i}", "passed": cm < bm,
                                   "candidate": cm, "comparator": bm, "rule": "remaining-parent mean MSE < control"})
                    loo.append({"excluded_parent": i, "candidate_mse": cm, "control_mse": bm})
                differences = b - c
                net = float(differences.sum())
                best = int(np.argmax(differences))
                comparisons[panel][control][endpoint] = {
                    "family_rmse_improvement_percent": percent(candidate["rmse"], comparator["rmse"]),
                    "paired": paired, "parents_nonworse": count, "parent_mse_improvements": differences.tolist(),
                    "largest_gain_parent": best, "largest_gain_fraction_of_net": None if net <= 0 else float(differences[best] / net),
                    "leave_one_parent_out": loo}
    c, b = latency["transport"]["median_ms"], latency["body"]["median_ms"]
    checks.append({"name": "latency_vs_body", "passed": Decimal(str(c)) <= Decimal("1.5") * Decimal(str(b)),
                   "candidate": c, "comparator": b, "rule": "pooled median <=1.5*body"})
    require(len(checks) == 541, "gate count")
    return families, references, latency, comparisons, {"passed": all(x["passed"] for x in checks),
        "checks_passed": sum(x["passed"] for x in checks), "total_checks": 541, "checks": checks,
        "components": {"family": 36, "paired": 108, "parent_counts": 36, "leave_one_parent_out": 360, "latency": 1}}


def expected_members():
    files = {"started.json", "training-completed.json", "ridge16.pt", "completed.json"}
    files.update(f"{v}-{s}{suffix}" for s in SEEDS for v in VARIANTS
                 for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
    for panel in PANELS:
        files.add(panel + "-targets.npz")
        for variant in (*VARIANTS, *REFERENCES):
            for seed in SEEDS if variant in VARIANTS else (None,):
                label = variant if seed is None else f"{variant}-{seed}"
                files.update(f"{panel}-{label}{suffix}" for suffix in ("-predictions.npz", "-evaluation.json"))
    require(len(files) == 126, "internal file count")
    return files


def tensor_weights(path, variant, *, initial=False):
    values = torch.load(path, map_location="cpu", weights_only=True)
    inputs, width = (15 if variant == "world" else 9), 48
    shapes = {"scales": (4,), "readout.weight": (6, width), "readout.bias": (6,)}
    for prefix, count in (("observation_update", inputs), ("transition", inputs + 40)):
        shapes.update({f"{prefix}.weight_ih": (3 * width, count), f"{prefix}.weight_hh": (3 * width, width),
                       f"{prefix}.bias_ih": (3 * width,), f"{prefix}.bias_hh": (3 * width,)})
    require(isinstance(values, dict) and set(values) == set(shapes), "checkpoint field schema")
    for key, shape in shapes.items():
        value = values[key]
        require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == "cpu"
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "checkpoint tensor " + key)
    require(bool((values["scales"] > 0).all()), "motion scales")
    if initial:
        require(bool((values["readout.weight"] == 0).all()) and bool((values["readout.bias"] == 0).all()), "initial skip")
    return values


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "protocol digest")
    protocol = read_json(experiment / "protocol.json")
    required = {"study": "pose-transport-v1", "scope": "exposed-data development, not confirmation",
        "seeds": list(SEEDS), "variants": list(VARIANTS), "references": list(REFERENCES), "panels": list(PANELS),
        "epochs": 30, "batch": 32, "context": 32, "horizon": 25, "learning_rate": .001,
        "grad_norm_cap": 1., "optimizer": "Adam", "expected_fits": 12, "updates_per_fit": 690,
        "position_loss_scale_m": .1, "rotation_loss_scale_rad": .1,
        "checkpoint": "final only; no selection, retries or sweep", "gate": GATE}
    require(all(protocol[k] == value for k, value in required.items()), "protocol settings")
    require(set(protocol["sources"]) == SOURCES, "source membership")
    snapshot = {k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}
    require(snapshot == protocol["sources"] == {p: sha(ROOT / p) for p in SOURCES}, "source hashes")
    require(protocol["runtime"] == {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
        "numpy": np.__version__, "threads": 1, "platform": platform.platform()}, "runtime identity")
    data = Path(protocol["data"])
    bound = members(data)
    require({k: v["sha256"] for k, v in bound.items()} == protocol["data_hashes"], "data membership/hashes")
    require(set(bound) == {"train.npz", "dev.npz", "test_sin.npz", "test_zigzag.npz",
                           "normalization.npz", "manifest.json", "completed.json"}, "prepared data closure")
    manifest, completion = read_json(data / "manifest.json"), read_json(data / "completed.json")
    require(completion["status"] == "completed" and completion["files"] == {
        k: v for k, v in bound.items() if k != "completed.json"}, "prepared completion")
    require(completion["source_sha256"] == manifest["source_sha256"]
            and completion["input_sha256"] == {k: v["sha256"] for k, v in manifest["sources"].items()}, "data provenance")
    specifications = {"train": ("old", list(range(30)), list(range(0, 1151, 50))),
        "dev": ("old", list(range(41, 50)), [0, 594, 1189]),
        **{p: (p.removeprefix("test_"), list(range(10)), np.linspace(0, 9439, 16, dtype=int).tolist()) for p in PANELS}}
    for split, (archive, ids, starts) in specifications.items():
        require(manifest["splits"][split] == {"archive": archive, "source_ids": ids, "starts": starts}, "split identities")
        with np.load(data / f"{split}.npz", allow_pickle=False) as arrays:
            require(set(arrays.files) == {"obs", "actions", "source_ids", "window_starts"}, "prepared fields")
            require(np.array_equal(arrays["source_ids"], np.repeat(ids, len(starts)))
                    and np.array_equal(arrays["window_starts"], np.tile(starts, len(ids))), "prepared IDs")
            for key, tail in (("obs", (57, 9)), ("actions", (56, 40))):
                value = arrays[key]
                require(value.shape == (len(ids) * len(starts), *tail) and value.dtype == np.float32
                        and np.isfinite(value).all(), "prepared shapes/values")
    return protocol, manifest, data


def audit(experiment, out, *, protocol_sha256=None, completed_sha256=None):
    start = time.perf_counter()
    require((protocol_sha256 is None) == (completed_sha256 is None), "supply both external hashes or neither")
    external = protocol_sha256 is not None
    out.mkdir(parents=True, exist_ok=False)
    try:
        run = experiment / "run-01"
        protocol_sha256 = protocol_sha256 or sha(experiment / "protocol.json")
        completed_sha256 = completed_sha256 or sha(run / "completed.json")
        require(sha(run / "completed.json") == completed_sha256, "completion digest")
        protocol, _manifest, data = validate_inputs(experiment, protocol_sha256)
        before = members(run)
        require(set(before) == expected_members(), "exact completed file membership")
        done, boundary = read_json(run / "completed.json"), read_json(run / "training-completed.json")
        require(done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256
                and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "producer seal")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "start binding")
        require(boundary["development_panel_forecasts_started"] is False and boundary["fits"] == done["fits"]
                and len(done["fits"]) == 12, "fit-before-evaluation boundary")
        fit_records, initializations = {}, {}
        for index, (seed, variant) in enumerate((s, v) for s in SEEDS for v in VARIANTS):
            label = f"{variant}-{seed}"
            fit = read_json(run / (label + "-fit.json"))
            require(fit == done["fits"][index] and fit["variant"] == variant and fit["seed"] == seed
                    and fit["updates"] == 690, "fit order/identity/count")
            for key, suffix in (("initial_sha256", "-initial.pt"), ("checkpoint_sha256", ".pt")):
                require(fit[key] == before[label + suffix]["sha256"], "weight hash")
            initial = tensor_weights(run / (label + "-initial.pt"), variant, initial=True)
            final = tensor_weights(run / (label + ".pt"), variant)
            require(torch.equal(initial["scales"], final["scales"]) and fit["scales"] == final["scales"].tolist(), "scale identity")
            require(fit["parameters"] == sum(t.numel() for k, t in final.items() if k != "scales"), "parameter count")
            initializations[label] = initial
            losses = np.load(run / (label + "-losses.npy"), allow_pickle=False)
            require(losses.shape == (690,) and np.isfinite(losses).all() and (losses >= 0).all(), "finite loss records")
            finite(fit["training_seconds"], "training seconds", positive=True)
            fit_records[label] = fit
        for seed in SEEDS:
            body = initializations[f"body-{seed}"]
            for variant in ("transport", "history16"):
                require(all(torch.equal(t, initializations[f"{variant}-{seed}"][k]) for k, t in body.items()), "paired initial tensors")
        require(all(torch.equal(v["scales"], initializations[f"body-{SEEDS[0]}"]["scales"])
                    for v in initializations.values()), "shared training-only scales")
        with np.load(data / "normalization.npz", allow_pickle=False) as normal:
            mean, scale = normal["obs_mean"], normal["obs_scale"]
        require(mean.shape == scale.shape == (9,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), "bound normalization")
        rows, window_errors, row_index = {}, {}, 0
        require(len(done["rows"]) == 36, "all evaluation rows")
        for panel in PANELS:
            with np.load(data / (panel + ".npz"), allow_pickle=False) as arrays:
                ep, er = public_pose(arrays["obs"].astype(np.float64) * scale + mean)
                expected_ids = np.column_stack((arrays["source_ids"], arrays["window_starts"]))
            with np.load(run / (panel + "-targets.npz"), allow_pickle=False) as arrays:
                require(set(arrays.files) == {"p", "R", "ids"}, "target fields")
                tp, tr, ids = arrays["p"], arrays["R"], arrays["ids"]
            require(tp.shape == (160, 25, 3) and tr.shape == (160, 25, 3, 3)
                    and tp.dtype == tr.dtype == np.float32 and ids.dtype == np.int64
                    and np.array_equal(ids, expected_ids) and np.array_equal(tp, ep[:, 32:])
                    and np.allclose(tr, er[:, 32:], rtol=0, atol=2e-7), "target provenance")
            rows[panel] = {}
            window_errors[panel + "__ids"] = ids
            for variant in (*VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in VARIANTS else (None,):
                    label = variant if seed is None else f"{variant}-{seed}"
                    prefix = panel + "-" + label
                    row = read_json(run / (prefix + "-evaluation.json"))
                    require(row == done["rows"][row_index] and row["panel"] == panel
                            and row["variant"] == variant and row["seed"] == seed, "evaluation order/identity")
                    row_index += 1
                    with np.load(run / (prefix + "-predictions.npz"), allow_pickle=False) as arrays:
                        require(set(arrays.files) == {"p", "R"}, "prediction fields")
                        pp, rr = arrays["p"], arrays["R"]
                    require(pp.dtype == rr.dtype == (np.float64 if variant == "ridge16" else np.float32), "prediction precision")
                    error = error_arrays(pp, rr, tp, tr)
                    calculated = reduce_errors(error, ids)
                    validate_metrics(row["metrics"], calculated, rr)
                    if variant == "hold":
                        require(np.array_equal(pp, np.repeat(ep[:, 31:32], 25, axis=1))
                                and np.allclose(rr, np.repeat(er[:, 31:32], 25, axis=1), rtol=0, atol=2e-7), "hold reference")
                    samples = np.asarray(row["latency_ms"], dtype=np.float64)
                    require(samples.shape == (20,) and np.isfinite(samples).all() and (samples > 0).all(), "row timings")
                    for endpoint, array in error.items():
                        window_errors[prefix + "__" + endpoint] = array
                    rows[panel][label] = {"variant": variant, "seed": seed, "errors": calculated,
                        "latency_ms": row["latency_ms"], "checkpoint_sha256": fit_records[label]["checkpoint_sha256"]
                        if variant in VARIANTS else (before["ridge16.pt"]["sha256"] if variant == "ridge16" else None)}
        families, references, latency, contrasts, gate = aggregate(rows)
        wall = finite(done["wall_seconds"], "execution wall", positive=True)
        training = sum(f["training_seconds"] for f in fit_records.values())
        ridge = finite(done["ridge_seconds"], "ridge seconds", positive=True)
        require(boundary["ridge_seconds"] == ridge, "ridge boundary")
        timing = sum(sum(r["latency_ms"]) for group in rows.values() for r in group.values()) / 1000
        require(training + ridge + timing <= wall + 1e-6, "nested timing accounting")
        summary = {"status": "completed", "study": "pose-transport-v1", "scope": protocol["scope"],
            "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "fits": fit_records, "rows": rows, "families": families, "references": references,
            "latency": latency, "contrasts": contrasts, "continuation_gate": gate,
            "counts": {"fits": 12, "recorded_updates": 8280, "neural_rows": 24, "reference_rows": 12,
                "panels": 2, "windows_per_panel": 160, "parents_per_panel": 10,
                "position_and_rotation_error_pairs": 36 * 160 * 25, "execution_files": 126},
            "costs": {"execution_wall_seconds": wall, "training_seconds": training, "ridge_fit_seconds": ridge,
                "measured_prediction_calls_seconds": timing,
                "scope": "Nested components; not additive to execution wall. Receipt wall excludes final payload hashing/write. Timing includes complete-window reconstruction/forecast, excludes loading."},
            "limits": ["Exposed-data development screen, not independent confirmation or architectural novelty.",
                "All forecasts condition on supplied recorded future torques; no control utility or issued-action causality claim.",
                "Families pool squared errors before square roots, with all3 fits/all16 windows/all25 horizons per parent.",
                "No inference, training, optimizer, RNG or native calls. weights_only loading inspects tensor schemas/identities only.",
                "Recorded losses do not independently prove numerical optimizer updates; batch order/training scales/causal model execution remain source-bound.",
                "Neural and fitted-reference predictions are not regenerated. Classical predictions except hold are source-bound, not numerically replayed.",
                "Target conversion uses independent NumPy Euler arithmetic; tolerance2e-7 allows float32 library rounding.",
                "Rotation validity requires orthogonality/determinant errors <=1e-4; reported float32 quality reductions allow3e-7 arithmetic tolerance.",
                "Body/transport/history16 initial tensors match; world has larger inputs/parameter count and is not parameter matched.",
                "Latencies pool120 samples per neural family and40 per reference; no fake reference fit replication.",
                "Prior failed residual-dynamics gate remains unchanged; passing this screen would only justify fresh confirmation."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before, "execution changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "externally_supplied_hashes": external,
            "execution_members": before, "sources": protocol["sources"], "data_hashes": protocol["data_hashes"],
            "auditor_sha256": sha(__file__), "files": members(out), "qualification_passed": gate["passed"],
            "checks_passed": gate["checks_passed"], "total_checks": 541, "new_model_calls": 0,
            "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                "wall_seconds": time.perf_counter() - start, "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - retain the original failure
            error.add_note(f"Could not write failure receipt: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--completed-sha256")
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256,
                   completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "checks_passed", "total_checks", "wall_seconds")}))
