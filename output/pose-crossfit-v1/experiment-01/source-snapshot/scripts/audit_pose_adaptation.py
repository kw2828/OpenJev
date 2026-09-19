"""Saved-output audit of pose adaptation; no inference, fitting or random draws.

Independent NumPy pose arithmetic is reused from the explicitly pinned earlier
saved-output auditor. Torch is used only to inspect weights_only tensor payloads.
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
from types import ModuleType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARITHMETIC_SOURCE = "scripts/audit_pose_transport.py"
ARITHMETIC_SHA256 = "abca9f05a1043f8741a43c11342ae289be08f73763aafed02d89afa823676026"


def load_arithmetic(path):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ARITHMETIC_SHA256:
        raise ValueError("frozen arithmetic source digest")
    module = ModuleType("_pose_adaptation_bound_arithmetic")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102 - execute exactly hash-pinned source
    return module


_base = load_arithmetic(ROOT / ARITHMETIC_SOURCE)
require, sha, read_json, write_json = _base.require, _base.sha, _base.read_json, _base.write_json
members, finite = _base.members, _base.finite
rotation_quality, rotation_angles = _base.rotation_quality, _base.rotation_angles
public_pose, error_arrays, reduce_errors = _base.public_pose, _base.error_arrays, _base.reduce_errors
validate_metrics, pooled, percent = _base.validate_metrics, _base.pooled, _base.percent
SEEDS = (1101, 1202, 1303)
TRAIN_VARIANTS = ("meta", "static", "public", "gru")
VARIANTS = ("meta", "meta_prior", "static", "static_adapt", "public", "gru")
REFERENCES = ("hold", "cv1", "cv16", "ls16", "body16", "ridge16")
PANELS = ("test_sin", "test_zigzag")
ENDPOINTS = ("position", "rotation")
SOURCES = {
    "scripts/train_pose_adaptation.py", "scripts/audit_pose_adaptation.py", ARITHMETIC_SOURCE,
    "src/openjev/research/pose_adaptation.py", "src/openjev/research/pose_transport.py",
    "src/openjev/research/pose_references.py", "src/openjev/research/rigid_motion.py",
    "tests/test_pose_adaptation.py", "tests/test_pose_adaptation_training.py",
    "tests/test_audit_pose_adaptation.py", "research/pose-adaptation-protocol.md",
    "research/pose-adaptation-diagnosis.md",
}
GATE = ("meta at least10% better on each physical endpoint against every learned and classical control "
        "on both panels; all3 paired fit endpoints nonworse; at least8/10 parents nonworse for each "
        "endpoint/control/panel; strictly lower MSE after each leave-one-parent-out; median full-window "
        "latency<=1.5x gru. Failure stays failed.")


def backbone(variant):
    require(variant in VARIANTS, "unknown learned variant")
    return {"meta_prior": "meta", "static_adapt": "static"}.get(variant, variant)


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
        for control in (*[v for v in VARIANTS if v != "meta"], *REFERENCES):
            comparisons[panel][control] = {}
            for endpoint in ENDPOINTS:
                candidate = families[panel]["meta"][endpoint]
                comparator = (families[panel][control] if control in VARIANTS else references[panel][control])[endpoint]
                base = f"{panel}/{control}/{endpoint}"
                checks.append({"name": base + "/family_10percent", "passed": comparator["rmse"] > 0 and
                    Decimal(str(candidate["rmse"])) <= Decimal(".90") * Decimal(str(comparator["rmse"])),
                    "candidate": candidate["rmse"], "comparator": comparator["rmse"], "rule": "RMSE <=0.90*positive control"})
                paired = []
                for seed in SEEDS:
                    c = rows[panel][f"meta-{seed}"]["errors"][endpoint]["mse"]
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
    c, b = latency["meta"]["median_ms"], latency["gru"]["median_ms"]
    checks.append({"name": "latency_vs_gru", "passed": Decimal(str(c)) <= Decimal("1.5") * Decimal(str(b)),
                   "candidate": c, "comparator": b, "rule": "pooled median <=1.5*gru"})
    require(len(checks) == 661, "gate count")
    grouped = []
    for panel in PANELS:
        for endpoint in ENDPOINTS:
            for category, suffix in (("family", "/family_10percent"), ("paired", "_nonworse"),
                                     ("parents", "/parents_nonworse"), ("leave_one_out", "/leave_out_")):
                selected = [x for x in checks if x["name"].startswith(panel + "/")
                            and f"/{endpoint}/" in x["name"]
                            and (suffix in x["name"] if category == "leave_one_out" else x["name"].endswith(suffix))
                            and ("/pair_" in x["name"] if category == "paired" else True)]
                require(len(selected) == {"family": 11, "paired": 33, "parents": 11, "leave_one_out": 110}[category],
                        "grouped gate membership")
                grouped.append({"name": f"{panel}/{endpoint}/{category}",
                                "passed": all(x["passed"] for x in selected),
                                "comparisons_passed": sum(x["passed"] for x in selected),
                                "total_comparisons": len(selected),
                                "comparison_names": [x["name"] for x in selected]})
    grouped.append({"name": "latency_vs_gru", "passed": checks[-1]["passed"],
                    "comparisons_passed": int(checks[-1]["passed"]), "total_comparisons": 1,
                    "comparison_names": [checks[-1]["name"]]})
    require(len(grouped) == 17 and sum(x["total_comparisons"] for x in grouped) == 661, "group count")
    return families, references, latency, comparisons, {"passed": all(x["passed"] for x in checks),
        "requirements_passed": sum(x["passed"] for x in grouped), "total_requirements": 17,
        "requirements": grouped, "checks_passed": sum(x["passed"] for x in checks),
        "total_checks": 661, "checks": checks,
        "components": {"family": 44, "paired": 132, "parent_counts": 44, "leave_one_parent_out": 440, "latency": 1}}


def expected_members():
    files = {"started.json", "training-completed.json", "ridge16.pt", "completed.json"}
    files.update(f"{v}-{s}{suffix}" for s in SEEDS for v in TRAIN_VARIANTS
                 for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
    for panel in PANELS:
        files.add(panel + "-targets.npz")
        for variant in (*VARIANTS, *REFERENCES):
            for seed in SEEDS if variant in VARIANTS else (None,):
                label = variant if seed is None else f"{variant}-{seed}"
                files.update(f"{panel}-{label}{suffix}" for suffix in ("-predictions.npz", "-evaluation.json"))
    require(len(files) == 150, "internal file count")
    return files


def tensor_weights(path, variant, *, initial=False):
    require(variant in TRAIN_VARIANTS, "unknown training variant")
    if variant == "gru":
        return _base.tensor_weights(path, "body", initial=initial)
    values = torch.load(path, map_location="cpu", weights_only=True)
    shapes = {"scales": (4,), "prior": (50 if variant == "public" else 13, 6)}
    if variant != "public":
        shapes.update({"encoder.0.weight": (48, 49), "encoder.0.bias": (48,),
                       "encoder.2.weight": (12, 48), "encoder.2.bias": (12,)})
    require(isinstance(values, dict) and set(values) == set(shapes), "checkpoint field schema")
    for key, shape in shapes.items():
        value = values[key]
        require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == "cpu"
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "checkpoint tensor " + key)
    require(bool((values["scales"] > 0).all()), "motion scales")
    if initial:
        require(bool((values["prior"] == 0).all()), "initial prior must be zero")
    return values


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "protocol digest")
    protocol = read_json(experiment / "protocol.json")
    required = {"study": "pose-adaptation-v1", "scope": "exposed-data development, not confirmation",
        "seeds": list(SEEDS), "variants": list(VARIANTS), "train_variants": list(TRAIN_VARIANTS), "references": list(REFERENCES), "panels": list(PANELS),
        "epochs": 30, "batch": 32, "context": 32, "horizon": 25, "learning_rate": .001,
        "grad_norm_cap": 1., "optimizer": "Adam", "expected_fits": 12, "updates_per_fit": 690,
        "position_loss_scale_m": .1, "rotation_loss_scale_rad": .1,
        "support_rows": 30, "ridge_precision": 1., "posterior_solve": "float64 Cholesky then float32 weights",
        "feature_normalization": "L2 normalization eps1e-8 then intercept1",
        "gate_grouping": "17 grouped requirements retain all661 elementary comparisons",
        "checkpoint": "final only; no selection, retries or sweep", "gate": GATE}
    require(all(protocol[k] == value for k, value in required.items()), "protocol settings")
    require(set(protocol["sources"]) == SOURCES, "source membership")
    require(protocol["sources"][ARITHMETIC_SOURCE] == ARITHMETIC_SHA256, "arithmetic source binding")
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
        for index, (seed, variant) in enumerate((s, v) for s in SEEDS for v in TRAIN_VARIANTS):
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
            meta = initializations[f"meta-{seed}"]
            require(all(torch.equal(t, initializations[f"static-{seed}"][k]) for k, t in meta.items()),
                    "paired meta/static initial tensors")
        require(all(torch.equal(v["scales"], initializations[f"meta-{SEEDS[0]}"]["scales"])
                    for v in initializations.values()), "shared training-only scales")
        with np.load(data / "normalization.npz", allow_pickle=False) as normal:
            mean, scale = normal["obs_mean"], normal["obs_scale"]
        require(mean.shape == scale.shape == (9,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), "bound normalization")
        rows, window_errors, row_index = {}, {}, 0
        require(len(done["rows"]) == 48, "all evaluation rows")
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
                        "latency_ms": row["latency_ms"], "checkpoint_sha256": fit_records[f"{backbone(variant)}-{seed}"]["checkpoint_sha256"]
                        if variant in VARIANTS else (before["ridge16.pt"]["sha256"] if variant == "ridge16" else None)}
        families, references, latency, contrasts, gate = aggregate(rows)
        wall = finite(done["wall_seconds"], "execution wall", positive=True)
        training = sum(f["training_seconds"] for f in fit_records.values())
        ridge = finite(done["ridge_seconds"], "ridge seconds", positive=True)
        require(boundary["ridge_seconds"] == ridge, "ridge boundary")
        timing = sum(sum(r["latency_ms"]) for group in rows.values() for r in group.values()) / 1000
        require(training + ridge + timing <= wall + 1e-6, "nested timing accounting")
        summary = {"status": "completed", "study": "pose-adaptation-v1", "scope": protocol["scope"],
            "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "fits": fit_records, "rows": rows, "families": families, "references": references,
            "latency": latency, "contrasts": contrasts, "continuation_gate": gate,
            "counts": {"fits": 12, "recorded_updates": 8280, "neural_rows": 36, "reference_rows": 12,
                "panels": 2, "windows_per_panel": 160, "parents_per_panel": 10,
                "position_and_rotation_error_pairs": 48 * 160 * 25, "execution_files": 150},
            "costs": {"execution_wall_seconds": wall, "training_seconds": training, "ridge_fit_seconds": ridge,
                "measured_prediction_calls_seconds": timing,
                "scope": "Nested components; not additive to execution wall. Receipt wall excludes final payload hashing/write. Timing includes complete-window reconstruction/forecast, excludes loading."},
            "limits": ["Exposed-data development screen, not independent confirmation or architectural novelty.",
                "All forecasts condition on supplied recorded future torques; no control utility or issued-action causality claim.",
                "Families pool squared errors before square roots, with all 3 fits, 16 windows and 25 horizons per parent.",
                "No inference, training, optimizer, RNG or native calls. weights_only loading inspects tensor schemas/identities only.",
                "Recorded losses do not independently prove numerical optimizer updates; batch order/training scales/causal model execution remain source-bound.",
                "Neural and fitted-reference predictions are not regenerated. Classical predictions except hold are source-bound, not numerically replayed.",
                "Target conversion uses independent NumPy Euler arithmetic; tolerance2e-7 allows float32 library rounding.",
                "Rotation validity requires orthogonality/determinant errors <=1e-4; reported float32 quality reductions allow3e-7 arithmetic tolerance.",
                "Meta/static initial tensors match. Public features have 50 versus 13 columns; GRU is a separate architecture, not parameter matched.",
                "Meta_prior and static_adapt reuse their respective final meta/static checkpoint; execution of adaptation is source-bound, not replayed.",
                "Context ridge supports, training order, learned feature execution and frozen forecast weights are source-bound, not independently regenerated.",
                "The fixed ridge precision does not establish parameter identifiability or calibrated uncertainty.",
                "17 grouped requirements are conjunctions of 661 comparisons, not independent statistical tests.",
                "Latencies pool 120 samples per neural family and 40 per reference; no fake reference fit replication.",
                "Prior residual-dynamics and pose-transport failed gates remain unchanged; passing only justifies fresh confirmation."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before, "execution changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "externally_supplied_hashes": external,
            "execution_members": before, "sources": protocol["sources"], "data_hashes": protocol["data_hashes"],
            "auditor_sha256": sha(__file__), "arithmetic_source_sha256": ARITHMETIC_SHA256, "files": members(out), "qualification_passed": gate["passed"],
            "checks_passed": gate["checks_passed"], "total_checks": 661, "requirements_passed": gate["requirements_passed"], "total_requirements": 17, "new_model_calls": 0,
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
