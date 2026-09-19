"""Saved-output audit of fixed-checkpoint support weighting, without inference.

Pinned prior auditors supply NumPy pose arithmetic and byte/schema validation.
Compact support summaries do not permit independent reconstruction of learned
features, Huber residuals or solved weights; those remain source-bound.
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

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "scripts/audit_pose_transport.py": "abca9f05a1043f8741a43c11342ae289be08f73763aafed02d89afa823676026",
    "scripts/audit_pose_adaptation.py": "1147f387ec9499d66edbc60cad17afbbef00720f98495593001b6d1bd65aac01",
}


def load_pinned(path, digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("pinned auditor source digest")
    module = ModuleType("_pose_support_bound_auditor")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102 - exact hash-pinned local bytes
    return module


_previous = load_pinned(ROOT / "scripts/audit_pose_adaptation.py", DEPENDENCIES["scripts/audit_pose_adaptation.py"])
require, sha, read_json, write_json = _previous.require, _previous.sha, _previous.read_json, _previous.write_json
members, finite = _previous.members, _previous.finite
public_pose, error_arrays, reduce_errors = _previous.public_pose, _previous.error_arrays, _previous.reduce_errors
rotation_angles, rotation_quality = _previous.rotation_angles, _previous.rotation_quality
validate_metrics, pooled, percent = _previous.validate_metrics, _previous.pooled, _previous.percent
SEEDS = (1101, 1202, 1303)
SUPPORT_VARIANTS = ("prior", "full", "recent5", "recent5_mass", "decay5", "decay5_mass", "huber3", "huber3_mass",
                    "decay_huber3", "decay_huber3_mass")
OLD_VARIANTS = ("static", "static_adapt", "public", "gru")
VARIANTS = (*SUPPORT_VARIANTS, *OLD_VARIANTS)
REFERENCES = ("hold", "cv1", "cv16", "ls16", "body16", "ridge16")
PANELS = ("test_sin", "test_zigzag")
ENDPOINTS = ("position", "rotation")
PRIMARY = "decay_huber3"
PARITY_TOLERANCE = {"float32": {"rtol": 1e-6, "atol": 2e-7}, "float64": {"rtol": 1e-12, "atol": 1e-12}}
DIAGNOSTIC_FIELDS = {"version", "variant", "base_variant", "adapted", "mass_control", "support_rows",
    "huber_iterations", "ridge_precision", "solve_dtype", "batched_solve_calls", "cholesky_factorizations",
    "actual", "derivation", "correction_frobenius_norm", "derivation_correction_frobenius_norm"}
STATS_FIELDS = {"weight_sum", "effective_n", "fraction_weight_lt_one"}


def backbone(variant):
    require(variant in VARIANTS, "unknown learned variant")
    return "meta" if variant in SUPPORT_VARIANTS else ("static" if variant == "static_adapt" else variant)


def expected_members():
    files = {"started.json", "completed.json"}
    for panel in PANELS:
        files.add(panel + "-targets.npz")
        for variant in (*VARIANTS, *REFERENCES):
            for seed in SEEDS if variant in VARIANTS else (None,):
                label = variant if seed is None else f"{variant}-{seed}"
                files.update(f"{panel}-{label}{suffix}" for suffix in ("-predictions.npz", "-evaluation.json"))
    require(len(files) == 196, "internal file count")
    return files


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
        for control in (*[v for v in VARIANTS if v != PRIMARY], *REFERENCES):
            comparisons[panel][control] = {}
            for endpoint in ENDPOINTS:
                candidate = families[panel][PRIMARY][endpoint]
                comparator = (families[panel][control] if control in VARIANTS else references[panel][control])[endpoint]
                base = f"{panel}/{control}/{endpoint}"
                checks.append({"name": base + "/family_10percent", "passed": comparator["rmse"] > 0 and
                    Decimal(str(candidate["rmse"])) <= Decimal(".90") * Decimal(str(comparator["rmse"])),
                    "candidate": candidate["rmse"], "comparator": comparator["rmse"], "rule": "RMSE <=0.90*positive control"})
                paired = []
                for seed in SEEDS:
                    c = rows[panel][f"{PRIMARY}-{seed}"]["errors"][endpoint]["mse"]
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
    c, b = latency[PRIMARY]["median_ms"], latency["gru"]["median_ms"]
    checks.append({"name": "latency_vs_gru", "passed": Decimal(str(c)) <= Decimal("1.5") * Decimal(str(b)),
                   "candidate": c, "comparator": b, "rule": "pooled median <=1.5*gru"})
    require(len(checks) == 1141, "gate count")
    grouped = []
    for panel in PANELS:
        for endpoint in ENDPOINTS:
            for category, suffix in (("family", "/family_10percent"), ("paired", "_nonworse"),
                                     ("parents", "/parents_nonworse"), ("leave_one_out", "/leave_out_")):
                selected = [x for x in checks if x["name"].startswith(panel + "/")
                            and f"/{endpoint}/" in x["name"]
                            and (suffix in x["name"] if category == "leave_one_out" else x["name"].endswith(suffix))
                            and ("/pair_" in x["name"] if category == "paired" else True)]
                require(len(selected) == {"family": 19, "paired": 57, "parents": 19, "leave_one_out": 190}[category],
                        "grouped gate membership")
                grouped.append({"name": f"{panel}/{endpoint}/{category}",
                                "passed": all(x["passed"] for x in selected),
                                "comparisons_passed": sum(x["passed"] for x in selected),
                                "total_comparisons": len(selected),
                                "comparison_names": [x["name"] for x in selected]})
    grouped.append({"name": "latency_vs_gru", "passed": checks[-1]["passed"],
                    "comparisons_passed": int(checks[-1]["passed"]), "total_comparisons": 1,
                    "comparison_names": [checks[-1]["name"]]})
    require(len(grouped) == 17 and sum(x["total_comparisons"] for x in grouped) == 1141, "group count")
    return families, references, latency, comparisons, {"passed": all(x["passed"] for x in checks),
        "requirements_passed": sum(x["passed"] for x in grouped), "total_requirements": 17,
        "requirements": grouped, "checks_passed": sum(x["passed"] for x in checks),
        "total_checks": 1141, "checks": checks,
        "components": {"family": 76, "paired": 228, "parent_counts": 76, "leave_one_parent_out": 760, "latency": 1}}



def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all() and (result >= 0).all(), name)
    return result


def _close(left, right, name):
    require(np.allclose(left, right, rtol=1e-6, atol=2e-7), name)


def _known_stats(base):
    if base == "prior":
        return {k: 0. for k in STATS_FIELDS}
    if base == "full":
        weights = np.ones(30)
    elif base == "recent5":
        weights = np.concatenate((np.zeros(25), np.ones(5)))
    elif base == "decay5":
        weights = np.power(2., -np.arange(29, -1, -1, dtype=np.float64) / 5)
    else:
        raise ValueError("unknown fixed weights")
    return {"weight_sum": float(weights.sum()), "effective_n": float(weights.sum() ** 2 / np.square(weights).sum()),
            "fraction_weight_lt_one": float((weights < 1).mean())}


def validate_diagnostics(saved, variant, batch=160):
    if variant not in SUPPORT_VARIANTS:
        require(saved is None, "old control diagnostics must be null")
        return None
    require(isinstance(saved, dict) and set(saved) == DIAGNOSTIC_FIELDS, "diagnostic fields")
    mass = variant.endswith("_mass")
    base = variant.removesuffix("_mass")
    huber = "huber3" in base
    calls = 0 if base == "prior" else (3 if huber else 1) + int(mass)
    factorizations = batch if base == "full" else 6 * batch * calls
    expected = {"version": "pose-support-v1", "variant": variant, "base_variant": base,
        "adapted": base != "prior", "mass_control": mass, "support_rows": 30,
        "huber_iterations": 3 if huber else 0, "ridge_precision": 1., "solve_dtype": "float64",
        "batched_solve_calls": calls, "cholesky_factorizations": factorizations}
    require(all(saved[k] == value for k, value in expected.items()), "diagnostic identity/work")
    require(type(saved["adapted"]) is bool and type(saved["mass_control"]) is bool, "diagnostic booleans")
    stats = {}
    for category in ("actual", "derivation"):
        require(isinstance(saved[category], dict) and set(saved[category]) == STATS_FIELDS, "support statistic fields")
        stats[category] = {k: _array(v, (batch, 6), "support statistic " + k) for k, v in saved[category].items()}
        m, ess, fraction = (stats[category][k] for k in ("weight_sum", "effective_n", "fraction_weight_lt_one"))
        require((m <= 30 + 2e-7).all() and (ess <= 30 + 2e-7).all() and (fraction <= 1).all(), "support statistic bounds")
        _close(fraction * 30, np.round(fraction * 30), "downweighted fraction is not a support count")
        if base != "prior":
            require((m > 0).all() and (ess >= 1 - 2e-7).all(), "positive support mass/effective size")
        if not huber:
            fixed = _known_stats(base)
            for key in STATS_FIELDS:
                expected_value = fixed[key]
                if mass and category == "actual":
                    expected_value = {"weight_sum": fixed["weight_sum"], "effective_n": 30.,
                                      "fraction_weight_lt_one": 1.}[key]
                _close(stats[category][key], expected_value, "known support statistic " + category + "/" + key)
        elif base == "decay_huber3" and category == "derivation":
            require((m <= _known_stats("decay5")["weight_sum"] + 2e-7).all(), "decay-Huber maximum mass")
            require((fraction >= 29 / 30 - 2e-7).all(), "decay-Huber downweighted support")
    if not mass:
        for key in STATS_FIELDS:
            _close(stats["actual"][key], stats["derivation"][key], "base actual/derivation identity")
    else:
        _close(stats["actual"]["weight_sum"], stats["derivation"]["weight_sum"], "matched mass identity")
        _close(stats["actual"]["effective_n"], 30., "mass control uniform support ESS")
        _close(stats["actual"]["fraction_weight_lt_one"],
               (stats["derivation"]["weight_sum"] < 30).astype(float), "mass control uniform downweighting")
    norms = {k: _array(saved[k], (batch,), "correction norm") for k in
             ("correction_frobenius_norm", "derivation_correction_frobenius_norm")}
    if base == "prior":
        require(all((x == 0).all() for x in norms.values()), "prior correction must be zero")
    if not mass:
        _close(norms["correction_frobenius_norm"], norms["derivation_correction_frobenius_norm"],
               "base correction norm identity")
    return saved


def validate_mass_pairs(rows):
    for panel in PANELS:
        for seed in SEEDS:
            for base in ("recent5", "decay5", "huber3", "decay_huber3"):
                direct = rows[panel][f"{base}-{seed}"]["diagnostics"]
                mass = rows[panel][f"{base}_mass-{seed}"]["diagnostics"]
                for key in STATS_FIELDS:
                    _close(mass["derivation"][key], direct["actual"][key], "mass derivation differs from base " + key)
                _close(mass["derivation_correction_frobenius_norm"], direct["correction_frobenius_norm"],
                       "mass base correction identity")


def validate_replay(p, rotation, previous_path):
    with np.load(previous_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"p", "R"}, "prior prediction fields")
        previous = {k: saved[k] for k in ("p", "R")}
    report = {"prior_predictions_sha256": sha(previous_path), "tolerance": PARITY_TOLERANCE[str(p.dtype)]}
    for key, value in (("p", p), ("R", rotation)):
        require(value.shape == previous[key].shape and value.dtype == previous[key].dtype
                and np.isfinite(previous[key]).all(), "prior prediction shape/precision")
        require(np.allclose(value, previous[key], **report["tolerance"]), "prior prediction replay mismatch " + key)
        report[key] = {"max_abs_error": float(np.abs(value.astype(float) - previous[key].astype(float)).max()),
                       "exact_array_equal": bool(np.array_equal(value, previous[key]))}
    return report


def window_groups(error, ids):
    result = {}
    for name, mask in (("start_zero", ids[:, 1] == 0), ("remaining", ids[:, 1] != 0)):
        require(int(mask.sum()) == (10 if name == "start_zero" else 150), "descriptive window grouping")
        result[name] = {"windows": int(mask.sum()), "errors": reduce_errors({k: v[mask] for k, v in error.items()}, ids[mask])}
    return result

SOURCES = {
    "scripts/run_pose_support.py", "scripts/audit_pose_support.py", *DEPENDENCIES,
    "scripts/train_pose_adaptation.py", "src/openjev/research/pose_support.py",
    "src/openjev/research/pose_adaptation.py", "src/openjev/research/pose_transport.py",
    "src/openjev/research/pose_references.py", "src/openjev/research/rigid_motion.py",
    "tests/test_pose_support.py", "tests/test_pose_support_runner.py", "tests/test_audit_pose_support.py",
    "research/pose-support-protocol.md", "research/pose-adaptation-failure-diagnosis.md",
}
GATE = ("decay_huber3 RMSE <=0.90*each positive control on both physical endpoints/panels; "
        "all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out "
        "MSE strictly lower; median full-window latency<=1.5*gru. All17 groups required.")
PRIOR_PROTOCOL_SHA256 = "f029b98d96db6fe3aa0383290b22a71f3ae53305a1b5e506c5b37e48d2056615"
PRIOR_COMPLETED_SHA256 = "c7cc39977ee5dade2cf610aa7bb795e7f9eb34fad97b337c723437200988c799"


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "protocol digest")
    protocol = read_json(experiment / "protocol.json")
    required = {"study": "pose-support-v1", "scope": "exposed-data fixed-checkpoint mechanism screen",
        "seeds": list(SEEDS), "support_variants": list(SUPPORT_VARIANTS), "old_variants": list(OLD_VARIANTS),
        "variants": list(VARIANTS), "references": list(REFERENCES), "panels": list(PANELS), "primary": PRIMARY,
        "new_training_fits": 0, "new_optimizer_updates": 0, "context": 32, "horizon": 25,
        "support_rows": 30, "recent_rows": 5, "decay_half_life": 5., "huber_delta": 1.5,
        "huber_iterations": 3, "ridge_precision": 1., "position_loss_scale_m": .1, "rotation_loss_scale_rad": .1,
        "mass_control": "Derive paired weights fully, replace by per-output mean over30 supports, solve once more.",
        "warmups_per_row": 3, "timed_windows_per_row": 20, "evaluation_rows": 96,
        "gate": GATE, "gate_groups": 17, "gate_comparisons": 1141,
        "prior_protocol_sha256": PRIOR_PROTOCOL_SHA256, "prior_completed_sha256": PRIOR_COMPLETED_SHA256,
        "no_selection": "Fixed arms and checkpoints; no retries, sweeps, excluded windows or gate changes."}
    require(all(protocol[k] == value for k, value in required.items()), "protocol settings")
    require(set(protocol["sources"]) == SOURCES, "source membership")
    require(all(protocol["sources"][p] == h for p, h in DEPENDENCIES.items()), "pinned dependencies")
    snapshot = {k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}
    require(snapshot == protocol["sources"] == {p: sha(ROOT / p) for p in SOURCES}, "source hashes")
    require(protocol["runtime"] == {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
        "numpy": np.__version__, "threads": 1, "platform": platform.platform()}, "runtime identity")
    prior = Path(protocol["prior_experiment"])
    old_protocol, _manifest, data = _previous.validate_inputs(prior, PRIOR_PROTOCOL_SHA256)
    require(protocol["data"] == str(data.resolve()) and protocol["data_hashes"] == old_protocol["data_hashes"],
            "unchanged prior data")
    prior_run = prior / "run-01"
    old_members = members(prior_run)
    require(set(old_members) == _previous.expected_members()
            and old_members["completed.json"]["sha256"] == PRIOR_COMPLETED_SHA256, "prior completion/membership")
    done = read_json(prior_run / "completed.json")
    require(done["status"] == "completed" and done["protocol_sha256"] == PRIOR_PROTOCOL_SHA256
            and done["files"] == {k: v for k, v in old_members.items() if k != "completed.json"}, "prior payload seal")
    names = [f"{v}-{seed}" for seed in SEEDS for v in ("meta", "static", "public", "gru")] + ["ridge16"]
    require(set(protocol["checkpoints"]) == set(names) and len(done["fits"]) == 12, "prior checkpoint coverage")
    for i, name in enumerate(names):
        binding = protocol["checkpoints"][name]
        require(binding == {"path": str((prior_run / (name + ".pt")).resolve()),
                            "sha256": old_members[name + ".pt"]["sha256"]}, "checkpoint path/hash binding")
        if name != "ridge16":
            variant, seed = name.rsplit("-", 1)
            fit = read_json(prior_run / (name + "-fit.json"))
            require(fit == done["fits"][i] and fit["variant"] == variant and fit["seed"] == int(seed)
                    and fit["checkpoint_sha256"] == binding["sha256"], "prior fit/checkpoint identity")
            state = _previous.tensor_weights(Path(binding["path"]), variant)
            require(fit["scales"] == state["scales"].tolist(), "prior scale identity")
    return protocol, data, prior_run, old_members


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
        protocol, data, prior_run, old_members = validate_inputs(experiment, protocol_sha256)
        before = members(run)
        require(set(before) == expected_members(), "exact completed file membership")
        done = read_json(run / "completed.json")
        require(set(done) == {"status", "rows", "new_training_fits", "new_optimizer_updates", "wall_seconds",
                              "protocol_sha256", "files"}, "completion fields")
        require(done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256
                and done["new_training_fits"] == done["new_optimizer_updates"] == 0
                and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "producer seal")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "start binding")
        require(len(done["rows"]) == 96, "all evaluation rows")
        with np.load(data / "normalization.npz", allow_pickle=False) as normal:
            mean, scale = normal["obs_mean"], normal["obs_scale"]
        require(mean.shape == scale.shape == (9,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), "bound normalization")
        rows, window_errors, replays, descriptive, row_index = {}, {}, {}, {}, 0
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
            with np.load(prior_run / (panel + "-targets.npz"), allow_pickle=False) as previous_target:
                require(set(previous_target.files) == {"p", "R", "ids"}
                        and all(np.array_equal(previous_target[k], v) for k, v in (("p", tp), ("R", tr), ("ids", ids))),
                        "prior target equality")
            rows[panel], replays[panel], descriptive[panel] = {}, {}, {}
            window_errors[panel + "__ids"] = ids
            for variant in (*VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in VARIANTS else (None,):
                    label = variant if seed is None else f"{variant}-{seed}"
                    prefix = panel + "-" + label
                    row = read_json(run / (prefix + "-evaluation.json"))
                    require(set(row) == {"panel", "variant", "seed", "checkpoint_sha256", "metrics", "latency_ms", "diagnostics"},
                            "evaluation fields")
                    require(row == done["rows"][row_index] and row["panel"] == panel
                            and row["variant"] == variant and row["seed"] == seed, "evaluation order/identity")
                    row_index += 1
                    checkpoint_name = f"{backbone(variant)}-{seed}" if variant in VARIANTS else variant
                    expected_checkpoint = protocol["checkpoints"][checkpoint_name]["sha256"] if variant in VARIANTS or variant == "ridge16" else None
                    require(row["checkpoint_sha256"] == expected_checkpoint, "row checkpoint identity")
                    info = validate_diagnostics(row["diagnostics"], variant)
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
                    if variant in ("prior", "full", *OLD_VARIANTS, *REFERENCES):
                        previous_variant = {"prior": "meta_prior", "full": "meta"}.get(variant, variant)
                        previous_label = previous_variant if seed is None else f"{previous_variant}-{seed}"
                        replays[panel][label] = validate_replay(pp, rr, prior_run / (panel + "-" + previous_label + "-predictions.npz"))
                    samples = np.asarray(row["latency_ms"], dtype=np.float64)
                    require(samples.shape == (20,) and np.isfinite(samples).all() and (samples > 0).all(), "row timings")
                    for endpoint, array in error.items():
                        window_errors[prefix + "__" + endpoint] = array
                    rows[panel][label] = {"variant": variant, "seed": seed, "errors": calculated,
                        "latency_ms": row["latency_ms"], "checkpoint_sha256": expected_checkpoint, "diagnostics": info}
                    descriptive[panel][label] = window_groups(error, ids)
        validate_mass_pairs(rows)
        require(sum(len(v) for v in replays.values()) == 48, "prior replay coverage")
        families, references, latency, contrasts, gate = aggregate(rows)
        descriptive_families = {panel: {variant: {group: pooled([
            descriptive[panel][f"{variant}-{seed}"][group]["errors"] for seed in SEEDS])
            for group in ("start_zero", "remaining")} for variant in VARIANTS} for panel in PANELS}
        wall = finite(done["wall_seconds"], "execution wall", positive=True)
        timing = sum(sum(r["latency_ms"]) for group in rows.values() for r in group.values()) / 1000
        require(timing <= wall, "nested timing accounting")
        work = {"diagnostic_batched_solve_calls": sum(r["diagnostics"]["batched_solve_calls"]
                for group in rows.values() for r in group.values() if r["diagnostics"] is not None),
            "diagnostic_cholesky_factorizations": sum(r["diagnostics"]["cholesky_factorizations"]
                for group in rows.values() for r in group.values() if r["diagnostics"] is not None),
            "scope": "Reported full-batch support diagnostics only; excludes warmup/timed calls and old control solves."}
        summary = {"status": "completed", "study": "pose-support-v1", "scope": protocol["scope"],
            "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "prior_protocol_sha256": protocol["prior_protocol_sha256"], "prior_completed_sha256": protocol["prior_completed_sha256"],
            "rows": rows, "families": families, "references": references, "latency": latency,
            "contrasts": contrasts, "continuation_gate": gate, "prior_replay": replays,
            "descriptive_window_groups": descriptive, "descriptive_group_families": descriptive_families,
            "counts": {"new_training_fits": 0, "new_optimizer_updates": 0, "checkpoints": 13,
                "neural_configuration_rows": 84, "reference_rows": 12, "prior_replayed_rows": 48,
                "panels": 2, "windows_per_panel": 160, "parents_per_panel": 10,
                "position_and_rotation_error_pairs": 96 * 160 * 25, "execution_files": 196},
            "costs": {"execution_wall_seconds": wall, "measured_prediction_calls_seconds": timing,
                "support_work": work, "scope": "Timed calls are nested in execution wall; wall excludes final hashing/write. No training. Timings include deriving mass controls, fitting and forecasting; exclude diagnostic serialization/loading."},
            "limits": ["Fixed-checkpoint development on exposed archives; no independent confirmation or architecture novelty.",
                "All primary metrics retain all windows, parents, fits and 25 horizons; start-zero grouping is descriptive only.",
                "Future recorded applied torques are supplied; no control-utility or causal issued-action claim.",
                "No model, optimizer, native or random calls by this audit; Torch weights_only inspects bound tensor schemas.",
                "Compact support summaries permit bounds, fixed-weight and cross-row mass checks, not reconstruction of learned features, Huber residual weights or solved posteriors.",
                "Huber iteration count and the last-used third weights are source-bound. Correction norms permit float32 final versus float64 derivation rounding.",
                "Prior/full and retained controls are checked against authenticated earlier arrays with declared tolerances; predictions are not regenerated.",
                "Target conversion is independent NumPy Euler arithmetic with rotation tolerance 2e-7; proper rotations require errors <=1e-4.",
                "Families pool squared errors before square roots. References are not replicated as artificial fits.",
                "17 grouped conjunctions retain 1141 comparisons; they are not independent statistical tests.",
                "Earlier failed studies remain failed. Passing would only justify fresh confirmation."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before and members(prior_run) == old_members, "evidence changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "externally_supplied_hashes": external,
            "prior_completed_sha256": protocol["prior_completed_sha256"], "execution_members": before,
            "sources": protocol["sources"], "data_hashes": protocol["data_hashes"], "checkpoints": protocol["checkpoints"],
            "auditor_sha256": sha(__file__), "auditor_dependencies": DEPENDENCIES, "files": members(out),
            "qualification_passed": gate["passed"], "requirements_passed": gate["requirements_passed"], "total_requirements": 17,
            "checks_passed": gate["checks_passed"], "total_checks": 1141,
            "new_model_calls": 0, "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                "wall_seconds": time.perf_counter() - start, "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
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
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "requirements_passed",
                                           "total_requirements", "checks_passed", "total_checks", "wall_seconds")}))
