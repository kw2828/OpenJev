"""Saved-output audit of parent-excluded versus in-sample selector training.

No models, optimizers, native calls or random draws. Frozen audit arithmetic is
loaded from exact hash-bound source; new fold and scalar-certificate checks use
independent NumPy arithmetic. Saved learning trajectories are not retrained.
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
PREVIOUS_AUDITOR_SHA256 = "12711689e03d6dbee6101d96894457832b41c789a471addc60efd3153d292fc6"


def load_pinned(path, digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("pinned auditor source digest")
    module = ModuleType("_pose_crossfit_bound_auditor")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102 - exact hash-pinned bytes
    return module


_previous = load_pinned(ROOT / "scripts/audit_pose_coordination.py", PREVIOUS_AUDITOR_SHA256)
require, sha, read_json, write_json = _previous.require, _previous.sha, _previous.read_json, _previous.write_json
members, finite = _previous.members, _previous.finite
public_pose, error_arrays, reduce_errors = _previous.public_pose, _previous.error_arrays, _previous.reduce_errors
rotation_quality, rotation_angles = _previous.rotation_quality, _previous.rotation_angles
rotation_log, rotation_exp = _previous.rotation_log, _previous.rotation_exp
validate_metrics, pooled, percent = _previous.validate_metrics, _previous.pooled, _previous.percent
SEEDS = (1101, 1202, 1303)
STAGES = ("is", "oof")
TRAIN_VARIANTS = ("is_summary", "is_recurrent", "oof_summary", "oof_recurrent")
CONSTANT_VARIANTS = ("is_constant", "oof_constant", "full_constant")
NEW_VARIANTS = ("fast", "slow", "is_constant", "is_summary", "is_recurrent", "oof_constant", "oof_summary", "oof_recurrent", "full_constant")
VARIANTS = (*_previous.VARIANTS, *NEW_VARIANTS[2:])
REFERENCES, PANELS, ENDPOINTS = _previous.REFERENCES, _previous.PANELS, _previous.ENDPOINTS
ALIASES = {"fast": "decay_huber3", "slow": "gru"}
PRIMARY = "oof_recurrent"
MOTION_TOLERANCE = {"rtol": 1e-4, "atol": 1e-7}


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
    require(len(checks) == 1921, "gate count")
    grouped = []
    for panel in PANELS:
        for endpoint in ENDPOINTS:
            for category, suffix in (("family", "/family_10percent"), ("paired", "_nonworse"),
                                     ("parents", "/parents_nonworse"), ("leave_one_out", "/leave_out_")):
                selected = [x for x in checks if x["name"].startswith(panel + "/")
                            and f"/{endpoint}/" in x["name"]
                            and (suffix in x["name"] if category == "leave_one_out" else x["name"].endswith(suffix))
                            and ("/pair_" in x["name"] if category == "paired" else True)]
                require(len(selected) == {"family": 32, "paired": 96, "parents": 32, "leave_one_out": 320}[category],
                        "grouped gate membership")
                grouped.append({"name": f"{panel}/{endpoint}/{category}",
                                "passed": all(x["passed"] for x in selected),
                                "comparisons_passed": sum(x["passed"] for x in selected),
                                "total_comparisons": len(selected),
                                "comparison_names": [x["name"] for x in selected]})
    grouped.append({"name": "latency_vs_gru", "passed": checks[-1]["passed"],
                    "comparisons_passed": int(checks[-1]["passed"]), "total_comparisons": 1,
                    "comparison_names": [checks[-1]["name"]]})
    require(len(grouped) == 17 and sum(x["total_comparisons"] for x in grouped) == 1921, "group count")
    return families, references, latency, comparisons, {"passed": all(x["passed"] for x in checks),
        "requirements_passed": sum(x["passed"] for x in grouped), "total_requirements": 17,
        "requirements": grouped, "checks_passed": sum(x["passed"] for x in checks),
        "total_checks": 1921, "checks": checks,
        "components": {"family": 128, "paired": 384, "parent_counts": 128, "leave_one_parent_out": 1280, "latency": 1}}


def eligible_indices(ids, fold):
    require(type(fold) is int and fold in range(3) and ids.shape == (720, 2)
            and ids.dtype == np.int64 and len(np.unique(ids, axis=0)) == 720,
            "fold identifiers")
    require(np.array_equal(np.unique(ids[:, 0]), np.arange(30))
            and all(np.sum(ids[:, 0] == p) == 24 for p in range(30)), "whole training parents")
    result = np.flatnonzero(ids[:, 0] % 3 != fold)
    require(len(result) == 480 and len(np.unique(ids[result, 0])) == 20, "excluded parent coverage")
    return result


def fold_statistics(p, rotation, actions, indices):
    """Reconstruct eligible-only preprocessing; no held-out row enters a reduction."""
    require(p.shape == (720, 57, 3) and rotation.shape == (720, 57, 3, 3)
            and actions.shape == (720, 56, 40) and indices.shape == (480,)
            and all(x.dtype == np.float32 and np.isfinite(x).all() for x in (p, rotation, actions)),
            "fold preprocessing shapes")
    a = actions[indices].astype(np.float64)
    mean = a.mean(axis=(0, 1)).astype(np.float32)
    std = np.maximum(a.std(axis=(0, 1), ddof=0), 1e-5).astype(np.float32)
    dp = (p[indices, 1:] - p[indices, :-1]).astype(np.float64)
    r = rotation[indices].astype(np.float64)
    w = rotation_log(r[:, 1:] @ np.swapaxes(r[:, :-1], -1, -2))
    motion = np.asarray([max(float(np.sqrt(np.square(x).mean())), 1e-5)
                         for x in (dp, w, np.diff(dp, axis=1), np.diff(w, axis=1))], np.float32)
    return mean, std, motion


def validate_orders(orders, epochs, size):
    require(orders.dtype == np.int64 and orders.shape == (epochs, size)
            and np.array_equal(np.sort(orders, axis=1), np.broadcast_to(np.arange(size), orders.shape)),
            "complete shared minibatch permutations")


def verify_assignment(stage, ids, selected_folds, combined, folds):
    require(stage in STAGES, "cache regime")
    expected = (ids[:, 0] % 3 + (1 if stage == "is" else 0)) % 3
    require(selected_folds.dtype == np.int64 and np.array_equal(selected_folds, expected), "cache fold assignment")
    require(set(combined) == {"fp", "fR", "sp", "sR"} and set(folds) == {0, 1, 2}, "cache fields")
    for key, value in combined.items():
        shape = (len(ids), 25, 3, 3) if key.endswith("R") else (len(ids), 25, 3)
        require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(), "combined cache shape")
        if key.endswith("R"):
            rotation_quality(value)
        for fold in range(3):
            mask = expected == fold
            require(np.array_equal(value[mask], folds[fold][key][mask]), "exact fold forecast selection")
    return {"regime": stage, "windows": len(ids), "per_expert_windows": [int(np.sum(expected == k)) for k in range(3)],
            "experts_excluded_queried_parent": stage == "oof"}


def close_number(actual, expected, label):
    finite(actual, label)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11), label)


def projected_rotation_problem(fr, sr, tr):
    """Independent matrix Rodrigues path, rather than the fitter's quaternion path."""
    projected, stats = [], {}
    for name, original in zip(("fast", "slow", "target"), (fr, sr, tr), strict=True):
        raw = np.asarray(original, np.float64).reshape(-1, 3, 3)
        require(np.isfinite(raw).all(), "constant rotation inputs")
        u, singular, vh = np.linalg.svd(raw)
        signs = np.where(np.linalg.det(u @ vh) < 0, -1., 1.)
        fix = np.broadcast_to(np.eye(3), raw.shape).copy(); fix[:, 2, 2] = signs
        r = u @ fix @ vh
        projected.append(r)
        stats[name] = {"matrices": len(r), "max_frobenius_change": float(np.linalg.norm(r - raw, axis=(1, 2)).max()),
            "min_input_singular_value": float(singular.min()), "reflections_corrected": int(np.sum(signs < 0)),
            "max_orthogonality_error": float(np.abs(r @ r.swapaxes(-1, -2) - np.eye(3)).max()),
            "max_determinant_error": float(np.abs(np.linalg.det(r) - 1).max())}
    fast, slow, target = projected
    vector = rotation_log(fast @ slow.swapaxes(-1, -2)); speed = np.linalg.norm(vector, axis=-1)
    axis = np.divide(vector, speed[:, None], out=np.zeros_like(vector), where=speed[:, None] > 0)
    x, y, z = axis.T; zero = np.zeros_like(x)
    cross = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    e0 = slow @ target.swapaxes(-1, -2); e1 = cross @ e0; e2 = cross @ e1
    def terms(e):
        return np.trace(e, axis1=-2, axis2=-1), .5 * np.stack((e[:, 2, 1] - e[:, 1, 2],
            e[:, 0, 2] - e[:, 2, 0], e[:, 1, 0] - e[:, 0, 1]), -1)
    ts = [terms(e) for e in (e0, e1, e2)]
    def distances(alpha):
        if alpha in (0., 1.):
            e = (slow if alpha == 0 else fast) @ target.swapaxes(-1, -2)
            trace, skew = terms(e)
        else:
            sine, versine = np.sin(alpha * speed), 2 * np.sin(alpha * speed / 2) ** 2
            trace = ts[0][0] + sine * ts[1][0] + versine * ts[2][0]
            skew = ts[0][1] + sine[:, None] * ts[1][1] + versine[:, None] * ts[2][1]
        return np.arctan2(np.linalg.norm(skew, axis=1), np.clip((trace - 1) / 2, -1., 1.))
    return distances, speed, stats


def verify_constant(record, fp, fr, sp, sr, tp, tr, *, tolerance=1e-7, max_evaluations=4095):
    require(set(record) == {"version", "alpha", "input_shape", "examples", "position", "rotation"}
            and record["version"] == "pose-constant-v1" and record["input_shape"] == list(fp.shape)
            and record["examples"] == math.prod(fp.shape[:2]), "constant record identity")
    require(fp.ndim == 3 and fp.shape[-1] == 3 and fp.shape == sp.shape == tp.shape
            and fr.shape == sr.shape == tr.shape == (*fp.shape[:-1], 3, 3)
            and all(np.isfinite(x).all() for x in (fp, fr, sp, sr, tp, tr)), "constant input shapes")
    delta, residual = fp.astype(float) - sp.astype(float), tp.astype(float) - sp.astype(float)
    denominator, numerator = float(np.square(delta).sum()), float((delta * residual).sum())
    raw = None if denominator == 0 else numerator / denominator
    alpha_p = .5 if denominator == 0 else float(np.clip(raw, 0, 1))
    position = record["position"]
    require(set(position) == {"numerator", "denominator", "unclipped_alpha", "degenerate", "objective_m2"}
            and position["degenerate"] is (denominator == 0), "position LS schema")
    for key, value in (("numerator", numerator), ("denominator", denominator),
                       ("objective_m2", float(np.square(alpha_p * delta - residual).sum(-1).mean()))):
        close_number(position[key], value, "constant position " + key)
    if raw is None:
        require(position["unclipped_alpha"] is None, "degenerate position")
    else:
        close_number(position["unclipped_alpha"], raw, "position unconstrained alpha")
    rotation = record["rotation"]
    fixed = {"objective": "mean squared geodesic radians on float64 SVD-projected SO(3) inputs",
        "interpolation": "Exp(alpha*Log(Rfast*Rslow.T))*Rslow; principal log, largest-axis positive at exact pi",
        "certificate_scope": "numerical guarded bound for projected interpolation only; not float32 production or formal interval arithmetic",
        "projection_method": "float64 SVD U diag(1,1,det(UVt)) Vt; no input mutation",
        "tolerance": tolerance, "max_evaluations": max_evaluations,
        "angle_guard": 1e-12, "objective_guard": 1e-12,
        "tie_rule": "smallest evaluated alpha among exact equal float64 objectives",
        "queue_rule": "smallest interval lower bound, then left endpoint, then interval id"}
    require(set(rotation) == set(fixed) | {"projection", "call_count", "status", "certified", "alpha", "objective_at_alpha", "upper_bound",
            "lower_bound", "gap", "speed_min", "speed_max", "evaluations", "intervals", "leaves"}
            and all(rotation[k] == v for k, v in fixed.items()), "rotation certificate schema/status")
    distances, speed, projection = projected_rotation_problem(fr, sr, tr)
    require(set(rotation["projection"]) == set(projection), "projection fields")
    for name, stats in projection.items():
        require(set(stats) == set(rotation["projection"][name]), "projection statistic fields")
        for key, value in stats.items():
            close_number(rotation["projection"][name][key], value, "projection " + name + "/" + key)
    close_number(rotation["speed_min"], float(speed.min()), "minimum angular speed")
    close_number(rotation["speed_max"], float(speed.max()), "maximum angular speed")
    evaluations, intervals = rotation["evaluations"], rotation["intervals"]
    n = len(evaluations)
    require(type(rotation["call_count"]) is int and rotation["call_count"] == n and 3 <= n <= max_evaluations
            and n % 2 == 1 and len(intervals) == n - 2, "constant search work coverage")
    for i, evaluation in enumerate(evaluations):
        require(set(evaluation) == {"id", "alpha", "objective", "role", "radius", "lower_bound"}
                and evaluation["id"] == i and 0 <= evaluation["alpha"] <= 1 and 0 <= evaluation["radius"] <= .5,
                "scalar evaluation identity")
        d = distances(evaluation["alpha"])
        close_number(evaluation["objective"], float(np.square(d).mean()), "rotation objective")
        lower = max(0., float(np.square(np.maximum(0., d - 1e-12 - (speed + 1e-12) * evaluation["radius"])).mean()) - 1e-12)
        close_number(evaluation["lower_bound"], lower, "interval numerical lower bound")
        if i < 2:
            require(evaluation["alpha"] == float(i) and evaluation["radius"] == 0
                    and evaluation["role"] == "endpoint", "search endpoints")
    for i, interval in enumerate(intervals):
        require(set(interval) == {"id", "parent", "left", "right", "center_evaluation", "lower_bound", "children"}
                and interval["id"] == i and interval["center_evaluation"] == i + 2
                and 0 <= interval["left"] < interval["right"] <= 1, "interval identity")
        e = evaluations[i + 2]
        require(e["role"] == "interval_center" and e["alpha"] == (interval["left"] + interval["right"]) / 2
                and e["radius"] == (interval["right"] - interval["left"]) / 2
                and interval["lower_bound"] == e["lower_bound"], "interval center/bound binding")
    require(intervals[0]["parent"] is None and intervals[0]["left"] == 0 and intervals[0]["right"] == 1,
            "full interval root")
    leaves = {0}
    for i in range(1, len(intervals), 2):
        chosen = min(leaves, key=lambda k: (intervals[k]["lower_bound"], intervals[k]["left"], k))
        parent = intervals[chosen]; center = evaluations[chosen + 2]["alpha"]
        incumbent = min(e["objective"] for e in evaluations[:i + 2])
        require(incumbent + 1e-12 - parent["lower_bound"] > tolerance, "continued after certification")
        require(parent["children"] == [i, i + 1], "best-bound search branch")
        for child, left, right in ((i, parent["left"], center), (i + 1, center, parent["right"])):
            require(intervals[child]["parent"] == chosen and intervals[child]["left"] == left
                    and intervals[child]["right"] == right, "exhaustive child partition")
        leaves.remove(chosen); leaves.update((i, i + 1))
    ordered = sorted(leaves, key=lambda i: (intervals[i]["left"], i))
    require(rotation["leaves"] == ordered and all(intervals[i]["children"] is None for i in leaves), "final full leaf coverage")
    objective, alpha_r = min((e["objective"], e["alpha"]) for e in evaluations)
    lower = min(intervals[i]["lower_bound"] for i in leaves); upper = objective + 1e-12
    require(rotation["alpha"] == alpha_r and record["alpha"] == [alpha_p, alpha_r]
            and rotation["objective_at_alpha"] == objective and rotation["upper_bound"] == upper
            and rotation["lower_bound"] == lower and rotation["gap"] == upper - lower
            and 0 <= upper - lower, "constant optimum/gap")
    certified = upper - lower <= tolerance
    chosen = min(leaves, key=lambda k: (intervals[k]["lower_bound"], intervals[k]["left"], k))
    node = intervals[chosen]; center = evaluations[chosen + 2]["alpha"]
    status = ("tolerance_reached" if certified else "evaluation_cap" if n + 2 > max_evaluations
              else "floating_interval_limit" if center in (node["left"], node["right"]) else None)
    require(status is not None and rotation["status"] == status and rotation["certified"] is certified,
            "honest terminal constant status")
    discrepancy = None
    if all(x.dtype == np.float32 for x in (fp, fr, sp, sr, tp, tr)):
        coefficients = np.broadcast_to(np.asarray(record["alpha"], np.float32), (len(fp), 2))
        pp, rr, _ = _previous.reconstruct_blend(fp, fr, sp, sr, coefficients)
        errors = error_arrays(pp, rr, tp, tr)
        discrepancy = {"scope": "NumPy reconstruction of float32 output blend, not an actual Torch production evaluation or a production optimum certificate",
            "position_mse_m2": float(errors["position"].mean()), "rotation_mse_rad2": float(errors["rotation"].mean()),
            "position_mse_minus_analytic": float(errors["position"].mean()) - position["objective_m2"],
            "rotation_mse_minus_projected": float(errors["rotation"].mean()) - objective}
    return {"alpha": record["alpha"], "call_count": n, "gap": upper - lower, "certified": certified, "status": status,
            "objective": rotation["objective"], "independent_matrix_arithmetic_atol": 1e-11,
            "float32_blend_diagnostic": discrepancy}


COORDINATION_PROTOCOL = "ab7ed131f6b32f2ec1d15c533c24c79de05939dec9869aaad3e4854a00a62d45"
COORDINATION_COMPLETED = "711251ab051e0415b0b61ee234e5e59a5420b60246bc9e4565d2ec7e48461ac1"
COORDINATION_SUMMARY = "68d8283013c6c93d283a93ef17b1d0e7841113950cde6ce52af05b5dafad7449"
COORDINATION_RECEIPT = "b4ad57a79751cc79c88f1c55dd662fd9c4f45442a884123d4a02e62c4455a67d"
SOURCES = _previous.SOURCES | {"scripts/train_pose_crossfit.py", "scripts/audit_pose_crossfit.py",
    "src/openjev/research/pose_constant.py", "tests/test_pose_constant.py", "tests/test_pose_crossfit_training.py",
    "tests/test_audit_pose_crossfit.py", "research/pose-crossfit-protocol.md"}
GATE = ('oof_recurrent RMSE <=0.90*each positive control on both physical endpoints/panels; '
        'all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out '
        'MSE strictly lower; median full-window latency<=1.5*new slow. All17 groups required; '
        'all9 constant optimizations must also certify their projected-rotation objective gap.')


def expected_members():
    names = {"started.json", "completed.json", "training-completed.json", "train-tokens.npy"}
    for seed in SEEDS:
        names.add(f"gate-{seed}-orders.npy")
        for fold in range(3):
            names.update(f"fold-{fold}-{seed}-{suffix}" for suffix in ("normalization.npz", "orders.npy"))
            names.update(f"fold-{fold}-{v}-{seed}-cache.npz" for v in ("fast", "slow"))
            names.update(f"fold-{fold}-{v}-{seed}{suffix}" for v in ("meta", "gru")
                         for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
        names.update(f"cache-{stage}-{seed}.npz" for stage in STAGES)
        names.update(f"{v}-{seed}{suffix}" for v in TRAIN_VARIANTS
                     for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
        names.update(f"{v}-{seed}-fit.json" for v in CONSTANT_VARIANTS)
    for panel in PANELS:
        names.add(panel + "-targets.npz")
        names.update(f"{panel}-{v}-{seed}{suffix}" for v in NEW_VARIANTS for seed in SEEDS
                     for suffix in ("-predictions.npz", "-evaluation.json"))
    require(len(names) == 288, "internal payload count")
    return names


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "protocol digest")
    p = read_json(experiment / "protocol.json")
    required = {"study": "pose-crossfit-v1", "scope": "exposed-data parent-exclusion training mechanism screen",
        "seeds": list(SEEDS), "train_variants": list(TRAIN_VARIANTS), "constant_variants": list(CONSTANT_VARIANTS),
        "variants": list(NEW_VARIANTS), "panels": list(PANELS), "primary": PRIMARY, "folds": 3,
        "group_rule": "source_id modulo3", "fold_training": "exclude groupk; OOF uses expertk=g; IS uses expertk=(g+1)%3",
        "fold_windows": 480, "fold_epochs": 46, "gate_windows": 720, "gate_epochs": 30,
        "batch": 32, "updates_per_fit": 690, "optimizer": "Adam", "learning_rate": .001, "grad_norm_cap": 1.,
        "expert_fits": 18, "gate_fits": 12, "constant_fits": 9, "optimizer_updates": 20700,
        "constant_tolerance_rad_squared": 1e-7, "constant_max_evaluations": 4095,
        "constant_criterion": "Analytic position LS; global interval search on float64 SO3-projected rotation objective. Production float32 blend unchanged.",
        "normalization": "Fold-only action mean/populationSD per40coords, float64 thenfloat32, floor1e-5; fold-only four motion RMS scales. Gate tokens retain original full-data scales.",
        "action_normalization_tolerance": "exact float32 arrays from NumPy float64 reductions", "motion_scale_tolerance": MOTION_TOLERANCE,
        "context": 32, "tokens": 31, "token_dim": 49, "horizon": 25,
        "position_loss_scale_m": .1, "rotation_loss_scale_rad": .1, "evaluation_rows": 54, "execution_files": 288,
        "warmups_per_row": 3, "timed_windows_per_row": 20,
        "blend_tolerance": {**_previous.BLEND_TOLERANCE, "hard_endpoints": "exact"}, "token_tolerance": _previous.TOKEN_TOLERANCE,
        "expert_replay_tolerance": {"rtol": 1e-6, "atol": 2e-7}, "gate": GATE, "gate_groups": 17, "gate_comparisons": 1921,
        "coordination_protocol_sha256": COORDINATION_PROTOCOL, "coordination_completed_sha256": COORDINATION_COMPLETED,
        "coordination_summary_sha256": COORDINATION_SUMMARY, "coordination_receipt_sha256": COORDINATION_RECEIPT,
        "no_selection": "Fixed parent groups, final checkpoints and budgets; no retries, sweeps, replacements, exclusions or criterion changes."}
    require(all(p[k] == v for k, v in required.items()), "frozen study settings")
    dependencies = {"scripts/audit_pose_coordination.py": PREVIOUS_AUDITOR_SHA256, **_previous.DEPENDENCIES}
    require(set(p["sources"]) == SOURCES and all(p["sources"][k] == v for k, v in dependencies.items())
            and {k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}
            == p["sources"] == {k: sha(ROOT / k) for k in SOURCES}, "source identities")
    require(p["runtime"] == {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
        "numpy": np.__version__, "threads": 1, "platform": platform.platform()}, "runtime identity")
    old = Path(p["coordination_experiment"]); context = _previous.validate_inputs(old, COORDINATION_PROTOCOL)
    require(p["data"] == str(context["data"].resolve()) and p["data_hashes"] == context["protocol"]["data_hashes"]
            and p["checkpoints"] == context["protocol"]["checkpoints"], "unchanged data/deployment experts")
    run = old / "run-01"; old_members = members(run)
    done = read_json(run / "completed.json")
    require(set(old_members) == _previous.expected_members() and old_members["completed.json"]["sha256"] == COORDINATION_COMPLETED
            and done["status"] == "completed" and done["protocol_sha256"] == COORDINATION_PROTOCOL
            and done["files"] == {k: v for k, v in old_members.items() if k != "completed.json"}, "parent completed seal")
    report = Path(p["coordination_report"]); summary = read_json(report / "summary.json"); receipt = read_json(report / "receipt.json")
    report_members = members(report)
    require(set(report_members) == {"summary.json", "window-errors.npz", "receipt.json"}
            and sha(report / "summary.json") == COORDINATION_SUMMARY and sha(report / "receipt.json") == COORDINATION_RECEIPT
            and summary["status"] == receipt["status"] == "completed"
            and summary["execution_completed_sha256"] == receipt["execution_completed_sha256"] == COORDINATION_COMPLETED
            and receipt["execution_members"] == old_members
            and receipt["files"] == {k: v for k, v in report_members.items() if k != "receipt.json"}, "parent audited seal")
    return {"protocol": p, "parent": context, "data": context["data"], "prior_run": run,
            "prior_members": old_members, "prior_summary": summary, "scales": context["scales"]}


def fit_record(run, label, identity, order_file, before, *, expert=False):
    fit = read_json(run / (label + "-fit.json"))
    require(set(fit) == set(identity) | {"updates", "parameters", "training_seconds", "initial_sha256", "checkpoint_sha256", "orders_sha256"}
            and all(fit[k] == v for k, v in identity.items()) and fit["updates"] == 690
            and fit["orders_sha256"] == before[order_file]["sha256"], "paired fit identity")
    weights = _previous._previous._previous.tensor_weights if expert else _previous.tensor_weights
    mode = identity["variant"] if expert else identity["variant"].split('_', 1)[1]
    initial, final = weights(run / (label + "-initial.pt"), mode), weights(run / (label + ".pt"), mode)
    for key, suffix in (("initial_sha256", "-initial.pt"), ("checkpoint_sha256", ".pt")):
        require(fit[key] == before[label + suffix]["sha256"], "fit tensor binding")
    zero_keys = (("prior",) if mode == "meta" else ("readout.weight", "readout.bias")) if expert else (
        ("network.2.weight", "network.2.bias") if mode == "summary" else ("head.weight", "head.bias"))
    require(all(bool((initial[k] == 0).all()) for k in zero_keys), "initial uncorrected/half forecast")
    require(fit["parameters"] == sum(v.numel() for k, v in final.items() if k != "scales"), "parameter accounting")
    loss = np.load(run / (label + "-losses.npy"), allow_pickle=False)
    require(loss.dtype == np.float64 and loss.shape == (690,) and np.isfinite(loss).all() and (loss >= 0).all(), "all recorded updates")
    finite(fit["training_seconds"], "training seconds", positive=True)
    return fit, initial, final


def audit_training(run, done, context, before):
    boundary = read_json(run / "training-completed.json")
    fields = ("expert_fits", "gate_fits", "constant_fits", "cache_costs")
    require(set(boundary) == {*fields, "evaluation_started"} and boundary["evaluation_started"] is False
            and all(boundary[k] == done[k] for k in fields), "training completion boundary")
    p, r, a, ids = _previous.load_public(context["data"], "train")
    tokens = np.load(run / "train-tokens.npy", allow_pickle=False)
    expected = _previous.public_tokens(p[:, :32], r[:, :32], a[:, :31], context["scales"])
    require(tokens.shape == (720, 31, 49) and tokens.dtype == np.float32 and np.isfinite(tokens).all()
            and (np.abs(tokens) <= 1).all() and np.allclose(tokens, expected, **_previous.TOKEN_TOLERANCE), "gate public tokens")
    require(np.array_equal(tokens, np.load(context["prior_run"] / "train-tokens.npy", allow_pickle=False)), "unchanged gate context")
    indices = {k: eligible_indices(ids, k) for k in range(3)}
    expected_stats = {k: fold_statistics(p, r, a, indices[k]) for k in range(3)}
    experts, gates, constants, cache_checks, normal_checks, cost_rows = {}, {}, {}, {}, {}, []
    for seed in SEEDS:
        folds, initial_experts, relative_orders, gate_initials = {}, {}, None, {}
        for fold in range(3):
            stem = f"fold-{fold}-{seed}"; nf = stem + "-normalization.npz"; of = stem + "-orders.npy"
            with np.load(run / nf, allow_pickle=False) as data:
                require(set(data.files) == {"train_indices", "action_mean", "action_scale", "motion_scales"}, "normalizer fields")
                normal = {k: data[k] for k in data.files}
            em, es, motion = expected_stats[fold]
            require(normal["train_indices"].dtype == np.int64 and np.array_equal(normal["train_indices"], indices[fold])
                    and normal["action_mean"].dtype == normal["action_scale"].dtype == normal["motion_scales"].dtype == np.float32
                    and np.array_equal(normal["action_mean"], em) and np.array_equal(normal["action_scale"], es)
                    and normal["motion_scales"].shape == (4,) and np.allclose(normal["motion_scales"], motion, **MOTION_TOLERANCE),
                    "eligible-only normalization")
            orders = np.load(run / of, allow_pickle=False)
            require(orders.dtype == np.int64 and orders.shape == (46, 480)
                    and np.array_equal(np.sort(orders, axis=1), np.broadcast_to(indices[fold], orders.shape)), "fold batch exclusion")
            mapped = np.searchsorted(indices[fold], orders)
            if relative_orders is not None:
                require(np.array_equal(mapped, relative_orders), "paired fold relative orders")
            relative_orders = mapped
            normal_checks[stem] = {"eligible_windows": 480, "excluded_group": fold,
                "motion_max_abs_error": float(np.max(np.abs(normal["motion_scales"].astype(float) - motion.astype(float))))}
            for variant in ("meta", "gru"):
                label = f"fold-{fold}-{variant}-{seed}"
                fit, initial, final = fit_record(run, label, {"kind": "expert", "variant": variant,
                    "seed": seed, "fold": fold, "normalization_sha256": before[nf]["sha256"]}, of, before, expert=True)
                require(np.array_equal(initial["scales"].numpy(), normal["motion_scales"])
                        and torch.equal(initial["scales"], final["scales"]), "fold scale tensor identity")
                if variant in initial_experts:
                    require(all(torch.equal(v, initial_experts[variant][k]) for k, v in initial.items() if k != "scales"),
                            "paired expert initialization")
                initial_experts[variant] = initial
                require(fit == done["expert_fits"][len(experts)], "expert fit execution order")
                experts[label] = fit
            folds[fold] = {}
            for expert, keys in (("fast", ("fp", "fR")), ("slow", ("sp", "sR"))):
                pp, rr = _previous.load_prediction(run / f"fold-{fold}-{expert}-{seed}-cache.npz")
                require(pp.shape == (720, 25, 3) and rr.shape == (720, 25, 3, 3)
                        and pp.dtype == rr.dtype == np.float32 and np.isfinite(pp).all(), "fold forecast cache")
                rotation_quality(rr); folds[fold].update(zip(keys, (pp, rr), strict=True))
            cost_rows.append({"kind": "fold", "seed": seed, "fold": fold})
        go = f"gate-{seed}-orders.npy"; validate_orders(np.load(run / go, allow_pickle=False), 30, 720)
        caches = {}
        for stage in STAGES:
            cf = f"cache-{stage}-{seed}.npz"
            with np.load(run / cf, allow_pickle=False) as data:
                require(set(data.files) == {"fp", "fR", "sp", "sR", "ids", "folds"}
                        and data["ids"].dtype == np.int64 and np.array_equal(data["ids"], ids), "combined cache identities")
                cached = {k: data[k] for k in ("fp", "fR", "sp", "sR")}
                cache_checks[f"{stage}-{seed}"] = verify_assignment(stage, ids, data["folds"], cached, folds)
            caches[stage] = tuple(cached[k] for k in ("fp", "fR", "sp", "sR"))
            cost_rows.append({"kind": "assembly", "seed": seed, "regime": stage})
            for kind in ("summary", "recurrent"):
                variant = stage + "_" + kind; label = f"{variant}-{seed}"
                fit, initial, _final = fit_record(run, label, {"kind": "selector", "variant": variant,
                    "seed": seed, "cache_sha256": before[cf]["sha256"]}, go, before)
                if kind in gate_initials:
                    require(all(torch.equal(v, gate_initials[kind][k]) for k, v in initial.items()), "IS/OOF initial pairing")
                gate_initials[kind] = initial
                require(fit == done["gate_fits"][len(gates)], "selector fit execution order")
                gates[label] = fit
        caches["full"] = tuple(x for expert in ("fast", "slow")
                                for x in _previous.load_prediction(context["prior_run"] / f"train-{expert}-{seed}.npz"))
        for stage in (*STAGES, "full"):
            variant = stage + "_constant"; label = f"{variant}-{seed}"; file = label + "-fit.json"
            record = read_json(run / file); row = done["constant_fits"][len(constants)]
            require(set(record) == {"variant", "seed", "regime", "optimization", "seconds"}
                    and record["variant"] == variant and record["seed"] == seed and record["regime"] == stage
                    and row == {"variant": variant, "seed": seed, "fit_sha256": before[file]["sha256"], "seconds": record["seconds"]},
                    "constant training-only input/record identity")
            finite(record["seconds"], "constant seconds", positive=True)
            constants[label] = {**row, **verify_constant(record["optimization"], *caches[stage], p[:, 32:], r[:, 32:])}
    require(len(experts) == len(done["expert_fits"]) == 18 and len(gates) == len(done["gate_fits"]) == 12
            and len(constants) == len(done["constant_fits"]) == 9 and len(done["cache_costs"]) == len(cost_rows) == 15,
            "complete training coverage")
    for row, identity in zip(done["cache_costs"], cost_rows, strict=True):
        require(set(row) == set(identity) | {"seconds"} and all(row[k] == v for k, v in identity.items()), "cache cost identity")
        finite(row["seconds"], "cache seconds", positive=True)
    return {"experts": experts, "gates": gates, "constants": constants, "normalization": normal_checks,
            "cache_assignment": cache_checks, "token_max_abs_error": float(np.max(np.abs(tokens.astype(float) - expected.astype(float))))}


def apply_certification_gate(gate, constants):
    require(len(constants) == 9 and all(type(x["certified"]) is bool for x in constants.values()), "nine constant statuses")
    passed = all(x["certified"] for x in constants.values())
    return {**gate, "numerical_comparisons_passed": gate["passed"], "constant_certification_pass": passed,
            "passed": gate["passed"] and passed}


def replay_expert(p, rotation, context, panel, variant, seed):
    require(variant in ALIASES, "expert replay alias")
    # Canonical support outputs have p/R only; coordination also saves alpha.
    path = context["parent"]["prior_run"] / f"{panel}-{ALIASES[variant]}-{seed}-predictions.npz"
    return _previous._previous.validate_replay(p, rotation, path)


def audit(experiment, out, *, protocol_sha256, completed_sha256):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    try:
        run = experiment / "run-01"
        require(sha(run / "completed.json") == completed_sha256, "external completion digest")
        context = validate_inputs(experiment, protocol_sha256); protocol = context["protocol"]
        before = members(run); require(set(before) == expected_members(), "exact completed payload membership")
        done = read_json(run / "completed.json")
        require(set(done) == {"status", "expert_fits", "gate_fits", "constant_fits", "cache_costs", "rows",
            "new_optimizer_updates", "wall_seconds", "protocol_sha256", "files"}
            and done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256
            and done["new_optimizer_updates"] == 20700 and len(done["rows"]) == 54
            and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "producer seal")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "execution start binding")
        training = audit_training(run, done, context, before)
        rows, active_rows, window_errors, replay, blends, row_index = {}, {}, {}, {}, {}, 0
        for panel in PANELS:
            ep, er, _a, expected_ids = _previous.load_public(context["data"], panel)
            with np.load(run / (panel + "-targets.npz"), allow_pickle=False) as target:
                require(set(target.files) == {"p", "R", "ids"}, "target fields")
                tp, tr, ids = target["p"], target["R"], target["ids"]
            require(tp.shape == (160, 25, 3) and tr.shape == (160, 25, 3, 3) and tp.dtype == tr.dtype == np.float32
                    and ids.dtype == np.int64 and np.array_equal(ids, expected_ids) and np.array_equal(tp, ep[:, 32:])
                    and np.allclose(tr, er[:, 32:], rtol=0, atol=2e-7), "public target provenance")
            with np.load(context["prior_run"] / (panel + "-targets.npz"), allow_pickle=False) as old:
                require(all(np.array_equal(old[k], v) for k, v in (("p", tp), ("R", tr), ("ids", ids))), "unchanged development targets")
            rows[panel], active_rows[panel], replay[panel], blends[panel] = {}, {}, {}, {}
            window_errors[panel + "__ids"] = ids
            for variant in (*_previous.VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in _previous.VARIANTS else (None,):
                    label = variant if seed is None else f"{variant}-{seed}"; prefix = panel + "-" + label
                    inherited = context["prior_summary"]["rows"][panel][label]
                    source = (context["parent"]["prior_run"] if variant in (*_previous._previous.VARIANTS, *REFERENCES)
                              else context["prior_run"])
                    pp, rr = _previous.load_prediction(source / (prefix + "-predictions.npz"), alpha=source == context["prior_run"])[:2]
                    require(pp.dtype == rr.dtype == (np.float64 if variant == "ridge16" else np.float32), "inherited forecast precision")
                    error = error_arrays(pp, rr, tp, tr); calculated = reduce_errors(error, ids)
                    require(calculated == inherited["errors"], "inherited numerical errors")
                    old_row = read_json(source / (prefix + "-evaluation.json")); validate_metrics(old_row["metrics"], calculated, rr)
                    if variant in ("decay_huber3", "gru"):
                        alias = "fast" if variant == "decay_huber3" else "slow"
                        timing_row = read_json(context["prior_run"] / f"{panel}-{alias}-{seed}-evaluation.json")
                    else:
                        timing_row = old_row
                    require(timing_row["latency_ms"] == inherited["latency_ms"], "inherited latency binding")
                    rows[panel][label] = {"variant": variant, "seed": seed, "errors": calculated,
                        "latency_ms": inherited["latency_ms"], "checkpoint_sha256": inherited["checkpoint_sha256"],
                        "latency_scope": "inherited descriptive only"}
                    for endpoint, value in error.items():
                        window_errors[prefix + "__" + endpoint] = value
            experts = {seed: {v: _previous.load_prediction(run / f"{panel}-{v}-{seed}-predictions.npz", alpha=True)
                              for v in ("fast", "slow")} for seed in SEEDS}
            for variant in NEW_VARIANTS:
                for seed in SEEDS:
                    label = f"{variant}-{seed}"; prefix = panel + "-" + label
                    row = read_json(run / (prefix + "-evaluation.json"))
                    require(set(row) == {"panel", "variant", "seed", "checkpoint_sha256", "expert_sha256", "metrics", "latency_ms"}
                            and row == done["rows"][row_index] and row["panel"] == panel and row["variant"] == variant
                            and row["seed"] == seed, "evaluation identity/order")
                    row_index += 1
                    digest = (training["gates"][label]["checkpoint_sha256"] if variant in TRAIN_VARIANTS
                              else training["constants"][label]["fit_sha256"] if variant in CONSTANT_VARIANTS else None)
                    require(row["checkpoint_sha256"] == digest and row["expert_sha256"] == {
                        n: protocol["checkpoints"][f"{n}-{seed}"]["sha256"] for n in ("meta", "gru")}, "deployment checkpoint binding")
                    pp, rr, alpha = _previous.load_prediction(run / (prefix + "-predictions.npz"), alpha=True)
                    fp, fr, _ = experts[seed]["fast"]; sp, sr, _ = experts[seed]["slow"]
                    blends[panel][label] = _previous.validate_blend(variant if variant in ALIASES else "recurrent",
                        pp, rr, alpha, fp, fr, sp, sr)
                    if variant in CONSTANT_VARIANTS:
                        expected_alpha = np.asarray(training["constants"][label]["alpha"], np.float32)
                        require(np.array_equal(alpha, np.broadcast_to(expected_alpha, (160, 2))), "trained constant deployment alpha")
                    error = error_arrays(pp, rr, tp, tr); calculated = reduce_errors(error, ids)
                    validate_metrics(row["metrics"], calculated, rr)
                    samples = np.asarray(row["latency_ms"], float)
                    require(samples.shape == (20,) and np.isfinite(samples).all() and (samples > 0).all(), "fresh latency coverage")
                    active = {"variant": variant, "seed": seed, "errors": calculated, "latency_ms": row["latency_ms"],
                        "checkpoint_sha256": digest, "expert_sha256": row["expert_sha256"], "latency_scope": "fresh complete forecast"}
                    active_rows[panel][label] = active
                    for endpoint, value in error.items():
                        window_errors[prefix + "__" + endpoint] = value
                    if variant in ALIASES:
                        alias = f"{ALIASES[variant]}-{seed}"
                        replay[panel][label] = replay_expert(pp, rr, context, panel, variant, seed)
                        rows[panel][alias]["latency_ms"] = row["latency_ms"]
                        rows[panel][alias]["latency_scope"] = "fresh alias " + variant
                    else:
                        rows[panel][label] = active
        require(sum(map(len, rows.values())) == 174 and sum(map(len, replay.values())) == 12, "distinct control coverage")
        families, references, latency, contrasts, gate = aggregate(rows)
        gate = apply_certification_gate(gate, training["constants"])
        certificate_pass = gate["constant_certification_pass"]
        for variant in latency:
            latency[variant]["scope"] = ("fresh complete forecast" if variant in ("decay_huber3", "gru", *NEW_VARIANTS[2:])
                                          else "inherited descriptive only")
        protocol_contrasts = {}
        for panel in PANELS:
            protocol_contrasts[panel] = {}
            for kind in ("constant", "summary", "recurrent"):
                control = f"is_{kind}"; candidate = f"oof_{kind}"
                protocol_contrasts[panel][kind] = {e: {
                    "rmse_improvement_percent": percent(families[panel][candidate][e]["rmse"], families[panel][control][e]["rmse"]),
                    "paired_mse_difference": [rows[panel][f"{candidate}-{s}"]["errors"][e]["mse"]
                        - rows[panel][f"{control}-{s}"]["errors"][e]["mse"] for s in SEEDS]}
                    for e in ENDPOINTS}
        wall = finite(done["wall_seconds"], "execution wall seconds", positive=True)
        expert_seconds = sum(x["training_seconds"] for x in training["experts"].values())
        gate_seconds = sum(x["training_seconds"] for x in training["gates"].values())
        constant_seconds = sum(x["seconds"] for x in training["constants"].values())
        cache_seconds = sum(x["seconds"] for x in done["cache_costs"])
        timed_seconds = sum(sum(x["latency_ms"]) for group in active_rows.values() for x in group.values()) / 1000
        require(expert_seconds + gate_seconds + constant_seconds + cache_seconds + timed_seconds <= wall + 1e-6,
                "nested work timing")
        summary = {"status": "completed", "study": "pose-crossfit-v1", "scope": protocol["scope"],
            "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "parent_summary_sha256": COORDINATION_SUMMARY, "training": training,
            "rows": rows, "active_rows": active_rows, "families": families, "references": references,
            "latency": latency, "contrasts": contrasts, "oof_vs_is": protocol_contrasts, "continuation_gate": gate,
            "constant_certification_pass": certificate_pass, "expert_replay": replay, "blend_reconstruction": blends,
            "counts": {"expert_fits": 18, "selector_fits": 12, "adam_updates": 20700, "constant_fits": 9,
                "constant_objective_evaluations": sum(x["call_count"] for x in training["constants"].values()),
                "new_evaluation_rows": 54, "inherited_rows": 132, "aliased_new_rows": 12, "distinct_combined_rows": 174,
                "controls": 32, "new_execution_files": 288, "windows_per_panel": 160, "parents_per_panel": 10},
            "costs": {"execution_wall_seconds": wall, "fold_expert_training_seconds": expert_seconds,
                "selector_training_seconds": gate_seconds, "constant_optimization_seconds": constant_seconds,
                "cache_seconds": cache_seconds, "measured_active_calls_seconds": timed_seconds,
                "scope": "Components are nested, not additional to total. Full forecast calls include both deployment experts for mixtures. Earlier full-data training remains inherited. Final sealing is outside recorded wall."},
            "limits": ["Exposed development data and a selector-training mechanism screen, not new architecture or fresh confirmation.",
                "Fold preprocess statistics and actual eligible-index permutations are checked. Neural training, cached forecasts and learned selector alpha generation remain source-bound, not rerun.",
                "IS/OOF use the same cases and same folded experts equally but differing model-training composition; a contrast does not identify training membership as the sole cause.",
                "Both selectors deploy the unchanged full30-parent experts, leaving a20-to30-parent expert-distribution shift.",
                "Constant certificates cover guarded float64 SO3-projected objectives, not exact float32 deployment optimality or formal interval arithmetic. Capped searches remain descriptive and force continuation failure.",
                "Parent cache and fixed constants use training targets only. All original development cases and failed earlier gates remain retained.",
                "Same frozen experts evolve privately; blended poses never feed back. Recorded future applied torques remain inputs.",
                "No model, optimizer, native or random calls were made by this audit."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before and members(context["prior_run"]) == context["prior_members"], "evidence changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "parent_summary_sha256": COORDINATION_SUMMARY, "execution_members": before, "sources": protocol["sources"],
            "data_hashes": protocol["data_hashes"], "expert_checkpoints": protocol["checkpoints"],
            "auditor_sha256": sha(__file__), "previous_auditor_sha256": PREVIOUS_AUDITOR_SHA256, "files": members(out),
            "qualification_passed": gate["passed"], "constant_certification_pass": certificate_pass,
            "requirements_passed": gate["requirements_passed"], "total_requirements": 17,
            "checks_passed": gate["checks_passed"], "total_checks": 1921,
            "new_model_calls": 0, "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                "wall_seconds": time.perf_counter() - start, "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            error.add_note(f"Could not preserve audit failure: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256, completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "constant_certification_pass",
        "requirements_passed", "total_requirements", "checks_passed", "total_checks", "wall_seconds")}))
