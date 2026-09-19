"""Saved-output audit of retrospective context-probe blending, without inference."""
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
CAPACITY_AUDITOR_SHA256 = "5432ebed789519cfecea0038d1782735943dd442a7954fc2b3292d3d1e4040b4"


def load_pinned(path, digest):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("pinned audit source identity")
    module = ModuleType("_pose_probe_pinned_audit")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102 - exactly pinned local bytes
    return module


_capacity = load_pinned(ROOT / "scripts/audit_pose_capacity.py", CAPACITY_AUDITOR_SHA256)
_parent = _capacity._audit
_geometry = _parent._previous
require, sha, read_json, write_json = _parent.require, _parent.sha, _parent.read_json, _parent.write_json
members, finite, error_arrays, reduce_errors = _parent.members, _parent.finite, _parent.error_arrays, _parent.reduce_errors
pooled, percent = _parent.pooled, _parent.percent
PANELS, SEEDS, ENDPOINTS = _parent.PANELS, _parent.SEEDS, _parent.ENDPOINTS
NEW_VARIANTS = ("fast", "slow", "half", "probe_half", "probe_inverse", "probe_fit")
ALIASES = {"fast": "decay_huber3", "slow": "gru", "half": "half"}
VARIANTS = (*_parent.VARIANTS, "probe_half", "probe_inverse", "probe_fit")
REFERENCES, PRIMARY = _parent.REFERENCES, "probe_fit"
GRID = np.arange(17, dtype=np.float64) / 16
LOSS_TOLERANCE = {"rtol": 1e-10, "atol": 1e-11}
CAPACITY_PROTOCOL = "843c3bc1744e8927b536cd85d3f1d54b0f2a4a5ed854fb52815a4109fbdfc237"
CAPACITY_COMPLETED = "975cf6434c6cc406ec6fc3ce7d1dc9e7e093fc1ac92fb660d0c7339936bb1198"
CAPACITY_SUMMARY = "b85c4537118b72bec031a925f8a433a34629cdeea25d4e1ba883017263b7fb3a"
CAPACITY_RECEIPT = "a8f072f3afa6f301df3bf5e324fa02e96f62cfe1be1e96859d4bcf821ba60714"
SOURCES = _capacity.SOURCES | {"src/openjev/research/pose_probe_blend.py", "tests/test_pose_probe_blend.py",
    "scripts/run_pose_probe_blend.py", "scripts/audit_pose_probe_blend.py", "tests/test_audit_pose_probe_blend.py",
    "research/pose-probe-blend-protocol.md"}


def probe_arithmetic(fp, fr, sp, sr, context_p, context_r, fallback):
    """Only context32 is admitted. The five completed targets are27..31."""
    batch = len(fp)
    require(fp.shape == sp.shape == (batch, 5, 3) and fr.shape == sr.shape == (batch, 5, 3, 3)
            and context_p.shape == (batch, 32, 3) and context_r.shape == (batch, 32, 3, 3)
            and all(x.dtype == np.float32 and np.isfinite(x).all() for x in (fp, fr, sp, sr, context_p, context_r)),
            "probe/context shape, precision or finiteness")
    fallback = np.asarray(fallback)
    require(fallback.shape == (2,) and fallback.dtype == np.float32 and np.isfinite(fallback).all()
            and (fallback >= 0).all() and (fallback <= 1).all(), "training fallback alpha")
    tp, tr = context_p[:, 27:32], context_r[:, 27:32]
    errors = np.stack([np.stack([x[e].mean(1) for e in ENDPOINTS], -1)
                       for x in (error_arrays(fp, fr, tp, tr), error_arrays(sp, sr, tp, tr))], 1)
    delta, residual = fp.astype(float) - sp.astype(float), tp.astype(float) - sp.astype(float)
    denominator = np.square(delta).sum(-1).mean(1)
    numerator = (delta * residual).sum(-1).mean(1)
    threshold = np.maximum(np.finfo(np.float64).tiny,
                           np.finfo(np.float32).eps ** 2 * errors[:, :, 0].max(1))
    fallback_mask = denominator <= threshold
    alpha_p = np.divide(numerator, denominator, out=np.full_like(numerator, float(fallback[0])), where=~fallback_mask)
    alpha_p = np.clip(alpha_p, 0, 1).astype(np.float32)
    scale = errors.max(1)
    normalized = np.divide(errors, scale[:, None], out=np.zeros_like(errors), where=scale[:, None] > 0)
    total = normalized.sum(1)
    inverse = np.divide(normalized[:, 1], total,
                        out=np.broadcast_to(fallback.astype(float), total.shape).copy(), where=total > 0).astype(np.float32)
    losses = []
    for coefficient in GRID:
        alpha = np.full((batch, 2), coefficient, np.float32)
        _, rr, _ = _geometry.reconstruct_blend(fp, fr, sp, sr, alpha)
        losses.append(error_arrays(fp, rr, tp, tr)["rotation"].mean(1))
    return {"errors": errors, "position_numerator": numerator, "position_denominator": denominator,
            "position_threshold": threshold, "position_fallback": fallback_mask,
            "position_alpha": alpha_p, "inverse_alpha": inverse, "rotation_grid_losses": np.stack(losses, -1)}


def checked_grid_alpha(saved_losses, reconstructed):
    require(saved_losses.dtype == np.float64 and saved_losses.shape == reconstructed.shape
            and saved_losses.ndim == 2 and saved_losses.shape[1] == 17
            and np.isfinite(saved_losses).all() and (saved_losses >= 0).all()
            and np.allclose(saved_losses, reconstructed, **LOSS_TOLERANCE), "reconstructed rotation grid losses")
    # Producer losses retain its float32 blend rounding; exact ties choose the
    # first ascending grid point. Independent matrix reconstruction is tolerant.
    return GRID[np.argmin(saved_losses, axis=1)].astype(np.float32)


def verify_probe(payload, variant, context_p, context_r, fallback, alpha):
    require(variant in ("probe_half", "probe_inverse", "probe_fit"), "probe variant")
    base = {"fast_p", "fast_R", "slow_p", "slow_R", "errors", "alpha"}
    grid_fields = {"position_numerator", "position_denominator", "position_threshold", "position_fallback",
                   "grid_coefficients", "grid_rotations", "grid_mse", "grid_index", "fitted_alpha"}
    require(set(payload) == base | ({"inverse_zero_fallback"} if variant == "probe_inverse" else grid_fields),
            "probe evidence fields")
    fp, fr, sp, sr = (payload[k] for k in ("fast_p", "fast_R", "slow_p", "slow_R"))
    expected = probe_arithmetic(fp, fr, sp, sr, context_p, context_r, fallback)
    batch = len(fp)
    require(payload["errors"].dtype == np.float64 and payload["errors"].shape == (batch, 2, 2)
            and np.allclose(payload["errors"], expected["errors"], **LOSS_TOLERANCE), "completed past probe errors")
    require(payload["alpha"].dtype == np.float32 and payload["alpha"].shape == (batch, 2)
            and np.array_equal(payload["alpha"], alpha), "probe/prediction alpha binding")
    result = {"error_max_abs": float(np.abs(payload["errors"] - expected["errors"]).max())}
    if variant == "probe_inverse":
        zero = payload["inverse_zero_fallback"]
        require(zero.dtype == np.bool_ and zero.shape == (batch, 2)
                and np.array_equal(zero, expected["errors"].sum(1) == 0), "inverse zero fallback")
        expected_alpha = expected["inverse_alpha"]
        result["inverse_zero_fallback_counts"] = zero.sum(0).tolist()
    else:
        for key in ("position_numerator", "position_denominator", "position_threshold"):
            actual = payload[key]
            require(actual.dtype == np.float64 and actual.shape == (batch,)
                    and np.allclose(actual, expected[key], **LOSS_TOLERANCE), "probe LS " + key)
        tiny = payload["position_fallback"]
        require(tiny.dtype == np.bool_ and tiny.shape == (batch,)
                and np.array_equal(tiny, expected["position_fallback"]), "probe LS fallback")
        grid = payload["grid_coefficients"]
        require(grid.dtype == np.float32 and np.array_equal(grid, GRID.astype(np.float32)), "all17 fixed grid coefficients")
        rotations = payload["grid_rotations"]
        require(rotations.dtype == np.float32 and rotations.shape == (batch, 17, 5, 3, 3)
                and np.isfinite(rotations).all(), "production grid rotation shape/dtype")
        grid_loss, maximum = [], 0.
        for i, coefficient in enumerate(GRID):
            aa = np.full((batch, 2), coefficient, np.float32)
            _, rr, _ = _geometry.reconstruct_blend(fp, fr, sp, sr, aa)
            saved = rotations[:, i]
            require(np.allclose(saved, rr, **_geometry.BLEND_TOLERANCE), "production grid geometry")
            if i in (0, 16):
                require(np.array_equal(saved, sr if i == 0 else fr), "exact grid endpoints")
            maximum = max(maximum, float(np.abs(saved.astype(float) - rr.astype(float)).max()))
            grid_loss.append(error_arrays(fp, saved, context_p[:, 27:32], context_r[:, 27:32])["rotation"].mean(1))
        loss = np.stack(grid_loss, -1)
        selected = checked_grid_alpha(payload["grid_mse"], loss)
        index = payload["grid_index"]
        require(index.dtype == np.int64 and index.shape == (batch,)
                and np.array_equal(index, np.argmin(payload["grid_mse"], 1)), "exact grid first minimum")
        fitted = payload["fitted_alpha"]
        require(fitted.dtype == np.float32 and fitted.shape == (batch, 2)
                and np.array_equal(fitted[:, 1], selected)
                and np.allclose(fitted[:, 0], expected["position_alpha"], rtol=1e-6, atol=2e-7), "fitted probe coefficients")
        expected_alpha = np.full((batch, 2), .5, np.float32) if variant == "probe_half" else fitted
        result.update(grid_geometry_max_abs=maximum, grid_loss_max_abs=float(np.abs(payload["grid_mse"] - loss).max()),
                      position_fallback_count=int(tiny.sum()), grid_index_counts=np.bincount(index, minlength=17).tolist())
    require(np.allclose(alpha, expected_alpha, rtol=1e-6, atol=2e-7), "actual probe coefficients")
    if variant in ("probe_half", "probe_fit"):
        require(np.array_equal(alpha, expected_alpha), "exact grid/fitted coefficient use")
    return result


def expected_work(variant, batch):
    probed = variant.startswith("probe_")
    nf, ns = batch * (variant != "slow"), batch * (variant != "fast")
    return {"scope": "successful calls; no timing or partial-failure guarantee",
        "fast_support_rows": 25 * batch * probed + 30 * nf,
        "fast_feature_samples": 30 * batch * probed + 55 * nf,
        "fast_batched_solve_calls": 3 * probed + 3 * bool(nf),
        "fast_cholesky_factorizations": 18 * (batch * probed + nf),
        "fast_forecast_samples": 5 * batch * probed + 25 * nf,
        "slow_observation_samples": 32 * ns, "slow_context_advance_samples": 31 * ns,
        "slow_probe_advance_samples": 5 * batch * probed, "slow_forecast_advance_samples": 25 * ns,
        "grid_rotation_samples": 85 * batch * (variant in ("probe_fit", "probe_half")),
        "final_blend_samples": 25 * batch * (variant not in ("fast", "slow")), "probe_callback_calls": 0}


def validate_selection(selection, variant, batch, fallback, alpha):
    """Saved scalar arithmetic only; timed-call probe predictions are not saved."""
    errors = np.asarray(selection["errors"], dtype=float)
    require(errors.shape == (batch, 2, 2) and np.isfinite(errors).all() and (errors >= 0).all(), "selection error shape")
    if variant == "probe_inverse":
        require(set(selection) == {"errors", "inverse_zero_fallback", "alpha"}, "inverse selection fields")
        scale = errors.max(1)
        normalized = np.divide(errors, scale[:, None], out=np.zeros_like(errors), where=scale[:, None] > 0)
        total = normalized.sum(1)
        zero = total == 0
        require(np.array_equal(selection["inverse_zero_fallback"], zero), "selection inverse fallback")
        calculated = np.divide(normalized[:, 1], total, out=np.broadcast_to(fallback.astype(float), total.shape).copy(),
                               where=total > 0).astype(np.float32)
    else:
        fields = {"errors", "alpha", "position_numerator", "position_denominator", "position_threshold",
                  "position_fallback", "grid_coefficients", "grid_mse", "grid_index", "fitted_alpha"}
        require(set(selection) == fields, "LS/grid selection fields")
        n, d, threshold = (np.asarray(selection[k], float) for k in ("position_numerator", "position_denominator", "position_threshold"))
        require(all(x.shape == (batch,) and np.isfinite(x).all() for x in (n, d, threshold)) and (d >= 0).all(), "selection LS values")
        calculated_threshold = np.maximum(np.finfo(np.float64).tiny, np.finfo(np.float32).eps ** 2 * errors[:, :, 0].max(1))
        require(np.allclose(threshold, calculated_threshold, rtol=1e-10, atol=np.finfo(np.float64).tiny), "selection threshold")
        tiny = d <= threshold
        require(np.array_equal(selection["position_fallback"], tiny), "selection tiny fallback")
        position = np.clip(np.divide(n, d, out=np.full_like(n, float(fallback[0])), where=~tiny), 0, 1).astype(np.float32)
        require(selection["grid_coefficients"] == GRID.tolist(), "selection fixed grid")
        losses = np.asarray(selection["grid_mse"], float)
        selected = checked_grid_alpha(losses, losses)
        require(np.array_equal(selection["grid_index"], np.argmin(losses, 1)), "selection grid argmin")
        fitted = np.asarray(selection["fitted_alpha"], np.float32)
        require(fitted.shape == (batch, 2) and np.array_equal(fitted[:, 1], selected)
                and np.allclose(fitted[:, 0], position, rtol=1e-6, atol=2e-7), "selection fitted coefficients")
        calculated = np.full_like(fitted, .5) if variant == "probe_half" else fitted
    require(np.allclose(alpha, calculated, rtol=1e-6, atol=2e-7)
            and selection["alpha"] == alpha.tolist(), "selection alpha binding")


def validate_diagnostic(diag, variant, batch, fallback, *, payload=None):
    require(set(diag) == {"version", "variant", "probe", "fallback_alpha", "alpha", "selection", "timing", "work", "wall_seconds"}
            and diag["version"] == "pose-probe-blend-v1" and diag["variant"] == variant
            and diag["work"] == expected_work(variant, batch), "diagnostic schema/work")
    fallback_batch = np.broadcast_to(fallback, (batch, 2))
    require(diag["fallback_alpha"] == fallback_batch.tolist(), "diagnostic inherited fallback")
    alpha = np.asarray(diag["alpha"], np.float32)
    require(alpha.shape == (batch, 2) and np.isfinite(alpha).all() and (alpha >= 0).all() and (alpha <= 1).all(), "diagnostic alpha")
    probe = variant.startswith("probe_")
    if probe:
        require(diag["probe"] == {"origin": 26, "support_indices": [1, 25], "support_rows": 25,
            "target_indices": [27, 31], "action_indices": [26, 30], "age_half_life": 5, "huber_delta": 1.5,
            "huber_iterations": 3, "ridge_precision": 1., "solve_dtype": "float64"}, "completed probe scope")
        validate_selection(diag["selection"], variant, batch, fallback, alpha)
        if payload is not None:
            selected = {k: v.tolist() for k, v in payload.items() if k not in ("fast_p", "fast_R", "slow_p", "slow_R", "grid_rotations")}
            require(diag["selection"] == selected, "JSON/numeric probe identity")
    else:
        require(diag["probe"] is None and diag["selection"] is None and payload is None, "unprobed diagnostic")
        value = 1. if variant == "fast" else 0. if variant == "slow" else .5
        require(np.array_equal(alpha, np.full((batch, 2), value, np.float32)), "fixed coefficient")
    timing = diag["timing"]
    require(set(timing) == {"probe", "slow_context_seconds", "callback_seconds", "scoring_seconds", "full_forecast_seconds"}, "timing fields")
    subtotal = sum(finite(timing[k], k) for k in timing if k != "probe")
    require(timing["callback_seconds"] == 0, "no unrecorded probe callback")
    if probe:
        q = timing["probe"]
        require(set(q) == {"fast_fit_and_probe_seconds", "slow_context_and_probe_seconds", "wall_seconds"}, "probe timing fields")
        values = {k: finite(v, k) for k, v in q.items()}
        require(values["fast_fit_and_probe_seconds"] + values["slow_context_and_probe_seconds"] <= values["wall_seconds"] + 1e-9,
                "nested probe timings")
        require(timing["slow_context_seconds"] == 0, "probed context double charge")
        subtotal += values["wall_seconds"]
    else:
        require(timing["probe"] is None and timing["scoring_seconds"] == 0, "unprobed timing")
    wall = finite(diag["wall_seconds"], "module wall", positive=True)
    require(subtotal <= wall + 1e-9, "module disjoint timings")
    return alpha, wall


def aggregate(rows):
    families, references, comparisons, checks, latency = {}, {}, {}, [], {}
    for variant in (*VARIANTS, *REFERENCES):
        labels = [f"{variant}-{s}" for s in SEEDS] if variant in VARIANTS else [variant]
        samples = np.asarray([t for p in PANELS for label in labels for t in rows[p][label]["latency_ms"]])
        require(samples.shape == ((120,) if variant in VARIANTS else (40,))
                and np.isfinite(samples).all() and (samples > 0).all(), "latency coverage")
        latency[variant] = {"samples": len(samples), "median_ms": float(np.median(samples)),
                            "p95_ms": float(np.percentile(samples, 95))}
    for panel in PANELS:
        families[panel] = {v: pooled([rows[panel][f"{v}-{s}"]["errors"] for s in SEEDS]) for v in VARIANTS}
        references[panel] = {v: rows[panel][v]["errors"] for v in REFERENCES}
        comparisons[panel] = {}
        for control in (*[v for v in VARIANTS if v != PRIMARY], *REFERENCES):
            comparisons[panel][control] = {}
            for endpoint in ENDPOINTS:
                c = families[panel][PRIMARY][endpoint]
                b = (families[panel] if control in VARIANTS else references[panel])[control][endpoint]
                name = f"{panel}/{control}/{endpoint}"
                checks.append({"name": name + "/family", "passed": b["rmse"] > 0
                    and Decimal(str(c["rmse"])) <= Decimal(".90") * Decimal(str(b["rmse"])),
                    "candidate": c["rmse"], "control": b["rmse"]})
                paired = []
                for seed in SEEDS:
                    cm = rows[panel][f"{PRIMARY}-{seed}"]["errors"][endpoint]["mse"]
                    bm = rows[panel][f"{control}-{seed}" if control in VARIANTS else control]["errors"][endpoint]["mse"]
                    checks.append({"name": name + f"/pair_{seed}", "passed": cm <= bm, "candidate": cm, "control": bm})
                    paired.append({"seed": seed, "candidate_mse": cm, "control_mse": bm,
                                   "rmse_improvement_percent": percent(math.sqrt(cm), math.sqrt(bm))})
                cp, bp = np.asarray(c["parent_mse"]), np.asarray(b["parent_mse"])
                require(cp.shape == bp.shape == (10,) and c["parent_ids"] == b["parent_ids"] == list(range(10)), "parent coverage")
                count = int(np.sum(cp <= bp))
                checks.append({"name": name + "/parents", "passed": count >= 8, "count": count})
                loo = []
                for i in range(10):
                    cm, bm = float(np.delete(cp, i).mean()), float(np.delete(bp, i).mean())
                    checks.append({"name": name + f"/loo_{i}", "passed": cm < bm, "candidate": cm, "control": bm})
                    loo.append({"excluded_parent": i, "candidate_mse": cm, "control_mse": bm})
                comparisons[panel][control][endpoint] = {"family_rmse_improvement_percent": percent(c["rmse"], b["rmse"]),
                    "paired": paired, "parents_nonworse": count, "parent_mse_improvements": (bp - cp).tolist(),
                    "leave_one_parent_out": loo}
    c, b = latency[PRIMARY]["median_ms"], latency["gru"]["median_ms"]
    checks.append({"name": "latency_vs_gru", "passed": Decimal(str(c)) <= Decimal("1.5") * Decimal(str(b)),
                   "candidate": c, "control": b})
    require(len(checks) == 2101, "all2101 comparisons")
    groups = []
    for panel in PANELS:
        for endpoint in ENDPOINTS:
            for category, count in (("family", 35), ("pair", 105), ("parents", 35), ("loo", 350)):
                selected = [x for x in checks if x["name"].startswith(panel + "/")
                            and f"/{endpoint}/{category}" in x["name"]]
                require(len(selected) == count, "group coverage")
                groups.append({"name": f"{panel}/{endpoint}/{category}", "passed": all(x["passed"] for x in selected),
                    "comparisons_passed": sum(x["passed"] for x in selected), "total_comparisons": count,
                    "comparison_names": [x["name"] for x in selected]})
    groups.append({"name": "latency_vs_gru", "passed": checks[-1]["passed"], "comparisons_passed": int(checks[-1]["passed"]),
                   "total_comparisons": 1, "comparison_names": ["latency_vs_gru"]})
    gate = {"passed": all(x["passed"] for x in checks), "requirements_passed": sum(x["passed"] for x in groups),
            "total_requirements": 17, "requirements": groups, "checks_passed": sum(x["passed"] for x in checks),
            "total_checks": 2101, "checks": checks,
            "components": {"family": 140, "paired": 420, "parents": 140, "leave_one_out": 1400, "latency": 1}}
    return families, references, latency, comparisons, gate


def expected_members():
    result = {"started.json", "completed.json"}
    for panel in PANELS:
        result.add(f"{panel}-inputs.npz")
        for variant in NEW_VARIANTS:
            for seed in SEEDS:
                prefix = f"{panel}-{variant}-{seed}"
                result.update((prefix + "-predictions.npz", prefix + "-evaluation.json"))
                if variant.startswith("probe_"):
                    result.add(prefix + "-probe.npz")
    require(len(result) == 94, "internal94 file boundary")
    return result


def required_settings():
    required = {"study": "pose-probe-blend-v1", "scope": "exposed-data completed-past probe blending screen",
        "capacity_protocol_sha256": CAPACITY_PROTOCOL, "capacity_completed_sha256": CAPACITY_COMPLETED,
        "capacity_summary_sha256": CAPACITY_SUMMARY, "capacity_receipt_sha256": CAPACITY_RECEIPT,
        "seeds": list(SEEDS), "panels": list(PANELS), "variants": list(NEW_VARIANTS), "primary": PRIMARY,
        "context": 32, "horizon": 25, "windows_per_panel": 160, "parents_per_panel": 10,
        "probe_origin": 26, "probe_poses": 27, "probe_past_actions": 26,
        "probe_support_indices": [1, 25], "probe_action_indices": [26, 30], "probe_target_indices": [27, 31],
        "probe_age_half_life": 5, "probe_huber_delta": 1.5, "probe_huber_iterations": 3,
        "probe_ridge_precision": 1., "probe_solve_dtype": "float64", "rotation_grid": GRID.tolist(),
        "evaluation_rows": 36, "execution_files": 94, "canonical_configurations": 36,
        "distinct_combined_rows": 192, "controls": 35, "gate_groups": 17, "gate_comparisons": 2101,
        "warmups_per_row": 3, "timed_windows_per_row": 20, "warmup_indices": [0, 1, 2], "timed_indices": list(range(3, 23)),
        "forecast_calls": 864, "batch_forecast_calls": 36, "single_forecast_calls": 828,
        "new_neural_fits": 0, "new_optimizer_updates": 0, "new_native_calls": 0,
        "blend_tolerance": {"rtol": 2e-6, "atol": 2e-6, "hard_endpoints": "exact"},
        "expert_replay_tolerance": {"rtol": 1e-6, "atol": 2e-7}, "coefficient_arithmetic_tolerance": LOSS_TOLERANCE,
        "gate": "probe_fit RMSE <=0.90*each positive control on both physical endpoints/panels; all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out MSE strictly lower; median full-window latency<=1.5*new slow. All17 groups required.",
        "position_fit": "Mean dot(fast-slow,target-slow) divided by mean squared separation, clipped to[0,1], computed in float64.",
        "position_fallback": "D <= max(float64.tiny, eps(input_dtype)^2 * max(fast_position_MSE,slow_position_MSE)); use saved training-only full_constant position alpha.",
        "inverse_fit": "Opposite expert MSE divided by sum, after overflow-safe common rescaling; both zero uses saved training-only full_constant alpha.",
        "rotation_fit": "Actual float32 production blend on the17 ascending grid coefficients, scored in float64 squared geodesic radians over5 completed leads; exact ties choose first index.",
        "probe_half": "Compute the same position fit and rotation grid as probe_fit, then discard coefficients and use exact half.",
        "probe_inverse": "Compute probe forecasts and endpoint errors, then inverse-error weights without the fitted coefficient or rotation grid.",
        "causality": "At root31, retrospective probe uses only poses0..31 and recorded applied actions0..30. These actions were not necessarily known at historical origin26. No future observed pose enters any forecast or coefficient.",
        "deployment": "Coefficients remain fixed across25 future steps. Experts roll privately; slow actual-context state continues through31 without probe feedback and preserves CV1.",
        "future_actions": "Both full forecasts condition on supplied recorded applied actions31..55, as in all parent studies; no planned-command or closed-loop claim.",
        "timing_scope": "Full single-window forecast including input validation, probe fitting/rollout/scoring when used, both deployment experts when used, blending and returned diagnostics. Loading, saved probe-array packaging, metric calculation and artifact I/O excluded.",
        "no_selection": "One complete fixed run, all cases and seeds. No retries, sweeps, exclusions, grid refinement, threshold tuning or gate changes. Keep partial returned artifacts on failure."}
    return required


def validate_settings(protocol):
    required = required_settings()
    require(set(protocol) == set(required) | {"capacity_experiment", "capacity_report", "data", "data_hashes",
                "checkpoints", "fallback", "sources", "runtime"}
            and all(protocol[k] == value for k, value in required.items()), "frozen settings")


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "external protocol digest")
    protocol = read_json(experiment / "protocol.json")
    validate_settings(protocol)
    require(len(SOURCES) == 36 and set(protocol["sources"]) == SOURCES
            and protocol["sources"] == {k: sha(ROOT / k) for k in SOURCES}
            == {k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}, "all36 source bindings")
    require(protocol["runtime"] == {"python": platform.python_version(), "numpy": np.__version__,
        "torch": importlib.metadata.version("torch"), "platform": platform.platform(), "threads": 1,
        "deterministic_algorithms": True}, "runtime identity")
    capacity = Path(protocol["capacity_experiment"])
    _, parent = _capacity.validate_inputs(capacity, CAPACITY_PROTOCOL)
    cap_run, cap_report = capacity / "run-01", Path(protocol["capacity_report"])
    cap_members, report_members = members(cap_run), members(cap_report)
    done = read_json(cap_run / "completed.json")
    require(set(cap_members) == _capacity.expected_members() and cap_members["completed.json"]["sha256"] == CAPACITY_COMPLETED
            and done["status"] == "completed" and done["protocol_sha256"] == CAPACITY_PROTOCOL
            and done["files"] == {k: v for k, v in cap_members.items() if k != "completed.json"}, "capacity execution seal")
    require(set(report_members) == {"summary.json", "receipt.json", "window-errors.npz"}
            and report_members["summary.json"]["sha256"] == CAPACITY_SUMMARY
            and report_members["receipt.json"]["sha256"] == CAPACITY_RECEIPT, "capacity report pins")
    receipt = read_json(cap_report / "receipt.json")
    require(receipt["status"] == "completed" and receipt["execution_members"] == cap_members
            and receipt["files"] == {k: v for k, v in report_members.items() if k != "receipt.json"}, "capacity audit seal")
    inherited = parent["context"]["protocol"]
    require(all(protocol[k] == inherited[k] for k in ("data", "data_hashes", "checkpoints")), "unchanged data and deployment experts")
    require(set(protocol["fallback"]) == {str(s) for s in SEEDS}, "all3 fallbacks")
    for seed in SEEDS:
        path = parent["run"] / f"full_constant-{seed}-fit.json"
        record = protocol["fallback"][str(seed)]
        require(record == {"path": str(path.resolve()), "sha256": sha(path),
                           "alpha": read_json(path)["optimization"]["alpha"]}, "training-only fallback identity")
    require(all(sha(Path(v["path"])) == v["sha256"] for v in protocol["checkpoints"].values()), "expert checkpoint bytes")
    return {"protocol": protocol, "parent": parent, "capacity_run": cap_run, "capacity_members": cap_members,
            "capacity_report": cap_report, "capacity_report_members": report_members}


def inherited_path(parent, panel, variant, seed):
    label = variant if seed is None else f"{variant}-{seed}"
    if variant in _parent.NEW_VARIANTS[2:]:
        source, has_alpha = parent["run"], True
    elif variant in _geometry.NEW_VARIANTS[2:]:
        source, has_alpha = parent["context"]["prior_run"], True
    else:
        source, has_alpha = parent["context"]["parent"]["prior_run"], False
    return source / f"{panel}-{label}-predictions.npz", has_alpha


def read_inputs(path, data, panel, parent_run):
    with np.load(path, allow_pickle=False) as value:
        require(set(value.files) == {"p", "R", "actions", "ids"}, "saved input fields")
        p, rotation, actions, ids = (value[k] for k in ("p", "R", "actions", "ids"))
    ep, er, ea, expected_ids = _geometry.load_public(data, panel)
    require(p.shape == (160, 57, 3) and rotation.shape == (160, 57, 3, 3) and actions.shape == (160, 56, 40)
            and all(x.dtype == np.float32 and np.isfinite(x).all() for x in (p, rotation, actions))
            and ids.dtype == np.int64 and ids.shape == (160, 2) and np.array_equal(ids, expected_ids)
            and np.array_equal(p, ep) and np.allclose(rotation, er, rtol=0, atol=2e-7)
            and np.array_equal(actions, ea), "original public input provenance")
    with np.load(parent_run / f"{panel}-targets.npz", allow_pickle=False) as target:
        require(np.array_equal(p[:, 32:], target["p"]) and np.array_equal(rotation[:, 32:], target["R"])
                and np.array_equal(ids, target["ids"]), "unchanged future targets")
    return p, rotation, ids


def validate_replay(p, rotation, old_p, old_r, *, exact=False):
    require(np.allclose(p, old_p, rtol=1e-6, atol=2e-7)
            and np.allclose(rotation, old_r, rtol=1e-6, atol=2e-7), "frozen expert/equal-mixture replay")
    bitwise = p.dtype == old_p.dtype and rotation.dtype == old_r.dtype and p.tobytes() == old_p.tobytes() and rotation.tobytes() == old_r.tobytes()
    require(not exact or bitwise, "same-run compute-null bitwise parity")
    return {"position_max_abs_error": float(np.abs(p.astype(float) - old_p.astype(float)).max()),
            "rotation_max_abs_error": float(np.abs(rotation.astype(float) - old_r.astype(float)).max()),
            "bitwise_equal": bitwise, "exact_required": exact, "rtol": 1e-6, "atol": 2e-7}


def audit(experiment, out, *, protocol_sha256, completed_sha256):
    start = time.perf_counter()
    require(not out.resolve().is_relative_to(experiment.resolve()), "audit output inside execution")
    out.mkdir(parents=True, exist_ok=False)
    try:
        context = validate_inputs(experiment, protocol_sha256)
        protocol, parent = context["protocol"], context["parent"]
        run = experiment / "run-01"
        before = members(run)
        require(set(before) == expected_members() and before["completed.json"]["sha256"] == completed_sha256,
                "exact94 execution files and external completion digest")
        done = read_json(run / "completed.json")
        require(set(done) == {"status", "protocol_sha256", "rows", "forecast_calls", "batch_forecast_calls",
            "single_forecast_calls", "new_optimizer_updates", "wall_seconds", "files"}
            and done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256
            and (done["forecast_calls"], done["batch_forecast_calls"], done["single_forecast_calls"], done["new_optimizer_updates"])
            == (864, 36, 828, 0) and len(done["rows"]) == 36
            and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "execution seal/call counts")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "execution start")
        rows, active, replay, probe_checks, blends, errors_out = {}, {}, {}, {}, {}, {}
        row_index, paid_work = 0, {}
        wall_rows = batch_seconds = measured_seconds = warmup_seconds = 0.
        for panel in PANELS:
            p, rotation, ids = read_inputs(run / f"{panel}-inputs.npz", Path(protocol["data"]), panel, parent["run"])
            tp, tr = p[:, 32:], rotation[:, 32:]
            rows[panel], active[panel], replay[panel], probe_checks[panel], blends[panel] = {}, {}, {}, {}, {}
            errors_out[panel + "__ids"] = ids
            old_arrays = {}
            for variant in (*_parent.VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in _parent.VARIANTS else (None,):
                    label = variant if seed is None else f"{variant}-{seed}"
                    path, has_alpha = inherited_path(parent, panel, variant, seed)
                    pp, rr = _geometry.load_prediction(path, alpha=has_alpha)[:2]
                    values = error_arrays(pp, rr, tp, tr)
                    calculated = reduce_errors(values, ids)
                    inherited = parent["summary"]["rows"][panel][label]
                    require(calculated == inherited["errors"], "inherited control metric identity")
                    rows[panel][label] = {**inherited, "latency_scope": "inherited descriptive only"}
                    if variant in ALIASES.values():
                        old_arrays[label] = (pp, rr)
                    for endpoint, value in values.items():
                        errors_out[f"{panel}-{label}__{endpoint}"] = value
            fresh, probe_cache = {}, {}
            for variant in NEW_VARIANTS:
                for seed in SEEDS:
                    label = f"{variant}-{seed}"; prefix = f"{panel}-{label}"
                    row = read_json(run / f"{prefix}-evaluation.json")
                    require(set(row) == {"panel", "variant", "seed", "expert_sha256", "fallback_alpha", "prediction_sha256",
                        "probe_sha256", "metrics", "latency_ms", "warmup_ms", "timing_diagnostics", "batch_diagnostics",
                        "batch_forecast_seconds", "row_wall_seconds"} and row == done["rows"][row_index]
                        and (row["panel"], row["variant"], row["seed"]) == (panel, variant, seed), "ordered row identity")
                    row_index += 1
                    fallback = np.asarray(protocol["fallback"][str(seed)]["alpha"], np.float32)
                    require(row["fallback_alpha"] == fallback.tolist() and row["expert_sha256"] == {
                        name: protocol["checkpoints"][f"{name}-{seed}"]["sha256"] for name in ("meta", "gru")}
                        and row["prediction_sha256"] == before[prefix + "-predictions.npz"]["sha256"], "row source/checkpoint binding")
                    pp, rr, alpha = _geometry.load_prediction(run / f"{prefix}-predictions.npz", alpha=True)
                    require(pp.dtype == rr.dtype == alpha.dtype == np.float32, "new float32 forecast")
                    payload = None
                    if variant.startswith("probe_"):
                        require(row["probe_sha256"] == before[prefix + "-probe.npz"]["sha256"], "probe payload binding")
                        with np.load(run / f"{prefix}-probe.npz", allow_pickle=False) as value:
                            payload = {k: value[k] for k in value.files}
                        probe_checks[panel][label] = verify_probe(payload, variant, p[:, :32], rotation[:, :32], fallback, alpha)
                        if seed in probe_cache:
                            common = set(payload) & set(probe_cache[seed]) - {"alpha"}
                            require(all(np.array_equal(payload[k], probe_cache[seed][k]) for k in common), "same-input probe and selection identity")
                        else:
                            probe_cache[seed] = payload
                    else:
                        require(row["probe_sha256"] is None, "unprobed payload absent")
                    da, dwall = validate_diagnostic(row["batch_diagnostics"], variant, 160, fallback, payload=payload)
                    require(np.array_equal(da, alpha), "batch diagnostic/forecast alpha")
                    if variant in ("fast", "slow"):
                        fresh[(variant, seed)] = (pp, rr)
                    fp, fr = fresh[("fast", seed)]
                    sp, sr = fresh.get(("slow", seed), (fp, fr))
                    if variant == "fast":
                        blends[panel][label] = {"fixed_fast": True}
                    else:
                        blends[panel][label] = _geometry.validate_blend(variant if variant in ALIASES else "recurrent",
                            pp, rr, alpha, fp, fr, sp, sr)
                    if variant in ALIASES:
                        canonical = f"{ALIASES[variant]}-{seed}"
                        replay[panel][label] = validate_replay(pp, rr, *old_arrays[canonical])
                        rows[panel][canonical]["latency_ms"] = row["latency_ms"]
                        rows[panel][canonical]["latency_scope"] = "fresh alias " + variant
                    elif variant == "probe_half":
                        replay[panel][label] = validate_replay(pp, rr, *fresh[("half", seed)], exact=True)
                    if variant == "half":
                        fresh[("half", seed)] = (pp, rr)
                    values = error_arrays(pp, rr, tp, tr); calculated = reduce_errors(values, ids)
                    _geometry.validate_metrics(row["metrics"], calculated, rr)
                    for key, count in (("latency_ms", 20), ("warmup_ms", 3)):
                        times = np.asarray(row[key])
                        require(times.shape == (count,) and np.isfinite(times).all() and (times > 0).all(), "all recorded timings")
                    require(len(row["timing_diagnostics"]) == 23, "all23 single-call diagnostics")
                    for diag, elapsed in zip(row["timing_diagnostics"], [*row["warmup_ms"], *row["latency_ms"]], strict=True):
                        _, inner = validate_diagnostic(diag, variant, 1, fallback)
                        require(inner <= elapsed / 1000 + 1e-9, "single-call inner/outer timing")
                    for diag in [row["batch_diagnostics"], *row["timing_diagnostics"]]:
                        for key, value in diag["work"].items():
                            if key != "scope":
                                paid_work[key] = paid_work.get(key, 0) + value
                    batch_wall = finite(row["batch_forecast_seconds"], "batch wall", positive=True)
                    wall = finite(row["row_wall_seconds"], "row wall", positive=True)
                    singles = (sum(row["warmup_ms"]) + sum(row["latency_ms"])) / 1000
                    require(dwall <= batch_wall + 1e-9 and batch_wall + singles <= wall + 1e-8, "row nested cost")
                    batch_seconds += batch_wall; wall_rows += wall
                    measured_seconds += sum(row["latency_ms"]) / 1000; warmup_seconds += sum(row["warmup_ms"]) / 1000
                    item = {"variant": variant, "seed": seed, "errors": calculated, "latency_ms": row["latency_ms"],
                            "expert_sha256": row["expert_sha256"], "fallback_alpha": row["fallback_alpha"],
                            "latency_scope": "fresh complete callable", "alpha_mean": alpha.mean(0, dtype=float).tolist(),
                            "alpha_min": alpha.min(0).tolist(), "alpha_max": alpha.max(0).tolist(),
                            "alpha_std": alpha.std(0, dtype=float).tolist()}
                    active[panel][label] = item
                    if variant not in ALIASES:
                        rows[panel][label] = item
                    for endpoint, value in values.items():
                        errors_out[prefix + "__" + endpoint] = value
        wall = finite(done["wall_seconds"], "execution wall", positive=True)
        require(row_index == 36 and sum(map(len, rows.values())) == 192 and wall_rows <= wall + 1e-6, "full row/whole cost coverage")
        families, refs, latency, comparisons, gate = aggregate(rows)
        for variant, value in latency.items():
            value["scope"] = "fresh complete callable" if variant in (*ALIASES.values(), *NEW_VARIANTS[3:]) else "inherited descriptive only"
        summary = {"status": "completed", "study": "pose-probe-blend-v1", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "parent_summary_sha256": _capacity.PARENT_SUMMARY,
            "capacity_summary_sha256": CAPACITY_SUMMARY, "rows": rows, "active_rows": active,
            "families": families, "references": refs, "latency": latency, "contrasts": comparisons,
            "continuation_gate": gate, "probe_reconstruction": probe_checks, "blend_reconstruction": blends, "replay": replay,
            "counts": {"new_neural_fits": 0, "new_optimizer_updates": 0, "new_rows": 36, "inherited_rows": 174,
                "aliased_new_rows": 18, "distinct_combined_rows": 192, "controls": 35, "forecasts": 864,
                "batch_forecasts": 36, "single_forecasts": 828, "probe_payloads": 18, "execution_files": 94},
            "work": paid_work, "costs": {"execution_wall_seconds": wall, "row_wall_seconds": wall_rows,
                "batch_forecast_seconds": batch_seconds, "timed_single_seconds": measured_seconds, "warmup_single_seconds": warmup_seconds,
                "scope": "Nested components; do not add to execution total. Full callable timings include all requested probe/selection work; inherited training and initial evidence authentication are separate."},
            "limits": ["Repeatedly exposed development archives, not fresh generalization or closed-loop control.",
                "The retrospective probe uses completed past actions26..30 and targets27..31, not actions necessarily knowable at origin26. Full forecasts condition on recorded applied future actions31..55.",
                "Probe neural outputs and timed-call outputs are authenticated/source-bound, not regenerated. Batch probe errors, grid geometries/scores, coefficients and full blends are independently reconstructed.",
                "Recorded production grid losses determine exact first argmin after independent float64 metric checks; geometry reconstruction has the prospectively bound floating-point tolerance.",
                "Five-step past error is transferred to a25-step horizon with a different support length. It is not calibrated uncertainty.",
                "No new model, optimizer, native or random calls were made by this audit; no successful numerical gate implies architectural novelty."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **errors_out)
        require(members(run) == before and members(parent["run"]) == parent["members"]
                and members(context["capacity_run"]) == context["capacity_members"]
                and members(context["capacity_report"]) == context["capacity_report_members"], "evidence changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "parent_summary_sha256": _capacity.PARENT_SUMMARY, "capacity_summary_sha256": CAPACITY_SUMMARY,
            "execution_members": before, "sources": protocol["sources"], "data_hashes": protocol["data_hashes"],
            "expert_checkpoints": protocol["checkpoints"], "fallback": protocol["fallback"], "auditor_sha256": sha(__file__),
            "capacity_auditor_sha256": CAPACITY_AUDITOR_SHA256, "files": members(out),
            "qualification_passed": gate["passed"], "requirements_passed": gate["requirements_passed"], "total_requirements": 17,
            "checks_passed": gate["checks_passed"], "total_checks": 2101, "new_model_calls": 0,
            "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0, "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error), "wall_seconds": time.perf_counter() - start,
                "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            error.add_note(f"Could not preserve failure: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256, completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "requirements_passed", "total_requirements", "checks_passed", "total_checks", "wall_seconds")}))
