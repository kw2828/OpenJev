"""Synthetic geometric/coefficient and gate checks; no model or data calls."""
import math
from pathlib import Path

import audit_pose_probe_blend as audit
import numpy as np
import pytest


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def probe():
    fp = np.zeros((1, 5, 3), np.float32); fp[..., 0] = 2.
    sp = np.zeros_like(fp)
    fr = np.broadcast_to(rotation(1.), (1, 5, 3, 3)).copy()
    sr = np.broadcast_to(rotation(0.), fr.shape).copy()
    context_p = np.zeros((1, 32, 3), np.float32); context_p[:, 27:, 0] = 1.
    context_r = np.broadcast_to(rotation(0.), (1, 32, 3, 3)).copy()
    context_r[:, 27:] = rotation(.5)
    return fp, fr, sp, sr, context_p, context_r, np.array([.3, .7], np.float32)


def test_analytic_completed_probe_errors_ls_inverse_and_grid():
    result = audit.probe_arithmetic(*probe())
    np.testing.assert_allclose(result["errors"], [[[1., .25], [1., .25]]], atol=1e-7)
    assert result["position_numerator"].tolist() == [2.]
    assert result["position_denominator"].tolist() == [4.]
    assert result["position_alpha"].tolist() == [.5]
    assert not result["position_fallback"].any()
    np.testing.assert_allclose(result["inverse_alpha"], [[.5, .5]], atol=1e-7)
    assert audit.checked_grid_alpha(result["rotation_grid_losses"], result["rotation_grid_losses"]).tolist() == [.5]


def test_tiny_disagreement_and_zero_error_fallbacks_are_distinct():
    fp, fr, sp, sr, p, r, fallback = probe()
    fp[:] = sp
    fr[:] = sr
    r[:] = sr[:, :1]
    result = audit.probe_arithmetic(fp, fr, sp, sr, p, r, fallback)
    assert result["position_fallback"].all()
    assert result["position_alpha"][0] == fallback[0]
    assert result["inverse_alpha"][0, 0] == .5
    assert result["inverse_alpha"][0, 1] == fallback[1]
    assert audit.checked_grid_alpha(result["rotation_grid_losses"], result["rotation_grid_losses"]).tolist() == [0.]
    p[:] = 0
    both_zero = audit.probe_arithmetic(fp, fr, sp, sr, p, r, fallback)
    assert np.array_equal(both_zero["inverse_alpha"], fallback[None])


def test_declared_dtype_threshold_and_signed_numerator():
    fp, fr, sp, sr, p, r, fallback = probe()
    fp[..., 0] = 1e-9
    result = audit.probe_arithmetic(fp, fr, sp, sr, p, r, fallback)
    assert result["position_fallback"].all()
    assert result["position_threshold"][0] == np.finfo(np.float32).eps ** 2
    fp[..., 0] = 1e-6
    assert not audit.probe_arithmetic(fp, fr, sp, sr, p, r, fallback)["position_fallback"].any()
    fp[..., 0] = 2
    p[:, 27:, 0] = -1
    result = audit.probe_arithmetic(fp, fr, sp, sr, p, r, fallback)
    assert result["position_numerator"][0] == -2 and result["position_alpha"][0] == 0


def test_full_future_targets_are_not_admitted_and_earlier_context_not_scored():
    inputs = list(probe())
    baseline = audit.probe_arithmetic(*inputs)
    inputs[4][:, :27] = 123
    modified = audit.probe_arithmetic(*inputs)
    assert all(np.array_equal(baseline[k], modified[k]) for k in baseline)
    inputs[4] = np.concatenate((inputs[4], np.zeros((1, 25, 3), np.float32)), 1)
    with pytest.raises(ValueError, match="probe/context"):
        audit.probe_arithmetic(*inputs)


def test_grid_exact_ties_and_saved_rounding_near_ties():
    recomputed = np.ones((2, 17), np.float64)
    saved = recomputed.copy(); saved[1, 8] -= 1e-12
    assert audit.checked_grid_alpha(saved, recomputed).tolist() == [0., .5]
    saved[0, 3] -= .001
    with pytest.raises(ValueError):
        audit.checked_grid_alpha(saved, recomputed)


def errors(mse, parents=None):
    parents = [mse] * 10 if parents is None else parents
    return {e: {"mse": mse, "rmse": math.sqrt(mse), "horizon_mse": [mse] * 25,
                "horizon_rmse": [math.sqrt(mse)] * 25, "parent_ids": list(range(10)),
                "parent_mse": parents, "parent_rmse": np.sqrt(parents).tolist()} for e in audit.ENDPOINTS}


def rows():
    return {p: {f"{v}-{s}" if s is not None else v: {"errors": errors(.81 if v == audit.PRIMARY else 1.),
                "latency_ms": [1.5 if v == audit.PRIMARY else 1.] * 20}
                for v in (*audit.VARIANTS, *audit.REFERENCES)
                for s in (audit.SEEDS if v in audit.VARIANTS else (None,))} for p in audit.PANELS}


def test_all192_rows_2101_checks_17groups_and_exact_inclusive_boundaries():
    r = rows()
    assert sum(map(len, r.values())) == 192
    families, references, latency, comparisons, gate = audit.aggregate(r)
    assert gate["passed"] and gate["checks_passed"] == gate["total_checks"] == 2101
    assert gate["requirements_passed"] == gate["total_requirements"] == 17
    names = [name for group in gate["requirements"] for name in group["comparison_names"]]
    assert len(set(names)) == len(names) == 2101
    assert len(families["test_sin"]) == 30 and len(references["test_sin"]) == 6
    assert len(comparisons["test_sin"]) == 35
    assert latency["gru"]["samples"] == 120


@pytest.mark.parametrize("violation", ["family", "pair", "parents", "loo", "latency"])
def test_single_failed_category_cannot_be_rescued(violation):
    r = rows()
    for seed in audit.SEEDS:
        entry = r["test_sin"][f"{audit.PRIMARY}-{seed}"]
        if violation == "family":
            entry["errors"] = errors(.811)
        elif violation == "pair":
            entry["errors"] = errors(1.01 if seed == 1101 else .01)
        elif violation == "parents":
            entry["errors"] = errors(.8, [.5] * 7 + [1.5] * 3)
        elif violation == "loo":
            entry["errors"] = errors(.5)
            r["test_sin"][f"probe_half-{seed}"]["errors"] = errors(.9, [4.5] + [.5] * 9)
        else:
            for panel in audit.PANELS:
                r[panel][f"{audit.PRIMARY}-{seed}"]["latency_ms"] = [1.5001] * 20
    assert not audit.aggregate(r)[-1]["passed"]


def payload(variant="probe_fit"):
    fp, fr, sp, sr, p, r, fallback = probe()
    result = audit.probe_arithmetic(fp, fr, sp, sr, p, r, fallback)
    value = {"fast_p": fp, "fast_R": fr, "slow_p": sp, "slow_R": sr, "errors": result["errors"]}
    if variant == "probe_inverse":
        value.update(alpha=result["inverse_alpha"], inverse_zero_fallback=result["errors"].sum(1) == 0)
    else:
        value.update({k: result[k] for k in ("position_numerator", "position_denominator", "position_threshold", "position_fallback")})
        rotations = []
        for coefficient in audit.GRID:
            _, rr, _ = audit._geometry.reconstruct_blend(fp, fr, sp, sr, np.full((1, 2), coefficient, np.float32))
            rotations.append(rr)
        value.update(grid_coefficients=audit.GRID.astype(np.float32), grid_rotations=np.stack(rotations, 1),
                     grid_mse=result["rotation_grid_losses"], grid_index=np.argmin(result["rotation_grid_losses"], 1))
        fitted = np.stack((result["position_alpha"], audit.GRID[value["grid_index"]].astype(np.float32)), -1)
        value.update(fitted_alpha=fitted, alpha=np.full_like(fitted, .5) if variant == "probe_half" else fitted)
    return value, p, r, fallback


@pytest.mark.parametrize("variant", ["probe_half", "probe_inverse", "probe_fit"])
def test_full_numeric_probe_payload_and_json_selection_roundtrip(variant):
    value, p, r, fallback = payload(variant)
    audit.verify_probe(value, variant, p, r, fallback, value["alpha"])
    selection = {k: v.tolist() for k, v in value.items()
                 if k not in ("fast_p", "fast_R", "slow_p", "slow_R", "grid_rotations")}
    audit.validate_selection(selection, variant, 1, fallback, value["alpha"])


@pytest.mark.parametrize("damage", ["future_error", "signed_ls", "threshold", "fallback", "grid_coefficient", "grid_rotation",
                                    "grid_score", "grid_index", "fitted_alpha", "actual_alpha", "missing", "extra"])
def test_probe_numeric_tampering_rejected(damage):
    value, p, r, fallback = payload()
    if damage == "future_error":
        value["errors"] += .01
    elif damage == "signed_ls":
        value["position_numerator"] *= -1
    elif damage == "threshold":
        value["position_threshold"][:] = 1
    elif damage == "fallback":
        value["position_fallback"][:] = True
    elif damage == "grid_coefficient":
        value["grid_coefficients"][1] += .01
    elif damage == "grid_rotation":
        value["grid_rotations"][:, 8] = rotation(.3)
    elif damage == "grid_score":
        value["grid_mse"][:, 8] += .01
    elif damage == "grid_index":
        value["grid_index"][:] = 9
    elif damage == "fitted_alpha":
        value["fitted_alpha"][:, 1] = 9 / 16
    elif damage == "actual_alpha":
        value["alpha"][:, 1] = 9 / 16
    elif damage == "missing":
        value.pop("grid_rotations")
    else:
        value["future_targets"] = p
    with pytest.raises(ValueError):
        audit.verify_probe(value, "probe_fit", p, r, fallback, value["alpha"])


def diagnostic(variant, batch=1):
    value, _, _, fallback = payload(variant) if variant.startswith("probe_") else (None, None, None, np.array([.3, .7], np.float32))
    probed = variant.startswith("probe_")
    alpha = value["alpha"] if probed else np.full((1, 2), 1. if variant == "fast" else 0. if variant == "slow" else .5, np.float32)
    return {"version": "pose-probe-blend-v1", "variant": variant,
        "probe": {"origin": 26, "support_indices": [1, 25], "support_rows": 25, "target_indices": [27, 31],
            "action_indices": [26, 30], "age_half_life": 5, "huber_delta": 1.5, "huber_iterations": 3,
            "ridge_precision": 1., "solve_dtype": "float64"} if probed else None,
        "fallback_alpha": np.broadcast_to(fallback, (batch, 2)).tolist(), "alpha": alpha.tolist(),
        "selection": None if value is None else {k: v.tolist() for k, v in value.items()
            if k not in ("fast_p", "fast_R", "slow_p", "slow_R", "grid_rotations")},
        "work": audit.expected_work(variant, batch), "wall_seconds": .04,
        "timing": {"probe": {"fast_fit_and_probe_seconds": .001, "slow_context_and_probe_seconds": .001, "wall_seconds": .003} if probed else None,
            "slow_context_seconds": 0. if probed or variant == "fast" else .001,
            "callback_seconds": 0., "scoring_seconds": .001 if probed else 0., "full_forecast_seconds": .01}}, fallback


@pytest.mark.parametrize("variant", audit.NEW_VARIANTS)
def test_all_variant_full_work_and_nested_timing(variant):
    diag, fallback = diagnostic(variant)
    audit.validate_diagnostic(diag, variant, 1, fallback)
    if variant.startswith("probe_"):
        assert diag["work"]["fast_cholesky_factorizations"] == 36
        assert diag["work"]["fast_support_rows"] == 55
        assert diag["work"]["fast_feature_samples"] == 85
        assert diag["work"]["slow_probe_advance_samples"] == 5
        assert diag["work"]["grid_rotation_samples"] == (0 if variant == "probe_inverse" else 85)
    diag["work"]["fast_support_rows"] += 1
    with pytest.raises(ValueError):
        audit.validate_diagnostic(diag, variant, 1, fallback)


@pytest.mark.parametrize("damage", ["targets", "future_actions", "parent_time", "callback", "fallback"])
def test_diagnostic_causal_boundary_and_time_tampering(damage):
    diag, fallback = diagnostic("probe_fit")
    if damage == "targets":
        diag["probe"]["target_indices"] = [32, 36]
    elif damage == "future_actions":
        diag["probe"]["action_indices"] = [31, 35]
    elif damage == "parent_time":
        diag["wall_seconds"] = .001
    elif damage == "callback":
        diag["timing"]["callback_seconds"] = .01
    else:
        diag["fallback_alpha"] = [[.5, .5]]
    with pytest.raises(ValueError):
        audit.validate_diagnostic(diag, "probe_fit", 1, fallback)


def test_exact94_members_and_canonical_inherited_path_mapping():
    names = audit.expected_members()
    assert len(names) == 94 and sum(n.endswith("-probe.npz") for n in names) == 18
    assert sum(n.endswith("-evaluation.json") for n in names) == 36
    parent = {"run": Path("crossfit"), "context": {"prior_run": Path("coordination"), "parent": {"prior_run": Path("support")}}}
    assert audit.inherited_path(parent, "test_sin", "oof_recurrent", 1101)[0].parent == Path("crossfit")
    assert audit.inherited_path(parent, "test_sin", "half", 1101)[0].parent == Path("coordination")
    assert audit.inherited_path(parent, "test_sin", "decay_huber3", 1101) == (Path("support/test_sin-decay_huber3-1101-predictions.npz"), False)


def test_same_run_compute_null_rejects_even_one_ulp_while_old_replay_tolerance_stays():
    p = np.full((1, 25, 3), .5, np.float32)
    r = np.broadcast_to(rotation(0.), (1, 25, 3, 3)).copy()
    assert audit.validate_replay(p, r, p.copy(), r.copy(), exact=True)["bitwise_equal"]
    modified = p.copy(); modified[0, 0, 0] = np.nextafter(modified[0, 0, 0], np.float32(1))
    assert not audit.validate_replay(modified, r, p, r)["bitwise_equal"]
    with pytest.raises(ValueError, match="compute-null"):
        audit.validate_replay(modified, r, p, r, exact=True)


@pytest.mark.parametrize("damage", ["extra", "missing", "mechanism", "timing", "selection", "gate"])
def test_exact_protocol_fields_and_semantics(damage):
    protocol = {**audit.required_settings(), **{k: None for k in ("capacity_experiment", "capacity_report", "data",
                "data_hashes", "checkpoints", "fallback", "sources", "runtime")}}
    audit.validate_settings(protocol)
    if damage == "extra":
        protocol["hidden_extension"] = True
    elif damage == "missing":
        protocol.pop("probe_half")
    else:
        protocol[{"mechanism": "inverse_fit", "timing": "timing_scope", "selection": "no_selection", "gate": "gate"}[damage]] = "changed"
    with pytest.raises(ValueError, match="frozen settings"):
        audit.validate_settings(protocol)


@pytest.mark.parametrize("extra", ["failed.json", "completion-before-error.json", "unaccounted.npz"])
def test_outer_rejects_incomplete_failure_and_extra_members(tmp_path, monkeypatch, extra):
    experiment, out = tmp_path / "experiment", tmp_path / "report"
    run = experiment / "run-01"; run.mkdir(parents=True)
    for name in audit.expected_members() | {extra}:
        (run / name).write_text("{}")
    monkeypatch.setattr(audit, "validate_inputs", lambda *a: {"protocol": {}, "parent": {}})
    with pytest.raises(ValueError, match="exact94"):
        audit.audit(experiment, out, protocol_sha256="synthetic", completed_sha256=audit.sha(run / "completed.json"))
    assert audit.read_json(out / "failed.json")["status"] == "failed"
    assert not (out / "receipt.json").exists()


def test_runner_wrong_protocol_before_output_or_model_loading(tmp_path, monkeypatch):
    import run_pose_probe_blend as runner
    experiment = tmp_path / "experiment"; experiment.mkdir()
    (experiment / "protocol.json").write_text("{}")
    monkeypatch.setattr(runner.coordination, "experts", lambda *a: pytest.fail("must not load experts"))
    with pytest.raises(ValueError, match="external protocol digest"):
        runner.run(experiment, "wrong")
    assert not (experiment / "run-01").exists()


def test_runner_retains_batch_and_returned_timings_after_later_failure(tmp_path, monkeypatch):
    import run_pose_probe_blend as runner
    import torch
    experiment = tmp_path / "experiment"; experiment.mkdir()
    p = torch.zeros(160, 57, 3)
    r = torch.eye(3).expand(160, 57, 3, 3).clone()
    actions = torch.zeros(160, 56, 40)
    ids = np.column_stack((np.repeat(np.arange(10), 16), np.tile(np.arange(16), 10))).astype(np.int64)
    protocol = {"data": "unused", "fallback": {"1101": {"alpha": [.3, .7]}}}
    monkeypatch.setattr(runner, "validate_protocol", lambda *a: protocol)
    monkeypatch.setattr(runner, "load_data", lambda *a: (p, r, actions, ids))
    monkeypatch.setattr(runner.coordination, "experts", lambda *a: (object(), object()))
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(runner.torch, "use_deterministic_algorithms", lambda *a: None)
    monkeypatch.setattr(runner, "measure", lambda *a: {"synthetic": True})
    attempted = []
    original = RuntimeError("injected later forecast failure")
    def fake_forecast(*args, **kwargs):
        attempted.append(kwargs["return_probe"])
        if len(attempted) == 3:
            raise original
        batch = len(args[3])
        value = (p[:batch, :25], r[:batch, :25], torch.ones(batch, 2), {"synthetic_call": len(attempted)})
        return (*value, None) if kwargs["return_probe"] else value
    monkeypatch.setattr(runner, "forecast", fake_forecast)
    with pytest.raises(RuntimeError) as caught:
        runner.run(experiment, "synthetic")
    assert caught.value is original and attempted == [True, False, False]
    run = experiment / "run-01"
    failure = audit.read_json(run / "failed.json")
    assert failure["returned_forecast_calls"] == 2 and failure["completed_rows"] == 0
    assert failure["active_row"]["returned_calls"] == 2
    assert len(failure["active_row"]["warmup_ms"]) == 1 and not failure["active_row"]["latency_ms"]
    assert failure["active_row"]["prediction_sha256"] == audit.sha(run / "test_sin-fast-1101-predictions.npz")
    assert failure["active_row"]["timing_diagnostics"] == [{"synthetic_call": 2}]
    assert (run / "test_sin-inputs.npz").exists() and not (run / "completed.json").exists()
