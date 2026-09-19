"""Handbuilt saved-evidence tests; no models, native calls or random draws."""
import copy
import json
import math

import audit_pose_support as audit
import numpy as np
import pytest


def diagnostic(variant, batch=160):
    base = variant.removesuffix("_mass")
    mass = variant.endswith("_mass")
    huber = "huber3" in base
    fixed = audit._known_stats("decay5" if base == "decay_huber3" else "full" if base == "huber3" else base)
    actual = dict(fixed)
    if mass:
        actual.update(effective_n=30., fraction_weight_lt_one=float(fixed["weight_sum"] < 30))
    calls = 0 if base == "prior" else (3 if huber else 1) + int(mass)
    return {"version": "pose-support-v1", "variant": variant, "base_variant": base,
        "adapted": base != "prior", "mass_control": mass, "support_rows": 30,
        "huber_iterations": 3 if huber else 0, "ridge_precision": 1., "solve_dtype": "float64",
        "batched_solve_calls": calls, "cholesky_factorizations": batch if base == "full" else 6 * batch * calls,
        "actual": {k: np.full((batch, 6), v).tolist() for k, v in actual.items()},
        "derivation": {k: np.full((batch, 6), v).tolist() for k, v in fixed.items()},
        "correction_frobenius_norm": [0. if base == "prior" else 1.] * batch,
        "derivation_correction_frobenius_norm": [0. if base == "prior" else 1.] * batch}


@pytest.mark.parametrize("variant", audit.SUPPORT_VARIANTS)
def test_all_known_compact_diagnostics_and_work(variant):
    value = diagnostic(variant, 2)
    assert audit.validate_diagnostics(value, variant, 2) == value


@pytest.mark.parametrize("damage", ["count", "mass", "ess", "fraction", "norm", "schema", "finite"])
def test_compact_diagnostic_corruptions_rejected(damage):
    value = diagnostic("recent5_mass", 2)
    if damage == "count":
        value["cholesky_factorizations"] -= 1
    elif damage == "mass":
        value["actual"]["weight_sum"][0][0] = 6.
    elif damage == "ess":
        value["actual"]["effective_n"][0][0] = 5.
    elif damage == "fraction":
        value["actual"]["fraction_weight_lt_one"][0][0] = .123
    elif damage == "norm":
        value["correction_frobenius_norm"][0] = -1.
    elif damage == "finite":
        value["derivation"]["weight_sum"][0][0] = float("nan")
    else:
        value["extra"] = 1
    with pytest.raises(ValueError):
        audit.validate_diagnostics(value, "recent5_mass", 2)


def test_prior_zero_convention_and_old_null():
    value = diagnostic("prior", 2)
    value["actual"]["fraction_weight_lt_one"][0][0] = 1
    with pytest.raises(ValueError):
        audit.validate_diagnostics(value, "prior", 2)
    assert audit.validate_diagnostics(None, "gru") is None
    with pytest.raises(ValueError):
        audit.validate_diagnostics({}, "gru")


def test_huber_mass_controls_preserve_derived_mass_not_assignment():
    value = diagnostic("huber3_mass", 2)
    value["derivation"]["weight_sum"] = [[15.] * 6] * 2
    value["derivation"]["effective_n"] = [[20.] * 6] * 2
    value["derivation"]["fraction_weight_lt_one"] = [[.5] * 6] * 2
    value["actual"]["weight_sum"] = [[15.] * 6] * 2
    value["actual"]["fraction_weight_lt_one"] = [[1.] * 6] * 2
    audit.validate_diagnostics(value, "huber3_mass", 2)
    value["actual"]["effective_n"][0][0] = 20.
    with pytest.raises(ValueError, match="ESS"):
        audit.validate_diagnostics(value, "huber3_mass", 2)


def fake_errors(value, parent=None):
    parent = [value] * 10 if parent is None else parent
    return {k: {"mse": value, "rmse": math.sqrt(value), "horizon_mse": [value] * 25,
                "horizon_rmse": [math.sqrt(value)] * 25, "parent_ids": list(range(10)),
                "parent_mse": parent, "parent_rmse": np.sqrt(parent).tolist()} for k in audit.ENDPOINTS}


def rows():
    return {
        p: {f"{v}-{s}" if s else v: {"errors": fake_errors(.81 if v == audit.PRIMARY else 1.),
            "latency_ms": [1.5 if v == audit.PRIMARY else 1.] * 20}
            for v in (*audit.VARIANTS, *audit.REFERENCES)
            for s in (audit.SEEDS if v in audit.VARIANTS else (None,))} for p in audit.PANELS}


def test_all_1141_checks_partitioned_into_17_requirements_at_boundaries():
    families, refs, latency, _, gate = audit.aggregate(rows())
    assert gate["passed"] and gate["checks_passed"] == gate["total_checks"] == 1141
    assert gate["requirements_passed"] == gate["total_requirements"] == 17
    names = [n for r in gate["requirements"] for n in r["comparison_names"]]
    assert len(names) == len(set(names)) == 1141
    assert latency[audit.PRIMARY]["samples"] == 120 and latency["cv1"]["samples"] == 40
    assert len(families["test_sin"]) == 14 and len(refs["test_sin"]) == 6


def test_one_failing_fit_or_parent_cannot_be_rescued():
    values = rows()
    values["test_sin"]["decay_huber3-1101"]["errors"] = fake_errors(1.01)
    values["test_sin"]["decay_huber3-1202"]["errors"] = fake_errors(.01)
    values["test_sin"]["decay_huber3-1303"]["errors"] = fake_errors(.01)
    gate = audit.aggregate(values)[-1]
    assert not next(x for x in gate["requirements"] if x["name"] == "test_sin/position/paired")["passed"]
    assert next(x for x in gate["requirements"] if x["name"] == "test_sin/position/family")["passed"]
    for seed in audit.SEEDS:
        values["test_sin"][f"decay_huber3-{seed}"]["errors"] = fake_errors(.8, [.5] * 7 + [1.5] * 3)
    gate = audit.aggregate(values)[-1]
    assert not next(x for x in gate["requirements"] if x["name"] == "test_sin/position/parents")["passed"]


def test_loo_ties_and_zero_reference_fail():
    values = rows()
    for seed in audit.SEEDS:
        values["test_sin"][f"decay_huber3-{seed}"]["errors"] = fake_errors(.5, [.5] * 10)
        values["test_sin"][f"full-{seed}"]["errors"] = fake_errors(.9, [4.5] + [.5] * 9)
    gate = audit.aggregate(values)[-1]
    assert not next(x for x in gate["checks"] if x["name"] == "test_sin/full/position/leave_out_0")["passed"]
    values["test_sin"]["hold"]["errors"] = fake_errors(0.)
    assert not audit.aggregate(values)[-1]["passed"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prediction_replay_tolerance_and_rejection(tmp_path, dtype):
    p = np.zeros((2, 25, 3), dtype=dtype)
    r = np.broadcast_to(np.eye(3, dtype=dtype), (2, 25, 3, 3)).copy()
    path = tmp_path / "old.npz"; np.savez(path, p=p, R=r)
    assert audit.validate_replay(p, r, path)["p"]["exact_array_equal"]
    tiny = audit.PARITY_TOLERANCE[str(p.dtype)]["atol"] / 2
    p[0, 0, 0] = tiny
    assert not audit.validate_replay(p, r, path)["p"]["exact_array_equal"]
    p[0, 0, 0] = .1
    with pytest.raises(ValueError, match="replay mismatch"):
        audit.validate_replay(p, r, path)


def test_window_start_diagnostic_keeps_all_cases():
    ids = np.column_stack((np.repeat(np.arange(10), 16), np.tile(np.arange(16), 10)))
    e = {k: np.ones((160, 25)) for k in audit.ENDPOINTS}
    e["position"][ids[:, 1] == 0] = 10.
    group = audit.window_groups(e, ids)
    assert group["start_zero"]["windows"] == 10 and group["remaining"]["windows"] == 150
    assert group["start_zero"]["errors"]["position"]["mse"] == 10.
    assert audit.reduce_errors(e, ids)["position"]["mse"] == 1.5625


def build_tree(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"; experiment.mkdir()
    run = experiment / "run-01"; run.mkdir()
    data = tmp_path / "data"; data.mkdir()
    prior = tmp_path / "prior"; prior.mkdir()
    checkpoints = {f"{v}-{s}": {"sha256": f"{v}-{s}"} for s in audit.SEEDS for v in ("meta", "static", "public", "gru")}
    checkpoints["ridge16"] = {"sha256": "ridge"}
    protocol = {"scope": "synthetic fixture only", "sources": {}, "data_hashes": {}, "checkpoints": checkpoints,
                "prior_protocol_sha256": "synthetic", "prior_completed_sha256": "synthetic"}
    audit.write_json(experiment / "protocol.json", protocol)
    ph = audit.sha(experiment / "protocol.json")
    np.savez(data / "normalization.npz", obs_mean=np.zeros(9), obs_scale=np.ones(9))
    obs = np.zeros((160, 57, 9), np.float32); obs[..., 6:9] = 1
    ids = np.column_stack((np.repeat(np.arange(10), 16), np.tile(np.arange(16), 10))).astype(np.int64)
    tp, tr = audit.public_pose(obs.astype(float)); tp, tr = tp[:, 32:], tr[:, 32:]
    for panel in audit.PANELS:
        np.savez(data / (panel + ".npz"), obs=obs, source_ids=ids[:, 0], window_starts=ids[:, 1])
        np.savez(run / (panel + "-targets.npz"), p=tp, R=tr, ids=ids)
        np.savez(prior / (panel + "-targets.npz"), p=tp, R=tr, ids=ids)
    audit.write_json(run / "started.json", {"protocol_sha256": ph})
    saved_rows = []
    for panel in audit.PANELS:
        for variant in (*audit.VARIANTS, *audit.REFERENCES):
            for seed in audit.SEEDS if variant in audit.VARIANTS else (None,):
                label = variant if seed is None else f"{variant}-{seed}"; prefix = panel + "-" + label
                dtype = np.float64 if variant == "ridge16" else np.float32
                pp, rr = tp.astype(dtype), tr.astype(dtype)
                if variant != "hold":
                    pp = pp.copy(); pp[..., 0] = .5 if variant == audit.PRIMARY else 1.
                errors = audit.reduce_errors(audit.error_arrays(pp, rr, tp, tr), ids)
                quality = audit.rotation_quality(rr)
                metrics = {"position_rmse_m": errors["position"]["rmse"], "rotation_rmse_rad": errors["rotation"]["rmse"],
                    "composite": (errors["position"]["mse"] + errors["rotation"]["mse"]) / .01,
                    "rotation_orthogonality_max": quality[0], "rotation_determinant_max_error": quality[1]}
                key = f"{audit.backbone(variant)}-{seed}" if variant in audit.VARIANTS else variant
                checkpoint = checkpoints[key]["sha256"] if key in checkpoints else None
                row = {"panel": panel, "variant": variant, "seed": seed, "checkpoint_sha256": checkpoint,
                    "metrics": metrics, "latency_ms": [1.] * 20,
                    "diagnostics": diagnostic(variant) if variant in audit.SUPPORT_VARIANTS else None}
                audit.write_json(run / (prefix + "-evaluation.json"), row)
                np.savez(run / (prefix + "-predictions.npz"), p=pp, R=rr)
                if variant in ("prior", "full", *audit.OLD_VARIANTS, *audit.REFERENCES):
                    v = {"prior": "meta_prior", "full": "meta"}.get(variant, variant)
                    name = v if seed is None else f"{v}-{seed}"
                    np.savez(prior / (panel + "-" + name + "-predictions.npz"), p=pp, R=rr)
                saved_rows.append(row)
    old_members = audit.members(prior)
    monkeypatch.setattr(audit, "validate_inputs", lambda *args: (protocol, data, prior, old_members))
    done = {"status": "completed", "rows": saved_rows, "new_training_fits": 0, "new_optimizer_updates": 0,
            "wall_seconds": 100., "protocol_sha256": ph, "files": audit.members(run)}
    audit.write_json(run / "completed.json", done)
    return experiment, ph, audit.sha(run / "completed.json")


def test_full_shaped_audit_preserves_failed_gate_and_all_replays(tmp_path, monkeypatch):
    experiment, ph, ch = build_tree(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    receipt = audit.audit(experiment, out, protocol_sha256=ph, completed_sha256=ch)
    assert receipt["status"] == "completed" and receipt["qualification_passed"] is False
    assert receipt["total_checks"] == 1141 and receipt["total_requirements"] == 17
    assert len(receipt["execution_members"]) == 196
    summary = audit.read_json(out / "summary.json")
    assert summary["counts"]["prior_replayed_rows"] == 48
    assert summary["counts"]["position_and_rotation_error_pairs"] == 384000
    assert set(audit.members(out)) == {"summary.json", "window-errors.npz", "receipt.json"}
    with pytest.raises(FileExistsError):
        audit.audit(experiment, out, protocol_sha256=ph, completed_sha256=ch)


@pytest.mark.parametrize("damage", ["extra", "payload", "metrics", "checkpoint", "diagnostics", "target", "prior_replay"])
def test_saved_tree_corruption_rejected(tmp_path, monkeypatch, damage):
    experiment, ph, ch = build_tree(tmp_path, monkeypatch)
    run = experiment / "run-01"; done = audit.read_json(run / "completed.json")
    path = run / "test_sin-prior-1101-evaluation.json"
    row = audit.read_json(path)
    if damage == "extra":
        (run / "unexpected.json").write_text("{}")
    elif damage == "payload":
        path.write_text("{}")
    elif damage == "target":
        target = run / "test_sin-targets.npz"
        with np.load(target) as arrays:
            p, r, ids = arrays["p"], arrays["R"], arrays["ids"]
        p[0, 0, 0] = 1.; np.savez(target, p=p, R=r, ids=ids)
    elif damage == "prior_replay":
        old = tmp_path / "prior/test_sin-meta_prior-1101-predictions.npz"
        with np.load(old) as arrays:
            p, r = arrays["p"], arrays["R"]
        p[0, 0, 0] += 1.; np.savez(old, p=p, R=r)
    else:
        if damage == "metrics":
            row["metrics"]["position_rmse_m"] += 1
        elif damage == "checkpoint":
            row["checkpoint_sha256"] = "other"
        else:
            row["diagnostics"]["batched_solve_calls"] = 3
        path.write_text(json.dumps(row)); done["rows"][0] = row
    if damage not in ("extra", "payload"):
        done["files"] = {k: v for k, v in audit.members(run).items() if k != "completed.json"}
        (run / "completed.json").write_text(json.dumps(done)); ch = audit.sha(run / "completed.json")
    out = tmp_path / "audit"
    with pytest.raises(ValueError):
        audit.audit(experiment, out, protocol_sha256=ph, completed_sha256=ch)
    assert audit.read_json(out / "failed.json")["status"] == "failed"
    assert not (out / "receipt.json").exists()


def test_mass_pair_derivation_must_match_independent_base_row():
    values = {p: {f"{v}-{s}": {"diagnostics": diagnostic(v, 2)} for s in audit.SEEDS for v in audit.SUPPORT_VARIANTS}
              for p in audit.PANELS}
    audit.validate_mass_pairs(values)
    changed = copy.deepcopy(values)
    changed["test_sin"]["huber3_mass-1101"]["diagnostics"]["derivation"]["weight_sum"][0][0] = 20.
    with pytest.raises(ValueError, match="mass derivation"):
        audit.validate_mass_pairs(changed)


def test_reject_unpinned_source_before_execution(tmp_path):
    path = tmp_path / "unexpected.py"; path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="pinned auditor"):
        audit.load_pinned(path, "0" * 64)
