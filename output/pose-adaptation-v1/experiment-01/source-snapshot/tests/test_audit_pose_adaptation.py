"""Constructed geometry, gate and saved-byte tests; no models or random draws."""
import hashlib
import json
import math

import audit_pose_adaptation as audit
import numpy as np
import pytest
import torch


def rotation_z(angle):
    return np.array([[math.cos(angle), -math.sin(angle), 0.],
                     [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])


@pytest.mark.parametrize("angle", [0., 1e-9, .4, math.pi / 2, math.pi - 1e-9, math.pi, 2 * math.pi - .1])
def test_angles_include_wrap_identity_and_cut(angle):
    expected = min(angle, 2 * math.pi - angle)
    assert audit.rotation_angles(rotation_z(angle), np.eye(3)) == pytest.approx(expected, abs=1e-14)


def test_geodesic_invariant_to_common_left_rotation():
    a, b, q = rotation_z(.4), rotation_z(-.7), audit.public_pose(np.array([0., 0., 0., 1., 0., 0., 0., 1., 1.]))[1].astype(float)
    assert audit.rotation_angles(q @ a, q @ b) == pytest.approx(audit.rotation_angles(a, b))


@pytest.mark.parametrize("matrix", [np.diag([1., 1., -1.]), 2 * np.eye(3), np.full((3, 3), np.nan)])
def test_improper_or_nonfinite_rotations_rejected(matrix):
    with pytest.raises(ValueError):
        audit.rotation_angles(matrix, np.eye(3))


def test_public_euler_conversion_and_degenerate_pair():
    p, r = audit.public_pose(np.array([2., 3., 4., 0., 0., 1., 1., 1., 0.]))
    np.testing.assert_array_equal(p, [2, 3, 4])
    np.testing.assert_allclose(r, rotation_z(math.pi / 2), atol=1e-7)
    with pytest.raises(ValueError, match="degenerate"):
        audit.public_pose(np.zeros(9))


def test_error_reduction_sums_xyz_and_pools_before_sqrt():
    ids = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    p = np.zeros((4, 2, 3)); p[2:, :, 0] = 2
    r = np.broadcast_to(np.eye(3), (4, 2, 3, 3)).copy()
    values = audit.error_arrays(p, r, np.zeros_like(p), r)
    metrics = audit.reduce_errors(values, ids)
    assert metrics["position"]["mse"] == 2
    assert metrics["position"]["rmse"] == math.sqrt(2)
    assert metrics["position"]["parent_mse"] == [0, 4]
    pooled = audit.pooled([metrics, metrics])
    assert pooled == metrics
    p[0, 0, 2] = np.nan
    with pytest.raises(ValueError, match="position"):
        audit.error_arrays(p, r, np.zeros_like(p), r)


def fake_errors(value, parent=None):
    parent = [value] * 10 if parent is None else parent
    return {k: {"mse": value, "rmse": math.sqrt(value), "horizon_mse": [value] * 25,
                "horizon_rmse": [math.sqrt(value)] * 25, "parent_ids": list(range(10)),
                "parent_mse": parent, "parent_rmse": np.sqrt(parent).tolist()} for k in audit.ENDPOINTS}


def rows():
    result = {}
    for panel in audit.PANELS:
        result[panel] = {}
        for variant in (*audit.VARIANTS, *audit.REFERENCES):
            for seed in audit.SEEDS if variant in audit.VARIANTS else (None,):
                label = variant if seed is None else f"{variant}-{seed}"
                result[panel][label] = {"errors": fake_errors(.81 if variant == "meta" else 1.),
                    "latency_ms": [1.5 if variant == "meta" else 1.] * 20}
    return result


def test_all_661_checks_and_inclusive_mean_latency_boundaries():
    families, refs, latency, contrasts, gate = audit.aggregate(rows())
    assert gate["passed"] and gate["checks_passed"] == gate["total_checks"] == 661
    assert latency["meta"]["samples"] == 120 and latency["cv1"]["samples"] == 40
    assert len(refs["test_sin"]) == 6
    assert families["test_sin"]["meta"]["position"]["rmse"] == pytest.approx(.9)
    assert contrasts["test_sin"]["static"]["position"]["parents_nonworse"] == 10


def test_paired_loss_cannot_be_rescued_by_other_fit_gains():
    data = rows()
    data["test_sin"]["meta-1101"]["errors"] = fake_errors(1.01)
    data["test_sin"]["meta-1202"]["errors"] = fake_errors(.1)
    data["test_sin"]["meta-1303"]["errors"] = fake_errors(.1)
    checks = audit.aggregate(data)[-1]["checks"]
    assert not next(c for c in checks if c["name"] == "test_sin/static/position/pair_1101_nonworse")["passed"]
    assert next(c for c in checks if c["name"] == "test_sin/static/position/family_10percent")["passed"]


def test_leave_one_out_prevents_single_parent_rescue():
    data = rows()
    # Candidate improves nine parents slightly; the tenth is responsible for all net loss.
    for seed in audit.SEEDS:
        data["test_sin"][f"meta-{seed}"]["errors"] = fake_errors(.5, [.5] * 10)
        data["test_sin"][f"static-{seed}"]["errors"] = fake_errors(.81, [4.5] + [.4] * 9)
    checks = audit.aggregate(data)[-1]["checks"]
    assert not next(c for c in checks if c["name"] == "test_sin/static/position/leave_out_0")["passed"]
    assert next(c for c in checks if c["name"] == "test_sin/static/position/family_10percent")["passed"]


def test_parent_count_and_strict_loo_ties():
    data = rows()
    for seed in audit.SEEDS:
        data["test_sin"][f"meta-{seed}"]["errors"] = fake_errors(.8, [.5] * 7 + [1.5] * 3)
    checks = audit.aggregate(data)[-1]["checks"]
    assert not next(c for c in checks if c["name"] == "test_sin/static/position/parents_nonworse")["passed"]
    data = rows()
    for seed in audit.SEEDS:
        data["test_sin"][f"meta-{seed}"]["errors"] = fake_errors(1.)
    checks = audit.aggregate(data)[-1]["checks"]
    assert next(c for c in checks if c["name"] == "test_sin/static/position/pair_1101_nonworse")["passed"]
    assert not next(c for c in checks if c["name"] == "test_sin/static/position/leave_out_0")["passed"]


def weights(variant):
    if variant != "gru":
        result = {"scales": torch.ones(4), "prior": torch.zeros(50 if variant == "public" else 13, 6)}
        if variant != "public":
            result.update({"encoder.0.weight": torch.zeros(48, 49), "encoder.0.bias": torch.zeros(48),
                           "encoder.2.weight": torch.zeros(12, 48), "encoder.2.bias": torch.zeros(12)})
        return result
    width = 48
    result = {"scales": torch.ones(4), "readout.weight": torch.zeros(6, width), "readout.bias": torch.zeros(6)}
    for prefix, count in (("observation_update", 9), ("transition", 49)):
        for key, shape in (("weight_ih", (3 * width, count)), ("weight_hh", (3 * width, width)),
                           ("bias_ih", (3 * width,)), ("bias_hh", (3 * width,))):
            result[prefix + "." + key] = torch.zeros(shape)
    return result


@pytest.mark.parametrize("variant", audit.TRAIN_VARIANTS)
def test_safe_tensor_schema_and_initial_zero_prior(tmp_path, variant):
    path = tmp_path / "weights.pt"
    state = weights(variant); torch.save(state, path)
    assert audit.tensor_weights(path, variant, initial=True)["scales"].tolist() == [1.] * 4
    key = "readout.bias" if variant == "gru" else "prior"
    state[key].view(-1)[0] = 1; torch.save(state, path)
    with pytest.raises(ValueError, match="initial"):
        audit.tensor_weights(path, variant, initial=True)
    state[key].view(-1)[0] = float("nan"); torch.save(state, path)
    with pytest.raises(ValueError, match="tensor"):
        audit.tensor_weights(path, variant)


def build_saved_tree(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"; experiment.mkdir()
    run = experiment / "run-01"; run.mkdir()
    data = tmp_path / "data"; data.mkdir()
    protocol = {"scope": "synthetic engineering only", "sources": {}, "data_hashes": {}}
    audit.write_json(experiment / "protocol.json", protocol)
    ph = audit.sha(experiment / "protocol.json")
    monkeypatch.setattr(audit, "validate_inputs", lambda *args: (protocol, {}, data))
    np.savez(data / "normalization.npz", obs_mean=np.zeros(9), obs_scale=np.ones(9))
    obs = np.zeros((160, 57, 9), np.float32); obs[..., 6:9] = 1
    ids = np.column_stack((np.repeat(np.arange(10), 16), np.tile(np.arange(16), 10))).astype(np.int64)
    tp, tr = audit.public_pose(obs.astype(float)); tp, tr = tp[:, 32:], tr[:, 32:]
    for panel in audit.PANELS:
        np.savez(data / (panel + ".npz"), obs=obs, source_ids=ids[:, 0], window_starts=ids[:, 1])
        np.savez(run / (panel + "-targets.npz"), p=tp, R=tr, ids=ids)
    audit.write_json(run / "started.json", {"protocol_sha256": ph})
    fits = []
    for seed in audit.SEEDS:
        for variant in audit.TRAIN_VARIANTS:
            label = f"{variant}-{seed}"; state = weights(variant)
            torch.save(state, run / (label + "-initial.pt")); torch.save(state, run / (label + ".pt"))
            np.save(run / (label + "-losses.npy"), np.zeros(690))
            fit = {"variant": variant, "seed": seed, "updates": 690, "scales": [1.] * 4,
                "parameters": sum(v.numel() for k, v in state.items() if k != "scales"),
                "training_seconds": .1, "initial_sha256": audit.sha(run / (label + "-initial.pt")),
                "checkpoint_sha256": audit.sha(run / (label + ".pt"))}
            audit.write_json(run / (label + "-fit.json"), fit); fits.append(fit)
    (run / "ridge16.pt").write_bytes(b"authenticated opaque ridge fixture")
    audit.write_json(run / "training-completed.json", {"fits": fits, "ridge_seconds": .1,
                                                        "development_panel_forecasts_started": False})
    saved_rows = []
    for panel in audit.PANELS:
        for variant in (*audit.VARIANTS, *audit.REFERENCES):
            for seed in audit.SEEDS if variant in audit.VARIANTS else (None,):
                label = variant if seed is None else f"{variant}-{seed}"
                prefix = panel + "-" + label
                dtype = np.float64 if variant == "ridge16" else np.float32
                pp, rr = tp.astype(dtype), tr.astype(dtype)
                if variant != "hold":
                    pp = pp.copy(); pp[..., 0] = .5 if variant == "meta" else 1
                    rr = np.broadcast_to(rotation_z(.5 if variant == "meta" else 1), rr.shape).astype(dtype).copy()
                result = audit.reduce_errors(audit.error_arrays(pp, rr, tp, tr), ids)
                q = audit.rotation_quality(rr)
                metrics = {"position_rmse_m": result["position"]["rmse"], "rotation_rmse_rad": result["rotation"]["rmse"],
                    "composite": (result["position"]["mse"] + result["rotation"]["mse"]) / .01,
                    "rotation_orthogonality_max": q[0], "rotation_determinant_max_error": q[1]}
                row = {"panel": panel, "variant": variant, "seed": seed, "metrics": metrics, "latency_ms": [1.] * 20}
                audit.write_json(run / (prefix + "-evaluation.json"), row)
                np.savez(run / (prefix + "-predictions.npz"), p=pp, R=rr)
                saved_rows.append(row)
    done = {"status": "completed", "fits": fits, "rows": saved_rows, "ridge_seconds": .1,
            "wall_seconds": 100., "protocol_sha256": ph, "files": audit.members(run)}
    audit.write_json(run / "completed.json", done)
    return experiment, ph, audit.sha(run / "completed.json")


def test_full_shaped_saved_audit_preserves_failed_gate(tmp_path, monkeypatch):
    experiment, protocol_sha, completion_sha = build_saved_tree(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    result = audit.audit(experiment, out, protocol_sha256=protocol_sha, completed_sha256=completion_sha)
    assert result["status"] == "completed" and result["qualification_passed"] is False
    assert result["total_checks"] == 661 and result["externally_supplied_hashes"] is True
    assert set(audit.members(out)) == {"summary.json", "window-errors.npz", "receipt.json"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["recorded_updates"] == 8280
    assert len(result["execution_members"]) == 150
    with pytest.raises(FileExistsError):
        audit.audit(experiment, out, protocol_sha256=protocol_sha, completed_sha256=completion_sha)


@pytest.mark.parametrize("damage", ["payload", "extra", "target", "metrics", "initial"])
def test_resealed_or_unsealed_corruption_rejected(tmp_path, monkeypatch, damage):
    experiment, protocol_sha, completion_sha = build_saved_tree(tmp_path, monkeypatch)
    run = experiment / "run-01"
    done = json.loads((run / "completed.json").read_text())
    if damage == "payload":
        (run / "meta-1101-losses.npy").write_bytes(b"broken")
    elif damage == "extra":
        (run / "unexpected.json").write_text("{}")
    elif damage == "target":
        path = run / "test_sin-targets.npz"
        with np.load(path) as values:
            p, r, ids = values["p"], values["R"], values["ids"]
        p[0, 0, 0] = 1
        np.savez(path, p=p, R=r, ids=ids)
    elif damage == "metrics":
        path = run / "test_sin-meta-1101-evaluation.json"
        row = json.loads(path.read_text()); row["metrics"]["position_rmse_m"] += 1
        path.write_text(json.dumps(row)); done["rows"][0] = row
    else:
        path = run / "static-1101-initial.pt"
        value = weights("static"); value["encoder.0.bias"][0] = 1; torch.save(value, path)
        fit_path = run / "static-1101-fit.json"
        fit = json.loads(fit_path.read_text()); fit["initial_sha256"] = audit.sha(path)
        fit_path.write_text(json.dumps(fit)); done["fits"][1] = fit
        boundary_path = run / "training-completed.json"; boundary = json.loads(boundary_path.read_text())
        boundary["fits"][1] = fit; boundary_path.write_text(json.dumps(boundary))
    if damage not in ("payload", "extra"):
        done["files"] = {k: v for k, v in audit.members(run).items() if k != "completed.json"}
        (run / "completed.json").write_text(json.dumps(done))
        completion_sha = hashlib.sha256((run / "completed.json").read_bytes()).hexdigest()
    out = tmp_path / "audit"
    with pytest.raises(ValueError):
        audit.audit(experiment, out, protocol_sha256=protocol_sha, completed_sha256=completion_sha)
    assert json.loads((out / "failed.json").read_text())["status"] == "failed"
    assert not (out / "receipt.json").exists()


def test_expected_files_exact_and_no_reference_fit_replication():
    expected = audit.expected_members()
    assert len(expected) == 150
    assert "test_sin-cv16-predictions.npz" in expected
    assert "test_sin-cv16-1101-predictions.npz" not in expected


def test_latency_failure_and_zero_baseline_are_not_rescued():
    values = rows()
    for panel in audit.PANELS:
        for seed in audit.SEEDS:
            values[panel][f"meta-{seed}"]["latency_ms"] = [1.500001] * 20
    gate = audit.aggregate(values)[-1]
    assert gate["checks_passed"] == 660 and gate["checks"][-1]["passed"] is False
    values = rows(); values["test_sin"]["hold"]["errors"] = fake_errors(0)
    gate = audit.aggregate(values)[-1]
    assert not next(c for c in gate["checks"] if c["name"] == "test_sin/hold/position/family_10percent")["passed"]


def test_required_external_hash_pair():
    with pytest.raises(ValueError, match="both external"):
        audit.audit(None, None, protocol_sha256="0" * 64)


def test_seventeen_groups_partition_all_661_checks():
    gate = audit.aggregate(rows())[-1]
    assert gate["requirements_passed"] == gate["total_requirements"] == 17
    names = [n for group in gate["requirements"] for n in group["comparison_names"]]
    assert len(names) == len(set(names)) == 661
    assert set(names) == {c["name"] for c in gate["checks"]}
    data = rows()
    data["test_sin"]["meta-1101"]["errors"] = fake_errors(1.01)
    gate = audit.aggregate(data)[-1]
    assert not next(x for x in gate["requirements"] if x["name"] == "test_sin/position/paired")["passed"]


def test_aliases_bind_existing_fits_without_creating_extra_weights(tmp_path, monkeypatch):
    experiment, ph, ch = build_saved_tree(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    audit.audit(experiment, out, protocol_sha256=ph, completed_sha256=ch)
    summary = audit.read_json(out / "summary.json")
    for panel in audit.PANELS:
        for seed in audit.SEEDS:
            rows = summary["rows"][panel]
            assert rows[f"meta-{seed}"]["checkpoint_sha256"] == rows[f"meta_prior-{seed}"]["checkpoint_sha256"]
            assert rows[f"static-{seed}"]["checkpoint_sha256"] == rows[f"static_adapt-{seed}"]["checkpoint_sha256"]
    assert summary["counts"]["neural_rows"] == 36
    assert "meta_prior-1101.pt" not in audit.expected_members()
    assert "static_adapt-1101.pt" not in audit.expected_members()


def test_arithmetic_dependency_uses_exact_source_bytes(tmp_path):
    path = tmp_path / "arithmetic.py"
    path.write_text("raise RuntimeError('not executed')")
    with pytest.raises(ValueError, match="source digest"):
        audit.load_arithmetic(path)
