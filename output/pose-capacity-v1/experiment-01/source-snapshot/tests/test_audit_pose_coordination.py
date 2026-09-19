"""Handbuilt arithmetic and tamper tests; no models, native calls or random draws."""
import math

import audit_pose_coordination as audit
import numpy as np
import pytest
import torch


def errors(value, parents=None):
    parents = [value] * 10 if parents is None else parents
    return {e: {"mse": value, "rmse": math.sqrt(value), "horizon_mse": [value] * 25,
                "horizon_rmse": [math.sqrt(value)] * 25, "parent_ids": list(range(10)),
                "parent_mse": parents, "parent_rmse": np.sqrt(parents).tolist()} for e in audit.ENDPOINTS}


def rows():
    return {p: {f"{v}-{s}" if s else v: {"errors": errors(.81 if v == audit.PRIMARY else 1.),
                "latency_ms": [1.5 if v == audit.PRIMARY else 1.] * 20}
                for v in (*audit.VARIANTS, *audit.REFERENCES)
                for s in (audit.SEEDS if v in audit.VARIANTS else (None,))} for p in audit.PANELS}


def test_gate_boundary_and_exact_partition():
    families, refs, latency, contrasts, gate = audit.aggregate(rows())
    assert gate["passed"] and gate["checks_passed"] == gate["total_checks"] == 1501
    assert gate["requirements_passed"] == gate["total_requirements"] == 17
    names = [n for group in gate["requirements"] for n in group["comparison_names"]]
    assert len(names) == len(set(names)) == 1501
    assert len(families["test_sin"]) == 20 and len(refs["test_sin"]) == 6
    assert len(contrasts["test_sin"]) == 25
    assert latency["recurrent"]["samples"] == 120 and latency["cv1"]["samples"] == 40
    assert "fast" not in latency and "slow" not in latency


@pytest.mark.parametrize("violation", ["family", "pair", "parents", "loo", "latency", "zero"])
def test_single_requirement_failure_cannot_be_rescued(violation):
    values = rows()
    if violation == "family":
        for seed in audit.SEEDS:
            values["test_sin"][f"recurrent-{seed}"]["errors"] = errors(.810001)
    elif violation == "pair":
        for seed, value in zip(audit.SEEDS, [1.01, .01, .01], strict=True):
            values["test_sin"][f"recurrent-{seed}"]["errors"] = errors(value)
    elif violation == "parents":
        for seed in audit.SEEDS:
            values["test_sin"][f"recurrent-{seed}"]["errors"] = errors(.8, [.5] * 7 + [1.5] * 3)
    elif violation == "loo":
        for seed in audit.SEEDS:
            values["test_sin"][f"recurrent-{seed}"]["errors"] = errors(.5)
            values["test_sin"][f"half-{seed}"]["errors"] = errors(.9, [4.5] + [.5] * 9)
    elif violation == "latency":
        for panel in audit.PANELS:
            for seed in audit.SEEDS:
                values[panel][f"recurrent-{seed}"]["latency_ms"] = [1.500001] * 20
    else:
        values["test_sin"]["hold"]["errors"] = errors(0.)
    assert not audit.aggregate(values)[-1]["passed"]


def z_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def experts():
    fp = np.full((2, 3, 3), 2., np.float32)
    sp = np.zeros_like(fp)
    fr = np.broadcast_to(z_rotation(.8), (2, 3, 3, 3)).copy()
    sr = np.broadcast_to(np.eye(3, dtype=np.float32), fr.shape).copy()
    return fp, fr, sp, sr


@pytest.mark.parametrize("variant, coefficients", [
    ("fast", [1, 1]), ("slow", [0, 0]), ("position_fast", [1, 0]),
    ("position_slow", [0, 1]), ("half", [.5, .5]), ("recurrent", [.25, .75])])
def test_analytic_composition_and_gate_summaries(variant, coefficients):
    fp, fr, sp, sr = experts()
    alpha = np.tile(np.array(coefficients, np.float32), (2, 1))
    p = np.full_like(fp, 2 * coefficients[0])
    r = np.broadcast_to(z_rotation(.8 * coefficients[1]), fr.shape).copy()
    if coefficients[1] == 1:
        r = fr.copy()
    result = audit.validate_blend(variant, p, r, alpha, fp, fr, sp, sr)
    assert result["position_max_abs_error"] == 0
    assert result["alpha_std"] == [0., 0.]
    assert result["near_pi_relative_rotations"] == 0
    assert result["alpha_mean"] == coefficients


@pytest.mark.parametrize("damage", ["position", "rotation", "endpoint", "alpha", "fixed", "precision"])
def test_blend_corruption_rejected(damage):
    fp, fr, sp, sr = experts()
    alpha = np.ones((2, 2), np.float32)
    p, r = fp.copy(), fr.copy()
    if damage == "position":
        p[0, 0, 0] += .1
    elif damage == "rotation":
        r[0, 0] = z_rotation(.9)
    elif damage == "endpoint":
        p[0, 0, 0] = np.nextafter(p[0, 0, 0], np.float32(3.))
    elif damage == "alpha":
        alpha[0, 0] = np.nan
    elif damage == "fixed":
        alpha[0, 0] = .9
    else:
        p = p.astype(np.float64)
    with pytest.raises(ValueError):
        audit.validate_blend("fast", p, r, alpha, fp, fr, sp, sr)


@pytest.mark.parametrize("angle", [0., 1e-7, .5, math.pi - 1e-6, math.pi])
def test_rotation_log_exp_analytic_edge_cases(angle):
    r = z_rotation(angle).astype(np.float64)
    vector = audit.rotation_log(r)
    assert np.allclose(vector, [0, 0, angle], atol=1e-7, rtol=1e-7)
    assert np.allclose(audit.rotation_exp(vector), r, atol=1e-7, rtol=1e-7)


def test_noncommuting_left_relative_blend():
    fp, _fr, sp, _sr = experts()
    fast = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)
    slow = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    fr = np.broadcast_to(fast, (2, 3, 3, 3)).copy()
    sr = np.broadcast_to(slow, fr.shape).copy()
    alpha = np.full((2, 2), .5, np.float32)
    _p, r, _ = audit.reconstruct_blend(fp, fr, sp, sr, alpha)
    # Halfway quaternion between 90-degree rotations about X and Z.
    expected = np.array([[2/3, -2/3, 1/3], [2/3, 1/3, -2/3], [1/3, 2/3, 2/3]])
    assert np.allclose(r, expected, atol=1e-7)


def test_public_tokens_actual_completed_intervals_and_no_future():
    p = np.zeros((1, 32, 3), np.float32)
    p[0, :, 0] = np.arange(32, dtype=np.float32) * .25
    r = np.broadcast_to(np.eye(3, dtype=np.float32), (1, 32, 3, 3)).copy()
    a = np.zeros((1, 31, 40), np.float32)
    a[0, 30, 7] = .4
    tokens = audit.public_tokens(p, r, a, np.array([.5, .1, 1, 1]))
    assert tokens.shape == (1, 31, 49)
    assert np.allclose(tokens[0, :, 3], math.tanh(.5))
    assert np.all(tokens[0, :, 6:9] == 0)
    assert tokens[0, -1, 16] == np.float32(math.tanh(.4))
    p2 = p.copy(); p2[0, -1, 0] += .125
    changed = audit.public_tokens(p2, r, a, np.array([.5, .1, 1, 1]))
    assert np.array_equal(tokens[:, :-1], changed[:, :-1])
    assert changed[0, -1, 3] > tokens[0, -1, 3]
    with pytest.raises(ValueError, match="token inputs"):
        audit.public_tokens(np.pad(p, ((0, 0), (0, 1), (0, 0))), r, a, np.ones(4))


@pytest.mark.parametrize("damage", ["key", "shape", "dtype", "nonfinite", "initial"])
def test_safe_saved_weight_schema_and_initial_head(tmp_path, damage):
    state = {"logits": torch.zeros(2)}
    path = tmp_path / "weights.pt"
    torch.save(state, path)
    assert audit.tensor_weights(path, "constant", initial=True)["logits"].shape == (2,)
    if damage == "key":
        state["extra"] = torch.zeros(1)
    elif damage == "shape":
        state["logits"] = torch.zeros(3)
    elif damage == "dtype":
        state["logits"] = torch.zeros(2, dtype=torch.float64)
    elif damage == "nonfinite":
        state["logits"][0] = float("nan")
    else:
        state["logits"][0] = .1
    torch.save(state, path)
    with pytest.raises(ValueError):
        audit.tensor_weights(path, "constant", initial=True)


def test_exact_new_payload_contract():
    names = audit.expected_members()
    assert len(names) == 147 and len(audit.SOURCES) == 17
    assert sum(n.endswith("-predictions.npz") for n in names) == 48
    assert sum(n.endswith("-evaluation.json") for n in names) == 48
    assert sum(n.endswith("-initial.pt") for n in names) == 9
    assert sum(n.startswith("train-") for n in names) == 7
    assert "training-completed.json" in names


def test_failure_is_exclusive_and_preserves_original_exception(tmp_path):
    output = tmp_path / "audit"
    with pytest.raises(FileNotFoundError):
        audit.audit(tmp_path / "absent", output, protocol_sha256="a", completed_sha256="b")
    receipt = (output / "failed.json").read_bytes()
    assert audit.read_json(output / "failed.json")["status"] == "failed"
    assert not (output / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        audit.audit(tmp_path / "absent", output, protocol_sha256="a", completed_sha256="b")
    assert (output / "failed.json").read_bytes() == receipt


def test_exact_source_loading_rejects_unbound_bytes(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("raise AssertionError('must never execute')\n")
    with pytest.raises(ValueError, match="source digest"):
        audit.load_pinned(path, "0" * 64)
