"""Synthetic numeric arrays only; no upstream data/model/native/RNG calls."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_action_filter_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_action_filter_data", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture():
    ids = np.arange(50, dtype=np.float64)[:, None, None]
    t = np.arange(8, dtype=np.float64)[None, :, None]
    pos = np.broadcast_to(ids + t / 10, (50, 8, 4)).copy()
    euler = np.broadcast_to(ids / 100 + t / 1000, (50, 8, 3)).copy()
    action = np.broadcast_to(ids / 50 + t / 20, (50, 8, 4)).copy()
    return pos, euler, action


def test_fixed_splits_alignment_training_only_and_no_alias():
    pos, euler, action = fixture()
    result, norm, _ = module.prepare_arrays(pos, euler, action)
    assert result["train"]["obs"].shape == (30, 8, 9)
    assert result["test"]["actions"].shape == (11, 7, 4)
    assert result["dev"]["source_ids"].tolist() == list(range(41, 50))
    np.testing.assert_allclose(result["train"]["actions"] * norm["actions_scale"] + norm["actions_mean"], action[:30, :-1], atol=1e-7)
    assert result["train"]["obs"].dtype == np.float32
    assert result["train"]["source_ids"].dtype == np.int64
    for x in (pos, euler, action):
        assert not np.shares_memory(result["train"]["obs"], x)
    pos[30:] += 500
    action[30:] -= 100
    _, changed, _ = module.prepare_arrays(pos, euler, action)
    for key in norm:
        np.testing.assert_array_equal(norm[key], changed[key])


def test_slope_canary_excluded_and_feature_order():
    pos, euler, action = fixture()
    original, norm, _ = module.prepare_arrays(pos, euler, action)
    pos[..., 3] = np.nan
    changed, _, _ = module.prepare_arrays(pos, euler, action)
    np.testing.assert_array_equal(original["train"]["obs"], changed["train"]["obs"])
    actual = original["train"]["obs"][0, 0] * norm["obs_scale"] + norm["obs_mean"]
    np.testing.assert_allclose(actual, [0, 0, 0, 0, 0, 0, 1, 1, 1], atol=1e-7)


@pytest.mark.parametrize("change", ["nan", "dtype", "shape", "duplicate"])
def test_bad_data_rejected(change):
    pos, euler, action = fixture()
    if change == "nan":
        action[1, 0, 0] = np.inf
    elif change == "dtype":
        euler = euler.astype(np.float32)
    elif change == "shape":
        action = action[:, :-1]
    else:
        for x in (pos, euler, action):
            x[30] = x[0]
    with pytest.raises(ValueError):
        module.prepare_arrays(pos, euler, action)


def test_constant_scaling():
    pos, euler, action = fixture()
    action[..., 0] = 2
    result, norm, _ = module.prepare_arrays(pos, euler, action)
    assert norm["actions_std"][0] == 0
    assert norm["actions_scale"][0] == 1
    assert not result["test"]["actions"][..., 0].any()


def test_guard_failure_and_exclusive_output(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"not trusted")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="identity"):
        module.prepare(source, out)
    assert (out / "failed.json").exists()
    assert not (out / "completed.json").exists()
    before = hashlib.sha256((out / "failed.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        module.prepare(source, out)
    assert hashlib.sha256((out / "failed.json").read_bytes()).hexdigest() == before


def test_safe_serialization_ignores_unused_object_payload(tmp_path, monkeypatch):
    ids = np.arange(50, dtype=np.float64)[:, None, None]
    pos = np.broadcast_to(ids, (50, 1750, 4)).copy()
    euler = np.broadcast_to(ids / 100, (50, 1750, 3)).copy()
    torques = np.broadcast_to(ids / 50, (50, 1750, 4)).copy()
    source = tmp_path / "fixture.npz"
    poison = np.array([{"must_not_be_unpickled": True}], dtype=object)
    np.savez_compressed(source, pos=pos, orn_euler=euler,
                        jointAppliedTorques=torques, vel=poison, jointPos=poison,
                        jointVel=poison, jointReactionForces=poison)
    monkeypatch.setattr(module, "SOURCE_BYTES", source.stat().st_size)
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    out = tmp_path / "prepared"
    receipt = module.prepare(source, out)
    assert receipt["status"] == "completed"
    assert set(receipt["files"]) == {"train.npz", "dev.npz", "test.npz", "normalization.npz", "manifest.json"}
    for name, identity in receipt["files"].items():
        raw = (out / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == identity["sha256"]
        assert len(raw) == identity["bytes"]
    with np.load(out / "test.npz", allow_pickle=False) as data:
        assert data["obs"].shape == (11, 1750, 9)
        assert data["actions"].shape == (11, 1749, 4)
        assert data["source_ids"].tolist() == list(range(30, 41))
    assert json.loads((out / "manifest.json").read_text())["loaded_keys"] == ["pos", "orn_euler", "jointAppliedTorques"]
