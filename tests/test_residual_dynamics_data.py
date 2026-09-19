"""Handwritten numeric fixtures only; no real data, models, simulation or RNG."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_residual_dynamics_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_residual_dynamics_data", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture(n=50, t=1750):
    ids = np.arange(n, dtype=np.float64)[:, None, None]
    time = np.arange(t, dtype=np.float64)[None, :, None]
    pos = np.broadcast_to(ids + time / 10, (n, t, 4)).copy()
    euler = np.broadcast_to(ids / 100 + time / 1000, (n, t, 3)).copy()
    torque = np.broadcast_to(ids + time / 20, (n, t, 4)).copy()
    torque += np.arange(4, dtype=np.float64)
    return pos, euler, torque


def test_exact_predeclared_members_and_disjoint_test_windows():
    assert module.SPAN == 561
    assert module.TRAIN_IDS == tuple(range(30))
    assert module.DEV_IDS == tuple(range(41, 50))
    assert module.TRAIN_STARTS == tuple(range(0, 1190, 50))
    assert module.DEV_STARTS == (0, 594, 1189)
    assert len(module.TEST_STARTS) == 16
    assert module.TEST_STARTS[0] == 0 and module.TEST_STARTS[-1] == 9439
    assert np.all(np.diff(module.DEV_STARTS) >= 561)
    assert np.all(np.diff(module.TEST_STARTS) >= 561)
    assert not set(range(30, 41)) & (set(module.TRAIN_IDS) | set(module.DEV_IDS))


def test_57_observations_and_all_560_torques_exact_alignment():
    raw = fixture()
    obs, actions = module.public_features(*raw, (50, 1750))
    norm = module.normalize_from_training(obs, actions)
    result = module.make_windows(obs, actions, [4], [50], norm)
    assert result["obs"].shape == (1, 57, 9)
    assert result["actions"].shape == (1, 56, 40)
    restored_obs = result["obs"][0] * norm["obs_scale"] + norm["obs_mean"]
    restored_actions = (result["actions"][0].reshape(560, 4)
                        * norm["actions_scale"] + norm["actions_mean"])
    np.testing.assert_allclose(restored_obs, obs[4, 50:611:10], atol=1e-5)
    np.testing.assert_allclose(restored_actions, raw[2][4, 50:610], atol=1e-5)
    # Forecast action block31 begins at last context observation raw index360.
    np.testing.assert_allclose(restored_actions[310:320], raw[2][4, 360:370], atol=1e-5)
    assert result["source_ids"].tolist() == [4]
    assert result["window_starts"].tolist() == [50]
    assert all(not np.shares_memory(x, y) for x in result.values() for y in raw)


def test_normalization_and_delta_use_all_training_pairs_only():
    obs, actions = module.public_features(*fixture(), (50, 1750))
    norm = module.normalize_from_training(obs, actions)
    expected = (obs[:30, 10:] - obs[:30, :-10]).std(axis=(0, 1))
    np.testing.assert_array_equal(norm["delta_std_raw"], expected)
    np.testing.assert_array_equal(norm["delta_std_normalized"], expected / norm["obs_scale"])
    np.testing.assert_array_equal(norm["delta_scale_normalized"],
                                  np.maximum(expected / norm["obs_scale"], 1e-3))
    obs[30:] += 30000
    actions[30:] -= 9000
    changed = module.normalize_from_training(obs, actions)
    for key in norm:
        np.testing.assert_array_equal(norm[key], changed[key])


def test_constant_scale_delta_floor_and_slope_poison():
    raw = list(fixture())
    raw[0][..., 3] = np.nan
    raw[2][..., 1] = 5
    obs, actions = module.public_features(*raw, (50, 1750))
    norm = module.normalize_from_training(obs, actions)
    assert norm["actions_std"][1] == 0 and norm["actions_scale"][1] == 1
    assert np.all(norm["delta_scale_normalized"] >= 1e-3)
    assert np.isfinite(obs).all()
    np.testing.assert_array_equal(obs[0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("change", ["nan", "shape", "dtype"])
def test_invalid_public_inputs_rejected(change):
    raw = list(fixture())
    if change == "nan":
        raw[2][0, 0, 0] = np.inf
    elif change == "shape":
        raw[1] = raw[1][:, :-1]
    else:
        raw[1] = raw[1].astype(np.float32)
    with pytest.raises(ValueError):
        module.public_features(*raw, (50, 1750))


def test_window_overflow_rejected():
    obs, actions = module.public_features(*fixture(), (50, 1750))
    norm = module.normalize_from_training(obs, actions)
    with pytest.raises(ValueError, match="outside"):
        module.make_windows(obs, actions, [0], [1190], norm)


def test_bad_hash_exclusive_failure_preserves_original(tmp_path):
    source = tmp_path / "bad.npz"
    source.write_bytes(b"untrusted")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="identity"):
        module.prepare(source, source, source, out)
    before = (out / "failed.json").read_bytes()
    assert not (out / "completed.json").exists()
    with pytest.raises(FileExistsError):
        module.prepare(source, source, source, out)
    assert (out / "failed.json").read_bytes() == before


def test_actual_serialization_public_allowlist_shapes_and_hashes(tmp_path, monkeypatch):
    # Only tiny repeated numeric values, but exact production axes and serialization.
    paths = {}
    for name, spec in module.SOURCES.items():
        pos, euler, torque = fixture(*spec["shape"])
        pos[..., 3] = np.nan
        poison = np.array([{"unpickling_forbidden": True}], dtype=object)
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, pos=pos, orn_euler=euler, jointAppliedTorques=torque,
                            vel=poison, jointPos=poison, jointVel=poison,
                            jointReactionForces=poison)
        monkeypatch.setitem(spec, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setitem(spec, "bytes", path.stat().st_size)
        paths[name] = path
    out = tmp_path / "prepared"
    receipt = module.prepare(**paths, out=out)
    assert set(receipt["files"]) == {"train.npz", "dev.npz", "test_sin.npz", "test_zigzag.npz",
                                        "normalization.npz", "manifest.json"}
    for name, entry in receipt["files"].items():
        raw = (out / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert len(raw) == entry["bytes"]
    for name, n in (("train", 720), ("dev", 27), ("test_sin", 160), ("test_zigzag", 160)):
        with np.load(out / f"{name}.npz", allow_pickle=False) as arrays:
            assert set(arrays.files) == {"obs", "actions", "source_ids", "window_starts"}
            assert arrays["obs"].shape == (n, 57, 9)
            assert arrays["actions"].shape == (n, 56, 40)
            assert arrays["obs"].dtype == arrays["actions"].dtype == np.float32
            assert arrays["source_ids"].dtype == arrays["window_starts"].dtype == np.int64
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["excluded_old_ids"] == list(range(30, 41))
    assert manifest["loaded_keys"] == ["pos", "orn_euler", "jointAppliedTorques"]
