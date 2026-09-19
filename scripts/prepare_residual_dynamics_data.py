"""Local-only, fixed-window preparation from pinned simulated HiP-RSSM archives.

Every 20 ms transition retains its ten ordered recorded torque vectors. These
are applied torques, not authenticated commands. No upstream code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

REPOSITORY_SHA = "f4adef0ce5ed4b37c9847973df320f72965bd330"
BASE_URL = f"https://raw.githubusercontent.com/ALRhub/HiP-RSSM/{REPOSITORY_SHA}/"
SOURCES = {
    "old": {
        "path": "data/MobileRobot/sin2/ts_002_50x2000_w_grad.npz",
        "sha256": "361e6f45b917e9883b004fc90bebaab5d7fa5ca935b1423aac3f2f4fe49d4dc6",
        "bytes": 30162192, "shape": (50, 1750),
    },
    "sin": {
        "path": "data/MobileRobot/sin_infer/ts_0.002_10x10000_w_grad.npz",
        "sha256": "ce23ebe3b029799477c4c319ef4186aea5a917e53bfccc0686ad03e7e3d4b45b",
        "bytes": 33859211, "shape": (10, 10000),
    },
    "zigzag": {
        "path": "data/MobileRobot/sin_infer/zigzag_ts_0.002_10x10000_w_grad.npz",
        "sha256": "746361b67f82a409e6b7e587c574c83c1b832988e6be7c5365aa4231231bb9b3",
        "bytes": 34105554, "shape": (10, 10000),
    },
}
MEMBERS = {name + ".npy" for name in (
    "pos", "orn_euler", "vel", "jointPos", "jointVel",
    "jointReactionForces", "jointAppliedTorques")}
STRIDE, CONTEXT, HORIZON = 10, 32, 25
SPAN = (CONTEXT + HORIZON - 1) * STRIDE + 1
TRAIN_IDS = tuple(range(30))
DEV_IDS = tuple(range(41, 50))
TRAIN_STARTS = tuple(range(0, 1750 - SPAN + 1, 50))
DEV_STARTS = tuple(int(x) for x in np.linspace(0, 1750 - SPAN, 3, dtype=np.int64))
TEST_STARTS = tuple(int(x) for x in np.linspace(0, 10000 - SPAN, 16, dtype=np.int64))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def public_features(pos, euler, torques, shape):
    for name, value, width in (("pos", pos, 4), ("orn_euler", euler, 3),
                                ("jointAppliedTorques", torques, 4)):
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise ValueError(f"{name}: expected float64")
        if value.shape != (*shape, width):
            raise ValueError(f"{name}: unexpected shape")
    # The slope channel is deliberately neither used nor checked for finiteness.
    if not all(np.isfinite(x).all() for x in (pos[..., :3], euler, torques)):
        raise ValueError("nonfinite public input")
    obs = np.concatenate((pos[..., :3], np.sin(euler), np.cos(euler)), axis=-1)
    return obs, torques[:, :-1].copy()


def normalize_from_training(obs, actions):
    normalizer = {}
    for name, array in (("obs", obs), ("actions", actions)):
        train = array[list(TRAIN_IDS)]
        mean, std = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
        normalizer[name + "_mean"] = mean
        normalizer[name + "_std"] = std
        normalizer[name + "_scale"] = np.where(std == 0, 1., std)
    delta = obs[list(TRAIN_IDS), STRIDE:] - obs[list(TRAIN_IDS), :-STRIDE]
    # Prediction uses an uncentered delta; no delta mean is subtracted.
    normalizer["delta_std_raw"] = delta.std(axis=(0, 1))
    normalizer["delta_std_normalized"] = normalizer["delta_std_raw"] / normalizer["obs_scale"]
    normalizer["delta_scale_normalized"] = np.maximum(normalizer["delta_std_normalized"], 1e-3)
    if not all(np.isfinite(x).all() for x in normalizer.values()):
        raise ValueError("nonfinite training normalization")
    return normalizer


def make_windows(obs, actions, ids, starts, norm):
    values, blocks, window_ids = [], [], []
    for i in ids:
        for start in starts:
            if start < 0 or start + SPAN > obs.shape[1]:
                raise ValueError("window outside trajectory")
            sampled = obs[i, start:start + SPAN:STRIDE]
            ordered = actions[i, start:start + SPAN - 1]
            values.append((sampled - norm["obs_mean"]) / norm["obs_scale"])
            blocks.append(((ordered - norm["actions_mean"]) / norm["actions_scale"])
                          .reshape(CONTEXT + HORIZON - 1, STRIDE * 4))
            window_ids.append((i, start))
    result = {"obs": np.asarray(values, dtype=np.float32),
              "actions": np.asarray(blocks, dtype=np.float32),
              "source_ids": np.asarray(window_ids, dtype=np.int64)[:, 0].copy(),
              "window_starts": np.asarray(window_ids, dtype=np.int64)[:, 1].copy()}
    if not np.isfinite(result["obs"]).all() or not np.isfinite(result["actions"]).all():
        raise ValueError("nonfinite normalized output")
    return result


def read_source(path, name):
    raw = Path(path).read_bytes()
    spec = SOURCES[name]
    if len(raw) != spec["bytes"] or digest(raw) != spec["sha256"]:
        raise ValueError(f"{name}: source identity mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != len(MEMBERS) or {x.filename for x in entries} != MEMBERS:
            raise ValueError("archive member mismatch")
        if sum(x.file_size for x in entries) > 100_000_000:
            raise ValueError("unexpected decompressed size")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        return public_features(arrays["pos"], arrays["orn_euler"],
                               arrays["jointAppliedTorques"], spec["shape"])


def prepare(old: Path, sin: Path, zigzag: Path, out: Path):
    """Exclusive preparation only; no fits, outcome selection or random draws."""
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    try:
        paths = {"old": old, "sin": sin, "zigzag": zigzag}
        arrays = {name: read_source(path, name) for name, path in paths.items()}
        norm = normalize_from_training(*arrays["old"])
        plans = {"train": ("old", TRAIN_IDS, TRAIN_STARTS),
                 "dev": ("old", DEV_IDS, DEV_STARTS),
                 "test_sin": ("sin", tuple(range(10)), TEST_STARTS),
                 "test_zigzag": ("zigzag", tuple(range(10)), TEST_STARTS)}
        shapes = {}
        for name, (source, ids, starts) in plans.items():
            windows = make_windows(*arrays[source], ids, starts, norm)
            shapes[name] = {key: list(value.shape) for key, value in windows.items()}
            with (out / f"{name}.npz").open("xb") as stream:
                np.savez_compressed(stream, **windows)
        with (out / "normalization.npz").open("xb") as stream:
            np.savez_compressed(stream, **norm)
        manifest = {
            "version": "residual-dynamics-data-v1", "status": "prepared_not_trained",
            "source_sha256": digest(Path(__file__).read_bytes()),
            "sources": {name: {**spec, "url": BASE_URL + spec["path"],
                                "repository_sha": REPOSITORY_SHA} for name, spec in SOURCES.items()},
            "splits": {name: {"archive": source, "source_ids": list(ids), "starts": list(starts)}
                       for name, (source, ids, starts) in plans.items()},
            "excluded_old_ids": list(range(30, 41)),
            "excluded_old_ids_status": "previous test already exposed; development-only if subsequently used, excluded here",
            "shapes": shapes, "context_frames": CONTEXT, "forecast_steps": HORIZON,
            "raw_stride": STRIDE, "raw_window_frames": SPAN,
            "nominal_raw_dt_seconds": .002, "nominal_model_dt_seconds": .02,
            "nominal_forecast_seconds": .5,
            "feature_order": ["x", "y", "z", "sin_roll", "sin_pitch", "sin_yaw",
                              "cos_roll", "cos_pitch", "cos_yaw"],
            "action_order": "time-major: torque[t,0:4], torque[t+1,0:4], ..., torque[t+9,0:4]; no averaging",
            "alignment": "sampled obs[k] at raw start+10*k; action block k contains raw torques start+10*k through start+10*k+9; next obs at start+10*k+10",
            "normalization": "float64 mean/population std over old train IDs0..29 only, all raw frames; actions omit terminal torque; constant std uses scale1; window outputs float32",
            "delta_scale": "delta_std_raw: population std of all training raw obs[t+10]-obs[t]; delta_std_normalized=delta_std_raw/obs_scale; delta_scale_normalized=max(delta_std_normalized,1e-3); do not subtract delta mean",
            "structural_identity_review": {
                "scope": "exact identity only, no predictive outcome inspection; no independent-seed/terrain proof",
                "old_full_archive_sha256": "5966ea52c06a75c54e8a782d4c23f56918bbfb807ae5a1c64f721babaf391d97",
                "old_full_vs_old_used": "all50 old used trajectories equal full raw frames250:2000 for xyz, Euler and torques",
                "cross_archive_pairs": ["old_full:sin", "old_full:zigzag", "sin:zigzag"],
                "exact_xyz_euler_frame_matches_per_pair": [0, 0, 0],
                "exact_xyz_euler_torque_frame_matches_per_pair": [0, 0, 0],
                "fresh_archive_git_blob_identity_checked": True,
            },
            "loaded_keys": ["pos", "orn_euler", "jointAppliedTorques"],
            "excluded_fields": ["pos[...,3] terrain slope", "vel", "jointPos", "jointVel", "jointReactionForces"],
            "limits": ["Simulated PyBullet data, not real robot measurements.",
                       "Recorded applied torques, not authenticated issued commands; offline conditional forecasting only.",
                       "500Hz supported by publisher paper/configuration; no per-frame timestamps or collection generator supplied.",
                       "Archives have distinct hashes and no exact pose-frame matches in prior identity-only review; independent seeds/terrain topology remain unverified.",
                       "Training windows overlap and windows within each test trajectory are dependent; trajectory is the aggregation unit.",
                       "No explicit upstream data/code license found; all raw and normalized arrays remain local.",
                       "No model/native/RNG calls or predictive outcomes inspected."],
        }
        write_json(out / "manifest.json", manifest)
        for name, path in paths.items():
            if digest(Path(path).read_bytes()) != SOURCES[name]["sha256"]:
                raise ValueError("source changed during preparation")
        files = {p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                 for p in sorted(out.iterdir()) if p.is_file()}
        receipt = {"status": "completed", "scope": "local_offline_data_preparation_only",
                   "source_sha256": manifest["source_sha256"],
                   "input_sha256": {name: spec["sha256"] for name, spec in SOURCES.items()},
                   "files": files, "wall_seconds_before_receipt": time.monotonic() - start,
                   "new_model_calls": 0, "new_native_calls": 0, "new_random_draws": 0}
        write_json(out / "completed.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                       "wall_seconds": time.monotonic() - start})
        except BaseException as secondary:  # noqa: BLE001 - keep original exception
            error.add_note(f"Failure receipt could not be preserved: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "sin", "zigzag", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    prepare(args.old, args.sin, args.zigzag, args.out)
