"""Safe, fixed-trajectory HiP-RSSM data preparation; no upstream code execution.

The mobile robot is simulated. Recorded applied torques are conditioning inputs,
not authenticated commanded actions. Source and derivative data remain local.
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
SOURCE_SHA256 = "361e6f45b917e9883b004fc90bebaab5d7fa5ca935b1423aac3f2f4fe49d4dc6"
SOURCE_BYTES = 30162192
SOURCE_URL = (
    "https://raw.githubusercontent.com/ALRhub/HiP-RSSM/" + REPOSITORY_SHA
    + "/data/MobileRobot/sin2/ts_002_50x2000_w_grad.npz"
)
SPLITS = {"train": tuple(range(30)), "dev": tuple(range(41, 50)),
          "test": tuple(range(30, 41))}
MEMBERS = {name + ".npy" for name in (
    "pos", "orn_euler", "vel", "jointPos", "jointVel",
    "jointReactionForces", "jointAppliedTorques")}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_arrays(pos, euler, torques):
    """Pure fixed-ID transformation; synthetic fixtures may use shorter time axes."""
    for name, array, width in (("pos", pos, 4), ("orn_euler", euler, 3),
                               ("jointAppliedTorques", torques, 4)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise ValueError(f"{name}: expected float64 ndarray")
        if array.ndim != 3 or array.shape[0] != 50 or array.shape[2] != width:
            raise ValueError(f"{name}: invalid shape")
    if pos.shape[1] < 2 or pos.shape[:2] != euler.shape[:2] or pos.shape[:2] != torques.shape[:2]:
        raise ValueError("unaligned time axes")
    # The fourth position channel is terrain slope and never enters features.
    xyz = pos[..., :3]
    for array in (xyz, euler, torques):
        if not np.isfinite(array).all():
            raise ValueError("nonfinite public conditioning arrays")
    observations = np.concatenate((xyz, np.sin(euler), np.cos(euler)), axis=-1)
    actions = torques[:, :-1].copy()
    # Reject exact duplicate full public trajectories across the held-out boundary.
    owners = {}
    trajectory_hashes = {}
    for split, indices in SPLITS.items():
        for i in indices:
            sha = digest(observations[i].tobytes() + actions[i].tobytes())
            if sha in owners and owners[sha] != split:
                raise ValueError("duplicate trajectory across splits")
            owners[sha] = split
            trajectory_hashes[str(i)] = sha
    normalizer = {}
    for name, array in (("obs", observations), ("actions", actions)):
        train = array[list(SPLITS["train"])]
        mean, std = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
        normalizer[name + "_mean"] = mean
        normalizer[name + "_std"] = std
        # Exact constant dimensions use scale one; retain the actual zero std.
        normalizer[name + "_scale"] = np.where(std == 0, 1.0, std)
    result = {}
    for split, indices in SPLITS.items():
        ids = np.array(indices, dtype=np.int64)
        obs = ((observations[ids] - normalizer["obs_mean"]) / normalizer["obs_scale"]).astype(np.float32)
        act = ((actions[ids] - normalizer["actions_mean"]) / normalizer["actions_scale"]).astype(np.float32)
        if not np.isfinite(obs).all() or not np.isfinite(act).all():
            raise ValueError("nonfinite normalized output")
        result[split] = {"obs": obs, "actions": act, "source_ids": ids}
    return result, normalizer, trajectory_hashes


def prepare(source: Path, out: Path) -> dict:
    """Write once from the pinned safe NPZ. Failure preserves its output prefix."""
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    try:
        raw = source.read_bytes()
        if len(raw) != SOURCE_BYTES or digest(raw) != SOURCE_SHA256:
            raise ValueError("source byte identity mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) != len(MEMBERS) or {e.filename for e in entries} != MEMBERS:
                raise ValueError("unexpected archive members")
            if sum(e.file_size for e in entries) > 100_000_000:
                raise ValueError("unexpected decompressed size")
        # No pickle, remote imports, simulator, model, random draws or test metrics.
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            pos, euler, torques = arrays["pos"], arrays["orn_euler"], arrays["jointAppliedTorques"]
        if pos.shape != (50, 1750, 4) or euler.shape != (50, 1750, 3) or torques.shape != (50, 1750, 4):
            raise ValueError("unexpected pinned source dimensions")
        splits, normalizer, hashes = prepare_arrays(pos, euler, torques)
        for split, arrays in splits.items():
            with (out / f"{split}.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
        with (out / "normalization.npz").open("xb") as stream:
            np.savez_compressed(stream, **normalizer)
        manifest = {
            "version": "action-filter-data-v1", "status": "prepared_not_trained",
            "source": {"url": SOURCE_URL, "repository_sha": REPOSITORY_SHA,
                       "sha256": SOURCE_SHA256, "bytes": len(raw)},
            "upstream_loader": {
                "url": "https://github.com/ALRhub/HiP-RSSM/blob/" + REPOSITORY_SHA + "/data/mobileDataSeq.py",
                "sha256": "d975063583c7cedfaeb99048fdbd1a94e23870b907a9b035a99f474f31d9c383"},
            "source_sha256": digest(Path(__file__).read_bytes()),
            "splits": {k: list(v) for k, v in SPLITS.items()},
            "trajectory_sha256": hashes,
            "shapes": {k: {n: list(a.shape) for n, a in v.items()} for k, v in splits.items()},
            "feature_order": ["x", "y", "z", "sin_roll", "sin_pitch", "sin_yaw",
                              "cos_roll", "cos_pitch", "cos_yaw"],
            "alignment": "obs[t] to obs[t+1] conditioned on jointAppliedTorques[t]; upstream loader alignment",
            "normalization": "float64 mean/population std from train raw IDs only; constant std uses scale1; output float32",
            "split_rule": "upstream test30..40 retained; original training41..49 withheld for development; train0..29",
            "timing": {"nominal_dt_seconds": 0.002, "source": "upstream frequency500 configuration; no timestamps supplied"},
            "limits": ["simulated PyBullet robot, not real hardware",
                       "recorded applied torques, not authenticated issued commands",
                       "no terrain-disjoint or independently collected-run claim",
                       "exact whole-trajectory duplicate check only; no proof against shifted overlaps",
                       "no explicit upstream code/data license found; raw and derivative data stay local",
                       "no test outcomes inspected or model/simulator/RNG calls"],
            "loaded_keys": ["pos", "orn_euler", "jointAppliedTorques"],
            "excluded": ["pos fourth channel (slope)", "vel", "jointPos", "jointVel", "jointReactionForces"],
            "new_model_calls": 0, "new_native_calls": 0,
        }
        write_json(out / "manifest.json", manifest)
        files = {p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                 for p in sorted(out.iterdir()) if p.is_file()}
        if digest(source.read_bytes()) != SOURCE_SHA256:
            raise ValueError("source changed during preparation")
        receipt = {"status": "completed", "scope": "local_offline_data_preparation_only",
                   "source_sha256": manifest["source_sha256"], "input_sha256": SOURCE_SHA256,
                   "files": files, "wall_seconds_before_receipt": time.monotonic() - start,
                   "new_model_calls": 0, "new_native_calls": 0}
        write_json(out / "completed.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                       "wall_seconds": time.monotonic() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original error
            error.add_note(f"Failure receipt could not be preserved: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.out)
