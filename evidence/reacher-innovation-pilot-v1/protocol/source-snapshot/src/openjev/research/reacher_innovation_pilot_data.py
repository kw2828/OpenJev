"""Read-only public training-corpus preparation for a future development pilot.

The public entry point authenticates the exact original two files. Only public
packets, issued commands and total native reward labels are decompressed from
the NPZ; simulator states, clean hidden observations, reward components and
realized noise never enter the returned tensors. No RNG, model or native calls
are made and nothing is written. The caller owns saving and freezing results.

Development here is a subset of a previously used TRAINING corpus. The ten-gap
view only censors additional known public observations on those same recorded
trajectories. It is neither fresh physical data nor closed-loop evaluation.
Previously fitted models that used all 768 episodes are not clean comparators.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NPZ_SHA256 = "bcf12a01b521ea677871ab7686b6c05913c6123bbab77e718e53090b16da6d08"
JSON_SHA256 = "89a6a387f21affa72feca65510d7dd04cca0207dbee3d9f309e9023b4933a6a8"
RANK_DOMAIN = "OpenJev/reacher-innovation-pilot-v1/development"
PUBLIC_KEYS = {"packets": "policy__packets", "commands": "policy__commands", "rewards": "audit__rewards"}
EPISODES, STEPS, DT = 768, 50, 0.02
SEED_BASES = {"seed": 64100001, "noise_seed": 64500001, "action_seed": 64600001}
SCHEDULE_SEED_BASE = 64400001
POLICIES = ("ik_pd", "random_high", "random_low")
DEV_QUOTAS = ((17, 7, 8), (16, 7, 9), (15, 9, 8), (17, 7, 8))


@dataclass(frozen=True)
class PilotData:
    """Caller-owned tensors and manifest; frozen fields do not freeze contents."""

    train: dict[str, torch.Tensor]
    dev6: dict[str, torch.Tensor]
    dev10: dict[str, torch.Tensor]
    manifest: dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _phase(schedule):
    _require(isinstance(schedule, list) and len(schedule) == STEPS + 1
             and all(type(value) is bool for value in schedule), "Boolean 51-packet sensor schedule required")
    missing = [i for i, visible in enumerate(schedule) if not visible]
    _require(len(missing) == 12 and 8 <= missing[0] <= 11, "Two original six-packet gaps required")
    phase = missing[0] - 8
    expected = [True] * (STEPS + 1)
    for start in (8 + phase, 28 + phase):
        expected[start:start + 6] = [False] * 6
    _require(schedule == expected, "Original six-gap positions or lengths differ")
    return phase


def _ages(valid):
    indices = np.arange(STEPS + 1)[None]
    last = np.maximum.accumulate(np.where(valid, indices, -1), axis=1)
    return ((indices - last) * DT).astype(np.float32)


def _validate_public(data, metadata):
    _require(isinstance(data, dict) and set(data) == set(PUBLIC_KEYS), "Exact public learning allowlist required")
    shapes = {"packets": (EPISODES, STEPS + 1, 8), "commands": (EPISODES, STEPS, 2), "rewards": (EPISODES, STEPS)}
    for name, shape in shapes.items():
        value = data[name]
        dtype = np.float64 if name == "rewards" else np.float32
        _require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
                 and np.isfinite(value).all(), f"Finite original array shape/dtype required: {name}")
    packets = data["packets"]
    _require(((packets[..., 6] == 0) | (packets[..., 6] == 1)).all(), "Binary public validity required")
    valid = packets[..., 6] == 1
    _require(valid[:, 0].all() and (packets[..., :4][~valid] == 0).all(), "Visible startup and zero missing angles required")
    _require(np.array_equal(packets[..., 4:6], np.broadcast_to(packets[:, :1, 4:6], packets[..., 4:6].shape)),
             "Static public target required")
    _require((packets[..., 7][valid] == 0).all()
             and np.isclose(packets[..., 7], _ages(valid), rtol=1e-5, atol=1e-6).all(),
             "Public age disagrees with retained measurements")
    _require((np.abs(data["commands"]) <= 1).all(), "Issued commands must already be clipped to [-1,1]")
    _require(isinstance(metadata, list) and len(metadata) == EPISODES, "Exactly 768 episode metadata rows required")
    phases = []
    for index, row in enumerate(metadata):
        _require(isinstance(row, dict), "Episode metadata must be an object")
        for key, base in SEED_BASES.items():
            _require(type(row.get(key)) is int and row[key] == base + index, f"Original row/seed relationship: {key}")
        phase = _phase(row.get("sensor_schedule"))
        _require(np.array_equal(valid[index], row["sensor_schedule"]), "Metadata/public observation schedule mismatch")
        policy = row.get("collector_policy")
        _require(policy in POLICIES and row.get("requested_policy") == "mixed", "Original collection policy required")
        _require(row.get("horizon") == STEPS and row.get("dt") == DT and row.get("noise_std") == 0.05
                 and row.get("env_id") == "Reacher-v5" and row.get("policy_keys") == ["packets", "commands"],
                 "Original public cohort configuration required")
        _require(type(row.get("action_hold")) is int and row["action_hold"] == 4
                 and row.get("exploration_std") == {"ik_pd": 0.12, "random_high": 0.8, "random_low": 0.3}[policy],
                 "Original exploration metadata required")
        phases.append(phase)
    return phases


def _rank(npz_sha, json_sha, reset_seed):
    payload = json.dumps([RANK_DOMAIN, npz_sha, json_sha, reset_seed],
                         ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tensors(data, indices):
    return {name: torch.from_numpy(np.array(value[indices], dtype=np.float32, copy=True)) for name, value in data.items()}


def _prepare_public(data, metadata, source_hashes):
    """Internal deterministic core; supplied bindings are not authentication.

    ``prepare_data`` is the production entry point and checks the real file
    hashes first. Synthetic tests call this core with their own fixture hashes.
    """
    _require(set(source_hashes) == {"npz_sha256", "json_sha256"}
             and all(isinstance(value, str) and len(value) == 64
                     and all(c in "0123456789abcdef" for c in value) for value in source_hashes.values()),
             "Explicit source SHA-256 bindings required")
    phases = _validate_public(data, metadata)
    selected, strata = [], []
    for phase in range(4):
        for policy, quota in zip(POLICIES, DEV_QUOTAS[phase], strict=True):
            indices = [i for i, row in enumerate(metadata) if phases[i] == phase and row["collector_policy"] == policy]
            _require(len(indices) >= quota, "Insufficient episodes for fixed development quota")
            ranked = sorted(indices, key=lambda i: (_rank(source_hashes["npz_sha256"], source_hashes["json_sha256"], metadata[i]["seed"]), i))
            selected.extend(ranked[:quota])
            strata.append({"phase": phase, "collector_policy": policy, "available": len(indices), "development": quota})
    dev_indices = sorted(selected)
    dev_set = set(dev_indices)
    train_indices = [i for i in range(EPISODES) if i not in dev_set]
    _require(len(dev_indices) == len(dev_set) == 128 and len(train_indices) == 640, "Complete disjoint 640/128 partition")
    train, dev6 = _tensors(data, train_indices), _tensors(data, dev_indices)
    dev10 = {key: value.clone() for key, value in dev6.items()}
    for position, index in enumerate(dev_indices):
        for start in (8 + phases[index], 28 + phases[index]):
            _require(bool((dev6["packets"][position, start + 6:start + 10, 6] == 1).all()),
                     "Censoring may remove only already-public visible packets")
            dev10["packets"][position, start + 6:start + 10, :4] = 0
            dev10["packets"][position, start + 6:start + 10, 6] = 0
    valid10 = dev10["packets"][..., 6].numpy() == 1
    dev10["packets"][..., 7] = torch.from_numpy(_ages(valid10))

    def identities(indices):
        return [{"original_index": i, "reset_seed": metadata[i]["seed"],
                 "noise_seed": metadata[i]["noise_seed"], "action_seed": metadata[i]["action_seed"],
                 "schedule_seed": SCHEDULE_SEED_BASE + i, "phase": phases[i],
                 "collector_policy": metadata[i]["collector_policy"],
                 "selection_sha256": _rank(source_hashes["npz_sha256"], source_hashes["json_sha256"], metadata[i]["seed"])}
                for i in indices]

    manifest = {
        "version": "reacher-innovation-pilot-data-v1",
        "scope": "development split of previously used training corpus; not untouched evaluation",
        "source_hashes": dict(source_hashes), "source_authentication": "caller-supplied to internal core",
        "selection_rule": {"domain": RANK_DOMAIN,
            "canonical_json": '[domain,npz_sha256,json_sha256,reset_seed], ensure_ascii=True, separators=(",", ":"), ASCII bytes',
            "rank": "ascending SHA-256 hex within phase and collector policy, then original index",
            "output_order": "ascending original index for both partitions", "strata": strata},
        "counts": {"original": EPISODES, "train": 640, "development": 128,
                   "dev6": 128, "dev10": 128, "steps": STEPS, "packets_per_episode": STEPS + 1,
                   "development_per_phase": 32},
        "partitions": {"train": {"original_indices": train_indices, "episodes": identities(train_indices)},
                       "development": {"original_indices": dev_indices, "episodes": identities(dev_indices)}},
        "public_arrays": {name: {"npz_key": PUBLIC_KEYS[name], "source_dtype": str(value.dtype),
                                 "source_shape": list(value.shape), "returned_dtype": "torch.float32"} for name, value in data.items()},
        "development_views": {"dev6": "original public observations and original total reward/issued commands",
            "dev10": "hide four already-visible public packets immediately after each original six-gap; zero angles/validity and recompute age",
            "extra_censored_packets_per_episode": 8, "new_labels": 0,
            "privileged_observation_recovery": False, "commands_and_total_rewards_unchanged": True},
        "rng_calls": 0, "model_calls": 0, "native_calls": 0,
        "source_sensing": "ordinary six-gap source only; phases 0..3 are schedule strata, not independent evaluation panels",
        "source_policy_counts": dict(sorted(Counter(row["collector_policy"] for row in metadata).items())),
    }
    return PilotData(train, dev6, dev10, manifest)


def _read_public(npz_bytes):
    # Do not enumerate/materialize audit arrays; only these three archive
    # members are decompressed. Hashing opaque archive bytes is separate.
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        return {name: archive[key] for name, key in PUBLIC_KEYS.items()}


def prepare_data(npz_path, json_path) -> PilotData:
    """Authenticate fixed original bytes, prepare fresh tensors, write nothing.

    Both inputs are snapshotted once, so the arrays/metadata used are the bytes
    actually hashed. There is no production override for the accepted hashes.
    Returned partitions and the two development views do not alias each other.
    """
    npz_bytes, json_bytes = Path(npz_path).read_bytes(), Path(json_path).read_bytes()
    hashes = {"npz_sha256": hashlib.sha256(npz_bytes).hexdigest(),
              "json_sha256": hashlib.sha256(json_bytes).hexdigest()}
    _require(hashes == {"npz_sha256": NPZ_SHA256, "json_sha256": JSON_SHA256}, "Original training corpus hash mismatch")
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    metadata = json.loads(json_bytes, parse_constant=nonfinite)
    result = _prepare_public(_read_public(npz_bytes), metadata, hashes)
    result.manifest["source_authentication"] = "both original file byte snapshots matched fixed SHA-256 constants"
    return result
