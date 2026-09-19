"""Synthetic arrays/metadata only; no original corpus preparation or draws."""

import copy
import hashlib
import io
import json
from collections import Counter

import numpy as np
import pytest
import torch

from openjev.research import reacher_innovation_pilot_data as pilot

FIXTURE_HASHES = {"npz_sha256": "a" * 64, "json_sha256": "b" * 64}


@pytest.fixture
def fixture():
    # Hand-declared original stratum sizes; no historical seed is executed.
    sizes = ((94, 40, 43), (97, 41, 56), (89, 49, 45), (111, 46, 57))
    metadata = []
    for phase, counts in enumerate(sizes):
        schedule = [True] * 51
        for start in (8 + phase, 28 + phase):
            schedule[start:start + 6] = [False] * 6
        for policy, count in zip(("ik_pd", "random_high", "random_low"), counts, strict=True):
            for _ in range(count):
                index = len(metadata)
                metadata.append({"seed": 64100001 + index, "noise_seed": 64500001 + index,
                    "action_seed": 64600001 + index, "sensor_schedule": list(schedule),
                    "collector_policy": policy, "requested_policy": "mixed", "env_id": "Reacher-v5",
                    "horizon": 50, "dt": 0.02, "noise_std": 0.05, "policy_keys": ["packets", "commands"],
                    "action_hold": 4, "exploration_std": {"ik_pd": 0.12, "random_high": 0.8, "random_low": 0.3}[policy]})
    packets = np.zeros((768, 51, 8), dtype=np.float32)
    angle = np.arange(768 * 51 * 2, dtype=np.float32).reshape(768, 51, 2) / 100000
    packets[..., :2], packets[..., 2:4] = np.cos(angle), np.sin(angle)
    packets[..., 4:6] = np.array([0.1, -0.09], dtype=np.float32)
    for i, row in enumerate(metadata):
        last = 0
        for t, visible in enumerate(row["sensor_schedule"]):
            if visible:
                last = t
            else:
                packets[i, t, :4] = 0
            packets[i, t, 6] = int(visible)
            packets[i, t, 7] = (t - last) * 0.02
    data = {"packets": packets,
            "commands": np.full((768, 50, 2), 0.1, dtype=np.float32),
            "rewards": -np.arange(768 * 50, dtype=np.float64).reshape(768, 50) / 10000}
    return data, metadata


def tensor_equal(left, right):
    assert set(left) == set(right) == {"packets", "commands", "rewards"}
    for name in left:
        assert left[name].device.type == "cpu" and left[name].dtype == torch.float32
        assert torch.equal(left[name], right[name])


def test_fixed_partition_is_whole_episode_complete_deterministic_and_ranked_independently(fixture):
    data, metadata = fixture
    result = pilot._prepare_public(data, metadata, FIXTURE_HASHES)
    wanted_dev = []
    quotas = ((17, 7, 8), (16, 7, 9), (15, 9, 8), (17, 7, 8))
    for phase in range(4):
        for policy, count in zip(("ik_pd", "random_high", "random_low"), quotas[phase], strict=True):
            group = [i for i, row in enumerate(metadata)
                     if row["sensor_schedule"].index(False) == 8 + phase and row["collector_policy"] == policy]
            scores = []
            for index in group:
                raw = json.dumps(["OpenJev/reacher-innovation-pilot-v1/development", "a" * 64, "b" * 64, 64100001 + index],
                                 separators=(",", ":"), ensure_ascii=True).encode("ascii")
                scores.append((hashlib.sha256(raw).hexdigest(), index))
            wanted_dev.extend(index for _, index in sorted(scores)[:count])
    wanted_dev.sort()
    dev = result.manifest["partitions"]["development"]
    train = result.manifest["partitions"]["train"]
    assert dev["original_indices"] == wanted_dev
    assert len(dev["original_indices"]) == 128 and len(train["original_indices"]) == 640
    assert set(dev["original_indices"]).isdisjoint(train["original_indices"])
    assert sorted(dev["original_indices"] + train["original_indices"]) == list(range(768))
    assert Counter(row["phase"] for row in dev["episodes"]) == dict.fromkeys(range(4), 32)
    for split, output in ((train, result.train), (dev, result.dev6)):
        for name, array in data.items():
            expected = torch.from_numpy(array[split["original_indices"]].astype(np.float32))
            assert torch.equal(output[name], expected)
        for row in split["episodes"]:
            i = row["original_index"]
            assert (row["reset_seed"], row["noise_seed"], row["action_seed"], row["schedule_seed"]) == (
                64100001 + i, 64500001 + i, 64600001 + i, 64400001 + i)
    again = pilot._prepare_public(data, metadata, FIXTURE_HASHES)
    assert result.manifest == again.manifest
    tensor_equal(result.train, again.train)
    tensor_equal(result.dev6, again.dev6)
    tensor_equal(result.dev10, again.dev10)


def test_ten_gap_view_only_censors_eight_public_packets_and_recomputes_age(fixture):
    data, metadata = fixture
    result = pilot._prepare_public(data, metadata, FIXTURE_HASHES)
    assert torch.equal(result.dev6["commands"], result.dev10["commands"])
    assert torch.equal(result.dev6["rewards"], result.dev10["rewards"])
    six, ten = result.dev6["packets"], result.dev10["packets"]
    assert (six[..., 6].sum(1) == 39).all() and (ten[..., 6].sum(1) == 31).all()
    for pos, identity in enumerate(result.manifest["partitions"]["development"]["episodes"]):
        phase = identity["phase"]
        expected_mask = torch.ones(51, dtype=torch.bool)
        for start in (8 + phase, 28 + phase):
            expected_mask[start:start + 10] = False
            assert (six[pos, start + 6:start + 10, 6] == 1).all()
        assert torch.equal(ten[pos, :, 6].bool(), expected_mask)
        assert not ten[pos, ~expected_mask, :4].any()
        assert torch.equal(ten[pos, expected_mask, :4], six[pos, expected_mask, :4])
        assert torch.equal(ten[pos, :, 4:6], six[pos, :, 4:6])
        last = 0
        for t in range(51):
            if expected_mask[t]:
                last = t
            assert ten[pos, t, 7].item() == pytest.approx((t - last) * 0.02)
    assert result.manifest["development_views"]["new_labels"] == 0
    assert not result.manifest["development_views"]["privileged_observation_recovery"]


def test_no_input_or_partition_aliases_and_no_rng_calls(fixture, monkeypatch):
    data, metadata = fixture
    snapshot = {name: value.copy() for name, value in data.items()}
    meta_copy = copy.deepcopy(metadata)
    def forbidden(*args, **kwargs):
        raise AssertionError("No random generator or model execution permitted")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch, "randperm", forbidden)
    result = pilot._prepare_public(data, metadata, FIXTURE_HASHES)
    for name in data:
        assert np.array_equal(data[name], snapshot[name])
        assert len({part[name].data_ptr() for part in (result.train, result.dev6, result.dev10)}) == 3
    assert metadata == meta_copy
    result.dev10["rewards"].zero_()
    assert result.dev6["rewards"].abs().sum() > 0
    assert np.array_equal(data["rewards"], snapshot["rewards"])


def tiny_archive():
    output = io.BytesIO()
    np.savez(output, policy__packets=np.zeros((1, 2, 8), dtype=np.float32),
             policy__commands=np.zeros((1, 1, 2), dtype=np.float32),
             audit__rewards=np.zeros((1, 1), dtype=np.float64),
             audit__qvel=np.array([{"privileged": "must never unpickle"}], dtype=object),
             audit__raw_obs=np.array([{"privileged": "must never unpickle"}], dtype=object))
    return output.getvalue()


def test_npz_loader_never_accesses_privileged_members(monkeypatch):
    raw = tiny_archive()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__
    accessed = []
    def guarded(self, key):
        accessed.append(key)
        assert key in {"policy__packets", "policy__commands", "audit__rewards"}
        return original(self, key)
    monkeypatch.setattr(archive_type, "__getitem__", guarded)
    actual = pilot._read_public(raw)
    assert accessed == ["policy__packets", "policy__commands", "audit__rewards"]
    assert set(actual) == {"packets", "commands", "rewards"}


def test_wrong_source_hash_refused_before_any_npz_loading(tmp_path, monkeypatch):
    npz, meta = tmp_path / "fixture.npz", tmp_path / "fixture.json"
    npz.write_bytes(tiny_archive())
    meta.write_text("[]")
    def forbidden(_raw):
        raise AssertionError("Authentication must precede array loading")
    monkeypatch.setattr(pilot, "_read_public", forbidden)
    with pytest.raises(ValueError, match="hash mismatch"):
        pilot.prepare_data(npz, meta)


def test_byte_snapshot_auth_and_handoff_with_explicit_tiny_fixture_constants(tmp_path, monkeypatch):
    # Only this test replaces constants with tiny fixture hashes. Production
    # prepare_data exposes no override and is never invoked on real sources.
    npz, meta = tmp_path / "fixture.npz", tmp_path / "fixture.json"
    raw, raw_meta = tiny_archive(), b'[{"fixture":true}]'
    npz.write_bytes(raw)
    meta.write_bytes(raw_meta)
    hashes = {"npz_sha256": hashlib.sha256(raw).hexdigest(), "json_sha256": hashlib.sha256(raw_meta).hexdigest()}
    monkeypatch.setattr(pilot, "NPZ_SHA256", hashes["npz_sha256"])
    monkeypatch.setattr(pilot, "JSON_SHA256", hashes["json_sha256"])
    def core(data, metadata, actual_hashes):
        assert set(data) == {"packets", "commands", "rewards"}
        assert metadata == [{"fixture": True}] and actual_hashes == hashes
        # Mutating the files after read does not replace authenticated inputs.
        npz.write_bytes(b"changed after snapshot")
        return pilot.PilotData({}, {}, {}, {"source_hashes": actual_hashes})
    monkeypatch.setattr(pilot, "_prepare_public", core)
    result = pilot.prepare_data(npz, meta)
    assert result.manifest["source_hashes"] == hashes
    assert result.manifest["source_authentication"].startswith("both original file byte snapshots")


@pytest.mark.parametrize("defect", ["extra_learning_field", "shape", "dtype", "nonfinite", "missing_poison",
                                  "action_bound", "age", "target", "visible_age", "validity", "seed", "noise_seed",
                                  "action_seed", "row_order", "schedule_length", "gap_length", "phase", "packet_schedule",
                                  "policy", "hold", "noise_scale", "metadata_count"])
def test_malformed_public_data_or_metadata_rejected(fixture, defect):
    data, metadata = fixture
    if defect == "extra_learning_field":
        data["audit_angles"] = np.zeros((768, 51, 4))
    elif defect == "shape":
        data["commands"] = data["commands"][:, :-1]
    elif defect == "dtype":
        data["packets"] = data["packets"].astype(np.float64)
    elif defect == "nonfinite":
        data["rewards"][0, 0] = np.nan
    elif defect == "missing_poison":
        data["packets"][0, 8, 0] = 0.1
    elif defect == "action_bound":
        data["commands"][0, 0, 0] = 1.01
    elif defect == "age":
        data["packets"][0, 8, 7] += 0.01
    elif defect == "target":
        data["packets"][0, 1, 4] += 0.1
    elif defect == "visible_age":
        data["packets"][0, 0, 7] = 1e-7
    elif defect == "validity":
        data["packets"][0, 0, 6] = 0.5
    elif defect in {"seed", "noise_seed", "action_seed"}:
        metadata[0][defect] += 1
    elif defect == "row_order":
        metadata[0], metadata[1] = metadata[1], metadata[0]
    elif defect == "schedule_length":
        metadata[0]["sensor_schedule"].append(True)
    elif defect == "gap_length":
        metadata[0]["sensor_schedule"][14] = False
    elif defect == "phase":
        metadata[0]["sensor_schedule"][7] = False
    elif defect == "packet_schedule":
        metadata[0]["sensor_schedule"] = [True] * 51
        metadata[0]["sensor_schedule"][9:15] = [False] * 6
        metadata[0]["sensor_schedule"][29:35] = [False] * 6
    elif defect == "policy":
        metadata[0]["collector_policy"] = "learned_controller"
    elif defect == "hold":
        metadata[0]["action_hold"] = 3
    elif defect == "noise_scale":
        metadata[0]["noise_std"] = 0.1
    else:
        metadata.pop()
    with pytest.raises(ValueError):
        pilot._prepare_public(data, metadata, FIXTURE_HASHES)


def test_selection_given_fixed_source_identities_does_not_read_rewards_or_angle_values(fixture):
    data, metadata = fixture
    original = pilot._prepare_public(data, metadata, FIXTURE_HASHES)
    data["rewards"] *= 3
    data["packets"][..., :4] *= -1
    changed = pilot._prepare_public(data, metadata, FIXTURE_HASHES)
    assert original.manifest["partitions"] == changed.manifest["partitions"]
