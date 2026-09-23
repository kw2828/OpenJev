"""Fabricated feature vectors only; no scorers, model, filter or native calls."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research.otto_sparse_query import ENTROPY_INDEX, HORIZON, STEP_INDEX, SparseQueryGate


def features(step, entropy=0.5):
    result = np.zeros(31, dtype=np.float32)
    result[15] = np.float32(step / 2188)
    result[19] = np.float32(entropy)
    return result


@pytest.mark.parametrize("kind", ["period2", "random_pair", "entropy"])
def test_all_2188_prefixes_obey_budget_and_fixed_schedules(kind):
    gate = SparseQueryGate(kind, 19876543, .5 if kind == "entropy" else None)
    state = np.zeros(2, dtype=np.float32)
    choices = []
    assert STEP_INDEX == 15 and ENTROPY_INDEX == 19 and HORIZON == 2188
    for step in range(2188):
        before = state.tobytes()
        x = features(step, 0.25 if step % 5 == 0 else 0.75)
        query, new, record = gate.decision(x, state)
        assert state.tobytes() == before
        choices.append(query)
        assert type(query) is bool and new.dtype == np.float32
        assert new.tolist() == [step + 1, sum(choices)]
        assert sum(choices) <= (step + 2) // 2
        assert record["query_limit"] == (step + 2) // 2 and record["queries_after"] == sum(choices)
        if kind == "period2":
            assert query == (step % 2 == 0)
        if kind == "random_pair":
            oracle = hashlib.sha256(b"openjev.otto.sparse-query.random-pair.v1\x00"
                                    + struct.pack(">II", 19876543, step // 2)).digest()
            assert record["pair_sha256"] == oracle.hex() and record["pair_offset"] == (oracle[0] & 1)
            assert query == (step % 2 == (oracle[0] & 1))
        state = new
    if kind in ("period2", "random_pair"):
        assert sum(choices) == 1094
        assert all(sum(choices[start:start+2]) == 1 for start in range(0, 2188, 2))
    with pytest.raises(ValueError, match="horizon"):
        gate(features(2188), state)


def test_entropy_tie_saved_credit_and_inherited_feature_without_clipping():
    gate = SparseQueryGate("entropy", 7, .5)
    state = np.zeros(2, dtype=np.float32)
    choices = []
    # Skipping step0 leaves one credit: step1 and step2 can both query.
    for step, value in enumerate((.25, .5, 1.25, .5, -.25, .5)):
        query, state, row = gate.decision(features(step, value), state)
        choices.append(query)
        assert row["entropy_feature"] == float(np.float32(value))
    assert choices == [False, True, True, False, False, True]
    assert state.tolist() == [6, 3]
    low = np.nextafter(np.float32(.5), np.float32(0))
    assert gate(features(0, low), np.zeros(2, dtype=np.float32))[0] is False
    assert gate(features(0, .5), np.zeros(2, dtype=np.float32))[0] is True


def test_immutable_inputs_returns_and_pure_repeated_calls():
    gate = SparseQueryGate("random_pair", 2**32 - 1)
    x, state = features(0), np.zeros(2, dtype=np.float32)
    x.flags.writeable = state.flags.writeable = False
    before = x.tobytes(), state.tobytes()
    query, after, record = gate.decision(x, state)
    record["query"] = not query
    config = gate.configuration
    config["episode_seed"] = 0
    repeated, other = gate(x, state)
    assert repeated is query and after.tobytes() == other.tobytes()
    assert (x.tobytes(), state.tobytes()) == before and not after.flags.writeable
    with pytest.raises(ValueError):
        after.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        gate.episode_seed = 0


def test_reset_restarts_explicit_state_and_new_episode_seed_changes_schedule():
    def sequence(seed):
        gate = SparseQueryGate("random_pair", seed)
        state = np.zeros(2, dtype=np.float32)
        first = []
        for step in range(64):
            query, state = gate(features(step), state)
            first.append(query)
        state = np.zeros(2, dtype=np.float32)
        replay = []
        for step in range(64):
            query, state = gate(features(step), state)
            replay.append(query)
        assert first == replay
        return first
    assert sequence(0) != sequence(1)


def test_rejects_malformed_feature_state_and_nonchronological_prefix():
    gate = SparseQueryGate("entropy", 9, .5)
    zero = np.zeros(2, dtype=np.float32)
    for bad in (np.zeros(30, dtype=np.float32), features(0).astype(np.float64), features(0).tolist()):
        with pytest.raises(ValueError, match="features"):
            gate(bad, zero)
    for value in (np.nan, np.inf, -np.inf):
        for index in (0, 15, 19, 30):
            bad = features(0)
            bad[index] = value
            with pytest.raises(ValueError, match="features"):
                gate(bad, zero)
    for bad in (np.zeros(3, dtype=np.float32), zero.astype(np.float64), [0, 0],
                np.array([.5, 0], dtype=np.float32), np.array([-1, 0], dtype=np.float32),
                np.array([0, np.nan], dtype=np.float32), np.array([0, 2189], dtype=np.float32)):
        with pytest.raises(ValueError, match="state"):
            gate(features(0), bad)
    with pytest.raises(ValueError, match="chronological"):
        gate(features(1), zero)
    almost = features(1)
    almost[15] = np.nextafter(almost[15], np.float32(1))
    with pytest.raises(ValueError, match="chronological"):
        gate(almost, np.array([1, 0], dtype=np.float32))
    with pytest.raises(ValueError, match="prior prefix"):
        gate(features(2), np.array([2, 2], dtype=np.float32))


def test_fixed_schedules_reject_spliced_query_counters():
    with pytest.raises(ValueError, match="complete prefix"):
        SparseQueryGate("period2", 1)(features(3), np.array([3, 1], dtype=np.float32))
    gate = SparseQueryGate("random_pair", 1)
    first, _, _ = gate.decision(features(0), np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="complete prefix"):
        gate(features(1), np.array([1, int(not first)], dtype=np.float32))


def test_configuration_rejects_ambiguous_or_nonfinite_parameters():
    for kind, seed, threshold in (("unknown", 1, None), ("period2", True, None),
            ("period2", -1, None), ("random_pair", 2**32, None), ("entropy", 1, None),
            ("entropy", 1, True), ("entropy", 1, np.nan), ("entropy", 1, np.inf),
            ("period2", 1, .5), ("entropy", np.int64(1), .5)):
        with pytest.raises(ValueError):
            SparseQueryGate(kind, seed, threshold)
