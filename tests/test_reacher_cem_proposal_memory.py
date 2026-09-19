"""Synthetic explicit-array checks; no RNG, simulator or learned model calls."""

import json

import numpy as np
import pytest

from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_cem_proposal_memory import CEMProposalMemory


def inputs(chunks=4, cases=1):
    def values(k):
        return np.linspace(-1., 1., cases*k*chunks*2, dtype=np.float64).reshape(cases, k, chunks, 2)
    return SearchInputs(values(64), values(192), tuple(values(k) for k in (64, 64, 63)))


def target():
    return np.array([.12, -.04], np.float32)


def plan(h=12):
    return np.linspace(-.8, .8, h*2, dtype=np.float32).reshape(h, 2)


def advance(memory, draws, sequence, step=0, goal=None):
    transformed, trace = memory.prepare(draws, target() if goal is None else goal, step=step)
    memory.commit(sequence, sequence[0].copy())
    return transformed, trace


@pytest.mark.parametrize("mode", ("cold", "repeat_last", "shift_plan"))
def test_startup_returns_exact_input_and_commit_is_only_cache_boundary(mode):
    memory = CEMProposalMemory(mode, 20, 12, 3)
    draws, goal = inputs(), target()
    transformed, trace = memory.prepare(draws, goal, step=0)
    assert transformed is draws and trace["input_object_reused"]
    assert trace["reset_reason"] == "startup" and trace["shifted_sequence"] is None
    assert trace["base_input_identities"] == trace["transformed_input_identities"] == dict(draws.identities())
    assert memory.snapshot()["next_step"] == 0
    assert memory.snapshot()["last_selected_sequence"] is None
    sequence = plan()
    memory.commit(sequence, sequence[0])
    snap = memory.snapshot()
    assert snap["next_step"] == 1 and snap["pending"] is None
    np.testing.assert_array_equal(snap["last_selected_sequence"], sequence)
    assert snap["costs"]["prepare_completed"] == snap["costs"]["commit_completed"] == 1
    json.dumps(trace, allow_nan=False)
    json.dumps(snap, allow_nan=False)


def test_shift_one_action_hold_pad_column_means_and_only_initial_random_slots():
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    draws, sequence = inputs(), plan()
    original_ids = draws.identities()
    advance(memory, draws, sequence)
    transformed, trace = memory.prepare(draws, target(), step=1)
    shifted = np.concatenate((sequence[1:], sequence[-1:]))
    expected_center = np.array([[np.mean(shifted[b:b+3, column], dtype=np.float64)
                                 for column in range(2)] for b in range(0, 12, 3)])
    np.testing.assert_array_equal(trace["shifted_sequence"], shifted)
    np.testing.assert_array_equal(trace["center"], expected_center)
    np.testing.assert_array_equal(transformed.initial[:, :7], draws.initial[:, :7])
    scales = np.r_[np.full(32, .25), np.full(32, .75)]
    np.testing.assert_array_equal(transformed.initial[:, 7:],
                                 draws.initial[:, 7:]+expected_center[None, None]/scales[None, 7:, None, None])
    for actual, expected in zip((transformed.random_extra, *transformed.cem),
                               (draws.random_extra, *draws.cem), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.tobytes() == expected.tobytes()
    assert draws.identities() == original_ids
    with pytest.raises(ValueError):
        transformed.initial.flags.writeable = True
    assert trace["costs"]["projected_blocks"] == 4
    assert trace["costs"]["column_mean_calls"] == 8
    assert trace["costs"]["tail_actions_repeated"] == 1


def test_repeat_last_uses_actual_first_command_not_terminal_action():
    memory = CEMProposalMemory("repeat_last", 20, 12, 3)
    draws, sequence = inputs(), plan()
    advance(memory, draws, sequence)
    _, trace = memory.prepare(draws, target(), step=1)
    np.testing.assert_array_equal(trace["center"], np.tile(sequence[0].astype(np.float64), (4, 1)))
    assert trace["shifted_sequence"] is None
    assert trace["costs"]["column_mean_calls"] == trace["costs"]["tail_actions_repeated"] == 0


@pytest.mark.parametrize("mode", ("repeat_last", "shift_plan"))
def test_observed_target_change_resets_before_planning_and_commit_replaces_cache(mode):
    memory = CEMProposalMemory(mode, 20, 12, 3)
    draws = inputs()
    advance(memory, draws, plan())
    changed = np.array([-.12, .08], np.float32)
    transformed, trace = memory.prepare(draws, changed, step=1)
    assert transformed is draws and trace["reset_reason"] == "observed_target_change"
    assert trace["shifted_sequence"] is None
    np.testing.assert_array_equal(memory.snapshot()["last_target"], target())
    replacement = np.full((12, 2), .3, np.float32)
    memory.commit(replacement, replacement[0])
    _, trace = memory.prepare(draws, changed, step=2)
    np.testing.assert_array_equal(trace["center"], np.full((4, 2), np.float64(np.float32(.3))))
    assert trace["reset_reason"] == "none"
    assert trace["costs"]["target_change_prepares"] == 1


@pytest.mark.parametrize("mode", ("cold", "repeat_last", "shift_plan"))
def test_zero_center_is_bitwise_cem_parity_including_paid_mean_and_clipping(mode):
    draws = inputs()
    memory = CEMProposalMemory(mode, 20, 12, 3)
    advance(memory, draws, np.zeros((12, 2), np.float32))
    changed, _ = memory.prepare(draws, target(), step=1)
    calls = []
    def score(bank):
        calls.append(bank.shape[1])
        return -np.sum((bank-np.float32(.1))**2, axis=(2, 3), dtype=np.float32)
    original = search("cem256", draws, score, step=1, steps=20, planning_horizon=12, action_block=3)
    actual = search("cem256", changed, score, step=1, steps=20, planning_horizon=12, action_block=3)
    assert changed is draws and calls == [64]*8
    for field in ("sequences", "scores", "selected_ids", "selected_sequences", "selected_actions"):
        assert getattr(original, field).tobytes() == getattr(actual, field).tobytes()
    assert actual.candidate_evaluations_per_case == 256
    assert actual.imagined_transitions_per_case == 256*12


def test_cold_never_reuses_nonzero_history_center():
    memory = CEMProposalMemory("cold", 20, 12, 3)
    draws = inputs()
    advance(memory, draws, plan())
    actual, trace = memory.prepare(draws, target(), step=1)
    assert actual is draws and np.count_nonzero(trace["center"]) == 0


def test_terminal_shortening_no_postterminal_pad_and_inactive_blocks_zero():
    memory = CEMProposalMemory("shift_plan", 5, 5, 3)
    draws = inputs(chunks=2)
    sequence = plan(5)
    advance(memory, draws, sequence)
    for step in range(1, 5):
        _, trace = memory.prepare(draws, target(), step=step)
        expected = sequence[1:]
        np.testing.assert_array_equal(trace["shifted_sequence"], expected)
        h = 5-step
        center = np.zeros((2, 2), np.float64)
        for b in range((h+2)//3):
            for column in range(2):
                center[b, column] = np.mean(expected[b*3:min(b*3+3, h), column], dtype=np.float64)
        np.testing.assert_array_equal(trace["center"], center)
        assert trace["costs"]["tail_actions_repeated"] == 0
        memory.commit(expected, expected[0])
        sequence = expected
    assert memory.snapshot()["next_step"] == 5
    with pytest.raises(ValueError, match="terminal"):
        memory.prepare(draws, target(), step=5)
    assert memory.snapshot()["failed"]


def test_inputs_outputs_and_snapshots_cannot_alias_internal_cache():
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    draws, goal, sequence = inputs(), target(), plan()
    _, trace = memory.prepare(draws, goal, step=0)
    goal[:] = 0
    trace["public_target"][:] = [0, 0]
    trace["center"][0][0] = 99
    memory.commit(sequence, sequence[0])
    expected = sequence.copy()
    sequence[:] = 1
    snap = memory.snapshot()
    snap["last_selected_sequence"][0][0] = 99
    snap["pending"] = "poison"
    snap["configuration"]["mode"] = "cold"
    _, trace = memory.prepare(draws, target(), step=1)
    np.testing.assert_array_equal(trace["shifted_sequence"], np.concatenate((expected[1:], expected[-1:])))
    assert trace["reset_reason"] == "none"


def test_counted_payload_copies_hashes_retention_and_json_safe_times():
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    draws = inputs()
    payload = sum(v.nbytes for v in (draws.initial, draws.random_extra, *draws.cem))
    _, trace0 = memory.prepare(draws, target(), step=0)
    assert trace0["costs"]["retained_numpy_payload_bytes"] == 8+4*2*8
    sequence = plan()
    memory.commit(sequence, sequence[0])
    assert memory.snapshot()["costs"]["retained_numpy_payload_bytes"] == 8+96+8+64
    _, trace1 = memory.prepare(draws, target(), step=1)
    costs = trace1["costs"]
    assert costs["retained_numpy_payload_bytes"] == 8+96+8+8+64+96
    assert costs["input_payload_bytes_copied"] == draws.initial.nbytes+payload
    assert costs["input_identity_bytes_hashed"] == 3*payload
    assert costs["transformed_initial_scalars"] == 57*4*2
    assert costs["prepare_attempted"] == costs["prepare_completed"] == 2
    assert costs["commit_attempted"] == costs["commit_completed"] == 1
    for key, value in memory.snapshot()["costs"].items():
        assert np.isfinite(value) and value >= 0, key


@pytest.mark.parametrize("values", [("bad", 20, 12, 3), ("cold", True, 1, 1),
                                   ("cold", 0, 1, 1), ("cold", 20, 21, 3),
                                   ("cold", 20, 12, 13), ("cold", 20, 12., 3)])
def test_malformed_configuration(values):
    with pytest.raises(ValueError):
        CEMProposalMemory(*values)


@pytest.mark.parametrize("kind", ["skip", "bool_step", "batch", "chunks", "target_dtype", "target_nan", "target_shape"])
def test_prepare_validation_is_terminal_with_accurate_failed_prefix(kind):
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    draws, goal, step = inputs(), target(), 0
    if kind == "skip": step = 1
    elif kind == "bool_step": step = False
    elif kind == "batch": draws = inputs(cases=2)
    elif kind == "chunks": draws = inputs(chunks=3)
    elif kind == "target_dtype": goal = goal.astype(np.float64)
    elif kind == "target_nan": goal[0] = np.nan
    else: goal = goal[None]
    with pytest.raises(ValueError): memory.prepare(draws, goal, step=step)
    snap = memory.snapshot()
    assert snap["failed"] and snap["next_step"] == 0 and snap["pending"] is None
    assert snap["costs"]["prepare_attempted"] == snap["costs"]["failed_operations"] == 1
    assert snap["costs"]["prepare_completed"] == 0
    with pytest.raises(RuntimeError, match="terminal"): memory.prepare(inputs(), target(), step=0)
    assert memory.snapshot()["costs"]["prepare_attempted"] == 1


@pytest.mark.parametrize("kind", ["no_prepare", "wrong_command", "wrong_horizon", "dtype", "nan", "bounds"])
def test_commit_validation_never_advances_cache(kind):
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    sequence, command = plan(), plan()[0].copy()
    if kind != "no_prepare": memory.prepare(inputs(), target(), step=0)
    if kind == "wrong_command": command[:] = 0
    elif kind == "wrong_horizon": sequence = sequence[:-1]
    elif kind == "dtype": sequence = sequence.astype(np.float64)
    elif kind == "nan": sequence[1, 0] = np.nan
    elif kind == "bounds": sequence[1, 0] = 1.01
    with pytest.raises((ValueError, RuntimeError)): memory.commit(sequence, command)
    snap = memory.snapshot()
    assert snap["failed"] and snap["next_step"] == 0 and snap["last_selected_sequence"] is None
    assert snap["costs"]["commit_attempted"] == 1 and snap["costs"]["commit_completed"] == 0


def test_duplicate_prepare_preserves_pending_and_failed_no_commit():
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    memory.prepare(inputs(), target(), step=0)
    before = memory.snapshot()["pending"]
    with pytest.raises(RuntimeError, match="pending"): memory.prepare(inputs(), target(), step=0)
    assert memory.snapshot()["pending"] == before
    with pytest.raises(RuntimeError, match="terminal"): memory.commit(plan(), plan()[0])


def test_failure_after_projection_preserves_paid_prefix_and_prior_cache(monkeypatch):
    memory = CEMProposalMemory("shift_plan", 20, 12, 3)
    draws = inputs()
    advance(memory, draws, plan())
    def fail(self):
        raise ArithmeticError("identity hashing failed")
    monkeypatch.setattr(SearchInputs, "identities", fail)
    with pytest.raises(ArithmeticError, match="identity hashing"):
        memory.prepare(draws, target(), step=1)
    snap = memory.snapshot()
    assert snap["next_step"] == 1 and snap["pending"] is None
    assert snap["costs"]["prepare_attempted"] == 2 and snap["costs"]["prepare_completed"] == 1
    assert snap["costs"]["column_mean_calls"] == 8 and snap["costs"]["tail_actions_repeated"] == 1
    assert snap["failure"]["error_type"] == "ArithmeticError"
