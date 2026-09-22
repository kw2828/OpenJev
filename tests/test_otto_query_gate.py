"""Fabricated score functions/public packets only, no model or native calls."""
from __future__ import annotations

import numpy as np
import pytest

from openjev.research.otto_query_gate import FEATURE_DIM, FEATURE_NAMES, QueryGateActor
from openjev.research.otto_released_policy import PublicBeliefView
from openjev.research.otto_restricted_policy import select_inbounds_action


def kernel():
    x, y = np.indices((107, 107), dtype=np.float64)
    out = np.stack(((x + 1) / 432, (y + 1) / 432, np.full_like(x, .25), .75 - (x + y + 2) / 432))
    out[:, 53, 53] = 0
    return out


def packet(step=0, position=(26, 26), hit=1, done=False):
    return {"position": list(position), "step": step, "hit": hit, "done": done,
            "valid_actions": [] if done else [a for a in range(4)
                if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


def after(actor, action, *, hit=0, done=False):
    previous = actor.public
    position = list(previous["position"])
    axis = action // 2
    position[axis] = min(52, max(0, position[axis] + 2 * (action % 2) - 1))
    return packet(previous["step"] + 1, position, -2 if done else hit, done)


def analytic(view):
    out = np.asarray([5e-11, 0, 2, 3], dtype=np.float64)
    for a in range(4):
        if not view._move(a, view.agent)[1]:
            out[a] = np.inf
    return out


def neural(_view):
    return np.asarray([-2, 5e-11, -1, 0], dtype=np.float32)


def constant_gate(query):
    return lambda _features, state: (query, state)


def actor(query=True, **kwargs):
    return QueryGateActor(packet(), kernel(), 3, analytic, neural, constant_gate(query), **kwargs)


def reach(state, position):
    for axis in range(2):
        while state.public["position"][axis] != position[axis]:
            action = 2 * axis + int(position[axis] > state.public["position"][axis])
            state.update(action, after(state, action))


@pytest.mark.parametrize("query", [True, False])
def test_endpoint_policy_exact_raw_bytes_and_boundary_near_tie(query):
    state = actor(query)
    reach(state, (0, 0))
    expected = neural(state.view) if query else analytic(state.view)
    # Eligible1 is near-minimum; lower blocked scores must never be selected.
    action, raw = state.choose()
    assert action == 1
    if query:
        assert action == select_inbounds_action(expected, (1, 3))
    assert raw.dtype == expected.dtype and raw.tobytes() == expected.tobytes()
    raw[:] = 42
    assert state.progress["pending_action"] == 1
    assert state.progress["calls"]["neural_score"] == {"attempted": int(query), "returned": int(query)}
    assert state.progress["calls"]["analytic_score"] == {"attempted": 1, "returned": 1}


def test_gate_blindness_call_order_immutable_state_and_skipped_backend():
    sequence, seen = [], []
    calls = iter((False, True))

    def a(view):
        sequence.append("analytic")
        return analytic(view)

    def gate(features, memory):
        sequence.append("gate")
        assert type(features) is np.ndarray and features.dtype == np.float32 and features.shape == (31,)
        assert memory.dtype == np.float32 and memory.shape == (2,)
        assert not features.flags.writeable and not memory.flags.writeable
        with pytest.raises(ValueError):
            features.flags.writeable = True
        with pytest.raises(ValueError):
            memory[:] = 7
        seen.append((features.copy(), memory.copy()))
        return next(calls), memory + np.float32(1)

    def n(view):
        sequence.append("neural")
        return neural(view)

    state = QueryGateActor(packet(), kernel(), 3, a, n, gate, state_size=2)
    action, _ = state.choose()
    assert sequence == ["analytic", "gate"]
    state.update(action, after(state, action, hit=2))
    state.choose()
    assert sequence == ["analytic", "gate", "analytic", "gate", "neural"]
    np.testing.assert_array_equal(seen[0][1], [0, 0])
    np.testing.assert_array_equal(seen[1][1], [1, 1])
    np.testing.assert_array_equal(state.state, [2, 2])


def test_one_filter_updates_both_readers_every_actual_packet_and_terminal():
    snapshots = []

    def capture(view):
        snapshots.append((view.agent, view.p_source.tobytes()))
        return neural(view)

    state = QueryGateActor(packet(), kernel(), 3, analytic, capture, constant_gate(True))
    expected = PublicBeliefView(kernel())
    expected._reset(1)
    for hit in (0, 3, 2):
        action, _ = state.choose()
        assert snapshots[-1] == (expected.agent, expected.p_source.tobytes())
        observation = after(state, action, hit=hit)
        expected._observe({**observation, "position": tuple(observation["position"])})
        state.update(action, observation)
        assert state.view.agent == expected.agent
        assert state.view.p_source.tobytes() == state.belief.tobytes() == expected.p_source.tobytes()
    action, _ = state.choose()
    state.update(action, after(state, action, done=True))
    assert state.public["done"] and np.count_nonzero(state.belief) == 1
    assert state.progress["calls"]["public_update"] == {"attempted": 4, "returned": 4}
    with pytest.raises(ValueError, match="found"):
        state.choose()


def test_feature_scaling_compute_age_previous_action_and_explicit_reset():
    features, events = [], []
    queries = iter((False, True, False))

    def gate(x, memory):
        features.append(x.copy())
        return next(queries), memory + np.float32(1)

    state = QueryGateActor(packet(), kernel(), 4, analytic, neural, gate, state_size=1, emit=events.append)
    for hit in (2, 0):
        action, _ = state.choose()
        state.update(action, after(state, action, hit=hit))
    state.choose()
    assert FEATURE_DIM == 31 and len(set(FEATURE_NAMES)) == 31
    f0, f1, f2 = [dict(zip(FEATURE_NAMES, row, strict=True)) for row in features]
    assert f0["x/52"] == f0["y/52"] == .5 and f0["sensing_length/5"] == np.float32(.8)
    assert f0["hit_1"] == 1 and all(f0[f"last_action_{a}"] == 0 for a in range(4))
    assert f0["query_age/2188"] == f0["has_queried"] == 0
    assert f1["step/2188"] == f1["query_age/2188"] == np.float32(1 / 2188)
    assert f1["has_queried"] == 0 and f1["hit_2"] == 1 and f1["last_action_0"] == 1
    assert f2["query_age/2188"] == np.float32(1 / 2188) and f2["has_queried"] == 1
    assert all(f0[f"analytic_delta_{a}/64"] == 0 for a in range(4))
    counts = state.progress["calls"]
    state.reset(packet(hit=3))
    assert state.state.tolist() == [0] and state.last_choice is None and state.progress["pending_action"] is None
    assert state.progress["calls"]["gate"] == counts["gate"]
    resets = [event for event in events if event["channel"] == "public_reset"]
    assert [(event["episode"], event["step"], event["event"]) for event in resets] == [
        (0, 0, "attempt"), (0, 0, "return"), (1, 0, "attempt"), (1, 0, "return")]


def test_readonly_view_exposes_geometry_but_no_mutators_or_hidden_fields():
    state = actor()
    before = state.belief
    view = state.view
    view.p_source[:] = 0
    view.agent[:] = [0, 0]
    assert state.belief.tobytes() == before.tobytes() and state.public["position"] == (26, 26)
    for field in ("source", "seed", "draws", "native", "_observe", "_reset"):
        assert not hasattr(view, field)
    with pytest.raises(ValueError):
        view.p_Poisson.flags.writeable = True
    centered = view._centeragent(before, view.agent)
    assert centered.shape == (105, 105)
    np.testing.assert_array_equal(view._extract_N_from_2N(centered, view.agent), before)


@pytest.mark.parametrize("likelihood", [0., 1e-12])
def test_zero_and_subfloor_public_mass_are_not_repaired_or_renormalized(likelihood):
    seen = []

    def gate(x, memory):
        seen.append(dict(zip(FEATURE_NAMES, x, strict=True)))
        return False, memory

    k = np.full((4, 107, 107), likelihood, dtype=np.float64)
    k[:, 53, 53] = 0
    state = QueryGateActor(packet(), k, 3, analytic, neural, gate)
    state.choose()
    assert state.belief.sum() == pytest.approx(likelihood, abs=1e-25)
    assert seen[0]["mass"] == np.float32(state.belief.sum())
    if likelihood == 0:
        assert seen[0]["entropy/log2(2809)"] == seen[0]["distance/104"] == 0
    else:
        assert 0 < seen[0]["entropy/log2(2809)"] < 1e-10
        assert 0 < seen[0]["distance/104"] < 1e-12


@pytest.mark.parametrize("failure", ["neural", "return_journal", "gate_state"])
def test_failed_work_stays_visible_no_retry_and_reset_retains_evidence(failure):
    calls, events = [], []

    def n(view):
        calls.append("neural")
        if failure == "neural":
            raise RuntimeError("neural failed")
        return neural(view)

    def emit(event):
        events.append(dict(event))
        if failure == "return_journal" and event["event"] == "return" and event["channel"] == "neural_score":
            raise OSError("journal failed")

    def gate(_x, memory):
        return (True, np.asarray([np.nan], dtype=np.float32)) if failure == "gate_state" else (True, memory)

    state = QueryGateActor(packet(), kernel(), 3, analytic, n, gate, state_size=1, emit=emit)
    with pytest.raises((RuntimeError, OSError, ValueError)):
        state.choose()
    progress = state.progress
    assert progress["failed"] and len(progress["errors"]) == 1 and progress["pending_action"] is None
    if failure == "gate_state":
        assert calls == [] and progress["calls"]["gate"] == {"attempted": 1, "returned": 1}
    else:
        assert calls == ["neural"] and progress["calls"]["neural_score"] == {"attempted": 1, "returned": 0}
        assert progress["pending_operations"][-1]["channel"] == "neural_score"
    with pytest.raises(RuntimeError, match="reset"):
        state.choose()
    with pytest.raises(RuntimeError, match="reset"):
        state.update(0, after(state, 0))
    assert state.progress == progress
    state.reset(packet())
    assert not state.progress["failed"] and state.progress["errors"] == progress["errors"]
    assert state.progress["pending_operations"] == progress["pending_operations"]
    assert state.state.tolist() == [0]


def test_update_return_failure_preserves_pending_action_and_prevents_resume():
    def emit(event):
        if event["channel"] == "public_update" and event["event"] == "return":
            raise OSError("public update publication failed")

    state = actor(emit=emit)
    action, _ = state.choose()
    before = state.public
    with pytest.raises(OSError):
        state.update(action, after(state, action, hit=2))
    assert state.public == before and state.progress["pending_action"] == action
    assert state.progress["calls"]["public_update"] == {"attempted": 1, "returned": 0}
    with pytest.raises(RuntimeError, match="reset"):
        state.update(action, after(state, action))
    state.reset(packet())
    assert state.public["step"] == 0 and state.view.agent == [26, 26]


def test_budget_check_precedes_invocation_and_wrong_observation_cannot_resume():
    remaining = [100]

    def check():
        if remaining[0] == 0:
            raise TimeoutError("budget")

    state = actor(check=check)
    remaining[0] = 0
    with pytest.raises(TimeoutError):
        state.choose()
    assert state.progress["calls"]["analytic_score"] == {"attempted": 0, "returned": 0}
    remaining[0] = 100
    state.reset(packet())
    action, _ = state.choose()
    with pytest.raises(ValueError, match="pending"):
        state.update((action + 1) % 4, after(state, (action + 1) % 4))
    assert state.progress["pending_action"] == action
    assert state.progress["calls"]["public_update"] == {"attempted": 0, "returned": 0}
