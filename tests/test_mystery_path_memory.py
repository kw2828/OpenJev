"""Handwritten public transitions only: no environment, seed or hidden path."""

import inspect
import json
from collections import deque

import pytest

from openjev.research.mystery_path_memory import (
    MODES,
    PRIORITIES,
    MemoryController,
    _first_action,
)
from openjev.research.mystery_path_observation import Observation


def obs(x=3, y=3, heading=0, off_path=False):
    return Observation(x=x, y=y, heading=heading, off_path=off_path)


def turn(controller, before, action):
    after = obs(before.x, before.y, (before.heading + (1 if action == 1 else -1)) % 4)
    controller.observe(before, action, after, 0.0)
    return after


@pytest.mark.parametrize("mode", MODES)
def test_startup_forward_and_observed_failure_force_wait_return_ignores_action(mode):
    controller = MemoryController(mode)
    start = obs()
    controller.reset(start)
    assert controller.act(start) == 3
    failed = obs(3, 2, 0, True)
    controller.observe(start, 3, failed, 0.0)
    assert controller.act(failed) == 0
    # Even a turn command is ignored on the return transition.
    controller.observe(failed, 1, start, 0.0)
    assert controller.act(start) == 1  # west frontier wins the west/east tie
    if mode in ("full", "erase_on_failure"):
        assert controller._safe == {(3, 3)}
        assert controller._failed == {(3, 2)}


@pytest.mark.parametrize("priority", PRIORITIES)
def test_compass_priority_breaks_equal_action_cost_frontiers(priority):
    start = (3, 3, 0, False)
    action = _first_action({(3, 3)}, {(3, 2)}, start, priority)
    desired = min((1, 3), key=priority.index)
    assert action == (1 if desired == 1 else 2)


def test_action_cost_can_prefer_more_grid_steps_without_a_turn():
    safe = {(1, 1), (2, 1)}
    failed = {(1, 0), (1, 2), (2, 0), (2, 2)}
    # West is one grid edge but three actions; east is two forward actions.
    assert _first_action(safe, failed, (1, 1, 3, False), (0, 1, 2, 3)) == 3


def test_180_degree_turn_uses_left_first_and_does_not_oscillate():
    controller = MemoryController("full")
    current = obs()
    controller.reset(current)
    controller._failed.update({(3, 2), (2, 3), (4, 3)})
    assert controller.act(current) == 1
    current = turn(controller, current, 1)
    assert controller.act(current) == 1
    current = turn(controller, current, 1)
    assert controller.act(current) == 3


def test_bfs_uses_adjacency_of_safe_cells_not_visit_order_or_progress():
    safe = {(2, 2), (2, 1), (3, 1), (3, 2)}
    failed = {(2, 3), (1, 2), (1, 1), (2, 0), (3, 0), (4, 1), (3, 3)}
    # Safe cell east is directly adjacent, even if first visited by the upper detour.
    assert _first_action(safe, failed, (2, 2, 3, False), (0, 1, 2, 3)) == 3


def test_no_frontier_returns_wait_and_boundary_is_not_unknown():
    safe = {(x, y) for x in range(7) for y in range(7)}
    assert _first_action(safe, set(), (0, 0, 0, False), (0, 1, 2, 3)) == 0
    assert _first_action({(0, 0)}, set(), (0, 0, 0, False), (0, 1, 2, 3)) == 2


def test_erase_clears_old_evidence_then_keeps_both_new_transition_endpoints():
    controller = MemoryController("erase_on_failure")
    controller.reset(obs(3, 4))
    before = obs()
    controller.observe(obs(3, 4), 3, before, 0.0)
    controller._failed.add((0, 0))
    failed = obs(3, 2, 0, True)
    controller.observe(before, 3, failed, 0.0)
    assert controller._safe == {(3, 3)}
    assert controller._failed == {(3, 2)}
    assert (3, 2) not in controller._safe


@pytest.mark.parametrize("mode,limit", (("last16", 16), ("last32", 32)))
def test_window_eviction_forgets_failure_and_retains_exact_raw_transitions(mode, limit):
    controller = MemoryController(mode)
    start = obs()
    failed = obs(3, 2, 0, True)
    controller.reset(start)
    controller.observe(start, 3, failed, 0.0)
    controller.observe(failed, 0, start, 0.0)
    for index in range(limit - 1):
        controller.observe(start, 0, start, float(index))
    # The failure entry is evicted, but the oldest retained return still exposes it.
    assert len(controller._transitions) == limit
    assert controller.act(start) == 1
    controller.observe(start, 0, start, 99.0)
    assert controller.act(start) == 3
    assert controller._safe == controller._failed == set()
    assert list(controller._transitions)[-1] == ((3, 3, 0, False), 0, (3, 3, 0, False), 99.0)
    assert controller.memory_bytes()["retained_transitions"] == limit


@pytest.mark.parametrize("mode", ("last16", "last32"))
def test_oldest_before_observation_survives_as_k_plus_one_endpoint(mode):
    controller = MemoryController(mode)
    start = obs(3, 4)
    end = obs()
    controller.reset(start)
    controller.observe(start, 3, end, 0.0)
    limit = controller._transitions.maxlen
    for _ in range(limit - 1):
        controller.observe(end, 0, end, 0.0)
    assert controller._transitions[0][0] == (3, 4, 0, False)
    assert controller._transitions[-1][2] == (3, 3, 0, False)
    # Raw retention has no extra initial/current observation cache.
    assert MemoryController.__slots__ == ("_failed", "_mode", "_priority", "_ready", "_safe", "_transitions")


@pytest.mark.parametrize("mode", MODES)
def test_forced_return_labels_endpoints_without_inventing_intermediate_cells(mode):
    controller = MemoryController(mode)
    failed = obs(6, 6, 2, True)
    returned = obs(0, 0, 2)
    controller.reset(failed)
    controller.observe(failed, 3, returned, 0.0)
    assert controller.act(returned) == 3
    if mode in ("full", "erase_on_failure"):
        assert controller._safe == {(0, 0)} and controller._failed == {(6, 6)}
    else:
        assert list(controller._transitions) == [((6, 6, 2, True), 3, (0, 0, 2, False), 0.0)]


@pytest.mark.parametrize("priority", PRIORITIES)
def test_full_and_windows_have_identical_actions_before_eviction(priority):
    controllers = [MemoryController(mode, priority) for mode in MODES]
    current = obs()
    for controller in controllers:
        controller.reset(current)
    scripted = [(3, obs(3, 2, 0, True)), (0, obs()), (1, obs(3, 3, 1)),
                (3, obs(2, 3, 1)), (2, obs(2, 3, 0)), (3, obs(2, 2, 0))]
    for action, after in scripted:
        assert len({controller.act(current) for controller in controllers}) == 1
        for controller in controllers:
            controller.observe(current, action, after, 0.0)
        current = after


@pytest.mark.parametrize("mode", MODES)
def test_reset_episode_isolation_and_decisions_do_not_change_retained_state(mode):
    controller = MemoryController(mode)
    start = obs()
    controller.reset(start)
    controller.observe(start, 3, obs(3, 2, 0, True), 1.0)
    controller.reset(start)
    fresh = MemoryController(mode)
    fresh.reset(start)
    before = controller.memory_bytes()
    assert controller.act(start) == fresh.act(start) == 3
    assert controller.memory_bytes() == before
    assert controller._safe == fresh._safe and controller._failed == fresh._failed
    assert controller._transitions == fresh._transitions


def test_memory_serialization_measures_actual_retention_separately_from_python_size():
    controller = MemoryController("last16")
    start = obs()
    controller.reset(start)
    controller.observe(start, 0, start, 0.0)
    expected = {"mode": "last16", "priority": (0, 1, 2, 3), "ready": True,
                "transitions": [((3, 3, 0, False), 0, (3, 3, 0, False), 0.0)]}
    metrics = controller.memory_bytes()
    assert metrics["serialized_state_bytes"] == len(json.dumps(
        expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    assert metrics["retained_python_bytes"] > metrics["serialized_state_bytes"] > 0
    assert metrics["retained_transitions"] == 1
    assert "temporary search" in metrics["scope"]


@pytest.mark.parametrize("mode", ("last16", "last32"))
def test_rewards_are_retained_but_do_not_steer_actions(mode):
    left, right = MemoryController(mode), MemoryController(mode)
    start = obs()
    for controller, reward in ((left, -1000.0), (right, 1000.0)):
        controller.reset(start)
        controller.observe(start, 0, start, reward)
    assert left.act(start) == right.act(start)
    assert left._transitions[0][-1] != right._transitions[0][-1]


@pytest.mark.parametrize("priority", ((0, 2, 1, 3), [0, 1, 2, 3], (False, 1, 2, 3), (0, 1, 2)))
def test_invalid_priorities_rejected(priority):
    with pytest.raises(ValueError, match="rotation"):
        MemoryController("full", priority)


@pytest.mark.parametrize("action,after", ((0, obs(3, 2)), (1, obs()), (3, obs(3, 1)),
                                         (0, obs(3, 3, 0, True)), (True, obs())))
def test_malformed_transition_rejected_without_mutating_memory(action, after):
    controller = MemoryController("full")
    controller.reset(obs())
    before = controller.memory_bytes()
    with pytest.raises(ValueError):
        controller.observe(obs(), action, after, 0.0)
    assert controller.memory_bytes() == before


@pytest.mark.parametrize("after", (obs(3, 3, 1), obs(3, 3, 0, True)))
def test_forced_return_heading_or_flag_corruption_rejected(after):
    controller = MemoryController("last16")
    controller.reset(obs())
    with pytest.raises(ValueError, match="Forced return"):
        controller.observe(obs(3, 2, 0, True), 2, after, 0.0)
    assert not controller._transitions


@pytest.mark.parametrize("reward", (float("nan"), float("inf"), True, "1"))
def test_invalid_reward_rejected(reward):
    controller = MemoryController("last16")
    controller.reset(obs())
    with pytest.raises(ValueError, match="finite"):
        controller.observe(obs(), 0, obs(), reward)
    assert not controller._transitions


def test_lifecycle_and_public_api_cannot_accept_goal_map_or_environment():
    controller = MemoryController("full")
    with pytest.raises(RuntimeError, match="reset"):
        controller.act(obs())
    with pytest.raises(RuntimeError, match="reset"):
        controller.observe(obs(), 0, obs(), 0.0)
    with pytest.raises(TypeError, match="Observation"):
        controller.reset({"x": 3, "y": 3, "heading": 0, "off_path": False, "goal": [0, 0]})
    assert list(inspect.signature(MemoryController).parameters) == ["mode", "priority"]
    assert isinstance(controller._transitions, deque)
