"""Deterministic public-information controls for the official 7 by 7 grid task.

No environment, hidden path, goal, episode seed or path-progress information is
accepted. Safe/failed labels come only from the public off-path flag. Routes
minimize primitive actions (including turns), not grid distance. A forced return
after failure labels its endpoints but never creates an edge across the grid.

Window controls retain exactly the last K raw transitions, including both
endpoint observations. Derived maps and routes are rebuilt and discarded at
every decision. Rewards are retained as raw evidence and never used for search.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from numbers import Real

from openjev.research.mystery_path_observation import Observation

MODES = ("full", "last16", "last32", "erase_on_failure")
PRIORITIES = tuple(tuple((start + offset) % 4 for offset in range(4)) for start in range(4))
GRID_SIZE = 7
_DELTA = ((0, -1), (-1, 0), (0, 1), (1, 0))  # N, W, S, E
_ACTION_ORDER = (1, 2, 3)  # stable BFS order: left, right, forward


def _public(obs: Observation) -> tuple[int, int, int, bool]:
    if type(obs) is not Observation:
        raise TypeError("Expected exactly the public Observation class")
    if any(type(value) is not int for value in (obs.x, obs.y, obs.heading)):
        raise TypeError("Public coordinates and heading must be Python integers")
    if not (0 <= obs.x < GRID_SIZE and 0 <= obs.y < GRID_SIZE and 0 <= obs.heading < 4):
        raise ValueError("Public observation is outside the official grid/heading range")
    if type(obs.off_path) is not bool:
        raise TypeError("Public off_path must be bool")
    return obs.x, obs.y, obs.heading, obs.off_path


def _label(safe: set, failed: set, obs: tuple) -> None:
    cell = obs[:2]
    if obs[3]:
        safe.discard(cell)
        failed.add(cell)
    else:
        failed.discard(cell)
        safe.add(cell)


def _transition(before: Observation, action: int, after: Observation, reward: float) -> tuple:
    first, last = _public(before), _public(after)
    if type(action) is not int or action not in range(4):
        raise ValueError("Action must be an integer in 0..3")
    if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
        raise ValueError("Reward must be a finite real number")
    if first[3]:
        # Every supplied command is ignored here. Origin is not an extra input.
        if last[3] or last[2] != first[2]:
            raise ValueError("Forced return must clear off_path and preserve heading")
    else:
        x, y, heading, _ = first
        if action == 1:
            heading = (heading + 1) % 4
        elif action == 2:
            heading = (heading - 1) % 4
        elif action == 3:
            dx, dy = _DELTA[heading]
            x = min(GRID_SIZE - 1, max(0, x + dx))
            y = min(GRID_SIZE - 1, max(0, y + dy))
        if last[:3] != (x, y, heading):
            raise ValueError("Transition disagrees with public command kinematics")
        if last[3] and (action != 3 or last[:2] == first[:2]):
            raise ValueError("Only entering a different cell can newly fail")
    return first, action, last, float(reward)


def _first_action(safe: set, failed: set, public: tuple, priority: tuple) -> int:
    """BFS on safe (x,y,heading) states, followed by one unknown-cell entry.

    Frontier ties use final entry direction priority, then BFS discovery order
    (left, right, forward), then coordinates. Heading is part of distance, so
    a 180-degree turn costs two calls and consistently starts with left.
    """
    start = public[:3]
    queue = deque([(start, 0, None)])
    seen = {start}
    candidates = []
    best_distance = None
    discovery = 0
    while queue:
        (x, y, heading), distance, first_action = queue.popleft()
        if best_distance is not None and distance + 1 > best_distance:
            break
        dx, dy = _DELTA[heading]
        cell = (x + dx, y + dy)
        in_grid = all(0 <= value < GRID_SIZE for value in cell)
        if in_grid and cell not in safe and cell not in failed:
            best_distance = distance + 1
            candidates.append((priority.index(heading), discovery, cell, heading,
                               3 if first_action is None else first_action))
        discovery += 1
        for action in _ACTION_ORDER:
            if action == 1:
                next_state = (x, y, (heading + 1) % 4)
            elif action == 2:
                next_state = (x, y, (heading - 1) % 4)
            elif in_grid and cell in safe:
                next_state = (*cell, heading)
            else:
                continue
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, distance + 1, action if first_action is None else first_action))
    return min(candidates)[-1] if candidates else 0


def _retained_size(value: object, seen: set[int]) -> int:
    """Reachable Python object size, excluding class/module/function objects."""
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_retained_size(k, seen) + _retained_size(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return size + sum(_retained_size(item, seen) for item in value)
    if type(value) is MemoryController:
        return size + sum(_retained_size(getattr(value, name), seen) for name in value.__slots__)
    return size


class MemoryController:
    """One episode-local map or raw-transition window, with no stored plan.

    Call reset before an episode, then act on its current public observation and
    observe every actual transition, including turns, waits and forced returns.
    act and memory_bytes are read-only. Repeated reset clears all episode data.
    """

    __slots__ = ("_failed", "_mode", "_priority", "_ready", "_safe", "_transitions")

    def __init__(self, mode: str, priority: tuple[int, ...] = (0, 1, 2, 3)):
        if type(mode) is not str or mode not in MODES:
            raise ValueError(f"Mode must be one of {MODES}")
        if type(priority) is not tuple or any(type(item) is not int for item in priority) or priority not in PRIORITIES:
            raise ValueError("Priority must be a cyclic rotation tuple of (0,1,2,3)")
        self._mode, self._priority, self._ready = mode, priority, False
        self._safe: set[tuple[int, int]] = set()
        self._failed: set[tuple[int, int]] = set()
        self._transitions: deque = deque(maxlen={"last16": 16, "last32": 32}.get(mode, 0))

    def reset(self, obs: Observation) -> None:
        public = _public(obs)
        self._safe.clear()
        self._failed.clear()
        self._transitions.clear()
        if self._mode in ("full", "erase_on_failure"):
            _label(self._safe, self._failed, public)
        self._ready = True

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Call reset before using a controller")

    def act(self, obs: Observation) -> int:
        self._require_ready()
        public = _public(obs)
        if public[3]:
            return 0
        if self._mode in ("last16", "last32"):
            safe, failed = set(), set()
            for before, _action, after, _reward in self._transitions:
                _label(safe, failed, before)
                _label(safe, failed, after)
        else:
            safe, failed = self._safe.copy(), self._failed.copy()
        # The current safe position is admissible even after window eviction.
        _label(safe, failed, public)
        return _first_action(safe, failed, public, self._priority)

    def observe(self, before: Observation, action: int, after: Observation, reward: float) -> None:
        self._require_ready()
        transition = _transition(before, action, after, reward)
        first, _, last, _ = transition
        if self._mode in ("last16", "last32"):
            self._transitions.append(transition)
        else:
            if self._mode == "erase_on_failure" and last[3]:
                self._safe.clear()
                self._failed.clear()
            _label(self._safe, self._failed, first)
            _label(self._safe, self._failed, last)

    def memory_bytes(self) -> dict[str, int | str]:
        """Canonical UTF-8 state bytes and separately a Python object estimate.

        Serialization includes configuration/readiness and all retained episode
        data, but not temporary search workspace. Python size counts reachable
        objects once, including shared immutable values; it is not RSS or a
        claim about incremental allocator usage. Neither measure includes code.
        """
        state = {"mode": self._mode, "priority": self._priority, "ready": self._ready}
        if self._mode in ("last16", "last32"):
            state["transitions"] = list(self._transitions)
        else:
            state.update(safe=sorted(self._safe), failed=sorted(self._failed))
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {"serialized_state_bytes": len(encoded), "retained_python_bytes": _retained_size(self, set()),
                "retained_transitions": len(self._transitions),
                "scope": "retained state only; excludes code and temporary search workspace"}
