"""Public-state query allocation with injected score-only backends.

No model/runtime, simulator, random stream, checkpoint or training is loaded.
The gate sees cheap public features before any current neural evaluation. One
qualified filter owns the actual history; backends must be pure score readers.
Counters cover wrapper calls, not operations hidden inside an injected backend.
Failures prohibit continuation until an explicit new-episode reset. Reset does
not erase cumulative counters, unresolved operations or the failure history.
"""
from __future__ import annotations

import numpy as np

from openjev.research.otto_released_policy import EPSILON, PublicBeliefView, _integer, _move, _packet
from openjev.research.otto_restricted_policy import select_inbounds_action

VERSION = "otto-query-gate-v1"
HORIZON = 2188
FEATURE_NAMES = (
    "x/52", "y/52", *(f"legal_{a}" for a in range(4)), "sensing_length/5",
    *(f"hit_{h}" for h in range(4)), *(f"last_action_{a}" for a in range(4)),
    "step/2188", "query_age/2188", "has_queried", "mass", "entropy/log2(2809)", "distance/104",
    *(f"analytic_centered_{a}/64" for a in range(4)),
    *(f"analytic_delta_{a}/64" for a in range(4)), "entropy_delta", "distance_delta",
)
FEATURE_DIM = len(FEATURE_NAMES)
CHANNELS = ("view_initialization", "public_reset", "analytic_score", "feature_build", "gate",
            "neural_score", "public_update")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _immutable(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


class ReadOnlyBeliefView:
    """Original policy geometry with no public reset or observation mutator.

The posterior and position are detached copies; the shared kernel is backed by
immutable bytes. This is an API boundary for trusted scorers, not a sandbox.
"""

    __slots__ = ("__view",)
    N, Ndim, Nhits, Nactions, NN_input_shape = 53, 2, 4, 4, (105, 105)

    def __init__(self, view):
        self.__view = view

    @property
    def p_source(self):
        return self.__view.p_source

    @property
    def p_Poisson(self):
        return self.__view.p_Poisson

    @property
    def agent(self):
        return self.__view.agent

    def _move(self, action, agent):
        return self.__view._move(action, agent)

    def _extract_N_from_2N(self, input, origin):
        return self.__view._extract_N_from_2N(input, origin)

    def _centeragent(self, p, agent):
        return self.__view._centeragent(p, agent)


def _analytic(scores, allowed):
    _require(isinstance(scores, np.ndarray) and scores.dtype == np.float64 and scores.shape == (4,),
             "analytic scorer must return float64[4]")
    mask = np.asarray([a in allowed for a in range(4)], dtype=np.bool_)
    _require(np.isfinite(scores[mask]).all() and np.isposinf(scores[~mask]).all(),
             "analytic costs require finite eligible entries and +inf blocked entries")
    selected = int(np.flatnonzero(np.abs(scores - scores.min()) < EPSILON)[0])
    return selected, scores.copy()


def _features(view, public, sensing, analytic, last_action, last_query, previous):
    p = view.p_source
    positive = p > 0
    entropy = float(-np.sum(p[positive] * np.log2(p[positive])) / np.log2(53 * 53))
    axis = np.arange(53, dtype=np.float64)
    qx, qy = public["position"]
    distance = float(np.sum(p * (np.abs(axis[:, None] - qx) + np.abs(axis[None, :] - qy))) / 104)
    allowed = public["valid_actions"]
    centered = np.zeros(4, dtype=np.float64)
    centered[list(allowed)] = (analytic[list(allowed)] - np.mean(analytic[list(allowed)])) / 64
    statistics = np.concatenate((centered, [entropy, distance]))
    delta = np.zeros(6, dtype=np.float64) if previous is None else statistics - previous
    # Changing eligibility must not turn blocked cost placeholders into features.
    delta[:4][[a not in allowed for a in range(4)]] = 0
    age = public["step"] if last_query is None else public["step"] - last_query
    values = [qx / 52, qy / 52, *[float(a in allowed) for a in range(4)], sensing / 5,
              *[float(public["hit"] == h) for h in range(4)],
              *[float(last_action == a) for a in range(4)], public["step"] / HORIZON,
              age / HORIZON, float(last_query is not None), float(np.sum(p)), entropy, distance,
              *centered, *delta]
    result = np.asarray(values, dtype=np.float32)
    _require(result.shape == (FEATURE_DIM,) and np.isfinite(result).all(), "finite fixed public gate features")
    return _immutable(result), _immutable(statistics)


class QueryGateActor:
    """One actual action lifecycle around two functional score backends.

    analytic_scores(view) returns the in-bounds analytic float64[4] costs;
    neural_scores(view) returns all four unchanged neural float32 costs.
    The neural endpoint is the original released RestrictedPolicyActor: its
    strict near-tie subtraction stays float32, not the later Python-float rule
    used by separately trained action heads.
    gate(features, state) returns (Python bool, finite float32[state_size]).
    The gate never receives either scorer object, the view or neural costs.
    State/features passed to it are detached and immutable. State width may be
    zero for a stateless gate. Reset starts the native center/positive-hit prior.

    Optional check() runs before each operation attempt; emit(event) receives a
    small detached attempt/return record. Return is acknowledged only after emit
    succeeds. The caller owns durable storage, clocks and budgets. No retry or
    failed-operation cleanup is attempted. External backend state is not reset:
    backends are required to depend only on the supplied current public view.

    Query age follows gate decisions only. A future TRAIN collector's external
    teacher-annotation calls must have separate accounting and must not reset
    this age. Intermittent-query training requires explicit virtual schedules
    or collected gate behavior, not pretending every annotation was deployed.
    Entropy/distance features use raw public mass, including subnormalized or
    zero beliefs under the inherited floor; no normalization repair is made.
    """

    def __init__(self, initial_public, kernel, sensing_length, analytic_scores, neural_scores, gate,
                 *, state_size=0, check=None, emit=None):
        _require(all(callable(f) for f in (analytic_scores, neural_scores, gate)), "callable score backends and gate")
        _require(type(state_size) is int and 0 <= state_size <= 4096, "bounded nonnegative state size")
        _require(type(sensing_length) in (int, float) and np.isfinite(sensing_length) and sensing_length > 0,
                 "finite positive sensing length")
        _require(check is None or callable(check), "check must be callable")
        _require(emit is None or callable(emit), "emit must be callable")
        self._analytic_scores, self._neural_scores, self._gate = analytic_scores, neural_scores, gate
        self._sensing, self._state_size = float(sensing_length), state_size
        self._check, self._emit = check, emit
        self._counts = {name: {"attempted": 0, "returned": 0} for name in CHANNELS}
        self._pending_operations, self._errors = {}, []
        self._operation_id, self._episode, self._failed, self._active = 0, -1, False, False
        self._public, self._pending_action, self._last_choice = None, None, None
        self._view = self._call("view_initialization", lambda: PublicBeliefView(kernel))
        self._read_view = ReadOnlyBeliefView(self._view)
        self.reset(initial_public)

    @property
    def view(self):
        return self._read_view

    @property
    def public(self):
        return dict(self._public)

    @property
    def belief(self):
        return self._view.p_source

    @property
    def state(self):
        return _immutable(self._state)

    @property
    def last_choice(self):
        return None if self._last_choice is None else dict(self._last_choice)

    @property
    def progress(self):
        return {"episode": self._episode, "failed": self._failed, "pending_action": self._pending_action,
                "calls": {k: dict(v) for k, v in self._counts.items()},
                "pending_operations": [dict(v) for v in self._pending_operations.values()],
                "errors": [dict(v) for v in self._errors]}

    def _call(self, channel, function):
        if self._check is not None:
            self._check()
        self._operation_id += 1
        step = 0 if channel == "public_reset" else None if self._public is None else self._public["step"]
        event = {"id": self._operation_id, "channel": channel, "episode": self._episode,
                 "step": step}
        self._counts[channel]["attempted"] += 1
        self._pending_operations[event["id"]] = dict(event)
        if self._emit is not None:
            self._emit({"event": "attempt", **event})
        result = function()
        if self._emit is not None:
            self._emit({"event": "return", **event})
        self._counts[channel]["returned"] += 1
        del self._pending_operations[event["id"]]
        return result

    def _fail(self, error):
        self._failed = True
        self._errors.append({"episode": self._episode, "step": None if self._public is None else self._public["step"],
                             "operation_id": self._operation_id, "error": repr(error)})

    def _enter(self, *, reset=False):
        if self._active:
            raise RuntimeError("reentrant actor operation")
        if self._failed and not reset:
            raise RuntimeError("failed actor requires explicit new-episode reset")
        self._active = True

    def reset(self, initial_public):
        self._enter(reset=True)
        try:
            packet = _packet(initial_public, 0)
            _require(not packet["done"] and packet["hit"] in (1, 2, 3)
                     and packet["position"] == (26, 26), "reset requires positive-hit native center prior")
            self._episode += 1
            self._call("public_reset", lambda: self._view._reset(packet["hit"]))
            self._public, self._pending_action, self._failed = packet, None, False
            self._state = _immutable(np.zeros(self._state_size, dtype=np.float32))
            self._last_action = self._last_query = self._previous = self._last_choice = None
        except BaseException as error:  # Preserve interrupted operation evidence.
            self._fail(error)
            raise
        finally:
            self._active = False

    def choose(self):
        self._enter()
        try:
            _require(not self._public["done"], "cannot choose after source found")
            _require(self._pending_action is None, "chosen action is already awaiting its public outcome")
            allowed = self._public["valid_actions"]
            analytic_action, analytic = _analytic(
                self._call("analytic_score", lambda: self._analytic_scores(self.view)), allowed)
            features, statistics = self._call("feature_build", lambda: _features(
                self.view, self._public, self._sensing, analytic, self._last_action, self._last_query, self._previous))
            reply = self._call("gate", lambda: self._gate(features, self.state))
            _require(isinstance(reply, tuple) and len(reply) == 2 and type(reply[0]) is bool,
                     "gate must return (Python bool, float32 state)")
            query, next_state = reply
            _require(isinstance(next_state, np.ndarray) and next_state.dtype == np.float32
                     and next_state.shape == (self._state_size,) and np.isfinite(next_state).all(), "invalid gate state")
            next_state = _immutable(next_state)
            if query:
                scores = self._call("neural_score", lambda: self._neural_scores(self.view))
                action = select_inbounds_action(scores, allowed)
                scores = scores.copy()
            else:
                action, scores = analytic_action, analytic
            self._state, self._previous = next_state, statistics
            if query:
                self._last_query = self._public["step"]
            self._pending_action = action
            self._last_choice = {"step": self._public["step"], "queried": query, "action": action,
                                 "backend": "neural" if query else "analytic"}
            return action, scores
        except BaseException as error:  # No retry after ambiguous backend/journal work.
            self._fail(error)
            raise
        finally:
            self._active = False

    def update(self, action, public):
        self._enter()
        try:
            _require(not self._public["done"], "cannot update after source found")
            action = _integer(action, "action", 0, 3)
            _require(self._pending_action is None or action == self._pending_action, "observed action differs from pending")
            packet = _packet(public, self._public["step"] + 1)
            _require(packet["position"] == tuple(_move(action, self._public["position"])[0]),
                     "public position disagrees with actual action")
            self._call("public_update", lambda: self._view._observe(packet))
            self._public, self._pending_action, self._last_action = packet, None, action
        except BaseException as error:  # Filter or journal failure must block continuation.
            self._fail(error)
            raise
        finally:
            self._active = False
