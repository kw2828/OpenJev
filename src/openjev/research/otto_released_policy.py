"""Public-only belief adapter for the injected official OTTO RLPolicy.

Pure movement, crop, centering and belief updates follow SourceTracking at
1467029f399dc5eeac8652499a9c8326ecab4575. This module imports no simulator or
model runtime and loads no files. The caller supplies a frozen likelihood
kernel, an independently qualified model and the authenticated policy class.

All four actions exist, including blocked moves that stay and receive a hit.
The public ``valid_actions`` field describes in-bounds moves, not an action
mask. A caller-owned horizon stops the episode after incorporating its final
nonterminal packet; it must not fabricate a found event for censoring.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np

VERSION = "otto-released-public-policy-v1"
N, NHITS, EPSILON = 53, 4, 1e-10
PUBLIC_FIELDS = frozenset({"position", "hit", "done", "step", "valid_actions"})


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} outside the allowed range")
    return int(value)


def _position(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError("position must have two coordinates")
    return tuple(_integer(x, "coordinate", 0, N - 1) for x in value)


def _move(action, position):
    action = _integer(action, "action", 0, 3)
    result = list(_position(position))
    axis, delta = action // 2, 2 * (action % 2) - 1
    possible = 0 <= result[axis] + delta < N
    if possible:
        result[axis] += delta
    return result, possible


def _packet(public, expected_step):
    if isinstance(public, tuple) and hasattr(public, "_fields") and hasattr(public, "_asdict"):
        public = public._asdict()
    if not isinstance(public, Mapping) or set(public) != PUBLIC_FIELDS:
        raise ValueError("exact public packet fields required")
    step = _integer(public["step"], "step", 0)
    if step != expected_step:
        raise ValueError("public step is not the next expected step")
    if not isinstance(public["done"], (bool, np.bool_)):
        raise TypeError("done must be boolean")
    done = bool(public["done"])
    hit = _integer(public["hit"], "hit", -2, 3)
    if (done and hit != -2) or (not done and hit < 0):
        raise ValueError("found sentinel and done disagree")
    position = _position(public["position"])
    moves = public["valid_actions"]
    if not isinstance(moves, (list, tuple)):
        raise TypeError("valid_actions must be an ordered public sequence")
    moves = tuple(_integer(x, "valid action", 0, 3) for x in moves)
    expected = () if done else tuple(a for a in range(4) if _move(a, position)[1])
    if moves != expected:
        raise ValueError("valid_actions must describe precisely the in-bounds moves")
    return {"position": position, "hit": hit, "done": done, "step": step, "valid_actions": moves}


class PublicBeliefView:
    """Narrow official-policy view; its posterior is derived only from public inputs.

    ``p_source`` is an owned snapshot, not the simulator's posterior. Copies and
    immutable kernel backing prevent the injected policy or caller from mutating
    the actor's arrays through its public view. No trajectory is retained.
    """

    __slots__ = ("_agent", "_kernel", "_probability")
    N, Ndim, Nhits, Nactions = 53, 2, 4, 4
    NN_input_shape = (105, 105)

    def __init__(self, kernel):
        if (not isinstance(kernel, np.ndarray) or kernel.dtype != np.float64
                or kernel.shape != (4, 107, 107) or not np.isfinite(kernel).all()
                or (kernel < 0).any() or (kernel > 1).any()
                or np.any(kernel[:, 53, 53] != 0)):
            raise ValueError("kernel must be finite float64[4,107,107] in [0,1] with zero origin")
        self._kernel = np.frombuffer(kernel.tobytes(order="C"), dtype=np.float64).reshape(kernel.shape)
        self._probability = np.zeros((N, N), dtype=np.float64)
        self._agent = (26, 26)

    @property
    def p_Poisson(self):
        return self._kernel

    @property
    def p_source(self):
        return self._probability.copy()

    @property
    def agent(self):
        return list(self._agent)

    def _move(self, action, agent):
        return _move(action, agent)

    def _extract_N_from_2N(self, input, origin):
        position = _position(origin)
        if not isinstance(input, np.ndarray) or input.ndim < 2 or input.shape[-2:] not in ((107, 107), (105, 105)):
            raise ValueError("crop input must end in 107x107 or 105x105")
        center = 53 if input.shape[-1] == 107 else 52
        x, y = center - position[0], center - position[1]
        return input[..., x:x + N, y:y + N]

    def _centeragent(self, p, agent):
        position = _position(agent)
        if (not isinstance(p, np.ndarray) or p.shape != (N, N) or p.dtype.kind != "f"
                or not np.isfinite(p).all() or (p < 0).any()):
            raise ValueError("centering requires a finite nonnegative float grid")
        # Exact upstream float64 zero-padding. RLPolicy casts its final batch.
        result = np.zeros((105, 105), dtype=np.float64)
        x, y = 52 - position[0], 52 - position[1]
        result[x:x + N, y:y + N] = p
        return result

    def _updated(self, probability, position, hit, done):
        if done:
            result = np.zeros((N, N), dtype=np.float64)
            result[position] = 1.0
            return result  # Sentinel -2 is never a likelihood index.
        result = probability.copy()
        result[position] = 0
        result *= self._extract_N_from_2N(self._kernel, position)[hit]
        result[(result < 0) & (result > -1e-15)] = 0
        mass = np.sum(result)
        if not np.isfinite(result).all() or (result < 0).any() or not np.isfinite(mass):
            raise FloatingPointError("invalid public belief update")
        if mass > EPSILON:
            result /= mass
        return result

    def _reset(self, hit):
        prior = np.ones((N, N), dtype=np.float64) / (N * N - 1)
        prior[26, 26] = 0
        self._probability = self._updated(prior, (26, 26), hit, False)
        self._agent = (26, 26)

    def _observe(self, packet):
        result = self._updated(self._probability, packet["position"], packet["hit"], packet["done"])
        self._probability = result
        self._agent = packet["position"]

    def storage_bytes(self):
        return {"mutable_array_bytes": self._probability.nbytes,
                "immutable_array_bytes": self._kernel.nbytes,
                "scope": "Owned posterior and immutable kernel only; excludes external model, policy object, scalar metadata and temporary snapshots/workspace."}


class ReleasedPolicyActor:
    """Official-policy injection over public-only state, with explicit lifecycle.

    ``policy_class`` must provide the official constructor and ``_value_policy``
    return contract. ``update(action, public)`` also accepts prescribed public
    actions when there is no pending choice, for controlled integration tests.
    If choose() has run, the next update must use that exact chosen action.
    """

    __slots__ = ("_pending_action", "_policy", "_public", "_view")

    def __init__(self, initial_public, kernel, model, policy_class, *, sym_avg=True):
        if not isinstance(sym_avg, bool):
            raise TypeError("sym_avg must be bool")
        packet = _packet(initial_public, 0)
        if packet["done"] or packet["hit"] not in (1, 2, 3) or packet["position"] != (26, 26):
            raise ValueError("reset requires a positive initial hit at the native center")
        self._view = PublicBeliefView(kernel)
        self._view._reset(packet["hit"])
        self._public, self._pending_action = packet, None
        self._policy = policy_class(env=self._view, model=model, sym_avg=sym_avg)

    @property
    def belief(self):
        return self._view.p_source

    @property
    def public(self):
        return dict(self._public)

    def reset(self, initial_public):
        packet = _packet(initial_public, 0)
        if packet["done"] or packet["hit"] not in (1, 2, 3) or packet["position"] != (26, 26):
            raise ValueError("reset requires a positive initial hit at the native center")
        self._view._reset(packet["hit"])
        self._public, self._pending_action = packet, None

    def choose(self):
        if self._public["done"]:
            raise RuntimeError("cannot choose after source found")
        if self._pending_action is not None:
            raise RuntimeError("a chosen action is already awaiting its public outcome")
        action, scores = self._policy._value_policy()
        action = _integer(action, "policy action", 0, 3)
        if (not isinstance(scores, np.ndarray) or scores.shape != (4,) or scores.dtype != np.float32
                or not np.isfinite(scores).all()):
            raise ValueError("official policy must return four finite float32 costs")
        first = int(np.flatnonzero(np.abs(scores - scores.min()) < EPSILON)[0])
        if action != first:
            raise ValueError("policy action disagrees with the upstream first near-tie rule")
        self._pending_action = action
        return action, scores.copy()

    def update(self, action, public):
        if self._public["done"]:
            raise RuntimeError("cannot update after source found; reset is required")
        action = _integer(action, "action", 0, 3)
        if self._pending_action is not None and action != self._pending_action:
            raise ValueError("observed action differs from the pending policy action")
        packet = _packet(public, self._public["step"] + 1)
        moved, _ = _move(action, self._public["position"])
        if packet["position"] != tuple(moved):
            raise ValueError("public position disagrees with the supplied action")
        self._view._observe(packet)
        self._public, self._pending_action = packet, None

    def storage_bytes(self):
        return self._view.storage_bytes()
