"""Exact released-policy input construction and an episode-local value LRU.

This module imports NumPy only, never TensorFlow, a model or a simulator.
``build_inputs`` preserves the original RLPolicy branch order and arithmetic;
it does not compute Q, select actions, normalize deficient mass, or omit blocked
actions. The supplied view must implement the qualified, pure public geometry.

Branch construction is adapted from OTTO RLPolicy._value_policy, copyright (c)
2023 by Aurore Loisy, MIT license at third_party/otto/LICENSE. Pinned upstream:
tmp/otto-source-review-01/isotropic/classes/rlpolicy.py, SHA256
d32b98f903f90bfeed40111c9120a524442e31c0d28eb7cdac0d6db80c2af5fb.

One cache belongs to one immutable model/runtime/symmetry configuration. The
caller MUST clear it at every episode boundary, including TRAIN/VALID changes.
Full C-order input bytes are keys, not hashes or approximate public features.
Only the 16 model values are cached: current masses and the original Q reduction
remain outside this component. Cache hits establish no new model qualification.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from numbers import Integral

import numpy as np

VERSION = "otto-exact-value-cache-v1"
INPUT_SHAPE = (16, 105, 105)
VALUE_SHAPE = (16, 1)
EPSILON = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, dtype, shape, name, *, nonnegative=False):
    _require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == shape
             and np.isfinite(value).all(), f"{name} must be finite {np.dtype(dtype).name}{shape}")
    if nonnegative:
        _require((value >= 0).all(), f"{name} must be nonnegative")


def build_inputs(view):
    """Return owned float32[16,105,105] inputs and float32[4,4] masses.

The leading order is action 0..3, then hit 0..3. A blocked move still uses the
view's original stationary successor, as RLPolicy does before action masking.
Float64 branch products, np.sum, np.maximum and centering precede the sole
float32 cast. Zero and subfloor branch mass deliberately remain subnormalized.
The caller owns source authentication and public-posterior reconstruction.
"""
    _require(view.Nactions == 4 and view.Nhits == 4 and tuple(view.NN_input_shape) == (105, 105),
             "view must expose the original four-action/four-hit geometry")
    probability, kernel, agent = view.p_source, view.p_Poisson, view.agent
    _array(probability, np.float64, (53, 53), "public posterior", nonnegative=True)
    _array(kernel, np.float64, (4, 107, 107), "public kernel", nonnegative=True)
    _require(isinstance(agent, (list, tuple, np.ndarray)) and len(agent) == 2
             and all(isinstance(x, Integral) and not isinstance(x, (bool, np.bool_))
                     and 0 <= x < 53 for x in agent), "public position must be two inbounds integers")
    inputs, masses = [], []
    for action in range(4):
        successor, _ = view._move(action, list(agent))
        evidence = view._extract_N_from_2N(input=kernel, origin=successor)
        _array(evidence, np.float64, (4, 53, 53), "public evidence", nonnegative=True)
        # Same operations and dtypes as original RLPolicy._value_policy.
        joint = deepcopy(probability)[np.newaxis, :] * evidence
        action_inputs, action_masses = [], []
        for hit in range(4):
            mass = np.maximum(EPSILON, np.sum(joint[hit]))
            centered = view._centeragent(joint[hit] / mass, successor)
            _array(centered, np.float64, (105, 105), "centered branch", nonnegative=True)
            action_inputs.append(centered)
            action_masses.append(mass)
        inputs.append(action_inputs)
        masses.append(action_masses)
    inputs = np.asarray(inputs, dtype=np.float32).reshape(INPUT_SHAPE)
    masses = np.asarray(masses, dtype=np.float32)
    _array(inputs, np.float32, INPUT_SHAPE, "model inputs", nonnegative=True)
    _array(masses, np.float32, (4, 4), "branch masses", nonnegative=True)
    return inputs, masses


class ExactValueCache:
    """Single-threaded, non-reentrant LRU for one fixed model and one episode.

``get_or_compute(inputs, compute)`` calls ``compute(readonly_inputs)`` on misses.
The callback receives an immutable exact snapshot, including signed zeros, and
must return finite native float32[16,1]. Returned values are detached writable
copies; internal values are immutable bytes. Noncontiguous inputs are accepted
and canonicalized to C order without altering numeric values or bit patterns.

``clear`` removes entries and preserves lifetime counters. A fresh instance is
required to start new counters. Invalid requests are not misses; callback and
invalid-output failures are misses but never insert or evict entries. Exceptions
propagate, and failed requests may be retried explicitly by the caller, never
automatically. No checkpoint identity is inferred or mixed into a cache key.
    The caller must never switch the underlying callback model, weights, dtype,
    symmetry mode or runtime while entries exist; use a fresh cache instead.
"""

    def __init__(self, capacity=64):
        _require(type(capacity) is int and capacity > 0, "capacity must be a positive Python integer")
        self._capacity = capacity
        self._entries = OrderedDict()
        self._busy = False
        self._counts = dict.fromkeys(("requests", "hits", "misses", "evictions", "invalid_requests",
                                      "compute_failures", "invalid_outputs", "clears"), 0)

    @property
    def capacity(self):
        return self._capacity

    def statistics(self):
        """Detached JSON-compatible cumulative counts and current owned bytes."""
        return {**self._counts,
                "failures": sum(self._counts[k] for k in
                                ("invalid_requests", "compute_failures", "invalid_outputs")),
                "entries": len(self._entries), "capacity": self.capacity,
                "key_bytes": sum(len(key) for key in self._entries),
                "value_bytes": sum(len(value) for value in self._entries.values())}

    def clear(self):
        """Start an empty episode cache without erasing failure or call history."""
        _require(not self._busy, "cannot clear during a cache computation")
        self._entries.clear()
        self._counts["clears"] += 1

    def get_or_compute(self, inputs, compute):
        _require(not self._busy, "cache computations must not be reentrant")
        self._counts["requests"] += 1
        try:
            _array(inputs, np.float32, INPUT_SHAPE, "model inputs")
            _require(callable(compute), "compute must be callable")
        except ValueError:
            self._counts["invalid_requests"] += 1
            raise
        key = inputs.tobytes(order="C")
        if key in self._entries:
            self._entries.move_to_end(key)
            self._counts["hits"] += 1
            saved = self._entries[key]
        else:
            self._counts["misses"] += 1
            self._busy = True
            try:
                result = compute(np.frombuffer(key, dtype=np.float32).reshape(INPUT_SHAPE))
            except BaseException:  # Account interrupted/failed callbacks, then propagate unchanged.
                self._counts["compute_failures"] += 1
                raise
            finally:
                self._busy = False
            try:
                _array(result, np.float32, VALUE_SHAPE, "model values")
            except ValueError:
                self._counts["invalid_outputs"] += 1
                raise
            saved = result.tobytes(order="C")
            self._entries[key] = saved
            if len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
                self._counts["evictions"] += 1
        return np.frombuffer(saved, dtype=np.float32).reshape(VALUE_SHAPE).copy()
