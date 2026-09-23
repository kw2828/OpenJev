"""Pure public-feature query schedules with an exact prefix query allowance.

Use one immutable gate configuration per episode with QueryGateActor(state_size=2).
Actor reset supplies [0, 0]: the next pre-action step and queries already used.
The caller must carry the returned state through the actual action history.
No scorer, observation filter, random generator, model or native runtime is used.

Random-pair offset is SHA256(NAMESPACE + big-endian uint32 episode seed +
big-endian uint32 pair index)[0] & 1. This is a reproducible hashed schedule,
not an independence claim. Label acquisition must never advance this state.
Entropy is the existing float32 entropy/log2(2809) feature, including its raw
mass convention; neither the belief nor this feature is renormalized or clipped.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

from openjev.research.otto_query_gate import FEATURE_DIM, FEATURE_NAMES, HORIZON

VERSION = "otto-sparse-query-v1"
KINDS = ("period2", "random_pair", "entropy")
STATE_SIZE = 2
STEP_INDEX = FEATURE_NAMES.index("step/2188")
ENTROPY_INDEX = FEATURE_NAMES.index("entropy/log2(2809)")
NAMESPACE = b"openjev.otto.sparse-query.random-pair.v1\x00"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pair(seed, index):
    digest = hashlib.sha256(NAMESPACE + struct.pack(">II", seed, index)).digest()
    return digest[0] & 1, digest.hex()


@dataclass(frozen=True)
class SparseQueryGate:
    """Stateless callback implementation; all carried memory is explicit state.

    __call__(features, state) returns the QueryGateActor pair (bool, float32[2]).
    decision(features, state) returns (bool, immutable next state, detached
    JSON-compatible metadata). Calling it again is pure and does not spend a
    query. Auditors can use the recorded feature/state/configuration directly.
    Queries after t decisions are bounded by ceil(t/2), including every prefix.
    """

    kind: str
    episode_seed: int
    entropy_threshold: float | None = None

    def __post_init__(self):
        _require(type(self.kind) is str and self.kind in KINDS, "known sparse query kind")
        _require(type(self.episode_seed) is int and 0 <= self.episode_seed < 2**32,
                 "episode seed must be uint32 Python integer")
        if self.kind == "entropy":
            _require(type(self.entropy_threshold) in (int, float) and math.isfinite(self.entropy_threshold),
                     "entropy gate requires a finite frozen threshold")
            object.__setattr__(self, "entropy_threshold", float(self.entropy_threshold))
        else:
            _require(self.entropy_threshold is None, "threshold is only valid for entropy gate")

    @property
    def configuration(self):
        return {"version": VERSION, "kind": self.kind, "episode_seed": self.episode_seed,
                "entropy_threshold": self.entropy_threshold, "state_size": STATE_SIZE,
                "horizon": HORIZON, "step_feature_index": STEP_INDEX, "entropy_feature_index": ENTROPY_INDEX,
                "random_namespace_hex": NAMESPACE.hex(), "random_encoding": "uint32 big-endian seed, pair index",
                "random_bit": "SHA256 digest byte 0, low bit selects within-pair offset"}

    def decision(self, features, state):
        _require(isinstance(features, np.ndarray) and features.dtype == np.float32
                 and features.shape == (FEATURE_DIM,) and np.isfinite(features).all(),
                 "features must be finite float32[31]")
        _require(isinstance(state, np.ndarray) and state.dtype == np.float32 and state.shape == (STATE_SIZE,)
                 and np.isfinite(state).all() and np.equal(state, np.floor(state)).all()
                 and np.all((state >= 0) & (state <= HORIZON)), "state must contain two bounded integer float32 counters")
        step, used = (int(v) for v in state)
        _require(step < HORIZON, "no query decision after horizon")
        _require(features[STEP_INDEX].tobytes() == np.float32(step / HORIZON).tobytes(),
                 "public feature step must exactly match chronological state")
        _require(used <= (step + 1) // 2, "state exceeds prior prefix query allowance")
        limit = (step + 2) // 2
        entropy = float(features[ENTROPY_INDEX])
        offset = digest = None
        if self.kind == "period2":
            _require(used == (step + 1) // 2, "period2 counter differs from its complete prefix")
            query = step % 2 == 0
        elif self.kind == "random_pair":
            offset, digest = _pair(self.episode_seed, step // 2)
            expected_used = step // 2 + int(step % 2 == 1 and offset == 0)
            _require(used == expected_used, "random-pair counter differs from its complete prefix")
            query = step % 2 == offset
        else:
            query = entropy >= self.entropy_threshold and used < limit
        after = used + int(query)
        _require(after <= limit, "query exceeds current prefix allowance")
        next_state = np.frombuffer(np.asarray([step + 1, after], dtype=np.float32).tobytes(), dtype=np.float32)
        record = {"version": VERSION, "kind": self.kind, "episode_seed": self.episode_seed,
                  "step": step, "queries_before": used, "queries_after": after, "query_limit": limit,
                  "query": query, "entropy_feature": entropy, "entropy_threshold": self.entropy_threshold,
                  "pair_index": step // 2 if self.kind == "random_pair" else None,
                  "pair_offset": offset, "pair_sha256": digest}
        return query, next_state, record

    def __call__(self, features, state):
        query, next_state, _ = self.decision(features, state)
        return query, next_state
