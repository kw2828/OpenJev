"""Independent saved-public-history checks for the two-observation controller.

NumPy and the standard library only: no model, controller, simulator or Torch
imports. The caller must authenticate the supplied ACTUAL public packet/action
prefix and expected model identity. This component checks their relationship to
saved buffers; it cannot prove the supplied prefix was physically executed.

Learned hidden states and imagined angular predictions receive only shape,
dtype and finiteness checks, never numerical neural-replay certification. Native
replay, score/search checks, source/weight/training lineage and full execution
coverage remain responsibilities of a future enclosing saved-output auditor.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping

import numpy as np

VERSION = "reacher-two-observation-public-history-audit-v1"
MODEL_CLASS = "TwoObservationHistoryGRUWorldModel"
WINDOW, MAX_STEPS, DT = 12, 50, 0.02
AGE_RTOL, AGE_ATOL = 1e-5, 1e-6
REAL_KEYS = frozenset({"real_packets", "real_actions", "real_present", "real_indices", "real_index", "real_target"})
STATE_KEYS = REAL_KEYS | {"hidden", "packet", "pending_action", "imagined_depth"}
IDENTITY_KEYS = {"version", "kind", "model_class", "configuration", "width", "weight_tensor_sha256",
                 "parameter_count", "parameter_tensor_bytes"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Finite JSON metadata required") from error


def _configuration(config):
    _require(type(config) is dict, "External model configuration must be a dict")
    h = config.get("hidden_size")
    noise = config.get("noise_std")
    _require(type(h) is int and h > 0, "Positive integer hidden_size")
    _require(type(noise) in (int, float) and math.isfinite(noise) and noise >= 0, "Finite nonnegative noise_std")
    _require(type(config.get("dt")) in (int, float) and config["dt"] == DT, "Only dt=.02 supported")
    _require(config.get("residual_reward") is True, "Controller requires residual_reward=True")
    # Duplicated intentionally, not imported from the implementation being audited.
    expected = {
        "version": "two-valid-observation-history-gru-v1", "model_class": MODEL_CLASS,
        "hidden_size": h, "dt": config["dt"], "noise_std": noise, "residual_reward": True,
        "valid_observations": 2, "max_real_packets": WINDOW, "max_issued_commands": WINDOW - 1,
        "episode_steps": MAX_STEPS, "age_rtol": AGE_RTOL, "age_atol": AGE_ATOL,
        "anchor": "older of last two actual valid observations; first observation before second exists",
        "real_history": "sanitized actual public packets and issued commands; integer indices; left padding",
        "assimilation": "zero-start reconstruction at every real boundary; initial-or-one-selected-advance phase",
        "replay": "all twelve assimilations and eleven full transitions, including padding, without detach",
        "imagined_rollout": "private recurrent hidden/packet; never appends real evidence",
        "overflow": "reject before dropping required anchor or command",
        "state_keys": sorted(STATE_KEYS), "run_status_authority": "enclosing protocol and execution receipts",
    }
    _require(_json(config) == _json(expected), "Exact actual-class/public-history configuration required")
    return h


def _identity(recorded, expected):
    _require(type(recorded) is dict and type(expected) is dict
             and set(recorded) == set(expected) == IDENTITY_KEYS, "Exact controller model identity membership")
    _require(_json(recorded) == _json(expected), "Model identity differs from external authenticated binding")
    h = _configuration(expected["configuration"])
    _require(expected["version"] == "reacher-two-observation-control-v1"
             and expected["kind"] == "two_observation_gru" and expected["model_class"] == MODEL_CLASS,
             "Actual two-observation controller identity required")
    _require(type(expected["width"]) is int and expected["width"] == h, "Identity width differs from configuration")
    count = 8 * h * h + 63 * h + 5
    _require(type(expected["parameter_count"]) is int and expected["parameter_count"] == count
             and type(expected["parameter_tensor_bytes"]) is int and expected["parameter_tensor_bytes"] == 4 * count,
             "Parameter count/float32 bytes do not match GRU schema")
    _require(type(expected["weight_tensor_sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", expected["weight_tensor_sha256"]) is not None,
             "Externally bound lowercase weight tensor SHA256 required")
    return h


def _array(value, shape, dtype, label, *, finite=True):
    _require(type(value) is np.ndarray and value.shape == shape and value.dtype == np.dtype(dtype),
             f"Exact array shape/dtype: {label}")
    if finite:
        _require(bool(np.isfinite(value).all()), f"Nonfinite array: {label}")
    return value


def reconstruct_public_history(public_packets, issued_commands, *, step, expected_configuration):
    """Return fresh expected root buffers for one synchronized batch boundary.

    Inputs are exactly [B,step+1,8] actual packets and [B,step,2] issued
    commands, not an episode with future rows. Step50 permits terminal-buffer
    inspection but cannot be passed to ``audit_decision``. Every prior boundary
    must fit the declared twelve-slot history, even if the final suffix fits.
    Unavailable angular placeholders, including NaN/inf, are discarded. All
    known public fields and visible angles must be finite. No inputs are mutated.
    """
    _configuration(expected_configuration)
    _require(type(step) is int and 0 <= step <= MAX_STEPS, "Integer real step in [0,50] required")
    _require(type(public_packets) is np.ndarray and public_packets.ndim == 3 and len(public_packets) > 0,
             "Nonempty batched public packet prefix required")
    b = len(public_packets)
    raw = _array(public_packets, (b, step + 1, 8), np.float32, "public_packets", finite=False)
    commands = _array(issued_commands, (b, step, 2), np.float32, "issued_commands")
    _require(bool((np.abs(commands) <= 1).all()), "Issued commands must already be clipped to [-1,1]")
    valid = raw[..., 6]
    _require(bool(((valid == 0) | (valid == 1)).all()), "Binary public validity required")
    visible = valid == 1
    clean = raw.copy()
    clean[..., :4] = np.where(visible[..., None], raw[..., :4], np.float32(0))
    _require(bool(np.isfinite(clean).all()), "Nonfinite visible angles or known public fields")
    _require(bool(visible[:, 0].all()), "First actual public packet must be visible")
    _require(bool((clean[..., 7] >= 0).all()) and bool((clean[..., 7][visible] == 0).all()),
             "Nonnegative age and exactly zero visible age required")
    target = clean[:, 0, 4:6]
    _require(np.array_equal(clean[..., 4:6], np.broadcast_to(target[:, None], (b, step + 1, 2))),
             "Public target must remain static across the entire prefix")
    result = {
        "real_packets": np.zeros((b, WINDOW, 8), dtype=np.float32),
        "real_actions": np.zeros((b, WINDOW - 1, 2), dtype=np.float32),
        "real_present": np.zeros((b, WINDOW), dtype=np.bool_),
        "real_indices": np.full((b, WINDOW), -1, dtype=np.int64),
        "real_index": np.full((b, 1), step, dtype=np.int64),
        "real_target": target.copy(), "packet": clean[:, -1].copy(),
        "pending_action": np.zeros((b, 2), dtype=np.float32),
        "imagined_depth": np.zeros((b, 1), dtype=np.int64),
    }
    for case in range(b):
        measurements = []
        for time in range(step + 1):
            if visible[case, time]:
                measurements.append(time)
            age = np.float32(time - measurements[-1]) * np.float32(DT)
            _require(np.isclose(clean[case, time, 7], age, rtol=AGE_RTOL, atol=AGE_ATOL),
                     "Public age disagrees with actual measurement indices")
            anchor = measurements[-2] if len(measurements) >= 2 else measurements[0]
            _require(time - anchor < WINDOW, "Required two-observation suffix exceeds twelve packets/eleven commands")
        length = step - anchor + 1
        start = WINDOW - length
        result["real_packets"][case, start:] = clean[case, anchor:]
        result["real_present"][case, start:] = True
        result["real_indices"][case, start:] = np.arange(anchor, step + 1, dtype=np.int64)
        if length > 1:
            result["real_actions"][case, start:] = commands[case, anchor:step]
    return result


def _state(state, batch, hidden_size, label):
    _require(isinstance(state, Mapping) and set(state) == STATE_KEYS, f"Exact state membership: {label}")
    shapes = {"hidden": (batch, hidden_size), "packet": (batch, 8),
              "real_packets": (batch, WINDOW, 8), "real_actions": (batch, WINDOW - 1, 2),
              "real_present": (batch, WINDOW), "real_indices": (batch, WINDOW),
              "real_index": (batch, 1), "real_target": (batch, 2),
              "pending_action": (batch, 2), "imagined_depth": (batch, 1)}
    for name, shape in shapes.items():
        dtype = np.bool_ if name == "real_present" else np.int64 if name in {
            "real_indices", "real_index", "imagined_depth"} else np.float32
        _array(state[name], shape, dtype, f"{label}.{name}")


def audit_decision(*, public_packets, issued_commands, step, root_state, carried_state,
                   selected_issued_command, recorded_identity, expected_identity):
    """Audit one root/selected-carried pair without learned-model replay.

    The selected command is supplied separately as the caller's actual issued
    command, not trusted merely because it appears in the carried state. All
    public buffers must match independently reconstructed evidence exactly.
    Carried age allows the frozen float32 age tolerance for one .02s increment.
    Learned hidden values and carried angular predictions remain unverified.
    """
    h = _identity(recorded_identity, expected_identity)
    _require(type(step) is int and 0 <= step < MAX_STEPS, "A decision requires real step in [0,49]")
    expected = reconstruct_public_history(public_packets, issued_commands, step=step,
                                          expected_configuration=expected_identity["configuration"])
    b = len(public_packets)
    _state(root_state, b, h, "root")
    _state(carried_state, b, h, "carried")
    action = _array(selected_issued_command, (b, 2), np.float32, "selected_issued_command")
    _require(bool((np.abs(action) <= 1).all()), "Selected issued command outside [-1,1]")
    for name, value in expected.items():
        _require(np.array_equal(root_state[name], value), f"Root differs from actual public prefix: {name}")
    for name in REAL_KEYS:
        _require(np.array_equal(carried_state[name], expected[name]), f"Selected advance mutated real evidence: {name}")
    _require(np.array_equal(carried_state["pending_action"], action), "Selected pending command differs from actually issued command")
    _require(bool((carried_state["imagined_depth"] == 1).all()), "Selected carried state must have depth one")
    packet = carried_state["packet"]
    _require(np.array_equal(packet[:, 4:6], expected["real_target"]) and bool((packet[:, 6] == 0).all()),
             "Carried target/validity differs from public clock")
    wanted_age = expected["packet"][:, 7] + np.float32(DT)
    _require(bool(np.isclose(packet[:, 7], wanted_age, rtol=AGE_RTOL, atol=AGE_ATOL).all()),
             "Carried age must increment by one real action interval")
    return {"version": VERSION, "status": "passed", "step": step, "cases": b,
            "retained_packets_per_case": expected["real_present"].sum(1).tolist(),
            "anchor_indices": [int(row[row >= 0][0]) for row in expected["real_indices"]],
            "expected_identity_sha256": hashlib.sha256(_json(expected_identity).encode()).hexdigest(),
            "public_history_verified": True, "selected_command_alignment_verified": True,
            "learned_hidden_numerically_verified": False, "predicted_angles_numerically_verified": False,
            "new_model_calls": 0, "new_native_calls": 0,
            "scope": "One supplied actual public prefix and saved root/carried pair; caller authenticates prefix and identity.",
            "excludes": ["native execution replay", "neural recomputation", "search/score arithmetic", "weight/source/training lineage", "full episode coverage"]}
