"""Causal, explicit CEM proposal centers; no RNG, physics or learned state.

One instance belongs to one real episode/case. ``commit`` is an acknowledgement
boundary: its caller must invoke it only after the selected command actually
returns from the environment. This module can verify the command/sequence
identity but cannot certify that a native transition happened. Candidate work
never changes this cache. Any failed operation is terminal for this instance.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from openjev.research.reacher_adaptive_search import SearchInputs

VERSION = "reacher-cem-proposal-memory-v1"
MODES = ("cold", "repeat_last", "shift_plan")
ARITHMETIC = "initial[:,7:64] + center / scales[7:64]; unchanged kernel then multiplies scales"


def _integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive builtin int")
    return value


def _array(value, shape, label):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != np.float32 or not np.isfinite(value).all()):
        raise ValueError(f"{label} must be finite float32{shape}")
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(shape)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return copy.deepcopy(value)


def _bytes(values):
    seen = set()

    def visit(value):
        if isinstance(value, np.ndarray):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            return value.nbytes
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        return 0

    return visit(values)


class CEMProposalMemory:
    """Center initial proposals while preserving every downstream CEM choice.

    Scales remain .25 for initial slots0:32 and .75 for32:64. Raw slots0:7
    are unchanged because the search kernel overwrites them with its anchors.
    ``random_extra`` and all CEM generations are byte-identical to the supplied
    inputs. A zero center returns the original SearchInputs object itself.

    A shifted plan removes one *real action*, hold-pads the far planning tail
    only if necessary, then takes explicit float64 column means within each
    current action block. This is a projection, not an exact block-aligned
    trajectory shift. No future target, gain or event schedule is accepted.
    """

    def __init__(self, mode, steps, planning_horizon, action_block):
        started = time.perf_counter()
        if type(mode) is not str or mode not in MODES:
            raise ValueError("mode must be cold, repeat_last or shift_plan")
        self._mode = mode
        self._steps = _integer(steps, "steps")
        self._horizon = _integer(planning_horizon, "planning_horizon")
        self._block = _integer(action_block, "action_block")
        if not self._block <= self._horizon <= self._steps:
            raise ValueError("action_block <= planning_horizon <= steps")
        self._chunks = (self._horizon + self._block - 1) // self._block
        self._next_step = 0
        self._pending = None
        self._last_target = self._last_sequence = self._last_command = None
        self._last_trace = None
        self._failed, self._failure = False, None
        self._counts = dict.fromkeys((
            "prepare_attempted", "prepare_completed", "commit_attempted", "commit_completed",
            "failed_operations", "startup_prepares", "target_change_prepares", "zero_center_prepares",
            "transformed_prepares", "projected_blocks", "column_mean_calls", "tail_actions_repeated",
            "transformed_initial_scalars", "input_payload_bytes_copied", "input_identity_bytes_hashed",
            "snapshot_calls"), 0)
        self._prepare_seconds = self._commit_seconds = self._snapshot_seconds = 0.
        self._setup_seconds = time.perf_counter() - started

    def configuration(self):
        return {"version": VERSION, "mode": self._mode, "steps": self._steps,
                "planning_horizon": self._horizon, "action_block": self._block,
                "cases_per_instance": 1, "rng_draws": 0, "native_calls": 0,
                "reset_rule": "startup or observed current public target change only",
                "cache_boundary": "commit after actual issued-command acknowledgement",
                "projection": "one-action shift, hold-pad, explicit float64 column means in current blocks",
                "transform_arithmetic": ARITHMETIC,
                "payload_bytes_scope": "unique retained ndarray payloads; excludes Python/JSON overhead and returned copies",
                "timing_scope": "setup/prepare/commit/snapshot separately; caller adds native work and trace storage",
                "run_status_authority": "enclosing protocol and execution receipts"}

    def _live(self):
        if self._failed:
            raise RuntimeError("failed proposal memory is terminal")

    def _fail(self, operation, error):
        self._failed = True
        self._counts["failed_operations"] += 1
        self._failure = {"operation": operation, "next_step": self._next_step,
                         "error_type": type(error).__name__, "message": str(error)}

    def _costs(self):
        retained = _bytes((self._last_target, self._last_sequence, self._last_command,
                           self._pending, self._last_trace))
        return {**self._counts, "setup_wall_seconds": self._setup_seconds,
                "prepare_wall_seconds": self._prepare_seconds,
                "commit_wall_seconds": self._commit_seconds,
                "snapshot_wall_seconds": self._snapshot_seconds,
                "retained_numpy_payload_bytes": retained}

    def prepare(self, inputs: SearchInputs, public_target, *, step):
        """Return immutable transformed inputs and a detached JSON-safe trace.

        Neither the cursor nor the acknowledged trajectory advances here.
        Copies and input hashing are paid even though no candidate is scored.
        Cost fields are attached after measuring trace materialization; their
        final dictionary construction is not included in the measured duration.
        """
        self._live()
        started = time.perf_counter()
        self._counts["prepare_attempted"] += 1
        try:
            if self._pending is not None:
                raise RuntimeError("one pending preparation must be committed first")
            if type(step) is not int or step != self._next_step or step >= self._steps:
                raise ValueError("step must be the exact next real decision before terminal")
            if type(inputs) is not SearchInputs or inputs.initial.shape != (1, 64, self._chunks, 2):
                raise ValueError("single-case full-horizon SearchInputs required")
            target = _array(public_target, (2,), "public_target")
            h = min(self._horizon, self._steps-step)
            active_chunks = (h + self._block - 1) // self._block
            reason = ("startup" if self._last_target is None else
                      "observed_target_change" if not np.array_equal(target, self._last_target) else "none")
            center = np.zeros((self._chunks, 2), dtype=np.float64)
            shifted = None
            if reason == "none" and self._mode == "repeat_last":
                center[:active_chunks] = self._last_command
            elif reason == "none" and self._mode == "shift_plan":
                tail = self._last_sequence[1:]
                repeats = max(0, h-len(tail))
                shifted = np.concatenate((tail, np.repeat(self._last_sequence[-1:], repeats, axis=0)), axis=0)[:h].copy()
                for block in range(active_chunks):
                    segment = shifted[block*self._block:min((block+1)*self._block, h)]
                    for column in range(2):
                        center[block, column] = np.mean(segment[:, column], dtype=np.float64)
                self._counts["projected_blocks"] += active_chunks
                self._counts["column_mean_calls"] += 2*active_chunks
                self._counts["tail_actions_repeated"] += repeats
            payload_bytes = sum(value.nbytes for value in (inputs.initial, inputs.random_extra, *inputs.cem))
            base_ids = dict(inputs.identities())
            self._counts["input_identity_bytes_hashed"] += payload_bytes
            zero = bool(np.all(center == 0))
            transformed = inputs
            if not zero:
                initial = inputs.initial.copy()
                self._counts["input_payload_bytes_copied"] += initial.nbytes
                scales = np.r_[np.full(32, .25), np.full(32, .75)]
                with np.errstate(over="ignore", invalid="ignore"):
                    initial[:, 7:64] = initial[:, 7:64] + center[None, None]/scales[None, 7:64, None, None]
                if not np.isfinite(initial).all():
                    raise FloatingPointError("nonfinite transformed proposal innovations")
                transformed = SearchInputs(initial, inputs.random_extra, inputs.cem)
                self._counts["input_payload_bytes_copied"] += payload_bytes
                self._counts["transformed_initial_scalars"] += 57*self._chunks*2
                transformed_ids = dict(transformed.identities())
                self._counts["input_identity_bytes_hashed"] += payload_bytes
            else:
                transformed_ids = dict(base_ids)
            self._pending = {"step": step, "horizon": h, "public_target": target}
            self._last_trace = {"version": VERSION, "mode": self._mode, "step": step,
                "horizon": h, "action_block": self._block, "reset_reason": reason,
                "public_target": target, "center": center, "shifted_sequence": shifted,
                "base_input_identities": base_ids, "transformed_input_identities": transformed_ids,
                "input_object_reused": zero, "transform_arithmetic": ARITHMETIC}
            trace = _json(self._last_trace)
            self._counts["prepare_completed"] += 1
            self._counts["zero_center_prepares" if zero else "transformed_prepares"] += 1
            if reason == "startup":
                self._counts["startup_prepares"] += 1
            elif reason == "observed_target_change":
                self._counts["target_change_prepares"] += 1
        except BaseException as error:
            self._fail("prepare", error)
            raise
        finally:
            self._prepare_seconds += time.perf_counter()-started
        trace["costs"] = self._costs()
        return transformed, trace

    def commit(self, selected_sequence, issued_command):
        """Acknowledge one actual command; validate everything before cache change."""
        self._live()
        started = time.perf_counter()
        self._counts["commit_attempted"] += 1
        try:
            if self._pending is None:
                raise RuntimeError("commit requires exactly one pending preparation")
            sequence = _array(selected_sequence, (self._pending["horizon"], 2), "selected_sequence")
            command = _array(issued_command, (2,), "issued_command")
            if (np.abs(sequence) > 1).any() or (np.abs(command) > 1).any():
                raise ValueError("selected sequence and command must already lie in [-1,1]")
            if not np.array_equal(sequence[0], command):
                raise ValueError("acknowledged issued command differs from selected first action")
            self._last_target = self._pending["public_target"]
            self._last_sequence, self._last_command = sequence, command
            self._next_step += 1
            self._pending = None
            self._counts["commit_completed"] += 1
        except BaseException as error:
            self._fail("commit", error)
            raise
        finally:
            self._commit_seconds += time.perf_counter()-started

    def snapshot(self):
        """Detached JSON-safe evidence, also available after failure or terminal."""
        started = time.perf_counter()
        self._counts["snapshot_calls"] += 1
        result = _json({"configuration": self.configuration(), "next_step": self._next_step,
            "pending": self._pending, "last_target": self._last_target,
            "last_selected_sequence": self._last_sequence, "last_issued_command": self._last_command,
            "failed": self._failed, "failure": self._failure, "last_trace": self._last_trace})
        self._snapshot_seconds += time.perf_counter()-started
        result["costs"] = self._costs()
        return result
