"""Public-angle finite-window identification for a custom Reacher tracking task.

This is a supplied-physics engineering reference, not a learned world model or
calibrated Bayesian estimator. A three-point backward derivative estimates
endpoint velocity. Each declared gain predicts the next angle from that same
public state. Velocity error and hidden actuator noise can bias inferred gain;
task qualification must measure that bias before using this reference.

Full, regularly sampled angle observations are required. New goals are public
at decision boundaries; neither their future values nor hidden gain schedules
enter this interface. Candidate planning is deliberately outside this module.
"""

from __future__ import annotations

import copy
import time
from collections import deque

import mujoco
import numpy as np

from openjev.research.reacher_physics_control import (
    _angles,
    _integer,
    _model,
    _packet,
    _real,
    _scalar,
    _step,
    _wrap,
)

VERSION = "reacher-tracking-window-identification-v1"


def _visible(value):
    result = _packet(value)
    if result[6] != 1:
        raise ValueError("Tracking identification requires full visible observations")
    return result


class TrackingAngleObserver:
    """Estimate state from at most three public, regularly sampled packets.

    Startup velocity is zero; the first completed transition uses a backward
    difference. Subsequently use (3*d_latest-d_previous)/(2*dt). Unwrapping
    assumes each observed displacement is less than pi. Initial winding is
    unidentifiable. Commands do not affect this measurement-only observer.
    """

    def __init__(self, initial_packet, *, dt):
        initial = _visible(initial_packet)
        self.dt = _scalar(dt, "dt", positive=True)
        self._position = _angles(initial)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._previous_delta = None
        self._last_packet = initial.copy()
        self._packets = deque([initial.copy()], maxlen=3)
        self._step_index = 0

    def estimate(self):
        return (np.r_[self._position, self._last_packet[4:6]],
                np.r_[self._velocity, [0., 0.]])

    def update(self, next_packet):
        packet = _visible(next_packet)
        delta = _wrap(_angles(packet) - _angles(self._last_packet))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            velocity = (delta / self.dt if self._previous_delta is None
                        else (3 * delta - self._previous_delta) / (2 * self.dt))
            position = self._position + delta
        if not np.isfinite(velocity).all() or not np.isfinite(position).all():
            raise FloatingPointError("Nonfinite public state estimate")
        self._position, self._velocity = position, velocity
        self._previous_delta = delta.copy()
        self._last_packet = packet.copy()
        self._packets.append(packet.copy())
        self._step_index += 1
        return self.estimate()

    def snapshot(self):
        qpos, qvel = self.estimate()
        return {"step_index": self._step_index, "dt": self.dt,
                "packets": np.stack(self._packets), "qpos_estimate": qpos,
                "qvel_estimate": qvel, "velocity_source": "public backward differences",
                "initial_velocity": "zero", "current_target_only": True,
                "missing_observations_supported": False}


class WindowedGainIdentifier:
    """Finite-window scalar grid identification with an explicit nominal model.

    Gains, window size and prior must be supplied. The model must have nominal
    native gears 200; independent copies carry each candidate multiplier. An
    exactly flat residual bank retains the previous estimate rather than
    manufacturing evidence for an endpoint gain. Otherwise earliest minimum
    wins. A nonzero residual spread is not a calibrated confidence measure.
    """

    def __init__(self, model, initial_packet, *, gains, window, prior_gain, frame_skip):
        started = time.perf_counter()
        nominal = _model(model)
        self.frame_skip = _integer(frame_skip, "frame_skip", 1)
        self.window = _integer(window, "window", 1)
        expected_gear = np.zeros((2, 6), dtype=np.float64)
        expected_gear[:, 0] = 200.
        if not np.array_equal(nominal.actuator_gear, expected_gear):
            raise ValueError("Supply independent nominal Reacher gears 200, not hidden current gain")
        raw = np.asarray(gains)
        if raw.ndim != 1 or len(raw) < 2:
            raise ValueError("gains must be a one-dimensional bank with at least two entries")
        values = _real(raw, raw.shape, "gains")
        if (values <= 0).any() or (np.diff(values) <= 0).any() or not np.isfinite(values * 200).all():
            raise ValueError("gains must be finite positive strictly increasing values")
        prior = _scalar(prior_gain, "prior_gain", positive=True)
        if not np.any(values == prior):
            raise ValueError("prior_gain must be present exactly in the candidate bank")
        self._gains = values.copy()
        self._gain = prior
        self._prior_gain = prior
        self._observer = TrackingAngleObserver(initial_packet,
            dt=_scalar(nominal.opt.timestep, "native timestep", positive=True) * self.frame_skip)
        self._models = [copy.copy(nominal) for _ in values]
        for member, gain in zip(self._models, values, strict=True):
            member.actuator_gear[:] = expected_gear * gain
        self._data = [mujoco.MjData(member) for member in self._models]
        self._residuals = deque(maxlen=self.window)
        self._last_trace = None
        self._step_index = 0
        self._failed = False
        self._candidate_attempts = self._candidate_completed = 0
        self._substeps_requested = self._substeps_completed = 0
        self._forward_calls_completed = 0
        self._update_seconds = 0.
        self._setup_seconds = time.perf_counter() - started

    def configuration(self):
        return {"version": VERSION, "gains": self._gains.copy(), "window": self.window,
                "prior_gain": self._prior_gain, "frame_skip": self.frame_skip,
                "dt": self._observer.dt, "learned": False, "privileged_live_state": False,
                "model_knowledge": "supplied full nominal Reacher dynamics; explicit scalar gear grid",
                "objective": "sum over last window of mean squared four cosine/sine errors",
                "tie_rule": "retain prior estimate if entire bank exactly flat; otherwise earliest minimum",
                "noise_prediction": "zero; actual applied actuator noise unavailable",
                "measurement_source": "full public angle packets and issued commands only",
                "limit": "endpoint velocity approximation and hidden noise can bias gain",
                "residual_spread_is_confidence": False}

    def estimate(self):
        if self._failed:
            raise RuntimeError("Failed identifier is terminal")
        qpos, qvel = self._observer.estimate()
        return qpos, qvel, float(self._gain)

    def update(self, issued_command, next_packet):
        if self._failed:
            raise RuntimeError("Failed identifier is terminal")
        started = time.perf_counter()
        packet = _visible(next_packet)
        command = np.clip(_real(issued_command, (2,), "issued_command"), -1., 1.).astype(np.float32)
        # Validate the derivative before making any simulator call or mutation.
        next_observer = copy.deepcopy(self._observer)
        next_observer.update(packet)
        qpos, qvel = self._observer.estimate()
        predictions = np.empty((len(self._gains), 4), dtype=np.float64)
        try:
            for index, (model, data) in enumerate(zip(self._models, self._data, strict=True)):
                self._candidate_attempts += 1
                self._substeps_requested += self.frame_skip
                mujoco.mj_resetData(model, data)
                data.qpos[:], data.qvel[:] = qpos, qvel
                mujoco.mj_forward(model, data)
                self._forward_calls_completed += 1
                _step(model, data, command.astype(np.float64), self.frame_skip)
                self._candidate_completed += 1
                self._substeps_completed += self.frame_skip
                predictions[index] = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])]
            residual = np.mean((predictions - packet[:4]) ** 2, axis=1)
            retained = [*self._residuals, residual][-self.window:]
            scores = np.sum(np.stack(retained), axis=0, dtype=np.float64)
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite candidate residuals")
            flat = bool(np.all(scores == scores[0]))
            gain = self._gain if flat else float(self._gains[int(np.argmin(scores))])
            self._last_trace = {"step_index": self._step_index, "root_qpos": qpos,
                "root_qvel": qvel, "issued_command": command.copy(),
                "next_packet": packet.copy(), "candidate_predictions": predictions,
                "candidate_residuals": residual, "window_residual_sums": scores,
                "retained_transitions": len(retained), "exactly_flat_bank": flat,
                "residual_spread": float(np.max(scores) - np.min(scores)),
                "gain_before": float(self._gain), "gain_after": gain}
            self._residuals.append(residual.copy())
            self._observer, self._gain = next_observer, gain
            self._step_index += 1
            return self.estimate()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._update_seconds += time.perf_counter() - started

    def snapshot(self):
        return {"configuration": self.configuration(), "step_index": self._step_index,
                "failed": self._failed, "gain_estimate": float(self._gain),
                "observer": self._observer.snapshot(), "last_trace": copy.deepcopy(self._last_trace),
                "window_residuals": (np.stack(self._residuals) if self._residuals
                                     else np.empty((0, len(self._gains)), dtype=np.float64)),
                "costs": {"setup_wall_seconds": self._setup_seconds,
                    "update_wall_seconds": self._update_seconds,
                    "native_transition_attempts": self._candidate_attempts,
                    "native_transitions_completed": self._candidate_completed,
                    "native_substeps_requested": self._substeps_requested,
                    "native_substeps_completed": self._substeps_completed,
                    "root_forward_calls_completed": self._forward_calls_completed,
                    "postconstraint_refresh_calls_completed": self._candidate_completed,
                    "native_model_copies": len(self._models), "native_data_instances": len(self._data),
                    "retained_residual_scalars": len(self._residuals) * len(self._gains)}}
