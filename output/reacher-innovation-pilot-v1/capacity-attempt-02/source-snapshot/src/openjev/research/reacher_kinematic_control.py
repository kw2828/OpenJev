"""Public-measurement kinematics with explicitly supplied nominal Reacher physics.

This is a competence reference, not a learned controller or true-state oracle.
Only the supplied model's parameters, public packets and issued commands enter
its estimate. Velocity is the shortest wrapped angle displacement between the
last two valid measurements divided by their elapsed control time. It is an
interval average, not an instantaneous velocity measurement. Rotations larger
than pi between measurements are ambiguous. The first estimate uses principal
angles and exactly zero velocity; a single packet cannot identify winding.

At later measurements, positions are reanchored on the branch nearest the
nominal prediction. This preserves an elbow that transiently crosses its soft
joint limit instead of jumping to the opposite stop. Missing measurements are
strict: angular placeholders must be finite zeros. During blackouts the private
simulator propagates the issued, clipped float32 commands with no disturbance.
The unknown realized actuator noise is never recovered or supplied.
"""

from __future__ import annotations

import copy
import time

import mujoco
import numpy as np

from openjev.research.reacher_physics_control import (
    PhysicsMPC,
    _angles,
    _integer,
    _model,
    _packet,
    _real,
    _scalar,
    _step,
    _wrap,
)

VERSION = "reacher-public-two-measurement-kinematics-v1"


class KinematicObserver:
    """One private nominal simulator, initialized from one valid public packet.

    Each update represents exactly ``frame_skip`` native substeps with one
    issued command. Elapsed time between observations is accumulated across
    every update, including missing packets; packet age is checked against it.
    Both model parameters and input arrays are copied. No environment or live
    ``MjData`` is accepted, stored or inspected.
    """

    def __init__(self, model, initial_packet, *, frame_skip=2):
        started = time.perf_counter()
        supplied = _model(model)
        self.frame_skip = _integer(frame_skip, "frame_skip", 1)
        self._model = copy.copy(supplied)
        self.native_timestep = _scalar(self._model.opt.timestep, "native timestep", positive=True)
        self.dt = _scalar(self.native_timestep * self.frame_skip, "decision dt", positive=True)
        initial = _packet(initial_packet)
        if not initial[6]:
            raise ValueError("Initial packet must contain a valid angle observation")
        self._target = initial[4:6].copy()
        self._data = mujoco.MjData(self._model)
        self._data.qpos[:] = np.concatenate((_angles(initial), self._target))
        self._data.qvel[:] = 0.
        mujoco.mj_forward(self._model, self._data)
        self._step_index = self._last_valid_step = 0
        self._last_valid_angles = _angles(initial)
        self._valid_measurements = [(0, initial.copy())]
        self._last_command = None
        self._last_packet = initial.copy()
        self._last_velocity_interval = None
        self._failed = False
        self._native_transitions = self._native_substeps = 0
        self._native_transition_attempts = self._native_substeps_requested = 0
        self._reanchor_forward_calls = 0
        self._update_wall_seconds = 0.
        self._setup_wall_seconds = time.perf_counter() - started

    def configuration(self):
        return {
            "version": VERSION,
            "classification": "supplied-physics public-observation competence reference",
            "learned": False,
            "privileged_live_state": False,
            "supplied_knowledge": "Full copied Reacher MjModel geometry, masses, actuators, joint limits and dynamics",
            "initial_velocity": "exactly zero, including target coordinates",
            "initial_angle_branch": "principal atan2; initial winding is unidentifiable",
            "measurement_velocity": "wrapped shortest displacement / actual time since previous valid packet",
            "measurement_position": "nearest nominally predicted angle branch, including soft-limit crossings",
            "missing_angles": "finite zero placeholders required; never use unobserved angles",
            "target": "static public target; target velocity zero",
            "disturbance_model": "nominal zero disturbance; realized noise unavailable",
            "command": "clip [-1,1], then float32 quantization before nominal native propagation",
            "frame_skip": self.frame_skip,
            "native_timestep": self.native_timestep,
            "decision_dt": self.dt,
            "integrator": int(self._model.opt.integrator),
            "gravity": self._model.opt.gravity.tolist(),
            "measurement_limit": "interval-average velocity; displacement exceeding pi can alias",
            "run_status_authority": "enclosing protocol and execution receipts",
        }

    def estimate(self):
        """Independent float64 qpos[4], qvel[4] arrays; no live-state oracle."""
        if self._failed:
            raise RuntimeError("Failed observer is terminal")
        return self._data.qpos.copy(), self._data.qvel.copy()

    def update(self, issued_command, next_packet):
        """Consume a command actually issued from the previous decision state.

        Invalid public input is rejected before changing nominal state. A
        native propagation failure is terminal; counts distinguish requested
        from successfully completed native work.
        """
        if self._failed:
            raise RuntimeError("Failed observer is terminal")
        started = time.perf_counter()
        packet = _packet(next_packet)
        command = np.clip(_real(issued_command, (2,), "issued_command"), -1., 1.).astype(np.float32)
        if not np.array_equal(packet[4:6], self._target):
            raise ValueError("Static target changed within episode")
        next_step = self._step_index + 1
        elapsed = (next_step - self._last_valid_step) * self.dt
        expected_age = 0. if packet[6] else elapsed
        if not np.isclose(packet[7], expected_age, atol=1e-6, rtol=1e-6):
            raise ValueError("Packet age does not match elapsed control time")
        self._native_transition_attempts += 1
        self._native_substeps_requested += self.frame_skip
        try:
            _step(self._model, self._data, command.astype(np.float64), self.frame_skip)
            self._native_transitions += 1
            self._native_substeps += self.frame_skip
            if packet[6]:
                measured = _angles(packet)
                velocity = _wrap(measured - self._last_valid_angles) / elapsed
                if not np.isfinite(velocity).all():
                    raise FloatingPointError("Nonfinite measured-interval velocity")
                self._data.qpos[:2] += _wrap(measured - self._data.qpos[:2])
                self._data.qpos[2:] = self._target
                self._data.qvel[:2] = velocity
                self._data.qvel[2:] = 0.
                mujoco.mj_forward(self._model, self._data)
                self._reanchor_forward_calls += 1
                self._last_valid_angles = measured.copy()
                self._last_valid_step = next_step
                self._valid_measurements = (self._valid_measurements + [(next_step, packet.copy())])[-2:]
                self._last_velocity_interval = elapsed
            self._step_index = next_step
            self._last_command, self._last_packet = command.copy(), packet.copy()
            return self.estimate()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._update_wall_seconds += time.perf_counter() - started

    def snapshot(self):
        """Detached public history, estimated state and charged nominal work.

        This snapshot is audit material, not a resumable native-state payload.
        No true simulator trajectory, hidden velocity or realized noise occurs.
        """
        return {
            "configuration": self.configuration(),
            "step_index": self._step_index,
            "elapsed_seconds": self._step_index * self.dt,
            "last_valid_step": self._last_valid_step,
            "last_velocity_interval_seconds": self._last_velocity_interval,
            "valid_measurements": [{"step": step, "time": step * self.dt, "packet": packet.copy()}
                                   for step, packet in self._valid_measurements],
            "last_packet": self._last_packet.copy(),
            "last_issued_command": None if self._last_command is None else self._last_command.copy(),
            "qpos_estimate": self._data.qpos.copy(),
            "qvel_estimate": self._data.qvel.copy(),
            "failed": self._failed,
            "costs": {
                "setup_wall_seconds": self._setup_wall_seconds,
                "update_wall_seconds": self._update_wall_seconds,
                "native_transition_attempts": self._native_transition_attempts,
                "native_substeps_requested": self._native_substeps_requested,
                "native_transitions_completed": self._native_transitions,
                "native_substeps_completed": self._native_substeps,
                "setup_forward_calls": 1,
                "measurement_reanchor_forward_calls": self._reanchor_forward_calls,
                "postconstraint_refresh_calls_completed": self._native_transitions,
                "counts_exclude_forward_and_postconstraint_from_native_substeps": True,
            },
        }


class KinematicMPC:
    """Nominal PhysicsMPC from the public-only two-measurement state estimate.

    Candidate banks are supplied by the caller; no proposal RNG or candidate
    selection policy is introduced. Costs are positive summed native distance
    plus command-square costs, with the existing PhysicsMPC earliest tie rule.
    Planning never updates the observer or implies that an action was issued.
    """

    def __init__(self, model, initial_packet, *, frame_skip=2):
        started = time.perf_counter()
        self.observer = KinematicObserver(model, initial_packet, frame_skip=frame_skip)
        self._planner = PhysicsMPC(self.observer._model, frame_skip=frame_skip)
        self._calls = self._candidate_evaluations = self._native_transitions = self._native_substeps = 0
        self._requested_transitions = self._requested_substeps = 0
        self._failed = False
        self._planning_wall_seconds = 0.
        self._setup_wall_seconds = time.perf_counter() - started

    def estimate(self):
        return self.observer.estimate()

    def update(self, issued_command, next_packet):
        if self._failed:
            raise RuntimeError("Failed planner is terminal")
        return self.observer.update(issued_command, next_packet)

    def plan(self, candidates):
        if self._failed:
            raise RuntimeError("Failed planner is terminal")
        started = time.perf_counter()
        raw = np.asarray(candidates)
        if raw.ndim != 3 or raw.shape[-1] != 2 or min(raw.shape[:2]) < 1:
            raise ValueError("candidates must have nonempty shape [K,H,2]")
        bank = _real(raw, raw.shape, "candidates")
        qpos, qvel = self.observer.estimate()
        count, horizon = bank.shape[:2]
        transitions = count * horizon
        self._requested_transitions += transitions
        self._requested_substeps += transitions * self.observer.frame_skip
        try:
            result = self._planner.plan(qpos, qvel, bank)
            self._calls += 1
            self._candidate_evaluations += count
            self._native_transitions += transitions
            self._native_substeps += transitions * self.observer.frame_skip
            return result
        except BaseException:
            self._failed = True
            raise
        finally:
            self._planning_wall_seconds += time.perf_counter() - started

    def snapshot(self):
        return {
            "observer": self.observer.snapshot(),
            "failed": self._failed or self.observer._failed,
            "planner": "Existing PhysicsMPC; supplied physics and public-derived estimate, not true state",
            "reward_weights": {"distance": 1., "control": 1.},
            "costs": {
                "setup_wall_seconds_including_observer": self._setup_wall_seconds,
                "planning_wall_seconds": self._planning_wall_seconds,
                "successful_plan_calls": self._calls,
                "candidate_evaluations_completed": self._candidate_evaluations,
                "candidate_reset_forward_calls_completed": self._candidate_evaluations,
                "native_transitions_requested": self._requested_transitions,
                "native_substeps_requested": self._requested_substeps,
                "native_transitions_completed": self._native_transitions,
                "native_substeps_completed": self._native_substeps,
                "postconstraint_refresh_calls_completed": self._native_transitions,
                "setup_and_observer_costs_overlap_do_not_sum": True,
                "failed_plan_partial_work_is_bounded_by_requested_counts": True,
            },
        }
