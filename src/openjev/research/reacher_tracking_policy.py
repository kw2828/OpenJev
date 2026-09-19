"""Causal controller roles for the separate changing-dynamics qualification.

No environment, reward, schedule or RNG enters a policy. Privileged inputs are
accepted only by explicitly named reference roles. Public angle reconstruction
is identical in all roles, including the two references' diagnostic estimates.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from openjev.research.reacher_physics_control import _integer, _model, _real
from openjev.research.reacher_tracking_control import GainPlanningBank
from openjev.research.reacher_tracking_identification import (
    TrackingAngleObserver,
    WindowedGainIdentifier,
    _visible,
)

ARMS = ("nominal", "adaptive", "frozen", "public_gain", "true_state", "zero")
VERSION = "reacher-tracking-policy-v1"


class TrackingPolicy:
    """One decision followed by one acknowledgement of the issued action.

    The adaptive/frozen roles also maintain the same public observer inside
    their identifier; this duplicate measurement work is paid and checked for
    exact agreement. At freeze_after completed transitions, the frozen role
    stops updating the identifier and retains that gain. Its public observer
    continues. A private candidate forecast never changes either observer.
    """

    def __init__(self, model, initial_packet, *, arm, steps, gain_grid, window,
                 freeze_after, noise_std, planning_horizon, action_block):
        started = time.perf_counter()
        if arm not in ARMS:
            raise ValueError("Unknown tracking controller role")
        nominal = _model(model)
        self.arm = arm
        self.steps = _integer(steps, "steps", 1)
        self.freeze_after = _integer(freeze_after, "freeze_after", 1)
        if self.freeze_after > self.steps:
            raise ValueError("freeze_after exceeds episode horizon")
        self._packet = _visible(initial_packet).astype(np.float32)
        self._observer = TrackingAngleObserver(self._packet, dt=float(nominal.opt.timestep)*2)
        self._identifier = (WindowedGainIdentifier(nominal, self._packet, gains=gain_grid,
            window=window, prior_gain=1., frame_skip=2) if arm in ("adaptive", "frozen") else None)
        self._bank = (GainPlanningBank(nominal, allowed_gains=gain_grid, steps=self.steps,
            noise_std=noise_std, frame_skip=2, planning_horizon=planning_horizon,
            action_block=action_block) if arm != "zero" else None)
        self._step = 0
        self._pending = None
        self._gain = 1.
        self._failed = False
        self._last_decision = None
        self._last_observation = None
        self._decisions = self._observations = self._identifications = 0
        self._decision_seconds = self._observation_seconds = 0.
        self._setup_seconds = time.perf_counter()-started

    def decide(self, inputs, *, current_gain=None, true_state=None, deadline=float("inf")):
        if self._failed or self._pending is not None or self._step >= self.steps:
            raise RuntimeError("Policy is failed, awaiting its issued action, or terminal")
        public_oracle = self.arm in ("public_gain", "true_state")
        if (current_gain is not None) != public_oracle:
            raise ValueError("Only explicit gain-reference roles receive current_gain")
        if (true_state is not None) != (self.arm == "true_state"):
            raise ValueError("Only the true_state reference receives privileged state")
        qpos, qvel = self._observer.estimate()
        public_qpos, public_qvel = qpos.copy(), qvel.copy()
        if self.arm == "true_state":
            if not isinstance(true_state, tuple) or len(true_state) != 2:
                raise ValueError("true_state must be explicit (qpos[4],qvel[4]) arrays")
            qpos = _real(true_state[0], (4,), "true qpos")
            qvel = _real(true_state[1], (4,), "true qvel")
            if not np.array_equal(qpos[2:].astype(np.float32), self._packet[4:6]) or np.any(qvel[2:] != 0):
                raise ValueError("Privileged state must share the current public target")
            # Keep target precision identical across every role; only arm state is privileged.
            qpos[2:] = self._packet[4:6].astype(np.float64)
        gain = current_gain if public_oracle else self._gain
        started = time.perf_counter()
        try:
            result = journal = bank_snapshot = None
            if self.arm == "zero":
                action = np.zeros(2, dtype=np.float32)
            else:
                result, journal, bank_snapshot = self._bank.plan(
                    qpos[None], qvel[None], self._packet[None, 4:6], gain, inputs,
                    step=self._step, deadline=deadline)
                action = result.selected_actions[0].copy()
            trace = {"step": self._step, "arm": self.arm, "public_packet": self._packet.copy(),
                "public_qpos": public_qpos, "public_qvel": public_qvel,
                "root_qpos": qpos.copy(), "root_qvel": qvel.copy(),
                "planning_gain": float(gain), "action": action.copy(),
                "search": result, "journal": journal, "bank_snapshot": bank_snapshot}
            self._pending = action.copy()
            self._last_decision = {key: copy.deepcopy(value) for key, value in trace.items()
                                   if key not in ("search", "journal", "bank_snapshot")}
            self._decisions += 1
            return action, trace
        except BaseException:
            self._failed = True
            raise
        finally:
            self._decision_seconds += time.perf_counter()-started

    def observe(self, issued_command, next_packet):
        if self._failed or self._pending is None:
            raise RuntimeError("Observation requires one pending actual action")
        command = _real(issued_command, (2,), "issued command").astype(np.float32)
        packet = _visible(next_packet).astype(np.float32)
        if not np.array_equal(command, self._pending):
            raise ValueError("Acknowledged command differs from the selected issued action")
        started = time.perf_counter()
        try:
            qpos, qvel = self._observer.update(packet)
            update_id = self._identifier is not None and (self.arm == "adaptive" or self._step < self.freeze_after)
            trace = None
            if update_id:
                identified_qpos, identified_qvel, self._gain = self._identifier.update(command, packet)
                if not np.array_equal(identified_qpos, qpos) or not np.array_equal(identified_qvel, qvel):
                    raise RuntimeError("Identifier and common public observer differ")
                self._identifications += 1
                trace = self._identifier.snapshot()["last_trace"]
            self._step += 1
            self._observations += 1
            self._packet = packet.copy()
            self._pending = None
            self._last_observation = {"step": self._step, "public_qpos": qpos.copy(),
                "public_qvel": qvel.copy(), "identifier_updated": bool(update_id),
                "identified_gain": float(self._gain) if self._identifier is not None else None,
                "identifier_trace": trace}
            return copy.deepcopy(self._last_observation)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._observation_seconds += time.perf_counter()-started

    def snapshot(self):
        return {"version": VERSION, "arm": self.arm, "steps": self.steps,
            "step": self._step, "freeze_after": self.freeze_after, "failed": self._failed,
            "pending_command": None if self._pending is None else self._pending.copy(),
            "observer": self._observer.snapshot(), "gain_estimate": self._gain,
            "identifier": None if self._identifier is None else self._identifier.snapshot(),
            "planner": None if self._bank is None else self._bank.snapshot(),
            "last_decision": copy.deepcopy(self._last_decision),
            "last_observation": copy.deepcopy(self._last_observation),
            "costs": {"setup_wall_seconds": self._setup_seconds, "decision_wall_seconds": self._decision_seconds,
                "observation_wall_seconds": self._observation_seconds, "decisions_completed": self._decisions,
                "observations_completed": self._observations, "identifier_updates_completed": self._identifications,
                "diagnostic_public_observer_updated_for_true_state": self.arm == "true_state",
                "identifier_has_duplicate_paid_public_observer": self._identifier is not None}}
