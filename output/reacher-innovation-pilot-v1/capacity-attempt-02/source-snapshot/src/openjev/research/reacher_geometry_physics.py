"""Nominal supplied-physics CEM with the frozen approximate geometry score.

Only copied MjModel parameters and explicit qpos/qvel/public-target arrays enter
this adapter. The caller owns their information provenance. This is nominal
qpos/qvel initialization, with reset solver/integration state and zero applied
disturbance, not continuation of a complete native integration state or a
stochastic optimal-control oracle. Expected actuator cost is charged once.

All innovations are supplied. Search is the unchanged four-call CEM256 kernel.
The selected one-action advance starts again at the supplied root; its returned
state is an auditable prediction, never authority to overwrite an observer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import mujoco
import numpy as np
import torch

from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_geometry_reward import (
    geometry_reward_components,
    geometry_reward_configuration,
)
from openjev.research.reacher_physics_control import _integer, _model, _scalar

VERSION = "reacher-nominal-geometry-cem-v1"
_COUNTS = (
    "candidate_sequences_requested", "candidate_sequences_initialized",
    "candidate_sequences_native_completed", "candidate_sequences_scored",
    "native_transitions_attempted", "native_transitions_completed",
    "native_substeps_attempted", "native_substeps_completed",
    "reset_calls_attempted", "reset_calls_completed", "forward_calls_attempted", "forward_calls_completed",
    "postconstraint_calls_attempted", "postconstraint_calls_completed",
    "geometry_calls_attempted", "geometry_calls_completed", "geometry_samples_attempted", "geometry_samples_completed",
)
_COMPONENTS = {"joint_angles": 2, "pair_norms": 2, "fingertip": 2,
               "distance": None, "action_cost": None, "reward": None}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash_array(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def _frozen(value):
    result = np.asarray(value)
    return np.frombuffer(result.tobytes(order="C"), dtype=result.dtype).reshape(result.shape)


def _model_hash(model):
    payload = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=payload)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _deadline(value):
    _require(type(value) in (int, float) and not math.isnan(value), "Explicit monotonic deadline")
    if time.monotonic() >= value:
        raise TimeoutError("Nominal geometry physics deadline exceeded")


def _array(value, shape, dtype, name):
    _require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
             and np.isfinite(value).all(), f"{name} must be finite {np.dtype(dtype).name}{shape}")
    return value.copy()


class PhysicsGeometryCEM:
    """Private nominal simulator; complete success/failure accounting via snapshot.

    Inputs use a leading case dimension, including for a single root. Public
    targets are explicitly float32, matching learned scorer inputs; native
    qpos/qvel remain float64. No RNG is created and no environment/MjData is
    accepted. A failure after an operation starts is terminal; its partial
    arrays, completion masks and attempted/completed counters remain available.
    Pure input validation failures occur before work and leave the adapter usable.
    """

    def __init__(self, model, *, frame_skip=2, noise_std=.05, steps=50, planning_horizon=12, action_block=3):
        start = time.perf_counter()
        self._model = copy.copy(_model(model))
        self.frame_skip = _integer(frame_skip, "frame_skip", 1)
        self.noise_std = _scalar(noise_std, "noise_std")
        self.steps = _integer(steps, "steps", 1)
        self.planning_horizon = _integer(planning_horizon, "planning_horizon", 1)
        self.action_block = _integer(action_block, "action_block", 1)
        _require(self.planning_horizon <= self.steps and self.action_block <= self.planning_horizon,
                 "Action block <= planning horizon <= episode steps")
        self.dt = _scalar(float(self._model.opt.timestep) * self.frame_skip, "decision timestep", positive=True)
        self._validate_geometry()
        self._data = mujoco.MjData(self._model)
        self._model_sha = _model_hash(self._model)
        self._failed = False
        self._journal = None
        self._lifetime = {name: 0 for name in _COUNTS}
        self._lifetime.update(operations_completed=0, operations_failed=0, operation_wall_seconds=0.)
        self._setup_seconds = time.perf_counter() - start

    def _validate_geometry(self):
        def body(name):
            index = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            _require(index >= 0, "Frozen planar Reacher body layout")
            return index
        first, second, tip, target = (body(name) for name in ("body0", "body1", "fingertip", "target"))
        for index, expected in ((first, [0, 0, .01]), (second, [.10, 0, 0]), (tip, [.11, 0, 0])):
            _require(np.array_equal(self._model.body_pos[index], expected)
                     and np.array_equal(self._model.body_quat[index], [1, 0, 0, 0]), "Frozen planar geometry offsets/orientation")
        _require(self._model.body_parentid[second] == first and self._model.body_parentid[tip] == second,
                 "Frozen planar link parentage")
        for coordinate, (name, index) in enumerate((("joint0", first), ("joint1", second))):
            joint = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            _require(joint >= 0 and self._model.jnt_bodyid[joint] == index
                     and self._model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                     and np.array_equal(self._model.jnt_axis[joint], [0, 0, 1])
                     and np.array_equal(self._model.jnt_pos[joint], [0, 0, 0])
                     and self._model.jnt_qposadr[joint] == coordinate
                     and self._model.jnt_dofadr[joint] == coordinate, "Planar hinge geometry")
        for axis, name in enumerate(("target_x", "target_y")):
            joint = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            expected_axis = [1, 0, 0] if axis == 0 else [0, 1, 0]
            _require(joint >= 0 and self._model.jnt_bodyid[joint] == target
                     and self._model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
                     and np.array_equal(self._model.jnt_axis[joint], expected_axis)
                     and self._model.jnt_qposadr[joint] == 2 + axis
                     and self._model.qpos0[2 + axis] == self._model.body_pos[target, axis], "Public target coordinates denote XY")

    def configuration(self):
        return {"version": VERSION, "classification": "supplied-physics nominal competence reference",
            "model_binary_sha256": self._model_sha, "model_copied": True, "native_timestep": float(self._model.opt.timestep),
            "frame_skip": self.frame_skip, "decision_dt": self.dt, "steps": self.steps,
            "planning_horizon": self.planning_horizon, "action_block": self.action_block,
            "planner": "cem256", "callback_sizes": [64, 64, 64, 64], "warm_start": False,
            "reward_clip": [-2.5, 0.], "score_sum": "sequential float32, per-step clipping",
            "geometry": geometry_reward_configuration(self.noise_std), "rng_draws": 0,
            "root_semantics": "Copied qpos/qvel; static explicit public target; mj_resetData then mj_forward; no full integration-state continuation.",
            "transition_semantics": "Nominal issued command held across frame_skip substeps, zero realized noise, followed by mj_rnePostConstraint.",
            "angle_precision": "cos/sin of nominal float64 qpos, quantized to float32 before the shared geometry scorer",
            "information_boundary": "Caller declares known-state, filtered or public-kinematic root provenance; this adapter cannot infer it.",
            "selected_advance": "Separate one-action prediction from the original root; not permission to replace an observer with privileged state.",
            "limits": "Equal CEM proposal budgets do not imply matched wall time or stochastic optimal MPC; FK scoring is not exact native reward.",
            "run_status_authority": "enclosing protocol and execution receipts"}

    def _roots(self, qpos, qvel, public_target, step):
        if self._failed:
            raise RuntimeError("Failed nominal physics adapter is terminal; inspect snapshot")
        _require(type(step) is int and 0 <= step < self.steps, "Decision step before terminal boundary")
        _require(isinstance(qpos, np.ndarray) and qpos.ndim == 2 and qpos.shape[1] == 4 and len(qpos) > 0,
                 "Explicit qpos batch[N,4]")
        n = len(qpos)
        position = _array(qpos, (n, 4), np.float64, "qpos")
        velocity = _array(qvel, (n, 4), np.float64, "qvel")
        target = _array(public_target, (n, 2), np.float32, "public_target")
        _require((velocity[:, 2:] == 0).all(), "Static target velocity must be zero")
        _require(np.array_equal(position[:, 2:].astype(np.float32), target), "Public target must match supplied root goal after float32 quantization")
        return {"qpos": position, "qvel": velocity, "public_target": target}

    def _begin(self, operation, roots, step, deadline, started):
        root_hashes = {name: _hash_array(value) for name, value in roots.items()}
        self._journal = {"operation": operation, "status": "running", "step": step,
            "configuration": self.configuration(), "roots": roots, "root_sha256": root_hashes,
            "banks": [], "selected": None, "search": None, "started": started, "wall_seconds": 0.}
        _deadline(deadline)
        _require(_model_hash(self._model) == self._model_sha, "Private model changed")

    def _counter(self, record, name, value=1):
        record["metadata"][name] += value
        self._lifetime[name] += value

    def _bank(self, roots, bank, step, deadline, *, selected=False):
        n, k, h, _ = bank.shape
        arrays = {"commands": bank.copy(), "predicted_angles": np.zeros((n, k, h, 4), np.float32),
            "qpos": np.zeros((n, k, h + 1, 4), np.float64), "qvel": np.zeros((n, k, h + 1, 4), np.float64),
            "initialized": np.zeros((n, k), bool), "native_completed": np.zeros((n, k, h), bool),
            "scored": np.zeros((n, k, h), bool), "scores": np.zeros((n, k), np.float32),
            "native_substeps_completed": np.zeros((n, k, h), np.int64)}
        arrays.update({"geometry_" + name: np.zeros((n, k, h) + (() if size is None else (size,)), np.float32)
                       for name, size in _COMPONENTS.items()})
        metadata = {name: 0 for name in _COUNTS}
        metadata.update(bank_sha256=_hash_array(bank), root_sha256=dict(self._journal["root_sha256"]),
            purpose="selected_root_advance" if selected else "candidate_scoring",
            candidate_start=None if selected else sum(row["arrays"]["commands"].shape[1] for row in self._journal["banks"]),
            bank_shape=list(bank.shape), cursor=None, native_seconds=0., geometry_seconds=0., wall_seconds=0.,
            sum_offsets_completed=0, status="running")
        record = {"arrays": arrays, "metadata": metadata}
        if selected:
            self._journal["selected"] = record
        else:
            self._journal["banks"].append(record)
        self._counter(record, "candidate_sequences_requested", n * k)
        started = time.perf_counter()
        data = self._data
        try:
            tick = time.perf_counter()
            try:
                for i in range(n):
                    for j in range(k):
                        metadata["cursor"] = {"case": i, "candidate": j, "offset": -1}
                        _deadline(deadline)
                        self._counter(record, "reset_calls_attempted")
                        mujoco.mj_resetData(self._model, data)
                        self._counter(record, "reset_calls_completed")
                        data.qpos[:], data.qvel[:], data.time = roots["qpos"][i], roots["qvel"][i], step * self.dt
                        self._counter(record, "forward_calls_attempted")
                        mujoco.mj_forward(self._model, data)
                        self._counter(record, "forward_calls_completed")
                        arrays["qpos"][i, j, 0], arrays["qvel"][i, j, 0] = data.qpos, data.qvel
                        arrays["initialized"][i, j] = True
                        self._counter(record, "candidate_sequences_initialized")
                        for offset in range(h):
                            metadata["cursor"] = {"case": i, "candidate": j, "offset": offset}
                            _deadline(deadline)
                            self._counter(record, "native_transitions_attempted")
                            data.ctrl[:] = bank[i, j, offset]
                            for _ in range(self.frame_skip):
                                _deadline(deadline)
                                self._counter(record, "native_substeps_attempted")
                                mujoco.mj_step(self._model, data)
                                self._counter(record, "native_substeps_completed")
                                arrays["native_substeps_completed"][i, j, offset] += 1
                            self._counter(record, "postconstraint_calls_attempted")
                            mujoco.mj_rnePostConstraint(self._model, data)
                            self._counter(record, "postconstraint_calls_completed")
                            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                                raise FloatingPointError("Nonfinite nominal physics")
                            arrays["qpos"][i, j, offset + 1], arrays["qvel"][i, j, offset + 1] = data.qpos, data.qvel
                            arrays["native_completed"][i, j, offset] = True
                            self._counter(record, "native_transitions_completed")
                            arrays["predicted_angles"][i, j, offset] = np.concatenate((np.cos(data.qpos[:2]), np.sin(data.qpos[:2]))).astype(np.float32)
                        self._counter(record, "candidate_sequences_native_completed")
            finally:
                metadata["native_seconds"] += time.perf_counter() - tick
            _deadline(deadline)
            self._counter(record, "geometry_calls_attempted")
            self._counter(record, "geometry_samples_attempted", n * k * h)
            target = np.broadcast_to(roots["public_target"][:, None, None], (n, k, h, 2)).copy()
            tick = time.perf_counter()
            try:
                with torch.no_grad():
                    components = geometry_reward_components(torch.from_numpy(arrays["predicted_angles"]),
                        torch.from_numpy(target), torch.from_numpy(bank.copy()), self.noise_std)
            finally:
                metadata["geometry_seconds"] += time.perf_counter() - tick
            self._counter(record, "geometry_calls_completed")
            self._counter(record, "geometry_samples_completed", n * k * h)
            for name in _COMPONENTS:
                arrays["geometry_" + name][:] = getattr(components, name).numpy()
            arrays["scored"][:] = True
            for offset in range(h):
                _deadline(deadline)
                arrays["scores"] += np.clip(arrays["geometry_reward"][:, :, offset], -2.5, 0.)
                metadata["sum_offsets_completed"] += 1
            self._counter(record, "candidate_sequences_scored", n * k)
            metadata["status"] = "completed"
            return arrays["scores"].copy()
        except BaseException:
            metadata["status"] = "failed"
            record["failure_native_state"] = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(), "time": float(data.time)}
            raise
        finally:
            metadata["wall_seconds"] = time.perf_counter() - started

    def _finish(self, roots, deadline):
        _deadline(deadline)
        _require(all(_hash_array(value) == self._journal["root_sha256"][name] for name, value in roots.items()), "Copied roots mutated")
        _require(_model_hash(self._model) == self._model_sha, "Private physics model changed")
        self._journal["status"] = "completed"
        self._lifetime["operations_completed"] += 1

    def _failed_operation(self, error):
        self._failed = True
        self._lifetime["operations_failed"] += 1
        self._journal["status"] = "failed"
        self._journal["error"] = {"type": type(error).__name__, "message": str(error)}
        if self._journal["search"] is not None and self._journal["search"]["status"] == "running":
            self._journal["search"]["status"] = "failed"

    def _timed_finish(self):
        self._journal["wall_seconds"] = time.perf_counter() - self._journal["started"]
        self._lifetime["operation_wall_seconds"] += self._journal["wall_seconds"]

    def score_bank(self, qpos, qvel, public_target, bank, *, step, deadline=float("inf")):
        """Return (float32 scores[N,K], persistable diagnostics) for explicit commands."""
        started = time.perf_counter()
        roots = self._roots(qpos, qvel, public_target, step)
        _require(isinstance(bank, np.ndarray) and bank.ndim == 4 and bank.shape[0] == len(qpos)
                 and 0 < bank.shape[1] <= 256 and bank.shape[-1] == 2
                 and 0 < bank.shape[2] <= min(self.planning_horizon, self.steps - step), "Bank case/candidate/horizon boundary")
        commands = _array(bank, bank.shape, np.float32, "bank")
        _require((np.abs(commands) <= 1).all(), "Issued commands already in [-1,1]; no candidate clipping")
        try:
            self._begin("score_bank", roots, step, deadline, started)
            scores = self._bank(roots, commands, step, deadline)
            self._finish(roots, deadline)
        except BaseException as error:
            self._failed_operation(error)
            raise
        finally:
            self._timed_finish()
        return _frozen(scores), self.snapshot()["last_operation"]

    def plan(self, qpos, qvel, public_target, inputs, *, step, deadline=float("inf")):
        """Return (unchanged SearchResult, diagnostics including selected root advance).

        Every candidate is evaluated from the original supplied root. The paid
        final mean, global-best tie order and exact proposal accounting are
        provided unchanged by reacher_adaptive_search.search. Selection does
        not assimilate information or install candidate terminal state anywhere.
        """
        started = time.perf_counter()
        roots = self._roots(qpos, qvel, public_target, step)
        chunks = (self.planning_horizon + self.action_block - 1) // self.action_block
        _require(isinstance(inputs, SearchInputs) and inputs.initial.shape == (len(qpos), 64, chunks, 2), "Complete aligned SearchInputs")
        try:
            self._begin("cem256", roots, step, deadline, started)
            self._journal["search"] = {"status": "running", "input_identities": list(inputs.identities())}
            result = search("cem256", inputs, lambda bank: self._bank(roots, bank, step, deadline),
                step=step, steps=self.steps, planning_horizon=self.planning_horizon, action_block=self.action_block)
            self._journal["search"] = {"status": "completed", "candidate_evaluations": result.candidate_evaluations,
                "imagined_transitions": result.imagined_transitions, "input_identities": list(result.input_identities),
                "selected_ids": result.selected_ids.copy(), "selected_actions": result.selected_actions.copy(),
                "selected_sequences": result.selected_sequences.copy(), "candidate_ids": list(result.candidate_ids)}
            _require([bank["arrays"]["commands"].shape[1] for bank in self._journal["banks"]] == [64] * 4, "Exactly four paid CEM callbacks")
            self._bank(roots, result.selected_actions[:, None, None], step, deadline, selected=True)
            self._finish(roots, deadline)
        except BaseException as error:
            self._failed_operation(error)
            raise
        finally:
            self._timed_finish()
        return result, self.snapshot()["last_operation"]

    def snapshot(self):
        """Detached arrays+metadata, including failed native prefixes; no rerun.

        Uncomputed cells are zeros with explicit initialized/native_completed/
        scored masks. Completed substeps mean native calls returned; a failed
        substep may have changed scratch state, saved separately without calling
        it a completed transition. Root arrays and model parameters stay intact.
        Setup and operation times are separate, and snapshot copying itself is
        caller work rather than hidden inside these recorded operation timings.
        """
        return copy.deepcopy({"configuration": self.configuration(), "setup_seconds": self._setup_seconds,
            "failed": self._failed, "lifetime": self._lifetime, "last_operation": self._journal})
