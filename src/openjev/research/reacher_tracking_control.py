"""Explicit-gain planning bank for the separate Reacher tracking task.

No plant, estimator, target schedule, RNG or environment enters this interface.
The caller authenticates the nominal model and the provenance of every root and
gain. A true-gain caller is privileged; an estimated-gain caller is not made
privileged by this shared planning implementation.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM, _deadline, _model_hash
from openjev.research.reacher_physics_control import _integer, _model, _real, _scalar

VERSION = 'reacher-tracking-gain-planning-bank-v1'


class GainPlanningBank:
    """One immutable physics adapter per declared gain, with paid eager setup.

    ``steps`` is explicit (the prospective tracking task may use 200). Scientific
    H12/block3 uses the unchanged CEM256 kernel; smaller explicit horizons are
    supported for engineering checks. Gain lookup is exact, with no interpolation
    or rounding. Each call predicts one root only. The selected action is paid
    again from that root, and its state never replaces a public observer.

    A native/planning failure is terminal for the bank, including attempts to
    switch to another gain. Pure input rejection before planner dispatch leaves
    the bank usable. Full failed native evidence remains in snapshot().
    """

    def __init__(self, nominal_model, *, allowed_gains, steps, noise_std,
                 frame_skip=2, planning_horizon=12, action_block=3):
        started = time.perf_counter()
        nominal = _model(nominal_model)
        gear = np.zeros((2, 6), np.float64)
        gear[:, 0] = 200.
        if (not np.array_equal(nominal.actuator_gear, gear)
                or not np.array_equal(nominal.dof_damping, [1., 1., 0., 0.])):
            raise ValueError('supply an independent nominal gear200/fixed-damping model')
        values = np.asarray(allowed_gains)
        if values.ndim != 1 or len(values) < 1:
            raise ValueError('allowed_gains must be a nonempty vector')
        values = _real(values, values.shape, 'allowed_gains')
        if (np.any(values <= 0) or np.any(np.diff(values) <= 0)
                or np.any(values > np.finfo(np.float64).max / 200.)):
            raise ValueError('allowed_gains must be strictly increasing, positive, finite gears')
        self._gains = np.frombuffer(values.tobytes(), dtype=np.float64)
        self.steps = _integer(steps, 'steps', 1)
        self.frame_skip = _integer(frame_skip, 'frame_skip', 1)
        self.planning_horizon = _integer(planning_horizon, 'planning_horizon', 1)
        self.action_block = _integer(action_block, 'action_block', 1)
        self.noise_std = _scalar(noise_std, 'noise_std')
        if not self.action_block <= self.planning_horizon <= self.steps:
            raise ValueError('action block <= planning horizon <= steps')
        self._nominal_sha256 = _model_hash(nominal)
        self._planners, self._member_summaries = [], []
        for value in self._gains:
            specific = copy.copy(nominal)
            specific.actuator_gear[:] = gear * value
            planner = PhysicsGeometryCEM(specific, frame_skip=self.frame_skip,
                noise_std=self.noise_std, steps=self.steps,
                planning_horizon=self.planning_horizon, action_block=self.action_block)
            self._planners.append(planner)
            self._member_summaries.append(self._compact(planner.snapshot()))
        self._failed, self._last, self._last_journal = False, None, None
        self._calls = self._completed = self._rejected = self._failures = 0
        self._plan_seconds = 0.
        self._setup_seconds = time.perf_counter() - started

    @staticmethod
    def _compact(snapshot):
        return {key: snapshot[key] for key in ('configuration', 'setup_seconds', 'failed', 'lifetime')}

    def configuration(self):
        return {'version': VERSION, 'allowed_gains': self._gains.tolist(),
                'gain_lookup': 'exact float64 equality, no rounding/interpolation',
                'nominal_model_binary_sha256': self._nominal_sha256,
                'member_model_binary_sha256': [row['configuration']['model_binary_sha256']
                                                for row in self._member_summaries],
                'steps': self.steps, 'planning_horizon': self.planning_horizon,
                'action_block': self.action_block, 'frame_skip': self.frame_skip,
                'noise_std': self.noise_std, 'planner': 'cem256', 'cases_per_call': 1,
                'rng_draws': 0, 'estimator_updates': 0,
                'model_setup': 'one temporary gain-specific copy plus one retained frozen-adapter copy per gain; all eager',
                'target_semantics': 'current public target only, held static inside this decision planning horizon',
                'gain_semantics': 'explicit current estimate/oracle value, held constant inside this decision planning horizon',
                'root_semantics': 'caller qpos/qvel; reset nominal solver state, not full plant integration-state continuation',
                'scoring': 'unchanged approximate geometry and expected normalized applied-action cost; zero realized rollout noise',
                'source_boundary': 'caller authenticates independent nominal XML and state/gain information provenance',
                'cost_scope': 'bank setup plus inclusive plan-call wall; nested adapter times overlap, do not add them again',
                'snapshot_scope': 'snapshot calls outside plan are caller copy/bookkeeping work',
                'run_status_authority': 'enclosing protocol and execution receipts'}

    def _validate(self, qpos, qvel, public_target, gain, inputs, step):
        value = _scalar(gain, 'gain', positive=True)
        matches = np.flatnonzero(self._gains == value)
        if len(matches) != 1:
            raise ValueError('gain must equal a declared allowed value exactly')
        if type(step) is not int or not 0 <= step < self.steps:
            raise ValueError('step must precede the explicit terminal boundary')
        for array, shape, dtype, name in ((qpos, (1, 4), np.float64, 'qpos'),
                (qvel, (1, 4), np.float64, 'qvel'),
                (public_target, (1, 2), np.float32, 'public_target')):
            if not (isinstance(array, np.ndarray) and array.shape == shape
                    and array.dtype == dtype and np.isfinite(array).all()):
                raise ValueError(f'{name} must be finite {dtype}{shape}')
        if np.any(qvel[:, 2:] != 0) or not np.array_equal(qpos[:, 2:].astype(np.float32), public_target):
            raise ValueError('root must contain the same static current public target')
        chunks = (self.planning_horizon + self.action_block - 1) // self.action_block
        if not isinstance(inputs, SearchInputs) or inputs.initial.shape != (1, 64, chunks, 2):
            raise ValueError('aligned single-case SearchInputs required')
        return int(matches[0]), value

    def _refresh(self, index):
        # Only the selected member is copied. Other adapters retain their cached
        # compact summaries, avoiding full-journal copies across the gain bank.
        snapshot = self._planners[index].snapshot()
        self._member_summaries[index] = self._compact(snapshot)
        self._last_journal = snapshot['last_operation']

    def plan(self, qpos, qvel, public_target, gain, inputs, *, step, deadline):
        """Return (SearchResult, unchanged native journal, compact bank snapshot).

        No stateful observer is accepted or updated. The complete native journal
        includes all four CEM banks and the selected one-action root restart.
        Copying that journal, selected-member accounting and the returned compact
        snapshot are charged in this bank's inclusive call wall time.
        """
        if self._failed:
            raise RuntimeError('failed gain planning bank is terminal')
        started, dispatched, index = time.perf_counter(), False, None
        self._calls += 1
        try:
            index, value = self._validate(qpos, qvel, public_target, gain, inputs, step)
            self._last = {'gain_index': index, 'gain': value, 'step': step, 'status': 'running',
                          'model_binary_sha256': self._member_summaries[index]['configuration']['model_binary_sha256'],
                          'wall_seconds': 0.}
            dispatched = True
            self._last_journal = None
            result, journal = self._planners[index].plan(qpos, qvel, public_target, inputs,
                                                         step=step, deadline=deadline)
            self._refresh(index)
            self._last['status'] = 'completed'
            returned_snapshot = self.snapshot(include_journal=False)
            _deadline(deadline)
            self._completed += 1
        except BaseException as error:
            if dispatched:
                self._failed = True
                self._failures += 1
                self._last.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
                try:
                    self._refresh(index)
                except BaseException as capture_error:  # noqa: BLE001 - preserve the original failure.
                    self._last['snapshot_error'] = type(capture_error).__name__
            else:
                self._rejected += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            self._plan_seconds += elapsed
            if dispatched:
                self._last['wall_seconds'] = elapsed
        returned_snapshot['costs']['plan_call_wall_seconds'] = self._plan_seconds
        returned_snapshot['costs']['plan_calls_completed'] = self._completed
        returned_snapshot['last_operation']['wall_seconds'] = elapsed
        return result, journal, returned_snapshot

    def snapshot(self, *, include_journal=True):
        """Independent snapshot, optionally including the last selected native journal.

        Partial masks/cursors and native states are unchanged from the underlying
        adapter. Aggregate counters include every gain, including failed work.
        Calling this method does no planning, simulation or estimator update.
        """
        totals = {key: sum(member['lifetime'][key] for member in self._member_summaries)
                  for key in self._member_summaries[0]['lifetime']}
        result = {'configuration': self.configuration(), 'failed': self._failed,
                  'members': [{'gain': float(gain), **member}
                              for gain, member in zip(self._gains, self._member_summaries, strict=True)],
                  'costs': {'setup_wall_seconds': self._setup_seconds,
                            'plan_call_wall_seconds': self._plan_seconds,
                            'plan_calls': self._calls, 'plan_calls_completed': self._completed,
                            'input_rejections': self._rejected, 'plan_calls_failed': self._failures,
                            'temporary_gain_model_copies': len(self._gains),
                            'retained_adapter_model_copies': len(self._gains),
                            'native_data_instances': len(self._gains),
                            'nested_adapter_setup_seconds': sum(m['setup_seconds'] for m in self._member_summaries),
                            'aggregate_adapter_lifetime': totals},
                  'last_operation': self._last}
        if include_journal:
            result['last_native_journal'] = self._last_journal
        return copy.deepcopy(result)
