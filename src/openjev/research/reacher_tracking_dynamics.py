"""Explicit-input native Reacher tracking task, not the native 50-step benchmark.

Only reset/step packets are public controller inputs. Audit methods deliberately
expose privileged state and all supplied schedules, including future events.
There is no controller, schedule generator, hidden RNG, or scientific default.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import mujoco
import numpy as np

from openjev.research import robotics_reacher as base

VERSION = 'reacher-tracking-dynamics-v1'
BASE_GEAR = 200.
TARGET_LIMIT = .27


def _array(value, name, shape):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite real {shape}')
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be representable in float64')
    result.flags.writeable = False
    return result


def _native_env(horizon):
    env = gym.make(base.ENV_ID, max_episode_steps=horizon, frame_skip=2,
                   reward_dist_weight=1., reward_control_weight=1.)
    try:
        model = env.unwrapped.model
        if (env.spec.max_episode_steps != horizon or env.unwrapped.dt != .02
                or (model.nq, model.nv, model.nu) != (4, 4, 2)
                or not np.array_equal(model.dof_damping, [1., 1., 0., 0.])
                or not np.array_equal(model.actuator_gear,
                                      [[BASE_GEAR, 0., 0., 0., 0., 0.]] * 2)
                or not np.array_equal(model.actuator_ctrlrange, [[-1., 1.]] * 2)
                or not np.array_equal(model.jnt_range[2:], [[-TARGET_LIMIT, TARGET_LIMIT]] * 2)):
            raise RuntimeError('unsupported native Reacher physics')
        return env
    except BaseException:
        env.close()
        raise


def nominal_model():
    """Independent unmodified XML model for a supplied-physics comparator.

    This creates and copies a fresh gear200, fixed-damping model, never the live
    task model. No reset, random draw, simulation, or plant parameter read occurs.
    Its use is known-model-class privilege, not a learned dynamics model.
    """
    env = _native_env(base.HORIZON)
    try:
        return copy.copy(env.unwrapped.model)
    finally:
        env.close()


class TrackingDynamicsEpisode:
    """One episode with caller-owned event generation and packet-only decisions.

    ``horizon`` is an explicit override of native Reacher's 50-decision limit.
    Every step clips/quantizes the command to float32, adds exactly the supplied
    noise realization, clips the applied normalized control, then uses gear200*g.
    The native reward charges that applied control once, not the scaled torque.
    Target coordinates must lie in the native target-joint box; reachability and
    any gain/noise distribution are responsibilities of the enclosing protocol.

    Instances are single-use. Validation before a native attempt leaves the
    current prefix intact; any failure after an attempt begins is terminal and
    its attempted command and partial native evidence remain audit-accessible.
    """

    def __init__(self, *, horizon, target_path, gear_multiplier, sensor_schedule, noise):
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        self.horizon = int(horizon)
        self._targets = _array(target_path, 'target_path', (self.horizon + 1, 2))
        if np.any(np.abs(self._targets) > TARGET_LIMIT):
            raise ValueError('target_path exceeds native target-joint bounds')
        self._gains = _array(gear_multiplier, 'gear_multiplier', (self.horizon,))
        if np.any(self._gains <= 0) or np.any(self._gains > np.finfo(np.float64).max / BASE_GEAR):
            raise ValueError('gear_multiplier must be positive with finite gear200*g')
        self._noise = _array(noise, 'noise', (self.horizon, 2))
        schedule = np.asarray(sensor_schedule)
        if schedule.shape != (self.horizon + 1,) or schedule.dtype != np.bool_ or not schedule[0]:
            raise ValueError('sensor_schedule must be bool[H+1] with visible startup')
        self._schedule = schedule.copy()
        self._schedule.flags.writeable = False
        self._env = _native_env(self.horizon)
        self._identity = {**base.native_identity(self._env), 'version': VERSION,
                          'base_native_horizon': base.HORIZON, 'horizon': self.horizon,
                          'task': 'custom explicit-target tracking with supplied shared actuator gear',
                          'base_gear': BASE_GEAR, 'fixed_dof_damping': [1., 1., 0., 0.],
                          'public_features': ['cos_q0', 'cos_q1', 'sin_q0', 'sin_q1',
                                              'target_x', 'target_y', 'valid', 'age_seconds'],
                          'reward_timing': 'simulate with gear_t and target_t; capture native reward/raw state; install target_(t+1); emit packet_(t+1)',
                          'noise': 'explicit supplied realization; clip command, quantize float32, add noise, clip applied normalized action',
                          'reward_control_units': 'squared normalized applied action, independent of gear multiplier',
                          'target_event': 'set target qpos/goal and zero target qvel, then mj_forward only on a change; never reset arms',
                          'run_status_authority': 'enclosing protocol and execution receipts'}
        self._status, self._index, self._last_valid = 'created', 0, 0
        self._closed, self._failure, self._seed = False, None, None
        self._packets, self._commands, self._states, self._transitions = [], [], [], []

    @property
    def dt(self):
        return self._identity['dt']

    @property
    def step_index(self):
        return self._index

    @property
    def finished(self):
        return self._status in {'completed', 'failed'}

    def _state(self, raw=None):
        native = self._env.unwrapped
        if raw is None:
            raw = native._get_obs()
        result = {'qpos': native.data.qpos.copy(), 'qvel': native.data.qvel.copy(),
                  'raw_obs': np.array(raw, dtype=np.float64, copy=True),
                  'integration_state': base._integration_state(self._env),
                  'time': float(native.data.time)}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('nonfinite native state')
        return result

    def _set_target(self, target):
        native = self._env.unwrapped
        native.data.qpos[2:] = target
        native.data.qvel[2:] = 0.
        native.goal = target.copy()
        mujoco.mj_forward(native.model, native.data)

    def _packet(self, index):
        return base.packet(self._env.unwrapped.data.qpos[:2], self._targets[index],
                           valid=self._schedule[index],
                           age_seconds=(index - self._last_valid) * self.dt)

    def _fail(self, error, stage):
        self._status = 'failed'
        self._failure = {'type': type(error).__name__, 'message': str(error), 'stage': stage,
                         'completed_decisions': self._index}
        # Preserve an original BaseException even if collecting more evidence fails.
        try:
            self._failure['last_native_state'] = self._state()
        except BaseException as capture_error:  # noqa: BLE001 - preserve the original terminal error.
            self._failure['state_capture_error'] = type(capture_error).__name__

    def reset(self, seed):
        if self._closed or self._status != 'created':
            raise RuntimeError('episode instances are single-use; construct a new episode')
        seed = base._seed(seed, 'seed')
        self._seed = seed
        stage = 'native_reset'
        try:
            self._env.unwrapped.model.actuator_gear[:, 0] = BASE_GEAR
            self._env.reset(seed=seed)
            stage = 'initial_target'
            self._set_target(self._targets[0])
            state, observed = self._state(), self._packet(0)
            self._states.append(state)
            self._packets.append(observed.copy())
            self._status = 'running'
            return observed
        except BaseException as error:
            self._fail(error, stage)
            raise

    def step(self, command):
        if self._closed or self._status != 'running':
            raise RuntimeError('step requires a running nonterminal episode')
        command = np.clip(base._vector(command, 'command'), -1., 1.).astype(np.float32)
        t = self._index
        applied = np.clip(command.astype(np.float64) + self._noise[t], -1., 1.)
        record = {'step': t, 'command': command.copy(), 'actuator_noise': self._noise[t].copy(),
                  'applied_action': applied.copy(), 'gear_multiplier': float(self._gains[t]),
                  'actuator_gear': BASE_GEAR * float(self._gains[t]),
                  'reward_target': self._targets[t].copy(), 'next_target': self._targets[t+1].copy(),
                  'status': 'attempted', 'native_returned': False,
                  'target_changed': not np.array_equal(self._targets[t], self._targets[t+1])}
        self._transitions.append(record)
        stage = 'native_step'
        try:
            native = self._env.unwrapped
            if not np.array_equal(native.model.dof_damping, self._identity['fixed_dof_damping']):
                raise RuntimeError('fixed damping was modified')
            native.model.actuator_gear[:, 0] = record['actuator_gear']
            raw, reward, terminated, truncated, info = self._env.step(applied)
            record['native_returned'] = True
            stage = 'transition_record'
            record['native_after'] = self._state(raw)
            stage = 'reward_record'
            record.update(reward=float(reward),
                          reward_dist=float(info['reward_dist']), reward_ctrl=float(info['reward_ctrl']),
                          terminated=bool(terminated), truncated=bool(truncated))
            if not np.isfinite([record['reward'], record['reward_dist'], record['reward_ctrl']]).all():
                raise FloatingPointError('nonfinite native reward')
            if terminated or bool(truncated) != (t + 1 == self.horizon):
                raise RuntimeError('unexpected native terminal boundary')
            stage = 'next_target'
            if record['target_changed']:
                self._set_target(self._targets[t+1])
            if self._schedule[t+1]:
                self._last_valid = t+1
            stage = 'decision_record'
            state, observed = self._state(), self._packet(t+1)
            self._states.append(state)
            self._packets.append(observed.copy())
            self._commands.append(command.copy())
            self._index += 1
            record['status'] = 'completed'
            self._status = 'completed' if self._index == self.horizon else 'running'
            return observed
        except BaseException as error:
            record['status'] = 'failed'
            self._fail(error, stage)
            raise

    def metadata(self):
        """Audit-only metadata, including future schedules and hidden gain/noise."""
        return copy.deepcopy({**self._identity, 'seed': self._seed, 'status': self._status,
                              'closed': self._closed, 'completed_decisions': self._index,
                              'attempted_transitions': len(self._transitions),
                              'returned_native_transitions': sum(r['native_returned'] for r in self._transitions),
                              'target_path': self._targets.tolist(), 'gear_multiplier': self._gains.tolist(),
                              'sensor_schedule': self._schedule.tolist(), 'noise': self._noise.tolist(),
                              'failure': self._failure})

    def audit_record(self):
        """Latest privileged evidence, never a public controller input."""
        return copy.deepcopy({'step': self._index, 'status': self._status,
                              'decision_state': self._states[-1] if self._states else None,
                              'transition': self._transitions[-1] if self._transitions else None,
                              'failure': self._failure})

    def episode_record(self):
        """Successful public prefix plus all attempts, including a failed last one.

        For M completed decisions, policy arrays are [M+1,8]/[M,2] and
        audit.decision_states have M+1 entries. Failed reset has zero packets.
        audit.transitions may contain one extra failed attempt. Each returned
        native transition preserves its pre-target-event state separately.
        """
        return {'policy': {'packets': (np.stack(self._packets) if self._packets else np.empty((0, 8), np.float32)),
                           'commands': (np.stack(self._commands) if self._commands else np.empty((0, 2), np.float32))},
                'audit': copy.deepcopy({'decision_states': self._states, 'transitions': self._transitions}),
                'metadata': self.metadata()}

    def close(self):
        if not self._closed:
            self._closed = True
            self._env.close()
