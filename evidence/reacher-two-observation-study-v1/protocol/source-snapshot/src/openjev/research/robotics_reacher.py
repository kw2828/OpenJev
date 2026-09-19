"""Masked Reacher packets and auditable collection, not a trained controller.

Only ``record['policy']`` is a policy-input dataset. Rewards are separate audit
values that a caller may explicitly select as supervised targets. Clean hidden
angles, velocities, applied controls and simulator state are audit-only.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ENV_ID = 'Reacher-v5'
HORIZON = 50
REWARD_DIST_WEIGHT = 1.
REWARD_CONTROL_WEIGHT = 1.  # Installed constructor default, despite prose docs saying .1.
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _seed(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _vector(value, name, size=2):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must be a finite real vector of length {size}')
    return raw.astype(np.float64)


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be finite and nonnegative')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be finite and nonnegative') from error
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _schedule(value):
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.shape != (HORIZON + 1,) or not result[0]:
        raise ValueError('sensor_schedule must be bool[51] with a valid initial observation')
    return result.copy()


def packet(angles, target, *, valid=True, age_seconds=0.):
    """Build the only policy packet, independently of the native observation."""
    angles, target = _vector(angles, 'angles'), _vector(target, 'target')
    if not isinstance(valid, (bool, np.bool_)):
        raise TypeError('valid must be boolean')
    age_seconds = _nonnegative(age_seconds, 'age_seconds')
    if valid and age_seconds != 0:
        raise ValueError('a fresh valid observation must have zero age')
    joint = np.concatenate((np.cos(angles), np.sin(angles))) if valid else np.zeros(4)
    result = np.concatenate((joint, target, [float(valid), age_seconds])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('packet must be representable in float32')
    return result


def make_env():
    """Native 50-step task with both actual constructor reward defaults explicit."""
    env = gym.make(ENV_ID, reward_dist_weight=REWARD_DIST_WEIGHT,
                   reward_control_weight=REWARD_CONTROL_WEIGHT, frame_skip=2)
    if env.spec.max_episode_steps != HORIZON:
        env.close()
        raise RuntimeError('Unexpected native Reacher episode horizon')
    return env


def native_identity(env):
    from gymnasium.envs.mujoco import reacher_v5

    native = env.unwrapped
    return {
        'env_id': ENV_ID, 'gymnasium': gym.__version__, 'mujoco': mujoco.__version__,
        'numpy': np.__version__, 'horizon': HORIZON, 'dt': native.dt,
        'frame_skip': native.frame_skip, 'state_spec': int(STATE_SPEC),
        'reward_dist_weight': REWARD_DIST_WEIGHT, 'reward_control_weight': REWARD_CONTROL_WEIGHT,
        'xml_sha256': hashlib.sha256(Path(native.fullpath).read_bytes()).hexdigest(),
        'native_source_sha256': hashlib.sha256(Path(reacher_v5.__file__).read_bytes()).hexdigest(),
    }


def _integration_state(env):
    native = env.unwrapped
    result = np.empty(mujoco.mj_stateSize(native.model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(native.model, native.data, result, STATE_SPEC)
    return result


def restore_native(env, record):
    """Restore native physics and time-limit index, not wrapper noise RNG state.

    Replay uses the explicitly saved applied actions. This function does not
    claim to regenerate future noisy wrapper actions or a sensor schedule.
    """
    native = env.unwrapped
    state = _vector(record['integration_state'], 'integration_state',
                    mujoco.mj_stateSize(native.model, STATE_SPEC))
    index = _seed(record['step'], 'step')
    if index > HORIZON:
        raise ValueError('step exceeds native horizon')
    mujoco.mj_setState(native.model, native.data, state, STATE_SPEC)
    mujoco.mj_forward(native.model, native.data)
    env._elapsed_steps = index


class ReacherEpisode:
    """Packet-only reset/step API; privileged values require audit_record()."""

    def __init__(self, noise_std=0.):
        self.noise_std = _nonnegative(noise_std, 'noise_std')
        self._env = make_env()
        self._record = None
        self._index = 0
        self._finished = False

    @property
    def dt(self):
        return self._env.unwrapped.dt

    @property
    def step_index(self):
        return self._index

    @property
    def finished(self):
        return self._finished

    def reset(self, seed, sensor_schedule, noise_seed=None):
        seed = _seed(seed, 'seed')
        noise_seed = _seed(seed + 1900033 if noise_seed is None else noise_seed, 'noise_seed')
        self._schedule = _schedule(sensor_schedule)
        self._noise_rng = np.random.default_rng(noise_seed)
        raw, _ = self._env.reset(seed=seed)
        self._index, self._last_valid, self._finished = 0, 0, False
        self._target = self._env.unwrapped.data.qpos[2:4].copy()
        self._metadata = {**native_identity(self._env), 'seed': seed, 'noise_seed': noise_seed,
                          'noise_std': self.noise_std, 'sensor_schedule': self._schedule.tolist(),
                          'policy_keys': ['packets', 'commands'],
                          'noise_timing': 'Independent Gaussian draw per decision, held over two physics substeps; clip command then add noise then clip applied action.'}
        self._record_state(raw, None, None, None, None, {}, False, False)
        result = self._packet()
        self._packets, self._records = [result.copy()], [self._record]
        return result

    def _packet(self):
        return packet(self._env.unwrapped.data.qpos[:2], self._target,
                      valid=self._schedule[self._index],
                      age_seconds=(self._index - self._last_valid) * self.dt)

    def _record_state(self, raw, command, noise, applied, reward, info, terminated, truncated):
        data = self._env.unwrapped.data
        self._record = {
            'step': self._index, 'time': float(data.time), 'qpos': data.qpos.copy(),
            'qvel': data.qvel.copy(), 'raw_obs': np.array(raw, copy=True),
            'integration_state': _integration_state(self._env), 'command': command,
            'actuator_noise': noise, 'applied_action': applied, 'reward': reward,
            'reward_dist': info.get('reward_dist'), 'reward_ctrl': info.get('reward_ctrl'),
            'terminated': bool(terminated), 'truncated': bool(truncated),
        }

    def step(self, command):
        if self._record is None or self._finished:
            raise RuntimeError('reset is required before stepping or after episode end')
        # The issued command stored for learning is exactly the quantized value
        # used by the simulator wrapper, before the hidden disturbance.
        command = np.clip(_vector(command, 'command'), -1., 1.).astype(np.float32)
        noise = self._noise_rng.normal(0., self.noise_std, 2)
        applied = np.clip(command.astype(np.float64) + noise, -1., 1.)
        raw, reward, terminated, truncated, info = self._env.step(applied)
        self._index += 1
        if self._schedule[self._index]:
            self._last_valid = self._index
        self._finished = bool(terminated or truncated)
        self._record_state(raw, command, noise, applied, float(reward), info, terminated, truncated)
        result = self._packet()
        self._packets.append(result.copy())
        self._records.append(self._record)
        return result

    def audit_record(self):
        if self._record is None:
            raise RuntimeError('reset before requesting an audit record')
        return copy.deepcopy(self._record)

    def metadata(self):
        if self._record is None:
            raise RuntimeError('reset before requesting metadata')
        return copy.deepcopy(self._metadata)

    def episode_record(self):
        """Return independent partial/full policy arrays and explicit audit arrays."""
        if self._record is None:
            raise RuntimeError('reset before requesting an episode record')
        audit = {key: np.stack([r[key] for r in self._records])
                 for key in ('qpos', 'qvel', 'raw_obs', 'integration_state', 'time')}
        for key, singular, tail in [('rewards', 'reward', ()), ('reward_dist', 'reward_dist', ()),
                                   ('reward_ctrl', 'reward_ctrl', ()),
                                   ('applied_actions', 'applied_action', (2,)),
                                   ('actuator_noise', 'actuator_noise', (2,))]:
            audit[key] = (np.stack([r[singular] for r in self._records[1:]]) if self._index
                          else np.empty((0, *tail), dtype=np.float64))
        commands = (np.stack([r['command'] for r in self._records[1:]]) if self._index
                    else np.empty((0, 2), dtype=np.float32))
        return {'policy': {'packets': np.stack(self._packets), 'commands': commands},
                'audit': audit, 'metadata': self.metadata()}

    def close(self):
        self._env.close()


def ik_pd_command(angles, velocities, target, *, kp=16., kd=6.):
    """Supplied-physics exploration teacher using privileged current qpos/qvel.

    Link lengths (.1,.11), elbow limit (+/-3), and motor gear200 come from the
    installed XML. Gains are torque units; divide by gear before control clipping.
    Infeasible targets are projected by clipping IK, never dropped. This teacher
    is not an observation-only policy and does not claim optimal native reward.
    """
    angles, velocities, target = (_vector(angles, 'angles'), _vector(velocities, 'velocities'),
                                 _vector(target, 'target'))
    kp, kd = _nonnegative(kp, 'kp'), _nonnegative(kd, 'kd')
    first, second = .1, .11
    cosine = np.clip((np.dot(target, target) - first**2 - second**2) / (2*first*second), -1., 1.)
    elbows = np.clip(np.array([1., -1.]) * np.arccos(cosine), -3., 3.)
    shoulders = np.arctan2(target[1], target[0]) - np.arctan2(second*np.sin(elbows),
                                                           first + second*np.cos(elbows))
    desired = np.stack((shoulders, elbows), axis=-1)
    # Only the unlimited shoulder admits angle wrapping. Wrapping elbow error
    # would drive through its physical joint stop when switching IK branches.
    errors = desired - angles
    errors[:, 0] = (errors[:, 0] + np.pi) % (2*np.pi) - np.pi
    error = errors[np.argmin(np.sum(errors**2, axis=1))]
    return np.clip((kp*error - kd*velocities) / 200., -1., 1.).astype(np.float32)


def collect_episode(seed, sensor_schedule, noise_std=0., noise_seed=None, action_seed=None, policy='mixed'):
    """Collect one episode; exploration is explicitly supplied-physics privileged.

    Mixed chooses *once per episode*: 1/2 IK/PD + Gaussian(.12), 1/4 Gaussian(.3),
    1/4 Gaussian(.8). Gaussian exploration is held for four decisions, independently
    of hidden actuator noise. The PD feedback itself updates each decision.
    """
    seed = _seed(seed, 'seed')
    action_seed = _seed(seed + 1900043 if action_seed is None else action_seed, 'action_seed')
    if policy not in {'mixed', 'ik_pd', 'random_low', 'random_high'}:
        raise ValueError('Unknown collection policy')
    rng = np.random.default_rng(action_seed)
    mode = rng.choice(['ik_pd', 'random_low', 'random_high'], p=[.5, .25, .25]) if policy == 'mixed' else policy
    scale = {'ik_pd': .12, 'random_low': .3, 'random_high': .8}[mode]
    env = ReacherEpisode(noise_std)
    try:
        env.reset(seed, sensor_schedule, noise_seed)
        for t in range(HORIZON):
            if t % 4 == 0:
                perturbation = rng.normal(0., scale, 2)
            current = env.audit_record()
            teacher = ik_pd_command(current['qpos'][:2], current['qvel'][:2], current['qpos'][2:4]) if mode == 'ik_pd' else np.zeros(2)
            env.step(teacher + perturbation)
        result = env.episode_record()
        result['metadata'].update({'action_seed': action_seed, 'requested_policy': policy,
                    'collector_policy': str(mode), 'action_hold': 4, 'exploration_std': scale,
                    'teacher_privilege': 'IK/PD receives current clean angles, velocities and target. Audit state is not a learned-model input or observation target.',
                    'policy_keys': ['packets', 'commands']})
        return result
    finally:
        env.close()


def native_replay(record, *, atol=1e-10):
    """Saved-output-only native replay; no controller, new noise or model calls.

    Independently reconstruct noise draws and clipping, then replay recorded
    applied actions from the saved integration state. Check every physical and
    policy record. The episode dictionary is the collect_episode return value.
    """
    atol = _nonnegative(atol, 'atol')
    policy, audit, meta = record['policy'], record['audit'], record['metadata']
    if set(policy) != {'packets', 'commands'}:
        raise ValueError('Unexpected policy inputs')
    env = make_env()
    try:
        if any(meta.get(key) != value for key, value in native_identity(env).items()):
            raise ValueError('Native environment identity differs from recording')
        sensor_schedule = _schedule(meta['sensor_schedule'])
        state_size = mujoco.mj_stateSize(env.unwrapped.model, STATE_SPEC)
        shapes = {'qpos': (51, 4), 'qvel': (51, 4), 'raw_obs': (51, 10),
                  'integration_state': (51, state_size), 'time': (51,), 'rewards': (50,),
                  'reward_dist': (50,), 'reward_ctrl': (50,), 'applied_actions': (50, 2),
                  'actuator_noise': (50, 2)}
        if set(audit) != set(shapes):
            raise ValueError('Unexpected audit member set')
        for key, shape in shapes.items():
            if audit[key].shape != shape or not np.isfinite(audit[key]).all():
                raise ValueError(f'Invalid saved {key}')
        if (policy['packets'].shape != (51, 8) or policy['commands'].shape != (50, 2)
                or policy['packets'].dtype != np.float32 or policy['commands'].dtype != np.float32
                or not all(np.isfinite(value).all() for value in policy.values())
                or np.any(np.abs(policy['commands']) > 1)):
            raise ValueError('Invalid policy arrays')
        expected_noise = np.random.default_rng(_seed(meta['noise_seed'], 'noise_seed')).normal(
            0., _nonnegative(meta['noise_std'], 'noise_std'), (50, 2))
        if not np.array_equal(audit['actuator_noise'], expected_noise):
            raise ValueError('Actuator noise differs from frozen stream')
        applied = np.clip(policy['commands'].astype(float) + expected_noise, -1., 1.)
        if not np.array_equal(audit['applied_actions'], applied):
            raise ValueError('Applied actions differ from recorded issued controls and noise')
        raw, _ = env.reset(seed=_seed(meta['seed'], 'seed'))
        if not np.array_equal(_integration_state(env), audit['integration_state'][0]):
            raise ValueError('Initial integration state differs from native reset seed')
        restore_native(env, {'integration_state': audit['integration_state'][0], 'step': 0})
        last_valid, max_error = 0, 0.
        target = audit['qpos'][0, 2:4]
        for t in range(51):
            if sensor_schedule[t]:
                last_valid = t
            wanted = packet(env.unwrapped.data.qpos[:2], target, valid=sensor_schedule[t],
                            age_seconds=(t-last_valid)*env.unwrapped.dt)
            if not np.array_equal(wanted, policy['packets'][t]):
                raise ValueError('Policy packet replay mismatch')
            actual = {'qpos': env.unwrapped.data.qpos, 'qvel': env.unwrapped.data.qvel,
                      'raw_obs': raw, 'integration_state': _integration_state(env),
                      'time': env.unwrapped.data.time}
            for key, value in actual.items():
                error = float(np.max(np.abs(np.asarray(value) - audit[key][t])))
                max_error = max(max_error, error)
                if not np.isfinite(error) or error > atol:
                    raise ValueError(f'Native {key} replay mismatch')
            if t == HORIZON:
                break
            raw, reward, terminated, truncated, info = env.step(applied[t])
            if terminated or truncated != (t == HORIZON-1):
                raise ValueError('Unexpected native episode ending')
            for key, value in [('rewards', reward), ('reward_dist', info['reward_dist']),
                               ('reward_ctrl', info['reward_ctrl'])]:
                error = abs(float(value) - audit[key][t])
                max_error = max(max_error, error)
                if not np.isfinite(error) or error > atol:
                    raise ValueError(f'Native {key} replay mismatch')
        return {'transitions': HORIZON, 'max_abs_error': max_error,
                'new_policy_calls': 0, 'saved_output_only': True}
    finally:
        env.close()
