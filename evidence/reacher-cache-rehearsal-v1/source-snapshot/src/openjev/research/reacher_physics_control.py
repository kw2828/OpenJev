"""Supplied-physics controls for the masked Reacher qualification.

The filter sees only packets and issued commands, but knows the simulator and
noise law. Gaussian measurement weighting followed by exact angle reanchoring
is a heuristic particle filter, not exact Bayesian inference. MPC uses nominal
noise-free dynamics for every supplied action sequence. No live MjData or
environment enters either constructor.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _real(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must be finite real data with shape {shape}')
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be representable in float64')
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _scalar(value, name, positive=False):
    raw = _real(value, (), name)
    if raw < 0 or (positive and raw == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(raw)


def _model(model):
    if not isinstance(model, mujoco.MjModel):
        raise TypeError('model must be an MjModel, never an environment or MjData')
    if (model.nq, model.nv, model.nu) != (4, 4, 2):
        raise ValueError('Expected the native two-joint Reacher model')
    if not np.array_equal(model.actuator_ctrlrange, [[-1., 1.], [-1., 1.]]):
        raise ValueError('Expected native normalized Reacher action limits')
    for name in ('fingertip', 'target'):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f'Missing Reacher body {name}')
    return model


def _packet(value):
    value = _real(value, (8,), 'packet')
    if value[6] not in (0., 1.) or value[7] < 0:
        raise ValueError('Invalid packet validity or age')
    if value[6]:
        if value[7] != 0 or not np.allclose(value[:2]**2 + value[2:4]**2, 1., atol=1e-6):
            raise ValueError('Valid packets require unit-circle angles and zero age')
    elif np.any(value[:4] != 0):
        raise ValueError('Missing observations require zero angle placeholders')
    return value


def _angles(value):
    return np.arctan2(value[2:4], value[:2])


def _wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def _step(model, data, action, frame_skip):
    data.ctrl[:] = action
    mujoco.mj_step(model, data, nstep=frame_skip)
    # Match Gymnasium's do_simulation, including its post-constraint refresh.
    mujoco.mj_rnePostConstraint(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError('Nonfinite Reacher physics')


class PhysicsMPC:
    """Nominal supplied-physics MPC, with fixed native reward weights 1/1."""

    def __init__(self, model, frame_skip=2):
        self.model = _model(model)
        self.frame_skip = _integer(frame_skip, 'frame_skip', 1)
        self._data = mujoco.MjData(self.model)
        self._finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fingertip')
        self._target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')

    def plan(self, qpos, qvel, candidates):
        """Return (first_action[2] float32, positive summed costs[K] float64).

        Inputs are copied. Clipping and float32 command quantization match the
        masked environment before disturbance. Ties select the first candidate.
        States may be filter estimates or explicitly privileged reference state.
        """
        qpos, qvel = _real(qpos, (4,), 'qpos'), _real(qvel, (4,), 'qvel')
        raw = np.asarray(candidates)
        if raw.ndim != 3 or raw.shape[-1] != 2 or min(raw.shape[:2]) < 1:
            raise ValueError('candidates must have nonempty shape [K,H,2]')
        actions = np.clip(_real(raw, raw.shape, 'candidates'), -1., 1.).astype(np.float32)
        costs = np.zeros(len(actions), dtype=np.float64)
        data = self._data
        for k, sequence in enumerate(actions):
            mujoco.mj_resetData(self.model, data)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(self.model, data)
            for action in sequence:
                applied = action.astype(np.float64)
                _step(self.model, data, applied, self.frame_skip)
                costs[k] += np.linalg.norm(data.xipos[self._finger] - data.xipos[self._target])
                costs[k] += np.dot(applied, applied)
        if not np.isfinite(costs).all():
            raise FloatingPointError('Nonfinite Reacher plan costs')
        return actions[int(np.argmin(costs)), 0].copy(), costs


class ParticleFilter:
    """History-only state estimate with explicitly supplied simulator knowledge.

    Initial angles/target come from the packet; initial joint velocities follow
    the native uniform [-.005,.005] prior. Seed must be a separately frozen
    estimator seed, not a recovered environment RNG state. Each valid packet
    weights predicted particles, resamples, and reanchors angles while retaining
    inferred velocities. Invalid packets only advance the dynamics.
    """

    def __init__(self, model, initial_packet, *, seed, particles=32,
                 noise_std=.05, measurement_std=.02, frame_skip=2):
        self.model = _model(model)
        self.frame_skip = _integer(frame_skip, 'frame_skip', 1)
        self.particles = _integer(particles, 'particles', 1)
        self.noise_std = _scalar(noise_std, 'noise_std')
        self.measurement_std = _scalar(measurement_std, 'measurement_std', positive=True)
        initial = _packet(initial_packet)
        if not initial[6]:
            raise ValueError('Initial packet must contain a valid angle observation')
        streams = np.random.SeedSequence(_integer(seed, 'seed')).spawn(3)
        initial_rng, self._noise_rng, self._resample_rng = [np.random.default_rng(s) for s in streams]
        self._target = initial[4:6].copy()
        self._age = 0.
        self._data = [mujoco.MjData(model) for _ in range(self.particles)]
        for data in self._data:
            data.qpos[:] = np.concatenate((_angles(initial), self._target))
            data.qvel[:2] = initial_rng.uniform(-.005, .005, 2)
            data.qvel[2:] = 0.
            mujoco.mj_forward(model, data)

    def estimate(self):
        """Independent qpos[4]/qvel[4] arrays, with a circular shoulder mean."""
        angles = np.stack([d.qpos[:2] for d in self._data])
        position = angles.mean(axis=0)
        position[0] = np.arctan2(np.sin(angles[:, 0]).mean(), np.cos(angles[:, 0]).mean())
        velocity = np.mean([d.qvel[:2] for d in self._data], axis=0)
        return np.concatenate((position, self._target)), np.concatenate((velocity, [0., 0.]))

    def update(self, issued_command, next_packet):
        """Advance one decision, assimilate only the next allowed packet."""
        packet = _packet(next_packet)
        command = np.clip(_real(issued_command, (2,), 'issued_command'), -1., 1.).astype(np.float32)
        if not np.array_equal(packet[4:6], self._target):
            raise ValueError('Static target changed within episode')
        expected_age = 0. if packet[6] else self._age + self.model.opt.timestep * self.frame_skip
        if not np.isclose(packet[7], expected_age, atol=1e-6, rtol=1e-6):
            raise ValueError('Packet age does not match one control decision')
        noise = self._noise_rng.normal(0., self.noise_std, (self.particles, 2))
        for data, disturbance in zip(self._data, noise, strict=True):
            _step(self.model, data, np.clip(command.astype(float) + disturbance, -1., 1.), self.frame_skip)
        if packet[6]:
            measured = _angles(packet)
            residual = _wrap(np.stack([d.qpos[:2] for d in self._data]) - measured)
            squared = np.sum(residual**2, axis=1)
            # Center before scaling, and divide twice rather than squaring a
            # tiny bandwidth. At least one log weight remains exactly zero;
            # other weights may safely underflow for very sharp likelihoods.
            with np.errstate(over='ignore', under='ignore'):
                logweights = -.5 * ((squared - squared.min()) / self.measurement_std) / self.measurement_std
                weights = np.exp(logweights)
            weights /= weights.sum()
            points = (self._resample_rng.random() + np.arange(self.particles)) / self.particles
            cumulative = np.cumsum(weights)
            cumulative[-1] = 1.
            indices = np.searchsorted(cumulative, points, side='right')
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            states = np.empty((self.particles, mujoco.mj_stateSize(self.model, spec)))
            for i, data in enumerate(self._data):
                mujoco.mj_getState(self.model, data, states[i], spec)
            for data, index in zip(self._data, indices, strict=True):
                mujoco.mj_setState(self.model, data, states[index], spec)
                # Joint limits are soft: the elbow can transiently cross pi.
                # Preserve each predicted branch rather than replacing +pi by
                # -pi and moving to the opposite physical joint stop.
                data.qpos[:2] += _wrap(measured - data.qpos[:2])
                data.qpos[2:] = self._target
                data.qvel[2:] = 0.
                mujoco.mj_forward(self.model, data)
        self._age = float(packet[7])
        return self.estimate()
