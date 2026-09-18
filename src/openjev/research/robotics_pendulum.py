"""CPU Pendulum qualification kernels, not a scored benchmark or learned model.

The history estimator has privileged knowledge of the simulator equations and
physical constants. Its *inputs* contain only angle observations and issued
commands. It assumes a constant gain within the supplied history and noiseless
unit-circle observations (allowing floating-point rounding).
"""

from dataclasses import dataclass, fields
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class Config:
    g: float = 10.
    mass: float = 1.
    length: float = 1.
    max_speed: float = 8.
    max_torque: float = 2.
    dt: float = .05

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0 or (field.name != 'g' and value == 0)):
                raise ValueError(f'{field.name} must be finite and {"nonnegative" if field.name == "g" else "positive"}')
            object.__setattr__(self, field.name, float(value))


DEFAULT_CONFIG = Config()


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must contain finite real numbers')
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be representable in float64')
    return result


def _config(config):
    if not isinstance(config, Config):
        raise TypeError('config must be a Config')
    return config


def _state(state):
    state = _array(state, 'state')
    if state.ndim != 2 or state.shape[1] != 2 or not len(state):
        raise ValueError('state must have nonempty shape [N,2]')
    return state


def normalize_angle(angle):
    """Normalize finite scalar/array angles to [-pi, pi), as Gymnasium does."""
    angle = _array(angle, 'angle')
    return (angle + np.pi) % (2 * np.pi) - np.pi


def observation(state):
    """Return [cos(theta), sin(theta)]; no velocity, gain, reward or info fields."""
    theta = _state(state)[:, 0]
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1)


def transition(state, commands, gains, config=DEFAULT_CONFIG):
    """One semi-implicit step; returns float64 next_state[N,2], reward[N].

    Gain acts after command clipping and before applied-torque clipping. Any
    finite signed gain is supported; experimental gain ranges belong in the
    caller's protocol. Reward uses the current state and *applied* torque.
    Inputs are neither mutated nor implicitly broadcast across environments.
    """
    config = _config(config)
    state = _state(state)
    commands, gains = _array(commands, 'commands'), _array(gains, 'gains')
    if commands.shape != (len(state),) or gains.shape != (len(state),):
        raise ValueError('commands and gains must have shape [N] matching state')
    theta, omega = state.T
    # A finite but enormous gain may overflow multiplication; clipping is still
    # well-defined. Other numerical failures are rejected below.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        torque = np.clip(np.clip(commands, -config.max_torque, config.max_torque) * gains,
                         -config.max_torque, config.max_torque)
        reward = -(normalize_angle(theta)**2 + .1 * omega**2 + .001 * torque**2)
        next_omega = omega + (3 * config.g / (2 * config.length) * np.sin(theta)
                             + 3 / (config.mass * config.length**2) * torque) * config.dt
        next_omega = np.clip(next_omega, -config.max_speed, config.max_speed)
        result = np.stack((theta + next_omega * config.dt, next_omega), axis=-1)
    if not np.isfinite(result).all() or not np.isfinite(reward).all():
        raise ValueError('configuration and inputs produce nonfinite dynamics or reward')
    return result, reward


def history_estimate(observations, commands, config=DEFAULT_CONFIG):
    """Return (final_omega[B], constant_gain[B], informative_count[B]).

    For T observations, commands has length T-1. Semi-implicit integration makes
    angle differences reveal omega_1..omega_(T-1), not omega_0. Thus the first
    gain equation uses command_1 and requires three observations. Least squares
    pools unsaturated equations. A zero command, clipped next velocity, or
    inferred torque at its limit contributes no equality identifying gain.
    Zero *gain* remains identifiable from nonzero commands.

    T=1 uses omega=0; fewer than three observations cannot identify gain. The
    gain fallback is always 1 when informative_count=0, not a measured gain.
    Angular sampling must satisfy dt*max_speed < pi. Rounding tolerances scale
    with observation dtype and finite-difference amplification; noisy sensing,
    irregular timing and changing gain require a separately specified filter.
    """
    config = _config(config)
    raw = np.asarray(observations)
    eps = np.finfo(raw.dtype).eps if raw.dtype.kind == 'f' else np.finfo(np.float64).eps
    observed, commands = _array(raw, 'observations'), _array(commands, 'commands')
    if observed.ndim != 3 or observed.shape[-1] != 2 or min(observed.shape[:2]) < 1:
        raise ValueError('observations must have nonempty shape [B,T,2]')
    batch, steps, _ = observed.shape
    if commands.shape != (batch, steps - 1):
        raise ValueError('commands must have shape [B,T-1]')
    if not np.allclose(np.sum(observed**2, axis=-1), 1., rtol=1e-5, atol=1e-6):
        raise ValueError('observations must be unit-circle cosine/sine pairs')
    if config.dt * config.max_speed >= np.pi:
        raise ValueError('dt*max_speed must be below pi to avoid angular aliasing')
    gain, count = np.ones(batch), np.zeros(batch, dtype=np.int64)
    if steps == 1:
        return np.zeros(batch), gain, count
    theta = np.arctan2(observed[..., 1], observed[..., 0])
    omega = normalize_angle(np.diff(theta, axis=1)) / config.dt
    angle_tol = 16 * eps
    speed_tol = angle_tol / config.dt + 1e-10 * config.max_speed
    if np.any(np.abs(omega) > config.max_speed + speed_tol):
        raise ValueError('observed angular increments exceed max_speed')
    omega = np.clip(omega, -config.max_speed, config.max_speed)
    if steps == 2:
        return omega[:, -1].copy(), gain, count
    gravity = 3 * config.g / (2 * config.length)
    torque_coefficient = 3 / (config.mass * config.length**2)
    torque = (np.diff(omega, axis=1) / config.dt - gravity * np.sin(theta[:, 1:-1]))
    torque /= torque_coefficient
    issued = np.clip(commands[:, 1:], -config.max_torque, config.max_torque)
    torque_tol = ((2 * speed_tol / config.dt + abs(gravity) * angle_tol) / torque_coefficient
                  + 1e-10 * config.max_torque)
    valid = ((np.abs(issued) > 1e-8 * config.max_torque)
             & (np.abs(omega[:, 1:]) < config.max_speed - speed_tol)
             & (np.abs(torque) < config.max_torque - torque_tol))
    count = valid.sum(axis=1, dtype=np.int64)
    denominator = np.sum(np.where(valid, issued**2, 0.), axis=1)
    numerator = np.sum(np.where(valid, issued * torque, 0.), axis=1)
    np.divide(numerator, denominator, out=gain, where=count > 0)
    if not np.isfinite(gain).all():
        raise ValueError('configuration and history produce a nonfinite gain estimate')
    return omega[:, -1].copy(), gain, count
