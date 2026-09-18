"""Synthetic arithmetic and installed-native parity, not scored study results."""

from dataclasses import replace

import numpy as np
import pytest

from openjev.research.robotics_pendulum import (
    DEFAULT_CONFIG,
    Config,
    history_estimate,
    normalize_angle,
    observation,
    transition,
)


def history(initial, commands, gains, config=DEFAULT_CONFIG):
    states = [np.asarray(initial, dtype=float)]
    for command in np.asarray(commands).T:
        states.append(transition(states[-1], command, gains, config)[0])
    return np.stack([observation(state) for state in states], axis=1), np.stack(states, axis=1)


def test_native_installed_gymnasium_scalar_steps_match_vectorized_gain_one():
    gym = pytest.importorskip('gymnasium.envs.classic_control.pendulum')
    cfg = Config(g=7.4, mass=.8, length=1.3, max_speed=6., max_torque=2.5, dt=.037)
    states = np.array([[0., 0.], [3.13, 3.], [-3.12, -2.], [.7, 5.9], [-1.2, -5.8]])
    commands = np.array([1., -4., 7., 2.5, -2.5])
    expected_state, expected_reward = [], []
    for state, command in zip(states, commands, strict=True):
        env = gym.PendulumEnv(g=cfg.g)
        env.m, env.l, env.max_speed, env.max_torque, env.dt = (
            cfg.mass, cfg.length, cfg.max_speed, cfg.max_torque, cfg.dt)
        env.state = state.copy()
        _, reward, terminated, truncated, _ = env.step(np.array([command]))
        assert not terminated and not truncated
        expected_state.append(env.state.copy())
        expected_reward.append(reward)
        env.close()
    actual_state, actual_reward = transition(states, commands, np.ones(len(states)), cfg)
    np.testing.assert_allclose(actual_state, expected_state, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual_reward, expected_reward, rtol=0, atol=1e-14)


def test_clipping_order_applied_torque_reward_and_semi_implicit_position():
    state = np.array([[0., 0.], [0., 0.], [0., 7.9]])
    commands, gains = np.array([100., -100., 2.]), np.array([.25, 3., 5.])
    copies = [value.copy() for value in (state, commands, gains)]
    result, reward = transition(state, commands, gains)
    np.testing.assert_allclose(result[:, 1], [.075, -.3, 8.])
    np.testing.assert_allclose(result[:, 0], result[:, 1] * .05)
    np.testing.assert_allclose(reward, [-.00025, -.004, -.1 * 7.9**2 - .004])
    for value, original in zip((state, commands, gains), copies, strict=True):
        np.testing.assert_array_equal(value, original)


def test_signed_gain_and_large_finite_gain_are_well_defined_after_clipping():
    result, _ = transition([[0., 0.], [0., 0.]], [1., 2.], [-.5, 1e308])
    np.testing.assert_allclose(result[:, 1], [-.075, .3])


def test_wrong_config_type_is_rejected():
    with pytest.raises(TypeError, match='Config'):
        transition([[0., 0.]], [0.], [1.], {})
    with pytest.raises(TypeError, match='Config'):
        history_estimate([[[1., 0.]]], [[]], None)


def test_normalization_observation_wrap_and_final_velocity():
    np.testing.assert_allclose(normalize_angle([0., np.pi, -np.pi, 5*np.pi]), [0., -np.pi, -np.pi, -np.pi])
    inputs, states = history([[np.pi-.02, 2.], [-np.pi+.01, -2.]], [[.4, .7], [-.7, -.4]], [1.2, .8])
    omega, gain, count = history_estimate(inputs, [[.4, .7], [-.7, -.4]])
    np.testing.assert_allclose(omega, states[:, -1, 1], atol=1e-12)
    np.testing.assert_allclose(gain, [1.2, .8], atol=1e-11)
    np.testing.assert_array_equal(count, [1, 1])
    np.testing.assert_allclose(observation([[0., 8.], [np.pi/2, -8.]]), [[1., 0.], [0., 1.]], atol=1e-15)


@pytest.mark.parametrize('dtype,tolerance', [(np.float64, 1e-10), (np.float32, 3e-4)])
def test_history_gain_recovery_uses_correct_command_index_and_clipped_commands(dtype, tolerance):
    commands = np.array([[.01, 4., -.3, .8], [-.9, .2, -.7, .4], [.1, -.6, .3, -.5]])
    gains = [.35, 1.4, 0.]
    inputs, states = history([[.1, .2], [-.1, -.3], [.2, .4]], commands, gains)
    omega, actual, count = history_estimate(inputs.astype(dtype), commands)
    np.testing.assert_allclose(omega, states[:, -1, 1], atol=tolerance)
    np.testing.assert_allclose(actual, gains, atol=tolerance)
    np.testing.assert_array_equal(count, [3, 3, 3])


def test_zero_commands_are_unidentifiable_despite_distinct_hidden_gains():
    commands = np.zeros((2, 8))
    inputs, _ = history([[.4, .1], [.4, .1]], commands, [.2, 2.7])
    np.testing.assert_array_equal(inputs[0], inputs[1])
    _, gain, count = history_estimate(inputs, commands)
    np.testing.assert_array_equal(gain, [1., 1.])
    np.testing.assert_array_equal(count, [0, 0])


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_torque_and_speed_saturation_are_excluded(dtype):
    commands = np.full((2, 5), 2.)
    inputs, _ = history([[0., 0.], [np.pi/2, 8.]], commands, [3., 1.])
    _, gain, count = history_estimate(inputs.astype(dtype), commands)
    np.testing.assert_array_equal(gain, [1., 1.])
    np.testing.assert_array_equal(count, [0, 0])


def test_saturated_and_informative_torques_are_not_pooled_together():
    commands = np.array([[0., 2., .2, -.3]])
    inputs, _ = history([[0., 0.]], commands, [2.])
    _, gain, count = history_estimate(inputs, commands)
    np.testing.assert_allclose(gain, [2.], atol=1e-11)
    np.testing.assert_array_equal(count, [2])


def test_previous_speed_saturation_does_not_discard_an_unclipped_next_step():
    cfg = Config(g=0.)
    commands = np.array([[0., -1.]])
    inputs, states = history([[0., 8.]], commands, [.5], cfg)
    omega, gain, count = history_estimate(inputs, commands, cfg)
    np.testing.assert_allclose(omega, states[:, -1, 1], atol=1e-12)
    np.testing.assert_allclose(gain, [.5], atol=1e-11)
    np.testing.assert_array_equal(count, [1])


def test_short_history_fallback_is_explicit_and_zero_gain_is_not_missing_data():
    inputs = observation([[.5, 2.]])[:, None]
    omega, gain, count = history_estimate(inputs, np.empty((1, 0)))
    np.testing.assert_array_equal(omega, [0.])
    np.testing.assert_array_equal(gain, [1.])
    np.testing.assert_array_equal(count, [0])
    inputs, states = history([[.5, 2.]], [[1.]], [0.])
    omega, gain, count = history_estimate(inputs, [[1.]])
    np.testing.assert_allclose(omega, states[:, -1, 1], atol=1e-12)
    np.testing.assert_array_equal(gain, [1.])
    np.testing.assert_array_equal(count, [0])


def test_history_is_causal_and_independent_across_batch_rows():
    commands = np.array([[.2, -.3, .4, -.5], [-.7, .6, -.5, .4]])
    inputs, _ = history([[.1, .2], [-.2, -.1]], commands, [.7, 1.3])
    prefix = inputs[:, :3].copy(), commands[:, :2].copy()
    expected = history_estimate(*prefix)
    inputs[:, 3:] = observation([[2., 0.], [-2., 0.]])[:, None]
    commands[:, 2:] = 100.
    actual = history_estimate(inputs[:, :3], commands[:, :2])
    for seen, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(seen, wanted)
    first = history_estimate(prefix[0][:1], prefix[1][:1])
    for seen, wanted in zip(first, expected, strict=True):
        np.testing.assert_array_equal(seen, wanted[:1])
    np.testing.assert_array_equal(prefix[0], inputs[:, :3])
    np.testing.assert_array_equal(prefix[1], commands[:, :2])


def test_gain_is_least_squares_not_unweighted_mean_of_ratios():
    cfg = Config(g=0.)
    # Deliberately inconsistent constant-gain observations distinguish pooling rules.
    theta = np.array([[0., 0., .0025, .0125]])
    inputs = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    commands = np.array([[0., .5, 1.]])
    _, gain, count = history_estimate(inputs, commands, cfg)
    applied = np.array([.05/.05/3, .15/.05/3])
    np.testing.assert_allclose(gain, [np.dot([.5, 1.], applied) / 1.25], atol=1e-12)
    assert not np.isclose(gain[0], np.mean(applied / [.5, 1.]))
    np.testing.assert_array_equal(count, [2])


@pytest.mark.parametrize('field,value', [('g', -1), ('mass', 0), ('length', -1), ('dt', 0),
                                        ('max_speed', np.inf), ('max_torque', np.nan), ('dt', True)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        replace(Config(), **{field: value})


@pytest.mark.parametrize('state,commands,gains', [
    ([0., 0.], [1.], [1.]), ([], [], []), ([[0., 0.]], [[1.]], [1.]),
    ([[0., 0.]], [1.], 1.), ([[0., np.nan]], [1.], [1.]),
    ([[0., 0.]], [np.inf], [1.]), ([[0., 0.]], [1.], [1j]),
    ([[0., 0.]], [True], [1.]), ([[0., 0.]], ['1'], [1.]),
])
def test_invalid_transition_inputs(state, commands, gains):
    with pytest.raises(ValueError):
        transition(state, commands, gains)


@pytest.mark.parametrize('observed,commands,config', [
    ([[1., 0.]], [], Config()), (np.empty((1, 0, 2)), np.empty((1, 0)), Config()),
    ([[[1., 0.]]], [[0.]], Config()), ([[[0., 0.]]], [[]], Config()),
    ([[[np.nan, 0.]]], [[]], Config()), ([[[1., 0.], [0., 1.]]], [[0.]], Config()),
    ([[[1., 0.]]], [[]], Config(dt=1.)),
])
def test_invalid_history_inputs(observed, commands, config):
    with pytest.raises(ValueError):
        history_estimate(observed, commands, config)
