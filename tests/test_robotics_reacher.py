"""Small native-engine engineering fixtures, not scored study episodes."""

import copy

import numpy as np
import pytest

from openjev.research.robotics_reacher import (
    HORIZON,
    ReacherEpisode,
    collect_episode,
    ik_pd_command,
    make_env,
    native_replay,
    packet,
    restore_native,
)


def schedule():
    result = np.ones(HORIZON + 1, dtype=bool)
    result[2:5] = False
    return result


def test_no_noise_matches_native_every_step_with_explicit_actual_default_rewards():
    wrapper, native = ReacherEpisode(), make_env()
    try:
        first = wrapper.reset(8300001, schedule())
        raw, _ = native.reset(seed=8300001)
        np.testing.assert_array_equal(first[:6], raw[:6].astype(np.float32))
        commands = np.random.default_rng(8300002).uniform(-2, 2, (50, 2))
        for t, command in enumerate(commands):
            observed = wrapper.step(command)
            assert observed.shape == (8,) and observed.dtype == np.float32
            record = wrapper.audit_record()
            raw, reward, terminated, truncated, info = native.step(np.clip(command, -1, 1).astype(np.float32).astype(float))
            np.testing.assert_array_equal(record['qpos'], native.unwrapped.data.qpos)
            np.testing.assert_array_equal(record['qvel'], native.unwrapped.data.qvel)
            np.testing.assert_array_equal(record['raw_obs'], raw)
            assert record['reward'] == reward
            assert record['reward_dist'] == info['reward_dist']
            assert record['reward_ctrl'] == info['reward_ctrl']
            assert not terminated and truncated == (t == 49)
        assert wrapper.finished and wrapper.step_index == 50
        with pytest.raises(RuntimeError):
            wrapper.step([0., 0.])
    finally:
        wrapper.close()
        native.close()


def test_packet_mask_age_static_target_and_no_velocity_or_displacement_leakage():
    wrapper = ReacherEpisode(.05)
    supplied = schedule()
    try:
        first = wrapper.reset(8300011, supplied, noise_seed=8300012)
        supplied[:] = True  # Schedule ownership must be independent.
        target = first[4:6].copy()
        for t in range(1, 6):
            got = wrapper.step([.1, -.1])
            np.testing.assert_array_equal(got[4:6], target)
            if 2 <= t <= 4:
                np.testing.assert_array_equal(got[:4], 0.)
                assert got[6] == 0 and got[7] == pytest.approx((t-1)*.02)
            else:
                assert got[6] == 1 and got[7] == 0
        # Even radical hidden-state changes cannot alter a missing angle packet.
        np.testing.assert_array_equal(packet([0., 0.], target, valid=False, age_seconds=.2),
                                      packet([2., -2.], target, valid=False, age_seconds=.2))
        record = wrapper.audit_record()
        record['qpos'][:] = 99.
        assert np.max(np.abs(wrapper.audit_record()['qpos'])) < 99.
        # Packet builder has no velocity, native displacement, reward or info input.
        assert set(wrapper.audit_record()) >= {'qvel', 'raw_obs', 'reward', 'applied_action'}
    finally:
        wrapper.close()


def test_episode_record_supports_empty_partial_and_full_independent_histories():
    env = ReacherEpisode(.05)
    try:
        with pytest.raises(RuntimeError):
            env.episode_record()
        first = env.reset(8300015, schedule())
        expected_first = first.copy()
        first[:] = 99.
        empty = env.episode_record()
        assert empty['policy']['packets'].shape == (1, 8)
        assert empty['policy']['commands'].shape == (0, 2)
        assert empty['audit']['rewards'].shape == (0,)
        np.testing.assert_array_equal(empty['policy']['packets'][0], expected_first)
        for _ in range(3):
            env.step([.1, -.2])
        partial = env.episode_record()
        assert partial['policy']['packets'].shape == (4, 8)
        assert partial['audit']['raw_obs'].shape == (4, 10)
        partial['audit']['qvel'][:] = 999.
        assert not np.any(env.episode_record()['audit']['qvel'] == 999.)
        for _ in range(47):
            env.step([.1, -.2])
        assert native_replay(env.episode_record())['transitions'] == 50
        env.reset(8300015, schedule())
        assert env.episode_record()['policy']['packets'].shape == (1, 8)
    finally:
        env.close()


def test_command_then_noise_then_applied_clipping_and_reward_uses_applied_action():
    wrapper = ReacherEpisode(.7)
    try:
        wrapper.reset(8300021, schedule(), noise_seed=8300022)
        command = np.array([-100., 100.])
        wrapper.step(command)
        record = wrapper.audit_record()
        noise = np.random.default_rng(8300022).normal(0, .7, 2)
        applied = np.clip([-1., 1.] + noise, -1, 1)
        np.testing.assert_array_equal(record['actuator_noise'], noise)
        np.testing.assert_array_equal(record['applied_action'], applied)
        np.testing.assert_array_equal(record['command'], [-1., 1.])
        np.testing.assert_array_equal(command, [-100., 100.])
        assert record['reward_ctrl'] == pytest.approx(-np.square(applied).sum())
        assert record['reward'] == pytest.approx(record['reward_dist'] + record['reward_ctrl'])
        assert record['reward_ctrl'] != pytest.approx(-2.)
    finally:
        wrapper.close()


def test_noise_stream_is_reproducible_and_independent_of_sensor_schedule():
    first, second = ReacherEpisode(.05), ReacherEpisode(.05)
    try:
        first.reset(8300031, schedule(), noise_seed=8300032)
        second.reset(8300031, np.ones(51, bool), noise_seed=8300032)
        for command in ([.2, -.1], [.5, .7], [-.3, .4]):
            first.step(command)
            second.step(command)
            a, b = first.audit_record(), second.audit_record()
            for key in ('qpos', 'qvel', 'integration_state', 'actuator_noise', 'applied_action'):
                np.testing.assert_array_equal(a[key], b[key])
    finally:
        first.close()
        second.close()


def test_saved_mid_episode_integration_state_replays_multiple_successor_steps():
    wrapper, native = ReacherEpisode(.05), make_env()
    try:
        wrapper.reset(8300041, schedule(), noise_seed=8300042)
        for _ in range(7):
            wrapper.step([.1, -.2])
        saved = wrapper.audit_record()
        native.reset(seed=99)
        restore_native(native, saved)
        for command in ([.2, .3], [-.1, .7], [-.8, .1]):
            wrapper.step(command)
            record = wrapper.audit_record()
            raw, reward, _, _, _ = native.step(record['applied_action'])
            np.testing.assert_allclose(native.unwrapped.data.qpos, record['qpos'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(native.unwrapped.data.qvel, record['qvel'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(raw, record['raw_obs'], atol=1e-12, rtol=0)
            assert reward == pytest.approx(record['reward'], abs=1e-12)
    finally:
        wrapper.close()
        native.close()


def test_ik_pd_uses_fingertip_length_motor_gear_and_respects_elbow_limits():
    theta = np.array([.3, .7])
    target = .1*np.array([np.cos(theta[0]), np.sin(theta[0])]) + .11*np.array([
        np.cos(theta.sum()), np.sin(theta.sum())])
    np.testing.assert_allclose(ik_pd_command(theta, [0., 0.], target), 0., atol=1e-7)
    np.testing.assert_allclose(ik_pd_command(theta, [.2, -.3], target), [-.006, .009], atol=1e-7)
    for unreachable in ([0., 0.], [2., 2.]):
        value = ik_pd_command([0., 0.], [0., 0.], unreachable)
        assert np.isfinite(value).all() and np.max(abs(value)) <= 1
    # At the positive elbow limit, PD must not choose a wrapped shortcut through the stop.
    value = ik_pd_command([0., 3.], [0., 0.], [0., 0.])
    assert value[1] <= 0
    np.testing.assert_array_equal(ik_pd_command(theta, [1e5, -1e5], target), [-1., 1.])


@pytest.mark.parametrize('policy', ['ik_pd', 'random_low', 'random_high', 'mixed'])
def test_collector_is_deterministic_has_separate_policy_arrays_and_exact_native_replay(policy):
    kwargs = {'seed': 8300051, 'sensor_schedule': schedule(), 'noise_std': .05,
              'noise_seed': 8300052, 'action_seed': 8300053, 'policy': policy}
    first, second = collect_episode(**kwargs), collect_episode(**kwargs)
    assert set(first) == {'policy', 'audit', 'metadata'}
    assert set(first['policy']) == {'packets', 'commands'}
    assert first['policy']['packets'].shape == (51, 8)
    assert first['policy']['commands'].shape == (50, 2)
    assert first['audit']['rewards'].shape == (50,)
    assert first['audit']['raw_obs'].shape == (51, 10)
    for group in ('policy', 'audit'):
        for key in first[group]:
            np.testing.assert_array_equal(first[group][key], second[group][key])
    assert first['metadata'] == second['metadata']
    assert 'privilege' in str(first['metadata'])
    result = native_replay(first)
    assert result['transitions'] == 50 and result['max_abs_error'] < 1e-10
    assert result['new_policy_calls'] == 0
    if first['metadata']['collector_policy'].startswith('random'):
        for start in range(0, 50, 4):
            np.testing.assert_array_equal(first['policy']['commands'][start:min(start+4, 50)],
                                          np.broadcast_to(first['policy']['commands'][start],
                                                          (min(4, 50-start), 2)))


@pytest.mark.parametrize('field', ['packet', 'command', 'noise', 'applied', 'reward', 'state', 'identity'])
def test_native_replay_rejects_meaningful_tampering(field):
    record = collect_episode(8300061, schedule(), .05, 8300062, 8300063, policy='random_low')
    altered = copy.deepcopy(record)
    if field == 'packet':
        altered['policy']['packets'][3, 0] = 1.
    elif field == 'command':
        altered['policy']['commands'][3, 0] += .01
    elif field == 'noise':
        altered['audit']['actuator_noise'][3, 0] += .01
    elif field == 'applied':
        altered['audit']['applied_actions'][3, 0] += .01
    elif field == 'reward':
        altered['audit']['rewards'][3] += .01
    elif field == 'state':
        altered['audit']['qvel'][3, 0] += .01
    else:
        altered['metadata']['reward_control_weight'] = .1
    with pytest.raises(ValueError):
        native_replay(altered)


@pytest.mark.parametrize('bad', [np.ones(50, bool), np.ones(51, int), np.zeros(51, bool)])
def test_invalid_sensor_schedule(bad):
    env = ReacherEpisode()
    try:
        with pytest.raises(ValueError):
            env.reset(1, bad)
    finally:
        env.close()


@pytest.mark.parametrize('bad', [[np.nan, 0], [np.inf, 0], [0.], [[0., 0.]], [True, False]])
def test_invalid_commands(bad):
    env = ReacherEpisode()
    try:
        with pytest.raises(RuntimeError):
            env.step([0., 0.])
        env.reset(8300071, schedule())
        before = env.audit_record()
        with pytest.raises(ValueError):
            env.step(bad)
        np.testing.assert_array_equal(before['integration_state'], env.audit_record()['integration_state'])
    finally:
        env.close()


def test_packet_and_noise_validation():
    with pytest.raises(ValueError):
        packet([0., 0.], [0., 0.], valid=True, age_seconds=.1)
    with pytest.raises(ValueError):
        packet([0., 0.], [0., 0.], valid=False, age_seconds=-.1)
    for bad in (-1., np.inf, np.nan, True):
        with pytest.raises(ValueError):
            ReacherEpisode(bad)
