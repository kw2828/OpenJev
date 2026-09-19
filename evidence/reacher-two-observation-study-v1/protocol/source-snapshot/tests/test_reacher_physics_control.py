"""Engineering-only physics parity and information-boundary tests."""

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('gymnasium.envs.mujoco.reacher_v5')

from openjev.research.reacher_physics_control import ParticleFilter, PhysicsMPC
from openjev.research.robotics_reacher import make_env, packet


@pytest.fixture
def env():
    environment = make_env()
    environment.reset(seed=17)
    yield environment
    environment.close()


def test_mpc_native_multistep_cost_parity_and_no_mutations(env):
    native = env.unwrapped
    qpos, qvel = native.data.qpos.copy(), native.data.qvel.copy()
    candidates = np.array([[[.1, -.2], [2., -.1], [.03, .4]],
                           [[-.2, .05], [.1, -.4], [0., 0.]]])
    copies = [x.copy() for x in (qpos, qvel, candidates)]
    expected = []
    for sequence in candidates:
        env.reset(seed=17)
        native.set_state(qpos, qvel)
        cost = 0.
        for command in sequence:
            _, reward, _, _, _ = env.step(np.clip(command, -1., 1.).astype(np.float32).astype(float))
            cost -= reward
        expected.append(cost)
    live_state = native.data.qpos.copy(), native.data.qvel.copy()
    selected, costs = PhysicsMPC(native.model).plan(qpos, qvel, candidates)
    np.testing.assert_allclose(costs, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(selected, candidates[np.argmin(expected), 0].astype(np.float32))
    for original, copied in zip((qpos, qvel, candidates), copies, strict=True):
        np.testing.assert_array_equal(original, copied)
    np.testing.assert_array_equal(native.data.qpos, live_state[0])
    np.testing.assert_array_equal(native.data.qvel, live_state[1])


def test_mpc_candidates_independent_of_order_and_ties(env):
    native = env.unwrapped
    planner = PhysicsMPC(native.model)
    candidates = np.array([[[.1, 0.], [.2, -.1]], [[-.1, .1], [0., 0.]]])
    first, costs = planner.plan(native.data.qpos, native.data.qvel, candidates)
    other, reordered = planner.plan(native.data.qpos, native.data.qvel, candidates[::-1])
    np.testing.assert_array_equal(costs, reordered[::-1])
    np.testing.assert_array_equal(first, other)
    action, costs = planner.plan(native.data.qpos, native.data.qvel, np.zeros((2, 2, 2)))
    np.testing.assert_array_equal(action, [0., 0.])
    assert costs[0] == costs[1]


def initial_packet(env):
    return packet(env.unwrapped.data.qpos[:2], env.unwrapped.data.qpos[2:])


def test_filter_has_no_live_state_dependency_and_copies_estimates(env):
    initial = initial_packet(env)
    initial_copy = initial.copy()
    first = ParticleFilter(env.unwrapped.model, initial, seed=12)
    env.unwrapped.data.qpos[:] = [2., -2., .1, .1]
    env.unwrapped.data.qvel[:] = [9., -9., 0., 0.]
    second = ParticleFilter(env.unwrapped.model, initial, seed=12)
    missing = packet([0., 0.], initial[4:6], valid=False, age_seconds=.02)
    one = first.update([.1, -.2], missing)
    two = second.update([.1, -.2], missing)
    for a, b in zip(one, two, strict=True):
        np.testing.assert_array_equal(a, b)
    one[0][:] = 99.
    one[1][:] = 99.
    for a, b in zip(first.estimate(), two, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(initial, initial_copy)


@pytest.mark.parametrize('measurement_std', [1e-8, 1e-300])
def test_filter_reanchors_observed_angles_and_handles_extreme_likelihood(env, measurement_std):
    initial = initial_packet(env)
    filt = ParticleFilter(env.unwrapped.model, initial, seed=3, measurement_std=measurement_std)
    measured = packet([2.7, -2.5], initial[4:6])
    qpos, qvel = filt.update([0., 0.], measured)
    np.testing.assert_allclose(qpos[:2], np.arctan2(measured[2:4], measured[:2]), atol=2e-7)
    assert np.isfinite(qvel).all()
    np.testing.assert_array_equal(qpos[2:], initial[4:6])
    np.testing.assert_array_equal(qvel[2:], [0., 0.])


def test_filter_clips_commands_before_noise_and_is_deterministic(env):
    initial = initial_packet(env)
    a = ParticleFilter(env.unwrapped.model, initial, seed=1)
    b = ParticleFilter(env.unwrapped.model, initial, seed=1)
    missing = packet([0., 0.], initial[4:6], valid=False, age_seconds=.02)
    for first, second in zip(a.update([100., -100.], missing), b.update([1., -1.], missing), strict=True):
        np.testing.assert_array_equal(first, second)


def test_filter_reanchor_preserves_elbow_branch_beyond_soft_joint_limit(env):
    initial = initial_packet(env)
    filt = ParticleFilter(env.unwrapped.model, initial, seed=3, noise_std=0.)
    for data in filt._data:
        data.qpos[1] = 3.2
        data.qvel[:] = 0.
    measured = packet([0., 3.2], initial[4:6])
    qpos, _ = filt.update([0., 0.], measured)
    assert qpos[1] > np.pi
    np.testing.assert_allclose(qpos[1], 3.2, atol=1e-7)


def test_filter_initial_velocity_prior_and_no_hidden_targets(env):
    initial = initial_packet(env)
    filt = ParticleFilter(env.unwrapped.model, initial, seed=1)
    qpos, qvel = filt.estimate()
    np.testing.assert_allclose(qpos[:2], np.arctan2(initial[2:4], initial[:2]), atol=2e-8)
    assert np.all(np.abs(qvel[:2]) <= .005)
    with pytest.raises(TypeError):
        ParticleFilter(env, initial, seed=1)
    with pytest.raises(TypeError):
        ParticleFilter(env.unwrapped.data, initial, seed=1)


def test_filter_rejects_invalid_packet_before_advancing(env):
    initial = initial_packet(env)
    first = ParticleFilter(env.unwrapped.model, initial, seed=5)
    reference = ParticleFilter(env.unwrapped.model, initial, seed=5)
    invalid = packet([0., 0.], initial[4:6], valid=False, age_seconds=.5)
    with pytest.raises(ValueError, match='age'):
        first.update([0., 0.], invalid)
    invalid = packet([0., 0.], initial[4:6] + .1, valid=False, age_seconds=.02)
    with pytest.raises(ValueError, match='target'):
        first.update([0., 0.], invalid)
    valid = packet([0., 0.], initial[4:6], valid=False, age_seconds=.02)
    for a, b in zip(first.update([.1, 0.], valid), reference.update([.1, 0.], valid), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('candidates', [[], np.zeros((0, 2, 2)), np.zeros((2, 0, 2)),
                                        np.zeros((2, 2, 3)), np.full((1, 1, 2), np.nan)])
def test_invalid_candidates_rejected(env, candidates):
    with pytest.raises(ValueError):
        PhysicsMPC(env.unwrapped.model).plan(env.unwrapped.data.qpos, env.unwrapped.data.qvel, candidates)


@pytest.mark.parametrize('kwargs', [{'particles': 0}, {'seed': True}, {'noise_std': -1},
                                  {'measurement_std': 0}, {'frame_skip': 0}])
def test_invalid_filter_configuration_rejected(env, kwargs):
    with pytest.raises(ValueError):
        ParticleFilter(env.unwrapped.model, initial_packet(env), **({'seed': 1} | kwargs))
