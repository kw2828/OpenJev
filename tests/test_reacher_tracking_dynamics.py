"""Explicit synthetic schedules; native engineering resets use seed410 only."""

import copy

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from openjev.research import reacher_tracking_dynamics as tracking


def inputs(horizon=5):
    targets = np.tile([.12, -.08], (horizon + 1, 1))
    targets[2:] = [-.06, .14]
    gains = np.ones(horizon)
    gains[2:] = .7
    schedule = np.ones(horizon + 1, dtype=bool)
    schedule[1:3] = False
    return {'horizon': horizon, 'target_path': targets, 'gear_multiplier': gains,
            'sensor_schedule': schedule, 'noise': np.zeros((horizon, 2))}


@pytest.fixture
def episode():
    env = tracking.TrackingDynamicsEpisode(**inputs())
    try:
        yield env
    finally:
        env.close()


def set_native_target(env, target):
    native = env.unwrapped
    native.data.qpos[2:] = target
    native.data.qvel[2:] = 0.
    native.goal = target.copy()
    mujoco.mj_forward(native.model, native.data)


def test_every_native_transition_matches_independent_manual_schedule_and_reward():
    spec = inputs()
    spec['noise'][:] = [[.03, -.02], [.5, -.4], [.02, .04], [0., 0.], [-.1, .1]]
    commands = [[.1, -.2], [2., -3.], [-.13, .21], [.5, .25], [-.01, .06]]
    env = tracking.TrackingDynamicsEpisode(**spec)
    native = gym.make('Reacher-v5', max_episode_steps=5, frame_skip=2,
                      reward_dist_weight=1., reward_control_weight=1.)
    try:
        env.reset(410)
        native.reset(seed=410)
        set_native_target(native, spec['target_path'][0])
        for t, command in enumerate(commands):
            observed = env.step(command)
            quantized = np.clip(command, -1, 1).astype(np.float32)
            applied = np.clip(quantized.astype(np.float64) + spec['noise'][t], -1, 1)
            native.unwrapped.model.actuator_gear[:, 0] = 200 * spec['gear_multiplier'][t]
            raw, reward, terminated, truncated, info = native.step(applied)
            record = env.audit_record()['transition']
            np.testing.assert_array_equal(record['native_after']['qpos'], native.unwrapped.data.qpos)
            np.testing.assert_array_equal(record['native_after']['qvel'], native.unwrapped.data.qvel)
            np.testing.assert_array_equal(record['native_after']['raw_obs'], raw)
            np.testing.assert_array_equal(record['command'], quantized)
            np.testing.assert_array_equal(record['applied_action'], applied)
            assert record['reward'] == reward
            assert record['reward_dist'] == info['reward_dist']
            assert record['reward_ctrl'] == info['reward_ctrl'] == -float(applied @ applied)
            assert record['terminated'] == terminated is False
            assert record['truncated'] == truncated == (t == 4)
            if not np.array_equal(spec['target_path'][t], spec['target_path'][t+1]):
                set_native_target(native, spec['target_path'][t+1])
            latest = env.audit_record()['decision_state']
            np.testing.assert_array_equal(latest['qpos'], native.unwrapped.data.qpos)
            np.testing.assert_array_equal(latest['qvel'], native.unwrapped.data.qvel)
            np.testing.assert_array_equal(latest['raw_obs'], native.unwrapped._get_obs())
            np.testing.assert_array_equal(observed[4:6], spec['target_path'][t+1].astype(np.float32))
        assert env.finished and env.step_index == 5
    finally:
        env.close()
        native.close()


def test_target_event_keeps_arm_state_and_time_but_native_reward_uses_old_target(episode):
    episode.reset(410)
    episode.step([.03, -.04])
    observed = episode.step([.03, -.04])
    record = episode.audit_record()
    transition, decision = record['transition'], record['decision_state']
    old = transition['native_after']
    assert transition['target_changed']
    np.testing.assert_array_equal(old['qpos'][:2], decision['qpos'][:2])
    np.testing.assert_array_equal(old['qvel'][:2], decision['qvel'][:2])
    assert old['time'] == decision['time'] == pytest.approx(.04)
    np.testing.assert_array_equal(old['raw_obs'][4:6], transition['reward_target'])
    np.testing.assert_array_equal(decision['raw_obs'][4:6], transition['next_target'])
    assert transition['reward_dist'] == -np.linalg.norm(old['raw_obs'][-2:])
    assert transition['reward_dist'] != pytest.approx(-np.linalg.norm(decision['raw_obs'][-2:]))
    np.testing.assert_array_equal(observed[4:6], transition['next_target'].astype(np.float32))
    np.testing.assert_array_equal(observed[:4], 0.)
    assert observed[6] == 0 and observed[7] == np.float32(.04)


def test_public_packet_has_only_current_target_mask_and_age_not_hidden_schedules():
    first_spec, second_spec = inputs(), inputs()
    second_spec['gear_multiplier'][:] = 1.3
    second_spec['noise'][:] = .2
    first = tracking.TrackingDynamicsEpisode(**first_spec)
    second = tracking.TrackingDynamicsEpisode(**second_spec)
    try:
        np.testing.assert_array_equal(first.reset(410), second.reset(410))
        for _ in range(2):
            a, b = first.step([.1, -.2]), second.step([.1, -.2])
            np.testing.assert_array_equal(a, b)  # Masked angles, same known targets.
            assert a.dtype == np.float32 and a.shape == (8,)
        assert not np.array_equal(first.audit_record()['decision_state']['qpos'],
                                  second.audit_record()['decision_state']['qpos'])
        observed = first.step([.1, -.2])
        assert observed[6] == 1 and observed[7] == 0
    finally:
        first.close()
        second.close()


def test_gain_scales_physics_before_action_but_never_native_control_cost():
    spec = inputs(1)
    spec['noise'][0] = [-.2, .2]
    low = tracking.TrackingDynamicsEpisode(**{**spec, 'gear_multiplier': np.array([.7])})
    high = tracking.TrackingDynamicsEpisode(**{**spec, 'gear_multiplier': np.array([1.3])})
    try:
        low.reset(410)
        high.reset(410)
        command = np.array([9., -9.])
        low.step(command)
        high.step(command)
        a, b = low.audit_record()['transition'], high.audit_record()['transition']
        np.testing.assert_array_equal(a['command'], [1., -1.])
        np.testing.assert_array_equal(a['applied_action'], [.8, -.8])
        assert a['reward_ctrl'] == b['reward_ctrl'] == pytest.approx(-1.28)
        assert a['actuator_gear'] == 140 and b['actuator_gear'] == 260
        assert not np.array_equal(a['native_after']['qvel'], b['native_after']['qvel'])
        np.testing.assert_array_equal(command, [9., -9.])
    finally:
        low.close()
        high.close()


def test_command_quantization_precedes_noise_and_applied_clipping():
    spec = inputs(1)
    spec['noise'][0] = [.8, -.8]
    env = tracking.TrackingDynamicsEpisode(**spec)
    try:
        env.reset(410)
        command = np.array([.300000003, -.300000003])
        env.step(command)
        record = env.audit_record()['transition']
        assert record['command'].dtype == np.float32
        np.testing.assert_array_equal(record['command'], command.astype(np.float32))
        np.testing.assert_array_equal(record['applied_action'], [1., -1.])
        assert record['reward_ctrl'] == -2.
    finally:
        env.close()


def assert_same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_same(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_complete_saved_schedule_and_commands_replay_deterministically_without_rng():
    env = tracking.TrackingDynamicsEpisode(**inputs())
    try:
        env.reset(410)
        for _ in range(5):
            env.step([.17, -.21])
        saved = env.episode_record()
    finally:
        env.close()
    meta = saved['metadata']
    replica = tracking.TrackingDynamicsEpisode(
        horizon=meta['horizon'], target_path=meta['target_path'], gear_multiplier=meta['gear_multiplier'],
        sensor_schedule=np.array(meta['sensor_schedule'], bool), noise=meta['noise'])
    try:
        replica.reset(meta['seed'])
        for command in saved['policy']['commands']:
            replica.step(command)
        assert_same(saved, replica.episode_record())
    finally:
        replica.close()


def test_schedule_and_all_returned_evidence_are_independent_copies():
    spec = inputs()
    expected = copy.deepcopy(spec)
    env = tracking.TrackingDynamicsEpisode(**spec)
    try:
        spec['target_path'][:] = .2
        spec['gear_multiplier'][:] = 9
        spec['noise'][:] = 9
        spec['sensor_schedule'][:] = True
        first = env.reset(410)
        first[:] = 99
        env.step([.1, .2])
        record = env.episode_record()
        np.testing.assert_array_equal(record['metadata']['target_path'], expected['target_path'])
        np.testing.assert_array_equal(record['metadata']['gear_multiplier'], expected['gear_multiplier'])
        np.testing.assert_array_equal(record['policy']['packets'][1, :4], 0)
        record['audit']['transitions'][0]['native_after']['qpos'][:] = 99
        record['policy']['packets'][:] = 99
        record['metadata']['noise'][0][0] = 99
        latest = env.episode_record()
        assert np.max(np.abs(latest['policy']['packets'])) < 99
        assert np.max(np.abs(latest['audit']['transitions'][0]['native_after']['qpos'])) < 99
        assert latest['metadata']['noise'][0][0] == 0
        for private in ('_targets', '_gains', '_noise', '_schedule'):
            assert not getattr(env, private).flags.writeable
    finally:
        env.close()


def test_custom_horizon_does_not_silently_keep_native50_and_terminal_is_strict():
    env = tracking.TrackingDynamicsEpisode(**inputs(51))
    try:
        with pytest.raises(RuntimeError):
            env.step([0., 0.])
        env.reset(410)
        for _ in range(50):
            env.step([0., 0.])
        assert not env.finished
        env.step([0., 0.])
        saved = env.episode_record()
        assert saved['metadata']['horizon'] == 51
        assert saved['metadata']['base_native_horizon'] == 50
        assert saved['metadata']['status'] == 'completed'
        assert saved['policy']['packets'].shape == (52, 8)
        assert saved['policy']['commands'].shape == (51, 2)
        assert len(saved['audit']['decision_states']) == 52
        assert len(saved['audit']['transitions']) == 51
        with pytest.raises(RuntimeError):
            env.step([0., 0.])
        with pytest.raises(RuntimeError):
            env.reset(410)
    finally:
        env.close()
    env.close()
    assert env.metadata()['closed']


def test_nominal_factory_is_independent_and_never_copies_altered_live_plant(episode, monkeypatch):
    episode.reset(410)
    for _ in range(3):
        episode.step([.1, .2])
    assert episode.audit_record()['transition']['actuator_gear'] == 140
    # Model construction must not reset or step an environment.
    from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv
    monkeypatch.setattr(ReacherEnv, 'reset', lambda *a, **kw: pytest.fail('factory reset'))
    monkeypatch.setattr(ReacherEnv, 'step', lambda *a, **kw: pytest.fail('factory step'))
    first, second = tracking.nominal_model(), tracking.nominal_model()
    np.testing.assert_array_equal(first.actuator_gear[:, 0], [200, 200])
    np.testing.assert_array_equal(first.dof_damping, [1, 1, 0, 0])
    first.actuator_gear[:, 0] = 500
    np.testing.assert_array_equal(second.actuator_gear[:, 0], [200, 200])
    assert episode.audit_record()['transition']['actuator_gear'] == 140


@pytest.mark.parametrize('field,value', [
    ('horizon', 0), ('horizon', True), ('horizon', 1.5),
    ('target_path', np.zeros((5, 2))), ('target_path', np.full((6, 2), np.nan)),
    ('target_path', np.full((6, 2), .271)), ('target_path', np.zeros((6, 2), bool)),
    ('gear_multiplier', np.zeros(5)), ('gear_multiplier', -np.ones(5)),
    ('gear_multiplier', np.ones(4)), ('gear_multiplier', np.full(5, np.inf)),
    ('gear_multiplier', np.full(5, np.finfo(float).max)),
    ('noise', np.zeros((5, 3))), ('noise', np.full((5, 2), np.nan)),
    ('sensor_schedule', np.ones(6)), ('sensor_schedule', np.zeros(6, bool)),
    ('sensor_schedule', np.ones(5, bool)),
])
def test_bad_schedules_reject_before_backend_creation(field, value, monkeypatch):
    spec = {**inputs(), field: value}
    monkeypatch.setattr(tracking, '_native_env', lambda *a: pytest.fail('created backend'))
    with pytest.raises(ValueError):
        tracking.TrackingDynamicsEpisode(**spec)


@pytest.mark.parametrize('command', [[np.nan, 0], [np.inf, 0], [1], [True, False], [1j, 0]])
def test_invalid_command_preserves_running_prefix_without_native_attempt(episode, command):
    episode.reset(410)
    before = episode.episode_record()
    with pytest.raises(ValueError):
        episode.step(command)
    assert_same(before, episode.episode_record())


def test_native_baseexception_preserves_original_failure_and_attempted_command(episode, monkeypatch):
    episode.reset(410)
    episode.step([.1, .2])
    original = KeyboardInterrupt('synthetic interrupted native step')

    def fail(*args):
        raise original

    monkeypatch.setattr(episode._env, 'step', fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        episode.step([.3, .4])
    assert caught.value is original
    record = episode.episode_record()
    assert record['metadata']['status'] == 'failed'
    assert record['metadata']['completed_decisions'] == 1
    assert record['metadata']['attempted_transitions'] == 2
    assert record['metadata']['returned_native_transitions'] == 1
    assert record['policy']['packets'].shape == (2, 8)
    assert record['policy']['commands'].shape == (1, 2)
    assert record['audit']['transitions'][-1]['status'] == 'failed'
    np.testing.assert_array_equal(record['audit']['transitions'][-1]['command'], np.float32([.3, .4]))
    with pytest.raises(RuntimeError):
        episode.step([0, 0])


def test_post_simulation_event_failure_keeps_native_reward_and_old_target_evidence(episode, monkeypatch):
    episode.reset(410)
    episode.step([.1, .2])
    original = RuntimeError('synthetic target event failure')

    def fail(*args):
        raise original

    monkeypatch.setattr(episode, '_set_target', fail)
    with pytest.raises(RuntimeError) as caught:
        episode.step([.1, .2])
    assert caught.value is original
    record = episode.episode_record()
    last = record['audit']['transitions'][-1]
    assert last['status'] == 'failed' and last['native_returned']
    assert np.isfinite(last['reward'])
    np.testing.assert_array_equal(last['native_after']['raw_obs'][4:6], last['reward_target'])
    assert record['metadata']['failure']['stage'] == 'next_target'
    assert record['metadata']['returned_native_transitions'] == 2
    assert record['metadata']['completed_decisions'] == 1


def test_malformed_native_info_retains_actual_return_before_reward_parsing(episode, monkeypatch):
    episode.reset(410)
    native_step = episode._env.step

    def malformed_info(command):
        raw, reward, terminated, truncated, info = native_step(command)
        del info['reward_ctrl']
        return raw, reward, terminated, truncated, info

    monkeypatch.setattr(episode._env, 'step', malformed_info)
    with pytest.raises(KeyError, match='reward_ctrl'):
        episode.step([.1, .2])
    record = episode.episode_record()
    last = record['audit']['transitions'][-1]
    assert last['native_returned'] and last['status'] == 'failed'
    assert last['native_after']['time'] == pytest.approx(.02)
    np.testing.assert_array_equal(last['native_after']['qpos'], episode._env.unwrapped.data.qpos)
    assert record['metadata']['returned_native_transitions'] == 1
    assert record['metadata']['completed_decisions'] == 0
    assert record['metadata']['failure']['type'] == 'KeyError'
    assert record['metadata']['failure']['stage'] == 'reward_record'


def test_failed_reset_and_failed_evidence_capture_do_not_replace_first_error(episode, monkeypatch):
    original = RuntimeError('synthetic reset failure')

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(episode._env, 'reset', fail)
    monkeypatch.setattr(episode, '_state', lambda *a: (_ for _ in ()).throw(ValueError('capture failure')))
    with pytest.raises(RuntimeError) as caught:
        episode.reset(410)
    assert caught.value is original
    saved = episode.episode_record()
    assert saved['policy']['packets'].shape == (0, 8)
    assert saved['metadata']['failure']['state_capture_error'] == 'ValueError'
    assert saved['metadata']['failure']['message'] == str(original)
    with pytest.raises(RuntimeError):
        episode.reset(410)


def test_invalid_seed_and_closed_instance_are_rejected_without_reset(episode):
    with pytest.raises(ValueError):
        episode.reset(True)
    assert episode.metadata()['status'] == 'created'
    episode.close()
    with pytest.raises(RuntimeError):
        episode.reset(410)


def test_external_damping_mutation_fails_instead_of_silently_changing_task(episode):
    episode.reset(410)
    episode._env.unwrapped.model.dof_damping[0] = 2
    with pytest.raises(RuntimeError, match='damping'):
        episode.step([0, 0])
    assert episode.metadata()['status'] == 'failed'
    assert episode.metadata()['returned_native_transitions'] == 0
