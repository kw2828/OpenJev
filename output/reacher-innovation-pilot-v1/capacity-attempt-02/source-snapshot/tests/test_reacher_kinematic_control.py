"""Synthetic public-input physics reference checks; only engineering seed410."""

import copy

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")

from openjev.research import reacher_kinematic_control as kinematic
from openjev.research.reacher_physics_control import PhysicsMPC
from openjev.research.robotics_reacher import make_env, packet


@pytest.fixture
def env():
    result = make_env()
    result.reset(seed=410)
    yield result
    result.close()


def initial(angles=(.1, -.1), target=(.12, -.04)):
    return packet(angles, target)


def hidden(public, age):
    return packet([0., 0.], public[4:6], valid=False, age_seconds=age)


def advance_native(model, data, command, skip):
    data.ctrl[:] = np.clip(np.asarray(command, dtype=np.float64), -1, 1).astype(np.float32)
    mujoco.mj_step(model, data, nstep=skip)
    mujoco.mj_rnePostConstraint(model, data)


def test_initialization_uses_only_public_angles_static_target_and_zero_velocity(env):
    public = initial()
    copy_packet = public.copy()
    observer = kinematic.KinematicObserver(env.unwrapped.model, public)
    qpos, qvel = observer.estimate()
    np.testing.assert_allclose(qpos[:2], [.1, -.1], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(qpos[2:], public[4:6])
    np.testing.assert_array_equal(qvel, np.zeros(4))
    np.testing.assert_array_equal(public, copy_packet)
    assert observer.configuration()["privileged_live_state"] is False
    assert observer.configuration()["learned"] is False
    assert observer.snapshot()["costs"]["native_transitions_completed"] == 0
    for wrong in (env, env.unwrapped.data):
        with pytest.raises(TypeError, match="never an environment or MjData"):
            kinematic.KinematicObserver(wrong, public)


@pytest.mark.parametrize("start,end", [(3.12, 3.2), (-3.12, -3.2)])
def test_wrapped_delta_and_soft_elbow_stop_branch_preserved(env, start, end):
    public = initial((start, start))
    observer = kinematic.KinematicObserver(env.unwrapped.model, public)
    measured = packet([end, end], public[4:6])
    qpos, qvel = observer.update([0., 0.], measured)
    np.testing.assert_allclose(qpos[:2], [end, end], atol=5e-8, rtol=0)
    # Independent circular delta, not a principal-angle subtraction across the cut.
    delta = np.arctan2(np.sin(end - start), np.cos(end - start))
    np.testing.assert_allclose(qvel[:2], delta / .02, atol=2e-6, rtol=0)
    assert np.all(np.sign(qpos[:2]) == np.sign(end))


def test_velocity_uses_elapsed_time_between_valid_packets_not_missing_count_or_last_dt(env):
    public = initial((0., 0.))
    observer = kinematic.KinematicObserver(env.unwrapped.model, public)
    observer.update([.01, 0.], hidden(public, .02))
    observer.update([0., -.01], hidden(public, .04))
    measured = packet([.12, -.06], public[4:6])
    _, velocity = observer.update([0., 0.], measured)
    np.testing.assert_allclose(velocity[:2], [2., -1.], atol=1e-7, rtol=0)
    assert observer.snapshot()["last_velocity_interval_seconds"] == pytest.approx(.06)
    _, velocity = observer.update([0., 0.], packet([.14, -.08], public[4:6]))
    np.testing.assert_allclose(velocity[:2], [1., -1.], atol=5e-7, rtol=0)
    saved = observer.snapshot()
    assert [m["step"] for m in saved["valid_measurements"]] == [3, 4]
    assert saved["costs"]["measurement_reanchor_forward_calls"] == 2


def test_blackout_propagation_matches_private_nominal_native_dynamics(env):
    public = initial()
    observer = kinematic.KinematicObserver(env.unwrapped.model, public, frame_skip=3)
    model = copy.copy(env.unwrapped.model)
    reference = mujoco.MjData(model)
    qpos, qvel = observer.estimate()
    reference.qpos[:], reference.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, reference)
    commands = np.random.default_rng(410).uniform(-.1, .1, (4, 2))
    for step, command in enumerate(commands, 1):
        advance_native(model, reference, command, 3)
        qpos, qvel = observer.update(command, hidden(public, step * .03))
        np.testing.assert_array_equal(qpos, reference.qpos)
        np.testing.assert_array_equal(qvel, reference.qvel)
    saved = observer.snapshot()
    assert saved["step_index"] == 4
    assert saved["costs"]["native_transitions_completed"] == 4
    assert saved["costs"]["native_substeps_completed"] == 12
    assert saved["last_valid_step"] == 0


def test_second_observation_after_blackout_uses_measurements_not_predicted_displacement(env):
    public = initial((0., 0.))
    observers = [kinematic.KinematicObserver(env.unwrapped.model, public) for _ in range(2)]
    for obs, sign in zip(observers, (-1, 1), strict=True):
        obs.update([sign, -.5 * sign], hidden(public, .02))
        obs.update([sign, -.5 * sign], hidden(public, .04))
    assert not np.allclose(observers[0].estimate()[0], observers[1].estimate()[0])
    measurement = packet([.03, -.06], public[4:6])
    velocities = [obs.update([0, 0], measurement)[1] for obs in observers]
    np.testing.assert_array_equal(*velocities)
    np.testing.assert_allclose(velocities[0][:2], [.5, -1], atol=1e-7, rtol=0)


@pytest.mark.parametrize("kind", ["nan_angles", "hidden_angles", "age", "target", "valid_age", "circle", "command"])
def test_invalid_update_rejected_before_propagation_without_hidden_state_leakage(env, kind):
    public = initial()
    observer = kinematic.KinematicObserver(env.unwrapped.model, public)
    bad = hidden(public, .02)
    command = [0., 0.]
    if kind == "nan_angles":
        bad[0] = np.nan
    elif kind == "hidden_angles":
        bad[1] = .4
    elif kind == "age":
        bad[7] = .04
    elif kind == "target":
        bad[4] += .01
    elif kind == "valid_age":
        bad = public.copy()
        bad[7] = .02
    elif kind == "circle":
        bad = public.copy()
        bad[0] = 3.
    else:
        command = [np.nan, 0.]
    before = observer.snapshot()
    with pytest.raises(ValueError):
        observer.update(command, bad)
    after = observer.snapshot()
    np.testing.assert_array_equal(before["qpos_estimate"], after["qpos_estimate"])
    np.testing.assert_array_equal(before["qvel_estimate"], after["qvel_estimate"])
    assert after["costs"] == before["costs"]
    assert not after["failed"]


def test_no_input_output_model_or_live_data_aliasing(env):
    public = initial()
    observer = kinematic.KinematicObserver(env.unwrapped.model, public)
    reference = kinematic.KinematicObserver(env.unwrapped.model, public.copy())
    model_mass = env.unwrapped.model.body_mass.copy()
    env.unwrapped.model.body_mass[:] *= 2
    env.unwrapped.model.opt.timestep *= 2
    env.unwrapped.data.qpos[:] = [2, -2, .1, .1]
    env.unwrapped.data.qvel[:] = [10, -10, 0, 0]
    public[:] = 17
    snapshot = observer.snapshot()
    snapshot["qpos_estimate"][:] = 22
    snapshot["valid_measurements"][0]["packet"][:] = 22
    snapshot["configuration"]["gravity"][:] = [0, 0, 0]
    a, b = observer.estimate()
    a[:] = b[:] = 23
    public = initial()
    command = np.array([.03, -.05])
    copies = command.copy(), hidden(public, .02)
    for left, right in zip(observer.update(command, copies[1]), reference.update(command, copies[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(command, copies[0])
    np.testing.assert_array_equal(observer._model.body_mass, model_mass)
    assert observer.dt == .02
    # Nominal prediction never writes back to the live simulator.
    np.testing.assert_array_equal(env.unwrapped.data.qpos, [2, -2, .1, .1])
    np.testing.assert_array_equal(env.unwrapped.data.qvel, [10, -10, 0, 0])


def test_clipping_and_quantization_match_nominal_reference(env):
    public = initial()
    first = kinematic.KinematicObserver(env.unwrapped.model, public)
    second = kinematic.KinematicObserver(env.unwrapped.model, public)
    for a, b in zip(first.update([100., -100.], hidden(public, .02)),
                    second.update([1., -1.], hidden(public, .02)), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.snapshot()["last_issued_command"], np.array([1, -1], np.float32))


def test_mpc_matches_existing_planner_and_does_not_assimilate_imagined_states(env):
    public = initial()
    control = kinematic.KinematicMPC(env.unwrapped.model, public)
    control.update([.1, -.2], hidden(public, .02))
    qpos, qvel = control.estimate()
    candidates = np.random.default_rng(410).uniform(-.3, .3, (5, 4, 2))
    original = candidates.copy()
    expected = PhysicsMPC(env.unwrapped.model).plan(qpos, qvel, candidates)
    actual = control.plan(candidates)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(candidates, original)
    for before, after in zip((qpos, qvel), control.estimate(), strict=True):
        np.testing.assert_array_equal(before, after)
    cost = control.snapshot()["costs"]
    assert cost["candidate_evaluations_completed"] == 5
    assert cost["native_transitions_completed"] == 20
    assert cost["native_substeps_completed"] == 40
    control.plan(candidates[::-1])
    assert control.snapshot()["costs"]["native_transitions_completed"] == 40
    assert control.snapshot()["observer"]["step_index"] == 1
    action, costs = control.plan(np.zeros((2, 2, 2)))
    np.testing.assert_array_equal(action, [0., 0.])
    assert costs[0] == costs[1]


@pytest.mark.parametrize("value", [[], np.zeros((0, 2, 2)), np.zeros((2, 0, 2)), np.zeros((2, 2, 3)), np.full((1, 1, 2), np.nan)])
def test_invalid_planning_bank_does_not_change_observer_or_cost_counters(env, value):
    control = kinematic.KinematicMPC(env.unwrapped.model, initial())
    before = control.snapshot()["costs"]
    with pytest.raises(ValueError):
        control.plan(value)
    assert control.snapshot()["costs"] == before


def test_native_failure_is_terminal_and_records_requested_work(env, monkeypatch):
    observer = kinematic.KinematicObserver(env.unwrapped.model, initial())

    def fail(*args):
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(kinematic, "_step", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        observer.update([0., 0.], hidden(initial(), .02))
    saved = observer.snapshot()
    assert saved["failed"]
    assert saved["costs"]["native_transition_attempts"] == 1
    assert saved["costs"]["native_transitions_completed"] == 0
    with pytest.raises(RuntimeError, match="terminal"):
        observer.estimate()
    with pytest.raises(RuntimeError, match="terminal"):
        observer.update([0., 0.], hidden(initial(), .02))


def test_failed_plan_retains_observer_and_requested_work_without_claiming_completed_cost(env, monkeypatch):
    control = kinematic.KinematicMPC(env.unwrapped.model, initial())
    before = control.estimate()

    def fail(*args):
        raise KeyboardInterrupt("synthetic interrupted plan")

    monkeypatch.setattr(control._planner, "plan", fail)
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        control.plan(np.zeros((3, 4, 2)))
    saved = control.snapshot()
    assert saved["failed"]
    assert saved["costs"]["native_transitions_requested"] == 12
    assert saved["costs"]["native_substeps_requested"] == 24
    assert saved["costs"]["native_transitions_completed"] == 0
    for a, b in zip(before, control.estimate(), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(RuntimeError, match="terminal"):
        control.plan(np.zeros((3, 4, 2)))
    with pytest.raises(RuntimeError, match="terminal"):
        control.update([0, 0], hidden(initial(), .02))


@pytest.mark.parametrize("skip", [0, -1, 1.5, True])
def test_invalid_frame_skip_rejected(env, skip):
    with pytest.raises(ValueError, match="frame_skip"):
        kinematic.KinematicObserver(env.unwrapped.model, initial(), frame_skip=skip)


def test_initial_packet_requires_valid_measurement(env):
    with pytest.raises(ValueError, match="Initial packet"):
        kinematic.KinematicObserver(env.unwrapped.model, hidden(initial(), 0.))


def test_configuration_delegates_run_status_to_protocol_and_receipts(env):
    value = kinematic.KinematicObserver(env.unwrapped.model, initial()).configuration()
    assert value["run_status_authority"] == "enclosing protocol and execution receipts"
    assert "engineering_only" not in value
