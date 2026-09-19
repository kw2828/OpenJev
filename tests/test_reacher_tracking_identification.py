"""Engineering-only tests for public-angle tracking identification, seed410."""

import copy

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")

from openjev.research import reacher_tracking_identification as identification
from openjev.research.robotics_reacher import make_env, packet


@pytest.fixture
def nominal():
    env = make_env()
    env.reset(seed=410)
    model = copy.copy(env.unwrapped.model)
    env.close()
    return model


def public(angles=(.1, -.1), target=(.12, -.04)):
    return packet(angles, target)


def identifier(nominal, **overrides):
    kwargs = {"gains": np.array([.5, 1., 1.5]), "window": 3,
              "prior_gain": 1., "frame_skip": 2}
    kwargs.update(overrides)
    return identification.WindowedGainIdentifier(nominal, public(), **kwargs)


def test_three_point_derivative_recovers_quadratic_endpoint_velocity():
    dt = .02
    q0, velocity, acceleration = np.array([.1, -.1]), np.array([.4, -.3]), np.array([.2, -.1])
    observer = identification.TrackingAngleObserver(public(q0), dt=dt)
    np.testing.assert_array_equal(observer.estimate()[1], np.zeros(4))
    for step in range(1, 6):
        t = step * dt
        qpos, qvel = observer.update(public(q0 + velocity*t + .5*acceleration*t*t))
        expected = velocity + acceleration * (t if step > 1 else t / 2)
        np.testing.assert_allclose(qpos[:2], q0 + velocity*t + .5*acceleration*t*t, atol=1e-8)
        np.testing.assert_allclose(qvel[:2], expected, atol=1e-6)
    assert observer.snapshot()["packets"].shape == (3, 8)


@pytest.mark.parametrize("sign", [-1, 1])
def test_wrap_boundary_preserves_continuous_angle_branch(sign):
    observer = identification.TrackingAngleObserver(public(np.array([3.12, 3.12])*sign), dt=.02)
    for value in [3.15, 3.18]:
        qpos, qvel = observer.update(public(np.array([value, value])*sign))
        np.testing.assert_allclose(qpos[:2], value*sign, atol=1e-7)
        np.testing.assert_allclose(qvel[:2], 1.5*sign, atol=1e-5)


def test_target_change_does_not_change_arm_estimate():
    observers = [identification.TrackingAngleObserver(public(), dt=.02) for _ in range(2)]
    same = observers[0].update(public((.11, -.12)))
    changed = observers[1].update(public((.11, -.12), (-.13, .08)))
    np.testing.assert_array_equal(same[0][:2], changed[0][:2])
    np.testing.assert_array_equal(same[1], changed[1])
    np.testing.assert_array_equal(changed[0][2:], public(target=(-.13, .08))[4:6])


def test_input_and_returned_arrays_are_isolated(nominal):
    initial, gains = public(), np.array([.5, 1., 1.5])
    model = identification.WindowedGainIdentifier(nominal, initial, gains=gains,
        window=3, prior_gain=1., frame_skip=2)
    initial[:] = 0
    gains[:] = 99
    nominal.actuator_gear[:] = 7
    command, next_packet = np.array([.001, -.002]), public((.11, -.12))
    model.update(command, next_packet)
    before = model.snapshot()
    command[:] = 1
    next_packet[:] = 0
    model.estimate()[0][:] = 12
    model.snapshot()["last_trace"]["candidate_predictions"][:] = 18
    np.testing.assert_array_equal(model.snapshot()["last_trace"]["candidate_predictions"],
                                  before["last_trace"]["candidate_predictions"])
    np.testing.assert_array_equal(model.configuration()["gains"], [.5, 1., 1.5])
    assert model.snapshot()["last_trace"]["next_packet"][6] == 1


def test_gain_bank_predictions_match_independent_native_steps(nominal):
    model = identifier(nominal)
    original_gear = nominal.actuator_gear.copy()
    qpos, qvel, _ = model.estimate()
    command = np.array([.004, -.006])
    expected = []
    for gain in [.5, 1., 1.5]:
        native = copy.copy(nominal)
        native.actuator_gear[:] = original_gear * gain
        data = mujoco.MjData(native)
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(native, data)
        data.ctrl[:] = command.astype(np.float32).astype(np.float64)
        mujoco.mj_step(native, data, nstep=2)
        mujoco.mj_rnePostConstraint(native, data)
        expected.append(np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])])
    next_packet = public((.101, -.102), (-.14, .02))
    model.update(command, next_packet)
    trace = model.snapshot()["last_trace"]
    np.testing.assert_array_equal(trace["candidate_predictions"], expected)
    np.testing.assert_array_equal(trace["candidate_residuals"],
        np.mean((np.array(expected)-next_packet[:4])**2, axis=1))
    np.testing.assert_array_equal(nominal.actuator_gear, original_gear)
    np.testing.assert_array_equal(trace["root_qpos"][2:], public()[4:6])
    np.testing.assert_array_equal(model.estimate()[0][2:], next_packet[4:6])
    costs = model.snapshot()["costs"]
    assert costs["native_transitions_completed"] == 3
    assert costs["native_substeps_completed"] == 6
    assert costs["root_forward_calls_completed"] == 3


def test_flat_zero_action_bank_retains_prior_and_claims_no_confidence(nominal):
    model = identifier(nominal)
    for _ in range(4):
        model.update([0, 0], public())
        trace = model.snapshot()["last_trace"]
        assert trace["exactly_flat_bank"] is True
        assert trace["residual_spread"] == 0.
        assert model.estimate()[2] == 1.
    assert model.configuration()["residual_spread_is_confidence"] is False


def test_window_uses_only_declared_number_of_completed_transitions(nominal):
    model = identifier(nominal, window=2)
    all_residuals = []
    for step in range(5):
        model.update([.002, -.003], public((.1+step*.01, -.1-step*.01)))
        trace = model.snapshot()["last_trace"]
        all_residuals.append(trace["candidate_residuals"])
        np.testing.assert_array_equal(trace["window_residual_sums"], np.sum(all_residuals[-2:], axis=0))
    assert model.snapshot()["window_residuals"].shape == (2, 3)
    assert model.snapshot()["costs"]["retained_residual_scalars"] == 6


def test_commands_are_clipped_and_float32_quantized(nominal):
    models = [identifier(nominal) for _ in range(2)]
    models[0].update([5., -9.], public((.2, -.3)))
    models[1].update([1., -1.], public((.2, -.3)))
    traces = [model.snapshot()["last_trace"] for model in models]
    np.testing.assert_array_equal(traces[0]["candidate_predictions"], traces[1]["candidate_predictions"])
    assert traces[0]["issued_command"].dtype == np.float32


@pytest.mark.parametrize("kind", ["hidden", "poison_hidden", "nan", "circle", "age", "command"])
def test_bad_public_update_rejected_before_native_work(nominal, kind):
    model = identifier(nominal)
    command, value = np.zeros(2), public()
    if kind in ("hidden", "poison_hidden"):
        value[6:] = [0, .02]
        if kind == "hidden":
            value[:4] = 0
    elif kind == "nan":
        value[0] = np.nan
    elif kind == "circle":
        value[0] = 3
    elif kind == "age":
        value[7] = .02
    else:
        command[0] = np.nan
    with pytest.raises(ValueError):
        model.update(command, value)
    assert model.snapshot()["costs"]["native_transition_attempts"] == 0
    assert model.snapshot()["step_index"] == 0
    assert not model.snapshot()["failed"]


@pytest.mark.parametrize("kwargs", [
    {"gains": [1.]}, {"gains": [[.5, 1.]]}, {"gains": [.5, .5, 1.]},
    {"gains": [1., .5]}, {"gains": [-1., 1.]}, {"gains": [.5, np.inf]},
    {"prior_gain": .9}, {"window": 0}, {"window": True}, {"frame_skip": 0},
])
def test_invalid_configuration(nominal, kwargs):
    with pytest.raises(ValueError):
        identifier(nominal, **kwargs)


def test_current_hidden_gain_model_is_rejected(nominal):
    nominal.actuator_gear *= .7
    with pytest.raises(ValueError, match="nominal"):
        identifier(nominal)
    with pytest.raises(TypeError):
        identifier(mujoco.MjData(nominal))


def test_native_failure_records_attempt_and_is_terminal(nominal, monkeypatch):
    model = identifier(nominal)
    def fail(*args):
        raise RuntimeError("engineering injected native failure")
    monkeypatch.setattr(identification, "_step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model.update([.01, 0.], public())
    receipt = model.snapshot()
    assert receipt["failed"]
    assert receipt["costs"]["native_transition_attempts"] == 1
    assert receipt["costs"]["native_transitions_completed"] == 0
    assert receipt["costs"]["native_substeps_requested"] == 2
    assert receipt["step_index"] == 0
    with pytest.raises(RuntimeError, match="terminal"):
        model.estimate()
    with pytest.raises(RuntimeError, match="terminal"):
        model.update([0, 0], public())
