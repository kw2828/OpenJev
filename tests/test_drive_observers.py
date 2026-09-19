"""Deterministic analytical fixtures only: no episodes, environments or RNG."""
import copy

import numpy as np
import pytest

from openjev.research.drive_observers import (
    LANDMARK_STD,
    NAMES,
    PROCESS_STD,
    SENSOR_STD,
    Observer,
    joint_odometry,
    joint_propagation,
    joseph_update,
    landmark_prediction,
    odometry_prediction,
    wrap,
)

ANGLES = np.arange(6) * np.pi / 3
LANDMARKS = 5 * np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
INITIAL = np.array([0., 0., np.pi / 4])


def missing():
    return {"values": np.full((6, 2), np.nan), "valid": np.zeros(6, dtype=bool)}


def visible(pose):
    return {"values": landmark_prediction(pose, LANDMARKS)[0], "valid": np.ones(6, dtype=bool)}


def numerical_jacobian(function, value):
    result = []
    for index in range(len(value)):
        delta = np.zeros_like(value)
        delta[index] = 1e-6
        result.append((function(value + delta) - function(value - delta)) / 2e-6)
    return np.stack(result, axis=1)


@pytest.mark.parametrize("n", [3, 5])
def test_odometry_jacobians_finite_difference(n):
    mean = np.array([.3, -.2, .7, .18, -.05])[:n]
    odom = np.array([.9, -.2])
    _, F, G = odometry_prediction(mean, odom)
    np.testing.assert_allclose(F, numerical_jacobian(lambda x: odometry_prediction(x, odom)[0], mean), atol=2e-10)
    np.testing.assert_allclose(G, numerical_jacobian(lambda e: odometry_prediction(mean, odom - e)[0], np.zeros(2)),
                               atol=2e-10)


def test_joint_measurement_and_propagation_jacobians():
    mean = np.array([.2, -.1, .8, .2, -.15, .1, .06, -.02, .04])
    command = np.array([.9, -.4])
    _, H = joint_odometry(mean, command)
    _, J = joint_propagation(mean, command)
    np.testing.assert_allclose(H, numerical_jacobian(lambda x: joint_odometry(x, command)[0], mean), atol=3e-10)
    np.testing.assert_allclose(J, numerical_jacobian(lambda x: joint_propagation(x, command)[0], mean), atol=3e-10)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_landmark_jacobian(n):
    mean = np.zeros(n)
    mean[:3] = [.4, -.3, .3]
    _, H = landmark_prediction(mean, LANDMARKS)
    numerical = numerical_jacobian(lambda x: landmark_prediction(x, LANDMARKS)[0].reshape(-1), mean)
    np.testing.assert_allclose(H, numerical, atol=1e-9)
    assert np.array_equal(H[:, 3:], np.zeros_like(H[:, 3:]))


@pytest.mark.parametrize("name", NAMES)
def test_known_initial_pose_and_parameter_priors(name):
    observer = Observer(name, LANDMARKS, INITIAL)
    np.testing.assert_array_equal(observer.mean[:3], INITIAL)
    np.testing.assert_array_equal(observer.cov[:3], np.zeros_like(observer.cov[:3]))
    np.testing.assert_array_equal(observer.mean[3:], np.zeros_like(observer.mean[3:]))
    if name == "joint_ekf":
        np.testing.assert_array_equal(np.diag(observer.cov)[3:], np.array([.2, .2, .15, .12])**2)
    elif name == "calibration_ekf":
        np.testing.assert_array_equal(np.diag(observer.cov)[3:], np.array([.15, .12])**2)
    else:
        assert not observer.cov.any()


def test_dr_inputs_are_distinct_and_pose_ekf_uses_odometry():
    estimates = {}
    for name in ("command_dr", "odom_dr", "pose_ekf"):
        estimates[name] = Observer(name, LANDMARKS, [0., 0., 0.]).step([1., .2], [.5, -.1], missing())
    np.testing.assert_allclose(estimates["command_dr"], [.1, 0., .02], atol=1e-14)
    np.testing.assert_allclose(estimates["odom_dr"], [.05, 0., -.01], atol=1e-14)
    np.testing.assert_array_equal(estimates["odom_dr"], estimates["pose_ekf"])


def test_odom_covariance_does_not_double_count_actual_velocity_noise():
    first = Observer("pose_ekf", LANDMARKS, [0., 0., 0.], process_std=[0., 0.])
    second = Observer("pose_ekf", LANDMARKS, [0., 0., 0.], process_std=[10., 10.])
    for observer in (first, second):
        observer.step([1., 0.], [1., 0.], missing())
    np.testing.assert_array_equal(first.cov, second.cov)
    np.testing.assert_allclose(np.diag(first.cov), [.1**2 * .03**2, 0., .1**2 * .03**2])


def test_joint_conditioned_process_noise_matches_scalar_gaussian_reference():
    observer = Observer("parameter_oracle", LANDMARKS, [0., 0., 0.])
    pose = observer.step([1., 0.], [1.2, .05], missing(), oracle_params=np.zeros(4))
    np.testing.assert_allclose(pose, [.11, 0., .0025], atol=1e-14)
    np.testing.assert_allclose(observer.diagnostics["conditioned_rates"], [1.1, .025], atol=1e-14)
    np.testing.assert_allclose(np.diag(observer.cov)[:3], [.1**2 * .03**2 / 2, 0., .1**2 * .03**2 / 2])
    assert not observer.cov[3:].any()
    # Scale changes both the process-to-sensor gain and measurement covariance.
    other = Observer("parameter_oracle", LANDMARKS, [0., 0., 0.])
    other.step([1., 0.], [2.2, .05], missing(), oracle_params=[0., 0., np.log(2), 0.])
    expected_eta = .03**2 * 2 / (4 * .03**2 + .03**2) * .2
    assert other.mean[0] == pytest.approx(.1 * (1 + expected_eta))


def test_oracle_parameters_remain_exact_through_landmarks_and_switch():
    observer = Observer("parameter_oracle", LANDMARKS, INITIAL)
    for parameters in ([.2, -.1, .15, .06], [-.2, .1, .15, .06]):
        public_odom = [1., .3]
        observer.step([.8, .2], public_odom, visible(INITIAL), oracle_params=parameters)
        np.testing.assert_array_equal(observer.mean[3:], parameters)
        np.testing.assert_array_equal(observer.cov[3:], np.zeros((4, 7)))
        np.testing.assert_array_equal(observer.cov[:, 3:], np.zeros((7, 4)))
    assert np.linalg.norm(observer.mean[:2] - INITIAL[:2]) > 0  # No true-pose reset.


def test_noiseless_oracle_matches_three_hand_euler_steps():
    parameters = np.array([.2, -.1, .15, .06])
    observer = Observer("parameter_oracle", LANDMARKS, INITIAL,
                        sensor_std=[0., 0.], process_std=[0., 0.], landmark_std=[0., 0.])
    pose = INITIAL.copy()
    for command in ([.5, .2], [.6, -.1], [.7, .3]):
        rates = np.exp(parameters[:2]) * command
        odom = np.array([np.exp(parameters[2]) * rates[0], rates[1] + parameters[3]])
        pose += .1 * np.array([rates[0] * np.cos(pose[2]), rates[0] * np.sin(pose[2]), rates[1]])
        pose[2] = wrap(pose[2])
        result = observer.step(command, odom, visible(pose), oracle_params=parameters)
        np.testing.assert_allclose(result, pose, atol=1e-14)
        assert not observer.cov.any()


def test_calibration_random_walk_is_added_once_per_interval():
    observer = Observer("calibration_ekf", LANDMARKS, INITIAL, parameter_prior_std=[0.] * 4)
    observer.step([0., 0.], [0., 0.], missing())
    np.testing.assert_allclose(np.diag(observer.cov)[3:], [.003**2, .003**2])
    joint = Observer("joint_ekf", LANDMARKS, INITIAL, parameter_prior_std=[0.] * 4)
    joint.step([0., 0.], [0., 0.], missing())
    np.testing.assert_allclose(np.diag(joint.cov)[3:], np.array([.015, .015, .003, .003])**2)


def test_joseph_covariance_and_singular_consistency():
    mean = np.array([0., 0., .2])
    P = np.diag([.2, .3, .4])
    H = np.array([[1., .3, .1], [.2, 1., -.1]])
    R = np.diag([.05, .04])
    innovation = np.array([.1, -.2])
    result, covariance = joseph_update(mean, P, innovation, H, R)
    expected_covariance = np.linalg.inv(np.linalg.inv(P) + H.T @ np.linalg.inv(R) @ H)
    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-15)
    np.testing.assert_allclose(result, mean + expected_covariance @ H.T @ np.linalg.solve(R, innovation))
    _, zero = joseph_update(mean, np.zeros((3, 3)), np.zeros(2), H, np.zeros((2, 2)))
    assert not zero.any()
    with pytest.raises(ValueError, match="deterministic"):
        joseph_update(mean, np.zeros((3, 3)), [.1, 0.], H, np.zeros((2, 2)))


def test_bearing_innovation_wraps_across_branch_cut():
    observer = Observer("pose_ekf", LANDMARKS, [0., 0., 1e-5])
    packet = visible([0., 0., -1e-5])
    observer.step([0., 0.], [0., 0.], packet)
    assert abs(observer.mean[2]) < 1e-4
    assert -np.pi <= observer.mean[2] < np.pi


@pytest.mark.parametrize("name", ["pose_ekf", "calibration_ekf", "joint_ekf"])
def test_landmark_update_reduces_pose_uncertainty_and_psd(name):
    missing_observer = Observer(name, LANDMARKS, INITIAL)
    measured_observer = Observer(name, LANDMARKS, INITIAL)
    target = odometry_prediction(INITIAL, [.8, .1])[0]
    missing_observer.step([.7, .2], [.8, .1], missing())
    measured_observer.step([.7, .2], [.8, .1], visible(target))
    assert np.trace(measured_observer.cov[:3, :3]) < np.trace(missing_observer.cov[:3, :3])
    assert np.linalg.eigvalsh(measured_observer.cov).min() >= -1e-14


def test_defensive_ownership_and_failure_atomicity():
    landmarks, initial = LANDMARKS.copy(), INITIAL.copy()
    observer = Observer("joint_ekf", landmarks, initial)
    landmarks[:] = initial[:] = 999.
    command, odom, packet = np.array([.5, .2]), np.array([.6, .3]), visible(INITIAL)
    saved = copy.deepcopy((command, odom, packet))
    pose = observer.step(command, odom, packet)
    np.testing.assert_array_equal(command, saved[0])
    np.testing.assert_array_equal(odom, saved[1])
    np.testing.assert_array_equal(packet["values"], saved[2]["values"])
    old_mean, old_covariance = observer.mean, observer.cov
    pose[:] = observer.mean[:] = observer.cov[:] = 999.
    observer.diagnostics["conditioned_rates"][0] = 999.
    np.testing.assert_array_equal(observer.mean, old_mean)
    np.testing.assert_array_equal(observer.cov, old_covariance)
    bad = missing()
    bad["values"][0] = 0.
    with pytest.raises(ValueError):
        observer.step(command, odom, bad)
    np.testing.assert_array_equal(observer.mean, old_mean)
    np.testing.assert_array_equal(observer.cov, old_covariance)
    assert observer.diagnostics["steps"] == 1


@pytest.mark.parametrize("name", ["pose_ekf", "calibration_ekf", "joint_ekf", "parameter_oracle"])
def test_signed_gaussian_range_is_assimilated_without_clipping(name):
    initial = np.array([4.89, 0., 0.])  # Positive physical range .11m to landmark0.
    signed = missing()
    signed["valid"][0] = True
    signed["values"][0] = [-.03, 0.]
    clipped = copy.deepcopy(signed)
    clipped["values"][0, 0] = 0.
    kwargs = {"oracle_params": np.zeros(4)} if name == "parameter_oracle" else {}
    observer = Observer(name, LANDMARKS, initial)
    control = Observer(name, LANDMARKS, initial)
    actual = observer.step([0., 0.], [0., 0.], signed, **kwargs)
    clipped_result = control.step([0., 0.], [0., 0.], clipped, **kwargs)
    assert np.isfinite(actual).all() and np.isfinite(observer.cov).all()
    assert actual[0] > clipped_result[0] > initial[0]
    assert signed["values"][0, 0] == -.03
    assert observer.diagnostics["landmark_updates"] == 1


@pytest.mark.parametrize("kind", ["private", "nan", "mask", "metadata", "missing", "bearing", "overflow"])
def test_invalid_boundary_rejected(kind):
    observer = Observer("joint_ekf", LANDMARKS, INITIAL)
    packet, command, odom, params = missing(), [1., .1], [1., .1], None
    if kind == "private":
        params = np.zeros(4)
    elif kind == "nan":
        odom[0] = float("nan")
    elif kind == "mask":
        packet["valid"] = np.zeros(6, dtype=int)
    elif kind == "metadata":
        packet["true_pose"] = INITIAL
    elif kind == "missing":
        packet["values"][0] = 0.
    elif kind == "bearing":
        packet["valid"][0] = True
        packet["values"][0] = [1., np.pi + .01]
    else:
        with pytest.raises(FloatingPointError):
            joint_odometry([0., 0., 0., 1000., 0., 0., 0., 0., 0.], [1., 1.])
        return
    with pytest.raises(ValueError):
        observer.step(command, odom, packet, params)
    assert observer.diagnostics["steps"] == 0


def test_bad_configuration_and_singular_landmark():
    with pytest.raises(ValueError):
        Observer("pose_oracle", LANDMARKS, INITIAL)
    with pytest.raises(ValueError):
        Observer("pose_ekf", LANDMARKS, INITIAL, sensor_std=[-.1, .1])
    with pytest.raises(ValueError):
        landmark_prediction([0., 0., 0.], [[0., 0.]])
    np.testing.assert_array_equal(SENSOR_STD, [.03, .03])
    np.testing.assert_array_equal(PROCESS_STD, [.03, .03])
    np.testing.assert_array_equal(LANDMARK_STD, [.12, .06])
