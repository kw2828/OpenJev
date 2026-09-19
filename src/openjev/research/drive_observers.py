"""Equation-based public observers for Euler differential-drive qualification.

Units: position metres, heading radians, velocities per second, and log gains.
The joint filter conditions temporary physical velocity disturbances on public
odometry before propagating pose. It is a nonlinear EKF approximation, not an
exact Bayesian filter. No environment, random generator or hidden metadata is used.
"""
from __future__ import annotations

import copy

import numpy as np

DT = .1
SENSOR_STD = np.array([.03, .03])
PROCESS_STD = np.array([.03, .03])
LANDMARK_STD = np.array([.12, .06])
PARAMETER_PRIOR_STD = np.array([.2, .2, .15, .12])
PARAMETER_RW_STD = np.array([.015, .015, .003, .003])
NAMES = ("command_dr", "odom_dr", "pose_ekf", "calibration_ekf", "joint_ekf", "parameter_oracle")


def _array(value, shape, name, *, finite=True):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if finite and not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _std(value, length, name):
    result = _array(value, (length,), name)
    if np.any(result < 0):
        raise ValueError(name + " must be nonnegative")
    return result


def _dt(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("dt must be positive and finite")
    return float(value)


def wrap(angle):
    """Principal angle in [-pi, pi); derivatives are local away from the cut."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _exp(value):
    with np.errstate(over="raise", invalid="raise"):
        result = np.exp(value)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite exponential gain")
    return result


def odometry_prediction(mean, odom, dt=DT):
    """Return f,F,G for pose3 or pose3+[kappa,b], with odometry sensor noise.

    G maps [epsilon_v,epsilon_w]. Actual plant velocity noise is already in the
    observed odometry and is not independently added a second time here.
    """
    n = np.asarray(mean).size
    if n not in (3, 5):
        raise ValueError("Odometry state must have3 or5 coordinates")
    mean = _array(mean, (n,), "mean")
    odom = _array(odom, (2,), "odom")
    dt = _dt(dt)
    theta = mean[2]
    scale = _exp(-mean[3]) if n == 5 else 1.
    speed = scale * odom[0]
    turn = odom[1] - (mean[4] if n == 5 else 0.)
    cosine, sine = np.cos(theta), np.sin(theta)
    result = mean.copy()
    result[:3] += dt * np.array([speed * cosine, speed * sine, turn])
    result[2] = wrap(result[2])
    F = np.eye(n)
    F[0, 2], F[1, 2] = -dt * speed * sine, dt * speed * cosine
    if n == 5:
        F[0, 3], F[1, 3], F[2, 4] = -dt * speed * cosine, -dt * speed * sine, -dt
    G = np.zeros((n, 2))
    G[0, 0], G[1, 0], G[2, 1] = -dt * scale * cosine, -dt * scale * sine, -dt
    return result, F, G


def joint_odometry(mean9, command):
    """Return predicted odometry and H for [pose,a,c,kappa,b,eta_v,eta_w]."""
    mean = _array(mean9, (9,), "joint mean")
    command = _array(command, (2,), "command")
    speed_drive, turn_drive = _exp(mean[3:5]) * command
    scale = _exp(mean[5])
    actual_speed, actual_turn = speed_drive + mean[7], turn_drive + mean[8]
    predicted = np.array([scale * actual_speed, actual_turn + mean[6]])
    H = np.zeros((2, 9))
    H[0, 3], H[0, 5], H[0, 7] = scale * speed_drive, scale * actual_speed, scale
    H[1, 4], H[1, 6], H[1, 8] = turn_drive, 1., 1.
    return predicted, H


def joint_propagation(mean9, command, dt=DT):
    """Conditioned9-state to persistent7-state Euler prediction and Jacobian."""
    mean = _array(mean9, (9,), "joint mean")
    command = _array(command, (2,), "command")
    dt = _dt(dt)
    speed_drive, turn_drive = _exp(mean[3:5]) * command
    speed, turn = speed_drive + mean[7], turn_drive + mean[8]
    cosine, sine = np.cos(mean[2]), np.sin(mean[2])
    predicted = mean[:7].copy()
    predicted[:3] += dt * np.array([speed * cosine, speed * sine, turn])
    predicted[2] = wrap(predicted[2])
    J = np.zeros((7, 9))
    J[:, :7] = np.eye(7)
    J[0, 2], J[1, 2] = -dt * speed * sine, dt * speed * cosine
    J[0, 3], J[1, 3] = dt * speed_drive * cosine, dt * speed_drive * sine
    J[2, 4] = dt * turn_drive
    J[0, 7], J[1, 7], J[2, 8] = dt * cosine, dt * sine, dt
    return predicted, J


def landmark_prediction(mean, landmarks):
    """Range/bearing in supplied landmark order, and stacked analytic H."""
    n = np.asarray(mean).size
    if n not in (3, 5, 7):
        raise ValueError("Landmark state must have3,5 or7 coordinates")
    mean = _array(mean, (n,), "mean")
    raw = np.asarray(landmarks)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("Landmarks require shape [L,2]")
    points = _array(landmarks, raw.shape, "landmarks")
    difference = points - mean[:2]
    squared = np.sum(difference**2, axis=1)
    if np.any(squared <= 1e-18):
        raise ValueError("Bearing Jacobian undefined within1e-9m of landmark")
    radius = np.sqrt(squared)
    predicted = np.column_stack((radius, wrap(np.arctan2(difference[:, 1], difference[:, 0]) - mean[2])))
    H = np.zeros((2 * len(points), n))
    H[0::2, :2] = -difference / radius[:, None]
    H[1::2, 0] = difference[:, 1] / squared
    H[1::2, 1] = -difference[:, 0] / squared
    H[1::2, 2] = -1.
    return predicted, H


def _covariance(value, n, name):
    covariance = _array(value, (n, n), name)
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError(name + " is not symmetric")
    covariance = (covariance + covariance.T) * .5
    scale = max(1., float(np.linalg.norm(covariance, ord=2)))
    if np.linalg.eigvalsh(covariance).min() < -1e-12 * scale:
        raise ValueError(name + " is not positive semidefinite")
    return covariance


def joseph_update(mean, covariance, innovation, H, R):
    """Functional EKF update; singular deterministic fixtures use a PSD solve.

    No diagonal clipping or extra ridge is introduced. A zero-variance direction
    with incompatible innovation raises instead of pretending it was observed.
    """
    n = np.asarray(mean).size
    mean = _array(mean, (n,), "mean")
    covariance = _covariance(covariance, n, "covariance")
    m = np.asarray(innovation).size
    innovation = _array(innovation, (m,), "innovation")
    H = _array(H, (m, n), "H")
    R = _covariance(R, m, "R")
    S = H @ covariance @ H.T + R
    S = (S + S.T) * .5
    # Positive measurement noise in production makes S strictly positive.
    if np.linalg.eigvalsh(R).min() > 0:
        K = np.linalg.solve(S, H @ covariance).T
    else:
        eigenvalues, vectors = np.linalg.eigh(S)
        threshold = np.finfo(np.float64).eps * max(1, m) * max(float(eigenvalues.max()), 1e-30) * 16
        if eigenvalues.min() < -threshold:
            raise ValueError("Innovation covariance is not positive semidefinite")
        keep = eigenvalues > threshold
        outside = vectors[:, ~keep].T @ innovation
        if np.linalg.norm(outside) > 1e-10 * (1 + np.linalg.norm(innovation)):
            raise ValueError("Incompatible deterministic observation")
        inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
        K = covariance @ H.T @ inverse
    result = mean + K @ innovation
    result[2] = wrap(result[2])
    A = np.eye(n) - K @ H
    updated = A @ covariance @ A.T + K @ R @ K.T
    updated = _covariance((updated + updated.T) * .5, n, "updated covariance")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite updated state")
    return result, updated


class Observer:
    """One public causal observer; parameter_oracle alone accepts private parameters.

    Each step is atomic: invalid measurements or numerical failures leave the
    previous persistent mean/covariance unchanged. Returned arrays are owned.
    """

    def __init__(self, name, landmarks, initial_pose, *, dt=DT, sensor_std=SENSOR_STD,
                 process_std=PROCESS_STD, landmark_std=LANDMARK_STD,
                 parameter_prior_std=PARAMETER_PRIOR_STD, parameter_rw_std=PARAMETER_RW_STD):
        if name not in NAMES:
            raise ValueError("Unknown public observer; pose_oracle belongs to the evaluator")
        self.name = name
        self.dt = _dt(dt)
        self._landmarks = _array(landmarks, (6, 2), "landmarks")
        if len(np.unique(self._landmarks, axis=0)) != 6:
            raise ValueError("Distinct identified landmarks required")
        self._sensor_std = _std(sensor_std, 2, "sensor_std")
        self._process_std = _std(process_std, 2, "process_std")
        self._landmark_std = _std(landmark_std, 2, "landmark_std")
        prior = _std(parameter_prior_std, 4, "parameter_prior_std")
        self._parameter_rw_std = _std(parameter_rw_std, 4, "parameter_rw_std")
        n = 7 if name in ("joint_ekf", "parameter_oracle") else 5 if name == "calibration_ekf" else 3
        self._mean = np.zeros(n)
        self._mean[:3] = _array(initial_pose, (3,), "initial_pose")
        self._mean[2] = wrap(self._mean[2])
        self._covariance = np.zeros((n, n))
        if name == "joint_ekf":
            self._covariance[3:, 3:] = np.diag(prior**2)
        elif name == "calibration_ekf":
            self._covariance[3:, 3:] = np.diag(prior[2:]**2)
        self._steps = 0
        self._diagnostics = {"steps": 0, "name": name, "landmark_updates": 0}

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    @property
    def cov(self):
        return self.covariance

    @property
    def diagnostics(self):
        return copy.deepcopy(self._diagnostics)

    def step(self, command, odom, packet, oracle_params=None):
        command = _array(command, (2,), "command")
        odom = _array(odom, (2,), "odom")
        if type(packet) is not dict or set(packet) != {"values", "valid"}:
            raise ValueError("Packet must contain only public values and valid")
        values = _array(packet["values"], (6, 2), "landmark values", finite=False)
        valid = np.asarray(packet["valid"])
        if valid.shape != (6,) or valid.dtype != np.bool_:
            raise ValueError("Landmark valid must be bool[6]")
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError("Visible landmark values must be finite; missing values must be NaN")
        # Gaussian additive range noise has signed support, even though the
        # predicted physical distance is positive. Do not truncate observations.
        if np.any(np.abs(values[valid, 1]) > np.pi):
            raise ValueError("Invalid public bearing")
        if self.name != "parameter_oracle" and oracle_params is not None:
            raise ValueError("Private parameters are forbidden for this observer")
        mean, covariance = self.mean, self.covariance
        if self.name in ("joint_ekf", "parameter_oracle"):
            if self.name == "parameter_oracle":
                mean[3:] = _array(oracle_params, (4,), "oracle_params [a,c,kappa,b]")
                covariance[3:, :] = 0.
                covariance[:, 3:] = 0.
            temporary = np.concatenate((mean, np.zeros(2)))
            temporary_cov = np.zeros((9, 9))
            temporary_cov[:7, :7] = covariance
            temporary_cov[7:, 7:] = np.diag(self._process_std**2)
            predicted_odom, H = joint_odometry(temporary, command)
            temporary, temporary_cov = joseph_update(
                temporary, temporary_cov, odom - predicted_odom, H, np.diag(self._sensor_std**2))
            mean, J = joint_propagation(temporary, command, self.dt)
            covariance = J @ temporary_cov @ J.T
            if self.name != "parameter_oracle":
                covariance[3:, 3:] += np.diag(self._parameter_rw_std**2)
            rate_estimate = (_exp(temporary[3:5]) * command + temporary[7:]).tolist()
        else:
            observed = command if self.name == "command_dr" else odom
            mean, F, G = odometry_prediction(mean, observed, self.dt)
            noise = self._process_std if self.name == "command_dr" else self._sensor_std
            covariance = F @ covariance @ F.T + G @ np.diag(noise**2) @ G.T
            if self.name == "calibration_ekf":
                covariance[3:, 3:] += np.diag(self._parameter_rw_std[2:]**2)
            rate_estimate = None
        count = int(valid.sum()) if self.name not in ("command_dr", "odom_dr") else 0
        if count:
            prediction, H = landmark_prediction(mean, self._landmarks[valid])
            residual = values[valid] - prediction
            residual[:, 1] = wrap(residual[:, 1])
            mean, covariance = joseph_update(mean, covariance, residual.reshape(-1), H,
                                             np.diag(np.tile(self._landmark_std**2, count)))
        covariance = _covariance((covariance + covariance.T) * .5, len(mean), "final covariance")
        if not np.isfinite(mean).all():
            raise ValueError("Nonfinite final state")
        self._mean, self._covariance = mean, covariance
        self._steps += 1
        self._diagnostics = {"steps": self._steps, "name": self.name, "landmark_updates": count,
                             "conditioned_rates": rate_estimate,
                             "parameter_order": (["a", "c", "kappa", "b"] if len(mean) == 7 else
                                                 ["kappa", "b"] if len(mean) == 5 else [])}
        return mean[:3].copy()
