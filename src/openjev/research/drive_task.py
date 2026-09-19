"""Independent equation-based mobile-robot qualification, not paper replication.

Private noise/parameters remain in the evaluator. Observers receive issued
commands, interval odometry and timestamp-aligned endpoint landmark packets.
"""

import time

import numpy as np

DT = .1
HORIZON = 300
LANDMARKS = 5 * np.column_stack((np.cos(np.arange(6) * np.pi / 3),
                                np.sin(np.arange(6) * np.pi / 3)))
INITIAL_POSE = np.array([0., 0., np.pi / 4])
PANELS = ("stationary", "gain_switch", "gain_switch_long_gap")
METHODS = ("command_dr", "odom_dr", "pose_ekf", "calibration_ekf", "joint_ekf",
           "parameter_oracle", "pose_oracle")
SEEDS = tuple(range(24101, 24113))


class EpisodeFailure(RuntimeError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace
        self.native_steps = len(trace["poses"])


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reference(time):
    phase = .22 * time
    position = np.array([3 * np.sin(phase), 1.5 * np.sin(2 * phase)])
    velocity = np.array([.66 * np.cos(phase), .66 * np.cos(2 * phase)])
    acceleration = np.array([-.1452 * np.sin(phase), -.2904 * np.sin(2 * phase)])
    heading_rate = (velocity[0] * acceleration[1] - velocity[1] * acceleration[0]) / np.dot(velocity, velocity)
    return position, velocity, heading_rate


def command(pose, time):
    """Same clocked path controller for every observer, with no private progress."""
    position, velocity, heading_rate = reference(time)
    desired_velocity = velocity + 1.2 * (position - pose[:2])
    direction = np.array([np.cos(pose[2]), np.sin(pose[2])])
    speed = np.clip(desired_velocity @ direction, 0., 1.5)
    heading = np.arctan2(desired_velocity[1], desired_velocity[0])
    turn = np.clip(heading_rate + 3 * wrap(heading - pose[2]), -3., 3.)
    return np.array([speed, turn])


def motion(pose, rates, dt=DT):
    return np.array([pose[0] + dt * rates[0] * np.cos(pose[2]),
                     pose[1] + dt * rates[0] * np.sin(pose[2]),
                     wrap(pose[2] + dt * rates[1])])


def make_streams(seed):
    """One immutable exogenous stream shared by all methods and three panels."""
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(6)]
    initial = np.concatenate((np.clip(streams[0].normal(0, .15, 2), -.3, .3),
                              [np.clip(streams[0].normal(0, .1), -.2, .2),
                               np.clip(streams[0].normal(0, .08), -.16, .16)]))
    changed = initial.copy()
    changed[:2] = np.clip(streams[0].normal(0, .15, 2), -.3, .3)
    return {"initial_params": initial, "changed_params": changed,
            "process_noise": streams[1].normal(0, .03, (HORIZON, 2)),
            "odometry_noise": streams[2].normal(0, .03, (HORIZON, 2)),
            "landmark_noise": streams[3].normal(size=(HORIZON, 6, 2)) * np.array([.12, .06]),
            "outage_phase": np.array(int(streams[4].integers(0, 45))),
            "switch_step": np.array(int(streams[5].integers(120, 181)))}


def private_parameters(streams, panel, step):
    if panel not in PANELS:
        raise ValueError("Unknown panel")
    return streams["changed_params"] if panel != "stationary" and step >= streams["switch_step"] else streams["initial_params"]


def sense(pose, noise, valid_time):
    displacement = LANDMARKS - pose[:2]
    distance = np.linalg.norm(displacement, axis=1)
    values = np.column_stack((distance, wrap(np.arctan2(displacement[:, 1], displacement[:, 0]) - pose[2])))
    values += noise
    values[:, 1] = wrap(values[:, 1])
    valid = (distance < 6.) & (distance > .1) & valid_time
    values[~valid] = np.nan
    return {"values": values, "valid": valid}


def transition(pose, issued_command, streams, panel, step):
    parameters = private_parameters(streams, panel, step)
    rates = np.exp(parameters[:2]) * issued_command + streams["process_noise"][step]
    next_pose = motion(pose, rates)
    odometry = rates * np.array([np.exp(parameters[2]), 1.])
    odometry[1] += parameters[3]
    odometry += streams["odometry_noise"][step]
    gap = 28 if panel == "gain_switch_long_gap" else 14
    visible = (step + 1 + int(streams["outage_phase"])) % 45 >= gap
    packet = sense(next_pose, streams["landmark_noise"][step], visible)
    return next_pose, odometry, packet


def metrics(poses, estimates, commands, switch_step, panel):
    targets = np.array([reference((i + 1) * DT)[0] for i in range(HORIZON)])
    tracking = np.sum((poses[:, :2] - targets) ** 2, axis=1)
    localization = np.sum((poses[:, :2] - estimates[:, :2]) ** 2, axis=1)
    start = 150 if panel == "stationary" else int(switch_step) + 30
    return {"tracking_mse": float(tracking.mean()),
            "late_tracking_mse": float(tracking[start:].mean()), "late_start_step": start,
            "localization_mse": float(localization.mean()),
            "late_localization_mse": float(localization[start:].mean()),
            "heading_mse": float(np.mean(wrap(poses[:, 2] - estimates[:, 2]) ** 2)),
            "mean_command_squared": float(np.mean(np.sum(commands ** 2, axis=1))),
            "max_tracking_distance": float(np.sqrt(tracking.max())),
            "diverged": bool(np.any(np.sqrt(tracking) > 3.))}


def run_episode(method, panel, streams):
    from openjev.research.drive_observers import Observer

    if method not in METHODS:
        raise ValueError("Unknown observer")
    observer = None if method == "pose_oracle" else Observer(method, LANDMARKS, INITIAL_POSE)
    pose = INITIAL_POSE.copy()
    estimate = pose.copy()
    poses, estimates, commands, odometry, packets, masks = [], [], [], [], [], []
    parameters = []
    observer_seconds = controller_seconds = 0.

    def snapshot():
        return {"poses": np.array(poses), "estimates": np.array(estimates),
                "commands": np.array(commands), "odometry": np.array(odometry),
                "landmarks": np.array(packets), "valid": np.array(masks),
                "private_parameters": np.array(parameters)}

    try:
        for step in range(HORIZON):
            tick = time.perf_counter()
            action = command(estimate, step * DT)
            controller_seconds += time.perf_counter() - tick
            pose, odom, packet = transition(pose, action, streams, panel, step)
            # Retain every paid native transition even if the observer fails.
            poses.append(pose.copy())
            commands.append(action.copy())
            odometry.append(odom.copy())
            packets.append(packet["values"].copy())
            masks.append(packet["valid"].copy())
            parameters.append(private_parameters(streams, panel, step).copy())
            tick = time.perf_counter()
            if method == "pose_oracle":
                estimate = pose.copy()
            else:
                oracle = private_parameters(streams, panel, step) if method == "parameter_oracle" else None
                estimate = observer.step(action, odom, packet, oracle_params=oracle)
            observer_seconds += time.perf_counter() - tick
            estimates.append(estimate.copy())
            if not np.isfinite(pose).all() or not np.isfinite(estimate).all():
                raise ValueError("Nonfinite episode state")
    except BaseException as exc:
        raise EpisodeFailure(f"{type(exc).__name__}: {exc}", snapshot()) from exc
    trace = snapshot()
    result = metrics(trace["poses"], trace["estimates"], trace["commands"], streams["switch_step"], panel)
    return trace, result, {"observer_wall_seconds": observer_seconds,
                           "controller_wall_seconds": controller_seconds}
