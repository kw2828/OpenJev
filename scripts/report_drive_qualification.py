"""Authenticate and reconstruct saved drive traces without observer/task imports."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = ("stationary", "gain_switch", "gain_switch_long_gap")
METHODS = ("command_dr", "odom_dr", "pose_ekf", "calibration_ekf", "joint_ekf",
           "parameter_oracle", "pose_oracle")
SEEDS = tuple(range(24101, 24113))
HORIZON = 300
REQUIRED_SOURCES = {
    "src/openjev/research/drive_task.py", "src/openjev/research/drive_observers.py",
    "tests/test_drive_task.py", "tests/test_drive_observers.py",
    "scripts/run_drive_qualification.py", "research/drive-qualification-protocol.md",
    "scripts/report_drive_qualification.py", "tests/test_report_drive_qualification.py",
}
STREAM_NAMES = {"initial_params", "changed_params", "process_noise", "odometry_noise",
                "landmark_noise", "outage_phase", "switch_step"}
TRACE_NAMES = {"poses", "estimates", "commands", "odometry", "landmarks", "valid",
               "private_parameters"}
RULE = {"required_panels": list(PANELS[1:]), "relative_improvement": .10,
        "absolute_mse_improvement": .001, "paired_wins": 9,
        "maximum_divergent_episodes": 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid nonnegative scalar: {name}")
    return float(value)


def equal(actual, expected, name, *, atol=2e-12):
    require(np.shape(actual) == np.shape(expected), f"Shape mismatch: {name}")
    require(np.allclose(actual, expected, rtol=1e-12, atol=atol, equal_nan=False),
            f"Numeric mismatch: {name}")
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.))


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "Duplicate archive fields")
        return {name: saved[name].copy() for name in saved.files}


def wrap(angle):
    return np.remainder(angle + np.pi, 2 * np.pi) - np.pi


def clock_target(t):
    angle = .22 * t
    p = np.array([3 * np.sin(angle), 1.5 * np.sin(2 * angle)])
    v = np.array([.66 * np.cos(angle), .66 * np.cos(2 * angle)])
    a = np.array([-.1452 * np.sin(angle), -.2904 * np.sin(2 * angle)])
    return p, v, (v[0] * a[1] - v[1] * a[0]) / np.dot(v, v)


def issued_command(estimate, t):
    p, v, omega = clock_target(t)
    desired = v + 1.2 * (p - estimate[:2])
    forward = np.cos(estimate[2]) * desired[0] + np.sin(estimate[2]) * desired[1]
    heading = np.arctan2(desired[1], desired[0])
    return np.array([np.clip(forward, 0., 1.5),
                     np.clip(omega + 3 * wrap(heading - estimate[2]), -3., 3.)])


def validate_streams(streams):
    require(set(streams) == STREAM_NAMES, "Stream fields")
    shapes = {"initial_params": (4,), "changed_params": (4,),
              "process_noise": (300, 2), "odometry_noise": (300, 2),
              "landmark_noise": (300, 6, 2)}
    for name, shape in shapes.items():
        x = streams[name]
        require(x.shape == shape and x.dtype == np.float64 and np.isfinite(x).all(),
                f"Stream schema: {name}")
    for name, maximum in (("outage_phase", 44), ("switch_step", 180)):
        value = streams[name]
        minimum = 120 if name == "switch_step" else 0
        require(value.shape == () and value.dtype.kind in "iu"
                and minimum <= int(value) <= maximum, f"Stream integer: {name}")
    for name in ("initial_params", "changed_params"):
        require(np.all(np.abs(streams[name]) <= [.3, .3, .2, .16]), "Parameter support")
    require(np.array_equal(streams["initial_params"][2:], streams["changed_params"][2:]),
            "Sensor calibration changed")


def reconstruct(trace, streams, panel, method):
    """Independent equations over saved arrays; no environment or filter invocation."""
    require(panel in PANELS and method in METHODS, "Episode identity")
    validate_streams(streams)
    require(set(trace) == TRACE_NAMES, "Trace fields")
    shapes = {"poses": (300, 3), "estimates": (300, 3), "commands": (300, 2),
              "odometry": (300, 2), "landmarks": (300, 6, 2),
              "private_parameters": (300, 4), "valid": (300, 6)}
    for name, shape in shapes.items():
        x = trace[name]
        require(x.shape == shape and x.dtype == (np.bool_ if name == "valid" else np.float64),
                f"Trace schema: {name}")
        if name not in ("landmarks", "valid"):
            require(np.isfinite(x).all(), f"Nonfinite trace: {name}")
    require(np.isnan(trace["landmarks"][~trace["valid"]]).all(), "Missing packet encoding")
    require(np.isfinite(trace["landmarks"][trace["valid"]]).all(), "Valid packet nonfinite")
    landmarks = np.array([[5 * np.cos(j * np.pi / 3), 5 * np.sin(j * np.pi / 3)]
                          for j in range(6)])
    pose = np.array([0., 0., np.pi / 4])
    previous_estimate = pose.copy()
    errors = dict.fromkeys(("pose", "odometry", "landmark", "command", "parameters"), 0.)
    for step in range(HORIZON):
        expected_command = issued_command(previous_estimate, step * .1)
        errors["command"] = max(errors["command"], equal(trace["commands"][step], expected_command,
                                                        "previous-estimate command"))
        use_changed = panel != "stationary" and step >= int(streams["switch_step"])
        params = streams["changed_params" if use_changed else "initial_params"]
        errors["parameters"] = max(errors["parameters"], equal(trace["private_parameters"][step],
                                                              params, "parameter timeline", atol=0.))
        rates = np.exp(params[:2]) * trace["commands"][step] + streams["process_noise"][step]
        x, y, theta = pose
        pose = np.array([x + .1 * rates[0] * np.cos(theta),
                         y + .1 * rates[0] * np.sin(theta), wrap(theta + .1 * rates[1])])
        errors["pose"] = max(errors["pose"], equal(trace["poses"][step], pose, "native endpoint"))
        odom = np.array([np.exp(params[2]) * rates[0], rates[1] + params[3]])
        odom += streams["odometry_noise"][step]
        errors["odometry"] = max(errors["odometry"], equal(trace["odometry"][step], odom,
                                                          "interval odometry"))
        displacement = landmarks - pose[:2]
        distance = np.sqrt(np.sum(displacement ** 2, axis=1))
        visible = (step + 1 + int(streams["outage_phase"])) % 45 >= (
            28 if panel == "gain_switch_long_gap" else 14)
        valid = (distance < 6.) & (distance > .1) & visible
        require(np.array_equal(trace["valid"][step], valid), "Endpoint validity/timestamp")
        packet = np.column_stack((distance, wrap(np.arctan2(displacement[:, 1], displacement[:, 0])
                                                 - pose[2])))
        packet += streams["landmark_noise"][step]
        packet[:, 1] = wrap(packet[:, 1])
        errors["landmark"] = max(errors["landmark"], equal(trace["landmarks"][step, valid],
                                                           packet[valid], "endpoint packet"))
        previous_estimate = trace["estimates"][step]
    if method == "pose_oracle":
        equal(trace["estimates"], trace["poses"], "pose oracle")
    targets = np.stack([clock_target((step + 1) * .1)[0] for step in range(HORIZON)])
    tracking = np.sum((trace["poses"][:, :2] - targets) ** 2, axis=1)
    localization = np.sum((trace["poses"][:, :2] - trace["estimates"][:, :2]) ** 2, axis=1)
    start = 150 if panel == "stationary" else int(streams["switch_step"]) + 30
    metrics = {"tracking_mse": float(tracking.mean()),
               "late_tracking_mse": float(tracking[start:].mean()), "late_start_step": start,
               "localization_mse": float(localization.mean()),
               "late_localization_mse": float(localization[start:].mean()),
               "heading_mse": float(np.mean(wrap(trace["poses"][:, 2] - trace["estimates"][:, 2]) ** 2)),
               "mean_command_squared": float(np.mean(np.sum(trace["commands"] ** 2, axis=1))),
               "max_tracking_distance": float(np.sqrt(tracking.max())),
               "diverged": bool(np.any(np.sqrt(tracking) > 3.))}
    return metrics, errors


def compare_metrics(saved, calculated):
    require(set(saved) == set(calculated), "Metric fields")
    for name, value in calculated.items():
        if type(value) in (bool, int):
            require(type(saved[name]) is type(value) and saved[name] == value, f"Metric: {name}")
        else:
            equal(finite(saved[name], name), value, f"Metric: {name}")


def aggregate(records):
    expected = {(p, s, m) for p in PANELS for s in SEEDS for m in METHODS}
    keys = [(r["panel"], r["seed"], r["method"]) for r in records]
    require(len(keys) == 252 and len(set(keys)) == 252 and set(keys) == expected, "Episode coverage")
    indexed = dict(zip(keys, records, strict=True))
    panels, checks = {}, []
    for panel in PANELS:
        methods = {}
        for method in METHODS:
            rows = [indexed[panel, seed, method] for seed in SEEDS]
            methods[method] = {
                "seeds": list(SEEDS),
                "late_tracking_mse": [r["metrics"]["late_tracking_mse"] for r in rows],
                "mean_late_tracking_mse": float(np.mean([r["metrics"]["late_tracking_mse"] for r in rows])),
                "mean_metrics": {key: float(np.mean([r["metrics"][key] for r in rows]))
                                 for key in ("tracking_mse", "localization_mse", "late_localization_mse",
                                             "heading_mse", "mean_command_squared", "max_tracking_distance")},
                "divergent_episodes": sum(r["metrics"]["diverged"] for r in rows),
                "costs": {key: sum(r["costs"][key] for r in rows)
                          for key in ("observer_wall_seconds", "controller_wall_seconds")},
                "episode_wall_seconds": sum(r["episode_wall_seconds"] for r in rows),
            }
        joint = methods["joint_ekf"]
        oracle = methods["parameter_oracle"]
        j = sum(Decimal(str(x)) for x in joint["late_tracking_mse"]) / Decimal(12)
        o = sum(Decimal(str(x)) for x in oracle["late_tracking_mse"]) / Decimal(12)
        wins = sum(a > b for a, b in zip(joint["late_tracking_mse"], oracle["late_tracking_mse"], strict=True))
        outcomes = {"relative_improvement": j > 0 and o <= Decimal(".9") * j,
                    "absolute_mse_improvement": j - o >= Decimal(".001"),
                    "paired_wins": wins >= 9,
                    "joint_zero_divergence": joint["divergent_episodes"] == 0,
                    "parameter_oracle_zero_divergence": oracle["divergent_episodes"] == 0}
        panel_checks = [{"panel": panel, "name": name, "passed": bool(passed),
                         "required": panel != "stationary"} for name, passed in outcomes.items()]
        checks.extend(panel_checks)
        panels[panel] = {"methods": methods, "paired_wins": wins,
                         "paired_differences_joint_minus_parameter_oracle": [a - b for a, b in zip(
                             joint["late_tracking_mse"], oracle["late_tracking_mse"], strict=True)],
                         "relative_improvement": float((j - o) / j) if j > 0 else None,
                         "absolute_mse_improvement": float(j - o),
                         "checks_passed": sum(outcomes.values()), "passed": all(outcomes.values())}
    required = [x for x in checks if x["required"]]
    return panels, {"passed": all(x["passed"] for x in required),
                    "checks_passed": sum(x["passed"] for x in required), "total_checks": 10,
                    "all_panel_checks_passed": sum(x["passed"] for x in checks),
                    "all_panel_checks": 15, "checks": checks,
                    "positive_means": "necessary screen only; next classical MHE stage",
                    "negative_means": "close this fixed recipe; no post-outcome retuning"}


def authenticate(experiment, protocol_sha256, completed_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "External protocol digest")
    run = experiment / "run-01"
    require(sha(run / "completed.json") == completed_sha256, "External completion digest")
    protocol, completed = read(experiment / "protocol.json"), read(run / "completed.json")
    require(set(protocol) == {"version", "seeds", "panels", "methods", "horizon", "episodes",
                              "source_sha256", "streams_sha256", "rule", "runtime"}, "Protocol fields")
    require(protocol["version"] == "drive-qualification-v1" and protocol["seeds"] == list(SEEDS)
            and protocol["panels"] == list(PANELS) and protocol["methods"] == list(METHODS)
            and protocol["horizon"] == 300 and protocol["episodes"] == 252
            and protocol["rule"] == RULE, "Protocol settings")
    require(REQUIRED_SOURCES == set(protocol["source_sha256"]), "Source binding membership")
    for name, digest in protocol["source_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and sha(ROOT / path) == digest,
                f"Source digest: {name}")
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "numpy": importlib.metadata.version("numpy")}
    require(protocol["runtime"] == runtime, "Runtime identity")
    streams = {f"seed-{seed}.npz" for seed in SEEDS}
    require(set(protocol["streams_sha256"]) == streams
            and {p.name for p in (experiment / "streams").iterdir()} == streams, "Stream membership")
    for name, digest in protocol["streams_sha256"].items():
        require(sha(experiment / "streams" / name) == digest, f"Stream digest: {name}")
    require(set(completed) == {"status", "episodes", "native_steps", "wall_seconds", "protocol_sha256",
                               "records_sha256", "neural_fits", "model_api_calls"}, "Completion fields")
    require(completed["status"] == "complete" and completed["episodes"] == 252
            and completed["native_steps"] == 75600 and completed["neural_fits"] == 0
            and completed["model_api_calls"] == 0 and completed["protocol_sha256"] == protocol_sha256,
            "Execution completion")
    require(sha(run / "records.json") == completed["records_sha256"], "Records digest")
    names = {f"{p}-{s}-{m}.{ext}" for p in PANELS for s in SEEDS for m in METHODS for ext in ("json", "npz")}
    names |= {"records.json", "completed.json"}
    require({str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()} == names,
            "Execution membership")
    require(not any(p.is_symlink() for p in run.rglob("*")), "Symlink execution member")
    return protocol, completed


def figures(out, panels, first_traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#8795a1", "#b3bdc5", "#408ab3", "#6d62ab", "#c96a34", "#288365", "#26343e")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, panel in zip(axes, PANELS, strict=True):
        for i, (method, color) in enumerate(zip(METHODS, colors, strict=True)):
            values = panels[panel]["methods"][method]["late_tracking_mse"]
            ax.scatter(np.full(12, i) + np.linspace(-.16, .16, 12), values, s=16, color=color, alpha=.8)
            ax.plot([i - .22, i + .22], [np.mean(values)] * 2, color="black", linewidth=2)
        joint = panels[panel]["methods"]["joint_ekf"]["late_tracking_mse"]
        oracle = panels[panel]["methods"]["parameter_oracle"]["late_tracking_mse"]
        for a, b in zip(joint, oracle, strict=True):
            ax.plot([4, 5], [a, b], color="#777777", alpha=.35, linewidth=.7)
        all_values = [v for m in METHODS for v in panels[panel]["methods"][m]["late_tracking_mse"]]
        if min(all_values) > 0:
            ax.set_yscale("log")
            scale_label = "log scale"
        else:
            scale_label = "linear scale; zero retained"
        ax.set_xticks(range(7), [m.replace("_", "\n") for m in METHODS], fontsize=8)
        ax.set_title(panel.replace("_", " "))
        ax.set_ylabel(f"Late tracking MSE (m²); {scale_label}")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Necessary control-opportunity screen | 12 paired seeds; all methods retained")
    fig.savefig(out / "paired-costs.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    target = np.stack([clock_target((i + 1) * .1)[0] for i in range(300)])
    for ax, panel in zip(axes, PANELS, strict=True):
        ax.plot(target[:, 0], target[:, 1], "k--", linewidth=1, label="Clocked target")
        for method, color in zip(METHODS, colors, strict=True):
            pose = first_traces[panel, method]["poses"]
            ax.plot(pose[:, 0], pose[:, 1], color=color, linewidth=1, label=method)
        ax.set_title(panel.replace("_", " "))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Fixed first seed 24101 | synthetic trajectories, not a selected success")
    fig.savefig(out / "first-seed-trajectories.png", dpi=170)
    plt.close(fig)


def report(experiment, out, protocol_sha256, completed_sha256):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        protocol, completed = authenticate(experiment, protocol_sha256, completed_sha256)
        run = experiment / "run-01"
        records = read(run / "records.json")
        aggregate(records)  # Coverage before any indexed processing.
        max_errors = dict.fromkeys(("pose", "odometry", "landmark", "command", "parameters"), 0.)
        first_traces = {}
        stream_cache = {s: archive(experiment / f"streams/seed-{s}.npz") for s in SEEDS}
        expected_order = [(s, p, m) for s in SEEDS for p in PANELS for m in METHODS]
        require([(r["seed"], r["panel"], r["method"]) for r in records] == expected_order, "Row order")
        members = {}
        for record in records:
            require(set(record) == {"seed", "panel", "method", "trace", "trace_sha256", "metrics", "costs",
                                    "episode_wall_seconds", "native_steps"}, "Row fields")
            seed, panel, method = record["seed"], record["panel"], record["method"]
            stem = f"{panel}-{seed}-{method}"
            require(record == read(run / f"{stem}.json"), "Row record binding")
            require(record["trace"] == f"{stem}.npz"
                    and sha(run / record["trace"]) == record["trace_sha256"], "Trace digest")
            require(record["native_steps"] == 300, "Row native work")
            require(set(record["costs"]) == {"observer_wall_seconds", "controller_wall_seconds"}, "Cost fields")
            wall = finite(record["episode_wall_seconds"], "episode wall")
            require(sum(finite(v, k) for k, v in record["costs"].items()) <= wall + 1e-9, "Nested row timing")
            trace = archive(run / record["trace"])
            metrics, errors = reconstruct(trace, stream_cache[seed], panel, method)
            compare_metrics(record["metrics"], metrics)
            record["metrics"] = metrics
            for name, value in errors.items():
                max_errors[name] = max(max_errors[name], value)
            if seed == SEEDS[0]:
                first_traces[panel, method] = trace
        panels, screen = aggregate(records)
        row_wall = sum(r["episode_wall_seconds"] for r in records)
        require(row_wall <= finite(completed["wall_seconds"], "execution wall") + 1e-9, "Whole execution timing")
        for path in sorted(run.iterdir()):
            members[path.name] = sha(path)
        summary = {"study": "drive-qualification-v1", "status": "completed",
                   "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
                   "panels": panels, "screen": screen, "records": records,
                   "counts": {"episodes": 252, "native_transitions_checked": 75600,
                              "stream_files": 12, "execution_files": 506, "new_model_calls": 0},
                   "maximum_reconstruction_error": max_errors,
                   "costs": {"execution_wall_seconds": completed["wall_seconds"],
                             "episode_wall_seconds": row_wall,
                             "observer_wall_seconds": sum(r["costs"]["observer_wall_seconds"] for r in records),
                             "controller_wall_seconds": sum(r["costs"]["controller_wall_seconds"] for r in records)},
                   "scope": "Saved-output equation reconstruction; observer algebra is source/test-bound, not replayed. "
                            "Stream hashes bind supplied draws; no RNG regeneration. Necessary screen only; "
                            "native labels refer only to the synthetic plant, not a physical robot. "
                            "No neural or architecture performance claim. Equal weight per seed.",
                   "protocol_document": "research/drive-qualification-protocol.md"}
        write(out / "summary.json", summary)
        figures(out, panels, first_traces)
        authenticate(experiment, protocol_sha256, completed_sha256)
        require(all(sha(run / name) == digest for name, digest in members.items()), "Execution changed during audit")
        write(out / "receipt.json", {"status": "completed", "qualification_passed": screen["passed"],
                                    "protocol_sha256": protocol_sha256,
                                    "execution_completed_sha256": completed_sha256,
                                    "source_sha256": protocol["source_sha256"],
                                    "streams_sha256": protocol["streams_sha256"],
                                    "execution_members": members,
                                    "files": {p.name: sha(p) for p in sorted(out.iterdir())},
                                    "wall_seconds": time.perf_counter() - start,
                                    "time_scope": "authentication, equations, aggregation and figures; excludes receipt write",
                                    "scope": summary["scope"]})
        return summary
    except BaseException as exc:
        try:
            if (out / "receipt.json").exists():
                (out / "receipt.json").rename(out / "receipt-before-error.json")
            write(out / "failed.json", {"status": "failed", "error_type": type(exc).__name__,
                                       "error": str(exc), "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - keep the original exception
            exc.add_note(f"Failure preservation also failed: {secondary}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    args = parser.parse_args()
    result = report(args.experiment, args.out, args.protocol_sha256, args.completed_sha256)
    print(json.dumps({"status": result["status"], "screen": result["screen"]}), flush=True)
