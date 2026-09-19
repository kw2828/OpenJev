"""Handwritten synthetic traces and receipts; no task, observer, or RNG calls."""

import copy
import importlib.metadata
import json
import math
import platform

import numpy as np
import pytest
import report_drive_qualification as audit


def synthetic(panel="gain_switch"):
    """Scalar handwritten fixture, with nonzero noise and a physical gain change."""
    streams = {"initial_params": np.array([.1, -.12, .05, -.03]),
               "changed_params": np.array([-.2, .2, .05, -.03]),
               "process_noise": np.tile([.01, -.02], (300, 1)),
               "odometry_noise": np.tile([-.015, .025], (300, 1)),
               "landmark_noise": np.tile([.03, -.01], (300, 6, 1)),
               "outage_phase": np.array(5), "switch_step": np.array(150)}
    rows = {name: [] for name in audit.TRACE_NAMES}
    x, y, theta = 0., 0., math.pi / 4
    turn_wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    for i in range(300):
        phase = .022 * i
        rx, ry = 3 * math.sin(phase), 1.5 * math.sin(2 * phase)
        vx, vy = .66 * math.cos(phase), .66 * math.cos(2 * phase)
        ax, ay = -.1452 * math.sin(phase), -.2904 * math.sin(2 * phase)
        gx, gy = vx + 1.2 * (rx - x), vy + 1.2 * (ry - y)
        command_v = min(1.5, max(0., math.cos(theta) * gx + math.sin(theta) * gy))
        command_w = min(3., max(-3., (vx * ay - vy * ax) / (vx * vx + vy * vy)
                                + 3 * turn_wrap(math.atan2(gy, gx) - theta)))
        params = streams["changed_params" if panel != "stationary" and i >= 150 else "initial_params"]
        v, w = math.exp(params[0]) * command_v + .01, math.exp(params[1]) * command_w - .02
        x, y, theta = x + .1 * v * math.cos(theta), y + .1 * v * math.sin(theta), turn_wrap(theta + .1 * w)
        values, valid = [], []
        for j in range(6):
            dx, dy = 5 * math.cos(j * math.pi / 3) - x, 5 * math.sin(j * math.pi / 3) - y
            distance = math.hypot(dx, dy)
            seen = .1 < distance < 6 and (i + 6) % 45 >= (28 if panel == "gain_switch_long_gap" else 14)
            valid.append(seen)
            values.append([distance + .03, turn_wrap(turn_wrap(math.atan2(dy, dx) - theta) - .01)]
                          if seen else [math.nan, math.nan])
        for name, value in {"poses": [x, y, theta], "estimates": [x, y, theta],
                            "commands": [command_v, command_w], "odometry": [math.exp(params[2]) * v - .015, w + params[3] + .025],
                            "landmarks": values, "valid": valid, "private_parameters": params.copy()}.items():
            rows[name].append(value)
    return {name: np.array(value) for name, value in rows.items()}, streams


@pytest.mark.parametrize("panel", audit.PANELS)
def test_handwritten_equations_and_late_window(panel):
    trace, streams = synthetic(panel)
    metrics, errors = audit.reconstruct(trace, streams, panel, "pose_oracle")
    assert metrics["late_start_step"] == (150 if panel == "stationary" else 180)
    assert metrics["heading_mse"] == metrics["localization_mse"] == 0
    assert max(errors.values()) < 2e-12
    target = np.array([[3 * math.sin(.022 * (i + 1)), 1.5 * math.sin(.044 * (i + 1))]
                       for i in range(300)])
    assert metrics["late_tracking_mse"] == pytest.approx(np.mean(np.sum(
        (trace["poses"][metrics["late_start_step"]:, :2] - target[metrics["late_start_step"]:]) ** 2, axis=1)))


@pytest.mark.parametrize("field,index", [
    ("poses", (180, 0)), ("commands", (25, 1)), ("odometry", (150, 0)),
    ("private_parameters", (150, 2)), ("estimates", (24, 0)),
])
def test_resealed_numeric_corruption_rejected(field, index):
    trace, streams = synthetic()
    trace[field][index] += .001
    with pytest.raises(ValueError, match="mismatch"):
        audit.reconstruct(trace, streams, "gain_switch", "pose_oracle")


def test_packet_value_mask_and_time_alignment():
    trace, streams = synthetic()
    i, j = np.argwhere(trace["valid"])[0]
    bad = copy.deepcopy(trace)
    bad["landmarks"][i, j, 0] += .01
    with pytest.raises(ValueError, match="endpoint packet"):
        audit.reconstruct(bad, streams, "gain_switch", "pose_oracle")
    bad = copy.deepcopy(trace)
    bad["valid"][i, j] = False
    bad["landmarks"][i, j] = np.nan
    with pytest.raises(ValueError, match="timestamp"):
        audit.reconstruct(bad, streams, "gain_switch", "pose_oracle")
    streams["outage_phase"] += 1
    with pytest.raises(ValueError, match="timestamp"):
        audit.reconstruct(trace, streams, "gain_switch", "pose_oracle")


def test_sensor_calibration_cannot_change_at_physical_switch():
    trace, streams = synthetic()
    streams["changed_params"][3] += .01
    with pytest.raises(ValueError, match="Sensor calibration"):
        audit.reconstruct(trace, streams, "gain_switch", "pose_oracle")


def fake_records(joint=.01, oracle=.009):
    records = []
    for seed in audit.SEEDS:
        for panel in audit.PANELS:
            for method in audit.METHODS:
                value = oracle if method == "parameter_oracle" else joint
                metrics = dict.fromkeys(("tracking_mse", "localization_mse", "late_localization_mse",
                                         "heading_mse", "mean_command_squared", "max_tracking_distance"), 0.)
                metrics.update(late_tracking_mse=value, diverged=False)
                records.append({"seed": seed, "panel": panel, "method": method, "metrics": metrics,
                                "costs": {"observer_wall_seconds": .01, "controller_wall_seconds": .01},
                                "episode_wall_seconds": .03})
    return records


def test_inclusive_decimal_thresholds_and_stationary_not_required():
    records = fake_records()
    for row in records:
        if row["panel"] == "stationary" and row["method"] == "parameter_oracle":
            row["metrics"]["late_tracking_mse"] = .02
    panels, gate = audit.aggregate(records)
    assert gate["passed"] and gate["checks_passed"] == gate["total_checks"] == 10
    assert gate["all_panel_checks"] == len(gate["checks"]) == 15
    assert not panels["stationary"]["passed"]
    assert panels["gain_switch"]["relative_improvement"] == .1
    assert panels["gain_switch"]["absolute_mse_improvement"] == .001


def test_just_below_threshold_and_zero_denominator_fail():
    assert not audit.aggregate(fake_records(oracle=.009000000000000002))[1]["passed"]
    panels, gate = audit.aggregate(fake_records(0., 0.))
    assert not gate["passed"]
    assert panels["gain_switch"]["relative_improvement"] is None


def test_strict_nine_paired_wins_and_separate_divergence_rules():
    records = fake_records(1., .8)
    for row in records:
        if row["method"] == "parameter_oracle" and row["seed"] in audit.SEEDS[-3:]:
            row["metrics"]["late_tracking_mse"] = 1.
    assert audit.aggregate(records)[0]["gain_switch"]["paired_wins"] == 9
    assert audit.aggregate(records)[1]["passed"]
    for row in records:
        if row["method"] == "joint_ekf" and row["panel"] == "gain_switch":
            row["metrics"]["diverged"] = True
            break
    checks = audit.aggregate(records)[1]["checks"]
    assert not next(c["passed"] for c in checks if c["panel"] == "gain_switch" and c["name"] == "joint_zero_divergence")
    assert next(c["passed"] for c in checks if c["panel"] == "gain_switch" and c["name"] == "parameter_oracle_zero_divergence")


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_seed"])
def test_exact_episode_coverage(change):
    rows = fake_records()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["seed"] = 410
    with pytest.raises(ValueError, match="coverage"):
        audit.aggregate(rows)


def envelope(tmp_path, monkeypatch):
    root = tmp_path / "root"
    experiment = tmp_path / "experiment"
    run = experiment / "run-01"
    run.mkdir(parents=True)
    (experiment / "streams").mkdir()
    for name in audit.REQUIRED_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source\n")
    monkeypatch.setattr(audit, "ROOT", root)
    for seed in audit.SEEDS:
        (experiment / f"streams/seed-{seed}.npz").write_bytes(b"synthetic bytes")
        for panel in audit.PANELS:
            for method in audit.METHODS:
                for ext in ("json", "npz"):
                    (run / f"{panel}-{seed}-{method}.{ext}").write_bytes(b"synthetic bytes")
    protocol = {"version": "drive-qualification-v1", "seeds": list(audit.SEEDS),
                "panels": list(audit.PANELS), "methods": list(audit.METHODS), "horizon": 300,
                "episodes": 252, "rule": audit.RULE,
                "source_sha256": {name: audit.sha(root / name) for name in audit.REQUIRED_SOURCES},
                "streams_sha256": {p.name: audit.sha(p) for p in (experiment / "streams").iterdir()},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": importlib.metadata.version("numpy")}}
    audit.write(experiment / "protocol.json", protocol)
    audit.write(run / "records.json", [])
    completed = {"status": "complete", "episodes": 252, "native_steps": 75600,
                 "neural_fits": 0, "model_api_calls": 0, "wall_seconds": 1.,
                 "protocol_sha256": audit.sha(experiment / "protocol.json"),
                 "records_sha256": audit.sha(run / "records.json")}
    audit.write(run / "completed.json", completed)
    return experiment, completed["protocol_sha256"], audit.sha(run / "completed.json")


def test_complete_byte_envelope_and_external_pins(tmp_path, monkeypatch):
    experiment, p, c = envelope(tmp_path, monkeypatch)
    audit.authenticate(experiment, p, c)
    with pytest.raises(ValueError, match="External protocol"):
        audit.authenticate(experiment, "0" * 64, c)
    with pytest.raises(ValueError, match="External completion"):
        audit.authenticate(experiment, p, "0" * 64)


@pytest.mark.parametrize("target", ["source", "stream", "failed", "records"])
def test_mutated_binding_or_extra_failure_rejected(tmp_path, monkeypatch, target):
    experiment, p, c = envelope(tmp_path, monkeypatch)
    if target == "source":
        path = audit.ROOT / min(audit.REQUIRED_SOURCES)
    elif target == "stream":
        path = experiment / f"streams/seed-{audit.SEEDS[0]}.npz"
    elif target == "failed":
        path = experiment / "run-01/failed.json"
    else:
        path = experiment / "run-01/records.json"
    path.write_text("tampered")
    with pytest.raises(ValueError):
        audit.authenticate(experiment, p, c)


def test_exclusive_output_and_preserved_original_failure(tmp_path, monkeypatch):
    def fail(*_):
        raise ValueError("deliberate authentication failure")
    monkeypatch.setattr(audit, "authenticate", fail)
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="deliberate authentication"):
        audit.report(tmp_path, out, "p", "c")
    assert json.loads((out / "failed.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        audit.report(tmp_path, out, "p", "c")


def test_failure_receipt_error_does_not_mask_original(tmp_path, monkeypatch):
    def fail(*_):
        raise ValueError("original")
    def fail_write(*_):
        raise OSError("disk")
    monkeypatch.setattr(audit, "authenticate", fail)
    monkeypatch.setattr(audit, "write", fail_write)
    with pytest.raises(ValueError, match="original") as caught:
        audit.report(tmp_path, tmp_path / "report", "p", "c")
    assert "disk" in caught.value.__notes__[0]


def test_authenticated_outer_success_retains_negative_screen(tmp_path, monkeypatch):
    """Real byte authentication, with numerical replay/plotting stubbed and tested separately."""
    experiment, _, _ = envelope(tmp_path, monkeypatch)
    run = experiment / "run-01"
    trace, streams = synthetic()
    metrics, errors = audit.reconstruct(trace, streams, "gain_switch", "pose_oracle")
    for seed in audit.SEEDS:
        np.savez_compressed(experiment / f"streams/seed-{seed}.npz", **streams)
    records = []
    for record in fake_records():
        stem = f"{record['panel']}-{record['seed']}-{record['method']}"
        np.savez_compressed(run / f"{stem}.npz", **trace)
        record.update(trace=f"{stem}.npz", trace_sha256=audit.sha(run / f"{stem}.npz"),
                      native_steps=300, metrics=metrics)
        (run / f"{stem}.json").write_text(json.dumps(record))
        records.append(record)
    (run / "records.json").write_text(json.dumps(records))
    protocol = audit.read(experiment / "protocol.json")
    protocol["streams_sha256"] = {p.name: audit.sha(p) for p in (experiment / "streams").iterdir()}
    (experiment / "protocol.json").write_text(json.dumps(protocol))
    p = audit.sha(experiment / "protocol.json")
    completed = audit.read(run / "completed.json")
    completed.update(protocol_sha256=p, records_sha256=audit.sha(run / "records.json"), wall_seconds=10.)
    (run / "completed.json").write_text(json.dumps(completed))
    c = audit.sha(run / "completed.json")
    calls = []
    def fake_reconstruct(*args):
        calls.append(args[2:])
        return metrics, errors
    def fake_figures(out, *_):
        (out / "paired-costs.png").write_bytes(b"synthetic figure stub")
        (out / "first-seed-trajectories.png").write_bytes(b"synthetic figure stub")
    monkeypatch.setattr(audit, "reconstruct", fake_reconstruct)
    monkeypatch.setattr(audit, "figures", fake_figures)
    out = tmp_path / "report"
    summary = audit.report(experiment, out, p, c)
    receipt = audit.read(out / "receipt.json")
    assert summary["status"] == receipt["status"] == "completed"
    assert not receipt["qualification_passed"]
    assert len(calls) == 252 and len(receipt["execution_members"]) == 506
    assert set(receipt["files"]) == {"summary.json", "paired-costs.png", "first-seed-trajectories.png"}
    assert receipt["files"]["summary.json"] == audit.sha(out / "summary.json")
