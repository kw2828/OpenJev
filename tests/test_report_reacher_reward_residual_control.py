"""Synthetic completed-audit fixtures; no model, simulator or execution inputs."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_reacher_reward_residual_control.py"
spec = importlib.util.spec_from_file_location("residual_control_report_fixture", SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rebind(folder, summary):
    write_json(folder / "summary.json", summary)
    receipt = {
        "status": "completed", "version": r.VERSION, "saved_output_only": True,
        **{key: summary[key] for key in ("plan_sha256", "execution_completed_sha256", "costs",
                                        "training_source", "fit_source")},
        "files": {name: r.digest((folder / name).read_bytes()) for name in ("summary.json", "README.md")},
    }
    write_json(folder / "receipt.json", receipt)
    return r.digest((folder / "receipt.json").read_bytes())


@pytest.fixture
def saved(tmp_path):
    folder = tmp_path / "synthetic-audit"
    folder.mkdir()
    (folder / "README.md").write_text("Synthetic audit, not empirical evidence.\n")
    costs = dict(zip(r.COST_FIELDS, (6.0, 9.0, 12.0, 10.0, 21.0, 18.0, 0.5), strict=True))
    costs.update(new_fits=0, accounting="Synthetic cost accounting only")
    fits, predictions = {}, {}
    for index, name in enumerate(r.FIT_ORDER):
        kind, seed = name.rsplit("-", 1)
        fits[name] = {
            "kind": kind, "seed": int(seed), "updates": 1152, "parameters": 36805,
            "wall_seconds": 1.0, "residual_reward": kind == "residual", "noise_std": 0.05,
            "initial_weights_sha256": "a" * 64, "minibatch_order_sha256": "b" * 64,
            "epochs": [{key: epoch if key == "epoch" else (index + 1) / (epoch + 1)
                        for key in r.LOG_FIELDS} for epoch in range(1, 49)],
        }
        predictions[name] = {
            key: 0.125 * (index + 1) if key == "reward_mse"
            else {"mse": 0.1 * (index + 1), "targets": 64, "components": 256}
            for key in r.PREDICTION_FIELDS
        }
    control = {}
    for panel in r.PANELS:
        control[panel] = {}
        for index, name in enumerate(r.arms(panel)):
            item = {
                "episode_costs": [100.0 + index] * 64, "mean_cost": 100.0 + index,
                "planner_used": name not in ("zero", "uniform"), "setup_seconds": 0.01,
                "decision_wall_seconds": 0.02,
                "batch_latency_seconds": {"mean": 0.0004, "median": 0.0003, "p95": 0.0008},
                "per_case_amortized_seconds": 0.0004 / 64,
            }
            if name not in r.REFERENCES:
                item["on_policy_prediction"] = {
                    "all_angle": {"mse": 0.2, "targets": 3200, "components": 12800},
                    "observed_angle": {"mse": 0.1, "targets": 2400, "components": 9600},
                    "blackout_angle": {"mse": None if panel == "full" else 0.3,
                                       "targets": 0 if panel == "full" else 800,
                                       "components": 0 if panel == "full" else 3200},
                    "reward_mse": 0.04, "mean_reward_prediction_bias": -0.12345678901234567,
                }
            control[panel][name] = item
    checks = [
        {"name": name, "actual": 1.0, "threshold": 2.0,
         "direction": "lt" if name.endswith("vs_paired_free") else "le", "passed": True}
        for name in r.criterion_names()
    ]
    summary = {
        "status": "completed", "version": r.VERSION, "saved_output_only": True,
        "plan_sha256": "c" * 64, "execution_completed_sha256": "d" * 64,
        # Deliberately inaccessible paths demonstrate that reporting never follows them.
        "training_source": {"execution_path": "/nonexistent/do-not-read-training"},
        "fit_source": {"execution_path": "/nonexistent/do-not-read-prior-scores"},
        "new_model_calls": 0, "new_policy_calls": 0, "new_mpc_calls": 0, "new_fits": 0,
        "native_transitions_checked": 2250, "native_max_abs_error": 0.0,
        "random_streams": {"streams": 692, "generators": 820, "priors_checked": 2,
                           "draws_for_manifest": 0},
        "fits": fits, "predictions": predictions, "control": control, "costs": costs,
        "continuation_gate": {"passed": True, "checks": checks},
        "paired_descriptive_comparisons": {
            "ordinary": {"residual_minus_free": {
                "mean_cost_difference": -1.0, "episode_paired_percentile_95": [-2.0, 0.5],
                "cases": 64, "conditional_on_saved_fits": True,
            }},
        },
        "limits": ["Synthetic fixture, not empirical results."],
    }
    expected = rebind(folder, summary)
    return folder, summary, expected


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_complete_report_preserves_every_arm_epoch_metric_and_cost(saved, tmp_path):
    folder, summary, expected = saved
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    out = tmp_path / "report"
    receipt = r.generate(folder, expected, out)
    assert receipt["control_rows"] == 42
    assert receipt["training_rows"] == 288
    assert receipt["prediction_rows"] == 6
    assert receipt["criteria_rows"] == 17
    assert receipt["new_model_calls"] == receipt["new_simulator_calls"] == receipt["new_fits"] == 0
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}
    control = rows(out / "all-fit-control.csv")
    assert [(row["panel"], row["arm"]) for row in control] == [
        (panel, name) for panel in r.PANELS for name in r.arms(panel)
    ]
    for row in control:
        source = summary["control"][row["panel"]][row["arm"]]
        assert json.loads(row["episode_costs_json"]) == source["episode_costs"]
        assert float(row["mean_cost"]) == source["mean_cost"]
        if row["arm"] not in r.REFERENCES:
            assert float(row["on_policy_prediction.mean_reward_prediction_bias"]) == -0.12345678901234567
    assert control[0]["on_policy_prediction.blackout_angle.mse"] == ""
    training = rows(out / "all-fit-training.csv")
    for row in training:
        source = summary["fits"][row["fit"]]["epochs"][int(row["epoch"]) - 1]
        assert set(row) == {"fit", *r.LOG_FIELDS}
        assert all(float(row[key]) == value for key, value in source.items())
    for row in rows(out / "all-fit-prediction.csv"):
        flat = r.flatten(summary["predictions"][row["fit"]])
        assert set(row) == {"fit", *flat}
        assert all(float(row[key]) == value for key, value in flat.items())
    assert [row["name"] for row in rows(out / "continuation-criteria.csv")] == r.criterion_names()
    costs = {row["metric"]: float(row["seconds"]) for row in rows(out / "costs.csv")}
    assert costs == {key: summary["costs"][key] for key in r.COST_FIELDS}
    report = (out / "report.md").read_text()
    assert "no biological or connectome advantage" in report
    assert "conditional on the six inherited fits (three paired seeds)" in report
    assert "prior invalid attempt already contains inherited training" in report
    assert "42 controller/panel rows" in report
    assert all(name in report for name in r.criterion_names())
    assert "All 48 training epochs" in report
    assert receipt["files"] == {path.name: r.digest(path.read_bytes()) for path in out.iterdir()
                                if path.name != "receipt.json"}


@pytest.mark.parametrize("member", ["receipt.json", "summary.json", "README.md"])
def test_every_audit_artifact_is_authenticated_before_output(saved, tmp_path, member):
    folder, _, expected = saved
    path = folder / member
    path.write_bytes(path.read_bytes() + b"\nmodified")
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="identity mismatch|artifact changed"):
        r.generate(folder, expected, out)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["old_version", "incomplete", "not_saved", "new_model_calls",
                                      "new_policy_calls", "new_mpc_calls", "new_fits",
                                      "missing_fit", "missing_prediction", "missing_control",
                                      "missing_reset", "missing_case", "missing_epoch", "missing_metric",
                                      "missing_criterion", "wrong_pass", "wrong_gate", "double_cost"])
def test_incomplete_or_mislabelled_summary_cannot_be_rehashed_away(saved, tmp_path, mutation):
    folder, summary, _ = saved
    if mutation == "old_version":
        summary["version"] = "reacher-reward-residual-v1"
    elif mutation == "incomplete":
        summary["status"] = "failed"
    elif mutation == "not_saved":
        summary["saved_output_only"] = False
    elif mutation.startswith("new_"):
        summary[mutation] = 1
    elif mutation == "missing_fit":
        del summary["fits"]["free-271"]
    elif mutation == "missing_prediction":
        del summary["predictions"]["residual-293"]
    elif mutation == "missing_control":
        del summary["control"]["shift"]["uniform"]
    elif mutation == "missing_reset":
        del summary["control"]["ordinary"]["residual-283-reset"]
    elif mutation == "missing_case":
        summary["control"]["full"]["zero"]["episode_costs"].pop()
    elif mutation == "missing_epoch":
        summary["fits"]["free-293"]["epochs"].pop()
    elif mutation == "missing_metric":
        del summary["predictions"]["residual-283"]["reward_mse"]
    elif mutation == "missing_criterion":
        summary["continuation_gate"]["checks"].pop()
    elif mutation == "wrong_pass":
        summary["continuation_gate"]["checks"][0]["passed"] = False
    elif mutation == "wrong_gate":
        summary["continuation_gate"]["passed"] = False
    else:
        summary["costs"]["cumulative_attempt_wall_seconds"] += summary["costs"]["inherited_fit_wall_seconds"]
    expected = rebind(folder, summary)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError):
        r.generate(folder, expected, out)
    assert not out.exists()


def test_failed_gate_is_reported_without_selecting_successful_fits(saved, tmp_path):
    folder, summary, _ = saved
    summary["continuation_gate"]["checks"][0].update(actual=3.0, passed=False)
    summary["continuation_gate"]["passed"] = False
    expected = rebind(folder, summary)
    out = tmp_path / "failed-gate-report"
    r.generate(folder, expected, out)
    assert "**FAIL** (16/17 checks)" in (out / "report.md").read_text()
    assert len(rows(out / "all-fit-control.csv")) == 42
    assert len(rows(out / "all-fit-prediction.csv")) == 6


def test_receipt_scope_and_summary_binding_are_verified(saved, tmp_path):
    folder, _, _ = saved
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt["saved_output_only"] = False
    write_json(folder / "receipt.json", receipt)
    expected = r.digest((folder / "receipt.json").read_bytes())
    with pytest.raises(ValueError, match="completed saved-output-only v2 audit"):
        r.generate(folder, expected, tmp_path / "not-saved")
    receipt["saved_output_only"] = True
    receipt["plan_sha256"] = "e" * 64
    write_json(folder / "receipt.json", receipt)
    expected = r.digest((folder / "receipt.json").read_bytes())
    with pytest.raises(ValueError, match="Summary/receipt binding"):
        r.generate(folder, expected, tmp_path / "not-bound")


def test_symlinks_unbound_members_and_nonexclusive_outputs_are_rejected(saved, tmp_path):
    folder, _, expected = saved
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        r.generate(folder, expected, out)
    with pytest.raises(ValueError, match="cannot alter"):
        r.generate(folder, expected, folder / "report")
    extra = folder / "unbound.txt"
    extra.write_text("not authenticated")
    with pytest.raises(ValueError, match="Unexpected or missing"):
        r.generate(folder, expected, tmp_path / "unbound")
    extra.unlink()
    target = tmp_path / "readme-copy"
    target.write_bytes((folder / "README.md").read_bytes())
    (folder / "README.md").unlink()
    (folder / "README.md").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks forbidden"):
        r.generate(folder, expected, tmp_path / "symlink")


@pytest.mark.parametrize("payload", [b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}'])
def test_ambiguous_or_nonfinite_json_rejected(payload):
    with pytest.raises(ValueError):
        r.read_json(payload)


def test_command_line_uses_only_completed_audit_directory(saved, tmp_path):
    folder, _, expected = saved
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--audit", str(folder),
        "--expected-audit-receipt-sha256", expected, "--out", str(tmp_path / "cli-report"),
    ], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == {
        "status": "completed", "control_rows": 42, "training_rows": 288, "criteria_rows": 17,
    }
