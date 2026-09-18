"""Publish complete tables from an externally authenticated, completed v2 audit.

Standard-library only. Never reads execution records, loads a model, runs native
replay, or computes new bootstrap samples. The audit directory is the sole input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path, PurePosixPath

VERSION = "reacher-reward-residual-control-v2"
FIT_ORDER = ("free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "zero", "uniform")
COST_FIELDS = (
    "inherited_fit_wall_seconds", "prior_invalid_attempt_wall_seconds",
    "new_evaluation_wall_seconds", "new_control_setup_and_decision_seconds",
    "cumulative_attempt_wall_seconds", "fresh_evaluation_plus_inherited_fits_seconds",
    "audit_validation_wall_seconds",
)
LOG_FIELDS = (
    "epoch", "loss", "observation_mse", "reward_mse", "kl_nats",
    "rollout_observation_mse", "rollout_reward_mse", "gradient_norm",
    "valid_observation_targets", "valid_rollout_starts",
)
PREDICTION_FIELDS = (
    "one_step", "persistence_one_step", "open_loop", "persistence_open_loop",
    "open_loop_valid_root_and_endpoint", "blackout_angle", "reward_mse",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_json(data):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique_keys(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_keys)

    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), "Nonfinite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    finite(result)
    return result


def arms(panel):
    names = list(FIT_ORDER)
    if panel != "full":
        names += [name + "-reset" for name in FIT_ORDER]
    return names + list(REFERENCES)


def criterion_names():
    names = ["physics_ordinary_vs_zero"]
    for panel in ("ordinary", "shift"):
        for seed in (271, 283, 293):
            names += [f"residual-{seed}/{panel}/vs_zero", f"residual-{seed}/{panel}/vs_paired_free"]
        names.append(f"residual/{panel}/mean_vs_free")
    return names + ["residual/reward_mse_vs_free", "residual/one_step_angle_noninferiority"]


def authenticate(audit, expected):
    """Authenticate every audit artifact without following execution/source paths."""
    audit = Path(audit)
    require(valid_sha(expected), "External audit receipt SHA-256 required")
    require(audit.is_dir() and not audit.is_symlink(), "Audit must be an ordinary directory")
    require(not any(path.is_symlink() for path in audit.rglob("*")), "Audit symlinks forbidden")
    receipt_bytes = (audit / "receipt.json").read_bytes()
    require(digest(receipt_bytes) == expected, "Audit receipt identity mismatch")
    receipt = read_json(receipt_bytes)
    require(receipt.get("status") == "completed" and receipt.get("version") == VERSION
            and receipt.get("saved_output_only") is True, "Require completed saved-output-only v2 audit")
    files = receipt["files"]
    require(isinstance(files, dict) and set(files) == {"summary.json", "README.md"}, "Audit artifact membership")
    contents = {}
    for name, expected_hash in files.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and valid_sha(expected_hash),
                "Audit artifact path/hash")
        contents[name] = (audit / name).read_bytes()
        require(digest(contents[name]) == expected_hash, f"Audit artifact changed: {name}")
    actual = {str(path.relative_to(audit)) for path in audit.rglob("*") if path.is_file()}
    require(actual == set(files) | {"receipt.json"}, "Unexpected or missing audit artifact")
    summary = read_json(contents["summary.json"])
    require(summary.get("status") == "completed" and summary.get("version") == VERSION
            and summary.get("saved_output_only") is True, "Require completed saved-output-only v2 summary")
    for key in ("plan_sha256", "execution_completed_sha256", "training_source", "fit_source", "costs"):
        require(summary[key] == receipt[key], f"Summary/receipt binding: {key}")
    require(valid_sha(summary["plan_sha256"]) and valid_sha(summary["execution_completed_sha256"]),
            "Summary experiment identities")
    validate_summary(summary)
    return receipt, summary


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid {name}")


def validate_summary(summary):
    for name in ("new_model_calls", "new_policy_calls", "new_mpc_calls", "new_fits"):
        require(type(summary[name]) is int and summary[name] == 0, f"Saved-output-only scope: {name}")
    require(set(summary["fits"]) == set(FIT_ORDER) and set(summary["predictions"]) == set(FIT_ORDER),
            "All six inherited fits and prediction records required")
    require(set(summary["control"]) == set(PANELS), "All three control panels required")
    for panel in PANELS:
        require(set(summary["control"][panel]) == set(arms(panel)), f"Complete control membership: {panel}")
        for name, result in summary["control"][panel].items():
            nonnegative(result["mean_cost"], "mean episode cost")
            require(isinstance(result["episode_costs"], list) and len(result["episode_costs"]) == 64,
                    f"All 64 paired cases required: {panel}/{name}")
            for value in result["episode_costs"]:
                nonnegative(value, "episode cost")
    for name in FIT_ORDER:
        fit = summary["fits"][name]
        kind, seed = name.rsplit("-", 1)
        require(fit["kind"] == kind and fit["seed"] == int(seed)
                and fit["residual_reward"] is (kind == "residual"), "Inherited fit identity")
        require(isinstance(fit["epochs"], list) and len(fit["epochs"]) == 48, "All 48 training epochs required")
        for epoch, row in enumerate(fit["epochs"], 1):
            require(set(row) == set(LOG_FIELDS) and type(row["epoch"]) is int and row["epoch"] == epoch,
                    "Complete ordered training metrics required")
        require(set(summary["predictions"][name]) == set(PREDICTION_FIELDS), "Complete prediction metrics required")
    checks = summary["continuation_gate"]["checks"]
    require(isinstance(checks, list) and [item["name"] for item in checks] == criterion_names(),
            "All 17 original continuation criteria required in order")
    for item in checks:
        require(item["direction"] in ("le", "lt") and type(item["passed"]) is bool, "Criterion schema")
        actual, threshold = item["actual"], item["threshold"]
        nonnegative(actual, "criterion actual")
        nonnegative(threshold, "criterion threshold")
        require(item["passed"] == (actual < threshold if item["direction"] == "lt" else actual <= threshold),
                "Criterion pass arithmetic")
    require(type(summary["continuation_gate"]["passed"]) is bool
            and summary["continuation_gate"]["passed"] == all(item["passed"] for item in checks),
            "Continuation gate arithmetic")
    costs = summary["costs"]
    for name in COST_FIELDS:
        nonnegative(costs[name], name)
    require(type(costs["new_fits"]) is int and costs["new_fits"] == 0, "Cost new-fit scope")
    require(math.isclose(costs["cumulative_attempt_wall_seconds"],
                         costs["prior_invalid_attempt_wall_seconds"] + costs["new_evaluation_wall_seconds"],
                         rel_tol=1e-12, abs_tol=1e-9), "Historical attempt cost double-counted or omitted")
    require(math.isclose(costs["fresh_evaluation_plus_inherited_fits_seconds"],
                         costs["inherited_fit_wall_seconds"] + costs["new_evaluation_wall_seconds"],
                         rel_tol=1e-12, abs_tol=1e-9), "Successful-pipeline-equivalent cost arithmetic")
    require(costs["inherited_fit_wall_seconds"] <= costs["prior_invalid_attempt_wall_seconds"],
            "Inherited fitting already belongs to prior attempt")


def flatten(value, prefix=""):
    """Preserve every scalar and serialize arrays losslessly into one CSV cell."""
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name + "_json"] = json.dumps(item, separators=(",", ":"), allow_nan=False)
        else:
            result[name] = item
    return result


def write_csv(path, rows, first=()):
    columns = list(first) + sorted({key for row in rows for key in row} - set(first))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def display(value):
    if value is None:
        return "not applicable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(display(value) for value in row) + " |" for row in rows]
    return lines


def report_lines(summary, expected):
    checks = summary["continuation_gate"]["checks"]
    passed = sum(item["passed"] for item in checks)
    lines = [
        "# Reacher reward-residual evaluation: complete saved-output report", "",
        f"Continuation gate: **{'PASS' if summary['continuation_gate']['passed'] else 'FAIL'}** ({passed}/17 checks).",
        "Six unchanged inherited fits; zero new fits. All 42 controller/panel rows are included.", "",
        "This is a conventional known-reward-structure comparison. It establishes no biological or connectome advantage, new RL algorithm, memory advantage, or ICLR readiness.", "",
        f"Authenticated audit receipt SHA-256: `{expected}`.",
        f"Frozen plan SHA-256: `{summary['plan_sha256']}`.",
        f"Execution completion SHA-256: `{summary['execution_completed_sha256']}`.", "",
        "This report only reads authenticated audit files. It performs no model inference, simulator replay, training, new evaluation, or bootstrap sampling.", "",
        "## Costs", "",
        "Times below are seconds. The prior invalid attempt already contains inherited training. Actual cumulative attempt time adds prior attempt and new evaluation once. The successful-pipeline equivalent is descriptive and is not actual cumulative cost. Audit validation is separate and excludes audit publication writes and final rehashing; report generation is also separate.", "",
        *table(("Metric", "Seconds"), [(key, summary["costs"][key]) for key in COST_FIELDS]), "",
        "The control setup/decision subtotal is included in new evaluation time. Batch decision latency includes the full case batch; per-case values are amortized and do not establish an individual deadline.", "",
        "## All control results", "",
        "Lower episode cost is better. Physics references use supplied dynamics/state information. Reset variants are diagnostics. Floor references do not use a planner. [All control metrics and all 64 case costs for every row](all-fit-control.csv) preserve numeric precision.", "",
        *table(("Panel", "Arm", "Mean episode cost", "Setup seconds", "Decision seconds", "Batch p95 seconds"), [
            (panel, name, item["mean_cost"], item["setup_seconds"], item["decision_wall_seconds"],
             item["batch_latency_seconds"]["p95"])
            for panel in PANELS for name in arms(panel) for item in [summary["control"][panel][name]]
        ]), "", "## Every continuation criterion", "",
        "All 17 checks must pass. Thresholds and directions are unchanged. [Machine-readable criteria](continuation-criteria.csv).", "",
        *table(("Criterion", "Actual", "Comparison", "Threshold", "Passed"), [
            (item["name"], item["actual"], "<" if item["direction"] == "lt" else "<=", item["threshold"], item["passed"])
            for item in checks
        ]), "", "## Paired comparisons and uncertainty", "",
        "These are saved episode-paired percentile bootstrap intervals, conditional on the six inherited fits (three paired seeds). They are not uncertainty intervals over architectures or newly trained models. Reset comparisons remain descriptive. No new bootstrap samples were generated.", "",
        *table(("Saved comparison metric", "Value"), flatten(summary["paired_descriptive_comparisons"]).items()), "",
        "## Every fit's training and prediction metrics", "",
        "All 48 logged epochs for each inherited fit appear below. They are historical training records, not rerun optimization. Prediction metrics come from the fresh common held-out corpus. Tables round for display; CSV files retain the parsed numeric precision.", "",
        "[Fit metadata](all-fit-metadata.csv) | [All training epochs](all-fit-training.csv) | [All prediction metrics](all-fit-prediction.csv)", "",
    ]
    for name in FIT_ORDER:
        fit = summary["fits"][name]
        lines += [f"### {name}", "", "Fit metadata:", "",
                  *table(("Field", "Value"), flatten({key: value for key, value in fit.items() if key != "epochs"}).items()),
                  "", "Held-out prediction metrics:", "",
                  *table(("Metric", "Value"), flatten(summary["predictions"][name]).items()), "",
                  "<details>", f"<summary>All 48 training epochs for {name}</summary>", "",
                  *table(LOG_FIELDS, [[row[key] for key in LOG_FIELDS] for row in fit["epochs"]]), "", "</details>", ""]
    lines += ["## Audit scope and limits", "",
              *table(("Audit fact", "Value"), [
                  ("Native transitions checked", summary["native_transitions_checked"]),
                  ("Maximum native replay absolute error", summary["native_max_abs_error"]),
                  *flatten(summary["random_streams"]).items(),
              ]), "", *(f"- {display(limit)}" for limit in summary["limits"]), ""]
    return lines


def generate(audit, expected, out):
    start = time.monotonic()
    audit, out = Path(audit), Path(out)
    require(not out.exists() and not out.is_symlink(), "Report output already exists")
    require(not out.resolve().is_relative_to(audit.resolve()), "Report cannot alter its authenticated audit directory")
    receipt, summary = authenticate(audit, expected)
    control_rows = [{"panel": panel, "arm": name, **flatten(summary["control"][panel][name])}
                    for panel in PANELS for name in arms(panel)]
    training_rows = [{"fit": name, **row} for name in FIT_ORDER for row in summary["fits"][name]["epochs"]]
    prediction_rows = [{"fit": name, **flatten(summary["predictions"][name])} for name in FIT_ORDER]
    metadata_rows = [{"fit": name, **flatten({key: value for key, value in summary["fits"][name].items()
                                             if key != "epochs"})} for name in FIT_ORDER]
    text = "\n".join(report_lines(summary, expected))
    # Reauthenticate inputs before publication without reading any execution paths.
    require(authenticate(audit, expected) == (receipt, summary), "Audit changed during reporting")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "all-fit-control.csv", control_rows, ("panel", "arm", "mean_cost"))
    write_csv(out / "all-fit-training.csv", training_rows, ("fit", "epoch"))
    write_csv(out / "all-fit-prediction.csv", prediction_rows, ("fit",))
    write_csv(out / "all-fit-metadata.csv", metadata_rows, ("fit",))
    write_csv(out / "continuation-criteria.csv", summary["continuation_gate"]["checks"], ("name",))
    write_csv(out / "costs.csv", [{"metric": name, "seconds": summary["costs"][name]} for name in COST_FIELDS],
              ("metric", "seconds"))
    (out / "report.md").write_text(text)
    members = {path.name: digest(path.read_bytes()) for path in sorted(out.iterdir())}
    result = {
        "status": "completed", "version": VERSION, "saved_output_only": True,
        "audit_receipt_sha256": expected, "plan_sha256": summary["plan_sha256"],
        "source_sha256": digest(Path(__file__).read_bytes()), "control_rows": len(control_rows),
        "training_rows": len(training_rows), "prediction_rows": len(prediction_rows), "criteria_rows": 17,
        "new_model_calls": 0, "new_simulator_calls": 0, "new_fits": 0,
        "report_generation_wall_seconds": time.monotonic() - start, "files": members,
    }
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.audit, args.expected_audit_receipt_sha256, args.out)
    print(json.dumps({key: result[key] for key in ("status", "control_rows", "training_rows", "criteria_rows")}))


if __name__ == "__main__":
    main()
