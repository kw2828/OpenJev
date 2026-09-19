"""Render only completed, authenticated Reacher cache-audit artifacts.

No runner, model, simulator, optimizer or research component is imported.
Engineering output is explicitly labeled and cannot masquerade as scored output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = "reacher-cache-ablation-v1"
ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "packet_mlp", "cached_mlp")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "zero", "uniform", "public_kinematic")
LABELS = {
    "residual_gru": "Persistent GRU",
    "encoded_current_gru": "Encoded-current GRU",
    "cached_gru": "Cached-angle GRU",
    "packet_mlp": "Packet MLP",
    "cached_mlp": "Cached-angle MLP",
    "known_state": "Known-state physics",
    "particle": "Particle physics",
    "public_kinematic": "Public kinematics",
    "zero": "Zero action",
    "uniform": "Uniform action",
}
COLORS = ("#007C91", "#E08E35", "#8E6BB4", "#4778B3", "#B05B86")
CONTRASTS = (
    ("persistent_minus_cached_gru", "residual_gru", "cached_gru", "primary"),
    ("persistent_minus_cached_mlp", "residual_gru", "cached_mlp", "primary"),
    ("gru_cache_information", "cached_gru", "encoded_current_gru", "secondary"),
    ("mlp_cache_information", "cached_mlp", "packet_mlp", "secondary"),
)
SCORED_PLAN_SHA = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON member")
            result[key] = value
        return result

    value = json.loads(
        Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    finite_json(value)
    return value


def finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON value")
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def close(left, right, label):
    require(
        type(left) in (int, float)
        and type(right) in (int, float)
        and math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-7),
        label,
    )


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checked(path, digest):
    path = Path(path)
    require(
        isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
        "External SHA-256 required",
    )
    require(
        path.is_file() and not path.is_symlink() and sha(path) == digest, f"Artifact hash mismatch: {path}"
    )
    return path


def member(folder, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact member path")
    path = Path(folder) / relative
    require(path.resolve().is_relative_to(Path(folder).resolve()), "Artifact escaped its parent")
    require(
        not any(parent.is_symlink() for parent in (path, *path.parents) if parent != parent.parent),
        "Artifact symlink",
    )
    return path


def authenticate(plan_path, plan_sha, audit_path, audit_sha, execution, *, engineering=False):
    plan = read(checked(plan_path, plan_sha))
    receipt = read(checked(audit_path, audit_sha))
    require(plan["study"] == receipt["version"] == STUDY, "Cache study identity")
    require(
        plan["engineering"] is receipt["engineering"] is engineering, "Explicit engineering/scored boundary"
    )
    require(
        plan["rng_namespace"].startswith("reacher-cache-engineering-")
        if engineering
        else plan["rng_namespace"] == STUDY + "-scored",
        "Declared cache RNG scope",
    )
    require(
        receipt["status"] == "completed"
        and receipt["saved_output_only"] is True
        and receipt["plan_sha256"] == plan_sha
        and receipt["source_sha256"] == plan["sources"]
        and receipt["runtime"] == plan["runtime"],
        "Completed, source-bound saved-output audit required",
    )
    if not engineering:
        require(
            plan_sha == SCORED_PLAN_SHA and len(plan["sources"]) == 70,
            "Exact frozen scored plan and all 70 sources required",
        )
    for name, digest in plan["sources"].items():
        checked(member(ROOT, name), digest)
    folder = Path(audit_path).parent
    require(set(receipt["files"]) == {"summary.json", "README.md"}, "Exact completed audit membership")
    require(
        not (folder / "failed.json").exists() and not (folder / "over-cap-receipt.json").exists(),
        "Failed audit cannot be reported as completed",
    )
    for name, digest in receipt["files"].items():
        checked(member(folder, name), digest)
    summary = read(folder / "summary.json")
    require(
        summary["status"] == "completed"
        and summary["version"] == STUDY
        and summary["engineering"] is engineering
        and summary["plan_sha256"] == plan_sha
        and summary["execution_completed_sha256"] == receipt["execution_completed_sha256"]
        and summary["saved_output_only"] is True
        and summary["costs"] == receipt["costs"],
        "Audit summary identity",
    )
    require(
        all(summary[key] == 0 for key in ("new_model_calls", "new_policy_calls", "new_fits")),
        "Saved-artifact-only audit scope",
    )
    execution = Path(execution)
    completed = read(checked(execution / "completed.json", receipt["execution_completed_sha256"]))
    require(
        completed["status"] == "completed"
        and completed["study"] == STUDY
        and completed["plan_sha256"] == plan_sha
        and completed["files"] == receipt["execution_members"]
        and completed["fits"] == 15
        and completed["control_rows"] == 60
        and completed["prediction_episodes"] == plan["prediction_episodes"],
        "Completed full execution binding",
    )
    require(
        0 < completed["wall_seconds"] <= plan["cap_seconds"]
        and 0 < summary["costs"]["audit_validation_wall_seconds"] <= plan["audit_cap_seconds"],
        "Completed execution and audit within frozen caps",
    )
    close(completed["wall_seconds"], summary["costs"]["new_execution_wall_seconds"], "Execution wall binding")
    close(
        completed["cumulative_attempt_wall_seconds"],
        summary["costs"]["cumulative_attempt_wall_seconds"],
        "Cumulative wall binding",
    )
    paths = list(execution.rglob("*"))
    require(
        not any(path.is_symlink() for path in paths)
        and {path.relative_to(execution).as_posix() for path in paths if path.is_file()}
        == set(completed["files"]) | {"completed.json"},
        "Exact completed execution membership",
    )
    for name, digest in completed["files"].items():
        checked(member(execution, name), digest)
    validate_summary(plan, summary)
    return (
        plan,
        summary,
        {
            "plan_sha256": plan_sha,
            "audit_receipt_sha256": audit_sha,
            "audit_summary_sha256": receipt["files"]["summary.json"],
            "execution_completed_sha256": receipt["execution_completed_sha256"],
            "execution_member_count": len(completed["files"]),
            "frozen_source_count": len(plan["sources"]),
        },
    )


def validate_summary(plan, summary):
    finite_json(summary)
    names = {f"{arm}-{pair}" for arm in ARMS for pair in PAIRS}
    n, steps = plan["control_episodes"], plan["steps"]
    require(type(n) is int and n > 0 and steps == 50, "Real 50-step horizon and case denominator")
    if not plan["engineering"]:
        require(
            n == 64
            and plan["prediction_episodes"] == 96
            and plan["train_episodes"] == 768
            and plan["epochs"] == 48
            and plan["batch_size"] == 32,
            "Frozen production coverage",
        )
    require(
        plan["arms"] == list(ARMS)
        and plan["pairs"] == list(PAIRS)
        and plan["panels"] == list(PANELS)
        and plan["references"] == list(REFERENCES),
        "Exact study arms and panels",
    )
    require(
        set(summary["fits"]) == set(summary["prediction"]) == names
        and set(summary["control"]) == set(PANELS),
        "All fifteen fits and three panels required",
    )
    require(
        summary["coverage"]["fits"] == 15 and summary["coverage"]["control_rows"] == 60,
        "All 60 control rows retained",
    )
    coverage = summary["coverage"]
    require(
        coverage["fit_names"] == coverage["prediction_names"] == plan["fit_order"]
        and set(plan["fit_order"]) == names
        and len(plan["fit_order"]) == 15
        and coverage["control_paths"] == [row["path"] for row in plan["execution_order"]]
        and len(set(coverage["control_paths"])) == 60
        and coverage["learned_control_rows"] == 45
        and coverage["reference_rows"] == 15
        and coverage["control_cases_per_row"] == n
        and coverage["native_control_transitions"] == 60 * n * steps
        and coverage["native_prediction_transitions"] == plan["prediction_episodes"] * steps
        and coverage["criterion_checks"] == 28,
        "Exact fit/row/transition coverage",
    )
    updates = plan["epochs"] * (plan["train_episodes"] // plan["batch_size"])
    require(
        coverage["optimizer_updates_per_fit"] == updates
        and coverage["total_optimizer_updates"] == 15 * updates,
        "Complete update coverage",
    )
    for panel, rows in summary["control"].items():
        require(set(rows) == names | set(REFERENCES), f"All fits/references in {panel}")
        for name, row in rows.items():
            costs = np.asarray(row["episode_costs"], dtype=float)
            require(
                costs.shape == (plan["control_episodes"],)
                and np.isfinite(costs).all()
                and math.isclose(float(costs.mean()), row["mean_cost"], rel_tol=1e-12, abs_tol=1e-12),
                "Native mean cost arithmetic",
            )
            decisions = np.asarray(row["decision_seconds"], dtype=float)
            require(
                decisions.shape == (steps,) and np.isfinite(decisions).all() and (decisions > 0).all(),
                "All measured decision times retained",
            )
            close(row["decision_wall_seconds"], float(decisions.sum()), "Decision time sum")
            close(
                row["per_case_amortized_seconds"],
                float(decisions.mean() / n),
                "Amortized decision denominator",
            )
            for key, value in {
                "mean": decisions.mean(),
                "p50": np.quantile(decisions, 0.5),
                "p95": np.quantile(decisions, 0.95),
                "max": decisions.max(),
            }.items():
                close(row["batch_latency_seconds"][key], float(value), "Batch latency arithmetic")
            require(
                row["setup_seconds"] >= 0
                and row["native_step_seconds"] >= 0
                and 0 < row["row_wall_seconds"]
                and row["setup_seconds"] + row["decision_wall_seconds"] + row["native_step_seconds"]
                <= row["row_wall_seconds"] + 1e-6,
                "Nonoverlapping control row timing",
            )
            if name in names:
                work = row["state_and_work"]
                require(
                    type(work["state_arrays_bytes"]) is int
                    and work["state_arrays_bytes"] > 0
                    and work["state_arrays_bytes"] % (2 * n * steps) == 0
                    and work["public_state_buffers_checked"] is True,
                    "Saved root and carried state payloads",
                )
    for name in names:
        require(
            set(summary["prediction"][name]["endpoints"]) == {"1", "3", "7"}, "All endpoint horizons retained"
        )
        fit = summary["fits"][name]
        arm, pair = name.rsplit("-", 1)
        require(
            fit["arm"] == arm
            and fit["pair"] == pair
            and fit["updates"] == updates
            and fit["model_configuration"]["model_class"] == plan["model_classes"][arm],
            "Each actual-class paired fit identity",
        )
        require(
            0 <= fit["setup_seconds"]
            and 0 < fit["training_seconds"]
            and fit["setup_seconds"] + fit["training_seconds"] <= fit["wall_seconds"] + 1e-6,
            "Complete fit wall time",
        )
    gate = summary["continuation_gate"]
    require(
        len(gate["checks"]) == plan["criterion"]["expected_checks"] == 28
        and len({row["name"] for row in gate["checks"]}) == 28,
        "All 28 continuation checks retained",
    )
    for row in gate["checks"]:
        require(row["comparison"] in ("le", "lt"), "Declared comparison operator")
        passed = row["left"] < row["right"] if row["comparison"] == "lt" else row["left"] <= row["right"]
        require(row["passed"] is bool(passed), "Continuation comparison arithmetic")
    require(
        gate["passed"] is all(row["passed"] for row in gate["checks"]),
        "Whole gate preserves every failed check",
    )
    expected = expected_gate(summary["control"])
    require(
        [row["name"] for row in gate["checks"]] == [row["name"] for row in expected],
        "Exact 28 prospective comparison identities",
    )
    for actual, wanted in zip(gate["checks"], expected, strict=True):
        close(actual["left"], wanted["left"], "Gate observed control arithmetic")
        close(actual["right"], wanted["right"], "Gate frozen threshold arithmetic")
        require(
            actual["comparison"] == wanted["comparison"] and actual["passed"] is wanted["passed"],
            "All failed checks retained",
        )
    paired_rows(plan, summary)
    rows = [row for panel in summary["control"].values() for row in panel.values()]
    costs = summary["costs"]
    for cost_key, row_key in (
        ("control_row_wall_seconds", "row_wall_seconds"),
        ("control_setup_seconds", "setup_seconds"),
        ("control_decision_seconds", "decision_wall_seconds"),
        ("control_native_step_seconds", "native_step_seconds"),
    ):
        close(costs[cost_key], sum(row[row_key] for row in rows), "Whole control phase cost")
    close(
        costs["fit_wall_seconds"],
        sum(row["wall_seconds"] for row in summary["fits"].values()),
        "All 15 fit costs",
    )
    close(
        costs["cumulative_attempt_wall_seconds"],
        costs["prior_costs"]["cumulative_attempt_wall_seconds"] + costs["new_execution_wall_seconds"],
        "Prior attempts charged once",
    )
    require(
        costs["new_fits"] == 15 and costs["astra_calls"] == 0 and costs["compute_matched"] is False,
        "Actual training and compute scope",
    )


def expected_gate(controls):
    """Independent saved-number arithmetic, no scientific rerun or new criterion."""

    def family(panel, arm):
        return np.mean([controls[panel][f"{arm}-{pair}"]["episode_costs"] for pair in PAIRS], axis=0)

    rows = []

    def add(name, left, right):
        rows.append(
            {
                "name": name,
                "left": float(left),
                "right": float(right),
                "comparison": "le",
                "passed": bool(left <= right),
            }
        )

    for panel in ("ordinary", "shift"):
        for arm in ("cached_gru", "cached_mlp"):
            add(
                f"{panel}: persistent mean improves {arm} by 3%",
                family(panel, "residual_gru").mean(),
                0.97 * family(panel, arm).mean(),
            )
            for pair in PAIRS:
                add(
                    f"{panel}/{pair}: persistent nonworse than {arm}",
                    controls[panel][f"residual_gru-{pair}"]["mean_cost"],
                    controls[panel][f"{arm}-{pair}"]["mean_cost"],
                )
        zero = controls[panel]["zero"]["mean_cost"]
        for pair in PAIRS:
            add(
                f"{panel}/{pair}: persistent beats zero by 10%",
                controls[panel][f"residual_gru-{pair}"]["mean_cost"],
                0.9 * zero,
            )
        for reference in ("known_state", "particle"):
            add(
                f"{panel}/{reference}: physics beats zero by 10%",
                controls[panel][reference]["mean_cost"],
                0.9 * zero,
            )
    for arm in ("cached_gru", "cached_mlp"):
        add(
            f"full: persistent mean degrades {arm} at most 2%",
            family("full", "residual_gru").mean(),
            1.02 * family("full", arm).mean(),
        )
    return rows


def paired_rows(plan, summary):
    """Reuse audited case intervals; compute only all-fit descriptive differences."""
    result = []
    require(set(summary["paired_descriptive_comparisons"]) == set(PANELS), "Every paired panel retained")
    for panel in PANELS:
        expected = {f"persistent_minus_{arm}" for arm in ARMS if arm != "residual_gru"}
        expected |= {"gru_cache_information", "mlp_cache_information", "persistent_minus_public_kinematic"}
        require(
            set(summary["paired_descriptive_comparisons"][panel]) == expected,
            "All seven audited contrasts retained",
        )
        for name, treatment, control, purpose in CONTRASTS:
            one = [summary["control"][panel][f"{treatment}-{pair}"] for pair in PAIRS]
            two = [summary["control"][panel][f"{control}-{pair}"] for pair in PAIRS]
            differences = np.mean([row["episode_costs"] for row in one], axis=0) - np.mean(
                [row["episode_costs"] for row in two], axis=0
            )
            audited = summary["paired_descriptive_comparisons"][panel][name]
            close(
                audited["mean_cost_difference"], float(differences.mean()), "Audited paired mean difference"
            )
            interval = np.asarray(audited["episode_paired_percentile_95"], dtype=float)
            require(
                interval.shape == (2,)
                and np.isfinite(interval).all()
                and interval[0] <= interval[1]
                and audited["cases"] == plan["control_episodes"]
                and audited["conditional_on_all_three_saved_fit_pairs"] is True,
                "Case-only paired interval scope",
            )
            result.append(
                {
                    "panel": panel,
                    "name": name,
                    "treatment": treatment,
                    "control": control,
                    "purpose": purpose,
                    "pair_ids": list(PAIRS),
                    "paired_fit_mean_differences": [
                        left["mean_cost"] - right["mean_cost"] for left, right in zip(one, two, strict=True)
                    ],
                    "mean_cost_difference": audited["mean_cost_difference"],
                    "episode_paired_percentile_95": audited["episode_paired_percentile_95"],
                    "interval_scope": "Audited case bootstrap conditional on all 3 saved fit pairs and one corpus; not fit uncertainty, multiplicity-adjusted inference or independent confirmation.",
                }
            )
    return result


def summarize(plan, summary, bindings):
    families, per_fit, references = [], [], []
    n, steps = plan["control_episodes"], plan["steps"]
    for panel in PANELS:
        for arm in ARMS:
            costs = []
            for pair in PAIRS:
                name = f"{arm}-{pair}"
                control, fit = summary["control"][panel][name], summary["fits"][name]
                work = control["state_and_work"]
                row = {
                    "panel": panel,
                    "arm": arm,
                    "fit": name,
                    "native_cost_mean": control["mean_cost"],
                    "native_episode_costs": control["episode_costs"],
                    "decision_ms_per_case_amortized": 1000 * control["per_case_amortized_seconds"],
                    "whole_row_ms_per_case_decision": 1000 * control["row_wall_seconds"] / (n * steps),
                    "batch_latency_seconds": control["batch_latency_seconds"],
                    "setup_seconds": control["setup_seconds"],
                    "decision_seconds": control["decision_wall_seconds"],
                    "native_step_seconds": control["native_step_seconds"],
                    "row_wall_seconds": control["row_wall_seconds"],
                    "candidate_evaluations": control["candidate_evaluations"],
                    "imagined_transitions": control["imagined_transitions"],
                    "clipping_fraction": control["clipping_fraction"],
                    "parameters": fit["parameters"],
                    "root_state_bytes_per_case": work["state_arrays_bytes"] // (2 * n * steps),
                    "saved_root_and_carried_state_bytes": work["state_arrays_bytes"],
                    "maximum_single_candidate_batch_state_bytes": work[
                        "maximum_candidate_state_payload_bytes"
                    ],
                    "model_work": work["model_work"],
                }
                per_fit.append(row)
                costs.append(row["native_cost_mean"])
            families.append(
                {
                    "panel": panel,
                    "arm": arm,
                    "fits": [f"{arm}-{pair}" for pair in PAIRS],
                    "native_cost_mean": float(np.mean(costs)),
                    "fit_means": costs,
                    "fit_mean_min": min(costs),
                    "fit_mean_max": max(costs),
                    "spread_definition": "Range across all three fit means, not a confidence interval.",
                }
            )
        for reference in REFERENCES:
            control = summary["control"][panel][reference]
            references.append(
                {
                    "panel": panel,
                    "reference": reference,
                    **control,
                    "whole_row_ms_per_case_decision": 1000 * control["row_wall_seconds"] / (n * steps),
                }
            )
    return {
        "study": STUDY,
        "scope": "engineering_only_no_efficacy"
        if plan["engineering"]
        else "completed_scored_development_study",
        "engineering": plan["engineering"],
        "bindings": bindings,
        "continuation_gate": summary["continuation_gate"],
        "coverage": summary["coverage"],
        "family_control": families,
        "per_fit_control": per_fit,
        "reference_control": references,
        "utility_cost_plot": {
            "x": "whole_row_ms_per_case_decision",
            "y": "native episode cost mean",
            "x_scale": "log",
            "learned_points": len(per_fit),
            "reference_points": len(references),
            "timing_definition": "1000 * recorded control row wall seconds / (cases * real decisions); includes row setup, native steps and saved traces. Global validation/hashing and training are separate.",
            "selection": "All three fits individually, plus every reference; no selected frontier or winner.",
        },
        "fits": summary["fits"],
        "prediction": summary["prediction"],
        "paired_control": paired_rows(plan, summary),
        "paired_descriptive_comparisons": summary["paired_descriptive_comparisons"],
        "costs": summary["costs"],
        "limits": summary["limits"]
        + [
            "Native cost is lower-is-better; all family means weight the three fits equally, without selecting a winner.",
            "Learned controls use CEM256; supplied-physics references use 64 proposals. These are not matched-total-compute comparisons.",
            "Decision times are shared-host batch-amortized throughput, not isolated single-case latency. Setup, native steps and full row time are separate.",
            "The utility-versus-cost plot uses whole recorded control row time, including setup, native steps and traces, amortized per case and decision. Training remains separate; neither total compute nor hardware-isolated latency is matched.",
            "State sizes count explicit tensor payloads; they are not peak resident or accelerator memory. Counts and affine MAC estimates are not measured FLOPs.",
            "Prediction horizons 1/3/7 use observed root/endpoint masks. Hidden blackout angles are labels for audit diagnostics only.",
        ],
    }


def fmt(value):
    return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |" for row in rows),
        ]
    )


def markdown(report):
    gate = report["continuation_gate"]
    status = "PASS" if gate["passed"] else "FAIL"
    lines = [
        "# Reacher explicit-cache comparison",
        "",
        "**ENGINEERING FIXTURE ONLY. These values are not scientific results.**"
        if report["engineering"]
        else "Completed scored development study. Every fit and reference is retained.",
        "",
        f"Prespecified continuation: **{status}** ({sum(row['passed'] for row in gate['checks'])}/28 checks).",
        "",
        "## Native control costs",
        "",
        "Lower is better. Three-fit spread is descriptive, not a confidence interval.",
        "",
        table(
            ["Panel", "Family", "Mean of 3 fits", "Smallest fit mean", "Largest fit mean"],
            [
                (r["panel"], LABELS[r["arm"]], r["native_cost_mean"], r["fit_mean_min"], r["fit_mean_max"])
                for r in report["family_control"]
            ],
        ),
        "",
        "## Every control fit",
        "",
        "Decision milliseconds are per case after batch amortization; state bytes are payload per case at one real root.",
        "",
        table(
            [
                "Panel",
                "Fit",
                "Native cost",
                "Decision ms/case",
                "Whole-row ms/case/decision",
                "Root state bytes",
                "Parameters",
                "Full row s",
            ],
            [
                (
                    r["panel"],
                    r["fit"],
                    r["native_cost_mean"],
                    r["decision_ms_per_case_amortized"],
                    r["whole_row_ms_per_case_decision"],
                    r["root_state_bytes_per_case"],
                    r["parameters"],
                    r["row_wall_seconds"],
                )
                for r in report["per_fit_control"]
            ],
        ),
        "",
        "## Every reference",
        "",
        "Physics references use 64 proposals; learned controls use 256. Public kinematics has supplied physics and public observations; its result is descriptive only.",
        "",
        table(
            [
                "Panel",
                "Reference",
                "Native cost",
                "Decision ms/case",
                "Whole-row ms/case/decision",
                "Full row s",
            ],
            [
                (
                    r["panel"],
                    LABELS[r["reference"]],
                    r["mean_cost"],
                    1000 * r["per_case_amortized_seconds"],
                    r["whole_row_ms_per_case_decision"],
                    r["row_wall_seconds"],
                )
                for r in report["reference_control"]
            ],
        ),
        "",
        "## Native cost versus full control-row time",
        "",
        "See utility-vs-cost.png. All 45 learned fit/panel points and 15 reference points are shown, with a logarithmic time axis and no selected frontier.",
        "",
        report["utility_cost_plot"]["timing_definition"],
        "",
        "Times are shared-host and amortized, not isolated latency. The comparison is not matched total compute. Training costs remain in the separate table below.",
        "",
        "## Held-out predictions",
        "",
        "Endpoint angle MSE uses observed roots and endpoints. Counts for each mask are retained in report.json.",
        "",
        table(
            ["Fit", "Horizon 1 MSE", "Horizon 3 MSE", "Horizon 7 MSE", "Reward MSE", "Blackout angle MSE"],
            [
                (
                    name,
                    *(row["endpoints"][str(h)]["mse"] for h in (1, 3, 7)),
                    row["one_step_reward_mse"],
                    row["one_step_native_blackout_angles"]["mse"],
                )
                for name, row in report["prediction"].items()
            ],
        ),
        "",
        "## Training and deployment work",
        "",
        "Equal optimizer updates do not imply equal training compute. The cache controls reset learned state at every real packet while retaining explicit public measurements. Their bookkeeping and unconditional encoding are charged.",
        "",
        table(
            ["Fit", "Fit wall s", "Training s", "Forward s", "Backward s", "Updates", "Parameters"],
            [
                (
                    name,
                    row["wall_seconds"],
                    row["training_seconds"],
                    row["phase_seconds"]["forward_seconds"],
                    row["phase_seconds"]["backward_seconds"],
                    row["updates"],
                    row["parameters"],
                )
                for name, row in report["fits"].items()
            ],
        ),
        "",
        table(
            ["Recorded study cost", "Seconds"],
            [(name, value) for name, value in report["costs"].items() if name.endswith("seconds")],
        ),
        "",
        report["costs"]["accounting"],
        "",
        "Actual operation counts, search counts, tensor sizes, prediction costs, setup/native costs and prior costs are retained in report.json.",
        "",
        "## Every primary and secondary paired comparison",
        "",
        "Treatment minus control native cost: negative favors treatment. Dots in paired-comparisons.png retain all three fit pairs. Intervals are the audited case-bootstrap 95% percentiles, conditional on the three saved fits and one corpus. They do not measure fit uncertainty or independent confirmation, and no new bootstrap was drawn.",
        "",
        "Persistent-versus-cache contrasts are primary (3% improvement in gap panels, at most 2% degradation under full sensing, plus the complete 28-check gate). Cache-versus-current contrasts are secondary and cannot rescue a failed gate.",
        "",
        table(
            [
                "Panel",
                "Purpose",
                "Treatment - control",
                "Pair0",
                "Pair1",
                "Pair2",
                "Mean difference",
                "Conditional case interval",
            ],
            [
                (
                    r["panel"],
                    r["purpose"],
                    LABELS[r["treatment"]] + " - " + LABELS[r["control"]],
                    *r["paired_fit_mean_differences"],
                    r["mean_cost_difference"],
                    ", ".join(fmt(v) for v in r["episode_paired_percentile_95"]),
                )
                for r in report["paired_control"]
            ],
        ),
        "",
        "## All 28 continuation checks",
        "",
        table(
            ["Check", "Left", "Operator", "Threshold", "Result"],
            [
                (
                    row["name"],
                    row["left"],
                    "<" if row["comparison"] == "lt" else "<=",
                    row["right"],
                    "PASS" if row["passed"] else "FAIL",
                )
                for row in gate["checks"]
            ],
        ),
        "",
        "## Limits",
        "",
        *(f"- {value}" for value in report["limits"]),
        "",
        "## Artifact bindings",
        "",
        *(f"- {name}: `{value}`" for name, value in report["bindings"].items()),
        "",
    ]
    return "\n".join(lines)


def figure(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refs = ("known_state", "particle", "public_kinematic", "zero", "uniform")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), sharey=True, sharex=True)
    fig.patch.set_facecolor("white")
    gate = report["continuation_gate"]
    passed = sum(row["passed"] for row in gate["checks"])
    status = "PASS" if gate["passed"] else "FAIL"
    scope = (
        "ENGINEERING FIXTURE ONLY - NOT SCIENTIFIC RESULTS"
        if report["engineering"]
        else "Completed development study"
    )
    fig.suptitle(
        f"Explicit public cache in Reacher | continuation {status} ({passed}/28)",
        x=0.05,
        ha="left",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.05,
        0.915,
        scope + " | Fixed CEM256 learned controls | Lower native cost is better",
        fontsize=11,
        color="#9A3412" if report["engineering"] else "#333333",
    )
    families = {(row["panel"], row["arm"]): row for row in report["family_control"]}
    references = {(row["panel"], row["reference"]): row for row in report["reference_control"]}
    for ax, panel in zip(axes, PANELS, strict=True):
        for index, arm in enumerate(ARMS):
            row = families[panel, arm]
            ax.barh(index, row["native_cost_mean"], height=0.54, color=COLORS[index], alpha=0.28)
            ax.plot([row["fit_mean_min"], row["fit_mean_max"]], [index, index], color=COLORS[index], lw=2)
            ax.scatter(
                row["fit_means"],
                np.asarray([index - 0.14, index, index + 0.14]),
                s=34,
                facecolors=COLORS[index],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
        for index, reference in enumerate(refs, start=len(ARMS)):
            row = references[panel, reference]
            ax.scatter(row["mean_cost"], index, marker="D", s=38, color="#666666", zorder=3)
        ax.axhline(len(ARMS) - 0.5, color="#D6D6D6", linewidth=0.8)
        ax.set_title(
            {"full": "Full sensing", "ordinary": "6-step observation gaps", "shift": "10-step gap shift"}[
                panel
            ],
            loc="left",
            fontsize=12,
            pad=12,
        )
        ax.set_yticks(range(10), [LABELS[name] for name in (*ARMS, *refs)], fontsize=10)
        ax.set_xlabel("Mean native episode cost", fontsize=10)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    upper = max(
        [row["fit_mean_max"] for row in report["family_control"]]
        + [row["mean_cost"] for row in report["reference_control"]]
    )
    axes[0].set_xlim(0, max(upper * 1.06, 0.01))
    axes[0].invert_yaxis()
    fig.text(
        0.05,
        0.055,
        "Colored bars: mean across all 3 fits. Dots: each fit. Lines: range of fit means, not confidence intervals.\n"
        "Gray diamonds: all 5 references (physics planners use 64 proposals). Shared-host timings and full costs are in tables.md.\n"
        "One environment; incremental persistence beyond explicit caches does not identify inferred velocity or establish biological novelty.",
        fontsize=10,
        color="#444444",
        va="bottom",
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.83, bottom=0.22, wspace=0.16)
    fig.savefig(
        path,
        dpi=180,
        facecolor="white",
        metadata={"Description": scope, "Software": "OpenJev saved-artifact reporter"},
    )
    plt.close(fig)


def utility_cost_figure(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    learned, references = report["per_fit_control"], report["reference_control"]
    require(len(learned) == 45 and len(references) == 15, "Every utility/cost point retained")
    times = np.asarray([row["whole_row_ms_per_case_decision"] for row in (*learned, *references)])
    costs = np.asarray(
        [row["native_cost_mean"] for row in learned] + [row["mean_cost"] for row in references]
    )
    require(
        np.isfinite(times).all() and np.all(times > 0) and np.isfinite(costs).all(),
        "Finite costs and strictly positive whole-row times",
    )
    shapes = ("o", "^", "s")
    ref_shapes = {"known_state": "D", "particle": "P", "public_kinematic": "X", "zero": "v", "uniform": ">"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.4), sharex=True, sharey=True)
    gate = report["continuation_gate"]
    status = "PASS" if gate["passed"] else "FAIL"
    scope = (
        "ENGINEERING FIXTURE ONLY - NOT SCIENTIFIC RESULTS"
        if report["engineering"]
        else "Completed development study"
    )
    fig.suptitle(
        f"Native cost versus whole-row time | continuation {status} ({sum(r['passed'] for r in gate['checks'])}/28)",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.915,
        scope + " | All 15 fits and 5 references in each panel | Lower and left is better",
        fontsize=11,
        color="#9A3412" if report["engineering"] else "#333333",
    )
    for ax, panel in zip(axes, PANELS, strict=True):
        for row in learned:
            if row["panel"] != panel:
                continue
            pair = PAIRS.index(row["fit"].rsplit("-", 1)[1])
            ax.scatter(
                row["whole_row_ms_per_case_decision"],
                row["native_cost_mean"],
                marker=shapes[pair],
                color=COLORS[ARMS.index(row["arm"])],
                s=63,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.85,
                zorder=3,
            )
        for row in references:
            if row["panel"] == panel:
                ax.scatter(
                    row["whole_row_ms_per_case_decision"],
                    row["mean_cost"],
                    marker=ref_shapes[row["reference"]],
                    color="#555555",
                    s=65,
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=0.8,
                    zorder=2,
                )
        ax.set_xscale("log")
        ax.set_title(
            {"full": "Full sensing", "ordinary": "6-step observation gaps", "shift": "10-step gap shift"}[
                panel
            ],
            loc="left",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Whole-row ms / case / decision (log scale)", fontsize=10)
        ax.grid(True, which="major", color="#E4E4E4", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean native episode cost", fontsize=11)
    axes[0].set_xlim(times.min() * 0.75, times.max() * 1.4)
    axes[0].set_ylim(0, max(float(costs.max()) * 1.10, 0.01))
    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS[i], label=LABELS[arm] + " (CEM256)")
        for i, arm in enumerate(ARMS)
    ]
    handles.extend(
        Line2D(
            [],
            [],
            marker=ref_shapes[ref],
            linestyle="",
            color="#555555",
            label=LABELS[ref]
            + (
                " (64 proposals)"
                if ref in ("known_state", "particle", "public_kinematic")
                else " (no search)"
            ),
        )
        for ref in ("known_state", "particle", "public_kinematic", "zero", "uniform")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.245),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=2,
        handletextpad=0.5,
    )
    fig.text(
        0.055,
        0.035,
        "Circle / triangle / square: paired fits 0 / 1 / 2; all fits retained. Overlapping points remain at their measured coordinates.\n"
        "Time = full recorded control-row wall / (cases x decisions), including setup, native steps and traces; training/global hashing are separate.\n"
        "Shared-host amortized throughput, not isolated latency or matched total compute. Supplied-physics references use 64 proposals; learned controls use CEM256.",
        fontsize=9.5,
        color="#444444",
        va="bottom",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.33, wspace=0.15)
    fig.savefig(
        path,
        dpi=180,
        facecolor="white",
        metadata={"Description": scope, "Software": "OpenJev saved-artifact reporter"},
    )
    plt.close(fig)


def paired_figure(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = report["paired_control"]
    require(len(rows) == 12, "All 4 contrasts across all 3 panels required")
    coordinates = [
        value
        for row in rows
        for value in (
            *row["paired_fit_mean_differences"],
            *row["episode_paired_percentile_95"],
            row["mean_cost_difference"],
        )
    ]
    extent = max(max(abs(v) for v in coordinates), 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=True, sharey=True)
    labels = (
        "Persistent - cached GRU",
        "Persistent - cached MLP",
        "Cached - current GRU",
        "Cached - current MLP",
    )
    shapes, offsets = ("o", "^", "s"), (-0.15, 0, 0.15)
    for ax, panel in zip(axes, PANELS, strict=True):
        ax.axvline(0, color="#666666", lw=1)
        ax.axhline(1.5, color="#D6D6D6", lw=0.8)
        for index, (name, _, _, purpose) in enumerate(CONTRASTS):
            row = next(row for row in rows if row["panel"] == panel and row["name"] == name)
            color = "#007C91" if purpose == "primary" else "#8E6BB4"
            for pair, value in enumerate(row["paired_fit_mean_differences"]):
                ax.scatter(
                    value,
                    index + offsets[pair],
                    marker=shapes[pair],
                    color=color,
                    s=43,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
            low, high = row["episode_paired_percentile_95"]
            ax.plot((low, high), (index + 0.30, index + 0.30), color="#222222", lw=2)
            ax.scatter(row["mean_cost_difference"], index + 0.30, marker="D", color="#222222", s=34, zorder=4)
        ax.set_yticks(range(4), labels, fontsize=10)
        ax.set_ylim(3.65, -0.5)
        ax.set_xlim(-extent * 1.12, extent * 1.12)
        ax.set_title(
            {"full": "Full sensing", "ordinary": "6-step observation gaps", "shift": "10-step gap shift"}[
                panel
            ],
            loc="left",
            fontsize=12,
        )
        ax.set_xlabel("Treatment - control native cost", fontsize=10)
        ax.grid(axis="x", color="#E6E6E6", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    gate = report["continuation_gate"]
    scope = (
        "ENGINEERING FIXTURE ONLY - NOT SCIENTIFIC RESULTS"
        if report["engineering"]
        else "Completed development study"
    )
    fig.suptitle(
        f"Persistence beyond explicit caches | {'PASS' if gate['passed'] else 'FAIL'} ({sum(row['passed'] for row in gate['checks'])}/28)",
        x=0.045,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.905,
        scope + " | Negative favors the first model; all paired fits retained",
        fontsize=11,
        color="#9A3412" if report["engineering"] else "#333333",
    )
    handles = [
        Line2D([], [], linestyle="", marker=shape, color="#007C91", label=f"Fit pair {i}")
        for i, shape in enumerate(shapes)
    ]
    handles.append(Line2D([], [], marker="D", color="#222222", label="Mean + conditional case interval"))
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.20), ncol=4, frameon=False, fontsize=9
    )
    fig.text(
        0.045,
        0.035,
        "Blue: primary persistent-versus-cache contrasts. Purple: secondary cache-versus-current contrasts, which cannot rescue a failed gate.\n"
        "Black intervals: audited case-bootstrap 95% percentiles conditional on the three saved fits and one training corpus; not uncertainty across fits.\n"
        "Full sensing has a 2% maximum mean degradation threshold; gap panels require 3% mean improvement plus every paired fit nonworse. All 28 checks remain separate.\n"
        "No biological novelty, velocity-inference mechanism, cache equivalence or independent confirmation is established by this plot.",
        fontsize=9,
        color="#444444",
        va="bottom",
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.29, wspace=0.15)
    fig.savefig(
        path,
        dpi=180,
        facecolor="white",
        metadata={"Description": scope, "Software": "OpenJev saved-artifact reporter"},
    )
    plt.close(fig)


def render(args):
    out = Path(args.out)
    require(not out.exists(), "Exclusive report directory required")
    begin = time.monotonic()
    try:
        plan, summary, bindings = authenticate(
            args.plan,
            args.expected_plan_sha256,
            args.audit,
            args.expected_audit_sha256,
            args.execution,
            engineering=args.engineering,
        )
        report = summarize(plan, summary, bindings)
        out.mkdir(parents=True, exist_ok=False)
        write(out / "report.json", report)
        (out / "tables.md").write_text(markdown(report))
        figure(report, out / "native-costs.png")
        utility_cost_figure(report, out / "utility-vs-cost.png")
        paired_figure(report, out / "paired-comparisons.png")
        write(
            out / "receipt.json",
            {
                "status": "completed",
                "scope": report["scope"],
                "engineering": args.engineering,
                "source_sha256": sha(__file__),
                "inputs": bindings,
                "new_model_calls": 0,
                "new_simulator_calls": 0,
                "wall_seconds": time.monotonic() - begin,
                "files": {
                    name: sha(out / name)
                    for name in (
                        "report.json",
                        "tables.md",
                        "native-costs.png",
                        "utility-vs-cost.png",
                        "paired-comparisons.png",
                    )
                },
            },
        )
        return {
            "status": "completed",
            "engineering": args.engineering,
            "out": str(out),
            "continuation_passed": report["continuation_gate"]["passed"],
            "receipt_sha256": sha(out / "receipt.json"),
        }
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
        write(
            out / "failed.json",
            {
                "status": "failed",
                "error": repr(error),
                "engineering": args.engineering,
                "wall_seconds": time.monotonic() - begin,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True, help="Completed audit receipt.json")
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true", help="Explicit synthetic-development mode only")
    print(json.dumps(render(parser.parse_args())))


if __name__ == "__main__":
    main()
