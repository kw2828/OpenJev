"""Publish descriptive plots and complete tables from the authenticated audit.

Reads saved audit outputs only. No execution replay, learned inference, model
selection, resampling or new experiment occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = ("full", "ordinary", "shift")
PLANNERS = ("rs64", "rs256", "cem256")
SEEDS = (271, 283, 293)
KINDS = (("free", "Free reward", "#596e86"), ("residual", "Known-cost residual", "#137f87"))
MARKERS = ("o", "s", "^")
PANEL_LABELS = ("Full sensing", "Six-step sensor gaps", "Ten-step sensor gaps")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FIT_ORDER = ("free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293")
REFERENCES = {"known_state", "particle", "zero", "uniform"}
LEARNED = {f"{name}-{method}" for name in FIT_ORDER for method in PLANNERS}
METRIC_FIELDS = (
    "native_selected_return",
    "finite_set_regret",
    "selected_raw_prediction_bias",
    "selected_clipped_prediction_bias",
    "common_bank_raw_return_mse",
    "common_bank_clipped_return_mse",
    "reward_prediction_clipping_fraction",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(np.isfinite(value), "Nonfinite JSON number")


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=pairs)
    finite_tree(value)  # Also catches finite-looking literals such as 1e999.
    return value


def number(value, label, *, minimum=None):
    require(type(value) in (int, float) and np.isfinite(value), f"Invalid numeric {label}")
    require(minimum is None or value >= minimum, f"Invalid range {label}")
    return value


def equal(actual, expected, label):
    number(actual, label)
    require(np.isclose(actual, expected, rtol=1e-12, atol=1e-8), f"Inconsistent {label}")


def vector(value, length, label, *, minimum=None):
    require(isinstance(value, list) and len(value) == length, f"Incomplete {label}")
    return np.array([number(v, label, minimum=minimum) for v in value], dtype=float)


def gate_checks(gate, expected):
    require(
        isinstance(gate, dict) and type(gate.get("passed")) is bool and isinstance(gate.get("checks"), list),
        "Invalid gate schema",
    )
    checks = gate["checks"]
    require(
        len(checks) == len(expected) and {row["name"] for row in checks} == set(expected),
        "Incomplete gate checks",
    )
    for row in checks:
        actual, threshold, strict = expected[row["name"]]
        equal(row["actual"], actual, "gate actual")
        equal(row["threshold"], threshold, "gate threshold")
        require(
            row["direction"] == ("lt" if strict else "le")
            and type(row["passed"]) is bool
            and row["passed"] == bool(actual < threshold if strict else actual <= threshold),
            "Inconsistent gate result",
        )
    require(gate["passed"] == all(row["passed"] for row in checks), "Inconsistent overall gate")


def validate_control(summary):
    controls = summary["control"]
    require(set(controls) == set(PANELS), "Incomplete control panels")
    for panel in PANELS:
        require(set(controls[panel]) == LEARNED | REFERENCES, "Expected all 66 control rows")
        for label, row in controls[panel].items():
            costs = vector(row["episode_costs"], 64, "paired episode costs", minimum=0)
            equal(row["mean_cost"], costs.mean(), "mean episode cost")
            decisions = vector(row["decision_seconds"], 50, "decision timings", minimum=0)
            require(np.all(decisions > 0), "Nonpositive decision timing")
            equal(row["decision_wall_seconds"], decisions.sum(), "decision timing sum")
            equal(row["per_case_amortized_seconds"], decisions.mean() / 64, "amortized timing denominator")
            latency = row["batch_latency_seconds"]
            require(set(latency) == {"mean", "p50", "p95", "max"}, "Latency summary schema")
            for key, value in {
                "mean": decisions.mean(),
                "p50": np.quantile(decisions, 0.5),
                "p95": np.quantile(decisions, 0.95),
                "max": decisions.max(),
            }.items():
                equal(latency[key], value, "latency statistic")
            for key in ("setup_seconds", "native_step_seconds", "row_wall_seconds"):
                require(number(row[key], key) > 0, "Nonpositive row timing")
            require(
                row["setup_seconds"] + row["decision_wall_seconds"] + row["native_step_seconds"]
                <= row["row_wall_seconds"] + 1e-6,
                "Nested timings exceed row",
            )
            learned = label in LEARNED
            planned = learned or label in ("known_state", "particle")
            require(
                type(row["planner_used"]) is bool and row["planner_used"] == planned, "Planner scope mismatch"
            )
            count = (64 if label.endswith("rs64") or label in REFERENCES else 256) if planned else 0
            for key, value in {
                "candidate_evaluations": 64 * 50 * count,
                "imagined_transitions": 64 * count * sum(min(12, 50 - t) for t in range(50)),
            }.items():
                require(type(row[key]) is int and row[key] == value, "Candidate/imagined coverage")
            traces = vector(row["search_seconds"], 50 if learned else 0, "search timings", minimum=0)
            if learned:
                require(np.all(traces > 0) and np.all(traces <= decisions + 1e-6), "Uncharged search timing")
                require(
                    0 <= number(row["clipping_fraction"], "clipping fraction") <= 1, "Clipping fraction range"
                )
            else:
                require(row["clipping_fraction"] is None, "Reference is not a learned prediction")
    expected, competence = {}, {method: {} for method in PLANNERS}

    def cost(panel, kind, seed, method):
        return controls[panel][f"{kind}-{seed}-{method}"]["mean_cost"]

    def mean(panel, kind, method):
        cases = [controls[panel][f"{kind}-{seed}-{method}"]["episode_costs"] for seed in SEEDS]
        return np.asarray(cases, dtype=float).mean(0).mean()

    for panel in PANELS[1:]:
        expected[f"residual/{panel}/cem_mean_vs_rs256"] = (
            mean(panel, "residual", "cem256"),
            0.95 * mean(panel, "residual", "rs256"),
            False,
        )
        for seed in SEEDS:
            expected[f"residual-{seed}/{panel}/cem_vs_rs256"] = (
                cost(panel, "residual", seed, "cem256"),
                cost(panel, "residual", seed, "rs256"),
                True,
            )
    gate_checks(summary["continuation_gate"], expected)
    require(set(summary["control_competence"]) == set(PLANNERS), "Incomplete competence planners")
    for method, checks in competence.items():
        checks["physics_ordinary_vs_zero"] = (
            controls["ordinary"]["known_state"]["mean_cost"],
            0.9 * controls["ordinary"]["zero"]["mean_cost"],
            False,
        )
        for panel in PANELS[1:]:
            for seed in SEEDS:
                residual = cost(panel, "residual", seed, method)
                checks[f"residual-{seed}/{panel}/vs_zero"] = (
                    residual,
                    0.9 * controls[panel]["zero"]["mean_cost"],
                    False,
                )
                checks[f"residual-{seed}/{panel}/vs_paired_free"] = (
                    residual,
                    cost(panel, "free", seed, method),
                    True,
                )
            checks[f"residual/{panel}/mean_vs_free"] = (
                mean(panel, "residual", method),
                0.95 * mean(panel, "free", method),
                False,
            )
        gate_checks(summary["control_competence"][method], checks)


def validate_diagnostics(summary):
    diagnostics = summary["diagnostics"]
    roots = diagnostics["roots"]
    require(
        isinstance(roots, list)
        and len(roots) == 64
        and {(r["episode_index"], r["root_ordinal"]) for r in roots}
        == {(e, r) for e in range(16) for r in range(4)},
        "Expected 64 unique physical roots",
    )
    order = [
        {"panel": panel, "fit": fit, "planner": method, "union_index": 64 + i}
        for i, (panel, fit, method) in enumerate(
            (p, f, m) for p in PANELS for f in FIT_ORDER for m in PLANNERS
        )
    ]
    ids = [f"common/{i}" for i in range(64)] + [
        f"selected/{r['panel']}/{r['fit']}/{r['planner']}" for r in order
    ]
    for root in roots:
        for key in (
            "episode_index",
            "root_ordinal",
            "phase",
            "step",
            "horizon",
            "identity_slots",
            "unique_sequence_count",
        ):
            require(type(root[key]) is int, "Root integer identity")
        phase = root["phase"]
        require(
            0 <= phase <= 3
            and root["step"] == (6, 10 + phase, 14 + phase, 47)[root["root_ordinal"]]
            and root["horizon"] == min(12, 50 - root["step"]),
            "Root phase/terminal denominator",
        )
        require(
            root["identity_slots"] == 118
            and 1 <= root["unique_sequence_count"] <= 118
            and root["duplicates_retained"] is True
            and root["union_ids"] == ids
            and root["search_order"] == order,
            "Incomplete 118-slot diagnostic union",
        )
        require(
            set(root["history_members"]) == {f"history-{p}.npz" for p in PANELS}
            and all(digest(v) for v in root["history_members"].values()),
            "Diagnostic history binding",
        )
        require(set(root["metrics"]) == set(PANELS), "Diagnostic panel coverage")
        for panel in PANELS:
            require(set(root["metrics"][panel]) == LEARNED, "Expected every diagnostic arm at every root")
            for row in root["metrics"][panel].values():
                returns = vector(row["native_selected_branch_returns"], 4, "native branch returns")
                equal(row["native_selected_return"], returns.mean(), "native branch mean")
                require(row["evaluated_union_slots"] == 118, "Diagnostic finite-set denominator")
                correlation = row["common_bank_spearman"]
                require(correlation is None or -1 <= number(correlation, "Spearman") <= 1, "Spearman range")
                for key in METRIC_FIELDS:
                    number(row[key], key)
                require(
                    row["finite_set_regret"] >= 0
                    and row["common_bank_raw_return_mse"] >= 0
                    and row["common_bank_clipped_return_mse"] >= 0
                    and 0 <= row["reward_prediction_clipping_fraction"] <= 1,
                    "Diagnostic metric range",
                )
        replay, timing = root["native_replay"], root["timing"]
        transitions = 118 * 4 * root["horizon"]
        require(
            replay["transitions"] == transitions
            and replay["saved_output_only"] is True
            and type(replay["new_policy_calls"]) is int
            and replay["new_policy_calls"] == 0,
            "Diagnostic replay scope",
        )
        require(0 <= number(replay["max_abs_error"], "native error") <= 1e-10, "Native replay error")
        require(
            timing["identity_slots"] == 118
            and timing["native_transitions"] == transitions
            and timing["observation_assimilations"] == 18 * (root["step"] + 1)
            and timing["history_action_advances"] == 18 * root["step"],
            "Diagnostic model-step counts",
        )
        for key in ("belief_seconds", "search_seconds", "native_and_storage_seconds", "row_wall_seconds"):
            require(number(timing[key], key) > 0, "Nonpositive diagnostic timing")
        require(
            sum(timing[k] for k in ("belief_seconds", "search_seconds", "native_and_storage_seconds"))
            <= timing["row_wall_seconds"] + 1e-6,
            "Diagnostic nested timing",
        )
    aggregate = diagnostics["aggregate"]
    require(set(aggregate) == set(PANELS), "Diagnostic aggregate panels")
    for panel in PANELS:
        require(set(aggregate[panel]) == LEARNED, "Expected all 54 diagnostic aggregate rows")
        for name, row in aggregate[panel].items():
            raw = [root["metrics"][panel][name] for root in roots]
            ranks = [r["common_bank_spearman"] for r in raw if r["common_bank_spearman"] is not None]
            require(
                row["rank_defined_roots"] == len(ranks) and row["rank_undefined_roots"] == 64 - len(ranks),
                "Rank coverage",
            )
            if ranks:
                equal(row["common_bank_spearman_mean"], np.mean(ranks), "rank mean excluding undefined roots")
            else:
                require(row["common_bank_spearman_mean"] is None, "Undefined ranks must remain undefined")
            for key in METRIC_FIELDS:
                equal(row[key + "_mean"], np.mean([r[key] for r in raw]), "diagnostic root mean")


def validate_costs(summary):
    costs = summary["costs"]
    control = [row for panel in summary["control"].values() for row in panel.values()]
    roots = summary["diagnostics"]["roots"]
    for key, expected in {
        "control_row_wall_seconds": sum(r["row_wall_seconds"] for r in control),
        "control_setup_seconds": sum(r["setup_seconds"] for r in control),
        "control_decision_seconds": sum(r["decision_wall_seconds"] for r in control),
        "control_native_step_seconds": sum(r["native_step_seconds"] for r in control),
        "diagnostic_root_wall_seconds": sum(r["timing"]["row_wall_seconds"] for r in roots),
        "new_evaluation_wall_seconds": summary["wall_seconds"],
    }.items():
        equal(costs[key], expected, key)
    require(0 < number(summary["wall_seconds"], "wall time") <= 3600, "Execution cap")
    for key in (
        "inherited_setup_seconds",
        "innovation_generation_and_storage_seconds",
        "diagnostic_collection_seconds",
        "inherited_fit_wall_seconds",
        "prior_cumulative_attempt_wall_seconds",
        "audit_validation_wall_seconds",
    ):
        require(number(costs[key], key) > 0, "Nonpositive cost")
    require(
        sum(
            costs[key]
            for key in (
                "inherited_setup_seconds",
                "innovation_generation_and_storage_seconds",
                "control_row_wall_seconds",
                "diagnostic_collection_seconds",
                "diagnostic_root_wall_seconds",
            )
        )
        <= summary["wall_seconds"] + 1e-6,
        "Costs exceed whole-run time",
    )
    equal(
        costs["cumulative_attempt_wall_seconds"],
        costs["prior_cumulative_attempt_wall_seconds"] + summary["wall_seconds"],
        "cumulative cost",
    )
    equal(
        costs["fresh_evaluation_plus_inherited_fits_seconds"],
        costs["inherited_fit_wall_seconds"] + summary["wall_seconds"],
        "inherited fit cost split",
    )
    equal(
        costs["inherited_fit_wall_seconds"],
        sum(r["wall_seconds"] for r in summary["fits"].values()),
        "summed inherited fit time",
    )
    require(costs["prior_costs"] == summary["source_lineage"]["prior_costs"], "Prior cost provenance")
    equal(
        costs["inherited_fit_wall_seconds"],
        costs["prior_costs"]["inherited_fit_wall_seconds"],
        "inherited cost binding",
    )
    equal(
        costs["prior_cumulative_attempt_wall_seconds"],
        costs["prior_costs"]["cumulative_attempt_wall_seconds"],
        "prior cumulative binding",
    )


def validate_summary(summary, receipt):
    require(
        summary.get("status") == "completed"
        and summary.get("version") == "reacher-search-v1"
        and summary.get("saved_output_only") is True,
        "Unexpected summary scope",
    )
    for key in ("plan_sha256", "execution_completed_sha256"):
        require(
            digest(summary.get(key)) and summary[key] == receipt.get(key),
            "Receipt/summary provenance mismatch",
        )
    require(
        summary["source_lineage"] == receipt["parent_source"] and summary["costs"] == receipt["costs"],
        "Receipt/summary lineage or cost mismatch",
    )
    for key in ("new_model_calls", "new_policy_calls", "new_mpc_calls", "new_fits"):
        require(type(summary.get(key)) is int and summary[key] == 0, "Audit inference scope must be zero")
    require(set(summary["fits"]) == set(FIT_ORDER), "Expected exactly six unchanged fits")
    require(
        summary["native_transition_counts"]
        == {
            "control": 211200,
            "diagnostic_cohort": 800,
            "diagnostic_branches": 294528,
            "inherited_training": 38400,
            "fresh_evaluation": 506528,
        }
        and summary["native_transitions_checked"] == 544928
        and summary["learned_control_imagined_transitions"] == 354336768,
        "Incomplete transition coverage",
    )
    require(
        0 <= number(summary["native_max_abs_error"], "native replay error") <= 1e-10, "Native replay mismatch"
    )
    historical = summary["historical_parent_qualification"]
    require(
        historical["rerun"] is False
        and historical["plan_sha256"] == receipt["parent_source"]["plan_sha256"]
        and historical["audit_receipt_sha256"] == receipt["parent_source"]["audit_receipt_sha256"],
        "Historical qualification provenance",
    )
    validate_control(summary)
    validate_diagnostics(summary)
    validate_costs(summary)


def authenticate(audit, expected):
    require(
        digest(expected)
        and not audit.is_symlink()
        and not (audit / "receipt.json").is_symlink()
        and sha(audit / "receipt.json") == expected,
        "External audit receipt identity mismatch",
    )
    receipt = read(audit / "receipt.json")
    require(
        receipt.get("status") == "completed"
        and receipt.get("version") == "reacher-search-v1"
        and receipt.get("saved_output_only") is True,
        "A completed saved-output search audit is required",
    )
    require(set(receipt["files"]) == {"summary.json", "README.md"}, "Exact audit member coverage required")
    for name, expected_digest in receipt["files"].items():
        path = PurePosixPath(name)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and not (audit / name).is_symlink()
            and digest(expected_digest)
            and sha(audit / name) == expected_digest,
            "Audit member identity mismatch",
        )
    require(
        digest(receipt.get("plan_canonical_sha256"))
        and isinstance(receipt.get("source_sha256"), dict)
        and bool(receipt["source_sha256"])
        and all(digest(v) for v in receipt["source_sha256"].values())
        and isinstance(receipt.get("runtime"), dict)
        and bool(receipt["runtime"])
        and isinstance(receipt.get("execution_members"), dict)
        and bool(receipt["execution_members"])
        and all(digest(v) for v in receipt["execution_members"].values()),
        "Missing completed audit provenance",
    )
    summary = read(audit / "summary.json")
    try:
        validate_summary(summary, receipt)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Incomplete or malformed audit summary") from error
    return summary


def rows(summary, panel, kind, planner):
    return [summary["control"][panel][f"{kind}-{seed}-{planner}"] for seed in SEEDS]


def write_csv(path, records):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save(fig, path):
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def render(audit, expected, out):
    summary = authenticate(audit, expected)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#20303b",
        }
    )
    control_table, diagnostic_table = [], []
    for panel in PANELS:
        for name, row in summary["control"][panel].items():
            control_table.append(
                {
                    "panel": panel,
                    "controller": name,
                    "mean_episode_cost": row["mean_cost"],
                    "setup_seconds": row["setup_seconds"],
                    "decision_wall_seconds": row["decision_wall_seconds"],
                    "native_step_seconds": row["native_step_seconds"],
                    "row_wall_seconds": row["row_wall_seconds"],
                }
            )
        for name, row in summary["diagnostics"]["aggregate"][panel].items():
            diagnostic_table.append({"panel": panel, "controller": name, **row})
    write_csv(out / "all-control-rows.csv", control_table)
    write_csv(out / "all-diagnostic-rows.csv", diagnostic_table)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
    labels = [f"{label} / {planner.upper()}" for _, label, _ in KINDS for planner in PLANNERS]
    for ax, panel, title in zip(axes, PANELS, PANEL_LABELS, strict=True):
        for group, (kind, _, color) in enumerate(KINDS):
            for p, planner in enumerate(PLANNERS):
                y = group * 3 + p
                values = [row["mean_cost"] for row in rows(summary, panel, kind, planner)]
                ax.plot([min(values), max(values)], [y, y], color=color, alpha=0.55)
                ax.scatter(
                    values, y + np.array([-0.09, 0, 0.09]), color=color, marker=MARKERS[p], s=40, zorder=3
                )
        for name, color, label in (
            ("zero", "#a55a4c", "Zero command"),
            ("known_state", "#c58b2a", "Known-state physics"),
        ):
            ax.axvline(
                summary["control"][panel][name]["mean_cost"],
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=label,
            )
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("Episode cost (lower is better)")
        ax.set_yticks(range(6), labels)
        ax.grid(axis="x", alpha=0.15)
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.98, 0.91), ncol=2, fontsize=8)
    gate = "PASS" if summary["continuation_gate"]["passed"] else "FAIL"
    fig.suptitle(
        "Does adaptive search improve actual control?",
        x=0.02,
        ha="left",
        y=0.98,
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.915,
        f"Frozen CEM256 versus RS256 continuation rule: {gate}. All six inherited fits are shown.",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.035,
        "Each dot is one fit averaged over 64 paired episodes; lines span fits, not confidence intervals.\n"
        "RS64 scores 64 sequences per decision. RS256 and CEM256 each score 256. Physics uses supplied dynamics and 64 sequences.",
        fontsize=10,
        linespacing=1.6,
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.82, bottom=0.19, wspace=0.20)
    save(fig, out / "search-control.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)
    for ax, panel, title in zip(axes, PANELS[1:], PANEL_LABELS[1:], strict=True):
        for kind, label, color in KINDS:
            for p, planner in enumerate(PLANNERS):
                selected = rows(summary, panel, kind, planner)
                # Includes proposal work, score calls and trace storage. This is
                # amortized CPU cost across 64 simultaneous cases, not latency.
                x = [1000 * row["decision_wall_seconds"] / (64 * 50) for row in selected]
                y = [row["mean_cost"] for row in selected]
                ax.scatter(x, y, color=color, marker=MARKERS[p], s=45, label=f"{label} / {planner.upper()}")
        ax.axhline(
            summary["control"][panel]["zero"]["mean_cost"],
            color="#a55a4c",
            linestyle="--",
            linewidth=1.2,
            label="Zero command",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Decision time per case (ms, amortized)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Episode cost (lower is better)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, fontsize=8)
    fig.suptitle("Measured control versus computation", x=0.025, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.025,
        0.035,
        "All seeds; descriptive timings on a shared CPU host. Decision time includes search and trace storage.\n"
        "Environment stepping, inherited training, shared setup and separate diagnostic costs are reported separately.",
        fontsize=10,
        linespacing=1.6,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.80))
    save(fig, out / "search-compute.png")

    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.5))
    fields = (
        ("native_selected_return_mean", "Native selected return\n(higher is better)"),
        ("finite_set_regret_mean", "Regret in evaluated finite set\n(lower is better)"),
        (
            "selected_raw_prediction_bias_mean",
            "Raw predicted minus native return\n(positive means optimistic)",
        ),
    )
    for r, panel in enumerate(PANELS[1:]):
        for c, (field, title) in enumerate(fields):
            ax = axes[r, c]
            for kind, label, color in KINDS:
                for p, planner in enumerate(PLANNERS):
                    values = [
                        summary["diagnostics"]["aggregate"][panel][f"{kind}-{seed}-{planner}"][field]
                        for seed in SEEDS
                    ]
                    shift = -0.10 if kind == "free" else 0.10
                    ax.scatter(
                        np.full(3, p + shift) + [-0.025, 0, 0.025],
                        values,
                        color=color,
                        marker=MARKERS[p],
                        s=32,
                        label=label if p == 0 else None,
                    )
                    ax.plot(
                        [p + shift - 0.06, p + shift + 0.06], [np.mean(values)] * 2, color=color, linewidth=2
                    )
            ax.set_xticks(range(3), [p.upper() for p in PLANNERS])
            ax.set_title(title, fontsize=11)
            ax.grid(axis="y", alpha=0.15)
            if c == 0:
                ax.set_ylabel(PANEL_LABELS[r + 1])
            if c == 2:
                ax.axhline(0, color="#66757d", linewidth=0.8)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Check model optimism at the same physical roots", x=0.025, ha="left", fontsize=18, fontweight="bold"
    )
    fig.text(
        0.025,
        0.025,
        "Each dot: one fit averaged over 64 fixed roots from 16 independent physical episodes, with four shared noise branches.\n"
        "Native outcomes do not select actions. Regret compares 118 evaluated identity slots, not an optimal controller. No new training.",
        fontsize=10,
        linespacing=1.6,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.925), h_pad=2.2)
    save(fig, out / "search-model-error.png")
    receipt = {
        "status": "completed",
        "version": "reacher-search-v1-figures",
        "audit_receipt_sha256": expected,
        "source_sha256": sha(Path(__file__)),
        "saved_output_only": True,
        "new_model_calls": 0,
        "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "limits": "Descriptive plots over the six inherited fits. No fit selection, additional model calls or new bootstrap samples.",
    }
    with (out / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.audit, args.audit_sha256, args.out), indent=2))
