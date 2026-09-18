# SPDX-License-Identifier: GPL-3.0-only
"""Render completed chess pin-quality-v3 evidence without neural/engine calls.

Authenticate the externally supplied audit receipt SHA, its plan, the complete
primary manifest, and all six native replay files before reading metrics.
No runner, chess, Torch, checkpoint loader, model or engine is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

VERSION = "canonical-matched-pin-quality-v3"
CORE = ("wldn", "joint", "separable", "pairwise", "root_only", "counts", "graph_mlp")
UNION = ("union:child", "union:union", "union:edits", "union:rotated")
SEEDS = (97, 109, 127)
SPLITS = ("dev", "shift")
PANELS = {"dev": "Ordinary development panel", "shift": "Shifted development panel"}
LABELS = {"base": "Frozen backbone", "wldn": "WLDN", "joint": "Joint pin", "separable": "Separable",
          "pairwise": "Pairwise", "root_only": "Root only", "counts": "Counts", "graph_mlp": "Graph MLP",
          **{f"union:{arm}": f"Union {arm}" for arm in ("child", "union", "edits", "rotated")}}
MARKERS = ("o", "s", "^")
FIGURES = ("agreement", "target-nll", "joint-comparisons", "native-cached-parity", "costs")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0), name)
    return float(value)


def checked(path, digest):
    path = Path(path)
    require(valid_hash(digest) and path.is_file() and not path.is_symlink(), "Regular hash-bound file required")
    require(sha(path) == digest, f"Artifact SHA-256 mismatch: {path.name}")
    return path


def member(folder, name):
    require(isinstance(name, str), "Manifest member name")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name and "\\" not in name
            and all(part not in (".", "..") for part in relative.parts), "Unsafe manifest path")
    path = Path(folder) / name
    require(path.resolve().is_relative_to(Path(folder).resolve()) and not path.is_symlink(), "Contained regular member")
    for parent in path.parents:
        if parent == Path(folder).parent:
            break
        require(not parent.is_symlink(), "Symlinked member parent")
    return path


def fit_name(arm, seed):
    return f"{arm.replace(':', '_')}-{seed}"


def expected_members(protocol):
    result = {"started.json", "training-targets.json", "all-training-complete.json", "evaluation-started.json", "summary.json"}
    for seed in SEEDS:
        for arm in protocol["arms"]:
            result.update(f"fits/{fit_name(arm, seed)}/{name}"
                          for name in ("initial.pt", "weights.pt", "learning.jsonl", "training.json"))
        for split in SPLITS:
            result.add(f"native/{seed}-{split}.jsonl")
            result.update(f"predictions/{fit_name(arm, seed)}-{split}.jsonl" for arm in protocol["methods"])
    return result


def validate_protocol(protocol):
    arms = tuple(protocol["arms"])
    require(protocol["version"] == VERSION and len(arms) == 8 and arms[:7] == CORE and arms[7] in UNION,
            "Exact v3 eight-arm scope required")
    require(protocol["methods"] == ["base", *arms] and protocol["seeds"] == list(SEEDS), "All frozen backbone seeds/methods")
    expected = {"fits": 24, "training_updates": 36864, "updates_per_fit": 1536, "training_roots": 32768,
                "batch_size": 128, "evaluation_roots_per_panel": 2048, "prediction_records": 110592,
                "candidate_scores": 3226149, "native_root_backbone_reconstructions": 12288, "quality_checks": 16}
    for name, value in expected.items():
        require(type(protocol[name]) is int and protocol[name] == value, f"Frozen full coverage: {name}")
    require(protocol["score_tolerance"] == 1e-5, "Frozen numerical tolerance")


def authenticate(plan_path, audit, execution, expected_receipt_sha256):
    audit, execution = Path(audit), Path(execution)
    require(audit.is_dir() and execution.is_dir() and not audit.is_symlink() and not execution.is_symlink(),
            "Regular completed artifact directories")
    require(not (audit / "failed.json").exists() and not (execution / "failed.json").exists(), "Conflicting failure receipt")
    receipt = read(checked(audit / "receipt.json", expected_receipt_sha256))
    require(receipt["status"] == "completed", "Completed audit required")
    plan = read(checked(plan_path, receipt["plan_sha256"]))
    protocol = plan["protocol"]
    validate_protocol(protocol)
    completed = read(checked(execution / "completed.json", receipt["execution_receipt_sha256"]))
    require(completed["status"] == "completed" and completed["plan_sha256"] == receipt["plan_sha256"], "Execution audit binding")
    expected = expected_members(protocol)
    require(set(completed["files"]) == expected, "Exact complete primary manifest")
    found = {str(path.relative_to(execution)) for path in execution.rglob("*") if path.is_file()}
    require(found == expected | {"completed.json"}, "Missing, partial or extra execution members")
    for name, digest in completed["files"].items():
        checked(member(execution, name), digest)
    native = {f"native/{seed}-{split}.jsonl" for seed in SEEDS for split in SPLITS}
    require(set(receipt["replay_native_files_sha256"]) == native, "All six complete native replay files")
    for name, digest in receipt["replay_native_files_sha256"].items():
        checked(member(audit, name), digest)
        require(digest == completed["files"][name], "Audit replay must reproduce primary native bytes")
    summary = read(checked(execution / "summary.json", receipt["summary_sha256"]))
    require(receipt["summary_sha256"] == completed["files"]["summary.json"], "Summary manifest binding")
    fits = {f"{arm}-{seed}": read(execution / f"fits/{fit_name(arm, seed)}/training.json")
            for arm in protocol["arms"] for seed in SEEDS}
    return plan, receipt, summary, fits, completed


def quality_checks(metrics, arms):
    checks = []
    for split in SPLITS:
        for arm in ("base", *[value for value in arms if value != "joint"]):
            gains = [metrics[f"joint-{seed}"][split]["agreement"] - metrics[f"{arm}-{seed}"][split]["agreement"]
                     for seed in SEEDS]
            mean, minimum = statistics.mean(gains), min(gains)
            required = 0. if arm == "base" else .01
            checks.append({"split": split, "comparator": arm, "paired_seed_gains": gains,
                           "mean_gain": mean, "minimum_paired_seed_gain": minimum,
                           "required_mean_gain": required, "required_seed_floor": -.005,
                           "passed": mean >= required and minimum >= -.005})
    return checks


def figure_data(plan, receipt, summary, fits, completed, receipt_sha):
    p = plan["protocol"]
    require(summary["status"] == "completed" and summary["plan_sha256"] == receipt["plan_sha256"], "Completed summary identity")
    for key, expected in (("fits", 24), ("training_updates", 36864), ("prediction_records", 110592)):
        require(type(summary[key]) is int and summary[key] == expected, f"Full summary scope: {key}")
    require(all(type(summary[key]) is int and summary[key] == 0
                for key in ("new_engine_calls", "external_model_calls", "copied_reference_predictions")), "Execution scope")
    require(summary["original_failed_criteria_unchanged"] is True and summary["limits"] == p["limits"], "Historical limits/criteria")
    scope = {"training_updates_checked": 36864, "fresh_initial_states_exact": 24, "final_checkpoint_identities_checked": 24,
             "prediction_records_replayed": 110592, "candidate_scores_replayed": 3226149,
             "native_root_backbone_reconstructions": 12288, "bootstrap_intervals_recomputed": 16}
    require(all(type(receipt[key]) is int and receipt[key] == value for key, value in scope.items()), "Complete numerical audit scope")
    require(receipt["cached_scores_and_nll_exact"] is True and receipt["native_vectors_exact"] is True,
            "Audit must exactly reproduce saved kernels; this does not assert native/cached parity")
    metrics = summary["metrics"]
    require(set(metrics) == {f"{arm}-{seed}" for arm in p["methods"] for seed in SEEDS}, "All 24 fits plus three backbone baselines")
    for rows in metrics.values():
        require(set(rows) == set(SPLITS), "Complete ordinary and shifted panels")
        for row in rows.values():
            require(row["examples"] == 2048 and 0 <= finite(row["agreement"], "Agreement") <= 1
                    and finite(row["target_nll"], "Target NLL") >= 0, "Complete finite panel metrics")
    gate = quality_checks(metrics, p["arms"])
    require(gate == summary["gate_checks"] == receipt["gate_recomputed"] and len(gate) == 16,
            "All 16 designated-joint gate checks must reproduce")
    require(summary["quality_gate_passed"] is all(row["passed"] for row in gate), "Quality gate state")
    intervals = summary["conditional_game_bootstrap"]
    require([(row["split"], row["comparator"]) for row in intervals]
            == [(row["split"], row["comparator"]) for row in gate], "All 16 audited intervals in frozen order")
    for row, check in zip(intervals, gate, strict=True):
        bounds = row["percentile95"]
        require(row["roots"] == 2048 and row["draws"] == 2000 and 0 < row["games"] <= 2048
                and row["seed"] == 996101 + (row["split"] == "shift") and len(bounds) == 2
                and all(math.isfinite(value) for value in bounds) and -1 <= bounds[0] <= bounds[1] <= 1
                and math.isclose(row["point_gain"], check["mean_gain"], abs_tol=1e-12), "Audited interval identities")
    numerical = receipt["numerical_gate_recomputed"]
    require(all(summary[key] == value for key, value in numerical.items()), "Native numerical audit binding")
    require(numerical["native_root_backbone_reconstructions"] == 12288
            and numerical["native_method_comparisons"] == 110592 and numerical["candidate_score_comparisons"] == 3226149
            and set(numerical["native_by_method"]) == set(p["methods"]), "Complete native comparison counts")
    for row in numerical["native_by_method"].values():
        require(finite(row["max_score_error"], "Native score error") >= 0
                and type(row["failed_cases"]) is int and type(row["choice_changes"]) is int
                and 0 <= row["choice_changes"] <= row["failed_cases"] <= 12288, "Finite native failures retained")
        require(row["failed_cases"] > 0 or row["max_score_error"] <= p["score_tolerance"], "Parity failure count")
    failures = sum(row["failed_cases"] for row in numerical["native_by_method"].values())
    require(numerical["failed_native_method_cases"] == failures
            and numerical["native_choice_changes"] == sum(row["choice_changes"] for row in numerical["native_by_method"].values())
            and numerical["max_native_cached_score_error"] == max(row["max_score_error"] for row in numerical["native_by_method"].values())
            and numerical["numerical_gate_passed"] is (failures == 0), "Native parity aggregation/state")
    continuation = summary["quality_gate_passed"] and numerical["numerical_gate_passed"]
    require(summary["continuation_passed"] is continuation and receipt["continuation_passed"] is continuation,
            "Failed quality/numerical gate cannot become a passed continuation")
    require(set(fits) == {f"{arm}-{seed}" for arm in p["arms"] for seed in SEEDS}, "All training cost receipts")
    for name, row in fits.items():
        require(row["status"] == "completed" and name == f"{row['arm']}-{row['seed']}"
                and row["plan_sha256"] == receipt["plan_sha256"] and row["updates"] == 1536
                and row["examples_seen"] == 1536 * 128 and row["parameters"] == p["parameters"][row["arm"]],
                "Matched fit training receipts")
        finite(row["seconds"], "Per-fit wall time", positive=True)
        for key, file in (("initial_sha256", "initial.pt"), ("weights_sha256", "weights.pt"), ("learning_sha256", "learning.jsonl")):
            require(row[key] == completed["files"][f"fits/{fit_name(row['arm'], row['seed'])}/{file}"], "Per-fit member identities")
    primary = finite(summary["wall_seconds"], "Execution wall", positive=True)
    audit_wall = finite(receipt["wall_seconds"], "Audit wall", positive=True)
    require(primary <= p["time_cap_seconds"] and audit_wall <= p["audit_time_cap_seconds"]
            and sum(row["seconds"] for row in fits.values()) <= primary + 1e-6, "Phase cost coverage and frozen caps")
    recovery = plan["recovery"]
    prior_updates = finite(recovery["prior_update_seconds_retained"], "Prior logged update lower bound")
    prior_fits = finite(recovery["prior_fit_seconds_overlapping"], "Prior overlapping completed fits")
    require(prior_updates >= 0 and prior_fits >= 0 and recovery["prior_termination_cause_known"] is False,
            "Interrupted prior attempt must remain incomplete")
    family = {split: {arm: {key: statistics.mean(metrics[f"{arm}-{seed}"][split][key] for seed in SEEDS)
                           for key in ("agreement", "target_nll")} for arm in p["methods"]} for split in SPLITS}
    return {"version": VERSION, "engineering": plan.get("publication_fixture") is True,
            "provenance": {"audit_receipt_sha256": receipt_sha, "plan_sha256": receipt["plan_sha256"],
                           "execution_receipt_sha256": receipt["execution_receipt_sha256"],
                           "summary_sha256": receipt["summary_sha256"], "auditor_sha256": receipt["auditor_sha256"],
                           "execution_members": completed["files"], "native_replay_members": receipt["replay_native_files_sha256"]},
            "methods": p["methods"], "arms": p["arms"], "seeds": list(SEEDS), "metrics": metrics,
            "family_means": family, "gate_checks": gate, "quality_gate_passed": summary["quality_gate_passed"],
            "numerical": numerical, "continuation_passed": continuation, "conditional_game_bootstrap": intervals,
            "fits": fits, "coverage": scope,
            "costs": {"execution_wall_seconds": primary, "audit_wall_seconds": audit_wall,
                      "fit_wall_seconds_nested": sum(row["seconds"] for row in fits.values()),
                      "v2_logged_update_seconds_lower_bound": prior_updates,
                      "v2_completed_fit_seconds_overlapping": prior_fits,
                      "v2_termination_cause_known": False,
                      "v1_and_other_historical_costs": "Not quantified by these inputs; do not present v3 as total project cost."},
            "limits": p["limits"], "prior_knowledge": p["prior_knowledge"], "audit_scope": receipt["scope"],
            "notes": ["Previously exposed development panels, not independent confirmation. No Elo or gameplay conclusion.",
                      "The designated joint arm must beat every prespecified comparator. A stronger control cannot replace it after outcomes.",
                      "Native/cached failures are retained even when audit replay exactly reproduces all saved results.",
                      "Intervals are the authenticated, conditional source-game bootstrap results. No new bootstrap or seed-population inference is performed.",
                      "The chess audit reran existing neural kernels on saved inputs; this renderer performs zero model or engine calls.",
                      "Per-fit work is nested in execution wall time. Prior logged updates and completed-fit times overlap and must not be added.",
                      "Shared-host wall measurements are not isolated latency or throughput estimates; this quality study establishes no speed claim."]}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#BAC4CC",
        "axes.labelcolor": "#233340", "text.color": "#233340", "xtick.color": "#425360", "ytick.color": "#425360",
        "grid.color": "#E0E6EB", "grid.linewidth": .6, "pdf.fonttype": 42, "svg.hashsalt": "openjev-pin-quality-v3"})


def finish(fig, data, title, note, out, name, *, legend=True):
    fig.suptitle(title, x=.055, y=.975, ha="left", fontsize=15, fontweight="bold")
    if legend:
        handles = [Line2D([], [], marker=marker, color="#637786", linestyle="none", label=f"Backbone seed {seed}")
                   for marker, seed in zip(MARKERS, SEEDS, strict=True)]
        handles.append(Line2D([], [], marker="D", color="#233340", linestyle="none", label="Three-seed mean"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .922), ncol=4, frameon=False, fontsize=9)
    fig.text(.055, .105, note, fontsize=8.5, va="top", linespacing=1.4)
    footer = ("SYNTHETIC ENGINEERING FIXTURE: NOT RESEARCH RESULTS" if data["engineering"] else
              "EXPOSED DEVELOPMENT PANELS | No independent confirmation, Elo or gameplay claim")
    fig.text(.055, .018, footer, fontsize=9, color="#A44238", weight="bold")
    fig.subplots_adjust(left=.16, right=.97, top=.83, bottom=.20, wspace=.31)
    for suffix in ("png", "svg", "pdf"):
        metadata = {"Creator": "OpenJev authenticated saved-artifact renderer"}
        if suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(out / f"{name}.{suffix}", dpi=220, metadata=metadata, facecolor="white")
    plt.close(fig)


def chart_axis(ax, methods):
    ax.set_yticks(np.arange(len(methods)), [LABELS[name] for name in methods])
    ax.invert_yaxis()
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="x")


def color(arm):
    return "#277F72" if arm == "joint" else "#89949D" if arm == "base" else "#496E8B"


def render_metric(data, out, key):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.1), sharey=True)
    factor = 100 if key == "agreement" else 1
    for split, ax in zip(SPLITS, axes, strict=True):
        for position, arm in enumerate(data["methods"]):
            for index, seed in enumerate(SEEDS):
                ax.scatter(factor * data["metrics"][f"{arm}-{seed}"][split][key], position + (index - 1) * .14,
                           marker=MARKERS[index], color=color(arm), s=38, edgecolor="white", linewidth=.4)
            ax.scatter(factor * data["family_means"][split][arm][key], position, marker="D", color="#233340", s=24)
        chart_axis(ax, data["methods"])
        ax.set_title(PANELS[split], pad=12)
        ax.set_xlabel("Target move agreement (%)" if key == "agreement" else "Target negative log-likelihood (nats)")
        ax.set_xlim(left=0, right=100 if key == "agreement" else None)
    # Shared axes should be inverted once, not toggled by the second panel.
    axes[0].set_ylim(len(data["methods"]) - .5, -.5)
    finish(fig, data, "All 24 fits and three frozen-backbone baselines" if key == "agreement" else
           "Target likelihood across every fit and baseline",
           "2,048 positions per panel, with all three backbone seeds retained. " +
           ("Higher agreement is better." if key == "agreement" else "Lower target NLL is better.") +
           "\nThe joint arm is the designated treatment; displayed controls are not post-outcome replacements.",
           out, "agreement" if key == "agreement" else "target-nll")


def render_comparisons(data, out):
    methods = [arm for arm in data["methods"] if arm != "joint"]
    intervals = {(row["split"], row["comparator"]): row for row in data["conditional_game_bootstrap"]}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.1), sharey=True)
    for split, ax in zip(SPLITS, axes, strict=True):
        for position, arm in enumerate(methods):
            check = next(row for row in data["gate_checks"] if row["split"] == split and row["comparator"] == arm)
            interval = intervals[split, arm]
            for index, gain in enumerate(check["paired_seed_gains"]):
                ax.scatter(100 * gain, position + (index - 1) * .13, marker=MARKERS[index], color="#657D8C", s=29)
            low, high = [100 * value for value in interval["percentile95"]]
            ax.plot([low, high], [position, position], color="#2D8075" if check["passed"] else "#B35243", lw=1.7)
            ax.scatter(100 * check["mean_gain"], position, marker="D", s=26,
                       color="#2D8075" if check["passed"] else "#B35243", zorder=4)
            ax.scatter(100 * check["required_mean_gain"], position, marker="|", color="#263846", s=100)
        ax.axvline(0, color="#A6B2BB", lw=.7)
        ax.axvline(-.5, color="#AF655C", linestyle=":", lw=.9)
        chart_axis(ax, methods)
        ax.set_title(PANELS[split], pad=12)
        ax.set_xlabel("Joint minus comparator agreement (percentage points)")
    axes[0].set_ylim(len(methods) - .5, -.5)
    finish(fig, data, "The designated joint arm against every comparator: all 16 checks",
           "Mean gain required: 0 pp versus backbone, +1 pp versus trained controls (vertical ticks). Every paired seed must be at least -0.5 pp (dotted).\n"
           "Green/red denotes the complete check passing/failing. Bars are the saved conditional game-bootstrap intervals; they are not the gate.",
           out, "joint-comparisons")


def render_parity(data, out):
    methods = data["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.1), sharey=True)
    for ax, key, title in zip(axes, ("max_score_error", "failed_cases", "choice_changes"),
                             ("Maximum native/cached score error", "Failed method-position cases", "Changed chosen moves"), strict=True):
        values = [data["numerical"]["native_by_method"][arm][key] for arm in methods]
        ax.barh(np.arange(len(methods)), values, height=.5, color=[color(arm) for arm in methods])
        for position, value in enumerate(values):
            label = f"{value:.2g}" if key == "max_score_error" else str(value)
            ax.annotate(label, (value, position), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8)
        chart_axis(ax, methods)
        ax.set_title(title, pad=13, fontsize=10)
        if key == "max_score_error":
            ax.set_xscale("symlog", linthresh=1e-6)
            ax.axvline(1e-5, color="#A44D42", linestyle="--", lw=1)
            ax.set_xlim(left=0, right=max(2e-5, max(values) * 1.4))
            ax.set_xlabel("Absolute score difference")
        else:
            ax.set_xlim(0, max(1, max(values) * 1.2))
            ax.set_xlabel("Count")
    axes[0].set_ylim(len(methods) - .5, -.5)
    finish(fig, data, "Native/cached parity is separate from exact audit replay",
           "All 12,288 native roots and 110,592 method-position comparisons are retained. Dashed: score tolerance 1e-5.\n"
           "An exact replay audit can reproduce a failed parity gate. Finite score mismatches and changed choices are never discarded.",
           out, "native-cached-parity", legend=False)


def render_costs(data, out):
    fig, (fits, phases) = plt.subplots(1, 2, figsize=(12.6, 7.1))
    for position, arm in enumerate(data["arms"]):
        values = [data["fits"][f"{arm}-{seed}"]["seconds"] / 60 for seed in SEEDS]
        for index, value in enumerate(values):
            fits.scatter(value, position + (index - 1) * .14, marker=MARKERS[index], color=color(arm), s=36)
        fits.scatter(statistics.mean(values), position, marker="D", color="#233340", s=24)
    chart_axis(fits, data["arms"])
    fits.set_xlim(left=0)
    fits.set_title("Complete training runs per fit", pad=12)
    fits.set_xlabel("Minutes; fit work is nested in execution")
    keys = ("execution_wall_seconds", "audit_wall_seconds", "v2_logged_update_seconds_lower_bound")
    phases.barh(np.arange(3), [data["costs"][key] / 60 for key in keys], color=["#496E8B", "#657F95", "#B0804C"])
    phases.set_yticks(np.arange(3), ["v3 execution", "v3 audit", "v2 logged updates\n(lower bound)"])
    phases.invert_yaxis()
    phases.tick_params(axis="both", length=0, pad=7)
    phases.grid(axis="x", alpha=.7)
    phases.set_axisbelow(True)
    phases.set_xlabel("Minutes; these are not a full-program total")
    phases.set_title("Current phases and known prior work", pad=12)
    finish(fig, data, "Training and audit costs, with interrupted work kept visible",
           "v2 logged update time and completed-fit time overlap. Its terminal wall time is unknown; earlier v1 and other work are not totaled here.\n"
           "Shared-host measurements include concurrent workloads. This quality/replay experiment establishes no isolated latency or speed claim.",
           out, "costs")


def report(data):
    lines = ["# Chess pin-quality v3", "",
             "**" + ("Synthetic engineering fixture, not research results." if data["engineering"] else
                       "Previously exposed development panels, not independent confirmation.") + "**", "",
             "All 24 trained fits and three frozen-backbone baselines are retained. No Elo or gameplay claim follows from these metrics.", "",
             "| Method | Ordinary agreement | Shift agreement | Ordinary target NLL | Shift target NLL |",
             "|---|---:|---:|---:|---:|"]
    for arm in data["methods"]:
        rows = [data["family_means"][split][arm] for split in SPLITS]
        lines.append(f"| {LABELS[arm]} | {100*rows[0]['agreement']:.3f}% | {100*rows[1]['agreement']:.3f}% | "
                     f"{rows[0]['target_nll']:.6f} | {rows[1]['target_nll']:.6f} |")
    lines += ["", "Three-seed means are descriptive. Every individual seed remains in the figures and figure-data.json.", "",
              (f"Quality gate: **{'PASS' if data['quality_gate_passed'] else 'FAIL'}** "
              f"({sum(row['passed'] for row in data['gate_checks'])}/16 checks). "
              f"Native/cached numerical gate: **{'PASS' if data['numerical']['numerical_gate_passed'] else 'FAIL'}**. "
              f"Continuation: **{'PASS' if data['continuation_passed'] else 'FAIL'}**."), "",
              "| Panel | Joint minus comparator | Mean gain (pp) | Minimum paired seed (pp) | Required mean (pp) | Check |",
              "|---|---|---:|---:|---:|---|"]
    for row in data["gate_checks"]:
        lines.append(f"| {row['split']} | {LABELS[row['comparator']]} | {100*row['mean_gain']:+.3f} | "
                     f"{100*row['minimum_paired_seed_gain']:+.3f} | {100*row['required_mean_gain']:.1f} | "
                     f"{'PASS' if row['passed'] else 'FAIL'} |")
    lines += ["", "Every comparison also requires a paired-seed floor of -0.5 percentage points. The designated joint arm is unchanged.", ""]
    for name in FIGURES:
        lines += [f"![{name.replace('-', ' ').capitalize()}]({name}.png)", ""]
    lines += ["| Method | Maximum native/cached score error | Failed method-position cases | Changed chosen moves |",
              "|---|---:|---:|---:|"]
    for arm in data["methods"]:
        row = data["numerical"]["native_by_method"][arm]
        lines.append(f"| {LABELS[arm]} | {row['max_score_error']:.8g} | {row['failed_cases']} | {row['choice_changes']} |")
    lines += ["", "Cost ledger (seconds):", ""]
    lines += [f"- {key.replace('_', ' ')}: {value}" for key, value in data["costs"].items()]
    lines += ["", "Interpretation limits:", ""] + [f"- {value}" for value in data["notes"]]
    lines += ["", data["limits"], "", "Prior knowledge: " + data["prior_knowledge"], "",
              "Audit scope: " + data["audit_scope"], "",
              "[All plotted values, fit records, gates and audited intervals](figure-data.json).", "",
              f"Audit receipt SHA-256: `{data['provenance']['audit_receipt_sha256']}`.", ""]
    return "\n".join(lines)


def render(plan_path, audit, execution, expected_receipt_sha256, out):
    values = authenticate(plan_path, audit, execution, expected_receipt_sha256)
    data = figure_data(*values, expected_receipt_sha256)
    out = Path(out)
    require(not out.exists(), "Exclusive figure output required")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write(out / "figure-data.json", data)
        style()
        render_metric(data, out, "agreement")
        render_metric(data, out, "target_nll")
        render_comparisons(data, out)
        render_parity(data, out)
        render_costs(data, out)
        (out / "README.md").write_text(report(data))
        require(authenticate(plan_path, audit, execution, expected_receipt_sha256) == values, "Artifacts changed while rendering")
        receipt = {"status": "completed", "version": VERSION, "engineering": data["engineering"],
                   "audit_receipt_sha256": expected_receipt_sha256, "plan_sha256": data["provenance"]["plan_sha256"],
                   "source_sha256": sha(__file__), "saved_artifact_only": True, "new_model_calls": 0, "new_engine_calls": 0,
                   "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__},
                   "files": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}}
        write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        plt.close("all")
        write(out / "failed.json", {"status": "failed", "error": repr(error), "audit_receipt_sha256": expected_receipt_sha256})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.plan, args.audit, args.execution, args.expected_audit_receipt_sha256, args.out)
    print(json.dumps({"status": result["status"], "files": len(result["files"]), "engineering": result["engineering"]}))


if __name__ == "__main__":
    main()
