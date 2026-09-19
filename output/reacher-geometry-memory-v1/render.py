"""Render a completed, authenticated geometry-scored memory study without inference.

This reporting helper imports no research, model, runner or native simulator
modules. External completion authorization and plan/audit hashes are required
before any outcome files are read. It deliberately rejects engineering data.
The replay selection is fixed here before any scored outcomes: ordinary case0,
pair0, all four model classes under geometry scoring, every one of the 50 saved actions.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = "reacher-geometry-memory-v1"
PLAN_SHA = "23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee"
FAMILIES = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")
MODES = ("geometry",)
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "public_kinematic", "zero", "uniform")
GROUPS = tuple((family, mode) for family in FAMILIES for mode in MODES)
LABELS = ("Persistent GRU", "Reset GRU / current packet", "Reset GRU / cached angles", "Cached-angle MLP")
REFERENCE_LABELS = ("Known-state physics / CEM256", "Particle physics / CEM256", "Public kinematics / CEM256", "Zero action", "Uniform action")
COLORS = ("#3468a6", "#138574", "#9954a0", "#d08025")
PANEL_LABELS = ("Full observations", "Ordinary: six-step gaps", "Shift: ten-step gaps")
PLOT_NAMES = ("native-costs", "utility-vs-cost", "gate-checks")
FILES = ("report.json", "tables.md", "fixed-case-replay.gif", "replay-preview.png",
         *(name + "." + extension for name in PLOT_NAMES for extension in ("png", "svg", "pdf")))
GEOMETRY_PLAN_SHA = "94c90f7585303f4edd88e1b0cb0dc8a488a2e2f07237aae4cfb489ade2068a2b"
GEOMETRY_AUDIT_SHA = "9cd5fdde29601ae5227afb9b3725be1fc855d4c7ba72aa7ca4c83248ae8de72e"
PARENT_PLAN_SHA = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"
PARENT_AUDIT_SHA = "d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790"
LIMITS = (
    "Zero new fits: all twelve inherited models and all 51 control rows are retained. The older cache failure and score-intervention success remain separate claims.",
    "The GRU arms are all recurrent within imagined rollouts. This compares separately trained real-assimilation policies and validity gating, not recurrence in isolation.",
    "All nonfloor controllers use CEM256 with the same geometry interface, horizon 12 and block 3. Equal candidate budgets do not match total compute or information.",
    "Physics references use supplied nominal dynamics. Known-state has privileged current qpos/qvel; public observers do not. None is an optimal stochastic-control oracle.",
    "Geometry approximates native RK4 cached-body reward; distance at projected mean angles is not expected distance.",
    "Conditional paired-case intervals fix these three saved fits and training corpus. They are not training-seed population uncertainty or multiplicity-corrected confirmation.",
    "Shared-host whole-row wall includes setup, native stepping and evidence storage. Amortized throughput is not isolated latency, FLOPs or a cross-machine speed result.",
    "The replay uses ordinary case0, pair0, every saved action and all four classes by a fixed pre-outcome rule. It is not a selected winner or aggregate result.",
    "Saved qpos is used only for schematic centerline drawing. Public-availability shading concerns decision inputs; no clean hidden angles are supplied to the controller.",
    "No connectome advantage, biological superiority, novel architecture, calibrated uncertainty or cross-environment generality is established.",
)


def require(value, label):
    if not value:
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
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "Nonfinite JSON value")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def member(folder, name):
    relative = Path(name)
    require(isinstance(name, str) and name and not relative.is_absolute()
            and ".." not in relative.parts, "Safe relative artifact member")
    folder, path = Path(folder), Path(folder) / relative
    require(path.resolve().is_relative_to(folder.resolve())
            and not any(part.is_symlink() for part in (path, *path.parents)), "Artifact path/symlink boundary")
    return path


def checked(path, expected):
    require(isinstance(expected, str) and len(expected) == 64
            and set(expected) <= set("0123456789abcdef"), "External SHA256 required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, f"Hash mismatch: {path}")
    return path


def complete_members(folder, hashes, extra):
    folder = Path(folder)
    paths = list(folder.rglob("*"))
    require(not folder.is_symlink() and not any(path.is_symlink() for path in paths), "No artifact symlinks")
    require({path.relative_to(folder).as_posix() for path in paths if path.is_file()} == set(hashes) | set(extra),
            "Exact completed artifact membership")
    for name, expected in hashes.items():
        checked(member(folder, name), expected)


def close(a, b, label):
    require(type(a) in (float, int) and type(b) in (float, int)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), label)


def policy(family, pair, mode="geometry"):
    require(mode == "geometry", "Geometry-only policy label")
    return f"{family}-{pair}"

def authenticate(args):
    require(args.completed_authorized is True, "Explicit completed-study authorization required before outcome reads")
    require(args.expected_plan_sha256 == PLAN_SHA, "Exact frozen geometry-memory plan required")
    plan = read(checked(args.plan, PLAN_SHA))
    receipt = read(checked(Path(args.audit) / "receipt.json", args.expected_audit_receipt_sha256))
    require(plan["study"] == receipt["version"] == STUDY
            and plan["engineering"] is receipt["engineering"] is False
            and plan["rng_namespace"] == STUDY + "-scored", "Scored frozen study only")
    require(receipt["status"] == "completed" and receipt["saved_output_only"] is True
            and receipt["plan_sha256"] == PLAN_SHA and receipt["source_sha256"] == plan["sources"]
            and len(plan["sources"]) == 90 and receipt["runtime"] == plan["runtime"], "Completed independent source-bound audit")
    for source_name, digest, audit_digest, previous_pass in (
        ("cache_source", PARENT_PLAN_SHA, PARENT_AUDIT_SHA, False),
        ("geometry_source", GEOMETRY_PLAN_SHA, GEOMETRY_AUDIT_SHA, True)):
        parent = plan[source_name]
        require(parent["plan_sha256"] == digest and parent["audit_receipt_sha256"] == audit_digest
                and parent["previous_scientific_gate_passed"] is previous_pass, "Preserve both completed historical lineages")
    for name, expected in plan["sources"].items():
        checked(member(ROOT, name), expected)
    require(set(receipt["files"]) == {"summary.json", "README.md"}, "Exact terminal audit schema")
    complete_members(args.audit, receipt["files"], {"receipt.json"})
    completed = read(checked(Path(args.execution) / "completed.json", receipt["execution_completed_sha256"]))
    require(completed["status"] == "completed" and completed["study"] == STUDY
            and completed["plan_sha256"] == PLAN_SHA and completed["new_fits"] == completed["astra_calls"] == 0
            and completed["restored_models"] == 12 and completed["control_rows"] == 51
            and completed["diagnostic_roots"] == 0 and completed["files"] == receipt["execution_members"],
            "Complete twelve-model/51-row/zero-diagnostic execution")
    complete_members(args.execution, completed["files"], {"completed.json"})
    summary = read(Path(args.audit) / "summary.json")
    require(summary["status"] == "completed" and summary["version"] == STUDY and summary["engineering"] is False
            and summary["saved_output_only"] is True and summary["plan_sha256"] == PLAN_SHA
            and summary["execution_completed_sha256"] == receipt["execution_completed_sha256"]
            and summary["costs"] == receipt["costs"]
            and summary["new_model_calls"] == summary["new_policy_calls"] == summary["new_fits"] == 0,
            "Authenticated saved-output scope")
    require(plan["cap_seconds"] == 5700 and plan["audit_cap_seconds"] == 3300
            and 0 < completed["wall_seconds"] <= 5700
            and 0 < summary["costs"]["audit_validation_wall_seconds"] <= 3300,
            "Both phases within frozen caps")
    close(completed["wall_seconds"], summary["costs"]["execution_wall_seconds"], "Execution timing identity")
    close(completed["cumulative_attempt_wall_seconds"], summary["costs"]["cumulative_attempt_wall_seconds"], "Nested cost identity")
    validate_summary(plan, summary)
    inputs = {"plan_sha256": PLAN_SHA, "audit_receipt_sha256": args.expected_audit_receipt_sha256,
              "audit_summary_sha256": receipt["files"]["summary.json"],
              "execution_completed_sha256": receipt["execution_completed_sha256"],
              "execution_member_count": len(completed["files"]), "frozen_source_count": 90}
    return plan, receipt, summary, inputs

def validate_summary(plan, summary):
    require(plan["arms"] == list(FAMILIES) and plan["score_modes"] == ["geometry"]
            and plan["pairs"] == list(PAIRS) and plan["panels"] == list(PANELS)
            and plan["references"] == list(REFERENCES) and plan["control_episodes"] == 64
            and plan["steps"] == 50 and plan["dt"] == .02
            and plan["planner"] == plan["physics_planner"] == "cem256"
            and plan["planning_horizon"] == 12 and plan["action_block"] == 3, "Complete frozen comparison")
    fits = {policy(family, pair) for family in FAMILIES for pair in PAIRS}
    require(set(summary["inherited_fits"]) == fits and set(summary["control"]) == set(PANELS), "Every inherited fit and panel")
    expected_counts = {"native_control_transitions_checked": 163200,
        "native_nominal_candidate_transitions_checked": 78741504,
        "native_nominal_selected_transitions_checked": 28800,
        "public_observer_transitions_checked": 310464, "native_transitions_checked": 78933504}
    for key, expected in expected_counts.items():
        require(type(summary[key]) is int and summary[key] == expected, "Complete native scope: " + key)
    for key in ("native_control_max_abs_error", "native_nominal_max_abs_error", "public_observer_max_abs_error"):
        require(0 <= summary[key] <= 1e-10, "Declared native replay tolerance")
    for panel in PANELS:
        require(set(summary["control"][panel]) == fits | set(REFERENCES), "All17 rows per panel")
        for name, row in summary["control"][panel].items():
            costs = np.asarray(row["episode_costs"], np.float64)
            require(costs.shape == (64,) and np.isfinite(costs).all() and (costs >= 0).all(), "All64 native cases retained")
            close(float(costs.mean()), row["mean_cost"], "Per-case cost arithmetic")
            require(row["row_wall_seconds"] > 0 and row["setup_seconds"] >= 0
                    and len(row["decision_seconds"]) == 50 and min(row["decision_seconds"]) >= 0, "Complete measured timing")
            close(sum(row["decision_seconds"]), row["decision_wall_seconds"], "Decision timing sum")
            require(row["row_wall_seconds"] + 1e-6 >= row["setup_seconds"] + row["decision_wall_seconds"] + row["native_step_seconds"],
                    "Row wall contains nested timing scopes")
            planned = name not in ("zero", "uniform")
            require(row["planner_used"] is planned and row["candidate_evaluations"] == (819200 if planned else 0), "Paid candidate coverage")
            if name in fits:
                require(row["scoring_work"]["imagined_transitions"] == 8749056
                        and row["scoring_work"]["selected_geometry_samples"] == 3200, "Candidate and selected learned work")
            else:
                physics = row["physics_work"]
                require(physics["candidate_native_transitions_replayed"] == (8749056 if planned else 0)
                        and physics["selected_native_transitions_replayed"] == (3200 if planned else 0), "Separate nominal work")
    saved, checks = summary["continuation_gate"], independent_checks(summary)
    require(len(saved["checks"]) == len(checks) == 25 and type(saved["passed"]) is bool, "All25 checks retained")
    for actual, expected in zip(saved["checks"], checks, strict=True):
        require(actual["name"] == expected["name"] and actual["comparison"] == "le"
                and type(actual["passed"]) is bool and actual["passed"] == expected["passed"], "Gate identity/result")
        close(actual["left"], expected["left"], "Gate left arithmetic")
        close(actual["right"], expected["right"], "Gate right arithmetic")
    require(saved["passed"] == all(check["passed"] for check in checks)
            and saved["secondary_cannot_rescue_primary"] is True, "Do not suppress or rescue failed checks")
    controls = [row for panel in PANELS for row in summary["control"][panel].values()]
    for cost_key, row_key in (("control_row_wall_seconds", "row_wall_seconds"), ("control_setup_seconds", "setup_seconds"),
                              ("control_decision_seconds", "decision_wall_seconds"), ("control_native_step_seconds", "native_step_seconds")):
        close(summary["costs"][cost_key], sum(row[row_key] for row in controls), "Complete cost sum: " + cost_key)
    learned = summary["costs"]["learned_control_scoring"]
    physics = summary["costs"]["physics_control_scoring"]
    require(learned["candidate_evaluations"] == 29491200 and learned["imagined_transitions"] == 314966016
            and learned["selected_geometry_samples"] == 115200, "Aggregate paid learned work")
    require(physics["candidate_native_transitions_replayed"] == 78741504
            and physics["selected_native_transitions_replayed"] == 28800
            and physics["counts"]["native_substeps_completed"] == 157540608, "Aggregate candidate/selected physics work")
    require(summary["costs"]["compute_matched"] is False and summary["costs"]["new_fits"] == summary["costs"]["new_optimizer_steps"] == 0,
            "Zero new fitting, candidate-only budget matching")
    contrasts = {"persistent_minus_encoded_current_gru": ("residual_gru", "encoded_current_gru"),
        "persistent_minus_cached_gru": ("residual_gru", "cached_gru"), "persistent_minus_cached_mlp": ("residual_gru", "cached_mlp"),
        "cached_gru_minus_encoded_current_gru": ("cached_gru", "encoded_current_gru"), "cached_gru_minus_cached_mlp": ("cached_gru", "cached_mlp")}
    require(set(summary["paired_descriptive_comparisons"]) == set(PANELS), "All conditional comparison panels")
    for panel in PANELS:
        require(set(summary["paired_descriptive_comparisons"][panel]) == set(contrasts), "All declared contrasts retained")
        for name, (left, right) in contrasts.items():
            item = summary["paired_descriptive_comparisons"][panel][name]
            a, b = (np.array([summary["control"][panel][policy(family, pair)]["episode_costs"] for pair in PAIRS]) for family in (left, right))
            delta = a - b
            close(item["mean_cost_difference"], float(delta.mean(0).mean()), "Paired difference arithmetic")
            for saved_delta, expected_delta in zip(item["paired_fit_cost_differences"], delta.mean(1), strict=True):
                close(saved_delta, float(expected_delta), "Every fit-pair contrast")
            require(item["all_three_pairs_nonworse"] == bool((delta.mean(1) <= 0).all())
                    and item["conditional_on_all_three_saved_fit_pairs"] is True and item["cases"] == 64,
                    "Conditional interval scope")
            interval = np.asarray(item["episode_paired_percentile_95"], float)
            require(interval.shape == (2,) and np.isfinite(interval).all() and interval[0] <= interval[1], "Saved ordered conditional interval")
            if float(b.mean()):
                close(item["percent_cost_change"], 100 * float(delta.mean(0).mean()) / float(b.mean()), "Family percent-change arithmetic")
            else:
                require(item["percent_cost_change"] is None, "Undefined relative change is retained")

def family_cost(summary, panel, family, mode="geometry"):
    require(mode == "geometry", "Geometry-only family mean")
    return float(np.mean([summary["control"][panel][policy(family, pair)]["episode_costs"] for pair in PAIRS]))

def independent_checks(summary):
    checks = []
    def add(name, left, right):
        checks.append({"name": name, "left": float(left), "right": float(right), "comparison": "le", "passed": bool(left <= right)})
    for comparator in ("encoded_current_gru", "cached_gru"):
        for panel in ("ordinary", "shift"):
            add(f"gap_mean/{panel}/{comparator}", family_cost(summary, panel, "residual_gru"), .97 * family_cost(summary, panel, comparator))
            for pair in PAIRS:
                add(f"gap_pair/{panel}/{comparator}/{pair}", summary["control"][panel][policy("residual_gru", pair)]["mean_cost"],
                    summary["control"][panel][policy(comparator, pair)]["mean_cost"])
        add(f"full_mean/{comparator}", family_cost(summary, "full", "residual_gru"), 1.02 * family_cost(summary, "full", comparator))
    for panel in ("ordinary", "shift"):
        for pair in PAIRS:
            add(f"competence/{panel}/residual_gru-{pair}", summary["control"][panel][policy("residual_gru", pair)]["mean_cost"],
                .9 * summary["control"][panel]["zero"]["mean_cost"])
    add("competence/ordinary/known_state", summary["control"]["ordinary"]["known_state"]["mean_cost"],
        .9 * summary["control"]["ordinary"]["zero"]["mean_cost"])
    return checks

def rows_for_report(summary):
    rows = []
    for panel in PANELS:
        for family, mode in GROUPS:
            for pair in PAIRS:
                name = policy(family, pair, mode)
                source = summary["control"][panel][name]
                rows.append({"panel": panel, "policy": name, "family": family, "mode": mode, "pair": pair,
                    "mean_native_cost": source["mean_cost"], "whole_row_wall_seconds": source["row_wall_seconds"],
                    "whole_row_amortized_ms": source["row_wall_seconds"] * 1000 / (64 * 50),
                    "setup_seconds": source["setup_seconds"], "decision_seconds": source["decision_wall_seconds"],
                    "native_seconds": source["native_step_seconds"], "candidates_per_decision": 256,
                    "episode_costs": source["episode_costs"],
                    "candidate_sequences": source["candidate_evaluations"],
                    "candidate_transition_samples": source["scoring_work"]["imagined_transitions"],
                    "selected_transition_samples": source["scoring_work"]["selected_geometry_samples"]})
        for name in REFERENCES:
            source = summary["control"][panel][name]
            rows.append({"panel": panel, "policy": name, "reference": True,
                "mean_native_cost": source["mean_cost"], "whole_row_wall_seconds": source["row_wall_seconds"],
                "whole_row_amortized_ms": source["row_wall_seconds"] * 1000 / (64 * 50),
                "setup_seconds": source["setup_seconds"], "decision_seconds": source["decision_wall_seconds"],
                "native_seconds": source["native_step_seconds"],
                "candidates_per_decision": 256 if name in ("known_state", "particle", "public_kinematic") else 0,
                "episode_costs": source["episode_costs"],
                "candidate_sequences": source["candidate_evaluations"],
                "candidate_transition_samples": source["physics_work"]["candidate_native_transitions_replayed"],
                "selected_transition_samples": source["physics_work"]["selected_native_transitions_replayed"]})
    return rows


def save_figure(fig, out, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / (name + "." + extension), dpi=180)


def plot_results(summary, report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), sharey=True)
    for ax, panel, title in zip(axes, PANELS, PANEL_LABELS, strict=True):
        for index, ((family, mode), color) in enumerate(zip(GROUPS, COLORS, strict=True)):
            values = [summary["control"][panel][policy(family, pair, mode)]["mean_cost"] for pair in PAIRS]
            for offset, value, marker in zip((-.15, 0, .15), values, markers, strict=True):
                ax.scatter(value, index + offset, c=color, marker=marker, s=46, zorder=4)
            ax.scatter(np.mean(values), index, marker="|", s=260, color="#152332", linewidths=2, zorder=5)
        for index, name in enumerate(REFERENCES, 4):
            ax.scatter(summary["control"][panel][name]["mean_cost"], index, marker="D", s=33, color="#626d78")
        ax.set_title(title)
        ax.set_xlabel("Mean episode native cost (lower is better)")
        ax.grid(axis="x", alpha=.18)
        ax.set_xlim(left=0)
        ax.set_yticks(range(9), (*LABELS, *REFERENCE_LABELS))
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker=marker, color="#425362", linestyle="None", label=pair)
               for marker, pair in zip(markers, PAIRS, strict=True)]
    handles.append(Line2D([], [], marker="|", color="#152332", linestyle="None", markersize=13, label="All-three-fit mean"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.55, .06))
    fig.suptitle("OpenJev: real-observation memory policies, fixed geometry scoring", fontsize=15)
    fig.text(.02, .02, "All 36 learned points and 15 references. All planned rows use CEM256; floors use no planner. No best-fit selection.", fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .93))
    save_figure(fig, out, "native-costs")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7), sharey=True)
    ref_markers = ("D", "P", "X", "+", "*")
    for ax, panel, title in zip(axes, PANELS, PANEL_LABELS, strict=True):
        for row in (row for row in report["control_rows"] if row["panel"] == panel):
            if row.get("reference"):
                marker = ref_markers[REFERENCES.index(row["policy"])]
                ax.scatter(row["whole_row_amortized_ms"], row["mean_native_cost"], marker=marker, color="#626d78", s=75, zorder=4)
            else:
                group = GROUPS.index((row["family"], row["mode"]))
                ax.scatter(row["whole_row_amortized_ms"], row["mean_native_cost"], marker=markers[PAIRS.index(row["pair"])],
                           color=COLORS[group], s=50, zorder=5)
        ax.set_xscale("log")
        ax.grid(alpha=.18)
        ax.set_title(title)
        ax.set_xlabel("Whole-row wall / (64 cases × 50 actions), ms")
    axes[0].set_ylabel("Mean native episode cost (lower is better)")
    handles = [Line2D([], [], marker="o", color=color, linestyle="None", label=label)
               for color, label in zip(COLORS, LABELS, strict=True)]
    handles += [Line2D([], [], marker=marker, color="#626d78", linestyle="None", label=label)
                for marker, label in zip(ref_markers, REFERENCE_LABELS, strict=True)]
    handles += [Line2D([], [], marker=marker, color="#425362", linestyle="None", label=pair)
                for marker, pair in zip(markers, PAIRS, strict=True)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .07))
    fig.suptitle("Utility versus measured control cost: every fit and reference", fontsize=15)
    fig.text(.02, .02, "Shared host; setup, native steps and traces included. Amortized throughput, not isolated latency, matched total compute or a selected frontier.", fontsize=9)
    fig.tight_layout(rect=(0, .26, 1, .92))
    save_figure(fig, out, "utility-vs-cost")
    plt.close(fig)

    checks = report["continuation_gate"]["checks"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8))
    for ax, subset in zip(axes, (checks[:12], checks[12:]), strict=True):
        ax.axis("off")
        for index, row in enumerate(subset):
            y = .95 - index * .069
            ax.text(0, y, "PASS" if row["passed"] else "FAIL", color="#137e67" if row["passed"] else "#b43b3b", weight="bold", fontsize=10)
            ax.text(.1, y, row["name"], fontsize=9.2)
            ax.text(.1, y - .025, f'{row["left"]:.6f} ≤ {row["right"]:.6f}', fontsize=8.5, color="#617180")
    passed = sum(row["passed"] for row in checks)
    fig.suptitle(f'Continuation gate: {"PASS" if report["continuation_gate"]["passed"] else "FAIL"} ({passed}/25)', fontsize=17)
    fig.text(.03, .025, "All checks are required. Cached-MLP and other descriptive contrasts cannot rescue a failed primary gate.", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .93))
    save_figure(fig, out, "gate-checks")
    plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "numpy": np.__version__}


def xml_geometry(plan):
    path = Path(importlib.metadata.distribution("gymnasium").locate_file("gymnasium/envs/mujoco/assets/reacher.xml"))
    checked(path, plan["runtime"]["native_xml_sha256"])
    tree = ET.parse(path).getroot()
    first = tree.find("./worldbody/body[@name='body0']")
    second = first.find("./body[@name='body1']")
    tip = second.find("./body[@name='fingertip']")
    target = tree.find("./worldbody/body[@name='target']")
    def vector(item, key):
        return np.fromstring(item.attrib[key], sep=" ", dtype=np.float64)
    require(np.array_equal(vector(first, "pos")[:2], [0, 0]), "Planar origin")
    lengths = [vector(second, "pos"), vector(tip, "pos")]
    require(all(np.array_equal(value[1:], [0, 0]) for value in lengths), "Planar link offsets")
    for body, name in ((first, "joint0"), (second, "joint1")):
        joint = body.find(f"./joint[@name='{name}']")
        require(joint.attrib["type"] == "hinge" and np.array_equal(vector(joint, "axis"), [0, 0, 1])
                and np.array_equal(vector(joint, "pos"), [0, 0, 0]), "Planar zero-origin hinge")
    refs = [float(target.find(f"./joint[@name='{name}']").attrib["ref"]) for name in ("target_x", "target_y")]
    require(np.array_equal(refs, vector(target, "pos")[:2]), "Saved target qpos denotes world XY")
    return {"xml_sha256": sha(path), "link_lengths_m": [float(value[0]) for value in lengths]}


def replay_rows(plan, summary, execution, geometry, expected_members):
    rows, bindings = [], {}
    for family, mode in GROUPS:
        name = policy(family, "pair0", mode)
        folder = Path(execution) / "control" / "ordinary" / name
        relative = f"control/ordinary/{name}/episodes"
        expected_json, expected_npz = (expected_members[relative + extension] for extension in (".json", ".npz"))
        metadata = read(checked(folder / "episodes.json", expected_json))
        require(len(metadata) == 64, "All recorded cases retained")
        with np.load(checked(folder / "episodes.npz", expected_npz), allow_pickle=False) as saved:
            values = {key: saved[key][0].copy() for key in ("audit__qpos", "audit__raw_obs", "audit__time", "audit__rewards", "policy__packets")}
        q, raw, timestamps, rewards, public = (values[key] for key in values)
        require(q.shape == (51, 4) and raw.shape == (51, 10) and timestamps.shape == (51,)
                and rewards.shape == (50,) and public.shape == (51, 8)
                and all(np.isfinite(value).all() for value in values.values()), "Complete finite native replay arrays")
        require(np.allclose(np.diff(timestamps), .02, rtol=0, atol=1e-12)
                and metadata[0]["dt"] == plan["dt"] and metadata[0]["horizon"] == 50, "Native replay clock")
        require(np.array_equal(q[:, 2:4], np.broadcast_to(q[0, 2:4], (51, 2)))
                and np.allclose(public[:, 4:6], q[:, 2:4], rtol=0, atol=1e-8), "Static public target")
        valid = np.asarray(metadata[0]["sensor_schedule"], dtype=bool)
        require(valid.shape == (51,) and valid[0] and valid[-1] and (~valid).sum() == 12
                and np.array_equal(valid, public[:, 6].astype(bool)), "Actual ordinary decision visibility")
        require(np.allclose(raw[:, :2], np.cos(q[:, :2]), rtol=0, atol=1e-12)
                and np.allclose(raw[:, 2:4], np.sin(q[:, :2]), rtol=0, atol=1e-12), "Recorded angle channels")
        l0, l1 = geometry["link_lengths_m"]
        elbow = l0 * np.stack((np.cos(q[:, 0]), np.sin(q[:, 0])), -1)
        tip = elbow + l1 * np.stack((np.cos(q[:, 0] + q[:, 1]), np.sin(q[:, 0] + q[:, 1])), -1)
        cumulative = np.concatenate(([0.], -np.cumsum(rewards, dtype=np.float64)))
        close(float(cumulative[-1]), summary["control"]["ordinary"][name]["episode_costs"][0], "Replay case0 cost matches audit")
        rows.append({"name": name, "q": q, "elbow": elbow, "tip": tip, "times": timestamps,
                     "valid": valid, "cost": cumulative})
        checked(folder / "episodes.npz", expected_npz)
        checked(folder / "episodes.json", expected_json)
        bindings[name] = {"episodes_npz_sha256": expected_npz, "episodes_json_sha256": expected_json,
            "case_index": 0, "pair": "pair0", "panel": "ordinary", "native_cost": float(cumulative[-1]),
            "decision_visibility": valid[:-1].tolist(),
            "fk_vs_cached_native_displacement_max_abs_error": float(np.max(np.abs(tip - q[:, 2:4] - raw[:, -2:])))}
    for row in rows[1:]:
        require(np.array_equal(row["q"][0], rows[0]["q"][0]) and np.array_equal(row["valid"], rows[0]["valid"])
                and np.array_equal(row["times"], rows[0]["times"]), "Paired replay reset, clock and sensing schedule")
    return rows, bindings


def replay_frame(rows, step, fonts):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1200, 1030), "#f3f6fa")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1200, 90), fill="#172c40")
    draw.text((24, 12), "OpenJev | geometry-scored memory: saved schematic replay", font=fonts[26], fill="white")
    draw.text((25, 52), "Ordinary case0, pair0 | all four model classes | not a performance summary", font=fonts[18], fill="#dde8f0")
    for index, (row, label, color) in enumerate(zip(rows, LABELS, COLORS, strict=True)):
        x, y = 24 + (index % 2) * 588, 112 + (index // 2) * 362
        missing = not row["valid"][step - 1]
        draw.rounded_rectangle((x, y, x + 564, y + 342), radius=12,
                               fill="#fff0cf" if missing else "white", outline="#d3dde8")
        draw.text((x + 17, y + 12), label, font=fonts[22], fill=color)
        draw.text((x + 17, y + 43), "ANGLES MISSING" if missing else "ANGLES OBSERVED", font=fonts[16], fill="#8d5d18" if missing else "#607283")
        cx, cy, scale = x + 282, y + 180, 470
        def point(value, cx=cx, cy=cy, scale=scale):
            return round(cx + float(value[0]) * scale), round(cy - float(value[1]) * scale)
        for value in (-.2, -.1, 0, .1, .2):
            draw.line((point((value, -.22)), point((value, .22))), fill="#e4e9ee")
            draw.line((point((-.22, value)), point((.22, value))), fill="#e4e9ee")
        draw.line([point(value) for value in row["tip"][:step + 1]], fill="#c2d3df", width=2)
        tx, ty = point(row["q"][0, 2:4])
        draw.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), outline="#b13f48", width=3)
        draw.line((tx - 11, ty, tx + 11, ty), fill="#b13f48")
        draw.line((tx, ty - 11, tx, ty + 11), fill="#b13f48")
        origin, elbow, tip = point((0, 0)), point(row["elbow"][step]), point(row["tip"][step])
        draw.line((origin, elbow), fill=color, width=10)
        draw.line((elbow, tip), fill="#476c83", width=8)
        for dot in (origin, elbow, tip):
            draw.ellipse((dot[0] - 5, dot[1] - 5, dot[0] + 5, dot[1] + 5), fill="#172c40")
        draw.text((x + 17, y + 311), f'Saved cumulative native cost: {row["cost"][step]:.3f}', font=fonts[18], fill="#172c40")
    draw.text((26, 854), f'Step {step:02d}/50 | native time {rows[0]["times"][step] - rows[0]["times"][0]:.2f}s | 5x slower playback | axes: meters',
              font=fonts[18], fill="#172c40")
    lines = ("Amber = angles missing at this action's decision. State shown is saved post-action qpos.",
             "Red cross = target; faint trail = fingertip path. Clean angles never entered model inputs in gaps.",
             "Native cost = negative reward. Schematic FK can differ from cached native body positions.",
             "All 50 saved actions, 100ms per frame: one native second shown over five seconds. No selected winning clip.")
    for index, line in enumerate(lines):
        draw.text((26, 890 + index * 27), line, font=fonts[16], fill="#526777")
    return image


def render_replay(plan, summary, execution, out, expected_members):
    from PIL import Image, ImageFont
    geometry = xml_geometry(plan)
    rows, bindings = replay_rows(plan, summary, execution, geometry, expected_members)
    candidates = (Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    font_path = next((path for path in candidates if path.is_file()), None)
    require(font_path is not None, "Hashable Arial or DejaVuSans font required")
    fonts = {size: ImageFont.truetype(str(font_path), size=size) for size in (16, 18, 22, 26)}
    frames = [replay_frame(rows, step, fonts) for step in range(1, 51)]
    gif = out / "fixed-case-replay.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False, disposal=2)
    preview = Image.new("RGB", (3600, 1030))
    first_missing = int(np.flatnonzero(~rows[0]["valid"][:-1])[0])
    for index, image in enumerate((frames[0], frames[first_missing], frames[-1])):
        preview.paste(image, (1200 * index, 0))
    preview.save(out / "replay-preview.png")
    with Image.open(gif) as saved:
        require(saved.n_frames == 50, "All50 frames retained")
        for index in range(50):
            saved.seek(index)
            require(saved.info["duration"] == 100, "Exact100ms frame duration")
    return {"selection": "ordinary case0/pair0; all four model classes; fixed before outcomes", "models": bindings,
        "frames": 50, "frame_duration_ms": 100, "playback_seconds": 5, "native_episode_seconds": 1,
        "geometry": geometry, "font": {"path": str(font_path), "sha256": sha(font_path)},
        "frame_native_timestamps_seconds": rows[0]["times"][1:].tolist(),
        "visibility": "Pre-action packet t determines shading; saved post-action qpos t+1 determines motion.",
        "pillow": importlib.metadata.version("pillow"), "native_pixels": False}


def tables(report):
    gate, costs = report["continuation_gate"], report["costs"]
    lines = ["# Geometry-scored real-observation memory study", "",
        f'Continuation: **{"PASS" if gate["passed"] else "FAIL"}** ({sum(row["passed"] for row in gate["checks"])}/25).', "",
        "All twelve inherited fits and 51 control rows are retained. Zero new fits or training updates. All planned rows use CEM256; total wall time and information are not matched.", "",
        "| Panel | Policy | Mean native cost | Whole-row wall (s) | Amortized wall (ms/action/case) |",
        "|---|---|---:|---:|---:|"]
    for row in report["control_rows"]:
        lines.append(f'| {row["panel"]} | {row["policy"]} | {row["mean_native_cost"]:.6f} | {row["whole_row_wall_seconds"]:.3f} | {row["whole_row_amortized_ms"]:.6f} |')
    lines += ["", "Every row contains 64 paired cases and 50 decisions. Wall time includes setup, native stepping and trace storage on a shared host.", "",
        "| Gate check | Measured cost | Required upper bound | Result |", "|---|---:|---:|---|"]
    for row in gate["checks"]:
        lines.append(f'| {row["name"]} | {row["left"]:.6f} | {row["right"]:.6f} | {"PASS" if row["passed"] else "FAIL"} |')
    lines += ["", "| Panel / contrast | Mean cost difference | Percent cost change | Conditional paired-case 95% interval | Pair 0 / 1 / 2 differences |",
              "|---|---:|---:|---|---|"]
    for panel in PANELS:
        for name, row in report["paired_descriptive_comparisons"][panel].items():
            percent = "undefined" if row["percent_cost_change"] is None else f'{row["percent_cost_change"]:.3f}%'
            interval = row["episode_paired_percentile_95"]
            pairs = " / ".join(f"{value:.6f}" for value in row["paired_fit_cost_differences"])
            lines.append(f'| {panel} / {name} | {row["mean_cost_difference"]:.6f} | {percent} | [{interval[0]:.6f}, {interval[1]:.6f}] | {pairs} |')
    lines += ["", "Intervals are copied from the authenticated audit without new resampling. They condition on the three saved fits and training corpus; they are not population-level seed uncertainty or multiplicity-corrected confirmation.", "",
              "| Paid scope | Wall seconds |", "|---|---:|"]
    for key in ("inheritance_copy_wall_seconds", "restore_wall_seconds", "innovation_generation_and_storage_seconds",
                "control_row_wall_seconds", "execution_wall_seconds", "audit_validation_wall_seconds"):
        lines.append(f'| {key} | {costs[key]:.6f} |')
    lines += [f'| Prior geometry lineage, including its cache ancestry | {costs["geometry_context_costs"]["cumulative_attempt_wall_seconds"]:.6f} |',
              f'| Cumulative through this execution | {costs["cumulative_attempt_wall_seconds"]:.6f} |', "",
              "Phase and kernel timings are nested. Do not add the cache parent a second time to the geometry lineage. Engineering preparation and publication are separate costs.", "",
              "| Work scope | Count |", "|---|---:|"]
    for name, value in report["work_counts"].items():
        lines.append(f"| {name} | {value:,} |")
    lines += ["", "The nominal planning replay count is separate from executed native-control replay and public-observer reconstruction. The selected one-action predictions are charged separately from candidate trajectories.", ""]
    lines += ["- " + limit for limit in report["limits"]]
    return "\n".join(lines) + "\n"

def render(args):
    require(args.completed_authorized is True, "Completed-study authorization is required before result reads")
    out = Path(args.out)
    require(not out.exists(), "Exclusive first rendering attempt required")
    for source in (Path(args.execution), Path(args.audit), Path(args.plan).parent):
        require(not out.resolve().is_relative_to(source.resolve()), "Output must not alter input artifact trees")
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    renderer_sha = sha(Path(__file__))
    try:
        plan, audit, summary, inputs = authenticate(args)
        report = {"status": "completed", "study": STUDY, "engineering": False, "inputs": inputs,
            "continuation_gate": summary["continuation_gate"], "control_rows": rows_for_report(summary),
            "family_means": {panel: {family: family_cost(summary, panel, family, mode)
                for family, mode in GROUPS} for panel in PANELS},
            "paired_descriptive_comparisons": summary["paired_descriptive_comparisons"],
            "costs": summary["costs"],
            "work_counts": {
                "learned_candidate_sequences": summary["costs"]["learned_control_scoring"]["candidate_evaluations"],
                "learned_candidate_transitions": summary["costs"]["learned_control_scoring"]["imagined_transitions"],
                "learned_selected_advances": summary["costs"]["learned_control_scoring"]["selected_geometry_samples"],
                "physics_candidate_sequences": 7372800,
                "physics_candidate_transitions": summary["native_nominal_candidate_transitions_checked"],
                "physics_selected_advances": summary["native_nominal_selected_transitions_checked"],
                "physics_candidate_integration_substeps": 157483008,
                "physics_selected_integration_substeps": 57600,
                "executed_native_control_transitions": summary["native_control_transitions_checked"],
                "public_observer_native_transitions": summary["public_observer_transitions_checked"]},
            "limits": list(LIMITS) + summary["limits"], "new_model_calls": 0, "new_native_calls": 0}
        runtime = plot_results(summary, report, out)
        replay = render_replay(plan, summary, args.execution, out, audit["execution_members"])
        report["replay"] = replay
        write(out / "report.json", report)
        with (out / "tables.md").open("x") as handle:
            handle.write(tables(report))
        # Guard the externally authenticated receipt and used source bindings
        # again after rendering, without invoking inference or native replay.
        checked(args.plan, args.expected_plan_sha256)
        checked(Path(args.audit) / "receipt.json", args.expected_audit_receipt_sha256)
        checked(Path(args.audit) / "summary.json", audit["files"]["summary.json"])
        checked(Path(args.execution) / "completed.json", audit["execution_completed_sha256"])
        for name, expected in plan["sources"].items():
            checked(member(ROOT, name), expected)
        for name, binding in replay["models"].items():
            folder = Path(args.execution) / "control" / "ordinary" / name
            checked(folder / "episodes.npz", binding["episodes_npz_sha256"])
            checked(folder / "episodes.json", binding["episodes_json_sha256"])
        require(sha(Path(__file__)) == renderer_sha, "Renderer source changed during rendering")
        receipt = {"status": "completed", "study": STUDY, "engineering": False,
            "scope": "saved-artifact reporting and fixed-case schematic replay only", "inputs": inputs,
            "renderer_source_sha256": renderer_sha, "source_sha256": plan["sources"],
            "new_model_calls": 0, "new_policy_calls": 0, "new_native_calls": 0,
            "control_rows": 51, "learned_rows": 36, "reference_rows": 15, "inherited_models": 12, "work_counts": report["work_counts"],
            "checks_passed": sum(row["passed"] for row in report["continuation_gate"]["checks"]),
            "gate_passed": report["continuation_gate"]["passed"], "replay": replay,
            "runtime": runtime, "wall_seconds": time.perf_counter() - begin,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "files": {name: sha(out / name) for name in FILES}}
        write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        write(out / "failed.json", {"status": "failed", "error": repr(error),
            "scope": "reporting_only_no_retry", "wall_seconds": time.perf_counter() - begin})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completed-authorized", action="store_true")
    args = parser.parse_args()
    receipt = render(args)
    print(json.dumps({"status": receipt["status"], "receipt_sha256": sha(args.out / "receipt.json"),
                      "files": list(receipt["files"]), "gate_passed": receipt["gate_passed"]}))


if __name__ == "__main__":
    main()
