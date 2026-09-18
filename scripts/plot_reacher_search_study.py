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


def authenticate(audit, expected):
    if sha(audit / "receipt.json") != expected:
        raise ValueError("External audit receipt identity mismatch")
    receipt = json.loads((audit / "receipt.json").read_text())
    if (receipt.get("status") != "completed" or receipt.get("version") != "reacher-search-v1"
            or receipt.get("saved_output_only") is not True):
        raise ValueError("A completed saved-output search audit is required")
    for name, digest in receipt["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or (audit / name).is_symlink():
            raise ValueError("Invalid audit member path")
        if sha(audit / name) != digest:
            raise ValueError(f"Audit member changed: {name}")
    if "summary.json" not in receipt["files"]:
        raise ValueError("Summary is not authenticated")
    summary = json.loads((audit / "summary.json").read_text())
    if (summary.get("status") != "completed" or summary.get("version") != "reacher-search-v1"
            or summary.get("saved_output_only") is not True):
        raise ValueError("Unexpected summary scope")
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
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#20303b"})
    control_table, diagnostic_table = [], []
    for panel in PANELS:
        for name, row in summary["control"][panel].items():
            control_table.append({"panel": panel, "controller": name,
                                  "mean_episode_cost": row["mean_cost"],
                                  "setup_seconds": row["setup_seconds"],
                                  "decision_wall_seconds": row["decision_wall_seconds"],
                                  "native_step_seconds": row["native_step_seconds"],
                                  "row_wall_seconds": row["row_wall_seconds"]})
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
                ax.plot([min(values), max(values)], [y, y], color=color, alpha=.55)
                ax.scatter(values, y + np.array([-.09, 0, .09]), color=color,
                           marker=MARKERS[p], s=40, zorder=3)
        for name, color, label in (("zero", "#a55a4c", "Zero command"),
                                   ("known_state", "#c58b2a", "Known-state physics")):
            ax.axvline(summary["control"][panel][name]["mean_cost"], color=color,
                       linestyle="--", linewidth=1.4, label=label)
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("Episode cost (lower is better)")
        ax.set_yticks(range(6), labels)
        ax.grid(axis="x", alpha=.15)
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(.98, .91), ncol=2, fontsize=8)
    gate = "PASS" if summary["continuation_gate"]["passed"] else "FAIL"
    fig.suptitle("Does adaptive search improve actual control?", x=.02, ha="left", y=.98,
                 fontsize=19, fontweight="bold")
    fig.text(.02, .915, f"Frozen CEM256 versus RS256 continuation rule: {gate}. All six inherited fits are shown.", fontsize=11)
    fig.text(.02, .035, "Each dot is one fit averaged over 64 paired episodes; lines span fits, not confidence intervals.\n"
             "RS64 scores 64 sequences per decision. RS256 and CEM256 each score 256. Physics uses supplied dynamics and 64 sequences.",
             fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=.24, right=.985, top=.82, bottom=.19, wspace=.20)
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
                ax.scatter(x, y, color=color, marker=MARKERS[p], s=45,
                           label=f"{label} / {planner.upper()}")
        ax.axhline(summary["control"][panel]["zero"]["mean_cost"], color="#a55a4c",
                   linestyle="--", linewidth=1.2, label="Zero command")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Decision time per case (ms, amortized)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Episode cost (lower is better)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .91), ncol=4, fontsize=8)
    fig.suptitle("Measured control versus computation", x=.025, ha="left", fontsize=18, fontweight="bold")
    fig.text(.025, .035, "All seeds; descriptive timings on a shared CPU host. Decision time includes search and trace storage.\n"
             "Environment stepping, inherited training, shared setup and separate diagnostic costs are reported separately.", fontsize=10, linespacing=1.6)
    fig.tight_layout(rect=(0, .13, 1, .80))
    save(fig, out / "search-compute.png")

    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.5))
    fields = (("native_selected_return_mean", "Native selected return\n(higher is better)"),
              ("finite_set_regret_mean", "Regret in evaluated finite set\n(lower is better)"),
              ("selected_raw_prediction_bias_mean", "Raw predicted minus native return\n(positive means optimistic)"))
    for r, panel in enumerate(PANELS[1:]):
        for c, (field, title) in enumerate(fields):
            ax = axes[r, c]
            for kind, label, color in KINDS:
                for p, planner in enumerate(PLANNERS):
                    values = [summary["diagnostics"]["aggregate"][panel][f"{kind}-{seed}-{planner}"][field]
                              for seed in SEEDS]
                    shift = -.10 if kind == "free" else .10
                    ax.scatter(np.full(3, p + shift) + [-.025, 0, .025], values,
                               color=color, marker=MARKERS[p], s=32,
                               label=label if p == 0 else None)
                    ax.plot([p + shift - .06, p + shift + .06], [np.mean(values)] * 2,
                            color=color, linewidth=2)
            ax.set_xticks(range(3), [p.upper() for p in PLANNERS])
            ax.set_title(title, fontsize=11)
            ax.grid(axis="y", alpha=.15)
            if c == 0:
                ax.set_ylabel(PANEL_LABELS[r + 1])
            if c == 2:
                ax.axhline(0, color="#66757d", linewidth=.8)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Check model optimism at the same physical roots", x=.025, ha="left", fontsize=18, fontweight="bold")
    fig.text(.025, .025, "Each dot: one fit averaged over 64 fixed roots from 16 independent physical episodes, with four shared noise branches.\n"
             "Native outcomes do not select actions. Regret compares 118 evaluated identity slots, not an optimal controller. No new training.",
             fontsize=10, linespacing=1.6)
    fig.tight_layout(rect=(0, .11, 1, .925), h_pad=2.2)
    save(fig, out / "search-model-error.png")
    receipt = {"status": "completed", "version": "reacher-search-v1-figures",
               "audit_receipt_sha256": expected, "source_sha256": sha(Path(__file__)),
               "saved_output_only": True, "new_model_calls": 0,
               "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.iterdir()) if p.is_file()},
               "limits": "Descriptive plots over the six inherited fits. No fit selection, additional model calls or new bootstrap samples."}
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
