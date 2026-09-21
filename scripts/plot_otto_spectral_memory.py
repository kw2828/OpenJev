"""Plot every fixed spectral arm from one hash-pinned saved summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(args):
    raw = args.summary.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.summary_sha256:
        raise ValueError("Summary hash mismatch")
    summary = json.loads(raw)
    arms = summary["configuration"]["arms"]
    if len(arms) != 12 or summary["progress"]["prefixes"] != 2164:
        raise ValueError("Incomplete twelve-arm study")
    if summary["original_learned_pilot_opportunity"] is not False:
        raise ValueError("Original continuation decision must remain unchanged")
    labels = [arm.replace("full_bayes", "Full Bayes").replace("exact_log", "Exact log")
              .replace("dct", "DCT ").replace("recent32_hard", "Recent 32 + hard")
              .replace("recent32", "Recent 32").replace("_neutral", " / zero")
              .replace("_nearest", " / near") for arm in arms]
    colors = ["#487caa" if "neutral" in arm else "#d29138" if "nearest" in arm
              else "#546b67" if arm in ("full_bayes", "exact_log") else "#85808f" for arm in arms]
    all_means = summary["all_prefixes"]["mixture_weighted_prefix_means"]
    case_means = summary["all_prefixes"]["mixture_weighted_case_means"]
    late = summary["after32_prefixes"]["mixture_weighted_prefix_means"]
    storage = summary["storage_by_arm"]
    state = [storage[a].get("state_array_bytes", storage[a].get("mutable_array_bytes", 0)) / 1024 for a in arms]
    immutable = [storage[a].get("immutable_array_bytes", 0) / 1024 for a in arms]
    duration = summary["mixture_weighted_per_case_timing_seconds"]
    stages = ("initialization_seconds", "update_seconds", "decode_seconds", "planner_seconds")
    values = {
        "all_agreement_percent": [all_means[f"{a}.matches_full_action"] * 100 for a in arms],
        "all_case_agreement_percent": [case_means[f"{a}.matches_full_action"] * 100 for a in arms],
        "after32_agreement_percent": [late[f"{a}.matches_full_action"] * 100 for a in arms],
        "after32_tv": [late[f"{a}.tv_to_full"] for a in arms],
        "after32_objective_excess": [late[f"{a}.full_objective_excess"] for a in arms],
        "state_kib": state, "local_immutable_kib": immutable,
        "timing_ms_per_weighted_case": {stage: [duration[a][stage] * 1000 for a in arms] for stage in stages},
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(left=.105, right=.98, top=.81, bottom=.18, wspace=.62, hspace=.35)
    fig.text(.045, .96, "Can compact additive memory preserve older odor evidence?", fontsize=21, weight="bold")
    fig.text(.045, .92, "Fixed DCT compression | 96 saved trajectories | 2,164 decisions | All 12 arms shown", fontsize=12)
    fig.text(.045, .887, "After 32: 538 decisions in 23 cases. No new gameplay, training or architecture advantage.", fontsize=11)
    specifications = [
        ("all_agreement_percent", "All decisions: weighting sensitivity", "% matching the full-history action"),
        ("after32_agreement_percent", "After 32: action agreement", "% matching the full-history action"),
        ("after32_tv", "After 32: belief distortion", "Total variation from full history"),
        ("after32_objective_excess", "After 32: decision cost proxy", "Full-belief heuristic excess, lower is better"),
    ]
    y = np.arange(len(arms))
    for ax, (key, title, xlabel) in zip(axes.flat, specifications, strict=False):
        ax.barh(y, values[key], color=colors, height=.65)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        if "percent" in key:
            ax.set_xlim(0, 106)
        elif key == "after32_tv":
            ax.set_xlim(0, 1)
    axes[0, 0].scatter(values["all_case_agreement_percent"], y, marker="D", s=19,
                      color="#222222", zorder=3, label="Case means (bars: prefix means)")
    axes[0, 0].set_title("All decisions: weighting sensitivity", loc="left", weight="bold", pad=28)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1), fontsize=7, frameon=False)
    ax = axes[1, 1]
    ax.barh(y, state, color=colors, height=.65, label="Evolving state and mask")
    ax.barh(y, immutable, left=state, color="#cad0d4", height=.65, label="Local immutable prior")
    ax.set_title("Retained per-actor array payload", loc="left", weight="bold", pad=28)
    ax.set_xlabel("KiB; shared arrays and temporary workspaces excluded")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=7, frameon=False)
    ax = axes[1, 2]
    bottom = np.zeros(len(arms))
    for stage, color in zip(stages, ("#a5abb0", "#9b6fb0", "#dfad56", "#497d88"), strict=True):
        vals = np.array(values["timing_ms_per_weighted_case"][stage])
        ax.barh(y, vals, left=bottom, height=.65, color=color, label=stage.removesuffix("_seconds"))
        bottom += vals
    ax.set_title("Complete controller computation", loc="left", weight="bold", pad=28)
    ax.set_xlabel("ms per mixture-weighted trajectory, one rotated pass")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=4, fontsize=7, frameon=False)
    for ax in axes.flat:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.17)
        ax.set_axisbelow(True)
    shared = summary["shared_model"]["shared_array_bytes"] / 1024
    planner = summary["additional_readonly_planner_kernel_bytes"] / 1024
    lines = [
        "Quality bars use mixture-weighted prefixes. Diamonds use mixture-weighted case means; that all-case summary favors recent-32 over DCT 16.",
        "DCT zero/near are both predeclared finite fills for excluded cells. Every spectral arm also stores a persistent 2,809-byte hard-support mask.",
        f"Memory excludes Python metadata. Spectral shared model: {shared:.1f} KiB; additional planner kernel: {planner:.1f} KiB. Dense decoding/planning workspace remains.",
        "Exact log and both DCT 53 arms are numerical controls. Timing includes initialization, updates, decoding and planning, not whole-study overhead.",
        "Recorded full-policy paths only; proposed actions were never executed. No efficacy threshold or reversal of the original continuation failure.",
    ]
    for i, line in enumerate(lines):
        fig.text(.045, .122 - .021 * i, line, fontsize=9, color="#475569")
    fig.savefig(output / "spectral-memory.png", dpi=180)
    fig.savefig(output / "spectral-memory.pdf")
    plt.close(fig)
    (output / "plotted-values.json").write_text(json.dumps({"summary_sha256": args.summary_sha256,
        "arms": arms, "labels": labels, **values}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args())
