"""Render paired-outcome concentration and saved-prefix evidence ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ("recent32", "old_positive", "old_zero", "full")
LABELS = ("Recent 32", "+ old detections", "+ old zero readings", "Full reference")


def read_pinned(path, pin):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError(f"Input hash mismatch: {path}")
    return json.loads(raw)


def plot(args):
    paired = read_pinned(args.paired, args.paired_sha256)
    diagnostic = read_pinned(args.diagnostic, args.diagnostic_sha256)
    if paired["original_continuation"]["learned_pilot_opportunity"] is not False:
        raise ValueError("Original failure must remain unchanged")
    if diagnostic["original_learned_pilot_opportunity"] is not False:
        raise ValueError("Diagnostic cannot reverse original decision")
    rows = paired["pairs"]
    if [row["seed"] for row in rows] != list(range(610001, 610097)):
        raise ValueError("Every original pair must be present in order")
    contributions = np.array([row["weighted_contribution"] for row in rows])
    if abs(float(contributions.sum()) - paired["weighted_mean_gain"]) > 1e-10:
        raise ValueError("Contribution sum mismatch")
    populations = [diagnostic["first_disagreement"], diagnostic["all_full_path_prefixes"]]
    values = [
        [population[metric][f"{mode}.matches_full_action"] * 100 for mode in MODES]
        for population, metric in zip(populations, (
            "stratum_weighted_eligible_case_means", "stratum_weighted_eligible_prefix_means"), strict=True)
    ]
    if not np.isfinite(values).all() or np.any(np.array(values) < 0) or np.any(np.array(values) > 100 + 1e-10):
        raise ValueError("Invalid action agreement")
    top3 = next(row for row in paired["concentration"] if row["k"] == 3)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.7), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    fig.subplots_adjust(left=.055, right=.98, top=.73, bottom=.29, wspace=.48)
    fig.text(.055, .94, "Which older odor observations change decisions?", fontsize=20, weight="bold")
    fig.text(.055, .875, "Post hoc saved-data analysis | No new gameplay or trained model | Original continuation still fails: 9/10 passed", fontsize=11)
    ax = axes[0]
    ax.bar(np.arange(1, 97), contributions, color=["#177568" if value >= 0 else "#b14f50" for value in contributions], width=.9)
    ax.axhline(0, color="#555555", linewidth=.7)
    for seed in top3["seeds"]:
        case = seed - 610000
        value = contributions[case - 1]
        ax.annotate(str(case), (case, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("Benefit concentrates in a few cases", loc="left", fontsize=12, weight="bold", pad=18)
    ax.set_xlabel("All 96 paired cases, original order")
    ax.set_ylabel("Contribution to mean moves saved")
    ax.set_xlim(0, 97)
    ax.set_ylim(min(-1.6, contributions.min() * 1.3), contributions.max() * 1.28)
    ax.text(.02, .97, f"Top 3: {top3['fraction_of_net_gain'] * 100:.1f}% of net gain", transform=ax.transAxes, va="top", fontsize=10)
    colors = ["#767676", "#4580b6", "#b07622", "#177568"]
    for index, ax in enumerate(axes[1:]):
        y = np.arange(4)
        ax.barh(y, values[index], color=colors, height=.55)
        ax.set_yticks(y, LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 117)
        ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
        ax.set_xlabel("Chosen action matches full history")
        for j, value in enumerate(values[index]):
            ax.text(value + 2, j, f"{value:.1f}%", va="center", fontsize=9)
        title = (f"First shared-history divergence\n{populations[0]['eligible_cases']} cases" if index == 0 else
                 f"Later prefixes on the full-policy path\n{populations[1]['eligible_prefixes']} decisions, {populations[1]['eligible_cases']} cases")
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=18)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.18)
    later = populations[1]["stratum_weighted_eligible_prefix_means"]
    foot = [
        "Left: all paired outcomes; top-three contribution ranking is post hoc. Positive values favor full history.",
        "Middle: mixture-weighted cases at their first differing action. Recent-32 disagreement there is true by construction.",
        "Right: mixture-weighted eligible prefixes after more than 32 observations, on the recorded full-history path.",
        "All reconstructions retain the initial hit and permanent visitation ledger. Matching full history is not proof of better autonomous search.",
        f"Old-zero retention raises agreement, but full-belief heuristic excess rises: {later['recent32.full_objective_excess']:.5f} to {later['old_zero.full_objective_excess']:.5f}.",
    ]
    for i, line in enumerate(foot):
        fig.text(.055, .19 - i * .034, line, fontsize=9, color="#4b5563")
    fig.savefig(output / "memory-evidence.png", dpi=180)
    fig.savefig(output / "memory-evidence.pdf")
    plt.close(fig)
    plotted = {"paired_sha256": args.paired_sha256, "diagnostic_sha256": args.diagnostic_sha256,
               "case_contributions": contributions.tolist(), "modes": MODES, "labels": LABELS,
               "first_divergence_agreement_percent": values[0], "full_path_agreement_percent": values[1],
               "top_three": top3, "scope": "descriptive and teacher-forced; original continuation failure unchanged"}
    (output / "plotted-values.json").write_text(json.dumps(plotted, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--paired-sha256", required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--diagnostic-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args())
