"""Plot the complete saved policy-value screen; no policy or simulator calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("exit", "full", "recent128", "latest", "quality", "privileged")
LABELS = ("Exit immediately", "Full public history", "Recent 128 steps", "Latest check / rock",
          "Quality only", "True map + qualities*")
COLORS = ("#858d96", "#1c6b48", "#527bba", "#ab6a1c", "#8b5d9c", "#53616b")


def main(args):
    raw = args.summary.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.summary_sha256:
        raise ValueError("externally pinned summary changed")
    summary = json.loads(raw)
    maps = [m["map_seed"] for m in summary["by_map"]]
    if summary["episodes"] != 192 or len(maps) != 8 or set(maps) != set(range(12001, 12009)):
        raise ValueError("figure requires all eight maps and 192 episodes")
    expected_conditions = {f"{kind}_vs_{control}" for kind in ("raw_gain", "positive_maps")
                           for control in ("recent128", "latest", "quality")}
    expected_conditions.update(("full_gain_vs_exit", "privileged_gain_vs_exit", "finite_particle_adequacy", "complete_cohort"))
    criteria = summary["criteria"]
    if (len(criteria) != 10 or {c["name"] for c in criteria} != expected_conditions
            or any(type(c["passes"]) is not bool for c in criteria)
            or summary["learned_pilot_admitted"] != all(c["passes"] for c in criteria)):
        raise ValueError("inconsistent continuation decision")
    for arm in ARMS:
        for metric in ("raw_return", "controller_seconds"):
            samples = [m["scores"][arm][metric] for m in summary["by_map"]]
            mean = summary["scores_equal_map_means"][arm][metric]
            if not np.isfinite([*samples, mean]).all() or not np.isclose(np.mean(samples), mean, atol=1e-12, rtol=0):
                raise ValueError("invalid plotted values or aggregate")
            if metric == "controller_seconds" and min(*samples, mean) <= 0:
                raise ValueError("log-axis times must be strictly positive")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    values = {"summary_sha256": hashlib.sha256(raw).hexdigest(), "arms": {}, "paired_map_gains": {}}
    for ax, metric, title in zip(axes[:2], ("raw_return", "controller_seconds"),
                                 ("Native task return", "Measured controller time"), strict=True):
        for index, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS, strict=True)):
            per_map = [m["scores"][arm][metric] for m in summary["by_map"]]
            mean = summary["scores_equal_map_means"][arm][metric]
            ax.scatter(per_map, index + np.linspace(-.16, .16, 8), color=color, s=29, alpha=.7)
            ax.scatter([mean], [index], color=color, marker="|", s=500, linewidths=2.7)
            values["arms"].setdefault(arm, {})[metric] = {"per_map": per_map, "equal_map_mean": mean}
        ax.set_yticks(range(6), LABELS if ax is axes[0] else [""] * 6)
        ax.set_ylim(5.65, -.65)
        ax.grid(axis="x", alpha=.18)
        ax.set_title(title, loc="left", weight="bold", pad=13)
    axes[0].axvline(10, color="#777777", linestyle=":", alpha=.5, linewidth=1)
    axes[0].set_xlabel("Reward per episode (higher is better)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Seconds per episode (log scale)")
    controls = ("recent128", "latest", "quality")
    for index, control in enumerate(controls):
        per_map = [m["scores"]["full"]["raw_return"] - m["scores"][control]["raw_return"]
                   for m in summary["by_map"]]
        values["paired_map_gains"][control] = per_map
        axes[2].scatter(per_map, index + np.linspace(-.16, .16, 8), color=COLORS[index + 2], s=30, alpha=.7)
        axes[2].scatter([np.mean(per_map)], [index], color=COLORS[index + 2], marker="|", s=500, linewidths=2.7)
    axes[2].axvline(0, color="#777777", linewidth=1)
    axes[2].axvline(5, color="#1c6b48", linestyle="--", linewidth=1, label="Required mean gain: +5")
    axes[2].set_yticks(range(3), ("vs Recent 128", "vs Latest", "vs Quality"))
    axes[2].set_ylim(2.65, -.65)
    axes[2].set_title("Full-history reward difference", loc="left", weight="bold", pad=13)
    axes[2].set_xlabel("Paired difference per map")
    axes[2].grid(axis="x", alpha=.18)
    axes[2].legend(loc="lower left", fontsize=9, frameon=False)
    passed = sum(c["passes"] for c in summary["criteria"])
    decision = "PASS" if summary["learned_pilot_admitted"] else "FAIL"
    figure.suptitle("RockSample: does older public information improve control?", x=.025, ha="left", y=.98,
                    fontsize=19, weight="bold")
    figure.text(.025, .902, f"192 episodes | 8 maps x 4 resets x 6 controllers | Continuation {decision}: {passed}/10 conditions",
                fontsize=12, color="#424b53")
    figure.text(.025, .055, "Each dot is one map's four-reset mean; bars mark equal-map means. No confidence intervals shown.\n"
                "*Privileged reference sees the true map and qualities. Controller time includes filtering and planning; excludes simulator and trace I/O.\n"
                "Classical filters and a fixed planner only. This experiment trains no architecture and measures no biological-wiring advantage.",
                fontsize=9, color="#424b53", linespacing=1.5)
    figure.subplots_adjust(left=.16, right=.985, top=.79, bottom=.24, wspace=.47)
    for suffix in ("png", "pdf"):
        figure.savefig(args.output / f"policy-value.{suffix}", dpi=180, facecolor="white")
    plt.close(figure)
    (args.output / "plotted-values.json").write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
