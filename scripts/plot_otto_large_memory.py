"""Render all saved larger-setting OTTO arms and blocks without new policy/environment calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("space_full", "space_recent32", "space_recent8", "space_initial",
        "info_full", "info_recent32", "info_recent8", "info_initial")
LABELS = ("Space-aware: full", "Space-aware: recent 32", "Space-aware: recent 8", "Space-aware: initial + visits",
          "Infotaxis: full", "Infotaxis: recent 32", "Infotaxis: recent 8", "Infotaxis: initial + visits")


def main(args):
    raw = args.summary.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.summary_sha256:
        raise ValueError("externally pinned summary changed")
    s = json.loads(raw)
    if s["episodes"] != 768 or [b["block"] for b in s["blocks"]] != list(range(8)) or len(s["criteria"]) != 10:
        raise ValueError("incomplete experiment")
    if all(c["passes"] for c in s["criteria"]) != s["learned_pilot_opportunity"]:
        raise ValueError("inconsistent continuation decision")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    colors = ["#136f63"] * 4 + ["#426bb0"] * 4
    plotted = {"summary_sha256": args.summary_sha256, "arms": {}, "paired_block_differences": {}}
    for ax, metric, title, xlabel, scale in zip(axes[:2], ("capped_time", "controller_seconds"),
            ("Search time", "Measured controller computation"),
            ("Moves to source, capped at 2188", "Milliseconds per episode"), (1, 1000), strict=True):
        for i, arm in enumerate(ARMS):
            values = [b["means"][arm][metric] for b in s["blocks"]]
            mean = s["means"][arm][metric]
            if not np.isfinite([*values, mean]).all() or abs(np.mean(values) - mean) > 1e-10:
                raise ValueError("invalid plotted mean")
            ax.scatter(np.array(values) * scale, i + np.linspace(-.15, .15, 8), color=colors[i], alpha=.65, s=30)
            ax.scatter([mean * scale], [i], marker="|", color=colors[i], s=500, linewidths=2.5)
            plotted["arms"].setdefault(arm, {})[metric] = {"blocks": values, "mean": mean}
        ax.set_yticks(range(8), LABELS if ax is axes[0] else [""] * 8)
        ax.set_ylim(7.65, -.65)
        ax.set_title(title, loc="left", weight="bold", pad=14)
        ax.set_xlabel(xlabel + " (lower is better)")
        ax.grid(axis="x", alpha=.18)
    for i, control in enumerate(("space_recent32", "space_recent8", "space_initial")):
        differences = [b["means"][control]["capped_time"] - b["means"]["space_full"]["capped_time"] for b in s["blocks"]]
        plotted["paired_block_differences"][control] = differences
        axes[2].scatter(differences, i + np.linspace(-.15, .15, 8), color="#136f63", alpha=.65, s=30)
        axes[2].scatter([np.mean(differences)], [i], marker="|", color="#136f63", s=500, linewidths=2.5)
    axes[2].axvline(0, color="#6b7280", linewidth=1)
    axes[2].axvline(2, color="#aa6924", linestyle="--", linewidth=1, label="Recent-32 required gain: 2 moves")
    axes[2].set_yticks(range(3), ("vs Recent 32", "vs Recent 8", "vs Initial + visits"))
    axes[2].set_ylim(2.55, -.65)
    axes[2].set_title("Full-history time saved", loc="left", weight="bold", pad=14)
    axes[2].set_xlabel("Paired moves saved (higher is better)")
    axes[2].legend(loc="lower left", frameon=False, fontsize=9)
    axes[2].grid(axis="x", alpha=.18)
    decision = "PASS" if s["learned_pilot_opportunity"] else "FAIL"
    fig.suptitle("Olfactory search: testing longer memory in the 53x53 setting", x=.025, ha="left", y=.98,
                 weight="bold", fontsize=19)
    fig.text(.025, .90, f"768 sampled-source searches | 96 paired cases | 8 controllers | Continuation {decision}: {sum(c['passes'] for c in s['criteria'])}/10", fontsize=12)
    fig.text(.025, .055,
             "Dots: eight fixed blocks; bars: means. Three initial-hit strata weighted by the upstream mixture. No confidence intervals shown.\n"
             "All controllers retain the initial hit and visited-cell ledger. Controller cost includes actor initialization, filtering and planning.\n"
             "Classical references only; no trained model or biological-wiring result. This is not the official survival-weighted evaluator.",
             fontsize=9, color="#424b53", linespacing=1.5)
    fig.subplots_adjust(left=.205, right=.985, top=.79, bottom=.24, wspace=.47)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"otto-large-memory.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    (args.output / "plotted-values.json").write_text(json.dumps(plotted, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
