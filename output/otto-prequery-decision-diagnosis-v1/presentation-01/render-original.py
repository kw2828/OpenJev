"""Render the closed saved-output diagnosis without loading prediction arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("pinned closed summary")
    data = json.loads(raw)
    args.output.mkdir(exist_ok=False)
    colors = ["#ba5a1f", "#2274a5", "#252525"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    plotted = []
    for row, architecture in enumerate(("innovation", "innovation_gru")):
        for col, regime in enumerate(("lambda3", "lambda4")):
            ax = axes[row, col]
            values = [x for x in data["means"] if x["architecture"] == architecture and x["group"] == regime]
            if len(values) != 1:
                raise ValueError("one complete three-seed aggregate")
            item = values[0]
            means = [100 * item["initial_agreement_contribution"], 100 * item["post_agreement_contribution"],
                     100 * (item["candidate_full_agreement"] - item["baseline_full_agreement"])]
            if abs(means[0] + means[1] - means[2]) > 1e-10:
                raise ValueError("additive full-denominator contributions")
            ax.bar(range(3), means, color=colors, width=.62, alpha=.76)
            for seed, marker, shift in zip((275000001, 275000002, 275000003), ("o", "s", "^"), (-.14, 0., .14), strict=True):
                record = next(x for x in data["paired"] if x["architecture"] == architecture
                              and x["group"] == regime and x["seed"] == seed)
                points = [100 * record["initial_agreement_contribution"],
                          100 * record["post_agreement_contribution"],
                          100 * (record["candidate_full_agreement"] - record["baseline_full_agreement"])]
                ax.scatter([x + shift for x in range(3)], points, color="black", marker=marker, s=32, zorder=4)
            for x, value in enumerate(means):
                ax.annotate(f"{value:+.2f}", (x, value), xytext=(0, 9 if value >= 0 else -17),
                            textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
            name = "Explicit correction" if architecture == "innovation" else "Ordinary error-fed GRU"
            ax.set_title(name + " | " + regime.replace("lambda", "λ"), fontsize=12)
            ax.set_xticks(range(3), ["Initial steps 1-3", "Later steps ≥5", "Full trajectory"])
            ax.axhline(0, color="#777777", linewidth=.8)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel("Agreement change (percentage points)")
            plotted.append({"architecture": architecture, "regime": regime, "means_pp": means})
    fig.suptitle("Extra prior supervision helps later decisions, hurts initial decisions", fontsize=17, y=.98)
    fig.text(.5, .927, "Auxiliary minus original MSE | Initial + later contribution = full change", ha="center", fontsize=11)
    fig.legend([Line2D([], [], color="black", marker=m, linestyle="none") for m in ("o", "s", "^")],
               ["Seed 275000001", "Seed 275000002", "Seed 275000003"], loc="lower center",
               bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
    fig.text(.5, .029, "Bars average all three paired fits. Contributions retain each full episode denominator, including empty episodes.",
             ha="center", fontsize=9)
    fig.text(.5, .008, "Retrospective diagnosis on exposed recorded paths. Original study remains FAIL 51/55; no new training or autonomous evaluation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .91))
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"prequery-decision-decomposition.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in args.output.iterdir()}
    record = {"status": "completed", "scope": "saved-summary rendering only", "input_sha256": args.sha256,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "files": outputs, "plotted": plotted}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
