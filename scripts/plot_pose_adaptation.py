"""Render sealed pose-adaptation measurements without running any predictors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

NAMES = {"meta": "Adaptation-trained", "meta_prior": "Same prior only", "static": "Static-trained",
         "static_adapt": "Static + adaptation", "public": "Public-feature ridge", "gru": "GRU (CV1)",
         "ridge16": "Torque ridge16", "hold": "Hold last",
         "cv1": "Velocity1", "cv16": "Velocity16", "ls16": "Least-squares16", "body16": "Body twist16"}
ORDER = tuple(NAMES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(completion, out):
    data = json.loads(completion.read_text())
    if data["status"] != "completed":
        raise ValueError("completed run required")
    for rel, binding in data["files"].items():
        if sha(completion.parent / rel) != binding["sha256"]:
            raise ValueError("changed input: " + rel)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#086c94", "#6b94aa", "#c16b19", "#91633e", "#9767b4", "#bd8b49"] + ["#477d66"] * 6
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for col, panel in enumerate(("test_sin", "test_zigzag")):
        for row, metric in enumerate(("position_rmse_m", "rotation_rmse_rad")):
            ax = axes[row, col]
            for y, variant in enumerate(ORDER):
                scores = np.array([x["metrics"][metric] for x in data["rows"]
                                   if x["panel"] == panel and x["variant"] == variant])
                pooled = float(np.sqrt(np.mean(scores ** 2)))
                ax.scatter(scores, np.full(len(scores), y), color=colors[y], s=22)
                ax.scatter(pooled, y, color=colors[y], marker="|", s=150)
            ax.set_yticks(np.arange(len(ORDER)), [NAMES[x] for x in ORDER])
            ax.set_xscale("log")
            low, high = ax.get_xlim()
            ticks = np.geomspace(low, high, 4)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.2g}" for x in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            ax.set_xlabel(("Position RMSE, meters" if row == 0 else "Rotation RMSE, radians")
                          + " (log scale; lower is better)")
            if row == 0:
                ax.set_title("Plain archive" if col == 0 else "Zigzag archive")
    axes[0, 0].invert_yaxis()
    fig.suptitle("OpenJev: learning to adapt pose dynamics", fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.text(.015, .02, "Exposed development data, 500 ms simulated-robot forecasts. Dots: individual fits; bars: pooled RMSE.\n"
                       "Three fits per neural model; fixed references shown once. Physical endpoints remain separate.", fontsize=10)
    fig.savefig(out / "physical-errors.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    for y, variant in enumerate(ORDER):
        values = np.concatenate([x["latency_ms"] for x in data["rows"] if x["variant"] == variant])
        med, p95 = np.percentile(values, [50, 95])
        ax.scatter(med, y, color=colors[y], s=45)
        ax.plot([med, p95], [y, y], color=colors[y], linewidth=3)
        ax.text(p95 * 1.1, y, f"{med:.2f} / {p95:.2f}", va="center", fontsize=10)
    ax.set_yticks(np.arange(len(ORDER)), [NAMES[x] for x in ORDER])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.5)
    ax.set_xlabel("Complete-window prediction, ms (log scale; median / p95)")
    ax.set_title("CPU cost includes context and all 25 forecast steps")
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.text(.015, .02, "One CPU thread; 120 timing samples per neural family, 40 per reference.\n"
                       "Excludes loading, normalization and service overhead.", fontsize=10)
    fig.savefig(out / "prediction-cost.png", dpi=160)
    plt.close(fig)
    receipt = {"source_sha256": sha(__file__), "completion_sha256": sha(completion),
               "files": {p.name: sha(p) for p in sorted(out.glob('*.png'))},
               "new_model_calls": 0, "new_training_calls": 0}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.completion, args.out)
