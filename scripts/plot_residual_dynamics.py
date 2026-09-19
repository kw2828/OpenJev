"""Plot every saved residual-dynamics configuration without executing models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

LABELS = {"latent": "Latent RLS", "public": "Public-feature RLS", "bias": "Bias correction",
          "none": "Residual GRU", "history16": "History16 GRU", "ridge16": "Direct ridge16",
          "hold_last": "Hold last"}
COLORS = ["#106c9e", "#755ba1", "#47a3a8", "#d67a21", "#dbad6b", "#378659", "#6d737c"]


def plot(run, output):
    output.mkdir(parents=True, exist_ok=False)
    saved = json.loads((run / "completed.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, panel, label in zip(axes, ("test_sin", "test_zigzag"), ("Plain sin_infer archive", "Zigzag archive"), strict=True):
        for i, (method, color) in enumerate(zip(LABELS, COLORS, strict=True)):
            rows = [r for r in saved["evaluations"] if r["variant"] == method and r["panel"] == panel]
            values = [r["test"]["mse"] for r in rows] if rows else [saved["references"][panel][method]["mse"]]
            mean = float(np.mean(values))
            ax.scatter(values, [i] * len(values), color=color, s=45)
            ax.plot([mean, mean], [i - .2, i + .2], color=color, linewidth=3)
            ax.annotate(f" {mean:.4g}", (max(values), i), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=10, color=color)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Normalized forecast MSE (log scale, lower is better)")
        ax.grid(axis="x", alpha=.15)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 1.4)
        if panel == "test_sin":
            ax.set_xticks([5, 10, 20, 30], ["5", "10", "20", "30"])
        else:
            ax.set_xticks([160000, 200000, 250000, 320000], ["160k", "200k", "250k", "320k"])
    axes[0].set_yticks(range(len(LABELS)), list(LABELS.values()))
    axes[0].invert_yaxis()
    fig.suptitle("OpenJev: 500 ms robot forecasts", fontsize=19, weight="bold", y=1.02)
    fig.text(.01, .01, "Same three full-context backbones for all RLS corrections; three separately trained history controls.\n"
             "160 windows per archive, 10 parent trajectories each. Simulated data with recorded future applied torques.\n"
             "Dots are fits; bars are means. This measures offline prediction, not closed-loop robot control.", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, .15, 1, .98))
    fig.savefig(output / "forecast-results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (variant, color) in enumerate(zip(list(LABELS)[:5], COLORS[:5], strict=True)):
        rows = [r for r in saved["evaluations"] if r["variant"] == variant]
        samples = [v for r in rows for v in r["window_latency_ms"]["samples"]]
        median, p95 = np.median(samples), np.percentile(samples, 95)
        ax.barh(i, median, color=color, height=.55)
        ax.plot([median, p95], [i, i], color=color, linewidth=3)
        ax.scatter([p95], [i], color=color, marker="|")
        ax.annotate(f" {median:.2f} / {p95:.2f}", (p95, i), xytext=(5, 0), textcoords="offset points", va="center")
    ax.set_yticks(range(5), list(LABELS.values())[:5])
    ax.invert_yaxis()
    ax.set_xlabel("Complete-window prediction, ms (median / p95)")
    ax.set_title("CPU cost includes context reconstruction and online updates")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.3)
    fig.text(.01, .01, "300 measured samples per configuration; one CPU thread. Excludes loading, normalization and service overhead.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "prediction-cost.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    receipt = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((run / "completed.json").read_bytes()).hexdigest(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.png"))},
        "new_model_calls": 0, "new_training_calls": 0}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plot(a.run, a.output)
