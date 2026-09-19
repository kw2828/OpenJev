"""Render the offline dynamics pilot using saved measurements only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"transported_delta": "Action transport", "decay_delta": "Scalar decay",
          "gru": "GRU", "history16": "History16 GRU", "diagonal_filter": "Diagonal filter",
          "ridge16": "Linear ridge16", "hold_last": "Hold last"}
COLORS = ["#116c9c", "#64a6ba", "#d77b22", "#e5b067", "#755ba1", "#32865b", "#656b73"]


def plot(run, out):
    data = json.loads((run / "completed.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    modes = list(LABELS)
    grouped = {m: [fit for fit in data["fits"] if fit["mode"] == m] for m in modes[:5]}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("OpenJev: short-horizon robot-dynamics pilot", fontsize=18, weight="bold", y=1.01)
    for i, (mode, color) in enumerate(zip(modes, COLORS, strict=True)):
        values = ([fit["test"]["mse"] for fit in grouped[mode]] if mode in grouped
                  else [data["references"][mode]["mse"]])
        mean = float(np.mean(values))
        axes[0].scatter(values, np.full(len(values), i), color=color, s=45, zorder=3)
        axes[0].plot([mean, mean], [i - .2, i + .2], color=color, linewidth=3)
        axes[0].annotate(f" {mean:.4g}", (max(values), i), xytext=(8, 0), textcoords="offset points",
                         va="center", fontsize=10, color=color)
        curve = (np.mean([fit["test"]["horizon_mse"] for fit in grouped[mode]], axis=0) if mode in grouped
                 else data["references"][mode]["horizon_mse"])
        axes[1].plot(np.arange(1, 11) * 2, curve, color=color, label=LABELS[mode], linewidth=2)
    axes[0].set_yticks(range(len(modes)), [LABELS[m] for m in modes])
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("10-step normalized MSE (log scale, lower is better)")
    axes[0].set_title("All 15 fits and both simple references")
    axes[0].grid(axis="x", alpha=.15)
    low, high = axes[0].get_xlim()
    axes[0].set_xlim(low, high * 4)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Forecast time (nominal ms)")
    axes[1].set_ylabel("Normalized MSE (log scale)")
    axes[1].set_title("Error at each forecast horizon")
    axes[1].legend(fontsize=8, loc="best", frameon=False)
    axes[1].grid(alpha=.15)
    fig.text(.01, -.025, "Simulated HiP-RSSM data; 11 held-out trajectories, 176 windows. Dots = paired training seeds.\n"
             "Recorded torques condition forecasts. Offline prediction only; no robot-control or architecture-novelty claim.",
             fontsize=10, color="#555555")
    fig.tight_layout()
    fig.savefig(out / "forecast-results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    receipt = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_sha256": hashlib.sha256((run / "completed.json").read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256((out / "forecast-results.png").read_bytes()).hexdigest(),
               "new_model_calls": 0, "new_training_calls": 0}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.out)
