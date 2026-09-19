"""Plot sealed support-weighting forecasts; no model calls or training."""
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

NAMES = {
    "prior": "Same prior only", "full": "Full 30 supports",
    "recent5": "Recent 5", "recent5_mass": "Recent 5: uniform mass",
    "decay5": "Exponential recency", "decay5_mass": "Recency: uniform mass",
    "huber3": "Robust residuals", "huber3_mass": "Robust: uniform mass",
    "decay_huber3": "Recency + robust", "decay_huber3_mass": "Combined: uniform mass",
    "static": "Static-trained", "static_adapt": "Static + adaptation",
    "public": "Public-feature ridge", "gru": "GRU (CV1)",
    "hold": "Hold last", "cv1": "Velocity1", "cv16": "Velocity16",
    "ls16": "Least-squares16", "body16": "Body twist16", "ridge16": "Torque ridge16",
}
PANELS = ("test_sin", "test_zigzag")
SEEDS = (1101, 1202, 1303)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(completion, out):
    data = json.loads(completion.read_text())
    if data["status"] != "completed" or len(data["rows"]) != 96:
        raise ValueError("completed96-row run required")
    for rel, binding in data["files"].items():
        if sha(completion.parent / rel) != binding["sha256"]:
            raise ValueError("changed input: " + rel)
    expected = {(p, v, s) for p in PANELS for i, v in enumerate(NAMES)
                for s in (SEEDS if i < 14 else (None,))}
    if {(r["panel"], r["variant"], r["seed"]) for r in data["rows"]} != expected:
        raise ValueError("row identities")
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#48556a", "#95631b", "#326e9e", "#9bb4c7", "#426bb2", "#a3b4cd",
              "#946ea7", "#c4b0ce", "#007e68", "#8cbaa9"] + ["#c78e48"] * 4 + ["#647564"] * 6
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharey=True)
    for col, panel in enumerate(PANELS):
        for row, metric in enumerate(("position_rmse_m", "rotation_rmse_rad")):
            ax = axes[row, col]
            for y, variant in enumerate(NAMES):
                scores = np.array([r["metrics"][metric] for r in data["rows"]
                                   if r["panel"] == panel and r["variant"] == variant])
                ax.scatter(scores, np.full(len(scores), y), color=colors[y], s=23)
                ax.scatter(np.sqrt(np.mean(scores ** 2)), y, color=colors[y], marker="|", s=170)
            ax.set_yticks(np.arange(len(NAMES)), list(NAMES.values()))
            ax.set_xscale("log")
            ticks = np.geomspace(*ax.get_xlim(), 4)
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
    fig.suptitle("OpenJev: which observations should adaptation trust?", fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, .07, 1, .97))
    fig.text(.015, .015, "Same frozen checkpoints; no new training. All windows retained, including source-start transients.\n"
             "Dots: individual fits; bars: pooled RMSE. Exposed simulated-robot data, 500 ms forecasts.\n"
             "Uniform-mass controls preserve total support weight while removing its assignment to observations.", fontsize=10)
    fig.savefig(out / "physical-errors.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 10))
    for y, variant in enumerate(NAMES):
        values = np.concatenate([r["latency_ms"] for r in data["rows"] if r["variant"] == variant])
        med, p95 = np.percentile(values, [50, 95])
        ax.scatter(med, y, color=colors[y], s=40)
        ax.plot([med, p95], [y, y], color=colors[y], linewidth=3)
        ax.text(p95 * 1.08, y, f"{med:.2f} / {p95:.2f}", va="center", fontsize=10)
    ax.set_yticks(np.arange(len(NAMES)), list(NAMES.values()))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.5)
    ax.set_xlabel("Complete-window prediction, ms (log scale; median / p95)")
    ax.set_title("Context fitting and all 25 forecast steps included")
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.text(.015, .015, "One CPU thread; 120 samples per neural configuration, 40 per reference.\n"
             "Mass controls pay for deriving their weights. Excludes loading, reporting and service work.", fontsize=10)
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
