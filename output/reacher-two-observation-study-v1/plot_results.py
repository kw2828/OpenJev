"""Plot all completed saved rows, without model or simulator calls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "evidence/reacher-two-observation-study-v1/audit/summary.json"
AUDIT = SOURCE.with_name("receipt.json")
OUT = ROOT / "evidence/reacher-two-observation-study-v1/report"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text())
    assert audit["status"] == "completed" and sha(SOURCE) == audit["files"]["summary.json"]
    summary = json.loads(SOURCE.read_text())
    assert summary["status"] == "completed" and summary["engineering"] is False
    assert summary["saved_output_only"] and summary["new_model_calls"] == 0
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for panel, controls in summary["control"].items():
        for label, value in controls.items():
            rows.append({"panel": panel, "controller": label, "mean_native_cost": value["mean_cost"]})
    assert len(rows) == 42
    with (OUT / "all-rows.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = ["Persistent GRU", "Two observations + actions", "One observation + actions",
              "True-state physics", "Public particle filter", "Public kinematics", "Zero command", "Uniform commands"]
    families = ["residual_gru", "two_observation_gru", "cached_gru"]
    references = ["known_state", "particle", "public_kinematic", "zero", "uniform"]
    colors = ["#147D92", "#D07523", "#7454A3"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.4), sharex=True, sharey=True)
    for ax, panel, title in zip(axes, ["full", "ordinary", "shift"],
                               ["Full sensing", "Six-step gaps", "Ten-step gaps"], strict=True):
        for index, (name, color) in enumerate(zip(families, colors, strict=True)):
            values = summary["families"][panel][name]
            ax.scatter(values["fit_mean_costs"], [index-.13, index, index+.13],
                       s=40, color=color, alpha=.8, zorder=3)
            ax.scatter([values["mean_cost"]], [index], marker="|", s=220, color="black", zorder=4)
            ax.annotate(f'{values["mean_cost"]:.3f}', (values["mean_cost"], index),
                        xytext=(0, 13), textcoords="offset points", fontsize=9, ha="center")
        for index, name in enumerate(references, 3):
            mean = summary["control"][panel][name]["mean_cost"]
            ax.scatter([mean], [index], s=38, marker="s", color="#4B5563", zorder=3)
            ax.annotate(f"{mean:.3f}", (mean, index), xytext=(0, 11),
                        textcoords="offset points", fontsize=9, ha="center")
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xscale("log")
        ax.set_xlim(3.9, 48)
        ax.set_xticks([4, 6, 10, 20, 40])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_ylim(7.7, -.65)
        ax.set_yticks(range(8), labels, fontsize=10)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Native cost (log scale, lower is better)", fontsize=10)
    fig.suptitle("Persistent memory narrowly misses the stronger comparison", x=.04, y=.98,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(.04, .91, "Continuation FAIL: 24/25 checks. Benefit over two observations: 4.44% / 2.94%; required 3% on both gaps.",
             fontsize=11, color="#A13732")
    fig.text(.04, .04, "All 9 models and 5 references, each on 64 paired cases in every panel. Dots: fits; black ticks: family means.\n"
             "Supplied-physics references remain better. Equal search budgets do not equal total compute. No biological or equivalence claim.",
             fontsize=10, color="#374151")
    fig.subplots_adjust(left=.20, right=.985, top=.81, bottom=.18, wspace=.14)
    fig.savefig(OUT / "native-costs.png", dpi=180, facecolor="white")
    plt.close(fig)
    receipt = {"status": "completed", "source_sha256": sha(Path(__file__)),
        "audit_receipt_sha256": sha(AUDIT), "summary_sha256": sha(SOURCE),
        "rows": 42, "learned_fit_points": 27, "reference_points": 15,
        "model_calls": 0, "native_calls": 0, "new_random_draws": 0,
        "files": {name: sha(OUT / name) for name in ["native-costs.png", "all-rows.csv"]}}
    with (OUT / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
