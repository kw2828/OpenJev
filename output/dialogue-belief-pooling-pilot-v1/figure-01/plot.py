"""Saved-report-only figure; no weights, predictions or model calls."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "report-01/summary.json"
PIN = None  # Bound in the render receipt together with the report receipt.
OUT = BASE / "figure-01/render-01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    summary = json.loads(SOURCE.read_text())
    receipt = json.loads((BASE / "report-01/receipt.json").read_text())
    assert sha(SOURCE) == receipt["files"]["summary.json"]["sha256"]
    assert summary["status"] == "completed" and summary["complete_fits"] == 8
    arms = ["pooled", "schema_attention", "belief_query", "state_token"]
    labels = ["Simple\npooling", "Schema\nattention", "Belief-guided\nattention", "State\ntoken"]
    colors = ["#63758B", "#4169AB", "#D25C35", "#158477"]
    strata = ["unmentioned_retention", "assigned_retention", "changed"]
    values = {arm: {"macro_accuracy_percent": [100 * summary["fits"][f"{arm}-{seed}"]["panels"]["all"]["macro_accuracy"]
                   for seed in (7101, 7102)],
                   "stratum_accuracy_percent": {s: [100 * summary["fits"][f"{arm}-{seed}"]["panels"]["all"]["strata"][s]["accuracy"]
                    for seed in (7101, 7102)] for s in strata}} for arm in arms}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor("#FAFBFD")
    for ax in axes:
        ax.set_facecolor("#FAFBFD")
        ax.set_ylim(-3, 104)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", color="#E0E5EC", zorder=0)
    for i, arm in enumerate(arms):
        v = values[arm]["macro_accuracy_percent"]
        axes[0].scatter([i-.05, i+.05], v, c=colors[i], s=42, zorder=3)
        axes[0].plot([i-.20, i+.20], [np.mean(v)] * 2, c=colors[i], linewidth=2.5)
        axes[0].text(i, np.mean(v)+4, f"{np.mean(v):.2f}%", ha="center", color=colors[i], weight="bold")
        means = [np.mean(values[arm]["stratum_accuracy_percent"][s]) for s in strata]
        axes[1].bar(np.arange(3)+(i-1.5)*.18, means, width=.17, color=colors[i],
                    label=labels[i].replace("\n", " "), zorder=3)
    axes[0].set_title("No accuracy gain over attention controls", pad=15)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_xlim(-.6, 3.6)
    axes[1].set_title("Every fit fails to recognize changed state", pad=15)
    axes[1].set_xticks(range(3), ["Unmentioned\nretention", "Assigned\nretention", "Changed\ndecisions"])
    axes[1].text(2, 8, "0% in all 8 fits", ha="center", weight="bold", color="#A4442B")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.02, .83), frameon=False, fontsize=9)
    gate = summary["continuation"]
    fig.suptitle("OpenJev: belief-conditioned pooling pilot", x=.06, ha="left", fontsize=19, weight="bold")
    fig.text(.06, .905, f"128 training + 128 DEV dialogues | Frozen MiniLM | 2 seeds, 3 epochs | Rule: FAIL ({gate['checks_passed']}/18 passed)", color="#536076")
    fig.text(.06, .065, "Left: equal average of three transition strata; dots show both seeds. Right: equal seed means.\n"
             "Development screen with freshly initialized heads. This result does not establish an architecture advantage.", fontsize=10, color="#536076")
    fig.subplots_adjust(left=.06, right=.985, bottom=.24, top=.80, wspace=.27)
    fig.savefig(OUT / "pooling-pilot.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUT / "plotted-values.json").write_text(json.dumps(values, indent=2, sort_keys=True)+"\n")
    record = {"status": "completed", "source_sha256": sha(Path(__file__)),
              "summary_sha256": sha(SOURCE), "report_receipt_sha256": sha(BASE / "report-01/receipt.json"),
              "model_calls": 0, "figure_scope": "All eight fits, full selected DEV population; no outcomes omitted",
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in OUT.iterdir() if p.is_file()}}
    (OUT / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": "completed", "figure": str(OUT / "pooling-pilot.png")}))


if __name__ == "__main__":
    main()
