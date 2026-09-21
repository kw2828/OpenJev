"""All-seed primary result figure, only from a completed independently checked report."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "output/dialogue-warm-pooling-v1"
OUT = BASE / "figure-01/render-01"
ARMS = ("untouched", "pooled", "schema_attention", "state_token", "belief_query")
LABELS = ("Untouched", "Pooled\ncontinuation", "Schema\nattention", "State\ntoken", "Belief\nquery")
COLORS = ("#87939D", "#506580", "#B88847", "#A76B81", "#167C75")
SEEDS = (6901, 6902, 6903)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    source = BASE / "report-01/summary.json"
    report = BASE / "report-01/receipt.json"
    audit = BASE / "audit-01/result-01/receipt.json"
    receipt, verified, summary = read(report), read(audit), read(source)
    assert receipt["status"] == verified["status"] == summary["status"] == "completed"
    assert verified["agreement"] is True
    assert sha(source) == receipt["files"]["summary.json"]["sha256"] == verified["producer_summary_sha256"]
    assert sha(report) == verified["producer_receipt_sha256"]
    assert summary["scored_fits"] == 15 and summary["probabilities_transformed"] is False
    specs = (("macro_three", "accuracy", 100, "Macro accuracy (%)", "Higher is better"),
             ("strata", "changed", 100, "Changed-decision accuracy (%)", "Higher is better"),
             ("micro", "nll", 1, "Log loss", "Lower is better"),
             ("micro", "brier", 1, "Brier score", "Lower is better"))
    values = {}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3))
    fig.patch.set_facecolor("#FAFBFC")
    for ax, (group, metric, scale, label, direction) in zip(axes.flat, specs, strict=True):
        key = "changed_accuracy" if group == "strata" else metric
        values[key] = {}
        ax.set_facecolor("#FAFBFC")
        for x, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
            points = []
            for seed in SEEDS:
                panel = summary["fits"][f"{arm}-{seed}"]["panels"]["unseen"]
                value = panel[group][metric]["accuracy"] if group == "strata" else panel[group][metric]
                assert np.isfinite(value)
                points.append(scale * value)
            values[key][arm] = {"seeds": dict(zip(map(str, SEEDS), points, strict=True)), "mean": float(np.mean(points))}
            ax.plot([x-.24, x+.24], [np.mean(points)]*2, color=color, lw=3, zorder=2)
            for offset, value, marker in zip((-.12, 0, .12), points, ("o", "s", "^"), strict=True):
                ax.scatter(x+offset, value, color=color, marker=marker, s=48, edgecolors="white", linewidths=.8, zorder=3)
        ax.set_xticks(range(5), LABELS, fontsize=9)
        ax.set_title(f"{label}  |  {direction}", loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="y", color="#DDE2E6", lw=.6)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.margins(y=.18, x=.08)
        for spine in ax.spines.values():
            spine.set_visible(False)
    gate = summary["continuation"]
    result = "PASS" if gate["passed"] else "FAIL"
    fig.text(.055, .956, "Does belief-guided attention improve a trained memory?", fontsize=19, fontweight="bold", color="#182E3F")
    fig.text(.055, .919, f"Unseen-service DEV  |  All three paired seeds  |  Fixed continuation rule: {result}, {gate['checks_passed']}/32", fontsize=11, color="#4B5B68")
    fig.text(.055, .050, "Markers: seeds 6901 (circle), 6902 (square), 6903 (triangle). Lines: equal-seed means. Raw probabilities.", fontsize=9, color="#4B5B68")
    fig.text(.055, .025, "Full exposed development cohort; trained encoder frozen. Seen-service safeguards also enter the rule. No test-set or novelty claim.", fontsize=9, color="#4B5B68")
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.14, hspace=.47, wspace=.21)
    OUT.mkdir(exist_ok=False)
    fig.savefig(OUT / "warm-pooling.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(OUT / "warm-pooling.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUT / "plotted-values.json").write_text(json.dumps(values, indent=2, sort_keys=True)+"\n")
    record = {"status": "completed", "source_sha256": sha(Path(__file__)), "summary_sha256": sha(source),
        "report_receipt_sha256": sha(report), "audit_receipt_sha256": sha(audit), "model_calls": 0,
        "scope": "All15 primary unseen fits, paired seeds and all32 continuation status. No outcome omitted.",
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir())}}
    (OUT / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": "completed", "figure": str(OUT / "warm-pooling.png")}))


if __name__ == "__main__":
    main()
