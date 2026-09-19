"""Render authenticated saved dialogue results, including deterministic references."""

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METHODS = ("current", "gru", "attention", "gated_delta", "kalman", "innovation_kalman", "carry")
LABELS = ("Current\nturn", "GRU", "Attention", "Gated\ndelta", "Kalman", "Innovation\nKalman", "Learned\ncarry")
SEEDS = (1729, 2718, 3141)
PANELS = ("seen", "unseen")
FILES = ("comparison.png", "comparison.svg", "revisions.png", "revisions.svg", "plot-data.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def probability(value):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
            "Accuracy must be finite and in [0,1]")
    return float(value)


def authenticate(summary_path, summary_sha256, receipt_sha256):
    receipt_path = summary_path.parent / "receipt.json"
    require(sha(summary_path) == summary_sha256, "Summary SHA mismatch")
    require(sha(receipt_path) == receipt_sha256, "Report receipt SHA mismatch")
    for name in ("failed.json", "late-completion.json", "cleanup-error.json"):
        require(not (summary_path.parent / name).exists(), "Report has a failure marker")
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "completed" and receipt["fit_count"] == 21,
            "Completed 21-fit report required")
    require(set(receipt["files"]) == {"accuracy.png", "report.md", "started.json", "summary.json"},
            "Report payload membership mismatch")
    for name, item in receipt["files"].items():
        path = summary_path.parent / name
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                f"Report payload mismatch: {name}")
    require(receipt["files"]["summary.json"]["sha256"] == summary_sha256,
            "Receipt does not bind this summary")
    summary = json.loads(summary_path.read_text())
    require(summary["status"] == "completed" and summary["study"] == "dialogue-memory-v1"
            and summary["development_only"] is True, "Unexpected study scope")
    require(tuple(summary["method_order"]) == METHODS and tuple(summary["seeds"]) == SEEDS,
            "Method/seed membership mismatch")
    require(set(summary["rows"]) == {f"{method}-{seed}" for method in METHODS for seed in SEEDS},
            "Exact 21 rows required")
    require(set(summary["references"]) == {"literal", "none"}, "Both references required")
    gate = summary["continuation_gate"]
    require(gate["checks_total"] == 11 and type(gate["checks_passed"]) is int
            and 0 <= gate["checks_passed"] <= 11 and type(gate["passed"]) is bool
            and gate["passed"] == receipt["continuation_passed"]
            and gate["passed"] == (gate["checks_passed"] == 11), "Gate identity mismatch")
    data = {"continuation_gate": gate, "seeds": list(SEEDS), "methods": list(METHODS),
            "metrics": {}}
    for metric in ("macro_three", "revision"):
        data["metrics"][metric] = {}
        for panel in PANELS:
            entry = {"methods": {}, "references": {}}
            for method in METHODS:
                values = []
                for seed in SEEDS:
                    row = summary["rows"][f"{method}-{seed}"]
                    require(row["method"] == method and row["seed"] == seed, "Row identity mismatch")
                    values.append(probability(row["metrics"][panel][metric]["accuracy"]))
                mean = math.fsum(values) / len(SEEDS)
                require(math.isclose(mean, summary["families"][method][panel][metric]["accuracy"],
                                     rel_tol=0, abs_tol=1e-12), "Family mean mismatch")
                entry["methods"][method] = {"seed_accuracy": values, "mean_accuracy": mean}
            for reference in ("literal", "none"):
                entry["references"][reference] = probability(
                    summary["references"][reference][panel][metric]["accuracy"])
            count = summary["references"]["literal"][panel]["revision"]["count"]
            require(type(count) is int and count > 0, "Revision support required for companion plot")
            require(summary["references"]["none"][panel]["revision"]["count"] == count,
                    "Reference revision support mismatch")
            for row in summary["rows"].values():
                require(row["metrics"][panel]["revision"]["count"] == count,
                        "Revision query count differs across fits")
            entry["revision_queries"] = count
            data["metrics"][metric][panel] = entry
    require(data["metrics"]["macro_three"]["unseen"]["references"]["literal"] > max(
        row["mean_accuracy"] for row in data["metrics"]["macro_three"]["unseen"]["methods"].values()),
        "This descriptive title requires literal carry to lead every unseen neural family")
    return data, receipt


def figure(data, metric, out):
    macro = metric == "macro_three"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "dialogue-memory-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    title = ("Dialogue memory: literal carry wins on unseen services" if macro
             else "Dialogue revisions: deterministic references remain important")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
    gate = data["continuation_gate"]
    fig.text(0.5, 0.925, f"Development only | Continuation: {'PASS' if gate['passed'] else 'FAIL'} "
             f"({gate['checks_passed']}/{gate['checks_total']} checks passed) | All 21 final fits",
             ha="center", color="#424242", fontsize=10)
    handles = [Patch(facecolor="#c4ced9", label="Mean of 3 fits")]
    markers = ("o", "s", "^")
    for seed, marker in zip(SEEDS, markers, strict=True):
        handles.append(Line2D([], [], color="#253746", marker=marker, linestyle="none",
                              markersize=5, label=f"Seed {seed}"))
    handles += [Line2D([], [], color="#a43923", linestyle="--", linewidth=2, label="Literal mention + carry"),
                Line2D([], [], color="#686868", linestyle=":", linewidth=2, label="Always NOT_MENTIONED")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3,
               frameon=False, fontsize=9.5, columnspacing=1.8)
    for ax, panel in zip(axes, PANELS, strict=True):
        entry = data["metrics"][metric][panel]
        x = np.arange(len(METHODS))
        values = np.asarray([entry["methods"][m]["seed_accuracy"] for m in METHODS]) * 100
        means = np.asarray([entry["methods"][m]["mean_accuracy"] for m in METHODS]) * 100
        colors = ["#4c7898" if m == "innovation_kalman" else "#c4ced9" for m in METHODS]
        ax.bar(x, means, color=colors, width=0.7, zorder=2)
        for index, marker in enumerate(markers):
            ax.scatter(x + (index - 1) * 0.14, values[:, index], marker=marker, s=24,
                       facecolor="#253746", edgecolor="white", linewidth=0.4, zorder=5)
        for index, mean in enumerate(means):
            ax.text(index, max(values[index]) + 2.2, f"{mean:.1f}", ha="center", fontsize=9)
        literal, none = (entry["references"][name] * 100 for name in ("literal", "none"))
        ax.axhline(literal, color="#a43923", linestyle="--", linewidth=1.8, zorder=4)
        ax.axhline(none, color="#686868", linestyle=":", linewidth=1.8, zorder=4)
        count_note = "" if macro else f" ({entry['revision_queries']} queries)"
        ax.set_title(f"{panel.title()} services{count_note}\nLiteral {literal:.1f}% | NOT_MENTIONED {none:.1f}%",
                     fontsize=11, pad=12)
        ax.set_xticks(x, LABELS, fontsize=9)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_ylim(-1, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Three-stratum macro accuracy (%)" if macro else "Revision accuracy (%)")
    note = ("Macro = equal mean of unmentioned retention, assigned retention, and changed state."
            if macro else "Revision = an assigned value changes to another assigned value; not an extra continuation test.")
    fig.text(0.5, 0.075, note, ha="center", fontsize=9, color="#454545")
    fig.text(0.5, 0.035, "Each fit uses the same development queries. Points show initialization variability, not independent datasets.",
             ha="center", fontsize=9, color="#454545")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.205, top=0.70, wspace=0.12)
    stem = "comparison" if macro else "revisions"
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white", metadata={"Software": "OpenJev saved-summary renderer"})
    fig.savefig(out / f"{stem}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def render(summary, out, *, summary_sha256, receipt_sha256):
    summary, out = Path(summary).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_hash = sha(__file__)
    inputs = {"summary_path": str(summary), "summary_sha256": summary_sha256,
              "report_receipt_sha256": receipt_sha256}
    try:
        write_json(out / "started.json", {"inputs": inputs, "renderer_source_sha256": source_hash,
                                         "created_utc": datetime.now(UTC).isoformat()})
        data, report_receipt = authenticate(summary, summary_sha256, receipt_sha256)
        figure(data, "macro_three", out)
        figure(data, "revision", out)
        write_json(out / "plot-data.json", data)
        require(sha(summary) == summary_sha256 and sha(summary.parent / "receipt.json") == receipt_sha256
                and sha(__file__) == source_hash, "Input/source changed during rendering")
        receipt = {"status": "completed", "study": "dialogue-memory-v1", "inputs": inputs,
                   "renderer_source_sha256": source_hash, "neural_calls": 0, "encoder_calls": 0,
                   "scope": "Descriptive visualization of authenticated saved aggregates; no new selection rule or independent prediction audit",
                   "development_only": True, "continuation_gate": data["continuation_gate"],
                   "execution_completed_sha256": report_receipt["execution_completed_sha256"],
                   "plan_sha256": report_receipt["plan_sha256"], "fits": 21, "deterministic_references": 2,
                   "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__},
                   "wall_seconds": time.perf_counter() - started,
                   "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in FILES}}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error), "inputs": inputs,
                                           "renderer_source_sha256": source_hash,
                                           "wall_seconds": time.perf_counter() - started})
        except BaseException as preservation_error:  # noqa: BLE001 - retain the original failure
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(f"Failure receipt could not be written: {preservation_error!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = render(args.summary, args.out, summary_sha256=args.summary_sha256,
                    receipt_sha256=args.receipt_sha256)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
