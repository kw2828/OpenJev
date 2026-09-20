"""Plot authenticated, independently checked primary commitment diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("flat", "MM", "AA", "MA", "AM")
SEEDS = (6201, 6202, 6203)
LABELS = ("Flat\nreference", "MM\nMean model", "AA\nAligned model", "MA\nMean mass /\naligned scores", "AM\nAligned mass /\nmean scores")
SUPPORT = {"all": 7819, "changed": 578, "retained": 7241,
           "unmentioned_retention": 4032, "assigned_retention": 3209}
VERSION = "dialogue-commitment-primary-figure-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_pinned(path, pin):
    require(len(pin) == 64 and digest(path) == pin, "External file pin: "+str(path))
    return json.loads(Path(path).read_text())


def values(summary, stratum, metric, weighting="row"):
    output = []
    for method in METHODS:
        row = []
        for seed in SEEDS:
            cell = summary["fits"][f"{method}-{seed}"]["cells"]["heldout_service/"+stratum]
            require(cell["rows"] == SUPPORT[stratum], "Fixed primary support")
            value = cell["metrics"][metric][weighting]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    "Finite diagnostic rate")
            row.append(100*value)
        output.append(row)
    return output


def execute(args):
    out = Path(args.out).resolve()
    require(all(not out.is_relative_to(Path(p).resolve().parent)
                for p in (args.summary, args.receipt, args.audit)), "Separate figure directory")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source_pin = digest(__file__)
    try:
        summary = read_pinned(args.summary, args.summary_sha256)
        receipt = read_pinned(args.receipt, args.receipt_sha256)
        audit = read_pinned(args.audit, args.audit_sha256)
        require(receipt["status"] == "completed" and summary["status"] == "completed",
                "Completed producer required")
        require(receipt["files"]["summary.json"]["sha256"] == args.summary_sha256,
                "Producer summary binding")
        require(audit["status"] == "completed" and audit["agreement"] is True
                and audit["producer_receipt_sha256"] == args.receipt_sha256
                and audit["producer_summary_sha256"] == args.summary_sha256,
                "Matching independent primary audit required")
        require(set(summary["fits"]) == {f"{m}-{s}" for m in METHODS for s in SEEDS},
                "All fifteen construction/seed cells required")
        fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
        panels = (("changed", "accuracy", "row", "A  Changed-value accuracy", True),
                  ("retained", "error", "row", "B  Retained-value errors", False),
                  ("unmentioned_retention", "error", "row", "C  False assignments to unmentioned slots", False),
                  ("assigned_retention", "error", "row", "D  Errors preserving assigned values", False),
                  ("all", "accuracy", "equal_service", "E  Equal-service overall accuracy", True),
                  ("retained", "error", "equal_service", "F  Equal-service retained-value errors", False))
        colors, markers = ("#306aa7", "#c37737", "#428565"), ("o", "s", "^")
        plotted = {}
        for ax, (stratum, metric, weighting, title, higher) in zip(axes.flat, panels, strict=True):
            panel = values(summary, stratum, metric, weighting)
            plotted[stratum+"/"+weighting] = {"metric": metric, "weighting": weighting, "seed_order": SEEDS,
                "percentages": dict(zip(METHODS, panel, strict=True))}
            for x, numbers in enumerate(panel):
                for i, value in enumerate(numbers):
                    ax.scatter(x+(i-1)*.06, value, color=colors[i], marker=markers[i], s=48,
                               label=str(SEEDS[i]) if x == 0 else None, zorder=3)
                mean = math.fsum(numbers)/3
                ax.plot([x-.25, x+.25], [mean, mean], color="#243444", lw=2)
            ax.set_xticks(range(5), LABELS)
            ax.set_xlim(-.5, 4.5)
            maximum = max(max(row) for row in panel)
            ax.set_ylim(0, 100 if higher else max(5, math.ceil(maximum*1.2/5)*5))
            weight_label = f"Percent of {SUPPORT[stratum]:,} rows" if weighting == "row" else "Mean service percent"
            ax.set_ylabel(f"{weight_label}; {'higher' if higher else 'lower'} is better")
            ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
            ax.grid(axis="y", color="#dce3eb")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=9)
        overall = values(summary, "all", "accuracy")
        means = [math.fsum(row)/3 for row in overall]
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="#243444", lw=2))
        labels.append("Three-seed mean")
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.045, .92), ncol=4, frameon=False)
        fig.suptitle("Separating stay/change mass from alternative scores", x=.05, ha="left", fontsize=19, fontweight="bold", y=.98)
        fig.text(.05, .94, "Held-out services; saved distributions only. All three seeds. Original training study: FAIL (18/22).", fontsize=11)
        fig.text(.05, .115, "Held-out-service accuracy (all 7,819 rows; seed means):  "+"   |   ".join(f"{m} {v:.2f}%" for m, v in zip(METHODS, means, strict=True)), fontsize=10)
        fig.text(.05, .075, "MM/AA reconstruct the original models. MA/AM combine two source distributions; both source predictions have a cost.", fontsize=10, color="#47566b")
        fig.text(.05, .05, "Exposed TRAIN development with correct previous values supplied. Seeds repeat optimization, not independent service sampling.", fontsize=10, color="#47566b")
        fig.text(.05, .025, "Probability-factor diagnostic, not a trained model, causal gate intervention, fresh evaluation, or reversal of the failed study.", fontsize=10, color="#47566b")
        fig.subplots_adjust(left=.075, right=.98, top=.875, bottom=.175, hspace=.65, wspace=.2)
        fig.savefig(out/"commitment.png", dpi=180, facecolor="white")
        fig.savefig(out/"commitment.pdf", facecolor="white")
        plt.close(fig)
        write(out/"plotted-values.json", {"panels": plotted, "all_row_accuracy_percent": dict(zip(METHODS, means, strict=True))})
        require(digest(args.summary) == args.summary_sha256 and digest(args.receipt) == args.receipt_sha256
                and digest(args.audit) == args.audit_sha256 and digest(__file__) == source_pin, "Final input stability")
        files = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        require(time.monotonic()-start <= 60 and sum(v["bytes"] for v in files.values()) <= 32*1024**2, "Figure resource limits")
        write(out/"receipt.json", {"version": VERSION, "status": "completed", "source_sha256": source_pin,
            "summary_sha256": args.summary_sha256, "producer_receipt_sha256": args.receipt_sha256,
            "audit_receipt_sha256": args.audit_sha256, "files": files, "wall_seconds": time.monotonic()-start,
            "model_calls": 0, "prediction_arrays_read": 0, "scope": "Audited primary aggregate visualization only"})
    except BaseException as error:
        try:
            write(out/"failed.json", {"version": VERSION, "status": "failed", "source_sha256": source_pin,
                "error": repr(error), "wall_seconds": time.monotonic()-start, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve original plotting failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Figure failure preservation: "+repr(secondary))
        raise


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "receipt", "audit"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    cli()
