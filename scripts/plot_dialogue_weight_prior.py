"""Plot authenticated, independently checked primary training-weight diagnostics."""
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

METHODS = tuple(f"{method}-{readout}" for method in ("flat_stratum", "token_mean", "token_aligned")
                for readout in ("original", "corrected"))
SEEDS = (6201, 6202, 6203)
LABELS = tuple(f"{method}\n{readout}" for method in ("Flat", "Mean", "Aligned")
               for readout in ("original", "corrected"))
SUPPORT = {"all": 7819, "changed": 578, "retained": 7241,
           "unmentioned_retention": 4032, "assigned_retention": 3209}
VERSION = "dialogue-weight-prior-primary-figure-v1"


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
                "All eighteen method/readout/seed cells required")
        fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
        panels = (("changed", "accuracy", "row", "A  Changed-value accuracy: rows", True),
                  ("changed", "accuracy", "equal_service", "B  Changed-value accuracy: services", True),
                  ("retained", "error", "row", "C  Retained-value errors: rows", False),
                  ("retained", "error", "equal_service", "D  Retained-value errors: services", False),
                  ("all", "accuracy", "row", "E  Overall accuracy: rows", True),
                  ("all", "accuracy", "equal_service", "F  Overall accuracy: services", True))
        colors, markers = ("#306aa7", "#c37737", "#428565"), ("o", "s", "^")
        plotted = {}
        for ax, (stratum, metric, weighting, title, higher) in zip(axes.flat, panels, strict=True):
            panel = values(summary, stratum, metric, weighting)
            plotted[stratum+"/"+weighting] = {"metric": metric, "weighting": weighting, "seed_order": SEEDS,
                "percentages": dict(zip(METHODS, panel, strict=True))}
            for x in (0, 2, 4):
                for i in range(3):
                    ax.plot([x+(i-1)*.06, x+1+(i-1)*.06], [panel[x][i], panel[x+1][i]],
                            color=colors[i], alpha=.3, lw=1, zorder=1)
            for x, numbers in enumerate(panel):
                for i, value in enumerate(numbers):
                    ax.scatter(x+(i-1)*.06, value, color=colors[i], marker=markers[i], s=48,
                               label=str(SEEDS[i]) if x == 0 else None, zorder=3)
                mean = math.fsum(numbers)/3
                ax.plot([x-.25, x+.25], [mean, mean], color="#243444", lw=2)
            ax.set_xticks(range(6), LABELS)
            ax.set_xlim(-.5, 5.5)
            maximum = max(max(row) for row in panel)
            minimum = min(min(row) for row in panel)
            if higher:
                margin = max(1., (maximum-minimum)*.15)
                ax.set_ylim(max(0., minimum-margin), min(100., maximum+margin))
            else:
                ax.set_ylim(0, max(2, math.ceil(maximum*1.2/2)*2))
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
        fig.suptitle("Undoing the prescribed training-loss weights", x=.05, ha="left", fontsize=19, fontweight="bold", y=.98)
        fig.text(.05, .94, "All nine original fits plus one fixed correction each. Original training study: FAIL (18/22).", fontsize=11)
        fig.text(.05, .115, "Correction fixed from FIT counts: 17,666 unmentioned retentions / 9,246 assigned retentions / 2,299 changes.", fontsize=10)
        fig.text(.05, .075, "Each pair uses the same trained model. Faint lines connect paired seeds. Accuracy axes zoom to observed ranges.", fontsize=10, color="#47566b")
        fig.text(.05, .05, "Held-out services from exposed TRAIN; correct previous values supplied. No tuning, training or calibration guarantee.", fontsize=10, color="#47566b")
        fig.text(.05, .025, "Row and equal-service weighting are both shown. Saved-output arithmetic does not establish a new architecture.", fontsize=10, color="#47566b")
        fig.subplots_adjust(left=.075, right=.98, top=.875, bottom=.175, hspace=.65, wspace=.2)
        fig.savefig(out/"weight-prior.png", dpi=180, facecolor="white")
        fig.savefig(out/"weight-prior.pdf", facecolor="white")
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
