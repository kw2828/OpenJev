"""Visualize metadata-only stopped collection status, without scientific reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

VERSION = "otto-score-forecast-stop-render-v1"
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular nonsymlink file")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return {"path": str(path), "sha256": value.hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def values(report):
    require(report["version"] == "otto-score-forecast-stop-summary-v1"
            and report["status"] == "incomplete_collection_verified" and report["rule"] == "not_evaluated"
            and report["arrays_read"] == report["scientific_calls"] == 0, "verified metadata-only stop report")
    require(report["terminal_timed_out"] is True and report["terminal_returncode"] != 0
            and report["terminal_cleanup"]["group_absent"] is True
            and report["terminal_cleanup"]["reaped"] is True, "closed original timeout")
    require(report["forecast_criteria"] == "all 45 not evaluated"
            and report["fitting"] == "not performed by this collection worker; not admitted by this report",
            "no admitted fitting or forecast criterion evaluation")
    stages = []
    for name, total in (("train", 54), ("valid", 36)):
        source = report["phases"][name]
        complete, unstarted = source["acknowledged_complete_episodes"], source["unstarted_episodes"]
        require(source["declared_episodes"] == total and type(complete) is int and type(unstarted) is int
                and 0 <= complete <= total and 0 <= unstarted <= total - complete, "complete phase coverage")
        stages.append({"stage": name, "declared": total, "complete": complete,
                       "interrupted": total - complete - unstarted, "unstarted": unstarted})
    require(sum(row["complete"] for row in stages) == report["acknowledged_complete_episodes"]
            and sum(row["declared"] for row in stages) == report["declared_episodes"] == 90,
            "all declared paths accounted for")
    require(sum(row["interrupted"] for row in stages) == 1
            and report["pending_episode_relation"] == "next_unacknowledged_episode"
            and report["pending_episode"] == report["boundary_pending_episode"], "one retained interrupted path")
    elapsed = report["parent_wall_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= report["original_cap_seconds"] == 900,
            "original parent elapsed includes timeout cleanup")
    return {"version": VERSION, "phases": stages,
            "complete": report["acknowledged_complete_episodes"], "declared": 90,
            "fits_completed": 0, "fits_planned": 12,
            "criteria_planned": 45, "criteria_status": "NOT EVALUATED",
            "parent_wall_seconds": elapsed, "original_cap_seconds": 900,
            "status": "COLLECTION TIMED OUT", "scope": "Partial collection, not task success or an architecture result.",
            "timing_scope": "Original parent elapsed includes cleanup; operation timers are not added or plotted."}


def figure(output, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(12, 6.6), facecolor="white")
    fig.suptitle("Collection stopped at the fixed time limit", fontsize=20, y=.955)
    fig.text(.5, .875, f"{data['complete']}/{data['declared']} paths completed | Partial collection, not task success",
             ha="center", fontsize=13, color="#8C3B21")
    ax = fig.add_axes((.13, .47, .8, .30))
    colors = {"complete": "#347AA6", "interrupted": "#D78820", "unstarted": "#D9DEE3"}
    for y, row in zip((1, 0), data["phases"], strict=True):
        left = 0
        for field in ("complete", "interrupted", "unstarted"):
            count = row[field]
            if count:
                ax.barh(y, count, left=left, height=.48, color=colors[field], edgecolor="white", linewidth=.8)
                if field != "interrupted":
                    ax.text(left + count / 2, y, f"{count} {field}", ha="center", va="center",
                            color="white" if field == "complete" else "#303840", fontsize=12)
                else:
                    ax.annotate(f"{count} interrupted", xy=(left + count / 2, y - .24),
                                xytext=(left + count / 2 + 4, y - .68), ha="center", fontsize=11,
                                color="#85520C", arrowprops={"arrowstyle": "-", "color": "#A46A16"})
            left += count
    ax.set_yticks((1, 0), ("TRAIN\n54 planned", "VALID\n36 planned"))
    ax.set_xlim(0, 55)
    ax.set_ylim(-.88, 1.6)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor=colors[key], label=label) for key, label in
                        (("complete", "Acknowledged complete"), ("interrupted", "Interrupted"), ("unstarted", "Unstarted"))],
               loc="center", bbox_to_anchor=(.5, .415), ncol=3, frameon=False)
    for x, headline, caption in ((.2, "0 / 12", "fits performed"),
                                  (.5, "45 criteria", "NOT EVALUATED"),
                                  (.8, f"{data['parent_wall_seconds']:,.3f} s", "original parent elapsed")):
        fig.text(x, .29, headline, ha="center", fontsize=22, weight="bold", color="#273746")
        fig.text(x, .235, caption, ha="center", fontsize=11, color="#43515D")
    fig.text(.5, .14, "The 900-second allocation was not extended. Parent elapsed includes process cleanup.",
             ha="center", fontsize=10)
    fig.text(.5, .095, "Complete TRAIN collection does not admit fitting after incomplete VALID collection.",
             ha="center", fontsize=10)
    fig.text(.5, .045, "No forecast accuracy, autonomous performance or architecture result is established.",
             ha="center", fontsize=11, color="#8C3B21")
    for suffix in ("png", "svg"):
        with (output / f"score-forecast-stop.{suffix}").open("xb") as stream:
            fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.18)
            stream.flush()
            os.fsync(stream.fileno())
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.report.is_absolute() and args.output.is_absolute()
            and not any(p.is_symlink() for p in args.output.parents), "absolute paths and regular output parents")
    args.output.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS, "arrays_read": 0, "scientific_calls": 0}

    def check():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt["peak_rss_bytes"] = rss
        require(time.monotonic() - start < LIMITS["seconds"] and rss <= LIMITS["rss_bytes"]
                and sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "bounded metadata presentation")

    try:
        report_pin = descriptor(args.report)
        require(report_pin["sha256"] == args.report_sha256 and report_pin["bytes"] <= 1024**2, "external stop-report pin")
        receipt.update(input=report_pin, source=descriptor(Path(__file__).absolute()))
        data = values(json.loads(args.report.read_text()))
        write(args.output / "plotted-values.json", data)
        check()
        receipt["matplotlib_version"] = figure(args.output, data)
        check()
        require(descriptor(args.report) == report_pin and descriptor(Path(__file__).absolute()) == receipt["source"],
                "unchanged input and source")
        receipt.update(status="completed", files={p.name: descriptor(p) for p in sorted(args.output.iterdir())},
                       wall_seconds=time.monotonic() - start, scope=data["scope"])
        write(args.output / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "completed", "receipt": descriptor(args.output / "receipt.json")}), flush=True)
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc(), wall_seconds=time.monotonic() - start)
        try:
            if (args.output / "receipt.json").exists():
                (args.output / "receipt.json").rename(args.output / "receipt.invalid.json")
            receipt["files"] = {p.name: descriptor(p) for p in args.output.iterdir() if p.is_file()}
            write(args.output / "receipt.json", receipt)
        except BaseException as publication:  # noqa: BLE001 - retain original presentation failure
            error.add_note(f"Failure receipt publication: {publication!r}")
        raise


if __name__ == "__main__":
    main()
