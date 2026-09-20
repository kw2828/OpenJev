"""Render authenticated synthetic timing observations; no model or data calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

METHODS = ("flat_stratum", "token_mean", "token_aligned")
PHASES = ("train", "evaluation")
LABELS = {"flat_stratum": "Flat baseline", "token_mean": "Token mean", "token_aligned": "Token aligned"}
COLORS = {"flat_stratum": "#738297", "token_mean": "#197d89", "token_aligned": "#8257a6", "overhead": "#b8c1cc"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def near(a, b):
    require(type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), "Timing arithmetic")


def authenticate(run, completion_pin):
    require(sha(run / "completed.json") == completion_pin, "External completion pin")
    done = read(run / "completed.json")
    require(done["status"] == "completed" and done["version"] == "dialogue-token-alignment-capacity-v1"
            and done["phase"] == "run" and done["technical_complete"] is True
            and done["encoder_calls"] == done["real_feature_arrays_decoded"] == 0 and done["quality_metrics"] is False,
            "Completed synthetic-only probe")
    expected = {"started.json", "plan.json", "events.jsonl", "timings.jsonl", "projection.json"}
    expected |= {f"profile-{phase}-{stratum}.json" for phase in PHASES for stratum in range(3)}
    require(set(done["files"]) == expected and {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
            == expected | {"completed.json"}, "Exact twelve-file successful closure")
    for name, item in done["files"].items():
        path = run / name
        require(not path.is_symlink() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Payload identity: " + name)
    plan = read(run / "plan.json")
    require(sha(run / "plan.json") == done["plan_sha256"] and plan["source_sha256"] == done["source_sha256"], "Frozen plan/source map")
    recipe = plan["recipe"]
    require(recipe["seeds"] == [6201, 6202, 6203] and recipe["epochs"] == 20 and recipe["batch_size"] == 256
            and recipe["warm_events"] == 1 and recipe["measured_events"] == 3 and recipe["strata"] == 3
            and recipe["admission_seconds"] == 2880 and recipe["study_ceiling_seconds"] == 3600,
            "Frozen projection recipe")
    events = [json.loads(line) for line in (run / "timings.jsonl").read_text().splitlines()]
    journal = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    identities = [(phase, s, method, rep) for phase in PHASES for s in range(3) for rep in range(4)
                  for method in METHODS[rep % 3:] + METHODS[:rep % 3]]
    require(len(events) == len(journal) == done["events"] == 72
            and [(e["phase"], e["stratum"], e["method"], e["repeat"]) for e in events] == identities,
            "All 72 events, controls and rotation")
    for event, recorded in zip(events, journal, strict=True):
        require({k: v for k, v in event.items() if k != "seconds"} == recorded
                and event["warmup"] is (event["repeat"] == 0)
                and type(event["seconds"]) in (int, float) and math.isfinite(event["seconds"]) and event["seconds"] > 0,
                "Exact timing/journal join")
    progress = done["progress"]
    require(progress["events_started"] == progress["events_completed"] == 72
            and progress["optimizer_attempted"] == progress["optimizer_returned"] == 36
            and progress["active"] is None, "Complete paid event/update coverage")
    projection = read(run / "projection.json")
    require(projection == done["projection"] and len(projection["cells"]) == 18, "Terminal projection identity")
    saved = {(c["phase"], c["stratum"], c["method"]): c for c in projection["cells"]}
    require(len(saved) == 18, "Unique eighteen cells")
    cells, totals = [], {method: 0. for method in METHODS}
    for phase in PHASES:
        strata = plan["geometry_strata"][phase]
        require([s["stratum"] for s in strata] == [0, 1, 2]
                and sum(s["batches_per_arm"] for s in strata) == (6900 if phase == "train" else 162), "Population coverage")
        for stratum in range(3):
            for method in METHODS:
                times = [e["seconds"] for e in events if (e["phase"], e["stratum"], e["method"]) == (phase, stratum, method)
                         and not e["warmup"]]
                require(len(times) == 3, "Three observations per displayed cell")
                cell = saved[phase, stratum, method]
                require(cell["batches"] == strata[stratum]["batches_per_arm"], "Stratum multiplier")
                near(cell["maximum_measured_seconds"], max(times))
                near(cell["projected_seconds"], max(times) * cell["batches"])
                cells.append({**cell, "measurements_seconds": times})
                totals[method] += cell["projected_seconds"]
    overhead = projection["observed_nonmeasured_seconds"]
    require(type(overhead) in (int, float) and math.isfinite(overhead) and overhead >= 0, "Observed remainder")
    total = math.fsum(totals.values()) + overhead
    near(projection["projected_seconds"], total)
    admitted = total <= 2880 and done["process_lifetime_peak_rss_bytes"] <= 6 * 1024**3
    require(projection["admission_seconds"] == 2880 and projection["study_ceiling_seconds"] == 3600
            and projection["admitted"] is admitted and done["admitted"] is admitted, "Frozen admission result")
    return done, cells, totals, overhead, total


def render(out, done, cells, totals, overhead, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(3.7, 1.65), hspace=.49, wspace=.40)
    for column, phase in enumerate(PHASES):
        ax = fig.add_subplot(grid[0, column])
        panel = [c for c in cells if c["phase"] == phase]
        xmax = max(max(c["measurements_seconds"]) * 1000 for c in panel)
        labels = []
        for row, cell in enumerate(panel):
            color = COLORS[cell["method"]]
            times = [x * 1000 for x in cell["measurements_seconds"]]
            if row in (3, 6): ax.axhline(row - .5, color="#dce2e8", lw=1)
            ax.plot([min(times), max(times)], [row, row], color=color, alpha=.55, lw=2)
            ax.scatter(times, [row - .10, row, row + .10], s=26, color=color, alpha=.7, zorder=3)
            ax.scatter([max(times)], [row], marker="D", facecolor="white", edgecolor=color, linewidth=1.6, s=55, zorder=4)
            ax.text(xmax * 1.22, row, f"{max(times):.1f}", ha="right", va="center", color=color, fontsize=9)
            labels.append(f"S{cell['stratum'] + 1}  {LABELS[cell['method']]}")
        ax.set_yticks(range(9), labels)
        ax.invert_yaxis(); ax.set_xlim(0, xmax * 1.26)
        ax.set_xlabel("Measured milliseconds per effective batch")
        ax.set_title("Training: forward, backward + update" if phase == "train" else "Evaluation: forward + output materialization",
                     loc="left", fontsize=12, weight="bold", pad=14)
        ax.xaxis.grid(True, alpha=.18); ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    ax = fig.add_subplot(grid[1, :])
    left = 0.
    for method in METHODS:
        width = totals[method]
        ax.barh(0, width, left=left, height=.42, color=COLORS[method], edgecolor="white")
        if width > total * .08:
            ax.text(left + width / 2, 0, f"{width / 60:.1f} min", ha="center", va="center", color="white", weight="bold")
        left += width
    ax.barh(0, overhead, left=left, height=.42, color=COLORS["overhead"], edgecolor="white")
    ax.axvline(2880, color="#a45139", ls="--", lw=1.5)
    ax.axvline(3600, color="#293c51", ls=":", lw=1.6)
    ax.text(2880, .39, "2,880 s admission", ha="right", va="center", color="#a45139", fontsize=9)
    ax.text(3600, .67, "3,600 s study ceiling", ha="right", va="center", color="#293c51", fontsize=9)
    ax.set_ylim(-.48, .87); ax.set_xlim(0, max(total, 3600) * 1.05); ax.set_yticks([])
    ax.set_xlabel("Projected seconds for all nine fits, including final evaluations")
    ax.set_title(f"Nine-fit projection: {total:,.1f} seconds ({total / 60:.2f} minutes)", loc="left", fontsize=12, weight="bold", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.xaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
    handles = [Patch(color=COLORS[m], label=f"{LABELS[m]}: three fits") for m in METHODS]
    handles.append(Patch(color=COLORS["overhead"], label=f"Shared observed remainder: {overhead:.1f} s"))
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0, -.32), fontsize=9)
    outcome = "ADMITTED" if done["admitted"] else "NOT ADMITTED - stop before training"
    fig.suptitle("Token alignment: synthetic capacity screen", x=.035, y=.978, ha="left", fontsize=19, weight="bold")
    fig.text(.035, .943, outcome, color="#197d89" if done["admitted"] else "#a45139", weight="bold", fontsize=12)
    fig.text(.035, .914, "All 18 timing cells shown. Synthetic values and labels on fixed public geometry; no task-quality results.", fontsize=11)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="#738297", linestyle="none", label="Each of 3 measured events"),
                        Line2D([0], [0], marker="D", markerfacecolor="white", color="#738297", linestyle="none", label="Maximum used for projection")],
               loc="upper right", bbox_to_anchor=(.978, .951), frameon=False, fontsize=9)
    fig.text(.035, .035, "Projection = each stratum population × its observed maximum + shared remainder. Heuristic, not a measured full-study duration.", fontsize=9, color="#526174")
    fig.text(.035, .016, f"S1-S3 are deterministic workload strata. Warmups are in the remainder. Schema preparation ({done['schema_preparation']['wall_seconds']:.2f} s) is recorded separately.", fontsize=9, color="#526174")
    fig.subplots_adjust(left=.14, right=.965, top=.857, bottom=.17)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"capacity.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    source_pin = sha(__file__)
    try:
        done, cells, totals, overhead, total = authenticate(Path(args.run), args.completion_sha256)
        render(out, done, cells, totals, overhead, total)
        require(sha(Path(args.run) / "completed.json") == args.completion_sha256 and sha(__file__) == source_pin, "End source/input stability")
        for name, item in done["files"].items(): require(sha(Path(args.run) / name) == item["sha256"], "End payload stability")
        record = {"status": "completed", "source_sha256": source_pin, "execution_completed_sha256": args.completion_sha256,
                  "plan_sha256": done["plan_sha256"], "authenticated_execution_files": done["files"], "events": 72,
                  "measured_cells": 18, "measurements_per_cell": 3, "admitted": done["admitted"],
                  "projected_seconds": total, "projected_arm_seconds": totals, "shared_observed_remainder_seconds": overhead,
                  "outputs": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())},
                  "scope": "Saved synthetic timing visualization only; no model, encoder, RNG or task-quality calls. All three observations and maximum shown per cell; ranges are not confidence intervals. Projection is not measured training cost."}
        with (out / "receipt.json").open("x") as stream: json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"status": "completed", "admitted": done["admitted"], "receipt_sha256": sha(out / "receipt.json")}))
    except BaseException as error:
        try:
            with (out / "failed.json").open("x") as stream:
                json.dump({"status": "failed", "error": repr(error), "source_sha256": source_pin,
                           "execution_completed_sha256": args.completion_sha256}, stream, indent=2)
        except BaseException as secondary:  # noqa: BLE001 - preserve original error
            if callable(getattr(error, "add_note", None)): error.add_note("Failure receipt: " + repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
