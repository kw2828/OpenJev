"""Presentation-only figures from a complete externally pinned saved report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

VERSION = "dialogue-observation-figure-v1"
REPORT_VERSION = "dialogue-observation-report-v1"
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
SEEDS = (6901, 6902, 6903)
LABELS = ("Frozen / original", "Frozen / number words", "Trainable / original", "Trainable / number words")
COLORS = ("#64748b", "#2563a6", "#c16925", "#873a9d")
LIMITS = {"wall_seconds": 60, "rss_bytes": 1024**3, "output_bytes": 32 * 1024**2}
METRICS = (
    ("seen_macro", "seen", "macro_three", "accuracy", "Seen services: macro accuracy", 100., "%", "Higher is better"),
    ("unseen_macro", "unseen", "macro_three", "accuracy", "Unseen services: macro accuracy", 100., "%", "Higher is better"),
    ("unseen_nll", "unseen", "micro", "nll", "Unseen services: log loss", 1., "nats", "Lower is better"),
    ("unseen_brier", "unseen", "micro", "brier", "Unseen services: Brier score", 1., "squared probability error", "Lower is better"),
    ("seen_assigned_error", "seen", "assigned_retention", "error", "Seen services: assigned-retention error", 100., "%", "Lower is better"),
    ("unseen_assigned_error", "unseen", "assigned_retention", "error", "Unseen services: assigned-retention error", 100., "%", "Lower is better"),
)
CONDITIONS = (
    ("unseen_macro_gain_1pp", "Unseen macro gain", "Mean delta >= +1.0 pp", .01, "ge", 100., "pp"),
    ("unseen_macro_strict_paired_wins", "Strict unseen macro wins", "At least 2 of 3 paired seeds", None, None, 1., "wins"),
    ("unseen_micro_nll_nonworse", "Unseen micro NLL", "Primary mean <= control mean", 0., "le", 1., "nats"),
    ("unseen_micro_brier_nonworse", "Unseen micro Brier", "Primary mean <= control mean", 0., "le", 1., "score"),
    ("seen_macro_deficit_at_most_1pp", "Seen macro change", "Mean delta >= -1.0 pp", -.01, "ge", 100., "pp"),
    ("seen_assigned_retention_error_increase_at_most_half_pp", "Seen assigned error", "Mean delta <= +0.5 pp", .005, "le", 100., "pp"),
    ("unseen_assigned_retention_error_increase_at_most_half_pp", "Unseen assigned error", "Mean delta <= +0.5 pp", .005, "le", 100., "pp"),
)
SCOPE = ("Saved-report presentation only. No raw predictions, corpus, checkpoints, model, or independent audit "
         "is loaded or replayed. Root separately requires the independent result audit before a real render. "
         "Official DEV is exposed development data; training seeds repeat the same examples, not independent "
         "test samples. No confidence intervals, new architecture, calibration, or world-model claim.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def authenticate(args):
    report = args.report.resolve()
    receipt_path, summary_path = report / "receipt.json", report / "summary.json"
    require(sha(receipt_path) == args.receipt_sha256 and sha(summary_path) == args.summary_sha256,
            "External report receipt and summary pins")
    receipt = read(receipt_path)
    require(receipt["status"] == "completed" and receipt["version"] == REPORT_VERSION
            and receipt["technical_validity_passed"] is True and receipt["model_calls"] == 0,
            "Completed technically valid saved-only report")
    names = {"started.json", "summary.json", "report.md"}
    require(set(receipt["files"]) == names
            and {p.relative_to(report).as_posix() for p in report.rglob("*") if p.is_file()} == names | {"receipt.json"},
            "Exact report membership")
    bindings = {receipt_path: descriptor(receipt_path)}
    for name in names:
        path = report / name
        require(not path.is_symlink() and descriptor(path) == receipt["files"][name], "Report payload identity: " + name)
        bindings[path] = descriptor(path)
    summary = read(summary_path)  # All external pins and payloads verified before quality decode.
    require(summary["version"] == REPORT_VERSION and summary["status"] == "completed"
            and summary["technical_validity_passed"] is True and summary["technical_complete_fits"] == 12,
            "All twelve fits technically complete")
    require(all(summary[k] == receipt[k] for k in ("plan_sha256", "execution_completed_sha256")), "Report lineage join")
    require(all(v.get("synthetic", False) is bool(args.synthetic) for v in (receipt, summary)), "Explicit synthetic scope")
    require(set(summary["fits"]) == {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}, "Exact twelve-fit membership")
    gate = summary["continuation"]
    require(gate["complete_fit_membership"] is True and gate["checks_total"] == receipt["scientific_checks_total"] == 7,
            "Seven original conditions")
    require([c["name"] for c in gate["checks"]] == [c[0] for c in CONDITIONS], "Exact condition identity/order")
    for check, definition in zip(gate["checks"], CONDITIONS, strict=True):
        require(type(check["passed"]) is bool, "Boolean condition outcome")
        if definition[3] is None:
            require(check["required"] == 2 and check["wins"] in (None, 0, 1, 2, 3), "Strict paired wins condition")
        else:
            require(check["threshold"] == definition[3] and check["comparison"] == definition[4], "Unchanged logical boundary")
            require(check["paired_seed_values"] is None or len(check["paired_seed_values"]) == 3, "Three paired seed values")
    require(gate["checks_passed"] == receipt["scientific_checks_passed"] == sum(c["passed"] for c in gate["checks"])
            and gate["passed"] is receipt["continuation_passed"] is all(c["passed"] for c in gate["checks"]), "Condition conjunction")
    return summary, bindings


def extract(summary):
    result = {}
    for key, panel, group, metric, *_ in METRICS:
        values, support = {}, None
        for arm in ARMS:
            points = []
            for seed in SEEDS:
                p = summary["fits"][f"{arm}-{seed}"]["panels"][panel]
                if group == "macro_three":
                    n = {s: p["strata"][s]["count"] for s in ("unmentioned_retention", "assigned_retention", "changed")}
                    value = p[group][metric]
                else:
                    cell = p["micro"] if group == "micro" else p["strata"][group]
                    n, value = {group: cell["count"]}, cell[metric]
                require(all(type(v) is int and v >= 0 for v in n.values()) and (support is None or n == support), "Same panel support across all fits")
                support = n
                require(value is None or finite(value) and value >= 0, "Defined finite nonnegative plotted value")
                require(metric not in ("accuracy", "error") or value is None or value <= 1, "Valid plotted rate")
                points.append(value)
            values[arm] = {"seeds": dict(zip(map(str, SEEDS), points, strict=True)),
                           "mean": None if any(p is None for p in points) else math.fsum(points) / 3}
        result[key] = {"support": support, "arms": values}
    return result


def render(summary, values, out, synthetic):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180, "svg.hashsalt": VERSION})
    status = f"Continuation {'PASS' if summary['continuation']['passed'] else 'FAIL'}: {summary['continuation']['checks_passed']}/7"
    prefix = "SYNTHETIC FIXTURE - NOT RESULTS\n" if synthetic else ""
    legend = [Line2D([], [], linestyle="none", marker=m, color="#334155", label=str(seed), markersize=6)
              for seed, m in zip(SEEDS, ("o", "s", "^"), strict=True)]
    legend.append(Line2D([], [], linestyle="none", marker="|", color="black", label="Three-seed mean", markersize=13, markeredgewidth=2))

    def panel(ax, definition):
        key, _p, _g, _m, title, scale, unit, direction = definition
        data, points = values[key], []
        for i, arm in enumerate(ARMS):
            arm_data = data["arms"][arm]
            for seed, offset, marker in zip(SEEDS, (-.15, 0, .15), ("o", "s", "^"), strict=True):
                value = arm_data["seeds"][str(seed)]
                if value is not None:
                    points.append(value * scale)
                    ax.scatter(value * scale, i + offset, marker=marker, color=COLORS[i], s=43, zorder=3)
            mean = arm_data["mean"]
            if mean is not None:
                ax.plot(mean * scale, i, marker="|", color="black", markersize=19, markeredgewidth=1.7, zorder=4)
            else:
                ax.text(.5, i, "undefined", transform=ax.get_yaxis_transform(), ha="center")
        if points:
            low, high = min(points), max(points)
            margin = max((high - low) * .16, .08 if scale == 100 else .001)
            ax.set_xlim(max(0, low - margin), high + margin)
        ax.set_yticks(range(4), LABELS)
        ax.set_ylim(3.45, -.45)
        ax.grid(axis="x", color="#dce2e8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"{unit}  |  {direction}")
        counts = data["support"]
        if len(counts) == 3:
            support = "n by stratum: " + " / ".join(f"{counts[s]:,}" for s in ("unmentioned_retention", "assigned_retention", "changed"))
        else:
            support = f"n = {next(iter(counts.values())):,}"
        ax.set_title(title + "\n" + support, loc="left", fontsize=11, pad=10)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    def decorate(fig, title):
        fig.suptitle(prefix + title, x=.06, y=.982, ha="left", fontsize=17, fontweight="bold")
        fig.text(.06, .900 if synthetic else .941, "Exposed official DEV | Autonomous normalized scalar memory | " + status, fontsize=10)
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .053), ncol=4, frameon=False)
        fig.text(.06, .038, "Dots are training seeds on the same DEV examples. Mean ticks are descriptive; no confidence intervals. Dot axes are zoomed.", fontsize=8.5)
        fig.text(.06, .021, "Macro = equal mean of unmentioned retention, assigned retention, and changed accuracy. No novel architecture or calibration claim.", fontsize=8.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, definition in zip(axes.flat, METRICS[:4], strict=True):
        panel(ax, definition)
    decorate(fig, "Observation learning: every arm and seed")
    fig.subplots_adjust(left=.20, right=.97, top=.845 if synthetic else .88, bottom=.18, wspace=.66, hspace=.67)
    for extension in ("png", "svg"):
        fig.savefig(out / f"observation.{extension}", metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.3), left=.20, right=.97,
                            top=.845 if synthetic else .88, bottom=.17, hspace=.64, wspace=.66)
    panel(fig.add_subplot(grid[0, 0]), METRICS[4])
    panel(fig.add_subplot(grid[0, 1]), METRICS[5])
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    ax.set_title("Seven fixed conditions: trainable + number words minus frozen + number words", loc="left", fontsize=11, pad=14)
    cells = []
    for check, definition in zip(summary["continuation"]["checks"], CONDITIONS, strict=True):
        _name, label, rule, _threshold, _comparison, scale, unit = definition
        if unit == "wins":
            shown = "undefined" if check["wins"] is None else f"{check['wins']} / 3"
            seeds = "Exact positive paired macro differences"
        else:
            shown = "undefined" if check["mean"] is None else f"{check['mean'] * scale:+.4f} {unit}"
            seed_values = check["paired_seed_values"]
            seeds = "undefined" if seed_values is None else " / ".join("NA" if v is None else f"{v * scale:+.4f}" for v in seed_values)
        cells.append([label, rule, seeds, shown, "PASS" if check["passed"] else "FAIL"])
    table = ax.table(cellText=cells, colLabels=["Condition", "Exact rule", "Seed deltas: 6901 / 6902 / 6903", "Mean / wins", "Status"],
                     cellLoc="left", colLoc="left", bbox=[-.18, 0, 1.18, .98], colWidths=[.19, .25, .29, .16, .08])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dce4")
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(weight="bold")
        elif column == 4:
            cell.set_text_props(color="#1d6b46" if cells[row - 1][4] == "PASS" else "#a43130", weight="bold")
    decorate(fig, "Retention and the original continuation conditions")
    for extension in ("png", "svg"):
        fig.savefig(out / f"retention-and-conditions.{extension}", metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)


def execute(args):
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source = {}

    def check():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        require(time.perf_counter() - started <= LIMITS["wall_seconds"] and peak <= LIMITS["rss_bytes"], "Figure wall/RSS cap")
        require(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Figure output cap")
        return peak

    def timeout(*_):
        raise TimeoutError("Figure wall cap")

    handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
    try:
        source = descriptor(Path(__file__).resolve())
        write(args.out / "started.json", {"version": VERSION, "request": request, "source": source, "limits": LIMITS})
        summary, bindings = authenticate(args)
        values = extract(summary)
        write(args.out / "plotted-values.json", {"synthetic": bool(args.synthetic), "metrics": values, "conditions": summary["continuation"]})
        render(summary, values, args.out, args.synthetic)
        for path, record in bindings.items():
            require(descriptor(path) == record, "Input identity unchanged through render")
        require(descriptor(Path(__file__).resolve()) == source, "Plot source unchanged")
        peak = check()
        names = ("started.json", "plotted-values.json", "observation.png", "observation.svg", "retention-and-conditions.png", "retention-and-conditions.svg")
        receipt = {"version": VERSION, "status": "completed", "synthetic": bool(args.synthetic), "request": request,
                   "source": source, "inputs": {str(p): v for p, v in bindings.items()}, "scope": SCOPE,
                   "plan_sha256": summary["plan_sha256"], "execution_completed_sha256": summary["execution_completed_sha256"],
                   "continuation_passed": summary["continuation"]["passed"], "seed_points": 72,
                   "files": {name: descriptor(args.out / name) for name in names},
                   "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak, "limits": LIMITS, "model_calls": 0}
        write(args.out / "receipt.json", receipt)
        check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request, "source": source,
                  "error_type": type(error).__name__, "error": str(error), "wall_seconds": time.perf_counter() - started})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original error
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps({"status": result["status"], "synthetic": result["synthetic"]}))
