"""Presentation-only figures from a complete externally pinned saved report."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
from pathlib import Path

VERSION = "dialogue-observation-figure-v2"
REPORT_VERSION = "dialogue-observation-report-v2"
AUDIT_VERSION = "dialogue-observation-independent-audit-v2"
FAILED_PIN = "41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f"
FAILED_MANIFEST = Path(__file__).resolve().parents[3] / "output/dialogue-observation-learning-v1/failed-scientific-publication-01/manifest.json"
CLOCK_PATH = Path(__file__).resolve().parents[3] / "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
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
SCOPE = ("Saved-report presentation only, requiring a completed independently audited V2 result. "
         "Report and audit aggregates/receipts are authenticated; displayed values are compared to the saved audit. "
         "No raw predictions, corpus, checkpoints, model, or audit computation is loaded or replayed. "
         "Official DEV is exposed development data; training seeds repeat the same examples, not independent "
         "test samples. No confidence intervals, new architecture, calibration, or world-model claim. "
         "V2 root execution time is suspend-inclusive; nested fit diagnostics use perf_counter and may exclude "
         "suspend. No cost is plotted. The failed V1 attempt remains separate, unsuccessful work.")


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
    audit = args.audit.resolve()
    receipt_path, summary_path = report / "receipt.json", report / "summary.json"
    audit_receipt_path, audit_summary_path = audit / "receipt.json", audit / "summary.json"
    for path, pin in ((receipt_path, args.receipt_sha256), (summary_path, args.summary_sha256),
                      (audit_receipt_path, args.audit_receipt_sha256), (audit_summary_path, args.audit_summary_sha256)):
        require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)
                and not path.is_symlink() and sha(path) == pin, "External report and audit pins")
    receipt, audit_receipt = read(receipt_path), read(audit_receipt_path)
    require(receipt["status"] == "completed" and receipt["version"] == REPORT_VERSION
            and receipt["technical_validity_passed"] is True and receipt["model_calls"] == 0,
            "Completed technically valid saved-only report")
    require(audit_receipt["status"] == "completed" and audit_receipt["version"] == AUDIT_VERSION
            and audit_receipt["agreement"] is True and audit_receipt["model_calls"] == 0
            and audit_receipt["fit_cells"] == 144 and audit_receipt["literal_cells"] == 24
            and type(audit_receipt["scalar_checks"]) is int and audit_receipt["scalar_checks"] > 0,
            "Successful complete independent audit")
    require(audit_receipt["producer_receipt_sha256"] == args.receipt_sha256
            and audit_receipt["producer_summary_sha256"] == args.summary_sha256
            and all(audit_receipt[k] == receipt[k] for k in (
                "plan_sha256", "execution_completed_sha256", "launch_sha256", "terminal_sha256",
                "continuation_passed", "scientific_checks_passed")), "Audit binds this exact completed production report")
    bindings = {}
    for folder, record, names in ((report, receipt, {"started.json", "summary.json", "report.md"}),
                                   (audit, audit_receipt, {"started.json", "summary.json"})):
        paths = list(folder.rglob("*"))
        require(not folder.is_symlink() and not any(p.is_symlink() for p in paths)
                and set(record["files"]) == names
                and {p.relative_to(folder).as_posix() for p in paths if p.is_file()} == names | {"receipt.json"},
                "Exact report/audit membership")
        bindings[folder / "receipt.json"] = descriptor(folder / "receipt.json")
        for name in names:
            path = folder / name
            require(descriptor(path) == record["files"][name], "Report/audit payload identity: " + name)
            bindings[path] = descriptor(path)
    require(all(v.get("synthetic", False) is bool(args.synthetic) for v in (receipt, audit_receipt)), "Explicit synthetic receipts")
    prior = failed_history(args.failed_manifest)
    bindings[args.failed_manifest.resolve()] = descriptor(args.failed_manifest)
    # Complete receipts, payloads and prior failure identity precede either quality decode.
    summary, audited = read(summary_path), read(audit_summary_path)
    require(summary["version"] == REPORT_VERSION and summary["status"] == "completed"
            and summary["technical_validity_passed"] is True and summary["technical_complete_fits"] == 12,
            "All twelve fits technically complete")
    require(all(summary[k] == receipt[k] for k in ("plan_sha256", "execution_completed_sha256")), "Report lineage join")
    require(audited["version"] == AUDIT_VERSION and audited["status"] == "completed" and audited["agreement"] is True
            and all(audited[k] == summary[k] for k in ("plan_sha256", "execution_completed_sha256"))
            and all(audited[k] == audit_receipt[k] for k in ("fit_cells", "literal_cells", "scalar_checks")), "Audit summary identity")
    require(all(v.get("synthetic", False) is bool(args.synthetic) for v in (summary, audited)), "Explicit synthetic scope")
    require(set(summary["fits"]) == {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}, "Exact twelve-fit membership")
    require(set(audited["fits"]) == set(summary["fits"]), "All twelve audited fits")
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
    compare(extract(audited), extract(summary))
    for field in ("checks_total", "checks_passed", "passed"):
        require(audited["continuation"][field] == gate[field], "Audited condition totals")
    require(len(audited["continuation"]["checks"]) == 7, "Seven independently audited conditions")
    for actual, expected in zip(gate["checks"], audited["continuation"]["checks"], strict=True):
        keys = ("name", "passed", "wins", "required") if "wins" in expected else (
            "name", "passed", "threshold", "comparison", "mean", "paired_seed_values")
        compare({k: expected[k] for k in keys}, {k: actual[k] for k in keys})
    return summary, bindings, prior


def failed_history(path):
    require(not path.is_symlink() and sha(path) == FAILED_PIN, "Fixed failed V1 manifest identity")
    record = read(path)
    require(record["status"] == "failed_technical_timing" and record["required_fits"] == 12
            and all(record[k] is False for k in ("quality_metrics_opened", "individual_predictions_decoded",
                                                "weights_loaded", "resume_permitted", "partial_scoring_permitted")),
            "Prior attempt remains failed and unscored")
    return {"manifest_sha256": FAILED_PIN, "status": record["status"],
            "completed_fit_records": [{k: fit[k] for k in ("fit_id", "counts", "wall_seconds")}
                                      for fit in record["completed_fits"]],
            "incomplete_fit": record["incomplete_fit"],
            "parent_monotonic_seconds": record["parent_monotonic_seconds"],
            "civil_elapsed_seconds": record["civil_elapsed_seconds"],
            "worker_perf_counter_seconds": record["worker_perf_counter_seconds"],
            "scope": "Preserved failed V1 work only. Civil and monotonic/performance-counter readings are distinct, not summed, and none is successful V2 execution time. No referenced old artifact is opened."}


def compare(expected, actual):
    """Compare saved displayed values; never recompute or relax scientific decisions."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), "Audit display key agreement")
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), "Audit display list agreement")
        for a, b in zip(expected, actual, strict=True):
            compare(a, b)
    elif type(expected) is float:
        require(finite(actual) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), "Audit displayed float agreement")
    else:
        require(type(actual) is type(expected) and actual == expected, "Audit exact count/status agreement")


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
        fig.text(.06, .023, "Macro = equal mean of unmentioned retention, assigned retention, and changed accuracy. No novel architecture or calibration claim.", fontsize=8.5)
        fig.text(.06, .008, "V1's failed timing attempt is preserved separately and is not successful V2 work. These figures contain no timing comparison.", fontsize=8.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, definition in zip(axes.flat, METRICS[:4], strict=True):
        panel(ax, definition)
    decorate(fig, "Observation learning V2: every arm and seed")
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


def load_clock():
    require(sha(CLOCK_PATH) == CLOCK_PIN, "Qualified suspend clock source")
    spec = importlib.util.spec_from_file_location("observation_figure_suspend_clock", CLOCK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SuspendClock


def timing(clock, deadline, *, terminal=False, failure=False):
    result = {"clock_backend": None if clock is None else clock.backend,
              "started_ns": None if deadline is None else deadline.started_ns,
              "deadline_ns": None if deadline is None else deadline.expires_ns,
              "finished_ns": None, "elapsed_ns": None, "wall_seconds": None,
              "timing_available": clock is not None and deadline is not None}
    if terminal and result["timing_available"]:
        try:
            end = clock.now_ns()
            require(end >= deadline.started_ns, "Terminal clock precedes start")
            elapsed = end - deadline.started_ns
            result.update(finished_ns=end, elapsed_ns=elapsed, wall_seconds=elapsed / 1e9)
        except BaseException as error:
            if not failure:
                raise
            result.update(timing_available=False, timing_error=repr(error))
    return result


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source = {}
    clock = deadline = handler = None

    def check():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        deadline.check()
        require(peak <= LIMITS["rss_bytes"], "Figure RSS cap")
        require(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Figure output cap")
        return peak

    def timeout(*_):
        # Never read the locking native clock from a signal handler.
        raise TimeoutError("Supplementary awake-time figure timeout")

    try:
        clock = load_clock()()
        deadline = clock.deadline_after(LIMITS["wall_seconds"])
        handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        source = descriptor(Path(__file__).resolve())
        write(args.out / "started.json", {"version": VERSION, "request": request, "source": source, "limits": LIMITS,
              "clock_source_sha256": CLOCK_PIN, **timing(clock, deadline)})
        summary, bindings, prior = authenticate(args)
        values = extract(summary)
        write(args.out / "plotted-values.json", {"synthetic": bool(args.synthetic), "metrics": values, "conditions": summary["continuation"]})
        render(summary, values, args.out, args.synthetic)
        for path, record in bindings.items():
            require(descriptor(path) == record, "Input identity unchanged through render")
        require(descriptor(Path(__file__).resolve()) == source and sha(CLOCK_PATH) == CLOCK_PIN, "Plot/clock sources unchanged")
        peak = check()
        names = ("started.json", "plotted-values.json", "observation.png", "observation.svg", "retention-and-conditions.png", "retention-and-conditions.svg")
        receipt = {"version": VERSION, "status": "completed", "synthetic": bool(args.synthetic), "request": request,
                   "source": source, "inputs": {str(p): v for p, v in bindings.items()}, "scope": SCOPE,
                   "plan_sha256": summary["plan_sha256"], "execution_completed_sha256": summary["execution_completed_sha256"],
                   "independent_audit_receipt_sha256": args.audit_receipt_sha256,
                   "independent_audit_summary_sha256": args.audit_summary_sha256,
                   "preserved_failed_v1": prior,
                   "continuation_passed": summary["continuation"]["passed"], "seed_points": 72,
                   "files": {name: descriptor(args.out / name) for name in names},
                   "clock_source_sha256": CLOCK_PIN, **timing(clock, deadline, terminal=True),
                   "wall_scope": "Suspend-inclusive elapsed from native deadline initialization through payload hashing; completion publication is checked against the same strict deadline. Clock-source authentication precedes initialization.",
                   "peak_rss_bytes": peak, "limits": LIMITS, "model_calls": 0}
        check()
        write(args.out / "receipt.json", receipt)
        check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request, "source": source,
                  "clock_source_sha256": CLOCK_PIN, "error_type": type(error).__name__, "error": str(error),
                  **timing(clock, deadline, terminal=True, failure=True)})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original error
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--audit-summary-sha256", required=True)
    parser.add_argument("--failed-manifest", type=Path, default=FAILED_MANIFEST,
                        help="Preserved V1 metadata only; must match the fixed published SHA256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps({"status": result["status"], "synthetic": result["synthetic"]}))
