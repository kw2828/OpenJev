"""Render an authenticated completed Qwen report; never open prediction arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

VERSION = "dialogue-qwen-observation-figure-v1"
REPORT_VERSION = "dialogue-qwen-observation-report-v1"
LIMITS = {"wall_seconds": 60, "rss_bytes": 1024**3, "output_bytes": 32*1024**2}
SEEDS = (6201, 6202, 6203)
ARMS = ("current", "history4")
SUPPORT = {"all": 7819, "changed": 578, "retained": 7241, "unmentioned_retention": 4032, "assigned_retention": 3209}
PANELS = (("changed", "accuracy", "Changed-state accuracy", "Higher is better", 100., "%"),
          ("retained", "error", "Retained-state error", "Lower is better", 100., "%"),
          ("all", "nll", "Overall log loss", "Lower is better", 1., "nats"),
          ("all", "brier", "Overall Brier score", "Lower is better", 1., "squared probability error"))
SCOPE = ("Descriptive rendering of an authenticated main report, not an independent numerical audit. "
         "Exposed official TRAIN; correct previous value supplied. Qwen has one frozen inference per arm; "
         "the three historical marks are optimizer seeds of earlier corrected-flat fits on the same rows. "
         "No architecture, autonomous-memory or calibration claim.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def item(path):
    return {"sha256": digest(path), "bytes": Path(path).stat().st_size}


def pin(value):
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def read(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON scalar: "+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def authenticate(args):
    summary_path, receipt_path = Path(args.summary).resolve(), Path(args.receipt).resolve()
    require(summary_path.name == "summary.json" and receipt_path.name == "receipt.json"
            and summary_path.parent == receipt_path.parent, "One completed report directory")
    require(pin(args.summary_sha256) and pin(args.receipt_sha256)
            and digest(summary_path) == args.summary_sha256 and digest(receipt_path) == args.receipt_sha256,
            "External summary/receipt pins")
    receipt = read(receipt_path)
    require(receipt["status"] == "completed" and receipt["version"] == REPORT_VERSION
            and receipt["technical_validity_passed"] is True
            and receipt["model_calls"] == receipt["encoder_calls"] == receipt["checkpoint_deserializations"] == 0,
            "Successful saved-only report")
    root, bindings = receipt_path.parent, {receipt_path: item(receipt_path)}
    require(set(receipt["files"]) == {"started.json", "summary.json", "report.md"}
            and {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            == set(receipt["files"]) | {"receipt.json"}, "Exact report closure")
    for name, descriptor in receipt["files"].items():
        path = root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root)
                and item(path) == descriptor, "Report payload: "+name)
        bindings[path] = descriptor
    summary = read(summary_path)
    require(summary["status"] == "completed" and summary["version"] == REPORT_VERSION
            and summary["technical_validity_passed"] is True
            and pin(summary["execution_completed_sha256"]) and pin(summary["plan_sha256"])
            and summary["execution_completed_sha256"] == receipt["execution_completed_sha256"]
            and summary["plan_sha256"] == receipt["plan_sha256"]
            and summary["source_sha256"] == receipt["study_source_sha256"], "Report identity joins")
    require(summary.get("synthetic", False) is bool(args.synthetic)
            and receipt.get("synthetic", False) is bool(args.synthetic), "Explicit matching synthetic scope")
    require(summary["support"] == SUPPORT and summary["counts"]["rows"] == 7819
            and summary["counts"]["decisions"] == 15638 and set(summary["arms"]) == set(ARMS)
            and len(summary["services"]) == len(set(summary["services"])) == 6, "Complete fixed cohort")
    require(set(summary["historical"]["corrected_flat"]["seeds"]) == {str(s) for s in SEEDS}, "All historical seeds")
    for name in ("semantic_strength", "added_history"):
        for kind in ("behavioral", "proper_score_nonregression"):
            rule = summary["decisions"][name][kind]
            require(len(rule["checks"]) == 4 and all(type(c["passed"]) is bool for c in rule["checks"].values())
                    and rule["passed"] is all(c["passed"] for c in rule["checks"].values()), "Separate four-check conclusion")
    return summary, bindings


def plotted_values(summary):
    result = {}
    historical = summary["historical"]["corrected_flat"]
    for stratum, metric, title, direction, scale, unit in PANELS:
        for weighting in ("row", "equal_service"):
            def value(fit, stratum=stratum, metric=metric, weighting=weighting):
                cell = fit["cells"][stratum]
                v = cell["metrics"][metric][weighting]
                require(cell["rows"] == SUPPORT[stratum] and type(v) in (float, int) and math.isfinite(v)
                        and 0 <= v and (metric == "nll" or v <= (2 if metric == "brier" else 1)), "Finite supported metric")
                return v
            seeds = [value(historical["seeds"][str(s)]) for s in SEEDS]
            mean = math.fsum(seeds)/3
            require(math.isclose(mean, value(historical["mean"]), rel_tol=1e-12, abs_tol=1e-12), "Historical mean identity")
            result[f"{stratum}/{metric}/{weighting}"] = {
                "stratum": stratum, "metric": metric, "weighting": weighting, "rows": SUPPORT[stratum],
                "title": title, "direction": direction, "scale": scale, "unit": unit,
                "qwen": {arm: value(summary["arms"][arm]) for arm in ARMS},
                "historical_seed_order": list(SEEDS), "historical_values": seeds, "historical_mean": mean}
    return result


def render(summary, data, out, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(12, 11.4), facecolor="white")
    for row, (stratum, metric, title, direction, _, _) in enumerate(PANELS):
        for col, weighting in enumerate(("row", "equal_service")):
            ax, panel = axes[row, col], data[f"{stratum}/{metric}/{weighting}"]
            scale = panel["scale"]
            qwen = [panel["qwen"][arm]*scale for arm in ARMS]
            seeds = [v*scale for v in panel["historical_values"]]
            mean = panel["historical_mean"]*scale
            ax.axhspan(1.5, 2.45, color="#f0f2f4", zorder=0)
            ax.scatter(qwen, [0, 1], s=56, c=["#176d9c", "#b45627"], marker="D", zorder=4)
            for i, (v, marker) in enumerate(zip(seeds, ("o", "s", "^"), strict=True)):
                ax.scatter(v, 2+(i-1)*.2, s=35, marker=marker, color="#687786", zorder=3)
            ax.plot([mean, mean], [1.65, 2.35], color="#192e41", lw=2, zorder=2)
            values = qwen+seeds+[mean]
            low, high = min(values), max(values)
            margin = max(.01 if scale == 100 else .0001, .12*(high-low))
            ax.set_xlim(max(0., low-margin), high+margin)
            ax.set_ylim(2.5, -.5)
            ax.set_yticks([0, 1, 2], ["Qwen current", "Qwen history4", "Historical flat"] if col == 0 else [""]*3)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(axis="x", color="#d9e0e6", lw=.7)
            ax.set_axisbelow(True)
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.set_title(f"{title} | {'Row' if col == 0 else 'Equal service'}", loc="left", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{panel['unit']}; {direction.lower()}", fontsize=9)
            if synthetic:
                ax.text(.5, .5, "SYNTHETIC", transform=ax.transAxes, ha="center", va="center", fontsize=21,
                        rotation=15, color="#a63737", alpha=.2, fontweight="bold")
    status = []
    for name, label in (("semantic_strength", "Current vs historical flat"), ("added_history", "History vs current")):
        rule = summary["decisions"][name]
        status.append(label+": behavior "+("PASS" if rule["behavioral"]["passed"] else "FAIL")+
                      "; proper scores "+("PASS" if rule["proper_score_nonregression"]["passed"] else "FAIL"))
    fig.suptitle(("SYNTHETIC LAYOUT ONLY\n" if synthetic else "")+"Qwen observation baseline with a correct previous value",
                 x=.04, y=.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(.04, .952, "Exposed official TRAIN | 578 changed and 7,241 retained rows per arm | Zoomed metric axes", fontsize=10)
    fig.text(.04, .938, "\n".join(status), fontsize=9, linespacing=1.5, va="top")
    handles = [Line2D([], [], linestyle="none", marker=m, color="#687786", label=f"Historical seed {s}")
               for s, m in zip(SEEDS, ("o", "s", "^"), strict=True)]
    handles.append(Line2D([], [], linestyle="none", marker="|", markersize=15, markeredgewidth=2,
                          color="#192e41", label="Historical mean"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .04), ncol=4, frameon=False, fontsize=9)
    fig.text(.04, .021, "Historical points are three earlier corrected-flat fits, not repeated Qwen runs. "
             "No probability-calibration or architecture claim.", fontsize=9)
    fig.subplots_adjust(left=.18, right=.985, top=.88, bottom=.115, wspace=.16, hspace=.66)
    fig.savefig(out/"comparison.png", dpi=180, facecolor="white")
    fig.savefig(out/"comparison.pdf", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def peak_rss():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == "darwin" else n*1024)


def execute(args):
    out = Path(args.out).resolve()
    parents = {Path(args.summary).resolve().parent, Path(args.receipt).resolve().parent}
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in parents), "Separate figure directory")
    out.mkdir(parents=True, exist_ok=False)
    started, previous, source = time.monotonic(), None, {}
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    def check():
        require(time.monotonic()-started <= LIMITS["wall_seconds"], "Figure wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Figure RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Figure storage cap")
    def timeout(*_):
        raise TimeoutError("Figure wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing timer")
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        source = {Path(__file__).name: digest(__file__)}
        write(out/"started.json", {"version": VERSION, "request": request, "limits": LIMITS, "source_sha256": source})
        summary, bindings = authenticate(args)
        data = plotted_values(summary)
        check()
        write(out/"plotted-values.json", {"version": VERSION, "synthetic": bool(args.synthetic), "scope": SCOPE,
              "panels": data, "decisions": summary["decisions"], "costs": summary["costs"]})
        render(summary, data, out, args.synthetic)
        for path, descriptor in bindings.items():
            require(item(path) == descriptor, "End report identity")
        require(digest(__file__) == source[Path(__file__).name], "End plotting source identity")
        check()
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "scope": SCOPE,
              "synthetic": bool(args.synthetic), "request": request, "source_sha256": source,
              "summary_sha256": args.summary_sha256, "report_receipt_sha256": args.receipt_sha256,
              "execution_completed_sha256": summary["execution_completed_sha256"], "plan_sha256": summary["plan_sha256"],
              "input_members": {str(p): v for p, v in bindings.items()}, "limits": LIMITS,
              "files": {n: item(out/n) for n in ("started.json", "plotted-values.json", "comparison.png", "comparison.pdf")},
              "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": peak_rss(),
              "model_calls": 0, "prediction_reads": 0, "no_retry": True})
        check()
    except BaseException as error:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                  "source_sha256": source, "error": repr(error), "wall_seconds": time.monotonic()-started,
                  "model_calls": 0, "prediction_reads": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "summary-sha256", "receipt", "receipt-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--synthetic", action="store_true", help="Both source JSON files must explicitly be synthetic")
    execute(parser.parse_args())
