"""Plot all fixed objective-study readouts from an authenticated report only.

No prediction, checkpoint, feature, encoder, corpus or producer imports. Two
figures show every seed under row and equal-service weighting. Historical fits
remain explicitly historical. --synthetic accepts only marked artificial input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

VERSION = "dialogue-objective-figure-v1"
REPORT_VERSION = "dialogue-objective-v1"
SEEDS = (6201, 6202, 6203)
FRESH = ("stratum-original", "stratum-corrected", "uniform-original", "uniform-reweighted")
HISTORICAL = tuple(f"{m}-{r}" for m in ("flat_stratum", "token_mean", "token_aligned") for r in ("original", "corrected"))
CATEGORIES = FRESH+HISTORICAL
LABELS = ("Stratum / original", "Stratum / corrected", "Uniform / original", "Uniform / reweighted",
          "Flat / original", "Flat / corrected", "Token mean / original", "Token mean / corrected",
          "Aligned / original", "Aligned / corrected")
SUPPORT = {"all": 7819, "changed": 578, "retained": 7241, "unmentioned_retention": 4032, "assigned_retention": 3209}
METRICS = ("accuracy", "error", "wrong_selected_branch", "wrong_value", "nll", "brier")
WEIGHTINGS = ("row", "equal_service", "equal_dialogue")
CHECKS = {"objective_changed_accuracy_row", "objective_changed_accuracy_equal_service",
          "objective_retained_error_row", "objective_retained_error_equal_service", "objective_nll_row",
          "objective_nll_equal_service", "objective_joint_seeds", "practical_accuracy_row",
          "practical_accuracy_equal_service", "practical_nll_row", "practical_nll_equal_service",
          "practical_true_false_positive", "practical_joint_seeds"}
COLORS, MARKERS = ("#246b9e", "#bd7134", "#337d64"), ("o", "s", "^")
PANELS = {"decisions": (("changed", "accuracy", "Changed-state accuracy", True),
                        ("retained", "error", "Retained-state error", False)),
          "overall": (("all", "accuracy", "Overall accuracy", True),
                      ("all", "nll", "Overall log loss", False))}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value): raise ValueError("Nonfinite JSON scalar: "+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def number(value, name, low=0., high=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= low
            and (high is None or value <= high), "Invalid "+name)
    return value


def authenticate(args):
    summary_path, receipt_path = Path(args.summary).resolve(), Path(args.receipt).resolve()
    require(summary_path.name == "summary.json" and receipt_path.name == "receipt.json"
            and summary_path.parent == receipt_path.parent, "One completed report directory")
    require(pin(args.summary_sha256) and pin(args.receipt_sha256) and pin(args.run_sha256)
            and digest(summary_path) == args.summary_sha256 and digest(receipt_path) == args.receipt_sha256,
            "External summary/receipt identities")
    receipt = read(receipt_path)
    require(receipt["status"] == "completed" and receipt["version"] == REPORT_VERSION
            and receipt["technical_validity_passed"] is True, "Successful report required")
    root = receipt_path.parent
    require(set(receipt["files"]) == {"started.json", "summary.json", "report.md"}
            and {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            == set(receipt["files"]) | {"receipt.json"}, "Exact report payload closure")
    for name, item in receipt["files"].items():
        path = root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.stat().st_size == item["bytes"]
                and digest(path) == item["sha256"], "Report payload identity: "+name)
    require(receipt["files"]["summary.json"]["sha256"] == args.summary_sha256, "Receipt binds requested summary")
    summary = read(summary_path)
    require(summary["status"] == "completed" and summary["version"] == REPORT_VERSION
            and summary["technical_validity_passed"] is True
            and summary["execution_completed_sha256"] == receipt["execution_completed_sha256"] == args.run_sha256
            and pin(summary["plan_sha256"]) and summary["plan_sha256"] == receipt["plan_sha256"], "Execution/plan joins")
    require(summary.get("synthetic", False) is bool(args.synthetic)
            and receipt.get("synthetic", False) is bool(args.synthetic), "Explicit matching synthetic scope")
    validate(summary, receipt)
    return summary


def validate(summary, receipt):
    require(set(summary["fits"]) == {f"{c}-{s}" for c in FRESH for s in SEEDS}
            and set(summary["means"]) == set(FRESH), "All twelve new readouts")
    require(set(summary["historical"]["fits"]) == {f"{c}-{s}" for c in HISTORICAL for s in SEEDS}
            and set(summary["historical"]["means"]) == set(HISTORICAL), "All eighteen historical readouts")
    for category in CATEGORIES:
        source = summary if category in FRESH else summary["historical"]
        for seed in SEEDS:
            fit = source["fits"][f"{category}-{seed}"]
            for stratum, support in SUPPORT.items():
                cell = fit["cells"]["heldout_service/"+stratum]
                require(type(cell["rows"]) is int and cell["rows"] == support, "Fixed primary stratum rows")
                counts = cell["counts"]
                require(set(counts) == {"correct", "error", "wrong_selected_branch", "wrong_value"}
                        and all(type(v) is int and 0 <= v <= support for v in counts.values())
                        and counts["correct"]+counts["error"] == support
                        and counts["wrong_selected_branch"]+counts["wrong_value"] == counts["error"], "Decision partition")
                require(set(cell["metrics"]) == set(METRICS), "All declared metrics")
                for metric in METRICS:
                    require(set(cell["metrics"][metric]) == set(WEIGHTINGS), "All declared metric weightings")
                    for weighting, value in cell["metrics"][metric].items():
                        number(value, category+" "+metric+" "+weighting, high=None if metric == "nll" else 2 if metric == "brier" else 1)
                    if metric in ("accuracy", "error", "wrong_selected_branch", "wrong_value"):
                        key = "correct" if metric == "accuracy" else metric
                        require(math.isclose(cell["metrics"][metric]["row"], counts[key]/support, rel_tol=1e-12, abs_tol=1e-12), "Rate/count identity")
            require(sum(fit["cells"]["heldout_service/"+s]["counts"]["correct"] for s in ("changed", "retained"))
                    == fit["cells"]["heldout_service/all"]["counts"]["correct"], "Changed/retained count closure")
        for stratum in SUPPORT:
            for metric in METRICS:
                for weighting in WEIGHTINGS:
                    value = math.fsum(source["fits"][f"{category}-{s}"]["cells"]["heldout_service/"+stratum]["metrics"][metric][weighting] for s in SEEDS)/3
                    claimed = source["means"][category]["cells"]["heldout_service/"+stratum]["metrics"][metric][weighting]
                    require(type(claimed) in (int, float) and math.isfinite(claimed)
                            and math.isclose(value, claimed, rel_tol=1e-12, abs_tol=1e-12), "Equal-three-seed summary mean")
    rule = summary["continuation"]
    require(set(rule["checks"]) == CHECKS and rule["total_checks"] == 13
            and all(type(c["passed"]) is bool for c in rule["checks"].values())
            and rule["checks_passed"] == sum(c["passed"] for c in rule["checks"].values())
            and rule["objective_passed"] is all(v["passed"] for k, v in rule["checks"].items() if k.startswith("objective_"))
            and rule["practical_passed"] is all(v["passed"] for k, v in rule["checks"].items() if k.startswith("practical_"))
            and rule["passed"] is (rule["objective_passed"] and rule["practical_passed"])
            and receipt["continuation_passed"] is rule["passed"], "Fixed thirteen-check status")
    costs = summary["costs"]
    require(set(costs["per_fit"]) == {f"{m}-{s}" for m in ("stratum", "uniform") for s in SEEDS}, "Six trained fits, twelve readouts")
    whole = number(costs["whole_wall_seconds"], "whole wall", high=6000)
    number(costs["process_lifetime_peak_rss_bytes"], "peak RSS", high=6*1024**3)
    for cost in costs["per_fit"].values():
        train, evaluation, total = (number(cost[k], k) for k in ("training_wall_seconds", "evaluation_wall_seconds", "wall_seconds"))
        require(train > 0 and evaluation > 0 and train+evaluation <= total <= whole, "Nested fit cost")
    require(math.fsum(c["wall_seconds"] for c in costs["per_fit"].values()) <= whole, "Whole versus nested fit time")


def plotted_values(summary, synthetic):
    panels = {}
    for figure, specs in PANELS.items():
        for stratum, metric, _, _ in specs:
            for weighting in ("row", "equal_service"):
                rows = {}
                for category, label in zip(CATEGORIES, LABELS, strict=True):
                    source = summary if category in FRESH else summary["historical"]
                    values = [source["fits"][f"{category}-{s}"]["cells"]["heldout_service/"+stratum]["metrics"][metric][weighting] for s in SEEDS]
                    scale = 1. if metric == "nll" else 100.
                    rows[category] = {"label": label, "origin": "fresh" if category in FRESH else "historical",
                        "seed_order": list(SEEDS), "values": values, "mean": math.fsum(values)/3,
                        "display_values": [v*scale for v in values], "display_mean": scale*math.fsum(values)/3,
                        "row_counts": [source["fits"][f"{category}-{s}"]["cells"]["heldout_service/"+stratum]["counts"] for s in SEEDS]}
                key = f"{stratum}/{metric}/{weighting}"
                panels[key] = {"figure": figure, "stratum": stratum, "rows": SUPPORT[stratum], "metric": metric,
                               "weighting": weighting, "unit": "nats" if metric == "nll" else "percent", "categories": rows}
    return {"version": VERSION, "synthetic": synthetic, "category_order": list(CATEGORIES), "panels": panels,
            "continuation": summary["continuation"], "costs": summary["costs"],
            "scope": "All 30 fixed method/readout/seed cells, under two weightings. Points are optimizer seeds on the same exposed TRAIN rows, not independent data replicates."}


def render(summary, plotted, out, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    rule = summary["continuation"]
    status = "PASS" if rule["passed"] else "FAIL"
    status_line = (f"Continuation {status}: {rule['checks_passed']}/13 checks  |  "
                   f"Objective {'PASS' if rule['objective_passed'] else 'FAIL'}  |  Practical {'PASS' if rule['practical_passed'] else 'FAIL'}")
    for figure, specs in PANELS.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 13.4), facecolor="white")
        for row, (stratum, metric, title, higher) in enumerate(specs):
            for column, weighting in enumerate(("row", "equal_service")):
                ax = axes[row, column]; data = plotted["panels"][f"{stratum}/{metric}/{weighting}"]
                ax.axhspan(-.55, 3.5, facecolor="#eaf3fa", zorder=0)
                ax.axhspan(3.5, 9.55, facecolor="#f3f4f5", zorder=0)
                all_values = []
                for y, category in enumerate(CATEGORIES):
                    values = data["categories"][category]["display_values"]
                    for j, value in enumerate(values):
                        ax.scatter(value, y+(j-1)*.20, marker=MARKERS[j], color=COLORS[j], s=38, zorder=3,
                                   edgecolors="white", linewidths=.4)
                    mean = data["categories"][category]["display_mean"]
                    ax.plot([mean, mean], [y-.28, y+.28], color="#243444", lw=1.7, zorder=2)
                    all_values.extend(values)
                low, high = min(all_values), max(all_values)
                margin = max(.015 if metric == "nll" else .8, .12*(high-low))
                left, right = max(0., low-margin), high+margin
                if metric == "error": left = 0.
                if metric != "nll": right = min(100., right)
                if left == right: right = left+(1 if metric != "nll" else .1)
                ax.set_xlim(left, right); ax.set_ylim(9.6, -.6)
                ax.set_yticks(range(10), LABELS if column == 0 else [""]*10)
                ax.tick_params(axis="y", length=0, labelsize=10)
                ax.tick_params(axis="x", labelsize=9)
                ax.axhline(3.5, color="#9baab7", lw=.9)
                ax.grid(axis="x", color="#d5dee6", lw=.7); ax.set_axisbelow(True)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.set_xlabel(("Percent" if metric != "nll" else "Nats")+f"; {'higher' if higher else 'lower'} is better")
                panel = chr(65+row*2+column)
                ax.set_title(f"{panel}  {title}\n{'Row weighting' if weighting == 'row' else 'Equal-service weighting'}", loc="left", pad=12)
                if synthetic:
                    ax.text(.5, .5, "SYNTHETIC", transform=ax.transAxes, ha="center", va="center",
                            rotation=25, fontsize=33, weight="bold", alpha=.065, color="#9c3434", zorder=1)
        prefix = "SYNTHETIC FIXTURE ONLY | " if synthetic else ""
        main = "Objective choice: change versus retention" if figure == "decisions" else "Objective choice: overall decisions and log loss"
        fig.suptitle(prefix+main, x=.06, y=.98, ha="left", fontsize=17 if synthetic else 19, weight="bold",
                     color="#8b3030" if synthetic else "#172b3b")
        fig.text(.06, .946, ("Artificial data and rule status, not an empirical result. " if synthetic else "")+status_line, fontsize=11)
        handles = [Line2D([0], [0], color=COLORS[j], marker=MARKERS[j], linestyle="none", label=f"Seed {seed}") for j, seed in enumerate(SEEDS)]
        handles += [Line2D([0], [0], color="#243444", marker="|", linestyle="none", markersize=13, label="Three-seed mean"),
                    Patch(facecolor="#eaf3fa", edgecolor="#9baab7", label="Fresh: 6 fits, 12 readouts"),
                    Patch(facecolor="#f3f4f5", edgecolor="#9baab7", label="Historical: 9 fits, 18 readouts")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .927), ncol=6, frameon=False, fontsize=9,
                   handletextpad=.5, columnspacing=1.25)
        fig.text(.06, .104, "Primary support per seed: 578 changed + 7,241 retained = 7,819 rows. Every category and all three seeds are shown.", fontsize=10)
        fig.text(.06, .080, "Axes zoom to plotted values; error axes start at zero. Historical fits are separate development references, not contemporaneous controls.", fontsize=9.5, color="#485a6c")
        fig.text(.06, .056, "Exposed official TRAIN, correct previous value supplied. A training-objective/readout comparison, not architecture or calibration evidence.", fontsize=9.5, color="#485a6c")
        costs = summary["costs"]
        fig.text(.06, .032, ("Synthetic cost placeholders: " if synthetic else "Fresh campaign cost: ")+
                 f"{costs['whole_wall_seconds']/60:.2f} min whole / 100 min cap; {costs['process_lifetime_peak_rss_bytes']/1024**3:.2f} GiB lifetime peak RSS. "
                 "Training/evaluation times remain nested, not added again.", fontsize=9.5, color="#485a6c")
        fig.subplots_adjust(left=.18, right=.97, top=.855, bottom=.17, hspace=.32, wspace=.13)
        for suffix in ("png", "pdf"): fig.savefig(out/f"objective-{figure}.{suffix}", dpi=180, facecolor="white")
        plt.close(fig)


def execute(args):
    out = Path(args.out).resolve(); inputs = (Path(args.summary).resolve(), Path(args.receipt).resolve())
    require(all(not out.is_relative_to(p.parent) and not p.parent.is_relative_to(out) for p in inputs), "Separate figure output")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); source_pin = None
    try:
        source_pin = digest(__file__)
        summary = authenticate(args)
        plotted = plotted_values(summary, bool(args.synthetic))
        write(out/"plotted-values.json", plotted)
        render(summary, plotted, out, bool(args.synthetic))
        require(digest(args.summary) == args.summary_sha256 and digest(args.receipt) == args.receipt_sha256
                and digest(__file__) == source_pin, "Final source/input identity")
        files = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        require(time.monotonic()-start <= 60 and sum(i["bytes"] for i in files.values()) <= 32*1024**2, "Figure wall/output bounds")
        write(out/"receipt.json", {"version": VERSION, "status": "completed", "synthetic": bool(args.synthetic),
            "source_sha256": source_pin, "summary_path": str(Path(args.summary).resolve()), "summary_sha256": args.summary_sha256,
            "report_receipt_path": str(Path(args.receipt).resolve()), "report_receipt_sha256": args.receipt_sha256,
            "execution_completed_sha256": args.run_sha256, "plan_sha256": summary["plan_sha256"],
            "continuation_passed": summary["continuation"]["passed"], "fresh_readouts": 12, "historical_readouts": 18,
            "panels": 8, "seed_marks": 240, "files": files, "wall_seconds": time.monotonic()-start,
            "model_calls": 0, "prediction_arrays_read": 0,
            "scope": ("Conspicuously synthetic layout fixture, not a scientific result. " if args.synthetic else "")+
                "Externally authenticated report visualization only. No independent re-audit of metrics or execution; no arrays, weights or corpus read. All category means computed from the three displayed seed values."})
    except BaseException as error:
        try:
            write(out/"failed.json", {"version": VERSION, "status": "failed", "source_sha256": source_pin,
                "synthetic": bool(args.synthetic), "error": repr(error), "wall_seconds": time.monotonic()-start,
                "model_calls": 0, "prediction_arrays_read": 0})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            if callable(getattr(error, "add_note", None)): error.add_note("Figure failure receipt: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", help="Require marked artificial inputs and watermark every figure")
    execute(parser.parse_args())
