"""Plot the completed typed-observation reporter summary, never live fit files.

This is a saved-summary visualization, not an independent numerical audit.
Rendering requires all twelve final fits and successful technical validation.
No model, checkpoint, corpus or producer module is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

METHODS = ("flat_stratum", "flat_balanced", "typed_stratum", "typed_balanced")
SEEDS = (6101, 6102, 6103)
LABELS = ("Flat\nstratum", "Flat\nbalanced", "Typed\nstratum", "Typed\nbalanced")
SHORT = ("FS", "FB", "TS", "TB")
MARKERS = ("o", "s", "^")
COLORS = ("#3066a5", "#bd642d", "#51835b")
FITS = {f"{method}-{seed}" for method in METHODS for seed in SEEDS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_summary(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON scalar: "+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def number(value, name, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "Invalid "+name)
    return float(value)


def count(summary, method, seed, metric):
    value = summary["fits"][f"{method}-{seed}"]["decisions"][metric]
    n, d = value["numerator"], value["denominator"]
    require(type(n) is int and type(d) is int and d > 0 and 0 <= n <= d, "Invalid count: "+metric)
    return n, d


def rate(summary, method, seed, metric):
    n, d = count(summary, method, seed, metric)
    return 100*n/d


def nll(summary, method, seed):
    return number(summary["fits"][f"{method}-{seed}"]["cells"]["heldout_service/changed"]["nll"]["equal_service"], "primary NLL")


def validate(summary):
    require(summary["version"] == "dialogue-typed-v1" and summary["status"] == "completed"
            and summary["technical_validity_passed"] is True, "Only a technically complete final report can be plotted")
    require(set(summary["fits"]) == FITS and set(summary["costs"]["per_fit"]) == FITS, "All twelve fits and costs required")
    for key in ("execution_completed_sha256", "plan_sha256"):
        value = summary[key]
        require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Invalid "+key)
    rule = summary["continuation"]
    require(type(rule["passed"]) is bool and summary["continuation_allowed"] is rule["passed"]
            and rule["total_checks"] == 9 and len(rule["checks"]) == 9
            and all(type(c["passed"]) is bool for c in rule["checks"])
            and rule["checks_passed"] == sum(c["passed"] for c in rule["checks"])
            and rule["passed"] == all(c["passed"] for c in rule["checks"]), "Inconsistent frozen report decision")
    require(rule["comparison"] == "typed_balanced versus flat_balanced", "Wrong primary comparison")
    for metric in ("true_changed_recall", "dontcare_changed_recall", "retained_error",
                   "true_false_positive_rate", "dontcare_false_positive_rate"):
        denominators = {count(summary, m, s, metric)[1] for m in METHODS for s in SEEDS}
        require(len(denominators) == 1, "Different fit populations for "+metric)
        if metric.endswith("changed_recall"):
            require(denominators == {29 if metric.startswith("true") else 5}, "Unexpected fixed rare-category support")
    total = number(summary["costs"]["whole_wall_seconds"], "whole wall", positive=True)
    number(summary["costs"]["process_lifetime_peak_rss_bytes"], "RSS", positive=True)
    walls = []
    for method in METHODS:
        for seed in SEEDS:
            nll(summary, method, seed)
            fit = summary["fits"][f"{method}-{seed}"]
            require(fit["cells"]["heldout_service/changed"]["rows"] == 578, "Unexpected primary changed-row population")
            for value, size in (("true", 29), ("dontcare", 5)):
                require(fit["cells"][f"heldout_service/value/{value}/changed"]["rows"] == size, "Rare support disagrees with cells")
            cost = summary["costs"]["per_fit"][f"{method}-{seed}"]
            train = number(cost["training_wall_seconds"], "training wall", positive=True)
            evaluation = number(cost["evaluation_wall_seconds"], "evaluation wall", positive=True)
            wall = number(cost["wall_seconds"], "fit wall", positive=True)
            require(train+evaluation <= wall <= total, "Inconsistent recorded cost scopes")
            walls.append(wall)
    require(math.fsum(walls) <= total, "Fit wall sum exceeds whole run")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def arm_points(ax, summary, metric, *, with_counts=False):
    maximum = 0.
    labels = []
    for i, method in enumerate(METHODS):
        values = [rate(summary, method, seed, metric) for seed in SEEDS]
        maximum = max(maximum, *values)
        for j, value in enumerate(values):
            ax.scatter(i+(j-1)*.14, value, marker=MARKERS[j], color=COLORS[j], s=46, zorder=3)
        ax.hlines(math.fsum(values)/3, i-.27, i+.27, color="#202830", linewidth=2, zorder=4)
        label = LABELS[i]
        if with_counts:
            label += "\n"+" | ".join(f"{n}/{d}" for n, d in (count(summary, method, seed, metric) for seed in SEEDS))
        labels.append(label)
    ax.set_xticks(range(4), labels, fontsize=8 if with_counts else 9)
    ax.set_xlim(-.5, 3.5)
    # Full recall scale avoids visually magnifying gains on five examples.
    ax.set_ylim((-.8, 103) if with_counts else (-.03*max(maximum, 1.), 1.2*max(maximum, 1.)))
    ax.set_ylabel("Percent")


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.8))
    fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.83, wspace=.30, hspace=.52)
    rule = summary["continuation"]
    status = "PASS" if rule["passed"] else "FAIL"
    fig.suptitle(f"Typed output and rare-category support: {status} ({rule['checks_passed']}/9 fixed checks)",
                 x=.07, y=.97, ha="left", fontsize=19, fontweight="bold")
    fig.text(.07, .925, "Historically exposed internal TRAIN service split; correct previous value supplied. All 12 final fits shown.", fontsize=11)
    legend = [Line2D([0], [0], marker=MARKERS[j], color=COLORS[j], label=str(seed), linewidth=1.2)
              for j, seed in enumerate(SEEDS)]
    legend.append(Line2D([0], [0], color="#202830", linewidth=2, label="Three-seed mean"))
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.064, .905), ncol=4, frameon=False, fontsize=9)
    for ax in axes.flat:
        style(ax)

    ax = axes[0, 0]
    for j, seed in enumerate(SEEDS):
        values = [nll(summary, method, seed) for method in ("flat_balanced", "typed_balanced")]
        ax.plot([0, 1], values, marker=MARKERS[j], color=COLORS[j], linewidth=1.4, markersize=6)
    means = [math.fsum(nll(summary, m, s) for s in SEEDS)/3 for m in ("flat_balanced", "typed_balanced")]
    ax.plot([0, 1], means, color="#202830", linewidth=2.4)
    target = .95*means[0]
    ax.axhline(target, color="#666666", linestyle="--", linewidth=1, label="5% below flat mean")
    ax.set_xticks([0, 1], ["Flat balanced", "Typed balanced"])
    ax.set_xlim(-.18, 1.18)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Equal-service changed-state NLL (nats)")
    ax.set_title("A  Primary paired comparison", loc="left")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.text(.02, -.17, "578 changed rows; lower is better.\nEvery paired seed must be nonworse.",
            transform=ax.transAxes, fontsize=8, va="top")

    for ax, metric, title in ((axes[0, 1], "true_changed_recall", "B  TRUE-change recall: 29 rows"),
                              (axes[0, 2], "dontcare_changed_recall", "C  DONTCARE recall: only 5 rows")):
        arm_points(ax, summary, metric, with_counts=True)
        ax.set_title(title, loc="left")
        ax.text(.02, -.26, "Correct/total counts follow seed order 6101 | 6102 | 6103.", transform=ax.transAxes, fontsize=8, va="top")
    axes[0, 2].text(.02, .95, "One schema query; weak descriptive evidence", transform=axes[0, 2].transAxes,
                   fontsize=8, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})

    ax = axes[1, 0]
    arm_points(ax, summary, "retained_error")
    ax.set_title("D  Retained-state error", loc="left")
    ax.set_ylabel("Error rate (%)")
    cap = math.fsum(rate(summary, "flat_balanced", seed, "retained_error") for seed in SEEDS)/3+.5
    ax.axhline(cap, color="#666666", linestyle="--", linewidth=1, label="Flat-balanced mean + 0.5 pp")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, cap*1.12))
    ax.legend(loc="best", frameon=False, fontsize=8)

    ax = axes[1, 1]
    maximum = 0.
    for metric, offset, color, label in (("true_false_positive_rate", -.19, "#4b73ad", "TRUE"),
                                        ("dontcare_false_positive_rate", .19, "#bd7434", "DONTCARE")):
        for i, method in enumerate(METHODS):
            values = [rate(summary, method, seed, metric) for seed in SEEDS]
            maximum = max(maximum, *values)
            for j, value in enumerate(values):
                ax.scatter(i+offset+(j-1)*.055, value, color=color, marker=MARKERS[j], s=38,
                           label=label if i == j == 0 else None, zorder=3)
            ax.hlines(math.fsum(values)/3, i+offset-.10, i+offset+.10, color=color, linewidth=2)
    ax.set_xticks(range(4), LABELS)
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(-.03*max(maximum, .5), 1.2*max(maximum, .5))
    ax.set_ylabel("False positives / supported non-target rows (%)")
    ax.set_title("E  Rare-type false positives", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.text(.02, -.17, "Lower is better; allowed mean increase is 0.5 pp per type.\nDenominators exclude rows where that candidate type is absent.",
            transform=ax.transAxes, fontsize=8, va="top")

    ax = axes[1, 2]
    ticklabels = []
    for i, method in enumerate(METHODS):
        for j, seed in enumerate(SEEDS):
            x = i*3+j
            cost = summary["costs"]["per_fit"][f"{method}-{seed}"]
            training = cost["training_wall_seconds"]
            evaluation = cost["evaluation_wall_seconds"]
            other = cost["wall_seconds"]-training-evaluation
            bottom = 0.
            for value, color, label in ((training, "#627f99", "Training + checkpoint"),
                                         (evaluation, "#bc8249", "Evaluation + save"),
                                         (other, "#c4c8cc", "Other fit work")):
                ax.bar(x, value, bottom=bottom, width=.76, color=color, label=label if x == 0 else None)
                bottom += value
            ticklabels.append(f"{SHORT[i]}\n{seed}")
    ax.set_xticks(range(12), ticklabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Whole-fit seconds")
    ax.set_title("F  Actual paid fit cost", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    costs = summary["costs"]
    footer = (f"Whole run: {costs['whole_wall_seconds']:.1f} s; process-lifetime peak RSS: "
              f"{costs['process_lifetime_peak_rss_bytes']/1024**3:.2f} GiB. Fit time is not inference latency.\n"
              "Three seeds are repeated optimizations, not independent service samples. No FALSE/clear changes in the primary panel.\n"
              "Inherited feature preparation and separate preflight/reporting are additional costs. This figure does not establish calibration, recurrence, or fresh confirmation.")
    fig.text(.07, .025, footer, fontsize=9, va="bottom", linespacing=1.5)
    try:
        fig.savefig(out/"typed-observation.png", dpi=170, facecolor="white")
        fig.savefig(out/"typed-observation.pdf", facecolor="white")
    finally:
        plt.close(fig)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def execute(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source_pin = digest(__file__)
    try:
        path = Path(args.summary).resolve()
        summary_pin = digest(path)
        if args.summary_sha256 is not None:
            require(summary_pin == args.summary_sha256, "External summary digest mismatch")
        summary = read_summary(path)
        validate(summary)
        render(summary, out)
        require(digest(path) == summary_pin and digest(__file__) == source_pin, "Summary or plot source changed during rendering")
        write(out/"receipt.json", {"status": "completed", "version": "dialogue-typed-plot-v1",
              "summary_path": str(path), "summary_sha256": summary_pin, "source_sha256": source_pin,
              "execution_completed_sha256": summary["execution_completed_sha256"], "plan_sha256": summary["plan_sha256"],
              "fits": sorted(FITS), "fixed_report_continuation": summary["continuation_allowed"],
              "files": {name: {"sha256": digest(out/name), "bytes": (out/name).stat().st_size}
                        for name in ("typed-observation.png", "typed-observation.pdf")},
              "wall_seconds": time.monotonic()-start, "model_calls": 0,
              "scope": "Visualization of complete reporter summary only; no predictions, checkpoints or corpus read. Not an independent metric audit; no inference-latency or fresh-confirmation claim."})
    except BaseException as error:
        try:
            write(out/"failed.json", {"status": "failed", "error": repr(error), "source_sha256": source_pin,
                                      "wall_seconds": time.monotonic()-start, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve original rendering error
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt error: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256")
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())
