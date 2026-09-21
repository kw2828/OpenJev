"""Render authenticated saved report metrics only; no arrays, models or fitting.

The full report payload manifest is checked before reading metrics and again
after rendering. Each displayed mean weights the three seeds equally. Lines
connect the same seed before and after calibration; they are not uncertainty
intervals. Execution is intentionally separate from source preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import time
from pathlib import Path, PurePosixPath

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
VERSION = "dialogue-calibration-figure-v1"
REPORT_VERSION = "dialogue-calibration-report-v2"
SEEDS = (6901, 6902, 6903)
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
FIT_IDS = {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}
PRIMARY = ("frozen_numbers", "trainable_numbers")
ROUTES = ("raw", "calibrated")
OUTPUT_FILES = ("calibration-control.png", "calibration-control.svg", "plotted-values.json")
SCOPE = ("Exposed official DEV, unseen-service panel. Calibration uses excluded TRAIN dialogues. "
         "Primary number-aware arms only; all twelve fits remain in the report. "
         "Equal-seed descriptive means, no confidence intervals or new statistical tests. "
         "No architecture, untouched-generalization or causal improvement claim.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe_member(directory, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and str(relative) == name and name not in ("", ".")
            and all(part not in (".", "..") for part in relative.parts), "Canonical relative report member")
    path = directory
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), "No symlink in report member")
    require(path.is_file(), "Regular report payload required: " + name)
    return path


def authenticate(report, receipt_sha256):
    require(not report.is_symlink() and report.is_dir(), "Regular report directory required")
    require(len(receipt_sha256) == 64 and all(c in "0123456789abcdef" for c in receipt_sha256),
            "External report receipt SHA256 required")
    receipt_path = safe_member(report, "receipt.json")
    require(sha(receipt_path) == receipt_sha256, "Report receipt pin changed")
    receipt = read(receipt_path)
    require(receipt["version"] == REPORT_VERSION and receipt["status"] == "completed"
            and receipt["technical_validity_passed"] is True and receipt["complete_fits"] == 12
            and receipt["model_calls"] == 0 and receipt["official_test_opened"] is False
            and receipt["original_raw_failure_preserved"] is True
            and receipt["legacy_v1_cost_failure_preserved"] is True
            and receipt["runtime_v2_admission_passed"] is True
            and receipt["original_raw_checks_passed"] == 6 and receipt["original_raw_checks_total"] == 7
            and receipt["checks_total"] == 11, "Complete report with unchanged historical result required")
    expected = {"started.json", "summary.json", "report.md"} | {f"fits/{name}.json" for name in FIT_IDS}
    require(set(receipt["files"]) == expected, "Exact complete report payload closure")
    present = set()
    for path in report.rglob("*"):
        require(not path.is_symlink(), "No report symlinks")
        if path.is_file():
            present.add(path.relative_to(report).as_posix())
    require(present == expected | {"receipt.json"}, "No missing or additional report payloads")
    for name, expected_descriptor in receipt["files"].items():
        require(set(expected_descriptor) == {"bytes", "sha256"}
                and descriptor(safe_member(report, name)) == expected_descriptor, "Report member changed: " + name)
    summary = read(report / "summary.json")
    require(summary["version"] == REPORT_VERSION and summary["status"] == "completed"
            and summary["technical_validity_passed"] is True and summary["complete_fits"] == 12
            and set(summary["fits"]) == FIT_IDS and set(summary["fit_panels"]) == FIT_IDS,
            "All twelve completed report fits required")
    require(summary["plan_sha256"] == receipt["plan_sha256"]
            and summary["control_plan_sha256"] == receipt["control_plan_sha256"]
            and summary["phases"] == receipt["phases"], "Summary and receipt lineage agree")
    original, gate = summary["original_raw_result"], summary["continuation"]
    require(original["passed"] is False and original["checks_passed"] == 6 and original["checks_total"] == 7
            and original["all_fit_metrics_exactly_reproduced"] is True
            and gate["raw_result_revised"] is False and gate["decision_conditions_unchanged"] is True
            and gate["checks_total"] == len(gate["checks"]) == 11
            and all(type(check["passed"]) is bool for check in gate["checks"])
            and len({check["name"] for check in gate["checks"]}) == 11
            and gate["checks_passed"] == sum(check["passed"] for check in gate["checks"])
            and gate["passed"] is all(check["passed"] for check in gate["checks"])
            and gate["passed"] is receipt["continuation_passed"]
            and gate["checks_passed"] == receipt["checks_passed"], "Recorded scientific gate agrees exactly")
    fits = {}
    for name, index in summary["fits"].items():
        require(index["fit_id"] == name and index["path"] == f"fits/{name}.json"
                and {key: index[key] for key in ("bytes", "sha256")} == receipt["files"][index["path"]],
                "Canonical per-fit index and manifest agree")
        fits[name] = read(report / index["path"])
        for route in ("raw", "normalized", "calibrated"):
            for panel in ("all", "seen", "unseen"):
                full = fits[name][route]["panels"][panel]
                require({key: full[key] for key in ("micro", "strata", "macro_three")}
                        == summary["fit_panels"][name][route][panel], "Summary and full fit metrics agree")
        for route in ("normalized", "calibrated"):
            require(fits[name][route]["validation"]["choices_unchanged"] is True
                    and fits[name][route]["validation"]["top_tie_masks_unchanged"] is True,
                    "All decision and full top-tie witnesses unchanged")
    return receipt, summary, fits


def plotted_values(receipt, summary, fits, report, receipt_sha256):
    groups = []
    for arm in PRIMARY:
        for route in ROUTES:
            points = []
            for seed in SEEDS:
                fit_id = f"{arm}-{seed}"
                panel = fits[fit_id][route]["panels"]["unseen"]
                values = {"macro_accuracy": panel["macro_three"]["accuracy"],
                          "nll": panel["micro"]["nll"], "brier": panel["micro"]["brier"]}
                require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                            for value in values.values()) and values["macro_accuracy"] <= 1,
                        "Finite supported plotted metrics")
                require(values["macro_accuracy"] == fits[fit_id]["raw"]["panels"]["unseen"]["macro_three"]["accuracy"],
                        "Raw and calibrated accuracy must agree for each seed")
                points.append({"fit_id": fit_id, "seed": seed, **values})
            means = {key: math.fsum(point[key] for point in points) / len(SEEDS)
                     for key in ("macro_accuracy", "nll", "brier")}
            published = summary["factorial"][route]["families"][arm]["mean"]["panels"]["unseen"]
            require(means == {"macro_accuracy": published["macro_three"]["accuracy"],
                              "nll": published["micro"]["nll"], "brier": published["micro"]["brier"]},
                    "Displayed equal-seed means exactly reproduce reported factorial means")
            groups.append({"arm": arm, "route": route, "points": points, "equal_seed_mean": means})
    return {"version": VERSION, "title": "Same decisions, calibrated probabilities", "seeds": list(SEEDS),
            "groups": groups, "continuation": summary["continuation"],
            "original_raw_result": summary["original_raw_result"],
            "report": {"path": str(report), "receipt_sha256": receipt_sha256,
                       "summary": receipt["files"]["summary.json"]},
            "definitions": {"macro_accuracy": "Equal mean of the three unseen-stratum accuracies",
                            "nll": "Unseen endpoint mean negative log likelihood, natural logarithm",
                            "brier": "Unseen endpoint mean candidate-sum Brier score",
                            "means": "Arithmetic mean of all three seeds with equal seed weights",
                            "pairing": "Each line joins the same seed within an arm before and after calibration"},
            "scope": SCOPE}


def render(values, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelcolor": "#344054", "text.color": "#17212e",
                         "xtick.color": "#344054", "ytick.color": "#344054",
                         "svg.hashsalt": VERSION, "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    figure.subplots_adjust(left=.068, right=.984, bottom=.265, top=.725, wspace=.31)
    figure.suptitle(values["title"], x=.5, y=.972, fontsize=23, fontweight="bold")
    gate = values["continuation"]
    status = "PASS" if gate["passed"] else "FAIL"
    figure.text(.5, .902, f"Calibration control: {status} ({gate['checks_passed']}/11)  |  Original raw study: FAIL (6/7)",
                ha="center", fontsize=14)
    figure.text(.5, .857, "Exposed DEV, unseen services  |  Primary number-aware arms  |  All three seeds shown",
                ha="center", fontsize=13, color="#536170")
    colors = ("#2567a8", "#bd5b20")
    offsets, markers = (-.11, 0., .11), ("o", "s", "^")
    metrics = (("macro_accuracy", "Unseen macro accuracy", "Higher is better"),
               ("nll", "Unseen log loss", "Lower is better"),
               ("brier", "Unseen Brier score", "Lower is better"))
    labels = ("Frozen\nraw", "Frozen\ncalibrated", "Trainable\nraw", "Trainable\ncalibrated")
    for ax, (metric, title, direction) in zip(axes, metrics, strict=True):
        all_values = [point[metric] for group in values["groups"] for point in group["points"]]
        lo, hi = min(all_values), max(all_values)
        span = max(hi - lo, .02 if metric == "macro_accuracy" else .04)
        ax.set_ylim(max(0., lo - .24 * span), min(1., hi + .40 * span) if metric == "macro_accuracy" else hi + .40 * span)
        ax.set_xlim(-.5, 3.5)
        for pair, color in enumerate(colors):
            left = 2 * pair
            ax.axvspan(left - .48, left + 1.48, color=color, alpha=.045, linewidth=0, zorder=0)
            for seed_index, (offset, marker) in enumerate(zip(offsets, markers, strict=True)):
                ys = [values["groups"][index]["points"][seed_index][metric] for index in (left, left + 1)]
                ax.plot([left + offset, left + 1 + offset], ys, color=color, alpha=.50, linewidth=1.3, zorder=2)
                for index, y in zip((left, left + 1), ys, strict=True):
                    ax.scatter(index + offset, y, marker=marker, s=72,
                               facecolor="white" if index % 2 == 0 else color,
                               edgecolor=color, linewidth=1.45, zorder=4)
        for index, group in enumerate(values["groups"]):
            mean = group["equal_seed_mean"][metric]
            ax.hlines(mean, index - .23, index + .23, color="#17212e", linewidth=3.1, zorder=5)
            label = f"{100 * mean:.2f}%" if metric == "macro_accuracy" else f"{mean:.3f}"
            ax.text(index, .957, label, ha="center", va="top", transform=ax.get_xaxis_transform(),
                    fontsize=12, fontweight="bold", color=colors[index // 2])
        ax.set_title(title + "\n" + direction, fontsize=15, pad=16, linespacing=1.45)
        ax.set_xticks(range(4), labels=labels, fontsize=12)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if metric == "macro_accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d9e1e8", linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [Line2D([0], [0], color="#667085", marker=marker, markersize=7,
                      linewidth=1, label=f"Seed {seed}") for seed, marker in zip(SEEDS, markers, strict=True)]
    handles.append(Line2D([0], [0], color="#17212e", linewidth=3.1, label="Equal-seed mean"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .107), ncol=4,
                  frameon=False, fontsize=13, columnspacing=1.7, handlelength=1.4)
    figure.text(.5, .067, "Lines pair the same seed; no confidence intervals. Labels show equal-seed means.",
                ha="center", fontsize=12, color="#536170")
    figure.text(.5, .027, "Accuracy averages three strata. Log loss and Brier average endpoints. Temperatures fit on excluded TRAIN only.",
                ha="center", fontsize=11.5, color="#536170")
    figure.savefig(out / OUTPUT_FILES[0], dpi=200, metadata={"Title": values["title"], "Description": SCOPE})
    figure.savefig(out / OUTPUT_FILES[1], metadata={"Title": values["title"], "Description": SCOPE, "Date": None})
    plt.close(figure)
    return {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
            "png_pixels": [3000, 1280], "mean_interval": "none", "displayed_fit_count": 6, "report_fit_count": 12}


def execute(args):
    require(args.report.absolute() == ROOT / "output/dialogue-calibration-runtime-v2/report-01",
            "Fixed completed runtime V2 report")
    require(args.out.absolute() == SOURCE.parent / "render-01", "Exclusive first render directory")
    args.out.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    source_pin = descriptor(SOURCE)
    previous_handler = None
    try:
        def expired(*_):
            raise TimeoutError("Presentation exceeded 120 seconds")

        previous_handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 120)
        receipt, summary, fits = authenticate(args.report, args.receipt_sha256)
        values = plotted_values(receipt, summary, fits, args.report, args.receipt_sha256)
        write(args.out / OUTPUT_FILES[2], values)
        runtime = render(values, args.out)
        authenticate(args.report, args.receipt_sha256)
        require(descriptor(SOURCE) == source_pin, "Presentation source unchanged during render")
        result = {"version": VERSION, "status": "completed", "source": {"path": str(SOURCE), **source_pin},
                  "inputs": {"report": str(args.report), "receipt_sha256": args.receipt_sha256,
                             "payloads": receipt["files"]},
                  "outputs": {name: descriptor(args.out / name) for name in OUTPUT_FILES},
                  "runtime": runtime, "model_calls": 0, "raw_prediction_arrays_read": False,
                  "new_metrics_or_tests": False, "scientific_gate_revised": False,
                  "wall_seconds": time.perf_counter() - started, "scope": SCOPE,
                  "timing_scope": "Presentation only, separate from scientific and audit costs; excludes receipt publication"}
        require(result["wall_seconds"] < 120, "Presentation deadline")
        write(args.out / "receipt.json", result)
        return result
    except BaseException as error:
        if (args.out / "receipt.json").exists():
            (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
        write(args.out / "failed.json", {"version": VERSION, "status": "failed", "source": source_pin,
                                        "error_type": type(error).__name__, "error": str(error), "model_calls": 0,
                                        "scope": "Partial presentation retained; no automatic retry"})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    completed = execute(arguments)
    print(json.dumps({"status": completed["status"], "outputs": list(completed["outputs"]),
                      "receipt_sha256": sha(arguments.out / "receipt.json")}))
