"""Render only the externally pinned, completed diagnostic aggregate summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import signal
import sys
from pathlib import Path

from openjev.research.suspend_clock import SuspendClock

SUMMARY_PIN = "7cdd2533910826e95760e3d0144b2163060a85c05702b4413cafda857c13260c"
RECEIPT_PIN = "a1d2e52a2c00a67e477378dee5e4e50b0279136b474841fcc5d4102f70d0b321"
PLAN_PIN = "b251d8d63713f2908f79fc327b90aa3e52c1e5108c8fc44c07f32b90bff04d36"
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
SEEDS = (6901, 6902, 6903)
ARMS = (
    ("frozen_original", "Frozen / original", "#0072B2"),
    ("frozen_numbers", "Frozen / numbers", "#D08700"),
    ("trainable_original", "Trainable / original", "#009E73"),
    ("trainable_numbers", "Trainable / numbers", "#BB55A1"),
)
OUTPUTS = ("started.json", "plotted-values.json", "probability-sensitivity.png", "receipt.json", "failed.json")
SCOPE = (
    "Presentation of pinned aggregate values only; all 12 fits and all eight fixed temperatures. "
    "No temperature selection, new predictions, model calls, fitted calibration or confidence intervals. "
    "Original scientific continuation remains FAIL, six of seven conditions passed."
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": digest(data)}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute(summary_path, out):
    out.mkdir(parents=True, exist_ok=True)
    need(not any((out / name).exists() for name in OUTPUTS), "Exclusive presentation outputs required")
    clock = SuspendClock()
    deadline = clock.deadline_after(120)
    source = descriptor(Path(__file__))

    def timeout(*_):
        raise TimeoutError("Presentation exceeded its 120-second allocation")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        write_json(out / "started.json", {
            "status": "started", "summary_sha256": SUMMARY_PIN, "source": source,
            "clock_backend": clock.backend, "started_ns": deadline.started_ns,
            "deadline_ns": deadline.expires_ns, "wall_cap_seconds": 120, "scope": SCOPE,
        })
        raw = summary_path.read_bytes()
        need(digest(raw) == SUMMARY_PIN, "Completed summary pin mismatch")
        summary = json.loads(raw)
        need(summary["status"] == "completed" and summary["complete_fits"] == 12,
             "Complete twelve-fit diagnostic required")
        need(summary["plan_sha256"] == PLAN_PIN, "Diagnostic plan identity")
        need(summary["temperature_grid"] == list(TEMPERATURES)
             and summary["temperature_selected"] is None, "Fixed, unselected temperature grid")
        need(summary["original_continuation_passed"] is False
             and summary["original_conditions_passed"] == 6
             and summary["original_conditions_total"] == 7, "Preserved scientific failure")
        expected_fits = {f"{arm}-{seed}" for arm, _, _ in ARMS for seed in SEEDS}
        need(set(summary["fits"]) == expected_fits, "Exact four-arm, three-seed membership")

        values = {"scope": SCOPE, "summary_sha256": SUMMARY_PIN, "panel": "unseen/micro",
                  "endpoints_per_fit": 33093, "temperatures": list(TEMPERATURES),
                  "seed_ids": list(SEEDS), "temperature_selected": None, "arms": {}}
        for arm, label, color in ARMS:
            by_seed = {}
            for seed in SEEDS:
                fit = summary["fits"][f"{arm}-{seed}"]
                need(fit["temperature_selected"] is None, "No per-fit temperature selection")
                need(set(fit["temperature_grid"]) == {str(t) for t in TEMPERATURES}, "Complete per-fit grid")
                data = {"nll": [], "brier": []}
                for temperature in TEMPERATURES:
                    cell = fit["temperature_grid"][str(temperature)]["unseen/micro"]
                    need(cell["count"] == 33093 and cell["choices_unchanged"] is True,
                         "Unchanged choices and complete unseen panel")
                    for metric in data:
                        value = cell[metric]
                        need(type(value) in (float, int) and math.isfinite(value), "Finite aggregate metric")
                        data[metric].append(value)
                by_seed[str(seed)] = data
            means = {metric: [math.fsum(by_seed[str(seed)][metric][i] for seed in SEEDS) / 3
                              for i in range(len(TEMPERATURES))] for metric in ("nll", "brier")}
            values["arms"][arm] = {"label": label, "color": color, "seeds": by_seed,
                                    "equal_seed_mean": means}
        write_json(out / "plotted-values.json", values)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig = plt.figure(figsize=(12.5, 5.0), dpi=200, facecolor="white")
        fig.text(0.5, 0.958, "Probability sensitivity on exposed DEV", ha="center", va="top",
                 fontsize=18, fontweight="bold", color="#172033")
        fig.text(0.5, 0.898,
                 "All 12 fits | output only | no temperature selected | original study failed (6/7 conditions passed)",
                 ha="center", va="top", fontsize=9.4, color="#465365")
        markers = ("o", "s", "^")
        for metric, rectangle, title, ylabel in (
            ("nll", [0.07, 0.25, 0.405, 0.505], "Unseen services: NLL", "Log loss (nats, lower is better)"),
            ("brier", [0.565, 0.25, 0.405, 0.505], "Unseen services: Brier", "Candidate-sum Brier (lower is better)"),
        ):
            ax = fig.add_axes(rectangle)
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=11)
            ax.grid(axis="y", color="#DDE3EB", linewidth=0.65, zorder=0)
            for arm, _, color in ARMS:
                packet = values["arms"][arm]
                for seed, marker in zip(SEEDS, markers, strict=True):
                    ax.plot(TEMPERATURES, packet["seeds"][str(seed)][metric], color=color,
                            linewidth=0.8, alpha=0.48, marker=marker, markersize=3.6,
                            markerfacecolor="white", markeredgewidth=0.85, zorder=3)
                ax.plot(TEMPERATURES, packet["equal_seed_mean"][metric], color=color,
                        linewidth=2.5, alpha=0.97, zorder=2)
            ax.axvline(1, color="#4B5563", linestyle=(0, (4, 3)), linewidth=1.05, alpha=0.8)
            ax.text(1.035, 0.965, "T=1", transform=ax.get_xaxis_transform(), va="top", fontsize=8,
                    color="#374151", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4})
            ax.set_xlim(0.40, 4.10)
            ax.set_xticks(TEMPERATURES, ["0.5", "0.75", "1", "1.25", "1.5", "2", "3", "4"])
            ax.set_xlabel("Fixed output temperature", labelpad=7)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.tick_params(axis="both", labelsize=8.2)
        handles = [Line2D([0], [0], color=color, lw=2.7, label=label) for _, label, color in ARMS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.860), ncol=4,
                   frameon=False, fontsize=9, handlelength=2.3, columnspacing=2.5)
        fig.text(0.5, 0.142,
                 "Thin lines and markers: seeds 6901 (circle), 6902 (square), 6903 (triangle). "
                 "Thick lines: equal-seed means. No confidence intervals.",
                 ha="center", va="center", fontsize=8.4, color="#465365")
        fig.text(0.5, 0.096,
                 "33,093 endpoints per fit; the same exposed DEV examples across seeds.",
                 ha="center", va="center", fontsize=8.4, color="#465365")
        fig.text(0.5, 0.05,
                 "T=1 preserves raw scores. Other temperatures renormalize outputs; choices and recurrent state are unchanged.",
                 ha="center", va="center", fontsize=8.1, color="#465365")
        with (out / "probability-sensitivity.png").open("xb") as stream:
            fig.savefig(stream, format="png", dpi=200, facecolor="white",
                        metadata={"Title": "Probability sensitivity on exposed DEV", "Description": SCOPE})
        plt.close(fig)
        need(descriptor(summary_path)["sha256"] == SUMMARY_PIN, "Summary changed during plotting")
        need(descriptor(Path(__file__)) == source, "Presentation source changed")
        finished = clock.now_ns()
        need(finished < deadline.expires_ns, "Presentation exceeded suspend-inclusive allocation")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt = {
            "version": "dialogue-probability-sensitivity-figure-v1", "status": "completed", "scope": SCOPE,
            "source": {"plot.py": source},
            "inputs": {str(summary_path): {"bytes": len(raw), "sha256": SUMMARY_PIN}},
            "upstream_provenance": {
                "diagnostic_receipt_sha256": RECEIPT_PIN, "diagnostic_plan_sha256": PLAN_PIN,
                "scope": "Receipt identity supplied by the parent; not read. Plan identity checked in the pinned summary.",
            },
            "outputs": {name: descriptor(out / name) for name in
                        ("started.json", "plotted-values.json", "probability-sensitivity.png")},
            "image_dimensions": [2500, 1000], "complete_fits": 12, "seed_curves_per_panel": 12,
            "mean_curves_per_panel": 4, "grid_temperatures": list(TEMPERATURES), "temperature_selected": None,
            "model_calls": 0, "clock_backend": clock.backend, "started_ns": deadline.started_ns,
            "finished_ns": finished, "elapsed_ns": finished - deadline.started_ns,
            "wall_seconds": (finished - deadline.started_ns) / 1e9, "wall_cap_seconds": 120,
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        }
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        if not (out / "failed.json").exists():
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                       "scope": SCOPE, "model_calls": 0, "source": source})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("output/dialogue-probability-diagnostic-v1/run-01/summary.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    completed = execute(args.summary, args.out)
    print(json.dumps({"status": completed["status"], "wall_seconds": completed["wall_seconds"],
                      "image": str(args.out / "probability-sensitivity.png"), "model_calls": 0}))
