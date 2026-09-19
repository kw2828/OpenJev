"""Plot a completed development audit summary; no model or experiment calls.

Only the externally hash-bound summary is consumed. This renderer does not
replace the saved-output audit or establish native-control effectiveness.
Outputs are exclusive, and a failed render is retained without an automatic retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

VARIANTS = ("constant", "age", "raw", "normalized")
PANELS, STAGES = ("dev6", "dev10"), ("initial", "final")
PRIMARY = "post_reacquisition_mean_mse"
METRICS = (PRIMARY, "one_step_angle_mse", "one_step_reward_mse")
FILES = ("recovery-mse.png", "recovery-mse.svg")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(*parts):
    return "/".join(map(str, parts))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def validate_summary(summary):
    """Check the plotted aggregates and exact predeclared arithmetic, not provenance."""
    require(summary.get("status") == "completed" and summary.get("complete_evidence_passed") is True
            and summary.get("development_only") is True, "Completed development audit required")
    for name, expected in {"fits": 12, "evaluations": 48, "episodes_per_evaluation": 128,
                           "optimizer_updates": 1920, "new_model_calls": 0,
                           "new_optimizer_steps": 0, "new_native_calls": 0}.items():
        require(type(summary.get(name)) is int and summary[name] == expected, "Audit coverage: " + name)
    for name in ("plan_sha256", "execution_completed_sha256"):
        require(isinstance(summary.get(name), str) and re.fullmatch(r"[0-9a-f]{64}", summary[name]),
                "Audit input SHA: " + name)
    fits, families = summary["fit_means"], summary["family_means"]
    require(set(fits) == {key(v, p, s, panel) for v in VARIANTS for p in range(3)
                         for s in STAGES for panel in PANELS}, "All 48 fit aggregates required")
    require(set(families) == {key(v, s, panel) for v in VARIANTS for s in STAGES for panel in PANELS},
            "All 16 family aggregates required")
    for values in (*fits.values(), *families.values()):
        for metric in METRICS:
            value = values.get(metric)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "Finite nonnegative aggregate required: " + metric)
    for variant in VARIANTS:
        for stage in STAGES:
            for panel in PANELS:
                for metric in METRICS:
                    mean = math.fsum(fits[key(variant, pair, stage, panel)][metric] for pair in range(3)) / 3
                    require(math.isclose(families[key(variant, stage, panel)][metric], mean,
                                         rel_tol=1e-12, abs_tol=1e-15), "All-three-fit family arithmetic")

    expected = []

    def add(name, left, control, ratio):
        expected.append({"name": name, "left": left, "control": control, "ratio": ratio,
                         "threshold": ratio * control, "comparison": "<=", "passed": left <= ratio * control})

    for panel in PANELS:
        for control in ("age", "raw"):
            add(f"{panel}/normalized/{control}/family_recovery",
                families[key("normalized", "final", panel)][PRIMARY], families[key(control, "final", panel)][PRIMARY], .97)
            for pair in range(3):
                add(f"{panel}/normalized/{control}/pair{pair}_recovery",
                    fits[key("normalized", pair, "final", panel)][PRIMARY], fits[key(control, pair, "final", panel)][PRIMARY], 1.)
        add(f"{panel}/normalized/constant/family_recovery",
            families[key("normalized", "final", panel)][PRIMARY], families[key("constant", "final", panel)][PRIMARY], 1.)
        for control in ("age", "raw"):
            for metric, ratio in (("one_step_angle_mse", 1.02), ("one_step_reward_mse", 1.05)):
                add(f"{panel}/normalized/{control}/{metric}", families[key("normalized", "final", panel)][metric],
                    families[key(control, "final", panel)][metric], ratio)
    for pair in range(3):
        add(f"dev6/normalized/pair{pair}/initial_learning", fits[key("normalized", pair, "final", "dev6")][PRIMARY],
            fits[key("normalized", pair, "initial", "dev6")][PRIMARY], .9)
    gate = summary["continuation"]
    require(len(expected) == 29 and gate.get("checks") == expected, "Exact 29 continuation checks; no threshold tolerance")
    require(type(gate.get("checks_passed")) is int and gate["checks_passed"] == sum(row["passed"] for row in expected)
            and type(gate.get("total_checks")) is int and gate["total_checks"] == 29
            and type(gate.get("passed")) is bool and gate["passed"] == all(row["passed"] for row in expected),
            "Continuation status disagrees with retained checks")
    return summary


def draw(summary, out, *, synthetic=False):
    """Draw all final points; synthetic layout fixtures require an explicit watermark."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "openjev-innovation-development-pilot-v1"})
    colors, markers = ("#2563A6", "#BB6335", "#268075"), ("o", "^", "s")
    offsets = (-.12, 0., .12)
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 6.3), sharey=True)
    figure.subplots_adjust(left=.09, right=.975, bottom=.26, top=.73, wspace=.13)
    try:
        for axis, panel in zip(axes, PANELS, strict=True):
            for pair in range(3):
                values = [summary["fit_means"][key(v, pair, "final", panel)][PRIMARY] for v in VARIANTS]
                x = [i + offsets[pair] for i in range(4)]
                axis.plot(x, values, color=colors[pair], alpha=.32, linewidth=1, zorder=2)
                axis.scatter(x, values, color=colors[pair], marker=markers[pair], s=53,
                             edgecolors="white", linewidths=.5, zorder=4)
            means = [summary["family_means"][key(v, "final", panel)][PRIMARY] for v in VARIANTS]
            for x, value in enumerate(means):
                axis.plot([x - .23, x + .23], [value, value], color="#17202A", linewidth=2.1, zorder=3)
                # Offset the mean so it cannot hide initialization1 when their
                # values coincide. The horizontal mark still spans the group.
                axis.scatter([x + .29], [value], color="#17202A", marker="D", s=36, zorder=5)
            axis.set_xticks(range(4), ("Constant", "Age", "Raw error", "Normalized\nerror"))
            axis.set_xlim(-.45, 3.45)
            axis.set_title("Original sensing\nTwo six-packet gaps" if panel == "dev6" else
                           "Additional censoring\nTwo ten-packet gaps", fontsize=12, pad=13)
            axis.grid(axis="y", color="#DCE2E7", linewidth=.7)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        maximum = max(summary["fit_means"][key(v, p, "final", panel)][PRIMARY]
                      for v in VARIANTS for p in range(3) for panel in PANELS)
        axes[0].set_ylim(0, maximum * 1.16 if maximum else 1.)
        axes[0].set_ylabel("Post-reacquisition angle MSE\n(lower is better)", labelpad=10)
        gate = summary["continuation"]
        figure.suptitle("Development prediction pilot; not native control", fontsize=17, fontweight="bold", y=.97)
        status = "PASS: continue to confirmation" if gate["passed"] else "FAIL: continuation criterion not met"
        figure.text(.5, .893, f"{gate['checks_passed']}/29 checks passed  |  {status}", ha="center", fontsize=11,
                    color="#236944" if gate["passed"] else "#9F3A32")
        if synthetic:
            figure.text(.5, .82, "SYNTHETIC LAYOUT TEST - NOT RESULTS", ha="center", fontsize=12, color="#9F3A32")
        legend = [Line2D([0], [0], color=colors[p], marker=markers[p], linewidth=1,
                         markersize=6, label=f"Paired initialization {p}") for p in range(3)]
        legend.append(Line2D([0], [0], color="#17202A", marker="D", linewidth=2,
                             markersize=5, label="Mean of all three fits"))
        figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .13), ncol=4, frameon=False)
        figure.text(.09, .083, "Final epoch 8; all 12 fits retained. Each point averages 128 development episodes, "
                    "two recovery points and horizons 1 and 3.", fontsize=9, color="#44515C")
        figure.text(.09, .047, "Error uses four public cosine/sine coordinates. Ten-gap views censor the same recorded "
                    "trajectories; no fresh native control or calibrated-uncertainty claim.", fontsize=8.7, color="#44515C")
        figure.savefig(out / FILES[0], dpi=180, facecolor="white")
        figure.savefig(out / FILES[1], facecolor="white", metadata={"Date": None})
    finally:
        plt.close(figure)


def render(summary_path, expected_sha256, out):
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256), "External summary SHA required")
    summary_path, out = Path(summary_path), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    try:
        write(out / "started.json", {"summary_sha256": expected_sha256, "automatic_retry": False})
        raw = summary_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Audit summary digest mismatch")
        summary = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))
        require(summary.get("synthetic_fixture") is not True, "Synthetic fixtures cannot be published as results")
        validate_summary(summary)
        draw(summary, out)
        require(sha(summary_path) == expected_sha256, "Audit summary changed during rendering")
        receipt = {"status": "completed", "development_only": True,
            "scope": "Presentation of authenticated saved audit arithmetic; no new evaluation",
            "summary_sha256": expected_sha256, "plan_sha256": summary["plan_sha256"],
            "execution_completed_sha256": summary["execution_completed_sha256"],
            "renderer_source_sha256": sha(Path(__file__)), "fits_retained": 12, "final_fit_points": 24,
            "family_means_shown": 8, "checks_passed": summary["continuation"]["checks_passed"],
            "total_checks": 29, "continuation_passed": summary["continuation"]["passed"],
            "new_model_calls": 0, "new_native_calls": 0, "new_optimizer_steps": 0,
            "created_utc": datetime.now(UTC).isoformat(), "wall_seconds": time.monotonic() - begin,
            "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in FILES}}
        write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error": repr(error), "automatic_retry": False})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note("Render failure preservation: " + repr(preservation_error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    result = render(arguments.summary, arguments.summary_sha256, arguments.out)
    print(json.dumps({"status": result["status"], "files": result["files"]}, sort_keys=True))
