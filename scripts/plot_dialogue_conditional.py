"""Render pinned, completed conditional-observation summaries without inference.

The original saved-output reporter authenticates predictions and technical work.
This presentation layer consumes only its summary/receipt and run completion.
It neither reads predictions/checkpoints nor recomputes scientific metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
MODES = ("mean", "slot", "candidate")
LABELS = ("Mean pooling", "Slot attention", "Candidate attention")
SEEDS = (5301, 5302, 5303)
FIT_ORDER = [f"{mode}-{seed}" for seed, modes in zip(
    SEEDS, (MODES, ("slot", "candidate", "mean"), ("candidate", "mean", "slot")), strict=True)
    for mode in modes]
MARKERS = ("o", "s", "^")
COLORS = ("#3274a1", "#b15c24", "#387c58")
FAILURES = ("failed.json", "late-completion.json", "cleanup-error.json", "completion-before-cleanup-error.json")
FIT_WALL_SCOPE = (
    "Includes optimizer setup, training/batch assembly/validation, evaluation, checkpoint and prediction I/O, "
    "and per-fit payload hashing. Excludes shared authentication/loading, model initialization and final receipt/whole-run hashing."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid, object_pairs_hook=unique)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def number(value, name, *, upper=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid " + name)
    require(upper is None or value <= upper, "Out-of-range " + name)
    return float(value)


def close(a, b, name):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), "Inconsistent " + name)


def authenticate(completed, summary_path, pins):
    """Reject partial/failed runs before decoding any summary metric."""
    completed, summary_path = Path(completed).resolve(), Path(summary_path).resolve()
    receipt_path = summary_path.parent / "receipt.json"
    require(completed.name == "completed.json" and summary_path.name == "summary.json", "Input filenames")
    require(sha(completed) == pins["completed_sha256"], "Run completion hash")
    require(not any((folder / name).exists() for folder in (completed.parent, summary_path.parent)
                    for name in FAILURES), "Failed or demoted run/report")
    done = read_json(completed)
    require(done["status"] == "completed" and done["phase"] == "train"
            and done["version"] == "dialogue-conditional-training-v1"
            and done["completed_fits"] == done["expected_fits"] == FIT_ORDER
            and done["no_retry"] is True and done["quality_metrics_computed"] is False
            and done["encoder_calls"] == 0 and done["test_contents_accessed"] is False,
            "Completed nine-fit diagnostic required")
    require(sha(receipt_path) == pins["receipt_sha256"], "Report receipt hash")
    receipt = read_json(receipt_path)
    require(receipt["status"] == "completed" and receipt["technical_admission_passed"] is True
            and receipt["source_run_completed_sha256"] == pins["completed_sha256"]
            and Path(receipt["source_run"]).resolve() == completed.parent and receipt["model_calls"] == 0,
            "Authenticated completed saved-output analysis required")
    reporter = "scripts/report_dialogue_conditional.py"
    require(receipt["reporter_sha256"] == done["source_sha256"][reporter] == sha(ROOT / reporter),
            "Frozen reporter source identity")
    require(receipt["summary_sha256"] == pins["summary_sha256"] == sha(summary_path), "Summary hash")
    summary = read_json(summary_path)
    require(summary["technical_admission_passed"] is True
            and summary["source_run_completed_sha256"] == pins["completed_sha256"]
            and summary["plan_sha256"] == done["plan_sha256"]
            and summary["architecture_advantage_established"] is False, "Summary scope/binding")
    require(set(summary["fits"]) == set(FIT_ORDER) == set(summary["cost"]["fits"]), "Exact nine-fit membership")
    primary = summary["primary"]
    require(primary["panel"] == "unseen/changed" and [p["seed"] for p in primary["paired"]] == list(SEEDS),
            "Fixed primary seed coverage")
    require(type(primary["rows"]) is int and primary["rows"] > 0, "Nonempty primary required for this figure")
    data = {"primary": {"rows": primary["rows"], "paired": []}, "accuracy": {}, "fit_wall": {},
            "cost": summary["cost"], "plan_sha256": done["plan_sha256"]}
    for pair in primary["paired"]:
        seed = pair["seed"]
        slot, candidate = (number(pair[k], k) for k in ("slot_nll", "candidate_nll"))
        for mode, value in (("slot", slot), ("candidate", candidate)):
            cell = summary["fits"][f"{mode}-{seed}"]["cells"]["unseen/changed"]
            close(value, number(cell["nll"], "primary fit NLL"), "paired NLL")
            require(cell["rows"] == primary["rows"], "Primary denominator")
        delta = candidate - slot
        close(delta, pair["candidate_minus_slot_nll"], "paired difference")
        data["primary"]["paired"].append({"seed": seed, "slot_nll": slot, "candidate_nll": candidate,
                                           "candidate_minus_slot_nll": delta})
    mean_difference = math.fsum(p["candidate_minus_slot_nll"] for p in data["primary"]["paired"]) / 3
    close(mean_difference, primary["mean_candidate_minus_slot_nll"], "mean paired difference")
    passed = mean_difference < 0 and all(p["candidate_minus_slot_nll"] < 0 for p in data["primary"]["paired"])
    require(type(receipt["continuation_allowed"]) is bool
            and passed == primary["primary_nll_rule_passed"] == summary["continuation_allowed"]
            == receipt["continuation_allowed"], "Continuation rule mismatch")
    data["primary"].update(mean_candidate_minus_slot_nll=mean_difference, continuation_allowed=passed)
    for panel in ("seen", "unseen"):
        cells = {mode: [summary["fits"][f"{mode}-{seed}"]["cells"][panel + "/changed"]
                        for seed in SEEDS] for mode in MODES}
        count = cells["mean"][0]["rows"]
        require(type(count) is int and count > 0 and all(c["rows"] == count for v in cells.values() for c in v),
                "Changed-state common support")
        references = {}
        for name in ("previous_gold_carry", "literal_carry"):
            cell = summary["references"][name]["cells"][panel + "/changed"]
            require(cell["rows"] == count, "Reference denominator")
            references[name] = number(cell["accuracy"], "reference accuracy", upper=1)
        data["accuracy"][panel] = {"rows": count, "seed_accuracy": {
            mode: [number(c["accuracy"], "accuracy", upper=1) for c in v] for mode, v in cells.items()},
            "references": references}
    for mode in MODES:
        data["fit_wall"][mode] = [number(summary["cost"]["fits"][f"{mode}-{seed}"]["wall_seconds"],
                                              "fit wall") for seed in SEEDS]
    close(number(summary["cost"]["whole_run_wall_seconds"], "whole run wall"), done["wall_seconds"], "whole run cost")
    for name in ("pooled", "lexical", "tokens"):
        number(data["cost"]["inherited_cache_preparation"][name]["recorded_wall_seconds"], "cache wall")
    number(data["cost"]["conditional_metadata_preparation"]["recorded_wall_seconds"], "metadata wall")
    data["fit_wall_scope"] = FIT_WALL_SCOPE
    return data


def draw(data, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    fig.suptitle("Conditional observation with the correct previous value", fontsize=17, fontweight="bold", y=.98)
    result = "PASS" if data["primary"]["continuation_allowed"] else "FAIL"
    fig.text(.5, .943, f"Primary continuation: {result} | All 9 final fits | Privileged prior; exposed development data",
             ha="center", fontsize=11)
    ax = axes[0, 0]
    for pair, color, marker in zip(data["primary"]["paired"], COLORS, MARKERS, strict=True):
        ax.plot([0, 1], [pair["slot_nll"], pair["candidate_nll"]], color=color, marker=marker,
                linewidth=1.7, markersize=6,
                label=f"{pair['seed']}: candidate - slot = {pair['candidate_minus_slot_nll']:+.4f}")
    ax.set(xticks=[0, 1], xticklabels=["Slot attention", "Candidate attention"], xlim=(-.2, 1.2),
           ylabel="Unweighted NLL (nats), lower is better")
    ax.set_ylim(bottom=0)
    ax.set_title(f"A. Primary: unseen changed state ({data['primary']['rows']:,} rows)", loc="left", fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=8.8)
    for ax, panel, letter in ((axes[0, 1], "seen", "B"), (axes[1, 0], "unseen", "C")):
        entry = data["accuracy"][panel]
        means = [100*math.fsum(entry["seed_accuracy"][mode])/3 for mode in MODES]
        ax.bar(range(3), means, color=["#d5dce3", "#b2c2d0", "#82b19b"], width=.62, zorder=2)
        for i, (seed, color, marker) in enumerate(zip(SEEDS, COLORS, MARKERS, strict=True)):
            ax.scatter([x+(i-1)*.13 for x in range(3)], [100*entry["seed_accuracy"][m][i] for m in MODES],
                       color=color, marker=marker, edgecolor="white", linewidth=.6, s=40, zorder=5, label=str(seed))
        ax.axhline(100*entry["references"]["literal_carry"], color="#a13d34", linestyle="--", linewidth=1.4)
        ax.axhline(100*entry["references"]["previous_gold_carry"], color="#555555", linestyle=":", linewidth=1.4)
        ax.set(xticks=range(3), xticklabels=LABELS, ylim=(-2, 103), ylabel="Changed-state accuracy (%)")
        ax.set_title(f"{letter}. {panel.title()} services ({entry['rows']:,} rows)", loc="left", fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
    ax = axes[1, 1]
    bottoms = [0., 0., 0.]
    for i, (seed, color) in enumerate(zip(SEEDS, COLORS, strict=True)):
        values = [data["fit_wall"][mode][i] for mode in MODES]
        ax.bar(range(3), values, bottom=bottoms, color=color, edgecolor="white", width=.62, label=str(seed))
        for x, value in enumerate(values):
            ax.text(x, bottoms[x]+value/2, f"{value:.1f}s", ha="center", va="center", color="white", fontsize=9)
        bottoms = [old+value for old, value in zip(bottoms, values, strict=True)]
    for x, total in enumerate(bottoms):
        ax.annotate(f"{total:.1f}s total", (x, total), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xticks=range(3), xticklabels=LABELS, ylim=(0, max(bottoms)*1.16), ylabel="Sum of 3 fit wall times (seconds)")
    ax.set_title("D. Per-fit training + evaluation + I/O", loc="left", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7e7e7", linewidth=.6)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", label=f"Seed {s}")
               for s, c, m in zip(SEEDS, COLORS, MARKERS, strict=True)]
    handles += [Patch(facecolor="#b2c2d0", label="Accuracy bar: 3-fit mean"),
                Line2D([], [], color="#a13d34", linestyle="--", label="Literal carry"),
                Line2D([], [], color="#555555", linestyle=":", label="Previous-gold carry")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .914), ncol=3, frameon=False, fontsize=9)
    cost = data["cost"]
    cached = cost["inherited_cache_preparation"]
    fig.text(.5, .105, "Same query rows across seeds; points are paired initializations, not independent datasets. No rollout or architecture claim.",
             ha="center", fontsize=9)
    fig.text(.5, .08, "Fit totals include optimizer/training, validation, evaluation, checkpoint/prediction I/O and per-fit hashing.", ha="center", fontsize=9)
    fig.text(.5, .056, f"Whole run: {cost['whole_run_wall_seconds']:.1f}s through payload hashing, including shared authentication/loading and initialization; completion write is cap-checked separately.",
             ha="center", fontsize=9)
    fig.text(.5, .031, "Separate inherited cache wall: " + "; ".join(
        f"{name} {cached[name]['recorded_wall_seconds']:.2f}s" for name in ("pooled", "lexical", "tokens"))
        + f". Conditional metadata: {cost['conditional_metadata_preparation']['recorded_wall_seconds']:.2f}s. No sum of nested scopes.",
        ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.985, top=.80, bottom=.19, hspace=.47, wspace=.28)
    fig.savefig(out / "comparison.png", dpi=180, facecolor="white", metadata={"Software": "OpenJev saved-summary renderer"})
    fig.savefig(out / "comparison.pdf", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def render(completed, summary, out, *, completed_sha256, summary_sha256, receipt_sha256):
    completed, summary, out = Path(completed).resolve(), Path(summary).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = {"completed_sha256": completed_sha256, "summary_sha256": summary_sha256, "receipt_sha256": receipt_sha256}
    inputs = {"completed": str(completed), "summary": str(summary), **pins}
    source = sha(__file__)
    try:
        write_json(out / "started.json", {"inputs": inputs, "plotter_sha256": source})
        data = authenticate(completed, summary, pins)
        draw(data, out)
        write_json(out / "plotted-values.json", data)
        require(sha(completed) == completed_sha256 and sha(summary) == summary_sha256
                and sha(summary.parent / "receipt.json") == receipt_sha256 and sha(__file__) == source,
                "Input/source drift during rendering")
        files = ("started.json", "comparison.png", "comparison.pdf", "plotted-values.json")
        receipt = {"status": "completed", "inputs": inputs, "plotter_sha256": source,
                   "plan_sha256": data["plan_sha256"], "fits": 9, "model_calls": 0, "encoder_calls": 0,
                   "predictions_or_checkpoints_loaded": False, "continuation_allowed": data["primary"]["continuation_allowed"],
                   "scope": "Descriptive presentation of authenticated saved aggregates; original reporter owns prediction and technical audit",
                   "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
                   "files": {name: {"sha256": sha(out/name), "bytes": (out/name).stat().st_size} for name in files}}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "inputs": inputs, "plotter_sha256": source, "error": repr(error)})
        except BaseException as preservation:  # noqa: BLE001 - retain original exception
            note = getattr(error, "add_note", None)
            if callable(note):
                note("Plot failure receipt could not be written: " + repr(preservation))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", required=True, type=Path)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render(**vars(args))
