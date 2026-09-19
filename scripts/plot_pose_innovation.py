"""Plot a hash-bound training-parent screen without model or data inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

STUDY = "pose-innovation-v1"
SEEDS = (1101, 1202, 1303)
VARIANTS = ("base", "bias", "ridge_summary", "ridge_ordered", "summary", "recurrent",
            "shuffled", "noerror", "error_shuffled")
LABELS = ("Frozen GRU", "Shrinkage bias", "Summary ridge", "Ordered ridge", "Summary MLP",
          "Ordered GRU (primary)", "Whole-token shuffle", "No-error GRU", "Error-only shuffle")
ENDPOINTS = ("position", "rotation")
FOLD_COLORS = ("#276b9b", "#ba762b", "#795ca1")
SEED_MARKERS = ("o", "s", "^")
FILES = ("started.json", "physical-errors.png", "physical-errors.svg", "plotted-values.json")
AUDIT_FILES = {"summary.json", "window-errors.npz", "README.md"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid " + name)
    return float(value)


def close(actual, expected, name):
    require(math.isclose(nonnegative(actual, name), nonnegative(expected, name),
                         rel_tol=1e-9, abs_tol=1e-12), "Inconsistent " + name)


def authenticate(summary_path, *, summary_sha256, receipt_sha256):
    summary_path = Path(summary_path)
    report, receipt_path = summary_path.parent, summary_path.parent / "receipt.json"
    require(summary_path.name == "summary.json", "Expected audited summary.json")
    require(sha(summary_path) == summary_sha256 and sha(receipt_path) == receipt_sha256,
            "External summary or receipt digest mismatch")
    summary, receipt = read_json(summary_path), read_json(receipt_path)
    require(summary["status"] == receipt["status"] == "completed" and summary["study"] == receipt["study"] == STUDY,
            "Completed innovation screen required")
    require(summary["variants"] == list(VARIANTS), "Frozen nine-method order required")
    for key in ("protocol_sha256", "execution_completed_sha256"):
        require(summary[key] == receipt[key] and len(summary[key]) == 64, "Audit identity: " + key)
    require(set(receipt["files"]) == AUDIT_FILES, "Exact three audit payloads required")
    expected = {**receipt["files"], "receipt.json": binding(receipt_path)}
    actual = {p.relative_to(report).as_posix() for p in report.rglob("*") if p.is_file()}
    require(actual == set(expected), "Extra, missing or failed audit payload")
    for name, record in expected.items():
        path = report / name
        require(not path.is_symlink() and path.resolve().is_relative_to(report.resolve()), "Unsafe audit payload")
        require(binding(path) == record, "Altered audit payload: " + name)
    screen = summary["screen"]
    checks = screen["checks"]
    require(type(screen["total_checks"]) is int and screen["total_checks"] == len(checks) == 368,
            "All368 frozen screen checks required")
    require(all(type(check["passed"]) is bool for check in checks), "Boolean screen checks required")
    require(len({check["name"] for check in checks}) == 368, "Unique screen check names required")
    passed = sum(check["passed"] for check in checks)
    require(type(screen["checks_passed"]) is int and screen["checks_passed"] == passed,
            "Screen pass count")
    require(type(screen["passed"]) is bool and screen["passed"] == (passed == 368)
            and type(receipt["qualification_passed"]) is bool
            and receipt["qualification_passed"] == screen["passed"], "Completed versus passed screen status")
    return summary, {"summary_sha256": summary_sha256, "audit_receipt_sha256": receipt_sha256,
                     "protocol_sha256": summary["protocol_sha256"],
                     "execution_completed_sha256": summary["execution_completed_sha256"],
                     "audit_members": expected}


def metric(value, parent_ids, name):
    mse = nonnegative(value["mse"], name + " MSE")
    close(value["rmse"], math.sqrt(mse), name + " RMSE")
    require(value["parent_ids"] == parent_ids, name + " parent identities")
    parents = value["parent_mse"]
    horizon = value["horizon_mse"]
    require(type(parents) is list and len(parents) == len(parent_ids), name + " parent coverage")
    require(type(horizon) is list and len(horizon) == 25, name + " horizon coverage")
    close(math.fsum(nonnegative(x, name) for x in parents) / len(parents), mse, name + " parent pooling")
    close(math.fsum(nonnegative(x, name) for x in horizon) / 25, mse, name + " horizon pooling")
    return {"mse": mse, "rmse": math.sqrt(mse), "parent_ids": parent_ids,
            "parent_mse": parents, "horizon_mse": horizon}


def plot_values(summary):
    """Check the equal-parent/equal-seed pooled arithmetic used in the figure."""
    expected_rows = {f"fold-{fold}-{seed}" for fold in range(3) for seed in SEEDS}
    require(set(summary["rows"]) == expected_rows, "All9 fold/seed rows required")
    require(set(summary["families"]) == {"train", "test"}, "Both train and test summaries required")
    result = {"train": {}, "test": {}}
    for split in ("train", "test"):
        families = summary["families"][split]
        require(set(families) == set(VARIANTS), "All9 families required")
        ranks = (1, 4, 7, 9) if split == "test" else (0, 2, 3, 5, 6, 8)
        parent_ids = sorted(fold + 3 * rank for fold in range(3) for rank in ranks)
        for variant in VARIANTS:
            result[split][variant] = {}
            for endpoint in ENDPOINTS:
                pooled = metric(families[variant][endpoint], parent_ids, split + "/" + variant + "/" + endpoint)
                marks = []
                parent_values = {parent: [] for parent in parent_ids}
                horizon_values = []
                for fold in range(3):
                    for seed in SEEDS:
                        row = summary["rows"][f"fold-{fold}-{seed}"]
                        require(set(row) == set(VARIANTS), "All9 methods in every fold/seed")
                        row_parents = [fold + 3 * rank for rank in ranks]
                        one = metric(row[variant][split][endpoint], row_parents, "paired " + endpoint)
                        marks.append({"fold": fold, "seed": seed, "mse": one["mse"], "rmse": one["rmse"]})
                        for parent, value in zip(row_parents, one["parent_mse"], strict=True):
                            parent_values[parent].append(value)
                        horizon_values.append(one["horizon_mse"])
                close(math.fsum(x["mse"] for x in marks) / 9, pooled["mse"], "Equal fold/seed MSE pooling")
                for parent, value in zip(parent_ids, pooled["parent_mse"], strict=True):
                    close(math.fsum(parent_values[parent]) / 3, value, "Equal seed parent pooling")
                for h, value in enumerate(pooled["horizon_mse"]):
                    close(math.fsum(row[h] for row in horizon_values) / 9, value, "Equal fold/seed horizon pooling")
                result[split][variant][endpoint] = {**pooled, "fold_seed_marks": marks}
    return result


def figure(values, screen, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": STUDY})
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, endpoint in zip(axes, ENDPOINTS, strict=True):
        for y, variant in enumerate(VARIANTS):
            data = values["test"][variant][endpoint]
            primary = variant == "recurrent"
            ax.barh(y, data["rmse"], height=.62, color="#168779" if primary else "#b4bfc6",
                    alpha=.85 if primary else .5)
            for item in data["fold_seed_marks"]:
                fold, seed_index = item["fold"], SEEDS.index(item["seed"])
                offset = (fold - 1) * .17 + (seed_index - 1) * .045
                ax.scatter(item["rmse"], y + offset, marker=SEED_MARKERS[seed_index],
                           color=FOLD_COLORS[fold], s=25, linewidth=.35, edgecolor="white", zorder=4)
        control_rmse = min(values["test"][variant][endpoint]["rmse"]
                           for variant in VARIANTS if variant != "recurrent")
        ax.axvline(.9 * control_rmse, color="#b24143", linestyle="--", linewidth=1.25)
        ax.set_yticks(range(len(VARIANTS)), LABELS)
        ax.set_ylim(len(VARIANTS) - .35, -.65)
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.set_title("Position error" if endpoint == "position" else "Rotation error", fontsize=13)
        ax.set_xlabel("RMSE (meters)" if endpoint == "position" else "Geodesic RMSE (radians)")
    fig.suptitle("Does ordered prediction-error memory help?", fontsize=19, weight="bold", y=.98)
    status = "PASS" if screen["passed"] else "FAIL"
    fig.text(.5, .935,
             f"Training-parent held-out screen | {status}: {screen['checks_passed']}/368 descriptive checks | Lower is better",
             ha="center", fontsize=11)
    legend = [Patch(facecolor="#b4bfc6", alpha=.6, label="Pooled RMSE"),
              Patch(facecolor="#168779", label="Primary: ordered GRU"),
              Line2D([], [], color="#b24143", linestyle="--", label="10% below strongest control")]
    legend.extend(Line2D([], [], marker="o", color=color, linestyle="none", label=f"Fold {fold}")
                  for fold, color in enumerate(FOLD_COLORS))
    legend.extend(Line2D([], [], marker=marker, color="#555555", linestyle="none", label=f"Seed {seed}")
                  for seed, marker in zip(SEEDS, SEED_MARKERS, strict=True))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .907),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(.01, .16, 1, .795), w_pad=3)
    fig.text(.025, .035,
             "Bars pool squared errors before square root. Marks show all 9 paired fold/seed results, not independent replicates.\n"
             "12 held-out original training parents; 24 overlapping windows per parent; 25 forecast steps per window.\n"
             "Dashed lines show only the pooled mean requirement. The screen also requires paired, parent and leave-one-parent-out checks.\n"
             "Backbones are frozen; correction heads use completed history. No external benchmark, closed-loop control or architecture claim.",
             fontsize=9, linespacing=1.5)
    fig.savefig(out / "physical-errors.png", dpi=170, metadata={"Software": "OpenJev saved-output plotter"})
    fig.savefig(out / "physical-errors.svg", metadata={"Date": None})
    plt.close(fig)


def render(summary, out, *, summary_sha256, receipt_sha256):
    summary, out = Path(summary), Path(out)
    require(not out.resolve().is_relative_to(summary.parent.resolve()), "Output must be outside sealed audit")
    out.mkdir(parents=True, exist_ok=False)
    begin, source = time.perf_counter(), sha(__file__)
    try:
        write_json(out / "started.json", {"status": "started", "source_sha256": source,
            "summary_sha256": summary_sha256, "audit_receipt_sha256": receipt_sha256})
        audited, inputs = authenticate(summary, summary_sha256=summary_sha256, receipt_sha256=receipt_sha256)
        values = plot_values(audited)
        write_json(out / "plotted-values.json", {"study": STUDY, "scope": "training-parent held-out signal screen",
            "screen": audited["screen"], "families": values, "methods": list(VARIANTS)})
        figure(values, audited["screen"], out)
        require(sha(__file__) == source, "Plotter changed during rendering")
        _, final_inputs = authenticate(summary, summary_sha256=summary_sha256, receipt_sha256=receipt_sha256)
        require(final_inputs == inputs, "Audited inputs changed during rendering")
        files = {name: binding(out / name) for name in FILES}
        require({p.name for p in out.iterdir()} == set(FILES), "Unexpected render payload")
        result = {"status": "completed", "study": STUDY, "inputs": inputs, "source_sha256": source,
            "screen_passed": audited["screen"]["passed"], "checks_passed": audited["screen"]["checks_passed"],
            "total_checks": 368, "methods_retained": 9, "paired_rows_per_method": 9, "held_out_parents": 12,
            "new_model_calls": 0, "new_optimizer_calls": 0, "new_native_calls": 0,
            "scope": "Authenticated audited aggregates and figures; no new forecasts or independent numerical re-audit",
            "files": files, "wall_seconds": time.perf_counter() - begin}
        write_json(out / "receipt.json", result)
        return result
    except BaseException as error:
        plt.close("all")
        try:
            if (out / "receipt.json").exists():
                (out / "receipt.json").rename(out / "receipt-before-error.json")
            write_json(out / "failed.json", {"status": "failed", "source_sha256": source,
                "error": repr(error), "wall_seconds": time.perf_counter() - begin})
        except BaseException as secondary:  # noqa: BLE001 - retain the original failure
            error.add_note("Unable to retain plotting failure: " + repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.out, summary_sha256=args.summary_sha256, receipt_sha256=args.receipt_sha256)
