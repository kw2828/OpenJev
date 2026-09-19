"""Plot authenticated fixed-expert hindsight capacity, without reoptimization."""
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

PANELS = ("test_sin", "test_zigzag")
SEEDS = (1101, 1202, 1303)
ENDPOINTS = ("position", "rotation")
METHODS = ("hard_window", "hard_step", "continuous")
MARKERS = ("o", "s", "^")
FILES = ("capacity.png", "capacity.svg", "plotted-values.json")
FAILURES = {"failed.json", "late-completion.json", "cleanup-error.json",
            "completion-before-cleanup-error.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def verify_members(directory, manifest):
    require(type(manifest) is dict and manifest, "Nonempty manifest required")
    for name, record in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name, "Unsafe member name")
        path = directory / relative
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(directory.resolve()), "Missing or unsafe payload: " + name)
        require(binding(path) == record, "Altered payload: " + name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(actual == set(manifest), "Extra or missing sealed payload")


def nonnegative(value, name):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "Invalid " + name)
    return float(value)


def close(actual, expected, name):
    require(math.isclose(nonnegative(actual, name), nonnegative(expected, name),
                         rel_tol=1e-10, abs_tol=1e-12), "Inconsistent " + name)


def authenticate(completed, summary, completed_sha256, summary_sha256, receipt_sha256=None):
    require(sha(completed) == completed_sha256 and sha(summary) == summary_sha256, "External input hash mismatch")
    run, report = completed.parent, summary.parent
    require(completed.name == "completed.json" and summary.name == "summary.json", "Expected input filenames")
    require(not any((folder / name).exists() for folder in (run, report) for name in FAILURES),
            "Failed or demoted execution/audit")
    receipt_path = report / "receipt.json"
    if receipt_sha256 is not None:
        require(sha(receipt_path) == receipt_sha256, "External audit receipt hash mismatch")
    receipt = read_json(receipt_path)
    require(receipt["status"] == "completed" and receipt["execution_completed_sha256"] == completed_sha256,
            "Completed saved-output audit required")
    require(set(receipt["files"]) == {"summary.json", "window-errors.npz"}, "Audit payload membership")
    verify_members(report, {**receipt["files"], "receipt.json": binding(receipt_path)})
    done, audited = read_json(completed), read_json(summary)
    require(done["status"] == audited["status"] == "completed" and audited["study"] == "pose-capacity-v1"
            and audited["execution_completed_sha256"] == completed_sha256, "Completed capacity diagnosis required")
    expected_names = {"started.json", "completed.json"} | {
        f"{p}-{s}-{suffix}" for p in PANELS for s in SEEDS
        for suffix in ("traces.json.gz", "bounds.npz", "row.json")}
    manifest = {**done["files"], "completed.json": binding(completed)}
    require(set(manifest) == expected_names and receipt["execution_members"] == manifest, "Exact20 execution members")
    verify_members(run, manifest)
    protocol_path = run.parent / "protocol.json"
    protocol = read_json(protocol_path)
    require(sha(protocol_path) == done["protocol_sha256"] == audited["protocol_sha256"] == receipt["protocol_sha256"],
            "Protocol hash binding")
    require(protocol["study"] == "pose-capacity-v1" and protocol["rows"] == 6
            and protocol["seeds"] == list(SEEDS) and protocol["panels"] == list(PANELS)
            and protocol["windows_per_row"] == 160 and protocol["horizon"] == 25
            and protocol["constant_searches"] == done["constant_searches"] == 960,
            "Fixed capacity coverage")
    require(audited["no_qualification_gate"] is True and receipt["no_qualification_gate"] is True,
            "Diagnosis must not become a qualification gate")
    counts = audited["counts"]
    for key, value in {"rows": 6, "windows_per_row": 160, "horizon": 25, "controls": 33,
                       "execution_files": 20, "verified_searches": 960, "new_model_calls": 0,
                       "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
                       "new_constant_optimizations_by_audit": 0}.items():
        require(type(counts[key]) is int and counts[key] == value, "Audit coverage: " + key)
    require(set(audited["rows"]) == set(PANELS) and len(done["rows"]) == 6, "All6 rows required")
    identities = set()
    for row in done["rows"]:
        key = row["panel"], row["seed"]
        require(key[0] in PANELS and key[1] in SEEDS and key not in identities, "Duplicate or foreign row")
        identities.add(key)
        require(row == audited["rows"][key[0]][str(key[1])]["work"]
                == read_json(run / f"{key[0]}-{key[1]}-row.json"), "Saved row binding")
    certificates = audited["certificate_summary"]
    require(certificates["searches"] == 960 and type(certificates["certified"]) is int
            and 0 <= certificates["certified"] <= 960
            and certificates["uncertified"] == 960 - certificates["certified"]
            and certificates["all_certified"] == (certificates["certified"] == 960)
            and done["certified_searches"] == receipt["certified_searches"] == certificates["certified"],
            "Certification counts")
    statuses = certificates["statuses"]
    require(set(statuses) <= {"tolerance_reached", "evaluation_cap", "floating_interval_limit"}
            and all(type(x) is int and x >= 0 for x in statuses.values())
            and sum(statuses.values()) == 960
            and statuses.get("tolerance_reached", 0) == certificates["certified"], "Certification status counts")
    return audited, protocol, {"completed_sha256": completed_sha256, "summary_sha256": summary_sha256,
                              "audit_receipt_sha256": sha(receipt_path), "protocol_sha256": sha(protocol_path),
                              "external_audit_receipt_pin": receipt_sha256 is not None,
                              "execution_members_verified": 20, "audit_members_verified": 3}


def plot_values(summary, protocol):
    """Pool the three seed MSE values before square root; never average RMSEs."""
    output = {}
    for panel in PANELS:
        require(set(summary["rows"][panel]) == {str(s) for s in SEEDS}, "All three paired seeds required")
        families = summary["controls"]["families"][panel]
        references = summary["controls"]["references"][panel]
        require(not (set(families) & set(references)), "Ambiguous control identity")
        controls = {**families, **references}
        require(len(controls) == 33 and sorted(controls) == protocol["controls"], "All33 original controls required")
        output[panel] = {}
        for endpoint in ENDPOINTS:
            control_mse = {name: nonnegative(row[endpoint]["mse"], "control MSE") for name, row in controls.items()}
            strongest = min(control_mse, key=lambda name: (control_mse[name], name))
            target_mse = .81 * control_mse[strongest]
            result = {"strongest_control": strongest, "strongest_control_mse": control_mse[strongest],
                      "strongest_control_rmse": math.sqrt(control_mse[strongest]),
                      "target_mse": target_mse, "target_rmse": math.sqrt(target_mse),
                      "fast_rmse": math.sqrt(control_mse["decay_huber3"]),
                      "slow_rmse": math.sqrt(control_mse["gru"]),
                      "all_33_controls": {name: {"mse": value, "rmse": math.sqrt(value),
                          "target_mse": .81 * value, "target_rmse": math.sqrt(.81 * value)}
                          for name, value in sorted(control_mse.items())}, "methods": {}}
            for method in METHODS:
                rows = [summary["rows"][panel][str(seed)]["metrics"][method][endpoint] for seed in SEEDS]
                lowers = [nonnegative(row["lower_mse"], "lower MSE") for row in rows]
                uppers = [nonnegative(row["upper_mse"], "upper MSE") for row in rows]
                require(all(low <= high for low, high in zip(lowers, uppers, strict=True)), "Reversed interval")
                lower, upper = math.fsum(lowers) / 3, math.fsum(uppers) / 3
                family = summary["families"][panel][method][endpoint]
                close(family["lower_mse"], lower, "pooled lower MSE")
                close(family["upper_mse"], upper, "pooled upper MSE")
                if endpoint == "position" or method != "continuous":
                    require(lowers == uppers, "Point optimum must have equal endpoints")
                comparison = summary["comparisons"][panel][method][endpoint]
                require(comparison["control_count"] == 33 and set(comparison["controls"]) == set(controls),
                        "All-control comparison coverage")
                close(comparison["controls"][strongest]["threshold_mse"], target_mse, "10percent threshold")
                result["methods"][method] = {"lower_mse": lower, "upper_mse": upper,
                    "lower_rmse": math.sqrt(lower), "upper_rmse": math.sqrt(upper),
                    "seeds": [{"seed": seed, "lower_rmse": math.sqrt(low), "upper_rmse": math.sqrt(high)}
                              for seed, low, high in zip(SEEDS, lowers, uppers, strict=True)],
                    "projected_rotation_only": method == "continuous" and endpoint == "rotation"}
            output[panel][endpoint] = result
    return output


def figure(values, certificate, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pose-capacity-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = ("#366a81", "#64868d", "#42464b", "#c17c32", "#d69b58", "#7855a0")
    for col, panel in enumerate(PANELS):
        for row, endpoint in enumerate(ENDPOINTS):
            ax, data = axes[row, col], values[panel][endpoint]
            labels = ["Fast expert", "GRU expert", "Strongest of 33 controls",
                      "Fixed hard choice (hindsight)", "Per-step hard choice (hindsight)",
                      "Continuous position optimum" if endpoint == "position" else "Continuous rotation (projected)"]
            base_values = [data["fast_rmse"], data["slow_rmse"], data["strongest_control_rmse"]]
            for y, value in enumerate(base_values):
                ax.barh(y, value, height=.52, color=colors[y], alpha=.75)
            for y, method in enumerate(METHODS, start=3):
                metric = data["methods"][method]
                low, high = metric["lower_rmse"], metric["upper_rmse"]
                if metric["projected_rotation_only"]:
                    ax.plot([low, high], [y, y], color=colors[y], linewidth=6, solid_capstyle="butt")
                    ax.scatter([low, high], [y, y], marker="|", color=colors[y], s=150, zorder=4)
                    ax.scatter(high, y, marker="D", color=colors[y], s=34, zorder=6)
                    ax.text(.99, .03, f"Projected pooled interval: [{low:.6g}, {high:.6g}] rad",
                            transform=ax.transAxes, ha="right", fontsize=9, color=colors[y])
                else:
                    ax.barh(y, high, height=.52, color=colors[y], alpha=.32)
                    ax.plot([high, high], [y - .27, y + .27], color=colors[y], linewidth=2)
                for i, (seed_row, marker) in enumerate(zip(metric["seeds"], MARKERS, strict=True)):
                    ordinate = y + (i - 1) * .13
                    ax.plot([seed_row["lower_rmse"], seed_row["upper_rmse"]], [ordinate, ordinate],
                            color=colors[y], linewidth=1.2, alpha=.8)
                    ax.scatter(seed_row["upper_rmse"], ordinate, color=colors[y], marker=marker,
                               s=28, edgecolor="white", linewidth=.4, zorder=5)
            ax.axvline(data["target_rmse"], color="#b93538", linestyle="--", linewidth=1.5)
            ax.axhline(2.5, color="#aaaaaa", linewidth=.7)
            ax.set_yticks(range(6), labels)
            ax.set_ylim(5.65, -.6)
            ax.set_xlim(left=0)
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            unit = "meters" if endpoint == "position" else "radians"
            ax.set_xlabel(f"{endpoint.capitalize()} RMSE ({unit}); lower is better")
            panel_name = "Plain archive" if col == 0 else "Zigzag archive"
            ax.set_title(f"{panel_name}\nStrongest: {data['strongest_control']}; 10% target = {data['target_rmse']:.5g}",
                         fontsize=11, pad=11)
    fig.suptitle("Hindsight capacity of frozen pose-expert outputs", fontsize=19, weight="bold")
    legend = [Patch(facecolor="#aaaaaa", alpha=.6, label="Pooled RMSE"),
              Line2D([], [], color="#b93538", linestyle="--", label="10% below strongest control"),
              Line2D([], [], color="#7855a0", marker="D", label="Projected interval; diamond = upper bound")]
    legend.extend(Line2D([], [], marker=marker, color="#555555", linestyle="none", label=f"Seed {seed}")
                  for seed, marker in zip(SEEDS, MARKERS, strict=True))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .938), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .19, 1, .9))
    capped = certificate["statuses"].get("evaluation_cap", 0)
    limited = certificate["statuses"].get("floating_interval_limit", 0)
    fig.text(.025, .025,
             "Future targets choose all oracle routes/coefficients. These are hindsight limits, not deployable forecasts or a qualification pass.\n"
             "All 6 rows retained: 3 seeds × 160 windows × 25 steps per panel. Squared errors are pooled before taking square roots.\n"
             "Fixed hard and continuous choices stay fixed across each forecast; per-step hard routing is not a bound on continuous mixing.\n"
             "Rotation interval: guarded float64 SVD-projected interpolation only; it does not bound original float32 production error.\n"
             f"Rotation certificates: {certificate['certified']}/960; evaluation-cap exits: {capped}; floating-interval exits: {limited}. "
             "Every uncertified interval is retained.\n"
             "The dashed threshold is a necessary mean-margin comparison only; prior gates, paired/parent checks and latency remain unchanged.",
             fontsize=9.5, linespacing=1.55)
    fig.savefig(out / "capacity.png", dpi=160, metadata={"Software": "OpenJev saved-output plotter"})
    fig.savefig(out / "capacity.svg", metadata={"Date": None})
    plt.close(fig)


def render(completed, summary, out, *, completed_sha256, summary_sha256, receipt_sha256=None):
    completed, summary, out = Path(completed), Path(summary), Path(out)
    require(not out.resolve().is_relative_to(completed.parent.resolve())
            and not out.resolve().is_relative_to(summary.parent.resolve()), "Output must be outside sealed input directories")
    out.mkdir(parents=True, exist_ok=False)
    start, source = time.perf_counter(), sha(__file__)
    try:
        write_json(out / "started.json", {"status": "started", "source_sha256": source,
            "expected_completed_sha256": completed_sha256, "expected_summary_sha256": summary_sha256,
            "expected_audit_receipt_sha256": receipt_sha256})
        audited, protocol, inputs = authenticate(completed, summary, completed_sha256, summary_sha256, receipt_sha256)
        values = plot_values(audited, protocol)
        write_json(out / "plotted-values.json", {"study": "pose-capacity-v1", "scope": "future-target hindsight only",
            "no_qualification_gate": True, "panels": values, "certificate_summary": audited["certificate_summary"]})
        figure(values, audited["certificate_summary"], out)
        require(sha(completed) == completed_sha256 and sha(summary) == summary_sha256
                and sha(summary.parent / "receipt.json") == inputs["audit_receipt_sha256"], "Inputs changed while plotting")
        receipt = {"status": "completed", "study": "pose-capacity-v1", "inputs": inputs,
            "source_sha256": source, "all_rows_retained": 6, "no_qualification_gate": True,
            "new_model_calls": 0, "new_native_calls": 0, "new_optimization_calls": 0,
            "scope": "saved audited arithmetic and visualization; not a new numerical certificate or forecast",
            "wall_seconds": time.perf_counter() - start,
            "files": {name: binding(out / name) for name in ("started.json", *FILES)}}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        plt.close("all")
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                "source_sha256": source, "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - retain original exception
            if callable(getattr(error, "add_note", None)):
                error.add_note(f"Unable to save failure receipt: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = render(args.completed, args.summary, args.out, completed_sha256=args.completed_sha256,
                    summary_sha256=args.summary_sha256, receipt_sha256=args.receipt_sha256)
    print(json.dumps({"status": result["status"], "out": str(args.out), "files": list(FILES)}))
