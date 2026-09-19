"""Render externally pinned, audited probe-blend results without inference."""
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
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PANELS = ("test_sin", "test_zigzag")
SEEDS = (1101, 1202, 1303)
ENDPOINTS = ("position", "rotation")
VARIANTS = ("fast", "slow", "half", "probe_half", "probe_inverse", "probe_fit")
ALIASES = {"fast": "decay_huber3", "slow": "gru", "half": "half"}
NEW = {"probe_half", "probe_inverse", "probe_fit"}
LABELS = ("Fast expert", "GRU expert", "Half blend", "Probe + half", "Probe inverse", "Probe fit (primary)")
COLORS = ("#63818d", "#63818d", "#a5afb3", "#a5afb3", "#c89b50", "#087f78")
MARKERS = ("o", "s", "^")
FILES = ("physical-errors.png", "physical-errors.svg", "prediction-cost.png", "prediction-cost.svg", "plotted-values.json")
FAILURES = {"failed.json", "late-completion.json", "cleanup-error.json", "completion-before-cleanup-error.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def verify_members(directory, manifest):
    require(type(manifest) is dict and manifest, "Nonempty member manifest required")
    for name, record in manifest.items():
        require(type(name) is str, "String member name required")
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == name,
                "Unsafe member name")
        path = directory / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()),
                "Missing or unsafe member: " + name)
        require(binding(path) == record, "Altered payload: " + name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(actual == set(manifest), "Extra or missing sealed payload")


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            "Invalid " + name)
    return float(value)


def close(actual, expected, name):
    require(math.isclose(number(actual, name), number(expected, name), rel_tol=1e-9, abs_tol=1e-12),
            "Inconsistent " + name)


def authenticate(completed, summary, *, completed_sha256, summary_sha256, receipt_sha256, protocol_sha256):
    run, report = completed.parent, summary.parent
    require(completed.name == "completed.json" and summary.name == "summary.json", "Expected completed/summary filenames")
    require(not any((folder / marker).exists() for folder in (run, report) for marker in FAILURES),
            "Failed or demoted execution/audit")
    receipt_path, protocol_path = report / "receipt.json", run.parent / "protocol.json"
    pins = ((completed, completed_sha256), (summary, summary_sha256), (receipt_path, receipt_sha256),
            (protocol_path, protocol_sha256))
    require(all(sha(path) == digest for path, digest in pins), "External artifact hash mismatch")
    done, audited, receipt, protocol = map(read_json, (completed, summary, receipt_path, protocol_path))
    require(done["status"] == audited["status"] == receipt["status"] == "completed"
            and audited["study"] == protocol["study"] == "pose-probe-blend-v1", "Completed probe study required")
    require(audited["execution_completed_sha256"] == receipt["execution_completed_sha256"] == completed_sha256
            and done["protocol_sha256"] == audited["protocol_sha256"] == receipt["protocol_sha256"] == protocol_sha256,
            "Audit/protocol identity binding")
    require(set(receipt["files"]) == {"summary.json", "window-errors.npz"}, "Exact audit payload membership")
    verify_members(report, {**receipt["files"], "receipt.json": binding(receipt_path)})
    names = {"started.json", "completed.json", *(p + "-inputs.npz" for p in PANELS)}
    for panel in PANELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                prefix = f"{panel}-{variant}-{seed}"
                names.update((prefix + "-predictions.npz", prefix + "-evaluation.json"))
                if variant in NEW:
                    names.add(prefix + "-probe.npz")
    manifest = {**done["files"], "completed.json": binding(completed)}
    require(len(names) == 94 and set(manifest) == names and receipt["execution_members"] == manifest,
            "Exact94 execution member binding")
    verify_members(run, manifest)
    require(protocol["variants"] == list(VARIANTS) and protocol["panels"] == list(PANELS)
            and protocol["seeds"] == list(SEEDS) and protocol["primary"] == "probe_fit"
            and protocol["windows_per_panel"] == 160 and protocol["context"] == 32 and protocol["horizon"] == 25
            and protocol["execution_files"] == 94 and protocol["evaluation_rows"] == 36
            and protocol["controls"] == 35 and protocol["canonical_configurations"] == 36,
            "Fixed study coverage")
    for key, value in {"forecast_calls": 864, "batch_forecast_calls": 36, "single_forecast_calls": 828}.items():
        require(type(done[key]) is int and done[key] == protocol[key] == value, "Complete call count: " + key)
    require(len(protocol["sources"]) == 36 and receipt["sources"] == protocol["sources"], "Source closure identity")
    snapshots = run.parent / "source-snapshot"
    require({p.relative_to(snapshots).as_posix(): sha(p) for p in snapshots.rglob("*") if p.is_file()}
            == protocol["sources"], "Frozen source snapshot identity")
    gate = audited["continuation_gate"]
    require(gate["total_requirements"] == len(gate["requirements"]) == 17
            and gate["total_checks"] == len(gate["checks"]) == 2101
            and len({x["name"] for x in gate["checks"]}) == 2101, "Complete gate coverage")
    for rows, count in ((gate["checks"], "checks_passed"), (gate["requirements"], "requirements_passed")):
        require(all(type(x["passed"]) is bool for x in rows) and gate[count] == sum(x["passed"] for x in rows),
                "Gate status/count binding")
    require(type(gate["passed"]) is bool and gate["passed"] == all(x["passed"] for x in gate["checks"])
            == all(x["passed"] for x in gate["requirements"])
            and receipt["qualification_passed"] == gate["passed"], "Gate qualification binding")
    identities = set()
    require(len(done["rows"]) == 36 and set(audited["active_rows"]) == set(PANELS), "All36 fresh rows required")
    for row in done["rows"]:
        identity = row["panel"], row["variant"], row["seed"]
        require(identity[0] in PANELS and identity[1] in VARIANTS and identity[2] in SEEDS and identity not in identities,
                "Duplicate or foreign fresh row")
        identities.add(identity)
        require(row == read_json(run / ("-".join(map(str, identity)) + "-evaluation.json")), "Fresh row payload binding")
        active = audited["active_rows"][identity[0]][f"{identity[1]}-{identity[2]}"]
        require(active["variant"] == identity[1] and active["seed"] == identity[2]
                and active["latency_ms"] == row["latency_ms"], "Fresh audited row identity/timing")
        for endpoint, metric in (("position", "position_rmse_m"), ("rotation", "rotation_rmse_rad")):
            close(active["errors"][endpoint]["rmse"], row["metrics"][metric], "audited fresh RMSE")
    require(identities == {(p, v, s) for p in PANELS for v in VARIANTS for s in SEEDS}, "Exact fresh identity coverage")
    return audited, {"completed_sha256": completed_sha256, "summary_sha256": summary_sha256,
                     "audit_receipt_sha256": receipt_sha256, "protocol_sha256": protocol_sha256,
                     "execution_members_verified": 94, "audit_members_verified": 3,
                     "source_snapshots_verified": 36}


def plot_values(summary):
    """Use fresh row errors; canonical aliases retain the prior audit's errors."""
    panels, latency = {}, {}
    for panel in PANELS:
        active = summary["active_rows"][panel]
        require(set(active) == {f"{v}-{s}" for v in VARIANTS for s in SEEDS}, "All18 fresh rows in each panel")
        families, references = summary["families"][panel], summary["references"][panel]
        require(not set(families) & set(references) and NEW <= set(families)
                and len(families) == 30 and len(references) == 6, "All36 canonical configurations")
        retained = {**{k: v for k, v in families.items() if k not in NEW}, **references}
        require(len(retained) == 33, "All33 retained prior controls")
        panels[panel] = {}
        for endpoint in ENDPOINTS:
            prior_mse = {name: number(value[endpoint]["mse"], "prior MSE") for name, value in retained.items()}
            best = min(prior_mse, key=lambda name: (prior_mse[name], name))
            variants = {}
            for variant in VARIANTS:
                seed_values = []
                for seed in SEEDS:
                    metric = active[f"{variant}-{seed}"]["errors"][endpoint]
                    mse = number(metric["mse"], "fresh MSE")
                    close(metric["rmse"], math.sqrt(mse), "fresh MSE/RMSE")
                    seed_values.append({"seed": seed, "mse": mse, "rmse": math.sqrt(mse)})
                pooled = math.fsum(x["mse"] for x in seed_values) / 3
                if variant not in ALIASES:
                    close(families[variant][endpoint]["mse"], pooled, "pooled fresh MSE")
                variants[variant] = {"mse": pooled, "rmse": math.sqrt(pooled), "seeds": seed_values,
                                     "canonical_alias": ALIASES.get(variant, variant)}
            panels[panel][endpoint] = {"variants": variants, "strongest_prior_control": best,
                "strongest_prior_rmse": math.sqrt(prior_mse[best]), "prior_mean_margin_rmse": .9 * math.sqrt(prior_mse[best]),
                "all_33_prior_controls": {name: {"mse": mse, "rmse": math.sqrt(mse)} for name, mse in sorted(prior_mse.items())}}
    for variant in VARIANTS:
        samples = []
        for panel in PANELS:
            for seed in SEEDS:
                row = summary["active_rows"][panel][f"{variant}-{seed}"]["latency_ms"]
                require(type(row) is list and len(row) == 20, "All20 timed calls per row")
                samples.extend(number(x, "latency", positive=True) for x in row)
        median, p95 = float(np.median(samples)), float(np.percentile(samples, 95))
        canonical = summary["latency"][ALIASES.get(variant, variant)]
        require(canonical["samples"] == len(samples) == 120, "All120 timed calls per variant")
        close(canonical["median_ms"], median, "fresh latency median")
        close(canonical["p95_ms"], p95, "fresh latency p95")
        latency[variant] = {"samples": 120, "median_ms": median, "p95_ms": p95, "all_samples_ms": samples}
    return {"panels": panels, "latency": latency, "latency_limit_ms": 1.5 * latency["slow"]["median_ms"]}


def gate_label(gate):
    return (f"Gate {'PASS' if gate['passed'] else 'FAIL'}: {gate['requirements_passed']}/17 groups, "
            f"{gate['checks_passed']}/2,101 checks")


def save_figure(fig, out, stem):
    fig.savefig(out / (stem + ".png"), dpi=160, metadata={"Software": "OpenJev saved-output plotter"})
    fig.savefig(out / (stem + ".svg"), metadata={"Date": None})
    plt.close(fig)


def figures(values, gate, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pose-probe-blend-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    for col, panel in enumerate(PANELS):
        for row, endpoint in enumerate(ENDPOINTS):
            ax, data = axes[row, col], values["panels"][panel][endpoint]
            for y, (variant, color) in enumerate(zip(VARIANTS, COLORS, strict=True)):
                metric = data["variants"][variant]
                ax.barh(y, metric["rmse"], height=.57, color=color, alpha=.8 if variant == "probe_fit" else .36)
                for i, (seed, marker) in enumerate(zip(metric["seeds"], MARKERS, strict=True)):
                    ax.scatter(seed["rmse"], y + (i - 1) * .13, marker=marker, color=color,
                               s=28, edgecolor="white", linewidth=.35, zorder=4)
            ax.barh(6, data["strongest_prior_rmse"], height=.57, color="#41464a", alpha=.75)
            ax.axhline(5.5, color="#bbbbbb", linewidth=.8)
            ax.axvline(data["prior_mean_margin_rmse"], color="#b43d45", linestyle="--", linewidth=1.25)
            ax.set_yticks(range(7), (*LABELS, "Strongest prior control"))
            ax.set_ylim(6.65, -.6)
            ax.set_xlim(left=0)
            ax.grid(axis="x", alpha=.17)
            ax.set_axisbelow(True)
            unit = "meters" if endpoint == "position" else "radians"
            ax.set_xlabel(f"{endpoint.capitalize()} RMSE ({unit}); lower is better")
            archive = "Plain archive" if col == 0 else "Zigzag archive"
            ax.set_title(f"{archive} | {endpoint.capitalize()}\nPrior control: {data['strongest_prior_control']}", fontsize=11)
    fig.suptitle("Past-error probe blending", fontsize=19, weight="bold", y=.987)
    fig.text(.5, .945, "Exposed development data | " + gate_label(gate), ha="center", fontsize=11)
    legend = [Patch(facecolor="#63818d", alpha=.4, label="Pooled RMSE across all three seeds"),
              Line2D([], [], color="#b43d45", linestyle="--", label="10% below strongest prior control")]
    legend.extend(Line2D([], [], marker=m, color="#555555", linestyle="none", label=f"Seed {s}")
                  for s, m in zip(SEEDS, MARKERS, strict=True))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .919), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .855))
    fig.text(.025, .025,
             "All 36 fresh rows retained: 3 seeds × 160 windows × 25 steps per archive. Pool squared errors before square root.\n"
             "Coefficients use five completed past steps, then stay fixed for 25 future steps. No hindsight oracle is plotted.\n"
             "Dashed lines show only the prior-control mean margin; the gate also requires every new control, paired/parent checks and latency.\n"
             "Repeatedly exposed simulated-robot data and supplied future applied torques: no fresh confirmation or closed-loop control claim.",
             fontsize=9, linespacing=1.5)
    save_figure(fig, out, "physical-errors")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for y, (variant, color) in enumerate(zip(VARIANTS, COLORS, strict=True)):
        timing = values["latency"][variant]
        median, p95 = timing["median_ms"], timing["p95_ms"]
        ax.barh(y, median, height=.55, color=color, alpha=.8 if variant == "probe_fit" else .55)
        ax.plot([median, p95], [y, y], color=color, linewidth=2)
        ax.scatter(p95, y, marker="D", color=color, s=35, zorder=3)
        ax.annotate(f"{median:.2f} / {p95:.2f}", (p95, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.axvline(values["latency_limit_ms"], color="#b43d45", linestyle="--", linewidth=1.4)
    ax.set_yticks(range(6), LABELS)
    ax.set_ylim(5.65, -.65)
    ax.set_xlim(0, max(max(x["p95_ms"] for x in values["latency"].values()), values["latency_limit_ms"]) * 1.23)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.set_xlabel("Full single-window forecast time (ms); labels show median / p95")
    fig.legend(handles=[Patch(facecolor="#63818d", label="Median"),
                        Line2D([], [], color="#63818d", marker="D", label="95th percentile"),
                        Line2D([], [], color="#b43d45", linestyle="--", label="1.5 × fresh GRU median")],
               loc="upper center", bbox_to_anchor=(.5, .885), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Probe and selection costs are included", fontsize=18, weight="bold", y=.985)
    fig.text(.5, .925, "Exposed development data | " + gate_label(gate), ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .17, 1, .835))
    fig.text(.025, .025,
             "120 timed calls per variant: 20 windows × 3 seeds × 2 archives; 3 warmups per row excluded. One CPU thread.\n"
             "Includes probe fitting/rollouts, selection, deployment experts, blending, validation and returned diagnostics.\n"
             "Probe + half computes and discards the primary selector. Inverse weighting skips the LS/rotation grid.\n"
             "Excludes loading, saved-array packaging, metrics and artifact I/O; this is instrumented forecast time, not robot latency.",
             fontsize=9, linespacing=1.5)
    save_figure(fig, out, "prediction-cost")


def render(completed, summary, out, *, completed_sha256, summary_sha256, receipt_sha256, protocol_sha256,
           presentation_note=None):
    completed, summary, out = Path(completed), Path(summary), Path(out)
    require(not out.resolve().is_relative_to(completed.parent.resolve())
            and not out.resolve().is_relative_to(summary.parent.resolve()), "Output must be outside sealed inputs")
    out.mkdir(parents=True, exist_ok=False)
    start, source = time.perf_counter(), sha(__file__)
    pins = {"completed_sha256": completed_sha256, "summary_sha256": summary_sha256,
            "receipt_sha256": receipt_sha256, "protocol_sha256": protocol_sha256}
    try:
        write_json(out / "started.json", {"status": "started", "source_sha256": source, "expected_inputs": pins,
            "presentation_note": presentation_note})
        audited, inputs = authenticate(completed, summary, **pins)
        values = plot_values(audited)
        write_json(out / "plotted-values.json", {"study": "pose-probe-blend-v1", "scope": "saved audited development forecasts",
            "continuation_gate": audited["continuation_gate"], "fresh_error_rows": 36,
            "prior_controls_for_reference_selection": 33, **values})
        figures(values, audited["continuation_gate"], out)
        require(sha(__file__) == source and sha(completed) == completed_sha256 and sha(summary) == summary_sha256
                and sha(summary.parent / "receipt.json") == receipt_sha256
                and sha(completed.parent.parent / "protocol.json") == protocol_sha256, "Source/input changed during rendering")
        result = {"status": "completed", "study": "pose-probe-blend-v1", "inputs": inputs,
            "source_sha256": source, "all_fresh_rows_retained": 36, "gate_passed": audited["continuation_gate"]["passed"],
            "new_model_calls": 0, "new_native_calls": 0, "new_optimization_calls": 0,
            "scope": "saved audited arithmetic and figures; no new forecast, independent numerical audit or qualification",
            "presentation_note": presentation_note,
            "files": {name: binding(out / name) for name in ("started.json", *FILES)},
            "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", result)
        return result
    except BaseException as error:
        plt.close("all")
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error), "source_sha256": source,
                "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note(f"Unable to retain plotting failure: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--presentation-note")
    for key in ("completed", "summary", "receipt", "protocol"):
        parser.add_argument(f"--{key}-sha256", required=True)
    args = parser.parse_args()
    render(args.completed, args.summary, args.out, completed_sha256=args.completed_sha256,
           summary_sha256=args.summary_sha256, receipt_sha256=args.receipt_sha256, protocol_sha256=args.protocol_sha256,
           presentation_note=args.presentation_note)
