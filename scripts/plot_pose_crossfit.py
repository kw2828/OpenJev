"""Plot a completed, audited pose-crossfit study from saved outputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SEEDS = (1101, 1202, 1303)
PANELS = ("test_sin", "test_zigzag")
VARIANTS = ("fast", "slow", "is_constant", "is_summary", "is_recurrent",
            "oof_constant", "oof_summary", "oof_recurrent", "full_constant")
NAMES = {
    "fast": "Fast expert", "slow": "GRU expert",
    "is_constant": "IS constant", "oof_constant": "OOF constant",
    "is_summary": "IS summary", "oof_summary": "OOF summary",
    "is_recurrent": "IS recurrent", "oof_recurrent": "OOF recurrent",
    "full_constant": "Full-training constant",
}
PAIRS = (("is_constant", "oof_constant"), ("is_summary", "oof_summary"),
         ("is_recurrent", "oof_recurrent"))
MARKERS = ("o", "s", "^")
COLORS = {v: ("#ad6727" if v.startswith("is_") else "#087f78" if v.startswith("oof_")
              else "#71569a" if v == "full_constant" else "#42627b") for v in NAMES}
FILES = ("physical-errors.png", "prediction-cost.png", "selector-weights.png")
FAILURE_MARKERS = {"failed.json", "late-completion.json", "cleanup-error.json",
                   "completion-before-cleanup-error.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(), parse_constant=nonfinite)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def verify_members(directory, expected):
    require(isinstance(expected, dict) and expected, "Nonempty member manifest required")
    actual = {}
    for name, record in expected.items():
        relative = Path(name)
        require(isinstance(name, str) and not relative.is_absolute()
                and relative.as_posix() == name and ".." not in relative.parts,
                "Unsafe member path")
        path = directory / relative
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(directory.resolve()), "Missing/unsafe member: " + name)
        actual[name] = binding(path)
        require(actual[name] == record, "Altered member: " + name)
    disk = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(disk == set(expected), "Extra or missing completed members")
    return actual


def authenticate(completed, summary):
    """Verify sealed execution and saved audit before loading plotting arrays.

    Hash consistency relies on the supplied local audit receipt as trust root;
    this is not a replacement numerical audit or an external signed receipt.
    """
    run, audit_dir = completed.parent, summary.parent
    require(completed.name == "completed.json" and summary.name == "summary.json", "Expected completed/summary names")
    require(not any((folder / marker).exists() for folder in (run, audit_dir)
                    for marker in FAILURE_MARKERS), "Failed or demoted execution/audit")
    audit_path = audit_dir / "receipt.json"
    audit = read_json(audit_path)
    require(audit["status"] == "completed", "Completed saved-output audit required")
    audited_members = {**audit["files"], "receipt.json": binding(audit_path)}
    verify_members(audit_dir, audited_members)
    done, audited = read_json(completed), read_json(summary)
    completion_hash = sha(completed)
    require(done["status"] == audited["status"] == "completed", "Completed study/summary required")
    require(audit["execution_completed_sha256"] == audited["execution_completed_sha256"] == completion_hash,
            "Audit completion binding")
    expected = {**done["files"], "completed.json": binding(completed)}
    require(len(expected) == 288 and audit["execution_members"] == expected, "Exact288 execution member binding")
    verify_members(run, expected)
    protocol_path = run.parent / "protocol.json"
    protocol = read_json(protocol_path)
    require(sha(protocol_path) == done["protocol_sha256"] == audit["protocol_sha256"], "Protocol binding")
    require(protocol["study"] == "pose-crossfit-v1" and protocol["evaluation_rows"] == 54
            and protocol["seeds"] == list(SEEDS) and protocol["panels"] == list(PANELS)
            and protocol["variants"] == list(VARIANTS), "Fixed study coverage")
    gate = audited["continuation_gate"]
    require(type(gate["passed"]) is bool and gate["total_requirements"] == 17
            and gate["total_checks"] == 1921 and len(gate["requirements"]) == 17
            and len(gate["checks"]) == 1921, "Complete continuation gate")
    require(all(type(x["passed"]) is bool for x in (*gate["requirements"], *gate["checks"]))
            and gate["requirements_passed"] == sum(x["passed"] for x in gate["requirements"])
            and gate["checks_passed"] == sum(x["passed"] for x in gate["checks"])
            and type(gate["numerical_comparisons_passed"]) is bool
            and gate["numerical_comparisons_passed"] == all(x["passed"] for x in gate["checks"]),
            "Gate status/count binding")
    certificate = audited["constant_certification_pass"]
    require(type(certificate) is bool and gate["constant_certification_pass"] == certificate
            and gate["passed"] == (gate["numerical_comparisons_passed"] and certificate)
            and audit["qualification_passed"] == gate["passed"],
            "Constant certificate and continuation status")
    rows, alphas = {}, {}
    require(len(done["rows"]) == 54, "Exactly54 evaluation rows required")
    for row in done["rows"]:
        key = row["panel"], row["variant"], row["seed"]
        require(key[0] in PANELS and key[1] in VARIANTS and key[2] in SEEDS and key not in rows,
                "Duplicate or foreign evaluation row")
        prefix = f"{key[0]}-{key[1]}-{key[2]}"
        require(read_json(run / (prefix + "-evaluation.json")) == row, "Evaluation row binding")
        for metric in ("position_rmse_m", "rotation_rmse_rad"):
            value = row["metrics"][metric]
            require(type(value) in (float, int) and np.isfinite(value) and value >= 0, "Invalid physical RMSE")
        latencies = np.asarray(row["latency_ms"], dtype=float)
        require(latencies.shape == (20,) and np.isfinite(latencies).all() and (latencies > 0).all(),
                "All20 finite positive row timings required")
        with np.load(run / (prefix + "-predictions.npz"), allow_pickle=False) as payload:
            require(set(payload.files) == {"p", "R", "alpha"}, "Prediction array membership")
            alpha = payload["alpha"].copy()
        require(alpha.dtype == np.float32 and alpha.shape == (160, 2) and np.isfinite(alpha).all()
                and (alpha >= 0).all() and (alpha <= 1).all(), "All160 bounded selector weights required")
        rows[key], alphas[key] = row, alpha
    require(set(rows) == {(p, v, s) for p in PANELS for v in VARIANTS for s in SEEDS}, "All54 identities required")
    return done, audited, rows, alphas, {
        "completed_sha256": completion_hash, "summary_sha256": sha(summary),
        "audit_receipt_sha256": sha(audit_path), "protocol_sha256": sha(protocol_path),
        "execution_members_verified": len(expected), "audit_members_verified": len(audited_members),
        "audit_receipt_trust": "supplied local receipt, hash consistency; no external signature"}


def setup_axes(ax):
    ax.set_yticks(range(len(NAMES)), NAMES.values())
    ax.set_ylim(len(NAMES) - .5, -.5)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for boundary in (1.5, 3.5, 5.5, 7.5):
        ax.axhline(boundary, color="#dddddd", linewidth=.6)


def paired_lines(ax, values, spacing=.12):
    order = list(NAMES)
    for first, second in PAIRS:
        for i in range(3):
            offset = (i - 1) * spacing
            ax.plot([values[first][i], values[second][i]], [order.index(first) + offset, order.index(second) + offset],
                    color="#aaaaaa", linewidth=.8, alpha=.7, zorder=2)


def seed_legend():
    return [Line2D([], [], marker=m, color="#444444", linestyle="none", label=f"Seed {s}", markersize=6)
            for s, m in zip(SEEDS, MARKERS, strict=True)]


def physical_errors(rows, audited, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for col, panel in enumerate(PANELS):
        for index, (metric, unit) in enumerate((("position_rmse_m", "Position RMSE (meters)"),
                                               ("rotation_rmse_rad", "Rotation RMSE (radians)"))):
            ax = axes[index, col]
            values = {v: np.asarray([rows[panel, v, s]["metrics"][metric] for s in SEEDS]) for v in NAMES}
            paired_lines(ax, values)
            for y, v in enumerate(NAMES):
                # Same160 windows and25 horizons for every seed: square-mean-root pools squared errors.
                pooled = float(np.sqrt(np.mean(values[v] ** 2)))
                ax.barh(y, pooled, height=.58, color=COLORS[v], alpha=.18, zorder=1)
                ax.plot([pooled, pooled], [y - .3, y + .3], color=COLORS[v], linewidth=2, zorder=3)
                for i, (value, marker) in enumerate(zip(values[v], MARKERS, strict=True)):
                    ax.scatter(value, y + (i - 1) * .12, marker=marker, s=34, edgecolor="white", linewidth=.35,
                               color=COLORS[v], zorder=4)
            setup_axes(ax)
            ax.set_xlim(left=0)
            ax.set_xlabel(unit + "; lower is better")
            if index == 0:
                ax.set_title("Plain archive" if col == 0 else "Zigzag archive", fontsize=13)
    gate = audited["continuation_gate"]
    certificate = audited["constant_certification_pass"]
    fig.suptitle("OpenJev: parent-excluded selector training", fontsize=18, weight="bold")
    fig.legend(handles=[*seed_legend(), Patch(facecolor="#bbbbbb", alpha=.5, label="Pooled RMSE bar")],
               loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .13, 1, .9))
    fig.text(.025, .025, f"Performance comparisons: {'PASS' if gate['numerical_comparisons_passed'] else 'FAIL'} "
             f"({gate['requirements_passed']}/17 groups; {gate['checks_passed']}/1,921 comparisons). "
             f"All nine constant certificates: {'PASS' if certificate else 'FAIL'}.\n"
             "Bars pool squared errors across all three seeds; thin lines pair IS/OOF seeds. All 160 windows × 25 steps per archive.\n"
             "IS = included-parent expert caches; OOF = excluded-parent expert caches. Shared full-training experts at deployment.\n"
             "Exposed simulated-robot development archives; no fresh confirmation or native control claim. The gate retains all 32 controls.",
             fontsize=10, linespacing=1.5)
    fig.savefig(out / FILES[0], dpi=160)
    plt.close(fig)


def prediction_cost(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for col, panel in enumerate(PANELS):
        ax = axes[col]
        for y, v in enumerate(NAMES):
            values = np.asarray([rows[panel, v, s]["latency_ms"] for s in SEEDS])
            median, p95 = np.percentile(values, [50, 95])
            ax.plot([median, p95], [y, y], color=COLORS[v], linewidth=3)
            ax.plot([median, median], [y - .23, y + .23], color=COLORS[v], linewidth=2)
            for i, (seed_values, marker) in enumerate(zip(values, MARKERS, strict=True)):
                ax.scatter(np.median(seed_values), y + (i - 1) * .12, marker=marker, s=34,
                           color=COLORS[v], edgecolor="white", linewidth=.4, zorder=3)
        setup_axes(ax)
        ax.set_xlim(left=0)
        ax.set_xlabel("Complete forecast milliseconds; lower is better")
        ax.set_title("Plain archive" if col == 0 else "Zigzag archive")
    fig.suptitle("Measured context plus 25-step forecasting cost", fontsize=18, weight="bold")
    fig.legend(handles=seed_legend(), loc="upper center", bbox_to_anchor=(.5, .92), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .16, 1, .88))
    fig.text(.025, .025, "Dots: each seed's median. Vertical ticks: pooled median. Segments: pooled median to p95.\n"
             "One CPU thread; 20 timed windows per seed/panel after 3 warmups (120 samples/configuration across both panels).\n"
             "Mixed outputs pay for both experts, context processing, selector and blending. Loading, normalization, metrics and I/O excluded.\n"
             "These are instrumented local forecast timings, not matched total training compute or closed-loop deployment latency.",
             fontsize=10, linespacing=1.5)
    fig.savefig(out / FILES[1], dpi=160)
    plt.close(fig)


def selector_weights(alphas, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for col, panel in enumerate(PANELS):
        for dimension, endpoint in enumerate(("Position", "Rotation")):
            ax = axes[dimension, col]
            means = {v: [float(alphas[panel, v, s][:, dimension].mean()) for s in SEEDS] for v in NAMES}
            paired_lines(ax, means, spacing=.17)
            for y, v in enumerate(NAMES):
                for i, (seed, marker) in enumerate(zip(SEEDS, MARKERS, strict=True)):
                    values = alphas[panel, v, seed][:, dimension]
                    # Deterministic index offsets display every window, without random jitter.
                    offsets = np.linspace(-.055, .055, len(values)) + (i - 1) * .17
                    ax.scatter(values, y + offsets, color=COLORS[v], s=5, alpha=.16, linewidth=0)
                    ax.scatter(means[v][i], y + (i - 1) * .17, color=COLORS[v], marker=marker, s=38,
                               edgecolor="white", linewidth=.4, zorder=4)
            setup_axes(ax)
            ax.set_xlim(-.025, 1.025)
            ax.set_xlabel(f"{endpoint} coefficient alpha: 0 = GRU expert, 1 = fast expert")
            if dimension == 0:
                ax.set_title("Plain archive" if col == 0 else "Zigzag archive")
    fig.suptitle("Saved selector weights on every evaluation window", fontsize=18, weight="bold")
    fig.legend(handles=seed_legend(), loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .11, 1, .9))
    fig.text(.025, .025, "Faint points: all 160 windows per seed. Large markers: within-seed means; lines pair IS/OOF seed means.\n"
             "Position and rotation use separate coefficients fixed over each 25-step forecast. Neither expert receives mixed-output feedback.\n"
             "Coefficient differences are descriptive; this plot does not establish a coherent joint dynamics model or a new memory architecture.",
             fontsize=10, linespacing=1.5)
    fig.savefig(out / FILES[2], dpi=160)
    plt.close(fig)


def render(completed, summary, out):
    completed, summary, out = Path(completed), Path(summary), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = sha(__file__)
    try:
        write_json(out / "started.json", {"status": "started", "source_sha256": source,
                                          "completed": str(completed), "summary": str(summary)})
        _done, audited, rows, alphas, inputs = authenticate(completed, summary)
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        physical_errors(rows, audited, out)
        prediction_cost(rows, out)
        selector_weights(alphas, out)
        require(sha(completed) == inputs["completed_sha256"] and sha(summary) == inputs["summary_sha256"]
                and sha(summary.parent / "receipt.json") == inputs["audit_receipt_sha256"], "Inputs changed during plotting")
        gate = audited["continuation_gate"]
        receipt = {"status": "completed", "study": "pose-crossfit-v1", "inputs": inputs,
                   "source_sha256": source, "new_model_calls": 0, "new_native_calls": 0,
                   "evaluation_rows": 54, "seeds": list(SEEDS), "variants": list(VARIANTS),
                   "all_fits_retained": True, "performance_gate_passed": gate["numerical_comparisons_passed"],
                   "constant_certification_pass": audited["constant_certification_pass"],
                   "qualification_passed": gate["passed"] and audited["constant_certification_pass"],
                   "scope": "authenticated saved-output visualization; no independent numerical re-audit",
                   "wall_seconds": time.perf_counter() - start,
                   "files": {name: binding(out / name) for name in ("started.json", *FILES)}}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        plt.close("all")
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                                             "source_sha256": source, "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original error
            if callable(getattr(error, "add_note", None)):
                error.add_note(f"Failure receipt could not be written: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.completed, args.summary, args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out), "files": list(FILES)}))
