"""Presentation-only plots from authenticated residual-memory scalar JSON.

No NumPy archives, checkpoints, predictors or scientific metrics are replayed.
Invoke only after the original producer and independent audit have closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("sor", "query_only", "fic", "full", "true_gp")
LABELS = ("SoR", "Query-only", "FIC", "Fitted GP", "True GP")
PHASES = ("base", "shift", "long")
SEEDS = (11, 23, 37)
SOURCES = {
    "src/openjev/research/query_feature_data.py", "src/openjev/research/residual_memory.py",
    "scripts/residual_memory_study.py", "scripts/audit_residual_memory_study.py",
    "tests/test_residual_memory.py", "tests/test_residual_memory_study.py",
    "tests/test_audit_residual_memory_study.py", "research/residual-memory-protocol.md",
    "scripts/audit_query_feature_study.py",
}
DEFAULT_AUDIT = ROOT / "output/residual-memory-delivery-v1/audit.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), f"ordinary evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def compare(left, right):
    if isinstance(right, dict):
        require(isinstance(left, dict) and left.keys() == right.keys(), "saved dictionary schema")
        for key in right:
            compare(left[key], right[key])
    elif isinstance(right, list):
        require(isinstance(left, list) and len(left) == len(right), "saved list schema")
        for a, b in zip(left, right, strict=True):
            compare(a, b)
    elif type(right) is float:
        require(type(left) in (float, int) and math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12), "audited scalar agreement")
    else:
        require(type(left) is type(right) and left == right, "audited identity agreement")


def authenticate(study, audit_path):
    """Verify opaque inventory and source pins before any scientific scalar read."""
    registration_pin = descriptor(study / "registration.json")
    expected_commands = {
        "qualification": [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
                          str(ROOT / "tests/test_residual_memory.py"),
                          str(ROOT / "tests/test_residual_memory_study.py"),
                          str(ROOT / "tests/test_audit_residual_memory_study.py")],
        "run": [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/residual_memory_study.py"),
                "run", "--out", str(study), "--registration-sha256", registration_pin["sha256"]],
        "audit": [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/audit_residual_memory_study.py"),
                  "--folder", str(study), "--output", str(audit_path)],
    }
    path_positions = {"qualification": (0, 4, 5, 6), "run": (0, 1, 4), "audit": (0, 1, 3, 5)}
    closures = {}
    for phase in ("qualification", "run", "audit"):
        path = audit_path.parent / (phase + "-process.json")
        pin = descriptor(path)
        record = read(path)
        require(record["state"] == "EXITED" and type(record["returncode"]) is int
                and record["returncode"] == 0, f"original {phase} process closed successfully")
        require(type(record["elapsed_seconds"]) in (int, float)
                and math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] > 0
                and isinstance(record["command"], list) and record["command"]
                and all(type(part) is str for part in record["command"]), f"original {phase} process metadata")
        command = record["command"].copy()
        require(len(command) == len(expected_commands[phase]), f"exact {phase} argument count")
        for index in path_positions[phase]:
            # Receipts use the repository working directory. Normalize lexical
            # paths without resolving the virtualenv interpreter's symlink.
            command[index] = os.path.abspath(ROOT / command[index])
        require(command == expected_commands[phase], f"exact original {phase} command and study binding")
        require(descriptor(path) == pin, f"unchanged {phase} closure during authentication")
        closures[phase] = {"descriptor": pin, "record": record}
    manifest_path = study / "manifest.json"
    manifest_pin = descriptor(manifest_path)
    manifest = read(manifest_path)
    require(isinstance(manifest, dict) and len(manifest) == 147, "complete 138 payloads plus nine sources")
    for name, expected in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name, "safe manifest member")
        path = study / relative
        require(path.resolve().is_relative_to(study), "manifest member inside study")
        require(descriptor(path) == expected, f"unchanged evidence: {name}")
    actual = {p.relative_to(study).as_posix() for p in study.rglob("*")
              if p.is_file() and p != manifest_path}
    require(actual == set(manifest), "exact producer inventory")
    plan = read(study / "registration.json")
    require(plan["config"]["version"] == "residual-memory-v1"
            and plan["config"]["training_updates"] == 0 and set(plan["sources"]) == SOURCES,
            "registered diagnostic and exact source roster")
    for name, expected in plan["sources"].items():
        require(descriptor(ROOT / name) == expected == descriptor(study / "source" / name),
                f"current and snapshotted source identity: {name}")
    receipt = read(study / "run-receipt.json")
    require(receipt["state"] == "EXITED" and receipt["exit_code"] == 0
            and receipt["training_updates"] == 0, "successful zero-training producer receipt")
    audit_pin = descriptor(audit_path)
    audit = read(audit_path)
    require(audit["version"] == "residual-memory-audit-v1" and audit["agreement"] is True
            and audit["independent_predictions_agree"] is True, "independent saved audit agreement")
    for name, path in (("registration", study / "registration.json"),
                       ("manifest", manifest_path), ("receipt", study / "run-receipt.json")):
        require(audit["admission"][name] == descriptor(path), f"audit {name} binding")
    require(audit["admission"]["sources"] == plan["sources"]
            and audit["admission"]["parent"] == plan["parent"], "audited source and parent bindings")
    require(audit["result"]["gate"] == receipt["gate"], "audited outcome binding")
    require(descriptor(study / "registration.json") == registration_pin,
            "registration unchanged during process admission")
    require(descriptor(manifest_path) == manifest_pin, "unchanged manifest during authentication")
    return {"manifest": manifest_pin, "audit": audit_pin,
            "registration": descriptor(study / "registration.json"),
            "producer_receipt": descriptor(study / "run-receipt.json"),
            "sources": plan["sources"], "parent": plan["parent"], "manifest_members": manifest,
            "original_processes": closures}, audit


def plotted_values(study, audit):
    rows, pairs, resources = (read(study / name) for name in ("metrics.json", "pairwise.json", "resources.json"))
    compare(rows, audit["rows"])
    compare(pairs, audit["pairwise"])
    compare(resources, audit["resources"])
    compare(read(study / "summary.json"), audit["result"])
    expected = {(phase, cohort, mode, seed) for phase in PHASES for cohort in range(3)
                for mode in MODES for seed in ((None,) if mode == "true_gp" else SEEDS)}
    require(len(rows) == 117 and {(r["phase"], r["cohort"], r["mode"], r["fit_seed"]) for r in rows}
            == expected, "all 117 metric rows")
    require(len(pairs) == 81 and {(r["phase"], r["cohort"], r["mode"], r["fit_seed"]) for r in pairs}
            == {r for r in expected if r[2] in MODES[:3]}, "all 81 same-kernel comparisons")
    require(len(resources) == 39 and {(r["phase"], r["mode"], r["fit_seed"]) for r in resources}
            == {(r[0], r[2], r[3]) for r in expected}, "all 39 latency observations")
    points, means, pair_points, pair_means, latencies, defer = [], [], [], [], [], []
    for phase in PHASES:
        for mode in MODES:
            local = []
            for cohort in range(3):
                group = [r for r in rows if (r["phase"], r["mode"], r["cohort"]) == (phase, mode, cohort)]
                point = {"phase": phase, "mode": mode, "cohort": cohort, "fit_count": len(group),
                         **{key: statistics.mean(r[key] for r in group) for key in ("regret", "nll")}}
                points.append(point)
                local.append(point)
            means.append({"phase": phase, "mode": mode,
                          **{key: statistics.mean(r[key] for r in local) for key in ("regret", "nll")}})
            timings = [r for r in resources if (r["phase"], r["mode"]) == (phase, mode)]
            latencies.append({"phase": phase, "mode": mode,
                "mean_fit_median_context_seconds": statistics.mean(r["median_context_seconds"] for r in timings),
                "fit_medians": [{"fit_seed": r["fit_seed"], "seconds": r["median_context_seconds"]} for r in timings]})
            if mode in MODES[:3]:
                local_pairs = []
                for cohort in range(3):
                    group = [r for r in pairs if (r["phase"], r["mode"], r["cohort"]) == (phase, mode, cohort)]
                    point = {"phase": phase, "mode": mode, "cohort": cohort,
                             "mean_weight_l1": statistics.mean(r["mean_weight_l1"] for r in group)}
                    pair_points.append(point)
                    local_pairs.append(point)
                pair_means.append({"phase": phase, "mode": mode,
                    "mean_weight_l1": statistics.mean(r["mean_weight_l1"] for r in local_pairs)})
        # The true-GP rows occur once per cohort, avoiding repeated inherited-fit entries.
        defer.append({"phase": phase, "regret": statistics.mean(r["always_defer_regret"] for r in rows
                                                                 if r["phase"] == phase and r["mode"] == "true_gp")})
    return {"version": "residual-memory-plot-v1", "mode_order": list(MODES), "labels": list(LABELS),
            "gate": audit["result"]["gate"], "registered_result": audit["result"],
            "metric_rows": rows, "pairwise_rows": pairs, "resources": resources,
            "cohort_points": points, "group_means": means, "pairwise_cohort_points": pair_points,
            "pairwise_group_means": pair_means, "latency_groups": latencies, "always_defer": defer,
            "definitions": {
                "metric_points": "Each point is one cohort's mean across three inherited fits; True GP has one reference.",
                "metric_means": "Equal mean of three cohort scores, not ensemble predictions.",
                "pairwise": "Mean identity-weight L1 from each approximation to full GP with the same fitted kernel; True GP is not that reference.",
                "latency": "Mean of fit-specific medians of 20 repeats, after three warmups, on the first context; archive build plus four requests.",
                "units": "Stored seconds are unchanged. Only the latency figure converts them to milliseconds.",
                "scope": "Presentation-only scalar averages. No training, architecture novelty, calibration or general speed claim."}}


def figure(values, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#3478A3", "shift": "#D57937", "long": "#488264"}
    offsets = {"base": -.22, "shift": 0., "long": .22}
    phase_labels = {"base": "BASE", "shift": "Compound SHIFT", "long": "LONG archive"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, title, ylabel in (
        (axes[0, 0], "regret", "Decision regret under the true public-history GP", "Mean conditional regret"),
        (axes[0, 1], "nll", "Predictive negative log likelihood", "Nats per request"),
    ):
        for phase in PHASES:
            x = [i + offsets[phase] for i in range(len(MODES))]
            means = [next(r[metric] for r in values["group_means"] if (r["phase"], r["mode"]) == (phase, mode))
                     for mode in MODES]
            ax.plot(x, means, "D", markersize=6, color=colors[phase], label=phase_labels[phase], zorder=4)
            for i, mode in enumerate(MODES):
                points = [r[metric] for r in values["cohort_points"] if (r["phase"], r["mode"]) == (phase, mode)]
                ax.scatter([x[i] + (j-1)*.035 for j in range(3)], points,
                           s=22, color=colors[phase], alpha=.45, zorder=3)
            if metric == "regret":
                baseline = next(r["regret"] for r in values["always_defer"] if r["phase"] == phase)
                ax.axhline(baseline, linestyle=":", color=colors[phase], alpha=.7, linewidth=1.2,
                           label=phase_labels[phase]+" defer")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(MODES)), LABELS, fontsize=9)
        ax.legend(frameon=False, fontsize=8, ncols=2 if metric == "regret" else 1)
        ax.grid(axis="y", alpha=.2)
        if metric == "regret":
            maximum = max(abs(r[metric]) for r in values["cohort_points"])
            ax.set_ylim(bottom=-max(.0001, .06*maximum))
            ax.axhline(0, color="#777777", linewidth=.7, zorder=1)
    ax = axes[1, 0]
    for phase in PHASES:
        x = [i + offsets[phase] for i in range(3)]
        means = [next(r["mean_weight_l1"] for r in values["pairwise_group_means"]
                      if (r["phase"], r["mode"]) == (phase, mode)) for mode in MODES[:3]]
        ax.plot(x, means, "D", markersize=6, color=colors[phase], label=phase_labels[phase], zorder=4)
        for i, mode in enumerate(MODES[:3]):
            points = [r["mean_weight_l1"] for r in values["pairwise_cohort_points"]
                      if (r["phase"], r["mode"]) == (phase, mode)]
            ax.scatter([x[i]+(j-1)*.035 for j in range(3)], points, s=22, color=colors[phase], alpha=.45)
    ax.set_title("Block-identity weights vs the same fitted full GP", fontsize=12, pad=10)
    ax.set_ylabel("Mean L1 distance (0 to 2)")
    ax.set_xticks(range(3), LABELS[:3], fontsize=9)
    ax.set_ylim(bottom=-.025)
    ax.axhline(0, color="#777777", linewidth=.7)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for phase in PHASES:
        records = [next(r for r in values["latency_groups"] if (r["phase"], r["mode"]) == (phase, mode))
                   for mode in MODES]
        x = [i + offsets[phase] for i in range(len(MODES))]
        ax.bar(x, [1000*r["mean_fit_median_context_seconds"] for r in records], width=.20,
               color=colors[phase], alpha=.75, label=phase_labels[phase])
        for i, record in enumerate(records):
            count = len(record["fit_medians"])
            ax.scatter([x[i]+(j-(count-1)/2)*.025 for j in range(count)],
                       [1000*r["seconds"] for r in record["fit_medians"]], s=13, color="#263238", zorder=4)
    ax.set_title("Archive build plus four cached requests", fontsize=12, pad=10)
    ax.set_ylabel("Milliseconds per context")
    ax.set_xticks(range(len(MODES)), LABELS, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Residual covariance with fixed learned kernels", fontsize=17, y=.98)
    fig.text(.5, .018,
             "Dots: three independent cohorts; fitted methods average three inherited fits. Diamonds: cohort means, not ensembles.\n"
             "Fitted GP retains each inherited kernel; True GP knows the generating kernel. No models are trained in this diagnostic.\n"
             "All modes cache archives. Latency is descriptive on one context per population, not a general speed claim.",
             ha="center", va="bottom", fontsize=9, color="#424A50")
    fig.tight_layout(rect=(0, .105, 1, .95), h_pad=2.4, w_pad=2.0)
    fig.savefig(out / "benchmark.png", dpi=150)
    fig.savefig(out / "benchmark.pdf")
    plt.close(fig)


def render(study, out, audit_path=DEFAULT_AUDIT):
    study, out, audit_path = Path(study).resolve(), Path(out).resolve(), Path(audit_path).resolve()
    require(not out.exists() and not out.is_relative_to(study), "exclusive presentation output outside study")
    inputs, audit = authenticate(study, audit_path)
    values = plotted_values(study, audit)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plotted-values.json", values)
    figure(values, out)
    require(authenticate(study, audit_path)[0] == inputs, "evidence unchanged after rendering")
    receipt = {"version": "residual-memory-plot-v1", "study": str(study), "audit_path": str(audit_path),
               "inputs": inputs, "renderer": descriptor(Path(__file__).resolve()),
               "outputs": {name: descriptor(out / name) for name in ("plotted-values.json", "benchmark.png", "benchmark.pdf")},
               "metric_rows": 117, "pairwise_rows": 81, "resource_records": 39,
               "scope": "Saved scalar visualization only; no arrays, models, training or metric replay. Visual QA remains separate."}
    write(out / "plot-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    result = render(args.study, args.out, args.audit)
    print(json.dumps({"outputs": result["outputs"], "metric_rows": result["metric_rows"]}))


if __name__ == "__main__":
    main()
