"""Presentation-only plots from authenticated, independently audited saved JSON.

No array decoding, checkpoint loading, predictor execution or metric replay.
Root invokes this helper only after the original run and audit processes close.
All numerical transformations here are displayed scalar averages and units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("static16", "centered16", "static32", "nystrom16", "centered_nystrom16", "full_gp")
LABELS = ("Static16", "Centered16", "Static32", "Nystrom16", "CenteredNystrom16", "FullGP")
SEEDS = (11, 23, 37)
PHASES = ("base", "shift")
DEFAULT_AUDIT = ROOT / "output/query-feature-delivery-v1/audit.json"


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
        require(isinstance(left, dict) and left.keys() == right.keys(), "saved scalar dictionary schema")
        for key in right:
            compare(left[key], right[key])
    elif isinstance(right, list):
        require(isinstance(left, list) and len(left) == len(right), "saved scalar list schema")
        for a, b in zip(left, right, strict=True):
            compare(a, b)
    elif type(right) is float:
        require(type(left) in (float, int) and math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12), "audited scalar agreement")
    else:
        require(type(left) is type(right) and left == right, "audited identity agreement")


def authenticate(study, audit_path):
    """Hash every manifest member before opening any metrics or audit result."""
    manifest_path = study / "manifest.json"
    manifest_pin = descriptor(manifest_path)
    manifest = read(manifest_path)
    require(isinstance(manifest, dict) and manifest, "nonempty producer manifest")
    for name, expected in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name, "safe manifest member")
        path = study / relative
        require(path.resolve().is_relative_to(study), "manifest member stays inside study")
        require(descriptor(path) == expected, f"unchanged manifest member: {name}")
    actual_names = {p.relative_to(study).as_posix() for p in study.rglob("*")
                    if p.is_file() and p != manifest_path}
    require(actual_names == set(manifest), "complete producer file inventory")
    plan = read(study / "registration.json")
    for relative, expected in plan["sources"].items():
        require(descriptor(ROOT / relative)["sha256"] == expected
                and descriptor(study / "source" / relative)["sha256"] == expected,
                f"current and snapshotted source match: {relative}")
    receipt = read(study / "run-receipt.json")
    require(receipt["state"] == "EXITED" and receipt["exit_code"] == 0, "successful saved producer receipt")
    audit_pin = descriptor(audit_path)
    audit = read(audit_path)
    require(audit["agreement"] is True, "independent saved audit agreement")
    for name, path in (("registration", study / "registration.json"),
                       ("manifest", manifest_path), ("receipt", study / "run-receipt.json")):
        require(audit["admission"][name] == descriptor(path), f"audit {name} binding")
    require(audit["result"]["gate"] == receipt["gate"], "audit/producer outcome binding")
    require(descriptor(manifest_path) == manifest_pin, "manifest unchanged during authentication")
    return {"manifest": manifest_pin, "audit": audit_pin,
            "registration": descriptor(study / "registration.json"),
            "producer_receipt": descriptor(study / "run-receipt.json"),
            "sources": plan["sources"], "manifest_members": manifest}, audit


def plotted_values(study, audit):
    rows, fits, resources = (read(study / name) for name in ("metrics.json", "fits.json", "resources.json"))
    compare(rows, audit["rows"])
    compare(resources, audit["resources"])
    compare(read(study / "summary.json"), audit["result"])
    expected_rows = {(phase, cohort, arm, seed) for phase in PHASES for cohort in range(3)
                     for arm in ARMS for seed in ((None,) if arm == "full_gp" else SEEDS)}
    require(len(rows) == 96 and {(r["phase"], r["cohort"], r["arm"], r["fit_seed"]) for r in rows}
            == expected_rows, "all96 metric rows")
    require(len(fits) == 15 and {(r["arm"], r["fit_seed"]) for r in fits}
            == {(arm, seed) for arm in ARMS[:-1] for seed in SEEDS}, "all15 fits")
    require(len(resources) == 32 and {(r["phase"], r["arm"], r["fit_seed"]) for r in resources}
            == {(phase, arm, seed) for phase in PHASES for arm in ARMS
                for seed in ((None,) if arm == "full_gp" else SEEDS)}, "all32 latency records")
    for fit in fits:
        require(fit["updates"] == 512 and [r["epoch"] for r in fit["curve"]] == list(range(1, 17)),
                "fixed512 updates and complete16-epoch fit curve")
        compare(read(study / f'{fit["arm"]}-{fit["fit_seed"]}-fit.json'), fit)
    cohort_points, group_means, curves, latencies = [], [], [], []
    for phase in PHASES:
        for arm in ARMS:
            points = []
            for cohort in range(3):
                subset = [r for r in rows if (r["phase"], r["arm"], r["cohort"]) == (phase, arm, cohort)]
                point = {"phase": phase, "arm": arm, "cohort": cohort, "fit_count": len(subset),
                         **{key: statistics.mean(r[key] for r in subset) for key in ("regret", "nll")}}
                points.append(point)
                cohort_points.append(point)
            group_means.append({"phase": phase, "arm": arm,
                                **{key: statistics.mean(r[key] for r in points) for key in ("regret", "nll")}})
            subset = [r for r in resources if (r["phase"], r["arm"]) == (phase, arm)]
            latencies.append({"phase": phase, "arm": arm,
                              "mean_fit_median_context_seconds": statistics.mean(r["median_context_seconds"]
                                                                                   for r in subset),
                              "fit_medians": [{"fit_seed": r["fit_seed"],
                                               "seconds": r["median_context_seconds"]} for r in subset]})
    for arm in ARMS[:-1]:
        subset = [r for r in fits if r["arm"] == arm]
        for epoch in range(1, 17):
            values = [r["curve"][epoch - 1]["train_nll"] for r in subset]
            curves.append({"arm": arm, "epoch": epoch, "mean": statistics.mean(values),
                           "minimum": min(values), "maximum": max(values),
                           "fits": [{"fit_seed": r["fit_seed"], "nll": r["curve"][epoch - 1]["train_nll"]}
                                    for r in subset]})
    return {"version": "query-feature-plot-v1", "arm_order": list(ARMS), "labels": list(LABELS),
            "gate": audit["result"]["gate"], "metric_rows": rows, "fits": fits, "resources": resources,
            "cohort_points": cohort_points, "group_means": group_means,
            "training_curves": curves, "latency_groups": latencies,
            "definitions": {"metric_points": "Each cohort averages three fit scores; FullGP has one score.",
                            "metric_means": "Equal average over three independent evaluation cohorts.",
                            "training_band": "Minimum and maximum of three fit NLLs, not a confidence interval.",
                            "latency": "Mean across fit-specific medians of20 full-context repeats; four requests each.",
                            "units": "Seconds stored unchanged; latency chart multiplies by1000 for milliseconds.",
                            "scope": "Scalar presentation only. No novelty, calibrated-uncertainty or inference-speed claim."}}


def figure(values, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#3478A3", "shift": "#D57937"}
    arm_colors = ("#58778C", "#B33D50", "#4D8C74", "#9465A7", "#C58B29")
    phase_labels = {"base": "BASE", "shift": "Compound SHIFT"}
    offsets = {"base": -.16, "shift": .16}
    display_labels = ("Static16", "Centered16", "Static32", "Nystrom16", "Centered\nNystrom16", "FullGP")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, title, ylabel in ((axes[0, 0], "regret", "Decision regret under the exact public GP", "Mean conditional regret"),
                                      (axes[0, 1], "nll", "Predictive negative log likelihood", "Nats per request")):
        for phase in PHASES:
            x = [i + offsets[phase] for i in range(len(ARMS))]
            means = [next(r[metric] for r in values["group_means"] if (r["phase"], r["arm"]) == (phase, arm))
                     for arm in ARMS]
            ax.plot(x, means, "D", markersize=6, color=colors[phase], label=phase_labels[phase], zorder=4)
            for i, arm in enumerate(ARMS):
                points = [r[metric] for r in values["cohort_points"] if (r["phase"], r["arm"]) == (phase, arm)]
                ax.scatter([x[i] + (j - 1) * .045 for j in range(3)], points,
                           s=22, color=colors[phase], alpha=.5, zorder=3)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(ARMS)), display_labels, fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=9)
        if metric == "regret":
            maximum = max(abs(r[metric]) for r in values["cohort_points"])
            ax.set_ylim(bottom=-max(.0001, .06 * maximum))
            ax.axhline(0, color="#777777", linewidth=.7, zorder=1)
    ax = axes[1, 0]
    for arm, label, color in zip(ARMS[:-1], LABELS[:-1], arm_colors, strict=True):
        records = [r for r in values["training_curves"] if r["arm"] == arm]
        x = [r["epoch"] for r in records]
        ax.fill_between(x, [r["minimum"] for r in records], [r["maximum"] for r in records], color=color, alpha=.10)
        ax.plot(x, [r["mean"] for r in records], label=label, color=color, linewidth=1.8)
    ax.set_title("TRAIN NLL: mean and range of three fits", fontsize=12, pad=10)
    ax.set_xlabel("Epoch (32 updates each)")
    ax.set_ylabel("Nats per request")
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax = axes[1, 1]
    for phase in PHASES:
        records = [next(r for r in values["latency_groups"] if (r["phase"], r["arm"]) == (phase, arm)) for arm in ARMS]
        x = [i + offsets[phase] for i in range(len(ARMS))]
        ax.bar(x, [1000 * r["mean_fit_median_context_seconds"] for r in records], width=.28,
               color=colors[phase], alpha=.75, label=phase_labels[phase])
        for i, record in enumerate(records):
            count = len(record["fit_medians"])
            ax.scatter([x[i] + (j - (count - 1)/2) * .035 for j in range(count)],
                       [1000 * row["seconds"] for row in record["fit_medians"]], color="#263238", s=13, zorder=4)
    ax.set_title("Four-request context latency, including archive work", fontsize=12, pad=10)
    ax.set_ylabel("Milliseconds per context")
    ax.set_xticks(range(len(ARMS)), display_labels, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Query-centered nonlinear features: a bounded learning pilot", fontsize=17, y=.98)
    fig.text(.5, .018,
             "Metric dots: three cohorts; fitted models average three fits, FullGP uses one reference. Diamonds average cohorts; no ensemble.\n"
             "Static methods cache each archive; centered methods rebuild per request. FullGP uses the known kernel and unoptimized NumPy.\n"
             "Latencies are descriptive measurements on the first context, not an equal-compute comparison or a speed claim.",
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
    receipt = {"version": "query-feature-plot-v1", "study": str(study), "audit_path": str(audit_path),
               "inputs": inputs, "renderer": descriptor(Path(__file__).resolve()),
               "outputs": {name: descriptor(out / name) for name in ("plotted-values.json", "benchmark.png", "benchmark.pdf")},
               "metric_rows": 96, "fits": 15, "resource_records": 32,
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
