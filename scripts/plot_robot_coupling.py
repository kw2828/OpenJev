"""Presentation only: opaque authentication and saved-scalar robot figures.

Run after the independent audit. No model imports, measurement decoding, fitting,
or metric replay. Arithmetic means below summarize already saved seed metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "robot-coupling-plot-v1"
STUDY_VERSION = "robot-coupling-study-v1"
ARMS = ("chain_memory", "rewired_memory", "chain_instant", "gru_residual", "quadratic_ar2")
FIXED = ("linear_frozen", "quadratic_frozen", "persistence", "velocity")
DIRECT = ("direct_ridge_1", "direct_ridge_100")
LABELS = {
    "chain_memory": "Chain memory", "rewired_memory": "Rewired memory",
    "chain_instant": "Chain instantaneous", "gru_residual": "GRU residual",
    "quadratic_ar2": "Quadratic AR2", "linear_frozen": "Frozen linear AR2",
    "quadratic_frozen": "Frozen quadratic AR2", "persistence": "Persistence",
    "velocity": "Constant velocity", "direct_ridge_1": "Direct ridge (1)",
    "direct_ridge_100": "Direct ridge (100)", "direct_missing": "Direct ridge",
}
CAPTION = (
    "Conditional offline prediction given future realized measured torque; 32 observed steps, 128 forecast steps (12.8 s).\n"
    "New causal per-recording preprocessing. Internal DEV after two-rate selection; not an official benchmark score.\n"
    "Dots show every selected fit; diamonds show arithmetic family means, not an ensemble. No CONFIRM or official TEST access.\n"
    "Direct-history ridge reads all future torques jointly; autoregressive models consume them stepwise.\n"
    "That ridge is a high-resource conditional reference, not a causally matched world model."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def relative_path(folder, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "safe relative path required")
    path = folder / name
    require(path.resolve().is_relative_to(folder.resolve()), "path escapes evidence root")
    return path


def authenticate(study):
    """Hash every study payload and current source before reading metric JSON."""
    study = Path(study).resolve()
    receipt = read_json(study / "receipt.json")
    require(receipt["status"] == "PASS" and not (study / "failure.json").exists(), "closed successful study required")
    require(receipt["sources_before"] == receipt["sources_after"], "source closure mismatch")
    require(receipt["confirmation_decodes"] == receipt["official_test_decodes"] == 0, "closed partition access")
    manifest = read_json(study / "manifest.json")["files"]
    actual = {str(path.relative_to(study)) for path in study.rglob("*") if path.is_file()}
    require(actual == set(manifest) | {"manifest.json", "receipt.json"}, "complete study file roster mismatch")
    inputs = {}
    for name, expected in manifest.items():
        path = relative_path(study, name)
        require(pin(path) == expected, "study pin mismatch: " + name)
        inputs[str(path)] = expected
    for name, expected in receipt["sources_before"].items():
        path = relative_path(ROOT, name)
        require(pin(path) == expected, "source pin mismatch: " + name)
        inputs[str(path)] = expected
    for name in ("manifest.json", "receipt.json"):
        inputs[str(study / name)] = pin(study / name)
    require({"results.json", "resources.json", "fits.json"} <= set(manifest), "missing scalar artifacts")
    return {"inputs": inputs, "receipt": receipt, "study": str(study)}


def finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def extract(results, resources, fits):
    cfg, rows, selection = results["config"], results["rows"], results["selection"]
    require(results["version"] == STUDY_VERSION and tuple(cfg["arms"]) == ARMS, "study version/arms")
    require(cfg["context"] == 32 and cfg["dev_horizon"] == 128 and cfg["horizons"] == [64, 128], "forecast scope")
    require(len(cfg["seeds"]) == 3 and len(set(cfg["seeds"])) == 3
            and len(cfg["learning_rates"]) == 2 and len(cfg["partitions"]["dev"]) == 2, "complete design")
    expected = {(recording, arm, seed, rate, h) for recording in cfg["partitions"]["dev"]
                for arm in ARMS for seed in cfg["seeds"] for rate in cfg["learning_rates"] for h in (64, 128)}
    expected |= {(recording, arm, None, None, h) for recording in cfg["partitions"]["dev"]
                 for arm in (*FIXED, *DIRECT) for h in (64, 128)}
    keyed = {(r["recording"], r["arm"], r["seed"], r["learning_rate"], r["horizon"]): r for r in rows}
    require(len(keyed) == len(rows) and set(keyed) == expected, "complete unique candidate/reference rows required")
    for row in rows:
        require(row["status"] in ("PASS", "FAILED"), "unknown metric status")
        if row["status"] == "PASS":
            require(finite(row["metrics"]["standardized_rmse"]), "invalid saved RMSE")
        else:
            require(row["metrics"] is None, "failed row must not carry a metric")
    fit_keys = {(r["arm"], r["seed"], r["learning_rate"]) for r in fits}
    require(len(fits) == 30 and fit_keys == {(a, s, r) for a in ARMS for s in cfg["seeds"]
                                         for r in cfg["learning_rates"]}, "all 30 fit attempts required")
    require(set(selection["selected_rates"]) == set(ARMS), "selected family roster")
    chosen_direct = selection["selected_direct"]
    require(chosen_direct is None or chosen_direct in DIRECT, "direct selection identity")
    order = [*ARMS, *FIXED, chosen_direct or "direct_missing"]
    panels = []
    for recording in cfg["partitions"]["dev"]:
        entries = []
        for arm in order:
            rate = selection["selected_rates"].get(arm)
            eligible = arm not in ARMS or rate is not None
            require(arm not in ARMS or rate is None or rate in cfg["learning_rates"], "selected rate")
            cohort = cfg["seeds"] if arm in ARMS else [None]
            selected = [keyed[(recording, arm, seed, rate, 128)] for seed in cohort] if eligible and arm != "direct_missing" else []
            points = [{"seed": r["seed"], "value": r["metrics"]["standardized_rmse"]}
                      for r in selected if r["status"] == "PASS"]
            complete = len(points) == len(cohort) and bool(selected)
            entries.append({"arm": arm, "learning_rate": rate, "points": points,
                            "mean": statistics.mean(p["value"] for p in points) if complete else None,
                            "status": "PASS" if complete else "INELIGIBLE" if not selected else "FAILED"})
        panels.append({"recording": recording, "entries": entries})
    resource_keys = {(r["arm"], r["seed"]): r for r in resources}
    require(len(resource_keys) == len(resources), "duplicate resource rows")
    latency = []
    for arm in order:
        seeds = cfg["seeds"] if arm in ARMS else [None]
        points = []
        for seed in seeds:
            row = resource_keys.get((arm, seed))
            if row is not None and row.get("timing") is not None:
                value = row["timing"]["median_seconds"]
                require(finite(value) and value > 0, "invalid request latency")
                points.append({"seed": seed, "value": value * 1000.0})
        complete = len(points) == len(seeds)
        latency.append({"arm": arm, "points": points,
                        "mean": statistics.mean(p["value"] for p in points) if complete else None,
                        "status": "PASS" if complete else "NOT TIMED"})
    return {"version": VERSION, "scope": CAPTION, "panels": panels, "latency_ms": latency,
            "selection": selection, "result": results["result"], "all_rows": rows,
            "resources": resources, "fits": fits, "order": order}


def figure(values, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), sharey=True)
    datasets = [p["entries"] for p in values["panels"]] + [values["latency_ms"]]
    scales = []
    for panel_index, (ax, entries) in enumerate(zip(axes, datasets, strict=True)):
        all_values = [p["value"] for entry in entries for p in entry["points"]]
        positive = [v for v in all_values if v > 0]
        scale = "linear"
        if positive and max(positive) / min(positive) >= 100:
            scale = "log" if len(positive) == len(all_values) else "symlog"
            ax.set_xscale(scale, **({"linthresh": min(positive) / 10} if scale == "symlog" else {}))
        scales.append(scale)
        for y, entry in enumerate(entries):
            color = "#126e82" if entry["arm"] == "chain_memory" else "#68788b" if entry["arm"] in ARMS else "#b16b29"
            offsets = [-.14, 0, .14] if len(entry["points"]) == 3 else [0] * len(entry["points"])
            for offset, point in zip(offsets, entry["points"], strict=True):
                ax.scatter(point["value"], y + offset, s=29, alpha=.8, color=color, zorder=3)
            if entry["mean"] is not None:
                ax.scatter(entry["mean"], y, marker="D", s=62, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
            if entry["status"] != "PASS":
                ax.text(.98, y, entry["status"], transform=ax.get_yaxis_transform(), ha="right", va="center", color="#a33030", fontsize=9)
        ax.set_ylim(len(entries) - .5, -.5)
        ax.set_yticks(range(len(entries)), [LABELS[a] for a in values["order"]])
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(x=.12)
        if panel_index < 2:
            name = values["panels"][panel_index]["recording"].removeprefix("recording_2021_12_15_").removesuffix(".mat")
            ax.set_title(f"DEV {panel_index + 1}: {name.replace('H_', ':').replace('M', '')}\nFull 128-step forecast")
            ax.set_xlabel("FIT-standardized RMSE" + (f" ({scale} scale)" if scale != "linear" else ""))
        else:
            ax.set_title("Selected models: full request\n20 repeats per fit after 3 warmups")
            ax.set_xlabel("Median request latency, ms" + (f" ({scale} scale)" if scale != "linear" else ""))
    gate = values["result"]
    fig.suptitle(f"Robot residual coupling: {gate['passed']}/{gate['total']} screen conditions passed", fontsize=16, y=.965)
    fig.legend([Line2D([], [], marker="o", linestyle="", color="#68788b"),
                Line2D([], [], marker="D", linestyle="", markerfacecolor="white", color="#68788b")],
               ["Individual fit / deterministic reference", "Family arithmetic mean"],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
    fig.text(.04, .025, CAPTION + "\nLatency includes normalization, conditioning, forecasting and denormalization; excludes model loading.", fontsize=10, va="bottom")
    fig.subplots_adjust(left=.14, right=.98, bottom=.25, top=.83, wspace=.22)
    fig.savefig(folder / "benchmark.png", dpi=150)
    fig.savefig(folder / "benchmark.pdf")
    plt.close(fig)
    return scales


def fmt(value):
    return "n/a" if value is None else f"{value:.6g}"


def clean(value):
    return str(value).replace("|", "/").replace("\n", " ")


def tables(values):
    lines = ["# Robot coupling saved results", "", CAPTION.replace("\n", " "), "",
             f"Saved screen outcome: **{values['result']['status']}**, {values['result']['passed']}/{values['result']['total']} conditions.", "",
             "## Every candidate rate and reference", "",
             "Both rates and both reported horizons are retained below. Selection uses pooled DEV H128; these are screening results.", "",
             "| Recording | Model | Rate | Seed | Horizon | Status | Standardized RMSE | Physical RMSE (deg) | Error |",
             "|---|---|---:|---:|---:|---|---:|---:|---|"]
    for row in values["all_rows"]:
        metric = row["metrics"] or {}
        name = row["recording"].removeprefix("recording_2021_12_15_").removesuffix(".mat")
        lines.append(f"| {name} | {LABELS[row['arm']]} | {fmt(row['learning_rate'])} | {row['seed']} | {row['horizon']} | {row['status']} | {fmt(metric.get('standardized_rmse'))} | {fmt(metric.get('physical_rmse_deg'))} | {clean(row.get('error') or '')} |")
    lines += ["", "## Selection", "", "| Family | Selected rate / ridge | Candidate eligibility |", "|---|---|---|"]
    for arm in ARMS:
        candidates = values["selection"]["rate_options"][arm]
        description = "; ".join(f"{r['learning_rate']:g}: {'eligible' if r['eligible'] else 'INELIGIBLE'}" for r in candidates)
        lines.append(f"| {LABELS[arm]} | {fmt(values['selection']['selected_rates'][arm])} | {description} |")
    lines.append(f"| Direct ridge | {values['selection']['selected_direct'] or 'INELIGIBLE'} | {clean(values['selection']['direct_options'])} |")
    lines += ["", "## Costs", "", "All trained attempts are shown; request timing exists only for selected eligible recipes. Missing timing is not zero.", "",
              "| Model | Rate | Seed | Fit status | Weights (B) | State (B) | Buffers (B) | Normalizer (B) | Request median (ms) |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    resources = {(r["arm"], r["seed"], r["learning_rate"]): r for r in values["resources"]}
    cost_rows = []
    for fit in values["fits"]:
        row = {**fit["resources"], "arm": fit["arm"], "seed": fit["seed"], "learning_rate": fit["learning_rate"],
               "fit_status": fit["fit"]["status"], "timing": resources.get((fit["arm"], fit["seed"], fit["learning_rate"]), {}).get("timing")}
        cost_rows.append(row)
    cost_rows += [{**r, "fit_status": "reference"} for r in values["resources"] if r["arm"] not in ARMS]
    for r in cost_rows:
        timing = r.get("timing")
        lines.append(f"| {LABELS[r['arm']]} | {fmt(r['learning_rate'])} | {r['seed']} | {r['fit_status']} | {r['parameter_bytes']} | {r['state_bytes']} | {r['buffer_bytes']} | {r['normalizer_bytes']} | {fmt(timing['median_seconds'] * 1000 if timing else None)} |")
    lines += ["", "Storage covers declared weights, explicit recurrent/history state, buffers and normalizers. Request inputs/outputs are reported in plotted-values.json; temporary workspace is not measured. Models differ in size and precision, so this is not an equal-memory or hardware-independent speed claim.", ""]
    return "\n".join(lines)


def render(study, output):
    auth = authenticate(study)
    study, output = Path(auth["study"]), Path(output).resolve()
    require(not output.is_relative_to(study), "publication must be outside immutable study")
    values = extract(read_json(study / "results.json"), read_json(study / "resources.json"), read_json(study / "fits.json"))
    output.mkdir(parents=True, exist_ok=False)
    values["axis_scales"] = figure(values, output)
    (output / "table.md").write_text(tables(values))
    with (output / "all-candidates.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recording", "arm", "seed", "learning_rate", "horizon", "status", "standardized_rmse", "physical_rmse_deg", "error"])
        writer.writeheader()
        for r in values["all_rows"]:
            writer.writerow({**{k: r[k] for k in ("recording", "arm", "seed", "learning_rate", "horizon", "status")},
                             **{k: (r["metrics"] or {}).get(k) for k in ("standardized_rmse", "physical_rmse_deg")},
                             "error": json.dumps(r.get("error"), sort_keys=True)})
    (output / "plotted-values.json").write_text(json.dumps(values, sort_keys=True, indent=2, allow_nan=False) + "\n")
    require(authenticate(study) == auth, "inputs changed during presentation")
    receipt = {"version": VERSION, "study": str(study), "inputs": auth["inputs"], "script": pin(Path(__file__)),
               "registration_sha256": auth["receipt"]["registration_sha256"],
               "scope": "saved-scalar presentation; independent audit must precede invocation; no numerical replay",
               "outputs": {p.name: pin(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.study, args.output)


if __name__ == "__main__":
    main()
