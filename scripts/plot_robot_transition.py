"""Saved-scalar presentation after original robot-transition run/audit closure.

This helper authenticates opaque producer evidence. The invoking delivery step
must admit original process/audit closure; this is not an independent audit.
No model, measurement decoder, fitting or scoring function is imported.
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
VERSION = "robot-transition-plot-v1"
STUDY_VERSION = "robot-transition-study-v1"
ARMS = ("lpv_recurrent", "lpv_instant", "lpv_constant", "gru_residual")
REFERENCES = ("causal_ridge_1", "causal_ridge_100", "linear_frozen", "persistence")
SOURCES = (
    "src/openjev/research/joint_coupling.py", "src/openjev/research/industrial_robot_data.py",
    "scripts/robot_coupling_study.py", "tests/test_joint_coupling.py",
    "tests/test_industrial_robot_data.py", "tests/test_robot_coupling_study.py",
    "research/robot-coupling-protocol.md", "src/openjev/research/bounded_robot_transition.py",
    "src/openjev/research/causal_robot_ridge.py", "scripts/robot_transition_study.py",
    "tests/test_bounded_robot_transition.py", "tests/test_causal_robot_ridge.py",
    "tests/test_robot_transition_study.py", "research/robot-transition-protocol.md",
)
LABELS = {
    "lpv_recurrent": "Recurrent LPV", "lpv_instant": "Instant LPV",
    "lpv_constant": "Constant LPV", "gru_residual": "GRU residual",
    "causal_ridge_1": "Causal ridge (1)", "causal_ridge_100": "Causal ridge (100)",
    "linear_frozen": "Frozen linear AR2", "persistence": "Persistence",
}
CAPTION = (
    "Development on the same two previously exposed recordings. Both rates are retained in the complete table.\n"
    "Selected-rate dots show all three individual fits; accuracy diamonds are arithmetic means, not an ensemble.\n"
    "32 observed steps followed by 128 forecast steps (12.8 s), conditional on realized measured torque.\n"
    "Each forecast uses only preceding torque inputs. Internal CONFIRM and official TEST remain closed.\n"
    "Different model sizes and precision; no novelty, control-safety or official benchmark claim."
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
    """Opaque payload/source joins before scalar reads; no numerical replay."""
    study = Path(study).resolve()
    receipt = read_json(study / "receipt.json")
    require(receipt["status"] == "PASS" and not (study / "failure.json").exists(), "closed producer required")
    require([receipt[k] for k in ("fits", "rows", "raw_decodes", "saved_fit_loads", "saved_dev_loads")]
            == [24, 112, 0, 7, 2], "producer count roster")
    require(receipt["confirmation_access"] is False and receipt["official_test_access"] is False,
            "closed partition access")
    registration = ROOT / "research/robot-transition-registration.json"
    require(pin(registration)["sha256"] == receipt["registration_sha256"], "original registration hash")
    plan = read_json(registration)
    require(plan["version"] == STUDY_VERSION and set(plan["sources"]) == set(SOURCES), "registered source roster")
    manifest = read_json(study / "manifest.json")
    require(set(manifest) == {"files"}, "manifest schema")
    inventory = manifest["files"]
    paths = list(study.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "symlinks not admitted")
    require({str(p.relative_to(study)) for p in paths if p.is_file()}
            == set(inventory) | {"manifest.json", "receipt.json"}, "complete study inventory")
    inputs = {}
    for name, expected in inventory.items():
        path = relative_path(study, name)
        require(pin(path) == expected, "study pin mismatch: " + name)
        inputs[str(path)] = expected
    require({"registration.json", "results.json", "resources.json", "fits.json"} <= set(inventory),
            "missing scalar artifacts")
    require(read_json(study / "registration.json") == plan, "saved registration join")
    for name, expected in plan["sources"].items():
        path = relative_path(ROOT, name)
        snapshot = relative_path(study / "sources", name)
        require(pin(path) == expected == pin(snapshot), "current/snapshot source mismatch: " + name)
        inputs[str(path)] = expected
    # The original parent proof remains external; authenticate its recorded bytes.
    require(set(plan["parent_closure"]) == {"manifest", "receipt", "process", "audit", "audit_process"},
            "parent closure roster")
    parent = {}
    for key, item in plan["parent_closure"].items():
        expected = {k: item[k] for k in ("sha256", "bytes")}
        require(pin(item["path"]) == expected, "parent closure pin: " + key)
        inputs[item["path"]] = expected
        parent[key] = read_json(item["path"])
    require(parent["receipt"]["status"] == "PASS" and parent["process"]["returncode"] == 0
            and parent["audit_process"]["returncode"] == 0 and parent["audit"]["status"] == "PASS"
            and parent["audit"]["agreement"] is True, "parent execution/audit closure")
    require(parent["receipt"]["registration_sha256"] == plan["parent_registration_sha256"], "parent registration join")
    for name, item in plan["data"].items():
        expected = {k: item[k] for k in ("sha256", "bytes")}
        require(pin(item["path"]) == expected == parent["manifest"]["files"][name], "parent data pin: " + name)
        inputs[item["path"]] = expected
    qualification = plan["qualification"]
    require(pin(qualification["path"]) == {k: qualification[k] for k in ("sha256", "bytes")}, "qualification pin")
    inputs[qualification["path"]] = pin(qualification["path"])
    for path in (registration, study / "manifest.json", study / "receipt.json"):
        inputs[str(path)] = pin(path)
    return {"study": str(study), "plan": plan, "receipt": receipt, "inputs": inputs}


def finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def extract(results, resources, fits):
    cfg, rows, selection = results["config"], results["rows"], results["selection"]
    require(results["version"] == cfg["version"] == STUDY_VERSION and tuple(cfg["arms"]) == ARMS, "study version/arms")
    require(cfg["context"] == 32 and cfg["dev_horizon"] == 128 and cfg["horizons"] == [64, 128], "forecast scope")
    require(cfg["seeds"] == [8101, 8102, 8103] and cfg["learning_rates"] == [.001, .003]
            and len(set(cfg["partitions"]["dev"])) == 2, "complete design")
    expected = {(f, a, s, r, h) for f in cfg["partitions"]["dev"] for a in ARMS
                for s in cfg["seeds"] for r in cfg["learning_rates"] for h in (64, 128)}
    expected |= {(f, a, None, None, h) for f in cfg["partitions"]["dev"] for a in REFERENCES for h in (64, 128)}
    keyed = {(r["recording"], r["arm"], r["seed"], r["learning_rate"], r["horizon"]): r for r in rows}
    require(len(keyed) == len(rows) == 112 and set(keyed) == expected, "complete unique score roster")
    for row in rows:
        require(row["status"] in ("PASS", "FAILED"), "metric status")
        if row["status"] == "PASS":
            metric = row["metrics"]
            require(len(metric["per_joint_rmse_deg"]) == 6 and all(finite(v) for v in
                    [metric["standardized_rmse"], metric["physical_rmse_deg"], *metric["per_joint_rmse_deg"]]), "finite saved metrics")
        else:
            require(row["metrics"] is None, "failed row carries metric")
    require(len(fits) == 24 and {(r["arm"], r["seed"], r["learning_rate"]) for r in fits}
            == {(a, s, r) for a in ARMS for s in cfg["seeds"] for r in cfg["learning_rates"]}, "all 24 fit attempts")
    require(set(selection["selected_rates"]) == set(ARMS)
            and selection["selected_ridge"] in (*REFERENCES[:2], None), "selection roster")
    gate = results["result"]
    require(gate["total"] == len(gate["conditions"]) == 45
            and len({c["name"] for c in gate["conditions"]}) == 45
            and all(type(c["passed"]) is bool for c in gate["conditions"])
            and gate["passed"] == sum(c["passed"] for c in gate["conditions"]), "saved 45-condition result")
    require(gate["status"] == ("QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL" if gate["passed"] == 45
                               else "DO_NOT_ADVANCE_TRANSITION"), "saved outcome consistency")
    order = [*ARMS, *REFERENCES]
    panels = []
    for recording in cfg["partitions"]["dev"]:
        entries = []
        for arm in order:
            rate = selection["selected_rates"].get(arm)
            require(arm not in ARMS or rate in (*cfg["learning_rates"], None), "selected rate")
            seeds = cfg["seeds"] if arm in ARMS else [None]
            selected = [keyed[(recording, arm, seed, rate, 128)] for seed in seeds] if arm not in ARMS or rate is not None else []
            points = [{"seed": r["seed"], "value": r["metrics"]["standardized_rmse"]} for r in selected if r["status"] == "PASS"]
            complete = len(points) == len(seeds)
            entries.append({"arm": arm, "learning_rate": rate, "points": points,
                            "center": statistics.mean(p["value"] for p in points) if complete else None,
                            "status": "PASS" if complete else "INELIGIBLE" if not selected else "FAILED"})
        panels.append({"recording": recording, "entries": entries})
    resource_keys = {(r["arm"], r["seed"]): r for r in resources}
    expected_resources = {(a, s) for a in ARMS if selection["selected_rates"][a] is not None for s in cfg["seeds"]}
    expected_resources |= {(a, None) for a in REFERENCES}
    require(len(resource_keys) == len(resources) and set(resource_keys) == expected_resources, "resource roster")
    latency = []
    for arm in order:
        points = []
        seeds = cfg["seeds"] if arm in ARMS else [None]
        for seed in seeds:
            row = resource_keys.get((arm, seed))
            if row is not None:
                require(row.get("learning_rate") == selection["selected_rates"].get(arm), "resource selected-rate join")
                value = row["timing"]["median_seconds"]
                require(finite(value) and value > 0, "positive saved request latency")
                points.append({"seed": seed, "value": value * 1000.})
        complete = len(points) == len(seeds)
        latency.append({"arm": arm, "points": points,
                        "center": statistics.median(p["value"] for p in points) if complete else None,
                        "status": "PASS" if complete else "NOT TIMED"})
    return {"version": VERSION, "scope": CAPTION, "config": cfg, "panels": panels, "latency_ms": latency,
            "accuracy_center": "arithmetic mean of individual fit RMSEs", "latency_center": "median of per-fit request medians",
            "selection": selection, "result": gate, "all_rows": rows, "resources": resources, "fits": fits, "order": order}


def figure(values, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    datasets = [p["entries"] for p in values["panels"]] + [values["latency_ms"]]
    scales = []
    seed_offsets = {8101: -.15, 8102: 0., 8103: .15, None: 0.}
    for panel, (ax, entries) in enumerate(zip(axes, datasets, strict=True)):
        numbers = [p["value"] for entry in entries for p in entry["points"]]
        positive = [v for v in numbers if v > 0]
        scale = "linear"
        if positive and max(positive) / min(positive) >= 100:
            scale = "log" if len(positive) == len(numbers) else "symlog"
            ax.set_xscale(scale, **({"linthresh": min(positive) / 10} if scale == "symlog" else {}))
        scales.append(scale)
        for y, entry in enumerate(entries):
            color = "#126e82" if entry["arm"] == "lpv_recurrent" else "#68788b" if entry["arm"] in ARMS else "#b16b29"
            for point in entry["points"]:
                ax.scatter(point["value"], y + seed_offsets[point["seed"]], s=30, alpha=.85, color=color, zorder=3)
            if entry["center"] is not None:
                ax.scatter(entry["center"], y, marker="D", s=64, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
            if entry["status"] != "PASS":
                ax.text(.98, y, entry["status"], transform=ax.get_yaxis_transform(), ha="right", va="center", color="#a33030", fontsize=9)
        ax.set_ylim(len(entries) - .5, -.5)
        ax.set_yticks(range(len(entries)), [LABELS[a] for a in values["order"]])
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(x=.12)
        if panel < 2:
            name = values["panels"][panel]["recording"].removeprefix("recording_2021_12_15_").removesuffix(".mat")
            ax.set_title(f"Exposed DEV {panel + 1}: {name}\nH128 accuracy; lower is better")
            ax.set_xlabel("FIT-standardized RMSE" + (f" ({scale})" if scale != "linear" else ""))
        else:
            ax.set_title("Full request latency; lower is better\n20 repeats per fit after 3 warmups")
            ax.set_xlabel("Median request latency, ms" + (f" ({scale})" if scale != "linear" else ""))
    result = values["result"]
    fig.suptitle(f"Bounded robot transition: {result['passed']}/45 development conditions passed", fontsize=16, y=.97)
    fig.legend([Line2D([], [], marker="o", linestyle="", color="#68788b"),
                Line2D([], [], marker="D", linestyle="", markerfacecolor="white", color="#68788b")],
               ["Individual seed fit / deterministic reference", "Accuracy mean / latency median"],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
    fig.text(.035, .025, CAPTION + "\nLatency includes normalization, casting, conditioning, spectral norms, forecast and denormalization; excludes model loading.", fontsize=9.5, va="bottom")
    fig.subplots_adjust(left=.14, right=.98, bottom=.25, top=.82, wspace=.20)
    fig.savefig(folder / "benchmark.png", dpi=160)
    fig.savefig(folder / "benchmark.pdf")
    plt.close(fig)
    return scales


def fmt(value):
    return "n/a" if value is None else f"{value:.6g}"


def clean(value):
    return str(value).replace("|", "/").replace("\n", " ")


def tables(values):
    lines = ["# Bounded robot transition: complete saved results", "", CAPTION.replace("\n", " "), "",
             f"Saved outcome: **{values['result']['status']}**, {values['result']['passed']}/45 conditions.", "",
             "## Every candidate and reference", "",
             "All 112 rows are retained. H64 is descriptive; two-rate selection and the rule use pooled DEV H128. Both causal ridge penalties are visible.", "",
             "| Recording | Model | Rate | Seed | Horizon | Status | Standardized RMSE | Physical RMSE (deg) | Per-joint RMSE (deg) | Error |",
             "|---|---|---:|---:|---:|---|---:|---:|---|---|"]
    for row in values["all_rows"]:
        metric = row["metrics"] or {}
        name = row["recording"].removeprefix("recording_2021_12_15_").removesuffix(".mat")
        joints = ", ".join(fmt(v) for v in metric.get("per_joint_rmse_deg", []))
        lines.append(f"| {name} | {LABELS[row['arm']]} | {fmt(row['learning_rate'])} | {row['seed']} | {row['horizon']} | {row['status']} | {fmt(metric.get('standardized_rmse'))} | {fmt(metric.get('physical_rmse_deg'))} | {joints} | {clean(row.get('error') or '')} |")
    lines += ["", "## Selection", "", "| Family | Selected rate / ridge | All options |", "|---|---|---|"]
    for arm in ARMS:
        text = "; ".join(f"{r['rate']:g}: {'eligible' if r['eligible'] else 'INELIGIBLE'}, pooled RMSE {fmt(r['pooled_rmse'])}" for r in values["selection"]["options"][arm])
        lines.append(f"| {LABELS[arm]} | {fmt(values['selection']['selected_rates'][arm])} | {text} |")
    lines.append(f"| Causal ridge | {values['selection']['selected_ridge'] or 'INELIGIBLE'} | {clean(values['selection']['ridge_options'])} |")
    lines += ["", "## Every fitting attempt and request cost", "",
              "All 24 attempts are retained. Optimizer-loop duration is not inference latency; absent request timing is not zero.", "",
              "| Model | Rate | Seed | Fit status | Updates | Optimizer loop (s) | Parameters | Inactive | Weight / state / buffer / normalizer (B) | Request median (ms) |",
              "|---|---:|---:|---|---:|---:|---:|---:|---|---:|"]
    timing = {(r["arm"], r["seed"], r.get("learning_rate")): r["timing"] for r in values["resources"]}
    costs = []
    for fit in values["fits"]:
        costs.append({**fit["resources"], "arm": fit["arm"], "seed": fit["seed"], "learning_rate": fit["learning_rate"],
                      "fit": fit["fit"], "timing": timing.get((fit["arm"], fit["seed"], fit["learning_rate"]))})
    costs += [{**r, "fit": None} for r in values["resources"] if r["arm"] in REFERENCES]
    for row in costs:
        fit = row["fit"] or {}
        duration = row.get("timing")
        storage = " / ".join(str(row[k]) for k in ("parameter_bytes", "state_bytes", "buffer_bytes", "normalizer_bytes"))
        lines.append(f"| {LABELS[row['arm']]} | {fmt(row.get('learning_rate'))} | {row['seed']} | {fit.get('status', 'reference')} | {fit.get('completed_updates', 'n/a')} | {fmt(fit.get('optimizer_seconds'))} | {row['parameters']} | {row.get('inactive_parameters', 0)} | {storage} | {fmt(duration['median_seconds'] * 1000 if duration else None)} |")
    lines += ["", "Storage is persistent numeric weights, explicit state, buffers and normalizers. Request input/output arrays are separate; temporary workspace and Python overhead are not measured. Instant LPV stores 192 inactive recurrent weights. These controls are not equal in effective capacity or numeric precision.", "",
              "## All 45 registered conditions", "", "| Condition | Passed |", "|---|---|"]
    lines.extend(f"| {clean(c['name'])} | {c['passed']} |" for c in values["result"]["conditions"])
    lines += ["", "The forced-state bound for LPV applies with finite fixed weights and bounded inputs in exact arithmetic. It is not incremental contraction, a bounded-gradient guarantee, or a robot-safety result. A pass only qualifies a separately designed confirmation protocol; the current recordings remain development data.", ""]
    return "\n".join(lines)


def render(study, output):
    auth = authenticate(study)
    study, output = Path(auth["study"]), Path(output).resolve()
    require(not output.is_relative_to(study), "publication must be outside immutable study")
    script_pin = pin(Path(__file__))
    results = read_json(study / "results.json")
    require(results["config"] == auth["plan"]["config"] and results["result"]["status"] == auth["receipt"]["scientific_result"], "result/plan/receipt join")
    values = extract(results, read_json(study / "resources.json"), read_json(study / "fits.json"))
    output.mkdir(parents=True, exist_ok=False)
    values["axis_scales"] = figure(values, output)
    (output / "table.md").write_text(tables(values))
    metric_keys = ["standardized_rmse", "physical_rmse_deg", "standardized_sse", "scalars", "windows", "per_joint_rmse_deg"]
    row_keys = ["recording", "arm", "seed", "learning_rate", "horizon", "status"]
    with (output / "all-candidates.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*row_keys, *metric_keys, "error"])
        writer.writeheader()
        for row in values["all_rows"]:
            metrics = {k: (row["metrics"] or {}).get(k) for k in metric_keys}
            metrics["per_joint_rmse_deg"] = json.dumps(metrics["per_joint_rmse_deg"])
            writer.writerow({**{k: row[k] for k in row_keys}, **metrics, "error": json.dumps(row.get("error"), sort_keys=True)})
    (output / "plotted-values.json").write_text(json.dumps(values, sort_keys=True, indent=2, allow_nan=False) + "\n")
    require(authenticate(study) == auth and pin(Path(__file__)) == script_pin, "presentation inputs/source changed")
    receipt = {"version": VERSION, "study": str(study), "inputs": auth["inputs"], "script": script_pin,
               "registration_sha256": auth["receipt"]["registration_sha256"],
               "scope": "saved-scalar presentation; invoking delivery admits original process/audit closure; no numerical replay",
               "outputs": {p.name: pin(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.study, args.output)


if __name__ == "__main__":
    main()
