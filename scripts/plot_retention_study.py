"""Plot saved audited retention scalars after original process closure.

This presentation helper never loads NPZ/checkpoint files, imports scientific
implementations, regenerates data, or replays an audit. Hashing is opaque.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "retention-plot-v1"
PHASES = ("base", "shift", "long")
SEEDS = (11, 23, 37)
CONTROLS = ("kl8", "kl9", "fifo9", "fic9", "coverage41", "recent41")
METHODS = tuple(f"learned-{seed}" for seed in SEEDS)+CONTROLS+("full", "diagonal")
TESTS = ("tests/test_retention_memory.py", "tests/test_retention_data.py",
         "tests/test_retention_controls.py", "tests/test_retention_policy.py",
         "tests/test_retention_study.py", "tests/test_audit_retention_study.py")
SOURCES = {"src/openjev/research/retention_memory.py", "src/openjev/research/retention_data.py",
           "src/openjev/research/retention_controls.py", "src/openjev/research/retention_policy.py",
           "scripts/retention_study.py", "scripts/audit_retention_study.py",
           "research/retention-protocol.md", *TESTS}
LABELS = ("Learned 11", "Learned 23", "Learned 37", "KL8", "KL9", "FIFO9",
          "FIC9", "Coverage41", "Recent41", "Full GP", "Diagonal diagnostic")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), "ordinary evidence file: "+str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _constant(value):
    raise ValueError("nonfinite JSON constant: "+value)


def read(path):
    descriptor(path)
    return json.loads(path.read_text(), parse_constant=_constant)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def compare(actual, expected):
    if isinstance(expected, dict):
        require(type(actual) is dict and actual.keys() == expected.keys(), "matching scalar-object schema")
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), "matching scalar-list schema")
        for left, right in zip(actual, expected, strict=True):
            compare(left, right)
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8), "audited scalar agreement")
    else:
        require(type(actual) is type(expected) and actual == expected, "exact scalar identity")


def expected_files():
    names = {"registration.json", "started.json", "run-receipt.json", "metrics.json", "resources.json",
             "diagnostics.json", "summary.json", "train-data.npz", "train-reference.npz"}
    for seed in SEEDS:
        names.update({f"policy-{seed}-initial.npz", f"policy-{seed}-final.npz", f"optimizer-{seed}-final.pt",
                      f"training-{seed}.npz", f"training-{seed}.json"})
    for phase in PHASES:
        for cohort in range(3):
            stem = f"{phase}-{cohort}"
            names.add(stem+"-data.npz")
            for method in METHODS:
                names.add(f"{stem}-{method}-prediction.npz")
                if method not in ("full", "diagonal"):
                    names.add(f"{stem}-{method}-state.npz")
    return names | {"source/"+name for name in SOURCES}


def condition_names():
    names = set()
    for phase in PHASES:
        names.update(f"{phase}_regret_vs_{control}" for control in CONTROLS)
        names.update((f"{phase}_nll", f"{phase}_beats_defer"))
        names.update(f"{phase}_cohort_{cohort}" for cohort in range(3))
        names.update(f"{phase}_learned-{seed}" for seed in SEEDS)
    return names


def authenticate(study, audit_path):
    require(study == ROOT/"output/retention-v1" and audit_path == ROOT/"output/retention-delivery-v1/audit.json",
            "fixed original study and delivery paths")
    plan_path = study/"registration.json"
    plan_pin = descriptor(plan_path)
    plan = read(plan_path)
    require(set(plan["sources"]) == SOURCES and plan["config"]["version"] == "retention-v1",
            "registered thirteen-source study")
    for name, pin in plan["sources"].items():
        require(descriptor(ROOT/name) == pin == descriptor(study/"source"/name), "unchanged source and snapshot")
    parent = ROOT/"research/residual-memory-results/summary.json"
    require(descriptor(parent) == plan["parent_result"], "closed parent-result identity")
    commands = {
        "qualification": [".venv/bin/python", "-m", "pytest", "-q", *TESTS],
        "run": [".venv/bin/python", "scripts/retention_study.py", "run", "--out", "output/retention-v1",
                "--registration-sha256", plan_pin["sha256"]],
        "audit": [".venv/bin/python", "scripts/audit_retention_study.py", "--folder", "output/retention-v1",
                  "--output", "output/retention-delivery-v1/audit.json"],
    }
    processes, process_pins = {}, {}
    for phase, command in commands.items():
        path = audit_path.parent/(phase+"-process.json")
        record = read(path)
        require(record["state"] == "EXITED" and type(record["returncode"]) is int
                and record["returncode"] == 0 and record["command"] == command
                and type(record["elapsed_seconds"]) in (int, float)
                and math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] > 0,
                "original successful process and exact command: "+phase)
        if phase == "run":
            require(record["elapsed_seconds"] <= 1800, "registered original run cap")
        processes[phase], process_pins[phase] = record, descriptor(path)
    manifest = read(study/"manifest.json")
    require(set(manifest) == expected_files(), "complete producer evidence roster")
    actual = {path.relative_to(study).as_posix(): descriptor(path) for path in study.rglob("*")
              if path.is_file() and path != study/"manifest.json"}
    require(actual == manifest, "opaque full inventory matches original manifest")
    receipt = read(study/"run-receipt.json")
    require(receipt["state"] == "EXITED" and receipt["exit_code"] == 0
            and receipt["training_updates"] == 1536
            and 0 < receipt["elapsed_seconds"] <= 1800, "closed inner producer receipt")
    admission = {"registration": plan_pin, "manifest": descriptor(study/"manifest.json"),
                 "receipt": descriptor(study/"run-receipt.json"), "sources": plan["sources"],
                 "parent_result": plan["parent_result"], "producer_seconds": receipt["elapsed_seconds"],
                 "manifest_members": len(manifest)}
    # Scientific scalar reads begin only after all original closures and pins.
    audit = read(audit_path)
    require(audit["version"] == "retention-audit-v1" and audit["agreement"] is True
            and audit["admission"] == admission, "independent audit bound to this producer")
    compare(read(study/"summary.json"), audit["result"])
    compare(read(study/"metrics.json"), audit["rows"])
    compare(read(study/"resources.json"), audit["resources"])
    expected_rows = {(phase, cohort, method) for phase in PHASES for cohort in range(3) for method in METHODS}
    require(len(audit["rows"]) == 99 and {(r["phase"], r["cohort"], r["method"]) for r in audit["rows"]}
            == expected_rows, "all99 saved score groups")
    require(len(audit["resources"]) == 33 and {(r["phase"], r["method"]) for r in audit["resources"]}
            == {(phase, method) for phase in PHASES for method in METHODS}, "all33 resource rows")
    checks = audit["result"]["checks"]
    require(len(checks) == 42 and {c["name"] for c in checks} == condition_names()
            and all(type(c["passed"]) is bool for c in checks), "all42 registered conditions")
    gate = "ADVANCE_RETENTION" if all(c["passed"] for c in checks) else "DO_NOT_ADVANCE_RETENTION"
    require(audit["result"]["gate"] == gate == receipt["gate"], "unchanged registered outcome")
    for key, value in {"npz_decodes": 200, "array_loads": 843, "prediction_groups": 99,
                       "metric_conditions": 42, "state_archives": 81, "optimizer_decodes": 0,
                       "model_calls": 0, "generator_calls": 0, "training_updates_replayed": 0}.items():
        require(audit["counts"][key] == value, "complete independent audit scope: "+key)
    for phase in PHASES:
        require(set(audit["result"]["means"][phase]) == {*METHODS, "learned_mean"}, "all method means")
    inputs = {"manifest": admission["manifest"], "audit": descriptor(audit_path), "registration": plan_pin,
              "producer_receipt": admission["receipt"], "sources": plan["sources"],
              "parent_result": plan["parent_result"], "manifest_members": manifest,
              "original_processes": {phase: {"descriptor": process_pins[phase], "record": processes[phase]}
                                     for phase in commands}}
    return inputs, audit


def figure(out, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resources = {(row["phase"], row["method"]): row for row in audit["resources"]}
    panels = (
        ("Conditional decision regret", "regret", "Lower is better; mean of three cohorts"),
        ("Latent-exposure NLL", "nll", "Nats/path; mean of three cohorts"),
        ("Retained logical state", "resident_bytes", "Bytes/context, including policy weights"),
        ("Complete stream + four requests", "milliseconds", "Median ms/context; first context only"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.8), layout="constrained")
    plotted = {}
    for ax, (title, key, subtitle) in zip(axes.flat, panels, strict=True):
        values = []
        for method in METHODS:
            row = []
            for phase in PHASES:
                if key in ("regret", "nll"):
                    value = audit["result"]["means"][phase][method][key]
                else:
                    record = resources[phase, method]
                    value = record["resident_bytes"] if key == "resident_bytes" else 1000*record["median_seconds"]
                require(type(value) in (int, float) and math.isfinite(value), "finite plot scalar")
                row.append(value)
            values.append(row)
        plotted[key] = values
        if key == "resident_bytes":
            colors = ("#2471a3", "#d68910", "#148f77")
            for j, phase in enumerate(PHASES):
                positions = [i+(j-1)*.23 for i in range(len(METHODS))]
                ax.barh(positions, [row[j] for row in values], height=.21, color=colors[j], label=phase.upper())
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    if len(set(row)) == 1 and j != 1:
                        continue
                    ax.annotate(f"{int(value):,}", (value, i+(j-1)*.23), xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=8)
            ax.axvline(1024, color="#a93226", linestyle="--", linewidth=1.3, label="1,024-byte cap")
            ax.set_xlim(0, max(max(row) for row in values)*1.17)
            ax.invert_yaxis()
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        else:
            image = ax.imshow(values, cmap="YlOrBr", aspect="auto")
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    ax.text(j, i, f"{value:.4g}", ha="center", va="center", fontsize=9,
                            color="white" if image.norm(value) > .67 else "#17212b")
            ax.set_xticks(range(3), [phase.upper() for phase in PHASES], fontsize=10)
        ax.set_yticks(range(len(METHODS)), LABELS, fontsize=9)
        ax.set_title(title+"\n"+subtitle, fontsize=12, loc="left")
        ax.axhline(2.5, color="#566573", linewidth=.8)
        ax.axhline(8.5, color="#566573", linewidth=.8)
    passed = sum(item["passed"] for item in audit["result"]["checks"])
    fig.suptitle(f"Bounded GP retention: {audit['result']['gate']} ({passed}/42 conditions)\n"
                 "All three learned fits and every control; colors are linear within each panel",
                 fontsize=16)
    fig.supxlabel("Score panels average cohorts separately for each fitted model; no ensemble. All 99 score rows are retained.\n"
                  "Full GP and diagonal are larger references. Python overhead/native workspace excluded; timings are descriptive.",
                  fontsize=10)
    fig.savefig(out/"benchmark.png", dpi=180)
    fig.savefig(out/"benchmark.pdf")
    plt.close(fig)
    return plotted


def render(study, out, audit_path):
    study, audit_path, out = Path(study).resolve(), Path(audit_path).resolve(), Path(out).resolve()
    inputs, audit = authenticate(study, audit_path)
    require(out == ROOT/"research/retention-results" and not out.exists(), "exclusive fixed publication directory")
    out.mkdir(parents=True, exist_ok=False)
    script_pin = descriptor(Path(__file__).resolve())
    plotted = figure(out, audit)
    write(out/"plotted-values.json", {"version": VERSION, "methods": list(METHODS), "phases": list(PHASES),
          "panels": plotted, "result": audit["result"], "rows": audit["rows"], "resources": audit["resources"],
          "audit_counts": audit["counts"], "process_seconds": {
              key: value["record"]["elapsed_seconds"] for key, value in inputs["original_processes"].items()},
          "scope": "Saved audited scalars only. No NPZ decoding, model calls, generation or audit replay."})
    require(authenticate(study, audit_path)[0] == inputs, "inputs and source pins unchanged after plotting")
    require(descriptor(Path(__file__).resolve()) == script_pin, "plotter unchanged")
    outputs = {name: descriptor(out/name) for name in ("benchmark.png", "benchmark.pdf", "plotted-values.json")}
    receipt = {"version": VERSION, "study": str(study), "audit_path": str(audit_path),
          "inputs": inputs, "renderer": script_pin, "outputs": outputs,
          "metric_rows": 99, "resource_records": 33, "condition_rows": 42,
          "result_reads_after_original_closure": True,
          "npz_decodes": 0, "model_calls": 0, "audit_replays": 0,
          "visual_review_required": True,
          "chronology_scope": "Process records provide successful completion and exact commands; no cross-phase timestamps are recorded."}
    write(out/"plot-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.study, args.out, args.audit)
    print(json.dumps({"outputs": result["outputs"], "metric_rows": result["metric_rows"]}))


if __name__ == "__main__":
    main()
