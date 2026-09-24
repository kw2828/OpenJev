"""Publish only closed, independently audited function-reference evidence.

No prediction archive is decoded and no numerical metric is recomputed. Plot
conversions are limited to bytes-to-KiB and recorded seconds per context. The
audit is copied byte-for-byte. Historical component-launch failure is retained.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "output/function-reuse-reference-v1"
ENGINEERING = ROOT / "output/function-reuse-reference-engineering-v1"
DELIVERY = ROOT / "output/function-reuse-reference-delivery-v1"
OUTPUT = ROOT / "research/function-reuse-reference-results"
VERSION = "function-reuse-reference-publication-v1"
REGISTRATION_SHA = "9c48893f50c87f99241bbb584e39636b94dd14ab3a427f6ff78addce58707d5a"
AUDIT_SHA = "8127f830dafd13837c4905266e1a76f19965a86d30053fedca8845e90b801681"
DELIVERY_PINS = {
    "qualification-process.json": "9b75e02d4dd6f12e70730378530167480224474f34a4c61807bb61f35a8f1b89",
    "run-process.json": "514f649b039570fc38f2ad66641946da6ea3f17ce004b2c842e902e6b0c8f72b",
    "audit-process.json": "5f09fc248a7b7fa3fb6970f83030333ea0409e43f68478b94f1f276d2104bbbe",
    "qualification.log": "66be3913276c650e7bd3701c2bde755b28fa2eb6ec930f5bfb3f4027bc075a97",
    "run.log": "135d05c656e1c3eef1d2c0e8747d39dd294b7146f9919a4d111cb4f316ccf8cb",
    "audit.log": "cc0dbf835cb37a48c9ba8de8d947ac5c0c8bd0032d0e99762fa9cb0362af009c",
}
SOURCES = {
    "scripts/function_reuse_reference_study.py", "scripts/audit_function_reuse_reference.py",
    "src/openjev/research/function_reuse_reference.py", "tests/test_function_reuse_reference.py",
    "tests/test_function_reuse_reference_study.py", "tests/test_function_reuse_reference_audit.py",
    "research/function-reuse-reference-protocol.md",
}
CONFIG = {"namespace": 440260924, "cohorts": 5, "blocks": [1, 3, 8, 16], "cases": 64,
          "dimensions": 8, "basis_examples": 16, "fewshot_examples": 4, "queries": 8, "choices": 6}
MSE_ZERO_FLOOR = 1e-32


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(ROOT), "evidence must be inside repository")
    return path.relative_to(ROOT).as_posix()


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular evidence file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink(), "ordinary evidence directory")
    result = {}
    for path in sorted(folder.rglob("*")):
        require(not path.is_symlink(), "no evidence symlinks")
        if path.is_file():
            result[relative(path)] = descriptor(path)
        else:
            require(path.is_dir(), "ordinary evidence entries")
    return result


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def authenticate():
    """Check closed process, frozen source and opaque payload bytes before metrics."""
    registration_path = STUDY / "registration.json"
    require(descriptor(registration_path)["sha256"] == REGISTRATION_SHA, "exact original registration")
    plan = read(registration_path)
    require(plan["config"] == CONFIG and set(plan["sources"]) == SOURCES
            and plan["seconds_cap"] == 120 and plan["run_attempts"] == 1
            and plan["training_updates"] == 0 and plan["data_selection"] == "none", "fixed original scope")
    inputs = inventory(DELIVERY)
    require(set(inputs) == {relative(DELIVERY / name) for name in DELIVERY_PINS}, "exact original process files")
    for name, digest in DELIVERY_PINS.items():
        require(descriptor(DELIVERY / name)["sha256"] == digest, "original process/log pin: " + name)
    python = str(ROOT / ".venv/bin/python")
    commands = {
        "qualification": [python, "-m", "pytest", "-q", "tests/test_function_reuse_reference.py",
                          "tests/test_function_reuse_reference_study.py", "tests/test_function_reuse_reference_audit.py"],
        "run": [python, "scripts/function_reuse_reference_study.py", "run", "--folder", relative(STUDY)],
        "audit": [python, "scripts/audit_function_reuse_reference.py", "--folder", relative(STUDY)],
    }
    phases = {}
    for phase, command in commands.items():
        record = read(DELIVERY / (phase + "-process.json"))
        require(record["command"] == command and type(record["returncode"]) is int
                and record["returncode"] == 0 and type(record["elapsed_seconds"]) in (float, int)
                and math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] > 0,
                "original successful process closure: " + phase)
        if phase != "qualification":
            require(record["status"] == "EXITED" and record["registration_sha256"] == REGISTRATION_SHA,
                    "original registration and exited status: " + phase)
        phases[phase] = record
    require("69 passed" in (DELIVERY / "qualification.log").read_text(), "original qualification completion")
    for name, pin in plan["sources"].items():
        require(descriptor(ROOT / name) == descriptor(STUDY / "sources" / name) == pin,
                "current and snapshotted original source: " + name)
        inputs[name] = pin
    expected_snapshot = {relative(STUDY / "sources" / name): pin for name, pin in plan["sources"].items()}
    require(inventory(STUDY / "sources") == expected_snapshot, "exact source snapshot")
    run_receipt = read(STUDY / "run/receipt.json")
    require(run_receipt["status"] == "COMPLETE"
            and run_receipt["registration"] == descriptor(registration_path)
            and 0 < run_receipt["elapsed_seconds"] <= 120, "original completed producer receipt")
    run_names = {f"run/cohort-{cohort:02d}-k-{blocks:02d}.npz"
                 for cohort in range(5) for blocks in CONFIG["blocks"]} | {"run/started.json", "run/summary.json"}
    require(set(run_receipt["files"]) == run_names, "all twenty original groups")
    require(set(inventory(STUDY / "run")) == {relative(STUDY / name) for name in run_names | {"run/receipt.json"}},
            "complete original run file inventory")
    for name, pin in run_receipt["files"].items():
        require(descriptor(STUDY / name) == pin, "unchanged opaque producer payload: " + name)
    require(read(STUDY / "run/started.json")["registration"] == descriptor(registration_path), "original start join")
    audit_receipt = read(STUDY / "audit.receipt.json")
    require(audit_receipt["status"] == "COMPLETE" and audit_receipt["agreement"] is True
            and audit_receipt["registration"] == descriptor(registration_path)
            and audit_receipt["producer_receipt"] == descriptor(STUDY / "run/receipt.json")
            and audit_receipt["files"] == {"audit.json": descriptor(STUDY / "audit.json")}
            and descriptor(STUDY / "audit.json")["sha256"] == AUDIT_SHA, "original independent audit binding")
    expected_study = ({relative(STUDY / name) for name in run_names | {
        "registration.json", "audit.json", "audit.receipt.json", "run/receipt.json"}} | set(expected_snapshot))
    require(set(inventory(STUDY)) == expected_study, "complete exact child study")
    require(set(inventory(ENGINEERING)) == {relative(ENGINEERING / name)
            for name in ("component-test.txt", "component-test-02.txt")}, "preserved component attempts")
    inputs.update(inventory(STUDY)); inputs.update(inventory(ENGINEERING))
    inputs[relative(Path(__file__).absolute())] = descriptor(Path(__file__).absolute())
    # No metric-bearing JSON is read until every preceding byte and closure check passes.
    audited = read(STUDY / "audit.json")
    require(audited["agreement"] is True and audited["architecture_claim"] is False
            and audited["external_process_closure_required"] is True and audited["config"] == CONFIG
            and audited["registration"] == descriptor(registration_path)
            and audited["producer_receipt"] == descriptor(STUDY / "run/receipt.json")
            and audited["status"] == audit_receipt["scientific_status"], "audited scope and receipt agreement")
    require([(row["cohort"], row["blocks"]) for row in audited["rows"]]
            == [(cohort, blocks) for cohort in range(5) for blocks in CONFIG["blocks"]]
            and len(audited["conditions"]) == 80, "all twenty audited groups and eighty conditions")
    return {"inputs": inputs, "plan": plan, "audit": audited, "phases": phases}


def plotted_values(audited):
    rows = []
    for row in audited["rows"]:
        plotted = {"cohort": row["cohort"], "blocks": row["blocks"], "cases": row["cases"],
                   "queries": row["queries"], "lookup_mse": row["lookup_mse"], "fewshot_mse": row["fewshot_mse"],
                   "lookup_regret": row["lookup_regret"], "fewshot_regret": row["fewshot_regret"],
                   "cached_maps_and_diagnostics_kib": (row["map_array_bytes_per_context"]
                                                       + row["diagnostic_array_bytes_per_context"]) / 1024,
                   "raw_basis_kib": row["raw_basis_array_bytes_per_context"] / 1024,
                   "lookup_fit_seconds_per_context": row["lookup_fit_seconds"] / row["cases"],
                   "lookup_query_seconds_per_context": row["lookup_query_seconds"] / row["cases"],
                   "lookup_total_seconds_per_context": (row["lookup_fit_seconds"]
                                                        + row["lookup_query_seconds"]) / row["cases"],
                   "fewshot_seconds_per_context": row["fewshot_query_seconds"] / row["cases"]}
        for name, value in plotted.items():
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative plot: " + name)
        rows.append(plotted)
    return {"rows": rows, "mse_zero_plot_floor": MSE_ZERO_FLOOR,
            "scope": "Every cohort is shown. Timing covers one fit plus eight requests for lookup and eight few-shot fits for the control; generation, file I/O and publication are excluded. Storage is numeric arrays, not peak process memory. Zero MSE, if present, is plotted at the disclosed floor without changing saved values."}


def figure(values, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    colors = ("#2166ac", "#d6604d")
    panels = [
        ("Prediction error", "MSE (log scale)", ("lookup_mse", "fewshot_mse"), ("Cached block maps", "Few-shot only")),
        ("Decision regret", "Mean regret", ("lookup_regret", "fewshot_regret"), ("Cached block maps", "Few-shot only")),
        ("Stored arrays per context", "KiB", ("cached_maps_and_diagnostics_kib", "raw_basis_kib"), ("Maps + diagnostics", "Raw basis X/Y")),
        ("Recorded computation per context", "Seconds: fit + 8 requests", ("lookup_total_seconds_per_context", "fewshot_seconds_per_context"), ("Cached block maps", "Few-shot only")),
    ]
    for axis, (title, ylabel, keys, labels) in zip(axes.flat, panels, strict=True):
        for method, (key, label) in enumerate(zip(keys, labels, strict=True)):
            for cohort in range(5):
                rows = [r for r in values["rows"] if r["cohort"] == cohort]
                xs = [CONFIG["blocks"].index(r["blocks"]) + (cohort-2)*.025 + (method-.5)*.16 for r in rows]
                ys = [MSE_ZERO_FLOOR if "_mse" in key and r[key] == 0 else r[key] for r in rows]
                axis.plot(xs, ys, marker="o" if method == 0 else "s", markersize=4,
                          color=colors[method], alpha=.65, linewidth=.7, label=label if cohort == 0 else None)
        axis.set(title=title, ylabel=ylabel, xlabel="Public function blocks (K)",
                 xticks=range(4), xticklabels=CONFIG["blocks"])
        axis.grid(alpha=.2); axis.legend(frameon=False, fontsize=9)
        if title == "Prediction error":
            axis.set_yscale("log")
        elif title == "Decision regret":
            axis.set_ylim(bottom=-0.02)
        else:
            axis.set_ylim(bottom=0)
    fig.suptitle("Classical reference for noiseless function reuse\nFive cohorts at every K; no neural training", fontsize=15)
    fig.supxlabel("Each point is one cohort/K group (64 contexts, 512 requests). Array bytes and host timings are descriptive.\n"
                  "Zero MSE, if present, is shown at 1e-32; saved values are unchanged.", fontsize=9)
    fig.savefig(folder / "benchmark.png", dpi=180)
    fig.savefig(folder / "benchmark.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def archive(folder, entries):
    manifest = {"version": VERSION, "members": entries, "scope": "Complete named study with each original source snapshotted once, original delivery logs and both component attempts, plus these publication artifacts and publisher source. Current scientific sources are authenticated against those identical snapshots, not stored twice. No parent archives or external neural checkpoints."}
    write(folder / "manifest.json", manifest)
    roster = {**entries, relative(folder / "manifest.json"): descriptor(folder / "manifest.json")}
    target = folder / "evidence.tar.gz"
    with (target.open("xb") as raw,
          gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
          tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar):
        for name in sorted(roster):
            payload = (ROOT / name).read_bytes()
            require({"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} == roster[name], "archive input unchanged")
            info = tarfile.TarInfo(name); info.size = len(payload); info.mode = 0o644; info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(target, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == len(roster) and {m.name for m in members} == set(roster), "exact archive roster")
        for member in members:
            require(member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts,
                    "regular safe archive member")
            payload = tar.extractfile(member).read()
            require({"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} == roster[member.name],
                    "opaque archive roundtrip")
    return len(roster)


def publish(output=OUTPUT):
    output = Path(output).absolute()
    require(output.is_relative_to(ROOT) and not output.exists(), "exclusive publication directory inside repository")
    before = authenticate()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("xb") as stream:
        stream.write((STUDY / "audit.json").read_bytes())
    values = plotted_values(before["audit"])
    write(output / "plotted-values.json", values)
    figure(values, output)
    payloads = {relative(output / name): descriptor(output / name)
                for name in ("summary.json", "plotted-values.json", "benchmark.png", "benchmark.pdf")}
    members = archive(output, {**{name: pin for name, pin in before["inputs"].items() if name not in SOURCES},
                               **payloads})
    after = authenticate()
    require(before == after, "all evidence and sources unchanged after publication")
    files = {name: descriptor(output / name) for name in ("summary.json", "plotted-values.json", "benchmark.png",
                                                         "benchmark.pdf", "manifest.json", "evidence.tar.gz")}
    receipt = {"version": VERSION, "status": "COMPLETE", "registration_sha256": REGISTRATION_SHA,
               "inputs": before["inputs"], "files": files, "archive_members": members,
               "original_processes": before["phases"], "plotted_groups": 20,
               "audit_summary_byte_copy": descriptor(output / "summary.json") == descriptor(STUDY / "audit.json"),
               "publication_array_decodes": 0, "publication_model_calls": 0,
               "publication_metric_recomputations": 0,
               "component_attempts": "First launch failed before collection because the resolved base interpreter lacked pytest; second launch passed. Both original logs are included.",
               "scope": "Saved-output presentation only. Audit status is retained verbatim; no architecture, neural speed or general reasoning claim."}
    require(receipt["audit_summary_byte_copy"], "byte-identical audited summary")
    write(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(publish(args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
