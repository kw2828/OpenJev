"""Package closed learned-retention evidence without numerical replay."""
import hashlib
import importlib.util
import json
import math
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def descriptor(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    study = ROOT / "output/retention-v1"
    delivery = ROOT / "output/retention-delivery-v1"
    out = ROOT / "research/retention-results"
    spec = importlib.util.spec_from_file_location("retention_plot_package", ROOT / "scripts/plot_retention_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pins, audit = module.authenticate(study, delivery / "audit.json")
    closure = json.loads((delivery / "plot-process.json").read_text())
    if closure["state"] != "EXITED" or closure["returncode"] != 0:
        raise ValueError("original plot process did not close")
    if not isinstance(closure["elapsed_seconds"], (int, float)) or not math.isfinite(closure["elapsed_seconds"]) or closure["elapsed_seconds"] <= 0:
        raise ValueError("finite positive original plot elapsed time")
    command = closure["command"].copy()
    expected = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/plot_retention_study.py"),
                "--study", str(study), "--audit", str(delivery / "audit.json"), "--out", str(out)]
    if len(command) != len(expected):
        raise ValueError("plot process argument count")
    for index in (0, 1, 3, 5, 7):
        command[index] = os.path.abspath(ROOT / command[index])
    if command != expected:
        raise ValueError("plot process command and study binding")
    plot = json.loads((out / "plot-receipt.json").read_text())
    expected_plot = {"version": "retention-plot-v1", "metric_rows": 99, "resource_records": 33,
                     "condition_rows": 42, "result_reads_after_original_closure": True,
                     "npz_decodes": 0, "model_calls": 0, "audit_replays": 0}
    if any(type(plot.get(key)) is not type(value) or plot[key] != value for key, value in expected_plot.items()):
        raise ValueError("plot version, counts and read-after-closure contract")
    if plot["inputs"] != pins or plot["study"] != str(study) or plot["audit_path"] != str(delivery / "audit.json"):
        raise ValueError("plot input binding")
    if set(plot["outputs"]) != {"benchmark.png", "benchmark.pdf", "plotted-values.json"}:
        raise ValueError("plot output roster")
    if plot["renderer"] != descriptor(ROOT / "scripts/plot_retention_study.py"):
        raise ValueError("renderer changed")
    for name, expected in plot["outputs"].items():
        if descriptor(out / name) != expected:
            raise ValueError("plot artifact changed")
    for name, original in (("audit.json", delivery / "audit.json"), ("summary.json", study / "summary.json")):
        if (out / name).read_bytes() != original.read_bytes():
            raise ValueError("published evidence differs from original: " + name)
    members = {}
    for prefix, directory in (("study", study), ("delivery", delivery)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                members[f"{prefix}/{path.relative_to(directory)}"] = path
    for name in ("retention-memory-engineering-v1", "retention-data-engineering-v1",
                 "retention-controls-engineering-v1", "retention-runner-engineering-v1",
                 "retention-audit-engineering-v1", "retention-plot-v1"):
        folder = ROOT / "output" / name
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                members[f"engineering/{name}/{path.relative_to(folder)}"] = path
    parent_summary = ROOT / "research/residual-memory-results/summary.json"
    if descriptor(parent_summary) != audit["admission"]["parent_result"]:
        raise ValueError("prior-result lineage pin changed")
    members["parent-lineage/research/residual-memory-results/summary.json"] = parent_summary
    for name in ("benchmark.png", "benchmark.pdf", "plotted-values.json", "plot-receipt.json", "summary.json"):
        members["presentation/" + name] = out / name
    for name in ("scripts/plot_retention_study.py", "scripts/package_retention_study.py",
                 "research/retention-results.md", "research/retention-next.md", "research/retention-protocol.md",
                 "README.md", "docs/decision-model-architecture.md"):
        members["publication/" + name] = ROOT / name
    if any(path.is_symlink() for path in members.values()):
        raise ValueError("ordinary evidence files only")
    manifest = {name: descriptor(path) for name, path in sorted(members.items())}
    archive = out / "evidence.tar.gz"
    with archive.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        for name, path in sorted(members.items()):
            bundle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as bundle:
        if set(bundle.getnames()) != set(manifest) or len(bundle.getnames()) != len(manifest):
            raise ValueError("archive roster")
        for item in bundle.getmembers():
            data = bundle.extractfile(item).read()
            if {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} != manifest[item.name]:
                raise ValueError("archive content differs")
    if module.authenticate(study, delivery / "audit.json")[0] != pins:
        raise ValueError("evidence changed during packaging")
    index = {
        "version": "retention-publication-v1", "gate": audit["result"]["gate"],
        "archive": descriptor(archive), "member_count": len(manifest), "members": manifest,
        "registration": pins["registration"], "audit": pins["audit"],
        "parent_result": {"url": "https://github.com/kw2828/OpenJev/blob/main/research/residual-memory-results.md",
                          **descriptor(parent_summary),
                          "scope": "Prior negative result retained for provenance. No prior weights or data reused."},
        "scope": "Opaque packaging and hash verification only; no generation, training or model evaluation."}
    with (out / "archive-index.json").open("x") as stream:
        json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"archive": index["archive"], "member_count": index["member_count"]}))


if __name__ == "__main__":
    main()
