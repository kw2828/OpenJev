"""Package closed residual-memory evidence without numerical replay."""
import hashlib
import importlib.util
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def descriptor(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    study = ROOT / "output/residual-memory-v1"
    delivery = ROOT / "output/residual-memory-delivery-v1"
    out = ROOT / "research/residual-memory-results"
    parent = ROOT / "output/query-feature-v1"
    spec = importlib.util.spec_from_file_location("residual_plot_package", ROOT / "scripts/plot_residual_memory_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pins, audit = module.authenticate(study, delivery / "audit.json")
    closure = json.loads((delivery / "plot-process.json").read_text())
    if closure["state"] != "EXITED" or closure["returncode"] != 0:
        raise ValueError("original plot process did not close")
    command = closure["command"].copy()
    expected = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/plot_residual_memory_study.py"),
                "--study", str(study), "--audit", str(delivery / "audit.json"), "--out", str(out)]
    if len(command) != len(expected):
        raise ValueError("plot process argument count")
    for index in (0, 1, 3, 5, 7):
        command[index] = os.path.abspath(ROOT / command[index])
    if command != expected:
        raise ValueError("plot process command and study binding")
    if "61 passed" not in (delivery / "qualification.log").read_text():
        raise ValueError("qualification receipt")
    plot = json.loads((out / "plot-receipt.json").read_text())
    if plot["inputs"] != pins or plot["study"] != str(study) or plot["audit_path"] != str(delivery / "audit.json"):
        raise ValueError("plot input binding")
    if set(plot["outputs"]) != {"benchmark.png", "benchmark.pdf", "plotted-values.json"}:
        raise ValueError("plot output roster")
    if plot["renderer"] != descriptor(ROOT / "scripts/plot_residual_memory_study.py"):
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
    for name in ("residual-memory-component-v1", "residual-memory-engineering-v1",
                 "residual-memory-plot-engineering-v1"):
        folder = ROOT / "output" / name
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                members[f"engineering/{name}/{path.relative_to(folder)}"] = path
    for name in ("registration.json", "manifest.json"):
        if descriptor(parent / name) != pins["parent"][name.removesuffix(".json")]:
            raise ValueError("parent lineage differs")
        members["parent-lineage/" + name] = parent / name
    for name in ("benchmark.png", "benchmark.pdf", "plotted-values.json", "plot-receipt.json", "summary.json"):
        members["presentation/" + name] = out / name
    for name in ("scripts/plot_residual_memory_study.py", "scripts/package_residual_memory_study.py",
                 "research/residual-memory-results.md", "research/residual-memory-next.md",
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
    parent_archive = ROOT / "research/query-feature-results/evidence.tar.gz"
    index = {
        "version": "residual-memory-publication-v1", "gate": audit["result"]["gate"],
        "archive": descriptor(archive), "member_count": len(manifest), "members": manifest,
        "registration": pins["registration"], "audit": pins["audit"],
        "parent_dependency": {
            "url": "https://github.com/kw2828/OpenJev/raw/main/research/query-feature-results/evidence.tar.gz",
            **descriptor(parent_archive),
            "scope": "The full parent bundle is required for the auditor's parent-manifest authentication. "
                     "This archive includes lineage pins and all three unchanged checkpoints, not duplicate parent predictions."},
        "scope": "Opaque packaging and hash verification only; no generation, training or model evaluation."}
    with (out / "archive-index.json").open("x") as stream:
        json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"archive": index["archive"], "member_count": index["member_count"]}))


if __name__ == "__main__":
    main()
