"""Package the closed pilot and its original evidence without numerical replay."""
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def descriptor(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    study = ROOT / "output/query-feature-v1"
    delivery = ROOT / "output/query-feature-delivery-v1"
    out = ROOT / "research/query-feature-results"
    spec = importlib.util.spec_from_file_location("query_feature_plot_package", ROOT / "scripts/plot_query_feature_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pins, audit = module.authenticate(study, delivery / "audit.json")
    for name in ("qualification-process-02.json", "run-process.json", "audit-process.json", "plot-process.json"):
        record = json.loads((delivery / name).read_text())
        if record["state"] != "EXITED" or record["exit_code"] != 0:
            raise ValueError("original process did not close successfully: " + name)
    if "107 passed" not in (delivery / "qualification-02.log").read_text():
        raise ValueError("final qualification receipt")
    plot = json.loads((out / "plot-receipt.json").read_text())
    for name, expected in plot["outputs"].items():
        if descriptor(out / name) != expected:
            raise ValueError("plot artifact changed")
    if (out / "audit.json").read_bytes() != (delivery / "audit.json").read_bytes():
        raise ValueError("published audit is not the original")
    members = {}
    for prefix, directory in (("study", study), ("delivery", delivery)):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                members[f"{prefix}/{path.relative_to(directory)}"] = path
    for name in ("query-feature-component-v1", "query-feature-data-engineering-v1", "replay-evidence-engineering-v1"):
        for path in sorted((ROOT / "output" / name).rglob("*")):
            if path.is_file():
                members[f"engineering/{name}/{path.relative_to(ROOT / 'output' / name)}"] = path
    for name in ("benchmark.png", "benchmark.pdf", "plotted-values.json", "plot-receipt.json", "summary.json"):
        members["presentation/" + name] = out / name
    for name in ("scripts/plot_query_feature_study.py", "scripts/package_query_feature_study.py",
                 "research/query-feature-results.md", "README.md", "docs/decision-model-architecture.md"):
        members["publication/" + name] = ROOT / name
    for path in members.values():
        if path.is_symlink():
            raise ValueError("ordinary files only")
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
                raise ValueError("archive content changed")
    if module.authenticate(study, delivery / "audit.json")[0] != pins:
        raise ValueError("source evidence changed during packaging")
    index = {"version": "query-feature-publication-v1", "gate": audit["result"]["gate"],
             "archive": descriptor(archive), "member_count": len(manifest), "members": manifest,
             "registration": pins["registration"], "audit": pins["audit"],
             "scope": "Opaque packaging and hash verification only. No data generation, training or model evaluation."}
    with (out / "archive-index.json").open("x") as stream:
        json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"archive": index["archive"], "member_count": index["member_count"]}))


if __name__ == "__main__":
    main()
