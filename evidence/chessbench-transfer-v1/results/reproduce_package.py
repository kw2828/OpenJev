"""Package a completed ChessBench transfer report without model or engine calls.

Publication requires a new output directory. Raw bytes and every archive member
are checked again after writing. This script is not part of the frozen study.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

VERSION = "chessbench-transfer-publication-v1"
PLAN = "evidence/chessbench-transfer-v1/protocol/plan.json"
PLAN_SHA = "10c280ddd417965b2a24aef85d564b34e0a5b171138d416e880327a2c2304323"
EXECUTION = "runs/chessbench-transfer-v1/execution"
REPORT = "runs/chessbench-transfer-v1/report"
ARMS = ("direct", "action_only", "delta", "full_afterstate")
SEEDS = (97, 109, 127)
ATTRIBUTION = """# ChessBench source data attribution

Source: ChessBench / Amortized Planning with Large-Scale Transformers:
A Case Study on Chess, Anian Ruoss et al., NeurIPS 2024.
Copyright 2024 DeepMind Technologies Limited.
Repository: https://github.com/google-deepmind/searchless_chess
Pinned revision: 90ae0e6b121673fc3079aaeffa047580bb600c0a
License notice: https://github.com/google-deepmind/searchless_chess/blob/90ae0e6b121673fc3079aaeffa047580bb600c0a/README.md#license-and-disclaimer
Data URL: https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag

The source repository identifies portions from https://lichess.org/ as CC0:
https://creativecommons.org/publicdomain/zero/1.0/
The remainder is Creative Commons Attribution 4.0:
https://creativecommons.org/licenses/by/4.0/legalcode
The original bag is redistributed byte-for-byte, with its acquisition receipt.
OpenJev's selected/rejected rows and prediction traces are derived from this
source; they add selection identities and our local models' outputs. The
original source has not been relabeled or changed. These data retain their
upstream terms and are not relicensed under OpenJev's MIT code license.
No endorsement by Google DeepMind, Lichess, or the source authors is implied.
Data and derived evidence are supplied without warranties. Original model
weights and independent OpenJev code retain the included project MIT license.
"""
LIMITS = """# Publication scope

This archive preserves all twelve final fits on the same 4,096 selected public
ChessBench behavioral-cloning positions. No checkpoint, seed, or result was
selected for publication. The frozen selection excludes prior natural/mirrored
root and native-successor exposures. Source-game identities and repetition
history are unavailable. Reported seed variation is descriptive, not an
independent-game confidence interval.

No model calls, engine calls, or new training occur during packaging. The
official report was completed separately and its summary/completion bytes are
retained unchanged. The archive includes original acquisition bytes, all saved
predictions and receipts, exclusions, frozen source snapshots, all twelve
published checkpoint bytes, and a per-file SHA-256 manifest.

Agreement is agreement with one reference move. Legal-menu NLL is not a
calibrated winning probability. No value targets, value MAE, engine regret,
Elo, novelty, or successful continuation gate are established by this panel.
The original candidate-v2 continuation gate remains failed.

Evaluation wall time includes CPU batch construction, model forward calls and
output writing; it is not single-decision latency. Brief concurrent synthetic
code tests were observed, so these execution timings do not support a clean
speed comparison. Model weights were unchanged during the evaluation.
"""


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def regular_tree(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Nonregular input: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def validate(root):
    plan_path, execution, report = root / PLAN, root / EXECUTION, root / REPORT
    plan, completed, summary = read(plan_path), read(report / "completed.json"), read(report / "summary.json")
    if sha(plan_path) != PLAN_SHA or completed["status"] != "completed":
        raise ValueError("Frozen plan or official report status differs")
    if (completed["summary_sha256"] != sha(report / "summary.json")
            or completed["plan_sha256"] != PLAN_SHA or summary["status"] != "completed"
            or summary["plan_sha256"] != PLAN_SHA
            or summary["candidate_gates_changed"] is not False
            or summary["elo_estimate"] is not None or summary["novelty_established"] is not False):
        raise ValueError("Official report binding or scope differs")
    receipt = read(execution / "completed.json")
    members = regular_tree(execution)
    members.pop("completed.json")
    expected = {"started.json", "selected.jsonl", "selection.json", "rejections.jsonl"} | {
        f"{config['name']}.{extension}" for config in plan["configurations"] for extension in ("json", "jsonl")
    }
    if (receipt["status"] != "completed" or receipt["plan_sha256"] != PLAN_SHA
            or receipt["files"] != members or set(members) != expected
            or completed["execution_receipt_sha256"] != sha(execution / "completed.json")
            or summary["execution_receipt_sha256"] != sha(execution / "completed.json")):
        raise ValueError("Execution is incomplete or changed")
    configs = plan["configurations"]
    if len(configs) != 12 or {(c["arm"], c["seed"]) for c in configs} != {
        (arm, seed) for arm in ARMS for seed in SEEDS
    } or set(summary["metrics"]) != {c["name"] for c in configs}:
        raise ValueError("All twelve fixed configurations are required")
    for arm in ARMS:
        for metric in ("agreement", "mean_nll"):
            calculated = math.fsum(summary["metrics"][f"{arm}-{seed}"][metric] for seed in SEEDS) / 3
            if calculated != summary["means"][arm][metric]:
                raise ValueError("Equal-seed aggregate differs")
    if any(metric["positions"] != 4096 for metric in summary["metrics"].values()):
        raise ValueError("Incomplete panel")
    candidate = read(root / plan["candidate_plan"])
    if sha(root / plan["candidate_plan"]) != plan["candidate_plan_sha256"]:
        raise ValueError("Candidate plan changed")
    for name in set(candidate["sources"]) & set(plan["sources"]):
        if candidate["sources"][name] != plan["sources"][name]:
            raise ValueError("Conflicting frozen source identities")
    frozen = {**candidate["sources"], **plan["sources"]}
    for name, digest in frozen.items():
        if sha(root / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for config in configs:
        if sha(root / config["path"]) != config["sha256"]:
            raise ValueError(f"Published checkpoint changed: {config['name']}")
    original = root / "evidence/chess-candidate-v2/results"
    for name, field in (("manifest.json", "publication_manifest_sha256"),
                        ("completed.json", "publication_receipt_sha256")):
        if sha(original / name) != plan[field]:
            raise ValueError("Candidate publication binding changed")
    if (sha(root / plan["input"]) != plan["input_sha256"]
            or (root / plan["input"]).stat().st_size != plan["input_bytes"]
            or sha(root / "runs/chessbench-inputs-v1/download.json") != plan["download_receipt_sha256"]
            or read(root / "runs/chessbench-inputs-v1/download.json") != plan["download_receipt"]
            or sha(root / Path(PLAN).parent / plan["exclusion_file"]) != plan["exclusion_sha256"]):
        raise ValueError("Source or exclusion bytes changed")
    return plan, summary, frozen


def render(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure_path = root / "docs/assets/chessbench-transfer-results.png"
    receipt_path = root / "docs/assets/chessbench-transfer-results.json"
    if figure_path.exists() or receipt_path.exists():
        if not (figure_path.is_file() and receipt_path.is_file()):
            raise ValueError("Existing figure requires its matching provenance receipt")
        receipt = read(receipt_path)
        expected = {
            "status": "rendered", "version": VERSION,
            "figure": figure_path.relative_to(root).as_posix(),
            "figure_sha256": sha(figure_path), "plan_sha256": PLAN_SHA,
            "official_summary": REPORT + "/summary.json",
            "official_summary_sha256": sha(root / REPORT / "summary.json"),
            "official_report_receipt_sha256": sha(root / REPORT / "completed.json"),
            "metric_scope": "All four equal-seed means and all twelve seed results; no intervals or winner selection",
            "means": summary["means"], "per_configuration": summary["metrics"],
            "matplotlib_version": receipt.get("matplotlib_version"),
        }
        if (receipt != expected or not isinstance(receipt.get("matplotlib_version"), str)
                or not receipt["matplotlib_version"]):
            raise ValueError("Existing figure receipt differs from the official report or image")
        previous = root / "evidence/chessbench-transfer-v1/results"
        if (previous / "manifest.json").exists():
            manifest, complete = read(previous / "manifest.json"), read(previous / "completed.json")
            if (complete["status"] != "completed"
                    or complete["manifest_sha256"] != sha(previous / "manifest.json")
                    or manifest["summary_sha256"] != expected["official_summary_sha256"]
                    or manifest["figure"] != {figure_path.name: sha(figure_path),
                                             receipt_path.name: sha(receipt_path)}):
                raise ValueError("Existing figure differs from the prior publication bindings")
        return figure_path, receipt_path
    labels = ("Direct", "Action only", "Native delta", "Full afterstate")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    marker_styles = ("o", "s", "^")
    offsets = (-0.13, 0, 0.13)
    for axis, metric, scale, xlabel in zip(
        axes, ("agreement", "mean_nll"), (100, 1),
        ("Reference-move agreement (%)  |  higher is better", "Legal-menu NLL (nats)  |  lower is better"),
        strict=True,
    ):
        values = []
        for index, arm in enumerate(ARMS):
            for seed, marker, offset in zip(SEEDS, marker_styles, offsets, strict=True):
                value = summary["metrics"][f"{arm}-{seed}"][metric] * scale
                values.append(value)
                axis.scatter(value, index + offset, marker=marker, s=42, color="#3787a5", alpha=0.85)
            mean = summary["means"][arm][metric] * scale
            axis.scatter(mean, index, marker="|", s=330, linewidths=2.8, color="#172a3a", zorder=4)
        span = max(values) - min(values)
        axis.set_xlim(min(values) - 0.22 * span, max(values) + 0.30 * span)
        axis.set_xlabel(xlabel, labelpad=12, fontsize=10)
        axis.set_yticks(range(4), labels)
        axis.grid(axis="x", color="#dde4e8", linewidth=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    figure.suptitle("OpenJev on ChessBench: all twelve frozen models", x=0.08, ha="left",
                   y=0.97, fontsize=16, fontweight="bold")
    figure.text(0.08, 0.90, "Same 4,096 public selected positions; no further training or checkpoint selection.",
                ha="left", fontsize=11, color="#405260")
    legend = [Line2D([], [], marker=marker, linestyle="", color="#3787a5", label=f"Seed {seed}")
              for seed, marker in zip(SEEDS, marker_styles, strict=True)]
    legend.append(Line2D([], [], marker="|", linestyle="", color="#172a3a", markersize=13,
                         markeredgewidth=2.8, label="Equal-seed mean"))
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.14), ncol=4, frameon=False)
    figure.text(0.08, 0.075,
                "Source-game IDs unavailable; seed variation is descriptive, with no independent-game interval.\n"
                "Agreement is not Elo. The original candidate continuation gate remains failed.",
                fontsize=9, color="#405260", va="top")
    figure.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.33, wspace=0.27)
    with figure_path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=180, facecolor="white",
                       metadata={"Software": "OpenJev transfer publication, matplotlib " + matplotlib.__version__})
    plt.close(figure)
    write(receipt_path, {
        "status": "rendered", "version": VERSION, "figure": figure_path.relative_to(root).as_posix(),
        "figure_sha256": sha(figure_path), "plan_sha256": PLAN_SHA,
        "official_summary": REPORT + "/summary.json",
        "official_summary_sha256": sha(root / REPORT / "summary.json"),
        "official_report_receipt_sha256": sha(root / REPORT / "completed.json"),
        "metric_scope": "All four equal-seed means and all twelve seed results; no intervals or winner selection",
        "means": summary["means"], "per_configuration": summary["metrics"],
        "matplotlib_version": matplotlib.__version__,
    })
    return figure_path, receipt_path


def audit(out):
    out = Path(out)
    manifest, complete = read(out / "manifest.json"), read(out / "completed.json")
    if complete["status"] != "completed" or complete["manifest_sha256"] != sha(out / "manifest.json"):
        raise ValueError("Publication receipt mismatch")
    actual_files = regular_tree(out)
    actual_files.pop("completed.json")
    if actual_files != complete["files"]:
        raise ValueError("Publication file membership or bytes changed")
    archive = out / manifest["archive"]["path"]
    if sha(archive) != manifest["archive"]["sha256"] or archive.stat().st_size != manifest["archive"]["bytes"]:
        raise ValueError("Archive bytes changed")
    observed = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or member.name in observed or member.name not in manifest["members"]):
                raise ValueError("Unexpected or unsafe archive member")
            digest = hashlib.file_digest(stream.extractfile(member), "sha256").hexdigest()
            observed[member.name] = {"sha256": digest, "bytes": member.size}
    if observed != manifest["members"]:
        raise ValueError("Archive membership, sizes or content differ")
    for copy_name, member_name in manifest["byte_identical_copies"].items():
        if sha(out / copy_name) != observed[member_name]["sha256"]:
            raise ValueError("Public copy differs from archived original")
    return {"status": "verified", "members": len(observed), "archive_sha256": sha(archive)}


def publish(root, out, source):
    root, out = Path(root).resolve(), Path(out).resolve()
    plan, summary, frozen = validate(root)
    out.mkdir(parents=True, exist_ok=False)
    try:
        with (out / "reproduce_package.py").open("x") as stream:
            stream.write(source)
        with (out / "source-attribution.md").open("x") as stream:
            stream.write(ATTRIBUTION)
        with (out / "scope.md").open("x") as stream:
            stream.write(LIMITS)
        figure_paths = render(root, summary)
        paths = {}
        for directory in (Path(PLAN).parent, Path(EXECUTION), Path(REPORT), Path("models/chess-candidate-v2")):
            for name in regular_tree(root / directory):
                relative = (directory / name).as_posix()
                paths[relative] = root / relative
        additional = set(frozen) | {
            plan["candidate_plan"], plan["input"], "runs/chessbench-inputs-v1/download.json",
            "evidence/chess-candidate-v2/results/manifest.json",
            "evidence/chess-candidate-v2/results/completed.json", "uv.lock", "pyproject.toml", "LICENSE",
        }
        for name in additional:
            paths[name] = root / name
        for path in figure_paths:
            paths[path.relative_to(root).as_posix()] = path
        for name in ("reproduce_package.py", "source-attribution.md", "scope.md"):
            paths["publication/" + name] = out / name
        members = {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                   for name, path in sorted(paths.items())}
        archive = out / "chessbench-transfer-v1.tar.gz"
        with (
            archive.open("xb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as stream,
        ):
            for name, path in sorted(paths.items()):
                data = path.read_bytes()
                if {"sha256": sha_bytes(data), "bytes": len(data)} != members[name]:
                    raise ValueError("Input changed during archive writing")
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                stream.addfile(info, io.BytesIO(data))
        copies = {
            "plan.json": PLAN, "summary.json": REPORT + "/summary.json",
            "report-completed.json": REPORT + "/completed.json",
            "execution-completed.json": EXECUTION + "/completed.json",
            "download.json": "runs/chessbench-inputs-v1/download.json",
        }
        for name, member in copies.items():
            with (out / name).open("xb") as stream:
                stream.write(paths[member].read_bytes())
        manifest = {
            "version": VERSION, "archive": {"path": archive.name, "sha256": sha(archive),
                                           "bytes": archive.stat().st_size},
            "members": members, "byte_identical_copies": copies,
            "plan_sha256": PLAN_SHA, "summary_sha256": sha(root / REPORT / "summary.json"),
            "official_report_receipt_sha256": sha(root / REPORT / "completed.json"),
            "execution_receipt_sha256": sha(root / EXECUTION / "completed.json"),
            "weights": {c["name"]: {"path": c["path"], "sha256": c["sha256"]} for c in plan["configurations"]},
            "frozen_sources": frozen,
            "source_data": {"path": plan["input"], "sha256": plan["input_sha256"],
                            "download_receipt_sha256": plan["download_receipt_sha256"],
                            "attribution": "source-attribution.md"},
            "figure": {path.name: sha(path) for path in figure_paths},
            "verification": "Every archive member and public copy checked after writing; no inference or engine calls",
        }
        write(out / "manifest.json", manifest)
        # Recheck source bytes after archive construction before sealing publication.
        if any(sha(paths[name]) != item["sha256"] for name, item in members.items()):
            raise ValueError("Source bytes changed before completion")
        write(out / "completed.json", {
            "status": "completed", "version": VERSION,
            "manifest_sha256": sha(out / "manifest.json"), "plan_sha256": PLAN_SHA,
            "official_summary_sha256": sha(out / "summary.json"),
            "files": regular_tree(out),
            "candidate_gates_changed": False, "model_or_engine_calls": 0,
            "scope": "Packaging and rendering only; official report retained without rerunning it",
        })
        return audit(out)
    except Exception as error:
        write(out / "failed.json", {"status": "failed", "error": str(error), "version": VERSION})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("publish")
    create.add_argument("--repository", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.directory)
    else:
        source = globals().get("_BOOTSTRAP_SOURCE") or Path(__file__).read_text()
        result = publish(args.repository, args.out, source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
