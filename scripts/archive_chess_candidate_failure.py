"""Archive the terminal candidate-v1 failed attempt without modifying its files."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "runs/chess-candidate-v1/execution"
PLAN = ROOT / "evidence/chess-candidate-v1/protocol/plan.json"
OUTPUT = ROOT / "evidence/chess-candidate-v1/failed-attempt"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def capture(directory):
    members = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in execution: {path}")
        if path.is_file():
            members[path.relative_to(directory).as_posix()] = path.read_bytes()
    return members


def publish(execution=EXECUTION, out=OUTPUT):
    execution, out = Path(execution), Path(out)
    members = capture(execution)
    required = {
        "started.json", "failed.json", "training-cache.json", "panels.json", "baselines.json",
        "action_only-97/learning.jsonl", "data/started.json", "data/completed.json",
        "data/analyses.jsonl", "data/dev.jsonl", "data/shift.jsonl",
        "data/excluded-states.json", "data/games.jsonl",
    }
    if set(members) != required:
        raise ValueError("Unexpected files: re-audit the failed attempt before archiving")
    failed = json.loads(members["failed.json"])
    started = json.loads(members["started.json"])
    plan_hash = sha(PLAN.read_bytes())
    if (failed["status"] != "failed" or failed["error"] != "[Errno 32] Broken pipe"
            or failed["plan_sha256"] != plan_hash or started["plan_sha256"] != plan_hash):
        raise ValueError("Failure identity does not match the frozen v1 plan")
    learning = [json.loads(line) for line in members["action_only-97/learning.jsonl"].splitlines()]
    if ([row["step"] for row in learning] != list(range(1, 129))
            or any(row["epoch"] != 1 or row["examples"] != 128 for row in learning)):
        raise ValueError("Partial optimization journal does not match the audited 128 updates")
    keys = ("examples", "microbatches", "candidate_evaluations", "root_core_iterations",
            "candidate_refinement_iterations", "successor_root_iterations", "total_core_iterations",
            "delta_convolutions", "successor_encoders")
    totals = {key: sum(row[key] for row in learning) for key in keys}
    expected = dict(zip(keys, (16384, 128, 484721, 65536, 969442, 0, 1034978, 0, 0)))
    if totals != expected:
        raise ValueError("Partial optimization arithmetic differs from the audited totals")
    data = json.loads(members["data/completed.json"])
    if data["status"] != "completed" or data["counts"] != {"dev": 2048, "shift": 2048}:
        raise ValueError("Fresh data did not complete")
    for name, digest in data["files"].items():
        if sha(members["data/" + name]) != digest:
            raise ValueError(f"Data receipt hash mismatch: {name}")
    analyses = [json.loads(line) for line in members["data/analyses.jsonl"].splitlines()]
    if len(analyses) != data["teacher_calls"] or any(row["status"] != "completed" for row in analyses):
        raise ValueError("Teacher call coverage differs from the receipt")
    for key in ("requested_nodes", "reported_nodes"):
        if sum(row[key] for row in analyses) != data[key]:
            raise ValueError(f"Teacher journal {key} does not reproduce the receipt")
    file_manifest = {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in members.items()}
    summary = {
        "status": "failed",
        "study": "chess-candidate-v1",
        "plan_sha256": plan_hash,
        "failure": failed,
        "partial_fit": {
            "configuration": "action_only-97", "recorded_optimizer_updates": len(learning),
            "planned_updates_per_fit": 1536, "computation": totals,
            "journal_sha256": sha(members["action_only-97/learning.jsonl"]),
            "journal_scope": "Each row is flushed after optimizer.step; no checkpoint was saved.",
            "training_wall_seconds": None,
            "timing_limit": "No completed fit receipt exists; total attempt wall time includes data and cache work.",
        },
        "complete_checkpoints": 0,
        "complete_fits": 0,
        "neural_evaluation_files": 0,
        "latency_files": 0,
        "engine_grading_calls": 0,
        "arena_games": 0,
        "native_baselines_present": True,
        "training_cache": json.loads(members["training-cache.json"]),
        "fresh_data_cost": data,
        "data_generation_wall_seconds": data["completed_unix"] - data["started_unix"],
        "cause_evidence": {
            "observed": "Terminal receipt reports [Errno 32] Broken pipe; journal ends at update 128.",
            "inference": "The first stdout progress print follows the flushed update-128 journal row; failure is consistent with that print.",
            "limit": "No traceback was saved, so the exact failing callsite is not independently established.",
        },
        "accounting": {
            "original_attempt_retained": True,
            "success_claim": False,
            "automatic_retry": False,
            "recovery_requires_separate_amendment": True,
            "extra_updates_to_charge_if_restarted": 128,
            "reuse_data_rule": "A separately declared recovery may bind these exact completed data bytes; do not silently regenerate, replace or relabel them.",
        },
        "archive_scope": "Every regular file present in the terminal execution, byte for byte; no model inference or training was run by this publisher.",
    }
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "execution.tar.gz"
    with (
        archive.open("xb") as stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(raw), 0o644, 0
            tar.addfile(info, io.BytesIO(raw))
    with tarfile.open(archive, "r:gz") as tar:
        infos = tar.getmembers()
        if len(infos) != len(members) or {info.name for info in infos} != set(members):
            raise ValueError("Archive member coverage differs")
        for info in infos:
            if not info.isfile() or tar.extractfile(info).read() != members[info.name]:
                raise ValueError(f"Archive did not roundtrip exact bytes: {info.name}")
    if capture(execution) != members:
        raise ValueError("Execution changed while being archived")
    with (out / "summary.json").open("xb") as stream:
        stream.write(encode(summary))
    readme = """# Candidate chess v1: failed attempt

This attempt failed with `Broken pipe` after **128 recorded optimizer updates** of the first fit, `action_only-97`. There are **zero complete checkpoints and zero neural evaluations**. This is a retained failed attempt, not a chess-performance result.

The journal records 16,384 training presentations and 484,721 candidate evaluations. The first progress print occurs immediately after flushing update 128, which is consistent with the error. The receipt does not contain a traceback, so that callsite is an inference.

Fresh-data generation completed: 4,096 positions, 4,756 Stockfish calls, 9,512,000 requested nodes and 9,497,297 reported nodes. The training cache was built for 32,768 roots and 968,036 legal successors. Native baseline summaries exist; trained-model predictions, latency measurements, stronger-engine grading and games do not.

`execution.tar.gz` preserves all 13 execution files losslessly. `manifest.json` records every original file's SHA-256 and byte size. The archiver verified each decompressed member against the original bytes and checked that the execution remained unchanged. `summary.json` retains the failed attempt's partial work and data costs.

Any recovery needs a separate, explicit protocol amendment. The failed attempt must remain in cumulative accounting, including these 128 extra updates. No checkpoint exists to resume. Reusing the completed evaluation data requires binding the same bytes and disclosing that reuse; it does not justify claiming newly generated data or dropping this failure.
"""
    with (out / "README.md").open("x") as stream:
        stream.write(readme)
    manifest = {
        "format": "chess-candidate-failed-attempt-v1",
        "status": "archived_failed_attempt",
        "plan_sha256": plan_hash,
        "source_sha256": sha(Path(__file__).read_bytes()),
        "execution_files": file_manifest,
        "execution_file_count": len(members),
        "execution_bytes": sum(len(raw) for raw in members.values()),
        "archive": {"path": archive.name, "sha256": sha(archive.read_bytes()), "bytes": archive.stat().st_size},
        "artifacts": {name: {"sha256": sha((out / name).read_bytes()), "bytes": (out / name).stat().st_size}
                      for name in ("summary.json", "README.md")},
        "verified": {"byte_roundtrip": True, "source_unchanged": True, "all_execution_files": True},
    }
    with (out / "manifest.json").open("xb") as stream:
        stream.write(encode(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=EXECUTION)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(publish(args.execution, args.out), indent=2))
