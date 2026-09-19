"""Saved-output-only packaging. Never imports or calls a learned model."""

import datetime
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/reacher-cache-capacity-v1"
ATTEMPT = BASE / "attempt"
EVIDENCE = ROOT / "evidence/reacher-cache-capacity-v1"
PUBLICATION = BASE / "publication-v1"
ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "packet_mlp", "cached_mlp")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

    def check(item):
        if isinstance(item, float):
            assert math.isfinite(item), path
        elif isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)

    check(value)
    return value


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


started, completed = read(ATTEMPT / "started.json"), read(ATTEMPT / "completed.json")
assert {p.name for p in ATTEMPT.iterdir()} == {"started.json", "completed.json"} | {
    f"{arm}.{suffix}" for arm in ARMS for suffix in ("pt", "json")
}
assert started["data_seed"] == 410
assert completed["status"] == "completed" and completed["scope"] == "engineering_capacity_only"
assert [row["kind"] for row in completed["rows"]] == list(ARMS)
assert 0 < completed["wall_seconds"] <= started["cap_seconds"] == 120
assert sha(BASE / "measure.py") == started["script_sha256"]
assert all(sha(ROOT / name) == digest for name, digest in started["source_sha256"].items())
original_hashes = {
    str(p.relative_to(BASE)): sha(p)
    for p in sorted(BASE.rglob("*"))
    if p.is_file() and p.name != Path(__file__).name
}
assert set(original_hashes) == {
    f"attempt/{name}"
    for name in [
        "started.json",
        "completed.json",
        *(f"{arm}.{suffix}" for arm in ARMS for suffix in ("json", "pt")),
    ]
} | {"measure.py"}
checkpoints = {}
shared_initials = {}
for row in completed["rows"]:
    arm = row["kind"]
    assert read(ATTEMPT / f"{arm}.json") == row
    assert row["updates"] == 4 and len(row["all_update_wall_seconds"]) == 4
    assert len(row["three_full_batch_cem_seconds"]) == 3
    assert all(
        math.isfinite(v) and v > 0
        for v in row["all_update_wall_seconds"] + row["three_full_batch_cem_seconds"]
    )
    assert (
        sum(row["all_update_wall_seconds"])
        <= row["whole_trainer_training_seconds"]
        <= row["whole_measurement_seconds"]
    )
    close(row["projected_3fit_training_seconds"], row["whole_trainer_training_seconds"] / 4 * 1152 * 3)
    close(
        row["projected_3fit_3panel_search_seconds"], sum(row["three_full_batch_cem_seconds"]) / 3 * 50 * 3 * 3
    )
    # Safe tensor loading only. No model constructor, forward, optimization or replay.
    payload = torch.load(ATTEMPT / f"{arm}.pt", weights_only=True, map_location="cpu")
    assert payload["kind"] == arm and payload["failed"] is False
    assert payload["successful_updates"] == payload["optimizer_steps"] == 4
    assert payload["cursor"] == {"epoch": 0, "batch": 4}
    assert payload["source_sha256"] == started["source_sha256"]
    assert payload["data_sha256"] == started["data_sha256"]
    assert payload["runtime"] == {"purpose": "synthetic-capacity", "threads": 2}
    assert sum(t.numel() for t in payload["student_state"].values()) == row["parameters"]
    assert all(torch.isfinite(t).all().item() for t in payload["student_state"].values())
    for key in ("hidden_size", "mlp_width", "train_episodes", "steps", "epochs", "batch_size"):
        assert payload["settings"][key] == started["plan"][key]
    assert payload["model_configuration"]["model_class"] == started["plan"]["model_classes"][arm]
    assert payload["settings"]["device"] == "cpu" and payload["settings"]["dtype"] == "torch.float32"
    assert payload["training_wall_seconds"] == row["whole_trainer_training_seconds"]
    assert set(payload["initialization"]["states"]) == {"gru", "mlp"}
    for family, initial in payload["initialization"]["states"].items():
        assert family in ("gru", "mlp")
        if family not in shared_initials:
            shared_initials[family] = initial
        else:
            other = shared_initials[family]
            assert list(initial) == list(other) and all(torch.equal(initial[k], other[k]) for k in initial)
    checkpoints[arm] = {
        "sha256": sha(ATTEMPT / f"{arm}.pt"),
        "successful_updates": 4,
        "cursor": payload["cursor"],
        "parameters": row["parameters"],
        "model_class": payload["model_configuration"]["model_class"],
        "data_sha256": payload["data_sha256"],
        "runtime_recorded_at_execution": payload["runtime"],
    }
close(
    completed["projected_training_seconds"],
    sum(r["projected_3fit_training_seconds"] for r in completed["rows"]),
)
close(
    completed["projected_search_seconds"],
    sum(r["projected_3fit_3panel_search_seconds"] for r in completed["rows"]),
)
EVIDENCE.mkdir(exist_ok=False)
PUBLICATION.mkdir(exist_ok=False)
snapshot = EVIDENCE / "source-snapshot"
snapshot.mkdir()
for name, digest in started["source_sha256"].items():
    dest = snapshot / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, dest)
    assert sha(dest) == digest
shutil.copyfile(BASE / "measure.py", snapshot / "measure.py")
shutil.copyfile(ATTEMPT / "completed.json", EVIDENCE / "raw-timings.json")
(EVIDENCE / "raw-receipts").mkdir()
for arm in ARMS:
    shutil.copyfile(ATTEMPT / f"{arm}.json", EVIDENCE / "raw-receipts" / f"{arm}.json")
# This is intentionally a later context observation, never relabeled as run metadata.
context = {
    "collected_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": "post-run context only; not recorded at execution and not proof of execution-time hardware/runtime",
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "logical_cpu_count": os.cpu_count(),
    "hardware": {
        k: subprocess.check_output(["sysctl", "-n", k], text=True).strip()
        for k in ("machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu")
    },
    "packages": {k: importlib.metadata.version(k) for k in ("torch", "numpy", "gymnasium", "mujoco")},
}
write(EVIDENCE / "post-run-context.json", context)
member_sources = {f"raw/{p.name}": p for p in sorted(ATTEMPT.iterdir())}
member_sources.update(
    {
        f"source-snapshot/{p.relative_to(snapshot).as_posix()}": p
        for p in sorted(snapshot.rglob("*"))
        if p.is_file()
    }
)
member_sources["packaging/package_evidence.py"] = Path(__file__)
manifest = {
    "schema": "reacher-cache-capacity-bundle-v1",
    "scope": "completed synthetic engineering capacity only",
    "members": {
        name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for name, path in sorted(member_sources.items())
    },
}
write(EVIDENCE / "bundle-manifest.json", manifest)
archive = PUBLICATION / "reacher-cache-capacity-v1.tar.gz"
with (
    archive.open("xb") as raw,
    gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as gz,
    tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
):
    for name, path in sorted(member_sources.items()):
        content = path.read_bytes()
        info = tarfile.TarInfo(name)
        info.size = len(content)
        info.mode = 0o644
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        tar.addfile(info, io.BytesIO(content))
with tarfile.open(archive, "r:gz") as tar:
    members = tar.getmembers()
    assert len(members) == len(manifest["members"]) and [m.name for m in members] == sorted(
        manifest["members"]
    )
    for member in members:
        assert member.isfile() and not member.name.startswith("/") and ".." not in Path(member.name).parts
        content = tar.extractfile(member).read()
        expected = manifest["members"][member.name]
        assert len(content) == member.size == expected["bytes"]
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
train, search = completed["projected_training_seconds"], completed["projected_search_seconds"]
subset = train + search
summary = {
    "scope": "engineering_capacity_only_not_scientific_result",
    "all_arm_count": len(ARMS),
    "projection_fit_count": len(ARMS) * 3,
    "projection_learned_control_rows": len(ARMS) * 3 * 3,
    "raw_completed_sha256": sha(ATTEMPT / "completed.json"),
    "raw_started_sha256": sha(ATTEMPT / "started.json"),
    "measurement_script_sha256": started["script_sha256"],
    "captured_source_sha256": started["source_sha256"],
    "recorded_plan_namespace": started["plan"]["rng_namespace"],
    "synthetic_data_seed": started["data_seed"],
    "synthetic_data_sha256": started["data_sha256"],
    "observed": {
        "whole_measurement_seconds": completed["wall_seconds"],
        "update_count_per_arm": 4,
        "search_count_per_arm": 3,
        "search_batch_cases": 64,
        "search_horizon": 12,
        "candidate_evaluations_per_case": 256,
        "train_batch_episodes": 32,
        "episode_steps": 50,
        "training_data_episodes": 768,
        "hidden_size": 64,
        "mlp_width": 107,
        "torch_threads_recorded": 2,
        "device_recorded": "cpu",
        "initialization_pairs_exercised": 1,
        "native_control_episodes": 0,
        "scientific_scores": 0,
    },
    "checkpoints": checkpoints,
    "projection": {
        "training_formula": "sum(whole_trainer_training_seconds / 4 * 1152 updates * 3 fits)",
        "search_formula": "sum(mean(three_full_batch_cem_seconds) * 50 decisions * 3 fits * 3 panels)",
        "training_seconds": train,
        "learned_search_seconds": search,
        "measured_component_projection_seconds": subset,
        "two_times_measured_component_projection_seconds": 2 * subset,
        "proposed_run_cap_seconds": 3600,
        "unmeasured_headroom_at_two_times_projection_seconds": 3600 - 2 * subset,
        "cap_is_proposed_not_authorized_by_this_receipt": True,
        "excluded": [
            "real packet assimilation and executed-action advance",
            "native simulation and data collection",
            "all reference controllers and filters",
            "training preparation/initialization and restoration",
            "prediction evaluation",
            "serialization/trace storage and manifest hashing",
            "independent audit and publication",
        ],
        "limits": "Four training updates and three same-startup-root searches per arm; no steady-state claim, confidence interval, end-to-end runtime guarantee, utility or speed superiority claim. All four updates, including the first, enter training projection. Full 12-step searches are extrapolated across all 50 decisions although real terminal horizons shrink.",
    },
}
write(EVIDENCE / "summary.json", summary)
rows = [
    "| Model | Parameters | Four-update trainer wall (s) | Three CEM calls (s) | Projected training, 3 fits (s) | Projected search, 9 rows (s) |",
    "|---|---:|---:|---|---:|---:|",
]
for row in completed["rows"]:
    rows.append(
        f"| {row['kind']} | {row['parameters']:,} | {row['whole_trainer_training_seconds']:.6f} | {', '.join(f'{v:.6f}' for v in row['three_full_batch_cem_seconds'])} | {row['projected_3fit_training_seconds']:.3f} | {row['projected_3fit_3panel_search_seconds']:.3f} |"
    )
report = f"""# Reacher public-cache engineering capacity

A completed synthetic capacity check projects **{train / 60:.2f} minutes of training plus {search / 60:.2f} minutes of learned search**, about **{subset / 60:.2f} minutes for those components**. This informs consideration of a fixed 3,600-second execution cap. It does not demonstrate end-to-end completion within that cap or scientific effectiveness.

The one recorded attempt completed in {completed["wall_seconds"]:.6f} seconds. It ran four optimizer updates per architecture on generated CPU float32 data, then three noninteractive CEM searches per architecture. The training batches contained 32 complete 50-step sequences; searches used 64 cases, 256 candidate evaluations per case, and horizon 12. The three GRUs had width 64; the two MLPs had width 107. One paired initialization was exercised, not all three planned fits. No native environment episode, utility measurement, real training corpus, or scientific score was used. The completed attempt is preserved without a rerun.

{chr(10).join(rows)}

Training is projected as each recorded trainer wall divided by four, multiplied by 1,152 updates and three fits. All four update timings, including the first, are retained. Search is projected from the mean of three full-batch calls times 50 decisions, three fits and three panels. The aggregate covers 15 projected fits and 45 projected learned control rows across all five arms. The searches use the same real startup root with different saved-plan innovation streams; they do not exercise evolving closed-loop state. Terminal search horizons would shrink in the actual study. These are short engineering samples, not a convergence or latency study.

Twice the projected measured components is {2 * subset / 60:.2f} minutes, leaving {((3600 - 2 * subset) / 60):.2f} minutes inside a proposed 60-minute cap for unmeasured execution work. This arithmetic is a planning margin, not an upper bound. **The projections exclude native simulation/data collection, reference controllers and filters, real assimilation and selected-action advances, training preparation and restoration, prediction evaluation, serialization/trace storage, manifest hashing, audit and publication.** A separate enclosing protocol must fix the actual execution and audit limits.

[Raw timings](raw-timings.json) preserve the exact completed measurement JSON, including every recorded per-update and CEM time. The five original per-arm JSON receipts are also copied unchanged under `raw-receipts/`; the larger original started receipt and all five checkpoints are retained losslessly in the archive. [Summary](summary.json) provides independently checked projection arithmetic. [Manifest](bundle-manifest.json) binds all twelve original attempt files, including five safe-loadable tensor checkpoints, all nine source/script snapshot members, and this saved-output packaging helper. The compact source snapshot contains the eight files hashed at execution plus the exact [measurement script](source-snapshot/measure.py). This is the captured source subset, not a claim of a complete transitive dependency archive.

The raw verification archive remains local at `output/reacher-cache-capacity-v1/publication-v1/reacher-cache-capacity-v1.tar.gz`; it is not asserted to be uploaded or publicly downloadable. Every archive member was reopened and checked for exact path, byte length and SHA-256. Binary checkpoints remain outside this evidence directory. [Receipt](receipt.json) binds the report, arithmetic, snapshot and archive. The recorded execution runtime says CPU and two Torch threads. [Post-run context](post-run-context.json) was collected while packaging and records the current host and installed package versions; it was **not captured at execution** and does not establish execution-time host/runtime identity. No attempt was made to infer execution timestamps from file modification times.
"""
(EVIDENCE / "README.md").write_text(report)
assert sha(EVIDENCE / "raw-timings.json") == sha(ATTEMPT / "completed.json")
assert all(sha(EVIDENCE / "raw-receipts" / f"{arm}.json") == sha(ATTEMPT / f"{arm}.json") for arm in ARMS)
assert all(sha(BASE / name) == digest for name, digest in original_hashes.items())
assert all(sha(ROOT / name) == digest for name, digest in started["source_sha256"].items())
receipt = {
    "status": "completed",
    "scope": "saved-output packaging and arithmetic only; no measurement rerun or scientific result",
    "packaged_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "measurement_status": completed["status"],
    "original_inputs_unchanged": True,
    "captured_source_hashes_match": True,
    "archive": {
        "path": archive.relative_to(ROOT).as_posix(),
        "sha256": sha(archive),
        "bytes": archive.stat().st_size,
        "member_count": len(manifest["members"]),
        "uncompressed_member_bytes": sum(v["bytes"] for v in manifest["members"].values()),
        "reopened_all_members_verified": True,
        "deterministic_headers": True,
        "uploaded": False,
    },
    "verification": {
        "strict_finite_json": True,
        "projection_arithmetic": True,
        "all_five_arm_rows_retained": True,
        "checkpoint_scope": "weights_only CPU load: class, recorded counters and parameter totals, finite weights, paired initial tensors, source/data/runtime metadata; no numerical optimizer replay or learned inference",
        "new_model_calls": 0,
        "new_optimizer_steps": 0,
        "new_native_calls": 0,
        "historical_runtime_identity_independently_proven": False,
    },
    "packager": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__))},
    "files": {p.relative_to(EVIDENCE).as_posix(): sha(p) for p in sorted(EVIDENCE.rglob("*")) if p.is_file()},
}
write(EVIDENCE / "receipt.json", receipt)
checksum_paths = [*sorted(p for p in EVIDENCE.rglob("*") if p.is_file()), archive, Path(__file__)]
(EVIDENCE / "SHA256SUMS").write_text(
    "".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in checksum_paths)
)
print(
    json.dumps(
        {
            "evidence": str(EVIDENCE),
            "receipt_sha256": sha(EVIDENCE / "receipt.json"),
            "archive": receipt["archive"],
            "projected_seconds": subset,
            "measured_attempt_seconds": completed["wall_seconds"],
        }
    )
)
