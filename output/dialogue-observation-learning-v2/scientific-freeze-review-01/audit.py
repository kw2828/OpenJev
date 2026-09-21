"""Metadata-only V2 freeze verifier; no project/model/numerical imports.

Prints one JSON result for external capture. The caller separately authenticates
the actual freeze process terminal. No scientific payload is deserialized here.
"""

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "output/dialogue-observation-learning-v1"
VERSION = "dialogue-observation-scientific-v2"
OLD_PLAN_PIN = "acb79b4600c66966762895d28eb2dc1d2be15c761d677c5e87c5750dde47f237"
PREPARED_PIN = "d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83"
PREPARED_PLAN_PIN = "4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e"
FAILED_PIN = "41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f"
PROTOCOL = "research/dialogue-observation-learning-protocol-v2.md"
NEW_SOURCES = {
    "scripts/study_dialogue_observation_v2.py",
    "tests/test_study_dialogue_observation_v2.py",
    "scripts/report_dialogue_observation_v2.py",
    "tests/test_report_dialogue_observation_v2.py",
    "scripts/supervise_dialogue_observation_v2.py",
    "tests/test_supervise_dialogue_observation_v2.py",
    "src/openjev/research/suspend_clock.py",
    "tests/test_suspend_clock.py",
    PROTOCOL,
    "output/dialogue-observation-learning-v2/result-audit-01/audit.py",
    "output/dialogue-observation-learning-v2/result-audit-01/test_audit.py",
}
TRAIN_LIMITS = {"wall_seconds": 28800, "rss_bytes": 8 * 1024**3,
                "mps_driver_bytes": 8 * 1024**3, "output_bytes": 2 * 1024**3}
FREEZE_LIMITS = {"wall_seconds": 300, "rss_bytes": 8 * 1024**3,
                 "output_bytes": 512 * 1024**2}
SAME_FIELDS = (
    "runtime", "config", "prepared", "prepared_completed_sha256", "prepared_plan_sha256",
    "fit_order", "loss_counts", "loss_weights", "orders_sha256", "expected",
    "evaluation_rows_sha256", "references_sha256", "quality_scoring_in_runner",
    "no_retry", "checkpoint_scope",
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "Regular file required: " + str(path))
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    # Called only for plans, receipts and request metadata, never scientific rows.
    return json.loads(Path(path).read_text())


def relative(name):
    need(type(name) is str and name and not Path(name).is_absolute()
         and ".." not in Path(name).parts and Path(name).as_posix() == name,
         "Safe canonical member path")
    return name


def manifest(folder, files):
    paths = list(folder.rglob("*"))
    need(not folder.is_symlink() and not any(p.is_symlink() for p in paths), "No symlinks")
    actual = {p.relative_to(folder).as_posix() for p in paths if p.is_file()}
    need(actual == set(files) | {"completed.json"}, "Exact freeze file closure")
    for name, entry in files.items():
        path = folder / relative(name)
        need(set(entry) == {"sha256", "bytes"} and digest(entry["sha256"])
             and type(entry["bytes"]) is int and entry["bytes"] >= 0,
             "Manifest entry schema")
        need(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
             "Payload identity: " + name)
    return sum(p.stat().st_size for p in paths if p.is_file())


def timing(started, done):
    need(started["clock_backend"] == done["clock_backend"] == "mach_continuous_time",
         "Qualified Mac native suspend-inclusive clock")
    for field in ("started_ns", "parent_started_ns", "deadline_ns"):
        need(type(started[field]) is int and started[field] >= 0
             and type(done[field]) is int and done[field] == started[field], "Stable clock " + field)
    need(done["parent_started_ns"] == done["started_ns"]
         and done["deadline_ns"] - done["started_ns"] == 300_000_000_000,
         "Freeze owns one fixed 300-second deadline")
    need(started["timing_available"] is True and done["timing_available"] is True
         and started["supervision_sha256"] is None and done["supervision_sha256"] is None,
         "Complete own-worker freeze timing")
    need(all(started[k] is None for k in ("finished_ns", "elapsed_ns", "wall_seconds")),
         "Started record has no fabricated terminal measurement")
    need(type(done["finished_ns"]) is int and type(done["elapsed_ns"]) is int
         and done["started_ns"] <= done["finished_ns"] < done["deadline_ns"]
         and done["elapsed_ns"] == done["finished_ns"] - done["started_ns"],
         "Strict suspend-inclusive deadline and elapsed identity")
    need(type(done["wall_seconds"]) in (int, float) and math.isfinite(done["wall_seconds"])
         and done["wall_seconds"] == done["elapsed_ns"] / 1e9,
         "Worker wall time derives only from native elapsed nanoseconds")
    need(done["wall_scope"] == "Suspend-inclusive worker elapsed through payload manifest; parent absolute deadline checked through completion write/hash"
         and done["phase_timing_scope"] == "Fit/update/evaluation/checkpoint durations retain V1 perf_counter diagnostics and may exclude suspend; never used for admission",
         "Explicit root and diagnostic timing scopes")


def inspect(freeze, plan_pin, completed_pin=None):
    need(digest(plan_pin) and (completed_pin is None or digest(completed_pin)), "External pin syntax")
    need(sha(freeze / "plan.json") == plan_pin, "External V2 plan pin")
    actual_done_pin = sha(freeze / "completed.json")
    need(completed_pin is None or actual_done_pin == completed_pin, "External completion pin")
    old_path = OLD / "scientific-freeze-01/plan.json"
    need(sha(old_path) == OLD_PLAN_PIN, "Pinned original scientific plan")
    old, plan, done = read(old_path), read(freeze / "plan.json"), read(freeze / "completed.json")
    need(plan["version"] == done["version"] == VERSION and done["status"] == "completed"
         and done["phase"] == "freeze" and done["plan_sha256"] == plan_pin
         and done["no_retry"] is True, "Completed V2 metadata freeze")
    need(type(done["model_calls"]) is int and type(done["encoder_calls"]) is int
         and done["model_calls"] == done["encoder_calls"] == 0, "No recorded model or encoder calls")
    need(plan["previous_plan_sha256"] == OLD_PLAN_PIN
         and all(plan[k] == old[k] for k in SAME_FIELDS), "Unchanged scientific recipe and opaque identities")
    need(plan["quality_scoring_in_runner"] is False and plan["no_retry"] is True,
         "No scoring or retry in scientific runner")
    sources, old_sources = plan["source_sha256"], old["source_sha256"]
    need(len(old_sources) == 53 and len(sources) == 64 and not set(old_sources) & NEW_SOURCES
         and set(sources) == set(old_sources) | NEW_SOURCES
         and all(sources[n] == p for n, p in old_sources.items())
         and done["source_sha256"] == sources, "Exact unchanged53 plus new11 source map")
    expected_files = {"started.json", "allocation.json", "evaluation-rows.jsonl", "references.json", "plan.json"}
    expected_files |= {"sources/" + relative(n) for n in sources}
    need(len(expected_files) == 69 and set(done["files"]) == expected_files, "69 payloads and one completion")
    total_bytes = manifest(freeze, done["files"])
    for name, pin in sources.items():
        need(digest(pin) and sha(ROOT / name) == sha(freeze / "sources" / name) == pin,
             "Source and frozen snapshot: " + name)
    for name, key in (("evaluation-rows.jsonl", "evaluation_rows_sha256"), ("references.json", "references_sha256")):
        need(sha(freeze / name) == plan[key], "Opaque unchanged evaluator/reference identity")
    need(done["expected"] == plan["expected"] == old["expected"], "Unchanged complete operation totals")

    spec = read(freeze / "allocation.json")
    need(spec == plan["allocation"] and sha(freeze / "allocation.json") == plan["allocation_sha256"], "Allocation snapshot")
    need(set(spec) == {"version", "prepared_completed_sha256", "prepared_plan_sha256", "cost_completed",
                      "cost_audit", "protocol", "limits", "freeze_limits", "failed_attempt"}
         and spec["version"] == "dialogue-observation-allocation-v2"
         and spec["limits"] == TRAIN_LIMITS and spec["freeze_limits"] == FREEZE_LIMITS,
         "Exact prospective allocation and phase caps")
    for key in ("cost_completed", "cost_audit", "protocol", "failed_attempt"):
        ref = spec[key]
        need(set(ref) == {"path", "sha256"} and digest(ref["sha256"])
             and sha(ref["path"]) == ref["sha256"], "Allocation prerequisite: " + key)
    need(all(spec[k] == old["allocation"][k] for k in ("cost_completed", "cost_audit")), "Original qualified cost inputs")
    need(Path(spec["protocol"]["path"]).resolve() == ROOT / PROTOCOL
         and spec["protocol"]["sha256"] == sources[PROTOCOL], "V2 protocol source join")
    need(Path(spec["failed_attempt"]["path"]).resolve() == OLD / "failed-scientific-publication-01/manifest.json"
         and spec["failed_attempt"]["sha256"] == FAILED_PIN, "Failed V1 attempt identity")
    failed = read(spec["failed_attempt"]["path"])
    need(failed["status"] == "failed_technical_timing"
         and all(failed[k] is False for k in ("resume_permitted", "partial_scoring_permitted",
                    "quality_metrics_opened", "individual_predictions_decoded", "weights_loaded")),
         "Prior attempt remains failed, unscored and unresumed")
    cost_done, cost_audit = read(spec["cost_completed"]["path"]), read(spec["cost_audit"]["path"])
    need(cost_done["status"] == cost_audit["status"] == "completed" and cost_audit["agreement"] is True
         and cost_audit["execution_completed_sha256"] == spec["cost_completed"]["sha256"], "Successful inherited cost audit")
    prepared = Path(plan["prepared"])
    need(prepared.resolve() == OLD / "preparation-02"
         and sha(prepared / "completed.json") == plan["prepared_completed_sha256"] == spec["prepared_completed_sha256"] == PREPARED_PIN
         and sha(prepared / "plan.json") == plan["prepared_plan_sha256"] == spec["prepared_plan_sha256"] == PREPARED_PLAN_PIN,
         "Qualified full preparation receipts")
    parent = read(prepared / "plan.json")
    need(all(plan[k] == parent[k] for k in ("config", "runtime", "loss_counts", "loss_weights"))
         and sha(prepared / "orders.json") == plan["orders_sha256"], "Prepared recipe/runtime and opaque paired orders")
    started = read(freeze / "started.json")
    request = started["request"]
    need(set(request) == {"command", "allocation", "allocation_sha256", "prepared", "out"}
         and request["command"] == "freeze" and Path(request["out"]).resolve() == freeze.resolve()
         and Path(request["prepared"]).resolve() == prepared.resolve()
         and request["allocation_sha256"] == plan["allocation_sha256"]
         and sha(request["allocation"]) == plan["allocation_sha256"], "Exact bounded freeze request")
    need(started["version"] == VERSION and started["runtime"] == plan["runtime"]
         and started["no_retry"] is True, "Started runtime/version/retry join")
    timing(started, done)
    need(type(done["peak_rss_bytes"]) is int and 0 < done["peak_rss_bytes"] <= FREEZE_LIMITS["rss_bytes"]
         and total_bytes <= FREEZE_LIMITS["output_bytes"], "Recorded RSS and actual output caps")
    # Rehash every checked byte identity before accepting a stable saved snapshot.
    need(manifest(freeze, done["files"]) == total_bytes
         and sha(freeze / "completed.json") == actual_done_pin
         and sha(old_path) == OLD_PLAN_PIN, "Stable end manifest and completion")
    for name, pin in sources.items():
        need(sha(ROOT / name) == pin, "Source changed during audit: " + name)
    for key in ("cost_completed", "cost_audit", "protocol", "failed_attempt"):
        need(sha(spec[key]["path"]) == spec[key]["sha256"], "Prerequisite changed during audit")
    for path, pin in ((prepared / "completed.json", PREPARED_PIN),
                      (prepared / "plan.json", PREPARED_PLAN_PIN),
                      (prepared / "orders.json", plan["orders_sha256"]),
                      (Path(request["allocation"]), plan["allocation_sha256"])):
        need(sha(path) == pin, "Prepared/allocation input changed during audit")
    return {"status": "completed", "agreement": True, "version": "dialogue-observation-freeze-review-v2",
            "plan_sha256": plan_pin, "execution_completed_sha256": actual_done_pin,
            "previous_plan_sha256": OLD_PLAN_PIN, "failed_attempt_sha256": FAILED_PIN,
            "scientific_sources_verified": 64, "original_sources_unchanged": 53,
            "freeze_payloads_verified": 69, "freeze_total_files": 70,
            "unchanged_scientific_fields": list(SAME_FIELDS), "expected_work": plan["expected"],
            "allocation": spec, "worker_timing": {k: done[k] for k in (
                "clock_backend", "started_ns", "parent_started_ns", "deadline_ns", "finished_ns", "elapsed_ns", "wall_seconds")},
            "freeze_peak_rss_bytes": done["peak_rss_bytes"], "freeze_output_bytes": total_bytes,
            "reported_model_calls": 0, "reported_encoder_calls": 0, "reviewer_model_calls": 0,
            "task_metrics_computed": False, "process_terminal_verified": False,
            "scope": ["Plans, request and receipt metadata are decoded. Evaluator rows, references and orders are only hashed.",
                      "Scientific counts and semantics inherit the pinned V1 plan and qualified preparation; this check establishes exact V2 equality.",
                      "Native clock and RSS are source-bound worker witnesses, not independent live measurements.",
                      "The caller must separately authenticate the actual freeze process terminal and exit status; this verifier does not establish process completion.",
                      "No training, quality scoring, weight/cache decoding or model operation occurs."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--completed-sha256", help="Optional externally supplied completion pin")
    args = parser.parse_args()
    start = time.perf_counter()
    source_pin = sha(Path(__file__))
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    def timeout(*_):
        raise TimeoutError("60-second saved-metadata review cap")

    handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        result = inspect(args.freeze, args.plan_sha256, args.completed_sha256)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        elapsed = time.perf_counter() - start
        need(elapsed < 60 and rss <= 2 * 1024**3 and sha(Path(__file__)) == source_pin, "Review resource/source limits")
        result.update(request=request, source_sha256=source_pin, review_wall_seconds=elapsed,
                      review_peak_rss_bytes=rss, completion_externally_pinned=args.completed_sha256 is not None)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps({"status": "failed", "request": request, "source_sha256": source_pin,
                          "error_type": type(error).__name__, "error": str(error)}, allow_nan=False), file=sys.stderr)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    main()
