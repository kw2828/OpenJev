"""Portable primary-metric recheck from an authenticated saved-output subset.

No training, cache, corpus, model, or main-reporter module is loaded. The exact
independent audit arithmetic is reused by source hash. Saved absolute paths are
never followed. This is not a repeat of the full technical/provenance audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-token-alignment-portable-metrics-v1"
STUDY_VERSION = "dialogue-token-alignment-scientific-v1"
HELPER = "output/dialogue-token-alignment-scientific-v1/audit-01/audit.py"
HELPER_SHA256 = "fc0f252a4ce12a092993bc9033dd5bb11250c6ed0522ec49d5e097eac62080b1"
METHODS = ("flat_stratum", "token_mean", "token_aligned")
SEEDS = (6201, 6202, 6203)
ORDER = [f"{m}-{s}" for i, s in enumerate(SEEDS) for m in METHODS[i:]+METHODS[:i]]
SUPPORT = {"evaluation_rows": 13599, "heldout_rows": 7819, "changed": 578, "retained": 7241,
           "heldout_services": 6, "changed_types": [0, 5, 29, 0]}
WALL_SECONDS = 60
OUTPUT_BYTES = 32*1024**2
SCOPE = (
    "Metric-only recheck: authenticated selected payloads, all-nine row and probability coverage, "
    "raw-log NLL/Brier/accuracy and selected-candidate branch errors, 22 exact behavioral checks, "
    "and paired correctness tables for heldout_service all/changed/retained. Float metric agreement "
    "uses inherited numerical tolerances; count/rule agreement is exact. Other descriptive groups "
    "are outside scope. Reuses the SHA-pinned independent audit arithmetic. "
    "Does not repeat original-corpus identity, training/source/cache/checkpoint authentication, "
    "runtime, budget, gradient or internal normalization-witness verification, reference metrics, "
    "calibration assessment, or model calls. Full technical admission is an authenticated main-report "
    "claim, not established by this recheck. This is exposed TRAIN development with privileged "
    "previous state, not autonomous recurrence or fresh evaluation."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            check(); h.update(chunk)
    check()
    return h.hexdigest()


def pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_arithmetic(check=lambda: None):
    path = ROOT/HELPER
    require(digest(path, check) == HELPER_SHA256, "Arithmetic helper source pin")
    spec = importlib.util.spec_from_file_location("_portable_alignment_arithmetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_manifest():
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"}
    names |= {f"orders-{s}.npy" for s in SEEDS}
    return names | {f"fits/{fit}/{name}" for fit in ORDER
                    for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}


def manifest_schema(members):
    require(type(members) is dict and set(members) == expected_manifest(), "Original 44-file manifest schema")
    for name, item in members.items():
        require(type(item) is dict and set(item) == {"sha256", "bytes"} and pin(item["sha256"])
                and type(item["bytes"]) is int and item["bytes"] >= 0, "Manifest record: "+name)


def bind(path, item, check):
    require(Path(path).is_file() and Path(path).stat().st_size == item["bytes"]
            and digest(path, check) == item["sha256"], "Selected payload binding: "+str(path))


def authenticate(args, arithmetic, check):
    """All selected hashes precede any plan, row, summary or NPZ decoding."""
    run, report = Path(args.run).resolve(), Path(args.report).resolve()
    require(pin(args.completed_sha256) and pin(args.report_receipt_sha256), "External SHA256 pins")
    require(digest(run/"completed.json", check) == args.completed_sha256, "External completion pin")
    require(digest(report/"receipt.json", check) == args.report_receipt_sha256, "External report pin")
    done, rr = arithmetic.read(run/"completed.json"), arithmetic.read(report/"receipt.json")
    require(done["status"] == "completed" and done["phase"] == "train" and done["version"] == STUDY_VERSION
            and done["completed_fits"] == done["expected_fits"] == ORDER, "All nine completed identities")
    require(done["progress"]["completed_fits"] == ORDER and done["progress"]["active_fit"] is None,
            "No incomplete active fit")
    require(rr["status"] == "completed" and rr["version"] == STUDY_VERSION
            and rr["technical_validity_passed"] is True
            and rr["execution_completed_sha256"] == args.completed_sha256
            and rr["plan_sha256"] == done["plan_sha256"], "Matching completed main report")
    manifest_schema(done["files"])
    require(rr["execution_files"] == 44 and rr["execution_members"] == done["files"], "Report execution manifest identity")
    require(rr["source_sha256"] == done["source_sha256"]["scripts/report_dialogue_alignment.py"],
            "Recorded reporter source identity")
    selected = {name: done["files"][name] for name in ("plan.json", "evaluation-rows.jsonl")}
    selected.update({f"fits/{name}/predictions.npz": done["files"][f"fits/{name}/predictions.npz"] for name in ORDER})
    for name, item in selected.items():
        bind(run/name, item, check)
    summary_member = rr["files"]["summary.json"]
    require(set(summary_member) == {"sha256", "bytes"} and pin(summary_member["sha256"])
            and type(summary_member["bytes"]) is int and summary_member["bytes"] > 0, "Report summary member")
    bind(report/"summary.json", summary_member, check)
    require(done["plan_sha256"] == selected["plan.json"]["sha256"], "Original plan binding")
    plan, summary = arithmetic.read(run/"plan.json"), arithmetic.read(report/"summary.json")
    require(plan["version"] == STUDY_VERSION and plan["expected_fits"] == ORDER
            and plan["config"]["methods"] == list(METHODS) and plan["config"]["seeds"] == list(SEEDS)
            and plan["source_sha256"] == done["source_sha256"], "Plan identity")
    require(summary["status"] == "completed" and summary["version"] == STUDY_VERSION
            and summary["technical_validity_passed"] is True
            and summary["execution_completed_sha256"] == args.completed_sha256
            and summary["plan_sha256"] == done["plan_sha256"]
            and summary["source_sha256"] == plan["source_sha256"], "Main summary identity")
    rows = [arithmetic.decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    meta = arithmetic.row_metadata(rows)
    require(meta["ids"].tolist() == plan["evaluation_row_indices"], "Original evaluation order")
    held = plan["split"]["heldout_services"]
    require(len(held) == len(set(held)) == SUPPORT["heldout_services"]
            and all(row["heldout_service"] == (row["service"] in held) for row in rows), "Recorded held-out service membership")
    changed = meta["held"] & meta["changed"]
    actual = {"evaluation_rows": len(rows), "heldout_rows": int(meta["held"].sum()),
              "changed": int(changed.sum()), "retained": int((meta["held"] & ~meta["changed"]).sum()),
              "heldout_services": len(held),
              "changed_types": [int((changed & (meta["target_types"] == c)).sum()) for c in range(4)]}
    require(actual == SUPPORT, "Fixed primary support")
    return rows, meta, summary, selected, summary_member, done["plan_sha256"]


def execute(args):
    out = Path(args.out).resolve()
    # Do not create outputs inside an authenticated input tree or over an input.
    inputs = [Path(args.run).resolve(), Path(args.report).resolve()]
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in inputs), "Separate exclusive output directory")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    request = {k: str(v) for k, v in vars(args).items()}
    source_pin = digest(__file__)
    def check():
        require(time.monotonic()-start <= WALL_SECONDS, "Portable recheck wall cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= OUTPUT_BYTES, "Portable output cap")
    try:
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": source_pin,
            "arithmetic_source": HELPER, "arithmetic_source_sha256": HELPER_SHA256, "scope": SCOPE,
            "no_retry": True, "model_calls": 0, "wall_seconds_limit": WALL_SECONDS})
        arithmetic = load_arithmetic(check)
        rows, meta, report, selected, summary_member, plan_pin = authenticate(args, arithmetic, check)
        fits, choices = {}, {}
        for name in ORDER:
            check()
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                require(len(archive.files) == 2 and set(archive.files) == {"row_indices", "log_probs"}, "Prediction NPZ schema")
                fits[name], choices[name] = arithmetic.score(rows, meta, {k: archive[k] for k in archive.files})
        pairs = {control: {str(seed): arithmetic.pair_counts(meta, choices[f"token_aligned-{seed}"], choices[f"{control}-{seed}"])
                           for seed in SEEDS} for control in METHODS[:2]}
        rule = arithmetic.continuation(fits)
        arithmetic.compare_report(fits, pairs, rule, report)
        result = {"status": "completed", "version": VERSION, "metric_agreement": True,
            "recorded_main_report_technical_validity": True, "scope": SCOPE, "fits": fits, "pairs": pairs,
            "continuation": rule, "primary_support_per_fit": SUPPORT,
            "execution_completed_sha256": args.completed_sha256, "plan_sha256": plan_pin,
            "report_receipt_sha256": args.report_receipt_sha256,
            "selected_run_members": selected, "report_summary_member": summary_member,
            "model_calls": 0, "checkpoint_deserializations": 0}
        write(out/"summary.json", result)
        # Detect input mutation during arithmetic without decoding any further data.
        for name, item in selected.items():
            bind(Path(args.run)/name, item, check)
        bind(Path(args.report)/"summary.json", summary_member, check)
        require(digest(Path(args.run)/"completed.json", check) == args.completed_sha256
                and digest(Path(args.report)/"receipt.json", check) == args.report_receipt_sha256
                and digest(ROOT/HELPER, check) == HELPER_SHA256 and digest(__file__, check) == source_pin,
                "Final input/source stability")
        files = {p.name: {"sha256": digest(p, check), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "request": request,
            "metric_agreement": True, "continuation_passed": rule["passed"], "scope": SCOPE,
            "source_sha256": source_pin, "arithmetic_source": HELPER, "arithmetic_source_sha256": HELPER_SHA256,
            "files": files, "wall_seconds": time.monotonic()-start, "model_calls": 0,
            "encoder_calls": 0, "checkpoint_deserializations": 0, "no_retry": True})
        check()
        return result
    except BaseException as error:
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                "source_sha256": source_pin, "arithmetic_source_sha256": HELPER_SHA256, "scope": SCOPE,
                "error": repr(error), "wall_seconds": time.monotonic()-start, "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original checker failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure preservation: "+repr(secondary))
        raise


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({"metric_agreement": result["metric_agreement"], "continuation_passed": result["continuation"]["passed"]}))


if __name__ == "__main__":
    cli()
