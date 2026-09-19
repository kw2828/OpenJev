"""Independent saved-output arithmetic for the fixed shared-prefix systems pilot.

No scorer, tokenizer, MLX model, random generator or inference is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("serial", "batch", "shared_serial", "shared_batch")
LABELS = ("Serial full", "Batch full", "Shared + serial", "Shared + batch")
SOURCES = {
    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "scripts/benchmark_shared_prefix.py", "tests/test_shared_prefix.py",
    "tests/test_benchmark_shared_prefix.py", "research/shared-prefix-protocol.md",
}
FAILURES = {"failed.json", "late-completion.json", "cleanup-error.json",
            "completion-before-cleanup-error.json"}
TOLERANCE = .005


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")
    return value


def parse_json(text):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(text, parse_constant=invalid, object_pairs_hook=pairs)


def read_json(path):
    return parse_json(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "Invalid " + name)
    return float(value)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid " + name)
    return value


def close(actual, expected, name):
    require(type(actual) in (float, int) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            "Inconsistent " + name)


def prefix_length(tokens):
    limit = min(map(len, tokens)) - 1
    for index in range(limit):
        if any(row[index] != tokens[0][index] for row in tokens[1:]):
            return index
    return limit


def expected_work(tokens, method):
    lengths = [len(row) for row in tokens]
    prefix = prefix_length(tokens) if method.startswith("shared_") else 0
    suffixes = [n - prefix for n in lengths]
    batched = method in ("batch", "shared_batch")
    slots = len(lengths) * max(suffixes) if batched else sum(suffixes)
    branches = 1 if batched else len(lengths)
    return {"calls": branches + int(prefix > 0), "prefill_calls": int(prefix > 0),
            "branch_calls": branches, "input_token_slots": slots + prefix,
            "prefix_tokens": prefix, "padding_token_slots": slots - sum(suffixes)}


def validate_requests(requests):
    require(type(requests) is list and len(requests) == 54, "Exactly54 requests required")
    expected = [(q, size, c, case) for q in (1, 2, 4) for size in ("short", "long")
                for c in (2, 4, 12) for case in range(3)]
    for row, identity in zip(requests, expected):
        q, size, c, case = identity
        require((row["question_count"], row["context_size"], row["candidate_count"], row["case"])
                == identity and all(type(row[k]) is int for k in ("question_count", "candidate_count", "case")),
                "Request cell/order mismatch")
        require(row["id"] == f"q{q}-{size}-c{c}-case{case}", "Request identity mismatch")
        request = row["request"]
        require(type(request["context"]) is str and request["context"].strip(), "Missing context")
        require(len(request["questions"]) == q and len(row["tokens"]) == q, "Question/token count")
        for j, question in enumerate(request["questions"]):
            require(question["id"] == f"field{j}" and type(question["question"]) is str
                    and question["question"].strip(), "Question identity")
            require([x["id"] for x in question["candidates"]] == [f"c{i}" for i in range(c)],
                    "Candidate identity")
            descriptions = [x["description"] for x in question["candidates"]]
            require(all(type(x) is str and x.strip() for x in descriptions)
                    and len(set(descriptions)) == c, "Candidate descriptions")
        require(all(type(t) is list and 1 <= len(t) <= 4096
                    and all(type(x) is int and 0 <= x < 151936 for x in t) for t in row["tokens"]),
                "Invalid frozen token arrays")
        require(type(row["prefix_tokens"]) is int and row["prefix_tokens"] == prefix_length(row["tokens"]),
                "Frozen token prefix mismatch")


def validate_answers(record, request):
    questions = request["request"]["questions"]
    require(len(record["answers"]) == len(record["candidate_logits"]) == len(questions), "Answer count")
    for index, (answer, logits, question) in enumerate(zip(record["answers"], record["candidate_logits"], questions)):
        ordered = sorted(question["candidates"], key=lambda c: c["description"])
        ids = [c["id"] for c in ordered]
        require(answer["id"] == question["id"] and set(answer["probabilities"]) == set(ids),
                "Answer/candidate identity mismatch")
        require(type(logits) is list and len(logits) == len(ids)
                and all(type(v) in (int, float) and math.isfinite(v) for v in logits), "Invalid candidate logits")
        values = np.asarray(logits, dtype=np.float64)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        actual = [number(answer["probabilities"][key], "candidate probability") for key in ids]
        require(all(x <= 1 for x in actual), "Probability above1")
        close(math.fsum(actual), 1., "probability sum")
        for value, target in zip(actual, probabilities):
            close(value, float(target), "candidate softmax")
        require(answer["choice"] == ids[int(np.argmax(values))], "Choice does not match saved logits")
        require(number(answer["candidate_token_mass"], "candidate mass") <= 1., "Mass above1")
        entropy = -math.fsum(p * math.log(p) for p in actual if p > 0)
        close(answer["entropy_nats"], entropy, "entropy")
        require(type(answer["input_tokens"]) is int
                and answer["input_tokens"] == len(request["tokens"][index]), "Input token count")


def parity(reference, record):
    choices = 0
    probability = mass = logit = 0.
    for left, right, a, b in zip(reference["answers"], record["answers"],
                                  reference["candidate_logits"], record["candidate_logits"]):
        choices += left["choice"] != right["choice"]
        probability = max(probability, *(abs(v - right["probabilities"][k])
                                         for k, v in left["probabilities"].items()))
        mass = max(mass, abs(left["candidate_token_mass"] - right["candidate_token_mass"]))
        logit = max(logit, *(abs(x - y) for x, y in zip(a, b)))
    return {"choice_mismatches": choices, "max_probability_difference": probability,
            "max_mass_difference": mass, "max_candidate_logit_difference": logit,
            "pass": choices == 0 and probability <= TOLERANCE and mass <= TOLERANCE}


def audit_records(requests, records):
    validate_requests(requests)
    expected = []
    for index, request in enumerate(requests):
        expected.extend((request["id"], "allocator_cold", 0, method) for method in METHODS)
        expected.extend((request["id"], "warm", repetition, METHODS[(index + repetition + j) % 4])
                        for repetition in range(5) for j in range(4))
    require(len(records) == len(expected) == 1296, "Exactly1296 records required")
    by_request = {r["id"]: r for r in requests}
    references = {}
    checked = []
    for record, key in zip(records, expected):
        require((record["request_id"], record["phase"], record["repetition"], record["method"]) == key
                and type(record["repetition"]) is int, "Missing, duplicate, foreign or reordered record")
        request = by_request[record["request_id"]]
        validate_answers(record, request)
        number(record["latency_ms"], "latency", positive=True)
        for field in ("peak_active_bytes", "baseline_active_bytes", "cached_allocator_bytes"):
            integer(record[field], field)
        require(record["peak_active_bytes"] >= record["baseline_active_bytes"], "Peak below baseline memory")
        work = expected_work(request["tokens"], record["method"])
        require(set(record["work"]) == set(work)
                and all(type(record["work"][k]) is int and record["work"][k] == v for k, v in work.items()),
                "Model call/token/padding count mismatch")
        require(record["questions_sequential"] is (record["method"] in ("serial", "shared_serial")),
                "Execution mode mismatch")
        if key[1:] == ("allocator_cold", 0, "serial"):
            references[key[0]] = record
        result = parity(references[key[0]], record)
        require(set(record["parity"]) == set(result)
                and type(record["parity"]["pass"]) is bool
                and type(record["parity"]["choice_mismatches"]) is int, "Parity schema mismatch")
        for name, value in result.items():
            if name in ("pass", "choice_mismatches"):
                require(record["parity"][name] == value, "Saved parity mismatch")
            else:
                close(record["parity"][name], value, "saved parity " + name)
        checked.append({**record, "parity": result})
    return checked


def authenticate(experiment, protocol_sha256, completed_sha256, root=ROOT):
    run = experiment / "run-01"
    require(not any((folder / name).exists() for folder in (experiment, run) for name in FAILURES),
            "Failed or demoted execution")
    require(sha(experiment / "protocol.json") == digest(protocol_sha256)
            and sha(run / "completed.json") == digest(completed_sha256), "External hash mismatch")
    protocol, done = read_json(experiment / "protocol.json"), read_json(run / "completed.json")
    require(protocol["version"] == "shared-prefix-v1"
            and protocol["model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
            and protocol["revision"] == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
            and protocol["methods"] == list(METHODS), "Fixed model/method identity")
    digest(protocol["prompt_protocol_sha256"])
    for key, value in {"request_count": 54, "warm_repetitions": 5, "allocator_cold_calls": 216,
                       "warm_calls": 1080}.items():
        require(type(protocol[key]) is int and protocol[key] == value, "Frozen workload: " + key)
    require(protocol["parity"]["choice_mismatches"] == 0
            and protocol["parity"]["max_probability_difference"] == TOLERANCE
            and protocol["parity"]["max_candidate_mass_difference"] == TOLERANCE, "Frozen tolerance changed")
    require(set(protocol["source_sha256"]) == SOURCES, "Exact six source bindings")
    for name, value in protocol["source_sha256"].items():
        require(sha(root / name) == digest(value), "Bound source changed: " + name)
    require(type(protocol["local_model_files_sha256"]) is dict
            and {"config.json", "tokenizer.json", "model.safetensors"} <= set(protocol["local_model_files_sha256"]),
            "Missing local model identities")
    for name, value in protocol["local_model_files_sha256"].items():
        require(Path(name).name == name, "Unsafe model filename")
        digest(value)
    require(done["status"] == "complete" and done["protocol_sha256"] == protocol_sha256,
            "Completed bound execution required")
    require(type(done["records"]) is int and done["records"] == 1296, "Completion record count")
    number(done["wall_seconds"], "execution wall", positive=True)
    for path, value in ((experiment / "requests.json", protocol["requests_sha256"]),
                        (run / "records.jsonl", done["records_sha256"]), (run / "runtime.json", done["runtime_sha256"])):
        require(sha(path) == digest(value), "Payload hash mismatch: " + path.name)
    runtime = read_json(run / "runtime.json")
    require(runtime["protocol_sha256"] == protocol_sha256
            and set(runtime["versions"]) == {"mlx", "mlx-lm", "numpy"}, "Runtime binding")
    require(all(type(v) is str and v for v in runtime["versions"].values())
            and all(type(runtime[k]) is str and runtime[k] for k in ("python", "platform", "device", "hardware")),
            "Runtime identity incomplete")
    number(runtime["model_load_ms"], "model load", positive=True)
    # The records are JSONL; reuse the strict object parser without importing the producer.
    records = []
    for line in (run / "records.jsonl").read_text().splitlines():
        require(line.strip(), "Blank record")
        record = parse_json(line)
        records.append(record)
    requests = read_json(experiment / "requests.json")
    records = audit_records(requests, records)
    failed = sum(not r["parity"]["pass"] for r in records)
    require(type(done["parity_failed_rows"]) is int and done["parity_failed_rows"] == failed
            and done["all_parity_passed"] is (failed == 0), "Completion parity totals")
    return protocol, done, runtime, requests, records


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95, method="linear")),
            "min": float(x.min()), "max": float(x.max())}


def aggregate(records, requests, selected_ids):
    group = [r for r in records if r["request_id"] in selected_ids]
    warm = [r for r in group if r["phase"] == "warm"]
    medians = {method: {identity: float(np.median([r["latency_ms"] for r in warm
                                                 if r["method"] == method and r["request_id"] == identity]))
                        for identity in selected_ids} for method in METHODS}
    result = {}
    for method in METHODS:
        rows = [r for r in group if r["method"] == method]
        warm_rows = [r for r in rows if r["phase"] == "warm"]
        failed = [dict(request_id=r["request_id"], phase=r["phase"], repetition=r["repetition"],
                       **r["parity"]) for r in rows if not r["parity"]["pass"]]
        ratios = {identity: medians["serial"][identity] / medians[method][identity]
                  for identity in sorted(selected_ids)}
        result[method] = {
            "records": len(rows), "warm_latency_ms": distribution([r["latency_ms"] for r in warm_rows]),
            "allocator_cold_latency_ms": distribution([r["latency_ms"] for r in rows if r["phase"] == "allocator_cold"]),
            "paired_request_median_speed_ratios": ratios, "speed_ratio_distribution": distribution(list(ratios.values())),
            "all_parity_passed": not failed, "failed_records": failed,
            "choice_mismatches": sum(r["parity"]["choice_mismatches"] for r in rows),
            **{key: max(r["parity"][key] for r in rows) for key in
               ("max_probability_difference", "max_mass_difference", "max_candidate_logit_difference")},
            "warm_peak_active_bytes": distribution([r["peak_active_bytes"] for r in warm_rows]),
            "maximum_peak_active_bytes_all_phases": max(r["peak_active_bytes"] for r in rows),
            "warm_baseline_active_bytes": distribution([r["baseline_active_bytes"] for r in warm_rows]),
            "warm_cached_allocator_bytes": distribution([r["cached_allocator_bytes"] for r in warm_rows]),
            "work_totals": {k: sum(r["work"][k] for r in rows) for k in rows[0]["work"]},
        }
    return result


def summarize(protocol, done, runtime, requests, records):
    cells = {}
    for q in (1, 2, 4):
        for context in ("short", "long"):
            for c in (2, 4, 12):
                ids = {r["id"] for r in requests if (r["question_count"], r["context_size"], r["candidate_count"])
                       == (q, context, c)}
                cells[f"q{q}-{context}-c{c}"] = aggregate(records, requests, ids)
    strata = {}
    for field in ("question_count", "context_size"):
        strata[field] = {str(value): aggregate(records, requests, {r["id"] for r in requests if r[field] == value})
                         for value in sorted({r[field] for r in requests})}
    overall = aggregate(records, requests, {r["id"] for r in requests})
    return {"status": "completed", "study": "shared-prefix-v1", "records": 1296,
            "request_count": 54, "allocator_cold_records": 216, "warm_records": 1080,
            "all_parity_passed": done["all_parity_passed"], "failed_parity_records": done["parity_failed_rows"],
            "methods": overall, "cells": cells, "strata": strata, "runtime": runtime,
            "execution_wall_seconds": done["wall_seconds"], "source_sha256": protocol["source_sha256"],
            "local_model_files_sha256": protocol["local_model_files_sha256"],
            "speed_claim_eligible": {m: overall[m]["all_parity_passed"] for m in METHODS},
            "quantile_convention": "NumPy linear sample quantile; no confidence intervals",
            "speed_ratio_convention": "Per request: median of five serial times / median of five method times",
            "limits": ["Synthetic serving workload, not labeled accuracy, calibration or training evidence.",
                       "All four methods and all 54 requests retained, including every mismatch.",
                       "Cold means allocator-cold with resident weights; no independent process-cold trials.",
                       "Memory is MLX allocator memory, not process RSS. Timings exclude filesystem recording.",
                       "Candidate softmax and choices independently reconstructed from saved label logits.",
                       "Full-vocabulary mass is range-checked and compared; full logits were not retained.",
                       "Tokenizer/transformer execution is source-bound, not independently rerun.",
                       "Model-file identities are retained from the bound protocol; weights are not reopened by this reporter."],
            "new_model_calls": 0, "new_native_calls": 0, "new_random_draws": 0}


def plot(summary, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#666666", "#4477AA", "#CC8844", "#228833")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for row, context in enumerate(("short", "long")):
        for column, questions in enumerate((1, 2, 4)):
            ax = axes[row, column]
            for i, method in enumerate(METHODS):
                cells = [summary["cells"][f"q{questions}-{context}-c{c}"][method] for c in (2, 4, 12)]
                median = [x["warm_latency_ms"]["median"] for x in cells]
                p95 = [x["warm_latency_ms"]["p95"] for x in cells]
                label = LABELS[i] + (" [MISMATCH]" if not summary["methods"][method]["all_parity_passed"] else "")
                ax.plot(range(3), median, "o-", color=colors[i], label=label, linewidth=1.8)
                ax.plot(range(3), p95, ":", color=colors[i], linewidth=1.2, alpha=.85)
            ax.set_title(f"{questions} question{'s' if questions > 1 else ''} | {context} context")
            ax.set_xticks(range(3), ["2", "4", "12"])
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
            if column == 0:
                ax.set_ylabel("Warm request latency (ms)")
            if row == 1:
                ax.set_xlabel("Candidates per question")
    flag = "ALL PARITY CHECKS PASS" if summary["all_parity_passed"] else f"MISMATCHES RETAINED: {summary['failed_parity_records']} / 1296 records"
    fig.suptitle("Shared-prefix inference: fixed synthetic workload\n" + flag, fontsize=14, y=.99)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .90), ncol=4, fontsize=9, frameon=False)
    fig.text(.5, .025, "Solid: median; dotted: p95. Each cell retains 3 requests × 5 warm repetitions.\n"
             "Full request time includes tokenization, prefix/cache work and readback. No accuracy or calibration claim.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.80, bottom=.14, hspace=.34, wspace=.26)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def report(experiment, protocol_sha256, completed_sha256, out, *, root=ROOT):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        write(out / "started.json", {"protocol_sha256": protocol_sha256, "completed_sha256": completed_sha256,
                                     "scope": "saved-output audit and plotting only", "source_sha256": sha(Path(__file__))})
        protocol, done, runtime, requests, records = authenticate(experiment, protocol_sha256, completed_sha256, root)
        summary = summarize(protocol, done, runtime, requests, records)
        summary.update(protocol_sha256=protocol_sha256, execution_completed_sha256=completed_sha256)
        write(out / "summary.json", summary)
        plot(summary, out / "warm-latency.png")
        # Recheck mutable input/source bytes after rendering before sealing output.
        require(sha(experiment / "protocol.json") == protocol_sha256
                and sha(experiment / "run-01/completed.json") == completed_sha256
                and sha(experiment / "run-01/records.jsonl") == done["records_sha256"]
                and sha(experiment / "run-01/runtime.json") == done["runtime_sha256"]
                and sha(experiment / "requests.json") == protocol["requests_sha256"]
                and all(sha(root / p) == s for p, s in protocol["source_sha256"].items()), "Inputs changed during report")
        receipt = {"status": "completed", "study": "shared-prefix-v1", "protocol_sha256": protocol_sha256,
                   "execution_completed_sha256": completed_sha256, "records_sha256": done["records_sha256"],
                   "runtime_sha256": done["runtime_sha256"], "requests_sha256": protocol["requests_sha256"],
                   "reporter_source_sha256": sha(Path(__file__)), "source_sha256": protocol["source_sha256"],
                   "records_checked": 1296, "all_parity_passed": summary["all_parity_passed"],
                   "failed_parity_records": summary["failed_parity_records"], "new_model_calls": 0,
                   "new_native_calls": 0, "new_random_draws": 0, "wall_seconds": time.perf_counter() - start,
                   "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                             for name in ("started.json", "summary.json", "warm-latency.png")}}
        write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                         "message": str(error), "wall_seconds": time.perf_counter() - start})
        except BaseException as preservation_error:  # noqa: BLE001 - preserve the original audit failure
            note = getattr(error, "add_note", None)
            if callable(note):
                note("Failure receipt could not be written: " + repr(preservation_error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.experiment, args.protocol_sha256, args.completed_sha256, args.out)
    print(json.dumps({"records_checked": result["records_checked"], "all_parity_passed": result["all_parity_passed"],
                      "receipt_sha256": sha(args.out / "receipt.json")}, allow_nan=False))
