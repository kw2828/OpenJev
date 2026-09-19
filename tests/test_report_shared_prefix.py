"""Pure synthetic saved-record tests; no tokenizer, model, cache or device calls."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/report_shared_prefix.py"
SPEC = importlib.util.spec_from_file_location("report_shared_prefix", PATH)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def fixture():
    requests, records = [], []
    for q in (1, 2, 4):
        for context in ("short", "long"):
            for count in (2, 4, 12):
                for case in range(3):
                    tokens = [[7, 8] + [10 + j] * (j + 2) for j in range(q)]
                    row = {"id": f"q{q}-{context}-c{count}-case{case}", "question_count": q,
                           "context_size": context, "candidate_count": count, "case": case,
                           "tokens": tokens, "prefix_tokens": len(tokens[0]) - 1 if q == 1 else 2,
                           "request": {"context": "Clearly synthetic fixture", "questions": [
                               {"id": f"field{j}", "question": "Synthetic question", "candidates": [
                                   {"id": f"c{k}", "description": f"Option{k:02d}"} for k in range(count)]}
                               for j in range(q)]}}
                    index = len(requests)
                    requests.append(row)
                    sequence = [("allocator_cold", 0, m) for m in reporter.METHODS]
                    sequence += [("warm", r, reporter.METHODS[(index + r + j) % 4])
                                 for r in range(5) for j in range(4)]
                    reference = None
                    for phase, repetition, method in sequence:
                        result = {"request_id": row["id"], "phase": phase, "repetition": repetition,
                                  "method": method, "answers": [
                                      {"id": f"field{j}", "choice": "c0",
                                       "probabilities": {f"c{k}": 1 / count for k in range(count)},
                                       "candidate_token_mass": .5, "entropy_nats": math.log(count),
                                       "input_tokens": len(tokens[j])} for j in range(q)],
                                  "candidate_logits": [[0.] * count for _ in range(q)],
                                  "latency_ms": (10 if method == "serial" else 5) + repetition,
                                  "peak_active_bytes": 1100, "baseline_active_bytes": 1000,
                                  "cached_allocator_bytes": 100, "work": reporter.expected_work(tokens, method),
                                  "questions_sequential": method in ("serial", "shared_serial")}
                        if phase == "allocator_cold" and method == "serial":
                            reference = result
                        result["parity"] = reporter.parity(reference, result)
                        records.append(result)
    return requests, records


@pytest.fixture(scope="module")
def saved():
    return fixture()


def test_exact_work_counts_and_zero_prefix():
    tokens = [[1, 2, 3, 4], [1, 2, 8, 9, 10, 11]]
    assert reporter.expected_work(tokens, "serial") == {
        "calls": 2, "prefill_calls": 0, "branch_calls": 2, "input_token_slots": 10,
        "prefix_tokens": 0, "padding_token_slots": 0}
    assert reporter.expected_work(tokens, "batch")["input_token_slots"] == 12
    assert reporter.expected_work(tokens, "shared_serial")["input_token_slots"] == 8
    assert reporter.expected_work(tokens, "shared_batch") == {
        "calls": 2, "prefill_calls": 1, "branch_calls": 1, "input_token_slots": 10,
        "prefix_tokens": 2, "padding_token_slots": 2}
    assert reporter.expected_work([[1, 2], [3, 4, 5]], "shared_batch")["prefill_calls"] == 0
    assert reporter.expected_work([[1, 2, 3]], "shared_serial")["calls"] == 2


def test_complete_coverage_and_paired_request_median_arithmetic(saved):
    requests, records = saved
    checked = reporter.audit_records(requests, records)
    summary = reporter.summarize({"source_sha256": {}, "local_model_files_sha256": {}},
                                {"all_parity_passed": True, "parity_failed_rows": 0, "wall_seconds": 1},
                                {}, requests, checked)
    assert len(summary["cells"]) == 18
    assert summary["methods"]["serial"]["warm_latency_ms"]["count"] == 270
    assert summary["methods"]["batch"]["speed_ratio_distribution"]["median"] == 12 / 7
    assert summary["cells"]["q4-long-c12"]["batch"]["warm_latency_ms"]["count"] == 15
    assert summary["new_model_calls"] == 0


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "order", "question", "candidate",
                                  "nan", "sum", "softmax", "choice", "tokens", "work", "memory", "parity"])
def test_corrupt_records_rejected(saved, kind):
    requests, original = saved
    records = copy.deepcopy(original)
    first = records[0]
    if kind == "missing":
        records.pop()
    elif kind == "duplicate":
        records[-1] = copy.deepcopy(first)
    elif kind == "foreign":
        first["request_id"] = "other"
    elif kind == "order":
        records[1], records[2] = records[2], records[1]
    elif kind == "question":
        first["answers"][0]["id"] = "other"
    elif kind == "candidate":
        first["answers"][0]["probabilities"]["other"] = 0.
    elif kind == "nan":
        first["latency_ms"] = float("nan")
    elif kind == "sum":
        first["answers"][0]["probabilities"]["c0"] = .6
    elif kind == "softmax":
        first["candidate_logits"][0][0] = 1.
    elif kind == "choice":
        first["answers"][0]["choice"] = "c1"
    elif kind == "tokens":
        first["answers"][0]["input_tokens"] += 1
    elif kind == "work":
        first["work"]["padding_token_slots"] += 1
    elif kind == "memory":
        first["peak_active_bytes"] = 0
    elif kind == "parity":
        first["parity"]["max_mass_difference"] = .01
    with pytest.raises(ValueError):
        reporter.audit_records(requests, records)


def test_mismatch_is_retained_not_dropped(saved):
    requests, original = saved
    records = copy.deepcopy(original)
    records[1]["answers"][0]["candidate_token_mass"] = .51
    records[1]["parity"] = reporter.parity(records[0], records[1])
    checked = reporter.audit_records(requests, records)
    assert len(checked) == 1296 and not checked[1]["parity"]["pass"]
    group = reporter.aggregate(checked, requests, {r["id"] for r in requests})
    assert not group["batch"]["all_parity_passed"]
    assert len(group["batch"]["failed_records"]) == 1
    assert group["batch"]["warm_latency_ms"]["count"] == 270


def test_parity_inclusive_boundary_without_epsilon():
    left = {"answers": [{"choice": "a", "probabilities": {"a": 1.}, "candidate_token_mass": 0.}],
            "candidate_logits": [[0.]]}
    right = copy.deepcopy(left)
    right["answers"][0]["candidate_token_mass"] = .005
    assert reporter.parity(left, right)["pass"]
    right["answers"][0]["candidate_token_mass"] = math.nextafter(.005, math.inf)
    assert not reporter.parity(left, right)["pass"]


def tree(tmp_path, saved):
    requests, records = saved
    root = tmp_path / "repository"
    experiment = root / "experiment"
    run = experiment / "run-01"
    run.mkdir(parents=True)
    for source in reporter.SOURCES:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source placeholder\n")
    reporter.write(experiment / "requests.json", requests)
    protocol = {"version": "shared-prefix-v1", "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                "revision": "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b", "methods": list(reporter.METHODS),
                "prompt_protocol_sha256": "0" * 64, "request_count": 54, "warm_repetitions": 5,
                "allocator_cold_calls": 216, "warm_calls": 1080,
                "parity": {"choice_mismatches": 0, "max_probability_difference": .005,
                           "max_candidate_mass_difference": .005},
                "source_sha256": {p: reporter.sha(root / p) for p in reporter.SOURCES},
                "local_model_files_sha256": {p: "0" * 64 for p in ("config.json", "tokenizer.json", "model.safetensors")},
                "requests_sha256": reporter.sha(experiment / "requests.json")}
    reporter.write(experiment / "protocol.json", protocol)
    protocol_sha = reporter.sha(experiment / "protocol.json")
    reporter.write(run / "runtime.json", {"model_load_ms": 1., "python": "synthetic", "platform": "synthetic",
                                          "device": "synthetic", "hardware": "synthetic",
                                          "versions": dict.fromkeys(("mlx", "mlx-lm", "numpy"), "synthetic"),
                                          "protocol_sha256": protocol_sha})
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    reporter.write(run / "completed.json", {"status": "complete", "records": 1296, "parity_failed_rows": 0,
                                            "all_parity_passed": True, "wall_seconds": 1.,
                                            "protocol_sha256": protocol_sha,
                                            "records_sha256": reporter.sha(run / "records.jsonl"),
                                            "runtime_sha256": reporter.sha(run / "runtime.json")})
    return root, experiment, protocol_sha, reporter.sha(run / "completed.json")


def test_exclusive_full_saved_report_and_figure(tmp_path, saved):
    root, experiment, protocol_sha, done_sha = tree(tmp_path, saved)
    out = tmp_path / "synthetic-report"
    result = reporter.report(experiment, protocol_sha, done_sha, out, root=root)
    assert result["records_checked"] == 1296
    assert result["all_parity_passed"]
    assert (out / "warm-latency.png").read_bytes().startswith(b"\x89PNG")
    assert all(reporter.sha(out / name) == item["sha256"] for name, item in result["files"].items())
    with pytest.raises(FileExistsError):
        reporter.report(experiment, protocol_sha, done_sha, out, root=root)


@pytest.mark.parametrize("kind", ["source", "records", "runtime", "failure_marker", "completion"])
def test_hash_and_failure_authentication_preserves_failed_report(tmp_path, saved, kind):
    root, experiment, protocol_sha, done_sha = tree(tmp_path, saved)
    if kind == "source":
        (root / "src/openjev/decisions.py").write_text("changed")
    elif kind in ("records", "runtime"):
        (experiment / "run-01" / ("records.jsonl" if kind == "records" else "runtime.json")).write_text("changed")
    elif kind == "failure_marker":
        (experiment / "run-01/failed.json").write_text("{}")
    else:
        done_sha = "f" * 64
    out = tmp_path / "failed-report"
    with pytest.raises(ValueError):
        reporter.report(experiment, protocol_sha, done_sha, out, root=root)
    assert (out / "failed.json").is_file()
    assert not (out / "receipt.json").exists()
