"""Freeze and run a bounded 54-request, four-method inference comparison."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from openjev.decisions import MODEL_ID, MODEL_REVISION, PROTOCOL_HASH, DecisionRequest, MLXScorer
from openjev.research.shared_prefix import METHODS, SharedPrefixExperiment, common_prefix_length

ROOT = Path(__file__).resolve().parents[1]
BOUND_FILES = (
    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "scripts/benchmark_shared_prefix.py", "tests/test_shared_prefix.py",
    "tests/test_benchmark_shared_prefix.py", "research/shared-prefix-protocol.md",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def requests():
    pools = [
        ["Billing", "Technical support", "Account access", "Security", "Shipping", "Returns",
         "Sales", "Legal", "Procurement", "Partnerships", "Product feedback", "Human resources"],
        ["London", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto", "Seattle", "Madrid",
         "Rome", "Dublin", "Boston", "Singapore"],
        ["Monthly invoice", "Annual invoice", "Password reset", "Shipment tracking",
         "Refund request", "Subscription renewal", "Account closure", "Product repair",
         "Contract review", "Purchase order", "Data export", "Service outage"],
        ["Email reply", "Telephone call", "Video call", "Support portal", "Postal letter",
         "In-person visit", "Text message", "Chat session", "Status page", "No reply requested",
         "Scheduled meeting", "Callback queue"],
    ]
    prompts = ["Which team is responsible for the current ticket?",
               "In which city is this customer located? Use the current ticket rather than earlier records.",
               "What is the subject of the current ticket?",
               "Which contact channel did the customer request for the response to this ticket?"]
    rows = []
    for question_count in (1, 2, 4):
        for context_size in ("short", "long"):
            for candidate_count in (2, 4, 12):
                for case in range(3):
                    index = case % candidate_count
                    current = (f"Current ticket {701 + case}: team {pools[0][index]}; customer city "
                               f"{pools[1][index]}; subject {pools[2][index]}; requested response "
                               f"channel {pools[3][index]}. The customer asks for a response today.")
                    history = ""
                    if context_size == "long":
                        history = "Historical records, separate from the current ticket:\n" + "\n".join(
                            f"Record {i}: the {pools[0][i % 12]} team in {pools[1][i % 12]} "
                            f"handled {pools[2][i % 12].lower()} by {pools[3][i % 12].lower()}. Closed."
                            for i in range(24)
                        ) + "\nEnd historical records.\n"
                    request = {"context": history + current, "questions": [
                        {"id": f"field{j}", "question": prompts[j], "candidates": [
                            {"id": f"c{k}", "description": value}
                            for k, value in enumerate(pools[j][:candidate_count])
                        ]} for j in range(question_count)
                    ]}
                    DecisionRequest.model_validate(request)
                    rows.append({"id": f"q{question_count}-{context_size}-c{candidate_count}-case{case}",
                                 "question_count": question_count, "context_size": context_size,
                                 "candidate_count": candidate_count, "case": case, "request": request})
    return rows


def compare(reference, actual):
    if len(reference["answers"]) != len(actual["answers"]):
        raise ValueError("Answer count changed")
    choices = 0
    max_probability = max_mass = max_logit = 0.
    for left, right, ll, rl in zip(reference["answers"], actual["answers"],
                                  reference["candidate_logits"], actual["candidate_logits"]):
        if left["id"] != right["id"] or left["probabilities"].keys() != right["probabilities"].keys():
            raise ValueError("Candidate or question identity changed")
        choices += left["choice"] != right["choice"]
        max_probability = max(max_probability, max(abs(p - right["probabilities"][key])
                                                   for key, p in left["probabilities"].items()))
        max_mass = max(max_mass, abs(left["candidate_token_mass"] - right["candidate_token_mass"]))
        max_logit = max(max_logit, float(np.max(np.abs(np.array(ll) - np.array(rl)))))
    return {"choice_mismatches": choices, "max_probability_difference": max_probability,
            "max_mass_difference": max_mass, "max_candidate_logit_difference": max_logit,
            "pass": choices == 0 and max_probability <= .005 and max_mass <= .005}


def prepare(output):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    experiment = SharedPrefixExperiment.__new__(SharedPrefixExperiment)
    experiment.scorer = type("TokenizerOnly", (), {"tokenizer": tokenizer})()
    rows = requests()
    for row in rows:
        prepared = experiment.prepare(DecisionRequest.model_validate(row["request"]))
        row["tokens"] = [entry[2] for entry in prepared]
        row["prefix_tokens"] = common_prefix_length(row["tokens"])
    write(output / "requests.json", rows)
    bindings = {name: sha(ROOT / name) for name in BOUND_FILES}
    weights = {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()}
    write(output / "protocol.json", {
        "version": "shared-prefix-v1", "model": MODEL_ID, "revision": MODEL_REVISION,
        "prompt_protocol_sha256": PROTOCOL_HASH, "methods": list(METHODS),
        "requests_sha256": sha(output / "requests.json"), "request_count": 54,
        "warm_repetitions": 5, "allocator_cold_calls": 216, "warm_calls": 1080,
        "parity": {"choice_mismatches": 0, "max_probability_difference": .005,
                   "max_candidate_mass_difference": .005,
                   "candidate_logits": "record differences, no separate raw-logit acceptance threshold"},
        "source_sha256": bindings, "local_model_files_sha256": weights,
        "interpretation": "Synthetic serving workload; not labeled quality, calibration or training evidence",
    })
    print(json.dumps({"prepared": len(rows), "protocol_sha256": sha(output / "protocol.json"),
                      "min_tokens": min(len(t) for r in rows for t in r["tokens"]),
                      "max_tokens": max(len(t) for r in rows for t in r["tokens"])}), flush=True)


def run(output):
    protocol = json.loads((output / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if sha(output / "requests.json") != protocol["requests_sha256"]:
        raise ValueError("Frozen requests changed")
    run_dir = output / "run-01"
    run_dir.mkdir(exist_ok=False)
    start = time.perf_counter()
    scorer = MLXScorer()
    scorer.mx.eval(scorer.model.parameters())
    scorer.mx.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    experiment = SharedPrefixExperiment(scorer)
    rows = json.loads((output / "requests.json").read_text())
    for row in rows:
        actual = [entry[2] for entry in experiment.prepare(DecisionRequest.model_validate(row["request"]))]
        if actual != row["tokens"]:
            raise ValueError("Tokenizer differs from frozen full prompts")
    write(run_dir / "runtime.json", {
        "model_load_ms": load_ms, "python": platform.python_version(),
        "platform": platform.platform(), "device": str(scorer.mx.default_device()),
        "hardware": subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                             text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
        "protocol_sha256": sha(output / "protocol.json"),
    })
    record_count = 0
    failures = 0
    with (run_dir / "records.jsonl").open("x") as handle:
        for i, row in enumerate(rows):
            request = DecisionRequest.model_validate(row["request"])
            reference = None
            cold = {}
            # First calls have an empty allocator cache, with model still resident.
            # Rotate method order to reduce systematic temporal-order confounding.
            for j in range(4):
                method = METHODS[(i + j) % 4]
                scorer.mx.synchronize()
                scorer.mx.clear_cache()
                cold[method] = experiment.score(request, method)
            reference = cold["serial"]
            for method in METHODS:
                result = cold[method]
                parity = compare(reference, result)
                handle.write(json.dumps({"request_id": row["id"], "phase": "allocator_cold",
                                         "repetition": 0, "method": method, **result, "parity": parity},
                                        allow_nan=False) + "\n")
                failures += not parity["pass"]
                record_count += 1
            # Clear-cache first calls warm all four method/shape combinations.
            for repetition in range(5):
                for j in range(4):
                    method = METHODS[(i + repetition + j) % 4]
                    result = experiment.score(request, method)
                    parity = compare(reference, result)
                    handle.write(json.dumps({"request_id": row["id"], "phase": "warm",
                                             "repetition": repetition, "method": method,
                                             **result, "parity": parity}, allow_nan=False) + "\n")
                    failures += not parity["pass"]
                    record_count += 1
                handle.flush()
            print(json.dumps({"request": i + 1, "of": len(rows), "id": row["id"],
                              "records": record_count, "parity_failed_rows": failures}), flush=True)
    write(run_dir / "completed.json", {
        "status": "complete", "records": record_count, "parity_failed_rows": failures,
        "all_parity_passed": failures == 0, "wall_seconds": time.perf_counter() - start,
        "protocol_sha256": sha(output / "protocol.json"),
        "records_sha256": sha(run_dir / "records.jsonl"), "runtime_sha256": sha(run_dir / "runtime.json"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "output/shared-prefix-v1")
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.output)
