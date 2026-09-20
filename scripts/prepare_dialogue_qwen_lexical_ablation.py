"""Authenticated local-tokenizer preparation for a fixed lexical-only ablation.

Labels are copied and hashed as opaque bytes, never decoded. Model weights are
hashed for identity only. No baseline score file, corpus or neural model is read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "dialogue-qwen-lexical-ablation-v1"
BASE_PLAN = "2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111"
BASE_COMPLETED = "6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f"
ROW_COUNT = 7819
BASE_SOURCES = {
    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "src/openjev/research/dialogue_qwen_observation.py", "scripts/prepare_dialogue_qwen_observation.py",
    "scripts/run_dialogue_qwen_observation.py", "tests/test_dialogue_qwen_observation.py",
    "tests/test_run_dialogue_qwen_observation.py", "tests/test_prepare_dialogue_qwen_observation.py",
    "research/dialogue-qwen-observation-protocol.md",
}
PROTOCOL = "research/dialogue-qwen-lexical-ablation-protocol.md"
NEW_SOURCES = {"src/openjev/research/dialogue_qwen_lexical_ablation.py",
    "scripts/prepare_dialogue_qwen_lexical_ablation.py", "tests/test_dialogue_qwen_lexical_ablation.py",
    "tests/test_prepare_dialogue_qwen_lexical_ablation.py", PROTOCOL}
LIMITS = {"wall_seconds": 180, "rss_bytes": 2*1024**3, "output_bytes": 512*1024**2}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path, check=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
            check()
    return value.hexdigest()


def item(path, check=lambda: None):
    return {"sha256": sha(path, check), "bytes": Path(path).stat().st_size}


def decode(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError("Nonfinite JSON: "+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def read(path):
    return decode(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def safe(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "Safe relative artifact path")
    return Path(root)/p


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def authenticate_baseline(args, check):
    root = Path(args.baseline).resolve()
    require(args.baseline_plan_sha256 == BASE_PLAN and args.baseline_completed_sha256 == BASE_COMPLETED,
            "Fixed baseline pins")
    require(sha(root/"plan.json", check) == BASE_PLAN and sha(root/"completed.json", check) == BASE_COMPLETED,
            "Baseline external identity")
    plan, done = read(root/"plan.json"), read(root/"completed.json")
    require(done["status"] == "completed" and done["plan_sha256"] == BASE_PLAN
            and done["model_calls"] == done["encoder_calls"] == 0 and done["tokenizer_only"] is True,
            "Successful tokenizer-only baseline preparation")
    require(set(done["files"]) == {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"}
            and {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            == set(done["files"]) | {"completed.json"}, "Exact successful baseline closure")
    bindings = {root/"completed.json": item(root/"completed.json", check)}
    for name, descriptor in done["files"].items():
        path = safe(root, name)
        require(item(path, check) == descriptor, "Baseline payload identity: "+name)
        bindings[path] = descriptor
    require(set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}
            and all(done["files"][name] == desc for name, desc in plan["files"].items()), "Baseline payload joins")
    require(set(plan["source_sha256"]) == BASE_SOURCES, "Exact inherited source closure")
    for name, digest in plan["source_sha256"].items():
        require(sha(safe(ROOT, name), check) == digest, "Frozen baseline source: "+name)
    require(plan["version"] == "dialogue-qwen-observation-v1" and plan["method"] == "batch"
            and plan["row_count"] == ROW_COUNT and plan["decisions"] == 2*ROW_COUNT, "Fixed complete baseline scope")
    require(sha(ROOT/PROTOCOL, check) == args.protocol_sha256, "External ablation protocol pin")
    return root, plan, bindings


def load_runner():
    spec = importlib.util.spec_from_file_location("frozen_qwen_ablation_runner_contract", ROOT/"scripts/run_dialogue_qwen_observation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def tokenize_record(original, tokenizer):
    from openjev.decisions import MAX_TOKENS, DecisionRequest, messages_for
    from openjev.research.dialogue_qwen_lexical_ablation import transform_request, verify_transform

    result = transform_request(original)
    request = DecisionRequest.model_validate(result["request"])
    require(request.model_dump() == result["request"], "No normalization of remaining prompt characters")
    tokens = []
    for q, mapping, ordered in zip(request.questions, result["canonical_id_maps"],
                                   result["ordered_candidate_ids"], strict=True):
        messages, candidates = messages_for(request.context, q)
        require([mapping[c.id] for c in candidates] == ordered, "Retokenized label order unchanged")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sequence = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        require(0 < len(sequence) <= MAX_TOKENS and all(type(x) is int and 0 <= x < 151936 for x in sequence),
                "Full prompt token range/length; no truncation")
        tokens.append(sequence)
    result["tokens"] = tokens
    verify_transform(original, result)
    return result


def execute(args):
    start, handler, sources = time.monotonic(), None, {}
    progress = {"requests_completed": 0, "questions_completed": 0, "tokenizer_load_attempted": False,
                "tokenizer_loaded": False, "model_calls": 0}
    out, baseline = Path(args.out).resolve(), Path(args.baseline).resolve()
    require(not out.is_relative_to(baseline) and not baseline.is_relative_to(out), "Separate output directory")
    out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) for k, v in vars(args).items()}
    def check():
        if time.monotonic()-start > LIMITS["wall_seconds"]:
            raise TimeoutError("Whole preparation wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Preparation process-lifetime RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Preparation output cap")
    def alarm(*_):
        raise TimeoutError("Whole preparation wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        handler = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        for key in THREAD_ENV:
            os.environ[key] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        write(out/"started.json", {"experiment_id": EXPERIMENT, "version": "dialogue-qwen-observation-preparation-v1",
            "request": request, "limits": LIMITS, "model_calls": 0, "labels_decoded": False, "cpu_threads": 1})
        baseline, parent, bindings = authenticate_baseline(args, check)
        sources = {name: sha(ROOT/name, check) for name in sorted(BASE_SOURCES | NEW_SOURCES)}
        runner = load_runner()  # Source was authenticated before any project imports.
        require(parent["runtime"] == runner.runtime() and parent["limits"] == runner.LIMITS, "Frozen runtime/limits")
        model = parent["model"]
        require(model["id"] == runner.MODEL_ID and model["revision"] == runner.REVISION
                and set(model["files_sha256"]) == runner.MODEL_FILES, "Frozen local model identity")
        snapshot = Path(model["snapshot_path"])
        require(snapshot.name == runner.REVISION, "Pinned snapshot directory")
        for name, digest in model["files_sha256"].items():
            path = safe(snapshot, name)
            require(sha(path, check) == digest, "Model/tokenizer file identity: "+name)
            bindings[path] = {"sha256": digest, "bytes": path.stat().st_size}
        progress["tokenizer_load_attempted"] = True
        tokenizer = load_tokenizer(snapshot)
        progress["tokenizer_loaded"] = True
        from openjev.decisions import LABELS
        from openjev.research.dialogue_qwen_observation import pilot_ids

        label_ids = [tokenizer.encode(label, add_special_tokens=False, truncation=False) for label in LABELS]
        require(label_ids == [[x] for x in runner.LABEL_IDS], "Unchanged unique label token IDs")
        records, old_slots, old_counts = [], dict.fromkeys(runner.ARMS, 0), dict.fromkeys(runner.ARMS, 0)
        with (baseline/"requests.jsonl").open() as stream, (out/"requests.jsonl").open("x") as target:
            for line in stream:
                original = decode(line)
                old_slots[original["arm"]] += runner.work_for(original)["input_token_slots"]
                old_counts[original["arm"]] += 1
                require(original["label_ids"] == runner.LABEL_IDS, "Baseline token-label binding")
                record = tokenize_record(original, tokenizer)
                target.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n")
                records.append(record)
                progress["requests_completed"] += 1
                progress["questions_completed"] += len(record["row_indices"])
                check()
        require(old_slots == parent["input_token_slots"] and old_counts == parent["request_counts"], "Inherited complete workload")
        shutil.copyfile(baseline/"labels.jsonl", out/"labels.jsonl")  # Deliberately opaque, including no row counting.
        require(item(out/"labels.jsonl", check) == parent["files"]["labels.jsonl"], "Byte-identical opaque labels")
        plan = copy.deepcopy(parent)
        plan.update(experiment_id=EXPERIMENT, source_sha256=sources, protocol_sha256=args.protocol_sha256,
            parent={"prepared_path": str(baseline), "plan_sha256": BASE_PLAN, "completed_sha256": BASE_COMPLETED,
                    "files": parent["files"], "source_sha256": parent["source_sha256"]},
            transformation="Remove only lexical candidate suffixes, exact TASK explanation and trailing flag-order legend.",
            files={name: item(out/name, check) for name in ("requests.jsonl", "labels.jsonl")},
            pilot_request_ids=pilot_ids(records),
            input_token_slots={a: sum(runner.work_for(r)["input_token_slots"] for r in records if r["arm"] == a) for a in runner.ARMS},
            request_counts={a: sum(r["arm"] == a for r in records) for a in runner.ARMS},
            prompt_tokens={a: {"min": min(len(t) for r in records if r["arm"] == a for t in r["tokens"]),
                              "max": max(len(t) for r in records if r["arm"] == a for t in r["tokens"]),
                              "sum": sum(len(t) for r in records if r["arm"] == a for t in r["tokens"])} for a in runner.ARMS})
        runner.validate_requests(records, plan)
        require(plan["request_counts"] == parent["request_counts"]
                and progress["questions_completed"] == parent["decisions"], "Unchanged full question/request membership")
        write(out/"plan.json", plan)
        for path, descriptor in bindings.items():
            require(item(path, check) == descriptor, "End inherited/model identity")
        require({name: sha(ROOT/name, check) for name in sources} == sources, "End source identity")
        files = {p.name: item(p, check) for p in out.iterdir() if p.is_file()}
        write(out/"completed.json", {"status": "completed", "experiment_id": EXPERIMENT,
            "model_calls": 0, "encoder_calls": 0, "tokenizer_only": True, "labels_decoded": False,
            "checkpoint_deserializations": 0, "baseline_scores_accessed": False, "official_dev_dialogues_accessed": False,
            "test_contents_accessed": False, "plan_sha256": sha(out/"plan.json", check), "files": files,
            "parent_plan_sha256": BASE_PLAN, "parent_completed_sha256": BASE_COMPLETED, "source_sha256": sources,
            "limits": LIMITS, "progress": progress, "cpu_threads": 1, "wall_seconds": time.monotonic()-start,
            "peak_rss_bytes": peak_rss(), "wall_scope": "Authentication, model-byte hashing, local tokenizer load, "
                "complete retokenization, opaque labels copy and final payload hashes; terminal write/return cap checked."})
        check()
        return plan
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"completed.json").exists():
                (out/"completed.json").rename(out/"late-completion.json")
            write(out/"failed.json", {"status": "failed", "experiment_id": EXPERIMENT, "request": request,
                "source_sha256": sources, "progress": progress, "error": repr(error), "model_calls": 0,
                "labels_decoded": False, "wall_seconds": time.monotonic()-start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "baseline-plan-sha256", "baseline-completed-sha256", "protocol-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
