"""Single independent saved-input audit; standard library only, no label decoding."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import signal
import sys
import time
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "output/dialogue-qwen-observation-v1/preparation-02"
NEW = ROOT / "output/dialogue-qwen-lexical-ablation-v1/preparation-01"
PINS = {
    "base_plan": "2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111",
    "base_completed": "6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f",
    "new_plan": "89b90d4a7decddf35e2dfbfacd53ce15a61e455cb32be9a719092839367caa36",
    "new_completed": "868f8306f98091e192a6d88a9f2660fdf22f89c18165471162cd6746002dea7f",
}
EXPERIMENT = "dialogue-qwen-lexical-ablation-v1"
LIMITS = {"wall_seconds": 60, "rss_bytes": 2 * 1024**3, "output_bytes": 32 * 1024**2}
BASE_SOURCES = {
    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "src/openjev/research/dialogue_qwen_observation.py", "scripts/prepare_dialogue_qwen_observation.py",
    "scripts/run_dialogue_qwen_observation.py", "tests/test_dialogue_qwen_observation.py",
    "tests/test_run_dialogue_qwen_observation.py", "tests/test_prepare_dialogue_qwen_observation.py",
    "research/dialogue-qwen-observation-protocol.md",
}
ADDED = {
    "src/openjev/research/dialogue_qwen_lexical_ablation.py": "6b9477a212e71f786398df2510449ad5ca0bbc898bd7b1ec1a1cdbba33f15484",
    "scripts/prepare_dialogue_qwen_lexical_ablation.py": "3adbf8406807841d47aab218f594f29638b9cb350a34ed165228689e7ea6c3cc",
    "tests/test_dialogue_qwen_lexical_ablation.py": "4d49de6b990bb8e8303cbc7130b2ac9d7d57820e720477969b743bb5080abf04",
    "tests/test_prepare_dialogue_qwen_lexical_ablation.py": "0907689da5f58c5dfe5502abefa545b897f092698bfae8155524c92245937fc2",
    "research/dialogue-qwen-lexical-ablation-protocol.md": "c04a1a0d77bedd028465f954de4261828dd23a75b83ea069da173455f787a329",
}
PREFIX = (
    "What is the user's currently committed value for this slot after the final USER turn? "
    "Use the supplied previous value as the state immediately before that turn. "
    "Update it only when the dialogue supports a change; a SYSTEM proposal alone is not "
    "a user commitment. NOT_MENTIONED means no constraint has been stated; DONTCARE "
    "means the user explicitly has no preference. Literal ontology values, including "
    "a value named None, remain distinct from these reserved states. "
)
EXPLANATION = (
    "Lexical flags are noisy public string-match observations, not answers. "
    "The literal register carries the last unique longest USER mention across the "
    "public prefix; it does not understand negation or relevance. "
)
LEGEND = "\nLexical flag order: user_match,system_match,unique_longest_user,unique_longest_system,literal_current,literal_previous,is_none,is_dontcare,affirmative_cue_for_true,negative_cue_for_false"
SCOPE = (
    "Independent standard-library reconstruction of the permitted paired string subtraction, "
    "all request/candidate mappings, opaque label-byte identity and token-array workload/pilot selection. "
    "No project imports, label decoding, score access, tokenizer, model or checkpoint calls. "
    "Token IDs are checked for structure/range/workload, not retokenized; tokenizer correctness and "
    "original public chronology/previous-value provenance inherit the authenticated preparation chain. "
    "Model/runtime/input descriptors and source files are checked, but model weights and earlier corpus "
    "payloads are not rehashed or loaded. This is paired-input verification, not cost-pilot admission or quality evidence."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
            check()
    return h.hexdigest()


def descriptor(path, check=lambda: None):
    return {"sha256": sha(path, check), "bytes": path.stat().st_size}


def run():
    started = time.monotonic()
    progress = {"authenticated_files": 0, "requests": 0, "questions": 0, "candidates": 0}
    handler = None
    bindings = {}
    def check():
        require(time.monotonic() - started <= LIMITS["wall_seconds"], "Audit wall cap")
        require(rss() <= LIMITS["rss_bytes"], "Audit RSS cap")
        require(sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")
    def expired(*_):
        raise TimeoutError("Audit wall cap")
    def bind(path, expected):
        actual = descriptor(path, check)
        require(actual == expected, "Hash/size mismatch: " + str(path))
        bindings[str(path)] = actual
        progress["authenticated_files"] += 1
    try:
        require({p.name for p in OUT.iterdir()} == {"audit.py"}, "Exclusive fresh audit directory")
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing timer")
        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
            os.environ[name] = "1"
        source_sha = sha(Path(__file__), check)
        write(OUT / "started.json", {"status": "started", "pins": PINS, "source_sha256": source_sha,
              "scope": SCOPE, "limits": LIMITS, "cpu_threads": 1, "command": [sys.executable, *sys.argv]})
        docs = []
        # Authenticate every payload from both manifests before decoding any request.
        for prefix, directory in (("base", BASE), ("new", NEW)):
            for name, key in (("plan.json", "plan"), ("completed.json", "completed")):
                require(sha(directory / name, check) == PINS[prefix + "_" + key], "External " + prefix + " " + key)
            plan, done = (decode((directory / name).read_bytes()) for name in ("plan.json", "completed.json"))
            require(done["status"] == "completed" and done["plan_sha256"] == PINS[prefix + "_plan"]
                    and done["model_calls"] == done["encoder_calls"] == 0 and done["tokenizer_only"] is True,
                    "Successful tokenizer-only preparation")
            expected = {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"}
            require(set(done["files"]) == expected and {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
                    == expected | {"completed.json"}, "Exact successful closure")
            for name, value in done["files"].items():
                bind(directory / name, value)
            bind(directory / "completed.json", descriptor(directory / "completed.json", check))
            require(set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}
                    and all(done["files"][k] == v for k, v in plan["files"].items()), "Plan manifest join")
            docs.append((plan, done))
        (old, _old_done), (new, done) = docs
        require(set(old["source_sha256"]) == BASE_SOURCES and set(new["source_sha256"]) == BASE_SOURCES | set(ADDED), "Exact source closures")
        require(all(new["source_sha256"][k] == v for k, v in old["source_sha256"].items())
                and all(new["source_sha256"][k] == v for k, v in ADDED.items()), "Preserved/appended source pins")
        for name, digest in new["source_sha256"].items():
            require(sha(ROOT / name, check) == digest, "Frozen source identity: " + name)
            bindings[str(ROOT / name)] = descriptor(ROOT / name, check)
        require(new["experiment_id"] == done["experiment_id"] == EXPERIMENT
                and new["protocol_sha256"] == ADDED["research/dialogue-qwen-lexical-ablation-protocol.md"], "New experiment identity")
        require(new["parent"] == {"prepared_path": str(BASE.resolve()), "plan_sha256": PINS["base_plan"],
                "completed_sha256": PINS["base_completed"], "files": old["files"], "source_sha256": old["source_sha256"]}, "Exact parent identity")
        modified = {"source_sha256", "files", "pilot_request_ids", "input_token_slots", "request_counts", "prompt_tokens"}
        added = {"experiment_id", "protocol_sha256", "parent", "transformation"}
        require(set(new) == set(old) | added and all(new[k] == v for k, v in old.items() if k not in modified), "Only declared plan changes")
        require(old["version"] == "dialogue-qwen-observation-v1" and old["method"] == "batch"
                and old["row_count"] == 7819 and old["decisions"] == 15638, "Fixed workload")
        require(old["model"]["id"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
                and old["model"]["revision"] == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
                and len(old["model"]["files_sha256"]) == 11, "Inherited model metadata")
        require(old["limits"] == {phase: {"wall_seconds": seconds, "rss_bytes": 12*1024**3, "output_bytes": 512*1024**2}
                for phase, seconds in (("pilot", 300), ("run", 7200))}, "Runner allocation")
        require(done["labels_decoded"] is False and done["baseline_scores_accessed"] is False
                and done["checkpoint_deserializations"] == 0 and done["cpu_threads"] == 1, "New preparation scope")
        require(new["files"]["labels.jsonl"] == old["files"]["labels.jsonl"], "Opaque label descriptor identity")
        with (BASE / "labels.jsonl").open("rb") as a, (NEW / "labels.jsonl").open("rb") as b:
            while True:
                aa, bb = a.read(1024**2), b.read(1024**2)
                require(aa == bb, "Opaque label-byte equality")
                check()
                if not aa:
                    break
        stats = {side: {arm: {"requests": 0, "questions": 0, "slots": 0, "tokens": 0, "min": 4097, "max": 0}
                        for arm in ("current", "history4")} for side in ("old", "new")}
        indices = {arm: set() for arm in ("current", "history4")}
        request_ids, compact = set(), []
        with (BASE / "requests.jsonl").open() as a, (NEW / "requests.jsonl").open() as b:
            for old_line, new_line in zip_longest(a, b):
                require(old_line is not None and new_line is not None, "Complete paired request order")
                left, right = decode(old_line), decode(new_line)
                require(set(left) == set(right) and {k: v for k, v in left.items() if k not in ("request", "tokens")}
                        == {k: v for k, v in right.items() if k not in ("request", "tokens")}, "Unchanged request metadata")
                arm = right["arm"]
                require(arm in indices and right["request_id"] not in request_ids and right["label_ids"] == list(range(32, 44)), "Arm/request/label identity")
                request_ids.add(right["request_id"])
                lr, rr = left["request"], right["request"]
                require(set(lr) == set(rr) == {"context", "questions"} and lr["context"] == rr["context"], "Unchanged public context")
                require(0 < len(rr["context"]) <= 12000, "Context length")
                n = len(rr["questions"])
                require(1 <= n <= 4 and len(lr["questions"]) == len(right["row_indices"]) == len(right["canonical_id_maps"])
                        == len(right["ordered_candidate_ids"]) == n, "Question routing coverage")
                for before, after, mapping, order, index in zip(lr["questions"], rr["questions"], right["canonical_id_maps"], right["ordered_candidate_ids"], right["row_indices"], strict=True):
                    require(type(index) is int and index >= 0 and index not in indices[arm], "Unique native row index")
                    indices[arm].add(index)
                    require(set(before) == set(after) == {"id", "question", "candidates"} and before["id"] == after["id"] == f"r{index}", "Question identity")
                    q = after["question"]
                    require(q.startswith(PREFIX + "\n") and before["question"] == PREFIX + EXPLANATION + q[len(PREFIX):] + LEGEND,
                            "Exact inverse question subtraction")
                    require(0 < len(q) <= 2000 and len(before["candidates"]) == len(after["candidates"]), "Question/candidate geometry")
                    ids = [c["id"] for c in after["candidates"]]
                    require(2 <= len(ids) <= 12 and len(set(ids)) == len(ids) and set(ids) == set(mapping)
                            and len(set(mapping.values())) == len(ids), "Candidate bijection")
                    for bc, ac in zip(before["candidates"], after["candidates"], strict=True):
                        require(set(bc) == set(ac) == {"id", "description"} and bc["id"] == ac["id"], "Candidate identity")
                        require(bc["description"].startswith(ac["description"]), "Unchanged candidate prefix")
                        tail = bc["description"][len(ac["description"]):]
                        require(re.fullmatch(r"\nPublic lexical flags: [01](?:,[01]){9}", tail) is not None, "Only exact flag suffix removed")
                        cid = mapping[ac["id"]]
                        kind = "NOT_MENTIONED" if cid == "reserved:NOT_MENTIONED" else "DONTCARE" if cid == "reserved:DONTCARE" else cid[6:].upper() if cid.startswith("value:") and cid[6:].casefold() in ("true", "false") else "OTHER"
                        require(ac["description"].endswith("\nCandidate type: " + kind) and 0 < len(ac["description"]) <= 1000, "Preserved candidate type/size")
                        progress["candidates"] += 1
                    require(order == [mapping[c["id"]] for c in sorted(before["candidates"], key=lambda c: c["description"])]
                            == [mapping[c["id"]] for c in sorted(after["candidates"], key=lambda c: c["description"])], "Unchanged description-to-label order")
                maxima = {}
                for side, row in (("old", left), ("new", right)):
                    require(len(row["tokens"]) == n, "Token question count")
                    lengths = []
                    for ts in row["tokens"]:
                        require(isinstance(ts, list) and 0 < len(ts) <= 4096 and all(type(t) is int and 0 <= t < 151936 for t in ts), "Integral full token arrays")
                        lengths.append(len(ts))
                    st = stats[side][arm]
                    st["requests"] += 1
                    st["questions"] += n
                    st["slots"] += n * max(lengths)
                    st["tokens"] += sum(lengths)
                    st["min"], st["max"] = min(st["min"], min(lengths)), max(st["max"], max(lengths))
                    maxima[side] = max(lengths)
                compact.append((right["request_id"], arm, right["dialogue_id"], right["time"], maxima))
                progress["requests"] += 1
                progress["questions"] += n
                check()
        require(progress["requests"] == 7444 and progress["questions"] == 15638
                and indices["current"] == indices["history4"] and len(indices["current"]) == 7819, "Complete unchanged two-arm cohort")
        groups = sorted({(r[2], r[3]) for r in compact})
        pilot_counts = {}
        for side, plan in (("old", old), ("new", new)):
            for arm, st in stats[side].items():
                require(plan["request_counts"][arm] == st["requests"] and plan["input_token_slots"][arm] == st["slots"]
                        and plan["prompt_tokens"][arm] == {"min": st["min"], "max": st["max"], "sum": st["tokens"]}, "Complete token workload agreement")
            selected = set(groups[:12])
            for arm in ("current", "history4"):
                longest = min((r for r in compact if r[1] == arm), key=lambda r: (-r[4][side], r[2], r[3], r[0]))
                selected.add((longest[2], longest[3]))
            expected = [r[0] for r in compact if (r[2], r[3]) in selected]
            require(plan["pilot_request_ids"] == expected, "Exact first12/longest union and original request order")
            pilot_counts[side] = {"groups": len(selected), "requests": len(expected)}
        require(done["progress"]["requests_completed"] == progress["requests"] and done["progress"]["questions_completed"] == progress["questions"], "Preparation recorded coverage")
        require(done["limits"] == {"wall_seconds": 180, "rss_bytes": 2*1024**3, "output_bytes": 512*1024**2}
                and math.isfinite(done["wall_seconds"]) and 0 < done["wall_seconds"] <= 180
                and 0 < done["peak_rss_bytes"] <= 2*1024**3, "Preparation resource witness")
        for path, value in bindings.items():
            require(descriptor(Path(path), check) == value, "Closing input/source identity")
        require(sha(Path(__file__), check) == source_sha, "Audit source identity")
        result = {"status": "completed", "agreement": True, "paired_inputs_clear_for_pilot": True,
            "experiment_id": EXPERIMENT, "source_sha256": source_sha, "scope": SCOPE, "pins": PINS,
            "counts": progress, "rows_per_arm": 7819, "source_files": len(new["source_sha256"]),
            "opaque_labels": new["files"]["labels.jsonl"], "workload": stats, "pilot": pilot_counts,
            "authenticated_inputs": bindings, "limits": LIMITS, "cpu_threads": 1,
            "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0,
            "scores_accessed": False, "labels_decoded": False, "wall_seconds": time.monotonic()-started,
            "process_lifetime_peak_rss_bytes": rss(), "files": {p.name: descriptor(p, check) for p in OUT.iterdir() if p.is_file()}}
        write(OUT / "receipt.json", result)
        check()
        print(json.dumps({"status": "completed", "agreement": True, "counts": progress, "receipt_sha256": sha(OUT / "receipt.json", check)}))
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (OUT / "receipt.json").exists():
                (OUT / "receipt.json").rename(OUT / "late-receipt.json")
            write(OUT / "failed.json", {"status": "failed", "pins": PINS, "scope": SCOPE,
                "progress": progress, "error": repr(error), "wall_seconds": time.monotonic()-started})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original audit failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    run()
