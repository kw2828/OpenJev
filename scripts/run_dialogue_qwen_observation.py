"""One bounded, label-blind Qwen observation pilot or full saved-distribution run.

The frozen requests are the only task data decoded here. Candidate log probabilities
and vocabulary mass use float64 log-domain arithmetic, without probability floors.
Model and tokenizer imports occur only after authentication inside the whole cap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-observation-v1"
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
ARMS = ("current", "history4")
ROW_COUNT = 7819
LABEL_IDS = list(range(32, 44))
VOCABULARY_SIZE = 151936
LIMITS = {phase: {"wall_seconds": seconds, "rss_bytes": 12*1024**3,
                  "output_bytes": 512*1024**2} for phase, seconds in (("pilot", 300), ("run", 7200))}
REQUIRED_SOURCES = {"scripts/run_dialogue_qwen_observation.py", "tests/test_run_dialogue_qwen_observation.py",
                    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py"}
MODEL_FILES = {"added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json",
               "merges.txt", "model.safetensors", "model.safetensors.index.json", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    def invalid(value):
        raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def sha(path, check=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
            check()
    return value.hexdigest()


def safe(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    return Path(root)/path


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            **{name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")}}


class Budget:
    def __init__(self, out, start, phase):
        self.out, self.start, self.limits = out, start, LIMITS[phase]
        self.progress = {"requests_completed": 0, "questions_completed": 0, "forward_calls_attempted": 0,
                         "forward_calls_returned": 0, "active_request_id": None}

    def check(self):
        if time.monotonic()-self.start > self.limits["wall_seconds"]:
            raise TimeoutError("Whole Qwen phase wall cap exceeded")
        require(peak_rss() <= self.limits["rss_bytes"], "Process-lifetime RSS cap exceeded")

    def storage(self, extra=0):
        self.check()
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        require(size+extra <= self.limits["output_bytes"], "Output storage cap exceeded")
        return size


def manifest(out, check=lambda: None):
    return {str(path.relative_to(out)): {"sha256": sha(path, check), "bytes": path.stat().st_size}
            for path in sorted(out.rglob("*")) if path.is_file()}


def authenticate_item(path, item, check):
    require(set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int
            and item["bytes"] >= 0, "Invalid artifact descriptor")
    require(path.is_file() and path.stat().st_size == item["bytes"]
            and sha(path, check) == item["sha256"], "Artifact identity: "+str(path))


def authenticate_prepared(prepared, pin, budget):
    require(sha(prepared/"plan.json", budget.check) == pin, "External plan digest")
    require(not any((prepared/name).exists() for name in ("failed.json", "late-completion.json")),
            "Preparation failed or was demoted")
    done = read(prepared/"completed.json")
    require(done["status"] == "completed" and done["plan_sha256"] == pin and done["model_calls"] == 0
            and done["tokenizer_only"] is True,
            "Preparation not completed")
    require(set(done["files"]) == {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"}
            and {str(p.relative_to(prepared)) for p in prepared.rglob("*") if p.is_file()}
            == set(done["files"]) | {"completed.json"}, "Preparation exact file closure")
    for name in ("started.json", "plan.json"):
        authenticate_item(prepared/name, done["files"][name], budget.check)
    plan = read(prepared/"plan.json")
    require(plan["version"] == VERSION and plan["method"] == "batch" and plan["limits"] == LIMITS,
            "Frozen version/method/limits")
    require(plan["runtime"] == runtime(), "Runtime drift")
    require(plan["row_count"] == ROW_COUNT and plan["decisions"] == 2*ROW_COUNT, "Fixed question scope")
    require(REQUIRED_SOURCES <= set(plan["source_sha256"]), "Missing runner/source bindings")
    for name, digest in plan["source_sha256"].items():
        require(sha(safe(ROOT, name), budget.check) == digest, "Source drift: "+name)
    model = plan["model"]
    require(model["id"] == MODEL_ID and model["revision"] == REVISION
            and set(model["files_sha256"]) == MODEL_FILES, "Frozen model identity")
    snapshot = Path(model["snapshot_path"])
    require(snapshot.name == REVISION, "Model snapshot revision path")
    for name, digest in model["files_sha256"].items():
        require(sha(snapshot/name, budget.check) == digest, "Model file drift: "+name)
    require(set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}, "Prepared manifest scope")
    require(all(done["files"][name] == item for name, item in plan["files"].items()),
            "Preparation receipt payload binding")
    # The sealed labels descriptor is never dereferenced, hashed or decoded.
    authenticate_item(prepared/"requests.jsonl", plan["files"]["requests.jsonl"], budget.check)
    return plan


def work_for(row):
    lengths = [len(tokens) for tokens in row["tokens"]]
    slots = len(lengths)*max(lengths)
    return {"calls": 1, "prefill_calls": 0, "branch_calls": 1, "input_token_slots": slots,
            "prefix_tokens": 0, "padding_token_slots": slots-sum(lengths)}


def validate_requests(rows, plan):
    require(rows and len({r["request_id"] for r in rows}) == len(rows), "Request membership")
    require(all(set(plan[key]) == set(ARMS)
                and all(type(v) is int and v > 0 for v in plan[key].values())
                for key in ("input_token_slots", "request_counts")), "Integral workload counts")
    slots, counts, identities = Counter(), Counter(), {arm: [] for arm in ARMS}
    for row in rows:
        require(row["arm"] in ARMS and type(row["request_id"]) is str
                and type(row["dialogue_id"]) is str and type(row["time"]) is int and row["time"] >= 0,
                "Request metadata")
        questions = row["request"]["questions"]
        n = len(questions)
        require(1 <= n <= 4 and len(row["tokens"]) == len(row["row_indices"])
                == len(row["ordered_candidate_ids"]) == len(row["canonical_id_maps"]) == n, "Question alignment")
        require(row["label_ids"] == LABEL_IDS and all(type(x) is int for x in row["label_ids"]), "Label token IDs")
        for question, tokens, index, ordered, mapping in zip(questions, row["tokens"], row["row_indices"],
                row["ordered_candidate_ids"], row["canonical_id_maps"], strict=True):
            require(type(index) is int and index >= 0, "Native row index")
            require(1 <= len(tokens) <= 4096 and all(type(t) is int and 0 <= t < VOCABULARY_SIZE for t in tokens),
                    "Frozen full prompt tokens")
            candidates = question["candidates"]
            ids = [c["id"] for c in candidates]
            require(2 <= len(ids) <= 12 and len(set(ids)) == len(ids) and set(mapping) == set(ids)
                    and all(type(v) is str and v for v in mapping.values())
                    and len(set(mapping.values())) == len(ids), "Canonical candidate map")
            require(ordered == [mapping[c["id"]] for c in sorted(candidates, key=lambda c: c["description"])],
                    "Prompt label/canonical order")
            identities[row["arm"]].append(index)
        slots[row["arm"]] += work_for(row)["input_token_slots"]
        counts[row["arm"]] += 1
    require(dict(slots) == plan["input_token_slots"] and dict(counts) == plan["request_counts"], "Full workload counts")
    require(all(len(indices) == len(set(indices)) == plan["row_count"] for indices in identities.values())
            and set(identities["current"]) == set(identities["history4"]), "Two-arm exact row coverage")
    pilot = plan["pilot_request_ids"]
    require(pilot and len(pilot) == len(set(pilot)) and set(pilot) <= {r["request_id"] for r in rows}, "Pilot membership")
    require({r["arm"] for r in rows if r["request_id"] in pilot} == set(ARMS), "Pilot both-arm coverage")
    groups = sorted({(r["dialogue_id"], r["time"]) for r in rows})
    selected_groups = set(groups[:12])
    for arm in ARMS:
        longest = min((r for r in rows if r["arm"] == arm),
                      key=lambda r: (-max(map(len, r["tokens"])), r["dialogue_id"], r["time"], r["request_id"]))
        selected_groups.add((longest["dialogue_id"], longest["time"]))
    require(pilot == [r["request_id"] for r in rows if (r["dialogue_id"], r["time"]) in selected_groups],
            "Fixed first12/longest pilot selection")


def summarize_distribution(logits, label_ids):
    """Retain log probabilities even when the displayed probability underflows."""
    values = np.asarray(logits, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "Nonfinite/invalid vocabulary logits")
    require(2 <= len(label_ids) <= 12 and len(set(label_ids)) == len(label_ids)
            and all(type(i) is int and 0 <= i < len(values) for i in label_ids), "Candidate label indices")
    candidate = values[label_ids]
    full_max, candidate_max = float(values.max()), float(candidate.max())
    full_shift = math.log(math.fsum(np.exp(values-full_max).tolist()))
    candidate_shift = math.log(math.fsum(np.exp(candidate-candidate_max).tolist()))
    log_probs = (candidate-candidate_max)-candidate_shift
    log_mass = (candidate_max-full_max)+candidate_shift-full_shift
    probabilities = np.exp(log_probs)
    require(np.isfinite(log_probs).all() and math.isfinite(log_mass) and log_mass <= 2e-12
            and abs(math.fsum(probabilities.tolist())-1) <= 2e-12, "Invalid stable distribution")
    return {"candidate_logits": candidate.tolist(), "log_probs": log_probs.tolist(),
            "probabilities": probabilities.tolist(), "log_candidate_token_mass": float(log_mass),
            "candidate_token_mass": math.exp(log_mass), "full_vocabulary_max_logit": full_max,
            "full_vocabulary_shifted_log_partition": full_shift, "candidate_max_logit": candidate_max,
            "candidate_shifted_log_partition": candidate_shift, "vocabulary_size": len(values),
            "candidate_max_tie_count": int(np.count_nonzero(candidate == candidate_max))}


def output_questions(row, values):
    require(values.shape == (len(row["row_indices"]), VOCABULARY_SIZE), "Full vocabulary output shape")
    output = []
    for i, (question, mapping, ordered, index) in enumerate(zip(row["request"]["questions"],
            row["canonical_id_maps"], row["ordered_candidate_ids"], row["row_indices"], strict=True)):
        result = summarize_distribution(values[i], row["label_ids"][:len(ordered)])
        selected = ordered[int(np.argmax(result["candidate_logits"]))]
        canonical = [mapping[c["id"]] for c in question["candidates"]]
        positions = [ordered.index(candidate) for candidate in canonical]
        for field in ("candidate_logits", "log_probs", "probabilities"):
            result[field] = [result[field][j] for j in positions]
        output.append({"row_index": index, "question_id": question["id"], "candidate_ids": canonical,
                       "label_ids": [row["label_ids"][j] for j in positions], "selected_id": selected,
                       "tie_break": "first frozen prompt label", **result})
    return output


def load_backend(plan):
    from huggingface_hub import snapshot_download

    from openjev.decisions import MLXScorer
    from openjev.research.shared_prefix import SharedPrefixExperiment
    # MLXScorer is pinned, local-files-only, and never invokes text generation.
    resolved = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True,
                                 allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    require(Path(resolved).resolve() == Path(plan["model"]["snapshot_path"]).resolve(), "Loaded model snapshot path")
    scorer = MLXScorer()
    require(scorer.label_ids == LABEL_IDS, "Loaded tokenizer label IDs")
    scorer.mx.eval(scorer.model.parameters())
    scorer.mx.synchronize()
    return scorer, SharedPrefixExperiment(scorer)


def verify_tokens(row, experiment):
    from openjev.decisions import DecisionRequest
    prepared = experiment.prepare(DecisionRequest.model_validate(row["request"]))
    require([tokens for _, _, tokens in prepared] == row["tokens"], "Retokenized prompt drift")
    require([[mapping[c.id] for c in ordered] for (_, ordered, _), mapping in
             zip(prepared, row["canonical_id_maps"], strict=True)] == row["ordered_candidate_ids"],
            "Retokenized candidate identity drift")


def projection(timings, plan, pilot_wall):
    arms = {}
    for arm in ARMS:
        rows = [row for row in timings if row["arm"] == arm]
        require(rows and all(math.isfinite(row["seconds"]) and row["seconds"] > 0 for row in rows), "Pilot timing support")
        per_slot = max(row["seconds"]/row["work"]["input_token_slots"] for row in rows)
        per_request = max(row["seconds"] for row in rows)
        token_bound, request_bound = per_slot*plan["input_token_slots"][arm], per_request*plan["request_counts"][arm]
        arms[arm] = {"max_seconds_per_charged_slot": per_slot, "max_seconds_per_request": per_request,
                     "token_scaled_seconds": token_bound, "request_scaled_seconds": request_bound,
                     "variable_seconds": token_bound}
    projected = math.ceil(2*sum(row["variable_seconds"] for row in arms.values())+pilot_wall+60)
    return {"arms": arms, "factor": 2, "pilot_whole_wall_seconds": pilot_wall, "fixed_seconds": 60,
            "projected_full_seconds": projected, "admission_limit_seconds": LIMITS["run"]["wall_seconds"],
            "admitted": projected <= LIMITS["run"]["wall_seconds"],
            "request_scaled_projection_seconds_descriptive_only": math.ceil(
                2*sum(row["request_scaled_seconds"] for row in arms.values())+pilot_wall+60),
            "scope": "Heuristic token-rate admission, not a wall-time guarantee; request-rate estimate is descriptive only"}


def authenticate_pilot(path, pin, plan, plan_pin, requests, budget):
    require(sha(path/"completed.json", budget.check) == pin, "External pilot digest")
    done = read(path/"completed.json")
    require(done["status"] == "completed" and done["version"] == VERSION and done["phase"] == "pilot"
            and done["plan_sha256"] == plan_pin and done["runtime"] == plan["runtime"]
            and done["source_sha256"] == plan["source_sha256"] and done["model"] == plan["model"]
            and done["labels_accessed"] is False and done["quality_outputs_saved"] is False, "Pilot identity/scope")
    require(set(done["files"]) == {"started.json", "plan.json", "timings.jsonl"}
            and {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
            == set(done["files"]) | {"completed.json"}, "Pilot exact file closure")
    for name, item in done["files"].items():
        authenticate_item(path/name, item, budget.check)
    require(sha(path/"plan.json", budget.check) == plan_pin, "Pilot copied plan")
    started = read(path/"started.json")
    require(started["version"] == VERSION and started["phase"] == "pilot"
            and started["request"]["plan_sha256"] == plan_pin and started["labels_accessed"] is False,
            "Pilot start identity")
    timings = [json.loads(line) for line in (path/"timings.jsonl").read_text().splitlines()]
    selected = [r for r in requests if r["request_id"] in plan["pilot_request_ids"]]
    require([r["request_id"] for r in timings] == [r["request_id"] for r in selected], "Pilot exact request order")
    for result, request in zip(timings, selected, strict=True):
        require(result["arm"] == request["arm"] and result["work"] == work_for(request)
                and result["questions"] == len(request["row_indices"]) and result["finite_logits"] is True,
                "Pilot work/numerical witness")
    require(0 < done["wall_seconds"] <= LIMITS["pilot"]["wall_seconds"]
            and done["process_lifetime_peak_rss_bytes"] <= LIMITS["pilot"]["rss_bytes"], "Pilot caps")
    require(done["progress"] == {"requests_completed": len(selected),
                "questions_completed": sum(len(r["row_indices"]) for r in selected),
                "forward_calls_attempted": len(selected), "forward_calls_returned": len(selected),
                "active_request_id": None}
            and done["work_totals"] == {key: sum(t["work"][key] for t in timings) for key in work_for(selected[0])}
            and sum(t["seconds"] for t in timings) <= done["wall_seconds"]
            and sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= LIMITS["pilot"]["output_bytes"],
            "Pilot complete paid work/caps")
    expected = projection(timings, plan, done["wall_seconds"])
    require(done["projection"] == expected and expected["admitted"] is True, "Pilot did not admit the full allocation")
    return done


def note(error, prefix, action):
    try:
        action()
    except BaseException as secondary:  # noqa: BLE001 - never replace the original terminal error
        if callable(getattr(error, "add_note", None)):
            error.add_note(prefix+repr(secondary))


def execute(args):
    start = time.monotonic()
    out, prepared = Path(args.out), Path(args.prepared)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, start, args.phase)
    request = {key: str(value) for key, value in vars(args).items()}
    old_signal, plan, timings = None, None, []
    def timeout(_signal, _frame):
        raise TimeoutError("Whole Qwen phase wall cap exceeded")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        old_signal = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, budget.limits["wall_seconds"])
        write(out/"started.json", {"version": VERSION, "phase": args.phase, "request": request,
                                    "limits": budget.limits, "no_retry": True, "labels_accessed": False})
        plan = authenticate_prepared(prepared, args.plan_sha256, budget)
        rows = [json.loads(line) for line in (prepared/"requests.jsonl").read_text().splitlines()]
        validate_requests(rows, plan)
        with (out/"plan.json").open("xb") as stream:
            stream.write((prepared/"plan.json").read_bytes())
        if args.phase == "run":
            require(args.pilot is not None and args.pilot_sha256 is not None, "Full run requires completed pilot pin")
            authenticate_pilot(Path(args.pilot), args.pilot_sha256, plan, args.plan_sha256, rows, budget)
        selected = rows if args.phase == "run" else [r for r in rows if r["request_id"] in plan["pilot_request_ids"]]
        budget.check()
        load_start = time.monotonic()
        scorer, experiment = load_backend(plan)
        model_load_seconds = time.monotonic()-load_start
        budget.check()
        score_stream = None
        try:
            if args.phase == "run":
                score_stream = (out/"scores.jsonl").open("xb")
            with (out/"timings.jsonl").open("xb") as timing_stream:
                for row in selected:
                    budget.check()
                    budget.progress["active_request_id"] = row["request_id"]
                    scorer.mx.synchronize()
                    scorer.mx.reset_peak_memory()
                    began = time.monotonic()
                    verify_tokens(row, experiment)
                    budget.check()
                    budget.progress["forward_calls_attempted"] += 1
                    values, work = experiment.logits(row["tokens"], "batch")
                    budget.progress["forward_calls_returned"] += 1
                    require(vars(work) == work_for(row), "Actual model work differs from full batch")
                    questions = output_questions(row, values)
                    score = {"request_id": row["request_id"], "arm": row["arm"], "dialogue_id": row["dialogue_id"],
                             "time": row["time"], "questions": questions}
                    payload = encoded(score)  # Pilot pays the same CPU reconstruction/serialization, then discards it.
                    scorer.mx.synchronize()
                    elapsed = time.monotonic()-began
                    timing = {"request_id": row["request_id"], "arm": row["arm"], "seconds": elapsed,
                              "questions": len(questions), "work": vars(work), "finite_logits": True,
                              "maximum_probability_mass_error": max(abs(math.fsum(q["probabilities"])-1) for q in questions),
                              "serialized_score_bytes": len(payload), "mlx_peak_active_bytes": int(scorer.mx.get_peak_memory()),
                              "mlx_cached_allocator_bytes": int(scorer.mx.get_cache_memory())}
                    timing_bytes = encoded(timing)
                    budget.storage(len(timing_bytes)+(len(payload) if score_stream is not None else 0))
                    if score_stream is not None:
                        score_stream.write(payload)
                        score_stream.flush()
                    timing_stream.write(timing_bytes)
                    timing_stream.flush()
                    timings.append(timing)
                    budget.progress["requests_completed"] += 1
                    budget.progress["questions_completed"] += len(questions)
                    budget.progress["active_request_id"] = None
                    budget.check()
                    if budget.progress["requests_completed"] % 100 == 0:
                        print(json.dumps({"phase": args.phase, "requests_completed": budget.progress["requests_completed"],
                                          "requests_total": len(selected), "wall_seconds": time.monotonic()-start}), flush=True)
        finally:
            if score_stream is not None:
                score_stream.close()
        require(authenticate_prepared(prepared, args.plan_sha256, budget) == plan, "End-of-run source/model/input drift")
        files = manifest(out, budget.check)
        wall = time.monotonic()-start
        result = {"status": "completed", "version": VERSION, "phase": args.phase, "plan_sha256": args.plan_sha256,
                  "source_sha256": plan["source_sha256"], "model": plan["model"], "runtime": plan["runtime"],
                  "limits": budget.limits, "files": files, "progress": budget.progress, "wall_seconds": wall,
                  "model_load_seconds": model_load_seconds, "request_seconds": sum(t["seconds"] for t in timings),
                  "process_lifetime_peak_rss_bytes": peak_rss(), "labels_accessed": False,
                  "quality_outputs_saved": args.phase == "run", "generated_tokens": 0, "optimizer_updates": 0,
                  "model_calls": budget.progress["forward_calls_returned"],
                  "no_retry": True, "pilot_completed_sha256": getattr(args, "pilot_sha256", None),
                  "work_totals": {key: sum(t["work"][key] for t in timings) for key in work_for(selected[0])},
                  "timing_scope": "Per request: token verification, inference/readback, float64 readout, JSON encoding; file writes outside request timer",
                  "wall_scope": "Whole phase includes authentication/load/all requests/serialization/closing hashes; completion write/hash/return cap-checked",
                  "mass_scope": "Candidate logits and shifted full-vocabulary log partition saved; unsaved vocabulary logits remain execution witnesses"}
        if args.phase == "pilot":
            result["projection"] = projection(timings, plan, wall)
        budget.storage(len(encoded(result)))
        write(out/"completed.json", result)
        budget.storage()
        return sha(out/"completed.json", budget.check)
    except BaseException as error:
        if old_signal is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if (out/"completed.json").exists():
            note(error, "Completion demotion: ", lambda: (out/"completed.json").rename(out/"late-completion.json"))
        note(error, "Failure receipt: ", lambda error=error: write(out/"failed.json", {
            "status": "failed", "version": VERSION, "phase": args.phase, "request": request, "error": repr(error),
            "progress": budget.progress, "wall_seconds": time.monotonic()-start,
            "process_lifetime_peak_rss_bytes": peak_rss(), "labels_accessed": False, "no_retry": True}))
        raise
    finally:
        if old_signal is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_signal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for phase in ("pilot", "run"):
        child = sub.add_parser(phase)
        child.add_argument("--prepared", type=Path, required=True)
        child.add_argument("--plan-sha256", required=True)
        child.add_argument("--out", type=Path, required=True)
        if phase == "run":
            child.add_argument("--pilot", type=Path, required=True)
            child.add_argument("--pilot-sha256", required=True)
    print(json.dumps({"completed_sha256": execute(parser.parse_args())}))


if __name__ == "__main__":
    main()
