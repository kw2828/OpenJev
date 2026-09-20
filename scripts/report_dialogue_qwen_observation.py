"""Saved-only Qwen observation report. No model, scorer, tokenizer or cache imports."""
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
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-observation-report-v1"
RUN_VERSION = "dialogue-qwen-observation-v1"
ARMS = ("current", "history4")
SEEDS = (6201, 6202, 6203)
STRATA = ("all", "changed", "retained", "unmentioned_retention", "assigned_retention")
WEIGHTINGS = ("row", "equal_service", "equal_dialogue")
SUPPORT = dict(zip(STRATA, (7819, 578, 7241, 4032, 3209), strict=True))
NONE, DC = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
TYPE_NAMES = ("none", "dontcare", "true", "false", "other")
MODEL = ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b")
MODEL_FILES = {"added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json",
               "merges.txt", "model.safetensors", "model.safetensors.index.json", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json"}
SOURCES = {"src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
           "src/openjev/research/dialogue_qwen_observation.py", "scripts/prepare_dialogue_qwen_observation.py",
           "scripts/run_dialogue_qwen_observation.py", "tests/test_dialogue_qwen_observation.py",
           "tests/test_run_dialogue_qwen_observation.py", "tests/test_prepare_dialogue_qwen_observation.py",
           "research/dialogue-qwen-observation-protocol.md"}
REFERENCE = "output/dialogue-objective-v1/report-01/summary.json"
REFERENCE_SHA = "d744753d9edb545b9060867500e3c390d03da69937ffd6d8b3fd1e0c12c4cf6b"
RUN_LIMITS = {p: {"wall_seconds": s, "rss_bytes": 12*1024**3, "output_bytes": 512*1024**2}
              for p, s in (("pilot", 300), ("run", 7200))}
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
SCOPE = ("Saved-only complete-cohort observation comparison on exposed official TRAIN with correct previous gold. "
         "Candidate log probabilities, exact prompt-order ties, metrics and decisions are reconstructed. "
         "Public text/lexical provenance and actual inference/model identity are source-bound execution witnesses, "
         "not replayed. Unsaved full-vocabulary logits cannot be reconstructed: their saved partition witnesses "
         "are checked for mathematical consistency only. No architecture, autonomous-memory or calibration claim. "
         "Historical controls are separately fitted references; aggregate seed comparisons are not row-paired repairs.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode(value):
    def pairs(items):
        out = {}
        for key, item in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = item
        return out
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def sha(path, check=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
            check()
    return value.hexdigest()


def item(path, check=lambda: None):
    return {"sha256": sha(path, check), "bytes": Path(path).stat().st_size}


def safe(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe member")
    result = Path(root)/path
    require(not result.is_symlink() and result.resolve().is_relative_to(Path(root).resolve()), "Escaping member")
    return result


def bind(path, descriptor, bindings, check):
    require(set(descriptor) == {"sha256", "bytes"} and type(descriptor["bytes"]) is int
            and descriptor["bytes"] >= 0 and pin(descriptor["sha256"]), "File descriptor")
    require(item(path, check) == descriptor, "Payload identity: " + str(path))
    bindings[Path(path)] = descriptor


def pin(value):
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def manifest(path, files, expected, terminal, bindings, check):
    require(set(files) == set(expected) and {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
            == set(expected) | {terminal}, "Exact artifact membership")
    for name, descriptor in files.items():
        bind(safe(path, name), descriptor, bindings, check)
    bindings[path/terminal] = item(path/terminal, check)


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            **{k: importlib.metadata.version(k) for k in ("mlx", "mlx-lm", "transformers", "tokenizers")}}


def number(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            "Finite nonnegative numerical witness")
    return value


def close(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "ifu" and np.isfinite(a).all()
            and np.allclose(a, b, atol=2e-12, rtol=2e-12), "Numerical witness: " + name)


def work_for(row):
    lengths = [len(v) for v in row["tokens"]]
    slots = len(lengths)*max(lengths)
    return {"calls": 1, "prefill_calls": 0, "branch_calls": 1, "input_token_slots": slots,
            "prefix_tokens": 0, "padding_token_slots": slots-sum(lengths)}


def authenticate(args, check):
    """Bind every selected file before decoding task labels or prediction records."""
    prepared, run = Path(args.prepared).resolve(), Path(args.run).resolve()
    require(pin(args.plan_sha256) and sha(prepared/"plan.json", check) == args.plan_sha256, "External plan pin")
    require(pin(args.run_sha256) and sha(run/"completed.json", check) == args.run_sha256, "External run pin")
    plan, prep, done = read(prepared/"plan.json"), read(prepared/"completed.json"), read(run/"completed.json")
    bindings = {}
    require(prep["status"] == "completed" and prep["plan_sha256"] == args.plan_sha256
            and prep["model_calls"] == prep["encoder_calls"] == 0 and prep["tokenizer_only"] is True
            and prep["official_dev_dialogues_accessed"] is False and prep["test_contents_accessed"] is False,
            "Successful model-free preparation")
    manifest(prepared, prep["files"], {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"},
             "completed.json", bindings, check)
    require(0 < number(prep["wall_seconds"]) <= 180 and 0 < number(prep["peak_rss_bytes"]) <= 2*1024**3
            and sum(p.stat().st_size for p in prepared.rglob("*") if p.is_file()) <= 512*1024**2, "Preparation caps")
    require(plan["version"] == RUN_VERSION and plan["method"] == "batch" and plan["limits"] == RUN_LIMITS
            and plan["runtime"] == runtime() and plan["row_count"] == SUPPORT["all"]
            and plan["decisions"] == 2*SUPPORT["all"] and set(plan["source_sha256"]) == SOURCES,
            "Frozen recipe/runtime/source closure")
    for name, digest in plan["source_sha256"].items():
        path = safe(ROOT, name)
        require(pin(digest) and sha(path, check) == digest, "Frozen source identity: " + name)
        bindings[path] = item(path, check)
    require((plan["model"]["id"], plan["model"]["revision"]) == MODEL
            and set(plan["model"]["files_sha256"]) == MODEL_FILES
            and all(pin(v) for v in plan["model"]["files_sha256"].values()), "Model manifest identity")
    require(set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}
            and all(prep["files"][n] == v for n, v in plan["files"].items()), "Prepared payload joins")
    require(done["status"] == "completed" and done["phase"] == "run" and done["version"] == RUN_VERSION,
            "Full completed run required")
    manifest(run, done["files"], {"started.json", "plan.json", "timings.jsonl", "scores.jsonl"},
             "completed.json", bindings, check)
    require(sha(run/"plan.json", check) == args.plan_sha256, "Copied plan identity")
    start = read(run/"started.json")
    phase_identity(done, start, plan, args.plan_sha256, "run")
    pilot_path = Path(start["request"]["pilot"]).resolve()
    pilot_pin = start["request"]["pilot_sha256"]
    require(pin(pilot_pin) and pilot_pin == done["pilot_completed_sha256"]
            and sha(pilot_path/"completed.json", check) == pilot_pin, "Pilot completion pin")
    pilot = read(pilot_path/"completed.json")
    manifest(pilot_path, pilot["files"], {"started.json", "plan.json", "timings.jsonl"},
             "completed.json", bindings, check)
    require(sha(pilot_path/"plan.json", check) == args.plan_sha256, "Pilot copied plan")
    phase_identity(pilot, read(pilot_path/"started.json"), plan, args.plan_sha256, "pilot")
    for path, receipt, phase in ((run, done, "run"), (pilot_path, pilot, "pilot")):
        require(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= RUN_LIMITS[phase]["output_bytes"],
                "Phase storage cap")
    reference = safe(ROOT, REFERENCE)
    require(plan["inputs"][REFERENCE]["sha256"] == REFERENCE_SHA, "Fixed historical reference pin")
    bind(reference, plan["inputs"][REFERENCE], bindings, check)
    history = read(reference)
    require(history["status"] == "completed" and history["technical_validity_passed"] is True
            and history["version"] == "dialogue-objective-v1", "Completed historical report")
    return plan, prep, done, pilot, pilot_path, history, bindings


def phase_identity(done, start, plan, plan_pin, phase):
    require(done["version"] == start["version"] == RUN_VERSION and done["phase"] == start["phase"] == phase
            and done["status"] == "completed" and done["plan_sha256"] == start["request"]["plan_sha256"] == plan_pin
            and done["source_sha256"] == plan["source_sha256"] and done["model"] == plan["model"]
            and done["runtime"] == plan["runtime"] and done["limits"] == start["limits"] == RUN_LIMITS[phase]
            and done["labels_accessed"] is start["labels_accessed"] is False
            and done["quality_outputs_saved"] is (phase == "run") and done["generated_tokens"] == done["optimizer_updates"] == 0
            and done["no_retry"] is start["no_retry"] is True, "Phase identity/scope")
    require(0 < number(done["wall_seconds"]) <= RUN_LIMITS[phase]["wall_seconds"]
            and 0 < number(done["process_lifetime_peak_rss_bytes"]) <= RUN_LIMITS[phase]["rss_bytes"]
            and number(done["model_load_seconds"]) <= done["wall_seconds"], "Phase wall/RSS caps")


def candidate_type(cid):
    if cid in (NONE, DC):
        return 0 if cid == NONE else 1
    return {"true": 2, "false": 3}.get(cid.removeprefix("value:").casefold(), 4) if cid.startswith("value:") else 4


def request_layout(requests, labels, plan):
    require(len(labels) == SUPPORT["all"] and [r["row_index"] for r in labels]
            == sorted({r["row_index"] for r in labels}), "Unique sorted evaluator rows")
    by_id = {r["row_index"]: r for r in labels}
    layouts, slots, counts = {a: {} for a in ARMS}, Counter(), Counter()
    require(len({r["request_id"] for r in requests}) == len(requests), "Unique requests")
    for request in requests:
        arm = request["arm"]
        require(arm in ARMS and type(request["time"]) is int and request["time"] >= 0
                and type(request["dialogue_id"]) is str and request["label_ids"] == list(range(32, 44)), "Request identity")
        questions = request["request"]["questions"]
        n = len(questions)
        require(1 <= n <= 4 and n == len(request["tokens"]) == len(request["row_indices"])
                == len(request["ordered_candidate_ids"]) == len(request["canonical_id_maps"]), "Request alignment")
        for question, tokens, index, ordered, mapping in zip(questions, request["tokens"], request["row_indices"],
                request["ordered_candidate_ids"], request["canonical_id_maps"], strict=True):
            require(type(index) is int and index in by_id and index not in layouts[arm], "Exact decision membership")
            require(1 <= len(tokens) <= 4096 and all(type(t) is int and 0 <= t < 151936 for t in tokens), "Prompt tokens")
            row = by_id[index]
            require(row["dialogue_id"] == request["dialogue_id"] and row["time"] == request["time"]
                    and question["id"] == f"r{index}", "Label/request identity join")
            candidates = question["candidates"]
            require(len(candidates) in (4, 5, 6, 7, 11) and len(candidates) == row["candidate_count"]
                    and set(mapping) == {c["id"] for c in candidates} and len(mapping) == len(candidates)
                    and len(set(mapping.values())) == len(mapping), "Candidate membership")
            ids = [mapping[c["id"]] for c in candidates]
            require(ordered == [mapping[c["id"]] for c in sorted(candidates, key=lambda c: c["description"])]
                    and NONE in ids and DC in ids and row["previous_candidate_id"] in ids
                    and row["current_candidate_id"] in ids, "Supported targets/previous/prompt order")
            flags = []
            for c in candidates:
                prefix, suffix = c["description"].rsplit("\nPublic lexical flags: ", 1)
                parts = suffix.split(",")
                require(len(parts) == 10 and all(v in ("0", "1") for v in parts), "Exact binary lexical suffix")
                public_type = ("NOT_MENTIONED", "DONTCARE", "TRUE", "FALSE", "OTHER")[candidate_type(mapping[c["id"]])]
                require(prefix.endswith("\nCandidate type: " + public_type),
                        "Public candidate type suffix")
                flags.append([int(v) for v in parts])
            require(sum(f[4] for f in flags) == 1, "Unique inherited literal-current register")
            layouts[arm][index] = {"candidate_ids": ids, "ordered": ordered, "flags": flags,
                                  "literal": next(i for i, f in enumerate(flags) if f[4]), "question": question}
        slots[arm] += work_for(request)["input_token_slots"]
        counts[arm] += 1
    require(dict(slots) == plan["input_token_slots"] and dict(counts) == plan["request_counts"], "Full planned work")
    require(all(set(v) == set(by_id) for v in layouts.values()), "Both arms complete")
    require(layouts["current"] == layouts["history4"], "Both arms same question/previous/lexical inputs")
    groups = sorted({(r["dialogue_id"], r["time"]) for r in requests})
    selected = set(groups[:12])
    for arm in ARMS:
        longest = min((r for r in requests if r["arm"] == arm),
                      key=lambda r: (-max(map(len, r["tokens"])), r["dialogue_id"], r["time"], r["request_id"]))
        selected.add((longest["dialogue_id"], longest["time"]))
    require(plan["pilot_request_ids"] == [r["request_id"] for r in requests if (r["dialogue_id"], r["time"]) in selected],
            "Fixed pilot membership/order")
    rows = []
    for row in labels:
        layout = layouts["current"][row["row_index"]]
        ids = layout["candidate_ids"]
        old, target = ids.index(row["previous_candidate_id"]), ids.index(row["current_candidate_id"])
        derived = ("unmentioned_retention" if old == target and ids[target] == NONE else "assigned_retention" if old == target
                   else "first_assignment" if ids[old] == NONE else "clear" if ids[target] == NONE else "revision")
        require(row["derived_bin"] == derived and row["current_value_group"] == TYPE_NAMES[candidate_type(ids[target])],
                "Exact transition/type labels")
        rows.append({**row, "target": target, "previous": old, "types": [candidate_type(cid) for cid in ids],
                     "literal": layout["literal"]})
    require(len({r["service"] for r in rows}) == 6, "All six primary services")
    require({k: int(v.sum()) for k, v in masks(rows).items()} == SUPPORT, "Fixed primary stratum support")
    return rows, layouts


def validate_timings(timings, requests, done):
    require([t["request_id"] for t in timings] == [r["request_id"] for r in requests], "Timing exact order")
    for timing, request in zip(timings, requests, strict=True):
        require(timing["arm"] == request["arm"] and timing["questions"] == len(request["row_indices"])
                and timing["work"] == work_for(request) and timing["finite_logits"] is True, "Paid model work")
        number(timing["seconds"], positive=True)
        require(number(timing["maximum_probability_mass_error"]) <= 2e-12
                and type(timing["serialized_score_bytes"]) is int and timing["serialized_score_bytes"] > 0,
                "Numerical timing witness")
        number(timing["mlx_peak_active_bytes"]); number(timing["mlx_cached_allocator_bytes"])
    require(done["progress"] == {"requests_completed": len(requests), "questions_completed": sum(len(r["row_indices"]) for r in requests),
                "forward_calls_attempted": len(requests), "forward_calls_returned": len(requests), "active_request_id": None}
            and done["model_calls"] == len(requests)
            and done["work_totals"] == {k: sum(t["work"][k] for t in timings) for k in work_for(requests[0])},
            "Complete call/work totals")
    close(done["request_seconds"], sum(t["seconds"] for t in timings), "Sum request seconds")
    require(done["request_seconds"] + done["model_load_seconds"] <= done["wall_seconds"], "Nested phase time")


def validate_projection(pilot, timings, plan):
    arms = {}
    for arm in ARMS:
        values = [t for t in timings if t["arm"] == arm]
        rate = max(t["seconds"]/t["work"]["input_token_slots"] for t in values)
        request_rate = max(t["seconds"] for t in values)
        arms[arm] = {"max_seconds_per_charged_slot": rate, "max_seconds_per_request": request_rate,
                     "token_scaled_seconds": rate*plan["input_token_slots"][arm],
                     "request_scaled_seconds": request_rate*plan["request_counts"][arm],
                     "variable_seconds": rate*plan["input_token_slots"][arm]}
    projection = pilot["projection"]
    require(projection["arms"] == arms and projection["factor"] == 2 and projection["fixed_seconds"] == 60
            and projection["pilot_whole_wall_seconds"] == pilot["wall_seconds"]
            and projection["admission_limit_seconds"] == 7200, "Pilot projection arithmetic")
    expected = math.ceil(2*sum(v["variable_seconds"] for v in arms.values()) + pilot["wall_seconds"] + 60)
    request_bound = math.ceil(2*sum(v["request_scaled_seconds"] for v in arms.values())+pilot["wall_seconds"]+60)
    require(projection["projected_full_seconds"] == expected <= 7200 and projection["admitted"] is True
            and projection["request_scaled_projection_seconds_descriptive_only"] == request_bound, "Admitted fixed token projection")


def reconstruct_question(saved, layout, row_index, question_id):
    ids, ordered = layout["candidate_ids"], layout["ordered"]
    require(saved["row_index"] == row_index and saved["question_id"] == question_id and saved["candidate_ids"] == ids
            and saved["label_ids"] == [32+ordered.index(cid) for cid in ids]
            and saved["tie_break"] == "first frozen prompt label" and saved["vocabulary_size"] == 151936,
            "Saved candidate/label identity")
    z = np.asarray(saved["candidate_logits"], np.float64)
    require(z.shape == (len(ids),) and np.isfinite(z).all(), "Finite supported logits")
    maximum = float(z.max())
    centered = z-maximum
    normalizer = float(np.logaddexp.reduce(centered))
    logs = centered-normalizer
    probability = np.exp(logs)
    require(np.isfinite(logs).all(), "Finite direct log probabilities")
    close(saved["log_probs"], logs, "Log probabilities")
    close(saved["probabilities"], probability, "Probabilities")
    require(saved["candidate_max_logit"] == maximum and saved["candidate_max_tie_count"] == int((z == maximum).sum()),
            "Candidate max/tie witness")
    close(saved["candidate_shifted_log_partition"], normalizer, "Candidate partition")
    full_max, full_shift = saved["full_vocabulary_max_logit"], saved["full_vocabulary_shifted_log_partition"]
    require(type(full_max) in (int, float) and math.isfinite(full_max) and full_max >= maximum
            and 0 <= number(full_shift) <= math.log(saved["vocabulary_size"])+2e-12, "Full-vocabulary witness bounds")
    log_mass = (maximum-full_max)+normalizer-full_shift
    require(math.isfinite(log_mass) and log_mass <= 2e-12, "Candidate mass bounds")
    close(saved["log_candidate_token_mass"], log_mass, "Log candidate mass")
    close(saved["candidate_token_mass"], math.exp(log_mass), "Candidate mass")
    choice_id = next(cid for cid in ordered if z[ids.index(cid)] == maximum)
    require(saved["selected_id"] == choice_id, "Exact frozen-label tie selection")
    return logs, ids.index(choice_id)


def reconstruct(requests, scores, timings, layouts, rows):
    require([s["request_id"] for s in scores] == [r["request_id"] for r in requests], "Score exact order/membership")
    positions = {r["row_index"]: i for i, r in enumerate(rows)}
    logs = {arm: np.full((len(rows), 12), -np.inf, np.float64) for arm in ARMS}
    choices = {arm: np.full(len(rows), -1, np.int64) for arm in ARMS}
    validations = {arm: {"exact_tie_rows": 0, "underflowed_target_probabilities": 0, "maximum_mass_error": 0.,
                         "candidate_mass_min": 1., "candidate_mass_max": 0.} for arm in ARMS}
    for request, score, timing in zip(requests, scores, timings, strict=True):
        arm = request["arm"]
        require(all(score[k] == request[k] for k in ("request_id", "arm", "dialogue_id", "time"))
                and len(score["questions"]) == len(request["row_indices"]), "Score request join")
        require(len(encoded(score)) == timing["serialized_score_bytes"], "Serialized score work witness")
        mass_errors = []
        for index, question, saved in zip(request["row_indices"], request["request"]["questions"], score["questions"], strict=True):
            values, selected = reconstruct_question(saved, layouts[arm][index], index, question["id"])
            i = positions[index]
            require(choices[arm][i] == -1, "No duplicate scored row")
            logs[arm][i, :len(values)] = values; choices[arm][i] = selected
            v = validations[arm]
            error = abs(math.fsum(np.exp(values).tolist())-1)
            mass_errors.append(error)
            v["exact_tie_rows"] += saved["candidate_max_tie_count"] > 1
            v["underflowed_target_probabilities"] += bool(math.exp(values[rows[i]["target"]]) == 0)
            v["maximum_mass_error"] = max(v["maximum_mass_error"], error)
            v["candidate_mass_min"] = min(v["candidate_mass_min"], saved["candidate_token_mass"])
            v["candidate_mass_max"] = max(v["candidate_mass_max"], saved["candidate_token_mass"])
        close(timing["maximum_probability_mass_error"], max(mass_errors), "Timing probability witness")
    require(all((v >= 0).all() for v in choices.values()), "Complete reconstructed predictions")
    return logs, choices, validations


def masks(rows):
    changed = np.asarray([r["target"] != r["previous"] for r in rows])
    none = np.asarray([r["types"][r["target"]] == 0 for r in rows])
    return dict(zip(STRATA, (np.ones(len(rows), bool), changed, ~changed, ~changed & none, ~changed & ~none), strict=True))


def means(values, rows, selected):
    indices = np.flatnonzero(selected)
    if not len(indices):
        return dict.fromkeys(WEIGHTINGS)
    result = {"row": math.fsum(float(values[i]) for i in indices)/len(indices)}
    for name, field in (("equal_service", "service"), ("equal_dialogue", "dialogue_id")):
        groups = {}
        for i in indices:
            groups.setdefault(rows[i][field], []).append(float(values[i]))
        result[name] = math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups)
    return result


def vectors(rows, logs, choices):
    target = np.asarray([r["target"] for r in rows])
    types = np.asarray([r["types"]+[4]*(12-len(r["types"])) for r in rows])
    selected_type, target_type = types[np.arange(len(rows)), choices], types[np.arange(len(rows)), target]
    correct = choices == target
    branch = np.minimum(selected_type, 2) != np.minimum(target_type, 2)
    errors = np.exp(logs); errors[np.arange(len(rows)), target] -= 1
    return {"accuracy": correct.astype(float), "error": (~correct).astype(float),
            "wrong_selected_branch": branch.astype(float), "wrong_value": (~correct & ~branch).astype(float),
            "nll": -logs[np.arange(len(rows)), target], "brier": np.square(errors).sum(1)}


def cell(rows, choices, values, selected):
    result = {"rows": int(selected.sum()), "counts": {k: int(values[m][selected].sum()) for k, m in
              (("correct", "accuracy"), ("error", "error"), ("wrong_selected_branch", "wrong_selected_branch"), ("wrong_value", "wrong_value"))},
              "metrics": {k: means(v, rows, selected) for k, v in values.items()}, "rare": {}}
    target = np.asarray([r["types"][r["target"]] for r in rows])
    chosen = np.asarray([r["types"][c] for r, c in zip(rows, choices, strict=True)])
    for code, name in enumerate(TYPE_NAMES):
        supported = np.asarray([code in r["types"] for r in rows])
        for metric, eligible, event in (("recall", selected & (target == code), values["accuracy"] == 1),
                                        ("false_positive", selected & supported & (target != code), chosen == code)):
            n, d = int((eligible & event).sum()), int(eligible.sum())
            result["rare"][name+"_"+metric] = {"numerator": n, "denominator": d, "rate": n/d if d else None}
    return result


def tree_difference(a, b):
    if isinstance(a, dict):
        require(set(a) == set(b), "Difference schema")
        return {k: tree_difference(a[k], b[k]) for k in a}
    require((a is None) == (b is None), "Common support")
    return None if a is None else a-b


def mean_tree(values):
    if isinstance(values[0], dict):
        require(all(set(v) == set(values[0]) for v in values), "Seed mean schema")
        return {k: mean_tree([v[k] for v in values]) for k in values[0]}
    if values[0] is None:
        require(all(v is None for v in values), "Seed mean support")
        return None
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Finite seed mean")
    return math.fsum(values)/len(values)


def exact_rate(fit, stratum, metric, weighting):
    def rate(c):
        n, d = c["counts"]["correct" if metric == "accuracy" else "error"], c["rows"]
        require(type(n) is int and type(d) is int and 0 <= n <= d, "Exact decision counts")
        return Fraction(n, d) if d else None
    if weighting == "row":
        return rate(fit["cells"][stratum])
    values = [rate(s[stratum]) for s in fit["services"].values() if s[stratum]["rows"]]
    return sum(values, Fraction())/len(values) if values else None


def decisions(candidate, controls):
    behavioral, scoring = {}, {}
    for stratum, metric, bound in (("changed", "accuracy", Fraction(1, 50)), ("retained", "error", Fraction(0))):
        for weighting in ("row", "equal_service"):
            a = exact_rate(candidate, stratum, metric, weighting)
            bs = [exact_rate(c, stratum, metric, weighting) for c in controls]
            delta = a-sum(bs, Fraction())/len(bs) if a is not None and all(v is not None for v in bs) else None
            passed = delta is not None and (delta >= bound if metric == "accuracy" else delta <= bound)
            behavioral[f"{stratum}_{metric}_{weighting}"] = {"passed": bool(passed), "difference": None if delta is None else float(delta),
                                                            "threshold": float(bound), "relation": ">=" if metric == "accuracy" else "<="}
    for metric in ("nll", "brier"):
        for weighting in ("row", "equal_service"):
            a = candidate["cells"]["all"]["metrics"][metric][weighting]
            bs = [c["cells"]["all"]["metrics"][metric][weighting] for c in controls]
            delta = a-math.fsum(bs)/len(bs) if a is not None and all(b is not None for b in bs) else None
            scoring[f"{metric}_{weighting}"] = {"passed": delta is not None and math.isfinite(delta) and delta <= 0,
                                                "difference": delta, "threshold": 0., "relation": "<="}
    return {"behavioral": {"passed": all(v["passed"] for v in behavioral.values()), "checks": behavioral},
            "proper_score_nonregression": {"passed": all(v["passed"] for v in scoring.values()), "checks": scoring}}


def aggregate(rows, logs, choices, historical):
    groups = masks(rows)
    services = sorted({r["service"] for r in rows})
    service_masks = {s: np.asarray([r["service"] == s for r in rows]) for s in services}
    arms, values = {}, {}
    for arm in ARMS:
        values[arm] = vectors(rows, logs[arm], choices[arm])
        arms[arm] = {"cells": {k: cell(rows, choices[arm], values[arm], mask) for k, mask in groups.items()},
                     "services": {s: {k: cell(rows, choices[arm], values[arm], mask & sm) for k, mask in groups.items()}
                                  for s, sm in service_masks.items()}}
    references = {}
    target = np.asarray([r["target"] for r in rows])
    for name, key in (("carry", "previous"), ("literal", "literal")):
        selected = np.asarray([r[key] for r in rows])
        correct = selected == target
        def accuracy(mask, correct=correct):
            return {"rows": int(mask.sum()), "correct": int((mask & correct).sum()), "accuracy": means(correct, rows, mask)}
        references[name] = {"cells": {k: accuracy(m) for k, m in groups.items()},
                            "services": {s: {k: accuracy(m & sm) for k, m in groups.items()} for s, sm in service_masks.items()}}
    def pair(mask):
        a, b = choices["history4"] == target, choices["current"] == target
        return {"rows": int(mask.sum()), "wrong_to_correct": int((mask & a & ~b).sum()),
                "correct_to_wrong": int((mask & ~a & b).sum()), "both_correct": int((mask & a & b).sum()),
                "both_wrong": int((mask & ~a & ~b).sum()),
                "metric_differences": {k: means(values["history4"][k]-values["current"][k], rows, mask) for k in values["current"]}}
    paired = {"cells": {k: pair(m) for k, m in groups.items()},
              "services": {s: {k: pair(m & sm) for k, m in groups.items()} for s, sm in service_masks.items()}}
    flat = {}
    for seed in SEEDS:
        source = historical["historical"]["fits"][f"flat_stratum-corrected-{seed}"]
        fit = {"cells": {k: source["cells"]["heldout_service/"+k] for k in STRATA},
               "services": {s: source["services"][s] for s in services}}
        for k in STRATA:
            require(fit["cells"][k]["rows"] == arms["current"]["cells"][k]["rows"], "Historical fixed cohort support")
            for s in services:
                require(fit["services"][s][k]["rows"] == arms["current"]["services"][s][k]["rows"], "Historical per-service support")
        flat[str(seed)] = fit
    average = mean_tree(list(flat.values()))
    comparisons = {arm: {"seeds": {seed: {k: tree_difference(arms[arm]["cells"][k]["metrics"], fit["cells"][k]["metrics"])
                                         for k in STRATA} for seed, fit in flat.items()},
                         "mean": {k: tree_difference(arms[arm]["cells"][k]["metrics"], average["cells"][k]["metrics"]) for k in STRATA}}
                   for arm in ARMS}
    return {"arms": arms, "controls": references, "paired": {"history4_minus_current": paired},
            "historical": {"corrected_flat": {"seeds": flat, "mean": average}, "comparisons": comparisons,
                           "other_fixed_readouts": {kind: {name: {k: fit["cells"]["heldout_service/"+k]
                                                        for k in STRATA} for name, fit in source.items()}
                                                    for kind, source in (("fresh_objective", historical["fits"]),
                                                                         ("historical", historical["historical"]["fits"]))},
                           "reference_summary_sha256": REFERENCE_SHA,
                           "scope": "Historical three-fit aggregate comparisons, not fresh paired model runs or row-paired repairs"},
            "decisions": {"semantic_strength": decisions(arms["current"], list(flat.values())),
                          "added_history": decisions(arms["history4"], [arms["current"]])},
            "support": {k: int(v.sum()) for k, v in groups.items()}, "services": services}


def report_text(summary):
    lines = ["# Qwen semantic observation baseline", "", SCOPE, "",
             "| Arm | Changed accuracy | Retained error | Overall NLL | Overall Brier |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        c = summary["arms"][arm]["cells"]
        lines.append(f"| {arm} | {100*c['changed']['metrics']['accuracy']['row']:.4f}% | "
                     f"{100*c['retained']['metrics']['error']['row']:.4f}% | "
                     f"{c['all']['metrics']['nll']['row']:.6f} | {c['all']['metrics']['brier']['row']:.6f} |")
    for name, result in summary["decisions"].items():
        lines.extend(("", (f"{name}: behavioral {'PASS' if result['behavioral']['passed'] else 'FAIL'}; "
                       f"proper-score nonregression {'PASS' if result['proper_score_nonregression']['passed'] else 'FAIL'}.")))
    lines.extend(("", ("All row, equal-service and equal-dialogue metrics, all historical seeds, sparse-category denominators, "
                  "per-service results and paired repairs/harms are retained in summary.json. No combined architecture gate."),
                  "", (f"Preparation {summary['costs']['preparation_seconds']:.3f}s; pilot {summary['costs']['pilot_seconds']:.3f}s; "
                  f"full run {summary['costs']['run_seconds']:.3f}s. Pilot decisions were repeated and paid separately.")))
    return "\n".join(lines)+"\n"


def execute(args):
    out = Path(args.out).resolve()
    for value in (args.prepared, args.run):
        parent = Path(value).resolve()
        require(not out.is_relative_to(parent) and not parent.is_relative_to(out), "Separate output tree")
    out.mkdir(parents=True, exist_ok=False)
    started, prior, sources = time.monotonic(), None, {}
    request = {k: str(v) for k, v in vars(args).items()}
    def check():
        if time.monotonic()-started > LIMITS["wall_seconds"]:
            raise TimeoutError("Report wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Report RSS cap")
    def storage():
        check()
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Report output cap")
    def expired(_signal, _frame):
        raise TimeoutError("Report wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing timer")
        prior = signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        sources = {n: sha(ROOT/n, check) for n in ("scripts/report_dialogue_qwen_observation.py", "tests/test_report_dialogue_qwen_observation.py")}
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources, "limits": LIMITS})
        plan, prep, done, pilot, pilot_path, history, bindings = authenticate(args, check)
        def lines(path):
            result = []
            with Path(path).open() as stream:
                for i, line in enumerate(stream):
                    result.append(decode(line))
                    if i % 100 == 0:
                        check()
            return result
        requests = lines(Path(args.prepared)/"requests.jsonl")
        rows, layouts = request_layout(requests, lines(Path(args.prepared)/"labels.jsonl"), plan)
        timings = lines(Path(args.run)/"timings.jsonl")
        validate_timings(timings, requests, done)
        pilot_timings = lines(pilot_path/"timings.jsonl")
        validate_timings(pilot_timings, [r for r in requests if r["request_id"] in plan["pilot_request_ids"]], pilot)
        validate_projection(pilot, pilot_timings, plan)
        logs, choices, validations = reconstruct(requests, lines(Path(args.run)/"scores.jsonl"), timings, layouts, rows)
        check()
        summary = aggregate(rows, logs, choices, history)
        summary.update(status="completed", version=VERSION, technical_validity_passed=True, scope=SCOPE,
                       plan_sha256=args.plan_sha256, execution_completed_sha256=args.run_sha256,
                       source_sha256=plan["source_sha256"], validations=validations,
                       counts={"rows": len(rows), "decisions": 2*len(rows), "requests": len(requests)},
                       costs={"preparation_seconds": prep["wall_seconds"], "pilot_seconds": pilot["wall_seconds"],
                              "run_seconds": done["wall_seconds"], "combined_seconds": prep["wall_seconds"]+pilot["wall_seconds"]+done["wall_seconds"],
                              "pilot_projection": pilot["projection"], "run_work": done["work_totals"], "pilot_work": pilot["work_totals"],
                              "run_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
                              "per_arm_request_seconds": {a: math.fsum(t["seconds"] for t in timings if t["arm"] == a) for a in ARMS},
                              "scope": "Whole phases include load/authentication/IO; request timings exclude file writes and nest within the full run."})
        write(out/"summary.json", summary)
        with (out/"report.md").open("x") as stream:
            stream.write(report_text(summary))
        for path, descriptor in bindings.items():
            require(item(path, check) == descriptor, "End input identity")
        for name, digest in sources.items():
            require(sha(ROOT/name, check) == digest, "End report source identity")
        storage()
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "technical_validity_passed": True,
              "request": request, "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256,
              "source_sha256": sources, "study_source_sha256": plan["source_sha256"], "scope": SCOPE,
              "files": {n: item(out/n, check) for n in ("started.json", "summary.json", "report.md")},
              "input_members": {str(p): v for p, v in bindings.items()}, "limits": LIMITS,
              "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": peak_rss(),
              "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0, "no_retry": True,
              "wall_scope": "Through all input stability checks and output hashing; terminal write and return cap-checked"})
        storage()
        return summary
    except BaseException as error:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                  "source_sha256": sources, "error": repr(error), "wall_seconds": time.monotonic()-started,
                  "no_retry": True, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original exception
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "plan-sha256", "run", "run-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
