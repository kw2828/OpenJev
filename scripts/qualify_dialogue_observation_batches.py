"""Bounded effective-batch cost qualification with synthetic endpoint targets."""
from __future__ import annotations

import argparse
import contextlib
import json
import signal
import time
from collections import Counter
from pathlib import Path

import qualify_dialogue_finetune as qualified

ROOT = qualified.ROOT
VERSION = "dialogue-observation-batches-v1"
PREPARED_PIN = "d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83"
PREPARED_PLAN_PIN = "4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e"
PROTOCOL = "research/dialogue-observation-learning-cost-protocol.md"
NEW_SOURCES = ["scripts/qualify_dialogue_observation_batches.py",
               "tests/test_qualify_dialogue_observation_batches.py",
               "src/openjev/research/dialogue_finetune_training.py",
               "tests/test_dialogue_finetune_training.py", PROTOCOL]
ARMS = ["frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers"]
CAPS = {"wall_seconds": 300, "rss_bytes": 8 * 1024**3,
        "mps_driver_bytes": 8 * 1024**3, "output_bytes": 256 * 1024**2}
CONFIG = {"seed": 6901, "arms": ARMS, "encoder_batch": 32, "chunk_tokens": 254,
          "memory_lr": .001, "encoder_lr": .00002, "weight_decay": .0001, "clip": 1.,
          "parity_tolerance": 2e-5, "cpu_threads": 1, "interop_threads": 1,
          "encoder_device": "mps", "memory_device": "cpu", "dtype": "float32"}
BATCH_METRICS = ("encoder_calls", "padded_token_positions", "padded_attention_positions",
                 "real_question_updates", "padded_candidate_positions", "encoder_sequences")
SINGLE_METRICS = ("padded_attention_positions", "encoder_sequences", "max_chunk_tokens_with_special",
                  "padded_candidate_positions", "real_question_updates", "public_user_turns")
PREPARED_MEMBERS = {"started.json", "source-pins.json", "plan.json", "actors-train.jsonl",
                    "actors-dev.jsonl", "targets-train.jsonl", "targets-dev.jsonl", "index.json",
                    "lexical-original.npy", "lexical-numbers.npy", "workloads.json", "orders.json",
                    "effective-batches.json"}
ENCODER_KEYS = ("input_texts", "content_tokens", "encoder_sequences", "encoder_calls",
                "special_token_positions", "valid_token_positions", "padded_token_positions",
                "padded_attention_positions", "padding_token_positions", "overlength_texts_chunked",
                "truncated_tokens")
EXPECTED = {"cells": 40, "optimizer_updates": 52, "training_dialogue_visits": 1168,
            "evaluation_forwards": 24, "parity_dialogue_visits": 412,
            "backward_calls": 1168, "memory_forwards": 1192, "encoding_passes": 1604,
            "checkpoints": 4}
require, sha, read, write = qualified.require, qualified.sha, qualified.read, qualified.write


def bind(mapping):
    for path, pin in mapping.items():
        require(sha(ROOT / path) == pin, "Changed pinned source/input: " + path)


def authenticate_prepared(path, pin):
    require(pin == PREPARED_PIN and sha(path / "completed.json") == pin, "Fixed prepared completion")
    done = read(path / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "prepare"
            and done["plan_sha256"] == PREPARED_PLAN_PIN and done["encoder_calls"] == 0
            and done["model_weights_loaded"] is False and done["official_test_opened"] is False,
            "Successful model-free prepared identity")
    require(set(done["files"]) == PREPARED_MEMBERS
            and {p.name for p in path.iterdir()} == PREPARED_MEMBERS | {"completed.json"}, "Prepared file closure")
    for name, record in done["files"].items():
        require((path / name).stat().st_size == record["bytes"] and sha(path / name) == record["sha256"],
                "Prepared payload identity: " + name)
    require(sha(path / "plan.json") == PREPARED_PLAN_PIN, "Prepared plan identity")
    plan = read(path / "plan.json")
    require(plan["preparation_version"] == 2 and plan["study"] == "dialogue-observation-learning-v1"
            and plan["runtime"] == qualified.runtime() and plan["config"]["arms"] == ARMS,
            "Prepared version/runtime/arms")
    require(plan["data_files"] == {k: v for k, v in done["files"].items() if k != "plan.json"}, "Prepared data map")
    require(len(plan["source_sha256"]) == 39 and read(path / "source-pins.json") == plan["source_sha256"],
            "Exact inherited 39-source map")
    bind(plan["source_sha256"])
    bind(plan["input_sha256"])
    for model in plan["model_files"].values():
        require(sha(model["path"]) == model["sha256"], "Frozen model asset")
    return plan


@contextlib.contextmanager
def attempt(out, phase, request):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": phase, "operation": "authenticate"}

    def check():
        require(time.perf_counter() - start <= CAPS["wall_seconds"], "Whole-phase wall cap")
        require(qualified.rss() <= CAPS["rss_bytes"], "Process RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= CAPS["output_bytes"], "Output cap")

    def timeout(*_):
        raise TimeoutError("Whole-phase wall cap")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, CAPS["wall_seconds"])
    try:
        write(out / "started.json", {"version": VERSION, "phase": phase, "request": request,
                                    "config": CONFIG, "caps": CAPS, "runtime": qualified.runtime()})
        yield start, progress, check
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "invalid-completion.json")
            write(out / "failed.json", {"status": "failed", "version": VERSION, "request": request,
                "error_type": type(error).__name__, "error": str(error), "progress": progress,
                "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": qualified.rss(),
                "scope": "One failed qualification attempt; no task scoring or automatic admission"})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            error.add_note("Failure receipt error: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def select_cases(workloads, batches, orders):
    """Only fixed public geometry and order metadata select workloads."""
    profiles = {(p["split"], p["dialogue_id"]): p["work"] for p in workloads["profiles"]}
    require(len(profiles) == len(workloads["profiles"]), "Distinct split-qualified profiles")
    require(set(batches["maxima"]) == set(BATCH_METRICS), "Complete effective-batch maxima")
    ids, maxima = orders["dialogue_ids"], {}
    require(len(ids) == len(set(ids)), "Distinct TRAIN order IDs")
    for metric, record in batches["maxima"].items():
        seed, epoch, start = record["seed"], record["epoch"], record["batch_start"]
        key = (seed, epoch, start)
        indices = orders["orders"][str(seed)][epoch][start:start + 32]
        dids = [ids[i] for i in indices]
        require(len(dids) == 32 and record["dialogue_indices"] == indices
                and record["dialogue_ids"] == dids, "Maximum batch/order membership")
        require(record["value"] == sum(profiles[("train", d)][metric] for d in dids), "Maximum batch work value")
        if key not in maxima:
            maxima[key] = {"dialogue_ids": dids, "maximizes": []}
        require(maxima[key]["dialogue_ids"] == dids, "Consistent duplicate maximum")
        maxima[key]["maximizes"].append(metric)
    require(len(maxima) == 3, "Exactly three distinct maximum full batches")
    cases = [{"case_id": f"batch-{i}", "kind": "batch", "split": "train", "seed": key[0],
              "epoch": key[1], "batch_start": key[2], "iterations": 3, **value}
             for i, (key, value) in enumerate(sorted(maxima.items()))]
    for split in ("train", "dev"):
        group = {d: w for (s, d), w in profiles.items() if s == split}
        winners = {}
        for metric in SINGLE_METRICS:
            did = min(group, key=lambda d: (-group[d][metric], d))
            winners.setdefault(did, []).append(metric)
        require(len(winners) == 3, "Exactly three maximum singles per split")
        for i, (did, reasons) in enumerate(sorted(winners.items())):
            cases.append({"case_id": f"{split}-single-{i}", "kind": "single" if split == "train" else "eval",
                          "split": split, "dialogue_ids": [did], "maximizes": reasons,
                          "iterations": 1 if split == "train" else 2})
    tail = orders["orders"]["6901"][0][2016:]
    require(len(ids) == 2017 and len(tail) == 1, "Fixed first-seed epoch-zero one-dialogue tail")
    cases.insert(6, {"case_id": "tail", "kind": "tail", "split": "train", "seed": 6901,
                    "epoch": 0, "batch_start": 2016, "dialogue_ids": [ids[tail[0]]], "iterations": 1})
    return cases


def expected_counts(cases):
    train = [c for c in cases if c["split"] == "train"]
    evaluation = [c for c in cases if c["split"] == "dev"]
    visits = 4 * sum(len(c["dialogue_ids"]) * c["iterations"] for c in train)
    evals = 4 * sum(len(c["dialogue_ids"]) * c["iterations"] for c in evaluation)
    parity = 4 * sum(len(c["dialogue_ids"]) for c in cases)
    return {"cells": 4 * len(cases), "optimizer_updates": 4 * sum(c["iterations"] for c in train),
            "training_dialogue_visits": visits, "evaluation_forwards": evals,
            "parity_dialogue_visits": parity, "backward_calls": visits,
            "memory_forwards": visits + evals, "encoding_passes": visits + evals + parity, "checkpoints": 4}


def prepare(prepared, pin, protocol_pin, out):
    request = {"prepared": str(prepared.resolve()), "prepared_sha256": pin, "protocol_sha256": protocol_pin}
    with attempt(out, "prepare-cost-plan", request) as (start, _, check):
        parent = authenticate_prepared(prepared, pin)
        require(sha(ROOT / PROTOCOL) == protocol_pin, "Prospective protocol pin")
        sources = {**parent["source_sha256"], **{name: sha(ROOT / name) for name in NEW_SOURCES}}
        require(len(sources) == 44, "Exact 44-source closure")
        cases = select_cases(read(prepared / "workloads.json"), read(prepared / "effective-batches.json"),
                             read(prepared / "orders.json"))
        require(expected_counts(cases) == EXPECTED, "Frozen complete pilot work schedule")
        plan = {"version": VERSION, "config": CONFIG, "caps": CAPS, "runtime": qualified.runtime(),
                "prepared": str(prepared.resolve()), "prepared_completed_sha256": pin,
                "prepared_plan_sha256": PREPARED_PLAN_PIN, "protocol_sha256": protocol_pin,
                "source_sha256": sources, "cases": cases, "expected_counts": EXPECTED,
                "scope": "Raw measured costs only, synthetic labels at scored positions; no full-study admission"}
        write(out / "plan.json", plan)
        authenticate_prepared(prepared, pin)
        bind(sources)
        check()
        write(out / "completed.json", {"version": VERSION, "status": "completed", "phase": "prepare-cost-plan",
              "plan_sha256": sha(out / "plan.json"), "files": qualified.manifest(out),
              "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": qualified.rss(), "model_calls": 0})
        result = sha(out / "completed.json")
        check()
    return result


def authenticate_plan(path, pin):
    require(sha(path / "completed.json") == pin, "Cost preparation completion pin")
    done = read(path / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "prepare-cost-plan"
            and set(done["files"]) == {"started.json", "plan.json"}
            and {p.name for p in path.iterdir()} == {"started.json", "plan.json", "completed.json"}, "Cost plan closure")
    for name, value in done["files"].items():
        require(sha(path / name) == value["sha256"] and (path / name).stat().st_size == value["bytes"], "Cost plan member")
    require(sha(path / "plan.json") == done["plan_sha256"], "Cost plan hash")
    plan = read(path / "plan.json")
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["caps"] == CAPS
            and plan["runtime"] == qualified.runtime(), "Cost recipe/runtime")
    prepared = Path(plan["prepared"])
    parent = authenticate_prepared(prepared, plan["prepared_completed_sha256"])
    require(plan["prepared_plan_sha256"] == PREPARED_PLAN_PIN, "Cost plan parent identity")
    expected_sources = {**parent["source_sha256"], **{n: sha(ROOT / n) for n in NEW_SOURCES}}
    require(plan["source_sha256"] == expected_sources and len(expected_sources) == 44, "Frozen source closure")
    require(plan["protocol_sha256"] == sha(ROOT / PROTOCOL), "Frozen cost protocol")
    cases = select_cases(read(prepared / "workloads.json"), read(prepared / "effective-batches.json"),
                         read(prepared / "orders.json"))
    require(plan["cases"] == cases and plan["expected_counts"] == expected_counts(cases) == EXPECTED, "Exact frozen cases")
    return plan, parent


def synthetic_rows(payload, positions, offset):
    """Only public support plus scored positions; no official target/bin input."""
    result, seen = [], set()
    for position in positions:
        t, q = position["time"], position["query_position"]
        require(type(t) is int and 0 <= t < len(payload["turn_text_ids"])
                and type(q) is int and 0 <= q < len(payload["candidate_ids"]), "Synthetic endpoint coordinates")
        require((t, q) not in seen, "Duplicate synthetic endpoint")
        seen.add((t, q))
        result.append({"time": t, "query_position": q,
                       "label_index": (t + q + offset) % len(payload["candidate_ids"][q]),
                       "stratum_index": (t + q + offset) % 3})
    require(result, "Nonempty scored position schedule")
    return result


def load_selected(prepared, cases):
    """Decode actors and whitelist scored coordinates, never use gold values."""
    wanted = {(c["split"], d) for c in cases for d in c["dialogue_ids"]}
    actors, positions = {}, {}
    layouts = {(e["split"], e["dialogue_id"]): e for e in read(prepared / "index.json")}
    for split in ("train", "dev"):
        for prefix, target in (("actors", actors), ("targets", positions)):
            with (prepared / f"{prefix}-{split}.jsonl").open() as stream:
                for line in stream:
                    value = json.loads(line)
                    key = (split, value["dialogue_id"])
                    if key not in wanted:
                        continue
                    require(value["split"] == split and key not in target, "Selected split-qualified payload identity")
                    if prefix == "actors":
                        target[key] = value
                    else:
                        target[key] = [{"time": r["time"], "query_position": r["query_position"]} for r in value["rows"]]
    require(set(actors) == set(positions) == wanted, "All selected actors/positions")
    for key, actor in actors.items():
        entry = layouts[key]
        require(actor["lexical_offset"] == entry["offset"] and actor["lexical_shape"] == entry["shape"]
                and len(positions[key]) == entry["scored_rows"], "Selected layout and endpoint count")
        synthetic_rows(actor, positions[key], 0)
    return actors, positions


def append_event(out, record):
    with (out / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def run(preparation, pin, out):
    request = {"preparation": str(preparation.resolve()), "preparation_sha256": pin}
    with attempt(out, "model-pilot", request) as (start, progress, check):
        plan, parent = authenticate_plan(preparation, pin)
        # All source/input/runtime validation precedes neural-library/model dispatch.
        import gc

        import numpy as np
        import torch
        from study_dialogue_copy_v2 import MonitoredCopyMemoryV2, empty_invariants, merge_invariants
        from transformers import AutoModel

        from openjev.research.dialogue_finetune_inputs import workload_profile
        from openjev.research.dialogue_finetune_training import supervised_loss
        from openjev.research.dialogue_trainable_encoder import build_actor, encode_token_lists

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.backends.mps.is_available(), "MPS required; no CPU fallback")
        prepared = Path(plan["prepared"])
        actors, positions = load_selected(prepared, plan["cases"])
        profiles = {(p["split"], p["dialogue_id"]): p["work"] for p in read(prepared / "workloads.json")["profiles"]}
        lexical = {name: np.load(prepared / f"lexical-{name}.npy", mmap_mode="r", allow_pickle=False)
                   for name in ("original", "numbers")}
        references = np.load(ROOT / "runs/sgd-state-v1/features-02/features.npy", mmap_mode="r", allow_pickle=False)
        counts, work_total, invariants = Counter(), Counter(), empty_invariants()
        cells, checkpoints = [], []
        progress.update({"counts": counts, "successful_encoder_work": work_total, "completed_cells": cells})
        max_driver = max_current = 0
        first_initial = None

        def sync_check():
            nonlocal max_driver, max_current
            torch.mps.synchronize()
            max_driver = max(max_driver, torch.mps.driver_allocated_memory())
            max_current = max(max_current, torch.mps.current_allocated_memory())
            require(max_driver <= CAPS["mps_driver_bytes"], "Sampled MPS driver cap")
            check()

        def payload_for(key, arm):
            payload = dict(actors[key])
            offset, shape = payload["lexical_offset"], payload["lexical_shape"]
            # Copy/restore lexical assembly is charged on every real timed visit.
            payload["lexical"] = np.array(lexical[arm.split("_", 1)[1]][offset:offset + int(np.prod(shape))],
                                          dtype=np.float32).reshape(shape)
            observed = workload_profile(payload)
            require(all(observed[k] == profiles[key][k] for k in observed), "Prepared actor work geometry")
            return payload

        def encode(encoder, payload, trainable, key):
            counts["encoding_attempts"] += 1
            vectors, work = encode_token_lists(encoder, payload["tokens"], **parent["tokenizer_ids"],
                trainable=trainable, chunk_tokens=CONFIG["chunk_tokens"], chunk_batch_size=CONFIG["encoder_batch"])
            counts["encoding_passes"] += 1
            work_total.update(work)
            require(all(work[k] == profiles[key][k] for k in work), "Actual encoder work")
            return vectors, work

        def forward(memory, vectors, payload):
            actor, none = build_actor(vectors, **{k: payload[k] for k in
                ("turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids", "lexical")}, memory_device="cpu")
            memory.begin_batch(actor, [{"layout": {"shape": payload["lexical_shape"]}}])
            counts["memory_forward_attempts"] += 1
            logs = memory(*actor, none_index=none)
            counts["memory_forwards"] += 1
            mask = actor[4][:, None].expand_as(logs)
            require(torch.isfinite(logs[mask]).all().item() and torch.isneginf(logs[~mask]).all().item(),
                    "Finite supported memory outputs and masked padding")
            merge_invariants(invariants, memory.audit)
            return logs

        for case in plan["cases"]:
            keys = [(case["split"], d) for d in case["dialogue_ids"]]
            for arm in ARMS:
                cell_start = time.perf_counter()
                progress.update({"case": case["case_id"], "arm": arm, "operation": "load"})
                torch.manual_seed(CONFIG["seed"])
                encoder = AutoModel.from_pretrained(parent["snapshot"], local_files_only=True,
                    use_safetensors=True, attn_implementation="eager", dtype=torch.float32).to("mps").eval()
                encoder.register_forward_pre_hook(lambda *_: counts.update({"encoder_forward_attempts": 1}))
                encoder.register_forward_hook(lambda *_: counts.update({"encoder_forward_returns": 1}))
                torch.manual_seed(CONFIG["seed"])
                memory = MonitoredCopyMemoryV2("scalar").cpu()
                initial = {"encoder": qualified.tensor_digest(encoder), "memory": qualified.tensor_digest(memory)}
                if first_initial is None:
                    first_initial = initial
                require(initial == first_initial, "All cases/arms start at identical tensors")
                cell = {"case_id": case["case_id"], "arm": arm, "kind": case["kind"], "split": case["split"],
                        "dialogue_ids": case["dialogue_ids"], "initial_sha256": initial, "parity": [], "events": []}
                progress["active_cell"] = cell
                for key in keys:
                    progress["operation"] = "initial-vector-parity"
                    parity_start = time.perf_counter()
                    payload = payload_for(key, arm)
                    vectors, work = encode(encoder, payload, False, key)
                    actual = vectors.detach().cpu().numpy()
                    expected = references[payload["original_feature_ids"]]
                    difference = float(np.max(np.abs(actual - expected)))
                    require(np.isfinite(actual).all() and difference <= CONFIG["parity_tolerance"], "Historical vector parity")
                    counts["parity_dialogue_visits"] += 1
                    cell["parity"].append({"dialogue_id": key[1], "max_abs": difference, "encoder_work": work,
                                           "wall_seconds": time.perf_counter() - parity_start})
                    del vectors, actual, expected
                    sync_check()
                trainable = arm.startswith("trainable_")
                encoder.requires_grad_(trainable)
                groups = [{"params": list(memory.parameters()), "lr": CONFIG["memory_lr"]}]
                if trainable:
                    groups.append({"params": list(encoder.parameters()), "lr": CONFIG["encoder_lr"]})
                optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])
                parameters = [p for group in groups for p in group["params"]]
                denominator = sum(len(positions[key]) for key in keys)
                for iteration in range(case["iterations"]):
                    sync_check()
                    event_start = time.perf_counter()
                    event_work, event_audit = Counter(), empty_invariants()
                    event = {"case_id": case["case_id"], "arm": arm, "iteration": iteration,
                             "phase": "eval" if case["kind"] == "eval" else "update",
                             "warm": iteration == 0 and case["kind"] in ("batch", "eval"),
                             "dialogue_visits": len(keys), "batch_endpoint_count": denominator}
                    progress.update({"operation": event["phase"], "iteration": iteration})
                    optimizer.zero_grad(set_to_none=True)
                    synthetic_loss_sum = 0.
                    for micro, key in enumerate(keys):
                        payload = payload_for(key, arm)
                        with torch.no_grad() if case["kind"] == "eval" else contextlib.nullcontext():
                            vectors, work = encode(encoder, payload, trainable and case["kind"] != "eval", key)
                            logs = forward(memory, vectors, payload)
                            if case["kind"] != "eval":
                                loss = supervised_loss(logs, synthetic_rows(payload, positions[key], iteration * len(keys) + micro),
                                                       parent["loss_weights"], denominator)
                                counts["backward_attempts"] += 1
                                loss.backward()
                                counts["backward_calls"] += 1
                                synthetic_loss_sum += float(loss.detach())
                                del loss
                                counts["training_dialogue_visits"] += 1
                            else:
                                counts["evaluation_forwards"] += 1
                        event_work.update(work)
                        merge_invariants(event_audit, memory.audit)
                        del vectors, logs
                        sync_check()
                    if case["kind"] != "eval":
                        head_norm = sum(float(p.grad.detach().double().square().sum())
                                        for p in memory.parameters() if p.grad is not None)
                        require(0 < head_norm < float("inf"), "Finite nonzero memory gradient")
                        gradient = {}
                        named = dict(encoder.named_parameters())
                        for name in ("embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight"):
                            grad = named[name].grad
                            if trainable:
                                require(grad is not None, "Encoder gradient present")
                                value = float(grad.detach().cpu().double().square().sum())
                                require(0 < value < float("inf"), "Finite nonzero encoder gradient")
                                gradient[name] = value
                            else:
                                require(grad is None, "Frozen encoder gradient absent")
                                gradient[name] = None
                        require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters), "Finite gradients")
                        norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG["clip"], error_if_nonfinite=True, foreach=False)
                        require(torch.isfinite(norm).item(), "Finite global gradient norm")
                        counts["optimizer_attempts"] += 1
                        optimizer.step()
                        counts["optimizer_updates"] += 1
                        require(all(torch.isfinite(p).all().item() for p in parameters), "Finite updated parameters")
                        if not trainable:
                            require(all(p.grad is None for p in encoder.parameters()), "All frozen gradients absent")
                        event.update({"memory_gradient_squared_norm": head_norm, "encoder_gradient_squared_norms": gradient,
                                      "synthetic_weighted_loss": synthetic_loss_sum, "gradient_norm_before_clip": float(norm)})
                    event.update({"encoder_work": dict(event_work), "invariants": event_audit})
                    sync_check()
                    event["wall_seconds"] = time.perf_counter() - event_start
                    io_start = time.perf_counter()
                    append_event(out, event)
                    event["journal_write_seconds"] = time.perf_counter() - io_start
                    cell["events"].append(event)
                final_encoder = qualified.tensor_digest(encoder)
                require((final_encoder != initial["encoder"]) == (trainable and case["kind"] != "eval"), "Encoder change/freeze witness")
                cell["final_encoder_sha256"] = final_encoder
                if case["case_id"] == "batch-0":
                    sync_check()
                    io_start = time.perf_counter()
                    state = {"memory": {k: v.detach().cpu() for k, v in memory.state_dict().items()}}
                    if trainable:
                        state["encoder"] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
                    checkpoint = out / f"checkpoint-{arm}.pt"
                    with checkpoint.open("xb") as stream:
                        torch.save(state, stream)
                    entry = {"arm": arm, "file": checkpoint.name, "sha256": sha(checkpoint),
                             "bytes": checkpoint.stat().st_size, "wall_seconds": time.perf_counter() - io_start,
                             "scope": "Synthetic qualification weights; local only, no scientific initialization"}
                    checkpoints.append(entry)
                    counts["checkpoints"] += 1
                    del state
                    sync_check()
                cell["cell_wall_seconds"] = time.perf_counter() - cell_start
                cells.append(cell)
                counts["cells"] += 1
                del encoder, memory, optimizer, groups, parameters
                if case["kind"] != "eval":
                    del named, grad
                gc.collect()
                torch.mps.empty_cache()
                sync_check()
        require({k: counts[k] for k in EXPECTED} == EXPECTED, "All frozen pilot operations complete")
        for attempted, returned in (("encoding_attempts", "encoding_passes"),
                                    ("memory_forward_attempts", "memory_forwards"),
                                    ("backward_attempts", "backward_calls"),
                                    ("optimizer_attempts", "optimizer_updates")):
            require(counts[attempted] == counts[returned], "Attempt/return coverage: " + attempted)
        require(counts["encoder_forward_attempts"] == counts["encoder_forward_returns"] == work_total["encoder_calls"],
                "All actual encoder dispatches accounted for")
        expected_work = {key: sum(profiles[(case["split"], did)][key] * (1 + case["iterations"]) * 4
                                  for case in plan["cases"] for did in case["dialogue_ids"]) for key in ENCODER_KEYS}
        require(dict(work_total) == expected_work, "All parity/training/eval encoder work")
        expected_state = sum(profiles[(case["split"], did)]["real_question_updates"] * case["iterations"] * 4
                             for case in plan["cases"] for did in case["dialogue_ids"])
        require(all(invariants[k] == expected_state for k in ("incoming_checks", "feature_checks", "result_checks", "mass_checks")),
                "Every real state step audited")
        summary = {"version": VERSION, "status": "qualified", "cells": cells, "counts": dict(counts),
                   "all_encoder_work": dict(work_total), "invariants": invariants, "checkpoints": checkpoints,
                   "sampled_mps_driver_max_bytes": max_driver, "sampled_mps_current_max_bytes": max_current,
                   "official_targets_used": False, "official_test_opened": False,
                   "scope": "Raw synthetic-target cost qualification; no task scores, projection or automatic admission",
                   "timing_scope": "Per-event assembly/encoding/monitor/loss/backward/clip/optimizer/checks; journal I/O separately measured, all charged to whole phase"}
        write(out / "summary.json", summary)
        authenticate_plan(preparation, pin)
        sync_check()
        write(out / "completed.json", {"version": VERSION, "status": "completed", "phase": "model-pilot",
              "cost_plan_sha256": sha(preparation / "plan.json"), "cost_preparation_completed_sha256": pin,
              "prepared_completed_sha256": PREPARED_PIN, "files": qualified.manifest(out),
              "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": qualified.rss(),
              "counts": dict(counts), "official_targets_used": False, "official_test_opened": False,
              "wall_scope": "Through payload manifest before completion write; cap checks continue through return"})
        result = sha(out / "completed.json")
        check()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--prepared-sha256", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--preparation-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.prepared, args.prepared_sha256, args.protocol_sha256, args.out) if args.command == "prepare" else run(
        args.preparation, args.preparation_sha256, args.out)
    print(json.dumps({"status": "completed", "completed_sha256": result}))
