"""Frozen MiniLM gradient/cost pilot with synthetic targets, never task scoring."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CONFIG = {"seed": 6801, "arms": ["frozen", "trainable"], "updates": 3,
          "warm_updates": 1, "accumulation": 4, "encoder_batch": 32,
          "chunk_tokens": 254, "encoder_lr": 2e-5, "memory_lr": 1e-3,
          "weight_decay": 1e-4, "clip": 1., "parity_tolerance": 2e-5,
          "wall_seconds": 300, "rss_bytes": 8 * 1024**3,
          "mps_driver_bytes": 8 * 1024**3, "output_bytes": 256 * 1024**2}
NEW_SOURCES = ["scripts/qualify_dialogue_finetune.py", "tests/test_qualify_dialogue_finetune.py",
    "src/openjev/research/dialogue_trainable_encoder.py", "tests/test_dialogue_trainable_encoder.py",
    "src/openjev/research/dialogue_finetune_inputs.py", "tests/test_dialogue_finetune_inputs.py",
    "research/dialogue-finetune-qualification-protocol.md"]
INPUTS = {
    "runs/dialogue-copy-v2/study-01/plan.json": "9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905",
    "runs/sgd-state-v1/features-02/completed.json": "e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308",
    "runs/sgd-state-v1/features-02/encoder-plan.json": "6d9bd36d1233db0ec24b92b2ec14079d565842b45b385d4d739f23864d3b8307",
    "runs/sgd-state-v1/features-02/packet.json": "14dd89a633bfa406394945d279c44ce6796d3b8dc412f772e2abda794ec943cb",
    "runs/sgd-state-v1/features-02/features.npy": "cec3066bf0b02b2364c32c76c78abeffcf89faef195bfa341a2d97baad44d5a1",
    "runs/sgd-state-v1/data/completed.json": "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
    "runs/sgd-state-v1/data/catalog.json": "b9bf362e454e46a524b0137aad7ebfcc8ad9b298076ce720d82937d0fb568fe5",
    "runs/sgd-state-v1/data/train-dialogues.jsonl": "04bf8797b35495d02c9a809a9b83259aefc24ff04a560f13aec989ba117b15aa",
    "runs/dialogue-copy-v1/lexical-01/index.json": "9661f99a7efc7b1aac4830b9098cdf1275811c341f467dec4f47491cf8e048ae",
    "runs/dialogue-copy-v1/lexical-01/lexical.npy": "d70dce8074fb8f5fcf8f8174cbbade325397873cc227204f73d96faa3e613591",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "tokenizers")}}


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def bind(paths, root=ROOT):
    for name, digest in paths.items():
        require(sha(root / name) == digest, "Changed input/source: " + name)


def sources():
    name = "runs/dialogue-copy-v2/study-01/plan.json"
    bind({name: INPUTS[name]})
    inherited = read(ROOT / name)["source_sha256"]
    bind(inherited)
    return {**inherited, **{p: sha(ROOT / p) for p in NEW_SOURCES}}


@contextlib.contextmanager
def attempt(out, phase, request):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": phase}

    def check():
        require(time.perf_counter() - start <= CONFIG["wall_seconds"], "Whole-phase wall cap")
        require(rss() <= CONFIG["rss_bytes"], "Process RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= CONFIG["output_bytes"],
                "Output cap")

    def timeout(*_):
        raise TimeoutError("Whole-phase wall cap")

    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, CONFIG["wall_seconds"])
    try:
        write(out / "started.json", {"phase": phase, "request": request,
                                    "config": CONFIG, "runtime": runtime()})
        yield start, progress, check
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "incomplete-completion.json")
            write(out / "failed.json", {"status": "failed", "phase": phase,
                "error_type": type(error).__name__, "error": str(error), "progress": progress,
                "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": rss(),
                "scope": "Qualification failure; no quality result or training admission"})
        except BaseException as secondary:  # noqa: BLE001 - preserve execution error
            error.add_note("Failure receipt: " + str(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def manifest(out):
    return {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())
            if p.is_file() and p.name != "completed.json"}


def prepare(out):
    with attempt(out, "prepare", {}) as (start, progress, check):
        source_map = sources()
        bind(INPUTS)
        write(out / "source-pins.json", source_map)
        import numpy as np
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        from openjev.research.dialogue_finetune_inputs import (
            build_dialogue_inputs,
            select_representatives,
            workload_profile,
        )
        inherited = read(ROOT / "runs/sgd-state-v1/features-02/encoder-plan.json")
        model_files = {}
        for name, digest in inherited["model_files_sha256"].items():
            p = Path(hf_hub_download(ENCODER, name, revision=REVISION, local_files_only=True))
            require(sha(p) == digest, "Pinned encoder file: " + name)
            model_files[name] = {"path": str(p), "sha256": digest}
        snapshot = str(Path(model_files["config.json"]["path"]).parent)
        require(all(str(Path(f["path"]).parent) == snapshot for f in model_files.values()), "One snapshot")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        memo = {}

        def tokenize(text):
            if text not in memo:
                memo[text] = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            return list(memo[text])

        packet = read(ROOT / "runs/sgd-state-v1/features-02/packet.json")
        cohort = packet["cohorts"]["train"]
        require(len(cohort) == 2017, "Complete original TRAIN cohort")
        wanted = {d["id"] for d in cohort}
        public = {}
        with (ROOT / "runs/sgd-state-v1/data/train-dialogues.jsonl").open() as f:
            for line in f:
                d = json.loads(line)
                if d["dialogue_id"] in wanted:
                    require(d["dialogue_id"] not in public, "Duplicate public dialogue")
                    public[d["dialogue_id"]] = d
        require(set(public) == wanted, "Full public dialogue coverage")
        catalog = read(ROOT / "runs/sgd-state-v1/data/catalog.json")["train"]
        entries = read(ROOT / "runs/dialogue-copy-v1/lexical-01/index.json")["cohorts"]["train"]
        layouts = {e["id"]: e for e in entries}
        require(set(layouts) == wanted, "Lexical layout cohort")
        cached_lex = np.load(ROOT / "runs/dialogue-copy-v1/lexical-01/lexical.npy", allow_pickle=False, mmap_mode="r")
        payloads, profiles = {}, []
        for d in cohort:
            layout = layouts[d["id"]]
            payload = build_dialogue_inputs(d, public[d["id"]], catalog, packet["queries"], layout, tokenize)
            shape, offset = layout["shape"], layout["offset"]
            count = int(np.prod(shape))
            expected = cached_lex[offset:offset + count].reshape(shape)
            require(np.array_equal(payload["lexical"], expected), "Original lexical observations differ")
            profiles.append({"dialogue_id": d["id"], "work": workload_profile(payload)})
            payload["lexical"] = payload["lexical"].tolist()
            payloads[d["id"]] = payload
            progress["prepared_dialogues"] = len(profiles)
            check()
        selected = select_representatives(profiles)
        require(len(selected) == 3 and len({r["dialogue_id"] for r in selected}) == 3, "Three representatives")
        write(out / "profiles.json", profiles)
        write(out / "representatives.json", [{"selection": r, "payload": payloads[r["dialogue_id"]]} for r in selected])
        plan = {"study": "dialogue-finetune-qualification-v1", "config": CONFIG, "runtime": runtime(),
            "source_sha256": source_map, "input_sha256": INPUTS, "encoder": ENCODER,
            "revision": REVISION, "snapshot": snapshot, "model_files": model_files,
            "tokenizer_ids": {"cls_id": tokenizer.cls_token_id, "sep_id": tokenizer.sep_token_id,
                              "pad_id": tokenizer.pad_token_id},
            "selected": selected, "prepared_files": {n: sha(out / n) for n in
                                                       ("profiles.json", "representatives.json")},
            "unique_tokenized_texts": len(memo), "cohort_dialogues": len(cohort),
            "scope": "Only synthetic-target implementation/cost qualification, no task scoring"}
        write(out / "plan.json", plan)
        bind(source_map)
        bind(INPUTS)
        check()
        write(out / "completed.json", {"status": "completed", "phase": "prepare",
              "plan_sha256": sha(out / "plan.json"), "files": manifest(out),
              "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": rss(),
              "encoder_calls": 0, "official_targets_used": False, "official_test_opened": False})
    return sha(out / "completed.json")


def authenticate_preparation(path, pin):
    require(sha(path / "completed.json") == pin and not (path / "failed.json").exists(), "Preparation completion")
    done = read(path / "completed.json")
    expected = {"started.json", "source-pins.json", "profiles.json", "representatives.json", "plan.json"}
    require(done["status"] == "completed" and set(done["files"]) == expected
            and {p.name for p in path.iterdir()} == expected | {"completed.json"}, "Preparation closure")
    for name, entry in done["files"].items():
        require(sha(path / name) == entry["sha256"] and (path / name).stat().st_size == entry["bytes"],
                "Preparation payload identity")
    plan = read(path / "plan.json")
    require(sha(path / "plan.json") == done["plan_sha256"], "Prepared plan pin")
    require(plan["study"] == "dialogue-finetune-qualification-v1" and plan["config"] == CONFIG
            and plan["runtime"] == runtime() and plan["input_sha256"] == INPUTS, "Frozen run configuration")
    require(plan["source_sha256"] == sources(), "Frozen source closure")
    bind(INPUTS)
    for f in plan["model_files"].values():
        require(sha(f["path"]) == f["sha256"], "Frozen model file")
    return plan


def tensor_digest(module):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((str(v.dtype), tuple(v.shape))).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore_payload(payload):
    """Restore the typed lexical actor after its authenticated JSON round trip."""
    import numpy as np

    from openjev.research.dialogue_finetune_inputs import workload_profile
    payload = {**payload, "lexical": np.asarray(payload["lexical"], dtype=np.float32)}
    workload_profile(payload)
    return payload


def synthetic_loss(logp, payload, offset):
    import torch

    from openjev.research.dialogue_finetune_inputs import synthetic_targets
    targets = torch.as_tensor(synthetic_targets(payload, offset=offset), dtype=torch.long)
    require(tuple(logp.shape[:3]) == (1, *targets.shape), "Full public T by Q loss support")
    return -logp[0].gather(-1, targets[..., None]).mean() / CONFIG["accumulation"]


def run(preparation, preparation_pin, out):
    with attempt(out, "model-pilot", {"preparation": str(preparation), "pin": preparation_pin}) as (start, progress, check):
        plan = authenticate_preparation(preparation, preparation_pin)
        import gc

        import numpy as np
        import torch
        from study_dialogue_copy_v2 import MonitoredCopyMemoryV2
        from transformers import AutoModel

        from openjev.research.dialogue_trainable_encoder import build_actor, encode_token_lists
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.backends.mps.is_available(), "MPS required; no fallback after freeze")
        representatives = read(preparation / "representatives.json")
        references = np.load(ROOT / "runs/sgd-state-v1/features-02/features.npy", mmap_mode="r", allow_pickle=False)
        cells, work_total, monitor_total = [], Counter(), Counter()
        progress.update({"cells": cells, "successful_encoder_work": work_total,
                         "monitored_state_checks": monitor_total})
        initial_memory = initial_encoder = None
        max_driver = max_current = 0

        def sync_check():
            nonlocal max_driver, max_current
            torch.mps.synchronize()
            driver, current = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
            max_driver, max_current = max(max_driver, driver), max(max_current, current)
            require(driver <= CONFIG["mps_driver_bytes"], "Sampled MPS driver allocation cap")
            check()

        def encode(encoder, payload, trainable):
            vectors, work = encode_token_lists(encoder, payload["tokens"], **plan["tokenizer_ids"],
                trainable=trainable, chunk_tokens=CONFIG["chunk_tokens"], chunk_batch_size=CONFIG["encoder_batch"])
            work_total.update(work)
            require(all(work[k] == active_work[k] for k in work), "Observed encoder work equals prepared workload")
            return vectors, work

        def actor(vectors, payload):
            return build_actor(vectors, turn_text_ids=payload["turn_text_ids"],
                query_text_ids=payload["query_text_ids"], candidate_text_ids=payload["candidate_text_ids"],
                candidate_ids=payload["candidate_ids"], lexical=payload["lexical"], memory_device="cpu")

        def forward(memory, encoded, none_index, payload):
            memory.begin_batch(encoded, [{"layout": {"shape": [len(payload["turn_text_ids"]),
                len(payload["query_text_ids"]), encoded[4].shape[-1], 10]}}])
            result = memory(*encoded, none_index=none_index)
            for k in ("incoming_checks", "feature_checks", "result_checks", "mass_checks", "valid_turns"):
                monitor_total[k] += memory.audit[k]
            return result

        for selection in representatives:
            payload = restore_payload(selection["payload"])
            active_work = selection["selection"]["work"]
            for arm in CONFIG["arms"]:
                cell_start = time.perf_counter()
                progress.update({"dialogue_id": payload["dialogue_id"], "arm": arm,
                                 "completed_cells": len(cells), "operation": "load"})
                torch.manual_seed(CONFIG["seed"])
                encoder = AutoModel.from_pretrained(plan["snapshot"], local_files_only=True,
                    use_safetensors=True, attn_implementation="eager", dtype=torch.float32).to("mps").eval()
                # Re-seed after encoder construction so its unused initialization cannot change the head.
                torch.manual_seed(CONFIG["seed"])
                memory = MonitoredCopyMemoryV2("scalar").cpu()
                enc_digest, mem_digest = tensor_digest(encoder), tensor_digest(memory)
                if initial_encoder is None:
                    initial_encoder, initial_memory = enc_digest, mem_digest
                require(enc_digest == initial_encoder and mem_digest == initial_memory, "Paired initial tensors")
                sync_check()
                progress["operation"] = "pre-update-parity"
                vectors, parity_work = encode(encoder, payload, False)
                actual = vectors.detach().cpu().numpy()
                expected = references[payload["original_feature_ids"]]
                parity = float(np.max(np.abs(actual - expected)))
                require(np.isfinite(actual).all() and parity <= CONFIG["parity_tolerance"], "Historical pooled-vector parity")
                progress["active_parity"] = {"max_abs": parity, "encoder_work": parity_work}
                del vectors, actual, expected
                trainable = arm == "trainable"
                encoder.requires_grad_(trainable)
                groups = [{"params": list(memory.parameters()), "lr": CONFIG["memory_lr"]}]
                if trainable:
                    groups.append({"params": list(encoder.parameters()), "lr": CONFIG["encoder_lr"]})
                optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])
                updates = []
                progress["active_updates"] = updates
                for update in range(CONFIG["updates"]):
                    sync_check()
                    update_start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    work_update = Counter()
                    for micro in range(CONFIG["accumulation"]):
                        progress.update({"operation": "encode/backward", "update": update, "microbatch": micro})
                        vectors, work = encode(encoder, payload, trainable)
                        encoded, none = actor(vectors, payload)
                        logp = forward(memory, encoded, none, payload)
                        loss = synthetic_loss(logp, payload, offset=update * CONFIG["accumulation"] + micro)
                        require(torch.isfinite(loss).item(), "Finite synthetic loss")
                        loss.backward()
                        work_update.update(work)
                        del vectors, encoded, none, logp, loss
                        sync_check()
                    progress["operation"] = "gradient-check/optimizer"
                    head_grad = sum(float(p.grad.detach().double().square().sum()) for p in memory.parameters() if p.grad is not None)
                    require(0 < head_grad < float("inf"), "Finite nonzero memory gradient")
                    gradient_witnesses = {}
                    named = dict(encoder.named_parameters())
                    for name in ("embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight"):
                        grad = named[name].grad
                        if trainable:
                            require(grad is not None, "Missing cross-device encoder gradient: " + name)
                            value = float(grad.detach().cpu().double().square().sum())
                            require(0 < value < float("inf"), "Finite nonzero encoder gradient: " + name)
                            gradient_witnesses[name] = value
                        else:
                            require(grad is None, "Frozen encoder acquired gradients")
                            gradient_witnesses[name] = None
                    parameters = [p for group in optimizer.param_groups for p in group["params"]]
                    # PyTorch clips CPU and MPS gradients together, with total norm on the first device.
                    norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG["clip"], error_if_nonfinite=True, foreach=False)
                    require(torch.isfinite(norm).item(), "Finite global gradient norm")
                    optimizer.step()
                    require(all(torch.isfinite(p).all().item() for p in parameters), "Finite updated parameters")
                    if not trainable:
                        require(all(p.grad is None for p in encoder.parameters()), "All frozen encoder gradients absent")
                    sync_check()
                    record = {"update": update, "warm": update < CONFIG["warm_updates"],
                        "wall_seconds": time.perf_counter() - update_start, "dialogue_visits": 4,
                        "memory_gradient_squared_norm": head_grad, "encoder_gradient_squared_norms": gradient_witnesses,
                        "encoder_work": dict(work_update), "sampled_mps_driver_bytes": torch.mps.driver_allocated_memory()}
                    updates.append(record)
                    with (out / "events.jsonl").open("a") as f:
                        f.write(json.dumps({"dialogue_id": payload["dialogue_id"], "arm": arm, **record}, allow_nan=False) + "\n")
                progress["operation"] = "evaluation"
                sync_check()
                eval_start = time.perf_counter()
                with torch.no_grad():
                    vectors, eval_work = encode(encoder, payload, False)
                    encoded, none = actor(vectors, payload)
                    evaluation = forward(memory, encoded, none, payload)
                    require(not torch.isnan(evaluation).any().item(), "Evaluation finite supported output")
                sync_check()
                eval_seconds = time.perf_counter() - eval_start
                final_encoder_digest = tensor_digest(encoder)
                if not trainable:
                    require(final_encoder_digest == enc_digest, "Frozen encoder weights changed")
                else:
                    require(final_encoder_digest != enc_digest, "Trainable encoder weights unchanged")
                cells.append({"selection": selection["selection"], "arm": arm, "updates": updates,
                    "initial_encoder_sha256": enc_digest, "initial_memory_sha256": mem_digest,
                    "final_encoder_sha256": final_encoder_digest, "parity_max_abs": parity,
                    "parity_encoder_work": parity_work, "evaluation_encoder_work": eval_work,
                    "evaluation_seconds": eval_seconds, "cell_wall_seconds": time.perf_counter() - cell_start})
                del encoder, memory, optimizer, groups, parameters, named, evaluation, vectors, encoded, none
                gc.collect()
                torch.mps.empty_cache()
                sync_check()
        require(len(cells) == 6, "All six workload/arm cells")
        expected_work = {k: 28 * sum(r["selection"]["work"][k] for r in representatives) for k in work_total}
        require(dict(work_total) == expected_work, "All 84 encoder passes accounted for")
        for k in ("incoming_checks", "feature_checks", "result_checks", "mass_checks"):
            require(monitor_total[k] == 26 * sum(r["selection"]["work"]["real_question_updates"]
                                               for r in representatives), "All 78 state forwards accounted for")
        summary = {"status": "qualified", "cells": cells, "optimizer_updates": 18, "dialogue_visits": 72,
            "all_encoder_work": dict(work_total), "monitored_state_checks": dict(monitor_total),
            "sampled_mps_driver_max_bytes": max_driver, "sampled_mps_current_max_bytes": max_current,
            "scope": "Three workload qualification only; synthetic targets, no task quality, no automatic scientific admission"}
        write(out / "summary.json", summary)
        authenticate_preparation(preparation, preparation_pin)
        sync_check()
        progress["operation"] = "close"
        write(out / "completed.json", {"status": "completed", "phase": "model-pilot",
            "preparation_completed_sha256": preparation_pin, "plan_sha256": sha(preparation / "plan.json"),
            "files": manifest(out), "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": rss(),
            "completed_cells": 6, "optimizer_updates": 18, "dialogue_visits": 72,
            "official_targets_used": False, "official_test_opened": False,
            "weights_saved": False, "scope": summary["scope"]})
    return sha(out / "completed.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--preparation-sha256", required=True)
    args = parser.parse_args()
    result = prepare(args.out) if args.command == "prepare" else run(args.preparation, args.preparation_sha256, args.out)
    print(json.dumps({"status": "completed", "completed_sha256": result}))
