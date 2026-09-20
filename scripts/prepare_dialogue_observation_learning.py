"""Prepare complete actor streams, separate targets and cost metadata, without neural calls."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import signal
import time
from collections import Counter
from pathlib import Path

import qualify_dialogue_finetune as qualified

ROOT = qualified.ROOT
STUDY = "dialogue-observation-learning-v1"
CONFIG = {"arms": ["frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers"],
          "seeds": [6901, 6902, 6903], "epochs": 20, "effective_batch": 32,
          "microbatch": 1, "encoder_batch": 32, "chunk_tokens": 254,
          "memory_lr": .001, "encoder_lr": .00002, "weight_decay": .0001, "clip": 1.,
          "memory_method": "scalar", "dropout": False, "cpu_threads": 1,
          "loss": "sum(weight[stratum] * endpoint NLL) / eligible endpoints in effective batch",
          "selection": "All final fits; no epoch, seed or variant selection"}
CAPS = {"wall_seconds": 300, "rss_bytes": 8 * 1024**3, "output_bytes": 512 * 1024**2}
INPUTS = {**qualified.INPUTS,
    "runs/sgd-state-v1/data/dev-dialogues.jsonl": "1309f6e5a8c511fe4c703a336337798cd221b1f3b0f3b9a234bf557bb2227be7",
    "output/dialogue-finetune-qualification-v1/preparation-01/plan.json": "f0e35e69f5c614e02a516edfa5bd049f3fda3beff48ed8c221b75907a0adee83",
    "output/dialogue-finetune-qualification-v1/pilot-01/completed.json": "a0f91d8f80a6edab85ef5a8e9eff1d989ab53b7605f9fd59a7ee9343f8dd0250",
    "output/dialogue-finetune-qualification-v1/pilot-audit-01.json": "eed44eaa50d5e95713a45f82c1c8437d8aacf6bda80cc8dc18db623d1eca8584"}
SOURCES = ["scripts/prepare_dialogue_observation_learning.py", "tests/test_prepare_dialogue_observation_learning.py",
           "src/openjev/research/dialogue_finetune_dataset.py", "tests/test_dialogue_finetune_dataset.py",
           "src/openjev/research/dialogue_number_lexical.py", "tests/test_dialogue_number_lexical.py",
           "research/dialogue-observation-learning-preparation.md"]
require, sha, read, write, bind = qualified.require, qualified.sha, qualified.read, qualified.write, qualified.bind


@contextlib.contextmanager
def attempt(out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "prepare", "dialogues": 0, "encoder_calls": 0}

    def check():
        require(time.perf_counter() - start <= CAPS["wall_seconds"], "Preparation wall cap")
        require(qualified.rss() <= CAPS["rss_bytes"], "Preparation RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= CAPS["output_bytes"], "Preparation output cap")

    def timeout(*_):
        raise TimeoutError("Preparation wall cap")

    handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, CAPS["wall_seconds"])
    try:
        write(out / "started.json", {"study": STUDY, "config": CONFIG, "caps": CAPS, "runtime": qualified.runtime()})
        yield start, progress, check
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "invalid-completion.json")
            write(out / "failed.json", {"status": "failed", "error": repr(error), "progress": progress,
                  "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": qualified.rss()})
        except BaseException as secondary:  # noqa: BLE001 - retain original execution error
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def public_subset(path, wanted):
    found = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            did = row["dialogue_id"]
            if did in wanted:
                require(did not in found, "Duplicate public dialogue")
                found[did] = row
    require(set(found) == set(wanted), "Complete public cohort")
    return found


def inherited_sources(prior):
    sources = qualified.sources()
    require(sources == prior["source_sha256"], "All inherited sources must match passed qualification")
    return sources


def build_lexical_pair(payload, public, catalog, packet_queries):
    import numpy as np

    from openjev.research.dialogue_number_lexical import lexical_observations

    normalized = np.zeros_like(payload["lexical"])
    registers = {"original": [], "numbers": []}
    by_id = {q["query_id"]: q for q in catalog}
    require(len(by_id) == len(catalog), "Unique public schemas")
    for j, qi in enumerate(payload["query_ids"]):
        ids = payload["candidate_ids"][j]
        n = len(ids)
        values, literal = lexical_observations(public, by_id[packet_queries[qi]["id"]], ids)
        require(values.shape == (len(payload["turn_text_ids"]), n, 10), "Normalized lexical shape")
        normalized[:, j, :n] = values
        original = payload["lexical"][:, j, :n, 4]
        require(np.all(original.sum(-1) == 1), "One original literal state")
        registers["original"].append(original.argmax(-1).tolist())
        registers["numbers"].append(literal.tolist())
    # Registers are deterministic public baselines and stay outside actor maps.
    return normalized, registers


def effective_batches(orders, profiles):
    """Account exact effective-batch membership, including each one-dialogue tail."""
    by_id = {p["dialogue_id"]: p["work"] for p in profiles if p["split"] == "train"}
    dids = orders["dialogue_ids"]
    require(len(by_id) == len(dids) and set(by_id) == set(dids), "Training workload/order join")
    metrics = ("encoder_calls", "padded_token_positions", "padded_attention_positions",
               "real_question_updates", "padded_candidate_positions", "encoder_sequences")
    maxima = {}
    counts = Counter()
    for seed in CONFIG["seeds"]:
        for epoch, order in enumerate(orders["orders"][str(seed)]):
            require(sorted(order) == list(range(len(dids))), "Complete epoch permutation")
            for start in range(0, len(order), CONFIG["effective_batch"]):
                indices = order[start:start + CONFIG["effective_batch"]]
                counts[str(len(indices))] += 1
                for metric in metrics:
                    total = sum(by_id[dids[i]][metric] for i in indices)
                    if metric not in maxima or total > maxima[metric]["value"]:
                        maxima[metric] = {"value": total, "seed": seed, "epoch": epoch,
                                          "batch_start": start, "dialogue_indices": indices,
                                          "dialogue_ids": [dids[i] for i in indices]}
    return {"scope": "Membership shared by four arms; counts below are before multiplying by arms",
            "sizes": dict(counts), "batches": sum(counts.values()), "maxima": maxima}


def prepare(out):
    with attempt(out) as (start, progress, check):
        bind(INPUTS)
        prior = read(ROOT / "output/dialogue-finetune-qualification-v1/preparation-01/plan.json")
        source_map = {**inherited_sources(prior), **{name: sha(ROOT / name) for name in SOURCES}}
        write(out / "source-pins.json", source_map)
        import numpy as np
        from transformers import AutoTokenizer

        from openjev.research.dialogue_finetune_dataset import (
            aggregate_work_profiles,
            assert_lexical_parity,
            build_actor_payload,
            build_loss_rows,
            paired_epoch_orders,
        )

        for entry in prior["model_files"].values():
            require(sha(entry["path"]) == entry["sha256"], "Pinned model/tokenizer asset")
        tokenizer = AutoTokenizer.from_pretrained(prior["snapshot"], local_files_only=True)
        memo = {}

        def tokenize(text):
            if text not in memo:
                memo[text] = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            return list(memo[text])

        packet = read(ROOT / "runs/sgd-state-v1/features-02/packet.json")
        catalog = read(ROOT / "runs/sgd-state-v1/data/catalog.json")
        layouts = read(ROOT / "runs/dialogue-copy-v1/lexical-01/index.json")["cohorts"]
        original = np.load(ROOT / "runs/dialogue-copy-v1/lexical-01/lexical.npy", mmap_mode="r", allow_pickle=False)
        require(original.dtype == np.float32 and original.ndim == 1, "Original lexical cache")
        require({k: len(packet["cohorts"][k]) for k in ("train", "dev")} == {"train": 2017, "dev": 2363}, "Complete fixed cohorts")
        arrays = {arm: np.lib.format.open_memmap(out / f"lexical-{arm}.npy", mode="w+", dtype=np.float32,
                                               shape=original.shape) for arm in ("original", "numbers")}
        counts = {split: Counter() for split in ("train", "dev")}
        index, changed_by_feature = [], np.zeros(10, dtype=np.int64)
        offset = 0

        def produce():
            nonlocal offset
            for split in ("train", "dev"):
                cohort = packet["cohorts"][split]
                public = public_subset(ROOT / f"runs/sgd-state-v1/data/{split}-dialogues.jsonl", {d["id"] for d in cohort})
                require(len(layouts[split]) == len(cohort), "Complete layout cohort")
                with (out / f"actors-{split}.jsonl").open("x") as actors, (out / f"targets-{split}.jsonl").open("x") as targets:
                    for d, layout in zip(cohort, layouts[split], strict=True):
                        payload = build_actor_payload(d, public[d["id"]], catalog[split], packet["queries"], layout, tokenize, split=split)
                        shape, count = list(payload["lexical"].shape), int(payload["lexical"].size)
                        require(layout["offset"] == offset and layout["id"] == d["id"], "Original ordered lexical closure")
                        assert_lexical_parity(payload, original[offset:offset + count].reshape(shape))
                        normalized, registers = build_lexical_pair(payload, public[d["id"]], catalog[split], packet["queries"])
                        require(np.array_equal(normalized[..., 6:], payload["lexical"][..., 6:]), "Reserved and Boolean features unchanged")
                        changed_by_feature[:] += np.count_nonzero(normalized != payload["lexical"], axis=(0, 1, 2))
                        arrays["original"][offset:offset + count] = payload["lexical"].reshape(-1)
                        arrays["numbers"][offset:offset + count] = normalized.reshape(-1)
                        rows = build_loss_rows(d, payload, packet["queries"])
                        counts[split].update(r["stratum_index"] for r in rows)
                        actor = {k: v for k, v in payload.items() if k != "lexical"}
                        actor.update({"lexical_offset": offset, "lexical_shape": shape})
                        actors.write(json.dumps(actor, separators=(",", ":"), allow_nan=False) + "\n")
                        targets.write(json.dumps({"split": split, "dialogue_id": d["id"], "rows": rows,
                                                  "literal_registers": registers}, separators=(",", ":"), allow_nan=False) + "\n")
                        index.append({"split": split, "dialogue_id": d["id"], "shape": shape, "offset": offset,
                                      "scored_rows": len(rows)})
                        offset += count
                        progress["dialogues"] += 1
                        check()
                        yield payload

        profiles = aggregate_work_profiles(produce())
        require(offset == original.size, "Every lexical position accounted for")
        require(sum(counts["train"].values()) == 51741 and sum(counts["dev"].values()) == 62329, "All original scored endpoints")
        for array in arrays.values():
            array.flush()
        arrays.clear()
        ids = [d["id"] for d in packet["cohorts"]["train"]]
        orders = paired_epoch_orders(ids, CONFIG["seeds"], CONFIG["epochs"])
        batches = effective_batches(orders, profiles["profiles"])
        require(batches["batches"] == 3840, "1280 updates per seed before four arms")
        write(out / "workloads.json", profiles)
        write(out / "orders.json", orders)
        write(out / "index.json", index)
        write(out / "effective-batches.json", batches)
        weights = [51741 / (3 * counts["train"][i]) for i in range(3)]
        old = read(ROOT / "runs/dialogue-copy-v2/study-01/plan.json")
        require(weights == old["loss_weights"], "Historical three-stratum weights unchanged")
        plan = {"study": STUDY, "config": CONFIG, "caps": CAPS, "runtime": qualified.runtime(),
                "source_sha256": source_map, "input_sha256": INPUTS,
                "encoder": qualified.ENCODER, "revision": qualified.REVISION,
                "model_files": prior["model_files"], "snapshot": prior["snapshot"],
                "tokenizer_ids": prior["tokenizer_ids"], "loss_counts": counts, "loss_weights": weights,
                "unique_tokenized_texts": len(memo), "cohort_sizes": {"train": 2017, "dev": 2363},
                "updates_per_fit": math.ceil(2017 / 32) * 20, "fits": 12,
                "training_dialogue_visits": 2017 * 20 * 12,
                "training_supervised_presentations": 51741 * 20 * 12,
                "lexical_changed_positions_by_feature": changed_by_feature.tolist(),
                "scope": "Complete input/work preparation only; cost admission and scientific execution not authorized by this receipt",
                "data_files": qualified.manifest(out)}
        write(out / "plan.json", plan)
        bind(source_map)
        bind(INPUTS)
        check()
        write(out / "completed.json", {"status": "completed", "study": STUDY, "phase": "prepare",
            "plan_sha256": sha(out / "plan.json"), "files": qualified.manifest(out),
            "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": qualified.rss(),
            "encoder_calls": 0, "model_weights_loaded": False, "official_test_opened": False,
            "scope": plan["scope"]})
    return sha(out / "completed.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "completed", "completed_sha256": prepare(args.out)}))
