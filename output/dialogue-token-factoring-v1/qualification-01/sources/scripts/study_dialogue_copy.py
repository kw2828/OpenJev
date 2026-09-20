"""Fixed development study of value-preserving categorical transitions.

Questions are independent streams, packed once per dialogue to avoid redundant
prefix replay. Gold availability only masks loss, never a recurrent update.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from prepare_dialogue_copy import authenticate, sha, write
from study_dialogue_memory import load_packet, loss_bin, reference_predictions, tensor_digest
from torch import nn

from openjev.research.dialogue_copy_memory import DialogueCopyMemory

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["readout", "scalar", "selective", "selective_no_lexical", "candidate_gru"]
SEEDS = [4101, 4102, 4103]
CONFIG = {"methods": METHODS, "seeds": SEEDS, "epochs": 20, "batch_size": 32,
          "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1., "threads": 4,
          "projection_dim": 64, "hidden_dim": 64, "gru_width": 16,
          "loss": "Equal training weight across three state-transition strata",
          "selection": "All final fits; no best epoch, seed or variant; development only"}
PRACTICAL_CHECKS = {"panels": ["seen", "unseen"], "primary": "selective",
                    "macro_gain_over_literal": .03, "macro_gain_over_scalar": .005,
                    "micro_nll_no_worse_than": "scalar", "strict_paired_macro_wins_over_scalar": 2,
                    "conventional_controls": ["readout", "candidate_gru"],
                    "maximum_revision_deficit_to_literal": .01,
                    "required_complete_fits": 15, "total_panel_checks": 12}
SOURCES = ["scripts/study_dialogue_copy.py", "scripts/prepare_dialogue_copy.py",
           "scripts/report_dialogue_copy.py", "src/openjev/research/dialogue_copy_features.py",
           "src/openjev/research/dialogue_copy_memory.py", "tests/test_dialogue_copy_features.py",
           "tests/test_dialogue_copy_memory.py", "tests/test_study_dialogue_copy.py",
           "tests/test_report_dialogue_copy.py", "research/dialogue-copy-protocol.md",
           "scripts/study_dialogue_memory.py", "scripts/report_dialogue_memory.py",
           "src/openjev/research/dialogue_carry.py"]


def runtime():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform()}


def failure(out, error, details):
    try:
        write(Path(out) / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                         "error": str(error), **details})
    except BaseException as secondary:  # noqa: BLE001 - preserve original exception
        error.add_note("Failure receipt: " + str(secondary))


def load_inputs(packet_path, lexical_path):
    packet, features = load_packet(packet_path)
    lexical_path = Path(lexical_path)
    receipt = authenticate(lexical_path, sha(lexical_path / "completed.json"))
    if receipt["packet_completed_sha256"] != sha(Path(packet_path) / "completed.json"):
        raise ValueError("Lexical/encoder packet mismatch")
    for path, digest in receipt["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Lexical preparation source changed")
    index = json.loads((lexical_path / "index.json").read_text())
    lexical = np.load(lexical_path / "lexical.npy", allow_pickle=False, mmap_mode="r")
    if lexical.ndim != 1 or lexical.dtype != np.float32 or lexical.size != receipt["float32_scalars"]:
        raise ValueError("Lexical array shape/dtype")
    cohorts = {}
    expected_offset = 0
    for split in ("train", "dev"):
        rows = []
        if len(index["cohorts"][split]) != len(packet["cohorts"][split]):
            raise ValueError("Lexical cohort membership")
        for d, entry in zip(packet["cohorts"][split], index["cohorts"][split], strict=True):
            qids = sorted({r["query"] for r in d["queries"]})
            c = max(len(packet["queries"][qi]["candidates"]) for qi in qids)
            shape = [len(d["turns"]), len(qids), c, 10]
            if (entry["id"] != d["id"] or entry["query_ids"] != qids
                    or entry["shape"] != shape or entry["offset"] != expected_offset):
                raise ValueError("Lexical/actor alignment")
            expected_offset += math.prod(shape)
            rows.append({**d, "layout": entry})
        cohorts[split] = rows
    if expected_offset != lexical.size:
        raise ValueError("Unaccounted lexical entries")
    return cohorts, packet["queries"], features, lexical


def make_batch(ds, queries, features, lexical):
    b = len(ds)
    t, q, c = (max(d["layout"]["shape"][i] for d in ds) for i in range(3))
    turns = np.zeros((b, t, 384), np.float32)
    valid = np.zeros((b, t), bool)
    qe, ce = np.zeros((b, q, 384), np.float32), np.zeros((b, q, c, 384), np.float32)
    mask = np.zeros((b, q, c), bool)
    lex = np.zeros((b, t, q, c, 10), np.float32)
    labels = np.full((b, t, q), -100, np.int64)
    bins = np.zeros((b, t, q), np.int64)
    for i, d in enumerate(ds):
        layout = d["layout"]
        dt, dq, dc, _ = layout["shape"]
        turns[i, :dt] = features[d["turns"]]
        valid[i, :dt] = True
        offset = layout["offset"]
        lex[i, :dt, :dq, :dc] = lexical[offset:offset + dt * dq * dc * 10].reshape(dt, dq, dc, 10)
        positions = {qi: j for j, qi in enumerate(layout["query_ids"])}
        for qi, j in positions.items():
            entry = queries[qi]
            n = len(entry["candidates"])
            qe[i, j] = features[entry["text"]]
            ce[i, j, :n] = features[entry["candidates"]]
            mask[i, j, :n] = True
        mask[i, dq:, 0] = True
        for row in d["queries"]:
            j = positions[row["query"]]
            ti = row["time"]
            if type(ti) is not int or not 0 <= ti < dt:
                raise ValueError("Invalid target time")
            label = row["label"]
            if type(label) is not int or not 0 <= label < len(queries[row["query"]]["candidates"]):
                raise ValueError("Invalid target label")
            if labels[i, ti, j] != -100:
                raise ValueError("Duplicated or future target")
            labels[i, ti, j] = label
            bins[i, ti, j] = loss_bin(row["bin"])
    return [torch.from_numpy(x) for x in (turns, valid, qe, ce, mask, lex)], torch.from_numpy(labels), torch.from_numpy(bins)


def work_counts(ds, actor):
    b, t = actor[1].shape
    q, c = actor[4].shape[1:]
    return {"real_turns": int(actor[1].sum()), "padded_turn_positions": b * t,
            "padded_query_positions": b * t * q, "padded_candidate_positions": b * t * q * c,
            "real_question_steps": sum(d["layout"]["shape"][0] * d["layout"]["shape"][1] for d in ds)}


def freeze(args):
    ds, _, _, _ = load_inputs(args.packet, args.lexical)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    counts = Counter(loss_bin(r["bin"]) for d in ds["train"] for r in d["queries"])
    if set(counts) != {0, 1, 2}:
        raise ValueError("Missing training stratum")
    total = sum(counts.values())
    plan = {"study": "dialogue-copy-v1", "config": CONFIG, "practical_checks": PRACTICAL_CHECKS,
            "runtime": runtime(), "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
            "packet_completed_sha256": sha(Path(args.packet) / "completed.json"),
            "lexical_completed_sha256": sha(Path(args.lexical) / "completed.json"),
            "loss_counts": dict(counts), "loss_weights": [total / (3 * counts[i]) for i in range(3)],
            "scope": "New architecture comparison on previously exposed development queries; not confirmation",
            "optimizer_updates_per_fit": math.ceil(len(ds["train"]) / CONFIG["batch_size"]) * CONFIG["epochs"]}
    write(out / "plan.json", plan)
    print(json.dumps({"plan_sha256": sha(out / "plan.json")}), flush=True)


def validate(args):
    out = Path(args.out)
    if sha(out / "plan.json") != args.plan_sha256:
        raise ValueError("Plan digest")
    plan = json.loads((out / "plan.json").read_text())
    if plan["config"] != CONFIG or plan["practical_checks"] != PRACTICAL_CHECKS or plan["runtime"] != runtime():
        raise ValueError("Frozen configuration/runtime changed")
    if set(plan["source_sha256"]) != set(SOURCES):
        raise ValueError("Source closure changed")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed: " + name)
    for folder, key in ((args.packet, "packet_completed_sha256"), (args.lexical, "lexical_completed_sha256")):
        if sha(Path(folder) / "completed.json") != plan[key]:
            raise ValueError("Input receipt changed")
    return plan


def evaluate(model, ds, queries, features, lexical, destination):
    model.eval()
    arrays = {k: [] for k in ("probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query")}
    started = time.perf_counter()
    actor_work = Counter()
    with torch.inference_mode():
        for start in range(0, len(ds), CONFIG["batch_size"]):
            subset = ds[start:start + CONFIG["batch_size"]]
            actor, _, _ = make_batch(subset, queries, features, lexical)
            actor_work.update(work_counts(subset, actor))
            probabilities = model(*actor).softmax(-1).numpy()
            for i, d in enumerate(subset):
                positions = {qi: j for j, qi in enumerate(d["layout"]["query_ids"])}
                for row in d["queries"]:
                    n = len(queries[row["query"]]["candidates"])
                    p = probabilities[i, row["time"], positions[row["query"]], :n]
                    if not np.isfinite(p).all() or not np.isclose(p.sum(), 1., atol=1e-5):
                        raise ValueError("Invalid probabilities")
                    padded = np.zeros(12, np.float32)
                    padded[:n] = p
                    for key, value in (("probabilities", padded), ("labels", row["label"]),
                                       ("choice", int(p.argmax())), ("bin", row["bin"]),
                                       ("unseen", row["unseen"]), ("dialogue", d["id"]),
                                       ("time", row["time"]), ("query", row["query"])):
                        arrays[key].append(value)
    np.savez_compressed(destination, **{k: np.asarray(v) for k, v in arrays.items()})
    return {"queries": len(arrays["labels"]), "wall_seconds": time.perf_counter() - started,
            "actor_shapes": dict(actor_work), "scope": "Batch assembly and output storage included; encoder excluded"}


def train(args):
    plan = validate(args)
    cohorts, queries, features, lexical = load_inputs(args.packet, args.lexical)
    root = Path(args.out)
    (root / "fits").mkdir(exist_ok=False)
    torch.set_num_threads(CONFIG["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = cohorts["train"]
    weight = torch.tensor(plan["loss_weights"], dtype=torch.float32)
    started = time.perf_counter()
    records, progress = [], {}
    try:
        references = reference_predictions(cohorts["dev"], queries, root / "references.npz")
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            orders = [rng.permutation(len(data)) for _ in range(CONFIG["epochs"])]
            paired = None
            for method in METHODS:
                progress = {"method": method, "seed": seed, "completed_optimizer_steps": 0,
                            "completed_supervised_queries": 0, "completed_epoch_losses": [],
                            "actor_shapes_attempted": {}}
                active = root / "fits" / f"{method}-{seed}"
                active.mkdir(exist_ok=False)
                torch.manual_seed(seed)
                model = DialogueCopyMemory(method, projection_dim=CONFIG["projection_dim"],
                                           hidden_dim=CONFIG["hidden_dim"], gru_width=CONFIG["gru_width"])
                initial = tensor_digest(model)
                if method in ("scalar", "selective", "selective_no_lexical"):
                    if paired is None:
                        paired = initial
                    if initial != paired:
                        raise ValueError("Unpaired categorical transport initialization")
                optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                              weight_decay=CONFIG["weight_decay"])
                losses, updates, native_queries = [], 0, 0
                actor_work = Counter()
                fit_start = time.perf_counter()
                model.train()
                for epoch, order in enumerate(orders):
                    loss_sum, count = 0., 0
                    for start in range(0, len(data), CONFIG["batch_size"]):
                        subset = [data[int(i)] for i in order[start:start + CONFIG["batch_size"]]]
                        actor, labels, bins = make_batch(subset, queries, features, lexical)
                        actor_work.update(work_counts(subset, actor))
                        progress["actor_shapes_attempted"] = dict(actor_work)
                        optimizer.zero_grad(set_to_none=True)
                        scores = model(*actor)
                        eligible = labels != -100
                        ce = nn.functional.cross_entropy(scores[eligible], labels[eligible], reduction="none")
                        loss = (ce * weight[bins[eligible]]).mean()
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
                        optimizer.step()
                        n = int(eligible.sum())
                        native_queries += n
                        updates += 1
                        progress["completed_optimizer_steps"] = updates
                        progress["completed_supervised_queries"] = native_queries
                        count += n
                        loss_sum += float(loss.detach()) * n
                    losses.append(loss_sum / count)
                    progress["completed_epoch_losses"] = list(losses)
                    print(json.dumps({"method": method, "seed": seed, "epoch": epoch + 1,
                                      "training_loss": losses[-1], "seconds": time.perf_counter() - fit_start}), flush=True)
                train_seconds = time.perf_counter() - fit_start
                torch.save(model.state_dict(), active / "weights.pt")
                evaluation = evaluate(model, cohorts["dev"], queries, features, lexical,
                                      active / "dev-predictions.npz")
                row = {"method": method, "seed": seed, "status": "completed", "epochs": len(losses),
                       "updates": updates, "training_queries": native_queries, "training_losses": losses,
                       "training_actor_shapes": dict(actor_work), "initial_tensors_sha256": initial,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "parameters_with_final_gradient": sum(p.numel() for p in model.parameters() if p.grad is not None),
                       "configuration": model.configuration(), "train_wall_seconds": train_seconds,
                       "evaluation": evaluation, "weights_sha256": sha(active / "weights.pt"),
                       "predictions_sha256": sha(active / "dev-predictions.npz")}
                write(active / "completed.json", row)
                records.append(row)
        validate(args)
        write(root / "completed.json", {"status": "completed", "fit_count": len(records), "fits": records,
              "plan_sha256": args.plan_sha256, "references": references,
              "wall_seconds": time.perf_counter() - started, "external_model_api_calls": 0,
              "scope": "All15 final fits on exposed development; test untouched; no model selection"})
    except BaseException as error:
        failure(root, error, {"completed_fits": records, "active_progress": progress,
                             "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "train"))
    for name in ("packet", "lexical", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"freeze": freeze, "train": train}[args.mode](args)
