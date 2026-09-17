"""Prospective recovery of candidate study after recorded stdout-pipe failure.

The exact-delta arm tests native one-ply consequences, not learned dynamics.
All final fits precede neural evaluation; report audits saved evidence only.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import time
from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import torch
from torch.nn import functional as F

from openjev.research import chess_candidate_arena as arena
from openjev.research import chess_candidate_attempt as attempt
from openjev.research import chess_candidate_data as data
from openjev.research.chess_anchor_eval import probe
from openjev.research.chess_candidate import ARMS, CandidateChess
from openjev.research.chess_candidate_eval import (
    CachedPositions,
    _diagnostic,
    audit_cache_metadata,
    audit_diagnostic,
    audit_predictions,
    evaluate,
)
from openjev.research.chess_spatial_baselines import simple_baselines

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "runs/chess-inputs/stockfish/stockfish-macos-universal"
PREFLIGHTS = ["evidence/chess-candidate-v1/preflight.json", "evidence/chess-candidate-v1/preflight-128.json"]
SOURCES = [
    "scripts/chess_candidate_recovery.py", "src/openjev/research/chess_candidate_attempt.py",
    "tests/test_chess_candidate_attempt.py",
    "scripts/chess_candidate_study.py", "tests/test_chess_candidate_study.py",
    "tests/test_chess_candidate_integration.py",
    "scripts/preflight_chess_candidate.py", "scripts/preflight_chess_candidate_128.py",
    "src/openjev/research/chess_candidate.py", "tests/test_chess_candidate.py",
    "src/openjev/research/chess_candidate_data.py", "tests/test_chess_candidate_data.py",
    "src/openjev/research/chess_candidate_eval.py", "tests/test_chess_candidate_eval.py",
    "src/openjev/research/chess_candidate_arena.py", "tests/test_chess_candidate_arena.py",
    "src/openjev/research/chess_capacity_data.py", "src/openjev/research/chess_capacity_arena.py",
    "src/openjev/research/chess_anchor.py", "src/openjev/research/chess_anchor_data.py",
    "src/openjev/research/chess_anchor_eval.py", "src/openjev/research/chess_spatial.py",
    "src/openjev/research/chess_spatial_data.py", "src/openjev/research/chess_spatial_baselines.py",
    "src/openjev/research/chess_arena.py", "src/openjev/research/chess_compute.py",
    "scripts/chess_compute_study.py",
]
PROTOCOL = {
    "version": "chess-candidate-v2", "arms": list(ARMS), "width": 32,
    "seeds": [97, 109, 127], "depth": 4, "branch_depth": 2,
    "train_examples": 32768, "epochs": 6, "batch_size": 128, "microbatch_size": 128,
    "candidate_chunk_size": 128, "updates_per_fit": 1536,
    "training_device": "mps", "evaluation_device": "cpu", "torch_threads": 2,
    "deterministic_algorithms": False,
    "optimizer": {"name": "Adam", "lr": 0.001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0},
    "gradient_clip": 1., "value_loss_weight": .5,
    "loss": "Legal-candidate CE + 0.5 root-value MSE once per position; unused auxiliary decoder.",
    "batch_order_seed_base": 11300029, "fit_order_seed": 11300017,
    "regret_positions_per_split": 128, "regret_selection_seed": 11300043,
    "regret_nodes": 20000, "regret_call_ceiling": 3328, "regret_node_ceiling": 66560000,
    "latency_positions_per_split": 64, "latency_selection_seed": 11300059, "latency_warmups": 3,
    "bootstrap_replicates": 2000, "bootstrap_seed": 11300071,
    "permutation_seed": 11300083, "permutation_scope": "Delta only, fixed nonidentity within-root cyclic shift; singleton menus unchanged and reported separately; exploratory corruption diagnostic.",
    "relative_regret_reduction": .20, "gate_absolute_tolerance": 1e-12,
    "arena_score_threshold": .60,
    "gate": "Delta must reduce mean signed bounded regret by20% versus BOTH direct/action_only on BOTH panels, positive reference means required; AND arena lower point bound at least60% versus each, zero failures across288 games.",
    "selection": "All final fits. One explicitly recorded recovery from fresh initialization after v1 output-pipe failure; no best epoch/seed, further retries, replacement games or budget extension.",
    "matching": "Same labels, positions, minibatches, updates and shared root initialization. Parameters, FLOPs and wall time are NOT matched. All three refinement arms share active modules.",
    "uncertainty": "Pointwise source-game bootstrap conditional on three fitted seeds; not simultaneous or seed-population uncertainty. Shared afterstates across same-panel source games can leave residual dependence.",
    "scope": "Development mechanism diagnostic on previously studied generators. Native one-ply consequences are not learned dynamics. No Elo, novelty, calibration or world-model benefit established.",
}

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def tree_files(directory):
    directory = Path(directory)
    found = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence must not contain symlinks")
        if path.is_file():
            found[path.relative_to(directory).as_posix()] = sha(path)
    return found

def digest_state(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()

def grading_module():
    spec = importlib.util.spec_from_file_location(
        "capacity_frozen_grading", ROOT / "scripts/chess_compute_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.PROTOCOL["regret_nodes"] != PROTOCOL["regret_nodes"]
        or module.PROTOCOL["value_cp_scale"] != data.CONFIG["value_cp_scale"]
        or module.PROTOCOL["mate_cp"] != data.CONFIG["mate_cp"]
    ):
        raise ValueError("Frozen grader budget or scale mismatch")
    return module

def panels(eval_rows):
    secondary, latency, combined = {}, [], []
    for offset, split in enumerate(("dev", "shift")):
        n = len(eval_rows[split])
        secondary[split] = sorted(
            random.Random(PROTOCOL["regret_selection_seed"] + offset).sample(
                range(n), PROTOCOL["regret_positions_per_split"]
            )
        )
        selected = sorted(
            random.Random(PROTOCOL["latency_selection_seed"] + offset).sample(
                range(n), PROTOCOL["latency_positions_per_split"]
            )
        )
        latency.extend(len(combined) + i for i in selected)
        combined.extend(eval_rows[split])
    return {
        "secondary_indices": secondary,
        "latency_indices": latency,
        "graded_configurations": [{"id": c["name"]} for c in configs()],
    }, combined

def audit_latency(measurement, combined, selected):
    required = {
        "device",
        "torch_threads",
        "depth",
        "timing_scope",
        "warmup_policy",
        "warmup_records",
        "records",
        "warmup_wall_ms",
        "total_wall_ms",
    }
    if (
        set(measurement) != required
        or measurement["device"] != "cpu"
        or measurement["torch_threads"] != PROTOCOL["torch_threads"]
        or measurement["depth"] != PROTOCOL["depth"]
        or measurement["timing_scope"]
        != "Full single-position choose call, including board construction, encoding and legal-response validation"
        or measurement["warmup_policy"]
        != "Three standard-starting-board calls before the fixed selected positions"
        or len(measurement["warmup_records"]) != PROTOCOL["latency_warmups"]
        or len(measurement["records"]) != len(selected)
    ):
        raise ValueError("Latency identity or coverage mismatch")
    for index, value in enumerate(measurement["warmup_records"]):
        if (
            value["id"] != "starting-board"
            or value["index"] != index
            or chess.Move.from_uci(value["choice"]) not in chess.Board().legal_moves
        ):
            raise ValueError("Invalid latency warmup")
    for value, index in zip(measurement["records"], selected, strict=True):
        if (
            value["id"] != combined[index]["id"]
            or value["index"] != index
            or chess.Move.from_uci(value["choice"]) not in chess.Board(combined[index]["fen"]).legal_moves
        ):
            raise ValueError("Latency selection or legality mismatch")
    for key, total in (("records", "total_wall_ms"), ("warmup_records", "warmup_wall_ms")):
        if any(not math.isfinite(r["wall_ms"]) or r["wall_ms"] < 0 for r in measurement[key]):
            raise ValueError("Invalid latency duration")
        if not math.isclose(
            measurement[total], math.fsum(r["wall_ms"] for r in measurement[key]), rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("Latency duration total mismatch")


def configs():
    result = [{"arm": arm, "width": PROTOCOL["width"], "seed": seed, "name": f"{arm}-{seed}"}
              for arm in PROTOCOL["arms"] for seed in PROTOCOL["seeds"]]
    random.Random(PROTOCOL["fit_order_seed"]).shuffle(result)
    return result


def make_model(config):
    return CandidateChess(config["arm"], config["seed"], config["width"],
                          PROTOCOL["depth"], PROTOCOL["branch_depth"])


def load_model(path, config, plan_hash):
    return CandidateChess.load(path, expected_plan_sha256=plan_hash, expected_arm=config["arm"],
                               expected_seed=config["seed"], expected_width=config["width"],
                               expected_root_depth=PROTOCOL["depth"],
                               expected_branch_depth=PROTOCOL["branch_depth"])


def signature(exclusions=None):
    exclusions = data.collect_exclusions(ROOT) if exclusions is None else exclusions
    preflights = {}
    for path in PREFLIGHTS:
        record = read(ROOT/path)
        if (record["status"] != "completed" or set(record["arms"]) != set(ARMS)
                or any(row["weights_unchanged"] is not True for row in record["arms"].values())
                or any(sha(ROOT/name) != digest for name, digest in record["sources"].items())):
            raise ValueError("Synthetic preflight source or completion mismatch")
        preflights[path] = sha(ROOT/path)
    return {
        "protocol": PROTOCOL, "data_config": data.CONFIG, "configurations": configs(),
        "recovery": attempt.audit(ROOT, SimpleNamespace(computation=computation, schedule=schedule)),
        "openings": arena.OPENINGS, "arena_schedule": arena.schedule(),
        "arena_protocol": arena.ARENA_PROTOCOL,
        "sources": {p: sha(ROOT/p) for p in SOURCES}, "preflights": preflights,
        "training_source_sha256": sha(ROOT/data.TRAIN),
        "exclusion_files": exclusions["files"], "exclusion_counts": exclusions["counts"],
        "exclusion_states_sha256": hashlib.sha256(json.dumps(exclusions["states"], separators=(",", ":")).encode()).hexdigest(),
        "engine_sha256": sha(ROOT/ENGINE),
        "parameters": {a: CandidateChess(a, PROTOCOL["seeds"][0], PROTOCOL["width"],
                                         PROTOCOL["depth"], PROTOCOL["branch_depth"]).parameter_counts()
                       for a in PROTOCOL["arms"]},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "mps_available": torch.backends.mps.is_available(),
                        "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "chess", "numpy")},
                        "lock_sha256": sha(ROOT/"uv.lock"), "pyproject_sha256": sha(ROOT/"pyproject.toml")},
    }


def prepare(out):
    frozen = signature()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"plan.json", frozen)
    write_new(out/"prepared.json", {"status": "prepared", "plan_sha256": sha(out/"plan.json"),
                                   "created_unix": time.time(), "training_or_scoring_started": False})
    return out/"plan.json"


def verify_plan(path, exclusions=None):
    plan = read(path)
    if plan != json.loads(json.dumps(signature(exclusions))):
        raise ValueError("Frozen source, input, budget or environment changed")
    return plan


def schedule(seed):
    generator = torch.Generator().manual_seed(PROTOCOL["batch_order_seed_base"]+seed)
    result = []
    for epoch in range(PROTOCOL["epochs"]):
        order = torch.randperm(PROTOCOL["train_examples"], generator=generator).tolist()
        for start in range(0, len(order), PROTOCOL["batch_size"]):
            result.append({"step": len(result)+1, "epoch": epoch+1,
                           "indices": order[start:start+PROTOCOL["batch_size"]]})
    if len(result) != PROTOCOL["updates_per_fit"]:
        raise ValueError("Training schedule differs from its frozen update budget")
    return result


def sync():
    if PROTOCOL["training_device"] == "mps":
        torch.mps.synchronize()


def computation(arm, examples, candidates):
    root = examples*PROTOCOL["depth"]
    refinement = 0 if arm == "direct" else candidates*PROTOCOL["branch_depth"]
    afterstate = candidates*PROTOCOL["depth"] if arm == "full_afterstate" else 0
    return {"candidate_evaluations": candidates, "root_core_iterations": root,
            "candidate_refinement_iterations": refinement, "successor_root_iterations": afterstate,
            "total_core_iterations": root+refinement+afterstate,
            "delta_convolutions": candidates if arm == "delta" else 0,
            "successor_encoders": candidates if arm == "full_afterstate" else 0}


def fit(config, cache, out, plan_hash, data_hash):
    out = Path(out)
    out.mkdir()
    model = make_model(config)
    initial = digest_state(model)
    model.to(PROTOCOL["training_device"])
    opt = PROTOCOL["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt["lr"], betas=tuple(opt["betas"]),
                                 eps=opt["eps"], weight_decay=opt["weight_decay"])
    steps = schedule(config["seed"])
    totals = computation(config["arm"], 0, 0)
    sync()
    begun = time.perf_counter()
    with (out/"learning.jsonl").open("x") as stream:
        for step in steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ce_total, mse_total, loss_total = 0., 0., 0.
            microbatches = 0
            sampled_allocation = 0
            for start in range(0, len(step["indices"]), PROTOCOL["microbatch_size"]):
                indices = step["indices"][start:start+PROTOCOL["microbatch_size"]]
                inputs, targets, values = cache.batch(indices, config["arm"], device=PROTOCOL["training_device"])
                logits, predicted, _ = model(**inputs, depth=PROTOCOL["depth"],
                                             candidate_chunk_size=PROTOCOL["candidate_chunk_size"])
                ce = F.cross_entropy(logits, targets)
                mse = F.mse_loss(predicted, values)
                loss = ce+PROTOCOL["value_loss_weight"]*mse
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                weight = len(indices)/len(step["indices"])
                (weight*loss).backward()
                ce_total += weight*float(ce.detach().cpu())
                mse_total += weight*float(mse.detach().cpu())
                loss_total += weight*float(loss.detach().cpu())
                microbatches += 1
                if PROTOCOL["training_device"] == "mps":
                    sampled_allocation = max(sampled_allocation, torch.mps.current_allocated_memory())
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip"],
                                                     error_if_nonfinite=True)
            optimizer.step()
            counts = computation(config["arm"], len(step["indices"]),
                                 sum(len(cache.menus[i]) for i in step["indices"]))
            for key, value in counts.items():
                totals[key] += value
            record = {"step": step["step"], "epoch": step["epoch"], "examples": len(step["indices"]),
                      "root_depth": PROTOCOL["depth"], "branch_depth": PROTOCOL["branch_depth"],
                      "indices_sha256": hashlib.sha256(json.dumps(step["indices"]).encode()).hexdigest(),
                      "microbatches": microbatches, "policy_ce": ce_total, "value_mse": mse_total,
                      "loss": loss_total, "gradient_norm": float(gradient.detach().cpu()),
                      "sampled_mps_allocated_bytes_after_backward": sampled_allocation, **counts}
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            if step["step"] % 128 == 0:
                print(json.dumps({"fit": config["name"], "updates": step["step"]}), flush=True)
    sync()
    elapsed = time.perf_counter()-begun
    model.cpu().save(out/"weights.pt", plan_sha256=plan_hash)
    result = {"status": "completed", **config, "plan_sha256": plan_hash, "data_receipt_sha256": data_hash,
              "cache_sha256": cache.metadata["cache_sha256"], "initial_state_sha256": initial,
              "updates": len(steps), "examples_seen": sum(len(s["indices"]) for s in steps),
              "training_seconds": elapsed, "computation": totals,
              "memory_scope": "Allocation sampled after microbatch backward, not a measured peak.",
              "weights_sha256": sha(out/"weights.pt"), "learning_sha256": sha(out/"learning.jsonl")}
    write_new(out/"training.json", result)
    return result


def run(plan_path, out):
    exclusions = data.collect_exclusions(ROOT)
    frozen = verify_plan(plan_path, exclusions)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan_hash = sha(plan_path)
    write_new(out/"started.json", {"status": "started", "plan_sha256": plan_hash, "started_unix": time.time()})
    begun = time.perf_counter()
    try:
        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.use_deterministic_algorithms(PROTOCOL["deterministic_algorithms"])
        attempt.copy_data(ROOT, out/"data", frozen["recovery"])
        data.validate(out/"data", exclusions["states"])
        del exclusions
        training = data.training_rows(ROOT)
        if len(training) != PROTOCOL["train_examples"]:
            raise ValueError("Training row count mismatch")
        cache = CachedPositions(training)
        write_new(out/"training-cache.json", cache.metadata)
        eval_rows = {s: rows(out/"data"/f"{s}.jsonl") for s in ("dev", "shift")}
        panel, combined = panels(eval_rows)
        write_new(out/"panels.json", panel)
        write_new(out/"baselines.json", {s: simple_baselines(training, value) for s, value in eval_rows.items()})
        for config in frozen["configurations"]:
            fit(config, cache, out/config["name"], plan_hash, sha(out/"data/completed.json"))
            gc.collect()
            if PROTOCOL["training_device"] == "mps":
                torch.mps.empty_cache()
        del cache
        gc.collect()
        # All twelve final fits complete before fresh neural evaluation or diagnostics.
        prepared = {s: CachedPositions(value) for s, value in eval_rows.items()}
        write_new(out/"evaluation-caches.json", {s: c.metadata for s, c in prepared.items()})
        decisions, model_map = [], {}
        for config in frozen["configurations"]:
            directory = out/config["name"]
            model = load_model(directory/"weights.pt", config, plan_hash)
            model_map[(config["arm"], config["seed"])] = model
            evaluation, diagnostic = {}, {}
            for split in eval_rows:
                evaluation[split] = evaluate(model, prepared[split], directory/f"{split}.jsonl")
                decisions.extend({"configuration": config["name"], "split": split, "panel_index": i,
                                  "id": p["id"], "choice": p["choice"]}
                                 for i, p in enumerate(rows(directory/f"{split}.jsonl")))
                if config["arm"] == "delta":
                    diagnostic[split] = evaluate(model, prepared[split], directory/f"permuted-{split}.jsonl", permuted=True)
            write_new(directory/"latency.json", probe(model, combined, PROTOCOL["depth"], panel["latency_indices"]))
            write_new(directory/"completed.json", {"status": "completed", **config, "plan_sha256": plan_hash,
                                                    "evaluation": evaluation, "diagnostic": diagnostic,
                                                    "files": tree_files(directory)})
            print(json.dumps({"evaluated": config["name"],
                              "agreement": {s: r["metrics"]["agreement"] for s, r in evaluation.items()}}), flush=True)
        directory = out/"regret"
        directory.mkdir()
        grading_plan = {"engine_path": str((ROOT/ENGINE).resolve()), "panels": eval_rows,
                        "secondary_indices": panel["secondary_indices"], "configurations": panel["graded_configurations"]}
        cost = grading_module().score_secondary(grading_plan, directory, decisions)
        if cost["calls"] > PROTOCOL["regret_call_ceiling"] or cost["requested_nodes"] > PROTOCOL["regret_node_ceiling"]:
            raise ValueError("Grading budget exceeded")
        write_new(directory/"completed.json", {"status": "completed", "cost": cost, "files": tree_files(directory)})
        arena.run(out/"arena", model_map, plan_hash)
        write_new(out/"completed.json", {"status": "completed", "plan_sha256": plan_hash,
                                         "wall_seconds": time.perf_counter()-begun, "files": tree_files(out)})
    except Exception as exc:
        write_new(out/"failed.json", {"status": "failed", "plan_sha256": plan_hash,
                                     "error": str(exc), "wall_seconds": time.perf_counter()-begun})
        raise
    return out/"completed.json"


def cluster_interval(differences, game_ids, seed):
    keys = sorted(set(game_ids))
    sums, counts = np.zeros(len(keys)), np.zeros(len(keys))
    index = {k: i for i, k in enumerate(keys)}
    for difference, key in zip(differences, game_ids, strict=True):
        sums[index[key]] += difference
        counts[index[key]] += 1
    samples = np.random.default_rng(seed).integers(len(keys), size=(PROTOCOL["bootstrap_replicates"], len(keys)))
    values = sums[samples].sum(-1)/counts[samples].sum(-1)
    return {"games": len(keys), "positions": len(game_ids), "mean": float(np.mean(differences)),
            "lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975)),
            "scope": PROTOCOL["uncertainty"]}


def comparisons(metrics, predictions, regret, eval_rows, arena_summary):
    names = {c["name"] for c in configs()}
    if (set(eval_rows) != {"dev", "shift"} or any(set(c) != names for c in (metrics, predictions, regret))
            or any(set(v) != {"dev", "shift"} for c in (metrics, predictions, regret) for v in c.values())):
        raise ValueError("Incomplete candidate comparison coverage")
    means = {a: {s: {key: float(np.mean([metrics[f"{a}-{seed}"][s][key] for seed in PROTOCOL["seeds"]]))
                     for key in ("agreement", "target_nll", "value_mae", "mean_confidence")}
                 for s in ("dev", "shift")} for a in PROTOCOL["arms"]}
    checks, compared = [], []
    for i, comparator in enumerate(("direct", "action_only", "full_afterstate")):
        for j, split in enumerate(("dev", "shift")):
            reference = float(np.mean([regret[f"{comparator}-{s}"][split] for s in PROTOCOL["seeds"]]))
            delta = float(np.mean([regret[f"delta-{s}"][split] for s in PROTOCOL["seeds"]]))
            if not math.isfinite(reference) or not math.isfinite(delta):
                raise ValueError("Nonfinite regret")
            reduction = (reference-delta)/reference if reference > 0 else None
            differences = np.mean([[int(a["correct"])-int(b["correct"])
                                     for a, b in zip(predictions[f"delta-{s}"][split],
                                                     predictions[f"{comparator}-{s}"][split], strict=True)]
                                    for s in PROTOCOL["seeds"]], axis=0)
            compared.append({"split": split, "comparator": comparator, "reference_bounded_regret": reference,
                             "delta_bounded_regret": delta, "bounded_regret_change": delta-reference,
                             "relative_reduction": reduction, "primary": comparator != "full_afterstate",
                             "agreement_interval": cluster_interval(differences, [r["game_id"] for r in eval_rows[split]],
                                                                      PROTOCOL["bootstrap_seed"]+i*2+j)})
            if comparator != "full_afterstate":
                checks.append({"metric": "relative_regret_reduction", "split": split, "comparator": comparator,
                               "observed": reduction, "threshold": PROTOCOL["relative_regret_reduction"],
                               "passed": reduction is not None and (reduction >= PROTOCOL["relative_regret_reduction"]
                                          or math.isclose(reduction, PROTOCOL["relative_regret_reduction"], rel_tol=0,
                                                          abs_tol=PROTOCOL["gate_absolute_tolerance"]))})
    checks.append({"metric": "arena", "passed": arena_summary["gate_passed"]})
    return {"means": means, "comparisons": compared, "checks": checks,
            "continuation_passed": all(c["passed"] for c in checks)}


def audit_training(directory, config, plan_hash, data_hash, cache_metadata, training):
    directory = Path(directory)
    trained = read(directory/"training.json")
    learning = rows(directory/"learning.jsonl")
    steps = schedule(config["seed"])
    if (trained["status"] != "completed" or any(trained[k] != config[k] for k in config)
            or trained["plan_sha256"] != plan_hash or trained["data_receipt_sha256"] != data_hash
            or trained["cache_sha256"] != cache_metadata["cache_sha256"]
            or trained["weights_sha256"] != sha(directory/"weights.pt")
            or trained["learning_sha256"] != sha(directory/"learning.jsonl")
            or len(learning) != len(steps) or trained["updates"] != len(steps)
            or trained["examples_seen"] != sum(len(s["indices"]) for s in steps)
            or trained["initial_state_sha256"] != digest_state(make_model(config))
            or not math.isfinite(trained["training_seconds"]) or trained["training_seconds"] < 0):
        raise ValueError("Training budget, initialization or provenance mismatch")
    counts = [chess.Board(r["fen"]).legal_moves.count() for r in training]
    total = computation(config["arm"], 0, 0)
    for record, step in zip(learning, steps, strict=True):
        expected = computation(config["arm"], len(step["indices"]), sum(counts[i] for i in step["indices"]))
        if (record["step"] != step["step"] or record["epoch"] != step["epoch"]
                or record["examples"] != len(step["indices"])
                or record["root_depth"] != PROTOCOL["depth"] or record["branch_depth"] != PROTOCOL["branch_depth"]
                or record["indices_sha256"] != hashlib.sha256(json.dumps(step["indices"]).encode()).hexdigest()
                or record["microbatches"] != math.ceil(len(step["indices"])/PROTOCOL["microbatch_size"])
                or any(record[k] != v for k, v in expected.items())
                or any(not math.isfinite(record[k]) or record[k] < 0
                       for k in ("policy_ce", "value_mse", "loss", "gradient_norm"))
                or type(record["sampled_mps_allocated_bytes_after_backward"]) is not int
                or record["sampled_mps_allocated_bytes_after_backward"] < 0
                or not math.isclose(record["loss"], record["policy_ce"]+PROTOCOL["value_loss_weight"]*record["value_mse"],
                                    rel_tol=2e-6, abs_tol=1e-6)):
            raise ValueError("Learning journal or candidate computation mismatch")
        for key, value in expected.items():
            total[key] += value
    if trained["computation"] != total:
        raise ValueError("Training computation totals mismatch")
    return {k: trained[k] for k in ("updates", "examples_seen", "training_seconds", "computation", "memory_scope")}


def audit_evaluation_receipt(receipt, metrics, diagnostic, cache_hash):
    if (receipt["metrics"] != metrics or receipt["diagnostic"] != diagnostic
            or receipt["cache_sha256"] != cache_hash
            or receipt["timing_scope"] != "Cached CPU batch evaluation and prediction writing; native cache construction excluded and recorded separately."
            or not math.isfinite(receipt["evaluation_wall_seconds"]) or receipt["evaluation_wall_seconds"] < 0):
        raise ValueError("Evaluation receipt, cache or diagnostics mismatch")


def report(plan_path, execution, out):
    exclusions = data.collect_exclusions(ROOT)
    plan = verify_plan(plan_path, exclusions)
    plan_hash = sha(plan_path)
    execution = Path(execution)
    complete = read(execution/"completed.json")
    actual = tree_files(execution)
    actual.pop("completed.json")
    if ((execution/"failed.json").exists() or complete["status"] != "completed"
            or complete["plan_sha256"] != plan_hash or complete["files"] != actual
            or read(execution/"started.json")["plan_sha256"] != plan_hash
            or not math.isfinite(complete["wall_seconds"]) or complete["wall_seconds"] < 0):
        raise ValueError("Incomplete execution or changed evidence")
    data.validate(execution/"data", exclusions["states"])
    del exclusions
    training = data.training_rows(ROOT)
    if len(training) != PROTOCOL["train_examples"]:
        raise ValueError("Training row count mismatch")
    eval_rows = {s: rows(execution/"data"/f"{s}.jsonl") for s in ("dev", "shift")}
    panel, combined = panels(eval_rows)
    if read(execution/"panels.json") != panel:
        raise ValueError("Evaluation panel selection changed")
    cache_metadata = read(execution/"training-cache.json")
    audit_cache_metadata(cache_metadata, training)
    evaluation_caches = read(execution/"evaluation-caches.json")
    if set(evaluation_caches) != {"dev", "shift"}:
        raise ValueError("Incomplete evaluation caches")
    for split, value in eval_rows.items():
        audit_cache_metadata(evaluation_caches[split], value)
    baseline = read(execution/"baselines.json")
    if baseline != {s: simple_baselines(training, value) for s, value in eval_rows.items()}:
        raise ValueError("Native baseline aggregates inconsistent")
    metrics, predictions, costs, latencies, decisions, diagnostics, model_digests = {}, {}, {}, {}, [], {}, {}
    for config in plan["configurations"]:
        name, arm = config["name"], config["arm"]
        directory = execution/name
        receipt = read(directory/"completed.json")
        required = {"weights.pt", "learning.jsonl", "training.json", "latency.json", "dev.jsonl", "shift.jsonl"}
        if arm == "delta":
            required.update(("permuted-dev.jsonl", "permuted-shift.jsonl"))
        if (set(receipt["files"]) != required or receipt["status"] != "completed"
                or receipt["plan_sha256"] != plan_hash or any(receipt[k] != config[k] for k in config)
                or any(sha(directory/n) != h for n, h in receipt["files"].items())
                or set(receipt["evaluation"]) != {"dev", "shift"}
                or set(receipt["diagnostic"]) != ({"dev", "shift"} if arm == "delta" else set())):
            raise ValueError("Fit identity, receipt or evaluation coverage mismatch")
        model = load_model(directory/"weights.pt", config, plan_hash)
        model_digests[name] = digest_state(model)
        costs[name] = audit_training(directory, config, plan_hash, sha(execution/"data/completed.json"),
                                     cache_metadata, training)
        metrics[name], predictions[name] = {}, {}
        for split, value in eval_rows.items():
            calculated, preds = audit_predictions(directory/f"{split}.jsonl", value, expected_arm=arm)
            audit_evaluation_receipt(receipt["evaluation"][split], calculated, _diagnostic(preds, False),
                                     evaluation_caches[split]["cache_sha256"])
            metrics[name][split], predictions[name][split] = calculated, preds
            decisions.extend({"configuration": name, "split": split, "panel_index": i,
                              "id": p["id"], "choice": p["choice"]} for i, p in enumerate(preds))
            if arm == "delta":
                checked = audit_diagnostic(directory/f"permuted-{split}.jsonl", value)
                audit_evaluation_receipt(receipt["diagnostic"][split], checked["metrics"], checked["diagnostic"],
                                         evaluation_caches[split]["cache_sha256"])
                diagnostics.setdefault(name, {})[split] = checked
        measurement = read(directory/"latency.json")
        audit_latency(measurement, combined, panel["latency_indices"])
        latencies[name] = measurement
    directory = execution/"regret"
    receipt = read(directory/"completed.json")
    if (receipt["status"] != "completed"
            or set(receipt["files"]) != {"engine.json", "analyses.jsonl", "regret.jsonl"}
            or any(sha(directory/n) != h for n, h in receipt["files"].items())):
        raise ValueError("Grading receipt mismatch")
    engine = read(directory/"engine.json")
    if engine["sha256"] != plan["engine_sha256"] or not engine["id"]["name"].startswith("Stockfish 19"):
        raise ValueError("Grading engine mismatch")
    records, analyses = rows(directory/"regret.jsonl"), rows(directory/"analyses.jsonl")
    grading_plan = {"panels": eval_rows, "secondary_indices": panel["secondary_indices"],
                    "configurations": panel["graded_configurations"]}
    grading_module().validate_secondary(grading_plan, decisions, analyses, records, receipt["cost"])
    if (receipt["cost"]["calls"] > PROTOCOL["regret_call_ceiling"]
            or receipt["cost"]["requested_nodes"] > PROTOCOL["regret_node_ceiling"]):
        raise ValueError("Grading budget exceeded")
    regret = {c["name"]: {s: float(np.mean([r["bounded_regret"] for r in records
                                           if r["configuration"] == c["name"] and r["split"] == s]))
                           for s in ("dev", "shift")} for c in configs()}
    arena_summary = arena.report(execution/"arena", plan_hash)
    bindings = read(execution/"arena/started.json")["models"]
    if (set(bindings) != set(model_digests)
            or any(bindings[name]["state_sha256"] != value for name, value in model_digests.items())):
        raise ValueError("Arena weights differ from trained checkpoints")
    result = {"status": "completed", "plan_sha256": plan_hash,
              "execution_receipt_sha256": sha(execution/"completed.json"), "metrics": metrics, "regret": regret,
              "costs": costs, "latency": latencies, "permutation_diagnostic": diagnostics,
              "training_cache": cache_metadata, "evaluation_caches": evaluation_caches,
              "engine_cost": receipt["cost"], "fresh_data_cost": read(execution/"data/completed.json"),
              "baselines": {s: {n: v["metrics"] for n, v in baseline[s].items()} for s in baseline},
              "arena": arena_summary, **comparisons(metrics, predictions, regret, eval_rows, arena_summary),
              "scope": PROTOCOL["scope"], "novelty_established": False, "elo_estimate": None,
              "recovery": plan["recovery"],
              "attempt_accounting": {"failed_wall_seconds": plan["recovery"]["failure"]["wall_seconds"],
                  "successful_execution_wall_seconds": complete["wall_seconds"],
                  "total_attempt_wall_seconds": plan["recovery"]["failure"]["wall_seconds"] + complete["wall_seconds"],
                  "final_fit_updates": sum(c["updates"] for c in costs.values()),
                  "discarded_updates": plan["recovery"]["completed_optimizer_updates"],
                  "total_optimizer_updates": sum(c["updates"] for c in costs.values()) + plan["recovery"]["completed_optimizer_updates"],
                  "data_generated_once": True}}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"summary.json", result)
    write_new(out/"completed.json", {"status": "completed", "plan_sha256": plan_hash,
                                     "summary_sha256": sha(out/"summary.json"),
                                     "execution_receipt_sha256": sha(execution/"completed.json")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--out", required=True)
        if name == "report":
            p.add_argument("--execution", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.out))
    elif args.command == "run":
        print(run(args.plan, args.out))
    else:
        result = report(args.plan, args.execution, args.out)
        print(json.dumps({k: result[k] for k in ("means", "comparisons", "checks", "continuation_passed")}, indent=2))


if __name__ == "__main__":
    main()
