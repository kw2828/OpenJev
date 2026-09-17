"""Matched-data chess capacity experiment with fresh panels and actual games.

Width is the only within-study architectural change. This is a strength
baseline, not a new architecture or equal-compute comparison. Prepare freezes
inputs, sources, openings and budgets before any data generation or fitting.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch
from torch.nn import functional as F

from openjev.research import chess_capacity_arena as arena
from openjev.research import chess_capacity_data as data
from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_anchor_eval import audit_predictions, evaluate, probe, tensorize
from openjev.research.chess_spatial_baselines import simple_baselines

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "runs/chess-inputs/stockfish/stockfish-macos-universal"
SOURCES = [
    "scripts/chess_capacity_study.py",
    "tests/test_chess_capacity_study.py",
    "src/openjev/research/chess_capacity_data.py",
    "tests/test_chess_capacity_data.py",
    "src/openjev/research/chess_capacity_arena.py",
    "tests/test_chess_capacity_arena.py",
    "src/openjev/research/chess_anchor.py",
    "src/openjev/research/chess_anchor_data.py",
    "src/openjev/research/chess_anchor_eval.py",
    "src/openjev/research/chess_spatial.py",
    "src/openjev/research/chess_spatial_data.py",
    "src/openjev/research/chess_spatial_baselines.py",
    "src/openjev/research/chess_arena.py",
    "scripts/chess_compute_study.py",
    "src/openjev/research/chess_compute.py",
]
PROTOCOL = {
    "version": "chess-capacity-v1",
    "widths": [32, 128],
    "seeds": [53, 67, 83],
    "depth": 4,
    "train_examples": 98304,
    "epochs": 8,
    "batch_size": 128,
    "updates_per_fit": 6144,
    "core_iterations_per_fit": 24576,
    "training_device": "mps",
    "evaluation_device": "cpu",
    "torch_threads": 2,
    "deterministic_algorithms": False,
    "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "gradient_clip": 1.0,
    "value_loss_weight": 0.5,
    "loss": "Legal-move CE + 0.5 bounded-value MSE; auxiliary head unused.",
    "recurrence": "Ordinary residual at depth four; width changes encoder, recurrent core and heads together.",
    "batch_order_seed_base": 10800029,
    "fit_order_seed": 10800017,
    "regret_positions_per_split": 128,
    "regret_selection_seed": 10800043,
    "regret_nodes": 20000,
    "regret_call_ceiling": 1792,
    "regret_node_ceiling": 35840000,
    "latency_positions_per_split": 64,
    "latency_selection_seed": 10800059,
    "latency_warmups": 3,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 10800071,
    "relative_regret_reduction": 0.20,
    "gate_absolute_tolerance": 1e-12,
    "arena_score_threshold": 0.60,
    "gate": "At least20% lower mean signed bounded engine-score loss on both fresh panels, requiring positive reference means, AND the96-game arena lower point bound at least60% with no failed games.",
    "selection": "All final checkpoints. No early stopping, retries, replacement games, selected seeds or budget extension.",
    "matching": "Same data, minibatches, optimizer updates and recurrent depth; parameter count, FLOPs and wall time are NOT matched.",
    "uncertainty": "Pointwise game-cluster bootstrap, conditional on these three trained seeds; not simultaneous or seed-population uncertainty.",
    "scope": "Controlled capacity scaling on previously studied generators and shift. No Elo, new architecture, calibrated confidence or world-model planning claim.",
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


def configs():
    result = [
        {"width": w, "seed": s, "name": f"width{w}-{s}"}
        for w in PROTOCOL["widths"]
        for s in PROTOCOL["seeds"]
    ]
    random.Random(PROTOCOL["fit_order_seed"]).shuffle(result)
    return result


def digest_state(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def signature():
    exclusions = data.collect_exclusions(ROOT)
    counts = {}
    for width in PROTOCOL["widths"]:
        model = AnchorChess("residual", PROTOCOL["seeds"][0], width, PROTOCOL["depth"])
        counts[str(width)] = {
            "stored": model.parameter_count(),
            "active": model.parameter_count() - sum(p.numel() for p in model.aux_head.parameters()),
        }
    return {
        "protocol": PROTOCOL,
        "data_config": data.CONFIG,
        "configurations": configs(),
        "openings": arena.OPENINGS,
        "arena_schedule": arena.schedule(),
        "arena_protocol": arena.ARENA_PROTOCOL,
        "sources": {name: sha(ROOT / name) for name in SOURCES},
        "exclusion_files": exclusions["files"],
        "exclusion_counts": exclusions["counts"],
        "exclusion_states_sha256": hashlib.sha256(
            json.dumps(exclusions["states"], separators=(",", ":")).encode()
        ).hexdigest(),
        "engine_sha256": sha(ROOT / ENGINE),
        "parameters": counts,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mps_available": torch.backends.mps.is_available(),
            "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "chess", "numpy")},
            "lock_sha256": sha(ROOT / "uv.lock"),
            "pyproject_sha256": sha(ROOT / "pyproject.toml"),
        },
    }


def prepare(out):
    frozen = signature()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "plan.json", frozen)
    write_new(
        out / "prepared.json",
        {
            "status": "prepared",
            "plan_sha256": sha(out / "plan.json"),
            "created_unix": time.time(),
            "training_or_scoring_started": False,
        },
    )
    return out / "plan.json"


def verify_plan(path):
    value = read(path)
    if value != json.loads(json.dumps(signature())):
        raise ValueError("Frozen source, input, budget or environment changed")
    return value


def schedule(seed):
    generator = torch.Generator().manual_seed(PROTOCOL["batch_order_seed_base"] + seed)
    result = []
    for epoch in range(PROTOCOL["epochs"]):
        order = torch.randperm(PROTOCOL["train_examples"], generator=generator).tolist()
        for start in range(0, len(order), PROTOCOL["batch_size"]):
            result.append(
                {
                    "step": len(result) + 1,
                    "epoch": epoch + 1,
                    "indices": order[start : start + PROTOCOL["batch_size"]],
                }
            )
    if (
        len(result) != PROTOCOL["updates_per_fit"]
        or len(result) * PROTOCOL["depth"] != PROTOCOL["core_iterations_per_fit"]
    ):
        raise ValueError("Training budget does not match schedule")
    return result


def sync():
    if PROTOCOL["training_device"] == "mps":
        torch.mps.synchronize()


def fit(config, tensors, out, plan_hash, data_hash):
    out.mkdir()
    model = AnchorChess("residual", config["seed"], config["width"], PROTOCOL["depth"])
    initial = digest_state(model)
    model.to(PROTOCOL["training_device"])
    opt = PROTOCOL["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt["lr"],
        betas=tuple(opt["betas"]),
        eps=opt["eps"],
        weight_decay=opt["weight_decay"],
    )
    steps = schedule(config["seed"])
    sync()
    begun = time.perf_counter()
    with (out / "learning.jsonl").open("x") as stream:
        for step in steps:
            batch = {
                key: value[step["indices"]].to(PROTOCOL["training_device"]) for key, value in tensors.items()
            }
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, values, _ = model(
                batch["observations"], batch["candidates"], batch["mask"], depth=PROTOCOL["depth"]
            )
            ce = F.cross_entropy(logits, batch["targets"])
            mse = F.mse_loss(values, batch["values"])
            loss = ce + PROTOCOL["value_loss_weight"] * mse
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            record = {
                "step": step["step"],
                "epoch": step["epoch"],
                "depth": PROTOCOL["depth"],
                "examples": len(step["indices"]),
                "policy_ce": float(ce.detach().cpu()),
                "value_mse": float(mse.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient.detach().cpu()),
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if step["step"] % 768 == 0:
                print(json.dumps({"fit": config["name"], "updates": step["step"]}), flush=True)
    sync()
    elapsed = time.perf_counter() - begun
    model.cpu().save(out / "weights.pt", plan_sha256=plan_hash)
    result = {
        "status": "completed",
        **config,
        "plan_sha256": plan_hash,
        "data_receipt_sha256": data_hash,
        "initial_state_sha256": initial,
        "updates": len(steps),
        "core_iterations": len(steps) * PROTOCOL["depth"],
        "examples_seen": sum(len(s["indices"]) for s in steps),
        "training_seconds": elapsed,
        "weights_sha256": sha(out / "weights.pt"),
        "learning_sha256": sha(out / "learning.jsonl"),
    }
    write_new(out / "training.json", result)
    return result


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


def run(plan_path, out):
    frozen = verify_plan(plan_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "started.json",
        {"status": "started", "plan_sha256": sha(plan_path), "started_unix": time.time()},
    )
    begun = time.perf_counter()
    try:
        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.use_deterministic_algorithms(PROTOCOL["deterministic_algorithms"])
        exclusions = data.collect_exclusions(ROOT)
        data.generate(out / "data", ROOT / ENGINE, exclusions["states"])
        data.validate(out / "data", exclusions["states"])
        training = data.training_rows(ROOT, out / "data")
        if len(training) != PROTOCOL["train_examples"]:
            raise ValueError("Training row count mismatch")
        tensors, _ = tensorize(training)
        eval_rows = {split: rows(out / "data" / f"{split}.jsonl") for split in ("dev", "shift")}
        prepared = {split: tensorize(value) for split, value in eval_rows.items()}
        panel, combined = panels(eval_rows)
        write_new(out / "panels.json", panel)
        write_new(
            out / "baselines.json", {s: simple_baselines(training, value) for s, value in eval_rows.items()}
        )
        data_hash = sha(out / "data/completed.json")
        for config in frozen["configurations"]:
            fit(config, tensors, out / config["name"], sha(plan_path), data_hash)
        # Every fit completes before any fresh neural evaluation is exposed.
        decisions, model_map = [], {}
        for config in frozen["configurations"]:
            directory = out / config["name"]
            model = AnchorChess.load(
                directory / "weights.pt",
                expected_plan_sha256=sha(plan_path),
                expected_recurrence="residual",
                expected_seed=config["seed"],
            )
            model_map[(config["width"], config["seed"])] = model
            evaluation = {}
            for split, value in eval_rows.items():
                ts, menus = prepared[split]
                evaluation[split] = evaluate(
                    model, value, ts, menus, PROTOCOL["depth"], directory / f"{split}.jsonl"
                )
                decisions.extend(
                    {
                        "configuration": config["name"],
                        "split": split,
                        "panel_index": i,
                        "id": p["id"],
                        "choice": p["choice"],
                    }
                    for i, p in enumerate(rows(directory / f"{split}.jsonl"))
                )
            write_new(
                directory / "latency.json",
                probe(model, combined, PROTOCOL["depth"], panel["latency_indices"]),
            )
            write_new(
                directory / "completed.json",
                {
                    "status": "completed",
                    **config,
                    "plan_sha256": sha(plan_path),
                    "evaluation": evaluation,
                    "files": tree_files(directory),
                },
            )
            print(
                json.dumps(
                    {
                        "evaluated": config["name"],
                        "agreement": {s: r["metrics"]["agreement"] for s, r in evaluation.items()},
                    }
                ),
                flush=True,
            )
        grader = grading_module()
        directory = out / "regret"
        directory.mkdir()
        grading_plan = {
            "engine_path": str((ROOT / ENGINE).resolve()),
            "panels": eval_rows,
            "secondary_indices": panel["secondary_indices"],
            "configurations": panel["graded_configurations"],
        }
        cost = grader.score_secondary(grading_plan, directory, decisions)
        if (
            cost["calls"] > PROTOCOL["regret_call_ceiling"]
            or cost["requested_nodes"] > PROTOCOL["regret_node_ceiling"]
        ):
            raise ValueError("Grading budget exceeded")
        write_new(
            directory / "completed.json",
            {"status": "completed", "cost": cost, "files": tree_files(directory)},
        )
        arena.run(out / "arena", model_map, sha(plan_path))
        write_new(
            out / "completed.json",
            {
                "status": "completed",
                "plan_sha256": sha(plan_path),
                "wall_seconds": time.perf_counter() - begun,
                "files": tree_files(out),
            },
        )
    except Exception as exc:
        write_new(
            out / "failed.json",
            {
                "status": "failed",
                "plan_sha256": sha(plan_path),
                "error": str(exc),
                "wall_seconds": time.perf_counter() - begun,
            },
        )
        raise
    return out / "completed.json"


def cluster_interval(differences, game_ids, seed):
    keys = sorted(set(game_ids))
    sums, counts = np.zeros(len(keys)), np.zeros(len(keys))
    index = {key: i for i, key in enumerate(keys)}
    for difference, key in zip(differences, game_ids, strict=True):
        sums[index[key]] += difference
        counts[index[key]] += 1
    samples = np.random.default_rng(seed).integers(
        len(keys), size=(PROTOCOL["bootstrap_replicates"], len(keys))
    )
    values = sums[samples].sum(-1) / counts[samples].sum(-1)
    return {
        "games": len(keys),
        "positions": len(game_ids),
        "mean": float(np.mean(differences)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "scope": PROTOCOL["uncertainty"],
    }


def comparisons(metrics, predictions, regret, eval_rows, arena_summary):
    names = {c["name"] for c in configs()}
    if (
        set(metrics) != names
        or set(predictions) != names
        or set(regret) != names
        or set(eval_rows) != {"dev", "shift"}
        or any(
            set(v) != {"dev", "shift"}
            for collection in (metrics, predictions, regret)
            for v in collection.values()
        )
    ):
        raise ValueError("Incomplete capacity comparison coverage")
    means = {
        str(w): {
            s: {
                metric: float(np.mean([metrics[f"width{w}-{seed}"][s][metric] for seed in PROTOCOL["seeds"]]))
                for metric in ("agreement", "target_nll", "value_mae", "mean_confidence")
            }
            for s in ("dev", "shift")
        }
        for w in PROTOCOL["widths"]
    }
    checks, compared = [], []
    for offset, split in enumerate(("dev", "shift")):
        small = float(np.mean([regret[f"width32-{s}"][split] for s in PROTOCOL["seeds"]]))
        large = float(np.mean([regret[f"width128-{s}"][split] for s in PROTOCOL["seeds"]]))
        reduction = (small - large) / small if small > 0 else None
        differences = np.mean(
            [
                [
                    int(a["correct"]) - int(b["correct"])
                    for a, b in zip(
                        predictions[f"width128-{s}"][split], predictions[f"width32-{s}"][split], strict=True
                    )
                ]
                for s in PROTOCOL["seeds"]
            ],
            axis=0,
        )
        compared.append(
            {
                "split": split,
                "small_bounded_regret": small,
                "large_bounded_regret": large,
                "bounded_regret_change": large - small,
                "relative_reduction": reduction,
                "agreement_interval": cluster_interval(
                    differences, [r["game_id"] for r in eval_rows[split]], PROTOCOL["bootstrap_seed"] + offset
                ),
            }
        )
        checks.append(
            {
                "metric": "relative_regret_reduction",
                "split": split,
                "observed": reduction,
                "threshold": PROTOCOL["relative_regret_reduction"],
                "passed": reduction is not None
                and (
                    reduction >= PROTOCOL["relative_regret_reduction"]
                    or math.isclose(
                        reduction,
                        PROTOCOL["relative_regret_reduction"],
                        rel_tol=0,
                        abs_tol=PROTOCOL["gate_absolute_tolerance"],
                    )
                ),
            }
        )
    # Arena's audited gate uses the all-scheduled-game lower bound and rejects failed games.
    checks.append({"metric": "arena", "passed": arena_summary["gate_passed"]})
    return {
        "means": means,
        "comparisons": compared,
        "checks": checks,
        "continuation_passed": all(c["passed"] for c in checks),
    }


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


def report(plan_path, execution, out):
    plan = verify_plan(plan_path)
    execution = Path(execution)
    complete = read(execution / "completed.json")
    actual = tree_files(execution)
    actual.pop("completed.json")
    if (
        (execution / "failed.json").exists()
        or complete["status"] != "completed"
        or complete["plan_sha256"] != sha(plan_path)
        or complete["files"] != actual
        or read(execution / "started.json")["plan_sha256"] != sha(plan_path)
        or not math.isfinite(complete["wall_seconds"])
        or complete["wall_seconds"] < 0
    ):
        raise ValueError("Incomplete execution or evidence changed")
    exclusions = data.collect_exclusions(ROOT)
    data.validate(execution / "data", exclusions["states"])
    training = data.training_rows(ROOT, execution / "data")
    if len(training) != PROTOCOL["train_examples"]:
        raise ValueError("Training row count mismatch")
    eval_rows = {s: rows(execution / "data" / f"{s}.jsonl") for s in ("dev", "shift")}
    panel, combined = panels(eval_rows)
    if read(execution / "panels.json") != panel:
        raise ValueError("Evaluation selection changed")
    baseline = read(execution / "baselines.json")
    if baseline != {s: simple_baselines(training, value) for s, value in eval_rows.items()}:
        raise ValueError("Native baseline aggregates inconsistent")
    metrics, predictions, costs, latencies, decisions = {}, {}, {}, {}, []
    model_digests = {}
    for config in plan["configurations"]:
        name = config["name"]
        directory = execution / name
        receipt = read(directory / "completed.json")
        required = {
            "weights.pt",
            "learning.jsonl",
            "training.json",
            "latency.json",
            "dev.jsonl",
            "shift.jsonl",
        }
        if (
            set(receipt["files"]) != required
            or receipt["status"] != "completed"
            or receipt["plan_sha256"] != sha(plan_path)
            or any(receipt[k] != config[k] for k in config)
            or any(sha(directory / n) != h for n, h in receipt["files"].items())
        ):
            raise ValueError("Fit identity or receipt mismatch")
        model = AnchorChess.load(
            directory / "weights.pt",
            expected_plan_sha256=sha(plan_path),
            expected_recurrence="residual",
            expected_seed=config["seed"],
        )
        if model.width != config["width"] or model.depth != PROTOCOL["depth"]:
            raise ValueError("Checkpoint architecture mismatch")
        model_digests[name] = digest_state(model)
        trained, learning, steps = (
            read(directory / "training.json"),
            rows(directory / "learning.jsonl"),
            schedule(config["seed"]),
        )
        if (
            trained["status"] != "completed"
            or any(trained[k] != config[k] for k in config)
            or trained["plan_sha256"] != sha(plan_path)
            or trained["data_receipt_sha256"] != sha(execution / "data/completed.json")
            or trained["weights_sha256"] != sha(directory / "weights.pt")
            or trained["learning_sha256"] != sha(directory / "learning.jsonl")
            or len(learning) != len(steps)
            or trained["updates"] != len(steps)
            or trained["core_iterations"] != len(steps) * PROTOCOL["depth"]
            or trained["examples_seen"] != sum(len(s["indices"]) for s in steps)
            or not math.isfinite(trained["training_seconds"])
            or trained["training_seconds"] < 0
        ):
            raise ValueError("Training budget or provenance mismatch")
        expected_initial = digest_state(
            AnchorChess("residual", config["seed"], config["width"], PROTOCOL["depth"])
        )
        if trained["initial_state_sha256"] != expected_initial:
            raise ValueError("Initialization mismatch")
        for record, step in zip(learning, steps, strict=True):
            if (
                record["step"] != step["step"]
                or record["epoch"] != step["epoch"]
                or record["depth"] != PROTOCOL["depth"]
                or record["examples"] != len(step["indices"])
                or any(
                    not math.isfinite(record[k]) or record[k] < 0
                    for k in ("policy_ce", "value_mse", "loss", "gradient_norm")
                )
                or not math.isclose(
                    record["loss"],
                    record["policy_ce"] + PROTOCOL["value_loss_weight"] * record["value_mse"],
                    rel_tol=2e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("Learning journal mismatch")
        costs[name] = {
            k: trained[k] for k in ("updates", "core_iterations", "examples_seen", "training_seconds")
        }
        if set(receipt["evaluation"]) != {"dev", "shift"}:
            raise ValueError("Missing evaluation split")
        metrics[name], predictions[name] = {}, {}
        for split, value in eval_rows.items():
            calculated, preds = audit_predictions(directory / f"{split}.jsonl", value)
            elapsed = receipt["evaluation"][split]["evaluation_wall_seconds"]
            if (
                calculated != receipt["evaluation"][split]["metrics"]
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("Evaluation aggregate or time mismatch")
            metrics[name][split], predictions[name][split] = calculated, preds
            decisions.extend(
                {
                    "configuration": name,
                    "split": split,
                    "panel_index": i,
                    "id": p["id"],
                    "choice": p["choice"],
                }
                for i, p in enumerate(preds)
            )
        measurement = read(directory / "latency.json")
        audit_latency(measurement, combined, panel["latency_indices"])
        latencies[name] = measurement
    directory = execution / "regret"
    receipt = read(directory / "completed.json")
    if (
        receipt["status"] != "completed"
        or set(receipt["files"]) != {"engine.json", "analyses.jsonl", "regret.jsonl"}
        or any(sha(directory / n) != h for n, h in receipt["files"].items())
    ):
        raise ValueError("Grading receipt mismatch")
    engine = read(directory / "engine.json")
    if engine["sha256"] != plan["engine_sha256"] or not engine["id"]["name"].startswith("Stockfish 19"):
        raise ValueError("Grading engine mismatch")
    records, analyses = rows(directory / "regret.jsonl"), rows(directory / "analyses.jsonl")
    grading_plan = {
        "panels": eval_rows,
        "secondary_indices": panel["secondary_indices"],
        "configurations": panel["graded_configurations"],
    }
    grading_module().validate_secondary(grading_plan, decisions, analyses, records, receipt["cost"])
    if (
        receipt["cost"]["calls"] > PROTOCOL["regret_call_ceiling"]
        or receipt["cost"]["requested_nodes"] > PROTOCOL["regret_node_ceiling"]
    ):
        raise ValueError("Grading budget exceeded")
    regret = {
        c["name"]: {
            s: float(
                np.mean(
                    [
                        r["bounded_regret"]
                        for r in records
                        if r["configuration"] == c["name"] and r["split"] == s
                    ]
                )
            )
            for s in ("dev", "shift")
        }
        for c in configs()
    }
    arena_summary = arena.report(execution / "arena", sha(plan_path))
    bindings = read(execution / "arena/started.json")["models"]
    if set(bindings) != set(model_digests) or any(
        bindings[name]["state_sha256"] != digest for name, digest in model_digests.items()
    ):
        raise ValueError("Arena weights differ from the trained checkpoints")
    result = {
        "status": "completed",
        "plan_sha256": sha(plan_path),
        "execution_receipt_sha256": sha(execution / "completed.json"),
        "metrics": metrics,
        "regret": regret,
        "costs": costs,
        "latency": latencies,
        "engine_cost": receipt["cost"],
        "fresh_data_cost": read(execution / "data/completed.json"),
        "baselines": {s: {n: v["metrics"] for n, v in baseline[s].items()} for s in baseline},
        "arena": arena_summary,
        **comparisons(metrics, predictions, regret, eval_rows, arena_summary),
        "scope": PROTOCOL["scope"],
        "novelty_established": False,
        "elo_estimate": None,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "summary.json", result)
    write_new(
        out / "completed.json",
        {
            "status": "completed",
            "plan_sha256": sha(plan_path),
            "summary_sha256": sha(out / "summary.json"),
            "execution_receipt_sha256": sha(execution / "completed.json"),
        },
    )
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
        print(
            json.dumps(
                {k: result[k] for k in ("means", "comparisons", "checks", "continuation_passed")}, indent=2
            )
        )


if __name__ == "__main__":
    main()
