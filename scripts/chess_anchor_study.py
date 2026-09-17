"""Parameter- and iteration-matched recurrence ablation on fresh chess positions.

Four fixed arms separate the skip-input change from balanced random-depth
training. There is no new world-model, reinforcement-learning or novelty claim.
Prepare freezes sources, original training inputs, exclusions and all budgets.
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

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_anchor_data import (
    FRESH_CONFIG,
    collect_exclusions,
    generate_fresh,
    validate_fresh,
)
from openjev.research.chess_anchor_eval import audit_predictions, evaluate, probe, tensorize
from openjev.research.chess_spatial_baselines import simple_baselines

ROOT = Path(__file__).resolve().parents[1]
TRAIN = "runs/chess-spatial-v1/execution/data/train.jsonl"
OLD_PLAN = "evidence/chess-spatial-v1/plan.json"
ENGINE = "runs/chess-inputs/stockfish/stockfish-macos-universal"
SOURCES = [
    "scripts/chess_anchor_study.py",
    "tests/test_chess_anchor_study.py",
    "src/openjev/research/chess_anchor.py",
    "tests/test_chess_anchor.py",
    "src/openjev/research/chess_anchor_data.py",
    "tests/test_chess_anchor_data.py",
    "src/openjev/research/chess_anchor_eval.py",
    "tests/test_chess_anchor_eval.py",
    "src/openjev/research/chess_spatial.py",
    "src/openjev/research/chess_spatial_data.py",
    "src/openjev/research/chess_spatial_baselines.py",
    "src/openjev/research/chess_arena.py",
    "scripts/chess_compute_study.py",
    "src/openjev/research/chess_compute.py",
]
ARMS = {
    "residual_fixed": ("residual", "fixed"),
    "residual_mixed": ("residual", "mixed"),
    "anchor_fixed": ("anchor", "fixed"),
    "anchor_mixed": ("anchor", "mixed"),
}
PROTOCOL = {
    "version": "chess-anchor-v1",
    "arms": ARMS,
    "seeds": [17, 29, 43],
    "width": 32,
    "default_depth": 4,
    "train_examples": 32768,
    "epochs": 6,
    "batch_size": 128,
    "updates_per_fit": 1536,
    "training_device": "mps",
    "evaluation_device": "cpu",
    "torch_threads": 2,
    "deterministic_algorithms": False,
    "mixed_depths": [2, 4, 6],
    "evaluation_depths": [2, 4, 8, 16],
    "batch_order_seed_base": 10400017,
    "depth_order_seed_base": 10400029,
    "fit_order_seed": 10400043,
    "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "gradient_clip": 1.0,
    "value_loss_weight": 0.5,
    "loss": "Legal-move CE + 0.5 bounded-value MSE; auxiliary head unused in every arm.",
    "initialization": "Fresh same-seed identical state_dict across all four arms. First iteration also matches.",
    "recurrence": "h0=x=encoder(board); residual: ReLU(h+F(h)); anchor: ReLU(x+F(h)); F has the same two convolutions.",
    "training_depth": "Fixed4 versus a separately seeded shuffle of exactly512 updates at each of2/4/6. Same minibatches per seed.",
    "core_iterations_per_fit": 6144,
    "regret_positions_per_split": 128,
    "regret_selection_seed": 10400059,
    "regret_nodes": 20000,
    "regret_configuration": "Both mixed arms, depths4/8, allthree seeds; atmost3328 calls,66560000 requestednodes.",
    "regret_call_ceiling": 3328,
    "regret_node_ceiling": 66560000,
    "latency_positions_per_split": 32,
    "latency_selection_seed": 10400071,
    "latency_warmups": 3,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 10400089,
    "uncertainty": "Pointwise game-cluster bootstrap, conditional on the three fitted seeds; not simultaneous or seed-population uncertainty.",
    "gate": {"primary_gain": 0.02, "worst_seed_deficit": 0.01, "extra_depth_gain": 0.01},
    "evaluation": "Fresh prospectively frozen positions from previously studied generators and shift; not broad independent confirmation.",
    "selection": "Final checkpoints only. No retries, early stopping, budget extension, or choosing a different winning depth after results.",
    "scope": "Controlled skip-source and depth-training ablation; no proved stability mechanism, biological claim, Elo, world-model planning, or novelty.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def digest_state(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configs():
    values = [
        {"arm": arm, "seed": seed, "name": f"{arm}-{seed}", "recurrence": recurrence, "regime": regime}
        for arm, (recurrence, regime) in ARMS.items()
        for seed in PROTOCOL["seeds"]
    ]
    random.Random(PROTOCOL["fit_order_seed"]).shuffle(values)
    return values


def schedule(seed, regime):
    if regime not in ("fixed", "mixed"):
        raise ValueError("Unknown depth regime")
    updates = PROTOCOL["updates_per_fit"]
    if updates % 3:
        raise ValueError("Mixed budget must divide exactly into three depths")
    depths = PROTOCOL["mixed_depths"] * (updates // 3)
    random.Random(PROTOCOL["depth_order_seed_base"] + seed).shuffle(depths)
    generator = torch.Generator().manual_seed(PROTOCOL["batch_order_seed_base"] + seed)
    result = []
    for epoch in range(PROTOCOL["epochs"]):
        order = torch.randperm(PROTOCOL["train_examples"], generator=generator).tolist()
        for start in range(0, len(order), PROTOCOL["batch_size"]):
            result.append(
                {
                    "step": len(result) + 1,
                    "epoch": epoch + 1,
                    "depth": 4 if regime == "fixed" else depths[len(result)],
                    "indices": order[start : start + PROTOCOL["batch_size"]],
                }
            )
    if len(result) != updates or sum(r["depth"] for r in result) != PROTOCOL["core_iterations_per_fit"]:
        raise ValueError("Training schedule does not match update/core budget")
    return result


def signature():
    exclusions = collect_exclusions(ROOT)
    original = read(ROOT / OLD_PLAN)
    source = ROOT / "runs/chess-spatial-v1/execution/data/completed.json"
    if read(source)["files"]["train.jsonl"] != sha(ROOT / TRAIN):
        raise ValueError("Original training data changed")
    if original["sources"]["src/openjev/research/chess_spatial.py"] != sha(
        ROOT / "src/openjev/research/chess_spatial.py"
    ):
        raise ValueError("Original model source changed")
    model = AnchorChess("residual", 17)
    aux = sum(p.numel() for p in model.aux_head.parameters())
    return {
        "protocol": PROTOCOL,
        "fresh_data": FRESH_CONFIG,
        "configurations": configs(),
        "sources": {name: sha(ROOT / name) for name in SOURCES},
        "exclusion_files": exclusions["files"],
        "exclusion_counts": exclusions["counts"],
        "exclusion_states_sha256": hashlib.sha256(
            json.dumps(exclusions["states"], separators=(",", ":")).encode()
        ).hexdigest(),
        "original_plan_sha256": sha(ROOT / OLD_PLAN),
        "train_sha256": sha(ROOT / TRAIN),
        "train_receipt_sha256": sha(source),
        "engine_sha256": sha(ROOT / ENGINE),
        "stored_parameters": model.parameter_count(),
        "active_parameters": model.parameter_count() - aux,
        "environment": {
            "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "chess", "numpy")},
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mps_available": torch.backends.mps.is_available(),
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
    frozen = read(path)
    # JSON represents the two-element arm tuples as lists.
    if frozen != json.loads(json.dumps(signature())):
        raise ValueError("Frozen input/source/environment mismatch")
    return frozen


def _sync():
    if PROTOCOL["training_device"] == "mps":
        torch.mps.synchronize()


def fit(config, train, out, plan_hash, data_hash):
    out.mkdir()
    model = AnchorChess(config["recurrence"], config["seed"], PROTOCOL["width"], 4)
    initial = digest_state(model)
    model.to(PROTOCOL["training_device"])
    opt = PROTOCOL["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt["lr"], betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=0.0
    )
    steps = schedule(config["seed"], config["regime"])
    _sync()
    started = time.perf_counter()
    with (out / "learning.jsonl").open("x") as stream:
        for step in steps:
            b = {key: value[step["indices"]].to(PROTOCOL["training_device"]) for key, value in train.items()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, values, _ = model(b["observations"], b["candidates"], b["mask"], depth=step["depth"])
            ce = F.cross_entropy(logits, b["targets"])
            mse = F.mse_loss(values, b["values"])
            loss = ce + PROTOCOL["value_loss_weight"] * mse
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            append(
                stream,
                {
                    "step": step["step"],
                    "epoch": step["epoch"],
                    "depth": step["depth"],
                    "examples": len(step["indices"]),
                    "policy_ce": float(ce.detach().cpu()),
                    "value_mse": float(mse.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient.detach().cpu()),
                },
            )
            if step["step"] % 256 == 0:
                print(json.dumps({"fit": config["name"], "updates": step["step"]}), flush=True)
    _sync()
    elapsed = time.perf_counter() - started
    model.cpu().save(out / "weights.pt", plan_sha256=plan_hash)
    result = {
        "status": "completed",
        **config,
        "plan_sha256": plan_hash,
        "data_receipt_sha256": data_hash,
        "initial_state_sha256": initial,
        "updates": len(steps),
        "core_iterations": sum(s["depth"] for s in steps),
        "examples_seen": sum(len(s["indices"]) for s in steps),
        "training_seconds": elapsed,
        "weights_sha256": sha(out / "weights.pt"),
        "learning_sha256": sha(out / "learning.jsonl"),
    }
    write_new(out / "training.json", result)
    return result


def grading_module():
    spec = importlib.util.spec_from_file_location(
        "frozen_compute_grading", ROOT / "scripts/chess_compute_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.PROTOCOL["regret_nodes"] != PROTOCOL["regret_nodes"]
        or module.PROTOCOL["value_cp_scale"] != FRESH_CONFIG["value_cp_scale"]
        or module.PROTOCOL["mate_cp"] != FRESH_CONFIG["mate_cp"]
    ):
        raise ValueError("Frozen grading helper budget/scale mismatch")
    return module


def panels(eval_rows):
    secondary = {}
    latency = []
    combined = []
    for offset, split in enumerate(("dev", "shift")):
        n = len(eval_rows[split])
        secondary[split] = sorted(
            random.Random(PROTOCOL["regret_selection_seed"] + offset).sample(
                range(n), PROTOCOL["regret_positions_per_split"]
            )
        )
        sampled = sorted(
            random.Random(PROTOCOL["latency_selection_seed"] + offset).sample(
                range(n), PROTOCOL["latency_positions_per_split"]
            )
        )
        latency.extend(len(combined) + index for index in sampled)
        combined.extend(eval_rows[split])
    graded = [
        {"id": f"{c['name']}-d{depth}"} for c in configs() if c["regime"] == "mixed" for depth in (4, 8)
    ]
    return {
        "secondary_indices": secondary,
        "latency_indices": latency,
        "graded_configurations": graded,
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
        torch.use_deterministic_algorithms(False)
        exclusion = collect_exclusions(ROOT)
        generate_fresh(out / "data", ROOT / ENGINE, exclusion["states"])
        validate_fresh(out / "data", exclusion["states"])
        original_rows = rows(ROOT / TRAIN)
        if len(original_rows) != PROTOCOL["train_examples"]:
            raise ValueError("Training row count mismatch")
        train, _ = tensorize(original_rows)
        eval_rows = {split: rows(out / "data" / f"{split}.jsonl") for split in ("dev", "shift")}
        prepared = {split: tensorize(value) for split, value in eval_rows.items()}
        panel, combined = panels(eval_rows)
        write_new(out / "panels.json", panel)
        write_new(
            out / "baselines.json",
            {split: simple_baselines(original_rows, value) for split, value in eval_rows.items()},
        )
        data_hash = sha(out / "data/completed.json")
        initial = {}
        for config in frozen["configurations"]:
            result = fit(config, train, out / config["name"], sha(plan_path), data_hash)
            previous = initial.setdefault(config["seed"], result["initial_state_sha256"])
            if result["initial_state_sha256"] != previous:
                raise ValueError("Same-seed initialization differs across arms")
        # All fitting is complete before any fresh neural evaluation is exposed.
        decisions = []
        fit_hashes = {}
        for config in frozen["configurations"]:
            directory = out / config["name"]
            model = AnchorChess.load(
                directory / "weights.pt",
                expected_plan_sha256=sha(plan_path),
                expected_recurrence=config["recurrence"],
                expected_seed=config["seed"],
            )
            reports = {}
            latencies = {}
            for depth in PROTOCOL["evaluation_depths"]:
                for split, value in eval_rows.items():
                    data, menus = prepared[split]
                    evaluation = evaluate(
                        model, value, data, menus, depth, directory / f"{split}-d{depth}.jsonl"
                    )
                    reports[f"{split}-d{depth}"] = evaluation
                    if config["regime"] == "mixed" and depth in (4, 8):
                        for index, prediction in enumerate(rows(directory / f"{split}-d{depth}.jsonl")):
                            decisions.append(
                                {
                                    "configuration": f"{config['name']}-d{depth}",
                                    "split": split,
                                    "panel_index": index,
                                    "id": prediction["id"],
                                    "choice": prediction["choice"],
                                }
                            )
                latencies[str(depth)] = probe(model, combined, depth, panel["latency_indices"])
            write_new(directory / "latency.json", latencies)
            write_new(
                directory / "completed.json",
                {
                    "status": "completed",
                    **config,
                    "plan_sha256": sha(plan_path),
                    "evaluation": reports,
                    "files": {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()},
                },
            )
            fit_hashes[config["name"]] = sha(directory / "completed.json")
            print(
                json.dumps(
                    {
                        "evaluated": config["name"],
                        "agreement": {s: v["metrics"]["agreement"] for s, v in reports.items()},
                    }
                ),
                flush=True,
            )
        grader = grading_module()
        regret = out / "regret"
        regret.mkdir()
        grading_plan = {
            "engine_path": str((ROOT / ENGINE).resolve()),
            "panels": eval_rows,
            "secondary_indices": panel["secondary_indices"],
            "configurations": panel["graded_configurations"],
        }
        cost = grader.score_secondary(grading_plan, regret, decisions)
        if (
            cost["calls"] > PROTOCOL["regret_call_ceiling"]
            or cost["requested_nodes"] > PROTOCOL["regret_node_ceiling"]
        ):
            raise ValueError("Stronger-engine call ceiling exceeded")
        write_new(
            regret / "completed.json",
            {
                "status": "completed",
                "cost": cost,
                "files": {p.name: sha(p) for p in sorted(regret.iterdir()) if p.is_file()},
            },
        )
        write_new(
            out / "completed.json",
            {
                "status": "completed",
                "plan_sha256": sha(plan_path),
                "fits": fit_hashes,
                "data_receipt_sha256": data_hash,
                "panels_sha256": sha(out / "panels.json"),
                "baselines_sha256": sha(out / "baselines.json"),
                "regret_receipt_sha256": sha(regret / "completed.json"),
                "wall_seconds": time.perf_counter() - begun,
            },
        )
    except Exception as exc:
        write_new(
            out / "failed.json",
            {
                "status": "failed",
                "plan_sha256": sha(plan_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - begun,
            },
        )
        raise
    return out / "completed.json"


def cluster_interval(differences, game_ids, seed):
    keys = sorted(set(game_ids))
    sums = np.zeros(len(keys))
    counts = np.zeros(len(keys))
    index = {key: i for i, key in enumerate(keys)}
    for difference, key in zip(differences, game_ids, strict=True):
        sums[index[key]] += difference
        counts[index[key]] += 1
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(keys), size=(PROTOCOL["bootstrap_replicates"], len(keys)))
    values = sums[samples].sum(-1) / counts[samples].sum(-1)
    return {
        "games": len(keys),
        "positions": len(game_ids),
        "mean": float(np.mean(differences)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "scope": PROTOCOL["uncertainty"],
    }


def gate_and_comparisons(metrics, predictions, regret, eval_rows):
    expected_models = {c["name"] for c in configs()}
    expected_panels = {
        f"{split}-d{depth}" for split in ("dev", "shift") for depth in PROTOCOL["evaluation_depths"]
    }
    expected_grades = {
        f"{c['name']}-d{depth}" for c in configs() if c["regime"] == "mixed" for depth in (4, 8)
    }
    if (
        set(metrics) != expected_models
        or set(predictions) != expected_models
        or set(regret) != expected_grades
        or set(eval_rows) != {"dev", "shift"}
        or any(
            set(metrics[name]) != expected_panels or set(predictions[name]) != expected_panels
            for name in expected_models
        )
        or any(set(value) != {"dev", "shift"} for value in regret.values())
    ):
        raise ValueError("Gate requires complete fixed model, depth and split coverage")
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for depth in PROTOCOL["evaluation_depths"]:
            means[arm][str(depth)] = {
                split: float(
                    np.mean(
                        [
                            metrics[f"{arm}-{seed}"][f"{split}-d{depth}"]["agreement"]
                            for seed in PROTOCOL["seeds"]
                        ]
                    )
                )
                for split in ("dev", "shift")
            }
    checks = []
    comparisons = []
    for offset, split in enumerate(("dev", "shift")):
        for label, candidate, cd, reference, rd in [
            ("primary", "anchor_mixed", 4, "residual_mixed", 4),
            ("extra_depth", "anchor_mixed", 8, "anchor_mixed", 4),
            ("depth8_architecture", "anchor_mixed", 8, "residual_mixed", 8),
        ]:
            deltas = [
                metrics[f"{candidate}-{seed}"][f"{split}-d{cd}"]["agreement"]
                - metrics[f"{reference}-{seed}"][f"{split}-d{rd}"]["agreement"]
                for seed in PROTOCOL["seeds"]
            ]
            position_deltas = np.mean(
                [
                    [
                        int(a["correct"]) - int(b["correct"])
                        for a, b in zip(
                            predictions[f"{candidate}-{seed}"][f"{split}-d{cd}"],
                            predictions[f"{reference}-{seed}"][f"{split}-d{rd}"],
                            strict=True,
                        )
                    ]
                    for seed in PROTOCOL["seeds"]
                ],
                axis=0,
            )
            interval = cluster_interval(
                position_deltas, [r["game_id"] for r in eval_rows[split]], PROTOCOL["bootstrap_seed"] + offset
            )
            engine_delta = float(
                np.mean(
                    [
                        regret[f"{candidate}-{seed}-d{cd}"][split]
                        - regret[f"{reference}-{seed}-d{rd}"][split]
                        for seed in PROTOCOL["seeds"]
                    ]
                )
            )
            comparisons.append(
                {
                    "comparison": label,
                    "split": split,
                    "seed_agreement_deltas": deltas,
                    "game_cluster_interval": interval,
                    "bounded_regret_delta": engine_delta,
                }
            )
            threshold = (
                PROTOCOL["gate"]["primary_gain"]
                if label == "primary"
                else PROTOCOL["gate"]["extra_depth_gain"]
                if label == "extra_depth"
                else 0.0
            )
            checks.append(
                {
                    "gate": label,
                    "split": split,
                    "metric": "agreement_gain",
                    "observed": float(np.mean(deltas)),
                    "threshold": threshold,
                    "passed": float(np.mean(deltas)) >= threshold
                    if label != "depth8_architecture"
                    else float(np.mean(deltas)) > 0,
                }
            )
            checks.append(
                {
                    "gate": label,
                    "split": split,
                    "metric": "bounded_regret_change",
                    "observed": engine_delta,
                    "threshold": 0.0,
                    "passed": engine_delta < 0 if label == "primary" else engine_delta <= 0,
                }
            )
            if label == "primary":
                checks.append(
                    {
                        "gate": label,
                        "split": split,
                        "metric": "worst_seed_gain",
                        "observed": min(deltas),
                        "threshold": -PROTOCOL["gate"]["worst_seed_deficit"],
                        "passed": min(deltas) >= -PROTOCOL["gate"]["worst_seed_deficit"],
                    }
                )
    interactions = {
        split: means["anchor_mixed"]["4"][split]
        - means["residual_mixed"]["4"][split]
        - means["anchor_fixed"]["4"][split]
        + means["residual_fixed"]["4"][split]
        for split in ("dev", "shift")
    }
    return {
        "means": means,
        "comparisons": comparisons,
        "interaction_at_depth4": interactions,
        "primary_passed": all(c["passed"] for c in checks if c["gate"] == "primary"),
        "extra_compute_passed": all(c["passed"] for c in checks if c["gate"] != "primary"),
        "checks": checks,
    }


def report(plan_path, execution, out):
    plan = verify_plan(plan_path)
    execution = Path(execution)
    complete = read(execution / "completed.json")
    expected = {c["name"] for c in plan["configurations"]}
    if (
        (execution / "failed.json").exists()
        or complete["status"] != "completed"
        or complete["plan_sha256"] != sha(plan_path)
        or set(complete["fits"]) != expected
        or read(execution / "started.json")["plan_sha256"] != sha(plan_path)
    ):
        raise ValueError("Incomplete or incorrectly bound execution")
    exclusion = collect_exclusions(ROOT)
    validate_fresh(execution / "data", exclusion["states"])
    if sha(execution / "data/completed.json") != complete["data_receipt_sha256"]:
        raise ValueError("Fresh data changed")
    eval_rows = {split: rows(execution / "data" / f"{split}.jsonl") for split in ("dev", "shift")}
    panel, combined = panels(eval_rows)
    if (
        read(execution / "panels.json") != panel
        or sha(execution / "panels.json") != complete["panels_sha256"]
    ):
        raise ValueError("Grading or latency panel changed")
    if sha(execution / "baselines.json") != complete["baselines_sha256"]:
        raise ValueError("Baseline evidence changed")
    baseline = read(execution / "baselines.json")
    original_rows = rows(ROOT / TRAIN)
    if baseline != {split: simple_baselines(original_rows, value) for split, value in eval_rows.items()}:
        raise ValueError("Native baseline evidence inconsistent")
    metrics = {}
    predictions = {}
    initial = {}
    costs = {}
    latency = {}
    decisions = []
    for config in plan["configurations"]:
        name = config["name"]
        directory = execution / name
        receipt = read(directory / "completed.json")
        if sha(directory / "completed.json") != complete["fits"][name]:
            raise ValueError("Fit receipt changed")
        required = {"weights.pt", "learning.jsonl", "training.json", "latency.json"} | {
            f"{s}-d{d}.jsonl" for s in ("dev", "shift") for d in PROTOCOL["evaluation_depths"]
        }
        if (
            set(receipt["files"]) != required
            or receipt["status"] != "completed"
            or receipt["plan_sha256"] != sha(plan_path)
            or any(receipt[k] != config[k] for k in config)
        ):
            raise ValueError("Fit identity or membership mismatch")
        for filename, digest in receipt["files"].items():
            if sha(directory / filename) != digest:
                raise ValueError("Fit evidence hash mismatch")
        model = AnchorChess.load(
            directory / "weights.pt",
            expected_plan_sha256=sha(plan_path),
            expected_recurrence=config["recurrence"],
            expected_seed=config["seed"],
        )
        if model.width != PROTOCOL["width"] or model.depth != 4:
            raise ValueError("Checkpoint architecture mismatch")
        trained = read(directory / "training.json")
        steps = schedule(config["seed"], config["regime"])
        learning = rows(directory / "learning.jsonl")
        if (
            len(learning) != len(steps)
            or trained["status"] != "completed"
            or trained["plan_sha256"] != sha(plan_path)
            or trained["data_receipt_sha256"] != complete["data_receipt_sha256"]
            or any(trained[k] != config[k] for k in config)
            or trained["weights_sha256"] != sha(directory / "weights.pt")
            or trained["learning_sha256"] != sha(directory / "learning.jsonl")
            or trained["updates"] != len(steps)
            or trained["core_iterations"] != sum(s["depth"] for s in steps)
            or trained["examples_seen"] != sum(len(s["indices"]) for s in steps)
        ):
            raise ValueError("Training provenance or compute budget mismatch")
        for row, step in zip(learning, steps, strict=True):
            if (
                any(row[k] != step[k] for k in ("step", "epoch", "depth"))
                or row["examples"] != len(step["indices"])
                or any(
                    not math.isfinite(row[k]) or row[k] < 0
                    for k in ("policy_ce", "value_mse", "loss", "gradient_norm")
                )
                or not math.isclose(
                    row["loss"],
                    row["policy_ce"] + PROTOCOL["value_loss_weight"] * row["value_mse"],
                    rel_tol=2e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("Training schedule or loss journal mismatch")
        expected_initial = digest_state(
            AnchorChess(config["recurrence"], config["seed"], PROTOCOL["width"], 4)
        )
        if trained["initial_state_sha256"] != expected_initial:
            raise ValueError("Initialization digest mismatch")
        previous = initial.setdefault(config["seed"], expected_initial)
        if previous != expected_initial:
            raise ValueError("Across-arm initializer mismatch")
        if not math.isfinite(trained["training_seconds"]) or trained["training_seconds"] < 0:
            raise ValueError("Invalid training time")
        costs[name] = {
            k: trained[k] for k in ("training_seconds", "updates", "core_iterations", "examples_seen")
        }
        metrics[name] = {}
        predictions[name] = {}
        if set(receipt["evaluation"]) != {
            f"{s}-d{d}" for s in ("dev", "shift") for d in PROTOCOL["evaluation_depths"]
        }:
            raise ValueError("Evaluation depth/split coverage mismatch")
        for depth in PROTOCOL["evaluation_depths"]:
            for split, value in eval_rows.items():
                key = f"{split}-d{depth}"
                calculated, preds = audit_predictions(directory / f"{key}.jsonl", value)
                if calculated != receipt["evaluation"][key]["metrics"]:
                    raise ValueError("Evaluation aggregate mismatch")
                elapsed = receipt["evaluation"][key]["evaluation_wall_seconds"]
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Invalid evaluation wall time")
                metrics[name][key] = calculated
                predictions[name][key] = preds
                if config["regime"] == "mixed" and depth in (4, 8):
                    decisions.extend(
                        {
                            "configuration": f"{name}-d{depth}",
                            "split": split,
                            "panel_index": index,
                            "id": p["id"],
                            "choice": p["choice"],
                        }
                        for index, p in enumerate(preds)
                    )
        latency[name] = read(directory / "latency.json")
        if set(latency[name]) != {str(d) for d in PROTOCOL["evaluation_depths"]}:
            raise ValueError("Latency depth coverage mismatch")
        # The helper validates warmup and decision output shape; independently audit selected IDs, legality and timing.
        for depth, measurement in latency[name].items():
            required_latency = {
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
                set(measurement) != required_latency
                or measurement["device"] != "cpu"
                or measurement["torch_threads"] != PROTOCOL["torch_threads"]
                or measurement["depth"] != int(depth)
                or measurement["timing_scope"]
                != "Full single-position choose call, including board construction, encoding and legal-response validation"
                or measurement["warmup_policy"]
                != "Three standard-starting-board calls before the fixed selected positions"
            ):
                raise ValueError("Latency device, depth, thread count or timing scope mismatch")
            warmups = measurement["warmup_records"]
            if len(warmups) != PROTOCOL["latency_warmups"]:
                raise ValueError("Latency warmup count mismatch")
            for index, warmup in enumerate(warmups):
                if (
                    warmup["id"] != "starting-board"
                    or warmup["index"] != index
                    or chess.Move.from_uci(warmup["choice"]) not in chess.Board().legal_moves
                    or not math.isfinite(warmup["wall_ms"])
                    or warmup["wall_ms"] < 0
                ):
                    raise ValueError("Invalid latency warmup record")
            if not math.isclose(
                measurement["warmup_wall_ms"], math.fsum(w["wall_ms"] for w in warmups), abs_tol=1e-8
            ):
                raise ValueError("Latency warmup total mismatch")
            records = measurement["records"]
            if len(records) != len(panel["latency_indices"]):
                raise ValueError("Incomplete latency panel")
            for measured, index in zip(records, panel["latency_indices"], strict=True):
                source = combined[index]
                if (
                    measured["id"] != source["id"]
                    or measured["index"] != index
                    or chess.Move.from_uci(measured["choice"]) not in chess.Board(source["fen"]).legal_moves
                    or not math.isfinite(measured["wall_ms"])
                    or measured["wall_ms"] < 0
                ):
                    raise ValueError("Invalid latency record")
            if not math.isclose(
                measurement["total_wall_ms"], math.fsum(r["wall_ms"] for r in records), abs_tol=1e-8
            ):
                raise ValueError("Latency total mismatch")
    directory = execution / "regret"
    regret_receipt = read(directory / "completed.json")
    if (
        regret_receipt["status"] != "completed"
        or sha(directory / "completed.json") != complete["regret_receipt_sha256"]
        or set(regret_receipt["files"]) != {"engine.json", "analyses.jsonl", "regret.jsonl"}
    ):
        raise ValueError("Incomplete engine grading")
    for name, digest in regret_receipt["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Engine grading changed")
    engine = read(directory / "engine.json")
    if engine["sha256"] != plan["engine_sha256"] or not engine["id"]["name"].startswith("Stockfish 19"):
        raise ValueError("Grading engine identity mismatch")
    grading_plan = {
        "panels": eval_rows,
        "secondary_indices": panel["secondary_indices"],
        "configurations": panel["graded_configurations"],
    }
    records = rows(directory / "regret.jsonl")
    analyses = rows(directory / "analyses.jsonl")
    grading_module().validate_secondary(grading_plan, decisions, analyses, records, regret_receipt["cost"])
    if (
        regret_receipt["cost"]["calls"] > PROTOCOL["regret_call_ceiling"]
        or regret_receipt["cost"]["requested_nodes"] > PROTOCOL["regret_node_ceiling"]
    ):
        raise ValueError("Grading budget exceeded")
    regret = {
        c["id"]: {
            split: float(
                np.mean(
                    [
                        r["bounded_regret"]
                        for r in records
                        if r["configuration"] == c["id"] and r["split"] == split
                    ]
                )
            )
            for split in ("dev", "shift")
        }
        for c in panel["graded_configurations"]
    }
    result = {
        "status": "completed",
        "plan_sha256": sha(plan_path),
        "execution_receipt_sha256": sha(execution / "completed.json"),
        "metrics": metrics,
        "regret": regret,
        "costs": costs,
        "latency": latency,
        "engine_cost": regret_receipt["cost"],
        "fresh_data_cost": read(execution / "data/completed.json"),
        "baselines": {s: {n: v["metrics"] for n, v in baseline[s].items()} for s in baseline},
        **gate_and_comparisons(metrics, predictions, regret, eval_rows),
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
                {key: result[key] for key in ("means", "primary_passed", "extra_compute_passed", "checks")},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
