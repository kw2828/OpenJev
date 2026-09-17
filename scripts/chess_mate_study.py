"""Fixed-budget mate-set versus single-move fine-tuning of our spatial model.

This is targeted policy supervision, not a new architecture or reinforcement
learning. All training labels stay outside model inputs; deployment performs no
mate enumeration. A fresh puzzle-game split tests the narrow mating skill, while
previously observed ordinary/shifted panels check forgetting.
"""

import argparse
import hashlib
import importlib.metadata
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

from openjev.research.chess_mate_data import select_mates
from openjev.research.chess_spatial import SpatialChess, encode_board, encode_candidates
from openjev.research.chess_spatial_data import state_key

ROOT = Path(__file__).resolve().parents[1]
CSV = "runs/chess-inputs/lichess-prefix.csv"
OLD_PLAN = "evidence/chess-v1/plan.json"
SPATIAL_PLAN = "evidence/chess-spatial-v1/plan.json"
DATA = "runs/chess-spatial-v1/execution/data"
EXCLUSIONS = [f"evidence/chess-student-v1/results/execution/data/{s}.jsonl" for s in ("train", "dev")]
EXCLUSIONS += [f"{DATA}/{s}.jsonl" for s in ("train", "dev", "shift")]
SOURCES = [
    "scripts/chess_mate_study.py",
    "tests/test_chess_mate_study.py",
    "src/openjev/research/chess_mate_data.py",
    "tests/test_chess_mate_data.py",
    "src/openjev/research/chess_spatial.py",
    "src/openjev/research/chess_spatial_data.py",
]
PROTOCOL = {
    "version": "chess-mate-v1",
    "seeds": [17, 29, 43],
    "arms": ["single", "set"],
    "base_mode": "predict",
    "depth": 4,
    "width": 32,
    "mate_sizes": [1024, 256, 512],
    "selection_seed": 10110017,
    "epochs": 12,
    "batch_size": 128,
    "mate_per_batch": 64,
    "updates_per_fit": 192,
    "replay_count_per_fit": 12288,
    "optimizer": {"name": "Adam", "lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "value_weight": 0.5,
    "gradient_clip": 1.0,
    "device": "mps",
    "torch_threads": 2,
    "batch_seed_base": 10110029,
    "fit_order_seed": 10110043,
    "deterministic_algorithms": False,
    "loss": "Mean policy loss over 64 mate and 64 replay examples + 0.5 mean value MSE. Mate value=1; replay keeps original teacher value.",
    "single": "Cross entropy on the recorded Lichess mating move.",
    "set": "Negative log total probability of all legal immediate mating moves; ordinary replay uses single teacher CE.",
    "replay": "12288 distinct original training rows sampled per seed, used once each; identical across arms.",
    "initialization": "Existing final predict checkpoint for each seed; fresh identical optimizer; same batches across arms.",
    "auxiliary": "Future-board head is unused during fine-tuning and inference in both arms.",
    "evaluation": "Every original and final checkpoint, mate dev/confirm and full spatial dev/shift. No checkpoint selection.",
    "gate": {"mate_gain_over_frozen": 0.20, "max_ordinary_mean_drop": 0.01, "set_gain_over_single": 0.05},
    "scope": "Narrow mate-in-one generalization to previously unused source games from one cached prefix. Ordinary panels are reused development data. No Elo, RL or architectural novelty claim.",
    "retries": 0,
    "training_extension": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
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


def signature():
    names = SOURCES + EXCLUSIONS + [CSV, OLD_PLAN, SPATIAL_PLAN, "pyproject.toml", "uv.lock"]
    names += [f"models/chess-spatial-v1/predict-{seed}/weights.pt" for seed in PROTOCOL["seeds"]]
    return {
        "protocol": PROTOCOL,
        "inputs": {name: sha(ROOT / name) for name in names},
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "chess")},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def selection():
    old = json.loads((ROOT / OLD_PLAN).read_text())
    games = {row["source_game"] for row in old["puzzles"]["positions"]}
    excluded = {state_key(chess.Board(row["fen"])) for name in EXCLUSIONS for row in read_rows(ROOT / name)}
    excluded.update(state_key(chess.Board(row["solver_fen"])) for row in old["puzzles"]["positions"])
    return select_mates(
        ROOT / CSV, games, excluded, sizes=tuple(PROTOCOL["mate_sizes"]), seed=PROTOCOL["selection_seed"]
    )


def prepare(out):
    value = signature()
    selected = selection()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "selection.json", selected)
    write_new(out / "plan.json", {**value, "selection_sha256": sha(out / "selection.json")})
    write_new(
        out / "prepared.json",
        {
            "status": "prepared",
            "plan_sha256": sha(out / "plan.json"),
            "created_unix": time.time(),
            "model_scoring_started": False,
        },
    )
    return out / "plan.json"


def verify_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    if plan != {**signature(), "selection_sha256": sha(path.parent / "selection.json")}:
        raise ValueError("Frozen source, data, checkpoint or environment mismatch")
    selected = json.loads((path.parent / "selection.json").read_text())
    if selected != selection():
        raise ValueError("Frozen puzzle selection does not reproduce")
    return plan, selected


def tensors(rows):
    """Encode only the board and legal move IDs; label fields form separate tensors."""
    boards = [chess.Board(row["fen"]) for row in rows]
    menus = [encode_candidates(board) for board in boards]
    longest = max(len(names) for names, _ in menus)
    features = np.zeros((len(rows), longest, 5), dtype=np.int64)
    legal = np.zeros((len(rows), longest), dtype=bool)
    mating = np.zeros_like(legal)
    targets = []
    for i, (row, (names, values)) in enumerate(zip(rows, menus, strict=True)):
        features[i, : len(names)] = values
        legal[i, : len(names)] = True
        targets.append(names.index(row["target_uci"]))
        wins = row.get("mating_uci", [row["target_uci"]])
        for name in wins:
            mating[i, names.index(name)] = True
    arrays = {
        "observations": np.stack([encode_board(board) for board in boards]),
        "candidates": features,
        "mask": legal,
        "targets": np.array(targets),
        "mating": mating,
        "values": np.array([row.get("target_value", 1.0) for row in rows], dtype=np.float32),
    }
    return {key: torch.from_numpy(value) for key, value in arrays.items()}, [names for names, _ in menus]


def policy_losses(logits, targets, mating, arm, mate_count):
    """Return one loss per row. Replay always uses its original single target."""
    if arm not in ("single", "set") or type(mate_count) is not int or not 0 <= mate_count <= len(logits):
        raise ValueError("Unknown arm or invalid mate row count")
    if mating.dtype != torch.bool or mating.shape != logits.shape:
        raise ValueError("Mating mask must be boolean and match logits")
    if not mating[:mate_count].any(-1).all():
        raise ValueError("Every mate row requires a nonempty target set")
    if (mating & ~torch.isfinite(logits)).any():
        raise ValueError("Target sets must contain only legal moves")
    if (
        targets.dtype != torch.long
        or targets.shape != (len(logits),)
        or (targets < 0).any()
        or (targets >= logits.shape[1]).any()
    ):
        raise ValueError("Recorded targets must be valid integer candidate indices")
    if not mating[:mate_count].gather(1, targets[:mate_count, None]).all():
        raise ValueError("Recorded mate target must belong to its mating set")
    losses = F.cross_entropy(logits, targets, reduction="none")
    if arm == "set" and mate_count:
        logp = logits[:mate_count].log_softmax(-1)
        set_losses = -logp.masked_fill(~mating[:mate_count], -torch.inf).logsumexp(-1)
        losses = torch.cat((set_losses, losses[mate_count:]))
    return losses


def batch_schedule(seed, replay_size):
    if replay_size < PROTOCOL["replay_count_per_fit"]:
        raise ValueError("Not enough replay rows for sampling without replacement")
    rng = random.Random(PROTOCOL["batch_seed_base"] + seed)
    replay = rng.sample(range(replay_size), PROTOCOL["replay_count_per_fit"])
    batches = []
    for epoch in range(PROTOCOL["epochs"]):
        mates = list(range(PROTOCOL["mate_sizes"][0]))
        rng.shuffle(mates)
        for start in range(0, len(mates), PROTOCOL["mate_per_batch"]):
            offset = len(batches) * PROTOCOL["mate_per_batch"]
            batches.append(
                {
                    "epoch": epoch + 1,
                    "mate": mates[start : start + PROTOCOL["mate_per_batch"]],
                    "replay": replay[offset : offset + PROTOCOL["mate_per_batch"]],
                }
            )
    return batches


def combined_batch(mates, replay, batch, device):
    # Candidate padding can differ between the two source pools.
    width = max(mates["mask"].shape[1], replay["mask"].shape[1])
    result = {}
    for key in mates:
        pair = []
        for source, indices in ((mates, batch["mate"]), (replay, batch["replay"])):
            value = source[key][indices]
            if key in ("candidates", "mask", "mating"):
                padding = width - value.shape[1]
                value = (
                    F.pad(value, (0, 0, 0, padding)) if key == "candidates" else F.pad(value, (0, padding))
                )
            pair.append(value)
        result[key] = torch.cat(pair).to(device)
    return result


def _sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def evaluate(model, rows, prepared, menus, out):
    data = prepared
    parameter = next(model.parameters())
    model.eval()
    result = []
    _sync(parameter.device)
    started = time.perf_counter()
    for start in range(0, len(rows), 128):
        end = min(len(rows), start + 128)
        batch = {key: value[start:end].to(parameter.device) for key, value in data.items()}
        logits, value, _ = model(batch["observations"], batch["candidates"], batch["mask"], depth=4)
        probs = logits.softmax(-1).cpu()
        values = value.cpu()
        for offset, (row, names) in enumerate(zip(rows[start:end], menus[start:end], strict=True)):
            target = names.index(row["target_uci"])
            winning = row.get("mating_uci", [row["target_uci"]])
            choice = names[int(probs[offset].argmax())]
            result.append(
                {
                    "id": row["id"],
                    "choice": choice,
                    "correct": choice in winning,
                    "recorded_target_match": choice == row["target_uci"],
                    "target_probability": float(probs[offset, target]),
                    "set_probability": float(probs[offset, [names.index(m) for m in winning]].sum()),
                    "max_probability": float(probs[offset].max()),
                    "value": float(values[offset]),
                    "target_value": row.get("target_value", 1.0),
                    "mating_moves": len(winning) if "mating_uci" in row else None,
                }
            )
    _sync(parameter.device)
    elapsed = time.perf_counter() - started
    with Path(out).open("x") as stream:
        for row in result:
            append(stream, row)
    return {
        "examples": len(rows),
        "correct": sum(r["correct"] for r in result),
        "accuracy": sum(r["correct"] for r in result) / len(result),
        "recorded_accuracy": sum(r["recorded_target_match"] for r in result) / len(result),
        "value_mae": math.fsum(abs(r["value"] - r["target_value"]) for r in result) / len(result),
        "batch_inference_seconds": elapsed,
        "timing_scope": "Batched inference with transfers and parsing; excludes pre-encoding; not single-move latency",
    }


def run(plan_path, out):
    _plan, selected = verify_plan(plan_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "started.json", {"plan_sha256": sha(plan_path), "started_unix": time.time()})
    started = time.perf_counter()
    try:
        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.use_deterministic_algorithms(False)
        device = torch.device(PROTOCOL["device"])
        rows = {f"mate_{s}": selected["splits"][s] for s in ("dev", "confirm")}
        rows.update({s: read_rows(ROOT / f"{DATA}/{s}.jsonl") for s in ("dev", "shift")})
        prepared = {key: tensors(value) for key, value in rows.items()}
        mate_tensors, _ = tensors(selected["splits"]["train"])
        replay = read_rows(ROOT / f"{DATA}/train.jsonl")
        replay_tensors, _ = tensors(replay)
        configs = [(arm, seed) for arm in PROTOCOL["arms"] for seed in PROTOCOL["seeds"]]
        random.Random(PROTOCOL["fit_order_seed"]).shuffle(configs)
        write_new(
            out / "schedule.json",
            {
                "fit_order": configs,
                "batches": {str(seed): batch_schedule(seed, len(replay)) for seed in PROTOCOL["seeds"]},
            },
        )
        fits = []
        # All originals evaluated before any new fits, without model selection.
        configs = [("frozen", seed) for seed in PROTOCOL["seeds"]] + configs
        for arm, seed in configs:
            name = f"{arm}-{seed}"
            fit = out / name
            fit.mkdir()
            model = SpatialChess.load(
                ROOT / f"models/chess-spatial-v1/predict-{seed}/weights.pt",
                expected_plan_sha256=sha(ROOT / SPATIAL_PLAN),
            ).to(device)
            training_seconds = 0.0
            if arm != "frozen":
                opt = PROTOCOL["optimizer"]
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=opt["lr"],
                    betas=tuple(opt["betas"]),
                    eps=opt["eps"],
                    weight_decay=0.0,
                )
                _sync(device)
                training_started = time.perf_counter()
                with (fit / "learning.jsonl").open("x") as stream:
                    for step, batch in enumerate(batch_schedule(seed, len(replay)), 1):
                        model.train()
                        b = combined_batch(mate_tensors, replay_tensors, batch, device)
                        optimizer.zero_grad(set_to_none=True)
                        logits, value, _ = model(b["observations"], b["candidates"], b["mask"], depth=4)
                        losses = policy_losses(
                            logits, b["targets"], b["mating"], arm, PROTOCOL["mate_per_batch"]
                        )
                        value_loss = F.mse_loss(value, b["values"])
                        loss = losses.mean() + PROTOCOL["value_weight"] * value_loss
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True
                        )
                        optimizer.step()
                        append(
                            stream,
                            {
                                "step": step,
                                "epoch": batch["epoch"],
                                "loss": float(loss.detach().cpu()),
                                "mate_policy_loss": float(
                                    losses[: PROTOCOL["mate_per_batch"]].mean().detach().cpu()
                                ),
                                "replay_policy_loss": float(
                                    losses[PROTOCOL["mate_per_batch"] :].mean().detach().cpu()
                                ),
                                "value_loss": float(value_loss.detach().cpu()),
                            },
                        )
                _sync(device)
                training_seconds = time.perf_counter() - training_started
                model.cpu().save(fit / "weights.pt", plan_sha256=sha(plan_path))
                model.to(device)
            metrics = {}
            for split, values in rows.items():
                tensors_, menus = prepared[split]
                metrics[split] = evaluate(model, values, tensors_, menus, fit / f"{split}.jsonl")
            write_new(
                fit / "completed.json",
                {
                    "status": "completed",
                    "arm": arm,
                    "seed": seed,
                    "plan_sha256": sha(plan_path),
                    "updates": 0 if arm == "frozen" else PROTOCOL["updates_per_fit"],
                    "training_seconds": training_seconds,
                    "metrics": metrics,
                    "files": {p.name: sha(p) for p in sorted(fit.iterdir()) if p.is_file()},
                },
            )
            fits.append(name)
            print(
                json.dumps(
                    {
                        "completed": name,
                        "training_seconds": training_seconds,
                        "accuracy": {s: m["accuracy"] for s, m in metrics.items()},
                    }
                ),
                flush=True,
            )
            del model
        write_new(
            out / "completed.json",
            {
                "status": "completed",
                "plan_sha256": sha(plan_path),
                "wall_seconds": time.perf_counter() - started,
                "fits": {name: sha(out / name / "completed.json") for name in fits},
                "schedule_sha256": sha(out / "schedule.json"),
            },
        )
    except Exception as exc:
        write_new(
            out / "failed.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise
    return out / "completed.json"


def report(plan_path, execution, out):
    _plan, selected = verify_plan(plan_path)
    execution = Path(execution)
    complete = json.loads((execution / "completed.json").read_text())
    identities = {f"{arm}-{seed}" for arm in ["frozen"] + PROTOCOL["arms"] for seed in PROTOCOL["seeds"]}
    if (
        (execution / "failed.json").exists()
        or complete["status"] != "completed"
        or complete["plan_sha256"] != sha(plan_path)
        or set(complete["fits"]) != identities
        or sha(execution / "schedule.json") != complete["schedule_sha256"]
    ):
        raise ValueError("Incomplete, changed or unbound execution")
    fit_order = [[arm, seed] for arm in PROTOCOL["arms"] for seed in PROTOCOL["seeds"]]
    random.Random(PROTOCOL["fit_order_seed"]).shuffle(fit_order)
    replay_size = len(read_rows(ROOT / f"{DATA}/train.jsonl"))
    expected_schedule = {
        "fit_order": fit_order,
        "batches": {str(seed): batch_schedule(seed, replay_size) for seed in PROTOCOL["seeds"]},
    }
    if json.loads((execution / "schedule.json").read_text()) != expected_schedule:
        raise ValueError("Saved training schedule differs from the frozen protocol")
    rows = {f"mate_{s}": selected["splits"][s] for s in ("dev", "confirm")}
    rows.update({s: read_rows(ROOT / f"{DATA}/{s}.jsonl") for s in ("dev", "shift")})
    result = {}
    for name in sorted(identities):
        fit = execution / name
        if sha(fit / "completed.json") != complete["fits"][name]:
            raise ValueError("Fit receipt mismatch")
        receipt = json.loads((fit / "completed.json").read_text())
        arm, seed = name.split("-")
        expected = {f"{s}.jsonl" for s in rows} | (
            {"weights.pt", "learning.jsonl"} if arm != "frozen" else set()
        )
        if (
            receipt["status"] != "completed"
            or receipt["arm"] != arm
            or receipt["seed"] != int(seed)
            or receipt["plan_sha256"] != sha(plan_path)
        ):
            raise ValueError("Fit identity or plan mismatch")
        if set(receipt["files"]) != expected or receipt["updates"] != (
            0 if arm == "frozen" else PROTOCOL["updates_per_fit"]
        ):
            raise ValueError("Fit file membership or update count mismatch")
        for filename, digest in receipt["files"].items():
            if sha(fit / filename) != digest:
                raise ValueError("Fit evidence hash mismatch")
        if arm != "frozen":
            history = read_rows(fit / "learning.jsonl")
            if [r["step"] for r in history] != list(range(1, PROTOCOL["updates_per_fit"] + 1)):
                raise ValueError("Training update journal incomplete")
            if any(
                not math.isfinite(r[k])
                for r in history
                for k in ("loss", "mate_policy_loss", "replay_policy_loss", "value_loss")
            ):
                raise ValueError("Nonfinite training journal")
            schedule = expected_schedule["batches"][seed]
            for row, batch in zip(history, schedule, strict=True):
                expected_loss = (
                    0.5 * (row["mate_policy_loss"] + row["replay_policy_loss"])
                    + PROTOCOL["value_weight"] * row["value_loss"]
                )
                if row["epoch"] != batch["epoch"] or not math.isclose(
                    row["loss"], expected_loss, rel_tol=2e-6, abs_tol=1e-6
                ):
                    raise ValueError("Training epoch or weighted loss arithmetic mismatch")
            loaded = SpatialChess.load(fit / "weights.pt", expected_plan_sha256=sha(plan_path))
            if (loaded.mode, loaded.seed, loaded.width, loaded.depth) != ("predict", int(seed), 32, 4):
                raise ValueError("Checkpoint identity changed")
        metrics = {}
        for split, source in rows.items():
            predictions = read_rows(fit / f"{split}.jsonl")
            if len(predictions) != len(source):
                raise ValueError("Missing predictions")
            correct = recorded = 0
            mae = []
            multiple = []
            for row, pred in zip(source, predictions, strict=True):
                legal = {move.uci() for move in chess.Board(row["fen"]).legal_moves}
                wins = row.get("mating_uci", [row["target_uci"]])
                if pred["id"] != row["id"] or pred["choice"] not in legal:
                    raise ValueError("Position identity or legal move mismatch")
                truth = pred["choice"] in wins
                recorded_truth = pred["choice"] == row["target_uci"]
                if (
                    pred["correct"] != truth
                    or pred["recorded_target_match"] != recorded_truth
                    or pred["target_value"] != row.get("target_value", 1.0)
                    or pred["mating_moves"] != (len(wins) if "mating_uci" in row else None)
                ):
                    raise ValueError("Prediction correctness or metadata mismatch")
                if any(
                    not math.isfinite(pred[k]) or not -1e-6 <= pred[k] <= 1 + 1e-6
                    for k in ("target_probability", "set_probability", "max_probability")
                ):
                    raise ValueError("Invalid probability")
                if not math.isfinite(pred["value"]) or not -1 <= pred["value"] <= 1:
                    raise ValueError("Invalid value")
                correct += truth
                recorded += recorded_truth
                mae.append(abs(pred["value"] - pred["target_value"]))
                if len(wins) > 1:
                    multiple.append(truth)
            calculated = {
                "examples": len(source),
                "correct": correct,
                "accuracy": correct / len(source),
                "recorded_accuracy": recorded / len(source),
                "value_mae": math.fsum(mae) / len(mae),
            }
            for key, value in calculated.items():
                if not math.isclose(receipt["metrics"][split][key], value, abs_tol=1e-12):
                    raise ValueError("Saved aggregate mismatch")
            metrics[split] = {
                **calculated,
                "multiple_mate_positions": len(multiple),
                "multiple_mate_accuracy": sum(multiple) / len(multiple) if multiple else None,
            }
        result[name] = {
            "metrics": metrics,
            "training_seconds": receipt["training_seconds"],
            "updates": receipt["updates"],
        }
    means = {
        arm: {
            split: math.fsum(result[f"{arm}-{s}"]["metrics"][split]["accuracy"] for s in PROTOCOL["seeds"])
            / len(PROTOCOL["seeds"])
            for split in rows
        }
        for arm in ["frozen"] + PROTOCOL["arms"]
    }
    gate = PROTOCOL["gate"]
    improvement = {split: means["set"][split] - means["frozen"][split] for split in rows}
    objective = {split: means["set"][split] - means["single"][split] for split in rows}
    summary = {
        "scope": PROTOCOL["scope"],
        "plan_sha256": sha(plan_path),
        "execution_receipt_sha256": sha(execution / "completed.json"),
        "results": result,
        "mean_accuracy": means,
        "set_minus_frozen": improvement,
        "set_minus_single": objective,
        "gates": {
            "targeted_skill": all(
                improvement[s] >= gate["mate_gain_over_frozen"] for s in ("mate_dev", "mate_confirm")
            ),
            "ordinary_retention": all(
                improvement[s] >= -gate["max_ordinary_mean_drop"] for s in ("dev", "shift")
            ),
            "set_objective": all(
                objective[s] >= gate["set_gain_over_single"] for s in ("mate_dev", "mate_confirm")
            ),
        },
        "updates": sum(r["updates"] for r in result.values()),
        "training_seconds": math.fsum(r["training_seconds"] for r in result.values()),
        "new_training_engine_calls": 0,
    }
    write_new(out, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--plan", required=True)
    p.add_argument("--execution", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.out))
    elif args.command == "run":
        print(run(args.plan, args.out))
    else:
        print(json.dumps(report(args.plan, args.execution, args.out), indent=2))


if __name__ == "__main__":
    main()
