"""Bounded offline dynamics pilot. No simulator, policy rollout, or test tuning.

The input is created by prepare_action_filter_data.py. Model checkpoints, raw
predictions, every training loss and timing samples are retained in a new run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research.action_filter_models import ActionFilterModel

MODES = ("transported_delta", "decay_delta", "gru", "history16", "diagonal_filter")
SEEDS = (101, 202, 303)
CONTEXT, HORIZON, WINDOWS = 32, 10, 16
EPOCHS, BATCH = 20, 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_hashes(path):
    return {name: sha(path / name) for name in
            ("train.npz", "dev.npz", "test.npz", "normalization.npz", "manifest.json", "completed.json")}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def windows(obs, actions):
    """Disjoint windows within each raw trajectory; never split trajectories."""
    if obs.ndim != 3 or actions.ndim != 3 or obs.shape[:2] != (len(actions), actions.shape[1] + 1):
        raise ValueError("observations must include exactly one extra frame")
    span = CONTEXT + HORIZON
    starts = np.linspace(0, obs.shape[1] - span, WINDOWS, dtype=int)
    if np.any(np.diff(starts) < span):
        raise ValueError("trajectory too short for nonoverlapping windows")
    observations, controls, identities = [], [], []
    for episode in range(len(obs)):
        for start in starts:
            observations.append(obs[episode, start:start + span])
            controls.append(actions[episode, start:start + span - 1])
            identities.append((episode, int(start)))
    return (torch.from_numpy(np.stack(observations)),
            torch.from_numpy(np.stack(controls)), np.asarray(identities))


def forecast(model, obs, actions, context=CONTEXT):
    """Read only context observations; actions advance a private state thereafter."""
    if context not in (16, CONTEXT):
        raise ValueError("unsupported context")
    state = model.initial(len(obs))
    start = CONTEXT - context
    for t in range(start, CONTEXT):
        if t > start:
            state, _ = model.advance(state, actions[:, t - 1])
        state = model.assimilate(state, obs[:, t])
    predicted = []
    for t in range(CONTEXT - 1, CONTEXT + HORIZON - 1):
        state, prediction = model.advance(state, actions[:, t])
        predicted.append(prediction)
    return torch.stack(predicted, dim=1)


def predict_batches(model, obs, actions, context):
    with torch.no_grad():
        return torch.cat([forecast(model, obs[s:s + BATCH], actions[s:s + BATCH], context)
                          for s in range(0, len(obs), BATCH)])


def metrics(predictions, target):
    error = (predictions.double() - target.double()).square()
    return {"mse": float(error.mean()),
            "horizon_mse": error.mean(dim=(0, 2)).tolist()}


def ridge_features(obs, actions, horizon):
    """Use no observation after root or action after forecast horizon."""
    return torch.cat((torch.ones(len(obs), 1, dtype=obs.dtype),
                      obs[:, CONTEXT - 16:CONTEXT].flatten(1),
                      actions[:, CONTEXT - 16:CONTEXT - 1 + horizon].flatten(1)), dim=1).double()


def ridge_predict(train_obs, train_act, eval_obs, eval_act):
    predictions, weights = [], []
    for h in range(1, HORIZON + 1):
        x = ridge_features(train_obs, train_act, h)
        y = train_obs[:, CONTEXT - 1 + h].double()
        penalty = torch.eye(x.shape[1], dtype=torch.float64)
        penalty[0, 0] = 0
        weight = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
        predictions.append(ridge_features(eval_obs, eval_act, h) @ weight)
        weights.append(weight)
    return torch.stack(predictions, dim=1), weights


def source_files():
    return [Path("scripts/train_action_filter.py"), Path("scripts/prepare_action_filter_data.py"),
            Path("src/openjev/research/action_filter_models.py"),
            Path("tests/test_action_filter_models.py"), Path("tests/test_action_filter_data.py"),
            Path("tests/test_action_filter_training.py"), Path("research/action-filter-protocol.md")]


def freeze(data, output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {str(p): sha(p) for p in source_files()}
    protocol = {"schema": "action-filter-v1", "data_hashes": data_hashes(data), "data_path": str(data.resolve()),
                "sources": bindings, "modes": MODES, "seeds": SEEDS,
                "context": CONTEXT, "horizon": HORIZON, "windows_per_trajectory": WINDOWS,
                "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": .003,
                "optimizer": "Adam", "clip_gradient_norm": 1., "checkpoint": "last epoch, no selection",
                "primary": "mean normalized squared error over 10 forecast steps and all 9 channels",
                "continuation": "transported_delta improves >=5% over every other neural family, ridge16 and hold_last; "
                                "all 3 seeds improve over paired neural fits and both references; "
                                "median full-window predictor latency <=1.5x GRU",
                "claim": "exploratory offline prediction only; not policy, real robot, terrain shift, or novel principle",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "platform": platform.platform(), "threads": 1}}
    write_json(output / "protocol.json", protocol)
    for path in source_files():
        target = output / "source-snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    print(json.dumps({"frozen": str(output), "protocol_sha256": sha(output / "protocol.json")}))


def run(data, output):
    protocol = json.loads((output / "protocol.json").read_text())
    if data_hashes(data) != protocol["data_hashes"]:
        raise ValueError("data changed after freeze")
    for path, digest in protocol["sources"].items():
        if sha(path) != digest:
            raise ValueError(f"source changed after freeze: {path}")
    run_dir = output / "run-01"
    run_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(run_dir / "started.json", {"protocol_sha256": sha(output / "protocol.json")})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        # Test arrays are not accessed until every neural fit is terminal.
        with np.load(data / "train.npz", allow_pickle=False) as archive:
            train_obs, train_act, train_ids = windows(archive["obs"], archive["actions"])
        with np.load(data / "dev.npz", allow_pickle=False) as archive:
            dev_obs, dev_act, dev_ids = windows(archive["obs"], archive["actions"])
        dimensions = (train_obs.shape[-1], train_act.shape[-1])
        train_target = train_obs[:, CONTEXT:]
        fits = []
        for seed in SEEDS:
            # Identical batches and targets for every family in this paired seed.
            generator = torch.Generator().manual_seed(seed + 5000)
            orders = [torch.randperm(len(train_obs), generator=generator) for _ in range(EPOCHS)]
            for mode in MODES:
                fit_start = time.perf_counter()
                torch.manual_seed(seed)
                model = ActionFilterModel("gru" if mode == "history16" else mode, *dimensions)
                # Preserve identical common-module initialization for delta ablation.
                if mode == "decay_delta":
                    transport = torch.load(run_dir / f"transported_delta-{seed}-initial.pt", weights_only=True)
                    state = model.state_dict()
                    model.load_state_dict({name: transport[name] for name in state})
                initial = run_dir / f"{mode}-{seed}-initial.pt"
                torch.save(model.state_dict(), initial)
                context = 16 if mode == "history16" else CONTEXT
                optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
                losses = []
                model.train()
                for epoch, order in enumerate(orders):
                    total = 0.
                    for s in range(0, len(order), BATCH):
                        indices = order[s:s + BATCH]
                        prediction = forecast(model, train_obs[indices], train_act[indices], context)
                        loss = (prediction - train_target[indices]).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"nonfinite loss {mode} {seed} {epoch}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        total += losses[-1] * len(indices)
                    print(json.dumps({"fit": f"{mode}-{seed}", "epoch": epoch + 1,
                                      "train_mse": total / len(order)}), flush=True)
                train_seconds = time.perf_counter() - fit_start
                model.eval()
                dev_predictions = predict_batches(model, dev_obs, dev_act, context)
                checkpoint = run_dir / f"{mode}-{seed}.pt"
                torch.save(model.state_dict(), checkpoint)
                np.save(run_dir / f"{mode}-{seed}-losses.npy", np.asarray(losses), allow_pickle=False)
                fit = {"mode": mode, "seed": seed, "checkpoint_sha256": sha(checkpoint),
                       "initial_sha256": sha(initial), "updates": len(losses),
                       "train_seconds": train_seconds, "development": metrics(dev_predictions, dev_obs[:, CONTEXT:]),
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "state_floats": sum(v.numel() for v in model.initial(1).values()),
                       "context_frames": context}
                fits.append(fit)
                write_json(run_dir / f"{mode}-{seed}-fit.json", fit)
        write_json(run_dir / "training-completed.json", {"fits": fits, "test_accessed": False})
        with np.load(data / "test.npz", allow_pickle=False) as archive:
            test_obs, test_act, test_ids = windows(archive["obs"], archive["actions"])
            test_raw_ids = archive["source_ids"]
        target = test_obs[:, CONTEXT:]
        np.savez_compressed(run_dir / "evaluation-inputs.npz", target=target.numpy(),
                            window_ids=test_ids, raw_trajectory_ids=test_raw_ids,
                            train_window_ids=train_ids, dev_window_ids=dev_ids)
        for fit in fits:
            mode, seed = fit["mode"], fit["seed"]
            model = ActionFilterModel("gru" if mode == "history16" else mode, *dimensions)
            checkpoint = run_dir / f"{mode}-{seed}.pt"
            if sha(checkpoint) != fit["checkpoint_sha256"]:
                raise ValueError("checkpoint changed")
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            model.eval()
            predictions = predict_batches(model, test_obs, test_act, fit["context_frames"])
            np.save(run_dir / f"{mode}-{seed}-predictions.npy", predictions.numpy(), allow_pickle=False)
            fit["test"] = metrics(predictions, target)
            timing = []
            with torch.no_grad():
                for i in range(103):
                    case = i % len(test_obs)
                    tick = time.perf_counter_ns()
                    forecast(model, test_obs[case:case + 1], test_act[case:case + 1], fit["context_frames"])
                    elapsed = (time.perf_counter_ns() - tick) / 1e6
                    if i >= 3:
                        timing.append(elapsed)
            fit["window_latency_ms"] = {"median": float(np.median(timing)),
                                        "p95": float(np.percentile(timing, 95)), "samples": timing}
            write_json(run_dir / f"{mode}-{seed}-evaluation.json", fit)
        reference_start = time.perf_counter()
        ridge, weights = ridge_predict(train_obs, train_act, test_obs, test_act)
        ridge_seconds = time.perf_counter() - reference_start
        torch.save(weights, run_dir / "ridge16.pt")
        hold = test_obs[:, CONTEXT - 1:CONTEXT].expand(-1, HORIZON, -1)
        references = {}
        for name, prediction in (("ridge16", ridge), ("hold_last", hold)):
            np.save(run_dir / f"{name}-predictions.npy", prediction.numpy(), allow_pickle=False)
            references[name] = metrics(prediction, target)
        references["ridge16"]["fit_and_batch_evaluate_seconds"] = ridge_seconds
        write_json(run_dir / "completed.json", {"fits": fits, "references": references,
                   "counts": {"train_windows": len(train_obs), "dev_windows": len(dev_obs),
                              "test_windows": len(test_obs), "neural_fits": len(fits)},
                   "wall_seconds": time.perf_counter() - started,
                   "protocol_sha256": sha(output / "protocol.json")})
        print(json.dumps({"completed": str(run_dir), "wall_seconds": time.perf_counter() - started}), flush=True)
    except BaseException as exc:
        write_json(run_dir / "failed.json", {"type": type(exc).__name__, "error": str(exc),
                                              "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command](args.data, args.output)
