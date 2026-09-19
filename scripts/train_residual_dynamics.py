"""Frozen 500 ms offline forecaster and causal prefix-adaptation comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research.residual_dynamics import ResidualDynamics, RLSAdapter

CONTEXT, HORIZON, EPOCHS, BATCH = 32, 25, 30, 32
SEEDS = (411, 512, 613)
VARIANTS = ("none", "bias", "public", "latent", "history16")
PANELS = ("test_sin", "test_zigzag")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def data_hashes(data):
    return {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()}


def load_split(data, split):
    with np.load(data / f"{split}.npz", allow_pickle=False) as arrays:
        return torch.from_numpy(arrays["obs"]), torch.from_numpy(arrays["actions"])


def forecast(model, obs, actions, variant="none"):
    """Adapt only on completed context transitions; freeze fast weights at root."""
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    start = 16 if variant == "history16" else 0
    state = model.assimilate(model.initial(len(obs)), obs[:, start])
    adapter = fast = None
    for t in range(start, CONTEXT - 1):
        state, base, features = model.advance(state, actions[:, t])
        if variant in ("bias", "public", "latent"):
            phi = features[variant]
            if adapter is None:
                adapter = RLSAdapter(phi.shape[-1], obs.shape[-1], model.delta_scale)
                fast = adapter.initial(len(obs))
            target = (obs[:, t + 1] - base) / model.delta_scale
            fast = adapter.observe(fast, phi, target)
        state = model.assimilate(state, obs[:, t + 1])
    predictions = []
    for t in range(CONTEXT - 1, CONTEXT + HORIZON - 1):
        state, pred, features = model.advance(state, actions[:, t])
        if adapter is not None:
            correction = adapter.correct(fast, features[variant])
            pred = pred + correction
            # Correct the private forecast accumulator only. Fast weights never
            # receive this prediction, a pseudo-label, or a covariance update.
            state = {**state, "obs": pred, "delta": state["delta"] + correction / model.delta_scale}
        predictions.append(pred)
    return torch.stack(predictions, dim=1)


def predict_batches(model, obs, actions, variant):
    with torch.no_grad():
        return torch.cat([forecast(model, obs[s:s + BATCH], actions[s:s + BATCH], variant)
                          for s in range(0, len(obs), BATCH)])


def metrics(pred, targets):
    if not torch.isfinite(pred).all():
        raise FloatingPointError("nonfinite forecast")
    error = (pred.double() - targets.double()).square()
    return {"mse": float(error.mean()), "horizon_mse": error.mean(dim=(0, 2)).tolist()}


def ridge_features(obs, actions, horizon):
    history = obs[:, CONTEXT - 16:CONTEXT].clone()
    history[:, :, :3] -= obs[:, CONTEXT - 1:CONTEXT, :3]
    return torch.cat((history.flatten(1), actions[:, CONTEXT - 16:CONTEXT - 1 + horizon].flatten(1)), 1).double()


def fit_ridge(obs, actions):
    """Direct residual ridge, unit penalty, unpenalized intercept, dual solve."""
    weights = []
    for h in range(1, HORIZON + 1):
        x = ridge_features(obs, actions, h)
        y = (obs[:, CONTEXT - 1 + h] - obs[:, CONTEXT - 1]).double()
        mean_x, mean_y = x.mean(0), y.mean(0)
        xc, yc = x - mean_x, y - mean_y
        # The dual is exact ridge, and avoids a large action-history normal matrix.
        dual = torch.linalg.solve(xc @ xc.T + torch.eye(len(x), dtype=torch.float64), yc)
        weight = xc.T @ dual
        weights.append({"weight": weight, "intercept": mean_y - mean_x @ weight})
    return weights


def predict_ridge(weights, obs, actions):
    return torch.stack([ridge_features(obs, actions, h) @ weights[h - 1]["weight"]
                        + weights[h - 1]["intercept"] + obs[:, CONTEXT - 1].double()
                        for h in range(1, HORIZON + 1)], 1)


def sources():
    return [Path("scripts/train_residual_dynamics.py"), Path("scripts/prepare_residual_dynamics_data.py"),
            Path("src/openjev/research/residual_dynamics.py"), Path("tests/test_residual_dynamics.py"),
            Path("tests/test_residual_dynamics_data.py"), Path("tests/test_residual_dynamics_training.py"),
            Path("research/residual-dynamics-protocol.md")]


def freeze(data, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"study": "residual-dynamics-v1", "sources": {str(p): sha(p) for p in sources()},
        "data_path": str(data.resolve()), "data_hashes": data_hashes(data), "context": CONTEXT,
        "horizon": HORIZON, "epochs": EPOCHS, "batch": BATCH, "seeds": SEEDS, "variants": VARIANTS,
        "panels": PANELS, "learning_rate": .001, "optimizer": "Adam", "gradient_norm_cap": 1.,
        "checkpoint_selection": "final epoch, no dev/test selection", "rls_prior_precision": 1.,
        "rls_forgetting": 1., "neural_fits": 6, "per_fit_updates": 690,
        "primary": "mean normalized observation squared error across 25-step forecast and all channels",
        "gate": "latent RLS >=10% mean improvement against each other variant, ridge and hold on BOTH panels; "
                "all 3 fits improve against paired variants and references on BOTH panels; "
                "pooled median full-window latency <=2x same-backbone no-adaptation (49 checks)",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform(), "threads": 1}}
    write_json(out / "protocol.json", protocol)
    for p in sources():
        dst = out / "source-snapshot" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    print(json.dumps({"frozen": str(out), "protocol_sha256": sha(out / "protocol.json")}))


def run(data, out):
    protocol = json.loads((out / "protocol.json").read_text())
    if data_hashes(data) != protocol["data_hashes"]:
        raise ValueError("prepared data changed")
    for p, digest in protocol["sources"].items():
        if sha(p) != digest:
            raise ValueError(f"source changed: {p}")
    dest = out / "run-01"
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    write_json(dest / "started.json", {"protocol_sha256": sha(out / "protocol.json")})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x, a = load_split(data, "train")
        dx, da = load_split(data, "dev")
        with np.load(data / "normalization.npz", allow_pickle=False) as arrays:
            scale = torch.tensor(arrays["delta_scale_normalized"], dtype=torch.float32)
        fits = []
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(seed + 1000)
            orders = [torch.randperm(len(x), generator=gen) for _ in range(EPOCHS)]
            for variant in ("none", "history16"):
                fit_start = time.perf_counter()
                torch.manual_seed(seed)
                model = ResidualDynamics(delta_scale=scale)
                initial = dest / f"{variant}-{seed}-initial.pt"
                torch.save(model.state_dict(), initial)
                opt = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
                losses = []
                model.train()
                for epoch, order in enumerate(orders):
                    total = 0.
                    for s in range(0, len(x), BATCH):
                        indices = order[s:s + BATCH]
                        pred = forecast(model, x[indices], a[indices], variant)
                        loss = (pred - x[indices, CONTEXT:]).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite training loss")
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        opt.step()
                        losses.append(float(loss.detach()))
                        total += losses[-1] * len(indices)
                    print(json.dumps({"fit": f"{variant}-{seed}", "epoch": epoch + 1,
                                      "train_mse": total / len(x)}), flush=True)
                fit_seconds = time.perf_counter() - fit_start
                model.eval()
                dev = predict_batches(model, dx, da, variant)
                checkpoint = dest / f"{variant}-{seed}.pt"
                torch.save(model.state_dict(), checkpoint)
                np.save(dest / f"{variant}-{seed}-losses.npy", np.asarray(losses), allow_pickle=False)
                fit = {"variant": variant, "seed": seed, "initial_sha256": sha(initial),
                    "checkpoint_sha256": sha(checkpoint), "updates": len(losses), "train_seconds": fit_seconds,
                    "development": metrics(dev, dx[:, CONTEXT:]),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "state_elements": {k: v.numel() for k, v in model.initial(1).items()}}
                fits.append(fit)
                write_json(dest / f"{variant}-{seed}-fit.json", fit)
        tick = time.perf_counter()
        ridge = fit_ridge(x, a)
        ridge_fit_seconds = time.perf_counter() - tick
        torch.save(ridge, dest / "ridge16.pt")
        write_json(dest / "training-completed.json", {"fits": fits, "test_arrays_opened": False,
                                                       "ridge_fit_seconds": ridge_fit_seconds})
        rows, references = [], {}
        for panel in PANELS:
            tx, ta = load_split(data, panel)
            np.save(dest / f"{panel}-targets.npy", tx[:, CONTEXT:].numpy(), allow_pickle=False)
            for seed in SEEDS:
                for variant in VARIANTS:
                    backbone = "history16" if variant == "history16" else "none"
                    fit = next(f for f in fits if f["variant"] == backbone and f["seed"] == seed)
                    model = ResidualDynamics(delta_scale=scale)
                    checkpoint = dest / f"{backbone}-{seed}.pt"
                    if sha(checkpoint) != fit["checkpoint_sha256"]:
                        raise ValueError("checkpoint changed")
                    model.load_state_dict(torch.load(checkpoint, weights_only=True))
                    model.eval()
                    pred = predict_batches(model, tx, ta, variant)
                    prefix = f"{panel}-{variant}-{seed}"
                    np.save(dest / f"{prefix}-predictions.npy", pred.numpy(), allow_pickle=False)
                    row = {"panel": panel, "variant": variant, "seed": seed,
                           "backbone_sha256": fit["checkpoint_sha256"], "test": metrics(pred, tx[:, CONTEXT:])}
                    # Equal 50 measured windows per fit/panel, not streaming latency.
                    timings = []
                    with torch.no_grad():
                        for i in range(53):
                            case = i % len(tx)
                            tick = time.perf_counter_ns()
                            forecast(model, tx[case:case + 1], ta[case:case + 1], variant)
                            elapsed = (time.perf_counter_ns() - tick) / 1e6
                            if i >= 3:
                                timings.append(elapsed)
                    row["window_latency_ms"] = {"samples": timings, "median": float(np.median(timings)),
                                                "p95": float(np.percentile(timings, 95))}
                    rows.append(row)
                    write_json(dest / f"{prefix}-evaluation.json", row)
            references[panel] = {}
            for name, pred in (("ridge16", predict_ridge(ridge, tx, ta)),
                               ("hold_last", tx[:, CONTEXT - 1:CONTEXT].expand(-1, HORIZON, -1))):
                np.save(dest / f"{panel}-{name}-predictions.npy", pred.numpy(), allow_pickle=False)
                references[panel][name] = metrics(pred, tx[:, CONTEXT:])
        # Seal all payloads after successful run. The completion receipt is outside its own map.
        write_json(dest / "completed.json", {"status": "completed", "fits": fits, "evaluations": rows,
                   "references": references, "ridge_fit_seconds": ridge_fit_seconds,
                   "wall_seconds": time.perf_counter() - start,
                   "protocol_sha256": sha(out / "protocol.json"),
                   "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                             for p in sorted(dest.iterdir()) if p.is_file()}})
        print(json.dumps({"completed": str(dest), "wall_seconds": time.perf_counter() - start}), flush=True)
    except BaseException as error:
        write_json(dest / "failed.json", {"error": repr(error), "wall_seconds": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"freeze": freeze, "run": run}[args.command](args.data, args.out)
