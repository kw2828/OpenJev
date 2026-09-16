"""Reproducible behavioral cloning of a transparent teacher, not RLCD."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .domain import FEATURES, STEERING, Directive, Observation, teacher_action
from .policies import DEFAULT_WEIGHTS, LocalPolicy, softmax


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n):
        # Oversample aiming boundaries, plus wide-screen and target-absent cases.
        error = rng.uniform(-1, 1) if rng.random() < 0.4 else rng.uniform(-0.16, 0.16)
        obs = Observation(
            visible=bool(rng.random() > 0.12),
            aim_error=error,
            half_width=float(rng.uniform(0.005, 0.25)),
            distance=float(rng.uniform(30, 1100)),
            health=float(rng.uniform(1, 100)),
            ammo=float(rng.integers(0, 51)),
            directive=str(rng.choice(list(Directive))),
            scenario="basic" if rng.random() < 0.25 else "defend_the_center",
        )
        if not obs.visible:
            from dataclasses import replace

            obs = replace(obs, aim_error=0, half_width=0, distance=0)
        steer, fire = teacher_action(obs)
        x.append(obs.features())
        y.append([STEERING.index(steer), int(fire)])
    return np.array(x), np.array(y, dtype=np.int64)


def train(output=DEFAULT_WEIGHTS, *, samples=60000, epochs=40, seed=7):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x, y = dataset(samples, seed)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    model = nn.Sequential(
        nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(samples)
        total = 0
        for ids in order.split(512):
            logits = model(tx[ids])
            loss = nn.functional.cross_entropy(logits[:, :3], ty[ids, 0])
            loss = loss + nn.functional.cross_entropy(logits[:, 3:], ty[ids, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}/{epochs}: loss={total / samples:.5f}", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = {}
    for i, layer in enumerate([model[0], model[2], model[4]], 1):
        params[f"w{i}"] = layer.weight.detach().numpy().T
        params[f"b{i}"] = layer.bias.detach().numpy()
    np.savez_compressed(output, **params)
    test_x, test_y = dataset(10000, seed + 1)
    logits = LocalPolicy(output).logits(test_x)
    p = softmax(logits[:, :3])
    f = softmax(logits[:, 3:])
    report = {
        "method": "supervised imitation of teacher_action on synthetic structured states",
        "not_rlcd": True,
        "seed": seed,
        "train_samples": samples,
        "epochs": epochs,
        "test_seed": seed + 1,
        "synthetic_test_samples": len(test_y),
        "features": list(FEATURES),
        "architecture": [len(FEATURES), 64, 64, 5],
        "parameters": sum(p.size for p in params.values()),
        "steer_teacher_agreement": float((p.argmax(1) == test_y[:, 0]).mean()),
        "fire_teacher_agreement": float((f.argmax(1) == test_y[:, 1]).mean()),
        "fire_brier_vs_teacher": float(np.mean((f[:, 1] - test_y[:, 1]) ** 2)),
        "training_seconds": time.perf_counter() - started,
        "weights_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": "Synthetic teacher agreement is not gameplay ability or real-world calibration.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report
