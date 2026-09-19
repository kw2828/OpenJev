"""One fixed development comparison of pose geometry and transported memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research.pose_references import fit_ridge16, predict_reference, predict_ridge16
from openjev.research.pose_transport import PoseTransport, training_scales
from openjev.research.rigid_motion import geodesic_angle, legacy_obs_to_pose

CONTEXT, HORIZON, EPOCHS, BATCH = 32, 25, 30, 32
SEEDS = (701, 802, 903)
VARIANTS = ("world", "body", "transport", "history16")
REFERENCES = ("hold", "cv1", "cv16", "ls16", "body16", "ridge16")
PANELS = ("test_sin", "test_zigzag")
POSITION_SCALE, ANGLE_SCALE = .1, .1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    with Path(p).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def load_data(data, split):
    with np.load(data / "normalization.npz", allow_pickle=False) as normal:
        mean, scale = normal["obs_mean"], normal["obs_scale"]
    with np.load(data / (split + ".npz"), allow_pickle=False) as arrays:
        raw = arrays["obs"].astype(np.float64) * scale + mean
        p, rotation = legacy_obs_to_pose(torch.from_numpy(raw))
        actions = torch.from_numpy(arrays["actions"].copy())
        ids = np.stack((arrays["source_ids"], arrays["window_starts"]), axis=-1)
    return p.float(), rotation.float(), actions, ids


def loss_function(pred_p, pred_r, target_p, target_r):
    return ((pred_p - target_p).square().sum(-1) / POSITION_SCALE ** 2
            + geodesic_angle(pred_r, target_r).square() / ANGLE_SCALE ** 2).mean()


def measure(pred_p, pred_r, target_p, target_r):
    if not torch.isfinite(pred_p).all() or not torch.isfinite(pred_r).all():
        raise FloatingPointError("nonfinite forecast")
    pos_sq = (pred_p.double() - target_p.double()).square().sum(-1)
    ang_sq = geodesic_angle(pred_r.double(), target_r.double()).square()
    return {"position_rmse_m": float(pos_sq.mean().sqrt()),
            "rotation_rmse_rad": float(ang_sq.mean().sqrt()),
            "composite": float((pos_sq / POSITION_SCALE ** 2 + ang_sq / ANGLE_SCALE ** 2).mean()),
            "rotation_orthogonality_max": float((pred_r.transpose(-1, -2) @ pred_r
                                                 - torch.eye(3)).abs().max()),
            "rotation_determinant_max_error": float((torch.linalg.det(pred_r) - 1).abs().max())}


def predict(model, p, rotation, actions):
    with torch.no_grad():
        outputs = [model.forecast(p[s:s + BATCH], rotation[s:s + BATCH], actions[s:s + BATCH])
                   for s in range(0, len(p), BATCH)]
    return tuple(torch.cat([x[i] for x in outputs]) for i in range(2))


def sources():
    return [Path(p) for p in (
        "scripts/train_pose_transport.py", "scripts/audit_pose_transport.py",
        "src/openjev/research/pose_transport.py", "src/openjev/research/pose_references.py",
        "src/openjev/research/rigid_motion.py", "tests/test_pose_transport.py",
        "tests/test_pose_references.py", "tests/test_rigid_motion.py", "tests/test_audit_pose_transport.py",
        "tests/test_pose_transport_training.py", "research/pose-transport-protocol.md")]


def freeze(data, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"study": "pose-transport-v1", "scope": "exposed-data development, not confirmation",
                "data": str(data.resolve()), "data_hashes": {p.name: sha(p) for p in sorted(data.glob('*')) if p.is_file()},
                "sources": {str(p): sha(p) for p in sources()}, "seeds": SEEDS,
                "variants": VARIANTS, "references": REFERENCES, "panels": PANELS,
                "epochs": EPOCHS, "batch": BATCH, "context": CONTEXT, "horizon": HORIZON,
                "learning_rate": .001, "grad_norm_cap": 1., "optimizer": "Adam",
                "expected_fits": 12, "updates_per_fit": 690,
                "position_loss_scale_m": POSITION_SCALE, "rotation_loss_scale_rad": ANGLE_SCALE,
                "checkpoint": "final only; no selection, retries or sweep",
                "primary": "position and geodesic rotation RMSE separately, all25 steps; family pools3 fits",
                "gate": "transport at least10% better on each physical endpoint against every learned and classical control on both panels; all3 paired fit endpoints nonworse; at least8/10 parents nonworse for each endpoint/control/panel; strictly lower MSE after each leave-one-parent-out; median full-window latency<=1.5x body. Failure stays failed.",
                "prior_failure": "residual-dynamics-v1 remains failed36/49; this is new training and a new objective",
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "threads": 1, "platform": platform.platform()}}
    write_json(out / "protocol.json", protocol)
    for p in sources():
        dst = out / "source-snapshot" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    print(json.dumps({"frozen": str(out), "sha256": sha(out / "protocol.json")}))


def run(data, out):
    protocol = json.loads((out / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        if sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    for name, digest in protocol["data_hashes"].items():
        if sha(data / name) != digest:
            raise ValueError("input data changed: " + name)
    dest = out / "run-01"
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    write_json(dest / "started.json", {"protocol_sha256": sha(out / "protocol.json")})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        p, r, a, _ = load_data(data, "train")
        scale = training_scales(p, r)
        fits = []
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(seed + 1000)
            orders = [torch.randperm(len(p), generator=gen) for _ in range(EPOCHS)]
            for variant in VARIANTS:
                tick = time.perf_counter()
                torch.manual_seed(seed)
                model = PoseTransport(variant, scale)
                stem = f"{variant}-{seed}"
                torch.save(model.state_dict(), dest / (stem + "-initial.pt"))
                opt = torch.optim.Adam(model.parameters(), lr=.001)
                losses = []
                for epoch, order in enumerate(orders):
                    total = 0.
                    for s in range(0, len(p), BATCH):
                        idx = order[s:s + BATCH]
                        pp, rr = model.forecast(p[idx], r[idx], a[idx])
                        loss = loss_function(pp, rr, p[idx, CONTEXT:], r[idx, CONTEXT:])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite training objective")
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        opt.step()
                        losses.append(float(loss.detach()))
                        total += losses[-1] * len(idx)
                    print(json.dumps({"fit": stem, "epoch": epoch + 1, "train_loss": total / len(p)}), flush=True)
                torch.save(model.state_dict(), dest / (stem + ".pt"))
                np.save(dest / (stem + "-losses.npy"), np.asarray(losses), allow_pickle=False)
                fit = {"variant": variant, "seed": seed, "updates": len(losses),
                       "scales": scale.tolist(), "parameters": sum(x.numel() for x in model.parameters()),
                       "training_seconds": time.perf_counter() - tick,
                       "initial_sha256": sha(dest / (stem + "-initial.pt")),
                       "checkpoint_sha256": sha(dest / (stem + ".pt"))}
                write_json(dest / (stem + "-fit.json"), fit)
                fits.append(fit)
        tick = time.perf_counter()
        ridge = fit_ridge16(p, r, a)
        ridge_seconds = time.perf_counter() - tick
        torch.save(ridge, dest / "ridge16.pt")
        write_json(dest / "training-completed.json", {"fits": fits, "ridge_seconds": ridge_seconds,
                                                       "development_panel_forecasts_started": False})
        rows = []
        for panel in PANELS:
            tp, tr, ta, ids = load_data(data, panel)
            np.savez_compressed(dest / (panel + "-targets.npz"), p=tp[:, CONTEXT:].numpy(),
                                R=tr[:, CONTEXT:].numpy(), ids=ids)
            for variant in (*VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in VARIANTS else (None,):
                    stem = f"{variant}-{seed}" if seed is not None else variant
                    if variant in VARIANTS:
                        fit = next(x for x in fits if x["variant"] == variant and x["seed"] == seed)
                        if sha(dest / (stem + ".pt")) != fit["checkpoint_sha256"]:
                            raise ValueError("checkpoint changed: " + stem)
                        model = PoseTransport(variant, scale)
                        model.load_state_dict(torch.load(dest / (stem + ".pt"), weights_only=True))
                        model.eval()
                        fn = model.forecast
                    elif variant == "ridge16":
                        fn = lambda xp, xr, xa: predict_ridge16(ridge, xp, xr, xa)
                    else:
                        fn = lambda xp, xr, xa, name=variant: predict_reference(name, xp, xr, xa)
                    with torch.no_grad():
                        pp, rr = fn(tp, tr, ta)
                        row = {"panel": panel, "variant": variant, "seed": seed,
                               "metrics": measure(pp, rr, tp[:, CONTEXT:], tr[:, CONTEXT:])}
                        timings = []
                        for i in range(23):
                            tick = time.perf_counter_ns()
                            fn(tp[i:i + 1], tr[i:i + 1], ta[i:i + 1])
                            ms = (time.perf_counter_ns() - tick) / 1e6
                            if i >= 3:
                                timings.append(ms)
                    row["latency_ms"] = timings
                    rows.append(row)
                    prefix = panel + "-" + stem
                    np.savez_compressed(dest / (prefix + "-predictions.npz"), p=pp.numpy(), R=rr.numpy())
                    write_json(dest / (prefix + "-evaluation.json"), row)
        write_json(dest / "completed.json", {"status": "completed", "fits": fits, "rows": rows,
                    "ridge_seconds": ridge_seconds, "wall_seconds": time.perf_counter() - start,
                    "protocol_sha256": sha(out / "protocol.json"),
                    "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                              for p in sorted(dest.iterdir()) if p.is_file()}})
        print(json.dumps({"completed": str(dest), "wall_seconds": time.perf_counter() - start}), flush=True)
    except BaseException as error:
        write_json(dest / "failed.json", {"error": repr(error), "wall_seconds": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (freeze if args.mode == "freeze" else run)(args.data, args.out)
