"""Frozen-checkpoint recency and robust-influence screen; no weight training."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from train_pose_adaptation import load_data, make_model, measure, predict

from openjev.research.pose_references import predict_reference, predict_ridge16
from openjev.research.pose_support import PoseSupport

SEEDS = (1101, 1202, 1303)
SUPPORT_VARIANTS = ("prior", "full", "recent5", "recent5_mass", "decay5", "decay5_mass",
                    "huber3", "huber3_mass", "decay_huber3", "decay_huber3_mass")
OLD_VARIANTS = ("static", "static_adapt", "public", "gru")
VARIANTS = SUPPORT_VARIANTS + OLD_VARIANTS
REFERENCES = ("hold", "cv1", "cv16", "ls16", "body16", "ridge16")
PANELS = ("test_sin", "test_zigzag")
PRIMARY = "decay_huber3"
GATE = ("decay_huber3 RMSE <=0.90*each positive control on both physical endpoints/panels; "
        "all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out "
        "MSE strictly lower; median full-window latency<=1.5*gru. All17 groups required.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sources():
    return [Path(p) for p in (
        "scripts/run_pose_support.py", "scripts/audit_pose_support.py",
        "scripts/audit_pose_adaptation.py", "scripts/audit_pose_transport.py",
        "scripts/train_pose_adaptation.py", "src/openjev/research/pose_support.py",
        "src/openjev/research/pose_adaptation.py", "src/openjev/research/pose_transport.py",
        "src/openjev/research/pose_references.py", "src/openjev/research/rigid_motion.py",
        "tests/test_pose_support.py", "tests/test_pose_support_runner.py",
        "tests/test_audit_pose_support.py", "research/pose-support-protocol.md",
        "research/pose-adaptation-failure-diagnosis.md")]


def checkpoint_name(variant, seed):
    if variant in SUPPORT_VARIANTS:
        return f"meta-{seed}"
    if variant in OLD_VARIANTS:
        return f"{'static' if variant == 'static_adapt' else variant}-{seed}"
    if variant == "ridge16":
        return variant
    raise ValueError("variant has no checkpoint")


def forecast_support(model, context_p, context_r, past_actions, future_actions,
                     variant, *, diagnostics=False):
    """The public forecasting boundary accepts context and actions, not targets."""
    if variant not in SUPPORT_VARIANTS:
        raise ValueError("unknown support variant")
    if future_actions.ndim != 3 or future_actions.shape[1:] != (25, 40):
        raise ValueError("forecast requires exactly25 action blocks")
    state = model.fit_context(context_p, context_r, past_actions, variant=variant,
                              return_diagnostics=diagnostics)
    if diagnostics:
        state, info = state
        pp, rr = model.rollout(state, future_actions)
        return pp, rr, info
    return model.rollout(state, future_actions)


def freeze(data, prior, out):
    old = json.loads((prior / "protocol.json").read_text())
    done = json.loads((prior / "run-01/completed.json").read_text())
    if done["status"] != "completed" or done["protocol_sha256"] != sha(prior / "protocol.json"):
        raise ValueError("completed prior experiment required")
    if old["study"] != "pose-adaptation-v1" or old["data"] != str(data.resolve()):
        raise ValueError("prior study/data identity")
    data_hashes = {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()}
    if data_hashes != old["data_hashes"]:
        raise ValueError("prepared data differ from the prior experiment")
    for name, binding in done["files"].items():
        if sha(prior / "run-01" / name) != binding["sha256"]:
            raise ValueError("prior payload changed: " + name)
    for name, digest in old["sources"].items():
        if sha(name) != digest:
            raise ValueError("prior source changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    names = [f"{v}-{s}" for s in SEEDS for v in ("meta", "static", "public", "gru")] + ["ridge16"]
    protocol = {
        "study": "pose-support-v1", "scope": "exposed-data fixed-checkpoint mechanism screen",
        "data": str(data.resolve()),
        "data_hashes": data_hashes,
        "prior_experiment": str(prior.resolve()), "prior_protocol_sha256": sha(prior / "protocol.json"),
        "prior_completed_sha256": sha(prior / "run-01/completed.json"),
        "checkpoints": {n: {"path": str((prior / "run-01" / (n + ".pt")).resolve()),
                            "sha256": sha(prior / "run-01" / (n + ".pt"))} for n in names},
        "sources": {str(p): sha(p) for p in sources()}, "seeds": SEEDS,
        "support_variants": SUPPORT_VARIANTS, "old_variants": OLD_VARIANTS,
        "variants": VARIANTS, "references": REFERENCES, "panels": PANELS, "primary": PRIMARY,
        "new_training_fits": 0, "new_optimizer_updates": 0, "context": 32, "horizon": 25,
        "support_rows": 30, "recent_rows": 5, "decay_half_life": 5., "huber_delta": 1.5,
        "huber_iterations": 3, "ridge_precision": 1.,
        "mass_control": "Derive paired weights fully, replace by per-output mean over30 supports, solve once more.",
        "position_loss_scale_m": .1, "rotation_loss_scale_rad": .1,
        "warmups_per_row": 3, "timed_windows_per_row": 20, "evaluation_rows": 96,
        "gate": GATE, "gate_groups": 17, "gate_comparisons": 1141,
        "no_selection": "Fixed arms and checkpoints; no retries, sweeps, excluded windows or gate changes.",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "threads": 1, "platform": platform.platform()},
    }
    write_json(out / "protocol.json", protocol)
    for p in sources():
        dst = out / "source-snapshot" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    print(json.dumps({"frozen": str(out), "sha256": sha(out / "protocol.json")}))


def run(out):
    protocol = json.loads((out / "protocol.json").read_text())
    prior = Path(protocol["prior_experiment"])
    if (sha(prior / "protocol.json") != protocol["prior_protocol_sha256"]
            or sha(prior / "run-01/completed.json") != protocol["prior_completed_sha256"]):
        raise ValueError("prior experiment binding changed")
    for name, digest in protocol["sources"].items():
        if sha(name) != digest:
            raise ValueError("frozen source changed: " + name)
    data = Path(protocol["data"])
    for name, digest in protocol["data_hashes"].items():
        if sha(data / name) != digest:
            raise ValueError("data changed: " + name)
    for name, binding in protocol["checkpoints"].items():
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("checkpoint changed: " + name)
    dest = out / "run-01"
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    write_json(dest / "started.json", {"protocol_sha256": sha(out / "protocol.json")})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        rows = []
        for panel in PANELS:
            p, r, a, ids = load_data(data, panel)
            np.savez_compressed(dest / (panel + "-targets.npz"), p=p[:, 32:].numpy(), R=r[:, 32:].numpy(), ids=ids)
            for variant in (*VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in VARIANTS else (None,):
                    checkpoint = None
                    if variant in VARIANTS or variant == "ridge16":
                        binding = protocol["checkpoints"][checkpoint_name(variant, seed)]
                        checkpoint = binding["sha256"]
                        weights = torch.load(binding["path"], map_location="cpu", weights_only=True)
                    if variant in SUPPORT_VARIANTS:
                        # Module initialization is overwritten by the pinned final checkpoint.
                        model = PoseSupport(weights["scales"])
                        model.load_state_dict(weights, strict=True)
                        model.eval()
                        fn = lambda xp, xr, xa, m=model, v=variant: forecast_support(
                            m, xp[:, :32], xr[:, :32], xa[:, :31], xa[:, 31:], v)
                    elif variant in OLD_VARIANTS:
                        model = make_model("static" if variant == "static_adapt" else variant, weights["scales"])
                        model.load_state_dict(weights, strict=True)
                        model.eval()
                        fn = lambda xp, xr, xa, m=model, v=variant: predict(m, xp, xr, xa, v)
                    elif variant == "ridge16":
                        fn = lambda xp, xr, xa, w=weights: predict_ridge16(w, xp, xr, xa)
                    else:
                        fn = lambda xp, xr, xa, v=variant: predict_reference(v, xp, xr, xa)
                    info = None
                    with torch.no_grad():
                        if variant in SUPPORT_VARIANTS:
                            pp, rr, info = forecast_support(model, p[:, :32], r[:, :32],
                                a[:, :31], a[:, 31:], variant, diagnostics=True)
                        else:
                            pp, rr = fn(p, r, a)
                        metrics = measure(pp, rr, p[:, 32:], r[:, 32:])
                        timings = []
                        for i in range(23):
                            tick = time.perf_counter_ns()
                            fn(p[i:i + 1], r[i:i + 1], a[i:i + 1])
                            elapsed = (time.perf_counter_ns() - tick) / 1e6
                            if i >= 3:
                                timings.append(elapsed)
                    row = {"panel": panel, "variant": variant, "seed": seed,
                           "checkpoint_sha256": checkpoint, "metrics": metrics,
                           "latency_ms": timings, "diagnostics": info}
                    rows.append(row)
                    label = variant if seed is None else f"{variant}-{seed}"
                    prefix = panel + "-" + label
                    np.savez_compressed(dest / (prefix + "-predictions.npz"), p=pp.numpy(), R=rr.numpy())
                    write_json(dest / (prefix + "-evaluation.json"), row)
                    print(json.dumps({"row": prefix, "metrics": metrics}), flush=True)
        write_json(dest / "completed.json", {"status": "completed", "rows": rows,
            "new_training_fits": 0, "new_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - start, "protocol_sha256": sha(out / "protocol.json"),
            "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                      for p in sorted(dest.iterdir()) if p.is_file()}})
        print(json.dumps({"completed": str(dest), "wall_seconds": time.perf_counter() - start}), flush=True)
    except BaseException as error:
        write_json(dest / "failed.json", {"error": repr(error), "wall_seconds": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.data is None or args.prior is None:
            parser.error("freeze requires --data and --prior")
        freeze(args.data, args.prior, args.out)
    else:
        run(args.out)
