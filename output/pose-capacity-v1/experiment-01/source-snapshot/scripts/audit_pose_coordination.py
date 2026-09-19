"""Saved-output audit of context-conditioned pose expert coordination.

No model, optimizer, native or random calls. Pose errors and gate composition
are independently reconstructed from authenticated arrays with NumPy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "scripts/audit_pose_support.py": "715b80a7aa544a1c3d28851750611e863ed096a85d7095af0b9732ccd14e16d3",
    "scripts/audit_pose_adaptation.py": "1147f387ec9499d66edbc60cad17afbbef00720f98495593001b6d1bd65aac01",
    "scripts/audit_pose_transport.py": "abca9f05a1043f8741a43c11342ae289be08f73763aafed02d89afa823676026",
}


def load_pinned(path, digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("pinned auditor source digest")
    module = ModuleType("_pose_coordination_bound_auditor")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102 - execute only hash-pinned local bytes
    return module


_previous = load_pinned(ROOT / "scripts/audit_pose_support.py", DEPENDENCIES["scripts/audit_pose_support.py"])
require, sha, read_json, write_json = _previous.require, _previous.sha, _previous.read_json, _previous.write_json
members, finite = _previous.members, _previous.finite
public_pose, error_arrays, reduce_errors = _previous.public_pose, _previous.error_arrays, _previous.reduce_errors
rotation_quality, rotation_angles = _previous.rotation_quality, _previous.rotation_angles
validate_metrics, pooled, percent = _previous.validate_metrics, _previous.pooled, _previous.percent
SEEDS = (1101, 1202, 1303)
TRAIN_VARIANTS = ("constant", "summary", "recurrent")
NEW_VARIANTS = ("fast", "slow", "position_fast", "position_slow", "half", *TRAIN_VARIANTS)
ALIASES = {"fast": "decay_huber3", "slow": "gru"}
VARIANTS = (*_previous.VARIANTS, "position_fast", "position_slow", "half", *TRAIN_VARIANTS)
REFERENCES = _previous.REFERENCES
PANELS, ENDPOINTS = _previous.PANELS, _previous.ENDPOINTS
PRIMARY = "recurrent"
BLEND_TOLERANCE = {"rtol": 2e-6, "atol": 2e-6}


def aggregate(rows):
    families, references, latency, comparisons, checks = {}, {}, {}, {}, []
    for variant in (*VARIANTS, *REFERENCES):
        labels = [f"{variant}-{seed}" for seed in SEEDS] if variant in VARIANTS else [variant]
        samples = np.asarray([t for panel in PANELS for label in labels for t in rows[panel][label]["latency_ms"]])
        require(samples.shape == ((120,) if variant in VARIANTS else (40,))
                and np.isfinite(samples).all() and (samples > 0).all(), "latency coverage")
        latency[variant] = {"samples": len(samples), "median_ms": float(np.median(samples)),
                            "p95_ms": float(np.percentile(samples, 95))}
    for panel in PANELS:
        families[panel] = {v: pooled([rows[panel][f"{v}-{seed}"]["errors"] for seed in SEEDS]) for v in VARIANTS}
        references[panel] = {v: rows[panel][v]["errors"] for v in REFERENCES}
        comparisons[panel] = {}
        for control in (*[v for v in VARIANTS if v != PRIMARY], *REFERENCES):
            comparisons[panel][control] = {}
            for endpoint in ENDPOINTS:
                candidate = families[panel][PRIMARY][endpoint]
                comparator = (families[panel][control] if control in VARIANTS else references[panel][control])[endpoint]
                base = f"{panel}/{control}/{endpoint}"
                checks.append({"name": base + "/family_10percent", "passed": comparator["rmse"] > 0 and
                    Decimal(str(candidate["rmse"])) <= Decimal(".90") * Decimal(str(comparator["rmse"])),
                    "candidate": candidate["rmse"], "comparator": comparator["rmse"], "rule": "RMSE <=0.90*positive control"})
                paired = []
                for seed in SEEDS:
                    c = rows[panel][f"{PRIMARY}-{seed}"]["errors"][endpoint]["mse"]
                    b = rows[panel][f"{control}-{seed}" if control in VARIANTS else control]["errors"][endpoint]["mse"]
                    checks.append({"name": base + f"/pair_{seed}_nonworse", "passed": c <= b,
                                   "candidate": c, "comparator": b, "rule": "MSE <= control"})
                    paired.append({"seed": seed, "candidate_mse": c, "control_mse": b,
                                   "rmse_improvement_percent": percent(math.sqrt(c), math.sqrt(b))})
                c = np.asarray(candidate["parent_mse"])
                b = np.asarray(comparator["parent_mse"])
                require(c.shape == b.shape == (10,) and candidate["parent_ids"] == comparator["parent_ids"] == list(range(10)),
                        "parent coverage")
                count = int(np.sum(c <= b))
                checks.append({"name": base + "/parents_nonworse", "passed": count >= 8,
                               "count": count, "total": 10, "rule": "at least8/10 parent MSE <= control"})
                loo = []
                for i in range(10):
                    cm, bm = float(np.delete(c, i).mean()), float(np.delete(b, i).mean())
                    checks.append({"name": base + f"/leave_out_{i}", "passed": cm < bm,
                                   "candidate": cm, "comparator": bm, "rule": "remaining-parent mean MSE < control"})
                    loo.append({"excluded_parent": i, "candidate_mse": cm, "control_mse": bm})
                differences = b - c
                net = float(differences.sum())
                best = int(np.argmax(differences))
                comparisons[panel][control][endpoint] = {
                    "family_rmse_improvement_percent": percent(candidate["rmse"], comparator["rmse"]),
                    "paired": paired, "parents_nonworse": count, "parent_mse_improvements": differences.tolist(),
                    "largest_gain_parent": best, "largest_gain_fraction_of_net": None if net <= 0 else float(differences[best] / net),
                    "leave_one_parent_out": loo}
    c, b = latency[PRIMARY]["median_ms"], latency["gru"]["median_ms"]
    checks.append({"name": "latency_vs_gru", "passed": Decimal(str(c)) <= Decimal("1.5") * Decimal(str(b)),
                   "candidate": c, "comparator": b, "rule": "pooled median <=1.5*gru"})
    require(len(checks) == 1501, "gate count")
    grouped = []
    for panel in PANELS:
        for endpoint in ENDPOINTS:
            for category, suffix in (("family", "/family_10percent"), ("paired", "_nonworse"),
                                     ("parents", "/parents_nonworse"), ("leave_one_out", "/leave_out_")):
                selected = [x for x in checks if x["name"].startswith(panel + "/")
                            and f"/{endpoint}/" in x["name"]
                            and (suffix in x["name"] if category == "leave_one_out" else x["name"].endswith(suffix))
                            and ("/pair_" in x["name"] if category == "paired" else True)]
                require(len(selected) == {"family": 25, "paired": 75, "parents": 25, "leave_one_out": 250}[category],
                        "grouped gate membership")
                grouped.append({"name": f"{panel}/{endpoint}/{category}",
                                "passed": all(x["passed"] for x in selected),
                                "comparisons_passed": sum(x["passed"] for x in selected),
                                "total_comparisons": len(selected),
                                "comparison_names": [x["name"] for x in selected]})
    grouped.append({"name": "latency_vs_gru", "passed": checks[-1]["passed"],
                    "comparisons_passed": int(checks[-1]["passed"]), "total_comparisons": 1,
                    "comparison_names": [checks[-1]["name"]]})
    require(len(grouped) == 17 and sum(x["total_comparisons"] for x in grouped) == 1501, "group count")
    return families, references, latency, comparisons, {"passed": all(x["passed"] for x in checks),
        "requirements_passed": sum(x["passed"] for x in grouped), "total_requirements": 17,
        "requirements": grouped, "checks_passed": sum(x["passed"] for x in checks),
        "total_checks": 1501, "checks": checks,
        "components": {"family": 100, "paired": 300, "parent_counts": 100, "leave_one_parent_out": 1000, "latency": 1}}




def rotation_log(matrix):
    """Independent principal SO(3) log, matching the declared near-pi sign convention."""
    rotation_quality(matrix)
    r = matrix.astype(np.float64)
    skew = .5 * np.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                          r[..., 1, 0] - r[..., 0, 1]), axis=-1)
    sine = np.linalg.norm(skew, axis=-1)
    cosine = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) * .5, -1., 1.)
    theta = np.arctan2(sine, cosine)
    factor = np.where(theta < 1e-3, 1 + theta ** 2 / 6 + 7 * theta ** 4 / 360,
                      theta / np.where(sine > 1e-4, sine, 1.))
    result = factor[..., None] * skew
    for index in np.argwhere(cosine < -.99):
        index = tuple(index)
        outer = (r[index] + r[index].T - 2 * cosine[index] * np.eye(3)) / (2 * max(1 - cosine[index], 1e-4))
        largest = int(np.argmax(np.diag(outer)))
        axis = outer[:, largest] / math.sqrt(max(outer[largest, largest], 1e-8))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        if np.dot(axis, skew[index]) < 0:
            axis = -axis
        result[index] = theta[index] * axis
    return result


def rotation_exp(vector):
    require(vector.shape[-1] == 3 and np.isfinite(vector).all(), "rotation exponential input")
    v = vector.astype(np.float64)
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    cross = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(*v.shape[:-1], 3, 3)
    theta = np.linalg.norm(v, axis=-1)
    return np.eye(3) + np.sinc(theta / np.pi)[..., None, None] * cross + (
        .5 * np.sinc(theta / (2 * np.pi)) ** 2)[..., None, None] * (cross @ cross)


def reconstruct_blend(fast_p, fast_r, slow_p, slow_r, alpha):
    require(fast_p.shape == slow_p.shape and fast_p.ndim == 3 and fast_p.shape[-1] == 3
            and fast_r.shape == slow_r.shape == (*fast_p.shape[:-1], 3, 3), "expert forecast shapes")
    require(all(x.dtype == np.float32 and np.isfinite(x).all() for x in (fast_p, fast_r, slow_p, slow_r, alpha))
            and alpha.shape == (len(fast_p), 2) and (alpha >= 0).all() and (alpha <= 1).all(), "float32 experts/gates")
    rotation_quality(fast_r); rotation_quality(slow_r)
    ap = alpha[:, 0, None, None].astype(float)
    ar = alpha[:, 1, None, None].astype(float)
    positions = (ap * fast_p.astype(float) + (1 - ap) * slow_p.astype(float)).astype(np.float32)
    relative = fast_r.astype(float) @ np.swapaxes(slow_r.astype(float), -1, -2)
    log = rotation_log(relative)
    rotations = (rotation_exp(ar * log) @ slow_r.astype(float)).astype(np.float32)
    positions = np.where(ap == 0, slow_p, np.where(ap == 1, fast_p, positions))
    rotations = np.where(ar[..., None] == 0, slow_r, np.where(ar[..., None] == 1, fast_r, rotations))
    return positions, rotations, {"relative_angle_max_rad": float(np.linalg.norm(log, axis=-1).max()),
        "near_pi_relative_rotations": int(np.sum((np.trace(relative, axis1=-2, axis2=-1) - 1) * .5 < -.99))}


def validate_blend(variant, p, rotation, alpha, fast_p, fast_r, slow_p, slow_r):
    require(variant in NEW_VARIANTS, "unknown new variant")
    fixed = {"fast": (1., 1.), "slow": (0., 0.), "position_fast": (1., 0.),
             "position_slow": (0., 1.), "half": (.5, .5)}
    if variant in fixed:
        require(np.array_equal(alpha, np.broadcast_to(np.array(fixed[variant], np.float32), alpha.shape)),
                "fixed composition gates")
    ep, er, diagnostic = reconstruct_blend(fast_p, fast_r, slow_p, slow_r, alpha)
    require(p.shape == ep.shape and rotation.shape == er.shape and p.dtype == rotation.dtype == np.float32
            and np.allclose(p, ep, **BLEND_TOLERANCE) and np.allclose(rotation, er, **BLEND_TOLERANCE),
            "independent blend mismatch")
    for channel, actual, slow, fast in ((0, p, slow_p, fast_p), (1, rotation, slow_r, fast_r)):
        for coefficient, expert in ((0, slow), (1, fast)):
            mask = alpha[:, channel] == coefficient
            require(np.array_equal(actual[mask], expert[mask]), "exact blend endpoint")
    return {**diagnostic, "position_max_abs_error": float(np.abs(p.astype(float) - ep.astype(float)).max()),
        "rotation_max_abs_error": float(np.abs(rotation.astype(float) - er.astype(float)).max()),
        "alpha_mean": alpha.mean(axis=0, dtype=np.float64).tolist(),
        "alpha_min": alpha.min(axis=0).tolist(), "alpha_max": alpha.max(axis=0).tolist(),
        "alpha_std": alpha.std(axis=0, dtype=np.float64).tolist(),
        "alpha_fraction_lt_005": (alpha < .05).mean(axis=0).tolist(),
        "alpha_fraction_gt_095": (alpha > .95).mean(axis=0).tolist(),
        "tolerance": BLEND_TOLERANCE}


def tensor_weights(path, variant, *, initial=False):
    require(variant in TRAIN_VARIANTS, "unknown gate variant")
    shapes = {"constant": {"logits": (2,)},
        "summary": {"network.0.weight": (22, 147), "network.0.bias": (22,),
                    "network.2.weight": (2, 22), "network.2.bias": (2,)},
        "recurrent": {"gru.weight_ih_l0": (48, 49), "gru.weight_hh_l0": (48, 16),
                      "gru.bias_ih_l0": (48,), "gru.bias_hh_l0": (48,),
                      "head.weight": (2, 16), "head.bias": (2,)}}[variant]
    values = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(values, dict) and set(values) == set(shapes), "gate weight fields")
    for key, shape in shapes.items():
        value = values[key]
        require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == "cpu"
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "gate tensor " + key)
    if initial:
        keys = {"constant": ("logits",), "summary": ("network.2.weight", "network.2.bias"),
                "recurrent": ("head.weight", "head.bias")}[variant]
        require(all(bool((values[k] == 0).all()) for k in keys), "initial half mixture")
    return values

TOKEN_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}
SUPPORT_PROTOCOL_SHA256 = "983848f161fe7d96a3fb4cf1b932115248da264c308b5b5a707badadb1151841"
SUPPORT_COMPLETED_SHA256 = "a946f05381c1b882e6760260e3fc424f5cc15e29971782dcbbdfff643f73978f"
SUPPORT_SUMMARY_SHA256 = "78d9d18f2204d8a54319b82ecd38de87a220d9fe74296e44dd7dece477e0f1f5"
SUPPORT_RECEIPT_SHA256 = "0c411aa9fc7bfc187106c0b3c5e53ea7408830e06a75cc2ced3e9f5da1b91cd2"
SOURCES = {"scripts/train_pose_coordination.py", "scripts/audit_pose_coordination.py",
    "src/openjev/research/pose_coordination.py", "tests/test_pose_coordination.py",
    "tests/test_pose_coordination_training.py", "tests/test_audit_pose_coordination.py",
    "research/pose-coordination-protocol.md", "scripts/run_pose_support.py", *DEPENDENCIES,
    "scripts/train_pose_adaptation.py", "src/openjev/research/pose_support.py",
    "src/openjev/research/pose_adaptation.py", "src/openjev/research/pose_transport.py",
    "src/openjev/research/pose_references.py", "src/openjev/research/rigid_motion.py"}
GATE = ('recurrent RMSE <=0.90*each positive control on both physical endpoints/panels; '
        'all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out '
        'MSE strictly lower; median full-window latency<=1.5*new slow. All17 groups required.')


def expected_members():
    files = {"started.json", "completed.json", "training-completed.json", "train-tokens.npy"}
    for seed in SEEDS:
        files.add(f"orders-{seed}.npy")
        files.update(f"train-{v}-{seed}.npz" for v in ("fast", "slow"))
        files.update(f"{v}-{seed}{suffix}" for v in TRAIN_VARIANTS
                     for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
    for panel in PANELS:
        files.add(panel + "-targets.npz")
        files.update(f"{panel}-{v}-{seed}{suffix}" for v in NEW_VARIANTS for seed in SEEDS
                     for suffix in ("-predictions.npz", "-evaluation.json"))
    require(len(files) == 147, "internal file count")
    return files


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "protocol digest")
    protocol = read_json(experiment / "protocol.json")
    required = {"study": "pose-coordination-v1", "scope": "exposed-data frozen-expert coordination screen",
        "seeds": list(SEEDS), "train_variants": list(TRAIN_VARIANTS), "variants": list(NEW_VARIANTS),
        "panels": list(PANELS), "primary": PRIMARY, "context": 32, "tokens": 31, "token_dim": 49, "horizon": 25,
        "epochs": 30, "batch": 32, "optimizer": "Adam", "learning_rate": .001, "grad_norm_cap": 1.,
        "expected_fits": 9, "updates_per_fit": 690, "evaluation_rows": 48, "execution_files": 147,
        "position_loss_scale_m": .1, "rotation_loss_scale_rad": .1, "warmups_per_row": 3, "timed_windows_per_row": 20,
        "gate": GATE, "gate_groups": 17, "gate_comparisons": 1501,
        "blend": "alpha is fast fraction; linear position, Exp(alpha_R*Log(R_fast*R_slow.T))*R_slow; exact endpoints",
        "blend_tolerance": {**BLEND_TOLERANCE, "hard_endpoints": "exact"},
        "expert_replay_tolerance": {"rtol": 1e-6, "atol": 2e-7}, "token_tolerance": TOKEN_TOLERANCE,
        "support_protocol_sha256": SUPPORT_PROTOCOL_SHA256, "support_completed_sha256": SUPPORT_COMPLETED_SHA256,
        "support_summary_sha256": SUPPORT_SUMMARY_SHA256, "support_receipt_sha256": SUPPORT_RECEIPT_SHA256,
        "no_selection": "Fixed arms, final checkpoints and seeds; no retries, sweeps, exclusions or gate changes."}
    require(all(protocol[k] == v for k, v in required.items()), "protocol settings")
    require(set(protocol["sources"]) == SOURCES and all(protocol["sources"][k] == v for k, v in DEPENDENCIES.items()),
            "source membership/pinned dependencies")
    require({k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}
            == protocol["sources"] == {p: sha(ROOT / p) for p in SOURCES}, "source hashes")
    require(protocol["runtime"] == {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
        "numpy": np.__version__, "threads": 1, "platform": platform.platform()}, "runtime identity")
    old = Path(protocol["support_experiment"])
    previous_protocol, data, _adaptation_run, _adaptation_members = _previous.validate_inputs(old, SUPPORT_PROTOCOL_SHA256)
    require(protocol["data"] == str(data.resolve()) and protocol["data_hashes"] == previous_protocol["data_hashes"],
            "unchanged public data")
    prior_run = old / "run-01"; prior_members = members(prior_run)
    require(set(prior_members) == _previous.expected_members()
            and prior_members["completed.json"]["sha256"] == SUPPORT_COMPLETED_SHA256, "support completed closure")
    previous = read_json(prior_run / "completed.json")
    require(previous["status"] == "completed" and previous["protocol_sha256"] == SUPPORT_PROTOCOL_SHA256
            and previous["files"] == {k: v for k, v in prior_members.items() if k != "completed.json"}, "support payload seal")
    report = Path(protocol["support_report"])
    require(sha(report / "summary.json") == SUPPORT_SUMMARY_SHA256 and sha(report / "receipt.json") == SUPPORT_RECEIPT_SHA256,
            "support report identities")
    receipt, summary = read_json(report / "receipt.json"), read_json(report / "summary.json")
    require(receipt["status"] == summary["status"] == "completed"
            and receipt["execution_completed_sha256"] == summary["execution_completed_sha256"] == SUPPORT_COMPLETED_SHA256
            and receipt["execution_members"] == prior_members, "support audit binding")
    require(set(members(report)) == {"summary.json", "receipt.json", "window-errors.npz"}
            and receipt["files"] == {k: v for k, v in members(report).items() if k != "receipt.json"}, "support report seal")
    expected_checkpoints = {f"{v}-{s}": previous_protocol["checkpoints"][f"{v}-{s}"] for s in SEEDS for v in ("meta", "gru")}
    require(protocol["checkpoints"] == expected_checkpoints, "unchanged expert checkpoint identities")
    scales = []
    for name, binding in expected_checkpoints.items():
        state = _previous._previous.tensor_weights(Path(binding["path"]), name.split('-')[0])
        scales.append(state["scales"].numpy())
    require(all(np.array_equal(scales[0], x) for x in scales), "expert scale equality")
    return {"protocol": protocol, "data": data, "prior_run": prior_run, "prior_members": prior_members,
            "prior_summary": summary, "scales": scales[0]}


def load_public(data, split):
    with np.load(data / "normalization.npz", allow_pickle=False) as normal:
        mean, scale = normal["obs_mean"], normal["obs_scale"]
    require(mean.shape == scale.shape == (9,) and np.isfinite(mean).all() and np.isfinite(scale).all()
            and (scale > 0).all(), "normalization")
    with np.load(data / (split + ".npz"), allow_pickle=False) as arrays:
        p, r = public_pose(arrays["obs"].astype(float) * scale + mean)
        a = arrays["actions"].copy()
        ids = np.column_stack((arrays["source_ids"], arrays["window_starts"]))
    return p, r, a, ids


def public_tokens(p, rotation, actions, scales):
    require(p.shape[1:] == (32, 3) and rotation.shape == (len(p), 32, 3, 3)
            and actions.shape == (len(p), 31, 40) and scales.shape == (4,) and (scales > 0).all(), "context token inputs")
    current = rotation[:, 1:].astype(float)
    inverse = np.swapaxes(current, -1, -2)
    dp = (p[:, 1:] - p[:, :-1]).astype(float)
    w = rotation_log(current @ np.swapaxes(rotation[:, :-1].astype(float), -1, -2))
    features = np.concatenate((current[:, :, 2, :], (inverse @ dp[..., None])[..., 0] / scales[0],
        (inverse @ w[..., None])[..., 0] / scales[1], actions.astype(float)), axis=-1)
    return np.tanh(features).astype(np.float32)


def load_prediction(path, *, alpha=False):
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == ({"p", "R", "alpha"} if alpha else {"p", "R"}), "prediction fields")
        return tuple(arrays[k] for k in (("p", "R", "alpha") if alpha else ("p", "R")))


def audit_training(run, done, context, before):
    boundary = read_json(run / "training-completed.json")
    require(set(boundary) == {"fits", "caches", "evaluation_started"} and boundary["evaluation_started"] is False
            and boundary["fits"] == done["fits"] and boundary["caches"] == done["caches"]
            and len(done["fits"]) == 9 and len(done["caches"]) == 3, "training-before-evaluation boundary")
    p, r, actions, _ids = load_public(context["data"], "train")
    require(p.shape == (720, 57, 3) and actions.shape == (720, 56, 40), "training public shape")
    tokens = np.load(run / "train-tokens.npy", allow_pickle=False)
    expected = public_tokens(p[:, :32], r[:, :32], actions[:, :31], context["scales"])
    require(tokens.shape == (720, 31, 49) and tokens.dtype == np.float32 and np.isfinite(tokens).all()
            and (np.abs(tokens) <= 1).all() and np.allclose(tokens, expected, **TOKEN_TOLERANCE), "training context token identity")
    fits, final_weights = {}, {}
    for i, seed in enumerate(SEEDS):
        require(set(done["caches"][i]) == {"seed", "seconds"} and done["caches"][i]["seed"] == seed, "cache order")
        finite(done["caches"][i]["seconds"], "cache seconds", positive=True)
        orders = np.load(run / f"orders-{seed}.npy", allow_pickle=False)
        require(orders.shape == (30, 720) and orders.dtype == np.int64
                and np.array_equal(np.sort(orders, axis=1), np.broadcast_to(np.arange(720), (30, 720))), "complete batch permutations")
        for expert in ("fast", "slow"):
            pp, rr = load_prediction(run / f"train-{expert}-{seed}.npz")
            require(pp.shape == (720, 25, 3) and rr.shape == (720, 25, 3, 3)
                    and pp.dtype == rr.dtype == np.float32 and np.isfinite(pp).all(), "training expert cache")
            rotation_quality(rr)
        for j, variant in enumerate(TRAIN_VARIANTS):
            label = f"{variant}-{seed}"; fit = read_json(run / (label + "-fit.json"))
            require(set(fit) == {"variant", "seed", "updates", "parameters", "training_seconds", "initial_sha256",
                                "checkpoint_sha256", "orders_sha256"}, "fit fields")
            require(fit == done["fits"][3 * i + j] and fit["variant"] == variant and fit["seed"] == seed
                    and fit["updates"] == 690 and fit["orders_sha256"] == before[f"orders-{seed}.npy"]["sha256"], "fit pairing/order")
            for key, suffix in (("initial_sha256", "-initial.pt"), ("checkpoint_sha256", ".pt")):
                require(fit[key] == before[label + suffix]["sha256"], "gate weight binding")
            tensor_weights(run / (label + "-initial.pt"), variant, initial=True)
            state = tensor_weights(run / (label + ".pt"), variant)
            require(fit["parameters"] == sum(x.numel() for x in state.values()), "gate parameter count")
            losses = np.load(run / (label + "-losses.npy"), allow_pickle=False)
            require(losses.shape == (690,) and losses.dtype == np.float64 and np.isfinite(losses).all()
                    and (losses >= 0).all(), "loss record coverage")
            finite(fit["training_seconds"], "fit seconds", positive=True)
            fits[label], final_weights[label] = fit, state
    return fits, final_weights, {"max_abs_error": float(np.abs(tokens.astype(float) - expected.astype(float)).max()),
                                 "tolerance": TOKEN_TOLERANCE}


def audit(experiment, out, *, protocol_sha256=None, completed_sha256=None):
    start = time.perf_counter()
    require((protocol_sha256 is None) == (completed_sha256 is None), "supply both external hashes or neither")
    external = protocol_sha256 is not None
    out.mkdir(parents=True, exist_ok=False)
    try:
        run = experiment / "run-01"
        protocol_sha256 = protocol_sha256 or sha(experiment / "protocol.json")
        completed_sha256 = completed_sha256 or sha(run / "completed.json")
        require(sha(run / "completed.json") == completed_sha256, "completion digest")
        context = validate_inputs(experiment, protocol_sha256); protocol = context["protocol"]
        before = members(run); require(set(before) == expected_members(), "exact completed membership")
        done = read_json(run / "completed.json")
        require(set(done) == {"status", "fits", "caches", "rows", "new_training_fits", "new_optimizer_updates", "wall_seconds",
                              "protocol_sha256", "files"} and done["status"] == "completed"
                and done["protocol_sha256"] == protocol_sha256 and done["new_training_fits"] == 9
                and done["new_optimizer_updates"] == 6210 and len(done["rows"]) == 48
                and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "producer seal/counts")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "start binding")
        fits, gate_weights, token_check = audit_training(run, done, context, before)
        rows, active_rows, window_errors, replay, blending, row_index = {}, {}, {}, {}, {}, 0
        for panel in PANELS:
            ep, er, _actions, expected_ids = load_public(context["data"], panel)
            with np.load(run / (panel + "-targets.npz"), allow_pickle=False) as target:
                require(set(target.files) == {"p", "R", "ids"}, "target fields")
                tp, tr, ids = target["p"], target["R"], target["ids"]
            require(tp.shape == (160, 25, 3) and tr.shape == (160, 25, 3, 3) and tp.dtype == tr.dtype == np.float32
                    and ids.dtype == np.int64 and np.array_equal(ids, expected_ids)
                    and np.array_equal(tp, ep[:, 32:]) and np.allclose(tr, er[:, 32:], rtol=0, atol=2e-7), "target provenance")
            with np.load(context["prior_run"] / (panel + "-targets.npz"), allow_pickle=False) as target:
                require(all(np.array_equal(target[k], v) for k, v in (("p", tp), ("R", tr), ("ids", ids))), "unchanged prior targets")
            rows[panel], active_rows[panel], replay[panel], blending[panel] = {}, {}, {}, {}
            window_errors[panel + "__ids"] = ids
            for variant in (*_previous.VARIANTS, *REFERENCES):
                for seed in SEEDS if variant in _previous.VARIANTS else (None,):
                    label = variant if seed is None else f"{variant}-{seed}"; prefix = panel + "-" + label
                    pp, rr = load_prediction(context["prior_run"] / (prefix + "-predictions.npz"))
                    require(pp.dtype == rr.dtype == (np.float64 if variant == "ridge16" else np.float32), "prior precision")
                    error = error_arrays(pp, rr, tp, tr); calculated = reduce_errors(error, ids)
                    inherited = context["prior_summary"]["rows"][panel][label]
                    require(calculated == inherited["errors"], "independent inherited error arithmetic")
                    old_row = read_json(context["prior_run"] / (prefix + "-evaluation.json"))
                    validate_metrics(old_row["metrics"], calculated, rr)
                    require(old_row["latency_ms"] == inherited["latency_ms"], "inherited latency identity")
                    rows[panel][label] = {"variant": variant, "seed": seed, "errors": calculated,
                        "latency_ms": inherited["latency_ms"], "latency_scope": "inherited descriptive only",
                        "checkpoint_sha256": inherited["checkpoint_sha256"]}
                    for endpoint, value in error.items():
                        window_errors[prefix + "__" + endpoint] = value
            experts = {seed: {v: load_prediction(run / f"{panel}-{v}-{seed}-predictions.npz", alpha=True) for v in ("fast", "slow")}
                       for seed in SEEDS}
            for variant in NEW_VARIANTS:
                for seed in SEEDS:
                    label = f"{variant}-{seed}"; prefix = panel + "-" + label
                    row = read_json(run / (prefix + "-evaluation.json"))
                    require(set(row) == {"panel", "variant", "seed", "checkpoint_sha256", "expert_sha256", "metrics", "latency_ms"}
                            and row == done["rows"][row_index] and row["panel"] == panel and row["variant"] == variant
                            and row["seed"] == seed, "evaluation identity/order")
                    row_index += 1
                    checkpoint = fits[label]["checkpoint_sha256"] if variant in TRAIN_VARIANTS else None
                    require(row["checkpoint_sha256"] == checkpoint and row["expert_sha256"] == {
                        n: protocol["checkpoints"][f"{n}-{seed}"]["sha256"] for n in ("meta", "gru")}, "row weights binding")
                    pp, rr, alpha = load_prediction(run / (prefix + "-predictions.npz"), alpha=True)
                    fp, fr, _fa = experts[seed]["fast"]; sp, sr, _sa = experts[seed]["slow"]
                    blending[panel][label] = validate_blend(variant, pp, rr, alpha, fp, fr, sp, sr)
                    if variant == "constant":
                        logits = gate_weights[label]["logits"].numpy().astype(float)
                        probabilities = np.array([1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x)) for x in logits])
                        require(np.allclose(alpha, probabilities[None], rtol=1e-6, atol=2e-7), "constant logits/gates")
                    error = error_arrays(pp, rr, tp, tr); calculated = reduce_errors(error, ids)
                    validate_metrics(row["metrics"], calculated, rr)
                    samples = np.asarray(row["latency_ms"], dtype=float)
                    require(samples.shape == (20,) and np.isfinite(samples).all() and (samples > 0).all(), "active latency samples")
                    active = {"variant": variant, "seed": seed, "errors": calculated, "latency_ms": row["latency_ms"],
                        "checkpoint_sha256": checkpoint, "expert_sha256": row["expert_sha256"], "latency_scope": "fresh complete forecast"}
                    active_rows[panel][label] = active
                    for endpoint, value in error.items():
                        window_errors[prefix + "__" + endpoint] = value
                    if variant in ALIASES:
                        alias = f"{ALIASES[variant]}-{seed}"
                        replay[panel][label] = _previous.validate_replay(pp, rr, context["prior_run"] / f"{panel}-{alias}-predictions.npz")
                        # One canonical control: inherited errors, freshly measured latency.
                        rows[panel][alias]["latency_ms"] = row["latency_ms"]
                        rows[panel][alias]["latency_scope"] = "fresh alias " + variant
                    else:
                        rows[panel][label] = active
        require(sum(len(x) for x in replay.values()) == 12 and sum(len(x) for x in rows.values()) == 132, "alias/control coverage")
        families, references, latency, contrasts, gate = aggregate(rows)
        for variant in latency:
            latency[variant]["scope"] = ("fresh complete forecast" if variant in ("decay_huber3", "gru", *NEW_VARIANTS[2:])
                                           else "inherited descriptive only")
        wall = finite(done["wall_seconds"], "execution seconds", positive=True)
        training = sum(x["training_seconds"] for x in fits.values())
        caching = sum(x["seconds"] for x in done["caches"])
        timing = sum(sum(row["latency_ms"]) for group in active_rows.values() for row in group.values()) / 1000
        require(training + caching + timing <= wall + 1e-6, "nested execution costs")
        summary = {"status": "completed", "study": "pose-coordination-v1", "scope": protocol["scope"],
            "protocol_sha256": protocol_sha256, "execution_completed_sha256": completed_sha256,
            "support_completed_sha256": SUPPORT_COMPLETED_SHA256, "support_summary_sha256": SUPPORT_SUMMARY_SHA256,
            "fits": fits, "rows": rows, "active_rows": active_rows, "families": families, "references": references,
            "latency": latency, "contrasts": contrasts, "continuation_gate": gate,
            "expert_replay": replay, "blend_reconstruction": blending, "training_token_check": token_check,
            "counts": {"new_fits": 9, "recorded_updates": 6210, "new_evaluation_rows": 48, "inherited_rows": 96,
                "aliased_new_rows": 12, "distinct_combined_rows": 132, "controls": 25, "new_execution_files": 147,
                "windows_per_panel": 160, "parents_per_panel": 10, "evaluated_error_pairs": 144 * 160 * 25},
            "costs": {"execution_wall_seconds": wall, "gate_training_seconds": training, "expert_cache_seconds": caching,
                "measured_active_calls_seconds": timing, "scope": "Nested components, not additive to execution wall. Wall excludes final hashing/write. Complete forecast timing includes both experts for mixed arms; inherited timings are descriptive only."},
            "limits": ["Exposed development panels, not confirmation, closed-loop utility or a novel joint dynamics architecture.",
                "Gate training uses the original training parents; frozen experts saw those windows, so this is not out-of-fold stacking.",
                "No neural inference, training, optimizer, native calls or random draws in the audit. Safe tensor loading checks saved schemas/identities.",
                "Training tokens and all output blends are independently reconstructed. Learned summary/recurrent alpha generation, cached training forecasts and optimizer updates remain source-bound.",
                "Saved permutations are complete and shared across gate arms; RNG seed generation and optimizer numerical steps are not replayed.",
                "Same frozen experts evolve privately; composed pose outputs never feed back. Valid rotations do not establish coherent shared physical dynamics.",
                "Fast/slow aliases use authenticated older errors as canonical controls and fresh timing only. Their fresh errors and parity discrepancies remain separately visible.",
                "Rotation Log inherits a deterministic near-pi convention, not global smoothness. Blend tolerance is fixed prospectively.",
                "All windows, seeds and parents remain included. The 17 grouped requirements retain 1501 dependent comparisons.",
                "Recorded future applied torques are supplied. Previous failed continuation rules remain failed."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **window_errors)
        require(members(run) == before and members(context["prior_run"]) == context["prior_members"], "evidence changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "externally_supplied_hashes": external,
            "support_completed_sha256": SUPPORT_COMPLETED_SHA256, "support_summary_sha256": SUPPORT_SUMMARY_SHA256,
            "execution_members": before, "sources": protocol["sources"], "data_hashes": protocol["data_hashes"],
            "expert_checkpoints": protocol["checkpoints"], "auditor_sha256": sha(__file__), "dependencies": DEPENDENCIES,
            "files": members(out), "qualification_passed": gate["passed"], "requirements_passed": gate["requirements_passed"],
            "total_requirements": 17, "checks_passed": gate["checks_passed"], "total_checks": 1501,
            "new_model_calls": 0, "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                "wall_seconds": time.perf_counter() - start, "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            error.add_note(f"Could not write failure receipt: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--completed-sha256")
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256,
                   completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "requirements_passed", "total_requirements",
                                           "checks_passed", "total_checks", "wall_seconds")}))
