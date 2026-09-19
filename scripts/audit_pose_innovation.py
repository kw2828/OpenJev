"""Independent saved-output arithmetic for a training-parent mechanism screen.

No model/trainer imports, forecasts, optimizers, fitting or random draws. Torch
is used only for safe checkpoint tensor loading; geometry and reductions use
NumPy. Saved neural predictions and learning trajectories remain source-bound.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = "output/residual-dynamics-v1/data-01"
PARENT = "output/pose-crossfit-v1/experiment-01"
PARENT_PROTOCOL = "b9e0d4c9a43527cd14d1931242cd130fd224eb948cf6f7d8662e43cb9d2b5143"
PARENT_COMPLETED = "3f57c8727e574a01e078ffea2e3cce5179619a72c8753e1b928d4f3115759be2"
SEEDS, FOLDS = (1101, 1202, 1303), (0, 1, 2)
NEURAL = ("summary", "recurrent", "shuffled", "noerror", "error_shuffled")
VARIANTS = ("base", "bias", "ridge_summary", "ridge_ordered", *NEURAL)
ENDPOINTS = ("position", "rotation")
TEST_RANKS = (1, 4, 7, 9)
SOURCES = {"scripts/run_pose_innovation.py", "scripts/audit_pose_innovation.py",
    "src/openjev/research/pose_innovation.py", "tests/test_pose_innovation.py",
    "tests/test_pose_innovation_runner.py", "tests/test_audit_pose_innovation.py",
    "src/openjev/research/pose_transport.py", "src/openjev/research/rigid_motion.py",
    "src/openjev/research/pose_coordination.py", "src/openjev/research/pose_adaptation.py",
    "src/openjev/research/pose_references.py", "scripts/train_pose_adaptation.py",
    "research/pose-innovation-protocol.md"}
CACHE_FIELDS = {"ids", "indices", "train_mask", "test_mask", "tokens", "target", "base_p", "base_R",
    "target_p", "target_R", "root_R", "one_errors", "permutations", "context_p", "context_R",
    "past_actions", "one_p", "one_R", "motion_scales"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def members(folder):
    paths = sorted(folder.rglob("*"))
    require(not folder.is_symlink() and not any(p.is_symlink() for p in paths), "symlink in evidence")
    return {p.relative_to(folder).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in paths if p.is_file()}


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key].copy() for key in value.files}


def finite(value, name, positive=False):
    require(type(value) in (float, int) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            "invalid " + name)
    return float(value)


def tensor(value, shape, name, dtype=np.float32):
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
            and np.isfinite(value).all(), "invalid " + name)
    return value


def rotation_quality(r):
    require(r.shape[-2:] == (3, 3) and np.isfinite(r).all(), "rotation schema")
    r = r.astype(float)
    require(np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3))) <= 1e-4
            and np.max(np.abs(np.linalg.det(r) - 1)) <= 1e-4, "proper rotation required")


def rotation_log(r):
    """Independent principal logarithm, matching the declared zero/pi branches."""
    r = np.asarray(r, dtype=float)
    skew = .5 * np.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), -1)
    sine = np.linalg.norm(skew, axis=-1)
    cosine = np.clip((np.trace(r, axis1=-2, axis2=-1)-1)*.5, -1, 1)
    angle = np.arctan2(sine, cosine)
    factor = np.where(angle < 1e-3, 1+angle**2/6+7*angle**4/360,
                      angle / np.where(sine > 1e-4, sine, 1))
    result = factor[..., None] * skew
    for index in np.argwhere(cosine < -.99):
        index = tuple(index)
        outer = (r[index]+r[index].T-2*cosine[index]*np.eye(3))/(2*max(1-cosine[index], 1e-4))
        k = int(np.argmax(np.diag(outer)))
        axis = outer[:, k] / math.sqrt(max(outer[k, k], 1e-8))
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        if np.dot(axis, skew[index]) < 0:
            axis = -axis
        result[index] = angle[index]*axis
    return result


def rotation_exp(w):
    w = np.asarray(w, dtype=float)
    x, y, z = np.moveaxis(w, -1, 0)
    zero = np.zeros_like(x)
    k = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*w.shape[:-1], 3, 3)
    theta = np.linalg.norm(w, axis=-1)
    return np.eye(3)+np.sinc(theta/np.pi)[..., None, None]*k + .5*np.sinc(theta/(2*np.pi))[..., None, None]**2*(k@k)


def residual(base_p, base_r, target_p, target_r, root_r, scales):
    frame = np.swapaxes(root_r.astype(float), -1, -2)[:, None]
    dp = (frame @ (target_p.astype(float)-base_p.astype(float))[..., None])[..., 0]/float(scales[0])
    w = rotation_log(target_r.astype(float) @ np.swapaxes(base_r.astype(float), -1, -2))
    return np.concatenate((dp, (frame@w[..., None])[..., 0]/float(scales[1])), -1)


def corrected(base_p, base_r, root_r, correction):
    frame = root_r.astype(float)[:, None]
    dp = (frame @ (.1*correction[..., :3].astype(float))[..., None])[..., 0]
    w = (frame @ (.1*correction[..., 3:].astype(float))[..., None])[..., 0]
    return base_p.astype(float)+dp, rotation_exp(w)@base_r.astype(float)


def errors(p, r, tp, tr):
    rotation_quality(r); rotation_quality(tr)
    relative = np.swapaxes(r.astype(float), -1, -2) @ tr.astype(float)
    skew = .5*np.stack((relative[..., 2, 1]-relative[..., 1, 2], relative[..., 0, 2]-relative[..., 2, 0],
                        relative[..., 1, 0]-relative[..., 0, 1]), -1)
    angle = np.arctan2(np.linalg.norm(skew, axis=-1), np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)*.5, -1, 1))
    return {"position": np.square(p.astype(float)-tp.astype(float)).sum(-1), "rotation": angle**2}


def reduce_errors(values, ids):
    parent_ids = sorted(np.unique(ids[:, 0]).tolist())
    return {endpoint: {"mse": float(value.mean()), "rmse": float(np.sqrt(value.mean())),
        "horizon_mse": value.mean(0).tolist(), "parent_ids": parent_ids,
        "parent_mse": [float(value[ids[:, 0] == parent].mean()) for parent in parent_ids]}
        for endpoint, value in values.items()}


def validate_metrics(saved, calculated):
    for name, actual in (("position_rmse_m", calculated["position"]["rmse"]),
                         ("rotation_rmse_rad", calculated["rotation"]["rmse"]),
                         ("composite", (calculated["position"]["mse"]+calculated["rotation"]["mse"])/.01)):
        require(math.isclose(finite(saved[name], name), actual, rel_tol=1e-9, abs_tol=1e-12), "metric " + name)
    for name in ("rotation_orthogonality_max", "rotation_determinant_max_error"):
        require(finite(saved[name], name) <= 1e-4, "saved rotation quality")


def split(ids, fold):
    expected = np.column_stack((np.repeat(np.arange(30), 24), np.tile(np.arange(24)*50, 30)))
    require(ids.dtype == np.int64 and np.array_equal(ids, expected) and type(fold) is int and fold in FOLDS,
            "original training IDs and fold")
    index = np.flatnonzero(ids[:, 0] % 3 == fold)
    test = np.isin(ids[index, 0]//3, TEST_RANKS)
    return index, ~test, test


def expected_members():
    result = {"started.json", "completed.json"}
    for seed in SEEDS:
        for fold in FOLDS:
            stem = f"fold-{fold}-{seed}"
            result.update((stem+"-cache.npz", stem+"-orders.npy"))
            result.update(stem+"-"+v+"-weights.npz" for v in ("ridge_summary", "ridge_ordered"))
            for variant in NEURAL:
                result.update(stem+"-"+variant+suffix for suffix in ("-initial.pt", ".pt", "-losses.npy", "-fit.json"))
            for variant in VARIANTS:
                result.update((stem+"-"+variant+"-predictions.npz", stem+"-"+variant+"-evaluation.json"))
    require(len(result) == 380, "internal file coverage")
    return result


def validate_inputs(experiment, protocol_sha256, root=ROOT):
    require(sha(experiment/"protocol.json") == protocol_sha256, "external protocol identity")
    p = read_json(experiment/"protocol.json")
    fixed = {"study": "pose-innovation-v1", "seeds": list(SEEDS), "folds": list(FOLDS), "variants": list(VARIANTS),
        "neural_variants": list(NEURAL), "primary": "recurrent", "test_ranks": list(TEST_RANKS),
        "windows_per_parent": 24, "train_windows_per_fold": 144, "test_windows_per_fold": 96,
        "context": 32, "tokens": 31, "token_dim": 55, "horizon": 25, "epochs": 40, "batch": 32,
        "updates_per_fit": 200, "fits": 45, "updates": 9000, "optimizer": "Adam", "learning_rate": .001,
        "clip_norm": 1., "correction_scales": [.1, .1]}
    require(all(p.get(k) == v for k, v in fixed.items()), "fixed prospective settings")
    require(set(p["sources"]) == SOURCES and p["sources"] == {k: sha(root/k) for k in SOURCES}
            == {k: v["sha256"] for k, v in members(experiment/"source-snapshot").items()}, "13 source bindings")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": importlib.metadata.version("torch"),
               "platform": platform.platform(), "threads": 1}
    require(p["runtime"] == runtime, "runtime binding")
    expected = {f"{DATA}/{n}" for n in ("train.npz", "normalization.npz", "manifest.json", "completed.json")}
    expected.update((f"{PARENT}/protocol.json", f"{PARENT}/run-01/completed.json"))
    for seed in SEEDS:
        for fold in FOLDS:
            expected.update(f"{PARENT}/run-01/{name}" for name in (f"fold-{fold}-gru-{seed}.pt",
                f"fold-{fold}-gru-{seed}-fit.json", f"fold-{fold}-{seed}-normalization.npz", f"fold-{fold}-slow-{seed}-cache.npz"))
    require(set(p["bindings"]) == expected and len(expected) == 42
            and p["bindings"] == {k: sha(root/k) for k in expected}, "exact training-only lineage bindings")
    require(p["bindings"][f"{PARENT}/protocol.json"] == PARENT_PROTOCOL
            and p["bindings"][f"{PARENT}/run-01/completed.json"] == PARENT_COMPLETED, "original parent pins")
    parent, completed = read_json(root/PARENT/"protocol.json"), read_json(root/PARENT/"run-01/completed.json")
    require(completed["status"] == "completed" and completed["protocol_sha256"] == PARENT_PROTOCOL, "completed parent")
    for name, digest in parent["sources"].items():
        require(sha(root/name) == digest, "frozen inherited source")
    for name in expected:
        if name.startswith(DATA+"/"):
            require(p["bindings"][name] == parent["data_hashes"][Path(name).name], "parent data identity")
        elif "/run-01/" in name and not name.endswith("completed.json"):
            require(p["bindings"][name] == completed["files"][Path(name).name]["sha256"], "parent artifact identity")
    return p


def original_training(root):
    normal, raw = arrays(root/DATA/"normalization.npz"), arrays(root/DATA/"train.npz")
    obs = raw["obs"].astype(float)*normal["obs_scale"]+normal["obs_mean"]
    tensor(obs, (720, 57, 9), "original observations", np.float64)
    angle = np.arctan2(obs[..., 3:6], obs[..., 6:9])
    roll, pitch, yaw = np.moveaxis(angle, -1, 0)
    sr, sp, sy, cr, cp, cy = np.sin(roll), np.sin(pitch), np.sin(yaw), np.cos(roll), np.cos(pitch), np.cos(yaw)
    r = np.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, sy*cp, sy*sp*sr+cy*cr,
                  sy*sp*cr-cy*sr, -sp, cp*sr, cp*cr), -1).reshape(720, 57, 3, 3).astype(np.float32)
    ids = np.stack((raw["source_ids"], raw["window_starts"]), -1)
    split(ids, 0)
    return obs[..., :3].astype(np.float32), r, tensor(raw["actions"], (720, 56, 40), "recorded actions"), ids


def validate_cache(c, fold, seed, original, root):
    require(set(c) == CACHE_FIELDS, "cache fields")
    p, r, actions, ids = original
    index, train, test = split(ids, fold)
    for key, expected in (("indices", index), ("ids", ids[index]), ("train_mask", train), ("test_mask", test)):
        require(c[key].dtype == expected.dtype and np.array_equal(c[key], expected), "cache split " + key)
    shapes = {"tokens": (240, 31, 55), "target": (240, 25, 6), "base_p": (240, 25, 3), "base_R": (240, 25, 3, 3),
        "target_p": (240, 25, 3), "target_R": (240, 25, 3, 3), "root_R": (240, 3, 3), "one_errors": (240, 31, 6),
        "context_p": (240, 32, 3), "context_R": (240, 32, 3, 3), "past_actions": (240, 31, 40),
        "one_p": (240, 31, 3), "one_R": (240, 31, 3, 3), "motion_scales": (4,)}
    for key, shape in shapes.items():
        tensor(c[key], shape, key)
    for key in ("base_R", "target_R", "root_R", "context_R", "one_R"):
        rotation_quality(c[key])
    require(np.array_equal(c["context_p"], p[index, :32]) and np.array_equal(c["target_p"], p[index, 32:])
            and np.allclose(c["context_R"], r[index, :32], rtol=0, atol=2e-7)
            and np.allclose(c["target_R"], r[index, 32:], rtol=0, atol=2e-7)
            and np.array_equal(c["root_R"], c["context_R"][:, -1]), "original training pose/root provenance")
    norm = arrays(root/PARENT/"run-01"/f"fold-{fold}-{seed}-normalization.npz")
    require(np.array_equal(norm["train_indices"], np.flatnonzero(ids[:, 0] % 3 != fold))
            and np.array_equal(c["motion_scales"], norm["motion_scales"]) and (c["motion_scales"] > 0).all(), "fold-only scales")
    normalized = (actions[index]-norm["action_mean"])/norm["action_scale"]
    require(np.array_equal(c["past_actions"], normalized[:, :31]), "fold-normalized completed actions")
    old = arrays(root/PARENT/"run-01"/f"fold-{fold}-slow-{seed}-cache.npz")
    for name, key in (("p", "base_p"), ("R", "base_R")):
        require(np.allclose(c[key], old[name][index], rtol=2e-6, atol=2e-7), "frozen backbone replay " + name)
    permutation = tensor(c["permutations"], (240, 31), "permutation", np.int64)
    require((permutation[:, 0] == 0).all() and (permutation[:, -1] == 30).all()
            and np.array_equal(np.sort(permutation, axis=1), np.broadcast_to(np.arange(31), (240, 31))), "fixed-endpoint permutations")
    predicted_target = residual(c["base_p"], c["base_R"], c["target_p"], c["target_R"], c["root_R"], [.1, .1])
    predicted_error = residual(c["one_p"], c["one_R"], c["context_p"][:, 1:], c["context_R"][:, 1:], c["root_R"], [.1, .1])
    require(np.allclose(c["target"], predicted_target, rtol=2e-5, atol=5e-6)
            and np.allclose(c["one_errors"], predicted_error, rtol=2e-5, atol=5e-6), "signed root-frame residuals")
    current = c["context_R"][:, 1:].astype(float)
    inverse = np.swapaxes(current, -1, -2)
    dp = np.diff(c["context_p"].astype(float), axis=1)
    w = rotation_log(current @ np.swapaxes(c["context_R"][:, :-1].astype(float), -1, -2))
    e = residual(c["one_p"], c["one_R"], c["context_p"][:, 1:], c["context_R"][:, 1:], c["root_R"], c["motion_scales"][2:])
    feature = np.concatenate((current[..., 2, :], (inverse@dp[..., None])[..., 0]/c["motion_scales"][0],
        (inverse@w[..., None])[..., 0]/c["motion_scales"][1], c["past_actions"].astype(float), e), -1)
    # Roundoff of float32 rotations is amplified by the frozen positive scales.
    tolerance = np.r_[np.full(3, 2e-7), np.full(3, 5e-7/float(c["motion_scales"][0])),
        np.full(3, 5e-7/float(c["motion_scales"][1])), np.full(40, 2e-7),
        np.full(3, 5e-7/float(c["motion_scales"][2])), np.full(3, 5e-7/float(c["motion_scales"][3]))]
    require((np.abs(c["tokens"].astype(float)-np.tanh(feature)) <= tolerance+2e-6*np.abs(np.tanh(feature))).all(),
            "completed-transition token arithmetic")
    return {"tokens_max_abs_error": float(np.max(np.abs(c["tokens"]-np.tanh(feature)))),
            "token_atol_by_channel": tolerance.tolist(), "target_max_abs_error": float(np.max(np.abs(c["target"]-predicted_target))),
            "one_error_max_abs_error": float(np.max(np.abs(c["one_errors"]-predicted_error)))}


def ridge_check(saved, tokens, target, train, kind):
    if kind == "ridge_summary":
        features = np.concatenate((tokens[:, -1], tokens.mean(1, dtype=np.float32), tokens[:, -1]-tokens[:, 0]), -1)
    else:
        features = tokens.reshape(len(tokens), -1)
    x = features.astype(float)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y = target.astype(float).reshape(len(target), -1)
    require(set(saved) == {"x_mean", "y_mean", "weights"}, "ridge fields")
    xm, ym, w = saved["x_mean"], saved["y_mean"], saved["weights"]
    tensor(xm, (x.shape[1],), "ridge feature mean", np.float64)
    tensor(ym, (150,), "ridge target mean", np.float64)
    tensor(w, (x.shape[1], 150), "ridge weights", np.float64)
    require(np.allclose(xm, x[train].mean(0), rtol=1e-7, atol=1e-8)
            and np.allclose(ym, y[train].mean(0), rtol=1e-10, atol=1e-10), "training-only ridge means")
    xc, yc = x[train]-xm, y[train]-ym
    rhs = xc.T@yc
    residual_normal = w+xc.T@(xc@w-yc)
    bound = 2e-6*(1+float(np.max(np.abs(rhs))))
    require(float(np.max(np.abs(residual_normal))) <= bound, "ridge normal-equation witness")
    return ((x-xm)@w+ym).reshape(target.shape), float(np.max(np.abs(residual_normal)))


def checkpoint(path, variant, initial=False):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    shapes = ({"network.0.weight": (9, 165), "network.0.bias": (9,), "network.2.weight": (150, 9), "network.2.bias": (150,)}
              if variant == "summary" else {"gru.weight_ih_l0": (24, 55), "gru.weight_hh_l0": (24, 8),
                "gru.bias_ih_l0": (24,), "gru.bias_hh_l0": (24,), "head.weight": (150, 8), "head.bias": (150,)})
    require(type(state) in (dict, __import__("collections").OrderedDict) and set(state) == set(shapes), "checkpoint class schema")
    result = {}
    for key, shape in shapes.items():
        value = state[key]
        require(type(value) is torch.Tensor and value.device.type == "cpu" and value.dtype == torch.float32
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "checkpoint tensor " + key)
        result[key] = value.numpy().copy()
        if initial and key.startswith(("head.", "network.2.")):
            require(np.count_nonzero(result[key]) == 0, "zero initial output head")
    return result


def aggregate(rows):
    families, checks, comparisons = {}, [], {}
    for partition in ("train", "test"):
        families[partition] = {}
        for variant in VARIANTS:
            endpoint_results = {}
            for endpoint in ENDPOINTS:
                metrics = [rows[f"fold-{f}-{s}"][variant][partition][endpoint] for s in SEEDS for f in FOLDS]
                parents = sorted({p for m in metrics for p in m["parent_ids"]})
                require(len(parents) == (18 if partition == "train" else 12), "partition parent count")
                parent_mse = [float(np.mean([m["parent_mse"][m["parent_ids"].index(p)] for m in metrics if p in m["parent_ids"]])) for p in parents]
                mean = float(np.mean([m["mse"] for m in metrics]))
                endpoint_results[endpoint] = {"mse": mean, "rmse": math.sqrt(mean), "parent_ids": parents,
                    "parent_mse": parent_mse, "horizon_mse": np.mean([m["horizon_mse"] for m in metrics], axis=0).tolist()}
            families[partition][variant] = endpoint_results
    for control in (v for v in VARIANTS if v != "recurrent"):
        comparisons[control] = {}
        for endpoint in ENDPOINTS:
            c, b = families["test"]["recurrent"][endpoint], families["test"][control][endpoint]
            prefix = f"{control}/{endpoint}"
            checks.append({"name": prefix+"/mean", "passed": b["mse"] > 0 and Decimal(str(c["mse"])) <= Decimal(".81")*Decimal(str(b["mse"])),
                           "candidate_mse": c["mse"], "control_mse": b["mse"]})
            paired = []
            for seed in SEEDS:
                for fold in FOLDS:
                    a = rows[f"fold-{fold}-{seed}"]["recurrent"]["test"][endpoint]["mse"]
                    z = rows[f"fold-{fold}-{seed}"][control]["test"][endpoint]["mse"]
                    item = {"name": prefix+f"/fold-{fold}-{seed}", "passed": a <= z, "candidate_mse": a, "control_mse": z}
                    checks.append(item); paired.append(item)
            require(c["parent_ids"] == b["parent_ids"], "paired parent identity")
            cp, bp = np.asarray(c["parent_mse"]), np.asarray(b["parent_mse"])
            count = int(np.sum(cp <= bp))
            checks.append({"name": prefix+"/parents", "passed": count >= 10, "parents_nonworse": count})
            for index, parent in enumerate(c["parent_ids"]):
                a, z = float(np.delete(cp, index).mean()), float(np.delete(bp, index).mean())
                checks.append({"name": prefix+f"/leave-out-{parent}", "passed": a < z, "candidate_mse": a, "control_mse": z})
            comparisons[control][endpoint] = {"paired": paired, "parents_nonworse": count,
                "rmse_improvement_percent": 100*(1-c["rmse"]/b["rmse"]) if b["rmse"] else None}
    require(len(checks) == 368, "368 screening comparisons")
    return families, comparisons, {"passed": all(x["passed"] for x in checks), "checks_passed": sum(x["passed"] for x in checks),
        "total_checks": 368, "checks": checks, "scope": "training-parent held-out mechanism screen; not the old external-panel gate"}


def audit(experiment, out, *, protocol_sha256, completed_sha256, root=ROOT):
    experiment, out, root = Path(experiment), Path(out), Path(root)
    require(not out.resolve().is_relative_to(experiment.resolve()), "audit output inside experiment")
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        protocol = validate_inputs(experiment, protocol_sha256, root)
        run = experiment/"run-01"
        before = members(run)
        require(set(before) == expected_members() and before["completed.json"]["sha256"] == completed_sha256,
                "exact380 execution files/external completion pin")
        done = read_json(run/"completed.json")
        require(done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256
                and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}
                and done["optimizer_updates"] == 9000 and done["external_panel_calls"] == 0
                and done["frozen_backbone_batch_calls"] == 9
                and len(done["fits"]) == 45 and len(done["rows"]) == 81 and len(done["caches"]) == 9,
                "completed full fixed execution")
        require(read_json(run/"started.json")["protocol_sha256"] == protocol_sha256, "start binding")
        original = original_training(root)
        rows, fits, cache_checks, saved_errors, initial_by_seed = {}, {}, {}, {}, {}
        fit_index = row_index = cache_index = 0
        for seed in SEEDS:
            for fold in FOLDS:
                stem = f"fold-{fold}-{seed}"
                c = arrays(run/(stem+"-cache.npz"))
                cache_checks[stem] = validate_cache(c, fold, seed, original, root)
                record = done["caches"][cache_index]; cache_index += 1
                require(record["seed"] == seed and record["fold"] == fold and record["cache_sha256"] == sha(run/(stem+"-cache.npz")), "cache record")
                finite(record["seconds"], "cache seconds", True)
                orders = np.load(run/(stem+"-orders.npy"), allow_pickle=False)
                train_indices = np.flatnonzero(c["train_mask"])
                require(orders.dtype == np.int64 and orders.shape == (40, 144)
                        and np.array_equal(np.sort(orders, axis=1), np.broadcast_to(train_indices, orders.shape)), "40 complete shared training permutations")
                for variant in NEURAL:
                    fs = stem+"-"+variant
                    fit = read_json(run/(fs+"-fit.json"))
                    require(fit == done["fits"][fit_index] and (fit["seed"], fit["fold"], fit["variant"]) == (seed, fold, variant)
                            and fit["updates"] == 200 and fit["parameters"] == (2994 if variant == "summary" else 2910)
                            and fit["initial_sha256"] == sha(run/(fs+"-initial.pt")) and fit["final_sha256"] == sha(run/(fs+".pt")), "fit identity/coverage")
                    fit_index += 1
                    initial = checkpoint(run/(fs+"-initial.pt"), variant, True)
                    checkpoint(run/(fs+".pt"), variant)
                    key = (seed, "summary" if variant == "summary" else "recurrent")
                    if key in initial_by_seed:
                        require(all(np.array_equal(x, initial_by_seed[key][k]) for k, x in initial.items()), "paired same-seed initial tensors")
                    else:
                        initial_by_seed[key] = initial
                    losses = np.load(run/(fs+"-losses.npy"), allow_pickle=False)
                    tensor(losses, (200,), "all losses", np.float64)
                    require((losses >= 0).all() and losses[0] == fit["first_loss"] and losses[-1] == fit["last_loss"], "loss boundary")
                    finite(fit["seconds"], "fit wall", True)
                    for name, count in (("head_latency_ms", 12), ("warmup_ms", 3)):
                        require(len(fit[name]) == count, "head timing count")
                        for value in fit[name]:
                            finite(value, "head timing", True)
                    fits[fs] = fit
                rows[stem] = {}
                for variant in VARIANTS:
                    prefix = stem+"-"+variant
                    payload = arrays(run/(prefix+"-predictions.npz"))
                    require(set(payload) == {"p", "R", "correction"}, "prediction schema")
                    pp, rr, correction = payload["p"], payload["R"], payload["correction"]
                    tensor(pp, (240, 25, 3), "predicted position"); tensor(rr, (240, 25, 3, 3), "predicted rotation")
                    tensor(correction, (240, 25, 6), "correction")
                    xp, xr = corrected(c["base_p"], c["base_R"], c["root_R"], correction)
                    require(np.allclose(pp, xp, rtol=2e-6, atol=2e-6) and np.allclose(rr, xr, rtol=2e-6, atol=2e-6), "root-frame correction reconstruction")
                    if variant == "base":
                        require(np.count_nonzero(correction) == 0 and np.array_equal(pp, c["base_p"])
                                and np.array_equal(rr, c["base_R"]), "exact unchanged base")
                    elif variant == "bias":
                        expected = c["one_errors"].astype(float).sum(1)[:, None]/32*np.arange(1, 26)[None, :, None]
                        require(np.allclose(correction, expected, rtol=2e-5, atol=2e-6), "fixed past-only shrinkage bias")
                    elif variant.startswith("ridge_"):
                        expected, witness = ridge_check(arrays(run/(prefix+"-weights.npz")), c["tokens"], c["target"], c["train_mask"], variant)
                        require(np.allclose(correction, expected, rtol=2e-5, atol=2e-6), "ridge saved prediction arithmetic")
                        cache_checks[stem][variant+"_normal_residual"] = witness
                    values = errors(pp, rr, c["target_p"], c["target_R"])
                    row = read_json(run/(prefix+"-evaluation.json"))
                    require(row == done["rows"][row_index] and (row["seed"], row["fold"], row["variant"]) == (seed, fold, variant), "row identity")
                    row_index += 1
                    metrics = {}
                    for partition in ("train", "test"):
                        mask = c[partition+"_mask"]
                        metrics[partition] = reduce_errors({k: v[mask] for k, v in values.items()}, c["ids"][mask])
                        validate_metrics(row[partition], metrics[partition])
                    rows[stem][variant] = metrics
                    for endpoint, value in values.items():
                        saved_errors[prefix+"__"+endpoint] = value
        families, comparisons, screen = aggregate(rows)
        wall = finite(done["wall_seconds"], "execution wall", True)
        fit_seconds = sum(finite(v["seconds"], "fit wall", True) for v in fits.values())
        require(fit_seconds <= wall+1e-6, "fit/whole nested timing")
        summary = {"status": "completed", "study": "pose-innovation-v1", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "variants": list(VARIANTS), "rows": rows,
            "families": families, "comparisons": comparisons, "screen": screen, "fits": fits, "cache_checks": cache_checks,
            "counts": {"rows": 81, "caches": 9, "fits": 45, "optimizer_updates": 9000, "test_windows": 864,
                "train_windows": 1296, "test_parents": 12, "train_parents": 18, "execution_files": 380,
                "neural_checkpoint_reads": 90, "external_panel_calls": 0},
            "costs": {"execution_wall_seconds": wall, "fit_wall_seconds": fit_seconds,
                "cache_seconds": sum(v["seconds"] for v in done["caches"]),
                "scope": "Nested measured components, not additive; cached-head timings exclude backbone and preprocessing."},
            "limits": ["Training-parent development screen only, not old external-panel qualification or new architecture efficacy.",
                "Saved neural outputs and gradients/optimizer updates are source/test-bound, not independently replayed.",
                "Orders/permutations are checked for exact legal membership and pairing; random seed generation is not rerun.",
                "Ridge normal equations and saved predictions are verified without another fit or solve.",
                "Token arithmetic uses explicit scale-dependent floating-point bounds; causal pre-assimilation provenance remains source/test-bound.",
                "Windows, folds and seeds are correlated; 368 conjunctive checks are not independent significance tests."]}
        write_json(out/"summary.json", summary)
        with (out/"window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **saved_errors)
        text = (f"# Training-parent error-order screen\n\nScreen **{'PASS' if screen['passed'] else 'FAIL'}**: "
                f"{screen['checks_passed']}/368 checks. All 81 rows, 45 fits and 864 held-out window predictions retained.\n\n"
                "| Method | Test position RMSE (m) | Test rotation RMSE (rad) | Train position RMSE (m) | Train rotation RMSE (rad) |\n"
                "|---|---:|---:|---:|---:|\n")
        for variant in VARIANTS:
            t, v = families["train"][variant], families["test"][variant]
            text += f"| {variant} | {v['position']['rmse']:.7g} | {v['rotation']['rmse']:.7g} | {t['position']['rmse']:.7g} | {t['rotation']['rmse']:.7g} |\n"
        text += "\n"+"\n\n".join(summary["limits"])+"\n"
        with (out/"README.md").open("x") as stream:
            stream.write(text)
        require(members(run) == before, "execution changed during audit")
        validate_inputs(experiment, protocol_sha256, root)
        receipt = {"status": "completed", "study": "pose-innovation-v1", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "execution_members": before, "sources": protocol["sources"],
            "files": members(out), "qualification_passed": screen["passed"], "checks_passed": screen["checks_passed"],
            "total_checks": 368, "auditor_sha256": sha(__file__), "new_model_calls": 0, "new_native_calls": 0,
            "new_optimizer_calls": 0, "new_ridge_fits": 0, "new_random_draws": 0, "wall_seconds": time.perf_counter()-start}
        write_json(out/"receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out/"failed.json", {"status": "failed", "error": repr(error), "wall_seconds": time.perf_counter()-start})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note(f"Could not preserve audit failure: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256, completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "checks_passed", "total_checks")}))
