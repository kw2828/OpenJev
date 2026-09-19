"""Independent saved-tensor development audit; no model, trainer or native calls.

This authenticates recorded provenance and recomputes prediction arithmetic.
It does not replay gradients, Adam transitions or neural predictions, and saved
timestamps cannot independently prove chronological execution. Completed source
and phase bindings support those narrower provenance statements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

VARIANTS = ("constant", "age", "raw", "normalized")
PANELS, STAGES = ("dev6", "dev10"), ("initial", "final")
FIT_FILES = {"started.json", "initial-weights.pt", "epoch-orders.pt", "training.jsonl", "weights.pt", "checkpoint.pt"}
EVAL_FILES = {"started.json", "predictions.pt", "summary.json"}
ROOT_FILES = {"started.json", "all-fits-completed.json", "evaluation-started.json", "evaluation-index.json"}
RECIPE = {"learning_rate": .001, "gradient_clip": {"backbone": 10., "variance_head": 10.},
          "rollout_horizon": 5, "rollout_weight": .5, "reward_scale": 4., "torch_threads": 1}
ADAM = {"betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
        "foreach": False, "fused": False, "maximize": False, "capturable": False,
        "differentiable": False, "decoupled_weight_decay": False}
SELECTION = {"primary": "post_reacquisition_mean_mse", "normalized_recovery_ratio": .97,
    "paired_nonworse": True, "constant_family_nonworse": True, "angle_regression_ratio": 1.02,
    "reward_regression_ratio": 1.05, "ordinary_initial_recovery_ratio": .9,
    "confirm_age_table_before_learned_uncertainty_claim": True}
BASE_CONFIG = {"hidden_size": 64, "context_size": 16, "dt": .02, "noise_std": .05,
    "eta": .1, "variance_min": 1e-4, "variance_max": 4., "epochs": 8, "batch_size": 32,
    "variance_score_weight": .1}
CORPUS_HASHES = {"npz_sha256": "bcf12a01b521ea677871ab7686b6c05913c6123bbab77e718e53090b16da6d08",
                 "json_sha256": "89a6a387f21affa72feca65510d7dd04cca0207dbee3d9f309e9023b4933a6a8"}

EXTRA_SOURCES = {
    *(f"src/openjev/research/reacher_innovation_{name}.py" for name in ("context", "loss", "pilot", "pilot_data")),
    *(f"tests/test_reacher_innovation_{name}.py" for name in ("context", "loss", "pilot", "pilot_data")),
    *("output/reacher-innovation-pilot-v1/" + name for name in (
        "run_pilot.py", "design.md", "test_run_pilot.py", "capacity.py", "capacity-review.json",
        "launch.py", "test_launch.py", "audit_pilot.py", "test_audit_pilot.py",
        "capacity-process-01/terminal.json", "capacity-process-02/terminal.json",
        "capacity-attempt-02/completed.json")),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved audit cap exceeded")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def child(root, name):
    path = root / name
    require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
            and path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "Safe relative member")
    return path


def tensor_hash(values):
    digest = hashlib.sha256(b"OpenJev named tensors v1\0")
    for name in sorted(values):
        value = values[name]
        require(isinstance(name, str) and isinstance(value, torch.Tensor) and value.layout == torch.strided
                and value.device.type == "cpu" and bool(torch.isfinite(value).all()), "Finite CPU named tensor")
        meta = json.dumps([name, str(value.dtype), list(value.shape)], ensure_ascii=False, separators=(",", ":")).encode()
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(meta)) + meta + struct.pack(">Q", len(raw)) + raw)
    return digest.hexdigest()


def state_hash(value):
    def describe(item):
        if isinstance(item, torch.Tensor):
            return ["tensor", tensor_hash({"value": item})]
        if isinstance(item, dict):
            return ["dict", [[type(k).__name__, k, describe(item[k])]
                for k in sorted(item, key=lambda k: (type(k).__name__, str(k)))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [describe(x) for x in item]]
        require(item is None or type(item) in (str, bool, int, float), "Plain metadata scalar")
        require(type(item) is not float or math.isfinite(item), "Finite metadata scalar")
        return [type(item).__name__, item]
    return hashlib.sha256(json.dumps(describe(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def inventory(folder, deadline):
    result = {}
    for path in sorted(folder.rglob("*")):
        check(deadline)
        require(not path.is_symlink(), "Symlink not admitted")
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    check(deadline)
    return result


def bound_tree(folder, receipt, expected, deadline, *, member_key="files"):
    actual = inventory(folder, deadline)
    require(set(receipt[member_key]) == expected and set(actual) == expected | {"completed.json"}, "Exact completed file membership")
    require(all(actual[k] == v for k, v in receipt[member_key].items()), "Completed member hashes and sizes")
    return actual


def close(actual, expected, label):
    if expected is None:
        require(actual is None, label + " absent target must remain null")
    else:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=3e-6, abs_tol=3e-8), label + " arithmetic mismatch")


def array(value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and tuple(value.shape) == tuple(shape)
            and value.dtype == dtype and not value.requires_grad and bool(torch.isfinite(value).all()), label + " tensor schema")
    return value.numpy()


def parameter_shapes(config):
    h, c = config["hidden_size"], config["context_size"]
    result = {}
    for prefix, width in (("observation_update", 8), ("transition", 6 + c)):
        result.update({prefix + ".weight_ih": (3 * h, width), prefix + ".weight_hh": (3 * h, h),
                       prefix + ".bias_ih": (3 * h,), prefix + ".bias_hh": (3 * h,)})
    for prefix in ("observation_head", "variance_head"):
        result.update({prefix + ".weight": (4, h), prefix + ".bias": (4,)})
    result.update({"reward_head.0.weight": (h, h + 2), "reward_head.0.bias": (h,),
        "reward_head.2.weight": (1, h), "reward_head.2.bias": (1,),
        "context_innovation.weight": (c, 4), "gate.weight": (1, 2), "gate.bias": (1,)})
    return result


def weights_schema(weights, config):
    shapes = parameter_shapes(config)
    require(set(weights) == set(shapes), "Exact innovation parameter set")
    for name, shape in shapes.items():
        array(weights[name], shape, torch.float32, name)
    return tensor_hash(weights)


def binding(actual, plan, variant):
    require(actual["version"] == "reacher-innovation-pilot-v1"
            and actual["config"] == plan["configurations"][variant]
            and json.loads(json.dumps(actual["recipe"])) == RECIPE
            and json.loads(json.dumps(actual["adam"])) == ADAM
            and actual["source_sha256"] == plan["source_sha256"] and actual["runtime"] == plan["runtime"]
            and actual["constructor_seed"] == 410 and actual["constructor_tensors_overwritten"] is True,
            "Exact fit/evaluation configuration, source and runtime")


def work_check(work, batch, steps):
    require(set(work) == {"modules", "expected_action_cost_samples", "completed_forward_only", "excludes"}
            and work["completed_forward_only"] is True, "Forward work receipt schema")
    expected = {}
    for name in ("observation_update", "context_innovation", "gate"):
        expected[name] = {"calls": steps, "samples": batch * steps}
    samples = batch * (steps + 5 * (steps - 4))
    for name in ("transition", "observation_head", "variance_head", "reward_head.0", "reward_head.2"):
        expected[name] = {"calls": steps + 5, "samples": samples}
    require(work["modules"] == expected and work["expected_action_cost_samples"] == samples, "All executed forward samples including auxiliary heads")


def audit_fit(folder, plan, pair, variant, train, initial, orders, deadline):
    done = read(folder / "completed.json")
    members = bound_tree(folder, done, FIT_FILES, deadline)
    binding(done, plan, variant)
    cfg = plan["configurations"][variant]
    episodes, steps = train["commands"].shape[:2]
    total = cfg["epochs"] * math.ceil(episodes / cfg["batch_size"])
    counts = {"attempted_updates": total, "optimizer_steps": total, "flushed_updates": total}
    require(done["status"] == "completed" and done["counts"] == counts and done["episodes"] == episodes
            and done["steps"] == steps and done["evaluation_performed"] is False
            and done["automatic_retry"] is False and done["resume_authorized"] is False
            and 0 < done["wall_seconds"] < plan["cap_seconds"], "Complete bounded fit")
    started = read(folder / "started.json")
    binding(started, plan, variant)
    for row in (done, started):
        require(row["initial_sha256"] == pair["initial_tensor_sha256"]
                and row["orders_sha256"] == pair["orders_tensor_sha256"]
                and row["data_sha256"] == plan["data_tensor_sha256"]["train"], "Paired original inputs")
    require(tensor_hash(load(folder / "initial-weights.pt")) == tensor_hash(initial)
            and torch.equal(load(folder / "epoch-orders.pt"), orders), "Copied initial/order identity")
    weights = load(folder / "weights.pt")
    require(weights_schema(weights, cfg) == done["final_weights_sha256"], "Final weight identity")
    checkpoint = load(folder / "checkpoint.pt")
    require(set(checkpoint) == {"version", "binding", "model_configuration", "weights", "optimizer", "parameter_names", "counts", "resume_authorized"}
            and checkpoint["counts"] == counts and checkpoint["resume_authorized"] is False
            and state_hash(checkpoint) == done["checkpoint_sha256"]
            and tensor_hash(checkpoint["weights"]) == done["final_weights_sha256"], "Exact saved checkpoint identity")
    binding(checkpoint["binding"], plan, variant)
    require(checkpoint["version"] == done["version"]
            and all(checkpoint["binding"][key] == done[key] for key in ("initial_sha256", "orders_sha256", "data_sha256")), "Checkpoint training input provenance")
    require(checkpoint["parameter_names"] == list(parameter_shapes(cfg)), "Original named parameter order")
    for name in ("variant", "hidden_size", "context_size", "dt", "noise_std", "eta", "variance_min", "variance_max"):
        require(checkpoint["model_configuration"][name] == cfg[name], "Actual model configuration: " + name)
    require(checkpoint["model_configuration"]["model_class"] == "InnovationContextWorldModel", "Declared actual model class")
    opt = checkpoint["optimizer"]
    require(set(opt) == {"state", "param_groups"} and len(opt["param_groups"]) == 1, "Single Adam group")
    group = opt["param_groups"][0]
    require(set(group) == set(ADAM) | {"params", "lr"}
            and group["params"] == list(range(len(checkpoint["parameter_names"]))) and group["lr"] == .001
            and all(json.loads(json.dumps(group[k])) == v for k, v in ADAM.items()), "Unchanged Adam recipe")
    require(set(opt["state"]) == set(group["params"]), "All Adam slots retained")
    for index, name in enumerate(checkpoint["parameter_names"]):
        saved = opt["state"][index]
        require(set(saved) == {"step", "exp_avg", "exp_avg_sq"}, "Exact Adam slot fields")
        require(float(array(saved["step"], (), torch.float32, "Adam step")) == total, "Every parameter Adam step")
        array(saved["exp_avg"], weights[name].shape, torch.float32, "Adam first moment")
        require((array(saved["exp_avg_sq"], weights[name].shape, torch.float32, "Adam second moment") >= 0).all(), "Nonnegative second moment")
    logs = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
    require(len(logs) == total, "Every optimizer update logged")
    elapsed = 0.
    for update, row in enumerate(logs):
        check(deadline)
        epoch, batch = divmod(update, math.ceil(episodes / cfg["batch_size"]))
        indices = orders[epoch, batch * cfg["batch_size"]:(batch + 1) * cfg["batch_size"]]
        require(row["epoch"] == epoch and row["batch"] == batch and row["update"] == update + 1
                and row["indices"] == indices.tolist() and row["indices_sha256"] == tensor_hash({"indices": indices}), "Exact full-epoch update order")
        metrics = row["metrics"]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()), "Finite recorded loss components")
        prediction = metrics["observation_mse"] + 4 * metrics["reward_mse"] + .5 * (metrics["rollout_observation_mse"] + 4 * metrics["rollout_reward_mse"])
        close(metrics["prediction_objective"], prediction, "Recorded prediction loss")
        close(metrics["loss"], prediction + cfg["variance_score_weight"] * metrics["residual_moment_score"], "Recorded total loss")
        valid = train["packets"][indices, :, 6]
        require(metrics["valid_observation_targets"] == metrics["valid_variance_targets"] == float(valid[:, 1:].sum())
                and metrics["valid_rollout_starts"] == float(valid[:, :steps - 4].sum()), "Public loss mask counts")
        require(row["gradient_clip"] == RECIPE["gradient_clip"] and set(row["gradient_norm_before_clip"]) == {"backbone", "variance_head"}, "Separate clipping groups")
        for name, norm in row["gradient_norm_before_clip"].items():
            require(type(norm) in (int, float) and math.isfinite(norm) and norm >= 0
                    and row["gradient_clipped"][name] is (norm > 10.), "Finite recorded norm and clip flag")
        require(row["fit_elapsed_seconds_before_log"] >= elapsed and row["update_seconds_before_log"] > 0
                and row["update_seconds_before_log"] <= row["fit_elapsed_seconds_before_log"] <= done["wall_seconds"], "Nested update timings")
        elapsed = row["fit_elapsed_seconds_before_log"]
        work_check(row["work"], len(indices), steps)
    return {"counts": counts, "wall_seconds": done["wall_seconds"], "final_weights_sha256": done["final_weights_sha256"],
            "completed_sha256": members["completed.json"]["sha256"], "receipt": done}


def prediction_metrics(pred, data, cfg):
    p, a, r = (data[k].numpy().astype(np.float64) for k in ("packets", "commands", "rewards"))
    n, steps = a.shape[:2]
    starts = steps - 4
    shapes = {"one_step_mean": (n, steps, 4), "one_step_variance": (n, steps, 4), "one_step_reward": (n, steps),
        "open_loop_mean": (n, starts, 5, 4), "open_loop_variance": (n, starts, 5, 4), "open_loop_reward": (n, starts, 5),
        "one_step_target_valid": (n, steps), "reacquisition_prior_mask": (n, steps),
        "post_reacquisition_root_mask": (n, starts), "post_reacquisition_target_valid": (n, starts, 2),
        "post_reacquisition_squared_error": (n, starts, 2)}
    bools = {"one_step_target_valid", "reacquisition_prior_mask", "post_reacquisition_root_mask", "post_reacquisition_target_valid"}
    require(set(pred) == set(shapes), "Complete prediction and mask tensors")
    arrays = {key: array(pred[key], shape, torch.bool if key in bools else torch.float32, key) for key, shape in shapes.items()}
    for key in ("one_step_variance", "open_loop_variance"):
        require(((arrays[key] >= cfg["variance_min"]) & (arrays[key] <= cfg["variance_max"])).all(), "Bounded predicted moments")
    visible = p[:, 1:, 6] == 1
    prior_return = visible & (p[:, :-1, 6] == 0)
    roots = p[:, :starts, 6] == 1
    targets = np.stack([p[:, t + 1:t + starts + 1] for t in range(5)], 2)
    target_rewards = np.stack([r[:, t:t + starts] for t in range(5)], 2)
    recovery = np.zeros_like(roots)
    recovery[:, 1:] = roots[:, 1:] & (p[:, :starts - 1, 6] == 0)
    valid_recovery = recovery[..., None] & (targets[:, :, [0, 2], 6] == 1)
    err = ((arrays["one_step_mean"].astype(np.float64) - p[:, 1:, :4]) ** 2).mean(-1)
    future_err = ((arrays["open_loop_mean"].astype(np.float64) - targets[..., :4]) ** 2).mean(-1)
    recovery_error = future_err[:, :, [0, 2]]
    for key, expected in (("one_step_target_valid", visible), ("reacquisition_prior_mask", prior_return),
            ("post_reacquisition_root_mask", recovery), ("post_reacquisition_target_valid", valid_recovery)):
        require(np.array_equal(arrays[key], expected), "Independent public mask: " + key)
    require(np.allclose(arrays["post_reacquisition_squared_error"], recovery_error, rtol=3e-6, atol=3e-8), "Saved recovery errors")
    result = []
    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None
    for i in range(n):
        complete = valid_recovery[i].all(-1)
        require(prior_return[i].sum() == complete.sum() == 2, "Exactly two fully covered recovery roots per episode")
        variance = arrays["one_step_variance"][i].astype(np.float64)
        error4 = (arrays["one_step_mean"][i].astype(np.float64) - p[i, 1:, :4]) ** 2
        mask = np.broadcast_to(visible[i, :, None], variance.shape)
        count = int(visible[i].sum())
        result.append({"episode": i, "one_step_angle_mse": mean(err[i], visible[i]),
            "one_step_reward_mse": float(((arrays["one_step_reward"][i].astype(np.float64) - r[i]) ** 2).mean()),
            "reacquisition_prior_angle_mse": mean(err[i], prior_return[i]),
            "post_reacquisition_h1_mse": mean(recovery_error[i, :, 0], valid_recovery[i, :, 0]),
            "post_reacquisition_h3_mse": mean(recovery_error[i, :, 1], valid_recovery[i, :, 1]),
            "post_reacquisition_mean_mse": float(recovery_error[i, complete].mean()),
            "post_reacquisition_complete_two_roots": True,
            "open_loop_angle_mse": mean(future_err[i], (targets[i, ..., 6] == 1) & roots[i, :, None]),
            "open_loop_reward_mse": mean((arrays["open_loop_reward"][i].astype(np.float64) - target_rewards[i]) ** 2,
                                          np.broadcast_to(roots[i, :, None], (starts, 5))),
            "residual_moment_score": float((.5 * (np.log(variance) + error4 / variance).sum(-1))[visible[i]].mean()),
            "variance_lower_bound_fraction": float((variance[mask] <= float(np.float32(1.01 * cfg["variance_min"]))).mean()),
            "variance_upper_bound_fraction": float((variance[mask] >= float(np.float32(.99 * cfg["variance_max"]))).mean()),
            "mean_predicted_residual_variance": float(variance[mask].mean()),
            "counts": {"one_step_actions": steps, "visible_targets": count, "reacquisition_targets": 2,
                "open_loop_windows": starts, "post_reacquisition_roots": int(recovery[i].sum()),
                "post_reacquisition_complete_roots": 2, "post_reacquisition_h1_targets": 2, "post_reacquisition_h3_targets": 2,
                "visible_open_loop_roots": int(roots[i].sum()),
                "visible_open_loop_targets": int(((targets[i, ..., 6] == 1) & roots[i, :, None]).sum()),
                "open_loop_reward_targets": int(roots[i].sum()) * 5, "moment_coordinate_targets": count * 4}})
    return result


def compare_rows(recorded, actual):
    require(len(recorded) == len(actual), "Every development episode retained")
    for saved, computed in zip(recorded, actual, strict=True):
        require(set(saved) == set(computed), "Per-episode metric schema")
        for key, value in computed.items():
            if key in {"episode", "counts", "post_reacquisition_complete_two_roots"}:
                require(saved[key] == value, "Episode/mask count identity")
            else:
                close(saved[key], value, key)


def criteria(evaluations):
    require(set(evaluations) == {(v, p, s, panel) for v in VARIANTS for p in range(3) for s in STAGES for panel in PANELS}, "All48 evaluations")
    metric_names = [k for k, v in next(iter(evaluations.values()))[0].items() if type(v) in (int, float) and k != "episode"]
    fit_means = {key: {m: float(np.mean([r[m] for r in rows])) for m in metric_names} for key, rows in evaluations.items()}
    family = {(v, stage, panel): {m: float(np.mean([fit_means[v, p, stage, panel][m] for p in range(3)])) for m in metric_names}
              for v in VARIANTS for stage in STAGES for panel in PANELS}
    checks = []
    def add(name, left, right, ratio):
        require(all(math.isfinite(x) and x >= 0 for x in (left, right)), "Finite nonnegative criterion inputs")
        checks.append({"name": name, "left": left, "control": right, "ratio": ratio,
                       "threshold": ratio * right, "comparison": "<=", "passed": left <= ratio * right})
    primary = "post_reacquisition_mean_mse"
    for panel in PANELS:
        for control in ("age", "raw"):
            add(f"{panel}/normalized/{control}/family_recovery", family["normalized", "final", panel][primary], family[control, "final", panel][primary], .97)
            for pair in range(3):
                add(f"{panel}/normalized/{control}/pair{pair}_recovery", fit_means["normalized", pair, "final", panel][primary], fit_means[control, pair, "final", panel][primary], 1.)
        add(f"{panel}/normalized/constant/family_recovery", family["normalized", "final", panel][primary], family["constant", "final", panel][primary], 1.)
        for control in ("age", "raw"):
            for metric, ratio in (("one_step_angle_mse", 1.02), ("one_step_reward_mse", 1.05)):
                add(f"{panel}/normalized/{control}/{metric}", family["normalized", "final", panel][metric], family[control, "final", panel][metric], ratio)
    for pair in range(3):
        add(f"dev6/normalized/pair{pair}/initial_learning", fit_means["normalized", pair, "final", "dev6"][primary], fit_means["normalized", pair, "initial", "dev6"][primary], .9)
    require(len(checks) == 29, "Exactly29 predeclared checks")
    return {"fit_means": {"/".join(map(str, k)): v for k, v in fit_means.items()},
            "family_means": {"/".join(k): v for k, v in family.items()},
            "continuation": {"passed": all(r["passed"] for r in checks), "checks_passed": sum(r["passed"] for r in checks),
                             "total_checks": 29, "checks": checks}}


def public_splits(public, split):
    require(split["source_hashes"] == CORPUS_HASHES and split["rng_calls"] == split["model_calls"] == split["native_calls"] == 0,
            "Public fixed-corpus split provenance")
    train_rows = split["partitions"]["train"]["episodes"]
    dev_rows = split["partitions"]["development"]["episodes"]
    require(len(train_rows) == 640 and len(dev_rows) == 128, "Whole episode partition sizes")
    all_rows = sorted(train_rows + dev_rows, key=lambda row: row["original_index"])
    require([row["original_index"] for row in all_rows] == list(range(768)), "No overlapping or omitted episode")
    for row in all_rows:
        i = row["original_index"]
        require(row["reset_seed"] == 64100001 + i and row["noise_seed"] == 64500001 + i
                and row["action_seed"] == 64600001 + i and row["schedule_seed"] == 64400001 + i
                and type(row["phase"]) is int and 0 <= row["phase"] < 4
                and row["collector_policy"] in {"ik_pd", "random_high", "random_low"}, "Original episode/phase/policy identities")
        raw = json.dumps(["OpenJev/reacher-innovation-pilot-v1/development", CORPUS_HASHES["npz_sha256"],
            CORPUS_HASHES["json_sha256"], row["reset_seed"]], ensure_ascii=True, separators=(",", ":")).encode("ascii")
        require(row["selection_sha256"] == hashlib.sha256(raw).hexdigest(), "Canonical split rank")
    selected = []
    for phase, quotas in enumerate(((17, 7, 8), (16, 7, 9), (15, 9, 8), (17, 7, 8))):
        for policy, quota in zip(("ik_pd", "random_high", "random_low"), quotas, strict=True):
            candidates = sorted([r for r in all_rows if r["phase"] == phase and r["collector_policy"] == policy],
                key=lambda r: (r["selection_sha256"], r["original_index"]))
            require(len(candidates) >= quota, "Every predeclared split stratum")
            selected.extend(r["original_index"] for r in candidates[:quota])
    require([r["original_index"] for r in dev_rows] == sorted(selected)
            and [r["original_index"] for r in train_rows] == [i for i in range(768) if i not in set(selected)], "Independent fixed quota selection")
    for name, rows in (("train", train_rows), ("dev6", dev_rows), ("dev10", dev_rows)):
        data = public[name]
        p, a = data["packets"].numpy(), data["commands"].numpy()
        require((np.abs(a) <= 1).all(), "Recorded clipped commands")
        require(np.array_equal(p[..., 4:6], np.broadcast_to(p[:, :1, 4:6], p[..., 4:6].shape)), "Static public target")
        valid = np.ones((len(rows), 51), dtype=np.bool_)
        for i, row in enumerate(rows):
            for start in (8 + row["phase"], 28 + row["phase"]):
                valid[i, start:start + (10 if name == "dev10" else 6)] = False
        require(np.array_equal(p[..., 6], valid) and (p[..., :4][~valid] == 0).all(), "Exact public sensing schedule")
        times = np.arange(51)[None]
        last = np.maximum.accumulate(np.where(valid, times, -1), 1)
        require((p[..., 7][valid] == 0).all() and np.allclose(p[..., 7], (times - last) * .02, rtol=1e-5, atol=1e-6), "Public integer-derived ages")
    six, ten = public["dev6"], public["dev10"]
    require(torch.equal(six["commands"], ten["commands"]) and torch.equal(six["rewards"], ten["rewards"]), "Same recorded commands/rewards")
    keep = ten["packets"][..., 6] == 1
    require(torch.equal(six["packets"][..., :4][keep], ten["packets"][..., :4][keep])
            and torch.equal(six["packets"][..., 4:6], ten["packets"][..., 4:6]), "Ten-gap view cannot fill hidden angles or change targets")


def validate_plan(plan, protocol, root, deadline):
    require(plan["schema"] == "reacher-innovation-development-pilot-v1" and plan["status"] == "frozen_before_training"
            and plan["development_only"] is True and plan["no_retry"] is True and plan["native_control"] is False
            and plan["untouched_test_performance"] is False and plan["fit_count"] == 12
            and plan["updates_per_fit"] == 160 and plan["total_updates"] == 1920
            and plan["all_fits_before_evaluation"] is True and plan["variants"] == list(VARIANTS)
            and plan["panels"] == list(PANELS) and plan["evaluations"] == list(STAGES)
            and plan["selection"] == SELECTION, "Exact prospective development design")
    require(type(plan["cap_seconds"]) is int and plan["cap_seconds"] == 1800, "Frozen finite execution cap")
    require(plan["configurations"] == {v: {"variant": v, **BASE_CONFIG} for v in VARIANTS}, "All four fixed configurations")
    current = {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
               "numpy": np.__version__, "dtype": "torch.float32", "device": "cpu", "torch_threads": 1}
    require(plan["runtime"] == current, "Same arithmetic runtime")
    parent = root / "evidence/reacher-two-observation-study-v1/protocol"
    require(sha(parent / "plan.json") == plan["parent_plan_sha256"]
            and sha(parent / "freeze.json") == plan["parent_freeze_sha256"], "Declared frozen parent identities")
    inherited = read(parent / "freeze.json")["source_sha256"]
    require(len(inherited) == 116 and all(plan["source_sha256"].get(k) == v for k, v in inherited.items()), "All116 inherited sources unchanged")
    require(set(plan["source_sha256"]) == set(inherited) | EXTRA_SOURCES, "Exact parent and pilot source closure")
    prepared = inventory(protocol, deadline)
    expected_prepared = {"started.json", "split.json", "train.pt", "dev6.pt", "dev10.pt"}
    expected_prepared |= {f"{kind}-pair{p}.pt" for p in range(3) for kind in ("initial", "orders")}
    expected_prepared |= {"source-snapshot/" + name for name in plan["source_sha256"]}
    require(set(plan["prepared_members"]) == expected_prepared, "Complete declared prepared/source artifacts")
    require(set(prepared) == set(plan["prepared_members"]) | {"plan.json"}, "Exact prepared membership")
    require(all(prepared[k] == v for k, v in plan["prepared_members"].items()), "Prepared byte identities")
    for name, digest in plan["source_sha256"].items():
        check(deadline)
        require(sha(child(root, name)) == digest and sha(child(protocol, "source-snapshot/" + name)) == digest, "Source and snapshot identity")
    require([pair["pair"] for pair in plan["pairs"]] == [0, 1, 2]
            and len({p["initial_tensor_sha256"] for p in plan["pairs"]}) == 3, "All3 distinct paired initializations")
    validate_registry(plan)
    return prepared


def validate_registry(plan):
    registry = plan["random_registry"]
    roles = {f"pair{p}/{kind}" for p in range(3) for kind in ("initialization", "orders")}
    require(set(registry) == roles and all(type(v) is int and 0 <= v < 2**32 for v in registry.values())
            and len(set(registry.values())) == 6, "Six distinct named registry roles")
    for pair in plan["pairs"]:
        p = pair["pair"]
        require(pair["initialization_seed"] == registry[f"pair{p}/initialization"]
                and pair["orders_seed"] == registry[f"pair{p}/orders"], "Every pair bound to registry roles")
        for field in ("torch_initial_rng_sha256", "torch_final_rng_sha256"):
            digest = pair[field]
            require(isinstance(digest, str) and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest), "Recorded initializer RNG identity")


def audit(plan_path, expected_plan_sha256, execution, expected_completed_sha256, out, *, root, audit_cap_seconds):
    begin, out, execution, root = time.monotonic(), Path(out).resolve(), Path(execution).resolve(), Path(root).resolve()
    protocol, plan_path = Path(plan_path).resolve().parent, Path(plan_path).resolve()
    require(not out.is_relative_to(execution) and not out.is_relative_to(protocol), "Audit output cannot change input trees")
    out.mkdir(parents=True, exist_ok=False)
    phase = "authenticate"
    try:
        require(type(audit_cap_seconds) in (float, int) and math.isfinite(audit_cap_seconds) and audit_cap_seconds > 0, "Explicit finite audit cap")
        deadline = begin + audit_cap_seconds
        require(sha(plan_path) == expected_plan_sha256 and sha(execution / "completed.json") == expected_completed_sha256, "External plan/completion hashes")
        plan, completed = read(plan_path), read(execution / "completed.json")
        prepared = validate_plan(plan, protocol, root, deadline)
        expected = set(ROOT_FILES)
        for pair in range(3):
            for variant in VARIANTS:
                name = f"pair{pair}-{variant}"
                expected |= {f"fits/{name}/{file}" for file in FIT_FILES | {"completed.json"}}
                for stage in STAGES:
                    for panel in PANELS:
                        expected |= {f"evaluation/{name}/{stage}/{panel}/{file}" for file in EVAL_FILES | {"completed.json"}}
        members = bound_tree(execution, completed, expected, deadline, member_key="members")
        require(completed["plan_sha256"] == expected_plan_sha256 and completed["fits"] == 12
                and completed["evaluations"] == 48 and completed["development_only"] is True
                and completed["native_control_measured"] is False and 0 < completed["wall_seconds"] < plan["cap_seconds"], "Complete bounded pilot execution")
        started, boundary = read(execution / "started.json"), read(execution / "all-fits-completed.json")
        eval_start = read(execution / "evaluation-started.json")
        require(started["plan_sha256"] == expected_plan_sha256 and started["no_retry"] is True
                and eval_start["all_fits_completed_sha256"] == sha(execution / "all-fits-completed.json")
                and datetime.fromisoformat(started["utc"]) <= datetime.fromisoformat(boundary["utc"]) <= datetime.fromisoformat(eval_start["utc"]) <= datetime.fromisoformat(completed["utc"]), "Declared all-fits-before-evaluation boundary")
        phase = "prepared-tensors"
        public = {name: load(protocol / (name + ".pt")) for name in ("train", "dev6", "dev10")}
        for name, data in public.items():
            require(set(data) == {"packets", "commands", "rewards"} and tensor_hash(data) == plan["data_tensor_sha256"][name], "Named public data identity")
            n = 640 if name == "train" else 128
            for key, shape in (("packets", (n, 51, 8)), ("commands", (n, 50, 2)), ("rewards", (n, 50))):
                array(data[key], shape, torch.float32, name + "/" + key)
        public_splits(public, read(protocol / "split.json"))
        require(torch.equal(public["dev6"]["commands"], public["dev10"]["commands"])
                and torch.equal(public["dev6"]["rewards"], public["dev10"]["rewards"]), "Same development trajectories/actions/rewards")
        fits, evaluations, reports, evaluation_costs = {}, {}, {}, {}
        for pair in plan["pairs"]:
            p = pair["pair"]
            initial, orders = load(protocol / f"initial-pair{p}.pt"), load(protocol / f"orders-pair{p}.pt")
            require(weights_schema(initial, BASE_CONFIG) == pair["initial_tensor_sha256"]
                    and tensor_hash({"orders": orders}) == pair["orders_tensor_sha256"], "Original pair tensors")
            array(orders, (8, 640), torch.int64, "Complete paired orders")
            require(torch.equal(orders.sort(1).values, torch.arange(640).expand(8, -1)), "Eight complete permutations")
            generator = np.random.default_rng(pair["orders_seed"])
            require(generator.bit_generator.state == pair["numpy_initial_rng"], "Historical order generator initial state")
            replayed = np.stack([generator.permutation(640) for _ in range(8)])
            require(np.array_equal(replayed, orders.numpy()) and generator.bit_generator.state == pair["numpy_final_rng"], "Historical supplied-order replay")
            for variant in VARIANTS:
                phase, name = "fits", f"pair{p}-{variant}"
                fits[name] = audit_fit(execution / "fits" / name, plan, pair, variant, public["train"], initial, orders, deadline)
                require(boundary["fits"][name] == {"completed_sha256": fits[name]["completed_sha256"], "receipt": fits[name]["receipt"]}, "Every fit sealed before evaluation")
                for stage in STAGES:
                    for panel in PANELS:
                        check(deadline)
                        phase = "evaluation"
                        folder = execution / "evaluation" / name / stage / panel
                        done, saved = read(folder / "completed.json"), read(folder / "summary.json")
                        bound_tree(folder, done, EVAL_FILES, deadline)
                        binding(done, plan, variant)
                        declared = read(folder / "started.json")
                        binding(declared, plan, variant)
                        weight_hash = pair["initial_tensor_sha256"] if stage == "initial" else fits[name]["final_weights_sha256"]
                        require(done["status"] == "completed" and done["split"] == "development"
                                and done["new_optimizer_steps"] == 0 and done["counts"] == {"optimizer_steps": 0, "completed_batches": 4}
                                and done["weights_sha256"] == declared["weights_sha256"] == weight_hash
                                and done["data_sha256"] == declared["data_sha256"] == plan["data_tensor_sha256"][panel]
                                and 0 < done["wall_seconds"] < completed["wall_seconds"], "Completed development provenance")
                        computed = prediction_metrics(load(folder / "predictions.pt"), public[panel], plan["configurations"][variant])
                        compare_rows(saved["per_episode"], computed)
                        require(saved["episodes"] == 128 and saved["steps"] == 50 and saved["horizon"] == 5
                                and saved["counts"] == done["counts"] and len(saved["work_by_batch"]) == 4, "All evaluation batches")
                        for work in saved["work_by_batch"]:
                            work_check(work, 32, 50)
                        evaluations[variant, p, stage, panel] = computed
                        reports[variant, p, stage, panel] = saved
                        evaluation_costs[f"{name}/{stage}/{panel}"] = done["wall_seconds"]
        require(set(boundary["fits"]) == set(fits), "Exactly12 fitted models")
        index = read(execution / "evaluation-index.json")
        keys = [(v, p, s, panel) for p in range(3) for v in VARIANTS for s in STAGES for panel in PANELS]
        require(len(index) == 48 and all(row == {"pair": p, "variant": v, "stage": s, "panel": panel, "summary": reports[v, p, s, panel]}
                for row, (v, p, s, panel) in zip(index, keys, strict=True)), "Complete ordered evaluation index")
        numbers = criteria(evaluations)
        phase = "exit-authentication"
        require(inventory(execution, deadline) == members and inventory(protocol, deadline) == prepared, "Input trees unchanged during audit")
        for name, digest in plan["source_sha256"].items():
            check(deadline)
            require(sha(child(root, name)) == digest, "Source remained unchanged")
        fit_seconds = sum(r["wall_seconds"] for r in fits.values())
        evaluation_seconds = sum(evaluation_costs.values())
        require(fit_seconds + evaluation_seconds < completed["wall_seconds"], "All sequential fit/evaluation costs within full execution")
        summary = {"status": "completed", "complete_evidence_passed": True, "development_only": True,
            "plan_sha256": expected_plan_sha256, "execution_completed_sha256": expected_completed_sha256,
            "fits": 12, "evaluations": 48, "episodes_per_evaluation": 128, "optimizer_updates": 1920,
            "per_episode": {"/".join(map(str, k)): v for k, v in evaluations.items()}, **numbers,
            "costs": {"preparation_wall_seconds": plan["preparation_wall_seconds"], "execution_wall_seconds": completed["wall_seconds"],
                "fit_wall_seconds_nested": fit_seconds, "evaluation_wall_seconds_nested": evaluation_seconds,
                "fit_wall_seconds_by_model": {name: row["wall_seconds"] for name, row in fits.items()},
                "evaluation_wall_seconds_by_row": evaluation_costs,
                "audit_wall_seconds_before_report": time.monotonic() - begin},
            "limits": ["Exposed development corpus and off-policy censoring stress; not native-control or fresh-test results.",
                "No neural inference, gradients or numerical optimizer transitions replayed; source/receipts bind those provenance claims.",
                "Initializer Torch generation and original-corpus-to-prepared tensor copying are bound to the frozen preparer, not independently replayed.",
                "Prediction metrics recomputed in float64 from saved float32 predictions; no tolerance added to29 gate thresholds.",
                "Recorded phase ordering is authenticated, not independently established by timestamps.",
                "Actual subprocess exits and external wall-clock termination require the separately authenticated launcher receipt.",
                "Forward counts omit backward/Adam/nonmodule work; whole execution time includes them. Shared-host timings are descriptive."],
            "new_model_calls": 0, "new_optimizer_steps": 0, "new_native_calls": 0}
        write(out / "summary.json", summary)
        receipt = {"status": "completed", "plan_sha256": expected_plan_sha256,
            "execution_completed_sha256": expected_completed_sha256, "source_sha256": plan["source_sha256"],
            "audit_source_sha256": sha(Path(__file__)), "execution_members": members, "prepared_members": prepared,
            "files": {"summary.json": {"sha256": sha(out / "summary.json"), "bytes": (out / "summary.json").stat().st_size}},
            "wall_seconds": time.monotonic() - begin, "audit_cap_seconds": audit_cap_seconds,
            "complete_evidence_passed": True, "qualification_passed": numbers["continuation"]["passed"],
            "checks_passed": numbers["continuation"]["checks_passed"], "total_checks": 29,
            "new_model_calls": 0, "new_optimizer_steps": 0, "new_native_calls": 0}
        write(out / "receipt.json", receipt)
        check(deadline)
        return summary
    except BaseException as error:
        error_text = repr(error)
        for action in (
            lambda: (out / "receipt.json").rename(out / "invalid-receipt.json") if (out / "receipt.json").exists() else None,
            lambda: write(out / "failed.json", {"status": "failed", "phase": phase, "error": error_text,
                "wall_seconds": time.monotonic() - begin, "complete_evidence_passed": False, "no_retry": True}),
        ):
            try:
                action()
            except BaseException as preservation_error:  # noqa: BLE001
                error.add_note(repr(preservation_error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-cap-seconds", type=float, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.plan_sha256, args.execution, args.completed_sha256, args.out,
                   root=args.root, audit_cap_seconds=args.audit_cap_seconds)
    print(json.dumps({"status": result["status"], "continuation": result["continuation"]}, allow_nan=False))
