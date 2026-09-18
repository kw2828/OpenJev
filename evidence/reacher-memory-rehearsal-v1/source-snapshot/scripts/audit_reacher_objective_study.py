"""Independent numerical audit of saved Reacher objective-study artifacts.

No learned model is instantiated or called. The training component and runner
are not imported. Native replay executes saved commands; CEM reconstruction
reads saved scalar scores. Saved gradients establish norm arithmetic, not
correctness of backprop. Shared protocol helpers are restricted to RNG/schema.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import struct
import sys
import time
from contextvars import ContextVar
from pathlib import Path

import audit_reacher_search_study as search_audit
import numpy as np
import torch

from openjev.research import reacher_objective_protocol as protocol
from openjev.research.reacher_random_streams import concrete_streams

base = search_audit.base
ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-objective-ablation-v1"
ARMS = ("anchor", "raw", "latent")
HORIZONS = (1, 3, 7)
CAPACITY_PATH = "evidence/reacher-objective-ablation-v1/engineering-capacity-v1.json"
CAPACITY_SHA = "3d77541ef843e531a66e7de55c89401ef9083a6bcbcdabca442d59f3f5cadb63"
TRAINING_TEST_TORCH_SEEDS = {
    "engineering/training-tests/public_data": 410,
    "engineering/training-tests/student/initial": 1121,
    "engineering/training-tests/predictor/initial": 1123,
    "engineering/training-tests/predictor/alternate": 1151,
    "engineering/training-tests/minibatch": 1129,
    **{f"engineering/training-tests/student/calibration/{index}": 1201 + index for index in range(3)},
    **{f"engineering/training-tests/predictor/calibration/{index}": 1301 + index for index in range(3)},
}
TRAINING_VERSION = "reacher-objective-training-v1"
TRAINING_FIELDS = ("hidden_size", "dt", "residual_reward", "noise_std", "train_episodes", "steps",
                   "epochs", "batch_size", "learning_rate", "gradient_clip", "rollout_horizon",
                   "rollout_weight", "reward_scale", "kl_weight", "kl_balance", "free_nats",
                   "auxiliary_horizons", "ema_momentum", "variance_weight", "covariance_weight")
_DEADLINE = ContextVar("objective_audit_deadline", default=float("inf"))


def check_budget():
    if time.monotonic() > _DEADLINE.get():
        raise TimeoutError("Frozen objective audit cap exceeded")


def canonical_tensor_hash(values):
    base.require(isinstance(values, dict) and all(isinstance(name, str) for name in values),
                 "Canonical named tensor mapping")
    digest = hashlib.sha256(b"OpenJev named tensors v1\0")
    for name in sorted(values):
        value = values[name]
        base.require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                     and (not value.is_floating_point() or torch.isfinite(value).all().item()),
                     "Canonical finite dense tensor")
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], ensure_ascii=False,
                              separators=(",", ":")).encode()
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(metadata)) + metadata + struct.pack(">Q", len(raw)) + raw)
    return digest.hexdigest()


def canonical_state_hash(value):
    def structure(item):
        if isinstance(item, torch.Tensor):
            return ["tensor", canonical_tensor_hash({"value": item})]
        if isinstance(item, dict):
            base.require(all(type(key) in (str, int) for key in item), "Checkpoint dictionary key type")
            return ["dict", [[type(key).__name__, key, structure(item[key])]
                             for key in sorted(item, key=lambda key: (type(key).__name__, str(key)))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [structure(part) for part in item]]
        base.require(item is None or type(item) in (bool, str, int, float), "Checkpoint scalar type")
        if type(item) is float:
            base.require(math.isfinite(item), "Checkpoint finite scalar")
        return [type(item).__name__, item]
    return hashlib.sha256(json.dumps(structure(value), ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def training_configuration(plan):
    return {**{key: plan[key] for key in TRAINING_FIELDS}, "optimizer": plan["optimizer"],
            "calibration": plan["calibration"], "dtype": "torch.float32", "device": "cpu",
            "model_class": "GRUResidualRewardWorldModel"}


def finite(value, label, *, positive=False):
    return base.finite_number(value, label, positive=positive)


def exact_integer(value, expected, label):
    base.require(type(value) is int and value == expected, label)


def checked(path, digest):
    check_budget()
    return search_audit.checked(Path(path), digest)


def read_tensor_file(path):
    path = Path(path)
    base.require(path.is_file() and not path.is_symlink(), "Tensor file identity")
    value = torch.load(path, map_location="cpu", weights_only=True)
    base.require(isinstance(value, dict), "Tensor checkpoint must be a dictionary")
    return value


def tensor_mapping(value, shapes, label, *, dtype=torch.float32):
    base.require(isinstance(value, dict) and set(value) == set(shapes), f"{label} tensor membership")
    for name, shape in shapes.items():
        item = value[name]
        base.require(isinstance(item, torch.Tensor) and tuple(item.shape) == tuple(shape)
                     and item.dtype == dtype and item.device.type == "cpu"
                     and torch.isfinite(item).all().item(), f"{label} tensor: {name}")
    return value


def student_shapes(plan):
    # Algebraic schema only, with no construction of a trainable module.
    return base.checkpoint_shapes({"width": 0, "stochastic_size": 0, **plan}, "gru")


def predictor_shapes(plan):
    hidden = plan["hidden_size"]
    return {"weight": (hidden, hidden), "bias": (hidden,)}


def reset_mask(schedules, steps):
    schedules = np.asarray(schedules)
    search_audit.array(schedules, (len(schedules), steps + 1), np.bool_, "Reset schedules")
    # Reset immediately before the last measured packet preceding each gap.
    return schedules[:, :steps] & ~schedules[:, 1:]


def audit_reset_events(path, records, *, enabled):
    packets = base.stack(records, "policy", "packets")
    n, points, width = packets.shape
    base.require(width == 8, "Public packet width")
    steps = points - 1
    schedules = np.asarray([record["metadata"]["sensor_schedule"] for record in records])
    wanted = reset_mask(schedules, steps) if enabled else np.zeros((n, steps), dtype=bool)
    saved = base.load_npz(Path(path), {"mask", "packets"})
    search_audit.array(saved["mask"], (n, steps), np.bool_, "Reset event mask")
    search_audit.array(saved["packets"], (n, steps, 8), np.float32, "Reset event packets")
    base.require(np.array_equal(saved["mask"], wanted), "Reset precursor times changed")
    current = packets[:, :steps].copy()
    current[..., :4] = np.where(current[..., 6:7] > .5, current[..., :4], 0)
    expected = np.where(wanted[..., None], current, 0)
    base.require(np.array_equal(saved["packets"], expected), "Reset public measurement identity")
    base.require(not wanted.any() or np.all(current[wanted, 6] == 1), "Reset lost current measurement")
    if enabled:
        base.require(np.all(wanted.sum(1) == 2), "Both scheduled gap precursors required")
    return {"events": int(wanted.sum()), "events_per_case": wanted.sum(1).tolist(),
            "steps_per_case": [np.flatnonzero(row).tolist() for row in wanted],
            "scope": "Recorded reset timing and current public packet; no neural state recomputation."}


def prediction_metrics(plan, records, values):
    n, steps = len(records), plan["steps"]
    horizons = tuple(plan.get("auxiliary_horizons", HORIZONS))
    expected_keys = {"one_step_angles", "one_step_rewards"}
    expected_keys.update(f"h{h}_{field}" for h in horizons for field in ("angles", "valid"))
    base.require(set(values) == expected_keys, "Held-out prediction artifact membership")
    one = search_audit.array(values["one_step_angles"], (n, steps, 4), np.float32,
                             "Held-out one-step angles")
    reward = search_audit.array(values["one_step_rewards"], (n, steps), np.float32,
                                "Held-out one-step rewards")
    packets = base.stack(records, "policy", "packets")
    native_angles = base.stack(records, "audit", "raw_obs")[:, :, :4]
    observed = packets[:, :, 6] > .5
    endpoints = {}
    for h in horizons:
        width = max(0, steps + 1 - h)
        predicted = search_audit.array(values[f"h{h}_angles"], (n, width, 4), np.float32,
                                       "Held-out endpoint angles")
        saved_mask = search_audit.array(values[f"h{h}_valid"], (n, width), np.bool_,
                                        "Held-out endpoint mask")
        mask = observed[:, :width] & observed[:, h:]
        base.require(np.array_equal(mask, saved_mask), "Held-out observed root/endpoint mask")
        if h == 1:
            base.require(np.array_equal(predicted, one), "One-step and horizon-one predictions differ")
        endpoints[str(h)] = base.masked_mse(predicted, packets[:, h:, :4], mask)
    reward_truth = base.stack(records, "audit", "rewards")
    return {
        "one_step_angles": base.masked_mse(one, packets[:, 1:, :4], observed[:, 1:]),
        "one_step_native_blackout_angles": base.masked_mse(one, native_angles[:, 1:], ~observed[:, 1:]),
        "one_step_reward_mse": float(np.mean((reward.astype(float) - reward_truth) ** 2)),
        "endpoints": endpoints,
        "scope": "Common held-out public history; native blackout angles are audit-only labels.",
    }


def control_qualification(plan, controls):
    pairs = tuple(plan.get("pairs", ("pair0", "pair1", "pair2")))
    expected = {f"{arm}-{pair}" for arm in ARMS for pair in pairs}
    for panel in ("full", "ordinary", "shift"):
        names = expected | set(search_audit.protocol.REFERENCES)
        if panel != "full":
            names |= {f"{name}-reset" for name in expected}
        base.require(set(controls[panel]) == names, "Complete objective control coverage")
    base.require(set(controls) == {"full", "ordinary", "shift"}, "Control panel membership")
    for rows in controls.values():
        for row in rows.values():
            values = np.asarray(row["episode_costs"], dtype=float)
            base.require(values.shape == (plan["control_episodes"],) and np.isfinite(values).all(),
                         "Paired control costs")
            base.require(math.isclose(float(values.mean()), row["mean_cost"], rel_tol=1e-12),
                         "Control mean arithmetic")

    def family(panel, arm, suffix=""):
        return np.mean([controls[panel][f"{arm}-{pair}{suffix}"]["episode_costs"] for pair in pairs], axis=0)

    checks, comparisons = [], {}

    def add(name, actual, threshold, direction):
        actual, threshold = finite(float(actual), name), finite(float(threshold), name)
        passed = {"le": actual <= threshold, "lt": actual < threshold,
                  "ge": actual >= threshold, "gt": actual > threshold}[direction]
        checks.append({"name": name, "actual": actual, "threshold": threshold,
                       "direction": direction, "passed": bool(passed)})

    for panel in ("ordinary", "shift"):
        latent = family(panel, "latent")
        comparisons[panel] = {}
        for comparator in ("anchor", "raw"):
            other = family(panel, comparator)
            add(f"latent/{panel}/mean_vs_{comparator}", latent.mean(), .95 * other.mean(), "le")
            comparisons[panel][f"latent_minus_{comparator}"] = (latent - other).tolist()
            for pair in pairs:
                add(f"latent-{pair}/{panel}/vs_{comparator}",
                    controls[panel][f"latent-{pair}"]["mean_cost"],
                    controls[panel][f"{comparator}-{pair}"]["mean_cost"], "le")
        reset = family(panel, "latent", "-reset")
        add(f"latent/{panel}/reset_effect", reset.mean(), 1.05 * latent.mean(), "ge")
        comparisons[panel]["latent_reset_minus_intact"] = (reset - latent).tolist()
        zero = controls[panel]["zero"]["mean_cost"]
        for pair in pairs:
            intact = controls[panel][f"latent-{pair}"]["mean_cost"]
            add(f"latent-{pair}/{panel}/reset_effect",
                controls[panel][f"latent-{pair}-reset"]["mean_cost"], intact, "gt")
            add(f"latent-{pair}/{panel}/vs_zero", intact, .9 * zero, "le")
    add("physics/ordinary/vs_zero", controls["ordinary"]["known_state"]["mean_cost"],
        .9 * controls["ordinary"]["zero"]["mean_cost"], "le")
    return {"passed": all(row["passed"] for row in checks), "checks": checks,
            "scope": "Development gate conditional on all saved fits; reset is sensitivity, not proof of useful memory."}, comparisons


def audit_ema_witness(witness, momentum, shapes):
    expected = {"teacher_before", "student_after_optimizer", "teacher_after_ema", "buffers"}
    base.require(set(witness) == expected, "EMA witness membership")
    before = tensor_mapping(witness["teacher_before"], shapes, "EMA teacher before")
    student = tensor_mapping(witness["student_after_optimizer"], shapes, "EMA student after optimizer")
    after = tensor_mapping(witness["teacher_after_ema"], shapes, "EMA teacher after")
    for name in shapes:
        computed = before[name].clone().mul_(momentum).add_(student[name], alpha=1 - momentum)
        base.require(torch.equal(computed, after[name]), f"EMA numerical transition: {name}")
    buffers = witness["buffers"]
    base.require(set(buffers) == {"student", "teacher"}, "EMA buffer witness membership")
    base.require(set(buffers["student"]) == set(buffers["teacher"]), "EMA buffer witness schema")
    for name, value in buffers["student"].items():
        base.require(torch.equal(value, buffers["teacher"][name]), "EMA buffers must copy exactly")
    return {"parameter_tensors": len(shapes), "buffer_tensors": len(buffers["student"])}


def audit_cohort(plan, records, split, panel):
    base.require(split in ("prediction", "control") and len(records) == plan[f"{split}_episodes"],
                 "Fresh cohort coverage")
    transitions, maximum = 0, 0.
    for index, record in enumerate(records):
        check_budget()
        meta = record["metadata"]
        base.require(meta["seed"] == protocol.seed(plan, f"{split}/reset/{index}")
                     and meta["noise_seed"] == protocol.seed(plan, f"{split}/actuator_noise/{index}")
                     and meta["noise_std"] == plan["noise_std"]
                     and meta["sensor_schedule"] == protocol.schedule(plan, index, panel, split).tolist()
                     and meta["policy_keys"] == ["packets", "commands"],
                     "Fresh cohort stream, public-input or schedule binding")
        if split == "prediction":
            seed = protocol.seed(plan, f"prediction/exploration/{index}")
            mode = str(np.random.default_rng(seed).choice(
                ["ik_pd", "random_low", "random_high"], p=[.5, .25, .25]))
            base.require(meta["action_seed"] == seed and meta["requested_policy"] == "mixed"
                         and meta["collector_policy"] == mode and meta["action_hold"] == 4
                         and meta["exploration_std"] == {"ik_pd": .12, "random_low": .3,
                                                          "random_high": .8}[mode],
                         "Prediction collection mixture identity")
        replay = search_audit.native.native_replay(record)
        base.require(replay["transitions"] == plan["steps"] and replay["new_policy_calls"] == 0
                     and replay["saved_output_only"] is True, "Native replay coverage")
        transitions += replay["transitions"]
        maximum = max(maximum, replay["max_abs_error"])
    return {"episodes": len(records), "transitions": transitions, "max_abs_error": maximum}


def validate_lineage(plan, execution):
    """Authenticate bytes and cohort identities without reading old efficacy metrics."""
    source = plan["training_source"]
    base.require(set(source) == {"execution_path", "plan_path", "plan_sha256", "audit_receipt_path",
                                "audit_receipt_sha256", "members", "cohort_plan"},
                 "Training provenance schema")
    base.require(base.read(execution / "train-provenance.json") == source, "Copied training provenance")
    parent = base.read(checked(ROOT / source["plan_path"], source["plan_sha256"]))
    receipt = base.read(checked(ROOT / source["audit_receipt_path"], source["audit_receipt_sha256"]))
    base.require(receipt["status"] == "completed" and receipt["plan_sha256"] == source["plan_sha256"],
                 "Original training-source completion")
    cohort = {name: parent[name] for name in search_audit.inherited.prior.COHORT_FIELDS}
    base.require(source["cohort_plan"] == cohort and set(source["members"]) == {"train.npz", "train.json"},
                 "Original training corpus identity")
    for name in ("train_episodes", "steps", "noise_std"):
        base.require(plan[name] == cohort[name], "Training corpus task/count")
    original = ROOT / source["execution_path"]
    checked(original / "completed.json", receipt["execution_completed_sha256"])
    for name, digest in source["members"].items():
        base.require(receipt["execution_members"][name] == digest, "Original training audit membership")
        checked(original / name, digest)
        checked(execution / name, digest)
    baseline = plan["search_source"]
    base.require(set(baseline) == {"plan_path", "plan_sha256", "audit_path", "audit_receipt_sha256",
                                  "prior_costs", "training_weights_reused"}, "Search source schema")
    baseline_plan = base.read(checked(ROOT / baseline["plan_path"], baseline["plan_sha256"]))
    baseline_receipt_path = checked(ROOT / baseline["audit_path"], baseline["audit_receipt_sha256"])
    baseline_receipt = base.read(baseline_receipt_path)
    base.require(baseline_receipt["status"] == "completed" and baseline_receipt["saved_output_only"] is True
                 and baseline_receipt["plan_sha256"] == baseline["plan_sha256"]
                 and baseline_receipt["source_sha256"] == baseline_plan["sources"]
                 and baseline_receipt["costs"] == baseline["prior_costs"]
                 and baseline["training_weights_reused"] is False,
                 "Completed search lineage; no inherited checkpoint reuse")
    for name, digest in baseline_plan["sources"].items():
        checked(ROOT / name, digest)
    for name, digest in baseline_receipt["files"].items():
        checked(baseline_receipt_path.parent / name, digest)
    return cohort, baseline_plan


def validate_runtime_sources(plan):
    import gymnasium.envs.mujoco.reacher_v5 as reacher
    import mujoco

    runtime = plan["runtime"]
    base.require(set(runtime) == {"python", "platform", "packages", "native_source_sha256",
                                  "native_xml_sha256", "mujoco_init_sha256"}
                 and set(runtime["packages"]) == {"numpy", "torch", "gymnasium", "mujoco"}, "Runtime exact schema")
    base.require(runtime["python"] == sys.version.split()[0]
                 and runtime["platform"] == platform.platform(), "Pinned Python/platform runtime")
    for name, version in runtime["packages"].items():
        base.require(importlib.metadata.version(name) == version, f"Pinned package runtime: {name}")
    for name, path in {"native_source_sha256": reacher.__file__,
                       "native_xml_sha256": Path(reacher.__file__).parent / "assets/reacher.xml",
                       "mujoco_init_sha256": mujoco.__file__}.items():
        checked(path, runtime[name])
    base.require(isinstance(plan["sources"], dict) and bool(plan["sources"]), "Frozen implementation sources")
    for path, digest in plan["sources"].items():
        checked(ROOT / path, digest)


def validate_streams(plan, baseline, execution):
    descriptors = list(baseline["random_stream_contract"]["priors"])
    numpy_registries = [concrete_streams(base.read(checked(ROOT / row["plan_path"], row["plan_sha256"])),
                                        include_train=row["include_train"]) for row in descriptors]
    numpy_registries.append(search_audit.protocol.registry(baseline))
    descriptors.append({"plan_path": plan["search_source"]["plan_path"],
                        "plan_sha256": plan["search_source"]["plan_sha256"],
                        "registry_kind": "named-search", "include_train": False})
    for previous in baseline["random_stream_contract"]["engineering_exclusions"]:
        numpy_registries.append(previous["registry"])
        descriptors.append({"plan_path": plan["search_source"]["plan_path"],
                            "plan_sha256": plan["search_source"]["plan_sha256"],
                            "registry_kind": "named-search-engineering",
                            "namespace": previous["namespace"], "include_train": False})
    torch_seeds = {}
    for prefix, seeds in (("world", (211, 223, 239)), ("reward", (271, 283, 293))):
        for seed in seeds:
            torch_seeds[f"{prefix}/student/{seed}"] = seed
            torch_seeds[f"{prefix}/minibatch/{seed}"] = seed + 4_100_000
    capacity = base.read(checked(ROOT / CAPACITY_PATH, CAPACITY_SHA))
    capacity_seed = capacity["synthetic_data"]["seed"]
    torch_seeds["engineering/capacity/synthetic/public_data"] = capacity_seed
    torch_seeds["engineering/temporary-overwritten-construction"] = 0
    capacity_state = protocol.torch_generator_manifest({"capacity": capacity_seed})
    base.require(capacity_state["capacity"]["initial_state_sha256"]
                 == capacity["synthetic_data"]["initial_state_sha256"], "Capacity synthetic RNG identity")
    descriptors.append({"artifact_path": CAPACITY_PATH, "artifact_sha256": CAPACITY_SHA,
                        "registry_kind": "additional-engineering-torch",
                        "roles": ["engineering/capacity/synthetic/public_data",
                                  "engineering/temporary-overwritten-construction"]})
    torch_seeds.update(TRAINING_TEST_TORCH_SEEDS)
    test_source = "tests/test_reacher_objective_training.py"
    descriptors.append({"artifact_path": test_source, "artifact_sha256": base.sha(ROOT / test_source),
                        "registry_kind": "additional-engineering-torch",
                        "roles": sorted(TRAINING_TEST_TORCH_SEEDS),
                        "seeds": dict(sorted(TRAINING_TEST_TORCH_SEEDS.items()))})
    audit_test_source = "tests/test_audit_reacher_objective_study.py"
    audit_numpy = {"engineering/auditor-tests/public_data": 716}
    audit_torch = {"engineering/auditor-tests/corrupted-minibatch-state": 72}
    numpy_registries.append(audit_numpy)
    torch_seeds.update(audit_torch)
    for kind, seeds in (("numpy", audit_numpy), ("torch", audit_torch)):
        descriptors.append({"artifact_path": audit_test_source,
                            "artifact_sha256": base.sha(ROOT / audit_test_source),
                            "registry_kind": f"additional-engineering-{kind}",
                            "roles": sorted(seeds), "seeds": dict(sorted(seeds.items()))})
    expected = protocol.stream_contract(plan, numpy_registries, [torch_seeds], descriptors)
    base.require(plan["random_stream_contract"] == expected
                 and base.read(execution / "random-streams.json") == expected,
                 "Named stream manifest, prior exclusions and generator identities")
    return {"numpy_generators": len(expected["generators"]),
            "torch_generators": len(expected["torch_generators"]),
            "prior_numpy_registries": len(numpy_registries), "prior_torch_registries": 1,
            "draws_for_manifest": 0}


def audit_orders(plan, execution):
    result = {}
    for pair in protocol.PAIRS:
        path = execution / "orders" / f"{pair}.npy"
        value = np.load(path, allow_pickle=False)
        search_audit.array(value, (plan["epochs"], plan["train_episodes"]), np.int64,
                           "Training epoch permutations")
        wanted = protocol.minibatch_orders(plan, pair).numpy()
        base.require(np.array_equal(value, wanted), "Training order differs from named generator")
        meta = base.read(path.with_suffix(".json"))
        base.require(meta == {"pair": pair, "role": f"fit/minibatch/{pair}",
                              "seed": protocol.seed(plan, f"fit/minibatch/{pair}"), "shape": list(value.shape),
                              "file_sha256": base.sha(path),
                              "tensor_sha256": canonical_tensor_hash({"orders": torch.from_numpy(value.copy())})},
                     "Order sidecar provenance")
        result[pair] = value
    return result


def audit_initializations(plan, execution):
    expected_keys = {"version", "settings", "seeds", "student_state", "predictor_state", "student_buffers",
                     "rng_states", "hashes", "initialization_wall_seconds"}
    result = {}
    for pair in protocol.PAIRS:
        initial = read_tensor_file(execution / "initializations" / f"{pair}.pt")
        base.require(set(initial) == expected_keys and initial["version"] == TRAINING_VERSION
                     and initial["settings"] == training_configuration(plan), "Paired initialization schema/config")
        tensor_mapping(initial["student_state"], student_shapes(plan), "Paired initial student")
        tensor_mapping(initial["predictor_state"], predictor_shapes(plan), "Paired initial predictor")
        base.require(initial["student_buffers"] == {}, "Unexpected initial model buffers")
        base.require(initial["seeds"] == {name: protocol.seed(plan, f"fit/{name}/{pair}")
                                         for name in ("student", "predictor")}, "Initialization named generators")
        base.require(set(initial["rng_states"]) == {f"{name}_{point}" for name in ("student", "predictor")
                                                  for point in ("before", "after")}, "Initialization RNG schema")
        for name in ("student", "predictor"):
            wanted = torch.Generator().manual_seed(initial["seeds"][name]).get_state()
            before, after = (initial["rng_states"][f"{name}_{point}"] for point in ("before", "after"))
            base.require(torch.equal(before, wanted) and after.dtype == torch.uint8
                         and after.shape == wanted.shape and not torch.equal(before, after),
                         "Initialization RNG state binding")
        base.require(set(initial["hashes"]) == {"student_state", "predictor_state", "student_buffers", "rng_states"},
                     "Initialization hash membership")
        for name, digest in initial["hashes"].items():
            base.require(canonical_tensor_hash(initial[name]) == digest, "Initialization tensor identity")
        finite(initial["initialization_wall_seconds"], "Initialization wall", positive=True)
        path = execution / "initializations" / f"{pair}.pt"
        base.require(base.read(path.with_suffix(".json")) == {
            "pair": pair, "file_sha256": base.sha(path), "settings": initial["settings"],
            "seeds": initial["seeds"], "hashes": initial["hashes"],
            "initialization_wall_seconds": initial["initialization_wall_seconds"]},
            "Initialization sidecar identity")
        result[pair] = initial
    return result


def expected_work(plan, arm, batch):
    steps, horizon = plan["steps"], plan["rollout_horizon"]
    if arm == "anchor":
        starts = max(0, steps - horizon + 1)
        calls = {"student_assimilate": steps, "student_prefix_advance": steps,
                 "student_rollout_advance": horizon if starts and plan["rollout_weight"] else 0,
                 "student_posterior_kl": steps}
        samples = {key: value * batch for key, value in calls.items()}
        samples["student_rollout_advance"] *= starts
    else:
        horizons = plan["auxiliary_horizons"]
        calls = {"student_assimilate": steps + 1, "student_prefix_advance": steps,
                 "student_open_loop_advance": sum(min(max(horizons), steps - t) for t in range(steps)),
                 "student_predictor": sum(max(0, steps + 1 - h) for h in horizons)}
        if arm == "raw":
            calls["student_endpoint_decoder"] = calls["student_predictor"]
        else:
            calls.update(teacher_assimilate=steps + 1, teacher_advance=steps)
        samples = {key: value * batch for key, value in calls.items()}
    return {"batch_forward_calls": calls, "sample_forward_evaluations": samples}


def audit_work(plan, arm, batch, saved, *, calibration=False):
    expected = expected_work(plan, arm, batch)
    if arm == "anchor":
        expected["counts_are_not_flops"] = True
    else:
        key = ("collapse_diagnostics_included_in_forward_wall" if calibration
               else "collapse_diagnostics_included_in_auxiliary_wall")
        expected[key] = arm == "latent"
    base.require(saved == expected, f"{arm} batch/sample compute accounting")


def audit_calibration(plan, execution, initializations):
    receipt = base.read(execution / "calibration.json")
    gradients = read_tensor_file(execution / "calibration-gradients.pt")
    keys = {"version", "rule", "batch_size", "pairs", "training_indices", "rows", "norms", "multipliers",
            "backbone_parameter_names", "initialization_hashes", "initial_artifact_hashes_before",
            "initial_artifact_hashes_after", "gradients_sha256", "optimizer_updates", "ema_updates",
            "global_rng_unchanged", "costs"}
    base.require(set(receipt) == keys and receipt["version"] == TRAINING_VERSION
                 and receipt["rule"] == plan["calibration"] and receipt["training_indices"] == list(range(128)),
                 "Calibration rule/source-order membership")
    exact_integer(receipt["batch_size"], 32, "Calibration complete batch size")
    exact_integer(receipt["pairs"], 3, "Calibration paired initializations")
    exact_integer(receipt["optimizer_updates"], 0, "Calibration cannot optimize")
    exact_integer(receipt["ema_updates"], 0, "Calibration cannot update teacher")
    base.require(receipt["global_rng_unchanged"] is True, "Calibration global RNG receipt")
    shapes = {key: shape for key, shape in student_shapes(plan).items()
              if key.startswith(("observation_update.", "transition."))}
    base.require(receipt["backbone_parameter_names"] == list(shapes), "Calibration named backbone")
    ordered_initial = [initializations[pair] for pair in protocol.PAIRS]
    hashes = [canonical_state_hash(value) for value in ordered_initial]
    base.require(receipt["initialization_hashes"] == [value["hashes"] for value in ordered_initial]
                 and receipt["initial_artifact_hashes_before"] == hashes
                 and receipt["initial_artifact_hashes_after"] == hashes,
                 "Calibration bound unchanged initial artifacts")
    base.require(len(receipt["rows"]) == 12, "All twelve calibration batches required")
    norms = {arm: [] for arm in ARMS}
    wanted_keys = set()
    for ordinal, row in enumerate(receipt["rows"]):
        check_budget()
        pair, batch = divmod(ordinal, 4)
        base.require(set(row) == {"pair", "batch", "indices", "losses", "work"}
                     and row["pair"] == pair and row["batch"] == batch
                     and row["indices"] == list(range(batch * 32, (batch + 1) * 32))
                     and set(row["losses"]) == set(ARMS) and set(row["work"]) == set(ARMS),
                     "Calibration first four complete corpus-order batches")
        for arm in ARMS:
            prefix = f"pair{pair}/batch{batch}/{arm}/"
            wanted_keys.update(prefix + name for name in shapes)
            values = {name: gradients[prefix + name] for name in shapes}
            tensor_mapping(values, shapes, "Calibration gradient")
            measured = math.sqrt(sum(float(value.double().square().sum()) for value in values.values()))
            finite(measured, "Calibration gradient norm", positive=True)
            saved = row["losses"][arm]
            base.require(set(saved) == {"loss", "backbone_norm", "gradient_sha256"}
                         and math.isclose(saved["backbone_norm"], measured, rel_tol=1e-12, abs_tol=1e-12)
                         and saved["gradient_sha256"] == canonical_tensor_hash(values),
                         "Calibration gradient-to-norm arithmetic")
            base.require(finite(saved["loss"], "Calibration loss") >= 0, "Nonnegative calibration objective")
            norms[arm].append(measured)
            audit_work(plan, arm, 32, row["work"][arm], calibration=True)
    base.require(set(gradients) == wanted_keys and receipt["gradients_sha256"] == canonical_tensor_hash(gradients),
                 "Complete calibration gradient membership/hash")
    for arm in ARMS:
        base.require(np.allclose(receipt["norms"][arm], norms[arm], rtol=1e-12, atol=1e-12),
                     "Calibration pooled twelve norms")
    multipliers = {}
    anchor_median = float(np.median(norms["anchor"]))
    base.require(set(receipt["multipliers"]) == {"raw", "latent"}, "Calibration multiplier membership")
    for arm in ("raw", "latent"):
        auxiliary_median = float(np.median(norms[arm]))
        unbounded = .5 * anchor_median / auxiliary_median
        expected = {"value": float(np.clip(unbounded, .01, 100.)), "unclipped": unbounded,
                    "bound_hit": not .01 <= unbounded <= 100., "anchor_median": anchor_median,
                    "auxiliary_median": auxiliary_median}
        saved = receipt["multipliers"][arm]
        base.require(set(saved) == set(expected) and type(saved["bound_hit"]) is bool, "Multiplier schema")
        for key, value in expected.items():
            base.require(saved[key] == value if type(value) is bool else
                         math.isclose(saved[key], value, rel_tol=1e-12, abs_tol=1e-12),
                         "Frozen calibration multiplier formula")
        multipliers[arm] = saved["value"]
    costs = receipt["costs"]
    phases = {"setup_seconds", "anchor_forward_seconds", "raw_forward_seconds", "latent_forward_seconds",
              "gradient_seconds"}
    base.require(set(costs) == phases | {"wall_seconds"}, "Calibration cost schema")
    for key, value in costs.items():
        finite(value, f"Calibration cost {key}", positive=True)
    base.require(sum(costs[key] for key in phases) <= costs["wall_seconds"] + 1e-6,
                 "Uncharged calibration phase work")
    return {"multipliers": multipliers, "norms": norms, "costs": costs, "calibration_batches": 12,
            "scope": "Saved gradient-vector norms and rule reconstructed; neural gradient origin not rerun."}


def audit_checkpoint(plan, payload, initial, arm, pair, multiplier, order):
    keys = {"version", "arm", "settings", "loss_multiplier", "source_sha256", "runtime", "initialization",
            "student_state", "student_buffers", "auxiliary_state", "auxiliary_buffers", "auxiliary_configuration",
            "optimizer_state", "optimizer_group_names", "minibatch_seed", "minibatch_rng_state", "cursor",
            "successful_updates", "optimizer_steps", "ema_updates", "failed", "log_chain_sha256",
            "setup_wall_seconds", "training_wall_seconds", "integrity_sha256"}
    base.require(set(payload) == keys and payload["version"] == TRAINING_VERSION and payload["arm"] == arm,
                 "Final training checkpoint schema")
    base.require(payload["integrity_sha256"] == canonical_state_hash(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}), "Checkpoint integrity")
    base.require(payload["settings"] == training_configuration(plan) and payload["source_sha256"] == plan["sources"]
                 and payload["runtime"] == plan["runtime"] and payload["loss_multiplier"] == multiplier,
                 "Checkpoint settings, scalar loss weight, source and runtime")
    base.require(canonical_state_hash(payload["initialization"]) == canonical_state_hash(initial),
                 "Checkpoint paired initialization identity")
    base.require(payload["failed"] is False, "Failed attempt cannot pass completed audit")
    batch_count = plan["train_episodes"] // plan["batch_size"]
    updates = plan["epochs"] * batch_count
    for key in ("successful_updates", "optimizer_steps"):
        exact_integer(payload[key], updates, "Final successful optimizer count")
    exact_integer(payload["ema_updates"], updates if arm == "latent" else 0, "Final teacher EMA count")
    base.require(payload["cursor"] == {"epoch": plan["epochs"] - 1, "batch": batch_count - 1},
                 "Final epoch/batch cursor")
    student = tensor_mapping(payload["student_state"], student_shapes(plan), "Final student")
    base.require(payload["student_buffers"] == {}, "Unexpected student buffers")
    base.require(payload["minibatch_seed"] == protocol.seed(plan, f"fit/minibatch/{pair}"),
                 "Checkpoint minibatch generator role")
    generator = torch.Generator().manual_seed(payload["minibatch_seed"])
    wanted_order = torch.stack([torch.randperm(plan["train_episodes"], generator=generator)
                               for _ in range(plan["epochs"])])
    base.require(np.array_equal(order, wanted_order.numpy())
                 and torch.equal(payload["minibatch_rng_state"], generator.get_state()),
                 "Final minibatch generator/cursor state")
    if arm == "anchor":
        base.require(payload["auxiliary_state"] is None and payload["auxiliary_buffers"] is None
                     and payload["auxiliary_configuration"] is None, "Anchor has no auxiliary")
    else:
        shapes = {"predictor." + name: shape for name, shape in predictor_shapes(plan).items()}
        if arm == "latent":
            shapes.update({"teacher." + name: shape for name, shape in student_shapes(plan).items()})
        tensor_mapping(payload["auxiliary_state"], shapes, "Final auxiliary/actual teacher")
        base.require(payload["auxiliary_buffers"] == {}, "Unexpected auxiliary buffers")
        wanted = {"kind": arm, "horizons": plan["auxiliary_horizons"],
                  "momentum": plan["ema_momentum"] if arm == "latent" else None,
                  "variance_weight": 0., "covariance_weight": 0.,
                  "model": {key: plan[key] for key in ("hidden_size", "dt", "noise_std", "residual_reward")},
                  "teacher_present": arm == "latent"}
        base.require(payload["auxiliary_configuration"] == wanted, "Auxiliary Python/teacher configuration")
    named = {"student." + name: shape for name, shape in student_shapes(plan).items()}
    if arm != "anchor":
        named.update({"predictor." + name: shape for name, shape in predictor_shapes(plan).items()})
    base.require(payload["optimizer_group_names"] == [list(named)], "Optimizer ownership/order; teacher excluded")
    optimizer = payload["optimizer_state"]
    base.require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
                 "Single Adam parameter group")
    group = optimizer["param_groups"][0]
    wanted = {key: value for key, value in plan["optimizer"].items() if key != "name"}
    wanted.update(lr=plan["learning_rate"], betas=tuple(wanted["betas"]), params=list(range(len(named))))
    base.require(group == wanted, "Exact Adam scalar/group configuration")
    base.require(set(optimizer["state"]) == set(group["params"]), "Adam state for every trainable tensor")
    for index, (name, shape) in enumerate(named.items()):
        state = optimizer["state"][index]
        tensor_mapping(state, {"step": (), "exp_avg": shape, "exp_avg_sq": shape}, f"Adam state {name}")
        base.require(float(state["step"]) == updates and torch.all(state["exp_avg_sq"] >= 0).item(),
                     "Adam recorded numerical steps and second moments")
    for key in ("setup_wall_seconds", "training_wall_seconds"):
        finite(payload[key], f"Checkpoint cost {key}", positive=True)
    return student


def audit_training_logs(plan, path, arm, order, train, checkpoint):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    batches = plan["train_episodes"] // plan["batch_size"]
    updates = plan["epochs"] * batches
    base.require(len(rows) == updates, "Exactly every successful training update must be recorded")
    keys = {"arm", "epoch", "batch", "update", "optimizer_steps", "ema_updates", "indices", "indices_sha256",
            "anchor", "auxiliary", "loss_multiplier", "anchor_loss", "auxiliary_loss", "total_loss",
            "backbone_gradient_norm", "predictor_gradient_norm", "combined_gradient_norm", "gradient_clipped",
            "anchor_work", "auxiliary_work", "ema_witness_captured", "costs", "previous_log_sha256", "log_sha256"}
    packets = base.stack(train, "policy", "packets")
    previous = hashlib.sha256(b"").hexdigest()
    costs, clipped = {}, 0
    for offset, row in enumerate(rows):
        check_budget()
        epoch, batch = divmod(offset, batches)
        base.require(set(row) == keys and row["arm"] == arm, "Training log schema/arm")
        for key, value in {"epoch": epoch, "batch": batch, "update": offset + 1,
                           "optimizer_steps": offset + 1, "ema_updates": offset + 1 if arm == "latent" else 0}.items():
            exact_integer(row[key], value, f"Training update counter {key}")
        selected = order[epoch, batch * plan["batch_size"]:(batch + 1) * plan["batch_size"]]
        base.require(row["indices"] == selected.tolist() and row["indices_sha256"] == canonical_tensor_hash(
            {"indices": torch.from_numpy(selected.copy())}), "Exact paired minibatch order")
        body = {key: value for key, value in row.items() if key not in ("previous_log_sha256", "log_sha256")}
        current = hashlib.sha256(bytes.fromhex(previous) + json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        base.require(row["previous_log_sha256"] == previous and row["log_sha256"] == current,
                     "Per-update log chain")
        previous = current
        for key in ("anchor_loss", "auxiliary_loss", "total_loss", "backbone_gradient_norm",
                    "predictor_gradient_norm", "combined_gradient_norm"):
            base.require(finite(row[key], key) >= 0, "Nonnegative training metric")
        base.require(row["loss_multiplier"] == checkpoint["loss_multiplier"] and math.isclose(
            row["total_loss"], row["anchor_loss"] + row["loss_multiplier"] * row["auxiliary_loss"],
            rel_tol=1e-5, abs_tol=1e-6), "Combined training objective arithmetic")
        anchor = row["anchor"]
        base.require(set(anchor) == base.LOG_KEYS - {"epoch", "gradient_norm"}, "Anchor metrics schema")
        for key, value in anchor.items():
            base.require(finite(value, key) >= 0, "Finite nonnegative anchor metrics")
        objective = anchor["observation_mse"] + plan["reward_scale"] * anchor["reward_mse"] + plan["rollout_weight"] * (
            anchor["rollout_observation_mse"] + plan["reward_scale"] * anchor["rollout_reward_mse"])
        base.require(anchor["kl_nats"] == 0 and math.isclose(anchor["loss"], objective, rel_tol=1e-5, abs_tol=1e-6)
                     and row["anchor_loss"] == anchor["loss"], "Unchanged native anchor objective")
        public = packets[selected]
        starts = max(0, plan["steps"] - plan["rollout_horizon"] + 1)
        base.require(anchor["valid_observation_targets"] == float(public[:, 1:, 6].sum())
                     and anchor["valid_rollout_starts"] == float(public[:, :starts, 6].sum()),
                     "Training anchor valid targets from bound corpus")
        audit_work(plan, "anchor", len(selected), row["anchor_work"])
        if arm == "anchor":
            base.require(row["auxiliary"] is None and row["auxiliary_work"] is None
                         and row["auxiliary_loss"] == row["predictor_gradient_norm"] == 0,
                         "Anchor cannot hide auxiliary computation")
        else:
            auxiliary = row["auxiliary"]
            counts = {str(h): int(((public[:, :-h, 6] > .5) & (public[:, h:, 6] > .5)).sum())
                      for h in plan["auxiliary_horizons"]}
            base.require(auxiliary["engineering_preview"] is True and auxiliary["batch_size"] == len(selected)
                         and auxiliary["valid_pairs_by_horizon"] == counts
                         and auxiliary["nonempty_horizons"] == sum(value > 0 for value in counts.values())
                         and auxiliary["prediction_loss"] == row["auxiliary_loss"], "Auxiliary masks/loss/counts")
            audit_work(plan, arm, len(selected), row["auxiliary_work"])
            base.require(auxiliary["batch_forward_calls"] == row["auxiliary_work"]["batch_forward_calls"],
                         "Auxiliary direct forward accounting")
            if arm == "latent":
                for kind in ("student", "teacher"):
                    diagnostic = auxiliary[kind]
                    exact_integer(diagnostic["samples"], sum(counts.values()), "Collapse sample coverage")
                    exact_integer(diagnostic["dimensions"], plan["hidden_size"], "Collapse latent dimension")
                    for name in ("min_std", "mean_std", "effective_rank"):
                        base.require(finite(diagnostic[name], name) >= 0, "Collapse diagnostic finite")
                    base.require(diagnostic["effective_rank"] <= max(0, min(sum(counts.values()) - 1, plan["hidden_size"])) + 1e-5,
                                 "Effective rank bound")
        combined = row["combined_gradient_norm"]
        base.require(row["backbone_gradient_norm"] ** 2 + row["predictor_gradient_norm"] ** 2
                     <= combined ** 2 * (1 + 1e-5) + 1e-8, "Disjoint gradient norm accounting")
        base.require(type(row["gradient_clipped"]) is bool
                     and row["gradient_clipped"] == (combined > plan["gradient_clip"]), "Gradient clipping flag")
        clipped += int(row["gradient_clipped"])
        base.require(row["ema_witness_captured"] is (arm == "latent" and offset in (0, updates - 1)),
                     "First/last EMA witnesses required")
        phase_names = {"anchor_forward_seconds", "auxiliary_forward_seconds", "backward_seconds",
                       "gradient_norm_and_clip_seconds", "optimizer_seconds", "ema_and_witness_seconds"}
        base.require(set(row["costs"]) == phase_names | {"batch_wall_seconds"}, "Training cost schema")
        for key, value in row["costs"].items():
            base.require(finite(value, key) >= 0, "Training phase timing")
            costs[key] = costs.get(key, 0.) + value
        base.require(sum(row["costs"][key] for key in phase_names) <= row["costs"]["batch_wall_seconds"] + 1e-6,
                     "Uncharged training phase work")
    base.require(previous == checkpoint["log_chain_sha256"] and costs["batch_wall_seconds"]
                 <= checkpoint["training_wall_seconds"] + 1e-6, "Final log-chain/training time binding")
    return {"updates": updates, "ema_updates": checkpoint["ema_updates"], "costs": costs,
            "gradient_clipped_updates": clipped, "gradient_clipping_fraction": clipped / updates,
            "log_chain_sha256": previous}


def fit_members(arm):
    members = {"initial-weights.pt", "weights.pt", "checkpoint.pt", "training.jsonl"}
    return members | ({"ema-first.pt", "ema-last.pt"} if arm == "latent" else set())


def audit_fit(plan, execution, row, initial, order, multiplier, train):
    check_budget()
    name, arm, pair = row["name"], row["arm"], row["pair"]
    folder = execution / "fits" / name
    receipt = base.read(folder / "completed.json")
    keys = {"name", "arm", "pair", "settings", "updates", "optimizer_steps", "ema_updates", "loss_multiplier",
            "student_parameters", "trainable_parameters", "initial_hashes", "order_tensor_sha256", "final_weights",
            "checkpoint_integrity_sha256", "log_chain_sha256", "wall_seconds", "files"}
    base.require(set(receipt) == keys and receipt["name"] == name and receipt["arm"] == arm
                 and receipt["pair"] == pair and receipt["settings"] == training_configuration(plan)
                 and set(receipt["files"]) == fit_members(arm), "Fit receipt exact schema/identity")
    for member, digest in receipt["files"].items():
        checked(folder / member, digest)
    beginning = read_tensor_file(folder / "initial-weights.pt")
    tensor_mapping(beginning, student_shapes(plan), "Saved fit initial weights")
    base.require(canonical_tensor_hash(beginning) == initial["hashes"]["student_state"]
                 and receipt["initial_hashes"] == initial["hashes"], "All arms share fresh paired initial tensors")
    checkpoint = read_tensor_file(folder / "checkpoint.pt")
    final = audit_checkpoint(plan, checkpoint, initial, arm, pair, multiplier, order)
    deployment = read_tensor_file(folder / "weights.pt")
    tensor_mapping(deployment, student_shapes(plan), "Deployment tensors")
    weight_hash = canonical_tensor_hash(final)
    base.require(canonical_tensor_hash(deployment) == weight_hash and receipt["final_weights"] == {
        "canonical_tensor_sha256": weight_hash, "file_sha256": receipt["files"]["weights.pt"]},
        "Final deployment tensors match checkpoint")
    base.require(receipt["checkpoint_integrity_sha256"] == checkpoint["integrity_sha256"]
                 and receipt["log_chain_sha256"] == checkpoint["log_chain_sha256"]
                 and receipt["loss_multiplier"] == multiplier
                 and receipt["order_tensor_sha256"] == canonical_tensor_hash({"orders": torch.from_numpy(order.copy())}),
                 "Fit scalar/checkpoint/log/order binding")
    for key, expected in {"updates": checkpoint["successful_updates"], "optimizer_steps": checkpoint["optimizer_steps"],
                          "ema_updates": checkpoint["ema_updates"]}.items():
        exact_integer(receipt[key], expected, f"Fit receipt {key}")
    parameters = sum(value.numel() for value in final.values())
    exact_integer(receipt["student_parameters"], parameters, "Student parameter count")
    exact_integer(receipt["trainable_parameters"], parameters + (0 if arm == "anchor" else
                  plan["hidden_size"] ** 2 + plan["hidden_size"]), "Matched auxiliary trainable capacity")
    logs = audit_training_logs(plan, folder / "training.jsonl", arm, order, train, checkpoint)
    witnesses = {}
    if arm == "latent":
        for label, update in (("first", 1), ("last", checkpoint["successful_updates"])):
            witness = read_tensor_file(folder / f"ema-{label}.pt")
            base.require(set(witness) == {"momentum", "update", "teacher_before", "student_after_optimizer",
                                          "teacher_after", "sha256"}
                         and witness["sha256"] == canonical_state_hash(
                             {key: value for key, value in witness.items() if key != "sha256"}),
                         "EMA witness integrity/schema")
            exact_integer(witness["update"], update, "EMA witness update index")
            base.require(witness["momentum"] == plan["ema_momentum"], "EMA frozen momentum")
            for key in ("teacher_before", "student_after_optimizer", "teacher_after"):
                base.require(set(witness[key]) == {"parameters", "buffers"} and witness[key]["buffers"] == {},
                             "EMA witness named state schema")
            witnesses[label] = audit_ema_witness({
                "teacher_before": witness["teacher_before"]["parameters"],
                "student_after_optimizer": witness["student_after_optimizer"]["parameters"],
                "teacher_after_ema": witness["teacher_after"]["parameters"],
                "buffers": {"student": witness["student_after_optimizer"]["buffers"],
                            "teacher": witness["teacher_after"]["buffers"]}},
                plan["ema_momentum"], student_shapes(plan))
            if label == "first":
                base.require(canonical_tensor_hash(witness["teacher_before"]["parameters"])
                             == initial["hashes"]["student_state"], "Teacher starts at exact paired student")
            else:
                teacher = {key.removeprefix("teacher."): value for key, value in checkpoint["auxiliary_state"].items()
                           if key.startswith("teacher.")}
                base.require(canonical_tensor_hash(witness["teacher_after"]["parameters"])
                             == canonical_tensor_hash(teacher) and
                             canonical_tensor_hash(witness["student_after_optimizer"]["parameters"]) == weight_hash,
                             "Actual saved final EMA teacher restoration")
    wall = finite(receipt["wall_seconds"], "Whole fit wall", positive=True)
    base.require(checkpoint["setup_wall_seconds"] + checkpoint["training_wall_seconds"] <= wall + 1e-6,
                 "Uncharged fit setup/training")
    return {"name": name, "arm": arm, "pair": pair, "loss_multiplier": multiplier,
            "parameters": parameters, "trainable_parameters": receipt["trainable_parameters"],
            "initial_hashes": initial["hashes"], "student_tensor_sha256": weight_hash,
            "wall_seconds": wall, "setup_seconds": checkpoint["setup_wall_seconds"],
            "training_seconds": checkpoint["training_wall_seconds"], "ema_witnesses": witnesses, **logs}


def expected_members(plan):
    members = {"started.json", "random-streams.json", "train.npz", "train.json", "train-provenance.json",
               "calibration-gradients.pt", "calibration.json", "calibration-completed.json", "all-fits-completed.json",
               "restored-models.json", "evaluation-started.json", "prediction.npz", "prediction.json", "costs.json"}
    for pair in protocol.PAIRS:
        members.update({f"initializations/{pair}.pt", f"initializations/{pair}.json",
                        f"orders/{pair}.npy", f"orders/{pair}.json"})
    for row in protocol.fit_manifest(plan):
        members.update(f"fits/{row['name']}/{name}" for name in fit_members(row["arm"]) | {"completed.json"})
        members.update(f"predictions/{row['name']}.{suffix}" for suffix in ("npz", "json"))
    for step in range(plan["steps"]):
        members.update(f"innovations/control/{step:03d}.{suffix}" for suffix in ("npz", "json"))
    for row in protocol.execution_order(plan):
        folder = row["path"]
        members.update(f"{folder}/{name}" for name in ("episodes.npz", "episodes.json", "timings.json"))
        if "fit" in row:
            members.update(f"{folder}/{name}" for name in ("executed_predictions.npz", "reset-events.npz"))
            members.update(f"{folder}/decisions/{step:03d}.{suffix}"
                           for step in range(plan["steps"]) for suffix in ("npz", "json"))
        else:
            members.add(f"{folder}/planning.npz")
    return members


def validate_members(plan, expected, execution):
    base.require(isinstance(expected, str) and len(expected) == 64
                 and all(char in "0123456789abcdef" for char in expected), "External frozen plan SHA")
    completed = base.read(execution / "completed.json")
    keys = {"status", "study", "plan_sha256", "fits", "control_rows", "prediction_episodes", "astra_calls",
            "wall_seconds", "evaluation_started_elapsed_seconds", "cumulative_attempt_wall_seconds", "files"}
    base.require(set(completed) == keys and completed["status"] == "completed"
                 and completed["study"] == VERSION and completed["plan_sha256"] == expected,
                 "Completed objective execution identity")
    for name, value in {"fits": 9, "control_rows": 57, "prediction_episodes": plan["prediction_episodes"],
                        "astra_calls": 0}.items():
        exact_integer(completed[name], value, f"Completed coverage: {name}")
    base.require(finite(completed["wall_seconds"], "Whole-run wall", positive=True) <= plan["cap_seconds"],
                 "Frozen execution cap exceeded")
    paths = list(execution.rglob("*"))
    base.require(not execution.is_symlink() and not any(path.is_symlink() for path in paths), "Artifact symlinks")
    members = expected_members(plan)
    base.require({str(path.relative_to(execution)) for path in paths if path.is_file()} == members | {"completed.json"}
                 and set(completed["files"]) == members, "Exact complete execution membership; partials forbidden")
    for path, digest in completed["files"].items():
        checked(execution / path, digest)
    return completed


def audit_boundaries(plan, expected, execution, completed, fits):
    started = base.read(execution / "started.json")
    calibration = base.read(execution / "calibration-completed.json")
    all_fits = base.read(execution / "all-fits-completed.json")
    restored = base.read(execution / "restored-models.json")
    evaluation = base.read(execution / "evaluation-started.json")
    base.require(set(started) == {"plan_sha256", "unix_time"} and started["plan_sha256"] == expected,
                 "Execution start receipt")
    base.require(set(calibration) == {"files", "initialization_files", "optimizer_updates", "unix_time"}
                 and calibration["optimizer_updates"] == 0
                 and calibration["files"] == {name: base.sha(execution / name)
                      for name in ("calibration.json", "calibration-gradients.pt")}
                 and calibration["initialization_files"] == {f"initializations/{pair}.pt":
                     base.sha(execution / "initializations" / f"{pair}.pt") for pair in protocol.PAIRS},
                 "Calibration completed before fitting boundary")
    base.require(set(all_fits) == {"plan_sha256", "fit_order", "files", "calibration_completed_sha256",
                                   "unix_time", "elapsed_seconds"}
                 and all_fits["plan_sha256"] == expected and all_fits["fit_order"] == plan["fit_order"]
                 and all_fits["files"] == {f"fits/{name}/completed.json":
                     base.sha(execution / "fits" / name / "completed.json") for name in plan["fit_order"]}
                 and all_fits["calibration_completed_sha256"] == base.sha(execution / "calibration-completed.json"),
                 "All nine authenticated fit completions")
    base.require(set(restored) == {"models", "wall_seconds", "unix_time"}
                 and set(restored["models"]) == set(plan["fit_order"]), "All restored models before evaluation")
    for name, fit in fits.items():
        base.require(restored["models"][name] == {
            "checkpoint_sha256": base.sha(execution / "fits" / name / "checkpoint.pt"),
            "weights_sha256": base.sha(execution / "fits" / name / "weights.pt"),
            "student_tensor_sha256": fit["student_tensor_sha256"], "successful_updates": fit["updates"],
            "restored_auxiliary_for_validation_only": fit["arm"] != "anchor"}, "Restored tensor/count identity")
    base.require(set(evaluation) == {"plan_sha256", "all_fits_completed_sha256", "restored_models_sha256",
                                     "unix_time", "elapsed_seconds"}
                 and evaluation["plan_sha256"] == expected
                 and evaluation["all_fits_completed_sha256"] == base.sha(execution / "all-fits-completed.json")
                 and evaluation["restored_models_sha256"] == base.sha(execution / "restored-models.json")
                 and evaluation["elapsed_seconds"] == completed["evaluation_started_elapsed_seconds"],
                 "Evaluation boundary binds completed fits and actual restored tensors")
    dates = [finite(value["unix_time"], "Phase Unix time", positive=True)
             for value in (started, calibration, all_fits, restored, evaluation)]
    base.require(dates == sorted(dates) and 0 < all_fits["elapsed_seconds"] < evaluation["elapsed_seconds"]
                 < completed["wall_seconds"], "Logged phase order")
    restore_seconds = finite(restored["wall_seconds"], "Restoration wall", positive=True)
    base.require(all_fits["elapsed_seconds"] + restore_seconds <= evaluation["elapsed_seconds"] + 1e-6,
                 "Restoration must fit before fresh evaluation")
    return {"calibration_before_fits": True, "all_nine_restored_before_evaluation": True,
            "fit_completed_elapsed_seconds": all_fits["elapsed_seconds"],
            "evaluation_started_elapsed_seconds": evaluation["elapsed_seconds"],
            "restore_seconds": restore_seconds, "scope": "Hash-bound logged order, not an independent process observer."}


def audit_inherited_training(cohort_plan, records):
    return base.audit_cohort(cohort_plan, records, "train")


def audit_costs(plan, execution, completed, initializations, calibration, fits, boundary, predictions, controls):
    saved = base.read(execution / "costs.json")
    phases = {"initialization_and_order_seconds", "calibration_seconds", "preparation_wall_seconds", "fit_wall_seconds",
              "restore_wall_seconds", "prediction_collection_seconds", "prediction_model_seconds",
              "innovation_generation_and_storage_seconds", "control_row_wall_seconds", "control_setup_seconds",
              "control_decision_seconds", "control_native_step_seconds"}
    base.require(set(saved) == phases | {"new_fits", "astra_calls", "prior_costs", "accounting"}, "Whole-study cost schema")
    for key in phases:
        base.require(finite(saved[key], key) >= 0, "Nonnegative phase costs")
    rows = [row for panel in controls.values() for row in panel.values()]
    calculated = {
        "calibration_seconds": calibration["costs"]["wall_seconds"],
        "fit_wall_seconds": sum(row["wall_seconds"] for row in fits.values()),
        "restore_wall_seconds": boundary["restore_seconds"],
        "prediction_model_seconds": sum(row["timing"]["wall_seconds"] for row in predictions.values()),
        "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in rows),
        "control_setup_seconds": sum(row["setup_seconds"] for row in rows),
        "control_decision_seconds": sum(row["decision_wall_seconds"] for row in rows),
        "control_native_step_seconds": sum(row["native_step_seconds"] for row in rows),
    }
    for key, value in calculated.items():
        base.require(math.isclose(saved[key], value, rel_tol=1e-12, abs_tol=1e-7), f"Cost reconstruction: {key}")
    initialization = sum(value["initialization_wall_seconds"] for value in initializations.values())
    base.require(initialization <= saved["initialization_and_order_seconds"] + 1e-6
                 and saved["initialization_and_order_seconds"] + saved["calibration_seconds"]
                 <= saved["preparation_wall_seconds"] + 1e-6, "Calibration/initialization nested cost")
    before_fit_end = saved["preparation_wall_seconds"] + saved["fit_wall_seconds"]
    before_eval = before_fit_end + saved["restore_wall_seconds"]
    evaluation = sum(saved[key] for key in ("prediction_collection_seconds", "prediction_model_seconds",
                                           "innovation_generation_and_storage_seconds", "control_row_wall_seconds"))
    base.require(before_fit_end <= boundary["fit_completed_elapsed_seconds"] + 1e-6
                 and before_eval <= boundary["evaluation_started_elapsed_seconds"] + 1e-6
                 and boundary["evaluation_started_elapsed_seconds"] + evaluation <= completed["wall_seconds"] + 1e-6,
                 "Nonoverlapping phase costs exceed execution time")
    exact_integer(saved["new_fits"], 9, "All nine fits charged")
    exact_integer(saved["astra_calls"], 0, "No Astra calls in local objective study")
    base.require(saved["prior_costs"] == plan["search_source"]["prior_costs"], "Prior cumulative cost identity")
    old = finite(saved["prior_costs"]["cumulative_attempt_wall_seconds"], "Earlier attempts", positive=True)
    base.require(completed["cumulative_attempt_wall_seconds"] == old + completed["wall_seconds"],
                 "Cumulative costs add new execution exactly once")
    return {**saved, "new_execution_wall_seconds": completed["wall_seconds"],
            "cumulative_attempt_wall_seconds": completed["cumulative_attempt_wall_seconds"]}


def audit_saved(plan, expected_plan_sha256, execution, out, *, engineering=False):
    """Audit source-bound saved evidence; caller authenticates external plan bytes."""
    execution, out = Path(execution), Path(out)
    base.require(not out.exists(), "Exclusive objective audit output required")
    begin = time.monotonic()
    base.require(type(plan["audit_cap_seconds"]) is int and plan["audit_cap_seconds"] > 0,
                 "Frozen positive audit cap")
    token = _DEADLINE.set(begin + plan["audit_cap_seconds"])
    try:
        protocol.validate_settings(plan, engineering=engineering)
        base.require(plan["engineering"] is engineering, "Explicit engineering audit boundary")
        base.require(plan["criterion"] == {
            "objective_improvement": .05, "every_pair_nonworse": True, "reset_increase": .05,
            "every_pair_reset_positive": True, "versus_zero_improvement": .10,
            "physics_versus_zero_improvement": .10, "panels": ["ordinary", "shift"], "expected_checks": 31},
            "Frozen continuation thresholds")
        validate_runtime_sources(plan)
        completed = validate_members(plan, expected_plan_sha256, execution)
        completion_hash = base.sha(execution / "completed.json")
        cohort_plan, baseline = validate_lineage(plan, execution)
        extra = {
            "src/openjev/research/reacher_latent_consistency.py", "tests/test_reacher_latent_consistency.py",
            "src/openjev/research/reacher_raw_endpoint.py", "tests/test_reacher_raw_endpoint.py",
            "src/openjev/research/reacher_objective_protocol.py", "tests/test_reacher_objective_protocol.py",
            "src/openjev/research/reacher_objective_training.py", "tests/test_reacher_objective_training.py",
            "scripts/reacher_objective_study.py", "tests/test_reacher_objective_study.py",
            "scripts/audit_reacher_objective_study.py", "tests/test_audit_reacher_objective_study.py"}
        base.require(set(plan["sources"]) == set(baseline["sources"]) | extra, "Exact frozen source membership")
        streams = validate_streams(plan, baseline, execution)
        train = base.load_records(execution / "train", plan["train_episodes"])
        replay = {"inherited_training": audit_inherited_training(cohort_plan, train)}
        check_budget()
        initializations = audit_initializations(plan, execution)
        orders = audit_orders(plan, execution)
        calibration = audit_calibration(plan, execution, initializations)
        fits = {}
        for row in protocol.fit_manifest(plan):
            multiplier = 0. if row["arm"] == "anchor" else calibration["multipliers"][row["arm"]]
            fits[row["name"]] = audit_fit(plan, execution, row, initializations[row["pair"]], orders[row["pair"]],
                                           multiplier, train)
        boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed, fits)
        prediction = base.load_records(execution / "prediction", plan["prediction_episodes"])
        replay["prediction"] = audit_cohort(plan, prediction, "prediction", "ordinary")
        predictions = {}
        for name in plan["fit_order"]:
            check_budget()
            predictions[name] = {
                **prediction_metrics(plan, prediction, base.load_npz(execution / "predictions" / f"{name}.npz")),
                "timing": audit_prediction_timing(plan, base.read(execution / "predictions" / f"{name}.json"))}
        inputs = [search_audit.audit_innovations(plan, execution / "innovations" / "control" / f"{step:03d}",
                   f"planner/control/{step}", plan["control_episodes"]) for step in range(plan["steps"])]
        controls, states, disturbances = {panel: {} for panel in protocol.PANELS}, None, None
        for row in protocol.execution_order(plan):
            check_budget()
            folder = execution / row["path"]
            records = base.load_records(folder / "episodes", plan["control_episodes"])
            replay[row["path"]] = audit_cohort(plan, records, "control", row["panel"])
            current = base.stack(records, "audit", "integration_state")[:, 0]
            noise = base.stack(records, "audit", "actuator_noise")
            if states is None:
                states, disturbances = current, noise
            base.require(np.array_equal(states, current) and np.array_equal(disturbances, noise),
                         "All intact/reset/objective control cases share reset and exogenous noise")
            control = search_audit.audit_control({**plan, "planners": ["cem256"]}, folder, row, records, inputs)
            if "fit" in row:
                control["reset_intervention"] = audit_reset_events(folder / "reset-events.npz", records,
                                                                    enabled=row["reset"])
            controls[row["panel"]][row["label"]] = control
        gate, differences = control_qualification(plan, controls)
        exact_integer(len(gate["checks"]), 31, "All prespecified continuation checks")
        costs = audit_costs(plan, execution, completed, initializations, calibration, fits, boundary, predictions, controls)
        base.require(validate_members(plan, expected_plan_sha256, execution) == completed
                     and base.sha(execution / "completed.json") == completion_hash, "Execution changed during audit")
        check_budget()
        summary = {
            "status": "completed", "version": VERSION, "engineering": engineering,
            "plan_sha256": expected_plan_sha256, "execution_completed_sha256": completion_hash,
            "saved_output_only": True, "new_model_calls": 0, "new_policy_calls": 0, "new_fits": 0,
            "native_transitions_checked": sum(row["transitions"] for row in replay.values()),
            "native_max_abs_error": max(row["max_abs_error"] for row in replay.values()),
            "coverage": {"fits": len(fits), "control_rows": sum(map(len, controls.values())),
                         "control_episodes_per_row": plan["control_episodes"], "prediction_episodes": len(prediction)},
            "random_streams": streams, "calibration": calibration, "fits": fits, "phase_boundary": boundary,
            "control": controls, "prediction": predictions, "continuation_gate": gate,
            "paired_descriptive_comparisons": paired_comparisons(plan, differences), "cohorts": replay,
            "costs": {**costs, "audit_validation_wall_seconds": time.monotonic() - begin},
            "limits": [
                "Saved gradient vectors establish norm and multiplier arithmetic, not their neural forward/backward origin.",
                "First/last EMA witnesses are checked numerically. Every other optimizer/EMA transition is source-bound logged evidence, not independently rerun training.",
                "Checkpoint tensor/counter restoration is authenticated by receipts and synthetic roundtrip tests; the saved-output audit does not instantiate the learned model.",
                "All nine fits are retained. Bootstrap intervals condition on three paired initializations and one shared corpus.",
                "Reset penalties establish intervention sensitivity; they do not alone prove useful memory versus a trained current-observation comparator.",
                "Auxiliary capacity/update matching is not FLOP, gradient-path or wall-time matching. Teacher, decoder and collapse diagnostics are paid training work.",
                "Native replay validates stored actions/outcomes; CEM proposals reconstruct from recorded scores without independently rerunning neural scoring.",
                "This is one development environment. Neither lower auxiliary loss nor this gate establishes JEPA reproduction, biological superiority or architecture novelty."]}
        out.mkdir(parents=True, exist_ok=False)
        base.write(out / "summary.json", summary)
        text = (f"# Reacher objective saved-output audit\n\n"
                f"Continuation: **{'PASS' if gate['passed'] else 'FAIL'}** "
                f"({sum(row['passed'] for row in gate['checks'])}/31 checks).\n\n"
                f"All nine fits and 57 control rows retained; {summary['native_transitions_checked']:,} native transitions replayed.\n\n"
                "Calibration vector norms and first/last EMA arithmetic are independently reconstructed. "
                "Neural forward/backward and every optimizer transition are not rerun. See summary.json for all limits.\n")
        (out / "README.md").write_text(text)
        receipt = {"status": "completed", "version": VERSION, "engineering": engineering,
                   "plan_sha256": expected_plan_sha256, "source_sha256": plan["sources"], "runtime": plan["runtime"],
                   "execution_completed_sha256": completion_hash, "execution_members": completed["files"],
                   "saved_output_only": True, "costs": summary["costs"],
                   "files": {name: base.sha(out / name) for name in ("summary.json", "README.md")}}
        check_budget()
        base.write(out / "receipt.json", receipt)
        check_budget()
        return summary
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "receipt.json").exists():
            (out / "receipt.json").rename(out / "over-cap-receipt.json")
        base.write(out / "failed.json", {"status": "failed", "error": repr(error),
                    "plan_sha256": expected_plan_sha256, "wall_seconds": time.monotonic() - begin,
                    "saved_output_only": True})
        raise
    finally:
        _DEADLINE.reset(token)


def audit_prediction_timing(plan, meta):
    n, steps = plan["prediction_episodes"], plan["steps"]
    maximum = max(plan["auxiliary_horizons"])
    calls = {"student_assimilations": n * steps, "prefix_advances": n * steps,
             "imagined_advances": n * sum(min(maximum, steps - t) for t in range(steps)),
             "teacher_or_auxiliary_calls": 0}
    base.require(set(meta) == {"wall_seconds", *calls}, "Prediction cost schema")
    for name, expected in calls.items():
        exact_integer(meta[name], expected, f"Prediction call accounting: {name}")
    return {"wall_seconds": finite(meta["wall_seconds"], "Prediction wall", positive=True), **calls}


def paired_comparisons(plan, differences):
    seed = protocol.seed(plan, "analysis/bootstrap/0")
    result = {}
    for panel, rows in differences.items():
        result[panel] = {}
        for name, values in rows.items():
            value = np.asarray(values, dtype=float)
            generator = np.random.default_rng(seed)
            indices = generator.integers(0, len(value), (plan["bootstrap_samples"], len(value)))
            interval = np.quantile(value[indices].mean(1), [.025, .975])
            result[panel][name] = {"mean_cost_difference": float(value.mean()),
                                  "episode_paired_percentile_95": interval.tolist(),
                                  "cases": len(value), "conditional_on_saved_fits": True}
    return result


def audit_plan(path, expected_sha256, execution, out):
    """Authenticate an externally frozen plan before production-only replay."""
    out = Path(out)
    base.require(not out.exists(), "Exclusive objective audit output required")
    try:
        plan = base.read(checked(path, expected_sha256))
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=False)
        base.write(out / "failed.json", {"status": "failed", "error": repr(error),
                    "plan_sha256": expected_sha256, "stage": "plan_authentication",
                    "saved_output_only": True})
        raise
    return audit_saved(plan, expected_sha256, execution, out, engineering=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_plan(args.plan, args.expected_plan_sha256, args.execution, args.out)
    print(json.dumps({"status": summary["status"], "coverage": summary["coverage"],
                      "continuation_passed": summary["continuation_gate"]["passed"]}))


if __name__ == "__main__":
    main()
