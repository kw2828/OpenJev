"""Independent saved-output audit for the frozen-transition reward intervention.

No runner, trainer or learned inference is invoked. Geometry is independently
reconstructed in NumPy. Search replays recorded scalar scores; native replay
executes saved actions. Protocol imports only define their model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextvars import ContextVar
from itertools import pairwise
from pathlib import Path

import audit_reacher_cache_study as inherited
import numpy as np
import torch

from openjev.research import reacher_geometry_protocol as protocol

base, search_audit = inherited.base, inherited.search_audit
require, read, write, sha = base.require, base.read, base.write, base.sha
array = search_audit.array
ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-geometry-score-v1"
PARENT_PLAN_SHA = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"
PARENT_AUDIT_SHA = "d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790"
FAMILIES = ("residual_gru", "cached_mlp")
MODES = ("learned", "geometry")
PANELS = ("full", "ordinary", "shift")
PAIRS = ("pair0", "pair1", "pair2")
MIN_PAIR_NORM = 1e-6
GEOMETRY_FIELDS = ("joint_angles", "pair_norms", "fingertip", "distance", "action_cost")
NEW_SOURCES = {f"{folder}/{name}.py" for folder, names in {
    "src/openjev/research": ("reacher_geometry_reward", "reacher_geometry_control", "reacher_geometry_protocol"),
    "scripts": ("reacher_geometry_study", "audit_reacher_geometry_study"),
    "tests": ("test_reacher_geometry_reward", "test_reacher_geometry_control", "test_reacher_geometry_protocol",
              "test_reacher_geometry_study", "test_audit_reacher_geometry_study"),
}.items() for name in names}
finite, integer = inherited.finite, inherited.exact_integer
tensor_hash, state_hash = inherited.canonical_tensor_hash, inherited.canonical_state_hash
_DEADLINE = ContextVar("geometry_audit_deadline", default=float("inf"))


def check_budget():
    if time.monotonic() > _DEADLINE.get():
        raise TimeoutError("Frozen geometry audit cap exceeded")


def checked(path, digest):
    check_budget()
    return search_audit.checked(Path(path), digest)


def expected_action_cost(commands, noise_std):
    """Independent normal partial moments, with the original float32 rounding.

    No Torch scoring helper is called. The study fixes sigma=.05, but zero and
    other finite scales are handled without a Monte Carlo approximation.
    """
    require(isinstance(commands, np.ndarray) and commands.dtype in (np.float32, np.float64)
            and commands.ndim >= 1 and commands.shape[-1] == 2 and commands.size > 0
            and np.isfinite(commands).all() and (np.abs(commands) <= 1).all(),
            "Finite in-range floating issued commands")
    require(type(noise_std) in (int, float) and math.isfinite(noise_std) and noise_std >= 0,
            "Fixed finite nonnegative noise scale")
    u, sigma = commands.astype(np.float64), float(noise_std)
    if sigma < math.sqrt(np.finfo(np.float64).tiny):
        per_action = u * u
    elif sigma > 8:
        inverse = 1.0 / sigma
        coefficient = inverse / math.sqrt(2 * math.pi)
        correction = np.zeros_like(u)
        for order in range(9):
            moment = sum(math.comb(2 * order, power) * 4.0 / ((power + 1) * (power + 3))
                         * u ** (2 * order - power) for power in range(0, 2 * order + 1, 2))
            correction += coefficient * moment
            coefficient *= -(inverse * inverse) / (2 * (order + 1))
        per_action = 1 - correction
    else:
        lower = np.clip((-1 - u) / sigma, -38, 38)
        upper = np.clip((1 - u) / sigma, -38, 38)
        erf = np.vectorize(math.erf, otypes=[np.float64])
        mass = .5 * (erf(upper / math.sqrt(2)) - erf(lower / math.sqrt(2)))
        phi_l = np.exp(-.5 * lower**2) / math.sqrt(2 * math.pi)
        phi_u = np.exp(-.5 * upper**2) / math.sqrt(2 * math.pi)
        per_action = ((u**2 + sigma**2) * mass + 2 * u * sigma * (phi_l - phi_u)
                      + sigma**2 * (lower * phi_l - upper * phi_u) + 1 - mass)
    result = np.clip(per_action, 0, 1).sum(-1).astype(commands.dtype)
    require(np.isfinite(result).all(), "Finite expected actuator cost")
    return result


def geometry_components(predicted4, targets, commands, noise_std):
    """Approximate planar geometry; target joint refs cancel body offsets."""
    require(isinstance(predicted4, np.ndarray) and predicted4.dtype in (np.float32, np.float64)
            and predicted4.ndim >= 1 and predicted4.shape[-1] == 4 and predicted4.size > 0
            and np.isfinite(predicted4).all(), "Finite decoded angular features")
    leading = predicted4.shape[:-1]
    for value, name in ((targets, "Public targets"), (commands, "Issued commands")):
        array(value, (*leading, 2), predicted4.dtype, name)
    cosine, sine = predicted4[..., :2], predicted4[..., 2:]
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.hypot(cosine, sine)
    require(np.isfinite(norms).all() and (norms >= MIN_PAIR_NORM).all(),
            "Finite angle-pair norms >=1e-6; no scoring fallback")
    angles = np.arctan2(sine, cosine)
    q0, q01 = angles[..., 0], angles.sum(-1)
    l0, l1 = np.asarray(.10, dtype=predicted4.dtype), np.asarray(.11, dtype=predicted4.dtype)
    tip = np.stack((l0 * np.cos(q0) + l1 * np.cos(q01),
                    l0 * np.sin(q0) + l1 * np.sin(q01)), axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        distance = np.sqrt(np.sum((tip - targets)**2, axis=-1))
    cost = expected_action_cost(commands, noise_std)
    reward = -distance - cost
    require(np.isfinite(distance).all() and np.isfinite(reward).all(),
            "Finite derived geometry reward")
    return {"joint_angles": angles, "pair_norms": norms, "fingertip": tip,
            "distance": distance, "action_cost": cost, "reward": reward}


def close_geometry(actual, expected, label):
    array(actual, expected.shape, expected.dtype, label)
    # Float32 Torch/NumPy atan2, trigonometry and norm kernels may differ by a
    # few ULPs. Score-to-trace equality is checked separately without tolerance.
    tolerance = 5e-7 if expected.dtype == np.float32 else 2e-14
    require(np.allclose(actual, expected, rtol=2e-6 if expected.dtype == np.float32 else 2e-13,
                        atol=tolerance), label + " independent geometry arithmetic")


def audit_bank_arrays(plan, mode, values, commands, root_target):
    """Verify one bank's decoded quantities; never re-evaluate its model."""
    require(mode in MODES, "Declared reward scoring mode")
    require(commands.ndim == 4 and commands.shape[-1] == 2, "Bank command dimensions")
    n, k, h, _ = commands.shape
    array(commands, (n, k, h, 2), np.float32, "Bank issued commands")
    require(n > 0 and 0 < k <= 256 and h > 0 and (np.abs(commands) <= 1).all(),
            "Nonempty bounded command bank")
    array(root_target, (n, 2), np.float32, "Bank public root target")
    keys = {"commands", "root_target", "predicted_angles", "learned_rewards", "selected_rewards"}
    if mode == "geometry":
        keys |= {"geometry_" + name for name in GEOMETRY_FIELDS}
    require(set(values) == keys, "Exact bank scoring array membership")
    array(values["commands"], commands.shape, commands.dtype, "Recorded commands dtype/shape")
    array(values["root_target"], root_target.shape, root_target.dtype, "Recorded target dtype/shape")
    require(np.array_equal(values["commands"], commands)
            and np.array_equal(values["root_target"], root_target), "Scoring bank/root binding")
    predictions = array(values["predicted_angles"], (n, k, h, 4), np.float32, "Bank angle predictions")
    learned = array(values["learned_rewards"], (n, k, h), np.float32, "Original learned rewards")
    selected = array(values["selected_rewards"], (n, k, h), np.float32, "Chosen-mode rewards")
    if mode == "learned":
        require(np.array_equal(selected, learned), "Learned baseline reward changed")
        return {"candidate_evaluations": n * k, "imagined_transitions": n * k * h,
                "geometry_samples": 0, "minimum_pair_norm": None, "joint1_limit_violations": 0}
    target = np.broadcast_to(root_target[:, None, None], (n, k, h, 2))
    expected = geometry_components(predictions, target, commands, plan["noise_std"])
    for name in GEOMETRY_FIELDS:
        close_geometry(values["geometry_" + name], expected[name], name)
    close_geometry(selected, expected["reward"], "Chosen geometry reward")
    # Geometry residual and actuator term are each charged exactly once in the
    # recorded values, independently of the cross-library comparison above.
    require(np.array_equal(selected, -values["geometry_distance"] - values["geometry_action_cost"]),
            "Recorded geometry distance and actuator cost charged exactly once")
    return {"candidate_evaluations": n * k, "imagined_transitions": n * k * h,
            "geometry_samples": n * k * h,
            "minimum_pair_norm": float(values["geometry_pair_norms"].min()),
            "joint1_limit_violations": int((np.abs(values["geometry_joint_angles"][..., 1]) > 3).sum())}


def first_occurrence_union(slots):
    """Canonical byte-exact union preserving every selector's original slot."""
    require(isinstance(slots, np.ndarray) and slots.dtype == np.float32 and slots.ndim == 3
            and slots.shape[-1] == 2 and len(slots) > 0 and np.isfinite(slots).all(),
            "Finite diagnostic command slots")
    unique, mapping, seen = [], [], {}
    for sequence in slots:
        identity = np.ascontiguousarray(sequence).tobytes()
        if identity not in seen:
            seen[identity] = len(unique)
            unique.append(sequence.copy())
        mapping.append(seen[identity])
    return np.stack(unique), np.asarray(mapping, dtype=np.int64)


def qualification(plan, controls):
    """Recompute all25 checks from complete saved per-case native costs."""
    expected = {f"{family}-{pair}--{mode}" for family in FAMILIES for pair in PAIRS for mode in MODES}
    references = {"known_state", "particle", "zero", "uniform", "public_kinematic"}
    require(set(controls) == set(PANELS), "All three geometry control panels")
    for panel in PANELS:
        require(set(controls[panel]) == expected | references, "All twelve learned rows plus five references")
        for row in controls[panel].values():
            values = np.asarray(row["episode_costs"], dtype=float)
            require(values.shape == (plan["control_episodes"],) and np.isfinite(values).all()
                    and math.isclose(float(values.mean()), row["mean_cost"], rel_tol=1e-12, abs_tol=1e-12),
                    "Complete paired native cost arithmetic")
    def mean(panel, family, mode):
        return np.mean([controls[panel][f"{family}-{pair}--{mode}"]["episode_costs"] for pair in PAIRS], axis=0)
    checks = []
    def add(name, left, right):
        checks.append({"name": name, "left": float(left), "right": float(right),
                       "comparison": "le", "passed": bool(left <= right)})
    for panel in ("ordinary", "shift"):
        add(f"{panel}: GRU geometry mean improves learned by 3%",
            mean(panel, "residual_gru", "geometry").mean(), .97 * mean(panel, "residual_gru", "learned").mean())
        for pair in PAIRS:
            add(f"{panel}/{pair}: GRU geometry nonworse than learned",
                controls[panel][f"residual_gru-{pair}--geometry"]["mean_cost"],
                controls[panel][f"residual_gru-{pair}--learned"]["mean_cost"])
        zero = controls[panel]["zero"]["mean_cost"]
        for mode in MODES:
            for pair in PAIRS:
                add(f"{panel}/{pair}/{mode}: GRU beats zero by 10%",
                    controls[panel][f"residual_gru-{pair}--{mode}"]["mean_cost"], .9 * zero)
        for reference in ("known_state", "particle"):
            add(f"{panel}/{reference}: physics beats zero by 10%",
                controls[panel][reference]["mean_cost"], .9 * zero)
    add("full: GRU geometry degrades learned at most 2%", mean("full", "residual_gru", "geometry").mean(),
        1.02 * mean("full", "residual_gru", "learned").mean())
    require(len(checks) == 25, "Exactly25 prospective checks")
    differences = {panel: {family: (mean(panel, family, "geometry") - mean(panel, family, "learned")).tolist()
                           for family in FAMILIES} for panel in PANELS}
    return {"passed": all(row["passed"] for row in checks), "checks": checks,
            "cached_mlp": "descriptive_only", "diagnostic": "exposed_roots_descriptive_only"}, differences


def inherited_members():
    result = {"train.npz", "train.json"}
    for pair in PAIRS:
        result |= {f"{folder}/{pair}.{suffix}" for folder in ("initializations", "orders") for suffix in ("pt", "json")}
        for family in FAMILIES:
            result |= {f"fits/{family}-{pair}/{name}" for name in
                       ("initial-weights.pt", "weights.pt", "checkpoint.pt", "training.jsonl", "completed.json")}
    result |= {f"control/{panel}/residual_gru-pair0/episodes.{suffix}" for panel in PANELS for suffix in ("npz", "json")}
    return result


def validate_lineage(plan, execution):
    source = plan["parent_source"]
    require(set(source) == {"plan_path", "plan_sha256", "audit_path", "audit_receipt_sha256", "execution_path",
        "completed_sha256", "summary_sha256", "members", "prior_costs", "engineering", "training_weights_reused",
        "new_fits", "previous_scientific_gate_passed"}, "Exact inherited source contract")
    engineering = plan["engineering"]
    require(source["engineering"] is engineering and source["training_weights_reused"] is True
            and type(source["new_fits"]) is int and source["new_fits"] == 0,
            "Unchanged completed inherited weights")
    if not engineering:
        require(source["plan_sha256"] == PARENT_PLAN_SHA and source["audit_receipt_sha256"] == PARENT_AUDIT_SHA
                and source["previous_scientific_gate_passed"] is False, "Pinned failed parent scientific gate")
    else:
        require(source["previous_scientific_gate_passed"] is None, "Engineering is not scientific qualification")
    parent = read(checked(ROOT / source["plan_path"], source["plan_sha256"]))
    audit = read(checked(ROOT / source["audit_path"], source["audit_receipt_sha256"]))
    completed = read(checked(ROOT / source["execution_path"] / "completed.json", source["completed_sha256"]))
    summary_path = (ROOT / source["audit_path"]).parent / "summary.json"
    summary = read(checked(summary_path, source["summary_sha256"]))
    require(audit["status"] == completed["status"] == summary["status"] == "completed"
            and audit["saved_output_only"] is True
            and audit["engineering"] is parent["engineering"] is summary["engineering"] is engineering
            and audit["plan_sha256"] == completed["plan_sha256"] == summary["plan_sha256"] == source["plan_sha256"]
            and audit["execution_completed_sha256"] == source["completed_sha256"]
            and audit["execution_members"] == completed["files"]
            and audit["source_sha256"] == parent["sources"]
            and audit["files"]["summary.json"] == source["summary_sha256"]
            and source["prior_costs"] == audit["costs"], "Authenticated completed parent evidence")
    require(set(parent["sources"]) | NEW_SOURCES == set(plan["sources"])
            and len(parent["sources"]) == 70
            and all(plan["sources"][name] == digest for name, digest in parent["sources"].items()),
            "Exact 70 frozen sources plus ten intervention sources")
    require(all(plan[k] == parent[k] for k in ("hidden_size", "mlp_width", "dt", "noise_std", "residual_reward")),
            "Unchanged model scale and dynamics assumptions")
    if not engineering:
        gate = summary["continuation_gate"]
        require(gate["passed"] is False and len(gate["checks"]) == 28
                and sum(item["passed"] for item in gate["checks"]) == 15, "Historical failed 15/28 gate preserved")
    for name, digest in audit["files"].items():
        checked((ROOT / source["audit_path"]).parent / name, digest)
    require(set(source["members"]) == inherited_members(), "Exactly all selected checkpoint/history members")
    for name, digest in source["members"].items():
        require(completed["files"].get(name) == digest, "Inherited member was independently audited")
        checked(ROOT / source["execution_path"] / name, digest)
        checked(execution / "inherited" / name, digest)
    for name, digest in (("source-plan.json", source["plan_sha256"]), ("source-audit.json", source["audit_receipt_sha256"]),
                         ("source-completed.json", source["completed_sha256"]), ("source-summary.json", source["summary_sha256"])):
        checked(execution / "inherited" / name, digest)
    require(read(execution / "inheritance.json") == source, "Copied inheritance receipt")
    return parent, audit


def validate_runtime_sources(plan):
    check_budget()
    inherited.validate_runtime_sources(plan)
    check_budget()


def prior_streams():
    """Independently reconstruct the runner's authenticated historical chain."""
    cache_path = "evidence/reacher-cache-ablation-v1/protocol/plan.json"
    cache = read(checked(ROOT / cache_path, PARENT_PLAN_SHA))
    memory_lineage = cache["memory_source"]
    memory = read(checked(ROOT / memory_lineage["plan_path"], memory_lineage["plan_sha256"]))
    objective_lineage = memory["objective_source"]
    objective = read(checked(ROOT / objective_lineage["plan_path"], objective_lineage["plan_sha256"]))
    numpy_prior, torch_prior, descriptors = inherited.inherited_memory_priors(objective, objective_lineage)
    require(memory["random_stream_contract"] == inherited.memory_protocol.stream_contract(
        memory, numpy_prior, torch_prior, descriptors), "Authenticated memory registry reconstruction")
    for parent, lineage, label in ((memory, memory_lineage, "memory"), (cache, {"plan_path": cache_path,
            "plan_sha256": PARENT_PLAN_SHA}, "cache")):
        contract = parent["random_stream_contract"]
        numpy_prior.append(contract["registry"])
        torch_prior.append(contract["torch_registry"])
        descriptor = {"plan_path": lineage["plan_path"], "plan_sha256": lineage["plan_sha256"],
                      "registry_kind": label + "-scored"}
        if label == "cache":
            descriptor["namespace"] = contract.get("namespace")
        descriptors.append(descriptor)
        for item in contract["engineering_exclusions"]:
            numpy_prior.append(item["registry"])
            torch_prior.append(item["torch_registry"])
            descriptors.append({"plan_path": lineage["plan_path"], "plan_sha256": lineage["plan_sha256"],
                                "registry_kind": label + "-engineering", "namespace": item["namespace"]})
        if label == "memory":
            numpy_prior.append({"cache/engineering/fixture": 410})
            torch_prior.append({"cache/engineering/fixture": 410, "cache/engineering/restore": 0})
            descriptors.append({"registry_kind": "cache-engineering-literal", "numpy_seeds": [410], "torch_seeds": [0, 410]})
            require(cache["random_stream_contract"] == inherited.protocol.stream_contract(
                cache, numpy_prior, torch_prior, descriptors), "Authenticated cache registry reconstruction")
    return numpy_prior, torch_prior, descriptors


def validate_streams(plan, execution):
    expected = protocol.stream_contract(plan, *prior_streams())
    require(plan["random_stream_contract"] == expected and read(execution / "random-streams.json") == expected,
            "Complete fresh/prior/engineering random stream contract")
    require(expected["torch_registry"] == {} and expected["torch_generators"] == {}
            and expected["new_torch_scored_streams"] == 0
            and expected["training_and_stochastic_inference_rng_draws"] == 0, "No new training random streams")
    return {"numpy_generators": len(expected["generators"]), "torch_generators": 0,
            "draws_for_manifest": 0, "full_prior_namespaces_excluded": True}


def expected_members(plan):
    result = {"started.json", "random-streams.json", "inheritance.json", "all-models-restored.json",
              "evaluation-started.json", "diagnostic-completed.json", "control-completed.json",
              "final-models.json", "costs.json"}
    result |= {"inherited/" + name for name in inherited_members()}
    result |= {"inherited/source-" + name + ".json" for name in ("plan", "audit", "completed", "summary")}
    for fit in protocol.fit_manifest(plan):
        result |= {f"model-states/{fit['name']}-{phase}.pt" for phase in ("before", "after")}
    for step in range(plan["steps"]):
        result |= {f"innovations/control/{step:03d}.{suffix}" for suffix in ("npz", "json")}
    for row in protocol.execution_order(plan):
        prefix = row["path"]
        result |= {prefix + "/" + name for name in ("episodes.npz", "episodes.json", "timings.json")}
        if "fit" in row:
            result |= {prefix + "/" + name for name in ("executed_predictions.npz", "states.npz", "state-work.json")}
            result |= {f"{prefix}/{folder}/{step:03d}.{suffix}" for folder in ("decisions", "scoring")
                       for step in range(plan["steps"]) for suffix in ("npz", "json")}
        else:
            result.add(prefix + "/planning.npz")
            if row["reference"] == "public_kinematic":
                result.add(prefix + "/observer-final.json")
    for row in protocol.diagnostic_manifest(plan):
        prefix = row["path"]
        result |= {prefix + "/" + name for name in ("innovations.npz", "innovations.json", "history.npz", "union.npz",
                                                    "native-root.npz", "root.json", "native.npz", "timings.json")}
        for fit in protocol.fit_manifest(plan):
            result.add(f"{prefix}/states/{fit['name']}.npz")
            result |= {f"{prefix}/{folder}/{fit['name']}--{mode}.{suffix}" for folder in ("search", "search-scoring", "scores")
                       for mode in MODES for suffix in ("npz", "json")}
    return result


def validate_members(plan, expected, execution):
    require(isinstance(expected, str) and len(expected) == 64 and set(expected) <= set("0123456789abcdef"), "External plan SHA")
    completed = read(execution / "completed.json")
    require(set(completed) == {"status", "study", "plan_sha256", "new_fits", "restored_models", "control_rows", "diagnostic_roots",
        "astra_calls", "wall_seconds", "evaluation_started_elapsed_seconds", "cumulative_attempt_wall_seconds", "files"}
        and completed["status"] == "completed" and completed["study"] == VERSION and completed["plan_sha256"] == expected,
        "Complete geometry execution identity")
    for key, wanted in (("new_fits", 0), ("restored_models", 6), ("control_rows", 51),
                        ("diagnostic_roots", len(protocol.diagnostic_manifest(plan))), ("astra_calls", 0)):
        integer(completed[key], wanted, "Completed " + key)
    require(finite(completed["wall_seconds"], "Whole execution seconds", positive=True) <= plan["cap_seconds"], "Frozen execution cap")
    paths = list(execution.rglob("*"))
    require(not execution.is_symlink() and not any(path.is_symlink() for path in paths), "No artifact symlinks")
    expected_files = expected_members(plan)
    require({path.relative_to(execution).as_posix() for path in paths if path.is_file()} == expected_files | {"completed.json"}
            and set(completed["files"]) == expected_files, "Exact completed files; no failed or partial artifacts")
    for name, digest in completed["files"].items():
        checked(execution / name, digest)
    return completed


def audit_restoration(plan, parent, execution):
    restored, final = read(execution / "all-models-restored.json"), read(execution / "final-models.json")
    require(set(restored) == {"models", "optimizer_constructed", "new_updates", "constructor_seed",
            "constructor_rng_isolated_and_weights_overwritten", "wall_seconds", "unix_time"}
            and restored["optimizer_constructed"] is False and restored["constructor_rng_isolated_and_weights_overwritten"] is True,
            "Deployment-only restoration boundary")
    integer(restored["new_updates"], 0, "No restoration updates")
    integer(restored["constructor_seed"], 0, "Isolated overwritten constructor seed")
    finite(restored["wall_seconds"], "Restoration wall seconds", positive=True)
    require(set(final) == {"student_tensor_sha256", "files", "new_updates"}, "Final model receipt schema")
    integer(final["new_updates"], 0, "No new optimizer updates")
    names = {row["name"] for row in protocol.fit_manifest(plan)}
    require(set(restored["models"]) == set(final["student_tensor_sha256"]) == names
            and set(final["files"]) == {f"model-states/{name}-after.pt" for name in names}, "All six before/after model identities")
    fits = {}
    for row in protocol.fit_manifest(plan):
        name, arm = row["name"], row["arm"]
        folder = execution / "inherited" / "fits" / name
        done = read(folder / "completed.json")
        checkpoint = inherited.load_tensors(folder / "checkpoint.pt")
        inherited.unseal(checkpoint)
        weights = inherited.load_tensors(folder / "weights.pt")
        schema = inherited.shapes(parent, arm)
        inherited.tensors(weights, schema, "Inherited student parameters")
        digest = tensor_hash(weights)
        require(digest == done["student_tensor_sha256"] == tensor_hash(checkpoint["student_state"]), "Inherited deployment weights match checkpoint")
        config = inherited.model_configuration(parent, arm)
        updates = parent["epochs"] * (parent["train_episodes"] // parent["batch_size"])
        require(checkpoint["model_configuration"] == done["model_configuration"] == config
                and checkpoint["settings"] == done["settings"] == inherited.training_configuration(parent)
                and checkpoint["source_sha256"] == parent["sources"]
                and checkpoint["runtime"] == parent["runtime"] == plan["runtime"]
                and checkpoint["successful_updates"] == checkpoint["optimizer_steps"] == done["updates"] == updates
                and checkpoint["cursor"] == {"epoch": parent["epochs"], "batch": 0}
                and checkpoint["failed"] is False and checkpoint["kind"] == done["arm"] == arm
                and done["name"] == name and done["pair"] == row["pair"], "Completed inherited class/configuration/schedule")
        for phase in ("before", "after"):
            snapshot = execution / "model-states" / f"{name}-{phase}.pt"
            actual = inherited.load_tensors(snapshot)
            inherited.tensors(actual, schema, "Actual deployment snapshot parameters")
            require(all(torch.equal(weights[key], actual[key]) for key in weights) and tensor_hash(actual) == digest,
                    "Observed before/after tensors unchanged")
            if phase == "after":
                checked(snapshot, final["files"][snapshot.relative_to(execution).as_posix()])
        require(final["student_tensor_sha256"][name] == digest, "Final tensor identity")
        expected = {"kind": arm, "model_class": protocol.MODEL_CLASSES[arm], "checkpoint_sha256": sha(folder / "checkpoint.pt"),
                    "weights_sha256": sha(folder / "weights.pt"), "student_tensor_sha256": digest, "successful_updates": updates,
                    "configuration": config, "snapshot_path": f"model-states/{name}-before.pt",
                    "snapshot_sha256": sha(execution / "model-states" / f"{name}-before.pt")}
        require(restored["models"][name] == expected, "Exact class restoration receipt")
        fits[name] = {"arm": arm, "pair": row["pair"], "parameters": sum(value.numel() for value in weights.values()),
                      "student_tensor_sha256": digest, "inherited_optimizer_updates": updates,
                      "new_optimizer_updates": 0, "observed_before_after_equal": True, "model_configuration": config}
    return fits, restored


def np_state_hash(values):
    return tensor_hash({name: torch.from_numpy(value.copy()) for name, value in values.items()})


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def scoring_configuration(plan, mode):
    return {"version": "reacher-geometry-control-v1", "score_mode": mode,
        "reward_clipping": [-2.5, 0.0], "sum_dtype": "float32_sequential",
        "geometry": plan["score_contract"]["geometry"], "geometry_computed": mode == "geometry",
        "learned_head_work_retained": True,
        "joint_limit_diagnostic": "abs(projected q1) > 3 radians; counted only, no penalty",
        "timing_scope": "Geometry seconds time its function only; copies, validation and diagnostics remain charged in total search/decision/row wall time.",
        "run_status_authority": "enclosing protocol and execution receipts"}


def audit_bank_metadata(plan, mode, meta, values, root, *, standalone=False, fit=None):
    n, k, h, _ = values["commands"].shape
    count = n * k * h
    geometry_count = count if mode == "geometry" else 0
    required = {"score_mode", "root_sha256", "bank_shape", "bank_dtype", "bank_sha256", "completed_model_offsets",
        "completed_scored_offsets", "model_advance_attempted_samples", "model_advance_samples", "geometry_attempted_samples",
        "geometry_samples", "model_advance_seconds", "geometry_seconds", "bank_wall_seconds", "maximum_candidate_state_tensor_bytes",
        "minimum_pair_norm", "joint_limit_violation_samples", "terminal_state_sha256", "prefix_state_sha256",
        "geometry_diagnostics_omitted_in_learned_mode"}
    if standalone:
        required |= {"model", "configuration", "wall_seconds"}
    require(set(meta) == required, "Exact bank metadata schema")
    require(meta["score_mode"] == mode and meta["root_sha256"] == np_state_hash(root)
            and meta["bank_shape"] == [n, k, h, 2] and meta["bank_dtype"] == "float32"
            and meta["bank_sha256"] == hashlib.sha256(values["commands"].tobytes(order="C")).hexdigest(),
            "Bank mode/root/bytes binding")
    for name, expected in (("completed_model_offsets", h), ("completed_scored_offsets", h),
        ("model_advance_attempted_samples", count), ("model_advance_samples", count),
        ("geometry_attempted_samples", geometry_count), ("geometry_samples", geometry_count),
        ("maximum_candidate_state_tensor_bytes", sum(value.nbytes for value in root.values()) * k)):
        integer(meta[name], expected, "Bank work " + name)
    for name in ("model_advance_seconds", "geometry_seconds", "bank_wall_seconds"):
        finite(meta[name], name)
    require(meta["model_advance_seconds"] + meta["geometry_seconds"] <= meta["bank_wall_seconds"] + 1e-6,
            "Scoring work charged inside bank time")
    if mode == "learned":
        require(meta["minimum_pair_norm"] is None and meta["joint_limit_violation_samples"] is None
                and meta["geometry_seconds"] == 0, "Learned scorer makes no geometry calls")
    else:
        require(meta["minimum_pair_norm"] == float(values["geometry_pair_norms"].min()), "Minimum norm arithmetic")
        integer(meta["joint_limit_violation_samples"], int((np.abs(values["geometry_joint_angles"][..., 1]) > 3).sum()),
                "Joint-limit diagnostic only")
    require(meta["geometry_diagnostics_omitted_in_learned_mode"] is (mode == "learned")
            and isinstance(meta["prefix_state_sha256"], list) and len(meta["prefix_state_sha256"]) == h
            and all(valid_digest(value) for value in meta["prefix_state_sha256"])
            and meta["terminal_state_sha256"] == meta["prefix_state_sha256"][-1], "Complete branch prefix identities")
    if standalone:
        require(meta["configuration"] == scoring_configuration(plan, mode)
                and meta["model"] == inherited.model_identity(plan, fit["arm"], fit), "Standalone score class/configuration")
        require(finite(meta["wall_seconds"], "Standalone scoring time", positive=True) >= meta["bank_wall_seconds"], "Complete standalone time")
    return {"model_advance_samples": count, "geometry_samples": geometry_count,
            "geometry_seconds": meta["geometry_seconds"], "model_advance_seconds": meta["model_advance_seconds"]}


def audit_selected(plan, mode, selected, meta, root, carried, trace, executed_angles, executed_rewards):
    n = len(executed_rewards)
    common = {"selected_angles", "selected_learned_reward", "selected_reward"}
    require(set(selected) == common | ({"selected_geometry_" + key for key in GEOMETRY_FIELDS} if mode == "geometry" else set()),
            "Selected advance array schema")
    array(selected["selected_angles"], (n, 4), np.float32, "Selected angles")
    for key in ("selected_learned_reward", "selected_reward"):
        array(selected[key], (n,), np.float32, key)
    require(np.array_equal(selected["selected_angles"], executed_angles)
            and np.array_equal(selected["selected_learned_reward"], executed_rewards), "Executed predictions retain original learned head")
    result = trace["result"]
    if mode == "geometry":
        expected = geometry_components(selected["selected_angles"], root["packet"][:, 4:6], result.selected_actions, plan["noise_std"])
        for key in GEOMETRY_FIELDS:
            close_geometry(selected["selected_geometry_" + key], expected[key], "Selected " + key)
        close_geometry(selected["selected_reward"], expected["reward"], "Selected geometry reward")
        require(np.array_equal(selected["selected_reward"], -selected["selected_geometry_distance"] - selected["selected_geometry_action_cost"]),
                "Selected actuator cost paid exactly once")
    else:
        require(np.array_equal(selected["selected_reward"], executed_rewards), "Learned selected reward unchanged")
    fields = {"model_advance_attempted_samples", "model_advance_samples", "geometry_attempted_samples", "geometry_samples",
              "geometry_seconds", "root_sha256", "minimum_pair_norm", "joint_limit_violation_samples",
              "model_advance_seconds", "carried_sha256"}
    require(set(meta) == fields and meta["root_sha256"] == np_state_hash(root)
            and meta["carried_sha256"] == np_state_hash(carried), "Selected root/carried state identity")
    for key in ("model_advance_attempted_samples", "model_advance_samples"):
        integer(meta[key], n, "Selected one-action model work")
    for key in ("geometry_attempted_samples", "geometry_samples"):
        integer(meta[key], n if mode == "geometry" else 0, "Selected one-action geometry work")
    finite(meta["model_advance_seconds"], "Selected advance time", positive=True)
    finite(meta["geometry_seconds"], "Selected geometry time")
    require(meta["minimum_pair_norm"] == (float(selected["selected_geometry_pair_norms"].min()) if mode == "geometry" else None)
            and meta["joint_limit_violation_samples"] == (int((np.abs(selected["selected_geometry_joint_angles"][..., 1]) > 3).sum()) if mode == "geometry" else None),
            "Selected geometry diagnostics")
    if mode == "learned":
        require(meta["geometry_seconds"] == 0, "No selected baseline geometry call")


def audit_scoring_trace(plan, stem, mode, root, fit, trace, step, *, carried=None, executed_angles=None, executed_rewards=None):
    values, meta = base.load_npz(stem.with_suffix(".npz")), read(stem.with_suffix(".json"))
    result, raw = trace["result"], trace["raw_rewards"]
    commands = result.sequences
    # SearchResult carries compressed chunks in older versions; expanded
    # sequences are an explicit property in the frozen planner.
    n, k, h, _ = commands.shape
    require(k == 256, "All paid CEM candidates retained")
    common = {"commands", "root_target", "predicted_angles", "learned_rewards", "selected_rewards"}
    bank_keys = common | ({"geometry_" + key for key in GEOMETRY_FIELDS} if mode == "geometry" else set())
    selected_keys = {"selected_angles", "selected_learned_reward", "selected_reward"}
    if mode == "geometry":
        selected_keys |= {"selected_geometry_" + key for key in GEOMETRY_FIELDS}
    wanted = bank_keys | {"selected_ids", "selected_actions"} | (selected_keys if carried is not None else set())
    require(set(values) == wanted, "Exact search scoring arrays")
    counts = audit_bank_arrays(plan, mode, {key: values[key] for key in bank_keys}, commands, root["packet"][:, 4:6])
    require(np.array_equal(values["selected_rewards"], raw)
            and np.array_equal(result.scores, search_audit.clipped_returns(raw)), "Exact clipped sequential score-to-CEM binding")
    require(np.array_equal(array(values["selected_ids"], (n,), np.int64, "Saved global-best IDs"), result.selected_ids)
            and np.array_equal(array(values["selected_actions"], (n, 2), np.float32, "Saved first actions"), result.selected_actions),
            "Selected candidate identity")
    expected_keys = {"version", "model", "step", "score_mode", "configuration", "root_sha256", "callbacks", "work", "search_seconds"}
    if carried is not None:
        expected_keys.add("selected_advance")
    require(set(meta) == expected_keys and meta["version"] == "reacher-geometry-control-v1"
            and meta["model"] == inherited.model_identity(plan, fit["arm"], fit)
            and meta["step"] == step and meta["score_mode"] == mode
            and meta["configuration"] == scoring_configuration(plan, mode)
            and meta["root_sha256"] == np_state_hash(root) and len(meta["callbacks"]) == 4
            and meta["search_seconds"] == trace["search_seconds"], "Search scoring metadata identity")
    callback_costs = []
    for index, callback in enumerate(meta["callbacks"]):
        require(callback["candidate_start"] == index * 64 and callback["candidate_stop"] == (index + 1) * 64,
                "Complete paid callback intervals")
        part = {key: value if key == "root_target" else value[:, index * 64:(index + 1) * 64] for key, value in values.items() if key in bank_keys}
        callback_costs.append(audit_bank_metadata(plan, mode,
            {key: value for key, value in callback.items() if key not in ("candidate_start", "candidate_stop")}, part, root))
    work = meta["work"]
    require(set(work) == {"candidate_evaluations", "imagined_transitions", "root_tensor_bytes", "max_single_candidate_state_tensor_bytes",
        "tensor_bytes_are_not_peak_process_memory", "model_work", "geometry_samples", "geometry_seconds", "model_advance_seconds", "diagnostic_array_bytes"},
        "Complete search work accounting schema")
    integer(work["candidate_evaluations"], n * 256, "Paid candidate count")
    integer(work["imagined_transitions"], n * 256 * h, "Paid imagined transition count")
    integer(work["root_tensor_bytes"], sum(value.nbytes for value in root.values()), "Root state payload")
    integer(work["max_single_candidate_state_tensor_bytes"], work["root_tensor_bytes"] * 64, "Maximum paid branch payload")
    integer(work["geometry_samples"], counts["geometry_samples"], "Total search geometry work")
    require(work["tensor_bytes_are_not_peak_process_memory"] is True, "Payload versus resident memory distinction")
    inherited.audit_work(plan, fit["arm"], work["model_work"], 0, n * 256 * h)
    for key in ("geometry_seconds", "model_advance_seconds"):
        require(math.isclose(work[key], sum(row[key] for row in callback_costs), rel_tol=1e-12, abs_tol=1e-12), "Summed search time " + key)
    integer(work["diagnostic_array_bytes"], sum(value.nbytes for key, value in values.items() if key not in selected_keys), "Search array payload")
    require(sum(callback["bank_wall_seconds"] for callback in meta["callbacks"]) <= meta["search_seconds"] + 1e-6, "All callbacks charged")
    if carried is not None:
        audit_selected(plan, mode, {key: values[key] for key in selected_keys}, meta["selected_advance"], root, carried,
                       trace, executed_angles, executed_rewards)
        chosen = (np.arange(n), result.selected_ids, np.zeros(n, dtype=np.int64))
        for selected_key, bank_key in (("selected_angles", "predicted_angles"),
                                       ("selected_learned_reward", "learned_rewards"),
                                       ("selected_reward", "selected_rewards")):
            # Re-advance uses N roots rather than the N*64 search batch. CPU
            # affine reductions can differ slightly while sharing the action.
            require(np.allclose(values[selected_key], values[bank_key][chosen], rtol=2e-5, atol=2e-6),
                    "Selected re-advance agrees with chosen candidate first transition")
    return values, meta, counts


def audit_public_root(plan, state, history, arm):
    schema = inherited.state_shapes(plan, arm)
    require(set(state) == set(schema), "Exact public-root state fields")
    for name, shape in schema.items():
        array(state[name], (1, *shape), inherited.state_dtype(name), "Public-root " + name)
    packets, commands = history["packets"], history["commands"]
    require(packets.ndim == 2 and packets.shape[1] == 8 and commands.shape == (len(packets) - 1, 2)
            and packets.dtype == commands.dtype == np.float32 and np.isfinite(packets).all()
            and np.isfinite(commands).all() and np.isin(packets[:, 6], (0, 1)).all()
            and packets[0, 6] == 1 and (packets[:, :4][packets[:, 6] == 0] == 0).all(), "Causal sanitized public history")
    require(np.array_equal(state["packet"], packets[-1:]), "Root carries actual public packet")
    if arm == "cached_mlp":
        visible = np.flatnonzero(packets[:, 6] == 1)
        current, last = len(packets) - 1, int(visible[-1])
        expected = {"real_index": np.array([[current]], np.int64), "last_visible_index": np.array([[last]], np.int64),
                    "real_target": packets[-1:, 4:6], "imagined_depth": np.zeros((1, 1), np.int64),
                    "cached_angles": packets[last:last + 1, :4],
                    "encoder_features": np.concatenate((packets[last:last + 1, :4], packets[-1:, 4:]), axis=1)}
        require(all(np.array_equal(state[key], value) for key, value in expected.items()),
                "Root cache is reconstructed from actual prior valid measurements only")


def audit_diagnostic_root(plan, execution, row, source_episode, fits):
    folder, step = execution / row["path"], row["step"]
    meta = read(folder / "root.json")
    require(set(meta) == {"panel", "case_index", "step", "horizon", "source_controller", "source_episode_path", "identity_slots",
        "unique_sequence_count", "slot_ids", "search_order", "beliefs", "history_sha256", "deduplication", "scope"}, "Diagnostic root schema")
    require(meta["panel"] == row["panel"] and meta["case_index"] == row["case_index"] and meta["step"] == step
            and meta["horizon"] == 12 and meta["source_controller"] == "residual_gru-pair0"
            and meta["source_episode_path"] == f"inherited/control/{row['panel']}/residual_gru-pair0/episodes"
            and meta["deduplication"] == "first occurrence of exact float32 bytes"
            and meta["scope"] == "exposed prior development histories; native hidden state used for offline labels only", "Exposed prior root identity")
    history = base.load_npz(folder / "history.npz", {"packets", "commands"})
    for name, expected in {"packets": source_episode["policy"]["packets"][:step + 1],
                           "commands": source_episode["policy"]["commands"][:step]}.items():
        array(history[name], expected.shape, np.float32, "Causal history " + name)
        require(np.array_equal(history[name], expected), "Prefix excludes future data")
    require(meta["history_sha256"] == sha(folder / "history.npz") and set(meta["beliefs"]) == set(fits), "All six history reconstructions")
    inputs = search_audit.audit_innovations(plan, folder / "innovations", row["input_prefix"], 1)
    slots = list(protocol.common_bank(inputs, step, plan)[0].copy())
    slot_ids, search_order, traces, state_by_fit = [f"common/{i}" for i in range(64)], [], {}, {}
    scoring_cost = {"model_advance_samples": 0, "geometry_samples": 0, "geometry_seconds": 0., "model_advance_seconds": 0.}
    search_time, belief_time = 0., 0.
    for fit_row in protocol.fit_manifest(plan):
        name, fit = fit_row["name"], fits[fit_row["name"]]
        state = base.load_npz(folder / "states" / f"{name}.npz")
        audit_public_root(plan, state, history, fit["arm"])
        state_by_fit[name] = state
        belief = meta["beliefs"][name]
        require(set(belief) == {"history_sha256", "observation_assimilations", "history_action_advances", "state_sha256",
                              "model_tensor_sha256", "wall_seconds"}
                and belief["history_sha256"] == meta["history_sha256"]
                and belief["observation_assimilations"] == step + 1 and belief["history_action_advances"] == step
                and belief["state_sha256"] == np_state_hash(state)
                and belief["model_tensor_sha256"] == fit["student_tensor_sha256"], "Causal reconstruction work/identity")
        belief_time += finite(belief["wall_seconds"], "Root belief time", positive=True)
        mode_values, mode_meta = {}, {}
        for mode in MODES:
            label = protocol.policy_name(name, mode)
            trace = search_audit.audit_trace({**plan, "planners": ["cem256"]}, inputs, folder / "search" / label, step=step)
            values, score_meta, counts = audit_scoring_trace(plan, folder / "search-scoring" / label, mode, state, fit, trace, step)
            traces[label], mode_values[mode], mode_meta[mode] = trace, values, score_meta
            slots.append(trace["result"].selected_sequences[0].copy())
            slot_ids.append("selected/" + label)
            search_order.append({"fit": name, "score_mode": mode, "label": label, "slot_index": len(slots) - 1})
            search_time += trace["search_seconds"]
            scoring_cost["model_advance_samples"] += counts["imagined_transitions"]
            scoring_cost["geometry_samples"] += counts["geometry_samples"]
            for key in ("geometry_seconds", "model_advance_seconds"):
                scoring_cost[key] += score_meta["work"][key]
        for key in ("commands", "predicted_angles", "learned_rewards"):
            require(np.array_equal(mode_values["learned"][key][:, :64], mode_values["geometry"][key][:, :64]),
                    "Scoring modes preserve identical common-bank dynamics and original reward outputs")
        require(mode_meta["learned"]["callbacks"][0]["prefix_state_sha256"] == mode_meta["geometry"]["callbacks"][0]["prefix_state_sha256"],
                "Common-bank branch state unchanged by scoring mode")
    slots = np.stack(slots)
    unique, mapping = first_occurrence_union(slots)
    first = np.asarray([np.flatnonzero(mapping == i)[0] for i in range(len(unique))], dtype=np.int64)
    noise = protocol.diagnostic_noise(plan, row["panel"], row["case_index"], step)
    union = base.load_npz(folder / "union.npz", {"slot_commands", "commands", "slot_to_unique", "unique_first_slots", "noise"})
    for name, expected in {"slot_commands": slots, "commands": unique, "slot_to_unique": mapping, "unique_first_slots": first, "noise": noise}.items():
        array(union[name], expected.shape, expected.dtype, "Union " + name)
        require(np.array_equal(union[name], expected), "Canonical deduplicated union and fresh shared branch noise")
    require(meta["identity_slots"] == len(slots) == 76 and meta["unique_sequence_count"] == len(unique)
            and meta["slot_ids"] == slot_ids and meta["search_order"] == search_order, "All selection identities retained")
    root = base.load_npz(folder / "native-root.npz", {"integration_state"})
    expected_root = source_episode["audit"]["integration_state"][step]
    array(root["integration_state"], expected_root.shape, np.float64, "Native root integration state")
    require(np.array_equal(root["integration_state"], expected_root), "Authenticated exposed physical root")
    native = base.load_npz(folder / "native.npz")
    replay = search_audit.replay_branches(source_episode, step, unique, noise, native)
    check_budget()
    native_returns = native["rewards"].sum(-1)
    native_mean = native_returns.mean(1)
    metrics, union_times = {}, 0.
    for fit_row in protocol.fit_manifest(plan):
        name, fit, state = fit_row["name"], fits[fit_row["name"]], state_by_fit[fit_row["name"]]
        mode_values, mode_meta = {}, {}
        for mode in MODES:
            label = protocol.policy_name(name, mode)
            saved = base.load_npz(folder / "scores" / f"{label}.npz")
            scores = saved.pop("scores")
            array(scores, (1, len(unique)), np.float32, "All-union clipped scores")
            counts = audit_bank_arrays(plan, mode, saved, unique[None], state["packet"][:, 4:6])
            require(np.array_equal(scores, search_audit.clipped_returns(saved["selected_rewards"]).astype(np.float32)), "All-union score arithmetic")
            score_meta = read(folder / "scores" / f"{label}.json")
            work = audit_bank_metadata(plan, mode, score_meta, saved, state, standalone=True, fit=fit)
            for key in scoring_cost:
                scoring_cost[key] += work[key]
            union_times += score_meta["wall_seconds"]
            mode_values[mode], mode_meta[mode] = saved, score_meta
            slot = slot_ids.index("selected/" + label)
            selected = mapping[slot]
            predicted_raw = saved["selected_rewards"][0].astype(np.float64).sum(-1)
            metrics[label] = {"common_bank_spearman": search_audit.rank_agreement(scores[0, mapping[:64]], native_mean[mapping[:64]]),
                "union_spearman": search_audit.rank_agreement(scores[0], native_mean),
                "native_selected_return": float(native_mean[selected]),
                "native_selected_branch_returns": native_returns[selected].tolist(),
                "finite_union_regret": float(native_mean.max() - native_mean[selected]),
                "union_rescore_best_native_return": float(native_mean[int(scores[0].argmax())]),
                "common_bank_raw_return_mse": float(np.mean((predicted_raw[mapping[:64]] - native_mean[mapping[:64]]) ** 2)),
                "common_bank_clipped_return_mse": float(np.mean((scores[0, mapping[:64]] - native_mean[mapping[:64]]) ** 2)),
                "selected_raw_prediction_bias": float(predicted_raw[selected] - native_mean[selected]),
                "selected_clipped_prediction_bias": float(scores[0, selected] - native_mean[selected]),
                "identity_slots": 76, "unique_sequences": len(unique),
                "scope": "Exposed development root and finite union only; four common noise labels, not optimal control."}
            # Identical sequence scored at different batch sizes can differ by
            # CPU affine reduction order. This is not claimed as bitwise parity.
            trace = traces[label]
            require(np.allclose(saved["selected_rewards"][0, selected], trace["raw_rewards"][0, trace["result"].selected_ids[0]],
                                rtol=2e-5, atol=2e-6), "Selected sequence retains score under union rescore")
            require(counts["imagined_transitions"] == work["model_advance_samples"], "Union sample accounting")
        for key in ("commands", "root_target", "predicted_angles", "learned_rewards"):
            require(np.array_equal(mode_values["learned"][key], mode_values["geometry"][key]), "All-union scoring modes share identical dynamics")
        require(mode_meta["learned"]["prefix_state_sha256"] == mode_meta["geometry"]["prefix_state_sha256"], "All-union mode-independent latent branches")
    timing = read(folder / "timings.json")
    require(set(timing) == {"belief_seconds", "search_seconds", "union_score_seconds", "native_and_storage_seconds", "row_wall_seconds",
        "identity_slots", "unique_sequences", "native_transitions", "observation_assimilations", "history_action_advances"}, "Diagnostic cost schema")
    for key in ("belief_seconds", "search_seconds", "union_score_seconds", "native_and_storage_seconds", "row_wall_seconds"):
        finite(timing[key], key, positive=True)
    require(math.isclose(timing["belief_seconds"], belief_time, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(timing["search_seconds"], search_time, rel_tol=1e-12, abs_tol=1e-12)
            and union_times <= timing["union_score_seconds"] + 1e-6
            and sum(timing[key] for key in ("belief_seconds", "search_seconds", "union_score_seconds", "native_and_storage_seconds"))
                <= timing["row_wall_seconds"] + 1e-6, "All diagnostic work charged")
    for key, expected in (("identity_slots", 76), ("unique_sequences", len(unique)), ("native_transitions", replay["transitions"]),
                          ("observation_assimilations", 6 * (step + 1)), ("history_action_advances", 6 * step)):
        integer(timing[key], expected, "Diagnostic " + key)
    return {"path": row["path"], "panel": row["panel"], "case_index": row["case_index"], "step": step,
            "metrics": metrics, "native_replay": replay, "timing": timing, "scoring_work": scoring_cost}


def audit_cohort(plan, records, panel):
    require(len(records) == plan["control_episodes"], "All fresh control cases")
    total, maximum = 0, 0.
    for index, record in enumerate(records):
        check_budget()
        meta = record["metadata"]
        require(meta["seed"] == protocol.seed(plan, f"control/reset/{index}")
                and meta["noise_seed"] == protocol.seed(plan, f"control/actuator_noise/{index}")
                and meta["noise_std"] == plan["noise_std"] and meta["policy_keys"] == ["packets", "commands"]
                and meta["sensor_schedule"] == protocol.schedule(plan, index, panel).tolist(), "Fresh control reset/noise/public sensor binding")
        replay = search_audit.native.native_replay(record)
        require(replay["transitions"] == plan["steps"] and replay["new_policy_calls"] == 0 and replay["saved_output_only"] is True,
                "Native replay scope and terminal boundary")
        total += replay["transitions"]
        maximum = max(maximum, replay["max_abs_error"])
    return {"episodes": len(records), "transitions": total, "max_abs_error": maximum}


def audit_learned_control(plan, folder, row, records, inputs, fit):
    n, steps, mode = plan["control_episodes"], plan["steps"], row["score_mode"]
    times = search_audit.timing_row(plan, folder / "timings.json", True)
    state_work = inherited.audit_states(plan, folder, records, fit["arm"], fit)
    states = base.load_npz(folder / "states.npz")
    predictions = base.load_npz(folder / "executed_predictions.npz", {"angles", "rewards"})
    angle = array(predictions["angles"], (n, steps, 4), np.float32, "Executed native-head angles")
    reward = array(predictions["rewards"], (n, steps), np.float32, "Executed original learned reward")
    commands, native_rewards = base.stack(records, "policy", "commands"), base.stack(records, "audit", "rewards")
    costs = -native_rewards.sum(1)
    work = {"candidate_evaluations": 0, "imagined_transitions": 0, "geometry_samples": 0,
            "geometry_seconds": 0., "model_advance_seconds": 0., "selected_geometry_samples": 0,
            "selected_geometry_seconds": 0., "selected_model_advance_seconds": 0.}
    searches, clipped, count = [], 0, 0
    selected_rewards = np.zeros((n, steps), np.float32)
    schema = inherited.state_shapes(plan, fit["arm"])
    for step in range(steps):
        check_budget()
        trace = search_audit.audit_trace({**plan, "planners": ["cem256"]}, inputs[step], folder / "decisions" / f"{step:03d}", step=step)
        require(np.array_equal(commands[:, step], trace["result"].selected_actions), "Executed action is recorded CEM global best")
        root = {name: states["root__" + name][:, step] for name in schema}
        carried = {name: states["carried__" + name][:, step] for name in schema}
        values, meta, counts = audit_scoring_trace(plan, folder / "scoring" / f"{step:03d}", mode, root, fit, trace, step,
            carried=carried, executed_angles=angle[:, step], executed_rewards=reward[:, step])
        require(trace["search_seconds"] + meta["selected_advance"]["model_advance_seconds"]
                + meta["selected_advance"]["geometry_seconds"] <= times["decision_seconds"][step] + 1e-6,
                "Search plus selected action computation charged to decision")
        searches.append(trace["search_seconds"])
        for key in ("candidate_evaluations", "imagined_transitions", "geometry_samples"):
            work[key] += counts[key]
        for key in ("geometry_seconds", "model_advance_seconds"):
            work[key] += meta["work"][key]
        for key in ("geometry_samples", "geometry_seconds", "model_advance_seconds"):
            work["selected_" + key] += meta["selected_advance"][key]
        selected_rewards[:, step] = values["selected_reward"]
        clipped += trace["clipped_predictions"]
        count += trace["reward_predictions"]
    decisions = np.asarray(times["decision_seconds"], dtype=float)
    valid = base.stack(records, "policy", "packets")[:, 1:, 6].astype(bool)
    truth = base.stack(records, "audit", "raw_obs")[:, 1:, :4]
    return {"episode_costs": costs.tolist(), "mean_cost": float(costs.mean()),
        "setup_seconds": times["setup_seconds"], "decision_wall_seconds": float(decisions.sum()),
        "native_step_seconds": float(sum(times["native_step_seconds"])), "row_wall_seconds": times["row_wall_seconds"],
        "decision_seconds": times["decision_seconds"], "search_seconds": searches,
        "batch_latency_seconds": {"mean": float(decisions.mean()), "p50": float(np.quantile(decisions, .5)),
            "p95": float(np.quantile(decisions, .95)), "max": float(decisions.max())},
        "per_case_amortized_seconds": float(decisions.mean() / n), "candidate_evaluations": work["candidate_evaluations"],
        "imagined_transitions": work["imagined_transitions"], "clipping_fraction": clipped / count, "planner_used": True,
        "on_policy_angle_mse": base.masked_mse(angle, truth, np.ones_like(valid)),
        "on_policy_valid_angle_mse": base.masked_mse(angle, truth, valid),
        "on_policy_blackout_angle_mse": base.masked_mse(angle, truth, ~valid) if (~valid).any() else None,
        "on_policy_original_learned_reward_mse": float(np.mean((reward.astype(float) - native_rewards) ** 2)),
        "on_policy_selected_reward_mse": float(np.mean((selected_rewards.astype(float) - native_rewards) ** 2)),
        "state_and_work": state_work, "scoring_work": work,
        "reward_error_limit": "Different closed-loop states/actions across scorers; not a common-input causal reward-quality comparison."}


def audit_boundaries(plan, expected, execution, completed, restored, diagnostics):
    started, evaluation, diagnostic, control = (read(execution / name) for name in
        ("started.json", "evaluation-started.json", "diagnostic-completed.json", "control-completed.json"))
    require(set(started) == {"plan_sha256", "unix_time"}
            and set(evaluation) == {"plan_sha256", "all_models_restored_sha256", "unix_time", "elapsed_seconds"}
            and set(diagnostic) == {"roots", "unix_time", "native_transitions", "timings", "plan_sha256"}
            and set(control) == {"rows", "unix_time", "plan_sha256", "diagnostic_completed_sha256"}, "Exact execution phase receipt fields")
    require(all(row["plan_sha256"] == expected for row in (started, evaluation, diagnostic, control))
            and evaluation["all_models_restored_sha256"] == sha(execution / "all-models-restored.json")
            and control["diagnostic_completed_sha256"] == sha(execution / "diagnostic-completed.json"), "Hash-bound phase order")
    timestamps = [finite(row["unix_time"], "Phase timestamp", positive=True) for row in (started, restored, evaluation, diagnostic, control)]
    require(all(a <= b for a, b in pairwise(timestamps)), "Restore precedes exposed diagnostic precedes fresh control")
    require(evaluation["elapsed_seconds"] == completed["evaluation_started_elapsed_seconds"]
            and 0 < evaluation["elapsed_seconds"] <= completed["wall_seconds"]
            and restored["wall_seconds"] <= evaluation["elapsed_seconds"] + 1e-6, "Pre-evaluation cost boundary")
    require(diagnostic["roots"] == len(diagnostics) and control["rows"] == 51
            and diagnostic["timings"] == [row["timing"] for row in diagnostics]
            and diagnostic["native_transitions"] == sum(row["native_replay"]["transitions"] for row in diagnostics), "All stage work retained")
    return {"timestamps_monotonic": True, "restored_models_before_diagnostic": 6,
            "diagnostic_before_fresh_controls": True, "new_fits": 0, "new_optimizer_updates": 0,
            "pre_evaluation_wall_seconds": evaluation["elapsed_seconds"]}


def paired_comparisons(plan, controls, differences):
    result = {}
    for panel in PANELS:
        family = {f"{arm}/{mode}": np.asarray([controls[panel][f"{arm}-{pair}--{mode}"]["episode_costs"]
            for pair in PAIRS], dtype=np.float64) for arm in FAMILIES for mode in MODES}
        contrasts = {"residual_gru_geometry_minus_learned": ("residual_gru/geometry", "residual_gru/learned"),
            "cached_mlp_geometry_minus_learned": ("cached_mlp/geometry", "cached_mlp/learned"),
            "geometry_gru_minus_learned_cached_mlp": ("residual_gru/geometry", "cached_mlp/learned"),
            "geometry_gru_minus_geometry_cached_mlp": ("residual_gru/geometry", "cached_mlp/geometry")}
        result[panel] = {}
        for label, (a, b) in contrasts.items():
            diff = family[a] - family[b]
            cases = diff.mean(0)
            rng = np.random.default_rng(protocol.seed(plan, "analysis/bootstrap/0"))
            index = rng.integers(0, len(cases), (plan["bootstrap_samples"], len(cases)))
            result[panel][label] = {"mean_cost_difference": float(cases.mean()),
                "percent_cost_change": float(100 * cases.mean() / family[b].mean()),
                "paired_fit_cost_differences": diff.mean(1).tolist(),
                "paired_fit_percent_changes": (100 * diff.mean(1) / family[b].mean(1)).tolist(),
                "all_three_pairs_nonworse": bool((diff.mean(1) <= 0).all()),
                "episode_paired_percentile_95": np.quantile(cases[index].mean(1), [.025, .975]).tolist(),
                "cases": len(cases), "conditional_on_all_three_saved_fit_pairs": True,
                "scope": "Conditional descriptive interval; no training-seed population uncertainty or multiple-comparison correction."}
        for arm in FAMILIES:
            require(np.allclose(np.asarray(differences[panel][arm]), (family[f"{arm}/geometry"] - family[f"{arm}/learned"]).mean(0), rtol=1e-13, atol=1e-13),
                    "Qualification and reporting contrast arithmetic")
    return result


def audit_costs(plan, execution, completed, restored, diagnostics, controls):
    saved = read(execution / "costs.json")
    require(set(saved) == {"restore_wall_seconds", "new_fits", "new_optimizer_steps", "diagnostic_root_wall_seconds",
        "diagnostic_native_transitions", "innovation_generation_and_storage_seconds", "control_row_wall_seconds",
        "control_setup_seconds", "control_decision_seconds", "control_native_step_seconds", "control_native_transitions",
        "astra_calls", "prior_costs", "compute_matched", "accounting"}, "Complete cost schema")
    rows = [controls[row["panel"]][row["label"]] for row in protocol.execution_order(plan)]
    sums = {"restore_wall_seconds": restored["wall_seconds"],
            "diagnostic_root_wall_seconds": sum(row["timing"]["row_wall_seconds"] for row in diagnostics),
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in rows),
            "control_setup_seconds": sum(row["setup_seconds"] for row in rows),
            "control_decision_seconds": sum(row["decision_wall_seconds"] for row in rows),
            "control_native_step_seconds": sum(row["native_step_seconds"] for row in rows)}
    for key, expected in sums.items():
        finite(saved[key], key, positive=True)
        require(math.isclose(saved[key], expected, rel_tol=1e-12, abs_tol=1e-8), "Summed observed cost " + key)
    for key, expected in (("new_fits", 0), ("new_optimizer_steps", 0), ("astra_calls", 0),
        ("diagnostic_native_transitions", sum(row["native_replay"]["transitions"] for row in diagnostics)),
        ("control_native_transitions", len(rows) * plan["control_episodes"] * plan["steps"])):
        integer(saved[key], expected, "Cost scope " + key)
    require(saved["compute_matched"] is False and saved["prior_costs"] == plan["parent_source"]["prior_costs"]
            and isinstance(saved["accounting"], str) and saved["accounting"], "Cumulative accounting and timing limitations")
    innovation = finite(saved["innovation_generation_and_storage_seconds"], "Innovation storage seconds", positive=True)
    require(saved["restore_wall_seconds"] + saved["diagnostic_root_wall_seconds"] + saved["control_row_wall_seconds"] + innovation
            <= completed["wall_seconds"] + 1e-6, "Entire paid execution includes all phases")
    cumulative = saved["prior_costs"]["cumulative_attempt_wall_seconds"] + completed["wall_seconds"]
    require(math.isclose(completed["cumulative_attempt_wall_seconds"], cumulative, rel_tol=1e-12, abs_tol=1e-8), "Inherited cumulative wall cost")
    return {**saved, "execution_wall_seconds": completed["wall_seconds"], "cumulative_attempt_wall_seconds": cumulative,
            "learned_control_scoring": {key: sum(row["scoring_work"][key] for row in rows if "scoring_work" in row)
                for key in ("candidate_evaluations", "imagined_transitions", "geometry_samples", "geometry_seconds",
                            "model_advance_seconds", "selected_geometry_samples", "selected_geometry_seconds", "selected_model_advance_seconds")},
            "diagnostic_scoring": {key: sum(row["scoring_work"][key] for row in diagnostics)
                for key in ("model_advance_samples", "geometry_samples", "geometry_seconds", "model_advance_seconds")}}


def audit_saved(plan, expected_plan_sha256, execution, out, *, engineering=False):
    """Authenticate saved evidence, native replay and arithmetic; no inference."""
    execution, out = Path(execution), Path(out)
    require(not out.exists(), "Exclusive geometry audit output")
    begin = time.monotonic()
    require(type(plan["audit_cap_seconds"]) is int and plan["audit_cap_seconds"] > 0, "Positive frozen audit cap")
    token = _DEADLINE.set(begin + plan["audit_cap_seconds"])
    try:
        require(type(plan["cap_seconds"]) is int and plan["cap_seconds"] > 0, "Positive frozen execution cap")
        protocol.validate_settings(plan, engineering=engineering)
        require(plan["engineering"] is engineering, "Explicit engineering boundary")
        validate_runtime_sources(plan)
        completed = validate_members(plan, expected_plan_sha256, execution)
        completion_hash = sha(execution / "completed.json")
        parent, _ = validate_lineage(plan, execution)
        streams = validate_streams(plan, execution)
        fits, restored = audit_restoration(plan, parent, execution)
        inherited_episodes = {panel: base.load_records(execution / "inherited" / "control" / panel / "residual_gru-pair0" / "episodes",
                                                     parent["control_episodes"]) for panel in PANELS}
        diagnostics = []
        for row in protocol.diagnostic_manifest(plan):
            check_budget()
            diagnostics.append(audit_diagnostic_root(plan, execution, row,
                inherited_episodes[row["panel"]][row["case_index"]], fits))
        inputs = [search_audit.audit_innovations(plan, execution / "innovations" / "control" / f"{step:03d}",
                  f"planner/control/{step}", plan["control_episodes"]) for step in range(plan["steps"])]
        controls, replay, resets, disturbances = {panel: {} for panel in PANELS}, {}, None, None
        for row in protocol.execution_order(plan):
            check_budget()
            folder = execution / row["path"]
            records = base.load_records(folder / "episodes", plan["control_episodes"])
            replay[row["path"]] = audit_cohort(plan, records, row["panel"])
            initial, noise = base.stack(records, "audit", "integration_state")[:, 0], base.stack(records, "audit", "actuator_noise")
            if resets is None:
                resets, disturbances = initial, noise
            require(np.array_equal(initial, resets) and np.array_equal(noise, disturbances), "All 51 rows share fresh resets and exogenous noise")
            if "fit" in row:
                result = audit_learned_control(plan, folder, row, records, inputs, fits[row["fit"]])
            elif row["reference"] == "public_kinematic":
                result = inherited.audit_kinematic_control(plan, folder, records, inputs)
            else:
                result = search_audit.audit_control({**plan, "planners": ["cem256"]}, folder, row, records, inputs)
            controls[row["panel"]][row["label"]] = result
        gate, differences = qualification(plan, controls)
        comparisons = paired_comparisons(plan, controls, differences)
        boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed, restored, diagnostics)
        costs = audit_costs(plan, execution, completed, restored, diagnostics, controls)
        require(validate_members(plan, expected_plan_sha256, execution) == completed
                and sha(execution / "completed.json") == completion_hash, "Execution stable during audit")
        validate_runtime_sources(plan)
        check_budget()
        cohorts = list(replay.values()) + [row["native_replay"] for row in diagnostics]
        summary = {"status": "completed", "version": VERSION, "engineering": engineering,
            "plan_sha256": expected_plan_sha256, "execution_completed_sha256": completion_hash,
            "saved_output_only": True, "new_model_calls": 0, "new_policy_calls": 0, "new_fits": 0,
            "native_transitions_checked": sum(row["transitions"] for row in cohorts),
            "native_control_transitions_checked": sum(row["transitions"] for row in replay.values()),
            "native_diagnostic_transitions_checked": sum(row["native_replay"]["transitions"] for row in diagnostics),
            "native_max_abs_error": max(row["max_abs_error"] for row in cohorts),
            "public_observer_transitions_checked": sum(controls[panel]["public_kinematic"]["public_observer"]["observer_native_transitions_replayed"] for panel in PANELS),
            "coverage": protocol.coverage(plan), "random_streams": streams, "inherited_fits": fits,
            "phase_boundary": boundary, "control": controls, "diagnostic": diagnostics,
            "continuation_gate": gate, "paired_descriptive_comparisons": comparisons, "cohorts": replay,
            "costs": {**costs, "audit_validation_wall_seconds": time.monotonic() - begin},
            "limits": [
                "This zero-fit intervention preserves all six inherited students and their failed parent 15/28 result; it cannot rescue that earlier claim.",
                "Before/after observed tensors are equal. No trainer or optimizer is invoked by this audit; absence of intermediate updates is source-bound, not proven by endpoints alone.",
                "Neural predictions and latent vectors are saved source-bound evidence, not independently recomputed. Cache/public-clock fields and same-bank mode invariance are checked.",
                "Geometry is independent NumPy forward kinematics with clipped-action expected cost once. Final-angle FK is not exact native RK4 cached-body reward; projected mean angles do not imply expected distance.",
                "Native replay validates saved actions/outcomes. CEM is reconstructed from all paid proposals and saved scores with exact sequential float32 clipping.",
                "Diagnostic histories were already exposed in the prior completed cache study. Rank and regret describe only the finite union and four shared noise branches; they are outside the gate.",
                "Both learned and geometry modes retain original motion/reward heads. Geometry adds arithmetic and recording work; equal candidate counts are not equal compute.",
                "All six fits, twelve policies and 51 control rows are retained. Conditional episode bootstrap intervals do not capture training-corpus or seed-population uncertainty.",
                "Timing is shared-host batched throughput including trace work, not isolated per-case latency, total FLOPs or cross-machine speed.",
                "A positive gate would support this specific approximate scoring intervention, not biological novelty, a new world-model architecture, or general robotics capability.",
            ]}
        out.mkdir(parents=True, exist_ok=False)
        write(out / "summary.json", summary)
        (out / "README.md").write_text(
            "# Reacher frozen-transition geometry-scoring audit\n\n"
            f"Continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({sum(row['passed'] for row in gate['checks'])}/25 checks).\n\n"
            f"Six inherited fits, zero new fits, all 51 control rows and {len(diagnostics)} exposed diagnostic roots retained. "
            f"Replayed {summary['native_transitions_checked']:,} saved native transitions.\n\n"
            "All reward arithmetic, candidate selection, public cache fields, unchanged tensor snapshots, provenance and nested costs are checked. "
            "No learned inference is invoked by this audit. Predictions remain saved source-bound evidence. "
            "Geometry is an approximation to native reward, and the exposed finite-union diagnostics are descriptive. "
            "See summary.json for every gate, fit and limitation.\n")
        receipt = {"status": "completed", "version": VERSION, "engineering": engineering,
            "plan_sha256": expected_plan_sha256, "source_sha256": plan["sources"], "runtime": plan["runtime"],
            "execution_completed_sha256": completion_hash, "execution_members": completed["files"],
            "saved_output_only": True, "costs": summary["costs"],
            "files": {name: sha(out / name) for name in ("summary.json", "README.md")}}
        check_budget()
        write(out / "receipt.json", receipt)
        check_budget()
        return summary
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "receipt.json").exists():
            (out / "receipt.json").rename(out / "over-cap-receipt.json")
        write(out / "failed.json", {"status": "failed", "error": repr(error), "plan_sha256": expected_plan_sha256,
            "wall_seconds": time.monotonic() - begin, "saved_output_only": True})
        raise
    finally:
        _DEADLINE.reset(token)


def audit_plan(path, expected_sha256, execution, out):
    """CLI authenticates external scored plan and never enables engineering."""
    require(not Path(out).exists(), "Exclusive audit output")
    try:
        plan = read(checked(Path(path), expected_sha256))
        require(plan["engineering"] is False, "Scored plan required by CLI")
    except BaseException as error:
        Path(out).mkdir(parents=True, exist_ok=False)
        write(Path(out) / "failed.json", {"status": "failed", "error": repr(error), "plan_sha256": expected_sha256,
                                         "saved_output_only": True, "phase": "authenticate-plan"})
        raise
    return audit_saved(plan, expected_sha256, execution, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_plan(args.plan, args.expected_plan_sha256, args.execution, args.out)
    print(json.dumps({"status": summary["status"], "native_transitions_checked": summary["native_transitions_checked"],
                      "continuation_gate_passed": summary["continuation_gate"]["passed"]}))


if __name__ == "__main__":
    main()
