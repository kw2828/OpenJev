"""Prospective RNG and saved-artifact contract for Reacher search allocation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_random_streams import generator_manifest
from openjev.research.robotics_reacher import packet

STUDY = "reacher-search-v1"
SCORED_NAMESPACE = "reacher-search-v1-scored"
ENGINEERING_NAMESPACES = (
    "reacher-search-v1", "reacher-search-engineering-capacity-v1",
    "reacher-search-engineering-unit-v1", "reacher-search-engineering-whole-tree-v1",
)
PANELS = ("full", "ordinary", "shift")
PLANNERS = ("rs64", "rs256", "cem256")
REFERENCES = ("known_state", "particle", "zero", "uniform")
INPUT_NAMES = ("initial", "random_extra", "cem/1", "cem/2", "cem/3")
INPUT_COUNTS = (64, 192, 64, 64, 63)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def save_npz(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def named_seed(namespace, role):
    return int.from_bytes(hashlib.sha256(f"OpenJev/{namespace}/{role}".encode()).digest()[:8], "big")


def registry(plan):
    require(isinstance(plan["rng_namespace"], str) and bool(plan["rng_namespace"].strip()),
            "An explicit RNG namespace is required")
    roles = []
    for split, count in (("control", plan["control_episodes"]), ("diagnostic", plan["diagnostic_episodes"])):
        for name in ("reset", "actuator_noise", "sensor_schedule"):
            roles.extend(f"{split}/{name}/{i}" for i in range(count))
        if split == "diagnostic":
            roles.extend(f"diagnostic/exploration/{i}" for i in range(count))
    roles.extend(f"planner/particle_filter/{i}" for i in range(plan["control_episodes"]))
    roles.extend(("floor/uniform/0", "analysis/bootstrap/0", "diagnostic/native_template/0"))
    prefixes = [f"planner/control/{t}" for t in range(plan["steps"])]
    for episode in range(plan["diagnostic_episodes"]):
        for root in range(4):
            prefixes.append(f"planner/diagnostic/{episode}/{root}")
            roles.extend(f"diagnostic/branch_noise/{episode}/{root}/{b}"
                         for b in range(plan["diagnostic_branches"]))
    for prefix in prefixes:
        roles.extend(f"{prefix}/{name}" for name in INPUT_NAMES)
    require(len(set(roles)) == len(roles), "Duplicate prospective stream role")
    return {role: named_seed(plan["rng_namespace"], role) for role in roles}


def stream_contract(plan, prior_registries=(), prior_plans=()):
    current = registry(plan)
    require(len(set(current.values())) == len(current), "Prospective root seed collision")
    generators = generator_manifest(current)
    states = {item["initial_state_sha256"] for item in generators.values()}
    require(len(states) == len(generators), "Prospective generator-state collision")
    namespaces = plan["engineering_rng_namespaces"]
    require(isinstance(namespaces, list) and all(isinstance(value, str) and value.strip() for value in namespaces)
            and len(set(namespaces)) == len(namespaces), "Engineering namespace exclusion schema")
    engineering = []
    for namespace in namespaces:
        coverage = {"control_episodes": 64, "diagnostic_episodes": 16, "steps": 50, "diagnostic_branches": 4}
        previous = registry({**plan, **coverage, "rng_namespace": namespace})
        previous_generators = generator_manifest(previous)
        require(not (set(current.values()) & set(previous.values())), "Engineering root seed collision")
        previous_states = {item["initial_state_sha256"] for item in previous_generators.values()}
        require(not (states & previous_states), "Engineering generator-state collision")
        engineering.append({"namespace": namespace, "coverage": coverage,
                            "registry": previous, "generators": previous_generators})
    for previous in prior_registries:
        require(not (set(current.values()) & set(previous.values())), "Prior root seed collision")
        previous_states = {item["initial_state_sha256"] for item in generator_manifest(previous).values()}
        require(not (states & previous_states), "Prior generator-state collision")
    return {
        "namespace": plan["rng_namespace"], "registry": current, "generators": generators,
        "engineering_exclusions": engineering,
        "priors": list(prior_plans), "draws_for_manifest": 0,
        "full_horizon_innovations": True,
        "intentional_reuse": [
            "All controllers/panels share control case resets and actuator noise; schedules share their phase.",
            "All models/panels/planners share named initial and innovation arrays at each decision.",
            "Particle filter children are separate by purpose and reused by corresponding case across panels.",
            "Uniform commands and analysis bootstrap draws are paired across comparisons.",
            "Diagnostic models/panels share physical histories, root proposals and each native noise branch.",
            "Diagnostic branches are independent of one another and every control/training/evaluation role.",
        ],
    }


def seed(plan, role):
    return plan["random_stream_contract"]["registry"][role]


def phase(plan, index, split="control"):
    return int(np.random.default_rng(seed(plan, f"{split}/sensor_schedule/{index}")).integers(0, 4))


def schedule(plan, index, panel, split="control"):
    require(panel in PANELS, "Unknown sensing panel")
    valid = np.ones(plan["steps"] + 1, dtype=bool)
    if panel != "full":
        offset = phase(plan, index, split)
        gap = plan["shift_gap"] if panel == "shift" else plan["ordinary_gap"]
        for start in (8 + offset, 28 + offset):
            valid[start:start + gap] = False
    return valid


def root_steps(plan, index):
    offset = phase(plan, index, "diagnostic")
    return (6, 10 + offset, 8 + offset + plan["ordinary_gap"], 47)


def draw_inputs(plan, prefix, count):
    chunks = (plan["planning_horizon"] + plan["action_block"] - 1) // plan["action_block"]
    values = [np.random.default_rng(seed(plan, f"{prefix}/{name}")).normal(size=(count, k, chunks, 2))
              for name, k in zip(INPUT_NAMES, INPUT_COUNTS, strict=True)]
    return SearchInputs(values[0], values[1], tuple(values[2:]))


def save_inputs(stem, inputs, prefix):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    values = (inputs.initial, inputs.random_extra, *inputs.cem)
    save_npz(stem.with_suffix(".npz"), **{name.replace("/", "_"): value
             for name, value in zip(INPUT_NAMES, values, strict=True)})
    write(stem.with_suffix(".json"), {
        "prefix": prefix, "input_identities": dict(inputs.identities()),
        "shapes": {name: list(value.shape) for name, value in zip(INPUT_NAMES, values, strict=True)},
        "unused_anchor_draws": 7, "full_horizon_innovations": True,
    })


def load_inputs(stem):
    with np.load(Path(stem).with_suffix(".npz"), allow_pickle=False) as saved:
        require(set(saved.files) == {name.replace("/", "_") for name in INPUT_NAMES}, "Innovation member set")
        values = [saved[name.replace("/", "_")] for name in INPUT_NAMES]
    result = SearchInputs(values[0], values[1], tuple(values[2:]))
    meta = read(Path(stem).with_suffix(".json"))
    require(meta["input_identities"] == dict(result.identities()), "Saved innovation identity")
    return result


def public_history(record, valid, root_step):
    """Materialize the allowed observation/command prefix, never a hidden-state input."""
    require(type(root_step) is int and 0 <= root_step < 50, "Invalid diagnostic root")
    require(np.asarray(valid).dtype == np.bool_ and np.asarray(valid).shape == (51,) and valid[0],
            "Invalid diagnostic sensing schedule")
    packets, last = [], 0
    for t in range(root_step + 1):
        if valid[t]:
            last = t
        packets.append(packet(record["audit"]["qpos"][t, :2], record["audit"]["qpos"][t, 2:4],
                              valid=valid[t], age_seconds=(t - last) * record["metadata"]["dt"]))
    return {"packets": np.array(packets, dtype=np.float32),
            "commands": record["policy"]["commands"][:root_step].copy()}


def trace_payload(result, raw_rewards, callback_sizes, search_seconds):
    arrays = {
        "chunks": result.sequences[:, :, ::result.action_block].copy(),
        "scores": result.scores, "raw_rewards": np.asarray(raw_rewards), "selected_ids": result.selected_ids,
    }
    require(arrays["raw_rewards"].dtype == np.float32
            and arrays["raw_rewards"].shape == (*result.scores.shape, result.horizon)
            and np.isfinite(arrays["raw_rewards"]).all(), "Raw reward trace shape/dtype/finite")
    meta = {key: getattr(result, key) for key in (
        "method", "horizon", "action_block", "candidate_evaluations_per_case", "candidate_evaluations",
        "imagined_transitions_per_case", "imagined_transitions",
    )}
    meta.update(candidate_ids=list(result.candidate_ids), input_identities=dict(result.input_identities),
                callback_sizes=list(callback_sizes), search_seconds=search_seconds, stages=[])
    for i, stage in enumerate(result.stages):
        row, fields = {}, {}
        for field in dataclasses.fields(stage):
            value = getattr(stage, field.name)
            if isinstance(value, np.ndarray):
                key = f"stage_{i}_{field.name}"
                arrays[key] = value
                fields[field.name] = key
            elif field.name not in {"proposal_mean", "proposal_std", "fixed_scales", "source_elite_ids"}:
                row[field.name] = value
        row["array_fields"] = fields
        meta["stages"].append(row)
    return arrays, meta


def save_trace(stem, result, raw_rewards, callback_sizes, search_seconds):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = trace_payload(result, raw_rewards, callback_sizes, search_seconds)
    save_npz(stem.with_suffix(".npz"), **arrays)
    write(stem.with_suffix(".json"), meta)
