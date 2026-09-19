"""Prospective role separation for the legacy Reacher helper interfaces.

Each helper still adds its documented episode/decision offsets. Distinct
hashed blocks separate those roles, and the expanded concrete seed registry
is checked across roles and against every supplied prior registry. A repeated
seed across paired controllers/panels is intentional and represented once.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

FIELDS = (
    "prediction_seed", "control_seed", "schedule_seed", "noise_seed",
    "exploration_seed", "candidate_seed", "filter_seed", "bootstrap_seed",
)


def domain_bases(study: str) -> dict[str, int]:
    if not isinstance(study, str) or not study.strip():
        raise ValueError("A nonempty study namespace is required")
    # Reserve 2**20 consecutive values per role for the existing <300k offsets.
    result = {
        field: int.from_bytes(hashlib.sha256(f"OpenJev/reacher/{study}/{field}".encode()).digest()[:6], "big") & ~((1 << 20) - 1)
        for field in FIELDS
    }
    if len(set(result.values())) != len(result):
        raise ValueError("Hashed role blocks collided; choose a new pre-run study namespace")
    return result


def concrete_streams(plan, *, include_train=False):
    """Enumerate actual NumPy generator seeds, not merely base field values."""
    registry = {}

    def add(prefix, base, count):
        if type(base) is not int or type(count) is not int or count < 1:
            raise ValueError("Seed bases/counts must be positive-count integer values")
        for index in range(count):
            registry[f"{prefix}/{index}"] = base + index

    for split, offset in (("prediction", 100000), ("control", 200000)):
        count = plan[f"{split}_episodes"]
        add(f"{split}/reset", plan[f"{split}_seed"], count)
        add(f"{split}/actuator_noise", plan["noise_seed"] + offset, count)
        add(f"{split}/sensor_schedule", plan["schedule_seed"] + (0 if split == "control" else offset), count)
        if split == "prediction":
            add("prediction/exploration", plan["exploration_seed"] + offset, count)
    add("planner/candidate_bank", plan["candidate_seed"], plan["steps"])
    add("planner/particle_filter", plan["filter_seed"], plan["control_episodes"])
    add("floor/uniform", plan["candidate_seed"] + 200000, 1)
    add("analysis/bootstrap", plan["bootstrap_seed"], 1)
    if include_train:
        for role, field in (("reset", "train_seed"), ("actuator_noise", "noise_seed"),
                            ("sensor_schedule", "schedule_seed"), ("exploration", "exploration_seed")):
            add(f"train/{role}", plan[field], plan["train_episodes"])
    return registry


def collisions(registry):
    by_seed = defaultdict(list)
    for name, seed in registry.items():
        by_seed[seed].append(name)
    return [{"seed": seed, "roles": roles} for seed, roles in sorted(by_seed.items()) if len(roles) > 1]


def generator_manifest(registry):
    """Identify initial generator states, including the filter's spawned children.

    This allocates RNG objects but draws no samples. The filter parent is a
    SeedSequence, not a separately consumed generator. Controller/panel pairing
    reuses the same named experimental stream and appears only once here.
    """
    result = {}
    for role, seed in registry.items():
        if role.startswith("planner/particle_filter/"):
            sequences = list(zip(("initial", "process_noise", "resample"),
                                 np.random.SeedSequence(seed).spawn(3), strict=True))
        else:
            sequences = [(None, np.random.SeedSequence(seed))]
        for child, sequence in sequences:
            generator = np.random.default_rng(sequence)
            state = json.dumps(generator.bit_generator.state, sort_keys=True, separators=(",", ":"))
            result[role + (f"/{child}" if child else "")] = {
                "entropy": seed, "spawn_key": list(sequence.spawn_key),
                "bit_generator": type(generator.bit_generator).__name__,
                "initial_state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                "draws_for_manifest": 0,
            }
    return result


def validate_separation(plan, *, prior_registries=()):
    current = concrete_streams(plan)
    found = collisions(current)
    if found:
        raise ValueError(f"Cross-role random-stream collisions: {found[:3]}")
    current_seeds = set(current.values())
    current_states = [item["initial_state_sha256"] for item in generator_manifest(current).values()]
    if len(set(current_states)) != len(current_states):
        raise ValueError("Current concrete generators have identical initial states")
    for prior in prior_registries:
        if current_seeds & set(prior.values()):
            raise ValueError("Current evaluation streams overlap a prior random-stream registry")
        previous_states = {item["initial_state_sha256"] for item in generator_manifest(prior).values()}
        if previous_states & set(current_states):
            raise ValueError("Current generator states overlap a prior generator manifest")
    return current
