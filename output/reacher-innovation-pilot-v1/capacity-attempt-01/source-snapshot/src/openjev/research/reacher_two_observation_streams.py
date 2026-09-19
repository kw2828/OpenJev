"""Explicit RNG binding for the prospective two-observation study.

No artifacts are read and no RNG is allocated at import time. The future runner
must authenticate the caller-supplied historical inventory against its actual
lineage. This module verifies the inventory's structure and generator identities,
not that supplied hashes correspond to files or exhaust previously consumed RNGs.

``stream_contract`` allocates local generators to hash their initial states but
draws no samples. Validate that JSON contract at study entry/exit; pass the
returned immutable ``BoundStreams`` to cheap per-decision helpers. An engineering
contract deliberately never derives the prospective scored namespace. Production
preparation must separately check that namespace against all four full-size
engineering spaces and the complete authenticated historical inventory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType

import numpy as np

from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_random_streams import generator_manifest

VERSION = "reacher-two-observation-streams-v1"
GEOMETRY_MEMORY_NAMESPACES = (
    "reacher-geometry-memory-v1-scored",
    "reacher-geometry-memory-engineering-unit-v1",
    "reacher-geometry-memory-engineering-runner-v1",
    "reacher-geometry-memory-engineering-capacity-v1",
    "reacher-geometry-memory-engineering-whole-tree-v1",
)
# Actual literal calls in the additive components/tests at authoring. Additional
# future calls must be added to the externally authenticated inventory as well.
REQUIRED_LITERAL_CALLS = {
    "src/openjev/research/reacher_two_observation_history.py": {
        "numpy": {}, "torch": {"overwritten_constructor": 410}},
    "src/openjev/research/reacher_two_observation_training.py": {
        "numpy": {}, "torch": {"overwritten_constructor": 410}},
    "tests/test_reacher_two_observation_history.py": {
        "numpy": {}, "torch": {"synthetic_model_and_rng_checks": 410}},
    "tests/test_reacher_two_observation_training.py": {
        "numpy": {}, "torch": {"synthetic_model": 410, "synthetic_minibatch_order": 410}},
    "tests/test_reacher_two_observation_control.py": {
        "numpy": {"synthetic_innovations": 410}, "torch": {"synthetic_model": 410}},
}
_BOUND_SEAL = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Canonical finite JSON required") from error


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _path(value):
    return (isinstance(value, str) and bool(value) and not PurePosixPath(value).is_absolute()
            and all(part not in ("", ".", "..") for part in value.split("/")) and "\\" not in value)


def _registry(value, label, *, allow_empty=False):
    _require(type(value) is dict and (allow_empty or bool(value))
             and all(isinstance(role, str) and role.strip() == role and role
                     and type(number) is int and 0 <= number < 2**64
                     for role, number in value.items()), f"{label}: named uint64 registry required")
    return dict(value)


def _settings(plan):
    _require(type(plan) is dict, "Prospective settings dictionary required")
    protocol.validate_settings(plan, engineering=plan.get("engineering") is True)


def named_seed(namespace, role):
    """Pure derivation in the established OpenJev namespace format; no draw."""
    _require(isinstance(namespace, str) and namespace.strip() == namespace and namespace
             and isinstance(role, str) and role.strip() == role and role, "Nonempty namespace and role required")
    return int.from_bytes(hashlib.sha256(f"OpenJev/{namespace}/{role}".encode()).digest()[:8], "big")


def registry(plan):
    """Derive only this explicitly requested plan's root seeds, without RNGs."""
    _settings(plan)
    return {role: named_seed(plan["rng_namespace"], role)
            for role in protocol.role_manifest(plan)["root_roles"]}


def torch_generator_manifest(values):
    """Historical CPU seed32 convention, identical to the prior cache protocol.

    Torch is imported only on explicit validation. Local CPU generators do not
    change the global state and draw no values. These identities document reused
    original orders/literal construction, not fresh training streams. Full-width
    Torch seed consumers require their own separately reviewed manifest schema.
    """
    import torch

    result = {}
    for role, number in _registry(values, "Torch", allow_empty=True).items():
        effective = number & ((1 << 32) - 1)
        generator = torch.Generator(device="cpu").manual_seed(effective)
        result[role] = {
            "seed": number, "effective_seed32": effective, "device": "cpu",
            "initial_state_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
            "draws_for_manifest": 0,
        }
    return result


def _check_manifest(values, recorded, kind, label):
    expected = generator_manifest(values) if kind == "numpy" else torch_generator_manifest(values)
    _require(type(recorded) is dict and _json(recorded) == _json(expected),
             f"{label}: exact initial generator states and spawned children required")
    return expected


def _descriptor(row, kind, label):
    _require(type(row) is dict and set(row) == {"name", "artifact_sha256", "registry", "generators"}
             and isinstance(row["name"], str) and row["name"].strip() == row["name"] and row["name"]
             and _sha(row["artifact_sha256"]), f"{label}: complete bound descriptor required")
    values = _registry(row["registry"], label)
    manifest = _check_manifest(values, row["generators"], kind, label)
    return {"name": row["name"], "artifact_sha256": row["artifact_sha256"],
            "registry": values, "generators": manifest}


def _full_roles():
    # Symbolic full-size roles only; this does not derive any scored seed.
    return protocol.role_manifest(protocol.settings(engineering=True))["root_roles"]


def _validate_history(history):
    """Validate explicit supplied history, never discover lineage from disk."""
    keys = {"lineage_sha256", "numpy_priors", "torch_priors", "geometry_memory", "literal_calls"}
    _require(type(history) is dict and set(history) == keys and _sha(history["lineage_sha256"]),
             "Caller-bound complete historical inventory required")
    clean = {"lineage_sha256": history["lineage_sha256"]}
    for kind in ("numpy", "torch"):
        rows = history[f"{kind}_priors"]
        _require(type(rows) is list and rows, f"Explicit nonempty historical {kind} registry inventory required")
        clean[f"{kind}_priors"] = [_descriptor(row, kind, f"Historical {kind}") for row in rows]
        names = [row["name"] for row in rows]
        _require(len(names) == len(set(names)), f"Unique historical {kind} descriptor names required")
    rows = history["geometry_memory"]
    _require(type(rows) is list and len(rows) == len(GEOMETRY_MEMORY_NAMESPACES),
             "All five completed geometry-memory namespaces required")
    full_roles = set(_full_roles())
    clean["geometry_memory"] = []
    for namespace, row in zip(GEOMETRY_MEMORY_NAMESPACES, rows, strict=True):
        item = _descriptor(row, "numpy", "Geometry-memory")
        _require(item["name"] == namespace and set(item["registry"]) == full_roles,
                 "Exact ordered geometry-memory namespaces and full64 root membership required")
        clean["geometry_memory"].append(item)
    rows = history["literal_calls"]
    _require(type(rows) is list and rows, "Source-bound literal engineering calls required")
    clean["literal_calls"] = []
    for row in rows:
        fields = {"source_path", "source_sha256", "numpy_registry", "numpy_generators", "torch_registry", "torch_generators"}
        _require(type(row) is dict and set(row) == fields and _path(row["source_path"])
                 and _sha(row["source_sha256"]), "Source-bound literal descriptor required")
        item = {"source_path": row["source_path"], "source_sha256": row["source_sha256"]}
        for kind in ("numpy", "torch"):
            values = _registry(row[f"{kind}_registry"], "Literal", allow_empty=True)
            item[f"{kind}_registry"] = values
            item[f"{kind}_generators"] = _check_manifest(values, row[f"{kind}_generators"], kind, "Literal")
        _require(item["numpy_registry"] or item["torch_registry"], "A literal descriptor must identify an actual call")
        clean["literal_calls"].append(item)
    indexed = {row["source_path"]: row for row in clean["literal_calls"]}
    _require(len(indexed) == len(rows), "Unique literal source paths required")
    for path, required in REQUIRED_LITERAL_CALLS.items():
        _require(path in indexed, f"Known literal source missing: {path}")
        for kind in ("numpy", "torch"):
            values = indexed[path][f"{kind}_registry"]
            _require(all(values.get(role) == number and type(values.get(role)) is int
                         for role, number in required[kind].items()), f"Known literal call missing: {path}/{kind}")
    return clean


def _states(manifest):
    return [value["initial_state_sha256"] for value in manifest.values()]


def _separated(current, current_generators, previous, previous_generators, label):
    _require(not set(current.values()) & set(previous.values()), f"NumPy root seed overlap: {label}")
    _require(not set(_states(current_generators)) & set(_states(previous_generators)),
             f"NumPy initial generator state overlap: {label}")


def stream_contract(plan, *, history):
    """Build auditable JSON without drawing samples or authorizing execution.

    ``history`` has a caller-authenticated lineage hash, ordered nonempty
    numpy_priors/torch_priors descriptors, all five full64 geometry_memory
    descriptors and source-bound literal_calls. Registry descriptors contain
    name, artifact_sha256, registry and generators. Literal descriptors contain
    source_path/source_sha256 plus numpy_registry/numpy_generators and
    torch_registry/torch_generators. Old repeated streams are retained: only the
    new control streams must be disjoint. No implicit history defaults exist.
    """
    _settings(plan)
    old = _validate_history(history)
    current = registry(plan)
    generators = generator_manifest(current)
    _require(len(set(current.values())) == len(current), "Current root seeds collide")
    _require(len(set(_states(generators))) == len(generators), "Current generator states collide")
    for item in old["numpy_priors"] + old["geometry_memory"]:
        _separated(current, generators, item["registry"], item["generators"], item["name"])
    for item in old["literal_calls"]:
        _separated(current, generators, item["numpy_registry"], item["numpy_generators"], item["source_path"])
    exclusions = []
    for namespace in protocol.ENGINEERING_NAMESPACES:
        if namespace == plan["rng_namespace"]:
            continue
        values = {role: named_seed(namespace, role) for role in _full_roles()}
        states = generator_manifest(values)
        _separated(current, generators, values, states, namespace)
        exclusions.append({"namespace": namespace, "control_episodes": 64,
                           "registry": values, "generators": states})
    engineering = plan["engineering"]
    return {
        "version": VERSION, "study": protocol.STUDY, "engineering": engineering,
        "namespace": plan["rng_namespace"], "settings_sha256": _digest(plan),
        "role_manifest_sha256": _digest(protocol.role_manifest(plan)),
        "registry": current, "generators": generators,
        "root_role_count": len(current), "generator_role_count": len(generators),
        "history": old, "history_sha256": _digest(old), "own_engineering_exclusions": exclusions,
        "allocation": {"manifest_sample_draws": 0, "import_time_rng_allocations": 0,
                       "new_torch_namespace_streams": 0, "local_generator_state_hashing": True},
        "training_rng": {
            "new_random_initializations": 0, "new_minibatch_orders": 0,
            "new_fits": 3, "original_initial_tensors": "Intentional reuse of all three authenticated paired initial tensors.",
            "historical_order_replay": "Intentional replay of original Torch permutations for validation; these are real draws, not fresh orders.",
            "overwritten_construction": "Isolated literal410 construction draws are real, then all tensors are replaced and ambient Torch RNG restored. Additional inherited restorers retain their separately bound literal conventions.",
            "constructor_call_counts": "Future runner must measure actual construction/restoration calls; no assumed twelve-call or zero-draw claim.",
        },
        "sampling": {"innovation_shapes": protocol.role_manifest(plan)["innovation_shapes"],
                     "full_horizon_draws_at_every_step": True, "unused_random_extra_candidates": 192,
                     "unused_draws": "Anchors, unused horizon tails and random_extra remain paid generation/provenance overhead, not extra scored candidates.",
                     "pairing": "Same role seeds across all rows/panels; initial proposals and innovations shared, later CEM banks adapt to scores."},
        "authority": {
            "scope": "Structure and separation of supplied initial generator identities only; no artifact authentication, freeze or execution authorization.",
            "execution_authorized": False, "production_freshness_certified": False,
            "scored_namespace_checked": not engineering,
            "pending": (["Engineering validation skips scored namespace derivation; production preparation must check its full64 registry against all four own engineering namespaces."] if engineering else [])
                       + ["Caller must authenticate complete historical artifact/source inventory, including all older inherited registries and any additional literal calls, at entry and exit."],
        },
    }


@dataclass(frozen=True, init=False)
class BoundStreams:
    """Immutable settings/registry binding created only by explicit validation.

    This is an in-process misuse guard, not a security boundary or run approval.
    Never serialize it instead of the independently hash-bound JSON contract.
    """

    settings_sha256: str
    contract_sha256: str
    registry: Mapping[str, int]
    _seal: object


def validate_stream_contract(plan, contract, *, history):
    """Recompute the full binding once at study entry/exit, then return a handle."""
    expected = stream_contract(plan, history=history)
    _require(type(contract) is dict and _json(contract) == _json(expected), "Exact stream contract required")
    result = object.__new__(BoundStreams)
    for name, value in {"settings_sha256": expected["settings_sha256"],
                        "contract_sha256": _digest(expected),
                        "registry": MappingProxyType(dict(expected["registry"])), "_seal": _BOUND_SEAL}.items():
        object.__setattr__(result, name, value)
    return result


def _bound(plan, contract):
    _require(type(contract) is BoundStreams and getattr(contract, "_seal", None) is _BOUND_SEAL,
             "Explicit prevalidated BoundStreams required")
    _require(type(plan) is dict and _digest(plan) == contract.settings_sha256, "Settings changed after stream validation")
    return contract.registry


def seed(plan, role, *, contract):
    """Read one bound seed without reconstructing any historical generators."""
    values = _bound(plan, contract)
    _require(isinstance(role, str) and role in values, "Unknown named stream role")
    return values[role]


def schedule(plan, index, panel, *, contract):
    """Paired sensor phase from one local named stream, independent of panel."""
    _bound(plan, contract)
    _require(type(index) is int and 0 <= index < plan["control_episodes"], "Control case index out of range")
    _require(panel in protocol.PANELS, "Unknown sensing panel")
    result = np.ones(plan["steps"] + 1, dtype=np.bool_)
    if panel != "full":
        generator = np.random.default_rng(seed(plan, f"control/sensor_schedule/{index}", contract=contract))
        offset = int(generator.integers(0, 4))
        length = plan["ordinary_gap"] if panel == "ordinary" else plan["shift_gap"]
        for start in (8 + offset, 28 + offset):
            result[start:start + length] = False
    # A caller cannot alter the paired schedule or later make its copy writable.
    return np.frombuffer(result.tobytes(), dtype=np.bool_)


def draw_control_inputs(plan, step, *, contract):
    """Generate all five full-horizon innovations, including unused draw traces."""
    _bound(plan, contract)
    _require(type(step) is int and 0 <= step < plan["steps"], "Control step out of range")
    chunks = (plan["planning_horizon"] + plan["action_block"] - 1) // plan["action_block"]
    values = []
    for name, count in zip(protocol.INPUT_NAMES, protocol.INPUT_COUNTS, strict=True):
        generator = np.random.default_rng(seed(plan, f"planner/control/{step}/{name}", contract=contract))
        values.append(generator.normal(size=(plan["control_episodes"], count, chunks, 2)))
    return SearchInputs(values[0], values[1], tuple(values[2:]))
