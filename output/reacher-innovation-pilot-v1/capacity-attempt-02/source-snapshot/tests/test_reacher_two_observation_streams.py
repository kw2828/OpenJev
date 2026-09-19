"""Synthetic historical dictionaries and engineering namespace only.

Invented prior integers initialize local generators solely for state manifests;
they are not historical evidence or sampled data. Only named engineering control
streams are sampled. No production artifacts, scored seed derivation or models.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_streams as streams
from openjev.research.reacher_adaptive_search import search


def descriptor(name, values, *, kind="numpy"):
    manifest = streams.generator_manifest if kind == "numpy" else streams.torch_generator_manifest
    return {"name": name, "artifact_sha256": "a" * 64,
            "registry": values, "generators": manifest(values)}


@pytest.fixture(scope="module")
def full_history():
    # Synthetic numerical values, never derived from old/scored namespaces.
    roles = protocol.role_manifest(protocol.settings(engineering=True))["root_roles"]
    value = {"lineage_sha256": "b" * 64,
             "numpy_priors": [descriptor("synthetic/older", {"old/reset/0": 1001,
                                "planner/particle_filter/0": 1002})],
             "torch_priors": [descriptor("synthetic/old-order", {"historical/order": 1003}, kind="torch")],
             "geometry_memory": [descriptor(namespace, {role: 10000 + 1000 * i + j for j, role in enumerate(roles)})
                                 for i, namespace in enumerate(streams.GEOMETRY_MEMORY_NAMESPACES)],
             "literal_calls": []}
    for path, kinds in streams.REQUIRED_LITERAL_CALLS.items():
        value["literal_calls"].append({"source_path": path, "source_sha256": "c" * 64,
            "numpy_registry": dict(kinds["numpy"]), "numpy_generators": streams.generator_manifest(kinds["numpy"]),
            "torch_registry": dict(kinds["torch"]), "torch_generators": streams.torch_generator_manifest(kinds["torch"])})
    return value


@pytest.fixture
def history(full_history):
    return copy.deepcopy(full_history)


@pytest.fixture
def plan():
    value = protocol.settings(engineering=True)
    value["control_episodes"] = 2
    return value


@pytest.fixture(autouse=True)
def no_scored_derivation(monkeypatch):
    original = streams.named_seed

    def checked(namespace, role):
        assert namespace in protocol.ENGINEERING_NAMESPACES, "Tests must never derive scored/historical namespaces"
        return original(namespace, role)

    monkeypatch.setattr(streams, "named_seed", checked)


@pytest.fixture
def bound(plan, full_history):
    contract = streams.stream_contract(plan, history=full_history)
    return streams.validate_stream_contract(plan, contract, history=full_history)


def test_import_surface_has_no_models_io_or_import_time_allocation():
    tree = ast.parse(Path(streams.__file__).read_text())
    # No top-level call is allowed except definitions' decorators; constants are
    # literals and one object sentinel. This catches accidental RNG initialization.
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    assert isinstance(child.func, ast.Name) and child.func.id == "object"
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("study" in (name or "") or "world_model" in (name or "") for name in imports)
    with patch("builtins.open", side_effect=AssertionError("No file I/O")), patch.object(
            streams.np.random, "default_rng", side_effect=AssertionError("No RNG allocation")):
        values = streams.registry(protocol.settings(engineering=True))
    assert len(values) == len(set(values.values())) == 508


def test_full_registry_and_spawned_filter_manifest_preserve_real_roles(plan, full_history):
    plan["control_episodes"] = 64
    contract = streams.stream_contract(plan, history=full_history)
    assert contract["root_role_count"] == 508
    assert contract["generator_role_count"] == 636
    role = "planner/particle_filter/63"
    assert role in contract["registry"] and role not in contract["generators"]
    for index, child in enumerate(protocol.FILTER_CHILDREN):
        state = contract["generators"][f"{role}/{child}"]
        assert state["spawn_key"] == [index]
        assert state["entropy"] == contract["registry"][role]
        assert state["draws_for_manifest"] == 0
    assert all(item["draws_for_manifest"] == 0 for item in contract["generators"].values())


def test_engineering_exclusions_are_full64_and_production_obligation_explicit(plan, full_history):
    contract = streams.stream_contract(plan, history=full_history)
    assert contract["root_role_count"] == 260 and contract["generator_role_count"] == 264
    assert len(contract["history"]["geometry_memory"]) == 5
    assert [row["namespace"] for row in contract["own_engineering_exclusions"]] == list(protocol.ENGINEERING_NAMESPACES[1:])
    assert all(len(row["registry"]) == 508 and len(row["generators"]) == 636
               and row["control_episodes"] == 64 for row in contract["own_engineering_exclusions"])
    authority = contract["authority"]
    assert authority["scored_namespace_checked"] is False
    assert authority["production_freshness_certified"] is authority["execution_authorized"] is False
    assert "production preparation" in authority["pending"][0]
    assert contract["allocation"]["manifest_sample_draws"] == 0
    assert contract["allocation"]["new_torch_namespace_streams"] == 0
    assert contract["training_rng"]["new_fits"] == 3
    assert "real draws" in contract["training_rng"]["historical_order_replay"]


def test_manifest_binding_retains_all_history_and_does_not_mutate_or_change_global_rng(plan, history):
    before = copy.deepcopy(history)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    contract = streams.stream_contract(plan, history=history)
    assert history == before
    assert contract["history"] == before
    assert torch.equal(torch.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    contract["history"]["numpy_priors"][0]["registry"]["old/reset/0"] = 7
    assert history == before


@pytest.mark.parametrize("field", ["numpy_priors", "torch_priors", "geometry_memory", "literal_calls"])
def test_complete_explicit_history_required(plan, history, field):
    history[field] = []
    with pytest.raises(ValueError):
        streams.stream_contract(plan, history=history)


@pytest.mark.parametrize("mutation", ["missing_root", "missing_child", "wrong_child", "wrong_order", "wrong_namespace", "truncated64"])
def test_geometry_memory_full_scope_and_children_cannot_be_truncated(plan, history, mutation):
    row = history["geometry_memory"][0]
    if mutation == "missing_root":
        del row["registry"]["control/reset/63"]
        row["generators"] = streams.generator_manifest(row["registry"])
    elif mutation == "missing_child":
        del row["generators"]["planner/particle_filter/63/resample"]
    elif mutation == "wrong_child":
        row["generators"]["planner/particle_filter/0/initial"]["spawn_key"] = [1]
    elif mutation == "wrong_order":
        history["geometry_memory"].reverse()
    elif mutation == "wrong_namespace":
        row["name"] = "unbound-engineering-space"
    else:
        row["registry"] = {key: value for key, value in row["registry"].items()
                           if not key.startswith("control/") or key.endswith("/0")}
        row["generators"] = streams.generator_manifest(row["registry"])
    with pytest.raises(ValueError):
        streams.stream_contract(plan, history=history)


@pytest.mark.parametrize("where", ["older", "geometry", "literal"])
def test_new_root_collision_with_any_historical_inventory_rejects(plan, history, where):
    number = streams.registry(plan)["control/reset/0"]
    if where == "older":
        history["numpy_priors"][0] = descriptor("synthetic/collision", {"old/collision": number})
    elif where == "geometry":
        row = history["geometry_memory"][0]
        row["registry"]["control/reset/63"] = number
        row["generators"] = streams.generator_manifest(row["registry"])
    else:
        row = history["literal_calls"][-1]
        row["numpy_registry"]["future/collision"] = number
        row["numpy_generators"] = streams.generator_manifest(row["numpy_registry"])
    with pytest.raises(ValueError, match="root seed overlap"):
        streams.stream_contract(plan, history=history)


def test_current_root_collision_and_cross_namespace_collision_reject(plan, full_history, monkeypatch):
    original = streams.named_seed
    monkeypatch.setattr(streams, "named_seed", lambda namespace, role: 9 if namespace == plan["rng_namespace"] else original(namespace, role))
    with pytest.raises(ValueError, match="Current root seeds collide"):
        streams.stream_contract(plan, history=full_history)
    monkeypatch.setattr(streams, "named_seed", lambda namespace, role: original(plan["rng_namespace"], role))
    with pytest.raises(ValueError, match="root seed overlap"):
        streams.stream_contract(plan, history=full_history)


def test_state_collision_is_checked_separately_from_root_numbers(plan, history, monkeypatch):
    # Controlled fake hash collision only in this additive module. No frozen
    # module/global rebinding. Different root numbers remain different.
    current = streams.registry(plan)
    target = streams.generator_manifest(current)["planner/particle_filter/0/initial"]["initial_state_sha256"]
    original = streams.generator_manifest

    def forced(values):
        result = original(values)
        if "old/reset/0" in values:
            result["old/reset/0"]["initial_state_sha256"] = target
        return result

    monkeypatch.setattr(streams, "generator_manifest", forced)
    history["numpy_priors"][0]["generators"] = forced(history["numpy_priors"][0]["registry"])
    with pytest.raises(ValueError, match="initial generator state overlap"):
        streams.stream_contract(plan, history=history)


@pytest.mark.parametrize("mutation", ["sha", "unknown_key", "bool_seed", "negative_seed", "huge_seed", "duplicate_name", "forged_state", "torch_effective"])
def test_malformed_prior_bindings_rejected(plan, history, mutation):
    row = history["numpy_priors"][0]
    if mutation == "sha":
        row["artifact_sha256"] = "not-a-digest"
    elif mutation == "unknown_key":
        row["authenticated"] = True
    elif mutation == "bool_seed":
        row["registry"]["old/reset/0"] = True
    elif mutation == "negative_seed":
        row["registry"]["old/reset/0"] = -1
    elif mutation == "huge_seed":
        row["registry"]["old/reset/0"] = 2**64
    elif mutation == "duplicate_name":
        history["numpy_priors"].append(copy.deepcopy(row))
    elif mutation == "forged_state":
        row["generators"]["old/reset/0"]["initial_state_sha256"] = "d" * 64
    else:
        history["torch_priors"][0]["generators"]["historical/order"]["effective_seed32"] += 1
    with pytest.raises(ValueError):
        streams.stream_contract(plan, history=history)


@pytest.mark.parametrize("mutation", ["source_missing", "seed_missing", "bad_path", "duplicate_path", "bad_sha", "bad_state"])
def test_known_literal_inventory_and_source_attribution_required(plan, history, mutation):
    row = history["literal_calls"][-1]
    if mutation == "source_missing":
        history["literal_calls"].pop()
    elif mutation == "seed_missing":
        row["numpy_registry"] = {}
        row["numpy_generators"] = {}
    elif mutation == "bad_path":
        row["source_path"] = "../outside.py"
    elif mutation == "duplicate_path":
        history["literal_calls"].append(copy.deepcopy(row))
    elif mutation == "bad_sha":
        row["source_sha256"] = "A" * 64
    else:
        row["numpy_generators"]["synthetic_innovations"]["draws_for_manifest"] = 1
    with pytest.raises(ValueError):
        streams.stream_contract(plan, history=history)


def test_torch_aliases_in_intentionally_reused_history_are_retained(plan, history):
    history["torch_priors"].append(descriptor("synthetic/alias", {"alias": 1003 + 2**32}, kind="torch"))
    contract = streams.stream_contract(plan, history=history)
    states = contract["history"]["torch_priors"]
    a = states[0]["generators"]["historical/order"]
    b = states[1]["generators"]["alias"]
    assert a["seed"] != b["seed"] and a["effective_seed32"] == b["effective_seed32"]
    assert a["initial_state_sha256"] == b["initial_state_sha256"]


def test_validated_handle_immutable_and_fast_path_does_not_rebuild_history(plan, bound):
    with pytest.raises(TypeError):
        bound.registry["control/reset/0"] = 1
    with pytest.raises(FrozenInstanceError):
        bound.settings_sha256 = "a" * 64
    with patch.object(streams, "stream_contract", side_effect=AssertionError("Expensive history rebuild")), patch.object(
            streams, "_validate_history", side_effect=AssertionError("Historical lookup")):
        assert isinstance(streams.seed(plan, "control/reset/0", contract=bound), int)
        assert streams.schedule(plan, 0, "shift", contract=bound).shape == (51,)
        assert streams.draw_control_inputs(plan, 49, contract=bound).initial.shape == (2, 64, 4, 2)


def test_unvalidated_or_tampered_contract_and_mutated_settings_rejected(plan, history, bound):
    contract = streams.stream_contract(plan, history=history)
    with pytest.raises(ValueError, match="prevalidated"):
        streams.seed(plan, "control/reset/0", contract=contract)
    contract["authority"]["production_freshness_certified"] = True
    with pytest.raises(ValueError, match="Exact stream contract"):
        streams.validate_stream_contract(plan, contract, history=history)
    plan["control_episodes"] = 1
    with pytest.raises(ValueError, match="Settings changed"):
        streams.seed(plan, "control/reset/0", contract=bound)


def test_history_binding_revalidated_even_if_only_lineage_hash_changes(plan, history):
    contract = streams.stream_contract(plan, history=history)
    history["lineage_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Exact stream contract"):
        streams.validate_stream_contract(plan, contract, history=history)


def test_paired_schedule_lengths_offsets_and_immutable_boundary(plan, bound):
    for case in range(plan["control_episodes"]):
        full = streams.schedule(plan, case, "full", contract=bound)
        ordinary = streams.schedule(plan, case, "ordinary", contract=bound)
        shift = streams.schedule(plan, case, "shift", contract=bound)
        assert full.all() and full.dtype == np.bool_
        assert ordinary[0] and ordinary[-1] and shift[0] and shift[-1]
        missing6, missing10 = np.flatnonzero(~ordinary), np.flatnonzero(~shift)
        assert len(missing6) == 12 and len(missing10) == 20
        assert 8 <= missing6[0] <= 11 and missing6[6] == missing6[0] + 20
        assert missing10[0] == missing6[0] and missing10[10] == missing6[6]
        assert set(missing6) <= set(missing10)
        assert np.array_equal(ordinary, streams.schedule(plan, case, "ordinary", contract=bound))
        with pytest.raises(ValueError):
            ordinary.flags.writeable = True


@pytest.mark.parametrize("index,panel", [(True, "full"), (-1, "full"), (2, "full"), (0, "missing")])
def test_schedule_argument_boundaries(plan, bound, index, panel):
    with pytest.raises(ValueError):
        streams.schedule(plan, index, panel, contract=bound)


@pytest.mark.parametrize("step", [True, -1, 50, 1.0])
def test_draw_argument_boundaries(plan, bound, step):
    with pytest.raises(ValueError):
        streams.draw_control_inputs(plan, step, contract=bound)


def test_full_horizon_draw_trace_terminal_clipping_and_exact_common_inputs(plan, bound):
    value = streams.draw_control_inputs(plan, 49, contract=bound)
    again = streams.draw_control_inputs(plan, 49, contract=bound)
    assert value.identities() == again.identities()
    arrays = [value.initial, value.random_extra, *value.cem]
    for name, count, array in zip(protocol.INPUT_NAMES, protocol.INPUT_COUNTS, arrays, strict=True):
        assert array.shape == (2, count, 4, 2) and array.dtype == np.float64
        # Independent primitive replay on the SAME named engineering role.
        role = f"planner/control/49/{name}"
        generator = np.random.Generator(np.random.PCG64(bound.registry[role]))
        assert np.array_equal(array, generator.normal(size=array.shape))
        with pytest.raises(ValueError):
            array.flags.writeable = True
    result = search("cem256", value, lambda commands: np.zeros(commands.shape[:2], dtype=np.float32), step=49)
    assert result.horizon == 1 and result.sequences.shape == (2, 256, 1, 2)
    assert result.candidate_evaluations == result.imagined_transitions == 512
    assert [stage.stop - stage.start for stage in result.stages] == [64] * 4
    assert result.input_identities == value.identities()
    assert value.identities() != streams.draw_control_inputs(plan, 48, contract=bound).identities()


def test_contract_digest_has_exact_serializable_content(plan, full_history, bound):
    contract = streams.stream_contract(plan, history=full_history)
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(payload.encode()).hexdigest() == bound.contract_sha256
    assert streams.validate_stream_contract(plan, json.loads(payload), history=full_history).registry == bound.registry
    with pytest.raises(ValueError, match="Unknown named stream role"):
        streams.seed(plan, "control/nonexistent/0", contract=bound)
