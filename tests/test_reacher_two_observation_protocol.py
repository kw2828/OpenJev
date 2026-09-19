"""Pure synthetic dictionaries/AST only: no RNG, models, simulator or assets."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from openjev.research import reacher_two_observation_protocol as protocol


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def synthetic_bindings(plan):
    inherited = {f"synthetic/old-{i}.py": f"{i:064x}" for i in range(90)}
    sources = {**inherited, **{name: "a" * 64 for name in protocol.NEW_SOURCE_PATHS}}
    bindings = {
        "lineage": {"cache_plan_sha256": protocol.CACHE_PLAN_SHA256,
                    "prerequisite_plan_sha256": protocol.PREREQUISITE_PLAN_SHA256,
                    "prerequisite_terminal_sha256": protocol.PREREQUISITE_TERMINAL_SHA256,
                    "completed_positive_gate_required": True},
        "paired_training": {"cohort_sha256": "b" * 64, "public_tensor_sha256": "c" * 64,
            "pairs": {pair: {"initial_tensor_sha256": "d" * 64, "orders_sha256": "e" * 64,
                             "initial_state_kind": "original_initialization"} for pair in protocol.PAIRS}},
        "inherited_checkpoints": {row["name"]: {"model_class": row["model_class"],
            "checkpoint_sha256": "f" * 64, "weights_sha256": "1" * 64, "configuration_sha256": "2" * 64}
            for row in protocol.restore_manifest(plan)},
        "sources": sources,
        "runtime": {**dict.fromkeys(("python", "numpy", "torch", "mujoco", "gymnasium", "machine", "determinism"), "synthetic"),
                    "threads": plan["threads"]},
        "fresh_streams": {"namespace": plan["rng_namespace"], "role_manifest_sha256": digest(protocol.role_manifest(plan)),
                          "prior_lineage_sha256": "3" * 64, "generator_separation_receipt_sha256": "4" * 64},
        "capacity_and_caps": {"execution_seconds": 1, "audit_seconds": 1,
                              "capacity_receipt_sha256": "5" * 64, "rehearsal_receipt_sha256": "6" * 64},
        "audit_contract": {"member_schema_sha256": "7" * 64, "cost_schema_sha256": "8" * 64,
                           "independent_review_sha256": "9" * 64},
    }
    return bindings, copy.deepcopy(sources), inherited


def test_no_seed_generator_model_native_or_io_surface():
    # Reading this source for static inspection is the only operation outside
    # the blocked-I/O context; no production artifact or numerical seed exists.
    source = Path(protocol.__file__).read_text()
    tree = ast.parse(source)
    imports = [node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(imports) <= {"__future__", "copy", "hashlib", "json", "math", "pathlib"}
    with patch("builtins.open", side_effect=AssertionError("No I/O")):
        plan = protocol.settings()
        assert protocol.validate_settings(plan) is plan
        assert protocol.coverage(plan)["control_rows"] == 42
        roles = protocol.role_manifest(plan)
        assert roles["numerical_seed_allocation"] == "not_implemented_not_performed"
    assert not hasattr(protocol, "seed") and not hasattr(protocol, "registry")
    assert not hasattr(protocol, "draw_control_inputs")
    assert "cap_seconds" not in plan and "audit_cap_seconds" not in plan


def test_three_new_history_fits_and_six_unchanged_actual_class_fits_preserve_pairs():
    plan = protocol.settings()
    fits = protocol.fit_manifest(plan)
    assert len(fits) == len({row["name"] for row in fits}) == 9
    assert plan["fit_order"] == [row["name"] for row in fits]
    new, inherited = protocol.training_manifest(plan), protocol.restore_manifest(plan)
    assert [row["name"] for row in new] == [f"two_observation_gru-{pair}" for pair in protocol.PAIRS]
    assert len(inherited) == 6
    assert {row["arm"] for row in inherited} == {"residual_gru", "cached_gru"}
    assert all(row["new_optimizer_updates"] == 1152 and not row["weights_reused"] for row in new)
    assert all(row["new_optimizer_updates"] == 0 and row["weights_reused"] for row in inherited)
    for pair in protocol.PAIRS:
        local = [row for row in fits if row["pair"] == pair]
        assert len({row["initialization_path"] for row in local}) == 1
        assert len({row["order_path"] for row in local}) == 1
        assert next(row for row in new if row["pair"] == pair)["initial_source_fit"] == f"residual_gru-{pair}"
    members = protocol.inherited_members(plan)
    assert len(members) == 44 and sum(name.endswith("checkpoint.pt") for name in members) == 6
    assert not any(name.startswith("control/") for name in members)


def test_exact42_row_order_all_fits_references_and_geometry_cem():
    plan = protocol.settings()
    rows = protocol.execution_order(plan)
    assert len(rows) == len({row["path"] for row in rows}) == 42
    assert rows == plan["execution_order"]
    for panel in protocol.PANELS:
        local = [row for row in rows if row["panel"] == panel]
        assert len(local) == 14
        assert len([row for row in local if "fit" in row]) == 9
        assert [row["reference"] for row in local if "reference" in row] == list(protocol.REFERENCES)
        for row in local:
            if row.get("reference") in ("zero", "uniform"):
                assert row["planner"] is row["score_mode"] is None
            else:
                assert row["planner"] == "cem256" and row["score_mode"] == "geometry"
    assert "MLP" not in " ".join(row.get("fit", "") for row in rows)
    assert plan["search_parameters"]["callback_sizes"] == [64] * 4
    assert "paid mean" in plan["search_parameters"]["final_stage"]


def test_exact_shortened_candidate_selected_native_and_reconstruction_counts():
    counts = protocol.coverage(protocol.settings())
    assert counts["sum_shortened_planning_horizons"] == 534
    assert counts["new_fits"] == 3 and counts["inherited_fits"] == 6 and counts["evaluated_models"] == 9
    assert counts["new_optimizer_updates"] == 3456
    assert counts["native_control_transitions"] == 134400
    assert counts["learned_candidate_evaluations"] == 22118400
    assert counts["physics_candidate_evaluations"] == 7372800
    assert counts["learned_imagined_transitions"] == 236224512
    assert counts["physics_nominal_transitions"] == 78741504
    assert counts["geometry_candidate_samples"] == 314966016
    assert counts["learned_selected_advances"] == 86400
    assert counts["physics_selected_advances"] == 28800
    assert counts["geometry_selected_samples"] == 115200
    assert counts["physics_candidate_native_substeps"] == 157483008
    assert counts["physics_selected_native_substeps"] == 57600
    assert counts["history_public_assimilate_samples"] == 28800
    assert counts["history_replayed_observation_update_samples"] == 345600
    assert counts["history_replayed_transition_samples"] == 316800
    assert counts["new_training_public_assimilate_samples"] == 5529600
    assert counts["new_training_replayed_observation_update_samples"] == 66355200
    assert counts["new_training_replayed_transition_samples"] == 60825600
    assert counts["new_training_prefix_and_rollout_advance_samples"] == 30965760
    assert counts["new_training_total_transition_samples"] == counts["new_training_analytic_reward_samples"] == 91791360
    assert counts["new_training_linear_layer_samples"] == 4 * 91791360
    assert counts["new_native_training_episodes"] == counts["new_prediction_episodes"] == 0


def test_exact25_proposed_checks_and_no_silent_equivalence_or_secondary_rescue():
    plan = protocol.settings()
    checks = protocol.criterion_manifest(plan)
    assert len(checks) == len({row["name"] for row in checks}) == 25
    assert {prefix: sum(row["name"].startswith(prefix) for row in checks)
            for prefix in ("gap_mean/", "gap_pair/", "full_mean/", "competence/")} == {
                "gap_mean/": 4, "gap_pair/": 12, "full_mean/": 2, "competence/": 7}
    for row in checks:
        prefix = row["name"].split("/")[0]
        assert row["maximum_cost_multiplier"] == {"gap_mean": .97, "gap_pair": 1., "full_mean": 1.02, "competence": .9}[prefix]
        assert row["treatment"] not in ("particle", "public_kinematic")
    assert [row["name"] for row in checks if row["treatment"] == "known_state"] == ["competence/ordinary/known_state"]
    assert plan["criterion"]["status"] == "proposed_until_complete_protocol_is_frozen"
    assert "does not prove equivalence" in plan["criterion"]["fail_interpretation"]
    assert plan["criterion"]["secondary_cannot_rescue_primary"] is True


def test_symbolic_roles_include_all_filter_children_unused_innovations_and_shared_phases():
    roles = protocol.role_manifest(protocol.settings())
    assert roles["root_role_count"] == len(set(roles["root_roles"])) == 508
    assert roles["generator_role_count"] == len({row["role"] for row in roles["generator_roles"]}) == 636
    assert roles["new_training_roles"] == []
    assert len(roles["own_namespace_exclusions"]) == 4
    for case in range(64):
        parent = f"planner/particle_filter/{case}"
        children = [row for row in roles["generator_roles"] if row["parent_role"] == parent]
        assert [row["role"].split("/")[-1] for row in children] == list(protocol.FILTER_CHILDREN)
        assert [row["spawn_key"] for row in children] == [[0], [1], [2]]
        assert parent not in {row["role"] for row in roles["generator_roles"]}
    assert roles["innovation_shapes"]["random_extra"] == [64, 192, 4, 2]
    assert roles["schedule"]["base_gap_starts"] == [8, 28]
    assert roles["schedule"]["ordinary_length"] == 6 and roles["schedule"]["shift_length"] == 10
    assert all(isinstance(role, str) for role in roles["root_roles"])


@pytest.mark.parametrize("key,value", [
    ("arms", ["residual_gru", "two_observation_gru"]), ("pairs", ["pair0"]),
    ("references", ["known_state", "zero"]), ("panels", ["ordinary", "shift"]),
    ("train_episodes", 767), ("epochs", 47), ("batch_size", 16), ("hidden_size", 63),
    ("steps", 49), ("learning_rate", .002), ("gradient_clip", 1.), ("rollout_horizon", 7),
    ("rollout_weight", 0.), ("reward_scale", 1.), ("control_episodes", 63),
    ("planner", "rs256"), ("physics_planner", "rs64"), ("planning_horizon", 11),
    ("action_block", 2), ("ordinary_gap", 5), ("shift_gap", 11),
    ("score_modes", ["learned"]), ("status", "frozen"), ("rng_namespace", protocol.ENGINEERING_NAMESPACES[0]),
])
def test_production_scope_drift_rejected_without_preparing_any_stream(key, value):
    plan = protocol.settings()
    plan[key] = value
    with pytest.raises(ValueError):
        protocol.validate_settings(plan)


def test_engineering_shrinks_only_sizes_without_removing_fits_physics_or_checks():
    plan = protocol.settings(engineering=True)
    plan.update(train_episodes=3, epochs=2, batch_size=2, hidden_size=3, control_episodes=1,
                threads=1, bootstrap_samples=8, rng_namespace=protocol.ENGINEERING_NAMESPACES[-1])
    assert protocol.validate_settings(plan, engineering=True) is plan
    counts = protocol.coverage(plan)
    assert counts["control_rows"] == 42 and counts["criterion_checks"] == 25
    assert counts["new_optimizer_updates_per_fit"] == 4 and counts["new_optimizer_updates"] == 12
    assert counts["native_control_transitions"] == 2100
    roles = protocol.role_manifest(plan)
    assert protocol.SCORED_NAMESPACE in roles["own_namespace_exclusions"]
    assert "Full64-case" in roles["exclusion_coverage"]
    plan["planning_horizon"] = 2
    with pytest.raises(ValueError):
        protocol.validate_settings(plan, engineering=True)


@pytest.mark.parametrize("key,value", [("control_episodes", True), ("epochs", 0), ("batch_size", 33),
                                       ("hidden_size", 65), ("steps", 4), ("rng_namespace", protocol.SCORED_NAMESPACE)])
def test_engineering_cannot_relax_scientific_or_type_boundaries(key, value):
    plan = protocol.settings(engineering=True)
    plan[key] = value
    with pytest.raises(ValueError):
        protocol.validate_settings(plan, engineering=True)


def test_returned_definitions_are_independent_and_missing_caps_not_invented():
    plan = protocol.settings()
    plan["criterion"]["mean_improvement"] = 0
    plan["optimizer"]["betas"].clear()
    plan["preparation_requirements"]["sources"]["required_new_paths"].clear()
    fresh = protocol.settings()
    assert fresh["criterion"]["mean_improvement"] == .03
    assert fresh["optimizer"]["betas"] == [.9, .999]
    assert len(fresh["preparation_requirements"]["sources"]["required_new_paths"]) == 14
    assert fresh["preparation_requirements"]["execution_authorized"] is False


def test_adam_and_training_recipe_match_existing_source_as_data_not_imported_models():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/openjev/research/reacher_objective_training.py").read_text())
    adam = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(name, ast.Name) and name.id == "ADAM" for name in node.targets))
    assert protocol.settings()["optimizer"] == adam
    tree = ast.parse((root / "src/openjev/research/reacher_memory_training.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MemoryTrainingSettings")
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert protocol.settings()[node.target.id] == ast.literal_eval(node.value)


def test_complete_synthetic_bindings_only_validate_metadata_never_authorize_execution():
    plan = protocol.settings(engineering=True)
    bindings, sources, inherited = synthetic_bindings(plan)
    result = protocol.validate_preparation_bindings(plan, bindings, expected_sources=sources, inherited_sources=inherited)
    assert result["validated_scope"] == "metadata_structure_only" and result["execution_authorized"] is False
    assert "actual source and lineage bytes" in result["remaining_checks"]
    assert len(sources) == 104


@pytest.mark.parametrize("section", protocol.preparation_requirements()["required_bindings"])
def test_missing_external_preparation_binding_rejected(section):
    plan = protocol.settings(engineering=True)
    bindings, sources, inherited = synthetic_bindings(plan)
    del bindings[section]
    with pytest.raises(ValueError):
        protocol.validate_preparation_bindings(plan, bindings, expected_sources=sources, inherited_sources=inherited)


@pytest.mark.parametrize("corruption", ["source", "old_source", "closure", "prerequisite", "pair", "fitted_initial", "fit", "class", "runtime", "roles", "cap", "audit"])
def test_external_bindings_reject_partial_or_mismatched_synthetic_contract(corruption):
    plan = protocol.settings(engineering=True)
    bindings, sources, inherited = synthetic_bindings(plan)
    if corruption == "source":
        bindings["sources"][protocol.NEW_SOURCE_PATHS[0]] = "0" * 64
    elif corruption == "old_source":
        sources["synthetic/old-0.py"] = bindings["sources"]["synthetic/old-0.py"] = "a" * 64
    elif corruption == "closure":
        del sources[protocol.NEW_SOURCE_PATHS[-1]]
        del bindings["sources"][protocol.NEW_SOURCE_PATHS[-1]]
    elif corruption == "prerequisite":
        bindings["lineage"]["completed_positive_gate_required"] = False
    elif corruption == "pair":
        del bindings["paired_training"]["pairs"]["pair2"]
    elif corruption == "fitted_initial":
        bindings["paired_training"]["pairs"]["pair0"]["initial_state_kind"] = "fitted_weights"
    elif corruption == "fit":
        del bindings["inherited_checkpoints"]["cached_gru-pair2"]
    elif corruption == "class":
        bindings["inherited_checkpoints"]["cached_gru-pair2"]["model_class"] = "GRUResidualRewardWorldModel"
    elif corruption == "runtime":
        bindings["runtime"]["torch"] = ""
    elif corruption == "roles":
        bindings["fresh_streams"]["role_manifest_sha256"] = "a" * 64
    elif corruption == "cap":
        bindings["capacity_and_caps"]["execution_seconds"] = True
    else:
        del bindings["audit_contract"]["cost_schema_sha256"]
    with pytest.raises(ValueError):
        protocol.validate_preparation_bindings(plan, bindings, expected_sources=sources, inherited_sources=inherited)
