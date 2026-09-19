"""Pure prospective contracts; only engineering namespaces consume samples.

Read completed protocol metadata to verify old pins, never weights or efficacy.
The frozen runner's prior_streams enumerates authenticated old metadata only.
No future study plan, manifest, readiness receipt or scored samples are created.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import reacher_geometry_study as previous
import torch

from openjev.research import reacher_geometry_memory_protocol as protocol
from openjev.research import reacher_search_protocol as search
from openjev.research.reacher_adaptive_search import search as adaptive_search

ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def previous_plan():
    path = ROOT / "evidence/reacher-geometry-score-v1/protocol/plan.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol.GEOMETRY_PLAN_SHA256
    return json.loads(raw)


@pytest.fixture(scope="module")
def priors():
    with (patch.object(previous.training, "_construct", side_effect=AssertionError("No model")),
          patch.object(previous, "make_env", side_effect=AssertionError("No native call")),
          patch.object(torch, "load", side_effect=AssertionError("No checkpoint"))):
        return previous.prior_streams()


@pytest.fixture(scope="module")
def engineering_plan(priors):
    plan = protocol.settings(engineering=True)
    plan.update(control_episodes=2, hidden_size=4, mlp_width=7, bootstrap_samples=8)
    plan["random_stream_contract"] = protocol.stream_contract(plan, *priors)
    return plan


def test_settings_do_not_read_allocate_rng_construct_models_or_derive_seeds():
    with (patch.object(search, "named_seed", side_effect=AssertionError("Seed derivation")),
          patch.object(np.random, "default_rng", side_effect=AssertionError("NumPy RNG")),
          patch.object(torch, "Generator", side_effect=AssertionError("Torch RNG")),
          patch("builtins.open", side_effect=AssertionError("File I/O"))):
        plan = protocol.settings()
        protocol.validate_settings(plan)
        assert len(protocol.restore_manifest(plan)) == 12
        assert len(protocol.criterion_manifest(plan)) == 25
        assert protocol.coverage(plan)["native_control_transitions"] == 163200
    assert plan["new_fits"] == plan["new_optimizer_updates"] == 0
    assert plan["stage_order"] == ["authenticate_and_restore_all_twelve", "fresh_closed_loop_control"]
    assert plan["score_contract"]["geometry"]["native_reward_exact"] is False
    assert plan["score_contract"]["geometry"]["control_weight"] == 1.
    plan["criterion"]["expected_checks"] = 1
    plan["source_contract"]["new_sources"].clear()
    assert protocol.settings()["criterion"]["expected_checks"] == 25
    assert len(protocol.settings()["source_contract"]["new_sources"]) == 10


def test_all_twelve_actual_classes_original_files_and_no_exposed_trajectories():
    plan = protocol.settings()
    fits = protocol.fit_manifest(plan)
    assert len(fits) == 12 and len({row["name"] for row in fits}) == 12
    assert [row["name"] for row in fits] == plan["fit_order"]
    assert {row["model_class"] for row in fits} == {
        "GRUResidualRewardWorldModel", "EncodedCurrentGRUWorldModel",
        "CachedObservationGRUWorldModel", "CachedObservationMLPWorldModel"}
    restored = protocol.restore_manifest(plan)
    assert len(restored) == 12
    for row in restored:
        assert row["weights_reused"] and row["new_optimizer_updates"] == 0
        assert row["source_path"] == f"fits/{row['name']}"
        assert row["initialization_path"] == f"initializations/{row['pair']}.pt"
        assert row["order_path"] == f"orders/{row['pair']}.pt"
        assert len(row["members"]) == 5
    names = protocol.inherited_members(plan)
    assert len(names) == 74
    assert sum(name.endswith("checkpoint.pt") for name in names) == 12
    assert not any(name.startswith("control/") for name in names)
    assert {"train.npz", "train.json"} <= set(names)


def test_exact_51_rows_geometry_and_matched_physics_search():
    plan = protocol.settings()
    rows = protocol.execution_order(plan)
    assert len(rows) == len({row["path"] for row in rows}) == 51
    assert rows == plan["execution_order"]
    assert rows[0]["path"] == "control/full/residual_gru-pair0"
    assert rows[3]["path"] == "control/full/cached_mlp-pair0"
    for panel in ("full", "ordinary", "shift"):
        local = [row for row in rows if row["panel"] == panel]
        assert len(local) == 17
        assert len([row for row in local if "fit" in row]) == 12
        assert [row["reference"] for row in local if "reference" in row] == [
            "known_state", "particle", "public_kinematic", "zero", "uniform"]
        for row in local:
            if row.get("reference") in ("zero", "uniform"):
                assert row["planner"] is row["score_mode"] is None
            else:
                assert row["planner"] == "cem256" and row["score_mode"] == "geometry"
    coverage = protocol.coverage(plan)
    assert coverage["learned_control_rows"] == 36
    assert coverage["physics_reference_rows"] == 9
    assert coverage["reference_rows"] == 15 and coverage["floor_rows"] == 6
    assert coverage["native_control_transitions"] == 163200
    assert coverage["learned_candidate_evaluations"] == 29491200
    assert coverage["physics_candidate_evaluations"] == 7372800
    assert coverage["learned_imagined_transitions"] == 314966016
    assert coverage["physics_nominal_transitions"] == 78741504
    assert coverage["geometry_candidate_samples"] == 393707520
    assert coverage["learned_selected_advances"] == 115200
    assert coverage["physics_selected_advances"] == 28800
    assert coverage["geometry_selected_samples"] == 144000
    assert coverage["physics_candidate_native_substeps"] == 157483008
    assert coverage["physics_selected_native_substeps"] == 57600
    assert coverage["exposed_diagnostic_roots"] == coverage["new_prediction_episodes"] == 0
    assert "not total-compute matched" in plan["cost_reporting"]["comparison"]


def test_gate_exactly_design_not_previous_geometry_or_cache_gate():
    plan = protocol.settings()
    checks = protocol.criterion_manifest(plan)
    assert len(checks) == len({row["name"] for row in checks}) == 25
    counts = {prefix: sum(row["name"].startswith(prefix) for row in checks)
              for prefix in ("gap_mean/", "gap_pair/", "full_mean/", "competence/")}
    assert counts == {"gap_mean/": 4, "gap_pair/": 12, "full_mean/": 2, "competence/": 7}
    assert [row for row in checks if row["treatment"] == "known_state"] == [{
        "name": "competence/ordinary/known_state", "panel": "ordinary",
        "treatment": "known_state", "control": "zero", "aggregation": "reference_vs_reference",
        "maximum_cost_multiplier": .9}]
    for row in checks:
        if row["name"].startswith("gap_mean/"):
            assert row["maximum_cost_multiplier"] == .97
        if row["name"].startswith("gap_pair/"):
            assert row["maximum_cost_multiplier"] == 1.
        if row["name"].startswith("full_mean/"):
            assert row["maximum_cost_multiplier"] == 1.02
        assert "cached_mlp" not in (row["treatment"], row["control"])
        assert row["treatment"] not in ("particle", "public_kinematic")
    assert plan["criterion"]["secondary_cannot_rescue_primary"]
    assert len(plan["secondary_contrasts"]) == 3
    assert "real-assimilation gating differs" in plan["criterion"]["limits"]


@pytest.mark.parametrize("key,value", [
    ("arms", ["residual_gru", "cached_mlp"]), ("pairs", ["pair0"]),
    ("score_modes", ["learned", "geometry"]), ("panels", ["ordinary", "shift"]),
    ("references", ["known_state", "zero"]), ("control_episodes", 63),
    ("steps", 49), ("dt", .01), ("noise_std", .06), ("hidden_size", 63),
    ("mlp_width", 106), ("ordinary_gap", 5), ("shift_gap", 9),
    ("new_fits", 1), ("new_optimizer_updates", 1), ("weights_unchanged", False),
    ("planner", "rs256"), ("physics_planner", "rs64"), ("planning_horizon", 6),
    ("action_block", 2), ("candidates", 256), ("particles", 16),
    ("exposed_diagnostic_roots", 48), ("new_prediction_episodes", 96),
    ("reset_interventions", True), ("diagnostic_cases", 8), ("score_mode", "learned"),
])
def test_production_rejects_scope_drift_and_obsolete_fields_without_any_draw(key, value):
    plan = protocol.settings()
    plan[key] = value
    with patch.object(np.random, "default_rng", side_effect=AssertionError("RNG activity")), pytest.raises(ValueError):
        protocol.validate_settings(plan)


@pytest.mark.parametrize("key,value", [
    ("control_episodes", True), ("control_episodes", 65), ("control_episodes", 0),
    ("hidden_size", 65), ("mlp_width", 108), ("bootstrap_samples", 0),
    ("steps", 3), ("physics_planner", "rs64"), ("planning_horizon", 2),
    ("rng_namespace", protocol.SCORED_NAMESPACE), ("engineering", False),
])
def test_engineering_only_shrinks_declared_dimensions(key, value):
    plan = protocol.settings(engineering=True)
    plan[key] = value
    with pytest.raises(ValueError):
        protocol.validate_settings(plan, engineering=True)


def test_engineering_cannot_drop_models_criteria_or_physics_costs(engineering_plan):
    plan = copy.deepcopy(engineering_plan)
    assert protocol.validate_settings(plan, engineering=True) is plan
    assert protocol.coverage(plan)["control_rows"] == 51
    assert protocol.coverage(plan)["native_control_transitions"] == 5100
    for key in ("criterion", "score_contract", "reference_contract", "cost_reporting", "lineage_contract"):
        changed = copy.deepcopy(plan)
        changed[key]["unbound"] = True
        with pytest.raises(ValueError, match="setting changed"):
            protocol.validate_settings(changed, engineering=True)


def test_completed_metadata_binds_prior_hash_lists_and_source_contract(previous_plan, priors):
    stream = previous_plan["random_stream_contract"]
    assert identity(stream["prior_numpy_registry_sha256"]) == protocol.PRIOR_BINDINGS["numpy_hash_list_sha256"]
    assert identity(stream["prior_torch_registry_sha256"]) == protocol.PRIOR_BINDINGS["torch_hash_list_sha256"]
    assert identity(stream["priors"]) == protocol.PRIOR_BINDINGS["descriptors_sha256"]
    assert identity(previous_plan["sources"]) == protocol.INHERITED_SOURCE_MANIFEST_SHA256
    actual_np, actual_torch = protocol.validate_prior_registries(*priors)
    assert [identity(row) for row in actual_np] == stream["prior_numpy_registry_sha256"]
    assert [identity(row) for row in actual_torch] == stream["prior_torch_registry_sha256"]
    assert protocol.lineage_contract()["cache"]["historical_gate_passed"] is False
    assert protocol.lineage_contract()["geometry"]["historical_gate_passed"] is True


def test_no_source_readiness_claim_or_missing_future_files_accepted(previous_plan):
    old = previous_plan["sources"]
    members = protocol.source_membership(old)
    assert len(members) == 90 and len(set(members)) == 90
    assert "scripts/reacher_geometry_memory_study.py" in members
    assert "scripts/audit_reacher_geometry_memory_study.py" in members
    assert "src/openjev/research/reacher_geometry_physics.py" in members
    with pytest.raises(ValueError, match="ninety-source"):
        protocol.validate_source_hashes(old, old)
    synthetic = {**old, **{name: "f" * 64 for name in protocol.NEW_SOURCES}}
    assert protocol.validate_source_hashes(synthetic, old) is synthetic
    # These are metadata-only hashes, not a claim that future files exist.
    changed = dict(synthetic)
    changed[next(iter(old))] = "0" * 64
    with pytest.raises(ValueError, match="inherited source changed"):
        protocol.validate_source_hashes(changed, old)
    del changed[protocol.NEW_SOURCES[-1]]
    with pytest.raises(ValueError, match="ninety-source"):
        protocol.validate_source_hashes(changed, old)
    with pytest.raises(ValueError, match="eighty-source"):
        protocol.source_membership({**old, "unbound.py": "0" * 64})


@pytest.mark.parametrize("kind", ["missing_numpy", "unknown_numpy", "reordered_numpy", "changed_numpy",
    "missing_torch", "unknown_torch", "changed_torch", "missing_descriptor", "changed_descriptor"])
def test_missing_unknown_or_changed_prior_cannot_be_silently_dropped(priors, kind):
    np_prior, torch_prior, descriptors = copy.deepcopy(priors)
    if kind == "missing_numpy":
        np_prior.pop()
    elif kind == "unknown_numpy":
        np_prior.append({"undeclared": 410})
    elif kind == "reordered_numpy":
        np_prior[0], np_prior[1] = np_prior[1], np_prior[0]
    elif kind == "changed_numpy":
        np_prior[0][next(iter(np_prior[0]))] += 1
    elif kind == "missing_torch":
        torch_prior.pop()
    elif kind == "unknown_torch":
        torch_prior.append({"undeclared": 410})
    elif kind == "changed_torch":
        torch_prior[0][next(iter(torch_prior[0]))] += 1
    elif kind == "missing_descriptor":
        descriptors.pop()
    else:
        descriptors[-1]["namespace"] = "unknown"
    with patch.object(np.random, "default_rng", side_effect=AssertionError("RNG before prior authentication")), pytest.raises(ValueError, match="prior"):
        protocol.stream_contract(protocol.settings(engineering=True), np_prior, torch_prior, descriptors)


def test_empty_caller_priors_fail_closed():
    with pytest.raises(ValueError, match="prior NumPy"):
        protocol.stream_contract(protocol.settings(engineering=True))


def test_full_exclusions_have_no_draws_and_no_global_rng_changes(priors):
    plan = protocol.settings(engineering=True)
    torch_before, np_before = torch.get_rng_state().clone(), np.random.get_state()
    original = np.random.default_rng

    class NoDraws:
        def __init__(self, *args, **kwargs):
            self.bit_generator = original(*args, **kwargs).bit_generator

        def __getattr__(self, name):
            raise AssertionError(f"Manifest requested sampling method {name}")

    with patch.object(np.random, "default_rng", NoDraws):
        contract = protocol.stream_contract(plan, *priors)
    assert torch.equal(torch_before, torch.get_rng_state())
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert len(contract["registry"]) == 508
    assert len(contract["generators"]) == 636
    assert len({row["initial_state_sha256"] for row in contract["generators"].values()}) == 636
    assert not any("diagnostic" in name or "prediction" in name for name in contract["registry"])
    assert contract["new_torch_scored_streams"] == contract["training_and_stochastic_inference_rng_draws"] == 0
    assert contract["torch_registry"] == contract["torch_generators"] == {}
    assert contract["discarded_constructor_rng"]["calls"] == 12
    assert contract["discarded_constructor_rng"]["seed"] == 0
    assert contract["draws_for_manifest"] == 0
    assert len(contract["namespace_exclusions"]) == 4
    assert protocol.SCORED_NAMESPACE in {row["namespace"] for row in contract["namespace_exclusions"]}
    assert len(contract["geometry_study_exclusions"]) == 5
    for excluded in contract["geometry_study_exclusions"]:
        assert len(excluded["registry"]) == 941
        assert len(excluded["generators"]) == 1069
        assert "diagnostic/native_template/0" in excluded["registry"]
        assert "diagnostic/branch_noise/shift/7/32/3" in excluded["registry"]
    assert len(contract["cache_study_exclusions"]) == len(contract["memory_study_exclusions"]) == 5
    for excluded in contract["namespace_exclusions"]:
        assert len(excluded["registry"]) == 508 and len(excluded["generators"]) == 636
    assert contract["prior_bindings"] == protocol.PRIOR_BINDINGS


def test_tiny_engineering_still_excludes_full_size_other_namespaces(engineering_plan):
    contract = engineering_plan["random_stream_contract"]
    assert len(contract["registry"]) == 260 and len(contract["generators"]) == 264
    assert all(len(item["registry"]) == 508 for item in contract["namespace_exclusions"])
    for item in contract["namespace_exclusions"] + contract["geometry_study_exclusions"]:
        assert not set(contract["registry"].values()) & set(item["registry"].values())


def test_known_new_literal_generators_attributed_by_source_and_concrete_state(engineering_plan):
    items = engineering_plan["random_stream_contract"]["engineering_literal_exclusions"]
    assert len(items) == 4
    controller, physics, runner, auditor = items
    assert controller["source_path"] == "tests/test_reacher_geometry_memory_control.py"
    assert controller["numpy_registry"] == {
        "controller/innovations": 410, "controller/native_reset": 410,
        "controller/native_actuator_noise": 410}
    assert controller["torch_registry"] == {"controller/global_fixture": 410, "controller/model_construction": 410}
    assert physics["source_path"] == "tests/test_reacher_geometry_physics.py"
    assert physics["numpy_registry"] == {"physics/innovations": 410, "physics/native_reset": 410}
    assert physics["torch_registry"] == physics["torch_generators"] == {}
    assert runner["source_path"] == "tests/test_reacher_geometry_memory_study.py"
    assert runner["numpy_registry"] == {"runner/innovations": 410, "runner/native_reset": 410,
                                        "runner/native_actuator_noise": 410}
    assert runner["torch_registry"] == {"runner/model_construction": 410}
    assert auditor["source_path"] == "tests/test_audit_reacher_geometry_memory_study.py"
    assert auditor["numpy_registry"] == {"auditor/innovations": 410, "auditor/native_reset": 410}
    assert auditor["torch_registry"] == auditor["torch_generators"] == {}
    np_hashes = {row["initial_state_sha256"] for item in items for row in item["generators"].values()}
    assert len(np_hashes) == 1  # Deliberate repeated engineering literal, not independent samples.
    current = engineering_plan["random_stream_contract"]["generators"]
    assert not np_hashes & {row["initial_state_sha256"] for row in current.values()}
    assert all(item["source_hash_binding"] == "plan.sources[source_path]" for item in items)


def test_root_collision_with_authenticated_literal_prior_is_rejected(priors):
    original = search.named_seed

    def force_collision(namespace, role):
        if namespace == protocol.ENGINEERING_NAMESPACES[0] and role == "control/reset/0":
            return 410
        return original(namespace, role)

    with patch.object(search, "named_seed", force_collision), pytest.raises(ValueError, match="Prior NumPy root collision"):
        protocol.stream_contract(protocol.settings(engineering=True), *priors)


def test_bound_registry_must_be_complete_and_unmodified_before_drawing(engineering_plan):
    for change in ("missing", "changed", "namespace"):
        plan = copy.deepcopy(engineering_plan)
        contract = plan["random_stream_contract"]
        if change == "missing":
            contract["registry"].pop("analysis/bootstrap/0")
        elif change == "changed":
            contract["registry"]["analysis/bootstrap/0"] += 1
        else:
            contract["namespace"] = protocol.SCORED_NAMESPACE
        with patch.object(np.random, "default_rng", side_effect=AssertionError("Drew before binding check")), pytest.raises(ValueError, match="registry differs"):
            protocol.draw_control_inputs(plan, 0)
    with pytest.raises(ValueError, match="Known geometry-memory"):
        protocol.seed(engineering_plan, "diagnostic/native_template/0")


def test_public_schedule_actual_six_and_ten_missing_steps_with_same_phase(engineering_plan):
    plan = engineering_plan
    for index in range(2):
        offset = protocol.phase(plan, index)
        assert 0 <= offset < 4
        full = protocol.schedule(plan, index, "full")
        ordinary = protocol.schedule(plan, index, "ordinary")
        shifted = protocol.schedule(plan, index, "shift")
        assert full.dtype == np.bool_ and full.shape == (51,) and full.all()
        assert (~ordinary).sum() == 12 and (~shifted).sum() == 20
        for start in (8 + offset, 28 + offset):
            assert ordinary[start - 1] and ordinary[start + 6]
            assert not ordinary[start:start + 6].any()
            assert shifted[start - 1] and shifted[start + 10]
            assert not shifted[start:start + 10].any()
        assert np.array_equal(protocol.schedule(plan, index, "ordinary"), ordinary)
        assert np.all(shifted <= ordinary) and ordinary[0] and shifted[-1]
    for args in ((2, "full", "control"), (0, "unknown", "control"), (0, "full", "prediction"), (True, "full", "control")):
        with pytest.raises(ValueError):
            protocol.schedule(plan, *args)


def test_engineering_proposals_shared_reproducible_terminal_clipped_and_cem_budget(engineering_plan, tmp_path):
    plan = engineering_plan
    inputs = protocol.draw_control_inputs(plan, 49)
    repeated = protocol.draw_control_inputs(plan, 49)
    assert inputs.initial.shape == (2, 64, 4, 2)
    assert inputs.random_extra.shape == (2, 192, 4, 2)
    assert [value.shape for value in inputs.cem] == [(2, 64, 4, 2), (2, 64, 4, 2), (2, 63, 4, 2)]
    assert np.array_equal(inputs.initial, repeated.initial)
    assert not np.array_equal(inputs.initial, protocol.draw_control_inputs(plan, 48).initial)
    initial = protocol.common_bank(inputs, 49, plan)
    assert initial.shape == (2, 64, 1, 2) and initial.dtype == np.float32
    observed = []

    def score(bank):
        observed.append(bank.copy())
        return -(bank ** 2).sum(axis=(2, 3), dtype=np.float32)

    result = adaptive_search("cem256", inputs, score, step=49, steps=50, planning_horizon=12, action_block=3)
    assert [bank.shape[1] for bank in observed] == [64, 64, 64, 64]
    assert np.array_equal(observed[0], initial)
    assert np.isfinite(result.selected_actions).all()
    assert all(bank.shape[2] == 1 for bank in observed)
    stem = tmp_path / "engineering-inputs"
    protocol.save_inputs(stem, inputs, "planner/control/49")
    loaded = protocol.load_inputs(stem)
    assert np.array_equal(loaded.initial, inputs.initial)
    for step in (-1, 50, True, 1.):
        with pytest.raises(ValueError):
            protocol.draw_control_inputs(plan, step)
