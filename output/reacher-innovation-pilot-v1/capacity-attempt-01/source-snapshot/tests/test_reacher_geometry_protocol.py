"""Synthetic scope/RNG/union checks, using engineering namespaces only."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openjev.research import reacher_cache_protocol as cache
from openjev.research import reacher_geometry_protocol as protocol
from openjev.research import reacher_search_protocol as search
from openjev.research.reacher_adaptive_search import search as adaptive_search


def small_plan():
    plan = protocol.settings(engineering=True)
    plan.update(control_episodes=2, diagnostic_cases=1, diagnostic_root_steps=[12],
                hidden_size=4, mlp_width=7, bootstrap_samples=8)
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    return plan


def state_hashes(manifest):
    return {row["initial_state_sha256"] for row in manifest.values()}


def test_settings_no_rng_model_or_artifact_activity_and_independent_constants():
    with (patch.object(search, "named_seed", side_effect=AssertionError("settings derived seeds")),
          patch.object(np.random, "default_rng", side_effect=AssertionError("settings allocated RNG")),
          patch.object(torch, "Generator", side_effect=AssertionError("settings allocated Torch RNG")),
          patch("builtins.open", side_effect=AssertionError("settings read files"))):
        plan = protocol.settings()
    assert plan["new_fits"] == plan["new_optimizer_updates"] == 0
    assert plan["weights_unchanged"] is True
    assert plan["score_contract"]["geometry"]["native_reward_exact"] is False
    assert plan["score_contract"]["geometry"]["control_weight"] == 1.
    assert plan["score_contract"]["geometry"]["joint_limit_penalty"] is None
    assert protocol.validate_settings(plan) is plan
    plan["criterion"]["expected_checks"] = 99
    assert protocol.settings()["criterion"]["expected_checks"] == 25


def test_exact_fits_policies_coverage_and_exposed_root_order():
    plan = protocol.settings()
    fits = protocol.fit_manifest(plan)
    assert len(fits) == 6 and all(row["weights_reused"] is True for row in fits)
    assert [row["name"] for row in fits] == plan["fit_order"]
    assert {row["model_class"] for row in fits} == {"GRUResidualRewardWorldModel", "CachedObservationMLPWorldModel"}
    rows = protocol.execution_order(plan)
    assert len(rows) == 51 and len({row["path"] for row in rows}) == 51
    assert rows[0]["label"] == "residual_gru-pair0--learned"
    assert rows[1]["label"] == "residual_gru-pair0--geometry"
    assert rows[12]["reference"] == "known_state"
    coverage = protocol.coverage(plan)
    assert coverage["learned_control_rows"] == 36 and coverage["reference_rows"] == 15
    assert coverage["native_control_transitions"] == 163200
    assert coverage["diagnostic_roots"] == 48
    assert coverage["diagnostic_identity_slots_per_root"] == 76
    assert coverage["diagnostic_native_branches_maximum"] == 14592
    assert coverage["diagnostic_native_transitions_maximum"] == 175104
    assert coverage["diagnostic_model_searches"] == coverage["diagnostic_union_score_calls"] == 576
    assert coverage["diagnostic_search_candidate_evaluations"] == 147456
    assert coverage["diagnostic_union_candidate_evaluations_maximum"] == 43776
    assert coverage["diagnostic_search_imagined_transitions"] == 1769472
    assert coverage["diagnostic_union_imagined_transitions_maximum"] == 525312
    roots = protocol.diagnostic_manifest(plan)
    assert roots[0] == {"panel": "full", "case_index": 0, "step": 12, "history_fit": "residual_gru-pair0",
        "source_path": "control/full/residual_gru-pair0", "path": "diagnostic/full/000/012",
        "input_prefix": "planner/diagnostic/full/0/12", "horizon": 12}
    assert roots[1]["step"] == 32 and roots[16]["panel"] == "ordinary"
    assert roots[-1]["panel"] == "shift" and roots[-1]["case_index"] == 7
    assert "Previous completed" in plan["diagnostic_contract"]["history_scope"]
    assert plan["stage_order"] == ["authenticate_and_restore_all_six", "exposed_history_diagnostic", "fresh_closed_loop_control"]


def test_twenty_five_criteria_and_secondary_not_a_rescue():
    criteria = protocol.settings()["criterion"]
    assert criteria["expected_checks"] == 2 + 6 + 1 + 12 + 4 == 25
    assert criteria["geometry_vs_own_learned"] == {"mean_improvement": .03, "every_pair_nonworse": True}
    assert criteria["full_sensing_vs_own_learned"]["maximum_mean_degradation"] == .02
    assert criteria["competence"]["score_modes"] == ["learned", "geometry"]
    assert criteria["competence"]["gru_vs_zero_improvement"] == .10
    assert "secondary_only" in criteria["cached_mlp_comparisons"]
    assert "never rescue" in criteria["diagnostic"]
    assert "not prove equivalent" in criteria["fail_interpretation"]
    assert len(protocol.settings()["secondary_contrasts"]) == 3


@pytest.mark.parametrize("key,value", [
    ("arms", ["residual_gru"]), ("score_modes", ["geometry"]), ("pairs", ["pair0"]),
    ("panels", ["ordinary"]), ("references", ["zero"]), ("control_episodes", 63),
    ("steps", 49), ("ordinary_gap", 5), ("shift_gap", 11), ("new_fits", 1),
    ("new_optimizer_updates", 1), ("weights_unchanged", False), ("diagnostic_cases", 7),
    ("diagnostic_root_steps", [12]), ("diagnostic_branches", 3), ("planning_horizon", 11),
    ("hidden_size", 63), ("mlp_width", 106), ("noise_std", .06), ("reset_interventions", True),
])
def test_production_scope_drift_fails_before_any_draw(key, value):
    plan = protocol.settings()
    plan[key] = value
    with patch.object(np.random, "default_rng", side_effect=AssertionError("drew samples")), pytest.raises(ValueError):
        protocol.validate_settings(plan)


@pytest.mark.parametrize("key,value", [
    ("control_episodes", True), ("control_episodes", 65), ("diagnostic_cases", 9),
    ("diagnostic_root_steps", [32]), ("diagnostic_root_steps", [12., 32.]),
    ("diagnostic_root_steps", [12, 32, 40]), ("hidden_size", 65), ("mlp_width", 108),
    ("diagnostic_branches", 1), ("steps", 12), ("rng_namespace", protocol.SCORED_NAMESPACE),
])
def test_engineering_can_only_shrink_declared_sizes_and_root_prefix(key, value):
    plan = protocol.settings(engineering=True)
    plan[key] = value
    with pytest.raises(ValueError):
        protocol.validate_settings(plan, engineering=True)


def test_engineering_sizes_do_not_drop_fits_modes_panels_or_noise_branches():
    plan = small_plan()
    protocol.validate_settings(plan, engineering=True)
    assert protocol.coverage(plan)["diagnostic_roots"] == 3
    assert protocol.coverage(plan)["control_rows"] == 51
    assert plan["diagnostic_branches"] == 4
    for key in ("criterion", "score_contract", "diagnostic_contract", "search_parameters"):
        changed = copy.deepcopy(plan)
        changed[key]["unbound_change"] = True
        with pytest.raises(ValueError, match="setting changed"):
            protocol.validate_settings(changed, engineering=True)


def test_full_registry_and_all_concrete_filter_states_without_global_rng_changes():
    plan = protocol.settings(engineering=True)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    contract = protocol.stream_contract(plan)
    torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert len(contract["registry"]) == 941 and len(contract["generators"]) == 1069
    assert contract["torch_registry"] == contract["torch_generators"] == {}
    assert contract["new_torch_scored_streams"] == contract["training_and_stochastic_inference_rng_draws"] == 0
    assert contract["draws_for_manifest"] == 0
    assert contract["discarded_constructor_rng"]["calls"] == 6
    assert contract["discarded_constructor_rng"]["seed"] == 0
    assert contract["discarded_constructor_rng"]["outer_rng_restored"] is True
    assert "diagnostic/native_template/0" in contract["registry"]
    assert len(state_hashes(contract["generators"])) == 1069
    assert len(contract["engineering_exclusions"]) == 3
    assert len(contract["cache_study_exclusions"]) == len(contract["memory_study_exclusions"]) == 5
    assert all(value["draws_for_manifest"] == 0 for value in contract["generators"].values())
    assert not any(role.startswith(("fit/", "prediction/", "diagnostic/reset/", "diagnostic/sensor_schedule/"))
                   for role in contract["registry"])
    assert len([role for role in contract["registry"] if role.startswith("diagnostic/branch_noise/")]) == 192
    assert json.loads(json.dumps(contract)) == contract


def test_reduced_fixture_excludes_full_engineering_cache_and_memory_coverage():
    plan = small_plan()
    contract = plan["random_stream_contract"]
    for item in contract["engineering_exclusions"]:
        assert len(item["registry"]) == 941 and len(item["generators"]) == 1069
        assert item["coverage"]["diagnostic_cases"] == 8 and item["coverage"]["control_episodes"] == 64
    for item in contract["cache_study_exclusions"] + contract["memory_study_exclusions"]:
        assert len(item["registry"]) == 892 and len(item["generators"]) == 1020
        assert len(item["torch_registry"]) == 9
        assert not state_hashes(contract["generators"]) & state_hashes(item["generators"])
    assert {item["namespace"] for item in contract["cache_study_exclusions"]} == {
        cache.SCORED_NAMESPACE, *cache.ENGINEERING_NAMESPACES}


def test_manifests_never_call_numpy_sampling_methods_or_read_prior_files(monkeypatch):
    original = np.random.default_rng

    class StateOnly:
        def __init__(self, seed):
            self.bit_generator = original(seed).bit_generator

        def __getattr__(self, name):
            raise AssertionError(f"Manifest requested sampler {name}")

    monkeypatch.setattr(np.random, "default_rng", StateOnly)
    plan = protocol.settings(engineering=True)
    with patch("builtins.open", side_effect=AssertionError("prior artifact I/O")):
        result = protocol.stream_contract(plan, [{"old/fixture": 410}], [{"old/torch": 0}],
                                          [{"scope": "synthetic-prior"}])
    assert result["draws_for_manifest"] == 0
    assert len(result["prior_numpy_registry_sha256"]) == len(result["prior_torch_registry_sha256"]) == 1
    assert result["priors"] == [{"scope": "synthetic-prior"}]


def test_current_and_prior_aliases_fail_closed(monkeypatch):
    plan = protocol.settings(engineering=True)
    fresh = next(iter(protocol.registry(plan).values()))
    with pytest.raises(ValueError, match="Prior NumPy root"):
        protocol.stream_contract(plan, [{"renamed/prior": fresh}])
    with pytest.raises(ValueError, match="uint64"):
        protocol.stream_contract(plan, [{"invalid/bool": True}])
    with monkeypatch.context() as context:
        context.setattr(search, "named_seed", lambda namespace, role: 410)
        with pytest.raises(ValueError, match="collision"):
            protocol.stream_contract(plan)


@pytest.mark.parametrize("which", ["cache_root", "cache_child", "memory_child"])
def test_automatic_old_stream_exclusions_compare_concrete_state_aliases(which, monkeypatch):
    plan = protocol.settings(engineering=True)
    current = protocol.registry(plan)
    from openjev.research.reacher_random_streams import generator_manifest

    generators = generator_manifest(current)
    if which == "cache_root":
        entries = protocol.cache_stream_exclusions()
        entries[0]["registry"]["renamed_old_role"] = next(iter(current.values()))
        monkeypatch.setattr(protocol, "cache_stream_exclusions", lambda: entries)
    else:
        entry_fn = protocol.cache_stream_exclusions if which == "cache_child" else cache.memory_stream_exclusions
        entries = entry_fn()
        entries[0]["generators"]["renamed/old/child"] = generators["planner/particle_filter/0/process_noise"]
        if which == "cache_child":
            monkeypatch.setattr(protocol, "cache_stream_exclusions", lambda: entries)
        else:
            monkeypatch.setattr(cache, "memory_stream_exclusions", lambda: entries)
    with pytest.raises(ValueError, match="collision"):
        protocol.stream_contract(plan)


def test_control_schedules_match_frozen_reference_helpers_and_no_diagnostic_regeneration():
    plan = small_plan()
    for index in range(plan["control_episodes"]):
        phase = protocol.phase(plan, index)
        for panel in protocol.PANELS:
            actual = protocol.schedule(plan, index, panel)
            np.testing.assert_array_equal(actual, search.schedule(plan, index, panel))
            assert actual.dtype == np.bool_ and actual[0] and actual[-1]
        ordinary = protocol.schedule(plan, index, "ordinary")
        shifted = protocol.schedule(plan, index, "shift")
        assert (~ordinary).sum() == 12 and (~shifted).sum() == 20
        assert not ordinary[8+phase:14+phase].any() and not shifted[8+phase:18+phase].any()
    with patch.object(np.random, "default_rng", side_effect=AssertionError("full panel sampled phase")):
        assert protocol.schedule(plan, 0, "full").all()
    for split in ("diagnostic", "prediction"):
        with pytest.raises(ValueError, match="inherited"):
            protocol.schedule(plan, 0, "ordinary", split)


def test_bound_seed_validation_precedes_innovation_sampling(monkeypatch):
    plan = small_plan()
    role = "planner/control/0/cem/2"
    assert role in plan["random_stream_contract"]["registry"]
    plan["random_stream_contract"]["registry"][role] ^= 1
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("innovation draw before all bound roles checked"))
    with pytest.raises(ValueError, match="Bound seed"):
        protocol.draw_control_inputs(plan, 0)


def test_full_horizon_control_inputs_terminal_banks_and_serialized_roundtrip(tmp_path):
    plan = small_plan()
    for step in (0, 47, 49):
        inputs = protocol.draw_control_inputs(plan, step)
        assert inputs.initial.shape == (2, 64, 4, 2) and inputs.random_extra.shape == (2, 192, 4, 2)
        assert inputs.initial.dtype == np.float64 and not inputs.initial.flags.writeable
        stem = tmp_path / str(step)
        protocol.save_inputs(stem, inputs, f"planner/control/{step}")
        loaded = protocol.load_inputs(stem)
        assert inputs.identities() == loaded.identities()
        bank = protocol.common_bank(inputs, step, plan)
        np.testing.assert_array_equal(bank, cache.common_bank(inputs, step, plan))
        result = adaptive_search("rs64", inputs, lambda commands: -np.sum(commands**2, axis=(2, 3)),
                                 step=step, planning_horizon=12, action_block=3)
        np.testing.assert_array_equal(bank, result.sequences)
    with patch.object(np.random, "default_rng", side_effect=AssertionError("terminal draw")), pytest.raises(ValueError):
        protocol.draw_control_inputs(plan, 50)


def test_diagnostic_inputs_and_noise_are_common_reproducible_and_distinct():
    plan = small_plan()
    inputs = protocol.diagnostic_inputs(plan, "ordinary", 0, 12)
    assert inputs.initial.shape == (1, 64, 4, 2)
    assert inputs.identities() == protocol.diagnostic_inputs(plan, "ordinary", 0, 12).identities()
    assert inputs.identities() != protocol.diagnostic_inputs(plan, "shift", 0, 12).identities()
    assert protocol.common_bank(inputs, 12, plan).shape == (1, 64, 12, 2)
    noise = protocol.diagnostic_noise(plan, "ordinary", 0, 12)
    assert noise.shape == (4, 12, 2) and noise.dtype == np.float64 and np.isfinite(noise).all()
    np.testing.assert_array_equal(noise, protocol.diagnostic_noise(plan, "ordinary", 0, 12))
    assert all(not np.array_equal(noise[i], noise[j]) for i in range(4) for j in range(i))
    assert not np.array_equal(noise, protocol.diagnostic_noise(plan, "shift", 0, 12))
    real_noise = np.random.default_rng(protocol.seed(plan, "control/actuator_noise/0")).normal(0, .05, (4, 12, 2))
    assert not np.array_equal(noise, real_noise)
    for args in (("ordinary", 1, 12), ("ordinary", 0, 32), ("ordinary", 0, 13)):
        with pytest.raises(ValueError, match="Preselected"):
            protocol.diagnostic_inputs(plan, *args)
        with pytest.raises(ValueError, match="Preselected"):
            protocol.diagnostic_noise(plan, *args)


def union_inputs():
    plan = protocol.settings(engineering=True)
    bank = np.broadcast_to(np.arange(64, dtype=np.float32)[:, None, None] / 128., (64, 12, 2)).copy()
    labels = [protocol.policy_name(row["name"], mode) for row in protocol.fit_manifest(plan) for mode in protocol.SCORE_MODES]
    selected = {label: bank[index % 4].copy() for index, label in enumerate(labels)}
    return plan, bank, selected


def test_union_first_occurrence_mapping_all_slots_and_no_aliasing():
    plan, bank, selected = union_inputs()
    original = bank.copy()
    result = protocol.diagnostic_union(plan, bank, dict(reversed(list(selected.items()))))
    assert result["slot_commands"].shape == (76, 12, 2)
    assert result["commands"].shape == (64, 12, 2)
    np.testing.assert_array_equal(result["slot_to_unique"], list(range(64)) + [i % 4 for i in range(12)])
    np.testing.assert_array_equal(result["unique_first_slots"], np.arange(64))
    assert result["slot_ids"] == [f"common/{i}" for i in range(64)] + [f"selected/{label}" for label in selected]
    np.testing.assert_array_equal(result["commands"][result["slot_to_unique"]], result["slot_commands"])
    result["commands"][0] = -1
    np.testing.assert_array_equal(bank, original)
    assert np.all(selected[next(iter(selected))] == 0)


def test_union_maximum_and_byte_exact_signed_zero_contract():
    plan, bank, selected = union_inputs()
    for index, label in enumerate(selected):
        selected[label].fill(-float(index + 1) / 32.)
    result = protocol.diagnostic_union(plan, bank, selected)
    assert len(result["commands"]) == 76
    selected[next(iter(selected))].fill(-0.)
    result = protocol.diagnostic_union(plan, bank, selected)
    assert len(result["commands"]) == 76  # +0 initial and -0 selected are different bytes.
    assert result["unique_first_slots"][64] == 64


@pytest.mark.parametrize("kind", ["missing", "float64", "nan", "range", "horizon"])
def test_union_rejects_missing_or_malformed_candidates(kind):
    plan, bank, selected = union_inputs()
    label = next(iter(selected))
    if kind == "missing":
        del selected[label]
    elif kind == "float64":
        selected[label] = selected[label].astype(np.float64)
    elif kind == "horizon":
        selected[label] = selected[label][:11]
    else:
        selected[label][0, 0] = np.nan if kind == "nan" else 1.001
    with pytest.raises(ValueError):
        protocol.diagnostic_union(plan, bank, selected)
