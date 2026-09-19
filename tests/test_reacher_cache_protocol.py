"""Synthetic engineering-only checks; no training, models or scored cohorts."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openjev.research import reacher_cache_protocol as protocol
from openjev.research import reacher_memory_protocol as memory_protocol
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_adaptive_search import search
from openjev.research.reacher_memory_training import MemoryTrainingSettings


def small_plan():
    plan = protocol.settings(engineering=True)
    plan.update(train_episodes=128, epochs=2, control_episodes=2, prediction_episodes=3,
                hidden_size=4, mlp_width=7, bootstrap_samples=8)
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    return plan


def test_exact_five_arm_membership_rotation_and_no_reset_rows():
    plan = protocol.settings()
    protocol.validate_settings(plan)
    fits = protocol.fit_manifest(plan)
    assert len(fits) == len({row["name"] for row in fits}) == 15
    for index, pair in enumerate(protocol.PAIRS):
        rows = [r for r in fits if r["pair"] == pair]
        assert [r["arm"] for r in rows] == [*protocol.ARMS[index:], *protocol.ARMS[:index]]
        assert {r["initialization_role"] for r in rows if r["arm"] not in ("packet_mlp", "cached_mlp")} == {f"fit/gru/{pair}"}
        assert {r["initialization_role"] for r in rows if r["arm"] in ("packet_mlp", "cached_mlp")} == {f"fit/mlp/{pair}"}
        assert {r["minibatch_role"] for r in rows} == {f"fit/minibatch/{pair}"}
        assert all(r["model_class"] == protocol.MODEL_CLASSES[r["arm"]] for r in rows)
    rows = protocol.execution_order(plan)
    assert len(rows) == len({r["path"] for r in rows}) == 60
    assert sum("fit" in r for r in rows) == 45
    assert sum("reference" in r for r in rows) == 15
    assert not any(r["reset"] for r in rows)
    assert sum(r.get("reference") == "public_kinematic" for r in rows) == 3
    scope = protocol.coverage(plan)
    assert scope["optimizer_updates_per_fit"] == 1152
    assert scope["total_optimizer_updates"] == 17280
    assert scope["native_control_transitions"] == 192000
    assert scope["native_prediction_transitions"] == 4800
    assert scope["criterion_checks"] == 28
    assert scope["fit_names"] == scope["prediction_names"] == plan["fit_order"]


def test_recipe_matches_anchor_trainer_without_constructing_models():
    plan = protocol.settings()
    trainer = MemoryTrainingSettings().configuration()
    for key, value in trainer.items():
        if key == "optimizer":
            assert json.loads(json.dumps(value)) == plan[key]
        elif key not in ("run_status_authority", "dtype", "device"):
            assert plan[key] == value
    assert plan["prediction_horizons"] == [1, 3, 7]
    assert plan["auxiliary_horizons"] == plan["prediction_horizons"]
    assert "later CEM proposals adapt" in plan["search_pairing"]
    assert "not equal" in plan["cost_reporting"]["comparison"]
    assert "status" not in protocol.CRITERION
    assert plan["engineering_rng_namespaces"] == list(protocol.ENGINEERING_NAMESPACES)


def test_defaults_do_not_derive_seeds_and_mutation_does_not_change_future_settings():
    with patch.object(search_protocol, "named_seed", side_effect=AssertionError("defaults selected a seed")):
        plan = protocol.settings()
    plan["criterion"]["competence"]["references"].clear()
    plan["search_parameters"]["callback_sizes"]["cem256"].pop()
    assert protocol.settings()["criterion"]["competence"]["references"] == ["known_state", "particle"]
    assert protocol.settings()["search_parameters"]["callback_sizes"]["cem256"] == [64]*4


@pytest.mark.parametrize("key,value", [
    ("arms", list(protocol.ARMS)[:-1]), ("pairs", ["pair0", "pair1"]),
    ("references", ["known_state", "particle", "zero", "uniform"]),
    ("planner", "rs256"), ("ordinary_gap", 7), ("shift_gap", 11),
    ("prediction_horizons", [1, 3, 6]), ("auxiliary_horizons", [1, 3, 6]),
    ("search_pairing", "all full CEM banks are identical"), ("objective", "latent"),
    ("reset_interventions", True), ("hidden_size", 63), ("mlp_width", 106),
    ("train_episodes", 767), ("prediction_episodes", 95), ("control_episodes", 63),
    ("steps", 49), ("epochs", 47), ("batch_size", 16),
    ("rng_namespace", "reacher-objective-ablation-v1-scored"),
])
def test_production_settings_fail_closed_on_scope_or_recipe_changes(key, value):
    plan = protocol.settings()
    plan[key] = value
    with pytest.raises(ValueError):
        protocol.validate_settings(plan)


@pytest.mark.parametrize("mutation", [
    lambda p: p["criterion"]["persistent_vs_cache"].update(mean_improvement=.04),
    lambda p: p["criterion"]["persistent_vs_cache"].update(every_pair_nonworse=1),
    lambda p: p["criterion"].update(expected_checks=23),
    lambda p: p["criterion"]["competence"]["references"].pop(),
    lambda p: p["optimizer"].update(foreach=0),
    lambda p: p["model_classes"].update(cached_gru="GRUResidualRewardWorldModel"),
    lambda p: p["fit_order"].reverse(),
    lambda p: p["execution_order"].pop(),
    lambda p: p["engineering_rng_namespaces"].pop(),
])
def test_nested_type_membership_order_and_gate_drift_rejected_for_engineering_too(mutation):
    plan = protocol.settings(engineering=True)
    mutation(plan)
    with pytest.raises(ValueError):
        protocol.validate_settings(plan, engineering=True)


def test_engineering_reductions_must_stay_inside_declared_full_size_exclusions():
    plan = small_plan()
    protocol.validate_settings(plan, engineering=True)
    with pytest.raises(ValueError):
        protocol.validate_settings(plan)
    for key, value in (("control_episodes", 65), ("prediction_episodes", 97), ("epochs", 49),
                       ("hidden_size", True), ("mlp_width", 108), ("batch_size", 31)):
        changed = copy.deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError):
            protocol.validate_settings(changed, engineering=True)
    for namespace in (protocol.SCORED_NAMESPACE, "reacher-cache-unregistered"):
        changed = copy.deepcopy(plan)
        changed["rng_namespace"] = namespace
        with pytest.raises(ValueError):
            protocol.validate_settings(changed, engineering=True)


def test_complete_engineering_streams_are_isolated_without_draws_or_global_mutation():
    plan = protocol.settings(engineering=True)
    torch_before, numpy_before = torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state())
    with (patch.object(torch, "randperm", side_effect=AssertionError("manifest drew samples")),
          patch.object(torch, "randn", side_effect=AssertionError("manifest drew samples")),
          patch.object(search_protocol, "draw_inputs", side_effect=AssertionError("manifest drew inputs"))):
        contract = protocol.stream_contract(plan)
    torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert len(contract["registry"]) == 892 and len(contract["generators"]) == 1020
    assert len(contract["torch_registry"]) == len(contract["torch_generators"]) == 9
    assert contract["draws_for_manifest"] == 0
    assert all(row["draws_for_manifest"] == 0 for row in contract["generators"].values())
    assert all(row["draws_for_manifest"] == 0 for row in contract["torch_generators"].values())
    assert set(contract["generators"]) - set(contract["registry"]) == {
        f"planner/particle_filter/{i}/{purpose}" for i in range(64)
        for purpose in ("initial", "process_noise", "resample")}
    assert len(contract["engineering_exclusions"]) == 3
    assert json.loads(json.dumps(contract)) == contract


def test_small_fixture_excludes_full_coverage_of_each_other_engineering_namespace():
    plan = small_plan()
    contract = plan["random_stream_contract"]
    assert len(contract["registry"]) < 892
    for exclusion in contract["engineering_exclusions"]:
        assert len(exclusion["registry"]) == 892 and len(exclusion["generators"]) == 1020
        assert len(exclusion["torch_registry"]) == 9
        assert exclusion["coverage"]["control_episodes"] == 64
        assert exclusion["coverage"]["prediction_episodes"] == 96
        assert not set(contract["registry"].values()) & set(exclusion["registry"].values())
        assert not set(contract["torch_registry"].values()) & set(exclusion["torch_registry"].values())
    assert all("public_kinematic" not in role for role in contract["registry"])


def test_prior_registries_and_provenance_bound_without_reading_files():
    plan = protocol.settings(engineering=True)
    prior_numpy = {"historical/environment": 271}
    prior_torch = {"historical/initialization": 271, "historical/minibatch": 4100271}
    descriptors = [{"kind": "synthetic-prior", "plan_sha256": "a"*64}]
    with patch("builtins.open", side_effect=AssertionError("protocol read artifacts")):
        contract = protocol.stream_contract(plan, [prior_numpy], [prior_torch], descriptors)
    assert len(contract["prior_numpy_registry_sha256"]) == len(contract["prior_torch_registry_sha256"]) == 1
    assert contract["priors"] == descriptors
    descriptors[0]["kind"] = "mutated"
    assert contract["priors"][0]["kind"] == "synthetic-prior"


def test_prior_numpy_and_torch_high_bit_alias_collisions_rejected():
    plan = small_plan()
    numpy_seed = next(iter(protocol.registry(plan).values()))
    torch_seed = next(iter(protocol.torch_registry(plan).values()))
    with pytest.raises(ValueError, match="Prior NumPy"):
        protocol.stream_contract(plan, [{"historical/renamed": numpy_seed}])
    for old in (torch_seed, torch_seed+(1 << 32)):
        with pytest.raises(ValueError, match="Prior Torch"):
            protocol.stream_contract(plan, prior_torch_registries=[{"historical/renamed": old}])
    with pytest.raises(ValueError, match="uint64"):
        protocol.stream_contract(plan, [{"historical/bool": True}])


def test_cross_role_collisions_fail_closed(monkeypatch):
    plan = protocol.settings(engineering=True)
    with monkeypatch.context() as context:
        context.setattr(search_protocol, "named_seed", lambda namespace, role: 19)
        with pytest.raises(ValueError, match="NumPy root"):
            protocol.stream_contract(plan)
    with monkeypatch.context() as context:
        context.setattr(protocol, "seed32", lambda namespace, role: 19)
        with pytest.raises(ValueError, match="Torch root"):
            protocol.stream_contract(plan)


def test_shared_initialization_and_orders_with_separate_mlp_and_exception_restoration():
    plan = small_plan()
    before = torch.get_rng_state().clone()
    draws = []
    for _ in range(3):
        with protocol.initialization_rng(plan, "pair0", "gru"):
            draws.append(torch.randn(8))
    for value in draws[1:]:
        torch.testing.assert_close(draws[0], value, rtol=0, atol=0)
    with protocol.initialization_rng(plan, "pair0", "mlp"):
        mlp_draw = torch.randn(8)
    assert not torch.equal(draws[0], mlp_draw)
    with (pytest.raises(RuntimeError, match="synthetic failure"),
          protocol.initialization_rng(plan, "pair0", "gru")):
        torch.randn(3)
        raise RuntimeError("synthetic failure")
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    orders = protocol.minibatch_orders(plan, "pair0")
    assert orders.shape == (2,128)
    torch.testing.assert_close(orders, protocol.minibatch_orders(plan,"pair0"), rtol=0, atol=0)
    for row in orders:
        torch.testing.assert_close(row.sort().values, torch.arange(128))
    assert not torch.equal(orders, protocol.minibatch_orders(plan,"pair1"))
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


def test_wrong_bound_seed_namespace_and_unknown_role_fail_before_draw():
    plan = small_plan()
    for key, role in (("registry", "control/reset/0"), ("torch_registry", "fit/gru/pair0")):
        changed = copy.deepcopy(plan)
        changed["random_stream_contract"][key][role] += 1
        with pytest.raises(ValueError, match="role-derived"):
            protocol.seed(changed, role)
    with pytest.raises(ValueError, match="Unknown"):
        protocol.seed(plan,"fit/predictor/pair0")
    changed = copy.deepcopy(plan)
    changed["random_stream_contract"]["namespace"] = "wrong"
    with pytest.raises(ValueError, match="namespace"):
        protocol.seed(changed,"control/reset/0")
    changed = copy.deepcopy(plan)
    changed["random_stream_contract"]["registry"]["planner/control/0/cem/3"] += 1
    with (patch.object(np.random,"default_rng",side_effect=AssertionError("partial innovation draw")),
          pytest.raises(ValueError, match="role-derived")):
        protocol.draw_control_inputs(changed,0)


def test_schedule_phase_pairing_gap_lengths_and_terminal_observation():
    plan = small_plan()
    for split in ("control", "prediction"):
        for index in range(plan[f"{split}_episodes"]):
            offset = protocol.phase(plan,index,split)
            masks = {panel:protocol.schedule(plan,index,panel,split) for panel in protocol.PANELS}
            assert all(v.dtype == np.bool_ and v.shape == (51,) and v[0] and v[-1] for v in masks.values())
            assert masks["full"].all()
            assert (~masks["ordinary"]).sum() == 12 and (~masks["shift"]).sum() == 20
            for panel in ("ordinary", "shift"):
                starts = np.flatnonzero(masks[panel][:-1] & ~masks[panel][1:])+1
                assert starts.tolist() == [8+offset,28+offset]
                np.testing.assert_array_equal(masks[panel], search_protocol.schedule(plan,index,panel,split))
    for index, split in ((-1,"control"),(2,"control"),(3,"prediction"),(0,"train"),(True,"control")):
        with pytest.raises(ValueError):
            protocol.schedule(plan,index,"ordinary",split)


def test_full_innovations_and_reference_bank_match_frozen_search_without_terminal_wrap(tmp_path):
    plan = small_plan()
    for step in (0,47,49):
        inputs = protocol.draw_control_inputs(plan,step)
        reference = search_protocol.draw_inputs(plan,f"planner/control/{step}",2)
        assert inputs.identities() == reference.identities()
        assert inputs.initial.shape == (2,64,4,2)
        assert inputs.random_extra.shape == (2,192,4,2)
        assert [x.shape for x in inputs.cem] == [(2,64,4,2),(2,64,4,2),(2,63,4,2)]
        bank = protocol.common_bank(inputs,step,plan)
        expected = search("rs64",inputs,lambda seq:np.zeros(seq.shape[:2],dtype=np.float32),step=step)
        np.testing.assert_array_equal(bank,expected.sequences)
        assert bank.shape == (2,64,min(12,50-step),2)
        stem = tmp_path/f"inputs-{step}"
        search_protocol.save_inputs(stem,inputs,f"planner/control/{step}")
        assert search_protocol.load_inputs(stem).identities() == inputs.identities()
    for step in (-1,50,51,True):
        with pytest.raises(ValueError,match="terminal"):
            protocol.draw_control_inputs(plan,step)
        with pytest.raises(ValueError,match="terminal"):
            protocol.common_bank(inputs,step,plan)


def test_actual_classes_feature_contract_and_gate_control_are_bound():
    expected = {
        'residual_gru': 'GRUResidualRewardWorldModel',
        'encoded_current_gru': 'EncodedCurrentGRUWorldModel',
        'cached_gru': 'CachedObservationGRUWorldModel',
        'packet_mlp': 'FeedForwardObservationWorldModel',
        'cached_mlp': 'CachedObservationMLPWorldModel',
    }
    plan = protocol.settings(engineering=True)
    assert plan['model_classes'] == expected
    feature = plan['feature_contract']
    assert len(feature['encoder_order']) == 8
    assert feature['encoder_order'][-2:] == ['actual_validity', 'actual_age_seconds']
    assert feature['age_tolerance'] == {'atol': 1e-6, 'rtol': 1e-5, 'visible_age_exact_zero': True}
    assert feature['no_motion_features'] is True
    assert 'isolated gate-removal effect is not estimated' in feature['gate_control']
    for key,value in (
        ('no_motion_features',False),
        ('encoder_order',['angle']*8),
        ('age_tolerance',{'atol':1e-4,'rtol':1e-5,'visible_age_exact_zero':True}),
    ):
        changed=copy.deepcopy(plan)
        changed['feature_contract'][key]=value
        with pytest.raises(ValueError,match='feature_contract'):
            protocol.validate_settings(changed,engineering=True)


def test_28_primary_checks_keep_both_caches_and_secondary_effects_separate():
    plan=protocol.settings()
    criterion=plan['criterion']
    cache=criterion['persistent_vs_cache']
    assert cache=={'comparators':['cached_gru','cached_mlp'],'mean_improvement':.03,'every_pair_nonworse':True}
    assert criterion['full_sensing_vs_cache']=={
        'comparators':['cached_gru','cached_mlp'],'maximum_mean_degradation':.02}
    assert criterion['competence']['persistent_vs_zero_improvement']==.10
    assert criterion['competence']['reference_vs_zero_improvement']==.10
    gaps, pairs, comparators = 2, 3, 2
    assert criterion['expected_checks']==gaps*comparators + gaps*pairs*comparators + comparators + gaps*pairs + gaps*2 == 28
    assert criterion['all_checks_required'] is True
    assert 'not proof of equivalence' in criterion['fail_interpretation']
    assert plan['secondary_contrasts']==[
        {'name':'gru_cache_information','treatment':'cached_gru','control':'encoded_current_gru'},
        {'name':'mlp_cache_information','treatment':'cached_mlp','control':'packet_mlp'},
    ]
    assert 'different blackout-training gradients' in plan['secondary_interpretation']
    for key in ('persistent_vs_cache','full_sensing_vs_cache'):
        changed=copy.deepcopy(plan)
        changed['criterion'][key]['comparators'].pop()
        with pytest.raises(ValueError):
            protocol.validate_settings(changed)
    changed=copy.deepcopy(plan)
    changed['secondary_contrasts'][0]['control']='residual_gru'
    with pytest.raises(ValueError):
        protocol.validate_settings(changed)


def test_all_prior_memory_namespaces_are_automatically_excluded_at_full_coverage():
    plan=small_plan()
    contract=plan['random_stream_contract']
    excluded=contract['memory_study_exclusions']
    assert [row['namespace'] for row in excluded]==[memory_protocol.SCORED_NAMESPACE,*memory_protocol.ENGINEERING_NAMESPACES]
    current_np={row['initial_state_sha256'] for row in contract['generators'].values()}
    current_torch={row['initial_state_sha256'] for row in contract['torch_generators'].values()}
    for previous in excluded:
        old=memory_protocol.settings()
        old['rng_namespace']=previous['namespace']
        assert previous['registry']==memory_protocol.registry(old)
        assert previous['torch_registry']==memory_protocol.torch_registry(old)
        assert len(previous['registry'])==892 and len(previous['generators'])==1020
        assert len(previous['torch_registry'])==9
        assert previous['coverage']['arms']==list(memory_protocol.ARMS)
        assert previous['coverage']['control_episodes']==64
        assert previous['coverage']['prediction_episodes']==96
        assert not current_np & {row['initial_state_sha256'] for row in previous['generators'].values()}
        assert not current_torch & {row['initial_state_sha256'] for row in previous['torch_generators'].values()}


@pytest.mark.parametrize('collision',['numpy_root','numpy_child_state','torch_state'])
def test_automatic_memory_exclusion_rejects_even_renamed_generator_aliases(monkeypatch,collision):
    plan=small_plan()
    excluded=copy.deepcopy(plan['random_stream_contract']['memory_study_exclusions'])
    if collision=='numpy_root':
        excluded[0]['registry']['synthetic_alias']=next(iter(protocol.registry(plan).values()))
    elif collision=='numpy_child_state':
        child=plan['random_stream_contract']['generators']['planner/particle_filter/0/process_noise']
        excluded[0]['generators']['synthetic_alias']=copy.deepcopy(child)
    else:
        state=next(iter(plan['random_stream_contract']['torch_generators'].values()))
        excluded[0]['torch_generators']['synthetic_alias']=copy.deepcopy(state)
    monkeypatch.setattr(protocol,'memory_stream_exclusions',lambda:excluded)
    with pytest.raises(ValueError,match='Memory-study'):
        protocol.stream_contract(plan)


def test_manifest_construction_cannot_sample_numpy_including_memory_scored_exclusions(monkeypatch):
    default_rng=np.random.default_rng
    class ManifestOnly:
        def __init__(self,seed):
            self.bit_generator=default_rng(seed).bit_generator
        def __getattr__(self,name):
            raise AssertionError('Manifest must not sample '+name)
    monkeypatch.setattr(np.random,'default_rng',ManifestOnly)
    plan=protocol.settings(engineering=True)
    contract=protocol.stream_contract(plan)
    assert contract['draws_for_manifest']==0
    assert len(contract['memory_study_exclusions'])==5
