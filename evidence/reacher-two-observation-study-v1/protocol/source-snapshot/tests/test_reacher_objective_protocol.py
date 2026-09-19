"""Synthetic objective-protocol checks; no scored namespace draws or real fits."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openjev.research import reacher_objective_protocol as protocol
from openjev.research import reacher_search_protocol as old_protocol
from openjev.research.reacher_adaptive_search import search
from openjev.research.reacher_raw_endpoint import RawEndpointAuxiliary
from openjev.research.reacher_world_models import GRUWorldModel


@pytest.fixture(autouse=True)
def preserve_global_state():
    threads, state = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def small_plan():
    plan = protocol.settings(engineering=True)
    plan.update(control_episodes=2, prediction_episodes=3, train_episodes=128,
                epochs=2, hidden_size=4, engineering_rng_namespaces=[])
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    return plan


def model_for(plan):
    with protocol.initialization_rng(plan, "pair0"):
        return GRUWorldModel(hidden_size=plan["hidden_size"])


def public_packets(batch=2):
    result = torch.tensor([[1., 0., 0., 1., .1, -.2, 1., 0.]]).repeat(batch, 1)
    return result


def fixed_schedules(batch=2):
    result = np.ones((batch, 51), dtype=bool)
    result[:, 8:14] = False
    result[:, 28:34] = False
    return result


def test_settings_are_independent_and_have_all_nine_fits_and_57_control_rows():
    plan = protocol.settings()
    assert protocol.validate_settings(plan) is plan
    assert plan["rng_namespace"] == protocol.SCORED_NAMESPACE
    assert plan["engineering_rng_namespaces"] == list(protocol.ENGINEERING_NAMESPACES)
    assert plan["fit_order"] == ["anchor-pair0", "raw-pair0", "latent-pair0",
                                  "raw-pair1", "latent-pair1", "anchor-pair1",
                                  "latent-pair2", "anchor-pair2", "raw-pair2"]
    fits = protocol.fit_manifest(plan)
    assert len(fits) == 9
    for pair in protocol.PAIRS:
        triple = [row for row in fits if row["pair"] == pair]
        assert {row["arm"] for row in triple} == set(protocol.ARMS)
        assert {row["initialization_role"] for row in triple} == {f"fit/student/{pair}"}
        assert {row["minibatch_role"] for row in triple} == {f"fit/minibatch/{pair}"}
        assert {row["predictor_role"] for row in triple if row["arm"] != "anchor"} == {f"fit/predictor/{pair}"}
    rows = protocol.execution_order(plan)
    assert len(rows) == len({row["path"] for row in rows}) == 57
    learned = [row for row in rows if "fit" in row]
    references = [row for row in rows if "reference" in row]
    assert len(learned) == 45 and len(references) == 12
    assert sum(row["reset"] for row in learned) == 18
    assert all(row["planner"] == "cem256" for row in learned)
    assert not any(row["reset"] for row in references)
    assert not any(row["reset"] for row in rows if row["panel"] == "full")
    plan["optimizer"]["betas"][0] = 0
    assert protocol.settings()["optimizer"]["betas"] == [.9, .999]


@pytest.mark.parametrize("key,value", [("planner", "rs256"), ("auxiliary_horizons", [1, 3, 11]),
                                      ("ema_momentum", .8), ("variance_weight", .1), ("residual_reward", False),
                                      ("prediction_episodes", 95), ("control_episodes", 63),
                                      ("rng_namespace", "reacher-search-v1-scored"), ("steps", 49)])
def test_fixed_settings_drift_is_rejected_without_draws(key, value):
    plan = protocol.settings()
    plan[key] = value
    with pytest.raises(ValueError):
        protocol.validate_settings(plan)


def test_nested_optimizer_type_order_and_exclusion_drift_rejected():
    for mutate in (
        lambda p: p["optimizer"].update(foreach=True),
        lambda p: p["optimizer"].update(foreach=0),
        lambda p: p["fit_order"].reverse(),
        lambda p: p["execution_order"].pop(),
        lambda p: p["engineering_rng_namespaces"].pop(),
    ):
        plan = protocol.settings()
        mutate(plan)
        with pytest.raises(ValueError):
            protocol.validate_settings(plan)


def test_engineering_reduction_requires_engineering_namespace():
    plan = small_plan()
    protocol.validate_settings(plan, engineering=True)
    with pytest.raises(ValueError):
        protocol.validate_settings(plan)
    plan["rng_namespace"] = protocol.SCORED_NAMESPACE
    with pytest.raises(ValueError, match="engineering namespace"):
        protocol.validate_settings(plan, engineering=True)
    plan["rng_namespace"] = "reacher-objective-engineering-unregistered-v1"
    with pytest.raises(ValueError, match="engineering namespace"):
        protocol.validate_settings(plan, engineering=True)


def test_full_engineering_stream_contract_has_exact_roles_children_and_no_global_rng_change():
    plan = protocol.settings(engineering=True)
    before = torch.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    with patch.object(torch, "randperm", side_effect=AssertionError("manifest drew samples")):
        contract = protocol.stream_contract(plan)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert len(contract["registry"]) == 892
    assert len(contract["generators"]) == 1020
    assert len(contract["torch_registry"]) == len(contract["torch_generators"]) == 9
    assert contract["draws_for_manifest"] == 0
    assert all(row["draws_for_manifest"] == 0 for row in contract["generators"].values())
    assert all(row["draws_for_manifest"] == 0 for row in contract["torch_generators"].values())
    assert set(contract["generators"]) - set(contract["registry"]) == {
        f"planner/particle_filter/{i}/{child}" for i in range(64)
        for child in ("initial", "process_noise", "resample")}
    assert len(contract["engineering_exclusions"]) == 3
    for previous in contract["engineering_exclusions"]:
        assert len(previous["registry"]) == 892 and len(previous["torch_registry"]) == 9
        assert not set(contract["registry"].values()) & set(previous["registry"].values())
        assert not set(contract["torch_registry"].values()) & set(previous["torch_registry"].values())
    assert json.loads(json.dumps(contract)) == contract


def test_manifest_binds_supplied_prior_registries_and_includes_all_parent_torch_purposes():
    plan = small_plan()
    previous_numpy = {"old/environment/0": 271}
    previous_torch = {"old/student/271": 271, "old/minibatch/271": 4100271}
    contract = protocol.stream_contract(plan, [previous_numpy], [previous_torch], [{"plan_sha256": "synthetic"}])
    assert len(contract["prior_numpy_registry_sha256"]) == len(contract["prior_torch_registry_sha256"]) == 1
    assert contract["priors"] == [{"plan_sha256": "synthetic"}]
    assert contract["unused_random_extra"].startswith("Generated and saved")


def test_prior_numpy_and_torch_seed_aliases_and_engineering_self_overlap_rejected():
    plan = small_plan()
    numpy_seed = next(iter(protocol.registry(plan).values()))
    torch_seed = next(iter(protocol.torch_registry(plan).values()))
    with pytest.raises(ValueError, match="Prior NumPy"):
        protocol.stream_contract(plan, [{"prior/other_name": numpy_seed}])
    for previous in (torch_seed, torch_seed + (1 << 32)):
        with pytest.raises(ValueError, match="Prior Torch"):
            protocol.stream_contract(plan, prior_torch_registries=[{"old/purpose": previous}])
    plan["engineering_rng_namespaces"] = [plan["rng_namespace"]]
    with pytest.raises(ValueError, match="Engineering NumPy"):
        protocol.stream_contract(plan)


def test_torch_high_bit_alias_check_matches_pinned_cpu_generator_behavior():
    plan = small_plan()
    value = protocol.seed(plan, "fit/student/pair0")
    first = torch.Generator().manual_seed(value)
    aliased = torch.Generator().manual_seed(value + (1 << 32))
    # Only engineering streams are sampled here, with no models involved.
    torch.testing.assert_close(torch.randn(32, generator=first), torch.randn(32, generator=aliased),
                               rtol=0, atol=0)


def test_cross_role_numpy_and_torch_collisions_fail_closed(monkeypatch):
    plan = small_plan()
    with monkeypatch.context() as context:
        context.setattr(old_protocol, "named_seed", lambda namespace, role: 19)
        with pytest.raises(ValueError, match="NumPy root"):
            protocol.stream_contract(plan)
    with monkeypatch.context() as context:
        context.setattr(protocol, "seed32", lambda namespace, role: 19)
        with pytest.raises(ValueError, match="Torch root"):
            protocol.stream_contract(plan)


def test_named_generators_pair_student_predictor_and_minibatches_without_global_contamination():
    plan = small_plan()
    before = torch.get_rng_state().clone()
    models, heads = [], []
    for _ in protocol.ARMS:
        with protocol.initialization_rng(plan, "pair0", "student"):
            model = GRUWorldModel(hidden_size=4)
        with protocol.initialization_rng(plan, "pair0", "predictor"):
            head = RawEndpointAuxiliary(model)
        models.append(model.state_dict())
        heads.append(head.state_dict())
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    for states in (models, heads):
        for key, value in states[0].items():
            for other in states[1:]:
                torch.testing.assert_close(other[key], value, rtol=0, atol=0)
    orders = protocol.minibatch_orders(plan, "pair0")
    torch.testing.assert_close(orders, protocol.minibatch_orders(plan, "pair0"), rtol=0, atol=0)
    assert orders.shape == (2, 128) and orders.dtype == torch.long
    for row in orders:
        torch.testing.assert_close(row.sort().values, torch.arange(128))
    assert not torch.equal(orders, protocol.minibatch_orders(plan, "pair1"))
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_initialization_context_restores_global_rng_after_exception_and_rejects_role_tampering():
    plan = small_plan()
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="synthetic"), protocol.initialization_rng(plan, "pair0"):
        torch.rand(3)
        raise RuntimeError("synthetic")
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    plan["random_stream_contract"]["torch_registry"]["fit/student/pair0"] += 1
    with pytest.raises(ValueError, match="role-derived"):
        protocol.torch_generator(plan, "fit/student/pair0")
    with pytest.raises(ValueError, match="Unknown"):
        protocol.torch_generator(plan, "control/reset/0")


def test_schedule_and_search_inputs_match_existing_search_helpers_on_engineering_streams(tmp_path):
    plan = small_plan()
    for split in ("prediction", "control"):
        for index in range(plan[f"{split}_episodes"]):
            for panel in protocol.PANELS:
                np.testing.assert_array_equal(protocol.schedule(plan, index, panel, split),
                                              old_protocol.schedule(plan, index, panel, split))
            ordinary = protocol.schedule(plan, index, "ordinary", split)
            shifted = protocol.schedule(plan, index, "shift", split)
            assert (~ordinary).sum() == 12 and (~shifted).sum() == 20
            assert protocol.reset_steps(ordinary) == protocol.reset_steps(shifted)
            phase = protocol.phase(plan, index, split)
            assert protocol.reset_steps(ordinary) == (7 + phase, 27 + phase)
    inputs = protocol.draw_control_inputs(plan, 49)
    reference = old_protocol.draw_inputs(plan, "planner/control/49", plan["control_episodes"])
    assert inputs.identities() == reference.identities()
    assert inputs.initial.shape == (2, 64, 4, 2)
    assert inputs.random_extra.shape == (2, 192, 4, 2)
    bank = protocol.common_bank(inputs, 49, plan)
    assert bank.shape == (2, 64, 1, 2)
    expected = search("rs64", inputs, lambda sequences: np.zeros(sequences.shape[:2], dtype=np.float32), step=49)
    np.testing.assert_array_equal(bank, expected.sequences)
    stem = tmp_path / "inputs"
    old_protocol.save_inputs(stem, inputs, "planner/control/49")
    assert old_protocol.load_inputs(stem).identities() == inputs.identities()
    with pytest.raises(ValueError, match="terminal"):
        protocol.draw_control_inputs(plan, 50)
    with pytest.raises(ValueError, match="terminal"):
        protocol.common_bank(inputs, 50, plan)


def test_public_reset_erases_history_but_preserves_current_measurement_and_input_state():
    plan = small_plan()
    model = model_for(plan)
    packets, schedules = public_packets(), fixed_schedules()
    state = model.initial(2)
    state["hidden"][0].fill_(3)
    state["hidden"][1].fill_(-4)
    state["packet"][0].fill_(7)
    state["packet"][1].fill_(-8)
    original = copy.deepcopy(state)
    packets_before, schedules_before = packets.clone(), schedules.copy()
    intact, intact_mask = protocol.assimilate_control(model, state, packets, 7, schedules)
    reset, mask = protocol.assimilate_control(model, state, packets, 7, schedules, reset=True)
    assert mask.tolist() == [True, True] and not intact_mask.any()
    torch.testing.assert_close(reset["hidden"][0], reset["hidden"][1], rtol=0, atol=0)
    assert not torch.equal(intact["hidden"][0], intact["hidden"][1])
    for result in (reset, intact):
        torch.testing.assert_close(result["packet"], packets, rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(state[key], original[key], rtol=0, atol=0)
        assert reset[key].data_ptr() != state[key].data_ptr()
    torch.testing.assert_close(packets, packets_before, rtol=0, atol=0)
    np.testing.assert_array_equal(schedules, schedules_before)
    assert protocol.reset_events(7, mask, packets) == [
        {"step": 7, "case": index, "packet": packets[index].tolist()} for index in range(2)]


def test_reset_occurs_before_one_assimilation_never_advances_actions_or_changes_rng():
    plan = small_plan()
    model = model_for(plan)
    state, packets, schedules = model.initial(2), public_packets(), fixed_schedules()
    state["hidden"].fill_(2)
    before = torch.get_rng_state().clone()
    with (patch.object(model, "advance", side_effect=AssertionError("action advanced")),
          patch.object(model, "assimilate", wraps=model.assimilate) as assimilate):
        result, _ = protocol.assimilate_control(model, state, packets, 7, schedules, reset=True)
    assert assimilate.call_count == 1
    initial_argument, packet_argument = assimilate.call_args.args
    assert torch.count_nonzero(initial_argument["hidden"]) == 0
    torch.testing.assert_close(packet_argument, packets, rtol=0, atol=0)
    expected = model.assimilate(model.initial(2), packets)
    torch.testing.assert_close(result["hidden"], expected["hidden"], rtol=0, atol=0)
    assert result["hidden"].abs().sum() > 0
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_case_specific_precursors_trigger_exactly_twice_with_independent_batch_states():
    plan = small_plan()
    model = model_for(plan)
    schedules = fixed_schedules()
    schedules[1] = np.roll(schedules[1], 1)
    seen = [[] for _ in range(2)]
    for step in range(50):
        packets = public_packets()
        packets[:, 6] = torch.from_numpy(schedules[:, step].astype(np.float32))
        packets[packets[:, 6] == 0, :4] = float("nan")
        state = model.initial(2)
        state["hidden"].fill_(3)
        result, mask = protocol.assimilate_control(model, state, packets, step, schedules, reset=True)
        assert torch.isfinite(result["packet"]).all()
        for index, triggered in enumerate(mask.tolist()):
            if triggered:
                seen[index].append(step)
    assert seen == [[7, 27], [8, 28]]


def test_missing_placeholders_discarded_and_full_sensing_never_resets():
    plan = small_plan()
    model = model_for(plan)
    schedules = fixed_schedules()
    state, packets = model.initial(2), public_packets()
    packets[:, 6] = 0
    packets[:, :4] = float("nan")
    observed, mask = protocol.assimilate_control(model, state, packets, 8, schedules, reset=True)
    assert not mask.any() and torch.count_nonzero(observed["packet"][:, :4]) == 0
    schedules[:] = True
    _, mask = protocol.assimilate_control(model, state, public_packets(), 7, schedules, reset=True)
    assert not mask.any() and protocol.reset_steps(schedules[0]) == ()


@pytest.mark.parametrize("bad", ["missing_precursor", "wrong_dtype", "wrong_schedule", "initial_missing",
                               "terminal_step", "visible_nan", "negative_age", "flag_fraction", "bad_state"])
def test_invalid_reset_inputs_fail_closed(bad):
    plan = small_plan()
    model = model_for(plan)
    state, packets, schedules, step = model.initial(2), public_packets(), fixed_schedules(), 7
    if bad == "missing_precursor":
        packets[:, 6] = 0
    elif bad == "wrong_dtype":
        packets = packets.double()
    elif bad == "wrong_schedule":
        schedules = schedules.astype(np.float32)
    elif bad == "initial_missing":
        schedules[:, 0] = False
    elif bad == "terminal_step":
        step = 50
    elif bad == "visible_nan":
        packets[:, 0] = float("nan")
    elif bad == "negative_age":
        packets[:, 7] = -1
    elif bad == "flag_fraction":
        packets[:, 6] = .5
    else:
        state["hidden"] = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        protocol.assimilate_control(model, state, packets, step, schedules, reset=True)


def test_reset_event_packet_is_detached_and_rejects_missing_or_nonfinite_values():
    packets = public_packets().requires_grad_()
    mask = torch.tensor([True, False])
    events = protocol.reset_events(7, mask, packets)
    assert events == [{"step": 7, "case": 0, "packet": packets[0].tolist()}]
    changed = packets.detach().clone()
    changed[0, 6] = 0
    with pytest.raises(ValueError, match="observed"):
        protocol.reset_events(7, mask, changed)
