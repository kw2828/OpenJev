"""Synthetic causal/gradient checks only; no environment, fits or scored inputs."""

import io

import pytest
import torch
from torch import nn

from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_two_observation_history import (
    REAL_KEYS,
    TwoObservationHistoryGRUWorldModel,
    parameter_and_operation_accounting,
)
from openjev.research.reacher_world_models import repeat_index, sequence_loss


@pytest.fixture(autouse=True)
def engineering_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(previous)


def packets(batch=2, points=15, missing=()):
    angle = torch.linspace(-0.3, 0.7, batch * points * 2).reshape(batch, points, 2)
    result = torch.zeros(batch, points, 8)
    result[..., :2], result[..., 2:4] = angle.cos(), angle.sin()
    result[..., 4:6] = torch.tensor([0.1, -0.09])
    result[..., 6] = 1
    last = 0
    for point in range(points):
        if point in missing:
            result[:, point, :4] = 0
            result[:, point, 6] = 0
            result[:, point, 7] = (point - last) * 0.02
        else:
            last = point
    return result


def commands(batch=2, steps=14):
    return torch.linspace(-0.7, 0.8, batch * steps * 2).reshape(batch, steps, 2)


def rollout_real(model, public, issued):
    assert issued.shape[1] == public.shape[1] - 1
    state = model.initial(len(public))
    for point in range(public.shape[1]):
        if point:
            state, _, _ = model.advance(state, issued[:, point - 1])
        state = model.assimilate(state, public[:, point])
    return state


def equal_state(left, right):
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


def test_constructor_parameter_order_initialization_and_rng_are_identical():
    torch.manual_seed(410)
    original = GRUResidualRewardWorldModel()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(410)
    bounded = TwoObservationHistoryGRUWorldModel()
    assert torch.equal(rng, torch.get_rng_state())
    assert bounded.__init__.__func__ is GRUResidualRewardWorldModel.__init__
    assert list(original.state_dict()) == list(bounded.state_dict())
    equal_state(original.state_dict(), bounded.state_dict())
    assert sum(p.numel() for p in bounded.parameters()) == 36805
    assert not list(bounded.named_buffers())


def test_visible_stream_evicts_everything_before_penultimate_measurement():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public, issued = packets(points=6), commands(steps=5)
    state = model.initial(2)
    for point in range(6):
        if point:
            state, _, _ = model.advance(state, issued[:, point - 1])
        state = model.assimilate(state, public[:, point])
        count = min(2, point + 1)
        assert state["real_present"].tolist() == [[False] * (12 - count) + [True] * count] * 2
        assert state["real_indices"].tolist() == [[-1] * (12 - count) + list(range(point - count + 1, point + 1))] * 2
        assert (state["real_index"] == point).all()
        torch.testing.assert_close(state["real_packets"][:, -count:], public[:, point - count + 1:point + 1], rtol=0, atol=0)
        assert not state["real_packets"][:, :-count].any()
        if point:
            torch.testing.assert_close(state["real_actions"][:, -1], issued[:, point - 1], rtol=0, atol=0)
        assert not state["real_actions"][:, :-1].any()
        assert not state["pending_action"].any() and not state["imagined_depth"].any()


def test_ten_missing_packet_boundary_and_thirteen_slot_reacquisition():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public, issued = packets(points=15, missing=range(3, 13)), commands(steps=14)
    state = rollout_real(model, public[:, :13], issued[:, :12])
    assert state["real_indices"].tolist() == [list(range(1, 13))] * 2
    assert state["real_present"].all()
    torch.testing.assert_close(state["real_actions"], issued[:, 1:12], rtol=0, atol=0)
    state, _, _ = model.advance(state, issued[:, 12])
    reacquired = model.assimilate(state, public[:, 13])
    assert reacquired["real_indices"].tolist() == [list(range(2, 14))] * 2
    assert reacquired["real_present"].all()
    assert (reacquired["real_packets"][..., 6].sum(1) == 2).all()
    torch.testing.assert_close(reacquired["real_actions"], issued[:, 2:13], rtol=0, atol=0)
    reacquired, _, _ = model.advance(reacquired, issued[:, 13])
    next_root = model.assimilate(reacquired, public[:, 14])
    assert next_root["real_indices"].tolist() == [[-1] * 10 + [13, 14]] * 2


@pytest.mark.parametrize("startup_only", [False, True])
def test_suffix_overflow_rejected_without_mutating_any_incoming_state(startup_only):
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    missing = range(1, 14) if startup_only else range(2, 14)
    public, issued = packets(points=14, missing=missing), commands(steps=13)
    last = 11
    state = rollout_real(model, public[:, :last + 1], issued[:, :last])
    state, _, _ = model.advance(state, issued[:, last])
    snapshot = {k: v.clone() for k, v in state.items()}
    with pytest.raises(ValueError, match="exceeds twelve"):
        model.assimilate(state, public[:, last + 1])
    equal_state(state, snapshot)


@pytest.mark.parametrize("missing", [(), (10, 11, 12, 13, 14)])
def test_same_allowed_suffix_different_past_gives_exact_same_predictions(missing):
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public, issued = packets(missing=missing), commands()
    earlier = 8 if missing else 13
    changed, alternate = public.clone(), issued.clone()
    changed[:, :earlier, :4] *= -1
    alternate[:, :earlier] *= -1
    left = rollout_real(model, public, issued)
    right = rollout_real(model, changed, alternate)
    equal_state(left, right)
    for _ in range(4):
        left, prediction, reward = model.advance(left, torch.full((2, 2), 0.3))
        right, other_prediction, other_reward = model.advance(right, torch.full((2, 2), 0.3))
        equal_state(left, right)
        torch.testing.assert_close(prediction, other_prediction, rtol=0, atol=0)
        torch.testing.assert_close(reward, other_reward, rtol=0, atol=0)


def test_reconstruction_exactly_matches_original_parent_replay_of_retained_suffix():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    parent = GRUResidualRewardWorldModel(hidden_size=4)
    parent.load_state_dict(model.state_dict())
    public, issued = packets(missing=range(10, 15)), commands()
    actual = rollout_real(model, public, issued)
    reference = parent.initial(2)
    for point in range(8, 15):
        reference = parent.assimilate(reference, public[:, point])
        if point < 14:
            reference, _, _ = parent.advance(reference, issued[:, point])
    equal_state({key: actual[key] for key in ("hidden", "packet")}, reference)


def test_discarded_hidden_and_prediction_poison_cannot_leak_at_real_boundary():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public = packets(points=2)
    root = model.assimilate(model.initial(2), public[:, 0])
    selected, _, _ = model.advance(root, torch.full((2, 2), 0.2))
    poisoned = {key: value.clone() for key, value in selected.items()}
    poisoned["hidden"].fill_(float("nan"))
    poisoned["packet"].fill_(float("inf"))
    equal_state(model.assimilate(selected, public[:, 1]), model.assimilate(poisoned, public[:, 1]))


def test_missing_poison_sanitized_but_known_fields_and_clock_rejected():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public = packets(points=2, missing=(1,))
    root = model.assimilate(model.initial(2), public[:, 0])
    selected, _, _ = model.advance(root, torch.zeros(2, 2))
    poison = public[:, 1].clone()
    poison[:, :4] = torch.tensor([float("nan"), float("inf"), -1e6, 1e6])
    equal_state(model.assimilate(selected, poison), model.assimilate(selected, public[:, 1]))
    poison[:, 4] = float("nan")
    with pytest.raises(ValueError, match="known public"):
        model.assimilate(selected, poison)


def test_age_tolerance_accepts_float_rounding_but_preserves_actual_packet():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public = packets(points=2, missing=(1,))
    root = model.assimilate(model.initial(2), public[:, 0])
    selected, _, _ = model.advance(root, torch.zeros(2, 2))
    public[:, 1, 7] += 1e-7
    actual = model.assimilate(selected, public[:, 1])
    torch.testing.assert_close(actual["packet"], public[:, 1], rtol=0, atol=0)
    model.advance(actual, torch.zeros(2, 2))


@pytest.mark.parametrize("change", ["first_missing", "target", "age", "visible_age", "validity"])
def test_public_packet_contract_rejects_invalid_boundaries(change):
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public = packets(points=2)
    state = model.initial(2)
    incoming = public[:, 0].clone()
    if change == "first_missing":
        incoming[:, 6] = 0
    else:
        state = model.assimilate(state, incoming)
        state, _, _ = model.advance(state, torch.zeros(2, 2))
        incoming = public[:, 1].clone()
        if change == "target":
            incoming[:, 4] += 0.01
        elif change == "age":
            incoming[:, 6] = 0
            incoming[:, 7] = 0.021
        elif change == "visible_age":
            incoming[:, 7] = 0.02
        else:
            incoming[:, 6] = 0.5
    with pytest.raises(ValueError):
        model.assimilate(state, incoming)


def test_private_rollout_phase_issued_action_alignment_and_candidate_independence():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public, issued = packets(points=4), commands(steps=3)
    initial = model.initial(2)
    with pytest.raises(ValueError, match="Initial real packet"):
        model.advance(initial, torch.zeros(2, 2))
    root = rollout_real(model, public[:, :3], issued[:, :2])
    snapshot = {key: value.clone() for key, value in root.items()}
    repeated = repeat_index(root, torch.tensor([0, 0, 1]))
    actions = torch.tensor([[0.7, -0.2], [-0.7, 0.2], [0.1, 0.3]])
    advanced, _, _ = model.advance(repeated, actions)
    equal_state(root, snapshot)
    for key in REAL_KEYS:
        torch.testing.assert_close(advanced[key], repeated[key], rtol=0, atol=0)
    for row, origin in enumerate((0, 0, 1)):
        independent, _, _ = model.advance({k: v[origin:origin + 1] for k, v in root.items()}, actions[row:row + 1])
        for key in root:
            torch.testing.assert_close(advanced[key][row:row + 1], independent[key], rtol=1e-6, atol=1e-7)
    imagined, _, _ = model.advance(advanced, actions)
    with pytest.raises(ValueError, match="exactly one"):
        model.assimilate(imagined, public[[0, 0, 1], 3])
    with pytest.raises(ValueError, match="exactly one"):
        model.assimilate(root, public[:, 3])
    selected, _, _ = model.advance(root, issued[:, 2])
    fresh = model.assimilate(selected, public[:, 3])
    torch.testing.assert_close(fresh["real_actions"][:, -1], issued[:, 2], rtol=0, atol=0)
    advanced["real_packets"].fill_(99)
    advanced["real_actions"].fill_(99)
    equal_state(root, snapshot)


def test_replay_gradients_only_depend_on_retained_actual_suffix():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public = packets(missing=range(10, 15)).requires_grad_()
    issued = commands().requires_grad_()
    state = rollout_real(model, public, issued)
    _, prediction, reward = model.advance(state, torch.zeros(2, 2))
    (prediction.square().sum() + reward.square().sum()).backward()
    assert not public.grad[:, :8].any()
    assert not issued.grad[:, :8].any()
    assert public.grad[:, 8:10, :4].abs().sum() > 0
    assert not public.grad[:, 10:, :4].any()  # Masked observations never enter replay.
    assert issued.grad[:, 8:].abs().sum() > 0
    for name, value in model.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
    assert model.observation_update.weight_ih.grad.abs().sum() > 0
    assert model.transition.weight_ih.grad.abs().sum() > 0


def test_sequence_loss_flattens_integer_history_and_keeps_gradients_through_fifty_steps():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public = packets(points=51, missing=tuple(range(8, 14)) + tuple(range(28, 38)))
    issued = commands(steps=50)
    # Different masks across the batch also exercise per-row anchor and clocks.
    public[1] = packets(batch=1, points=51, missing=tuple(range(9, 15)) + tuple(range(29, 39)))[0]
    rewards = -issued.square().sum(-1) - 0.12
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loss, metrics = sequence_loss(model, public, issued, rewards, rollout_horizon=5)
    loss.backward()
    assert torch.isfinite(loss) and metrics["kl_nats"] == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    equal_state(model.state_dict(), before)
    poisoned = public.clone()
    poisoned[..., :4][public[..., 6] == 0] = 1e6
    again, repeated = sequence_loss(model, poisoned, issued, rewards, rollout_horizon=5)
    torch.testing.assert_close(loss, again, rtol=0, atol=0)
    assert metrics == repeated


def test_episode_reset_and_terminal_observation_fifty():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    public, issued = packets(points=51), commands(steps=50)
    state = rollout_real(model, public, issued)
    assert (state["real_index"] == 50).all()
    with pytest.raises(ValueError, match="beyond terminal50"):
        model.advance(state, torch.zeros(2, 2))
    initial = model.initial(2)
    assert (initial["real_index"] == -1).all() and (initial["real_indices"] == -1).all()
    assert not initial["real_present"].any()
    fresh = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    fresh.load_state_dict(model.state_dict())
    equal_state(model.assimilate(initial, public[:, 0]), fresh.assimilate(fresh.initial(2), public[:, 0]))


def test_late_root_private_rollout_cannot_pass_terminal_fifty():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    state = rollout_real(model, packets(points=49), commands(steps=48))
    for _ in range(2):
        state, _, _ = model.advance(state, torch.zeros(2, 2))
    assert (state["real_index"] == 48).all() and (state["imagined_depth"] == 2).all()
    with pytest.raises(ValueError, match="beyond terminal50"):
        model.advance(state, torch.zeros(2, 2))


def test_full_masked_neural_work_is_counted_by_actual_forward_hooks():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    observed = {"gru": 0, "linear": 0}
    handles = []
    def hook(kind):
        def count(_module, args, _out):
            observed[kind] += len(args[0])
        return count
    for module in model.modules():
        if isinstance(module, nn.GRUCell):
            handles.append(module.register_forward_hook(hook("gru")))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook("linear")))
    try:
        root = model.assimilate(model.initial(2), packets()[:, 0])
        model.advance(root, torch.zeros(2, 2))
    finally:
        for handle in handles:
            handle.remove()
    counts = parameter_and_operation_accounting(model, assimilate_samples=2, advance_samples=2)
    assert counts["replayed_observation_update_samples"] == 24
    assert counts["replayed_transition_samples"] == 22
    assert counts["gru_cell_sample_calls"] == observed["gru"] == 48
    assert counts["linear_layer_sample_calls"] == observed["linear"] == 96
    assert counts["analytic_reward_sample_calls"] == 24
    assert counts["dense_affine_macs"] == 24 * 3 * 4 * 12 + 24 * (5 * 16 + 25 * 4)
    assert counts["startup_masked_work_is_counted"] and not counts["compute_matched"]


def test_actual_safe_deployment_restore_binds_semantics_and_preserves_rng():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    handle = io.BytesIO()
    torch.save(model.deployment_checkpoint(), handle)
    handle.seek(0)
    payload = torch.load(handle, weights_only=True)
    before = torch.get_rng_state().clone()
    restored = TwoObservationHistoryGRUWorldModel.from_deployment_checkpoint(payload)
    assert torch.equal(before, torch.get_rng_state())
    assert type(restored) is TwoObservationHistoryGRUWorldModel
    assert restored.configuration() == model.configuration()
    equal_state(restored.state_dict(), model.state_dict())
    public, issued = packets(missing=range(10, 15)), commands()
    equal_state(rollout_real(restored, public, issued), rollout_real(model, public, issued))
    assert set(payload) == {"weights", "configuration"}


def test_nondefault_constructor_configuration_roundtrips_without_reinterpreting_parameters():
    model = TwoObservationHistoryGRUWorldModel(hidden_size=5, dt=0.04, noise_std=0.1, residual_reward=False)
    restored = TwoObservationHistoryGRUWorldModel.from_deployment_checkpoint(model.deployment_checkpoint())
    assert restored.configuration() == model.configuration()
    equal_state(restored.state_dict(), model.state_dict())
    counts = parameter_and_operation_accounting(restored, assimilate_samples=1, advance_samples=1)
    assert counts["analytic_reward_sample_calls"] == 0
    assert counts["replayed_transition_samples"] == 11


@pytest.mark.parametrize("change", ["class", "window", "phase", "extra", "shape", "nan", "dtype", "hidden_bool", "dt_nan", "noise_negative", "reward_integer"])
def test_deployment_rejects_wrong_semantics_scalars_or_weight_schema(change):
    payload = TwoObservationHistoryGRUWorldModel(hidden_size=4).deployment_checkpoint()
    key = next(iter(payload["weights"]))
    if change == "class":
        payload["configuration"]["model_class"] = "GRUResidualRewardWorldModel"
    elif change == "window":
        payload["configuration"]["max_real_packets"] = 11
    elif change == "phase":
        payload["configuration"]["assimilation"] = "reuse old hidden"
    elif change == "extra":
        payload["weights"]["extra"] = torch.ones(1)
    elif change == "shape":
        payload["weights"][key] = torch.zeros(1)
    elif change == "nan":
        payload["weights"][key].fill_(float("nan"))
    elif change == "dtype":
        payload["weights"][key] = payload["weights"][key].double()
    elif change == "hidden_bool":
        payload["configuration"]["hidden_size"] = True
    elif change == "dt_nan":
        payload["configuration"]["dt"] = float("nan")
    elif change == "noise_negative":
        payload["configuration"]["noise_std"] = -0.1
    else:
        payload["configuration"]["residual_reward"] = 1
    with pytest.raises(ValueError):
        TwoObservationHistoryGRUWorldModel.from_deployment_checkpoint(payload)


@pytest.mark.parametrize("change", ["indices", "mask", "missing_poison", "age", "target", "padding", "action_padding", "float_index", "extra", "pending", "three_valid"])
def test_state_schema_rejects_corrupted_history_before_advancing(change):
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    root = rollout_real(model, packets(points=5, missing=(3, 4)), commands(steps=4))
    if change == "indices":
        root["real_indices"][:, -2] -= 1
    elif change == "mask":
        root["real_present"][:, -2] = False
    elif change == "missing_poison":
        root["real_packets"][:, -1, 0] = 999
    elif change == "age":
        root["real_packets"][:, -1, 7] += 0.02
    elif change == "target":
        root["real_target"] += 0.1
    elif change == "padding":
        root["real_packets"][:, 0, 0] = 1
    elif change == "action_padding":
        root["real_actions"][:, 0, 0] = 1
    elif change == "float_index":
        root["real_index"] = root["real_index"].float()
    elif change == "extra":
        root["extra"] = torch.zeros(2)
    elif change == "pending":
        root["pending_action"] += 0.1
    else:
        root["real_packets"][:, -2, 6] = 1
    with pytest.raises(ValueError):
        model.advance(root, torch.zeros(2, 2))


@pytest.mark.parametrize("bad", [float("nan"), 1.01, float("inf")])
def test_invalid_issued_commands_fail_without_mutation(bad):
    model = TwoObservationHistoryGRUWorldModel(hidden_size=4)
    root = model.assimilate(model.initial(2), packets()[:, 0])
    snapshot = {key: value.clone() for key, value in root.items()}
    with pytest.raises(ValueError):
        model.advance(root, torch.full((2, 2), bad))
    equal_state(root, snapshot)
