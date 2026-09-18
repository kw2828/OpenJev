"""Synthetic interface proofs only, using already declared engineering seed410."""

import copy

import pytest
import torch
from torch import nn

from openjev.research.reacher_observation_baseline import (
    CurrentObservationGRUWorldModel,
    FeedForwardObservationWorldModel,
    operation_counts,
    parameter_accounting,
)
from openjev.research.reacher_reward_residual import (
    GRUResidualRewardWorldModel,
    expected_clipped_action_cost,
)
from openjev.research.reacher_world_models import repeat_index, sequence_loss


@pytest.fixture(autouse=True)
def isolated_engineering_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(threads)


def make(kind):
    if kind == "gru":
        return CurrentObservationGRUWorldModel(hidden_size=4)
    return FeedForwardObservationWorldModel(width=4)


def packet(batch=2, *, missing=False):
    result = torch.zeros(batch, 8)
    result[:, :2] = 1
    result[:, 4:6] = torch.tensor([0.12, -0.08])
    result[:, 6] = 0 if missing else 1
    result[:, 7] = 0.12 if missing else 0
    return result


def assert_state(left, right):
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)


def test_gru_constructor_parameter_order_values_and_rng_are_identical():
    torch.manual_seed(410)
    reference = GRUResidualRewardWorldModel(hidden_size=4)
    after = torch.get_rng_state().clone()
    torch.manual_seed(410)
    baseline = CurrentObservationGRUWorldModel(hidden_size=4)
    assert torch.equal(after, torch.get_rng_state())
    assert [name for name, _ in reference.named_parameters()] == [
        name for name, _ in baseline.named_parameters()
    ]
    assert list(reference.state_dict()) == list(baseline.state_dict())
    assert_state(reference.state_dict(), baseline.state_dict())
    assert CurrentObservationGRUWorldModel.__init__ is GRUResidualRewardWorldModel.__init__
    assert CurrentObservationGRUWorldModel.advance is GRUResidualRewardWorldModel.advance
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in baseline.parameters())


@pytest.mark.parametrize("missing", [False, True])
def test_gru_exactly_matches_parent_with_explicit_reset_before_each_packet(missing):
    baseline = make("gru")
    reference = GRUResidualRewardWorldModel(hidden_size=4)
    reference.load_state_dict(baseline.state_dict())
    current = packet(missing=missing)
    action = torch.tensor([[0.2, -0.3], [-0.1, 0.4]])
    state = baseline.initial(2)
    for _ in range(3):
        state["hidden"].fill_(123)
        state["packet"].fill_(-456)
        actual = baseline.assimilate(state, current)
        expected = reference.assimilate(reference.initial(2), current)
        assert_state(actual, expected)
        for _ in range(3):
            actual, angles, reward = baseline.advance(actual, action)
            expected, expected_angles, expected_reward = reference.advance(expected, action)
            assert_state(actual, expected)
            torch.testing.assert_close(angles, expected_angles, atol=0, rtol=0)
            torch.testing.assert_close(reward, expected_reward, atol=0, rtol=0)
        state = actual


@pytest.mark.parametrize("kind", ["gru", "mlp"])
@pytest.mark.parametrize("missing", [False, True])
def test_assimilation_erases_all_old_values_and_preserves_current_public_fields(kind, missing):
    model = make(kind)
    old = model.initial(2)
    for value in old.values():
        value.fill_(float("nan"))
        value.requires_grad_(True)
    incoming = packet(missing=missing)
    if missing:
        incoming[:, :4] = torch.tensor([float("nan"), float("inf"), -999.0, 999.0])
    incoming.requires_grad_(True)
    state = model.assimilate(old, incoming)
    expected = model.assimilate(model.initial(2), packet(missing=missing))
    assert_state(state, expected)
    torch.testing.assert_close(state["packet"][:, 4:], incoming[:, 4:], rtol=0, atol=0)
    if missing:
        assert not state["packet"][:, :4].any()
        if kind == "gru":
            assert not state["hidden"].any()
    next_state, angles, reward = model.advance(state, torch.zeros(2, 2))
    (angles.sum() + reward.sum()).backward()
    assert all(value.grad is None for value in old.values())
    assert incoming.grad is not None and torch.isfinite(incoming.grad).all()
    assert torch.isfinite(next_state["packet"]).all()


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_imagination_can_retain_action_effects_but_next_real_packet_overwrites_them(kind):
    model = make(kind)
    base = model.assimilate(model.initial(2), packet())
    left, _, _ = model.advance(base, torch.full((2, 2), 0.7))
    right, _, _ = model.advance(base, torch.full((2, 2), -0.7))
    assert any(not torch.equal(left[key], right[key]) for key in left)
    after_left, _, _ = model.advance(left, torch.zeros(2, 2))
    after_right, _, _ = model.advance(right, torch.zeros(2, 2))
    assert any(not torch.equal(after_left[key], after_right[key]) for key in left)
    incoming = packet(missing=True)
    assert_state(model.assimilate(after_left, incoming), model.assimilate(after_right, incoming))


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_candidate_branch_repetition_and_advances_do_not_mutate_roots_or_other_branches(kind):
    model = make(kind)
    root = model.assimilate(model.initial(2), packet())
    saved = {name: value.clone() for name, value in root.items()}
    expanded = repeat_index(root, torch.tensor([0, 0, 1]))
    actions = torch.tensor([[0.2, 0.4], [-0.4, -0.2], [0.2, 0.4]])
    branched, _, _ = model.advance(expanded, actions)
    assert_state(root, saved)
    for index, source in enumerate((0, 0, 1)):
        direct, _, _ = model.advance(
            {key: value[source : source + 1] for key, value in root.items()}, actions[index : index + 1]
        )
        for key in root:
            torch.testing.assert_close(branched[key][index : index + 1], direct[key], rtol=1e-6, atol=1e-7)
    assert not branched["packet"][:, 6].any()
    torch.testing.assert_close(branched["packet"][:, 4:6], expanded["packet"][:, 4:6], rtol=0, atol=0)
    torch.testing.assert_close(branched["packet"][:, 7], expanded["packet"][:, 7] + 0.02, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_unchanged_sequence_loss_matches_direct_one_step_targets_and_masks(kind):
    model = make(kind)
    packets = packet(12).reshape(2, 6, 8)
    packets[:, 2:4, 6] = 0
    packets[:, 2:4, 7] = torch.tensor([0.02, 0.04])
    commands = torch.linspace(-0.8, 0.8, 20).reshape(2, 5, 2)
    rewards = -commands.square().sum(-1) - 0.1
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loss, metrics = sequence_loss(model, packets, commands, rewards, rollout_horizon=3)
    loss.backward()
    assert torch.isfinite(loss) and metrics["kl_nats"] == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert_state(before, model.state_dict())
    changed = packets.clone()
    changed[:, 2:4, :4] = 1e6
    repeated, repeated_metrics = sequence_loss(model, changed, commands, rewards, rollout_horizon=3)
    # Frozen sequence_loss masks targets and both comparators discard missing inputs.
    torch.testing.assert_close(loss, repeated, rtol=0, atol=0)
    assert metrics == repeated_metrics
    predictions, predicted_rewards = [], []
    for step in range(5):
        state = model.assimilate(model.initial(2), packets[:, step])
        _, angles, reward = model.advance(state, commands[:, step])
        predictions.append(angles)
        predicted_rewards.append(reward)
    direct = torch.stack(predictions, 1)
    mask = packets[:, 1:, 6:7]
    mse = ((direct - packets[:, 1:, :4]).square() * mask).sum() / (mask.sum() * 4)
    torch.testing.assert_close(torch.tensor(metrics["observation_mse"]), mse, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.tensor(metrics["reward_mse"]),
        (torch.stack(predicted_rewards, 1) - rewards).square().mean(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_gru_training_loss_and_gradients_equal_an_explicitly_reset_parent():
    model = make("gru")
    reference = GRUResidualRewardWorldModel(hidden_size=4)
    reference.load_state_dict(model.state_dict())
    parent_assimilate = reference.assimilate
    reference.assimilate = lambda state, value: parent_assimilate(reference.initial(len(value)), value)
    packets = packet(10).reshape(2, 5, 8)
    packets[:, 1:3, 6] = 0
    commands = torch.full((2, 4, 2), 0.2)
    rewards = torch.full((2, 4), -0.25)
    losses = []
    for item in (model, reference):
        loss, _ = sequence_loss(item, packets, commands, rewards, rollout_horizon=3)
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0
        )


def test_mlp_reward_skip_changes_only_total_reward_and_uses_no_randomness():
    enabled = FeedForwardObservationWorldModel(width=4)
    disabled = FeedForwardObservationWorldModel(width=4, residual_reward=False)
    disabled.load_state_dict(enabled.state_dict())
    action = torch.tensor([[0.3, -0.1], [1.0, -1.0]])
    rng = torch.get_rng_state().clone()
    state = enabled.assimilate(enabled.initial(2), packet())
    one, angles, reward = enabled.advance(state, action)
    other, original_angles, original_reward = disabled.advance(state, action)
    assert_state(one, other)
    torch.testing.assert_close(angles, original_angles, rtol=0, atol=0)
    torch.testing.assert_close(
        reward, original_reward - expected_clipped_action_cost(action, 0.05), rtol=0, atol=0
    )
    assert torch.equal(rng, torch.get_rng_state())


def test_default_capacity_and_operation_estimates_match_actual_layer_shapes():
    for model in (CurrentObservationGRUWorldModel(), FeedForwardObservationWorldModel()):
        accounting = parameter_accounting(model)
        assert accounting["trainable_parameters"] == sum(p.numel() for p in model.parameters())
        assert accounting["trainable_parameters"] == (
            36805 if isinstance(model, CurrentObservationGRUWorldModel) else 36599
        )
        measured = []
        handles = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(
                    module.register_forward_hook(
                        lambda mod, inputs, result, measured=measured: measured.append(
                            inputs[0].shape[0] * mod.in_features * mod.out_features
                        )
                    )
                )
            elif isinstance(module, nn.GRUCell):
                handles.append(
                    module.register_forward_hook(
                        lambda mod, inputs, result, measured=measured: measured.append(
                            inputs[0].shape[0] * 3 * mod.hidden_size * (mod.input_size + mod.hidden_size)
                        )
                    )
                )
        model.advance(model.assimilate(model.initial(2), packet(missing=True)), torch.zeros(2, 2))
        for handle in handles:
            handle.remove()
        counts = operation_counts(model, assimilate_samples=2, advance_samples=2)
        assert counts["dense_affine_macs"] == sum(measured)
        assert counts["real_state_reset_samples"] == 2 and counts["analytic_reward_sample_calls"] == 2
        assert counts["counts_are_not_total_flops_or_measured_wall_time"]


@pytest.mark.parametrize("kind", ["gru", "mlp"])
@pytest.mark.parametrize("bad", ["shape", "dtype", "validity", "known_nan", "age", "extra_state"])
def test_invalid_interfaces_and_known_field_poison_rejected(kind, bad):
    model = make(kind)
    value = packet()
    state = model.initial(2)
    if bad == "shape":
        value = value[:, :7]
    elif bad == "dtype":
        value = value.double()
    elif bad == "validity":
        value[:, 6] = 0.5
    elif bad == "known_nan":
        value[:, 4] = float("nan")
    elif bad == "age":
        value[:, 7] = -0.1
    else:
        state["velocity"] = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        model.assimilate(state, value)


@pytest.mark.parametrize("bad", [0, True, -1, 1.5])
def test_invalid_mlp_width_rejected(bad):
    with pytest.raises(ValueError):
        FeedForwardObservationWorldModel(width=bad)


def test_actions_and_accounting_counts_reject_invalid_values():
    model = FeedForwardObservationWorldModel(width=4)
    state = model.assimilate(model.initial(2), packet())
    for value in (torch.zeros(2, 3), torch.full((2, 2), float("nan")), torch.full((2, 2), 1.1)):
        with pytest.raises(ValueError):
            model.advance(state, value)
    for value in (-1, True, 0.5):
        with pytest.raises(ValueError):
            operation_counts(model, assimilate_samples=value)


def test_state_and_packet_objects_are_not_mutated():
    for kind in ("gru", "mlp"):
        model = make(kind)
        old = model.initial(2)
        incoming = packet()
        saved_state, saved_packet = copy.deepcopy(old), incoming.clone()
        fresh = model.assimilate(old, incoming)
        fresh["packet"].fill_(17)
        assert_state(old, saved_state)
        torch.testing.assert_close(incoming, saved_packet, rtol=0, atol=0)
