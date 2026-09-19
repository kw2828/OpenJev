"""Synthetic bounded-history proofs, no corpus, fits or scored RNG streams."""

import io

import pytest
import torch
from torch import nn

from openjev.research.reacher_bounded_history import (
    BoundedThreePacketGRUWorldModel,
    parameter_and_operation_accounting,
)
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import repeat_index, sequence_loss


@pytest.fixture(autouse=True)
def engineering_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(previous)


def packets(batch=2, points=7):
    angle = torch.linspace(-0.3, 0.7, batch * points * 2).reshape(batch, points, 2)
    result = torch.zeros(batch, points, 8)
    result[..., :2], result[..., 2:4] = angle.cos(), angle.sin()
    result[..., 4:6] = torch.tensor([0.1, -0.09])
    result[..., 6] = 1
    return result


def commands(batch=2, steps=6):
    return torch.linspace(-0.7, 0.8, batch * steps * 2).reshape(batch, steps, 2)


def rollout_real(model, public, issued):
    state = model.initial(len(public))
    for point in range(public.shape[1]):
        state = model.assimilate(state, public[:, point])
        if point < issued.shape[1]:
            state, _, _ = model.advance(state, issued[:, point])
    return state


def equal_state(one, other):
    assert set(one) == set(other)
    for name in one:
        torch.testing.assert_close(one[name], other[name], rtol=0, atol=0)


def test_constructor_parameters_order_and_rng_exactly_match_original_gru():
    torch.manual_seed(410)
    original = GRUResidualRewardWorldModel(hidden_size=4)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(410)
    bounded = BoundedThreePacketGRUWorldModel(hidden_size=4)
    assert torch.equal(rng, torch.get_rng_state())
    assert bounded.__init__.__func__ is GRUResidualRewardWorldModel.__init__
    assert list(original.state_dict()) == list(bounded.state_dict())
    equal_state(original.state_dict(), bounded.state_dict())


@pytest.mark.parametrize("missing", [False, True])
def test_identical_final_three_real_packets_and_two_commands_erase_arbitrary_older_past(missing):
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    public, issued = packets(), commands()
    changed, alternate = public.clone(), issued.clone()
    changed[:, :-3, :6] += 10
    alternate[:, :-2] *= -1
    if missing:
        public[:, -2:, 6] = changed[:, -2:, 6] = 0
        public[:, -2:, 7] = changed[:, -2:, 7] = torch.tensor([0.02, 0.04])
        public[:, -2:, :4] = 123
        changed[:, -2:, :4] = -456
    left = rollout_real(model, public, issued)
    right = rollout_real(model, changed, alternate)
    equal_state(left, right)
    action = torch.full((2, 2), 0.3)
    for _ in range(4):
        left, observation, reward = model.advance(left, action)
        right, other_observation, other_reward = model.advance(right, action)
        equal_state(left, right)
        torch.testing.assert_close(observation, other_observation, rtol=0, atol=0)
        torch.testing.assert_close(reward, other_reward, rtol=0, atol=0)


def test_current_reconstruction_matches_parent_replay_of_only_the_final_window():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    reference = GRUResidualRewardWorldModel(hidden_size=4)
    reference.load_state_dict(model.state_dict())
    public, issued = packets(), commands()
    public[:, -2, 6] = 0
    public[:, -2, 7] = 0.02
    actual = rollout_real(model, public, issued)
    expected = reference.initial(2)
    for point in range(3):
        expected = reference.assimilate(expected, public[:, -3 + point])
        if point < 2:
            expected, _, _ = reference.advance(expected, issued[:, -2 + point])
    equal_state({key: actual[key] for key in ("hidden", "packet")}, expected)


def test_startup_masks_real_action_alignment_and_oldest_edge_eviction():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    public, issued = packets(points=5), commands(steps=4)
    state = model.initial(2)
    for point in range(5):
        if point:
            state, _, _ = model.advance(state, issued[:, point - 1])
        state = model.assimilate(state, public[:, point])
        count = min(3, point + 1)
        assert state["real_valid"].tolist() == [[False] * (3 - count) + [True] * count] * 2
        torch.testing.assert_close(
            state["real_packets"][:, -count:], public[:, point - count + 1 : point + 1], rtol=0, atol=0
        )
        assert not state["real_packets"][:, : 3 - count].any()
        edges = min(2, point)
        if edges:
            torch.testing.assert_close(
                state["real_actions"][:, -edges:], issued[:, point - edges : point], rtol=0, atol=0
            )
        assert not state["real_actions"][:, : 2 - edges].any()
        assert not state["pending_action"].any() and not state["imagined_depth"].any()


def test_no_imagined_packet_enters_history_and_multistep_terminal_cannot_be_real_evidence():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    public = packets(points=2)
    root = model.assimilate(model.initial(2), public[:, 0])
    imagined = root
    action = torch.tensor([[0.2, -0.3], [0.3, -0.2]])
    for depth in range(1, 5):
        imagined, _, _ = model.advance(imagined, action)
        for name in ("real_packets", "real_actions", "real_valid"):
            torch.testing.assert_close(imagined[name], root[name], rtol=0, atol=0)
        assert (imagined["imagined_depth"] == depth).all()
        assert not imagined["packet"][:, 6].any()
    with pytest.raises(ValueError, match="exactly one"):
        model.assimilate(imagined, public[:, 1])
    selected, _, _ = model.advance(root, action)
    # Old latent and predicted-packet values cannot become an evidence channel.
    selected["hidden"].fill_(float("nan"))
    selected["packet"].fill_(float("nan"))
    actual = model.assimilate(selected, public[:, 1])
    assert torch.isfinite(actual["hidden"]).all()
    torch.testing.assert_close(actual["real_actions"][:, -1], action, rtol=0, atol=0)


def test_duplicate_assimilation_and_action_before_first_packet_rejected():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    initial = model.initial(2)
    with pytest.raises(ValueError, match="initial real packet"):
        model.advance(initial, torch.zeros(2, 2))
    root = model.assimilate(initial, packets()[:, 0])
    with pytest.raises(ValueError, match="exactly one"):
        model.assimilate(root, packets()[:, 1])


def test_missing_angular_poison_is_discarded_but_known_field_poison_is_rejected():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    incoming = packets()[:, 0]
    incoming[:, 6] = 0
    incoming[:, 7] = 0.1
    clean = incoming.clone()
    incoming[:, :4] = torch.tensor([float("nan"), float("inf"), -123.0, 456.0])
    equal_state(model.assimilate(model.initial(2), incoming), model.assimilate(model.initial(2), clean))
    incoming[:, 4] = float("nan")
    with pytest.raises(ValueError, match="known public"):
        model.assimilate(model.initial(2), incoming)


def test_candidate_branches_are_independent_and_never_mutate_real_roots():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    root = rollout_real(model, packets(), commands())
    saved = {key: value.clone() for key, value in root.items()}
    repeated = repeat_index(root, torch.tensor([0, 0, 1]))
    actions = torch.tensor([[0.7, -0.2], [-0.7, 0.2], [0.1, 0.3]])
    advanced, _, _ = model.advance(repeated, actions)
    equal_state(root, saved)
    for row, origin in enumerate((0, 0, 1)):
        independent, _, _ = model.advance(
            {k: v[origin : origin + 1] for k, v in root.items()}, actions[row : row + 1]
        )
        for key in root:
            torch.testing.assert_close(advanced[key][row : row + 1], independent[key], rtol=1e-6, atol=1e-7)
    advanced["real_packets"].fill_(99)
    advanced["real_actions"].fill_(99)
    equal_state(root, saved)
    for key in ("real_packets", "real_actions", "real_valid"):
        torch.testing.assert_close(
            repeated[key], repeat_index(root, torch.tensor([0, 0, 1]))[key], rtol=0, atol=0
        )


def test_gradients_depend_only_on_three_real_packets_and_two_intervening_commands():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    public, issued = packets().requires_grad_(), commands().requires_grad_()
    state = rollout_real(model, public, issued)
    _, observation, reward = model.advance(state, torch.zeros(2, 2))
    (observation.square().sum() + reward.square().sum()).backward()
    assert not public.grad[:, :-3].any()
    assert not issued.grad[:, :-2].any()
    assert public.grad[:, -3:].abs().sum() > 0
    assert issued.grad[:, -2:].abs().sum() > 0
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())


def test_episode_reset_removes_history_and_has_no_global_mutable_state():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    rollout_real(model, packets(), commands())
    reset = model.initial(2)
    assert not any(bool(value.any()) for value in reset.values())
    fresh = BoundedThreePacketGRUWorldModel(hidden_size=4)
    fresh.load_state_dict(model.state_dict())
    equal_state(model.assimilate(reset, packets()[:, 0]), fresh.assimilate(fresh.initial(2), packets()[:, 0]))


def test_sequence_loss_public_targets_masks_backward_and_no_weight_mutation():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    public, issued = packets(), commands()
    public[:, 2:4, 6] = 0
    public[:, 2:4, 7] = torch.tensor([0.02, 0.04])
    rewards = -issued.square().sum(-1) - 0.12
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loss, metrics = sequence_loss(model, public, issued, rewards, rollout_horizon=5)
    loss.backward()
    assert torch.isfinite(loss) and metrics["kl_nats"] == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    equal_state(model.state_dict(), before)
    poison = public.clone()
    poison[:, 2:4, :4] = 1e6
    again, repeated = sequence_loss(model, poison, issued, rewards, rollout_horizon=5)
    torch.testing.assert_close(loss, again, rtol=0, atol=0)
    assert metrics == repeated


def test_actual_safe_deployment_restore_preserves_semantics_and_rng():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    payload = model.deployment_checkpoint()
    handle = io.BytesIO()
    torch.save(payload, handle)
    handle.seek(0)
    loaded = torch.load(handle, weights_only=True)
    before = torch.get_rng_state().clone()
    restored = BoundedThreePacketGRUWorldModel.from_deployment_checkpoint(loaded)
    assert torch.equal(before, torch.get_rng_state())
    assert restored.configuration() == model.configuration()
    equal_state(restored.state_dict(), model.state_dict())
    equal_state(rollout_real(restored, packets(), commands()), rollout_real(model, packets(), commands()))
    assert set(payload) == {"weights", "configuration"}  # No episode history or inferred fit provenance.


@pytest.mark.parametrize("change", ["class", "window", "phase", "extra", "shape", "nan", "dtype"])
def test_deployment_checkpoint_wrong_class_history_or_weights_rejected(change):
    payload = BoundedThreePacketGRUWorldModel(hidden_size=4).deployment_checkpoint()
    key = next(iter(payload["weights"]))
    if change == "class":
        payload["configuration"]["model_class"] = "GRUResidualRewardWorldModel"
    elif change == "window":
        payload["configuration"]["real_packet_window"] = 4
    elif change == "phase":
        payload["configuration"]["assimilation"] = "reuse old hidden"
    elif change == "extra":
        payload["weights"]["extra"] = torch.ones(1)
    elif change == "shape":
        payload["weights"][key] = torch.zeros(1)
    elif change == "nan":
        payload["weights"][key].fill_(float("nan"))
    else:
        payload["weights"][key] = payload["weights"][key].double()
    with pytest.raises(ValueError):
        BoundedThreePacketGRUWorldModel.from_deployment_checkpoint(payload)


def test_parameter_and_operation_counts_include_masked_startup_replay():
    model = BoundedThreePacketGRUWorldModel()
    counted, handles = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_hook(
                    lambda layer, inputs, result: counted.append(
                        inputs[0].shape[0] * layer.in_features * layer.out_features
                    )
                )
            )
        elif isinstance(module, nn.GRUCell):
            handles.append(
                module.register_forward_hook(
                    lambda layer, inputs, result: counted.append(
                        inputs[0].shape[0] * 3 * layer.hidden_size * (layer.input_size + layer.hidden_size)
                    )
                )
            )
    state = model.assimilate(model.initial(2), packets()[:, 0])
    state, _, _ = model.advance(state, torch.zeros(2, 2))
    model.advance(state, torch.zeros(2, 2))
    for handle in handles:
        handle.remove()
    accounting = parameter_and_operation_accounting(model, assimilate_samples=2, advance_samples=4)
    assert accounting["trainable_parameters"] == 36805 == sum(p.numel() for p in model.parameters())
    assert accounting["dense_affine_macs"] == sum(counted)
    assert accounting["replayed_observation_update_samples"] == 6
    assert accounting["replayed_transition_samples"] == 4
    assert accounting["analytic_reward_sample_calls"] == 8
    assert accounting["startup_masked_work_is_counted"] and not accounting["compute_matched"]


@pytest.mark.parametrize("change", ["extra", "mask", "padding", "actions", "depth", "known_nan"])
def test_invalid_state_schema_and_alignment_rejected(change):
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    state = model.initial(2)
    if change == "extra":
        state["privileged_velocity"] = torch.zeros(2, 2)
    elif change == "mask":
        state["real_valid"][:, 0] = True
    elif change == "padding":
        state["real_packets"][:, 0, 0] = 1
    elif change == "actions":
        state["real_actions"][:, 0] = 0.2
    elif change == "depth":
        state["imagined_depth"] = state["imagined_depth"].float()
    else:
        state["real_packets"][:, :, 4] = float("nan")
    with pytest.raises(ValueError):
        model.assimilate(state, packets()[:, 0])


def test_mixed_startup_lengths_replay_independently():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    public, issued = packets(), commands()
    first = model.initial(1)
    second = rollout_real(model, public[1:2, :3], issued[1:2, :2])
    second, _, _ = model.advance(second, issued[1:2, 2])
    mixed = {key: torch.cat((first[key], second[key]), 0) for key in first}
    actual = model.assimilate(mixed, public[:, 3])
    expected = [model.assimilate(first, public[0:1, 3]), model.assimilate(second, public[1:2, 3])]
    for index in range(2):
        for key in actual:
            torch.testing.assert_close(
                actual[key][index : index + 1], expected[index][key], rtol=1e-6, atol=1e-7
            )


def test_action_and_metadata_inputs_never_mutated():
    model = BoundedThreePacketGRUWorldModel(hidden_size=4)
    incoming = packets()[:, 0]
    original = incoming.clone()
    root = model.assimilate(model.initial(2), incoming)
    original_root = {key: value.clone() for key, value in root.items()}
    action = torch.full((2, 2), 0.3)
    before = action.clone()
    result, _, _ = model.advance(root, action)
    result["pending_action"].fill_(0.7)
    result["real_packets"].fill_(123)
    torch.testing.assert_close(incoming, original, rtol=0, atol=0)
    torch.testing.assert_close(action, before, rtol=0, atol=0)
    equal_state(root, original_root)


def test_configuration_has_static_provenance_not_execution_status():
    value = BoundedThreePacketGRUWorldModel(hidden_size=4).configuration()
    assert value["run_status_authority"] == "enclosing protocol and execution receipts"
    assert not {"engineering_only", "integrated_study"} & value.keys()
