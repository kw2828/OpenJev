"""Random-weight and analytic fixtures only; no training or collected data."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn

from openjev.research.reacher_latent_consistency import LatentConsistencyAuxiliary
from openjev.research.reacher_raw_endpoint import RawEndpointAuxiliary
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import GaussianRSSM, GRUWorldModel


@pytest.fixture(autouse=True)
def single_thread_seed():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1827)
    yield
    torch.set_num_threads(threads)


def sequence(batch=2, steps=6):
    packets = torch.zeros(batch, steps + 1, 8)
    angles = torch.randn(batch, steps + 1, 2)
    packets[..., :4] = torch.cat((angles.cos(), angles.sin()), -1)
    packets[..., 4:6] = torch.tensor([0.1, -0.2])
    packets[..., 6] = 1
    commands = torch.randn(batch, steps, 2).clamp(-1, 1)
    return packets, commands


class AdditiveGRU(GRUWorldModel):
    """An analytic causal fixture deliberately trusting all supplied angles."""

    def __init__(self):
        super().__init__(hidden_size=2)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.observation_head = nn.Linear(2, 4)
        with torch.no_grad():
            self.observation_head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 2.]]))
            self.observation_head.bias.zero_()

    def assimilate(self, state, packet):
        return {"packet": packet, "hidden": state["hidden"] + packet[:, :2]}

    def advance(self, state, action):
        hidden = state["hidden"] + self.gain * action
        return {"packet": state["packet"], "hidden": hidden}, action, action.sum(-1)


def identity_head(auxiliary):
    with torch.no_grad():
        auxiliary.predictor.weight.copy_(torch.eye(auxiliary.predictor.in_features))
        auxiliary.predictor.bias.zero_()


@pytest.mark.parametrize("model_class", [GRUWorldModel, GRUResidualRewardWorldModel])
def test_gradients_reach_student_predictor_and_shared_decoder(model_class):
    model = model_class(hidden_size=8)
    auxiliary = RawEndpointAuxiliary(model)
    loss, metrics = auxiliary(model, *sequence())
    assert torch.isfinite(loss)
    assert metrics["valid_pairs_by_horizon"] == {"1": 12, "3": 8, "7": 0}
    loss.backward()
    for parameter in (model.transition.weight_ih, model.observation_update.weight_ih,
                      model.observation_head[0].weight, auxiliary.predictor.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.reward_head.parameters())
    assert all(p is not q for p in model.parameters() for q in auxiliary.parameters())
    assert dict(auxiliary.named_children()) == {"predictor": auxiliary.predictor}
    assert set(auxiliary.state_dict()) == {"predictor.weight", "predictor.bias"}


@pytest.mark.parametrize("hidden_size", [2, 8, 64])
def test_extra_trainable_capacity_matches_latent_predictor(hidden_size):
    student = GRUWorldModel(hidden_size=hidden_size)
    raw = RawEndpointAuxiliary(student)
    latent = LatentConsistencyAuxiliary(student)
    raw_parameters = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    latent_parameters = sum(p.numel() for p in latent.parameters() if p.requires_grad)
    assert raw_parameters == latent_parameters == hidden_size * (hidden_size + 1)
    assert raw.configuration()["extra_trainable_parameters"] == raw_parameters


def test_action_alignment_raw_targets_and_detachment_are_exact():
    model = AdditiveGRU()
    auxiliary = RawEndpointAuxiliary(model, horizons=(1, 3))
    identity_head(auxiliary)
    packets, actions = sequence(batch=1, steps=3)
    packets[0, :, :2] = torch.tensor([[1, 2], [10, 20], [100, 200], [1000, 2000]])
    actions[0] = torch.tensor([[3, 5], [7, 11], [13, 17]])
    packets.requires_grad_()
    pairs = auxiliary.pairs(model, packets, actions)
    torch.testing.assert_close(pairs[1].prediction[0, 0], torch.tensor([4., 7., 8., 14.]))
    torch.testing.assert_close(pairs[1].prediction[0, 1], torch.tensor([21., 38., 42., 76.]))
    torch.testing.assert_close(pairs[3].prediction[0, 0], torch.tensor([24., 35., 48., 70.]))
    for horizon, pair in pairs.items():
        torch.testing.assert_close(pair.target, packets[:, horizon:, :4], rtol=0, atol=0)
        assert not pair.target.requires_grad


@pytest.mark.parametrize("model_class", [GRUWorldModel, AdditiveGRU])
def test_future_public_packets_never_enter_root_predictions(model_class):
    model = model_class() if model_class is AdditiveGRU else model_class(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model, horizons=(1, 3, 6))
    packets, commands = sequence(batch=1)
    original = auxiliary.pairs(model, packets, commands)
    changed = packets.clone()
    changed[:, 1:, :6] += 100
    changed[:, 1:, 7] += 1
    altered = auxiliary.pairs(model, changed, commands)
    for horizon in auxiliary.horizons:
        torch.testing.assert_close(original[horizon].prediction[:, 0], altered[horizon].prediction[:, 0],
                                   rtol=0, atol=0)
        torch.testing.assert_close(altered[horizon].target[:, 0], changed[:, horizon, :4], rtol=0, atol=0)
        assert not torch.equal(original[horizon].target[:, 0], altered[horizon].target[:, 0])
    only_nonangular = packets.clone()
    only_nonangular[:, 1:, 4:6] += 10
    targets = auxiliary.pairs(model, only_nonangular, commands)
    for horizon in auxiliary.horizons:
        torch.testing.assert_close(original[horizon].target, targets[horizon].target, rtol=0, atol=0)


def test_command_gradient_uses_exact_imagined_window_and_packets_only_at_root():
    model = AdditiveGRU()
    auxiliary = RawEndpointAuxiliary(model, horizons=(3,))
    identity_head(auxiliary)
    packets, commands = sequence(batch=1)
    packets.requires_grad_()
    commands.requires_grad_()
    prediction = auxiliary.pairs(model, packets, commands)[3].prediction[:, 0].sum()
    packet_gradient, command_gradient = torch.autograd.grad(prediction, (packets, commands))
    torch.testing.assert_close(command_gradient[:, :3], torch.full((1, 3, 2), 3.))
    torch.testing.assert_close(command_gradient[:, 3:], torch.zeros(1, 3, 2))
    torch.testing.assert_close(packet_gradient[:, 1:], torch.zeros(1, 6, 8))


@pytest.mark.parametrize("padding", [False, True])
def test_masks_and_pair_counts_exactly_match_latent_auxiliary(padding):
    model = GRUWorldModel(hidden_size=4)
    raw, latent = RawEndpointAuxiliary(model), LatentConsistencyAuxiliary(model)
    packets, commands = sequence(batch=3, steps=9)
    packets[0, :, 6] = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    packets[1, 1:7, 6] = 0
    packets[2, :, 6] = 0
    kwargs = {}
    if padding:
        kwargs["transition_valid"] = torch.tensor([[True] * 9, [True] * 7 + [False] * 2, [False] * 9])
    raw_pairs = raw.pairs(model, packets, commands, **kwargs)
    latent_pairs = latent.pairs(model, packets, commands, **kwargs)
    for horizon in raw.horizons:
        torch.testing.assert_close(raw_pairs[horizon].valid, latent_pairs[horizon].valid, rtol=0, atol=0)
    _, raw_metrics = raw(model, packets, commands, **kwargs)
    _, latent_metrics = latent(model, packets, commands, **kwargs)
    assert raw_metrics["valid_pairs_by_horizon"] == latent_metrics["valid_pairs_by_horizon"]
    assert raw_metrics["nonempty_horizons"] == latent_metrics["nonempty_horizons"]
    for key in ("student_assimilate", "student_prefix_advance", "student_open_loop_advance", "student_predictor"):
        assert raw_metrics["batch_forward_calls"][key] == latent_metrics["batch_forward_calls"][key]


def test_poisoned_missing_angles_are_removed_before_all_model_inputs_and_targets():
    model = AdditiveGRU()
    auxiliary = RawEndpointAuxiliary(model)
    packets, commands = sequence()
    packets[:, 2:4, 6] = 0
    packets[:, 2:4, :4] = 0
    clean = auxiliary.pairs(model, packets, commands)
    poisoned = packets.clone()
    poisoned[:, 2:4, :4] = float("nan")
    hidden_labels = packets.clone()
    hidden_labels[:, 2:4, :4] = 99999
    for altered in (poisoned, hidden_labels):
        pairs = auxiliary.pairs(model, altered, commands)
        for horizon in auxiliary.horizons:
            for field in ("prediction", "target", "valid"):
                torch.testing.assert_close(getattr(clean[horizon], field), getattr(pairs[horizon], field),
                                           rtol=0, atol=0)
        actual, _ = auxiliary(model, altered, commands)
        expected, _ = auxiliary(model, packets, commands)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("missing", [6, 10])
def test_gap_bridge_includes_step_to_next_valid_endpoint(missing):
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model, horizons=(1, 3, missing, missing + 1))
    packets, commands = sequence(batch=1, steps=missing + 1)
    packets[:, 1:-1, 6] = 0
    packets[:, 1:-1, :4] = float("nan")
    pairs = auxiliary.pairs(model, packets, commands)
    assert not pairs[missing].valid.any()
    assert pairs[missing + 1].valid.tolist() == [[True]]
    loss, metrics = auxiliary(model, packets, commands)
    assert metrics["valid_pairs_by_horizon"][str(missing)] == 0
    assert metrics["valid_pairs_by_horizon"][str(missing + 1)] == 1
    loss.backward()
    assert torch.isfinite(model.transition.weight_ih.grad).all()
    assert model.transition.weight_ih.grad.abs().sum() > 0


def test_default_horizons_bridge_six_missing_packets_only_at_seven():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    packets, commands = sequence(batch=1, steps=7)
    packets[:, 1:-1, 6] = 0
    _, metrics = auxiliary(model, packets, commands)
    assert metrics["valid_pairs_by_horizon"] == {"1": 0, "3": 0, "7": 1}


def test_terminal_endpoint_allowed_and_poisoned_padding_cannot_change_loss():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    packets, commands = sequence(batch=1)
    expected, expected_metrics = auxiliary(model, packets[:, :4], commands[:, :3])
    packets[:, 4:] = float("nan")
    commands[:, 3:] = float("nan")
    mask = torch.tensor([[True, True, True, False, False, False]])
    actual, metrics = auxiliary(model, packets, commands, transition_valid=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metrics["valid_pairs_by_horizon"] == expected_metrics["valid_pairs_by_horizon"] == {
        "1": 3, "3": 1, "7": 0}
    # Masked padding is still executed, so its cost must not disappear.
    assert metrics["batch_forward_calls"]["student_prefix_advance"] == 6
    assert expected_metrics["batch_forward_calls"]["student_prefix_advance"] == 3


def test_independent_terminal_lengths_and_empty_episode_are_masked():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    mask = torch.tensor([[True, True, False, False, False, False], [True] * 6, [False] * 6])
    _, metrics = auxiliary(model, *sequence(batch=3), transition_valid=mask)
    assert metrics["valid_pairs_by_horizon"] == {"1": 8, "3": 4, "7": 0}


def test_restarted_mask_cannot_join_episodes():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    mask = torch.tensor([[True, False, True, False, False, False]])
    with pytest.raises(ValueError, match="restart/wrap"):
        auxiliary(model, *sequence(batch=1), transition_valid=mask)


@pytest.mark.parametrize("empty_kind", ["no_observations", "no_transitions", "horizons_too_long"])
def test_no_targets_returns_differentiable_zero_without_fabricated_labels(empty_kind):
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model, horizons=(7, 9) if empty_kind == "horizons_too_long" else (1, 3, 7))
    packets, commands = sequence()
    kwargs = {}
    if empty_kind == "no_observations":
        packets[..., 6] = 0
        packets[..., :4] = float("nan")
    if empty_kind == "no_transitions":
        kwargs["transition_valid"] = torch.zeros(2, 6, dtype=torch.bool)
    loss, metrics = auxiliary(model, packets, commands, **kwargs)
    assert loss.item() == 0 and metrics["nonempty_horizons"] == 0
    assert all(count == 0 for count in metrics["valid_pairs_by_horizon"].values())
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())
    assert auxiliary.predictor.weight.grad.abs().sum() == 0
    if empty_kind == "horizons_too_long":
        for pair in auxiliary.pairs(model, packets, commands).values():
            assert pair.prediction.shape == pair.target.shape == (2, 0, 4)
            assert pair.valid.shape == (2, 0)


def test_horizons_have_equal_weight_not_pooled_pair_weight():
    model = AdditiveGRU()
    auxiliary = RawEndpointAuxiliary(model, horizons=(1, 3, 7))
    identity_head(auxiliary)
    packets, commands = sequence(batch=1, steps=3)
    packets[..., :4] = 0
    commands[:] = torch.tensor([1., 2.])
    pairs = auxiliary.pairs(model, packets, commands)
    horizon_losses = [(p.prediction[p.valid] - p.target[p.valid]).square().mean()
                      for p in pairs.values() if p.valid.any()]
    expected = torch.stack(horizon_losses).mean()
    pooled = torch.cat([(p.prediction[p.valid] - p.target[p.valid]).square()
                        for p in pairs.values()]).mean()
    actual, metrics = auxiliary(model, packets, commands)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.isclose(actual, pooled)
    assert metrics["valid_pairs_by_horizon"] == {"1": 3, "3": 1, "7": 0}


def test_call_accounting_matches_observed_interfaces_and_extra_decoder_work():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    decoder_widths, predictor_widths = [], []
    decoder_hook = model.observation_head.register_forward_hook(
        lambda module, inputs, output: decoder_widths.append(inputs[0].shape[0]))
    predictor_hook = auxiliary.predictor.register_forward_hook(
        lambda module, inputs, output: predictor_widths.append(inputs[0].shape[0]))
    try:
        with (patch.object(model, "assimilate", wraps=model.assimilate) as assimilate,
              patch.object(model, "advance", wraps=model.advance) as advance):
            _, metrics = auxiliary(model, *sequence())
        calls = metrics["batch_forward_calls"]
        assert calls == {
            "student_assimilate": 7, "student_prefix_advance": 6, "student_open_loop_advance": 21,
            "student_predictor": 10, "student_endpoint_decoder": 10}
        assert assimilate.call_count == calls["student_assimilate"]
        assert advance.call_count == calls["student_prefix_advance"] + calls["student_open_loop_advance"]
        assert len(predictor_widths) == calls["student_predictor"]
        assert len(decoder_widths) == advance.call_count + calls["student_endpoint_decoder"]
        assert sum(predictor_widths) == metrics["sample_forward_evaluations"]["student_predictor"]
        assert sum(decoder_widths) == 2 * (27 + 10)
        assert metrics["sample_forward_evaluations"] == {key: 2 * value for key, value in calls.items()}
        assert not any("teacher" in key for key in calls)
    finally:
        decoder_hook.remove()
        predictor_hook.remove()


def test_forward_does_not_mutate_inputs_modes_weights_or_existing_gradients():
    model = GRUResidualRewardWorldModel(hidden_size=4).eval()
    auxiliary = RawEndpointAuxiliary(model).train()
    packets, commands = sequence()
    mask = torch.ones(2, 6, dtype=torch.bool)
    before_inputs = [value.clone() for value in (packets, commands, mask)]
    before_weights = [copy.deepcopy(module.state_dict()) for module in (model, auxiliary)]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    auxiliary(model, packets, commands, transition_valid=mask)
    assert not model.training and auxiliary.training
    for current, before in zip((packets, commands, mask), before_inputs, strict=True):
        torch.testing.assert_close(current, before, rtol=0, atol=0)
    for module, before in zip((model, auxiliary), before_weights, strict=True):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())


def test_configuration_is_json_safe_detached_and_checkpoint_reconstructible():
    model = GRUResidualRewardWorldModel(hidden_size=4, dt=0.03, noise_std=0.12, residual_reward=False)
    auxiliary = RawEndpointAuxiliary(model, horizons=(7, 1, 3))
    metadata = auxiliary.configuration()
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["horizons"] == [1, 3, 7]
    assert metadata["model"]["noise_std"] == 0.12
    assert metadata["model"]["residual_reward"] is False
    assert metadata["model"]["dt"] == 0.03
    assert metadata["teacher"] is None
    assert metadata["decoder_receives_auxiliary_gradients"]
    clone = RawEndpointAuxiliary(model, horizons=metadata["horizons"])
    clone.load_state_dict(auxiliary.state_dict())
    packets, commands = sequence()
    expected, _ = auxiliary(model, packets, commands)
    actual, _ = clone(model, packets, commands)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    metadata["horizons"].clear()
    metadata["model"]["noise_std"] = 99
    assert auxiliary.configuration()["horizons"] == [1, 3, 7]
    assert auxiliary.configuration()["model"]["noise_std"] == 0.12


def test_joint_dtype_transfer_is_supported_but_unmatched_dtype_is_rejected():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    model.double()
    packets, commands = (x.double() for x in sequence())
    with pytest.raises(ValueError, match="dtype/device"):
        auxiliary(model, packets, commands)
    auxiliary.double()
    loss, metadata = auxiliary(model, packets, commands)
    assert loss.dtype == torch.float64 and torch.isfinite(loss)
    assert metadata["configuration"]["dtype"] == "torch.float64"


@pytest.mark.parametrize("setting,value", [("hidden_size", 8), ("dt", 0.04),
                                          ("residual_reward", False), ("noise_std", 0.2)])
def test_model_python_configuration_drift_is_rejected(setting, value):
    model = GRUResidualRewardWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    setattr(model, setting, value)
    with pytest.raises(ValueError, match="configuration"):
        auxiliary(model, *sequence())


def test_student_tensor_membership_and_class_drift_are_rejected():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    model.register_buffer("unexpected", torch.tensor(1))
    with pytest.raises(ValueError, match="membership"):
        auxiliary(model, *sequence())
    with pytest.raises(TypeError, match="class"):
        auxiliary(GRUResidualRewardWorldModel(hidden_size=4), *sequence())


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (1, 1), (True,), (1.5,), "1"])
def test_invalid_horizons_rejected(horizons):
    with pytest.raises(ValueError, match="horizons"):
        RawEndpointAuxiliary(GRUWorldModel(hidden_size=4), horizons=horizons)


def test_stochastic_models_are_unsupported():
    with pytest.raises(TypeError, match="deterministic"):
        RawEndpointAuxiliary(GaussianRSSM(hidden_size=4))


@pytest.mark.parametrize("bad", ["visible_nan", "command_nan", "goal_nan", "flag", "age", "dtype",
                               "shape", "command_shape", "mask_dtype", "mask_shape"])
def test_invalid_available_inputs_fail_closed(bad):
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    packets, commands = sequence()
    kwargs = {}
    if bad == "visible_nan":
        packets[0, 0, 0] = float("nan")
    elif bad == "command_nan":
        commands[0, 0, 0] = float("nan")
    elif bad == "goal_nan":
        packets[0, 0, 4] = float("nan")
    elif bad == "flag":
        packets[0, 0, 6] = 0.5
    elif bad == "age":
        packets[0, 0, 7] = -1
    elif bad == "dtype":
        commands = commands.double()
    elif bad == "shape":
        packets = packets[:, :-1]
    elif bad == "command_shape":
        commands = commands[:, :, :1]
    elif bad == "mask_dtype":
        kwargs["transition_valid"] = torch.ones(2, 6)
    else:
        kwargs["transition_valid"] = torch.ones(2, 7, dtype=torch.bool)
    with pytest.raises(ValueError):
        auxiliary(model, packets, commands, **kwargs)


def test_nonfinite_valid_predictions_are_rejected():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = RawEndpointAuxiliary(model)
    with torch.no_grad():
        auxiliary.predictor.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="predictions must be finite"):
        auxiliary(model, *sequence())
