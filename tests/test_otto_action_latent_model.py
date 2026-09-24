"""Fabricated action-latent contracts and interventions; no empirical inputs."""
from __future__ import annotations

import inspect

import pytest
import torch

from openjev.research import otto_action_latent_model as models


def packet(horizon=8):
    prefix = torch.linspace(-.3, .7, 2 * 5 * 31, dtype=torch.float32).reshape(2, 5, 31)
    lengths = torch.tensor([5, 3], dtype=torch.int64)
    prefix[1, 3:] = float("nan")
    actions = torch.arange(2 * horizon, dtype=torch.int64).reshape(2, horizon) % 4
    observations = torch.cos(torch.arange(2 * horizon * 31, dtype=torch.float32)).reshape(2, horizon, 31) * .4
    found = torch.zeros((2, horizon), dtype=torch.bool)
    return prefix, lengths, actions, observations, found


def same(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().contiguous().reshape(-1).view(torch.uint8))


def output_same(left, right, take=None):
    for name in ("outcome_logits", "cost_contrasts", "aux_features", "prior_states"):
        x, y = left[name], right[name]
        if take is not None:
            x, y = x[:, take], y[:, take]
        same(x, y)


def test_seeded_construction_exact_counts_shared_initial_modules_and_rng_preservation():
    before = torch.random.get_rng_state().clone()
    controls = {kind: models.make_model(kind, 7) for kind in models.KINDS}
    same(before, torch.random.get_rng_state())
    for kind, model in controls.items():
        replica = models.ActionLatentModel(kind, 7)
        assert model.parameter_metadata()["count"] == models.PARAMETER_COUNTS[kind]
        assert model.parameter_metadata()["trainable_count"] == models.PARAMETER_COUNTS[kind]
        assert sum(p.numel() for p in model.parameters()) == models.PARAMETER_COUNTS[kind]
        for name, value in model.state_dict().items():
            same(value, replica.state_dict()[name])
        common = controls["action_recurrent"].state_dict()
        for name, value in model.state_dict().items():
            if name.startswith(("assimilation.", "outcome_head.", "cost_head.", "aux_head.")):
                same(value, common[name])
    assert max(models.PARAMETER_COUNTS.values()) / min(models.PARAMETER_COUNTS.values()) < 1.15
    assert controls["action_blind"].parameter_metadata()["zero_proposed_action_weight_parameters"] == 336
    same(before, torch.random.get_rng_state())


@pytest.mark.parametrize("kind", models.KINDS)
def test_blind_shapes_centering_finite_outputs_and_exact_executed_work(kind):
    model = models.make_model(kind, 7)
    prefix, lengths, actions, _, _ = packet()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = model.blind_rollout(prefix, lengths, actions)
    assert set(result) == models.OUTPUT_FIELDS
    for name, shape in (("outcome_logits", (2, 8, 5)), ("cost_contrasts", (2, 8, 4)),
                        ("aux_features", (2, 8, 2)), ("prior_states", (2, 8, 28)), ("prefix_state", (2, 28))):
        assert result[name].dtype == torch.float32 and tuple(result[name].shape) == shape
        assert torch.isfinite(result[name]).all()
    assert result["posterior_states"] is None
    torch.testing.assert_close(result["cost_contrasts"].sum(-1), torch.zeros(2, 8), rtol=0, atol=1e-7)
    work = result["work"]
    assert work["prefix_assimilation_calls"] == 5 and work["prefix_assimilation_rows"] == 8
    assert work["outcome_head_calls"] == work["cost_head_calls"] == work["aux_head_calls"] == 8
    assert work["readout_rows"] == 16 and work["observation_assimilation_rows"] == 0
    if kind == "direct_horizon":
        assert work["direct_token_calls"] == work["direct_prior_calls"] == 8
        assert work["direct_token_rows"] == 2 * 36 and work["direct_prior_rows"] == 16
        assert work["transition_calls"] == work["transition_rows"] == 0
    else:
        assert work["transition_calls"] == 8 and work["transition_rows"] == 16
        assert work["direct_token_calls"] == work["direct_prior_calls"] == 0
    output_same(result, model(prefix, lengths, actions))
    for name, value in before.items():
        same(value, model.state_dict()[name])


def test_blind_signature_rejects_future_observations_found_and_teacher_inputs():
    assert tuple(inspect.signature(models.ActionLatentModel.blind_rollout).parameters) == (
        "self", "prefix", "prefix_lengths", "actions")
    prefix, lengths, actions, observations, found = packet()
    model = models.make_model("action_recurrent", 7)
    for extra in ({"continuation": observations}, {"found": found}, {"teacher_scores": torch.ones(2, 8, 4)}):
        with pytest.raises(TypeError):
            model.blind_rollout(prefix, lengths, actions, **extra)


@pytest.mark.parametrize("kind", models.KINDS)
def test_future_actions_cannot_change_earlier_horizons_and_short_prefix_repeats(kind):
    model = models.make_model(kind, 13)
    prefix, lengths, actions, _, _ = packet()
    changed = actions.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 4
    baseline = model.blind_rollout(prefix, lengths, actions)
    intervened = model.blind_rollout(prefix, lengths, changed)
    output_same(baseline, intervened, slice(0, 4))
    shorter = model.blind_rollout(prefix, lengths, actions[:, :4])
    for name in ("outcome_logits", "cost_contrasts", "aux_features", "prior_states"):
        same(baseline[name][:, :4], shorter[name])
    if kind == "action_blind":
        output_same(baseline, intervened)
    else:
        assert not torch.equal(baseline["prior_states"][:, 4], intervened["prior_states"][:, 4])


def test_direct_shared_position_encoder_is_causal_order_sensitive_and_not_a_slot_table():
    model = models.make_model("direct_horizon", 11)
    actions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int64)
    tokens = models.action_position_features(actions)
    assert tokens.shape == (1, 8, 9) and tokens.dtype == torch.float32
    same(tokens[:, :4], models.action_position_features(actions[:, :4]))
    assert tuple(model.action_encoder.weight.shape) == (48, 9)
    descriptor = model.direct_descriptor(actions)
    assert descriptor.shape == (1, 8, 49)
    same(descriptor[:, :4], model.direct_descriptor(actions[:, :4]))
    reordered = actions.clone()
    reordered[:, :4] = torch.tensor([3, 2, 1, 0])
    assert not torch.equal(descriptor[:, 3], model.direct_descriptor(reordered)[:, 3])
    assert torch.equal(descriptor[0, :, -1], torch.arange(1, 9, dtype=torch.float32) / 8)
    with pytest.raises(ValueError, match="direct descriptor"):
        models.make_model("action_recurrent").direct_descriptor(actions)


@pytest.mark.parametrize("kind", models.KINDS)
def test_poisoned_prefix_padding_is_ignored_without_mutation(kind):
    model = models.make_model(kind, 23)
    prefix, lengths, actions, _, _ = packet(4)
    before = prefix.detach().contiguous().view(torch.uint8).clone()
    first = model.blind_rollout(prefix, lengths, actions)
    altered = prefix.clone()
    altered[1, 3:] = float("inf")
    output_same(first, model.blind_rollout(altered, lengths, actions))
    same(before, prefix.detach().contiguous().view(torch.uint8))
    # No mutable carry leaks between independent invocations.
    model.blind_rollout(prefix.flip(0), lengths.flip(0), actions.flip(0))
    output_same(first, model.blind_rollout(prefix, lengths, actions))


@pytest.mark.parametrize("kind", models.KINDS)
def test_normal_forecast_precedes_own_observation_and_future_actions(kind):
    model = models.make_model(kind, 29)
    prefix, lengths, actions, observations, found = packet(4)
    first = model.normal_rollout(prefix, lengths, actions, observations, found=found)
    blind = model.blind_rollout(prefix, lengths, actions)
    output_same(first, blind, slice(0, 1))
    changed = observations.clone()
    changed[:, 1] += .7
    other = model.normal_rollout(prefix, lengths, actions, changed, found=found)
    output_same(first, other, slice(0, 2))
    assert not torch.equal(first["posterior_states"][:, 1], other["posterior_states"][:, 1])
    assert not torch.equal(first["prior_states"][:, 2], other["prior_states"][:, 2])
    future = observations.clone()
    future[:, 3] *= -5
    output_same(first, model.normal_rollout(prefix, lengths, actions, future, found=found))
    future_actions = actions.clone()
    future_actions[:, 3] = (future_actions[:, 3] + 1) % 4
    output_same(first, model.normal_rollout(prefix, lengths, future_actions, observations, found=found), slice(0, 3))
    assert first["posterior_states"].shape == (2, 4, 28)
    assert first["work"]["observation_assimilation_calls"] == 4
    assert first["work"]["observation_assimilation_rows"] == 8
    if kind == "direct_horizon":
        assert first["work"]["direct_token_rows"] == 8  # one action after every observation


@pytest.mark.parametrize("kind", models.KINDS)
def test_observed_found_is_used_after_prediction_and_freezes_future_state(kind):
    model = models.make_model(kind, 31)
    prefix, lengths, actions, observations, found = packet(4)
    baseline = model.normal_rollout(prefix, lengths, actions, observations, found=found)
    found[0, 1] = True  # Event flags need not be repeated after first detection.
    observations[0, 1:] = float("nan")
    terminal = model.normal_rollout(prefix, lengths, actions, observations, found=found)
    for name in ("outcome_logits", "cost_contrasts", "aux_features", "prior_states"):
        same(baseline[name][0, :2], terminal[name][0, :2])
        same(terminal[name][0, 1], terminal[name][0, 2])
        same(terminal[name][0, 2], terminal[name][0, 3])
    assert terminal["work"]["terminal_frozen_rows"] == 2
    assert terminal["work"]["observation_assimilation_rows"] == 5
    assert torch.isfinite(terminal["outcome_logits"]).all()
    assert (terminal["outcome_logits"].softmax(-1) > 0).all()  # No label-forced probability one.


@pytest.mark.parametrize("kind", models.KINDS)
def test_gradients_reach_shared_heads_dynamics_and_observation_encoder(kind):
    model = models.make_model(kind, 37)
    prefix, lengths, actions, observations, found = packet(4)
    for normal in (False, True):
        model.zero_grad(set_to_none=True)
        result = (model.normal_rollout(prefix, lengths, actions, observations, found=found) if normal
                  else model.blind_rollout(prefix, lengths, actions))
        labels = torch.arange(8, dtype=torch.int64).reshape(2, 4) % 5
        loss = torch.nn.functional.cross_entropy(result["outcome_logits"].reshape(-1, 5), labels.reshape(-1))
        target = torch.linspace(-.2, .4, 32).reshape(2, 4, 4)
        target -= target.mean(-1, keepdim=True)
        loss = loss + (result["cost_contrasts"] - target).square().mean()
        loss = loss + (result["aux_features"] - .2).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if kind == "action_blind" and name == "transition.weight_ih":
                assert not bool(parameter.grad.any())
            else:
                assert bool(parameter.grad.any()), name
        if kind == "direct_horizon":
            assert bool((model.action_encoder.weight.grad.abs().sum(0) > 0).all())


def test_normal_observation_gradient_cannot_reach_its_own_or_future_prior_forecast():
    model = models.make_model("action_recurrent", 43)
    prefix, lengths, actions, observations, found = packet(4)
    observations.requires_grad_()
    result = model.normal_rollout(prefix, lengths, actions, observations, found=found)
    gradient = torch.autograd.grad(result["outcome_logits"][:, 1, 0].sum(), observations)[0]
    assert bool(gradient[:, 0].abs().sum() > 0)
    assert not bool(gradient[:, 1:].any())


@pytest.mark.parametrize("defect", ("prefix_dtype", "prefix_shape", "prefix_nan", "length_dtype", "length_zero",
                                  "length_overflow", "action_dtype", "action_range", "horizon", "batch"))
def test_invalid_blind_inputs_fail_before_any_recurrent_call(defect):
    model = models.make_model("action_recurrent", 47)
    prefix, lengths, actions, _, _ = packet(4)
    if defect == "prefix_dtype":
        prefix = prefix.double()
    elif defect == "prefix_shape":
        prefix = prefix[:, :, :30]
    elif defect == "prefix_nan":
        prefix[0, 0, 0] = float("nan")
    elif defect == "length_dtype":
        lengths = lengths.float()
    elif defect == "length_zero":
        lengths[0] = 0
    elif defect == "length_overflow":
        lengths[0] = 6
    elif defect == "action_dtype":
        actions = actions.float()
    elif defect == "action_range":
        actions[0, 0] = 4
    elif defect == "horizon":
        actions = torch.zeros(2, 9, dtype=torch.int64)
    else:
        actions = actions[:1]
    calls = []
    hook = model.assimilation.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError):
            model.blind_rollout(prefix, lengths, actions)
    finally:
        hook.remove()
    assert not calls


@pytest.mark.parametrize("defect", ("continuation_shape", "continuation_nan", "found_dtype", "found_shape"))
def test_invalid_observed_panel_fails_before_prefix_assimilation(defect):
    model = models.make_model("direct_horizon", 51)
    prefix, lengths, actions, observations, found = packet(4)
    if defect == "continuation_shape":
        observations = observations[:, :, :30]
    elif defect == "continuation_nan":
        observations[0, 0, 0] = float("nan")
    elif defect == "found_dtype":
        found = found.long()
    else:
        found = found[:, :3]
    calls = []
    hook = model.assimilation.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError):
            model.normal_rollout(prefix, lengths, actions, observations, found=found)
    finally:
        hook.remove()
    assert not calls


@pytest.mark.parametrize("kind,seed", (("unknown", 1), ("action_recurrent", True), ("action_recurrent", -1),
                                     ("action_recurrent", 2**32)))
def test_invalid_factory_preserves_rng(kind, seed):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        models.make_model(kind, seed)
    same(before, torch.random.get_rng_state())


@pytest.mark.parametrize("kind", models.KINDS)
def test_fixed_scale_zero_initial_head_checkpoint_and_standardized_gradient(kind):
    first = models.make_model(kind, 57)
    second = models.make_model(kind, 57, cost_scale=.01)
    assert "cost_scale" in second.state_dict() and "cost_scale" not in dict(second.named_parameters())
    assert second.cost_scale.shape == () and not second.cost_scale.requires_grad
    assert second.parameter_metadata()["count"] == first.parameter_metadata()["count"]
    assert not bool(first.cost_head.weight.any()) and not bool(first.cost_head.bias.any())
    assert not bool(second.cost_head.weight.any()) and not bool(second.cost_head.bias.any())
    prior = torch.linspace(-.4, .6, 56).reshape(2, 28)
    target = torch.tensor([[.2, -.3, .4, -.3], [-.1, .3, .2, -.4]])
    for model in (first, second):
        with torch.no_grad():
            model.cost_head.weight.copy_(torch.arange(112).reshape(4, 28) / 2000)
            model.cost_head.bias.copy_(torch.tensor([.03, -.02, .01, .04]))
        prediction = model._read(prior, model._work())[1]
        scaled_target = target * model.cost_scale
        loss = ((prediction - scaled_target) / model.cost_scale).square().mean()
        loss.backward()
    torch.testing.assert_close(second.cost_head.weight.grad, first.cost_head.weight.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(second.cost_head.bias.grad, first.cost_head.bias.grad, rtol=1e-6, atol=1e-7)
    expected = first._read(prior, first._work())[1] * second.cost_scale
    torch.testing.assert_close(second._read(prior, second._work())[1], expected, rtol=0, atol=0)
    restored = models.make_model(kind, 99)
    restored.load_state_dict(second.state_dict())
    same(restored.cost_scale, second.cost_scale)
    torch.testing.assert_close(restored._read(prior, restored._work())[1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", (True, 0, -1, float("nan"), float("inf"), 1e100, 1e-100, 10**1000))
def test_invalid_fixed_cost_scale_preserves_rng(scale):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="scale"):
        models.make_model("action_recurrent", 61, cost_scale=scale)
    same(before, torch.random.get_rng_state())
