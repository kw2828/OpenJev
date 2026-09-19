"""Analytical tests of public-target residual moment supervision."""

import copy
import math

import pytest
import torch

from openjev.research.reacher_innovation_loss import diagonal_residual_moment_score


def score(mean, variance, targets):
    return diagonal_residual_moment_score(
        mean, variance, targets, variance_min=1e-4, variance_max=4.0,
    )


def targets(batch=2):
    result = torch.zeros(batch, 8, dtype=torch.float64)
    result[:, :2] = 1
    result[:, 6] = 1
    return result


def test_exact_score_and_only_variance_receives_supervision_gradient():
    mean = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    variance = torch.full_like(mean, 0.5, requires_grad=True)
    observed = targets().requires_grad_()
    loss, metrics = score(mean, variance, observed)
    assert float(loss.detach()) == pytest.approx(0.5 * (4 * math.log(0.5) + 4))
    loss.backward()
    assert mean.grad is None and observed.grad is None
    expected = torch.tensor([[-0.5, -0.5, 0.5, 0.5]] * 2, dtype=torch.float64)
    torch.testing.assert_close(variance.grad, expected)
    assert metrics["valid_variance_targets"] == 2


def test_missing_targets_are_excluded_and_poison_has_no_effect():
    mean = torch.zeros(2, 4, dtype=torch.float64)
    variance = torch.full_like(mean, 0.5, requires_grad=True)
    observed = targets()
    observed[1, 6] = 0
    observed[1, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e9])
    loss, metrics = score(mean, variance, observed)
    reference, _ = score(mean[:1], variance[:1], targets(1))
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    loss.backward()
    assert not variance.grad[1].any()
    assert metrics["valid_variance_targets"] == 1


def test_all_absent_targets_give_zero_loss_and_defined_zero_variance_gradient():
    observed = targets()
    observed[:, 6] = 0
    observed[:, :4] = float("nan")
    variance = torch.full((2, 4), 0.2, dtype=torch.float64, requires_grad=True)
    loss, metrics = score(torch.zeros_like(variance), variance, observed)
    loss.backward()
    assert loss.item() == 0 and variance.grad is not None and not variance.grad.any()
    assert all(value == 0 for value in metrics.values())


def test_conditional_residual_second_moment_is_stationary_even_for_biased_mean():
    # Both targets are valid cosine/sine pairs, and their mean is [0,1,0,0].
    observed = targets()
    observed[1, 0] = -1
    mean = torch.tensor([[0.5, 0.5, 0.25, -0.25]] * 2, dtype=torch.float64)
    residual_moment = (observed[:, :4] - mean).square().mean(0)
    shared = residual_moment.clone().requires_grad_()
    loss, _ = score(mean, shared.expand_as(mean), observed)
    loss.backward()
    torch.testing.assert_close(shared.grad, torch.zeros_like(shared), rtol=0, atol=1e-14)
    assert residual_moment[0] > observed[:, 0].var(unbiased=False)


def test_bound_fractions_and_extra_sequence_dimension():
    observed = targets().unsqueeze(0)
    mean = torch.zeros(1, 2, 4, dtype=torch.float64)
    variance = torch.tensor([[[1e-4, 4.0, 1.0, 1.0]] * 2], dtype=torch.float64)
    _, metrics = score(mean, variance, observed)
    assert metrics["variance_lower_bound_fraction"] == 0.25
    assert metrics["variance_upper_bound_fraction"] == 0.25


@pytest.mark.parametrize("defect", ["nonfinite_mean", "zero_variance", "oversize_variance",
                                    "invalid_validity", "visible_nan", "visible_age"])
def test_reject_invalid_public_targets_and_priors(defect):
    mean, variance, observed = torch.zeros(2, 4, dtype=torch.float64), torch.ones(2, 4, dtype=torch.float64), targets()
    if defect == "nonfinite_mean":
        mean[0, 0] = float("nan")
    elif defect == "zero_variance":
        variance[0, 0] = 0
    elif defect == "oversize_variance":
        variance[0, 0] = 5
    elif defect == "invalid_validity":
        observed[0, 6] = 0.5
    elif defect == "visible_nan":
        observed[0, 0] = float("nan")
    else:
        observed[0, 7] = 0.2
    with pytest.raises(ValueError):
        score(mean, variance, observed)


@pytest.fixture
def small_model():
    from openjev.research.reacher_innovation_context import InnovationContextWorldModel

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            yield InnovationContextWorldModel("normalized", hidden_size=4, context_size=4).double()
    finally:
        torch.set_num_threads(previous)


def sequence(steps=14):
    batch = 2
    angles = torch.linspace(-0.4, 0.8, batch * (steps + 1) * 2, dtype=torch.float64).reshape(batch, steps + 1, 2)
    packets = torch.zeros(batch, steps + 1, 8, dtype=torch.float64)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.tensor([0.1, -0.09], dtype=torch.float64)
    packets[..., 6] = 1
    for case, stop in enumerate((8, 12)):
        for point in range(3, min(stop + 1, steps + 1)):
            packets[case, point, :4] = 0
            packets[case, point, 6] = 0
            packets[case, point, 7] = (point - 2) * 0.02
    commands = torch.linspace(-0.7, 0.8, batch * steps * 2, dtype=torch.float64).reshape(batch, steps, 2)
    rewards = torch.linspace(-0.9, -0.1, batch * steps, dtype=torch.float64).reshape(batch, steps)
    return packets, commands, rewards


@pytest.mark.parametrize("variant", ["constant", "age", "raw", "normalized"])
def test_variance_score_adds_only_variance_head_gradients_to_original_objective(small_model, variant):
    from openjev.research.reacher_innovation_loss import innovation_sequence_loss
    from openjev.research.reacher_world_models import sequence_loss

    small_model.variant = variant
    reference = copy.deepcopy(small_model)
    packets, commands, rewards = sequence()
    loss, metrics = innovation_sequence_loss(
        small_model, packets, commands, rewards, variance_score_weight=0.1,
    )
    base, original = sequence_loss(reference, packets, commands, rewards)
    for name in ("observation_mse", "reward_mse", "rollout_observation_mse", "rollout_reward_mse",
                 "valid_observation_targets", "valid_rollout_starts"):
        assert metrics[name] == original[name]
    assert metrics["prediction_objective"] == float(base.detach())
    assert float(loss.detach()) == pytest.approx(float(base.detach()) + 0.1 * metrics["residual_moment_score"])
    loss.backward()
    base.backward()
    for (name, actual), (expected_name, expected) in zip(
        small_model.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert name == expected_name
        if name.startswith("variance_head."):
            assert expected.grad is None
            assert actual.grad is not None and actual.grad.abs().sum() > 0
        else:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=1e-12)


@pytest.mark.parametrize("horizon,weight", [(1, 0.5), (5, 0.0), (20, 0.5)])
def test_zero_auxiliary_weight_preserves_prediction_objective_and_window_masks(small_model, horizon, weight):
    from openjev.research.reacher_innovation_loss import innovation_sequence_loss
    from openjev.research.reacher_world_models import sequence_loss

    packets, commands, rewards = sequence()
    loss, metrics = innovation_sequence_loss(
        small_model, packets, commands, rewards, variance_score_weight=0,
        rollout_horizon=horizon, rollout_weight=weight,
    )
    base, original = sequence_loss(
        small_model, packets, commands, rewards, rollout_horizon=horizon, rollout_weight=weight,
    )
    torch.testing.assert_close(loss, base, rtol=0, atol=0)
    assert metrics["valid_rollout_starts"] == original["valid_rollout_starts"]


def test_auxiliary_score_uses_action_priors_and_terminal_is_target_only(small_model, monkeypatch):
    from openjev.research.reacher_innovation_loss import innovation_sequence_loss

    packets, commands, rewards = sequence(50)
    captured_mean, captured_variance, assimilations = [], [], []
    advance, assimilate = small_model.advance, small_model.assimilate

    def record_advance(state, action):
        next_state, mean, reward = advance(state, action)
        captured_mean.append(mean)
        captured_variance.append(next_state["prior_variance"])
        return next_state, mean, reward

    def record_assimilation(state, packet):
        assimilations.append(state["real_index"].clone())
        return assimilate(state, packet)

    monkeypatch.setattr(small_model, "advance", record_advance)
    monkeypatch.setattr(small_model, "assimilate", record_assimilation)
    _, metrics = innovation_sequence_loss(
        small_model, packets, commands, rewards, variance_score_weight=0.1, rollout_weight=0,
    )
    assert len(assimilations) == len(captured_mean) == 50
    assert assimilations[-1].unique().item() == 48
    expected, _ = score(torch.stack(captured_mean, 1), torch.stack(captured_variance, 1), packets[:, 1:])
    assert metrics["residual_moment_score"] == expected.item()
    assert metrics["valid_variance_targets"] == float(packets[:, 1:, 6].sum())


def test_short_optimizer_integration_keeps_all_four_variants_finite(small_model):
    from openjev.research.reacher_innovation_loss import innovation_sequence_loss

    packets, commands, rewards = sequence()
    for variant in ("constant", "age", "raw", "normalized"):
        model = copy.deepcopy(small_model)
        model.variant = variant
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = innovation_sequence_loss(model, packets, commands, rewards, variance_score_weight=0.1)
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
