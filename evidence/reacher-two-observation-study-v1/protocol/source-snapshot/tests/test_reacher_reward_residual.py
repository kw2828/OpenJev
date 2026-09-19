"""Analytical and interface engineering fixtures, not performance evidence."""

import inspect
import math

import numpy as np
import pytest
import torch

from openjev.research.reacher_reward_residual import (
    GRUResidualRewardWorldModel,
    expected_clipped_action_cost,
)
from openjev.research.reacher_world_models import GRUWorldModel, sequence_loss


def quadrature_cost(command, sigma):
    """Independent Gauss-Legendre integration in standard-normal coordinates."""
    issued = np.clip(np.asarray(command, dtype=np.float64), -1, 1)
    if sigma == 0:
        return np.square(issued).sum(-1)
    nodes, weights = np.polynomial.legendre.leggauss(160)
    result = np.zeros(issued.shape[:-1])
    for index in np.ndindex(issued.shape[:-1]):
        for u in issued[index]:
            lower, upper = (-1 - u) / sigma, (1 - u) / sigma
            lo, hi = max(lower, -12.0), min(upper, 12.0)
            z = (nodes + 1) * (hi - lo) / 2 + lo
            integral = np.dot(weights, np.square(u + sigma * z) * np.exp(-z * z / 2))
            integral *= (hi - lo) / (2 * math.sqrt(2 * math.pi))
            tails = 0.5 * math.erfc(-lower / math.sqrt(2)) + 0.5 * math.erfc(upper / math.sqrt(2))
            result[index] += integral + tails
    return result


@pytest.mark.parametrize("sigma", [0.0, 1e-5, 0.05, 0.5, 1.0, 8.0, 8.01, 50.0])
def test_formula_matches_independent_quadrature_at_interior_and_boundaries(sigma):
    commands = np.array([[0.0, 0.25], [0.95, -0.95], [1.0, -1.0], [1.7, -2.0]])
    actual = expected_clipped_action_cost(torch.tensor(commands), sigma).numpy()
    np.testing.assert_allclose(actual, quadrature_cost(commands, sigma), rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("sigma", [0.0, 1e-200, 1e-5, 0.05, 0.5, 8.0, 8.01, 1e100])
def test_gradients_bounds_symmetry_and_dtype(sigma):
    command = torch.tensor([[0.1, -0.2], [1.0, -1.0], [3.0, -4.0]], requires_grad=True)
    value = expected_clipped_action_cost(command, sigma)
    assert value.dtype == command.dtype and value.device == command.device
    assert value.shape == (3,)
    assert torch.isfinite(value).all() and (value >= 0).all() and (value <= 2).all()
    torch.testing.assert_close(value, expected_clipped_action_cost(-command, sigma), rtol=0, atol=0)
    value.sum().backward()
    assert torch.isfinite(command.grad).all()
    assert (command.grad[-1] == 0).all()


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.5, 8.0, 10.0])
def test_gradcheck_away_from_clipping_corners(sigma):
    command = torch.tensor([[0.17, -0.72]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: expected_clipped_action_cost(x, sigma), (command,))


def test_issued_clipping_precedes_noise_and_differs_from_unclipped_gaussian():
    extreme = torch.tensor([[2.0, -2.0]], dtype=torch.float64)
    bounded = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    actual = expected_clipped_action_cost(extreme, 0.05)
    torch.testing.assert_close(actual, expected_clipped_action_cost(bounded, 0.05), rtol=0, atol=0)
    assert actual.item() < 1.93  # Adding noise before initial clipping would be almost exactly two.


@pytest.mark.parametrize("invalid", [-0.1, float("inf"), float("-inf"), float("nan"), None, "bad"])
def test_reject_invalid_noise(invalid):
    with pytest.raises(ValueError, match="noise_std"):
        expected_clipped_action_cost(torch.zeros(2), invalid)
    with pytest.raises(ValueError, match="noise_std"):
        GRUResidualRewardWorldModel(noise_std=invalid)


@pytest.mark.parametrize("invalid", [torch.zeros(3), torch.zeros(2, dtype=torch.int64),
                                   torch.tensor([float("nan"), 0.0]), torch.tensor(0.0)])
def test_reject_invalid_commands(invalid):
    with pytest.raises(ValueError, match="command"):
        expected_clipped_action_cost(invalid, 0.05)


def initialized(model_type, **kwargs):
    torch.manual_seed(121)
    model = model_type(hidden_size=8, **kwargs)
    return model, torch.random.get_rng_state()


def public_packet(batch=2):
    packet = torch.zeros(batch, 8)
    packet[:, :2] = 1.0
    packet[:, 4:6] = torch.tensor([0.15, -0.07])
    packet[:, 6] = 1.0
    return packet


def test_parameter_identity_initialization_and_only_reward_is_changed():
    base, base_rng = initialized(GRUWorldModel)
    residual, residual_rng = initialized(GRUResidualRewardWorldModel)
    disabled, disabled_rng = initialized(GRUResidualRewardWorldModel, residual_reward=False)
    assert base.state_dict().keys() == residual.state_dict().keys() == disabled.state_dict().keys()
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in residual.parameters())
    torch.testing.assert_close(base_rng, residual_rng, rtol=0, atol=0)
    torch.testing.assert_close(base_rng, disabled_rng, rtol=0, atol=0)
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(parameter, dict(residual.named_parameters())[name], rtol=0, atol=0)
    action = torch.tensor([[0.2, 0.4], [-0.5, 0.9]])
    result = []
    for model in (base, residual, disabled):
        state = model.assimilate(model.initial(2), public_packet())
        result.append(model.advance(state, action))
    for state, observation, _ in result[1:]:
        torch.testing.assert_close(observation, result[0][1], rtol=0, atol=0)
        for key in state:
            torch.testing.assert_close(state[key], result[0][0][key], rtol=0, atol=0)
    torch.testing.assert_close(result[2][2], result[0][2], rtol=0, atol=0)
    torch.testing.assert_close(result[1][2], result[0][2] - expected_clipped_action_cost(action, 0.05),
                               rtol=0, atol=0)
    residual.load_state_dict(base.state_dict(), strict=True)


def test_original_total_reward_sequence_loss_requires_no_privileged_inputs():
    model, _ = initialized(GRUResidualRewardWorldModel)
    packets = public_packet(2 * 7).reshape(2, 7, 8)
    packets[:, 2:4, 6] = 0
    packets[:, 2:4, 7] = torch.tensor([0.02, 0.04])
    commands = torch.linspace(-0.7, 0.7, 24).reshape(2, 6, 2)
    rewards = -commands.square().sum(-1) - 0.2
    loss, metrics = sequence_loss(model, packets, commands, rewards)
    loss.backward()
    assert torch.isfinite(loss) and metrics["reward_mse"] > 0
    assert model.reward_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(inspect.signature(sequence_loss).parameters) == {
        "model", "packets", "commands", "rewards", "rollout_horizon", "rollout_weight",
        "reward_scale", "kl_weight", "kl_balance", "free_nats",
    }
    assert set(inspect.signature(model.advance).parameters) == {"state", "action"}
    changed = packets.clone()
    changed[:, 2:4, :4] = 1e6
    repeated_loss, repeated_metrics = sequence_loss(model, changed, commands, rewards)
    torch.testing.assert_close(loss, repeated_loss, rtol=0, atol=0)
    assert metrics == repeated_metrics


def test_disabled_treatment_matches_base_loss_and_gradients():
    base, _ = initialized(GRUWorldModel)
    disabled, _ = initialized(GRUResidualRewardWorldModel, residual_reward=False)
    packets = public_packet(2 * 4).reshape(2, 4, 8)
    commands = torch.full((2, 3, 2), 0.2)
    rewards = torch.full((2, 3), -0.3)
    losses = []
    for model in (base, disabled):
        loss, _ = sequence_loss(model, packets, commands, rewards, rollout_horizon=2)
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(disabled.named_parameters())[name].grad,
                                   rtol=0, atol=0)
