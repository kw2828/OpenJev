"""Fabricated invariants for policy initialization and group-relative learning."""
import numpy as np
import pytest
import torch

from openjev.research.retention_policy import RetentionPolicy, leave_one_out_advantage, select


def test_initial_policy_is_analytic_and_budgeted():
    torch.manual_seed(3)
    model = RetentionPolicy()
    statistics = {'features': np.zeros((2, 3, 6), np.float64),
                  'kl': np.array([[.5, .1, .2], [.2, .2, .5]])}
    actions, _, _ = select(model, statistics, False)
    assert actions.tolist() == [1, 0]
    assert model.parameter_bytes == 264
    assert sum(p.numel() for p in model.parameters()) == 33


def test_leave_one_out_excludes_own_reward():
    rewards = torch.tensor([[.0, .1, .4]], dtype=torch.float64, requires_grad=True)
    advantages = leave_one_out_advantage(rewards)
    assert not advantages.requires_grad
    np.testing.assert_allclose(advantages.numpy(), [[-5., -2., 7.]])
    baseline = rewards.detach()-.05*advantages
    assert baseline[0, 1] == .2


def test_policy_gradient_can_change_deletion_probabilities():
    torch.manual_seed(7)
    model = RetentionPolicy()
    features = torch.tensor([[[0., 0., 1., 0., .2, .5], [0., 1., -1., .1, .2, .5]]], dtype=torch.float64)
    kl = torch.full((1, 2), .2, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = model(features, kl).detach()
    # A fabricated favorable deletion changes the policy without GP/data calls.
    loss = -torch.log_softmax(-model(features, kl), -1)[0, 0]
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    after = model(features, kl).detach()
    assert after[0, 0]-after[0, 1] < before[0, 0]-before[0, 1]
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize('reward', [torch.zeros(4), torch.zeros(2, 1), torch.tensor([[float('nan'), 0.]])])
def test_invalid_group_rewards_fail(reward):
    with pytest.raises(ValueError):
        leave_one_out_advantage(reward)
