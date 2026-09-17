import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('stable_baselines3')
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from openjev.research.self_reference import (
    DynamicsEncoder,
    group_advantages,
    policy_loss,
    reference_rewards,
    success_centers,
)


def test_failure_reward_decreases_with_distance_and_never_beats_success():
    rewards = reference_rewards([[0.], [.1], [1.], [10.]], [True, False, False, False])
    assert rewards[0] == 1
    assert 0 < rewards[3] < rewards[2] < rewards[1] < .6


def test_no_success_cannot_borrow_reference_from_another_call():
    reference_rewards([[0.], [1.]], [True, False])
    assert np.array_equal(reference_rewards([[0.], [1.]], [False, False]), [0., 0.])
    assert np.array_equal(group_advantages([1., 1.]), [0., 0.])
    assert np.array_equal(group_advantages([0., 0.]), [0., 0.])


def test_single_failure_or_distance_ties_are_finite():
    assert np.allclose(reference_rewards([[0.], [2.]], [True, False]), [1., .3])
    assert np.allclose(reference_rewards([[0.], [2.], [-2.]], [True, False, False]), [1., .3, .3])
    with pytest.raises(ValueError):
        reference_rewards([[np.nan], [0.]], [True, False])


def test_dbscan_noise_excluded_and_all_noise_falls_back_to_mean():
    centers = success_centers([[0.], [.01], [10.]])
    assert np.allclose(centers, [[.005]])
    assert np.allclose(success_centers([[0.], [10.]]), [[5.]])
    assert np.allclose(success_centers([[3.]]), [[3.]])


def test_encoder_shape_and_freeze():
    torch.manual_seed(1)
    encoder = DynamicsEncoder()
    observations = torch.randn(6, 6)
    assert encoder(observations, torch.zeros(6)).shape == (6, 6)
    embedded = encoder.trajectory(observations.numpy())
    assert embedded.shape == (16,)
    assert np.isfinite(embedded).all()


def test_actor_loss_weights_trajectories_equally_and_penalizes_kl():
    torch.manual_seed(3)
    policy = ActorCriticPolicy(spaces.Box(-10., 10., shape=(18,), dtype=np.float32),
                              spaces.Discrete(2), lr_schedule=lambda _: .001,
                              net_arch=[64, 64])
    states = torch.randn(4, 18)
    actions = torch.tensor([0, 1, 0, 1])
    with torch.no_grad():
        dist = policy.get_distribution(states).distribution
        old, ref = dist.log_prob(actions), dist.probs.clone()
    # One positive-action trajectory, one three-action negative trajectory.
    advantage = torch.tensor([1., -1., -1., -1.])
    loss, metrics = policy_loss(policy, states, actions, old, ref, advantage, [1, 3],
                                kl_weight=0., entropy_weight=0.)
    assert float(loss.detach()) == pytest.approx(0., abs=1e-6)
    assert metrics['kl'] == pytest.approx(0., abs=1e-6)
    altered_ref = torch.tensor([[.999, .001]]).repeat(4, 1)
    penalized, _ = policy_loss(policy, states, actions, old, altered_ref, advantage,
                              [1, 3], kl_weight=1., entropy_weight=0.)
    assert penalized > loss
    penalized.backward()
    assert all(p.grad is None for p in policy.value_net.parameters())
    assert all(p.grad is None for p in policy.mlp_extractor.value_net.parameters())
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
