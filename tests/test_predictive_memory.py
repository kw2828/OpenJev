import pytest
import torch
from torch.nn import functional as F

from openjev.research.predictive_memory import (
    PredictiveMemory,
    advantages,
    auxiliary_loss,
    observation_loss,
)


def encoded_observations(length=4, batch=2, cells=2):
    channels = [F.one_hot(torch.randint(n, (length, batch, cells)), n)
                for n in (11, 6, 3)]
    image = torch.cat(channels, -1).reshape(length, batch, cells * 20)
    direction = F.one_hot(torch.randint(4, (length, batch)), 4)
    return torch.cat([image, direction], -1).float()


def fixture():
    torch.manual_seed(717)
    observations = encoded_observations()
    model = PredictiveMemory(observation_dim=observations.shape[-1], hidden=8)
    actions = torch.randint(7, (4, 2))
    resets = torch.tensor([[True, False], [False, False], [False, True], [True, False]])
    return model, observations, actions, resets


@pytest.mark.parametrize('current_only', [False, True])
def test_sequence_replay_matches_rollout_including_internal_resets(current_only):
    model, observations, previous_actions, resets = fixture()
    initial = torch.randn(2, model.hidden)
    state = initial.clone()
    step_logits, step_values, step_states = [], [], []
    with torch.no_grad():
        for t in range(len(observations)):
            logits, values, state = model.observe(
                observations[t], previous_actions[t], state, resets[t], current_only)
            step_logits.append(logits)
            step_values.append(values)
            step_states.append(state)
        replay = model.sequence(observations, previous_actions, initial, resets, current_only)
    for actual, expected in zip(replay, (step_logits, step_values, step_states), strict=True):
        torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    selected_actions = torch.tensor([[0, 2], [1, 4], [6, 0], [3, 5]])
    rollout_logprobs = torch.distributions.Categorical(logits=torch.stack(step_logits)).log_prob(
        selected_actions)
    replay_logprobs = torch.distributions.Categorical(logits=replay[0]).log_prob(selected_actions)
    torch.testing.assert_close(replay_logprobs, rollout_logprobs, rtol=0, atol=0)
    for t, env in ((2, 1), (3, 0)):
        fresh = model.observe(observations[t, env:env+1], previous_actions[t, env:env+1],
                              torch.zeros(1, model.hidden), torch.ones(1, dtype=torch.bool),
                              current_only)
        torch.testing.assert_close(replay[0][t, env], fresh[0][0])


def test_future_observations_and_actions_cannot_change_prefix_outputs():
    model, observations, previous_actions, resets = fixture()
    initial = torch.randn(2, model.hidden)
    reference = model.sequence(observations, previous_actions, initial, resets)
    changed_obs, changed_actions = observations.clone(), previous_actions.clone()
    changed_obs[2:] = torch.randn_like(changed_obs[2:]) * 20
    changed_actions[2:] = (changed_actions[2:] + 1) % 7
    changed = model.sequence(changed_obs, changed_actions, initial, resets)
    for a, b in zip(reference, changed, strict=True):
        torch.testing.assert_close(a[:2], b[:2], rtol=0, atol=0)


def test_current_only_ignores_prior_state_and_reset_masks_previous_action():
    model, observations, actions, _ = fixture()
    reset = torch.zeros(2, dtype=torch.bool)
    a = model.observe(observations[0], actions[0], torch.randn(2, model.hidden), reset, True)
    b = model.observe(observations[0], actions[0], torch.randn(2, model.hidden) * 100, reset, True)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    reset[:] = True
    a = model.observe(observations[0], actions[0], torch.randn(2, model.hidden), reset)
    b = model.observe(observations[0], (actions[0] + 3) % 7,
                      torch.randn(2, model.hidden), reset)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_gae_bootstraps_truncation_but_never_carries_advantage_across_resets():
    rewards = torch.tensor([[1., 2.], [100., 200.], [3., 4.]])
    values = torch.tensor([[.2, .4], [.5, .6], [.7, .8]])
    next_values = torch.tensor([[4., 999.], [7., 8.], [9., 10.]])
    terminated = torch.tensor([[False, True], [False, False], [False, True]])
    ended = torch.tensor([[True, True], [False, False], [False, True]])
    result, returns = advantages(rewards, values, next_values, terminated, ended,
                                  gamma=.9, lam=.8)
    final = torch.tensor([3. + .9 * 9. - .7, 4. - .8])
    middle = torch.tensor([100. + .9 * 7. - .5, 200. + .9 * 8. - .6]) + .9 * .8 * final
    expected = torch.stack([torch.tensor([1. + .9 * 4. - .2, 2. - .4]), middle, final])
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(returns, expected + values)
    changed_rewards = rewards.clone()
    changed_rewards[1:] += 10000
    changed, _ = advantages(changed_rewards, values, next_values, terminated, ended,
                             gamma=.9, lam=.8)
    torch.testing.assert_close(changed[0], result[0], rtol=0, atol=0)


def manual_prediction_losses(model, states, actions, next_obs, rewards, terminated, ended, full):
    """Independent start-by-start rollouts check target and episode-boundary indexing."""
    per_horizon = []
    for horizon in (1, 2, 4):
        predicted, reward_targets, done_targets, latent_targets, obs_targets = [], [], [], [], []
        for start in range(len(states) - horizon + 1):
            last = start + horizon - 1
            for env in range(states.shape[1]):
                # The transition that ends an episode is valid; subsequent ones are not.
                if ended[start:last, env].any():
                    continue
                state = states[start, env:env+1]
                for t in range(start, last + 1):
                    state = model.imagine(state, actions[t, env:env+1])
                predicted.append(state)
                reward_targets.append(rewards[last, env:env+1])
                done_targets.append(terminated[last, env:env+1])
                obs_targets.append(next_obs[last, env:env+1])
                with torch.no_grad():
                    target = model.observe(next_obs[last, env:env+1], actions[last, env:env+1],
                                           states[last, env:env+1],
                                           torch.zeros(1, dtype=torch.bool))[2]
                latent_targets.append(target)
        if not predicted:
            continue
        prediction = torch.cat(predicted)
        losses = [F.mse_loss(model.reward(prediction).squeeze(-1), torch.cat(reward_targets)),
                  F.binary_cross_entropy_with_logits(model.termination(prediction).squeeze(-1),
                                                     torch.cat(done_targets).float())]
        if full:
            losses += [F.mse_loss(prediction, torch.cat(latent_targets)),
                       observation_loss(model.observation(prediction), torch.cat(obs_targets))]
        else:
            losses += [prediction.new_tensor(0.), prediction.new_tensor(0.)]
        per_horizon.append(torch.stack(losses))
    return torch.stack(per_horizon).mean(0)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('with_boundaries', [False, True])
def test_auxiliary_rollouts_match_explicit_valid_paths_across_termination_and_truncation(
        full, with_boundaries):
    model, next_obs, actions, _ = fixture()
    states = torch.randn(4, 2, model.hidden)
    rewards = torch.tensor([[.2, 0.], [0., .3], [.4, 0.], [0., .8]])
    # Environment zero terminates at t=0; environment one truncates at t=1.
    terminated = torch.tensor([[True, False], [False, False], [False, False], [False, True]])
    ended = torch.tensor([[True, False], [False, True], [False, False], [False, True]])
    if not with_boundaries:
        terminated[0, 0] = False
        ended[:3] = False
    actual, measurements = auxiliary_loss(model, states, actions, next_obs, rewards,
                                          terminated, ended, full)
    expected = manual_prediction_losses(model, states, actions, next_obs, rewards,
                                         terminated, ended, full)
    torch.testing.assert_close(measurements, expected.detach())
    torch.testing.assert_close(actual, expected.sum())


def test_auxiliary_future_targets_are_detached_but_transition_receives_finite_gradients():
    model, next_obs, actions, _ = fixture()
    states = torch.randn(4, 2, model.hidden, requires_grad=True)
    next_obs.requires_grad_()
    terminal = torch.zeros(4, 2, dtype=torch.bool)
    loss, components = auxiliary_loss(model, states, actions, next_obs,
                                       torch.rand(4, 2), terminal, terminal, True)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(components).all()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert states.grad.abs().sum() > 0
    assert next_obs.grad is None
    # Encoder and posterior GRU are used only in the detached target branch here.
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.memory.parameters())
    for module in (model.dynamics, model.observation, model.reward, model.termination):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0


def test_categorical_observation_loss_prefers_correct_native_one_hot_channels():
    torch.manual_seed(990)
    targets = encoded_observations(length=2, batch=2, cells=49)
    correct = targets * 16 - 8
    wrong = -correct
    correct_loss = observation_loss(correct, targets)
    wrong_loss = observation_loss(wrong, targets)
    assert targets.shape[-1] == 984
    assert correct_loss < 1e-5
    assert wrong_loss > 10
    # A separate direction group gets equal weight to each entire image channel.
    wrong_direction = correct.clone()
    wrong_direction[..., -4:] = wrong[..., -4:]
    expected_delta = (F.cross_entropy(wrong[..., -4:].reshape(-1, 4),
                                     targets[..., -4:].argmax(-1).reshape(-1))
                      - F.cross_entropy(correct[..., -4:].reshape(-1, 4),
                                        targets[..., -4:].argmax(-1).reshape(-1))) / 4
    torch.testing.assert_close(observation_loss(wrong_direction, targets) - correct_loss, expected_delta)
