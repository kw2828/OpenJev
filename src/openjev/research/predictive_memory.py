"""Recurrent PPO with optional action-conditioned predictive auxiliary learning.

This is deterministic predictive state learning, not a Dreamer implementation.
No imagined transition is used as an environment sample for policy optimization.
"""

import torch
from torch import nn
from torch.nn import functional as F


class PredictiveMemory(nn.Module):
    def __init__(self, observation_dim=984, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh())
        self.memory = nn.GRUCell(hidden + 7, hidden)
        self.actor = nn.Linear(hidden, 7)
        self.value = nn.Linear(hidden, 1)
        self.dynamics = nn.GRUCell(7, hidden)
        self.observation = nn.Linear(hidden, observation_dim)
        self.reward = nn.Linear(hidden, 1)
        self.termination = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.value.weight, 1.)
        nn.init.zeros_(self.value.bias)

    def observe(self, observations, previous_actions, state, reset, current_only=False):
        state = state * (~reset).float().unsqueeze(-1)
        if current_only:
            state = torch.zeros_like(state)
        actions = F.one_hot(previous_actions.long(), 7).float()
        actions = actions * (~reset).float().unsqueeze(-1)
        state = self.memory(torch.cat([self.encoder(observations), actions], -1), state)
        return self.actor(state), self.value(state).squeeze(-1), state

    def sequence(self, observations, previous_actions, state, resets, current_only=False):
        logits, values, states = [], [], []
        for obs, act, reset in zip(observations, previous_actions, resets, strict=True):
            logit, value, state = self.observe(obs, act, state, reset, current_only)
            logits.append(logit)
            values.append(value)
            states.append(state)
        return torch.stack(logits), torch.stack(values), torch.stack(states)

    def imagine(self, state, actions):
        return self.dynamics(F.one_hot(actions.long(), 7).float(), state)


def observation_loss(logits, targets):
    """Equal weight for object/color/state/direction categorical predictions."""
    cells = (logits.shape[-1] - 4) // 20
    predicted = logits[..., :-4].reshape(-1, cells, 20)
    target = targets[..., :-4].reshape(-1, cells, 20)
    losses = []
    for lo, hi in [(0, 11), (11, 17), (17, 20)]:
        losses.append(F.cross_entropy(predicted[..., lo:hi].reshape(-1, hi-lo),
                                      target[..., lo:hi].argmax(-1).reshape(-1)))
    losses.append(F.cross_entropy(logits[..., -4:].reshape(-1, 4),
                                  targets[..., -4:].argmax(-1).reshape(-1)))
    return torch.stack(losses).mean()


def auxiliary_loss(model, states, actions, next_obs, rewards, terminated, ended, full,
                   horizons=(1, 2, 4)):
    """Predict up to four steps without observing intervening frames.

    Starts are all causal posterior states. Mask trajectories after the first
    terminal/truncated transition; retain that transition's true observation.
    Future posterior targets are detached and contain no optimizer gradient.
    Observation decoding anchors the full arm against latent collapse.
    """
    length, batch, width = states.shape
    with torch.no_grad():
        _, _, target_states = model.observe(
            next_obs.reshape(-1, next_obs.shape[-1]), actions.reshape(-1),
            states.detach().reshape(-1, width),
            torch.zeros(length * batch, dtype=torch.bool, device=states.device))
        target_states = target_states.reshape(length, batch, width)
    imagined = states
    valid = torch.ones((length, batch), dtype=torch.bool, device=states.device)
    objectives, measurements = [], []
    for step in range(1, max(horizons)+1):
        n = length-step+1
        imagined = model.imagine(imagined[:n].reshape(-1, width),
                                 actions[step-1:].reshape(-1)).reshape(n, batch, width)
        valid = valid[:n]
        if step > 1:
            valid = valid & ~ended[step-2:length-1]
        if step not in horizons or not valid.any():
            continue
        pred = imagined[valid]
        rt = rewards[step-1:][valid]
        tt = terminated[step-1:][valid].float()
        reward_loss = F.mse_loss(model.reward(pred).squeeze(-1), rt)
        done_loss = F.binary_cross_entropy_with_logits(model.termination(pred).squeeze(-1), tt)
        loss = reward_loss + done_loss
        if full:
            latent = F.mse_loss(pred, target_states[step-1:][valid])
            obs_loss = observation_loss(model.observation(pred), next_obs[step-1:][valid])
            loss = loss + latent + obs_loss
        else:
            latent = loss.new_tensor(0.)
            obs_loss = loss.new_tensor(0.)
        objectives.append(loss)
        measurements.append(torch.stack([reward_loss, done_loss, latent, obs_loss]).detach())
    if not objectives:
        raise ValueError('At least one valid one-step transition is required')
    return torch.stack(objectives).mean(), torch.stack(measurements).mean(0)


def advantages(rewards, values, next_values, terminated, ended, gamma=.99, lam=.95):
    """Bootstrap time limits using terminal observations, never across episodes."""
    output = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).float() * next_values[t] - values[t]
        carry = delta + gamma * lam * (~ended[t]).float() * carry
        output[t] = carry
    return output, output + values
