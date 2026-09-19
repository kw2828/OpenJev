"""Conventional small world models for public, intermittently sensed Reacher data.

These are a finite-history MLP, a deterministic GRU, and a Gaussian recurrent
state-space model. They are not Dreamer reproductions or connectome models.
Only eight-dimensional public packets, issued actions, and executed rewards
enter training. No joint velocity, simulator state, or privileged next state is
accepted. Reacher-v5 computes its native reward after the physics transition.

Packet order: cos(q0), cos(q1), sin(q0), sin(q1), target x, target y,
measurement valid, seconds since the last valid measurement. Missing external
angular entries are discarded. The autoregressive history baseline retains
its own predicted angles as estimates with valid=0. Imagination never marks
predictions as valid measurements. All state operations return fresh mappings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

State = dict[str, Tensor]
PACKET_SIZE = 8
ACTION_SIZE = 2
ANGLE_SIZE = 4


def repeat_index(state: Mapping[str, Tensor], indices: Tensor) -> State:
    """Select/repeat independent batch states for a candidate-action bank."""
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("indices must be a one-dimensional long tensor")
    return {key: value.index_select(0, indices) for key, value in state.items()}


def _sanitize(packet: Tensor) -> Tensor:
    angles = torch.where(packet[..., 6:7] > 0.5, packet[..., :4], 0.0)
    return torch.cat((angles, packet[..., 4:]), dim=-1)


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ELU(), nn.Linear(hidden, outputs))


class PublicWorldModel(nn.Module):
    """Explicit episode state; eval mode always uses Gaussian means."""

    def __init__(self, dt: float = 0.02) -> None:
        super().__init__()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.dt = float(dt)

    def _zeros(self, batch: int, width: int, device=None) -> Tensor:
        parameter = next(self.parameters())
        if not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        return torch.zeros(
            batch, width, device=parameter.device if device is None else device,
            dtype=parameter.dtype,
        )

    def _next_packet(self, packet: Tensor, angles: Tensor) -> Tensor:
        return torch.cat(
            (angles, packet[:, 4:6], torch.zeros_like(packet[:, 6:7]),
             packet[:, 7:8] + self.dt), dim=-1,
        )

    def initial(self, batch: int, device=None) -> State:
        raise NotImplementedError

    def assimilate(self, state: State, packet: Tensor) -> State:
        raise NotImplementedError

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        raise NotImplementedError

    def posterior_kl(self, state: State, balance: float, free_nats: float) -> Tensor:
        return state["packet"].new_zeros(state["packet"].shape[0])


class HistoryWorldModel(PublicWorldModel):
    """Autoregressive packet/action window with a two-layer MLP.

    The newest slot contains the current packet and then the issued action.
    Each imagined step appends its angle estimate with valid=0. Missing public
    packets preserve this estimate; valid measurements overwrite it. These
    estimates can retain older information, so the finite input window is not
    a claim of strictly bounded effective memory. Measurement/estimate flags
    remain explicit, and external missing-angle placeholders are never used.
    """

    def __init__(self, window: int = 12, width: int = 128, dt: float = 0.02) -> None:
        super().__init__(dt)
        if not isinstance(window, int) or window < 1 or width < 1:
            raise ValueError("window and width must be positive integers")
        self.window = window
        self.encoder = nn.Sequential(
            nn.Linear(window * (PACKET_SIZE + ACTION_SIZE), width), nn.ELU(),
            nn.Linear(width, width), nn.ELU(),
        )
        self.observation_head = nn.Linear(width, ANGLE_SIZE)
        self.reward_head = nn.Linear(width, 1)

    def initial(self, batch: int, device=None) -> State:
        packet = self._zeros(batch, PACKET_SIZE, device)
        return {
            "packet": packet,
            "history": packet.new_zeros(batch, self.window, PACKET_SIZE + ACTION_SIZE),
        }

    def assimilate(self, state: State, packet: Tensor) -> State:
        packet = _sanitize(packet)
        angles = torch.where(packet[:, 6:7] > 0.5, packet[:, :4], state["packet"][:, :4])
        packet = torch.cat((angles, packet[:, 4:]), -1)
        current = torch.cat((packet, state["history"][:, -1, PACKET_SIZE:]), dim=-1)
        return {
            "packet": packet,
            "history": torch.cat((state["history"][:, :-1], current[:, None]), dim=1),
        }

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        current = torch.cat((state["packet"], action), dim=-1)
        history = torch.cat((state["history"][:, :-1], current[:, None]), dim=1)
        features = self.encoder(history.flatten(1))
        angles = self.observation_head(features)
        reward = self.reward_head(features).squeeze(-1)
        packet = self._next_packet(state["packet"], angles)
        next_pair = torch.cat((packet, torch.zeros_like(action)), dim=-1)
        next_state = {
            "packet": packet,
            "history": torch.cat((history[:, 1:], next_pair[:, None]), dim=1),
        }
        return next_state, angles, reward


class GRUWorldModel(PublicWorldModel):
    """Separate valid-measurement updates and action-conditioned transitions."""

    def __init__(self, hidden_size: int = 64, dt: float = 0.02) -> None:
        super().__init__(dt)
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.observation_update = nn.GRUCell(PACKET_SIZE, hidden_size)
        self.transition = nn.GRUCell(ACTION_SIZE + 4, hidden_size)
        self.observation_head = _mlp(hidden_size, hidden_size, ANGLE_SIZE)
        self.reward_head = _mlp(hidden_size + ACTION_SIZE, hidden_size, 1)

    def initial(self, batch: int, device=None) -> State:
        return {
            "packet": self._zeros(batch, PACKET_SIZE, device),
            "hidden": self._zeros(batch, self.hidden_size, device),
        }

    def assimilate(self, state: State, packet: Tensor) -> State:
        packet = _sanitize(packet)
        candidate = self.observation_update(packet, state["hidden"])
        hidden = torch.where(packet[:, 6:7] > 0.5, candidate, state["hidden"])
        return {"packet": packet, "hidden": hidden}

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        hidden = self.transition(torch.cat((action, state["packet"][:, 4:]), -1),
                                 state["hidden"])
        angles = self.observation_head(hidden)
        reward = self.reward_head(torch.cat((hidden, action), -1)).squeeze(-1)
        return {"packet": self._next_packet(state["packet"], angles),
                "hidden": hidden}, angles, reward


class GaussianRSSM(PublicWorldModel):
    """Small diagonal-Gaussian RSSM, not an implementation of a full RL agent.

    During training both priors and posteriors are reparameterized samples;
    eval uses their means. Missing observations leave the current prior intact.
    KL balance weights the detached-posterior dynamics KL by ``balance`` and
    the detached-prior representation KL by ``1 - balance``. Free nats apply
    to each whole stochastic vector before masking missing posteriors.
    """

    def __init__(self, hidden_size: int = 64, stochastic_size: int = 16,
                 dt: float = 0.02) -> None:
        super().__init__(dt)
        if hidden_size < 1 or stochastic_size < 1:
            raise ValueError("hidden and stochastic sizes must be positive")
        self.hidden_size = hidden_size
        self.stochastic_size = stochastic_size
        self.transition = nn.GRUCell(stochastic_size + ACTION_SIZE + 4, hidden_size)
        self.prior = _mlp(hidden_size, hidden_size, 2 * stochastic_size)
        self.posterior = _mlp(hidden_size + PACKET_SIZE, hidden_size, 2 * stochastic_size)
        self.observation_head = _mlp(hidden_size + stochastic_size, hidden_size, ANGLE_SIZE)
        self.reward_head = _mlp(hidden_size + stochastic_size + ACTION_SIZE, hidden_size, 1)

    @staticmethod
    def _distribution(raw: Tensor) -> tuple[Tensor, Tensor]:
        mean, scale = raw.chunk(2, dim=-1)
        return mean, F.softplus(scale) + 0.1

    def _sample(self, mean: Tensor, scale: Tensor) -> Tensor:
        return mean + scale * torch.randn_like(mean) if self.training else mean

    def initial(self, batch: int, device=None) -> State:
        stochastic = self._zeros(batch, self.stochastic_size, device)
        return {
            "packet": self._zeros(batch, PACKET_SIZE, device),
            "hidden": self._zeros(batch, self.hidden_size, device),
            "stochastic": stochastic,
            "prior_mean": stochastic.clone(), "prior_scale": torch.ones_like(stochastic),
            "posterior_mean": stochastic.clone(),
            "posterior_scale": torch.ones_like(stochastic),
            "posterior_valid": stochastic.new_zeros(batch, 1),
        }

    def assimilate(self, state: State, packet: Tensor) -> State:
        packet = _sanitize(packet)
        mean, scale = self._distribution(self.posterior(torch.cat((state["hidden"], packet), -1)))
        valid = packet[:, 6:7] > 0.5
        return {
            **state, "packet": packet,
            "stochastic": torch.where(valid, self._sample(mean, scale), state["stochastic"]),
            "posterior_mean": torch.where(valid, mean, state["prior_mean"]),
            "posterior_scale": torch.where(valid, scale, state["prior_scale"]),
            "posterior_valid": valid.to(packet.dtype),
        }

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        transition_input = torch.cat((state["stochastic"], action, state["packet"][:, 4:]), -1)
        hidden = self.transition(transition_input, state["hidden"])
        mean, scale = self._distribution(self.prior(hidden))
        stochastic = self._sample(mean, scale)
        features = torch.cat((hidden, stochastic), -1)
        angles = self.observation_head(features)
        reward = self.reward_head(torch.cat((features, action), -1)).squeeze(-1)
        return {
            "packet": self._next_packet(state["packet"], angles), "hidden": hidden,
            "stochastic": stochastic, "prior_mean": mean, "prior_scale": scale,
            "posterior_mean": mean, "posterior_scale": scale,
            "posterior_valid": state["posterior_valid"].new_zeros(hidden.shape[0], 1),
        }, angles, reward

    def posterior_kl(self, state: State, balance: float = 0.8,
                     free_nats: float = 1.0) -> Tensor:
        if not 0 <= balance <= 1 or not math.isfinite(free_nats) or free_nats < 0:
            raise ValueError("invalid KL balance or free nats")

        def normal_kl(qm, qs, pm, ps):
            return (torch.log(ps / qs) + (qs.square() + (qm - pm).square())
                    / (2 * ps.square()) - 0.5).sum(-1)

        qm, qs = state["posterior_mean"], state["posterior_scale"]
        pm, ps = state["prior_mean"], state["prior_scale"]
        dynamics = normal_kl(qm.detach(), qs.detach(), pm, ps).clamp_min(free_nats)
        representation = normal_kl(qm, qs, pm.detach(), ps.detach()).clamp_min(free_nats)
        return (balance * dynamics + (1 - balance) * representation) * state["posterior_valid"][:, 0]


def make_world_model(kind: str, *, hidden_size: int = 64, stochastic_size: int = 16,
                     window: int = 12, width: int = 128, dt: float = 0.02) -> PublicWorldModel:
    if kind == "history":
        return HistoryWorldModel(window=window, width=width, dt=dt)
    if kind == "gru":
        return GRUWorldModel(hidden_size=hidden_size, dt=dt)
    if kind == "rssm":
        return GaussianRSSM(hidden_size=hidden_size, stochastic_size=stochastic_size, dt=dt)
    raise ValueError(f"unknown world model: {kind}")


def _validate_sequence(packets: Tensor, commands: Tensor, rewards: Tensor) -> None:
    if commands.ndim != 3 or commands.shape[-1] != ACTION_SIZE:
        raise ValueError("commands must have shape [batch, time, 2]")
    batch, steps, _ = commands.shape
    if batch < 1 or steps < 1 or packets.shape != (batch, steps + 1, PACKET_SIZE):
        raise ValueError("packets must have shape [batch, time + 1, 8]")
    if rewards.shape != (batch, steps):
        raise ValueError("rewards must have shape [batch, time]")
    if not all(value.is_floating_point() for value in (packets, commands, rewards)):
        raise ValueError("sequences must use floating point tensors")
    if not all(torch.isfinite(value).all() for value in (packets, commands, rewards)):
        raise ValueError("sequences must be finite")
    if not torch.all((packets[..., 6] == 0) | (packets[..., 6] == 1)):
        raise ValueError("measurement validity must be binary")
    if torch.any(packets[..., 7] < 0):
        raise ValueError("measurement age must be nonnegative")


def _masked_mean(square_error: Tensor, mask: Tensor) -> Tensor:
    """Zero loss with a defined zero gradient when all targets are absent."""
    mask = torch.broadcast_to(mask, square_error.shape).to(square_error.dtype)
    return (square_error * mask).sum() / mask.sum().clamp_min(1)


def sequence_loss(model: PublicWorldModel, packets: Tensor, commands: Tensor,
                  rewards: Tensor, *, rollout_horizon: int = 5,
                  rollout_weight: float = 0.5, reward_scale: float = 4.0,
                  kl_weight: float = 0.01, kl_balance: float = 0.8,
                  free_nats: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    """Causal one-step fitting plus open-loop loss from valid observation starts.

    All complete ``rollout_horizon`` windows are included. Their root states
    contain only each root's observation prefix; future measurements are used
    only as masked targets, never assimilated in the rollout. Rewards are
    supervised at every executed step. The reward scale multiplies MSE, rather
    than scaling the targets. KL is averaged over valid posterior updates only.
    """
    _validate_sequence(packets, commands, rewards)
    if not isinstance(rollout_horizon, int) or rollout_horizon < 1:
        raise ValueError("rollout_horizon must be a positive integer")
    if any(not math.isfinite(v) or v < 0 for v in (rollout_weight, reward_scale, kl_weight)):
        raise ValueError("loss weights must be nonnegative and finite")
    if not 0 <= kl_balance <= 1 or not math.isfinite(free_nats) or free_nats < 0:
        raise ValueError("invalid KL balance or free nats")
    batch, steps, _ = commands.shape
    state = model.initial(batch, packets.device)
    roots, angles, predicted_rewards, kls = [], [], [], []
    for step in range(steps):
        state = model.assimilate(state, packets[:, step])
        roots.append(state)
        kls.append(model.posterior_kl(state, kl_balance, free_nats))
        state, predicted_angles, predicted_reward = model.advance(state, commands[:, step])
        angles.append(predicted_angles)
        predicted_rewards.append(predicted_reward)
    observation_loss = _masked_mean(
        (torch.stack(angles, 1) - packets[:, 1:, :ANGLE_SIZE]).square(),
        packets[:, 1:, 6:7],
    )
    reward_loss = F.mse_loss(torch.stack(predicted_rewards, 1), rewards)
    valid_posteriors = packets[:, :-1, 6]
    kl_loss = torch.stack(kls, 1).sum() / valid_posteriors.sum().clamp_min(1)
    zero = observation_loss * 0
    rollout_observation, rollout_reward = zero, zero
    starts = steps - rollout_horizon + 1
    if rollout_weight and starts > 0:
        imagined = {
            key: torch.stack([root[key] for root in roots[:starts]], 1).reshape(
                batch * starts, *roots[0][key].shape[1:],
            )
            for key in roots[0]
        }
        predicted_angles, predicted_rewards, target_angles, target_rewards, masks = [], [], [], [], []
        prefix_valid = packets[:, :starts, 6].reshape(-1)
        for offset in range(rollout_horizon):
            action = commands[:, offset:offset + starts].reshape(batch * starts, ACTION_SIZE)
            imagined, observation, reward = model.advance(imagined, action)
            predicted_angles.append(observation)
            predicted_rewards.append(reward)
            target_angles.append(packets[:, offset + 1:offset + starts + 1, :ANGLE_SIZE].reshape(-1, ANGLE_SIZE))
            target_rewards.append(rewards[:, offset:offset + starts].reshape(-1))
            masks.append(packets[:, offset + 1:offset + starts + 1, 6].reshape(-1) * prefix_valid)
        rollout_observation = _masked_mean(
            (torch.stack(predicted_angles, 1) - torch.stack(target_angles, 1)).square(),
            torch.stack(masks, 1)[..., None],
        )
        rollout_reward = _masked_mean(
            (torch.stack(predicted_rewards, 1) - torch.stack(target_rewards, 1)).square(),
            prefix_valid[:, None],
        )
    loss = observation_loss + reward_scale * reward_loss + kl_weight * kl_loss
    loss = loss + rollout_weight * (rollout_observation + reward_scale * rollout_reward)
    metrics = {
        "loss": float(loss.detach()), "observation_mse": float(observation_loss.detach()),
        "reward_mse": float(reward_loss.detach()), "kl_nats": float(kl_loss.detach()),
        "rollout_observation_mse": float(rollout_observation.detach()),
        "rollout_reward_mse": float(rollout_reward.detach()),
        "valid_observation_targets": float(packets[:, 1:, 6].sum()),
        "valid_rollout_starts": float(packets[:, :max(starts, 0), 6].sum()) if starts > 0 else 0.0,
    }
    return loss, metrics
