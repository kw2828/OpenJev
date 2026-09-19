"""Kinematic recurrent predictors with an optional rotation of latent memory.

This is an exploratory inductive bias, not a fully equivariant GRU. A standard
coordinate-wise GRU does not implement vector-neuron equivariant nonlinearities.
All state is local to one forecast and has no deployment-time weight updates.
"""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.rigid_motion import so3_exp, so3_log


def rotate(rotation, vector):
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


class PoseTransport(nn.Module):
    """Predict changes in displacement, with a constant-motion output skip.

    R maps body coordinates to world coordinates. dp and w are displacement
    (meters) and spatial rotation increment (radians) per nominal 20 ms step.
    The transported arm groups its 48 hidden scalars into sixteen row vectors.
    It rotates those rows when the body frame changes, including correction
    from an imagined endpoint to the subsequently observed actual endpoint.
    """

    def __init__(self, variant, scales, hidden=48):
        super().__init__()
        if variant not in ("world", "body", "transport", "history16"):
            raise ValueError("unknown pose variant")
        if hidden <= 0 or hidden % 3:
            raise ValueError("hidden must be a positive multiple of three")
        if scales.shape != (4,) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("four positive motion scales required")
        self.variant, self.hidden = variant, hidden
        self.register_buffer("scales", scales.detach().clone())
        self.frame = "world" if variant == "world" else "body"
        self.transport = variant in ("transport", "history16")
        inputs = 15 if self.frame == "world" else 9
        self.observation_update = nn.GRUCell(inputs, hidden, dtype=scales.dtype)
        self.transition = nn.GRUCell(inputs + 40, hidden, dtype=scales.dtype)
        self.readout = nn.Linear(hidden, 6, dtype=scales.dtype)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def initial(self, p, rotation):
        return {"p": p.clone(), "R": rotation.clone(), "previous_p": p.clone(),
                "previous_R": rotation.clone(), "dp": torch.zeros_like(p),
                "w": torch.zeros_like(p), "h": p.new_zeros(len(p), self.hidden)}

    def move_hidden(self, hidden, previous, current):
        if not self.transport:
            return hidden
        transform = previous.transpose(-1, -2) @ current
        return (hidden.reshape(-1, self.hidden // 3, 3) @ transform).reshape(-1, self.hidden)

    def inputs(self, state):
        if self.frame == "world":
            orientation = state["R"].flatten(1)
            dp, w = state["dp"], state["w"]
        else:
            # Unit world vertical, expressed in the current body frame.
            orientation = state["R"][:, 2, :]
            inverse = state["R"].transpose(-1, -2)
            dp, w = rotate(inverse, state["dp"]), rotate(inverse, state["w"])
        return torch.cat((orientation, dp / self.scales[0], w / self.scales[1]), -1)

    def assimilate(self, state, p, rotation):
        dp = p - state["previous_p"]
        w = so3_log(rotation @ state["previous_R"].transpose(-1, -2))
        hidden = self.move_hidden(state["h"], state["R"], rotation)
        actual = {**state, "p": p.clone(), "R": rotation.clone(), "dp": dp, "w": w}
        actual["h"] = self.observation_update(self.inputs(actual), hidden)
        return actual

    def advance(self, state, action):
        hidden = self.transition(torch.cat((self.inputs(state), action), -1), state["h"])
        change = self.readout(hidden)
        dp_change = change[:, :3] * self.scales[2]
        w_change = change[:, 3:] * self.scales[3]
        if self.frame == "body":
            dp_change = rotate(state["R"], dp_change)
            w_change = rotate(state["R"], w_change)
        dp, w = state["dp"] + dp_change, state["w"] + w_change
        p = state["p"] + dp
        rotation = so3_exp(w) @ state["R"]
        return {"p": p, "R": rotation, "previous_p": state["p"],
                "previous_R": state["R"], "dp": dp, "w": w,
                "h": self.move_hidden(hidden, state["R"], rotation)}

    def forecast(self, p, rotation, actions, context=32, horizon=25):
        if p.ndim != 3 or rotation.shape != (*p.shape[:2], 3, 3):
            raise ValueError("invalid pose shapes")
        if context < 16 or p.shape[1] < context or actions.shape != (len(p), context + horizon - 1, 40):
            raise ValueError("invalid context or action shapes")
        start = context - 16 if self.variant == "history16" else 0
        state = self.initial(p[:, start], rotation[:, start])
        state = self.assimilate(state, p[:, start], rotation[:, start])
        for t in range(start, context - 1):
            state = self.advance(state, actions[:, t])
            state = self.assimilate(state, p[:, t + 1], rotation[:, t + 1])
        # Every learned arm begins imagination with exactly the same CV16 skip.
        state = {**state, "dp": (p[:, context - 1] - p[:, context - 16]) / 15,
                 "w": so3_log(rotation[:, context - 1] @ rotation[:, context - 16].transpose(-1, -2)) / 15}
        positions, rotations = [], []
        for t in range(context - 1, context + horizon - 1):
            state = self.advance(state, actions[:, t])
            positions.append(state["p"])
            rotations.append(state["R"])
        return torch.stack(positions, 1), torch.stack(rotations, 1)


def training_scales(p, rotation):
    """One scalar per vector, estimated solely from training windows."""
    dp = p[:, 1:] - p[:, :-1]
    w = so3_log(rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2))
    arrays = (dp, w, dp[:, 1:] - dp[:, :-1], w[:, 1:] - w[:, :-1])
    return torch.stack([x.square().mean().sqrt().clamp_min(1e-5) for x in arrays])
