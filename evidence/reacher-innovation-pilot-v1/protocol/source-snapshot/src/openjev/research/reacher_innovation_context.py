"""Isolated conventional fast/slow GRU with four slow-correction gates.

This is an untrained architecture prototype, not a novel or biological model,
calibrated filter or registered study arm. All variants instantiate and execute
the same modules; only gate-feature pathways differ. Slow context changes only
on real visible innovations after startup. Action-only imagination never writes
context. Public packet/action inputs have no simulator-state channel.

The four bounded positive outputs are uncalibrated predictive per-coordinate
residual second moments, not a joint covariance on the cosine/sine manifold.
They can include mean bias. Constructor bounds and eta are provisional, not
tuned scientific settings. Proper visible-target supervision and bound-hit
diagnostics belong in a future external objective. No loss is supplied here.
Variance-head inputs are detached; scalar gate features are also detached.
Thus variance supervision cannot directly change the mean/reward backbone, and
downstream loss cannot inflate variance to suppress the correction gate.
The innovation passed through the slow residual projection remains attached.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from openjev.research.reacher_observation_baseline import _public_packet, _require
from openjev.research.reacher_reward_residual import expected_clipped_action_cost
from openjev.research.reacher_world_models import PublicWorldModel, State

VARIANTS = ("constant", "age", "raw", "normalized")
STATE_KEYS = frozenset({"hidden", "context", "packet", "prior_mean", "prior_variance",
                        "real_index", "last_valid_index", "imagined_depth"})
MAX_STEPS = 50
AGE_RTOL, AGE_ATOL = 1e-5, 1e-6


class InnovationContextWorldModel(PublicWorldModel):
    """Action-conditioned fast state, innovation-corrected slow context.

    ``advance`` preserves slow context and predicts mean/variance before the
    returned observation can be seen. ``assimilate`` accepts startup or exactly
    one advance from a real root. The caller must establish that command was
    actually issued; an imagined terminal is not a real-observation boundary.
    The state stores actual integer clocks so a returning valid packet's zero
    age cannot erase the elapsed gap. No persistent state lives in the module.
    """

    def __init__(self, variant="constant", hidden_size=64, context_size=16, dt=0.02,
                 *, noise_std=0.05, eta=0.1, variance_min=1e-4, variance_max=4.0):
        _require(variant in VARIANTS, "Unknown gate variant")
        _require(type(hidden_size) is int and hidden_size > 0
                 and type(context_size) is int and context_size > 0, "Positive integer state widths")
        values = (dt, noise_std, eta, variance_min, variance_max)
        _require(all(type(x) in (float, int) and math.isfinite(x) for x in values)
                 and dt > 0 and noise_std >= 0 and 0 < eta < 1
                 and 0 < variance_min < variance_max, "Finite valid constructor scalars")
        super().__init__(dt)
        self.variant, self.hidden_size, self.context_size = variant, hidden_size, context_size
        self.noise_std, self.eta = float(noise_std), float(eta)
        self.variance_min, self.variance_max = float(variance_min), float(variance_max)
        self.observation_update = nn.GRUCell(8, hidden_size)
        self.transition = nn.GRUCell(6 + context_size, hidden_size)
        self.observation_head = nn.Linear(hidden_size, 4)
        self.variance_head = nn.Linear(hidden_size, 4)
        self.reward_head = nn.Sequential(nn.Linear(hidden_size + 2, hidden_size), nn.ELU(), nn.Linear(hidden_size, 1))
        self.context_innovation = nn.Linear(4, context_size, bias=False)
        self.gate = nn.Linear(2, 1)

    def initial(self, batch, device=None) -> State:
        _require(type(batch) is int and batch > 0, "Positive integer batch required")
        parameter = next(self.parameters())
        _require(parameter.device.type == "cpu" and parameter.dtype in (torch.float32, torch.float64)
                 and (device is None or torch.device(device) == parameter.device),
                 "Prototype uses CPU float32/float64 without fallback")
        packet = self._zeros(batch, 8, device)
        clock = torch.full((batch, 1), -1, dtype=torch.int64, device=packet.device)
        return {"hidden": packet.new_zeros(batch, self.hidden_size),
                "context": packet.new_zeros(batch, self.context_size), "packet": packet,
                "prior_mean": packet.new_zeros(batch, 4),
                "prior_variance": packet.new_full((batch, 4), self.variance_min),
                "real_index": clock, "last_valid_index": clock.clone(),
                "imagined_depth": torch.zeros_like(clock)}

    def _state(self, state):
        _require(isinstance(state, Mapping) and set(state) == STATE_KEYS, "Exact innovation-context state membership")
        parameter, packet = next(self.parameters()), state["packet"]
        _require(parameter.device.type == "cpu" and parameter.dtype in (torch.float32, torch.float64)
                 and isinstance(packet, Tensor) and packet.ndim == 2 and len(packet) > 0,
                 "Nonempty CPU float32/float64 state")
        batch = len(packet)
        widths = {"hidden": self.hidden_size, "context": self.context_size, "packet": 8,
                  "prior_mean": 4, "prior_variance": 4,
                  "real_index": 1, "last_valid_index": 1, "imagined_depth": 1}
        clocks = {"real_index", "last_valid_index", "imagined_depth"}
        for name, width in widths.items():
            value = state[name]
            dtype = torch.int64 if name in clocks else parameter.dtype
            _require(isinstance(value, Tensor) and value.shape == (batch, width)
                     and value.dtype == dtype and value.device == parameter.device,
                     f"State shape/dtype/device: {name}")
            _require(bool(torch.isfinite(value).all()), f"Nonfinite state: {name}")
        index, last, depth = (state[k] for k in ("real_index", "last_valid_index", "imagined_depth"))
        active = index >= 0
        _require(bool(((index >= -1) & (index <= MAX_STEPS) & (depth >= 0)
                       & (index + depth <= MAX_STEPS)).all()), "Real/imagined terminal clock bounds")
        _require(bool(torch.where(active, (last >= 0) & (last <= index), (last == -1) & (depth == 0)).all()),
                 "Actual observation clock bounds")
        age = (index + depth - last).to(parameter.dtype)
        valid = ((depth == 0) & (index == last) & active).to(parameter.dtype)
        _require(torch.equal(packet[:, 6:7], valid)
                 and bool(torch.isclose(packet[:, 7:8], age * self.dt, rtol=AGE_RTOL, atol=AGE_ATOL).all()),
                 "Public validity/age disagrees with actual clocks")
        missing_root = active[:, 0] & (depth[:, 0] == 0) & (valid[:, 0] == 0)
        _require(bool((packet[missing_root, :4] == 0).all()), "Missing root angles must be sanitized")
        variance = state["prior_variance"]
        _require(bool(((variance >= self.variance_min) & (variance <= self.variance_max)).all()),
                 "Stored predictive variance outside configured bounds")
        return batch

    def assimilate_with_diagnostics(self, state: State, packet: Tensor):
        batch, public = self._state(state), _public_packet(self, packet)
        _require(len(public) == batch, "Real packet batch alignment")
        startup = state["real_index"] == -1
        _require(torch.equal(state["imagined_depth"], (~startup).to(torch.int64)),
                 "Real assimilation requires startup or one issued-command advance")
        index = state["real_index"] + 1
        visible = public[:, 6:7] == 1
        _require(bool((index <= MAX_STEPS).all()) and bool((~startup | visible).all()),
                 "Initial visible packet and terminal50 required")
        _require(bool((public[:, 7:8][visible] == 0).all()), "Visible public age must be zero")
        _require(torch.equal(public[~startup[:, 0], 4:6], state["packet"][~startup[:, 0], 4:6]),
                 "Static public target changed")
        last = torch.where(visible, index, state["last_valid_index"])
        _require(bool(torch.isclose(public[:, 7:8], (index - last).to(public.dtype) * self.dt,
                                   rtol=AGE_RTOL, atol=AGE_ATOL).all()), "Incoming age disagrees with actual clocks")
        eligible = visible & ~startup
        elapsed = torch.where(startup, 0, index - state["last_valid_index"]).to(public.dtype) * self.dt
        innovation = torch.where(eligible, public[:, :4] - state["prior_mean"], 0)
        raw = innovation.square().sum(-1, keepdim=True)
        normalized = (innovation.square() / state["prior_variance"]).sum(-1, keepdim=True)
        # Compute every statistic before applying variant masks. All scalar
        # gate inputs stop gradient; Ue below remains differentiable.
        raw_feature, normalized_feature = torch.log1p(raw).detach(), torch.log1p(normalized).detach()
        error_feature = normalized_feature if self.variant == "normalized" else raw_feature
        mask = public.new_tensor([self.variant != "constant", self.variant in {"raw", "normalized"}])
        features = torch.cat((elapsed.detach(), error_feature), -1) * mask
        gate = torch.sigmoid(self.gate(features))
        delta = torch.tanh(self.context_innovation(innovation))
        observed_hidden = self.observation_update(public, state["hidden"])
        hidden = torch.where(visible, observed_hidden, state["hidden"])
        context = torch.where(eligible, state["context"] + self.eta * gate * delta, state["context"])
        updated = {"hidden": hidden, "context": context, "packet": public,
                   "prior_mean": state["prior_mean"].clone(), "prior_variance": state["prior_variance"].clone(),
                   "real_index": index, "last_valid_index": last,
                   "imagined_depth": torch.zeros_like(index)}
        diagnostics = {"innovation_valid": eligible, "startup": startup, "elapsed_seconds": elapsed,
                       "prior_mean": state["prior_mean"].clone(), "prior_variance": state["prior_variance"].clone(),
                       "innovation": innovation, "raw_squared_error": raw, "normalized_squared_error": normalized,
                       "gate_features": features, "gate": gate,
                       "effective_gate": torch.where(eligible, gate, 0), "context_delta": context - state["context"]}
        self._state(updated)
        return updated, diagnostics

    def assimilate(self, state: State, packet: Tensor) -> State:
        return self.assimilate_with_diagnostics(state, packet)[0]

    def advance(self, state: State, action: Tensor):
        batch = self._state(state)
        _require(bool((state["real_index"] >= 0).all())
                 and bool((state["real_index"] + state["imagined_depth"] < MAX_STEPS).all()),
                 "Visible startup required; no command beyond terminal50")
        packet = state["packet"]
        _require(isinstance(action, Tensor) and action.shape == (batch, 2)
                 and action.dtype == packet.dtype and action.device == packet.device
                 and bool(torch.isfinite(action).all()) and bool((action.abs() <= 1).all()),
                 "Finite issued commands already clipped to [-1,1] required")
        hidden = self.transition(torch.cat((action, packet[:, 4:], state["context"]), -1), state["hidden"])
        mean = self.observation_head(hidden)
        variance = self.variance_min + (self.variance_max - self.variance_min) * torch.sigmoid(self.variance_head(hidden.detach()))
        reward = self.reward_head(torch.cat((hidden, action), -1)).squeeze(-1)
        reward = reward - expected_clipped_action_cost(action, self.noise_std)
        advanced = {"hidden": hidden, "context": state["context"].clone(),
                    "packet": self._next_packet(packet, mean), "prior_mean": mean,
                    "prior_variance": variance, "real_index": state["real_index"].clone(),
                    "last_valid_index": state["last_valid_index"].clone(),
                    "imagined_depth": state["imagined_depth"] + 1}
        self._state(advanced)
        _require(bool(torch.isfinite(reward).all()), "Nonfinite predicted reward")
        return advanced, mean, reward

    def configuration(self):
        return {"model_class": type(self).__name__, "variant": self.variant,
                "hidden_size": self.hidden_size, "context_size": self.context_size, "dt": self.dt,
                "noise_std": self.noise_std, "eta": self.eta,
                "variance_min": self.variance_min, "variance_max": self.variance_max,
                "variance_semantics": "uncalibrated bounded per-coordinate predictive residual second moments",
                "variance_gradient": "head inputs detached; no backbone gradient from variance supervision",
                "gate_gradient": "all scalar features detached; residual context innovation remains attached",
                "gate_features": ["elapsed actual measurement gap in seconds", "log1p squared innovation statistic"],
                "slow_updates": "real visible innovation only; startup, missing and imagination hold context",
                "episode_steps": MAX_STEPS, "state_keys": sorted(STATE_KEYS),
                "provisional_constructor_choices": True,
                "run_status_authority": "enclosing protocol and execution receipts"}


@contextmanager
def capture_module_work(model):
    """Count completed leaf GRU/linear forwards and sample calls with hooks.

    Counts include discarded correction candidates. They exclude non-module
    arithmetic (including the reward skip), validation, copies, backward and
    optimizer work, and do not time anything. A failing leaf forward is not a
    completed call; this is not a full failed-work or compute accounting tool.
    Hook handles are removed even if the caller raises. No model state/RNG is
    changed; the yielded dictionary is a caller-owned mutable receipt.
    """
    _require(type(model) is InnovationContextWorldModel, "Exact innovation-context model required")
    counts, handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, (nn.GRUCell, nn.Linear)):
            counts[name] = {"calls": 0, "samples": 0}
            def record(_module, inputs, _output, name=name):
                counts[name]["calls"] += 1
                counts[name]["samples"] += len(inputs[0])
            handles.append(module.register_forward_hook(record))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()
