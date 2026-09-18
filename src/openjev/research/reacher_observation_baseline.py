"""Future current-observation comparators, outside all existing frozen studies.

Call ``assimilate`` before EVERY real decision, including missing observations.
It discards all preceding state. Consecutive ``advance`` calls may still retain
an imagined trajectory inside one planning call. That action-conditioned
open-loop state is necessary for multi-step planning and is not retained
observation history across real decisions.

The preferred comparator is CurrentObservationGRUWorldModel: exactly the same
parameters, initialization, readouts, reward skip and imagined transition as
the conventional GRU, trained with the history-erasing assimilation rule.
It is not a post-training reset intervention. The optional feedforward model
changes the transition architecture and is only approximately size matched.

The current eight-dimensional packet is NOT a sufficient physical Markov
state: velocity is absent and blackout packets omit joint angles. Both models
are observation-only estimators, not privileged physics models. They accept
the unchanged public sequence_loss inputs and total native reward targets.
No training, checkpoint loading, evaluation or scientific result lives here.
The existing frozen objective trainer binds GRUResidualRewardWorldModel by
class/configuration. State-dict compatibility does NOT authorize substituting
this class there. A future trainer must explicitly bind the actual constructor
and assimilation rule in its protocol and restore metadata.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from openjev.research.reacher_reward_residual import (
    GRUResidualRewardWorldModel,
    expected_clipped_action_cost,
)
from openjev.research.reacher_world_models import PublicWorldModel, State


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(model, value, shape, label, *, finite=True):
    parameter = next(model.parameters())
    _require(
        isinstance(value, Tensor)
        and tuple(value.shape) == tuple(shape)
        and value.dtype == parameter.dtype
        and value.device == parameter.device,
        f"{label} shape/dtype/device",
    )
    if finite:
        _require(bool(torch.isfinite(value).all()), f"Nonfinite {label}")


def _public_packet(model, packet):
    _require(
        isinstance(packet, Tensor) and packet.ndim == 2 and len(packet) > 0,
        "Public packet must have a nonempty batch",
    )
    _tensor(model, packet, (len(packet), 8), "Public packet", finite=False)
    _require(bool(((packet[:, 6] == 0) | (packet[:, 6] == 1)).all()), "Binary observation validity")
    # Unavailable angular placeholders cannot become a hidden-state channel.
    # Even NaN/inf in these unavailable entries is discarded before validation.
    public = torch.cat((torch.where(packet[:, 6:7] == 1, packet[:, :4], 0), packet[:, 4:]), -1)
    _require(
        bool(torch.isfinite(public).all()) and bool((public[:, 7] >= 0).all()),
        "Nonfinite known public fields or negative age",
    )
    return public


def _old_state_schema(model, state, batch, *, hidden=False):
    expected = {"packet", "hidden"} if hidden else {"packet"}
    _require(isinstance(state, Mapping) and set(state) == expected, "State key membership")
    # Deliberately inspect only schema, never any old value, at a real boundary.
    _tensor(model, state["packet"], (batch, 8), "Previous packet", finite=False)
    if hidden:
        _tensor(model, state["hidden"], (batch, model.hidden_size), "Previous hidden", finite=False)


class CurrentObservationGRUWorldModel(GRUResidualRewardWorldModel):
    """Parameter/init-identical GRU trained with no cross-decision history.

    Constructor and advance are inherited unchanged. Missing current angles
    are zeroed, current target/validity/age are preserved, and missing packets
    produce zero hidden state under the inherited valid-measurement update.
    Old hidden/packet values are unused, even if they contain invalid numbers.
    ``sequence_loss`` invokes this reset at each real packet while preserving
    recurrent imagined advances within its complete open-loop windows.
    """

    def assimilate(self, state: State, packet: Tensor) -> State:
        public = _public_packet(self, packet)
        _old_state_schema(self, state, len(public), hidden=True)
        fresh = self.initial(len(public), public.device)
        return super().assimilate(fresh, public)


class FeedForwardObservationWorldModel(PublicWorldModel):
    """Packet/action MLP with predicted packets as its only imagined state.

    Width107 gives 36,599 parameters, compared with the conventional GRU64's
    36,805. This is a near parameter-count control, NOT a compute match. Every
    real assimilation overwrites imagined angles even when sensing is absent.
    Imagined angles have valid=0 and are never relabeled measured observations.
    The known expected actuator cost is identical to the GRU residual skip.
    The inherited analytic reward helper requires float64-capable backends;
    current engineering tests and the intended future comparator target CPU.
    """

    def __init__(
        self, width: int = 107, dt: float = 0.02, *, noise_std: float = 0.05, residual_reward: bool = True
    ):
        _require(type(width) is int and width > 0, "Positive integer width required")
        _require(
            type(noise_std) in (float, int) and math.isfinite(noise_std) and noise_std >= 0,
            "Finite nonnegative noise_std required",
        )
        _require(type(residual_reward) is bool, "Boolean residual_reward required")
        super().__init__(dt)
        self.width = width
        self.noise_std = float(noise_std)
        self.residual_reward = residual_reward
        self.encoder = nn.Sequential(nn.Linear(10, width), nn.ELU(), nn.Linear(width, width), nn.ELU())
        self.observation_head = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, 4))
        self.reward_head = nn.Sequential(nn.Linear(width + 2, width), nn.ELU(), nn.Linear(width, 1))

    def initial(self, batch: int, device=None) -> State:
        return {"packet": self._zeros(batch, 8, device)}

    def assimilate(self, state: State, packet: Tensor) -> State:
        public = _public_packet(self, packet)
        _old_state_schema(self, state, len(public))
        return {"packet": public}

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        _require(isinstance(state, Mapping) and set(state) == {"packet"}, "State key membership")
        _require(
            isinstance(state["packet"], Tensor) and state["packet"].ndim == 2 and len(state["packet"]) > 0,
            "Nonempty imagined packet batch",
        )
        batch = len(state["packet"])
        _tensor(self, state["packet"], (batch, 8), "Imagined packet")
        _tensor(self, action, (batch, 2), "Issued action")
        _require(bool((action.abs() <= 1).all()), "Issued action must already be clipped to [-1, 1]")
        _require(
            bool(((state["packet"][:, 6] == 0) | (state["packet"][:, 6] == 1)).all())
            and bool((state["packet"][:, 7] >= 0).all()),
            "Imagined packet validity/age",
        )
        features = self.encoder(torch.cat((state["packet"], action), -1))
        angles = self.observation_head(features)
        reward = self.reward_head(torch.cat((features, action), -1)).squeeze(-1)
        if self.residual_reward:
            reward = reward - expected_clipped_action_cost(action, self.noise_std)
        return {"packet": self._next_packet(state["packet"], angles)}, angles, reward


def parameter_accounting(model):
    """Explicit architecture/configuration and dense-affine operation estimates.

    MACs exclude bias adds, activations, GRU gate arithmetic, analytic reward
    evaluation, validation, copying and backward/optimizer work. A future run
    must charge all of those in measured end-to-end wall time.
    """
    _require(
        type(model) in (CurrentObservationGRUWorldModel, FeedForwardObservationWorldModel),
        "Expected an isolated observation comparator",
    )
    if isinstance(model, CurrentObservationGRUWorldModel):
        h = model.hidden_size
        assimilation_macs = 3 * h * (8 + h)
        advance_macs = 3 * h * (6 + h) + h * h + 4 * h + (h + 2) * h + h
        width = h
    else:
        width = model.width
        assimilation_macs = 0
        advance_macs = 3 * width * width + 17 * width
    counts = {name: parameter.numel() for name, parameter in model.named_parameters()}
    return {
        "architecture": type(model).__name__,
        "engineering_only": True,
        "integrated_into_existing_study": False,
        "width": width,
        "dt": model.dt,
        "noise_std": model.noise_std,
        "residual_reward": model.residual_reward,
        "trainable_parameters": sum(counts.values()),
        "named_parameters": counts,
        "state_keys": ["packet", "hidden"]
        if isinstance(model, CurrentObservationGRUWorldModel)
        else ["packet"],
        "real_boundary": "Every public packet erases all preceding state, including during missing sensing.",
        "open_loop": "Repeated action-conditioned advance can retain its imagined trajectory until assimilation.",
        "dense_affine_macs_per_sample": {"assimilate": assimilation_macs, "advance": advance_macs},
        "macs_are_not_total_flops": True,
        "excluded_work": [
            "bias",
            "activations",
            "GRU gates",
            "analytic reward",
            "validation/copies",
            "backward/optimizer",
        ],
    }


def operation_counts(model, *, assimilate_samples=0, advance_samples=0):
    """Account all interfaces, including assimilation with missing packets.

    These are caller-supplied sample counts, not instrumented timing or proof
    of execution. Branch expansion must be included in advance_samples.
    """
    _require(
        all(type(value) is int and value >= 0 for value in (assimilate_samples, advance_samples)),
        "Nonnegative integer sample counts required",
    )
    accounting = parameter_accounting(model)
    macs = accounting["dense_affine_macs_per_sample"]
    recurrent = isinstance(model, CurrentObservationGRUWorldModel)
    return {
        "assimilate_samples": assimilate_samples,
        "advance_samples": advance_samples,
        "real_state_reset_samples": assimilate_samples,
        "gru_cell_sample_calls": assimilate_samples + advance_samples if recurrent else 0,
        "linear_layer_sample_calls": (4 if recurrent else 6) * advance_samples,
        "analytic_reward_sample_calls": advance_samples if model.residual_reward else 0,
        "dense_affine_macs": assimilate_samples * macs["assimilate"] + advance_samples * macs["advance"],
        "counts_are_not_total_flops_or_measured_wall_time": True,
    }
