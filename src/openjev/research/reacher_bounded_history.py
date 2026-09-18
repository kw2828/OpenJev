"""Engineering-only GRU with a strictly bounded three-real-packet history.

The real decision state is re-encoded from zero using only the final three
sanitized PUBLIC packets and their two intervening ISSUED commands. Explicit
startup masks distinguish absent history from a real but missing measurement.
Neither old hidden states nor earlier predicted angles are replay inputs.

Evidence boundaries are explicit: assimilate appends one real packet; advance
never changes the real-history buffers. After the first observation, a real
assimilation requires exactly one advance from the previous real root. This
records the selected issued command. A multi-step imagined state cannot be
assimilated as if it were real evidence. A caller must execute the chosen
action from the ROOT state, not reuse a candidate's terminal state. The caller
still owns verification that this command was actually issued to the system.

Constructor and parameter initialization are inherited unchanged. Imagination
retains ordinary recurrent transition state within the current planning call.
This component has no trainer, scored study, environment access or result.
Compatible weight names do not authorize substitution into any frozen trainer:
a future protocol must bind the actual class and history/reset semantics.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from openjev.research.reacher_observation_baseline import _public_packet
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import State

VERSION = "three-real-packet-gru-v1"
STATE_KEYS = {
    "hidden",
    "packet",
    "real_packets",
    "real_actions",
    "real_valid",
    "pending_action",
    "imagined_depth",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class BoundedThreePacketGRUWorldModel(GRUResidualRewardWorldModel):
    """Same GRU parameters, with a separately specified real-history contract.

    Replay always computes three observation updates and two transitions for
    every sample, including masked startup slots. Invalid startup transitions
    have no effect, but their computation is explicitly counted. Observation
    missingness uses the inherited GRU gate; it does not mean absent history.
    """

    def initial(self, batch: int, device=None) -> State:
        state = super().initial(batch, device)
        packet = state["packet"]
        return {
            **state,
            "real_packets": packet.new_zeros(batch, 3, 8),
            "real_actions": packet.new_zeros(batch, 2, 2),
            "real_valid": torch.zeros(batch, 3, dtype=torch.bool, device=packet.device),
            "pending_action": packet.new_zeros(batch, 2),
            "imagined_depth": torch.zeros(batch, 1, dtype=torch.int64, device=packet.device),
        }

    def _state_schema(self, state, *, real_boundary):
        _require(isinstance(state, Mapping) and set(state) == STATE_KEYS, "Exact bounded-state membership")
        parameter = next(self.parameters())
        packet = state["packet"]
        _require(
            isinstance(packet, Tensor) and packet.ndim == 2 and len(packet) > 0,
            "Nonempty packet state required",
        )
        batch = len(packet)
        shapes = {
            "packet": (batch, 8),
            "hidden": (batch, self.hidden_size),
            "real_packets": (batch, 3, 8),
            "real_actions": (batch, 2, 2),
            "real_valid": (batch, 3),
            "pending_action": (batch, 2),
            "imagined_depth": (batch, 1),
        }
        for name, shape in shapes.items():
            dtype = {"real_valid": torch.bool, "imagined_depth": torch.int64}.get(name, parameter.dtype)
            value = state[name]
            _require(
                isinstance(value, Tensor)
                and tuple(value.shape) == shape
                and value.dtype == dtype
                and value.device == parameter.device,
                f"Bounded-state schema: {name}",
            )
            if value.is_floating_point() and not (real_boundary and name in {"packet", "hidden"}):
                _require(bool(torch.isfinite(value).all()), f"Nonfinite bounded-state value: {name}")
        valid, packets = state["real_valid"], state["real_packets"]
        _require(not bool((valid[:, :-1] & ~valid[:, 1:]).any()), "Startup mask must be left padded")
        _require(bool((packets[~valid] == 0).all()), "Absent startup packets must be zero")
        _require(
            bool(((packets[..., 6] == 0) | (packets[..., 6] == 1)).all())
            and bool((packets[..., 7] >= 0).all()),
            "Real history validity/age",
        )
        _require(
            bool((packets[..., :4][packets[..., 6] == 0] == 0).all()),
            "Real missing-angle history must remain sanitized",
        )
        edges = valid[:, :-1] & valid[:, 1:]
        _require(bool((state["real_actions"][~edges] == 0).all()), "Absent startup edges must be zero")
        _require(
            bool((state["real_actions"].abs() <= 1).all())
            and bool((state["pending_action"].abs() <= 1).all()),
            "Stored issued command bounds",
        )
        depth = state["imagined_depth"][:, 0]
        _require(
            bool((depth >= 0).all()) and bool((depth < torch.iinfo(torch.int64).max).all()),
            "Imagined depth bounds",
        )
        _require(
            bool((state["pending_action"][depth == 0] == 0).all()), "No command is pending at a real root"
        )
        _require(not bool(((~valid.any(1)) & (depth != 0)).any()), "No action before an initial real packet")
        return batch

    def assimilate(self, state: State, packet: Tensor) -> State:
        public = _public_packet(self, packet)
        batch = self._state_schema(state, real_boundary=True)
        _require(len(public) == batch, "Real packet batch alignment")
        has_history = state["real_valid"].any(1)
        expected_depth = has_history.to(torch.int64)
        _require(
            torch.equal(state["imagined_depth"][:, 0], expected_depth),
            "Real assimilation requires episode start or exactly one issued-action advance; "
            "never assimilate a multi-step imagined terminal state",
        )
        real_packets = torch.cat((state["real_packets"][:, 1:], public[:, None]), 1)
        real_valid = torch.cat(
            (state["real_valid"][:, 1:], torch.ones(batch, 1, dtype=torch.bool, device=public.device)), 1
        )
        real_actions = torch.cat((state["real_actions"][:, 1:], state["pending_action"][:, None]), 1)
        edges = real_valid[:, :-1] & real_valid[:, 1:]
        real_actions = torch.where(edges[..., None], real_actions, 0)

        # Ignore old hidden/packet values completely. Only the raw bounded
        # public window participates in this reconstruction and its gradients.
        replay = super().initial(batch, public.device)
        for point in range(3):
            updated = super().assimilate(replay, real_packets[:, point])
            present = real_valid[:, point : point + 1]
            replay = {name: torch.where(present, updated[name], value) for name, value in replay.items()}
            if point < 2:
                advanced, _, _ = super().advance(replay, real_actions[:, point])
                present_edge = edges[:, point : point + 1]
                replay = {
                    name: torch.where(present_edge, advanced[name], value) for name, value in replay.items()
                }
        return {
            **replay,
            "real_packets": real_packets,
            "real_actions": real_actions,
            "real_valid": real_valid,
            "pending_action": torch.zeros_like(state["pending_action"]),
            "imagined_depth": torch.zeros_like(state["imagined_depth"]),
        }

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        batch = self._state_schema(state, real_boundary=False)
        _require(bool(state["real_valid"][:, -1].all()), "An initial real packet is required before advance")
        _require(
            isinstance(action, Tensor)
            and action.shape == (batch, 2)
            and action.dtype == state["packet"].dtype
            and action.device == state["packet"].device
            and bool(torch.isfinite(action).all())
            and bool((action.abs() <= 1).all()),
            "Finite aligned issued commands clipped to [-1,1] required",
        )
        imagined, angles, reward = super().advance(
            {"hidden": state["hidden"], "packet": state["packet"]}, action
        )
        return (
            {
                **imagined,
                **{name: state[name].clone() for name in ("real_packets", "real_actions", "real_valid")},
                "pending_action": action.clone(),
                "imagined_depth": state["imagined_depth"] + 1,
            },
            angles,
            reward,
        )

    def configuration(self):
        return {
            "version": VERSION,
            "model_class": type(self).__name__,
            "hidden_size": self.hidden_size,
            "dt": self.dt,
            "noise_std": self.noise_std,
            "residual_reward": self.residual_reward,
            "real_packet_window": 3,
            "intervening_issued_commands": 2,
            "real_history": "sanitized public packets only; left-padded explicit startup masks",
            "assimilation": "rebuild from zero at every real packet; initial-or-one-advance phase required",
            "imagined_rollout": "private recurrent hidden/packet state; never append imagined packets to history",
            "state_keys": sorted(STATE_KEYS),
            "engineering_only": True,
            "integrated_study": False,
        }

    def deployment_checkpoint(self):
        """Safe CPU float32 deployment payload, not optimizer state or resume authority.

        Episode histories are external state and deliberately not checkpointed.
        A future training runner must separately bind optimizer/order/source and
        runtime evidence; this helper only prevents wrong-class weight loading.
        """
        _require(type(self) is BoundedThreePacketGRUWorldModel, "Exact deployment class required")
        _require(
            all(p.device.type == "cpu" and p.dtype == torch.float32 for p in self.parameters()),
            "Deployment payload currently requires CPU float32",
        )
        _require(not list(self.named_buffers()), "Unexpected deployment buffers")
        weights = {name: value.detach().clone() for name, value in self.state_dict().items()}
        _require(all(bool(torch.isfinite(value).all()) for value in weights.values()), "Nonfinite weights")
        return {"configuration": self.configuration(), "weights": weights}

    @classmethod
    def from_deployment_checkpoint(cls, payload):
        _require(
            cls is BoundedThreePacketGRUWorldModel
            and isinstance(payload, dict)
            and set(payload) == {"configuration", "weights"},
            "Deployment checkpoint schema/class",
        )
        config = payload["configuration"]
        _require(isinstance(config, dict), "Deployment configuration")
        constructor = {key: config[key] for key in ("hidden_size", "dt", "noise_std", "residual_reward")}
        # Isolate unused construction RNG; exact saved tensors replace it.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = cls(**constructor)
        _require(config == model.configuration(), "Actual bounded-history class/configuration must match")
        weights, expected = payload["weights"], model.state_dict()
        _require(isinstance(weights, dict) and set(weights) == set(expected), "Deployment weight membership")
        for name, value in weights.items():
            _require(
                isinstance(value, Tensor)
                and value.device.type == "cpu"
                and value.dtype == torch.float32
                and value.shape == expected[name].shape
                and bool(torch.isfinite(value).all()),
                f"Deployment weight schema: {name}",
            )
        model.load_state_dict(weights, strict=True)
        return model


def parameter_and_operation_accounting(model, *, assimilate_samples=0, advance_samples=0):
    """Dense-affine MAC estimates, not total FLOPs or measured execution time.

    Full parent advance is used during bounded replay, so its readouts and
    analytic reward are computed and counted even though only state is kept.
    Validation, copies, masks, nonlinearities, gate arithmetic, analytic reward,
    backward and optimizer work still require measured wall-time accounting.
    """
    _require(type(model) is BoundedThreePacketGRUWorldModel, "Exact bounded-history model required")
    _require(
        all(type(value) is int and value >= 0 for value in (assimilate_samples, advance_samples)),
        "Nonnegative integer sample counts required",
    )
    h = model.hidden_size
    updates, transitions = 3 * assimilate_samples, 2 * assimilate_samples + advance_samples
    update_macs, transition_macs = 3 * h * (h + 8), 5 * h * h + 25 * h
    named = {name: value.numel() for name, value in model.named_parameters()}
    return {
        "configuration": model.configuration(),
        "trainable_parameters": sum(named.values()),
        "named_parameters": named,
        "public_assimilate_samples": assimilate_samples,
        "advance_samples": advance_samples,
        "replayed_observation_update_samples": updates,
        "replayed_transition_samples": 2 * assimilate_samples,
        "gru_cell_sample_calls": updates + transitions,
        "linear_layer_sample_calls": 4 * transitions,
        "analytic_reward_sample_calls": transitions if model.residual_reward else 0,
        "dense_affine_macs": updates * update_macs + transitions * transition_macs,
        "startup_masked_work_is_counted": True,
        "compute_matched": False,
        "counts_are_not_total_flops_or_measured_wall_time": True,
    }
