"""Prospective control: reconstruct from the last two valid public observations.

At each REAL boundary, discard previous learned/predicted state and replay from
zero the public packet/action suffix beginning at the older of the last TWO
valid observations. Before a second measurement, the first is the anchor.
The suffix includes every missing packet and intervening ISSUED command, with
explicit startup masks and integer indices. Twelve slots permit eleven actions;
unsupported longer suffixes are rejected, never silently truncated.

All twelve observation updates and eleven full parent advances execute even
for padding. Their gradients remain attached and their computation is charged.
Candidate advances have private learned rollout state, never new real evidence.
A real assimilation requires startup or exactly one advance from its real root;
the caller must separately prove that its pending command was actually issued.
This component explicitly enforces the Reacher fifty-action episode boundary;
it is not a generic unbounded-sequence recurrent model.

Same inherited GRU constructor, parameters and initialization. No training,
environment, scored streams, checkpoint selection or current-study integration.
A future trainer must bind this actual class, not just compatible tensor names.
This is a known-history control, not a novel architecture or efficacy result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from openjev.research.reacher_observation_baseline import _public_packet, _require
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import State

VERSION = "two-valid-observation-history-gru-v1"
WINDOW = 12
MAX_STEPS = 50
AGE_RTOL, AGE_ATOL = 1e-5, 1e-6
REAL_KEYS = frozenset({"real_packets", "real_actions", "real_present", "real_indices", "real_index", "real_target"})
STATE_KEYS = REAL_KEYS | {"hidden", "packet", "pending_action", "imagined_depth"}


class TwoObservationHistoryGRUWorldModel(GRUResidualRewardWorldModel):
    """Parameter-identical GRU with a hard real-evidence boundary.

    State is external and batch-indexable with the unchanged ``repeat_index``.
    Gradients flow through retained public packets, issued actions and replay
    weights. No gradient or numerical dependency survives from discarded past
    hidden state, predictions, packets or actions.
    """

    def initial(self, batch: int, device=None) -> State:
        _require(type(batch) is int and batch > 0, "Positive integer batch required")
        state = super().initial(batch, device)
        packet = state["packet"]
        return {
            **state,
            "real_packets": packet.new_zeros(batch, WINDOW, 8),
            "real_actions": packet.new_zeros(batch, WINDOW - 1, 2),
            "real_present": torch.zeros(batch, WINDOW, dtype=torch.bool, device=packet.device),
            "real_indices": torch.full((batch, WINDOW), -1, dtype=torch.int64, device=packet.device),
            "real_index": torch.full((batch, 1), -1, dtype=torch.int64, device=packet.device),
            "real_target": packet.new_zeros(batch, 2),
            "pending_action": packet.new_zeros(batch, 2),
            "imagined_depth": torch.zeros(batch, 1, dtype=torch.int64, device=packet.device),
        }

    def _state_schema(self, state, *, real_boundary):
        _require(isinstance(state, Mapping) and set(state) == STATE_KEYS, "Exact two-observation state membership")
        parameter, packet = next(self.parameters()), state["packet"]
        _require(isinstance(packet, Tensor) and packet.ndim == 2 and len(packet) > 0, "Nonempty packet state")
        batch = len(packet)
        shapes = {
            "hidden": (batch, self.hidden_size), "packet": (batch, 8),
            "real_packets": (batch, WINDOW, 8), "real_actions": (batch, WINDOW - 1, 2),
            "real_present": (batch, WINDOW), "real_indices": (batch, WINDOW),
            "real_index": (batch, 1), "real_target": (batch, 2),
            "pending_action": (batch, 2), "imagined_depth": (batch, 1),
        }
        for name, shape in shapes.items():
            dtype = (torch.bool if name == "real_present" else torch.int64
                     if name in {"real_indices", "real_index", "imagined_depth"} else parameter.dtype)
            value = state[name]
            _require(isinstance(value, Tensor) and tuple(value.shape) == shape
                     and value.dtype == dtype and value.device == parameter.device,
                     f"Two-observation state shape/dtype/device: {name}")
            if value.is_floating_point() and not (real_boundary and name in {"hidden", "packet"}):
                _require(bool(torch.isfinite(value).all()), f"Nonfinite two-observation state: {name}")
        index, depth = state["real_index"], state["imagined_depth"]
        _require(bool(((index >= -1) & (index <= MAX_STEPS)).all())
                 and bool(((depth >= 0) & (depth <= MAX_STEPS)).all()), "Real/imagined clock bounds")
        _require(bool((index + depth <= MAX_STEPS).all()), "Imagined rollout exceeds terminal observation50")
        present, indices, packets = state["real_present"], state["real_indices"], state["real_packets"]
        _require(not bool((present[:, :-1] & ~present[:, 1:]).any()), "Startup mask must be left padded")
        _require(torch.equal(present.any(1), index[:, 0] >= 0), "History and real clock disagree")
        expected = index - torch.arange(WINDOW - 1, -1, -1, device=parameter.device)
        _require(torch.equal(indices, torch.where(present, expected, -1))
                 and bool((indices[present] >= 0).all()), "Contiguous causal real indices")
        _require(bool((packets[~present] == 0).all()), "Absent startup packets must be zero")
        _require(bool(((packets[..., 6] == 0) | (packets[..., 6] == 1)).all())
                 and bool((packets[..., 7] >= 0).all()), "Stored public validity/age")
        visible = present & (packets[..., 6] == 1)
        counts = visible.sum(1)
        active = index[:, 0] >= 0
        _require(bool(((counts[active] >= 1) & (counts[active] <= 2)).all()), "One or two retained valid observations")
        first_slot = WINDOW - present.sum(1)
        first_index = indices.gather(1, first_slot.clamp_max(WINDOW - 1)[:, None])[:, 0]
        first_visible = visible.gather(1, first_slot.clamp_max(WINDOW - 1)[:, None])[:, 0]
        _require(bool(first_visible[active].all())
                 and bool((first_index[counts == 1] == 0).all()), "Suffix must start at its valid observation anchor")
        _require(bool((packets[..., :4][~visible] == 0).all()), "Missing angular history must remain sanitized")
        _require(torch.equal(packets[..., 4:6][present], state["real_target"][:, None].expand(-1, WINDOW, -1)[present]),
                 "Stored public target must be static")
        last_at_slot = torch.where(visible, indices, -1).cummax(1).values
        ages = (indices - last_at_slot).to(parameter.dtype) * self.dt
        _require(bool(torch.isclose(packets[..., 7][present], ages[present], rtol=AGE_RTOL, atol=AGE_ATOL).all()),
                 "Stored ages disagree with real measurement indices")
        edges = present[:, :-1] & present[:, 1:]
        _require(bool((state["real_actions"][~edges] == 0).all()), "Absent startup actions must be zero")
        _require(bool((state["real_actions"].abs() <= 1).all())
                 and bool((state["pending_action"].abs() <= 1).all()), "Issued command bounds")
        startup = ~active
        _require(bool((depth[startup] == 0).all()) and bool((state["real_target"][startup] == 0).all()),
                 "Empty episode clock/target")
        _require(bool((state["pending_action"][depth[:, 0] == 0] == 0).all()), "No pending command at a real root")
        if not real_boundary:
            last = last_at_slot[:, -1:]
            wanted_age = (index - last + depth).to(parameter.dtype) * self.dt
            wanted_valid = ((depth == 0) & (index == last)).to(parameter.dtype)
            _require(torch.equal(packet[active, 4:6], state["real_target"][active])
                     and torch.equal(packet[active, 6:7], wanted_valid[active])
                     and bool(torch.isclose(packet[active, 7:8], wanted_age[active], rtol=AGE_RTOL, atol=AGE_ATOL).all()),
                     "Imagined public target/validity/age disagrees with real clock")
            roots = active & (depth[:, 0] == 0)
            _require(torch.equal(packet[roots], packets[roots, -1]), "Real root packet differs from actual evidence")
        return batch

    def assimilate(self, state: State, packet: Tensor) -> State:
        public = _public_packet(self, packet)
        batch = self._state_schema(state, real_boundary=True)
        _require(len(public) == batch, "Real packet batch alignment")
        startup = state["real_index"] == -1
        _require(torch.equal(state["imagined_depth"], (~startup).to(torch.int64)),
                 "Real assimilation requires startup or exactly one issued-action advance, never an imagined terminal")
        index = state["real_index"] + 1
        _require(bool((index <= MAX_STEPS).all()), "No real observation after terminal50")
        visible = public[:, 6:7] == 1
        _require(bool((~startup | visible).all()), "First real packet must be visible")
        _require(bool((public[:, 7:8][visible] == 0).all()), "Visible public age must be zero")
        _require(torch.equal(public[~startup[:, 0], 4:6], state["real_target"][~startup[:, 0]]), "Static public target changed")
        previous_visible = state["real_present"] & (state["real_packets"][..., 6] == 1)
        last = torch.where(previous_visible, state["real_indices"], -1).max(1, keepdim=True).values
        newest = torch.where(visible, index, last)
        expected_age = (index - newest).to(public.dtype) * self.dt
        _require(bool(torch.isclose(public[:, 7:8], expected_age, rtol=AGE_RTOL, atol=AGE_ATOL).all()),
                 "Public age disagrees with real measurement indices")

        # Append to a thirteen-slot scratch suffix first. Compute its true anchor
        # before dropping any slot, so overflow can never silently lose evidence.
        appended = torch.cat((state["real_packets"], public[:, None]), 1)
        indices = torch.cat((state["real_indices"], index), 1)
        present = torch.cat((state["real_present"], torch.ones_like(visible)), 1)
        actual_valid = present & (appended[..., 6] == 1)
        latest_two = torch.where(actual_valid, indices, -1).topk(2, dim=1).values
        anchor = torch.where(latest_two[:, 1:2] >= 0, latest_two[:, 1:2], latest_two[:, :1])
        keep = present & (indices >= anchor)
        _require(bool((keep.sum(1) <= WINDOW).all()), "Two-observation suffix exceeds twelve packets/eleven commands")
        present = keep[:, -WINDOW:]
        real_packets = torch.where(present[..., None], appended[:, -WINDOW:], 0)
        real_indices = torch.where(present, indices[:, -WINDOW:], -1)
        actions = torch.cat((state["real_actions"], state["pending_action"][:, None]), 1)[:, -(WINDOW - 1):]
        edges = present[:, :-1] & present[:, 1:]
        real_actions = torch.where(edges[..., None], actions, 0)

        replay = super().initial(batch, public.device)
        for point in range(WINDOW):
            updated = super().assimilate(replay, real_packets[:, point])
            mask = present[:, point:point + 1]
            replay = {name: torch.where(mask, updated[name], value) for name, value in replay.items()}
            if point < WINDOW - 1:
                advanced, _, _ = super().advance(replay, real_actions[:, point])
                edge = edges[:, point:point + 1]
                replay = {name: torch.where(edge, advanced[name], value) for name, value in replay.items()}
        return {**replay, "real_packets": real_packets, "real_actions": real_actions,
                "real_present": present, "real_indices": real_indices, "real_index": index,
                "real_target": public[:, 4:6].clone(), "pending_action": torch.zeros_like(state["pending_action"]),
                "imagined_depth": torch.zeros_like(state["imagined_depth"])}

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        batch = self._state_schema(state, real_boundary=False)
        _require(bool((state["real_index"] >= 0).all()), "Initial real packet required before advance")
        _require(bool((state["real_index"] + state["imagined_depth"] < MAX_STEPS).all()), "No action beyond terminal50")
        _require(isinstance(action, Tensor) and action.shape == (batch, 2)
                 and action.dtype == state["packet"].dtype and action.device == state["packet"].device
                 and bool(torch.isfinite(action).all()) and bool((action.abs() <= 1).all()),
                 "Finite aligned issued commands already clipped to [-1,1] required")
        imagined, angles, reward = super().advance({"hidden": state["hidden"], "packet": state["packet"]}, action)
        return ({**imagined, **{name: state[name].clone() for name in REAL_KEYS},
                 "pending_action": action.clone(), "imagined_depth": state["imagined_depth"] + 1}, angles, reward)

    def configuration(self):
        _require(type(self) is TwoObservationHistoryGRUWorldModel, "Exact two-observation class required")
        _require(type(self.hidden_size) is int and self.hidden_size > 0
                 and type(self.dt) in (int, float) and math.isfinite(self.dt) and self.dt > 0
                 and type(self.noise_std) in (int, float) and math.isfinite(self.noise_std) and self.noise_std >= 0
                 and type(self.residual_reward) is bool, "Valid constructor configuration required")
        return {"version": VERSION, "model_class": type(self).__name__, "hidden_size": self.hidden_size,
                "dt": self.dt, "noise_std": self.noise_std, "residual_reward": self.residual_reward,
                "valid_observations": 2, "max_real_packets": WINDOW, "max_issued_commands": WINDOW - 1,
                "episode_steps": MAX_STEPS, "age_rtol": AGE_RTOL, "age_atol": AGE_ATOL,
                "anchor": "older of last two actual valid observations; first observation before second exists",
                "real_history": "sanitized actual public packets and issued commands; integer indices; left padding",
                "assimilation": "zero-start reconstruction at every real boundary; initial-or-one-selected-advance phase",
                "replay": "all twelve assimilations and eleven full transitions, including padding, without detach",
                "imagined_rollout": "private recurrent hidden/packet; never appends real evidence",
                "overflow": "reject before dropping required anchor or command",
                "state_keys": sorted(STATE_KEYS), "run_status_authority": "enclosing protocol and execution receipts"}

    def deployment_checkpoint(self):
        """CPU float32 deployment only; not a trainer/optimizer resume checkpoint."""
        configuration = self.configuration()
        _require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in self.parameters())
                 and not list(self.named_buffers()), "CPU float32 deployment without extra buffers")
        weights = {name: value.detach().clone() for name, value in self.state_dict().items()}
        _require(all(bool(torch.isfinite(value).all()) for value in weights.values()), "Finite deployment weights")
        return {"configuration": configuration, "weights": weights}

    @classmethod
    def from_deployment_checkpoint(cls, payload):
        _require(cls is TwoObservationHistoryGRUWorldModel and isinstance(payload, dict)
                 and set(payload) == {"configuration", "weights"}, "Actual-class deployment schema")
        config = payload["configuration"]
        _require(isinstance(config, dict), "Deployment configuration")
        keys = ("hidden_size", "dt", "noise_std", "residual_reward")
        _require(all(key in config for key in keys), "Missing deployment constructor configuration")
        _require(type(config["hidden_size"]) is int and config["hidden_size"] > 0
                 and type(config["residual_reward"]) is bool
                 and all(type(config[key]) in (int, float) and math.isfinite(config[key]) for key in ("dt", "noise_std"))
                 and config["dt"] > 0 and config["noise_std"] >= 0, "Invalid deployment constructor configuration")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)  # Isolated engineering construction; all tensors replaced.
            model = cls(**{key: config[key] for key in keys})
        _require(config == model.configuration(), "Actual two-observation class/history configuration must match")
        weights, expected = payload["weights"], model.state_dict()
        _require(isinstance(weights, dict) and set(weights) == set(expected), "Deployment weight membership")
        for name, value in weights.items():
            _require(isinstance(value, Tensor) and value.device.type == "cpu" and value.dtype == torch.float32
                     and value.shape == expected[name].shape and bool(torch.isfinite(value).all()),
                     "Deployment weight schema: " + name)
        model.load_state_dict(weights, strict=True)
        return model


def parameter_and_operation_accounting(model, *, assimilate_samples=0, advance_samples=0):
    """Affine MAC estimates, including full masked replay; not total FLOPs.

    Validation, buffers/copies, masks, nonlinearities, analytic reward arithmetic,
    backward, optimizer and storage must additionally be charged in wall time.
    Real evidence storage is bounded, not constant-time to reconstruct.
    These formulas count completed calls only; a future controller must retain
    failed-prefix counters and measure reconstruction separately in wall time.
    """
    _require(type(model) is TwoObservationHistoryGRUWorldModel, "Exact two-observation accounting class")
    _require(all(type(value) is int and value >= 0 for value in (assimilate_samples, advance_samples)),
             "Nonnegative integer sample counts")
    h = model.hidden_size
    updates, replayed = WINDOW * assimilate_samples, (WINDOW - 1) * assimilate_samples
    transitions = replayed + advance_samples
    named = {name: value.numel() for name, value in model.named_parameters()}
    return {"configuration": model.configuration(), "trainable_parameters": sum(named.values()), "named_parameters": named,
            "public_assimilate_samples": assimilate_samples, "advance_samples": advance_samples,
            "replayed_observation_update_samples": updates, "replayed_transition_samples": replayed,
            "gru_cell_sample_calls": updates + transitions, "linear_layer_sample_calls": 4 * transitions,
            "analytic_reward_sample_calls": transitions if model.residual_reward else 0,
            "dense_affine_macs": updates * (3 * h * (h + 8)) + transitions * (5 * h * h + 25 * h),
            "startup_masked_work_is_counted": True, "compute_matched": False,
            "counts_are_not_total_flops_or_measured_wall_time": True}
