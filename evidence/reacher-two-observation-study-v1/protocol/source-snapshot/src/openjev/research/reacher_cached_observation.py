"""Public last-measurement controls, outside existing frozen studies.

Every real assimilation erases learned hidden state and predicted angles. The
cached controls retain only the last ACTUALLY VISIBLE public angles. These are
private encoder features, never relabeled current measurements or loss targets.
The matched current GRU also encodes every packet from zero, including missing
packets, so availability of the cache is separate from removing a validity gate.

Advance is private imagination: it never updates the cache or real clock. The
caller must re-advance the selected issued command from its real root exactly
once before assimilating the next real packet. This phase check cannot prove
that the caller actually issued the command. Consecutive imagined advances are
permitted, but a multi-step candidate terminal cannot become real evidence.

Constructors are inherited unchanged, preserving parameter names/order and RNG
draws. Shared tensor schemas do not authorize reinterpreting old checkpoints:
an enclosing trainer must bind the actual class and configuration. This module
contains no trainer, environment, scored RNG, weights or efficacy result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from openjev.research.reacher_observation_baseline import (
    FeedForwardObservationWorldModel,
    _public_packet,
    _require,
    _tensor,
)
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import State

VERSION = "public-angle-cache-v1"
AGE_RTOL = 1e-5
AGE_ATOL = 1e-6
_REAL_KEYS = {"real_index", "last_visible_index", "real_target", "imagined_depth"}
_INTEGER_KEYS = {"real_index", "last_visible_index", "imagined_depth"}


class _PublicBoundary:
    """Episode bookkeeping only: no parameters, buffers or constructor draws."""

    _cached = False
    _recurrent = True

    def _keys(self):
        keys = _REAL_KEYS | {"packet"}
        keys |= {"hidden"} if self._recurrent else {"encoder_features"}
        return keys | ({"cached_angles"} if self._cached else set())

    def initial(self, batch: int, device=None) -> State:
        _require(type(batch) is int and batch > 0, "Positive integer batch required")
        state = super().initial(batch, device)
        packet = state["packet"]
        state.update(
            {
                "real_index": torch.full((batch, 1), -1, dtype=torch.int64, device=packet.device),
                "last_visible_index": torch.full((batch, 1), -1, dtype=torch.int64, device=packet.device),
                "real_target": packet.new_zeros(batch, 2),
                "imagined_depth": torch.zeros(batch, 1, dtype=torch.int64, device=packet.device),
            }
        )
        if self._cached:
            state["cached_angles"] = packet.new_zeros(batch, 4)
        if not self._recurrent:
            state["encoder_features"] = packet.clone()
        return state

    def _state_schema(self, state, *, real_boundary):
        _require(isinstance(state, Mapping) and set(state) == self._keys(), "Exact cache-state membership")
        packet = state["packet"]
        _require(
            isinstance(packet, Tensor) and packet.ndim == 2 and len(packet) > 0,
            "Nonempty packet state required",
        )
        batch = len(packet)
        parameter = next(self.parameters())
        widths = {"packet": 8, "real_target": 2, "cached_angles": 4, "encoder_features": 8}
        if self._recurrent:
            widths["hidden"] = self.hidden_size
        for name, value in state.items():
            if name in _INTEGER_KEYS:
                _require(
                    isinstance(value, Tensor)
                    and value.shape == (batch, 1)
                    and value.dtype == torch.int64
                    and value.device == parameter.device,
                    f"Cache-state integer schema: {name}",
                )
            else:
                # These old learned/predicted values are discarded at a real
                # boundary. Inspect their schema, never their numerical values.
                ignored = real_boundary and name in {"packet", "hidden", "encoder_features"}
                _tensor(self, value, (batch, widths[name]), f"Cache state {name}", finite=not ignored)
        index, last, depth = (state[key] for key in ("real_index", "last_visible_index", "imagined_depth"))
        bound = torch.iinfo(torch.int64).max // 2
        _require(
            bool(((index >= -1) & (index < bound)).all()) and bool(((depth >= 0) & (depth < bound)).all()),
            "Real index or imagined depth bounds",
        )
        startup = index == -1
        _require(
            bool(torch.where(startup, last == -1, (last >= 0) & (last <= index)).all()),
            "Last-visible index bounds",
        )
        _require(bool((depth[startup] == 0).all()), "No advance before the first real packet")
        _require(bool((state["real_target"][startup[:, 0]] == 0).all()), "Startup target must be zero")
        if self._cached:
            _require(bool((state["cached_angles"][startup[:, 0]] == 0).all()), "Startup cache must be zero")
        if not real_boundary:
            active = ~startup[:, 0]
            _require(
                bool(((packet[:, 6] == 0) | (packet[:, 6] == 1)).all()) and bool((packet[:, 7] >= 0).all()),
                "Imagined packet validity/age",
            )
            expected_valid = ((depth == 0) & (index == last)).to(packet.dtype)[:, 0]
            _require(
                torch.equal(packet[active, 6], expected_valid[active]),
                "Packet validity disagrees with real clock",
            )
            expected_age = ((index - last + depth).to(packet.dtype) * self.dt)[:, 0]
            _require(
                bool(
                    torch.isclose(packet[active, 7], expected_age[active], rtol=AGE_RTOL, atol=AGE_ATOL).all()
                ),
                "Imagined age disagrees with real clock",
            )
            _require(
                torch.equal(packet[active, 4:6], state["real_target"][active]), "Imagined target changed"
            )
            missing_roots = active & (depth[:, 0] == 0) & (packet[:, 6] == 0)
            _require(
                bool((packet[missing_roots, :4] == 0).all()), "Real missing packet must remain sanitized"
            )
            if not self._recurrent:
                features = state["encoder_features"]
                _require(torch.equal(features[:, 4:], packet[:, 4:]), "Encoder known fields changed")
                root = depth[:, 0] == 0
                _require(
                    torch.equal(features[root, :4], state["cached_angles"][root])
                    and torch.equal(features[~root], packet[~root]),
                    "Encoder feature phase mismatch",
                )
        return batch

    def assimilate(self, state: State, packet: Tensor) -> State:
        public = _public_packet(self, packet)
        batch = self._state_schema(state, real_boundary=True)
        _require(batch == len(public), "Real packet batch alignment")
        startup = state["real_index"] == -1
        _require(
            torch.equal(state["imagined_depth"], (~startup).to(torch.int64)),
            "Real assimilation requires episode start or exactly one issued-action advance; "
            "never assimilate a multi-step imagined terminal state",
        )
        visible = public[:, 6:7] == 1
        _require(bool((~startup | visible).all()), "The first real packet must be visible")
        _require(bool((public[:, 7:8][visible] == 0).all()), "Visible packet age must be zero")
        _require(
            torch.equal(public[~startup[:, 0], 4:6], state["real_target"][~startup[:, 0]]),
            "Static public target changed",
        )
        index = state["real_index"] + 1
        last = torch.where(visible, index, state["last_visible_index"])
        age = (index - last).to(public.dtype) * self.dt
        _require(
            bool(torch.isclose(public[:, 7:8], age, rtol=AGE_RTOL, atol=AGE_ATOL).all()),
            "Public age disagrees with real measurement indices",
        )
        real = {
            "real_index": index,
            "last_visible_index": last,
            "real_target": public[:, 4:6].clone(),
            "imagined_depth": torch.zeros_like(state["imagined_depth"]),
        }
        if self._cached:
            cache = torch.where(visible, public[:, :4], state["cached_angles"])
            real["cached_angles"] = cache
            # Bypass external missing-angle sanitization ONLY for these
            # internal features. Actual public packet/validity are untouched.
            features = torch.cat((cache, public[:, 4:]), -1)
        else:
            features = public
        if self._recurrent:
            fresh_hidden = self._zeros(batch, self.hidden_size, public.device)
            learned = {"hidden": self.observation_update(features, fresh_hidden)}
        else:
            learned = {"encoder_features": features}
        return {"packet": public, **learned, **real}

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        batch = self._state_schema(state, real_boundary=False)
        _require(bool((state["real_index"] >= 0).all()), "An initial real packet is required before advance")
        _tensor(self, action, (batch, 2), "Issued action")
        _require(bool((action.abs() <= 1).all()), "Issued action must already be clipped to [-1, 1]")
        if self._recurrent:
            base = {"packet": state["packet"], "hidden": state["hidden"]}
        else:
            # Parent MLP consumes this private feature vector as its imagined
            # packet; it does not sanitize away its honestly stale angle values.
            base = {"packet": state["encoder_features"]}
        imagined, angles, reward = super().advance(base, action)
        metadata = {key: state[key].clone() for key in _REAL_KEYS - {"imagined_depth"}}
        metadata["imagined_depth"] = state["imagined_depth"] + 1
        if self._cached:
            metadata["cached_angles"] = state["cached_angles"].clone()
        if not self._recurrent:
            imagined["encoder_features"] = imagined["packet"].clone()
        return {**imagined, **metadata}, angles, reward

    def configuration(self):
        cached = type(self) in (CachedObservationGRUWorldModel, CachedObservationMLPWorldModel)
        recurrent = type(self) in (EncodedCurrentGRUWorldModel, CachedObservationGRUWorldModel)
        _require(
            type(self._cached) is bool
            and self._cached == cached
            and type(self._recurrent) is bool
            and self._recurrent == recurrent,
            "Public-cache behavior flags changed",
        )
        expected = configuration_for(
            type(self),
            width=self.hidden_size if self._recurrent else self.width,
            dt=self.dt,
            noise_std=self.noise_std,
            residual_reward=self.residual_reward,
        )
        _require(sorted(self._keys()) == expected["state_keys"], "Public-cache state keys changed")
        return expected


class EncodedCurrentGRUWorldModel(_PublicBoundary, GRUResidualRewardWorldModel):
    """Current packet, reset and encoded unconditionally, with no angle cache."""


class CachedObservationGRUWorldModel(_PublicBoundary, GRUResidualRewardWorldModel):
    """Same GRU tensors, explicit last-visible-angle memory, no learned history."""

    _cached = True


class CachedObservationMLPWorldModel(_PublicBoundary, FeedForwardObservationWorldModel):
    """Same packet-MLP tensors, explicit last-visible-angle memory at real roots."""

    _cached = True
    _recurrent = False


def configuration_for(model_class, *, width, dt, noise_std, residual_reward):
    """Pure expected configuration: no construction, tensor work or RNG draws.

    ``width`` denotes hidden size for the GRUs and hidden-layer width for the
    MLP. Require the exact actual class, not merely a compatible weight schema.
    An enclosing trainer may compare this with model.configuration() at update
    and restore boundaries without creating or reinitializing another model.
    """
    _require(
        model_class
        in (EncodedCurrentGRUWorldModel, CachedObservationGRUWorldModel, CachedObservationMLPWorldModel),
        "Exact public-cache model class required",
    )
    _require(type(width) is int and width > 0, "Positive integer width required")
    _require(type(dt) in (int, float) and math.isfinite(dt) and dt > 0, "Finite positive dt required")
    _require(
        type(noise_std) in (int, float) and math.isfinite(noise_std) and noise_std >= 0,
        "Finite nonnegative noise_std required",
    )
    _require(type(residual_reward) is bool, "Boolean residual_reward required")
    # Derive semantics from exact identities, not mutable instance/class flags.
    cached = model_class in (CachedObservationGRUWorldModel, CachedObservationMLPWorldModel)
    recurrent = model_class in (EncodedCurrentGRUWorldModel, CachedObservationGRUWorldModel)
    keys = _REAL_KEYS | {"packet"} | ({"hidden"} if recurrent else {"encoder_features"})
    if cached:
        keys |= {"cached_angles"}
    return {
        "version": VERSION,
        "model_class": model_class.__name__,
        "hidden_size" if recurrent else "width": width,
        "dt": float(dt),
        "noise_std": float(noise_std),
        "residual_reward": residual_reward,
        "encoder_feature_order": [
            "cached_cos_q0" if cached else "current_cos_q0",
            "cached_cos_q1" if cached else "current_cos_q1",
            "cached_sin_q0" if cached else "current_sin_q0",
            "cached_sin_q1" if cached else "current_sin_q1",
            "current_target_x",
            "current_target_y",
            "actual_validity",
            "actual_age_seconds",
        ],
        "cache": "last actually visible public angles only" if cached else "none",
        "assimilation": "erase learned and predicted state; encode every real packet from zero",
        "phase": "visible episode start, then exactly one selected issued-action advance",
        "target": "static exact public target",
        "age_validation": {"rtol": AGE_RTOL, "atol": AGE_ATOL, "visible_age": 0.0},
        "imagined_rollout": "private action-conditioned state; no real-clock or cache updates",
        "state_keys": sorted(keys),
        "run_status_authority": "enclosing protocol and execution receipts",
    }


def parameter_accounting(model):
    """Parameter/state payload and affine forward estimates, not compute parity."""
    _require(
        type(model)
        in (EncodedCurrentGRUWorldModel, CachedObservationGRUWorldModel, CachedObservationMLPWorldModel),
        "Expected an exact public-cache control class",
    )
    width = model.hidden_size if model._recurrent else model.width
    macs = {
        "assimilate": 3 * width * (8 + width) if model._recurrent else 0,
        "advance": 5 * width * width + 25 * width if model._recurrent else 3 * width * width + 17 * width,
    }
    counts = {name: parameter.numel() for name, parameter in model.named_parameters()}
    floating = 8 + 2 + (width if model._recurrent else 8) + (4 if model._cached else 0)
    return {
        **model.configuration(),
        "trainable_parameters": sum(counts.values()),
        "named_parameters": counts,
        "state_payload_per_case": {
            "floating_elements": floating,
            "int64_elements": 3,
            "tensor_bytes": floating * next(model.parameters()).element_size() + 3 * 8,
            "excludes": "autograd graph, allocator, mapping/object overhead and temporary tensors",
        },
        "dense_affine_macs_per_sample": macs,
        "macs_are_not_total_flops": True,
        "excluded_work": [
            "bias",
            "activations",
            "GRU gates",
            "analytic reward",
            "cache and phase validation",
            "feature construction/copies",
            "backward/optimizer",
        ],
        "wall_time_requirement": "charge all neural, cache, validation, copying and optimizer work",
    }


def operation_counts(model, *, assimilate_samples=0, advance_samples=0):
    """Caller-supplied sample counts include every missing root and candidate.

    These are forward-operation estimates, not measured timing or proof of
    execution. Backward/optimizer/bookkeeping must be charged in wall time.
    """
    _require(
        all(type(value) is int and value >= 0 for value in (assimilate_samples, advance_samples)),
        "Nonnegative integer sample counts required",
    )
    macs = parameter_accounting(model)["dense_affine_macs_per_sample"]
    return {
        "assimilate_samples": assimilate_samples,
        "advance_samples": advance_samples,
        "real_state_reset_samples": assimilate_samples,
        "real_boundary_validation_samples": assimilate_samples,
        "cache_update_opportunity_samples": assimilate_samples if model._cached else 0,
        "gru_cell_sample_calls": assimilate_samples + advance_samples if model._recurrent else 0,
        "linear_layer_sample_calls": (4 if model._recurrent else 6) * advance_samples,
        "analytic_reward_sample_calls": advance_samples if model.residual_reward else 0,
        "dense_affine_macs": assimilate_samples * macs["assimilate"] + advance_samples * macs["advance"],
        "counts_are_not_total_flops_or_measured_wall_time": True,
    }
