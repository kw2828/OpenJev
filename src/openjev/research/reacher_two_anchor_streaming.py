"""Functional reuse of bounded two-observation GRU reconstruction.

This isolated sequence kernel has no persistent cache, parameters, RNG, trainer
or controller integration. Its caller keeps weights fixed during the call and
supplies only actual public packets and actually issued commands. Two branches
start from zero at actual valid measurements. A new measurement promotes the
previous newest branch and starts another from zero. Imagined predictions never
become measurements. Root functions match suffix reconstruction mathematically;
shared autograd accumulation can change floating-point gradient rounding.
For a root-only objective, discarded readouts can have ``None`` gradients here
but explicit zero gradients through the reference's masked padding. Those are
the same mathematical derivative, not necessarily the same optimizer behavior.

All parent advance heads and the configured analytical reward skip execute,
including unused outputs. Counts describe completed forward work, not FLOPs,
wall time, failed-prefix accounting or a measured speed advantage. Validation,
branch selection, graph storage and copies add work. The existing frozen study
and its twelve-slot reconstruction implementation are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openjev.research.reacher_observation_baseline import _public_packet, _require
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_two_observation_history import (
    AGE_ATOL,
    AGE_RTOL,
    MAX_STEPS,
    WINDOW,
)
from openjev.research.reacher_world_models import State


@dataclass(frozen=True)
class StreamingRoots:
    """Batch/time roots and actual measurement indices, with attached gradients.

    ``roots`` has hidden[B,T,H] and packet[B,T,8]; anchors are int64[B,T].
    Tensor contents remain caller-owned, not immutable or a resume authority.
    No newest-branch hidden state is exposed for reuse across optimizer steps.
    """

    roots: State
    older_anchor: Tensor
    newest_anchor: Tensor
    work: dict[str, int]


def _inputs(model, packets, commands):
    _require(type(model) is GRUResidualRewardWorldModel, "Exact parent residual GRU class required")
    _require(type(model.hidden_size) is int and model.hidden_size > 0
             and type(model.dt) in (int, float) and math.isfinite(model.dt) and model.dt > 0
             and type(model.noise_std) in (int, float) and math.isfinite(model.noise_std)
             and model.noise_std >= 0 and type(model.residual_reward) is bool,
             "Finite parent constructor configuration required")
    parameter = next(model.parameters())
    _require(all(p.dtype == parameter.dtype and p.device == parameter.device
                 and bool(torch.isfinite(p).all()) for p in model.parameters()),
             "Finite aligned model parameters required")
    _require(isinstance(packets, Tensor) and packets.ndim == 3
             and packets.shape[0] > 0 and 1 <= packets.shape[1] <= MAX_STEPS + 1
             and packets.shape[2] == 8, "Public packets require [B,T,8], 1<=T<=51")
    batch, points, _ = packets.shape
    # The shared validator discards even nonfinite missing-angle placeholders
    # before checking known fields. Preserve the actual validity and age.
    public = _public_packet(model, packets.reshape(batch * points, 8)).reshape(batch, points, 8)
    _require(isinstance(commands, Tensor) and commands.shape == (batch, points - 1, 2)
             and commands.dtype == parameter.dtype and commands.device == parameter.device
             and bool(torch.isfinite(commands).all()) and bool((commands.abs() <= 1).all()),
             "Finite issued commands [B,T-1,2] already within [-1,1] required")
    valid = public[..., 6] == 1
    _require(bool(valid[:, 0].all()), "First real packet must be visible")
    _require(bool((public[..., 7][valid] == 0).all()), "Visible public age must be zero")
    _require(torch.equal(public[..., 4:6], public[:, :1, 4:6].expand(-1, points, -1)),
             "Static public target changed")
    indices = torch.arange(points, device=parameter.device).expand(batch, -1)
    newest = torch.where(valid, indices, -1).cummax(1).values
    prior_newest = torch.cat((torch.zeros_like(newest[:, :1]), newest[:, :-1]), 1)
    older = torch.where(valid, prior_newest, -1).cummax(1).values
    expected_age = (indices - newest).to(public.dtype) * model.dt
    _require(bool(torch.isclose(public[..., 7], expected_age, rtol=AGE_RTOL, atol=AGE_ATOL).all()),
             "Public age disagrees with actual measurement indices")
    # Check every prefix, not merely the final (potentially reacquired) suffix.
    _require(bool((indices - older < WINDOW).all()),
             "Two-observation suffix exceeds twelve packets/eleven commands")
    return public, older, newest


def stream_roots(model: GRUResidualRewardWorldModel, packets: Tensor,
                 issued_commands: Tensor) -> StreamingRoots:
    """Return every real root from one fixed-parameter public sequence.

    There are at most fifty actual commands. Startup is visible. Missing slots
    retain their real age and validity but have zero angular inputs. Each prefix
    must fit the same last-two-valid twelve-packet bound as the original model.
    The newest branch resets only at actual visible packets; no detach or
    in-place update cuts gradients. The result is not a deployment checkpoint.

    All neural batches keep size B. Startup pays one parent assimilation; each
    later packet pays two full parent advances and three assimilations, even
    when a fresh branch or an old branch is discarded. No conditional batching
    or scientific random inputs are introduced.
    """
    public, older_anchor, newest_anchor = _inputs(model, packets, issued_commands)
    batch, points, _ = public.shape
    zero = model.initial(batch, public.device)
    older = model.assimilate(zero, public[:, 0])
    newest = older  # Safe shared prefix: all parent operations return new states.
    roots = [older]
    for point in range(1, points):
        action, packet = issued_commands[:, point - 1], public[:, point]
        older_advanced, _, _ = model.advance(older, action)
        newest_advanced, _, _ = model.advance(newest, action)
        old_updated = model.assimilate(older_advanced, packet)
        new_updated = model.assimilate(newest_advanced, packet)
        fresh = model.assimilate(zero, packet)
        visible = packet[:, 6:7] == 1
        older = {name: torch.where(visible, new_updated[name], old_updated[name]) for name in zero}
        newest = {name: torch.where(visible, fresh[name], new_updated[name]) for name in zero}
        roots.append(older)
    stacked = {name: torch.stack([root[name] for root in roots], dim=1) for name in zero}
    _require(all(bool(torch.isfinite(value).all()) for value in stacked.values()),
             "Nonfinite streamed root")
    observations, advances = 1 + 3 * (points - 1), 2 * (points - 1)
    return StreamingRoots(
        roots=stacked, older_anchor=older_anchor, newest_anchor=newest_anchor,
        work={"batch_size": batch, "real_packets_per_case": points,
              "parent_assimilate_calls": observations, "parent_advance_calls": advances,
              "observation_update_samples": batch * observations,
              "transition_samples": batch * advances,
              "observation_head_samples": batch * advances,
              "reward_head_samples": batch * advances,
              "expected_action_cost_samples": batch * advances if model.residual_reward else 0},
    )
