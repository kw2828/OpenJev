"""Approximate Reacher geometry scoring from decoded public angle features.

This prospective component makes no environment, model or random calls. It is
not a native-reward implementation: the pinned RK4 environment's cached body
positions need not equal final-qpos forward kinematics. Nor does the distance
of a projected prediction equal expected distance under actuator uncertainty.
Only the actuator penalty below is an analytical expectation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from openjev.research.reacher_reward_residual import expected_clipped_action_cost

MIN_PAIR_NORM = 1e-6
LINK_OFFSETS_METERS = (0.10, 0.11)


def _noise_std(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("noise_std must be a fixed finite nonnegative real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("noise_std must be a fixed finite nonnegative real scalar") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError("noise_std must be a fixed finite nonnegative real scalar")
    return result


def geometry_reward_configuration(noise_std: float) -> dict:
    """Return fresh, JSON-ready semantics; no tensor construction or RNG use."""
    return {
        "version": "reacher-geometry-reward-v1",
        "input_order": ["cos_q0", "cos_q1", "sin_q0", "sin_q1"],
        "projection": "atan2(sin_q, cos_q), independently for each joint",
        "minimum_pair_norm": MIN_PAIR_NORM,
        "norm_boundary": "accept >= threshold; reject below, no fallback",
        "link_offsets_meters": list(LINK_OFFSETS_METERS),
        "geometry_source": "reacher.xml body1.pos.x=0.10; fingertip.pos.x=0.11",
        "score": "-planar_fingertip_distance - expected_clipped_action_cost",
        "actuator_cost_function": "openjev.research.reacher_reward_residual.expected_clipped_action_cost",
        "noise_std": _noise_std(noise_std),
        "distance_weight": 1.0,
        "control_weight": 1.0,
        "issued_command_range": [-1.0, 1.0],
        "joint_limit_penalty": None,
        "reward_clipping": None,
        "tensor_contract": "matching leading shapes, CPU, common float32 or float64 dtype",
        "native_reward_exact": False,
        "limitation": "Final-angle FK approximates RK4 cached-body distance; projected distance is not expected distance.",
        "run_status_authority": "enclosing protocol and execution receipts",
    }


@dataclass(frozen=True)
class GeometryRewardComponents:
    """New output tensors, never aliases of the supplied observations/actions."""

    joint_angles: Tensor
    pair_norms: Tensor
    fingertip: Tensor
    distance: Tensor
    action_cost: Tensor
    reward: Tensor


def _validate_inputs(predicted4: Tensor, public_target: Tensor, issued_command: Tensor) -> None:
    values = (("predicted4", predicted4, 4), ("public_target", public_target, 2),
              ("issued_command", issued_command, 2))
    for name, value, size in values:
        if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} must be a float32 or float64 tensor")
        if value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError(f"{name} must be a strided CPU tensor; no implicit device fallback")
        if value.ndim < 1 or value.shape[-1] != size or value.numel() == 0:
            raise ValueError(f"{name} must be nonempty with last dimension {size}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if public_target.dtype != predicted4.dtype or issued_command.dtype != predicted4.dtype:
        raise ValueError("All inputs must share the same dtype")
    if public_target.shape[:-1] != predicted4.shape[:-1] or issued_command.shape[:-1] != predicted4.shape[:-1]:
        raise ValueError("All inputs must have exactly matching leading shapes; broadcasting is forbidden")
    if bool((issued_command.abs() > 1.0).any()):
        raise ValueError("issued_command must already lie in [-1,1]; candidate values are not changed")


def geometry_reward_components(predicted4: Tensor, public_target: Tensor,
                               issued_command: Tensor, noise_std: float) -> GeometryRewardComponents:
    """Project [...,4] decoded features and score matching [...,2] goals/actions.

    Pair norms must be at least 1e-6. Positive radial scaling therefore does not
    affect the score, provided inputs remain admissible. Projected joints may
    fall outside the native joint limits: no hidden penalty or correction is
    applied. Issued commands must already lie in [-1,1]. The reused analytical
    actuator-cost function accounts for clipping after adding noise and charges
    the expected applied squared norm exactly once; candidates are not changed.

    Output dtype is preserved. CPU is explicit because the existing actuator
    function needs float64; this helper never silently copies neural data from
    another device. Autograd is retained, except for the input-validity checks.
    The total reward is deliberately not clipped to the planner's score range.
    """
    sigma = _noise_std(noise_std)
    _validate_inputs(predicted4, public_target, issued_command)
    cosine, sine = predicted4[..., :2], predicted4[..., 2:]
    norms = torch.hypot(cosine, sine)
    if not bool(torch.isfinite(norms).all()) or bool((norms < MIN_PAIR_NORM).any()):
        raise ValueError("Angle-pair norms must be finite and at least 1e-6; no fallback")
    angles = torch.atan2(sine, cosine)
    q0, q01 = angles[..., 0], angles.sum(-1)
    l0, l1 = LINK_OFFSETS_METERS
    tip = torch.stack((l0 * torch.cos(q0) + l1 * torch.cos(q01),
                       l0 * torch.sin(q0) + l1 * torch.sin(q01)), dim=-1)
    distance = torch.linalg.vector_norm(tip - public_target, dim=-1)
    action_cost = expected_clipped_action_cost(issued_command, sigma)
    reward = -distance - action_cost
    if not bool(torch.isfinite(distance).all()) or not bool(torch.isfinite(reward).all()):
        raise ValueError("Geometry reward overflowed; no fallback")
    return GeometryRewardComponents(angles, norms, tip, distance, action_cost, reward)


def geometry_reward(predicted4: Tensor, public_target: Tensor,
                    issued_command: Tensor, noise_std: float) -> Tensor:
    """Return the approximate per-item reward, with explicit geometry semantics."""
    return geometry_reward_components(predicted4, public_target, issued_command, noise_std).reward
