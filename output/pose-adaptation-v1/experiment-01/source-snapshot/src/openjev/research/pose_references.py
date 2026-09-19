"""Classical pose forecasts and direct residual ridge, with no model or RNG.

R maps body to world. Position units follow the input; rotation vectors are
radians. References use context indices0..31 only. Each ridge horizon reads
actions16..30+h, including every recorded value in each40-dimensional block.
Rotation logs are principal logs; rotations beyond the pi cut cannot be
unwrapped from endpoints. No coordinate normalization or test-time fitting is
performed here. Returned tensors are owned outputs, not immutable objects.
"""
from __future__ import annotations

import torch
from torch import Tensor

from openjev.research.rigid_motion import constant_body_twist, so3_exp, so3_log

CONTEXT, HORIZON = 32, 25
REFERENCES = ("hold", "cv1", "cv16", "ls16", "body16")


def _inputs(p: Tensor, rotation: Tensor, actions: Tensor):
    if not isinstance(p, Tensor) or p.dtype not in (torch.float32, torch.float64):
        raise ValueError("positions must be float32/64 tensors")
    if p.ndim != 3 or p.shape[0] < 1 or tuple(p.shape[1:]) != (57, 3):
        raise ValueError("positions must have shape[B,57,3]")
    for name, value, shape in (("rotation", rotation, (p.shape[0], 57, 3, 3)),
                               ("actions", actions, (p.shape[0], 56, 40))):
        if (not isinstance(value, Tensor) or tuple(value.shape) != shape
                or value.dtype != p.dtype or value.device != p.device):
            raise ValueError(f"{name}: shape/dtype/device mismatch")
    if p.device.type != "cpu":
        raise ValueError("this reference implementation requires CPU inputs")
    # Future poses are neither checked nor consumed by prediction.
    if not bool(torch.isfinite(p[:, :CONTEXT]).all()):
        raise ValueError("nonfinite context position")
    if not bool(torch.isfinite(rotation[:, :CONTEXT]).all()):
        raise ValueError("nonfinite context rotation")


def _steps(p):
    return torch.arange(1, HORIZON + 1, dtype=p.dtype, device=p.device)


def _world_cv(p, rotation, start, steps):
    intervals = CONTEXT - 1 - start
    velocity = (p[:, 31] - p[:, start]) / intervals
    omega = so3_log(rotation[:, 31] @ rotation[:, start].transpose(-1, -2)) / intervals
    position = p[:, 31, None] + steps[None, :, None] * velocity[:, None]
    vectors = steps[None, :, None] * omega[:, None]
    return position, vectors


def predict_reference(name: str, p: Tensor, rotation: Tensor,
                      actions: Tensor) -> tuple[Tensor, Tensor]:
    """Return [B,25,3]/[B,25,3,3], retaining input dtype. No action values used."""
    _inputs(p, rotation, actions)
    if name not in REFERENCES:
        raise ValueError("unknown classical reference")
    steps = _steps(p)
    root = rotation[:, 31, None]
    if name == "hold":
        return p[:, 31:32].expand(-1, HORIZON, -1).clone(), root.expand(-1, HORIZON, -1, -1).clone()
    if name == "body16":
        return constant_body_twist(p[:, 16], rotation[:, 16], p[:, 31], rotation[:, 31], steps, 15)
    if name in ("cv1", "cv16"):
        position, vectors = _world_cv(p, rotation, 30 if name == "cv1" else 16, steps)
    else:
        offsets = torch.arange(-15, 1, dtype=p.dtype, device=p.device)
        weights = offsets / offsets.square().sum()
        velocity = ((p[:, 16:32] - p[:, 31, None]) * weights[None, :, None]).sum(1)
        relative = so3_log(rotation[:, 16:32] @ root.transpose(-1, -2))
        omega = (relative * weights[None, :, None]).sum(1)
        position = p[:, 31, None] + steps[None, :, None] * velocity[:, None]
        vectors = steps[None, :, None] * omega[:, None]
    return position, so3_exp(vectors) @ root


def ridge_features(p: Tensor, rotation: Tensor, actions: Tensor, horizon: int) -> Tensor:
    """Float64[B,96+(15+h)*40]; no future poses or actions after endpoint h."""
    _inputs(p, rotation, actions)
    if type(horizon) is not int or not 1 <= horizon <= HORIZON:
        raise ValueError("horizon must be an integer1..25")
    action_prefix = actions[:, 16:31 + horizon]
    if not bool(torch.isfinite(action_prefix).all()):
        raise ValueError("nonfinite required action prefix")
    # Convert before subtraction/logs so fitted ridge arithmetic is float64.
    positions = p[:, 16:32].double() - p[:, 31:32].double()
    relative = so3_log(rotation[:, 16:32].double()
                       @ rotation[:, 31:32].double().transpose(-1, -2))
    return torch.cat((positions.flatten(1), relative.flatten(1), action_prefix.double().flatten(1)), 1)


def _ridge_base(p, rotation, horizon):
    steps = torch.tensor([horizon], dtype=torch.float64, device=p.device)
    positions, vectors = _world_cv(p.double(), rotation.double(), 16, steps)
    return positions[:, 0], vectors[:, 0]


@torch.no_grad()
def fit_ridge16(p: Tensor, rotation: Tensor, actions: Tensor) -> list[dict[str, Tensor]]:
    """Fit25 independent six-output residual maps, unit ridge/unpenalized intercept.

    Targets are p_future-worldCV16 and Log(R_future R_root.T)-worldCV16_rotvec.
    Uses caller-supplied training trajectories only. No selection or data reads.
    """
    _inputs(p, rotation, actions)
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(rotation).all()):
        raise ValueError("nonfinite training target")
    result = []
    root = rotation[:, 31].double()
    for horizon in range(1, HORIZON + 1):
        features = ridge_features(p, rotation, actions, horizon)
        base_p, base_w = _ridge_base(p, rotation, horizon)
        target_p = p[:, 31 + horizon].double() - base_p
        target_w = so3_log(rotation[:, 31 + horizon].double() @ root.transpose(-1, -2)) - base_w
        targets = torch.cat((target_p, target_w), -1)
        mean_x, mean_y = features.mean(0), targets.mean(0)
        xc, yc = features - mean_x, targets - mean_y
        dual = torch.linalg.solve(xc @ xc.T + torch.eye(len(xc), dtype=torch.float64), yc)
        weight = xc.T @ dual
        intercept = mean_y - mean_x @ weight
        if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(intercept).all()):
            raise FloatingPointError("nonfinite ridge solution")
        result.append({"weight": weight.clone(), "intercept": intercept.clone()})
    return result


def predict_ridge16(weights: list[dict[str, Tensor]], p: Tensor, rotation: Tensor,
                    actions: Tensor) -> tuple[Tensor, Tensor]:
    """Return float64 predictions; targets/future poses never influence them."""
    _inputs(p, rotation, actions)
    if type(weights) is not list or len(weights) != HORIZON:
        raise ValueError("expected25 ridge horizon maps")
    positions, rotations = [], []
    root = rotation[:, 31].double()
    for horizon, mapping in enumerate(weights, 1):
        features = ridge_features(p, rotation, actions, horizon)
        if type(mapping) is not dict or set(mapping) != {"weight", "intercept"}:
            raise ValueError("invalid ridge map")
        for key, shape in (("weight", (features.shape[1], 6)), ("intercept", (6,))):
            value = mapping[key]
            if (not isinstance(value, Tensor) or value.dtype != torch.float64
                    or value.device != p.device or tuple(value.shape) != shape
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("invalid ridge tensor")
        correction = features @ mapping["weight"] + mapping["intercept"]
        base_p, base_w = _ridge_base(p, rotation, horizon)
        positions.append(base_p + correction[:, :3])
        rotations.append(so3_exp(base_w + correction[:, 3:]) @ root)
    return torch.stack(positions, 1), torch.stack(rotations, 1)
