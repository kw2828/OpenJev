"""Functional Torch pose geometry, with body-frame increments per time step.

Positions/displacements use meters, Euler/rotation vectors use radians. Rotation
matrices map body coordinates into world coordinates. Euler XYZ means roll,
pitch, yaw with R = Rz(yaw) @ Ry(pitch) @ Rx(roll). No time division, SciPy,
data access, model calls, random draws, or in-place updates are performed.

Inputs must have exactly matching batch shapes, float32/64 dtype and device.
Matrix inputs are assumed to be proper rotations: shape/finiteness are checked,
but matrices are not projected onto SO(3) or certified as orthogonal here.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _check(value: Tensor, tail: tuple[int, ...], name: str):
    if not (isinstance(value, Tensor) and value.dtype in (torch.float32, torch.float64)
            and value.ndim >= len(tail) and tuple(value.shape[-len(tail):]) == tail
            and bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be finite float32/64 with trailing shape {tail}')


def _matched(*items):
    """Each entry is (tensor, trailing shape, name); broadcasting is forbidden."""
    base, tail, _ = items[0]
    for value, shape, name in items:
        _check(value, shape, name)
        if (value.shape[:-len(shape)] != base.shape[:-len(tail)]
                or value.dtype != base.dtype or value.device != base.device):
            raise ValueError('Batch shapes, dtype and device must match exactly')


def _skew(vector: Tensor):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def _sine_axis(matrix: Tensor):
    return .5 * torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2],
                            matrix[..., 0, 2] - matrix[..., 2, 0],
                            matrix[..., 1, 0] - matrix[..., 0, 1]), -1)


def _angle_parts(matrix: Tensor):
    sine_axis = _sine_axis(matrix)
    sine = torch.linalg.vector_norm(sine_axis, dim=-1)
    cosine = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1, 1)
    return torch.atan2(sine, cosine), sine_axis, sine, cosine


def euler_xyz_to_matrix(euler: Tensor) -> Tensor:
    """Roll/pitch/yaw [...,3] -> Rz(yaw) Ry(pitch) Rx(roll) [...,3,3]."""
    _check(euler, (3,), 'Euler angles')
    roll, pitch, yaw = euler.unbind(-1)
    sr, sp, sy = roll.sin(), pitch.sin(), yaw.sin()
    cr, cp, cy = roll.cos(), pitch.cos(), yaw.cos()
    return torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr), -1).reshape(*euler.shape[:-1], 3, 3)


def so3_exp(rotation_vector: Tensor) -> Tensor:
    """Rodrigues exponential [...,3] -> [...,3,3], stable at zero.

    torch.sinc uses sin(pi*x)/(pi*x), including its analytic value at zero.
    The second Rodrigues coefficient is .5*sinc(theta/(2*pi))**2, so neither
    coefficient divides by a vanishing angle. Vectors beyond pi are supported;
    the logarithm returns the corresponding principal rotation instead.
    """
    _check(rotation_vector, (3,), 'Rotation vector')
    theta = torch.linalg.vector_norm(rotation_vector, dim=-1)
    first = torch.sinc(theta / math.pi)
    second = .5 * torch.sinc(theta / (2 * math.pi)).square()
    skew = _skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + first[..., None, None] * skew + second[..., None, None] * (skew @ skew)


def so3_log(matrix: Tensor) -> Tensor:
    """Principal rotation vector, norm in [0,pi], with stable zero/pi branches.

    Away from pi, Log(R)=theta/sin(theta)*vee((R-R.T)/2). Near pi recover the
    axis from its symmetric outer product and the largest diagonal entry. The
    skew part chooses the axis sign where resolvable. At exactly pi the two
    signs represent the same rotation; the largest-diagonal positive-axis
    convention is deterministic for a matrix with exactly zero skew.

    No globally continuous/differentiable logarithm exists at the pi cut locus.
    This function has finite identity gradients; near-pi numerical stability
    does not imply a unique or smooth axis at exactly pi.
    """
    _check(matrix, (3, 3), 'Rotation matrix')
    theta, sine_axis, sine, cosine = _angle_parts(matrix)
    theta2 = theta.square()
    series = 1 + theta2 / 6 + 7 * theta2.square() / 360
    # Keep unused branches finite too: torch.where still evaluates both sides.
    safe_sine = torch.where(sine > 1e-4, sine, torch.ones_like(sine))
    factor = torch.where(theta < 1e-3, series, theta / safe_sine)
    ordinary = factor[..., None] * sine_axis

    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    outer = (matrix + matrix.transpose(-1, -2) - 2 * cosine[..., None, None] * identity)
    outer = outer / (2 * (1 - cosine).clamp_min(1e-4))[..., None, None]
    diagonal = outer.diagonal(dim1=-2, dim2=-1)
    index = diagonal.argmax(-1, keepdim=True)
    largest = diagonal.gather(-1, index).clamp_min(1e-8).sqrt()
    column = outer.gather(-1, index.unsqueeze(-2).expand(*index.shape[:-1], 3, 1)).squeeze(-1)
    axis = column / largest
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    sign = torch.where((axis * sine_axis).sum(-1, keepdim=True) < 0,
                       -torch.ones_like(largest), torch.ones_like(largest))
    near_pi = theta[..., None] * axis * sign
    return torch.where((cosine < -.99)[..., None], near_pi, ordinary)


def geodesic_angle(matrix: Tensor, other: Tensor | None = None) -> Tensor:
    """Shortest rotation angle in radians, optionally between two rotations.

    atan2(norm(vee(skew)), (trace-1)/2) avoids acos's infinite derivative at
    identity. Torch's norm supplies a finite zero subgradient there, including
    when this angle itself is used as a loss. The distance is mathematically
    nonsmooth at zero and pi; this is not a claim of global differentiability.
    """
    _check(matrix, (3, 3), 'Rotation matrix')
    if other is not None:
        _matched((matrix, (3, 3), 'Rotation matrix'), (other, (3, 3), 'Other rotation'))
        matrix = matrix.transpose(-1, -2) @ other
    return _angle_parts(matrix)[0]


def legacy_obs_to_pose(observation: Tensor) -> tuple[Tensor, Tensor]:
    """Raw [xyz, sin(roll,pitch,yaw), cos(roll,pitch,yaw)] -> (p,R).

    This expects unnormalized physical coordinates. Each sin/cos pair must be
    nondegenerate, but need not have exactly unit radius. atan2 recovers angles;
    the corresponding Euler matrix is a proper rotation. No angle unwrapping
    or hidden velocity is inferred.
    """
    _check(observation, (9,), 'Legacy observation')
    sine, cosine = observation[..., 3:6], observation[..., 6:9]
    radius = torch.linalg.vector_norm(torch.stack((sine, cosine), -1), dim=-1)
    if not bool((radius >= 1e-8).all()):
        raise ValueError('Legacy sine/cosine pairs must have radius >=1e-8')
    euler = torch.atan2(sine, cosine)
    return observation[..., :3].clone(), euler_xyz_to_matrix(euler)


def relative_body_increment(position: Tensor, rotation: Tensor,
                            next_position: Tensor, next_rotation: Tensor) -> tuple[Tensor, Tensor]:
    """Body displacement R.T*(p_next-p) and rotation Log(R.T*R_next)."""
    _matched((position, (3,), 'Position'), (rotation, (3, 3), 'Rotation'),
             (next_position, (3,), 'Next position'), (next_rotation, (3, 3), 'Next rotation'))
    inverse = rotation.transpose(-1, -2)
    displacement = (inverse @ (next_position - position).unsqueeze(-1)).squeeze(-1)
    return displacement, so3_log(inverse @ next_rotation)


def advance_pose(position: Tensor, rotation: Tensor, displacement_body: Tensor,
                 rotation_body: Tensor) -> tuple[Tensor, Tensor]:
    """p_next=p+R*d_body; R_next=R*Exp(w_body), with no state mutation."""
    _matched((position, (3,), 'Position'), (rotation, (3, 3), 'Rotation'),
             (displacement_body, (3,), 'Body displacement'), (rotation_body, (3,), 'Body rotation'))
    next_position = position + (rotation @ displacement_body.unsqueeze(-1)).squeeze(-1)
    return next_position, rotation @ so3_exp(rotation_body)


def so3_left_jacobian(rotation_vector: Tensor) -> Tensor:
    """SO(3) left Jacobian J(w) used in the SE(3) exponential's translation."""
    _check(rotation_vector, (3,), 'Rotation vector')
    theta = torch.linalg.vector_norm(rotation_vector, dim=-1)
    theta2 = theta.square()
    second = .5 * torch.sinc(theta / (2 * math.pi)).square()
    safe_theta = torch.where(theta >= .1, theta, torch.ones_like(theta))
    third = torch.where(theta < .1, 1 / 6 - theta2 / 120 + theta2.square() / 5040,
                        (theta - theta.sin()) / safe_theta.pow(3))
    skew = _skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + second[..., None, None] * skew + third[..., None, None] * (skew @ skew)


def constant_body_twist(position0: Tensor, rotation0: Tensor, position1: Tensor,
                        rotation1: Tensor, steps: Tensor, intervals: int) -> tuple[Tensor, Tensor]:
    """Extrapolate T1*Exp(h/intervals * Log(T0^-1*T1)) for each offset h.

    steps is a nonempty 1D nonnegative numeric tensor on the pose device. h=0
    returns T1; offsets count steps AFTER T1, not after T0. intervals counts the
    positive number of time steps between the two observed poses. Output has
    shape batch_shape+[H,3] and batch_shape+[H,3,3].

    Translation uses the SE(3) left Jacobian, not a rotated position secant.
    Endpoint data cannot disambiguate rotation winding past pi; principal SO(3)
    Log is used. This classical two-pose reference ignores supplied actions and
    assumes a constant body twist. No velocities or hidden state are queried.
    """
    _matched((position0, (3,), 'Initial position'), (rotation0, (3, 3), 'Initial rotation'),
             (position1, (3,), 'Latest position'), (rotation1, (3, 3), 'Latest rotation'))
    if not (type(intervals) is int and intervals > 0):
        raise ValueError('intervals must be a positive integer')
    if not (isinstance(steps, Tensor) and steps.ndim == 1 and steps.numel() > 0
            and steps.dtype in (torch.float32, torch.float64, torch.int32, torch.int64)
            and steps.device == position0.device and bool(torch.isfinite(steps).all())
            and bool((steps >= 0).all())):
        raise ValueError('steps must be a finite nonnegative1D numeric tensor on the pose device')
    displacement, angular = relative_body_increment(position0, rotation0, position1, rotation1)
    linear = torch.linalg.solve(so3_left_jacobian(angular), displacement.unsqueeze(-1)).squeeze(-1)
    factor = steps.to(dtype=position0.dtype) / intervals
    extrapolated_angular = angular.unsqueeze(-2) * factor[:, None]
    extrapolated_linear = linear.unsqueeze(-2) * factor[:, None]
    local_displacement = (so3_left_jacobian(extrapolated_angular) @ extrapolated_linear.unsqueeze(-1)).squeeze(-1)
    rotations = rotation1.unsqueeze(-3) @ so3_exp(extrapolated_angular)
    positions = position1.unsqueeze(-2) + (rotation1.unsqueeze(-3) @ local_displacement.unsqueeze(-1)).squeeze(-1)
    return positions, rotations
