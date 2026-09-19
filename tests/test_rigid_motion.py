"""Synthetic geometry only, with isolated Torch seed410 for roundtrip coverage."""
import math

import pytest
import torch

from openjev.research.rigid_motion import (
    advance_pose,
    constant_body_twist,
    euler_xyz_to_matrix,
    geodesic_angle,
    legacy_obs_to_pose,
    relative_body_increment,
    so3_exp,
    so3_left_jacobian,
    so3_log,
)


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield


def tolerances(dtype):
    return {'rtol': 2e-5, 'atol': 2e-6} if dtype == torch.float32 else {'rtol': 2e-10, 'atol': 2e-11}


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_shape', [(), (7,), (2, 3)])
def test_exp_log_roundtrip_random_batched(dtype, batch_shape):
    axis = torch.randn(*batch_shape, 3, dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    angles = torch.rand(batch_shape, dtype=dtype) * (math.pi - .01)
    vector = axis * angles[..., None]
    matrix = so3_exp(vector)
    torch.testing.assert_close(so3_log(matrix), vector, **tolerances(dtype))
    torch.testing.assert_close(so3_exp(so3_log(matrix)), matrix, **tolerances(dtype))
    identity = torch.eye(3, dtype=dtype).expand_as(matrix)
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, identity, **tolerances(dtype))
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(batch_shape, dtype=dtype), **tolerances(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('angle', [0., 1e-12, 1e-7, 1e-4, .01, math.pi - .01, math.pi - 1e-4, math.pi])
def test_zero_and_pi_roundtrip_on_different_axes(dtype, angle):
    axis = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.],
                         [1., -2., 3.], [-1., -1., -1.]], dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    vector = axis * angle
    matrix = so3_exp(vector)
    logged = so3_log(matrix)
    torch.testing.assert_close(so3_exp(logged), matrix, **tolerances(dtype))
    torch.testing.assert_close(torch.linalg.vector_norm(logged, dim=-1),
                               torch.full((5,), angle, dtype=dtype), **tolerances(dtype))
    if angle < math.pi - 1e-5:
        torch.testing.assert_close(logged, vector, **tolerances(dtype))


def test_float64_resolves_axis_sign_just_before_pi():
    axis = torch.tensor([-.3, .4, -.5], dtype=torch.float64)
    axis /= axis.norm()
    vector = axis * (math.pi - 1e-10)
    torch.testing.assert_close(so3_log(so3_exp(vector)), vector, rtol=1e-9, atol=1e-9)


def test_exact_pi_zero_skew_is_finite_and_deterministic():
    matrix = torch.diag(torch.tensor([-1., 1., -1.], dtype=torch.float64))
    expected = torch.tensor([0., math.pi, 0.], dtype=torch.float64)
    torch.testing.assert_close(so3_log(matrix), expected)
    assert torch.equal(so3_log(matrix), so3_log(matrix))


def test_euler_known_axes_and_noncommuting_xyz_order():
    dtype = torch.float64
    axes = torch.eye(3, dtype=dtype) * (math.pi / 2)
    torch.testing.assert_close(euler_xyz_to_matrix(axes), so3_exp(axes), atol=1e-15, rtol=1e-15)
    angles = torch.tensor([.2, -.4, .7], dtype=dtype)
    rotations = so3_exp(torch.eye(3, dtype=dtype) * angles[:, None])
    expected = rotations[2] @ rotations[1] @ rotations[0]
    actual = euler_xyz_to_matrix(angles)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert not torch.allclose(actual, rotations[0] @ rotations[1] @ rotations[2])
    yaw = euler_xyz_to_matrix(torch.tensor([0., 0., math.pi / 2], dtype=dtype))
    torch.testing.assert_close(yaw @ torch.tensor([1., 0., 0.], dtype=dtype),
                               torch.tensor([0., 1., 0.], dtype=dtype), atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_body_increment_inverse_and_global_frame_equivariance(dtype):
    p = torch.tensor([[1., 2., -3.], [-1., 3., 4.]], dtype=dtype)
    rotation = euler_xyz_to_matrix(torch.tensor([[.2, -.4, .3], [-.2, .7, -.1]], dtype=dtype))
    d = torch.tensor([[.3, .2, -.1], [.1, -.5, .2]], dtype=dtype)
    w = torch.tensor([[.3, -.2, .1], [.5, .2, -.3]], dtype=dtype)
    next_p, next_r = advance_pose(p, rotation, d, w)
    d_back, w_back = relative_body_increment(p, rotation, next_p, next_r)
    torch.testing.assert_close(d_back, d, **tolerances(dtype))
    torch.testing.assert_close(w_back, w, **tolerances(dtype))
    global_r = so3_exp(torch.tensor([-.4, .3, .1], dtype=dtype)).expand(2, -1, -1)
    offset = torch.tensor([8., -3., 5.], dtype=dtype)
    transformed_p = (global_r @ p.unsqueeze(-1)).squeeze(-1) + offset
    transformed_r = global_r @ rotation
    actual_p, actual_r = advance_pose(transformed_p, transformed_r, d, w)
    expected_p = (global_r @ next_p.unsqueeze(-1)).squeeze(-1) + offset
    torch.testing.assert_close(actual_p, expected_p, **tolerances(dtype))
    torch.testing.assert_close(actual_r, global_r @ next_r, **tolerances(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_identity_gradients_finite_and_angle_zero_subgradient(dtype):
    vector = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    rotation = so3_exp(vector)
    loss = so3_log(rotation).sum() + so3_left_jacobian(vector).sum()
    loss.backward()
    assert torch.isfinite(vector.grad).all()
    torch.testing.assert_close(vector.grad, torch.ones_like(vector), **tolerances(dtype))
    vector2 = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    angle = geodesic_angle(so3_exp(vector2))
    angle.sum().backward()
    assert torch.equal(angle, torch.zeros(2, dtype=dtype))
    assert torch.equal(vector2.grad, torch.zeros_like(vector2))
    matrix = torch.eye(3, dtype=dtype).requires_grad_()
    geodesic_angle(matrix).backward()
    assert torch.isfinite(matrix.grad).all() and torch.equal(matrix.grad, torch.zeros_like(matrix))


def test_numerical_gradients_away_from_cut_locus():
    vector = torch.tensor([.2, -.3, .4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: so3_log(so3_exp(value)), (vector,))
    assert torch.autograd.gradcheck(so3_exp, (vector,))
    near_pi = torch.tensor([math.pi - .02, .01, -.01], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: so3_log(so3_exp(value)), (near_pi,), atol=1e-5)


def test_legacy_observation_convention_scale_and_owned_outputs():
    angles = torch.tensor([[.4, -.3, 2.7], [-2.4, .2, -.6]], dtype=torch.float64)
    position = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    observation = torch.cat((position, 2 * angles.sin(), 2 * angles.cos()), -1).requires_grad_()
    p, rotation = legacy_obs_to_pose(observation)
    torch.testing.assert_close(p, position)
    torch.testing.assert_close(rotation, euler_xyz_to_matrix(angles))
    (p.sum() + rotation.sum()).backward()
    assert torch.isfinite(observation.grad).all()
    p.detach().fill_(99.)
    assert torch.equal(observation[:, :3], position)


def test_private_pose_branches_never_mutate_inputs_or_siblings():
    p = torch.zeros(2, 3)
    rotation = torch.eye(3).expand(2, -1, -1).clone()
    d, w = torch.ones(2, 3) * .1, torch.ones(2, 3) * .2
    p_saved, r_saved, d_saved, w_saved = (value.clone() for value in (p, rotation, d, w))
    left_p, left_r = advance_pose(p, rotation, d, w)
    right_p, right_r = advance_pose(p, rotation, -d, -w)
    right_saved = (right_p.clone(), right_r.clone())
    left_p.fill_(12.)
    left_r.fill_(23.)
    for actual, expected in zip((p, rotation, d, w, right_p, right_r),
                                (p_saved, r_saved, d_saved, w_saved, *right_saved), strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_constant_twist_pure_translation_and_pure_rotation(dtype):
    p0 = torch.tensor([[1., 2., 3.]], dtype=dtype)
    r0 = euler_xyz_to_matrix(torch.tensor([[.3, .2, -.4]], dtype=dtype))
    delta = torch.tensor([[.2, -.1, .3]], dtype=dtype)
    steps = torch.tensor([0, 1, 2, 5], dtype=torch.int64)
    p1 = p0 + 3 * delta
    pred_p, pred_r = constant_body_twist(p0, r0, p1, r0, steps, 3)
    torch.testing.assert_close(pred_p, p1[:, None] + steps[None, :, None] * delta[:, None], **tolerances(dtype))
    torch.testing.assert_close(pred_r, r0[:, None].expand_as(pred_r), **tolerances(dtype))
    w = torch.tensor([[0., 0., .12]], dtype=dtype)
    r1 = r0 @ so3_exp(3 * w)
    pred_p, pred_r = constant_body_twist(p0, r0, p0, r1, steps, 3)
    torch.testing.assert_close(pred_p, p0[:, None].expand_as(pred_p), **tolerances(dtype))
    expected = r0[:, None] @ so3_exp((steps + 3)[None, :, None] * w[:, None])
    torch.testing.assert_close(pred_r, expected, **tolerances(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_constant_twist_reproduces_independent_analytic_screw(dtype):
    # A world-z screw: spatial origin circles while rising .2 per radian.
    # Its body twist has constant v=(0,1,.2), w=(0,0,1).
    def screw(angle):
        angle = torch.as_tensor(angle, dtype=dtype)
        position = torch.stack((angle.cos(), angle.sin(), .2 * angle), -1)
        euler = torch.stack((torch.zeros_like(angle), torch.zeros_like(angle), angle), -1)
        return position, euler_xyz_to_matrix(euler)
    p0, r0 = screw(.2)
    p1, r1 = screw(.6)
    steps = torch.tensor([0., 1., 2., 7.], dtype=dtype)
    pred_p, pred_r = constant_body_twist(p0, r0, p1, r1, steps, 4)
    expected_p, expected_r = screw(.6 + .1 * steps)
    torch.testing.assert_close(pred_p, expected_p, **tolerances(dtype))
    torch.testing.assert_close(pred_r, expected_r, **tolerances(dtype))
    # Rotating a fixed secant is not the same finite SE(3) exponential.
    wrong = p1 + steps[-1] * (r1 @ r0.T @ (p1 - p0)) / 4
    assert torch.linalg.vector_norm(pred_p[-1] - wrong) > .05


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_constant_twist_stationary_gradients_finite_and_batch_shape(dtype):
    p0 = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    p1 = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    w0 = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    w1 = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    steps = torch.arange(1, 26)
    pred_p, pred_r = constant_body_twist(p0, so3_exp(w0), p1, so3_exp(w1), steps, 15)
    assert pred_p.shape == (2, 25, 3) and pred_r.shape == (2, 25, 3, 3)
    (pred_p.sum() + pred_r.sum()).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in (p0, p1, w0, w1))


@pytest.mark.parametrize('function,value', [(so3_exp, torch.ones(3, dtype=torch.int64)),
                                          (so3_exp, torch.ones(4)),
                                          (so3_log, torch.ones(3, 2)),
                                          (euler_xyz_to_matrix, torch.full((3,), float('nan'))),
                                          (legacy_obs_to_pose, torch.zeros(9))])
def test_reject_invalid_inputs(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_no_implicit_batch_broadcasting_or_dtype_conversion():
    p = torch.zeros(2, 3)
    r = torch.eye(3)
    with pytest.raises(ValueError, match='match exactly'):
        advance_pose(p, r, p, p)
    with pytest.raises(ValueError, match='match exactly'):
        advance_pose(p, r.expand(2, -1, -1), p.double(), p)


@pytest.mark.parametrize('steps,intervals', [(torch.tensor([-1]), 2), (torch.tensor([1]), 0),
                                           (torch.tensor([1]), True), (torch.tensor([float('nan')]), 3),
                                           (torch.ones(2, 1), 1)])
def test_constant_twist_rejects_invalid_offsets(steps, intervals):
    with pytest.raises(ValueError):
        constant_body_twist(torch.zeros(3), torch.eye(3), torch.zeros(3), torch.eye(3), steps, intervals)
