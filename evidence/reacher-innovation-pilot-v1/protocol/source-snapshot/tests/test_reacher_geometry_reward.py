"""Synthetic analytical fixtures only, not native or learned-policy results."""

import json
import math

import pytest
import torch

from openjev.research.reacher_geometry_reward import (
    MIN_PAIR_NORM,
    geometry_reward,
    geometry_reward_components,
    geometry_reward_configuration,
)
from openjev.research.reacher_reward_residual import expected_clipped_action_cost


def features(angles):
    angles = torch.as_tensor(angles, dtype=torch.float64)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def inputs(dtype=torch.float64):
    return (torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=dtype),
            torch.tensor([[0.0, 0.0]], dtype=dtype),
            torch.tensor([[0.0, 0.0]], dtype=dtype))


@pytest.mark.parametrize("angles,expected", [
    ([0.0, 0.0], [0.21, 0.0]),
    ([math.pi / 2, 0.0], [0.0, 0.21]),
    ([0.0, math.pi / 2], [0.10, 0.11]),
    ([math.pi / 2, -math.pi / 2], [0.11, 0.10]),
    ([0.0, math.pi], [-0.01, 0.0]),
])
def test_independent_analytic_postures(angles, expected):
    prediction = features(angles)
    target = torch.tensor([0.03, -0.04], dtype=torch.float64)
    result = geometry_reward_components(prediction, target, torch.zeros_like(target), 0.0)
    torch.testing.assert_close(result.fingertip, torch.tensor(expected, dtype=torch.float64),
                               rtol=0, atol=3e-17)
    distance = math.hypot(expected[0] - 0.03, expected[1] + 0.04)
    assert result.reward.item() == pytest.approx(-distance, abs=5e-17)
    assert result.action_cost.item() == 0


def test_projection_is_unit_circle_and_independent_positive_radial_scale():
    prediction = features([[0.3, -1.1], [2.8, 0.6]])
    target = torch.tensor([[0.05, -0.01], [-0.02, 0.03]], dtype=torch.float64)
    action = torch.tensor([[0.2, -0.3], [1.0, -1.0]], dtype=torch.float64)
    scale = torch.tensor([[0.03, 17.0], [2.0, 0.005]], dtype=torch.float64)
    scaled = prediction * torch.cat((scale, scale), dim=-1)
    base = geometry_reward_components(prediction, target, action, 0.05)
    changed = geometry_reward_components(scaled, target, action, 0.05)
    torch.testing.assert_close(base.reward, changed.reward, rtol=0, atol=1e-15)
    torch.testing.assert_close(base.fingertip, changed.fingertip, rtol=0, atol=1e-15)
    torch.testing.assert_close(changed.pair_norms, scale, rtol=1e-15, atol=1e-15)
    unit = changed.joint_angles.cos().square() + changed.joint_angles.sin().square()
    torch.testing.assert_close(unit, torch.ones_like(unit), rtol=0, atol=2e-16)
    # Positive radial scaling is the invariance; changing signs changes angles.
    assert not torch.allclose(base.fingertip, geometry_reward_components(-prediction, target, action, .05).fingertip)


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.8, 10.0])
def test_existing_expected_actuator_cost_charged_once(sigma):
    prediction = features([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[0.21, 0.0], [0.21, 0.0]], dtype=torch.float64)
    action = torch.tensor([[0.3, -0.4], [1.0, -1.0]], dtype=torch.float64)
    result = geometry_reward_components(prediction, target, action, sigma)
    expected = expected_clipped_action_cost(action, sigma)
    torch.testing.assert_close(result.distance, torch.zeros_like(expected), rtol=0, atol=3e-17)
    torch.testing.assert_close(result.action_cost, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.reward, -result.distance - expected, rtol=0, atol=0)
    if sigma == 0:
        torch.testing.assert_close(result.reward, torch.tensor([-.25, -2.], dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_leading_dimensions_noncontiguous_inputs_dtype_and_no_mutation(dtype):
    prediction, target, action = inputs(dtype)
    prediction = prediction.repeat(2, 3, 1).transpose(0, 1)
    target = target.repeat(2, 3, 1).transpose(0, 1)
    action = action.repeat(2, 3, 1).transpose(0, 1)
    originals = [x.clone() for x in (prediction, target, action)]
    rng = torch.random.get_rng_state().clone()
    result = geometry_reward_components(prediction, target, action, .05)
    assert result.reward.shape == (3, 2) and result.fingertip.shape == (3, 2, 2)
    for value in vars(result).values():
        assert value.dtype == dtype and value.device.type == "cpu"
        assert all(value.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
                   for x in (prediction, target, action))
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
    for actual, original in zip((prediction, target, action), originals, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(geometry_reward(prediction, target, action, .05), result.reward, rtol=0, atol=0)


def test_unclipped_reward_and_no_joint_limit_penalty():
    # q1=pi lies outside native [-3,3], but the pure geometry has no penalty.
    prediction = features([0.0, math.pi])
    target = torch.tensor([-0.01, 10.0], dtype=torch.float64)
    action = torch.tensor([1.0, -1.0], dtype=torch.float64)
    result = geometry_reward_components(prediction, target, action, 0)
    assert result.reward.item() == pytest.approx(-12.0)
    target = torch.tensor([-0.01, 0.0], dtype=torch.float64)
    assert geometry_reward(prediction, target, torch.zeros_like(action), 0).item() == pytest.approx(0., abs=3e-17)


def test_gradients_match_finite_differences_and_radial_direction_has_zero_derivative():
    prediction = (features([[0.2, -0.7]]) * 1.7).requires_grad_()
    target = torch.tensor([[0.02, -0.04]], dtype=torch.float64, requires_grad=True)
    action = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p, t, a: geometry_reward(p, t, a, .05),
                                   (prediction, target, action))
    gradient = torch.autograd.grad(geometry_reward(prediction, target, action, .05).sum(), prediction)[0]
    radial = gradient[..., :2] * prediction[..., :2] + gradient[..., 2:] * prediction[..., 2:]
    torch.testing.assert_close(radial, torch.zeros_like(radial), rtol=0, atol=1e-15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_threshold_accepts_boundary_rejects_lower_and_zero(dtype):
    prediction, target, action = inputs(dtype)
    prediction[..., :2] = MIN_PAIR_NORM
    geometry_reward(prediction, target, action, 0)
    prediction[..., 0] = MIN_PAIR_NORM / 2
    with pytest.raises(ValueError, match="norms"):
        geometry_reward(prediction, target, action, 0)
    prediction[..., 0] = 0
    with pytest.raises(ValueError, match="norms"):
        geometry_reward(prediction, target, action, 0)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_all_nonfinite_inputs_rejected(position, invalid):
    values = list(inputs())
    values[position].flatten()[0] = invalid
    with pytest.raises(ValueError, match="finite"):
        geometry_reward(*values, .05)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float16, torch.bfloat16, torch.complex64])
def test_unsupported_dtypes_rejected(position, dtype):
    values = list(inputs())
    values[position] = values[position].to(dtype)
    with pytest.raises(ValueError, match="float32 or float64"):
        geometry_reward(*values, .05)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_non_tensor_scalar_empty_wrong_shapes_and_device_rejected(position):
    for invalid in ([1., 1.], torch.tensor(0.), torch.empty(0, 4 if position == 0 else 2),
                    torch.zeros(1, 3), torch.empty((1, 4 if position == 0 else 2), device="meta")):
        values = list(inputs())
        values[position] = invalid
        with pytest.raises(ValueError):
            geometry_reward(*values, .05)


@pytest.mark.parametrize("position", [1, 2])
def test_no_broadcasts_or_dtype_promotion(position):
    values = list(inputs())
    values[position] = values[position][0]
    with pytest.raises(ValueError, match="leading shapes"):
        geometry_reward(*values, .05)
    values = list(inputs())
    values[position] = values[position].float()
    with pytest.raises(ValueError, match="same dtype"):
        geometry_reward(*values, .05)


@pytest.mark.parametrize("invalid", [None, True, -0.1, float("nan"), float("inf"), "0.05", torch.tensor(.05)])
def test_noise_is_fixed_finite_scalar_configuration(invalid):
    with pytest.raises((ValueError, TypeError), match="noise_std"):
        geometry_reward(*inputs(), invalid)
    with pytest.raises((ValueError, TypeError), match="noise_std"):
        geometry_reward_configuration(invalid)


def test_derived_overflow_rejected():
    prediction, target, action = inputs(torch.float32)
    target[:] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="overflowed"):
        geometry_reward(prediction, target, action, .05)


@pytest.mark.parametrize("invalid", [1.000001, -1.000001, 2., -3.])
def test_out_of_range_issued_commands_rejected_not_clipped(invalid):
    prediction, target, action = inputs()
    action[0, 1] = invalid
    with pytest.raises(ValueError, match="already lie"):
        geometry_reward(prediction, target, action, .05)
    assert action[0, 1].item() == invalid


def test_configuration_binds_approximation_and_returns_independent_values():
    first = geometry_reward_configuration(.05)
    assert json.loads(json.dumps(first)) == first
    assert first["minimum_pair_norm"] == 1e-6
    assert first["native_reward_exact"] is False
    assert first["reward_clipping"] is None and first["joint_limit_penalty"] is None
    assert first["link_offsets_meters"] == [.10, .11]
    assert first["noise_std"] == .05
    first["link_offsets_meters"][0] = 100
    assert geometry_reward_configuration(.05)["link_offsets_meters"] == [.10, .11]
