"""Handwritten NumPy caches only: no random draws, models or native calls."""
import json
import math
from itertools import pairwise

import numpy as np
import pytest

from openjev.research.pose_constant import fit_constant


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def z(angle):
    return rotation([0, 0, 1], angle)


def poses(fast=None, slow=None, target=None, dtype=np.float64):
    p = np.zeros((1, 1, 3), dtype=dtype)
    rs = [np.eye(3) if r is None else r for r in (fast, slow, target)]
    return p.copy(), rs[0][None, None].astype(dtype), p.copy(), rs[1][None, None].astype(dtype), p.copy(), rs[2][None, None].astype(dtype)


def angle_between(a, b):
    relative = a @ b.T
    skew = .5 * np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                          relative[1, 0] - relative[0, 1]])
    return math.atan2(np.linalg.norm(skew), np.clip((np.trace(relative) - 1) / 2, -1, 1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_analytic_position_uses_all_windows_and_coordinates(dtype):
    data = list(poses(dtype=dtype))
    delta = np.array([[[1, 0, 0], [0, 2, 0]], [[-3, 0, 0], [0, 0, 4]]], dtype=dtype)
    slow = np.full_like(delta, 2)
    data[0], data[2], data[4] = slow + delta, slow, slow + .25 * delta
    data[1] = data[3] = data[5] = np.broadcast_to(np.eye(3, dtype=dtype), (2, 2, 3, 3)).copy()
    result = fit_constant(*data)
    assert result["alpha"] == [.25, 0.]
    assert result["position"]["denominator"] == 30
    assert result["position"]["numerator"] == 7.5
    assert result["position"]["objective_m2"] == 0
    assert result["examples"] == 4


@pytest.mark.parametrize("target,expected", [(-2., 0.), (2., 1.)])
def test_analytic_position_clips_endpoints(target, expected):
    data = list(poses())
    data[0][0, 0, 0] = 1
    data[4][0, 0, 0] = target
    record = fit_constant(*data)
    assert record["alpha"][0] == expected
    assert record["position"]["unclipped_alpha"] == target


def test_equal_experts_choose_half_position_and_first_rotation_tie():
    record = fit_constant(*poses(z(.2), z(.2), z(.8)))
    assert record["alpha"] == [.5, 0.]
    assert record["position"]["degenerate"]
    assert record["position"]["unclipped_alpha"] is None
    assert record["rotation"]["certified"]
    assert record["rotation"]["call_count"] == 3
    assert record["rotation"]["objective_at_alpha"] == pytest.approx(.36)


@pytest.mark.parametrize("fraction", [0., .25, .5, .75, 1.])
def test_single_axis_known_optimum_and_endpoints(fraction):
    record = fit_constant(*poses(z(.8), z(0), z(fraction * .8)))
    assert record["rotation"]["certified"]
    assert record["alpha"][1] == fraction
    assert record["rotation"]["objective_at_alpha"] < 1e-28
    assert record["rotation"]["gap"] <= 1e-7


def test_cut_locus_counterexample_does_not_assume_unimodality():
    record = fit_constant(*poses(z(.4), z(-.4), z(math.pi)))
    r = record["rotation"]
    initial = r["evaluations"][:3]
    assert initial[2]["objective"] == pytest.approx(math.pi ** 2)
    assert initial[0]["objective"] == pytest.approx((math.pi - .4) ** 2)
    assert initial[1]["objective"] == pytest.approx((math.pi - .4) ** 2)
    assert min(initial[0]["objective"], initial[1]["objective"]) < initial[2]["objective"]
    assert record["alpha"][1] in (0., 1.)
    assert r["certified"] and r["gap"] <= r["tolerance"]
    assert r["lower_bound"] <= (math.pi - .4) ** 2 <= r["upper_bound"]


def test_small_hard_cap_is_honest_and_does_not_split_half_a_node():
    args = poses(rotation([1, 0, 0], .6) @ z(.2), z(.2), rotation([0, 1, 0], .3) @ z(.4))
    r = fit_constant(*args, tolerance=1e-14, max_evaluations=8)["rotation"]
    assert not r["certified"] and r["status"] == "evaluation_cap"
    assert r["call_count"] == 7
    assert r["gap"] > r["tolerance"]
    assert len(r["intervals"]) == 5 and len(r["leaves"]) == 3


def test_noncommuting_objective_and_every_bound_are_independently_reconstructable():
    slow = z(.2)
    fast = rotation([1, 0, 0], .6) @ slow
    target = rotation([0, 1, 0], .3) @ z(.4)
    result = fit_constant(*poses(fast, slow, target), tolerance=1e-14, max_evaluations=31)
    r = result["rotation"]
    nodes = r["intervals"]
    for e in r["evaluations"]:
        reconstructed = rotation([1, 0, 0], .6 * e["alpha"]) @ slow
        distance = angle_between(reconstructed, target)
        assert e["objective"] == pytest.approx(distance ** 2, abs=1e-14)
        lower = max(0., max(0., distance - r["angle_guard"]
                            - (.6 + r["angle_guard"]) * e["radius"]) ** 2 - r["objective_guard"])
        assert e["lower_bound"] == pytest.approx(lower, abs=1e-14)
    leaves = [nodes[i] for i in r["leaves"]]
    assert leaves[0]["left"] == 0 and leaves[-1]["right"] == 1
    assert all(left["right"] == right["left"] for left, right in pairwise(leaves))
    assert r["lower_bound"] == min(n["lower_bound"] for n in leaves)
    assert r["upper_bound"] == r["objective_at_alpha"] + r["objective_guard"]
    assert r["gap"] == r["upper_bound"] - r["lower_bound"]
    for node in nodes:
        e = r["evaluations"][node["center_evaluation"]]
        assert e["alpha"] == (node["left"] + node["right"]) / 2
        assert node["lower_bound"] == e["lower_bound"]
        if node["children"] is not None:
            left, right = [nodes[i] for i in node["children"]]
            assert left["parent"] == right["parent"] == node["id"]
            assert (left["left"], left["right"], right["left"], right["right"]) == (
                node["left"], e["alpha"], e["alpha"], node["right"])
    # Independent dense samples merely witness bound validity in this fixture.
    sampled = [angle_between(rotation([1, 0, 0], .6 * a) @ slow, target) ** 2 for a in np.linspace(0, 1, 101)]
    assert r["lower_bound"] <= min(sampled)
    assert r["call_count"] == len(r["evaluations"]) == len(nodes) + 2


def test_svd_projection_is_explicit_deterministic_and_inputs_unchanged():
    deformation = np.diag([1.2, .9, 1.05])
    args = poses(z(.8) @ deformation, deformation, z(.4) @ deformation, dtype=np.float32)
    before = [a.copy() for a in args]
    first = fit_constant(*args)
    second = fit_constant(*args)
    assert first == second
    assert first["alpha"][1] == .5
    assert first["rotation"]["certified"]
    assert "projected" in first["rotation"]["certificate_scope"]
    for stats in first["rotation"]["projection"].values():
        assert stats["max_frobenius_change"] > .2
        assert stats["max_determinant_error"] < 1e-14
        assert stats["max_orthogonality_error"] < 1e-14
    for actual, old in zip(args, before, strict=True):
        np.testing.assert_array_equal(actual, old)
    assert json.loads(json.dumps(first, allow_nan=False)) == first


def test_reflection_projection_returns_proper_rotations():
    reflected = np.diag([1., 1., -.5])
    result = fit_constant(*poses(reflected, reflected, reflected))
    assert result["alpha"] == [.5, 0.]
    for stats in result["rotation"]["projection"].values():
        assert stats["reflections_corrected"] == 1
        assert stats["max_determinant_error"] == 0


def test_exact_pi_interpolation_is_finite_and_uses_declared_axis():
    fast = np.diag([1., -1., -1.])
    target = rotation([1, 0, 0], math.pi / 2)
    result = fit_constant(*poses(fast, np.eye(3), target))
    assert result["alpha"][1] == .5
    assert result["rotation"]["speed_max"] == pytest.approx(math.pi)
    assert result["rotation"]["certified"]


def test_batch_objective_weights_each_window_horizon_equally():
    args = list(poses())
    args[0] = args[2] = args[4] = np.zeros((1, 2, 3))
    args[1] = np.stack((z(.8), z(.8)))[None]
    args[3] = np.stack((z(0), z(0)))[None]
    args[5] = np.stack((z(.2), z(.6)))[None]
    r = fit_constant(*args, max_evaluations=3)["rotation"]
    assert r["objective_at_alpha"] == pytest.approx(.04)
    assert r["alpha"] == .5
    assert not r["certified"]


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, True, "1e-7"])
def test_invalid_tolerance_rejected(bad):
    with pytest.raises(ValueError, match="tolerance"):
        fit_constant(*poses(), tolerance=bad)


@pytest.mark.parametrize("bad", [0, 2, -1, 3., True])
def test_invalid_evaluation_cap_rejected(bad):
    with pytest.raises(ValueError, match="max_evaluations"):
        fit_constant(*poses(), max_evaluations=bad)


@pytest.mark.parametrize("index,value", [(0, math.nan), (1, math.inf), (4, -math.inf), (5, math.nan)])
def test_nonfinite_input_rejected(index, value):
    args = list(poses())
    args[index].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        fit_constant(*args)


def test_malformed_shapes_and_nonfloating_inputs_rejected():
    args = list(poses())
    args[0] = args[0].astype(np.int64)
    with pytest.raises(ValueError, match="float32/float64"):
        fit_constant(*args)
    args = list(poses()); args[1] = args[1][:, :, :2]
    with pytest.raises(ValueError, match="rotation shapes"):
        fit_constant(*args)
    args = list(poses()); args[0] = args[0][0]
    with pytest.raises(ValueError, match="shape"):
        fit_constant(*args)
    args = [x[:0] for x in poses()]
    with pytest.raises(ValueError, match="positive"):
        fit_constant(*args)


def test_finite_inputs_with_overflowed_solution_cannot_return_invalid_json():
    args = list(poses())
    args[0][0, 0, 0] = 1e-160
    args[4][0, 0, 0] = 1e154
    with pytest.raises(FloatingPointError, match="unconstrained"):
        fit_constant(*args)


def test_nonfinite_projection_diagnostic_is_rejected():
    args = list(poses())
    args[1] *= 1e200
    with np.errstate(over="ignore"), pytest.raises(FloatingPointError, match="projection diagnostics"):
        fit_constant(*args)
