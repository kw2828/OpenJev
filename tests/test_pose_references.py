"""Small analytic Torch fixtures only; no real forecasts, native calls or RNG."""
import pytest
import torch

from openjev.research.pose_references import (
    HORIZON,
    REFERENCES,
    fit_ridge16,
    predict_reference,
    predict_ridge16,
    ridge_features,
)
from openjev.research.rigid_motion import so3_exp


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def motion(dtype=torch.float64, batch=2):
    t = torch.arange(57, dtype=dtype)
    offsets = torch.arange(batch, dtype=dtype)[:, None, None]
    p = t[None, :, None] * torch.tensor([.03, -.02, .01], dtype=dtype) + offsets
    vectors = t[:, None] * torch.tensor([.003, -.002, .004], dtype=dtype)
    initial = so3_exp(torch.tensor([.4, -.3, .2], dtype=dtype))
    rotation = (so3_exp(vectors) @ initial).expand(batch, -1, -1, -1).clone()
    actions = torch.arange(batch * 56 * 40, dtype=dtype).reshape(batch, 56, 40) / 1000
    return p, rotation, actions


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["cv1", "cv16", "ls16"])
def test_exact_constant_world_motion(name, dtype):
    p, rotation, actions = motion(dtype)
    predicted_p, predicted_r = predict_reference(name, p, rotation, actions)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(predicted_p, p[:, 32:], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(predicted_r, rotation[:, 32:], atol=tolerance, rtol=tolerance)
    assert predicted_p.dtype == predicted_r.dtype == dtype


def test_hold_is_owned_current_pose_and_ignores_all_action_values():
    p, rotation, actions = motion()
    actions.fill_(float("nan"))
    result_p, result_r = predict_reference("hold", p, rotation, actions)
    assert torch.equal(result_p, p[:, 31:32].expand(-1, 25, -1))
    assert torch.equal(result_r, rotation[:, 31:32].expand(-1, 25, -1, -1))
    result_p.fill_(99)
    result_r.fill_(0)
    assert not torch.equal(p[:, 31], result_p[:, 0])
    assert rotation[:, 31].abs().sum() > 0


def test_body_twist_tracks_helical_motion_not_a_straight_rotated_secant():
    t = torch.arange(57, dtype=torch.float64)
    theta = .018 * t
    p = torch.stack((2 * (theta.cos() - 1), 2 * theta.sin(), .01 * t), -1)[None]
    w = torch.stack((torch.zeros_like(t), torch.zeros_like(t), theta), -1)
    rotation = so3_exp(w)[None]
    actions = torch.zeros(1, 56, 40, dtype=torch.float64)
    result_p, result_r = predict_reference("body16", p, rotation, actions)
    torch.testing.assert_close(result_p, p[:, 32:], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result_r, rotation[:, 32:], atol=2e-12, rtol=2e-12)
    straight_p, _ = predict_reference("cv16", p, rotation, actions)
    assert (straight_p - p[:, 32:]).abs().max() > .1


def test_rooted_ls_uses_all16_points_and_no_intercept_displacement():
    p, rotation, actions = motion(batch=1)
    p[:, 20, 0] += .12
    predicted_p, _ = predict_reference("ls16", p, rotation, actions)
    offsets = torch.arange(-15, 1, dtype=torch.float64)
    velocity_x = .03 + (-11 * .12) / offsets.square().sum()
    expected = p[:, 31, 0] + torch.arange(1, 26, dtype=torch.float64) * velocity_x
    torch.testing.assert_close(predicted_p[0, :, 0], expected.flatten())
    cv_p, _ = predict_reference("cv16", p, rotation, actions)
    assert not torch.equal(cv_p, predicted_p)


@pytest.mark.parametrize("name", REFERENCES)
def test_no_future_pose_or_prior_discarded_context_leakage(name):
    p, rotation, actions = motion()
    before = predict_reference(name, p, rotation, actions)
    p[:, 32:] = float("nan")
    rotation[:, 32:] = float("nan")
    actions.fill_(float("nan"))
    p[:, :16] += 800
    rotation[:, :16] = so3_exp(torch.tensor([1., .3, -.2], dtype=rotation.dtype))
    after = predict_reference(name, p, rotation, actions)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def maps():
    result = []
    for h in range(1, HORIZON + 1):
        weight = torch.zeros(96 + (15 + h) * 40, 6, dtype=torch.float64)
        # One explicit permitted action drives x. Others drive rotation residual.
        weight[-40, 0] = .2
        weight[-1, 5] = .002
        result.append({"weight": weight, "intercept": torch.zeros(6, dtype=torch.float64)})
    return result


def test_ridge_future_pose_poison_and_per_endpoint_future_action_causality():
    p, rotation, actions = motion()
    weights = maps()
    before_p, before_r = predict_ridge16(weights, p, rotation, actions)
    p[:, 32:] = float("nan")
    rotation[:, 32:] = float("nan")
    after_p, after_r = predict_ridge16(weights, p, rotation, actions)
    assert torch.equal(before_p, after_p) and torch.equal(before_r, after_r)
    actions[:, 36:] += 7  # h<=5 uses through block35, excluding changed block36.
    changed_p, changed_r = predict_ridge16(weights, p, rotation, actions)
    assert torch.equal(before_p[:, :5], changed_p[:, :5])
    assert torch.equal(before_r[:, :5], changed_r[:, :5])
    assert not torch.equal(before_p[:, 5:], changed_p[:, 5:])
    assert not torch.equal(before_r[:, 5:], changed_r[:, 5:])


def test_ridge_features_exact_order_and_ignored_future_action_nan():
    p, rotation, actions = motion(batch=1)
    features = ridge_features(p, rotation, actions, 1)
    assert features.shape == (1, 736)
    torch.testing.assert_close(features[:, :48], (p[:, 16:32] - p[:, 31:32]).flatten(1))
    assert torch.equal(features[:, 96:], actions[:, 16:32].flatten(1))
    actions[:, 32:] = float("nan")
    assert torch.equal(features, ridge_features(p, rotation, actions, 1))
    with pytest.raises(ValueError, match="required action"):
        ridge_features(p, rotation, actions, 2)


def test_unit_ridge_dual_matches_closed_form_two_training_rows():
    p, rotation, actions = motion()
    # Only future targets differ nonlinearly; no label enters a feature.
    h = torch.arange(1, 26, dtype=torch.float64)
    p[0, 32:, 0] += .001 * h.square()
    p[1, 32:, 0] -= .002 * h.square()
    rotation[0, 32:] = so3_exp(torch.stack((h * 0, h * 0, .004 * h), -1)) @ rotation[0, 31]
    rotation[1, 32:] = so3_exp(torch.stack((h * 0, h * 0, .009 * h), -1)) @ rotation[1, 31]
    weights = fit_ridge16(p, rotation, actions)
    # Validate position-output coefficients by the two-row exact ridge identity:
    # W = (x1-x0)(y1-y0)^T / (2 + ||x1-x0||²).
    for horizon in (1, 9, 25):
        x = ridge_features(p, rotation, actions, horizon)
        y = torch.tensor([.001 * horizon**2, -.002 * horizon**2], dtype=torch.float64)
        dx = x[1] - x[0]
        expected = dx * (y[1] - y[0]) / (2 + dx.square().sum())
        expected_intercept = y.mean() - x.mean(0) @ expected
        torch.testing.assert_close(weights[horizon - 1]["weight"][:, 0], expected, atol=1e-13, rtol=1e-10)
        torch.testing.assert_close(weights[horizon - 1]["intercept"][0], expected_intercept, atol=1e-13, rtol=1e-10)
    saved = [{k: v.clone() for k, v in m.items()} for m in weights]
    result_p, result_r = predict_ridge16(weights, p, rotation, actions)
    assert result_p.dtype == result_r.dtype == torch.float64
    assert torch.isfinite(result_p).all() and torch.isfinite(result_r).all()
    assert all(torch.equal(m[k], s[k]) for m, s in zip(weights, saved) for k in m)
    torch.testing.assert_close(result_r.transpose(-1, -2) @ result_r,
                               torch.eye(3, dtype=torch.float64).expand_as(result_r), atol=1e-12, rtol=1e-12)


def test_unpenalized_intercept_fits_constant_residual_and_zero_map_is_cv16():
    p, rotation, actions = motion(batch=1)
    p[:, 32:] += torch.tensor([.5, -.1, .2], dtype=p.dtype)
    weights = fit_ridge16(p, rotation, actions)
    result_p, result_r = predict_ridge16(weights, p, rotation, actions)
    torch.testing.assert_close(result_p, p[:, 32:], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(result_r, rotation[:, 32:], atol=1e-13, rtol=1e-13)
    for mapping in weights:
        assert not mapping["weight"].any()
        mapping["intercept"].zero_()
    got = predict_ridge16(weights, p, rotation, actions)
    expected = predict_reference("cv16", p, rotation, actions)
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "name", "weights"])
def test_input_and_weight_rejections(bad):
    p, rotation, actions = motion()
    if bad == "shape":
        p = p[:, :-1]
    elif bad == "dtype":
        actions = actions.float()
    elif bad == "nan":
        p[:, 31] = float("nan")
    with pytest.raises(ValueError):
        if bad == "weights":
            predict_ridge16([], p, rotation, actions)
        else:
            predict_reference("invalid" if bad == "name" else "cv16", p, rotation, actions)
