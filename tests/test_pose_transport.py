import pytest
import torch

from openjev.research.pose_transport import PoseTransport, training_scales
from openjev.research.rigid_motion import so3_exp


def sample():
    gen = torch.Generator().manual_seed(171)
    p = torch.randn(2, 57, 3, generator=gen) * .03
    rotation = so3_exp(torch.randn(2, 57, 3, generator=gen) * .1)
    actions = torch.randn(2, 56, 40, generator=gen)
    return p, rotation, actions


@pytest.mark.parametrize("variant", ["world", "body", "transport", "history16"])
def test_zero_head_is_valid_cv16(variant):
    p, r, a = sample()
    model = PoseTransport(variant, torch.tensor([.01, .02, .001, .002]))
    predicted_p, predicted_r = model.forecast(p, r, a)
    from openjev.research.rigid_motion import so3_log
    h = torch.arange(1, 26, dtype=p.dtype)[None, :, None]
    expected_p = p[:, 31:32] + h * (p[:, 31:32] - p[:, 16:17]) / 15
    w = so3_log(r[:, 31] @ r[:, 16].transpose(-1, -2)) / 15
    expected_r = so3_exp(h * w[:, None]) @ r[:, 31:32]
    torch.testing.assert_close(predicted_p, expected_p, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(predicted_r, expected_r, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("variant", ["world", "body", "transport", "history16"])
def test_no_future_pose_or_later_action_leakage(variant):
    p, r, a = sample()
    model = PoseTransport(variant, torch.tensor([.01, .02, .001, .002]))
    torch.nn.init.normal_(model.readout.weight, std=.01)
    baseline = model.forecast(p, r, a)
    changed_p, changed_r, changed_a = p.clone(), r.clone(), a.clone()
    changed_p[:, 32:] = 99
    changed_r[:, 32:] = torch.eye(3)
    changed_a[:, 41:] = 77
    changed = model.forecast(changed_p, changed_r, changed_a)
    for original, revised in zip(baseline, changed, strict=True):
        torch.testing.assert_close(original[:, :10], revised[:, :10], rtol=0, atol=0)


def test_hidden_transport_preserves_world_vectors_and_norm():
    model = PoseTransport("transport", torch.ones(4))
    p, r, _ = sample()
    h = p[:, :16].reshape(2, 48)
    moved = model.move_hidden(h, r[:, 0], r[:, 1]).reshape(2, 16, 3)
    torch.testing.assert_close(moved @ r[:, 1].transpose(-1, -2),
                               h.reshape(2, 16, 3) @ r[:, 0].transpose(-1, -2))
    torch.testing.assert_close(moved.norm(dim=-1), h.reshape(2, 16, 3).norm(dim=-1))


def test_actual_delta_uses_previous_actual_not_prediction():
    model = PoseTransport("transport", torch.ones(4))
    p, r, a = sample()
    state = model.assimilate(model.initial(p[:, 0], r[:, 0]), p[:, 0], r[:, 0])
    torch.nn.init.constant_(model.readout.bias, .4)
    prior = model.advance(state, a[:, 0])
    actual = model.assimilate(prior, p[:, 1], r[:, 1])
    torch.testing.assert_close(actual["dp"], p[:, 1] - p[:, 0])
    torch.testing.assert_close(state["p"], p[:, 0])


def test_body_translation_and_yaw_equivariance_of_forecast():
    p, r, a = sample()
    model = PoseTransport("transport", torch.tensor([.01, .02, .001, .002]))
    torch.nn.init.normal_(model.readout.weight, std=.01)
    q = so3_exp(torch.tensor([0., 0., 1.2]))
    b = torch.tensor([.2, -.5, .7])
    expected_p, expected_r = model.forecast(p, r, a)
    actual_p, actual_r = model.forecast(p @ q.T + b, q @ r, a)
    torch.testing.assert_close(actual_p, expected_p @ q.T + b, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual_r, q @ expected_r, atol=3e-6, rtol=1e-5)


def test_identical_initial_body_and_transport_weights():
    models = []
    for variant in ("body", "transport", "history16"):
        torch.manual_seed(318)
        models.append(PoseTransport(variant, torch.ones(4)))
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_identity_forecast_has_finite_parameter_gradients():
    model = PoseTransport("transport", torch.ones(4))
    p = torch.zeros(2, 57, 3)
    r = torch.eye(3).expand(2, 57, 3, 3).clone()
    pp, rr = model.forecast(p, r, torch.zeros(2, 56, 40))
    loss = (pp - .1).square().mean() + (rr - torch.eye(3)).square().mean()
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())


def test_training_scales_are_rotation_invariant_scalar_vector_scales():
    p, r, _ = sample()
    q = so3_exp(torch.tensor([.2, -.4, .8]))
    torch.testing.assert_close(training_scales(p, r), training_scales(p @ q.T, q @ r))
