"""Synthetic-only causal ridge and pose-frame checks, isolated seed410."""
import json

import pytest
import torch

from openjev.research.pose_adaptation import PoseAdaptation
from openjev.research.rigid_motion import so3_exp, so3_log


def fixture(mode="learned", dtype=torch.float64, batch=2):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        model = PoseAdaptation(mode, torch.tensor([.1, .05, .02, .01], dtype=dtype))
        p = torch.randn(batch, 32, 3, dtype=dtype).cumsum(1) * .01
        w = torch.randn(batch, 32, 3, dtype=dtype) * .04
        rotation = so3_exp(w)
        actions = torch.randn(batch, 31, 40, dtype=dtype) * .2
        future = torch.randn(batch, 5, 40, dtype=dtype) * .2
    return model, p, rotation, actions, future


@pytest.mark.parametrize("mode,parameters,scalars", [("learned", 3066, 96), ("public", 300, 318)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_accounting_and_diagnostics(mode, parameters, scalars, dtype):
    model, p, rotation, actions, future = fixture(mode, dtype)
    state, diag = model.fit_context(p, rotation, actions, return_diagnostics=True)
    predicted_p, predicted_r = model.rollout(state, future)
    assert predicted_p.shape == (2, 5, 3)
    assert predicted_r.shape == (2, 5, 3, 3)
    assert predicted_p.dtype == dtype
    assert diag["available_support_transitions"] == diag["fitted_support_transitions"] == 30
    assert diag["batched_solve_calls"] == 1
    assert all(1 <= c <= 61.0001 for c in diag["normal_condition_number"])
    counts = model.parameter_and_state_counts()
    assert counts["trainable_parameters"] == parameters
    assert counts["persistent_state_scalars_per_case"] == scalars
    json.dumps(model.configuration(), allow_nan=False)
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize("mode", ["learned", "public"])
def test_posterior_matches_independent_dense_ridge(mode):
    model, p, rotation, actions, _ = fixture(mode)
    with torch.no_grad():
        model.prior.copy_(torch.arange(model.prior.numel()).reshape_as(model.prior) * .0001)
    phi, target, _, _ = model._context_design(p, rotation, actions)
    expected = torch.linalg.solve(phi.transpose(1, 2) @ phi + torch.eye(model.feature_dim),
                                  phi.transpose(1, 2) @ target + model.prior)
    actual = model.fit_context(p, rotation, actions)["weights"]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_noncommuting_support_frames_and_action_index():
    model, p, rotation, actions, _ = fixture("public")
    phi, targets, dp, w = model._context_design(p, rotation, actions)
    assert phi.shape == (2, 30, 50)
    assert targets.shape == (2, 30, 6)
    for t in (1, 7, 30):
        inverse = rotation[:, t].transpose(-1, -2)
        backward = p[:, t] - p[:, t - 1]
        forward = p[:, t + 1] - p[:, t]
        old_w = so3_log(rotation[:, t] @ rotation[:, t - 1].transpose(-1, -2))
        next_w = so3_log(rotation[:, t + 1] @ rotation[:, t].transpose(-1, -2))
        expected = torch.cat(((inverse @ (forward - backward)[..., None]).squeeze(-1) / .02,
                              (inverse @ (next_w - old_w)[..., None]).squeeze(-1) / .01), -1)
        torch.testing.assert_close(targets[:, t - 1], expected)
        inputs = torch.cat((rotation[:, t, 2],
                            (inverse @ backward[..., None]).squeeze(-1) / .1,
                            (inverse @ old_w[..., None]).squeeze(-1) / .05,
                            actions[:, t]), -1).tanh()
        expected_phi = torch.cat((inputs / inputs.norm(dim=-1, keepdim=True),
                                  torch.ones(2, 1, dtype=inputs.dtype)), -1)
        torch.testing.assert_close(phi[:, t - 1], expected_phi)
    torch.testing.assert_close(dp, p[:, 31] - p[:, 30])
    torch.testing.assert_close(w, so3_log(rotation[:, 31] @ rotation[:, 30].transpose(-1, -2)))
    changed = actions.clone()
    changed[:, 0] = 999
    torch.testing.assert_close(model.fit_context(p, rotation, changed)["weights"],
                               model.fit_context(p, rotation, actions)["weights"], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["learned", "public"])
def test_prior_zero_is_world_cv1_and_no_solve(mode, monkeypatch):
    model, p, rotation, actions, future = fixture(mode)

    def forbidden(*args, **kwargs):
        raise AssertionError("static model must not solve")

    monkeypatch.setattr(torch, "cholesky_solve", forbidden)
    state, diag = model.fit_context(p, rotation, actions, adapt=False, return_diagnostics=True)
    predicted_p, predicted_r = model.rollout(state, future)
    steps = torch.arange(1, 6, dtype=p.dtype)
    expected_p = p[:, -1, None] + steps[None, :, None] * (p[:, -1] - p[:, -2])[:, None]
    expected_r = so3_exp(steps[None, :, None] * state["w"][:, None]) @ rotation[:, -1, None]
    torch.testing.assert_close(predicted_p, expected_p, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(predicted_r, expected_r, atol=1e-12, rtol=1e-12)
    assert diag["batched_solve_calls"] == diag["fitted_support_transitions"] == 0


def test_rollout_correction_is_current_body_to_world():
    model, p, rotation, actions, future = fixture("public", batch=1)
    state = model.fit_context(p, rotation, actions, adapt=False)
    with torch.no_grad():
        state["weights"][-1, -1] = torch.tensor([1., 2., 3., .4, .5, .6])
    predicted_p, predicted_r = model.rollout(state, future[:, :1])
    dp = state["dp"] + (state["R"] @ torch.tensor([.02, .04, .06], dtype=p.dtype)[..., None]).squeeze(-1)
    w = state["w"] + (state["R"] @ torch.tensor([.004, .005, .006], dtype=p.dtype)[..., None]).squeeze(-1)
    torch.testing.assert_close(predicted_p[:, 0], state["p"] + dp)
    torch.testing.assert_close(predicted_r[:, 0], so3_exp(w) @ state["R"])


@pytest.mark.parametrize("mode", ["learned", "public"])
def test_batch_and_branch_isolation_and_owned_roots(mode):
    model, p, rotation, actions, future = fixture(mode)
    originals = [x.clone() for x in (p, rotation, actions, future)]
    state = model.fit_context(p, rotation, actions)
    saved = {key: value.clone() for key, value in state.items()}
    output = model.rollout(state, future)
    single = model.fit_context(p[:1], rotation[:1], actions[:1])
    for actual, expected in zip(model.rollout(single, future[:1]), output, strict=True):
        torch.testing.assert_close(actual, expected[:1], atol=1e-10, rtol=1e-10)
    altered = future.clone()
    altered[:, 3:] += 100
    branched = model.rollout(state, altered)
    for actual, expected in zip(branched, output, strict=True):
        torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=0, rtol=0)
    for key in state:
        torch.testing.assert_close(state[key], saved[key], atol=0, rtol=0)
    for actual, expected in zip((p, rotation, actions, future), originals, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        state["p"].add_(100)
        state["R"].zero_()
    torch.testing.assert_close(p, originals[0], atol=0, rtol=0)
    torch.testing.assert_close(rotation, originals[1], atol=0, rtol=0)


def test_meta_gradients_through_solve_and_rollout_are_nonzero():
    model, p, rotation, actions, future = fixture()
    p.requires_grad_(True)
    actions.requires_grad_(True)
    state = model.fit_context(p, rotation, actions)
    solve_grads = torch.autograd.grad(state["weights"].square().sum(),
                                      tuple(model.encoder.parameters()), retain_graph=True)
    assert all(torch.isfinite(g).all() for g in solve_grads)
    assert sum(g.abs().sum() for g in solve_grads) > 0
    state["weights"].retain_grad()
    predicted_p, predicted_r = model.rollout(state, future)
    loss = predicted_p.square().mean() + (predicted_r - torch.eye(3)).square().mean()
    loss.backward()
    for value in (p.grad, actions.grad, state["weights"].grad, model.prior.grad):
        assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
    grads = [parameter.grad for parameter in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("mode", ["learned", "public"])
def test_constant_motion_zero_correction_and_finite_identity_gradients(mode):
    model, _, _, actions, future = fixture(mode)
    p = torch.arange(32, dtype=torch.float64)[None, :, None].expand(2, -1, 3).clone()
    rotation = torch.eye(3, dtype=p.dtype).expand(2, 32, 3, 3).clone().requires_grad_(True)
    state = model.fit_context(p, rotation, actions)
    torch.testing.assert_close(state["weights"], torch.zeros_like(state["weights"]), atol=0, rtol=0)
    predicted_p, predicted_r = model.rollout(state, future)
    assert torch.isfinite(predicted_p).all() and torch.isfinite(predicted_r).all()
    (predicted_p.square().mean() + predicted_r.square().mean()).backward()
    assert torch.isfinite(rotation.grad).all()


@pytest.mark.parametrize("bad", [torch.ones(3), torch.tensor([1., 1., 0., 1.]),
                                torch.tensor([1., 1., float("nan"), 1.]), torch.ones(4, dtype=torch.int64)])
def test_reject_invalid_scales(bad):
    with pytest.raises(ValueError):
        PoseAdaptation("learned", bad)


@pytest.mark.parametrize("which", ["short_context", "long_context", "short_actions", "dtype",
                                  "nonfinite", "bad_adapt", "extra_state", "empty_horizon"])
def test_strict_input_boundaries(which):
    model, p, rotation, actions, future = fixture()
    with pytest.raises(ValueError):
        if which == "short_context":
            model.fit_context(p[:, :-1], rotation, actions)
        elif which == "long_context":
            model.fit_context(torch.cat((p, p[:, -1:]), 1), rotation, actions)
        elif which == "short_actions":
            model.fit_context(p, rotation, actions[:, :-1])
        elif which == "dtype":
            model.fit_context(p.float(), rotation, actions)
        elif which == "nonfinite":
            p[0, 0, 0] = float("nan")
            model.fit_context(p, rotation, actions)
        elif which == "bad_adapt":
            model.fit_context(p, rotation, actions, adapt=1)
        elif which == "extra_state":
            state = model.fit_context(p, rotation, actions)
            model.rollout({**state, "future_pose": p[:, -1]}, future)
        else:
            model.rollout(model.fit_context(p, rotation, actions), future[:, :0])
