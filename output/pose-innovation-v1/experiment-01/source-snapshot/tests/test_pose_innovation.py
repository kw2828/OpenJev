"""Tiny synthetic error-history checks; isolated engineering seed410 only."""
import inspect

import pytest
import torch

from openjev.research.pose_coordination import context_tokens
from openjev.research.pose_innovation import (
    RecurrentInnovation,
    SummaryInnovation,
    apply_correction,
    innovation_tokens,
    residual_targets,
)
from openjev.research.rigid_motion import geodesic_angle, so3_exp, so3_log


@pytest.fixture(autouse=True)
def engineering_scope():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            yield
    finally:
        torch.set_num_threads(previous)


def context(dtype=torch.float64, transitions=7):
    p = torch.randn(2, transitions + 1, 3, dtype=dtype).cumsum(1) * .02
    R = so3_exp(torch.randn(2, transitions + 1, 3, dtype=dtype) * .15)
    actions = torch.randn(2, transitions, 40, dtype=dtype)
    pp = p[:, 1:].clone() + torch.randn(2, transitions, 3, dtype=dtype) * .01
    pr = so3_exp(torch.randn(2, transitions, 3, dtype=dtype) * .03) @ R[:, 1:]
    scales = torch.tensor([.1, .2, .03, .04], dtype=dtype)
    error_scales = scales[2:].clone()
    return p, R, actions, pp, pr, scales, error_scales


def poses(dtype=torch.float64, horizon=25):
    p = torch.randn(2, horizon, 3, dtype=dtype) * .2
    R = so3_exp(torch.randn(2, horizon, 3, dtype=dtype) * .2)
    root = so3_exp(torch.tensor([[.4, -.2, .1], [-.1, .3, .2]], dtype=dtype))
    scales = torch.tensor([.1, .1], dtype=dtype)
    return p, R, root, scales


def expose_head(model):
    with torch.no_grad():
        head = model.head if isinstance(model, RecurrentInnovation) else model.network[2]
        head.weight.copy_(torch.randn_like(head.weight) * .04)


def test_tokens_preserve_existing49_and_exact_transition_alignment():
    p, R, a, pp, pr, scales, errors = context(transitions=31)
    tokens = innovation_tokens(p, R, a, pp, pr, scales, errors)
    assert tokens.shape == (2, 31, 55)
    torch.testing.assert_close(tokens[..., :49], context_tokens(p, R, a, scales), rtol=0, atol=0)
    for t in (0, 13, 30):
        inv = R[:, -1].transpose(-1, -2)
        e_p = (inv @ (p[:, t + 1] - pp[:, t])[..., None])[..., 0] / errors[0]
        e_r = (inv @ so3_log(R[:, t + 1] @ pr[:, t].transpose(-1, -2))[..., None])[..., 0] / errors[1]
        torch.testing.assert_close(tokens[:, t, 49:], torch.cat((e_p, e_r), -1).tanh(),
                                   rtol=1e-14, atol=1e-14)


def test_predictions_and_actions_affect_only_their_own_completed_token():
    values = context()
    original = innovation_tokens(*values)
    changed = [v.clone() for v in values]
    changed[2][:, 2] += .5
    changed[3][:, 2] += .03
    changed[4][:, 2] = so3_exp(torch.tensor([.3, -.2, .1], dtype=values[0].dtype)) @ changed[4][:, 2]
    output = innovation_tokens(*changed)
    assert not torch.equal(output[:, 2], original[:, 2])
    assert torch.equal(output[:, :2], original[:, :2])
    assert torch.equal(output[:, 3:], original[:, 3:])


def test_later_context_cannot_change_earlier_public_features_but_root_frame_is_retrospective():
    values = context(); changed = [v.clone() for v in values]
    old = innovation_tokens(*values)
    changed[0][:, 5:] += .2
    changed[1][:, 5:] = so3_exp(torch.tensor([.4, .2, -.3], dtype=values[0].dtype)) @ changed[1][:, 5:]
    new = innovation_tokens(*changed)
    assert torch.equal(new[:, :4, :49], old[:, :4, :49])
    # Earlier signed residuals are intentionally transported to the known final
    # root frame, so they are not represented as previously emitted tokens.
    assert not torch.equal(new[:, :4, 49:], old[:, :4, 49:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_root_frame_left_error_recovers_noncommuting_targets_and_proper_rotations(dtype):
    p, R, root, scales = poses(dtype, horizon=5)
    known = torch.randn(2, 5, 6, dtype=dtype) * .8
    target_p, target_R = apply_correction(p, R, root, known, scales)
    inferred = residual_targets(p, R, target_p, target_R, root, scales)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(inferred, known, rtol=tolerance, atol=tolerance)
    pp, pr = apply_correction(p, R, root, inferred, scales)
    torch.testing.assert_close(pp, target_p, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(pr, target_R, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(pr.transpose(-1, -2) @ pr,
                               torch.eye(3, dtype=dtype).expand_as(pr), rtol=tolerance, atol=tolerance)
    world = (root[:, None] @ (known[..., 3:] * scales[1])[..., None])[..., 0]
    assert not torch.allclose(target_R, R @ so3_exp(world))


def test_target_frames_and_application_scales_can_differ_from_input_error_scales():
    p, R, a, pp, pr, scales, input_scales = context()
    tokens = innovation_tokens(p, R, a, pp, pr, scales, input_scales)
    output_scales = torch.tensor([.1, .1], dtype=p.dtype)
    target = residual_targets(pp, pr, p[:, 1:], R[:, 1:], R[:, -1], output_scales)
    normalized_input = target * torch.cat((output_scales[:1].repeat(3) / input_scales[0],
                                           output_scales[1:].repeat(3) / input_scales[1]))
    torch.testing.assert_close(tokens[..., 49:], normalized_input.tanh(), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("cls,count", [(RecurrentInnovation, 2910), (SummaryInnovation, 2994)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [1, 31])
def test_zero_initialization_size_and_exact_zero_correction(cls, count, dtype, length):
    model = cls(dtype=dtype)
    tokens = innovation_tokens(*context(dtype, transitions=length))
    out = model(tokens)
    assert sum(x.numel() for x in model.parameters()) == count
    assert out.shape == (2, 25, 6) and torch.equal(out, torch.zeros_like(out))
    p, R, root, scales = poses(dtype)
    pp, pr = apply_correction(p, R, root, out, scales)
    assert torch.equal(pp, p) and torch.equal(pr, R)


def test_summary_formula_and_interior_permutation_invariance():
    tokens = innovation_tokens(*context())
    model = SummaryInnovation(dtype=tokens.dtype); expose_head(model)
    expected = model.network(torch.cat((tokens[:, -1], tokens.mean(1), tokens[:, -1] - tokens[:, 0]), -1))
    assert torch.equal(model(tokens), expected.reshape(2, 25, 6))
    permutation = torch.tensor([0, 4, 2, 5, 1, 3, 6])
    shuffled = tokens[:, permutation]
    torch.testing.assert_close(model(tokens), model(shuffled), rtol=1e-13, atol=1e-13)


def test_recurrent_control_uses_same_weights_and_has_order_sensitivity_without_retained_state():
    tokens = innovation_tokens(*context())
    model = RecurrentInnovation(dtype=tokens.dtype); expose_head(model)
    control = RecurrentInnovation(dtype=tokens.dtype)
    control.load_state_dict(model.state_dict())
    permutation = torch.tensor([0, 4, 2, 5, 1, 3, 6])
    original = model(tokens)
    shuffled = control(tokens[:, permutation])
    assert not torch.allclose(original, shuffled, rtol=1e-6, atol=1e-8)
    assert torch.equal(model(tokens), original)
    torch.testing.assert_close(model(tokens[:1]), original[:1], rtol=1e-13, atol=1e-13)
    assert torch.equal(control(tokens), original)


@pytest.mark.parametrize("cls", [RecurrentInnovation, SummaryInnovation])
def test_physical_loss_reaches_zero_head_and_then_encoder_and_input(cls):
    model = cls(dtype=torch.float64)
    tokens = innovation_tokens(*context()).detach().requires_grad_()
    p, R, root, scales = poses()
    wanted = torch.full((2, 25, 6), .3, dtype=tokens.dtype)
    tp, tr = apply_correction(p, R, root, wanted, scales)
    pp, pr = apply_correction(p, R, root, model(tokens), scales)
    loss = (pp - tp).square().sum() + geodesic_angle(pr, tr).square().sum()
    loss.backward()
    head = model.head if cls is RecurrentInnovation else model.network[2]
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert head.bias.grad is not None and head.bias.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True); tokens.grad = None; expose_head(model)
    pp, pr = apply_correction(p, R, root, model(tokens), scales)
    ((pp - tp).square().sum() + geodesic_angle(pr, tr).square().sum()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all() and tokens.grad.abs().sum() > 0


def test_target_construction_is_separate_from_forward_and_does_not_mutate_inputs():
    values = context(); before = [v.clone() for v in values]
    tokens = innovation_tokens(*values); tokens_before = tokens.clone()
    model = RecurrentInnovation(dtype=tokens.dtype); expose_head(model)
    original = model(tokens)
    p, R, _, pp, pr, _, scales = values
    labels = residual_targets(pp, pr, p[:, 1:], R[:, 1:], R[:, -1], scales)
    labels.add_(1000)
    assert torch.equal(model(tokens), original) and torch.equal(tokens, tokens_before)
    assert list(inspect.signature(model.forward).parameters) == ["tokens"]
    for value, prior in zip(values, before, strict=True):
        assert torch.equal(value, prior)
    base_p, base_R, root, output_scales = poses()
    correction = original.clone(); original_inputs = [x.clone() for x in (base_p, base_R, root, correction, output_scales)]
    corrected = apply_correction(base_p, base_R, root, correction, output_scales)
    corrected[0].add_(10); corrected[1].zero_()
    for value, prior in zip((base_p, base_R, root, correction, output_scales), original_inputs, strict=True):
        assert torch.equal(value, prior)


@pytest.mark.parametrize("cls", [RecurrentInnovation, SummaryInnovation])
@pytest.mark.parametrize("bad", ["empty_batch", "empty_time", "long", "dimension", "nan", "dtype"])
def test_predictors_reject_bad_cached_tokens(cls, bad):
    model = cls(dtype=torch.float64)
    tokens = torch.zeros(2, 7, 55, dtype=torch.float64)
    if bad == "empty_batch":
        tokens = tokens[:0]
    elif bad == "empty_time":
        tokens = tokens[:, :0]
    elif bad == "long":
        tokens = torch.zeros(2, 32, 55, dtype=tokens.dtype)
    elif bad == "dimension":
        tokens = tokens[..., :54]
    elif bad == "nan":
        tokens[0, 0, 0] = float("nan")
    else:
        tokens = tokens.float()
    with pytest.raises(ValueError):
        model(tokens)


@pytest.mark.parametrize("bad", ["future_pose", "future_action", "prediction_length", "rotation", "dtype", "scale", "nonfinite"])
def test_token_builder_rejects_inconsistent_or_unsupported_inputs(bad):
    values = list(context(transitions=31))
    if bad == "future_pose":
        values[0] = torch.cat((values[0], values[0][:, -1:]), 1)
    elif bad == "future_action":
        values[2] = torch.cat((values[2], values[2][:, -1:]), 1)
    elif bad == "prediction_length":
        values[3] = values[3][:, :-1]
    elif bad == "rotation":
        values[1] = values[1][..., :2]
    elif bad == "dtype":
        values[4] = values[4].float()
    elif bad == "scale":
        values[6][0] = 0
    else:
        values[3][0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        innovation_tokens(*values)


@pytest.mark.parametrize("bad", ["target_batch", "target_horizon", "root", "residual", "scales", "nonfinite", "dtype"])
def test_target_and_application_reject_broadcasting_or_invalid_inputs(bad):
    p, R, root, scales = poses(horizon=3)
    correction = torch.zeros(2, 3, 6, dtype=p.dtype)
    if bad == "target_batch":
        with pytest.raises(ValueError):
            residual_targets(p, R, p[:1], R, root, scales)
        return
    if bad == "target_horizon":
        with pytest.raises(ValueError):
            residual_targets(p, R, p[:, :1], R, root, scales)
        return
    if bad == "root":
        root = root[:1]
    elif bad == "residual":
        correction = correction[:, :1]
    elif bad == "scales":
        scales[1] = -1
    elif bad == "nonfinite":
        correction[0, 0, 0] = float("nan")
    else:
        correction = correction.float()
    with pytest.raises(ValueError):
        apply_correction(p, R, root, correction, scales)


@pytest.mark.parametrize("cls", [RecurrentInnovation, SummaryInnovation])
def test_constructor_rejects_nonfloating_configuration(cls):
    with pytest.raises(ValueError):
        cls(dtype=torch.int64)
