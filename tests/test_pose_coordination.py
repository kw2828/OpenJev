"""Synthetic pose gates/blends only; isolated seed410 and one Torch thread."""
import pytest
import torch

from openjev.research.pose_coordination import ConstantGate, RecurrentGate, SummaryGate, blend, context_tokens
from openjev.research.rigid_motion import geodesic_angle, so3_exp, so3_log


@pytest.fixture(autouse=True)
def engineering_scope():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(previous)


def context(dtype=torch.float64):
    p = torch.randn(2, 32, 3, dtype=dtype).cumsum(1) * .01
    r = so3_exp(torch.randn(2, 32, 3, dtype=dtype) * .1)
    a = torch.randn(2, 31, 40, dtype=dtype)
    scales = torch.tensor([.1, .2, .03, .04], dtype=dtype)
    return p, r, a, scales


def experts(dtype=torch.float64):
    p, q = torch.randn(2, 5, 3, dtype=dtype), torch.randn(2, 5, 3, dtype=dtype)
    r, s = so3_exp(torch.randn(2, 5, 3, dtype=dtype) * .2), so3_exp(torch.randn(2, 5, 3, dtype=dtype) * .2)
    return p, r, q, s


def test_tokens_include_root_and_match_each_completed_transition():
    p, r, a, scales = context()
    actual = context_tokens(p, r, a, scales)
    assert actual.shape == (2, 31, 49) and actual.abs().max() <= 1
    for t in (1, 7, 31):
        inv = r[:, t].transpose(-1, -2)
        dp = (inv @ (p[:, t] - p[:, t - 1])[..., None])[..., 0] / scales[0]
        w = so3_log(r[:, t] @ r[:, t - 1].transpose(-1, -2))
        w = (inv @ w[..., None])[..., 0] / scales[1]
        expected = torch.cat((r[:, t, 2, :], dp, w, a[:, t - 1]), -1).tanh()
        # Scalar versus batched matrix products may differ by a float64 ULP.
        torch.testing.assert_close(actual[:, t - 1], expected, rtol=2e-15, atol=2e-15)
    altered = a.clone(); altered[:, 0] += 1
    changed = context_tokens(p, r, altered, scales)
    assert not torch.equal(changed[:, 0], actual[:, 0])
    torch.testing.assert_close(changed[:, 1:], actual[:, 1:], atol=0, rtol=0)
    altered_p = p.clone(); altered_p[:, -1] += .1
    changed = context_tokens(altered_p, r, a, scales)
    assert not torch.equal(changed[:, -1], actual[:, -1])
    torch.testing.assert_close(changed[:, :-1], actual[:, :-1], atol=0, rtol=0)


def test_later_context_changes_do_not_modify_earlier_tokens_or_inputs():
    values = context(); before = [v.clone() for v in values]
    old = context_tokens(*values)
    changed = [v.clone() for v in values]
    changed[0][:, 17:] += 1
    changed[1][:, 17:] = so3_exp(torch.tensor([.2, .3, -.1], dtype=torch.float64))
    changed[2][:, 16:] += 1
    new = context_tokens(*changed)
    torch.testing.assert_close(old[:, :16], new[:, :16], rtol=0, atol=0)
    for v, original in zip(values, before, strict=True):
        torch.testing.assert_close(v, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gate_class,count", [(ConstantGate, 2), (SummaryGate, 3302), (RecurrentGate, 3250)])
def test_initial_half_shapes_parameters_and_reset(gate_class, count, dtype):
    tokens = context_tokens(*context(dtype))
    gate = gate_class(dtype=dtype)
    assert sum(p.numel() for p in gate.parameters()) == count
    assert gate(tokens).shape == (2, 2)
    assert torch.equal(gate(tokens), torch.full((2, 2), .5, dtype=dtype))
    gate(tokens.flip(1))
    assert torch.equal(gate(tokens), gate(tokens[:1]).expand(2, -1))


def test_summary_feature_order_and_recurrent_reset_after_nonzero_head():
    tokens = context_tokens(*context())
    summary = SummaryGate(dtype=torch.float64)
    recurrent = RecurrentGate(dtype=torch.float64)
    with torch.no_grad():
        summary.network[2].weight.fill_(.03)
        recurrent.head.weight.fill_(.03)
    x = torch.cat((tokens[:, -1], tokens.mean(1), tokens[:, -1] - tokens[:, 0]), -1)
    torch.testing.assert_close(summary(tokens), summary.network(x).sigmoid(), rtol=0, atol=0)
    before = recurrent(tokens)
    recurrent(tokens.flip(1))
    torch.testing.assert_close(recurrent(tokens), before, rtol=0, atol=0)
    torch.testing.assert_close(recurrent(tokens[:1]), before[:1], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fraction", [0., 1.])
def test_blend_exact_endpoints_and_separate_endpoint_splice(dtype, fraction):
    fast_p, fast_r, slow_p, slow_r = experts(dtype)
    alpha = torch.full((2, 2), fraction, dtype=dtype)
    actual_p, actual_r = blend(fast_p, fast_r, slow_p, slow_r, alpha)
    assert torch.equal(actual_p, fast_p if fraction else slow_p)
    assert torch.equal(actual_r, fast_r if fraction else slow_r)
    alpha[:] = torch.tensor([1., 0.], dtype=dtype)
    p, r = blend(fast_p, fast_r, slow_p, slow_r, alpha)
    assert torch.equal(p, fast_p) and torch.equal(r, slow_r)
    assert p.data_ptr() != fast_p.data_ptr() and r.data_ptr() != slow_r.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noncommuting_geodesic_path_is_proper_and_not_matrix_average(dtype):
    slow = so3_exp(torch.tensor([.5, 0., 0.], dtype=dtype))
    fast = so3_exp(torch.tensor([0., .7, 0.], dtype=dtype))
    p = torch.zeros(1, 3, 3, dtype=dtype)
    rf = fast.expand(1, 3, 3, 3); rs = slow.expand_as(rf)
    alpha = torch.tensor([[.2, .3]], dtype=dtype)
    _, out = blend(p, rf, p, rs, alpha)
    expected = so3_exp(.3 * so3_log(fast @ slow.T)) @ slow
    tol = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(out, expected.expand_as(out), rtol=tol, atol=tol)
    torch.testing.assert_close(out.transpose(-1, -2) @ out, torch.eye(3, dtype=dtype).expand_as(out),
                               rtol=tol, atol=tol)
    torch.testing.assert_close(torch.linalg.det(out), torch.ones(1, 3, dtype=dtype), rtol=tol, atol=tol)
    torch.testing.assert_close(geodesic_angle(rs, out), .3 * geodesic_angle(rs, rf), rtol=tol, atol=tol)
    assert not torch.allclose(out, .3 * rf + .7 * rs)


def test_blend_gradient_reaches_both_alpha_coordinates_and_expert_inputs():
    fp, fr, sp, sr = experts()
    fp.requires_grad_(); sp.requires_grad_()
    alpha = torch.tensor([[.2, .4], [.7, .3]], dtype=torch.float64, requires_grad=True)
    p, r = blend(fp, fr, sp, sr, alpha)
    loss = p.square().sum() + geodesic_angle(r, sr).square().sum()
    loss.backward()
    for grad in (alpha.grad, fp.grad, sp.grad):
        assert grad is not None and torch.isfinite(grad).all()
    assert (alpha.grad.abs().sum(0) > 0).all()
    h = 1e-6
    for coord in range(2):
        ap, am = alpha.detach().clone(), alpha.detach().clone()
        ap[0, coord] += h; am[0, coord] -= h
        pp, rp = blend(fp.detach(), fr, sp.detach(), sr, ap)
        pm, rm = blend(fp.detach(), fr, sp.detach(), sr, am)
        difference = ((pp.square().sum() + geodesic_angle(rp, sr).square().sum())
                      - (pm.square().sum() + geodesic_angle(rm, sr).square().sum())) / (2 * h)
        torch.testing.assert_close(alpha.grad[0, coord], difference, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("gate_class", [ConstantGate, SummaryGate, RecurrentGate])
def test_training_loss_reaches_gate_and_no_expert_mutation(gate_class):
    gate = gate_class(dtype=torch.float64)
    tokens = context_tokens(*context())
    values = experts(); before = [v.clone() for v in values]
    alpha = gate(tokens)
    p, r = blend(*values, alpha)
    (p.square().mean() + geodesic_angle(r, values[3]).square().mean()).backward()
    head = gate.logits if gate_class is ConstantGate else gate.network[2].bias if gate_class is SummaryGate else gate.head.bias
    assert head.grad is not None and torch.isfinite(head.grad).all() and head.grad.abs().sum() > 0
    for original, value in zip(before, values, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["future_pose", "future_action", "short_pose", "nonfinite", "dtype", "scale"])
def test_context_rejects_future_and_invalid_inputs(bad):
    p, r, a, scales = context()
    with pytest.raises(ValueError):
        if bad == "future_pose":
            context_tokens(torch.cat((p, p[:, -1:]), 1), torch.cat((r, r[:, -1:]), 1), a, scales)
        elif bad == "future_action":
            context_tokens(p, r, torch.cat((a, a[:, -1:]), 1), scales)
        elif bad == "short_pose":
            context_tokens(p[:, :-1], r, a, scales)
        elif bad == "nonfinite":
            p[0, 0, 0] = float("nan"); context_tokens(p, r, a, scales)
        elif bad == "dtype":
            context_tokens(p, r, a.float(), scales)
        else:
            scales[0] = 0; context_tokens(p, r, a, scales)


@pytest.mark.parametrize("bad", ["shape", "nan", "dtype", "low", "high", "expert", "empty"])
def test_blend_rejects_invalid_inputs(bad):
    fp, fr, sp, sr = experts()
    alpha = torch.full((2, 2), .5, dtype=torch.float64)
    with pytest.raises(ValueError):
        if bad == "shape":
            blend(fp, fr, sp, sr, alpha[:, :1])
        elif bad == "nan":
            alpha[0, 0] = float("nan"); blend(fp, fr, sp, sr, alpha)
        elif bad == "dtype":
            blend(fp, fr, sp, sr, alpha.float())
        elif bad in ("low", "high"):
            alpha[0, 0] = -.01 if bad == "low" else 1.01; blend(fp, fr, sp, sr, alpha)
        elif bad == "expert":
            fr[0, 0, 0, 0] = float("inf"); blend(fp, fr, sp, sr, alpha)
        else:
            blend(fp[:, :0], fr[:, :0], sp[:, :0], sr[:, :0], alpha)


@pytest.mark.parametrize("gate_class", [ConstantGate, SummaryGate, RecurrentGate])
def test_gates_reject_nonfinite_wrong_shape_and_mixed_precision(gate_class):
    gate = gate_class(dtype=torch.float64)
    tokens = torch.zeros(2, 31, 49, dtype=torch.float64)
    for value in (tokens[:, :30], tokens.float(), torch.full_like(tokens, float("nan"))):
        with pytest.raises(ValueError):
            gate(value)
