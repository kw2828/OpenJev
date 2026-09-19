import pytest
import torch
import train_pose_coordination as runner

from openjev.research.pose_coordination import blend, context_tokens
from openjev.research.pose_support import PoseSupport
from openjev.research.pose_transport import PoseTransport
from openjev.research.rigid_motion import so3_exp


def sample():
    generator = torch.Generator().manual_seed(1783)
    p = torch.randn(2, 32, 3, generator=generator) * .01
    r = so3_exp(torch.randn(2, 32, 3, generator=generator) * .02)
    past = torch.randn(2, 31, 40, generator=generator) * .1
    future = torch.randn(2, 25, 40, generator=generator) * .1
    scales = torch.tensor([.01, .02, .005, .005])
    return p, r, past, future, PoseSupport(scales), PoseTransport('body', scales)


@pytest.mark.parametrize('variant', runner.VARIANTS)
def test_forecast_cannot_read_future_poses_or_later_actions(variant):
    p, r, past, future, fast, slow = sample()
    gate = runner.make_gate(variant) if variant in runner.TRAIN_VARIANTS else None
    first = runner.forecast(fast, slow, gate, variant, p, r, past, future)
    altered = future.clone()
    altered[:, 10:] = 99
    second = runner.forecast(fast, slow, gate, variant, p, r, past, altered)
    for x, y in zip(first[:2], second[:2], strict=True):
        torch.testing.assert_close(x[:, :10], y[:, :10], rtol=0, atol=0)
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    with pytest.raises(ValueError, match='context'):
        runner.forecast(fast, slow, gate, variant, torch.cat((p, p), 1), torch.cat((r, r), 1), past, future)


def test_fixed_splicing_keeps_private_forecasts_unchanged():
    p, r, past, future, fast, slow = sample()
    f = runner.forecast(fast, slow, None, 'fast', p, r, past, future)
    s = runner.forecast(fast, slow, None, 'slow', p, r, past, future)
    for variant, expected in [('position_fast', (f[0], s[1])), ('position_slow', (s[0], f[1]))]:
        got = runner.forecast(fast, slow, None, variant, p, r, past, future)
        for x, y in zip(got[:2], expected, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_training_updates_gate_without_changing_cached_experts():
    p, r, past, future, fast, slow = sample()
    fast.requires_grad_(False)
    slow.requires_grad_(False)
    with torch.no_grad():
        fp, fr, sp, sr = runner.private_rollouts(fast, slow, p, r, torch.cat((past, future), 1))
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in (fast, slow)]
    gate = runner.make_gate('constant')
    opt = torch.optim.Adam(gate.parameters(), lr=.1)
    tokens = context_tokens(p, r, past, fast.scales)
    pp, rr = blend(fp, fr, sp, sr, gate(tokens))
    loss = runner.loss_function(pp, rr, fp, fr)
    loss.backward()
    opt.step()
    assert gate(tokens)[0, 0] > .5
    for m, original in zip((fast, slow), before, strict=True):
        assert all(x.grad is None for x in m.parameters())
        for key, val in m.state_dict().items():
            torch.testing.assert_close(val, original[key], rtol=0, atol=0)


def test_invalid_variant_rejected():
    p, r, past, future, fast, slow = sample()
    with pytest.raises(ValueError, match='variant'):
        runner.forecast(fast, slow, None, 'future_oracle', p, r, past, future)
