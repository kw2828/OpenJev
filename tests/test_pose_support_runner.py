"""The runner's forecast boundary must not accept or inspect future poses."""
import pytest
import torch
from run_pose_support import SUPPORT_VARIANTS, checkpoint_name, forecast_support


class ContextRecorder:
    def __init__(self):
        self.received = None

    def fit_context(self, p, rotation, actions, *, variant, return_diagnostics):
        if p.shape != (2, 32, 3) or rotation.shape != (2, 32, 3, 3) or actions.shape != (2, 31, 40):
            raise ValueError("context boundary")
        self.received = (p, rotation, actions, variant)
        state = {"root": p[:, -1].clone()}
        return (state, {"variant": variant}) if return_diagnostics else state

    def rollout(self, state, actions):
        return state["root"][:, None].expand(-1, len(actions[0]), -1).clone(), actions.clone()


@pytest.mark.parametrize("variant", SUPPORT_VARIANTS)
def test_context_only_boundary_and_diagnostic_transparency(variant):
    model = ContextRecorder()
    p = torch.arange(2 * 32 * 3).reshape(2, 32, 3).float()
    r = torch.eye(3).expand(2, 32, 3, 3)
    past, future = torch.zeros(2, 31, 40), torch.ones(2, 25, 40)
    ordinary = forecast_support(model, p, r, past, future, variant)
    diagnostic = forecast_support(model, p, r, past, future, variant, diagnostics=True)
    assert all(torch.equal(x, y) for x, y in zip(ordinary, diagnostic[:2]))
    assert diagnostic[2] == {"variant": variant}
    assert model.received[0].shape == (2, 32, 3)
    assert model.received[2].shape == (2, 31, 40)
    # The future action block is routed only to rollout.
    assert torch.equal(ordinary[1], future)
    assert torch.equal(model.received[2], past)


def test_forecast_rejects_target_sized_context_and_incomplete_actions():
    p, r = torch.zeros(2, 57, 3), torch.eye(3).expand(2, 57, 3, 3)
    with pytest.raises(ValueError, match="context boundary"):
        forecast_support(ContextRecorder(), p, r, torch.zeros(2, 31, 40), torch.ones(2, 25, 40), "full")
    with pytest.raises(ValueError, match="exactly25"):
        forecast_support(ContextRecorder(), p[:, :32], r[:, :32], torch.zeros(2, 31, 40),
                         torch.ones(2, 24, 40), "full")


def test_reused_checkpoint_identity_does_not_follow_display_label():
    for variant in SUPPORT_VARIANTS:
        assert checkpoint_name(variant, 1101) == "meta-1101"
    assert checkpoint_name("static_adapt", 1202) == "static-1202"
    assert checkpoint_name("gru", 1303) == "gru-1303"
    assert checkpoint_name("ridge16", None) == "ridge16"
    with pytest.raises(ValueError, match="no checkpoint"):
        checkpoint_name("hold", None)
