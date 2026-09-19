import pytest
import torch
from train_residual_dynamics import CONTEXT, HORIZON, forecast, ridge_features

from openjev.research.residual_dynamics import ResidualDynamics, RLSAdapter


@pytest.mark.parametrize("variant", ["none", "bias", "public", "latent", "history16"])
def test_targets_do_not_enter_forecasts(variant):
    torch.manual_seed(88)
    model = ResidualDynamics(delta_scale=torch.full((9,), .1))
    obs = torch.randn(2, CONTEXT + HORIZON, 9)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 40)
    with torch.no_grad():
        expected = forecast(model, obs, act, variant)
        obs[:, CONTEXT:] = 10000
        torch.testing.assert_close(forecast(model, obs, act, variant), expected, rtol=0, atol=0)


def test_history_really_uses_only_last16():
    model = ResidualDynamics(delta_scale=torch.full((9,), .1))
    obs = torch.randn(2, CONTEXT + HORIZON, 9)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 40)
    expected = forecast(model, obs, act, "history16")
    obs[:, :16], act[:, :16] = 10000, 10000
    torch.testing.assert_close(forecast(model, obs, act, "history16"), expected, rtol=0, atol=0)


def test_ridge_root_relative_and_horizon_causal():
    obs = torch.randn(2, CONTEXT + HORIZON, 9)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 40)
    expected = ridge_features(obs, act, 10)
    obs[:, :, :3] += 50
    obs[:, CONTEXT:] = 10000
    act[:, CONTEXT - 1 + 10:] = 10000
    torch.testing.assert_close(ridge_features(obs, act, 10), expected, rtol=1e-4, atol=1e-5)


def test_imagination_never_updates_the_fast_head(monkeypatch):
    updates = []
    original = RLSAdapter.observe

    def observe(adapter, state, phi, target):
        updates.append(state["updates"].clone())
        return original(adapter, state, phi, target)

    monkeypatch.setattr(RLSAdapter, "observe", observe)
    model = ResidualDynamics(delta_scale=torch.full((9,), .1))
    obs = torch.randn(2, CONTEXT + HORIZON, 9)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 40)
    forecast(model, obs, act, "latent")
    assert len(updates) == CONTEXT - 1
    assert torch.all(updates[-1] == CONTEXT - 2)
