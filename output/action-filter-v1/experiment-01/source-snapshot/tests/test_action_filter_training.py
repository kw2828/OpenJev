import numpy as np
import pytest
import torch
from train_action_filter import CONTEXT, HORIZON, forecast, ridge_features, windows

from openjev.research.action_filter_models import ActionFilterModel


@pytest.mark.parametrize("mode", ["transported_delta", "decay_delta", "gru", "diagonal_filter"])
@pytest.mark.parametrize("context", [16, 32])
def test_future_observations_cannot_change_forecast(mode, context):
    torch.manual_seed(37)
    model = ActionFilterModel(mode, 3, 2)
    obs = torch.randn(2, CONTEXT + HORIZON, 3)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 2)
    expected = forecast(model, obs, act, context)
    obs[:, CONTEXT:] = 9999
    torch.testing.assert_close(forecast(model, obs, act, context), expected, rtol=0, atol=0)


def test_history16_has_no_older_observation_or_action_input():
    torch.manual_seed(38)
    model = ActionFilterModel("gru", 3, 2)
    obs = torch.randn(2, CONTEXT + HORIZON, 3)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 2)
    expected = forecast(model, obs, act, 16)
    obs[:, :16] = 9999
    act[:, :16] = 9999
    torch.testing.assert_close(forecast(model, obs, act, 16), expected, rtol=0, atol=0)


def test_ridge_cannot_see_future_actions_or_target_observations():
    obs = torch.randn(2, CONTEXT + HORIZON, 3)
    act = torch.randn(2, CONTEXT + HORIZON - 1, 2)
    expected = ridge_features(obs, act, 5)
    obs[:, CONTEXT:] = 9999
    act[:, CONTEXT - 1 + 5:] = 9999
    torch.testing.assert_close(ridge_features(obs, act, 5), expected, rtol=0, atol=0)


def test_windows_keep_trajectory_identity_and_do_not_overlap():
    obs = np.zeros((2, 1750, 3), np.float32)
    obs[1] = 1
    act = np.zeros((2, 1749, 2), np.float32)
    x, _, ids = windows(obs, act)
    assert len(x) == 32
    assert np.all(np.diff(ids[:16, 1]) >= 42)
    assert torch.all(x[:16] == 0) and torch.all(x[16:] == 1)
    with pytest.raises(ValueError):
        windows(obs[:, :50], act[:, :49])
