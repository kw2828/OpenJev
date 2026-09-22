"""Fabricated cost/loss checks only, with no data loading, fitting or simulation.

Torch cases use either scripted differentiable costs or one unfitted qualified
head. Direction permutations and expected losses are hand specified; no saved
labels, empirical arrays, native environment, or optimizer is involved.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openjev.research import otto_cost_regression as m

# Original action -> transformed action for -x,+x,-y,+y; reflected groups first
# reverse y and then rotate counterclockwise in the array coordinate convention.
DIRECTIONS = ((0, 1, 2, 3), (2, 3, 1, 0), (1, 0, 3, 2), (3, 2, 0, 1),
              (0, 1, 3, 2), (2, 3, 0, 1), (1, 0, 2, 3), (3, 2, 1, 0))


def test_episode_weights_equalize_contribution_and_duplicate_rows_do_not_change_scale():
    identities = ['short', 'long', 'long', 'long']
    weights = m.episode_weights(identities)
    np.testing.assert_allclose(weights, [2, 2 / 3, 2 / 3, 2 / 3], rtol=1e-15)
    assert weights.dtype == np.float64
    assert weights.sum() == pytest.approx(4)
    assert weights[0] == pytest.approx(weights[1:].sum())
    costs = np.array([[0, 2, np.inf, np.inf], [0, 6, np.inf, np.inf]], dtype=np.float64)
    base = m.build_targets(costs, np.isfinite(costs), ['a', 'b'], kind='continuation')
    repeated = costs[[0, 0, 0, 1]]
    expanded = m.build_targets(repeated, np.isfinite(repeated), ['a', 'a', 'a', 'b'], kind='continuation')
    assert base['scale'] == expanded['scale'] == pytest.approx(math.sqrt(5))
    np.testing.assert_array_equal(expanded['scaled_float32'], base['scaled_float32'][[0, 0, 0, 1]])


def test_centering_before_float32_preserves_small_gaps_without_panel_amplification():
    step = 2.0**-10
    costs = np.array([[2.0**30, 2.0**30 + step, 2.0**30 + 2 * step, np.inf],
                      [0, 2, np.inf, np.inf]], dtype=np.float64)
    mask = np.isfinite(costs)
    result = m.build_targets(costs, mask, ['a', 'b'], kind='continuation')
    assert np.float32(costs[0, 0]) == np.float32(costs[0, 1])
    np.testing.assert_array_equal(result['centered_float64'][0], [-step, 0, step, 0])
    assert result['scaled_float32'][0, 2] > result['scaled_float32'][0, 1] > result['scaled_float32'][0, 0]
    small_gap = result['scaled_float32'][0, 1] - result['scaled_float32'][0, 0]
    large_gap = result['scaled_float32'][1, 1] - result['scaled_float32'][1, 0]
    assert float(small_gap / large_gap) == pytest.approx(step / 2, rel=2e-7)
    shifted = np.where(mask, costs + np.array([[32], [1024]]), np.inf)
    same = m.build_targets(shifted, mask, ['a', 'b'], kind='continuation')
    np.testing.assert_array_equal(same['centered_float64'], result['centered_float64'])
    np.testing.assert_array_equal(same['scaled_float32'], result['scaled_float32'])
    assert same['scale'] == result['scale']


def test_analytic_centered_costs_encode_historical_preferences_including_range_floor():
    costs = np.array([[2, 3, 6, np.inf], [0, 2e-9, np.inf, np.inf]], dtype=np.float64)
    result = m.build_targets(costs, np.isfinite(costs), ['a', 'b'], kind='analytic')
    np.testing.assert_allclose(result['centered_float64'][0], [-5 / 3, -2 / 3, 7 / 3, 0], atol=1e-15)
    np.testing.assert_allclose(result['centered_float64'][1], [-0.4, 0.4, 0, 0], atol=1e-15)
    # Historical logits are [0,-1,-4] and [0,-0.8]. Test the represented
    # distribution before the separate global training scale is applied.
    for row, expected_logits in enumerate(((0, -1, -4), (0, -0.8))):
        expected = np.exp(expected_logits)
        expected /= expected.sum()
        observed = np.exp(-result['centered_float64'][row, :len(expected_logits)])
        observed /= observed.sum()
        np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-15)


def test_global_rms_weights_episodes_and_eligible_action_means():
    costs = np.array([[0, 2, np.inf, np.inf], [0, 2, 4, np.inf], [0, 4, 8, 12]], dtype=np.float64)
    result = m.build_targets(costs, np.isfinite(costs), ['a', 'a', 'b'], kind='continuation')
    expected_centered = np.array([[-1, 1, 0, 0], [-2, 0, 2, 0], [-6, -2, 2, 6]], dtype=np.float64)
    np.testing.assert_array_equal(result['centered_float64'], expected_centered)
    np.testing.assert_array_equal(result['weights_float64'], [0.75, 0.75, 1.5])
    # Per-row eligible means of squares are 1, 8/3, and 20. Episode weighting
    # gives (0.75 + 2 + 30)/3 = 131/12, not a uniform mean over action cells.
    assert result['scale'] == pytest.approx(math.sqrt(131 / 12), rel=1e-15)
    np.testing.assert_allclose(result['scaled_float32'], expected_centered / math.sqrt(131 / 12),
                               rtol=1e-7, atol=0)


def test_tied_and_single_action_rows_stay_zero_without_nan_or_hard_labels():
    costs = np.array([[3, 3, 3, np.inf], [9, np.inf, np.inf, np.inf]], dtype=np.float64)
    for kind in ('analytic', 'continuation'):
        result = m.build_targets(costs, np.isfinite(costs), ['a', 'b'], kind=kind)
        assert result['scale'] == 1e-8
        np.testing.assert_array_equal(result['centered_float64'], np.zeros((2, 4)))
        np.testing.assert_array_equal(result['scaled_float32'], np.zeros((2, 4), dtype=np.float32))


def test_matched_arms_detach_immutable_targets_masks_and_episode_weights():
    analytic = np.array([[1, 2, 4, np.inf], [0, 2, np.inf, np.inf]], dtype=np.float64)
    continuation = np.array([[40, 41, 42, np.inf], [3, 7, np.inf, np.inf]], dtype=np.float64)
    mask, identities = np.isfinite(analytic), ['first', 'second']
    before = (analytic.copy(), continuation.copy(), mask.copy(), identities.copy())
    result = m.targets(analytic, continuation, mask, identities)
    for actual, expected in zip((analytic, continuation, mask, identities), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for arm in ('analytic', 'continuation'):
        for value in result[arm].values():
            if isinstance(value, np.ndarray):
                assert not np.shares_memory(value, analytic)
                assert not np.shares_memory(value, continuation)
                assert not np.shares_memory(value, mask)
                with pytest.raises(ValueError):
                    value.setflags(write=True)
        np.testing.assert_array_equal(result[arm]['weights_float32'], [1, 1])
        assert result[arm]['scaled_float32'].dtype == np.float32
    saved = result['continuation']['centered_float64'].copy()
    analytic.fill(0)
    continuation.fill(100)
    mask.fill(False)
    identities[:] = ['changed', 'changed']
    np.testing.assert_array_equal(result['continuation']['centered_float64'], saved)
    assert result['continuation']['allowed'][0].tolist() == [True, True, True, False]
    np.testing.assert_array_equal(result['analytic']['weights_float64'], result['continuation']['weights_float64'])


@pytest.mark.parametrize('invalid', ['float32_before_centering', 'finite_blocked_cost', 'misaligned_episodes'])
def test_malformed_cost_or_row_contract_is_rejected(invalid):
    costs = np.array([[0, 2, np.inf, np.inf]], dtype=np.float64)
    mask, identities = np.isfinite(costs), ['a']
    if invalid == 'float32_before_centering':
        costs = costs.astype(np.float32)
    elif invalid == 'finite_blocked_cost':
        costs[0, 2] = 0
    else:
        identities.append('b')
    with pytest.raises(ValueError):
        m.build_targets(costs, mask, identities, kind='continuation')


def scripted_head(values):
    import torch

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
            self.received = []

        def forward(self, views):
            self.received.append(views.detach().clone())
            return self.values

    class Head(torch.nn.Module):
        kind = 'dense_augmented'

        def __init__(self):
            super().__init__()
            self.core = Core()
            self.register_buffer('permutations', torch.tensor(DIRECTIONS, dtype=torch.int64))

        def views(self, x):
            views = x[:, None, :].expand(-1, 8, -1).clone()
            views[:, :, 0] = torch.arange(8, dtype=torch.float32)
            return views

    return Head()


def loss_fixture():
    import torch

    values = np.zeros((2, 8, 4), dtype=np.float32)
    for group, permutation in enumerate(DIRECTIONS):
        # Values are specified in original coordinates and independently placed
        # into each transformed action slot. Blocked costs are deliberately huge.
        originals = ((group, 999, -group, -999), (2 * group + 5, 5, 5 - 2 * group, 777))
        for row in range(2):
            for action, transformed in enumerate(permutation):
                values[row, group, transformed] = originals[row][action]
    x = torch.zeros((2, 2836), dtype=torch.float32)
    target = torch.tensor([[-1, 0, 1, 0], [-2, 0, 2, 0]], dtype=torch.float32)
    mask = torch.tensor([[True, False, True, False], [True, True, True, False]])
    return values, x, target, mask


def test_all_eight_view_losses_relabel_masks_and_average_before_ensemble():
    import torch

    values, x, target, mask = loss_fixture()
    model = scripted_head(values)
    losses = m.training_losses(model, x, target, mask)
    # Eight row losses are (g+1)^2 and 8/3*(g+1)^2. Squaring after
    # averaging views instead would incorrectly give 20.25 and 54.
    torch.testing.assert_close(losses, torch.tensor([25.5, 68.0]), rtol=1e-7, atol=0)
    assert len(model.core.received) == 1
    views = model.core.received[0]
    assert views.shape == (2, 8, 2836)
    torch.testing.assert_close(views[:, :, 0], torch.arange(8, dtype=torch.float32).expand(2, 8))


def test_blocked_gradients_are_zero_and_loss_ignores_prediction_and_target_offsets():
    import torch

    values, x, target, mask = loss_fixture()
    model = scripted_head(values)
    losses = m.training_losses(model, x, target, mask)
    losses.sum().backward()
    expected = np.zeros_like(values)
    for group, permutation in enumerate(DIRECTIONS):
        expected[0, group, permutation[0]] = (group + 1) / 8
        expected[0, group, permutation[2]] = -(group + 1) / 8
        expected[1, group, permutation[0]] = (group + 1) / 6
        expected[1, group, permutation[2]] = -(group + 1) / 6
        blocked = ([permutation[1], permutation[3]], [permutation[3]])
        for row in range(2):
            assert torch.equal(model.core.values.grad[row, group, blocked[row]], torch.zeros(len(blocked[row])))
    torch.testing.assert_close(model.core.values.grad, torch.from_numpy(expected), rtol=2e-7, atol=1e-7)
    shifted = scripted_head(values + np.arange(16, dtype=np.float32).reshape(2, 8, 1) * 4)
    shifted_target = target + torch.tensor([[32], [-16]], dtype=torch.float32) * mask
    shifted_losses = m.training_losses(shifted, x, shifted_target, mask)
    torch.testing.assert_close(shifted_losses, losses, rtol=0, atol=0)
    shifted_losses.sum().backward()
    torch.testing.assert_close(shifted.core.values.grad, model.core.values.grad, rtol=0, atol=0)


def test_qualified_head_uses_actual_eight_feature_views_and_one_action_loss_is_zero():
    import torch

    from openjev.research import otto_symmetry_head as head

    model = head.make_head('dense_augmented', 7)
    x = torch.zeros((1, 2836), dtype=torch.float32)
    x[0, 10 * 53 + 17], x[0, 2809], x[0, 2810] = 3, -0.5, 0.25
    target = torch.tensor([[0, 23, 0, 0]], dtype=torch.float32)
    mask = torch.tensor([[False, True, False, False]])
    observed = []
    handle = model.core.register_forward_pre_hook(lambda _module, args: observed.append(args[0].detach().clone()))
    try:
        losses = m.training_losses(model, x, target, mask)
    finally:
        handle.remove()
    assert len(observed) == 1 and observed[0].shape == (1, 8, 2836)
    base_grid = x[0, :2809].numpy().reshape(53, 53)
    for group in range(8):
        expected = np.rot90(base_grid[:, ::-1] if group >= 4 else base_grid, group % 4)
        np.testing.assert_array_equal(observed[0][0, group, :2809].numpy().reshape(53, 53), expected)
    assert torch.equal(losses, torch.zeros(1))
    losses.sum().backward()
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in model.parameters())
