import numpy as np
import pytest
import torch
import train_pose_crossfit as runner

from openjev.research.pose_support import PoseSupport
from openjev.research.pose_transport import PoseTransport
from openjev.research.rigid_motion import so3_exp


def ids():
    return np.column_stack((np.repeat(np.arange(30), 24), np.tile(np.arange(24), 30)))


def test_whole_parent_split_balanced_disjoint_and_complete():
    values = ids()
    groups = runner.partition(values)
    for k in range(3):
        held = values[groups == k, 0]
        trained = values[groups != k, 0]
        assert len(held) == 240 and len(trained) == 480
        assert set(held).isdisjoint(set(trained))
        assert len(set(held)) == 10 and len(set(trained)) == 20


@pytest.mark.parametrize('kind', ['dtype', 'parents', 'counts'])
def test_invalid_parent_identity_rejected(kind):
    values = ids()
    if kind == 'dtype':
        values = values.astype(np.float32)
    elif kind == 'parents':
        values[0, 0] = 30
    else:
        values[0, 0] = 1
    with pytest.raises(ValueError):
        runner.partition(values)


def test_excluded_poses_and_actions_cannot_change_fold_normalization():
    generator = torch.Generator().manual_seed(2942)
    p = torch.randn(720, 57, 3, generator=generator) * .01
    r = so3_exp(torch.randn(720, 57, 3, generator=generator) * .04)
    a = torch.randn(720, 56, 40, generator=generator)
    groups = runner.partition(ids())
    first = runner.fold_normalization(p, r, a, groups, 1)
    p[groups == 1] = float('nan')
    r[groups == 1] = float('nan')
    a[groups == 1] = float('nan')
    second = runner.fold_normalization(p, r, a, groups, 1)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize('regime', ['is', 'oof'])
def test_cache_assignment_uses_correct_exclusion_without_reordering_cases(regime):
    groups = runner.partition(ids())
    caches = {k: tuple(torch.full((720, 25, *tail), 10*k+j, dtype=torch.float32)
                      for j, tail in enumerate(((3,), (3, 3), (3,), (3, 3)))) for k in range(3)}
    result, folds = runner.mix_cache(caches, groups, regime)
    for i in range(720):
        expected = groups[i] if regime == 'oof' else (groups[i]+1) % 3
        assert folds[i] == expected
        for j in range(4):
            torch.testing.assert_close(result[j][i], caches[int(expected)][j][i], rtol=0, atol=0)
    assert np.bincount(folds).tolist() == [240, 240, 240]


@pytest.mark.parametrize('kind', ['summary', 'recurrent'])
def test_gate_initial_weights_match_between_training_regimes(kind):
    states = []
    for regime in ('is', 'oof'):
        torch.manual_seed(1101)
        states.append(runner.make_gate(regime+'_'+kind).state_dict())
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


@pytest.mark.parametrize('variant', runner.VARIANTS)
def test_forecast_excludes_future_poses_and_later_actions(variant):
    generator = torch.Generator().manual_seed(732)
    p = torch.randn(2, 32, 3, generator=generator) * .01
    r = so3_exp(torch.randn(2, 32, 3, generator=generator) * .04)
    a = torch.randn(2, 56, 40, generator=generator) * .1
    scales = torch.tensor([.01, .02, .005, .005])
    fast, slow = PoseSupport(scales), PoseTransport('body', scales)
    gate = runner.make_gate(variant) if variant in runner.TRAIN_VARIANTS else None
    alpha = [.2, .7] if variant in runner.CONSTANT_VARIANTS else None
    first = runner.forecast(fast, slow, gate, alpha, variant, p, r, a[:, :31], a[:, 31:])
    changed = a.clone()
    changed[:, 41:] = 99
    second = runner.forecast(fast, slow, gate, alpha, variant, p, r, a[:, :31], changed[:, 31:])
    for x, y in zip(first[:2], second[:2], strict=True):
        torch.testing.assert_close(x[:, :10], y[:, :10], rtol=0, atol=0)
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    with pytest.raises(ValueError):
        runner.forecast(fast, slow, gate, alpha, variant, torch.cat((p, p), 1), torch.cat((r, r), 1), a[:, :31], a[:, 31:])
