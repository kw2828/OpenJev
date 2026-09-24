"""Fabricated checks for the paired objective, causality and TRAIN-only scaling."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_otto_action_effect as r

from openjev.research.otto_action_latent_model import make_model


def batch():
    n, h = 3, 4
    outcomes = torch.tensor([[0, 1, 2, 3], [0, 4, 4, 4], [4, 4, 4, 4]])
    normal = torch.full((n, h, 5), .2, dtype=torch.float64)
    normal[1, 2:] = torch.tensor([0., 0., 0., 0., 1.])
    normal[2, 1:] = torch.tensor([0., 0., 0., 0., 1.])
    return {'prefix': torch.zeros(n, 9, 31), 'prefix_lengths': torch.full((n,), 9),
            'actions': torch.arange(h).repeat(n, 1), 'continuation': torch.zeros(n, h, 31),
            'outcomes': outcomes, 'raw_costs': torch.zeros(n, h, 4),
            'gap_oracle': torch.full((n, h, 5), .2, dtype=torch.float64), 'normal_oracle': normal,
            'opposite_oracle': torch.tensor([.1, .3, .2, .2, .2], dtype=torch.float64).repeat(n, h, 1)}


def prediction(offset=0.):
    return {'outcome_logits': (torch.linspace(-1., 1., 60).reshape(3, 4, 5) + offset * torch.arange(5)).requires_grad_(),
            'cost_contrasts': torch.zeros(3, 4, 4, requires_grad=True),
            'aux_features': torch.zeros(3, 4, 2, requires_grad=True)}


def scalar_loss(gap, normal, opposite, b, *, weight, variance):
    ce = [0., 0., 0.]
    effect = 0.
    for i in range(3):
        for h in range(4):
            vectors = [p['outcome_logits'][i, h].detach().numpy().astype(float) for p in (gap, normal, opposite)]
            distributions = [np.exp(v - v.max()) / np.exp(v - v.max()).sum() for v in vectors]
            for k, target in enumerate(('gap_oracle', 'normal_oracle', 'opposite_oracle')):
                if k == 1 and bool((b['outcomes'][i, :h] == 4).any()):
                    continue
                ce[k] += sum(-float(b[target][i, h, j]) * math.log(distributions[k][j]) for j in range(5)) / 12
            effect += sum((distributions[0][j] - distributions[2][j]
                           - float(b['gap_oracle'][i, h, j] - b['opposite_oracle'][i, h, j]))**2
                          for j in range(5)) / 12
    return .25 * ce[0] + .5 * ce[1] + .25 * ce[2] + weight * effect / variance


@pytest.mark.parametrize('family', r.FAMILIES)
def test_paired_loss_matches_independent_scalar_all_cases_all_horizons(family):
    b = batch()
    gap, normal, opposite = prediction(), prediction(.2), prediction(-.3)
    actual = r.paired_loss(gap, normal, opposite, b, torch, 1., effect_variance=.05, family=family)
    expected = scalar_loss(gap, normal, opposite, b, weight=.1 if family == 'effect_recurrent' else 0., variance=.05)
    assert actual.item() == pytest.approx(expected, rel=2e-7)
    actual.backward()
    assert bool(gap['outcome_logits'].grad[2, 1:].abs().sum() > 0)
    assert bool(opposite['outcome_logits'].grad[2, 1:].abs().sum() > 0)
    assert not bool(normal['outcome_logits'].grad[2, 1:].any())
    assert bool(normal['outcome_logits'].grad[2, 0].abs().sum() > 0)
    # Counterfactual branches receive no invented costs or observations.
    assert opposite['cost_contrasts'].grad is None
    assert opposite['aux_features'].grad is None


def test_signed_effect_gradient_includes_both_branches_and_scale_has_no_square_root():
    b, gap, normal, opposite = batch(), prediction(.4), prediction(), prediction(-.2)
    candidate = r.paired_loss(gap, normal, opposite, b, torch, 1., effect_variance=.04, family='effect_recurrent')
    control = r.paired_loss(gap, normal, opposite, b, torch, 1., effect_variance=.04, family='paired_recurrent')
    delta = torch.softmax(gap['outcome_logits'].double(), -1) - torch.softmax(opposite['outcome_logits'].double(), -1)
    oracle = b['gap_oracle'] - b['opposite_oracle']
    expected = .1 * (delta - oracle).square().sum(-1).mean() / .04
    torch.testing.assert_close(candidate - control, expected, rtol=1e-12, atol=1e-12)
    for p in (gap, opposite):
        actual_gradient = torch.autograd.grad(candidate - control, p['outcome_logits'], retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, p['outcome_logits'], retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert bool(actual_gradient.abs().sum() > 0)


def test_all_controls_use_identical_loss_for_identical_predictions():
    args = prediction(.1), prediction(), prediction(-.1), batch(), torch, .5
    values = [r.paired_loss(*args, effect_variance=.2, family=family) for family in r.FAMILIES[1:]]
    assert all(torch.equal(values[0], v) for v in values[1:])


def test_cost_auxiliary_weights_support_and_no_opposite_cost_loss():
    b, g, n, o = batch(), prediction(), prediction(), prediction()
    base = r.paired_loss(g, n, o, b, torch, .5, effect_variance=1., family='paired_recurrent')
    b['raw_costs'][:] = torch.tensor([64., -64., 128., -128.])
    b['continuation'][..., 19:21] = 2.
    actual = r.paired_loss(g, n, o, b, torch, .5, effect_variance=1., family='paired_recurrent')
    # Two of three cases supported; each is averaged over its surviving rows.
    assert (actual - base).item() == pytest.approx((2.5 / .5**2 + .1 * 4) * 2 / 3, rel=1e-7)
    b['raw_costs'][b['outcomes'] == 4] = 1e6
    b['continuation'][b['outcomes'] == 4] = 1e6
    o['cost_contrasts'] = torch.full((3, 4, 4), 1e6)
    o['aux_features'] = torch.full((3, 4, 2), 1e6)
    check = r.paired_loss(g, n, o, b, torch, .5, effect_variance=1., family='paired_recurrent')
    torch.testing.assert_close(check, actual, rtol=0, atol=0)


def test_train_effect_variance_retains_zero_signal_cases_and_found_suffixes():
    d = {'gap_oracle': np.full((2, 4, 5), .2), 'opposite_oracle': np.full((2, 4, 5), .2)}
    assert r.train_effect_variance(d, np, return_variance=True) == (1e-6, 0.)
    d['opposite_oracle'][0] = [.1, .3, .2, .2, .2]
    scale, var = r.train_effect_variance(d, np, return_variance=True)
    assert scale == var == pytest.approx(.01)
    d['outcomes'] = np.full((2, 4), 4)
    assert r.train_effect_variance(d, np) == scale
    d['opposite_oracle'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='variance'):
        r.train_effect_variance(d, np)


@pytest.mark.parametrize('family', r.FAMILIES)
def test_input_projection_opposite_branch_and_paired_initialization(family):
    b = batch()
    model = make_model(r.MODEL_KINDS[family], 97, cost_scale=.01)
    a = model.blind_rollout(*(b[k] for k in r.INPUT_KEYS))
    opposite = model.blind_rollout(b['prefix'], b['prefix_lengths'], b['actions'] ^ 1)
    b['gap_oracle'][:] = 1e6
    b['opposite_oracle'][:] = 1e6
    b['raw_costs'][:] = 1e6
    again = model.blind_rollout(*(b[k] for k in r.INPUT_KEYS))
    torch.testing.assert_close(a['outcome_logits'], again['outcome_logits'], rtol=0, atol=0)
    if family == 'paired_blind':
        torch.testing.assert_close(a['outcome_logits'], opposite['outcome_logits'], rtol=0, atol=0)
    if family in ('effect_recurrent', 'paired_recurrent'):
        twin = make_model('action_recurrent', 97, cost_scale=.01)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, twin.state_dict()[k], rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', ({'effect_variance': 0.}, {'effect_variance': float('inf')}, {'family': 'unregistered'}))
def test_loss_rejects_invalid_condition(kwargs):
    options = {'effect_variance': 1., 'family': 'effect_recurrent', **kwargs}
    with pytest.raises(ValueError, match='inputs'):
        r.paired_loss(prediction(), prediction(), prediction(), batch(), torch, 1., **options)
