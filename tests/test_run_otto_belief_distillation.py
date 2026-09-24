"""Fabricated probability-target, terminal and privileged-input checks."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openjev.research.otto_action_latent_model import make_model

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('_belief_runner_test', SCRIPTS / 'run_otto_belief_distillation.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def data(h=4):
    n = 3
    outcomes = np.tile(np.arange(h, dtype=np.int64) % 4, (n, 1))
    outcomes[1, 1:] = 4
    outcomes[2] = 4
    alive = outcomes != 4
    normal = np.full((n, h, 5), .2, np.float64)
    normal[1, 2:] = [0, 0, 0, 0, 1]
    normal[2, 1:] = [0, 0, 0, 0, 1]
    return {'prefix': np.zeros((n, 9, 31), np.float32), 'prefix_lengths': np.full(n, 9, np.int64),
        'actions': np.tile(np.arange(h, dtype=np.int64) % 4, (n, 1)),
        'continuation': np.zeros((n, h, 31), np.float32), 'outcomes': outcomes,
        'raw_costs': np.zeros((n, h, 4), np.float32), 'legal': np.broadcast_to(alive[..., None], (n, h, 4)).copy(),
        'case_ids': np.array(['fake:a', 'fake:b', 'fake:c']), 'regimes': np.array(['lambda3'] * n),
        'initial_belief': np.full((n, 2809), 1 / 2809, np.float64), 'prefix_actions': np.zeros((n, 8), np.int64),
        'prefix_outcomes': np.zeros((n, 8), np.int64), 'prefix_position': np.full((n, 2), 26, np.int64),
        'gap_oracle': np.full((n, h, 5), .2, np.float64), 'normal_oracle': normal,
        'opposite_oracle': np.full((n, h, 5), .2, np.float64)}


def tensors(d):
    return {k: torch.from_numpy(d[k].copy()) for k in (*r.INPUT_KEYS, *r.TARGET_KEYS)}


def prediction():
    return {'outcome_logits': torch.zeros(3, 4, 5, requires_grad=True),
        'cost_contrasts': torch.zeros(3, 4, 4, requires_grad=True),
        'aux_features': torch.zeros(3, 4, 2, requires_grad=True)}


@pytest.mark.parametrize('h', (4, 8))
def test_archive_roundtrip_and_strict_schema(tmp_path, h):
    d = data(h)
    path = tmp_path / 'data.npz'
    r.save_npz(path, np, **d)
    loaded = r.load_data(path, np, h)
    assert set(d) == set(loaded)
    for key in d:
        np.testing.assert_array_equal(d[key], loaded[key])
        assert not np.shares_memory(d[key], loaded[key])
    with pytest.raises(FileExistsError):
        r.save_npz(path, np, **d)
    with pytest.raises(ValueError, match='schema'):
        r.load_data(path, np, 8 if h == 4 else 4)


@pytest.mark.parametrize('defect', ('extra', 'oracle_dtype', 'oracle_nan', 'oracle_negative', 'oracle_mass',
    'target_zero', 'resolved', 'unabsorbed', 'duplicate', 'future_nan', 'found_legal', 'prefix_outcome', 'action', 'position'))
def test_reject_bad_archives_before_fit(tmp_path, defect):
    d = data()
    if defect == 'extra':
        d['secret'] = np.ones(3)
    elif defect == 'oracle_dtype':
        d['gap_oracle'] = d['gap_oracle'].astype(np.float32)
    elif defect == 'oracle_nan':
        d['opposite_oracle'][0, 0, 0] = np.nan
    elif defect == 'oracle_negative':
        d['gap_oracle'][0, 0] = [-.1, .1, .2, .3, .5]
    elif defect == 'oracle_mass':
        d['initial_belief'][0] *= .9
    elif defect == 'target_zero':
        d['normal_oracle'][0, 1] = [.5, 0, .5, 0, 0]
    elif defect == 'resolved':
        d['normal_oracle'][2, 3] = [.2] * 5
    elif defect == 'unabsorbed':
        d['outcomes'][1, 3] = 0
    elif defect == 'duplicate':
        d['case_ids'][1] = d['case_ids'][0]
    elif defect == 'future_nan':
        d['continuation'][0, 0, 0] = np.nan
    elif defect == 'found_legal':
        d['legal'][2] = True
    elif defect == 'prefix_outcome':
        d['prefix_outcomes'][0, 0] = 4
    elif defect == 'action':
        d['actions'][0, 0] = -1
    else:
        d['prefix_position'][0, 0] = 52
    path = tmp_path / 'invalid.npz'
    np.savez(path, **d)
    with pytest.raises(ValueError):
        r.load_data(path, np, 4)


@pytest.mark.parametrize('condition', ('gap', 'normal'))
@pytest.mark.parametrize('soft', (True, False))
def test_loss_terminal_gradient_and_fixed_denominator(condition, soft):
    d, p = tensors(data()), prediction()
    actual = r.loss_for(p, d, torch, condition=condition, soft_targets=soft)
    expected_rows = 12 if condition == 'gap' else 7
    assert actual.item() == pytest.approx(math.log(5) * expected_rows / 12, rel=1e-7)
    actual.backward()
    resolved = torch.zeros((3, 4), dtype=torch.bool)
    resolved[1, 2:] = True
    resolved[2, 1:] = True
    if condition == 'normal':
        assert not bool(p['outcome_logits'].grad[resolved].any())
    elif not soft:
        assert bool(p['outcome_logits'].grad[resolved].abs().sum() > 0)


def test_soft_target_cross_entropy_and_gradient_matches_weighted_hard_losses():
    d = tensors(data())
    p = prediction()
    p['outcome_logits'] = torch.linspace(-2, 2, 60).reshape(3, 4, 5).requires_grad_()
    d['gap_oracle'][:] = torch.tensor([.1, .2, .3, .1, .3], dtype=torch.float64)
    soft = r.loss_for(p, d, torch, condition='gap', soft_targets=True)
    weighted = 0
    for k, weight in enumerate((.1, .2, .3, .1, .3)):
        target = {**d, 'outcomes': torch.full((3, 4), k, dtype=torch.int64)}
        weighted = weighted + weight * r.loss_for(p, target, torch, condition='gap', soft_targets=False)
    torch.testing.assert_close(soft, weighted.to(torch.float64), rtol=2e-7, atol=1e-7)
    a = torch.autograd.grad(soft, p['outcome_logits'], retain_graph=True)[0]
    b = torch.autograd.grad(weighted, p['outcome_logits'])[0]
    torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-8)


def test_prediction_shortcut_uses_only_earlier_found_and_blind_rejects_outcomes():
    d = tensors(data())
    logits = torch.zeros((3, 4, 5))
    normal = r.probabilities(logits, torch, condition='normal', outcomes=d['outcomes'])
    assert normal.dtype == torch.float64
    torch.testing.assert_close(normal[2, 0], torch.full((5,), .2, dtype=torch.float64))
    torch.testing.assert_close(normal[2, 1], torch.tensor([0, 0, 0, 0, 1], dtype=torch.float64))
    with pytest.raises(ValueError, match='blind'):
        r.probabilities(logits, torch, condition='gap', outcomes=d['outcomes'])
    gap = r.probabilities(logits, torch, condition='gap')
    assert bool((gap > 0).all())


def test_float64_softmax_keeps_small_positive_probabilities_without_epsilon():
    logits = torch.tensor([[[0., -150., -200., -50., -1.]]])
    probs = r.probabilities(logits, torch, condition='gap')
    assert 0 < probs[0, 0, 2] < 1e-80
    with pytest.raises(ValueError, match='positive'):
        r.probabilities(logits * 10, torch, condition='gap')


@pytest.mark.parametrize('family', r.FAMILIES)
def test_projection_causality_and_paired_initialization(family):
    d = tensors(data())
    model = make_model(r.MODEL_KINDS[family], 73, cost_scale=.01)
    gap = model.blind_rollout(*(d[k] for k in r.INPUT_KEYS))
    d['gap_oracle'][:] = .2
    d['normal_oracle'][:] = .2
    d['raw_costs'][:] = 1e6
    other = model.blind_rollout(*(d[k] for k in r.INPUT_KEYS))
    torch.testing.assert_close(gap['outcome_logits'], other['outcome_logits'], rtol=0, atol=0)
    normal = model.normal_rollout(*(d[k] for k in r.INPUT_KEYS), d['continuation'], found=d['outcomes'] == 4)
    torch.testing.assert_close(gap['outcome_logits'][:, 0], normal['outcome_logits'][:, 0], rtol=0, atol=0)
    if family in ('recurrent_soft', 'recurrent_sampled'):
        twin = make_model('action_recurrent', 73, cost_scale=.01)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, twin.state_dict()[k], rtol=0, atol=0)


def test_cost_normalization_uses_only_train_supported_rows():
    d = data()
    d['raw_costs'][:] = [64, -64, 128, -128]
    scale, var = r.train_cost_scale(d, np, return_variance=True)
    assert var == pytest.approx(5 / 3)
    assert scale == float(np.float32(math.sqrt(5 / 3)))
    d['raw_costs'][d['outcomes'] == 4] = 1e6
    assert r.train_cost_scale(d, np, return_variance=True) == (scale, var)
