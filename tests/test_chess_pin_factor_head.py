# SPDX-License-Identifier: GPL-3.0-only
import chess
import pytest
import torch

from openjev.research.chess_pin_factor_head import ChessPinFactorHead, pack_factors
from openjev.research.chess_pin_factors import candidate_factors
from test_chess_graph_contrast import fixture


def factors():
    return torch.tensor([[0, 0, 4, 12, 60, 0], [0, 1, 60, 52, 4, 1],
                         [2, 0, 4, 13, 31, 2], [4, 1, 60, 44, 12, 0]])


def head(arm):
    result = ChessPinFactorHead(arm, seed=311).double()
    with torch.no_grad():
        result.output.weight.copy_(torch.linspace(-.15, .15, 29)[None])
    return result


def test_parameter_count_common_initial_state_and_rng():
    rng = torch.get_rng_state().clone()
    joint, separate = ChessPinFactorHead('joint').double(), ChessPinFactorHead('separable').double()
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(p.numel() for p in joint.parameters()) == 16658
    assert all(torch.equal(v, separate.state_dict()[k]) for k, v in joint.state_dict().items())
    args = fixture()
    assert torch.equal(joint(*args, factors()), args[4])
    assert torch.equal(separate(*args, factors()), args[4])


@pytest.mark.parametrize('arm', ['joint', 'separable'])
def test_vectorized_reference_scores_and_gradients(arm):
    model = head(arm); args = list(fixture())
    args[0].requires_grad_(); args[1].requires_grad_()
    actual = model(*args, factors()); reference = model(*args, factors(), reference=True)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
    variables = list(model.parameters())+args[:2]
    a = torch.autograd.grad(actual[args[3]].square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(reference[args[3]].square().sum(), variables)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)
    factor_gradients = [gradient for (name, _), gradient in zip(model.named_parameters(), a) if name.startswith('factor.')]
    assert len(factor_gradients) == 4 and all(x.abs().sum() > 0 for x in factor_gradients)
    torch.testing.assert_close(actual, model(*args, factors().flip(0)), atol=1e-11, rtol=1e-11)
    compact = list(args); compact[-1] = args[-1][args[3]]
    torch.testing.assert_close(actual, model(*compact, factors()), atol=0, rtol=0)


def test_separable_matches_single_role_and_removes_constructed_cross_role_interaction():
    joint, separate = head('joint'), head('separable')
    x = torch.zeros(1, 197, dtype=torch.double)
    x[0, 0] = 1; x[0, 192] = 1
    torch.testing.assert_close(joint.factor_value(x), separate.factor_value(x), atol=1e-12, rtol=1e-12)
    for model in (joint, separate):
        with torch.no_grad():
            for parameter in model.factor.parameters(): parameter.zero_()
            model.factor[0].weight[0, 0] = 1
            model.factor[0].weight[0, 64] = 1
            model.factor[0].bias[0] = -1.5
            model.factor[2].weight[0, 0] = 1
    x[0, 64] = 1
    assert joint.factor_value(x)[0, 0] == .5
    assert separate.factor_value(x)[0, 0] == 0


def test_empty_factors_agree_and_pack_native_menus():
    empty = torch.empty(0, 6, dtype=torch.long)
    torch.testing.assert_close(head('joint')(*fixture(), empty), head('separable')(*fixture(), empty), atol=0, rtol=0)
    records = [candidate_factors(chess.Board()),
               candidate_factors(chess.Board('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1'))]
    packed = pack_factors(records)
    assert packed.dtype == torch.long and packed.shape[1] == 6
    assert packed[:, 0].min() >= len(records[0]['candidates'])
    assert packed[:, 0].max() < sum(len(x['candidates']) for x in records)


def test_padded_candidate_reordering_preserves_factor_alignment():
    args = list(fixture()); model = head('joint'); f = factors()
    actual = model(*args, f)
    order = torch.tensor([2, 0, 1])
    old_lookup = torch.full_like(args[3], -1, dtype=torch.long)
    old_lookup[args[3]] = torch.arange(int(args[3].sum()))
    permuted = list(args)
    for index in (1, 2, 3, 4, 6):
        permuted[index] = args[index][:, order]
    new_to_old = old_lookup[:, order][permuted[3]]
    old_to_new = new_to_old.argsort()
    changed = f.clone(); changed[:, 0] = old_to_new[f[:, 0]]
    torch.testing.assert_close(model(*permuted, changed), actual[:, order], atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('column,value', [(0, 5), (1, 2), (2, 64), (5, 3)])
def test_reject_bad_factor_indices(column, value):
    bad = factors().clone(); bad[0, column] = value
    with pytest.raises(ValueError):
        head('joint')(*fixture(), bad)
