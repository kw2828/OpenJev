# SPDX-License-Identifier: GPL-3.0-only
from itertools import product

import pytest
import torch

from openjev.research.chess_pin_factor_head import ChessPinFactorHead
from openjev.research.chess_pin_pairwise import ChessPinPairwiseHead
from test_chess_pin_factor_head import factors
from test_chess_graph_contrast import fixture


def head():
    result = ChessPinPairwiseHead(seed=311).double()
    with torch.no_grad(): result.output.weight.copy_(torch.linspace(-.15, .15, 29)[None])
    return result


def test_same_parameters_initial_state_rng_and_zero_output():
    state = torch.get_rng_state().clone(); model = ChessPinPairwiseHead(seed=311)
    joint = ChessPinFactorHead('joint', seed=311)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 16658
    assert all(torch.equal(v, joint.state_dict()[k]) for k, v in model.state_dict().items())
    args = fixture(); assert torch.equal(model.double()(*args, factors()), args[4])


def test_pairwise_slices_preserved_and_three_way_threshold_removed():
    model = head(); generator = torch.Generator().manual_seed(973)
    values = torch.randn(7, 197, dtype=torch.double, generator=generator)
    for role in range(3):
        anchored = values.clone(); anchored[:, role*64:(role+1)*64] = 0
        torch.testing.assert_close(model.factor_value(anchored), model.factor(anchored), atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        for p in model.factor.parameters(): p.zero_()
        model.factor[0].weight[0, [0, 64, 128]] = 1
        model.factor[0].bias[0] = -2.5; model.factor[2].weight[0, 0] = 1
    x = torch.zeros(1, 197, dtype=torch.double); x[0, [0, 64, 128, 192]] = 1
    assert model.factor(x)[0, 0] == .5
    assert model.factor_value(x)[0, 0] == 0
    with torch.no_grad():
        model.factor[0].weight[0, 128] = 0; model.factor[0].bias[0] = -1.5
    assert model.factor(x)[0, 0] == .5 and model.factor_value(x)[0, 0] == .5


def test_third_mixed_difference_vanishes_with_fixed_metadata():
    model = head(); generator = torch.Generator().manual_seed(981)
    a = torch.randn(9, 197, dtype=torch.double, generator=generator)
    b = torch.randn(9, 197, dtype=torch.double, generator=generator); b[:, 192:] = a[:, 192:]
    terms = []
    for bits in product((0, 1), repeat=3):
        x = torch.cat([*(b[:, i*64:(i+1)*64] if bit else a[:, i*64:(i+1)*64] for i, bit in enumerate(bits)), a[:, 192:]], -1)
        terms.append((-1)**(3-sum(bits))*model.factor_value(x))
    torch.testing.assert_close(torch.stack(terms).sum(0), torch.zeros_like(terms[0]), atol=1e-12, rtol=0)


def test_full_score_input_and_parameter_gradients_match_subset_reference():
    model = head(); args = list(fixture()); args[0].requires_grad_(); args[1].requires_grad_()
    actual = model(*args, factors()); expected = model(*args, factors(), reference=True)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    variables = list(model.parameters())+args[:2]
    left = torch.autograd.grad(actual[args[3]].square().sum(), variables, retain_graph=True)
    right = torch.autograd.grad(expected[args[3]].square().sum(), variables)
    for a, b in zip(left, right): torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    gradients = [g for (n, _), g in zip(model.named_parameters(), left) if n.startswith('factor.')]
    assert len(gradients) == 4 and all(g.abs().sum() > 0 for g in gradients)
    torch.testing.assert_close(actual, model(*args, factors().flip(0)), atol=1e-11, rtol=1e-11)
    compact = list(args); compact[-1] = args[-1][args[3]]
    torch.testing.assert_close(actual, model(*compact, factors()), atol=0, rtol=0)


def test_empty_factors_and_padded_candidate_reordering():
    model = head(); args = list(fixture()); empty = torch.empty(0, 6, dtype=torch.long)
    torch.testing.assert_close(model(*args, empty), model(*args, empty, reference=True), atol=0, rtol=0)
    f = factors(); actual = model(*args, f); order = torch.tensor([2, 0, 1])
    lookup = torch.full_like(args[3], -1, dtype=torch.long); lookup[args[3]] = torch.arange(int(args[3].sum()))
    new = list(args)
    for i in (1, 2, 3, 4, 6): new[i] = args[i][:, order]
    new_to_old = lookup[:, order][new[3]]; changed = f.clone(); changed[:, 0] = new_to_old.argsort()[f[:, 0]]
    torch.testing.assert_close(model(*new, changed), actual[:, order], atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('column,value', [(0, 5), (1, 2), (2, 64), (5, 3)])
def test_invalid_factors_rejected_in_both_paths(column, value):
    bad = factors(); bad[0, column] = value
    for reference in (False, True):
        with pytest.raises(ValueError): head()(*fixture(), bad, reference=reference)
