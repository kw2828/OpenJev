# SPDX-License-Identifier: GPL-3.0-only
import chess
import pytest
import torch

from openjev.research.chess_pin_factor_controls import ARMS, PARAMETERS, ChessPinFactorControl, pin_counts
from openjev.research.chess_pin_factor_head import ChessPinFactorHead, pack_factors
from openjev.research.chess_pin_factors import candidate_factors
from test_chess_graph_contrast import fixture
from test_chess_pin_factor_head import factors


def head(arm):
    model = ChessPinFactorControl(arm, seed=311).double()
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.15, .15, model.output.in_features)[None])
    return model


def test_parameter_budgets_backbone_preservation_and_common_encoder():
    args = fixture()
    rng = torch.get_rng_state().clone()
    original = ChessPinFactorHead(seed=311).double()
    for arm in ARMS:
        model = ChessPinFactorControl(arm, seed=311).double()
        assert torch.equal(torch.get_rng_state(), rng)
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[arm]
        for name, value in original.state_dict().items():
            if name.startswith(('encoder.', 'difference.')) or arm == 'root_only':
                assert torch.equal(value, model.state_dict()[name])
        assert torch.equal(model(*args, factors()), args[4])


@pytest.mark.parametrize('arm', ARMS)
def test_full_scores_parameter_input_gradients_and_active_branch(arm):
    args = list(fixture()); args[0].requires_grad_(); args[1].requires_grad_()
    model = head(arm)
    actual = model(*args, factors()); reference = model(*args, factors(), reference=True)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
    variables = list(model.parameters())+args[:2]
    a = torch.autograd.grad(actual[args[3]].square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(reference[args[3]].square().sum(), variables)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)
    gradients = [g for (name, _), g in zip(model.named_parameters(), a) if name.startswith('factor.')]
    assert len(gradients) == 4 and all(g.abs().sum() > 0 for g in gradients)
    torch.testing.assert_close(actual, model(*args, factors().flip(0)), atol=1e-11, rtol=1e-11)
    compact = list(args); compact[-1] = args[-1][args[3]]
    torch.testing.assert_close(actual, model(*compact, factors()), atol=0, rtol=0)


def test_counts_discard_identities_and_neural_features_but_keep_owner_and_status():
    f = factors(); expected = torch.tensor([[1, 0, 0, 0, 1, 0], [0]*6,
                                           [0, 0, 1, 0, 0, 0], [0]*6, [0, 0, 0, 1, 0, 0]], dtype=torch.double)
    assert torch.equal(pin_counts(f, 5, dtype=torch.double), expected)
    assert torch.equal(pin_counts(f, 5, dtype=torch.double, reference=True), expected)
    root = torch.randn(5, 64, 32, dtype=torch.double); delta = torch.randn_like(root)
    graph = delta.sum(1); model = head('counts')
    changed = f.clone(); changed[:, 2:5] = (changed[:, 2:5]+7) % 64
    actual = model.branch(root, delta, graph, f)
    assert torch.equal(actual, model.branch(root*3, -delta, -graph, changed))
    changed[0, 5] = 1
    assert not torch.equal(actual, model.branch(root, delta, graph, changed))


def test_root_branch_is_constant_across_native_candidates_and_ignores_child_deltas():
    record = candidate_factors(chess.Board('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1'))
    f = pack_factors([record]); count = len(record['candidates'])
    root = torch.randn(1, 64, 32, dtype=torch.double).expand(count, -1, -1)
    delta = torch.randn_like(root); graph = delta.sum(1); model = head('root_only')
    actual = model.branch(root, delta, graph, f)
    torch.testing.assert_close(actual, actual[:1].expand_as(actual), atol=0, rtol=0)
    assert torch.equal(actual, model.branch(root, -delta*100, graph*17, f))
    before = torch.tensor([(c, *w, 0) for c in range(count) for w in record['before']])
    direct = ChessPinFactorHead(seed=311).double().pool(root, torch.zeros_like(delta), before, reference=True)
    torch.testing.assert_close(actual, direct, atol=1e-12, rtol=1e-12)
    # Added factors must not leak into the root branch.
    added = torch.tensor([[0, 1, 0, 7, 63, 1]])
    assert torch.equal(actual, model.branch(root, delta, graph, torch.cat((f, added))))
    assert torch.equal(model.branch(root, delta, graph, added), torch.zeros(count, 32, dtype=torch.double))


def test_conventional_branch_uses_only_existing_pooled_graph():
    root = torch.randn(5, 64, 32, dtype=torch.double); delta = torch.randn_like(root)
    graph = delta.sum(1); model = head('graph_mlp')
    empty = torch.empty(0, 6, dtype=torch.long)
    actual = model.branch(root, delta, graph, factors())
    assert torch.equal(actual, model.branch(-root, -delta, graph, empty))
    assert not torch.equal(actual, model.branch(root, delta, -graph, factors()))


@pytest.mark.parametrize('arm', ARMS)
def test_candidate_reordering_with_padding(arm):
    args = list(fixture()); model = head(arm); f = factors()
    actual = model(*args, f); order = torch.tensor([2, 0, 1])
    old_lookup = torch.full_like(args[3], -1, dtype=torch.long)
    old_lookup[args[3]] = torch.arange(int(args[3].sum()))
    changed_args = list(args)
    for i in (1, 2, 3, 4, 6): changed_args[i] = args[i][:, order]
    old_to_new = old_lookup[:, order][changed_args[3]].argsort()
    changed_factors = f.clone(); changed_factors[:, 0] = old_to_new[f[:, 0]]
    torch.testing.assert_close(model(*changed_args, changed_factors), actual[:, order], atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('arm', ARMS)
def test_invalid_discarded_fields_still_rejected(arm):
    model = head(arm)
    for column, value in ((0, 5), (1, 2), (2, 64), (5, 3), (2, -1), (3, 4)):
        f = factors(); f[0, column] = value
        with pytest.raises(ValueError): model(*fixture(), f)
