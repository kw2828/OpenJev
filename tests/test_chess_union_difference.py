# SPDX-License-Identifier: GPL-3.0-only
import chess
import pytest
import torch

from openjev.research.chess_union_difference import (
    ARMS, ChessUnionDifferenceHead, difference_edges, reference_edges, relation_flags, rotate_classes)
from openjev.research.chess_graph_contrast import candidate_graphs
from openjev.research.wldn_baseline import ChessWLDNHead
from test_chess_graph_contrast import fixture


@pytest.mark.parametrize('arm', ARMS)
def test_packing_values_and_gradients_match_independent_reference(arm):
    args = list(fixture()); args[0].requires_grad_(); args[1].requires_grad_()
    head = ChessUnionDifferenceHead(arm, seed=251).double()
    assert sum(p.numel() for p in head.parameters()) == 16740
    assert torch.equal(head(*args), args[4])
    with torch.no_grad(): head.output.weight.copy_(torch.linspace(-.1,.1,58)[None])
    actual, expected = head(*args), head(*args, reference=True)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    variables = list(head.parameters())+args[:2]
    left = torch.autograd.grad(actual[args[3]].square().sum(), variables, retain_graph=True)
    right = torch.autograd.grad(expected[args[3]].square().sum(), variables)
    for a,b in zip(left,right): torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    compact = list(args); compact[-1] = args[-1][args[3]]
    assert torch.equal(head(*compact), actual)


def test_all_arms_start_identically_and_child_can_embed_frozen_wldn_exactly():
    heads = [ChessUnionDifferenceHead(a, seed=257).double() for a in ARMS]
    states = [h.state_dict() for h in heads]
    assert all(all(torch.equal(state[k], v) for k,v in states[0].items()) for state in states[1:])
    # The 58-unit readout embeds a59-unit original with its final unit disabled.
    old = ChessWLDNHead(seed=263).double(); head = heads[0]
    with torch.no_grad():
        old.output.weight.fill_(.1); old.output.weight[0,-1] = 0
        head.encoder.load_state_dict(old.encoder.state_dict())
        head.difference.message.weight[:,:36].copy_(old.difference.message.weight)
        head.difference.message.weight[:,36:].zero_()
        head.difference.message.bias.copy_(old.difference.message.bias)
        head.difference.update.load_state_dict(old.difference.update.state_dict())
        head.readout.weight.copy_(old.readout.weight[:58]); head.readout.bias.copy_(old.readout.bias[:58])
        head.output.weight.copy_(old.output.weight[:,:58])
    args = fixture()
    torch.testing.assert_close(head(*args), old(*args), atol=1e-12,rtol=1e-12)


def test_edit_classes_reconstruct_root_child_and_rotation_preserves_counts():
    args = fixture(); owner,_ = args[3].nonzero(as_tuple=True)
    root,child = args[5][owner],args[6][args[3]]
    index,flags = difference_edges(root,child,'edits',torch.double)
    before,after = relation_flags(root),relation_flags(child)
    dense = torch.zeros(len(child),64,64,12,dtype=torch.double); dense[tuple(index)] = flags
    retained,added,removed = dense.split(4,-1)
    assert torch.equal((retained+removed).bool(),before)
    assert torch.equal((retained+added).bool(),after)
    ir,fr = difference_edges(root,child,'rotated',torch.double)
    assert torch.equal(index,ir)
    corrupt = torch.zeros_like(dense);corrupt[tuple(ir)] = fr
    assert torch.equal(corrupt.reshape(len(child),64,64,3,4).sum(3), dense.reshape(len(child),64,64,3,4).sum(3))
    assert torch.equal(corrupt.sum((1,2)),dense.sum((1,2)))
    assert not torch.equal(corrupt,dense)
    # Same edit class is moved in some places and retained in others: no single
    # global relabeling maps all original classes to the changed classes.
    original = (retained+2*added+3*removed).to(torch.uint8)
    changed = rotate_classes(original)
    assert any(len(set(changed[original==role].tolist())) > 1 for role in (1,2,3))


@pytest.mark.parametrize('fen',[
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
    '4k3/8/8/8/8/8/p7/4K3 b - - 0 1'])
def test_native_special_move_graphs_match_independent_packing(fen):
    graph = candidate_graphs([chess.Board(fen)])
    child = graph['children'][graph['mask']];root = graph['root'].expand(len(child),-1,-1,-1)
    for arm in ARMS:
        actual = difference_edges(root,child,arm,torch.double)
        expected = reference_edges(root,child,arm,torch.double)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))


def test_empty_and_singleton_rotations_and_bad_graphs():
    empty = torch.zeros(2,2,64,64,dtype=torch.uint8)
    for arm in ARMS:
        index,flags = difference_edges(empty,empty,arm,torch.double)
        assert index.shape == (3,0) and flags.shape == (0,12)
    labels = torch.zeros(2,64,64,4,dtype=torch.uint8);labels[1,2,3,0]=3
    assert torch.equal(rotate_classes(labels),labels)
    bad = empty.clone();bad[0,0,0,0]=1
    with pytest.raises(ValueError): difference_edges(bad,empty,'edits',torch.double)
    with pytest.raises(ValueError): ChessUnionDifferenceHead('unknown')
