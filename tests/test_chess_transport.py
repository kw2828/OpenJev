import chess
import numpy as np
import pytest
import torch

from openjev.research.chess_candidate import CandidateChess, encode_batch
from openjev.research.chess_transport import (
    ARMS, TransportHead, degree_rewire, frozen_features, root_relations, transitions,
)


def fixture():
    boards = [chess.Board(), chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2')]
    inputs, _ = encode_batch(boards, 'direct')
    backbone = CandidateChess('direct', seed=19, width=8)
    nodes, features, logits, _ = frozen_features(backbone, **inputs)
    args = (nodes, features, inputs['candidates'], inputs['legal_mask'], logits,
            torch.from_numpy(np.stack([root_relations(b) for b in boards])))
    return backbone, args


def nonzero_head(arm='transport', steps=3):
    model = TransportHead(arm, width=8, seed=23, steps=steps)
    with torch.no_grad():
        model.output.weight.fill_(.2)
    return model


@pytest.mark.parametrize('arm', ARMS)
def test_zero_residual_preserves_backbone_and_padding(arm):
    _, args = fixture()
    model = TransportHead(arm, width=8)
    actual = model(*args)
    torch.testing.assert_close(actual, args[4], atol=0, rtol=0)
    assert torch.isneginf(actual[~args[3]]).all()


@pytest.mark.parametrize('arm', ARMS)
def test_conservation_and_candidate_permutation(arm):
    _, args = fixture()
    model = nonzero_head(arm)
    logits, trace = model(*args, trace=True)
    for step in trace:
        for key in ('left','right'):
            assert (step[key] >= 0).all()
            torch.testing.assert_close(step[key].sum(-1), torch.ones_like(logits), atol=2e-6, rtol=0)
    perm = torch.arange(logits.shape[1]-1,-1,-1)
    changed = (args[0], args[1][:,perm], args[2][:,perm], args[3][:,perm], args[4][:,perm], args[5])
    torch.testing.assert_close(model(*changed), logits[:,perm], atol=2e-6, rtol=0)


def test_transition_direction_empty_rows_and_reverse():
    edges = torch.zeros(1,2,64,64,dtype=torch.uint8)
    edges[0,0,2,7] = 1
    a = transitions(edges, torch.float32)
    assert a[0,0,2,7] == a[0,1,7,2] == 1
    assert a[0,0,7,7] == 1
    torch.testing.assert_close(a.sum(-1), torch.ones(1,4,64), atol=0, rtol=0)


def test_raw_graph_matches_native_attacks_and_color_mirror():
    board = chess.Board()
    edges = root_relations(board)
    for square in range(64):
        piece = board.piece_at(square)
        expected = set(board.attacks(square)) if piece else set()
        observed = set(np.flatnonzero(edges[:,square].any(0)))
        assert observed == expected
    np.testing.assert_array_equal(edges, root_relations(board.mirror()))
    assert not np.diagonal(edges,axis1=1,axis2=2).any()


def test_rewire_degrees_and_reproducibility():
    edges = root_relations(chess.Board())
    actual, records = degree_rewire(edges, seed=29)
    again, _ = degree_rewire(edges, seed=29)
    np.testing.assert_array_equal(actual, again)
    np.testing.assert_array_equal(actual.sum(1), edges.sum(1))
    np.testing.assert_array_equal(actual.sum(2), edges.sum(2))
    assert not np.array_equal(actual, edges)
    assert not np.diagonal(actual,axis1=1,axis2=2).any()
    assert all(r['accepted_swaps'] == r['target_swaps'] for r in records)
    empty = np.zeros_like(edges)
    singleton = empty.copy(); singleton[0,0,1] = 1
    for graph in (empty,singleton):
        rewired, _ = degree_rewire(graph, seed=31)
        np.testing.assert_array_equal(graph,rewired)


def test_router_and_overlap_have_gradients_backbone_has_none():
    backbone, args = fixture()
    model = nonzero_head()
    model(*args)[args[3]].square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in backbone.parameters())


def test_static_ignores_edges_uniform_ignores_router():
    _, args = fixture()
    static = nonzero_head('static')
    torch.testing.assert_close(static(*args), static(*args[:-1],torch.zeros_like(args[-1])),atol=0,rtol=0)
    uniform = nonzero_head('uniform')
    before = uniform(*args)
    with torch.no_grad():
        uniform.router.weight.add_(100)
        uniform.router.bias.add_(100)
    torch.testing.assert_close(before, uniform(*args),atol=0,rtol=0)


def test_overlap_ablation_is_specific_at_first_step():
    _, args = fixture()
    model, ablated = nonzero_head(steps=1), nonzero_head('no_overlap',steps=1)
    _, original = model(*args,trace=True)
    _, changed = ablated(*args,trace=True)
    torch.testing.assert_close(original[0]['left'],changed[0]['left'],atol=0,rtol=0)
    assert original[0]['overlap_mass'].max() > 0
    assert changed[0]['overlap_mass'].count_nonzero() == 0


def test_graph_node_permutation_equivariance():
    _, args = fixture()
    model = nonzero_head()
    perm = torch.arange(63,-1,-1)
    candidates = args[2].clone()
    candidates[...,:2] = 63-candidates[...,:2]
    changed = (args[0][:,perm],args[1],candidates,args[3],args[4],args[5][:,:,perm][:,:,:,perm])
    torch.testing.assert_close(model(*args),model(*changed),atol=2e-6,rtol=0)


def test_validation_and_rng_isolation():
    _, args = fixture()
    state = torch.get_rng_state().clone()
    model = nonzero_head()
    assert torch.equal(state,torch.get_rng_state())
    with pytest.raises(ValueError,match='binary'):
        model(*args[:-1],torch.full_like(args[-1],2))
    with pytest.raises(ValueError,match='nonempty'):
        model(args[0],args[1],args[2],torch.zeros_like(args[3]),args[4],args[5])
