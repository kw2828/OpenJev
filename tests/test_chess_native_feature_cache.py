import copy

import chess
import pytest
import torch

from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_graph_contrast import candidate_graphs
from openjev.research.chess_native_feature_cache import arguments, build
from openjev.research.chess_spatial import encode_board, encode_candidates
from openjev.research.chess_transport import candidate_features


def model():
    return CandidateChess('direct', seed=97, width=32, root_depth=1).eval().requires_grad_(False)


def boards():
    start = chess.Board(); start.push_uci('e2e4'); start.push_uci('c7c5')
    return [start, chess.Board('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1')]


def graph(positions):
    native = candidate_graphs(positions)
    return {**native, 'children': native['children'][native['mask']],
            'menus': list(map(tuple, native['menus'])), 'fens': [b.fen(en_passant='fen') for b in positions]}


def test_exact_native_features_preserve_boards_weights_and_rng():
    net = model(); positions = boards(); before = copy.deepcopy(positions)
    rng = torch.get_rng_state().clone(); state = {k: v.clone() for k, v in net.state_dict().items()}
    cache = build(net, positions)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in state.items())
    for i, board in enumerate(positions):
        assert board.fen() == before[i].fen() and board.move_stack == before[i].move_stack
        names, encoded = encode_candidates(board); candidates = torch.from_numpy(encoded)[None]
        scores, _, hidden = net(torch.from_numpy(encode_board(board))[None], candidates, torch.ones(1, len(names), dtype=torch.bool))
        nodes = hidden.flatten(2).transpose(1, 2); actions = candidate_features(net, nodes, candidates)
        lo, hi = cache['offsets'][i:i+2].tolist()
        assert torch.equal(cache['nodes'][i], nodes[0])
        assert torch.equal(cache['action_features'][lo:hi], actions[0])
        assert torch.equal(cache['base_logits'][lo:hi], scores[0])
    assert all(not v.requires_grad for v in cache.values() if torch.is_tensor(v))


def test_builder_grouping_and_order_do_not_change_any_features():
    net = model(); positions = boards(); together = build(net, positions)
    reverse = build(net, positions[::-1])
    for i, board in enumerate(positions):
        single = build(net, [board]); lo, hi = together['offsets'][i:i+2].tolist()
        j = 1-i; a, b = reverse['offsets'][j:j+2].tolist()
        assert torch.equal(together['nodes'][i], single['nodes'][0])
        assert torch.equal(together['nodes'][i], reverse['nodes'][j])
        for key in ('action_features', 'base_logits', 'candidates'):
            assert torch.equal(together[key][lo:hi], single[key])
            assert torch.equal(together[key][lo:hi], reverse[key][a:b])


def test_ragged_candidate_packing_reordering_and_graph_identity_checks():
    positions = boards(); cache = build(model(), positions); index = torch.tensor([1, 0, 1])
    selected = [positions[i] for i in index]; native = graph(selected)
    args = arguments(cache, index, native)
    assert torch.equal(args[3], native['mask']) and torch.isneginf(args[4][~args[3]]).all()
    for slot, i in enumerate(index):
        lo, hi = cache['offsets'][i:i+2].tolist()
        assert torch.equal(args[1][slot, :hi-lo], cache['action_features'][lo:hi])
        assert torch.equal(args[2][slot, :hi-lo], cache['candidates'][lo:hi])
        assert torch.equal(args[4][slot, :hi-lo], cache['base_logits'][lo:hi])
    bad = copy.deepcopy(native); bad['menus'][0] = bad['menus'][0][::-1]
    with pytest.raises(ValueError): arguments(cache, index, bad)
    bad = copy.deepcopy(native); bad['fens'][0] = chess.STARTING_FEN
    with pytest.raises(ValueError): arguments(cache, index, bad)
    with pytest.raises(ValueError): arguments(cache, torch.tensor([-1]), graph([positions[0]]))


def test_reject_training_or_mutable_or_wrong_precision_backbones_and_empty_inputs():
    for net in (model().train(), model().requires_grad_(True), model().double()):
        with pytest.raises(ValueError): build(net, boards())
    with pytest.raises(ValueError): build(model(), [])
    with pytest.raises(ValueError): build(model(), [chess.Board(None)])
