# SPDX-License-Identifier: GPL-3.0-only
import copy

import chess
import pytest
import torch

from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_child_graph_cache import ChildGraphCache, write_cache
from openjev.research.chess_native_feature_cache import build
from openjev.research.chess_pin_cache import pack
from openjev.research.chess_pin_factors import candidate_factors
from openjev.research.chess_training_inputs import TrainingInputs, merge_features, merge_pins


def fixture():
    start = chess.Board(); later = start.copy(); later.push_uci('e2e4')
    boards = [start, chess.Board('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1'), later]
    net = CandidateChess('direct', seed=97, width=32, root_depth=1).eval().requires_grad_(False)
    features = [build(net, boards[:2]), build(net, boards[2:])]
    # Different block boundaries must still yield identical root membership.
    pins = [pack([candidate_factors(b) for b in group], [b.fen(en_passant='fen') for b in group])
            for group in (boards[:1], boards[1:])]
    return boards, net, features, pins


def test_cross_block_shuffle_repeats_and_padding_match_fresh_native_inputs(tmp_path):
    boards, net, features, pins = fixture(); saved = copy.deepcopy((features, pins))
    data = TrainingInputs(features, pins)
    write_cache([{'fen': b.fen(en_passant='fen')} for b in boards], tmp_path/'graphs', provenance={'test': True})
    graphs = ChildGraphCache(tmp_path/'graphs')
    order = torch.tensor([2, 1, 0, 1]); selected = [boards[i] for i in order]
    args, factors = data.batch(order, graphs.batch(order)); fresh = build(net, selected)
    assert torch.equal(args[0], fresh['nodes'])
    for name, index in (('action_features', 1), ('candidates', 2), ('base_logits', 4)):
        assert torch.equal(args[index][args[3]], fresh[name])
    assert torch.isneginf(args[4][~args[3]]).all()
    assert torch.equal(factors, pack([candidate_factors(b) for b in selected], fresh['fens'])['factors'])
    assert data.batch(torch.tensor([0, 2]), graphs.batch([0, 2]))[1].shape == (0, 6)
    for old_blocks, new_blocks in zip(saved, (features, pins)):
        for old, new in zip(old_blocks, new_blocks):
            for key in old:
                assert torch.equal(old[key], new[key]) if torch.is_tensor(old[key]) else old[key] == new[key]


def test_reject_misaligned_blocks_corrupt_features_and_graph_identity(tmp_path):
    boards, _, features, pins = fixture()
    with pytest.raises(ValueError): TrainingInputs(features[::-1], pins)
    for name in ('base_logits', 'nodes', 'action_features'):
        bad = copy.deepcopy(features); bad[0][name].view(-1)[0] = float('nan')
        with pytest.raises(ValueError): merge_features(bad)
    bad = copy.deepcopy(features); bad[0]['offsets'][1] += 1
    with pytest.raises(ValueError): merge_features(bad)
    bad = copy.deepcopy(pins); bad[1]['factors'][0, 0] = int(bad[1]['candidate_offsets'][-1])
    with pytest.raises(ValueError): merge_pins(bad)
    with pytest.raises(ValueError): merge_features([])
    with pytest.raises(ValueError): merge_pins([])
    write_cache([{'fen': b.fen(en_passant='fen')} for b in boards], tmp_path/'graphs', provenance={'test': True})
    data = TrainingInputs(features, pins); graph = ChildGraphCache(tmp_path/'graphs').batch([1, 0])
    with pytest.raises(ValueError): data.batch(torch.tensor([0, 1]), graph)
