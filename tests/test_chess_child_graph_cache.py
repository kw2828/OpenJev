import json

import chess
import pytest
import torch

from openjev.research.chess_child_graph_cache import ChildGraphCache, write_cache
from openjev.research.chess_graph_contrast import candidate_graphs

FENS = [chess.STARTING_FEN, 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
        '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', '4k3/P7/8/8/8/8/8/4K3 w - - 0 1']


def test_round_trip_all_children_frames_special_moves_and_repeated_indices(tmp_path):
    fens = FENS + [chess.Board(f).mirror().fen(en_passant='fen') for f in FENS]
    rows = [{'fen': f, 'target_uci': 'SHOULD_NOT_BE_STORED'} for f in fens]
    out = tmp_path/'cache'; manifest = write_cache(rows, out, provenance={'test': True})
    assert manifest['stored_graph_bytes']*8 == manifest['unpacked_uint8_graph_bytes']
    assert 'SHOULD_NOT_BE_STORED' not in (out/'roots.jsonl').read_text()
    cache = ChildGraphCache(out)
    order = [5, 1, 7, 0, 3, 2, 6, 4, 5]
    actual = cache.batch(torch.tensor(order)); native = candidate_graphs([chess.Board(fens[i]) for i in order])
    assert torch.equal(actual['root'], native['root'])
    assert torch.equal(actual['children'], native['children'][native['mask']])
    assert torch.equal(actual['mask'], native['mask']) and torch.equal(actual['permutation'], native['permutation'])
    assert actual['menus'] == list(map(tuple, native['menus']))
    # Returned tensors do not alias the read-only packed data.
    original = actual['children'].clone(); actual['children'].zero_()
    assert torch.equal(cache.batch(order)['children'], original)


def test_corrupt_member_and_invalid_indices_are_rejected(tmp_path):
    out = tmp_path/'cache'; write_cache([{'fen': FENS[0]}], out, provenance={})
    cache = ChildGraphCache(out)
    for bad in ([], [-1], [1], [True], [.5]):
        with pytest.raises(ValueError, match='indices'): cache.batch(bad)
    with (out/'children.bin').open('r+b') as stream:
        stream.write(bytes([255]))
    with pytest.raises(ValueError, match='hash'): ChildGraphCache(out)


def test_incomplete_cache_is_not_accepted_and_existing_cache_is_not_overwritten(tmp_path):
    out = tmp_path/'cache'
    with pytest.raises(ValueError, match='empty'): write_cache([], out, provenance={})
    assert json.loads((out/'failed.json').read_text())['status'] == 'failed'
    with pytest.raises(FileNotFoundError): ChildGraphCache(out)
    with pytest.raises(FileExistsError): write_cache([{'fen': FENS[0]}], out, provenance={})
