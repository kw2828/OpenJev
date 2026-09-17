"""Synthetic checks for the new exposure boundary and isolated generator."""

import copy
import math

import chess
import pytest

from openjev.research import chess_candidate_data as candidate
from openjev.research import chess_connectome_data as data
from openjev.research.chess_spatial_data import state_key, symmetry_key


def tiny_config():
    config = copy.deepcopy(data.CONFIG)
    for index, split in enumerate(config['splits']):
        split.update(examples=3, game_cap=12, max_plies=12,
                     random_move_probability=1.0, seed_base=810000 + index*100)
    return config


def label(board):
    return {'target_uci': min(move.uci() for move in board.legal_moves),
            'score_cp': 100, 'target_value': math.tanh(100/600), 'mate': None,
            'requested_nodes': 2000, 'reported_nodes': 2001, 'wall_seconds': .001}


def test_new_seeds_and_module_isolation():
    before = copy.deepcopy(candidate.CONFIG)
    assert [s['seed_base'] for s in data.CONFIG['splits']] == [115000000, 116000000]
    a, b = data.generator(), data.generator(tiny_config())
    assert a is not b and a is not candidate
    assert a.CONFIG == data.CONFIG
    assert b.CONFIG == tiny_config()
    a.CONFIG['splits'][0]['seed_base'] = -1
    assert data.CONFIG['splits'][0]['seed_base'] == 115000000
    assert candidate.CONFIG == before


def test_every_prior_root_and_legal_successor_is_excluded():
    root = chess.Board()
    clock_variant = chess.Board(root.fen())
    clock_variant.halfmove_clock, clock_variant.fullmove_number = 7, 12
    states = {'prior-placeholder'}
    counts = data.add_exposures(states, [root.fen(), clock_variant.fen()])
    assert counts == {'distinct_roots': 1, 'native_successor_visits': 20}
    expected = {'prior-placeholder', state_key(root)}
    for move in root.legal_moves:
        child = root.copy()
        child.push(move)
        expected.add(state_key(child))
    assert states == expected
    for key in expected - {'prior-placeholder', state_key(root)}:
        admission = data.generator(tiny_config())._Admission([key], tiny_config())
        assert not admission.consider(root, 'dev')['accepted']
        assert not admission.consider(root.mirror(), 'dev')['accepted']


def test_invalid_prior_board_rejected():
    with pytest.raises(ValueError, match='Invalid prior board'):
        data.add_exposures(set(), ['8/8/8/8/8/8/8/8 w - - 0 1'])


def test_fresh_generator_and_validator_preserve_strict_contract(tmp_path):
    config = tiny_config()
    module = data.generator(config)
    excluded = [state_key(chess.Board())]
    path = tmp_path/'panel'
    receipt = module._generate(path, label, config, excluded)
    rows = module.validate(path, excluded)
    assert receipt['counts'] == {'dev': 3, 'shift': 3}
    exposures = {}
    for split, records in rows.items():
        seen = set()
        for row in records:
            board = chess.Board(row['fen'])
            seen.add(symmetry_key(board))
            legal = sorted(move.uci() for move in board.legal_moves)
            assert [t['uci'] for t in row['candidate_transitions']] == legal
            for transition in row['candidate_transitions']:
                child = board.copy()
                child.push_uci(transition['uci'])
                assert transition['next_fen'] == child.fen(en_passant='fen')
                seen.add(symmetry_key(child))
        exposures[split] = seen
    assert not exposures['dev'] & exposures['shift']
    assert symmetry_key(chess.Board()) not in exposures['dev'] | exposures['shift']
    with pytest.raises(FileExistsError):
        module._generate(path, label, config, excluded)
    (path/'unbound.json').write_text('{}')
    with pytest.raises(ValueError):
        module.validate(path, excluded)


def test_old_generator_cannot_validate_new_seed_contract(tmp_path):
    config = tiny_config()
    module = data.generator(config)
    path = tmp_path/'panel'
    module._generate(path, label, config, [])
    wrong = copy.deepcopy(config)
    wrong['splits'][0]['seed_base'] += 1
    with pytest.raises(ValueError):
        data.generator(wrong).validate(path, [])
