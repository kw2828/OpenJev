# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic identity, perspective and full-history tests; never calls a teacher."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import chess
import pytest

SPEC = importlib.util.spec_from_file_location(
    'transition_inventory', Path(__file__).resolve().parents[1]/'scripts/chess_transition_value_inventory.py')
inv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inv)


def dump(path, value):
    path.write_text(json.dumps(value))


def lines(path, values):
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


def fixture(tmp_path, moves=('e2e4', 'e7e5', 'g1f3', 'b8c6'), rejected=None, stop='ply_cap'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rejected = rejected or {}
    config = {'teacher_nodes': 2, 'value_cp_scale': 600., 'mate_cp': 10000,
              'teacher_threads': 1, 'teacher_hash_mb': 16,
              'splits': [{'name': 'train', 'examples': len(moves)-len(rejected),
                          'game_cap': 4, 'max_plies': len(moves), 'seed_base': 100}]}
    board = chess.Board()
    rows, calls, actions = [], [], []
    for ply, uci in enumerate(moves):
        cp = 60 + 60 * ply
        call = {'split': 'train', 'game_id': 0, 'ply': ply, 'fen': board.fen(en_passant='fen'),
                'state_key': inv.state_key(board), 'labelled_position': ply not in rejected,
                'excluded_position': rejected.get(ply) == 'excluded', 'status': 'completed',
                'target_uci': uci, 'target_value': math.tanh(cp/600), 'score_cp': cp, 'mate': None,
                'requested_nodes': 2, 'reported_nodes': 2, 'wall_seconds': .01, 'call_wall_seconds': .02}
        successor = board.copy(stack=True)
        successor.push_uci(uci)
        if ply not in rejected:
            rows.append({'id': f'train-{len(rows):05d}', 'split': 'train', 'game_id': 0,
                         'game_seed': 100, 'ply': ply, 'fen': call['fen'], 'state_key': call['state_key'],
                         **{k: call[k] for k in inv.LABEL_FIELDS}, 'behavior_uci': uci,
                         'next_fen': successor.fen(en_passant='fen')})
        calls.append(call)
        actions.append({'uci': uci, 'source': 'random' if ply == 0 else 'teacher'})
        board.push_uci(uci)
    games = [{'split': 'train', 'game_id': 0, 'seed': 100, 'sampled_positions': len(rows),
              'actions': actions, 'final_fen': board.fen(en_passant='fen'), 'stop': stop}]
    dump(tmp_path/'started.json', {'config': config})
    lines(tmp_path/'train.jsonl', rows)
    lines(tmp_path/'games.jsonl', games)
    lines(tmp_path/'analyses.jsonl', calls)
    receipt = {'status': 'completed', 'counts': {'train': len(rows)},
               **{k: 0 for k in inv.SOURCE_COST_FIELDS}, 'teacher_cost_scope': 'synthetic tests'}
    dump(tmp_path/'completed.json', receipt)
    return reseal(tmp_path)


def reseal(directory):
    receipt = inv.read(directory/'completed.json')
    receipt['files'] = {name: inv.sha(directory/name) for name in inv.FILES}
    dump(directory/'completed.json', receipt)
    return inv.sha(directory/'completed.json')


def edit_rows(directory, filename, mutate):
    rows = list(inv.jsonl(directory/filename))
    mutate(rows)
    lines(directory/filename, rows)
    return reseal(directory)


def test_child_orientation_and_random_teacher_overlap(tmp_path):
    digest = fixture(tmp_path)
    records, stats = inv.inventory_source(tmp_path, digest, 'fixture-a')
    assert stats['accepted'] == 3 and stats['skipped'] == 1
    assert records[0]['root_teacher']['target_value'] > 0
    assert records[0]['child_teacher']['target_value'] > 0
    assert records[0]['continuation_value'] == -math.tanh(120/600)
    assert records[0]['root_player'] == 'white' and records[1]['root_player'] == 'black'
    assert records[0]['behavior_source'] == 'random' and records[0]['behavior_equals_teacher']
    assert records[-1]['reason'] == 'child_unobserved'
    assert records[-1]['continuation_value'] is None
    assert stats['referenced_unique_call_costs']['calls'] == 4
    assert stats['child_role_costs']['calls'] == 3
    d = stats['nonterminal_consistency']['behavior_equals_teacher']
    assert d['absolute_difference']['count'] == 3
    assert d['continuation_exceeds_root_count'] == 0


@pytest.mark.parametrize('rejected,reason', [('excluded', 'child_excluded'), ('duplicate', 'child_not_accepted')])
def test_rejected_child_is_not_a_target(tmp_path, rejected, reason):
    digest = fixture(tmp_path, rejected={1: rejected})
    records, stats = inv.inventory_source(tmp_path, digest, 'fixture-a')
    assert records[0]['status'] == 'skipped' and records[0]['reason'] == reason
    assert records[0]['continuation_value'] is None and records[0]['child_teacher'] is None
    assert stats['train_roots'] == 3


def test_sources_are_separate_even_with_same_game_and_fen(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    ha = fixture(a, rejected={1: 'excluded'})
    hb = fixture(b)
    ar, _ = inv.inventory_source(a, ha, 'source-a')
    br, _ = inv.inventory_source(b, hb, 'source-b')
    assert ar[0]['next_fen'] == br[0]['next_fen']
    assert ar[0]['reason'] == 'child_excluded' and br[0]['status'] == 'accepted'
    assert {r['source'] for r in ar} == {'source-a'}


@pytest.mark.parametrize('stop', ['terminal', 'ply_cap', 'quota'])
def test_native_checkmate_is_exact_even_at_cap_or_quota(tmp_path, stop):
    digest = fixture(tmp_path, moves=('f2f3', 'e7e5', 'g2g4', 'd8h4'), stop=stop)
    records, stats = inv.inventory_source(tmp_path, digest, 'fixture')
    assert records[-1]['label_kind'] == 'native_terminal'
    assert records[-1]['continuation_value'] == 1.0  # Black, the root mover, wins.
    assert records[-1]['root_player'] == 'black'
    assert records[-1]['termination'] == 'checkmate'
    assert records[-1]['child_teacher'] is None
    assert stats['terminal_outcomes'] == {'1.0': 1}


def test_fivefold_requires_full_history_and_is_exact_draw(tmp_path):
    moves = ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 4
    digest = fixture(tmp_path, moves=moves, stop='terminal')
    records, _ = inv.inventory_source(tmp_path, digest, 'fixture')
    assert records[-1]['label_kind'] == 'native_terminal'
    assert records[-1]['termination'] == 'fivefold_repetition'
    assert records[-1]['continuation_value'] == 0.0
    assert chess.Board(records[-1]['next_fen']).outcome(claim_draw=False) is None


def test_ply_cutoff_is_never_zero_or_terminal(tmp_path):
    digest = fixture(tmp_path, moves=('e2e4',))
    records, _ = inv.inventory_source(tmp_path, digest, 'fixture')
    assert records[0]['continuation_value'] is None and records[0]['reason'] == 'child_unobserved'


@pytest.mark.parametrize('filename,field,value', [
    ('train.jsonl', 'game_id', 1), ('train.jsonl', 'ply', 2),
    ('train.jsonl', 'game_seed', 101), ('train.jsonl', 'fen', chess.STARTING_FEN),
    ('train.jsonl', 'next_fen', chess.STARTING_FEN), ('train.jsonl', 'behavior_uci', 'a2a4'),
    ('train.jsonl', 'target_value', 0.9), ('train.jsonl', 'split', 'dev'),
    ('analyses.jsonl', 'fen', chess.STARTING_FEN), ('analyses.jsonl', 'target_value', 0.9),
    ('analyses.jsonl', 'labelled_position', False), ('analyses.jsonl', 'excluded_position', True),
    ('analyses.jsonl', 'mate', 3), ('analyses.jsonl', 'requested_nodes', 99),
    ('analyses.jsonl', 'call_wall_seconds', -1), ('analyses.jsonl', 'game_id', 1),
    ('analyses.jsonl', 'split', 'dev'),
])
def test_invalid_identity_or_label_rejected_even_after_resealing(tmp_path, filename, field, value):
    fixture(tmp_path)
    digest = edit_rows(tmp_path, filename, lambda rows: rows[1].__setitem__(field, value))
    with pytest.raises(ValueError):
        inv.inventory_source(tmp_path, digest, 'fixture')


@pytest.mark.parametrize('filename', ['train.jsonl', 'games.jsonl', 'analyses.jsonl'])
def test_duplicate_records_rejected(tmp_path, filename):
    fixture(tmp_path)
    digest = edit_rows(tmp_path, filename, lambda rows: rows.append(copy.deepcopy(rows[0])))
    with pytest.raises(ValueError):
        inv.inventory_source(tmp_path, digest, 'fixture')


@pytest.mark.parametrize('filename', [*inv.FILES, 'completed.json'])
def test_corrupted_file_does_not_match_receipt(tmp_path, filename):
    digest = fixture(tmp_path)
    with (tmp_path/filename).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        inv.inventory_source(tmp_path, digest, 'fixture')


def test_false_terminal_and_wrong_game_seed_rejected(tmp_path):
    fixture(tmp_path)
    digest = edit_rows(tmp_path, 'games.jsonl', lambda rows: rows[0].__setitem__('stop', 'terminal'))
    with pytest.raises(ValueError, match='Terminal marker'):
        inv.inventory_source(tmp_path, digest, 'fixture')
    fixture(tmp_path)
    digest = edit_rows(tmp_path, 'games.jsonl', lambda rows: rows[0].__setitem__('seed', 101))
    with pytest.raises(ValueError, match='game seed'):
        inv.inventory_source(tmp_path, digest, 'fixture')


def test_heldout_poison_values_never_enter_targets(tmp_path):
    fixture(tmp_path)
    for file in ('games.jsonl', 'analyses.jsonl'):
        rows = list(inv.jsonl(tmp_path/file))
        rows.append({'split': 'dev', 'game_id': 0, 'ply': 0, 'fen': 'poison', 'target_value': 999})
        lines(tmp_path/file, rows)
    digest = reseal(tmp_path)
    records, stats = inv.inventory_source(tmp_path, digest, 'fixture')
    assert len(records) == 4 and stats['accepted'] == 3
    assert all(r['split'] == 'train' and r['root_teacher']['split'] == 'train' for r in records)


def test_continuation_rejects_wrong_child_game_ply_and_fen(tmp_path):
    fixture(tmp_path)
    rows = list(inv.jsonl(tmp_path/'train.jsonl'))
    calls = list(inv.jsonl(tmp_path/'analyses.jsonl'))
    board = chess.Board()
    board.push_uci(rows[0]['behavior_uci'])
    for field, value in [('game_id', 1), ('ply', 2), ('fen', chess.STARTING_FEN), ('game_seed', 101)]:
        child = {**rows[1], field: value}
        with pytest.raises(ValueError, match='identity mismatch'):
            inv.continuation(rows[0], child, calls[1], board, chess.WHITE)


def test_distribution_and_other_moves_keep_all_pairs():
    records = [{'label_kind': 'accepted_train_child_teacher', 'behavior_equals_teacher': equal,
                'continuation_value': value, 'root_teacher': {'target_value': 0.}}
               for equal, value in [(True, -.5), (True, .5), (False, .9), (False, -.9)]]
    stats = inv.consistency(records)
    assert stats['behavior_equals_teacher']['absolute_difference']['mean'] == .5
    assert stats['behavior_other']['absolute_difference']['count'] == 2
    assert stats['behavior_other']['continuation_exceeds_root_count'] == 1
    assert inv.distribution([])['mean'] is None


def test_output_receipt_hashes_and_exclusive_destination(tmp_path, monkeypatch):
    digest = fixture(tmp_path/'data')
    monkeypatch.setattr(inv, 'ROOT', tmp_path)
    monkeypatch.setattr(inv, 'SOURCES', {'data': digest})
    for relative in ('scripts/chess_transition_value_inventory.py', 'tests/test_chess_transition_value_inventory.py'):
        path = tmp_path/relative
        path.parent.mkdir(exist_ok=True)
        path.write_text('fixture')
    out = tmp_path/'inventory'
    inv.run(out)
    receipt = inv.read(out/'receipt.json')
    assert receipt['status'] == 'completed' and receipt['new_teacher_calls'] == 0
    assert set(receipt['files']) == {'started.json', 'summary.json', 'records.jsonl', 'README.md'}
    assert all(inv.sha(out/name) == digest for name, digest in receipt['files'].items())
    with pytest.raises(FileExistsError):
        inv.run(out)
