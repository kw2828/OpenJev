# SPDX-License-Identifier: GPL-3.0-only
"""Small authenticated synthetic roots; no real cache construction or inference."""

import copy
import hashlib
import json
import math

import chess
import pytest
import torch

from openjev.research import chess_continuation_data as data


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def jsonl(path, values):
    path.write_text(''.join(json.dumps(v, allow_nan=False) + '\n' for v in values))


def seal_inventory(root, monkeypatch):
    path = root/data.INVENTORY
    receipt = data._json(path/'receipt.json')
    receipt['files'] = {name: data._sha(path/name) for name in data.INVENTORY_FILES}
    receipt['sources'] = data.SOURCE_RECEIPTS
    write(path/'receipt.json', receipt)
    monkeypatch.setattr(data, 'INVENTORY_SHA256', data._sha(path/'receipt.json'))


def make_data(root, monkeypatch, *, terminal=False):
    all_records, source_counts, source_receipts = [], {}, {}
    for source in data.SOURCE_ORDER:
        directory = root/source
        directory.mkdir(parents=True)
        original, records, games, calls = [], [], [], []
        for gid in range(12):
            moves = ('f2f3', 'e7e5', 'g2g4', 'd8h4') if terminal else ('e2e4', 'e7e5', 'g1f3', 'b8c6')
            board, actions = chess.Board(), []
            start = len(original)
            for ply, uci in enumerate(moves):
                fen = board.fen(en_passant='fen')
                root_player = 'white' if board.turn else 'black'
                cp = 30 + 20 * ply
                teacher = {'split': 'train', 'game_id': gid, 'ply': ply, 'fen': fen,
                           'target_uci': uci, 'target_value': math.tanh(cp/600.), 'score_cp': cp, 'mate': None,
                           'requested_nodes': 2, 'reported_nodes': 2, 'wall_seconds': .01, 'call_wall_seconds': .02}
                board.push_uci(uci)
                row = {'id': f'train-{len(original):05d}', 'game_seed': 100 + gid,
                       **{k: teacher[k] for k in data.TEACHER_FIELDS}, 'behavior_uci': uci,
                       'next_fen': board.fen(en_passant='fen')}
                original.append(row)
                record = {'source': source, **{k: row[k] for k in (
                    'id', 'split', 'game_id', 'game_seed', 'ply', 'fen', 'next_fen', 'behavior_uci')},
                    'root_teacher': teacher, 'behavior_source': 'teacher', 'behavior_equals_teacher': True,
                    'root_player': root_player}
                records.append(record)
                actions.append({'uci': uci, 'source': 'teacher'})
                calls.append(teacher)
            for offset in range(4):
                record = records[start + offset]
                if offset < 3:
                    child = records[start + offset + 1]
                    record.update({'status': 'accepted', 'label_kind': 'accepted_train_child_teacher',
                                   'continuation_value': -child['root_teacher']['target_value'],
                                   'child_id': child['id'], 'child_teacher': child['root_teacher'],
                                   'termination': None, 'reason': None})
                elif terminal:
                    record.update({'status': 'accepted', 'label_kind': 'native_terminal',
                                   'continuation_value': 1., 'child_id': None, 'child_teacher': None,
                                   'termination': 'checkmate', 'reason': None})
                else:
                    record.update({'status': 'skipped', 'label_kind': None, 'continuation_value': None,
                                   'child_id': None, 'child_teacher': None, 'termination': None,
                                   'reason': 'child_unobserved'})
            games.append({'split': 'train', 'game_id': gid, 'seed': 100 + gid,
                          'actions': actions, 'final_fen': board.fen(en_passant='fen')})
        config = {'value_cp_scale': 600., 'mate_cp': 10000,
                  'splits': [{'name': 'train', 'examples': len(original), 'game_cap': 20,
                              'max_plies': 4, 'seed_base': 100}]}
        write(directory/'started.json', {'config': config})
        jsonl(directory/'train.jsonl', original)
        jsonl(directory/'games.jsonl', games)
        jsonl(directory/'analyses.jsonl', calls)
        write(directory/'completed.json', {'status': 'completed', 'counts': {'train': len(original)},
              'files': {name: data._sha(directory/name) for name in data.SOURCE_FILES}})
        source_receipts[source] = data._sha(directory/'completed.json')
        source_counts[source] = len(original)
        for record in records:
            record['source_receipt_sha256'] = source_receipts[source]
        all_records.extend(records)
    monkeypatch.setattr(data, 'SOURCE_ROOT_COUNTS', source_counts)
    monkeypatch.setattr(data, 'SOURCE_RECEIPTS', source_receipts)
    for code in [*data.INVENTORY_CODE, 'src/openjev/research/chess_continuation_data.py',
                 'tests/test_chess_continuation_data.py', 'src/openjev/research/chess_anchor_eval.py',
                 'src/openjev/research/chess_spatial.py']:
        file = root/code
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('synthetic bound code fixture\n')
    inventory = root/data.INVENTORY
    inventory.mkdir(parents=True)
    jsonl(inventory/'records.jsonl', all_records)
    for name in ('summary.json', 'started.json', 'README.md'):
        (inventory/name).write_text('{}')
    write(inventory/'receipt.json', {'status': 'completed', 'sources': source_receipts,
          'files': {}, 'code': {name: data._sha(root/name) for name in data.INVENTORY_CODE},
          'new_teacher_calls': 0, 'training_updates': 0})
    seal_inventory(root, monkeypatch)
    return all_records


def modify_records(root, monkeypatch, fn):
    path = root/data.INVENTORY/'records.jsonl'
    records = list(data._jsonl(path))
    fn(records)
    jsonl(path, records)
    seal_inventory(root, monkeypatch)


def test_exact_partition_formula():
    for source in data.SOURCE_ORDER:
        for gid in range(100):
            digest = hashlib.sha256(f'openjev-continuation-v1|{source}|{gid}'.encode('ascii')).digest()
            expected = 'diagnostic' if int.from_bytes(digest[:8], 'big') % 10 == 0 else 'train'
            assert data.partition(source, gid) == expected
    with pytest.raises(ValueError):
        data.partition(data.SOURCE_ORDER[0], True)
    with pytest.raises(ValueError):
        data.partition('another-source', 0)


def test_all_roots_order_and_source_qualified_identities(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch)
    monkeypatch.setattr(data, 'DIAGNOSTIC_LIMIT', 2)
    dataset = data.load_dataset(tmp_path)
    assert len(dataset.rows) == 96
    assert len({r['id'] for r in dataset.rows}) == 96
    assert len({r['game_id'] for r in dataset.rows}) == 24
    assert [r['source'] for r in dataset.rows] == [data.SOURCE_ORDER[0]] * 48 + [data.SOURCE_ORDER[1]] * 48
    for name in ('train', 'diagnostic'):
        entries = dataset.membership['partitions'][name]
        expected = [i for i, r in enumerate(dataset.rows) if data.partition(r['source'], r['source_game_id']) == name]
        assert entries['indices'] == expected
        assert entries['ids_sha256'] == data._digest([dataset.rows[i]['id'] for i in expected])
        assert dataset.membership['fixed_diagnostic_subsets'][name]['indices'] == expected[:2]
    assert sum(p['roots'] for p in dataset.membership['partitions'].values()) == 96
    by_id = {row['id']: row for row in dataset.rows}
    for row in dataset.rows:
        if row['continuation_child_id'] is not None:
            assert row['partition'] == by_id[row['continuation_child_id']]['partition']
    signature = dataset.signature()
    signature['membership']['roots'] = 0
    assert dataset.membership['roots'] == 96
    assert 'helper_sources' in signature['provenance']


def test_partition_does_not_select_by_outcomes(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch)
    dataset = data.load_dataset(tmp_path)
    changed = copy.deepcopy(dataset.rows)
    for row in changed:
        row['target_value'] *= -1
        if row['continuation_value'] is not None:
            row['continuation_value'] *= -1
    expected = {k: v for k, v in dataset.membership.items() if k != 'state_overlap_audit'}
    assert data._membership(changed) == expected


def test_tensorize_native_inputs_and_masked_missing_values(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch)
    rows = data.load_dataset(tmp_path).rows[:4]
    original = copy.deepcopy(rows)
    tensors, menus = data.tensorize(rows)
    assert tensors['observations'].shape == (4, 19, 8, 8)
    assert tensors['mask'].dtype == torch.bool
    assert tensors['continuation_mask'].tolist() == [True, True, True, False]
    assert tensors['continuation_values'][-1].item() == 0.
    for i, row in enumerate(rows):
        assert menus[i][tensors['behavior_indices'][i]] == row['behavior_uci']
        assert menus[i][tensors['targets'][i]] == row['target_uci']
    assert rows == original and rows[-1]['continuation_value'] is None
    prediction = torch.tensor([0., 0., 0., float('nan')], requires_grad=True)
    mask = tensors['continuation_mask']
    loss = (prediction[mask] - tensors['continuation_values'][mask]).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and prediction.grad[-1].item() == 0.


def test_behavior_index_follows_menu_reordering(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch)
    row = data.load_dataset(tmp_path).rows[0]
    base = data.chess_anchor_eval.encode_candidates
    def reverse(board):
        names, features = base(board)
        return tuple(reversed(names)), features[::-1].copy()
    monkeypatch.setattr(data.chess_anchor_eval, 'encode_candidates', reverse)
    tensors, menus = data.tensorize([row])
    index = tensors['behavior_indices'][0].item()
    assert menus[0][index] == row['behavior_uci']
    assert index == len(menus[0]) - 1 - row['behavior_index']


def test_native_terminal_values_replayed_with_history(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch, terminal=True)
    dataset = data.load_dataset(tmp_path)
    terminals = [row for row in dataset.rows if row['continuation_kind'] == 'native_terminal']
    assert len(terminals) == 24 and all(r['continuation_value'] == 1. for r in terminals)
    modify_records(tmp_path, monkeypatch, lambda records: records[3].__setitem__('continuation_value', -1.))
    with pytest.raises(ValueError, match='Native terminal value'):
        data.load_dataset(tmp_path)


@pytest.mark.parametrize('field,value', [
    ('source', 'another-source'), ('game_id', 999), ('game_seed', 999), ('ply', 2),
    ('fen', chess.STARTING_FEN), ('next_fen', chess.STARTING_FEN), ('id', 'train-00000'),
    ('behavior_uci', 'a1a8'), ('split', 'dev'), ('continuation_value', .99),
    ('child_id', 'train-00008'), ('termination', 'checkmate'),
])
def test_inventory_identity_or_orientation_corruption_rejected(tmp_path, monkeypatch, field, value):
    make_data(tmp_path, monkeypatch)
    modify_records(tmp_path, monkeypatch, lambda records: records[1].__setitem__(field, value))
    with pytest.raises(ValueError):
        data.load_dataset(tmp_path)


def test_child_teacher_provenance_cannot_be_substituted(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch)
    modify_records(tmp_path, monkeypatch,
                   lambda records: records[0]['child_teacher'].__setitem__('call_wall_seconds', 12.))
    with pytest.raises(ValueError, match='Child teacher-call provenance'):
        data.load_dataset(tmp_path)


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'cross-source'])
def test_no_dropped_duplicate_or_cross_source_roots(tmp_path, monkeypatch, mode):
    make_data(tmp_path, monkeypatch)
    def mutate(records):
        if mode == 'missing':
            records.pop()
        elif mode == 'duplicate':
            records.append(copy.deepcopy(records[0]))
        else:
            records[0]['child_teacher'] = records[49]['root_teacher']
            records[0]['child_id'] = records[49]['id']
            records[0]['source'] = data.SOURCE_ORDER[1]
    modify_records(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError):
        data.load_dataset(tmp_path)


@pytest.mark.parametrize('field,value', [('continuation_value', 0.), ('child_id', 'train-00000'),
                                      ('label_kind', 'native_terminal'), ('reason', 'unknown')])
def test_skipped_raw_values_must_remain_null(tmp_path, monkeypatch, field, value):
    make_data(tmp_path, monkeypatch)
    modify_records(tmp_path, monkeypatch, lambda records: records[3].__setitem__(field, value))
    with pytest.raises(ValueError, match='Skipped continuation'):
        data.load_dataset(tmp_path)


@pytest.mark.parametrize('relative', [f'{data.INVENTORY}/receipt.json', f'{data.INVENTORY}/records.jsonl',
                                    f'{data.INVENTORY}/README.md', next(iter(data.INVENTORY_CODE)),
                                    f'{data.SOURCE_ORDER[0]}/train.jsonl',
                                    f'{data.SOURCE_ORDER[1]}/analyses.jsonl',
                                    f'{data.SOURCE_ORDER[0]}/completed.json'])
def test_all_bound_corruption_rejected_before_root_parsing(tmp_path, monkeypatch, relative):
    make_data(tmp_path, monkeypatch)
    with (tmp_path/relative).open('a') as stream:
        stream.write(' ')
    def forbidden(*args):
        raise AssertionError('Parsed roots before authentication completed')
    monkeypatch.setattr(data, '_jsonl', forbidden)
    with pytest.raises(ValueError, match='hash mismatch'):
        data.load_dataset(tmp_path)


def test_no_dev_or_shift_targets_are_opened(tmp_path, monkeypatch):
    make_data(tmp_path, monkeypatch)
    for source in data.SOURCE_ORDER:
        for name in ('dev.jsonl', 'shift.jsonl'):
            (tmp_path/source/name).write_text('unparseable forbidden target file')
    original_jsonl = data._jsonl
    def restricted(path):
        assert path.name not in ('dev.jsonl', 'shift.jsonl', 'analyses.jsonl')
        return original_jsonl(path)
    monkeypatch.setattr(data, '_jsonl', restricted)
    assert len(data.load_dataset(tmp_path).rows) == 96


@pytest.mark.parametrize('value,mask', [(0., False), (None, True), (float('nan'), True),
                                     (float('inf'), True), (2., True), (.1, 1)])
def test_invalid_tensor_mask_values_rejected(value, mask):
    with pytest.raises(ValueError, match='mask/value'):
        data.tensorize([{'continuation_value': value, 'continuation_mask': mask}])


def test_terminal_helper_preserves_repetition_history():
    moves = ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 4
    board = chess.Board()
    for uci in moves[:-1]:
        board.push_uci(uci)
    fen = board.fen(en_passant='fen')
    board.push_uci(moves[-1])
    next_fen = board.fen(en_passant='fen')
    assert chess.Board(next_fen).outcome(claim_draw=False) is None
    row = {'source_game_id': 0, 'game_seed': 100, 'ply': 15, 'fen': fen, 'next_fen': next_fen,
           'behavior_uci': moves[-1], 'continuation_value': 0., 'termination': 'fivefold_repetition'}
    game = {'split': 'train', 'game_id': 0, 'seed': 100, 'actions': [{'uci': u} for u in moves]}
    data._terminal(row, game)
    with pytest.raises(ValueError, match='root history'):
        data._terminal({**row, 'ply': 3}, game)


def test_natural_and_mirrored_state_overlap_without_filtering():
    board = chess.Board()
    board.push_uci('e2e4')
    fen = board.fen(en_passant='fen')
    mirrored = board.mirror().fen(en_passant='fen')
    rows = [{'partition': 'train', 'fen': fen, 'next_fen': chess.STARTING_FEN, 'continuation_mask': True},
            {'partition': 'diagnostic', 'fen': mirrored, 'next_fen': chess.STARTING_FEN, 'continuation_mask': False}]
    result = data._state_overlap(rows)
    assert result['views']['natural']['combined_unique_overlap'] == 0
    assert result['views']['symmetry_class']['combined_unique_overlap'] == 1
    assert result['status'] == 'grouped_but_not_state_disjoint' and result['rows_removed'] == 0
    rows[1]['fen'] = fen
    result = data._state_overlap(rows)
    assert result['views']['natural']['cross_partition_pairs']['train_root__diagnostic_root'] == 1
    rows[1]['fen'] = chess.STARTING_FEN
    result = data._state_overlap(rows)
    assert result['views']['natural']['cross_partition_pairs']['train_continuation__diagnostic_root'] == 1
    assert result['views']['natural']['affected_rows_by_role']['train']['continuation'] == 1
