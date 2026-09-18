# SPDX-License-Identifier: GPL-3.0-only
"""Authenticated training-root data and fixed game splits for continuation work.

No model, teacher, training, tensor-cache write, or dev/shift target is used.
``load_dataset`` returns all original roots in the inventory's explicit order.
``dataset.signature()`` is JSON-compatible and can be frozen before tensorizing.
Raw missing continuation values stay None. Tensor zero placeholders are valid
only with their Boolean mask; a loss must select masked rows before arithmetic.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from . import chess_anchor_eval

VERSION = 'chess-continuation-data-v1'
INVENTORY = 'evidence/chess-transition-value-inventory-v1'
INVENTORY_SHA256 = 'f85d74e390ef6b7e85dce26dff5f395584a8b0f34282d418b1494caf388b76ac'
SOURCE_ORDER = ('runs/chess-spatial-v1/execution/data', 'runs/chess-capacity-v1/execution/data')
SOURCE_RECEIPTS = {
    SOURCE_ORDER[0]: '312d97c0ed6bcf1c9ecd267621c2b48532cd1f521a7237e16986e83d99d50bda',
    SOURCE_ORDER[1]: '26f4975d62c48281592fb650dc74ed6b99ff00c2f946c410a989b992e738c058',
}
SOURCE_ROOT_COUNTS = {SOURCE_ORDER[0]: 32768, SOURCE_ORDER[1]: 65536}
SOURCE_FILES = ('started.json', 'train.jsonl', 'games.jsonl', 'analyses.jsonl')
INVENTORY_FILES = {'started.json', 'summary.json', 'records.jsonl', 'README.md'}
INVENTORY_CODE = {'scripts/chess_transition_value_inventory.py', 'tests/test_chess_transition_value_inventory.py'}
DIAGNOSTIC_LIMIT = 2048
PARTITION_RULE = "SHA256(ASCII('openjev-continuation-v1|<source path>|<game_id>'))[:8] big-endian modulo 10 == 0"
LABEL_FIELDS = ('target_uci', 'target_value', 'score_cp', 'mate')
TEACHER_FIELDS = ('split', 'game_id', 'ply', 'fen', *LABEL_FIELDS)
SKIP_REASONS = {'child_unobserved', 'child_excluded', 'child_not_accepted'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    _require(path.is_file() and not path.is_symlink(), f'Not a regular input file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _json(path):
    return json.loads(path.read_text())


def _jsonl(path):
    with path.open() as stream:
        for line in stream:
            _require(bool(line.strip()), 'Blank JSONL record')
            yield json.loads(line)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def partition(source, game_id):
    """Whole source-games are assigned without reading outcomes or labels."""
    _require(source in SOURCE_ORDER and type(game_id) is int and game_id >= 0,
             'Invalid source-game partition key')
    key = f'openjev-continuation-v1|{source}|{game_id}'.encode('ascii')
    return 'diagnostic' if int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 10 == 0 else 'train'


def _qualified_id(source, original_id):
    return f'{source}::{original_id}'


def authenticate(root):
    """Verify every bound receipt/file/code byte before parsing any root data."""
    root = Path(root)
    receipt_path = root/INVENTORY/'receipt.json'
    _require(_sha(receipt_path) == INVENTORY_SHA256, 'Inventory receipt hash mismatch')
    receipt = _json(receipt_path)
    _require(receipt.get('status') == 'completed' and receipt.get('sources') == SOURCE_RECEIPTS
             and receipt.get('new_teacher_calls') == 0 and receipt.get('training_updates') == 0,
             'Unexpected inventory receipt or sources')
    _require(set(receipt['files']) == INVENTORY_FILES and set(receipt['code']) == INVENTORY_CODE,
             'Unexpected inventory files or code contract')
    bound = {f'{INVENTORY}/receipt.json': INVENTORY_SHA256}
    for name, digest in receipt['files'].items():
        relative = f'{INVENTORY}/{name}'
        _require(_sha(root/relative) == digest, f'Inventory output hash mismatch: {name}')
        bound[relative] = digest
    for relative, digest in receipt['code'].items():
        _require(_sha(root/relative) == digest, f'Inventory code hash mismatch: {relative}')
        bound[relative] = digest
    _require(not (root/INVENTORY/'failed.json').exists(), 'Failed inventory evidence present')
    source_receipts = {}
    for source in SOURCE_ORDER:
        relative = f'{source}/completed.json'
        expected = SOURCE_RECEIPTS[source]
        _require(_sha(root/relative) == expected, 'Original source receipt hash mismatch')
        bound[relative] = expected
        original = _json(root/relative)
        _require(original.get('status') == 'completed' and not (root/source/'failed.json').exists(),
                 'Original source incomplete or failed')
        _require(original['counts']['train'] == SOURCE_ROOT_COUNTS[source], 'Original root count changed')
        for name in SOURCE_FILES:
            relative = f'{source}/{name}'
            _require(_sha(root/relative) == original['files'][name], f'Original source hash mismatch: {relative}')
            bound[relative] = original['files'][name]
        source_receipts[source] = original
    # No dev.jsonl or shift.jsonl contents are opened, hashed, or tensorized.
    helper_sources = {p: _sha(root/p) for p in (
        'src/openjev/research/chess_continuation_data.py', 'tests/test_chess_continuation_data.py',
        'src/openjev/research/chess_anchor_eval.py', 'src/openjev/research/chess_spatial.py')}
    return {'inventory_receipt_sha256': INVENTORY_SHA256, 'bound_files': bound,
            'helper_sources': helper_sources,
            'original_source_receipts': source_receipts,
            'python_chess_version': chess.__version__, 'torch_version': str(torch.__version__)}


def _check_original(row, source, index, config):
    _require(row.get('split') == 'train' and row.get('id') == f'train-{index:05d}',
             'Original training row order or split changed')
    gid, ply = row.get('game_id'), row.get('ply')
    _require(type(gid) is int and 0 <= gid < config['game_cap']
             and type(ply) is int and 0 <= ply < config['max_plies'], 'Invalid source-game-ply identity')
    _require(type(row.get('game_seed')) is int and row['game_seed'] == config['seed_base'] + gid,
             'Original game seed mismatch')
    board = chess.Board(row['fen'])
    _require(board.is_valid() and not board.chess960 and board.outcome(claim_draw=False) is None,
             'Expected a valid nonterminal standard root')
    _require(board.fen(en_passant='fen') == row['fen'], 'Noncanonical original FEN')
    names = tuple(sorted(move.uci() for move in board.legal_moves))
    _require(names and row['target_uci'] in names and row['behavior_uci'] in names,
             'Teacher or behavior action is not legal')
    _require(_finite(row['target_value']) and -1 <= row['target_value'] <= 1
             and type(row['score_cp']) is int
             and (row['mate'] is None or type(row['mate']) is int), 'Invalid original teacher score')
    _require(math.isclose(row['target_value'], math.tanh(row['score_cp']/config['value_cp_scale']),
                          rel_tol=0, abs_tol=1e-12), 'Original teacher score scaling mismatch')
    if row['mate'] is not None:
        mate = row['mate']
        cp = config['mate_cp'] - mate if mate > 0 else -config['mate_cp'] - mate
        _require(row['score_cp'] == cp, 'Original mate score mismatch')
    successor = board.copy(stack=False)
    successor.push_uci(row['behavior_uci'])
    _require(successor.fen(en_passant='fen') == row['next_fen'], 'Original behavior successor differs')
    return names


def _terminal(row, game):
    _require(game['split'] == 'train' and game['game_id'] == row['source_game_id']
             and game['seed'] == row['game_seed'], 'Terminal game provenance mismatch')
    _require(row['ply'] < len(game['actions']), 'Terminal move outside recorded game')
    board = chess.Board()
    for ply, action in enumerate(game['actions'][:row['ply'] + 1]):
        _require(board.outcome(claim_draw=False) is None, 'Post-terminal history action')
        if ply == row['ply']:
            _require(board.fen(en_passant='fen') == row['fen'] and action['uci'] == row['behavior_uci'],
                     'Terminal root history mismatch')
            root_turn = board.turn
        move = chess.Move.from_uci(action['uci'])
        _require(move in board.legal_moves, 'Illegal terminal history move')
        board.push(move)
    outcome = board.outcome(claim_draw=False)
    _require(board.fen(en_passant='fen') == row['next_fen'] and outcome is not None,
             'Terminal continuation not supported by full native history')
    value = 0.0 if outcome.winner is None else 1.0 if outcome.winner == root_turn else -1.0
    _require(row['continuation_value'] == value and row['termination'] == outcome.termination.name.lower(),
             'Native terminal value or termination differs')


@dataclass
class ContinuationDataset:
    rows: tuple[dict, ...]
    membership: dict
    provenance: dict

    def signature(self):
        """Frozen-compatible metadata only; does not allocate tensors or write files.

        Ordered membership hashes are SHA256 of compact, sorted-key ASCII JSON
        lists of source-qualified row IDs. Indices always address ``rows``.
        The first 2048 roots in each partition are fixed diagnostics only;
        every training-partition root remains in the full optimization pool.
        Diagnostic games are exposed historical data, not fresh confirmation.
        """
        return copy.deepcopy({'version': VERSION, 'membership': self.membership,
                              'provenance': self.provenance})


def _state_overlap(rows):
    """Report cross-partition overlap without selecting or removing any row."""
    states = {p: {kind: {'natural': [], 'symmetry_class': []} for kind in ('root', 'continuation')}
              for p in ('train', 'diagnostic')}
    for row in rows:
        for kind, fen in [('root', row['fen']), ('continuation', row['next_fen'])]:
            if kind == 'continuation' and not row['continuation_mask']:
                continue
            board = chess.Board(fen)
            natural = ' '.join(board.fen(en_passant='fen').split()[:4])
            mirrored = ' '.join(board.mirror().fen(en_passant='fen').split()[:4])
            bucket = states[row['partition']][kind]
            bucket['natural'].append(natural)
            bucket['symmetry_class'].append(min(natural, mirrored))
    views = {}
    for view in ('natural', 'symmetry_class'):
        sets = {p: {kind: set(states[p][kind][view]) for kind in ('root', 'continuation')}
                for p in ('train', 'diagnostic')}
        unions = {p: set.union(*sets[p].values()) for p in sets}
        intersection = unions['train'] & unions['diagnostic']
        views[view] = {
            'unique_states': {p: {kind: len(values) for kind, values in sets[p].items()} for p in sets},
            'cross_partition_pairs': {
                f'train_{left}__diagnostic_{right}': len(sets['train'][left] & sets['diagnostic'][right])
                for left in ('root', 'continuation') for right in ('root', 'continuation')},
            'combined_unique_overlap': len(intersection),
            'affected_rows_by_role': {
                p: {kind: sum(value in intersection for value in states[p][kind][view])
                    for kind in ('root', 'continuation')} for p in states},
            'combined_overlap_sha256': _digest(sorted(intersection)),
        }
    disjoint = views['symmetry_class']['combined_unique_overlap'] == 0
    return {'definition': 'First four FEN fields: pieces, side, castling and en-passant square. '
                           'symmetry_class is min(natural key, python-chess board.mirror key). '
                           'Continuation includes only supervised behavior successors. Clocks and '
                           'repetition histories are excluded from this state identity.',
            'views': views, 'state_disjoint_under_this_definition': disjoint,
            'status': ('grouped_and_state_disjoint_under_declared_identity' if disjoint
                       else 'grouped_but_not_state_disjoint'),
            'rows_removed': 0,
            'limits': 'Descriptive overlap audit only; no filtering. Does not cover every legal alternative successor, '
                      'repetition history, or previously exposed data outside these two partitions.'}


def _membership(rows):
    partitions, subsets = {}, {}
    for name in ('train', 'diagnostic'):
        indices = [i for i, row in enumerate(rows) if row['partition'] == name]
        ids = [rows[i]['id'] for i in indices]
        partitions[name] = {'roots': len(indices), 'games': len({rows[i]['game_id'] for i in indices}),
                            'continuation_labels': sum(rows[i]['continuation_mask'] for i in indices),
                            'indices': indices, 'ids_sha256': _digest(ids)}
        diagnostic = indices[:DIAGNOSTIC_LIMIT]
        subsets[name] = {'roots': len(diagnostic), 'indices': diagnostic,
                         'ids_sha256': _digest([rows[i]['id'] for i in diagnostic])}
    _require(set(partitions['train']['indices']).isdisjoint(partitions['diagnostic']['indices']),
             'Partition overlap')
    _require(sum(p['roots'] for p in partitions.values()) == len(rows), 'Partition drops roots')
    return {'source_order': list(SOURCE_ORDER), 'partition_rule': PARTITION_RULE,
            'diagnostic_limit_per_partition': DIAGNOSTIC_LIMIT, 'roots': len(rows),
            'all_ids_sha256': _digest([row['id'] for row in rows]),
            'partitions': partitions, 'fixed_diagnostic_subsets': subsets,
            'by_source': {s: {p: sum(r['source'] == s and r['partition'] == p for r in rows)
                              for p in ('train', 'diagnostic')} for s in SOURCE_ORDER},
            'claim_scope': 'Deterministic partition of exposed historical training games; no independent confirmation.'}


def load_dataset(root):
    """Return every original root; validate all continuation provenance before use."""
    root = Path(root)
    provenance = authenticate(root)
    inventory_rows = _jsonl(root/INVENTORY/'records.jsonl')
    rows, raw_records, keys, ids, games = [], [], {}, set(), {}
    for source in SOURCE_ORDER:
        configuration = _json(root/source/'started.json')['config']
        training = [s for s in configuration['splits'] if s['name'] == 'train']
        _require(len(training) == 1 and configuration['splits'][0]['name'] == 'train',
                 'Unexpected original training split')
        config = {**configuration, **training[0]}
        _require(config['examples'] == SOURCE_ROOT_COUNTS[source], 'Configured original count differs')
        _require(_finite(config['value_cp_scale']) and config['value_cp_scale'] > 0
                 and type(config['mate_cp']) is int and config['mate_cp'] > 0,
                 'Invalid source teacher score configuration')
        for game in _jsonl(root/source/'games.jsonl'):
            if game['split'] != 'train':
                continue
            key = source, game['game_id']
            _require(key not in games, 'Duplicate source-game history')
            games[key] = game
        count = 0
        for index, original in enumerate(_jsonl(root/source/'train.jsonl')):
            names = _check_original(original, source, index, config)
            record = next(inventory_rows, None)
            _require(record is not None and record.get('source') == source
                     and record.get('source_receipt_sha256') == SOURCE_RECEIPTS[source],
                     'Inventory source identity or order differs')
            for key in ('id', 'split', 'game_id', 'game_seed', 'ply', 'fen', 'next_fen', 'behavior_uci'):
                _require(record.get(key) == original[key], f'Inventory root {key} differs from original')
            teacher = record['root_teacher']
            _require(all(teacher.get(k) == original[k] for k in TEACHER_FIELDS),
                     'Inventory root teacher differs from original')
            _require(record.get('behavior_source') in ('random', 'teacher')
                     and type(record.get('behavior_equals_teacher')) is bool
                     and record['behavior_equals_teacher'] == (original['target_uci'] == original['behavior_uci']),
                     'Invalid behavior source or teacher-match metadata')
            key = source, original['game_id'], original['ply']
            qid = _qualified_id(source, original['id'])
            _require(key not in keys and qid not in ids, 'Duplicate source-qualified root')
            status, value, kind = record.get('status'), record.get('continuation_value'), record.get('label_kind')
            _require(status in ('accepted', 'skipped'), 'Unknown continuation status')
            if status == 'skipped':
                _require(value is None and kind is None and record.get('child_id') is None
                         and record.get('child_teacher') is None and record.get('termination') is None
                         and record.get('reason') in SKIP_REASONS, 'Skipped continuation must remain null')
            else:
                _require(_finite(value) and -1 <= value <= 1 and record.get('reason') is None
                         and kind in ('accepted_train_child_teacher', 'native_terminal'),
                         'Invalid accepted continuation target')
            row = {k: original[k] for k in ('fen', 'next_fen', 'game_seed', 'ply', *LABEL_FIELDS, 'behavior_uci')}
            row.update({'id': qid, 'original_id': original['id'], 'source': source, 'split': 'train',
                        'source_game_id': original['game_id'], 'game_id': f'{source}::game-{original["game_id"]}',
                        'partition': partition(source, original['game_id']),
                        'behavior_index': names.index(original['behavior_uci']),
                        'behavior_source': record['behavior_source'],
                        'behavior_equals_teacher': record['behavior_equals_teacher'],
                        'continuation_status': status, 'continuation_value': value,
                        'continuation_mask': status == 'accepted', 'continuation_kind': kind,
                        'continuation_reason': record['reason'], 'termination': record['termination'],
                        'continuation_child_id': (_qualified_id(source, record['child_id'])
                                                  if record['child_id'] is not None else None),
                        'inventory_record_index': len(rows), 'inventory_record_sha256': _digest(record)})
            rows.append(row)
            raw_records.append(record)
            keys[key] = len(rows)-1
            ids.add(qid)
            count += 1
        _require(count == SOURCE_ROOT_COUNTS[source], 'Source does not contain every original root')
    _require(next(inventory_rows, None) is None, 'Inventory contains extra or duplicated roots')
    for row, record in zip(rows, raw_records, strict=True):
        child_index = keys.get((row['source'], row['source_game_id'], row['ply'] + 1))
        kind = row['continuation_kind']
        if kind == 'accepted_train_child_teacher':
            _require(child_index is not None, 'Continuation child is not an accepted original training root')
            child, child_record = rows[child_index], raw_records[child_index]
            _require(row['continuation_child_id'] == child['id'] and row['next_fen'] == child['fen']
                     and row['game_seed'] == child['game_seed'] and row['game_id'] == child['game_id'],
                     'Continuation child source-game-ply-FEN provenance mismatch')
            _require(record['child_teacher'] == child_record['root_teacher'], 'Child teacher-call provenance differs')
            _require(row['continuation_value'] == -child['target_value'], 'Continuation orientation mismatch')
            _require(row['partition'] == child['partition'], 'Parent/child partition leakage')
            _require(row['termination'] is None, 'Nonterminal child has terminal metadata')
        elif kind == 'native_terminal':
            _require(child_index is None and record['child_id'] is None and record['child_teacher'] is None,
                     'Native terminal child has a next training root')
            _require((row['source'], row['source_game_id']) in games, 'Missing native terminal game')
            _terminal(row, games[row['source'], row['source_game_id']])
        else:
            _require(child_index is None, 'Skipped continuation has an accepted next training root')
    _require(len(rows) == sum(SOURCE_ROOT_COUNTS.values()), 'Combined original root coverage differs')
    # Detect source mutation after the initial verification as well.
    for relative, expected in provenance['bound_files'].items():
        _require(_sha(root/relative) == expected, 'Bound input changed during preparation')
    _require(all(_sha(root/p) == expected for p, expected in provenance['helper_sources'].items()),
             'Helper source changed during preparation')
    membership = _membership(rows)
    membership['state_overlap_audit'] = _state_overlap(rows)
    return ContinuationDataset(tuple(rows), membership, provenance)


def tensorize(rows):
    """Encode native inputs and separate supervision; never mutate raw null values.

    ``behavior_indices`` is recomputed from each returned legal menu, rather
    than trusting a stored index or assuming a particular candidate ordering.
    Use ``continuation_mask`` to select rows BEFORE any continuation loss.
    """
    rows = list(rows)
    _require(bool(rows), 'Cannot tensorize an empty continuation batch')
    for row in rows:
        mask, value = row.get('continuation_mask'), row.get('continuation_value')
        _require(type(mask) is bool and ((mask and _finite(value) and -1 <= value <= 1)
                                       or (not mask and value is None)), 'Invalid continuation mask/value pair')
    tensors, menus = chess_anchor_eval.tensorize(rows)
    behavior = []
    for row, menu in zip(rows, menus, strict=True):
        legal = {move.uci() for move in chess.Board(row['fen']).legal_moves}
        _require(len(menu) == len(set(menu)) and set(menu) == legal, 'Native legal menu mismatch')
        _require(row['behavior_uci'] in menu, 'Behavior action not in returned legal menu')
        behavior.append(menu.index(row['behavior_uci']))
    tensors['behavior_indices'] = torch.tensor(behavior, dtype=torch.long)
    tensors['continuation_mask'] = torch.tensor([row['continuation_mask'] for row in rows], dtype=torch.bool)
    tensors['continuation_values'] = torch.tensor(
        [row['continuation_value'] if row['continuation_mask'] else 0.0 for row in rows], dtype=torch.float32)
    return tensors, menus
