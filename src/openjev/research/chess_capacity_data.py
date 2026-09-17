"""Bounded capacity-study data, reusing frozen spatial labels and exclusions.

Only state identities enter exclusions. Anchor labels and outcomes never enter
new model inputs. The original spatial generator owns quota, deduplication,
teacher-cost, transition, and failure accounting; this wrapper never retries.
"""

import copy
import hashlib
import math
from pathlib import Path

import chess
import chess.engine

from openjev.research import chess_anchor_data as anchor_source
from openjev.research.chess_spatial_data import (
    generate_data,
    state_key,
    stockfish_label,
    symmetry_key,
    validate_data,
)

TRAIN = 'runs/chess-spatial-v1/execution/data/train.jsonl'
ORIGINAL_COUNT = 32768
ANCHOR_EXECUTION = 'runs/chess-anchor-v1/execution'
ANCHOR_CONFIG = copy.deepcopy(anchor_source.FRESH_CONFIG)
CONFIG = {
    'teacher_nodes': 2000, 'value_cp_scale': 600., 'mate_cp': 10000,
    'teacher_threads': 1, 'teacher_hash_mb': 16, 'aux_actions': 4,
    'splits': [
        {'name': 'train', 'examples': 65536, 'game_cap': 2400, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 105000000},
        {'name': 'dev', 'examples': 4096, 'game_cap': 300, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 106000000},
        {'name': 'shift', 'examples': 4096, 'game_cap': 300, 'max_plies': 96,
         'random_move_probability': .1, 'seed_base': 107000000},
    ],
}


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or nonregular required capacity source: {path}')
    return path.read_bytes()


def _decode(path):
    return anchor_source._decode(_read(path))


def _bound_data(directory, split_names, counts=None):
    """Verify the complete original-generator receipt and every bound member."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Missing or nonregular data directory')
    raw = _read(directory/'completed.json')
    receipt = anchor_source._decode(raw)
    required = {'started.json', 'excluded-states.json', 'games.jsonl', 'analyses.jsonl'}
    required.update(f'{name}.jsonl' for name in split_names)
    if (not isinstance(receipt, dict) or receipt.get('status') != 'completed'
            or (directory/'failed.json').exists() or set(receipt.get('files', {})) != required):
        raise ValueError('Completed data receipt membership/status mismatch')
    if counts is not None and receipt.get('counts') != counts:
        raise ValueError('Data receipt quotas differ from frozen source membership')
    members = {'completed.json': raw}
    for name, digest in receipt['files'].items():
        member = _read(directory/name)
        if hashlib.sha256(member).hexdigest() != digest:
            raise ValueError('Bound data file hash mismatch')
        members[name] = member
    return receipt, members


def collect_exclusions(root):
    """Return inherited exclusions plus all anchor inputs and stored successors.

    The anchor outer completion binds its complete data receipt. That receipt's
    exact membership and hashes are verified before consuming either panel.
    """
    root = Path(root)
    inherited = anchor_source.collect_exclusions(root)
    states, files = set(inherited['states']), dict(inherited['files'])
    counts = copy.deepcopy(inherited['counts'])
    by_file = counts['by_file']
    execution = root/ANCHOR_EXECUTION
    complete_raw = _read(execution/'completed.json')
    complete = anchor_source._decode(complete_raw)
    if not isinstance(complete, dict) or complete.get('status') != 'completed' or (execution/'failed.json').exists():
        raise ValueError('Anchor execution did not complete successfully')
    expected_counts = {split['name']: split['examples'] for split in ANCHOR_CONFIG['splits']}
    _, members = _bound_data(execution/'data', expected_counts, expected_counts)
    if hashlib.sha256(members['completed.json']).hexdigest() != complete.get('data_receipt_sha256'):
        raise ValueError('Anchor execution does not bind this data receipt')
    if anchor_source._decode(members['started.json']).get('config') != ANCHOR_CONFIG:
        raise ValueError('Anchor source generation configuration differs from its frozen contract')

    def bind(relative, raw, metadata):
        digest = hashlib.sha256(raw).hexdigest()
        if relative in files and files[relative] != digest:
            raise ValueError('Source identity changed while collecting exclusions')
        files[relative] = digest
        by_file[relative] = metadata

    bind(f'{ANCHOR_EXECUTION}/completed.json', complete_raw,
         {'kind': 'anchor_execution_receipt', 'rows': 0, 'positions': 0})
    for name, raw in members.items():
        bind(f'{ANCHOR_EXECUTION}/data/{name}', raw,
             {'kind': 'anchor_data_receipt_member', 'rows': 0, 'positions': 0})
    mirror_classes = counts['unique_mirror_classes']
    added_inputs, added_behavior, added_auxiliary = 0, 0, 0

    def add(board):
        nonlocal mirror_classes
        key = state_key(board)
        if key not in states and state_key(board.mirror()) not in states:
            mirror_classes += 1
        states.add(key)

    all_ids = set()
    for split, expected in expected_counts.items():
        relative = f'{ANCHOR_EXECUTION}/data/{split}.jsonl'
        records = anchor_source._rows(members[f'{split}.jsonl'], expected)
        behavior_count, auxiliary_count = 0, 0
        for index, row in enumerate(records):
            if (row.get('split') != split or row['id'] != f'{split}-{index:05d}'
                    or row['id'] in all_ids or 'fen' not in row):
                raise ValueError('Anchor input position membership differs from its fixed panel')
            all_ids.add(row['id'])
            board = anchor_source._board(row['fen'])
            if row.get('state_key') != state_key(board):
                raise ValueError('Anchor input state key differs from its FEN')
            add(board)
            if 'behavior_uci' not in row or 'next_fen' not in row:
                raise ValueError('Anchor input lacks its stored behavior successor')
            add(anchor_source._successor(board, {'uci': row['behavior_uci'], 'next_fen': row['next_fen']}))
            behavior_count += 1
            transitions = row.get('aux_transitions')
            if not isinstance(transitions, list) or len(transitions) != min(ANCHOR_CONFIG['aux_actions'], board.legal_moves.count()):
                raise ValueError('Anchor auxiliary transition membership is incomplete')
            actions = set()
            for transition in transitions:
                successor = anchor_source._successor(board, transition)
                if transition['uci'] in actions:
                    raise ValueError('Duplicate anchor auxiliary action')
                actions.add(transition['uci'])
                add(successor)
                auxiliary_count += 1
        by_file[relative] = {
            'kind': 'anchor_dataset', 'rows': len(records), 'input_positions': len(records),
            'behavior_successor_positions': behavior_count, 'auxiliary_successor_positions': auxiliary_count,
            'positions': len(records)+behavior_count+auxiliary_count,
        }
        added_inputs += len(records)
        added_behavior += behavior_count
        added_auxiliary += auxiliary_count
    # Bind the reused training receipt as well as the already inherited train file.
    old_directory = (root/TRAIN).parent
    old_receipt = _decode(old_directory/'completed.json')
    old_raw = _read(root/TRAIN)
    if (old_receipt.get('status') != 'completed' or (old_directory/'failed.json').exists()
            or old_receipt.get('counts', {}).get('train') != ORIGINAL_COUNT
            or old_receipt.get('files', {}).get('train.jsonl') != hashlib.sha256(old_raw).hexdigest()):
        raise ValueError('Original training source is not bound by its completed receipt')
    if TRAIN not in files or files[TRAIN] != hashlib.sha256(old_raw).hexdigest():
        raise ValueError('Original training source changed after inherited exclusion collection')
    relative = str((old_directory/'completed.json').relative_to(root))
    bind(relative, _read(old_directory/'completed.json'), {'kind': 'original_training_receipt', 'rows': 0, 'positions': 0})
    counts.update({
        'source_files': len(files), 'unique_state_keys': len(states), 'unique_mirror_classes': mirror_classes,
        'observed_positions': counts['observed_positions']+added_inputs+added_behavior+added_auxiliary,
        'dataset_rows': counts['dataset_rows']+added_inputs,
        'dataset_input_positions': counts['dataset_input_positions']+added_inputs,
        'dataset_behavior_successor_positions': counts['dataset_behavior_successor_positions']+added_behavior,
        'dataset_auxiliary_successor_positions': counts['dataset_auxiliary_successor_positions']+added_auxiliary,
        'anchor_input_positions': added_inputs, 'anchor_behavior_successor_positions': added_behavior,
        'anchor_auxiliary_successor_positions': added_auxiliary, 'by_file': dict(sorted(by_file.items())),
    })
    return {'states': sorted(states), 'files': dict(sorted(files.items())), 'counts': counts}


def generate(out, engine, states):
    """Generate one fixed-budget attempt using an explicit Stockfish binary path."""
    if Path(out).exists():
        raise FileExistsError(out)
    with chess.engine.SimpleEngine.popen_uci(str(engine)) as teacher:
        if not teacher.id.get('name', '').startswith('Stockfish 19'):
            raise ValueError('Expected the frozen Stockfish 19 teacher')
        teacher.configure({'Threads': CONFIG['teacher_threads'], 'Hash': CONFIG['teacher_hash_mb']})
        return generate_data(out, lambda board: stockfish_label(teacher, board, CONFIG), CONFIG, states)


def validate(out, states):
    """Validate quotas, exact transitions, exclusion manifest and all teacher calls."""
    return validate_data(out, CONFIG, states)


def training_rows(root, out):
    """Join old and fresh training with explicit source-prefixed row/game IDs.

    Caller validates the fresh data with the frozen exclusion manifest first.
    This helper additionally checks both training receipts, row identities,
    legal labels and cross-source mirrored-state uniqueness. Input files remain
    unchanged; source identity fields retain the original unprefixed values.
    """
    root, out = Path(root), Path(out)
    old_receipt, old_members = _bound_data((root/TRAIN).parent, ('train', 'dev', 'shift'))
    fresh_counts = {split['name']: split['examples'] for split in CONFIG['splits']}
    _, fresh_members = _bound_data(out, fresh_counts, fresh_counts)
    if old_receipt.get('counts', {}).get('train') != ORIGINAL_COUNT:
        raise ValueError('Original training quota mismatch')
    if anchor_source._decode(fresh_members['started.json']).get('config') != CONFIG:
        raise ValueError('Fresh training configuration mismatch')
    original = anchor_source._rows(old_members['train.jsonl'], ORIGINAL_COUNT)
    fresh = anchor_source._rows(fresh_members['train.jsonl'], fresh_counts['train'])
    joined, seen = [], set()
    for origin, records in (('old', original), ('fresh', fresh)):
        for index, row in enumerate(records):
            if row.get('split') != 'train' or row['id'] != f'train-{index:05d}':
                raise ValueError('Training row membership mismatch')
            if type(row.get('game_id')) is not int or row['game_id'] < 0:
                raise ValueError('Generated training game ID must be a nonnegative integer')
            board = anchor_source._board(row['fen'])
            if row.get('state_key') != state_key(board):
                raise ValueError('Training state key differs from its FEN')
            key = symmetry_key(board)
            if key in seen:
                raise ValueError('Duplicated or mirrored state across joined training rows')
            seen.add(key)
            anchor_source._legal_move(board, row['target_uci'])
            value = row.get('target_value')
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError('Invalid bounded training value')
            joined.append({**copy.deepcopy(row), 'id': f'{origin}:{row["id"]}',
                           'game_id': f'{origin}:{row["game_id"]}', 'training_source': origin,
                           'source_id': row['id'], 'source_game_id': row['game_id']})
    return joined
