"""Frozen-source exclusions and fresh development panels for anchored recurrence.

Only previously seen chess positions enter the exclusion manifest. Teacher
scores, preferred moves, probabilities and game results never enter generated
model inputs. Data generation delegates unchanged to the auditable spatial
generator, including its mirrored-state deduplication and full call accounting.
"""

import hashlib
import json
from pathlib import Path

import chess
import chess.engine

from openjev.research.chess_spatial_data import (
    generate_data,
    state_key,
    stockfish_label,
    symmetry_key,
    validate_data,
)

DATA_SOURCES = (
    ('evidence/chess-student-v1/results/execution/data/train.jsonl', 'train', 4096),
    ('evidence/chess-student-v1/results/execution/data/dev.jsonl', 'dev', 1024),
    ('runs/chess-spatial-v1/execution/data/train.jsonl', 'train', 32768),
    ('runs/chess-spatial-v1/execution/data/dev.jsonl', 'dev', 4096),
    ('runs/chess-spatial-v1/execution/data/shift.jsonl', 'shift', 4096),
)
MATE_SELECTION = 'evidence/chess-mate-v1/selection.json'
MATE_COUNTS = {'train': 1024, 'dev': 256, 'confirm': 512}
PUZZLE_PLAN = 'evidence/chess-v1/plan.json'
PUZZLE_COUNT = 24
ARENA_SOURCES = (
    ('evidence/chess-v1/results', 'game-game-', 18),
    ('evidence/chess-spatial-arena-v1/results', 'game-', 16),
    ('evidence/chess-mate-arena-v1/results', 'game-', 16),
)
FRESH_CONFIG = {
    'teacher_nodes': 2000, 'value_cp_scale': 600., 'mate_cp': 10000,
    'teacher_threads': 1, 'teacher_hash_mb': 16, 'aux_actions': 4,
    'splits': [
        {'name': 'dev', 'examples': 2048, 'game_cap': 200, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 102000000},
        {'name': 'shift', 'examples': 2048, 'game_cap': 200, 'max_plies': 96,
         'random_move_probability': .1, 'seed_base': 103000000},
    ],
}


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field in exclusion source')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite JSON value in exclusion source')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _board(fen):
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise ValueError('Exclusion rows require a complete six-field FEN')
    board = chess.Board(fen)
    if not board.is_valid() or board.chess960:
        raise ValueError('Exclusion source contains an invalid standard chess position')
    return board


def _legal_move(board, token):
    if not isinstance(token, str):
        raise TypeError('Recorded move must be a UCI string')
    move = chess.Move.from_uci(token)
    if move not in board.legal_moves:
        raise ValueError('Exclusion source contains an illegal or null move')
    return move


def _rows(raw, expected):
    if not raw.endswith(b'\n'):
        raise ValueError('Exclusion JSONL source must end with a complete newline-terminated row')
    lines = raw.splitlines()
    if len(lines) != expected or any(not line for line in lines):
        raise ValueError('Exclusion source row count differs from its fixed dataset membership')
    records = [_decode(line) for line in lines]
    if any(not isinstance(row, dict) or not isinstance(row.get('id'), str) for row in records):
        raise ValueError('Malformed exclusion dataset row')
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Duplicated row identity in exclusion source')
    return records


def _successor(board, transition):
    if (not isinstance(transition, dict) or not isinstance(transition.get('uci'), str) or
            'next_fen' not in transition):
        raise ValueError('Stored successor requires its legal UCI action and complete target FEN')
    successor = board.copy(stack=False)
    successor.push(_legal_move(board, transition['uci']))
    target = _board(transition['next_fen'])
    if target.fen(en_passant='fen') != successor.fen(en_passant='fen'):
        raise ValueError('Stored successor FEN disagrees with its exact legal transition')
    return successor


def collect_exclusions(root):
    """Return sorted natural state keys, every exact source hash and row counts.

    All named source files and fixed arena game members are required. No glob
    discovery of arbitrary JSON documents or silent skipping is permitted. The
    existing generator expands natural states to their mirrored equivalents.
    """
    root = Path(root)
    states, files, per_file = set(), {}, {}
    positions = 0
    behavior_positions, auxiliary_positions = 0, 0

    def load(relative):
        path = root/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or nonregular required exclusion source: {relative}')
        raw = path.read_bytes()
        files[relative] = hashlib.sha256(raw).hexdigest()
        return raw

    def add(board):
        nonlocal positions
        states.add(state_key(board))
        positions += 1

    for relative, split, expected in DATA_SOURCES:
        records = _rows(load(relative), expected)
        file_behavior, file_auxiliary = 0, 0
        for row in records:
            if row.get('split') != split or 'fen' not in row:
                raise ValueError('Exclusion row is missing its FEN or has the wrong split membership')
            board = _board(row['fen'])
            if 'state_key' in row and row['state_key'] != state_key(board):
                raise ValueError('Saved exclusion state key differs from its board')
            add(board)
            if 'next_fen' in row or 'behavior_uci' in row:
                if 'next_fen' not in row or 'behavior_uci' not in row:
                    raise ValueError('Behavior successor and behavior UCI must both be present')
                add(_successor(board, {'uci': row['behavior_uci'], 'next_fen': row['next_fen']}))
                file_behavior += 1
            if 'aux_transitions' in row:
                transitions = row['aux_transitions']
                if not isinstance(transitions, list) or not transitions:
                    raise ValueError('Stored auxiliary transitions must be a nonempty list')
                used = set()
                for transition in transitions:
                    successor = _successor(board, transition)
                    if transition['uci'] in used:
                        raise ValueError('Stored auxiliary actions must be distinct')
                    used.add(transition['uci'])
                    add(successor)
                    file_auxiliary += 1
        behavior_positions += file_behavior
        auxiliary_positions += file_auxiliary
        per_file[relative] = {'kind': 'dataset', 'rows': len(records), 'input_positions': len(records),
                              'behavior_successor_positions': file_behavior,
                              'auxiliary_successor_positions': file_auxiliary,
                              'positions': len(records)+file_behavior+file_auxiliary}

    selection = _decode(load(MATE_SELECTION))
    if not isinstance(selection, dict) or set(selection.get('splits', {})) != set(MATE_COUNTS):
        raise ValueError('Mate selection lacks the fixed train/dev/confirm membership')
    mate_ids, mate_count = set(), 0
    for split, count in MATE_COUNTS.items():
        records = selection['splits'][split]
        if not isinstance(records, list) or len(records) != count:
            raise ValueError('Mate selection split row count differs from its frozen size')
        for row in records:
            if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or
                    row['id'] in mate_ids or 'fen' not in row):
                raise ValueError('Malformed or duplicated mate selection row')
            mate_ids.add(row['id'])
            add(_board(row['fen']))
            mate_count += 1
    per_file[MATE_SELECTION] = {'kind': 'mate_selection', 'rows': mate_count,
                                'positions': mate_count, 'splits': dict(MATE_COUNTS)}

    puzzle_plan = _decode(load(PUZZLE_PLAN))
    puzzles = puzzle_plan['puzzles']['positions']
    if (not isinstance(puzzles, list) or len(puzzles) != PUZZLE_COUNT or
            len({row['puzzle_id'] for row in puzzles}) != PUZZLE_COUNT):
        raise ValueError('Original puzzle panel is incomplete or duplicated')
    for row in puzzles:
        board = _board(row['source_fen'])
        board.push(_legal_move(board, row['setup_uci']))
        if board.fen() != row['solver_fen']:
            raise ValueError('Original puzzle solver FEN differs from its legal setup')
        add(board)
    per_file[PUZZLE_PLAN] = {'kind': 'old_puzzle_solver_boards', 'rows': len(puzzles), 'positions': len(puzzles)}

    games, arena_positions = 0, 0
    for directory, prefix, expected in ARENA_SOURCES:
        folder = root/directory
        names = {f'{prefix}{index:02d}.json' for index in range(1, expected+1)}
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f'Missing or nonregular required arena directory: {directory}')
        # Match only root game JSONs; receipts, summaries, reports and PGNs are not positions.
        actual = {path.name for path in folder.glob(prefix+'*.json')}
        if actual != names:
            raise ValueError(f'Arena game membership differs from the fixed panel: {directory}')
        for index in range(1, expected+1):
            relative = f'{directory}/{prefix}{index:02d}.json'
            game = _decode(load(relative))
            if (not isinstance(game, dict) or game.get('game_id') != f'game-{index:02d}' or
                    not isinstance(game.get('moves'), list) or
                    game.get('status') not in ('completed', 'unfinished', 'failed')):
                raise ValueError('Malformed arena game identity, move list or terminal status')
            board = _board(game['initial_fen'])
            add(board)
            for ply, move in enumerate(game['moves'], 1):
                if (not isinstance(move, dict) or move.get('played') is not True or
                        move.get('ply') != ply or move.get('fen_before') != board.fen() or
                        move.get('side') != ('white' if board.turn else 'black')):
                    raise ValueError('Malformed accepted move or broken arena position chain')
                action = _legal_move(board, move['move_uci'])
                if move.get('san') != board.san(action):
                    raise ValueError('Accepted arena move SAN is inconsistent')
                board.push(action)
                if move.get('fen_after') != board.fen():
                    raise ValueError('Accepted arena successor FEN is inconsistent')
                add(board)
            if board.fen() != game['final_fen']:
                raise ValueError('Arena final FEN differs from its accepted move trace')
            if (not isinstance(game.get('attempts'), list) or
                    any(not isinstance(attempt, dict) for attempt in game['attempts'])):
                raise ValueError('Arena lacks its attempt ledger')
            if [attempt for attempt in game['attempts'] if attempt.get('played')] != game['moves']:
                raise ValueError('Arena accepted moves disagree with the attempt ledger')
            count = 1+len(game['moves'])
            per_file[relative] = {'kind': 'arena_game', 'rows': 1, 'positions': count,
                                  'accepted_moves': len(game['moves'])}
            games += 1
            arena_positions += count
    return {'states': sorted(states), 'files': dict(sorted(files.items())),
            'counts': {'source_files': len(files), 'observed_positions': positions,
                       'unique_state_keys': len(states),
                       'unique_mirror_classes': len({symmetry_key(chess.Board(key+' 0 1')) for key in states}),
                       'dataset_rows': sum(expected for _, _, expected in DATA_SOURCES),
                       'dataset_input_positions': sum(expected for _, _, expected in DATA_SOURCES),
                       'dataset_behavior_successor_positions': behavior_positions,
                       'dataset_auxiliary_successor_positions': auxiliary_positions,
                       'mate_selection_rows': mate_count, 'old_puzzle_solver_positions': len(puzzles),
                       'arena_games': games, 'arena_positions': arena_positions,
                       'by_file': dict(sorted(per_file.items()))}}


def generate_fresh(out, engine_path, excluded_states):
    """Run the fixed two-split generation once; caller freezes the engine hash.

    The original generator owns files, costs and failure receipts. No old labels
    are passed to the teacher or added to observations. There are no retries.
    """
    if Path(out).exists():
        raise FileExistsError(out)
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        if not engine.id.get('name', '').startswith('Stockfish 19'):
            raise ValueError('Expected the frozen Stockfish 19 teacher')
        engine.configure({'Threads': FRESH_CONFIG['teacher_threads'], 'Hash': FRESH_CONFIG['teacher_hash_mb']})
        return generate_data(out, lambda board: stockfish_label(engine, board, FRESH_CONFIG),
                             FRESH_CONFIG, excluded_states)


def validate_fresh(out, excluded_states):
    """Validate the original data receipt, exact transitions and exclusion manifest."""
    return validate_data(out, FRESH_CONFIG, excluded_states)
