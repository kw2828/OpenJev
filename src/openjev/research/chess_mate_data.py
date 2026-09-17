"""Deterministic, source-game-separated mate-in-one curriculum selection.

This module only reads the supplied public CSV. It does not write files, call a
model/engine, or use evaluation outcomes. Dataset labels stay separate from model
inputs: a legal solver board is obtained after the opponent's first listed move.
"""

import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import chess

from openjev.research.chess_spatial_data import symmetry_key

SPLITS = ('train', 'dev', 'confirm')


def source_game_id(value):
    """Normalize Lichess URLs with perspective/ply suffixes to the eight-char ID."""
    if re.fullmatch('[A-Za-z0-9]{8}', value):
        return value
    parsed = urlparse(value)
    if parsed.scheme not in ('http', 'https') or parsed.netloc.lower() not in ('lichess.org', 'www.lichess.org'):
        raise ValueError('Expected a Lichess source-game URL or eight-character game ID')
    identifier = parsed.path.strip('/').split('/')[0]
    if not re.fullmatch('[A-Za-z0-9]{8}', identifier):
        raise ValueError('Source-game URL does not contain an eight-character game ID')
    return identifier


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _excluded_keys(states):
    result = set()
    for value in states:
        if not isinstance(value, str) or len(value.split()) != 4:
            raise ValueError('Excluded states must be four-field FEN state keys')
        board = chess.Board(value+' 0 1')
        if not board.is_valid():
            raise ValueError('Invalid excluded chess state')
        result.add(symmetry_key(board))
    return result


def _whole_group_subset(groups, remaining, size):
    """First achievable exact quota in the fixed shuffled group order.

    Dynamic programming retains the first predecessor for each reachable size.
    It avoids splitting a source game when an earlier large group does not fit
    the remaining space. No seed retries, random replacement or scores are used.
    """
    reachable = {0: None}
    for index in remaining:
        weight = len(groups[index])
        for previous in sorted(reachable, reverse=True):
            total = previous+weight
            if total <= size and total not in reachable:
                reachable[total] = previous, index
        if size in reachable:
            break
    if size not in reachable:
        raise ValueError(f'Insufficient whole source-game groups for frozen split quota {size}')
    chosen, total = [], size
    while total:
        total, index = reachable[total]
        chosen.append(index)
    chosen.reverse()
    return chosen


def select_mates(csv_path, excluded_games, excluded_states, sizes=(1024, 256, 512), seed=10110017):
    """Return fixed train/dev/confirm rows and complete source-selection metadata.

    Rows are first sorted by PuzzleId. Source-game groups are sorted by their
    first PuzzleId and shuffled once with ``seed``. Each split takes the first
    reachable exact whole-group quota in that same order. Canonical mirrored
    solver-state duplicates are removed globally before splitting. Excluded game
    IDs may be supplied as canonical IDs or Lichess URLs. No files are written.
    """
    if len(sizes) != 3 or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError('Specify three positive integer train/dev/confirm sizes')
    if type(seed) is not int:
        raise ValueError('Selection seed must be an integer')
    excluded_games = {source_game_id(value) for value in excluded_games}
    exclusions = _excluded_keys(excluded_states)
    path = Path(csv_path)
    raw = path.read_bytes()
    cut = raw.rfind(b'\n')+1
    if not cut:
        raise ValueError('CSV has no complete newline-terminated rows')
    reader = csv.DictReader(io.StringIO(raw[:cut].decode('utf-8')), strict=True)
    required = {'PuzzleId', 'FEN', 'Moves', 'Rating', 'Themes', 'GameUrl'}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError('CSV is missing required Lichess puzzle columns')
    rejected, parsed, all_ids = Counter(), [], set()
    source_rows, themed_rows = 0, 0
    for source_line, row in enumerate(reader, 2):
        source_rows += 1
        if None in row or any(value is None for value in row.values()) or not row['PuzzleId']:
            rejected['incomplete_csv_row'] += 1
            continue
        if row['PuzzleId'] in all_ids:
            rejected['duplicate_puzzle_id'] += 1
            continue
        all_ids.add(row['PuzzleId'])
        if 'mateIn1' not in row['Themes'].split():
            rejected['not_mateIn1'] += 1
            continue
        themed_rows += 1
        try:
            game_id = source_game_id(row['GameUrl'])
            rating = int(row['Rating'])
            board = chess.Board(row['FEN'])
            moves = row['Moves'].split()
            if rating < 0 or not board.is_valid() or board.chess960 or len(moves) != 2:
                raise ValueError('Invalid rating, board, or mate-in-one move sequence')
            setup = chess.Move.from_uci(moves[0])
            if setup not in board.legal_moves:
                raise ValueError('Opponent setup must be a legal move')
            board.push(setup)
            if board.is_game_over(claim_draw=False):
                raise ValueError('Opponent setup produced a terminal solver board')
            answer = board.copy(stack=False)
            recorded = chess.Move.from_uci(moves[1])
            if recorded not in answer.legal_moves:
                raise ValueError('Recorded answer must be a legal move')
            answer.push(recorded)
            if not answer.is_checkmate():
                rejected['recorded_answer_not_mate'] += 1
                continue
            mates = []
            for move in sorted(board.legal_moves, key=lambda item: item.uci()):
                successor = board.copy(stack=False)
                successor.push(move)
                if successor.is_checkmate():
                    mates.append(move.uci())
            assert moves[1] in mates
            parsed.append({'id': 'lichess-'+row['PuzzleId'], 'puzzle_id': row['PuzzleId'],
                           'fen': board.fen(en_passant='fen'), 'target_uci': moves[1],
                           'mating_uci': mates, 'source_game': row['GameUrl'], 'source_game_id': game_id,
                           'rating': rating, 'source_line': source_line})
        except (ValueError, TypeError, IndexError):
            rejected['invalid_puzzle'] += 1
    # Also exclude mirror-equivalent candidate states from excluded source games,
    # even if another puzzle supplies that same solver board under a different ID.
    blocked = exclusions | {symmetry_key(chess.Board(row['fen'])) for row in parsed
                            if row['source_game_id'] in excluded_games}
    candidates, seen = [], set()
    for row in sorted(parsed, key=lambda item: item['puzzle_id']):
        canonical = symmetry_key(chess.Board(row['fen']))
        if row['source_game_id'] in excluded_games:
            rejected['excluded_source_game'] += 1
        elif canonical in blocked:
            rejected['excluded_solver_state_or_mirror'] += 1
        elif canonical in seen:
            rejected['duplicate_solver_state_or_mirror'] += 1
        else:
            seen.add(canonical)
            candidates.append(row)
    if len(candidates) < sum(sizes):
        raise ValueError(f'Only {len(candidates)} clean mate positions for frozen total quota {sum(sizes)}')
    by_game = defaultdict(list)
    for row in candidates:
        by_game[row['source_game_id']].append(row)
    groups = sorted(by_game.values(), key=lambda group: group[0]['puzzle_id'])
    random.Random(seed).shuffle(groups)
    remaining, result = list(range(len(groups))), {}
    for split, size in zip(SPLITS, sizes, strict=True):
        chosen = _whole_group_subset(groups, remaining, size)
        result[split] = [row for index in chosen for row in groups[index]]
        chosen_set = set(chosen)
        remaining = [index for index in remaining if index not in chosen_set]
    metadata = {
        'source_path': str(path.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'source_bytes': len(raw), 'source_complete_rows': source_rows,
        'discarded_incomplete_suffix_bytes': len(raw)-cut, 'source_columns': reader.fieldnames,
        'mateIn1_theme_rows': themed_rows, 'valid_mate_rows_before_exclusions': len(parsed),
        'clean_candidates': len(candidates), 'candidate_source_games': len(groups),
        'rejection_counts': dict(sorted(rejected.items())),
        'excluded_source_games': len(excluded_games), 'excluded_state_symmetry_classes': len(exclusions),
        'excluded_source_games_sha256': _digest(sorted(excluded_games)),
        'excluded_state_symmetry_classes_sha256': _digest(sorted(exclusions)),
        'sizes': dict(zip(SPLITS, sizes, strict=True)), 'seed': seed,
        'split_source_games': {split: len({row['source_game_id'] for row in records})
                               for split, records in result.items()},
        'selected_puzzle_ids': {split: [row['puzzle_id'] for row in records] for split, records in result.items()},
        'unused_candidates': sum(len(groups[index]) for index in remaining), 'unused_source_games': len(remaining),
        'candidate_manifest_sha256': _digest(candidates),
        'selection': 'PuzzleId sort; group by canonical game ID; one seeded group shuffle; first exact whole-group subset per split',
        'solver_board': 'Apply Moves[0] as the opponent setup; recorded answer is Moves[1]; all mating candidates verified natively',
        'deduplication': 'Global solver piece/turn/castling/EP key and mirrored equivalent; ignore move counters',
        'scope': 'Public convenience-prefix curriculum; no claim of independence from external models pretraining',
    }
    return {'splits': result, 'selection': metadata}
