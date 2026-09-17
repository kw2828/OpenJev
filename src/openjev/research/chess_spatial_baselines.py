"""Fixed, untrained baselines for bounded legal-candidate chess decisions.

Uniform reports its exact expected agreement without sampling a move. Frequency
uses only training target-UCI counts. Material reuses the arena's native one-ply
heuristic. Evaluation targets define scoring and descriptive slices, never moves.
"""

from collections import Counter

import chess

from openjev.research.chess_arena import greedy_material_policy


def _board(row):
    board = chess.Board(row['fen'])
    if not board.is_valid() or board.chess960:
        raise ValueError('Baseline expects a valid standard chess position')
    target = chess.Move.from_uci(row['target_uci'])
    if target not in board.legal_moves:
        raise ValueError('Baseline target must be a legal move')
    return board, target


def target_categories(board, target):
    """Native target-move slices; metadata for evaluation, never model features."""
    capture = board.is_capture(target)
    return {'capture': capture, 'noncapture': not capture,
            'check': board.gives_check(target), 'in_check': board.is_check(),
            'promotion': bool(target.promotion), 'castling': board.is_castling(target)}


def _metrics(predictions, expected=False):
    field = 'expected_correct' if expected else 'correct'
    agreement = ('expected_top1_teacher_agreement' if expected else
                 'top1_teacher_agreement')
    slices = {}
    for category in predictions[0]['categories']:
        selected = [row for row in predictions if row['categories'][category]]
        slices[category] = {
            'examples': len(selected),
            agreement: sum(row[field] for row in selected)/len(selected) if selected else None,
        }
    return {'examples': len(predictions),
            agreement: sum(row[field] for row in predictions)/len(predictions),
            'slices': slices}


def simple_baselines(train_rows, eval_rows):
    """Return uniform, training-UCI frequency and one-ply material evidence.

Each baseline contains ``method``, ``metrics`` and one prediction per evaluation
row. Uniform's ``choice`` and ``correct`` are None: its ``expected_correct`` and
``target_probability`` are exactly 1/legal_count. Deterministic baselines report
their actual choice and correctness. Sorted original UCI strings break ties.
No model, engine, random number generator, fit, or evaluation-label selection is
involved. Empty panels, duplicate IDs and illegal targets are rejected.
"""
    if not train_rows or not eval_rows:
        raise ValueError('Training and evaluation panels must be nonempty')
    for rows in (train_rows, eval_rows):
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Repeated position ID within a baseline panel')
        for row in rows:
            _board(row)
    frequency = Counter(row['target_uci'] for row in train_rows)
    material = greedy_material_policy()
    outputs = {name: [] for name in ('uniform_random', 'training_move_frequency', 'greedy_material')}
    for row in eval_rows:
        board, target = _board(row)
        legal = sorted(move.uci() for move in board.legal_moves)
        common = {'id': row['id'], 'game_id': row['game_id'], 'target': target.uci(),
                  'legal_count': len(legal), 'categories': target_categories(board, target)}
        outputs['uniform_random'].append({**common, 'choice': None, 'correct': None,
                                          'expected_correct': 1/len(legal),
                                          'target_probability': 1/len(legal)})
        choices = {'training_move_frequency': max(legal, key=lambda move: frequency[move]),
                   'greedy_material': material(board)['choice']}
        for name, choice in choices.items():
            correct = choice == target.uci()
            outputs[name].append({**common, 'choice': choice, 'correct': correct,
                                  'target_probability': float(correct)})
    methods = {
        'uniform_random': 'Exact mean of 1/legal_count; no sampled move or model inference',
        'training_move_frequency': 'Most frequent training target UCI among legal candidates; sorted-UCI ties',
        'greedy_material': 'Native one-ply material P/N/B/R/Q=1/3/3/5/9; sorted-UCI ties; no learned parameters',
    }
    return {name: {'method': methods[name],
                   'metrics': _metrics(predictions, expected=name == 'uniform_random'),
                   'predictions': predictions}
            for name, predictions in outputs.items()}
