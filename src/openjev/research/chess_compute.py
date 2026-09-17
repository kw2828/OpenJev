"""Measured inference controls for frozen spatial chess policies.

Exact successors use python-chess rules and preserve the caller's move history.
They are explicit symbolic lookahead, not learned dynamics. ``scores`` are
side-to-move values in [-1, 1], never calibrated probabilities. Draw claims are
not automatic: native outcomes use ``claim_draw=False`` throughout.

Decision ``compute`` counts distinguish full policy calls, nonterminal boards
processed by the value-only path, native terminal successors, and symbolic
transitions. Terminal checks count inspected successors, not input validation.
Wall time includes preparation, inference, result parsing and device barriers;
it excludes model loading. MPS/CUDA are synchronized at both ends of a decision.
"""

import math
import time

import chess
import numpy as np
import torch

from openjev.research.chess_spatial import encode_board


def _depth(model, depth):
    if type(depth) is not int or depth < 1:
        raise ValueError('depth must be a positive integer')
    return model._depth(depth)


def _validate_board(board):
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard chess board')
    if board.outcome(claim_draw=False) is not None:
        raise ValueError('Expected a nonterminal chess board')


def _legal(board):
    _validate_board(board)
    moves = sorted(board.legal_moves, key=lambda move: move.uci())
    if not moves:
        raise ValueError('Expected a nonempty legal move menu')
    return moves


def _device(model):
    return next(model.parameters()).device


def _synchronize(model):
    device = _device(model)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def _start(model):
    started = time.perf_counter()
    _synchronize(model)
    return started


def _counts(model, depth, legal_count):
    return {'device': str(_device(model)), 'depth': depth, 'legal_count': legal_count,
            'candidates_evaluated': 0, 'policy_forward_calls': 0,
            'boardvalue_evaluations': 0, 'boardvalue_forward_calls': 0,
            'terminal_evaluations': 0, 'terminal_checks': 0, 'symbolic_transition_count': 0}


def _finish(model, started, result, counts):
    _synchronize(model)
    elapsed = (time.perf_counter()-started)*1000
    return {**result, 'latency_ms': elapsed,
            'compute': {**counts, 'total_wall_ms': elapsed}}


def _policy(model, board, depth, moves):
    result = model.choose(board.copy(stack=True), depth=depth)
    names = {move.uci() for move in moves}
    probabilities = result.get('probabilities')
    if (result.get('choice') not in names or not isinstance(probabilities, dict)
            or set(probabilities) != names
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1., rel_tol=1e-5, abs_tol=1e-6)
            or not math.isfinite(result.get('value', math.nan))
            or not -1 <= result['value'] <= 1):
        raise ValueError('Invalid spatial policy response or legal probabilities')
    return result


@torch.inference_mode()
def _value_boards(model, boards, depth):
    """Internal path: boards already checked; never execute candidate/aux heads."""
    if not boards:
        return []
    model.eval()
    parameter = next(model.parameters())
    observations = torch.from_numpy(np.stack([encode_board(board) for board in boards])).to(parameter)
    hidden = model.encoder(observations)
    for index in range(depth):
        hidden = model.blocks[index](hidden) if model.mode == 'cnn' else model.core(hidden)
    values = model.value_head(hidden.mean(dim=(2, 3))).tanh().squeeze(-1).cpu().tolist()
    if len(values) != len(boards) or any(not math.isfinite(value) or not -1 <= value <= 1 for value in values):
        raise ValueError('Nonfinite or invalid bounded board value')
    return values


def value_boards(model, boards, depth=4):
    """Return values for nonterminal boards, each from its own side to move.

Runs encoder, requested recurrent/untied blocks and value head in one batch.
Neither legal candidate encoding nor policy/auxiliary heads are evaluated.
Input boards and their move stacks are untouched. Empty input returns ``[]``.
Terminal inputs are rejected so native terminal values cannot be overwritten by
a learned estimate. This helper returns values only; decision wrappers time and
count its internal value-only operation.
    """
    depth = _depth(model, depth)
    boards = list(boards)
    for board in boards:
        _validate_board(board)
    return _value_boards(model, boards, depth)


def choose_depth(model, board, depth):
    """Original legal-candidate policy at a fixed, explicitly requested depth."""
    started = _start(model)
    depth = _depth(model, depth)
    moves = _legal(board)
    result = _policy(model, board, depth, moves)
    counts = _counts(model, depth, len(moves))
    counts.update({'candidates_evaluated': len(moves), 'policy_forward_calls': 1})
    return _finish(model, started, {**result, 'method': 'fixed_depth_policy'}, counts)


def exact_successor_decision(model, board, top_k=None, depth=4):
    """Rank exact legal successors using native outcomes or negated values.

``top_k=None`` considers every legal move without a root policy call. A positive
integer first shortlists that many moves using the root policy (or all legal
moves if fewer). Probability ties and final score ties use sorted original UCI.
Only shortlisted moves are pushed or inspected for terminal outcomes.

A successor value uses its opponent-to-move perspective, so the root score is
its negative. Native wins score +1, draws 0 and losses -1. A proven terminal win
outranks an equally scored nonterminal value rounded to +1. Returned scores
cover the evaluated subset and are not normalized into probabilities.
    """
    started = _start(model)
    depth = _depth(model, depth)
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError('top_k must be None or a positive integer')
    moves = _legal(board)
    counts = _counts(model, depth, len(moves))
    if top_k is not None:
        policy = _policy(model, board, depth, moves)
        moves = sorted(moves, key=lambda move: (-policy['probabilities'][move.uci()], move.uci()))[:top_k]
        counts['policy_forward_calls'] = 1
    # Keep score and value-batch ordering independent of probability ordering.
    moves = sorted(moves, key=lambda move: move.uci())
    scores, terminals, nonterminal_names, successors = {}, {}, [], []
    for move in moves:
        successor = board.copy(stack=True)
        successor.push(move)
        outcome = successor.outcome(claim_draw=False)
        name = move.uci()
        if outcome is not None:
            scores[name] = 0. if outcome.winner is None else (1. if outcome.winner == board.turn else -1.)
            terminals[name] = {'value': scores[name], 'termination': outcome.termination.name.lower()}
        else:
            nonterminal_names.append(name)
            successors.append(successor)
    values = _value_boards(model, successors, depth)
    scores.update({name: -value for name, value in zip(nonterminal_names, values, strict=True)})
    scores = dict(sorted(scores.items()))
    choice = min(scores, key=lambda name: (-scores[name], -(name in terminals and scores[name] == 1.), name))
    counts.update({'candidates_evaluated': len(moves), 'boardvalue_evaluations': len(successors),
                   'boardvalue_forward_calls': int(bool(successors)), 'terminal_evaluations': len(terminals),
                   'terminal_checks': len(moves), 'symbolic_transition_count': len(moves)})
    result = {'choice': choice, 'scores': scores, 'score_semantics': 'root-side bounded successor values; not probabilities',
              'terminal_successors': terminals, 'method': 'exact_successor_value',
              'top_k': top_k, 'depth': depth, 'mode': model.mode, 'seed': model.seed}
    return _finish(model, started, result, counts)


def terminal_guard_decision(model, board, depth=4):
    """Explicit rule-based control: take a native mate in one, else original policy.

Scans every legal successor and resolves multiple mates by sorted UCI. The
fallback policy and probabilities are preserved exactly. On a winning mate it
returns native scores for winning moves without fabricating policy confidence.
    """
    started = _start(model)
    depth = _depth(model, depth)
    moves = _legal(board)
    counts = _counts(model, depth, len(moves))
    wins, terminals = [], 0
    for move in moves:
        successor = board.copy(stack=True)
        successor.push(move)
        outcome = successor.outcome(claim_draw=False)
        if outcome is not None:
            terminals += 1
            if outcome.termination == chess.Termination.CHECKMATE and outcome.winner == board.turn:
                wins.append(move.uci())
    counts.update({'candidates_evaluated': len(moves), 'terminal_evaluations': terminals,
                   'terminal_checks': len(moves), 'symbolic_transition_count': len(moves)})
    if wins:
        result = {'choice': wins[0], 'scores': dict.fromkeys(wins, 1.),
                  'score_semantics': 'native immediate winning mates; not probabilities',
                  'mode': model.mode, 'seed': model.seed, 'depth': depth}
    else:
        result = _policy(model, board, depth, moves)
        counts['policy_forward_calls'] = 1
    result = {**result, 'method': 'native_mate_guard_then_policy', 'guard_triggered': bool(wins),
              'winning_mates': wins}
    return _finish(model, started, result, counts)
