import chess
import numpy as np
import pytest

from openjev.research.chess_arena import chess_record
from openjev.research.chess_scorer import chess_messages, label_pool, move_distribution


def test_all_starting_moves_scored_without_twelve_move_limit():
    record = chess_record(chess.Board())
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    messages, candidates = chess_messages(record, labels)
    assert len(candidates) == 20
    assert candidates == sorted(candidates, key=lambda c: c['id'])
    answer = move_distribution(np.arange(100), list(range(20)), candidates)
    assert set(answer['probabilities']) == {c['id'] for c in candidates}
    assert sum(answer['probabilities'].values()) == pytest.approx(1)
    assert answer['choice'] == candidates[-1]['id']
    assert answer['candidate_token_mass'] < 1e-20
    assert 'Stockfish' not in str(messages)


def test_label_pool_filters_multitoken_and_duplicate_ids():
    class Tokenizer:
        def encode(self, label, **kwargs):
            if label == 'AA':
                return [1, 2]
            if label == 'AB':
                return [ord('A')]
            return [int.from_bytes(label.encode(), 'big')]
    labels, ids = label_pool(Tokenizer())
    assert 'AA' not in labels and 'AB' not in labels
    assert len(ids) == len(set(ids)) >= 218


def test_invalid_distributions_and_candidate_overflow_fail():
    with pytest.raises(ValueError):
        move_distribution([float('nan')], [0], [{'id': 'a1a2'}])
    with pytest.raises(ValueError):
        chess_messages(chess_record(chess.Board()), ['A'])
    with pytest.raises(ValueError):
        move_distribution([0, 1], [0, 0], [{'id': 'a1a2'}, {'id': 'a1b1'}])
