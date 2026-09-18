# SPDX-License-Identifier: GPL-3.0-only
import collections
import copy
from pathlib import Path
import sys

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_native_cost as cost
from openjev.research.chess_pin_factors import candidate_factors


def fixture():
    prediction = {'menus': ['e2e4', 'd2d4'], 'choice': 'e2e4', 'scores': [2., 1.]}
    expected = [{'seed': s, 'root_index': i, 'method': m, 'prediction': copy.deepcopy(prediction)}
                for s in cost.SEEDS for i in range(16) for m in cost.METHODS]
    records = [{'seed': s, 'root_index': i, 'repeat': r, 'method': m,
                'prediction': copy.deepcopy(prediction), 'cached_score_max_error': 0.,
                'milliseconds': float(cost.METHODS.index(m)+1)} for s, i, r, m in cost.timing_order()]
    return records, expected


def test_balanced_membership_and_paired_timing_arithmetic():
    order = cost.timing_order()
    assert len(order) == len(set(order)) == 2352
    slots = collections.Counter((s, i, index % 7, m) for index, (s, i, _r, m) in enumerate(order))
    assert len(slots) == 2352 and set(slots.values()) == {1}
    records, expected = fixture(); summary = cost.summarize(records, expected)
    assert summary['median_paired_ratio_to_wldn']['joint'] == 1.5
    assert summary['median_paired_ratio_to_wldn']['wldn'] == 1
    assert summary['median_complete_ms']['graph_mlp'] == 7


def test_corrupt_timing_scores_membership_and_error_fields_rejected():
    records, expected = fixture()
    for bad in (records[:-1], records[::-1]):
        with pytest.raises(AssertionError): cost.summarize(bad, expected)
    for patch in ({'milliseconds': float('nan')}, {'milliseconds': 0}, {'cached_score_max_error': 1.}):
        bad = copy.deepcopy(records); bad[0].update(patch)
        with pytest.raises(AssertionError): cost.summarize(bad, expected)
    for patch in ({'scores': [2., float('nan')]}, {'scores': [2., 1.01]},
                  {'choice': 'd2d4'}, {'menus': ['d2d4', 'e2e4']}, {'scores': [2.]}):
        bad = copy.deepcopy(records); bad[0]['prediction'].update(patch)
        with pytest.raises(AssertionError): cost.summarize(bad, expected)
    with pytest.raises(AssertionError): cost.summarize(records, expected[:-1])


@pytest.mark.parametrize('fen', ['k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1',
                               '4k3/4n3/8/8/8/8/8/K3R3 b - - 0 1', chess.STARTING_FEN])
def test_root_only_native_packing_matches_independent_candidate_filter(fen):
    board = chess.Board(fen); record = candidate_factors(board, reference=True)
    expected = [(i, *w[:4], 0) for i, candidate in enumerate(record['candidates'])
                for w in candidate['factors'] if w[4] in (0, 2)]
    assert torch.equal(cost.root_factors(board, len(record['candidates'])),
                       torch.tensor(expected, dtype=torch.long).reshape(-1, 6))


def test_native_paths_do_not_charge_controls_for_unused_pin_extraction(monkeypatch):
    calls = collections.Counter()
    original_candidates, original_root = cost.candidate_factors, cost.pin_witnesses
    def candidates(board):
        calls['candidates'] += 1
        return original_candidates(board)
    def root(board, perspective):
        calls['root'] += 1
        return original_root(board, perspective)
    monkeypatch.setattr(cost, 'candidate_factors', candidates)
    monkeypatch.setattr(cost, 'pin_witnesses', root)
    monkeypatch.setattr(cost.source.prior, 'candidate_features',
                        lambda model, nodes, candidates: torch.zeros(1, candidates.shape[1], 120))
    class Backbone:
        def __call__(self, observations, candidates, mask):
            return torch.zeros_like(mask, dtype=torch.float32), None, torch.zeros(1, 32, 8, 8)
    class Head:
        def __call__(self, *args):
            assert len(args) in (7, 8)
            if len(args) == 8: assert args[-1].shape[1] == 6
            return args[4]
    for method in cost.METHODS:
        calls.clear()
        result = cost.decision(Backbone(), None if method == 'base' else Head(), method,
                               'k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1')
        assert result['choice'] in result['menus']
        expected = {'candidates': 1} if method in ('joint', 'separable', 'counts') else (
            {'root': 1} if method == 'root_only' else {})
        assert dict(calls) == expected
