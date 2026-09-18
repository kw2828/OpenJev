# SPDX-License-Identifier: GPL-3.0-only
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_wldn_interventions as diagnostic


def example():
    rows = [{'id': 'a', 'game_id': 'game1', 'target_uci': 'e2e4'},
            {'id': 'b', 'game_id': 'game1', 'target_uci': 'd2d4'},
            {'id': 'c', 'game_id': 'game2', 'target_uci': 'a2a3'}]
    records = [{**{k: row[k] for k in ('id', 'game_id')}, 'index': i, 'choice': row['target_uci'],
                'correct': True, 'target_nll': 1.} for i, row in enumerate(rows)]
    return rows, records


def test_identity_alignment_rejects_omissions_duplicates_and_wrong_labels():
    rows, records = example()
    assert diagnostic.aligned(records[::-1], rows) == records
    for bad in [records[:-1], [records[0], records[0], records[2]],
                [dict(records[0], game_id='other'), *records[1:]],
                [dict(records[0], correct=False), *records[1:]],
                [dict(records[0], target_nll=float('nan')), *records[1:]]]:
        with pytest.raises(AssertionError): diagnostic.aligned(bad, rows)


def test_paired_transitions_include_changed_wrong_moves():
    _, native = example(); native[1].update(choice='h2h3', correct=False)
    native[2].update(choice='h2h4', correct=False)
    changed = [dict(r) for r in native]
    changed[0].update(choice='h2h3', correct=False, target_nll=2.)
    changed[1].update(choice='d2d4', correct=True)
    changed[2].update(choice='h2h3', correct=False)
    result = diagnostic.paired(native, changed)
    assert result == {'agreement_loss': 0., 'nll_increase': 1/3, 'decision_disagreement': 1.,
                      'native_correct_to_wrong': 1, 'native_wrong_to_correct': 1}
    with pytest.raises(AssertionError): diagnostic.paired(native, changed[::-1])


def test_game_bootstrap_averages_seeds_then_weights_positions():
    differences = np.array([[1, 1, -1], [0, 0, -1], [-1, -1, -1]])
    result = diagnostic.interval(differences, ['a', 'a', 'b'], 'dev')
    assert result == {'games': 2, 'point_loss': -1/3, 'percentile95': [-1., 0.]}
    assert diagnostic.interval(differences[::-1], ['a', 'a', 'b'], 'dev') == result
    with pytest.raises(AssertionError): diagnostic.interval(differences[:2], ['a', 'a', 'b'], 'dev')


def test_replay_rejects_changed_choice_and_nonfinite_loss():
    _, records = example()
    assert diagnostic.compare_records(records, records) == 0
    for change in [dict(records[0], choice='a2a4'), dict(records[0], target_nll=float('inf')),
                   dict(records[0], target_nll=1.001)]:
        with pytest.raises(AssertionError): diagnostic.compare_records(records, [change, *records[1:]])


def test_protocol_covers_all_modes_seeds_panels_without_training():
    p = diagnostic.PROTOCOL
    assert len(p['modes'])*len(p['seeds'])*len(p['panels'])*p['positions_per_panel'] == p['new_prediction_records'] == 73728
    assert p['copied_reference_records'] == len(p['seeds'])*len(p['panels'])*p['positions_per_panel']
    assert p['modes'] == ['native', 'zero_delta', 'root_diff_graph', 'zero_edge_flags', 'zero_pooled', 'permuted_delta']
