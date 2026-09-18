# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_native_evaluation_inputs as runner


def test_metadata_keeps_only_identity_and_rejects_misalignment_or_duplicate_ids():
    raw = [{'id': 'a', 'fen': 'A', 'target': 'unused'}, {'id': 'b', 'fen': 'B', 'value': 123}]
    graphs = [{'index': i, 'fen': row['fen']} for i, row in enumerate(raw)]
    assert runner.metadata_rows(raw, graphs) == [{'index': 0, 'id': 'a', 'fen': 'A'}, {'index': 1, 'id': 'b', 'fen': 'B'}]
    for bad in (raw[::-1], raw[:1], [raw[0], raw[0]]):
        with pytest.raises(AssertionError): runner.metadata_rows(bad, graphs)


def fixture():
    features, pins, batches = [], [], []
    for split, total in runner.SPLITS.items():
        for start in range(0, 2048, 128):
            count = total-15*3000 if start == 1920 else 3000
            pins.append({'split': split, 'start': start, 'stop': start+128, 'factor_rows': 2,
                         'coverage': {'roots': 128, 'candidates': count, 'witness_roles_by_owner_retained_added_removed': [[1, 0, 0], [0, 1, 0]]}})
        for seed in runner.SEEDS:
            for start in range(0, 2048, 128):
                count = total-15*3000 if start == 1920 else 3000
                features.append({'split': split, 'backbone_seed': seed, 'start': start, 'stop': start+128,
                                 'roots': 128, 'candidates': count, 'tensor_bytes': 100})
            for order in ('source', 'shuffled'):
                for start in range(0, 2048, 128):
                    count = total-15*3000 if start == 1920 else 3000
                    batches.append({'split': split, 'backbone_seed': seed, 'order': order, 'start': start,
                                    'roots': 128, 'candidates': count, 'factor_rows': 2})
    return features, pins, batches


def test_complete_panel_membership_counts_and_all_order_passes_required():
    records = fixture(); summary = runner.summarize(*records)
    assert (summary['feature_blocks'], summary['common_pin_blocks'], summary['alignment_batches']) == (96, 32, 192)
    for index in range(3):
        bad = copy.deepcopy(records); bad[index].pop()
        with pytest.raises(AssertionError): runner.summarize(*bad)
        bad = copy.deepcopy(records); bad[index].reverse()
        with pytest.raises(AssertionError): runner.summarize(*bad)
    for index, key in ((0, 'candidates'), (1, 'factor_rows'), (2, 'factor_rows')):
        bad = copy.deepcopy(records); bad[index][-1][key] += 1
        with pytest.raises(AssertionError): runner.summarize(*bad)
