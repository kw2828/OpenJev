# SPDX-License-Identifier: GPL-3.0-only
import collections
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_cost_coverage as cost


def test_graph_windows_preserve_global_indices():
    class Graphs:
        def batch(self, index): return index
    for start in range(0, 128, 16):
        actual = cost.GraphWindow(Graphs(), start).batch(torch.arange(16))
        assert torch.equal(actual, torch.arange(start, start+16))


def test_complete_balanced_membership_and_failure_retention():
    order = cost.timing_order()
    assert len(order) == len(set(order)) == 18816
    slots = collections.Counter((s, i, index % 7, m) for index, (s, i, _r, m) in enumerate(order))
    assert len(slots) == 18816 and set(slots.values()) == {1}
    p = {'menus': ['a2a3', 'a2a4'], 'choice': 'a2a3', 'scores': [2., 1.]}
    expected = [{'seed': s, 'root_index': i, 'method': m, 'prediction': p}
                for s in cost.original.SEEDS for i in range(128) for m in cost.original.METHODS]
    records = [{'seed': s, 'root_index': i, 'repeat': r, 'method': m, 'prediction': p,
                'milliseconds': float(cost.original.METHODS.index(m)+1), **cost.prior.compare(p, p)}
               for s, i, r, m in order]
    records[0]['prediction'] = {**p, 'scores': [2., 1.00002]}
    records[0].update(cost.prior.compare(records[0]['prediction'], p))
    summary = cost.summarize(records, expected)
    assert not summary['numerical_gate_passed'] and summary['failed_distinct_cases'] == 1
    assert summary['median_paired_ratio_to_wldn']['joint'] == 1.5
    with pytest.raises(AssertionError): cost.summarize(records[:-1], expected)
