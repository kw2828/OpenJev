# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_native_feature_multiseed as screen


def fixture():
    p = {'menus': ['a2a3', 'a2a4'], 'choice': 'a2a3', 'scores': [2., 1.]}
    return [{'backbone_seed': b, 'seed': s, 'root_index': i, 'method': m,
             'native': copy.deepcopy(p), 'cached': copy.deepcopy(p), **screen.parent.prior.prior.compare(p, p)}
            for b in screen.SEEDS for s in screen.original.SEEDS for start in range(0, 128, 16)
            for m in screen.original.METHODS for i in range(start, start+16)]


def test_every_backbone_and_case_required_and_failures_preserved():
    records = fixture(); summary = screen.summarize(records)
    assert summary['root_backbone_head_method_checks'] == 8064 and summary['numerical_gate_passed']
    for bad in (records[:-1], records[2688:]+records[:2688]):
        with pytest.raises(AssertionError): screen.summarize(bad)
    records[-1]['native']['scores'][1] += 2e-5
    records[-1].update(screen.parent.prior.prior.compare(records[-1]['native'], records[-1]['cached']))
    summary = screen.summarize(records)
    assert not summary['numerical_gate_passed'] and summary['failed_distinct_cases'] == 1
    assert summary['per_backbone_max_error']['127'] > 1e-5
    records[-1]['score_tolerance_passed'] = True
    with pytest.raises(AssertionError): screen.summarize(records)


def test_nan_and_choice_inconsistency_fail_closed():
    records = fixture(); records[0]['native']['scores'][1] = float('nan')
    with pytest.raises(AssertionError): screen.summarize(records)
    records = fixture(); records[0]['native']['choice'] = 'a2a4'
    with pytest.raises(AssertionError): screen.summarize(records)
