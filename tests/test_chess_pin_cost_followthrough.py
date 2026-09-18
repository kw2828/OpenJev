# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_cost_followthrough as follow
from test_chess_pin_native_cost import fixture


def test_finite_violation_is_retained_and_never_counts_as_a_pass():
    records, expected = fixture()
    lookup = {(r['seed'], r['root_index'], r['method']): r['prediction'] for r in expected}
    records[0]['prediction']['scores'][1] += 1.1e-5
    for r in records: r.update(follow.compare(r['prediction'], lookup[r['seed'], r['root_index'], r['method']]))
    summary = follow.summarize(records, expected)
    assert not summary['numerical_gate_passed']
    assert summary['failed_timing_records'] == summary['failed_distinct_cases'] == 1
    assert summary['choice_changes'] == 0
    records[0]['score_tolerance_passed'] = True
    with pytest.raises(AssertionError): follow.summarize(records, expected)


def test_choice_changes_are_failures_and_malformed_predictions_still_abort():
    _, expected = fixture(); prior = expected[0]['prediction']
    changed = copy.deepcopy(prior); changed.update(scores=[1., 2.], choice='d2d4')
    assert not follow.compare(changed, prior)['choice_matches']
    for patch in ({'scores': [2., float('nan')]}, {'choice': 'd2d4'}, {'menus': []}):
        malformed = copy.deepcopy(prior); malformed.update(patch)
        with pytest.raises(AssertionError): follow.compare(malformed, prior)
