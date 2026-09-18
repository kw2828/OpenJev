# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_pairwise_preflight as runner


def test_finite_failures_retained_without_relaxing_gate_and_bad_vectors_rejected():
    a = {'menus': ['a2a3', 'a2a4'], 'scores': [0., 1.], 'choice': 'a2a4'}
    b = copy.deepcopy(a); b['scores'][0] = 2e-5
    result = runner.compare(a, b)
    assert not result['score_tolerance_passed'] and result['choice_matches']
    bad = copy.deepcopy(b); bad['choice'] = 'a2a3'
    with pytest.raises(AssertionError): runner.compare(a, bad)
    bad = copy.deepcopy(b); bad['scores'][0] = float('nan')
    with pytest.raises(AssertionError): runner.compare(a, bad)


def test_complete_probe_membership_and_scores_required():
    records = []
    for b in runner.SEEDS:
        for s in runner.PROBES:
            for i in range(128):
                count = 3920-127*30 if i == 127 else 30
                prediction = {'menus': [str(j) for j in range(count)], 'scores': [0.]*count, 'choice': '0'}
                records.append({'backbone_seed': b, 'probe_seed': s, 'root_index': i,
                                'native': prediction, 'cached': prediction, **runner.compare(prediction, prediction)})
    stats = runner.summarize(records)
    assert stats['candidate_score_checks'] == 35280 and stats['numerical_gate_passed']
    with pytest.raises(AssertionError): runner.summarize(records[:-1])
    bad = copy.deepcopy(records); bad[0]['max_score_error'] = 1
    with pytest.raises(AssertionError): runner.summarize(bad)
    bad = copy.deepcopy(records); bad[0]['cached'] = copy.deepcopy(bad[0]['cached'])
    bad[0]['cached']['scores'][1] = 2e-5; bad[0]['cached']['choice'] = '1'
    bad[0].update(runner.compare(bad[0]['native'], bad[0]['cached']))
    assert runner.summarize(bad)['failed_distinct_cases'] == 1
