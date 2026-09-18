# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_update_profile as runner


def fixture():
    return [{'backbone_seed': seed, 'arm': arm, 'step': step, 'roots': 128, 'candidates': 3920,
             'indices_sha256': 'index', 'loss': 1., 'gradient_norm': 1.,
             'update_seconds': 1., 'complete_step_seconds': 100. if step == 1 else 2.+i}
            for seed in runner.study.SEEDS for i, arm in enumerate(runner.ARMS) for step in (1, 2, 3)]


def test_full_membership_and_finite_positive_timing_required():
    rows = fixture(); assert runner.summarize(rows)['artificial_updates'] == 99
    for bad in (rows[:-1], rows[::-1]):
        with pytest.raises(AssertionError): runner.summarize(bad)
    for key, value in (('roots', 127), ('candidates', 3921), ('loss', float('nan')), ('complete_step_seconds', .5)):
        bad = copy.deepcopy(rows); bad[0][key] = value
        with pytest.raises(AssertionError): runner.summarize(bad)


def test_projection_excludes_cold_step_and_uses_two_distinct_whole_arm_totals():
    stats = runner.summarize(fixture())
    expected_core = 1536*3*sum(range(2, 9))
    assert stats['core_21_fit_update_seconds'] == expected_core
    assert stats['largest_two_union_cost_arms'] == ['union:edits', 'union:rotated']
    assert stats['maximum_27_fit_update_seconds'] == expected_core+1536*3*(11+12)
