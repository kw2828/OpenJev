# SPDX-License-Identifier: GPL-3.0-only
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_factor_preflight as screen
from openjev.research.chess_pin_factor_head import ChessPinFactorHead
from test_chess_graph_contrast import fixture
from test_chess_pin_factor_head import factors


def test_individual_candidate_keeps_only_its_factors_and_compact_graph():
    args = list(fixture()); args[-1] = args[-1][args[3]]
    f = factors(); head = ChessPinFactorHead().double()
    with torch.no_grad(): head.output.weight.fill_(.1)
    actual = head(*args, f)
    for compact, (root, slot) in enumerate(args[3].nonzero().tolist()):
        subset = screen.single_factors(f, compact)
        assert not len(subset) or (subset[:, 0] == 0).all()
        expected = head(*screen.single_args(args, root, slot, compact), subset, reference=True)
        torch.testing.assert_close(actual[root, slot], expected[0, 0], atol=1e-11, rtol=1e-11)


def test_checkpoint_tolerance_membership_and_nonfinite_fail_closed():
    a = {'w': torch.tensor([1.], dtype=torch.double)}
    assert screen.compare_states(a, {'w': a['w']+1e-10}, 1e-9) < 1e-9
    for b in ({'w': a['w']+1e-8}, {'w': torch.tensor([float('nan')])}, {'other': a['w']}, {'w': a['w'][None]}):
        with pytest.raises(AssertionError): screen.compare_states(a, b, 1e-9)
    with pytest.raises(AssertionError):
        screen.compare_states({'ok': a['w'], 'bad': a['w']},
                              {'ok': a['w'], 'bad': torch.tensor([float('nan')], dtype=torch.double)}, 1e-9)


def test_frozen_budget_and_scope():
    assert screen.PROTOCOL['roots'] == 128
    assert screen.PROTOCOL['total_updates'] == 6
    assert screen.PROTOCOL['root_seed_arm_checks'] == 3*2*128
    assert screen.PROTOCOL['score_tolerance'] == 1e-5
    assert screen.PROTOCOL['update_replay_tolerance'] == 1e-9
