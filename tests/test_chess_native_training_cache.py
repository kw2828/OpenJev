# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_native_training_cache as cache


def fixture():
    menus = (('a2a3', 'a2a4'), ('b2b3',))
    metadata = [{'fen': chess.STARTING_FEN, 'moves': list(m)} for m in menus]
    block = {'version': cache.native.VERSION, 'nodes': torch.zeros(2, 64, 32),
             'action_features': torch.zeros(3, 120), 'base_logits': torch.zeros(3),
             'candidates': torch.zeros(3, 5, dtype=torch.long), 'offsets': torch.tensor([0, 2, 3]),
             'menus': menus, 'fens': (chess.STARTING_FEN,)*2}
    return block, metadata


def test_block_identity_offsets_precision_and_finiteness_are_checked():
    block, metadata = fixture(); result = cache.block_summary(block, metadata)
    assert result['roots'] == 2 and result['candidates'] == 3
    for patch in ({'offsets': torch.tensor([0, 1, 3])}, {'nodes': block['nodes'].double()},
                  {'base_logits': torch.tensor([0., 0., float('nan')])}, {'menus': block['menus'][::-1]},
                  {'action_features': torch.zeros(3, 119)}, {'version': 'other'}):
        bad = copy.deepcopy(block); bad.update(patch)
        with pytest.raises(AssertionError): cache.block_summary(bad, metadata)


def test_full_dataset_membership_and_candidate_totals_cannot_be_narrowed():
    records = []
    for seed, start, stop in cache.expected_membership():
        candidates = 968036-255*3781 if start == 32640 else 3781
        records.append({'backbone_seed': seed, 'start': start, 'stop': stop,
                        'roots': 128, 'candidates': candidates, 'tensor_bytes': 100})
    result = cache.summarize(records)
    assert result['blocks'] == 768 and result['total_root_feature_rows'] == 98304
    assert result['total_candidate_feature_rows'] == 2904108
    for bad in (records[:-1], records[::-1]):
        with pytest.raises(AssertionError): cache.summarize(bad)
    records[-1]['candidates'] -= 1
    with pytest.raises(AssertionError): cache.summarize(records)
