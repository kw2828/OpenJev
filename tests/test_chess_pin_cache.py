# SPDX-License-Identifier: GPL-3.0-only
import copy
from pathlib import Path
import sys

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_training_cache as runner
from openjev.research.chess_pin_cache import gather, pack, validate
from openjev.research.chess_pin_factors import candidate_factors


def fixture():
    boards = [chess.Board(), chess.Board('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1')]
    records = [candidate_factors(b) for b in boards]; fens = [b.fen(en_passant='fen') for b in boards]
    return records, fens, validate(pack(records, fens))


def test_independent_packing_and_reordered_repeated_root_gather():
    records, fens, cache = fixture(); runner.compare(cache, runner.reference_pack(records, fens))
    order = torch.tensor([1, 0, 1])
    expected = pack([records[i] for i in order], [fens[i] for i in order])
    assert torch.equal(gather(cache, order), expected['factors'])
    assert gather(cache, torch.tensor([0])).shape == (0, 6)
    for bad in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([], dtype=torch.long)):
        with pytest.raises(ValueError): gather(cache, bad)


def test_corrupt_offsets_root_ownership_and_factor_indices_rejected():
    _, _, cache = fixture()
    for field in ('candidate_offsets', 'factor_offsets'):
        bad = copy.deepcopy(cache); bad[field][-1] += 1
        with pytest.raises(ValueError): validate(bad)
    for column, value in ((0, 0), (1, 2), (2, 64), (5, 3), (3, 4)):
        bad = copy.deepcopy(cache); bad['factors'][0, column] = value
        with pytest.raises(ValueError): validate(bad)
    bad = copy.deepcopy(cache); bad['factors'] = bad['factors'].float()
    with pytest.raises(ValueError): validate(bad)
