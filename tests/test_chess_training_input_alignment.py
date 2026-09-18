# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_training_input_alignment as runner
from test_chess_training_inputs import fixture
from openjev.research.chess_child_graph_cache import ChildGraphCache, write_cache
from openjev.research.chess_training_inputs import TrainingInputs


def test_reference_original_block_lookup_matches_merged_shuffle_and_digest(tmp_path):
    boards, _, features, pins = fixture()
    write_cache([{'fen': b.fen(en_passant='fen')} for b in boards], tmp_path/'graphs', provenance={'test': True})
    graph = ChildGraphCache(tmp_path/'graphs').batch([2, 1, 0, 1]); indices = torch.tensor([2, 1, 0, 1])
    actual = TrainingInputs(features, pins).batch(indices, graph)
    expected = runner.reference_batch(features, pins, indices.tolist(), graph)
    assert all(torch.equal(a, b) for a, b in zip([*actual[0], actual[1]], [*expected[0], expected[1]]))
    digest = runner.digest_batch(indices.tolist(), graph, *actual)
    assert digest == runner.digest_batch(indices.tolist(), graph, *expected)
    expected[1][0, 5] = (expected[1][0, 5]+1) % 3
    assert digest != runner.digest_batch(indices.tolist(), graph, *expected)
    with pytest.raises(ValueError): runner.locate(features, [3])
    with pytest.raises(AssertionError): runner.summarize([])
