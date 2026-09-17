"""Synthetic-only checks for evaluation integrity and timing isolation."""

import copy
import json
import math

import chess
import numpy as np
import pytest
import torch
from torch import nn

from openjev.research.chess_anchor_eval import (
    audit_predictions,
    evaluate,
    probe,
    summarize,
    tensorize,
)
from openjev.research.chess_spatial import SpatialChess, encode_board, encode_candidates


def rows_fixture():
    fens = [chess.STARTING_FEN, '7k/P7/8/8/8/8/8/7K w - - 0 1',
            '4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1']
    rows = []
    for index, fen in enumerate(fens):
        names, _ = encode_candidates(chess.Board(fen))
        rows.append({'id': f'position-{index}', 'game_id': f'game-{index}', 'fen': fen,
                     'target_uci': names[0 if index != 1 else 1], 'target_value': (index-1)*.5})
    return rows


class SyntheticPolicy(nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.placeholder = nn.Parameter(torch.tensor(0.0))
        self.fault = fault
        self.forward_sizes = []
        self.choose_boards = []

    def forward(self, observations, candidates, legal_mask, depth=None):
        self.forward_sizes.append((len(observations), depth))
        logits = -torch.arange(candidates.shape[1], dtype=torch.float32).expand(len(observations), -1)*.125
        logits = logits.clone().masked_fill(~legal_mask, 999)
        hidden = torch.full((len(observations), 2, 8, 8), 1e30)
        value = torch.zeros(len(observations))
        if self.fault == 'logits':
            logits[0, 0] = torch.inf
        if self.fault == 'hidden':
            hidden[0, 0, 0, 0] = torch.nan
        if self.fault == 'value':
            value[0] = torch.nan
        if self.fault == 'unbounded_value':
            value[0] = 1.1
        return logits, value, hidden

    def choose(self, board, depth=None):
        self.choose_boards.append((board.fen(), depth))
        names, _ = encode_candidates(board)
        return {'choice': names[0], 'probabilities': {name: 1/len(names) for name in names}}


def run_synthetic(tmp_path, rows=None, model=None):
    rows = rows or rows_fixture()
    tensors, menus = tensorize(rows)
    model = model or SyntheticPolicy()
    path = tmp_path/'predictions.jsonl'
    result = evaluate(model, rows, tensors, menus, 4, path)
    return rows, model, path, result


def test_tensorize_uses_original_encoding_and_pads_only_menu():
    rows = rows_fixture()
    tensors, menus = tensorize(rows)
    assert set(tensors) == {'observations', 'candidates', 'mask', 'targets', 'values'}
    assert tensors['observations'].dtype == torch.float32
    assert tensors['candidates'].dtype == tensors['targets'].dtype == torch.int64
    assert tensors['mask'].dtype == torch.bool
    assert len(set(map(len, menus))) > 1
    for index, row in enumerate(rows):
        board = chess.Board(row['fen'])
        names, candidates = encode_candidates(board)
        assert menus[index] == names
        np.testing.assert_array_equal(tensors['observations'][index].numpy(), encode_board(board))
        np.testing.assert_array_equal(tensors['candidates'][index, :len(names)].numpy(), candidates)
        assert tensors['mask'][index].sum() == len(names)
        assert not tensors['candidates'][index, len(names):].any()
        assert menus[index][int(tensors['targets'][index])] == row['target_uci']


def test_labels_never_change_model_inputs_or_predictions(tmp_path):
    rows = rows_fixture()
    changed = copy.deepcopy(rows)
    for row in changed:
        row['target_uci'] = encode_candidates(chess.Board(row['fen']))[0][-1]
        row['target_value'] = .9
    left, menus = tensorize(rows)
    right, new_menus = tensorize(changed)
    for key in ('observations', 'candidates', 'mask'):
        assert torch.equal(left[key], right[key])
    assert menus == new_menus
    evaluate(SyntheticPolicy(), rows, left, menus, 4, tmp_path/'a.jsonl')
    evaluate(SyntheticPolicy(), changed, right, new_menus, 4, tmp_path/'b.jsonl')
    a = [json.loads(line) for line in (tmp_path/'a.jsonl').read_text().splitlines()]
    b = [json.loads(line) for line in (tmp_path/'b.jsonl').read_text().splitlines()]
    for p, q in zip(a, b, strict=True):
        for key in ('choice', 'value', 'max_probability', 'entropy', 'hidden_rms', 'logit_span'):
            assert p[key] == q[key]


def test_saved_arithmetic_padding_rms_and_metrics_roundtrip(tmp_path):
    rows, model, path, result = run_synthetic(tmp_path)
    metrics, predictions = audit_predictions(path, rows)
    assert metrics == result['metrics'] == summarize(predictions)
    assert metrics['examples'] == 3 and metrics['correct'] == 2
    assert metrics['agreement'] == 2/3
    assert metrics['value_mae'] == 1/3
    assert metrics['mismatch_confidence'] == predictions[1]['max_probability']
    assert result['evaluation_wall_seconds'] >= 0
    assert model.training
    assert model.forward_sizes == [(3, 4)]
    for index, prediction in enumerate(predictions):
        count = len(encode_candidates(chess.Board(rows[index]['fen']))[0])
        logits = -torch.arange(count, dtype=torch.float64)*.125
        distribution = logits.softmax(0)
        assert prediction['max_probability'] == pytest.approx(float(distribution.max()))
        assert prediction['entropy'] == pytest.approx(float(-(distribution*distribution.log()).sum()))
        assert prediction['logit_span'] == (count-1)*.125
        assert prediction['hidden_rms'] == pytest.approx(float(torch.tensor(1e30)))
        assert prediction['target_probability'] == math.exp(-prediction['target_nll'])


def test_integer_game_ids_preserved_through_tensorize_evaluate_and_audit(tmp_path):
    rows = rows_fixture()
    for index, row in enumerate(rows):
        row['game_id'] = index
    rows, _, path, result = run_synthetic(tmp_path, rows)
    metrics, predictions = audit_predictions(path, rows)
    assert metrics == result['metrics']
    assert [row['game_id'] for row in predictions] == [0, 1, 2]
    assert all(type(row['game_id']) is int for row in predictions)
    for replacement in ('0', False, 0.0):
        changed = copy.deepcopy(predictions)
        changed[0]['game_id'] = replacement
        path.write_text('\n'.join(json.dumps(row) for row in changed)+'\n')
        with pytest.raises(ValueError, match='identity'):
            audit_predictions(path, rows)


@pytest.mark.parametrize('invalid_game_id', [True, False, -1, 1.0, '', None])
def test_invalid_game_id_types_rejected(invalid_game_id):
    rows = rows_fixture()
    rows[0]['game_id'] = invalid_game_id
    with pytest.raises(ValueError, match='game ID'):
        tensorize(rows)


def test_exact_batch_boundary_and_no_overwrite_scoring(tmp_path):
    base = rows_fixture()[0]
    rows = [{**base, 'id': f'p-{index}'} for index in range(129)]
    rows, model, path, _ = run_synthetic(tmp_path, rows)
    assert model.forward_sizes == [(128, 4), (1, 4)]
    tensors, menus = tensorize(rows)
    with pytest.raises(FileExistsError):
        evaluate(model, rows, tensors, menus, 4, path)
    assert model.forward_sizes == [(128, 4), (1, 4)]


@pytest.mark.parametrize('fault', ['logits', 'hidden', 'value', 'unbounded_value'])
def test_nonfinite_or_unbounded_model_outputs_fail(tmp_path, fault):
    with pytest.raises(ValueError):
        run_synthetic(tmp_path, model=SyntheticPolicy(fault))


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(id='wrong'),
    lambda p: p.update(game_id='wrong'),
    lambda p: p.update(choice='e1e8'),
    lambda p: p.update(correct=not p['correct']),
    lambda p: p.update(target_probability=.999),
    lambda p: p.update(target_nll=-1),
    lambda p: p.update(value=float('nan')),
    lambda p: p.update(target_value=.99),
    lambda p: p.update(entropy=100),
    lambda p: p.update(hidden_rms=-1),
    lambda p: p.update(logit_span=float('inf')),
    lambda p: p.pop('max_probability'),
])
def test_mutated_predictions_rejected(tmp_path, mutation):
    rows, _, path, _ = run_synthetic(tmp_path)
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(predictions[0])
    path.write_text('\n'.join(json.dumps(row) for row in predictions)+'\n')
    with pytest.raises(ValueError):
        audit_predictions(path, rows)


def test_audit_rejects_missing_duplicate_reordered_records(tmp_path):
    rows, _, path, _ = run_synthetic(tmp_path)
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    for changed in (predictions[:-1], predictions+[predictions[0]], list(reversed(predictions))):
        path.write_text('\n'.join(json.dumps(row) for row in changed)+'\n')
        with pytest.raises(ValueError):
            audit_predictions(path, rows)


def test_perfect_predictions_have_no_mismatch_confidence(tmp_path):
    rows = rows_fixture()
    for row in rows:
        row['target_uci'] = encode_candidates(chess.Board(row['fen']))[0][0]
    _, _, _, result = run_synthetic(tmp_path, rows)
    assert result['metrics']['agreement'] == 1
    assert result['metrics']['mismatch_confidence'] is None


def test_probe_has_exact_warmups_and_fixed_positions_no_score_labels():
    rows = rows_fixture()
    model = SyntheticPolicy()
    result = probe(model, rows, 8, [2, 0])
    assert model.choose_boards[:3] == [(chess.STARTING_FEN, 8)]*3
    assert model.choose_boards[3:] == [(rows[2]['fen'], 8), (rows[0]['fen'], 8)]
    assert len(result['warmup_records']) == 3
    assert [row['index'] for row in result['records']] == [2, 0]
    assert result['total_wall_ms'] == math.fsum(row['wall_ms'] for row in result['records'])
    assert result['warmup_wall_ms'] == math.fsum(row['wall_ms'] for row in result['warmup_records'])
    assert result['device'] == 'cpu'
    assert all(set(row) == {'id', 'index', 'choice', 'wall_ms'} for row in result['records'])
    assert model.training


def test_probe_rejects_invalid_indices_and_cpu_violation_without_calls():
    rows = rows_fixture()
    model = SyntheticPolicy()
    for indices in ([], [0, 0], [True], [3]):
        with pytest.raises(ValueError):
            probe(model, rows, 4, indices)
    assert not model.choose_boards
    model.register_buffer('wrong_device', torch.empty(1, device='meta'))
    with pytest.raises(ValueError, match='CPU'):
        probe(model, rows, 4, [0])
    assert not model.choose_boards


@pytest.mark.parametrize('fault', ['illegal', 'probabilities', 'mutated_board'])
def test_probe_rejects_bad_policy_response(fault):
    class BadPolicy(SyntheticPolicy):
        def choose(self, board, depth=None):
            result = super().choose(board, depth)
            if fault == 'illegal':
                result['choice'] = 'e1e8'
            elif fault == 'probabilities':
                result['probabilities'][result['choice']] = float('nan')
            else:
                board.push_uci(result['choice'])
            return result
    with pytest.raises(ValueError):
        probe(BadPolicy(), rows_fixture(), 4, [0])


def test_rejects_bad_source_labels_and_tensor_target_before_model(tmp_path):
    rows = rows_fixture()
    bad = copy.deepcopy(rows)
    bad[0]['target_uci'] = 'e1e8'
    with pytest.raises(ValueError, match='legal'):
        tensorize(bad)
    tensors, menus = tensorize(rows)
    tensors['targets'][0] = 1
    model = SyntheticPolicy()
    with pytest.raises(ValueError, match='Target tensor'):
        evaluate(model, rows, tensors, menus, 4, tmp_path/'bad.jsonl')
    assert not model.forward_sizes


@pytest.mark.parametrize('architecture', ['original', 'residual', 'anchor'])
def test_real_untrained_spatial_model_on_synthetic_board_only(tmp_path, architecture):
    rows = rows_fixture()[:1]
    tensors, menus = tensorize(rows)
    if architecture == 'original':
        model = SpatialChess('recurrent', seed=7)
    else:
        from openjev.research.chess_anchor import AnchorChess
        model = AnchorChess(architecture, seed=7)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        logits, _, _ = model(tensors['observations'], tensors['candidates'], tensors['mask'], depth=2)
        expected = menus[0][int(logits.argmax(-1)[0])]
    path = tmp_path/'untrained.jsonl'
    evaluate(model, rows, tensors, menus, 2, path)
    _, predictions = audit_predictions(path, rows)
    assert predictions[0]['choice'] == expected
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
