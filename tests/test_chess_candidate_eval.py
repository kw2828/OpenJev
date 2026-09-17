"""Synthetic cache/evaluation invariants; no engine calls or optimizer updates."""

import copy
import json
import math

import chess
import numpy as np
import pytest
import torch
from torch import nn

from openjev.research import chess_anchor_eval as anchor_eval
from openjev.research.chess_candidate import ARMS, CandidateChess, encode_batch
from openjev.research.chess_candidate_eval import (
    BATCH_SIZE,
    PERMUTATION_SEED,
    CachedPositions,
    audit_cache_metadata,
    audit_diagnostic,
    audit_predictions,
    evaluate,
    permutation_offset,
    permute_successors,
    prepare_cache,
)
from openjev.research.chess_spatial import encode_board, encode_candidates


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def rows_fixture():
    fens = [chess.STARTING_FEN, 'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 12 20',
            '7k/8/8/8/3Pp3/8/8/7K b - d3 0 12', '7k/P7/8/8/8/8/8/7K w - - 7 19',
            '7k/6R1/8/5K2/8/8/8/8 b - - 0 1']
    result = []
    for i, fen in enumerate(fens):
        names, _ = encode_candidates(chess.Board(fen))
        result.append({'id': f'synthetic-{i}', 'game_id': i if i % 2 else f'game-{i}', 'fen': fen,
                       'target_uci': names[0], 'target_value': .1*i, 'teacher_note': 'Never model input'})
    assert len(encode_candidates(chess.Board(fens[-1]))[0]) == 1
    return result


def many_rows(n):
    row = rows_fixture()[0]
    return [{**row, 'id': f'position-{i}', 'game_id': i//3} for i in range(n)]


class SyntheticPolicy(nn.Module):
    def __init__(self, arm='delta', fault=None, seed=17):
        super().__init__()
        self.placeholder = nn.Parameter(torch.tensor(0.))
        self.arm, self.root_depth, self.branch_depth, self.seed = arm, 4, 2, seed
        self.fault, self.calls = fault, []

    def forward(self, **inputs):
        assert set(inputs) == {'observations', 'candidates', 'legal_mask', 'depth'} | (
            {'successors'} if self.arm in ('delta', 'full_afterstate') else set())
        assert inputs['depth'] == 4
        self.calls.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
        mask = inputs['legal_mask']
        logits = -torch.arange(mask.shape[1], dtype=torch.float32).expand(len(mask), -1)*.125
        logits = logits.clone().masked_fill(~mask, 1e30)
        values = torch.zeros(len(mask))
        hidden = torch.full((len(mask), 2, 8, 8), 1e30)
        if self.fault == 'logits':
            logits[0, 0] = torch.inf
        elif self.fault == 'hidden':
            hidden[0, 0, 0, 0] = torch.nan
        elif self.fault == 'value':
            values[0] = torch.nan
        elif self.fault == 'unbounded_value':
            values[0] = 1.1
        elif self.fault == 'shape':
            values = values[:, None]
        return logits, values, hidden


def read_predictions(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_preallocated_cache_matches_native_batch_and_records_actual_storage():
    rows = rows_fixture()
    cache = prepare_cache(rows)
    expected, menus = encode_batch([chess.Board(r['fen']) for r in rows], 'delta')
    actual, targets, values = cache.batch(range(len(rows)), 'delta')
    assert cache.menus == menus
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    assert cache.candidates.ndim == 2
    assert cache.successors.shape == (sum(map(len, menus)), 19, 8, 8)
    assert len(cache) == len(rows)
    assert targets.dtype == torch.long and values.dtype == torch.float32
    assert [menus[i][int(targets[i])] for i in range(len(rows))] == [r['target_uci'] for r in rows]
    tensors = [cache.observations, cache.candidates, cache.successors, cache.targets,
               cache.values, cache.counts, cache.offsets]
    meta = cache.metadata
    assert meta['cpu_tensor_bytes'] == sum(t.numel()*t.element_size() for t in tensors)
    assert meta['native_successors'] == meta['native_board_copies'] == meta['native_pushes'] == sum(map(len, menus))
    assert meta['root_encodings'] == len(rows)
    assert meta['native_construction_wall_seconds'] >= 0 and meta['fingerprinting_wall_seconds'] >= 0
    assert meta['total_wall_seconds'] >= meta['native_construction_wall_seconds']
    assert all(t.device.type == 'cpu' and t.is_contiguous() for t in tensors)
    assert all('teacher_note' not in r for r in cache.rows)


@pytest.mark.parametrize('arm', ARMS)
def test_reordered_repeated_and_single_position_batches_keep_flat_successor_alignment(arm):
    rows, selected = rows_fixture(), [3, 1, 3, 4, 0]
    cache = CachedPositions(rows)
    actual, targets, values = cache.batch(np.asarray(selected, dtype=np.int64), arm)
    expected, _ = encode_batch([chess.Board(rows[i]['fen']) for i in selected], arm)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert torch.equal(targets, cache.targets[selected]) and torch.equal(values, cache.values[selected])
    for source in selected:
        one, target, value = cache.batch(torch.tensor([source]), arm)
        other, _ = encode_batch([chess.Board(rows[source]['fen'])], arm)
        assert all(torch.equal(one[k], other[k]) for k in one)
        assert len(target) == len(value) == 1


def test_successors_keep_black_root_perspective_and_all_nineteen_channels():
    rows, cache = rows_fixture(), CachedPositions(rows_fixture())
    for i, row in enumerate(rows):
        board = chess.Board(row['fen'])
        for j, uci in enumerate(cache.menus[i]):
            child = board.copy(stack=True)
            child.push_uci(uci)
            actual = cache.successors[int(cache.offsets[i])+j].numpy()
            np.testing.assert_array_equal(actual, encode_board(child, perspective=board.turn))
            np.testing.assert_array_equal(actual-cache.observations[i].numpy(),
                                          encode_board(child, perspective=board.turn)-encode_board(board))


def test_cache_fingerprints_repeat_and_labels_cannot_change_inputs():
    rows, changed = rows_fixture(), rows_fixture()
    for row in changed:
        row['target_uci'] = encode_candidates(chess.Board(row['fen']))[0][-1]
        row['target_value'] = -.8
        row['teacher_note'] = 'different hidden label'
    left, repeat, right = CachedPositions(rows), CachedPositions(rows), CachedPositions(changed)
    assert left.metadata['cache_sha256'] == repeat.metadata['cache_sha256']
    assert left.metadata['input_rows_sha256'] == right.metadata['input_rows_sha256']
    assert left.metadata['labels_sha256'] != right.metadata['labels_sha256']
    assert left.metadata['cache_sha256'] != right.metadata['cache_sha256']
    a, targets, values = left.batch(range(len(rows)), 'delta')
    b, new_targets, new_values = right.batch(range(len(rows)), 'delta')
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(targets, new_targets) and not torch.equal(values, new_values)
    for key in ('observations', 'candidates', 'counts', 'offsets', 'successors'):
        assert left.metadata['tensors'][key] == right.metadata['tensors'][key]
    rows[0]['target_value'] = 999
    assert left.rows[0]['target_value'] == 0.


@pytest.mark.parametrize('include_successors', [False, True])
def test_cache_receipt_audit_does_not_rebuild_native_children(monkeypatch, include_successors):
    rows = rows_fixture()
    cache = CachedPositions(rows, include_successors=include_successors)
    def forbidden(*_, **__):
        raise AssertionError('Cheap audit must not push or encode native successors')
    monkeypatch.setattr(chess.Board, 'push_uci', forbidden)
    result = audit_cache_metadata(cache.metadata, rows, include_successors=include_successors)
    assert result['status'] == 'verified'
    assert result['cache_sha256'] == cache.metadata['cache_sha256']
    assert result['native_successors'] == (int(cache.counts.sum()) if include_successors else 0)


@pytest.mark.parametrize('fault', ['row', 'labels', 'counts', 'bytes', 'shape', 'feature_hash', 'source', 'timing'])
def test_cache_receipt_corruption_is_rejected(fault):
    rows = rows_fixture()
    cache = CachedPositions(rows)
    metadata = copy.deepcopy(cache.metadata)
    if fault == 'row':
        rows[0]['id'] = 'replaced'
    elif fault == 'labels':
        rows[0]['target_value'] = -.8
    elif fault == 'counts':
        metadata['native_pushes'] -= 1
    elif fault == 'bytes':
        metadata['cpu_tensor_bytes'] -= 4
    elif fault == 'shape':
        metadata['tensors']['successors']['shape'][0] -= 1
    elif fault == 'feature_hash':
        metadata['tensors']['successors']['sha256'] = 'a'*64
    elif fault == 'source':
        metadata['source_sha256'] = '0'*64
    else:
        metadata['total_wall_seconds'] = -1
    with pytest.raises(ValueError):
        audit_cache_metadata(metadata, rows)


def test_batch_mutation_never_changes_cached_buffers():
    cache = CachedPositions(rows_fixture())
    before = {k: getattr(cache, k).clone() for k in ('observations', 'candidates', 'successors', 'targets', 'values')}
    batch, targets, values = cache.batch([0], 'delta')
    for value in batch.values():
        value.zero_()
    targets.fill_(9)
    values.fill_(.9)
    assert all(torch.equal(getattr(cache, k), v) for k, v in before.items())


def test_skip_children_never_pushes_and_native_arm_is_rejected(monkeypatch):
    def fail(*_, **__):
        raise AssertionError('Unexpected native push')
    monkeypatch.setattr(chess.Board, 'push_uci', fail)
    cache = CachedPositions(rows_fixture(), include_successors=False)
    assert cache.successors is None and cache.metadata['native_pushes'] == 0
    assert 'successors' not in cache.metadata['tensors']
    for arm in ('direct', 'action_only'):
        batch, _, _ = cache.batch([0], arm)
        assert set(batch) == {'observations', 'candidates', 'legal_mask'}
    for arm in ('delta', 'full_afterstate'):
        with pytest.raises(ValueError, match='not cached'):
            cache.batch([0], arm)


@pytest.mark.parametrize('indices', [[], [True], [1.0], [-1], [5], torch.tensor([True]), torch.tensor([[0]])])
def test_invalid_indices_rejected(indices):
    cache = CachedPositions(rows_fixture(), include_successors=False)
    with pytest.raises(ValueError):
        cache.batch(indices, 'direct')


@pytest.mark.parametrize('fault', ['duplicate', 'illegal', 'nan', 'terminal', 'invalid_board'])
def test_invalid_source_rows_rejected(fault):
    rows = rows_fixture()
    if fault == 'duplicate':
        rows[1]['id'] = rows[0]['id']
    elif fault == 'illegal':
        rows[0]['target_uci'] = 'e2e5'
    elif fault == 'nan':
        rows[0]['target_value'] = math.nan
    elif fault == 'terminal':
        rows[0]['fen'] = 'k7/2Q5/2K5/8/8/8/8/8 b - - 0 1'
    else:
        rows[0]['fen'] = '8/8/8/8/8/8/8/8 w - - 0 1'
    with pytest.raises(ValueError):
        CachedPositions(rows)


def test_permutation_is_local_nonidentity_and_independent_of_rng_and_batch_order():
    cache = CachedPositions(rows_fixture())
    indices = [4, 1, 0]
    batch, _, _ = cache.batch(indices, 'delta')
    original = batch['successors'].clone()
    ids = [cache.rows[i]['id'] for i in indices]
    before = torch.random.get_rng_state().clone()
    changed, offsets = permute_successors(batch, ids)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(batch['successors'], original)
    begin = 0
    for source, offset in zip(indices, offsets, strict=True):
        count = len(cache.menus[source])
        assert offset == permutation_offset(cache.rows[source]['id'], count)
        assert (offset == 0) if count == 1 else (1 <= offset < count)
        assert torch.equal(changed['successors'][begin:begin+count], original[begin:begin+count].roll(-offset, 0))
        one, _, _ = cache.batch([source], 'delta')
        alone, only = permute_successors(one, [cache.rows[source]['id']])
        assert only == [offset]
        assert torch.equal(alone['successors'], changed['successors'][begin:begin+count])
        begin += count
    assert offsets[0] == 0
    assert changed['observations'] is batch['observations']
    assert changed['candidates'] is batch['candidates']


@pytest.mark.parametrize('arm', ARMS)
def test_evaluation_batches_all_rows_and_anchor_auditor_accepts_receipt(tmp_path, arm):
    rows = many_rows(2*BATCH_SIZE+1)
    cache, model = CachedPositions(rows), SyntheticPolicy(arm)
    model.train()
    path = tmp_path/'predictions.jsonl'
    result = evaluate(model, cache, path)
    assert [len(call['observations']) for call in model.calls] == [16, 16, 1]
    assert model.training
    metrics, predictions = anchor_eval.audit_predictions(path, rows)
    assert result['metrics'] == metrics == audit_predictions(path, rows, expected_arm=arm)[0]
    assert result['metrics']['hidden_rms'] == pytest.approx(1e30, rel=1e-7)
    assert all(math.isfinite(v) for k, v in metrics.items() if v is not None)
    assert result['diagnostic']['permuted_positions'] == 0
    assert result['diagnostic']['permutation_seed'] is None
    assert result['cache_sha256'] == cache.metadata['cache_sha256']
    assert len(predictions) == len(rows)
    with pytest.raises(FileExistsError):
        evaluate(model, cache, path)


def test_real_untrained_direct_matches_original_evaluation_and_root_value_once(tmp_path):
    rows, cache = rows_fixture(), CachedPositions(rows_fixture())
    model = CandidateChess('direct', seed=17, width=4)
    tensors, menus = anchor_eval.tensorize(rows)
    prior = anchor_eval.evaluate(model, rows, tensors, menus, 4, tmp_path/'old.jsonl')
    calls = []
    hook = model.value_head.register_forward_hook(lambda _, args, output: calls.append(output.shape))
    result = evaluate(model, cache, tmp_path/'new.jsonl')
    hook.remove()
    assert result['metrics'] == prior['metrics']
    assert calls == [torch.Size([len(rows), 1])]
    for old, new in zip(read_predictions(tmp_path/'old.jsonl'), read_predictions(tmp_path/'new.jsonl'), strict=True):
        assert all(new[k] == v for k, v in old.items())


def test_permutation_receipts_repeat_across_fit_seeds_and_do_not_change_root_value(tmp_path):
    rows, cache = rows_fixture(), CachedPositions(rows_fixture())
    diagnostics = []
    for seed in (17, 29):
        model = CandidateChess('delta', seed=seed, width=4)
        normal, shifted = tmp_path/f'{seed}-normal.jsonl', tmp_path/f'{seed}-permuted.jsonl'
        evaluate(model, cache, normal)
        result = evaluate(model, cache, shifted, permuted=True)
        diagnostics.append(result['diagnostic'])
        metrics, preds = audit_predictions(shifted, rows, permuted=True, expected_arm='delta')
        assert result['metrics'] == metrics
        assert audit_diagnostic(shifted, rows) == {'metrics': metrics, 'diagnostic': result['diagnostic']}
        assert result['diagnostic']['permutation_seed'] == PERMUTATION_SEED
        assert result['diagnostic']['one_legal_move_unchanged'] == 1
        assert result['diagnostic']['permuted_positions'] == len(rows)-1
        for left, right in zip(read_predictions(normal), preds, strict=True):
            assert left['value'] == right['value'] and left['hidden_rms'] == right['hidden_rms']
    assert diagnostics[0] == diagnostics[1]


@pytest.mark.parametrize('fault', ['logits', 'hidden', 'value', 'unbounded_value', 'shape'])
def test_invalid_model_output_raises_and_restores_training_state(tmp_path, fault):
    model = SyntheticPolicy(fault=fault)
    model.train()
    with pytest.raises(ValueError):
        evaluate(model, rows_fixture(), tmp_path/'bad.jsonl')
    assert model.training


@pytest.mark.parametrize('field,value', [('successor_permutation_offset', 0), ('one_legal_move_unchanged', True),
                                       ('evaluation_kind', 'intact'), ('arm', 'action_only'),
                                       ('choice', 'a1a8'), ('target_probability', .9)])
def test_saved_prediction_corruption_is_detected(tmp_path, field, value):
    rows, path = rows_fixture(), tmp_path/'p.jsonl'
    evaluate(SyntheticPolicy(), rows, path, permuted=True)
    predictions = read_predictions(path)
    predictions[0][field] = value
    path.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    with pytest.raises(ValueError):
        audit_predictions(path, rows, permuted=True, expected_arm='delta')


def test_permuted_artifact_cannot_be_audited_as_primary_or_other_arm(tmp_path):
    rows, path = rows_fixture(), tmp_path/'p.jsonl'
    evaluate(SyntheticPolicy(), rows, path, permuted=True)
    with pytest.raises(ValueError, match='intervention'):
        audit_predictions(path, rows)
    with pytest.raises(ValueError):
        evaluate(SyntheticPolicy('action_only'), rows, tmp_path/'forbidden.jsonl', permuted=True)
    assert not (tmp_path/'forbidden.jsonl').exists()


def test_changed_teacher_labels_never_enter_evaluation_model_inputs(tmp_path):
    rows, changed = rows_fixture(), copy.deepcopy(rows_fixture())
    for row in changed:
        row['target_uci'] = encode_candidates(chess.Board(row['fen']))[0][-1]
        row['target_value'] = -.9
    left, right = SyntheticPolicy(), SyntheticPolicy()
    evaluate(left, rows, tmp_path/'a.jsonl', permuted=True)
    evaluate(right, changed, tmp_path/'b.jsonl', permuted=True)
    for a, b in zip(left.calls, right.calls, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) else a[key] == b[key]
