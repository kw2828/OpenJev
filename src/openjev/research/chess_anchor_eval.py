"""CPU evaluation and saved-prediction auditing for the anchor chess pilot.

Teacher targets are separate tensors and never enter the model call. This module
does not train, select checkpoints, call an engine, or choose latency positions.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import chess
import numpy as np
import torch

from .chess_spatial import encode_board, encode_candidates

BATCH_SIZE = 128
WARMUPS = 3
FIELDS = {
    'id', 'game_id', 'target', 'choice', 'correct', 'target_nll',
    'target_probability', 'value', 'target_value', 'max_probability', 'entropy',
    'hidden_rms', 'logit_span',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), f'Nonfinite/invalid {name}')
    return value


def _game_id(value):
    return (type(value) is int and value >= 0) or (type(value) is str and bool(value))


def _rows(rows):
    _require(bool(rows), 'Evaluation rows must not be empty')
    seen = set()
    for row in rows:
        _require(isinstance(row['id'], str) and bool(row['id']), 'Invalid position ID')
        _require(row['id'] not in seen, 'Duplicate position ID')
        seen.add(row['id'])
        _require(_game_id(row['game_id']), 'Invalid game ID')
        _require(-1 <= _number(row['target_value'], 'target_value') <= 1, 'Target value outside [-1,1]')


def tensorize(rows):
    """Return CPU tensor dictionary and sorted legal UCI menus in row order."""
    _rows(rows)
    observations, features, menus, targets = [], [], [], []
    for row in rows:
        board = chess.Board(row['fen'])
        _require(board.is_valid(), 'Invalid chess board')
        names, candidates = encode_candidates(board)
        _require(bool(names), 'Empty legal-move menu')
        _require(row['target_uci'] in names, 'Teacher target is not legal')
        observations.append(encode_board(board))
        features.append(candidates)
        menus.append(names)
        targets.append(names.index(row['target_uci']))
    max_legal = max(map(len, menus))
    candidates = np.zeros((len(rows), max_legal, 5), dtype=np.int64)
    mask = np.zeros((len(rows), max_legal), dtype=bool)
    for index, feature in enumerate(features):
        candidates[index, :len(feature)] = feature
        mask[index, :len(feature)] = True
    arrays = {
        'observations': np.stack(observations), 'candidates': candidates, 'mask': mask,
        'targets': np.asarray(targets, dtype=np.int64),
        'values': np.asarray([row['target_value'] for row in rows], dtype=np.float32),
    }
    return {name: torch.from_numpy(array) for name, array in arrays.items()}, menus


def _cpu(model):
    _require(
        all(tensor.device.type == 'cpu' for tensor in list(model.parameters()) + list(model.buffers())),
        'Evaluation requires a CPU model',
    )


def _validate_prediction(prediction, *, legal_count=None):
    _require(FIELDS <= prediction.keys(), 'Missing prediction fields')
    for name in ('id', 'target', 'choice'):
        _require(isinstance(prediction[name], str) and bool(prediction[name]), f'Invalid {name}')
    _require(_game_id(prediction['game_id']), 'Invalid game ID')
    _require(type(prediction['correct']) is bool, 'Correctness must be Boolean')
    for name in ('target_nll', 'target_probability', 'value', 'target_value',
                 'max_probability', 'entropy', 'hidden_rms', 'logit_span'):
        _number(prediction[name], name)
    _require(prediction['correct'] == (prediction['choice'] == prediction['target']), 'Incorrect correctness flag')
    _require(prediction['target_nll'] >= 0, 'Negative target NLL')
    _require(0 <= prediction['target_probability'] <= 1, 'Target probability outside [0,1]')
    _require(0 < prediction['max_probability'] <= 1, 'Max probability outside (0,1]')
    _require(
        math.isclose(prediction['target_probability'], math.exp(-prediction['target_nll']),
                     rel_tol=1e-10, abs_tol=1e-14),
        'Target probability and NLL disagree',
    )
    _require(prediction['target_probability'] <= prediction['max_probability'] + 1e-12,
             'Target probability exceeds maximum')
    if prediction['correct']:
        _require(math.isclose(prediction['target_probability'], prediction['max_probability'],
                              rel_tol=1e-10, abs_tol=1e-14), 'Correct choice is not maximum probability')
    for name in ('value', 'target_value'):
        _require(-1 <= prediction[name] <= 1, f'{name} outside [-1,1]')
    _require(prediction['entropy'] >= 0 and prediction['hidden_rms'] >= 0
             and prediction['logit_span'] >= 0, 'Negative entropy/RMS/logit span')
    _require(prediction['entropy'] + 1e-10 >= -math.log(prediction['max_probability']),
             'Entropy below maximum-probability bound')
    if legal_count is not None:
        _require(prediction['max_probability'] + 1e-12 >= 1 / legal_count,
                 'Maximum probability below uniform')
        _require(prediction['entropy'] <= math.log(legal_count) + 1e-10,
                 'Entropy exceeds legal-menu bound')


def summarize(predictions):
    """Aggregate recorded scalar diagnostics; mismatch confidence is None if empty."""
    _require(bool(predictions), 'Predictions must not be empty')
    ids = []
    for row in predictions:
        _validate_prediction(row)
        ids.append(row['id'])
    _require(len(set(ids)) == len(ids), 'Duplicate prediction IDs')
    count = len(predictions)
    mean = lambda values: math.fsum(values) / count
    wrong = [row['max_probability'] for row in predictions if not row['correct']]
    correct = sum(row['correct'] for row in predictions)
    result = {
        'examples': count, 'correct': correct, 'agreement': correct / count,
        'target_nll': mean(row['target_nll'] for row in predictions),
        'value_mae': mean(abs(row['value'] - row['target_value']) for row in predictions),
        'mean_confidence': mean(row['max_probability'] for row in predictions),
        'mismatch_confidence': math.fsum(wrong) / len(wrong) if wrong else None,
        'entropy': mean(row['entropy'] for row in predictions),
        'hidden_rms': mean(row['hidden_rms'] for row in predictions),
        'logit_span': mean(row['logit_span'] for row in predictions),
    }
    for name, value in result.items():
        if value is not None:
            _number(value, name)
    return result


def _inputs(rows, tensors, menus):
    _rows(rows)
    count = len(rows)
    _require(len(menus) == count, 'Menu count mismatch')
    _require(set(tensors) == {'observations', 'candidates', 'mask', 'targets', 'values'},
             'Unexpected tensor fields')
    _require(all(value.device.type == 'cpu' and len(value) == count for value in tensors.values()),
             'Tensors must be row-aligned CPU tensors')
    _require(tensors['observations'].shape == (count, 19, 8, 8), 'Observation shape mismatch')
    _require(tensors['candidates'].ndim == 3 and tensors['candidates'].shape[-1] == 5,
             'Candidate shape mismatch')
    _require(tensors['mask'].dtype == torch.bool and tensors['mask'].shape == tensors['candidates'].shape[:2],
             'Mask shape/dtype mismatch')
    _require(tensors['targets'].dtype == torch.long and tensors['targets'].shape == (count,),
             'Target shape/dtype mismatch')
    _require(tensors['values'].shape == (count,), 'Value shape mismatch')
    _require(torch.isfinite(tensors['observations']).all().item(), 'Nonfinite observations')
    for index, (row, menu) in enumerate(zip(rows, menus, strict=True)):
        names = tuple(sorted(move.uci() for move in chess.Board(row['fen']).legal_moves))
        _require(tuple(menu) == names and bool(names), 'Legal menu differs from board')
        target = int(tensors['targets'][index])
        _require(0 <= target < len(menu) and menu[target] == row['target_uci'], 'Target tensor mismatch')
        expected_mask = torch.arange(tensors['mask'].shape[1]) < len(menu)
        _require(torch.equal(tensors['mask'][index], expected_mask), 'Legal mask mismatch')
        expected_value = torch.tensor(row['target_value'], dtype=tensors['values'].dtype)
        _require(tensors['values'][index].item() == expected_value.item(), 'Value tensor mismatch')


@torch.inference_mode()
def evaluate(model, rows, tensors, menus, depth, outfile: Path):
    """Evaluate in batches of 128 and write a new, complete prediction JSONL."""
    _cpu(model)
    _inputs(rows, tensors, menus)
    _require(type(depth) is int and depth > 0, 'Depth must be a positive integer')
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    was_training = model.training
    started = time.perf_counter()
    try:
        with outfile.open('x') as stream:
            model.eval()
            for start in range(0, len(rows), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(rows))
                mask = tensors['mask'][start:end]
                logits, values, hidden = model(
                    tensors['observations'][start:end], tensors['candidates'][start:end], mask, depth=depth,
                )
                _require(logits.shape == mask.shape and values.shape == (end-start,), 'Model output shape mismatch')
                _require(hidden.ndim == 4 and hidden.shape[0] == end-start, 'Hidden shape mismatch')
                _require(torch.isfinite(logits[mask]).all().item(), 'Nonfinite legal logits')
                _require(torch.isfinite(values).all().item() and torch.isfinite(hidden).all().item(),
                         'Nonfinite value/hidden output')
                log_probs = logits.double().masked_fill(~mask, -torch.inf).log_softmax(-1)
                probs = log_probs.exp()
                targets = tensors['targets'][start:end]
                nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                entropies = -torch.where(probs > 0, probs * log_probs, 0).sum(-1)
                # Cast before squaring: large finite float32 activations must not overflow float32.
                rms = hidden.double().square().flatten(1).mean(-1).sqrt()
                legal_logits = logits.double().masked_fill(~mask, -torch.inf)
                spans = legal_logits.max(-1).values - logits.double().masked_fill(~mask, torch.inf).min(-1).values
                choices = legal_logits.argmax(-1)
                for offset, row in enumerate(rows[start:end]):
                    choice = menus[start+offset][int(choices[offset])]
                    target_nll = float(nll[offset])
                    prediction = {
                        'id': row['id'], 'game_id': row['game_id'], 'target': row['target_uci'],
                        'choice': choice, 'correct': choice == row['target_uci'],
                        'target_nll': target_nll, 'target_probability': math.exp(-target_nll),
                        'value': float(values[offset]), 'target_value': row['target_value'],
                        'max_probability': float(probs[offset].max()), 'entropy': float(entropies[offset]),
                        'hidden_rms': float(rms[offset]), 'logit_span': float(spans[offset]),
                    }
                    _validate_prediction(prediction, legal_count=len(menus[start+offset]))
                    stream.write(json.dumps(prediction, allow_nan=False)+'\n')
                    predictions.append(prediction)
            stream.flush()
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started
    return {'metrics': summarize(predictions), 'evaluation_wall_seconds': elapsed}


def audit_predictions(path, rows):
    """Validate row identity, legality and recorded arithmetic, without model calls.

Scalar records cannot prove hidden activations or full probability distributions;
their authenticity additionally depends on the caller's frozen execution hashes.
"""
    _rows(rows)
    with Path(path).open() as stream:
        predictions = [json.loads(line) for line in stream]
    _require(len(predictions) == len(rows), 'Incomplete/excess prediction rows')
    for prediction, row in zip(predictions, rows, strict=True):
        _require(prediction.get('id') == row['id'] and prediction.get('game_id') == row['game_id']
                 and type(prediction.get('game_id')) is type(row['game_id']),
                 'Prediction position/game identity mismatch')
        _require(prediction.get('target') == row['target_uci']
                 and prediction.get('target_value') == row['target_value'], 'Prediction label mismatch')
        legal = {move.uci() for move in chess.Board(row['fen']).legal_moves}
        _require(prediction.get('choice') in legal and prediction.get('target') in legal,
                 'Illegal prediction choice/target')
        _validate_prediction(prediction, legal_count=len(legal))
    return summarize(predictions), predictions


@torch.inference_mode()
def probe(model, rows, depth, indices):
    """Time explicit CPU decisions, with exactly three separately counted warmups."""
    _cpu(model)
    _rows(rows)
    _require(type(depth) is int and depth > 0, 'Depth must be a positive integer')
    indices = list(indices)
    _require(bool(indices) and len(set(indices)) == len(indices), 'Latency indices must be nonempty and unique')
    _require(all(type(index) is int and 0 <= index < len(rows) for index in indices),
             'Invalid latency index')

    def call(fen, identity, index):
        start = time.perf_counter()
        board = chess.Board(fen)
        original_fen = board.fen(en_passant='fen')
        legal = {move.uci() for move in board.legal_moves}
        result = model.choose(board, depth=depth)
        _require(isinstance(result, dict) and isinstance(result.get('choice'), str), 'Malformed policy response')
        _require(board.fen(en_passant='fen') == original_fen, 'Policy mutated latency-probe board')
        _require(result['choice'] in legal, 'Illegal latency-probe choice')
        if 'probabilities' in result:
            probabilities = result['probabilities']
            _require(isinstance(probabilities, dict) and set(probabilities) == legal, 'Incomplete legal probabilities')
            for probability in probabilities.values():
                _require(0 <= _number(probability, 'probability') <= 1, 'Invalid probability')
            _require(math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-6), 'Probabilities do not sum to one')
        wall_ms = (time.perf_counter() - start) * 1000
        _require(_number(wall_ms, 'wall_ms') >= 0, 'Negative latency')
        return {'id': identity, 'index': index, 'choice': result['choice'], 'wall_ms': wall_ms}

    was_training = model.training
    try:
        model.eval()
        warmups = [call(chess.STARTING_FEN, 'starting-board', index) for index in range(WARMUPS)]
        records = [call(rows[index]['fen'], rows[index]['id'], index) for index in indices]
    finally:
        model.train(was_training)
    return {
        'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'depth': depth,
        'timing_scope': 'Full single-position choose call, including board construction, encoding and legal-response validation',
        'warmup_policy': 'Three standard-starting-board calls before the fixed selected positions',
        'warmup_records': warmups, 'records': records,
        'warmup_wall_ms': math.fsum(row['wall_ms'] for row in warmups),
        'total_wall_ms': math.fsum(row['wall_ms'] for row in records),
    }
