"""CPU caches and recorded evaluation for candidate-conditioned chess policies.

The cache contains exact native one-ply consequences, never teacher information
in model inputs. It performs no fitting or engine calls. Cached throughput is
not end-to-end latency; use the unchanged chess_anchor_eval.probe for that.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import chess
import numpy as np
import torch

from . import chess_anchor_eval as anchor_eval
from .chess_candidate import ARMS
from .chess_spatial import ENCODING_VERSION, encode_board, encode_candidates

BATCH_SIZE = 16
ROOT_DEPTH = 4
BRANCH_DEPTH = 2
PERMUTATION_SEED = 11300083
CACHE_VERSION = 'candidate-position-cache-v1'
ROW_FIELDS = ('id', 'game_id', 'fen', 'target_uci', 'target_value')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tensor_metadata(tensor):
    """Hash contiguous CPU storage in bounded views, without a whole-cache copy."""
    raw = memoryview(tensor.numpy()).cast('B')
    digest = hashlib.sha256()
    for start in range(0, len(raw), 1024*1024):
        digest.update(raw[start:start+1024*1024])
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'bytes': len(raw),
            'sha256': digest.hexdigest()}


class CachedPositions:
    """Preallocated CPU tensors, one flat successor row per legal candidate.

Only the five required row fields are copied; optional labels or previous
auxiliary transitions are not retained. Labels have their own buffers and
fingerprints. Returned batches are independent of cache storage. Metadata's
byte total counts tensor storage, not Python objects or temporary/device copies.
"""

    def __init__(self, rows, *, include_successors=True):
        started = time.perf_counter()
        _require(type(include_successors) is bool, 'include_successors must be Boolean')
        rows = list(rows)
        anchor_eval._rows(rows)
        self.rows = tuple({key: row[key] for key in ROW_FIELDS} for row in rows)
        self.include_successors = include_successors
        menus, counts = [], []
        # Count first. No giant Python list of child arrays is ever retained.
        for row in self.rows:
            board = chess.Board(row['fen'])
            names, _ = encode_candidates(board)
            _require(bool(names) and not board.is_game_over(claim_draw=False), 'Cache needs nonterminal roots')
            _require(row['target_uci'] in names, 'Teacher target must be legal')
            menus.append(names)
            counts.append(len(names))
        self.menus = tuple(menus)
        self.counts = torch.tensor(counts, dtype=torch.long)
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.long), self.counts.cumsum(0)))
        total, n = int(self.offsets[-1]), len(self.rows)
        self.observations = torch.empty((n, 19, 8, 8), dtype=torch.float32)
        self.candidates = torch.empty((total, 5), dtype=torch.long)
        self.targets = torch.empty(n, dtype=torch.long)
        self.values = torch.empty(n, dtype=torch.float32)
        self.successors = torch.empty((total, 19, 8, 8), dtype=torch.float32) if include_successors else None
        obs, actions = self.observations.numpy(), self.candidates.numpy()
        children = self.successors.numpy() if include_successors else None
        for index, row in enumerate(self.rows):
            board = chess.Board(row['fen'])
            names, features = encode_candidates(board)
            _require(names == self.menus[index], 'Legal menu changed during cache construction')
            begin, end = int(self.offsets[index]), int(self.offsets[index+1])
            obs[index] = encode_board(board)
            actions[begin:end] = features
            self.targets[index] = names.index(row['target_uci'])
            self.values[index] = row['target_value']
            if children is not None:
                for offset, uci in enumerate(names):
                    child = board.copy(stack=True)
                    child.push_uci(uci)
                    children[begin+offset] = encode_board(child, perspective=board.turn)
        construction_seconds = time.perf_counter()-started
        fingerprint_started = time.perf_counter()
        tensors = {name: _tensor_metadata(getattr(self, name)) for name in
                   ('observations', 'candidates', 'counts', 'offsets', 'targets', 'values')}
        if include_successors:
            tensors['successors'] = _tensor_metadata(self.successors)
        input_rows = [{key: row[key] for key in ('id', 'game_id', 'fen')} for row in self.rows]
        labels = [{key: row[key] for key in ('id', 'target_uci', 'target_value')} for row in self.rows]
        identity = {'version': CACHE_VERSION, 'encoding': ENCODING_VERSION,
                    'input_rows_sha256': _digest(input_rows), 'labels_sha256': _digest(labels),
                    'include_successors': include_successors, 'tensors': tensors}
        self.metadata = {
            **identity, 'cache_sha256': _digest(identity), 'positions': n, 'legal_candidates': total,
            'root_encodings': n, 'native_successors': total if include_successors else 0,
            'native_board_copies': total if include_successors else 0,
            'native_pushes': total if include_successors else 0,
            'successor_encodings': total if include_successors else 0,
            'cpu_tensor_bytes': sum(row['bytes'] for row in tensors.values()),
            'byte_scope': 'Allocated CPU tensor storage only; excludes Python rows/menus, temporary batches and device copies.',
            'native_construction_wall_seconds': construction_seconds,
            'construction_scope': 'Two passes over root FENs: legal menus/counts, allocation, encoding and optional native child copies/pushes. No hashing or model calls.',
            'fingerprinting_wall_seconds': time.perf_counter()-fingerprint_started,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'encoding_source_sha256': hashlib.sha256(Path(encode_board.__code__.co_filename).read_bytes()).hexdigest(),
        }
        self.metadata['total_wall_seconds'] = time.perf_counter()-started

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, arm, device='cpu'):
        """Return model inputs and separate labels in the requested row order.

Repeated/reordered positions are supported. Flat successors follow the valid
entries of this batch's mask, including nonconsecutive or repeated source rows.
No native transitions occur here; metadata counts cache construction separately.
"""
        _require(type(arm) is str and arm in ARMS, 'Unknown candidate arm')
        if isinstance(indices, torch.Tensor):
            _require(indices.device.type == 'cpu' and indices.ndim == 1
                     and indices.dtype in (torch.int32, torch.int64), 'Indices must be a CPU integer vector')
            indices = indices.tolist()
        indices = list(indices)
        _require(bool(indices) and all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
                 and 0 <= i < len(self) for i in indices), 'Invalid cache indices')
        indices = [int(i) for i in indices]
        native = arm in ('delta', 'full_afterstate')
        _require(not native or self.successors is not None, 'Native successors were not cached')
        selected = torch.tensor(indices, dtype=torch.long)
        counts = self.counts[selected]
        width, total = int(counts.max()), int(counts.sum())
        candidates = torch.zeros((len(indices), width, 5), dtype=torch.long)
        mask = torch.zeros((len(indices), width), dtype=torch.bool)
        successors = torch.empty((total, 19, 8, 8), dtype=torch.float32) if native else None
        destination = 0
        for index, source in enumerate(indices):
            begin, end = int(self.offsets[source]), int(self.offsets[source+1])
            count = end-begin
            candidates[index, :count].copy_(self.candidates[begin:end])
            mask[index, :count] = True
            if native:
                successors[destination:destination+count].copy_(self.successors[begin:end])
            destination += count
        inputs = {'observations': self.observations.index_select(0, selected).to(device),
                  'candidates': candidates.to(device), 'legal_mask': mask.to(device)}
        if native:
            inputs['successors'] = successors.to(device)
        return inputs, self.targets.index_select(0, selected).to(device), self.values.index_select(0, selected).to(device)


def prepare_cache(rows, *, include_successors=True):
    return CachedPositions(rows, include_successors=include_successors)


def audit_cache_metadata(metadata, rows, *, include_successors=True):
    """Check source/count/storage receipts without reconstructing float caches.

Small count/offset/label buffers are independently rehashed. Observation,
candidate-feature and successor hashes attest the transient cache created by
the frozen run; this cheap audit checks their receipt binding, not their former
in-memory contents. Full byte reconstruction would require encoding again.
"""
    _require(type(include_successors) is bool, 'include_successors must be Boolean')
    rows = list(rows)
    anchor_eval._rows(rows)
    rows = [{key: row[key] for key in ROW_FIELDS} for row in rows]
    counts, targets = [], []
    for row in rows:
        board = chess.Board(row['fen'])
        _require(board.is_valid() and not board.is_game_over(claim_draw=False), 'Invalid/nonterminal cache source')
        menu = tuple(sorted(move.uci() for move in board.legal_moves))
        _require(row['target_uci'] in menu, 'Cache source has illegal target')
        counts.append(len(menu))
        targets.append(menu.index(row['target_uci']))
    n, total = len(rows), sum(counts)
    identity = {
        'version': CACHE_VERSION, 'encoding': ENCODING_VERSION,
        'input_rows_sha256': _digest([{key: row[key] for key in ('id', 'game_id', 'fen')} for row in rows]),
        'labels_sha256': _digest([{key: row[key] for key in ('id', 'target_uci', 'target_value')} for row in rows]),
        'include_successors': include_successors, 'tensors': metadata['tensors'],
    }
    _require(all(metadata[key] == value for key, value in identity.items())
             and metadata['cache_sha256'] == _digest(identity), 'Cache source or fingerprint mismatch')
    expected = {'observations': ([n, 19, 8, 8], 'torch.float32', 4),
                'candidates': ([total, 5], 'torch.int64', 8),
                'counts': ([n], 'torch.int64', 8), 'offsets': ([n+1], 'torch.int64', 8),
                'targets': ([n], 'torch.int64', 8), 'values': ([n], 'torch.float32', 4)}
    if include_successors:
        expected['successors'] = ([total, 19, 8, 8], 'torch.float32', 4)
    tensors = metadata['tensors']
    _require(set(tensors) == set(expected), 'Cache tensor membership differs')
    for name, (shape, dtype, item_size) in expected.items():
        item = tensors[name]
        _require(set(item) == {'shape', 'dtype', 'bytes', 'sha256'} and item['shape'] == shape
                 and item['dtype'] == dtype and type(item['bytes']) is int
                 and item['bytes'] == math.prod(shape)*item_size
                 and type(item['sha256']) is str and len(item['sha256']) == 64
                 and set(item['sha256']) <= set('0123456789abcdef'), 'Invalid cache tensor receipt')
    counts_tensor = torch.tensor(counts, dtype=torch.long)
    small = {'counts': counts_tensor, 'offsets': torch.cat((torch.zeros(1, dtype=torch.long), counts_tensor.cumsum(0))),
             'targets': torch.tensor(targets, dtype=torch.long),
             'values': torch.tensor([row['target_value'] for row in rows], dtype=torch.float32)}
    _require(all(tensors[name] == _tensor_metadata(value) for name, value in small.items()),
             'Cache index or label tensor checksum differs')
    count_fields = {'positions': n, 'legal_candidates': total, 'root_encodings': n,
                    **{key: total if include_successors else 0 for key in
                       ('native_successors', 'native_board_copies', 'native_pushes', 'successor_encodings')},
                    'cpu_tensor_bytes': sum(item['bytes'] for item in tensors.values())}
    _require(all(type(metadata[key]) is int and metadata[key] == value for key, value in count_fields.items()),
             'Cache construction counts or byte total differ')
    for key in ('native_construction_wall_seconds', 'fingerprinting_wall_seconds', 'total_wall_seconds'):
        _require(type(metadata[key]) in (int, float) and math.isfinite(metadata[key]) and metadata[key] >= 0,
                 'Invalid cache timing')
    _require(metadata['total_wall_seconds'] >= metadata['native_construction_wall_seconds']
             + metadata['fingerprinting_wall_seconds'], 'Cache component timing exceeds total')
    _require(metadata['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
             and metadata['encoding_source_sha256'] == hashlib.sha256(Path(encode_board.__code__.co_filename).read_bytes()).hexdigest(),
             'Cache implementation source differs')
    return {'status': 'verified', 'cache_sha256': metadata['cache_sha256'], **count_fields,
            'scope': 'Source row, legal-count, storage-shape, index/label byte and frozen-construction hash receipts; '
                     'float feature buffers are not regenerated by this audit.'}


def permutation_offset(position_id, legal_count):
    """Candidate i receives child (i+offset) modulo L; L>1 has no fixed point."""
    _require(type(position_id) is str and bool(position_id), 'Invalid permutation position ID')
    _require(type(legal_count) is int and legal_count > 0, 'Invalid legal count')
    if legal_count == 1:
        return 0
    digest = _digest([PERMUTATION_SEED, position_id])
    return 1+int(digest, 16) % (legal_count-1)


def permute_successors(inputs, position_ids):
    """Copy and permute each root's child rows without mixing positions or labels."""
    mask = inputs['legal_mask']
    _require(mask.ndim == 2 and mask.dtype == torch.bool and len(position_ids) == len(mask),
             'Permutation rows do not match legal mask')
    counts = mask.sum(-1).cpu().tolist()
    children = inputs.get('successors')
    _require(children is not None and children.shape == (sum(counts), 19, 8, 8),
             'Permutation requires one successor per legal action')
    changed = torch.empty_like(children)
    offsets, begin = [], 0
    for identity, count in zip(position_ids, counts, strict=True):
        offset = permutation_offset(identity, count)
        changed[begin:begin+count] = children[begin:begin+count].roll(-offset, dims=0)
        offsets.append(offset)
        begin += count
    return {**inputs, 'successors': changed}, offsets


def _diagnostic(predictions, permuted):
    return {'kind': 'permuted_successors' if permuted else 'intact',
            'permutation_seed': PERMUTATION_SEED if permuted else None,
            'positions': len(predictions),
            'permuted_positions': sum(row['successor_permutation_offset'] > 0 for row in predictions),
            'one_legal_move_unchanged': sum(row['one_legal_move_unchanged'] for row in predictions),
            'mapping_sha256': _digest([[row['id'], row['successor_permutation_offset']] for row in predictions])}


@torch.inference_mode()
def evaluate(model, rows_or_cache, outfile, *, permuted=False):
    """Write a final-only CPU panel in batches of 16, with cached inputs timed separately."""
    _require(type(permuted) is bool, 'permuted must be Boolean')
    anchor_eval._cpu(model)
    _require(model.arm in ARMS and model.root_depth == ROOT_DEPTH and model.branch_depth == BRANCH_DEPTH,
             'Evaluation requires the fixed root-depth-4/branch-depth-2 architecture')
    _require(not permuted or model.arm == 'delta', 'Only the delta arm has the permutation diagnostic')
    _require(all(p.dtype == torch.float32 for p in model.parameters()), 'Evaluation requires float32 parameters')
    cache = rows_or_cache if isinstance(rows_or_cache, CachedPositions) else CachedPositions(rows_or_cache)
    _require(model.arm not in ('delta', 'full_afterstate') or cache.successors is not None,
             'Evaluation requires cached native successors')
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    predictions, was_training = [], model.training
    started = time.perf_counter()
    try:
        with outfile.open('x') as stream:
            model.eval()
            for start in range(0, len(cache), BATCH_SIZE):
                end = min(start+BATCH_SIZE, len(cache))
                rows = cache.rows[start:end]
                inputs, targets, _ = cache.batch(range(start, end), model.arm)
                offsets = [0]*len(rows)
                if permuted:
                    inputs, offsets = permute_successors(inputs, [row['id'] for row in rows])
                mask = inputs['legal_mask']
                logits, values, hidden = model(**inputs, depth=ROOT_DEPTH)
                _require(logits.shape == mask.shape and values.shape == (len(rows),), 'Model output shape mismatch')
                _require(hidden.ndim == 4 and hidden.shape[0] == len(rows), 'Hidden shape mismatch')
                _require(torch.isfinite(logits[mask]).all().item() and torch.isfinite(hidden).all().item()
                         and torch.isfinite(values).all().item(), 'Nonfinite model outputs')
                legal_logits = logits.double().masked_fill(~mask, -torch.inf)
                log_probs = legal_logits.log_softmax(-1)
                probabilities = log_probs.exp()
                nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
                entropy = -torch.where(probabilities > 0, probabilities*log_probs, 0).sum(-1)
                rms = hidden.double().square().flatten(1).mean(-1).sqrt()
                spans = legal_logits.max(-1).values-logits.double().masked_fill(~mask, torch.inf).min(-1).values
                choices = legal_logits.argmax(-1)
                for offset, row in enumerate(rows):
                    menu = cache.menus[start+offset]
                    choice = menu[int(choices[offset])]
                    target_nll = float(nll[offset])
                    prediction = {
                        'id': row['id'], 'game_id': row['game_id'], 'target': row['target_uci'],
                        'choice': choice, 'correct': choice == row['target_uci'],
                        'target_nll': target_nll, 'target_probability': math.exp(-target_nll),
                        'value': float(values[offset]), 'target_value': row['target_value'],
                        'max_probability': float(probabilities[offset].max()), 'entropy': float(entropy[offset]),
                        'hidden_rms': float(rms[offset]), 'logit_span': float(spans[offset]),
                        'arm': model.arm, 'evaluation_kind': 'permuted_successors' if permuted else 'intact',
                        'successor_permutation_offset': offsets[offset],
                        'one_legal_move_unchanged': permuted and len(menu) == 1,
                    }
                    anchor_eval._validate_prediction(prediction, legal_count=len(menu))
                    stream.write(json.dumps(prediction, allow_nan=False)+'\n')
                    predictions.append(prediction)
    finally:
        model.train(was_training)
    elapsed = time.perf_counter()-started
    return {'metrics': anchor_eval.summarize(predictions), 'evaluation_wall_seconds': elapsed,
            'diagnostic': _diagnostic(predictions, permuted),
            'timing_scope': 'Cached CPU batch evaluation and prediction writing; native cache construction excluded and recorded separately.',
            'cache_sha256': cache.metadata['cache_sha256']}


def audit_predictions(path, rows, *, permuted=False, expected_arm=None):
    """Return (metrics, predictions), also checking the recorded intervention identity."""
    _require(type(permuted) is bool, 'permuted must be Boolean')
    rows = list(rows)
    metrics, predictions = anchor_eval.audit_predictions(path, rows)
    arm = predictions[0].get('arm') if expected_arm is None else expected_arm
    _require(arm in ARMS and (not permuted or arm == 'delta'), 'Invalid evaluation arm/intervention')
    for pred, row in zip(predictions, rows, strict=True):
        count = chess.Board(row['fen']).legal_moves.count()
        offset = permutation_offset(row['id'], count) if permuted else 0
        _require(pred.get('arm') == arm and pred.get('evaluation_kind') == ('permuted_successors' if permuted else 'intact')
                 and type(pred.get('successor_permutation_offset')) is int
                 and pred['successor_permutation_offset'] == offset
                 and type(pred.get('one_legal_move_unchanged')) is bool
                 and pred['one_legal_move_unchanged'] == (permuted and count == 1),
                 'Prediction intervention metadata mismatch')
    return metrics, predictions


def audit_diagnostic(path, rows):
    """Validate the separate delta-mapping artifact and reproduce its receipt fields."""
    metrics, predictions = audit_predictions(path, rows, permuted=True, expected_arm='delta')
    return {'metrics': metrics, 'diagnostic': _diagnostic(predictions, True)}
