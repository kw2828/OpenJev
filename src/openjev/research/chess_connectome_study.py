"""Prospective controlled connectome fits, authenticated local checkpoints and evaluation.

This module does not select models, tune on evaluation results or estimate Elo.
Seeded schedules do not imply bitwise deterministic MPS execution. Graph-derived
checkpoints are restricted to local runs and retain the source graph's terms.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch
from torch.nn import functional as F

from . import chess_anchor_eval as anchor_eval
from .chess_candidate import CandidateChess
from .chess_candidate_eval import CachedPositions, _tensor_metadata, audit_cache_metadata
from .chess_connectome_adapter import ConnectomeChessAdapter
from .connectome_graph import graph_sha256

ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT/'runs'
VERSION = 'openjev-connectome-study-v1'
SEEDS = (97, 109, 127)
VARIANTS = ('biological', 'rewire151', 'rewire163', 'rewire179', 'dense', 'node_local')
CONFIG_FIELDS = {'name', 'variant', 'mode', 'seed', 'mapping_seed', 'backbone_seed'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _hash(value):
    _require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             'Expected a lowercase SHA-256 identity')
    return value


def _read(path):
    return json.loads(Path(path).read_text())


def _rows_file(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _local(path):
    path = Path(path)
    _require(path.resolve().is_relative_to(RUNS_ROOT.resolve()),
             'Graph-derived checkpoint/training artifacts must remain under local runs/')
    return path


def _state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        _require(type(name) is str and isinstance(tensor, torch.Tensor), 'Invalid tensor state')
        value = tensor.detach().cpu().contiguous()
        _require(value.dtype in (torch.float32, torch.int64), 'Checkpoint tensors must be float32 or int64')
        _require(not value.is_floating_point() or torch.isfinite(value).all(), 'Nonfinite checkpoint tensor')
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(model):
    return _state_sha(model.state_dict())


def configurations():
    return [
        {'name': f'{variant}-{seed}', 'variant': variant,
         'mode': variant if variant in ('dense', 'node_local') else 'sparse',
         'seed': seed, 'mapping_seed': seed, 'backbone_seed': seed}
        for variant in VARIANTS for seed in SEEDS
    ]


def baseline_configurations():
    return [{'name': f'direct-{seed}', 'variant': 'direct', 'mode': 'direct',
             'seed': seed, 'mapping_seed': seed, 'backbone_seed': seed} for seed in SEEDS]


def _configuration(config, *, baseline=True):
    _require(type(config) is dict and set(config) == CONFIG_FIELDS, 'Configuration fields differ')
    allowed = configurations()+(baseline_configurations() if baseline else [])
    _require(config in allowed and all(type(config[key]) is int for key in
                                     ('seed', 'mapping_seed', 'backbone_seed')), 'Unknown study configuration')
    return dict(config)


def make_model(config, backbone, graph):
    config = _configuration(config, baseline=False)
    _require(isinstance(backbone, CandidateChess) and backbone.seed == config['backbone_seed'],
             'Backbone seed/architecture differs')
    return ConnectomeChessAdapter(backbone, graph, mode=config['mode'], seed=config['seed'],
                                 mapping_seed=config['mapping_seed'], slot_channels=16, steps=4, alpha=.5)


def _identity(model, config):
    config = _configuration(config)
    if isinstance(model, ConnectomeChessAdapter):
        _require(config['variant'] != 'direct' and model.mode == config['mode']
                 and model.seed == config['seed'] and model.mapping_seed == config['mapping_seed']
                 and model.backbone.seed == config['backbone_seed'] and model.steps == 4
                 and model.slot_channels == 16 and model.alpha == .5, 'Adapter configuration differs')
        architecture = model.config
    else:
        _require(isinstance(model, CandidateChess) and config['variant'] == 'direct'
                 and model.arm == 'direct' and model.seed == config['seed']
                 and model.root_depth == 4 and model.recurrence == 'residual', 'Direct baseline differs')
        architecture = {'kind': 'unchanged-direct', 'seed': model.seed, 'width': model.width, 'root_depth': 4}
    return {'configuration': config, 'architecture': architecture, 'state_sha256': state_sha256(model)}


def save_checkpoint(model, path, *, plan_sha256, configuration, backbone_sha256):
    path = _local(path)
    _hash(plan_sha256)
    _hash(backbone_sha256)
    _require(isinstance(model, ConnectomeChessAdapter), 'Expected an adapter checkpoint')
    identity = _identity(model, configuration)
    _require(state_sha256(model.backbone) == backbone_sha256, 'Frozen backbone identity differs')
    payload = {
        'version': VERSION, 'plan_sha256': plan_sha256,
        'configuration': identity['configuration'], 'architecture': identity['architecture'],
        'graph_sha256': model.config['source_graph_sha256'],
        'backbone_sha256': backbone_sha256, 'state_sha256': identity['state_sha256'],
        'source_sha256': sha256(__file__),
        'state_dict': {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
    }
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return {'checkpoint_sha256': sha256(path), 'state_sha256': identity['state_sha256']}


def load_checkpoint(path, backbone, graph, *, expected_plan_sha256, expected_config,
                    expected_backbone_sha256, expected_checkpoint_sha256):
    path = _local(path)
    for value in (expected_plan_sha256, expected_backbone_sha256, expected_checkpoint_sha256):
        _hash(value)
    _require(sha256(path) == expected_checkpoint_sha256, 'Checkpoint file hash differs')
    _require(state_sha256(backbone) == expected_backbone_sha256, 'Supplied backbone identity differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    keys = {'version', 'plan_sha256', 'configuration', 'architecture', 'graph_sha256',
            'backbone_sha256', 'state_sha256', 'source_sha256', 'state_dict'}
    _require(type(payload) is dict and set(payload) == keys and payload['version'] == VERSION,
             'Checkpoint format differs')
    model = make_model(expected_config, backbone, graph).cpu()
    _require(payload['configuration'] == _configuration(expected_config, baseline=False)
             and payload['plan_sha256'] == expected_plan_sha256
             and payload['backbone_sha256'] == expected_backbone_sha256
             and payload['graph_sha256'] == graph_sha256(graph)
             and payload['architecture'] == model.config
             and payload['source_sha256'] == sha256(__file__), 'Checkpoint metadata binding differs')
    expected = model.state_dict()
    state = payload['state_dict']
    _require(type(state) is dict and set(state) == set(expected), 'Checkpoint tensor membership differs')
    buffers = dict(model.named_buffers())
    for name, tensor in state.items():
        _require(isinstance(tensor, torch.Tensor) and tensor.shape == expected[name].shape
                 and tensor.dtype == expected[name].dtype, 'Checkpoint tensor shape/dtype differs')
        if name.startswith('backbone.') or name in buffers:
            _require(torch.equal(tensor, expected[name]), 'Checkpoint frozen backbone or graph/mapping buffer differs')
    _require(_state_sha(state) == payload['state_sha256'], 'Checkpoint tensor hash differs')
    model.load_state_dict(state, strict=True)
    _require(state_sha256(model.backbone) == expected_backbone_sha256, 'Loaded backbone changed')
    return model.eval()


@dataclass(frozen=True)
class TrainingSettings:
    train_examples: int = 32768
    epochs: int = 6
    batch_size: int = 128
    device: str = 'mps'
    learning_rate: float = .001
    gradient_clip: float = 1.0
    batch_order_seed_base: int = 11500029
    torch_threads: int = 2
    deterministic_algorithms: bool = False

    def __post_init__(self):
        for key in ('train_examples', 'epochs', 'batch_size', 'torch_threads'):
            _require(type(getattr(self, key)) is int and getattr(self, key) > 0, f'Invalid {key}')
        _require(self.train_examples % self.batch_size == 0, 'Training size must be divisible by batch size')
        _require(self.device in ('cpu', 'mps') and type(self.device) is str, 'Unsupported training device')
        _require(type(self.batch_order_seed_base) is int and self.batch_order_seed_base >= 0,
                 'Invalid batch-order seed base')
        for key in ('learning_rate', 'gradient_clip'):
            value = getattr(self, key)
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0, f'Invalid {key}')
        _require(type(self.deterministic_algorithms) is bool, 'Determinism flag must be Boolean')

    @property
    def updates(self):
        return self.epochs*(self.train_examples//self.batch_size)


DEFAULT_TRAINING = TrainingSettings()


def schedule(seed, settings=DEFAULT_TRAINING):
    _require(type(seed) is int and seed >= 0 and isinstance(settings, TrainingSettings), 'Invalid schedule inputs')
    generator = torch.Generator(device='cpu').manual_seed(settings.batch_order_seed_base+seed)
    result = []
    for epoch in range(settings.epochs):
        indices = torch.randperm(settings.train_examples, generator=generator).tolist()
        for start in range(0, len(indices), settings.batch_size):
            result.append({'step': len(result)+1, 'epoch': epoch+1,
                           'indices': indices[start:start+settings.batch_size]})
    _require(len(result) == settings.updates, 'Schedule budget differs')
    return result


def _cache(rows_or_cache):
    cache = rows_or_cache if isinstance(rows_or_cache, CachedPositions) else CachedPositions(
        rows_or_cache, include_successors=False)
    _require(not cache.include_successors, 'Connectome study never consumes native successor inputs')
    audit_cache_metadata(cache.metadata, cache.rows, include_successors=False)
    for name, metadata in cache.metadata['tensors'].items():
        _require(_tensor_metadata(getattr(cache, name)) == metadata, 'Cached tensor bytes changed')
    return cache


def _sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def _frozen(model, expected_backbone, expected_buffers, *, hash_backbone=False):
    _require(not model.backbone.training and all(not p.requires_grad and p.grad is None
                                                for p in model.backbone.parameters()), 'Backbone is not frozen')
    if hash_backbone:
        _require(state_sha256(model.backbone) == expected_backbone, 'Frozen backbone tensor hash changed')
        _require(_state_sha(dict(model.named_buffers())) == expected_buffers, 'Fixed graph/mapping buffers changed')


def _add_counts(total, counts):
    for key, value in counts.items():
        if type(value) is int and key not in ('batch_size', 'candidate_slots_per_position'):
            total[key] = total.get(key, 0)+value


def fit(config, backbone, graph, rows_or_cache, out, *, plan_sha256, data_sha256,
        settings=DEFAULT_TRAINING):
    """Fit one final adapter, with no validation, restart or checkpoint selection."""
    config = _configuration(config, baseline=False)
    _hash(plan_sha256)
    _hash(data_sha256)
    _require(isinstance(settings, TrainingSettings), 'Expected explicit TrainingSettings')
    out = _local(out)
    out.mkdir(parents=False, exist_ok=False)
    _write(out/'started.json', {'status': 'started', 'configuration': config,
                               'plan_sha256': plan_sha256, 'settings': asdict(settings)})
    updates = 0
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        if settings.device == 'mps':
            _require(torch.backends.mps.is_available(), 'MPS unavailable; no fallback permitted')
            _require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') in (None, '0'), 'MPS fallback must be disabled')
        torch.set_num_threads(settings.torch_threads)
        torch.use_deterministic_algorithms(settings.deterministic_algorithms)
        cache = _cache(rows_or_cache)
        _require(len(cache) == settings.train_examples, 'Training row count differs')
        model = make_model(config, backbone, graph)
        initial = state_sha256(model)
        frozen_backbone = state_sha256(model.backbone)
        frozen_buffers = _state_sha(dict(model.named_buffers()))
        model.to(settings.device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=settings.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0)
        steps = schedule(config['seed'], settings)
        totals, seen_candidates = {}, 0
        _sync(settings.device)
        training_started = time.perf_counter()
        with (out/'learning.jsonl').open('x') as stream:
            for step in steps:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                inputs, targets, values = cache.batch(step['indices'], 'direct', device=settings.device)
                logits, predicted, _ = model(**inputs)
                ce, mse = F.cross_entropy(logits, targets), F.mse_loss(predicted, values)
                loss = ce+.5*mse
                _require(torch.isfinite(loss), 'Nonfinite training loss')
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                _require(all(p.grad is not None for p in trainable), 'Trainable gradient missing')
                norm = torch.nn.utils.clip_grad_norm_(trainable, settings.gradient_clip, error_if_nonfinite=True)
                _frozen(model, frozen_backbone, frozen_buffers)
                optimizer.step()
                updates += 1
                counts = model.computation_counts(len(step['indices']), inputs['candidates'].shape[1])
                legal = sum(len(cache.menus[index]) for index in step['indices'])
                seen_candidates += legal
                _add_counts(totals, counts)
                epoch_end = step['step'] % (settings.train_examples//settings.batch_size) == 0
                _frozen(model, frozen_backbone, frozen_buffers, hash_backbone=epoch_end)
                record = {
                    'step': step['step'], 'epoch': step['epoch'], 'examples': len(step['indices']),
                    'indices_sha256': _digest(step['indices']), 'policy_ce': float(ce.detach().cpu()),
                    'value_mse': float(mse.detach().cpu()), 'loss': float(loss.detach().cpu()),
                    'gradient_norm_before_clip': float(norm.detach().cpu()), 'actual_legal_candidates': legal,
                    'computation': counts, 'epoch_backbone_hash_checked': epoch_end,
                    'sampled_mps_allocated_bytes_after_step': (
                        torch.mps.current_allocated_memory() if settings.device == 'mps' else None),
                }
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                if epoch_end:
                    os.fsync(stream.fileno())
        _sync(settings.device)
        training_seconds = time.perf_counter()-training_started
        _frozen(model, frozen_backbone, frozen_buffers, hash_backbone=True)
        checkpoint = save_checkpoint(model, out/'weights.pt', plan_sha256=plan_sha256,
                                     configuration=config, backbone_sha256=frozen_backbone)
        receipt = {
            'status': 'completed', 'version': VERSION, 'configuration': config,
            'plan_sha256': plan_sha256, 'data_sha256': data_sha256, 'settings': asdict(settings),
            'cache_sha256': cache.metadata['cache_sha256'], 'initial_state_sha256': initial,
            'backbone_sha256': frozen_backbone, 'buffers_sha256': frozen_buffers,
            'graph_sha256': graph_sha256(graph), 'backbone_unchanged': True,
            'updates': updates, 'examples_seen': settings.epochs*settings.train_examples,
            'actual_legal_candidates': seen_candidates, 'computation': totals,
            'training_seconds': training_seconds, 'fit_wall_seconds': time.perf_counter()-started,
            'training_timing_scope': 'Fixed-batch loop including gradient/hash checks and logging; excludes cache validation, initialization and checkpoint serialization',
            'memory_scope': 'MPS allocation sampled after each optimizer step; not a measured peak',
            'determinism_scope': 'Seeded initialization and order; no bitwise MPS reproducibility claim',
            'mps_fallback_environment': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'parameter_counts': model.parameter_counts(), **checkpoint,
            'learning_sha256': sha256(out/'learning.jsonl'), 'source_sha256': sha256(__file__),
        }
        _write(out/'training.json', receipt)
        return receipt
    except Exception as error:
        _write(out/'failed.json', {'status': 'failed', 'version': VERSION, 'configuration': config,
                                  'plan_sha256': plan_sha256, 'updates_completed': updates,
                                  'error_type': type(error).__name__, 'error': str(error),
                                  'wall_seconds': time.perf_counter()-started})
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(actual, expected):
    return _number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def audit_training(directory, config, plan_sha256, data_sha256, cache_metadata, backbone, graph,
                   *, settings=DEFAULT_TRAINING, training_rows):
    """Audit every scheduled update and final weights without forward/engine calls."""
    directory = _local(directory)
    _configuration(config, baseline=False)
    _require({p.name for p in directory.iterdir()} ==
             {'started.json', 'learning.jsonl', 'weights.pt', 'training.json'},
             'Training artifact membership differs or fit is incomplete')
    receipt = _read(directory/'training.json')
    started = _read(directory/'started.json')
    _require(started == {'status': 'started', 'configuration': config,
                        'plan_sha256': plan_sha256, 'settings': asdict(settings)}, 'Started receipt differs')
    expected = {
        'status': 'completed', 'version': VERSION, 'configuration': config,
        'plan_sha256': plan_sha256, 'data_sha256': data_sha256, 'settings': asdict(settings),
        'cache_sha256': cache_metadata['cache_sha256'], 'graph_sha256': graph_sha256(graph),
        'backbone_sha256': state_sha256(backbone), 'backbone_unchanged': True,
        'updates': settings.updates, 'examples_seen': settings.epochs*settings.train_examples,
        'learning_sha256': sha256(directory/'learning.jsonl'), 'source_sha256': sha256(__file__),
        'checkpoint_sha256': sha256(directory/'weights.pt'),
    }
    _require(all(receipt.get(key) == value for key, value in expected.items()), 'Training receipt binding differs')
    _require(type(receipt['updates']) is int and receipt['backbone_unchanged'] is True,
             'Malformed training terminal state')
    _require(all(_number(receipt.get(key)) and receipt[key] >= 0 for key in
                 ('training_seconds', 'fit_wall_seconds')), 'Invalid training wall time')
    training_rows = list(training_rows)
    _require(len(training_rows) == settings.train_examples, 'Training source length differs')
    audit_cache_metadata(cache_metadata, training_rows, include_successors=False)
    counts = [chess.Board(row['fen']).legal_moves.count() for row in training_rows]
    initial = make_model(config, backbone, graph)
    _require(state_sha256(initial) == receipt['initial_state_sha256']
             and _state_sha(dict(initial.named_buffers())) == receipt['buffers_sha256']
             and initial.parameter_counts() == receipt['parameter_counts'], 'Initial model or parameter receipt differs')
    learning = _rows_file(directory/'learning.jsonl')
    steps = schedule(config['seed'], settings)
    _require(len(learning) == len(steps), 'Training update coverage differs')
    totals, candidates_seen = {}, 0
    for item, step in zip(learning, steps, strict=True):
        _require(set(item) == {'step', 'epoch', 'examples', 'indices_sha256', 'policy_ce', 'value_mse',
                              'loss', 'gradient_norm_before_clip', 'actual_legal_candidates', 'computation',
                              'epoch_backbone_hash_checked', 'sampled_mps_allocated_bytes_after_step'},
                 'Learning record schema differs')
        indices = step['indices']
        expected_counts = initial.computation_counts(len(indices), max(counts[i] for i in indices))
        legal = sum(counts[i] for i in indices)
        _require(type(item.get('step')) is int and item['step'] == step['step']
                 and type(item.get('epoch')) is int and item['epoch'] == step['epoch']
                 and item.get('examples') == len(indices) and item.get('indices_sha256') == _digest(indices)
                 and item.get('actual_legal_candidates') == legal
                 and item.get('computation') == expected_counts, 'Training schedule or computation differs')
        for name in ('policy_ce', 'value_mse', 'loss', 'gradient_norm_before_clip'):
            _require(_number(item.get(name)) and item[name] >= 0, 'Invalid training scalar')
        replayed_loss = float(torch.tensor(item['policy_ce'], dtype=torch.float32)
                              +.5*torch.tensor(item['value_mse'], dtype=torch.float32))
        _require(_close(item['loss'], replayed_loss), 'Training objective arithmetic differs')
        epoch_end = step['step'] % (settings.train_examples//settings.batch_size) == 0
        _require(type(item.get('epoch_backbone_hash_checked')) is bool
                 and item['epoch_backbone_hash_checked'] == epoch_end, 'Epoch hash-check coverage differs')
        memory = item.get('sampled_mps_allocated_bytes_after_step')
        _require((memory is None and settings.device == 'cpu') or
                 (settings.device == 'mps' and type(memory) is int and memory >= 0), 'Invalid memory sample')
        candidates_seen += legal
        _add_counts(totals, expected_counts)
    _require(receipt.get('computation') == totals and receipt.get('actual_legal_candidates') == candidates_seen,
             'Training aggregate computation differs')
    model = load_checkpoint(directory/'weights.pt', backbone, graph,
                            expected_plan_sha256=plan_sha256, expected_config=config,
                            expected_backbone_sha256=expected['backbone_sha256'],
                            expected_checkpoint_sha256=expected['checkpoint_sha256'])
    _require(state_sha256(model) == receipt['state_sha256'], 'Final training state hash differs')
    return receipt


def _direct_computation(model, batch, candidates):
    width = model.width
    root = batch*64*(19*width*9+4*2*width*width*9)
    policy = batch*candidates*((3*width+24)*2*width+2*width)
    value = batch*(width*width+width)
    return {
        'batch_size': batch, 'candidate_slots_per_position': candidates,
        'candidate_score_slots': batch*candidates, 'frozen_root_iterations': batch*4,
        'frozen_root_conv_macs': root, 'input_projection_macs': 0, 'output_projection_macs': 0,
        'graph_dense_matmul_macs': 0, 'graph_structural_message_products': 0,
        'graph_matrix_entries_materialized': 0, 'graph_edge_gain_evaluations': 0,
        'node_state_updates': 0, 'slot_gather_values': 0, 'slot_pool_accumulations': 0,
        'slot_occupancy_divisions': 0, 'policy_head_macs': policy, 'value_head_macs': value,
        'accounted_dense_mac_total': root+policy+value, 'native_successors': 0,
        'candidate_branch_evaluations': 0,
        'scope': 'Selected dense convolution/linear MACs; excludes encoding, activation and validation work',
    }


def _computation(model, batch, candidates):
    return model.computation_counts(batch, candidates) if isinstance(model, ConnectomeChessAdapter) else (
        _direct_computation(model, batch, candidates))


def _arithmetic(logits, legal_ids, target):
    _require(type(logits) is list and len(logits) == len(legal_ids) and bool(logits)
             and all(_number(value) for value in logits), 'Raw logits must be finite and menu-aligned')
    maximum = max(logits)
    shifted = [value-maximum for value in logits]
    normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    log_probabilities = [value-normalizer for value in shifted]
    probabilities = [math.exp(value) for value in log_probabilities]
    target_index = legal_ids.index(target)
    choice = legal_ids[logits.index(maximum)]
    return {
        'choice': choice, 'correct': choice == target, 'target': target,
        'target_nll': -log_probabilities[target_index], 'target_probability': probabilities[target_index],
        'max_probability': max(probabilities),
        'entropy': -math.fsum(p*log_p for p, log_p in zip(probabilities, log_probabilities, strict=True)),
        'logit_span': maximum-min(logits),
    }


@torch.inference_mode()
def evaluate(model, rows_or_cache, outfile, *, configuration, plan_sha256, batch_size=16):
    """Final-only CPU evaluation, preserving every legal ID/logit and all calls."""
    _hash(plan_sha256)
    _require(type(batch_size) is int and batch_size > 0, 'Invalid evaluation batch size')
    anchor_eval._cpu(model)
    identity = _identity(model, configuration)
    cache = _cache(rows_or_cache)
    previous_threads, was_training = torch.get_num_threads(), model.training
    predictions, totals, batches, legal_total = [], {}, 0, 0
    with Path(outfile).open('x') as stream:
        try:
            torch.set_num_threads(2)
            model.eval()
            started = time.perf_counter()
            for start in range(0, len(cache), batch_size):
                end = min(start+batch_size, len(cache))
                inputs, _, _ = cache.batch(range(start, end), 'direct')
                logits, values, hidden = model(**inputs)
                mask = inputs['legal_mask']
                _require(logits.shape == mask.shape and values.shape == (end-start,)
                         and hidden.shape == (end-start, model.width, 8, 8)
                         and torch.isfinite(logits[mask]).all() and torch.isfinite(values).all()
                         and torch.isfinite(hidden).all(), 'Invalid evaluation outputs')
                counts = _computation(model, end-start, inputs['candidates'].shape[1])
                _add_counts(totals, counts)
                batches += 1
                for offset, row in enumerate(cache.rows[start:end]):
                    menu = list(cache.menus[start+offset])
                    raw = logits[offset, :len(menu)].double().tolist()
                    record = {
                        'version': VERSION, **row, 'configuration': configuration['name'],
                        'model_configuration': configuration, 'plan_sha256': plan_sha256,
                        'model_state_sha256': identity['state_sha256'], 'legal_ids': menu, 'logits': raw,
                        **_arithmetic(raw, menu, row['target_uci']), 'value': float(values[offset]),
                        'hidden_rms': float(hidden[offset].double().square().mean().sqrt()),
                    }
                    anchor_eval._validate_prediction(record, legal_count=len(menu))
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                    predictions.append(record)
                    legal_total += len(menu)
            stream.flush()
            elapsed = time.perf_counter()-started
        finally:
            model.train(was_training)
            torch.set_num_threads(previous_threads)
    _require(_identity(model, configuration) == identity, 'Model changed during evaluation')
    metrics, saved = audit_predictions(
        outfile, cache.rows, expected_configuration=configuration,
        expected_plan_sha256=plan_sha256, expected_state_sha256=identity['state_sha256'])
    _require(saved == predictions, 'Saved predictions differ')
    return {
        'status': 'completed', 'version': VERSION, 'model': identity, 'plan_sha256': plan_sha256,
        'metrics': metrics, 'cache_sha256': cache.metadata['cache_sha256'],
        'batch_size': batch_size, 'batches': batches, 'actual_legal_candidates': legal_total,
        'computation': totals, 'predictions_sha256': sha256(outfile), 'model_state_unchanged': True,
        'evaluation_wall_seconds': elapsed,
        'timing_scope': 'CPU cached batch inference, graph/projection work, arithmetic and writing; cache construction/validation and post-audit excluded',
    }


def audit_predictions(path, rows, *, expected_configuration, expected_plan_sha256, expected_state_sha256):
    _configuration(expected_configuration)
    _hash(expected_plan_sha256)
    _hash(expected_state_sha256)
    rows = list(rows)
    anchor_eval._rows(rows)
    predictions = _rows_file(path)
    _require(len(predictions) == len(rows), 'Prediction coverage differs')
    for prediction, row in zip(predictions, rows, strict=True):
        board = chess.Board(row['fen'])
        _require(board.is_valid() and not board.is_game_over(claim_draw=False), 'Invalid evaluation board')
        menu = sorted(move.uci() for move in board.legal_moves)
        _require(prediction.get('version') == VERSION
                 and prediction.get('configuration') == expected_configuration['name']
                 and prediction.get('model_configuration') == expected_configuration
                 and prediction.get('plan_sha256') == expected_plan_sha256
                 and prediction.get('model_state_sha256') == expected_state_sha256
                 and prediction.get('legal_ids') == menu, 'Prediction identity or legal menu differs')
        for key in ('id', 'game_id', 'fen', 'target_uci', 'target_value'):
            _require(prediction.get(key) == row[key] and type(prediction.get(key)) is type(row[key]),
                     'Prediction source row differs')
        arithmetic = _arithmetic(prediction.get('logits'), menu, row['target_uci'])
        for key, value in arithmetic.items():
            _require((type(value) is str and prediction.get(key) == value)
                     or (type(value) is bool and type(prediction.get(key)) is bool and prediction[key] == value)
                     or (type(value) in (int, float) and _close(prediction.get(key), value)), 'Prediction arithmetic differs')
        anchor_eval._validate_prediction(prediction, legal_count=len(menu))
    return anchor_eval.summarize(predictions), predictions


def audit_evaluation(receipt, path, rows, *, configuration, plan_sha256, model, cache_metadata):
    identity = _identity(model, configuration)
    metrics, predictions = audit_predictions(path, rows, expected_configuration=configuration,
                                             expected_plan_sha256=plan_sha256,
                                             expected_state_sha256=identity['state_sha256'])
    audit_cache_metadata(cache_metadata, rows, include_successors=False)
    batch_size = receipt.get('batch_size')
    _require(type(batch_size) is int and batch_size > 0, 'Invalid evaluation batch size receipt')
    counts = [len(item['legal_ids']) for item in predictions]
    totals = {}
    for start in range(0, len(counts), batch_size):
        group = counts[start:start+batch_size]
        _add_counts(totals, _computation(model, len(group), max(group)))
    expected = {'status': 'completed', 'version': VERSION, 'model': identity, 'plan_sha256': plan_sha256,
                'metrics': metrics, 'cache_sha256': cache_metadata['cache_sha256'],
                'batches': (len(counts)+batch_size-1)//batch_size, 'actual_legal_candidates': sum(counts),
                'computation': totals, 'predictions_sha256': sha256(path), 'model_state_unchanged': True}
    _require(all(receipt.get(key) == value for key, value in expected.items()), 'Evaluation receipt differs')
    _require(_number(receipt.get('evaluation_wall_seconds')) and receipt['evaluation_wall_seconds'] >= 0,
             'Invalid evaluation wall time')
    return metrics, predictions


def grading_module():
    """Reuse the unchanged cached-choice Stockfish assessment and its auditor."""
    path = ROOT/'scripts/chess_compute_study.py'
    spec = importlib.util.spec_from_file_location('connectome_frozen_secondary_grading', path)
    _require(spec is not None and spec.loader is not None, 'Frozen grader cannot be imported')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {'regret_nodes': 20000, 'value_cp_scale': 600., 'mate_cp': 10000,
                'engine_threads': 1, 'engine_hash_mb': 16}
    _require(all(module.PROTOCOL[key] == value for key, value in expected.items()),
             'Frozen grader budget or score semantics differ')
    return module


def _grading_inputs(plan, decisions, call_ceiling, requested_node_ceiling):
    _require(type(plan) is dict and set(plan) ==
             {'engine_path', 'panels', 'secondary_indices', 'configurations'}, 'Grading plan fields differ')
    for ceiling in (call_ceiling, requested_node_ceiling):
        _require(type(ceiling) is int and ceiling > 0, 'Invalid engine budget ceiling')
    _require(type(plan['configurations']) is list and bool(plan['configurations'])
             and all(type(c) is dict and set(c) == {'id'} for c in plan['configurations']),
             'Grading configuration schema differs')
    names = [c['id'] for c in plan['configurations']]
    allowed = {c['name'] for c in configurations()+baseline_configurations()}
    _require(len(set(names)) == len(names) and set(names) <= allowed, 'Invalid/duplicate graded model names')
    _require(set(plan['panels']) == set(plan['secondary_indices']) == {'dev', 'shift'},
             'Grading split membership differs')
    menus, positions = {}, {}
    for split in ('dev', 'shift'):
        rows = plan['panels'][split]
        anchor_eval._rows(rows)
        indices = plan['secondary_indices'][split]
        _require(type(indices) is list and bool(indices) and all(type(i) is int for i in indices)
                 and indices == sorted(set(indices)) and min(indices) >= 0 and max(indices) < len(rows),
                 'Invalid fixed grading panel indices')
        for index, row in enumerate(rows):
            board = chess.Board(row['fen'])
            _require(board.is_valid() and not board.is_game_over(claim_draw=False), 'Invalid grading board')
            legal = {move.uci() for move in board.legal_moves}
            _require(row['target_uci'] in legal, 'Illegal grading source target')
            menus[split, index], positions[split, index] = legal, row
    decisions = list(decisions)
    expected = {(name, split, index) for name in names for split in ('dev', 'shift')
                for index in range(len(plan['panels'][split]))}
    lookup = {}
    for decision in decisions:
        key = decision.get('configuration'), decision.get('split'), decision.get('panel_index')
        _require(type(key[2]) is int and key in expected and key not in lookup, 'Extra/duplicate grading decision')
        row = positions[key[1:]]
        _require(decision.get('id') == row['id'] and decision.get('choice') in menus[key[1:]],
                 'Grading choice or source identity differs')
        lookup[key] = decision
    _require(set(lookup) == expected, 'Grading decisions do not cover every configured model and position')
    calls = sum(1+len({lookup[name, split, index]['choice'] for name in names})
                for split in ('dev', 'shift') for index in plan['secondary_indices'][split])
    _require(calls <= call_ceiling and calls*20000 <= requested_node_ceiling,
             'Planned unique engine calls exceed the frozen ceiling')
    return decisions, names, calls


def _grade_means(records, names):
    return {
        split: {
            name: {
                'mean_bounded_score_loss': math.fsum(r['bounded_regret'] for r in records
                                                   if r['split'] == split and r['configuration'] == name)
                /sum(r['split'] == split and r['configuration'] == name for r in records),
                'mean_cp_loss': math.fsum(r['cp_loss'] for r in records
                                         if r['split'] == split and r['configuration'] == name)
                /sum(r['split'] == split and r['configuration'] == name for r in records),
            } for name in names
        } for split in ('dev', 'shift')
    }


def score_stronger(grading_plan, out, decisions, *, plan_sha256, expected_engine_sha256,
                   call_ceiling=5632, requested_node_ceiling=112640000):
    """Grade each unique chosen move once at 20k nodes, retaining negative losses."""
    _hash(plan_sha256)
    _hash(expected_engine_sha256)
    decisions, names, calls = _grading_inputs(grading_plan, decisions, call_ceiling, requested_node_ceiling)
    _require(sha256(grading_plan['engine_path']) == expected_engine_sha256, 'Engine binary hash differs')
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    binding = {'plan_sha256': plan_sha256, 'grading_plan_sha256': _digest(grading_plan),
               'decisions_sha256': _digest(decisions), 'expected_engine_sha256': expected_engine_sha256,
               'call_ceiling': call_ceiling, 'requested_node_ceiling': requested_node_ceiling,
               'planned_unique_calls': calls}
    _write(out/'started.json', {'status': 'started', **binding})
    begun = time.perf_counter()
    try:
        grader = grading_module()
        cost = grader.score_secondary(grading_plan, out, decisions)
        analyses, records = _rows_file(out/'analyses.jsonl'), _rows_file(out/'regret.jsonl')
        grader.validate_secondary(grading_plan, decisions, analyses, records, cost)
        _require(cost['calls'] == calls and cost['requested_nodes'] == calls*20000,
                 'Completed grading cost differs from planned cached calls')
        receipt = {
            'status': 'completed', 'version': VERSION, **binding, 'cost': cost,
            'means': _grade_means(records, names),
            'files': {name: sha256(out/name) for name in ('engine.json', 'analyses.jsonl', 'regret.jsonl')},
            'source_sha256': sha256(__file__), 'grader_source_sha256': sha256(ROOT/'scripts/chess_compute_study.py'),
            'grading_wall_seconds': time.perf_counter()-begun,
            'semantics': 'Signed bounded-score loss: unrestricted 20k-node score minus chosen-root 20k-node score; finite-search negative differences retained; not game-theoretic regret',
        }
        _write(out/'completed.json', receipt)
        audit_stronger(grading_plan, out, decisions, expected_plan_sha256=plan_sha256,
                       expected_engine_sha256=expected_engine_sha256, call_ceiling=call_ceiling,
                       requested_node_ceiling=requested_node_ceiling)
        return receipt
    except Exception as error:
        _write(out/'failed.json', {'status': 'failed', **binding, 'error_type': type(error).__name__,
                                  'error': str(error), 'wall_seconds': time.perf_counter()-begun})
        raise


def audit_stronger(grading_plan, directory, decisions, *, expected_plan_sha256, expected_engine_sha256,
                   call_ceiling=5632, requested_node_ceiling=112640000):
    directory = Path(directory)
    _hash(expected_plan_sha256)
    _hash(expected_engine_sha256)
    decisions, names, calls = _grading_inputs(grading_plan, decisions, call_ceiling, requested_node_ceiling)
    required = {'started.json', 'completed.json', 'engine.json', 'analyses.jsonl', 'regret.jsonl'}
    _require({p.name for p in directory.iterdir()} == required, 'Grading artifacts are partial or contain extras')
    receipt = _read(directory/'completed.json')
    binding = {'plan_sha256': expected_plan_sha256, 'grading_plan_sha256': _digest(grading_plan),
               'decisions_sha256': _digest(decisions), 'expected_engine_sha256': expected_engine_sha256,
               'call_ceiling': call_ceiling, 'requested_node_ceiling': requested_node_ceiling,
               'planned_unique_calls': calls}
    _require(_read(directory/'started.json') == {'status': 'started', **binding}, 'Grading started binding differs')
    _require(receipt.get('status') == 'completed' and receipt.get('version') == VERSION
             and all(receipt.get(key) == value for key, value in binding.items()), 'Grading receipt binding differs')
    expected_files = {name: sha256(directory/name) for name in ('engine.json', 'analyses.jsonl', 'regret.jsonl')}
    _require(receipt.get('files') == expected_files and receipt.get('source_sha256') == sha256(__file__)
             and receipt.get('grader_source_sha256') == sha256(ROOT/'scripts/chess_compute_study.py'),
             'Grading file/source hashes differ')
    engine = _read(directory/'engine.json')
    _require(engine.get('sha256') == expected_engine_sha256
             and engine.get('path') == grading_plan['engine_path']
             and type(engine.get('id')) is dict
             and engine['id'].get('name', '').startswith('Stockfish 19'), 'Engine identity differs')
    analyses, records = _rows_file(directory/'analyses.jsonl'), _rows_file(directory/'regret.jsonl')
    for item in analyses:
        _require(set(item) == {'split', 'panel_index', 'id', 'root_move', 'pv_first', 'score_cp', 'mate',
                              'bounded_score', 'requested_nodes', 'reported_nodes', 'wall_seconds'}
                 and type(item.get('panel_index')) is int and type(item.get('score_cp')) is int
                 and (item.get('mate') is None or type(item['mate']) is int)
                 and _number(item.get('bounded_score')), 'Invalid engine analysis schema/types')
        _require(type(item.get('reported_nodes')) is int and item['reported_nodes'] >= 0
                 and type(item.get('requested_nodes')) is int and item['requested_nodes'] == 20000
                 and _number(item.get('wall_seconds')) and item['wall_seconds'] >= 0,
                 'Invalid engine call costs')
    for item in records:
        _require(set(item) == {'configuration', 'split', 'panel_index', 'id', 'choice', 'bounded_regret', 'cp_loss'}
                 and type(item.get('panel_index')) is int and type(item.get('cp_loss')) is int
                 and _number(item.get('bounded_regret')), 'Invalid signed-loss record schema/types')
    cost = receipt.get('cost')
    _require(type(cost) is dict and set(cost) == {'calls', 'requested_nodes', 'reported_nodes', 'wall_seconds'}
             and all(type(cost[key]) is int and cost[key] >= 0 for key in ('calls', 'requested_nodes', 'reported_nodes'))
             and _number(cost['wall_seconds']) and cost['wall_seconds'] >= 0, 'Invalid aggregate engine costs')
    grading_module().validate_secondary(grading_plan, decisions, analyses, records, cost)
    _require(cost['calls'] == calls and cost['requested_nodes'] == calls*20000
             and receipt.get('means') == _grade_means(records, names), 'Grading means or cache costs differ')
    _require(_number(receipt.get('grading_wall_seconds')) and receipt['grading_wall_seconds'] >= 0,
             'Invalid total grading wall time')
    return receipt


@torch.inference_mode()
def measure_latency(model, rows, selected_indices, *, configuration, plan_sha256, warmups=3):
    """Record every CPU choose call, including three separately charged warmups."""
    anchor_eval._cpu(model)
    identity = _identity(model, configuration)
    _hash(plan_sha256)
    rows, indices = list(rows), list(selected_indices)
    anchor_eval._rows(rows)
    _require(type(warmups) is int and warmups == 3, 'Exactly three warmups are required')
    _require(bool(indices) and len(set(indices)) == len(indices)
             and all(type(i) is int and 0 <= i < len(rows) for i in indices), 'Invalid latency indices')

    def call(fen, identifier, index):
        started = time.perf_counter()
        board = chess.Board(fen)
        original = board.fen(en_passant='fen')
        legal = sorted(move.uci() for move in board.legal_moves)
        _require(board.is_valid() and not board.is_game_over(claim_draw=False), 'Invalid latency position')
        decision = model.choose(board)
        _require(board.fen(en_passant='fen') == original and decision.get('choice') in legal,
                 'Latency call mutated the board or returned an illegal move')
        probabilities = decision.get('probabilities')
        _require(type(probabilities) is dict and set(probabilities) == set(legal)
                 and all(_number(value) and 0 <= value <= 1 for value in probabilities.values())
                 and math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-6),
                 'Invalid latency probability menu')
        _require(decision['choice'] == max(legal, key=lambda move: probabilities[move]), 'Latency argmax differs')
        return {'id': identifier, 'index': index, 'fen': fen, 'legal_ids': legal, 'decision': decision,
                'computation': _computation(model, 1, len(legal)),
                'wall_ms': (time.perf_counter()-started)*1000}

    previous_threads, was_training = torch.get_num_threads(), model.training
    try:
        torch.set_num_threads(2)
        model.eval()
        warmup_records = [call(chess.STARTING_FEN, 'starting-board', index) for index in range(warmups)]
        records = [call(rows[index]['fen'], rows[index]['id'], index) for index in indices]
    finally:
        torch.set_num_threads(previous_threads)
        model.train(was_training)
    _require(_identity(model, configuration) == identity, 'Model changed during latency measurement')
    result = {
        'version': VERSION, 'model': identity, 'plan_sha256': plan_sha256, 'device': 'cpu', 'torch_threads': 2,
        'warmup_records': warmup_records, 'records': records, 'forward_calls': warmups+len(indices),
        'warmup_wall_ms': math.fsum(record['wall_ms'] for record in warmup_records),
        'total_wall_ms': math.fsum(record['wall_ms'] for record in records),
        'model_state_unchanged': True,
        'timing_scope': 'Full native board construction, legal menu, choose with graph assembly/projections, synchronization and response validation; model load excluded; warmups separately charged',
    }
    audit_latency(result, rows, indices, configuration=configuration, plan_sha256=plan_sha256,
                  expected_state_sha256=identity['state_sha256'], warmups=warmups, model=model)
    return result


def audit_latency(saved, rows, indices, *, configuration, plan_sha256, expected_state_sha256,
                  warmups=3, model):
    _configuration(configuration)
    _hash(plan_sha256)
    _hash(expected_state_sha256)
    rows, indices = list(rows), list(indices)
    anchor_eval._rows(rows)
    _require(type(warmups) is int and warmups == 3 and bool(indices) and len(set(indices)) == len(indices)
             and all(type(i) is int and 0 <= i < len(rows) for i in indices), 'Invalid latency selection')
    _require(saved.get('version') == VERSION and saved.get('plan_sha256') == plan_sha256
             and saved.get('model', {}).get('configuration') == configuration
             and saved['model'].get('state_sha256') == expected_state_sha256
             and saved.get('device') == 'cpu' and saved.get('torch_threads') == 2
             and saved.get('model_state_unchanged') is True
             and saved.get('forward_calls') == warmups+len(indices), 'Latency receipt identity/call count differs')
    _require(saved['model'] == _identity(model, configuration), 'Latency architecture/tensor identity differs')
    _require(len(saved['warmup_records']) == warmups and len(saved['records']) == len(indices),
             'Latency record coverage differs')
    expected = [(chess.STARTING_FEN, 'starting-board', i) for i in range(warmups)]
    expected += [(rows[i]['fen'], rows[i]['id'], i) for i in indices]
    for record, (fen, identifier, index) in zip(saved['warmup_records']+saved['records'], expected, strict=True):
        board = chess.Board(fen)
        legal = sorted(move.uci() for move in board.legal_moves)
        _require(board.is_valid() and not board.is_game_over(claim_draw=False), 'Invalid latency source board')
        _require(record.get('id') == identifier and record.get('index') == index
                 and record.get('fen') == fen and record.get('legal_ids') == legal
                 and _number(record.get('wall_ms')) and record['wall_ms'] >= 0, 'Latency source or wall time differs')
        decision = record.get('decision', {})
        probabilities = decision.get('probabilities', {})
        _require(type(probabilities) is dict and set(probabilities) == set(legal)
                 and all(_number(value) and 0 <= value <= 1 for value in probabilities.values())
                 and math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-6)
                 and decision.get('choice') == max(legal, key=lambda move: probabilities[move]),
                 'Latency decision probabilities or choice differ')
        _require(_number(decision.get('value')) and -1 <= decision['value'] <= 1
                 and _number(decision.get('latency_ms')) and 0 <= decision['latency_ms'] <= record['wall_ms'],
                 'Invalid internal choose time or value')
        _require(record.get('computation') == _computation(model, 1, len(legal)), 'Latency computation differs')
    _require(_close(saved.get('warmup_wall_ms'), math.fsum(r['wall_ms'] for r in saved['warmup_records']))
             and _close(saved.get('total_wall_ms'), math.fsum(r['wall_ms'] for r in saved['records'])),
             'Latency timing totals differ')
    return saved
