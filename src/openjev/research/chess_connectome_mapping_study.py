"""Bounded training-only hard-square search on a frozen chess backbone.

This is a separate checkpoint and audit contract from the published fixed-map
study. Saved-output auditing verifies identities, schedules and arithmetic; it
does not prove recorded losses were produced by an exact optimizer execution.
Graph-derived weights remain in local runs/. No trial is launched here.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch
from torch.nn import functional as F

from . import chess_connectome_study as original
from .chess_candidate import CandidateChess
from .chess_candidate_eval import audit_cache_metadata
from .chess_connectome_interface import HardSquareConnectomeAdapter, evaluate_square_swap
from .connectome_graph import graph_sha256

ROOT = original.ROOT
RUNS_ROOT = ROOT / 'runs'
VERSION = 'openjev-connectome-hard-mapping-study-v1'
VARIANTS = ('biological', 'rewire151', 'rewire163', 'rewire179', 'node_local')
POLICIES = ('fixed', 'learned')
SEEDS = original.SEEDS
DERIVED_BUFFERS = {'node_slots', 'slot_occupancy', 'square_permutation'}
SOURCE_FILES = (
    'src/openjev/research/chess_connectome_mapping_study.py',
    'src/openjev/research/chess_connectome_interface.py',
    'src/openjev/research/chess_connectome_study.py',
    'src/openjev/research/chess_connectome_adapter.py',
    'src/openjev/research/connectome_graph.py',
    'src/openjev/research/chess_candidate.py',
    'src/openjev/research/chess_candidate_eval.py',
    'src/openjev/research/chess_anchor_eval.py',
    'src/openjev/research/chess_anchor.py',
    'src/openjev/research/chess_spatial.py',
    'src/openjev/research/chess_compute.py',
)
_require = original._require
_digest = original._digest
_hash = original._hash
_read = original._read
_write = original._write
_rows_file = original._rows_file
_number = original._number
_close = original._close
_state_sha = original._state_sha
state_sha256 = original.state_sha256
sha256 = original.sha256


def source_bindings():
    return {path: sha256(ROOT / path) for path in SOURCE_FILES}


def _local(path):
    path = Path(path)
    _require(path.resolve().is_relative_to(RUNS_ROOT.resolve()),
             'Graph-derived checkpoint/training artifacts must remain under local runs/')
    _require(not path.is_symlink(), 'Artifact must not be a symlink')
    return path


def configurations():
    return [dict(c, name=f"{c['variant']}-{policy}-{c['seed']}", mapping_policy=policy)
            for c in original.configurations() if c['variant'] in VARIANTS for policy in POLICIES]


def _configuration(config):
    _require(type(config) is dict and set(config) == original.CONFIG_FIELDS | {'mapping_policy'}
             and config in configurations()
             and all(type(config[key]) is int for key in ('seed', 'mapping_seed', 'backbone_seed')),
             'Unknown hard-mapping configuration')
    return dict(config)


def legacy_config(config):
    config = _configuration(config)
    result = {key: value for key, value in config.items() if key != 'mapping_policy'}
    result['name'] = f"{config['variant']}-{config['seed']}"
    return original._configuration(result, baseline=False)


def _immutable_buffers(model):
    return {key: value for key, value in model.named_buffers() if key not in DERIVED_BUFFERS}


class MappingStudyAdapter(HardSquareConnectomeAdapter):
    """The prototype's exact actor with explicit training/checkpoint identity."""

    def __init__(self, config, backbone, graph):
        self.study_configuration = _configuration(config)
        super().__init__(backbone, graph, mode=config['mode'], seed=config['seed'],
                         mapping_seed=config['mapping_seed'], slot_channels=16, steps=4, alpha=.5)
        self._initial_immutable_sha256 = _state_sha(_immutable_buffers(self))
        self._initial_backbone_sha256 = state_sha256(self.backbone)

    @property
    def config(self):
        return {**super().config, 'architecture': VERSION,
                'configuration': dict(self.study_configuration),
                'configuration_sha256': _digest(self.study_configuration),
                'checkpoint_contract': 'Separate authenticated local hard-mapping checkpoint; not an old adapter',
                'mapping_policy': self.study_configuration['mapping_policy']}

    @property
    def provenance(self):
        return {**super().provenance, 'experiment_status': 'Bounded mapping-training helper; no result implied',
                'mapping_policy': self.study_configuration['mapping_policy'],
                'interface_permutation_learnable': self.study_configuration['mapping_policy'] == 'learned',
                'derivative_asset_policy': 'Graph-derived checkpoint weights remain under local runs/'}

    @torch.no_grad()
    def choose(self, board):
        result = super().choose(board)
        result.update({'study_architecture': VERSION,
                       'mapping_policy': self.study_configuration['mapping_policy'],
                       'configuration': dict(self.study_configuration),
                       'model_kind': 'Frozen direct backbone with hard-square graph adapter; no cross-move state'})
        return result


def make_model(config, backbone, graph):
    config = _configuration(config)
    _require(isinstance(backbone, CandidateChess) and backbone.seed == config['backbone_seed'],
             'Backbone seed/architecture differs')
    return MappingStudyAdapter(config, backbone, graph)


def _frozen(model, *, hashes=False):
    model._assert_interface()
    _require(not model.backbone.training and all(not p.requires_grad and p.grad is None
                                                for p in model.backbone.parameters()), 'Backbone is not frozen')
    if model.study_configuration['mapping_policy'] == 'fixed':
        _require(model.square_permutation.cpu().tolist() == list(range(64)), 'Fixed mapping changed')
    if hashes:
        _require(state_sha256(model.backbone) == model._initial_backbone_sha256,
                 'Frozen backbone tensor hash changed')
        _require(_state_sha(_immutable_buffers(model)) == model._initial_immutable_sha256,
                 'Immutable graph or original mapping buffers changed')


def save_checkpoint(model, path, *, plan_sha256, configuration, backbone_sha256):
    path = _local(path)
    _hash(plan_sha256)
    _hash(backbone_sha256)
    config = _configuration(configuration)
    _require(isinstance(model, MappingStudyAdapter) and model.study_configuration == config,
             'Expected a matching hard-mapping study adapter')
    _frozen(model, hashes=True)
    _require(state_sha256(model.backbone) == backbone_sha256, 'Frozen backbone identity differs')
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {'version': VERSION, 'plan_sha256': plan_sha256, 'configuration': config,
               'configuration_sha256': _digest(config), 'architecture': model.config,
               'graph_sha256': model.config['source_graph_sha256'], 'backbone_sha256': backbone_sha256,
               'immutable_buffers_sha256': model._initial_immutable_sha256,
               'state_sha256': _state_sha(state), 'source_sha256': source_bindings(), 'state_dict': state}
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return {'checkpoint_sha256': sha256(path), 'state_sha256': payload['state_sha256']}


def load_checkpoint(path, backbone, graph, *, expected_plan_sha256, expected_config,
                    expected_backbone_sha256, expected_checkpoint_sha256):
    path = _local(path)
    config = _configuration(expected_config)
    for value in (expected_plan_sha256, expected_backbone_sha256, expected_checkpoint_sha256):
        _hash(value)
    _require(sha256(path) == expected_checkpoint_sha256, 'Checkpoint file hash differs')
    _require(state_sha256(backbone) == expected_backbone_sha256, 'Supplied backbone identity differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    keys = {'version', 'plan_sha256', 'configuration', 'configuration_sha256', 'architecture',
            'graph_sha256', 'backbone_sha256', 'immutable_buffers_sha256', 'state_sha256',
            'source_sha256', 'state_dict'}
    _require(type(payload) is dict and set(payload) == keys and payload['version'] == VERSION,
             'Checkpoint format differs')
    model = make_model(config, backbone, graph).cpu()
    expected = model.state_dict()
    state = payload['state_dict']
    _require(type(state) is dict and set(state) == set(expected), 'Checkpoint tensor membership differs')
    for key, tensor in state.items():
        _require(isinstance(tensor, torch.Tensor) and tensor.shape == expected[key].shape
                 and tensor.dtype == expected[key].dtype, 'Checkpoint tensor shape/dtype differs')
    model.apply_permutation(state['square_permutation'])
    expected = model.state_dict()
    buffers = dict(model.named_buffers())
    for key, tensor in state.items():
        if key.startswith('backbone.') or key in buffers:
            _require(torch.equal(tensor, expected[key]), 'Checkpoint frozen or derived mapping buffer differs')
    _require(payload['configuration'] == config and payload['configuration_sha256'] == _digest(config)
             and payload['plan_sha256'] == expected_plan_sha256
             and payload['backbone_sha256'] == expected_backbone_sha256
             and payload['graph_sha256'] == graph_sha256(graph)
             and payload['architecture'] == model.config
             and payload['immutable_buffers_sha256'] == model._initial_immutable_sha256
             and payload['source_sha256'] == source_bindings(), 'Checkpoint metadata binding differs')
    _require(_state_sha(state) == payload['state_sha256'], 'Checkpoint tensor hash differs')
    model.load_state_dict(state, strict=True)
    _frozen(model, hashes=True)
    return model.eval()


@dataclass(frozen=True)
class TrainingSettings(original.TrainingSettings):
    warmup_updates: int = 256
    proposal_interval: int = 4
    proposal_seed_base: int = 11600041

    def __post_init__(self):
        super().__post_init__()
        for key, minimum in [('warmup_updates', 0), ('proposal_interval', 1), ('proposal_seed_base', 0)]:
            _require(type(getattr(self, key)) is int and getattr(self, key) >= minimum, f'Invalid {key}')
        _require(self.warmup_updates < self.updates, 'Proposal warmup exhausts the training budget')

    @property
    def proposals(self):
        return len(range(self.warmup_updates + 1, self.updates + 1, self.proposal_interval))


DEFAULT_TRAINING = TrainingSettings()


def schedule(seed, settings=DEFAULT_TRAINING):
    _require(isinstance(settings, TrainingSettings), 'Expected mapping TrainingSettings')
    return original.schedule(seed, settings)


def proposal_schedule(seed, settings=DEFAULT_TRAINING):
    _require(type(seed) is int and seed >= 0 and isinstance(settings, TrainingSettings),
             'Invalid proposal schedule inputs')
    generator = random.Random(settings.proposal_seed_base + seed)
    return [{'proposal_index': index + 1, 'before_update': update,
             'swap': generator.sample(range(64), 2)}
            for index, update in enumerate(range(settings.warmup_updates + 1,
                                                settings.updates + 1, settings.proposal_interval))]


def fit(config, backbone, graph, rows_or_cache, out, *, plan_sha256, data_sha256,
        settings=DEFAULT_TRAINING):
    started = time.perf_counter()
    config = _configuration(config)
    _hash(plan_sha256)
    _hash(data_sha256)
    _require(isinstance(settings, TrainingSettings), 'Expected explicit mapping TrainingSettings')
    out = _local(out)
    out.mkdir(parents=False, exist_ok=False)
    updates = proposal_count = 0
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        sources = source_bindings()
        _write(out / 'started.json', {'status': 'started', 'configuration': config,
                                     'plan_sha256': plan_sha256, 'settings': asdict(settings),
                                     'source_sha256': sources})
        if settings.device == 'mps':
            _require(torch.backends.mps.is_available(), 'MPS unavailable; no fallback permitted')
            _require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') in (None, '0'), 'MPS fallback must be disabled')
        torch.set_num_threads(settings.torch_threads)
        torch.use_deterministic_algorithms(settings.deterministic_algorithms)
        cache = original._cache(rows_or_cache)
        _require(len(cache) == settings.train_examples, 'Training row count differs')
        model = make_model(config, backbone, graph)
        initial = state_sha256(model)
        model.to(settings.device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=settings.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0)
        steps = schedule(config['seed'], settings)
        proposals = {row['before_update']: row for row in proposal_schedule(config['seed'], settings)}
        totals, proposal_totals, seen_candidates, proposal_candidates, accepted = {}, {}, 0, 0, 0
        proposal_seconds = step_seconds = 0.
        original._sync(settings.device)
        training_started = time.perf_counter()
        with (out / 'learning.jsonl').open('x') as stream, (out / 'proposals.jsonl').open('x') as journal:
            for step in steps:
                original._sync(settings.device)
                step_started = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                inputs, targets, values = cache.batch(step['indices'], 'direct', device=settings.device)
                counts = model.computation_counts(len(step['indices']), inputs['candidates'].shape[1])
                legal = sum(len(cache.menus[index]) for index in step['indices'])
                proposal = None
                if step['step'] in proposals:
                    spec = proposals[step['step']]
                    proposal = evaluate_square_swap(model, dict(inputs, targets=targets, values=values),
                                                    *spec['swap'], policy=config['mapping_policy'])
                    proposal.update(spec)
                    proposal.update({'epoch': step['epoch'], 'indices_sha256': _digest(step['indices']),
                                     'configuration_sha256': _digest(config), 'plan_sha256': plan_sha256,
                                     'computation_per_forward': counts, 'actual_legal_candidates': 2 * legal})
                    original._sync(settings.device)
                    proposal['outer_wall_seconds'] = time.perf_counter() - step_started
                    proposal['outer_timing_scope'] = 'Start of update through proposal; includes batch creation, zero_grad, helper copies and synchronization'
                    proposal_count += 1
                    accepted += int(proposal['accepted'])
                    proposal_candidates += 2 * legal
                    proposal_seconds += proposal['outer_wall_seconds']
                    original._add_counts(proposal_totals, counts)
                    original._add_counts(proposal_totals, counts)
                    journal.write(json.dumps(proposal, allow_nan=False) + '\n')
                    journal.flush()
                logits, predicted, _ = model(**inputs)
                ce, mse = F.cross_entropy(logits, targets), F.mse_loss(predicted, values)
                loss = ce + .5 * mse
                _require(torch.isfinite(loss), 'Nonfinite training loss')
                scalars = {'policy_ce': float(ce.detach().cpu()), 'value_mse': float(mse.detach().cpu()),
                           'loss': float(loss.detach().cpu())}
                if proposal:
                    _selected_objective(proposal, scalars)
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                _require(all(p.grad is not None for p in trainable), 'Trainable gradient missing')
                norm = torch.nn.utils.clip_grad_norm_(trainable, settings.gradient_clip, error_if_nonfinite=True)
                _frozen(model)
                optimizer.step()
                updates += 1
                seen_candidates += legal
                original._add_counts(totals, counts)
                epoch_end = step['step'] % (settings.train_examples // settings.batch_size) == 0
                _frozen(model, hashes=epoch_end)
                original._sync(settings.device)
                wall = time.perf_counter() - step_started
                step_seconds += wall
                record = {'step': step['step'], 'epoch': step['epoch'], 'examples': len(step['indices']),
                          'indices_sha256': _digest(step['indices']), **scalars,
                          'gradient_norm_before_clip': float(norm.detach().cpu()), 'actual_legal_candidates': legal,
                          'computation': counts, 'epoch_backbone_hash_checked': epoch_end,
                          'sampled_mps_allocated_bytes_after_step': torch.mps.current_allocated_memory()
                          if settings.device == 'mps' else None,
                          'proposal_index': proposal['proposal_index'] if proposal else None,
                          'proposal_sha256': _digest(proposal) if proposal else None,
                          'permutation': model.square_permutation.cpu().tolist(), 'step_wall_seconds': wall}
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                if epoch_end:
                    os.fsync(stream.fileno())
                    os.fsync(journal.fileno())
        original._sync(settings.device)
        training_seconds = time.perf_counter() - training_started
        _frozen(model, hashes=True)
        _require(source_bindings() == sources, 'Helper sources changed during the fit')
        checkpoint = save_checkpoint(model, out / 'weights.pt', plan_sha256=plan_sha256,
                                     configuration=config, backbone_sha256=model._initial_backbone_sha256)
        receipt = {'status': 'completed', 'version': VERSION, 'configuration': config,
                   'configuration_sha256': _digest(config), 'plan_sha256': plan_sha256,
                   'data_sha256': data_sha256, 'settings': asdict(settings), 'source_sha256': sources,
                   'cache_sha256': cache.metadata['cache_sha256'], 'initial_state_sha256': initial,
                   'backbone_sha256': model._initial_backbone_sha256,
                   'immutable_buffers_sha256': model._initial_immutable_sha256,
                   'graph_sha256': graph_sha256(graph), 'backbone_unchanged': True,
                   'updates': updates, 'examples_seen': settings.epochs * settings.train_examples,
                   'actual_legal_candidates': seen_candidates, 'computation': totals,
                   'proposal_count': proposal_count, 'accepted_proposals': accepted,
                   'proposal_forward_calls': 2 * proposal_count, 'proposal_optimizer_updates': 0,
                   'proposal_examples_seen': 2 * proposal_count * settings.batch_size,
                   'proposal_actual_legal_candidates': proposal_candidates,
                   'proposal_computation': proposal_totals, 'proposal_wall_seconds': proposal_seconds,
                   'step_wall_seconds': step_seconds, 'training_seconds': training_seconds,
                   'fit_wall_seconds': time.perf_counter() - started,
                   'final_permutation': model.square_permutation.cpu().tolist(),
                   'training_timing_scope': 'Full update loop including proposal batch copies, parameter snapshots, mapping validation, synchronization and logging; setup/checkpoint also charged in fit_wall_seconds',
                   'computation_scope': 'Selected forward-operation estimates from architecture MAC/work formulas; backward, optimizer and bookkeeping charged in wall time, not exact total FLOPs',
                   'training_forward_calls': updates, 'backward_calls': updates,
                   'total_forward_calls': updates + 2 * proposal_count,
                   'mapping_administration': 'Explicit CPU discrete checks and transfers; no neural CPU fallback',
                   'proposal_update_consistency': 'Same-batch fixed-weight objective in eval/train modes (no dropout); rel_tol=1e-5, abs_tol=1e-6. Strict acceptance has no tolerance.',
                   'memory_scope': 'MPS allocation sampled after each update, not a measured peak',
                   'mps_fallback_environment': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
                   'parameter_counts': model.parameter_counts(), **checkpoint,
                   'learning_sha256': sha256(out / 'learning.jsonl'),
                   'proposals_sha256': sha256(out / 'proposals.jsonl')}
        _write(out / 'training.json', receipt)
        return receipt
    except BaseException as error:
        try:
            _write(out / 'failed.json', {'status': 'failed', 'version': VERSION, 'configuration': config,
                                        'plan_sha256': plan_sha256, 'updates_completed': updates,
                                        'proposals_completed': proposal_count,
                                        'error_type': type(error).__name__, 'error': str(error),
                                        'wall_seconds': time.perf_counter() - started})
        except BaseException as receipt_error:  # noqa: BLE001 - preserve the original deadline exception
            error.add_note(f'Failed to preserve failure receipt: {receipt_error!r}')
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def _objective(record):
    _require(type(record) is dict and set(record) == {'loss', 'policy_ce', 'value_mse'}
             and all(_number(v) and v >= 0 for v in record.values()), 'Invalid objective scalars')
    loss = float(torch.tensor(record['policy_ce'], dtype=torch.float32)
                 + .5 * torch.tensor(record['value_mse'], dtype=torch.float32))
    _require(_close(record['loss'], loss), 'Objective arithmetic differs')


def _selected_objective(proposal, record):
    """No-dropout actor is identical in train/eval; allow MPS rounding only."""
    chosen = proposal['proposed'] if proposal['accepted'] else proposal['current']
    _require(all(_number(record.get(key)) and math.isclose(record[key], chosen[key], rel_tol=1e-5, abs_tol=1e-6)
                 for key in ('loss', 'policy_ce', 'value_mse')), 'Proposal selected objective differs from training update')


def _counts_equal(actual, expected):
    return (type(actual) is dict and set(actual) == set(expected)
            and all(type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items()))


def _permutation(value):
    _require(type(value) is list and len(value) == 64 and all(type(i) is int for i in value)
             and sorted(value) == list(range(64)), 'Invalid recorded square permutation')
    return list(value)


def _audit_proposal(item, spec, step, permutation, counts, legal, config, plan_sha256, settings):
    keys = {'status', 'policy', 'swap', 'current', 'proposed', 'strictly_improved', 'accepted',
            'forward_calls', 'optimizer_updates', 'evaluated_examples', 'value_weight',
            'starting_permutation', 'proposed_permutation', 'final_permutation', 'wall_seconds',
            'timing_scope', 'neural_device', 'mapping_administration', 'proposal_index', 'before_update',
            'epoch', 'indices_sha256', 'configuration_sha256', 'plan_sha256', 'computation_per_forward',
            'outer_wall_seconds', 'outer_timing_scope', 'actual_legal_candidates'}
    _require(type(item) is dict and set(item) == keys, 'Proposal record schema differs')
    _require(all(item.get(key) == value for key, value in spec.items())
             and type(item['proposal_index']) is int and type(item['before_update']) is int
             and all(type(i) is int for i in item['swap'])
             and item['epoch'] == step['epoch'] and type(item['epoch']) is int
             and item['indices_sha256'] == _digest(step['indices'])
             and item['configuration_sha256'] == _digest(config) and item['plan_sha256'] == plan_sha256,
             'Proposal schedule or identity differs')
    _require(_permutation(item['starting_permutation']) == permutation, 'Proposal chain entry differs')
    proposed = list(permutation)
    first, second = spec['swap']
    proposed[first], proposed[second] = proposed[second], proposed[first]
    _require(_permutation(item['proposed_permutation']) == proposed, 'Proposed swap differs')
    _objective(item['current'])
    _objective(item['proposed'])
    improved = item['proposed']['loss'] < item['current']['loss']
    accepted = config['mapping_policy'] == 'learned' and improved
    expected = proposed if accepted else permutation
    _require(type(item['strictly_improved']) is bool and item['strictly_improved'] == improved
             and type(item['accepted']) is bool and item['accepted'] == accepted
             and _permutation(item['final_permutation']) == expected, 'Strict proposal acceptance differs')
    _require(item['status'] == 'completed_proposal' and item['policy'] == config['mapping_policy']
             and type(item['forward_calls']) is int and item['forward_calls'] == 2
             and type(item['optimizer_updates']) is int and item['optimizer_updates'] == 0
             and type(item['evaluated_examples']) is int and item['evaluated_examples'] == 2 * settings.batch_size
             and _number(item['value_weight']) and item['value_weight'] == .5
             and _counts_equal(item['computation_per_forward'], counts)
             and type(item['actual_legal_candidates']) is int and item['actual_legal_candidates'] == 2 * legal
             and item['neural_device'] in ({'mps', 'mps:0'} if settings.device == 'mps' else {'cpu'}),
             'Proposal computation or objective definition differs')
    _require(_number(item['wall_seconds']) and _number(item['outer_wall_seconds'])
             and 0 <= item['wall_seconds'] <= item['outer_wall_seconds'], 'Invalid proposal wall time')
    for key in ('timing_scope', 'outer_timing_scope', 'mapping_administration'):
        _require(type(item[key]) is str and bool(item[key]), 'Proposal accounting scope missing')
    return list(expected)


def audit_training(directory, config, plan_sha256, data_sha256, cache_metadata, backbone, graph,
                   *, settings=DEFAULT_TRAINING, training_rows):
    """Replay saved discrete decisions and scalar arithmetic; no neural/engine calls."""
    directory = _local(directory)
    config = _configuration(config)
    _hash(plan_sha256)
    _hash(data_sha256)
    _require(isinstance(settings, TrainingSettings), 'Expected mapping TrainingSettings')
    _require({p.name for p in directory.iterdir()} ==
             {'started.json', 'learning.jsonl', 'proposals.jsonl', 'weights.pt', 'training.json'}
             and all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),
             'Training artifact membership differs or fit is incomplete')
    receipt = _read(directory / 'training.json')
    sources = source_bindings()
    _require(_read(directory / 'started.json') == {'status': 'started', 'configuration': config,
             'plan_sha256': plan_sha256, 'settings': asdict(settings), 'source_sha256': sources},
             'Started receipt differs')
    training_rows = list(training_rows)
    _require(len(training_rows) == settings.train_examples, 'Training source length differs')
    audit_cache_metadata(cache_metadata, training_rows, include_successors=False)
    initial = make_model(config, backbone, graph)
    expected = {'status': 'completed', 'version': VERSION, 'configuration': config,
                'configuration_sha256': _digest(config), 'plan_sha256': plan_sha256, 'data_sha256': data_sha256,
                'settings': asdict(settings), 'cache_sha256': cache_metadata['cache_sha256'],
                'source_sha256': sources, 'graph_sha256': graph_sha256(graph),
                'backbone_sha256': state_sha256(backbone), 'backbone_unchanged': True,
                'immutable_buffers_sha256': initial._initial_immutable_sha256,
                'initial_state_sha256': state_sha256(initial), 'parameter_counts': initial.parameter_counts(),
                'updates': settings.updates, 'examples_seen': settings.epochs * settings.train_examples,
                'proposal_count': settings.proposals, 'proposal_forward_calls': 2 * settings.proposals,
                'proposal_optimizer_updates': 0, 'proposal_examples_seen': 2 * settings.proposals * settings.batch_size,
                'training_forward_calls': settings.updates, 'backward_calls': settings.updates,
                'total_forward_calls': settings.updates + 2 * settings.proposals,
                'learning_sha256': sha256(directory / 'learning.jsonl'),
                'proposals_sha256': sha256(directory / 'proposals.jsonl'),
                'checkpoint_sha256': sha256(directory / 'weights.pt')}
    _require(all(receipt.get(key) == value for key, value in expected.items()), 'Training receipt binding differs')
    for key in ('updates', 'examples_seen', 'proposal_count', 'proposal_forward_calls', 'proposal_optimizer_updates',
                'proposal_examples_seen', 'training_forward_calls', 'backward_calls', 'total_forward_calls'):
        _require(type(receipt[key]) is int, 'Malformed computation count')
    _require(receipt['backbone_unchanged'] is True, 'Malformed frozen-backbone status')
    for key in ('training_seconds', 'fit_wall_seconds', 'step_wall_seconds', 'proposal_wall_seconds'):
        _require(_number(receipt.get(key)) and receipt[key] >= 0, 'Invalid training wall time')
    _require(receipt['proposal_wall_seconds'] <= receipt['step_wall_seconds']
             <= receipt['training_seconds'] <= receipt['fit_wall_seconds'], 'Training timing nesting differs')
    if settings.device == 'mps':
        _require(receipt.get('mps_fallback_environment') in (None, '0'), 'Recorded MPS fallback enabled')
    counts = [chess.Board(row['fen']).legal_moves.count() for row in training_rows]
    learning, proposals = _rows_file(directory / 'learning.jsonl'), _rows_file(directory / 'proposals.jsonl')
    steps, specs = schedule(config['seed'], settings), proposal_schedule(config['seed'], settings)
    _require(len(learning) == len(steps) and len(proposals) == len(specs), 'Update or proposal coverage differs')
    proposal_by_step = {spec['before_update']: (item, spec) for item, spec in zip(proposals, specs, strict=True)}
    totals, proposal_totals, legal_total, proposal_legal_total, accepted = {}, {}, 0, 0, 0
    permutation = list(range(64))
    for item, step in zip(learning, steps, strict=True):
        keys = {'step', 'epoch', 'examples', 'indices_sha256', 'policy_ce', 'value_mse', 'loss',
                'gradient_norm_before_clip', 'actual_legal_candidates', 'computation',
                'epoch_backbone_hash_checked', 'sampled_mps_allocated_bytes_after_step',
                'proposal_index', 'proposal_sha256', 'permutation', 'step_wall_seconds'}
        _require(type(item) is dict and set(item) == keys, 'Learning record schema differs')
        indices = step['indices']
        expected_counts = initial.computation_counts(len(indices), max(counts[i] for i in indices))
        legal = sum(counts[i] for i in indices)
        _require(type(item['step']) is int and item['step'] == step['step']
                 and type(item['epoch']) is int and item['epoch'] == step['epoch']
                 and type(item['examples']) is int and item['examples'] == len(indices)
                 and item['indices_sha256'] == _digest(indices)
                 and type(item['actual_legal_candidates']) is int and item['actual_legal_candidates'] == legal
                 and _counts_equal(item['computation'], expected_counts), 'Training schedule or computation differs')
        _objective({key: item[key] for key in ('loss', 'policy_ce', 'value_mse')})
        _require(_number(item['gradient_norm_before_clip']) and item['gradient_norm_before_clip'] >= 0
                 and _number(item['step_wall_seconds']) and item['step_wall_seconds'] >= 0,
                 'Invalid learning gradient or timing')
        if step['step'] in proposal_by_step:
            proposal, spec = proposal_by_step[step['step']]
            permutation = _audit_proposal(proposal, spec, step, permutation, expected_counts, legal,
                                          config, plan_sha256, settings)
            _selected_objective(proposal, item)
            _require(type(item['proposal_index']) is int and item['proposal_index'] == spec['proposal_index']
                     and item['proposal_sha256'] == _digest(proposal)
                     and proposal['outer_wall_seconds'] <= item['step_wall_seconds'],
                     'Proposal/update linkage differs')
            accepted += int(proposal['accepted'])
            proposal_legal_total += 2 * legal
            original._add_counts(proposal_totals, expected_counts)
            original._add_counts(proposal_totals, expected_counts)
        else:
            _require(item['proposal_index'] is None and item['proposal_sha256'] is None,
                     'Unscheduled proposal present')
        _require(_permutation(item['permutation']) == permutation, 'Update mapping chain differs')
        epoch_end = step['step'] % (settings.train_examples // settings.batch_size) == 0
        _require(type(item['epoch_backbone_hash_checked']) is bool
                 and item['epoch_backbone_hash_checked'] == epoch_end, 'Epoch frozen hash checks differ')
        memory = item['sampled_mps_allocated_bytes_after_step']
        _require((memory is None and settings.device == 'cpu') or
                 (settings.device == 'mps' and type(memory) is int and memory >= 0), 'Invalid memory sample')
        legal_total += legal
        original._add_counts(totals, expected_counts)
    _require(_counts_equal(receipt.get('computation'), totals)
             and _counts_equal(receipt.get('proposal_computation'), proposal_totals)
             and type(receipt.get('accepted_proposals')) is int and receipt['accepted_proposals'] == accepted
             and type(receipt.get('actual_legal_candidates')) is int
             and receipt['actual_legal_candidates'] == legal_total
             and type(receipt.get('proposal_actual_legal_candidates')) is int
             and receipt['proposal_actual_legal_candidates'] == proposal_legal_total
             and _permutation(receipt.get('final_permutation')) == permutation,
             'Training aggregate computation or mapping differs')
    _require(_close(receipt['proposal_wall_seconds'], math.fsum(p['outer_wall_seconds'] for p in proposals))
             and _close(receipt['step_wall_seconds'], math.fsum(p['step_wall_seconds'] for p in learning)),
             'Training timing aggregates differ')
    model = load_checkpoint(directory / 'weights.pt', backbone, graph, expected_plan_sha256=plan_sha256,
                            expected_config=config, expected_backbone_sha256=expected['backbone_sha256'],
                            expected_checkpoint_sha256=expected['checkpoint_sha256'])
    _require(state_sha256(model) == receipt.get('state_sha256')
             and model.square_permutation.tolist() == permutation, 'Final training state or mapping differs')
    return receipt
