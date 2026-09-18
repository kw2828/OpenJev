"""Bounded hard-interface search on frozen chess backbones and exposed panels.

Thirty final fits precede all evaluation. Graph-derived weights stay in runs/.
The separate audit reads saved score vectors and journals without inference.
"""

import argparse
import dataclasses
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

from openjev.research import chess_connectome_data as data
from openjev.research import chess_connectome_study as evaluation
from openjev.research.chess_candidate_eval import CachedPositions, audit_cache_metadata

ROOT = Path(__file__).resolve().parents[1]
OLD_PLAN = 'evidence/chess-connectome-v1/protocol/plan.json'
OLD_EXECUTION = 'runs/chess-connectome-v1/execution'
OLD_SUMMARY = 'evidence/chess-connectome-v1/results/summary.json'
OLD_PLAN_SHA = '5543877d88d2c00b7085c93f2f01602f2d0b3439486b05661d1a8511af7f81db'
OLD_EXECUTION_SHA = '07948f91fc6dfc2aa9cca065a69a4637a0569ac5bca6c9c10aa0b68afbe290fd'
OLD_SUMMARY_SHA = '7c96b00490b7af01e630c647cb63a1d7a99f084d326edd4809d4e89ee983eae0'
PROFILE = 'evidence/chess-connectome-mapping-preflight-v1/profile.json'
CACHE_TIMINGS = {'native_construction_wall_seconds', 'fingerprinting_wall_seconds', 'total_wall_seconds'}
CACHE_FIELDS = {'version', 'encoding', 'input_rows_sha256', 'labels_sha256', 'include_successors', 'tensors',
                'cache_sha256', 'positions', 'legal_candidates', 'root_encodings', 'native_successors',
                'native_board_copies', 'native_pushes', 'successor_encodings', 'cpu_tensor_bytes', 'byte_scope',
                'construction_scope', 'source_sha256', 'encoding_source_sha256'} | CACHE_TIMINGS
NEW_SOURCES = [
    'scripts/chess_connectome_mapping_study.py', 'tests/test_chess_connectome_mapping_runner.py',
    'src/openjev/research/chess_connectome_mapping_study.py', 'tests/test_chess_connectome_mapping_study.py',
    'src/openjev/research/chess_connectome_interface.py', 'tests/test_chess_connectome_interface.py',
    'scripts/chess_connectome_mapping_profile.py', 'tests/test_chess_connectome_mapping_profile.py',
    'scripts/chess_compute_study.py', 'tests/test_chess_compute_study.py',
]
PROTOCOL = {
    'version': 'chess-connectome-mapping-v1', 'paired_seeds': [97, 109, 127],
    'variants': ['biological', 'rewire151', 'rewire163', 'rewire179', 'node_local'],
    'mapping_policies': ['fixed', 'learned'], 'fit_order_seed': 11600017,
    'train_examples': 32768, 'epochs': 6, 'updates_per_fit': 1536,
    'warmup_updates': 256, 'proposal_interval': 4, 'proposal_seed_base': 11600041,
    'proposals_per_fit': 320, 'all_final_fits_before_neural_evaluation': True,
    'torch_threads': 2, 'deterministic_algorithms': False, 'dtype': 'float32',
    'primary_wall_seconds': 7200, 'audit_wall_seconds': 1800,
    'regret_positions_per_split': 128, 'regret_nodes': 20000,
    'regret_call_ceiling': 8704, 'regret_node_ceiling': 174080000,
    'latency_positions_per_split': 64, 'latency_warmups': 3,
    'relative_loss_reduction': .10, 'gate_absolute_tolerance': 1e-12,
    'gate': 'Biological-learned mean signed bounded loss at least10% below biological-fixed and each learned '
            'rewire on BOTH panels, with positive reference means. Strictly better in EVERY paired seed than '
            'each learned rewire and no worse than same-seed direct. The mean fixed-minus-learned gain must '
            'exceed each rewire and node_local gain on BOTH panels. All30 fits and all33 evaluations required.',
    'matching': 'Five topologies times fixed/learned interfaces times three paired seeds. Same32768 roots, '
                'sixepochs,1536updates, paired batches and320proposal pairs. Fixed mode performs both proposal '
                'forwards but never accepts. Continuous adapter training and backbone are otherwise unchanged.',
    'mapping': 'Warmup256 optimizer updates; before updates257,261,...,1533 propose one paired-seed hard '
               'square swap. Exactly two no-gradient training-batch objectives per proposal. Learned mode '
               'accepts strict finite improvement; fixed mode retains entry mapping. No evaluation labels.',
    'data': 'Original authenticated connectome-v1 training roots and evaluation panels; original secondary '
            'and latency indices. No new teacher data. These are exposed historical development panels.',
    'pretraining': 'Three published direct candidate-v2 backbones reused, with original pretraining costs '
                   'and prior attempts retained. No low-data or sample-efficiency claim.',
    'selection': 'Final-only checkpoints and all fixed seeds; no retries, replacement fits, resume, '
                 'best-seed selection, extension or evaluation before all30 fits complete.',
    'timing': 'Measured shared-host training/proposal and full CPU decision wall time. Concurrent independent '
              'pin training is permitted and may contend for resources. Three separately charged warmups '
              'and fixed64 positions per panel. Host load recorded; no isolated speed claim.',
    'scope': 'Exposed development mechanism screen for a hard learned interface. Not an intact fly, recurrent '
             'world model, search policy, calibrated confidence, architectural novelty, gameplay or Elo result.',
    'arena': 'No gameplay in this screen. A passed all-control gate permits a separately frozen game followup.',
    'asset_policy': 'Biological arrays, rewires and derived checkpoint weights remain local under runs/.',
    'audit_scope': 'Saved-output source, cache, full-score arithmetic, checkpoint, training/proposal journal '
                   'and compute audit with no new model or engine calls. No proof of optimizer execution '
                   'or proposal objective values without inference.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected regular nonsymlink file')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'Nonfinite JSON: {value}')

    def floating(value):
        result = float(value)
        require(math.isfinite(result), 'Nonfinite JSON numeric overflow')
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def read(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Expected regular nonsymlink JSON file')
    return parse(Path(path).read_text())


def rows(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Expected regular nonsymlink journal')
    return [parse(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tree(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'Regular evidence directory required')
    result = {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Nonregular evidence member')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    require(spec is not None and spec.loader is not None, 'Required module cannot be imported')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def original():
    return module('scripts/chess_connectome_study.py', '_mapping_original_connectome_runner')


def helper():
    return importlib.import_module('openjev.research.chess_connectome_mapping_study')


def grader():
    result = module('scripts/chess_compute_study.py', '_mapping_original_secondary_grader')
    expected = {'regret_nodes': 20000, 'value_cp_scale': 600., 'mate_cp': 10000,
                'engine_threads': 1, 'engine_hash_mb': 16}
    require(all(result.PROTOCOL[k] == v for k, v in expected.items()), 'Inherited grading semantics differ')
    return result


def configurations():
    result = helper().configurations()
    random.Random(PROTOCOL['fit_order_seed']).shuffle(result)
    expected = {(v, p, s) for v in PROTOCOL['variants'] for p in PROTOCOL['mapping_policies']
                for s in PROTOCOL['paired_seeds']}
    require(len(result) == 30 and {(c['variant'], c['mapping_policy'], c['seed']) for c in result} == expected
            and all(c['name'] == f"{c['variant']}-{c['mapping_policy']}-{c['seed']}" for c in result),
            'Mapping configuration coverage differs')
    return result


def baselines():
    return evaluation.baseline_configurations()


def legacy_config(config):
    return config if config['variant'] == 'direct' else helper().legacy_config(config)


def prerequisites():
    bindings = {OLD_PLAN: OLD_PLAN_SHA, f'{OLD_EXECUTION}/completed.json': OLD_EXECUTION_SHA,
                OLD_SUMMARY: OLD_SUMMARY_SHA}
    require(all(sha(ROOT/p) == h for p, h in bindings.items()), 'Original prerequisite hash changed')
    old = original()
    previous, excluded = old.verify(ROOT/OLD_PLAN)
    old.completed(ROOT/OLD_EXECUTION, OLD_PLAN_SHA)
    old.artifact_membership(ROOT/OLD_EXECUTION, previous)
    old.completed((ROOT/OLD_SUMMARY).parent, OLD_PLAN_SHA)
    summary = read(ROOT/OLD_SUMMARY)
    require(summary['status'] == 'completed' and summary['plan_sha256'] == OLD_PLAN_SHA
            and summary['execution_receipt_sha256'] == OLD_EXECUTION_SHA
            and summary['report_new_model_or_engine_calls'] == 0, 'Original completed audit identity differs')
    panels = data.validate(ROOT/OLD_EXECUTION/'data', excluded)
    require(set(panels) == {'dev', 'shift'} and all(len(r) == 2048 for r in panels.values()),
            'Original panel coverage differs')
    selection, _ = old.panels(panels)
    require(read(ROOT/OLD_EXECUTION/'panels.json') == selection, 'Original secondary or timing indices changed')
    inputs = {'pinned_files': bindings, 'data_files': old.tree(ROOT/OLD_EXECUTION/'data'),
              'selection_sha256': sha(ROOT/OLD_EXECUTION/'panels.json'),
              'training_cache_sha256': sha(ROOT/OLD_EXECUTION/'training-cache.json'),
              'evaluation_cache_sha256': sha(ROOT/OLD_EXECUTION/'evaluation-cache.json')}
    return previous, panels, selection, inputs


def signature():
    previous, panels, selection, inputs = prerequisites()
    settings = helper().TrainingSettings()
    require(settings.train_examples == PROTOCOL['train_examples'] and settings.epochs == PROTOCOL['epochs']
            and settings.updates == PROTOCOL['updates_per_fit']
            and settings.warmup_updates == PROTOCOL['warmup_updates']
            and settings.proposal_interval == PROTOCOL['proposal_interval']
            and settings.proposal_seed_base == PROTOCOL['proposal_seed_base'] and settings.device == 'mps',
            'Helper training settings differ')
    for seed in PROTOCOL['paired_seeds']:
        proposals = helper().proposal_schedule(seed, settings)
        require(len(proposals) == PROTOCOL['proposals_per_fit']
                and [p['before_update'] for p in proposals] == list(range(257, 1537, 4)), 'Proposal schedule differs')
    profile = read(ROOT/PROFILE)
    profile_receipt_path = (ROOT/PROFILE).parent/'receipt.json'
    profile_receipt = read(profile_receipt_path)
    require(profile['status'] == 'completed' and profile['synthetic_only'] is True
            and type(profile['real_training_rows']) is int and profile['real_training_rows'] == 0,
            'Completed synthetic-only preflight required')
    require(profile_receipt['status'] == 'completed' and profile_receipt['profile_sha256'] == sha(ROOT/PROFILE)
            and not ((ROOT/PROFILE).parent/'failed.json').exists()
            and bool(profile_receipt['sources'])
            and all(not Path(p).is_absolute() and '..' not in Path(p).parts and sha(ROOT/p) == h
                    for p, h in profile_receipt['sources'].items()), 'Preflight receipt or sources changed')
    sources = {**previous['binding']['sources'], **{p: sha(ROOT/p) for p in NEW_SOURCES}}
    plan = {'protocol': PROTOCOL, 'training_settings': dataclasses.asdict(settings),
        'configurations': configurations(), 'baselines': baselines(), 'binding': previous['binding'],
        'original_inputs': inputs, 'selection': selection, 'sources': dict(sorted(sources.items())),
        'profile': {'path': PROFILE, 'sha256': sha(ROOT/PROFILE),
                    'receipt_sha256': sha(profile_receipt_path)},
        'environment': {**original().environment(), 'mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')},
        'training_or_new_evaluation_started': False}
    return plan, panels


def prepare(out):
    out = Path(out)
    require(not out.exists() and not out.is_symlink(), 'Protocol output already exists')
    plan, _ = signature()
    require(plan['environment']['mps_available'] is True and plan['environment']['mps_fallback'] == '0',
            'MPS required with fallback disabled')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    write(out/'prepared.json', {'status': 'prepared', 'plan_sha256': sha(out/'plan.json'),
        'created_unix': time.time(), 'training_or_new_evaluation_started': False})
    return {'path': str(out/'plan.json'), 'sha256': sha(out/'plan.json'), 'fits': 30, 'proposals_per_fit': 320}


def verify(plan_path):
    plan, prepared = read(plan_path), read(Path(plan_path).parent/'prepared.json')
    require(prepared['status'] == 'prepared' and prepared['plan_sha256'] == sha(plan_path)
            and prepared['training_or_new_evaluation_started'] is False, 'Prepared plan identity changed')
    expected, panels = signature()
    require(plan == expected, 'Frozen protocol, sources, environment, inputs or profile changed')
    require(plan['environment']['mps_available'] is True and plan['environment']['mps_fallback'] == '0',
            'MPS unavailable or fallback enabled')
    return plan, panels


class DeadlineExceeded(BaseException):
    """Escape inherited ordinary-exception handlers when a phase budget expires."""


def deadline(seconds):
    def expired(_signum, _frame):
        raise DeadlineExceeded('Frozen wall budget exhausted; no retry or extension')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)


def private_output(out):
    out = Path(out).resolve()
    require(out.is_relative_to((ROOT/'runs').resolve()), 'Graph-derived artifacts must remain in local runs/')
    return out


def validate_cache_identity(current, previous, source_rows):
    """Rebuilt bytes/labels must match; measured construction times are new work."""
    require(type(current) is dict and type(previous) is dict
            and set(current) == set(previous) == CACHE_FIELDS, 'Cache receipt schema differs')
    audit_cache_metadata(current, source_rows, include_successors=False)
    audit_cache_metadata(previous, source_rows, include_successors=False)
    require({k: v for k, v in current.items() if k not in CACHE_TIMINGS}
            == {k: v for k, v in previous.items() if k not in CACHE_TIMINGS},
            'Original deterministic cache identity differs')


def completed(directory, plan_hash, *, terminal='completed.json'):
    actual = tree(directory)
    require('failed.json' not in actual and terminal in actual, 'Incomplete or failed output')
    receipt = read(Path(directory)/terminal)
    actual.pop(terminal)
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == plan_hash
            and receipt['files'] == actual, 'Completed artifact manifest differs')
    return receipt


def artifact_membership(execution, plan):
    execution = Path(execution)
    expected = {'started.json', 'data-source.json', 'training-cache.json', 'panels.json', 'fits',
                'all-fits-completed.json', 'evaluation-started.json', 'evaluation-cache.json',
                'evaluation', 'decisions.json', 'regret', 'completed.json'}
    require({p.name for p in execution.iterdir()} == expected, 'Execution artifact membership differs')
    for subdir, configs in (('fits', plan['configurations']), ('evaluation', plan['baselines']+plan['configurations'])):
        directory = execution/subdir
        require({p.name for p in directory.iterdir()} == {c['name'] for c in configs}
                and all(p.is_dir() and not p.is_symlink() for p in directory.iterdir()),
                'Fit/evaluation configuration membership differs')
        members = ({'started.json', 'learning.jsonl', 'proposals.jsonl', 'weights.pt', 'training.json'}
                   if subdir == 'fits' else {'dev.jsonl', 'shift.jsonl', 'latency.json', 'completed.json'})
        for config in configs:
            require({p.name for p in (directory/config['name']).iterdir()} == members,
                    'Fit/evaluation file coverage differs')


def grading_plan(plan, panels):
    return {'engine_path': str((ROOT/plan['binding']['engine']).resolve()), 'panels': panels,
        'secondary_indices': plan['selection']['secondary_indices'],
        'configurations': [{'id': c['name']} for c in plan['baselines']+plan['configurations']]}


def grading_inputs(plan, decisions):
    names = [c['id'] for c in plan['configurations']]
    expected_names = {c['name'] for c in baselines()+configurations()}
    require(len(names) == len(set(names)) == 33 and set(names) == expected_names, 'All33 grading models required')
    positions, legal = {}, {}
    for split in ('dev', 'shift'):
        indices = plan['secondary_indices'][split]
        require(len(indices) == len(set(indices)) == 128
                and all(type(i) is int and 0 <= i < len(plan['panels'][split]) for i in indices),
                'Invalid secondary selection')
        for i, row in enumerate(plan['panels'][split]):
            board = chess.Board(row['fen'])
            require(board.is_valid() and not board.is_game_over(claim_draw=False), 'Invalid grading root')
            positions[split, i] = row['id']
            legal[split, i] = {m.uci() for m in board.legal_moves}
    expected = {(name, split, i) for name in names for split, i in positions}
    seen = {}
    for row in decisions:
        key = row['configuration'], row['split'], row['panel_index']
        require(set(row) == {'configuration', 'split', 'panel_index', 'id', 'choice'}
                and type(key[2]) is int and key in expected and key not in seen
                and row['id'] == positions[key[1:]] and row['choice'] in legal[key[1:]],
                'Duplicate, missing, illegal or unbound grading choice')
        seen[key] = row['choice']
    require(set(seen) == expected, 'Full choice coverage required')
    calls = sum(1+len({seen[name, split, i] for name in names})
                for split in ('dev', 'shift') for i in plan['secondary_indices'][split])
    require(calls <= PROTOCOL['regret_call_ceiling'] and calls*20000 <= PROTOCOL['regret_node_ceiling'],
            'Planned grading budget exceeded')
    return calls


def grading_binding(plan, decisions, plan_hash, engine_hash):
    require(valid_hash(plan_hash) and valid_hash(engine_hash), 'Invalid grading provenance hash')
    require(sha(plan['engine_path']) == engine_hash, 'Engine binary changed')
    return {'plan_sha256': plan_hash, 'grading_plan_sha256': digest(plan), 'decisions_sha256': digest(decisions),
            'expected_engine_sha256': engine_hash, 'planned_unique_calls': grading_inputs(plan, decisions),
            'call_ceiling': PROTOCOL['regret_call_ceiling'], 'requested_node_ceiling': PROTOCOL['regret_node_ceiling'],
            'source_sha256': sha(__file__), 'grader_source_sha256': sha(ROOT/'scripts/chess_compute_study.py')}


def audit_grading(plan, directory, decisions, *, plan_hash, engine_hash):
    directory = Path(directory)
    receipt = completed(directory, plan_hash)
    require({p.name for p in directory.iterdir()} ==
            {'started.json', 'engine.json', 'analyses.jsonl', 'regret.jsonl', 'completed.json'},
            'Grading artifact coverage differs')
    binding = grading_binding(plan, decisions, plan_hash, engine_hash)
    require(read(directory/'started.json') == {'status': 'started', **binding}
            and all(receipt[k] == v for k, v in binding.items()), 'Grading provenance differs')
    engine = read(directory/'engine.json')
    require(engine['path'] == plan['engine_path'] and engine['sha256'] == engine_hash
            and engine['id']['name'].startswith('Stockfish 19'), 'Grading engine identity differs')
    analyses, records, cost = rows(directory/'analyses.jsonl'), rows(directory/'regret.jsonl'), receipt['cost']
    require(set(cost) == {'calls', 'requested_nodes', 'reported_nodes', 'wall_seconds'}
            and all(type(cost[k]) is int and cost[k] >= 0 for k in ('calls', 'requested_nodes', 'reported_nodes'))
            and number(cost['wall_seconds']) and number(receipt['grading_wall_seconds']), 'Invalid grading cost')
    for row in analyses:
        require(set(row) == {'split', 'panel_index', 'id', 'root_move', 'pv_first', 'score_cp', 'mate',
                             'bounded_score', 'requested_nodes', 'reported_nodes', 'wall_seconds'}
                and type(row['panel_index']) is int and type(row['score_cp']) is int
                and (row['mate'] is None or type(row['mate']) is int)
                and number(row['bounded_score'], -1) and row['bounded_score'] <= 1
                and type(row['requested_nodes']) is int and row['requested_nodes'] == 20000
                and type(row['reported_nodes']) is int and row['reported_nodes'] >= 0
                and number(row['wall_seconds']), 'Invalid analysis record')
    for row in records:
        require(set(row) == {'configuration', 'split', 'panel_index', 'id', 'choice', 'bounded_regret', 'cp_loss'}
                and type(row['panel_index']) is int and type(row['cp_loss']) is int
                and number(row['bounded_regret'], -2) and row['bounded_regret'] <= 2, 'Invalid signed loss record')
    grader().validate_secondary(plan, decisions, analyses, records, cost)
    require(cost['calls'] == binding['planned_unique_calls'] and cost['requested_nodes'] == cost['calls']*20000,
            'Grading unique-call accounting differs')
    return receipt


def score_grading(plan, directory, decisions, *, plan_hash, engine_hash):
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    begun = time.perf_counter()
    try:
        binding = grading_binding(plan, decisions, plan_hash, engine_hash)
        write(directory/'started.json', {'status': 'started', **binding})
        cost = grader().score_secondary(plan, directory, decisions)
        write(directory/'completed.json', {'status': 'completed', **binding, 'cost': cost,
            'grading_wall_seconds': time.perf_counter()-begun, 'files': tree(directory)})
        return audit_grading(plan, directory, decisions, plan_hash=plan_hash, engine_hash=engine_hash)
    except BaseException as exc:
        write(directory/'failed.json', {'status': 'failed', 'plan_sha256': plan_hash,
            'error_type': type(exc).__name__, 'error': str(exc), 'wall_seconds': time.perf_counter()-begun})
        raise


def load_model(config, plan, execution, fit_records, topology, plan_hash):
    backbone = original().load_backbone(plan, config['seed'])
    if config['variant'] == 'direct':
        return backbone
    fit = next(f for f in fit_records if f['configuration'] == config)
    model = helper().load_checkpoint(Path(execution)/'fits'/config['name']/'weights.pt',
        backbone, topology[config['variant']], expected_plan_sha256=plan_hash, expected_config=config,
        expected_backbone_sha256=evaluation.state_sha256(backbone), expected_checkpoint_sha256=fit['weights_sha256'])
    require(evaluation.state_sha256(model) == fit['state_sha256'], 'Loaded model differs from fit-boundary state')
    return model


def run(plan_path, out):
    out = private_output(out)
    out.mkdir(parents=True, exist_ok=False)
    begun, plan_hash = time.perf_counter(), sha(plan_path)
    write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'started_unix': time.time()})
    deadline(PROTOCOL['primary_wall_seconds'])
    try:
        plan, panels = verify(plan_path)
        torch.set_num_threads(PROTOCOL['torch_threads'])
        torch.use_deterministic_algorithms(PROTOCOL['deterministic_algorithms'])
        old = original()
        topology = old.graphs(plan['binding']['graph_hashes'])
        write(out/'data-source.json', plan['original_inputs'])
        training = data.training_rows(ROOT)
        cache = CachedPositions(training, include_successors=False)
        validate_cache_identity(cache.metadata, read(ROOT/OLD_EXECUTION/'training-cache.json'), training)
        write(out/'training-cache.json', cache.metadata)
        write(out/'panels.json', plan['selection'])
        (out/'fits').mkdir()
        fit_records = []
        for config in plan['configurations']:
            identity = config['name']
            print(json.dumps({'fit_started': identity, 'unix': time.time()}), flush=True)
            receipt = helper().fit(config, old.load_backbone(plan, config['seed']), topology[config['variant']],
                cache, out/'fits'/identity, plan_sha256=plan_hash,
                data_sha256=plan['binding']['training_source_sha256'], settings=helper().TrainingSettings())
            require(receipt['status'] == 'completed' and receipt['configuration'] == config
                    and receipt['plan_sha256'] == plan_hash and valid_hash(receipt['state_sha256']),
                    'Fit returned an unbound terminal receipt')
            fit_records.append({'configuration': config, 'training_sha256': sha(out/'fits'/identity/'training.json'),
                'weights_sha256': sha(out/'fits'/identity/'weights.pt'), 'state_sha256': receipt['state_sha256']})
            print(json.dumps({'fit_completed': identity, 'completed': len(fit_records)}), flush=True)
            gc.collect()
            torch.mps.empty_cache()
        write(out/'all-fits-completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'fits': fit_records, 'completed_unix': time.time(), 'evaluation_started': False})
        del cache, training
        gc.collect()
        write(out/'evaluation-started.json', {'status': 'started', 'plan_sha256': plan_hash,
            'completed_fits': len(fit_records), 'started_unix': time.time(),
            'fit_boundary_sha256': sha(out/'all-fits-completed.json')})
        prepared = {s: CachedPositions(r, include_successors=False) for s, r in panels.items()}
        metadata = {s: c.metadata for s, c in prepared.items()}
        previous_cache = read(ROOT/OLD_EXECUTION/'evaluation-cache.json')
        require(set(metadata) == set(previous_cache) == {'dev', 'shift'}, 'Evaluation cache panels differ')
        for split in ('dev', 'shift'):
            validate_cache_identity(metadata[split], previous_cache[split], panels[split])
        write(out/'evaluation-cache.json', metadata)
        combined = panels['dev']+panels['shift']
        (out/'evaluation').mkdir()
        decisions = []
        for config in plan['baselines']+plan['configurations']:
            identity = config['name']
            model = load_model(config, plan, out, fit_records, topology, plan_hash)
            state = evaluation.state_sha256(model)
            directory = out/'evaluation'/identity
            directory.mkdir()
            measured = {}
            for split in ('dev', 'shift'):
                measured[split] = evaluation.evaluate(model, prepared[split], directory/f'{split}.jsonl',
                    configuration=legacy_config(config), plan_sha256=plan_hash)
                decisions.extend({'configuration': identity, 'split': split, 'panel_index': i,
                    'id': row['id'], 'choice': row['choice']} for i, row in enumerate(rows(directory/f'{split}.jsonl')))
            host_before = list(os.getloadavg())
            write(directory/'latency.json', evaluation.measure_latency(model, combined, plan['selection']['latency_indices'],
                configuration=legacy_config(config), plan_sha256=plan_hash, warmups=PROTOCOL['latency_warmups']))
            require(evaluation.state_sha256(model) == state, 'Evaluation mutated model state')
            write(directory/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                'configuration': config, 'legacy_configuration': legacy_config(config), 'state_sha256': state,
                'evaluation': measured, 'host_load_average_before_latency': host_before,
                'host_load_average_after_latency': list(os.getloadavg()), 'files': tree(directory)})
            print(json.dumps({'evaluated': identity}), flush=True)
            del model
            gc.collect()
        write(out/'decisions.json', decisions)
        score_grading(grading_plan(plan, panels), out/'regret', decisions,
                      plan_hash=plan_hash, engine_hash=plan['binding']['engine_sha256'])
        elapsed = time.perf_counter()-begun
        require(number(elapsed) and elapsed <= PROTOCOL['primary_wall_seconds'], 'Primary wall budget exceeded')
        write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'wall_seconds': elapsed, 'files': tree(out)})
        return {'status': 'completed', 'execution': str(out)}
    except BaseException as exc:
        write(out/'failed.json', {'status': 'failed', 'plan_sha256': plan_hash, 'error_type': type(exc).__name__,
            'error': str(exc), 'wall_seconds': time.perf_counter()-begun})
        raise
    finally:
        signal.alarm(0)


def continuation(losses):
    expected = {c['name'] for c in baselines()+configurations()}
    require(set(losses) == expected and all(set(p) == {'dev', 'shift'} for p in losses.values()),
            'All33 configurations and both panels required')
    require(all(number(v, -2) and v <= 2 for p in losses.values() for v in p.values()), 'Invalid signed bounded loss')
    seeds, tol = PROTOCOL['paired_seeds'], PROTOCOL['gate_absolute_tolerance']
    arms = ['direct']+[f'{v}-{p}' for v in PROTOCOL['variants'] for p in PROTOCOL['mapping_policies']]
    means = {arm: {s: math.fsum(losses[f'{arm}-{seed}'][s] for seed in seeds)/3 for s in ('dev', 'shift')}
             for arm in arms}
    checks = []
    for split in ('dev', 'shift'):
        biology = means['biological-learned'][split]
        for reference_name in ('biological-fixed', 'rewire151-learned', 'rewire163-learned', 'rewire179-learned'):
            reference = means[reference_name][split]
            reduction = (reference-biology)/reference if reference > 0 else None
            checks.append({'split': split, 'comparator': reference_name, 'criterion': 'mean_relative_reduction',
                'reference': reference, 'treatment': biology, 'reduction': reduction,
                'passed': reduction is not None and reduction+tol >= PROTOCOL['relative_loss_reduction']})
        for variant in ('rewire151', 'rewire163', 'rewire179'):
            for seed in seeds:
                difference = losses[f'biological-learned-{seed}'][split]-losses[f'{variant}-learned-{seed}'][split]
                checks.append({'split': split, 'comparator': f'{variant}-learned', 'seed': seed,
                    'criterion': 'strict_paired_improvement', 'difference': difference, 'passed': difference < -tol})
        for seed in seeds:
            difference = losses[f'biological-learned-{seed}'][split]-losses[f'direct-{seed}'][split]
            checks.append({'split': split, 'comparator': 'direct', 'seed': seed,
                'criterion': 'no_paired_degradation', 'difference': difference, 'passed': difference <= tol})
        biological_gain = means['biological-fixed'][split]-biology
        for variant in ('rewire151', 'rewire163', 'rewire179', 'node_local'):
            comparator_gain = means[f'{variant}-fixed'][split]-means[f'{variant}-learned'][split]
            difference = biological_gain-comparator_gain
            checks.append({'split': split, 'comparator': variant, 'criterion': 'mapping_difference_in_differences',
                'biological_gain': biological_gain, 'comparator_gain': comparator_gain,
                'difference': difference, 'passed': difference > tol})
    return {'means': means, 'checks': checks, 'continuation_passed': all(c['passed'] for c in checks),
            'scope': PROTOCOL['scope']}


def engine_metrics(records, configs):
    result = {}
    for config in configs:
        name = config['name']
        result[name] = {}
        for split in ('dev', 'shift'):
            selected = [r for r in records if r['configuration'] == name and r['split'] == split]
            require(len(selected) == PROTOCOL['regret_positions_per_split'], 'Engine summary coverage differs')
            raw = [r['cp_loss'] for r in selected]
            require(all(type(v) is int for v in raw), 'Raw centipawn losses must be integers')
            result[name][split] = {'positions': len(selected),
                'mean_signed_bounded_loss': math.fsum(r['bounded_regret'] for r in selected)/len(selected),
                'mean_cp_loss': math.fsum(raw)/len(raw), 'p95_cp_loss': float(np.quantile(raw, .95)),
                'max_cp_loss': max(raw)}
    return result


def audit(plan_path, execution, out):
    out, execution = Path(out), private_output(execution)
    out.mkdir(parents=True, exist_ok=False)
    begun, plan_hash = time.perf_counter(), sha(plan_path)
    write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'started_unix': time.time()})
    deadline(PROTOCOL['audit_wall_seconds'])
    try:
        plan, panels = verify(plan_path)
        finished = completed(execution, plan_hash)
        require(number(finished['wall_seconds']) and finished['wall_seconds'] <= PROTOCOL['primary_wall_seconds'],
                'Execution exceeded the frozen wall budget')
        artifact_membership(execution, plan)
        started, boundary, eval_start = (read(execution/p) for p in
            ('started.json', 'all-fits-completed.json', 'evaluation-started.json'))
        require(started['status'] == 'started' and started['plan_sha256'] == plan_hash
                and number(started['started_unix']) and boundary['status'] == 'completed'
                and boundary['plan_sha256'] == plan_hash and boundary['evaluation_started'] is False
                and number(boundary['completed_unix']) and boundary['completed_unix'] >= started['started_unix']
                and [f['configuration'] for f in boundary['fits']] == plan['configurations'],
                'All-final-fits boundary differs')
        require(eval_start['status'] == 'started' and eval_start['plan_sha256'] == plan_hash
                and type(eval_start['completed_fits']) is int and eval_start['completed_fits'] == 30
                and eval_start['fit_boundary_sha256'] == sha(execution/'all-fits-completed.json')
                and number(eval_start['started_unix']) and eval_start['started_unix'] >= boundary['completed_unix'],
                'Evaluation preceded complete fitting')
        require(read(execution/'data-source.json') == plan['original_inputs']
                and read(execution/'panels.json') == plan['selection'], 'Original data or panel selection differs')
        training = data.training_rows(ROOT)
        train_cache, eval_cache = read(execution/'training-cache.json'), read(execution/'evaluation-cache.json')
        validate_cache_identity(train_cache, read(ROOT/OLD_EXECUTION/'training-cache.json'), training)
        previous_cache = read(ROOT/OLD_EXECUTION/'evaluation-cache.json')
        require(set(eval_cache) == set(previous_cache) == {'dev', 'shift'}, 'Evaluation cache panels differ')
        for split in ('dev', 'shift'):
            validate_cache_identity(eval_cache[split], previous_cache[split], panels[split])
        topology = original().graphs(plan['binding']['graph_hashes'])
        fit_costs, metrics, latency, decisions = {}, {}, {}, []
        for config, fit_record in zip(plan['configurations'], boundary['fits'], strict=True):
            name = config['name']
            directory = execution/'fits'/name
            require(fit_record['training_sha256'] == sha(directory/'training.json')
                    and fit_record['weights_sha256'] == sha(directory/'weights.pt'), 'Fit changed after boundary')
            backbone = original().load_backbone(plan, config['seed'])
            fit_costs[name] = helper().audit_training(directory, config, plan_hash,
                plan['binding']['training_source_sha256'], train_cache, backbone, topology[config['variant']],
                training_rows=training, settings=helper().TrainingSettings())
            require(fit_costs[name]['state_sha256'] == fit_record['state_sha256'], 'Final model state differs')
            del backbone
        combined = panels['dev']+panels['shift']
        for config in plan['baselines']+plan['configurations']:
            name = config['name']
            directory = execution/'evaluation'/name
            receipt = completed(directory, plan_hash)
            model = load_model(config, plan, execution, boundary['fits'], topology, plan_hash)
            state = evaluation.state_sha256(model)
            require(receipt['configuration'] == config and receipt['legacy_configuration'] == legacy_config(config)
                    and receipt['state_sha256'] == state, 'Evaluation outer identity differs')
            metrics[name] = {}
            for split in ('dev', 'shift'):
                metric, predictions = evaluation.audit_evaluation(receipt['evaluation'][split],
                    directory/f'{split}.jsonl', panels[split], configuration=legacy_config(config),
                    plan_sha256=plan_hash, model=model, cache_metadata=eval_cache[split])
                metrics[name][split] = metric
                decisions.extend({'configuration': name, 'split': split, 'panel_index': i,
                    'id': row['id'], 'choice': row['choice']} for i, row in enumerate(predictions))
            measured = read(directory/'latency.json')
            evaluation.audit_latency(measured, combined, plan['selection']['latency_indices'],
                configuration=legacy_config(config), plan_sha256=plan_hash, expected_state_sha256=state,
                warmups=PROTOCOL['latency_warmups'], model=model)
            for key in ('host_load_average_before_latency', 'host_load_average_after_latency'):
                require(type(receipt[key]) is list and len(receipt[key]) == 3 and all(number(v) for v in receipt[key]),
                        'Invalid host-load observation')
            latency[name] = {'mean_ms': measured['total_wall_ms']/len(measured['records']),
                'total_wall_ms': measured['total_wall_ms'], 'warmup_wall_ms': measured['warmup_wall_ms'],
                **{k: receipt[k] for k in ('host_load_average_before_latency', 'host_load_average_after_latency')}}
            del model
        require(decisions == read(execution/'decisions.json'), 'Graded choices differ from saved full-score predictions')
        graded = audit_grading(grading_plan(plan, panels), execution/'regret', decisions,
            plan_hash=plan_hash, engine_hash=plan['binding']['engine_sha256'])
        values = rows(execution/'regret/regret.jsonl')
        engine = engine_metrics(values, plan['baselines']+plan['configurations'])
        losses = {name: {s: panel['mean_signed_bounded_loss'] for s, panel in panels.items()}
                  for name, panels in engine.items()}
        summary = {'status': 'completed', 'version': PROTOCOL['version'], 'plan_sha256': plan_hash,
            'execution_receipt_sha256': sha(execution/'completed.json'), 'metrics': metrics,
            'engine_loss': losses, 'engine_metrics': engine,
            'mapping_comparison': continuation(losses), 'latency': latency,
            'training_costs': fit_costs, 'grading_costs': graded['cost'],
            'backbone_pretraining_costs': plan['binding']['backbone_pretraining_costs'],
            'original_data_costs': read(ROOT/OLD_EXECUTION/'data/completed.json'),
            'original_inputs': plan['original_inputs'], 'wall_seconds': finished['wall_seconds'],
            'limits': [PROTOCOL[k] for k in ('matching', 'data', 'pretraining', 'selection', 'timing',
                                           'scope', 'arena', 'asset_policy', 'audit_scope')],
            'report_new_model_or_engine_calls': 0}
        write(out/'summary.json', summary)
        elapsed = time.perf_counter()-begun
        require(number(elapsed) and elapsed <= PROTOCOL['audit_wall_seconds'], 'Audit wall budget exceeded')
        write(out/'receipt.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'execution_receipt_sha256': sha(execution/'completed.json'), 'audit_wall_seconds': elapsed,
            'completed_unix': time.time(), 'new_model_calls': 0, 'new_engine_calls': 0, 'files': tree(out)})
        return {'status': 'completed', 'gate': summary['mapping_comparison']}
    except BaseException as exc:
        write(out/'failed.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
            'wall_seconds': time.perf_counter()-begun})
        raise
    finally:
        signal.alarm(0)


def launch(plan_path, execution, audit_out, launcher):
    launcher = Path(launcher).resolve()
    execution = private_output(execution)
    require(not execution.exists() and not Path(audit_out).exists(), 'Execution or audit output exists')
    request = {'plan': str(Path(plan_path).resolve()), 'execution': str(execution),
        'audit': str(Path(audit_out).resolve()), 'plan_sha256': sha(plan_path), 'created_unix': time.time()}
    launcher.mkdir(parents=True, exist_ok=False)
    try:
        write(launcher/'request.json', request)
        with (launcher/'supervisor.log').open('xb') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise', '--out', str(launcher)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        result = {'status': 'launched', 'supervisor_pid': child.pid, **request}
        write(launcher/'launched.json', result)
        return result
    except BaseException as exc:
        write(launcher/'failed.json', {'status': 'failed', 'phase': 'launch', 'error_type': type(exc).__name__,
                                      'error': str(exc), 'completed_unix': time.time()})
        raise


def supervise(launcher):
    launcher, command = Path(launcher), 'setup'
    try:
        request = read(launcher/'request.json')
        require(sha(request['plan']) == request['plan_sha256'], 'Launch plan changed')
        for command, target in (('run', request['execution']), ('audit', request['audit'])):
            args = [sys.executable, str(Path(__file__).resolve()), command, '--plan', request['plan'], '--out', target]
            if command == 'audit':
                args += ['--execution', request['execution']]
            budget = PROTOCOL['primary_wall_seconds' if command == 'run' else 'audit_wall_seconds']
            with (launcher/f'{command}.log').open('xb') as log:
                child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                write(launcher/f'{command}-launched.json', {'pid': child.pid, 'started_unix': time.time(), 'args': args})
                try:
                    code = child.wait(timeout=budget+10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                    raise TimeoutError('Phase watchdog exhausted the frozen budget') from None
            write(launcher/f'{command}-exit.json', {'returncode': code, 'completed_unix': time.time()})
            if code:
                result = {'status': 'failed', 'phase': command, 'returncode': code, 'completed_unix': time.time()}
                write(launcher/'failed.json', result)
                return result
            terminal = 'completed.json' if command == 'run' else 'receipt.json'
            receipt = completed(target, request['plan_sha256'], terminal=terminal)
            elapsed = receipt['wall_seconds' if command == 'run' else 'audit_wall_seconds']
            require(number(elapsed) and elapsed <= budget, 'Successful exit exceeded the frozen wall budget')
        write(launcher/'completed.json', {'status': 'completed', 'completed_unix': time.time()})
        return {'status': 'completed'}
    except BaseException as exc:
        if not (launcher/'failed.json').exists():
            write(launcher/'failed.json', {'status': 'failed', 'phase': command, 'error_type': type(exc).__name__,
                'error': str(exc), 'completed_unix': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'run', 'audit', 'launch', 'supervise'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--audit-out', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.out)
    elif args.command == 'verify':
        plan, _ = verify(args.plan)
        result = {'status': 'verified', 'plan_sha256': sha(args.plan), 'fits': len(plan['configurations'])}
    elif args.command == 'run':
        result = run(args.plan, args.out)
    elif args.command == 'audit':
        result = audit(args.plan, args.execution, args.out)
    elif args.command == 'launch':
        result = launch(args.plan, args.execution, args.audit_out, args.out)
    else:
        result = supervise(args.out)
    print(json.dumps(result, allow_nan=False), flush=True)
