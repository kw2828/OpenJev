"""Controlled action-outcome supervision for an unchanged recurrent chess actor.

All nine fits finish before any evaluation. Every epoch is saved; epoch eight
is the sole primary checkpoint. The exposed historical panels are development
evidence, not fresh confirmation. A critic used only while training is not a
world model, a planner, or evidence for biological wiring.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

from openjev.research import chess_continuation_arena as arena
from openjev.research import chess_continuation_data as data
from openjev.research.chess_anchor_eval import audit_predictions, evaluate, probe, tensorize
from openjev.research.chess_continuation_model import ContinuationChess, training_objective

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'runs/chess-inputs/stockfish/stockfish-macos-universal'
PANEL_SOURCE = 'runs/chess-capacity-v1/execution/data'
OBJECTIVES = ('policy', 'best_value', 'continuation')
SEEDS = (193, 211, 227)
SOURCES = [
    'scripts/chess_continuation_study.py', 'tests/test_chess_continuation_study.py',
    'scripts/chess_continuation_profile.py',
    'src/openjev/research/chess_continuation_model.py', 'tests/test_chess_continuation_model.py',
    'src/openjev/research/chess_continuation_data.py', 'tests/test_chess_continuation_data.py',
    'src/openjev/research/chess_continuation_arena.py', 'tests/test_chess_continuation_arena.py',
    'src/openjev/research/chess_capacity_arena.py', 'src/openjev/research/chess_arena.py',
    'src/openjev/research/chess_anchor.py', 'src/openjev/research/chess_anchor_eval.py',
    'src/openjev/research/chess_spatial.py', 'src/openjev/research/chess_compute.py',
    'scripts/chess_compute_study.py',
]
PROTOCOL = {
    'version': 'chess-continuation-v1', 'objectives': list(OBJECTIVES), 'seeds': list(SEEDS),
    'width': 32, 'depth': 4, 'epochs': 8, 'batch_size': 128,
    'training_device': 'mps', 'evaluation_device': 'cpu', 'torch_threads': 2,
    'deterministic_algorithms': False, 'dtype': 'float32',
    'optimizer': {'name': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.},
    'gradient_clip': 1., 'batch_seed_base': 10900029, 'fit_order_seed': 10900017,
    'root_value_weight': .5, 'candidate_value_weight': 1.,
    'objective_definition': 'CE + .5 root-value MSE. best_value adds teacher-action critic MSE. continuation adds the per-root mean over teacher action and an available distinct behavior action. Duplicate actions and missing labels add no target.',
    'primary_checkpoint_epoch': 8, 'diagnostic_epochs': list(range(1, 9)),
    'regret_selection_seed': 10900043, 'regret_positions_per_split': 128,
    'regret_nodes': 20000, 'regret_call_ceiling': 2560, 'regret_requested_node_ceiling': 51200000,
    'latency_selection_seed': 10900059, 'latency_positions_per_split': 64, 'latency_warmups': 3,
    'primary_wall_seconds': 14400, 'audit_wall_seconds': 2700,
    'relative_loss_reduction': .20, 'arena_lower_points_threshold': .60,
    'gate': 'Continuation must reduce mean signed bounded engine loss by at least20% against BOTH controls on BOTH panels, with positive reference means, and achieve at least60% lower all-game point bound against EACH control with no failed games.',
    'matching': 'Same roots, raw actor, paired initialization, minibatch order, optimizer updates and depth. Best-value and continuation have identical active parameters. Costs are measured, not assumed equal.',
    'selection': 'All9 fits before evaluation. Final epoch primary. All8 epochs reported on fixed diagnostic subsets. No early stopping, replacement seeds, retry, checkpoint selection or budget extension.',
    'scope': 'Auxiliary supervision development study on exposed historical games and panels. No novelty, biological advantage, calibrated win probability, planning, Astra comparison or Elo claim.',
    'audit_scope': 'Saved-output source and coverage audit, scalar prediction arithmetic, training-log arithmetic and native game replay. No model or engine reruns and no proof of recorded full score vectors or exact optimizer execution.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse_json(text):
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
    return parse_json(Path(path).read_text())


def rows(path):
    return [parse_json(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n')
    stream.flush()


def files(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        require(not path.is_symlink(), 'Evidence must not contain symlinks')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def configurations():
    result = [{'objective': o, 'seed': s, 'name': f'{o}-{s}'} for o in OBJECTIVES for s in SEEDS]
    random.Random(PROTOCOL['fit_order_seed']).shuffle(result)
    return result


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def batches(count, seed):
    require(type(count) is int and count > 0, 'Positive training count required')
    generator = torch.Generator().manual_seed(PROTOCOL['batch_seed_base']+seed)
    step = 0
    for epoch in range(1, PROTOCOL['epochs']+1):
        order = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count, PROTOCOL['batch_size']):
            step += 1
            yield {'epoch': epoch, 'step': step, 'indices': order[start:start+PROTOCOL['batch_size']]}


def panel_inputs():
    receipt = read(ROOT/PANEL_SOURCE/'completed.json')
    require(sha(ROOT/PANEL_SOURCE/'completed.json') == data.SOURCE_RECEIPTS[PANEL_SOURCE],
            'Original capacity receipt changed')
    panels, bindings, secondary, latency, combined = {}, {}, {}, [], []
    for offset, split in enumerate(('dev', 'shift')):
        path = ROOT/PANEL_SOURCE/f'{split}.jsonl'
        require(sha(path) == receipt['files'][f'{split}.jsonl'], 'Historical panel changed')
        bindings[f'{PANEL_SOURCE}/{split}.jsonl'] = sha(path)
        panel = rows(path)
        require(len(panel) == 4096 and all(r['split'] == split for r in panel), 'Panel scope changed')
        panels[split] = panel
        secondary[split] = sorted(random.Random(PROTOCOL['regret_selection_seed']+offset).sample(range(len(panel)), 128))
        chosen = sorted(random.Random(PROTOCOL['latency_selection_seed']+offset).sample(range(len(panel)), 64))
        latency.extend(len(combined)+i for i in chosen)
        combined.extend(panel)
    return panels, {'files': bindings, 'secondary_indices': secondary, 'latency_indices': latency}, combined


def signature(dataset):
    _, panel, _ = panel_inputs()
    count = dataset.membership['partitions']['train']['roots']
    return {
        'protocol': PROTOCOL, 'dataset': dataset.signature(), 'panels': panel,
        'configurations': configurations(), 'arena': arena.protocol(),
        'updates_per_fit': math.ceil(count/PROTOCOL['batch_size'])*PROTOCOL['epochs'],
        'examples_seen_per_fit': count*PROTOCOL['epochs'],
        'parameters': {o: ContinuationChess(o, SEEDS[0]).parameter_report() for o in OBJECTIVES},
        'initial_state_sha256': {str(s): state_hash(ContinuationChess('policy', s)) for s in SEEDS},
        'engine_sha256': sha(ROOT/ENGINE), 'sources': {p: sha(ROOT/p) for p in SOURCES},
        'synthetic_update_profile_sha256': sha(ROOT/'evidence/chess-continuation-preflight-v1/profile.json'),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'mps_available': torch.backends.mps.is_available(),
                        'mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'),
                        'dependencies': {n: importlib.metadata.version(n) for n in ('torch', 'chess', 'numpy')},
                        'lock_sha256': sha(ROOT/'uv.lock'), 'pyproject_sha256': sha(ROOT/'pyproject.toml')},
    }


def prepare(out):
    dataset = data.load_dataset(ROOT)
    plan = signature(dataset)
    require(plan['environment']['mps_available'], 'MPS required; no CPU substitution')
    require(plan['environment']['mps_fallback'] == '0', 'MPS fallback must be disabled')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    write(out/'prepared.json', {'status': 'prepared', 'plan_sha256': sha(out/'plan.json'),
                                'created_unix': time.time(), 'training_or_evaluation_started': False})
    return {'path': str(out/'plan.json'), 'sha256': sha(out/'plan.json'),
            'training_roots': plan['dataset']['membership']['partitions']['train']['roots'],
            'updates_per_fit': plan['updates_per_fit']}


def verify(plan_path):
    plan = read(plan_path)
    dataset = data.load_dataset(ROOT)
    require(plan == json.loads(json.dumps(signature(dataset))), 'Frozen protocol, source, input or environment changed')
    return plan, dataset


class DeadlineExceeded(BaseException):
    """A phase stop must escape policy wrappers that retain ordinary errors."""


def deadline(seconds):
    def expired(_signum, _frame):
        raise DeadlineExceeded('Frozen wall-clock budget exhausted; no retry or extension')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)


def load_model(config, path, plan_hash):
    return ContinuationChess.load(path, expected_plan_sha256=plan_hash,
                                  expected_seed=config['seed'], expected_objective=config['objective'])


def fit(config, tensors, out, plan, plan_hash):
    out.mkdir()
    model = ContinuationChess(config['objective'], config['seed'])
    initial = state_hash(model)
    require(initial == plan['initial_state_sha256'][str(config['seed'])], 'Paired initialization differs')
    model.to(PROTOCOL['training_device'])
    opt = PROTOCOL['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=tuple(opt['betas']),
                                 eps=opt['eps'], weight_decay=opt['weight_decay'])
    count = len(tensors['targets'])
    steps_per_epoch = math.ceil(count/PROTOCOL['batch_size'])
    torch.mps.synchronize()
    begun = time.perf_counter()
    checkpoints = {}
    with (out/'learning.jsonl').open('x') as journal:
        for step in batches(count, config['seed']):
            batch = {k: v[step['indices']].to(PROTOCOL['training_device']) for k, v in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss = training_objective(model, batch['observations'], batch['candidates'], batch['mask'],
                                      batch['targets'], batch['values'], batch['behavior_indices'],
                                      batch['continuation_values'], batch['continuation_mask'])
            require(bool(torch.isfinite(loss['total'])), 'Nonfinite training objective')
            loss['total'].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            metrics = {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else v for k, v in loss.items()}
            append(journal, {'step': step['step'], 'epoch': step['epoch'], 'batch_size': len(step['indices']),
                             'batch_indices_sha256': digest(step['indices']), **metrics,
                             'preclip_gradient_norm': float(gradient.detach().cpu()),
                             'elapsed_seconds': time.perf_counter()-begun})
            if step['step'] % steps_per_epoch == 0:
                torch.mps.synchronize()
                name = f"epoch-{step['epoch']:02d}.pt"
                model.save(out/name, plan_sha256=plan_hash)
                checkpoints[name] = sha(out/name)
                print(json.dumps({'fit': config['name'], 'epoch': step['epoch'], 'updates': step['step']}), flush=True)
    torch.mps.synchronize()
    require(step['step'] == plan['updates_per_fit'], 'Incomplete training schedule')
    result = {'status': 'completed', **config, 'plan_sha256': plan_hash,
              'initial_state_sha256': initial, 'updates': step['step'],
              'examples_seen': count*PROTOCOL['epochs'], 'training_seconds': time.perf_counter()-begun,
              'parameters': model.parameter_report(), 'checkpoints': checkpoints,
              'learning_sha256': sha(out/'learning.jsonl'), 'completed_unix': time.time()}
    write(out/'training.json', result)
    del optimizer, model
    torch.mps.empty_cache()
    return result


def grader():
    spec = importlib.util.spec_from_file_location('_continuation_frozen_grader', ROOT/'scripts/chess_compute_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.PROTOCOL['regret_nodes'] == PROTOCOL['regret_nodes']
            and module.PROTOCOL['value_cp_scale'] == 600 and module.PROTOCOL['mate_cp'] == 10000,
            'Inherited grader scale or budget differs')
    return module


def grading_plan(plan, panels):
    return {'engine_path': str(ROOT/ENGINE), 'panels': panels,
            'secondary_indices': plan['panels']['secondary_indices'],
            'configurations': [{'id': c['name']} for c in plan['configurations']]}


def run(plan_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan_hash, begun = sha(plan_path), time.perf_counter()
    write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'started_unix': time.time()})
    deadline(PROTOCOL['primary_wall_seconds'])
    try:
        plan, dataset = verify(plan_path)
        torch.set_num_threads(PROTOCOL['torch_threads'])
        torch.use_deterministic_algorithms(PROTOCOL['deterministic_algorithms'])
        training = [dataset.rows[i] for i in dataset.membership['partitions']['train']['indices']]
        train_tensors, _ = data.tensorize(training)
        (out/'fits').mkdir()
        fits = [fit(c, train_tensors, out/'fits'/c['name'], plan, plan_hash) for c in plan['configurations']]
        del train_tensors
        write(out/'training-completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
              'fits': fits, 'completed_unix': time.time(), 'evaluation_started': False})
        write(out/'evaluation-started.json', {'started_unix': time.time(), 'completed_fits': len(fits),
              'training_receipt_sha256': sha(out/'training-completed.json')})
        panels, _, combined = panel_inputs()
        diagnostic = {name: [dataset.rows[i] for i in subset['indices']]
                      for name, subset in dataset.membership['fixed_diagnostic_subsets'].items()}
        encoded = {s: tensorize(r) for s, r in {**panels, **diagnostic}.items()}
        curves, summaries, decisions, latencies, models = [], [], [], [], {}
        for config in plan['configurations']:
            name = config['name']
            for epoch in PROTOCOL['diagnostic_epochs']:
                model = load_model(config, out/'fits'/name/f'epoch-{epoch:02d}.pt', plan_hash)
                for split, records in diagnostic.items():
                    result = evaluate(model, records, *encoded[split], 4, out/'curves'/name/f'{epoch:02d}-{split}.jsonl')
                    curves.append({'configuration': name, 'epoch': epoch, 'partition': split, **result})
            # The fixed final checkpoint above is used without choosing an epoch.
            models[name] = model
            for split, records in panels.items():
                path = out/'predictions'/name/f'{split}.jsonl'
                result = evaluate(model, records, *encoded[split], 4, path)
                summaries.append({'configuration': name, 'split': split, **result})
                decisions.extend({'configuration': name, 'split': split, 'panel_index': i,
                                  'id': r['id'], 'choice': r['choice']} for i, r in enumerate(rows(path)))
            latencies.append({'configuration': name, **probe(model, combined, 4, plan['panels']['latency_indices'])})
        write(out/'curves.json', curves)
        write(out/'predictions.json', summaries)
        write(out/'decisions.json', decisions)
        write(out/'latency.json', latencies)
        (out/'engine').mkdir()
        cost = grader().score_secondary(grading_plan(plan, panels), out/'engine', decisions)
        require(cost['calls'] <= PROTOCOL['regret_call_ceiling']
                and cost['requested_nodes'] <= PROTOCOL['regret_requested_node_ceiling'], 'Engine budget exceeded')
        write(out/'engine'/'cost.json', cost)
        arena.run(models, out/'arena', plan_hash, arena_plan=plan['arena'])
        manifest = files(out)
        elapsed = time.perf_counter()-begun
        require(math.isfinite(elapsed) and elapsed <= PROTOCOL['primary_wall_seconds'], 'Primary wall budget exceeded')
        write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
              'completed_unix': time.time(), 'total_wall_seconds': elapsed, 'files': manifest})
        return {'status': 'completed', 'execution': str(out)}
    except BaseException as exc:
        write(out/'failed.json', {'status': 'failed', 'plan_sha256': plan_hash, 'completed_unix': time.time(),
                                  'error_type': type(exc).__name__, 'error': str(exc),
                                  'elapsed_seconds': time.perf_counter()-begun})
        raise
    finally:
        signal.alarm(0)


def validate_training(config, records, fit_receipt, training, plan):
    require(len(records) == plan['updates_per_fit'], 'Training update count differs')
    elapsed = 0.
    for record, step in zip(records, batches(len(training), config['seed']), strict=True):
        for key in ('step', 'epoch', 'batch_size', 'teacher_targets_used', 'continuation_targets_used', 'supervised_targets_used'):
            require(type(record[key]) is int and record[key] >= 0, 'Training counts must be integers')
        require(record['step'] == step['step'] and record['epoch'] == step['epoch']
                and record['batch_size'] == len(step['indices'])
                and record['batch_indices_sha256'] == digest(step['indices']), 'Training schedule differs')
        for key in ('total', 'policy_ce', 'root_value_mse', 'candidate_value_mse', 'preclip_gradient_norm', 'elapsed_seconds'):
            require(type(record[key]) in (int, float) and math.isfinite(record[key]) and record[key] >= 0,
                    'Nonfinite or negative training metric')
        require(record['elapsed_seconds'] >= elapsed, 'Training time moved backwards')
        elapsed = record['elapsed_seconds']
        require(math.isclose(record['total'], record['policy_ce']+.5*record['root_value_mse']+record['candidate_value_mse'],
                             rel_tol=2e-6, abs_tol=2e-6), 'Objective arithmetic differs')
        teacher = 0 if config['objective'] == 'policy' else len(step['indices'])
        continuation = sum(training[i]['continuation_mask'] and training[i]['behavior_uci'] != training[i]['target_uci']
                           for i in step['indices']) if config['objective'] == 'continuation' else 0
        require(record['teacher_targets_used'] == teacher and record['continuation_targets_used'] == continuation
                and record['supervised_targets_used'] == teacher+continuation, 'Supervised target counts differ')
        if config['objective'] == 'policy':
            require(record['candidate_value_mse'] == 0., 'Policy control acquired critic supervision')
    require(type(fit_receipt['updates']) is int and type(fit_receipt['examples_seen']) is int
            and type(fit_receipt['training_seconds']) in (int, float)
            and math.isfinite(fit_receipt['training_seconds'])
            and fit_receipt['updates'] == plan['updates_per_fit']
            and fit_receipt['examples_seen'] == plan['examples_seen_per_fit']
            and fit_receipt['initial_state_sha256'] == plan['initial_state_sha256'][str(config['seed'])]
            and fit_receipt['training_seconds'] >= elapsed, 'Training receipt differs')


def outcome_gate(regret, arena_summary):
    means, checks = {}, []
    expected_configs = {f'{o}-{s}' for o in OBJECTIVES for s in SEEDS}
    require(len(regret) == 9*2*PROTOCOL['regret_positions_per_split'], 'Incomplete grading coverage')
    seen, memberships = set(), {}
    for row in regret:
        key = row['configuration'], row['split'], row['panel_index']
        require(key not in seen and key[0] in expected_configs and key[1] in ('dev', 'shift')
                and type(key[2]) is int and key[2] >= 0, 'Duplicate or invalid grading identity')
        seen.add(key)
        memberships.setdefault(key[:2], set()).add(key[2])
        require(type(row['bounded_regret']) in (int, float) and math.isfinite(row['bounded_regret'])
                and -2 <= row['bounded_regret'] <= 2, 'Invalid bounded engine loss')
    for split in ('dev', 'shift'):
        panels = [memberships.get((name, split), set()) for name in expected_configs]
        require(len(panels[0]) == PROTOCOL['regret_positions_per_split']
                and all(p == panels[0] for p in panels), 'Grading configurations have different positions')
    for objective in OBJECTIVES:
        for split in ('dev', 'shift'):
            selected = [r for r in regret if r['configuration'] in {f'{objective}-{s}' for s in SEEDS} and r['split'] == split]
            require(len(selected) == 3*PROTOCOL['regret_positions_per_split'], 'Incomplete grading group')
            means[objective, split] = math.fsum(r['bounded_regret'] for r in selected)/len(selected)
    for control in ('policy', 'best_value'):
        for split in ('dev', 'shift'):
            baseline, treatment = means[control, split], means['continuation', split]
            checks.append({'control': control, 'split': split, 'metric': 'bounded_engine_loss',
                           'reference': baseline, 'continuation': treatment,
                           'relative_reduction': (baseline-treatment)/baseline if baseline > 0 else None,
                           'passed': baseline > 0 and treatment <= .8*baseline+1e-12})
    # Independently check the summarized point bounds; native game replay is
    # separately required by audit before this gate may support a claim.
    require(arena_summary['status'] == 'completed' and arena_summary['games'] == 192
            and set(arena_summary['by_opponent']) == {'policy', 'best_value'}, 'Incomplete arena coverage')
    arena_checks, total_failed = [], 0
    for control in ('policy', 'best_value'):
        panel = arena_summary['by_opponent'][control]
        counts, points = panel['status_counts'], panel['continuation']
        require(panel['games'] == 96 and all(type(counts[k]) is int and counts[k] >= 0
                for k in ('completed', 'unfinished', 'failed')) and sum(counts.values()) == 96,
                'Incomplete per-control arena coverage')
        require(all(type(points[k]) is int and points[k] >= 0 for k in ('wins', 'draws', 'losses'))
                and points['wins']+points['draws']+points['losses'] == counts['completed'],
                'Arena result counts differ')
        completed = points['wins']+.5*points['draws']
        lower, upper = completed/96, (completed+counts['unfinished']+counts['failed'])/96
        require(points['possible_points'] == 96 and points['completed_points'] == completed
                and points['score_lower_bound'] == lower and points['score_upper_bound'] == upper,
                'Arena point bounds differ')
        arena_checks.append({'opponent': control, 'threshold': .60, 'lower_score_bound': lower, 'passed': lower >= .60})
        total_failed += counts['failed']
    arena_gate = arena_summary['continuation_gate']
    expected_gate = {'checks': arena_checks, 'zero_failed_games': total_failed == 0,
                     'passed': total_failed == 0 and all(c['passed'] for c in arena_checks)}
    require(arena_gate == expected_gate, 'Arena gate disagrees with per-control outcomes')
    checks.append({'metric': 'paired_games', 'details': arena_gate, 'passed': arena_gate['passed']})
    return {'passed': all(c['passed'] for c in checks), 'checks': checks,
            'scope': PROTOCOL['scope'], 'thresholds': {'relative_loss_reduction': .20, 'lower_game_points': .60}}


def audit_latency(saved, plan, combined):
    require(len(saved) == 9 and {r['configuration'] for r in saved} == {c['name'] for c in plan['configurations']},
            'Latency configuration coverage differs')
    for row in saved:
        require(row['device'] == 'cpu' and row['torch_threads'] == 2 and row['depth'] == 4,
                'Latency device or inference depth differs')
        require(len(row['warmup_records']) == 3 and len(row['records']) == len(plan['panels']['latency_indices']),
                'Latency coverage differs')
        for record, index in zip(row['records'], plan['panels']['latency_indices'], strict=True):
            require(record['index'] == index and record['id'] == combined[index]['id']
                    and record['choice'] in {m.uci() for m in chess.Board(combined[index]['fen']).legal_moves},
                    'Native latency position or choice differs')
        for index, record in enumerate(row['warmup_records']):
            require(record['index'] == index and record['id'] == 'starting-board'
                    and record['choice'] in {m.uci() for m in chess.Board().legal_moves}, 'Invalid latency warmup')
        for record in row['records']+row['warmup_records']:
            require(type(record['wall_ms']) in (int, float) and math.isfinite(record['wall_ms'])
                    and record['wall_ms'] >= 0, 'Invalid latency duration')
        require(row['total_wall_ms'] == math.fsum(r['wall_ms'] for r in row['records'])
                and row['warmup_wall_ms'] == math.fsum(r['wall_ms'] for r in row['warmup_records']),
                'Latency accounting differs')
    return saved


def engine_metrics(regret):
    result = []
    for config in configurations():
        for split in ('dev', 'shift'):
            selected = [r for r in regret if (r['configuration'], r['split']) == (config['name'], split)]
            raw = [r['cp_loss'] for r in selected]
            result.append({'configuration': config['name'], 'split': split, 'positions': len(selected),
                           'mean_signed_bounded_loss': math.fsum(r['bounded_regret'] for r in selected)/len(selected),
                           'mean_cp_loss': math.fsum(raw)/len(raw),
                           'p95_cp_loss': float(np.quantile(raw, .95)), 'max_cp_loss': max(raw)})
    return result


def audit(plan_path, execution, out):
    out, execution = Path(out), Path(execution)
    out.mkdir(parents=True, exist_ok=False)
    begun, plan_hash = time.perf_counter(), sha(plan_path)
    write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'started_unix': time.time()})
    deadline(PROTOCOL['audit_wall_seconds'])
    try:
        plan, dataset = verify(plan_path)
        require(not (execution/'failed.json').exists(), 'Execution failed')
        receipt = read(execution/'completed.json')
        actual = files(execution)
        actual.pop('completed.json', None)
        require(receipt['status'] == 'completed' and receipt['plan_sha256'] == plan_hash
                and receipt['files'] == actual, 'Execution manifest differs')
        require(type(receipt['total_wall_seconds']) in (int, float)
                and math.isfinite(receipt['total_wall_seconds'])
                and 0 <= receipt['total_wall_seconds'] <= PROTOCOL['primary_wall_seconds'], 'Execution exceeded wall budget')
        train_done, eval_start = read(execution/'training-completed.json'), read(execution/'evaluation-started.json')
        require(train_done['evaluation_started'] is False and len(train_done['fits']) == 9
                and eval_start['completed_fits'] == 9 and train_done['completed_unix'] <= eval_start['started_unix']
                and eval_start['training_receipt_sha256'] == sha(execution/'training-completed.json'),
                'Evaluation preceded complete fitting')
        training = [dataset.rows[i] for i in dataset.membership['partitions']['train']['indices']]
        panels, _, combined = panel_inputs()
        diagnostic = {s: [dataset.rows[i] for i in p['indices']]
                      for s, p in dataset.membership['fixed_diagnostic_subsets'].items()}
        curves, primary, decisions, final_models = [], [], [], {}
        curve_records = read(execution/'curves.json')
        primary_records = read(execution/'predictions.json')
        require(len(curve_records) == 9*8*2 and len(primary_records) == 9*2, 'Evaluation summary coverage differs')
        for config, fit_result in zip(plan['configurations'], train_done['fits'], strict=True):
            name = config['name']
            folder = execution/'fits'/name
            require(fit_result == read(folder/'training.json')
                    and all(fit_result[k] == config[k] for k in config)
                    and fit_result['plan_sha256'] == plan_hash
                    and fit_result['learning_sha256'] == sha(folder/'learning.jsonl'), 'Fit identity differs')
            validate_training(config, rows(folder/'learning.jsonl'), fit_result, training, plan)
            require(set(fit_result['checkpoints']) == {f'epoch-{e:02d}.pt' for e in range(1, 9)}, 'Checkpoint coverage differs')
            for epoch in range(1, 9):
                path = folder/f'epoch-{epoch:02d}.pt'
                require(sha(path) == fit_result['checkpoints'][path.name], 'Checkpoint hash differs')
                model = load_model(config, path, plan_hash)
                require(model.parameter_report() == plan['parameters'][config['objective']], 'Active parameter count differs')
                for split, records in diagnostic.items():
                    metric, _ = audit_predictions(execution/'curves'/name/f'{epoch:02d}-{split}.jsonl', records)
                    saved = [r for r in curve_records if (r['configuration'], r['epoch'], r['partition']) == (name, epoch, split)]
                    require(len(saved) == 1 and saved[0]['metrics'] == metric, 'Curve summary differs')
                    curves.extend(saved)
            final_models[name] = model
            for split, records in panels.items():
                path = execution/'predictions'/name/f'{split}.jsonl'
                metric, _ = audit_predictions(path, records)
                saved = [r for r in primary_records if (r['configuration'], r['split']) == (name, split)]
                require(len(saved) == 1 and saved[0]['metrics'] == metric, 'Primary summary differs')
                primary.extend(saved)
                decisions.extend({'configuration': name, 'split': split, 'panel_index': i,
                                  'id': r['id'], 'choice': r['choice']} for i, r in enumerate(rows(path)))
        require(decisions == read(execution/'decisions.json'), 'Graded decisions differ from recorded predictions')
        cost = read(execution/'engine'/'cost.json')
        regret = rows(execution/'engine'/'regret.jsonl')
        grader().validate_secondary(grading_plan(plan, panels), decisions,
                                   rows(execution/'engine'/'analyses.jsonl'), regret, cost)
        require(cost['calls'] <= PROTOCOL['regret_call_ceiling']
                and cost['requested_nodes'] <= PROTOCOL['regret_requested_node_ceiling'], 'Engine budget exceeded')
        arena_summary = arena.report(execution/'arena', plan_hash, arena_plan=plan['arena'])
        require(read(execution/'arena'/'started.json')['models'] == arena._model_bindings(final_models),
                'Arena used different weights from the final checkpoints')
        latency = audit_latency(read(execution/'latency.json'), plan, combined)
        summary = {'version': PROTOCOL['version'], 'plan_sha256': plan_hash, 'training': train_done['fits'],
                   'curves': curves, 'primary': primary, 'secondary_cost': cost, 'arena': arena_summary,
                   'engine_metrics': engine_metrics(regret), 'native_latency': latency,
                   'continuation_gate': outcome_gate(regret, arena_summary),
                   'audit_scope': PROTOCOL['audit_scope'], 'claim_scope': PROTOCOL['scope']}
        write(out/'summary.json', summary)
        manifest = files(out)
        elapsed = time.perf_counter()-begun
        require(math.isfinite(elapsed) and elapsed <= PROTOCOL['audit_wall_seconds'], 'Audit wall budget exceeded')
        write(out/'receipt.json', {'status': 'completed', 'plan_sha256': plan_hash,
              'execution_receipt_sha256': sha(execution/'completed.json'), 'completed_unix': time.time(),
              'audit_wall_seconds': elapsed, 'new_model_calls': 0, 'new_engine_calls': 0,
              'files': manifest})
        return {'status': 'completed', 'gate': summary['continuation_gate']}
    except BaseException as exc:
        write(out/'failed.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        signal.alarm(0)


def launch(plan_path, execution, audit_out, launcher):
    launcher = Path(launcher).resolve()
    request = {'plan': str(Path(plan_path).resolve()), 'execution': str(Path(execution).resolve()),
               'audit': str(Path(audit_out).resolve()), 'plan_sha256': sha(plan_path), 'created_unix': time.time()}
    require(not Path(execution).exists() and not Path(audit_out).exists(), 'Execution or audit path already exists')
    launcher.mkdir(parents=True, exist_ok=False)
    try:
        write(launcher/'request.json', request)
        with (launcher/'supervisor.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise', '--out', str(launcher)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        result = {'status': 'launched', 'supervisor_pid': process.pid, **request}
        write(launcher/'launched.json', result)
        return result
    except BaseException as exc:
        write(launcher/'failed.json', {'status': 'failed', 'phase': 'launch', 'error_type': type(exc).__name__,
                                       'error': str(exc), 'completed_unix': time.time()})
        raise


def supervise(launcher):
    launcher = Path(launcher)
    command = 'setup'
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
                                         stderr=subprocess.STDOUT, start_new_session=True)
                write(launcher/f'{command}-launched.json', {'pid': child.pid, 'started_unix': time.time(), 'args': args})
                try:
                    code = child.wait(timeout=budget+10)
                except subprocess.TimeoutExpired:
                    # Only the process group created for this phase is stopped.
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
            terminal = Path(target)/('completed.json' if command == 'run' else 'receipt.json')
            receipt = read(terminal)
            require(receipt['status'] == 'completed' and receipt['plan_sha256'] == request['plan_sha256']
                    and not (Path(target)/'failed.json').exists(), 'Successful exit lacks a valid completion receipt')
            elapsed = receipt['total_wall_seconds' if command == 'run' else 'audit_wall_seconds']
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= budget,
                    'Successful exit exceeded the frozen wall budget')
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
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--audit-out', type=Path)
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
