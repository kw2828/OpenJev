"""Board-aware chess distillation with a controlled future-prediction auxiliary.

Prepare freezes code, budgets, exclusions and dependencies before new data.
Run never selects checkpoints or extends budgets. Report verifies saved evidence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.chess_spatial import (
    MODES,
    SpatialChess,
    encode_board,
    encode_candidates,
    piece_targets,
)
from openjev.research.chess_spatial_data import generate_data, stockfish_label, validate_data
from openjev.research.chess_student import state_key

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'runs/chess-inputs/stockfish/stockfish-macos-universal'
EXCLUSIONS = [f'evidence/chess-student-v1/results/execution/data/{split}.jsonl'
              for split in ('train', 'dev')]
SOURCES = ['scripts/train_chess_spatial.py', 'src/openjev/research/chess_spatial.py',
           'src/openjev/research/chess_spatial_data.py', 'tests/test_chess_spatial.py',
           'tests/test_chess_spatial_data.py', 'tests/test_chess_spatial_study.py',
           'src/openjev/research/chess_spatial_baselines.py',
           'src/openjev/research/chess_arena.py', 'src/openjev/research/chess_student.py',
           'src/openjev/research/chess_spatial_regret.py', 'tests/test_chess_spatial_regret.py']
PROTOCOL = {
    'version': 'chess-spatial-v1', 'modes': list(MODES), 'seeds': [17, 29, 43],
    'width': 32, 'depth': 4, 'input_channels': 19,
    'splits': [
        {'name': 'train', 'examples': 32768, 'game_cap': 1200, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 92000000},
        {'name': 'dev', 'examples': 4096, 'game_cap': 200, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 93000000},
        {'name': 'shift', 'examples': 4096, 'game_cap': 200, 'max_plies': 96,
         'random_move_probability': .1, 'seed_base': 94000000},
    ],
    'teacher': 'Stockfish 19', 'teacher_nodes': 2000, 'teacher_threads': 1, 'teacher_hash_mb': 16,
    'value_cp_scale': 600., 'mate_cp': 10000, 'aux_actions': 4,
    'epochs': 6, 'batch_size': 128, 'learning_rate': .001, 'value_loss_weight': .5,
    'aux_loss_weight': .25, 'gradient_clip': 1., 'torch_threads': 2, 'device': 'mps',
    'deterministic_algorithms': False,
    'reproducibility': 'Fixed data, seeds, initial weights and batch order; MPS indexing gradients are not bitwise deterministic',
    'fit_order_seed': 920017, 'epoch_order_seed_base': 920029,
    'optimizer': {'name': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.},
    'loss': 'Legal candidate CE + 0.5 bounded value MSE + 0.25 auxiliary CE for reconstruct/predict only',
    'auxiliary': 'Four fixed distinct uniformly sampled legal actions per board, or all if fewer; same in all fits',
    'aux_loss': 'Equal weight per example-action to mean changed-square and unchanged-square CE; masks from true transition',
    'targets': 'predict: successor pieces in original mover perspective; reconstruct: current pieces; other arms aux weight zero',
    'compute': 'Every arm computes and backpropagates through the auxiliary branch, including zero-weight controls',
    'initialization': 'All recurrent modes have identical same-seed initial state; CNN starts with four copies of the same core',
    'comparison': 'Recurrent aux arms match parameters and operations; CNN has more parameters at equal depth operations',
    'split': 'Separate generated games; global dedup of pieces/side/castling/EP and mirrored equivalents; exclude all v1 states',
    'inference': 'Four internal refinements per FEN, shared source/destination scorer, no search or across-move memory',
    'regret_nodes': 20000, 'regret_positions_per_split': 128, 'regret_selection_seed': 950001,
    'regret': 'Secondary: fresh unrestricted and each unique chosen root move at fixed nodes; bounded tanh score difference, can be negative',
    'selection': 'Final checkpoints only, no early stopping, retries, budget extension or test-driven depth selection',
    'evaluation': 'Every final fit on all dev and shifted positions; both are development, not untouched confirmation',
    'continuation_gate': {'min_mean_agreement_gain': .02, 'max_paired_seed_deficit': .01,
                          'min_future_changed_accuracy': .70,
                          'comparison_modes': ['cnn', 'recurrent', 'reconstruct'],
                          'scope': 'Descriptive development gate only; fresh confirmation required for efficacy claims'},
    'claim_scope': 'Dynamics-supervised representation learning; no novel planning, world-model control, biological or Elo claim',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n')
    stream.flush()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def excluded_states():
    return {state_key(chess.Board(row['fen'])) for name in EXCLUSIONS for row in jsonl(ROOT/name)}


def signature(engine_path=ENGINE):
    return {'protocol': PROTOCOL, 'sources': {name: sha(ROOT/name) for name in SOURCES},
            'exclusions': {name: sha(ROOT/name) for name in EXCLUSIONS},
            'excluded_positions': len(excluded_states()),
            'engine': {'path': str(Path(engine_path).resolve()), 'sha256': sha(engine_path)},
            'dependencies': {name: importlib.metadata.version(name) for name in ('chess', 'torch', 'numpy')},
            'python': platform.python_version(), 'hardware': platform.machine(),
            'platform': platform.platform(), 'lock_sha256': sha(ROOT/'uv.lock'),
            'pyproject_sha256': sha(ROOT/'pyproject.toml'),
            'parameters': {mode: SpatialChess(mode, 17).parameter_count() for mode in MODES}}


def prepare(out, engine_path=ENGINE):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', signature(engine_path))
    write_new(out/'prepared.json', {'status': 'prepared', 'created_unix': time.time(),
                                   'plan_sha256': sha(out/'plan.json'), 'training_started': False})
    return out/'plan.json'


def verify_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan != signature(plan['engine']['path']):
        raise ValueError('Frozen plan/source/dependency/engine/exclusion mismatch')
    return plan


def tensors(rows):
    """Teacher labels stay separate from board and legal-candidate model inputs."""
    encoded, candidates, targets, current, future, actions, aux_masks, ids = [], [], [], [], [], [], [], []
    max_legal = max(chess.Board(row['fen']).legal_moves.count() for row in rows)
    for row in rows:
        board = chess.Board(row['fen'])
        names, features = encode_candidates(board)
        ids.append(names)
        encoded.append(encode_board(board))
        padded = np.zeros((max_legal, 5), dtype=np.int64)
        padded[:len(names)] = features
        candidates.append(padded)
        targets.append(names.index(row['target_uci']))
        current.append(piece_targets(board, board.turn))
        next_targets = np.zeros((PROTOCOL['aux_actions'], 8, 8), dtype=np.int64)
        action_features = np.zeros((PROTOCOL['aux_actions'], 5), dtype=np.int64)
        mask = np.zeros(PROTOCOL['aux_actions'], dtype=bool)
        for index, transition in enumerate(row['aux_transitions']):
            action_features[index] = features[names.index(transition['uci'])]
            next_targets[index] = piece_targets(chess.Board(transition['next_fen']), board.turn)
            mask[index] = True
        future.append(next_targets)
        actions.append(action_features)
        aux_masks.append(mask)
    legal = np.arange(max_legal)[None, :] < np.array([len(names) for names in ids])[:, None]
    arrays = {'observations': np.stack(encoded), 'candidates': np.stack(candidates), 'mask': legal,
              'targets': np.array(targets), 'values': np.array([row['target_value'] for row in rows], np.float32),
              'current': np.stack(current), 'future': np.stack(future), 'actions': np.stack(actions),
              'aux_mask': np.stack(aux_masks)}
    return {name: torch.from_numpy(value) for name, value in arrays.items()}, ids


def auxiliary_loss(logits, target, changed, valid):
    """Balanced changed/unchanged loss, then average valid transitions per batch."""
    ce = F.cross_entropy(logits, target, reduction='none')
    changed = changed.to(ce.dtype)
    changed_loss = (ce*changed).sum((1, 2))/changed.sum((1, 2)).clamp_min(1)
    unchanged_loss = (ce*(1-changed)).sum((1, 2))/(1-changed).sum((1, 2)).clamp_min(1)
    if not valid.any():
        raise ValueError('No valid auxiliary action')
    return ((changed_loss+unchanged_loss)*.5)[valid].mean()


def auxiliary_forward(model, hidden, batch, mode):
    count = batch['actions'].shape[1]
    repeated = hidden.repeat_interleave(count, dim=0)
    logits = model.predict_board(repeated, batch['actions'].flatten(0, 1))
    current = batch['current'].repeat_interleave(count, dim=0)
    future = batch['future'].flatten(0, 1)
    changed = current != future
    target = current if mode == 'reconstruct' else future
    return logits, target, future, changed, batch['aux_mask'].flatten()


def batch_at(data, indices):
    return {name: value[indices].to(PROTOCOL['device']) for name, value in data.items()}


def summarize_predictions(rows):
    changes = sum(row['future_changed_count'] for row in rows)
    transitions = sum(row['aux_transition_count'] for row in rows)
    return {'examples': len(rows), 'top1_teacher_agreement': float(np.mean([r['correct'] for r in rows])),
            'target_nll': float(np.mean([r['target_nll'] for r in rows])),
            'value_mae': float(np.mean([abs(r['value']-r['target_value']) for r in rows])),
            'future_changed_accuracy': sum(r['future_changed_correct'] for r in rows)/changes,
            'future_square_accuracy': sum(r['future_square_correct'] for r in rows)/(64*transitions),
            'future_exact_board_accuracy': sum(r['future_exact_count'] for r in rows)/transitions,
            'aux_target_square_accuracy': sum(r['aux_target_square_correct'] for r in rows)/(64*transitions),
            'copy_current_square_accuracy': 1-changes/(64*transitions),
            'copy_current_changed_accuracy': 0., 'copy_current_exact_board_accuracy': 0.}


@torch.no_grad()
def evaluate(model, data, rows, names):
    model.eval()
    predictions = []
    started = time.perf_counter()
    for start in range(0, len(rows), PROTOCOL['batch_size']):
        stop = min(start+PROTOCOL['batch_size'], len(rows))
        batch = batch_at(data, slice(start, stop))
        logits, value, hidden = model(batch['observations'], batch['candidates'], batch['mask'])
        choices = logits.argmax(-1).cpu().tolist()
        log_probs = logits.log_softmax(-1)
        nll = (-log_probs.gather(1, batch['targets'][:, None]).squeeze(1)).cpu().tolist()
        aux, target, future, changed, valid = auxiliary_forward(model, hidden, batch, model.mode)
        guessed = aux.argmax(1)
        count = batch['actions'].shape[1]
        valid_squares = valid[:, None, None]
        aggregate = {
            'future_changed_correct': ((guessed == future) & changed & valid_squares).sum((1, 2)),
            'future_changed_count': (changed & valid_squares).sum((1, 2)),
            'future_square_correct': ((guessed == future) & valid_squares).sum((1, 2)),
            'future_exact_count': (guessed == future).flatten(1).all(1) & valid,
            'aux_target_square_correct': ((guessed == target) & valid_squares).sum((1, 2)),
            'aux_transition_count': valid,
        }
        aggregate = {key: val.reshape(-1, count).sum(1).cpu().tolist() for key, val in aggregate.items()}
        values = value.cpu().tolist()
        for offset, row in enumerate(rows[start:stop]):
            choice = names[start+offset][choices[offset]]
            predictions.append({'id': row['id'], 'game_id': row['game_id'], 'target': row['target_uci'],
                                'choice': choice, 'correct': choice == row['target_uci'],
                                'target_nll': nll[offset], 'target_probability': math.exp(-nll[offset]),
                                'value': values[offset], 'target_value': row['target_value'],
                                **{key: val[offset] for key, val in aggregate.items()}})
    return {'metrics': summarize_predictions(predictions), 'predictions': predictions,
            'evaluation_wall_seconds': time.perf_counter()-started}


def fit(mode, seed, tensor_sets, rows, names, out, plan_hash, data_hash):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'mode': mode, 'seed': seed, 'started_unix': time.time(),
                                 'plan_sha256': plan_hash, 'data_receipt_sha256': data_hash})
    model = SpatialChess(mode, seed).to(PROTOCOL['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 betas=tuple(PROTOCOL['optimizer']['betas']),
                                 eps=PROTOCOL['optimizer']['eps'], weight_decay=0.)
    generator = torch.Generator().manual_seed(PROTOCOL['epoch_order_seed_base']+seed)
    train = tensor_sets['train']
    aux_weight = PROTOCOL['aux_loss_weight'] if mode in ('reconstruct', 'predict') else 0.
    updates, examples = 0, 0
    start = time.perf_counter()
    with (out/'learning.jsonl').open('x') as stream:
        for epoch in range(PROTOCOL['epochs']):
            order = torch.randperm(len(rows['train']), generator=generator)
            for position in range(0, len(order), PROTOCOL['batch_size']):
                batch = batch_at(train, order[position:position+PROTOCOL['batch_size']])
                model.train()
                logits, values, hidden = model(batch['observations'], batch['candidates'], batch['mask'])
                policy = F.cross_entropy(logits, batch['targets'])
                value = F.mse_loss(values, batch['values'])
                aux_logits, target, _, changed, valid = auxiliary_forward(model, hidden, batch, mode)
                aux = auxiliary_loss(aux_logits, target, changed, valid)
                loss = policy+PROTOCOL['value_loss_weight']*value+aux_weight*aux
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss; fixed run stops without replacement')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['gradient_clip'],
                                                          error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                examples += len(batch['targets'])
                append(stream, {'epoch': epoch+1, 'update': updates, 'examples_seen': examples,
                                'policy_ce': float(policy.detach()), 'value_mse': float(value.detach()),
                                'aux_ce': float(aux.detach()), 'aux_weight': aux_weight,
                                'loss': float(loss.detach()), 'gradient_norm': float(gradient)})
            print(json.dumps({'mode': mode, 'seed': seed, 'epoch': epoch+1,
                              'updates': updates, 'wall_seconds': time.perf_counter()-start}), flush=True)
    train_seconds = time.perf_counter()-start
    model.save(out/'weights.pt', plan_sha256=plan_hash)
    for split in ('dev', 'shift'):
        write_new(out/f'{split}.json', evaluate(model, tensor_sets[split], rows[split], names[split]))
    files = ['weights.pt', 'learning.jsonl', 'dev.json', 'shift.json']
    result = {'status': 'completed', 'mode': mode, 'seed': seed, 'completed_unix': time.time(),
              'updates': updates, 'examples_seen': examples, 'training_wall_seconds': train_seconds,
              'parameter_count': model.parameter_count(), 'plan_sha256': plan_hash,
              'data_receipt_sha256': data_hash, 'files': {name: sha(out/name) for name in files}}
    write_new(out/'completed.json', result)
    return result


def score_regret(engine_path, out, rows, baselines):
    """Secondary stronger-engine assessment. Shared choices reuse one identical call."""
    directory = out/'regret'
    directory.mkdir(exist_ok=False)
    selected = {}
    choices = {}
    for split in ('dev', 'shift'):
        selected[split] = sorted(random.Random(PROTOCOL['regret_selection_seed']+(split == 'shift')).sample(
            range(len(rows[split])), min(len(rows[split]), PROTOCOL['regret_positions_per_split'])))
        choices[split] = {f'{mode}-{seed}': json.loads((out/f'{mode}-{seed}'/f'{split}.json').read_text())['predictions']
                          for mode in MODES for seed in PROTOCOL['seeds']}
        for name in ('greedy_material', 'training_move_frequency'):
            choices[split][name] = baselines[split][name]['predictions']
    write_new(directory/'selection.json', selected)
    costs = {'calls': 0, 'requested_nodes': 0, 'reported_nodes': 0, 'wall_seconds': 0.}
    results = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine, (directory/'analyses.jsonl').open('x') as stream:
        engine.configure({'Threads': 1, 'Hash': 16})
        for split in ('dev', 'shift'):
            for index in selected[split]:
                row, lookup = rows[split][index], {}
                board = chess.Board(row['fen'])
                moves = sorted({preds[index]['choice'] for preds in choices[split].values()})
                for chosen in [None]+moves:
                    engine.configure({'Clear Hash': None})
                    begin = time.perf_counter()
                    kwargs = {} if chosen is None else {'root_moves': [chess.Move.from_uci(chosen)]}
                    info = engine.analyse(board.copy(stack=False), chess.engine.Limit(nodes=PROTOCOL['regret_nodes']),
                                          info=chess.engine.INFO_SCORE | chess.engine.INFO_PV | chess.engine.INFO_BASIC,
                                          **kwargs)
                    score = info['score'].pov(board.turn)
                    cp = score.score(mate_score=PROTOCOL['mate_cp'])
                    if cp is None or (chosen is not None and info['pv'][0].uci() != chosen):
                        raise ValueError('Invalid secondary engine assessment')
                    analysis = {'split': split, 'id': row['id'], 'fen': row['fen'], 'root_move': chosen,
                                'pv_first': info['pv'][0].uci(), 'score_cp': cp, 'mate': score.mate(),
                                'bounded_score': math.tanh(cp/PROTOCOL['value_cp_scale']),
                                'requested_nodes': PROTOCOL['regret_nodes'], 'reported_nodes': info.get('nodes', 0),
                                'wall_seconds': time.perf_counter()-begin}
                    append(stream, analysis)
                    lookup[chosen] = analysis
                    costs['calls'] += 1
                    for key in ('requested_nodes', 'reported_nodes', 'wall_seconds'):
                        costs[key] += analysis[key]
                for name, predictions in choices[split].items():
                    chosen = predictions[index]['choice']
                    results.append({'split': split, 'id': row['id'], 'policy': name, 'choice': chosen,
                                    'best_score': lookup[None]['bounded_score'],
                                    'chosen_score': lookup[chosen]['bounded_score'],
                                    'bounded_regret': lookup[None]['bounded_score']-lookup[chosen]['bounded_score'],
                                    'cp_regret': lookup[None]['score_cp']-lookup[chosen]['score_cp']})
    write_new(directory/'predictions.json', results)
    means = {split: {name: {'mean_bounded_regret': float(np.mean([r['bounded_regret'] for r in results
                                                               if r['split'] == split and r['policy'] == name])),
                            'mean_cp_regret': float(np.mean([r['cp_regret'] for r in results
                                                           if r['split'] == split and r['policy'] == name]))}
                    for name in choices[split]} for split in ('dev', 'shift')}
    receipt = {'status': 'completed', 'cost': costs, 'means': means,
               'scope': 'Development secondary engine score; finite-search differences may be negative, not true regret or Elo',
               'files': {name: sha(directory/name) for name in ('selection.json', 'analyses.jsonl', 'predictions.json')}}
    write_new(directory/'completed.json', receipt)
    return receipt


def run(plan_path, out):
    plan = verify_plan(plan_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan_hash = sha(plan_path)
    write_new(out/'started.json', {'started_unix': time.time(), 'plan_sha256': plan_hash})
    try:
        torch.set_num_threads(PROTOCOL['torch_threads'])
        torch.use_deterministic_algorithms(PROTOCOL['deterministic_algorithms'])
        with chess.engine.SimpleEngine.popen_uci(plan['engine']['path']) as engine:
            if not engine.id.get('name', '').startswith(PROTOCOL['teacher']):
                raise ValueError('Wrong engine identity')
            engine.configure({'Threads': 1, 'Hash': 16})
            write_new(out/'engine.json', {'id': engine.id, **plan['engine']})
            generate_data(out/'data', lambda board: stockfish_label(engine, board, PROTOCOL),
                          PROTOCOL, excluded_states())
        rows = validate_data(out/'data', PROTOCOL, excluded_states())
        tensor_sets, names = {}, {}
        for split in rows:
            tensor_sets[split], names[split] = tensors(rows[split])
        jobs = [(mode, seed) for mode in MODES for seed in PROTOCOL['seeds']]
        random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
        results = []
        for mode, seed in jobs:
            results.append(fit(mode, seed, tensor_sets, rows, names, out/f'{mode}-{seed}',
                               plan_hash, sha(out/'data/completed.json')))
        from openjev.research.chess_spatial_baselines import simple_baselines
        baselines = {split: simple_baselines(rows['train'], rows[split]) for split in ('dev', 'shift')}
        write_new(out/'baselines.json', baselines)
        score_regret(plan['engine']['path'], out, rows, baselines)
        write_new(out/'completed.json', {'status': 'completed', 'completed_unix': time.time(),
                                        'baseline_sha256': sha(out/'baselines.json'),
                                        'regret_receipt_sha256': sha(out/'regret/completed.json'),
                                        'plan_sha256': plan_hash, 'fit_order': jobs, 'fits': len(results),
                                        'total_updates': sum(r['updates'] for r in results),
                                        'training_wall_seconds': sum(r['training_wall_seconds'] for r in results),
                                        'fit_receipts': {f'{mode}-{seed}': sha(out/f'{mode}-{seed}/completed.json')
                                                         for mode, seed in jobs}})
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'completed_unix': time.time(),
                                     'error_type': type(exc).__name__, 'error': str(exc),
                                     'plan_sha256': plan_hash})
        raise


def continuation_gate(fits):
    expected = {(mode, seed, split) for mode in MODES for seed in PROTOCOL['seeds']
                for split in ('dev', 'shift')}
    actual = {(row['mode'], row['seed'], row['split']) for row in fits}
    if actual != expected or len(fits) != len(expected):
        raise ValueError('Gate requires the complete fixed fit panel')
    lookup = {(r['mode'], r['seed'], r['split']): r for r in fits}
    config = PROTOCOL['continuation_gate']
    checks = []
    for split in ('dev', 'shift'):
        for control in config['comparison_modes']:
            deltas = [lookup['predict', seed, split]['top1_teacher_agreement']-
                      lookup[control, seed, split]['top1_teacher_agreement'] for seed in PROTOCOL['seeds']]
            checks.append({'split': split, 'control': control, 'metric': 'mean_agreement_gain',
                           'observed': float(np.mean(deltas)), 'threshold': config['min_mean_agreement_gain'],
                           'operator': '>=', 'passed': bool(np.mean(deltas) >= config['min_mean_agreement_gain'])})
            checks.append({'split': split, 'control': control, 'metric': 'worst_paired_seed_gain',
                           'observed': min(deltas), 'threshold': -config['max_paired_seed_deficit'],
                           'operator': '>=', 'passed': min(deltas) >= -config['max_paired_seed_deficit']})
        accuracy = np.mean([lookup['predict', seed, split]['future_changed_accuracy'] for seed in PROTOCOL['seeds']])
        checks.append({'split': split, 'metric': 'future_changed_accuracy', 'observed': float(accuracy),
                       'threshold': config['min_future_changed_accuracy'], 'operator': '>=',
                       'passed': bool(accuracy >= config['min_future_changed_accuracy'])})
    return {'passed': bool(all(c['passed'] for c in checks)), 'checks': checks, 'scope': config['scope']}


def validate_prediction(predicted, row):
    board = chess.Board(row['fen'])
    if (predicted['id'] != row['id'] or predicted['game_id'] != row['game_id'] or
            predicted['target'] != row['target_uci'] or
            predicted['correct'] != (predicted['choice'] == row['target_uci']) or
            predicted['target_value'] != row['target_value'] or
            chess.Move.from_uci(predicted['choice']) not in board.legal_moves or
            not math.isfinite(predicted['target_nll']) or predicted['target_nll'] < 0 or
            not math.isfinite(predicted['value']) or not -1 <= predicted['value'] <= 1 or
            not math.isclose(predicted['target_probability'], math.exp(-predicted['target_nll']),
                             rel_tol=1e-10, abs_tol=1e-15)):
        raise ValueError('Invalid evaluation prediction or probability')
    count = len(row['aux_transitions'])
    current = piece_targets(board, board.turn)
    changed = sum(int(np.count_nonzero(current != piece_targets(chess.Board(t['next_fen']), board.turn)))
                  for t in row['aux_transitions'])
    bounds = {'aux_transition_count': (count, count), 'future_changed_count': (changed, changed),
              'future_changed_correct': (0, changed), 'future_square_correct': (0, 64*count),
              'future_exact_count': (0, count), 'aux_target_square_correct': (0, 64*count)}
    if any(type(predicted[key]) is not int or not low <= predicted[key] <= high
           for key, (low, high) in bounds.items()):
        raise ValueError('Impossible auxiliary evaluation counts')
    if (predicted['future_changed_correct'] > predicted['future_square_correct'] or
            predicted['future_exact_count']*64 > predicted['future_square_correct']):
        raise ValueError('Inconsistent auxiliary evaluation counts')


def validate_learning(logs, mode, examples):
    steps = math.ceil(examples/PROTOCOL['batch_size'])
    weight = PROTOCOL['aux_loss_weight'] if mode in ('predict', 'reconstruct') else 0.
    if len(logs) != steps*PROTOCOL['epochs']:
        raise ValueError('Training update count mismatch')
    for index, row in enumerate(logs):
        epoch, batch = divmod(index, steps)
        expected_seen = epoch*examples+min((batch+1)*PROTOCOL['batch_size'], examples)
        if (row['update'] != index+1 or row['epoch'] != epoch+1 or row['examples_seen'] != expected_seen or
                row['aux_weight'] != weight or
                any(not math.isfinite(row[key]) or row[key] < 0
                    for key in ('policy_ce', 'value_mse', 'aux_ce', 'loss', 'gradient_norm')) or
                not math.isclose(row['loss'], row['policy_ce']+PROTOCOL['value_loss_weight']*row['value_mse']+
                                 weight*row['aux_ce'], rel_tol=1e-5, abs_tol=1e-5)):
            raise ValueError('Training log budget or loss mismatch')


def report(plan_path, execution, out):
    verify_plan(plan_path)
    execution = Path(execution)
    complete = json.loads((execution/'completed.json').read_text())
    plan_hash = sha(plan_path)
    if (complete['status'] != 'completed' or complete['plan_sha256'] != plan_hash or
            (execution/'failed.json').exists()):
        raise ValueError('Execution did not complete under this plan')
    rows = validate_data(execution/'data', PROTOCOL, excluded_states())
    data_hash = sha(execution/'data/completed.json')
    expected_updates = PROTOCOL['epochs']*math.ceil(len(rows['train'])/PROTOCOL['batch_size'])
    expected_names = {f'{mode}-{seed}' for mode in MODES for seed in PROTOCOL['seeds']}
    if (set(complete['fit_receipts']) != expected_names or complete['fits'] != len(expected_names) or
            complete['total_updates'] != expected_updates*len(expected_names)):
        raise ValueError('Incomplete or extra fit panel')
    if sha(execution/'baselines.json') != complete['baseline_sha256']:
        raise ValueError('Baseline hash mismatch')
    regret = json.loads((execution/'regret/completed.json').read_text())
    if sha(execution/'regret/completed.json') != complete['regret_receipt_sha256']:
        raise ValueError('Regret receipt hash mismatch')
    for name, digest in regret['files'].items():
        if sha(execution/'regret'/name) != digest:
            raise ValueError('Regret evidence hash mismatch')
    from openjev.research.chess_spatial_baselines import simple_baselines
    from openjev.research.chess_spatial_regret import validate_regret
    baselines = json.loads((execution/'baselines.json').read_text())
    recomputed = {split: simple_baselines(rows['train'], rows[split]) for split in ('dev', 'shift')}
    if baselines != recomputed:
        raise ValueError('Baseline predictions or aggregates disagree with their fixed rules')
    choices = {split: {f'{mode}-{seed}': json.loads((execution/f'{mode}-{seed}'/f'{split}.json').read_text())['predictions']
                       for mode in MODES for seed in PROTOCOL['seeds']} for split in ('dev', 'shift')}
    for split in ('dev', 'shift'):
        for name in ('greedy_material', 'training_move_frequency'):
            choices[split][name] = baselines[split][name]['predictions']
    validate_regret(execution/'regret', rows, choices, PROTOCOL)
    summaries = []
    for mode in MODES:
        for seed in PROTOCOL['seeds']:
            name = f'{mode}-{seed}'
            directory = execution/name
            receipt = json.loads((directory/'completed.json').read_text())
            if (sha(directory/'completed.json') != complete['fit_receipts'][name] or
                    receipt['plan_sha256'] != plan_hash or receipt['data_receipt_sha256'] != data_hash or
                    receipt['mode'] != mode or receipt['seed'] != seed or receipt['status'] != 'completed' or
                    receipt['updates'] != expected_updates or
                    receipt['examples_seen'] != len(rows['train'])*PROTOCOL['epochs']):
                raise ValueError('Fit receipt or budget mismatch')
            if set(receipt['files']) != {'weights.pt', 'learning.jsonl', 'dev.json', 'shift.json'}:
                raise ValueError('Missing required fit evidence')
            for filename, digest in receipt['files'].items():
                if sha(directory/filename) != digest:
                    raise ValueError('Fit evidence hash mismatch')
            model = SpatialChess.load(directory/'weights.pt', expected_plan_sha256=plan_hash)
            if model.mode != mode or model.seed != seed or model.parameter_count() != receipt['parameter_count']:
                raise ValueError('Model identity mismatch')
            logs = jsonl(directory/'learning.jsonl')
            validate_learning(logs, mode, len(rows['train']))
            for split in ('dev', 'shift'):
                result = json.loads((directory/f'{split}.json').read_text())
                predictions = result['predictions']
                if len(predictions) != len(rows[split]):
                    raise ValueError('Missing evaluation positions')
                for predicted, row in zip(predictions, rows[split], strict=True):
                    validate_prediction(predicted, row)
                metrics = summarize_predictions(predictions)
                if metrics != result['metrics']:
                    raise ValueError('Evaluation aggregate mismatch')
                summaries.append({'mode': mode, 'seed': seed, 'split': split, **metrics,
                                  'training_wall_seconds': receipt['training_wall_seconds'],
                                  'parameter_count': receipt['parameter_count'],
                                  'checkpoint_sha256': receipt['files']['weights.pt']})
    measured_training = sum(r['training_wall_seconds'] for r in summaries if r['split'] == 'dev')
    if not math.isclose(measured_training, complete['training_wall_seconds'], rel_tol=1e-12):
        raise ValueError('Training wall time mismatch')
    means = {split: {mode: {metric: float(np.mean([r[metric] for r in summaries
                                                  if r['mode'] == mode and r['split'] == split]))
                            for metric in ('top1_teacher_agreement', 'target_nll', 'value_mae',
                                           'future_changed_accuracy', 'future_square_accuracy',
                                           'future_exact_board_accuracy')}
                    for mode in MODES} for split in ('dev', 'shift')}
    summary = {'status': 'completed', 'plan_sha256': plan_hash, 'fits': summaries, 'means': means,
               'continuation_gate': continuation_gate(summaries), 'novelty_established': False,
               'elo_estimate': None, 'claim_scope': PROTOCOL['claim_scope'], 'protocol': PROTOCOL,
               'teacher_cost': json.loads((execution/'data/completed.json').read_text()),
               'training_wall_seconds': complete['training_wall_seconds'],
               'baselines': json.loads((execution/'baselines.json').read_text()), 'regret': regret}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'summary.json', summary)
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                                    'summary_sha256': sha(out/'summary.json'),
                                    'execution_receipt_sha256': sha(execution/'completed.json')})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ENGINE)
    for action in ('run', 'report'):
        p = sub.add_parser(action)
        p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        if action == 'report':
            p.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.out, args.engine))
    elif args.command == 'run':
        run(args.plan, args.out)
    else:
        print(json.dumps(report(args.plan, args.execution, args.out)['means'], indent=2))


if __name__ == '__main__':
    main()
