"""Frozen, bounded Stockfish distillation pilot for compact chess students.

prepare writes a plan only. run generates fresh, game-separated data and trains
all nine final fits. report verifies saved evidence and publishes aggregate
development results. No paid model calls, existing puzzle labels, Elo estimates,
checkpoint selection, or biological-connectome claims are part of this pilot.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.chess_student import (
    DEPTH,
    INPUT_SIZE,
    MODES,
    MOVE_TO_INDEX,
    MOVE_VOCAB_SHA256,
    UCI_MOVES,
    WIDTH,
    ChessStudent,
    circuit_topology,
    encode_board,
    legal_mask,
    state_key,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT/'runs/chess-inputs/stockfish/stockfish-macos-universal'
PROTOCOL = {
    'version': 'chess-student-v1', 'modes': list(MODES), 'seeds': [17, 29, 43],
    'width': WIDTH, 'depth': DEPTH, 'input_size': INPUT_SIZE,
    'vocabulary_size': len(UCI_MOVES), 'vocabulary_sha256': MOVE_VOCAB_SHA256,
    'train_examples': 4096, 'dev_examples': 1024, 'train_game_cap': 160, 'dev_game_cap': 64,
    'game_max_plies': 40, 'game_seed_base': 81_000_000, 'random_move_probability': .5,
    'teacher': 'Stockfish 19', 'teacher_nodes': 2000, 'teacher_threads': 1, 'teacher_hash_mb': 16,
    'clear_hash_each_analysis': True, 'teacher_history': 'FEN only, no earlier move stack',
    'value_target': 'tanh(side-to-move centipawns / 600); mate score represented as signed 10000 cp',
    'value_cp_scale': 600., 'mate_cp': 10000, 'value_loss_weight': .5,
    'epochs': 3, 'batch_size': 64, 'optimizer': 'Adam', 'learning_rate': .001,
    'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0., 'gradient_clip': 1.,
    'torch_threads': 2, 'device': 'cpu', 'fit_order_seed': 815017,
    'epoch_order_seed_base': 815029,
    'data_split': 'Disjoint generated game IDs; global exact state dedup ignores halfmove/fullmove counters',
    'topology': 'Synthetic signed 16 sensory/24 inter/16 command/8 motor circuit, 352 edges',
    'control': '20000 double-edge-swap proposals preserve each node signed in/out degrees',
    'observation': 'Numeric FEN pieces, side, castling, EP and counters; no teacher scores or history input',
    'loss': 'Final-depth legal-masked move cross entropy plus 0.5 bounded-value MSE',
    'evaluation': 'All final checkpoints on the same 1024 generated development positions',
    'selection': 'No early stopping, checkpoint selection, retries, replacements or budget extension',
    'claim_scope': 'Initial supervised chess distillation baseline, not RL, Elo, novelty or imported biology',
    'continuation_gate': {
        'circuit_mean_agreement_gain_over_rewired': .03,
        'circuit_mean_agreement_max_deficit_to_gru': .03,
        'circuit_paired_seed_max_deficit_to_rewired': .05,
        'circuit_mean_value_mae_max_excess_over_rewired': .03,
        'scope': 'Descriptive development gate to consider a larger study, not efficacy or publication evidence',
    },
}
SOURCES = ['scripts/train_chess_student.py', 'src/openjev/research/chess_student.py',
           'tests/test_chess_student.py']


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


def signature(engine_path):
    engine_path = Path(engine_path).resolve()
    return {'protocol': PROTOCOL, 'sources': {name: sha(ROOT/name) for name in SOURCES},
            'dependencies': {name: importlib.metadata.version(name)
                             for name in ('python-chess', 'chess', 'torch', 'numpy')},
            'python': platform.python_version(), 'lock_sha256': sha(ROOT/'uv.lock'),
            'engine': {'path': str(engine_path), 'sha256': sha(engine_path)},
            'model_parameters': {mode: ChessStudent(mode).parameter_count() for mode in MODES},
            'structured_topology_sha256': hashlib.sha256(circuit_topology().numpy().tobytes()).hexdigest(),
            'rewired_topology_sha256': hashlib.sha256(circuit_topology(True).numpy().tobytes()).hexdigest()}


def prepare(out, engine_path=DEFAULT_ENGINE):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', signature(engine_path))
    write_new(out/'prepared.json', {'status': 'prepared', 'plan_sha256': sha(out/'plan.json'),
                                   'created_unix': time.time(), 'training_started': False})
    return out/'plan.json'


def verify_plan(path):
    value = json.loads(Path(path).read_text())
    if value != signature(value['engine']['path']):
        raise ValueError('Frozen plan, source, dependency or engine hash mismatch')
    if value['dependencies']['python-chess'] != '1.999' or value['dependencies']['chess'] != '1.11.2':
        raise ValueError('Expected pinned python-chess==1.999 and chess==1.11.2')
    return value


def stockfish_label(engine, board, config):
    """Engine teacher evidence stays outside the numeric model observation."""
    engine.configure({'Clear Hash': None})
    begin = time.perf_counter()
    info = engine.analyse(board.copy(stack=False), chess.engine.Limit(nodes=config['teacher_nodes']),
                          info=chess.engine.INFO_SCORE | chess.engine.INFO_PV | chess.engine.INFO_BASIC)
    elapsed = time.perf_counter()-begin
    move = info['pv'][0]
    if move not in board.legal_moves:
        raise ValueError('Teacher returned an illegal or absent principal-variation move')
    score = info['score'].pov(board.turn)
    cp = score.score(mate_score=config['mate_cp'])
    if cp is None:
        raise ValueError('Teacher returned no numeric score')
    return {'target_uci': move.uci(), 'score_cp': int(cp), 'mate': score.mate(),
            'target_value': math.tanh(cp/config['value_cp_scale']),
            'reported_nodes': int(info.get('nodes', 0)), 'requested_nodes': config['teacher_nodes'],
            'wall_seconds': elapsed}


def generate_data(out, labeler, *, config=None):
    """Generate disjoint games and globally unique states, with durable partials.

    A caller-owned engine supplies labeler(board). An insufficient fixed game cap
    raises without replacing examples or extending the cap. This injection also
    permits small mechanical tests without running Stockfish or training models.
    """
    config = PROTOCOL if config is None else config
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_new(out/'started.json', {'started_unix': started, 'config': config})
    seen, counts = set(), {}
    engine_calls, requested_nodes, reported_nodes = 0, 0, 0
    label_requested_nodes, label_reported_nodes = 0, 0
    teacher_wall = 0.
    try:
        with (out/'games.jsonl').open('x') as games, (out/'analyses.jsonl').open('x') as analyses:
            for split, game_offset in [('train', 0), ('dev', config['train_game_cap'])]:
                count = 0
                with (out/f'{split}.jsonl').open('x') as records:
                    for local_game in range(config[f'{split}_game_cap']):
                        game_id = game_offset+local_game
                        seed = config['game_seed_base']+game_id
                        rng, board = random.Random(seed), chess.Board()
                        sampled, actions = 0, []
                        for ply in range(config['game_max_plies']):
                            if board.outcome(claim_draw=False) is not None:
                                break
                            fen, key = board.fen(en_passant='fen'), state_key(board)
                            random_action = rng.random() < config['random_move_probability']
                            label = None
                            if key not in seen or not random_action:
                                label = labeler(board)
                                target = chess.Move.from_uci(label['target_uci'])
                                if target not in board.legal_moves or not math.isfinite(label['target_value']):
                                    raise ValueError('Invalid teacher label')
                                if not -1 <= label['target_value'] <= 1:
                                    raise ValueError('Teacher value must lie in [-1,1]')
                                engine_calls += 1
                                requested_nodes += label['requested_nodes']
                                reported_nodes += label['reported_nodes']
                                teacher_wall += label['wall_seconds']
                                if key not in seen:
                                    label_requested_nodes += label['requested_nodes']
                                    label_reported_nodes += label['reported_nodes']
                                append(analyses, {'split': split, 'game_id': game_id, 'ply': ply,
                                                  'fen': fen, 'labelled_position': key not in seen, **label})
                            if key not in seen:
                                row = {'id': f'{split}-{count:05d}', 'split': split, 'game_id': game_id,
                                       'game_seed': seed, 'ply': ply, 'fen': fen, 'state_key': key, **label}
                                append(records, row)
                                seen.add(key)
                                sampled += 1
                                count += 1
                            if count == config[f'{split}_examples']:
                                break
                            legal = sorted(board.legal_moves, key=lambda move: move.uci())
                            action = rng.choice(legal) if random_action else chess.Move.from_uci(label['target_uci'])
                            actions.append({'uci': action.uci(), 'source': 'random' if random_action else 'teacher'})
                            board.push(action)
                        append(games, {'split': split, 'game_id': game_id, 'seed': seed,
                                       'sampled_positions': sampled, 'actions': actions,
                                       'final_fen': board.fen(en_passant='fen'),
                                       'stop': ('quota' if count == config[f'{split}_examples'] else
                                                'terminal' if board.is_game_over() else 'ply_cap')})
                        if count == config[f'{split}_examples']:
                            break
                counts[split] = count
                if count != config[f'{split}_examples']:
                    raise ValueError(f'{split} generated {count} unique positions before fixed game cap')
        receipt = {'status': 'completed', 'started_unix': started, 'completed_unix': time.time(),
                   'counts': counts, 'unique_states': len(seen), 'teacher_calls': engine_calls,
                   'requested_nodes': requested_nodes, 'reported_nodes': reported_nodes,
                   'label_requested_nodes': label_requested_nodes, 'label_reported_nodes': label_reported_nodes,
                   'rollout_only_teacher_calls': engine_calls-len(seen),
                   'teacher_cost_scope': 'Dataset labels plus teacher-selected rollout moves; no gameplay search',
                   'teacher_wall_seconds': teacher_wall,
                   'files': {name: sha(out/name) for name in
                             ('train.jsonl', 'dev.jsonl', 'games.jsonl', 'analyses.jsonl')}}
        write_new(out/'completed.json', receipt)
        return receipt
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                     'error': str(exc), 'completed_unix': time.time(),
                                     'counts_completed_splits': counts, 'unique_states': len(seen),
                                     'teacher_calls': engine_calls, 'requested_nodes': requested_nodes,
                                     'reported_nodes': reported_nodes, 'teacher_wall_seconds': teacher_wall})
        raise


def validate_data(out, *, config=None):
    config = PROTOCOL if config is None else config
    out = Path(out)
    receipt = json.loads((out/'completed.json').read_text())
    if receipt['status'] != 'completed' or (out/'failed.json').exists():
        raise ValueError('Data generation did not complete successfully')
    for name, digest in receipt['files'].items():
        if sha(out/name) != digest:
            raise ValueError('Data file hash mismatch')
    seen, game_sets, splits = set(), {}, {}
    for split in ('train', 'dev'):
        rows = jsonl(out/f'{split}.jsonl')
        if len(rows) != config[f'{split}_examples'] or receipt['counts'][split] != len(rows):
            raise ValueError('Incorrect frozen data count')
        games = set()
        for index, row in enumerate(rows):
            board = chess.Board(row['fen'])
            if not board.is_valid() or row['state_key'] != state_key(board) or row['state_key'] in seen:
                raise ValueError('Invalid or duplicated state within/across data splits')
            if row['split'] != split or row['id'] != f'{split}-{index:05d}':
                raise ValueError('Invalid data membership or row ordering')
            if chess.Move.from_uci(row['target_uci']) not in board.legal_moves:
                raise ValueError('Nonlegal target move')
            expected_value = math.tanh(row['score_cp']/config['value_cp_scale'])
            if not math.isclose(row['target_value'], expected_value, rel_tol=0., abs_tol=1e-12):
                raise ValueError('Value target does not match the declared teacher transform')
            seen.add(row['state_key'])
            games.add(row['game_id'])
        game_sets[split], splits[split] = games, rows
    if game_sets['train'] & game_sets['dev']:
        raise ValueError('Training and development contain the same generated game')
    analyses = jsonl(out/'analyses.jsonl')
    if (len(analyses) != receipt['teacher_calls'] or
            sum(row['reported_nodes'] for row in analyses) != receipt['reported_nodes'] or
            sum(row['requested_nodes'] for row in analyses) != receipt['requested_nodes'] or
            sum(row['requested_nodes'] for row in analyses if row['labelled_position']) !=
            receipt['label_requested_nodes'] or
            sum(row['reported_nodes'] for row in analyses if row['labelled_position']) !=
            receipt['label_reported_nodes'] or
            sum(not row['labelled_position'] for row in analyses) != receipt['rollout_only_teacher_calls']):
        raise ValueError('Teacher computation receipt mismatch')
    labelled = {(row['split'], row['game_id'], row['ply']): row for row in analyses if row['labelled_position']}
    all_rows = splits['train']+splits['dev']
    if len(labelled) != len(all_rows) or len(seen) != receipt['unique_states']:
        raise ValueError('Teacher labelled-position count mismatch')
    for row in all_rows:
        analysis = labelled.get((row['split'], row['game_id'], row['ply']))
        if analysis is None or any(analysis[key] != row[key] for key in
                                   ('fen', 'target_uci', 'score_cp', 'mate', 'target_value',
                                    'requested_nodes', 'reported_nodes', 'wall_seconds')):
            raise ValueError('Label disagrees with saved engine analysis')
    games = jsonl(out/'games.jsonl')
    game_lookup = {row['game_id']: row for row in games}
    if len(game_lookup) != len(games):
        raise ValueError('Repeated game receipt ID')
    for game in games:
        offset = 0 if game['split'] == 'train' else config['train_game_cap']
        if (not offset <= game['game_id'] < offset+config[f"{game['split']}_game_cap"] or
                game['seed'] != config['game_seed_base']+game['game_id'] or
                len(game['actions']) > config['game_max_plies']):
            raise ValueError('Generated game receipt violates the frozen stream')
        replay = chess.Board()
        positions = {0: replay.fen(en_passant='fen')}
        for ply, action in enumerate(game['actions'], start=1):
            replay.push_uci(action['uci'])
            positions[ply] = replay.fen(en_passant='fen')
        assigned = [row for row in all_rows if row['game_id'] == game['game_id']]
        if (replay.fen(en_passant='fen') != game['final_fen'] or len(assigned) != game['sampled_positions'] or
                any(row['split'] != game['split'] or row['game_seed'] != game['seed'] or
                    positions.get(row['ply']) != row['fen'] for row in assigned)):
            raise ValueError('Data positions do not replay from their declared generated game')
    if any(row['game_id'] not in game_lookup for row in all_rows):
        raise ValueError('Missing generated game receipt')
    return splits


def tensors(rows):
    boards = [chess.Board(row['fen']) for row in rows]
    return {'observations': torch.from_numpy(np.stack([encode_board(board) for board in boards])),
            'mask': torch.from_numpy(np.stack([legal_mask(board) for board in boards])),
            'targets': torch.tensor([MOVE_TO_INDEX[row['target_uci']] for row in rows]),
            'values': torch.tensor([row['target_value'] for row in rows], dtype=torch.float32)}


@torch.no_grad()
def evaluate(model, data, rows):
    model.eval()
    output = []
    begin = time.perf_counter()
    for start in range(0, len(rows), PROTOCOL['batch_size']):
        stop = start+PROTOCOL['batch_size']
        logits, values = model(data['observations'][start:stop], data['mask'][start:stop])
        probabilities = logits.softmax(-1)
        for i, row in enumerate(rows[start:stop]):
            target = MOVE_TO_INDEX[row['target_uci']]
            choice = int(probabilities[i].argmax())
            output.append({'id': row['id'], 'game_id': row['game_id'], 'choice': UCI_MOVES[choice],
                           'target': row['target_uci'], 'correct': choice == target,
                           'target_probability': float(probabilities[i, target]),
                           'value': float(values[i]), 'target_value': row['target_value']})
    metrics = summarize_predictions(output)
    metrics['evaluation_wall_seconds'] = time.perf_counter()-begin
    return {'metrics': metrics, 'predictions': output}


def summarize_predictions(rows):
    return {'examples': len(rows), 'top1_teacher_agreement': float(np.mean([row['correct'] for row in rows])),
            'target_nll': float(np.mean([-math.log(max(row['target_probability'], 1e-12)) for row in rows])),
            'value_mae': float(np.mean([abs(row['value']-row['target_value']) for row in rows])),
            'value_mse': float(np.mean([(row['value']-row['target_value'])**2 for row in rows]))}


def simple_baselines(train_rows, dev_rows):
    """Exact uniform expectation and legal argmax of training-label frequencies."""
    frequencies = Counter(row['target_uci'] for row in train_rows)
    uniform, frequency_rows = [], []
    for row in dev_rows:
        legal = sorted(move.uci() for move in chess.Board(row['fen']).legal_moves)
        uniform.append(1/len(legal))
        choice = max(legal, key=lambda move: frequencies[move])
        frequency_rows.append({'id': row['id'], 'choice': choice, 'target': row['target_uci'],
                               'correct': choice == row['target_uci']})
    return {
        'uniform_random': {'expected_top1_teacher_agreement': float(np.mean(uniform)),
                           'method': 'Exact mean of 1/legal_count; no sampled model inference'},
        'training_move_frequency': {
            'top1_teacher_agreement': float(np.mean([row['correct'] for row in frequency_rows])),
            'method': 'Highest training target-UCI frequency among legal moves; sorted-UCI tie break',
            'predictions': frequency_rows,
        },
    }


def continuation_gate(fits):
    gate = PROTOCOL['continuation_gate']
    by_mode = {mode: {row['seed']: row for row in fits if row['mode'] == mode} for mode in MODES}
    if any(set(rows) != set(PROTOCOL['seeds']) for rows in by_mode.values()):
        raise ValueError('Continuation gate requires every fixed fit seed in every arm')

    def mean(mode, metric):
        return float(np.mean([row[metric] for row in by_mode[mode].values()]))

    agreement = 'top1_teacher_agreement'
    gain = mean('circuit', agreement)-mean('rewired', agreement)
    gru_deficit = mean('gru', agreement)-mean('circuit', agreement)
    seed_deficit = max(by_mode['rewired'][seed][agreement]-by_mode['circuit'][seed][agreement]
                       for seed in PROTOCOL['seeds'])
    value_excess = mean('circuit', 'value_mae')-mean('rewired', 'value_mae')
    checks = [
        {'criterion': 'mean_agreement_gain_over_rewired', 'observed': gain, 'operator': '>=',
         'threshold': gate['circuit_mean_agreement_gain_over_rewired'],
         'passed': gain >= gate['circuit_mean_agreement_gain_over_rewired']},
        {'criterion': 'mean_agreement_deficit_to_gru', 'observed': gru_deficit, 'operator': '<=',
         'threshold': gate['circuit_mean_agreement_max_deficit_to_gru'],
         'passed': gru_deficit <= gate['circuit_mean_agreement_max_deficit_to_gru']},
        {'criterion': 'largest_paired_seed_agreement_deficit_to_rewired', 'observed': seed_deficit,
         'operator': '<=', 'threshold': gate['circuit_paired_seed_max_deficit_to_rewired'],
         'passed': seed_deficit <= gate['circuit_paired_seed_max_deficit_to_rewired']},
        {'criterion': 'mean_value_mae_excess_over_rewired', 'observed': value_excess, 'operator': '<=',
         'threshold': gate['circuit_mean_value_mae_max_excess_over_rewired'],
         'passed': value_excess <= gate['circuit_mean_value_mae_max_excess_over_rewired']},
    ]
    return {'passed': all(check['passed'] for check in checks), 'checks': checks, 'scope': gate['scope']}


def fit(mode, seed, train, dev, dev_rows, out, plan_hash, data_hash):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'started_unix': time.time(), 'mode': mode, 'seed': seed,
                                 'plan_sha256': plan_hash, 'data_receipt_sha256': data_hash})
    model = ChessStudent(mode, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 betas=tuple(PROTOCOL['betas']), eps=PROTOCOL['epsilon'],
                                 weight_decay=PROTOCOL['weight_decay'])
    generator = torch.Generator().manual_seed(PROTOCOL['epoch_order_seed_base']+seed)
    begin, updates, examples = time.perf_counter(), 0, 0
    with (out/'learning.jsonl').open('x') as stream:
        for epoch in range(PROTOCOL['epochs']):
            order = torch.randperm(len(train['targets']), generator=generator)
            for start in range(0, len(order), PROTOCOL['batch_size']):
                indices = order[start:start+PROTOCOL['batch_size']]
                model.train()
                logits, values = model(train['observations'][indices], train['mask'][indices])
                policy_loss = F.cross_entropy(logits, train['targets'][indices])
                value_loss = F.mse_loss(values, train['values'][indices])
                loss = policy_loss+PROTOCOL['value_loss_weight']*value_loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss; no retry or budget replacement')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['gradient_clip'],
                                                          error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                examples += len(indices)
                append(stream, {'epoch': epoch+1, 'update': updates, 'examples_seen': examples,
                                'policy_ce': float(policy_loss.detach()), 'value_mse': float(value_loss.detach()),
                                'loss': float(loss.detach()), 'gradient_norm': float(gradient)})
    training_seconds = time.perf_counter()-begin
    torch.save({'mode': mode, 'seed': seed, 'state_dict': model.state_dict(),
                'vocabulary_sha256': MOVE_VOCAB_SHA256, 'plan_sha256': plan_hash}, out/'weights.pt')
    write_new(out/'evaluation.json', evaluate(model, dev, dev_rows))
    receipt = {'status': 'completed', 'completed_unix': time.time(), 'mode': mode, 'seed': seed,
               'updates': updates, 'examples_seen': examples, 'parameter_count': model.parameter_count(),
               'training_wall_seconds': training_seconds, 'plan_sha256': plan_hash,
               'data_receipt_sha256': data_hash, 'files': {name: sha(out/name) for name in
                                                       ('weights.pt', 'learning.jsonl', 'evaluation.json')}}
    write_new(out/'completed.json', receipt)
    return receipt


def run(plan_path, out):
    plan = verify_plan(plan_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan_hash = sha(plan_path)
    write_new(out/'started.json', {'started_unix': time.time(), 'plan_sha256': plan_hash,
                                 'status': 'started', 'protocol': PROTOCOL})
    try:
        torch.set_num_threads(PROTOCOL['torch_threads'])
        torch.use_deterministic_algorithms(True)
        with chess.engine.SimpleEngine.popen_uci(plan['engine']['path']) as engine:
            if not engine.id.get('name', '').startswith(PROTOCOL['teacher']):
                raise ValueError('Engine identity differs from frozen teacher version')
            engine.configure({'Threads': PROTOCOL['teacher_threads'], 'Hash': PROTOCOL['teacher_hash_mb']})
            write_new(out/'engine.json', {'id': engine.id, **plan['engine']})
            generate_data(out/'data', lambda board: stockfish_label(engine, board, PROTOCOL))
        rows = validate_data(out/'data')
        train, dev = tensors(rows['train']), tensors(rows['dev'])
        jobs = [(mode, seed) for mode in MODES for seed in PROTOCOL['seeds']]
        random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
        receipts = []
        for mode, seed in jobs:
            receipts.append(fit(mode, seed, train, dev, rows['dev'], out/f'{mode}-{seed}',
                                plan_hash, sha(out/'data/completed.json')))
        write_new(out/'completed.json', {'status': 'completed', 'completed_unix': time.time(),
                                        'plan_sha256': plan_hash, 'fits': len(receipts),
                                        'fit_order': jobs, 'total_updates': sum(r['updates'] for r in receipts),
                                        'training_wall_seconds': sum(r['training_wall_seconds'] for r in receipts),
                                        'fit_receipts': {f'{mode}-{seed}': sha(out/f'{mode}-{seed}/completed.json')
                                                         for mode, seed in jobs}})
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'completed_unix': time.time(),
                                     'plan_sha256': plan_hash, 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def report(plan_path, execution, out):
    verify_plan(plan_path)
    execution, out = Path(execution), Path(out)
    complete = json.loads((execution/'completed.json').read_text())
    plan_hash = sha(plan_path)
    if complete['status'] != 'completed' or complete['plan_sha256'] != plan_hash or (execution/'failed.json').exists():
        raise ValueError('Execution lacks a matching successful completion')
    rows = validate_data(execution/'data')
    data_hash = sha(execution/'data/completed.json')
    expected_updates = PROTOCOL['epochs']*math.ceil(PROTOCOL['train_examples']/PROTOCOL['batch_size'])
    summaries = []
    for mode in MODES:
        for seed in PROTOCOL['seeds']:
            name, directory = f'{mode}-{seed}', execution/f'{mode}-{seed}'
            receipt = json.loads((directory/'completed.json').read_text())
            if sha(directory/'completed.json') != complete['fit_receipts'][name]:
                raise ValueError('Fit receipt hash mismatch')
            if (receipt['status'] != 'completed' or receipt['plan_sha256'] != plan_hash or
                    receipt['data_receipt_sha256'] != data_hash or receipt['mode'] != mode or receipt['seed'] != seed
                    or receipt['updates'] != expected_updates or
                    receipt['examples_seen'] != PROTOCOL['epochs']*PROTOCOL['train_examples']):
                raise ValueError('Fit protocol or budget mismatch')
            for filename, digest in receipt['files'].items():
                if sha(directory/filename) != digest:
                    raise ValueError('Checkpoint, log or evaluation hash mismatch')
            checkpoint = ChessStudent.load(directory/'weights.pt', expected_plan_sha256=plan_hash)
            if (checkpoint.mode != mode or checkpoint.seed != seed or
                    receipt['parameter_count'] != checkpoint.parameter_count()):
                raise ValueError('Checkpoint model identity or parameter count mismatch')
            logs = jsonl(directory/'learning.jsonl')
            if len(logs) != expected_updates or [row['update'] for row in logs] != list(range(1, expected_updates+1)):
                raise ValueError('Training update log mismatch')
            result = json.loads((directory/'evaluation.json').read_text())
            if len(result['predictions']) != len(rows['dev']):
                raise ValueError('Development coverage mismatch')
            for prediction, target in zip(result['predictions'], rows['dev'], strict=True):
                if (prediction['id'] != target['id'] or prediction['target'] != target['target_uci'] or
                        prediction['game_id'] != target['game_id'] or
                        prediction['target_value'] != target['target_value'] or
                        prediction['correct'] != (prediction['choice'] == prediction['target']) or
                        chess.Move.from_uci(prediction['choice']) not in chess.Board(target['fen']).legal_moves or
                        not 0 <= prediction['target_probability'] <= 1):
                    raise ValueError('Invalid development prediction or teacher mismatch')
            metrics = summarize_predictions(result['predictions'])
            for key, value in metrics.items():
                if not math.isclose(value, result['metrics'][key], rel_tol=0., abs_tol=1e-12):
                    raise ValueError('Evaluation aggregate mismatch')
            summaries.append({'mode': mode, 'seed': seed, **metrics,
                              'parameter_count': receipt['parameter_count'],
                              'training_wall_seconds': receipt['training_wall_seconds'],
                              'checkpoint_sha256': receipt['files']['weights.pt']})
    if complete['fits'] != 9 or len(complete['fit_receipts']) != 9 or complete['total_updates'] != expected_updates*9:
        raise ValueError('Outer fit count or budget mismatch')
    averages = {mode: {metric: float(np.mean([row[metric] for row in summaries if row['mode'] == mode]))
                      for metric in ('top1_teacher_agreement', 'target_nll', 'value_mae', 'value_mse')}
                for mode in MODES}
    out.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'completed', 'scope': PROTOCOL['claim_scope'], 'novelty_established': False,
               'elo_estimate': None, 'data_role': 'generated development set, not untouched confirmation',
               'fits': summaries, 'mean_by_mode': averages, 'protocol': PROTOCOL,
               'baselines': simple_baselines(rows['train'], rows['dev']),
               'continuation_gate': continuation_gate(summaries),
               'teacher_cost': json.loads((execution/'data/completed.json').read_text()),
               'training_wall_seconds': complete['training_wall_seconds'], 'plan_sha256': plan_hash}
    write_new(out/'summary.json', summary)
    write_new(out/'completed.json', {'status': 'completed', 'completed_unix': time.time(),
                                    'summary_sha256': sha(out/'summary.json'), 'plan_sha256': plan_hash,
                                    'execution_receipt_sha256': sha(execution/'completed.json')})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--out', required=True, type=Path)
    prepare_parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE)
    for name in ('run', 'report'):
        command = sub.add_parser(name)
        command.add_argument('--plan', required=True, type=Path)
        command.add_argument('--out', required=True, type=Path)
        if name == 'report':
            command.add_argument('--execution', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.out, args.engine))
    elif args.command == 'run':
        run(args.plan, args.out)
    else:
        print(json.dumps(report(args.plan, args.execution, args.out)['mean_by_mode'], indent=2))


if __name__ == '__main__':
    main()
