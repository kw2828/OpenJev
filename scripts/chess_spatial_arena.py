"""Preselected 16-game development arena for every spatial chess architecture.

Prepare is independent of training completion. Run requires all twelve final
fits, preserves every scheduled game and attempt, and never retries or substitutes
a policy. This small deterministic panel does not estimate Elo or general strength.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import time
from pathlib import Path

import chess
import chess.pgn
import torch

from openjev.research.chess_arena import greedy_material_policy, play_game
from openjev.research.chess_spatial import SpatialChess
from openjev.research.chess_student import ChessStudent

ROOT = Path(__file__).resolve().parents[1]
MODES = ('predict', 'reconstruct', 'recurrent', 'cnn')
SEEDS = (17, 29, 43)
SOURCES = ('scripts/chess_spatial_arena.py', 'tests/test_chess_spatial_arena.py',
           'src/openjev/research/chess_spatial.py', 'src/openjev/research/chess_student.py',
           'src/openjev/research/chess_arena.py', 'pyproject.toml', 'uv.lock')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def schedule():
    games = []
    for mode in MODES:
        spatial = f'{mode}-17'
        for opponent in ('greedy_material', 'gru_v1'):
            for white, black in ((spatial, opponent), (opponent, spatial)):
                games.append({'id': f'game-{len(games)+1:02d}', 'white': white, 'black': black})
    return games


def signature(training_plan, execution):
    training_plan = Path(training_plan).resolve()
    old_plan = ROOT/'evidence/chess-student-v1/plan.json'
    old_weights = ROOT/'models/chess-student-v1/gru-17/weights.pt'
    return {
        'format': 'openjev-chess-spatial-arena-v1',
        'training_plan': {'path': str(training_plan), 'sha256': sha(training_plan)},
        'training_execution': str(Path(execution).resolve()),
        'old_gru': {'path': str(old_weights), 'sha256': sha(old_weights), 'mode': 'gru', 'seed': 17,
                    'training_plan_path': str(old_plan), 'training_plan_sha256': sha(old_plan)},
        'sources': {name: sha(ROOT/name) for name in SOURCES},
        'dependencies': {name: importlib.metadata.version(name) for name in ('chess', 'torch')},
        'python': platform.python_version(),
        'protocol': {'initial_fen': chess.STARTING_FEN, 'clock_seconds': 300., 'max_plies': 160,
                     'claim_draw': False, 'device': 'cpu', 'torch_threads': 2, 'spatial_seed': 17,
                     'spatial_depth': 4, 'games': schedule(), 'illustrative_game': 'game-01',
                     'illustration_selection': 'First scheduled game, before outcomes; every accepted move',
                     'timing': 'Policy call wall time on CPU, including candidate encoding; model load excluded; no warmups',
                     'scope': 'Deterministic development games, not Elo, independent confirmation or a strength ranking',
                     'retries': 0, 'fallback': None},
    }


def prepare(out, training_plan=None, execution=None):
    training_plan = training_plan or ROOT/'evidence/chess-spatial-v1/plan.json'
    execution = execution or ROOT/'runs/chess-spatial-v1/execution'
    value = signature(training_plan, execution)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', value)
    write_new(out/'prepared.json', {'status': 'prepared', 'created_unix': time.time(),
                                   'plan_sha256': sha(out/'plan.json'), 'arena_started': False})
    return out/'plan.json'


def verify_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan != signature(plan['training_plan']['path'], plan['training_execution']):
        raise ValueError('Arena plan, source, dependency or old checkpoint binding changed')
    return plan


def verify_training(plan):
    """Bind all final fit receipts and weights before any gameplay or model load."""
    training_plan = Path(plan['training_plan']['path'])
    if sha(training_plan) != plan['training_plan']['sha256']:
        raise ValueError('Spatial training plan hash changed')
    frozen = json.loads(training_plan.read_text())
    protocol = frozen['protocol']
    if set(protocol['modes']) != set(MODES) or set(protocol['seeds']) != set(SEEDS):
        raise ValueError('Spatial training does not contain the required twelve fixed fits')
    for name, digest in frozen['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen spatial training source changed')
    for key, name in (('lock_sha256', 'uv.lock'), ('pyproject_sha256', 'pyproject.toml')):
        if key in frozen and sha(ROOT/name) != frozen[key]:
            raise ValueError('Frozen spatial training dependency file changed')
    execution = Path(plan['training_execution'])
    completed = json.loads((execution/'completed.json').read_text())
    expected = {f'{mode}-{seed}' for mode in MODES for seed in SEEDS}
    if ((execution/'failed.json').exists() or completed['status'] != 'completed' or
            completed['fits'] != 12 or set(completed['fit_receipts']) != expected or
            completed['plan_sha256'] != sha(training_plan)):
        raise ValueError('Arena requires the complete successful twelve-fit training execution')
    data_hash = sha(execution/'data/completed.json')
    if (json.loads((execution/'data/completed.json').read_text())['status'] != 'completed' or
            (execution/'data/failed.json').exists()):
        raise ValueError('Spatial training data did not complete')
    checkpoints, receipts = {}, {}
    for mode in MODES:
        for seed in SEEDS:
            name = f'{mode}-{seed}'
            fit = execution/name
            if sha(fit/'completed.json') != completed['fit_receipts'][name]:
                raise ValueError('Spatial fit receipt differs from the bound training execution')
            receipt = json.loads((fit/'completed.json').read_text())
            if (receipt['status'] != 'completed' or (fit/'failed.json').exists() or
                    receipt['mode'] != mode or receipt['seed'] != seed or
                    receipt['plan_sha256'] != sha(training_plan) or receipt['data_receipt_sha256'] != data_hash):
                raise ValueError('Spatial fit provenance or completion is invalid')
            if set(receipt['files']) != {'weights.pt', 'learning.jsonl', 'dev.json', 'shift.json'}:
                raise ValueError('Spatial fit receipt does not bind all required outputs')
            for filename, digest in receipt['files'].items():
                if sha(fit/filename) != digest:
                    raise ValueError('Spatial fit checkpoint or evidence changed')
            checkpoints[name] = {'path': str(fit/'weights.pt'), 'sha256': sha(fit/'weights.pt')}
            receipts[name] = sha(fit/'completed.json')
    return {'status': 'verified', 'training_plan_sha256': sha(training_plan),
            'training_completed_sha256': sha(execution/'completed.json'),
            'data_receipt_sha256': data_hash, 'fit_receipts': receipts, 'checkpoints': checkpoints,
            'selected_checkpoints': {f'{mode}-17': checkpoints[f'{mode}-17'] for mode in MODES}}


def load_policies(plan, bindings):
    policies, loads = {}, []
    for mode in MODES:
        name = f'{mode}-17'
        artifact = bindings['selected_checkpoints'][name]
        started = time.perf_counter()
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Selected spatial checkpoint changed before loading')
        model = SpatialChess.load(artifact['path'], expected_plan_sha256=bindings['training_plan_sha256'])
        if (model.mode != mode or model.seed != 17 or model.depth != 4 or model.width != 32 or
                next(model.parameters()).device.type != 'cpu'):
            raise ValueError('Loaded spatial model does not match its selected identity')
        policies[name] = model.choose
        loads.append({'policy': name, 'wall_seconds': time.perf_counter()-started, **artifact})
    old = plan['old_gru']
    started = time.perf_counter()
    if sha(old['path']) != old['sha256']:
        raise ValueError('Old GRU checkpoint changed before loading')
    model = ChessStudent.load(old['path'], expected_plan_sha256=old['training_plan_sha256'])
    if model.mode != 'gru' or model.seed != 17 or next(model.parameters()).device.type != 'cpu':
        raise ValueError('Loaded old GRU has an incorrect identity')
    policies['gru_v1'] = model.choose
    loads.append({'policy': 'gru_v1', 'wall_seconds': time.perf_counter()-started,
                  'path': old['path'], 'sha256': old['sha256']})
    policies['greedy_material'] = greedy_material_policy()
    return policies, loads


def validate_game(game, spec, protocol):
    """Replay every attempt, clock debit and accepted move; preserve non-scored states."""
    if any(game[key] != spec[key] for key in ('white', 'black')) or game['game_id'] != spec['id']:
        raise ValueError('Game does not match its scheduled players or ID')
    if (game['initial_fen'] != protocol['initial_fen'] or game['claim_draw'] != protocol['claim_draw'] or
            game['max_plies'] != protocol['max_plies']):
        raise ValueError('Game rules differ from the frozen protocol')
    board = chess.Board(game['initial_fen'])
    clocks = {side: float(protocol['clock_seconds']) for side in ('white', 'black')}
    if game['initial_clocks'] != clocks:
        raise ValueError('Incorrect initial game clocks')
    played = []
    for index, attempt in enumerate(game['attempts']):
        side = 'white' if board.turn else 'black'
        if (attempt['fen_before'] != board.fen() or attempt['side'] != side or
                attempt['ply'] != len(played)+1 or attempt['clocks_before'] != clocks):
            raise ValueError('Attempt does not match the authoritative position and clocks')
        elapsed = attempt['latency_ms']/1000
        if not math.isfinite(elapsed) or elapsed < 0 or attempt['wall_ms'] != attempt['latency_ms']:
            raise ValueError('Invalid attempt latency')
        clocks[side] = max(0., clocks[side]-elapsed)
        if any(not math.isclose(attempt['clocks'][color], clocks[color], abs_tol=1e-9, rel_tol=0.)
               for color in clocks):
            raise ValueError('Attempt clock debit is inconsistent')
        # Retain the original binary clock value for exact next-attempt linkage.
        clocks = dict(attempt['clocks'])
        if attempt['played']:
            move = chess.Move.from_uci(attempt['move_uci'])
            if (move not in board.legal_moves or attempt['san'] != board.san(move) or
                    attempt['disposition'] != 'played' or clocks[side] <= 0):
                raise ValueError('Invalid accepted move')
            board.push(move)
            played.append(attempt)
        elif index != len(game['attempts'])-1:
            raise ValueError('Play continued after a discarded policy attempt')
        if attempt['fen_after'] != board.fen():
            raise ValueError('Attempt successor is inconsistent')
    if played != game['moves'] or game['final_fen'] != board.fen() or game['clocks'] != clocks:
        raise ValueError('Accepted moves or final game state disagree with the trace')
    outcome = board.outcome(claim_draw=protocol['claim_draw'])
    if game['status'] == 'completed':
        if outcome is not None:
            if game['result'] != outcome.result() or game['termination'] != outcome.termination.name.lower():
                raise ValueError('Completed game outcome is inconsistent')
        else:
            last = game['attempts'][-1]
            expected = ('1/2-1/2' if board.has_insufficient_material(not board.turn) else
                        '0-1' if board.turn else '1-0')
            termination = 'timeout_insufficient_material' if expected == '1/2-1/2' else 'timeout'
            if (last['played'] or last['disposition'] != 'discarded_timeout' or
                    clocks[last['side']] != 0 or game['result'] != expected or game['termination'] != termination):
                raise ValueError('Completed game lacks a legal outcome or recorded flag fall')
    elif game['status'] == 'unfinished':
        if (game['result'] != '*' or game['termination'] != 'max_plies' or outcome is not None or
                len(played) != protocol['max_plies'] or len(game['attempts']) != len(played)):
            raise ValueError('Ply-cap game is not an unscored unfinished game')
    elif game['status'] == 'failed':
        if (game['result'] != '*' or not game['attempts'] or game['attempts'][-1]['played'] or
                game['termination'] not in {'policy_error', 'invalid_response', 'invalid_choice'} or
                game['attempts'][-1]['disposition'] != game['termination']):
            raise ValueError('Policy failure is not retained as an unscored failed game')
    else:
        raise ValueError('Unknown game terminal status')
    return game


def summarize(games):
    counts = {status: sum(game['status'] == status for game in games)
              for status in ('completed', 'unfinished', 'failed')}
    policies = {}
    for game in games:
        for side in ('white', 'black'):
            row = policies.setdefault(game[side], dict.fromkeys(('wins', 'draws', 'losses',
                                                                 'unfinished', 'failed'), 0))
            if game['status'] != 'completed':
                row[game['status']] += 1
            elif game['result'] == '1/2-1/2':
                row['draws'] += 1
            elif game['result'] == ('1-0' if side == 'white' else '0-1'):
                row['wins'] += 1
            else:
                row['losses'] += 1
    return {'games': len(games), 'status_counts': counts, 'policies': policies,
            'game_results': [{key: game[key] for key in ('game_id', 'white', 'black', 'status',
                                                        'result', 'termination')}
                             | {'played_plies': len(game['moves'])} for game in games],
            'scope': 'All scheduled deterministic development games; unfinished/failed games are unscored; no Elo'}


def execute_schedule(plan, policies, out):
    """Execute every scheduled game once, including after a recorded policy failure."""
    protocol, games = plan['protocol'], []
    for spec in protocol['games']:
        result = play_game(policies[spec['white']], policies[spec['black']],
                           white_name=spec['white'], black_name=spec['black'],
                           initial_fen=protocol['initial_fen'], clock_seconds=protocol['clock_seconds'],
                           max_plies=protocol['max_plies'], claim_draw=protocol['claim_draw'])
        game = {'game_id': spec['id'], **result.to_dict()}
        write_new(out/f'{spec["id"]}.json', game)
        pgn = result.to_pgn()
        with (out/f'{spec["id"]}.pgn').open('x') as stream:
            stream.write(pgn+'\n')
        validate_game(game, spec, protocol)
        parsed = chess.pgn.read_game(io.StringIO(pgn))
        if (parsed is None or parsed.errors or parsed.end().board().fen() != game['final_fen'] or
                parsed.headers['Result'] != game['result']):
            raise ValueError('Saved PGN does not reproduce the complete game')
        games.append(game)
        print(json.dumps({'game': spec['id'], 'status': game['status'], 'result': game['result'],
                          'plies': len(game['moves']), 'termination': game['termination']}), flush=True)
    summary = summarize(games)
    write_new(out/'summary.json', summary)
    return summary


def run(plan_path, out):
    plan = verify_plan(plan_path)
    bindings = verify_training(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'started_unix': time.time(), 'plan_sha256': sha(plan_path)})
    try:
        write_new(out/'training-bindings.json', bindings)
        torch.set_num_threads(plan['protocol']['torch_threads'])
        policies, loads = load_policies(plan, bindings)
        write_new(out/'loads.json', loads)
        started = time.perf_counter()
        summary = execute_schedule(plan, policies, out)
        receipt = {'status': 'completed', 'execution_complete': True,
                   'plan_sha256': sha(plan_path), 'completed_unix': time.time(),
                   'gameplay_wall_seconds': time.perf_counter()-started,
                   'scheduled_games': len(plan['protocol']['games']),
                   'game_status_counts': summary['status_counts'],
                   'illustrative_game': plan['protocol']['illustrative_game'],
                   'files': {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
                   'scope': 'Execution completion is separate from scored game status; no retries or Elo'}
        write_new(out/'completed.json', receipt)
        return receipt
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'plan_sha256': sha(plan_path),
                                     'error_type': type(exc).__name__, 'error': str(exc),
                                     'completed_unix': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--out', type=Path, default=ROOT/'evidence/chess-spatial-arena-v1')
    prepare_parser.add_argument('--training-plan', type=Path, default=ROOT/'evidence/chess-spatial-v1/plan.json')
    prepare_parser.add_argument('--execution', type=Path, default=ROOT/'runs/chess-spatial-v1/execution')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--plan', type=Path, default=ROOT/'evidence/chess-spatial-arena-v1/plan.json')
    run_parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = (prepare(args.out, args.training_plan, args.execution) if args.command == 'prepare' else
              run(args.plan, args.out))
    print(str(result) if isinstance(result, Path) else json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
