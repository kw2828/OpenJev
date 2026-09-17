"""Preselected 16-game development arena for mate-set supervision controls.

Prepare can precede fine-tuning completion. Run requires all six final fits and
three frozen references. Seed17 and both illustration games are selected before
outcomes. No warmups, retries, fallback moves, checkpoint selection or Elo.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import chess
import chess.pgn
import torch

from openjev.research.chess_arena import greedy_material_policy
from openjev.research.chess_compute import terminal_guard_decision
from openjev.research.chess_spatial import SpatialChess

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('frozen_spatial_arena', ROOT/'scripts/chess_spatial_arena.py')
_ARENA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ARENA)
validate_game = _ARENA.validate_game
execute_schedule = _ARENA.execute_schedule
summarize = _ARENA.summarize

ARMS = ('frozen_predict17', 'single17', 'set17', 'guarded_frozen_predict17')
SEEDS = (17, 29, 43)
SOURCES = ('scripts/chess_mate_arena.py', 'tests/test_chess_mate_arena.py',
           'scripts/chess_spatial_arena.py', 'src/openjev/research/chess_arena.py',
           'src/openjev/research/chess_compute.py', 'src/openjev/research/chess_spatial.py',
           'src/openjev/research/chess_student.py', 'pyproject.toml', 'uv.lock')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def schedule():
    games = []
    for policy in ARMS:
        for opponent in ('greedy_material', 'cnn17'):
            for white, black in ((policy, opponent), (opponent, policy)):
                games.append({'id': f'game-{len(games)+1:02d}', 'white': white, 'black': black})
    return games


def signature(training_plan, execution):
    training_plan = Path(training_plan).resolve()
    spatial_plan = ROOT/'evidence/chess-spatial-v1/plan.json'
    originals = {}
    for name, mode in (('frozen_predict17', 'predict'), ('cnn17', 'cnn')):
        path = ROOT/f'models/chess-spatial-v1/{mode}-17/weights.pt'
        originals[name] = {'path': str(path), 'sha256': sha(path), 'mode': mode, 'seed': 17,
                           'training_plan_path': str(spatial_plan), 'training_plan_sha256': sha(spatial_plan)}
    return {'format': 'openjev-chess-mate-arena-v1',
            'training_plan': {'path': str(training_plan), 'sha256': sha(training_plan)},
            'training_execution': str(Path(execution).resolve()), 'originals': originals,
            'sources': {name: sha(ROOT/name) for name in SOURCES},
            'dependencies': {name: importlib.metadata.version(name) for name in ('chess', 'torch', 'numpy')},
            'python': platform.python_version(), 'platform': platform.platform(),
            'protocol': {'initial_fen': chess.STARTING_FEN, 'clock_seconds': 300., 'max_plies': 160,
                         'claim_draw': False, 'device': 'cpu', 'torch_threads': 2, 'model_seed': 17,
                         'depth': 4, 'games': schedule(), 'illustrative_game': 'game-01',
                         'additional_illustration': 'game-09',
                         'illustration_selection': 'game-01 is the first frozen-reference game; game-09 is the first set-model game. Both fixed before outcomes, every accepted move.',
                         'illustration_caption': 'Predetermined development game, not a best game or strength estimate. Learned policies use depth4; the guarded reference explicitly checks native mate-in-one.',
                         'timing': 'CPU policy call wall time includes board/candidate processing and any explicit mate scan; model load excluded; no warmups',
                         'scope': 'Small deterministic development arena, not Elo, an independent strength confirmation or a model ranking',
                         'warmups': 0, 'retries': 0, 'fallback': None}}


def prepare(out, training_plan=None, execution=None):
    training_plan = training_plan or ROOT/'evidence/chess-mate-v1/plan.json'
    execution = execution or ROOT/'runs/chess-mate-v1/execution'
    value = signature(training_plan, execution)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', value)
    write_new(out/'prepared.json', {'status': 'prepared', 'created_unix': time.time(),
                                   'plan_sha256': sha(out/'plan.json'), 'arena_started': False})
    return out/'plan.json'


def verify_plan(path):
    plan = read(path)
    if plan != signature(plan['training_plan']['path'], plan['training_execution']):
        raise ValueError('Arena source, dependency, schedule, training plan or original binding changed')
    return plan


def verify_training(plan):
    """Verify all nine receipts and six new checkpoints, not just selected seed17."""
    training_path = Path(plan['training_plan']['path'])
    plan_hash = sha(training_path)
    if plan_hash != plan['training_plan']['sha256']:
        raise ValueError('Mate training plan changed')
    frozen = read(training_path)
    protocol = frozen['protocol']
    if (tuple(protocol['seeds']) != SEEDS or tuple(protocol['arms']) != ('single', 'set')
            or protocol['base_mode'] != 'predict' or protocol['depth'] != 4 or protocol['width'] != 32):
        raise ValueError('Training does not contain the fixed six fits and three references')
    if sha(training_path.parent/'selection.json') != frozen['selection_sha256']:
        raise ValueError('Mate training selection changed')
    for name, digest in frozen['inputs'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen mate training input changed')
    references = {}
    for seed in SEEDS:
        name = f'models/chess-spatial-v1/predict-{seed}/weights.pt'
        if name not in frozen['inputs']:
            raise ValueError('Mate plan lacks a required original reference')
        references[f'frozen-{seed}'] = {'path': str(ROOT/name), 'sha256': frozen['inputs'][name]}
    if references['frozen-17']['sha256'] != plan['originals']['frozen_predict17']['sha256']:
        raise ValueError('Arena reference differs from the mate-study initializer')
    execution = Path(plan['training_execution'])
    complete = read(execution/'completed.json')
    expected = {f'{arm}-{seed}' for arm in ('frozen', 'single', 'set') for seed in SEEDS}
    if (complete['status'] != 'completed' or (execution/'failed.json').exists()
            or complete['plan_sha256'] != plan_hash or set(complete['fits']) != expected
            or complete['schedule_sha256'] != sha(execution/'schedule.json')):
        raise ValueError('Arena requires the complete successful six-fit/three-reference execution')
    receipts, checkpoints = {}, {}
    for name in sorted(expected):
        directory = execution/name
        receipt = read(directory/'completed.json')
        arm, seed = name.split('-')
        required = {f'{split}.jsonl' for split in ('mate_dev', 'mate_confirm', 'dev', 'shift')}
        if arm != 'frozen':
            required |= {'weights.pt', 'learning.jsonl'}
        if (sha(directory/'completed.json') != complete['fits'][name]
                or receipt['status'] != 'completed' or (directory/'failed.json').exists()
                or receipt['arm'] != arm or receipt['seed'] != int(seed)
                or receipt['plan_sha256'] != plan_hash or set(receipt['files']) != required
                or receipt['updates'] != (0 if arm == 'frozen' else protocol['updates_per_fit'])):
            raise ValueError('Mate fit/reference receipt identity or completeness mismatch')
        for filename, digest in receipt['files'].items():
            if sha(directory/filename) != digest:
                raise ValueError('Mate fit/reference checkpoint or evidence changed')
        receipts[name] = sha(directory/'completed.json')
        if arm != 'frozen':
            checkpoints[name] = {'path': str(directory/'weights.pt'), 'sha256': sha(directory/'weights.pt')}
    return {'status': 'verified', 'training_plan_sha256': plan_hash,
            'training_completed_sha256': sha(execution/'completed.json'),
            'training_schedule_sha256': sha(execution/'schedule.json'),
            'fit_receipts': receipts, 'checkpoints': checkpoints, 'references': references,
            'selected_checkpoints': {name: checkpoints[name] for name in ('single-17', 'set-17')}}


def _load(artifact, plan_hash, mode, seed=17):
    if sha(artifact['path']) != artifact['sha256']:
        raise ValueError('Checkpoint changed before arena loading')
    model = SpatialChess.load(artifact['path'], expected_plan_sha256=plan_hash)
    if (model.mode != mode or model.seed != seed or model.width != 32 or model.depth != 4
            or next(model.parameters()).device.type != 'cpu'):
        raise ValueError('Loaded model does not match its frozen identity')
    return model


def load_policies(plan, bindings):
    policies, loads = {}, []
    for name, artifact in plan['originals'].items():
        started = time.perf_counter()
        model = _load(artifact, artifact['training_plan_sha256'], artifact['mode'])
        policies[name] = model.choose
        if name == 'frozen_predict17':
            policies['guarded_frozen_predict17'] = functools.partial(terminal_guard_decision, model, depth=4)
        loads.append({'policy': name, 'wall_seconds': time.perf_counter()-started,
                      'path': artifact['path'], 'sha256': artifact['sha256'],
                      'shared_with': ['guarded_frozen_predict17'] if name == 'frozen_predict17' else []})
    for arm in ('single', 'set'):
        name = f'{arm}17'
        artifact = bindings['selected_checkpoints'][f'{arm}-17']
        started = time.perf_counter()
        model = _load(artifact, bindings['training_plan_sha256'], 'predict')
        policies[name] = model.choose
        loads.append({'policy': name, 'wall_seconds': time.perf_counter()-started, **artifact, 'shared_with': []})
    policies['greedy_material'] = greedy_material_policy()
    return policies, loads


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
        receipt = {'status': 'completed', 'execution_complete': True, 'plan_sha256': sha(plan_path),
                   'completed_unix': time.time(), 'gameplay_wall_seconds': time.perf_counter()-started,
                   'scheduled_games': len(plan['protocol']['games']), 'game_status_counts': summary['status_counts'],
                   'illustrative_game': plan['protocol']['illustrative_game'],
                   'additional_illustration': plan['protocol']['additional_illustration'],
                   'files': {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
                   'scope': 'Execution completion is separate from scored outcomes; all games retained, no retries or Elo'}
        write_new(out/'completed.json', receipt)
        return receipt
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'plan_sha256': sha(plan_path),
                                     'error_type': type(exc).__name__, 'error': str(exc),
                                     'completed_unix': time.time()})
        raise


def report(plan_path, execution, out):
    """Audit completed arena without replaying a policy or choosing any games."""
    plan, execution = verify_plan(plan_path), Path(execution)
    bindings, complete = verify_training(plan), read(execution/'completed.json')
    protocol = plan['protocol']
    required = {'started.json', 'training-bindings.json', 'loads.json', 'summary.json'}
    required |= {f"{game['id']}.{suffix}" for game in protocol['games'] for suffix in ('json', 'pgn')}
    if (complete['status'] != 'completed' or complete['execution_complete'] is not True
            or (execution/'failed.json').exists() or complete['plan_sha256'] != sha(plan_path)
            or complete['scheduled_games'] != len(protocol['games']) or set(complete['files']) != required
            or complete['illustrative_game'] != protocol['illustrative_game']
            or complete['additional_illustration'] != protocol['additional_illustration']
            or not math.isfinite(complete['gameplay_wall_seconds']) or complete['gameplay_wall_seconds'] < 0):
        raise ValueError('Incomplete or inconsistent arena execution receipt')
    for name, digest in complete['files'].items():
        if sha(execution/name) != digest:
            raise ValueError('Arena evidence hash changed')
    if (read(execution/'started.json')['plan_sha256'] != sha(plan_path)
            or read(execution/'training-bindings.json') != bindings):
        raise ValueError('Arena training bindings changed')
    loads = read(execution/'loads.json')
    expected_loads = {name: artifact for name, artifact in plan['originals'].items()}
    expected_loads.update({f'{arm}17': bindings['selected_checkpoints'][f'{arm}-17'] for arm in ('single', 'set')})
    if len(loads) != len(expected_loads) or {row['policy'] for row in loads} != set(expected_loads):
        raise ValueError('Arena model-load coverage mismatch')
    for row in loads:
        artifact = expected_loads[row['policy']]
        if (row['path'] != artifact['path'] or row['sha256'] != artifact['sha256']
                or not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0
                or row['shared_with'] != (['guarded_frozen_predict17'] if row['policy'] == 'frozen_predict17' else [])):
            raise ValueError('Arena model-load provenance mismatch')
    games = []
    for spec in protocol['games']:
        game = validate_game(read(execution/f"{spec['id']}.json"), spec, protocol)
        with (execution/f"{spec['id']}.pgn").open() as stream:
            parsed = chess.pgn.read_game(stream)
            extra = chess.pgn.read_game(stream)
        if (parsed is None or parsed.errors or extra is not None
                or parsed.headers['White'] != spec['white'] or parsed.headers['Black'] != spec['black']
                or parsed.headers['Result'] != game['result']
                or parsed.end().board().fen() != game['final_fen']
                or [move.uci() for move in parsed.mainline_moves()] != [r['move_uci'] for r in game['moves']]):
            raise ValueError('Arena PGN does not match its complete move trace')
        games.append(game)
    summary = summarize(games)
    if summary != read(execution/'summary.json') or summary['status_counts'] != complete['game_status_counts']:
        raise ValueError('Arena outcome aggregate mismatch')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'completed', 'plan_sha256': sha(plan_path), **summary,
              'illustrative_game': protocol['illustrative_game'],
              'additional_illustration': protocol['additional_illustration'],
              'novelty_established': False, 'elo_estimate': None}
    write_new(out/'summary.json', result)
    write_new(out/'completed.json', {'status': 'completed', 'summary_sha256': sha(out/'summary.json'),
                                    'plan_sha256': sha(plan_path),
                                    'execution_receipt_sha256': sha(execution/'completed.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--out', type=Path, default=ROOT/'evidence/chess-mate-arena-v1')
    prep.add_argument('--training-plan', type=Path, default=ROOT/'evidence/chess-mate-v1/plan.json')
    prep.add_argument('--execution', type=Path, default=ROOT/'runs/chess-mate-v1/execution')
    for name in ('run', 'report'):
        command = commands.add_parser(name)
        command.add_argument('--plan', type=Path, default=ROOT/'evidence/chess-mate-arena-v1/plan.json')
        command.add_argument('--out', type=Path, required=True)
        if name == 'report':
            command.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.out, args.training_plan, args.execution))
    elif args.command == 'run':
        print(json.dumps(run(args.plan, args.out), indent=2))
    else:
        print(json.dumps(report(args.plan, args.execution, args.out), indent=2))


if __name__ == '__main__':
    main()
