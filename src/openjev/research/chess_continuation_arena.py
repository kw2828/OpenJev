"""Fixed paired arena for continuation-supervised actors, with native-history audit.

The unchanged capacity arena supplies play, history/clock replay and PGN kernels.
Every player truthfully remains width32; this module adds objective identities
without pretending objectives are capacities. Immediate-mate diagnostics run
only after each game and cannot affect actor choices or charged decision time.
"""

import io
import math
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from openjev.research import chess_capacity_arena as native
from openjev.research.chess_continuation_model import ContinuationChess

OBJECTIVES = ('policy', 'best_value', 'continuation')
OPPONENTS = ('policy', 'best_value')
SEEDS = (193, 211, 227)
OPENINGS = native.OPENINGS
WIDTH, DEPTH = 32, 4
BOOTSTRAP_SEED = 28100001
BOOTSTRAP_REPLICATES = 2000
_hash, _read, _write, _plan_hash = native._hash, native._read, native._write, native._plan_hash
to_pgn = native.to_pgn


def model_name(objective, seed):
    return f'{objective}-{seed}'


def schedule():
    games = []
    for opponent in OPPONENTS:
        for opening in native._openings(OPENINGS):
            for seed in SEEDS:
                for white, black in (('continuation', opponent), (opponent, 'continuation')):
                    games.append({'id': f'game-{len(games)+1:03d}', 'opponent': opponent,
                                  'opening_id': opening['id'], 'opening_name': opening['name'],
                                  'opening_moves': opening['moves'], 'seed': seed,
                                  'white': model_name(white, seed), 'black': model_name(black, seed),
                                  'white_objective': white, 'black_objective': black,
                                  'white_width': WIDTH, 'black_width': WIDTH})
    return games


def protocol():
    rules = native.protocol(OPENINGS)
    return {**rules, 'games': schedule(), 'width': WIDTH, 'depth': DEPTH,
            'objectives': list(OBJECTIVES), 'opponents': list(OPPONENTS), 'seeds': list(SEEDS),
            'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
            'zero_failed_games_scope': 'All 192 scheduled games',
            'inference_scope': 'Unchanged actor only; no candidate critic, search, guard, retry or fallback',
            'mate_diagnostic_scope': 'Post-game native checks on accepted actor turns only; no policy feedback',
            'source_dependencies': ['src/openjev/research/chess_capacity_arena.py',
                                    'src/openjev/research/chess_arena.py']}


def _rules(arena_plan):
    rules = protocol()
    if arena_plan is not None and arena_plan != rules:
        raise ValueError('Frozen continuation arena protocol differs from the fixed protocol')
    return rules


def _spec_identity(spec):
    opponent, seed = spec['opponent'], spec['seed']
    if (opponent not in OPPONENTS or type(seed) is not int or seed not in SEEDS
            or {spec['white_objective'], spec['black_objective']} != {'continuation', opponent}):
        raise ValueError('Invalid continuation pairing')
    for side in ('white', 'black'):
        if (spec[f'{side}_width'] != WIDTH or type(spec[f'{side}_width']) is not int
                or spec[side] != model_name(spec[f'{side}_objective'], seed)):
            raise ValueError('Objective/player identity mismatch')


def _objective_response(response, objective):
    if not isinstance(response, Mapping) or response.get('objective') != objective:
        raise ValueError('Actor response objective does not match the scheduled player')
    return response


def _diagnostic(game):
    board = chess.Board(game['initial_fen'])
    totals = {name: {'accepted_turns': 0, 'mate_opportunities': 0, 'mate_taken': 0, 'mate_missed': 0}
              for name in (game['white'], game['black'])}
    opportunities = []
    for row in game['moves']:
        move = chess.Move.from_uci(row['move_uci'])
        if row['ply'] > 6:
            player = game['white'] if board.turn else game['black']
            totals[player]['accepted_turns'] += 1
            mates = []
            for candidate in list(board.legal_moves):
                board.push(candidate)
                if board.is_checkmate():
                    mates.append(candidate.uci())
                board.pop()
            if mates:
                hit = row['move_uci'] in mates
                totals[player]['mate_opportunities'] += 1
                totals[player]['mate_taken' if hit else 'mate_missed'] += 1
                opportunities.append({'ply': row['ply'], 'player': player, 'fen': board.fen(),
                                      'mating_moves': sorted(mates), 'selected': row['move_uci'],
                                      'taken': hit})
        board.push(move)
    return {'by_player': totals, 'opportunities': opportunities,
            'scope': 'Accepted actor turns only, after six opening plies; dependent repeated turns, not puzzles'}


def play(spec, white, black, *, timer=None):
    _spec_identity(spec)

    def checked(policy, objective):
        def choose(board):
            return _objective_response(policy(board), objective)
        return choose

    game = native.play(spec, checked(white, spec['white_objective']),
                       checked(black, spec['black_objective']), timer=timer)
    game.update({key: spec[key] for key in ('white_objective', 'black_objective', 'opponent')})
    game['mate_diagnostic'] = _diagnostic(game)
    return game


def validate_game(game, spec):
    _spec_identity(spec)
    for key in ('white_objective', 'black_objective', 'opponent'):
        if game.get(key) != spec[key]:
            raise ValueError('Saved game objective identity mismatch')
    native.validate_game(game, spec)
    for attempt in game['attempts']:
        if attempt['policy'] is not None:
            _objective_response(attempt['policy'], spec[f"{attempt['side']}_objective"])
    if game.get('mate_diagnostic') != _diagnostic(game):
        raise ValueError('Immediate-mate diagnostic disagrees with native replay')
    return game


def _points(games):
    counts = Counter(g['status'] for g in games)
    clusters = {o['id']: [] for o in OPENINGS}
    wins = draws = losses = 0
    for game in games:
        if game['status'] != 'completed':
            pair = (0., 1.)
        elif game['result'] == '1/2-1/2':
            pair = (.5, .5)
            draws += 1
        elif game['result'] == ('1-0' if game['white_objective'] == 'continuation' else '0-1'):
            pair = (1., 1.)
            wins += 1
        else:
            pair = (0., 0.)
            losses += 1
        clusters[game['opening_id']].append(pair)
    points = wins + .5 * draws
    unresolved = counts['unfinished'] + counts['failed']
    values = np.array([np.sum(group, axis=0) for group in clusters.values()])
    sample = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    bootstrap = values[sample].sum(axis=1) / len(games)
    return {'games': len(games),
            'status_counts': {key: counts[key] for key in ('completed', 'unfinished', 'failed')},
            'continuation': {'wins': wins, 'draws': draws, 'losses': losses,
                             'completed_points': points, 'possible_points': len(games),
                             'score_lower_bound': points / len(games),
                             'score_upper_bound': (points + unresolved) / len(games)},
            'opening_cluster_bootstrap': {
                'openings': len(values), 'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
                'lower_bound_interval': np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
                'upper_bound_interval': np.quantile(bootstrap[:, 1], [.025, .975]).tolist(),
                'scope': 'Whole openings with both colors and all three fitted seeds; conditional on those fits. '
                         'Intervals describe unresolved score bounds, not imputed draws.'}}


def summarize(games, arena_plan=None):
    rules = _rules(arena_plan)
    specs = rules['games']
    if len(games) != len(specs) or [g['game_id'] for g in games] != [s['id'] for s in specs]:
        raise ValueError('Summary requires the full ordered 192-game schedule')
    for game, spec in zip(games, specs, strict=True):
        if (any(game[key] != spec[key] for key in
                ('white', 'black', 'seed', 'white_width', 'black_width',
                 'white_objective', 'black_objective', 'opponent', 'opening_id'))
                or game['status'] not in ('completed', 'unfinished', 'failed')
                or game['result'] not in
                ({'1-0', '0-1', '1/2-1/2'} if game['status'] == 'completed' else {'*'})):
            raise ValueError('Incorrect summary identity or terminal status')
    aggregate = _points(games)
    panels = {opponent: _points([g for g in games if g['opponent'] == opponent])
              for opponent in OPPONENTS}
    checks = [{'opponent': name, 'threshold': rules['gate_lower_score'],
               'lower_score_bound': panel['continuation']['score_lower_bound'],
               'passed': panel['continuation']['score_lower_bound'] >= rules['gate_lower_score']}
              for name, panel in panels.items()]
    zero_failed = aggregate['status_counts']['failed'] == 0
    passed = zero_failed and all(check['passed'] for check in checks)
    diagnostic = {}
    for game in games:
        for name, counts in game['mate_diagnostic']['by_player'].items():
            previous = diagnostic.setdefault(name, {key: 0 for key in counts})
            for key, value in counts.items():
                previous[key] += value
    return {'status': 'completed', 'games': len(games), 'by_opponent': panels, 'aggregate': aggregate,
            'status_counts': aggregate['status_counts'],
            'gate': {'checks': checks, 'zero_failed_games': zero_failed, 'passed': passed},
            'continuation_gate': {'checks': checks, 'zero_failed_games': zero_failed, 'passed': passed},
            'gate_passed': passed, 'policy_calls': sum(len(g['attempts']) for g in games),
            'played_plies': sum(g['played_plies'] for g in games),
            'policy_wall_seconds': math.fsum(g['wall_seconds'] for g in games),
            'mate_diagnostic_by_player': diagnostic,
            'mate_diagnostic_scope': 'Post-game native checks; accepted turns, not independent positions',
            'game_results': [{key: g[key] for key in
                              ('game_id', 'opening_id', 'opponent', 'seed', 'white', 'black', 'status',
                               'result', 'termination', 'played_plies')} for g in games],
            'elo_estimate': None, 'scope': rules['scope']}


def _model_bindings(models):
    import hashlib

    expected = {model_name(o, s): (o, s) for o in OBJECTIVES for s in SEEDS}
    if set(models) != set(expected):
        raise ValueError('Expected exactly nine named objective/seed models')
    bindings = {}
    for name, (objective, seed) in expected.items():
        model = models[name]
        if (not isinstance(model, ContinuationChess) or model.objective != objective
                or type(model.seed) is not int or model.seed != seed or model.width != WIDTH
                or model.depth != DEPTH or model.recurrence != 'residual' or model.training
                or any(value.device.type != 'cpu' or not torch.isfinite(value).all()
                       for value in model.state_dict().values())):
            raise ValueError('Model objective, seed, width, depth, eval mode or CPU state mismatch')
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            digest.update(key.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        bindings[name] = {'objective': objective, 'seed': seed, 'width': WIDTH, 'depth': DEPTH,
                          'recurrence': 'residual', 'state_sha256': digest.hexdigest()}
    return bindings


def _actor(model):
    def choose(board):
        response = model.choose(board, depth=DEPTH)
        if not isinstance(response, Mapping):
            raise TypeError('Actor did not return a response mapping')
        if 'objective' in response and response['objective'] != model.objective:
            raise ValueError('Actor returned a contradictory objective')
        return {**response, 'objective': model.objective}
    return choose


def run(models, out, plan_sha256, arena_plan=None):
    _plan_hash(plan_sha256)
    rules, bindings = _rules(arena_plan), _model_bindings(models)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    _write(out / 'started.json', {'status': 'started', 'plan_sha256': plan_sha256,
                                'protocol': rules, 'models': bindings})
    start = time.perf_counter()
    try:
        games = []
        for spec in rules['games']:
            game = play(spec, _actor(models[spec['white']]), _actor(models[spec['black']]))
            _write(out / f"{spec['id']}.json", game)
            with (out / f"{spec['id']}.pgn").open('x') as stream:
                stream.write(to_pgn(game))
            validate_game(game, spec)
            games.append(game)
        if _model_bindings(models) != bindings:
            raise ValueError('Models changed during the arena')
        summary = summarize(games, rules)
        _write(out / 'summary.json', summary)
        _write(out / 'completed.json', {'status': 'completed', 'plan_sha256': plan_sha256,
                                       'wall_seconds': time.perf_counter() - start,
                                       'files': {p.name: _hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
        return summary
    except Exception as exc:
        _write(out / 'failed.json', {'status': 'failed', 'plan_sha256': plan_sha256,
                                    'wall_seconds': time.perf_counter() - start,
                                    'error': {'type': type(exc).__name__, 'message': str(exc)},
                                    'files': {p.name: _hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
        raise


def report(saved, plan_sha256, arena_plan=None):
    """Reconstruct histories, PGNs, diagnostics and aggregates without inference."""
    _plan_hash(plan_sha256)
    rules = _rules(arena_plan)
    saved = Path(saved)
    receipt, started = _read(saved / 'completed.json'), _read(saved / 'started.json')
    if (receipt['status'] != 'completed' or started['status'] != 'started'
            or receipt['plan_sha256'] != plan_sha256 or started['plan_sha256'] != plan_sha256
            or started['protocol'] != rules or not math.isfinite(receipt['wall_seconds'])
            or receipt['wall_seconds'] < 0):
        raise ValueError('Incomplete or incorrectly bound arena')
    expected_models = {model_name(o, s): (o, s) for o in OBJECTIVES for s in SEEDS}
    if set(started['models']) != set(expected_models):
        raise ValueError('Incomplete objective/seed model bindings')
    for name, (objective, seed) in expected_models.items():
        binding = started['models'][name]
        _plan_hash(binding['state_sha256'])
        if binding != {'objective': objective, 'seed': seed, 'width': WIDTH, 'depth': DEPTH,
                       'recurrence': 'residual', 'state_sha256': binding['state_sha256']}:
            raise ValueError('Incorrect saved model identity')
    expected = {'started.json', 'summary.json'} | {
        f"{spec['id']}.{ext}" for spec in rules['games'] for ext in ('json', 'pgn')}
    if (set(receipt['files']) != expected
            or {p.name for p in saved.iterdir()} != expected | {'completed.json'}):
        raise ValueError('Saved arena does not cover exactly the fixed schedule')
    for name, digest in receipt['files'].items():
        path = saved / name
        if path.is_symlink() or not path.is_file() or _hash(path) != digest:
            raise ValueError('Arena source hash or regular-file mismatch')
    games = []
    for spec in rules['games']:
        game = validate_game(_read(saved / f"{spec['id']}.json"), spec)
        text = (saved / f"{spec['id']}.pgn").read_text()
        pgn = chess.pgn.read_game(io.StringIO(text))
        if (text != to_pgn(game) or pgn is None or pgn.errors
                or pgn.end().board().fen() != game['final_fen']
                or [move.uci() for move in pgn.mainline_moves()] != [r['move_uci'] for r in game['moves']]):
            raise ValueError('PGN does not preserve the complete native history')
        games.append(game)
    summary = summarize(games, rules)
    if summary != _read(saved / 'summary.json'):
        raise ValueError('Arena summary disagrees with saved-output replay')
    return summary


audit = report
