"""Frozen inference-only depth and exact-successor controls on seen development data.

No new training, checkpoint selection, fresh holdout, Elo or novelty claim.
Prepare binds all inputs and position panels. Run is a single durable attempt.
Report audits saved journals without querying a model or chess engine again.
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

from openjev.research.chess_compute import (
    choose_depth,
    exact_successor_decision,
    terminal_guard_decision,
)
from openjev.research.chess_spatial import MODES, SpatialChess

ROOT = Path(__file__).resolve().parents[1]
SPATIAL_PLAN = ROOT/'evidence/chess-spatial-v1/plan.json'
SPATIAL_EXECUTION = ROOT/'runs/chess-spatial-v1/execution'
SOURCES = ['scripts/chess_compute_study.py', 'tests/test_chess_compute_study.py',
           'src/openjev/research/chess_compute.py', 'tests/test_chess_compute.py',
           'src/openjev/research/chess_spatial.py']
PROTOCOL = {
    'version': 'chess-compute-v1', 'seeds': [17, 29, 43],
    'positions_per_split': 256, 'selection_seed': 10010017,
    'configuration_order_seed': 10010029, 'warmups_per_configuration': 3,
    'warmup_fen': chess.STARTING_FEN, 'device': 'cpu', 'torch_threads': 2,
    'deterministic_algorithms': True,
    'regret_positions_per_split': 64, 'regret_selection_seed': 10010043,
    'regret_nodes': 20000, 'engine_threads': 1, 'engine_hash_mb': 16,
    'value_cp_scale': 600., 'mate_cp': 10000,
    'selection': 'For each ordered split dev/shift use base_seed+split_index; sort sampled original indices; secondary samples sorted indices within selected panel',
    'gate': {'mode': 'predict', 'candidate': 'policy_d8', 'reference': 'policy_d4',
             'min_mean_agreement_gain': .02, 'max_paired_seed_deficit': .01,
             'scope': 'Descriptive gate on already-seen development panels, not untouched efficacy confirmation'},
    'claim_scope': 'Inference controls on previously scored spatial-v1 development data; no fresh holdout, fitting, selection, Elo or novelty claim',
    'timing': 'End-to-end decision wrapper CPU wall time; model loading excluded; three starting-board warmups/configuration reported separately',
    'terminal_policy': 'Native outcomes, claim_draw=False; only selected successors scanned by top-k; separate all-move mate guard',
    'secondary': 'One unrestricted and each unique selected root move at fixed nodes; Clear Hash each call; negative finite-search score differences retained',
}
COUNT_KEYS = ('candidates_evaluated', 'policy_forward_calls', 'boardvalue_evaluations',
              'boardvalue_forward_calls', 'terminal_evaluations', 'terminal_checks', 'symbolic_transition_count')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n')
    stream.flush()


def configurations():
    result = []
    for mode in MODES:
        methods = [{'variant': f'policy_d{depth}', 'method': 'policy', 'depth': depth, 'top_k': None}
                   for depth in ([2, 4] if mode == 'cnn' else [2, 4, 8, 16])]
        methods += [{'variant': 'guard_d4', 'method': 'guard', 'depth': 4, 'top_k': None}]
        methods += [{'variant': f'successor_{count}_d4', 'method': 'successor', 'depth': 4,
                     'top_k': None if count == 'all' else count} for count in (2, 4, 'all')]
        for seed in PROTOCOL['seeds']:
            for method in methods:
                result.append({'id': f"{mode}-{seed}-{method['variant']}", 'mode': mode, 'seed': seed, **method})
    random.Random(PROTOCOL['configuration_order_seed']).shuffle(result)
    return result


def signature(spatial_plan=SPATIAL_PLAN, execution=SPATIAL_EXECUTION, engine_path=None):
    spatial_plan, execution = Path(spatial_plan).resolve(), Path(execution).resolve()
    original = read(spatial_plan)
    complete = read(execution/'completed.json')
    original_hash = sha(spatial_plan)
    expected = {f'{mode}-{seed}' for mode in MODES for seed in PROTOCOL['seeds']}
    if (complete['status'] != 'completed' or complete['plan_sha256'] != original_hash
            or set(complete['fit_receipts']) != expected or complete['fits'] != len(expected)
            or (execution/'failed.json').exists()):
        raise ValueError('Original spatial study did not complete with the fixed model panel')
    if original['sources']['src/openjev/research/chess_spatial.py'] != sha(ROOT/'src/openjev/research/chess_spatial.py'):
        raise ValueError('Original spatial model source changed')
    data_receipt = read(execution/'data/completed.json')
    if data_receipt['status'] != 'completed':
        raise ValueError('Original data is incomplete')
    inputs = {str(spatial_plan): original_hash,
              str(execution/'completed.json'): sha(execution/'completed.json'),
              str(execution/'data/completed.json'): sha(execution/'data/completed.json')}
    weights = {}
    for name in sorted(expected):
        directory = execution/name
        receipt = read(directory/'completed.json')
        if (sha(directory/'completed.json') != complete['fit_receipts'][name]
                or receipt['plan_sha256'] != original_hash or receipt['status'] != 'completed'
                or f"{receipt['mode']}-{receipt['seed']}" != name
                or receipt['data_receipt_sha256'] != sha(execution/'data/completed.json')
                or sha(directory/'weights.pt') != receipt['files']['weights.pt']):
            raise ValueError('Original checkpoint receipt mismatch')
        inputs[str(directory/'completed.json')] = sha(directory/'completed.json')
        inputs[str(directory/'weights.pt')] = sha(directory/'weights.pt')
        weights[name] = str(directory/'weights.pt')
    panels, secondary = {}, {}
    for offset, split in enumerate(('dev', 'shift')):
        path = execution/'data'/f'{split}.jsonl'
        if sha(path) != data_receipt['files'][path.name]:
            raise ValueError('Original development data hash mismatch')
        inputs[str(path)] = sha(path)
        rows = jsonl(path)
        if len(rows) < PROTOCOL['positions_per_split'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Insufficient or duplicate original development panel')
        selected = sorted(random.Random(PROTOCOL['selection_seed']+offset).sample(
            range(len(rows)), PROTOCOL['positions_per_split']))
        panels[split] = [{'original_index': index, **{key: rows[index][key] for key in
                           ('id', 'game_id', 'fen', 'target_uci')}} for index in selected]
        for row in panels[split]:
            board = chess.Board(row['fen'])
            if (not board.is_valid() or board.outcome(claim_draw=False) is not None
                    or chess.Move.from_uci(row['target_uci']) not in board.legal_moves):
                raise ValueError('Invalid selected position or teacher move')
        if PROTOCOL['regret_positions_per_split'] > len(selected):
            raise ValueError('Secondary budget exceeds selected panel')
        secondary[split] = sorted(random.Random(PROTOCOL['regret_selection_seed']+offset).sample(
            range(len(selected)), PROTOCOL['regret_positions_per_split']))
    engine_path = Path(engine_path or original['engine']['path']).resolve()
    inputs[str(engine_path)] = sha(engine_path)
    return {'protocol': PROTOCOL, 'configurations': configurations(), 'panels': panels,
            'secondary_indices': secondary, 'spatial_plan': str(spatial_plan),
            'spatial_execution': str(execution), 'spatial_plan_sha256': original_hash,
            'weights': weights, 'inputs': inputs, 'engine_path': str(engine_path),
            'sources': {name: sha(ROOT/name) for name in SOURCES},
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                            'machine': platform.machine(), 'dependencies': {
                                name: importlib.metadata.version(name) for name in ('chess', 'torch', 'numpy')},
                            'lock_sha256': sha(ROOT/'uv.lock'), 'pyproject_sha256': sha(ROOT/'pyproject.toml')}}


def prepare(out, spatial_plan=SPATIAL_PLAN, execution=SPATIAL_EXECUTION, engine_path=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', signature(spatial_plan, execution, engine_path))
    write_new(out/'prepared.json', {'status': 'prepared', 'created_unix': time.time(),
                                   'plan_sha256': sha(out/'plan.json'), 'evaluation_started': False})
    return out/'plan.json'


def verify_plan(path):
    plan = read(path)
    if plan != signature(plan['spatial_plan'], plan['spatial_execution'], plan['engine_path']):
        raise ValueError('Frozen source, inputs, protocol or environment mismatch')
    return plan


def decide(model, board, config):
    if config['method'] == 'policy':
        return choose_depth(model, board, config['depth'])
    if config['method'] == 'guard':
        return terminal_guard_decision(model, board, config['depth'])
    return exact_successor_decision(model, board, config['top_k'], config['depth'])


class PolicyObserver:
    """Transparent model view that records an existing root policy call, never adds one."""

    def __init__(self, model):
        self.model = model
        self.policy = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def choose(self, board, depth):
        self.policy = self.model.choose(board, depth=depth)
        return self.policy


def decision_record(model, position, config, split, index):
    observer = PolicyObserver(model)
    result = decide(observer, chess.Board(position['fen']), config)
    target = position.get('target_uci')
    root_policy = observer.policy or {}
    probabilities = root_policy.get('probabilities')
    row = {'configuration': config['id'], 'split': split, 'panel_index': index, 'id': position['id'],
           'choice': result['choice'], 'correct': result['choice'] == target if target is not None else None,
           'latency_ms': result['latency_ms'], 'compute': result['compute'],
           'policy_entropy': None, 'policy_max_probability': None, 'target_probability': None,
           'value': root_policy.get('value'), 'chosen_score': result.get('scores', {}).get(result['choice']),
           'evaluated_candidates': sorted(result.get('scores', {})) if config['method'] == 'successor' else None,
           'guard_triggered': result.get('guard_triggered'), 'winning_mates': result.get('winning_mates')}
    if probabilities is not None:
        row.update({'policy_entropy': -sum(p*math.log(p) for p in probabilities.values() if p > 0),
                    'policy_max_probability': max(probabilities.values()),
                    'target_probability': probabilities[target] if target is not None else None})
    return row


def score_secondary(plan, out, decisions):
    costs = {'calls': 0, 'requested_nodes': 0, 'reported_nodes': 0, 'wall_seconds': 0.}
    lookup = {(row['configuration'], row['split'], row['panel_index']): row for row in decisions}
    with chess.engine.SimpleEngine.popen_uci(plan['engine_path']) as engine, \
            (out/'analyses.jsonl').open('x') as analyses, (out/'regret.jsonl').open('x') as stream:
        if not engine.id.get('name', '').startswith('Stockfish 19'):
            raise ValueError('Wrong secondary engine identity')
        engine.configure({'Threads': PROTOCOL['engine_threads'], 'Hash': PROTOCOL['engine_hash_mb']})
        write_new(out/'engine.json', {'id': engine.id, 'path': plan['engine_path'],
                                    'sha256': sha(plan['engine_path'])})
        for split in ('dev', 'shift'):
            for index in plan['secondary_indices'][split]:
                position = plan['panels'][split][index]
                board = chess.Board(position['fen'])
                choices = {cfg['id']: lookup[cfg['id'], split, index]['choice'] for cfg in plan['configurations']}
                scores = {}
                for move in [None]+sorted(set(choices.values())):
                    begin = time.perf_counter()
                    engine.configure({'Clear Hash': None})
                    extra = {} if move is None else {'root_moves': [chess.Move.from_uci(move)]}
                    info = engine.analyse(board.copy(stack=False), chess.engine.Limit(nodes=PROTOCOL['regret_nodes']),
                                          info=chess.engine.INFO_SCORE | chess.engine.INFO_PV | chess.engine.INFO_BASIC,
                                          **extra)
                    elapsed = time.perf_counter()-begin
                    score = info['score'].pov(board.turn)
                    cp = score.score(mate_score=PROTOCOL['mate_cp'])
                    if (cp is None or not info.get('pv') or info['pv'][0] not in board.legal_moves
                            or (move is not None and info['pv'][0].uci() != move)
                            or type(info.get('nodes')) is not int or info['nodes'] < 0):
                        raise ValueError('Invalid secondary engine response')
                    row = {'split': split, 'panel_index': index, 'id': position['id'], 'root_move': move,
                           'pv_first': info['pv'][0].uci(), 'score_cp': int(cp), 'mate': score.mate(),
                           'bounded_score': math.tanh(cp/PROTOCOL['value_cp_scale']),
                           'requested_nodes': PROTOCOL['regret_nodes'], 'reported_nodes': info['nodes'],
                           'wall_seconds': elapsed}
                    append(analyses, row)
                    scores[move] = row
                    costs['calls'] += 1
                    for key in ('requested_nodes', 'reported_nodes', 'wall_seconds'):
                        costs[key] += row[key]
                for identity, move in choices.items():
                    append(stream, {'configuration': identity, 'split': split, 'panel_index': index,
                                    'id': position['id'], 'choice': move,
                                    'bounded_regret': scores[None]['bounded_score']-scores[move]['bounded_score'],
                                    'cp_loss': scores[None]['score_cp']-scores[move]['score_cp']})
    return costs


def run(plan_path, out):
    plan = verify_plan(plan_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'status': 'started', 'started_unix': time.time(), 'plan_sha256': sha(plan_path)})
    try:
        torch.set_num_threads(PROTOCOL['torch_threads'])
        torch.use_deterministic_algorithms(PROTOCOL['deterministic_algorithms'])
        write_new(out/'panels.json', plan['panels'])
        decisions = []
        with (out/'warmups.jsonl').open('x') as warmups, (out/'decisions.jsonl').open('x') as journal:
            for config in plan['configurations']:
                model = SpatialChess.load(plan['weights'][f"{config['mode']}-{config['seed']}"],
                                          expected_plan_sha256=plan['spatial_plan_sha256']).to(PROTOCOL['device'])
                if model.mode != config['mode'] or model.seed != config['seed'] or model.depth != 4:
                    raise ValueError('Loaded checkpoint identity mismatch')
                for index in range(PROTOCOL['warmups_per_configuration']):
                    position = {'id': f'warmup-{index}', 'fen': PROTOCOL['warmup_fen']}
                    append(warmups, decision_record(model, position, config, 'warmup', index))
                for split in ('dev', 'shift'):
                    for index, position in enumerate(plan['panels'][split]):
                        row = decision_record(model, position, config, split, index)
                        append(journal, row)
                        decisions.append(row)
                print(json.dumps({'configuration_completed': config['id'], 'decisions': len(decisions)}), flush=True)
                del model
        cost = score_secondary(plan, out, decisions)
        files = ['panels.json', 'warmups.jsonl', 'decisions.jsonl', 'analyses.jsonl', 'regret.jsonl', 'engine.json']
        write_new(out/'completed.json', {'status': 'completed', 'completed_unix': time.time(),
                                        'plan_sha256': sha(plan_path), 'configurations': len(plan['configurations']),
                                        'decisions': len(decisions), 'secondary_cost': cost,
                                        'files': {name: sha(out/name) for name in files}})
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'completed_unix': time.time(), 'plan_sha256': sha(plan_path),
                                     'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def validate_decision(row, position, config, split, index):
    board = chess.Board(position['fen'])
    legal = sorted(move.uci() for move in board.legal_moves)
    correct = row['choice'] == position['target_uci'] if split != 'warmup' else None
    if (row['configuration'] != config['id'] or row['split'] != split or row['panel_index'] != index
            or row['id'] != position['id'] or row['choice'] not in legal or row['correct'] != correct
            or not math.isfinite(row['latency_ms']) or row['latency_ms'] < 0):
        raise ValueError('Decision identity, choice, correctness or timing mismatch')
    counts = row['compute']
    if (counts['depth'] != config['depth'] or counts['device'] != PROTOCOL['device']
            or counts['legal_count'] != len(legal) or counts['total_wall_ms'] != row['latency_ms']
            or any(type(counts[key]) is not int or counts[key] < 0 for key in COUNT_KEYS)):
        raise ValueError('Invalid compute accounting')
    expected = dict.fromkeys(COUNT_KEYS, 0)
    expected['candidates_evaluated'] = len(legal)
    has_policy = config['method'] == 'policy'
    if config['method'] == 'policy':
        expected['policy_forward_calls'] = 1
        if row['evaluated_candidates'] is not None or row['chosen_score'] is not None:
            raise ValueError('Unexpected successor score on plain policy')
    else:
        candidates = row['evaluated_candidates'] if config['method'] == 'successor' else legal
        if (not isinstance(candidates, list) or candidates != sorted(set(candidates))
                or not set(candidates) <= set(legal) or row['choice'] not in candidates):
            raise ValueError('Invalid evaluated candidate membership')
        required = min(config['top_k'], len(legal)) if config['top_k'] is not None else len(legal)
        if len(candidates) != required:
            raise ValueError('Incomplete successor menu')
        terminals, wins = {}, []
        for name in candidates:
            after = board.copy(stack=True)
            after.push_uci(name)
            outcome = after.outcome(claim_draw=False)
            if outcome is not None:
                terminals[name] = 0. if outcome.winner is None else (1. if outcome.winner == board.turn else -1.)
                if outcome.termination == chess.Termination.CHECKMATE and outcome.winner == board.turn:
                    wins.append(name)
        expected.update({'candidates_evaluated': len(candidates), 'terminal_checks': len(candidates),
                         'symbolic_transition_count': len(candidates), 'terminal_evaluations': len(terminals)})
        if config['method'] == 'successor':
            has_policy = config['top_k'] is not None
            expected.update({'boardvalue_evaluations': len(candidates)-len(terminals),
                             'boardvalue_forward_calls': int(len(candidates) > len(terminals)),
                             'policy_forward_calls': int(config['top_k'] is not None)})
            if (row['chosen_score'] is None or not math.isfinite(row['chosen_score'])
                    or not -1 <= row['chosen_score'] <= 1
                    or (row['choice'] in terminals and row['chosen_score'] != terminals[row['choice']])
                    or (wins and row['choice'] != wins[0])):
                raise ValueError('Invalid chosen successor/native terminal score')
        else:
            has_policy = not wins
            expected['policy_forward_calls'] = int(has_policy)
            if (row['guard_triggered'] != bool(wins) or row['winning_mates'] != wins
                    or (wins and (row['choice'] != wins[0] or row['chosen_score'] != 1.))):
                raise ValueError('Native mate guard mismatch')
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError('Compute counters do not match declared algorithm')
    fields = ('policy_entropy', 'policy_max_probability', 'target_probability', 'value')
    if has_policy:
        entropy, maximum, probability, value = (row[key] for key in fields)
        if (entropy is None or not math.isfinite(entropy) or not -1e-7 <= entropy <= math.log(len(legal))+1e-5
                or maximum is None or not math.isfinite(maximum) or not 1/len(legal)-1e-6 <= maximum <= 1
                or value is None or not math.isfinite(value) or not -1 <= value <= 1
                or (split != 'warmup' and (probability is None or not math.isfinite(probability)
                                          or not 0 <= probability <= maximum+1e-6))
                or (split == 'warmup' and probability is not None)):
            raise ValueError('Invalid policy confidence/value summary')
    elif any(row[key] is not None for key in fields):
        raise ValueError('Value-only or native terminal decision must not claim policy probabilities')


def validate_secondary(plan, decisions, analyses, records, cost):
    decision_lookup = {(r['configuration'], r['split'], r['panel_index']): r for r in decisions}
    expected_analyses, expected_records = set(), set()
    for split in ('dev', 'shift'):
        for index in plan['secondary_indices'][split]:
            moves = {decision_lookup[c['id'], split, index]['choice'] for c in plan['configurations']}
            expected_analyses.update((split, index, move) for move in {None, *moves})
            expected_records.update((c['id'], split, index) for c in plan['configurations'])
    if (len(analyses) != len(expected_analyses)
            or {(r['split'], r['panel_index'], r['root_move']) for r in analyses} != expected_analyses
            or len(records) != len(expected_records)
            or {(r['configuration'], r['split'], r['panel_index']) for r in records} != expected_records):
        raise ValueError('Secondary engine panel coverage mismatch')
    lookup = {}
    for row in analyses:
        position = plan['panels'][row['split']][row['panel_index']]
        board = chess.Board(position['fen'])
        if (row['id'] != position['id'] or chess.Move.from_uci(row['pv_first']) not in board.legal_moves
                or (row['root_move'] is not None and row['pv_first'] != row['root_move'])
                or type(row['score_cp']) is not int or row['requested_nodes'] != PROTOCOL['regret_nodes']
                or type(row['reported_nodes']) is not int or row['reported_nodes'] < 0
                or not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0
                or row['bounded_score'] != math.tanh(row['score_cp']/PROTOCOL['value_cp_scale'])
                or (row['mate'] is not None and (type(row['mate']) is not int
                    or row['score_cp'] != chess.engine.Mate(row['mate']).score(mate_score=PROTOCOL['mate_cp'])))):
            raise ValueError('Invalid secondary engine analysis')
        lookup[row['split'], row['panel_index'], row['root_move']] = row
    for row in records:
        key = row['configuration'], row['split'], row['panel_index']
        decision = decision_lookup[key]
        best = lookup[row['split'], row['panel_index'], None]
        chosen = lookup[row['split'], row['panel_index'], decision['choice']]
        if (row['id'] != decision['id'] or row['choice'] != decision['choice']
                or row['bounded_regret'] != best['bounded_score']-chosen['bounded_score']
                or row['cp_loss'] != best['score_cp']-chosen['score_cp']):
            raise ValueError('Secondary engine subtraction/choice mismatch')
    expected_cost = {'calls': len(analyses), **{key: sum(row[key] for row in analyses)
                                               for key in ('requested_nodes', 'reported_nodes', 'wall_seconds')}}
    if cost != expected_cost:
        raise ValueError('Secondary engine cost mismatch')


def gate(summaries):
    expected = {(c['id'], split) for c in configurations() for split in ('dev', 'shift')}
    if len(summaries) != len(expected) or {(r['configuration'], r['split']) for r in summaries} != expected:
        raise ValueError('Gate requires the complete fixed configuration panel')
    config, checks = PROTOCOL['gate'], []
    lookup = {(r['configuration'], r['split']): r for r in summaries}
    for split in ('dev', 'shift'):
        deltas = [lookup[f"{config['mode']}-{seed}-{config['candidate']}", split]['agreement']-
                  lookup[f"{config['mode']}-{seed}-{config['reference']}", split]['agreement']
                  for seed in PROTOCOL['seeds']]
        for metric, value, threshold in (
            ('mean_agreement_gain', float(np.mean(deltas)), config['min_mean_agreement_gain']),
            ('worst_seed_gain', min(deltas), -config['max_paired_seed_deficit']),
        ):
            checks.append({'split': split, 'metric': metric, 'observed': value,
                           'threshold': threshold, 'passed': value >= threshold})
    return {'passed': all(row['passed'] for row in checks), 'checks': checks, 'scope': config['scope']}


def summarize(plan, decisions, regret, warmups):
    summaries = []
    grouped = {}
    for row in decisions:
        grouped.setdefault((row['configuration'], row['split']), []).append(row)
    regroups = {}
    for row in regret:
        regroups.setdefault((row['configuration'], row['split']), []).append(row)
    for config in plan['configurations']:
        for split in ('dev', 'shift'):
            rows = grouped[config['id'], split]
            baseline = grouped[f"{config['mode']}-{config['seed']}-policy_d4", split]
            paired = [(r['correct'], b['correct']) for r, b in zip(rows, baseline, strict=True)]
            corrected = sum(now and not before for now, before in paired)
            broken = sum(before and not now for now, before in paired)
            latency = [row['latency_ms'] for row in rows]
            secondary = regroups[config['id'], split]
            summaries.append({'configuration': config['id'], 'mode': config['mode'], 'seed': config['seed'],
                              'variant': config['variant'], 'split': split, 'examples': len(rows),
                              'agreement': sum(r['correct'] for r in rows)/len(rows),
                              'latency_mean_ms': float(np.mean(latency)), 'latency_p50_ms': float(np.quantile(latency, .5)),
                              'latency_p95_ms': float(np.quantile(latency, .95)), 'total_wall_ms': sum(latency),
                              'compute_totals': {key: sum(r['compute'][key] for r in rows) for key in COUNT_KEYS},
                              'paired_vs_depth4': {'corrected': corrected, 'broken': broken,
                                                   'net_agreement_gain': (corrected-broken)/len(rows)},
                              'secondary_positions': len(secondary),
                              'mean_bounded_regret': float(np.mean([r['bounded_regret'] for r in secondary])),
                              'mean_cp_loss': float(np.mean([r['cp_loss'] for r in secondary]))})
    return {'configurations': summaries, 'continuation_gate': gate(summaries),
            'warmups': {'calls': len(warmups), 'total_wall_ms': sum(r['latency_ms'] for r in warmups),
                        'compute_totals': {key: sum(r['compute'][key] for r in warmups) for key in COUNT_KEYS}},
            'claim_scope': PROTOCOL['claim_scope'], 'novelty_established': False, 'elo_estimate': None}


def report(plan_path, execution, out):
    plan = verify_plan(plan_path)
    execution = Path(execution)
    complete = read(execution/'completed.json')
    started = read(execution/'started.json')
    required = {'panels.json', 'warmups.jsonl', 'decisions.jsonl', 'analyses.jsonl', 'regret.jsonl', 'engine.json'}
    if (complete['status'] != 'completed' or complete['plan_sha256'] != sha(plan_path)
            or started['status'] != 'started' or started['plan_sha256'] != sha(plan_path)
            or (execution/'failed.json').exists() or set(complete['files']) != required
            or complete['configurations'] != len(plan['configurations'])):
        raise ValueError('Execution did not complete under this plan')
    for name, digest in complete['files'].items():
        if sha(execution/name) != digest:
            raise ValueError('Execution evidence hash mismatch')
    if read(execution/'panels.json') != plan['panels']:
        raise ValueError('Shared position panel mismatch')
    engine = read(execution/'engine.json')
    if (engine['path'] != plan['engine_path'] or engine['sha256'] != plan['inputs'][plan['engine_path']]
            or not engine['id']['name'].startswith('Stockfish 19')):
        raise ValueError('Secondary engine identity mismatch')
    decisions, warmups = jsonl(execution/'decisions.jsonl'), jsonl(execution/'warmups.jsonl')
    expected_count = len(plan['configurations'])*sum(len(rows) for rows in plan['panels'].values())
    if len(decisions) != expected_count or complete['decisions'] != expected_count:
        raise ValueError('Incomplete decision journal')
    if len(warmups) != len(plan['configurations'])*PROTOCOL['warmups_per_configuration']:
        raise ValueError('Incomplete warmup journal')
    cursor, warm_cursor = 0, 0
    for config in plan['configurations']:
        for index in range(PROTOCOL['warmups_per_configuration']):
            position = {'id': f'warmup-{index}', 'fen': PROTOCOL['warmup_fen']}
            validate_decision(warmups[warm_cursor], position, config, 'warmup', index)
            warm_cursor += 1
        for split in ('dev', 'shift'):
            for index, position in enumerate(plan['panels'][split]):
                validate_decision(decisions[cursor], position, config, split, index)
                cursor += 1
    analyses, regret = jsonl(execution/'analyses.jsonl'), jsonl(execution/'regret.jsonl')
    validate_secondary(plan, decisions, analyses, regret, complete['secondary_cost'])
    result = {'status': 'completed', 'plan_sha256': sha(plan_path),
              'execution_receipt_sha256': sha(execution/'completed.json'),
              'secondary_cost': complete['secondary_cost'], **summarize(plan, decisions, regret, warmups)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'summary.json', result)
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(plan_path),
                                    'summary_sha256': sha(out/'summary.json'),
                                    'execution_receipt_sha256': sha(execution/'completed.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--spatial-plan', type=Path, default=SPATIAL_PLAN)
    prep.add_argument('--spatial-execution', type=Path, default=SPATIAL_EXECUTION)
    prep.add_argument('--engine', type=Path)
    for name in ('run', 'report'):
        command = commands.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        if name == 'report':
            command.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.out, args.spatial_plan, args.spatial_execution, args.engine))
    elif args.command == 'run':
        run(args.plan, args.out)
    else:
        print(json.dumps(report(args.plan, args.execution, args.out)['continuation_gate'], indent=2))


if __name__ == '__main__':
    main()
