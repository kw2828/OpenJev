# SPDX-License-Identifier: GPL-3.0-only
"""Complete native cost of pin-factor candidates and information controls."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess
import torch

import chess_pin_control_preflight as controls
from openjev.research.chess_pin_factor_head import ChessPinFactorHead, pack_factors
from openjev.research.chess_pin_factor_controls import ChessPinFactorControl
from openjev.research.chess_pin_factors import candidate_factors, pin_witnesses
from openjev.research.wldn_baseline import ChessWLDNHead

pins = controls.parent
source, ROOT = pins.source, pins.ROOT
PLAN = 'evidence/chess-pin-control-preflight-v1/protocol/plan.json'
RUN = 'runs/chess-pin-control-preflight-v1/execution'
AUDIT = 'evidence/chess-pin-control-preflight-v1/audit/receipt.json'
METHODS = ('base', 'wldn', 'joint', 'separable', 'root_only', 'counts', 'graph_mlp')
SEEDS = (211, 223, 227)
CODE = ['scripts/chess_pin_native_cost.py', 'tests/test_chess_pin_native_cost.py']
PROTOCOL = {
    'version': 'pin-factor-complete-native-cost-v1',
    'scope': 'Descriptive engineering cost, fresh nonzero-projection heads. No quality fitting or evaluation.',
    'methods': list(METHODS), 'head_seeds': list(SEEDS), 'backbone_seed': 97,
    'roots': 16, 'panel': 'First16 original training roots and every legal candidate; no quality-dependent selection.',
    'weights': 'Fresh heads with their prescribed seed initialization; output projection Normal(0,.1), generator900000+head_seed. No fitted adapter weights. Frozen backbone97.',
    'timing': 'Complete FEN parse, board/legal-candidate encoding, backbone, action features, native root/child graphs, required pin extraction/packing, head, argmax and score-list serialization. Models loaded outside timing. No cached positions, features or graphs in a timed call.',
    'factor_paths': 'Joint/separable/counts reconstruct all candidate pin transitions. Root-only extracts root witnesses once and repeats them as retained per candidate. Graph MLP/WLDN/base do not extract pins. Base skips graphs and extra action features.',
    'verification': 'All native menus, scores and selected moves checked against separately prepared cached-batch scores using the independent pin detector and scalar factor/count reference. Every timed result checked. Audit freshly repeats336 native decisions and recomputes all cached scores and timing arithmetic; durations are authenticated, not rerun.',
    'repeats': 7, 'warmups_per_method_seed': 2, 'timed_records': 2352,
    'order': 'Rotate seven methods by(seed_index+root_index+repeat)%7; each method occupies every slot once per seed/root.',
    'aggregation': 'Median full milliseconds and median matched ratio to WLDN over identical head-seed/root/repeat keys; also per-head-seed medians. No speed or quality acceptance threshold.',
    'absolute_score_tolerance': 1e-5, 'native_audit_decisions': 336,
    'runtime': 'CPU2 deterministic, same process, shared host with union training; host load recorded.',
    'time_cap_seconds': 900,
    'limits': 'Fresh heads and16 old training roots, one backbone, shared host. No trained-policy latency, throughput, hardware-general speed, equal FLOPs, teacher quality, engine strength, biological advantage or novelty claim. No new quality run authorized by a timing result.',
}


def read(path):
    return json.loads(Path(path).read_text())


def signature():
    frozen = read(ROOT/PLAN)
    if frozen != controls.signature():
        raise ValueError('Frozen control source changed')
    complete = pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = read(ROOT/AUDIT)
    replay = ROOT/Path(AUDIT).parent/'replay'
    pins.diagnostic.manifest(replay, source.file_hash(ROOT/PLAN))
    if (audit['status'] != 'passed' or audit['plan_sha256'] != source.file_hash(ROOT/PLAN)
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_completed_sha256'] != source.file_hash(replay/'completed.json')
            or audit['score_checks'] != 35280 or audit['artificial_updates_replayed'] != 9):
        raise ValueError('Control engineering audit identity changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'control_signature': frozen, 'control_audit_sha256': source.file_hash(ROOT/AUDIT)}


def deadline(begin):
    if time.monotonic()-begin > PROTOCOL['time_cap_seconds']:
        raise TimeoutError('Frozen fifteen-minute pin cost ceiling exceeded')


def timing_order():
    return [(s, i, r, m) for si, s in enumerate(SEEDS) for i in range(16) for r in range(7)
            for m in METHODS[(si+i+r)%7:]+METHODS[:(si+i+r)%7]]


def heads(seed):
    result = {'base': None, 'wldn': ChessWLDNHead(seed=seed),
              **{arm: ChessPinFactorHead(arm, seed=seed) for arm in ('joint', 'separable')},
              **{arm: ChessPinFactorControl(arm, seed=seed) for arm in controls.ARMS}}
    with torch.no_grad():
        for head in result.values():
            if head is not None:
                generator = torch.Generator().manual_seed(900000+seed)
                head.output.weight.copy_(torch.randn(head.output.weight.shape, generator=generator)*.1)
                head.eval()
    return result


def root_factors(board, count):
    before = pin_witnesses(board, board.turn)
    return torch.tensor([(candidate, *witness, 0) for candidate in range(count) for witness in before],
                        dtype=torch.long).reshape(-1, 6)


def result(names, scores):
    if scores.shape != (1, len(names)) or not len(names) or not torch.isfinite(scores).all():
        raise AssertionError('Invalid complete native score vector')
    return {'menus': list(names), 'choice': names[int(scores.argmax(-1))], 'scores': scores[0].tolist()}


@torch.no_grad()
def decision(model, head, method, fen):
    if method not in METHODS or (head is None) != (method == 'base'):
        raise ValueError('Invalid native method/head pair')
    board = chess.Board(fen)
    names, encoded = source.prior.encode_candidates(board)
    candidates = torch.from_numpy(encoded)[None]; mask = torch.ones(1, len(names), dtype=torch.bool)
    scores, _, hidden = model(torch.from_numpy(source.prior.encode_board(board))[None], candidates, mask)
    if head is not None:
        nodes = hidden.flatten(2).transpose(1, 2)
        actions = source.prior.candidate_features(model, nodes, candidates)
        graph = source.candidate_graphs([board])
        if list(names) != graph['menus'][0] or not torch.equal(mask, graph['mask']):
            raise AssertionError('Native graph/action menus differ')
        args = [nodes, actions, candidates, mask, scores, graph['root'], graph['children'][mask]]
        if method == 'wldn':
            scores = head(*args)
        else:
            if method == 'root_only':
                factors = root_factors(board, len(names))
            elif method == 'graph_mlp':
                factors = torch.empty(0, 6, dtype=torch.long)
            else:
                record = candidate_factors(board)
                if [c['uci'] for c in record['candidates']] != list(names):
                    raise AssertionError('Pin/action menus differ')
                factors = pack_factors([record])
            scores = head(*args, factors)
    return result(names, scores)


def score_error(observed, expected):
    if (observed['menus'] != expected['menus'] or observed['choice'] != expected['choice']
            or observed['choice'] not in observed['menus']
            or len(observed['scores']) != len(expected['scores'])
            or len(observed['scores']) != len(observed['menus'])):
        raise AssertionError('Native prediction membership differs')
    if any(not math.isfinite(v) for v in observed['scores']+expected['scores']):
        raise AssertionError('Nonfinite score')
    error = max(abs(a-b) for a, b in zip(observed['scores'], expected['scores']))
    if error > PROTOCOL['absolute_score_tolerance']:
        raise AssertionError('Frozen native score tolerance exceeded')
    return error


@torch.no_grad()
def cached(seed, model, loaded, data, features, graphs, rows):
    index = torch.arange(16); graph = graphs.batch(index)
    args = source.arguments(model, data, features, index, graph)
    boards = [chess.Board(row['fen']) for row in rows[:16]]
    native = source.candidate_graphs(boards)
    if (native['menus'] != [list(menu) for menu in graph['menus']]
            or not torch.equal(native['root'], graph['root'])
            or not torch.equal(native['children'][native['mask']], graph['children'])):
        raise AssertionError('Native graphs and cache differ')
    records = [candidate_factors(board, reference=True) for board in boards]
    if [tuple(c['uci'] for c in r['candidates']) for r in records] != graph['menus']:
        raise AssertionError('Reference pin/action menus differ')
    factors = pack_factors(records); expected = []
    for method in METHODS:
        scores = args[4] if method == 'base' else (loaded[method](*args) if method == 'wldn'
                 else loaded[method](*args, factors, reference=True))
        for i, menu in enumerate(graph['menus']):
            expected.append({'seed': seed, 'root_index': i, 'id': rows[i]['id'], 'method': method,
                             'prediction': result(menu, scores[i:i+1, :len(menu)])})
    coverage = pins.inputs.summarize(records)
    coverage['factor_rows'] = len(factors)
    return expected, coverage


def summarize(records, expected):
    keys = [(r['seed'], r['root_index'], r['repeat'], r['method']) for r in records]
    if keys != timing_order():
        raise AssertionError('Incomplete or misordered timing membership')
    lookup = {(r['seed'], r['root_index'], r['method']): r['prediction'] for r in expected}
    if len(lookup) != 336 or len(expected) != 336:
        raise AssertionError('Incomplete reference membership')
    for r in records:
        if not math.isfinite(r['milliseconds']) or r['milliseconds'] <= 0:
            raise AssertionError('Invalid timing')
        error = score_error(r['prediction'], lookup[r['seed'], r['root_index'], r['method']])
        if error != r['cached_score_max_error']:
            raise AssertionError('Stored score error differs')
    times = {k: r['milliseconds'] for k, r in zip(keys, records)}
    return {'median_complete_ms': {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m) for m in METHODS},
            'per_head_seed_median_ms': {str(s): {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m and r['seed'] == s) for m in METHODS} for s in SEEDS},
            'median_paired_ratio_to_wldn': {m: statistics.median(times[s, i, r, m]/times[s, i, r, 'wldn'] for s in SEEDS for i in range(16) for r in range(7)) for m in METHODS},
            'max_cached_score_error': max(r['cached_score_max_error'] for r in records)}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = read(plan_path)
    if plan != signature():
        raise ValueError('Frozen native cost signature changed')
    plan_hash = source.file_hash(plan_path); auditing = execution is not None
    if auditing:
        pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False)
    write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'pid': os.getpid(), 'plan_sha256': plan_hash,
                              'audit': auditing, 'host_load': list(os.getloadavg())})
    try:
        model, data, features, graphs, rows, setup = pins.parent.setup()
        expected, timings, replays = [], [], []
        coverage = None
        old_expected = read(execution/'expected.json') if auditing else None
        for seed in SEEDS:
            loaded = heads(seed)
            initial = {m: {k: v.clone() for k, v in h.state_dict().items()} for m, h in loaded.items() if h is not None}
            batch, pin_coverage = cached(seed, model, loaded, data, features, graphs, rows)
            expected.extend(batch)
            if coverage is not None and coverage != pin_coverage:
                raise AssertionError('Pin coverage changed across head seeds')
            coverage = pin_coverage
            lookup = {(r['root_index'], r['method']): r['prediction'] for r in batch}
            if auditing:
                previous = [r for r in old_expected if r['seed'] == seed]
                if batch != previous:
                    raise AssertionError('Cached reference scores did not reproduce exactly')
                for i in range(16):
                    for method in METHODS:
                        deadline(begin)
                        value = decision(model, loaded[method], method, rows[i]['fen'])
                        replays.append({'seed': seed, 'root_index': i, 'method': method,
                                        'cached_score_max_error': score_error(value, lookup[i, method])})
            else:
                for method in METHODS:
                    for _ in range(2): decision(model, loaded[method], method, chess.STARTING_FEN)
                for s, i, repeat, method in timing_order():
                    if s != seed: continue
                    deadline(begin); start = time.perf_counter()
                    value = decision(model, loaded[method], method, rows[i]['fen'])
                    elapsed = 1000*(time.perf_counter()-start)
                    timings.append({'seed': seed, 'root_index': i, 'repeat': repeat, 'method': method,
                                    'prediction': value, 'milliseconds': elapsed,
                                    'cached_score_max_error': score_error(value, lookup[i, method])})
            for method, state in initial.items():
                pins.compare_states(state, loaded[method].state_dict(), 0)
            print(json.dumps({'head_seed': seed, 'audit': auditing}), flush=True)
        deadline(begin)
        if auditing:
            old = read(execution/'summary.json'); timings = source.prior.rows(execution/'timings.jsonl')
            stats = summarize(timings, expected)
            if (old['status'] != 'completed' or old['plan_sha256'] != plan_hash
                    or any(old[k] != v for k, v in stats.items()) or len(replays) != 336
                    or not 0 < old['wall_seconds'] <= 900 or old['limits'] != PROTOCOL['limits']
                    or old['pin_input_coverage'] != coverage):
                raise AssertionError('Saved native cost summary differs')
            write(out/'replays.json', replays)
            receipt = {'status': 'passed', 'plan_sha256': plan_hash,
                       'summary_sha256': source.file_hash(execution/'summary.json'),
                       'completed_sha256': source.file_hash(execution/'completed.json'),
                       'replays_sha256': source.file_hash(out/'replays.json'),
                       'timing_records_checked': len(timings), 'native_decisions_replayed': len(replays),
                       'cached_reference_vectors_exact': len(expected),
                       'max_native_replay_score_error': max(r['cached_score_max_error'] for r in replays),
                       'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', receipt); print(json.dumps(receipt), flush=True)
        else:
            stats = summarize(timings, expected)
            with (out/'timings.jsonl').open('x') as stream:
                for record in timings: stream.write(json.dumps(record, allow_nan=False)+'\n')
            write(out/'expected.json', expected)
            summary = {'status': 'completed', 'plan_sha256': plan_hash, **stats,
                       'setup': setup, 'pin_input_coverage': coverage,
                       'host_load': list(os.getloadavg()), 'wall_seconds': time.monotonic()-begin,
                       'limits': PROTOCOL['limits']}
            write(out/'summary.json', summary)
            write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                  'files': {p.name: source.file_hash(p) for p in out.iterdir() if p.is_file()}})
            print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'failed.json', {'error': repr(error), 'wall_seconds': time.monotonic()-begin}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'audit'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.command == 'prepare':
        plan = signature(); args.out.mkdir(parents=True, exist_ok=False)
        source.prior.write(args.out/'plan.json', plan)
        print(json.dumps({'plan_sha256': source.file_hash(args.out/'plan.json')}), flush=True)
    else: execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
