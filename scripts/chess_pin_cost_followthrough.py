# SPDX-License-Identifier: GPL-3.0-only
"""Collect complete cost evidence while preserving the failed numerical gate."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess

import chess_pin_native_cost as original

ROOT, source = original.ROOT, original.source
PARENT_PLAN = 'evidence/chess-pin-native-cost-v1/protocol/plan.json'
FAILURE = 'runs/chess-pin-native-cost-v1/execution/failed.json'
DIAGNOSTIC = 'evidence/chess-pin-native-cost-v1/diagnostic/receipt.json'
CODE = ['scripts/chess_pin_cost_followthrough.py', 'tests/test_chess_pin_cost_followthrough.py']
PROTOCOL = {
    'version': 'pin-native-cost-failure-followthrough-v1',
    'scope': 'Post-failure completion of descriptive cost evidence. Original timing scope, methods, seeds,16 roots,2352 records and1e-5 score tolerance unchanged.',
    'known_failure': 'Original screen stopped at score tolerance; post-hoc336-case diagnostic found one WLDN case at1.0013580322265625e-5, zero changed choices, mainly float32 backbone batch-layout differences.',
    'change': 'Record every finite matching-menu prediction and its tolerance/choice result before continuing. Aggregate the original numerical criterion without aborting on a finite violation. No violation becomes a pass; original failed execution remains untouched.',
    'numerical_gate': 'All2352 predictions must have maximum absolute score error<=1e-5 and the same choice as the independently prepared reference. Any failure leaves numerical_gate_passed false.',
    'audit': 'Freshly recompute all336 cached reference vectors and native decisions; reproduce all raw timing error fields, choices, gate membership and summary arithmetic. Timing durations are authenticated, not rerun.',
    'time_cap_seconds': 900,
    'limits': 'The same shared-host, fresh-weight and one-backbone limitations as the parent apply. A completed execution or successful audit is not a passing numerical gate, quality result or permission to promote the architecture.',
}


def signature():
    plan = original.read(ROOT/PARENT_PLAN)
    if plan != original.signature():
        raise ValueError('Original frozen native screen changed')
    diagnostic = original.read(ROOT/DIAGNOSTIC)
    if (diagnostic['status'] != 'completed' or diagnostic['over_tolerance'] != 1
            or diagnostic['choice_changes'] != 0 or diagnostic['records'] != 336
            or diagnostic['plan_sha256'] != source.file_hash(ROOT/PARENT_PLAN)
            or diagnostic['failure_sha256'] != source.file_hash(ROOT/FAILURE)
            or diagnostic['records_sha256'] != source.file_hash(ROOT/Path(DIAGNOSTIC).parent/'records.json')
            or diagnostic['diagnostic_source_sha256'] != source.file_hash(ROOT/'scripts/chess_pin_cost_diagnostic.py')):
        raise ValueError('Preserved failure diagnostic changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'original_signature': plan, 'failure_sha256': source.file_hash(ROOT/FAILURE),
            'diagnostic_sha256': source.file_hash(ROOT/DIAGNOSTIC)}


def compare(observed, expected):
    if (observed['menus'] != expected['menus'] or not observed['menus']
            or observed['choice'] not in observed['menus'] or expected['choice'] not in expected['menus']
            or len(observed['scores']) != len(expected['scores'])
            or len(observed['scores']) != len(observed['menus'])
            or any(not math.isfinite(v) for v in observed['scores']+expected['scores'])):
        raise AssertionError('Malformed native/reference prediction')
    for result in (observed, expected):
        if result['menus'][max(range(len(result['scores'])), key=result['scores'].__getitem__)] != result['choice']:
            raise AssertionError('Choice is not the recorded argmax')
    error = max(abs(a-b) for a, b in zip(observed['scores'], expected['scores']))
    return {'cached_score_max_error': error, 'score_tolerance_passed': error <= 1e-5,
            'choice_matches': observed['choice'] == expected['choice']}


def summarize(records, expected):
    keys = [(r['seed'], r['root_index'], r['repeat'], r['method']) for r in records]
    if keys != original.timing_order():
        raise AssertionError('Timing membership differs')
    lookup = {(r['seed'], r['root_index'], r['method']): r['prediction'] for r in expected}
    if len(lookup) != 336 or len(expected) != 336:
        raise AssertionError('Reference membership differs')
    for r in records:
        checked = compare(r['prediction'], lookup[r['seed'], r['root_index'], r['method']])
        if any(r[k] != v for k, v in checked.items()) or not math.isfinite(r['milliseconds']) or r['milliseconds'] <= 0:
            raise AssertionError('Timing validation differs')
    times = {key: r['milliseconds'] for key, r in zip(keys, records)}
    failures = [r for r in records if not (r['score_tolerance_passed'] and r['choice_matches'])]
    return {'median_complete_ms': {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m) for m in original.METHODS},
            'median_paired_ratio_to_wldn': {m: statistics.median(times[s, i, r, m]/times[s, i, r, 'wldn'] for s in original.SEEDS for i in range(16) for r in range(7)) for m in original.METHODS},
            'per_head_seed_median_ms': {str(s): {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m and r['seed'] == s) for m in original.METHODS} for s in original.SEEDS},
            'numerical_gate_passed': not failures, 'failed_timing_records': len(failures),
            'failed_distinct_cases': len({(r['seed'], r['root_index'], r['method']) for r in failures}),
            'choice_changes': sum(not r['choice_matches'] for r in records),
            'max_cached_score_error': max(r['cached_score_max_error'] for r in records)}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen followthrough signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'pid': os.getpid(), 'plan_sha256': plan_hash,
                              'audit': audit, 'host_load': list(os.getloadavg())})
    try:
        model, data, features, graphs, rows, setup = original.pins.parent.setup()
        expected, records, replays = [], [], []; coverage = None
        old_expected = original.read(execution/'expected.json') if audit else None
        for seed in original.SEEDS:
            loaded = original.heads(seed)
            initial = {m: {k: v.clone() for k, v in h.state_dict().items()} for m, h in loaded.items() if h is not None}
            batch, current = original.cached(seed, model, loaded, data, features, graphs, rows)
            if coverage is not None and current != coverage: raise AssertionError('Coverage changed')
            coverage = current; expected.extend(batch)
            lookup = {(r['root_index'], r['method']): r['prediction'] for r in batch}
            if audit:
                if batch != [r for r in old_expected if r['seed'] == seed]:
                    raise AssertionError('Reference vectors did not reproduce exactly')
                for i in range(16):
                    for method in original.METHODS:
                        original.deadline(begin)
                        value = original.decision(model, loaded[method], method, rows[i]['fen'])
                        replays.append({'seed': seed, 'root_index': i, 'method': method,
                                        **compare(value, lookup[i, method])})
            else:
                for method in original.METHODS:
                    for _ in range(2): original.decision(model, loaded[method], method, chess.STARTING_FEN)
                # Persist each measurement, including a failing numerical result.
                with (out/'timings.jsonl').open('a' if records else 'x') as stream:
                    for s, i, repeat, method in original.timing_order():
                        if s != seed: continue
                        original.deadline(begin); start = time.perf_counter()
                        value = original.decision(model, loaded[method], method, rows[i]['fen'])
                        milliseconds = 1000*(time.perf_counter()-start)
                        record = {'seed': seed, 'root_index': i, 'repeat': repeat, 'method': method,
                                  'prediction': value, 'milliseconds': milliseconds, **compare(value, lookup[i, method])}
                        records.append(record); stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            for method, state in initial.items(): original.pins.compare_states(state, loaded[method].state_dict(), 0)
            print(json.dumps({'head_seed': seed, 'audit': audit}), flush=True)
        original.deadline(begin)
        if audit:
            records = source.prior.rows(execution/'timings.jsonl'); previous = original.read(execution/'summary.json')
            stats = summarize(records, expected)
            if (any(previous[k] != v for k, v in stats.items()) or previous['pin_input_coverage'] != coverage
                    or previous['plan_sha256'] != plan_hash or previous['status'] != 'completed'
                    or not 0 < previous['wall_seconds'] <= 900 or previous['limits'] != PROTOCOL['limits']
                    or len(replays) != 336): raise AssertionError('Saved summary differs')
            saved = {(r['seed'], r['root_index'], r['method']): r for r in records}
            for r in replays:
                prior = saved[r['seed'], r['root_index'], r['method']]
                if any(r[k] != prior[k] for k in ('cached_score_max_error', 'score_tolerance_passed', 'choice_matches')):
                    raise AssertionError('Native replay result differs')
            write(out/'replays.json', replays)
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'replays_sha256': source.file_hash(out/'replays.json'),
                      'timing_records_checked': len(records), 'native_decisions_replayed': len(replays),
                      'cached_reference_vectors_exact': len(expected), **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            result = {'status': 'completed', 'plan_sha256': plan_hash, **summarize(records, expected),
                      'pin_input_coverage': coverage, 'setup': setup, 'host_load': list(os.getloadavg()),
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'summary.json', result); write(out/'expected.json', expected)
            write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                  'files': {p.name: source.file_hash(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps(result), flush=True)
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
