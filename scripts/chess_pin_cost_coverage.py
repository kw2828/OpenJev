# SPDX-License-Identifier: GPL-3.0-only
"""Explicit adaptive extension of native cost to all128 engineering roots."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess

import chess_pin_cost_followthrough as prior

original = prior.original
ROOT, source = prior.ROOT, prior.source
COUNT = 128
PLAN = 'evidence/chess-pin-native-cost-followthrough-v1/protocol/plan.json'
RUN = 'runs/chess-pin-native-cost-followthrough-v1/execution'
AUDIT = 'evidence/chess-pin-native-cost-followthrough-v1/audit/receipt.json'
CODE = ['scripts/chess_pin_cost_coverage.py', 'tests/test_chess_pin_cost_coverage.py']
PROTOCOL = {
    'version': 'pin-native-cost-coverage-extension-v1',
    'scope': 'Adaptive engineering cost extension, not confirmation. First16-root costs and one numerical violation already known. The initial panel had zero root pins and only16 added factors.',
    'selection': 'All128 original training roots already used by both frozen preflights, including the original16. No subset selected by cost, move quality or neural outcomes.',
    'methods_and_weights': 'Same seven methods, three fresh head seeds, backbone97 and nonzero output projections as original frozen cost protocol.',
    'numerical_gate': 'Unchanged1e-5 maximum absolute score tolerance and exact choice equality for every measured decision. Preserve every finite violation; any violation fails the aggregate gate.',
    'reference': 'Reuse frozen cached reference constructor in eight consecutive16-root chunks with aligned data/features/cache indexing. Independent pin detector and scalar factor/control paths. No teacher supervision.',
    'repeats': 7, 'timed_records': 18816, 'native_audit_decisions': 2688,
    'order': 'Rotate seven methods by(seed_index+global_root_index+repeat)%7; each method occupies each slot once per seed/root. Two starting-position warmups per method/seed.',
    'timing_scope': original.PROTOCOL['timing'],
    'aggregation': 'Median complete milliseconds, per-head-seed medians and matched method/WLDN ratios; raw timings and coverage retained. Original16-root results retained separately; not pooled as independent evidence.',
    'audit': 'Reconstruct all2688 reference vectors exactly, freshly repeat2688 native decisions, authenticate all18816 measurements and recompute every numerical/cost aggregate. Durations not rerun.',
    'runtime': 'CPU2 deterministic, shared host with union training.', 'time_cap_seconds': 900,
    'limits': 'Adaptive extension, fresh heads, one backbone, old training positions and shared-host load. A passing audit can reproduce a failed numerical gate. No trained quality, independent confirmation, speed-advantage acceptance gate or architectural novelty claim.',
}


def signature():
    frozen = original.read(ROOT/PLAN)
    if frozen != prior.signature(): raise ValueError('Followthrough signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    summary = original.read(ROOT/RUN/'summary.json')
    if (audit['status'] != 'passed' or audit['numerical_gate_passed'] is not False
            or audit['timing_records_checked'] != 2352 or audit['native_decisions_replayed'] != 336
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['replays_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'replays.json')
            or summary['pin_input_coverage']['roots_with_pins'] != 0):
        raise ValueError('Prior negative/coverage result changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'followthrough_signature': frozen, 'followthrough_audit_sha256': source.file_hash(ROOT/AUDIT)}


class GraphWindow:
    def __init__(self, graphs, start):
        self.graphs, self.start = graphs, start

    def batch(self, index):
        return self.graphs.batch(index+self.start)


def timing_order():
    methods = original.METHODS
    return [(s, i, r, m) for si, s in enumerate(original.SEEDS) for i in range(COUNT) for r in range(7)
            for m in methods[(si+i+r)%7:]+methods[:(si+i+r)%7]]


def references(seed, model, loaded, data, features, graphs, rows):
    result = []
    for start in range(0, COUNT, 16):
        ds = {k: v[start:start+16] for k, v in data.items()}
        fs = {k: v[start:start+16] for k, v in features.items()}
        chunk, _ = original.cached(seed, model, loaded, ds, fs, GraphWindow(graphs, start), rows[start:start+16])
        for row in chunk: row['root_index'] += start
        result.extend(chunk)
    if len({(r['seed'], r['root_index'], r['method']) for r in result}) != COUNT*7:
        raise AssertionError('Reference window membership differs')
    return result


def summarize(records, expected):
    keys = [(r['seed'], r['root_index'], r['repeat'], r['method']) for r in records]
    if keys != timing_order(): raise AssertionError('Timing membership differs')
    lookup = {(r['seed'], r['root_index'], r['method']): r['prediction'] for r in expected}
    if len(lookup) != 2688 or len(expected) != 2688: raise AssertionError('Reference membership differs')
    for r in records:
        value = prior.compare(r['prediction'], lookup[r['seed'], r['root_index'], r['method']])
        if any(r[k] != v for k, v in value.items()) or not math.isfinite(r['milliseconds']) or r['milliseconds'] <= 0:
            raise AssertionError('Invalid measured record')
    times = {key: r['milliseconds'] for key, r in zip(keys, records)}
    failures = [r for r in records if not (r['score_tolerance_passed'] and r['choice_matches'])]
    return {'median_complete_ms': {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m) for m in original.METHODS},
            'per_head_seed_median_ms': {str(s): {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m and r['seed'] == s) for m in original.METHODS} for s in original.SEEDS},
            'median_paired_ratio_to_wldn': {m: statistics.median(times[s, i, r, m]/times[s, i, r, 'wldn'] for s in original.SEEDS for i in range(COUNT) for r in range(7)) for m in original.METHODS},
            'numerical_gate_passed': not failures, 'failed_timing_records': len(failures),
            'failed_distinct_cases': len({(r['seed'], r['root_index'], r['method']) for r in failures}),
            'choice_changes': sum(not r['choice_matches'] for r in records),
            'max_cached_score_error': max(r['cached_score_max_error'] for r in records)}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen coverage signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'pid': os.getpid(), 'plan_sha256': plan_hash,
                              'audit': audit, 'host_load': list(os.getloadavg())})
    try:
        model, data, features, graphs, rows, setup = original.pins.parent.setup()
        factor_records = [original.candidate_factors(chess.Board(row['fen']), reference=True) for row in rows]
        coverage = original.pins.inputs.summarize(factor_records)
        coverage['factor_rows'] = len(original.pack_factors(factor_records))
        expected, records, replays = [], [], []
        previous_expected = original.read(execution/'expected.json') if audit else None
        for seed in original.SEEDS:
            loaded = original.heads(seed)
            initial = {m: {k: v.clone() for k, v in h.state_dict().items()} for m, h in loaded.items() if h is not None}
            batch = references(seed, model, loaded, data, features, graphs, rows); expected.extend(batch)
            lookup = {(r['root_index'], r['method']): r['prediction'] for r in batch}
            if audit:
                if batch != [r for r in previous_expected if r['seed'] == seed]:
                    raise AssertionError('Cached references differ')
                for i in range(COUNT):
                    for method in original.METHODS:
                        original.deadline(begin)
                        value = original.decision(model, loaded[method], method, rows[i]['fen'])
                        replays.append({'seed': seed, 'root_index': i, 'method': method,
                                        **prior.compare(value, lookup[i, method])})
            else:
                for method in original.METHODS:
                    for _ in range(2): original.decision(model, loaded[method], method, chess.STARTING_FEN)
                with (out/'timings.jsonl').open('a' if records else 'x') as stream:
                    for s, i, repeat, method in timing_order():
                        if s != seed: continue
                        original.deadline(begin); start = time.perf_counter()
                        value = original.decision(model, loaded[method], method, rows[i]['fen'])
                        record = {'seed': seed, 'root_index': i, 'repeat': repeat, 'method': method,
                                  'prediction': value, 'milliseconds': 1000*(time.perf_counter()-start),
                                  **prior.compare(value, lookup[i, method])}
                        records.append(record); stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            for method, state in initial.items(): original.pins.compare_states(state, loaded[method].state_dict(), 0)
            print(json.dumps({'head_seed': seed, 'audit': audit}), flush=True)
        original.deadline(begin)
        if audit:
            records = source.prior.rows(execution/'timings.jsonl'); old = original.read(execution/'summary.json')
            stats = summarize(records, expected)
            if (old['status'] != 'completed' or old['plan_sha256'] != plan_hash
                    or any(old[k] != v for k, v in stats.items()) or old['pin_input_coverage'] != coverage
                    or not 0 < old['wall_seconds'] <= 900 or old['limits'] != PROTOCOL['limits']):
                raise AssertionError('Saved summary differs')
            saved = {(r['seed'], r['root_index'], r['method']): r for r in records}
            if len(replays) != 2688: raise AssertionError('Incomplete native replay')
            for r in replays:
                previous = saved[r['seed'], r['root_index'], r['method']]
                if any(r[k] != previous[k] for k in ('cached_score_max_error', 'score_tolerance_passed', 'choice_matches')):
                    raise AssertionError('Native replay differs')
            write(out/'replays.json', replays)
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'replays_sha256': source.file_hash(out/'replays.json'), 'timing_records_checked': len(records),
                      'native_decisions_replayed': len(replays), 'cached_reference_vectors_exact': len(expected),
                      **stats, 'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
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
