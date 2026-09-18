# SPDX-License-Identifier: GPL-3.0-only
"""Verify the versioned native feature contract on all three frozen backbones."""
import argparse
import json
from pathlib import Path
import time

import chess
import torch

import chess_native_feature_preflight as parent

ROOT, source, original, native = parent.ROOT, parent.source, parent.original, parent.native
SEEDS = (97, 109, 127)
PLAN = 'evidence/chess-native-feature-preflight-v1/protocol/plan.json'
RUN = 'runs/chess-native-feature-preflight-v1/execution'
AUDIT = 'evidence/chess-native-feature-preflight-v1/audit/receipt.json'
CODE = ['scripts/chess_native_feature_multiseed.py', 'tests/test_chess_native_feature_multiseed.py']
PROTOCOL = {
    'version': 'single-position-native-features-three-backbones-v1',
    'scope': 'Engineering extension of the frozen native-cache contract. All3 original backbones; same128 original training roots; no target-based selection or quality evaluation.',
    'backbone_seeds': list(SEEDS), 'head_seeds': list(original.SEEDS), 'methods': list(original.METHODS),
    'roots': 128, 'root_backbone_head_method_checks': 8064, 'candidate_score_checks': 246960,
    'absolute_score_tolerance': 1e-5,
    'criterion': 'Every complete cached/native score vector meets maximum absolute1e-5 and exact choice equality. Seed97 must exactly reproduce the prior single-backbone records. Every finite failure is retained and fails the aggregate numerical gate.',
    'reference': 'Primary uses frozen native-cache builder/assembler; audit uses previously frozen explicit reference loop and pad_sequence assembler. Neural kernels and board encoders shared; cached pins use independent detector and reference pooling.',
    'layouts': 'Backbone one position and its own legal menu per call; cached head batches16 consecutive roots; fresh nonzero-projection heads and native inference unchanged.',
    'runtime': 'CPU2 deterministic, shared host;900 seconds per execution/audit.', 'time_cap_seconds': 900,
    'limits': '128 already exposed engineering roots, fresh heads. Not full-dataset integrity, trained quality, game strength, independent confirmation, timing advantage or novelty. Historical input contracts and failures remain unchanged.',
}


def signature():
    plan = original.read(ROOT/PLAN)
    if plan != parent.signature(): raise ValueError('Frozen single-backbone feature signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or not audit['numerical_gate_passed'] or not audit['cache_exact']
            or audit['root_seed_method_checks'] != 2688 or audit['candidate_score_checks'] != 82320
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_records_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'records.jsonl')):
        raise ValueError('Single-backbone replay identity changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'parent_signature': plan, 'parent_audit_sha256': source.file_hash(ROOT/AUDIT)}


def summarize(records):
    keys = [(r['backbone_seed'], r['seed'], r['root_index'], r['method']) for r in records]
    expected = [(b, s, i, m) for b in SEEDS for s in original.SEEDS for start in range(0, 128, 16)
                for m in original.METHODS for i in range(start, start+16)]
    if keys != expected: raise AssertionError('Incomplete or misordered engineering comparisons')
    for r in records:
        checked = parent.prior.prior.compare(r['native'], r['cached'])
        if any(r[k] != v for k, v in checked.items()): raise AssertionError('Stored numerical comparison differs')
    checks = sum(len(r['native']['scores']) for r in records)
    failures = [r for r in records if not (r['score_tolerance_passed'] and r['choice_matches'])]
    return {'numerical_gate_passed': not failures, 'failed_distinct_cases': len(failures),
            'root_backbone_head_method_checks': len(records), 'candidate_score_checks': checks,
            'choice_changes': sum(not r['choice_matches'] for r in records),
            'max_cached_score_error': max(r['cached_score_max_error'] for r in records),
            'per_backbone_max_error': {str(b): max(r['cached_score_max_error'] for r in records if r['backbone_seed'] == b) for b in SEEDS}}


@torch.no_grad()
def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen three-backbone signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        rows = source.prior.rows(ROOT/source.prior.DATA['train'])[:128]
        boards = [chess.Board(row['fen']) for row in rows]
        graphs = source.ChildGraphCache(ROOT/source.CACHE/'train')
        old_plan = original.read(ROOT/source.PARENT)
        old_records = {(r['seed'], r['root_index'], r['method']): r for r in source.prior.rows(ROOT/RUN/'records.jsonl')}
        records = []; prior_replays = 0
        with (out/'records.jsonl').open('x') as stream:
            for backbone_seed in SEEDS:
                original.deadline(begin); model = source.prior.backbone(backbone_seed, old_plan)
                model_state = {k: v.clone() for k, v in model.state_dict().items()}
                cache = parent.reference_cache(model, rows) if audit else native.build(model, boards)
                if audit:
                    parent.compare_cache(torch.load(execution/f'cache-{backbone_seed}.pt', weights_only=True, map_location='cpu'), cache)
                else: source.prior.save(out/f'cache-{backbone_seed}.pt', cache)
                if backbone_seed == 97:
                    parent.compare_cache(torch.load(ROOT/RUN/'cache.pt', weights_only=True, map_location='cpu'), cache)
                for seed in original.SEEDS:
                    loaded = original.heads(seed)
                    state = {m: {k: v.clone() for k, v in h.state_dict().items()} for m, h in loaded.items() if h is not None}
                    for start in range(0, 128, 16):
                        original.deadline(begin); index = torch.arange(start, start+16); graph = graphs.batch(index)
                        args = parent.reference_arguments(cache, index.tolist(), graph) if audit else native.arguments(cache, index, graph)
                        factor_rows = [original.candidate_factors(boards[i], reference=True) for i in index]
                        if [tuple(c['uci'] for c in row['candidates']) for row in factor_rows] != graph['menus']:
                            raise AssertionError('Factor menu identity differs')
                        factors = original.pack_factors(factor_rows)
                        for method in original.METHODS:
                            logits = args[4] if method == 'base' else (loaded[method](*args) if method == 'wldn'
                                     else loaded[method](*args, factors, reference=True))
                            for slot, i in enumerate(index.tolist()):
                                names = graph['menus'][slot]
                                cached = original.result(names, logits[slot:slot+1, :len(names)])
                                current = original.decision(model, loaded[method], method, rows[i]['fen'])
                                record = {'backbone_seed': backbone_seed, 'seed': seed, 'root_index': i,
                                          'id': rows[i]['id'], 'method': method,
                                          **parent.prior.prior.compare(current, cached), 'cached': cached, 'native': current}
                                if backbone_seed == 97:
                                    previous = old_records[seed, i, method]
                                    if any(record[k] != previous[k] for k in record if k != 'backbone_seed'):
                                        raise AssertionError('Seed97 prior comparison changed')
                                    prior_replays += 1
                                records.append(record); stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                    for method, saved in state.items(): original.pins.compare_states(saved, loaded[method].state_dict(), 0)
                original.pins.compare_states(model_state, model.state_dict(), 0)
                print(json.dumps({'backbone_seed': backbone_seed, 'audit': audit}), flush=True)
        stats = summarize(records)
        if stats['candidate_score_checks'] != 246960 or prior_replays != 2688:
            raise AssertionError('Incomplete score or prior-record coverage')
        original.deadline(begin)
        if audit:
            old = original.read(execution/'summary.json')
            if (source.prior.rows(execution/'records.jsonl') != records or any(old[k] != v for k, v in stats.items())
                    or old['plan_sha256'] != plan_hash or old['prior_seed97_records_exact'] != prior_replays
                    or old['status'] != 'completed' or not 0 < old['wall_seconds'] <= 900):
                raise AssertionError('Saved all-backbone evidence differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash, 'summary_sha256': source.file_hash(execution/'summary.json'),
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'reference_records_sha256': source.file_hash(out/'records.jsonl'),
                      'caches_exact': 3, 'prior_seed97_records_exact': prior_replays, **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            result = {'status': 'completed', 'plan_sha256': plan_hash, 'prior_seed97_records_exact': prior_replays,
                      **stats, 'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'summary.json', result)
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
