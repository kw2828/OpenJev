# SPDX-License-Identifier: GPL-3.0-only
"""Versioned feature-cache repair with unchanged native policy and score gate."""
import argparse
import json
from pathlib import Path
import time

import chess
import torch
from torch.nn.utils.rnn import pad_sequence

import chess_pin_cost_coverage as prior
from openjev.research import chess_native_feature_cache as native

original = prior.original
ROOT, source = prior.ROOT, prior.source
PLAN = 'evidence/chess-pin-native-cost-coverage-v1/protocol/plan.json'
RUN = 'runs/chess-pin-native-cost-coverage-v1/execution'
AUDIT = 'evidence/chess-pin-native-cost-coverage-v1/audit/receipt.json'
CODE = ['src/openjev/research/chess_native_feature_cache.py', 'tests/test_chess_native_feature_cache.py',
        'scripts/chess_native_feature_preflight.py']
PROTOCOL = {
    'version': native.VERSION,
    'scope': 'New feature-generation contract for future quality studies; old frozen caches, studies and failed gates remain unchanged.',
    'known_issue': '128-root native cost extension failed1e-5 score tolerance on23 distinct cases, including base. No changed choices. Single-position versus batch128 backbone features differ in float32.',
    'change': 'Build each frozen backbone feature row using one board and its own unpadded legal menu. Save native action features before node layout changes. Compact candidate storage; minibatches gather these precomputed inputs without recomputing action features.',
    'unchanged': 'Native decision function, backbone weights, all seven fresh head factories, head seeds211/223/227, float32 precision,128 old training roots, legal menus and1e-5 maximum absolute score criterion.',
    'backbone_seed': 97, 'roots': 128, 'root_seed_method_checks': 2688,
    'candidate_score_checks': 82320, 'absolute_score_tolerance': 1e-5,
    'new_gate': 'Every cached/native score vector meets original1e-5 tolerance and exact choice equality. All native predictions must reproduce the prior saved native predictions exactly. A new passing cache contract does not retroactively pass an old gate.',
    'reference': 'Audit reconstructs the cache with a separate explicit loop and packs batches with pad_sequence, sharing board encoders/backbone/action-feature kernels. Independent pin detector and reference factor/count pooling as before. Exact cache and record replay required.',
    'batching': 'Eight consecutive16-root head batches, identical to prior coverage screen. Native calls remain single-position. Only cached feature generation and packing change.',
    'runtime': 'CPU2 deterministic, shared host,900 seconds per execution/audit.', 'time_cap_seconds': 900,
    'limits': 'No training, teacher outcomes, games, new positions, timing advantage, all-backbone verification or architecture novelty. Future fits and baselines must share this versioned cache contract; prior fitted outcomes cannot be silently relabelled.',
}


def signature():
    frozen = original.read(ROOT/PLAN)
    if frozen != prior.signature(): raise ValueError('Frozen coverage signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or audit['numerical_gate_passed'] is not False
            or audit['native_decisions_replayed'] != 2688 or audit['failed_distinct_cases'] != 23
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['replays_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'replays.json')):
        raise ValueError('Prior negative audit changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'prior_signature': frozen, 'prior_audit_sha256': source.file_hash(ROOT/AUDIT)}


@torch.no_grad()
def reference_cache(model, rows):
    states, features, scores, moves, menus, fens, lengths = [], [], [], [], [], [], []
    for row in rows:
        board = chess.Board(row['fen']); names, encoded = source.prior.encode_candidates(board)
        c = torch.from_numpy(encoded).unsqueeze(0); mask = torch.ones(c.shape[:2], dtype=torch.bool)
        logits, _, hidden = model(torch.from_numpy(source.prior.encode_board(board)).unsqueeze(0), c, mask)
        nodes = hidden.reshape(1, 32, 64).permute(0, 2, 1)
        states.append(nodes.squeeze(0)); features.append(source.prior.candidate_features(model, nodes, c).squeeze(0))
        scores.append(logits.squeeze(0)); moves.append(c.squeeze(0)); menus.append(tuple(names))
        fens.append(board.fen(en_passant='fen')); lengths.append(len(names))
    return {'version': native.VERSION, 'nodes': torch.stack(states), 'action_features': torch.cat(features),
            'base_logits': torch.cat(scores), 'candidates': torch.cat(moves),
            'offsets': torch.cat((torch.tensor([0]), torch.tensor(lengths).cumsum(0))),
            'menus': tuple(menus), 'fens': tuple(fens)}


def reference_arguments(cache, indices, graph):
    chunks = [slice(int(cache['offsets'][i]), int(cache['offsets'][i+1])) for i in indices]
    lengths = [s.stop-s.start for s in chunks]
    actions = pad_sequence([cache['action_features'][s] for s in chunks], batch_first=True)
    candidates = pad_sequence([cache['candidates'][s] for s in chunks], batch_first=True)
    base = pad_sequence([cache['base_logits'][s] for s in chunks], batch_first=True, padding_value=-torch.inf)
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    if (not torch.equal(mask, graph['mask']) or [cache['menus'][i] for i in indices] != list(graph['menus'])
            or [cache['fens'][i] for i in indices] != [chess.Board(f).fen(en_passant='fen') for f in graph['fens']]):
        raise AssertionError('Reference graph identity mismatch')
    return [torch.stack([cache['nodes'][i] for i in indices]), actions, candidates, mask, base,
            graph['root'], graph['children']]


def compare_cache(left, right):
    if set(left) != set(right): raise AssertionError('Cache key mismatch')
    for key in ('version', 'menus', 'fens'):
        if left[key] != right[key]: raise AssertionError('Cache metadata mismatch')
    return original.pins.compare_states({k: v for k, v in left.items() if torch.is_tensor(v)},
                                        {k: v for k, v in right.items() if torch.is_tensor(v)}, 0)


@torch.no_grad()
def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen native-feature signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        model, data, old_features, graphs, rows, setup = original.pins.parent.setup()
        cache = reference_cache(model, rows) if audit else native.build(model, [chess.Board(r['fen']) for r in rows])
        drift = {'node_max_error': float((cache['nodes']-old_features['nodes']).abs().max()),
                 'base_max_error': float((cache['base_logits']-old_features['base_logits'][data['mask']]).abs().max())}
        if audit:
            compare_cache(torch.load(execution/'cache.pt', weights_only=True, map_location='cpu'), cache)
        prior_native = {}
        for row in source.prior.rows(ROOT/RUN/'timings.jsonl'):
            key = row['seed'], row['root_index'], row['method']
            if key in prior_native and prior_native[key] != row['prediction']:
                raise AssertionError('Prior repeated native predictions differ')
            prior_native[key] = row['prediction']
        if len(prior_native) != 2688: raise AssertionError('Incomplete prior native predictions')
        records = []
        with (out/'records.jsonl').open('x') as stream:
            for seed in original.SEEDS:
                loaded = original.heads(seed)
                initial = {m: {k: v.clone() for k, v in h.state_dict().items()} for m, h in loaded.items() if h is not None}
                for start in range(0, 128, 16):
                    original.deadline(begin); index = torch.arange(start, start+16); graph = graphs.batch(index)
                    args = reference_arguments(cache, index.tolist(), graph) if audit else native.arguments(cache, index, graph)
                    factor_rows = [original.candidate_factors(chess.Board(rows[i]['fen']), reference=True) for i in index]
                    factors = original.pack_factors(factor_rows)
                    if [tuple(c['uci'] for c in row['candidates']) for row in factor_rows] != graph['menus']:
                        raise AssertionError('Reference factor menu mismatch')
                    for method in original.METHODS:
                        logits = args[4] if method == 'base' else (loaded[method](*args) if method == 'wldn'
                                 else loaded[method](*args, factors, reference=True))
                        for slot, i in enumerate(index.tolist()):
                            names = graph['menus'][slot]
                            cached = original.result(names, logits[slot:slot+1, :len(names)])
                            current = original.decision(model, loaded[method], method, rows[i]['fen'])
                            record = {'seed': seed, 'root_index': i, 'id': rows[i]['id'], 'method': method,
                                      **prior.prior.compare(current, cached),
                                      'native_prediction_unchanged': current == prior_native[seed, i, method],
                                      'cached': cached, 'native': current}
                            records.append(record); stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                for method, state in initial.items(): original.pins.compare_states(state, loaded[method].state_dict(), 0)
                print(json.dumps({'head_seed': seed, 'audit': audit}), flush=True)
        keys = {(r['seed'], r['root_index'], r['method']) for r in records}
        checks = sum(len(r['native']['scores']) for r in records)
        if len(keys) != len(records) or len(keys) != 2688 or checks != 82320:
            raise AssertionError('Incomplete native-feature verification')
        failed = [r for r in records if not (r['score_tolerance_passed'] and r['choice_matches'] and r['native_prediction_unchanged'])]
        stats = {'numerical_gate_passed': not failed, 'failed_distinct_cases': len(failed),
                 'root_seed_method_checks': len(records), 'candidate_score_checks': checks,
                 'max_cached_score_error': max(r['cached_score_max_error'] for r in records),
                 'choice_changes': sum(not r['choice_matches'] for r in records),
                 'native_prediction_changes': sum(not r['native_prediction_unchanged'] for r in records),
                 'old_to_new_feature_drift': drift}
        original.deadline(begin)
        if audit:
            old = original.read(execution/'summary.json')
            if (source.prior.rows(execution/'records.jsonl') != records or any(old[k] != v for k, v in stats.items())
                    or old['status'] != 'completed' or old['plan_sha256'] != plan_hash or not 0 < old['wall_seconds'] <= 900):
                raise AssertionError('Saved native-feature evidence differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash, 'summary_sha256': source.file_hash(execution/'summary.json'),
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'reference_records_sha256': source.file_hash(out/'records.jsonl'), 'cache_exact': True,
                      **stats, 'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            source.prior.save(out/'cache.pt', cache)
            result = {'status': 'completed', 'plan_sha256': plan_hash, **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
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
