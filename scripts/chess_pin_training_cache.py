# SPDX-License-Identifier: GPL-3.0-only
"""Full training absolute-pin inputs and independent detector replay."""
import argparse
import json
from pathlib import Path
import time

import chess
import torch

import chess_native_training_cache as features
from openjev.research.chess_pin_cache import VERSION, pack, validate

ROOT, source, original = features.ROOT, features.source, features.original
PLAN = 'evidence/chess-native-training-cache-v1/protocol/plan.json'
RUN = 'runs/chess-native-training-cache-v1/execution'
AUDIT = 'evidence/chess-native-training-cache-v1/audit/receipt.json'
CODE = ['src/openjev/research/chess_pin_cache.py', 'scripts/chess_pin_training_cache.py', 'tests/test_chess_pin_cache.py']
PROTOCOL = {
    'version': VERSION, 'roots': 32768, 'candidates': 968036, 'block_roots': 128, 'blocks': 256,
    'scope': 'All original training roots and legal candidates; pin inputs only. No dev/shift data, teacher outcomes, optimization or new positions.',
    'primary': 'Existing frozen ray-scan candidate_factors, packed into local candidate indices with separate root-to-candidate and root-to-factor offsets.',
    'reference': 'Independently detect every root/child pin and status using frozen bitboard/occupancy reference; explicitly pack each factor without the primary packer. Every tensor, menu, FEN, offset and coverage statistic must reproduce exactly.',
    'alignment': 'Menus/FENs must match the authenticated native child-graph cache and canonical feature-cache source rows. Common pin blocks are shared by all three backbone feature versions. Caller boards/history remain unchanged.',
    'coverage': 'Report full-set witness coverage and role counts, without a quality or coverage acceptance threshold. Prior mirror checks remain their original bounded evidence; this full run adds no mirror claim.',
    'runtime': 'CPU2 deterministic,128-root blocks,900 seconds per execution/audit.', 'time_cap_seconds': 900,
    'limits': 'Absolute-pin witnesses are established rule-derived input, not learned legality, game search or generic novelty. No trained-head result, full native-score test or architecture-acceptance gate. Historical evidence and active quality study remain unchanged.',
}


def signature():
    plan = original.read(ROOT/PLAN)
    if plan != features.signature(): raise ValueError('Frozen training-feature signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or not audit['all_cache_tensors_and_metadata_exact'] or audit['blocks'] != 768
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_blocks_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'blocks.jsonl')):
        raise ValueError('Training feature audit changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'feature_signature': plan, 'feature_audit_sha256': source.file_hash(ROOT/AUDIT)}


def reference_pack(records, fens):
    values = []; candidates = [0]; offsets = [0]; menus = []
    for row in records:
        menu = []
        for child in row['candidates']:
            index = candidates[-1]+len(menu)
            for owner, king, blocker, attacker, status in child['factors']:
                values.append([index, owner, king, blocker, attacker, status])
            menu.append(child['uci'])
        candidates.append(candidates[-1]+len(menu)); offsets.append(len(values)); menus.append(tuple(menu))
    return {'version': VERSION, 'factors': torch.tensor(values, dtype=torch.long).reshape(-1, 6),
            'candidate_offsets': torch.tensor(candidates), 'factor_offsets': torch.tensor(offsets),
            'menus': tuple(menus), 'fens': tuple(fens)}


def compare(left, right):
    if set(left) != set(right): raise AssertionError('Pin-cache fields differ')
    for key in ('version', 'menus', 'fens'):
        if left[key] != right[key]: raise AssertionError('Pin-cache metadata differs')
    original.pins.compare_states({k: v for k, v in left.items() if torch.is_tensor(v)},
                                  {k: v for k, v in right.items() if torch.is_tensor(v)}, 0)


def summarize(blocks):
    if [(b['start'], b['stop']) for b in blocks] != [(i, i+128) for i in range(0, 32768, 128)]:
        raise AssertionError('Incomplete pin input coverage')
    coverage = {}
    for key in blocks[0]['coverage']:
        if key == 'witness_roles_by_owner_retained_added_removed':
            coverage[key] = [[sum(b['coverage'][key][o][s] for b in blocks) for s in range(3)] for o in range(2)]
        else: coverage[key] = sum(b['coverage'][key] for b in blocks)
    if coverage['roots'] != 32768 or coverage['candidates'] != 968036:
        raise AssertionError('Wrong root or candidate membership')
    return {'blocks': len(blocks), 'factor_rows': sum(b['factor_rows'] for b in blocks), 'coverage': coverage}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen pin-training input signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        rows = source.prior.rows(ROOT/RUN/'rows.jsonl')
        graphs = source.ChildGraphCache(ROOT/source.CACHE/'train')
        if len(rows) != 32768 or any(r['index'] != i or r['fen'] != graphs.records[i]['fen'] for i, r in enumerate(rows)):
            raise AssertionError('Feature/pin source identity differs')
        blocks = []
        if not audit: (out/'blocks').mkdir()
        with (out/'blocks.jsonl').open('x') as stream:
            for start in range(0, 32768, 128):
                original.deadline(begin); stop = start+128; records, fens = [], []
                for i in range(start, stop):
                    board = chess.Board(rows[i]['fen']); before = board.fen(en_passant='fen'), list(board.move_stack)
                    record = original.candidate_factors(board, reference=audit)
                    if [r['uci'] for r in record['candidates']] != graphs.records[i]['moves']:
                        raise AssertionError('Pin/native graph menus differ')
                    if before != (board.fen(en_passant='fen'), list(board.move_stack)):
                        raise AssertionError('Pin extraction changed caller board')
                    records.append(record); fens.append(before[0])
                cache = validate(reference_pack(records, fens) if audit else pack(records, fens))
                relative = f'blocks/{start:05d}.pt'
                if audit:
                    saved = validate(torch.load(execution/relative, weights_only=True, map_location='cpu'))
                    compare(saved, cache); digest = source.file_hash(execution/relative)
                else:
                    source.prior.save(out/relative, cache); digest = source.file_hash(out/relative)
                feature = torch.load(ROOT/RUN/f'97/{start:05d}.pt', weights_only=True, map_location='cpu')
                if (feature['menus'] != cache['menus'] or feature['fens'] != cache['fens']
                        or not torch.equal(feature['offsets'], cache['candidate_offsets'])):
                    raise AssertionError('Canonical feature/pin packing mismatch')
                item = {'start': start, 'stop': stop, 'path': relative, 'sha256': digest,
                        'factor_rows': len(cache['factors']), 'coverage': original.pins.inputs.summarize(records)}
                blocks.append(item); stream.write(json.dumps(item, allow_nan=False)+'\n'); stream.flush()
                if stop % 2048 == 0: print(json.dumps({'roots_completed': stop, 'audit': audit}), flush=True)
        original.deadline(begin); stats = summarize(blocks)
        if audit:
            old = original.read(execution/'summary.json')
            if (source.prior.rows(execution/'blocks.jsonl') != blocks or any(old[k] != v for k, v in stats.items())
                    or old['status'] != 'completed' or old['plan_sha256'] != plan_hash
                    or not 0 < old['wall_seconds'] <= 900 or old['limits'] != PROTOCOL['limits']):
                raise AssertionError('Saved pin-cache evidence differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'reference_blocks_sha256': source.file_hash(out/'blocks.jsonl'),
                      'all_factors_and_metadata_exact': True, **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            result = {'status': 'completed', 'plan_sha256': plan_hash, **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'summary.json', result)
            write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                  'files': {str(p.relative_to(out)): source.file_hash(p) for p in out.rglob('*') if p.is_file()}})
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
