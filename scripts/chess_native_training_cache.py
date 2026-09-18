# SPDX-License-Identifier: GPL-3.0-only
"""Full training-only native feature cache, with bounded block reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import chess
import torch

import chess_native_feature_multiseed as prior

ROOT, source, original, native = prior.ROOT, prior.source, prior.original, prior.native
PLAN = 'evidence/chess-native-feature-multiseed-v1/protocol/plan.json'
RUN = 'runs/chess-native-feature-multiseed-v1/execution'
AUDIT = 'evidence/chess-native-feature-multiseed-v1/audit/receipt.json'
CODE = ['scripts/chess_native_training_cache.py', 'tests/test_chess_native_training_cache.py']
PROTOCOL = {
    'version': 'native-single-position-full-training-cache-v1',
    'scope': 'Full original training set only; no dev/shift feature construction, quality outcomes, optimization or new positions.',
    'backbone_seeds': list(prior.SEEDS), 'roots_per_backbone': 32768,
    'candidates_per_backbone': 968036, 'block_roots': 128, 'blocks_per_backbone': 256,
    'feature_contract': native.VERSION,
    'construction': 'Frozen native builder, one board and its unpadded legal menu per forward. Store compact nodes/action features/base logits/candidates/offsets/menus/FENs in256 blocks per backbone. Source metadata keeps only index,id,FEN; no target/value fields.',
    'alignment': 'Every source index/FEN must match the authenticated native child-graph cache. Every rebuilt legal menu and compact offset must match that cache. First block for each seed must exactly match its previously audited engineering cache.',
    'audit': 'Authenticate every primary file, independently reconstruct all768 blocks using the frozen explicit reference loop, and compare every tensor and metadata field exactly. Check all block membership, coverage counts, tensor byte counts, menu digests and checkpoint immutability. Shared neural/encoding kernels; no sampling.',
    'runtime': 'CPU2 deterministic;128-root working blocks; shared host with existing quality study.',
    'time_cap_seconds': 900,
    'limits': 'Feature construction and integrity only. This does not test trained head quality, full-set head score equivalence, fresh confirmation, pin-factor extraction on all roots, speed advantage or novelty. Historical caches and live fits remain unchanged. Future matched fits require the same new input contract and separately frozen quality protocol.',
}


def signature():
    plan = original.read(ROOT/PLAN)
    if plan != prior.signature(): raise ValueError('Frozen three-backbone signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    graph_manifest = original.read(ROOT/source.CACHE/'train/manifest.json')
    if (audit['status'] != 'passed' or not audit['numerical_gate_passed'] or audit['caches_exact'] != 3
            or audit['candidate_score_checks'] != 246960 or audit['root_backbone_head_method_checks'] != 8064
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_records_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'records.jsonl')
            or graph_manifest['roots'] != 32768 or graph_manifest['children'] != 968036):
        raise ValueError('Three-backbone audit or training source identity changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'parent_signature': plan, 'parent_audit_sha256': source.file_hash(ROOT/AUDIT),
            'training_rows_sha256': source.file_hash(ROOT/source.prior.DATA['train']),
            'native_training_manifest_sha256': source.file_hash(ROOT/source.CACHE/'train/manifest.json')}


def block_summary(cache, metadata):
    expected_keys = {'version', 'nodes', 'action_features', 'base_logits', 'candidates', 'offsets', 'menus', 'fens'}
    if set(cache) != expected_keys or cache['version'] != native.VERSION or not metadata:
        raise AssertionError('Invalid native feature block')
    count = len(metadata); menus = tuple(tuple(r['moves']) for r in metadata)
    lengths = [len(m) for m in menus]; candidates = sum(lengths)
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    fens = tuple(chess.Board(r['fen']).fen(en_passant='fen') for r in metadata)
    if cache['menus'] != menus or cache['fens'] != fens or not torch.equal(cache['offsets'], offsets):
        raise AssertionError('Native feature menu/FEN/offset mismatch')
    shapes = {'nodes': (count, 64, 32), 'action_features': (candidates, 120),
              'base_logits': (candidates,), 'candidates': (candidates, 5), 'offsets': (count+1,)}
    for key, shape in shapes.items():
        value = cache[key]; dtype = torch.long if key in ('candidates', 'offsets') else torch.float32
        if value.shape != shape or value.dtype != dtype or value.device.type != 'cpu' or value.requires_grad:
            raise AssertionError('Native feature shape/dtype/device differs')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError('Nonfinite native feature')
    return {'roots': count, 'candidates': candidates,
            'tensor_bytes': sum(cache[k].numel()*cache[k].element_size() for k in shapes),
            'menus_sha256': hashlib.sha256(json.dumps(menus, separators=(',', ':')).encode()).hexdigest()}


def expected_membership():
    return [(seed, start, start+128) for seed in prior.SEEDS for start in range(0, 32768, 128)]


def summarize(records):
    if [(r['backbone_seed'], r['start'], r['stop']) for r in records] != expected_membership():
        raise AssertionError('Incomplete or misordered cache blocks')
    by_seed = {}
    for seed in prior.SEEDS:
        rows = [r for r in records if r['backbone_seed'] == seed]
        totals = {k: sum(r[k] for r in rows) for k in ('roots', 'candidates', 'tensor_bytes')}
        if (totals['roots'] != 32768 or totals['candidates'] != 968036
                or any(r['roots'] != 128 or r['candidates'] < 128 or r['tensor_bytes'] <= 0 for r in rows)):
            raise AssertionError('Full training coverage differs')
        by_seed[str(seed)] = totals
    return {'backbone_caches': 3, 'blocks': len(records), 'by_backbone': by_seed,
            'total_root_feature_rows': sum(r['roots'] for r in records),
            'total_candidate_feature_rows': sum(r['candidates'] for r in records),
            'total_tensor_bytes': sum(r['tensor_bytes'] for r in records)}


@torch.no_grad()
def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen full-training cache signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        rows = [{'index': i, 'id': r['id'], 'fen': r['fen']} for i, r in enumerate(source.prior.rows(ROOT/source.prior.DATA['train']))]
        graph = source.ChildGraphCache(ROOT/source.CACHE/'train')
        if (len(rows) != 32768 or len(graph.records) != 32768 or len({r['id'] for r in rows}) != 32768
                or any(r['index'] != g['index'] or r['fen'] != g['fen'] for r, g in zip(rows, graph.records))):
            raise AssertionError('Original training rows and graph source differ')
        if audit:
            if source.prior.rows(execution/'rows.jsonl') != rows: raise AssertionError('Saved source metadata differs')
        else:
            with (out/'rows.jsonl').open('x') as stream:
                for r in rows: stream.write(json.dumps(r, allow_nan=False)+'\n')
        old_plan = original.read(ROOT/source.PARENT); records = []
        for seed in prior.SEEDS:
            original.deadline(begin); model = source.prior.backbone(seed, old_plan)
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            if not audit: (out/str(seed)).mkdir()
            with (out/'blocks.jsonl').open('a' if records else 'x') as stream:
                for start in range(0, 32768, 128):
                    original.deadline(begin); stop = start+128; selected = rows[start:stop]
                    cache = prior.parent.reference_cache(model, selected) if audit else native.build(model, [chess.Board(r['fen']) for r in selected])
                    stats = block_summary(cache, graph.records[start:stop])
                    relative = f'{seed}/{start:05d}.pt'
                    if audit:
                        saved = torch.load(execution/relative, weights_only=True, map_location='cpu')
                        prior.parent.compare_cache(saved, cache)
                        if block_summary(saved, graph.records[start:stop]) != stats: raise AssertionError('Saved block summary differs')
                        digest = source.file_hash(execution/relative)
                    else:
                        source.prior.save(out/relative, cache); digest = source.file_hash(out/relative)
                    if start == 0:
                        pilot = torch.load(ROOT/RUN/f'cache-{seed}.pt', weights_only=True, map_location='cpu')
                        prior.parent.compare_cache(pilot, cache)
                    record = {'backbone_seed': seed, 'start': start, 'stop': stop, 'path': relative,
                              'sha256': digest, **stats}
                    records.append(record); stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                    if stop % 2048 == 0:
                        print(json.dumps({'backbone_seed': seed, 'roots_completed': stop, 'audit': audit}), flush=True)
            original.pins.compare_states(initial, model.state_dict(), 0)
        original.deadline(begin); stats = summarize(records)
        if audit:
            old = original.read(execution/'summary.json')
            if (source.prior.rows(execution/'blocks.jsonl') != records or any(old[k] != v for k, v in stats.items())
                    or old['status'] != 'completed' or old['plan_sha256'] != plan_hash
                    or not 0 < old['wall_seconds'] <= 900 or old['limits'] != PROTOCOL['limits']):
                raise AssertionError('Full training cache summary differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'reference_blocks_sha256': source.file_hash(out/'blocks.jsonl'),
                      'all_cache_tensors_and_metadata_exact': True, 'prior_pilot_caches_exact': 3,
                      **stats, 'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            result = {'status': 'completed', 'plan_sha256': plan_hash, 'prior_pilot_caches_exact': 3,
                      **stats, 'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
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
