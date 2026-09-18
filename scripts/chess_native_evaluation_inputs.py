# SPDX-License-Identifier: GPL-3.0-only
"""Full old-panel native features, pin factors and input alignment, without quality evaluation."""
import argparse
import json
from pathlib import Path
import time

import chess
import torch

import chess_training_input_alignment as parent

ROOT, source, original = parent.ROOT, parent.source, parent.original
training = parent.parent.features
pins = parent.parent
PLAN = 'evidence/chess-training-input-alignment-v1/protocol/plan.json'
RUN = 'runs/chess-training-input-alignment-v1/execution'
AUDIT = 'evidence/chess-training-input-alignment-v1/audit/receipt.json'
SPLITS = {'dev': 60541, 'shift': 58946}
SEEDS = parent.SEEDS
CODE = ['scripts/chess_native_evaluation_inputs.py', 'tests/test_chess_native_evaluation_inputs.py']
PROTOCOL = {
    'version': 'native-canonical-old-evaluation-inputs-v1',
    'panels': {'dev': {'roots': 2048, 'candidates': 60541}, 'shift': {'roots': 2048, 'candidates': 58946}},
    'backbone_seeds': list(SEEDS), 'block_roots': 128,
    'feature_blocks': 96, 'common_pin_blocks': 32, 'alignment_batches': 192,
    'feature_contract': training.native.VERSION, 'pin_contract': pins.VERSION,
    'loader_contract': parent.VERSION, 'shuffle_seed': 862019,
    'scope': 'Both complete original development panels, already exposed in prior architecture studies. Source files are authenticated; only index,id,FEN retained. No teacher label/value use, fit, quality prediction comparison, new positions or independent confirmation.',
    'construction': 'One position and its unpadded legal menu per frozen backbone call; frozen native feature builder. Common pins use the ray detector and frozen packer. All block menus/FENs/candidate offsets must match authenticated native graph metadata; backbone weights and caller boards/history unchanged.',
    'alignment': 'For each split and backbone, select every root once in source order and once in a seeded permutation, using the frozen merged TrainingInputs loader. Save every batch digest of identities, features, pin rows and native graphs.',
    'reference': 'Reconstruct every feature block with the separate single-position loop, sharing neural/encoding kernels. Reconstruct every pin block with the separate bitboard/occupancy detector and explicit packer. Compare all saved tensor/metadata fields exactly. Replay every ordered/shuffled batch digest from original blocks without merging, using independent lookup/padding/factor translation; native graph decoder shared.',
    'runtime': 'CPU2 deterministic;900 seconds per execution/audit; existing quality run left unchanged.',
    'time_cap_seconds': 900,
    'limits': 'Input integrity and batching only. Not full-panel native head-score equivalence, trained architecture quality, new data, game strength, efficiency or novelty. Historical studies retain their actual input contract. Future matched fits require a separately frozen quality protocol.',
}


def signature():
    plan = original.read(ROOT/PLAN)
    if plan != parent.signature(): raise ValueError('Frozen training-loader signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or not audit['all_shuffled_inputs_exact'] or audit['batches'] != 768
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_batches_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'batches.jsonl')):
        raise ValueError('Training-loader audit changed')
    manifests = {split: original.read(ROOT/source.CACHE/split/'manifest.json') for split in SPLITS}
    if any(m['roots'] != 2048 or m['children'] != SPLITS[s] for s, m in manifests.items()):
        raise ValueError('Old panel membership changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'training_loader_signature': plan, 'training_loader_audit_sha256': source.file_hash(ROOT/AUDIT),
            'panel_rows_sha256': {s: source.file_hash(ROOT/source.prior.DATA[s]) for s in SPLITS},
            'panel_graph_manifests_sha256': {s: source.file_hash(ROOT/source.CACHE/s/'manifest.json') for s in SPLITS}}


def metadata_rows(raw, graphs):
    rows = [{'index': i, 'id': r['id'], 'fen': r['fen']} for i, r in enumerate(raw)]
    if (not rows or len(rows) != len(graphs) or len({r['id'] for r in rows}) != len(rows)
            or any(r['index'] != g['index'] or r['fen'] != g['fen'] for r, g in zip(rows, graphs))):
        raise AssertionError('Source row identity differs from native graphs')
    return rows


def coverage_sum(blocks):
    result = {}
    for key in blocks[0]['coverage']:
        if key == 'witness_roles_by_owner_retained_added_removed':
            result[key] = [[sum(b['coverage'][key][o][s] for b in blocks) for s in range(3)] for o in range(2)]
        else: result[key] = sum(b['coverage'][key] for b in blocks)
    return result


def summarize(features, pin_blocks, batches):
    expected_features = [(split, seed, start, start+128) for split in SPLITS for seed in SEEDS for start in range(0, 2048, 128)]
    expected_pins = [(split, start, start+128) for split in SPLITS for start in range(0, 2048, 128)]
    expected_batches = [(split, seed, order, start) for split in SPLITS for seed in SEEDS
                        for order in ('source', 'shuffled') for start in range(0, 2048, 128)]
    if ([(r['split'], r['backbone_seed'], r['start'], r['stop']) for r in features] != expected_features
            or [(r['split'], r['start'], r['stop']) for r in pin_blocks] != expected_pins
            or [(r['split'], r['backbone_seed'], r['order'], r['start']) for r in batches] != expected_batches):
        raise AssertionError('Incomplete or misordered evaluation inputs')
    by_split = {}
    for split, candidates in SPLITS.items():
        selected = [r for r in pin_blocks if r['split'] == split]
        coverage = coverage_sum(selected); factors = sum(r['factor_rows'] for r in selected)
        if (coverage['roots'] != 2048 or coverage['candidates'] != candidates
                or sum(sum(row) for row in coverage['witness_roles_by_owner_retained_added_removed']) != factors):
            raise AssertionError('Invalid evaluation pin coverage')
        byte_counts = {}
        for seed in SEEDS:
            fs = [r for r in features if (r['split'], r['backbone_seed']) == (split, seed)]
            if (sum(r['roots'] for r in fs) != 2048 or sum(r['candidates'] for r in fs) != candidates
                    or any(r['roots'] != 128 or r['tensor_bytes'] <= 0 for r in fs)):
                raise AssertionError('Invalid evaluation feature coverage')
            byte_counts[str(seed)] = sum(r['tensor_bytes'] for r in fs)
            for order in ('source', 'shuffled'):
                bs = [r for r in batches if (r['split'], r['backbone_seed'], r['order']) == (split, seed, order)]
                if (any(r['roots'] != 128 for r in bs) or sum(r['candidates'] for r in bs) != candidates
                        or sum(r['factor_rows'] for r in bs) != factors):
                    raise AssertionError('Invalid evaluation loader coverage')
        by_split[split] = {'coverage': coverage, 'factor_rows': factors, 'feature_tensor_bytes_by_backbone': byte_counts}
    return {'feature_blocks': len(features), 'common_pin_blocks': len(pin_blocks),
            'alignment_batches': len(batches), 'by_split': by_split}


def persist_block(cache, relative, out, execution, compare):
    if execution is None:
        path = out/relative; path.parent.mkdir(parents=True, exist_ok=True); source.prior.save(path, cache)
    else:
        path = execution/relative; compare(torch.load(path, weights_only=True, map_location='cpu'), cache)
    return source.file_hash(path)


@torch.no_grad()
def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen evaluation-input signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        feature_records, pin_records, batch_records = [], [], []
        with (out/'features.jsonl').open('x') as feature_stream, (out/'pins.jsonl').open('x') as pin_stream, (out/'batches.jsonl').open('x') as batch_stream:
            for split in SPLITS:
                graphs = source.ChildGraphCache(ROOT/source.CACHE/split)
                rows = metadata_rows(source.prior.rows(ROOT/source.prior.DATA[split]), graphs.records)
                if len(rows) != 2048: raise AssertionError('Wrong panel size')
                if audit and source.prior.rows(execution/f'{split}-rows.jsonl') != rows: raise AssertionError('Saved source rows differ')
                if not audit:
                    with (out/f'{split}-rows.jsonl').open('x') as stream:
                        for row in rows: stream.write(json.dumps(row, allow_nan=False)+'\n')
                common = []
                for start in range(0, 2048, 128):
                    original.deadline(begin); records, fens = [], []
                    for i in range(start, start+128):
                        board = chess.Board(rows[i]['fen']); before = board.fen(en_passant='fen'), list(board.move_stack)
                        record = original.candidate_factors(board, reference=audit)
                        if before != (board.fen(en_passant='fen'), list(board.move_stack)):
                            raise AssertionError('Caller board changed')
                        if [c['uci'] for c in record['candidates']] != graphs.records[i]['moves']:
                            raise AssertionError('Pin and native graph menus differ')
                        records.append(record); fens.append(before[0])
                    block = pins.validate(pins.reference_pack(records, fens) if audit else pins.pack(records, fens))
                    relative = f'{split}/pins/{start:05d}.pt'
                    digest = persist_block(block, relative, out, execution, pins.compare)
                    item = {'split': split, 'start': start, 'stop': start+128, 'path': relative, 'sha256': digest,
                            'factor_rows': len(block['factors']), 'coverage': original.pins.inputs.summarize(records)}
                    pin_records.append(item); pin_stream.write(json.dumps(item, allow_nan=False)+'\n'); pin_stream.flush(); common.append(block)
                for seed in SEEDS:
                    model = source.prior.backbone(seed, original.read(ROOT/source.PARENT))
                    initial = {k: v.clone() for k, v in model.state_dict().items()}; blocks = []
                    for start in range(0, 2048, 128):
                        original.deadline(begin); selected = rows[start:start+128]
                        block = (training.prior.parent.reference_cache(model, selected) if audit
                                 else training.native.build(model, [chess.Board(r['fen']) for r in selected]))
                        stats = training.block_summary(block, graphs.records[start:start+128])
                        pin = common[start//128]
                        if (block['menus'] != pin['menus'] or block['fens'] != pin['fens']
                                or not torch.equal(block['offsets'], pin['candidate_offsets'])):
                            raise AssertionError('Feature/pin identity differs')
                        relative = f'{split}/{seed}/{start:05d}.pt'
                        digest = persist_block(block, relative, out, execution, training.prior.parent.compare_cache)
                        item = {'split': split, 'backbone_seed': seed, 'start': start, 'stop': start+128,
                                'path': relative, 'sha256': digest, **stats}
                        feature_records.append(item); feature_stream.write(json.dumps(item, allow_nan=False)+'\n'); feature_stream.flush(); blocks.append(block)
                    original.pins.compare_states(initial, model.state_dict(), 0)
                    data = None if audit else parent.TrainingInputs(blocks, common)
                    orders = {'source': torch.arange(2048), 'shuffled': torch.randperm(2048, generator=torch.Generator().manual_seed(PROTOCOL['shuffle_seed']))}
                    for order_name, order in orders.items():
                        if sorted(order.tolist()) != list(range(2048)): raise AssertionError('Invalid panel order')
                        for start in range(0, 2048, 128):
                            original.deadline(begin); indices = order[start:start+128]; graph = graphs.batch(indices)
                            args, factors = (parent.reference_batch(blocks, common, indices.tolist(), graph) if audit else data.batch(indices, graph))
                            item = {'split': split, 'backbone_seed': seed, 'order': order_name, 'start': start,
                                    'roots': len(indices), 'candidates': int(args[3].sum()), 'factor_rows': len(factors),
                                    'sha256': parent.digest_batch(indices.tolist(), graph, args, factors)}
                            batch_records.append(item); batch_stream.write(json.dumps(item, allow_nan=False)+'\n'); batch_stream.flush()
                    print(json.dumps({'split': split, 'backbone_seed': seed, 'completed': True, 'audit': audit}), flush=True)
                    del data, blocks, model
        original.deadline(begin); stats = summarize(feature_records, pin_records, batch_records)
        if audit:
            old = original.read(execution/'summary.json')
            for name, records in (('features', feature_records), ('pins', pin_records), ('batches', batch_records)):
                if source.prior.rows(execution/f'{name}.jsonl') != records: raise AssertionError('Saved evaluation input records differ')
            if (any(old[k] != v for k, v in stats.items()) or old['status'] != 'completed'
                    or old['plan_sha256'] != plan_hash or old['limits'] != PROTOCOL['limits']
                    or not 0 < old['wall_seconds'] <= 900): raise AssertionError('Saved evaluation input summary differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'reference_records_sha256': {name: source.file_hash(out/f'{name}.jsonl') for name in ('features', 'pins', 'batches')},
                      'all_features_pins_and_batch_digests_exact': True, **stats,
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
