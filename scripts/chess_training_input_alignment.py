# SPDX-License-Identifier: GPL-3.0-only
"""Exhaustive shuffled alignment of native features, pin factors and graphs."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import time

import chess
import torch
from torch.nn.utils.rnn import pad_sequence

import chess_pin_training_cache as parent
from openjev.research.chess_training_inputs import TrainingInputs, VERSION

ROOT, source, original = parent.ROOT, parent.source, parent.original
PLAN = 'evidence/chess-pin-training-cache-v1/protocol/plan.json'
RUN = 'runs/chess-pin-training-cache-v1/execution'
AUDIT = 'evidence/chess-pin-training-cache-v1/audit/receipt.json'
CODE = ['src/openjev/research/chess_training_inputs.py', 'tests/test_chess_training_inputs.py',
        'scripts/chess_training_input_alignment.py', 'tests/test_chess_training_input_alignment.py']
SEEDS = (97, 109, 127)
PROTOCOL = {
    'version': VERSION, 'roots_per_backbone': 32768, 'candidates_per_backbone': 968036,
    'backbone_seeds': list(SEEDS), 'batch_roots': 128, 'shuffle_seed': 862013,
    'scope': 'Every original training root once per backbone in one seeded shuffled pass. No labels, optimization, head scores, dev/shift data or new positions.',
    'primary': 'Join all authenticated canonical feature and common pin blocks. Select shuffled minibatches through native arguments and pin gather. Authenticate native graph cache and bind its selected complete menus/FENs/masks.',
    'reference': 'Read original blocks without joining them; locate each selected root independently, use pad_sequence for candidate tensors and explicit factor-row translation. Reproduce each saved batch digest exactly, including every feature, pin row, graph tensor and selected root identity. Graph decoding is shared with the previously audited native graph cache.',
    'coverage': 'All 32768 roots and968036 legal candidates exactly once for each backbone;768 batches total. Repeated roots, mismatched block boundaries and empty-factor batches are separate unit-test coverage.',
    'runtime': 'CPU2 deterministic,900-second cap per execution/audit; one backbone in memory at a time.',
    'limits': 'Input alignment only. Does not establish trained quality, full-set native head-score equivalence, novel architecture, game strength or computational advantage.',
    'time_cap_seconds': 900,
}


def signature():
    plan = original.read(ROOT/PLAN)
    if plan != parent.signature(): raise ValueError('Frozen pin-cache signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or not audit['all_factors_and_metadata_exact'] or audit['blocks'] != 256
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_blocks_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'blocks.jsonl')):
        raise ValueError('Full pin-input audit changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'pin_signature': plan, 'pin_audit_sha256': source.file_hash(ROOT/AUDIT)}


def locate(blocks, indices):
    boundaries = [0]
    for block in blocks: boundaries.append(boundaries[-1]+len(block['menus']))
    result = []
    for index in indices:
        if not 0 <= index < boundaries[-1]: raise ValueError('Invalid reference root index')
        block = bisect_right(boundaries, index)-1
        result.append((blocks[block], index-boundaries[block]))
    return result


def reference_batch(feature_blocks, pin_blocks, indices, graph):
    features = locate(feature_blocks, indices); pins = locate(pin_blocks, indices)
    nodes, actions, candidates, base, lengths, factors, menus, fens = [], [], [], [], [], [], [], []
    candidate = 0
    for (feature, i), (pin, j) in zip(features, pins):
        menu = feature['menus'][i]; fen = feature['fens'][i]
        if menu != pin['menus'][j] or fen != pin['fens'][j]: raise AssertionError('Reference identities differ')
        lo, hi = map(int, feature['offsets'][i:i+2]); a, b = map(int, pin['candidate_offsets'][j:j+2])
        if hi-lo != len(menu) or b-a != len(menu): raise AssertionError('Reference candidate offsets differ')
        nodes.append(feature['nodes'][i]); actions.append(feature['action_features'][lo:hi])
        candidates.append(feature['candidates'][lo:hi]); base.append(feature['base_logits'][lo:hi])
        lengths.append(len(menu)); menus.append(menu); fens.append(fen)
        start, stop = map(int, pin['factor_offsets'][j:j+2])
        for row in pin['factors'][start:stop].tolist():
            if not a <= row[0] < b: raise AssertionError('Reference factor root differs')
            factors.append([candidate+row[0]-a, *row[1:]])
        candidate += len(menu)
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    if (menus != graph['menus'] or fens != [chess.Board(f).fen(en_passant='fen') for f in graph['fens']]
            or not torch.equal(mask, graph['mask'])): raise AssertionError('Reference graph identities differ')
    args = [torch.stack(nodes), pad_sequence(actions, batch_first=True), pad_sequence(candidates, batch_first=True),
            mask, pad_sequence(base, batch_first=True, padding_value=-torch.inf), graph['root'], graph['children']]
    return args, torch.tensor(factors, dtype=torch.long).reshape(-1, 6)


def digest_batch(indices, graph, args, factors):
    digest = hashlib.sha256()
    metadata = {'indices': indices, 'menus': graph['menus'], 'fens': graph['fens']}
    digest.update(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
    for tensor in [*args, factors]:
        digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize(records):
    expected = [(seed, start) for seed in SEEDS for start in range(0, 32768, 128)]
    if [(r['backbone_seed'], r['start']) for r in records] != expected:
        raise AssertionError('Incomplete shuffled batch membership')
    by_seed = {}
    for seed in SEEDS:
        rows = [r for r in records if r['backbone_seed'] == seed]
        stats = {key: sum(r[key] for r in rows) for key in ('roots', 'candidates', 'factor_rows')}
        if stats != {'roots': 32768, 'candidates': 968036, 'factor_rows': 182331}:
            raise AssertionError('Wrong shuffled input coverage')
        by_seed[str(seed)] = stats
    return {'batches': len(records), 'by_backbone': by_seed, 'every_root_once_per_backbone': True}


@torch.no_grad()
def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen alignment signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        graphs = source.ChildGraphCache(ROOT/source.CACHE/'train')
        pin_blocks = [torch.load(ROOT/RUN/f'blocks/{start:05d}.pt', weights_only=True, map_location='cpu')
                      for start in range(0, 32768, 128)]
        order = torch.randperm(32768, generator=torch.Generator().manual_seed(PROTOCOL['shuffle_seed']))
        if sorted(order.tolist()) != list(range(32768)): raise AssertionError('Invalid shuffled membership')
        saved = source.prior.rows(execution/'batches.jsonl') if audit else None; records = []
        with (out/'batches.jsonl').open('x') as stream:
            for seed in SEEDS:
                blocks = [torch.load(ROOT/parent.RUN/f'{seed}/{start:05d}.pt', weights_only=True, map_location='cpu')
                          for start in range(0, 32768, 128)]
                data = None if audit else TrainingInputs(blocks, pin_blocks)
                for start in range(0, 32768, 128):
                    original.deadline(begin); indices = order[start:start+128]; graph = graphs.batch(indices)
                    args, factors = (reference_batch(blocks, pin_blocks, indices.tolist(), graph) if audit
                                     else data.batch(indices, graph))
                    item = {'backbone_seed': seed, 'start': start, 'roots': len(indices),
                            'candidates': int(args[3].sum()), 'factor_rows': len(factors),
                            'sha256': digest_batch(indices.tolist(), graph, args, factors)}
                    if audit and (len(saved) <= len(records) or saved[len(records)] != item):
                        raise AssertionError('Independent shuffled batch differs')
                    records.append(item); stream.write(json.dumps(item, allow_nan=False)+'\n'); stream.flush()
                print(json.dumps({'backbone_seed': seed, 'batches_checked': 256, 'audit': audit}), flush=True)
                del data, blocks
        original.deadline(begin); stats = summarize(records)
        if audit:
            old = original.read(execution/'summary.json')
            if (saved != records or any(old[k] != v for k, v in stats.items()) or old['status'] != 'completed'
                    or old['plan_sha256'] != plan_hash or not 0 < old['wall_seconds'] <= 900
                    or old['limits'] != PROTOCOL['limits']): raise AssertionError('Saved alignment summary differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'reference_batches_sha256': source.file_hash(out/'batches.jsonl'),
                      'all_shuffled_inputs_exact': True, **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
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
