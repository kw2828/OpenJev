"""Frozen quality comparison of shared graph contrast and its ingredient controls."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import chess
import numpy as np
import torch
from torch.nn import functional as F

import chess_transport_study as prior
from openjev.research.chess_graph_contrast import ARMS, candidate_graphs
from openjev.research.chess_graph_contrast_shared import SharedRootGraphContrastHead
from openjev.research.chess_child_graph_cache import ChildGraphCache, file_hash

ROOT = prior.ROOT
PARENT = 'evidence/chess-transport-v1/protocol/plan.json'
PREVIOUS = 'runs/chess-transport-v1/execution'
CACHE = 'runs/chess-child-graph-cache-v1/execution'
CACHE_PLAN = 'evidence/chess-child-graph-cache-v1/protocol/plan.json'
VERSION = 'shared-native-graph-contrast-quality-v1'
CODE = ['src/openjev/research/chess_graph_contrast.py', 'src/openjev/research/chess_graph_contrast_shared.py',
        'src/openjev/research/chess_counterfactual_transport.py',
        'src/openjev/research/chess_child_graph_cache.py', 'scripts/chess_graph_contrast_study.py',
        'tests/test_chess_graph_contrast.py', 'tests/test_chess_graph_contrast_shared.py',
        'tests/test_chess_child_graph_cache.py', 'tests/test_chess_graph_contrast_study.py']
PROTOCOL = {
    'version': VERSION, 'arms': list(ARMS), 'seeds': [97, 109, 127],
    'hypothesis': 'An action-dependent shared-weight child-minus-root graph score improves decisions over root-only, child-only and candidate-permuted graph contrast. This is development evidence, not a novelty assertion.',
    'backbone': 'Same original frozen direct checkpoints as transport-v1; new zero-output heads, no reuse of fitted transport or contrast weights',
    'head_seed': 'backbone_seed+1100', 'head_parameters': 16744, 'steps': 3, 'width': 32,
    'architecture': 'Original root transport kernel shared across candidates, dense compact child operators, same head weights in both branches. Contrast subtracts the action-dependent root correction. Identical binary native graphs give exactly zero residual and gradient.',
    'information': 'Frozen root node features and existing120 action features. Native legal moves construct child attack graphs in the root-player frame; node features are not recomputed after the move. No teacher shortlist, engine-derived feature or learned transition.',
    'training_examples': 32768, 'epochs': 6, 'batch_size': 128, 'updates_per_fit': 1536,
    'fits': 12, 'total_updates': 18432, 'optimizer': 'Adam', 'learning_rate': .001,
    'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'gradient_clip': 1.,
    'loss': 'legal-move CE only', 'batch_order_seed': '700000+100*backbone_seed+epoch', 'fit_order_seed': 810031,
    'device': 'cpu', 'threads': 2, 'deterministic_algorithms': True,
    'cache': 'All native root/child graphs independently reconstructed before freezing; packed bits decode losslessly. Recompute all32768 frozen root feature caches for each seed and compare to the original cache within2e-6.',
    'evaluation': 'All12 final fits finish before any evaluation. All seeds and final checkpoints retained; no model replacement, checkpoint selection or outcome-driven extension. Both2048-position panels are already exposed development data.',
    'diagnostic': 'Also evaluate each fitted contrast head with child graphs permuted at test time; no additional gate based on this diagnostic.',
    'reference': 'Unchanged backbone outputs copied from authenticated completed transport records, with explicit provenance; not new inference.',
    'gate': 'On BOTH panels, contrast must beat root, child and separately trained permuted_contrast by at least0.01 mean agreement, and must not lose against the unchanged backbone. Every paired seed in every comparison must gain at least-0.005. All8 checks required. Previous failed criteria remain unchanged.',
    'bootstrap': '2000 source-game cluster resamples of paired correctness averaged across3 seeds; conditional descriptive95% intervals, no correction for adaptive development, multiple comparisons or seed population.',
    'latency': 'First16 old dev roots, seed97,2 warmups and3 repeats; base and4 trained heads rotate. Full decisions include native legal enumeration, board encoding, backbone, graph construction, head and UCI choice. Root-only avoids unnecessary child construction. Shared-host characterization, no hardware-general speed claim.',
    'time_cap_seconds': 7200,
    'limits': 'Equal head parameter counts are not equal FLOPs or optimal tuning. Root-only has less input information; child-only isolates the subtractive term, and permuted contrast tests action alignment. No engine quality, gameplay, independent confirmation or algorithmic novelty claim.'}


def fit_order():
    rng = random.Random(PROTOCOL['fit_order_seed']); result = {}
    for seed in PROTOCOL['seeds']:
        arms = list(ARMS); rng.shuffle(arms); result[str(seed)] = arms
    return result


def signature():
    original = json.loads((ROOT/PARENT).read_text()); parent_signature = prior.signature()
    assert all(original[k] == v for k, v in parent_signature.items())
    previous = json.loads((ROOT/PREVIOUS/'completed.json').read_text())
    cache = json.loads((ROOT/CACHE/'completed.json').read_text())
    assert previous['status'] == cache['status'] == 'completed'
    assert previous['plan_sha256'] == file_hash(ROOT/PARENT)
    assert cache['plan_sha256'] == file_hash(ROOT/CACHE_PLAN)
    inputs = {}
    names = [f'packed-{s}.pt' for s in ('train', 'dev', 'shift')]
    names += [f'root-cache-train-{seed}.pt' for seed in PROTOCOL['seeds']]
    names += [f'base-{seed}-{split}.jsonl' for seed in PROTOCOL['seeds'] for split in ('dev', 'shift')]
    for name in names:
        path = f'{PREVIOUS}/{name}'; inputs[path] = file_hash(ROOT/path)
        assert inputs[path] == previous['files'][name]
    for name, expected in cache['files'].items():
        path = f'{CACHE}/{name}'; inputs[path] = file_hash(ROOT/path); assert inputs[path] == expected
    for path in (f'{PREVIOUS}/completed.json', f'{CACHE}/completed.json', CACHE_PLAN,
                 'evidence/chess-child-graph-cache-v1/audit/receipt.json',
                 'runs/chess-graph-contrast-v3/execution/summary.json',
                 'runs/chess-graph-contrast-v3/execution/completed.json',
                 'evidence/chess-graph-contrast-v3/audit/receipt.json'):
        inputs[path] = file_hash(ROOT/path)
    audit = json.loads((ROOT/'evidence/chess-child-graph-cache-v1/audit/receipt.json').read_text())
    assert audit['status'] == 'completed' and audit['execution_receipt_sha256'] == inputs[f'{CACHE}/completed.json']
    assert {s: v['roots'] for s, v in audit['reconstructed'].items()} == {'train': 32768, 'dev': 2048, 'shift': 2048}
    engineering = json.loads((ROOT/'evidence/chess-graph-contrast-v3/audit/receipt.json').read_text())
    assert engineering['status'] == 'completed' and engineering['numerical_passed']
    assert engineering['execution_receipt_sha256'] == inputs['runs/chess-graph-contrast-v3/execution/completed.json']
    assert engineering['additional_artificial_update_replays'] == 12 and not any(engineering['final_checkpoint_max_errors'].values())
    return {'protocol': PROTOCOL, 'sources': {n: file_hash(ROOT/n) for n in CODE},
            'inputs': inputs, 'parent_signature': parent_signature, 'parent_plan_sha256': file_hash(ROOT/PARENT),
            'fit_order': fit_order()}


def prepare(out):
    plan = signature(); plan['prepared_unix'] = time.time()
    out.mkdir(parents=True, exist_ok=False); prior.write(out/'plan.json', plan)
    print(json.dumps({'plan_sha256': file_hash(out/'plan.json'), 'fits': 12, 'updates': 18432}))


def deadline(start):
    if time.time()-start > PROTOCOL['time_cap_seconds']:
        raise TimeoutError('Frozen120-minute execution ceiling exceeded')


def arguments(model, data, features, index, graphs):
    args = list(prior.arguments(model, data, features, index, 'transport'))
    width = graphs['mask'].shape[1]
    for i in (1, 2, 3, 4): args[i] = args[i][:, :width]
    assert torch.equal(args[3], graphs['mask']) and torch.equal(args[5], graphs['root'])
    return args + [graphs['children']]


def train(seed, arm, model, data, features, graphs, out, plan_hash, begin):
    out.mkdir(); start = time.perf_counter(); head = SharedRootGraphContrastHead(arm, seed=seed+1100)
    prior.save(out/'initial.pt', head.state_dict())
    optimizer = torch.optim.Adam(head.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    updates = 0
    with (out/'learning.jsonl').open('x') as stream:
        for epoch in range(6):
            order = np.random.default_rng(700000+100*seed+epoch).permutation(32768)
            for offset in range(0, 32768, 128):
                deadline(begin); index = torch.from_numpy(order[offset:offset+128]); graph = graphs.batch(index)
                optimizer.zero_grad(set_to_none=True)
                logits = head(*arguments(model, data, features, index, graph), permutation=graph['permutation'])
                loss = F.cross_entropy(logits, data['targets'][index]); assert torch.isfinite(loss)
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                record = {'update': updates, 'epoch': epoch, 'examples': 128, 'loss': float(loss.detach()),
                          'gradient_norm': float(norm), 'indices_sha256': hashlib.sha256(index.numpy().tobytes()).hexdigest()}
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            print(json.dumps({'fit': f'{arm}-{seed}', 'epoch': epoch+1, 'updates': updates}), flush=True)
    deadline(begin)
    prior.save(out/'weights.pt', {'version': VERSION, 'arm': arm, 'seed': seed,
                                'plan_sha256': plan_hash, 'state_dict': head.state_dict()})
    prior.write(out/'training.json', {'status': 'completed', 'updates': updates, 'examples_seen': updates*128,
        'seconds': time.perf_counter()-start, 'parameters': sum(p.numel() for p in head.parameters()),
        'weights_sha256': file_hash(out/'weights.pt'), 'initial_sha256': file_hash(out/'initial.pt'),
        'learning_sha256': file_hash(out/'learning.jsonl')})


def load(out, seed, arm, plan_hash):
    saved_arm = 'contrast' if arm == 'contrast_corrupted' else arm
    state = torch.load(out/f'{saved_arm}-{seed}/weights.pt', weights_only=True, map_location='cpu')
    assert state['version'] == VERSION and state['arm'] == saved_arm and state['seed'] == seed and state['plan_sha256'] == plan_hash
    selected = 'permuted_contrast' if arm == 'contrast_corrupted' else arm
    head = SharedRootGraphContrastHead(selected, seed=seed+1100); head.load_state_dict(state['state_dict']); return head.eval()


@torch.no_grad()
def evaluate(head, model, data, features, graphs, rows, path, begin):
    records = []
    for start in range(0, len(rows), 128):
        deadline(begin); index = torch.arange(start, min(start+128, len(rows))); graph = graphs.batch(index)
        logits = head(*arguments(model, data, features, index, graph), permutation=graph['permutation'])
        choices = logits.argmax(-1).tolist(); losses = F.cross_entropy(logits, data['targets'][index], reduction='none').tolist()
        for j, (choice, loss) in enumerate(zip(choices, losses)):
            i = start+j; row = rows[i]; move = graph['menus'][j][choice]
            records.append({'index': i, 'id': row['id'], 'game_id': row['game_id'], 'choice': move,
                            'correct': move == row['target_uci'], 'target_nll': loss})
    with path.open('x') as stream:
        for record in records: stream.write(json.dumps(record, allow_nan=False)+'\n')
    return {'agreement': statistics.mean(r['correct'] for r in records),
            'target_nll': statistics.mean(r['target_nll'] for r in records), 'examples': len(records)}


def gate(metrics):
    checks = []
    for split in ('dev', 'shift'):
        for arm in ('base', 'root', 'child', 'permuted_contrast'):
            gains = [metrics[f'contrast-{s}'][split]['agreement']-metrics[f'{arm}-{s}'][split]['agreement'] for s in (97, 109, 127)]
            required = 0. if arm == 'base' else .01
            checks.append({'split': split, 'comparator': arm, 'mean_gain': statistics.mean(gains),
                'minimum_paired_seed_gain': min(gains), 'required_mean_gain': required,
                'passed': statistics.mean(gains) >= required and min(gains) >= -.005})
    return checks


def summarize(out, metrics, rows):
    intervals = []
    for split in ('dev', 'shift'):
        full = [prior.rows(out/f'contrast-{s}-{split}.jsonl') for s in (97, 109, 127)]
        ids = np.array([r['game_id'] for r in rows[split]]); groups = [np.flatnonzero(ids == g) for g in sorted(set(ids))]
        for arm in ('base', 'root', 'child', 'permuted_contrast', 'contrast_corrupted'):
            other = [prior.rows(out/f'{arm}-{s}-{split}.jsonl') for s in (97, 109, 127)]
            difference = np.mean([[int(a['correct'])-int(b['correct']) for a, b in zip(x, y)] for x, y in zip(full, other)], axis=0)
            sums = np.array([difference[g].sum() for g in groups]); counts = np.array([len(g) for g in groups])
            sample = np.random.default_rng(820037+(split == 'shift')).integers(0, len(groups), (2000, len(groups)))
            intervals.append({'split': split, 'comparator': arm, 'games': len(groups), 'point_gain': float(difference.mean()),
                              'percentile95': np.quantile(sums[sample].sum(1)/counts[sample].sum(1), [.025, .975]).tolist()})
    checks = gate(metrics)
    return {'status': 'completed', 'scope': 'already exposed development panels', 'metrics': metrics,
        'gate_checks': checks, 'continuation_passed': all(c['passed'] for c in checks),
        'conditional_game_bootstrap': intervals, 'fits': 12, 'training_updates': 18432,
        'new_prediction_records': 61440, 'copied_reference_records': 12288,
        'new_engine_calls': 0, 'external_model_calls': 0, 'original_failed_criteria_unchanged': True,
        'limits': PROTOCOL['limits']}


@torch.no_grad()
def decision(model, head, fen):
    board = chess.Board(fen); names, candidates = prior.encode_candidates(board)
    candidates = torch.from_numpy(candidates)[None]; mask = torch.ones(1, len(names), dtype=torch.bool)
    observations = torch.from_numpy(prior.encode_board(board))[None]
    logits, _, hidden = model(observations, candidates, mask)
    if head is not None:
        nodes = hidden.flatten(2).transpose(1, 2)
        action_features = prior.candidate_features(model, nodes, candidates)
        if head.contrast_arm == 'root':
            edges = torch.from_numpy(prior.root_relations(board))[None]
            logits = head(nodes, action_features, candidates, mask, logits, edges)
        else:
            graph = candidate_graphs([board]); assert list(names) == graph['menus'][0]
            logits = head(nodes, action_features, candidates, mask, logits, graph['root'],
                          graph['children'][graph['mask']], permutation=graph['permutation'])
    return names[int(logits.argmax(-1))]


def run(plan_path, out):
    plan = json.loads(plan_path.read_text()); assert all(plan[k] == value for k, value in signature().items())
    old_plan = json.loads((ROOT/PARENT).read_text()); plan_hash = file_hash(plan_path)
    out.mkdir(parents=True, exist_ok=False); begin = time.time()
    prior.write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'pid': os.getpid(), 'unix': begin})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        data = torch.load(ROOT/PREVIOUS/'packed-train.pt', weights_only=True, map_location='cpu')
        graphs = ChildGraphCache(ROOT/CACHE/'train'); cache_checks = []
        for seed in PROTOCOL['seeds']:
            deadline(begin); model = prior.backbone(seed, old_plan); features = prior.root_cache(model, data)
            old = torch.load(ROOT/PREVIOUS/f'root-cache-train-{seed}.pt', weights_only=True, map_location='cpu')
            assert torch.equal(torch.isneginf(features['base_logits']), torch.isneginf(old['base_logits']))
            error = max(float((features['nodes']-old['nodes']).abs().max()),
                        float((features['base_logits'][data['mask']]-old['base_logits'][data['mask']]).abs().max()))
            assert error <= 2e-6; cache_checks.append({'seed': seed, 'training_roots': 32768, 'max_cache_error': error})
            del old
            for arm in plan['fit_order'][str(seed)]:
                train(seed, arm, model, data, features, graphs, out/f'{arm}-{seed}', plan_hash, begin)
            del features, model; gc.collect()
        del graphs, data; gc.collect(); deadline(begin)
        prior.write(out/'training-cache-check.json', cache_checks)
        prior.write(out/'all-training-complete.json', {'fits': 12, 'before_any_evaluation': True, 'unix': time.time(), 'plan_sha256': plan_hash})
        rows = {split: prior.rows(ROOT/prior.DATA[split]) for split in ('dev', 'shift')}; metrics = {}; references = []
        for split in ('dev', 'shift'):
            data = torch.load(ROOT/PREVIOUS/f'packed-{split}.pt', weights_only=True, map_location='cpu')
            graphs = ChildGraphCache(ROOT/CACHE/split)
            for seed in PROTOCOL['seeds']:
                deadline(begin); model = prior.backbone(seed, old_plan); features = prior.root_cache(model, data)
                for arm in (*ARMS, 'contrast_corrupted'):
                    head = load(out, seed, arm, plan_hash)
                    metrics.setdefault(f'{arm}-{seed}', {})[split] = evaluate(head, model, data, features, graphs, rows[split], out/f'{arm}-{seed}-{split}.jsonl', begin)
                name = f'base-{seed}-{split}.jsonl'; source = ROOT/PREVIOUS/name
                with (out/name).open('xb') as stream: stream.write(source.read_bytes())
                records = prior.rows(out/name)
                metrics.setdefault(f'base-{seed}', {})[split] = {'agreement': statistics.mean(r['correct'] for r in records),
                    'target_nll': statistics.mean(r['target_nll'] for r in records), 'examples': len(records)}
                references.append({'file': name, 'copied_from': str(source.relative_to(ROOT)), 'sha256': file_hash(source), 'fresh_inference': False})
            del data, graphs, features, model; gc.collect()
        prior.write(out/'reference-provenance.json', references); prior.write(out/'metrics.json', metrics)
        model = prior.backbone(97, old_plan); heads = {a: load(out, 97, a, plan_hash) for a in ARMS}; heads['base'] = None
        methods = ['base', *ARMS]; timing = []
        for arm in methods:
            for _ in range(2): decision(model, heads[arm], chess.STARTING_FEN)
        for i, row in enumerate(rows['dev'][:16]):
            for repeat in range(3):
                offset = (i+repeat) % len(methods)
                for arm in methods[offset:]+methods[:offset]:
                    deadline(begin); start = time.perf_counter(); choice = decision(model, heads[arm], row['fen'])
                    timing.append({'root_index': i, 'repeat': repeat, 'arm': arm, 'milliseconds': 1000*(time.perf_counter()-start), 'choice': choice})
        prior.write(out/'latency.json', {'records': timing, 'scope': PROTOCOL['latency'], 'host_load': list(os.getloadavg())})
        summary = summarize(out, metrics, rows); deadline(begin)
        summary.update(plan_sha256=plan_hash, wall_seconds=time.time()-begin)
        prior.write(out/'summary.json', summary)
        prior.write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'files': {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file()}})
        print(json.dumps({'status': 'completed', 'continuation_passed': summary['continuation_passed'], 'wall_seconds': summary['wall_seconds']}), flush=True)
    except BaseException as error:
        prior.write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.time()-begin})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.out) if args.command == 'prepare' else run(args.plan, args.out)
