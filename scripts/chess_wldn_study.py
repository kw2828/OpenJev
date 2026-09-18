# SPDX-License-Identifier: GPL-3.0-only
"""Frozen three-seed WLDN chess comparison, independent of pending contrast outcomes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import chess
import numpy as np
import torch
from torch.nn import functional as F

import chess_graph_contrast_study as contrast
import chess_wldn_preflight as preflight
from openjev.research.wldn_baseline import ChessWLDNHead

ROOT = contrast.ROOT
PARENT = 'evidence/chess-graph-contrast-quality-v1/protocol/plan.json'
PREFLIGHT_PLAN = 'evidence/chess-wldn-preflight-v1/protocol/plan.json'
VERSION = 'wldn-chess-matched-development-v1'
CODE = ['scripts/chess_wldn_study.py', 'tests/test_chess_wldn_study.py']
PROTOCOL = {
    'version': VERSION, 'seeds': [97, 109, 127], 'head_seed': 'backbone_seed+1100',
    'hypothesis': 'Characterize scalar graph contrast against a nodewise WLDN-style difference ranker; not a new method claim.',
    'parameters': 16638, 'backbone': 'Same original frozen direct policies as transport and graph contrast',
    'information': 'Same frozen root node features in both branches, same native child graphs and full legal menus, same120 action features appended after pooled difference encoding. No recomputed child node features or teacher shortlist.',
    'architecture': 'Published shared tied WL encoder,3 steps width32; subtract nodewise representations;1 distinct WL difference step on child graph;sum all64 squares;59-unit readout with original120 action features;zero scalar correction initially.',
    'initialization': preflight.PROTOCOL['initialization'],
    'adaptation_scope': 'Selected public method output/gradient checks with Torch shim, not TensorFlow runtime or official chemistry benchmark reproduction. Four multi-hot directed/color attack flags, no normalization/self-loop insertion. Fixed root-node features and action readout are explicit chess adaptations.',
    'training_examples': 32768, 'epochs': 6, 'batch_size': 128, 'updates_per_fit': 1536,
    'fits': 3, 'total_updates': 4608, 'optimizer': 'Adam', 'learning_rate': .001,
    'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'gradient_clip': 1.,
    'batch_order_seed': '700000+100*backbone_seed+epoch', 'loss': 'legal-move CE only',
    'device': 'cpu', 'threads': 2, 'deterministic_algorithms': True,
    'evaluation': 'All3 final WLDN fits finish before evaluation; retain every seed and final checkpoint. No best checkpoint, replacement, extension or tuning from outcomes.',
    'reference': 'Unchanged base decisions copied from authenticated original transport files with explicit provenance.',
    'comparison': 'Only after BOTH the12-fit contrast study and this3-fit study complete and are audited, join by row identity. Contrast must beat WLDN by>=.01 mean agreement on BOTH old panels, each paired seed>=-.005. Both new checks plus all8 original contrast checks required. All earlier failures remain unchanged.',
    'bootstrap': '2000 source-game resamples of paired correctness averaged across3 seeds, seed845101+shift_indicator; conditional descriptive95% intervals without adaptive/multiplicity correction.',
    'latency': 'During this run, base and WLDN rotate over first16 old dev roots,3 repeats,2 warmups. The later comparison rotates base, contrast and WLDN over those same roots and budgets in one process. Full native graph construction charged. Shared-host characterization.',
    'time_cap_seconds': 7200,
    'limits': 'Previously exposed development panels, not independent confirmation. Similar parameters do not imply equal FLOPs, tuning or active capacity. No new engine calls, gameplay or novelty claim. GPL reference/adaptation remains separately identified.'}


def signature():
    prepared = json.loads((ROOT/PREFLIGHT_PLAN).read_text()); current = preflight.signature()
    assert all(prepared[k] == v for k, v in current.items())
    directory = ROOT/'runs/chess-wldn-preflight-v1/execution'
    completed = json.loads((directory/'completed.json').read_text())
    assert completed['status'] == 'completed' and completed['plan_sha256'] == contrast.file_hash(ROOT/PREFLIGHT_PLAN)
    assert all(contrast.file_hash(directory/p) == h for p, h in completed['files'].items())
    audit_path = 'evidence/chess-wldn-preflight-v1/audit/receipt.json'
    audit = json.loads((ROOT/audit_path).read_text())
    assert audit['status'] == 'completed' and audit['numerical_passed']
    assert audit['execution_receipt_sha256'] == contrast.file_hash(directory/'completed.json')
    assert audit['summary_sha256'] == contrast.file_hash(directory/'summary.json')
    assert audit['numerical_replays'] == 24 and audit['additional_artificial_update_replays'] == 3
    assert audit['final_checkpoint_max_error'] == 0
    return {'protocol': PROTOCOL, 'sources': {p: contrast.file_hash(ROOT/p) for p in CODE},
        'preflight_signature': current, 'preflight_plan_sha256': contrast.file_hash(ROOT/PREFLIGHT_PLAN),
        'preflight_audit_sha256': contrast.file_hash(ROOT/audit_path),
        'contrast_plan_sha256': contrast.file_hash(ROOT/PARENT),
        'fit_order': [f'wldn-{seed}' for seed in PROTOCOL['seeds']]}


def prepare(out):
    plan = signature(); plan['prepared_unix'] = time.time()
    out.mkdir(parents=True, exist_ok=False); contrast.prior.write(out/'plan.json', plan)
    print(json.dumps({'plan_sha256': contrast.file_hash(out/'plan.json'), 'fits': 3, 'updates': 4608}))


def deadline(begin):
    if time.time()-begin > PROTOCOL['time_cap_seconds']: raise TimeoutError('Frozen120-minute ceiling exceeded')


def load(out, seed, plan_hash):
    saved = torch.load(out/f'wldn-{seed}/weights.pt', weights_only=True, map_location='cpu')
    assert saved['version'] == VERSION and saved['seed'] == seed and saved['plan_sha256'] == plan_hash
    head = ChessWLDNHead(seed=seed+1100); head.load_state_dict(saved['state_dict']); return head.eval()


def train(seed, model, data, features, graphs, out, plan_hash, begin):
    out.mkdir(); head = ChessWLDNHead(seed=seed+1100); contrast.prior.save(out/'initial.pt', head.state_dict())
    optimizer = torch.optim.Adam(head.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    start = time.perf_counter(); updates = 0
    with (out/'learning.jsonl').open('x') as stream:
        for epoch in range(6):
            order = np.random.default_rng(700000+100*seed+epoch).permutation(32768)
            for offset in range(0, 32768, 128):
                deadline(begin); index = torch.from_numpy(order[offset:offset+128]); graph = graphs.batch(index)
                optimizer.zero_grad(set_to_none=True)
                logits = head(*contrast.arguments(model, data, features, index, graph))
                loss = F.cross_entropy(logits, data['targets'][index]); assert torch.isfinite(loss)
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                record = {'update': updates, 'epoch': epoch, 'examples': 128, 'loss': float(loss.detach()),
                    'gradient_norm': float(norm), 'indices_sha256': hashlib.sha256(index.numpy().tobytes()).hexdigest()}
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            print(json.dumps({'fit': f'wldn-{seed}', 'epoch': epoch+1, 'updates': updates}), flush=True)
    deadline(begin)
    contrast.prior.save(out/'weights.pt', {'version': VERSION, 'seed': seed, 'plan_sha256': plan_hash, 'state_dict': head.state_dict()})
    contrast.prior.write(out/'training.json', {'status': 'completed', 'updates': updates,
        'examples_seen': updates*128, 'seconds': time.perf_counter()-start, 'parameters': sum(p.numel() for p in head.parameters()),
        'weights_sha256': contrast.file_hash(out/'weights.pt'), 'initial_sha256': contrast.file_hash(out/'initial.pt'),
        'learning_sha256': contrast.file_hash(out/'learning.jsonl')})


@torch.no_grad()
def evaluate(head, model, data, features, graphs, rows, path, begin):
    records = []
    for start in range(0, len(rows), 128):
        deadline(begin); index = torch.arange(start, min(start+128, len(rows))); graph = graphs.batch(index)
        logits = head(*contrast.arguments(model, data, features, index, graph))
        choices = logits.argmax(-1).tolist(); losses = F.cross_entropy(logits, data['targets'][index], reduction='none').tolist()
        for j, (choice, loss) in enumerate(zip(choices, losses)):
            i = start+j; row = rows[i]; name = graph['menus'][j][choice]
            records.append({'index': i, 'id': row['id'], 'game_id': row['game_id'], 'choice': name,
                            'correct': name == row['target_uci'], 'target_nll': loss})
    with path.open('x') as stream:
        for r in records: stream.write(json.dumps(r, allow_nan=False)+'\n')
    return {'agreement': statistics.mean(r['correct'] for r in records),
            'target_nll': statistics.mean(r['target_nll'] for r in records), 'examples': len(records)}


@torch.no_grad()
def decision(model, head, fen):
    board = chess.Board(fen); names, candidates = contrast.prior.encode_candidates(board)
    candidates = torch.from_numpy(candidates)[None]; mask = torch.ones(1, len(names), dtype=torch.bool)
    logits, _, hidden = model(torch.from_numpy(contrast.prior.encode_board(board))[None], candidates, mask)
    if head is not None:
        nodes = hidden.flatten(2).transpose(1, 2); features = contrast.prior.candidate_features(model, nodes, candidates)
        graph = contrast.candidate_graphs([board]); assert list(names) == graph['menus'][0]
        logits = head(nodes, features, candidates, mask, logits, graph['root'], graph['children'][graph['mask']])
    return names[int(logits.argmax(-1))]


def gate(metrics):
    checks = []
    for split in ('dev', 'shift'):
        gains = [metrics[f'contrast-{seed}'][split]['agreement']-metrics[f'wldn-{seed}'][split]['agreement'] for seed in (97, 109, 127)]
        checks.append({'split': split, 'comparator': 'wldn', 'mean_gain': statistics.mean(gains),
            'minimum_paired_seed_gain': min(gains), 'required_mean_gain': .01,
            'passed': statistics.mean(gains) >= .01 and min(gains) >= -.005})
    return checks


def run(plan_path, out):
    plan = json.loads(plan_path.read_text()); assert all(plan[k] == v for k, v in signature().items())
    out.mkdir(parents=True, exist_ok=False); begin = time.time(); plan_hash = contrast.file_hash(plan_path)
    contrast.prior.write(out/'started.json', {'status': 'started', 'pid': os.getpid(), 'unix': begin, 'plan_sha256': plan_hash})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        old_plan = json.loads((ROOT/contrast.PARENT).read_text())
        data = torch.load(ROOT/contrast.PREVIOUS/'packed-train.pt', weights_only=True, map_location='cpu')
        graphs = contrast.ChildGraphCache(ROOT/contrast.CACHE/'train'); checks = []
        for seed in PROTOCOL['seeds']:
            deadline(begin); model = contrast.prior.backbone(seed, old_plan); features = contrast.prior.root_cache(model, data)
            original = torch.load(ROOT/contrast.PREVIOUS/f'root-cache-train-{seed}.pt', weights_only=True, map_location='cpu')
            assert torch.equal(torch.isneginf(features['base_logits']), torch.isneginf(original['base_logits']))
            error = max(float((features['nodes']-original['nodes']).abs().max()),
                float((features['base_logits'][data['mask']]-original['base_logits'][data['mask']]).abs().max()))
            assert error <= 2e-6; checks.append({'seed': seed, 'training_roots': 32768, 'max_cache_error': error})
            del original
            train(seed, model, data, features, graphs, out/f'wldn-{seed}', plan_hash, begin)
            del features, model; gc.collect()
        del data, graphs; gc.collect(); deadline(begin)
        contrast.prior.write(out/'training-cache-check.json', checks)
        contrast.prior.write(out/'all-training-complete.json', {'fits': 3, 'before_any_evaluation': True, 'unix': time.time(), 'plan_sha256': plan_hash})
        rows = {s: contrast.prior.rows(ROOT/contrast.prior.DATA[s]) for s in ('dev', 'shift')}; metrics = {}; references = []
        for split in ('dev', 'shift'):
            data = torch.load(ROOT/contrast.PREVIOUS/f'packed-{split}.pt', weights_only=True, map_location='cpu')
            graphs = contrast.ChildGraphCache(ROOT/contrast.CACHE/split)
            for seed in PROTOCOL['seeds']:
                deadline(begin); model = contrast.prior.backbone(seed, old_plan); features = contrast.prior.root_cache(model, data)
                head = load(out, seed, plan_hash)
                metrics.setdefault(f'wldn-{seed}', {})[split] = evaluate(head, model, data, features, graphs, rows[split], out/f'wldn-{seed}-{split}.jsonl', begin)
                name = f'base-{seed}-{split}.jsonl'; origin = ROOT/contrast.PREVIOUS/name
                with (out/name).open('xb') as stream: stream.write(origin.read_bytes())
                records = contrast.prior.rows(out/name)
                metrics.setdefault(f'base-{seed}', {})[split] = {'agreement': statistics.mean(r['correct'] for r in records),
                    'target_nll': statistics.mean(r['target_nll'] for r in records), 'examples': len(records)}
                references.append({'file': name, 'copied_from': str(origin.relative_to(ROOT)), 'sha256': contrast.file_hash(origin), 'fresh_inference': False})
            del features, model, data, graphs; gc.collect()
        contrast.prior.write(out/'reference-provenance.json', references); contrast.prior.write(out/'metrics.json', metrics)
        model = contrast.prior.backbone(97, old_plan); head = load(out, 97, plan_hash); heads = {'base': None, 'wldn': head}; timing = []
        for a in heads:
            for _ in range(2): decision(model, heads[a], chess.STARTING_FEN)
        for i, row in enumerate(rows['dev'][:16]):
            for repeat in range(3):
                methods = ['base', 'wldn']; offset = (i+repeat) % 2
                for a in methods[offset:]+methods[:offset]:
                    deadline(begin); start = time.perf_counter(); choice = decision(model, heads[a], row['fen'])
                    timing.append({'root_index': i, 'repeat': repeat, 'arm': a, 'milliseconds': 1000*(time.perf_counter()-start), 'choice': choice})
        contrast.prior.write(out/'latency.json', {'records': timing, 'scope': PROTOCOL['latency'], 'host_load': list(os.getloadavg())})
        deadline(begin)
        summary = {'status': 'completed', 'plan_sha256': plan_hash, 'fits': 3, 'training_updates': 4608,
            'metrics': metrics, 'new_prediction_records': 12288, 'copied_reference_records': 12288,
            'comparison_status': 'Requires separately audited contrast outputs; no advantage decision in this execution',
            'new_engine_calls': 0, 'external_model_calls': 0, 'original_failed_criteria_unchanged': True,
            'wall_seconds': time.time()-begin, 'limits': PROTOCOL['limits']}
        contrast.prior.write(out/'summary.json', summary)
        contrast.prior.write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'files': {str(p.relative_to(out)): contrast.file_hash(p) for p in out.rglob('*') if p.is_file()}})
        print(json.dumps({'status': 'completed', 'wall_seconds': summary['wall_seconds']}), flush=True)
    except BaseException as error:
        contrast.prior.write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.time()-begin})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.out) if args.command == 'prepare' else run(args.plan, args.out)
