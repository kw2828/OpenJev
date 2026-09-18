# SPDX-License-Identifier: GPL-3.0-only
"""Frozen native-input and artificial-update screen of the WLDN chess adapter."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess
import torch
from torch.nn import functional as F

import chess_graph_contrast_study as source
from openjev.research.wldn_baseline import ChessWLDNHead, VERSION

ROOT = source.ROOT
PARENT = 'evidence/chess-graph-contrast-quality-v1/protocol/plan.json'
REFERENCES = 'evidence/chess-wldn-reference-v1/sources/manifest.json'
CODE = ['src/openjev/research/wldn_baseline.py', 'tests/test_wldn_baseline.py',
        'scripts/chess_wldn_preflight.py']
PROTOCOL = {
    'version': VERSION, 'scope': 'Engineering only, no chess-quality evaluation',
    'backbone_seed': 97, 'probe_seeds': [211, 223, 227], 'probe_roots': 8,
    'probe_order': 'First8 old training roots, all legal candidates, no label selection',
    'probe_projection': 'Replace zero output with normal(0,0.1), seed900000+head_seed',
    'comparison': 'Shared batched root/child processing versus a separate complete one-candidate forward call for every legal move',
    'absolute_score_tolerance': 1e-5,
    'tolerance_scope': 'Float32 summed-node scalar scores. Fixed before this screen; not a relaxation of any earlier architecture tolerance.',
    'training_roots': 128, 'training_order': 'First128 old training roots, reused for each artificial update',
    'head_seed': 1197, 'parameters': 16638, 'representation_width': 32,
    'representation_steps': 3, 'difference_steps': 1, 'readout_width': 59,
    'initialization': 'Normal affine weights with std=min(1/sqrt(fan_in),0.1), zero biases; final scalar projection zeroed for backbone preservation',
    'artificial_updates': 3, 'target': 'First legal move for every root; teacher targets unused',
    'optimizer': 'Adam', 'learning_rate': .001, 'gradient_clip': 1.,
    'runtime': 'CPU,2 threads, deterministic algorithms', 'time_cap_seconds': 600,
    'cost': 'Update timing includes lossless child-graph batch decoding, forward, CE, backward, clipping and optimizer step. Excludes source hashing and frozen-root feature setup, which are recorded separately. Shared host; no speed claim.',
    'limits': 'Selected official method parity uses a Torch compatibility layer, not TensorFlow binaries or its training pipeline. New root-feature chess adapter, all64 squares, multi-hot typed directed attacks, extra120 action-feature readout. No teacher agreement, engine scores, games, novelty or official benchmark reproduction.'}


def signature():
    parent = json.loads((ROOT/PARENT).read_text())
    assert all(parent[k] == value for k, value in source.signature().items())
    reference = json.loads((ROOT/REFERENCES).read_text())
    assert reference['commit'] == '5905475ae279a49fb318d7cf8d33a93b9a0c4e49'
    refs = {r['path']: r['sha256'] for r in reference['sources']}
    assert all(source.file_hash(ROOT/p) == h for p, h in refs.items())
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'parent_plan_sha256': source.file_hash(ROOT/PARENT),
            'reference_manifest_sha256': source.file_hash(ROOT/REFERENCES), 'reference_sources': refs,
            'parent_signature': {k: v for k, v in parent.items() if k != 'prepared_unix'}}


def prepare(out):
    plan = signature(); plan['prepared_unix'] = time.time()
    out.mkdir(parents=True, exist_ok=False); source.prior.write(out/'plan.json', plan)
    print(json.dumps({'plan_sha256': source.file_hash(out/'plan.json'), 'artificial_updates': 3}), flush=True)


def setup():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); start = time.perf_counter()
    packed = torch.load(ROOT/source.PREVIOUS/'packed-train.pt', weights_only=True, map_location='cpu')
    data = {k: v[:128].clone() for k, v in packed.items()}; del packed; gc.collect()
    parent = json.loads((ROOT/source.PARENT).read_text()); model = source.prior.backbone(97, parent)
    features = source.prior.root_cache(model, data)
    old = torch.load(ROOT/source.PREVIOUS/'root-cache-train-97.pt', weights_only=True, map_location='cpu')
    error = max(float((features['nodes']-old['nodes'][:128]).abs().max()),
                float((features['base_logits'][data['mask']]-old['base_logits'][:128][data['mask']]).abs().max()))
    assert error <= 2e-6 and torch.equal(torch.isneginf(features['base_logits']), torch.isneginf(old['base_logits'][:128]))
    del old; gc.collect()
    graphs = source.ChildGraphCache(ROOT/source.CACHE/'train')
    rows = source.prior.rows(ROOT/source.prior.DATA['train'])[:128]
    return model, data, features, graphs, rows, {'seconds': time.perf_counter()-start, 'cache_max_error': error}


@torch.no_grad()
def probes(model, data, features, graphs, rows):
    batch = graphs.batch(torch.arange(8)); args = source.arguments(model, data, features, torch.arange(8), batch)
    native = source.candidate_graphs([chess.Board(r['fen']) for r in rows[:8]])
    assert torch.equal(batch['root'], native['root']) and torch.equal(batch['children'], native['children'][native['mask']])
    assert batch['menus'] == list(map(tuple, native['menus']))
    records = []
    for seed in PROTOCOL['probe_seeds']:
        head = ChessWLDNHead(seed=seed)
        assert torch.equal(head(*args), args[4])
        generator = torch.Generator().manual_seed(900000+seed)
        head.output.weight.copy_(torch.randn(head.output.weight.shape, generator=generator)*.1)
        batched = head(*args); child_offset = 0
        for b, names in enumerate(batch['menus']):
            error = 0.
            for m in range(len(names)):
                single = [args[0][b:b+1], args[1][b:b+1, m:m+1], args[2][b:b+1, m:m+1],
                          torch.ones(1, 1, dtype=torch.bool), args[4][b:b+1, m:m+1],
                          args[5][b:b+1], args[6][child_offset+m:child_offset+m+1]]
                score = head(*single)[0, 0]
                assert torch.isfinite(score)
                error = max(error, float((score-batched[b, m]).abs()))
            child_offset += len(names)
            records.append({'seed': seed, 'root_index': b, 'id': rows[b]['id'], 'candidates': len(names),
                            'max_score_error': error, 'passed': error <= PROTOCOL['absolute_score_tolerance']})
    return records


def updates(model, data, features, graphs):
    head = ChessWLDNHead(seed=1197); initial = {k: v.clone() for k, v in head.state_dict().items()}
    assert sum(p.numel() for p in head.parameters()) == 16638
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    records = []
    for i in range(3):
        start = time.perf_counter(); index = torch.arange(128); graph = graphs.batch(index)
        optimizer.zero_grad(set_to_none=True)
        logits = head(*source.arguments(model, data, features, index, graph))
        loss = F.cross_entropy(logits, torch.zeros(128, dtype=torch.long)); assert torch.isfinite(loss)
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
        records.append({'update': i+1, 'artificial_target_loss': float(loss.detach()),
                        'gradient_norm': float(norm), 'seconds': time.perf_counter()-start})
    return initial, head.state_dict(), records


def execute(plan_path, out):
    plan = json.loads(plan_path.read_text()); assert all(plan[k] == value for k, value in signature().items())
    out.mkdir(parents=True, exist_ok=False); begin = time.time(); plan_hash = source.file_hash(plan_path)
    source.prior.write(out/'started.json', {'unix': begin, 'pid': os.getpid(), 'plan_sha256': plan_hash})
    try:
        model, data, features, graphs, rows, preparation = setup()
        records = probes(model, data, features, graphs, rows)
        with (out/'probes.jsonl').open('x') as stream:
            for r in records: stream.write(json.dumps(r)+'\n')
        assert all(r['passed'] for r in records), 'Frozen numerical criterion failed'
        initial, final, training = updates(model, data, features, graphs)
        source.prior.save(out/'initial.pt', initial); source.prior.save(out/'weights.pt', final)
        changes = {k: int((v-initial[k]).count_nonzero()) for k, v in final.items()}
        assert all(v > 0 for v in changes.values())
        summary = {'status': 'completed', 'plan_sha256': plan_hash, 'numerical_passed': True,
            'root_seed_checks': len(records), 'candidate_checks': sum(r['candidates'] for r in records),
            'max_score_error': max(r['max_score_error'] for r in records),
            'initial_policy_exact': True, 'parameters': 16638, 'updates': training,
            'artificial_training_updates': 3, 'median_update_seconds': statistics.median(r['seconds'] for r in training),
            'preparation': preparation, 'changed_parameter_coordinates': changes,
            'new_engine_calls': 0, 'new_development_evaluations': 0,
            'wall_seconds': time.time()-begin, 'limits': PROTOCOL['limits']}
        assert summary['wall_seconds'] <= 600
        source.prior.write(out/'summary.json', summary)
        source.prior.write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'files': {p.name: source.file_hash(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.time()-begin})
        raise


def audit(plan_path, execution, out):
    plan = json.loads(plan_path.read_text()); assert all(plan[k] == value for k, value in signature().items())
    plan_hash = source.file_hash(plan_path); completed = json.loads((execution/'completed.json').read_text())
    assert completed['status'] == 'completed' and completed['plan_sha256'] == plan_hash
    assert {p.name for p in execution.iterdir() if p.is_file()} == set(completed['files']) | {'completed.json'}
    assert all(source.file_hash(execution/name) == expected for name, expected in completed['files'].items())
    out.mkdir(parents=True, exist_ok=False); begin = time.time()
    source.prior.write(out/'started.json', {'unix': begin, 'plan_sha256': plan_hash})
    try:
        model, data, features, graphs, rows, preparation = setup()
        records = probes(model, data, features, graphs, rows)
        assert records == source.prior.rows(execution/'probes.jsonl')
        initial, final, training = updates(model, data, features, graphs)
        expected_initial = torch.load(execution/'initial.pt', weights_only=True)
        expected_final = torch.load(execution/'weights.pt', weights_only=True)
        assert initial.keys() == expected_initial.keys() == final.keys() == expected_final.keys()
        assert all(torch.equal(v, expected_initial[k]) for k, v in initial.items())
        error = max(float((v-expected_final[k]).abs().max()) for k, v in final.items()); assert error == 0
        summary = json.loads((execution/'summary.json').read_text())
        assert summary['status'] == 'completed' and summary['plan_sha256'] == plan_hash
        assert summary['numerical_passed'] == all(r['passed'] for r in records)
        assert summary['root_seed_checks'] == len(records) == 24
        assert summary['candidate_checks'] == sum(r['candidates'] for r in records)
        assert summary['max_score_error'] == max(r['max_score_error'] for r in records)
        assert summary['initial_policy_exact'] and summary['parameters'] == 16638
        assert summary['preparation']['cache_max_error'] == preparation['cache_max_error']
        assert summary['artificial_training_updates'] == len(training) == 3
        assert summary['new_engine_calls'] == summary['new_development_evaluations'] == 0
        assert summary['wall_seconds'] <= 600
        assert summary['changed_parameter_coordinates'] == {k: int((v-initial[k]).count_nonzero()) for k, v in final.items()}
        for a, b in zip(training, summary['updates']):
            assert {k:v for k,v in a.items() if k != 'seconds'} == {k:v for k,v in b.items() if k != 'seconds'}
            assert math.isfinite(b['seconds']) and b['seconds'] > 0
        assert summary['median_update_seconds'] == statistics.median(r['seconds'] for r in summary['updates'])
        result = {'status': 'completed', 'plan_sha256': plan_hash,
            'summary_sha256': source.file_hash(execution/'summary.json'),
            'execution_receipt_sha256': source.file_hash(execution/'completed.json'),
            'auditor_sha256': source.file_hash(__file__), 'numerical_replays': len(records),
            'candidate_checks': sum(r['candidates'] for r in records), 'numerical_passed': True,
            'additional_artificial_update_replays': 3, 'final_checkpoint_max_error': error,
            'new_engine_calls': 0, 'new_development_evaluations': 0, 'wall_seconds': time.time()-begin,
            'scope': 'Full source and artifact hashes, all native score comparisons, fresh frozen-root features, initial state and all3 artificial updates replayed. Timing records authenticated; no latency advantage or training-quality claim.'}
        source.prior.write(out/'receipt.json', result); print(json.dumps(result), flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'audit'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out)
    elif args.command == 'run': execute(args.plan, args.out)
    else: audit(args.plan, args.execution, args.out)
