# SPDX-License-Identifier: GPL-3.0-only
"""Bounded complete-update profiling; artificial targets and exact state replay."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import torch

import chess_pin_pairwise_preflight as parent
import chess_pin_comparator_selection as selection
from openjev.research import chess_pin_study as study

ROOT, source, original = parent.ROOT, parent.source, parent.original
PLAN = 'evidence/chess-pin-pairwise-preflight-v1/protocol/plan.json'
RUN = 'runs/chess-pin-pairwise-preflight-v1/execution'
AUDIT = 'evidence/chess-pin-pairwise-preflight-v1/audit/receipt.json'
SELECTION = 'evidence/chess-pin-comparator-selection-v1/protocol/plan.json'
ARMS = tuple(study.PARAMETERS)
CODE = ['scripts/chess_pin_update_profile.py', 'tests/test_chess_pin_update_profile.py']
PROTOCOL = {
    'version': 'pin-study-complete-update-profile-v1', 'arms': list(ARMS), 'backbone_seeds': list(study.SEEDS),
    'roots': 128, 'candidates': 3920, 'head_instances': 33, 'updates_per_head': 3, 'total_updates': 99,
    'scope': 'All11 supported head choices and3 canonical backbones on first128 original training roots. Artificial first-legal-move targets only; no teacher fields, quality metrics, comparator selection or new positions.',
    'recipe': 'Frozen shared factory, optimizer and update helper; CPU float32 heads/features, CPU2 deterministic. Three128-root epochs of the shared schedule per instance. Reset head/optimizer for each arm/backbone.',
    'measurement': 'Complete step begins before native graph-cache decoding and canonical feature/pin batch assembly; ends after Adam update. Includes forward, backward and gradient checks. Also retain helper update-only seconds. Constructor/cache loading/checkpoint saves excluded from step timing and included in run wall time.',
    'aggregation': 'Retain all99 timings; per-arm/per-backbone steady estimate is median of steps2 and3. Report cold step1 separately. Project1536 updates per future fit: sum over all7 core arms and3 backbones; additional maximum is the largest two distinct union-arm totals across all3 backbones. No timer threshold or quality gate.',
    'replay': 'Authenticate full primary manifest; exactly recreate all initial/final head tensors, losses, gradient norms and index hashes with the same production kernels. Recompute all original timing summaries; do not claim timing replication or an independent neural implementation.',
    'runtime': '900 seconds per execution/audit, existing main quality study unchanged.', 'time_cap_seconds': 900,
    'limits': 'Short fixed-prefix artificial-update profile on a shared host, not full-corpus training throughput, native decision latency, long-run budget guarantee or quality. First-step optimizer state, input variability, learned activation patterns, setup, evaluation and host contention limit extrapolation. Full quality budget must be frozen separately.',
}


def read(path): return json.loads(Path(path).read_text())


def signature():
    frozen = read(ROOT/PLAN)
    if frozen != parent.signature(): raise ValueError('Pairwise preflight signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or not audit['numerical_gate_passed'] or not audit['primary_numerical_gate_passed']
            or audit['candidate_score_checks'] != 35280 or audit['artificial_updates_replayed'] != 9
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['reference_probes_sha256'] != source.file_hash(ROOT/Path(AUDIT).parent/'probes.jsonl')):
        raise ValueError('Pairwise audit identity changed')
    selected = read(ROOT/SELECTION); current = selection.signature()
    if not all(selected[k] == v for k, v in current.items()): raise ValueError('Comparator rule changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'pairwise_signature': frozen, 'pairwise_audit_sha256': source.file_hash(ROOT/AUDIT),
            'selection_signature': current, 'selection_plan_sha256': source.file_hash(ROOT/SELECTION)}


def summarize(records):
    expected = [(seed, arm, step) for seed in study.SEEDS for arm in ARMS for step in (1, 2, 3)]
    if [(r['backbone_seed'], r['arm'], r['step']) for r in records] != expected:
        raise AssertionError('Incomplete update-profile membership')
    for r in records:
        if (r['roots'] != 128 or r['candidates'] != 3920 or not r['indices_sha256']
                or any(not math.isfinite(r[k]) for k in ('loss', 'gradient_norm', 'update_seconds', 'complete_step_seconds'))
                or r['loss'] < 0 or r['gradient_norm'] < 0 or not 0 < r['update_seconds'] <= r['complete_step_seconds']):
            raise AssertionError('Invalid update-profile record')
    by_arm = {}
    for arm in ARMS:
        by_arm[arm] = {}
        for seed in study.SEEDS:
            rows = [r for r in records if (r['arm'], r['backbone_seed']) == (arm, seed)]
            by_arm[arm][str(seed)] = {'cold_complete_seconds': rows[0]['complete_step_seconds'],
                'steady_complete_median_seconds': statistics.median(r['complete_step_seconds'] for r in rows[1:]),
                'steady_update_only_median_seconds': statistics.median(r['update_seconds'] for r in rows[1:])}
    cost = {a: 1536*sum(v['steady_complete_median_seconds'] for v in seeds.values()) for a, seeds in by_arm.items()}
    core = sum(cost[a] for a in study.CORE_ARMS)
    pairs = list(itertools.combinations(('union:'+a for a in study.UNION_ARMS), 2))
    worst = max(pairs, key=lambda pair: sum(cost[a] for a in pair))
    return {'head_instances': 33, 'artificial_updates': len(records), 'by_arm_and_backbone': by_arm,
            'projected_seconds_per_three_seed_arm': cost, 'core_21_fit_update_seconds': core,
            'largest_two_union_cost_arms': list(worst),
            'maximum_27_fit_update_seconds': core+sum(cost[a] for a in worst)}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = read(plan_path)
    if plan != signature(): raise ValueError('Frozen update-profile signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        pin = torch.load(ROOT/'runs/chess-pin-training-cache-v1/execution/blocks/00000.pt', weights_only=True, map_location='cpu')
        graphs = source.ChildGraphCache(ROOT/source.CACHE/'train')
        saved = source.prior.rows(execution/'updates.jsonl') if audit else None; records = []
        with (out/'updates.jsonl').open('x') as stream:
            for seed in study.SEEDS:
                feature = torch.load(ROOT/f'runs/chess-native-training-cache-v1/execution/{seed}/00000.pt', weights_only=True, map_location='cpu')
                data = parent.parent.parent.TrainingInputs([feature], [pin])
                for arm in ARMS:
                    model = study.make_head(arm, seed); opt = study.optimizer(model)
                    initial = {k: v.clone() for k, v in model.state_dict().items()}
                    for epoch, indices in study.schedule(128, 3, 128, seed):
                        original.deadline(begin); tick = time.perf_counter()
                        args, factors = data.batch(indices, graphs.batch(indices))
                        values = study.update(model, arm, opt, args, factors, torch.zeros(128, dtype=torch.long))
                        seconds = time.perf_counter()-tick
                        item = {'backbone_seed': seed, 'arm': arm, 'step': epoch+1, 'roots': 128,
                                'candidates': int(args[3].sum()), 'indices_sha256': study.index_digest(indices),
                                'complete_step_seconds': seconds, **values}
                        if audit:
                            old = saved[len(records)]
                            if any(item[k] != old[k] for k in item if k not in ('update_seconds', 'complete_step_seconds')):
                                raise AssertionError('Artificial float32 update replay differs')
                        records.append(item); stream.write(json.dumps(item, allow_nan=False)+'\n'); stream.flush()
                    name = f'{seed}-{arm.replace(":", "_")}'
                    final = model.state_dict()
                    if torch.equal(initial['output.weight'], final['output.weight']): raise AssertionError('Head did not update')
                    for label, state in (('initial', initial), ('final', final)):
                        path = f'{name}-{label}.pt'
                        if audit:
                            original.pins.compare_states(state, torch.load(execution/path, weights_only=True, map_location='cpu'), 0)
                        else: source.prior.save(out/path, state)
                    print(json.dumps({'backbone_seed': seed, 'arm': arm, 'updates': 3, 'audit': audit}), flush=True)
        original.deadline(begin); replay_stats = summarize(records)
        if audit:
            stats = summarize(saved); old = read(execution/'summary.json')
            if (len(saved) != len(records) or any(old[k] != v for k, v in stats.items())
                    or old['status'] != 'completed' or old['plan_sha256'] != plan_hash
                    or old['limits'] != PROTOCOL['limits'] or not 0 < old['wall_seconds'] <= 900):
                raise AssertionError('Saved profile summary differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'replay_updates_sha256': source.file_hash(out/'updates.jsonl'),
                      'initial_and_final_states_exact': 66, 'update_values_and_indices_exact': 99,
                      'timing_summaries_recomputed': True, 'timing_replication_claimed': False,
                      'head_instances': replay_stats['head_instances'], 'artificial_updates': replay_stats['artificial_updates'],
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            result = {'status': 'completed', 'plan_sha256': plan_hash, **replay_stats,
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
