"""Cue-visible memory diagnostic, reusing the frozen predictive PPO implementation.

The original source files remain immutable. An isolated imported module receives
an explicit environment and protocol substitution. All other training code and
losses are identical to recurrent-world-v1; the new signature hashes both layers.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.world_object import Ball, Key

from openjev.research.cue_memory_env import CueMemoryBatch
from openjev.research.memory_env import OBS_SIZE, MemoryBatch, encode_obs
from openjev.research.predictive_memory import PredictiveMemory

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_cue_memory_frozen_base', ROOT / 'scripts/recurrent_world_study.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.PROTOCOL = {
    **base.PROTOCOL,
    'version': 'cue-memory-v1', 'seeds': [61, 73, 89],
    'training_seed_base': 20000000, 'evaluation_seed_start': 40000000,
    'probe_seed_start': 50000000, 'fit_order_seed': 91637,
    'start_intervention': 'After native map generation: center row, x=1, facing east; no other change',
    'additional_evaluations': ['cue_swapped', 'native_start'],
    'cue_swapped': 'Exchange cue key/ball after reset; preserve original stored reward goal and branch objects',
    'memory_diagnostic': {
        'candidate': 'recurrent_ppo', 'same_size_success_minimum': .8,
        'each_fit_same_size_success_minimum': .7,
        'same_size_gain_vs_current_minimum': .2,
        'same_size_drop_on_reset_minimum': .2,
    },
    'claim': 'Cue-visible development diagnostic; not independent confirmation, Astra training or novelty',
}
base.MemoryBatch = CueMemoryBatch
base.FILES = [*base.FILES, 'scripts/cue_memory_study.py',
              'src/openjev/research/cue_memory_env.py',
              'tests/test_cue_memory_env.py', 'tests/test_cue_memory_study.py']
PROTOCOL = base.PROTOCOL


class SwappedCueBatch(CueMemoryBatch):
    """Paired input intervention, deliberately inconsistent with the reward goal."""

    def _reset_slot(self, index):
        super()._reset_slot(index)
        env = self.envs[index]
        x, y = 1, env.height // 2 - 1
        cue = env.grid.get(x, y)
        if not isinstance(cue, (Key, Ball)):
            raise TypeError('Expected the native cue object')
        env.grid.set(x, y, Ball(cue.color) if isinstance(cue, Key) else Key(cue.color))
        return encode_obs(env.gen_obs())


def memory_gate(summary):
    size = str(PROTOCOL['train_size'])
    averages = summary['averages']
    rule = PROTOCOL['memory_diagnostic']
    success = averages['recurrent_ppo'][size]['success']
    checks = {
        'same_size_success': success >= rule['same_size_success_minimum'],
        'each_fit_same_size_success': all(
            row['success'] >= rule['each_fit_same_size_success_minimum']
            for row in summary['per_fit']['recurrent_ppo'][size]),
        'gain_vs_current': (success - averages['current_ppo'][size]['success']
                            >= rule['same_size_gain_vs_current_minimum']),
        'drop_on_reset': (success - averages['recurrent_ppo_reset'][size]['success']
                          >= rule['same_size_drop_on_reset_minimum']),
    }
    return checks


def branch_reversal(intact, swapped):
    """Opposite success at two terminal choices means opposite physical branches.

Stored reward targets stay fixed for each paired seed. Timeout pairs remain in
the denominator and cannot count as a reversal.
"""
    left, right = intact['episodes'], swapped['episodes']
    if [r['seed'] for r in left] != [r['seed'] for r in right]:
        raise ValueError('Cue intervention episode identities differ')
    both = [not a['timeout'] and not b['timeout'] for a, b in zip(left, right, strict=True)]
    reversed_pair = [valid and a['success'] != b['success']
                     for a, b, valid in zip(left, right, both, strict=True)]
    return {'assigned_pairs': len(left), 'both_reach_branch': int(sum(both)),
            'reversed_branch': int(sum(reversed_pair)),
            'both_reach_branch_fraction': float(np.mean(both)),
            'branch_reversal_fraction': float(np.mean(reversed_pair))}


def run(plan, out):
    base.verify(plan)
    out.mkdir(parents=True, exist_ok=False)
    base.write_new(out / 'started.json', {'plan_sha256': base.sha(plan), 'started_unix': time.time()})
    base.run(plan, out / 'core')
    jobs = json.loads((out / 'core/fit_order.json').read_text())
    for arm, seed in jobs:
        checkpoint = out / 'core/fits' / f'{arm}-{seed}' / 'model.pt'
        model = PredictiveMemory(OBS_SIZE, PROTOCOL['hidden'])
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        for mode, environment in [('cue_swapped', SwappedCueBatch), ('native_start', MemoryBatch)]:
            base.MemoryBatch = environment
            try:
                results = [base.evaluate(model, size, current=arm == 'current_ppo')
                           for size in PROTOCOL['eval_sizes']]
            finally:
                base.MemoryBatch = CueMemoryBatch
            record = {'arm': arm, 'seed': seed, 'mode': mode,
                      'checkpoint_sha256': base.sha(checkpoint), 'results': results}
            base.write_new(out / 'interventions' / f'{arm}-{seed}-{mode}.json', record)
            print(json.dumps({'evaluation': f'{arm}-{seed}-{mode}',
                              'success_by_size': [r['success'] for r in results]}), flush=True)
    base.write_new(out / 'completed.json', {'status': 'completed', 'plan_sha256': base.sha(plan),
                                           'finished_unix': time.time(), 'additional_evaluations': 24})


def report(plan, execution, out):
    base.verify(plan)
    receipt = json.loads((execution / 'completed.json').read_text())
    if (receipt['status'] != 'completed' or receipt['plan_sha256'] != base.sha(plan)
            or receipt['additional_evaluations'] != 24):
        raise ValueError('Cue diagnostic is incomplete')
    base.report(plan, execution / 'core', out)
    summary = json.loads((out / 'summary.json').read_text())
    fits = {(r['arm'], r['seed']): r for r in summary['training']}
    intact = {(r['arm'], r['seed']): r for r in summary['evaluation']
              if r.get('mode') == 'intact'}
    expected = {(arm, seed, mode) for arm in base.ARMS for seed in PROTOCOL['seeds']
                for mode in PROTOCOL['additional_evaluations']}
    observed, records, aggregates, reversals = set(), [], {}, []
    for path in sorted((execution / 'interventions').glob('*.json')):
        record = json.loads(path.read_text())
        arm, seed, mode = record['arm'], record['seed'], record['mode']
        identity = (arm, seed, mode)
        if identity not in expected or identity in observed:
            raise ValueError('Unexpected or duplicate intervention identity')
        if record['checkpoint_sha256'] != fits[(arm, seed)]['checkpoint_sha256']:
            raise ValueError('Intervention checkpoint mismatch')
        observed.add(identity)
        records.append(record)
        results = record['results']
        if len(results) != 3 or {r['size'] for r in results} != set(PROTOCOL['eval_sizes']):
            raise ValueError('Intervention size coverage mismatch')
        bucket = aggregates.setdefault(mode, {}).setdefault(arm, {})
        for result in results:
            size, rows = result['size'], result['episodes']
            start = PROTOCOL['evaluation_seed_start'] + size * 10000
            if [r['seed'] for r in rows] != list(range(start, start + PROTOCOL['evaluation_episodes'])):
                raise ValueError('Missing or changed intervention seeds')
            for row in rows:
                if sum(bool(row[k]) for k in ['success', 'wrong_goal', 'timeout']) != 1:
                    raise ValueError('Intervention outcomes do not partition episodes')
            for metric, field in [('success', 'success'), ('wrong_goal', 'wrong_goal'),
                                  ('timeout', 'timeout'), ('mean_return', 'return'), ('mean_length', 'length')]:
                actual = float(np.mean([r[field] for r in rows]))
                if not np.isfinite(actual) or abs(actual - result[metric]) > 1e-12:
                    raise ValueError('Intervention metric mismatch')
            bucket.setdefault(str(size), []).append({k: result[k] for k in
                                                     ['success', 'wrong_goal', 'timeout',
                                                      'mean_return', 'mean_length']})
            if mode == 'cue_swapped':
                original = next(r for r in intact[(arm, seed)]['results'] if r['size'] == size)
                reversals.append({'arm': arm, 'seed': seed, 'size': size,
                                  **branch_reversal(original, result)})
    if observed != expected:
        raise ValueError('Missing intervention records')
    averages = {mode: {arm: {size: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                             for size, rows in sizes.items()} for arm, sizes in arms.items()}
                for mode, arms in aggregates.items()}
    checks = memory_gate(summary)
    diagnostic = {'protocol_version': PROTOCOL['version'], 'plan_sha256': base.sha(plan),
                  'summary_sha256': base.sha(out / 'summary.json'), 'evaluations': records,
                  'averages': averages, 'branch_reversals': reversals, 'memory_checks': checks,
                  'memory_gate_passed': all(checks.values()), 'novelty_established': False,
                  'independent_confirmation': False}
    base.write_new(out / 'diagnostic.json', diagnostic)
    base.write_new(out / 'report-completed.json', {
        'status': 'completed', 'protocol_version': PROTOCOL['version'],
        'plan_sha256': base.sha(plan), 'summary_sha256': base.sha(out / 'summary.json'),
        'diagnostic_sha256': base.sha(out / 'diagnostic.json'),
    })
    print(json.dumps({'memory_checks': checks, 'intervention_averages': averages}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run', 'report'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        base.write_new(args.out, base.signature())
    elif args.command == 'smoke':
        for arm in base.ARMS:
            base.fit(arm, 7, args.out / arm, updates=2)
    elif args.command == 'run':
        run(args.plan, args.out)
    else:
        report(args.plan, args.run, args.out)


if __name__ == '__main__':
    main()
