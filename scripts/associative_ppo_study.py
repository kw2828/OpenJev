"""Matched recurrent PPO comparison of global and selective associative writes."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.world_object import Ball, Key
from minigrid.envs import MemoryEnv
from torch.distributions import Categorical
from torch.nn import functional as F

from openjev.research.associative_policy import MODES, AssociativePolicy
from openjev.research.cue_memory_env import CueMemoryBatch
from openjev.research.memory_env import OBS_SIZE, MemoryBatch, encode_obs
from openjev.research.predictive_memory import advantages

ROOT = Path(__file__).resolve().parents[1]
ARMS = list(MODES)
PROTOCOL = {
    'version': 'associative-ppo-v1', 'arms': ARMS, 'seeds': [101, 113, 127],
    'train_size': 11, 'eval_sizes': [11, 17, 23], 'view_size': 7, 'max_steps': 128,
    'environments': 16, 'rollout': 64, 'updates': 1024, 'epochs': 2, 'minibatch_envs': 8,
    'hidden': 64, 'store_dimension': 16, 'adapter_hidden': 33,
    'learning_rate': .0003, 'gamma': .99, 'gae_lambda': .95,
    'clip': .2, 'value_coefficient': .5, 'entropy_coefficient': .01, 'gradient_clip': .5,
    'training_seed_base': 80000000, 'evaluation_seed_start': 110000000,
    'policy_rng_offset': 900000, 'evaluation_episodes': 128, 'torch_threads': 1,
    'fit_order_seed': 917031,
    'evaluation_actions': 'greedy_argmax', 'selection': 'final snapshot only; all fits',
    'evaluation_modes': ['intact', 'reset_all', 'cue_swapped', 'native_start'],
    'fast_additional_mode': 'reset_store',
    'environment': 'Native MiniGrid Memory except x1 east-facing cue-visible reset and max_steps128',
    'cue_swapped': 'Swap cue key/ball after reset; original stored reward goals unchanged',
    'supervised_warm_start': False, 'auxiliary_losses': False,
    'continuation': {'candidate': 'fast_selective', 'same_size_success_minimum': .8,
                     'each_fit_same_size_success_minimum': .7,
                     'controls': ['gru', 'feedforward', 'fast_global'],
                     'minimum_gain_each_size_vs_each_control': .05,
                     'same_size_store_reset_drop_minimum': .2},
    'claim': 'Development comparison of established fast-weight mechanisms, not novelty or confirmation',
}
FILES = ['scripts/associative_ppo_study.py', 'src/openjev/research/associative_policy.py',
         'src/openjev/research/cue_memory_env.py', 'src/openjev/research/memory_env.py',
         'src/openjev/research/predictive_memory.py', 'tests/test_associative_policy.py',
         'tests/test_associative_ppo_study.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def signature():
    return {'protocol': PROTOCOL, 'source_hashes': {p: sha(ROOT/p) for p in FILES},
            'environment_source_sha256': sha(inspect.getfile(MemoryEnv)),
            'dependencies': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'gymnasium', 'minigrid']},
            'python': platform.python_version(), 'platform': platform.platform(),
            'lock_sha256': sha(ROOT/'uv.lock')}


def verify(plan):
    if json.loads(plan.read_text()) != signature():
        raise ValueError('Frozen protocol, source or runtime mismatch')


class SwappedCueBatch(CueMemoryBatch):
    def _reset_slot(self, index):
        super()._reset_slot(index)
        env = self.envs[index]
        cue = env.grid.get(1, env.height//2-1)
        if not isinstance(cue, (Key, Ball)):
            raise TypeError('Expected the native cue object')
        env.grid.set(1, env.height//2-1, Ball(cue.color) if isinstance(cue, Key) else Key(cue.color))
        return encode_obs(env.gen_obs())


def modes(arm):
    return [*PROTOCOL['evaluation_modes'], *(['reset_store'] if arm.startswith('fast_') else [])]


def fit(arm, seed, out, *, updates=None):
    p = PROTOCOL
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AssociativePolicy(arm, OBS_SIZE, p['hidden'])
    # Match sampled-action RNG after architecture-specific initialization.
    torch.manual_seed(p['policy_rng_offset'] + seed)
    optimizer = torch.optim.Adam((q for q in model.parameters() if q.requires_grad), lr=p['learning_rate'], eps=1e-5)
    current = False
    n, length = p['environments'], p['rollout']
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'started.json', {'arm': arm, 'seed': seed, 'protocol': p['version'],
                                   'updates': updates or p['updates'], 'started_unix': time.time()})
    total = time.perf_counter()
    episodes, gradient_steps, collection_seconds, optimization_seconds = [], 0, 0., 0.
    gradient_parameters = set()
    with CueMemoryBatch(n, p['train_size'], p['training_seed_base'] + seed * 100000,
                     p['max_steps']) as envs, (out / 'learning.jsonl').open('x') as logfile:
        obs = torch.from_numpy(envs.reset())
        previous = torch.zeros(n, dtype=torch.long)
        state = model.initial_state(n)
        reset = torch.ones(n, dtype=torch.bool)
        for update in range(updates or p['updates']):
            start = time.perf_counter()
            initial = state.detach().clone()
            history = {key: [] for key in ['obs', 'previous', 'reset', 'actions', 'logprob', 'values',
                                         'next_obs', 'next_values', 'rewards', 'terminated', 'ended']}
            with torch.no_grad():
                for _ in range(length):
                    logits, value, state = model.observe(obs, previous, state, reset, current)
                    distribution = Categorical(logits=logits)
                    actions = distribution.sample()
                    new_obs, rewards, terminal, timeout, infos = envs.step(actions.numpy())
                    ended = torch.from_numpy(terminal | timeout)
                    true_next = new_obs.copy()
                    for i, info in enumerate(infos):
                        if 'episode' in info:
                            episodes.append(info['episode'])
                            true_next[i] = info['terminal_observation']
                    next_obs = torch.from_numpy(true_next)
                    _, next_value, _ = model.observe(next_obs, actions, state,
                                                     torch.zeros_like(reset), current)
                    values = {'obs': obs, 'previous': previous, 'reset': reset, 'actions': actions,
                              'logprob': distribution.log_prob(actions), 'values': value,
                              'next_obs': next_obs, 'next_values': next_value,
                              'rewards': torch.from_numpy(rewards), 'terminated': torch.from_numpy(terminal),
                              'ended': ended}
                    for key, entries in history.items():
                        entries.append(values[key])
                    obs, previous, reset = torch.from_numpy(new_obs), actions, ended
            data = {key: torch.stack(value) for key, value in history.items()}
            adv, returns = advantages(data['rewards'], data['values'], data['next_values'],
                                      data['terminated'], data['ended'], p['gamma'], p['gae_lambda'])
            adv = (adv-adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            collection_seconds += time.perf_counter()-start
            start = time.perf_counter()
            metrics = []
            for _ in range(p['epochs']):
                order = torch.randperm(n)
                for offset in range(0, n, p['minibatch_envs']):
                    indices = order[offset:offset+p['minibatch_envs']]
                    logits, value, _ = model.sequence(data['obs'][:, indices], data['previous'][:, indices],
                                                           initial[indices], data['reset'][:, indices], current)
                    distribution = Categorical(logits=logits)
                    ratio = (distribution.log_prob(data['actions'][:, indices]) - data['logprob'][:, indices]).exp()
                    gain = ratio * adv[:, indices]
                    clipped_gain = ratio.clamp(1-p['clip'], 1+p['clip']) * adv[:, indices]
                    actor = -torch.minimum(gain, clipped_gain).mean()
                    critic = F.mse_loss(value, returns[:, indices])
                    entropy = distribution.entropy().mean()
                    loss = actor + p['value_coefficient']*critic - p['entropy_coefficient']*entropy
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss; preserve receipts and stop')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            gradient_parameters.add(name)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip'])
                    optimizer.step()
                    gradient_steps += 1
                    metrics.append([loss.item(), entropy.item(), actor.item(), critic.item()])
            optimization_seconds += time.perf_counter()-start
            state = state.detach()
            recent = episodes[-100:]
            record = {'update': update+1, 'interactions': (update+1)*n*length,
                      'completed_episodes': len(episodes),
                      'recent_success': float(np.mean([e['success'] for e in recent])) if recent else None,
                      'recent_return': float(np.mean([e['return'] for e in recent])) if recent else None,
                      'mean_losses': np.mean(metrics, axis=0).tolist(),
                      'elapsed_seconds': time.perf_counter()-total}
            logfile.write(json.dumps(record, allow_nan=False)+'\n')
            logfile.flush()
            if (update+1) % 16 == 0 or updates:
                print(json.dumps({'arm': arm, 'seed': seed, **record}), flush=True)
    checkpoint = out / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    elapsed = time.perf_counter()-total
    report = {'arm': arm, 'seed': seed, 'status': 'completed',
              'interactions': (updates or p['updates'])*n*length, 'gradient_steps': gradient_steps,
              'completed_episodes': len(episodes), 'training_seconds': elapsed,
              'collection_seconds': collection_seconds, 'optimization_seconds': optimization_seconds,
              'registered_parameters': sum(t.numel() for t in model.parameters()),
              'trainable_parameters': sum(t.numel() for t in model.parameters() if t.requires_grad),
              'state_values_per_episode': model.state_size,
              'gradient_parameters': sum(t.numel() for name, t in model.named_parameters() if name in gradient_parameters),
              'gradient_parameter_names': sorted(gradient_parameters), 'checkpoint_sha256': sha(checkpoint),
              'learning_sha256': sha(out / 'learning.jsonl')}
    write_new(out / 'completed.json', report)
    return report


@torch.inference_mode()
def evaluate(model, arm, size, mode='intact', *, random_policy=False):
    p = PROTOCOL
    environment = {'native_start': MemoryBatch, 'cue_swapped': SwappedCueBatch}.get(mode, CueMemoryBatch)
    results, timings, executed = [], [], 0
    write_stats = {name: {'steps': 0, 'beta_sum': 0., 'update_norm_sum': 0.}
                   for name in ['cue_visible', 'cue_hidden']}
    begin = time.perf_counter()
    for offset in range(0, p['evaluation_episodes'], p['environments']):
        n = min(p['environments'], p['evaluation_episodes']-offset)
        first = p['evaluation_seed_start'] + size*10000 + offset
        rng = np.random.default_rng(first)
        with environment(n, size, first, p['max_steps']) as envs:
            obs = torch.from_numpy(envs.reset())
            previous = torch.zeros(n, dtype=torch.long)
            state = None if random_policy else model.initial_state(n)
            reset = torch.ones(n, dtype=torch.bool)
            active = np.ones(n, dtype=bool)
            counts = np.zeros((n, 7), dtype=np.int64)
            for _ in range(p['max_steps']):
                old_state = state
                start = time.perf_counter()
                if random_policy:
                    actions = torch.from_numpy(rng.integers(7, size=n))
                else:
                    logits, _, state = model.observe(obs, previous, state, reset,
                                                     current_only=mode == 'reset_all',
                                                     reset_store=mode == 'reset_store')
                    if not torch.isfinite(logits).all() or not torch.isfinite(state).all():
                        raise ValueError('Nonfinite evaluation output')
                    actions = logits.argmax(-1)
                timings.append(time.perf_counter()-start)
                # Privileged visibility is for telemetry only and never enters observe().
                if not random_policy and model.uses_store:
                    encoded = model.encoder(obs)
                    beta = (model.write_gate(encoded).sigmoid().flatten()
                            if arm == 'fast_selective' else model.write_gate.bias.sigmoid().expand(n))
                    old_store = old_state[:, model.hidden:]
                    old_store = (torch.zeros_like(old_store) if mode in ['reset_all', 'reset_store']
                                 else old_store * (~reset).float().unsqueeze(-1))
                    update_norm = (state[:, model.hidden:]-old_store).norm(dim=-1)
                    for i in np.flatnonzero(active):
                        env = envs.envs[i]
                        name = 'cue_visible' if env.agent_sees(1, env.height//2-1) else 'cue_hidden'
                        bucket = write_stats[name]
                        bucket['steps'] += 1
                        bucket['beta_sum'] += float(beta[i])
                        bucket['update_norm_sum'] += float(update_norm[i])
                for i in np.flatnonzero(active):
                    counts[i, actions[i]] += 1
                following, _, terminated, truncated, infos = envs.step(actions.numpy())
                executed += n
                for i, info in enumerate(infos):
                    if active[i] and 'episode' in info:
                        results.append({**info['episode'],
                                        'wrong_goal': bool(terminated[i] and not info['episode']['success']),
                                        'timeout': bool(truncated[i] and not terminated[i]),
                                        'action_counts': counts[i].tolist()})
                        active[i] = False
                obs, previous = torch.from_numpy(following), actions
                reset = torch.from_numpy(terminated | truncated)
                if not active.any():
                    break
            if active.any():
                raise ValueError('Evaluation did not finish every assigned episode')
    results.sort(key=lambda row: row['seed'])
    return {'size': size, 'episodes': results,
            'success': float(np.mean([r['success'] for r in results])),
            'wrong_goal': float(np.mean([r['wrong_goal'] for r in results])),
            'timeout': float(np.mean([r['timeout'] for r in results])),
            'mean_return': float(np.mean([r['return'] for r in results])),
            'mean_length': float(np.mean([r['length'] for r in results])),
            'model_forward_seconds': sum(timings), 'model_batch_p50_ms': float(np.median(timings)*1000),
            'model_batch_p95_ms': float(np.percentile(timings, 95)*1000),
            'wall_seconds': time.perf_counter()-begin, 'executed_environment_steps': executed,
            'timing_scope': 'Model+argmax for full batch; includes background reset slots; telemetry excluded',
            'write_audit': write_stats if not random_policy and model.uses_store else None}


def run(plan, out):
    verify(plan)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'plan_sha256': sha(plan), 'started_unix': time.time()})
    jobs = [(arm, seed) for arm in ARMS for seed in PROTOCOL['seeds']]
    random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
    write_new(out/'fit_order.json', jobs)
    training = [fit(arm, seed, out/'fits'/f'{arm}-{seed}') for arm, seed in jobs]
    write_new(out/'training-completed.json', {'status': 'completed', 'fits': training})
    for arm, seed in jobs:
        checkpoint = out/'fits'/f'{arm}-{seed}'/'model.pt'
        model = AssociativePolicy(arm)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        for mode in modes(arm):
            results = [evaluate(model, arm, size, mode) for size in PROTOCOL['eval_sizes']]
            write_new(out/'evaluation'/f'{arm}-{seed}-{mode}.json',
                      {'arm': arm, 'seed': seed, 'mode': mode,
                       'checkpoint_sha256': sha(checkpoint), 'results': results})
            print(json.dumps({'evaluation': f'{arm}-{seed}-{mode}',
                              'success_by_size': [r['success'] for r in results]}), flush=True)
    write_new(out/'evaluation/random.json',
              {'arm': 'random', 'mode': 'intact',
               'results': [evaluate(None, 'random', size, random_policy=True) for size in PROTOCOL['eval_sizes']]})
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(plan),
                                     'fits': len(training), 'evaluations': 55, 'finished_unix': time.time()})


def gates(averages, per_fit):
    rule, size = PROTOCOL['continuation'], str(PROTOCOL['train_size'])
    candidate = rule['candidate']
    checks = {
        'same_size_success': averages[candidate]['intact'][size]['success'] >= rule['same_size_success_minimum'],
        'each_fit_same_size_success': all(r['success'] >= rule['each_fit_same_size_success_minimum']
                                         for r in per_fit[candidate]['intact'][size]),
        'store_reset_effect': (averages[candidate]['intact'][size]['success']
                               - averages[candidate]['reset_store'][size]['success']
                               >= rule['same_size_store_reset_drop_minimum']),
    }
    for control in rule['controls']:
        for test_size in PROTOCOL['eval_sizes']:
            key = str(test_size)
            delta = averages[candidate]['intact'][key]['success']-averages[control]['intact'][key]['success']
            checks[f'gain_vs_{control}_size_{key}'] = delta >= rule['minimum_gain_each_size_vs_each_control']
    return checks


def validate_episode_result(result):
    size, rows = result['size'], result['episodes']
    start = PROTOCOL['evaluation_seed_start'] + size*10000
    if [r['seed'] for r in rows] != list(range(start, start+PROTOCOL['evaluation_episodes'])):
        raise ValueError('Evaluation seed coverage mismatch')
    for row in rows:
        if any(type(row[k]) is not bool for k in ['success', 'wrong_goal', 'timeout']):
            raise ValueError('Expected boolean episode outcomes')
        if sum(row[k] for k in ['success', 'wrong_goal', 'timeout']) != 1:
            raise ValueError('Episode outcomes do not partition examples')
        if (type(row['length']) is not int or not 1 <= row['length'] <= PROTOCOL['max_steps']
                or len(row['action_counts']) != 7
                or sum(row['action_counts']) != row['length']
                or any(not isinstance(v, int) or v < 0 for v in row['action_counts'])):
            raise ValueError('Invalid episode action count or length')
        expected_return = 1-.9*row['length']/PROTOCOL['max_steps'] if row['success'] else 0.
        if not np.isfinite(row['return']) or abs(expected_return-row['return']) > 1e-10:
            raise ValueError('Return differs from native time-sensitive reward')
    for metric, field in [('success', 'success'), ('wrong_goal', 'wrong_goal'), ('timeout', 'timeout'),
                          ('mean_return', 'return'), ('mean_length', 'length')]:
        if not np.isfinite(result[metric]) or abs(result[metric]-np.mean([r[field] for r in rows])) > 1e-12:
            raise ValueError('Episode aggregates do not match rows')


def report(plan, execution, out):
    verify(plan)
    completed = json.loads((execution/'completed.json').read_text())
    if (completed['status'] != 'completed' or completed['plan_sha256'] != sha(plan)
            or completed['fits'] != 12 or completed['evaluations'] != 55):
        raise ValueError('Study not completed under this plan')
    training = json.loads((execution/'training-completed.json').read_text())['fits']
    expected_fits = {(arm, seed) for arm in ARMS for seed in PROTOCOL['seeds']}
    if len(training) != 12 or {(v['arm'], v['seed']) for v in training} != expected_fits:
        raise ValueError('Training fit identities differ')
    for row in training:
        path = execution/'fits'/f"{row['arm']}-{row['seed']}"
        if (json.loads((path/'completed.json').read_text()) != row or row['status'] != 'completed'
                or sha(path/'model.pt') != row['checkpoint_sha256']
                or sha(path/'learning.jsonl') != row['learning_sha256']
                or row['interactions'] != PROTOCOL['updates']*PROTOCOL['rollout']*PROTOCOL['environments']
                or row['gradient_steps'] != PROTOCOL['updates']*PROTOCOL['epochs']*2):
            raise ValueError('Training receipt, budget or checkpoint mismatch')
    expected = {(arm, seed, mode) for arm, seed in expected_fits for mode in modes(arm)}
    observed, records, grouped, random_count = set(), [], {}, 0
    for path in sorted((execution/'evaluation').glob('*.json')):
        record = json.loads(path.read_text())
        arm, mode = record['arm'], record['mode']
        if arm == 'random':
            random_count += 1
            if mode != 'intact':
                raise ValueError('Random baseline mode changed')
        else:
            identity = (arm, record['seed'], mode)
            if identity not in expected or identity in observed:
                raise ValueError('Unexpected or duplicated evaluation identity')
            observed.add(identity)
            if sha(execution/'fits'/f"{arm}-{record['seed']}"/'model.pt') != record['checkpoint_sha256']:
                raise ValueError('Evaluation checkpoint mismatch')
        records.append(record)
        if len(record['results']) != 3 or {r['size'] for r in record['results']} != set(PROTOCOL['eval_sizes']):
            raise ValueError('Missing evaluation sizes')
        for result in record['results']:
            validate_episode_result(result)
            bucket = grouped.setdefault(arm, {}).setdefault(mode, {}).setdefault(str(result['size']), [])
            bucket.append({k: result[k] for k in ['success', 'wrong_goal', 'timeout', 'mean_return', 'mean_length']})
    if observed != expected or random_count != 1 or len(records) != 55:
        raise ValueError('Missing evaluation conditions')
    averages = {arm: {mode: {size: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                             for size, rows in sizes.items()} for mode, sizes in conditions.items()}
                for arm, conditions in grouped.items()}
    paired = []
    by_identity = {(r['arm'], r.get('seed'), r['mode']): r for r in records}
    for arm, seed in sorted(expected_fits):
        intact = by_identity[(arm, seed, 'intact')]
        for mode in modes(arm)[1:]:
            other = by_identity[(arm, seed, mode)]
            by_size = {r['size']: r for r in other['results']}
            for first in intact['results']:
                second = by_size[first['size']]
                a, b = first['episodes'], second['episodes']
                both = [not x['timeout'] and not y['timeout'] for x, y in zip(a, b, strict=True)]
                changed = [x['success'] != y['success'] for x, y in zip(a, b, strict=True)]
                paired.append({'arm': arm, 'seed': seed, 'mode': mode, 'size': first['size'],
                               'assigned_pairs': len(a), 'both_reach_branch': sum(both),
                               'reversed_branch': sum(x and y for x, y in zip(both, changed, strict=True)),
                               'success_difference': first['success']-second['success']})
    checks = gates(averages, grouped)
    summary = {'protocol': PROTOCOL, 'plan_sha256': sha(plan), 'averages': averages, 'per_fit': grouped,
               'checks': checks, 'continuation_passed': all(checks.values()), 'training': training,
               'training_seconds': sum(v['training_seconds'] for v in training),
               'interactions': sum(v['interactions'] for v in training), 'evaluation': records,
               'paired_interventions': paired, 'novelty_established': False, 'independent_confirmation': False}
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', json.loads(plan.read_text()))
    write_new(out/'summary.json', summary)
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(plan),
                                     'summary_sha256': sha(out/'summary.json')})
    print(json.dumps({'checks': checks, 'averages': averages}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run', 'report'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        write_new(args.out, signature())
    elif args.command == 'smoke':
        for arm in ARMS:
            fit(arm, 7, args.out/arm, updates=2)
    elif args.command == 'run':
        run(args.plan, args.out)
    else:
        report(args.plan, args.run, args.out)


if __name__ == '__main__':
    main()
