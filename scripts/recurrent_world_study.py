"""Frozen MiniGrid memory / predictive recurrent PPO development study."""

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
from minigrid.envs import MemoryEnv
from torch.distributions import Categorical
from torch.nn import functional as F

from openjev.research.memory_env import OBS_SIZE, MemoryBatch
from openjev.research.predictive_memory import PredictiveMemory, advantages, auxiliary_loss

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['current_ppo', 'recurrent_ppo', 'reward_prediction', 'world_prediction']
PROTOCOL = {
    'version': 'recurrent-world-v1', 'seeds': [17, 29, 43], 'arms': ARMS,
    'train_size': 11, 'eval_sizes': [11, 17, 23], 'view_size': 7, 'max_steps': 128,
    'environments': 16, 'rollout': 64, 'updates': 512, 'epochs': 2, 'minibatch_envs': 8,
    'hidden': 64, 'learning_rate': .0003, 'gamma': .99, 'gae_lambda': .95,
    'clip': .2, 'value_coefficient': .5, 'entropy_coefficient': .01,
    'auxiliary_coefficient': .1, 'gradient_clip': .5, 'prediction_horizons': [1, 2, 4],
    'evaluation_episodes': 128, 'evaluation_seed_start': 8000000,
    'training_seed_base': 100000, 'probe_seed_start': 9000000, 'probe_length': 64, 'fit_order_seed': 71680, 'torch_threads': 1,
    'evaluation_actions': 'greedy_argmax', 'diagnostic': 'zero_state_every_step',
    'native_observation': 'partial categorical image plus direction; previous action; no hidden map',
    'episode_cap': 'env.max_steps=128 changes native reward denominator; not reward shaping',
    'selection': 'final snapshot only; all three training seeds; no tuned selection',
    'continuation': {'candidate': 'world_prediction', 'same_size_success_minimum': .8,
                     'minimum_gain_each_size_vs_each_control': .05,
                     'controls': ['recurrent_ppo', 'reward_prediction']},
    'claim': 'Predictive recurrent PPO pilot, not imagined-policy training, Astra training or novelty',
}
FILES = ['scripts/recurrent_world_study.py', 'src/openjev/research/predictive_memory.py',
         'src/openjev/research/memory_env.py']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def signature():
    return {'protocol': PROTOCOL, 'source_hashes': {p: sha(ROOT / p) for p in FILES},
            'environment_source_sha256': sha(inspect.getfile(MemoryEnv)),
            'dependencies': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'gymnasium', 'minigrid']},
            'python': platform.python_version(), 'platform': platform.platform(),
            'lock_sha256': sha(ROOT / 'uv.lock')}


def verify(plan):
    if json.loads(plan.read_text()) != signature():
        raise ValueError('Frozen protocol, source or runtime mismatch')


def fit(arm, seed, out, *, updates=None):
    p = PROTOCOL
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PredictiveMemory(OBS_SIZE, p['hidden'])
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], eps=1e-5)
    current = arm == 'current_ppo'
    predictive = arm in ['reward_prediction', 'world_prediction']
    n, length = p['environments'], p['rollout']
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'started.json', {'arm': arm, 'seed': seed, 'protocol': p['version'],
                                   'updates': updates or p['updates'], 'started_unix': time.time()})
    total = time.perf_counter()
    episodes, gradient_steps, collection_seconds, optimization_seconds = [], 0, 0., 0.
    gradient_parameters = set()
    with MemoryBatch(n, p['train_size'], p['training_seed_base'] + seed * 100000,
                     p['max_steps']) as envs, (out / 'learning.jsonl').open('x') as logfile:
        obs = torch.from_numpy(envs.reset())
        previous = torch.zeros(n, dtype=torch.long)
        state = torch.zeros(n, p['hidden'])
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
                    logits, value, states = model.sequence(data['obs'][:, indices], data['previous'][:, indices],
                                                           initial[indices], data['reset'][:, indices], current)
                    distribution = Categorical(logits=logits)
                    ratio = (distribution.log_prob(data['actions'][:, indices]) - data['logprob'][:, indices]).exp()
                    gain = ratio * adv[:, indices]
                    clipped_gain = ratio.clamp(1-p['clip'], 1+p['clip']) * adv[:, indices]
                    actor = -torch.minimum(gain, clipped_gain).mean()
                    critic = F.mse_loss(value, returns[:, indices])
                    entropy = distribution.entropy().mean()
                    auxiliary = actor.new_tensor(0.)
                    components = torch.zeros(4)
                    if predictive:
                        auxiliary, components = auxiliary_loss(
                            model, states, data['actions'][:, indices], data['next_obs'][:, indices],
                            data['rewards'][:, indices], data['terminated'][:, indices], data['ended'][:, indices],
                            full=arm == 'world_prediction', horizons=p['prediction_horizons'])
                    loss = (actor + p['value_coefficient']*critic - p['entropy_coefficient']*entropy
                            + p['auxiliary_coefficient']*auxiliary)
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
                    metrics.append([loss.item(), entropy.item(), *components.tolist()])
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
              'gradient_parameters': sum(t.numel() for name, t in model.named_parameters() if name in gradient_parameters),
              'gradient_parameter_names': sorted(gradient_parameters), 'checkpoint_sha256': sha(checkpoint),
              'learning_sha256': sha(out / 'learning.jsonl')}
    write_new(out / 'completed.json', report)
    return report


def evaluate(model, size, *, current=False, random_policy=False):
    p = PROTOCOL
    results, model_seconds, total_steps = [], 0., 0
    begin = time.perf_counter()
    for offset in range(0, p['evaluation_episodes'], p['environments']):
        n = min(p['environments'], p['evaluation_episodes']-offset)
        seed_start = p['evaluation_seed_start'] + size*10000 + offset
        rng = np.random.default_rng(seed_start)
        with MemoryBatch(n, size, seed_start, p['max_steps']) as envs:
            obs = torch.from_numpy(envs.reset())
            previous = torch.zeros(n, dtype=torch.long)
            state = torch.zeros(n, p['hidden'])
            reset = torch.ones(n, dtype=torch.bool)
            active = np.ones(n, dtype=bool)
            for _ in range(p['max_steps']):
                start = time.perf_counter()
                if random_policy:
                    actions = torch.from_numpy(rng.integers(7, size=n))
                else:
                    with torch.no_grad():
                        logits, _, state = model.observe(obs, previous, state, reset, current)
                        actions = logits.argmax(-1)
                model_seconds += time.perf_counter()-start
                new_obs, _, terminal, timeout, infos = envs.step(actions.numpy())
                total_steps += n  # Includes background auto-reset slots while awaiting other first episodes.
                for i, info in enumerate(infos):
                    if active[i] and 'episode' in info:
                        episode = info['episode']
                        results.append({**episode, 'wrong_goal': bool(terminal[i] and not episode['success']),
                                        'timeout': bool(timeout[i] and not terminal[i])})
                        active[i] = False
                obs, previous = torch.from_numpy(new_obs), actions
                reset = torch.from_numpy(terminal | timeout)
                if not active.any():
                    break
            if active.any():
                raise ValueError('Evaluation failed to finish every assigned episode')
    results.sort(key=lambda row: row['seed'])
    return {'size': size, 'episodes': results, 'model_seconds': model_seconds,
            'wall_seconds': time.perf_counter()-begin, 'executed_environment_steps': total_steps,
            'success': float(np.mean([r['success'] for r in results])),
            'wrong_goal': float(np.mean([r['wrong_goal'] for r in results])),
            'timeout': float(np.mean([r['timeout'] for r in results])),
            'mean_return': float(np.mean([r['return'] for r in results])),
            'mean_length': float(np.mean([r['length'] for r in results]))}


def forecast_diagnostics(model, size):
    """Common random-policy probe trajectories, independent of fitted actions."""
    p = PROTOCOL
    n = p['environments']
    seed = p['probe_seed_start'] + size*10000
    rng = np.random.default_rng(seed)
    data = {key: [] for key in ['obs', 'previous', 'resets', 'actions', 'next_obs', 'rewards', 'terminated', 'ended']}
    with MemoryBatch(n, size, seed, p['max_steps']) as envs:
        obs = envs.reset()
        previous = np.zeros(n, dtype=np.int64)
        resets = np.ones(n, dtype=bool)
        for _ in range(p['probe_length']):
            actions = rng.integers(7, size=n)
            following, rewards, terminal, timeout, infos = envs.step(actions)
            true_next = following.copy()
            for i, info in enumerate(infos):
                if 'terminal_observation' in info:
                    true_next[i] = info['terminal_observation']
            values = {'obs': obs, 'previous': previous, 'resets': resets, 'actions': actions,
                      'next_obs': true_next, 'rewards': rewards, 'terminated': terminal, 'ended': terminal | timeout}
            for key, entries in data.items():
                entries.append(values[key])
            obs, previous, resets = following, actions, terminal | timeout
    data = {key: torch.from_numpy(np.stack(value)) for key, value in data.items()}
    with torch.no_grad():
        _, _, states = model.sequence(data['obs'], data['previous'], torch.zeros(n, p['hidden']), data['resets'])
        _, component = auxiliary_loss(model, states, data['actions'], data['next_obs'], data['rewards'],
                                      data['terminated'], data['ended'], full=True,
                                      horizons=p['prediction_horizons'])
        flat = states.reshape(-1, p['hidden'])
        actions = data['actions'].reshape(-1)
        _, _, target = model.observe(data['next_obs'].reshape(-1, OBS_SIZE), actions, flat,
                                     torch.zeros(len(flat), dtype=torch.bool))
        predicted = model.imagine(flat, actions)
        predicted_error = F.mse_loss(predicted, target).item()
        persistence_error = F.mse_loss(flat, target).item()
    return {'size': size, 'probe_seed_start': seed, 'steps': p['probe_length']*n,
            'observations_sha256': hashlib.sha256(data['obs'].numpy().tobytes()).hexdigest(),
            'actions_sha256': hashlib.sha256(data['actions'].numpy().tobytes()).hexdigest(),
            'latent_feature_variance': flat.var(0, unbiased=False).mean().item(),
            'one_step_latent_mse': predicted_error, 'copy_current_latent_mse': persistence_error,
            'one_step_mse_ratio': predicted_error / persistence_error if persistence_error else None,
            'horizon_mean_losses': dict(zip(['reward_mse', 'termination_bce', 'latent_mse', 'observation_ce'],
                                            component.tolist(), strict=True)),
            'observation_head_trained': None}


def run(plan, out):
    verify(plan)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'started.json', {'plan_sha256': sha(plan), 'started_unix': time.time()})
    jobs = [(arm, seed) for seed in PROTOCOL['seeds'] for arm in ARMS]
    random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
    write_new(out / 'fit_order.json', jobs)
    fits = []
    for arm, seed in jobs:
        fits.append(fit(arm, seed, out / 'fits' / f'{arm}-{seed}'))
    write_new(out / 'training-completed.json', {'fits': fits, 'status': 'completed'})
    for arm, seed in jobs:
        directory = out / 'fits' / f'{arm}-{seed}'
        model = PredictiveMemory(OBS_SIZE, PROTOCOL['hidden'])
        model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
        model.eval()
        if arm in ['reward_prediction', 'world_prediction']:
            probes = [forecast_diagnostics(model, size) for size in PROTOCOL['eval_sizes']]
            for probe in probes:
                probe['observation_head_trained'] = arm == 'world_prediction'
            write_new(out / 'diagnostics' / f'{arm}-{seed}.json', {'arm': arm, 'seed': seed, 'probes': probes})
        for mode in ['intact'] + (['reset'] if arm != 'current_ppo' else []):
            results = [evaluate(model, size, current=mode == 'reset' or arm == 'current_ppo')
                       for size in PROTOCOL['eval_sizes']]
            write_new(out / 'evaluation' / f'{arm}-{seed}-{mode}.json',
                      {'arm': arm, 'seed': seed, 'mode': mode, 'checkpoint_sha256': sha(directory / 'model.pt'),
                       'results': results})
            print(json.dumps({'evaluation': f'{arm}-{seed}-{mode}',
                              'success_by_size': [r['success'] for r in results]}), flush=True)
    write_new(out / 'evaluation' / 'random.json',
              {'arm': 'random', 'results': [evaluate(None, size, random_policy=True)
                                            for size in PROTOCOL['eval_sizes']]})
    write_new(out / 'completed.json', {'status': 'completed', 'plan_sha256': sha(plan),
                                      'fits': len(fits), 'finished_unix': time.time()})


def report(plan, run_dir, out):
    verify(plan)
    completed = json.loads((run_dir / 'completed.json').read_text())
    if completed['status'] != 'completed' or completed['plan_sha256'] != sha(plan):
        raise ValueError('Incomplete or mismatched run')
    expected_fits = {(arm, seed) for arm in ARMS for seed in PROTOCOL['seeds']}
    training = json.loads((run_dir / 'training-completed.json').read_text())['fits']
    if (completed['fits'] != 12 or len(training) != 12
            or {(r['arm'], r['seed']) for r in training} != expected_fits):
        raise ValueError('Wrong training fit identities/count')
    for fit_result in training:
        directory = run_dir / 'fits' / f"{fit_result['arm']}-{fit_result['seed']}"
        if (json.loads((directory / 'completed.json').read_text()) != fit_result
                or fit_result['status'] != 'completed'
                or fit_result['interactions'] != PROTOCOL['environments']*PROTOCOL['rollout']*PROTOCOL['updates']
                or fit_result['gradient_steps'] != PROTOCOL['updates']*PROTOCOL['epochs']*(PROTOCOL['environments']//PROTOCOL['minibatch_envs'])
                or sha(directory / 'model.pt') != fit_result['checkpoint_sha256']
                or sha(directory / 'learning.jsonl') != fit_result['learning_sha256']):
            raise ValueError('Training receipt, budget or checkpoint mismatch')
    expected_eval = {(arm, seed, mode) for arm, seed in expected_fits
                     for mode in ['intact'] + (['reset'] if arm != 'current_ppo' else [])}
    observed_eval = []
    random_count = 0
    records, conditions = [], {}
    for path in sorted((run_dir / 'evaluation').glob('*.json')):
        record = json.loads(path.read_text())
        if record['arm'] == 'random':
            random_count += 1
        else:
            observed_eval.append((record['arm'], record['seed'], record['mode']))
            checkpoint = run_dir / 'fits' / f"{record['arm']}-{record['seed']}" / 'model.pt'
            if sha(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('Checkpoint identity mismatch')
        records.append(record)
        name = record['arm'] + ('_reset' if record.get('mode') == 'reset' else '')
        bucket = conditions.setdefault(name, {str(size): [] for size in PROTOCOL['eval_sizes']})
        for result in record['results']:
            rows = result['episodes']
            expected = list(range(PROTOCOL['evaluation_seed_start']+result['size']*10000,
                                  PROTOCOL['evaluation_seed_start']+result['size']*10000+PROTOCOL['evaluation_episodes']))
            if [r['seed'] for r in rows] != expected:
                raise ValueError('Evaluation seeds are missing, duplicated or changed')
            for row in rows:
                if sum(bool(row[key]) for key in ['success', 'wrong_goal', 'timeout']) != 1:
                    raise ValueError('Episode terminal states are not a partition')
            for key in ['success', 'wrong_goal', 'timeout']:
                actual = float(np.mean([r[key] for r in rows]))
                if abs(actual-result[key]) > 1e-12:
                    raise ValueError('Episode metrics mismatch')
            for key, field in [('mean_return', 'return'), ('mean_length', 'length')]:
                if abs(float(np.mean([r[field] for r in rows]))-result[key]) > 1e-12:
                    raise ValueError('Return or length metric mismatch')
            bucket[str(result['size'])].append({k: result[k] for k in
                                               ['success', 'wrong_goal', 'timeout', 'mean_return', 'mean_length']})
    if len(observed_eval) != len(expected_eval) or set(observed_eval) != expected_eval or random_count != 1:
        raise ValueError('Evaluation condition identities/count mismatch')
    for name, sizes in conditions.items():
        expected = 1 if name == 'random' else 3
        if any(len(rows) != expected for rows in sizes.values()):
            raise ValueError('Missing training fits')
    if len(conditions) != 8:
        raise ValueError('Missing expected conditions')
    averages = {name: {size: {key: float(np.mean([r[key] for r in values])) for key in values[0]}
                      for size, values in sizes.items()} for name, sizes in conditions.items()}
    candidate = averages['world_prediction']
    checks = {'same_size_success': candidate['11']['success'] >= .8}
    differences = {}
    for control in ['recurrent_ppo', 'reward_prediction']:
        differences[control] = {}
        for size in PROTOCOL['eval_sizes']:
            delta = candidate[str(size)]['success'] - averages[control][str(size)]['success']
            differences[control][str(size)] = delta
            checks[f'{control}_size_{size}'] = delta >= .05
    probes = [json.loads(path.read_text()) for path in sorted((run_dir / 'diagnostics').glob('*.json'))]
    if len(probes) != 6 or {(r['arm'], r['seed']) for r in probes} != {
            (arm, seed) for arm in ['reward_prediction', 'world_prediction'] for seed in PROTOCOL['seeds']}:
        raise ValueError('Missing prediction diagnostics')
    for size in PROTOCOL['eval_sizes']:
        identities = {(r['observations_sha256'], r['actions_sha256']) for record in probes
                      for r in record['probes'] if r['size'] == size}
        if len(identities) != 1:
            raise ValueError('Probe trajectories differ across fits')
    out.mkdir(parents=True, exist_ok=False)
    summary = {'protocol': PROTOCOL, 'plan_sha256': sha(plan), 'averages': averages,
               'per_fit': conditions, 'differences': differences, 'checks': checks,
               'continuation_passed': all(checks.values()), 'training': training,
               'interactions': sum(r['interactions'] for r in training),
               'training_seconds': sum(r['training_seconds'] for r in training),
               'evaluation': records, 'forecast_diagnostics': probes, 'novelty_established': False, 'independent_confirmation': False}
    write_new(out / 'summary.json', summary)
    write_new(out / 'plan.json', json.loads(plan.read_text()))
    print(json.dumps({'continuation_passed': summary['continuation_passed'], 'averages': averages}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run', 'report'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        write_new(args.out, signature())
    elif args.command == 'smoke':
        for arm in ARMS:
            fit(arm, 7, args.out / arm, updates=2)
    elif args.command == 'run':
        run(args.plan, args.out)
    else:
        report(args.plan, args.run, args.out)


if __name__ == '__main__':
    main()
