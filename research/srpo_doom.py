"""Frozen, bounded structured-observation SRPO adaptation and matched ablations."""
import argparse
import copy
import gzip
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from rl_doom import ROOT, digest, episode, verify
from rl_doom import SOURCES as RL_SOURCES
from stable_baselines3 import PPO

from openjev.domain import Observation
from openjev.research.rl_env import FiringEnv, current_features
from openjev.research.self_reference import (
    DynamicsEncoder,
    group_advantages,
    policy_loss,
    raw_trajectory,
    reference_rewards,
)

PROTOCOL = ROOT/'research/protocols/srpo-doom-v1.json'
SOURCES = sorted(set(RL_SOURCES + ['research/srpo_doom.py', 'research/analyze_srpo_doom.py',
                                 'src/openjev/research/self_reference.py']))


def check_time(deadline):
    if time.monotonic() > deadline:
        raise RuntimeError('Frozen wall budget exhausted; no efficacy claim')


def fit_encoder(trace_path, seed, updates, deadline):
    """Use consecutive pre-action observations only; never cross episode boundaries."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs, actions, ys, previous = [], [], [], None
    with gzip.open(trace_path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            x = current_features(Observation(**row['observation']))[1:].astype(np.float32)
            if previous is not None:
                old, old_x = previous
                if (old['seed'] == row['seed'] and old['step']+1 == row['step']
                        and not old['terminated']):
                    xs.append(old_x)
                    actions.append(float(old['issued_fire']))
                    ys.append(x)
            previous = row, x
    x, a, y = (torch.as_tensor(np.asarray(v), dtype=torch.float32) for v in (xs, actions, ys))
    model = DynamicsEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    losses = []
    for _ in range(updates):
        check_time(deadline)
        indices = rng.integers(len(x), size=128)
        loss = torch.nn.functional.mse_loss(model(x[indices], a[indices]), y[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {'training_pairs': len(x), 'updates': updates, 'losses': losses,
                   'data_sha256': digest(trace_path),
                   'note': 'Training loss only, no out-of-sample dynamics accuracy claim'}


def rollout(model, game_seed, p, trace, label, deadline):
    env = FiringEnv('history', p['training_scenario'], horizon=p['horizon'], tics=p['tics'],
                    trace=trace, label=label)
    states, actions, log_probs, observations = [], [], [], []
    try:
        state, _ = env.reset(options={'game_seed': game_seed})
        while True:
            check_time(deadline)
            with torch.no_grad():
                dist = model.policy.get_distribution(torch.as_tensor(state[None])).distribution
                action = dist.sample()
                log_prob = dist.log_prob(action)
            states.append(state.copy())
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            observations.append(state[1:7].copy())
            state, _, done, _, info = env.step(actions[-1])
            if done:
                break
        return {'states': np.asarray(states), 'actions': np.asarray(actions),
                'old_log_probs': np.asarray(log_probs, dtype=np.float32),
                'observations': np.asarray(observations), 'result': info['result']}
    finally:
        env.close()


def update(model, reference, optimizer, trajectories, rewards, p):
    advantage = group_advantages(rewards)
    if np.max(np.abs(advantage)) < 1e-8:
        return {'updated': False, 'epochs': 0, 'reason': 'homogeneous_rewards'}
    lengths = [len(t['actions']) for t in trajectories]
    states = torch.as_tensor(np.concatenate([t['states'] for t in trajectories]))
    actions = torch.as_tensor(np.concatenate([t['actions'] for t in trajectories]))
    old_log_probs = torch.as_tensor(np.concatenate([t['old_log_probs'] for t in trajectories]))
    adv = torch.as_tensor(np.repeat(advantage, lengths), dtype=torch.float32)
    with torch.no_grad():
        ref_probs = reference.get_distribution(states).distribution.probs.detach()
    history = []
    for _ in range(p['epochs']):
        loss, metrics = policy_loss(model.policy, states, actions, old_log_probs, ref_probs,
                                   adv, lengths, **p['loss'])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite actor loss')
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
        if not torch.isfinite(norm):
            raise ValueError('Non-finite actor gradient')
        optimizer.step()
        history.append({'loss': float(loss.detach()), **metrics})
    return {'updated': True, 'epochs': len(history), 'history': history}


class AlwaysFire:
    def predict(self, state, deterministic=True):
        return 1, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    p = json.loads(PROTOCOL.read_text())
    if args.smoke:
        p.update(replicates=1, groups_per_fit=2, group_size=4, encoder_updates=2,
                 training_seed_starts=[159500], evaluation_seeds=[159900, 159901])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = ROOT/p['initial_run']
    parent = verify(source, 'completed_development')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+p['max_wall_seconds']
    manifest = {'status': 'running', 'smoke': args.smoke, 'protocol': p,
                'protocol_sha256': digest(PROTOCOL), 'source_sha256': {s: digest(ROOT/s) for s in SOURCES},
                'initial_manifest_sha256': digest(source/'manifest.json'),
                'initial_protocol_sha256': parent['protocol_sha256'],
                'python': platform.python_version(), 'platform': platform.platform(),
                'versions': {n: importlib.metadata.version(n) for n in
                             ('torch', 'numpy', 'stable-baselines3', 'vizdoom', 'gymnasium')},
                'fits_completed': 0, 'training_interactions': 0, 'evaluation_episodes': 0}
    mp = args.output/'manifest.json'
    mp.write_text(json.dumps(manifest, indent=2)+'\n')
    models, encoders = {}, {}
    try:
        for rep in range(p['replicates']):
            start = time.monotonic()
            encoder, receipt = fit_encoder(source/f'ppo_history-{rep}-training.jsonl.gz',
                                            p['model_seeds'][rep], p['encoder_updates'], deadline)
            receipt['wall_seconds'] = time.monotonic()-start
            torch.save(encoder.state_dict(), args.output/f'encoder-{rep}.pt')
            (args.output/f'encoder-{rep}.json').write_text(json.dumps(receipt, indent=2)+'\n')
            encoders[rep] = encoder
        jobs = [(arm, rep) for arm in p['arms'] for rep in range(p['replicates'])]
        np.random.default_rng(p['order_seed']).shuffle(jobs)
        for arm, rep in jobs:
            print(json.dumps({'training': arm, 'replicate': rep}), flush=True)
            model = PPO.load(source/f'ppo_history-{rep}.zip', device='cpu')
            reference = copy.deepcopy(model.policy).eval()
            for param in reference.parameters():
                param.requires_grad_(False)
            model.policy.set_training_mode(False)
            torch.manual_seed(p['model_seeds'][rep])
            optimizer = torch.optim.Adam(model.policy.parameters(), lr=p['learning_rate'])
            start = time.monotonic()
            steps, updated, groups = 0, 0, []
            arrays = {k: [] for k in ('states', 'actions', 'old_log_probs')}
            offsets = [0]
            with gzip.open(args.output/f'{arm}-{rep}-trace.jsonl.gz', 'wt') as trace:
                for group in range(p['groups_per_fit']):
                    trajectories = [rollout(model, p['training_seed_starts'][rep]+group, p, trace,
                                            {'arm': arm, 'replicate': rep, 'group': group, 'sample': sample}, deadline)
                                    for sample in range(p['group_size'])]
                    good = [t['result']['kills'] >= p['success_kills'] for t in trajectories]
                    if arm == 'group_binary':
                        rewards = np.asarray(good, dtype=float)
                    else:
                        embeddings = np.asarray([(encoders[rep].trajectory(t['observations'])
                                                  if arm == 'srpo_latent' else raw_trajectory(t['observations']))
                                                 for t in trajectories])
                        rewards = reference_rewards(embeddings, good, p['failure_reward_weight'])
                    result = update(model, reference, optimizer, trajectories, rewards, p)
                    updated += result['updated']
                    for t in trajectories:
                        steps += len(t['actions'])
                        offsets.append(steps)
                        for k, values in arrays.items():
                            values.append(t[k])
                    groups.append({'group': group, 'seed': p['training_seed_starts'][rep]+group,
                                   'successes': sum(good), 'rewards': rewards.tolist(),
                                   'episodes': [t['result'] for t in trajectories], 'optimization': result})
                    print(json.dumps({'arm': arm, 'replicate': rep, 'groups': group+1,
                                      'interactions': steps, 'successes': sum(good)}), flush=True)
            # Actor-only optimization must not silently train the inherited critic.
            for key, value in model.policy.state_dict().items():
                if 'value' in key and not torch.equal(value, reference.state_dict()[key]):
                    raise ValueError('Critic unexpectedly changed')
                if not torch.isfinite(value).all():
                    raise ValueError('Non-finite checkpoint')
            model.num_timesteps += steps
            model.save(args.output/f'{arm}-{rep}.zip')
            np.savez_compressed(args.output/f'{arm}-{rep}-rollouts.npz', offsets=offsets,
                                **{k: np.concatenate(v) for k, v in arrays.items()})
            (args.output/f'{arm}-{rep}-training.json').write_text(json.dumps({
                'groups': groups, 'interactions': steps, 'updated_groups': updated,
                'wall_seconds': time.monotonic()-start, 'initial_checkpoint_sha256': digest(source/f'ppo_history-{rep}.zip'),
                'critic_unchanged': True}, indent=2, allow_nan=False)+'\n')
            manifest['fits_completed'] += 1
            manifest['training_interactions'] += steps
            models[arm, rep] = model
            mp.write_text(json.dumps(manifest, indent=2)+'\n')
        for rep in range(p['replicates']):
            models['ppo_frozen', rep] = PPO.load(source/f'ppo_history-{rep}.zip', device='cpu')
        jobs = [(scenario, seed, arm, rep) for scenario in p['scenarios'] for seed in p['evaluation_seeds']
                for arm in [*p['arms'], 'ppo_frozen'] for rep in range(p['replicates'])]
        jobs.extend((s, seed, 'always_fire', None) for s in p['scenarios'] for seed in p['evaluation_seeds'])
        np.random.default_rng(p['order_seed']+1).shuffle(jobs)
        with gzip.open(args.output/'evaluation-trace.jsonl.gz', 'wt') as trace, (args.output/'episodes.jsonl').open('x') as out:
            for scenario, seed, arm, rep in jobs:
                check_time(deadline)
                model = AlwaysFire() if arm == 'always_fire' else models[arm, rep]
                row = episode(scenario, seed, arm, p, model, 'history', rep, trace)
                out.write(json.dumps(row, allow_nan=False)+'\n')
                out.flush()
                manifest['evaluation_episodes'] += 1
                if manifest['evaluation_episodes'] % 100 == 0:
                    print(json.dumps({'evaluation_episodes': manifest['evaluation_episodes']}), flush=True)
        manifest['status'] = 'completed_smoke_no_efficacy' if args.smoke else 'completed_pilot'
    except BaseException as e:
        manifest.update(status='failed_no_efficacy_claim', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic()-started
        manifest['artifact_sha256'] = {str(f.relative_to(args.output)): digest(f) for f in args.output.rglob('*')
                                      if f.is_file() and f != mp}
        mp.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
