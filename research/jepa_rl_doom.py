"""Equal-interaction local pilot: frozen V-JEPA rewards and categorical RL variants."""
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
from rl_doom import ROOT, Budget, digest, episode, verify
from rl_doom import SOURCES as RL_SOURCES
from srpo_doom import AlwaysFire, check_time
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure

from openjev.research.policy_variants import actor_loss, advantages
from openjev.research.rl_env import FiringEnv
from openjev.research.self_reference import reference_rewards
from openjev.research.video_reward import (
    FrozenVideoEncoder,
    clip_hash,
    pixel_embedding,
    prepare_frame,
    sample_clip,
)

PROTOCOL = ROOT/'research/protocols/jepa-rl-doom-v1.json'
SOURCES = sorted(set(RL_SOURCES+['research/jepa_rl_doom.py', 'research/analyze_jepa_rl_doom.py',
    'research/srpo_doom.py', 'src/openjev/research/self_reference.py',
    'src/openjev/research/policy_variants.py', 'src/openjev/research/video_reward.py']))


class SparseGoalEnv(FiringEnv):
    def __init__(self, success_kills=10, sparse_trace=None, **kwargs):
        super().__init__(**kwargs)
        self.success_kills, self.sparse_trace = success_kills, sparse_trace

    def step(self, action):
        before = self.obs.to_dict()
        state, dense, done, truncated, info = super().step(action)
        reward = float(done and info['result']['kills'] >= self.success_kills)
        if self.sparse_trace:
            self.sparse_trace.write(json.dumps({**self.label, 'seed': self.game_seed,
                'step': self.steps-1, 'observation': before, 'action': int(action),
                'reward': reward, 'dense_environment_reward': dense, 'terminated': done})+'\n')
        return state, reward, done, truncated, info


def make_actor_critic(arm, env, source_model, p, rep):
    cls = PPO if arm == 'ppo_sparse' else A2C
    config = p['ppo'] if cls is PPO else p['a2c']
    model = cls('MlpPolicy', env, device='cpu', seed=p['model_seeds'][rep],
                policy_kwargs={'net_arch': [64, 64]}, **config)
    # Copy actor only. The critic and optimizer are fresh for this sparse objective.
    state = model.policy.state_dict()
    for key, value in source_model.policy.state_dict().items():
        if key.startswith(('mlp_extractor.policy_net.', 'action_net.')):
            state[key] = value.clone()
    model.policy.load_state_dict(state)
    return model


def rollout(model, seed, p, trace, label, deadline, remaining, video):
    env = FiringEnv('history', p['training_scenario'], horizon=p['horizon'], tics=p['tics'],
                    trace=trace, label=label)
    states, actions, logs, frames = [], [], [], []
    try:
        state, _ = env.reset(options={'game_seed': seed})
        done = False
        for _ in range(remaining):
            check_time(deadline)
            with torch.no_grad():
                dist = model.policy.get_distribution(torch.as_tensor(state[None])).distribution
                action = dist.sample()
                log = dist.log_prob(action)
            states.append(state.copy())
            actions.append(int(action.item()))
            logs.append(float(log.item()))
            if video:
                frames.append(prepare_frame(env.doom.frame()))
            state, _, done, _, info = env.step(actions[-1])
            if done:
                break
        result = {'states': np.asarray(states), 'actions': np.asarray(actions),
                  'old_log_probs': np.asarray(logs, dtype=np.float32), 'complete': done,
                  'result': info['result'] if done else env.result()}
        if video and done:
            result['clip'], indices = sample_clip(frames)
            result['clip_indices'] = indices.tolist()
            result['clip_sha256'] = clip_hash(result['clip'])
        return result
    finally:
        env.close()


def update(model, reference, optimizer, trajectories, rewards, variant, p):
    advantage = advantages(rewards, variant)
    if np.max(np.abs(advantage)) < 1e-8:
        return {'updated': False, 'reason': 'homogeneous_rewards', 'epochs': 0}
    lengths = [len(t['actions']) for t in trajectories]
    states = torch.as_tensor(np.concatenate([t['states'] for t in trajectories]))
    actions = torch.as_tensor(np.concatenate([t['actions'] for t in trajectories]))
    logs = torch.as_tensor(np.concatenate([t['old_log_probs'] for t in trajectories]))
    adv = torch.as_tensor(np.repeat(advantage, lengths), dtype=torch.float32)
    with torch.no_grad():
        ref = reference.get_distribution(states).distribution.probs.detach()
    history = []
    for _ in range(1 if variant == 'rloo' else p['epochs']):
        loss, metrics = actor_loss(model.policy, states, actions, logs, ref, adv, lengths,
                                  variant, horizon=p['horizon'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
        history.append({'loss': float(loss.detach()), **metrics})
    return {'updated': True, 'epochs': len(history), 'history': history}


def train_group(arm, rep, source, p, output, deadline, encoder):
    model = PPO.load(source/f'ppo_history-{rep}.zip', device='cpu')
    reference = copy.deepcopy(model.policy).eval()
    reference.requires_grad_(False)
    model.policy.set_training_mode(False)
    torch.manual_seed(p['model_seeds'][rep])
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=p['learning_rate'])
    variant = 'grpo' if arm.startswith('srpo_') else arm
    video = arm.startswith('srpo_')
    steps, used, updated, groups, offsets = 0, 0, 0, [], [0]
    arrays = {k: [] for k in ('states', 'actions', 'old_log_probs')}
    embedding_records = {}
    started = time.monotonic()
    with gzip.open(output/f'{arm}-{rep}-trace.jsonl.gz', 'wt') as trace:
        while steps < p['steps_per_fit']:
            group = len(groups)
            trajectories = []
            for sample in range(p['group_size']):
                if steps == p['steps_per_fit']:
                    break
                t = rollout(model, p['training_seed_starts'][rep]+group, p, trace,
                    {'arm': arm, 'replicate': rep, 'group': group, 'sample': sample}, deadline,
                    p['steps_per_fit']-steps, video)
                trajectories.append(t)
                steps += len(t['actions'])
                offsets.append(steps)
                for k, values in arrays.items():
                    values.append(t[k])
            complete = len(trajectories) == p['group_size'] and all(t['complete'] for t in trajectories)
            record = {'group': group, 'seed': p['training_seed_starts'][rep]+group, 'complete': complete,
                'lengths': [len(t['actions']) for t in trajectories],
                'episodes': [t['result'] for t in trajectories],
                'trajectory_complete': [t['complete'] for t in trajectories]}
            if complete:
                good = np.array([t['result']['kills'] >= p['success_kills'] for t in trajectories])
                rewards = good.astype(float)
                encode_start = time.monotonic()
                if video:
                    record['clips'] = [{k: t[k] for k in ('clip_indices', 'clip_sha256')} for t in trajectories]
                    if good.any() and (~good).any():
                        embeddings = np.stack([encoder(t['clip']) for t in trajectories])
                        embedding_records[f'group_{group}'] = embeddings
                        rewards = reference_rewards(embeddings, good, p['failure_reward_weight'])
                    else:
                        record['encoding_skipped'] = 'homogeneous_success_labels'
                record['encoding_seconds'] = time.monotonic()-encode_start
                check_time(deadline)
                result = update(model, reference, optimizer, trajectories, rewards, variant, p)
                used += sum(record['lengths'])
                updated += result['updated']
                record.update(successes=int(good.sum()), rewards=rewards.tolist(), optimization=result)
            else:
                record['optimization'] = {'updated': False, 'epochs': 0, 'reason': 'incomplete_group_budget_end'}
            groups.append(record)
            print(json.dumps({'arm': arm, 'replicate': rep, 'group': group, 'interactions': steps,
                              'complete': complete}), flush=True)
    for key, value in model.policy.state_dict().items():
        if 'value' in key and not torch.equal(value, reference.state_dict()[key]):
            raise ValueError('Critic changed')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite parameters')
    model.num_timesteps += steps
    np.savez_compressed(output/f'{arm}-{rep}-rollouts.npz', offsets=offsets,
                        **{k: np.concatenate(v) for k, v in arrays.items()})
    if video:
        np.savez_compressed(output/f'{arm}-{rep}-embeddings.npz', **embedding_records)
    return model, {'interactions': steps, 'complete_group_interactions': used,
        'discarded_group_interactions': steps-used, 'updated_groups': updated, 'groups': groups,
        'critic_unchanged': True, 'wall_seconds': time.monotonic()-started}


def train_sb3(arm, rep, source, p, output, deadline):
    started = time.monotonic()
    with gzip.open(output/f'{arm}-{rep}-trace.jsonl.gz', 'wt') as trace:
        env = SparseGoalEnv(kind='history', scenario=p['training_scenario'],
            seed_start=p['training_seed_starts'][rep], horizon=p['horizon'], tics=p['tics'],
            success_kills=p['success_kills'], sparse_trace=trace, label={'arm': arm, 'replicate': rep})
        try:
            initial = PPO.load(source/f'ppo_history-{rep}.zip', device='cpu')
            model = make_actor_critic(arm, env, initial, p, rep)
            model.set_logger(configure(str(output/f'{arm}-{rep}-logs'), ['csv']))
            model.learn(total_timesteps=p['steps_per_fit'], callback=Budget(deadline))
            if model.num_timesteps != p['steps_per_fit']:
                raise ValueError('Interaction budget mismatch')
            receipt = {'interactions': model.num_timesteps, 'complete_group_interactions': None,
                'discarded_group_interactions': 0, 'updated_groups': None,
                'episodes': env.training_episodes, 'unfinished_episode': None if env.done else env.result(),
                'wall_seconds': time.monotonic()-started, 'fresh_critic': True}
            return model, receipt
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    p = json.loads(PROTOCOL.read_text())
    if args.smoke:
        p.update(replicates=1, steps_per_fit=512, group_size=2, horizon=40, success_kills=2,
                 training_seed_starts=[189000], evaluation_seeds=[189900])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = ROOT/p['initial_run']
    parent = verify(source, 'completed_development')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+p['max_wall_seconds']
    manifest = {'status': 'running', 'smoke': args.smoke, 'protocol': p,
        'protocol_sha256': digest(PROTOCOL), 'source_sha256': {s: digest(ROOT/s) for s in SOURCES},
        'initial_manifest_sha256': digest(source/'manifest.json'), 'initial_protocol_sha256': parent['protocol_sha256'],
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {n: importlib.metadata.version(n) for n in
                     ('torch', 'numpy', 'stable-baselines3', 'vizdoom', 'gymnasium', 'transformers', 'pillow')},
        'fits_completed': 0, 'training_interactions': 0, 'evaluation_episodes': 0}
    mp = args.output/'manifest.json'
    mp.write_text(json.dumps(manifest, indent=2)+'\n')
    models, pretrained = {}, None
    try:
        jobs = [(arm, rep) for arm in p['arms'] for rep in range(p['replicates'])]
        np.random.default_rng(p['order_seed']).shuffle(jobs)
        for arm, rep in jobs:
            check_time(deadline)
            print(json.dumps({'training': arm, 'replicate': rep}), flush=True)
            fit_start = time.monotonic()
            encoder = pixel_embedding
            if arm == 'srpo_vjepa':
                if pretrained is None:
                    pretrained = FrozenVideoEncoder()
                encoder = pretrained
            elif arm == 'srpo_random_video':
                encoder = FrozenVideoEncoder(random_seed=p['random_encoder_seeds'][rep])
            if isinstance(encoder, FrozenVideoEncoder):
                (args.output/f'{arm}-{rep}-encoder.json').write_text(json.dumps(encoder.receipt, indent=2)+'\n')
            if arm in ('ppo_sparse', 'a2c_sparse'):
                model, receipt = train_sb3(arm, rep, source, p, args.output, deadline)
            else:
                model, receipt = train_group(arm, rep, source, p, args.output, deadline, encoder)
            model.save(args.output/f'{arm}-{rep}.zip')
            cls = A2C if arm == 'a2c_sparse' else PPO
            restored = cls.load(args.output/f'{arm}-{rep}.zip', device='cpu')
            for key, value in model.policy.state_dict().items():
                if not torch.equal(value, restored.policy.state_dict()[key]):
                    raise ValueError('Checkpoint reload mismatch')
            receipt.update(total_fit_wall_seconds=time.monotonic()-fit_start,
                           initial_checkpoint_sha256=digest(source/f'ppo_history-{rep}.zip'))
            (args.output/f'{arm}-{rep}-training.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
            del encoder
            manifest['fits_completed'] += 1
            manifest['training_interactions'] += receipt['interactions']
            models[arm, rep] = restored
            mp.write_text(json.dumps(manifest, indent=2)+'\n')
        for rep in range(p['replicates']):
            models['ppo_frozen', rep] = PPO.load(source/f'ppo_history-{rep}.zip', device='cpu')
        jobs = [(s, seed, arm, rep) for s in p['scenarios'] for seed in p['evaluation_seeds']
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
        check_time(deadline)
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
