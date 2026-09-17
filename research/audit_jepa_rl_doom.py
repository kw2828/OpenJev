"""Independent artifact, budget, reward and actor-update replay audit."""
import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from jepa_rl_doom import ROOT, update
from rl_doom import digest, verify
from stable_baselines3 import A2C, PPO

from openjev.research.rl_env import FiringEnv
from openjev.research.self_reference import reference_rewards
from openjev.research.video_reward import clip_hash, prepare_frame, sample_clip


def audit(root):
    m = verify(root, 'completed_pilot')
    p = m['protocol']
    receipts = []
    torch.set_num_threads(1)
    for arm in p['arms']:
        for rep in range(p['replicates']):
            t = json.loads((root/f'{arm}-{rep}-training.json').read_text())
            with gzip.open(root/f'{arm}-{rep}-trace.jsonl.gz', 'rt') as stream:
                trace = [json.loads(line) for line in stream]
            assert len(trace) == t['interactions'] == p['steps_per_fit']
            cls = A2C if arm == 'a2c_sparse' else PPO
            final = cls.load(root/f'{arm}-{rep}.zip', device='cpu')
            assert all(torch.isfinite(v).all() for v in final.policy.state_dict().values())
            receipt = {'arm': arm, 'replicate': rep, 'trace_interactions': len(trace), 'finite_checkpoint': True}
            if 'groups' in t:
                initial = ROOT/p['initial_run']/f'ppo_history-{rep}.zip'
                assert digest(initial) == t['initial_checkpoint_sha256']
                model = PPO.load(initial, device='cpu')
                reference = copy.deepcopy(model.policy).eval()
                reference.requires_grad_(False)
                optimizer = torch.optim.Adam(model.policy.parameters(), lr=p['learning_rate'])
                data = np.load(root/f'{arm}-{rep}-rollouts.npz')
                embeddings = np.load(root/f'{arm}-{rep}-embeddings.npz') if arm.startswith('srpo_') else None
                offsets = data['offsets']
                assert offsets[0] == 0 and offsets[-1] == len(trace)
                index, used, updates = 0, 0, 0
                for group in t['groups']:
                    trajectories = []
                    for length in group['lengths']:
                        lo, hi = offsets[index:index+2]
                        assert hi-lo == length
                        trajectories.append({k: data[k][lo:hi] for k in ('states', 'actions', 'old_log_probs')})
                        assert all(r['group'] == group['group'] and r['seed'] == group['seed'] for r in trace[lo:hi])
                        index += 1
                    with torch.no_grad():
                        for trajectory in trajectories:
                            states = torch.as_tensor(trajectory['states'])
                            actions = torch.as_tensor(trajectory['actions'])
                            logs = model.policy.get_distribution(states).distribution.log_prob(actions).numpy()
                            np.testing.assert_allclose(logs, trajectory['old_log_probs'], atol=2e-5, rtol=2e-5)
                    if not group['complete']:
                        assert group is t['groups'][-1]
                        assert not group['optimization']['updated']
                        continue
                    assert all(group['trajectory_complete']) and len(trajectories) == p['group_size']
                    good = np.array([e['kills'] >= p['success_kills'] for e in group['episodes']])
                    assert int(good.sum()) == group['successes']
                    rewards = good.astype(float)
                    if embeddings is not None and good.any() and (~good).any():
                        z = embeddings[f"group_{group['group']}"]
                        np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1., atol=1e-5)
                        rewards = reference_rewards(z, good, p['failure_reward_weight'])
                    np.testing.assert_allclose(rewards, group['rewards'], atol=1e-8)
                    variant = 'grpo' if arm.startswith('srpo_') else arm
                    result = update(model, reference, optimizer, trajectories, rewards, variant, p)
                    assert result['epochs'] == group['optimization']['epochs']
                    assert result['updated'] == group['optimization']['updated']
                    updates += result['updated']
                    used += sum(group['lengths'])
                assert index == len(offsets)-1
                assert used == t['complete_group_interactions']
                assert len(trace)-used == t['discarded_group_interactions']
                assert updates == t['updated_groups']
                worst = max(float((v-final.policy.state_dict()[k]).abs().max())
                            for k, v in model.policy.state_dict().items())
                assert worst < 2e-5, worst
                if embeddings is not None and embeddings.files:
                    first_group = int(embeddings.files[0].split('_')[1])
                    group = t['groups'][first_group]
                    lo = sum(sum(g['lengths']) for g in t['groups'][:first_group])
                    length = group['lengths'][0]
                    env = FiringEnv('history', p['training_scenario'], horizon=p['horizon'], tics=p['tics'])
                    frames = []
                    try:
                        state, _ = env.reset(options={'game_seed': group['seed']})
                        for j in range(lo, lo+length):
                            np.testing.assert_array_equal(state, data['states'][j])
                            frames.append(prepare_frame(env.doom.frame()))
                            state, _, done, _, info = env.step(int(data['actions'][j]))
                        assert done and info['result']['kills'] == group['episodes'][0]['kills']
                    finally:
                        env.close()
                    clip, indices = sample_clip(frames)
                    assert clip_hash(clip) == group['clips'][0]['clip_sha256']
                    assert indices.tolist() == group['clips'][0]['clip_indices']
                    receipt['first_encoded_trajectory_frame_hash_replayed'] = True
                receipt.update(actor_update_replay_max_abs_error=worst, rewards_recomputed=True,
                               old_action_log_probabilities_verified=True, incomplete_group_excluded=True)
            else:
                terminal = {e['seed']: e for e in t['episodes']}
                for row in trace:
                    expected = float(row['terminated'] and terminal[row['seed']]['kills'] >= p['success_kills'])
                    assert row['reward'] == expected
                receipt['sparse_terminal_reward_verified'] = True
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    return {'status': 'passed', 'manifest_sha256': digest(root/'manifest.json'),
        'audit_source_sha256': digest(__file__), 'checks': receipts,
        'scope': 'Checks hashes, exact trace budgets, saved embeddings/rewards, old action probabilities, '
                 'group actor-update replay, and sparse reward traces. Does not prove visual semantic quality '
                 'or reproduce SB3 optimizer training from scratch.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.run)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
