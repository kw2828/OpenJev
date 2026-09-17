"""Check source/trace/checkpoint integrity after a terminal SRPO pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rl_doom import ROOT, digest, verify
from stable_baselines3 import PPO

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
m = verify(a.run, 'completed_pilot')
for name, sha in m['source_sha256'].items():
    if digest(ROOT/name) != sha:
        raise ValueError(f'Frozen execution source changed: {name}')
if digest(ROOT/'research/protocols/srpo-doom-v1.json') != m['protocol_sha256']:
    raise ValueError('Protocol changed')
protocol = m['protocol']
checks, steps = [], 0
for rep in range(protocol['replicates']):
    first = None
    for arm in protocol['arms']:
        with np.load(a.run/f'{arm}-{rep}-rollouts.npz') as data:
            expected = protocol['groups_per_fit']*protocol['group_size']+1
            assert len(data['offsets']) == expected
            assert data['offsets'][0] == 0 and np.all(np.diff(data['offsets']) > 0)
            n = int(data['offsets'][-1])
            steps += n
            assert all(len(data[k]) == n and np.isfinite(data[k]).all()
                       for k in ['states', 'actions', 'old_log_probs'])
            end = data['offsets'][protocol['group_size']]
            group = {k: data[k][:end].copy() for k in ['states', 'actions', 'old_log_probs']}
            if first is None:
                first = group
            else:
                assert all(np.array_equal(first[k], group[k]) for k in group)
        receipt = json.loads((a.run/f'{arm}-{rep}-training.json').read_text())
        assert receipt['interactions'] == n and receipt['critic_unchanged']
        checkpoint = PPO.load(a.run/f'{arm}-{rep}.zip', device='cpu')
        assert all(torch.isfinite(v).all() for v in checkpoint.policy.state_dict().values())
        checks.append({'arm': arm, 'replicate': rep, 'transitions': n,
                       'checkpoint_finite': True, 'first_group_matches_paired_arms': True})
assert steps == m['training_interactions']
report = {'status': 'integrity_checks_passed_not_efficacy', 'run_manifest_sha256': digest(a.run/'manifest.json'),
          'audit_source_sha256': digest(__file__), 'execution_sources_unchanged': True,
          'raw_artifact_hashes_verified': True, 'checks': checks,
          'training_interactions': steps, 'evaluation_episodes': m['evaluation_episodes']}
with a.output.open('x') as out:
    json.dump(report, out, indent=2)
    out.write('\n')
print(json.dumps(report, indent=2))
