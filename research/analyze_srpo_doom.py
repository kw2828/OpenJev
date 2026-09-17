"""Audit and summarize a completed SRPO adaptation pilot; never confirm efficacy."""
import argparse
import json
from pathlib import Path

import numpy as np
from rl_doom import digest, verify


def analyze(root):
    manifest = verify(root, 'completed_pilot')
    p = manifest['protocol']
    rows = [json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    arms = [*p['arms'], 'ppo_frozen', 'always_fire']
    expected = len(p['scenarios'])*len(p['evaluation_seeds'])*(p['replicates']*4+1)
    if len(rows) != expected or manifest['fits_completed'] != len(p['arms'])*p['replicates']:
        raise ValueError('Incomplete pilot')
    if manifest['training_interactions'] > p['max_training_interactions']:
        raise ValueError('Interaction bound exceeded')
    rng = np.random.default_rng(p['bootstrap_seed'])
    seeds = rng.integers(len(p['evaluation_seeds']), size=(p['bootstrap_draws'], len(p['evaluation_seeds'])))
    reps = rng.integers(p['replicates'], size=(p['bootstrap_draws'], p['replicates']))
    result = {'status': 'completed_development_pilot_not_confirmation',
              'manifest_sha256': digest(root/'manifest.json'), 'analyzer_sha256': digest(__file__),
              'summary': {}, 'contrasts': {}, 'continuation_checks': {}, 'training': {}}
    for arm in p['arms']:
        training = [json.loads((root/f'{arm}-{r}-training.json').read_text()) for r in range(p['replicates'])]
        result['training'][arm] = {
            'interactions': sum(t['interactions'] for t in training),
            'actor_training_wall_seconds': sum(t['wall_seconds'] for t in training),
            'updated_groups': [t['updated_groups'] for t in training],
            'homogeneous_groups': [p['groups_per_fit']-t['updated_groups'] for t in training],
            'encoder_training_wall_seconds': sum(json.loads((root/f'encoder-{r}.json').read_text())['wall_seconds']
                                                 for r in range(p['replicates'])) if arm == 'srpo_latent' else 0.,
        }
    for scenario in p['scenarios']:
        matrix = {}
        result['summary'][scenario], result['contrasts'][scenario] = {}, {}
        for arm in arms:
            count = 1 if arm == 'always_fire' else p['replicates']
            values = np.empty((count, len(p['evaluation_seeds']), 2))
            subset = [r for r in rows if r['scenario'] == scenario and r['arm'] == arm]
            for rep in range(count):
                for j, seed in enumerate(p['evaluation_seeds']):
                    cell = [r for r in subset if r['seed'] == seed and r['replicate'] ==
                            (None if arm == 'always_fire' else rep)]
                    if len(cell) != 1:
                        raise ValueError('Missing or duplicate evaluation cell')
                    values[rep, j] = [cell[0]['kills'], cell[0]['game_seconds']]
            matrix[arm] = values
            result['summary'][scenario][arm] = {
                'kills': float(values[:, :, 0].mean()), 'game_seconds': float(values[:, :, 1].mean()),
                'per_fit_kills': values[:, :, 0].mean(axis=1).tolist(), 'episodes': len(subset),
                'max_episode_p95_ms': max(r['p95_decision_ms'] for r in subset),
                'capped_episodes': sum(r['capped'] for r in subset)}
        checks = {}
        for control in ['group_binary', 'srpo_raw', 'ppo_frozen', 'always_fire']:
            diff = matrix['srpo_latent']-matrix[control]
            draws = diff[reps[:, :, None], seeds[:, None, :]].mean(axis=(1, 2))
            result['contrasts'][scenario][control] = {
                metric: {'difference': float(diff[:, :, k].mean()),
                         'exploratory_95_interval': np.quantile(draws[:, k], [.025, .975]).tolist()}
                for k, metric in enumerate(['kills', 'game_seconds'])}
            checks[control] = {'kills': float(diff[:, :, 0].mean()) >=
                              (.5 if scenario == 'defend_the_center' else -.5),
                              'duration': float(diff[:, :, 1].mean()) >= -.5}
        checks['latency'] = result['summary'][scenario]['srpo_latent']['max_episode_p95_ms'] < 1.
        result['continuation_checks'][scenario] = checks
    result['eligible_for_fresh_confirmation'] = all(
        (all(value.values()) if isinstance(value, dict) else value)
        for checks in result['continuation_checks'].values() for value in checks.values())
    result['claim_boundary'] = ('Exploratory pilot with three warm-start fits. Equal trajectory-group budgets, '
        'not equal interactions or compute. Structured dynamics adaptation, not V-JEPA/OpenVLA replication. '
        'No automatic confirmation or efficacy/novelty claim; preserve negative results.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(result, indent=2))
