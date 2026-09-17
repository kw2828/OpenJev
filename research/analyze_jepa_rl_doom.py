"""Validate the frozen JEPA/RL pilot and report descriptive paired contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
from jepa_rl_doom import PROTOCOL, ROOT
from rl_doom import digest, verify


def analyze(root):
    m = verify(root, 'completed_pilot')
    p = m['protocol']
    if digest(PROTOCOL) != m['protocol_sha256']:
        raise ValueError('Protocol changed')
    if any(digest(ROOT/s) != h for s, h in m['source_sha256'].items()):
        raise ValueError('Frozen source changed')
    rows = [json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    if (len(rows) != p['max_evaluation_episodes'] or m['evaluation_episodes'] != len(rows)
            or m['fits_completed'] != len(p['arms'])*p['replicates']
            or m['training_interactions'] != p['max_training_interactions']):
        raise ValueError('Incomplete or over-budget study')
    result = {'status': 'completed_development_pilot_not_confirmation', 'manifest_sha256': digest(root/'manifest.json'),
        'summary': {}, 'contrasts': {}, 'continuation_checks': {}, 'training': {}, 'claim_boundary': p['claim_boundary']}
    for arm in p['arms']:
        fits = [json.loads((root/f'{arm}-{r}-training.json').read_text()) for r in range(p['replicates'])]
        if any(t['interactions'] != p['steps_per_fit'] for t in fits):
            raise ValueError('Unequal interaction budget')
        result['training'][arm] = {'interactions': sum(t['interactions'] for t in fits),
            'total_fit_wall_seconds': sum(t['total_fit_wall_seconds'] for t in fits),
            'discarded_group_interactions': [t['discarded_group_interactions'] for t in fits],
            'updated_groups': [t['updated_groups'] for t in fits],
            'encoding_seconds': sum(g.get('encoding_seconds', 0.) for t in fits for g in t.get('groups', []))}
    rng = np.random.default_rng(p['bootstrap_seed'])
    seeds = rng.integers(len(p['evaluation_seeds']), size=(p['bootstrap_draws'], len(p['evaluation_seeds'])))
    reps = rng.integers(p['replicates'], size=(p['bootstrap_draws'], p['replicates']))
    controls = ['grpo', 'srpo_pixels', 'srpo_random_video', 'ppo_sparse', 'ppo_frozen', 'always_fire']
    for scenario in p['scenarios']:
        matrix = {}
        summary = result['summary'][scenario] = {}
        for arm in [*p['arms'], 'ppo_frozen', 'always_fire']:
            subset = [r for r in rows if r['scenario'] == scenario and r['arm'] == arm]
            count = 1 if arm == 'always_fire' else p['replicates']
            values = np.empty((count, len(p['evaluation_seeds']), 2))
            for rep in range(count):
                for j, seed in enumerate(p['evaluation_seeds']):
                    cell = [r for r in subset if r['seed'] == seed and
                            r['replicate'] == (None if arm == 'always_fire' else rep)]
                    if len(cell) != 1:
                        raise ValueError('Missing/duplicate evaluation cell')
                    values[rep, j] = [cell[0]['kills'], cell[0]['game_seconds']]
            matrix[arm] = values
            summary[arm] = {'kills': float(values[:, :, 0].mean()), 'game_seconds': float(values[:, :, 1].mean()),
                'per_fit_kills': values[:, :, 0].mean(axis=1).tolist(), 'episodes': len(subset),
                'max_episode_p95_ms': max(r['p95_decision_ms'] for r in subset),
                'capped_episodes': sum(r['capped'] for r in subset)}
        contrasts = result['contrasts'][scenario] = {}
        for arm in p['arms']:
            contrasts[arm] = {}
            for control in sorted(set(controls+['ppo_frozen'])):
                if arm == control:
                    continue
                diff = matrix[arm]-matrix[control]
                draws = diff[reps[:, :, None], seeds[:, None, :]].mean(axis=(1, 2))
                contrasts[arm][control] = {metric: {'difference': float(diff[:, :, k].mean()),
                    'exploratory_95_interval': np.quantile(draws[:, k], [.025, .975]).tolist()}
                    for k, metric in enumerate(['kills', 'game_seconds'])}
        checks = result['continuation_checks'][scenario] = {}
        for control in controls:
            c = contrasts['srpo_vjepa'][control]
            checks[control] = {'kills': c['kills']['difference'] >= (.5 if scenario == 'defend_the_center' else -.5),
                               'duration': c['game_seconds']['difference'] >= -.5}
        checks['latency'] = summary['srpo_vjepa']['max_episode_p95_ms'] < 1.
    result['eligible_for_fresh_confirmation'] = all(
        all(v.values()) if isinstance(v, dict) else v
        for checks in result['continuation_checks'].values() for v in checks.values())
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = analyze(args.run)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'eligible': result['eligible_for_fresh_confirmation'], 'summary': result['summary']}, indent=2))
