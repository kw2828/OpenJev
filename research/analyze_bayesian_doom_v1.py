"""Verify and summarize the frozen pilot. CIs are exploratory crossed bootstraps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openjev.research.bayesian import sigmoid


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'completed_development_pilot':
        raise ValueError('Only a completed pilot can be analyzed')
    for name, expected in manifest['artifact_sha256'].items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Artifact changed: {name}')
    result = json.loads((root/'results.json').read_text())
    episodes = result['episodes']
    protocol = manifest['protocol']
    output = {'status': 'development_results_not_confirmation', 'run_manifest_sha256':
              hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
              'episodes_completed': manifest['episodes_completed'], 'wall_seconds': manifest['wall_seconds'],
              'ci_method': '10000 paired crossed bootstrap draws over 5 training replicates and 10 episode seeds; exploratory unadjusted 95% intervals, seed 8675309',
              'summary': {}, 'paired_differences': {}, 'common_audit': result['common_audit']}
    rng = np.random.default_rng(8675309)
    replicate_indices = rng.integers(0, 5, size=(10000, 5))
    seed_indices = rng.integers(0, 10, size=(10000, 10))
    for scenario in protocol['evaluation_scenarios']:
        output['summary'][scenario] = {}
        matrices = {}
        for mode in protocol['policies']:
            rows = [r for r in episodes if r['scenario'] == scenario and r['policy'] == mode]
            metrics = ['utility', 'kills', 'shots', 'hits', 'successful_firing_windows',
                       'health', 'game_seconds', 'median_decision_ms']
            summary = {metric: float(np.mean([r[metric] for r in rows])) for metric in metrics}
            summary['episodes'] = len(rows)
            summary['truncated_episodes'] = sum(r['truncated'] for r in rows)
            matrix = np.zeros((5, 10))
            for replicate in range(5):
                for index, seed in enumerate(protocol['evaluation_episode_seeds']):
                    values = [r['utility'] for r in rows if r['seed'] == seed and
                              (r['replicate'] == replicate or r['replicate'] is None)]
                    if len(values) != 1:
                        raise ValueError('Missing or duplicate paired cell')
                    matrix[replicate, index] = values[0]
            summary['utility_by_training_replicate'] = list(matrix.mean(axis=1))
            matrices[mode] = matrix
            output['summary'][scenario][mode] = summary
        output['paired_differences'][scenario] = {}
        for mode in ['bayes_mean_hit', 'bayes_lower_hit', 'rules', 'tiny_imitation']:
            difference = matrices[mode]-matrices['map_hit']
            draws = difference[replicate_indices[:, :, None], seed_indices[:, None, :]].mean(axis=(1, 2))
            output['paired_differences'][scenario][mode+'_minus_map'] = {
                'mean_utility_difference': float(difference.mean()),
                'exploratory_95_interval': np.quantile(draws, [.025, .975]).tolist()}
    train = [[] for _ in range(5)]
    with (root/'trace.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['policy'] == 'collect' and row['replicate'] is not None and row['outcome'] is not None:
                train[row['replicate']].append(row)
    output['fit_checks'] = []
    for index, rows in enumerate(train):
        with np.load(root/f'posterior-{index}.npz', allow_pickle=False) as model:
            x = np.asarray([r['x'] for r in rows])
            y = np.asarray([r['outcome'] for r in rows])
            gradient = x.T @ (sigmoid(x @ model['mean'])-y) + protocol['prior_precision']*model['mean']
            output['fit_checks'].append({'replicate': index, 'n': len(rows),
                'gradient_norm': float(np.linalg.norm(gradient)),
                'covariance_min_eigenvalue': float(np.linalg.eigvalsh(model['covariance']).min())})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'summary': result['summary'], 'paired_differences': result['paired_differences']}, indent=2))
