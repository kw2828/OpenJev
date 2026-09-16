"""Evaluate the predeclared continuation gate; never tune the rule after a run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'completed_frozen_memory_ablation':
        raise ValueError('Incomplete run')
    for name, digest in manifest['artifact_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed artifact: {name}')
    protocol = manifest['protocol']
    result = json.loads((root/'results.json').read_text())
    seeds = protocol['evaluation_episode_seeds']
    reps = protocol['replicates']
    rng = np.random.default_rng(protocol['bootstrap_rng_seed'])
    ri = rng.integers(0,reps,size=(protocol['bootstrap_draws'],reps))
    si = rng.integers(0,len(seeds),size=(protocol['bootstrap_draws'],len(seeds)))
    out = {'status':'frozen_development_ablation','run_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
           'episodes_completed':manifest['episodes_completed'],'wall_seconds':manifest['wall_seconds'],
           'summary':{},'primary':{},'secondary_bayes_minus_map':{},
           'common_audit':result['common_audit'],'fit_checks':result['fit_checks']}
    decisions = []
    for scenario in protocol['evaluation_scenarios']:
        matrices = {}
        out['summary'][scenario] = {}
        for policy in protocol['policies']:
            rows = [r for r in result['episodes'] if r['scenario']==scenario and r['policy']==policy]
            assert len(rows) == len(seeds)*(1 if policy in ('rules','tiny_imitation') else reps)
            matrices[policy] = {}
            for metric in ['utility','net_utility']:
                matrix = np.empty((reps,len(seeds)))
                for rep in range(reps):
                    for j,seed in enumerate(seeds):
                        cells = [r[metric] for r in rows if r['seed']==seed and r['replicate'] in (None,rep)]
                        if len(cells)!=1:
                            raise ValueError('Missing or duplicate paired cell')
                        matrix[rep,j] = cells[0]
                matrices[policy][metric] = matrix
            summary = {k:float(np.mean([r[k] for r in rows])) for k in
                       ['utility','net_utility','kills','decision_compute_seconds','steps','firing_windows','success_windows','game_seconds']}
            summary.update(episodes=len(rows),truncated_episodes=sum(r['truncated'] for r in rows),
                           compute_ms_per_decision=1000*sum(r['decision_compute_seconds'] for r in rows)/sum(r['steps'] for r in rows),
                           max_episode_p95_decision_ms=max(r['p95_decision_ms'] for r in rows))
            out['summary'][scenario][policy] = summary
        out['primary'][scenario] = {}
        rule = protocol['continuation']
        for control in ('current_map','expanded_map'):
            diff = matrices['history_map']['net_utility']-matrices[control]['net_utility']
            draws = diff[ri[:,:,None],si[:,None,:]].mean(axis=(1,2))
            interval = np.quantile(draws,[.00625,.99375]).tolist()
            mean = float(diff.mean())
            passed = mean >= rule['minimum_mean_net_utility_gain'] and interval[0] > rule['minimum_simultaneous_interval_lower']
            out['primary'][scenario]['history_map_minus_'+control] = {
                'mean_net_utility_gain':mean,'bonferroni_98_75_interval':interval,'passes':passed,
                'raw_utility_gain':float((matrices['history_map']['utility']-matrices[control]['utility']).mean())}
            decisions.append(passed)
        h,c = out['summary'][scenario]['history_map'],out['summary'][scenario]['current_map']
        ratio = h['compute_ms_per_decision']/c['compute_ms_per_decision']
        latency_pass = h['max_episode_p95_decision_ms'] < rule['maximum_history_p95_decision_ms'] and ratio <= rule['maximum_aggregate_history_to_current_compute_per_decision_ratio']
        out['primary'][scenario]['compute_gate'] = {'ratio':ratio,'history_max_episode_p95_ms':h['max_episode_p95_decision_ms'],'passes':latency_pass}
        decisions.append(latency_pass)
        out['secondary_bayes_minus_map'][scenario] = {}
        for kind in protocol['features']:
            diff = matrices[kind+'_bayes']['net_utility']-matrices[kind+'_map']['net_utility']
            draws = diff[ri[:,:,None],si[:,None,:]].mean(axis=(1,2))
            out['secondary_bayes_minus_map'][scenario][kind] = {'mean_net_utility_gain':float(diff.mean()),
                'exploratory_95_interval':np.quantile(draws,[.025,.975]).tolist()}
    passed = all(decisions)
    out['continuation'] = {'passed':passed,'action':protocol['continuation']['pass_action' if passed else 'failure_action'],
                           'basis':'All four primary utility contrasts and both compute gates; no secondary-result override.'}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'primary':result['primary'],'continuation':result['continuation']},indent=2))
