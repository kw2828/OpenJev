"""Analyze one selected candidate without changing the prospective criterion."""
import argparse
import json
from pathlib import Path

import numpy as np
from cadence_doom import digest, verify_run


def analyze(root):
    manifest = verify_run(root, 'completed_confirmation')
    protocol = manifest['protocol']
    rows = [json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    seeds = protocol['confirmation_seeds']
    reps = protocol['history_replicates']
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    seed_draws = rng.integers(0,len(seeds),(protocol['bootstrap_draws'],len(seeds)))
    rep_draws = rng.integers(0,reps,(protocol['bootstrap_draws'],reps))
    gate = protocol['confirmation_gate']
    out = {'status':'frozen_selected_policy_confirmation', 'manifest_sha256':digest(root/'manifest.json'),
           'selected':manifest['selection']['selected'], 'summary':{}, 'primary':{}, 'secondary':{}}
    decisions = []
    for scenario in protocol['confirmation_scenarios']:
        out['summary'][scenario] = {}
        matrix = {}
        for arm in [*manifest['arms'],'history_map']:
            subset = [r for r in rows if r['scenario']==scenario and r['arm']==arm]
            count = reps if arm=='history_map' else 1
            if len(subset) != count*len(seeds):
                raise ValueError('Incomplete arm')
            matrix[arm] = {}
            for metric in ('net_utility','utility','kills'):
                a = np.empty((count,len(seeds)))
                for rep in range(count):
                    for j,seed in enumerate(seeds):
                        cells = [r[metric] for r in subset if r['seed']==seed and r['replicate']==(rep if arm=='history_map' else None)]
                        if len(cells) != 1:
                            raise ValueError('Missing or duplicated paired cell')
                        a[rep,j] = cells[0]
                matrix[arm][metric] = a
            summary = {m:float(np.mean([r[m] for r in subset])) for m in
                       ('utility','net_utility','kills','engine_reward','ammo_used','firing_windows',
                        'success_windows','game_seconds','decision_compute_seconds','steps')}
            summary.update(episodes=len(subset),truncated_episodes=sum(r['truncated'] for r in subset),
                           max_episode_p95_decision_ms=max(r['p95_decision_ms'] for r in subset),
                           compute_ms_per_decision=1000*sum(r['decision_compute_seconds'] for r in subset)/sum(r['steps'] for r in subset))
            out['summary'][scenario][arm] = summary
        out['primary'][scenario] = {}
        out['secondary'][scenario] = {}
        for control in ('rules','history_map','matched_no_rest','command_rest1'):
            item = {}
            for metric in ('net_utility','kills'):
                diff = matrix['selected'][metric]-matrix[control][metric]
                if control == 'history_map':
                    draws = diff[rep_draws[:,:,None],seed_draws[:,None,:]].mean(axis=(1,2))
                else:
                    draws = diff[0,seed_draws].mean(axis=1)
                item[metric] = {'mean_difference':float(diff.mean()),
                                'interval':np.quantile(draws,gate['interval_quantiles']).tolist()}
            if control in ('rules','history_map'):
                passed = (item['net_utility']['mean_difference'] >= gate['minimum_mean_net_gain']
                          and item['net_utility']['interval'][0] > gate['minimum_net_interval_lower']
                          and item['kills']['interval'][0] > gate['minimum_kill_interval_lower'])
                item['passed'] = passed
                decisions.append(passed)
                out['primary'][scenario][control] = item
            else:
                out['secondary'][scenario][control] = item
        time_pass = out['summary'][scenario]['selected']['max_episode_p95_decision_ms'] < gate['maximum_candidate_p95_decision_ms']
        decisions.append(time_pass)
        out['primary'][scenario]['latency_passed'] = time_pass
    out['confirmation_passed'] = all(decisions)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = analyze(args.run)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'selected':result['selected'],'primary':result['primary'],
                      'confirmation_passed':result['confirmation_passed']},indent=2))
