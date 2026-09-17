"""Paired, crossed-seed summaries of the frozen RL firing experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
from rl_doom import digest, verify


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text())
    stage = manifest['stage']
    verify(root,f'completed_{stage}')
    protocol = manifest['protocol']
    seeds = protocol[f'{stage}_seeds']
    reps = protocol['replicates']
    rows = [json.loads(s) for s in (root/'episodes.jsonl').read_text().splitlines()]
    arms = [*manifest['arms'],'rules','rule_event']
    expected_count = len(seeds)*len(protocol['scenarios'])*(reps*len(manifest['arms'])+2)
    if len(rows)!=expected_count:
        raise ValueError('Incorrect evaluation count')
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    sd = rng.integers(len(seeds),size=(protocol['bootstrap_draws'],len(seeds)))
    rd = rng.integers(reps,size=(protocol['bootstrap_draws'],reps))
    result = {'status':f'completed_{stage}_analysis','manifest_sha256':digest(root/'manifest.json'),
              'selection':manifest['selection'],'summary':{},'contrasts':{}}
    metrics = ('kills','game_seconds','utility','net_utility','engine_reward','firing_windows')
    for scenario in protocol['scenarios']:
        matrix = {}
        result['summary'][scenario] = {}
        for arm in arms:
            count = reps if arm in manifest['arms'] else 1
            subset = [r for r in rows if r['arm']==arm and r['scenario']==scenario]
            if len(subset)!=count*len(seeds):
                raise ValueError('Incomplete arm')
            matrix[arm] = {}
            for metric in metrics:
                values = np.empty((count,len(seeds)))
                for rep in range(count):
                    for j,seed in enumerate(seeds):
                        cell = [r[metric] for r in subset if r['seed']==seed and
                                r['replicate']==(rep if count>1 else None)]
                        if len(cell)!=1:
                            raise ValueError('Missing or duplicate cell')
                        values[rep,j] = cell[0]
                matrix[arm][metric] = values
            result['summary'][scenario][arm] = {m:float(matrix[arm][m].mean()) for m in metrics}
            result['summary'][scenario][arm].update(episodes=len(subset),
                capped_episodes=sum(r['capped'] for r in subset),
                mean_decision_ms=1000*sum(r['decision_compute_seconds'] for r in subset)/sum(r['steps'] for r in subset),
                max_episode_p95_decision_ms=max(r['p95_decision_ms'] for r in subset),
                per_fit_kills=matrix[arm]['kills'].mean(axis=1).tolist())
        result['contrasts'][scenario] = {}
        for arm in manifest['arms']:
            result['contrasts'][scenario][arm] = {}
            controls = ['rule_event','rules'] + (['ppo_current'] if arm!='ppo_current' and 'ppo_current' in matrix else [])
            for control in controls:
                entry = {}
                for metric in metrics[:4]:
                    diff = matrix[arm][metric]-matrix[control][metric]
                    draws = diff[rd[:,:,None],sd[:,None,:]].mean(axis=(1,2))
                    entry[metric] = {'difference':float(diff.mean()),
                        'interval':np.quantile(draws,protocol['interval_quantiles']).tolist()}
                result['contrasts'][scenario][arm][control] = entry
    if stage=='confirmation':
        selected = manifest['selection']['selected']
        checks = {}
        for scenario in protocol['scenarios']:
            c = result['contrasts'][scenario][selected]['rule_event']
            is_center = scenario=='defend_the_center'
            checks[scenario] = {
                'kills':c['kills']['interval'][0]>(0 if is_center else -1.) and
                        (c['kills']['difference']>=1. if is_center else True),
                'survival':c['game_seconds']['interval'][0]>-1.,
                'latency':result['summary'][scenario][selected]['max_episode_p95_decision_ms']<1.}
        result['gate_checks'] = checks
        result['confirmation_passed'] = all(v for c in checks.values() for v in c.values())
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=analyze(args.run)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'selection':result['selection'], 'summary':result['summary'],
                      'confirmation_passed':result.get('confirmation_passed')},indent=2))
