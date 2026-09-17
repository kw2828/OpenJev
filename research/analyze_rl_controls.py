"""Exploratory simple-control comparisons and delayed-credit diagnostics."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rl_doom import digest, verify


def trace_rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:
            yield json.loads(line)


def analyze(run,controls):
    parent=json.loads((run/'manifest.json').read_text())
    verify(run,'completed_'+parent['stage'])
    receipt=verify(controls,'completed_diagnostic_controls')
    if receipt['parent_manifest_sha256']!=digest(run/'manifest.json'):
        raise ValueError('Controls have different parent provenance')
    protocol=parent['protocol']
    seeds=protocol[parent['stage']+'_seeds']
    selected=parent['selection']['selected']
    if selected is None:
        raise ValueError('Selected family required for this comparison')
    rows=[json.loads(s) for s in (run/'episodes.jsonl').read_text().splitlines()]
    cr=[json.loads(s) for s in (controls/'episodes.jsonl').read_text().splitlines()]
    rng=np.random.default_rng(protocol['bootstrap_seed'])
    sd=rng.integers(len(seeds),size=(protocol['bootstrap_draws'],len(seeds)))
    rd=rng.integers(protocol['replicates'],size=(protocol['bootstrap_draws'],protocol['replicates']))
    means,contrasts={},{}
    for scenario in protocol['scenarios']:
        means[scenario],contrasts[scenario]={},{}
        for arm in ['always_fire','alternating_fire','visible_fire']:
            c=[r for r in cr if r['scenario']==scenario and r['arm']==arm]
            if len(c)!=len(seeds):raise ValueError('Incomplete control')
            means[scenario][arm]={m:float(np.mean([r[m] for r in c])) for m in
                                  ['kills','game_seconds','utility','firing_windows']}
            contrasts[scenario][arm]={}
            for metric in ['kills','game_seconds','utility']:
                diff=np.empty((protocol['replicates'],len(seeds)))
                for rep in range(protocol['replicates']):
                    for j,seed in enumerate(seeds):
                        a=[r[metric] for r in rows if r['scenario']==scenario and r['arm']==selected
                           and r['replicate']==rep and r['seed']==seed]
                        b=[r[metric] for r in c if r['seed']==seed]
                        if len(a)!=1 or len(b)!=1:raise ValueError('Missing or duplicate cell')
                        diff[rep,j]=a[0]-b[0]
                draws=diff[rd[:,:,None],sd[:,None,:]].mean(axis=(1,2))
                contrasts[scenario][arm][metric]={'difference':float(diff.mean()),
                    'interval':np.quantile(draws,protocol['interval_quantiles']).tolist()}
    hits=defaultdict(lambda:{'hit_windows_with_fire':0,'hit_windows_without_fire':0})
    actions=defaultdict(list)
    for r in trace_rows(controls/'trace.jsonl.gz'):
        actions[r['scenario'],r['seed'],r['arm']].append(r['issued_fire'])
        key='hit_windows_with_fire' if r['issued_fire'] else 'hit_windows_without_fire'
        hits[r['scenario']+':'+r['arm']][key]+=int(r['hit_count_change']>0)
    learned=defaultdict(list)
    for r in trace_rows(run/'evaluation-trace.jsonl.gz'):
        if r['arm']==selected:
            learned[r['scenario'],r['seed'],r['replicate']].append(r['issued_fire'])
    equivalence={}
    for scenario in protocol['scenarios']:
        matches=sum(sequence==actions[sc,seed,'always_fire'] for (sc,seed,_),sequence in learned.items() if sc==scenario)
        equivalence[scenario]={'matched_episodes':matches,'compared_episodes':len(seeds)*protocol['replicates']}
    pairs={}
    for scenario in protocol['scenarios']:
        a={r['seed']:r for r in cr if r['arm']=='always_fire' and r['scenario']==scenario}
        b={r['seed']:r for r in cr if r['arm']=='alternating_fire' and r['scenario']==scenario}
        pairs[scenario]={f:sum(a[s][f]==b[s][f] for s in seeds)
                         for f in ['kills','game_seconds','engine_reward','ammo_used','steps','dead','capped']}
    return {'status':'completed_exploratory_controls_analysis','selected':selected,
            'means':means,'selected_minus_control':contrasts,'issued_fire_equivalence_to_always_fire':equivalence,
            'always_alternating_matched_fields':pairs,'control_hit_timing':dict(hits),
            'source_sha256':digest(Path(__file__)),'run_manifest_sha256':digest(run/'manifest.json'),
            'control_manifest_sha256':digest(controls/'manifest.json'),
            'claim_boundary':'Secondary exploratory comparisons. Intervals use parent quantiles but are outside its four-primary-comparison adjustment family. No change to frozen selection or primary gate.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--controls',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    value=analyze(args.run,args.controls)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'means':value['means'],'equivalence':value['issued_fire_equivalence_to_always_fire']},indent=2))
