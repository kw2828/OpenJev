"""Prospective diagnostic controls; cannot change the frozen RL selection/gate."""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
from rl_doom import ROOT, digest, episode, verify

ADDENDUM=ROOT/'research/protocols/rl-doom-v1-simple-controls.json'


class SimplePolicy:
    def __init__(self,arm):
        self.arm=arm
        self.steps=0

    def predict(self,state,deterministic=True):
        action = (int(state[1] > .5) if self.arm=='visible_fire'
                  else 1 if self.arm=='always_fire' else int(self.steps%2==0))
        self.steps+=1
        return np.array(action),None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    parent=json.loads((args.run/'manifest.json').read_text())
    stage=parent['stage']
    verify(args.run,f'completed_{stage}')
    for file,expected in parent['source_sha256'].items():
        if digest(ROOT/file)!=expected:
            raise ValueError('Parent execution source changed')
    extra=json.loads(ADDENDUM.read_text())
    p=parent['protocol']
    jobs=[(sc,seed,arm) for sc in p['scenarios'] for seed in p[f'{stage}_seeds'] for arm in extra['arms']]
    np.random.default_rng(extra['schedule_seed']).shuffle(jobs)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','parent_manifest_sha256':digest(args.run/'manifest.json'),
             'source_sha256':digest(Path(__file__)),'addendum_sha256':digest(ADDENDUM),'stage':stage,
             'episodes':0,'claim_boundary':extra['claim_boundary']}
    started=time.monotonic()
    try:
        with gzip.open(args.output/'trace.jsonl.gz','wt') as trace, (args.output/'episodes.jsonl').open('x') as stream:
            for sc,seed,arm in jobs:
                if time.monotonic()-started>extra['max_wall_seconds']:
                    raise RuntimeError('Diagnostic wall budget exhausted')
                row=episode(sc,seed,arm,p,SimplePolicy(arm),'current',None,trace)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                receipt['episodes']+=1
        receipt['status']='completed_diagnostic_controls'
    except BaseException as exc:
        receipt.update(status='failed_no_efficacy_claim',error=str(exc))
        raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started
        receipt['artifact_sha256']={f.name:digest(f) for f in args.output.iterdir() if f.name!='manifest.json'}
        (args.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
