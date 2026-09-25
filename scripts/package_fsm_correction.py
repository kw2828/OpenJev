"""Package original study and engineering evidence, excluding the source archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--plots',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    closure=json.loads((args.study/'closure.json').read_text())
    if closure['status']!='completed':
        raise ValueError('original study must be terminal before packaging')
    if not args.audit.is_file():
        raise ValueError('independent audit must be retained')
    args.output.mkdir(parents=True,exist_ok=False)
    roots=[args.study,Path('output/fsm-correction-engineering-v1'),
           Path('output/predictive-state-core-engineering-v2'),Path('output/fsm-data-engineering-v1'),
           Path('output/fsm-gru-engineering-v1'),Path('output/fsm-audit-engineering-v1'),args.plots]
    files={p for root in roots for p in root.rglob('*') if p.is_file()}
    files.discard(Path('output/fsm-correction-engineering-v1/admission-01/combined_data.npz'))
    files|={args.audit,Path('research/fsm-correction-protocol.md'),Path('research/fsm-correction-registration.json'),
            Path('scripts/audit_fsm_correction.py'),Path('scripts/plot_fsm_correction.py'),Path(__file__).resolve().relative_to(Path.cwd())}
    entries=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    manifest={'scope':'All declared original model fits, predictions, checkpoints, failures, independent audit and qualification evidence. Source measurement archive excluded.',
              'source_data_license':'CC-BY-4.0; preserved normalized targets derive from CubeSpec FSM measurements',
              'source_data_author':'Merijn Floren, KU Leuven; Floren et al. ISMA-USD2024',
              'source_data_url':'https://github.com/merijnfloren/fsm-benchmark-data/tree/539a12fef384b086a8562b500498b2fa3899ef70',
              'source_data_changes':'100/200mV estimation arrays only; split by whole realization triplets and periods; FIT normalization; C100/H128 windows. No300mV or official-test arrays decoded.',
              'file_count':len(entries),'payload_bytes':sum(e['bytes'] for e in entries),'files':entries}
    mpath=args.output/'manifest.json';mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=args.output/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for item in entries:
            path=Path(item['path'])
            if sha(path)!=item['sha256']:
                raise ValueError('source changed during packaging: '+str(path))
            tar.add(path,arcname='evidence/'+str(path),recursive=False)
        tar.add(mpath,arcname='evidence/manifest.json',recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        for item in entries:
            member=tar.extractfile('evidence/'+item['path'])
            if member is None or hashlib.sha256(member.read()).hexdigest()!=item['sha256']:
                raise ValueError('archive verification failed')
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
             'manifest_sha256':sha(mpath),'file_count':len(entries),'payload_bytes':manifest['payload_bytes'],
             'all_archived_payloads_verified':True}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
