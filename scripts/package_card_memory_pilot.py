"""Package closed card-pilot evidence; never executes environments or models."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def package(root,out):
    out.mkdir(parents=True,exist_ok=False)
    base=root/'output/card-memory-pilot-v1'
    for phase in ('preflight-01','data-01','training-01','evaluation-01'):
        path=base/phase
        if not (path/'completed.json').is_file() or (path/'failed.json').exists():
            raise ValueError('Incomplete phase: '+phase)
    for path in (base/'report-01/summary.json',base/'visualization-01/receipt.json',base/'data-audit-01.json',base/'independent-results-review.json'):
        if not path.is_file():raise ValueError('Missing artifact: '+str(path))
    sources=[root/'evidence/card-memory-pilot-v1',root/'output/decision-memory-direction-v1']
    sources += [base/name for name in ('preflight-01','data-01','training-01','evaluation-01','report-01','visualization-01')]
    files=[p for folder in sources for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [base/'data-audit-01.json',base/'independent-results-review.json',base/'picker-diagnostic-01.json',root/'research/card-memory-pilot.md']
    files += [root/'scripts'/name for name in ('report_card_memory_pilot.py','visualize_card_memory_pilot.py','audit_card_memory_data.py','package_card_memory_pilot.py')]
    files=sorted(set(files))
    manifest={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    archive=out/'openjev-card-memory-pilot-full-evidence.tar.gz'
    with tarfile.open(archive,'x:gz') as packed:
        for p in files:packed.add(p,arcname=str(p.relative_to(root)),recursive=False)
    with tarfile.open(archive,'r:gz') as packed:
        members=packed.getmembers()
        if {m.name for m in members}!=set(manifest):raise ValueError('Archive membership mismatch')
        for m in members:
            stream=packed.extractfile(m)
            if hashlib.sha256(stream.read()).hexdigest()!=manifest[m.name]['sha256']:
                raise ValueError('Archive hash mismatch')
    receipt={'status':'complete','members':len(files),'uncompressed_bytes':sum(v['bytes'] for v in manifest.values()),
             'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'manifest_sha256':sha(out/'manifest.json'),
             'native_calls':0,'model_calls':0}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    package(**vars(parser.parse_args()))
