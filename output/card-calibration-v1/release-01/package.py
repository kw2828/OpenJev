"""Package the completed study without inference or native evaluation."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile
import time

root=Path(__file__).resolve().parents[3]
out=root/'output/card-calibration-v1/release-01'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def write(path,data):
    with path.open('x') as f: json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
start=time.monotonic()
paths=set()
for directory in ('evidence/card-calibration-v1','output/card-calibration-v1/calibration-01',
 'output/card-calibration-v1/evaluation-01','output/card-calibration-v1/report-01',
 'output/card-calibration-v1/visualization-01','output/card-calibration-v1/independent-review-01',
 'output/card-confidence-diagnostic-v1','output/card-confidence-direction-v1'):
    paths.update(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
bindings=json.loads((root/'evidence/card-calibration-v1/source-bindings.json').read_text())
paths.update(root/p for p in bindings['files'])
paths.update(root/p for p in ('README.md','research/card-calibration-results.md',
 'output/card-calibration-v1/independent-audit.py','output/card-calibration-v1/independent-review.md',
 'output/card-calibration-v1/next-direction.md',
 'output/card-calibration-v1/release-01/package.py'))
for relative in ('calibration-01/completed.json','evaluation-01/completed.json','report-01/receipt.json',
 'visualization-01/receipt.json','independent-review-01/completed.json'):
    doc=json.loads((root/'output/card-calibration-v1'/relative).read_text())
    assert doc['status']==('passed' if relative=='independent-review-01/completed.json' else 'complete'),relative
for p in paths: assert p.is_file() and not p.is_symlink(),str(p)
files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)}
manifest={'version':'card-calibration-evidence-v1','status':'complete','files':files,
 'payload_members':len(files),'payload_bytes':sum(x['bytes'] for x in files.values()),
 'inherited_releases':[
 {'url':'https://github.com/kw2828/OpenJev/releases/tag/research-card-memory-pilot-v1','archive_sha256':'dbdc50851322e2dc02b744e43eac83b315f83bcfa607b9931ef932e2f69b980f','contents':'Original training data, checkpoints and 78 bound source files'},
 {'url':'https://github.com/kw2828/OpenJev/releases/tag/research-card-controllers-v1','archive_sha256':'a1081454314e0dc7c5a694be95c1c65885714d9642e1140944482bba7448076c','contents':'Prior C calibration-training trajectories and controller source/evidence'}],
 'scope':'All new scalar fits, fresh trajectories, complete reports, figures, independent audit and prospective source snapshots. Prior weights and calibration-training trajectories remain in the linked immutable releases.'}
write(out/'manifest.json',manifest)
archive=out/'openjev-card-calibration-full-evidence.tar.gz'
with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
    for p in sorted(paths): tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    tar.add(out/'manifest.json',arcname='output/card-calibration-v1/release-01/manifest.json',recursive=False)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    assert {m.name for m in members}==set(files)|{'output/card-calibration-v1/release-01/manifest.json'}
    for m in members:
        if m.name in files:
            f=tar.extractfile(m);h=hashlib.sha256()
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
            assert h.hexdigest()==files[m.name]['sha256'],m.name
write(out/'receipt.json',{'status':'complete','version':'card-calibration-evidence-v1',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'archive':archive.name,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
 'archive_members':len(members),'manifest_sha256':sha(out/'manifest.json'),
 'wall_seconds':time.monotonic()-start,'all_archive_members_stream_verified':True,
 'new_model_calls':0,'new_native_calls':0})
print(json.dumps(json.loads((out/'receipt.json').read_text())))
