"""Package the complete engineering run and verify every archived byte."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/persistent-dynamics-qualification-v1'
OUT = BASE / 'control-evidence-release-v1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert json.loads((BASE/'control-engineering-01/completed.json').read_text())['status'] == 'completed'
    assert json.loads((BASE/'control-process-01/completed.json').read_text())['exit_code'] == 0
    assert json.loads((BASE/'control-results-review-01/receipt.json').read_text())['status'] == 'completed'
    paths=[]
    for folder in ('control-engineering-01','control-process-01','control-results-review-01'):
        paths.extend(p for p in (BASE/folder).rglob('*') if p.is_file())
    for name in ('run_control_engineering.py','review_control_engineering.py','plot_control_engineering.py',
                 'bundle_control_engineering.py','control-engineering-results.png','control-audit-design.md',
                 'control-audit-review.md','rollout-review.md','policy-review.md','qualification-protocol-review.md',
                 'next-decision.md'):
        paths.append(BASE/name)
    paths += list((ROOT/'tests').glob('test_reacher_tracking*.py'))
    paths.append(ROOT/'research/reacher-tracking-engineering.md')
    assert len(paths) == len(set(paths)) and all(p.is_file() and not p.is_symlink() for p in paths)
    manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(paths)}
    OUT.mkdir(exist_ok=False)
    (OUT/'member-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    archive=OUT/'openjev-tracking-engineering-full-evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for name in manifest:
            bundle.add(ROOT/name,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        assert len(members)==len(manifest) and {p.name for p in members}==set(manifest)
        for member in members:
            assert member.isfile() and member.size==manifest[member.name]['bytes']
            with bundle.extractfile(member) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest[member.name]['sha256']
    result={'status':'completed','engineering':True,'archive':archive.name,'archive_sha256':digest(archive),
            'archive_bytes':archive.stat().st_size,'member_count':len(manifest),
            'uncompressed_bytes':sum(v['bytes'] for v in manifest.values()),'every_member_reopened_and_hashed':True,
            'raw_execution_traces_included':True,'manifest_sha256':digest(OUT/'member-manifest.json')}
    (OUT/'bundle-receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (OUT/'SHA256SUMS').write_text(result['archive_sha256']+'  '+archive.name+'\n'+result['manifest_sha256']+'  member-manifest.json\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
