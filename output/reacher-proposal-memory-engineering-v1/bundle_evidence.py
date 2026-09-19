"""Package nine-row evidence, original innovations and exact cold references."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
OLD=ROOT/'output/persistent-dynamics-qualification-v1/control-engineering-01'
OUT=BASE/'release-v1'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert json.loads((BASE/'attempt-01/completed.json').read_text())['status']=='completed'
    assert json.loads((BASE/'process-01/completed.json').read_text())['exit_code']==0
    assert json.loads((BASE/'review-01/receipt.json').read_text())['status']=='completed'
    paths=[]
    for folder in ('attempt-01','process-01','review-01'):
        paths += [p for p in (BASE/folder).rglob('*') if p.is_file()]
    for name in ('protocol.json','run_engineering.py','report_results.py','bundle_evidence.py','implementation-review.md','audit-review.md','results-review.md','results-review.json','followup-design.md','next-decision.md'):
        paths.append(BASE/name)
    paths.append(ROOT/'research/reacher-proposal-memory.md')
    paths += [ROOT/f'tests/test_{n}.py' for n in ('reacher_cem_proposal_memory','reacher_proposal_memory_rollout','reacher_proposal_memory_audit')]
    paths += [OLD/name for name in ('started.json','completed.json','execution-completed.json','case.npz')]
    paths += [p for p in (OLD/'inputs').rglob('*') if p.is_file()]
    paths += [p for p in (OLD/'source-snapshot').rglob('*') if p.is_file()]
    for arm in ('nominal','public_gain','true_state'):
        paths += [p for p in (OLD/'rows'/arm).rglob('*') if p.is_file()]
    for arm in ('adaptive','frozen','zero'):
        paths.append(OLD/'rows'/arm/'completed.json')
    paths += list(OLD.glob('audit-*.json'))
    assert len(paths)==len(set(paths)) and all(p.is_file() and not p.is_symlink() for p in paths)
    manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)}
    OUT.mkdir(exist_ok=False)
    with (OUT/'members.json').open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
    archive=OUT/'openjev-proposal-memory-full-evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for name in manifest:bundle.add(ROOT/name,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        assert len(members)==len(manifest) and {m.name for m in members}==set(manifest)
        for member in members:
            assert member.isfile() and member.size==manifest[member.name]['bytes']
            with bundle.extractfile(member) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest[member.name]['sha256']
    receipt={'status':'completed','archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,
             'members':len(manifest),'raw_bytes':sum(x['bytes'] for x in manifest.values()),
             'all_members_reopened_and_hashed':True,'all_nine_raw_rows':True,'shared_original_inputs':True,
             'three_complete_cold_reference_rows':True,'other_old_row_raw_traces_included':False,
             'other_old_run_evidence':'https://github.com/kw2828/OpenJev/releases/tag/research-reacher-tracking-engineering-v1',
             'manifest_sha256':sha(OUT/'members.json')}
    with (OUT/'bundle-receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    (OUT/'SHA256SUMS').write_text(receipt['sha256']+'  '+archive.name+'\n'+receipt['manifest_sha256']+'  members.json\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
