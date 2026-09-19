"""Package complete diagnostic evidence plus its source roots and original draws."""
import json
import tarfile
from pathlib import Path

from openjev.research.reacher_tracking_rollout import sha, write

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT/'output/reacher-proposal-memory-engineering-v1/attempt-01'
OUT = BASE/'release-v1'


def main():
    completed = json.loads((BASE/'attempt-01/completed.json').read_text())
    assert completed['status'] == 'completed'
    assert json.loads((BASE/'process-01/completed.json').read_text())['exit_code'] == 0
    assert json.loads((BASE/'review-01/receipt.json').read_text())['status'] == 'completed'
    paths = []
    for name in ('attempt-01', 'process-01', 'review-01', 'inputs-v1'):
        paths += [p for p in (BASE/name).rglob('*') if p.is_file()]
    for name in ('README.md', 'protocol.json', 'report-arithmetic.json', 'prepare_inputs.py', 'run_diagnostic.py',
                 'report_results.py', 'bundle_evidence.py', 'protocol-review.md', 'report-review.md',
                 'results-review.md', 'results-review.json', 'next-decision.md', 'next-direction.md', 'release-notes.md'):
        paths.append(BASE/name)
    paths += [ROOT/'research/reacher-common-root-diagnostic.md', PRIOR/'completed.json']
    for role in ('nominal', 'public_gain', 'true_state'):
        paths += [PRIOR/'rows'/f'{role}--cold'/name for name in ('completed.json', 'started.json', 'episode.json')]
    prepared = json.loads((BASE/'inputs-v1/completed.json').read_text())
    for binding in prepared['original_inputs'].values():
        for suffix in ('npz', 'json'):
            path = Path(binding['stem']).with_suffix('.'+suffix)
            assert sha(path) == binding[suffix+'_sha256']
            paths.append(path)
    for name, digest in completed['source_sha256'].items():
        assert sha(ROOT/name) == digest
        paths.append(ROOT/name)
    paths = sorted(set(paths))
    assert all(p.is_file() and not p.is_symlink() for p in paths)
    manifest = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}
    OUT.mkdir(exist_ok=False); write(OUT/'members.json', manifest)
    archive = OUT/'openjev-common-root-full-evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in manifest:
            bundle.add(ROOT/name, arcname=name, recursive=False)
    import hashlib
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        assert len(members) == len(manifest) and {m.name for m in members} == set(manifest)
        for member in members:
            assert member.isfile() and member.size == manifest[member.name]['bytes']
            with bundle.extractfile(member) as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == manifest[member.name]['sha256']
    receipt = {'status': 'completed', 'archive': archive.name, 'bytes': archive.stat().st_size,
        'sha256': sha(archive), 'members': len(manifest), 'raw_bytes': sum(v['bytes'] for v in manifest.values()),
        'all_members_reopened_and_hashed': True, 'all_twelve_raw_slots_and_inputs': True,
        'original_source_episodes_and_four_input_stems': True, 'all_prior_raw_decision_traces_included': False,
        'prior_complete_evidence': 'https://github.com/kw2828/OpenJev/releases/tag/research-reacher-proposal-memory-v1',
        'manifest_sha256': sha(OUT/'members.json')}
    write(OUT/'bundle-receipt.json', receipt)
    (OUT/'SHA256SUMS').write_text(receipt['sha256']+'  '+archive.name+'\n'+receipt['manifest_sha256']+'  members.json\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
