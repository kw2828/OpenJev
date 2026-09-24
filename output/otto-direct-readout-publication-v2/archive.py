"""Package closed evidence as opaque bytes, with the prior archive's lineage."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
STUDY = ROOT / 'output/otto-direct-readout-v1'
REPORT = ROOT / 'research/otto-direct-readout-results'


def main():
    spec = importlib.util.spec_from_file_location('_direct_archive', ROOT / 'scripts/run_otto_direct_readout.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    closure = run.read(STUDY / 'closure-01.json')
    plan, _, _ = run.process_closure(closure['plan']['path'], closure['producer_receipt']['path'],
                                     closure['producer_terminal']['path'])
    run.process_closure(closure['plan']['path'], closure['audit_receipt']['path'], closure['audit_terminal']['path'])
    prior_base = ROOT / 'output/otto-readout-compute-publication-v1'
    verification = run.read(prior_base / 'archive-verification.json')
    run.verify(verification['manifest'])
    prior = run.read(verification['manifest']['path'])
    records = {}

    def add(value, expected=None):
        record = run.descriptor(value)
        name = str(Path(record['path']).relative_to(ROOT))
        run.require(Path(name).name != 'test.npz' and not Path(name).name.startswith('test-prediction-'),
                    'no original TEST payload')
        if expected is not None:
            run.require(all(record[k] == v for k, v in expected.items()), 'unchanged archived input')
        run.require(name not in records or records[name] == record, 'consistent archive member')
        records[name] = record

    for name, pin in prior['members'].items():
        add(ROOT / name, pin)
    add(prior_base / 'archive-manifest.json', verification['manifest'])
    add(prior_base / 'archive-verification.json')
    for name, pin in plan['sources'].items():
        add(ROOT / name, {'sha256': pin})
    for directory in (STUDY, STUDY/'engineering-01', STUDY/'engineering-02',
                      STUDY/'training-01', STUDY/'audit-01', REPORT):
        for path in directory.iterdir():
            if path.is_file():
                add(path)
    for name in ('scripts/close_otto_direct_readout.py', 'scripts/report_otto_direct_readout.py',
                 'research/otto-direct-readout-results.md', 'research/otto-action-latent-proposal.md'):
        add(ROOT / name)
    for path in (ROOT / 'output/otto-direct-readout-publication-v1/render-01').iterdir():
        add(path)
    add(ROOT / 'output/otto-direct-readout-publication-v1/renderer-01.py')
    add(ROOT / 'output/otto-direct-readout-publication-v1/visual-revision.json')
    add(Path(__file__))
    manifest = {'version': 'otto-direct-readout-archive-v1', 'status': closure['status'],
                'closure': run.descriptor(STUDY/'closure-01.json'), 'members': records,
                'member_count': len(records), 'uncompressed_bytes': sum(r['bytes'] for r in records.values()),
                'scope': 'Complete current study and every byte-verified member of the prior comparable-cost evidence archive.',
                'prior_manifest': verification['manifest'], 'external_file_inputs': prior['external_file_inputs'],
                'external_dependencies': prior['external_dependencies'], 'portability': prior['portability'],
                'dev_reused': True, 'fresh_dev': False, 'numerical_array_decodes': 0,
                'model_calls': 0, 'solver_calls': 0, 'test_payloads': 0}
    run.write(BASE/'archive-manifest.json', manifest)
    archive = BASE/'otto-direct-readout-v1.tar.gz'
    with tarfile.open(archive, 'x:gz') as stream:
        for name, record in sorted(records.items()):
            run.verify(record)
            stream.add(record['path'], arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        run.require(len(members) == len(records) and {m.name for m in members} == set(records), 'exact roster')
        for member in members:
            run.require(member.isfile() and member.size == records[member.name]['bytes'], 'regular expected member')
            raw, digest = stream.extractfile(member), hashlib.sha256()
            for block in iter(lambda raw=raw: raw.read(1024**2), b''):
                digest.update(block)
            run.require(digest.hexdigest() == records[member.name]['sha256'], 'archive byte roundtrip')
    for record in records.values():
        run.verify(record)
    result = {'status': 'passed', 'archive': run.descriptor(archive),
              'manifest': run.descriptor(BASE/'archive-manifest.json'), 'members_verified': len(records),
              'source_bytes_unchanged': True, 'numerical_array_decodes': 0}
    run.write(BASE/'archive-verification.json', result)
    with (BASE/'SHA256SUMS').open('x') as stream:
        for path in (archive, BASE/'archive-manifest.json', BASE/'archive-verification.json'):
            stream.write(run.descriptor(path)['sha256'] + '  ' + path.name + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
