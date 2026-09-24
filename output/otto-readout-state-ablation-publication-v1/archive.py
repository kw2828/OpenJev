"""Opaque evidence archive; no numerical decoding, inference or experiment work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
STUDY = ROOT / 'output/otto-readout-state-ablation-v1'
REPORT = ROOT / 'research/otto-readout-state-ablation-results'


def main():
    spec = importlib.util.spec_from_file_location('_readout_archive_runner', ROOT / 'scripts/run_otto_readout_ablation.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    closure = run.read(STUDY / 'closure-01.json')
    plan, _, _ = run.process_closure(closure['plan']['path'], closure['producer_receipt']['path'], closure['producer_terminal']['path'])
    run.process_closure(closure['plan']['path'], closure['audit_receipt']['path'], closure['audit_terminal']['path'])
    records = {}

    def add(path, expected=None):
        record = run.descriptor(path)
        relative = str(Path(record['path']).relative_to(ROOT))
        run.require(Path(relative).name != 'test.npz' and not Path(relative).name.startswith('test-prediction-'), 'no old TEST payload')
        if expected is not None:
            run.require(all(record[k] == v for k, v in expected.items()), 'archive input pin')
        run.require(relative not in records or records[relative] == record, 'consistent archive member')
        records[relative] = record

    for name, pin in plan['sources'].items():
        add(ROOT / name, {'sha256': pin})
    for record in plan['lineage']['evidence'].values():
        add(record['path'], record)
    for record in plan['bridge_inputs'].values():
        add(record['path'], record)
    for stage in ('train', 'dev'):
        add(plan[stage]['descriptor']['path'], plan[stage]['descriptor'])
    dev_receipt = run.read(plan['bridge_inputs']['collection_receipt']['path'])
    dev_directory = Path(plan['bridge_inputs']['collection_receipt']['path']).parent
    for name, pin in dev_receipt['files'].items():
        add(dev_directory / name, pin)
    for directory in (STUDY, STUDY / 'engineering-01', STUDY / 'training-01', STUDY / 'audit-01', REPORT):
        for path in directory.iterdir():
            if path.is_file():
                add(path)
    for name in ('scripts/close_otto_readout_ablation.py', 'scripts/report_otto_readout_ablation.py',
                 'research/otto-readout-state-ablation-results.md', 'research/otto-readout-learning-directions.md'):
        add(ROOT / name)
    add(Path(__file__))
    manifest = {'version': 'otto-readout-ablation-archive-v1', 'scope': 'complete current study plus authenticated prior training lineage, DEV collection and bound sources',
                'status': closure['status'], 'closure': run.descriptor(STUDY / 'closure-01.json'),
                'members': records, 'member_count': len(records), 'uncompressed_bytes': sum(r['bytes'] for r in records.values()),
                'numerical_array_decodes': 0, 'model_calls': 0, 'teacher_calls': 0, 'test_payloads': 0,
                'external_dependencies': ['Installed pinned Python environments', 'Original native teacher model/environment artifacts',
                                          'Earlier collection qualification and capacity dependencies not separately listed as members'],
                'prior_releases': ['https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-dev-v1',
                                   'https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-collection-v1',
                                   'https://github.com/kw2828/OpenJev/releases/tag/otto-residual-reanalysis-v1']}
    run.write(BASE / 'archive-manifest.json', manifest)
    archive = BASE / 'otto-readout-state-ablation-v1.tar.gz'
    with tarfile.open(archive, 'x:gz') as stream:
        for relative, record in sorted(records.items()):
            run.verify(record)
            stream.add(record['path'], arcname=relative, recursive=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        run.require(len(members) == len(records) and {m.name for m in members} == set(records), 'exact member roster')
        for member in members:
            run.require(member.isfile() and member.size == records[member.name]['bytes'], 'regular correctly sized member')
            raw = stream.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda raw=raw: raw.read(1024**2), b''):
                digest.update(block)
            run.require(digest.hexdigest() == records[member.name]['sha256'], 'opaque archive byte roundtrip')
    for record in records.values():
        run.verify(record)
    verification = {'status': 'passed', 'archive': run.descriptor(archive), 'manifest': run.descriptor(BASE / 'archive-manifest.json'),
                    'members_verified': len(records), 'source_bytes_unchanged': True, 'numerical_array_decodes': 0}
    run.write(BASE / 'archive-verification.json', verification)
    files = (archive, BASE / 'archive-manifest.json', BASE / 'archive-verification.json')
    with (BASE / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(run.descriptor(p)['sha256'] + '  ' + p.name + '\n' for p in files))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
