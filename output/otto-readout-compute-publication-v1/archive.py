"""Opaque evidence archive; no numerical decoding, inference or experiment work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
STUDY = ROOT / 'output/otto-readout-compute-v1'
REPORT = ROOT / 'research/otto-readout-compute-results'


def main():
    spec = importlib.util.spec_from_file_location('_readout_archive_runner', ROOT / 'scripts/run_otto_readout_compute.py')
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
    # The new data replace bridge_inputs in the plan, but authenticate_prior
    # still verifies this original native-to-numerical proof as opaque bytes.
    prior = run.authenticate_prior()
    add(ROOT / 'output/otto-residual-reanalysis-v1/dev-evaluation-plan-01.json')
    for record in prior['bridge_inputs'].values():
        add(record['path'], record)
    old_bridge = run.read(prior['bridge_inputs']['bridge_receipt']['path'])
    old_bridge_dir = Path(prior['bridge_inputs']['bridge_receipt']['path']).parent
    for name, pin in old_bridge['files'].items():
        add(old_bridge_dir / name, pin)
    add(old_bridge_dir.parent / 'process.launch.json')
    old_plan = run.read(prior['bridge_inputs']['collection_plan']['path'])
    old_collection = run.read(prior['bridge_inputs']['collection_receipt']['path'])
    old_collection_dir = Path(prior['bridge_inputs']['collection_receipt']['path']).parent
    for name, pin in old_collection['files'].items():
        add(old_collection_dir / name, pin)
    add(old_collection_dir.parent / 'dev-collection-native-01.launch.json')
    for record in old_plan['inputs'].values():
        add(record['path'], {k: record[k] for k in ('sha256', 'bytes')})
    old_qual_path = run.regular(old_plan['inputs']['engineering']['path'])
    for name, pin in run.read(old_qual_path)['files'].items():
        add(old_qual_path.parent / name, pin)
    external_files = {name: run.descriptor(record['path']) for name, record in old_plan['native_inputs'].items()}
    current_native = run.read(plan['bridge_inputs']['collection_plan']['path'])['native_inputs']
    run.require(set(current_native) == set(external_files)
                and all(run.descriptor(record['path']) == external_files[name]
                    for name, record in current_native.items()), 'same disclosed external native inputs')
    for directory in (STUDY, STUDY / 'engineering-01', STUDY / 'training-01', STUDY / 'audit-01', STUDY / 'native-bridge-01', REPORT):
        for path in directory.iterdir():
            if path.is_file():
                add(path)
    for name in ('scripts/close_otto_readout_compute.py', 'scripts/report_otto_readout_compute.py',
                 'scripts/report_otto_readout_ablation.py',
                 'research/otto-readout-compute-results.md', 'research/otto-readout-cache-proposal.md',
                 'research/otto-linear-probe-proposal.md'):
        add(ROOT / name)
    for path in (BASE / 'render-01').iterdir():
        add(path)
    add(BASE / 'renderer-01.py')
    add(BASE / 'visual-revision.json')
    add(Path(__file__))
    manifest = {'version': 'otto-readout-compute-archive-v1', 'scope': 'complete current study plus authenticated prior training lineage, DEV collection and bound sources',
                'status': closure['status'], 'closure': run.descriptor(STUDY / 'closure-01.json'),
                'members': records, 'member_count': len(records), 'uncompressed_bytes': sum(r['bytes'] for r in records.values()),
                'numerical_array_decodes': 0, 'model_calls': 0, 'teacher_calls': 0, 'test_payloads': 0,
                'external_file_inputs': external_files,
                'portability': 'Registered repository and interpreter paths are absolute; this is not a portable self-contained installation.',
                'external_dependencies': ['Installed pinned Python environments', 'Original native teacher model/environment artifacts',
                                          'Earlier collection qualification and capacity dependencies not separately listed as members'],
                'prior_releases': ['https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-dev-v1',
                                   'https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-collection-v1',
                                   'https://github.com/kw2828/OpenJev/releases/tag/otto-residual-reanalysis-v1']}
    run.write(BASE / 'archive-manifest.json', manifest)
    archive = BASE / 'otto-readout-compute-v1.tar.gz'
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
