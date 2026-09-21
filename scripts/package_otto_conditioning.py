"""Package the completed conditioning evidence without model execution."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/otto-conditioning-v1'
OUT = BASE / 'archive-01'


def descriptor(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    assert not OUT.exists(), 'exclusive archive'
    pins = {
        'run-01/receipt.json': '0c879a9c8f0f418080c01f1a98c7ef79255909a92a48ec7e6dd61289de7a9d90',
        'audit-01/receipt.json': 'bcee27c5083acfbe0d7611b9fce62328a1282cb3cd2ab9569f30dd917e752daf',
        'report-01/receipt.json': 'e9f0e404fb090856fcc57f7251fbdaf9d56e6db6e778e3b7ee16973113ad0bc8',
    }
    for name, pin in pins.items():
        path = BASE / name
        assert descriptor(path)['sha256'] == pin
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'completed'
        for child, expected in receipt['files'].items():
            assert descriptor(path.parent / child) == expected
    assert (BASE / 'report-review-01.json').is_file(), 'independent report review required'
    files = [p for p in BASE.rglob('*') if p.is_file()]
    plan = json.loads((BASE / 'plan-01.json').read_text())
    source_names = [
        'scripts/study_otto_conditioning.py', 'scripts/freeze_otto_conditioning.py',
        'scripts/audit_otto_conditioning.py', 'src/openjev/research/otto_conditioned_value.py',
        'tests/test_otto_conditioned_value.py', 'tests/test_otto_conditioning.py',
        'tests/test_audit_otto_conditioning.py', 'research/otto-conditioning-protocol.md',
    ]
    for name in source_names:
        assert descriptor(ROOT / name)['sha256'] == plan['sources'][name]
    source_names += ['scripts/report_otto_conditioning.py', 'scripts/package_otto_conditioning.py',
                     'research/otto-conditioning-results.md', 'research/experiment-index.md',
                     'README.md', 'docs/assets/otto-conditioning.png', 'docs/assets/otto-conditioning.svg']
    files += [ROOT / name for name in source_names]
    assert all(not p.is_symlink() and p.resolve().is_relative_to(ROOT) for p in files)
    manifest = {str(p.relative_to(ROOT)): descriptor(p) for p in sorted(files)}
    assert len(manifest) == len(files)
    OUT.mkdir()
    write(OUT / 'manifest.json', {'version': 'otto-conditioning-archive-v1', 'files': manifest,
          'dependencies': 'Historical caches and source closures remain in the original scalar and capacity releases.'})
    archive = OUT / 'openjev-otto-conditioning-v1.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for path in sorted(files) + [OUT / 'manifest.json']:
            target.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    expected = {**manifest, str((OUT / 'manifest.json').relative_to(ROOT)): descriptor(OUT / 'manifest.json')}
    with tarfile.open(archive, 'r:gz') as saved:
        assert len(saved.getmembers()) == len(expected)
        assert {member.name for member in saved.getmembers()} == expected.keys()
        for member in saved.getmembers():
            assert member.isfile()
            data = saved.extractfile(member).read()
            assert {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == expected[member.name]
    result = {'status': 'verified', 'archive': descriptor(archive), 'original_files': len(files),
              'verified_members': len(expected), 'model_calls': 0, 'native_steps': 0}
    write(OUT / 'receipt.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
