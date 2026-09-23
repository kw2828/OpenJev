"""Losslessly package a closed diagnostic; no arrays, models or new metrics."""
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'output/otto-ranking-diagnosis-v1'
OUTPUT = Path(__file__).resolve().parent / 'package-01'


def pin(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': size}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


OUTPUT.mkdir(exist_ok=False)
receipt = {'status': 'started', 'scope': 'Opaque-byte publication only.',
           'model_calls': 0, 'native_calls': 0, 'optimizer_calls': 0}
try:
    plan = read(STUDY / 'plan-01.json')
    assert pin(STUDY / 'plan-01.json')['sha256'] == '20a5c91430043af5e15ed013a98c9203e7e27d1e613e1a09088a09a9c49fc3ce'
    closure = read(STUDY / 'closure-01.json')
    assert closure['status'] == 'completed'
    for name, expected in plan['sources'].items():
        assert pin(ROOT / name) == expected
    for phase, closed in closure['phases'].items():
        for kind in ('worker', 'terminal'):
            item = closed[kind]
            assert pin(Path(item['path'])) == {k: item[k] for k in ('sha256', 'bytes')}
        worker = read(Path(closed['worker']['path']))
        terminal = read(Path(closed['terminal']['path']))
        assert worker['status'] == terminal['status'] == 'completed'
        assert terminal['returncode'] == 0 and terminal['group_absent'] and not terminal['timed_out']
        for name, expected in worker['files'].items():
            assert pin(STUDY / f'{phase}-01' / name) == expected
    assert read(STUDY / 'audit-01/audit.json')['agreement'] is True
    raw = STUDY / 'diagnosis-01/diagnosis.json'
    expected = pin(raw)
    packed = OUTPUT / 'openjev-otto-ranking-diagnosis-v1.json.gz'
    with raw.open('rb') as source, packed.open('xb') as sink:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=sink, mtime=0) as target:
            shutil.copyfileobj(source, target, length=1024**2)
    digest, size = hashlib.sha256(), 0
    with gzip.open(packed, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
            size += len(chunk)
    assert {'sha256': digest.hexdigest(), 'bytes': size} == expected == pin(raw)
    metadata = ['plan-01.json', 'closure-01.json',
                'diagnosis-01/receipt.json', 'diagnosis-01/started.json',
                'diagnosis-supervision-01.launch.json', 'diagnosis-supervision-01.terminal.json',
                'audit-01/receipt.json', 'audit-01/audit.json', 'audit-01/started.json',
                'audit-supervision-01.launch.json', 'audit-supervision-01.terminal.json']
    manifest = {'version': 'otto-ranking-diagnosis-release-v1',
                'scope': 'Complete retrospective ranking tables; original predictions are in the upstream release.',
                'raw_json': {'path': str(raw.relative_to(ROOT)), **expected},
                'gzip_asset': {'name': packed.name, **pin(packed)},
                'metadata': {str((STUDY / name).relative_to(ROOT)): pin(STUDY / name) for name in metadata},
                'source_pins': plan['sources'],
                'original_evidence': 'https://github.com/kw2828/OpenJev/releases/tag/otto-separate-prior-v1',
                'round_trip_verified': True, 'new_scientific_gate': False}
    save(OUTPUT / 'manifest.json', manifest)
    note = '''# OpenJev ranking diagnostic

Decompress `openjev-otto-ranking-diagnosis-v1.json.gz` to recover every original diagnostic record. The manifest records both compressed and uncompressed SHA-256 hashes and byte sizes; the local package was verified by a complete streaming decompression.

The release tag points to the repository containing the protocol, code, source pins, qualification, original process receipts, independent audit, CSV exports, figures and interpretation. Absolute paths in historical receipts identify the original run machine, not a portable runtime.

Original teacher trajectories, predictions and all 24 checkpoints remain in the `otto-separate-prior-v1` release. No new weights or model results were created in this diagnostic. All three original scientific gates remain failed. This release is retrospective descriptive evidence, not an autonomous or architecture result.
'''
    with (OUTPUT / 'RESTORE.md').open('x') as stream:
        stream.write(note)
    receipt.update(status='completed', uncompressed=expected, round_trip_verified=True,
                   files={p.name: pin(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
except BaseException as error:
    receipt.update(status='failed', error=repr(error))
finally:
    receipt['source'] = pin(Path(__file__))
    save(OUTPUT / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
if receipt['status'] != 'completed':
    raise SystemExit(1)
