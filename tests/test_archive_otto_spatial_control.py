"""Tiny fabricated byte/receipt checks only; no real archives or scientific inputs."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_archive_spatial_control_test', ROOT/'scripts/archive_otto_spatial_control.py')
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'ROOT', tmp_path)
    monkeypatch.setattr(M.B, 'ROOT', tmp_path)
    study = tmp_path/'output/otto-spatial-control-v1'
    study.mkdir(parents=True)
    monkeypatch.setattr(M, 'STUDY', study)
    return tmp_path, study


def member_fixture(root):
    originals = {'saved/a.txt': b'Every fixed case stays in the archive.\n',
                 'saved/b.bin': b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(32))}
    identities = {}
    for name, data in originals.items():
        path = root/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        identities[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return originals, identities


@pytest.mark.parametrize(('lengths', 'expected'), [([8], [8]), ([3, 13], [8, 8]), ([17, 3], [8, 8, 4])])
def test_exact_part_boundaries_have_no_empty_trailing_part(tmp_path, lengths, expected):
    writer = M.PartWriter(tmp_path, lambda: None, part_bytes=8)
    data = []
    for i, count in enumerate(lengths):
        piece = bytes([65+i])*count
        assert writer.write(piece) == count
        data.append(piece)
    writer.finish()
    paths = [tmp_path/name for name in writer.parts]
    assert [p.stat().st_size for p in paths] == expected
    assert [p.name for p in paths] == [f'{M.PREFIX}{i+1:04d}' for i in range(len(expected))]
    assert b''.join(p.read_bytes() for p in paths) == b''.join(data)
    assert writer.hasher.hexdigest() == hashlib.sha256(b''.join(data)).hexdigest()


def test_multipart_roundtrip_is_lossless_and_deterministic(workspace):
    root, study = workspace
    originals, members = member_fixture(root)
    first, second = study/'first', study/'second'
    first.mkdir()
    second.mkdir()
    result = M.pack(first, members, lambda: None, part_bytes=97)
    assert len(result['parts']) > 1
    joined = b''.join((first/p['name']).read_bytes() for p in result['parts'])
    assert result['sha256'] == hashlib.sha256(joined).hexdigest()
    assert result['bytes'] == len(joined)
    with tarfile.open(fileobj=io.BytesIO(joined), mode='r:gz') as archive:
        entries = archive.getmembers()
        assert [p.name for p in entries] == list(originals)
        for entry in entries:
            assert entry.isfile() and entry.mtime == 0 and entry.uid == entry.gid == 0
            assert archive.extractfile(entry).read() == originals[entry.name]
    # Original mtimes and chmod are intentionally absent from archive metadata.
    (root/'saved/a.txt').chmod(0o600)
    repeated = M.pack(second, members, lambda: None, part_bytes=97)
    assert repeated == result
    assert all((root/name).read_bytes() == data for name, data in originals.items())


@pytest.mark.parametrize('damage', ['corrupt_trailer', 'truncate', 'missing_part'])
def test_stream_corruption_and_truncation_are_refused(workspace, damage):
    root, study = workspace
    _, members = member_fixture(root)
    output = study/'parts'
    output.mkdir()
    stream = M.pack(output, members, lambda: None, part_bytes=97)
    paths = [output/p['name'] for p in stream['parts']]
    last = paths[-1].read_bytes()
    if damage == 'corrupt_trailer':
        paths[-1].write_bytes(last[:-1]+bytes([last[-1] ^ 1]))
    elif damage == 'truncate':
        paths[-1].write_bytes(last[:-1])
    else:
        paths = paths[:1]+paths[2:]
    with pytest.raises((gzip.BadGzipFile, EOFError, tarfile.ReadError, ValueError, zlib.error)):
        M.verify_members(paths, members, lambda: None)


def test_member_descriptor_mismatch_is_refused(workspace):
    root, study = workspace
    _, members = member_fixture(root)
    output = study/'parts'
    output.mkdir()
    stream = M.pack(output, members, lambda: None, part_bytes=97)
    members['saved/a.txt']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='lossless member'):
        M.verify_members([output/p['name'] for p in stream['parts']], members, lambda: None)


def archive_fixture(workspace):
    root, study = workspace
    for name in M.PUBLICATION_SOURCES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fabricated publication source '+name+'\n')
    args = SimpleNamespace(output=study/'archive-01')
    archive = M.Archive(args)
    archive.check = lambda: None
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in M.PUBLICATION_SOURCES}
    return archive, sources


def engineering_attempt(study, number, sources, *, passed):
    directory = study/f'publication-engineering-{number:02d}'
    directory.mkdir()
    log = directory/'pytest.log'
    log.write_text('fabricated test log\n')
    receipt = {'version': M.ENGINEERING_VERSION, 'status': 'passed' if passed else 'failed',
               'scope': 'Fabricated byte-only checks', 'sources': sources.copy(),
               'sources_before': sources.copy(), 'sources_after': sources.copy(),
               'results': [{'name': 'pytest', 'exit_code': 0 if passed else 1}],
               'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0,
               'files': {'pytest.log': {'bytes': log.stat().st_size,
                                        'sha256': hashlib.sha256(log.read_bytes()).hexdigest()}}}
    (directory/'receipt.json').write_text(json.dumps(receipt))
    return directory, receipt


def test_all_failed_attempts_are_preserved_and_current_pass_is_required(workspace):
    _, study = workspace
    archive, sources = archive_fixture(workspace)
    engineering_attempt(study, 1, sources, passed=False)
    with pytest.raises(ValueError, match='one passed publication attempt'):
        archive.publication_engineering()
    assert len(archive.categories['publication_engineering']) == 2
    engineering_attempt(study, 2, sources, passed=True)
    result = archive.publication_engineering()
    assert result['attempts'] == ['publication-engineering-01', 'publication-engineering-02']
    assert result['admitting_attempts'] == ['publication-engineering-02']
    assert len(archive.categories['publication_engineering']) == 4


def test_an_older_pass_is_preserved_but_cannot_admit_current_bytes(workspace):
    _, study = workspace
    archive, sources = archive_fixture(workspace)
    old = {**sources, M.PACKER: 'a'*64}
    engineering_attempt(study, 1, old, passed=True)
    with pytest.raises(ValueError, match='one passed publication attempt'):
        archive.publication_engineering()
    engineering_attempt(study, 2, sources, passed=True)
    result = archive.publication_engineering()
    assert result['admitting_attempts'] == ['publication-engineering-02']
    assert result['attempts'] == ['publication-engineering-01', 'publication-engineering-02']


@pytest.mark.parametrize('defect', ['unlisted', 'nested', 'wrong_hash', 'empty_results', 'boolean_exit',
                                  'nonzero_pass', 'missing_source', 'changed_source', 'bad_directory'])
def test_incomplete_or_misleading_engineering_is_refused(workspace, defect):
    _, study = workspace
    archive, sources = archive_fixture(workspace)
    directory, receipt = engineering_attempt(study, 1, sources, passed=True)
    if defect == 'unlisted':
        (directory/'extra.log').write_text('not declared')
    elif defect == 'nested':
        (directory/'nested').mkdir()
    elif defect == 'wrong_hash':
        receipt['files']['pytest.log']['sha256'] = '0'*64
    elif defect == 'empty_results':
        receipt['results'] = []
    elif defect == 'boolean_exit':
        receipt['results'][0]['exit_code'] = False
    elif defect == 'nonzero_pass':
        receipt['results'][0]['exit_code'] = 1
    elif defect == 'missing_source':
        del receipt['sources'][M.PACKER_TEST]
    elif defect == 'changed_source':
        receipt['sources_after'][M.PACKER] = '0'*64
    else:
        directory = directory.rename(study/'publication-engineering-extra')
    (directory/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        archive.publication_engineering()


def test_incomplete_worker_is_refused_before_any_archive_member(workspace):
    _, study = workspace
    run = study/'run-01'
    run.mkdir()
    receipt_path = run/'receipt.json'
    receipt_path.write_text(json.dumps({'status': 'running', 'files': {}}))
    pin = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    # Use the held report helper's real closure check; no producer/model imports.
    report_path = ROOT/M.REPORT
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == M.REPORT_PIN
    spec = importlib.util.spec_from_file_location('_archive_test_report', report_path)
    report = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = report
    spec.loader.exec_module(report)
    args = SimpleNamespace(plan=study/'plan-01.json', run=run, audit=study/'audit-01',
                           report=study/'report-01', terminal=study/'process-01.terminal.json', output=study/'archive-01')
    archive = M.Archive(args)
    archive.check = lambda: None

    def authenticate(_args, check):
        return report.closed(run, pin, report.worker_payloads(), check)

    with pytest.raises(ValueError, match='exact completed evidence closure'):
        archive.evidence(SimpleNamespace(authenticate=authenticate))
    assert archive.members == {}
    assert not list(study.glob('archive-01/*'))
