"""Tiny byte fixtures only; no scientific archives or model execution."""
from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_bellman_archive', ROOT / 'scripts/archive_otto_bellman_control.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture_files(tmp_path):
    root = tmp_path / 'inputs'
    root.mkdir()
    for name, value in (('nested/empty', b''), ('weights.npz', bytes(range(256)) * 1024), ('text.json', b'{"x": 1}\n')):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return root, {p.relative_to(root).as_posix(): M.digest(p) for p in root.rglob('*') if p.is_file()}


def test_exact_members_bytes_and_originals_without_extraction(tmp_path):
    root, members = fixture_files(tmp_path)
    archive = tmp_path / 'test.tar.gz'
    M.pack(root, members, archive)
    with tarfile.open(archive, 'r:gz') as saved:
        assert set(saved.getnames()) == set(members)
        for entry in saved:
            assert saved.extractfile(entry).read() == (root / entry.name).read_bytes()
    assert len(list(root.rglob('*'))) == 4  # Three original files, one directory.
    with pytest.raises(FileExistsError):
        M.pack(root, members, archive)


@pytest.mark.parametrize('kind', ['changed', 'missing', 'unsafe', 'symlink'])
def test_original_corruption_and_unsafe_paths_reject(tmp_path, kind):
    root, members = fixture_files(tmp_path)
    if kind == 'changed':
        (root / 'text.json').write_bytes(b'{"x": 2}\n')
        with pytest.raises(ValueError, match='lossless'):
            M.pack(root, members, tmp_path / 'changed.tar.gz')
    elif kind == 'missing':
        (root / 'text.json').unlink()
        with pytest.raises(ValueError):
            M.unchanged(root, members)
    elif kind == 'unsafe':
        with pytest.raises(ValueError, match='safe original'):
            M.pack(root, {'../escape': {'bytes': 0, 'sha256': '0' * 64}}, tmp_path / 'unsafe.tar.gz')
    else:
        (root / 'link').symlink_to(root / 'text.json')
        with pytest.raises(ValueError, match='nonsymlink'):
            M.contained(root, 'link')


def test_full_gzip_integrity_and_duplicate_members(tmp_path):
    root, members = fixture_files(tmp_path)
    archive = tmp_path / 'test.tar.gz'
    M.pack(root, members, archive)
    data = bytearray(archive.read_bytes())
    data[-8] ^= 1  # Gzip CRC, after the tar end marker.
    archive.write_bytes(data)
    with pytest.raises((gzip.BadGzipFile, tarfile.ReadError)):
        M.verify_archive(archive, members)
    duplicate = tmp_path / 'duplicate.tar.gz'
    with tarfile.open(duplicate, 'x:gz') as stream:
        for _ in range(2):
            info = tarfile.TarInfo('text.json')
            value = (root / 'text.json').read_bytes()
            info.size = len(value)
            stream.addfile(info, io.BytesIO(value))
    with pytest.raises(ValueError, match='exact regular'):
        M.verify_archive(duplicate, {'text.json': members['text.json']})


def test_failure_keeps_partial_archive_and_existing_output(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / 'release', plan_sha256='0' * 64)
    monkeypatch.setattr(M.Archive, 'evidence', lambda self: ({}, {}))
    def fail(root, members, path, check):
        path.write_bytes(b'partial archive')
        raise ValueError('fabricated copy failure')
    monkeypatch.setattr(M, 'pack', fail)
    with pytest.raises(ValueError, match='fabricated copy'):
        M.execute(args)
    assert (args.output / M.ARCHIVE).read_bytes() == b'partial archive'
    assert M.read(args.output / 'failed.json')['status'] == 'failed'
    assert not (args.output / 'receipt.json').exists()
    with pytest.raises(ValueError, match='exclusive'):
        M.execute(args)


def test_restore_discloses_prior_inputs_and_no_scientific_success():
    text = M.restore_text('a' * 64)
    assert '100 original paths' in text and '8d560b9' in text
    assert 'Prior scalar-study inputs are separately required' in text
    assert 'Archival success is not scientific success' in text
