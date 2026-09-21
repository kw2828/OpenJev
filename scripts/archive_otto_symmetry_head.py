"""Archive one completed audited study, preserving every original run byte."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return {'sha256': value.hexdigest(), 'bytes': path.stat().st_size}


def regular(path):
    require(path.is_absolute() and path.is_relative_to(ROOT) and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'regular contained input')
    return path


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--audit-sha256', required=True)
    args = parser.parse_args()
    study = ROOT/'output/otto-symmetry-head-v1'
    run, audit, release = study/'run-01', study/'audit-01', study/'release-01'
    worker_path, audit_path = regular(run/'receipt.json'), regular(audit/'receipt.json')
    require(descriptor(worker_path)['sha256'] == args.worker_sha256
            and descriptor(audit_path)['sha256'] == args.audit_sha256, 'external completion pins')
    worker, checked = json.loads(worker_path.read_text()), json.loads(audit_path.read_text())
    require(worker['status'] == checked['status'] == 'completed' and checked['agreement'] is True
            and worker['version'] == 'otto-symmetry-head-v1'
            and checked['version'] == 'otto-symmetry-head-saved-audit-v1'
            and checked['plan_sha256'] == worker['plan_sha256']
            and checked['worker_sha256'] == args.worker_sha256 and worker['completed_episodes'] == 720
            and worker['completed_stage_fits'] == 12 and not worker['pending'], 'completed audited study')
    names = set(worker['files']) | {'receipt.json'}
    require({p.name for p in run.iterdir()} == names and len(names) == 42, 'all original run files')
    paths = [regular(run/name) for name in sorted(names)]
    for path in paths:
        if path.name != 'receipt.json':
            require(descriptor(path) == worker['files'][path.name], 'closed scientific payload identity')
    paths += [regular(study/name) for name in ('plan-01.json', 'execution-witness.json',
              'run-process-01.launch.json', 'run-process-01.log', 'run-process-01.terminal.json')]
    terminal = json.loads((study/'run-process-01.terminal.json').read_text())
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['group_absent'] is True
            and descriptor(study/'run-process-01.terminal.json')['sha256'] == checked['terminal_sha256']
            and descriptor(study/'plan-01.json')['sha256'] == worker['plan_sha256'], 'audited parent and plan')
    for name, pin in worker['sources'].items():
        require(descriptor(regular(ROOT/name))['sha256'] == pin, 'unchanged scientific sources')
    members = {p.relative_to(ROOT).as_posix(): descriptor(p) for p in paths}
    require(len(members) == 47, 'complete unique archive members')
    release.mkdir(exist_ok=False)
    started = time.monotonic()
    archive_path = release/'openjev-otto-symmetry-head-v1.tar.gz'
    try:
        with tarfile.open(archive_path, 'x:gz', compresslevel=6) as archive:
            for name in members:
                archive.add(ROOT/name, arcname=name, recursive=False)
        seen = set()
        with tarfile.open(archive_path, 'r:gz') as archive:
            for entry in archive:
                require(entry.isfile() and entry.name in members and entry.name not in seen, 'exact archive member set')
                stream, digest, size = archive.extractfile(entry), hashlib.sha256(), 0
                for chunk in iter(lambda stream=stream: stream.read(1024**2), b''):
                    digest.update(chunk)
                    size += len(chunk)
                require({'sha256': digest.hexdigest(), 'bytes': size} == members[entry.name], 'lossless archive readback')
                seen.add(entry.name)
        require(seen == members.keys(), 'complete archive coverage')
        with gzip.open(archive_path, 'rb') as stream:
            for _ in iter(lambda: stream.read(1024**2), b''):
                pass
        for name, value in members.items():
            require(descriptor(ROOT/name) == value, 'unchanged originals after archival')
        manifest = {'version': 'otto-symmetry-head-raw-archive-v1', 'status': 'verified',
                    'worker_sha256': args.worker_sha256, 'audit_sha256': args.audit_sha256,
                    'source': descriptor(Path(__file__)), 'archive': {'name': archive_path.name, **descriptor(archive_path)},
                    'members': members, 'seconds': time.monotonic()-started,
                    'scope': 'All41 payloads and receipt, including12 checkpoints and full data/trajectories, plus plan, execution witness and3 supervisor files. Saved-only byte preservation; no model/native/training calls.'}
        write(release/'manifest.json', manifest)
        with (release/'SHA256SUMS.txt').open('x') as stream:
            for path in (archive_path, release/'manifest.json'):
                stream.write(descriptor(path)['sha256']+'  '+path.name+'\n')
        print(json.dumps({'status': 'verified', 'archive': manifest['archive'], 'members': len(members)}), flush=True)
    except BaseException as error:
        try:
            write(release/'failed.json', {'status': 'failed', 'error': repr(error)})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original archive failure.
            error.add_note(f'Failure receipt publication: {secondary!r}')
        raise


if __name__ == '__main__':
    main()
