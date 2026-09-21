"""Losslessly package one closed action-head run, verifying every member stream.

No array/model decoding, simulator calls, extraction, source-data mutation, or
network publication. The archive and packaging evidence use an exclusive folder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import signal
import stat
import tarfile
import time
import traceback
from pathlib import Path

VERSION = 'otto-action-head-archive-v1'
SECONDS = 180
MAX_OUTPUT = 2 * 1024**3
BLOCK = 1024**2
ARCHIVE_NAME = 'otto-action-head-v1-run-01.tar.gz'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path, check):
    value = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while data := stream.read(BLOCK):
            check()
            value.update(data)
            size += len(data)
    return {'bytes': size, 'sha256': value.hexdigest()}


def regular(path):
    return (path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents))
            and stat.S_ISREG(path.stat().st_mode))


class LimitedWriter:
    def __init__(self, stream, check):
        self.stream, self.check, self.bytes = stream, check, 0

    def write(self, data):
        self.check()
        require(self.bytes + len(data) <= MAX_OUTPUT - 1024**2, 'archive output budget')
        returned = self.stream.write(data)
        require(returned == len(data), 'complete compressed write')
        self.bytes += returned
        return returned

    def flush(self):
        return self.stream.flush()


def execute(args):
    require(all(p.is_absolute() for p in (args.run, args.terminal, args.output)), 'absolute explicit paths')
    require(args.run.name == 'run-01', 'fixed archive root name')
    require(not args.output.resolve().is_relative_to(args.run.resolve()), 'archive outside source run')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGALRM)

    def timeout(_signum, _frame):
        raise TimeoutError('180-second packaging deadline')

    def check():
        require(time.monotonic() - started < SECONDS, '180-second packaging deadline')

    manifest = {'version': VERSION, 'status': 'started', 'limits': {'seconds': SECONDS, 'output_bytes': MAX_OUTPUT},
                'source': {'path': str(Path(__file__).resolve())}, 'source_run': str(args.run),
                'receipt_sha256': args.receipt_sha256, 'terminal_sha256': args.terminal_sha256,
                'model_calls': 0, 'simulator_calls': 0, 'extracted_files': 0, 'published': False}
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, SECONDS)
    try:
        require(args.run.is_dir() and not any(p.is_symlink() for p in (args.run, *args.run.parents)), 'regular source directory')
        receipt_path = args.run / 'receipt.json'
        require(regular(receipt_path) and regular(args.terminal), 'regular completion artifacts')
        require(digest(receipt_path, check)['sha256'] == args.receipt_sha256, 'external worker receipt pin')
        require(digest(args.terminal, check)['sha256'] == args.terminal_sha256, 'external parent terminal pin')
        receipt = json.loads(receipt_path.read_text())
        terminal = json.loads(args.terminal.read_text())
        require(receipt['status'] == 'completed' and receipt['completed_fits'] == 12
                and receipt['completed_episodes'] == 1536, 'complete original scientific run')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0
                and terminal['timed_out'] is False and terminal['group_absent'] is True,
                'actual successful parent terminal')
        command = terminal['command']
        require(command.count('--output') == 1 and command[command.index('--output') + 1] == str(args.run), 'parent run path join')
        require(terminal['started_ns'] <= receipt['started_ns'] <= receipt['finished_ns'] <= terminal['finished_ns'],
                'worker timing contained by parent')
        expected = dict(receipt['files'])
        require(len(expected) == 48 and 'receipt.json' not in expected, 'complete forty-eight original payloads')
        expected['receipt.json'] = digest(receipt_path, check)
        require({p.name for p in args.run.iterdir()} == set(expected), 'exact forty-nine file closure')
        require(sum(name.startswith('head-') and name.endswith('.npz') for name in expected) == 12, 'all twelve checkpoints')
        for name, record in expected.items():
            require(Path(name).name == name and regular(args.run / name), 'safe flat regular archive member')
            require(set(record) == {'bytes', 'sha256'} and digest(args.run / name, check) == record,
                    f'closed original bytes: {name}')
        manifest['source'].update(digest(Path(__file__).resolve(), check))
        manifest['terminal'] = {'path': str(args.terminal), **digest(args.terminal, check)}
        manifest['member_count'] = len(expected)
        manifest['raw_bytes'] = sum(record['bytes'] for record in expected.values())
        archive = args.output / ARCHIVE_NAME
        with archive.open('xb') as raw:
            bounded = LimitedWriter(raw, check)
            with (
                gzip.GzipFile(filename='', mode='wb', fileobj=bounded, compresslevel=3, mtime=0) as compressed,
                tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as tar,
            ):
                for name in sorted(expected):
                    check()
                    info = tarfile.TarInfo(f'run-01/{name}')
                    info.size, info.mode, info.mtime = expected[name]['bytes'], 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with (args.run / name).open('rb') as source:
                        tar.addfile(info, source)
        checked = {}
        with gzip.open(archive, 'rb') as decoded, tarfile.open(fileobj=decoded, mode='r|') as tar:
            for member in tar:
                check()
                name = member.name.removeprefix('run-01/')
                require(member.name == f'run-01/{name}' and name in expected and name not in checked
                        and member.isfile() and member.size == expected[name]['bytes'], 'exact unique archive member')
                require(member.uid == member.gid == member.mtime == 0 and member.mode == 0o644,
                        'deterministic regular member metadata')
                archive_hash, original_hash, size = hashlib.sha256(), hashlib.sha256(), 0
                stream = tar.extractfile(member)
                require(stream is not None, 'readable regular archive member stream')
                with stream, (args.run / name).open('rb') as source:
                    while data := stream.read(BLOCK):
                        check()
                        original = source.read(len(data))
                        require(data == original, f'archive/source byte identity: {name}')
                        archive_hash.update(data)
                        original_hash.update(original)
                        size += len(data)
                    require(source.read(1) == b'', 'unchanged source length')
                observed = {'bytes': size, 'sha256': archive_hash.hexdigest()}
                require(observed == expected[name] and original_hash.hexdigest() == observed['sha256'],
                        f'archive/original/receipt triple hash equality: {name}')
                checked[name] = {'archive_path': member.name, **observed, 'original_byte_identity': True}
            # Consume the gzip trailer, verifying its CRC and complete length.
            while decoded.read(BLOCK):
                check()
        require(set(checked) == set(expected), 'all forty-nine archived streams verified')
        require(digest(receipt_path, check)['sha256'] == args.receipt_sha256
                and digest(args.terminal, check)['sha256'] == args.terminal_sha256, 'unchanged external completion pins')
        archive_record = digest(archive, check)
        instructions = (
            '# Restore the complete immutable action-head run\n\n'
            f'Archive: `{ARCHIVE_NAME}`\n\n'
            f'SHA-256: `{archive_record["sha256"]}`\n\n'
            '1. Verify the archive SHA-256 against this manifest before extraction.\n'
            '2. Create a new empty directory outside any existing scientific run.\n'
            f'3. Extract with `tar -xzf {ARCHIVE_NAME} -C /absolute/path/to/empty-directory`.\n'
            '4. The resulting `run-01/` contains all 49 original files, including the original receipt and 12 model checkpoints.\n'
            '5. Compare every restored file size and SHA-256 to the `members` entries in `manifest.json`. '
            'The 48 payload hashes must also equal `run-01/receipt.json`; verify the receipt itself against '
            f'`{args.receipt_sha256}`.\n\n'
            'No model loader is needed to restore or verify this archive. NPZ contents were never decoded during packaging. '
            'Study source, plan, parent terminal, and independent audit remain separate repository evidence. '
            'This archive preserves the completed run; it does not alter its scientific result.\n')
        with (args.output / 'RESTORE.md').open('x') as stream:
            stream.write(instructions)
        manifest.update(status='completed', archive={'path': ARCHIVE_NAME, **archive_record}, members=checked,
                        exact_payload_count=48, checkpoint_count=12, verified_member_count=len(checked),
                        restoration=digest(args.output / 'RESTORE.md', check),
                        compression={'format': 'tar.gz', 'gzip_level': 3, 'gzip_mtime': 0,
                                     'tar_format': 'USTAR', 'normalized_metadata': True},
                        wall_seconds=time.monotonic() - started)
        require(sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) < MAX_OUTPUT - 1024**2, 'total extra output budget')
        write(args.output / 'manifest.json', manifest)
        check()
        print(json.dumps({'status': 'completed', 'archive': str(archive), **archive_record,
                          'verified_members': len(checked), 'wall_seconds': manifest['wall_seconds'],
                          'manifest_sha256': digest(args.output / 'manifest.json', check)['sha256']}))
        return manifest
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                        wall_seconds=time.monotonic() - started)
        try:
            write(args.output / 'failed.json', manifest)
        except BaseException as publication_error:  # noqa: BLE001 - preserve the original packaging failure.
            error.add_note(f'Failed to save packaging failure: {publication_error!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'terminal', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    for name in ('receipt-sha256', 'terminal-sha256'):
        parser.add_argument(f'--{name}', required=True)
    return execute(parser.parse_args())


if __name__ == '__main__':
    main()
