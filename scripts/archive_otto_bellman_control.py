"""Losslessly package a completed audited study; no scientific computations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import resource
import signal
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'scripts/report_otto_bellman_control.py'
REPORT_PIN = '8fe5f1db28250e41c41291198ddb064dae93a8f1dc8a6f3c4cdd1537a2c0aa6c'
if hashlib.sha256(REPORT.read_bytes()).hexdigest() != REPORT_PIN:
    raise ValueError('qualified report source before import')
_spec = importlib.util.spec_from_file_location('_bellman_archive_report', REPORT)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
require, read, write, digest = R.require, R.read, R.write, R.digest
VERSION = 'otto-bellman-control-raw-archive-v1'
FREEZE_COMMIT = '8d560b9'
LIMITS = {'seconds': 600, 'rss_bytes': 4 * 1024**3, 'output_bytes': 8 * 1024**3}
ARCHIVE = 'openjev-otto-bellman-control-v1.tar.gz'


def contained(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name, 'safe original path')
    return R.regular(root / relative)


def verify_archive(path, members, check=lambda: None):
    seen = set()
    with tarfile.open(path, 'r:gz') as archive:
        for entry in archive:
            check()
            require(entry.isfile() and entry.name in members and entry.name not in seen, 'exact regular archive members')
            value, size = hashlib.sha256(), 0
            with archive.extractfile(entry) as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    check()
                    value.update(chunk)
                    size += len(chunk)
            require({'sha256': value.hexdigest(), 'bytes': size} == members[entry.name], 'lossless archived member')
            seen.add(entry.name)
    require(seen == members.keys(), 'complete archive member coverage')
    # Tar iteration may stop at its end marker before gzip verifies the trailer.
    with gzip.open(path, 'rb') as stream:
        for _ in iter(lambda: stream.read(1024**2), b''):
            check()


def unchanged(root, descriptors, check=lambda: None):
    for name, identity in descriptors.items():
        require(digest(contained(root, name), check) == identity, f'unchanged original: {name}')


def pack(root, members, path, check=lambda: None):
    class CheckedReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size):
            check()
            return self.stream.read(size)

    with tarfile.open(path, 'x:gz', compresslevel=6, copybufsize=1024**2) as archive:
        for name in members:
            source = contained(root, name)
            info = archive.gettarinfo(source, arcname=name)
            require(info.isfile() and info.size == members[name]['bytes'], 'regular original size')
            with source.open('rb') as stream:
                archive.addfile(info, CheckedReader(stream))
    verify_archive(path, members, check)
    unchanged(root, members, check)


class Archive(R.Report):
    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'archive native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'archive RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'archive output cap')

    def evidence(self):
        super().authenticate()
        a = self.args
        study = ROOT / 'output/otto-bellman-control-v1'
        require(a.run == study / 'run-01' and a.audit == study / 'audit-01' and a.plan == study / 'plan-01.json'
                and a.terminal == study / 'run-process-01.terminal.json', 'declared original study paths')
        witness_path = R.regular(study / 'execution-witness.json')
        require(digest(witness_path, self.check)['sha256'] == a.witness_sha256, 'external execution witness pin')
        witness = read(witness_path)
        require(witness['status'] == 'completed' and witness['actual_tool_exit_code'] == 0
                and type(witness['actual_session_id']) is int and witness['actual_session_id'] > 0
                and isinstance(witness['actual_completion_chunk'], str) and witness['actual_completion_chunk']
                and witness['worker_sha256'] == a.receipt_sha256 and witness['plan_sha256'] == a.plan_sha256
                and witness['terminal_sha256'] == a.terminal_sha256, 'original execution witness joins')
        launch = R.regular(study / 'run-process-01.launch.json')
        require(read(a.run / 'started.json')['request']['supervision'] == str(launch), 'original supervisor path')
        paths = [a.run / n for n in sorted(R.payload_names() | {'receipt.json'})]
        paths += [a.plan, witness_path, launch, study / 'run-process-01.log', a.terminal]
        paths += [a.audit / n for n in ('started.json', 'summary.json', 'receipt.json')]
        members = {p.relative_to(ROOT).as_posix(): digest(R.regular(p), self.check) for p in paths}
        require(len(members) == len(paths) == 100, 'all 100 distinct original paths')
        plan = read(a.plan)
        sources = {name: digest(contained(ROOT, name), self.check) for name in plan['sources']}
        for path, pin in ((REPORT, REPORT_PIN), (R.HELPER, R.HELPER_PIN), (ROOT / R.H.CLOCK, R.H.CLOCK_PIN)):
            require(digest(R.regular(path), self.check)['sha256'] == pin, 'qualified publication helper')
            sources[path.relative_to(ROOT).as_posix()] = digest(path, self.check)
        sources[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__), self.check)
        self.receipt['inputs']['execution_witness'] = {'path': str(witness_path), **digest(witness_path, self.check)}
        return members, sources


def restore_text(plan_pin):
    return f'''# Restore Bellman-control raw evidence

This archive preserves 100 original paths and bytes: 91 worker payloads and
receipt, the plan, root execution witness, three supervisor files, and three
independent audit files. It includes all continuation and target checkpoints,
target arrays, public trajectories, fit records and operation journals.
Archival success is not scientific success and does not revise any failed gate.

1. Download `{ARCHIVE}`, `manifest.json`, `SHA256SUMS.txt`, and `RESTORE.md`
   from the same release. Run `shasum -a 256 -c SHA256SUMS.txt` there.
2. Use a fresh checkout; do not overwrite an existing experiment directory:

```sh
git clone https://github.com/kw2828/OpenJev.git OpenJev-bellman-control
cd OpenJev-bellman-control
git checkout {FREEZE_COMMIT}
tar -xzf /absolute/path/to/{ARCHIVE}
```

The source freeze is `{FREEZE_COMMIT}`; plan SHA256 is `{plan_pin}`.
Members are regular files under `output/otto-bellman-control-v1/`. Extraction
is relative to the repository root. `manifest.json` lists each original path,
SHA256 and byte count; verify every restored file against that mapping.
Absolute runtime paths in historical receipts are preserved, not rewritten.

The publisher streams every archived member back, reads the gzip trailer,
then checks originals and pinned sources again. No arrays, models, training
or simulator are executed. These checks establish byte preservation only.

Prior scalar-study inputs are separately required for lineage authentication
and numerical replay. This archive does not duplicate the earlier TRAIN/VALID
caches, checkpoints, their upstream evidence, or the pinned runtime. Obtain
those earlier releases and source artifacts named by the frozen plan before
replaying the full audit. Byte verification needs only the archive and its
manifest, with no model frameworks or earlier datasets.
'''


def execute(args):
    a = Archive(args)
    a.receipt.update(version=VERSION, limits=LIMITS, source_freeze=FREEZE_COMMIT,
                     scope='Lossless saved evidence publication only; archival success is not scientific success.')
    require(args.output.is_absolute() and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), 'exclusive archive output')
    args.output.mkdir(parents=True, exist_ok=False)
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        require(digest(R.regular(ROOT / R.H.CLOCK))['sha256'] == R.H.CLOCK_PIN, 'qualified native clock')
        a.clock = R.H.load(ROOT / R.H.CLOCK, '_bellman_archive_clock').SuspendClock()
        a.start = a.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('archive emergency cap')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        members, sources = a.evidence()
        archive = args.output / ARCHIVE
        pack(ROOT, members, archive, a.check)
        unchanged(ROOT, sources, a.check)
        manifest = {'version': VERSION, 'status': 'verified', 'members': members, 'member_count': len(members),
                    'archive': {'name': ARCHIVE, **digest(archive, a.check)}, 'inputs': a.receipt['inputs'],
                    'source_freeze': FREEZE_COMMIT, 'sources_verified_unchanged': sources,
                    'scope': a.receipt['scope'], 'source': digest(Path(__file__), a.check)}
        write(args.output / 'manifest.json', manifest)
        with (args.output / 'RESTORE.md').open('x') as stream:
            stream.write(restore_text(args.plan_sha256))
        with (args.output / 'SHA256SUMS.txt').open('x') as stream:
            for path in (archive, args.output / 'manifest.json', args.output / 'RESTORE.md'):
                stream.write(f'{digest(path, a.check)["sha256"]}  {path.name}\n')
        a.check()
        a.receipt.update(status='completed', verified_members=len(members), source=manifest['source'],
                         clock_backend=a.clock.backend, wall_seconds=(a.clock.now_ns() - a.start) / 1e9,
                         files={p.name: digest(p, a.check) for p in args.output.iterdir() if p.is_file()})
        write(args.output / 'receipt.json', a.receipt)
        a.check()
        return a.receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.output / 'receipt.json').exists():
                (args.output / 'receipt.json').rename(args.output / 'invalid-completed-receipt.json')
            write(args.output / 'failed.json', {**a.receipt, 'status': 'failed', 'error': repr(error)})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure and any partial archive.
            error.add_note(f'Failure publication: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'audit', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-receipt-sha256', 'witness-sha256'):
        parser.add_argument('--' + flag, required=True)
    execute(parser.parse_args())
