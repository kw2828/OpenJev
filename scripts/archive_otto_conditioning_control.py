"""Stream a completed audited control study into a lossless release archive."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'scripts/report_otto_conditioning_control.py'
REPORT_PIN = 'a535b1e8cec66f7e8afa9302bb7fbaf14c077ab0215a5b79e3d4f0e542613140'
PACKER = 'scripts/archive_otto_bellman_control.py'
PACKER_PIN = 'd167f723918da146bab8db45565354fc2542239bb01e78ae906dff1524f4442f'
LICENSE = 'tmp/otto-source-review-01/LICENSE'
LICENSE_PIN = '84009d54953e8b853b5dfb71dabf39d744bed8172e081488a376e0b0a979b376'
REPOSITORY_LICENSE_PIN = '679461bc4c0d0e17e0e414f1a767b1f72e40424725394e30df16d9b15eba270f'


def load_source(name, pin, module_name):
    path = ROOT / name
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
        raise ValueError('pinned publication source: ' + name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_source(REPORT, REPORT_PIN, '_conditioning_archive_report')
A = load_source(PACKER, PACKER_PIN, '_conditioning_archive_streams')
require, read, write, digest = R.require, R.read, R.write, R.digest
VERSION, ARCHIVE = 'otto-conditioning-control-archive-v1', 'openjev-otto-conditioning-control-v1.tar.gz'
STUDY = ROOT / 'output/otto-conditioning-control-v1'
FREEZE = 'ff7fd838fab60c66b880792752f7ee56be78e60e'
LIMITS = {'seconds': 600, 'rss_bytes': 4 * 1024**3, 'output_bytes': 8 * 1024**3}
QUAL_PAYLOADS = (R.PAYLOADS - {'native-setup.json', 'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl'}) | {
    'preparation.json', 'parity.jsonl'}
REPORT_PAYLOADS = {'report.md', 'plotted-values.json', 'otto-conditioning-control.png', 'otto-conditioning-control.svg'}
# Metadata-only pins of already published dependencies; the archives are not copied or fetched.
DEPENDENCIES = (
    ('otto-conditioning-v1', 13070613, '65e8d636900de1b873b247f7dffb00e85e315c0955ec3330acbd98e64a90b086',
     'archive-01', '87df2470d9729367d92c1015a1e400b25c9e81fd92019af2a172ce376f314fbc'),
    ('otto-capacity-v1', 98270157, 'd7fc565022a9cd10d14181b1d73e4dd659b648925ef340e0bf9c593050cb6bb2',
     'archive-01', 'd54ec424665686613d20005946aef77be1be670d0ee924452423ad82d873f6a3'),
    ('otto-return-value-v1', 599736656, '421f8976e4b2005f734593ab5d3c70226bc123a91c2e45256b399d0215a38fb8',
     'release-01', 'a12401c668240f5103d104555c7e26b3c582aa6e4bfcff896e2e5846991b2622'),
    ('otto-symmetry-head-v1', 244249285, '34c18bf08d50ed875a72168d498d5010f4947e1d0162370886b7a30248c01158',
     'release-01', '0daf91bc5d8c6aa4ee20529bd3dda8031cd0406aa156f214093e64efb5ffe1fe'),
)


class Archive(R.Report):
    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'archive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'archive RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'archive output cap')

    def evidence(self):
        super().authenticate()
        a, members = self.args, {}
        require(a.run == STUDY / 'run-01' and a.audit == STUDY / 'audit-01' and a.report == STUDY / 'report-01'
                and a.plan == STUDY / 'plan-01.json' and a.terminal == STUDY / 'run-process-01.terminal.json', 'declared closed study paths')

        def add(path, expected=None):
            path = R.regular(path)
            require(path.is_relative_to(ROOT), 'original repository-relative archive path')
            descriptor = digest(path, self.check)
            if expected is not None:
                require(descriptor == expected, 'original descriptor before packaging')
            name = path.relative_to(ROOT).as_posix()
            require(name not in members or members[name] == descriptor, 'deduplicated identical original path')
            members[name] = descriptor
            return path

        def closure(directory, pin, expected):
            receipt = self.manifest(directory, pin, expected)
            add(directory / 'receipt.json')
            for name, descriptor in receipt['files'].items():
                add(directory / name, descriptor)
            return receipt

        def pinned_record(descriptor):
            path = Path(descriptor['path'])
            if not path.is_absolute():
                path = ROOT / path
            return add(path, {k: descriptor[k] for k in ('bytes', 'sha256')})

        closure(a.run, a.receipt_sha256, R.PAYLOADS)
        closure(a.audit, a.audit_receipt_sha256, {'started.json', 'summary.json'})
        report = closure(a.report, a.report_receipt_sha256, REPORT_PAYLOADS)
        require(report['version'] == R.VERSION and report['source']['sha256'] == REPORT_PIN
                and report['helper_sha256'] == R.HELPER_PIN
                and report['model_calls'] == report['simulator_calls'] == report['training_calls'] == 0, 'qualified saved reporter')
        for key, path, pin in (('plan', a.plan, a.plan_sha256), ('worker', a.run / 'receipt.json', a.receipt_sha256),
                               ('terminal', a.terminal, a.terminal_sha256), ('audit', a.audit / 'receipt.json', a.audit_receipt_sha256)):
            require(report['inputs'][key]['path'] == str(path) and report['inputs'][key]['sha256'] == pin, 'same published evidence')
        plan = read(add(a.plan))
        require(len(plan['sources']) == 209, 'unchanged 209-source scientific closure')
        qualification = {key: pinned_record(value) for key, value in plan['qualification'].items()}
        qplan = read(qualification['plan'])
        qworker = closure(qualification['receipt'].parent, plan['qualification']['receipt']['sha256'], QUAL_PAYLOADS)
        qaudit = closure(qualification['audit'].parent, plan['qualification']['audit']['sha256'], {'started.json', 'summary.json'})
        require(qplan['mode'] == qworker['mode'] == qaudit['mode'] == 'qualify' and qworker['parity_passed'] is True
                and qworker['completed_episodes'] == 0 and qworker['sources'] == qplan['sources'] == plan['sources']
                and qaudit['agreement'] is True and qaudit['plan_sha256'] == qworker['plan_sha256'] == plan['qualification']['plan']['sha256']
                and qaudit['worker_sha256'] == plan['qualification']['receipt']['sha256']
                and qaudit['terminal_sha256'] == plan['qualification']['terminal']['sha256']
                and qaudit['source']['sha256'] == R.AUDITOR_PIN, 'same independently qualified deployment')
        expected_calls = {'model_load': 6, 'parity_restore': 6, 'parity_numpy_forward': 96, 'parity_torch_forward': 96}
        require({k: v['returned'] for k, v in qworker['calls'].items()} == expected_calls
                and qworker['pending'] == [] and all(v['attempted'] == v['returned'] for v in qworker['calls'].values()), 'complete qualification work')
        for directory, terminal_path, receipt in ((a.run, a.terminal, read(a.run / 'receipt.json')),
                                                  (qualification['receipt'].parent, qualification['terminal'], qworker)):
            started = read(directory / 'started.json')
            launch = add(Path(started['request']['supervision']))
            require(digest(launch, self.check)['sha256'] == receipt['supervision_sha256'], 'original supervisor pin')
            terminal = read(add(terminal_path))
            require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['group_absent'] is True
                    and terminal['timed_out'] is False and terminal['cleanup']['reaped'] is True, 'original successful parent')
            require(read(launch) == started['launch'], 'original supervisor contents')
            for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds'):
                require(terminal[key] == started['launch'][key], 'same qualification/full process')
            require(receipt['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'], 'closed parent timing')
            add(launch.with_name(launch.name.removesuffix('.launch.json') + '.log'))
        require(a.witness.parent == STUDY and digest(R.regular(a.witness), self.check)['sha256'] == a.witness_sha256, 'external execution witness pin')
        witness = read(add(a.witness))
        require(witness['status'] == 'completed' and witness['actual_tool_exit_code'] == 0
                and type(witness['actual_session_id']) is int and witness['actual_session_id'] > 0
                and isinstance(witness['actual_completion_chunk'], str) and witness['actual_completion_chunk']
                and witness['worker_sha256'] == a.receipt_sha256
                and witness['plan_sha256'] == a.plan_sha256 and witness['terminal_sha256'] == a.terminal_sha256, 'original execution witness identity')
        for name in ('qualification-execution-01.json', 'run-dispatch-01.json', 'qualification-plan-verification-01.json',
                     'full-plan-verification-01.json', 'report-source-review-01.json', 'seed-ledger-review-01.json'):
            add(STUDY / name)
        # Includes the exact six heads, three kernels and provenance receipt inputs.
        inputs = {role: pinned_record(descriptor) for role, descriptor in plan['inputs'].items()}
        require(sum(key.startswith('head_') for key in inputs) == 6 and sum(key.startswith('kernel_') for key in inputs) == 3,
                'six unchanged heads and three kernels')
        previous = read(inputs['conditioning_receipt'])
        for name in ('summary.json', 'preparation.json'):
            add(inputs['conditioning_receipt'].parent / name, previous['files'][name])
        old_audit = read(inputs['conditioning_audit'])
        for name, descriptor in old_audit['files'].items():
            add(inputs['conditioning_audit'].parent / name, descriptor)
        sources = {}
        for name, pin in plan['sources'].items():
            path = A.contained(ROOT, name)
            require(digest(path, self.check)['sha256'] == pin, 'frozen scientific source')
            add(path)
            sources[name] = members[name]
        for path, pin in ((ROOT / REPORT, REPORT_PIN), (ROOT / PACKER, PACKER_PIN), (ROOT / LICENSE, LICENSE_PIN),
                          (ROOT / 'LICENSE', REPOSITORY_LICENSE_PIN), (A.REPORT, A.REPORT_PIN),
                          (R.HELPER, R.HELPER_PIN), (ROOT / R.H.CLOCK, R.H.CLOCK_PIN)):
            require(digest(R.regular(path), self.check)['sha256'] == pin, 'qualified publication helper')
            name = add(path).relative_to(ROOT).as_posix()
            sources[name] = members[name]
        name = add(Path(__file__).resolve()).relative_to(ROOT).as_posix()
        sources[name] = members[name]
        dependencies = []
        for tag, size, pin, folder, manifest_pin in DEPENDENCIES:
            path = ROOT / 'output' / tag / folder / 'manifest.json'
            require(digest(R.regular(path), self.check)['sha256'] == manifest_pin, 'historical dependency manifest')
            add(path)
            dependencies.append({'release': f'https://github.com/kw2828/OpenJev/releases/tag/{tag}',
                'archive_url': f'https://github.com/kw2828/OpenJev/releases/download/{tag}/openjev-{tag}.tar.gz',
                'archive': {'bytes': size, 'sha256': pin}, 'manifest_path': path.relative_to(ROOT).as_posix(),
                'manifest_sha256': manifest_pin})
        self.receipt['inputs'].update(report={'path': str(a.report / 'receipt.json'), **digest(a.report / 'receipt.json', self.check)},
                                     execution_witness={'path': str(a.witness), **digest(a.witness, self.check)})
        return dict(sorted(members.items())), sources, dependencies


def restore_text(plan_pin):
    return f'''# Restore conditioning-control evidence

This release preserves the complete current worker, qualification, both saved
audits, report, six unchanged fitted heads, three known kernels and all 209
scientific source files. It adds original process records and publication
source/byte witnesses. Archival success is not scientific success.

1. Download `{ARCHIVE}`, `manifest.json`, `dependencies.json`, `RESTORE.md`
   and `SHA256SUMS.txt` from the same release. Verify the latter before use:
   `shasum -a 256 -c SHA256SUMS.txt`.
2. In a fresh repository checkout at `{FREEZE}`, inspect the archive inventory
   and extract relative to that root. Do not overwrite an existing study.
3. Verify every restored relative path, byte count and SHA256 against
   `manifest.json`. The original plan is `{plan_pin}`. Historical absolute
   runtime paths in JSON are intentionally preserved, not rewritten.

The packager streams all archived members back, reads the complete gzip
trailer, then rechecks original files and source bytes. It runs no model,
simulator, training, posterior or scientific scoring operation.

This is self-contained CURRENT evidence, not a standalone recursive archive
of every earlier study or an installed runtime. `dependencies.json` lists
already published conditioning, capacity, scalar-return and original teacher
archives with exact hashes and sizes. Follow their own restoration manifests
for older lineage dependencies. The current archive includes six checkpoint
files and kernels, but does not duplicate old full TRAIN/VALID caches or every
earlier closed payload directory. The unchanged strict audit authenticator
requires those dependencies, including the old caches for qualification replay.

The source freeze and its package manifest remain authoritative. Preserve the
recorded NumPy/Python/Torch runtime when replaying numerical audits. A model-free
byte verification requires only the archive, manifests and checksum tools.
'''


def execute(args):
    archive = Archive(args)
    archive.receipt.update(version=VERSION, limits=LIMITS, source_freeze=FREEZE,
                           scope='Saved byte preservation only; historical dependencies remain explicit.')
    require(args.output.is_absolute() and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), 'exclusive archive output')
    args.output.mkdir(parents=True, exist_ok=False)
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        require(digest(R.regular(ROOT / R.H.CLOCK))['sha256'] == R.H.CLOCK_PIN, 'qualified native clock')
        archive.clock = R.H.load(ROOT / R.H.CLOCK, '_conditioning_archive_clock').SuspendClock()
        archive.start = archive.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('archive deadline')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        members, sources, dependencies = archive.evidence()
        target = args.output / ARCHIVE
        A.pack(ROOT, members, target, archive.check)
        A.unchanged(ROOT, sources, archive.check)
        identity = digest(target, archive.check)
        require(identity['bytes'] < 2 * 1024**3, 'single GitHub asset must be below 2 GiB; preserve archive for separately verified splitting')
        write(args.output / 'manifest.json', {'version': VERSION, 'status': 'verified', 'members': members,
              'member_count': len(members), 'archive': {'name': ARCHIVE, **identity}, 'source_freeze': FREEZE,
              'scientific_source_count': 209, 'sources_verified_unchanged': sources, 'inputs': archive.receipt['inputs'],
              'source': digest(Path(__file__).resolve(), archive.check)})
        write(args.output / 'dependencies.json', {'historical_archives': dependencies, 'downloaded_by_packager': False,
              'scope': 'Direct historical releases, including teacher data; their manifests describe additional older lineage dependencies.'})
        with (args.output / 'RESTORE.md').open('x') as stream:
            stream.write(restore_text(args.plan_sha256))
        with (args.output / 'SHA256SUMS.txt').open('x') as stream:
            for path in (target, args.output / 'manifest.json', args.output / 'dependencies.json', args.output / 'RESTORE.md'):
                stream.write(f'{digest(path, archive.check)["sha256"]}  {path.name}\n')
        archive.check()
        archive.receipt.update(status='completed', verified_members=len(members), scientific_sources=209,
            source=digest(Path(__file__).resolve(), archive.check), clock_backend=archive.clock.backend,
            wall_seconds=(archive.clock.now_ns() - archive.start) / 1e9,
            files={p.name: digest(p, archive.check) for p in args.output.iterdir() if p.is_file()})
        write(args.output / 'receipt.json', archive.receipt)
        archive.check()
        return archive.receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.output / 'receipt.json').exists():
                (args.output / 'receipt.json').rename(args.output / 'invalid-completed-receipt.json')
            write(args.output / 'failed.json', {**archive.receipt, 'status': 'failed', 'error': repr(error)})
        except BaseException as secondary:  # noqa: BLE001 - Retain original failure and partial archive.
            error.add_note(f'Failure publication: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'audit', 'report', 'witness', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-receipt-sha256', 'report-receipt-sha256', 'witness-sha256'):
        parser.add_argument('--' + flag, required=True)
    execute(parser.parse_args())
