"""Losslessly archive completed spatial-study evidence without numerical decoding.

Invoke only after the original worker, independent audit and report complete.
Historical datasets remain separate, explicitly pinned release dependencies.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import resource
import signal
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/otto-spatial-study-v1'
VERSION = 'otto-spatial-study-archive-v1'
ARCHIVE = 'openjev-otto-spatial-study-v1.tar.gz'
FREEZE = '1971472bd4992d888c8a8f18032d2c604ad6012a'
REPORT = 'scripts/report_otto_spatial_study.py'
REPORT_PIN = '192e0b8e3f7eb630848aa4df0881c7153d3180a1ca479b1a30ade25af0bface2'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
LIMITS = {'seconds': 600, 'rss_bytes': 4*1024**3, 'output_bytes': 2*1024**3}
ENGINEERING = {
    'output/otto-spatial-study-v1/engineering-01/receipt.json': '73d84ebab71d2466809aa2a62f3086afc16a8ac8a4693dd07a83e5bbbdf066df',
    'output/otto-spatial-study-v1/engineering-02/receipt.json': 'fc4fd666e017b9e822df7f106e90ffdb22c065846844f5ad100a6080041101ff',
    'output/otto-spatial-v1/audit-engineering-01/receipt.json': 'a153136b48f0ee1b8c92e273c2b662ec9ad748a2eceee73358bb467c454f0c21',
    'output/otto-spatial-v1/audit-engineering-02/receipt.json': '8f1d4b2476c9bac63fc25c3ae0d537b5023b3055aa5be13d438d38122c48fb5f',
    'output/otto-spatial-qualification-v1/engineering-01/receipt.json': '42e6d2a419a96149e15f3b5564e836056f4e15ccf399d5fd427ff67e73401ef8',
    'output/otto-spatial-qualification-v1/engineering-02/receipt.json': '8a43411a16e935fd4496d7413614bb221e77a70ad400b469d9cf82dc4ad380d5',
}
WITNESSES = {
    'output/otto-spatial-study-v1/pre-dispatch-witness-01.json': '0cb4ec2814a484472e2b3975f52c1f041405c0d9e413eece56b3bf1d5a2615d5',
    'output/otto-spatial-study-v1/dispatch-witness-01.json': 'b6558691751315c3e0b7e3533a52f6b014cad3ad4b4beaf8a4750c712c894fd2',
}
LICENSES = {
    'LICENSE': '679461bc4c0d0e17e0e414f1a767b1f72e40424725394e30df16d9b15eba270f',
    'third_party/otto/LICENSE': '84009d54953e8b853b5dfb71dabf39d744bed8172e081488a376e0b0a979b376',
    'tmp/otto-source-review-01/LICENSE': '84009d54953e8b853b5dfb71dabf39d744bed8172e081488a376e0b0a979b376',
}
# Original published archive identities, copied from the pinned prior packager.
DEPENDENCY_SOURCE = ('scripts/archive_otto_conditioning_control.py', '4469dfe09541423df35f1199d1b7ab8ba14532317a201627e5a7e7741a60c428')
DEPENDENCIES = (
    ('otto-capacity-v1', 'archive-01', 98270157, 'd7fc565022a9cd10d14181b1d73e4dd659b648925ef340e0bf9c593050cb6bb2',
     'd54ec424665686613d20005946aef77be1be670d0ee924452423ad82d873f6a3'),
    ('otto-return-value-v1', 'release-01', 599736656, '421f8976e4b2005f734593ab5d3c70226bc123a91c2e45256b399d0215a38fb8',
     'a12401c668240f5103d104555c7e26b3c582aa6e4bfcff896e2e5846991b2622'),
    ('otto-symmetry-head-v1', 'release-01', 244249285, '34c18bf08d50ed875a72168d498d5010f4947e1d0162370886b7a30248c01158',
     '0daf91bc5d8c6aa4ee20529bd3dda8031cd0406aa156f214093e64efb5ffe1fe'),
)
OUTPUTS = {ARCHIVE, 'manifest.json', 'dependencies.json', 'RESTORE.md', 'SHA256SUMS.txt'}
SCOPE = ('Opaque byte preservation only. No arrays, checkpoints, outcome summaries or scientific models are decoded; '
         'no fitting, scoring, simulator, local model readout or remote call. Successful archival is not scientific success.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    require(path.is_absolute() and path.is_relative_to(ROOT) and '..' not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'contained regular original file')
    return path


def contained(name):
    rel = Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name, 'safe canonical member name')
    return regular(ROOT / rel)


def digest(path, check=lambda: None):
    value, size = hashlib.sha256(), 0
    with regular(path).open('rb') as stream:
        while block := stream.read(1024**2):
            check()
            value.update(block)
            size += len(block)
    return {'sha256': value.hexdigest(), 'bytes': size}


def load_source(name, pin, module_name, check=lambda: None):
    path = contained(name)
    require(digest(path, check)['sha256'] == pin, 'pinned helper before import')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def pack(path, members, check):
    class CheckedReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size):
            check()
            return self.stream.read(size)

    with tarfile.open(path, 'x:gz', compresslevel=6, copybufsize=1024**2) as archive:
        for name, identity in members.items():
            check()
            source = contained(name)
            info = archive.gettarinfo(source, arcname=name)
            require(info.isfile() and info.size == identity['bytes'], 'regular member and recorded byte size')
            with source.open('rb') as stream:
                archive.addfile(info, CheckedReader(stream))
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for entry in archive:
            check()
            require(entry.isfile() and entry.name in members and entry.name not in seen
                    and entry.size == members[entry.name]['bytes'], 'exact regular readback membership')
            value, size = hashlib.sha256(), 0
            with archive.extractfile(entry) as stream:
                while block := stream.read(1024**2):
                    check()
                    value.update(block)
                    size += len(block)
            require({'sha256': value.hexdigest(), 'bytes': size} == members[entry.name], 'lossless member readback')
            seen.add(entry.name)
    require(seen == members.keys(), 'every archive member verified')
    # tar can stop at its end marker before gzip consumes and verifies its trailer.
    with gzip.open(path, 'rb') as stream:
        while stream.read(1024**2):
            check()


class Archive:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.rss, self.members, self.categories = 0, {}, {}

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['seconds']*10**9, 'archive native deadline')
        self.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        require(self.rss <= LIMITS['rss_bytes'], 'archive RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'archive output cap')

    def add(self, path, category, expected=None):
        identity = digest(regular(path), self.check)
        if expected is not None:
            require(identity['sha256'] == expected if isinstance(expected, str) else identity == expected,
                    'pinned archive input: '+str(path))
        name = path.relative_to(ROOT).as_posix()
        require(name not in self.members or self.members[name] == identity, 'identical duplicated membership')
        self.members[name] = identity
        self.categories.setdefault(category, set()).add(name)
        return identity

    def add_closed(self, directory, receipt, category):
        self.add(directory/'receipt.json', category)
        for name, identity in receipt['files'].items():
            require(Path(name).name == name, 'flat closed evidence filenames')
            self.add(directory/name, category, identity)

    def evidence(self, report):
        a = self.args
        require(a.plan == STUDY/'plan-01.json' and a.run == STUDY/'run-01' and a.audit == STUDY/'audit-01'
                and a.report == STUDY/'report-01' and a.terminal == STUDY/'process-01.terminal.json', 'original study paths')
        worker, audit, terminal = report.authenticate(a, self.check)
        publication = report.closed(a.report, a.report_sha256, report.OUTPUTS, self.check)
        request = {k: str(getattr(a, k)) for k in ('plan', 'plan_sha256', 'run', 'receipt_sha256', 'terminal',
                                                'terminal_sha256', 'audit', 'audit_sha256')}
        require(publication['version'] == report.VERSION and publication['source']['sha256'] == REPORT_PIN
                and publication['request'] == {**request, 'output': str(a.report)}
                and publication['model_calls'] == publication['training_calls'] == publication['simulator_calls'] == 0,
                'completed report from the same audited inputs')
        plan = report.read(a.plan)
        require(len(plan['sources']) == 209, 'all209 frozen scientific sources')
        for directory, receipt, category in ((a.run, worker, 'worker'), (a.audit, audit, 'audit'),
                                              (a.report, publication, 'report')):
            self.add_closed(directory, receipt, category)
        self.add(a.plan, 'plan', a.plan_sha256)
        launch = STUDY/'process-01.launch.json'
        require(report.read(a.run/'started.json')['request']['supervision'] == str(launch), 'original launch filename')
        self.add(launch, 'supervisor', worker['supervision_sha256'])
        self.add(a.terminal, 'supervisor', a.terminal_sha256)
        self.add(STUDY/'process-01.log', 'supervisor')
        self.add(a.witness, 'execution_witness', a.witness_sha256)
        witness = report.read(a.witness)
        require(witness['status'] == 'completed' and all(witness[k] == getattr(a, key) for k, key in (
            ('plan_sha256', 'plan_sha256'), ('worker_sha256', 'receipt_sha256'),
            ('terminal_sha256', 'terminal_sha256'), ('audit_sha256', 'audit_sha256'))), 'execution witness pins')
        for key in ('worker_execution', 'audit_execution'):
            event = witness[key]
            require(type(event['session_id']) is int and event['session_id'] > 0 and event['exit_code'] == 0
                    and isinstance(event['terminal_chunk'], str) and event['terminal_chunk'], 'actual original tool completion')
        for name, pin in WITNESSES.items():
            self.add(contained(name), 'dispatch_witnesses', pin)
        for name, pin in plan['sources'].items():
            self.add(contained(name), 'scientific_sources', pin)
        for name, pin in (*LICENSES.items(), (REPORT, REPORT_PIN), DEPENDENCY_SOURCE):
            self.add(contained(name), 'licenses' if name in LICENSES else 'publication_sources', pin)
        self.add(Path(__file__).resolve(), 'publication_sources')
        self.qualification(report, plan)
        for name, pin in ENGINEERING.items():
            path = contained(name)
            self.add(path, 'engineering', pin)
            # Preserve every completed preflight attempt, including initial failures.
            evidence = report.read(path)
            files = evidence.get('files', {})
            require({p.name for p in path.parent.iterdir()} == set(files)|{'receipt.json'}, 'closed engineering inventory')
            for child, identity in files.items():
                require(Path(child).name == child, 'safe engineering payload')
                self.add(path.parent/child, 'engineering', identity)
        dependencies = []
        for tag, folder, size, pin, manifest_pin in DEPENDENCIES:
            path = ROOT/'output'/tag/folder/'manifest.json'
            self.add(path, 'dependency_manifests', manifest_pin)
            dependencies.append({'release': f'https://github.com/kw2828/OpenJev/releases/tag/{tag}',
                'archive_url': f'https://github.com/kw2828/OpenJev/releases/download/{tag}/openjev-{tag}.tar.gz',
                'archive': {'bytes': size, 'sha256': pin}, 'manifest_path': path.relative_to(ROOT).as_posix(),
                'manifest_sha256': manifest_pin})
        for role, identity in plan['inputs'].items():
            if role.startswith('capacity_'):
                self.add(contained(identity['path']), 'dependency_descriptors', {k: identity[k] for k in ('bytes', 'sha256')})
        require(all(len(self.categories[k]) == n for k, n in (
            ('worker', 57), ('audit', 4), ('report', 4), ('scientific_sources', 209),
            ('supervisor', 3), ('qualification', 25))), 'exact primary artifact coverage')
        self.members = dict(sorted(self.members.items()))
        return dependencies, plan['inputs'], terminal['wall_seconds']

    def qualification(self, report, plan):
        q = {role: contained(item['path']) for role, item in plan['qualification'].items()}
        names = {'started.json', 'runtime.json', 'fixtures.npz', 'calls.jsonl', 'operations.jsonl',
                 'synthetic-losses.jsonl', 'summary.json'}
        names.update(f'{prefix}-{kind}.npz' for prefix in ('disposable', 'parity') for kind in report.KINDS)
        worker = report.closed(q['receipt'].parent, plan['qualification']['receipt']['sha256'], names, self.check)
        audit = report.closed(q['audit'].parent, plan['qualification']['audit']['sha256'], {'started.json', 'summary.json'}, self.check)
        qp = report.read(q['plan'])
        require(worker['version'] == qp['version'] == 'otto-spatial-qualification-v1' and worker['errors'] == []
                and audit['agreement'] is True and len(qp['sources']) == 9, 'completed synthetic qualification evidence')
        for directory, receipt in ((q['receipt'].parent, worker), (q['audit'].parent, audit)):
            self.add_closed(directory, receipt, 'qualification')
        for role in ('plan', 'terminal'):
            self.add(q[role], 'qualification', {k: plan['qualification'][role][k] for k in ('bytes', 'sha256')})
        launch = q['plan'].parent/'process-01.launch.json'
        original = report.read(q['receipt'].parent/'started.json')
        terminal = report.read(q['terminal'])
        require(original['request']['supervision'] == str(launch) and report.read(launch) == original['launch']
                and all(terminal[k] == v for k, v in original['launch'].items())
                and terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['group_absent'] is True,
                'closed original qualification supervision')
        self.add(launch, 'qualification', audit['inputs'][str(launch)])
        self.add(q['plan'].parent/'process-01.log', 'qualification')
        for name, pin in qp['sources'].items():
            self.add(contained(name), 'qualification_sources', pin)


def restore_text(plan_pin, members):
    return f'''# Restore the spatial-study evidence

This archive preserves {members} unique original relative paths: all 56 current
worker payloads and receipt, all 15 initial and final checkpoints and prediction
archives, independent audit, plots and derivation, plan and original supervisor
records, engineering attempts, synthetic qualification and its supervision,
209 frozen scientific source files, publication helpers and applicable licenses.
Archive verification proves byte preservation, not scientific superiority.

1. Download `{ARCHIVE}`, `manifest.json`, `dependencies.json`, `RESTORE.md`
   and `SHA256SUMS.txt` from release `otto-spatial-study-v1`.
2. Verify `shasum -a 256 -c SHA256SUMS.txt`. Inspect the archive inventory;
   extract only into a fresh checkout or empty restoration directory. Never
   overwrite an existing scientific run or historical evidence directory.
3. Paths are relative to the repository root. The source freeze is
   `{FREEZE}` and original plan SHA256 is `{plan_pin}`.
   A fresh checkout at that commit already contains some source files; verify
   those against the manifest, or first extract into a separate empty directory
   and copy only absent, verified evidence. All members are regular files.
4. Verify every restored path against `manifest.json` member byte counts and
   SHA256 values before reuse. The publisher streams all members back, verifies
   the gzip trailer, then rehashes original inputs and sources after packaging.

The archive contains the complete NEW study evidence, not a recursively bundled
runtime or all historical datasets. `dependencies.json` identifies the original
capacity, scalar-return and teacher-data releases, with their published archive
hashes/sizes and included manifests. Those releases may name further lineage
dependencies. No historical dataset was fetched or duplicated by this packager.
The original TRAIN/VALID descriptors remain in the current plan and dependency
list. Qualification checkpoints are disposable synthetic fixtures, not trained
scientific heads; they retain their original separate paths and identities.

Portable byte restoration does not make the strict numerical auditor relocatable.
Original absolute repository, input, output and interpreter paths are preserved
in receipts. Replaying the unchanged audit requires those recorded paths, earlier
evidence and exact runtime. A new layout needs a separately reviewed relocation
adapter; do not rewrite frozen plans or receipts to impersonate the old layout.
Byte verification needs no numerical framework, model, simulator or prior data.

The plotting and archival sources are publication helpers outside the 209-file
scientific closure. Their inclusion does not revise the protocol, results or any
historical failed gate. No autonomous competence or novel architecture claim is
established by this scalar study or by successful packaging.
'''


def execute(args):
    a = Archive(args)
    require(args.output == STUDY/'archive-01' and not args.output.exists()
            and '..' not in args.output.parts and not any(p.is_symlink() for p in args.output.parents), 'exclusive contained archive output')
    args.output.mkdir(parents=True, exist_ok=False)
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        a.clock = load_source(CLOCK, CLOCK_PIN, '_spatial_archive_clock').SuspendClock()
        a.start = a.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('archive emergency cap')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        report = load_source(REPORT, REPORT_PIN, '_spatial_archive_report', a.check)
        source = digest(Path(__file__).resolve(), a.check)
        dependencies, inputs, parent_seconds = a.evidence(report)
        require(a.members[Path(__file__).resolve().relative_to(ROOT).as_posix()] == source,
                'unchanged packager source during authentication')
        target = args.output/ARCHIVE
        pack(target, a.members, a.check)
        # Completion joins are metadata-only; never call the reporter's numeric values/render functions.
        report.authenticate(args, a.check)
        report.closed(args.report, args.report_sha256, report.OUTPUTS, a.check)
        for name, identity in a.members.items():
            require(digest(contained(name), a.check) == identity, 'unchanged original after archival')
        write(args.output/'dependencies.json', {'version': VERSION, 'historical_archives': dependencies,
              'original_input_descriptors': inputs, 'downloaded_by_packager': False,
              'historical_datasets_copied': False, 'archive_identity_source': DEPENDENCY_SOURCE,
              'scope': 'Direct dependencies; their original manifests describe further lineage. No new remote verification.'})
        with (args.output/'RESTORE.md').open('x') as stream:
            stream.write(restore_text(args.plan_sha256, len(a.members)))
        manifest = {'version': VERSION, 'status': 'verified', 'scope': SCOPE, 'source': source,
            'source_freeze': FREEZE, 'request': {k: str(v) for k, v in vars(args).items()},
            'archive': {'name': ARCHIVE, **digest(target, a.check)}, 'member_count': len(a.members),
            'members': a.members, 'categories': {k: sorted(v) for k, v in a.categories.items()},
            'category_counts': {k: len(v) for k, v in a.categories.items()},
            'category_scope': 'Categories can overlap; archive member paths are unique.',
            'scientific_sources': 209, 'original_worker_parent_seconds': parent_seconds,
            'limits': LIMITS, 'model_calls': 0, 'simulator_calls': 0, 'training_calls': 0}
        write(args.output/'manifest.json', manifest)
        with (args.output/'SHA256SUMS.txt').open('x') as stream:
            for name in sorted(OUTPUTS-{'SHA256SUMS.txt'}):
                stream.write(digest(args.output/name, a.check)['sha256']+'  '+name+'\n')
        require({p.name for p in args.output.iterdir()} == OUTPUTS, 'exact archive output payload closure')
        files = {name: digest(args.output/name, a.check) for name in sorted(OUTPUTS)}
        finish = a.clock.now_ns()
        a.check()
        receipt = {'version': VERSION, 'status': 'completed', 'verified_members': len(a.members), 'source': source,
            'request': manifest['request'], 'files': files, 'scope': SCOPE, 'limits': LIMITS,
            'clock_backend': a.clock.backend, 'started_ns': a.start, 'finished_ns': finish,
            'wall_seconds': (finish-a.start)/1e9, 'peak_rss_bytes': a.rss,
            'model_calls': 0, 'simulator_calls': 0, 'training_calls': 0}
        write(args.output/'receipt.json', receipt)
        a.check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.output/'receipt.json').exists():
                (args.output/'receipt.json').rename(args.output/'invalid-completed-receipt.json')
            if (args.output/'manifest.json').exists():
                (args.output/'manifest.json').rename(args.output/'invalid-manifest.json')
            write(args.output/'failed.json', {'version': VERSION, 'status': 'failed', 'error': repr(error),
                'request': {k: str(v) for k, v in vars(args).items()}, 'scope': SCOPE, 'started_ns': a.start,
                'wall_seconds': None, 'originals_unchanged_by_packager': True})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure and partial archive.
            error.add_note(f'Failure evidence publication: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'audit', 'report', 'witness', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-sha256', 'report-sha256', 'witness-sha256'):
        parser.add_argument('--'+flag, required=True)
    print(json.dumps(execute(parser.parse_args())), flush=True)
