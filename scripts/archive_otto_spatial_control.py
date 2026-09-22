"""Archive closed, independently audited spatial-control evidence as bounded parts.

No archive construction precedes completed worker, original supervisor, audit and
report authentication. Checkpoints and arrays are opaque byte streams throughout.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import resource
import shutil
import signal
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'output/otto-spatial-control-v1'
VERSION = 'otto-spatial-control-archive-v1'
PREFIX = 'openjev-otto-spatial-control-v1.tar.gz.part'
FREEZE = 'e8e97cd1e6bfa58261a8ced0845b597c1c2c0441'
PART_BYTES = 1024**3
LIMITS = {'seconds': 1800, 'rss_bytes': 4*1024**3, 'output_bytes': 16*1024**3}
BASE = 'scripts/archive_otto_spatial_study.py'
BASE_PIN = '1a347c8fdc7e4375132626f041b378c9f8d615f5c344f70f1448c773c7d51cf3'
REPORT = 'scripts/report_otto_spatial_control.py'
REPORT_PIN = '66c3f945f94d46b4e2cbb5c2074bfca2ce57e17952aeb8a8210bb8af09322631'
REPORT_TEST = 'tests/test_report_otto_spatial_control.py'
REPORT_TEST_PIN = '4c51bf6ee58a54c224cbed1d9b69436f9b1135bc7d76f7cca59d8f64bcd8c942'
PACKER = 'scripts/archive_otto_spatial_control.py'
PACKER_TEST = 'tests/test_archive_otto_spatial_control.py'
PUBLICATION_SOURCES = (PACKER, PACKER_TEST, REPORT, REPORT_TEST)
ENGINEERING_VERSION = 'otto-spatial-control-publication-engineering-v1'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
AUDITOR = 'scripts/audit_otto_spatial_control.py'
AUDITOR_PIN = '85fc1d059407a3e95a9365cb7cfc254ff60b8230dccc7629c427b9615b108baa'
PRODUCER = 'scripts/study_otto_spatial_control.py'
PRODUCER_PIN = 'cf87432697852209d323a3f809758fef5382d229a85092ca8f49380d9d275797'
ENGINEERING = {
    'engineering-01': '7315ef685687cf100c0843eb6d7688bd20d200b156e89d3a12bb3b55c31812ae',
    'engineering-02': '4b8721a54c4c7a1d1faf0389ae4e66daaddb32d0b76f711e89ec3794de932d0b',
    'seed-engineering-01': '54ea64ea88b648cdb0d35b79ba874e7893e68aa9a8f3f2b75823af4e67fa86fc',
    'seed-engineering-02': '7b1ee7bf120a7957ae4f1009fb658b161cb3bcb447021dc84cca5cf8b44d5df8',
}
WITNESSES = {
    'prefreeze-review-01.json': 'ca2c0bc19d9f9d2a22ca47c2ee82e97fe27acf3e463f688375fd7b0eceb95a03',
    'autonomous-admission-01.json': '5b30fe500ac2c0dfcfb178d9f9813b4e5a9658ca03c464e0ded1b26a7e92e93a',
    'launch-witness-01.json': '0887f1e4a26de4ca9b604555c36085473480eb8255eedb4c0bd7f7ed128b3f07',
    'qualification-execution-witness-01.json': 'd7c49f5ac2d83849c883b460c0f6423a686b3e12649b47ab361e588db09974fd',
}
SPATIAL_DEPENDENCY = (
    'otto-spatial-study-v1', 'archive-01', 61800541,
    '56ba6868ff0359ec6a27d8631085776602b63227c14fd63a0fddff0f72b51662',
    '6ab20033d406ce90b646819a9316824916714e6c2be790dc1c98f5f9d173e3ad',
)
SCOPE = ('Opaque byte preservation only; no NPZ decoding, model readout, training, simulator or remote calls. '
         'Archival completion is not scientific success. Historical TRAIN/VALID caches remain separate dependencies.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name, pin, module_name):
    path = ROOT/name
    require(not any(p.is_symlink() for p in (path, *path.parents))
            and hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'helper hash before import')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# These inherited utilities are stdlib-only byte/path functions, never numerical helpers.
B = load(BASE, BASE_PIN, '_spatial_control_archive_bytes')
regular, contained, digest, write = B.regular, B.contained, B.digest, B.write


class PartWriter:
    """Split one gzip stream at fixed byte offsets without a second full copy."""
    def __init__(self, directory, check, part_bytes=PART_BYTES):
        require(type(part_bytes) is int and part_bytes > 0, 'positive segment length')
        self.directory, self.check, self.part_bytes = directory, check, part_bytes
        self.stream, self.parts, self.current = None, [], 0
        self.bytes, self.hasher = 0, hashlib.sha256()

    def write(self, data):
        self.check()
        view = memoryview(data)
        total = len(view)
        while view:
            if self.stream is None:
                name = f'{PREFIX}{len(self.parts)+1:04d}'
                self.stream = (self.directory/name).open('xb')
                self.parts.append(name)
                self.current = 0
            take = min(len(view), self.part_bytes-self.current)
            block = view[:take]
            require(self.stream.write(block) == take, 'complete segment write')
            self.hasher.update(block)
            self.current += take
            self.bytes += take
            view = view[take:]
            self.check()
            if self.current == self.part_bytes:
                self.finish()
        return total

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def finish(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None


class PartReader(io.RawIOBase):
    def __init__(self, paths, check):
        self.paths, self.check, self.index, self.stream = paths, check, 0, None

    def readable(self):
        return True

    def read(self, size=-1):
        require(size >= 0, 'bounded segmented read')
        chunks = []
        while size:
            self.check()
            if self.stream is None:
                if self.index == len(self.paths):
                    break
                self.stream = self.paths[self.index].open('rb')
                self.index += 1
            block = self.stream.read(size)
            if block:
                chunks.append(block)
                size -= len(block)
            else:
                self.stream.close()
                self.stream = None
        return b''.join(chunks)

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def pack(directory, members, check, *, part_bytes=PART_BYTES):
    writer = PartWriter(directory, check, part_bytes)
    try:
        with (
            gzip.GzipFile(filename='', mode='wb', fileobj=writer, compresslevel=1, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT, copybufsize=1024**2) as archive,
        ):
            for name, identity in members.items():
                check()
                source = contained(name)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = identity['bytes'], 0o644, 0
                require(source.stat().st_size == info.size, 'unchanged original size before packing')
                with source.open('rb') as stream:
                    archive.addfile(info, stream)
    finally:
        writer.finish()
    paths = [directory/name for name in writer.parts]
    verify_members(paths, members, check)
    parts = [{'name': p.name, **digest(p, check)} for p in paths]
    require(all(p['bytes'] == part_bytes for p in parts[:-1]) and 0 < parts[-1]['bytes'] <= part_bytes,
            'fixed segmentation with one bounded final part')
    joined_hash = hashlib.sha256()
    for path in paths:
        with path.open('rb') as stream:
            while block := stream.read(1024**2):
                check()
                joined_hash.update(block)
    require(joined_hash.hexdigest() == writer.hasher.hexdigest()
            and sum(p['bytes'] for p in parts) == writer.bytes, 'complete compressed stream readback identity')
    return {'format': 'single gzip tar stream split into ordered byte parts', 'parts': parts,
            'bytes': writer.bytes, 'sha256': joined_hash.hexdigest(), 'part_bytes': part_bytes,
            'compression_level': 1, 'gzip_mtime': 0, 'normalized_tar_metadata': True}


def verify_members(paths, members, check):
    """Independently stream the exact member bytes and full gzip trailer back."""
    seen = set()
    with (PartReader(paths, check) as joined, gzip.GzipFile(fileobj=joined, mode='rb') as decoded,
          tarfile.open(fileobj=decoded, mode='r|') as archive):
        for entry in archive:
            check()
            require(entry.isfile() and entry.name in members and entry.name not in seen
                    and entry.size == members[entry.name]['bytes'], 'exact regular readback membership')
            hasher, size = hashlib.sha256(), 0
            with archive.extractfile(entry) as stream:
                while block := stream.read(1024**2):
                    check()
                    hasher.update(block)
                    size += len(block)
            require({'sha256': hasher.hexdigest(), 'bytes': size} == members[entry.name], 'lossless member readback')
            seen.add(entry.name)
    require(seen == members.keys(), 'complete segmented member coverage')
    # Read to gzip EOF independently of tar's earlier end marker, validating its trailer.
    with PartReader(paths, check) as joined, gzip.GzipFile(fileobj=joined, mode='rb') as decoded:
        while decoded.read(1024**2):
            check()


class Archive(B.Archive):
    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['seconds']*10**9, 'archive native deadline')
        self.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        require(self.rss <= LIMITS['rss_bytes'], 'archive RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'archive disk cap')

    def closure(self, directory, pin, names, category, *, completed=True):
        self.add(directory/'receipt.json', category, pin)
        receipt = json.loads((directory/'receipt.json').read_text())
        require(not completed or receipt['status'] == 'completed', 'completed evidence before archival')
        require(set(receipt['files']) == names and {p.name for p in directory.iterdir()} == names|{'receipt.json'},
                'exact closed directory inventory')
        for name, identity in receipt['files'].items():
            require(Path(name).name == name, 'flat closed filenames')
            self.add(directory/name, category, identity)
        return receipt

    def publication_engineering(self):
        """Preserve every direct attempt; only final-byte passing evidence admits packaging."""
        current = {name: digest(contained(name), self.check)['sha256'] for name in PUBLICATION_SOURCES}
        directories = sorted(p for p in STUDY.iterdir() if p.name.startswith('publication-engineering-'))
        require(directories, 'publication engineering evidence required')
        admitted = []
        for directory in directories:
            require(re.fullmatch(r'publication-engineering-[0-9]{2}', directory.name)
                    and directory.is_dir() and not directory.is_symlink(), 'canonical direct engineering attempt')
            path = regular(directory/'receipt.json')
            pin = digest(path, self.check)['sha256']
            receipt = json.loads(path.read_text())
            require(receipt['version'] == ENGINEERING_VERSION and receipt['status'] in {'passed', 'failed'}
                    and isinstance(receipt['scope'], str) and receipt['scope']
                    and all(type(receipt[k]) is int and receipt[k] == 0 for k in
                            ('model_calls', 'training_calls', 'simulator_calls')), 'synthetic-only engineering scope')
            results = receipt['results']
            require(isinstance(results, list) and results
                    and all(isinstance(r, dict) and isinstance(r.get('name'), str) and r['name']
                            and type(r.get('exit_code')) is int for r in results), 'nonempty explicit command results')
            sources = receipt['sources']
            require(isinstance(sources, dict) and set(PUBLICATION_SOURCES) <= set(sources)
                    and all(isinstance(pin, str) and re.fullmatch(r'[0-9a-f]{64}', pin)
                            for pin in sources.values()), 'all four publication source identities')
            self.closure(directory, pin, set(receipt['files']), 'publication_engineering', completed=False)
            if receipt['status'] == 'passed':
                require(all(r['exit_code'] == 0 for r in results), 'passed attempt has only successful commands')
                require(receipt['sources_before'] == receipt['sources_after'] == sources,
                        'passed attempt preserves source hashes before and after commands')
                if all(sources[name] == sha for name, sha in current.items()):
                    admitted.append(directory.name)
        require(admitted, 'one passed publication attempt must bind every current publication source')
        return {'attempts': [p.name for p in directories], 'current_sources': current,
                'admitting_attempts': admitted, 'scope': 'Failed and earlier-byte attempts preserved; only current-byte passed attempts establish readiness.'}

    def supervision(self, directory, plan_path, terminal_path, worker, category):
        started = json.loads((directory/'started.json').read_text())
        request, launch = started['request'], started['launch']
        path = Path(request['supervision'])
        self.add(path, category, worker['supervision_sha256'])
        terminal = json.loads(terminal_path.read_text())
        require(json.loads(path.read_text()) == launch and all(terminal[k] == v for k, v in launch.items())
                and request['plan'] == str(plan_path) and request['output'] == str(directory)
                and terminal['status'] == 'completed' and terminal['returncode'] == 0
                and terminal['timed_out'] is False and terminal['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == [],
                'completed original matching supervisor')
        require(path.name.endswith('.launch.json'), 'original supervisor filename')
        self.add(path.with_name(path.name.removesuffix('.launch.json')+'.log'), category)

    def qualification(self, plan, report):
        q = plan['qualification']
        require(set(q) == {'plan', 'receipt', 'terminal', 'audit'}, 'all qualification descriptors')
        for item in q.values():
            self.add(contained(item['path']), 'qualification', {k: item[k] for k in ('bytes', 'sha256')})
        paths = {role: contained(item['path']) for role, item in q.items()}
        qp = report.read(paths['plan'])
        names = {'started.json', 'runtime.json', 'inference-setup.json', 'work-contexts.jsonl', 'work.jsonl',
                 'summary.json', 'serialization-projection.json', 'preparation.json', 'qualification-arrays.npz',
                 'qualification-tuples.jsonl', 'parity.jsonl'}
        worker = self.closure(paths['receipt'].parent, q['receipt']['sha256'], names, 'qualification')
        audit = self.closure(paths['audit'].parent, q['audit']['sha256'], {'started.json', 'readouts.jsonl', 'summary.json'}, 'qualification')
        require(qp['mode'] == worker['mode'] == audit['mode'] == 'qualify'
                and qp['sources'] == plan['sources'] == worker['sources'] and worker['inputs'] == qp['inputs']
                and worker['version'] == qp['version'] == 'otto-spatial-control-v1'
                and worker['plan_sha256'] == q['plan']['sha256'] and worker['pending'] == []
                and worker['parity_passed'] is True and worker['completed_episodes'] == 0
                and all(qp['inputs'][k] == v for k, v in plan['inputs'].items()),
                'same completed all-head deployment qualification')
        require(audit['agreement'] is True and audit['pending'] is None and audit['source']['sha256'] == AUDITOR_PIN
                and audit['worker_sha256'] == q['receipt']['sha256'] and audit['plan_sha256'] == q['plan']['sha256']
                and audit['terminal_sha256'] == q['terminal']['sha256'] and audit['saved_checkpoint_readout_calls'] == 780,
                'closed independent qualification audit')
        require({k: v['returned'] for k, v in worker['calls'].items()} == {
                    'model_load': 15, 'parity_restore': 15, 'parity_numpy_forward': 780, 'parity_torch_forward': 780}
                and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete qualification work')
        self.supervision(paths['receipt'].parent, paths['plan'], paths['terminal'], worker, 'qualification')
        for role, item in qp['inputs'].items():
            if role not in {'train_data', 'valid_data'}:
                self.add(contained(item['path']), 'qualification_inputs', {k: item[k] for k in ('bytes', 'sha256')})
        return {role: qp['inputs'][role] for role in ('train_data', 'valid_data')}

    def evidence(self, report):
        a = self.args
        require(a.plan == STUDY/'plan-01.json' and a.run == STUDY/'run-01' and a.audit == STUDY/'audit-01'
                and a.report == STUDY/'report-01' and a.terminal == STUDY/'process-01.terminal.json', 'original study paths')
        worker, audit, terminal = report.authenticate(a, self.check)
        plan = report.read(a.plan)
        require(len(plan['sources']) == 219 and plan['sources'][PRODUCER] == PRODUCER_PIN
                and plan['sources'][AUDITOR] == AUDITOR_PIN and worker['completed_episodes'] == 1152
                and worker['mode'] == audit['mode'] == 'study', 'full original study only')
        for directory, pin, names, category in (
            (a.run, a.receipt_sha256, report.worker_payloads(), 'worker'),
            (a.audit, a.audit_sha256, {'started.json', 'readouts.jsonl', 'summary.json'}, 'audit'),
            (a.report, a.report_sha256, report.OUTPUTS, 'report'),
        ):
            value = self.closure(directory, pin, names, category)
            if category == 'report':
                require(value['version'] == report.VERSION and value['source']['sha256'] == REPORT_PIN
                        and value['request'] == {k: str(getattr(a, k)) for k in (
                            'plan', 'plan_sha256', 'run', 'receipt_sha256', 'terminal', 'terminal_sha256',
                            'audit', 'audit_sha256')} | {'output': str(a.report)}
                        and value['model_calls'] == value['training_calls'] == value['simulator_calls'] == 0,
                        'same authenticated report, with no model calls')
        for path, pin, category in ((a.plan, a.plan_sha256, 'plan'), (a.terminal, a.terminal_sha256, 'supervisor'),
                                    (a.witness, a.witness_sha256, 'execution_witness')):
            self.add(path, category, pin)
        self.supervision(a.run, a.plan, a.terminal, worker, 'supervisor')
        witness = report.read(a.witness)
        require(witness['status'] == 'completed' and all(witness[k] == getattr(a, key) for k, key in (
            ('plan_sha256', 'plan_sha256'), ('worker_sha256', 'receipt_sha256'),
            ('terminal_sha256', 'terminal_sha256'), ('audit_sha256', 'audit_sha256'))), 'original tool witness joins')
        for key in ('worker_execution', 'audit_execution'):
            event = witness[key]
            require(type(event['session_id']) is int and event['session_id'] > 0 and event['exit_code'] == 0
                    and isinstance(event['terminal_chunk'], str) and event['terminal_chunk'], 'actual successful original tool invocation')
        for name, pin in plan['sources'].items():
            self.add(contained(name), 'scientific_sources', pin)
        for role, item in plan['inputs'].items():
            self.add(contained(item['path']), 'final_heads' if role.startswith('head_') else 'direct_inputs',
                     {k: item[k] for k in ('bytes', 'sha256')})
        prior = report.read(contained(plan['inputs']['spatial_receipt']['path']))
        prior_plan = report.read(contained(plan['inputs']['spatial_plan']['path']))
        prior_audit = report.read(contained(plan['inputs']['spatial_audit']['path']))
        require(prior['status'] == 'completed' and prior['completed_fits'] == 15 and prior['pending'] == []
                and prior['sources'] == prior_plan['sources'] and all(plan['sources'][k] == v for k, v in prior_plan['sources'].items())
                and prior['plan_sha256'] == plan['inputs']['spatial_plan']['sha256']
                and prior_audit['status'] == 'completed' and prior_audit['agreement'] is True
                and prior_audit['worker_sha256'] == plan['inputs']['spatial_receipt']['sha256']
                and prior_audit['plan_sha256'] == plan['inputs']['spatial_plan']['sha256']
                and prior_audit['terminal_sha256'] == plan['inputs']['spatial_terminal']['sha256'], 'completed original fitted-head lineage')
        for role, item in plan['inputs'].items():
            if role.startswith('head_'):
                require({k: item[k] for k in ('bytes', 'sha256')} == prior['files'][Path(item['path']).name], 'original final checkpoint identity')
        cache_dependencies = self.qualification(plan, report)
        for name, pin in WITNESSES.items():
            self.add(STUDY/name, 'prior_witnesses', pin)
        for name, pin in ENGINEERING.items():
            value = report.read(STUDY/name/'receipt.json')
            self.closure(STUDY/name, pin, set(value['files']), 'engineering', completed=False)
        for name, pin in B.LICENSES.items():
            self.add(contained(name), 'licenses', pin)
        for name, pin in ((BASE, BASE_PIN), (REPORT, REPORT_PIN), (REPORT_TEST, REPORT_TEST_PIN),
                          (report.HELPER, report.HELPER_PIN)):
            self.add(contained(name), 'publication_sources', pin)
        self.add(Path(__file__).resolve(), 'publication_sources')
        self.add(contained(PACKER_TEST), 'publication_sources')
        publication_engineering = self.publication_engineering()
        dependencies = []
        for tag, folder, size, pin, manifest_pin in (SPATIAL_DEPENDENCY, *B.DEPENDENCIES):
            path = ROOT/'output'/tag/folder/'manifest.json'
            self.add(path, 'dependency_manifests', manifest_pin)
            dependencies.append({'tag': tag, 'archive_name': f'openjev-{tag}.tar.gz',
                'archive_url': f'https://github.com/kw2828/OpenJev/releases/download/{tag}/openjev-{tag}.tar.gz',
                'manifest_url': f'https://github.com/kw2828/OpenJev/releases/download/{tag}/manifest.json',
                'archive': {'bytes': size, 'sha256': pin}, 'manifest_sha256': manifest_pin,
                'included_manifest': path.relative_to(ROOT).as_posix()})
        require(all(len(self.categories[k]) == n for k, n in (
            ('worker', 12), ('audit', 4), ('report', 6), ('scientific_sources', 219), ('final_heads', 15),
            ('qualification', 20), ('supervisor', 3))), 'exact primary archive coverage')
        self.members = dict(sorted(self.members.items()))
        return {'historical_archives': dependencies, 'excluded_original_cache_descriptors': cache_dependencies,
                'publication_engineering': publication_engineering,
                'scope': 'Current evidence and all15 final heads included; prior fitting/data releases are separately named dependencies. '
                         'Those historical manifests can name additional lineage. No dependency archive was downloaded or byte-verified here.'}, terminal['wall_seconds']


def restore_text(plan_pin, members, parts):
    return f'''# Restore spatial-control evidence

The {len(parts)} numbered parts form ONE gzip-compressed tar stream, not separate
archives. They preserve {members} original relative paths and bytes: all current
worker, independent audit, report, qualification and qualification-audit payloads;
15 final heads, three known kernels, 219 frozen sources, source licenses, original
supervisors and execution witnesses, plus engineering attempts including failures.
Archive completion does not imply scientific success or architecture advantage.
Every direct publication-engineering-NN attempt is retained. At least one passed
attempt must bind the current reporter, packer and both test files; historical
attempts with older hashes cannot establish current publication readiness.

Download every numbered part, manifest.json, dependencies.json, RESTORE.md and
SHA256SUMS.txt. Run `shasum -a 256 -c SHA256SUMS.txt`, then concatenate parts in
the manifest's exact order. For these fixed four-digit names, shell lexical order
is identical: `cat {PREFIX}???? > openjev-otto-spatial-control-v1.tar.gz`.
Check that combined file's SHA256 and size against manifest.archive before use.
Inspect the tar inventory, extract into an EMPTY directory, then verify every
restored member using manifest.members. Never overwrite existing run evidence.
The publisher streams all members back, validates the gzip trailer, and rechecks
all original files and sources after compression. Part boundaries are fixed at
1 GiB, below GitHub's per-asset limit; no member is dropped to meet asset sizing.

Source freeze: `{FREEZE}`. Frozen plan SHA256: `{plan_pin}`.
The original TRAIN/VALID caches are not duplicated. dependencies.json identifies
their exact descriptors and the scalar-return release that contains them, plus
the original spatial-fit, capacity and teacher releases with archive hashes,
byte sizes and included manifests. Consult their manifests for recursive lineage;
this is not a standalone runtime or a recursive bundle of all historical data.
Qualification's already-saved 52 shared branch tuples ARE included.

Byte inspection and restoration are portable. Existing strict numerical auditors
also require recorded absolute repository/input/output/interpreter paths and exact
runtime, or a separately reviewed relocation adapter. No relocation adapter is
provided here. Do not rewrite historical plans or receipts to mimic another path.
Packaging performs no model, optimizer, simulator, API call or numerical replay.
Scientific outcomes and all failures remain those in the completed audit/report.
'''


def execute(args):
    archive = Archive(args)
    require(args.output == STUDY/'archive-01' and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), 'exclusive original archive directory')
    args.output.mkdir(parents=True, exist_ok=False)
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        archive.clock = load(CLOCK, CLOCK_PIN, '_spatial_control_archive_clock').SuspendClock()
        archive.start = archive.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('archive emergency cap')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        source = digest(Path(__file__).resolve(), archive.check)
        report = load(REPORT, REPORT_PIN, '_spatial_control_archive_report')
        dependencies, parent_seconds = archive.evidence(report)
        require(archive.members[Path(__file__).resolve().relative_to(ROOT).as_posix()] == source, 'unchanged packager source')
        raw = sum(v['bytes'] for v in archive.members.values())
        reserve = raw+raw//500+4096*len(archive.members)+32*1024**2
        require(reserve <= LIMITS['output_bytes'] and shutil.disk_usage(args.output).free >= reserve,
                'conservative incompressible input-plus-tar storage fits both allocation and available disk')
        stream = pack(args.output, archive.members, archive.check)
        # Recheck successful identities and every original after writing. No reporting or scoring calls.
        report.authenticate(args, archive.check)
        for name, identity in archive.members.items():
            require(digest(contained(name), archive.check) == identity, 'unchanged original after archival')
        write(args.output/'dependencies.json', {'version': VERSION, **dependencies})
        (args.output/'RESTORE.md').write_text(restore_text(args.plan_sha256, len(archive.members), stream['parts']))
        request = {k: str(v) for k, v in vars(args).items()}
        manifest = {'version': VERSION, 'status': 'verified', 'scope': SCOPE, 'source': source,
            'source_freeze': FREEZE, 'request': request, 'archive': stream, 'members': archive.members,
            'member_count': len(archive.members), 'original_bytes': raw,
            'categories': {k: sorted(v) for k, v in archive.categories.items()},
            'category_counts': {k: len(v) for k, v in archive.categories.items()},
            'category_scope': 'Membership categories overlap; tar paths are unique.',
            'original_worker_parent_seconds': parent_seconds, 'limits': LIMITS,
            'github_asset_rule_source': 'https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas',
            'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0}
        write(args.output/'manifest.json', manifest)
        names = {p['name'] for p in stream['parts']}|{'manifest.json', 'dependencies.json', 'RESTORE.md'}
        with (args.output/'SHA256SUMS.txt').open('x') as file:
            for name in sorted(names):
                file.write(digest(args.output/name, archive.check)['sha256']+'  '+name+'\n')
        names.add('SHA256SUMS.txt')
        require({p.name for p in args.output.iterdir()} == names, 'exact archive payload closure')
        files = {name: digest(args.output/name, archive.check) for name in sorted(names)}
        archive.check()
        finish = archive.clock.now_ns()
        receipt = {'version': VERSION, 'status': 'completed', 'source': source, 'request': request,
            'files': files, 'verified_members': len(archive.members), 'scope': SCOPE, 'limits': LIMITS,
            'clock_backend': archive.clock.backend, 'started_ns': archive.start, 'finished_ns': finish,
            'wall_seconds': (finish-archive.start)/1e9, 'peak_rss_bytes': archive.rss,
            'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0}
        write(args.output/'receipt.json', receipt)
        archive.check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            for name in ('receipt.json', 'manifest.json'):
                if (args.output/name).exists():
                    (args.output/name).rename(args.output/('invalid-'+name))
            failure = {'version': VERSION, 'status': 'failed', 'error': repr(error), 'scope': SCOPE,
                       'started_ns': archive.start, 'wall_seconds': None,
                       'request': {k: str(v) for k, v in vars(args).items()}, 'originals_modified': False}
            write(args.output/'failed.json', failure)
            write(args.output/'receipt.json', failure)
        except BaseException as secondary:  # noqa: BLE001 - Preserve primary failure and partial parts.
            error.add_note(f'Failure publication: {secondary!r}')
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
