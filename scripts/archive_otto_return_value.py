"""Archive the completed audited scalar-return study, retaining original bytes.

Publication-only streaming hashes/compression. No model, posterior, simulator,
training or scientific metric computation; prior frozen sources are unchanged.
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
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/otto-return-value-v1'
VERSION = 'otto-return-value-raw-archive-v1'
PLAN_PIN = '92df71d5e20ca48d8485bfa0a5f50bdf0e6e7a276c8e0c097bccd8e1c095aee2'
FREEZE_COMMIT = '8879b92ed9ce7586b4fae3bc8bf0eb07bb9a82dd'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
RUNNER = 'scripts/study_otto_return_value.py'
AUDITOR = 'scripts/audit_otto_return_value.py'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
LIMITS = {'seconds': 600, 'rss_bytes': 4 * 1024**3, 'output_bytes': 8 * 1024**3}
FAMILIES, SEEDS = ('min8', 'mlp8', 'homogeneous8'), (10101, 10102, 10103)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    require(path.is_absolute() and path.is_relative_to(ROOT) and '..' not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'regular contained input')
    return path


def descriptor(path, check=lambda: None):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            check()
            value.update(chunk)
            size += len(chunk)
    return {'sha256': value.hexdigest(), 'bytes': size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def payload_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'work-contexts.jsonl', 'work.jsonl', 'preparation.json', 'fits.jsonl', 'fit-curves.jsonl',
             'epoch-orders.jsonl', 'parity.jsonl', 'parity.json', 'inference-setup.json',
             'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json', 'training-costs.json'}
    names.update(f'kernel-lambda{r}.npz' for r in (3, 4, 5))
    names.update(f'{split}-{suffix}' for split in ('train', 'valid') for suffix in ('data.npz', 'rows.jsonl'))
    names.update(f'{prefix}-{family}-{seed}.npz' for prefix in ('final', 'predictions') for family in FAMILIES for seed in SEEDS)
    return names


class Archive:
    def __init__(self, args):
        self.args = args
        self.run, self.audit, self.out = STUDY / 'run-01', STUDY / 'audit-01', STUDY / 'release-01'
        self.clock = self.start = None
        self.peak_rss = 0

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'archive native deadline')
        self.peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(self.peak_rss <= LIMITS['rss_bytes'], 'archive RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'archive output cap')

    def closed(self, directory, pin, expected):
        require(directory.is_dir() and not any(p.is_symlink() for p in (directory, *directory.parents)), 'regular evidence directory')
        path = regular(directory / 'receipt.json')
        require(descriptor(path, self.check)['sha256'] == pin, 'external completion pin')
        result = read(path)
        require(result['status'] == 'completed' and set(result['files']) == expected
                and {p.name for p in directory.iterdir()} == expected | {'receipt.json'}, 'exact closed payload membership')
        for name, identity in result['files'].items():
            require(descriptor(regular(directory / name), self.check) == identity, 'closed payload identity')
        return result

    def authenticate(self):
        worker = self.closed(self.run, self.args.worker_sha256, payload_names())
        audit = self.closed(self.audit, self.args.audit_sha256, {'started.json', 'summary.json'})
        plan_path = regular(STUDY / 'plan-01.json')
        require(descriptor(plan_path, self.check)['sha256'] == PLAN_PIN == worker['plan_sha256'] == audit['plan_sha256'], 'exact frozen plan')
        plan = read(plan_path)
        require(worker['version'] == plan['version'] == 'otto-return-value-v1'
                and plan['status'] == 'frozen_before_native_run' and len(plan['sources']) == 162
                and worker['sources'] == plan['sources'] and worker['inputs'] == plan['inputs']
                and worker['limits'] == plan['limits'] and worker['completed_episodes'] == 720
                and worker['completed_fits'] == 9 and worker['prepared_episodes'] == 240
                and worker['training_updates'] == 31680 and worker['external_model_calls'] == 0
                and worker['pending'] == [] and all(v['attempted'] == v['returned'] for v in worker['calls'].values())
                and worker['calls']['native_reset']['returned'] == 731
                and worker['calls']['optimizer_update']['returned'] == 31680, 'complete scientific work')
        require(audit['version'] == 'otto-return-value-saved-audit-v1' and audit['agreement'] is True
                and audit['worker_sha256'] == self.args.worker_sha256
                and audit['producer_source_sha256'] == plan['sources'][RUNNER]
                and audit['source']['sha256'] == plan['sources'][AUDITOR]
                and audit['training_calls'] == audit['simulator_calls'] == audit['remote_model_calls'] == 0,
                'independent completed audit identity')
        for name, pin in plan['sources'].items():
            require(descriptor(regular(ROOT / name), self.check)['sha256'] == pin, 'unchanged frozen source')
        for value in plan['inputs'].values():
            require(descriptor(regular(ROOT / value['path']), self.check) == {k: value[k] for k in ('sha256', 'bytes')}, 'unchanged prior input')
        terminal_path = regular(STUDY / 'run-process-01.terminal.json')
        launch_path = regular(STUDY / 'run-process-01.launch.json')
        terminal_pin = descriptor(terminal_path, self.check)['sha256']
        require(terminal_pin == audit['terminal_sha256']
                and descriptor(launch_path, self.check)['sha256'] == worker['supervision_sha256'], 'audited process pins')
        terminal, launch, started = read(terminal_path), read(launch_path), read(self.run / 'started.json')
        request = {'plan': str(plan_path), 'plan_sha256': PLAN_PIN, 'output': str(self.run), 'supervision': str(launch_path)}
        require(started['request'] == request and started['launch'] == launch, 'original worker request and launch')
        audit_request = read(self.audit / 'started.json')['request']
        require(audit_request['run'] == str(self.run) and audit_request['plan'] == str(plan_path)
                and audit_request['terminal'] == str(terminal_path) and audit_request['plan_sha256'] == PLAN_PIN
                and audit_request['receipt_sha256'] == self.args.worker_sha256
                and audit_request['terminal_sha256'] == terminal_pin, 'audit command input joins')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                    'cap_seconds', 'watchdog_sha256', 'clock_source_sha256'):
            require(terminal[key] == launch[key], 'closed parent launch identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command[:2] == [plan['python_executable'], str(ROOT / RUNNER)] and len(command) == 10
                and dict(zip(command[2::2], command[3::2], strict=True))
                == {f'--{key.replace("_", "-")}': value for key, value in request.items()}, 'actual completed command')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True
                and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['clock_source_sha256'] == CLOCK_PIN
                and launch['watchdog_sha256'] == plan['sources'][SUPERVISOR]
                and launch['pid'] == launch['pgid'] != launch['parent_pid'] and launch['cwd'] == str(ROOT)
                and launch['cap_seconds'] == 5400 and launch['deadline_ns'] == launch['started_ns'] + 5400 * 10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
                and started['started_ns'] == worker['started_ns']
                and worker['wall_seconds'] == (worker['finished_ns'] - worker['started_ns']) / 1e9
                and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
                and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9, 'successful strict parent completion')
        require(0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in self.run.iterdir()) <= plan['limits']['output_bytes']
                and worker['calls']['native_step']['returned'] <= plan['limits']['native_steps'], 'saved run caps')
        witness_path = regular(STUDY / 'execution-witness.json')
        witness = read(witness_path)
        require(witness['status'] == 'completed' and witness['actual_tool_exit_code'] == 0
                and type(witness['actual_session_id']) is int and witness['actual_session_id'] > 0
                and isinstance(witness['actual_completion_chunk'], str) and witness['actual_completion_chunk']
                and witness['worker_sha256'] == self.args.worker_sha256 and witness['plan_sha256'] == PLAN_PIN
                and witness['terminal_sha256'] == terminal_pin, 'root terminal-tool witness joins')
        paths = [regular(self.run / name) for name in sorted(payload_names() | {'receipt.json'})]
        paths += [plan_path, witness_path, launch_path, regular(STUDY / 'run-process-01.log'), terminal_path]
        members = {p.relative_to(ROOT).as_posix(): descriptor(p, self.check) for p in paths}
        require(len(members) == 50, 'all50 unique original archive members')
        self.plan, self.worker = plan, worker
        return members, {'audit': descriptor(self.audit / 'receipt.json', self.check),
                         'audit_payloads': audit['files'], 'execution_witness': descriptor(witness_path, self.check)}

    def restore_text(self):
        return f'''# Restore the scalar-return study evidence

This archive preserves 50 original paths: all 44 worker payloads plus its
receipt, and the frozen plan, root execution witness, supervisor launch, log
and terminal. It includes every final checkpoint (nine), prediction archive
(nine), TRAIN/VALID dataset, public trajectory and operation journal.
No scientific success is implied by successful archive verification.

1. Download `openjev-otto-return-value-v1.tar.gz`, `manifest.json`,
   `SHA256SUMS.txt` and this `RESTORE.md` from the same release.
2. In that directory run `shasum -a 256 -c SHA256SUMS.txt`. Both the archive
   and its manifest must match; the sums also cover this document.
3. Use a fresh checkout, preserving any existing experiment directories:

```sh
git clone https://github.com/kw2828/OpenJev.git OpenJev-return-value
cd OpenJev-return-value
git checkout {FREEZE_COMMIT}
tar -xzf /absolute/path/to/openjev-otto-return-value-v1.tar.gz
```

The archive contains regular files under `output/otto-return-value-v1/` only;
extract at the repository root. Do not extract over an existing `run-01`.
The source freeze is `{FREEZE_COMMIT}`. The plan SHA256 is `{PLAN_PIN}`
and binds 162 source files. Subsequent publication helpers are outside that
scientific source closure. Original absolute runtime/request paths remain
unchanged in receipts; moving the archive does not rewrite those witnesses.

`manifest.json` maps each original path to its exact SHA256 and byte count.
Verify all 50 restored files against that mapping before numerical reuse.
The publisher performs streaming per-member readback, gzip integrity reading
and checks that the originals did not change. No arrays or models are loaded.

Audit receipts and reports are published separately as small repository
evidence; their identities are bound in the archive manifest. This archive
is the complete new worker run, not a bundle of all historical inputs.
Re-running lineage authentication also requires the earlier study artifacts
named by the plan and the pinned runtime. Byte verification itself requires
neither those earlier datasets nor NumPy, Torch, a model or a simulator.
'''

    def execute(self):
        require(not self.out.exists() and not any(p.is_symlink() for p in (self.out, *self.out.parents)), 'exclusive nonsymlink release')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            require(descriptor(regular(ROOT / CLOCK))['sha256'] == CLOCK_PIN, 'clock source before import')
            spec = importlib.util.spec_from_file_location('_return_archive_clock', ROOT / CLOCK)
            clock_module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = clock_module
            spec.loader.exec_module(clock_module)
            self.clock = clock_module.SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('archive emergency cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            members, evidence = self.authenticate()
            archive_path = self.out / 'openjev-otto-return-value-v1.tar.gz'
            check = self.check

            class CheckedReader:
                def __init__(self, stream):
                    self.stream = stream

                def read(self, size):
                    check()
                    return self.stream.read(size)

            with tarfile.open(archive_path, 'x:gz', compresslevel=6, copybufsize=1024**2) as archive:
                for name in members:
                    self.check()
                    info = archive.gettarinfo(ROOT / name, arcname=name)
                    require(info.isfile(), 'regular archive member')
                    with (ROOT / name).open('rb') as stream:
                        archive.addfile(info, CheckedReader(stream))
            seen = set()
            with tarfile.open(archive_path, 'r:gz') as archive:
                for entry in archive:
                    require(entry.isfile() and entry.name in members and entry.name not in seen, 'exact archive member set')
                    stream, digest, size = archive.extractfile(entry), hashlib.sha256(), 0
                    for chunk in iter(lambda stream=stream: stream.read(1024**2), b''):
                        self.check()
                        digest.update(chunk)
                        size += len(chunk)
                    require({'sha256': digest.hexdigest(), 'bytes': size} == members[entry.name], 'lossless archive readback')
                    seen.add(entry.name)
            require(seen == members.keys(), 'complete archive coverage')
            with gzip.open(archive_path, 'rb') as stream:
                for _ in iter(lambda: stream.read(1024**2), b''):
                    self.check()
            for name, identity in members.items():
                require(descriptor(regular(ROOT / name), self.check) == identity, 'unchanged original after archival')
            require(descriptor(self.audit / 'receipt.json', self.check) == evidence['audit'], 'unchanged external audit')
            for name, identity in evidence['audit_payloads'].items():
                require(descriptor(regular(self.audit / name), self.check) == identity, 'unchanged audit payload')
            for name, pin in self.plan['sources'].items():
                require(descriptor(regular(ROOT / name), self.check)['sha256'] == pin, 'unchanged frozen source after archival')
            with (self.out / 'RESTORE.md').open('x') as stream:
                stream.write(self.restore_text())
            archive_identity = descriptor(archive_path, self.check)
            manifest = {'version': VERSION, 'status': 'verified', 'worker_sha256': self.args.worker_sha256,
                        'audit_sha256': self.args.audit_sha256, 'plan_sha256': PLAN_PIN, 'freeze_commit': FREEZE_COMMIT,
                        'source': descriptor(Path(__file__), self.check), 'scientific_source_count': 162,
                        'limits': LIMITS, 'clock_backend': self.clock.backend, 'peak_rss_bytes': self.peak_rss,
                        'archive': {'name': archive_path.name, **archive_identity}, 'members': members,
                        'evidence': evidence, 'restore': descriptor(self.out / 'RESTORE.md', self.check),
                        'seconds': (self.clock.now_ns() - self.start) / 1e9,
                        'timing_scope': 'Native start through manifest preparation; publication rechecked before return.',
                        'scope': 'All44 payloads and receipt, including9 final checkpoints and9 predictions, plus plan, root execution witness and3 supervisor files. Saved-only byte preservation; no model, simulator, training or scientific scoring calls.'}
            write(self.out / 'manifest.json', manifest)
            with (self.out / 'SHA256SUMS.txt').open('x') as stream:
                for path in (archive_path, self.out / 'manifest.json', self.out / 'RESTORE.md'):
                    stream.write(descriptor(path, self.check)['sha256'] + '  ' + path.name + '\n')
            self.check()
            print(json.dumps({'status': 'verified', 'archive': manifest['archive'], 'members': len(members)}), flush=True)
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                if (self.out / 'manifest.json').exists():
                    (self.out / 'manifest.json').rename(self.out / 'manifest.invalid.json')
                write(self.out / 'failed.json', {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc(),
                      'worker_sha256': self.args.worker_sha256, 'audit_sha256': self.args.audit_sha256,
                      'clock_backend': self.clock.backend if self.clock else None, 'started_ns': self.start})
            except BaseException as secondary:  # noqa: BLE001 - Preserve the original archive failure.
                error.add_note(f'Failure receipt publication: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--audit-sha256', required=True)
    Archive(parser.parse_args()).execute()
