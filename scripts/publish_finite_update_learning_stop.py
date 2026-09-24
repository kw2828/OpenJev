"""Publish the original failed exact-update feasibility qualification, using JSON and opaque bytes.

No numerical library, model, generator, optimizer or array decoder is imported.
Original process/source/inventory authentication precedes all cost/report reads.
This publication cannot admit a scientific run or retry the closed qualification.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'finite-update-learning-v1'
VERSION = 'finite-update-learning-stopped-publication-v1'
REGISTRATION_SHA256 = 'df564a228c92291b7a3149037ee1bf94d3f03f6f289564dc21d40a6e9681bec5'
CLOSURE = {
    'engineering-01.receipt.json': 'a92438916caf461c94ed0b939de9cf1b4d3e7336394d4e1d58b29829a08362bf',
    'engineering-native-01.launch.json': '8b3f5b154c2dd89a52fab264ff6092e24e003540a3969928603a7d3edad8d3f7',
    'engineering-native-01.terminal.json': 'dcdbb9f63f8447e2b7c965656cd249f996c8945ee89e87d06f0f1dce3522b787',
}
KERNEL_SHA256 = '0922b1ecc92016b42410ef652d87f2a534be0b64b3750bff977947322e8ae375'
KERNEL_PUBLICATION_SHA256 = '1970b5920d5cb3b239bc2d2ddf79c99dc9280361195c5a9c403bb8bf9a9b78b7'
ARMS = ('original_free', 'matched_free', 'rounded')
CONFIG = {'seed_namespace': 433260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003, 'fit_seeds': [433261001, 433261002, 433261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}
EXPOSURE = {'config': {**CONFIG, 'seed_namespace': 942201, 'fit_seeds': [942301],
    'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.},
    'target_counts': {'prefix_updates': 1024, 'joint_updates': 3072}, 'maximum_projected_seconds': 90.}
PARENT_FILES = ('output/finite-rounded-learning-v1/study-registration.json',
    'research/finite-rounded-learning-results/summary.json', 'research/finite-rounded-learning-results/receipt.json',
    'output/finite-rounded-learning-package-v1/manifest.json',
    'output/finite-rounded-learning-delivery-v1/delivery-verification.json')
COUNTS = dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'optimizer_calls',
                       'generator_calls', 'native_calls', 'qualification_reruns'), 0)
OUTPUT = ROOT / 'research/finite-update-learning-stop-results'
REPORT = ROOT / 'research/finite-update-learning-stop-results.md'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
            and path.is_file(), 'ordinary absolute file without symlink components: ' + str(path))
    return path

def byte_descriptor(payload):
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}

def descriptor(path):
    return byte_descriptor(regular(path).read_bytes())

def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), 'finite saved JSON')

def read(path):
    value = json.loads(regular(path).read_text())
    finite_tree(value)
    return value

def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()

def write(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

def inventory(folder):
    require(folder.is_absolute() and folder.resolve() == folder
            and folder.is_dir() and not folder.is_symlink(), 'ordinary evidence directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no evidence symlinks')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = descriptor(path)
        else:
            require(path.is_dir(), 'regular files and directories only')
    return result

def archive(path, payloads):
    """Deterministic regular members followed by opaque, byte-exact verification."""
    with path.open('xb') as raw:
        with (gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed,
              tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as bundle):
            for name, payload in sorted(payloads.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(payload), 0o644, 0
                member.uid = member.gid = 0
                member.uname = member.gname = ''
                bundle.addfile(member, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    with tarfile.open(path, 'r:gz') as bundle:
        members = bundle.getmembers()
        require([member.name for member in members] == sorted(payloads), 'exact sorted archive roster')
        for member in members:
            require(member.isfile() and member.mode == 0o644 and member.mtime == 0
                    and member.uid == member.gid == 0 and member.uname == member.gname == '',
                    'deterministic ordinary archive members')
            with bundle.extractfile(member) as stream:
                require(stream.read() == payloads[member.name], 'byte-identical opaque archive roundtrip')
    return {'verified': True, 'members': len(payloads), 'bytes': sum(map(len, payloads.values()))}


def safe_relative(name):
    require(type(name) is str and name and '\\' not in name, 'portable relative member name')
    require(not PurePosixPath(name).is_absolute()
            and all(piece not in ('', '.', '..') for piece in name.split('/')), 'safe relative member')
    return name


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pytest', 'ruff')}}


def native_closure(plan, receipt, launch, terminal, command, cap, *, passed):
    require(receipt['launch'] == launch and all(terminal.get(k) == v for k, v in launch.items()),
            'original receipt, launch and terminal identities')
    require(launch['command'] == command and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == cap
            and launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and launch['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'exact original arguments, interpreter, cwd, native clock and deadline')
    require(all(type(launch[k]) is int and launch[k] > 0 for k in
                ('pid', 'pgid', 'parent_pid', 'started_ns', 'deadline_ns'))
            and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
            and launch['deadline_ns'] - launch['started_ns'] == cap * 10**9,
            'original process group and absolute deadline')
    cleanup = terminal['cleanup']
    require(terminal['status'] == ('completed' if passed else 'failed')
            and terminal['returncode'] == (0 if passed else 1)
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['timing_available'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and cleanup['reaped'] is True and cleanup['group_absent'] is True
            and cleanup['errors'] == [] and cleanup['signals'] == [], 'original child cleanly reaped without timeout')
    require(launch['started_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 10**9, 'native timing identities')


def scalar_agreement(saved, audited):
    """The original auditor's scalar tolerance; identities/counters stay exact."""
    if type(audited) is dict:
        require(type(saved) is dict and set(saved) == set(audited), 'exact saved metric mapping')
        for name in audited:
            scalar_agreement(saved[name], audited[name])
    elif type(audited) is list:
        require(type(saved) is list and len(saved) == len(audited), 'exact saved metric roster')
        for left, right in zip(saved, audited, strict=True):
            scalar_agreement(left, right)
    elif type(audited) is float:
        require(type(saved) in (int, float) and math.isfinite(saved) and math.isfinite(audited)
                and math.isclose(saved, audited, rel_tol=1e-10, abs_tol=1e-12),
                'saved independently audited scalar agrees at original tolerance')
    else:
        require(type(saved) is type(audited) and saved == audited, 'exact saved metric identity or count')


def authenticate_kernel(proof, sources):
    folder = ROOT / 'output/finite-rounded-transition-qualification-v1'
    publication = ROOT / 'research/finite-rounded-transition-qualification-results'
    require(proof['folder'] == str(folder) and proof['publication'] == str(publication)
            and inventory(folder) == proof['files'] and inventory(publication) == proof['publication_files'],
            'complete registered primitive proof and publication inventories')
    plan_path = folder / 'registration-01.json'
    require(descriptor(plan_path)['sha256'] == KERNEL_SHA256, 'original primitive registration pin')
    plan = read(plan_path)
    receipt = read(folder / 'engineering-01.receipt.json')
    launch, terminal = read(folder / 'native-01.launch.json'), read(folder / 'native-01.terminal.json')
    require(plan['version'] == 'finite-rounded-transition-qualification-v1' and plan['root'] == str(ROOT)
            and receipt['status'] == 'PASS' and receipt['version'] == plan['version']
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == KERNEL_SHA256
            and receipt['output'] == str(folder / 'engineering-01')
            and receipt['supervision'] == str(folder / 'native-01.launch.json')
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
            'original successful primitive qualification')
    command = [plan['runtime']['executable'], str(ROOT / 'scripts/qualify_finite_rounded_transition.py'),
        'worker', '--plan', str(plan_path), '--plan-sha256', KERNEL_SHA256,
        '--supervision', receipt['supervision'], '--output', receipt['output']]
    native_closure(plan, receipt, launch, terminal, command, 90, passed=True)
    require(len(receipt['commands']) == 2 and [r['command'] for r in receipt['commands']] == plan['commands']
            and all(r['returncode'] == 0 for r in receipt['commands'])
            and inventory(folder / 'engineering-01') == receipt['files'], 'original two primitive commands and log inventory')
    snapshot = folder / 'source-snapshot-01'
    manifest = snapshot / 'manifest.json'
    require(read(manifest) == {'registration': {'path': str(plan_path), **descriptor(plan_path)}, 'sources': plan['sources']}
            and inventory(snapshot) == {**plan['sources'], 'manifest.json': descriptor(manifest)},
            'exact original primitive source snapshot')
    require(all(sources.get(name) == pin == descriptor(ROOT / safe_relative(name))
                for name, pin in plan['sources'].items()), 'same primitive sources in current closure')
    path = publication / 'receipt.json'
    require(descriptor(path)['sha256'] == KERNEL_PUBLICATION_SHA256, 'original primitive publication receipt pin')
    published = read(path)
    require(published['status'] == 'PASS' and published['archive_roundtrip'] is True
            and published['numerical_calls'] == published['array_decodes'] == 0
            and published['inputs'] == proof['files']
            and proof['publication_files'] == {**published['outputs'], 'receipt.json': descriptor(path)},
            'unchanged opaque primitive publication and all its proof bytes')
    return {**{str(folder / name): pin for name, pin in proof['files'].items()},
            **{str(publication / name): pin for name, pin in proof['publication_files'].items()}}


def probe_run_names():
    names = {'config.json', 'train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
        'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz', 'fits.jsonl', 'training-orders.jsonl',
        'checkpoint-barrier.json', 'prediction-times.jsonl', 'oracle-train-check.json', 'oracle-base-check.json',
        'train-prefix.npz', 'base-prefix.npz', 'summary.json'}
    for arm in ARMS:
        names |= {f'prefix-{arm}-942301-base.npz', f'allocation-{arm}-942301.json'}
        for label in ('initial', 'boundary', 'final'):
            names |= {f'{label}-{arm}-942301.npz', f'{label}-optimizer-{arm}-942301.json'}
    return names


def authenticate(study, registration_sha256):
    study = Path(study)
    require(study == ROOT / 'output' / STUDY and Path.cwd() == ROOT, 'exact original study and cwd')
    plan_path = regular(study / 'engineering-registration-01.json')
    require(descriptor(plan_path)['sha256'] == registration_sha256 == REGISTRATION_SHA256,
            'externally supplied exact engineering registration')
    plan = read(plan_path)
    require(plan['version'] == STUDY and plan['mode'] == 'engineering' and plan['root'] == str(ROOT)
            and plan['runtime'] == runtime() and plan['config'] == CONFIG and plan['exposure_probe'] == EXPOSURE
            and len(plan['sources']) == 75 and plan['rss_limit_bytes'] == 4 * 1024**3
            and plan['output_limit_bytes'] == 512 * 1024**2 and 'qualification' not in plan,
            'original engineering-only plan, runtime and fixed exposure')
    expected_phases = {name: {'cap_seconds': cap, 'output': str(study / output),
                            'supervision': str(study / (prefix + '.launch.json'))}
        for name, cap, output, prefix in (('qualify', 300, 'engineering-01', 'engineering-native-01'),
            ('fit', 1200, 'run-01', 'fit-native-01'), ('audit', 600, 'audit-01', 'audit-native-01'))}
    require(plan['phases'] == expected_phases, 'exact original phase paths and caps')
    inputs = {str(plan_path): descriptor(plan_path)}
    for name, pin in plan['sources'].items():
        path = ROOT / safe_relative(name)
        require(descriptor(path) == pin, 'unchanged registered source: ' + name)
        inputs[str(path)] = pin
    snapshot = study / 'source-snapshot-engineering-01'
    snapshot_manifest = snapshot / 'manifest.json'
    require(read(snapshot_manifest) == {'registration': {'path': str(plan_path), **descriptor(plan_path)},
            'sources': plan['sources']}, 'original complete source snapshot manifest')
    snapshot_files = inventory(snapshot)
    require(snapshot_files == {**plan['sources'], 'manifest.json': descriptor(snapshot_manifest)},
            'all75 frozen copies plus manifest')
    for name, sha in CLOSURE.items():
        require(descriptor(study / name)['sha256'] == sha, 'exact original closure record: ' + name)
    phase = study / 'engineering-01'
    receipt_path = study / 'engineering-01.receipt.json'
    launch_path, terminal_path = study / 'engineering-native-01.launch.json', study / 'engineering-native-01.terminal.json'
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(receipt['phase'] == 'qualify' and receipt['status'] == 'FAILED'
            and receipt['error'] == "ValueError('qualification command 2 failed')"
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == registration_sha256
            and receipt['output'] == str(phase) and receipt['supervision'] == str(launch_path)
            and receipt['sources_before'] == plan['sources'] and 'sources_after' not in receipt and 'result' not in receipt,
            'original failed qualification, without inventing source-after success')
    command = [plan['runtime']['executable'], str(ROOT / 'scripts/finite_update_learning_worker.py'),
        '--plan', str(plan_path), '--plan-sha256', registration_sha256, '--phase', 'qualify',
        '--supervision', str(launch_path), '--output', str(phase)]
    native_closure(plan, receipt, launch, terminal, command, 300, passed=False)
    require(launch['started_unix'] <= receipt['started_unix'] <= receipt['finished_unix'] <= terminal['finished_unix'],
            'worker receipt lies inside original native process')
    phase_files = inventory(phase)
    expected_phase = {f'command-{index}.log' for index in range(3)} | {
        'exposure/audit.json', 'exposure/projections.json', 'exposure/receipt.json'} | {
        'exposure/run/' + name for name in probe_run_names()}
    require(len(expected_phase) == 47 and set(phase_files) == expected_phase and phase_files == receipt['files'],
            'all47 original qualification payloads, including the entire retained probe')
    commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
        [plan['runtime']['executable'], '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--noconftest', *plan['tests']],
        [plan['runtime']['executable'], str(ROOT / 'scripts/qualify_finite_update_exposure.py'),
            '--output', str(phase / 'exposure')]]
    require(len(receipt['commands']) == 3, 'exact three original commands')
    for record, argv, code in zip(receipt['commands'], commands, (0, 0, 1), strict=True):
        require(record['command'] == argv and record['returncode'] == code
                and type(record['seconds']) in (int, float) and math.isfinite(record['seconds'])
                and 0 <= record['seconds'] <= terminal['wall_seconds'], 'lint/tests passed and original cost probe failed')
    require(sum(row['seconds'] for row in receipt['commands']) <= terminal['wall_seconds'],
            'nested command durations fit original process duration')
    inputs.update(authenticate_kernel(plan['kernel_qualification'], plan['sources']))
    require(set(plan['parent_evidence']) == set(PARENT_FILES), 'exact five registered closed-parent evidence files')
    for name, pin in plan['parent_evidence'].items():
        require(descriptor(ROOT / name) == pin, 'unchanged opaque parent evidence: ' + name)
        inputs[str(ROOT / name)] = pin
    require(not (study / 'study-registration.json').exists(), 'no scientific registration')
    for name in ('fit', 'audit'):
        spec = plan['phases'][name]
        output, launch_file = Path(spec['output']), Path(spec['supervision'])
        for path in (output, Path(str(output) + '.receipt.json'), launch_file,
                     launch_file.with_name(launch_file.name.replace('.launch.json', '.terminal.json')),
                     launch_file.with_name(launch_file.name.replace('.launch.json', '.log'))):
            require(not path.exists(), 'no scientific phase artifact: ' + str(path))
    child_files = inventory(study)
    expected_child = {'engineering-registration-01.json', 'engineering-01.receipt.json',
        'engineering-native-01.launch.json', 'engineering-native-01.terminal.json', 'engineering-native-01.log'} | {
        'source-snapshot-engineering-01/' + name for name in snapshot_files} | {
        'engineering-01/' + name for name in phase_files}
    require(set(child_files) == expected_child, 'complete child study inventory without later attempts or science')
    inputs.update({str(study / name): pin for name, pin in child_files.items()})
    inputs[str(Path(__file__).resolve())] = descriptor(Path(__file__).resolve())
    # Nothing below this boundary reads an array. Cost interpretation begins only
    # after source, native closure, exact command and all byte-inventory checks.
    require((phase / 'command-0.log').read_text() == 'All checks passed!\n', 'original successful lint log')
    test_log = (phase / 'command-1.log').read_text()
    matches = re.findall(r'^273 passed, 1 warning in ([0-9]+(?:\.[0-9]+)?)s\s*$', test_log, re.MULTILINE)
    require(len(matches) == 1 and not re.search(r'^FAILED ', test_log, re.MULTILINE), '273passed/onewarning original test result')
    probe_folder = phase / 'exposure'
    probe = read(probe_folder / 'receipt.json')
    require(probe['version'] == 'finite-update-exposure-v1' and probe['status'] == 'FAILED'
            and probe['error'] == "ValueError('preselected science exposure must fit engineering feasibility bound')"
            and all(probe[key] == value for key, value in EXPOSURE.items())
            and 'audit' not in probe and 'run_files' not in probe,
            'original feasibility rejection before post-admission-only receipt fields')
    require(inventory(probe_folder) == {**probe['files_before_receipt'], 'receipt.json': descriptor(probe_folder / 'receipt.json')},
            'exact exposure subreceipt joins original retained probe bytes')
    run = probe_folder / 'run'
    summary, audited = read(run / 'summary.json'), read(probe_folder / 'audit.json')
    require(summary['version'] == STUDY and summary['config'] == EXPOSURE['config']
            and inventory(run) == {**summary['files'], 'summary.json': descriptor(run / 'summary.json')}
            and set(summary['files']) == probe_run_names() - {'summary.json'}, 'complete three-fit engineering producer')
    require(audited['version'] == 'finite-update-learning-audit-v1' and audited['profile'] == 'engineering-942201'
            and audited['agreement'] is True and audited['exact_oracle_agreement'] is True
            and audited['technical_complete'] is False and audited['requires_original_supervisor_closure'] is True
            and audited['metadata']['fits'] == 3 and audited['metadata']['payloads'] == 40
            and audited['counts'] == {'array_decodes': 21, 'checkpoint_decodes': 9, 'optimizer_json_decodes': 9,
                'model_calls': 0, 'native_calls': 0, 'optimizer_calls': 0, 'world_or_generator_calls': 0},
            'original independently saved engineering audit completed; no scientific audit claim')
    for name in ('rows', 'baseline_rows', 'prefix_rows'):
        scalar_agreement(summary[name], audited[name])
    require(audited['structural_work'] == summary['structural_work'], 'exact producer/audit structural-work agreement')
    fits = summary['fits']
    require([(r['arm'], r['seed']) for r in fits] == [(arm, 942301) for arm in ARMS]
            and fits == [json.loads(line) for line in regular(run / 'fits.jsonl').read_text().splitlines()],
            'all three original engineering fits, in registered order')
    projected = read(probe_folder / 'projections.json')
    require(projected == probe['projections'] and len(projected) == 3, 'durable unchanged original cost projections')
    stages_out = []
    for fit, row in zip(fits, projected, strict=True):
        arm = fit['arm']
        allocation = read(run / fit['allocation']['path'])
        require(fit['allocation'] == {'path': f'allocation-{arm}-942301.json',
                    **descriptor(run / f'allocation-{arm}-942301.json')}
                and allocation['status'] == 'PASS' and allocation['termination'] == 'completed_updates'
                and allocation['prefix_updates'] == allocation['accepted_prefix_updates'] == fit['accepted_prefix_updates'] == 32
                and allocation['joint_updates'] == allocation['accepted_joint_updates'] == fit['updates'] == 64
                and allocation['attempted_updates'] == allocation['accepted_updates'] == fit['attempted_updates'] == 96
                and len(allocation['trace']) == 96 and allocation['timed_seconds'] <= 30.
                and all(item['accepted'] is True and item['rolled_back'] is False for item in allocation['trace']),
                'exact completed probe updates, not a partial training fit')
        times = [item['stopped_elapsed'] - item['start_elapsed'] for item in allocation['stages']]
        require(len(times) == 2 and all(value >= 0 for value in times), 'both retained engineering stage durations')
        overhead = fit['seconds'] - sum(times)
        value = 2 * sum(seconds * scale for seconds, scale in zip(times, (32., 48.), strict=True)) + overhead
        expected = {'arm': arm, 'seed': 942301, 'stage_seconds': times, 'nonstage_seconds': overhead,
                    'safety_multiplier': 2., 'projected_seconds': value}
        require(row == expected and overhead >= 0 and value >= 90., 'all three original conservative projections fail fixed90sbound')
        stages_out.append({**row, 'probe_fit_seconds': fit['seconds'], 'accepted_prefix_updates': 32,
                           'accepted_joint_updates': 64, 'meets_feasibility_bound': False})
    require(summary['counts']['fit_count'] == 3 and summary['counts']['accepted_prefix_steps'] == 96
            and summary['counts']['accepted_joint_steps'] == 192 and summary['counts']['optimizer_steps'] == 288
            and summary['checkpoint_barrier']['fit_count'] == 3 and summary['checkpoint_barrier']['dev_generation_count'] == 0,
            'original exact engineering completion counts and pre-evaluation barrier')
    return {'plan': plan, 'plan_path': plan_path, 'receipt': receipt, 'terminal': terminal,
            'inputs': inputs, 'child_files': child_files, 'pytest_seconds': float(matches[0]),
            'projections': stages_out, 'probe_seconds': probe['seconds'],
            'probe_counts': summary['counts'], 'probe_audit_counts': audited['counts'],
            'probe_data_cases': audited['data_cases'], 'paired_batch_checks': audited['metadata']['paired_batch_checks']}

def summarize(auth):
    return {'version': VERSION, 'study': STUDY, 'status': 'QUALIFICATION_FAILED',
        'reason': 'PRESELECTED_EXPOSURE_COST_BOUND_FAILED',
        'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
        'tests_passed': 273, 'tests_failed': 0, 'test_warnings': 1, 'lint_passed': True,
        'original_process_returncode': 1, 'original_wall_seconds': auth['terminal']['wall_seconds'],
        'original_timing_scope': auth['terminal']['timing_scope'], 'pytest_seconds': auth['pytest_seconds'],
        'commands': auth['receipt']['commands'], 'registered_sources': 75,
        'child_study_files': len(auth['child_files']), 'qualification_payload_files': 47,
        'retained_probe_run_files': 41, 'probe_seconds': auth['probe_seconds'],
        'probe_config': EXPOSURE['config'], 'proposed_scientific_config': CONFIG,
        'maximum_projected_seconds': 90., 'strict_projection_rule': 'projected_seconds < 90.0',
        'projection_formula': '2 * (prefix_stage_seconds * 1024/32 + joint_stage_seconds * 3072/64) + measured_nonstage_fit_seconds',
        'projections': auth['projections'], 'engineering_producer_counts': auth['probe_counts'],
        'engineering_audit': {'agreement': True, 'profile': 'engineering-942201',
            'technical_complete': False, 'requires_original_supervisor_closure': True,
            'qualification_supervisor_status': 'failed', 'counts': auth['probe_audit_counts'],
            'data_cases': auth['probe_data_cases'], 'paired_batch_checks': auth['paired_batch_checks']},
        'scientific_registered': False, 'scientific_started': False, 'scientific_result': None,
        'advancement_eligible': False, 'publication_counts': COUNTS,
        'interpretation': 'The preselected update counts failed their frozen conservative feasibility rule. This is not a task-performance result, a failure of the completed engineering arithmetic audit, or evidence that a full fit would exceed its separate 120-second safety cap.',
        'limitations': [
            'The three projections are extrapolations from one engineering seed, not measured full scientific fits or runtime guarantees.',
            'Stage timings and historical execution are authenticated saved records, not replayed by this publisher.',
            'The original arithmetic audit completed inside the engineering command; the enclosing qualification failed on the subsequent cost rule.',
            'System-temporary pytest fixture outputs are not included or claimed preserved. Authoritative test logs and the complete original retained exposure probe are included.',
            'No scientific seed data were generated, no scientific run was admitted, and no threshold change, smaller schedule or retry is authorized by this report.',
            'Installed interpreter and packages remain external; the archive is evidence, not a bundled portable runtime.']}


def figure(summary):
    x, width, maximum = 195., 500., 120.
    threshold = x + width * 90 / maximum
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" viewBox="0 0 900 300">',
        '<rect width="900" height="300" fill="white"/>', '<g font-family="Arial,sans-serif" fill="#17202a">',
        '<text x="30" y="32" font-size="22">Exact-update qualification: cost bound failed</text>',
        '<text x="30" y="60" font-size="14">273 tests passed; engineering audit agreed. No scientific run.</text>',
        f'<line x1="{threshold}" y1="82" x2="{threshold}" y2="224" stroke="#17202a" stroke-dasharray="5,4"/>']
    for index, row in enumerate(summary['projections']):
        y = 105 + 47 * index
        extent = width * row['projected_seconds'] / maximum
        pieces.extend((f'<text x="30" y="{y + 5}" font-size="15">{html.escape(row["arm"])}</text>',
            f'<rect x="{x}" y="{y - 12}" width="{extent:.6f}" height="23" fill="#b74736"/>',
            f'<text x="{x + extent + 10:.6f}" y="{y + 5}" font-size="14">{row["projected_seconds"]:.6f} s</text>'))
    pieces.extend((f'<text x="{x}" y="244" font-size="12">0</text>',
        f'<text x="{threshold - 35}" y="244" font-size="12">strict &lt;90 s</text>',
        '<text x="30" y="280" font-size="13">Conservative projections, not measured full-fit times. Zero-origin seconds scale.</text>',
        '</g></svg>\n'))
    return ''.join(pieces).encode()


def report_text(summary):
    text = ('# Exact-update learning stopped at feasibility qualification\n\n'
        '**QUALIFICATION_FAILED.** All 273 tests passed with one warning, and the three retained '
        'engineering fits passed their independent saved-output checks. Each conservative full-fit '
        'projection exceeded the registered strict 90-second bound. No scientific study was registered or started.\n\n'
        '![Original feasibility stop](status.svg)\n\n'
        '| Arm | Prefix stage(s) | Joint stage(s) | Complete probe fit(s) | Projected full fit(s) | Feasible below 90s |\n'
        '| --- | ---: | ---: | ---: | ---: | --- |\n')
    for row in summary['projections']:
        text += (f'| {row["arm"]} | {row["stage_seconds"][0]:.9f} | {row["stage_seconds"][1]:.9f} | '
                 f'{row["probe_fit_seconds"]:.9f} | {row["projected_seconds"]:.9f} | FAIL |\n')
    text += ('\nThe engineering probe used namespace 942201, seed 942301, 512 TRAIN attempts and 8 DEV attempts. '
        'Each arm completed exactly 32 prefix updates and 64 joint updates. Its audit checked saved targets, '
        'predictions, paired batches and model/optimizer boundaries. These small engineering outputs are not '
        'scientific effectiveness results.\n\n'
        'The preselected scientific schedule was 1024 prefix and 3072 joint updates per fit. The frozen rule was '
        '`2 * (prefix_stage_seconds * 32 + joint_stage_seconds * 48) + measured_nonstage_fit_seconds < 90`. '
        'All three projections failed. This is a conservative feasibility rejection, not proof that a measured '
        'full fit would exceed the separate 120-second safety cap. Counts and thresholds were not adjusted.\n\n'
        f'The original supervised qualification closed with exit 1 after {summary["original_wall_seconds"]:.9f}s. '
        f'Tests took {summary["pytest_seconds"]:.2f}s and the exposure helper recorded {summary["probe_seconds"]:.9f}s internally; '
        f'the full exposure command took {summary["commands"][2]["seconds"]:.9f}s including child startup. '
        'These scopes are nested within the qualification, not additive. No retry, scientific registration, '
        'scientific fit, or scientific audit followed. The previous equal-time result remains separate and failed.\n\n'
        f'The archive preserves every {summary["child_study_files"]} original child-study file, '
        'all 75 current registered sources and their complete frozen snapshot, original command/process receipts '
        'and logs, the retained three-fit probe, the complete pinned rounded-kernel proof/publication, and '
        'the five registered parent-evidence files. Arrays and checkpoints are copied as opaque bytes. '
        'System-temporary pytest fixtures were not retained and are not claimed to be in this archive. '
        'The interpreter and installed packages remain external.\n\n'
        'The saved engineering audit reports arithmetic agreement while leaving original process admission '
        'to its caller. Its enclosing original qualification is authentically closed as failed. Publication '
        'does not relabel it as technical qualification success or establish a model/architecture gain.\n\n'
        '[Complete summary](summary.json) · [Exact member manifest](manifest.json) · '
        '[Evidence archive](evidence.tar.gz)\n')
    return text.encode()


def publish(study, output, registration_sha256):
    study, output = Path(study), Path(output)
    require(output == OUTPUT and output.parent.resolve() == output.parent and not output.exists()
            and not output.is_symlink() and not REPORT.exists() and not REPORT.is_symlink(),
            'exclusive prescribed output folder and overview')
    auth = authenticate(study, registration_sha256)
    summary = summarize(auth)
    report = report_text(summary)
    overview = report.decode()
    for name in ('status.svg', 'summary.json', 'manifest.json', 'evidence.tar.gz'):
        overview = overview.replace('(' + name + ')', '(finite-update-learning-stop-results/' + name + ')')
    outputs = {'summary.json': json_bytes(summary), 'report.md': report, 'status.svg': figure(summary)}
    payloads = {}
    for path_text, pin in auth['inputs'].items():
        path = regular(Path(path_text))
        require(path.is_relative_to(ROOT), 'archive inputs have repository-relative destinations')
        name = 'evidence/' + safe_relative(path.relative_to(ROOT).as_posix())
        payload = path.read_bytes()
        require(byte_descriptor(payload) == pin and name not in payloads, 'unique unchanged opaque input')
        payloads[name] = payload
    payloads.update({'publication/' + name: value for name, value in outputs.items()})
    payloads['publication/finite-update-learning-stop-results.md'] = overview.encode()
    manifest = {'version': VERSION, 'status': 'QUALIFICATION_FAILED', 'registration': summary['registration'],
        'sources': auth['plan']['sources'], 'inputs': auth['inputs'], 'child_study_files': auth['child_files'],
        'files': {name: byte_descriptor(value) for name, value in sorted(payloads.items())},
        'manifest_coverage': 'Every other archive member. MANIFEST.json equals this external manifest and excludes itself.',
        'scope': 'Original failed engineering qualification, all registered sources and snapshots, retained exposure probe, kernel proof/publication and registered parent evidence. No temporary pytest output preservation is claimed.',
        'archive_metadata': 'Sorted regular files, mode0644, uid/gid0, empty owner names, timestamps0, empty gzip filename.',
        'runtime': auth['plan']['runtime'], 'portable_environment_included': False,
        'scientific_registered': False, 'scientific_started': False, 'publication_counts': COUNTS}
    manifest_payload = json_bytes(manifest)
    payloads['MANIFEST.json'] = manifest_payload
    output.mkdir()
    for name, payload in outputs.items():
        write(output / name, payload)
    write(output / 'manifest.json', manifest_payload)
    roundtrip = archive(output / 'evidence.tar.gz', payloads)
    after = authenticate(study, registration_sha256)
    require(after == auth, 'all source, original evidence and dependency pins unchanged after publication')
    write(REPORT, overview.encode())
    receipt = {'version': VERSION, 'status': 'PASS', 'scientific_status': 'QUALIFICATION_FAILED',
        'publisher': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
        'registration': summary['registration'], 'output': str(output),
        'overview': {'path': str(REPORT), **descriptor(REPORT)},
        'files': {name: descriptor(output / name) for name in (*outputs, 'manifest.json', 'evidence.tar.gz')},
        'inputs_before': auth['inputs'], 'inputs_after': after['inputs'], 'inputs_unchanged': True,
        'opaque_archive_roundtrip': roundtrip, 'publication_counts': COUNTS,
        'scientific_registered': False, 'scientific_started': False}
    write(output / 'receipt.json', json_bytes(receipt))
    return {'output': str(output), 'report': str(REPORT), 'receipt': descriptor(output / 'receipt.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.output, args.registration_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
