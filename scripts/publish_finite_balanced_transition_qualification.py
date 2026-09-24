"""Publish one closed fabricated qualification using standard-library metadata.

This helper never imports the primitive, numerical packages, a model or a data
decoder. The archived environment description is not a portable environment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY_VERSION = 'finite-balanced-transition-qualification-v1'
VERSION = 'finite-balanced-transition-qualification-publication-v1'
REGISTRATION_SHA256 = 'afc981803d96830c4946c5f9e0702c42ae0f888c1cd9be59311d9338e0905bd2'
ORIGINAL_CLOSURE = {
    'engineering-01.receipt.json': '60d5bf44cb5359f5b08dab64dee40e0b00ca2dcaca17ad67eec7dbd544a81d93',
    'native-01.launch.json': '874ad0fc09baea86841b88b90a5145068eb3bbc2683edf30fad6206d7b4081eb',
    'native-01.terminal.json': '6fd9f2591980479072496efc968e1ffd31f8c90bdf046cfbd07bb9f26f85e91c',
    'native-01.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}
SOURCES = (
    'src/openjev/research/finite_balanced_transition.py',
    'tests/test_finite_balanced_transition.py',
    'scripts/qualify_finite_balanced_transition.py',
    'research/finite-balanced-transition-qualification-protocol.md',
    'scripts/supervise_dialogue_observation_v2.py',
    'src/openjev/research/suspend_clock.py',
    'src/openjev/__init__.py', 'src/openjev/research/__init__.py',
    'pyproject.toml', 'uv.lock',
)
ENVIRONMENT = {key: '1' for key in (
    'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
)}
ENVIRONMENT.update(PYTEST_ADDOPTS='', PYTEST_PLUGINS='')
PUBLICATION_COUNTS = {
    'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0,
    'optimizer_calls': 0, 'generator_calls': 0, 'numerical_qualification_reruns': 0,
}


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


def current_runtime():
    return {'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('torch', 'numpy', 'pytest', 'ruff')}}


def authenticate(study, registration_sha256):
    """Validate original process and all bytes before interpreting test logs."""
    study = Path(study).absolute()
    require(study == ROOT / 'output' / STUDY_VERSION and Path.cwd() == ROOT,
            'exact original study and repository cwd')
    plan_path = study / 'registration-01.json'
    require(descriptor(plan_path)['sha256'] == registration_sha256 == REGISTRATION_SHA256,
            'exact supplied original registration SHA256')
    plan = read(plan_path)
    phase = study / 'engineering-01'
    snapshot = study / 'source-snapshot-01'
    launch_path = study / 'native-01.launch.json'
    terminal_path = study / 'native-01.terminal.json'
    receipt_path = study / 'engineering-01.receipt.json'
    require(plan['version'] == STUDY_VERSION and plan['root'] == str(ROOT)
            and plan['output'] == str(phase) and plan['snapshot'] == str(snapshot)
            and plan['supervision'] == str(launch_path) and plan['cap_seconds'] == 90
            and plan['output_limit_bytes'] == 8 * 1024**2
            and plan['environment'] == ENVIRONMENT and plan['runtime'] == current_runtime(),
            'original fixed paths, limits, environment and current runtime')
    require(set(plan['sources']) == set(SOURCES) and len(plan['sources']) == 10,
            'exact ten-source closure')
    for name in SOURCES:
        require(descriptor(ROOT / name) == plan['sources'][name], 'unchanged current source: ' + name)
    snapshot_manifest = read(snapshot / 'manifest.json')
    require(snapshot_manifest == {
        'registration': {'path': str(plan_path), **descriptor(plan_path)}, 'sources': plan['sources'],
    }, 'snapshot manifest associated with original registration')
    require(inventory(snapshot) == {**plan['sources'], 'manifest.json': descriptor(snapshot / 'manifest.json')},
            'all ten original source copies plus snapshot manifest')
    commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *SOURCES[:3]],
                [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 '--noconftest', SOURCES[1]]]
    require(plan['commands'] == commands, 'exact two registered qualification commands')
    for name, digest in ORIGINAL_CLOSURE.items():
        require(descriptor(study / name)['sha256'] == digest, 'original closed process bytes: ' + name)
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(receipt['version'] == STUDY_VERSION and receipt['status'] == 'PASS'
            and 'error' not in receipt and receipt['plan'] == str(plan_path)
            and receipt['plan_sha256'] == registration_sha256
            and receipt['output'] == str(phase) and receipt['supervision'] == str(launch_path)
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['launch'] == launch, 'original successful worker receipt and source joins')
    expected = [sys.executable, str(ROOT / SOURCES[2]), 'worker', '--plan', str(plan_path),
                '--plan-sha256', registration_sha256, '--supervision', str(launch_path), '--output', str(phase)]
    require(launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['command'] == expected and launch['cwd'] == str(ROOT)
            and launch['cap_seconds'] == 90
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['watchdog_sha256'] == plan['sources'][SOURCES[4]]['sha256']
            and launch['clock_source_sha256'] == plan['sources'][SOURCES[5]]['sha256'],
            'exact original argv, cwd, cap and native clock source binding')
    require(all(type(launch[key]) is int and launch[key] > 0
                for key in ('pid', 'pgid', 'parent_pid', 'started_ns', 'deadline_ns'))
            and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid']
            and launch['deadline_ns'] - launch['started_ns'] == 90_000_000_000,
            'original process group and native deadline')
    require(all(terminal.get(key) == value for key, value in launch.items()),
            'complete original launch/terminal join')
    cleanup = terminal['cleanup']
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['timing_available'] is True and terminal['timed_out'] is False
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['group_absent'] is True and cleanup['group_absent'] is True
            and cleanup['reaped'] is True and cleanup['errors'] == [] and cleanup['signals'] == [],
            'original successful closure, no timeout, reaped child and clean process group')
    require(type(terminal['finished_ns']) is int and type(terminal['elapsed_ns']) is int
            and launch['started_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1_000_000_000,
            'consistent original native elapsed time including group cleanup')
    require(launch['started_unix'] <= receipt['started_unix'] <= receipt['finished_unix']
            <= terminal['finished_unix'], 'original civil timestamp ordering')
    files = inventory(phase)
    require(files == receipt['files'] and set(files) == {'command-0.log', 'command-1.log'}
            and sum(pin['bytes'] for pin in files.values()) <= plan['output_limit_bytes'],
            'complete bounded command log inventory')
    require(len(receipt['commands']) == 2, 'exact two completed commands')
    for index, row in enumerate(receipt['commands']):
        require(row['command'] == commands[index] and row['returncode'] == 0
                and row['log'] == files[f'command-{index}.log']
                and type(row['elapsed_ns']) is int and 0 <= row['elapsed_ns'] <= terminal['elapsed_ns'],
                'completed command and original log identity')
    require(sum(row['elapsed_ns'] for row in receipt['commands']) <= terminal['elapsed_ns'],
            'command durations nested inside original phase')
    child = inventory(study)
    expected_members = {'registration-01.json', 'engineering-01.receipt.json', 'native-01.launch.json',
                        'native-01.terminal.json', 'native-01.log', 'engineering-01/command-0.log',
                        'engineering-01/command-1.log', 'source-snapshot-01/manifest.json'}
    expected_members.update('source-snapshot-01/' + name for name in SOURCES)
    require(set(child) == expected_members, 'entire exact original eighteen-file study inventory')
    # Interpret only the authenticated logs; no numerical source is imported.
    require((phase / 'command-0.log').read_text() == 'All checks passed!\n', 'original lint passed')
    test_log = (phase / 'command-1.log').read_text()
    match = re.fullmatch(r'\.{42}\s+\[100%\]\n42 passed in ([0-9]+(?:\.[0-9]+)?)s\n', test_log)
    require(match is not None, 'exact original 42-pass test log without hidden failures or skips')
    inputs = {str(study / name): pin for name, pin in child.items()}
    inputs.update({str(ROOT / name): pin for name, pin in plan['sources'].items()})
    inputs[str(Path(__file__).resolve())] = descriptor(Path(__file__).resolve())
    return {'plan': plan, 'plan_path': plan_path, 'receipt': receipt, 'receipt_path': receipt_path,
            'launch_path': launch_path, 'terminal': terminal, 'terminal_path': terminal_path,
            'child_files': child, 'inputs': inputs, 'pytest_seconds': float(match.group(1))}


def summarize(auth):
    return {
        'version': VERSION, 'study_version': STUDY_VERSION, 'status': 'QUALIFICATION_PASS',
        'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
        'worker_receipt': {'path': str(auth['receipt_path']), **descriptor(auth['receipt_path'])},
        'original_terminal': {'path': str(auth['terminal_path']), **descriptor(auth['terminal_path'])},
        'tests_passed': 42, 'tests_failed': 0, 'lint_passed': True, 'source_files': 10,
        'snapshot_source_files': 10, 'study_files': len(auth['child_files']),
        'original_wall_seconds': auth['terminal']['wall_seconds'],
        'original_timing_scope': auth['terminal']['timing_scope'],
        'pytest_reported_seconds': auth['pytest_seconds'],
        'commands': auth['receipt']['commands'], 'runtime': auth['plan']['runtime'],
        'primitive': {'dtype': 'CPU float64', 'shape': [4, 8, 8], 'default_sweeps': 64,
                      'default_tolerance': 1e-12, 'order': 'row dim2, then column dim1',
                      'derivative': 'autograd through the finite normalization algorithm'},
        'scope': 'Fabricated numerical primitive qualification only; no learned model or empirical evaluation.',
        'limits': ['Finite sweeps are not a universal convergence guarantee for arbitrary positive matrices.',
                   'No training, integration, task performance, architectural novelty or scientific advancement is admitted.',
                   'Timings describe this original qualification, not model speed or a comparative benchmark.',
                   'The archive preserves source and evidence; interpreter and installed packages remain external.'],
        'training_advances': False, 'scientific_execution_admitted': False,
        'publication_counts': PUBLICATION_COUNTS,
    }


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


def publish(study, output, registration_sha256):
    study, output = Path(study).absolute(), Path(output).absolute()
    require(output == ROOT / 'research/finite-balanced-transition-qualification-results'
            and output.parent.resolve() == output.parent and not output.exists() and not output.is_symlink(),
            'exclusive prescribed publication folder')
    auth = authenticate(study, registration_sha256)
    summary = summarize(auth)
    summary_payload = json_bytes(summary)
    payloads = {}
    for name, pin in auth['child_files'].items():
        payload = regular(study / name).read_bytes()
        require(byte_descriptor(payload) == pin, 'evidence unchanged while assembling archive')
        payloads[f'evidence/{STUDY_VERSION}/{name}'] = payload
    source = Path(__file__).resolve()
    source_payload = source.read_bytes()
    require(byte_descriptor(source_payload) == auth['inputs'][str(source)], 'publisher source unchanged')
    payloads['publication/' + source.name] = source_payload
    payloads['publication/summary.json'] = summary_payload
    manifest = {
        'version': VERSION, 'registration': summary['registration'], 'scope': summary['scope'],
        'training_advances': False, 'sources': auth['plan']['sources'], 'inputs': auth['inputs'],
        'files': {name: byte_descriptor(payload) for name, payload in sorted(payloads.items())},
        'self_exclusion': 'files covers every other archive member; MANIFEST.json excludes itself.',
        'external_dependencies': {'runtime': auth['plan']['runtime'],
                                  'absolute_repository_root': str(ROOT), 'portable_execution_claimed': False},
    }
    manifest_payload = json_bytes(manifest)
    payloads['MANIFEST.json'] = manifest_payload
    output.mkdir()
    write(output / 'summary.json', summary_payload)
    write(output / 'manifest.json', manifest_payload)
    roundtrip = archive(output / 'evidence.tar.gz', payloads)
    after = authenticate(study, registration_sha256)
    require(after == auth, 'all original input hashes, runtime, source pins and closures unchanged after publication')
    receipt = {
        'version': VERSION, 'status': 'PASS', 'registration': summary['registration'],
        'publisher': {'path': str(source), **descriptor(source)}, 'output': str(output),
        'files': {name: descriptor(output / name) for name in ('summary.json', 'manifest.json', 'evidence.tar.gz')},
        'inputs_before': auth['inputs'], 'inputs_after': after['inputs'], 'inputs_unchanged': True,
        'archive_roundtrip': roundtrip, 'publication_counts': PUBLICATION_COUNTS,
        'training_advances': False, 'scientific_execution_admitted': False,
    }
    write(output / 'receipt.json', json_bytes(receipt))
    return {'output': str(output), 'receipt': descriptor(output / 'receipt.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.output, args.registration_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
