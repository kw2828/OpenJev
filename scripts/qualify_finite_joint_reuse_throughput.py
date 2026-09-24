"""Register and bound one integrated engineering throughput comparison.

Only standard-library metadata and the qualified native supervisor are imported
here. Numerical work occurs in the exact registered child commands. A worker
receipt is not an original process closure; later phases require both.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from benchmark_finite_joint_reuse_throughput import CONFIG, make_schedule
from qualify_finite_joint_reuse import SOURCES as PARENT_SOURCES
from supervise_dialogue_observation_v2 import publish

from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-joint-reuse-throughput-v1'
FOLDER = ROOT / 'output' / VERSION
PLAN_PATH = FOLDER / 'registration.json'
SNAPSHOT = FOLDER / 'source-snapshot'
PRIMITIVE = ROOT / 'output/finite-joint-reuse-qualification-v1'
PUBLICATION = ROOT / 'research/finite-joint-reuse-qualification-results'
PRIMITIVE_SHA256 = '9f0ce32aef4ae456a88bba8d79b2073404e053ac6fba4a9c54bed219deb7e6e0'
PUBLICATION_SHA256 = '8e1565b7eedd753fa18b0185841d0d0631b05bc12af7f7233b1bc1a257379909'
NEW_SOURCES = (
    'scripts/run_finite_joint_reuse_training.py',
    'tests/test_finite_joint_reuse_training.py',
    'scripts/benchmark_finite_joint_reuse_throughput.py',
    'scripts/audit_finite_joint_reuse_throughput.py',
    'tests/test_finite_joint_reuse_throughput_audit.py',
    'scripts/qualify_finite_joint_reuse_throughput.py',
    'research/finite-joint-reuse-throughput-protocol.md',
    'scripts/publish_finite_joint_reuse_qualification.py',
)
SOURCES = tuple(dict.fromkeys((*PARENT_SOURCES, *NEW_SOURCES)))
TESTS = ('tests/test_finite_joint_reuse.py', 'tests/test_finite_joint_reuse_training.py',
         'tests/test_finite_joint_reuse_throughput_audit.py')
ENV = {key: '1' for key in (
    'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
)}
ENV.update(PYTEST_ADDOPTS='', PYTEST_PLUGINS='')
OUTPUT_LIMIT_BYTES = 512 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'ordinary absolute file: ' + str(path))
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return {'sha256': value.hexdigest(), 'bytes': path.stat().st_size}


def files(folder):
    folder = Path(folder)
    require(folder.is_absolute() and folder.resolve() == folder and folder.is_dir()
            and not folder.is_symlink(), 'ordinary output directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no evidence symlinks')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = descriptor(path)
        else:
            require(path.is_dir(), 'ordinary evidence entries')
    return result


def read(path):
    descriptor(path)
    return json.loads(Path(path).read_text())


def sources():
    return {name: descriptor(ROOT / name) for name in SOURCES}


def runtime():
    return {'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('torch', 'numpy', 'pytest', 'ruff')}}


def phase_specs():
    phases = {}
    for phase, cap, output, prefix in (
            ('qualify', 180, 'engineering-01', 'qualify-native-01'),
            ('run', 120, 'run-01', 'run-native-01'),
            ('audit', 60, 'audit-01', 'audit-native-01')):
        phases[phase] = {'cap_seconds': cap, 'output': str(FOLDER / output),
                         'supervision': str(FOLDER / (prefix + '.launch.json'))}
    return phases


def commands(phase):
    phases = phase_specs()
    if phase == 'qualify':
        return [[str(ROOT / '.venv/bin/ruff'), 'check',
                 *[name for name in NEW_SOURCES if name.endswith('.py')]],
                [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 '--noconftest', *TESTS]]
    if phase == 'run':
        return [[sys.executable, str(ROOT / 'scripts/benchmark_finite_joint_reuse_throughput.py'),
                 '--output', phases['run']['output'] + '/benchmark']]
    require(phase == 'audit', 'declared phase')
    return [[sys.executable, str(ROOT / 'scripts/audit_finite_joint_reuse_throughput.py'),
             '--study', str(FOLDER), '--output', phases['audit']['output'] + '/audit.json']]


def prerequisite():
    """Revalidate the published primitive and every retained historical input."""
    receipt_path = PUBLICATION / 'receipt.json'
    require(descriptor(receipt_path)['sha256'] == PUBLICATION_SHA256,
            'exact original primitive publication receipt')
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS' and receipt['inputs_unchanged'] is True
            and receipt['inputs_before'] == receipt['inputs_after']
            and receipt['scientific_execution_admitted'] is False
            and receipt['training_advances'] is False
            and files(PUBLICATION) == {**receipt['files'], 'receipt.json': descriptor(receipt_path)},
            'complete unchanged published component qualification')
    for name, pin in receipt['inputs_before'].items():
        require(Path(name).is_relative_to(ROOT) and descriptor(Path(name)) == pin,
                'unchanged component and historical input')
    overview = receipt['overview']
    require(overview == {'path': str(ROOT / 'research/finite-joint-reuse-qualification-results.md'),
                         **descriptor(Path(overview['path']))}, 'receipt-bound component overview')
    original_path = PRIMITIVE / 'registration-01.json'
    require(descriptor(original_path)['sha256'] == PRIMITIVE_SHA256
            and receipt['registration'] == {'path': str(original_path), **descriptor(original_path)},
            'original component registration identity')
    original = read(original_path)
    require(set(original['sources']) == set(PARENT_SOURCES) and len(original['sources']) == 80,
            'exact80-source prerequisite')
    expected = {str(Path(name).relative_to(PRIMITIVE)): pin
                for name, pin in receipt['inputs_before'].items() if Path(name).is_relative_to(PRIMITIVE)}
    require(files(PRIMITIVE) == expected, 'complete original component inventory')
    prior = original['prior_stop']
    require(files(Path(prior['folder'])) == prior['files']
            and files(Path(prior['publication'])) == prior['publication_files'],
            'prior failed qualification and publication inventories remain unchanged')
    return {'registration': {'path': str(original_path), **descriptor(original_path)},
            'publication_receipt': {'path': str(receipt_path), **descriptor(receipt_path)},
            'publication_files': files(PUBLICATION), 'overview': overview,
            'original_files': expected, 'inputs': receipt['inputs_before'], 'prior_stop': prior}


def validate_plan(plan):
    require(Path.cwd() == ROOT and plan['version'] == VERSION and plan['root'] == str(ROOT)
            and plan['runtime'] == runtime() and plan['environment'] == ENV
            and plan['phases'] == phase_specs() and plan['config'] == CONFIG
            and plan['schedule'] == make_schedule() and plan['snapshot'] == str(SNAPSHOT)
            and plan['output_limit_bytes'] == OUTPUT_LIMIT_BYTES
            and plan['tests'] == list(TESTS)
            and plan['commands'] == {phase: commands(phase) for phase in phase_specs()},
            'exact registered settings, runtime, paths, commands and schedule')
    require(sources() == plan['sources'], 'all registered current source bytes unchanged')
    require(prerequisite() == plan['prerequisite'], 'full published prerequisite unchanged')
    require(read(SNAPSHOT / 'manifest.json') == {
        'registration': {'path': str(PLAN_PATH), **descriptor(PLAN_PATH)}, 'sources': plan['sources']},
        'snapshot manifest joins original registration')
    require(files(SNAPSHOT) == {**plan['sources'], 'manifest.json': descriptor(SNAPSHOT / 'manifest.json')},
            'complete unchanged registered source snapshot')


def expected_worker(phase, plan_sha256):
    spec = phase_specs()[phase]
    return [sys.executable, str(Path(__file__).resolve()), 'worker', '--plan', str(PLAN_PATH),
            '--plan-sha256', plan_sha256, '--phase', phase, '--supervision', spec['supervision'],
            '--output', spec['output']]


def validate_launch(plan, phase, launch, plan_sha256):
    spec = plan['phases'][phase]
    require(launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['command'] == expected_worker(phase, plan_sha256)
            and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == spec['cap_seconds']
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and launch['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256']
            and all(type(launch[key]) is int and launch[key] > 0
                    for key in ('pid', 'pgid', 'parent_pid', 'started_ns', 'deadline_ns'))
            and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid']
            and launch['deadline_ns'] - launch['started_ns'] == spec['cap_seconds'] * 10**9,
            'exact original worker argv, process identity, native deadline and source binding')


def closed_phase(plan, phase):
    """Read-only admission of one original successfully closed predecessor."""
    validate_plan(plan)
    plan_sha = descriptor(PLAN_PATH)['sha256']
    require(read(PLAN_PATH) == plan, 'original registered plan bytes')
    spec = plan['phases'][phase]
    output, launch_path = Path(spec['output']), Path(spec['supervision'])
    receipt_path = Path(str(output) + '.receipt.json')
    terminal_path = Path(str(launch_path).replace('.launch.json', '.terminal.json'))
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(receipt['version'] == VERSION and receipt['phase'] == phase and receipt['status'] == 'PASS'
            and 'error' not in receipt and receipt['plan'] == str(PLAN_PATH)
            and receipt['plan_sha256'] == plan_sha and receipt['output'] == str(output)
            and receipt['supervision'] == str(launch_path) and receipt['launch'] == launch
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and files(output) == receipt['files'], 'exact successful predecessor and payload inventory')
    validate_launch(plan, phase, launch, plan_sha)
    require(all(terminal.get(key) == value for key, value in launch.items())
            and terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['timed_out'] is False and terminal['timing_available'] is True
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
            and terminal['cleanup']['signals'] == []
            and terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 10**9,
            'original completed native process, exact times and clean group closure')
    require(len(receipt['commands']) == len(plan['commands'][phase]), 'complete predecessor command roster')
    for index, row in enumerate(receipt['commands']):
        require(row['command'] == plan['commands'][phase][index] and row['returncode'] == 0
                and row['log'] == descriptor(output / f'command-{index}.log')
                and type(row['elapsed_ns']) is int and 0 <= row['elapsed_ns'] <= terminal['elapsed_ns'],
                'exact completed predecessor command and log')
    require(sum(row['elapsed_ns'] for row in receipt['commands']) <= terminal['elapsed_ns'],
            'nested command timings')
    predecessors = {'qualify': (), 'run': ('qualify',), 'audit': ('qualify', 'run')}[phase]
    expected_predecessors = {}
    for previous in predecessors:
        previous_spec = plan['phases'][previous]
        paths = {'receipt_path': Path(previous_spec['output'] + '.receipt.json'),
                 'launch_path': Path(previous_spec['supervision']),
                 'terminal_path': Path(previous_spec['supervision'].replace('.launch.json', '.terminal.json'))}
        expected_predecessors[previous] = {key: descriptor(path) for key, path in paths.items()}
        previous_terminal = read(paths['terminal_path'])
        require(previous_terminal['clock_backend'] == launch['clock_backend']
                and previous_terminal['finished_ns'] <= launch['started_ns'],
                'predecessor originally closed before current launch')
    require(receipt['predecessors'] == expected_predecessors, 'exact unchanged predecessor closure pins')
    return {'receipt': receipt, 'receipt_path': receipt_path, 'launch_path': launch_path,
            'terminal': terminal, 'terminal_path': terminal_path, 'directory': output}


def register():
    require(Path.cwd() == ROOT and not PLAN_PATH.exists() and not SNAPSHOT.exists(),
            'exclusive registration in repository cwd')
    FOLDER.mkdir(exist_ok=True)
    require(not files(FOLDER), 'no previous throughput attempt or phase')
    plan = {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(), 'environment': ENV,
            'sources': sources(), 'snapshot': str(SNAPSHOT), 'phases': phase_specs(),
            'commands': {phase: commands(phase) for phase in phase_specs()}, 'tests': list(TESTS),
            'config': CONFIG, 'schedule': make_schedule(), 'output_limit_bytes': OUTPUT_LIMIT_BYTES,
            'prerequisite': prerequisite(),
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'scope': 'Engineering paired training throughput only; no DEV, task evaluation or scientific admission.'}
    publish(PLAN_PATH, plan)
    SNAPSHOT.mkdir()
    for name, pin in plan['sources'].items():
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        require(descriptor(target) == pin, 'unchanged source snapshot bytes')
    publish(SNAPSHOT / 'manifest.json', {'registration': {'path': str(PLAN_PATH), **descriptor(PLAN_PATH)},
                                       'sources': plan['sources']})
    validate_plan(plan)
    return {'plan': str(PLAN_PATH), **descriptor(PLAN_PATH)}


def worker(args):
    output, plan_path = args.output.resolve(), args.plan.resolve()
    receipt_path = Path(str(output) + '.receipt.json')
    require(plan_path == PLAN_PATH and output == Path(phase_specs()[args.phase]['output'])
            and args.supervision.resolve() == Path(phase_specs()[args.phase]['supervision'])
            and not output.exists() and not receipt_path.exists(), 'exclusive exact registered worker paths')
    record = {'version': VERSION, 'phase': args.phase, 'status': 'FAILED',
              'plan': str(plan_path), 'plan_sha256': args.plan_sha256, 'output': str(output),
              'supervision': str(args.supervision.resolve()), 'commands': [], 'started_unix': time.time()}
    try:
        require(descriptor(plan_path)['sha256'] == args.plan_sha256, 'externally supplied registration digest')
        plan = read(plan_path)
        validate_plan(plan)
        require(all(os.environ.get(key) == value for key, value in ENV.items()), 'registered numerical environment')
        record['sources_before'] = sources()
        predecessors = {'qualify': (), 'run': ('qualify',), 'audit': ('qualify', 'run')}[args.phase]
        record['predecessors'] = {}
        for phase in predecessors:
            closed = closed_phase(plan, phase)
            record['predecessors'][phase] = {key: descriptor(closed[key])
                for key in ('receipt_path', 'launch_path', 'terminal_path')}
        for _ in range(100):
            if args.supervision.is_file():
                break
            time.sleep(.01)
        launch = read(args.supervision.resolve())
        validate_launch(plan, args.phase, launch, args.plan_sha256)
        require(launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp(), 'original current child identity')
        for phase in predecessors:
            previous_terminal = read(Path(plan['phases'][phase]['supervision'].replace('.launch.json', '.terminal.json')))
            require(previous_terminal['clock_backend'] == launch['clock_backend']
                    and previous_terminal['finished_ns'] <= launch['started_ns'],
                    'predecessor originally closed before current launch')
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'original suspend-inclusive clock backend')
        record['launch'] = launch

        def check():
            require(launch['started_ns'] <= clock.now_ns() < launch['deadline_ns'], 'original native deadline')
            if output.exists():
                require(sum(pin['bytes'] for pin in files(output).values()) <= OUTPUT_LIMIT_BYTES,
                        'retained output bound at command boundaries')

        check()
        output.mkdir()
        for index, command in enumerate(plan['commands'][args.phase]):
            check()
            log, started = output / f'command-{index}.log', clock.now_ns()
            with log.open('xb') as stream:
                process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, check=False)
            record['commands'].append({'command': command, 'returncode': process.returncode,
                                       'elapsed_ns': clock.now_ns() - started, 'log': descriptor(log)})
            check()
            require(process.returncode == 0, 'registered phase command passed: ' + str(index))
        validate_plan(plan)
        for phase in predecessors:
            closed = closed_phase(plan, phase)
            require(record['predecessors'][phase] == {key: descriptor(closed[key])
                    for key in ('receipt_path', 'launch_path', 'terminal_path')}, 'predecessor unchanged after command')
        record['sources_after'] = sources()
        check()
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['files'] = files(output) if output.exists() else {}
        record['peak_rss_native_units'] = {
            'self': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'children': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            'platform': sys.platform, 'continuous_limit_claimed': False}
        record['finished_unix'] = time.time()
        require(math.isfinite(record['finished_unix']), 'finite final receipt time')
        publish(receipt_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('register')
    phase = sub.add_parser('worker')
    phase.add_argument('--plan', type=Path, required=True)
    phase.add_argument('--plan-sha256', required=True)
    phase.add_argument('--phase', choices=('qualify', 'run', 'audit'), required=True)
    phase.add_argument('--supervision', type=Path, required=True)
    phase.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register()), flush=True)
    else:
        worker(args)


if __name__ == '__main__':
    main()
