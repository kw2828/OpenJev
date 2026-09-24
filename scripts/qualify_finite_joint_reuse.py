"""Source-bound fabricated joint-reuse checks; no empirical fits or saved weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from finite_update_learning_worker import files
from register_finite_update_learning import SOURCES as PARENT_SOURCES
from supervise_dialogue_observation_v2 import publish

from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-joint-reuse-qualification-v1'
FOLDER = ROOT / 'output' / VERSION
SOURCES = tuple(dict.fromkeys((
    'src/openjev/research/finite_joint_reuse.py',
    'tests/test_finite_joint_reuse.py',
    'scripts/qualify_finite_joint_reuse.py',
    'research/finite-joint-reuse-qualification-protocol.md',
    'scripts/supervise_dialogue_observation_v2.py',
    'src/openjev/research/suspend_clock.py',
    'src/openjev/__init__.py', 'src/openjev/research/__init__.py',
    'pyproject.toml', 'uv.lock', *PARENT_SOURCES,
    'scripts/publish_finite_update_learning_stop.py',
)))
ENV = {key: '1' for key in ('PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')}
ENV.update(PYTEST_ADDOPTS='', PYTEST_PLUGINS='')
LINT = [SOURCES[i] for i in (0, 1, 2)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary file: ' + str(path))
    payload = path.read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def sources():
    return {name: descriptor(ROOT / name) for name in SOURCES}


def prior_stop():
    folder = ROOT / 'output/finite-update-learning-v1'
    plan = folder / 'engineering-registration-01.json'
    require(descriptor(plan)['sha256'] == 'df564a228c92291b7a3149037ee1bf94d3f03f6f289564dc21d40a6e9681bec5',
            'original stopped equal-update registration')
    original = json.loads(plan.read_text())
    require(all(descriptor(ROOT / name) == pin for name, pin in original['sources'].items()),
            'all75stopped sources unchanged')
    receipt = json.loads((folder / 'engineering-01.receipt.json').read_text())
    terminal = json.loads((folder / 'engineering-native-01.terminal.json').read_text())
    require(receipt['status'] == 'FAILED' and terminal['status'] == 'failed'
            and terminal['returncode'] == 1 and not terminal['timed_out']
            and terminal['group_absent'] and terminal['cleanup']['reaped']
            and not terminal['cleanup']['errors']
            and [row['returncode'] for row in receipt['commands']] == [0, 0, 1],
            'original closed engineering stop')
    require(not (folder / 'study-registration.json').exists()
            and not (folder / 'run-01').exists(), 'no scientific continuation of failed exposure')
    publication = ROOT / 'research/finite-update-learning-stop-results'
    pub_receipt = publication / 'receipt.json'
    require(descriptor(pub_receipt)['sha256'] == 'a9e386ed1da8f9a34dd51d9a7e6a10ed0efc5fd0ad72f6188c7fb71d4ab8d3d9',
            'exact authenticated stop publication')
    published = json.loads(pub_receipt.read_text())
    require(published['status'] == 'PASS' and published['scientific_status'] == 'QUALIFICATION_FAILED'
            and published['inputs_unchanged'] is True
            and files(publication) == {**published['files'], 'receipt.json': descriptor(pub_receipt)},
            'complete unchanged original stop publication')
    for name, pin in published['inputs_before'].items():
        require(descriptor(ROOT / name) == pin, 'original stop input still matches publication')
    return {'folder': str(folder), 'files': files(folder),
            'publication': str(publication), 'publication_files': files(publication)}


def runtime():
    return {'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pytest', 'ruff')}}


def commands():
    return [[str(ROOT / '.venv/bin/ruff'), 'check', *LINT],
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--noconftest', SOURCES[1]]]


def register(attempt):
    require(Path.cwd() == ROOT and type(attempt) is int and attempt > 0, 'cwd and positive attempt')
    FOLDER.mkdir(exist_ok=True)
    plan_path = FOLDER / f'registration-{attempt:02d}.json'
    prefix = FOLDER / f'native-{attempt:02d}'
    output = FOLDER / f'engineering-{attempt:02d}'
    snapshot = FOLDER / f'source-snapshot-{attempt:02d}'
    require(not any(p.exists() for p in (plan_path, output, snapshot, Path(str(output) + '.receipt.json'),
                                        Path(str(prefix) + '.launch.json'), Path(str(prefix) + '.terminal.json'))),
            'exclusive attempt')
    plan = {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(), 'sources': sources(),
            'environment': ENV, 'commands': commands(), 'cap_seconds': 180, 'output_limit_bytes': 8 * 1024**2,
            'output': str(output), 'supervision': str(prefix) + '.launch.json', 'snapshot': str(snapshot),
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'prior_stop': prior_stop(),
            'scope': 'Fabricated prediction, gradient, clipped-gradient and short Adam checks only; no saved checkpoint decoding, data generator, empirical fit, or speed claim.'}
    publish(plan_path, plan)
    snapshot.mkdir()
    for name, pin in plan['sources'].items():
        payload = (ROOT / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == pin['sha256'] and len(payload) == pin['bytes'],
                'source unchanged before snapshot')
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    publish(snapshot / 'manifest.json', {'registration': {'path': str(plan_path), **descriptor(plan_path)},
                                       'sources': plan['sources']})
    require(sources() == plan['sources'], 'source unchanged after snapshot')
    return {'plan': str(plan_path), **descriptor(plan_path)}


def worker(args):
    output, plan_path = args.output.resolve(), args.plan.resolve()
    receipt_path = Path(str(output) + '.receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'exclusive worker output')
    record = {'version': VERSION, 'status': 'FAILED', 'plan': str(plan_path),
              'plan_sha256': args.plan_sha256, 'output': str(output), 'supervision': str(args.supervision.resolve()),
              'commands': [], 'scope': 'Fabricated joint computation equivalence checks only', 'started_unix': time.time()}
    try:
        require(descriptor(plan_path)['sha256'] == args.plan_sha256, 'exact registration digest')
        plan = json.loads(plan_path.read_text())
        require(plan['version'] == VERSION and Path.cwd() == ROOT and plan['root'] == str(ROOT), 'root and version')
        require(plan['environment'] == ENV and all(os.environ.get(k) == v for k, v in ENV.items()),
                'single-thread environment and disabled plugin autoload')
        require(plan['commands'] == commands() and plan['runtime'] == runtime(), 'qualified commands and runtime')
        require(prior_stop() == plan['prior_stop'], 'unchanged closed equal-update stop before qualification')
        record['sources_before'] = sources()
        require(record['sources_before'] == plan['sources'], 'source closure before qualification')
        require(plan['cap_seconds'] == 180 and plan['output_limit_bytes'] == 8 * 1024**2
                and plan['output'] == str(output) and plan['supervision'] == str(args.supervision.resolve()),
                'exact paths and limits')
        for _ in range(100):
            if args.supervision.is_file():
                break
            time.sleep(.01)
        launch = json.loads(args.supervision.read_text())
        expected = [sys.executable, str(Path(__file__).resolve()), 'worker', '--plan', str(plan_path),
                    '--plan-sha256', args.plan_sha256, '--supervision', plan['supervision'], '--output', str(output)]
        require(launch['command'] == expected and launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp()
                and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == 180, 'original native launch identity')
        require(launch['watchdog_sha256'] == plan['sources'][SOURCES[4]]['sha256']
                and launch['clock_source_sha256'] == plan['sources'][SOURCES[5]]['sha256'], 'native source identity')
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'] and clock.backend in ('mach_continuous_time', 'CLOCK_BOOTTIME'),
                'native suspend-inclusive clock')
        record['launch'] = launch

        def check():
            require(launch['started_ns'] <= clock.now_ns() < launch['deadline_ns'], 'original native deadline')
            if output.exists():
                require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= plan['output_limit_bytes'],
                        'bounded qualification logs at command boundaries')

        check()
        output.mkdir()
        for i, command in enumerate(plan['commands']):
            check()
            log = output / f'command-{i}.log'
            started = clock.now_ns()
            with log.open('xb') as stream:
                process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, check=False)
            record['commands'].append({'command': command, 'returncode': process.returncode,
                                       'elapsed_ns': clock.now_ns() - started, 'log': descriptor(log)})
            check()
            require(process.returncode == 0, 'qualification command passed: ' + str(i))
        require(prior_stop() == plan['prior_stop'], 'closed equal-update stop unchanged after qualification')
        record['sources_after'] = sources()
        require(record['sources_after'] == plan['sources'], 'source closure after qualification')
        check()
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['files'] = {p.name: descriptor(p) for p in sorted(output.glob('*')) if p.is_file()}
        record['peak_rss_native_units'] = {'self': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                           'children': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                                           'platform': sys.platform, 'continuous_limit_claimed': False}
        record['finished_unix'] = time.time()
        publish(receipt_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    registration = sub.add_parser('register')
    registration.add_argument('--attempt', type=int, required=True)
    phase = sub.add_parser('worker')
    phase.add_argument('--plan', type=Path, required=True)
    phase.add_argument('--plan-sha256', required=True)
    phase.add_argument('--supervision', type=Path, required=True)
    phase.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register(args.attempt)), flush=True)
    else:
        worker(args)


if __name__ == '__main__':
    main()
