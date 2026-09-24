"""Bounded admission for one engineering, fit, or saved-output audit attempt."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev.research.suspend_clock import SuspendClock

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime():
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(),
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pytest', 'ruff')}}


def files(folder):
    if not folder.exists():
        return {}
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no symlinks in phase output')
        if path.is_file():
            result[str(path.relative_to(folder))] = descriptor(path)
    return result


def closed_producer(receipt):
    path = Path(receipt['supervision'].replace('.launch.json', '.terminal.json'))
    terminal = json.loads(path.read_text())
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['group_absent'] and not terminal['timed_out']
            and terminal['cleanup']['reaped'] and not terminal['cleanup']['errors']
            and terminal['timing_available'] and terminal['error'] is None
            and terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'original supervisor must close successfully')
    require(all(terminal[key] == receipt['launch'][key] for key in
                ('pid', 'pgid', 'command', 'cwd', 'cap_seconds', 'clock_backend',
                 'started_ns', 'deadline_ns', 'watchdog_sha256', 'clock_source_sha256')),
            'original launch must match terminal')
    return path


def qualification_commands(plan):
    root = Path(plan['root'])
    return [[str(root / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
            [plan['runtime']['executable'], '-m', 'pytest', '-q', '-p',
             'no:cacheprovider', '--noconftest', *plan['tests']]]


def validate_launch_binding(plan, receipt):
    phase = receipt['phase']
    spec = plan['phases'][phase]
    launch = receipt['launch']
    expected = [plan['runtime']['executable'],
                str(Path(plan['root']) / 'scripts/finite_rounded_learning_worker.py'),
                '--plan', receipt['plan'], '--plan-sha256', receipt['plan_sha256'],
                '--phase', phase, '--supervision', spec['supervision'], '--output', spec['output']]
    require(receipt['output'] == spec['output'] and receipt['supervision'] == spec['supervision'],
            'exact phase paths')
    require(launch['command'] == expected and launch['cwd'] == plan['root']
            and launch['cap_seconds'] == spec['cap_seconds'], 'exact phase launch binding')
    for launch_key, source in (
            ('watchdog_sha256', 'scripts/supervise_dialogue_observation_v2.py'),
            ('clock_source_sha256', 'src/openjev/research/suspend_clock.py')):
        require(launch[launch_key] == plan['sources'][source]['sha256'], 'registered launch source')


def admit_qualification(plan, receipt_path):
    receipt_path = Path(receipt_path).resolve()
    spec = plan['phases']['qualify']
    require(receipt_path == Path(spec['output'] + '.receipt.json'), 'exact qualification receipt path')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['phase'] == 'qualify' and receipt['status'] == 'PASS'
            and receipt['output'] == spec['output'] and receipt['supervision'] == spec['supervision'],
            'successful exact qualification phase')
    engineering_path = Path(receipt['plan'])
    require(descriptor(engineering_path)['sha256'] == receipt['plan_sha256'], 'original engineering plan identity')
    engineering = json.loads(engineering_path.read_text())
    require(engineering['mode'] == 'engineering', 'qualification came from engineering registration')
    for key in ('version', 'root', 'runtime', 'sources', 'config', 'phases', 'tests',
                'lint_sources', 'rss_limit_bytes', 'output_limit_bytes', 'kernel_qualification'):
        require(engineering[key] == plan[key], 'qualification/study binding: ' + key)
    require(receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
            'qualification exact source closure')
    require(files(Path(spec['output'])) == receipt['files'], 'all original qualification payloads')
    require(json.loads(Path(spec['supervision']).read_text()) == receipt['launch'],
            'original persisted qualification launch')
    validate_launch_binding(plan, receipt)
    commands = receipt['commands']
    require(len(commands) == 2 and [row['command'] for row in commands] == qualification_commands(plan),
            'exact lint and selected-test commands')
    require(all(row['returncode'] == 0 and type(row['seconds']) in (int, float)
                and math.isfinite(row['seconds']) and 0 <= row['seconds'] < spec['cap_seconds']
                for row in commands), 'successful finite qualification commands')
    return closed_producer(receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--phase', choices=('qualify', 'fit', 'audit'), required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    receipt = Path(str(output) + '.receipt.json')
    require(not output.exists() and not receipt.exists(), 'exclusive phase paths required')
    record = {'phase': args.phase, 'status': 'FAILED', 'started_unix': time.time(),
              'plan': str(args.plan.resolve()), 'plan_sha256': args.plan_sha256,
              'supervision': str(args.supervision.resolve()), 'output': str(output)}
    clock = None
    before = {}
    try:
        require(digest(args.plan) == args.plan_sha256, 'registered plan digest')
        plan = json.loads(args.plan.read_text())
        require(plan['version'] == 'finite-rounded-learning-v1', 'registered study version')
        require(plan['mode'] == ('engineering' if args.phase == 'qualify' else 'study'), 'registered phase mode')
        require(Path.cwd() == ROOT and plan['root'] == str(ROOT), 'registered working directory')
        spec = plan['phases'][args.phase]
        require(str(output) == spec['output'], 'registered output path')
        require(str(args.supervision.resolve()) == spec['supervision'], 'registered supervisor path')
        require(runtime() == plan['runtime'], 'registered runtime')
        require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread environment')
        require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
                and os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
                and os.environ.get('PYTEST_ADDOPTS') == os.environ.get('PYTEST_PLUGINS') == '',
                'no bytecode, plugin autoload or ambient pytest overrides')
        for name, expected in plan['sources'].items():
            before[name] = descriptor(ROOT / name)
            require(before[name] == expected, f'changed registered source: {name}')
        proof = plan['kernel_qualification']
        require(files(Path(proof['folder'])) == proof['files'], 'unchanged numerical qualification evidence')
        require(files(Path(proof['publication'])) == proof['publication_files'], 'unchanged numerical qualification publication')
        record['sources_before'] = before
        # The supervisor publishes immediately after starting this worker.
        for _ in range(100):
            if args.supervision.exists():
                break
            time.sleep(.01)
        launch = json.loads(args.supervision.read_text())
        require(launch['command'] == [sys.executable, *sys.argv], 'original supervised command')
        require(launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp(), 'original supervised process')
        require(launch['cwd'] == str(ROOT) and launch['cap_seconds'] == spec['cap_seconds'], 'original supervisor bounds')
        require(launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256'], 'supervisor source')
        require(launch['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'], 'clock source')
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'same native clock backend')
        record['launch'] = launch
        validate_launch_binding(plan, record)

        checks = 0

        def check(*, hard=False):
            nonlocal checks
            checks += 1
            require(clock.now_ns() < launch['deadline_ns'], 'phase deadline expired')
            if not hard and checks != 1 and checks % 256:
                return
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
            require(rss_bytes <= plan['rss_limit_bytes'], 'worker RSS limit')
            if output.exists():
                paths = list(output.rglob('*'))
                require(len(paths) <= 1024, 'phase file-count limit')
                require(sum(p.stat().st_size for p in paths if p.is_file()) <= plan['output_limit_bytes'], 'phase output limit')

        check(hard=True)
        if args.phase == 'qualify':
            output.mkdir()
            commands = qualification_commands(plan)
            record['commands'] = []
            for index, command in enumerate(commands):
                check()
                start = clock.now_ns()
                with (output / f'command-{index}.log').open('xb') as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               check=False, timeout=max(.01, (launch['deadline_ns'] - start) / 1e9))
                record['commands'].append({'command': command, 'returncode': completed.returncode,
                                           'seconds': (clock.now_ns() - start) / 1e9})
                require(completed.returncode == 0, f'qualification command {index} failed')
        elif args.phase == 'fit':
            qualify = json.loads(Path(plan['qualification']['path']).read_text())
            require(descriptor(plan['qualification']['path']) == plan['qualification']['descriptor'], 'qualification receipt digest')
            require(qualify['status'] == 'PASS', 'qualification must pass')
            qualification_terminal = admit_qualification(plan, plan['qualification']['path'])
            require(descriptor(qualification_terminal) == plan['qualification']['terminal'], 'original qualification terminal digest')
            require(json.loads(qualification_terminal.read_text())['finished_ns'] <= launch['started_ns'],
                    'qualification closed before fit launch')
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from run_finite_rounded_learning import run
            result = run(output, plan['config'], check)
            record['result'] = {'fits': len(result['fits']), 'rows': len(result['rows'])}
        else:
            fit_spec = plan['phases']['fit']
            fit_receipt_path = Path(fit_spec['output'] + '.receipt.json')
            fit_receipt = json.loads(fit_receipt_path.read_text())
            require(fit_receipt['phase'] == 'fit' and fit_receipt['plan'] == str(args.plan.resolve())
                    and fit_receipt['sources_before'] == fit_receipt['sources_after'] == plan['sources'],
                    'original fit phase and full source closure')
            validate_launch_binding(plan, fit_receipt)
            require(json.loads(Path(fit_spec['supervision']).read_text()) == fit_receipt['launch'],
                    'original persisted fit launch')
            terminal_path = closed_producer(fit_receipt)
            require(json.loads(terminal_path.read_text())['finished_ns'] <= launch['started_ns'],
                    'fit closed before audit launch')
            require(fit_receipt['status'] == 'PASS' and fit_receipt['plan_sha256'] == args.plan_sha256, 'original fit receipt')
            fit_folder = Path(fit_spec['output'])
            require(files(fit_folder) == fit_receipt['files'], 'authenticate every original producer file')
            record['producer_receipt'] = descriptor(fit_receipt_path)
            record['producer_terminal'] = descriptor(terminal_path)
            output.mkdir()
            from audit_finite_rounded_learning import audit
            result = audit(fit_folder, check=check)
            publish(output / 'audit.json', result)
            require(files(fit_folder) == fit_receipt['files'], 'producer unchanged by audit')
            record['result'] = {'gates': result['gates'], 'advance': result['advance']}
        check(hard=True)
        after = {name: descriptor(ROOT / name) for name in before}
        require(files(Path(proof['folder'])) == proof['files'], 'numerical prerequisite unchanged after phase')
        require(files(Path(proof['publication'])) == proof['publication_files'], 'numerical prerequisite publication unchanged after phase')
        record['sources_after'] = after
        require(after == before, 'registered source changed during phase')
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['finished_unix'] = time.time()
        record['files'] = files(output)
        if clock is not None and 'launch' in record:
            record['seconds'] = (clock.now_ns() - record['launch']['started_ns']) / 1e9
        publish(receipt, record)
        print(json.dumps({'phase': args.phase, 'status': record['status'], 'receipt': str(receipt)}), flush=True)


if __name__ == '__main__':
    main()
