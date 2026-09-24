"""Register and run a passive, source-bound diagnostic of saved recurrent weights.

Registration and qualification do not open checkpoint arrays. The diagnostic
authenticates the closed parent study before loading all 27 declared snapshots.
No model, optimizer, data generator or new evaluation is invoked.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from finite_training_allocation_worker import (
    ROOT,
    THREADS,
    SuspendClock,
    closed_producer,
    descriptor,
    files,
    publish,
    require,
    runtime,
)
from render_finite_training_allocation import ARMS, SEEDS, authenticate

VERSION = 'finite-transport-diagnostic-v1'
PARENT = ROOT / 'output/finite-training-allocation-v1'
FOLDER = ROOT / 'output' / VERSION
EXTRA_SOURCES = (
    'scripts/render_finite_training_allocation.py',
    'scripts/diagnose_finite_transport.py',
    'src/openjev/research/finite_transport_geometry.py',
    'tests/test_finite_transport_geometry.py',
    'research/finite-transport-diagnostic-protocol.md',
)
TESTS = ['tests/test_finite_transport_geometry.py']
LINT = [name for name in EXTRA_SOURCES if name.endswith('.py')]
PARAMETERS = {'transition_logits': (4, 8, 8), 'emission_logits': (4, 8),
              'hazard_logits': (4, 8), 'cost_logits': (4, 8)}
LABELS = ('initial', 'boundary', 'final')


def parent_identity(auth):
    return {'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
            'inputs': auth['inputs']}


def validate_launch_binding(plan, plan_path, phase, launch):
    spec = plan['phases'][phase]
    command = launch['command']
    expected = {'--plan': str(plan_path), '--plan-sha256': descriptor(plan_path)['sha256'],
                '--phase': phase, '--supervision': spec['supervision'], '--output': spec['output']}
    require(len(command) == 3 + 2 * len(expected) and command[0] == plan['runtime']['executable']
            and (ROOT / command[1]).resolve() == ROOT / 'scripts/diagnose_finite_transport.py'
            and command[2] == 'worker' and len(set(command[3::2])) == len(expected),
            'exact original diagnostic worker and unique flags')
    flags = dict(zip(command[3::2], command[4::2], strict=True))
    require(set(flags) == set(expected), 'exact phase flag names')
    for key in ('--plan', '--supervision', '--output'):
        flags[key] = str((ROOT / flags[key]).resolve())
    require(flags == expected and launch['cwd'] == str(ROOT)
            and launch['cap_seconds'] == spec['cap_seconds'], 'original phase identity and cap')
    require(launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and launch['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'original supervisor and clock sources')


def register(attempt):
    require(Path.cwd() == ROOT and type(attempt) is int and attempt > 0, 'root and positive attempt')
    auth = authenticate(PARENT)
    sources = {**auth['plan']['sources'], **{name: descriptor(ROOT / name) for name in EXTRA_SOURCES}}
    require(len(sources) == 52, 'complete diagnostic source closure')
    phases = {name: {'output': str(FOLDER / output),
                     'supervision': str(FOLDER / (prefix + '.launch.json')), 'cap_seconds': 90}
              for name, output, prefix in (
                  ('qualify', f'engineering-{attempt:02d}', f'engineering-native-{attempt:02d}'),
                  ('diagnose', 'run-01', 'diagnostic-native-01'))}
    require(not Path(phases['diagnose']['output']).exists(), 'no previous diagnostic execution')
    FOLDER.mkdir(exist_ok=True)
    plan = {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'parent': parent_identity(auth), 'phases': phases, 'tests': TESTS, 'lint_sources': LINT,
            'rss_limit_bytes': 1024**3, 'output_limit_bytes': 32 * 1024**2,
            'checkpoint_roster': [f'{label}-{arm}-{seed}.npz'
                                  for arm in ARMS for seed in SEEDS for label in LABELS],
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    path = FOLDER / f'registration-{attempt:02d}.json'
    publish(path, plan)
    return {'plan': str(path), **descriptor(path)}


def admit_qualification(plan, plan_path):
    spec = plan['phases']['qualify']
    path = Path(spec['output'] + '.receipt.json')
    receipt = json.loads(path.read_text())
    require(receipt['status'] == 'PASS' and receipt['phase'] == 'qualify'
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == descriptor(plan_path)['sha256']
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['parent_before'] == receipt['parent_after'] == plan['parent']
            and receipt['output'] == spec['output'] and receipt['supervision'] == spec['supervision'],
            'original qualification bound to this exact plan and parent')
    require(files(Path(spec['output'])) == receipt['files']
            and set(receipt['files']) == {'command-0.log', 'command-1.log'}, 'complete qualification logs')
    expected = [[str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
                [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *plan['tests']]]
    require([c['command'] for c in receipt['commands']] == expected
            and all(c['returncode'] == 0 for c in receipt['commands']), 'exact passed qualification commands')
    require(receipt['launch'] == json.loads(Path(spec['supervision']).read_text()), 'persisted original launch')
    validate_launch_binding(plan, plan_path, 'qualify', receipt['launch'])
    terminal = closed_producer(receipt)
    return path, terminal


def diagnose(auth, plan, check):
    import numpy as np

    from openjev.research.finite_transport_geometry import analyze_parameters

    require(plan['checkpoint_roster'] == [f'{label}-{arm}-{seed}.npz'
            for arm in ARMS for seed in SEEDS for label in LABELS], 'exact all-27 roster')
    rows, array_count = [], 0
    directory = auth['phases']['fit']['directory']
    for arm in ARMS:
        for seed in SEEDS:
            for label in LABELS:
                check()
                name = f'{label}-{arm}-{seed}.npz'
                expected = auth['summary']['files'][name]
                require(descriptor(directory / name) == expected, 'original checkpoint bytes')
                with np.load(directory / name, allow_pickle=False) as saved:
                    require(set(saved.files) == set(PARAMETERS), 'exact parameter archive keys')
                    parameters = {}
                    for key, shape in PARAMETERS.items():
                        value = saved[key]
                        array_count += 1
                        require(value.dtype == np.float64 and value.shape == shape
                                and np.isfinite(value).all(), 'finite original float64 parameter: ' + key)
                        parameters[key] = value.copy()
                rows.append({'arm': arm, 'seed': seed, 'label': label,
                             'checkpoint': {'path': str(directory / name), **expected},
                             'geometry': analyze_parameters(parameters, check=check)})
    require(len(rows) == 27 and array_count == 108, 'complete diagnostic input coverage')
    return {'version': VERSION, 'records': rows,
            'existing_endpoint_rows': auth['audit']['rows'], 'existing_prefix_rows': auth['audit']['prefix_rows'],
            'counts': {'checkpoint_decodes': 27, 'parameter_array_decodes': array_count,
                       'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0,
                       'new_evaluation_cases': 0, 'new_training_updates': 0},
            'scope': 'Post-hoc descriptive geometry of every saved checkpoint and existing error summaries. Uniform-action model sensitivity, not empirical information loss, latent recovery or a causal intervention.',
            'parent': plan['parent']['registration'], 'architecture_claim': False, 'candidate_advances': False,
            'requires_original_supervisor_closure': True}


def worker(args):
    output, plan_path = args.output.resolve(), args.plan.resolve()
    receipt_path = Path(str(output) + '.receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'exclusive phase paths')
    record = {'phase': args.phase, 'status': 'FAILED', 'plan': str(plan_path),
              'plan_sha256': args.plan_sha256, 'supervision': str(args.supervision.resolve()),
              'output': str(output), 'started_unix': time.time()}
    clock = None
    try:
        require(descriptor(plan_path)['sha256'] == args.plan_sha256, 'registered plan digest')
        plan = json.loads(plan_path.read_text())
        require(plan['version'] == VERSION and plan['root'] == str(ROOT) and Path.cwd() == ROOT,
                'exact diagnostic and working directory')
        require(runtime() == plan['runtime'], 'registered runtime')
        require(all(os.environ.get(k) == '1' for k in THREADS)
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'registered single-thread environment')
        before = {name: descriptor(ROOT / name) for name in plan['sources']}
        require(before == plan['sources'], 'unchanged source closure')
        record['sources_before'] = before
        spec = plan['phases'][args.phase]
        require(str(output) == spec['output'] and str(args.supervision.resolve()) == spec['supervision'],
                'registered phase paths')
        for _ in range(100):
            if args.supervision.exists():
                break
            time.sleep(.01)
        launch = json.loads(args.supervision.read_text())
        validate_launch_binding(plan, plan_path, args.phase, launch)
        require(launch['command'] == [sys.executable, *sys.argv] and launch['pid'] == os.getpid()
                and launch['pgid'] == os.getpgrp() and launch['cwd'] == str(ROOT)
                and launch['cap_seconds'] == spec['cap_seconds'], 'original supervised process')
        require(launch['watchdog_sha256'] == before['scripts/supervise_dialogue_observation_v2.py']['sha256']
                and launch['clock_source_sha256'] == before['src/openjev/research/suspend_clock.py']['sha256'],
                'original supervisor source identity')
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'same native clock')
        record['launch'] = launch

        def check():
            require(clock.now_ns() < launch['deadline_ns'], 'diagnostic phase deadline')
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            require((rss if sys.platform == 'darwin' else rss * 1024) <= plan['rss_limit_bytes'], 'worker RSS cap')
            if output.exists():
                paths = list(output.rglob('*'))
                require(len(paths) <= 16 and sum(p.stat().st_size for p in paths if p.is_file())
                        <= plan['output_limit_bytes'], 'bounded diagnostic output')

        check()
        auth = authenticate(PARENT)
        identity = parent_identity(auth)
        require(identity == plan['parent'], 'complete original closed evidence identity')
        record['parent_before'] = identity
        check()
        output.mkdir()
        if args.phase == 'qualify':
            commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
                        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *plan['tests']]]
            record['commands'] = []
            for index, command in enumerate(commands):
                check()
                start = clock.now_ns()
                with (output / f'command-{index}.log').open('xb') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                        timeout=max(.01, (launch['deadline_ns'] - start) / 1e9), check=False)
                record['commands'].append({'command': command, 'returncode': result.returncode,
                                           'seconds': (clock.now_ns() - start) / 1e9})
                require(result.returncode == 0, 'qualification command failed')
        else:
            qualification, terminal = admit_qualification(plan, plan_path)
            require(json.loads(terminal.read_text())['finished_ns'] <= launch['started_ns'], 'qualification precedes diagnosis')
            record['qualification_receipt'] = descriptor(qualification)
            record['qualification_terminal'] = descriptor(terminal)
            result = diagnose(auth, plan, check)
            publish(output / 'geometry.json', result)
            record['result'] = result['counts']
        check()
        after = parent_identity(authenticate(PARENT))
        require(after == identity, 'parent evidence unchanged')
        record['parent_after'] = after
        record['sources_after'] = {name: descriptor(ROOT / name) for name in before}
        require(record['sources_after'] == before, 'diagnostic sources unchanged')
        check()
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['files'] = files(output)
        record['finished_unix'] = time.time()
        if clock is not None and 'launch' in record:
            record['seconds'] = (clock.now_ns() - record['launch']['started_ns']) / 1e9
        publish(receipt_path, record)
        print(json.dumps({'phase': args.phase, 'status': record['status'], 'receipt': str(receipt_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    registration = commands.add_parser('register')
    registration.add_argument('--attempt', type=int, default=1)
    run = commands.add_parser('worker')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--plan-sha256', required=True)
    run.add_argument('--phase', choices=('qualify', 'diagnose'), required=True)
    run.add_argument('--supervision', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register(args.attempt)))
    else:
        worker(args)


if __name__ == '__main__':
    main()
