"""Register and supervise the independent-cohort blind-loss comparison."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from finite_head_learning_worker_v2 import (
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

VERSION = 'finite-action-range-study-v1'
FOLDER = ROOT / 'output' / VERSION
PRIMITIVE_SHA = '4730b3c017be7a360cad2d6fb6e60510d29e69f82fea41774236061c9b92649c'
CONFIG = {
    'cohorts': [{'seed_namespace': 437260924 + i, 'fit_seed': 437261001 + i} for i in range(5)],
    'train_attempts': 512, 'dev_attempts': 512, 'batch_size': 64, 'learning_rate': .003,
    'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
    'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.,
}
SMOKE = {**CONFIG, 'cohorts': [{'seed_namespace': 948001, 'fit_seed': 948101}],
         'train_attempts': 64, 'dev_attempts': 32, 'batch_size': 16,
         'prefix_updates': 2, 'joint_updates': 3, 'fit_cap_seconds': 30.}
OWN = (
    'scripts/run_finite_action_range_learning.py', 'tests/test_finite_action_range_learning.py',
    'scripts/audit_finite_action_range_learning.py', 'tests/test_finite_action_range_audit.py',
    'scripts/qualify_finite_action_range_exposure.py', 'tests/test_finite_action_range_exposure.py',
    'scripts/finite_action_range_study.py', 'tests/test_finite_action_range_study.py',
    'research/finite-action-range-study-protocol.md',
)
TESTS = [name for name in OWN if name.startswith('tests/')]
RSS_LIMIT = 4 * 1024**3
OUTPUT_LIMIT = 1024**3
FILE_LIMIT = 2048


def read(path):
    return json.loads(Path(path).read_text())


def prerequisite():
    from qualify_finite_action_range import authenticate_closed
    return authenticate_closed(PRIMITIVE_SHA)


def phase_specs():
    return {phase: {'cap_seconds': cap, 'output': str(FOLDER / output),
                    'supervision': str(FOLDER / (phase + '-native-01.launch.json'))}
            for phase, cap, output in (('qualify', 600, 'engineering-01'),
                                       ('fit', 1800, 'run-01'), ('audit', 600, 'audit-01'))}


def plan_path(mode):
    require(mode in ('engineering', 'study'), 'known registration mode')
    return FOLDER / (mode + '-registration.json')


def commands():
    return [[str(ROOT / '.venv/bin/ruff'), 'check', *[name for name in OWN if name.endswith('.py')]],
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--noconftest', *TESTS],
            [sys.executable, str(ROOT / 'scripts/qualify_finite_action_range_exposure.py'),
             '--output', str(FOLDER / 'engineering-01/exposure')]]


def settings(parent):
    # These imports are configuration only; no model, data or audit is executed.
    from audit_finite_action_range_learning import RULE
    from qualify_finite_action_range_exposure import EXPOSURE
    names = sorted(set(parent['sources']) | set(OWN))
    require(len(names) == 161, 'complete explicit integration source roster')
    return {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(),
            'sources': {name: descriptor(ROOT / name) for name in names},
            'primitive_qualification': parent, 'config': CONFIG, 'smoke': SMOKE,
            'rule': json.loads(json.dumps(dict(RULE))), 'exposure': EXPOSURE, 'phases': phase_specs(),
            'commands': commands(), 'rss_limit_bytes': RSS_LIMIT,
            'output_limit_bytes': OUTPUT_LIMIT, 'file_limit': FILE_LIMIT}


def validate(plan, sha):
    path = plan_path(plan['mode'])
    require(descriptor(path)['sha256'] == sha and Path.cwd() == ROOT,
            'exact immutable registration and working directory')
    wanted = settings(prerequisite())
    require(all(plan.get(key) == value for key, value in wanted.items()),
            'unchanged configuration, runtime, sources and original primitive qualification')
    snapshot = FOLDER / ('source-snapshot-' + plan['mode'])
    require(read(snapshot / 'manifest.json') == {'registration': descriptor(path), 'sources': plan['sources']}
            and files(snapshot) == {**plan['sources'], 'manifest.json': descriptor(snapshot / 'manifest.json')},
            'exact complete registration source snapshot')
    return path


def launch_binding(plan, sha, receipt):
    phase, launch = receipt['phase'], receipt['launch']
    spec = plan['phases'][phase]
    command = [sys.executable, str(ROOT / 'scripts/finite_action_range_study.py'), '--phase', phase,
               '--plan-sha256', sha, '--supervision', spec['supervision']]
    require(receipt['plan'] == str(plan_path(plan['mode'])) and receipt['plan_sha256'] == sha
            and receipt['output'] == spec['output'] and receipt['supervision'] == spec['supervision']
            and launch['command'] == command and launch['cwd'] == str(ROOT)
            and launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['cap_seconds'] == spec['cap_seconds']
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid']
            and launch['deadline_ns'] - launch['started_ns'] == spec['cap_seconds'] * 10**9,
            'exact original phase command and native supervision')
    for key, name in (('watchdog_sha256', 'scripts/supervise_dialogue_observation_v2.py'),
                      ('clock_source_sha256', 'src/openjev/research/suspend_clock.py')):
        require(launch[key] == plan['sources'][name]['sha256'], 'registered native supervision sources')


def closed_phase(plan, sha, phase):
    spec = plan['phases'][phase]
    path = Path(spec['output'] + '.receipt.json')
    receipt = read(path)
    require(receipt['version'] == VERSION and receipt['phase'] == phase and receipt['status'] == 'PASS'
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['files'] == files(spec['output']), 'complete original successful phase')
    launch_binding(plan, sha, receipt)
    require(read(spec['supervision']) == receipt['launch'], 'unchanged persisted original launch')
    terminal = closed_producer(receipt)
    return receipt, terminal


def admit_qualification(plan):
    engineering_path = plan_path('engineering')
    engineering = read(engineering_path)
    sha = descriptor(engineering_path)['sha256']
    validate(engineering, sha)
    for key in settings(prerequisite()):
        require(plan[key] == engineering[key], 'same engineering and scientific choices')
    receipt, terminal = closed_phase(engineering, sha, 'qualify')
    output = Path(engineering['phases']['qualify']['output'])
    require([row['command'] for row in receipt['commands']] == commands()
            and all(row['returncode'] == 0 and row['log'] == descriptor(output / f'command-{i}.log')
                    and type(row['seconds']) in (int, float) and math.isfinite(row['seconds'])
                    and row['seconds'] >= 0 for i, row in enumerate(receipt['commands'])),
            'all original integration commands passed')
    smoke = read(output / 'smoke/audit.json')
    require(smoke['agreement'] is True and smoke['exact_oracle_agreement'] is True
            and receipt['smoke']['config'] == SMOKE
            and receipt['smoke']['audit'] == descriptor(output / 'smoke/audit.json')
            and receipt['smoke']['run_files'] == files(output / 'smoke/run'),
            'complete original saved-output smoke audit')
    probe = read(output / 'exposure/receipt.json')
    from qualify_finite_action_range_exposure import ARMS
    from qualify_finite_action_range_exposure import VERSION as EXPOSURE_VERSION
    audit_path = output / 'exposure/audit.json'
    probe_audit = read(audit_path)
    require(probe['version'] == EXPOSURE_VERSION and probe['status'] == 'PASS'
            and probe['feasible'] is True and probe['config'] == plan['exposure']['config']
            and probe['target_counts'] == plan['exposure']['target_counts']
            and probe['maximum_projected_seconds'] == plan['exposure']['maximum_projected_seconds']
            and probe['projection_comparison'] == '<=' and probe['internal_cap_seconds'] == 240.
            and probe['internal_clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and type(probe['internal_wall_seconds']) in (int, float)
            and math.isfinite(probe['internal_wall_seconds']) and 0 <= probe['internal_wall_seconds'] < 240.
            and len(probe['projections']) == 6
            and [(row['arm'], row['seed']) for row in probe['projections']] == [(arm, 948301) for arm in ARMS]
            and all(type(row['projected_seconds']) in (int, float)
                    and math.isfinite(row['projected_seconds'])
                    and 0 <= row['projected_seconds'] <= 90 for row in probe['projections'])
            and probe['audit'] == descriptor(audit_path)
            and probe_audit['agreement'] is True and probe_audit['descriptive_only'] is True
            and probe_audit['no_dev_generated'] is True
            and probe['run_files'] == files(output / 'exposure/run')
            and probe['projections'] == read(output / 'exposure/projections.json')
            and probe['files_before_receipt'] == {name: pin for name, pin in files(output / 'exposure').items()
                                                 if name != 'receipt.json'},
            'complete unchanged six-arm engineering feasibility result')
    return {'registration': descriptor(engineering_path),
            'receipt': descriptor(Path(str(output) + '.receipt.json')), 'terminal': descriptor(terminal)}


def register(mode):
    path = plan_path(mode)
    require(Path.cwd() == ROOT and not path.exists(), 'exclusive registration path')
    if mode == 'engineering':
        require(not FOLDER.exists(), 'exclusive integration study folder')
        FOLDER.mkdir()
    else:
        require(FOLDER.is_dir() and not Path(phase_specs()['fit']['output']).exists(),
                'science not started')
    plan = {**settings(prerequisite()), 'mode': mode,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    if mode == 'study':
        plan['qualification'] = admit_qualification(plan)
    publish(path, plan)
    snapshot = FOLDER / ('source-snapshot-' + mode)
    snapshot.mkdir()
    for name, pin in plan['sources'].items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        require(descriptor(target) == pin, 'exact registered source snapshot')
    publish(snapshot / 'manifest.json', {'registration': descriptor(path), 'sources': plan['sources']})
    print(json.dumps({'plan': str(path), **descriptor(path), 'sources': len(plan['sources'])}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True,
                        choices=('register-engineering', 'register-study', 'qualify', 'fit', 'audit'))
    parser.add_argument('--plan-sha256')
    parser.add_argument('--supervision', type=Path)
    args = parser.parse_args()
    if args.phase.startswith('register-'):
        register(args.phase.removeprefix('register-'))
        return
    phase = args.phase
    mode = 'engineering' if phase == 'qualify' else 'study'
    output = Path(phase_specs()[phase]['output'])
    receipt_path = Path(str(output) + '.receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'one exclusive phase attempt')
    record = {'version': VERSION, 'phase': phase, 'status': 'FAILED',
              'plan': str(plan_path(mode)), 'plan_sha256': args.plan_sha256,
              'supervision': str(args.supervision.resolve()), 'output': str(output)}
    try:
        plan = read(plan_path(mode))
        validate(plan, args.plan_sha256)
        require(plan['mode'] == mode and all(os.environ.get(key) == '1' for key in THREADS)
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
                and os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
                and os.environ.get('PYTEST_ADDOPTS') == os.environ.get('PYTEST_PLUGINS') == '',
                'fixed phase and numerical environment')
        record['sources_before'] = plan['sources']
        require(str(args.supervision.resolve()) == plan['phases'][phase]['supervision'],
                'exact supervision path')
        for _ in range(100):
            if args.supervision.exists():
                break
            time.sleep(.01)
        record['launch'] = read(args.supervision)
        launch_binding(plan, args.plan_sha256, record)
        launch = record['launch']
        require(launch['pid'] == os.getpid() == os.getpgrp(), 'this original process group')
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'same native deadline clock')
        checks = 0

        def check(*, hard=False):
            nonlocal checks
            checks += 1
            require(clock.now_ns() < launch['deadline_ns'], 'registered native phase deadline')
            if not hard and checks != 1 and checks % 256:
                return
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            require((rss if sys.platform == 'darwin' else rss * 1024) <= RSS_LIMIT, 'sampled worker RSS bound')
            if output.exists():
                paths = list(output.rglob('*'))
                require(len(paths) <= FILE_LIMIT
                        and sum(p.stat().st_size for p in paths if p.is_file()) <= OUTPUT_LIMIT,
                        'phase output bounds')

        check(hard=True)
        if phase == 'qualify':
            output.mkdir()
            record['commands'] = []
            for index, command in enumerate(commands()):
                check(hard=True)
                start = clock.now_ns()
                with (output / f'command-{index}.log').open('xb') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                                            timeout=max(.01, (launch['deadline_ns'] - start) / 1e9))
                record['commands'].append({'command': command, 'returncode': result.returncode,
                    'seconds': (clock.now_ns() - start) / 1e9, 'log': descriptor(output / f'command-{index}.log')})
                require(result.returncode == 0, 'registered integration command failed')
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from audit_finite_action_range_learning import audit
            from run_finite_action_range_learning import run
            (output / 'smoke').mkdir()
            run(output / 'smoke/run', plan['smoke'], check)
            retained = files(output / 'smoke/run')
            audited = audit(output / 'smoke/run', check=check, profile='smoke', rule=plan['rule'])
            require(audited['agreement'] is True and audited['exact_oracle_agreement'] is True
                    and files(output / 'smoke/run') == retained, 'unchanged complete smoke independently agrees')
            publish(output / 'smoke/audit.json', audited)
            record['smoke'] = {'config': SMOKE, 'run_files': retained,
                               'audit': descriptor(output / 'smoke/audit.json')}
        else:
            qualified = admit_qualification(plan)
            require(plan['qualification'] == qualified, 'original integration qualification unchanged')
            qt = read(plan['phases']['qualify']['supervision'].replace('.launch.json', '.terminal.json'))
            require(qt['clock_backend'] == launch['clock_backend']
                    and qt['finished_ns'] <= launch['started_ns'], 'qualification closed before science')
            if phase == 'fit':
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                from run_finite_action_range_learning import run
                result = run(output, plan['config'], check)
                record['result'] = {'fits': len(result['fits']), 'rows': len(result['rows'])}
            else:
                producer, terminal = closed_phase(plan, args.plan_sha256, 'fit')
                prior = read(terminal)
                require(prior['clock_backend'] == launch['clock_backend']
                        and prior['finished_ns'] <= launch['started_ns'], 'fit closed before saved-output audit')
                record['producer_receipt'] = descriptor(Path(producer['output'] + '.receipt.json'))
                record['producer_terminal'] = descriptor(terminal)
                output.mkdir()
                from audit_finite_action_range_learning import audit
                result = audit(Path(producer['output']), check=check, profile='science', rule=plan['rule'])
                require(result['agreement'] is True and result['exact_oracle_agreement'] is True,
                        'independent saved-output audit agreement')
                publish(output / 'audit.json', result)
                require(files(producer['output']) == producer['files'], 'producer unchanged by audit')
                record['result'] = {'advance': result['advance']}
        check(hard=True)
        validate(plan, args.plan_sha256)
        record['sources_after'] = plan['sources']
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['files'] = files(output)
        publish(receipt_path, record)
        print(json.dumps({'phase': phase, 'status': record['status'], 'receipt': str(receipt_path)}), flush=True)


if __name__ == '__main__':
    main()
