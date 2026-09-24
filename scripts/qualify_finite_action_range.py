"""Freeze and qualify action-error loss primitives on fabricated inputs only."""
from __future__ import annotations

import argparse
import json
import os
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

VERSION = 'finite-action-range-qualification-v1'
FOLDER = ROOT / 'output' / VERSION
PLAN = FOLDER / 'registration.json'
OUTPUT = FOLDER / 'qualification-01'
SUPERVISION = FOLDER / 'qualification-native-01.launch.json'
PARENT = ROOT / 'research/finite-decision-error-results'
PARENT_SHA = '9dedc8f41b6742aff66c860e55d1849eac6771162264832725a072302a96bdf5'
PUBLISHER_SHA = 'f397dcfd80a5cd0d23e3d8099125a76f3337b4ea1e1d3796bc75d8ccbea07971'
OWN = ('src/openjev/research/finite_action_range_loss.py',
       'scripts/run_finite_action_range_training.py',
       'tests/test_finite_action_range_loss.py',
       'tests/test_finite_action_range_training.py',
       'scripts/qualify_finite_action_range.py',
       'research/finite-action-range-qualification-protocol.md')
TESTS = [name for name in OWN if name.startswith('tests/')]


def read(path):
    return json.loads(Path(path).read_text())


def prerequisite():
    require(descriptor(PARENT / 'receipt.json')['sha256'] == PARENT_SHA, 'exact closed diagnostic publication')
    receipt = read(PARENT / 'receipt.json')
    for name, pin in receipt['sources'].items():
        require(descriptor(ROOT / name) == pin, 'unchanged diagnostic source before helper import')
    for name, pin in receipt['files'].items():
        require(descriptor(PARENT / name) == pin, 'unchanged published diagnostic evidence')
    require(descriptor(ROOT / 'scripts/publish_finite_decision_error.py')['sha256'] == PUBLISHER_SHA,
            'reviewed diagnostic authenticator')
    from publish_finite_decision_error import authenticate
    auth = authenticate()
    require(auth['child_files'] == receipt['child_files']
            and auth['external_parent']['passing_conditions'] == 14
            and auth['external_parent']['conditions'] == 15,
            'complete closed diagnostic and original failed learning comparison')
    return {'receipt': descriptor(PARENT / 'receipt.json'), 'sources': receipt['sources'],
            'child_files': receipt['child_files'], 'parent_status': 'HEAD_INDEPENDENT_ADVANCE_FAIL'}


def source_names(parent):
    return sorted(set(parent['sources']) | set(OWN) |
                  {'scripts/publish_finite_decision_error.py', 'research/finite-decision-error-next.md'})


def commands():
    return [[str(ROOT / '.venv/bin/ruff'), 'check', *[n for n in OWN if n.endswith('.py')]],
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--noconftest', *TESTS]]


def register():
    require(Path.cwd() == ROOT and not FOLDER.exists(), 'exclusive new qualification registration')
    parent = prerequisite()
    names = source_names(parent)
    require(len(names) == 152, 'complete inherited and primitive source closure')
    sources = {name: descriptor(ROOT / name) for name in names}
    plan = {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'prerequisite': parent, 'cap_seconds': 120, 'output': str(OUTPUT),
            'supervision': str(SUPERVISION), 'commands': commands(),
            'scope': 'Fabricated loss, gradient and optimizer qualification only; no scientific data or admission.',
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    FOLDER.mkdir()
    publish(PLAN, plan)
    snapshot = FOLDER / 'source-snapshot'
    snapshot.mkdir()
    for name, pin in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        require(descriptor(target) == pin, 'exact qualification source snapshot')
    publish(snapshot / 'manifest.json', {'registration': descriptor(PLAN), 'sources': sources})
    print(json.dumps({'plan': str(PLAN), **descriptor(PLAN), 'sources': len(sources)}))


def validate(plan, sha):
    require(descriptor(PLAN)['sha256'] == sha and plan['version'] == VERSION
            and plan['root'] == str(ROOT) and Path.cwd() == ROOT
            and plan['runtime'] == runtime() and plan['cap_seconds'] == 120
            and plan['output'] == str(OUTPUT) and plan['supervision'] == str(SUPERVISION)
            and plan['commands'] == commands(), 'exact immutable qualification plan')
    parent = prerequisite()
    names = source_names(parent)
    require(len(names) == 152 and sorted(plan['sources']) == names
            and {name: descriptor(ROOT / name) for name in names} == plan['sources']
            and parent == plan['prerequisite'], 'complete unchanged source and prerequisite closure')
    snapshot = FOLDER / 'source-snapshot'
    require(read(snapshot / 'manifest.json') == {'registration': descriptor(PLAN), 'sources': plan['sources']}
            and files(snapshot) == {**plan['sources'], 'manifest.json': descriptor(snapshot / 'manifest.json')},
            'complete original source snapshot')


def launch_binding(plan, sha, launch):
    expected = [sys.executable, str(ROOT / 'scripts/qualify_finite_action_range.py'),
                '--phase', 'qualify', '--plan-sha256', sha, '--supervision', str(SUPERVISION)]
    require(launch['command'] == expected and launch['cwd'] == str(ROOT)
            and launch['cap_seconds'] == 120 and launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['deadline_ns'] - launch['started_ns'] == 120 * 10**9,
            'original bounded native launch')
    for key, name in (('watchdog_sha256', 'scripts/supervise_dialogue_observation_v2.py'),
                      ('clock_source_sha256', 'src/openjev/research/suspend_clock.py')):
        require(launch[key] == plan['sources'][name]['sha256'], 'registered supervision source')


def authenticate_closed(sha):
    plan = read(PLAN)
    validate(plan, sha)
    receipt_path = Path(str(OUTPUT) + '.receipt.json')
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS' and receipt['plan_sha256'] == sha
            and receipt['plan'] == str(PLAN) and receipt['output'] == str(OUTPUT)
            and receipt['supervision'] == str(SUPERVISION)
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['files'] == files(OUTPUT), 'complete successful original qualification')
    require(read(SUPERVISION) == receipt['launch'], 'persisted original launch')
    launch_binding(plan, sha, receipt['launch'])
    require(len(receipt['commands']) == 2 and [r['command'] for r in receipt['commands']] == commands()
            and all(r['returncode'] == 0 and r['log'] == descriptor(OUTPUT / f'command-{i}.log')
                    for i, r in enumerate(receipt['commands'])), 'both original fixed commands passed')
    terminal = closed_producer(receipt)
    return {'registration': descriptor(PLAN), 'receipt': descriptor(receipt_path),
            'terminal': descriptor(terminal), 'sources': plan['sources'],
            'scope': plan['scope'], 'scientific_admission': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('register', 'qualify'), required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--supervision', type=Path)
    args = parser.parse_args()
    if args.phase == 'register':
        register()
        return
    receipt_path = Path(str(OUTPUT) + '.receipt.json')
    require(not OUTPUT.exists() and not receipt_path.exists(), 'one exclusive qualification attempt')
    record = {'version': VERSION, 'status': 'FAILED', 'plan': str(PLAN), 'plan_sha256': args.plan_sha256,
              'supervision': str(args.supervision.resolve()), 'output': str(OUTPUT)}
    try:
        plan = read(PLAN)
        validate(plan, args.plan_sha256)
        require(str(args.supervision.resolve()) == str(SUPERVISION), 'exact original supervision path')
        require(all(os.environ.get(key) == '1' for key in THREADS)
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
                and os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
                and os.environ.get('PYTEST_ADDOPTS') == os.environ.get('PYTEST_PLUGINS') == '', 'fixed qualification environment')
        record['sources_before'] = plan['sources']
        for _ in range(100):
            if SUPERVISION.exists():
                break
            time.sleep(.01)
        launch = read(SUPERVISION)
        launch_binding(plan, args.plan_sha256, launch)
        require(launch['pid'] == os.getpid() == launch['pgid'] == os.getpgrp(), 'this original process group')
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'same native clock')
        record['launch'] = launch
        OUTPUT.mkdir()
        record['commands'] = []
        for index, command in enumerate(commands()):
            require(clock.now_ns() < launch['deadline_ns'], 'qualification native deadline')
            with (OUTPUT / f'command-{index}.log').open('xb') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                                        timeout=max(.01, (launch['deadline_ns'] - clock.now_ns()) / 1e9))
            record['commands'].append({'command': command, 'returncode': result.returncode,
                                        'log': descriptor(OUTPUT / f'command-{index}.log')})
            require(result.returncode == 0, 'qualification command failed')
        require(clock.now_ns() < launch['deadline_ns'], 'qualification closed within native bound')
        validate(plan, args.plan_sha256)
        record['sources_after'] = plan['sources']
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['files'] = files(OUTPUT)
        publish(receipt_path, record)
        print(json.dumps({'status': record['status'], 'receipt': str(receipt_path)}), flush=True)


if __name__ == '__main__':
    main()
